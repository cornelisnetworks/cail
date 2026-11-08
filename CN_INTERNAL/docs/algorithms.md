# CAIL Allreduce Algorithms

**Status**: Implemented (v0.1.0)
**Validated**: 2026-06-12 against `main` @ `91c5e4f`
**Source**: `src/coll/allreduce/cail_allreduce_{recursive_doubling,ring,rabenseifner}.c`

All three algorithms share these properties:

- **Blocking point-to-point only**: `PMPI_Send`, `PMPI_Recv`,
  `PMPI_Sendrecv`. No nonblocking overlap, no MPI collectives internally.
- **GPU-aware transport**: device pointers are passed straight to PMPI
  point-to-point calls; the underlying MPI must be GPU-aware
  (CUDA-aware Open MPI in the current deployment).
- **Receive flush before compute**: every receive that feeds a subsequent
  reduction kernel or remains in the working buffer is followed by
  `cail_gpu_flush_recv_buf(buf, bytes)` — a 1-byte device-to-host read of
  the buffer tail that drains posted PCIe writes (NIC RDMA / GDRCopy)
  before a reduction kernel reads the data. See `gpu-layer.md` §2.
  **KNOWN GAP (code bug)**: the Phase-3 unfold receives — even folded ranks
  receiving the final result into `recvbuf`
  (`cail_allreduce_recursive_doubling.c:131`,
  `cail_allreduce_rabenseifner.c:261`) — are **not** followed by a flush.
  CAIL launches no kernel after these receives, but the application may
  read `recvbuf` from a GPU kernel immediately after `MPI_Allreduce`
  returns; on RDMA-write delivery paths this can observe stale data. The
  fix is a one-line flush after each of the two `PMPI_Recv` calls.
- **Scratch via the buffer cache**: temp space comes from
  `cail_buf_get_tmp()`; `cail_buf_release_tmp()` in the cleanup path is a
  no-op by design (see `architecture.md` §6).
- **`goto cleanup` error structure**: on any PMPI or kernel failure, host
  metadata arrays are freed, the scratch ref released, and the error code
  returned — dispatch then falls back to `PMPI_Allreduce` (the application
  still gets a correct reduction unless the failure poisoned `recvbuf`;
  see §5).
- **MPI_IN_PLACE**: handled identically everywhere — if
  `sendbuf != MPI_IN_PLACE`, stage with
  `cail_gpu_memcpy(recvbuf, sendbuf, bufsize)`; the algorithm then operates
  on `recvbuf` as the working buffer.

## 1. Shared Concept: Rank Folding (non-power-of-two)

Recursive doubling and Rabenseifner both require a power-of-two participant
count. With `pof2` = largest power of two ≤ `nprocs` and
`rem = nprocs - pof2`:

- **Fold-in (phase 1)**: the first `2*rem` ranks pair up.
  - Even rank `r` sends its whole working buffer to `r+1` (`PMPI_Send`) and
    sits out (`newrank = -1`).
  - Odd rank `r` receives into `tmpbuf` (`PMPI_Recv`), flushes, reduces
    locally (`cail_gpu_reduce_local`), and joins with `newrank = r/2`.
  - Ranks `>= 2*rem` join with `newrank = rank - rem`.
- **Main phase**: exactly `pof2` virtual ranks participate. Virtual→physical
  mapping: `dst = (newdst < rem) ? newdst*2 + 1 : newdst + rem`.
- **Unfold (phase 3)**: odd rank `r` calls `PMPI_Send` with the final
  result; even rank `r-1` posts a matching `PMPI_Recv` into `recvbuf`
  (two distinct blocking calls on distinct ranks — not `PMPI_Sendrecv`),
  repopulating the ranks that sat out. **Note**: these receives lack the
  flush call (see KNOWN GAP above).

Cost: 2 extra latency steps and a full extra message per folded pair — the
origin of the dispatch threshold-halving rule
(`interception-and-dispatch.md` §2).

## 2. Recursive Doubling

**File**: `cail_allreduce_recursive_doubling.c` — small-message default.

After fold-in, each active virtual rank loops `mask = 1; mask < pof2;
mask <<= 1`:

1. Partner: `newdst = newrank ^ mask`, mapped to physical `dst`.
2. `PMPI_Sendrecv`: swap the **entire** working buffer `recvbuf` with `dst`,
   receiving into `tmpbuf`.
3. `cail_gpu_flush_recv_buf(tmpbuf, bufsize)`.
4. `cail_gpu_reduce_local(tmpbuf, recvbuf, count, dtype, op)`.

Then unfold.

- **Scratch**: one full-message `tmpbuf` (`bufsize` bytes).
- **Steps**: `log2(pof2)` exchanges (+2 fold steps if non-pof2).
- **Traffic per active rank**: `N * log2(pof2)` bytes — latency-optimal,
  bandwidth-suboptimal; hence the small-message slot.

## 3. Ring

**File**: `cail_allreduce_ring.c` — forced-only (`CAIL_ALGO=ring`); never
auto-selected (see `interception-and-dispatch.md` §2).

Buffer is split into `nprocs` chunks: `base = count / nprocs`,
first `count % nprocs` chunks get one extra element; `chunk_size[]` (int) and
`chunk_offset[]` (size_t) are heap-allocated per call. Neighbors:
`left = (rank-1+P) % P`, `right = (rank+1) % P`.

- **Phase 1 — reduce-scatter ring** (`P-1` steps): at step `s`, send chunk
  `(rank - s + P) % P` from within `recvbuf` to `right`; receive chunk
  `(rank - s - 1 + P) % P` from `left` into `tmp_buf` (`PMPI_Sendrecv`);
  flush; `cail_gpu_reduce_local` into the chunk's slice of `recvbuf`.
  After `P-1` steps each rank owns the fully-reduced chunk `(rank+1) % P`.
- **Phase 2 — allgather ring** (`P-1` steps): at step `s`, send chunk
  `(rank - s + 1 + P) % P` to `right`; receive chunk `(rank - s + P) % P`
  from `left` **directly into the destination slice** of `recvbuf`
  (no reduction, flush only).

- **Constraint**: requires `count >= nprocs` (each chunk non-empty);
  otherwise `CAIL_WARN` + fallback to `PMPI_Allreduce`.
- **Non-pof2**: natively supported (no folding).
- **Scratch**: `max_chunk * type_size` bytes only — smallest footprint of
  the three.
- **Steps**: `2(P-1)`. **Traffic per rank**: `2N(P-1)/P` bytes —
  bandwidth-optimal but linear latency.

## 4. Rabenseifner

**File**: `cail_allreduce_rabenseifner.c` — large-message default.

Fold-in as §1, then two sub-phases over `pof2` virtual ranks with per-call
heap arrays `cnts[]`/`disps[]` partitioning `count` into `pof2` blocks:

- **Phase 2a — reduce-scatter by recursive halving**
  (`mask = 1 → pof2/2`): partner `newdst = newrank ^ mask`; the rank's
  active block range halves each step. Lower-half rank sends the upper half
  of its active blocks and receives the lower half into `tmpbuf`
  (`PMPI_Sendrecv`); flush; `cail_gpu_reduce_local` on the received slice
  into `recvbuf`. After `log2(pof2)` steps each active rank owns `1/pof2`
  of the fully-reduced vector.
- **Phase 2b — allgather by recursive doubling**
  (`mask = pof2/2 → 1`): reverse traversal; ranks exchange owned blocks,
  receiving **directly into `recvbuf`** (flush, no reduction), doubling
  owned range each step.

Then unfold.

- **Constraint**: `count >= pof2`. Dispatch screens this; the function
  itself `PMPI_Abort`s if violated (defensive backstop — see
  `interception-and-dispatch.md` §3).
- **Scratch**: full-message `tmpbuf` (`bufsize` bytes).
- **Steps**: `2*log2(pof2)` (+2 fold). **Traffic per active rank**:
  `2N(pof2-1)/pof2` — ring-class bandwidth with logarithmic latency.

## 5. Complexity Summary

| Algorithm | Steps | Bytes/rank | Host allocs per call | GPU scratch |
|-----------|-------|-----------|----------------------|-------------|
| Recursive doubling | `log2(pof2)` (+2 fold) | `N·log2(pof2)` | none | `N` |
| Ring | `2(P-1)` | `2N(P-1)/P` | 2 arrays, `O(P)` | `⌈N/P⌉` |
| Rabenseifner | `2·log2(pof2)` (+2 fold) | `2N(pof2-1)/pof2` | 2 arrays, `O(pof2)` | `N` |

## 6. Failure Modes and Engineering Notes

- **Mid-collective failure hangs peers**: all communication is blocking with
  no timeouts. If one rank dies mid-algorithm, the others block in
  `PMPI_Sendrecv` until the MPI runtime kills the job. Standard for this
  class of library, but worth stating.
- **Fallback after partial mutation**: if an algorithm fails *after* staging
  `sendbuf → recvbuf` (or after some reduction steps) and dispatch falls
  back to `PMPI_Allreduce`, the fallback re-runs from `sendbuf`/`recvbuf`
  as the application provided them. For the `MPI_IN_PLACE` case, `recvbuf`
  may already be partially reduced — the PMPI retry would then operate on
  corrupted input. **KNOWN GAP**: low practical likelihood (failures at this
  point are PMPI transport errors that would also fail the retry), but the
  fallback-on-dispatch-error path is only strictly safe for failures that
  occur before the first buffer mutation (e.g. scratch allocation failure).
- **Per-call heap churn**: ring and Rabenseifner `malloc`/`free` their
  metadata arrays on every invocation. For allreduce-heavy workloads this is
  measurable CPU overhead; a per-communicator cache would remove it.
  **KNOWN GAP** (performance, not correctness).
