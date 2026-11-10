# CAIL Interception and Dispatch

**Status**: Implemented (v0.1.0)
**Validated**: 2026-06-12 against `main` @ `91c5e4f`
**Source**: `src/core/cail_pmpi.c`, `src/coll/allreduce/cail_allreduce.c`

This document covers the two decision layers between an application's
`MPI_Allreduce` call and an algorithm implementation:

1. **Eligibility gate** (`cail_pmpi.c`) — intercept or fall back to PMPI?
2. **Dispatch** (`cail_allreduce.c`) — which algorithm runs?

## 1. Eligibility Gate

Checks run in this exact order inside the intercepted `MPI_Allreduce`:

| # | Check | On failure |
|---|-------|-----------|
| 1 | Initialized? If not, `cail_init(comm)` | `PMPI_Abort` (init failure is fatal, no fallback) |
| 2 | `count == 0` | Return `MPI_SUCCESS` immediately (no PMPI call) |
| 3 | `cail_type_supported(datatype)` | `CAIL_WARN` + fall back to `PMPI_Allreduce` |
| 4 | `cail_op_supported(op)` | `CAIL_WARN` + fall back |
| 5 | `PMPI_Comm_test_inter` — intracommunicator? | `CAIL_DBG` + fall back |
| 6 | Device-pointer check (skipped if built with `CAIL_HOST_PATH`): `recvbuf` must be a device pointer; `sendbuf` must be a device pointer unless `MPI_IN_PLACE` | `CAIL_DBG` + fall back |
| 7 | `msg_size = count * type_size >= min_msg_size` (`CAIL_MIN_MSG_SIZE`, default 32 KB) | `CAIL_DBG` + fall back |
| 8 | `cail_allreduce_dispatch(...)` returned `MPI_SUCCESS`? | `CAIL_DBG` + fall back |

Notes:

- Check 2 means a zero-count allreduce never reaches native MPI. Native MPI
  also treats it as a no-op, so semantics are preserved, but error-checking
  behavior of the native implementation (e.g. invalid communicator detection)
  is bypassed for `count == 0`.
- Check 6 uses `cail_gpu_is_device_pointer()` →
  `cudaPointerGetAttributes`; managed (UVM) memory counts as device memory.
  On the host-path build this check is compiled out and *all* buffers are
  eligible.
- Check 8 makes dispatch failures (e.g. scratch allocation failure)
  transparent to the application — the reduction still completes via PMPI.

## 2. Auto-Dispatch (`CAIL_ALGO=auto`, the default)

`cail_allreduce_dispatch()` resolves `nprocs` via `PMPI_Comm_size` and
computes `pof2` = largest power of two ≤ `nprocs`. Decisions in order:

```
1. count < pof2(nprocs)        → dispatch_small_msg()   (Rabenseifner cannot
                                                          partition the buffer)
2. effective_small = msg_small_threshold                 (CAIL_MSG_SMALL_THRESHOLD,
                                                          default 256 KB)
   if !is_pof2(nprocs) && nprocs >= 16:
       effective_small /= 2                              ("threshold halving")
3. msg_size <  effective_small → dispatch_small_msg()    → recursive doubling
4. msg_size >= effective_small → dispatch_large_msg()    → Rabenseifner
```

### The Threshold-Halving Rule

For non-power-of-two process counts with `nprocs >= 16`, the small/large
boundary is halved. Rationale (from code comments): the rank fold/unfold
steps (see `algorithms.md` §2) add two extra latency steps and an extra
full-message transfer for folded ranks. At larger scale this overhead is
better amortized by Rabenseifner's superior bandwidth term, so the dispatcher
switches to the large-message algorithm at half the usual message size.

### Ring Is Forced-Only

`dispatch_large_msg()` always selects Rabenseifner. The ring algorithm is
**never** chosen automatically — performance testing at np=2..8 (GPU) showed
Rabenseifner wins at all tested scales (commit `91c5e4f` "Updated defaults
from performance testing"). Ring remains available via `CAIL_ALGO=ring` for
topology experiments. **Note for tuning docs**: `CAIL_NPROCS_THRESHOLD` is
parsed and stored but currently unused by auto-dispatch (reserved).

### Compile-Time Gating Interaction

Each algorithm can be compiled out (`--disable-ring` etc. →
`CAIL_ENABLE_*` undefined). Auto-dispatch wrappers (`call_ring`,
`call_rabenseifner`, `call_recursive_doubling`) then degrade gracefully:

```c
static inline int call_ring(...) {
#ifdef CAIL_ENABLE_RING
    return cail_allreduce_ring(...);
#else
    CAIL_WARN("ring selected ... but disabled at compile time, falling back to PMPI");
    return PMPI_Allreduce(...);
#endif
}
```

## 3. Forced Dispatch (`CAIL_ALGO` set)

When `force_algo != CAIL_ALGO_AUTO`, dispatch branches immediately:

| Forced value | Compiled in? | Constraint | Action |
|--------------|-------------|------------|--------|
| `recursive_doubling` | yes | — | run it |
| `recursive_doubling` | no | — | `CAIL_ERR` + `PMPI_Abort(comm, MPI_ERR_INTERN)` |
| `ring` | yes | — | run it |
| `ring` | no | — | `CAIL_ERR` + `PMPI_Abort` |
| `rabenseifner` | yes | `count >= pof2(nprocs)` | run it |
| `rabenseifner` | yes | `count < pof2(nprocs)` | `CAIL_WARN` + fall back to `PMPI_Allreduce` |
| `rabenseifner` | no | — | `CAIL_ERR` + `PMPI_Abort` |

Design rationale: a *forced* algorithm that was compiled out is a
configuration error the user must see immediately (abort), whereas a forced
Rabenseifner that merely can't partition this particular message is a
per-call condition (warn + fallback).

**Asymmetry note**: `cail_allreduce_rabenseifner()` itself also re-checks
`count < pof2` and calls `PMPI_Abort` if violated. The dispatch layer screens
this before calling, so the in-algorithm abort is a defensive backstop — but
any future caller bypassing dispatch must respect the precondition.

## 4. Decision Flow Summary

```
MPI_Allreduce
 ├─ count==0 ──────────────────────────────► return MPI_SUCCESS
 ├─ unsupported type/op ───────────────────► PMPI_Allreduce  [warn]
 ├─ intercommunicator ─────────────────────► PMPI_Allreduce
 ├─ host buffer (GPU build) ───────────────► PMPI_Allreduce
 ├─ msg_size < CAIL_MIN_MSG_SIZE ──────────► PMPI_Allreduce
 └─ cail_allreduce_dispatch
     ├─ CAIL_ALGO forced ──► algorithm (or abort if compiled out;
     │                       Rabenseifner small-count → PMPI fallback)
     └─ auto
         ├─ count < pof2 ──────────────────► recursive doubling
         ├─ msg < effective_small ─────────► recursive doubling
         └─ else ──────────────────────────► Rabenseifner
     └─ dispatch returned error ───────────► PMPI_Allreduce  [dbg]
```
