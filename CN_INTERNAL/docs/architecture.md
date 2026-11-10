# CAIL System Architecture

**Status**: Implemented (v0.1.0)
**Validated**: 2026-06-12 against `main` @ `91c5e4f`

## 1. System Purpose

CAIL (Cornelis Allreduce Interposition Library) accelerates GPU-resident
`MPI_Allreduce` operations without requiring application changes. It is loaded
via `LD_PRELOAD` (or direct linking) and uses the MPI profiling interface
(PMPI): the library defines the `MPI_Allreduce` symbol itself, evaluates
whether the call is eligible for optimization, and either executes one of
three optimized algorithms with device-side reduction kernels or delegates to
the MPI runtime's own implementation via `PMPI_Allreduce`.

### What Problem Does It Solve?

Stock MPI allreduce implementations are frequently not tuned for GPU buffers:
staging through host memory, generic reduction loops, and topology-unaware
algorithm selection leave bandwidth on the table. CAIL provides:

1. **Device-native reduction** — element-wise reduction kernels run directly
   on the GPU (CUDA), avoiding host staging for compute.
2. **GPU-aware MPI transport reuse** — point-to-point exchanges use the
   underlying MPI's GPU-aware path (`PMPI_Sendrecv` et al. on device
   pointers), so CAIL composes with CUDA-aware/ROCm-aware Open MPI.
3. **Message-size-aware algorithm selection** — latency-optimal recursive
   doubling for small messages, bandwidth-optimal Rabenseifner for large ones.

## 2. Layer Map

```
              Application
                  │  MPI_Allreduce(...)
                  ▼
┌───────────────────────────────────────────────┐
│ core/   cail_pmpi.c     PMPI interception     │
│         cail_init.c     lazy init, env config │
│         cail_types.c    MPI↔internal type map │
│         cail_buf.c      scratch buffer cache  │
└───────────────┬───────────────────────────────┘
                │ eligible            │ ineligible
                ▼                     ▼
┌───────────────────────────┐   PMPI_Allreduce (native MPI)
│ coll/allreduce/           │
│   cail_allreduce.c        │  dispatch (auto / forced)
│   ..._recursive_doubling.c│
│   ..._ring.c              │
│   ..._rabenseifner.c      │
└───────────────┬───────────┘
                ▼
┌───────────────────────────────────────────────┐
│ gpu/    cail_gpu.h        backend-agnostic API│
│   cuda/  CUDA kernels, UVA memcpy, PCIe fence │
│   host/  PMPI_Reduce_local + malloc           │
│   rocm/  stub only (no HIP code) — see        │
│          gpu-layer.md                         │
└───────────────────────────────────────────────┘
```

Each layer builds as a libtool convenience library (`libcail_core.la`,
`libcail_allreduce.la`, `libcail_gpu.la`) assembled into the single installed
artifact `libcail.so` (libtool `-version-info 0:0:0`).

## 3. PMPI Interception

Defined in `src/core/cail_pmpi.c`. CAIL intercepts exactly two C symbols:

| Symbol | Behavior |
|--------|----------|
| `MPI_Allreduce` | Eligibility gate → `cail_allreduce_dispatch()` or `PMPI_Allreduce()` fallback (see `interception-and-dispatch.md`) |
| `MPI_Finalize` | Runs `cail_finalize()` (GPU teardown + scratch buffer free), then `PMPI_Finalize()` |

**No Fortran bindings exist.** `mpi_allreduce_` / `MPI_ALLREDUCE` wrappers are
not provided. Fortran applications are only intercepted if their MPI Fortran
layer delegates to the C `MPI_Allreduce` symbol (Open MPI does not — its
Fortran bindings call the back-end directly). **KNOWN GAP** for Fortran
workloads.

## 4. Initialization Lifecycle

Initialization is **lazy**: it happens inside the first intercepted
`MPI_Allreduce` call, not at `MPI_Init` and not via a library constructor.

`cail_init(comm)` in `src/core/cail_init.c`:

1. Early-exit if already initialized.
2. Parse environment variables into `cail_global_state` (see §5).
3. Call `cail_gpu_init()` — creates the CUDA stream and kernel dispatch
   table (no-op on host path). On failure returns `MPI_ERR_INTERN`.
4. Set `initialized = 1` and emit a `CAIL_DBG` trace of the resolved config.

If `cail_init` fails, the interposer calls `PMPI_Abort(comm, rc)` — it does
**not** fall back to native MPI. A broken GPU runtime is treated as fatal.

`cail_finalize()` (from intercepted `MPI_Finalize`) calls
`cail_gpu_finalize()` and `cail_buf_finalize()`, then clears `initialized`.

### Thread Safety

**KNOWN GAP**: `cail_init()` has no `pthread_once`, atomics, or locking.
Concurrent first calls to `MPI_Allreduce` from multiple threads
(`MPI_THREAD_MULTIPLE`) race on `cail_global_state` and on `cail_gpu_init()`.
The scratch-buffer cache (§6) is similarly unsynchronized. CAIL is currently
safe only for single-threaded MPI usage (`MPI_THREAD_SINGLE`/`FUNNELED`).

## 5. Global State

One static instance of `cail_state_t` (`cail_global_state`, defined in
`cail_init.c`, declared in `cail_internal.h`):

| Field | Env var | Default | Meaning |
|-------|---------|---------|---------|
| `initialized` | — | 0 | Set after successful `cail_init()` |
| `debug` | `CAIL_DEBUG` | 0 | Enable `CAIL_DBG` stderr traces (non-empty, not `'0'`) |
| `warn` | `CAIL_WARN` | 1 | Enable `CAIL_WARN` stderr messages (`'0'` disables) |
| `msg_small_threshold` | `CAIL_MSG_SMALL_THRESHOLD` | 262144 (256 KB) | Small/large message boundary for auto-dispatch |
| `nprocs_threshold` | `CAIL_NPROCS_THRESHOLD` | 4 | Reserved; parsed/validated but not used by current auto-dispatch |
| `min_msg_size` | `CAIL_MIN_MSG_SIZE` | 32768 (32 KB) | Below this, pass through to native PMPI; 0 disables passthrough |
| `force_algo` | `CAIL_ALGO` | `CAIL_ALGO_AUTO` | `auto`, `recursive_doubling`, `ring`, `rabenseifner` |

Env parsing helpers (`cail_env_parse_positive_long`,
`cail_env_parse_positive_int`, `cail_env_parse_nonneg_long`) use
`strtol(..., 10)` and reject trailing garbage and sign violations; on parse
failure they log `CAIL_ERR` and keep the compiled-in default. An
unrecognized `CAIL_ALGO` string logs `CAIL_ERR` and resets to auto.
**Note**: only the `int` helper bounds-checks (`val > INT_MAX`); the `long`
helpers do not check `errno == ERANGE`, so a value that overflows `strtol`
saturates to `LONG_MAX` and is silently accepted rather than rejected.

**Note**: the `cail_global_state` designated initializer assigns
`.nprocs_threshold` twice (lines 14–15 of `cail_init.c`). Benign (same
constant) but should be cleaned up.

## 6. Scratch Buffer Cache (`cail_buf.c`)

All three algorithms need temporary device (or host) scratch space. Because
`cudaMalloc`/`cudaFree` are expensive and repeated alloc/free churns the MPI
registration cache, `cail_buf.c` implements a **single-slot high-water-mark
cache**:

- `cail_buf_get_tmp(&ptr, size)` — if `size <= cached_size`, returns the
  cached buffer without allocating. Otherwise frees the old buffer, allocates
  `size` bytes (via `cail_gpu_malloc`, or `malloc` on the host path), and
  records the new high-water mark. Returns `MPI_ERR_NO_MEM` on failure.
- `cail_buf_release_tmp(ptr)` — **intentional no-op**. The buffer is retained
  for the next collective.
- `cail_buf_finalize()` — actually frees, called from `MPI_Finalize`.

Implications:

- Memory equal to the largest allreduce message persists on the device for
  the life of the job. For workloads with one outlier-sized allreduce this
  can pressure GPU memory. **KNOWN GAP**: no shrink policy.
- Single slot = no concurrency: a second in-flight allreduce would corrupt
  the cache (consistent with the single-threaded constraint in §4).

## 7. Type and Op Mapping (`cail_types.c`)

CAIL bridges MPI handles to dense internal enums sized for the GPU kernel
dispatch table:

- `cail_datatype_t`: 12 values (`CAIL_CHAR`, `CAIL_INT`, `CAIL_LONG`,
  `CAIL_FLOAT`, `CAIL_DOUBLE`, `CAIL_LONG_LONG`, `CAIL_UCHAR`, `CAIL_UINT`,
  `CAIL_ULONG`, `CAIL_ULONGLONG`, `CAIL_SHORT`, `CAIL_USHORT`) plus
  `CAIL_NUM_DTYPES` and `CAIL_INVALID`.
- `cail_op_t`: 4 values (`CAIL_SUM`, `CAIL_PROD`, `CAIL_MAX`, `CAIL_MIN`)
  plus `CAIL_NUM_OPS` and `CAIL_OP_INVALID`.

`cail_mpi_type_to_dtype()` recognizes 23 MPI datatype handles, folding
aliases and C99 fixed-width types onto the 12 internal enums (e.g. `MPI_CHAR`,
`MPI_SIGNED_CHAR`, `MPI_INT8_T` → `CAIL_CHAR`; `MPI_BYTE`,
`MPI_UNSIGNED_CHAR`, `MPI_UINT8_T` → `CAIL_UCHAR`). The README's headline
says "20 MPI datatypes (12 base + 8 C99 aliases)" but its own table lists 23
distinct handles (slash-separated alias entries such as
`MPI_LONG_LONG / MPI_LONG_LONG_INT`); the recognizer explicitly checks all
23.

The reverse map `cail_dtype_to_mpi_type()` returns canonical MPI types only
(lossy but representation-identical) — used by the host path to drive
`PMPI_Reduce_local`. `cail_type_size()` uses native `sizeof`, so `CAIL_LONG`
is 8 bytes on LP64 and 4 bytes on ILP32/Windows ABIs.

## 8. Logging and Error Conventions

Macros in `cail_internal.h`, all writing to `stderr`:

| Macro | Gate | Prefix |
|-------|------|--------|
| `CAIL_DBG` | `cail_global_state.debug` | `[cail]` |
| `CAIL_WARN` | `cail_global_state.warn` | `[cail WARN]` |
| `CAIL_ERR` | always | `[cail ERROR]` |

`CAIL_CHECK(rc)` propagates any non-`MPI_SUCCESS` return code to the caller.
Within the interposer, internal failures degrade gracefully to
`PMPI_Allreduce`; within forced-algorithm paths, configuration errors abort
(see `interception-and-dispatch.md` §3).

## 9. Known Gaps Summary

| Gap | Severity | Where documented |
|-----|----------|------------------|
| Missing PCIe flush after Phase-3 unfold receives (potential stale GPU reads) | **High (code bug)** | `algorithms.md` (shared properties + §1) |
| No Fortran bindings | Medium | §3 |
| No thread safety (init, buffer cache) | Medium | §4, §6 |
| Scratch buffer never shrinks | Low | §6 |
| Duplicate designated initializer | Trivial | §5 |
| ROCm backend is a stub | High (for this workspace) | `gpu-layer.md` |
| Build breaks when CUDA absent and host-path not requested | Medium | `build-and-packaging.md` |
