# CAIL - Cornelis Allreduce Interposition Library

A GPU-aware `MPI_Allreduce` interposition library using PMPI. CAIL transparently
intercepts `MPI_Allreduce` calls and routes them through optimized GPU-aware
algorithms with CUDA kernel support and a host-path fallback.

## How It Works

CAIL defines its own `MPI_Allreduce` and `MPI_Finalize` symbols that override
the MPI library's versions via the PMPI profiling interface. When your application
calls `MPI_Allreduce`, CAIL intercepts the call and:

1. Checks whether the buffer resides on a GPU (via `cudaPointerGetAttributes`)
2. Checks whether the datatype and reduction op are supported
3. If both conditions are met, dispatches to an optimized GPU-aware algorithm
4. Otherwise, falls back transparently to the native `PMPI_Allreduce`

Fallback to native MPI occurs automatically for:
- Host (CPU) buffers (unless built with `--enable-host-path`)
- Unsupported datatypes or reduction operations
- Derived (non-contiguous) MPI datatypes
- Intercommunicators
- Zero-count calls
- Any internal error

## Algorithms

CAIL includes three allreduce algorithms:

| Algorithm           | Latency          | Bandwidth              | Auto-Selected          |
|---------------------|------------------|------------------------|------------------------|
| Recursive Doubling  | O(log₂ P)       | O(n · log₂ P)         | Yes (small messages)   |
| Rabenseifner        | O(2 · log₂ P)   | O(2n · (P-1)/P)       | Yes (large msgs, large scale) |
| Ring                | O(2 · (P-1))    | O(2n · (P-1)/P)       | Yes (large msgs, small scale) |

### Automatic Algorithm Selection (2D Dispatch)

Auto-dispatch selects an algorithm based on **message size** and **process
count** (nprocs):

**Small scale** (nprocs ≤ `CAIL_NPROCS_THRESHOLD`, default 4):

| Message Size                         | Algorithm            |
|------------------------------------- |----------------------|
| < `CAIL_MSG_SMALL_THRESHOLD`         | Recursive Doubling   |
| ≥ `CAIL_MSG_SMALL_THRESHOLD`         | Ring                 |

**Large scale** (nprocs > `CAIL_NPROCS_THRESHOLD`):

| Message Size                         | Algorithm            |
|------------------------------------- |----------------------|
| < `CAIL_MSG_SMALL_THRESHOLD`         | Recursive Doubling   |
| ≥ `CAIL_MSG_SMALL_THRESHOLD`         | Rabenseifner         |

**Guards applied before the matrix:**

- `count == 0` → immediate success (no communication)
- `count < pof2(nprocs)` → Recursive Doubling (Rabenseifner's reduce-scatter
  cannot distribute fewer elements than the nearest power-of-two ranks)
- Unsupported type/op → PMPI fallback
- Intercommunicator → PMPI fallback

**Non-power-of-two adjustment:** When nprocs is not a power of two and
nprocs ≥ 16, the effective small threshold is halved. Both recursive doubling
and Rabenseifner use rank folding for non-pof2, but the folding overhead makes
Rabenseifner relatively cheaper sooner.

Ring and Rabenseifner can also be forced via `CAIL_ALGO`.

Each algorithm can be individually enabled or disabled at configure time
(all enabled by default).

## GPU Kernels

CUDA reduction kernels (`cail_cuda_reduce.cu`) support all 20 MPI datatypes
across all 4 reduction operations (SUM, PROD, MAX, MIN) — a full 20×4 dispatch
matrix with no gaps. Float SUM uses vectorized `float4` access and double SUM
uses `double2` for maximum bandwidth.

## Prerequisites

- C compiler (gcc, icc, clang)
- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
- CUDA Toolkit (nvcc, cudart) — unless building with `--enable-host-path`
- Autotools (autoconf >= 2.69, automake, libtool)

## Building

```sh
./autogen.sh
./configure
make
make install
```

### Configure Options

| Option                          | Description                                          | Default  |
|---------------------------------|------------------------------------------------------|----------|
| `--enable-host-path`            | Build host-only path without GPU support             | no       |
| `--enable-debug`                | Debug build with `-g -O0`                            | no       |
| `--enable-recursive-doubling`   | Enable recursive-doubling algorithm                  | yes      |
| `--enable-ring`                 | Enable ring algorithm                                | yes      |
| `--enable-rabenseifner`         | Enable Rabenseifner algorithm                        | yes      |
| `--with-cuda=PATH`              | Path to CUDA toolkit installation                    | auto     |
| `--with-cuda-arch=SM`           | NVCC architecture flag (e.g. `sm_70`, `sm_80`)       | sm_70    |
| `--with-mpi=PATH`               | Path to MPI installation                             | auto     |

### Host-Only Build (No CUDA)

```sh
./configure --enable-host-path
```

This builds CAIL without any GPU dependency. The host-path and GPU backends
are mutually exclusive at compile time — `--enable-host-path` replaces CUDA
kernels with `MPI_Reduce_local` and GPU memory allocation with `malloc`/`free`.

## Usage

Link your application against `libcail` (the PMPI symbols override MPI
automatically):

```sh
mpicc -o my_app my_app.c -lcail
mpirun -np 8 ./my_app
```

Or use `LD_PRELOAD` to inject into an existing binary:

```sh
mpirun -np 8 -x LD_PRELOAD=/path/to/libcail.so ./my_app
```

A pkg-config file (`cail.pc`) is installed for build integration:

```sh
mpicc -o my_app my_app.c $(pkg-config --cflags --libs cail)
```

## Environment Variables

All environment variables are read once at first `MPI_Allreduce` call (lazy
initialization).

| Variable                 | Description                                                          | Default    |
|--------------------------|----------------------------------------------------------------------|------------|
| `CAIL_DEBUG`            | Enable debug logging to stderr. Set to any non-empty, non-`0` value. | off        |
| `CAIL_ALGO`             | Force a specific algorithm, bypassing auto-dispatch. See values below. | `auto`     |
| `CAIL_MIN_MSG_SIZE`     | Minimum message size (bytes) for CAIL to handle. Smaller messages pass through to native MPI. Set to `0` to disable passthrough. | `65536`    |
| `CAIL_MSG_SMALL_THRESHOLD`  | Message size (bytes) below which recursive doubling is used (within CAIL-handled range). | `8192`     |
| `CAIL_WARN`                 | Set to `0` to suppress `[cail WARN]` messages. | on         |

For non-power-of-two process counts ≥ 16, the effective small threshold is
automatically halved to account for rank-folding overhead.

For non-power-of-two process counts ≥ 16, the effective small threshold is
automatically halved to account for rank-folding overhead.

### `CAIL_ALGO` Values

| Value                | Algorithm            | Auto-Selected |
|----------------------|----------------------|---------------|
| `auto`               | Automatic (default)  | —             |
| `recursive_doubling` | Recursive Doubling   | Yes           |
| `rabenseifner`       | Rabenseifner         | Yes           |
| `ring`               | Ring                 | Yes           |

If a forced algorithm was disabled at compile time, CAIL aborts with an
error message (`[cail ERROR]`).

### Logging Levels

CAIL uses three logging levels on stderr:

| Prefix            | When                                                        |
|-------------------|-------------------------------------------------------------|
| `[cail]`         | Debug messages (algorithm selection, init). Only when `CAIL_DEBUG=1`. |
| `[cail WARN]`    | Unexpected fallbacks (e.g., preferred algorithm disabled, forced algo can't run). Disable with `CAIL_WARN=0`. |
| `[cail ERROR]`   | Initialization failures, invalid env var values. Always printed. |

### Tuning the 2D Dispatch Matrix

The environment variables map directly to the dispatch logic:

```
if msg_size < CAIL_MIN_MSG_SIZE:
    → PMPI_Allreduce  (passthrough to native MPI)

if count < pof2(nprocs):
    → recursive_doubling  (always, regardless of thresholds)

if msg < MSG_SMALL_THRESHOLD:
    → recursive_doubling

if msg >= MSG_SMALL_THRESHOLD:
    if nprocs <= NPROCS_THRESHOLD:
        → ring
    else:
        → rabenseifner

Non-pof2 adjustment: if nprocs is not a power of two and nprocs >= 16,
effective MSG_SMALL_THRESHOLD is halved.
```

### Examples

Force ring algorithm with debug output:
```sh
CAIL_DEBUG=1 CAIL_ALGO=ring mpirun -np 4 ./my_app
```

Lower the small-message threshold to use Rabenseifner for more message sizes:
```sh
CAIL_MSG_SMALL_THRESHOLD=4096 mpirun -np 8 ./my_app
```

Raise the threshold to keep recursive doubling active for larger messages:
```sh
CAIL_MSG_SMALL_THRESHOLD=32768 mpirun -np 8 ./my_app
```

Lower the passthrough threshold to let CAIL handle smaller messages:
```sh
CAIL_MIN_MSG_SIZE=4096 mpirun -np 4 ./my_app
```

Widen the "small scale" region for finer-grained tuning at higher process counts:
```sh
CAIL_NPROCS_THRESHOLD=16 mpirun -np 16 ./my_app
```

## Supported Types and Operations

**Base datatypes** (12):

| Type | Size |
|------|------|
| `MPI_CHAR` / `MPI_SIGNED_CHAR` | 1 byte |
| `MPI_UNSIGNED_CHAR` / `MPI_BYTE` | 1 byte |
| `MPI_SHORT` | 2 bytes |
| `MPI_UNSIGNED_SHORT` | 2 bytes |
| `MPI_INT` | 4 bytes |
| `MPI_UNSIGNED` | 4 bytes |
| `MPI_LONG` | 4 or 8 bytes |
| `MPI_UNSIGNED_LONG` | 4 or 8 bytes |
| `MPI_LONG_LONG` / `MPI_LONG_LONG_INT` | 8 bytes |
| `MPI_UNSIGNED_LONG_LONG` | 8 bytes |
| `MPI_FLOAT` | 4 bytes |
| `MPI_DOUBLE` | 8 bytes |

**C99 fixed-width aliases** (8 additional MPI handles, mapped to the base types above):

| Alias | Maps To |
|-------|---------|
| `MPI_INT8_T` | `MPI_CHAR` |
| `MPI_UINT8_T` | `MPI_UNSIGNED_CHAR` |
| `MPI_INT16_T` | `MPI_SHORT` |
| `MPI_UINT16_T` | `MPI_UNSIGNED_SHORT` |
| `MPI_INT32_T` | `MPI_INT` |
| `MPI_UINT32_T` | `MPI_UNSIGNED` |
| `MPI_INT64_T` | `MPI_LONG_LONG` |
| `MPI_UINT64_T` | `MPI_UNSIGNED_LONG_LONG` |

**Operations** (4): `MPI_SUM`, `MPI_PROD`, `MPI_MAX`, `MPI_MIN`

All 12 base types support all 4 operations (48 combinations, no gaps). Any
unsupported type or op falls back to native MPI with a warning:
`[cail WARN] falling back to PMPI_Allreduce: unsupported type=<name> op=<name>`

## Testing

The test suite is in `scripts/` (dev repository, not the cail submodule) and
runs across two hosts via `mpirun`.

### Test Scripts

| Script                | Purpose                                                      |
|-----------------------|--------------------------------------------------------------|
| `scripts/run_tests.sh`      | Full test suite: Phase 1 (debug verification), Phase 2 (correctness matrix), Phase 3 (env var dispatch verification) |
| `scripts/run_osu_compare.sh`| OSU Allreduce benchmark: baseline vs CAIL with latency/speedup comparison |

### `run_tests.sh` Phases

**Phase 1 — Debug Verification:** Runs with `CAIL_DEBUG=1` and verifies:
- cail initializes without errors
- Auto-dispatch selects recursive_doubling for small messages
- Forced algorithms (ring, rabenseifner) actually dispatch correctly
- Auto-dispatch selects different algorithms by message size

**Phase 2 — Correctness Matrix:** Runs all 6 test programs × 5 algorithms × 6
process counts (4, 3, 2, 1, 5, 7) = 180 test cases, verifying numerical
correctness.

**Phase 3 — Env Var Dispatch Verification:** Verifies that tunable
environment variables affect algorithm selection:
- `CAIL_MSG_SMALL_THRESHOLD`: lowering it shifts messages from recursive_doubling to rabenseifner
- `CAIL_NPROCS_THRESHOLD`: raising it changes the scale region for a given process count
- Non-pof2 threshold halving: verified at nprocs=17

### Test Programs

| Test                          | Coverage                                      |
|-------------------------------|-----------------------------------------------|
| `test_allreduce_basic`        | Basic correctness (float SUM) across 7 message sizes (small/medium/large) |
| `test_allreduce_inplace`      | `MPI_IN_PLACE` semantics                      |
| `test_allreduce_datatypes`    | All 20 supported datatypes × 4 ops             |
| `test_allreduce_algorithms`   | Each algorithm forced via `CAIL_ALGO`        |
| `test_allreduce_nonpof2`      | Non-power-of-2 process counts                 |
| `test_allreduce_edge`         | Edge cases (count=1, large counts)            |
| `bench_allreduce`             | Performance benchmark across message sizes    |

## Project Structure

```
cail/
  configure.ac              Autoconf input
  Makefile.am               Top-level automake input
  autogen.sh                Bootstrap script (runs autoreconf)
  cail.pc.in               pkg-config template
  m4/
    ax_check_mpi.m4         MPI detection macro
    ax_check_cuda.m4        CUDA detection macro
  src/
    core/
      cail.h               Public API header
      cail_internal.h      Internal macros, state, forward declarations
      cail_types.h/.c      MPI type/op mapping (20 types x 4 ops)
      cail_pmpi.c          PMPI interposition (MPI_Allreduce, MPI_Finalize)
      cail_init.c          Lazy init, env var parsing
      cail_buf.c           Single cached GPU buffer
    coll/allreduce/
      cail_allreduce.c     2D algorithm dispatch (msg_size × nprocs)
      cail_allreduce_impl.h  Algorithm function declarations
      cail_allreduce_recursive_doubling.c
      cail_allreduce_ring.c
      cail_allreduce_rabenseifner.c
    gpu/
      cail_gpu.h           Backend-agnostic GPU interface
      cail_host_reduce.c   Host-path fallback (MPI_Reduce_local)
      cuda/
        cail_cuda_reduce.cu  CUDA reduction kernels (vectorized)
        cail_cuda_mem.c      CUDA memory operations
      rocm/
        cail_rocm_reduce_stub.c  ROCm stubs (future)
  tests/
    test_runner.sh          mpirun wrapper script
    test_allreduce_*.c      Test programs
    bench_allreduce.c       Performance benchmark
```

## License

Copyright Cornelis Networks. All rights reserved.
