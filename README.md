# CAIL — Cornelis Allreduce Interposition Library

CAIL is a drop-in GPU-aware `MPI_Allreduce` optimization library. It uses the
MPI profiling interface (PMPI) to transparently intercept `MPI_Allreduce` calls
and route them through optimized algorithms with native CUDA/HIP reduction kernels.
No application changes are required, just `LD_PRELOAD` the library.

## Quick Start

### Prerequisites

- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
- CUDA Toolkit (nvcc, cudart) or ROCm Toolkit (hipcc, amdhip64)
- Autotools (autoconf >= 2.69, automake, libtool)

### Build

```sh
./autogen.sh
mkdir build && cd build
../configure --with-cuda=/path/to/cuda --with-cuda-arch=<arch>
make -j$(nproc)
make install
```

### Use

```sh
mpirun -np <N> -x LD_PRELOAD=/path/to/libcail.so ./my_app
```

That's it. CAIL intercepts `MPI_Allreduce` automatically. Unsupported calls
(host buffers, unsupported types, derived datatypes) fall back to native MPI.

### Verify

```sh
CAIL_DEBUG=1 mpirun -np <N> -x LD_PRELOAD=/path/to/libcail.so ./my_app
```

Look for `[cail] initialized:` and `[cail] algorithm=` on stderr to confirm CAIL is active and dispatching.

## Building

### Configure Options

| Option                          | Description                                          | Default  |
|---------------------------------|------------------------------------------------------|----------|
| `--with-cuda=PATH`              | Path to CUDA toolkit installation                    | auto     |
| `--with-cuda-arch=SM`           | NVCC architecture flag (e.g. `sm_70`, `sm_90`)       | `sm_70`  |
| `--with-rocm=PATH`              | Path to ROCm toolkit installation                    | auto     |
| `--with-rocm-arch=GFX`          | HIP offload architecture (e.g. `gfx90a`); empty lets hipcc autodetect | empty |
| `--with-mpi=PATH`               | Path to MPI installation                             | auto     |
| `--enable-host-path`            | Build without GPU support (host-only, uses `MPI_Reduce_local`) | no |
| `--enable-debug`                | Debug build with `-g -O0`                            | no       |
| `--enable-recursive-doubling`   | Enable recursive-doubling algorithm                  | yes      |
| `--enable-ring`                 | Enable ring algorithm                                | yes      |
| `--enable-rabenseifner`         | Enable Rabenseifner algorithm                        | yes      |

### ROCm Build

```sh
./configure --with-rocm=/opt/rocm --with-rocm-arch=gfx90a
```

### Host-Only Build (No CUDA/ROCm)

```sh
./configure --enable-host-path
```

This replaces CUDA kernels with `MPI_Reduce_local` and GPU memory operations
with `malloc`/`free`. Useful for development or CPU-only clusters.

### When CAIL Intercepts

CAIL intercepts `MPI_Allreduce` when all of these are true:

- Buffer resides on a CUDA or ROCm device (or built with `--enable-host-path`)
- Datatype is one of the 20 supported MPI types (see below)
- Operation is SUM, PROD, MAX, or MIN
- Communicator is an intracommunicator
- Message size >= `CAIL_MIN_MSG_SIZE` (default 32 KB)

Everything else falls back transparently to native `PMPI_Allreduce`.

## Environment Variables

All variables are read once at the first `MPI_Allreduce` call.

| Variable                     | Description                                              | Default  |
|------------------------------|----------------------------------------------------------|----------|
| `CAIL_ALGO`                  | Force a specific algorithm (see values below)            | `auto`   |
| `CAIL_MIN_MSG_SIZE`          | Minimum message size in bytes for CAIL to handle. Messages smaller than this pass through to native MPI. Set to `0` to disable passthrough. | `32768` (32 KB) |
| `CAIL_MSG_SMALL_THRESHOLD`   | Messages strictly below this size (in bytes) use the small-message algorithm; messages at or above use the large-message algorithm. | `262144` (256 KB) |
| `CAIL_NPROCS_THRESHOLD`      | Process count threshold (currently unused by auto-dispatch; reserved for future use). | `4` |
| `CAIL_DEBUG`                 | Enable debug logging to stderr. Set to any non-empty, non-`0` value. | off |
| `CAIL_WARN`                  | Set to `0` to suppress `[cail WARN]` messages.           | on       |

### `CAIL_ALGO` Values

| Value                | Algorithm            |
|----------------------|----------------------|
| `auto`               | Automatic (default)  |
| `recursive_doubling` | Recursive Doubling   |
| `ring`               | Ring                 |
| `rabenseifner`       | Rabenseifner         |

If a forced algorithm was disabled at compile time (`--disable-ring`, etc.),
CAIL aborts with `[cail ERROR]`.

### Logging

CAIL writes to stderr at three levels:

| Prefix            | When                                                        |
|-------------------|-------------------------------------------------------------|
| `[cail]`         | Debug messages (init, algorithm selection). Only when `CAIL_DEBUG=1`. |
| `[cail WARN]`    | Unexpected fallbacks (e.g. forced algorithm can't run for this count). Suppress with `CAIL_WARN=0`. |
| `[cail ERROR]`   | Fatal errors (invalid env var, forced algorithm compiled out). Always printed. |

## Supported Types and Operations

**20 MPI datatypes** (12 base + 8 C99 fixed-width aliases):

| Base Type | Size | C99 Alias |
|-----------|------|-----------|
| `MPI_CHAR` / `MPI_SIGNED_CHAR` | 1 byte | `MPI_INT8_T` |
| `MPI_UNSIGNED_CHAR` / `MPI_BYTE` | 1 byte | `MPI_UINT8_T` |
| `MPI_SHORT` | 2 bytes | `MPI_INT16_T` |
| `MPI_UNSIGNED_SHORT` | 2 bytes | `MPI_UINT16_T` |
| `MPI_INT` | 4 bytes | `MPI_INT32_T` |
| `MPI_UNSIGNED` | 4 bytes | `MPI_UINT32_T` |
| `MPI_LONG` | 4 or 8 bytes | — |
| `MPI_UNSIGNED_LONG` | 4 or 8 bytes | — |
| `MPI_LONG_LONG` / `MPI_LONG_LONG_INT` | 8 bytes | `MPI_INT64_T` |
| `MPI_UNSIGNED_LONG_LONG` | 8 bytes | `MPI_UINT64_T` |
| `MPI_FLOAT` | 4 bytes | — |
| `MPI_DOUBLE` | 8 bytes | — |

**4 operations:** `MPI_SUM`, `MPI_PROD`, `MPI_MAX`, `MPI_MIN`

All 20 types × 4 operations = 80 combinations are supported with no gaps.
Unsupported type/op combinations fall back to native MPI with a warning.

## Algorithm Selection

### Algorithms

| Algorithm           | Latency          | Bandwidth              | Best For |
|---------------------|------------------|------------------------|----------|
| Recursive Doubling  | O(log₂ P)       | O(n · log₂ P)         | Small messages, any process count |
| Ring                | O(2(P-1))        | O(2n · (P-1)/P)       | Large messages, small process counts |
| Rabenseifner        | O(2 · log₂ P)   | O(2n · (P-1)/P)       | Large messages, large process counts |

### Auto-Dispatch (Default)

When `CAIL_ALGO=auto` (the default), CAIL selects an algorithm based on
message size and process count. The dispatch logic, in order:

```
1. msg_size < CAIL_MIN_MSG_SIZE       → native MPI (passthrough)
2. count < pof2(nprocs)               → small-message algorithm
3. msg_size < CAIL_MSG_SMALL_THRESHOLD → small-message algorithm
4. otherwise                           → large-message algorithm
```

**Default algorithm mapping:**

| Dispatch Slot                      | Algorithm           |
|------------------------------------|---------------------|
| Small-message                      | Recursive Doubling  |
| Large-message                      | Rabenseifner        |

For non-power-of-two process counts >= 16, the effective small threshold is
automatically halved to account for rank-folding overhead.

### Tuning

All dispatch thresholds are tunable via environment variables. Here are common
tuning scenarios:

**Let CAIL handle smaller messages** (default passthrough is 32 KB):
```sh
CAIL_MIN_MSG_SIZE=4096 mpirun -np 4 -x LD_PRELOAD=... ./my_app
```

**Disable passthrough entirely** (CAIL handles all message sizes):
```sh
CAIL_MIN_MSG_SIZE=0 mpirun -np 4 -x LD_PRELOAD=... ./my_app
```

**Shift the small/large message boundary** (lower threshold → large-message algorithm used sooner):
```sh
CAIL_MSG_SMALL_THRESHOLD=4096 mpirun -np 8 -x LD_PRELOAD=... ./my_app
```

**Raise the boundary** (small-message algorithm used for larger messages):
```sh
CAIL_MSG_SMALL_THRESHOLD=32768 mpirun -np 8 -x LD_PRELOAD=... ./my_app
```

**Widen the "small scale" region** (use small-scale large-message algorithm at higher process counts):
```sh
CAIL_NPROCS_THRESHOLD=16 mpirun -np 16 -x LD_PRELOAD=... ./my_app
```

**Force a specific algorithm** (bypasses auto-dispatch entirely):
```sh
CAIL_ALGO=ring CAIL_DEBUG=1 mpirun -np 4 -x LD_PRELOAD=... ./my_app
```

## Testing

### Test Programs

| Test                          | Buffers | Coverage                                      |
|-------------------------------|---------|-----------------------------------------------|
| `test_allreduce_basic`        | CPU     | Float SUM across 7 message sizes, internal count sweep |
| `test_allreduce_correctness`  | CPU     | All 20 datatypes × 4 ops × {normal, MPI_IN_PLACE}. Requires `-c count`. |
| `test_allreduce_edge`         | CPU     | Edge cases: count=0, count=1, large counts, single process |
| `test_allreduce_correctness_gpu` | GPU  | All 20 datatypes × 4 ops × {normal, MPI_IN_PLACE}. Requires `-c count`. CUDA or ROCm build. |
| `bench_allreduce`             | CPU     | Performance benchmark across message sizes    |
| `bench_allreduce_gpu`         | GPU     | GPU performance benchmark. CUDA or ROCm build. |

### Test CLI

```sh
# Correctness test: -c count is required, -a algo is optional
mpirun -np 4 ./test_allreduce_correctness -c 4096 -a ring

# Basic test: -c and -a are optional (has internal count sweep)
mpirun -np 4 ./test_allreduce_basic

# Edge test: no arguments
mpirun -np 4 ./test_allreduce_edge
```

## License

Copyright (c) 2026 Cornelis Networks. All rights reserved.
