# Custom GPU-Aware Allreduce (CNCCL)

A high-performance, GPU-aware implementation of `MPI_Allreduce`. 

This library intercepts standard MPI calls to provide improved latency for small-to-medium message sizes using a **Recursive Doubling** algorithm.

## Overview

- **Location**: `cnccl/gpu/`
- **Goal**: Optimize collective communication latency by bypassing host staging and using efficient GPU kernels.
- **Method**: Uses `LD_PRELOAD` to override the default `MPI_Allreduce` symbol.

## Architecture & Features

### 1. Algorithm: Recursive Doubling
- Performs global reduction in $O(\log P)$ steps.
- Uses a virtual hypercube topology.
- **Non-Power-of-2 Support**: Automatically handles arbitrary process counts by folding extra ranks in a pre-processing step (Phase 1) and expanding them in a post-processing step (Phase 3).

### 2. GPU Optimizations
- **Direct GPU Access**: Detects device pointers and operates directly on GPU memory (CUDA).
- **Ping-Pong Buffering**: Alternates between `recvbuf` and a temporary buffer (`tmpbuf`) to overlap communication with computation and avoid strict in-place read/write dependencies.
- **Vectorized Kernels**: Custom reduction kernels (`gpu_reduce.cu`) using `float4` and `double2` vector types for maximum memory bandwidth.

## Build Instructions

### Prerequisites
- CUDA Toolkit (nvcc, cudart)
- MPI Implementation (OpenMPI, MPICH, etc.)
- C/C++ Compiler (gcc/g++)

### Compilation
Navigate to the `gpu` directory and run `make`:

```bash
cd gpu
sh build_gpu.sh
```

This generates the shared library: **`libcustom_allreduce.so`**.

## Usage

Inject the library into your MPI application using `LD_PRELOAD`.

```bash
# Example for OpenMPI
mpirun -np 8 -x LD_PRELOAD=$PWD/libcustom_allreduce.so ./your_gpu_app
```

_Note: Ensure the `LD_PRELOAD` environment variable is correctly propagated to all MPI ranks._

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUSTOM_ALLREDUCE_DEBUG` | Set to `1` to enable extensive debug logging (rank mapping, GPU UUIDs, etc). | `0` |
| `CUSTOM_ALLREDUCE_ALGO`  | *Experimental*. Select algorithm strategy (e.g., `recursive`). | `recursive` |

## Code Structure

- **`mpi_allreduce_wrapper.c`**: The interposition layer. Contains `dlsym` logic to find the real MPI function and handles device pointer detection.
- **`recursive_doubling.c`**: Core logic for the recursive doubling algorithm, including the optimized ping-pong buffering state machine.
- **`gpu_reduce.cu`**: CUDA kernels for summation and product reductions.
- **`common.h` / `recursive_doubling.h`**: Header files for shared type definitions and function prototypes.
- **`build_gpu.sh`**: Helper script for setting up environment variables.
