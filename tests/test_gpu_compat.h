/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#ifndef TEST_GPU_COMPAT_H
#define TEST_GPU_COMPAT_H

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
typedef hipError_t gpuError_t;
#define gpuSuccess hipSuccess
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetErrorString hipGetErrorString
#define GPU_KERNEL_LAUNCH(kernel, grid, block, shmem, stream, ...) \
    hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)
#define GPU_ERROR_PREFIX "HIP"
#else
#include <cuda_runtime.h>
typedef cudaError_t gpuError_t;
#define gpuSuccess cudaSuccess
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetErrorString cudaGetErrorString
#define GPU_KERNEL_LAUNCH(kernel, grid, block, shmem, stream, ...) \
    kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#define GPU_ERROR_PREFIX "CUDA"
#endif

#define GPU_CHECK(call) do {                                           \
    gpuError_t _e = (call);                                            \
    if (_e != gpuSuccess) {                                            \
        fprintf(stderr, "%s error %s:%d: %s\n", GPU_ERROR_PREFIX,   \
                __FILE__, __LINE__, gpuGetErrorString(_e));            \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                   \
} while (0)

#endif /* TEST_GPU_COMPAT_H */
