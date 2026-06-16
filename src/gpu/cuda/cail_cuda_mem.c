/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../cail_gpu.h"
#include <cuda_runtime.h>
#include <stddef.h>

int  cail_cuda_init(void);
void cail_cuda_finalize(void);
int  cail_cuda_synchronize(void);

int cail_gpu_is_device_pointer(const void *ptr) {
    struct cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
}

int cail_gpu_malloc(void **ptr, size_t size) {
    return (cudaMalloc(ptr, size) == cudaSuccess) ? 0 : -1;
}

int cail_gpu_free(void *ptr) {
    return (cudaFree(ptr) == cudaSuccess) ? 0 : -1;
}

int cail_gpu_memcpy(void *dst, const void *src, size_t size) {
    if (cudaMemcpy(dst, src, size, cudaMemcpyDefault) != cudaSuccess)
        return -1;
    /* Fence the staged copy before MPI/NIC reads the buffer. */
    return (cudaStreamSynchronize(0) == cudaSuccess) ? 0 : -1;
}

int cail_gpu_synchronize(void) { return cail_cuda_synchronize(); }
int cail_gpu_init(void)        { return cail_cuda_init(); }
void cail_gpu_finalize(void)   { cail_cuda_finalize(); }
