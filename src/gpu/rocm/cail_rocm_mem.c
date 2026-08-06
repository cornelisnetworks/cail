/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../cail_gpu.h"
#include <hip/hip_runtime_api.h>
#include <stddef.h>

int  cail_rocm_init(void);
void cail_rocm_finalize(void);
int  cail_rocm_synchronize(void);

int cail_gpu_is_device_pointer(const void *ptr) {
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes(&attr, ptr);
    if (err != hipSuccess) {
        hipGetLastError();
        return 0;
    }
    /* ROCm 6.x may report unregistered host pointers as hipSuccess with
       hipMemoryTypeUnregistered; only device/managed types are GPU-resident. */
    return (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeManaged);
}

int cail_gpu_malloc(void **ptr, size_t size) {
    return (hipMalloc(ptr, size) == hipSuccess) ? 0 : -1;
}

int cail_gpu_free(void *ptr) {
    return (hipFree(ptr) == hipSuccess) ? 0 : -1;
}

int cail_gpu_memcpy(void *dst, const void *src, size_t size) {
    if (hipMemcpy(dst, src, size, hipMemcpyDefault) != hipSuccess)
        return -1;
    /* Send-side release fence: the staged buffer may next be read by an
       external GPU-aware MPI / NIC / GDRCopy engine, not by HIP stream work.
       Synchronize the copy's (default) stream before exposing dst to PMPI.
       Stream-scoped (not device-wide) so unrelated device work is not serialized. */
    return (hipStreamSynchronize(0) == hipSuccess) ? 0 : -1;
}

int cail_gpu_synchronize(void) { return cail_rocm_synchronize(); }
int cail_gpu_init(void)        { return cail_rocm_init(); }
void cail_gpu_finalize(void)   { cail_rocm_finalize(); }
