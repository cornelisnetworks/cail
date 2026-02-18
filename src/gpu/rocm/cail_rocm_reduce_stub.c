/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail_rocm_reduce_stub.c — ROCm/HIP stub implementation
 * These are placeholder stubs for future ROCm support.
 * Compiled only when HAVE_ROCM is defined (never in this version).
 */
#include "../cail_gpu.h"

int cail_gpu_is_device_pointer(const void *ptr) { (void)ptr; return 0; }
int cail_gpu_reduce_local(const void *in, void *inout, size_t count,
                            cail_datatype_t dtype, cail_op_t op)
{ (void)in; (void)inout; (void)count; (void)dtype; (void)op; return -1; }
int cail_gpu_malloc(void **ptr, size_t size) { (void)ptr; (void)size; return -1; }
int cail_gpu_free(void *ptr) { (void)ptr; return -1; }
int cail_gpu_memcpy(void *dst, const void *src, size_t size)
{ (void)dst; (void)src; (void)size; return -1; }
int cail_gpu_synchronize(void) { return -1; }
int cail_gpu_init(void) { return -1; }
void cail_gpu_finalize(void) {}
int cail_gpu_flush_rdma_writes(void) { return 0; }
int cail_gpu_flush_recv_buf(const void *recv_buf, size_t recv_bytes)
{ (void)recv_buf; (void)recv_bytes; return 0; }
