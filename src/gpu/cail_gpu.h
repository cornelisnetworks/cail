/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail_gpu.h — Backend-agnostic GPU interface for cail
 * Implemented by: CUDA (cail_cuda_reduce.cu + cail_cuda_mem.c)
 *                 Host-path (cail_host_reduce.c)
 *                 ROCm stub (cail_rocm_reduce_stub.c)
 */
#ifndef CAIL_GPU_H
#define CAIL_GPU_H

#include "cail_types.h"
#include <stddef.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

int cail_gpu_is_device_pointer(const void *ptr);

/* Element-wise reduction on GPU: inout[i] = inout[i] op in[i] for i in [0, count) */
int cail_gpu_reduce_local(const void *in, void *inout, size_t count,
                            cail_datatype_t dtype, cail_op_t op);

/* Allocate GPU buffer of `size` bytes */
int cail_gpu_malloc(void **ptr, size_t size);

/* Free GPU buffer */
int cail_gpu_free(void *ptr);

/* Copy `size` bytes between GPU buffers or GPU<->host (direction auto-detected) */
int cail_gpu_memcpy(void *dst, const void *src, size_t size);

/* Synchronize all pending GPU operations */
int cail_gpu_synchronize(void);

/* Flush NIC-originated RDMA writes targeting a specific receive buffer so that
 * subsequent GPU kernels see the data.  A PCIe read from the last byte of the
 * buffer drains all prior posted writes to that region.
 * Returns 0 on success or if flush is not needed/supported. */
int cail_gpu_flush_recv_buf(const void *recv_buf, size_t recv_bytes);

/* Initialize GPU subsystem (create streams, verify device) */
int cail_gpu_init(void);

/* Finalize GPU subsystem (destroy streams, free resources) */
void cail_gpu_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* CAIL_GPU_H */
