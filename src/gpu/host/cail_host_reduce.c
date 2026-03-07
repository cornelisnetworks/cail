/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail_host_reduce.c — Host-path implementation of cail_gpu.h interface.
 * Compiled only when CAIL_HOST_PATH is defined (--enable-host-path).
 * Uses malloc/free for memory and MPI_Reduce_local for reductions.
 */
#include "../cail_gpu.h"
#include "../../core/cail_types.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>


/* Host-path always returns 0 (not device memory) */
int cail_gpu_is_device_pointer(const void *ptr)
{
    (void)ptr;
    return 0;
}

/* Perform element-wise reduction using MPI_Reduce_local */
int cail_gpu_reduce_local(const void *in, void *inout, size_t count,
                            cail_datatype_t dtype, cail_op_t op)
{
    MPI_Datatype mpi_dt = cail_dtype_to_mpi_type(dtype);
    MPI_Op       mpi_op = cail_optype_to_mpi_op(op);

    if (mpi_dt == MPI_DATATYPE_NULL || mpi_op == MPI_OP_NULL) {
        return -1;
    }

    return PMPI_Reduce_local(in, inout, (int)count, mpi_dt, mpi_op);
}

/* Allocate host memory using malloc */
int cail_gpu_malloc(void **ptr, size_t size)
{
    *ptr = malloc(size);
    return (*ptr) ? 0 : -1;
}

/* Free host memory using free */
int cail_gpu_free(void *ptr)
{
    free(ptr);
    return 0;
}

/* Copy memory using memcpy (host-to-host) */
int cail_gpu_memcpy(void *dst, const void *src, size_t size)
{
    memcpy(dst, src, size);
    return 0;
}

/* No-op synchronization for host-path */
int cail_gpu_synchronize(void)
{
    return 0;
}

/* No-op initialization for host-path */
int cail_gpu_init(void)
{
    return 0;
}

/* No-op finalization for host-path */
void cail_gpu_finalize(void)
{
}


int cail_gpu_flush_recv_buf(const void *recv_buf, size_t recv_bytes)
{
    (void)recv_buf;
    (void)recv_bytes;
    return 0;
}
