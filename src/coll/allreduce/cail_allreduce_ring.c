/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../../core/cail_internal.h"
#include "../../gpu/cail_gpu.h"

#include <mpi.h>
#include <stdlib.h>
#include <stddef.h>

int cail_allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rank, nprocs;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &nprocs);

    if (count == 0)
        return MPI_SUCCESS;

    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    if (dtype == CAIL_INVALID)
        return MPI_ERR_TYPE;

    cail_op_t optype = cail_mpi_op_to_optype(op);
    if (optype == CAIL_OP_INVALID)
        return MPI_ERR_OP;

    int type_size = cail_type_size(dtype);
    if (type_size <= 0)
        return MPI_ERR_TYPE;

    const void *src = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;

    if (nprocs == 1) {
        if (src != recvbuf) {
            size_t bytes = (size_t)count * (size_t)type_size;
            int rc = cail_gpu_memcpy(recvbuf, src, bytes);
            return (rc == 0) ? MPI_SUCCESS : MPI_ERR_INTERN;
        }
        return MPI_SUCCESS;
    }

    if (count < nprocs) {
        CAIL_WARN("ring: count=%d < nprocs=%d, cannot distribute chunks, falling back to PMPI",
                   count, nprocs);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    size_t total_bytes = (size_t)count * (size_t)type_size;
    if (src != recvbuf) {
        int rc = cail_gpu_memcpy(recvbuf, src, total_bytes);
        if (rc != 0)
            return MPI_ERR_INTERN;
    }

    int base = count / nprocs;
    int rem  = count % nprocs;

    int *chunk_size = (int *)malloc((size_t)nprocs * sizeof(int));
    size_t *chunk_offset = (size_t *)malloc((size_t)nprocs * sizeof(size_t));
    if (!chunk_size || !chunk_offset) {
        free(chunk_size);
        free(chunk_offset);
        return MPI_ERR_NO_MEM;
    }

    int max_chunk = 0;
    for (int i = 0; i < nprocs; i++) {
        chunk_size[i] = base + (i < rem ? 1 : 0);
        if (chunk_size[i] > max_chunk)
            max_chunk = chunk_size[i];
    }

    chunk_offset[0] = 0;
    for (int i = 1; i < nprocs; i++) {
        chunk_offset[i] = chunk_offset[i - 1] + (size_t)chunk_size[i - 1];
    }

    void *tmp_buf = NULL;
    size_t tmp_bytes = (size_t)max_chunk * (size_t)type_size;
    int mpi_errno = cail_buf_get_tmp(&tmp_buf, tmp_bytes);
    if (mpi_errno != MPI_SUCCESS) {
        free(chunk_size);
        free(chunk_offset);
        return mpi_errno;
    }

    char *recv_bytes = (char *)recvbuf;
    int left  = (rank - 1 + nprocs) % nprocs;
    int right = (rank + 1) % nprocs;

    for (int step = 0; step < nprocs - 1; step++) {
        int send_chunk = (rank - step + nprocs) % nprocs;
        int recv_chunk = (rank - step - 1 + nprocs) % nprocs;

        int send_count = chunk_size[send_chunk];
        int recv_count = chunk_size[recv_chunk];

        char *send_ptr = recv_bytes + chunk_offset[send_chunk] * (size_t)type_size;
        char *recv_ptr = recv_bytes + chunk_offset[recv_chunk] * (size_t)type_size;

        mpi_errno = PMPI_Sendrecv(send_ptr, send_count, datatype, right, 0,
                                   tmp_buf, recv_count, datatype, left, 0,
                                   comm, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS) goto cleanup;

        cail_gpu_flush_recv_buf(tmp_buf, (size_t)recv_count * (size_t)type_size);

        int rc = cail_gpu_reduce_local(tmp_buf, recv_ptr, (size_t)recv_count, dtype, optype);
        if (rc != 0) {
            mpi_errno = MPI_ERR_INTERN;
            goto cleanup;
        }
    }

    /* ---- Allgather phase ---- */
    for (int step = 0; step < nprocs - 1; step++) {
        int send_chunk = (rank - step + 1 + nprocs) % nprocs;
        int recv_chunk = (rank - step + nprocs) % nprocs;

        int send_count = chunk_size[send_chunk];
        int recv_count = chunk_size[recv_chunk];

        char *send_ptr = recv_bytes + chunk_offset[send_chunk] * (size_t)type_size;
        char *recv_ptr = recv_bytes + chunk_offset[recv_chunk] * (size_t)type_size;

        mpi_errno = PMPI_Sendrecv(send_ptr, send_count, datatype, right, 0,
                                   recv_ptr, recv_count, datatype, left, 0,
                                   comm, MPI_STATUS_IGNORE);
        if (mpi_errno != MPI_SUCCESS) goto cleanup;

        cail_gpu_flush_recv_buf(recv_ptr, (size_t)recv_count * (size_t)type_size);
    }

cleanup:
    cail_buf_release_tmp(tmp_buf);
    free(chunk_size);
    free(chunk_offset);
    return mpi_errno;
}
