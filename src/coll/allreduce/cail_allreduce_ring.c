/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* Ring Allreduce (reduce-scatter ring + allgather ring)
 *
 * Phase 1 (reduce-scatter): the buffer is split into P chunks. Over
 * P-1 steps, each rank sends one chunk to its right neighbor and
 * receives one from its left, reducing into the received chunk.
 * After P-1 steps each rank holds one fully-reduced chunk.
 *
 * Phase 2 (allgather): over another P-1 steps, each rank forwards
 * its reduced chunk around the ring until every rank has all chunks.
 *
 * Latency:    O(2 * (P-1)) messages
 * Bandwidth:  O(2n * (P-1)/P) — near-optimal, same as Rabenseifner
 *
 * Strengths: bandwidth-optimal for large messages, simple algorithm
 * with predictable nearest-neighbor communication pattern.
 *
 * Trade-offs: latency grows linearly with P (not logarithmically),
 * so it becomes expensive at high process counts. Each step only
 * involves nearest-neighbor communication, which can be advantageous
 * on topologies where nearest-neighbor is cheap.
 */

#include "../../core/cail_internal.h"
#include "../../gpu/cail_gpu.h"

#include <mpi.h>
#include <stdlib.h>
#include <stddef.h>

int cail_allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rank, nprocs;
    int rc;

    rc = PMPI_Comm_rank(comm, &rank);
    CAIL_CHECK(rc);
    rc = PMPI_Comm_size(comm, &nprocs);
    CAIL_CHECK(rc);

    if (count == 0)
        return MPI_SUCCESS;

    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    cail_op_t optype = cail_mpi_op_to_optype(op);
    int type_size = cail_type_size(dtype);
    if (dtype == CAIL_INVALID || optype == CAIL_OP_INVALID || type_size <= 0)
        return MPI_ERR_OP;

    size_t bufsize = (size_t)count * (size_t)type_size;

    if (nprocs == 1) {
        if (sendbuf != MPI_IN_PLACE && recvbuf != sendbuf) {
            rc = cail_gpu_memcpy(recvbuf, sendbuf, bufsize);
            if (rc != 0) return MPI_ERR_INTERN;
        }
        return MPI_SUCCESS;
    }

    if (count < nprocs) {
        CAIL_WARN("ring: count=%d < nprocs=%d, cannot distribute chunks, falling back to PMPI",
                   count, nprocs);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    const void *src = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;
    if (src != recvbuf) {
        if (cail_gpu_memcpy(recvbuf, src, bufsize) != 0)
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

    /* Phase 1: Reduce-scatter ring. Over P-1 steps, each rank sends
     * one chunk to its right neighbor and receives one from its left,
     * reducing into the received chunk. After P-1 steps each rank
     * holds exactly one chunk of the fully-reduced result. */
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

    /* Phase 2: Allgather ring. Over P-1 steps, each rank forwards
     * its fully-reduced chunk to the right. No reduction — just
     * copying. After P-1 steps every chunk has traveled the full
     * ring and every rank has the complete result. */
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
