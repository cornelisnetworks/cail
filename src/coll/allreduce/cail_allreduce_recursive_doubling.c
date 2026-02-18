/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../../core/cail_internal.h"
#include "../../gpu/cail_gpu.h"

#include <mpi.h>

int cail_allreduce_recursive_doubling(const void *sendbuf, void *recvbuf,
                                       int count, MPI_Datatype datatype,
                                       MPI_Op op, MPI_Comm comm)
{
    int rank, nprocs;
    int rc;

    rc = PMPI_Comm_rank(comm, &rank);
    CAIL_CHECK(rc);
    rc = PMPI_Comm_size(comm, &nprocs);
    CAIL_CHECK(rc);

    if (count == 0) {
        return MPI_SUCCESS;
    }

    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    cail_op_t optype = cail_mpi_op_to_optype(op);
    int type_size = cail_type_size(dtype);
    if (dtype == CAIL_INVALID || optype == CAIL_OP_INVALID || type_size <= 0) {
        return MPI_ERR_OP;
    }

    size_t bufsize = (size_t)count * (size_t)type_size;

    if (nprocs == 1) {
        if (sendbuf != MPI_IN_PLACE && recvbuf != sendbuf) {
            rc = cail_gpu_memcpy(recvbuf, sendbuf, bufsize);
            if (rc != 0) return MPI_ERR_INTERN;
        }
        return MPI_SUCCESS;
    }

    void *tmpbuf = NULL;
    rc = cail_buf_get_tmp(&tmpbuf, bufsize);
    if (rc != MPI_SUCCESS) {
        return rc;
    }

    const void *src = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;
    if (src != recvbuf) {
        if (cail_gpu_memcpy(recvbuf, src, bufsize) != 0) {
            rc = MPI_ERR_INTERN;
            goto cleanup;
        }
    }

    int pof2 = 1;
    while (pof2 <= nprocs) pof2 <<= 1;
    pof2 >>= 1;
    int rem = nprocs - pof2;
    int newrank;

    if (rank < 2 * rem) {
        if ((rank % 2) == 0) {
            rc = PMPI_Send(recvbuf, count, datatype, rank + 1, 0, comm);
            if (rc != MPI_SUCCESS) goto cleanup;
            newrank = -1;
        } else {
            rc = PMPI_Recv(tmpbuf, count, datatype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            if (rc != MPI_SUCCESS) goto cleanup;
            cail_gpu_flush_recv_buf(tmpbuf, bufsize);
            if (cail_gpu_reduce_local(tmpbuf, recvbuf, count, dtype, optype) != 0) {
                rc = MPI_ERR_INTERN;
                goto cleanup;
            }
            newrank = rank / 2;
        }
    } else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        int mask = 1;
        while (mask < pof2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? (newdst * 2 + 1) : (newdst + rem);

            rc = PMPI_Sendrecv(recvbuf, count, datatype, dst, 0,
                               tmpbuf, count, datatype, dst, 0,
                               comm, MPI_STATUS_IGNORE);
            if (rc != MPI_SUCCESS) goto cleanup;

            cail_gpu_flush_recv_buf(tmpbuf, bufsize);
            if (cail_gpu_reduce_local(tmpbuf, recvbuf, count, dtype, optype) != 0) {
                rc = MPI_ERR_INTERN;
                goto cleanup;
            }

            mask <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if ((rank % 2) == 0) {
            rc = PMPI_Recv(recvbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
            if (rc != MPI_SUCCESS) goto cleanup;
        } else {
            rc = PMPI_Send(recvbuf, count, datatype, rank - 1, 0, comm);
            if (rc != MPI_SUCCESS) goto cleanup;
        }
    }

cleanup:
    cail_buf_release_tmp(tmpbuf);
    return rc;
}
