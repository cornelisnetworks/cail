/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../../core/cail.h"
#include "../../core/cail_internal.h"
#include "../../gpu/cail_gpu.h"


#define CAIL_ALLREDUCE_TREE_TAG 0

int cail_allreduce_tree(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rc = MPI_SUCCESS;
    int rank = 0, nprocs = 0;

    rc = PMPI_Comm_rank(comm, &rank);
    if (rc != MPI_SUCCESS)
        return rc;
    rc = PMPI_Comm_size(comm, &nprocs);
    if (rc != MPI_SUCCESS)
        return rc;

    if (count == 0)
        return MPI_SUCCESS;

    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    cail_op_t optype = cail_mpi_op_to_optype(op);
    if (dtype == CAIL_INVALID)
        return MPI_ERR_TYPE;
    if (optype == CAIL_OP_INVALID)
        return MPI_ERR_OP;

    int type_size = cail_type_size(dtype);
    if (type_size == 0)
        return MPI_ERR_TYPE;

    size_t bytes = (size_t)count * (size_t)type_size;
    void *tmp_buf = NULL;
    rc = cail_buf_get_tmp(&tmp_buf, bytes);
    if (rc != MPI_SUCCESS)
        return rc;

    if (sendbuf != MPI_IN_PLACE) {
        int grc = cail_gpu_memcpy(recvbuf, sendbuf, bytes);
        if (grc != 0) {
            cail_buf_release_tmp(tmp_buf);
            return MPI_ERR_INTERN;
        }
    }

    if (nprocs == 1) {
        cail_buf_release_tmp(tmp_buf);
        return MPI_SUCCESS;
    }

    int parent = (rank == 0) ? -1 : (rank - 1) / 2;
    int left_child = 2 * rank + 1;
    int right_child = 2 * rank + 2;
    int has_left = left_child < nprocs;
    int has_right = right_child < nprocs;

    if (has_left) {
        rc = PMPI_Recv(tmp_buf, count, datatype, left_child,
                       CAIL_ALLREDUCE_TREE_TAG, comm, MPI_STATUS_IGNORE);
        if (rc != MPI_SUCCESS) {
            cail_buf_release_tmp(tmp_buf);
            return rc;
        }
        cail_gpu_flush_recv_buf(tmp_buf, bytes);
        int grc = cail_gpu_reduce_local(tmp_buf, recvbuf, (size_t)count, dtype, optype);
        if (grc != 0) {
            cail_buf_release_tmp(tmp_buf);
            return MPI_ERR_INTERN;
        }
    }

    if (has_right) {
        rc = PMPI_Recv(tmp_buf, count, datatype, right_child,
                       CAIL_ALLREDUCE_TREE_TAG, comm, MPI_STATUS_IGNORE);
        if (rc != MPI_SUCCESS) {
            cail_buf_release_tmp(tmp_buf);
            return rc;
        }
        cail_gpu_flush_recv_buf(tmp_buf, bytes);
        int grc = cail_gpu_reduce_local(tmp_buf, recvbuf, (size_t)count, dtype, optype);
        if (grc != 0) {
            cail_buf_release_tmp(tmp_buf);
            return MPI_ERR_INTERN;
        }
    }

    if (rank != 0) {
        rc = PMPI_Send(recvbuf, count, datatype, parent,
                       CAIL_ALLREDUCE_TREE_TAG, comm);
        if (rc != MPI_SUCCESS) {
            cail_buf_release_tmp(tmp_buf);
            return rc;
        }
    }

    if (rank != 0) {
        rc = PMPI_Recv(recvbuf, count, datatype, parent,
                       CAIL_ALLREDUCE_TREE_TAG, comm, MPI_STATUS_IGNORE);
        if (rc != MPI_SUCCESS) {
            cail_buf_release_tmp(tmp_buf);
            return rc;
        }
    }

    if (has_left) {
        rc = PMPI_Send(recvbuf, count, datatype, left_child,
                       CAIL_ALLREDUCE_TREE_TAG, comm);
        if (rc != MPI_SUCCESS) {
            cail_buf_release_tmp(tmp_buf);
            return rc;
        }
    }
    if (has_right) {
        rc = PMPI_Send(recvbuf, count, datatype, right_child,
                       CAIL_ALLREDUCE_TREE_TAG, comm);
        if (rc != MPI_SUCCESS) {
            cail_buf_release_tmp(tmp_buf);
            return rc;
        }
    }

    cail_buf_release_tmp(tmp_buf);
    return MPI_SUCCESS;
}
