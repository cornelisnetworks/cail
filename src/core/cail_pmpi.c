/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "cail_internal.h"
#include "cail_types.h"
#include "../gpu/cail_gpu.h"

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rc;

    if (!cail_is_initialized()) {
        rc = cail_init(comm);
        if (rc != MPI_SUCCESS) {
            CAIL_ERR("cail_init failed (rc=%d), falling back to PMPI", rc);
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        }
    }

    if (count == 0)
        return MPI_SUCCESS;

    if (!cail_type_supported(datatype) || !cail_op_supported(op)) {
        CAIL_WARN("falling back to PMPI_Allreduce: unsupported type=%s op=%s",
                   cail_mpi_type_name(datatype), cail_mpi_op_name(op));
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    int is_inter;
    PMPI_Comm_test_inter(comm, &is_inter);
    if (is_inter) {
        CAIL_DBG("falling back to PMPI_Allreduce: intercommunicator");
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

#ifndef CAIL_HOST_PATH
    const void *check_buf = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;
    if (!cail_gpu_is_device_pointer(check_buf)) {
        CAIL_DBG("falling back to PMPI_Allreduce: host buffer");
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
#endif

    int type_size;
    PMPI_Type_size(datatype, &type_size);
    size_t msg_size = (size_t)count * (size_t)type_size;
    if (msg_size < cail_global_state.min_msg_size) {
        CAIL_DBG("falling back to PMPI_Allreduce: msg_size=%zu < min_msg_size=%zu",
                  msg_size, cail_global_state.min_msg_size);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    rc = cail_allreduce_dispatch(sendbuf, recvbuf, count, datatype, op, comm);
    if (rc != MPI_SUCCESS) {
        CAIL_DBG("cail_allreduce_dispatch failed (rc=%d), falling back to PMPI", rc);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    return MPI_SUCCESS;
}

int MPI_Finalize(void)
{
    cail_finalize();
    return PMPI_Finalize();
}
