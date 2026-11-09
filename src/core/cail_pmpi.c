/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/*
 * PMPI interposition: this file defines MPI_Allreduce, MPI_Allreduce_c (when
 * the MPI library implements MPI-4 large-count bindings) and MPI_Finalize,
 * overriding the MPI library's versions at link time (or via LD_PRELOAD).
 * The real MPI implementations remain accessible through the PMPI_* prefix.
 *
 * Call chain:
 *   app calls MPI_Allreduce() or MPI_Allreduce_c()
 *     -> resolves to this file's definition
 *       -> CAIL handles it: cail_allreduce_dispatch() -> algorithm -> PMPI_*
 *       -> or passthrough:  PMPI_Allreduce() -> real MPI implementation
 */

#include "cail_internal.h"
#include "cail_types.h"
#include "../gpu/cail_gpu.h"

#include <limits.h>

/* Shared eligibility checks for every intercepted Allreduce entry point.
 * Returns 1 when CAIL should run the call, 0 when it must pass through to the
 * MPI library. `count` has already been checked to be > 0 and to fit in an
 * int. */
static int cail_should_intercept(const void *sendbuf, void *recvbuf, int count,
                                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    if (!cail_type_supported(datatype)) {
        CAIL_WARN("falling back to PMPI_Allreduce: unsupported type=%s",
                   cail_mpi_type_name(datatype));
        return 0;
    }
    if (!cail_op_supported(op)) {
        CAIL_WARN("falling back to PMPI_Allreduce: unsupported op=%s",
                   cail_mpi_op_name(op));
        return 0;
    }

    int is_inter;
    PMPI_Comm_test_inter(comm, &is_inter);
    if (is_inter) {
        CAIL_DBG("falling back to PMPI_Allreduce: intercommunicator");
        return 0;
    }

#ifndef CAIL_HOST_PATH
    if (!cail_gpu_is_device_pointer(recvbuf)) {
        CAIL_DBG("falling back to PMPI_Allreduce: recvbuf is not on GPU");
        return 0;
    }
    if (sendbuf != MPI_IN_PLACE && !cail_gpu_is_device_pointer(sendbuf)) {
        CAIL_DBG("falling back to PMPI_Allreduce: sendbuf is not on GPU");
        return 0;
    }
#else
    (void)sendbuf;
    (void)recvbuf;
#endif

    int type_size;
    PMPI_Type_size(datatype, &type_size);
    size_t msg_size = (size_t)count * (size_t)type_size;
    if (msg_size < cail_global_state.min_msg_size) {
        CAIL_DBG("falling back to PMPI_Allreduce: msg_size=%zu < min_msg_size=%zu",
                  msg_size, cail_global_state.min_msg_size);
        return 0;
    }

    return 1;
}

static void cail_ensure_initialized(MPI_Comm comm)
{
    if (!cail_is_initialized()) {
        int rc = cail_init(comm);
        if (rc != MPI_SUCCESS) {
            CAIL_ERR("cail_init failed (rc=%d), aborting", rc);
            PMPI_Abort(comm, rc);
        }
    }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    cail_ensure_initialized(comm);

    if (count == 0)
        return MPI_SUCCESS;

    if (!cail_should_intercept(sendbuf, recvbuf, count, datatype, op, comm))
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

    int rc = cail_allreduce_dispatch(sendbuf, recvbuf, count, datatype, op, comm);
    if (rc != MPI_SUCCESS) {
        CAIL_DBG("cail_allreduce_dispatch failed (rc=%d), falling back to PMPI", rc);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    return MPI_SUCCESS;
}

#if MPI_VERSION >= 4
/* MPI-4 large-count binding. Callers such as mpi4py >= 4.0 use this entry
 * point for every Allreduce, so without it they bypass CAIL entirely.
 * Counts that fit in an int take the same path as MPI_Allreduce; larger
 * counts pass through, since the algorithms index with int counts. */
int MPI_Allreduce_c(const void *sendbuf, void *recvbuf, MPI_Count count,
                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    cail_ensure_initialized(comm);

    if (count == 0)
        return MPI_SUCCESS;

    if (count < 0 || count > (MPI_Count)INT_MAX) {
        CAIL_DBG("falling back to PMPI_Allreduce_c: count=%lld exceeds INT_MAX",
                  (long long)count);
        return PMPI_Allreduce_c(sendbuf, recvbuf, count, datatype, op, comm);
    }

    int icount = (int)count;
    if (!cail_should_intercept(sendbuf, recvbuf, icount, datatype, op, comm))
        return PMPI_Allreduce_c(sendbuf, recvbuf, count, datatype, op, comm);

    int rc = cail_allreduce_dispatch(sendbuf, recvbuf, icount, datatype, op, comm);
    if (rc != MPI_SUCCESS) {
        CAIL_DBG("cail_allreduce_dispatch failed (rc=%d), falling back to PMPI", rc);
        return PMPI_Allreduce_c(sendbuf, recvbuf, count, datatype, op, comm);
    }

    return MPI_SUCCESS;
}
#endif

int MPI_Finalize(void)
{
    cail_finalize();
    return PMPI_Finalize();
}
