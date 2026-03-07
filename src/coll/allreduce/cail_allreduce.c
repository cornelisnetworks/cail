/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../../core/cail_internal.h"
#include "../../core/cail_types.h"
#include "cail_allreduce_impl.h"
#include <mpi.h>

/* ------------------------------------------------------------------ */
/* Algorithm call helpers                                              */
/* Each wraps the #ifdef guard so the dispatch matrix stays clean.     */
/* Forced mode: abort if compiled out (user explicitly requested it).  */
/* Auto mode:   WARN + PMPI fallback if compiled out.                  */
/* ------------------------------------------------------------------ */

static inline int call_recursive_doubling(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        const char *reason)
{
#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
    CAIL_DBG("algorithm=recursive_doubling (%s)", reason);
    return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count,
                                              datatype, op, comm);
#else
    CAIL_WARN("recursive_doubling selected (%s) but disabled at compile time, "
              "falling back to PMPI", reason);
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
}

static inline int call_rabenseifner(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        const char *reason)
{
#ifdef CAIL_ENABLE_RABENSEIFNER
    CAIL_DBG("algorithm=rabenseifner (%s)", reason);
    return cail_allreduce_rabenseifner(sendbuf, recvbuf, count,
                                        datatype, op, comm);
#else
    CAIL_WARN("rabenseifner selected (%s) but disabled at compile time, "
              "falling back to PMPI", reason);
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
}

static inline int call_ring(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        const char *reason)
{
#ifdef CAIL_ENABLE_RING
    CAIL_DBG("algorithm=ring (%s)", reason);
    return cail_allreduce_ring(sendbuf, recvbuf, count,
                                datatype, op, comm);
#else
    CAIL_WARN("ring selected (%s) but disabled at compile time, "
              "falling back to PMPI", reason);
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
}

/* ------------------------------------------------------------------ */
/* Dispatch helpers                                                    */
/* Map the two dispatch outcomes to the appropriate algorithm.         */
/* ------------------------------------------------------------------ */

static inline int dispatch_small_msg(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        const char *reason)
{
    return call_recursive_doubling(sendbuf, recvbuf, count, datatype, op,
                                    comm, reason);
}

static inline int dispatch_large_msg(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        const char *reason)
{
    /* Rabenseifner is the best large-message algorithm at all tested GPU
     * scales (np=2..8).  Ring is kept as a forced-only option. */
    return call_rabenseifner(sendbuf, recvbuf, count, datatype, op,
                              comm, reason);
}

/* ------------------------------------------------------------------ */
/* 2D dispatch matrix                                                  */
/* ------------------------------------------------------------------ */

int cail_allreduce_dispatch(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    int type_size = cail_type_size(dtype);
    size_t msg_size = (size_t)count * (size_t)type_size;

    int nprocs;
    PMPI_Comm_size(comm, &nprocs);

    /* --- Forced algorithm (CAIL_ALGO env var) --- */

    if (cail_global_state.force_algo != CAIL_ALGO_AUTO) {
        int pof2 = cail_pof2(nprocs);

        switch (cail_global_state.force_algo) {
        case CAIL_ALGO_RECURSIVE_DOUBLING:
#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
            CAIL_DBG("algorithm=recursive_doubling (forced)");
            return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_ERR("CAIL_ALGO=recursive_doubling forced but disabled at compile time, aborting");
            PMPI_Abort(comm, MPI_ERR_INTERN);
#endif
        case CAIL_ALGO_RING:
#ifdef CAIL_ENABLE_RING
            CAIL_DBG("algorithm=ring (forced)");
            return cail_allreduce_ring(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_ERR("CAIL_ALGO=ring forced but disabled at compile time, aborting");
            PMPI_Abort(comm, MPI_ERR_INTERN);
#endif
        case CAIL_ALGO_RABENSEIFNER:
#ifdef CAIL_ENABLE_RABENSEIFNER
            /* Rabenseifner needs count >= pof2; fall back to PMPI when it cannot run */
            if (count < pof2) {
                CAIL_WARN("forced rabenseifner but count=%d < pof2=%d, falling back to PMPI",
                          count, pof2);
                return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
            }
            CAIL_DBG("algorithm=rabenseifner (forced)");
            return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_ERR("CAIL_ALGO=rabenseifner forced but disabled at compile time, aborting");
            PMPI_Abort(comm, MPI_ERR_INTERN);
#endif
        default:
            break;
        }
    }

    /* --- Auto-dispatch --- */

    int pof2 = cail_pof2(nprocs);

    /* Rabenseifner reduce-scatter needs count >= pof2 */
    if (count < pof2)
        return dispatch_small_msg(sendbuf, recvbuf, count, datatype, op, comm,
                                   "count < pof2");

    /* Non-pof2 adjustment: halve threshold when rank folding adds overhead */
    size_t effective_small = cail_global_state.msg_small_threshold;
    if (!cail_is_pof2(nprocs) && nprocs >= 16)
        effective_small /= 2;

    if (msg_size < effective_small)
        return dispatch_small_msg(sendbuf, recvbuf, count, datatype, op, comm,
                                   "msg_size < msg_small_threshold");

    return dispatch_large_msg(sendbuf, recvbuf, count, datatype, op, comm,
                               "msg_size >= msg_small_threshold");
}
