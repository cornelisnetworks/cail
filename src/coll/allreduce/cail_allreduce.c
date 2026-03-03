/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../../core/cail_internal.h"
#include "../../core/cail_types.h"
#include "cail_allreduce_impl.h"
#include <mpi.h>

int cail_allreduce_dispatch(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    int type_size = cail_type_size(dtype);
    size_t msg_size = (size_t)count * (size_t)type_size;

    int nprocs;
    PMPI_Comm_size(comm, &nprocs);

    if (cail_global_state.force_algo != CAIL_ALGO_AUTO) {
        switch (cail_global_state.force_algo) {
        case CAIL_ALGO_RECURSIVE_DOUBLING:
#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
            CAIL_DBG("algorithm=recursive_doubling (forced)");
            return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("recursive_doubling forced but disabled at compile time, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        case CAIL_ALGO_RING:
#ifdef CAIL_ENABLE_RING
            CAIL_DBG("algorithm=ring (forced)");
            return cail_allreduce_ring(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("ring forced but disabled at compile time, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        case CAIL_ALGO_RABENSEIFNER:
#ifdef CAIL_ENABLE_RABENSEIFNER
            CAIL_DBG("algorithm=rabenseifner (forced)");
            return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("rabenseifner forced but disabled at compile time, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        case CAIL_ALGO_TREE:
#ifdef CAIL_ENABLE_TREE
            CAIL_DBG("algorithm=tree (forced)");
            return cail_allreduce_tree(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("tree forced but disabled at compile time, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        default:
            break;
        }
    }

    int pof2 = cail_pof2(nprocs);

    /* Rabenseifner reduce-scatter needs count >= pof2 */
    if (count < pof2) {
#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
        CAIL_DBG("algorithm=recursive_doubling (count=%d < pof2=%d)", count, pof2);
        return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#else
        CAIL_WARN("recursive_doubling needed (count=%d < pof2=%d) but disabled, falling back to PMPI",
                   count, pof2);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
    }

    size_t effective_small = cail_global_state.small_threshold;
    if (!cail_is_pof2(nprocs) && nprocs >= 16)
        effective_small /= 2;

    if (nprocs <= cail_global_state.nprocs_small) {
        if (msg_size < effective_small) {
#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
            CAIL_DBG("algorithm=recursive_doubling (small_scale, msg_size=%zu < threshold=%zu)",
                      msg_size, effective_small);
            return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#elif defined(CAIL_ENABLE_RABENSEIFNER)
            CAIL_WARN("recursive_doubling preferred but disabled, using rabenseifner");
            return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("recursive_doubling preferred but disabled, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        } else if (msg_size < cail_global_state.medium_threshold) {
#ifdef CAIL_ENABLE_RABENSEIFNER
            CAIL_DBG("algorithm=rabenseifner (small_scale, threshold=%zu <= msg_size=%zu < medium=%zu)",
                      effective_small, msg_size, cail_global_state.medium_threshold);
            return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#elif defined(CAIL_ENABLE_RECURSIVE_DOUBLING)
            CAIL_WARN("rabenseifner preferred but disabled, using recursive_doubling");
            return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("rabenseifner preferred but disabled, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        } else {
#ifdef CAIL_ENABLE_RABENSEIFNER
            CAIL_DBG("algorithm=rabenseifner (small_scale, msg_size=%zu >= medium=%zu)",
                      msg_size, cail_global_state.medium_threshold);
            return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#elif defined(CAIL_ENABLE_RECURSIVE_DOUBLING)
            CAIL_WARN("rabenseifner preferred but disabled, using recursive_doubling");
            return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#else
            CAIL_WARN("no suitable algorithm available, falling back to PMPI");
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
        }
    }

    if (msg_size < effective_small) {
#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
        CAIL_DBG("algorithm=recursive_doubling (nprocs=%d, msg_size=%zu < threshold=%zu)",
                  nprocs, msg_size, effective_small);
        return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#elif defined(CAIL_ENABLE_RABENSEIFNER)
        CAIL_WARN("recursive_doubling preferred but disabled, using rabenseifner");
        return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#else
        CAIL_WARN("recursive_doubling preferred but disabled, falling back to PMPI");
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
    }

#ifdef CAIL_ENABLE_RABENSEIFNER
    CAIL_DBG("algorithm=rabenseifner (nprocs=%d, msg_size=%zu >= threshold=%zu)",
              nprocs, msg_size, effective_small);
    return cail_allreduce_rabenseifner(sendbuf, recvbuf, count, datatype, op, comm);
#elif defined(CAIL_ENABLE_RECURSIVE_DOUBLING)
    CAIL_WARN("rabenseifner preferred but disabled, using recursive_doubling");
    return cail_allreduce_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm);
#else
    CAIL_WARN("no suitable algorithm available, falling back to PMPI");
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#endif
}
