/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* Rabenseifner Allreduce (reduce-scatter + allgather)
 *
 * Phase 1 (reduce-scatter): recursive halving — each step, ranks
 * exchange half their data with a partner and reduce locally. After
 * log2(P) steps each rank holds 1/P of the final result.
 *
 * Phase 2 (allgather): recursive doubling — each step, ranks
 * exchange their portion with a partner, doubling the amount of
 * final data each holds. After log2(P) steps every rank has the
 * complete result.
 *
 * Latency:    O(2 * log2 P) messages (log2 for each phase)
 * Bandwidth:  O(2n * (P-1)/P) — near-optimal, each byte sent ~twice
 *
 * Best for large messages at scale: the bandwidth term is independent
 * of P (approaches 2n), so it scales well. The reduce-scatter halves
 * the working set each step, maximizing network utilization.
 *
 * Trade-offs: higher latency than recursive doubling (2x the steps),
 * so it loses to recursive doubling for small messages. Requires
 * count >= pof2(nprocs) for the reduce-scatter partitioning.
 * Non-power-of-two process counts are handled by folding excess
 * ranks before the main algorithm and unfolding afterward.
 */
#include "../../core/cail_internal.h"
#include "../../gpu/cail_gpu.h"

#include <mpi.h>
#include <stdlib.h>

int cail_allreduce_rabenseifner(const void *sendbuf, void *recvbuf, int count,
                                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rc = MPI_SUCCESS;
    int rank, nprocs;

    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &nprocs);

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


    int pof2_local = cail_pof2(nprocs);
    if (count < pof2_local) {
        CAIL_ERR("rabenseifner: count=%d < pof2=%d (nprocs=%d), "
                  "cannot distribute in reduce-scatter, aborting",
                   count, pof2_local, nprocs);
        PMPI_Abort(comm, MPI_ERR_INTERN);
    }

    /* bufsize already computed above */
    void *tmpbuf = NULL;
    rc = cail_buf_get_tmp(&tmpbuf, bufsize);
    if (rc != MPI_SUCCESS) {
        return rc;
    }

    const void *src = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;
    if (src != recvbuf) {
        int copy_rc = cail_gpu_memcpy(recvbuf, src, bufsize);
        if (copy_rc != 0) {
            rc = MPI_ERR_INTERN;
            goto cleanup;
        }
    }

    int pof2 = 1;
    while (pof2 <= nprocs) {
        pof2 <<= 1;
    }
    pof2 >>= 1;
    int rem = nprocs - pof2;
    int newrank;

    /* Phase 1: Non-power-of-two fold-in. Excess ranks (rank < 2*rem)
     * send their data to a neighbor and sit out the main algorithm.
     * Odd ranks in this range receive, reduce, and participate. */
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            rc = PMPI_Send(recvbuf, count, datatype, rank + 1, 0, comm);
            if (rc != MPI_SUCCESS) {
                goto cleanup;
            }
            newrank = -1;
        } else {
            rc = PMPI_Recv(tmpbuf, count, datatype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            if (rc != MPI_SUCCESS) {
                goto cleanup;
            }
            cail_gpu_flush_recv_buf(tmpbuf, bufsize);
            int red_rc = cail_gpu_reduce_local(tmpbuf, recvbuf, (size_t)count, dtype, optype);
            if (red_rc != 0) {
                rc = MPI_ERR_INTERN;
                goto cleanup;
            }
            newrank = rank / 2;
        }
    } else {
        newrank = rank - rem;
    }

    int *cnts = NULL;
    int *disps = NULL;

    if (newrank != -1) {
        cnts = (int *)malloc((size_t)pof2 * sizeof(int));
        disps = (int *)malloc((size_t)pof2 * sizeof(int));
        if (!cnts || !disps) {
            rc = MPI_ERR_NO_MEM;
            goto cleanup;
        }

        for (int i = 0; i < pof2; i++) {
            cnts[i] = count / pof2;
        }
        for (int i = 0; i < (count % pof2); i++) {
            cnts[i] += 1;
        }

        if (pof2 > 0) {
            disps[0] = 0;
        }
        for (int i = 1; i < pof2; i++) {
            disps[i] = disps[i - 1] + cnts[i - 1];
        }

        int mask = 0x1;
        int send_idx = 0;
        int recv_idx = 0;
        int last_idx = pof2;

        /* Phase 2a: Reduce-scatter via recursive halving. Each step,
         * exchange half the remaining data with a partner and reduce
         * locally. After log2(pof2) steps each rank holds 1/pof2 of
         * the fully-reduced result. */
        while (mask < pof2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            int send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (int i = send_idx; i < last_idx; i++) {
                    send_cnt += cnts[i];
                }
                for (int i = recv_idx; i < send_idx; i++) {
                    recv_cnt += cnts[i];
                }
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++) {
                    send_cnt += cnts[i];
                }
                for (int i = recv_idx; i < last_idx; i++) {
                    recv_cnt += cnts[i];
                }
            }

            rc = PMPI_Sendrecv((char *)recvbuf + ((size_t)disps[send_idx] * (size_t)type_size),
                               send_cnt, datatype, dst, 0,
                               (char *)tmpbuf + ((size_t)disps[recv_idx] * (size_t)type_size),
                               recv_cnt, datatype, dst, 0,
                               comm, MPI_STATUS_IGNORE);
            if (rc != MPI_SUCCESS) {
                goto cleanup;
            }

            cail_gpu_flush_recv_buf((char *)tmpbuf + ((size_t)disps[recv_idx] * (size_t)type_size),
                                     (size_t)recv_cnt * (size_t)type_size);
            int red_rc = cail_gpu_reduce_local((char *)tmpbuf + ((size_t)disps[recv_idx] * (size_t)type_size),
                                                (char *)recvbuf + ((size_t)disps[recv_idx] * (size_t)type_size),
                                                (size_t)recv_cnt, dtype, optype);
            if (red_rc != 0) {
                rc = MPI_ERR_INTERN;
                goto cleanup;
            }

            send_idx = recv_idx;
            mask <<= 1;
            if (mask < pof2) {
                last_idx = recv_idx + pof2 / mask;
            }
        }

        mask >>= 1;
        /* Phase 2b: Allgather via recursive doubling. Each step,
         * exchange reduced portions with a partner, doubling the
         * amount of final data each rank holds. After log2(pof2)
         * steps every active rank has the complete result. */
        while (mask > 0) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            int send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                if (mask != pof2 / 2) {
                    last_idx = last_idx + pof2 / (mask * 2);
                }
                recv_idx = send_idx + pof2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++) {
                    send_cnt += cnts[i];
                }
                for (int i = recv_idx; i < last_idx; i++) {
                    recv_cnt += cnts[i];
                }
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (int i = send_idx; i < last_idx; i++) {
                    send_cnt += cnts[i];
                }
                for (int i = recv_idx; i < send_idx; i++) {
                    recv_cnt += cnts[i];
                }
            }

            rc = PMPI_Sendrecv((char *)recvbuf + ((size_t)disps[send_idx] * (size_t)type_size),
                               send_cnt, datatype, dst, 0,
                               (char *)recvbuf + ((size_t)disps[recv_idx] * (size_t)type_size),
                               recv_cnt, datatype, dst, 0,
                               comm, MPI_STATUS_IGNORE);
            if (rc != MPI_SUCCESS) {
                goto cleanup;
            }

            cail_gpu_flush_recv_buf((char *)recvbuf + ((size_t)disps[recv_idx] * (size_t)type_size),
                                     (size_t)recv_cnt * (size_t)type_size);

            if (newrank > newdst) {
                send_idx = recv_idx;
            }

            mask >>= 1;
        }
    }

    /* Phase 3: Non-power-of-two unfold. Ranks that sat out in
     * phase 1 receive the final result from their neighbor. */
    if (rank < 2 * rem) {
        if (rank % 2) {
            rc = PMPI_Send(recvbuf, count, datatype, rank - 1, 0, comm);
        } else {
            rc = PMPI_Recv(recvbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
            if (rc == MPI_SUCCESS) {
                /* Flush the GPU-aware receive before the caller reads recvbuf. */
                cail_gpu_flush_recv_buf(recvbuf, bufsize);
            }
        }
    }

cleanup:
    if (cnts) {
        free(cnts);
    }
    if (disps) {
        free(disps);
    }
    cail_buf_release_tmp(tmpbuf);
    return rc;
}
