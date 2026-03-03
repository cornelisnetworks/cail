/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail_allreduce_rabenseifner.c — monolithic Rabenseifner allreduce
 * Implements non-power-of-two folding, recursive-halving reduce-scatter,
 * and recursive-doubling allgather in a single routine.
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

    if (count == 0) {
        return MPI_SUCCESS;
    }

    if (nprocs == 1) {
        if (sendbuf != MPI_IN_PLACE && recvbuf != sendbuf) {
            cail_datatype_t dtype_single = cail_mpi_type_to_dtype(datatype);
            int type_size_single = cail_type_size(dtype_single);
            if (dtype_single == CAIL_INVALID || type_size_single < 0) {
                return MPI_ERR_OP;
            }
            size_t bytes_single = (size_t)count * (size_t)type_size_single;
            int copy_rc = cail_gpu_memcpy(recvbuf, sendbuf, bytes_single);
            return (copy_rc == 0) ? MPI_SUCCESS : MPI_ERR_INTERN;
        }
        return MPI_SUCCESS;
    }

    cail_datatype_t dtype = cail_mpi_type_to_dtype(datatype);
    cail_op_t optype = cail_mpi_op_to_optype(op);
    int type_size = cail_type_size(dtype);
    if (dtype == CAIL_INVALID || optype == CAIL_OP_INVALID || type_size < 0) {
        return MPI_ERR_OP;
    }

    int pof2_local = cail_pof2(nprocs);
    if (count < pof2_local) {
        CAIL_WARN("rabenseifner: count=%d < pof2=%d (nprocs=%d), "
                   "cannot distribute in reduce-scatter, falling back to PMPI",
                   count, pof2_local, nprocs);
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    size_t bufsize = (size_t)count * (size_t)type_size;
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

    if (rank < 2 * rem) {
        if (rank % 2) {
            rc = PMPI_Send(recvbuf, count, datatype, rank - 1, 0, comm);
        } else {
            rc = PMPI_Recv(recvbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
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
