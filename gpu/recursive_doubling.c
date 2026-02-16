/*
 * Recursive Doubling Allreduce Implementation
 * 
 * This file implements the recursive doubling algorithm for MPI_Allreduce.
 * The algorithm is latency-optimized and performs well for small to medium
 * message sizes where the number of communication steps (log P) matters more
 * than the bandwidth utilization per step.
 * 
 * Notation:
 *   - P = number of processes (nprocs)
 *   - N = total number of elements (count)
 * 
 * Algorithm overview:
 *   - In each of log2(P) steps, ranks exchange data with partners at
 *     increasing distances (1, 2, 4, 8, ...) in a hypercube pattern
 *   - After log2(P) steps, all ranks have the global reduced result
 *   - Communication volume per rank: N * log2(P)
 *   - Latency cost: log2(P) messages
 * 
 * GPU-aware features:
 *   - Detects GPU memory pointers and avoids unnecessary host staging
 *   - Uses CUDA kernels for local reductions when possible
 *   - Falls back to host reductions for unsupported operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "recursive_doubling.h"
#include "common.h"

/*
 * Recursive Doubling Allreduce Algorithm
 * 
 * Efficient for small-to-medium messages. O(log P) steps where P is the
 * number of processes. Each step exchanges with partner at distance 2^k,
 * then performs local reduction.
 * 
 * For non-power-of-2 process counts: extra processes send data to lower
 * ranks before the main algorithm, then receive final results after.
 * 
 * Example with 6 processes (pof2=4, rem=2):
 *   Phase 1: Ranks 0,1,2,3 form pairs: (0,1), (2,3)
 *            Ranks 1,3 send to 0,2 and wait
 *            Ranks 0,2,4,5 participate as virtual ranks 0,1,2,3
 *   Phase 2: Recursive doubling with 4 ranks (log2(4) = 2 steps)
 *   Phase 3: Ranks 0,2 send final result back to 1,3
 */
int recursive_doubling_allreduce(const void *sendbuf, void *recvbuf,
                                 int count, MPI_Datatype datatype,
                                 MPI_Op op, MPI_Comm comm)
{
    int rank, nprocs, type_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &type_size);
    
    size_t bufsize = (size_t)count * type_size;
    
    /*
     * Detect if we're working with GPU memory.
     * If either sendbuf or recvbuf is on GPU, we'll use GPU-aware operations.
     * This avoids unnecessary staging through host memory.
     */
    int use_gpu = is_device_pointer(recvbuf);
    if (sendbuf != MPI_IN_PLACE && is_device_pointer(sendbuf)) {
        use_gpu = 1;
    }
    
    /*
     * Handle MPI_IN_PLACE: when sendbuf is MPI_IN_PLACE, the input data
     * is already in recvbuf. Otherwise, we need to copy sendbuf to recvbuf
     * to initialize the accumulation buffer.
     */
    const void *src = sendbuf;
    if (sendbuf == MPI_IN_PLACE) {
        src = recvbuf;
    }
    
    /* Initialize recvbuf with input data (copy sendbuf -> recvbuf if needed) */
    if (src != recvbuf) {
        if (use_gpu) {
            cudaError_t err = cudaMemcpy(recvbuf, src, bufsize, cudaMemcpyDefault);
            if (err != cudaSuccess) {
                fprintf(stderr, "Error: cudaMemcpy failed in recursive_doubling_allreduce: %s\n",
                        cudaGetErrorString(err));
                return MPI_ERR_OTHER;
            }
        } else {
            memcpy(recvbuf, src, bufsize);
        }
    }
    
    /* Trivial case: single process, no communication needed */
    if (nprocs == 1) {
        return MPI_SUCCESS;
    }
    
    /*
     * Allocate temporary buffer for receiving data from partner ranks.
     * This buffer is reused across all communication steps.
     * For GPU operations, we allocate on the GPU to avoid host staging.
     */
    void *tmpbuf = NULL;
    if (use_gpu) {
        cudaError_t err = cudaMalloc(&tmpbuf, bufsize);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaMalloc failed in recursive_doubling_allreduce: %s\n",
                    cudaGetErrorString(err));
            return MPI_ERR_NO_MEM;
        }
    } else {
        tmpbuf = malloc(bufsize);
        if (tmpbuf == NULL) {
            fprintf(stderr, "Error: malloc failed in recursive_doubling_allreduce\n");
            return MPI_ERR_NO_MEM;
        }
    }
    
    /*
     * Find the largest power of 2 that is <= nprocs.
     * The main recursive doubling algorithm works on a virtual hypercube
     * of size pof2. Extra processes (rem = nprocs - pof2) are handled
     * separately in the pre/post-processing steps.
     * 
     * Example: nprocs=6 -> pof2=4, rem=2
     */
    int pof2 = 1;
    while (pof2 <= nprocs) pof2 <<= 1;
    pof2 >>= 1;
    
    int rem = nprocs - pof2;  /* Number of "extra" processes beyond power-of-2 */
    int newrank = -1;         /* Virtual rank for main algorithm (-1 = not participating) */
    
    /*
     * ========== Phase 1: Handle non-power-of-2 process counts ==========
     * 
     * When nprocs is not a power of 2, we have 'rem' extra processes.
     * Strategy:
     *   - First 2*rem processes form pairs: (0,1), (2,3), (4,5), ...
     *   - Odd ranks (1,3,5,...) send their data to even ranks (0,2,4,...)
     *   - Even ranks reduce received data and participate in main algorithm
     *   - Ranks >= 2*rem participate directly with adjusted newrank
     * 
     * Example with 6 processes:
     *   Ranks 0,1,2,3 form pairs. Rank 1 -> 0, Rank 3 -> 2
     *   Ranks 0,2,4,5 participate as virtual ranks 0,1,2,3
     *   Ranks 1,3 wait until after main algorithm
     */
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /*
             * Even rank: Receive data from odd partner, reduce it locally,
             * then participate in main algorithm with newrank = rank/2
             */
            MPI_Recv(tmpbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
            gpu_reduce_local(tmpbuf, recvbuf, count, datatype, op, use_gpu, bufsize);
            newrank = rank / 2;
        } else {
            /*
             * Odd rank: Send data to even partner and wait.
             * Will receive final result in Phase 3.
             */
            MPI_Send(recvbuf, count, datatype, rank - 1, 0, comm);
            newrank = -1;  /* Don't participate in main algorithm */
        }
    } else {
        /*
         * Ranks >= 2*rem participate in main algorithm with adjusted rank.
         * The adjustment accounts for the rem ranks that were paired off.
         * Example: rank 4 with rem=2 becomes newrank 2 (4 - 2 = 2)
         */
        newrank = rank - rem;
    }
    
    /*
     * ========== Phase 2: Main recursive doubling algorithm ==========
     * 
     * Performs log2(pof2) communication steps in a hypercube pattern.
     * Only ranks with newrank != -1 participate.
     * 
     * In step k (mask = 2^k), each rank exchanges data with partner at
     * distance 2^k and reduces locally. After log2(pof2) steps, all
     * participating ranks have the complete reduced result.
     * 
     * Communication pattern example with 4 ranks:
     *   Step 0 (mask=1): 0<->1, 2<->3  (distance 1)
     *   Step 1 (mask=2): 0<->2, 1<->3  (distance 2)
     * After 2 steps, all 4 ranks have the global result.
     */
    if (newrank != -1) {
        int mask = 1;
        while (mask < pof2) {
            /*
             * XOR with mask gives the partner rank in the virtual hypercube.
             * This creates the recursive doubling pattern: partners at
             * distance 1, 2, 4, 8, ... in successive steps.
             */
            int newdst = newrank ^ mask;
            
            /*
             * Convert virtual rank back to actual MPI rank.
             * Ranks in first 2*rem used even numbers (0,2,4,...)
             * Ranks >= 2*rem are offset by rem.
             */
            int dst;
            if (newdst < rem) {
                dst = newdst * 2;  /* Partner is an even rank in first 2*rem */
            } else {
                dst = newdst + rem;  /* Partner is a high rank, adjust offset */
            }
            
            /*
             * Exchange data with partner and reduce.
             * MPI_Sendrecv is used for deadlock-free bidirectional exchange.
             * After receiving, perform local reduction: recvbuf = recvbuf OP tmpbuf
             */
            MPI_Sendrecv(recvbuf, count, datatype, dst, 0,
                         tmpbuf, count, datatype, dst, 0,
                         comm, MPI_STATUS_IGNORE);
            
            /* Local reduction: recvbuf = recvbuf op tmpbuf */
            gpu_reduce_local(tmpbuf, recvbuf, count, datatype, op, use_gpu, bufsize);
            
            mask <<= 1;  /* Double the distance for next step */
        }
    }
    
    /*
     * ========== Phase 3: Send results back to waiting ranks ==========
     * 
     * Odd ranks in the first 2*rem that didn't participate in the main
     * algorithm now receive the final reduced result from their even partners.
     */
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /* Even rank: send final result to odd partner */
            MPI_Send(recvbuf, count, datatype, rank + 1, 0, comm);
        } else {
            /* Odd rank: receive final result from even partner */
            MPI_Recv(recvbuf, count, datatype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
        }
    }
    
    /* Clean up: free temporary buffer */
    if (use_gpu) {
        cudaFree(tmpbuf);
    } else {
        free(tmpbuf);
    }
    
    return MPI_SUCCESS;
}
