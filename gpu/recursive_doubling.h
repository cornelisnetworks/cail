/*
 * Recursive Doubling Allreduce Algorithm
 * 
 * Efficient for small-to-medium messages. O(log P) steps.
 * Each step: exchange with partner at distance 2^k, then local reduce.
 */

#ifndef RECURSIVE_DOUBLING_H
#define RECURSIVE_DOUBLING_H

#include <mpi.h>

/*
 * Recursive Doubling Allreduce Implementation
 * 
 * For non-power-of-2 processes, extra processes send data to lower ranks
 * before the main algorithm and receive results after.
 */
int recursive_doubling_allreduce(const void *sendbuf, void *recvbuf,
                                 int count, MPI_Datatype datatype,
                                 MPI_Op op, MPI_Comm comm);

#endif /* RECURSIVE_DOUBLING_H */
