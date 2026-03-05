/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#ifndef CAIL_ALLREDUCE_IMPL_H
#define CAIL_ALLREDUCE_IMPL_H

#include <mpi.h>

/* Forward declarations for all allreduce algorithm implementations.
 *
 * Each algorithm can be individually enabled or disabled at configure time
 * via --enable-<algorithm> (all enabled by default). The CAIL_ENABLE_*
 * defines gate compilation; disabled algorithms are not linked.
 */

int cail_allreduce_recursive_doubling(const void *sendbuf, void *recvbuf, int count,
                                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int cail_allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int cail_allreduce_rabenseifner(const void *sendbuf, void *recvbuf, int count,
                                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);


#endif /* CAIL_ALLREDUCE_IMPL_H */
