/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#ifndef CAIL_ALLREDUCE_IMPL_H
#define CAIL_ALLREDUCE_IMPL_H

#include <mpi.h>

/* Forward declarations for all allreduce algorithm implementations.
 *
 * All algorithms are always compiled and available. The dispatch logic in
 * cail_allreduce.c uses CAIL_ENABLE_* to select which algorithms are
 * reachable at runtime; disabled algorithms simply never get called.
 */

int cail_allreduce_recursive_doubling(const void *sendbuf, void *recvbuf, int count,
                                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int cail_allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int cail_allreduce_rabenseifner(const void *sendbuf, void *recvbuf, int count,
                                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int cail_allreduce_tree(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif /* CAIL_ALLREDUCE_IMPL_H */
