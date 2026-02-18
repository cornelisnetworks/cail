/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#ifndef CAIL_ALLREDUCE_IMPL_H
#define CAIL_ALLREDUCE_IMPL_H

#include <mpi.h>

/* Forward declarations for all allreduce algorithm implementations */

#ifdef CAIL_ENABLE_RECURSIVE_DOUBLING
int cail_allreduce_recursive_doubling(const void *sendbuf, void *recvbuf, int count,
                                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#endif

#ifdef CAIL_ENABLE_RING
int cail_allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#endif

#ifdef CAIL_ENABLE_RABENSEIFNER
int cail_allreduce_rabenseifner(const void *sendbuf, void *recvbuf, int count,
                                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#endif

#ifdef CAIL_ENABLE_TREE
int cail_allreduce_tree(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#endif

#endif /* CAIL_ALLREDUCE_IMPL_H */
