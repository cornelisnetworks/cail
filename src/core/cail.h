/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail.h — GPU-aware MPI_Allreduce interposition library public header */
#ifndef CAIL_H
#define CAIL_H

#include <mpi.h>

/* Library version — available for compile-time checks by consumers of cail.h */
#define CAIL_VERSION_MAJOR 0
#define CAIL_VERSION_MINOR 0
#define CAIL_VERSION_PATCH 0

/*
 * cail is a transparent MPI_Allreduce interposition library.
 * Link with -lcail before -lmpi to activate GPU-aware allreduce.
 * No API calls are needed — the library intercepts MPI_Allreduce via PMPI.
 */

#endif /* CAIL_H */

