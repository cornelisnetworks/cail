/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail.h — GPU-aware MPI_Allreduce interposition library public header */
#ifndef CAIL_H
#define CAIL_H

#include <mpi.h>

/* Library version */
#define CAIL_VERSION_MAJOR 0
#define CAIL_VERSION_MINOR 1
#define CAIL_VERSION_PATCH 0

/* Symbol visibility for shared library */
#if defined(__GNUC__) && __GNUC__ >= 4
#  define CAIL_API __attribute__((visibility("default")))
#else
#  define CAIL_API
#endif

/*
 * cail is a transparent MPI_Allreduce interposition library.
 * Link with -lcail before -lmpi to activate GPU-aware allreduce.
 * No API calls are needed — the library intercepts MPI_Allreduce via PMPI.
 */

#endif /* CAIL_H */

