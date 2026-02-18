/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#ifndef CAIL_INTERNAL_H
#define CAIL_INTERNAL_H

#include "cail.h"
#include "cail_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

/* Active when CAIL_DEBUG env var is set at runtime */
#define CAIL_DBG(fmt, ...) \
    do { if (cail_global_state.debug) \
        fprintf(stderr, "[cail] " fmt "\n", ##__VA_ARGS__); } while (0)

#define CAIL_ERR(fmt, ...) \
    fprintf(stderr, "[cail ERROR] " fmt "\n", ##__VA_ARGS__)

/* Warning: always printed (not gated by debug), for unexpected fallbacks */
#define CAIL_WARN(fmt, ...) \
    fprintf(stderr, "[cail WARN] " fmt "\n", ##__VA_ARGS__)

#define CAIL_CHECK(rc) \
    do { if ((rc) != MPI_SUCCESS) return (rc); } while (0)

#define CAIL_ENV_DEBUG            "CAIL_DEBUG"
#define CAIL_ENV_ALGO             "CAIL_ALGO"
#define CAIL_ENV_SMALL_THRESHOLD  "CAIL_SMALL_THRESHOLD"
#define CAIL_ENV_MEDIUM_THRESHOLD "CAIL_MEDIUM_THRESHOLD"
#define CAIL_ENV_NPROCS_SMALL     "CAIL_NPROCS_SMALL"
#define CAIL_ENV_NPROCS_LARGE     "CAIL_NPROCS_LARGE"
#define CAIL_ENV_MIN_MSG_SIZE     "CAIL_MIN_MSG_SIZE"

#define CAIL_DEFAULT_SMALL_THRESHOLD   8192U       /* 8 KB  */
#define CAIL_DEFAULT_MEDIUM_THRESHOLD  524288U     /* 512 KB */
#define CAIL_DEFAULT_NPROCS_SMALL      4U
#define CAIL_DEFAULT_NPROCS_LARGE      64U
#define CAIL_DEFAULT_MIN_MSG_SIZE      65536U      /* 64 KB */

/* Compute largest power-of-two ≤ n (n must be > 0) */
static inline int cail_pof2(int n)
{
    int pof2 = 1;
    while (pof2 <= n) pof2 <<= 1;
    return pof2 >> 1;
}

/* Check if n is an exact power of two */
static inline int cail_is_pof2(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

typedef struct cail_state {
    int    initialized;
    int    debug;
    size_t small_threshold;
    size_t medium_threshold;
    int    nprocs_small;      /* nprocs ≤ this → "small scale" rules */
    int    nprocs_large;      /* nprocs > this → "large scale" rules  */
    size_t min_msg_size;      /* msg_size below this → passthrough to PMPI */
    int    force_algo;        /* CAIL_ALGO_AUTO or forced cail_algo_t value */
} cail_state_t;

extern cail_state_t cail_global_state;

typedef int (*cail_allreduce_fn)(const void *, void *, int,
                                   MPI_Datatype, MPI_Op, MPI_Comm);

int  cail_init(MPI_Comm comm);
void cail_finalize(void);
int  cail_is_initialized(void);

int cail_allreduce_dispatch(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int  cail_buf_get_tmp(void **ptr, size_t size);
void cail_buf_release_tmp(void *ptr);
void cail_buf_finalize(void);

#endif /* CAIL_INTERNAL_H */
