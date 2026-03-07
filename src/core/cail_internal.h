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

/* Warning: gated by CAIL_WARN env var (default: enabled) */
#define CAIL_WARN(fmt, ...) \
    do { if (cail_global_state.warn) \
        fprintf(stderr, "[cail WARN] " fmt "\n", ##__VA_ARGS__); } while (0)

#define CAIL_CHECK(rc) \
    do { if ((rc) != MPI_SUCCESS) return (rc); } while (0)

#define CAIL_ENV_DEBUG            "CAIL_DEBUG"
#define CAIL_ENV_ALGO             "CAIL_ALGO"
#define CAIL_ENV_MSG_SMALL_THRESHOLD  "CAIL_MSG_SMALL_THRESHOLD"
#define CAIL_ENV_NPROCS_THRESHOLD     "CAIL_NPROCS_THRESHOLD"
#define CAIL_ENV_MIN_MSG_SIZE         "CAIL_MIN_MSG_SIZE"
#define CAIL_ENV_WARN                 "CAIL_WARN"

#define CAIL_DEFAULT_MSG_SMALL_THRESHOLD   262144U     /* 256 KB */
#define CAIL_DEFAULT_NPROCS_THRESHOLD      4U
#define CAIL_DEFAULT_MIN_MSG_SIZE          32768U      /* 32 KB */

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
    int    initialized;       /* 1 after cail_init() succeeds */
    int    debug;             /* emit debug traces to stderr (CAIL_DEBUG) */
    int    warn;              /* emit warnings to stderr (CAIL_WARN, default: on) */
    size_t msg_small_threshold;  /* msg bytes below this → recursive_doubling (CAIL_MSG_SMALL_THRESHOLD) */
    int    nprocs_threshold;  /* nprocs ≤ this → small-scale dispatch rules (CAIL_NPROCS_THRESHOLD) */
    size_t min_msg_size;      /* msg bytes below this → passthrough to PMPI (CAIL_MIN_MSG_SIZE) */
    int    force_algo;        /* CAIL_ALGO_AUTO or forced algorithm (CAIL_ALGO) */
} cail_state_t;

extern cail_state_t cail_global_state;


int  cail_init(MPI_Comm comm);
void cail_finalize(void);
int  cail_is_initialized(void);

int cail_allreduce_dispatch(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int  cail_buf_get_tmp(void **ptr, size_t size);
void cail_buf_release_tmp(void *ptr);
void cail_buf_finalize(void);

#endif /* CAIL_INTERNAL_H */
