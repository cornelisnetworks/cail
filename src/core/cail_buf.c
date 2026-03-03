/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/* cail_buf.c — Single-buffer cache for temporary GPU buffers
 *
 * Centralizes the static-buffer caching pattern previously inlined in
 * recursive_doubling.c (lines 98-115).  The cache grows on demand but
 * never shrinks until cail_buf_finalize() is called.
 *
 * Host-path: when CAIL_HOST_PATH is defined, uses malloc/free instead
 * of the GPU allocator so the library can be tested without a GPU.
 */

#include "cail_internal.h"

#ifndef CAIL_HOST_PATH
#  include "../gpu/cail_gpu.h"
#endif

#include <stdlib.h>

static void  *cached_buf  = NULL;
static size_t cached_size = 0;

int cail_buf_get_tmp(void **ptr, size_t size)
{
    if (size == 0) {
        *ptr = NULL;
        return MPI_SUCCESS;
    }

    if (size > cached_size) {
        if (cached_buf) {
#ifdef CAIL_HOST_PATH
            free(cached_buf);
#else
            cail_gpu_free(cached_buf);
#endif
            cached_buf  = NULL;
            cached_size = 0;
        }

#ifdef CAIL_HOST_PATH
        cached_buf = malloc(size);
        if (!cached_buf) {
            CAIL_ERR("malloc(%zu) failed", size);
            return MPI_ERR_NO_MEM;
        }
#else
        int rc = cail_gpu_malloc(&cached_buf, size);
        if (rc != 0) {
            CAIL_ERR("cail_gpu_malloc(%zu) failed (rc=%d)", size, rc);
            cached_buf = NULL;
            return MPI_ERR_NO_MEM;
        }
#endif
        cached_size = size;
        CAIL_DBG("buf: allocated %zu bytes", size);
    } else {
        CAIL_DBG("buf: reusing cached %zu bytes (requested %zu)",
                   cached_size, size);
    }

    *ptr = cached_buf;
    return MPI_SUCCESS;
}

void cail_buf_release_tmp(void *ptr)
{
    (void)ptr; /* intentional no-op: buffer stays in cache */
}

void cail_buf_finalize(void)
{
    if (cached_buf) {
#ifdef CAIL_HOST_PATH
        free(cached_buf);
#else
        cail_gpu_free(cached_buf);
#endif
        cached_buf  = NULL;
        cached_size = 0;
    }
}
