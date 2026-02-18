/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "cail_internal.h"
#include "../gpu/cail_gpu.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

cail_state_t cail_global_state = {
    .initialized      = 0,
    .debug            = 0,
    .small_threshold  = CAIL_DEFAULT_SMALL_THRESHOLD,
    .medium_threshold = CAIL_DEFAULT_MEDIUM_THRESHOLD,
    .nprocs_small     = CAIL_DEFAULT_NPROCS_SMALL,
    .nprocs_large     = CAIL_DEFAULT_NPROCS_LARGE,
    .min_msg_size     = CAIL_DEFAULT_MIN_MSG_SIZE,
    .force_algo       = CAIL_ALGO_AUTO
};

static int parse_positive_long(const char *env_name, const char *env_val,
                               size_t *out)
{
    char *end;
    long val = strtol(env_val, &end, 10);
    if (*end != '\0' || val <= 0) {
        CAIL_ERR("invalid %s='%s'", env_name, env_val);
        return -1;
    }
    *out = (size_t)val;
    return 0;
}

static int parse_nonneg_long(const char *env_name, const char *env_val,
                             size_t *out)
{
    char *end;
    long val = strtol(env_val, &end, 10);
    if (*end != '\0' || val < 0) {
        CAIL_ERR("invalid %s='%s'", env_name, env_val);
        return -1;
    }
    *out = (size_t)val;
    return 0;
}

static int parse_positive_int(const char *env_name, const char *env_val,
                              int *out)
{
    char *end;
    long val = strtol(env_val, &end, 10);
    if (*end != '\0' || val <= 0 || val > INT_MAX) {
        CAIL_ERR("invalid %s='%s'", env_name, env_val);
        return -1;
    }
    *out = (int)val;
    return 0;
}

int cail_init(MPI_Comm comm)
{
    (void)comm;

    if (cail_global_state.initialized)
        return MPI_SUCCESS;

    const char *env = getenv(CAIL_ENV_DEBUG);
    if (env && env[0] != '\0' && env[0] != '0')
        cail_global_state.debug = 1;

    env = getenv(CAIL_ENV_SMALL_THRESHOLD);
    if (env && env[0] != '\0') {
        if (parse_positive_long(CAIL_ENV_SMALL_THRESHOLD, env,
                                &cail_global_state.small_threshold) != 0)
            cail_global_state.small_threshold = CAIL_DEFAULT_SMALL_THRESHOLD;
    }

    env = getenv(CAIL_ENV_MEDIUM_THRESHOLD);
    if (env && env[0] != '\0') {
        if (parse_positive_long(CAIL_ENV_MEDIUM_THRESHOLD, env,
                                &cail_global_state.medium_threshold) != 0)
            cail_global_state.medium_threshold = CAIL_DEFAULT_MEDIUM_THRESHOLD;
    }

    if (cail_global_state.small_threshold >= cail_global_state.medium_threshold) {
        CAIL_ERR("CAIL_SMALL_THRESHOLD (%zu) >= CAIL_MEDIUM_THRESHOLD (%zu), using defaults",
                  cail_global_state.small_threshold,
                  cail_global_state.medium_threshold);
        cail_global_state.small_threshold  = CAIL_DEFAULT_SMALL_THRESHOLD;
        cail_global_state.medium_threshold = CAIL_DEFAULT_MEDIUM_THRESHOLD;
    }

    env = getenv(CAIL_ENV_NPROCS_SMALL);
    if (env && env[0] != '\0') {
        if (parse_positive_int(CAIL_ENV_NPROCS_SMALL, env,
                               &cail_global_state.nprocs_small) != 0)
            cail_global_state.nprocs_small = CAIL_DEFAULT_NPROCS_SMALL;
    }

    env = getenv(CAIL_ENV_NPROCS_LARGE);
    if (env && env[0] != '\0') {
        if (parse_positive_int(CAIL_ENV_NPROCS_LARGE, env,
                               &cail_global_state.nprocs_large) != 0)
            cail_global_state.nprocs_large = CAIL_DEFAULT_NPROCS_LARGE;
    }

    if (cail_global_state.nprocs_small >= cail_global_state.nprocs_large) {
        CAIL_ERR("CAIL_NPROCS_SMALL (%d) >= CAIL_NPROCS_LARGE (%d), using defaults",
                  cail_global_state.nprocs_small,
                  cail_global_state.nprocs_large);
        cail_global_state.nprocs_small = CAIL_DEFAULT_NPROCS_SMALL;
        cail_global_state.nprocs_large = CAIL_DEFAULT_NPROCS_LARGE;
    }


    env = getenv(CAIL_ENV_MIN_MSG_SIZE);
    if (env && env[0] != '\0') {
        if (parse_nonneg_long(CAIL_ENV_MIN_MSG_SIZE, env,
                              &cail_global_state.min_msg_size) != 0)
            cail_global_state.min_msg_size = CAIL_DEFAULT_MIN_MSG_SIZE;
    }

    env = getenv(CAIL_ENV_ALGO);
    if (env && env[0] != '\0') {
        if      (strcmp(env, "auto")               == 0) cail_global_state.force_algo = CAIL_ALGO_AUTO;
        else if (strcmp(env, "recursive_doubling") == 0) cail_global_state.force_algo = CAIL_ALGO_RECURSIVE_DOUBLING;
        else if (strcmp(env, "ring")               == 0) cail_global_state.force_algo = CAIL_ALGO_RING;
        else if (strcmp(env, "rabenseifner")       == 0) cail_global_state.force_algo = CAIL_ALGO_RABENSEIFNER;
        else if (strcmp(env, "tree")               == 0) cail_global_state.force_algo = CAIL_ALGO_TREE;
        else {
            CAIL_ERR("unknown CAIL_ALGO='%s', using auto", env);
            cail_global_state.force_algo = CAIL_ALGO_AUTO;
        }
    }

    int rc = cail_gpu_init();
    if (rc != 0) {
        CAIL_ERR("cail_gpu_init failed (rc=%d)", rc);
        return MPI_ERR_INTERN;
    }

    cail_global_state.initialized = 1;

    CAIL_DBG("initialized: min_msg_size=%zu small_threshold=%zu "
              "medium_threshold=%zu nprocs_small=%d nprocs_large=%d "
              "force_algo=%d debug=%d",
              cail_global_state.min_msg_size,
              cail_global_state.small_threshold,
              cail_global_state.medium_threshold,
              cail_global_state.nprocs_small,
              cail_global_state.nprocs_large,
              cail_global_state.force_algo,
              cail_global_state.debug);

    return MPI_SUCCESS;
}

void cail_finalize(void)
{
    if (!cail_global_state.initialized)
        return;

    cail_gpu_finalize();
    cail_buf_finalize();
    cail_global_state.initialized = 0;
}

int cail_is_initialized(void)
{
    return cail_global_state.initialized;
}
