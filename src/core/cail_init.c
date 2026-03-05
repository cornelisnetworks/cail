/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "cail_internal.h"
#include "../gpu/cail_gpu.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

cail_state_t cail_global_state = {
    .initialized          = 0,
    .debug                = 0,
    .warn                 = 1,
    .msg_small_threshold  = CAIL_DEFAULT_MSG_SMALL_THRESHOLD,
    .nprocs_threshold     = CAIL_DEFAULT_NPROCS_THRESHOLD,
    .nprocs_threshold     = CAIL_DEFAULT_NPROCS_THRESHOLD,
    .min_msg_size         = CAIL_DEFAULT_MIN_MSG_SIZE,
    .force_algo           = CAIL_ALGO_AUTO
};

static int cail_env_parse_positive_long(const char *env_name, const char *env_val,
                                        size_t *out)
{
    char *end;
    long val = strtol(env_val, &end, 10);
    if (*end != '\0' || val <= 0) {
        CAIL_ERR("invalid %s='%s', using default", env_name, env_val);
        return -1;
    }
    *out = (size_t)val;
    return 0;
}

static int cail_env_parse_nonneg_long(const char *env_name, const char *env_val,
                                      size_t *out)
{
    char *end;
    long val = strtol(env_val, &end, 10);
    if (*end != '\0' || val < 0) {
        CAIL_ERR("invalid %s='%s', using default", env_name, env_val);
        return -1;
    }
    *out = (size_t)val;
    return 0;
}

static int cail_env_parse_positive_int(const char *env_name, const char *env_val,
                                       int *out)
{
    char *end;
    long val = strtol(env_val, &end, 10);
    if (*end != '\0' || val <= 0 || val > INT_MAX) {
        CAIL_ERR("invalid %s='%s', using default", env_name, env_val);
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

    env = getenv(CAIL_ENV_WARN);
    if (env && env[0] == '0')
        cail_global_state.warn = 0;

    env = getenv(CAIL_ENV_MSG_SMALL_THRESHOLD);
    if (env && env[0] != '\0') {
        if (cail_env_parse_positive_long(CAIL_ENV_MSG_SMALL_THRESHOLD, env,
                                &cail_global_state.msg_small_threshold) != 0)
            cail_global_state.msg_small_threshold = CAIL_DEFAULT_MSG_SMALL_THRESHOLD;
    }


    env = getenv(CAIL_ENV_NPROCS_THRESHOLD);
    if (env && env[0] != '\0') {
        if (cail_env_parse_positive_int(CAIL_ENV_NPROCS_THRESHOLD, env,
                               &cail_global_state.nprocs_threshold) != 0)
            cail_global_state.nprocs_threshold = CAIL_DEFAULT_NPROCS_THRESHOLD;
    }

    env = getenv(CAIL_ENV_MIN_MSG_SIZE);
    if (env && env[0] != '\0') {
        if (cail_env_parse_nonneg_long(CAIL_ENV_MIN_MSG_SIZE, env,
                              &cail_global_state.min_msg_size) != 0)
            cail_global_state.min_msg_size = CAIL_DEFAULT_MIN_MSG_SIZE;
    }

    env = getenv(CAIL_ENV_ALGO);
    if (env && env[0] != '\0') {
        if      (strcmp(env, "auto")               == 0) cail_global_state.force_algo = CAIL_ALGO_AUTO;
        else if (strcmp(env, "recursive_doubling") == 0) cail_global_state.force_algo = CAIL_ALGO_RECURSIVE_DOUBLING;
        else if (strcmp(env, "ring")               == 0) cail_global_state.force_algo = CAIL_ALGO_RING;
        else if (strcmp(env, "rabenseifner")       == 0) cail_global_state.force_algo = CAIL_ALGO_RABENSEIFNER;
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

    CAIL_DBG("initialized: min_msg_size=%zu msg_small_threshold=%zu "
              "nprocs_threshold=%d force_algo=%d debug=%d warn=%d",
              cail_global_state.min_msg_size,
              cail_global_state.msg_small_threshold,
              cail_global_state.nprocs_threshold,
              cail_global_state.force_algo,
              cail_global_state.debug,
              cail_global_state.warn);

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
