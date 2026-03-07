/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/*
 * test_allreduce_correctness_gpu.cu — GPU-buffer correctness tests for cail
 *
 * Mirrors test_allreduce_correctness.c but with CUDA device buffers.
 * For a given element count (-c, required) and algorithm (-a, optional),
 * exercises every supported {datatype x op} combination with both
 * separate send/recv buffers and MPI_IN_PLACE.
 *
 * 20 datatypes x 4 ops x 2 modes = 160 sub-tests per invocation
 * (minus PROD skips for nprocs > 4 to avoid overflow).
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                                \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                  \
} while (0)

/* ------------------------------------------------------------------ */
/* Type / op descriptors (same 20 types as CPU correctness test)      */
/* ------------------------------------------------------------------ */

typedef struct {
    const char   *name;
    MPI_Datatype  mpi_type;
    size_t        size;
    int           is_float;   /* 1 = float, 2 = double, 0 = integer */
} type_info_t;

typedef struct {
    const char *name;
    MPI_Op      mpi_op;
} op_info_t;

/* ------------------------------------------------------------------ */
/* Expected-value computation                                         */
/* ------------------------------------------------------------------ */

static long long factorial(int n)
{
    long long r = 1;
    for (int i = 2; i <= n; i++) r *= i;
    return r;
}

static double compute_expected(MPI_Op op, int nprocs)
{
    if (op == MPI_SUM) {
        double s = 0.0;
        for (int r = 0; r < nprocs; r++) s += (double)(r + 1);
        return s;
    }
    if (op == MPI_PROD) return (double)factorial(nprocs);
    if (op == MPI_MAX)  return (double)nprocs;
    if (op == MPI_MIN)  return 1.0;
    return 0.0;
}

/* ------------------------------------------------------------------ */
/* Generic kernel: fill each element with a value                     */
/*                                                                    */
/* We write the value byte-by-byte into each element slot. This       */
/* handles all 12 base types without needing per-type kernels.        */
/* ------------------------------------------------------------------ */

__global__ void fill_elements(char *buf, int count, int elem_size,
                              const char *val_bytes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        char *dst = buf + (size_t)idx * elem_size;
        for (int b = 0; b < elem_size; b++)
            dst[b] = val_bytes[b];
    }
}

static void gpu_fill_buf(void *d_buf, int count, size_t elem_size,
                         int is_float, int rank)
{
    /* Build the value on host, then copy the raw bytes to constant memory
     * via a small device buffer. */
    char val_bytes[16] = {0};  /* large enough for any MPI type */

    if (is_float == 1) {
        float v = (float)(rank + 1);
        memcpy(val_bytes, &v, sizeof(v));
    } else if (is_float == 2) {
        double v = (double)(rank + 1);
        memcpy(val_bytes, &v, sizeof(v));
    } else {
        long long v = (long long)(rank + 1);
        memcpy(val_bytes, &v, elem_size);
    }

    /* Copy value template to device */
    char *d_val;
    CUDA_CHECK(cudaMalloc(&d_val, 16));
    CUDA_CHECK(cudaMemcpy(d_val, val_bytes, 16, cudaMemcpyHostToDevice));

    int grid = (count + 255) / 256;
    fill_elements<<<grid, 256>>>((char *)d_buf, count, (int)elem_size, d_val);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_val);
}

/* ------------------------------------------------------------------ */
/* Host-side verification (copy D2H then check)                       */
/* ------------------------------------------------------------------ */

static int check_buf(const void *h_buf, int count, const type_info_t *t,
                     double expected)
{
    if (t->is_float == 1) {
        const float *b = (const float *)h_buf;
        float exp_f = (float)expected;
        for (int i = 0; i < count; i++) {
            if (fabsf(b[i] - exp_f) > fabsf(exp_f) * 1e-3f + 1e-6f)
                return 0;
        }
        return 1;
    }
    if (t->is_float == 2) {
        const double *b = (const double *)h_buf;
        for (int i = 0; i < count; i++) {
            if (fabs(b[i] - expected) > fabs(expected) * 1e-9 + 1e-12)
                return 0;
        }
        return 1;
    }
    /* Integer types: mask expected to elem_size bytes. */
    const unsigned char *p = (const unsigned char *)h_buf;
    unsigned long long mask = (t->size >= sizeof(long long))
        ? ~0ULL : (1ULL << (t->size * 8)) - 1;
    unsigned long long exp_masked = (unsigned long long)(long long)expected & mask;
    for (int i = 0; i < count; i++) {
        unsigned long long val = 0;
        memcpy(&val, p + i * t->size, t->size);
        if ((val & mask) != exp_masked) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Usage                                                              */
/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s -c count [-a algo] [-h]\n", prog);
    fprintf(stderr, "  -c count   element count (required)\n");
    fprintf(stderr, "  -a algo    set CAIL_ALGO before MPI_Init (default: auto)\n");
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int count = 0;
    const char *algo_name = NULL;
    int opt;

    while ((opt = getopt(argc, argv, "c:a:h")) != -1) {
        switch (opt) {
        case 'c':
            count = atoi(optarg);
            break;
        case 'a':
            algo_name = optarg;
            setenv("CAIL_ALGO", optarg, 1);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    if (count <= 0) {
        fprintf(stderr, "%s: -c count is required and must be > 0\n", argv[0]);
        usage(argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % dev_count));

    if (!algo_name) algo_name = "auto";
    int is_pof2 = (nprocs > 0) && ((nprocs & (nprocs - 1)) == 0);

    if (rank == 0) {
        printf("=== cail GPU allreduce correctness tests ===\n");
        printf("nprocs=%d  GPUs_per_node=%d  count=%d  algo=%s\n",
               nprocs, dev_count, count, algo_name);
        fflush(stdout);
    }

    /* All 20 CAIL-supported datatypes. */
    type_info_t types[] = {
        {"MPI_CHAR",               MPI_CHAR,               sizeof(char),               0},
        {"MPI_SHORT",              MPI_SHORT,              sizeof(short),              0},
        {"MPI_INT",                MPI_INT,                sizeof(int),                0},
        {"MPI_LONG",               MPI_LONG,               sizeof(long),               0},
        {"MPI_LONG_LONG",          MPI_LONG_LONG,          sizeof(long long),          0},
        {"MPI_FLOAT",              MPI_FLOAT,              sizeof(float),              1},
        {"MPI_DOUBLE",             MPI_DOUBLE,             sizeof(double),             2},
        {"MPI_UNSIGNED_CHAR",      MPI_UNSIGNED_CHAR,      sizeof(unsigned char),      0},
        {"MPI_UNSIGNED_SHORT",     MPI_UNSIGNED_SHORT,     sizeof(unsigned short),     0},
        {"MPI_UNSIGNED",           MPI_UNSIGNED,           sizeof(unsigned int),       0},
        {"MPI_UNSIGNED_LONG",      MPI_UNSIGNED_LONG,      sizeof(unsigned long),      0},
        {"MPI_UNSIGNED_LONG_LONG", MPI_UNSIGNED_LONG_LONG, sizeof(unsigned long long), 0},
        {"MPI_INT8_T",             MPI_INT8_T,             sizeof(int8_t),             0},
        {"MPI_UINT8_T",            MPI_UINT8_T,            sizeof(uint8_t),            0},
        {"MPI_INT16_T",            MPI_INT16_T,            sizeof(int16_t),            0},
        {"MPI_UINT16_T",           MPI_UINT16_T,           sizeof(uint16_t),           0},
        {"MPI_INT32_T",            MPI_INT32_T,            sizeof(int32_t),            0},
        {"MPI_UINT32_T",           MPI_UINT32_T,           sizeof(uint32_t),           0},
        {"MPI_INT64_T",            MPI_INT64_T,            sizeof(int64_t),            0},
        {"MPI_UINT64_T",           MPI_UINT64_T,           sizeof(uint64_t),           0},
    };
    int ntypes = (int)(sizeof(types) / sizeof(types[0]));

    /* All 4 CAIL-supported ops. */
    op_info_t ops[] = {
        {"MPI_SUM",  MPI_SUM},
        {"MPI_PROD", MPI_PROD},
        {"MPI_MAX",  MPI_MAX},
        {"MPI_MIN",  MPI_MIN},
    };
    int nops = (int)(sizeof(ops) / sizeof(ops[0]));

    int g_pass = 0;
    int g_fail = 0;

    /* Pre-allocate GPU and host buffers at the maximum needed size
     * (count * largest_type_size).  Reusing a single allocation avoids
     * hammering the Open MPI rcache VMA interval tree with hundreds of
     * cudaMalloc/cudaFree cycles, which triggers a SEGV in
     * opal_interval_tree_traverse at certain power-of-2 buffer sizes. */
    size_t max_elem_size = 0;
    for (int ti = 0; ti < ntypes; ti++)
        if (types[ti].size > max_elem_size) max_elem_size = types[ti].size;
    size_t max_bufsize = (size_t)count * max_elem_size;

    void *d_send, *d_recv, *d_inplace;
    CUDA_CHECK(cudaMalloc(&d_send, max_bufsize));
    CUDA_CHECK(cudaMalloc(&d_recv, max_bufsize));
    CUDA_CHECK(cudaMalloc(&d_inplace, max_bufsize));
    void *h_check = malloc(max_bufsize);
    if (!h_check) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int ti = 0; ti < ntypes; ti++) {
        for (int oi = 0; oi < nops; oi++) {
            /* PROD overflows quickly — skip for large process counts. */
            if (ops[oi].mpi_op == MPI_PROD && nprocs > 4) continue;

            double expected = compute_expected(ops[oi].mpi_op, nprocs);
            size_t bufsize = (size_t)count * types[ti].size;

            /* --- Separate send/recv buffers --- */
            CUDA_CHECK(cudaMemset(d_recv, 0, bufsize));
            gpu_fill_buf(d_send, count, types[ti].size,
                         types[ti].is_float, rank);

            MPI_Allreduce(d_send, d_recv, count, types[ti].mpi_type,
                          ops[oi].mpi_op, MPI_COMM_WORLD);

            CUDA_CHECK(cudaMemcpy(h_check, d_recv, bufsize,
                                  cudaMemcpyDeviceToHost));

            {
                int pass = check_buf(h_check, count, &types[ti], expected);
                if (rank == 0) {
                    printf("%s: algo=%s %s x %s count=%d np=%d pof2=%s\n",
                           pass ? "PASS" : "FAIL", algo_name,
                           types[ti].name, ops[oi].name,
                           count, nprocs, is_pof2 ? "yes" : "no");
                    fflush(stdout);
                }
                if (pass) g_pass++; else g_fail++;
            }

            /* --- MPI_IN_PLACE --- */
            gpu_fill_buf(d_inplace, count, types[ti].size,
                         types[ti].is_float, rank);

            MPI_Allreduce(MPI_IN_PLACE, d_inplace, count, types[ti].mpi_type,
                          ops[oi].mpi_op, MPI_COMM_WORLD);

            CUDA_CHECK(cudaMemcpy(h_check, d_inplace, bufsize,
                                  cudaMemcpyDeviceToHost));

            {
                int pass = check_buf(h_check, count, &types[ti], expected);
                if (rank == 0) {
                    printf("%s: algo=%s MPI_IN_PLACE %s x %s count=%d np=%d pof2=%s\n",
                           pass ? "PASS" : "FAIL", algo_name,
                           types[ti].name, ops[oi].name,
                           count, nprocs, is_pof2 ? "yes" : "no");
                    fflush(stdout);
                }
                if (pass) g_pass++; else g_fail++;
            }
        }
    }

    free(h_check);
    cudaFree(d_send);
    cudaFree(d_recv);
    cudaFree(d_inplace);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
        fflush(stdout);
    }

    MPI_Finalize();
    return g_fail > 0 ? 1 : 0;
}
