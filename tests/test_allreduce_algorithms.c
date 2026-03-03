/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    const char   *name;
    MPI_Datatype  mpi_type;
    size_t        size;
    int           is_float;
} type_info_t;

typedef struct {
    const char *name;
    MPI_Op      mpi_op;
} op_info_t;

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

static int check_int_buf(const void *buf, int count, size_t elem_size, long long expected)
{
    const unsigned char *p = (const unsigned char *)buf;
    unsigned long long mask = (elem_size >= sizeof(long long))
        ? ~0ULL : (1ULL << (elem_size * 8)) - 1;
    unsigned long long exp_masked = (unsigned long long)expected & mask;
    for (int i = 0; i < count; i++) {
        unsigned long long val = 0;
        memcpy(&val, p + i * elem_size, elem_size);
        if ((val & mask) != exp_masked) return 0;
    }
    return 1;
}

static int check_float_buf(const float *buf, int count, float expected)
{
    for (int i = 0; i < count; i++) {
        if (fabsf(buf[i] - expected) > fabsf(expected) * 1e-3f + 1e-6f) return 0;
    }
    return 1;
}

static int check_double_buf(const double *buf, int count, double expected)
{
    for (int i = 0; i < count; i++) {
        if (fabs(buf[i] - expected) > fabs(expected) * 1e-9 + 1e-12) return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc > 1) {
        setenv("CAIL_ALGO", argv[1], 1);
    }

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char *algo = getenv("CAIL_ALGO");
    if (!algo) algo = "auto";

    type_info_t types[] = {
        {"MPI_INT",    MPI_INT,    sizeof(int),    0},
        {"MPI_FLOAT",  MPI_FLOAT,  sizeof(float),  1},
        {"MPI_DOUBLE", MPI_DOUBLE, sizeof(double), 1},
        {"MPI_UINT64_T", MPI_UINT64_T, sizeof(uint64_t), 0},
        {"MPI_SHORT",  MPI_SHORT,  sizeof(short),  0},
        {"MPI_UNSIGNED_LONG_LONG", MPI_UNSIGNED_LONG_LONG, sizeof(unsigned long long), 0},
    };
    int ntypes = (int)(sizeof(types) / sizeof(types[0]));

    op_info_t ops[] = {
        {"MPI_SUM",  MPI_SUM},
        {"MPI_PROD", MPI_PROD},
        {"MPI_MAX",  MPI_MAX},
        {"MPI_MIN",  MPI_MIN},
    };
    int nops = (int)(sizeof(ops) / sizeof(ops[0]));

    int counts[] = {10, 1000, 4096, 100000, 200000};
    int ncounts = (int)(sizeof(counts) / sizeof(counts[0]));

    int all_pass = 1;
    for (int ci = 0; ci < ncounts; ci++) {
        int count = counts[ci];
        for (int ti = 0; ti < ntypes; ti++) {
            for (int oi = 0; oi < nops; oi++) {
                if (ops[oi].mpi_op == MPI_PROD && nprocs > 4) continue;

                size_t bufsize = count * types[ti].size;
                void *sendbuf = malloc(bufsize);
                void *recvbuf = malloc(bufsize);
                if (!sendbuf || !recvbuf) { MPI_Abort(MPI_COMM_WORLD, 1); }

                if (types[ti].is_float && types[ti].size == sizeof(float)) {
                    float *s = (float *)sendbuf;
                    for (int i = 0; i < count; i++) s[i] = (float)(rank + 1);
                } else if (types[ti].is_float && types[ti].size == sizeof(double)) {
                    double *s = (double *)sendbuf;
                    for (int i = 0; i < count; i++) s[i] = (double)(rank + 1);
                } else {
                    unsigned char *s = (unsigned char *)sendbuf;
                    for (int i = 0; i < count; i++) {
                        long long val = (long long)(rank + 1);
                        memcpy(s + i * types[ti].size, &val, types[ti].size);
                    }
                }
                memset(recvbuf, 0, bufsize);

                MPI_Allreduce(sendbuf, recvbuf, count, types[ti].mpi_type,
                              ops[oi].mpi_op, MPI_COMM_WORLD);

                double expected = compute_expected(ops[oi].mpi_op, nprocs);
                int pass;

                if (types[ti].is_float && types[ti].size == sizeof(float)) {
                    pass = check_float_buf((const float *)recvbuf, count, (float)expected);
                } else if (types[ti].is_float && types[ti].size == sizeof(double)) {
                    pass = check_double_buf((const double *)recvbuf, count, expected);
                } else {
                    pass = check_int_buf(recvbuf, count, types[ti].size, (long long)expected);
                }

                if (rank == 0) {
                    printf("%s: algo=%s %s × %s count=%d np=%d\n",
                           pass ? "PASS" : "FAIL", algo,
                           types[ti].name, ops[oi].name, count, nprocs);
                }
                if (!pass) all_pass = 0;

                free(sendbuf);
                free(recvbuf);
            }
        }
    }

    MPI_Finalize();
    return all_pass ? 0 : 1;
}
