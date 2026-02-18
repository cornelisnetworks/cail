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

static void fill_buf(void *buf, int count, const type_info_t *t, int rank)
{
    if (t->is_float && t->size == sizeof(float)) {
        float *b = (float *)buf;
        for (int i = 0; i < count; i++) b[i] = (float)(rank + 1);
    } else if (t->is_float && t->size == sizeof(double)) {
        double *b = (double *)buf;
        for (int i = 0; i < count; i++) b[i] = (double)(rank + 1);
    } else {
        unsigned char *b = (unsigned char *)buf;
        for (int i = 0; i < count; i++) {
            long long val = (long long)(rank + 1);
            memcpy(b + i * t->size, &val, t->size);
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    type_info_t types[] = {
        {"MPI_CHAR",               MPI_CHAR,               sizeof(char),               0},
        {"MPI_SHORT",              MPI_SHORT,              sizeof(short),              0},
        {"MPI_INT",                MPI_INT,                sizeof(int),                0},
        {"MPI_LONG",               MPI_LONG,               sizeof(long),               0},
        {"MPI_LONG_LONG",          MPI_LONG_LONG,          sizeof(long long),          0},
        {"MPI_FLOAT",              MPI_FLOAT,              sizeof(float),              1},
        {"MPI_DOUBLE",             MPI_DOUBLE,             sizeof(double),             1},
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

    op_info_t ops[] = {
        {"MPI_SUM",  MPI_SUM},
        {"MPI_PROD", MPI_PROD},
        {"MPI_MAX",  MPI_MAX},
        {"MPI_MIN",  MPI_MIN},
    };
    int nops = (int)(sizeof(ops) / sizeof(ops[0]));

    int test_counts[] = {10, 4096, 200000};
    int ntest_counts = (int)(sizeof(test_counts) / sizeof(test_counts[0]));
    int all_pass = 1;

    for (int ci = 0; ci < ntest_counts; ci++) {
        int count = test_counts[ci];
        for (int ti = 0; ti < ntypes; ti++) {
            for (int oi = 0; oi < nops; oi++) {
                if (ops[oi].mpi_op == MPI_PROD && nprocs > 4) continue;

                size_t bufsize = count * types[ti].size;
                void *buf = malloc(bufsize);
                if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }

                fill_buf(buf, count, &types[ti], rank);

                MPI_Allreduce(MPI_IN_PLACE, buf, count, types[ti].mpi_type,
                              ops[oi].mpi_op, MPI_COMM_WORLD);

                double expected = compute_expected(ops[oi].mpi_op, nprocs);
                int pass;

                if (types[ti].is_float && types[ti].size == sizeof(float)) {
                    pass = check_float_buf((const float *)buf, count, (float)expected);
                } else if (types[ti].is_float && types[ti].size == sizeof(double)) {
                    pass = check_double_buf((const double *)buf, count, expected);
                } else {
                    pass = check_int_buf(buf, count, types[ti].size, (long long)expected);
                }

                if (rank == 0) {
                    printf("%s: MPI_IN_PLACE %s × %s count=%d np=%d\n",
                           pass ? "PASS" : "FAIL", types[ti].name, ops[oi].name, count, nprocs);
                }
                if (!pass) all_pass = 0;

                free(buf);
            }
        }
    }

    MPI_Finalize();
    return all_pass ? 0 : 1;
}
