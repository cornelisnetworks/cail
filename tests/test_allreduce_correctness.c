/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/*
 * test_allreduce_correctness.c — MPI_Allreduce correctness test
 *
 * For a given element count (-c) and algorithm (-a), exercises every
 * supported {datatype × op} combination with both separate send/recv
 * buffers and MPI_IN_PLACE.  The test runner sweeps counts, algorithms,
 * and process counts externally.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Type / op descriptors                                              */
/* ------------------------------------------------------------------ */

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
/* Buffer fill / check helpers                                        */
/* ------------------------------------------------------------------ */

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

static int check_buf(const void *buf, int count, const type_info_t *t,
                     double expected)
{
    if (t->is_float && t->size == sizeof(float)) {
        const float *b = (const float *)buf;
        float exp_f = (float)expected;
        for (int i = 0; i < count; i++) {
            if (fabsf(b[i] - exp_f) > fabsf(exp_f) * 1e-3f + 1e-6f)
                return 0;
        }
        return 1;
    }
    if (t->is_float && t->size == sizeof(double)) {
        const double *b = (const double *)buf;
        for (int i = 0; i < count; i++) {
            if (fabs(b[i] - expected) > fabs(expected) * 1e-9 + 1e-12)
                return 0;
        }
        return 1;
    }
    /* Integer types: mask expected to elem_size bytes. */
    const unsigned char *p = (const unsigned char *)buf;
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

    if (!algo_name) algo_name = "auto";
    int is_pof2 = (nprocs > 0) && ((nprocs & (nprocs - 1)) == 0);

    /* All 20 CAIL-supported datatypes. */
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

    /* All 4 CAIL-supported ops. */
    op_info_t ops[] = {
        {"MPI_SUM",  MPI_SUM},
        {"MPI_PROD", MPI_PROD},
        {"MPI_MAX",  MPI_MAX},
        {"MPI_MIN",  MPI_MIN},
    };
    int nops = (int)(sizeof(ops) / sizeof(ops[0]));

    int all_pass = 1;

    for (int ti = 0; ti < ntypes; ti++) {
        for (int oi = 0; oi < nops; oi++) {
            /* PROD overflows quickly — skip for large process counts. */
            if (ops[oi].mpi_op == MPI_PROD && nprocs > 4) continue;

            double expected = compute_expected(ops[oi].mpi_op, nprocs);
            size_t bufsize = (size_t)count * types[ti].size;

            /* --- Separate send/recv buffers --- */
            {
                void *sendbuf = malloc(bufsize);
                void *recvbuf = malloc(bufsize);
                if (!sendbuf || !recvbuf) { MPI_Abort(MPI_COMM_WORLD, 1); }

                fill_buf(sendbuf, count, &types[ti], rank);
                memset(recvbuf, 0, bufsize);

                MPI_Allreduce(sendbuf, recvbuf, count, types[ti].mpi_type,
                              ops[oi].mpi_op, MPI_COMM_WORLD);

                int pass = check_buf(recvbuf, count, &types[ti], expected);
                if (rank == 0) {
                    printf("%s: algo=%s %s × %s count=%d np=%d pof2=%s\n",
                           pass ? "PASS" : "FAIL", algo_name,
                           types[ti].name, ops[oi].name,
                           count, nprocs, is_pof2 ? "yes" : "no");
                }
                if (!pass) all_pass = 0;

                free(sendbuf);
                free(recvbuf);
            }

            /* --- MPI_IN_PLACE --- */
            {
                void *buf = malloc(bufsize);
                if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }

                fill_buf(buf, count, &types[ti], rank);

                MPI_Allreduce(MPI_IN_PLACE, buf, count, types[ti].mpi_type,
                              ops[oi].mpi_op, MPI_COMM_WORLD);

                int pass = check_buf(buf, count, &types[ti], expected);
                if (rank == 0) {
                    printf("%s: algo=%s MPI_IN_PLACE %s × %s count=%d np=%d pof2=%s\n",
                           pass ? "PASS" : "FAIL", algo_name,
                           types[ti].name, ops[oi].name,
                           count, nprocs, is_pof2 ? "yes" : "no");
                }
                if (!pass) all_pass = 0;

                free(buf);
            }
        }
    }

    MPI_Finalize();
    return all_pass ? 0 : 1;
}
