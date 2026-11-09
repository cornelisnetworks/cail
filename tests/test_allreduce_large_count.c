/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/*
 * test_allreduce_large_count.c: MPI-4 large-count entry point
 *
 * mpi4py >= 4.0 and other MPI-4 callers use MPI_Allreduce_c rather than
 * MPI_Allreduce. Checks that CAIL intercepts MPI_Allreduce_c (one
 * "[cail] algorithm=" debug line per eligible call, captured from stderr),
 * that results are correct for supported types/ops with separate buffers and
 * MPI_IN_PLACE, and that unsupported types still pass through to the MPI
 * library. Skips (exit 0) when the MPI library predates MPI-4.
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if MPI_VERSION >= 4

static int rank, nprocs;

/* stderr capture: CAIL_DEBUG lines are written to stderr. */
static FILE *cap_file;
static int saved_stderr = -1;

static void capture_begin(void)
{
    fflush(stderr);
    cap_file = tmpfile();
    if (!cap_file) { perror("tmpfile"); MPI_Abort(MPI_COMM_WORLD, 1); }
    saved_stderr = dup(STDERR_FILENO);
    dup2(fileno(cap_file), STDERR_FILENO);
}

/* Restore stderr and return how many "[cail] algorithm=" lines were logged. */
static int capture_end(void)
{
    fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    rewind(cap_file);
    int hits = 0;
    char line[512];
    while (fgets(line, sizeof line, cap_file))
        if (strstr(line, "[cail] algorithm="))
            hits++;
    fclose(cap_file);
    return hits;
}

/* Value contributed by `r` at index i. For PROD only every third element
 * varies so the product (nprocs!) stays exact in every type. */
static double contrib(int r, MPI_Count i, MPI_Op op)
{
    if (op == MPI_PROD)
        return (i % 3 == 0) ? (double)(r + 1) : 1.0;
    return (double)((r + 1) + (int)(i % 5));
}

static double expected(MPI_Count i, MPI_Op op)
{
    double acc = contrib(0, i, op);
    for (int r = 1; r < nprocs; r++) {
        double v = contrib(r, i, op);
        if (op == MPI_SUM) acc += v;
        else if (op == MPI_PROD) acc *= v;
        else if (op == MPI_MAX) acc = v > acc ? v : acc;
        else acc = v < acc ? v : acc;
    }
    return acc;
}

#define DEFINE_RUN(NAME, CTYPE)                                               \
static int NAME(MPI_Datatype dt, MPI_Op op, MPI_Count count, int in_place)    \
{                                                                             \
    CTYPE *sbuf = malloc((size_t)count * sizeof(CTYPE));                      \
    CTYPE *rbuf = malloc((size_t)count * sizeof(CTYPE));                      \
    if (!sbuf || !rbuf) MPI_Abort(MPI_COMM_WORLD, 1);                         \
    for (MPI_Count i = 0; i < count; i++) {                                   \
        sbuf[i] = (CTYPE)contrib(rank, i, op);                                \
        rbuf[i] = in_place ? sbuf[i] : (CTYPE)-1;                             \
    }                                                                         \
    int rc = MPI_Allreduce_c(in_place ? MPI_IN_PLACE : sbuf, rbuf, count,     \
                             dt, op, MPI_COMM_WORLD);                         \
    int ok = (rc == MPI_SUCCESS);                                             \
    for (MPI_Count i = 0; ok && i < count; i++)                               \
        if (fabs((double)rbuf[i] - expected(i, op)) > 1e-6) ok = 0;           \
    free(sbuf);                                                               \
    free(rbuf);                                                               \
    return ok;                                                                \
}

DEFINE_RUN(run_int, int)
DEFINE_RUN(run_float, float)
DEFINE_RUN(run_double, double)
DEFINE_RUN(run_int64, long long)

typedef int (*run_fn)(MPI_Datatype, MPI_Op, MPI_Count, int);

int main(int argc, char **argv)
{
    /* Every eligible call goes through CAIL and logs its algorithm. Set
     * before the first intercepted call, which is when CAIL reads its env. */
    setenv("CAIL_DEBUG", "1", 1);
    setenv("CAIL_MIN_MSG_SIZE", "0", 1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    struct { const char *name; MPI_Datatype dt; run_fn fn; } types[] = {
        { "MPI_INT",     MPI_INT,     run_int },
        { "MPI_FLOAT",   MPI_FLOAT,   run_float },
        { "MPI_DOUBLE",  MPI_DOUBLE,  run_double },
        { "MPI_INT64_T", MPI_INT64_T, run_int64 },
    };
    struct { const char *name; MPI_Op op; } ops[] = {
        { "SUM", MPI_SUM }, { "PROD", MPI_PROD }, { "MAX", MPI_MAX }, { "MIN", MPI_MIN },
    };
    const MPI_Count counts[] = { 1, 1000, 70001 };

    int failures = 0, calls = 0;

    capture_begin();
    for (size_t t = 0; t < sizeof types / sizeof types[0]; t++)
        for (size_t o = 0; o < sizeof ops / sizeof ops[0]; o++)
            for (size_t c = 0; c < sizeof counts / sizeof counts[0]; c++)
                for (int in_place = 0; in_place <= 1; in_place++) {
                    calls++;
                    if (!types[t].fn(types[t].dt, ops[o].op, counts[c], in_place)) {
                        failures++;
                        fprintf(stdout, "FAIL: rank %d %s %s count=%lld %s\n", rank,
                                types[t].name, ops[o].name, (long long)counts[c],
                                in_place ? "in-place" : "separate");
                    }
                }
    int intercepted = capture_end();

    if (intercepted != calls) {
        failures++;
        printf("FAIL: rank %d: CAIL intercepted %d of %d MPI_Allreduce_c calls\n",
               rank, intercepted, calls);
    }

    /* count == 0 is a successful no-op. */
    if (MPI_Allreduce_c(MPI_IN_PLACE, NULL, 0, MPI_INT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        failures++;
        printf("FAIL: rank %d: count=0\n", rank);
    }

    /* Unsupported type: correct result, no CAIL algorithm. */
    {
        MPI_Count n = 16;
        long double buf[16];
        for (int i = 0; i < 16; i++) buf[i] = (long double)(rank + 1);
        capture_begin();
        MPI_Allreduce_c(MPI_IN_PLACE, buf, n, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        int hits = capture_end();
        long double want = (long double)nprocs * (nprocs + 1) / 2;
        if (hits != 0 || fabsl(buf[0] - want) > 1e-9L) {
            failures++;
            printf("FAIL: rank %d: MPI_LONG_DOUBLE passthrough (hits=%d)\n", rank, hits);
        }
    }

    int total = 0;
    PMPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        printf("%s: MPI_Allreduce_c, %d calls per rank intercepted and verified (np=%d)\n",
               total == 0 ? "PASS" : "FAIL", calls, nprocs);

    MPI_Finalize();
    return total == 0 ? 0 : 1;
}

#else /* MPI_VERSION < 4 */

int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("SKIP: MPI library does not provide MPI-4 large-count bindings\n");
    MPI_Finalize();
    return 0;
}

#endif
