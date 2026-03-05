/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-c count] [-a algo] [-h]\n", prog);
    fprintf(stderr, "  -c count   element count (default: sweep 1..200000)\n");
    fprintf(stderr, "  -a algo    set CAIL_ALGO before MPI_Init\n");
}

int main(int argc, char **argv)
{
    int opt_count = 0;
    int opt;

    while ((opt = getopt(argc, argv, "c:a:h")) != -1) {
        switch (opt) {
        case 'c':
            opt_count = atoi(optarg);
            break;
        case 'a':
            setenv("CAIL_ALGO", optarg, 1);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int counts[] = {1, 10, 100, 1000, 4096, 100000, 200000};
    int ncounts = (int)(sizeof(counts) / sizeof(counts[0]));
    if (opt_count > 0) { counts[0] = opt_count; ncounts = 1; }

    float expected_sum = 0.0f;
    for (int r = 0; r < nprocs; r++) expected_sum += (float)(r + 1);

    int all_pass = 1;
    for (int ci = 0; ci < ncounts; ci++) {
        int count = counts[ci];
        float *sendbuf = (float *)malloc(count * sizeof(float));
        float *recvbuf = (float *)malloc(count * sizeof(float));
        if (!sendbuf || !recvbuf) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) sendbuf[i] = (float)(rank + 1);
        memset(recvbuf, 0, count * sizeof(float));
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (fabsf(recvbuf[i] - expected_sum) > 1e-3f) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: count=%d np=%d\n", pass ? "PASS" : "FAIL", count, nprocs);
        if (!pass) all_pass = 0;
        free(sendbuf); free(recvbuf);
    }
    MPI_Finalize();
    return all_pass ? 0 : 1;
}
