/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_WARMUP     10
#define DEFAULT_ITERATIONS 100

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t min_bytes  = 4;
    size_t max_bytes  = 16 * 1024 * 1024;
    int    iterations = DEFAULT_ITERATIONS;
    int    warmup     = DEFAULT_WARMUP;

    if (argc > 1) min_bytes  = (size_t)atol(argv[1]);
    if (argc > 2) max_bytes  = (size_t)atol(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    if (argc > 4) warmup     = atoi(argv[4]);

    if (rank == 0) {
        printf("# cail bench_allreduce: nprocs=%d iterations=%d warmup=%d\n",
               nprocs, iterations, warmup);
        printf("# size_bytes,min_us,max_us,avg_us,bw_gbps\n");
    }

    float *buf = (float *)malloc(max_bytes);
    if (!buf) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    memset(buf, 0, max_bytes);

    for (size_t size = min_bytes; size <= max_bytes; size *= 4) {
        int count = (int)(size / sizeof(float));
        if (count < 1) count = 1;

        for (int i = 0; i < warmup; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        double min_t = 1e30, max_t = 0.0, sum_t = 0.0;
        for (int i = 0; i < iterations; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            double dt = t1 - t0;
            if (dt < min_t) min_t = dt;
            if (dt > max_t) max_t = dt;
            sum_t += dt;
        }

        double avg_t = sum_t / iterations;
        double bw = 2.0 * (double)(nprocs - 1) / (double)nprocs * (double)size / avg_t / 1e9;

        if (rank == 0) {
            printf("%zu,%.3f,%.3f,%.3f,%.3f\n",
                   size, min_t * 1e6, max_t * 1e6, avg_t * 1e6, bw);
            fflush(stdout);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
