/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/*
 * bench_allreduce_gpu.cu — GPU-buffer benchmark for cail MPI_Allreduce
 *
 * Same as bench_allreduce.c but uses cudaMalloc'd device buffers so
 * cail's GPU path is exercised instead of falling back to PMPI.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_WARMUP     10
#define DEFAULT_ITERATIONS 100

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                                \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                  \
} while (0)

int main(int argc, char **argv)
{
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

    size_t min_bytes  = 4;
    size_t max_bytes  = 16 * 1024 * 1024;
    int    iterations = DEFAULT_ITERATIONS;
    int    warmup     = DEFAULT_WARMUP;

    if (argc > 1) min_bytes  = (size_t)atol(argv[1]);
    if (argc > 2) max_bytes  = (size_t)atol(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    if (argc > 4) warmup     = atoi(argv[4]);

    if (rank == 0) {
        printf("# cail bench_allreduce_gpu: nprocs=%d gpus_per_node=%d iterations=%d warmup=%d\n",
               nprocs, dev_count, iterations, warmup);
        printf("# size_bytes,min_us,max_us,avg_us,bw_gbps\n");
    }

    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));
    CUDA_CHECK(cudaMemset(d_buf, 0, max_bytes));

    for (size_t size = min_bytes; size <= max_bytes; size *= 4) {
        int count = (int)(size / sizeof(float));
        if (count < 1) count = 1;

        if (size > 8192) {
            warmup = 10;
            iterations = DEFAULT_ITERATIONS > 100 ? 100 : DEFAULT_ITERATIONS;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double timer = 0.0;
        for (int i = 0; i < iterations + warmup; i++) {
            double t0 = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, d_buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (i >= warmup) {
                timer += t1 - t0;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        double latency = (timer * 1e6) / iterations;
        double min_t, max_t, avg_t;
        MPI_Reduce(&latency, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_t /= nprocs;

        double bw = 2.0 * (double)(nprocs - 1) / (double)nprocs * (double)size / (avg_t / 1e6) / 1e9;

        if (rank == 0) {
            printf("%zu,%.3f,%.3f,%.3f,%.3f\n",
                   size, min_t, max_t, avg_t, bw);
            fflush(stdout);
        }
    }

    cudaFree(d_buf);
    MPI_Finalize();
    return 0;
}
