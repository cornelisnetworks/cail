/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

/*
 * test_allreduce_gpu.cu — GPU-buffer correctness tests for cail
 *
 * Allocates send/recv buffers on CUDA device memory, fills them via
 * kernels, calls MPI_Allreduce (which cail intercepts), then copies
 * results back to host for verification.
 *
 * Exercises: basic SUM, MPI_IN_PLACE, multiple datatypes/ops,
 *            multiple message sizes, all cail algorithms.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                                \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                  \
} while (0)

/* ------------------------------------------------------------------ */
/* Kernels to fill device buffers                                     */
/* ------------------------------------------------------------------ */

__global__ void fill_float(float *buf, int count, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) buf[idx] = val;
}

__global__ void fill_double(double *buf, int count, double val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) buf[idx] = val;
}

__global__ void fill_int(int *buf, int count, int val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) buf[idx] = val;
}

static int grid(int count) { return (count + 255) / 256; }

/* ------------------------------------------------------------------ */
/* Test helpers                                                       */
/* ------------------------------------------------------------------ */

static int g_pass = 0;
static int g_fail = 0;

static void report(int rank, int ok, const char *label) {
    if (rank == 0) {
        printf("%s: %s\n", ok ? "PASS" : "FAIL", label);
        fflush(stdout);
    }
    if (ok) g_pass++; else g_fail++;
}

/* ------------------------------------------------------------------ */
/* Test 1: float SUM, various counts                                  */
/* ------------------------------------------------------------------ */
static void test_float_sum(int rank, int nprocs) {
    int counts[] = {1, 10, 100, 1000, 4096, 100000, 200000};
    int ncounts = sizeof(counts) / sizeof(counts[0]);

    float expected = 0.0f;
    for (int r = 0; r < nprocs; r++) expected += (float)(r + 1);

    for (int ci = 0; ci < ncounts; ci++) {
        int count = counts[ci];
        float *d_send, *d_recv;
        CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_recv, 0, count * sizeof(float)));

        fill_float<<<grid(count), 256>>>(d_send, count, (float)(rank + 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Allreduce(d_send, d_recv, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        float *h_recv = (float *)malloc(count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(float), cudaMemcpyDeviceToHost));

        int ok = 1;
        int fail_idx = -1;
        float fail_val = 0.0f;
        for (int i = 0; i < count; i++) {
            if (fabsf(h_recv[i] - expected) > 1e-3f) {
                ok = 0; fail_idx = i; fail_val = h_recv[i]; break;
            }
        }

        char label[256];
        if (!ok) {
            snprintf(label, sizeof(label),
                     "GPU float SUM count=%d np=%d (rank=%d idx=%d got=%g expected=%g)",
                     count, nprocs, rank, fail_idx, (double)fail_val, (double)expected);
        } else {
            snprintf(label, sizeof(label), "GPU float SUM count=%d np=%d", count, nprocs);
        }
        report(rank, ok, label);

        free(h_recv);
        cudaFree(d_send);
        cudaFree(d_recv);
    }
}

/* ------------------------------------------------------------------ */
/* Test 2: MPI_IN_PLACE with GPU buffers                              */
/* ------------------------------------------------------------------ */
static void test_inplace(int rank, int nprocs) {
    int counts[] = {100, 4096, 200000};
    int ncounts = sizeof(counts) / sizeof(counts[0]);

    float expected = 0.0f;
    for (int r = 0; r < nprocs; r++) expected += (float)(r + 1);

    for (int ci = 0; ci < ncounts; ci++) {
        int count = counts[ci];
        float *d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, count * sizeof(float)));

        fill_float<<<grid(count), 256>>>(d_buf, count, (float)(rank + 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Allreduce(MPI_IN_PLACE, d_buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        float *h_buf = (float *)malloc(count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, count * sizeof(float), cudaMemcpyDeviceToHost));

        int ok = 1;
        for (int i = 0; i < count; i++) {
            if (fabsf(h_buf[i] - expected) > 1e-3f) { ok = 0; break; }
        }

        char label[128];
        snprintf(label, sizeof(label), "GPU MPI_IN_PLACE float SUM count=%d np=%d", count, nprocs);
        report(rank, ok, label);

        free(h_buf);
        cudaFree(d_buf);
    }
}

/* ------------------------------------------------------------------ */
/* Test 3: double SUM                                                 */
/* ------------------------------------------------------------------ */
static void test_double_sum(int rank, int nprocs) {
    int counts[] = {100, 4096, 100000};
    int ncounts = sizeof(counts) / sizeof(counts[0]);

    double expected = 0.0;
    for (int r = 0; r < nprocs; r++) expected += (double)(r + 1);

    for (int ci = 0; ci < ncounts; ci++) {
        int count = counts[ci];
        double *d_send, *d_recv;
        CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_recv, 0, count * sizeof(double)));

        fill_double<<<grid(count), 256>>>(d_send, count, (double)(rank + 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Allreduce(d_send, d_recv, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double *h_recv = (double *)malloc(count * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(double), cudaMemcpyDeviceToHost));

        int ok = 1;
        int fail_idx = -1;
        double fail_val = 0.0;
        for (int i = 0; i < count; i++) {
            if (fabs(h_recv[i] - expected) > 1e-9) {
                ok = 0; fail_idx = i; fail_val = h_recv[i]; break;
            }
        }

        char label[256];
        if (!ok && rank == 0) {
            snprintf(label, sizeof(label),
                     "GPU double SUM count=%d np=%d (idx=%d got=%.17g expected=%.17g)",
                     count, nprocs, fail_idx, fail_val, expected);
        } else {
            snprintf(label, sizeof(label), "GPU double SUM count=%d np=%d", count, nprocs);
        }
        report(rank, ok, label);

        free(h_recv);
        cudaFree(d_send);
        cudaFree(d_recv);
    }
}

/* ------------------------------------------------------------------ */
/* Test 4: int SUM + MAX + MIN                                        */
/* ------------------------------------------------------------------ */
static void test_int_ops(int rank, int nprocs) {
    int count = 4096;

    /* SUM */
    {
        int *d_send, *d_recv;
        CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_recv, 0, count * sizeof(int)));

        fill_int<<<grid(count), 256>>>(d_send, count, rank + 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Allreduce(d_send, d_recv, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        int *h_recv = (int *)malloc(count * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(int), cudaMemcpyDeviceToHost));

        int expected = 0;
        for (int r = 0; r < nprocs; r++) expected += r + 1;

        int ok = 1;
        for (int i = 0; i < count; i++) {
            if (h_recv[i] != expected) { ok = 0; break; }
        }

        char label[128];
        snprintf(label, sizeof(label), "GPU int SUM count=%d np=%d", count, nprocs);
        report(rank, ok, label);

        free(h_recv);
        cudaFree(d_send);
        cudaFree(d_recv);
    }

    /* MAX */
    {
        int *d_send, *d_recv;
        CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_recv, 0, count * sizeof(int)));

        fill_int<<<grid(count), 256>>>(d_send, count, rank + 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Allreduce(d_send, d_recv, count, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int *h_recv = (int *)malloc(count * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(int), cudaMemcpyDeviceToHost));

        int ok = 1;
        for (int i = 0; i < count; i++) {
            if (h_recv[i] != nprocs) { ok = 0; break; }
        }

        char label[128];
        snprintf(label, sizeof(label), "GPU int MAX count=%d np=%d", count, nprocs);
        report(rank, ok, label);

        free(h_recv);
        cudaFree(d_send);
        cudaFree(d_recv);
    }

    /* MIN */
    {
        int *d_send, *d_recv;
        CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_recv, 0, count * sizeof(int)));

        fill_int<<<grid(count), 256>>>(d_send, count, rank + 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Allreduce(d_send, d_recv, count, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        int *h_recv = (int *)malloc(count * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(int), cudaMemcpyDeviceToHost));

        int ok = 1;
        for (int i = 0; i < count; i++) {
            if (h_recv[i] != 1) { ok = 0; break; }
        }

        char label[128];
        snprintf(label, sizeof(label), "GPU int MIN count=%d np=%d", count, nprocs);
        report(rank, ok, label);

        free(h_recv);
        cudaFree(d_send);
        cudaFree(d_recv);
    }
}

/* ------------------------------------------------------------------ */
/* Test 5: float PROD (small nprocs only to avoid overflow)           */
/* ------------------------------------------------------------------ */
static void test_float_prod(int rank, int nprocs) {
    if (nprocs > 4) return;

    int count = 1000;
    float *d_send, *d_recv;
    CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_recv, 0, count * sizeof(float)));

    fill_float<<<grid(count), 256>>>(d_send, count, (float)(rank + 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Allreduce(d_send, d_recv, count, MPI_FLOAT, MPI_PROD, MPI_COMM_WORLD);

    float *h_recv = (float *)malloc(count * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = 1.0f;
    for (int r = 0; r < nprocs; r++) expected *= (float)(r + 1);

    int ok = 1;
    for (int i = 0; i < count; i++) {
        if (fabsf(h_recv[i] - expected) > fabsf(expected) * 1e-3f + 1e-6f) { ok = 0; break; }
    }

    char label[128];
    snprintf(label, sizeof(label), "GPU float PROD count=%d np=%d", count, nprocs);
    report(rank, ok, label);

    free(h_recv);
    cudaFree(d_send);
    cudaFree(d_recv);
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
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

    if (rank == 0) {
        printf("=== cail GPU allreduce tests ===\n");
        printf("nprocs=%d  GPUs_per_node=%d\n", nprocs, dev_count);
        fflush(stdout);
    }

    test_float_sum(rank, nprocs);
    test_inplace(rank, nprocs);
    test_double_sum(rank, nprocs);
    test_int_ops(rank, nprocs);
    test_float_prod(rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
        fflush(stdout);
    }

    MPI_Finalize();
    return g_fail > 0 ? 1 : 0;
}
