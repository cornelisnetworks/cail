/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int all_pass = 1;

    /* count=0: should succeed without error */
    {
        float dummy_send = 1.0f, dummy_recv = 0.0f;
        int rc = MPI_Allreduce(&dummy_send, &dummy_recv, 0,
                               MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        int pass = (rc == MPI_SUCCESS);
        if (rank == 0) printf("%s: count=0\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
    }

    /* count=1: single element */
    {
        float send = (float)(rank + 1);
        float recv = 0.0f;
        MPI_Allreduce(&send, &recv, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        float expected = 0.0f;
        for (int r = 0; r < nprocs; r++) expected += (float)(r + 1);
        int pass = (fabsf(recv - expected) < 1e-3f);
        if (rank == 0) printf("%s: count=1\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
    }

    /* large count: exercises ring path (>512KB with float = >131072 elements) */
    {
        int count = 200000;
        float *sendbuf = (float *)malloc(count * sizeof(float));
        float *recvbuf = (float *)malloc(count * sizeof(float));
        if (!sendbuf || !recvbuf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) sendbuf[i] = (float)(rank + 1);
        memset(recvbuf, 0, count * sizeof(float));
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        float expected = 0.0f;
        for (int r = 0; r < nprocs; r++) expected += (float)(r + 1);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (fabsf(recvbuf[i] - expected) > 1e-3f) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: count=200000 (large)\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(sendbuf);
        free(recvbuf);
    }

    /* MPI_LONG_DOUBLE: unsupported type falls back to PMPI */
    {
        int count = 10;
        long double *sendbuf = (long double *)malloc(count * sizeof(long double));
        long double *recvbuf = (long double *)malloc(count * sizeof(long double));
        if (!sendbuf || !recvbuf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) sendbuf[i] = (long double)(rank + 1);
        memset(recvbuf, 0, count * sizeof(long double));
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        long double expected = 0.0L;
        for (int r = 0; r < nprocs; r++) expected += (long double)(r + 1);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (fabsl(recvbuf[i] - expected) > 1e-6L) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_LONG_DOUBLE fallback\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(sendbuf);
        free(recvbuf);
    }

    /* MPI_BAND: unsupported op falls back to PMPI */
    {
        int count = 10;
        unsigned int *sendbuf = (unsigned int *)malloc(count * sizeof(unsigned int));
        unsigned int *recvbuf = (unsigned int *)malloc(count * sizeof(unsigned int));
        if (!sendbuf || !recvbuf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) sendbuf[i] = 0xFFFFFFFFu;
        memset(recvbuf, 0, count * sizeof(unsigned int));
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_UNSIGNED, MPI_BAND, MPI_COMM_WORLD);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (recvbuf[i] != 0xFFFFFFFFu) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_BAND fallback\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(sendbuf);
        free(recvbuf);
    }

    /* MPI_IN_PLACE count=0: should succeed */
    {
        float dummy = 1.0f;
        int rc = MPI_Allreduce(MPI_IN_PLACE, &dummy, 0,
                               MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        int pass = (rc == MPI_SUCCESS);
        if (rank == 0) printf("%s: MPI_IN_PLACE count=0\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
    }

    /* MPI_IN_PLACE count=1: single element */
    {
        float val = (float)(rank + 1);
        MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        float expected = 0.0f;
        for (int r = 0; r < nprocs; r++) expected += (float)(r + 1);
        int pass = (fabsf(val - expected) < 1e-3f);
        if (rank == 0) printf("%s: MPI_IN_PLACE count=1\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
    }

    /* MPI_IN_PLACE large count */
    {
        int count = 200000;
        float *buf = (float *)malloc(count * sizeof(float));
        if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) buf[i] = (float)(rank + 1);
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        float expected = 0.0f;
        for (int r = 0; r < nprocs; r++) expected += (float)(r + 1);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (fabsf(buf[i] - expected) > 1e-3f) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_IN_PLACE count=200000 (large)\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(buf);
    }

    /* MPI_IN_PLACE with int MAX */
    {
        int count = 4096;
        int *buf = (int *)malloc(count * sizeof(int));
        if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) buf[i] = rank + 1;
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (buf[i] != nprocs) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_IN_PLACE int MAX count=4096\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(buf);
    }

    /* MPI_IN_PLACE with double MIN */
    {
        int count = 4096;
        double *buf = (double *)malloc(count * sizeof(double));
        if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) buf[i] = (double)(rank + 1);
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (fabs(buf[i] - 1.0) > 1e-12) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_IN_PLACE double MIN count=4096\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(buf);
    }

    /* MPI_IN_PLACE with unsupported type (LONG_DOUBLE) falls back to PMPI */
    {
        int count = 10;
        long double *buf = (long double *)malloc(count * sizeof(long double));
        if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) buf[i] = (long double)(rank + 1);
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        long double expected = 0.0L;
        for (int r = 0; r < nprocs; r++) expected += (long double)(r + 1);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (fabsl(buf[i] - expected) > 1e-6L) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_IN_PLACE MPI_LONG_DOUBLE fallback\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(buf);
    }

    /* MPI_IN_PLACE with unsupported op (BAND) falls back to PMPI */
    {
        int count = 10;
        unsigned int *buf = (unsigned int *)malloc(count * sizeof(unsigned int));
        if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < count; i++) buf[i] = 0xFFFFFFFFu;
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_UNSIGNED, MPI_BAND, MPI_COMM_WORLD);
        int pass = 1;
        for (int i = 0; i < count; i++) {
            if (buf[i] != 0xFFFFFFFFu) { pass = 0; break; }
        }
        if (rank == 0) printf("%s: MPI_IN_PLACE MPI_BAND fallback\n", pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
        free(buf);
    }

    MPI_Finalize();
    return all_pass ? 0 : 1;
}
