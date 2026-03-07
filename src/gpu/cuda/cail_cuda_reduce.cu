/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "../cail_gpu.h"
#include "../../core/cail_types.h"
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Pinned host byte for PCIe read fence in cail_gpu_flush_recv_buf().
   cudaMemcpy D→H with a single byte forces a PCIe read transaction that
   flushes all prior posted writes (GDRCopy BAR stores, NIC RDMA) to GPU memory. */
static char  *flush_host_byte = NULL;

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4
#define MAX_GRID 65535

typedef void (*cail_cuda_kernel_fn)(const void*, void*, size_t, cudaStream_t);

static cudaStream_t cail_stream = 0;
static cail_cuda_kernel_fn kernel_table[CAIL_NUM_OPS][CAIL_NUM_DTYPES];

template<typename T>
struct OpSum {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct OpProd {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

template<typename T>
struct OpMax {
    __device__ __forceinline__ T operator()(T a, T b) const { return (a > b) ? a : b; }
};

template<typename T>
struct OpMin {
    __device__ __forceinline__ T operator()(T a, T b) const { return (a < b) ? a : b; }
};

template<typename T, typename Op>
__global__ void cail_reduce_kernel(const T *in, T *inout, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    Op op;
    for (size_t i = idx; i < count; i += stride) {
        inout[i] = op(inout[i], in[i]);
    }
}

__global__ void cail_reduce_float4_sum(const float *in, float *inout, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    size_t vec_count = count / 4;
    const float4 *in4 = (const float4*)in;
    float4 *inout4 = (float4*)inout;
    for (size_t i = idx; i < vec_count; i += stride) {
        float4 a = inout4[i];
        float4 b = in4[i];
        inout4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    size_t rem_start = vec_count * 4;
    for (size_t i = rem_start + idx; i < count; i += stride) {
        inout[i] += in[i];
    }
}

__global__ void cail_reduce_double2_sum(const double *in, double *inout, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    size_t vec_count = count / 2;
    const double2 *in2 = (const double2*)in;
    double2 *inout2 = (double2*)inout;
    for (size_t i = idx; i < vec_count; i += stride) {
        double2 a = inout2[i];
        double2 b = in2[i];
        inout2[i] = make_double2(a.x + b.x, a.y + b.y);
    }
    size_t rem_start = vec_count * 2;
    for (size_t i = rem_start + idx; i < count; i += stride) {
        inout[i] += in[i];
    }
}

static inline bool is_aligned(const void *ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

static inline int grid_for_scalar(size_t count) {
    if (count == 0) return 0;
    size_t blocks = (count + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    if (blocks > MAX_GRID) blocks = MAX_GRID;
    return (int)blocks;
}

static inline int grid_for_vec(size_t vec_count) {
    if (vec_count == 0) return 0;
    size_t blocks = (vec_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > MAX_GRID) blocks = MAX_GRID;
    return (int)blocks;
}

template<typename T, typename Op>
static void launch_scalar(const void *in, void *inout, size_t count, cudaStream_t s) {
    int grid = grid_for_scalar(count);
    if (grid > 0) {
        cail_reduce_kernel<T, Op><<<grid, BLOCK_SIZE, 0, s>>>((const T*)in, (T*)inout, count);
    }
}

static void launch_sum_float(const void *in, void *inout, size_t count, cudaStream_t s) {
    if (count == 0) return;
    if ((count % 4) == 0 && is_aligned(in, sizeof(float4)) && is_aligned(inout, sizeof(float4))) {
        int grid = grid_for_vec(count / 4);
        if (grid > 0) {
            cail_reduce_float4_sum<<<grid, BLOCK_SIZE, 0, s>>>((const float*)in, (float*)inout, count);
            return;
        }
    }
    launch_scalar<float, OpSum<float>>(in, inout, count, s);
}

static void launch_sum_double(const void *in, void *inout, size_t count, cudaStream_t s) {
    if (count == 0) return;
    if ((count % 2) == 0 && is_aligned(in, sizeof(double2)) && is_aligned(inout, sizeof(double2))) {
        int grid = grid_for_vec(count / 2);
        if (grid > 0) {
            cail_reduce_double2_sum<<<grid, BLOCK_SIZE, 0, s>>>((const double*)in, (double*)inout, count);
            return;
        }
    }
    launch_scalar<double, OpSum<double>>(in, inout, count, s);
}

#define DECL_LAUNCH_SUM(type, name) \
static void launch_sum_##name(const void *in, void *inout, size_t count, cudaStream_t s) { \
    launch_scalar<type, OpSum<type>>(in, inout, count, s); \
}

#define DECL_LAUNCH_PROD(type, name) \
static void launch_prod_##name(const void *in, void *inout, size_t count, cudaStream_t s) { \
    launch_scalar<type, OpProd<type>>(in, inout, count, s); \
}

#define DECL_LAUNCH_MAX(type, name) \
static void launch_max_##name(const void *in, void *inout, size_t count, cudaStream_t s) { \
    launch_scalar<type, OpMax<type>>(in, inout, count, s); \
}

#define DECL_LAUNCH_MIN(type, name) \
static void launch_min_##name(const void *in, void *inout, size_t count, cudaStream_t s) { \
    launch_scalar<type, OpMin<type>>(in, inout, count, s); \
}

DECL_LAUNCH_SUM(char, char)
DECL_LAUNCH_SUM(int, int)
DECL_LAUNCH_SUM(long, long)
DECL_LAUNCH_SUM(long long, longlong)
DECL_LAUNCH_SUM(unsigned char, uchar)
DECL_LAUNCH_SUM(unsigned int, uint)
DECL_LAUNCH_SUM(unsigned long, ulong)
DECL_LAUNCH_SUM(unsigned long long, ulonglong)
DECL_LAUNCH_SUM(short, short)
DECL_LAUNCH_SUM(unsigned short, ushort)

DECL_LAUNCH_PROD(char, char)
DECL_LAUNCH_PROD(short, short)
DECL_LAUNCH_PROD(int, int)
DECL_LAUNCH_PROD(long, long)
DECL_LAUNCH_PROD(float, float)
DECL_LAUNCH_PROD(double, double)
DECL_LAUNCH_PROD(long long, longlong)
DECL_LAUNCH_PROD(unsigned char, uchar)
DECL_LAUNCH_PROD(unsigned short, ushort)
DECL_LAUNCH_PROD(unsigned int, uint)
DECL_LAUNCH_PROD(unsigned long, ulong)
DECL_LAUNCH_PROD(unsigned long long, ulonglong)

DECL_LAUNCH_MAX(char, char)
DECL_LAUNCH_MAX(short, short)
DECL_LAUNCH_MAX(int, int)
DECL_LAUNCH_MAX(long, long)
DECL_LAUNCH_MAX(float, float)
DECL_LAUNCH_MAX(double, double)
DECL_LAUNCH_MAX(long long, longlong)
DECL_LAUNCH_MAX(unsigned char, uchar)
DECL_LAUNCH_MAX(unsigned short, ushort)
DECL_LAUNCH_MAX(unsigned int, uint)
DECL_LAUNCH_MAX(unsigned long, ulong)
DECL_LAUNCH_MAX(unsigned long long, ulonglong)

DECL_LAUNCH_MIN(char, char)
DECL_LAUNCH_MIN(short, short)
DECL_LAUNCH_MIN(int, int)
DECL_LAUNCH_MIN(long, long)
DECL_LAUNCH_MIN(float, float)
DECL_LAUNCH_MIN(double, double)
DECL_LAUNCH_MIN(long long, longlong)
DECL_LAUNCH_MIN(unsigned char, uchar)
DECL_LAUNCH_MIN(unsigned short, ushort)
DECL_LAUNCH_MIN(unsigned int, uint)
DECL_LAUNCH_MIN(unsigned long, ulong)
DECL_LAUNCH_MIN(unsigned long long, ulonglong)

static void init_kernel_table(void) {
    for (int op = 0; op < CAIL_NUM_OPS; ++op) {
        for (int dt = 0; dt < CAIL_NUM_DTYPES; ++dt) {
            kernel_table[op][dt] = 0;
        }
    }

    kernel_table[CAIL_SUM][CAIL_CHAR]      = launch_sum_char;
    kernel_table[CAIL_SUM][CAIL_INT]       = launch_sum_int;
    kernel_table[CAIL_SUM][CAIL_LONG]      = launch_sum_long;
    kernel_table[CAIL_SUM][CAIL_FLOAT]     = launch_sum_float;
    kernel_table[CAIL_SUM][CAIL_DOUBLE]    = launch_sum_double;
    kernel_table[CAIL_SUM][CAIL_LONG_LONG] = launch_sum_longlong;
    kernel_table[CAIL_SUM][CAIL_UCHAR]     = launch_sum_uchar;
    kernel_table[CAIL_SUM][CAIL_UINT]      = launch_sum_uint;
    kernel_table[CAIL_SUM][CAIL_ULONG]     = launch_sum_ulong;
    kernel_table[CAIL_SUM][CAIL_ULONGLONG] = launch_sum_ulonglong;
    kernel_table[CAIL_SUM][CAIL_SHORT]     = launch_sum_short;
    kernel_table[CAIL_SUM][CAIL_USHORT]    = launch_sum_ushort;

    kernel_table[CAIL_PROD][CAIL_CHAR]      = launch_prod_char;
    kernel_table[CAIL_PROD][CAIL_SHORT]     = launch_prod_short;
    kernel_table[CAIL_PROD][CAIL_INT]       = launch_prod_int;
    kernel_table[CAIL_PROD][CAIL_LONG]      = launch_prod_long;
    kernel_table[CAIL_PROD][CAIL_FLOAT]     = launch_prod_float;
    kernel_table[CAIL_PROD][CAIL_DOUBLE]    = launch_prod_double;
    kernel_table[CAIL_PROD][CAIL_LONG_LONG] = launch_prod_longlong;
    kernel_table[CAIL_PROD][CAIL_UCHAR]     = launch_prod_uchar;
    kernel_table[CAIL_PROD][CAIL_USHORT]    = launch_prod_ushort;
    kernel_table[CAIL_PROD][CAIL_UINT]      = launch_prod_uint;
    kernel_table[CAIL_PROD][CAIL_ULONG]     = launch_prod_ulong;
    kernel_table[CAIL_PROD][CAIL_ULONGLONG] = launch_prod_ulonglong;

    kernel_table[CAIL_MAX][CAIL_CHAR]      = launch_max_char;
    kernel_table[CAIL_MAX][CAIL_SHORT]     = launch_max_short;
    kernel_table[CAIL_MAX][CAIL_INT]       = launch_max_int;
    kernel_table[CAIL_MAX][CAIL_LONG]      = launch_max_long;
    kernel_table[CAIL_MAX][CAIL_FLOAT]     = launch_max_float;
    kernel_table[CAIL_MAX][CAIL_DOUBLE]    = launch_max_double;
    kernel_table[CAIL_MAX][CAIL_LONG_LONG] = launch_max_longlong;
    kernel_table[CAIL_MAX][CAIL_UCHAR]     = launch_max_uchar;
    kernel_table[CAIL_MAX][CAIL_USHORT]    = launch_max_ushort;
    kernel_table[CAIL_MAX][CAIL_UINT]      = launch_max_uint;
    kernel_table[CAIL_MAX][CAIL_ULONG]     = launch_max_ulong;
    kernel_table[CAIL_MAX][CAIL_ULONGLONG] = launch_max_ulonglong;

    kernel_table[CAIL_MIN][CAIL_CHAR]      = launch_min_char;
    kernel_table[CAIL_MIN][CAIL_SHORT]     = launch_min_short;
    kernel_table[CAIL_MIN][CAIL_INT]       = launch_min_int;
    kernel_table[CAIL_MIN][CAIL_LONG]      = launch_min_long;
    kernel_table[CAIL_MIN][CAIL_FLOAT]     = launch_min_float;
    kernel_table[CAIL_MIN][CAIL_DOUBLE]    = launch_min_double;
    kernel_table[CAIL_MIN][CAIL_LONG_LONG] = launch_min_longlong;
    kernel_table[CAIL_MIN][CAIL_UCHAR]     = launch_min_uchar;
    kernel_table[CAIL_MIN][CAIL_USHORT]    = launch_min_ushort;
    kernel_table[CAIL_MIN][CAIL_UINT]      = launch_min_uint;
    kernel_table[CAIL_MIN][CAIL_ULONG]     = launch_min_ulong;
    kernel_table[CAIL_MIN][CAIL_ULONGLONG] = launch_min_ulonglong;
}

extern "C" int cail_cuda_init(void) {
    if (cail_stream != 0) {
        return 0;
    }
    cudaError_t err = cudaStreamCreate(&cail_stream);
    if (err != cudaSuccess) {
        cail_stream = 0;
        return -1;
    }
    init_kernel_table();

    /* Allocate a pinned host byte for the PCIe read fence used by
       cail_gpu_flush_recv_buf(). */
    if (cudaHostAlloc(&flush_host_byte, 1, cudaHostAllocDefault) != cudaSuccess)
        flush_host_byte = NULL;

    return 0;
}

extern "C" void cail_cuda_finalize(void) {
    if (flush_host_byte) { cudaFreeHost(flush_host_byte); flush_host_byte = NULL; }
    if (cail_stream != 0) {
        cudaStreamDestroy(cail_stream);
        cail_stream = 0;
    }
}

extern "C" int cail_cuda_synchronize(void) {
    if (cail_stream == 0) return -1;
    return (cudaStreamSynchronize(cail_stream) == cudaSuccess) ? 0 : -1;
}

/* PCIe read fence: reads the last byte of the receive buffer into pinned
   host memory, forcing a D→H transaction that drains any prior posted
   writes (NIC RDMA, GDRCopy BAR stores) to the same GPU memory region. */
extern "C" int cail_gpu_flush_recv_buf(const void *recv_buf, size_t recv_bytes) {
    if (!flush_host_byte || !recv_buf || recv_bytes == 0) return 0;
    cudaMemcpy(flush_host_byte, (const char*)recv_buf + recv_bytes - 1,
               1, cudaMemcpyDeviceToHost);
    return 0;
}


extern "C" int cail_gpu_reduce_local(const void *in, void *inout, size_t count,
                                       cail_datatype_t dtype, cail_op_t op) {
    if (op < 0 || op >= CAIL_NUM_OPS || dtype < 0 || dtype >= CAIL_NUM_DTYPES) {
        fprintf(stderr, "[cail WARN] gpu_reduce_local: dtype=%d op=%d out of range\n",
                (int)dtype, (int)op);
        return -1;
    }
    cail_cuda_kernel_fn fn = kernel_table[op][dtype];
    if (fn == 0) {
        fprintf(stderr, "[cail WARN] gpu_reduce_local: no kernel for dtype=%s op=%s\n",
                cail_dtype_name(dtype), cail_op_name(op));
        return -1;
    }
    if (cail_stream == 0) {
        fprintf(stderr, "[cail WARN] gpu_reduce_local: CUDA stream not initialized\n");
        return -1;
    }
    fn(in, inout, count, cail_stream);
    cudaError_t err = cudaStreamSynchronize(cail_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cail WARN] gpu_reduce_local: cudaStreamSynchronize failed (%s) "
                "for dtype=%s op=%s count=%zu\n",
                cudaGetErrorString(err), cail_dtype_name(dtype), cail_op_name(op), count);
        return -1;
    }
    return 0;
}
