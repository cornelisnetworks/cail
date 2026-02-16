/*
 * CUDA reduction kernels for custom allreduce
 * 
 * Performs element-wise reduction: inout[i] = inout[i] op in[i]
 * 
 * Optimizations:
 * - Vectorized loads (float4/double2) for better memory bandwidth
 * - Increased ILP with multiple elements per thread
 * - Stream-based async execution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

/* Block size for reduction kernel - 512 often better for modern GPUs */
#define BLOCK_SIZE 512

/* Elements per thread for better instruction-level parallelism */
#define ELEMENTS_PER_THREAD 4

/* Global CUDA stream for async operations */
static cudaStream_t reduce_stream = NULL;
static int stream_initialized = 0;

static void ensure_stream_initialized() {
    if (!stream_initialized) {
        cudaError_t err = cudaStreamCreateWithFlags(&reduce_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error creating CUDA stream: %s\n", cudaGetErrorString(err));
            /* Fallback to default stream if creation fails, though highly unlikely */
            reduce_stream = 0; 
        }
        stream_initialized = 1;
    }
}

/*
 * Vectorized float4 SUM kernel - 4x memory bandwidth improvement
 */
__global__ void reduce_sum_float4_kernel(const float4 *in, float4 *inout, size_t count4)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count4; i += stride) {
        float4 a = inout[i];
        float4 b = in[i];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        inout[i] = a;
    }
}

/*
 * Vectorized double2 SUM kernel
 */
__global__ void reduce_sum_double2_kernel(const double2 *in, double2 *inout, size_t count2)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count2; i += stride) {
        double2 a = inout[i];
        double2 b = in[i];
        a.x += b.x; a.y += b.y;
        inout[i] = a;
    }
}

/*
 * Generic element-wise SUM reduction kernel
 * Each thread processes ELEMENTS_PER_THREAD elements for better ILP
 */
template<typename T>
__global__ void reduce_sum_kernel(const T *in, T *inout, size_t count)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    size_t stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    
    for (size_t base = idx; base < count; base += stride) {
        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD && base + k < count; k++) {
            inout[base + k] = inout[base + k] + in[base + k];
        }
    }
}

template<typename T>
__global__ void reduce_prod_kernel(const T *in, T *inout, size_t count)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    size_t stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    
    for (size_t base = idx; base < count; base += stride) {
        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD && base + k < count; k++) {
            inout[base + k] = inout[base + k] * in[base + k];
        }
    }
}

template<typename T>
__global__ void reduce_max_kernel(const T *in, T *inout, size_t count)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    size_t stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    
    for (size_t base = idx; base < count; base += stride) {
        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD && base + k < count; k++) {
            size_t i = base + k;
            inout[i] = (in[i] > inout[i]) ? in[i] : inout[i];
        }
    }
}

template<typename T>
__global__ void reduce_min_kernel(const T *in, T *inout, size_t count)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    size_t stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    
    for (size_t base = idx; base < count; base += stride) {
        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD && base + k < count; k++) {
            size_t i = base + k;
            inout[i] = (in[i] < inout[i]) ? in[i] : inout[i];
        }
    }
}

/* Launcher macro with stream support and adjusted block count for elements-per-thread */
#define LAUNCH_REDUCE(kernel, type, in, inout, count) do { \
    size_t total_threads = ((count) + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD; \
    int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE; \
    /* Modern GPUs support > 65535 blocks, but let's cap reasonably to avoid launch overhead */ \
    if (num_blocks > 2147483647) num_blocks = 2147483647; \
    kernel<type><<<num_blocks, BLOCK_SIZE, 0, reduce_stream>>>( \
        (const type*)(in), (type*)(inout), count); \
} while(0)

/* Launcher for vectorized kernels */
#define LAUNCH_REDUCE_VEC(kernel, in, inout, count_vec) do { \
    int num_blocks = ((count_vec) + BLOCK_SIZE - 1) / BLOCK_SIZE; \
    /* Modern GPUs support > 65535 blocks */ \
    if (num_blocks > 2147483647) num_blocks = 2147483647; \
    kernel<<<num_blocks, BLOCK_SIZE, 0, reduce_stream>>>( \
        (in), (inout), count_vec); \
} while(0)

/*
 * Check if memory is aligned for vectorized access
 */
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

/*
 * C interface for reduction operations
 * 
 * op_type: 0=SUM, 1=PROD, 2=MAX, 3=MIN
 * dtype: 0=char, 1=int, 2=long, 3=float, 4=double, 5=long long,
 *        6=uchar, 7=uint, 8=ulong, 9=ulonglong, 10=short, 11=ushort  
 * 
 * Uses vectorized kernels (float4/double2) when possible for 2-4x speedup
 */
extern "C" int cuda_reduce_local(const void *in, void *inout, size_t count,
                                  int dtype, int op_type)
{
    ensure_stream_initialized();
    
    switch (op_type) {
    case 0: /* SUM */
        switch (dtype) {
        case 0: LAUNCH_REDUCE(reduce_sum_kernel, char, in, inout, count); break;
        case 1: LAUNCH_REDUCE(reduce_sum_kernel, int, in, inout, count); break;
        case 2: LAUNCH_REDUCE(reduce_sum_kernel, long, in, inout, count); break;
        case 3: /* float - try vectorized first */
            if (count >= 4 && (count % 4 == 0) && 
                is_aligned(in, 16) && is_aligned(inout, 16)) {
                LAUNCH_REDUCE_VEC(reduce_sum_float4_kernel, 
                    (const float4*)in, (float4*)inout, count / 4);
            } else {
                LAUNCH_REDUCE(reduce_sum_kernel, float, in, inout, count);
            }
            break;
        case 4: /* double - try vectorized first */
            if (count >= 2 && (count % 2 == 0) && 
                is_aligned(in, 16) && is_aligned(inout, 16)) {
                LAUNCH_REDUCE_VEC(reduce_sum_double2_kernel,
                    (const double2*)in, (double2*)inout, count / 2);
            } else {
                LAUNCH_REDUCE(reduce_sum_kernel, double, in, inout, count);
            }
            break;
        case 5: LAUNCH_REDUCE(reduce_sum_kernel, long long, in, inout, count); break;
        case 6: LAUNCH_REDUCE(reduce_sum_kernel, unsigned char, in, inout, count); break;
        case 7: LAUNCH_REDUCE(reduce_sum_kernel, unsigned int, in, inout, count); break;
        case 8: LAUNCH_REDUCE(reduce_sum_kernel, unsigned long, in, inout, count); break;
        case 9: LAUNCH_REDUCE(reduce_sum_kernel, unsigned long long, in, inout, count); break;
        case 10: LAUNCH_REDUCE(reduce_sum_kernel, short, in, inout, count); break;
        case 11: LAUNCH_REDUCE(reduce_sum_kernel, unsigned short, in, inout, count); break;
        default: return -1;
        }
        break;
    case 1: /* PROD */
        switch (dtype) {
        case 0: LAUNCH_REDUCE(reduce_prod_kernel, char, in, inout, count); break;
        case 1: LAUNCH_REDUCE(reduce_prod_kernel, int, in, inout, count); break;
        case 2: LAUNCH_REDUCE(reduce_prod_kernel, long, in, inout, count); break;
        case 3: LAUNCH_REDUCE(reduce_prod_kernel, float, in, inout, count); break;
        case 4: LAUNCH_REDUCE(reduce_prod_kernel, double, in, inout, count); break;
        case 5: LAUNCH_REDUCE(reduce_prod_kernel, long long, in, inout, count); break;
        default: return -1;
        }
        break;
    case 2: /* MAX */
        switch (dtype) {
        case 0: LAUNCH_REDUCE(reduce_max_kernel, char, in, inout, count); break;
        case 1: LAUNCH_REDUCE(reduce_max_kernel, int, in, inout, count); break;
        case 2: LAUNCH_REDUCE(reduce_max_kernel, long, in, inout, count); break;
        case 3: LAUNCH_REDUCE(reduce_max_kernel, float, in, inout, count); break;
        case 4: LAUNCH_REDUCE(reduce_max_kernel, double, in, inout, count); break;
        case 5: LAUNCH_REDUCE(reduce_max_kernel, long long, in, inout, count); break;
        default: return -1;
        }
        break;
    case 3: /* MIN */
        switch (dtype) {
        case 0: LAUNCH_REDUCE(reduce_min_kernel, char, in, inout, count); break;
        case 1: LAUNCH_REDUCE(reduce_min_kernel, int, in, inout, count); break;
        case 2: LAUNCH_REDUCE(reduce_min_kernel, long, in, inout, count); break;
        case 3: LAUNCH_REDUCE(reduce_min_kernel, float, in, inout, count); break;
        case 4: LAUNCH_REDUCE(reduce_min_kernel, double, in, inout, count); break;
        case 5: LAUNCH_REDUCE(reduce_min_kernel, long long, in, inout, count); break;
        default: return -1;
        }
        break;
    default:
        return -1;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA reduce kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/*
 * Synchronous version - waits for stream completion
 * More efficient than cudaDeviceSynchronize() as it only waits on our stream
 */
extern "C" int cuda_reduce_local_sync(const void *in, void *inout, size_t count,
                                       int dtype, int op_type)
{
    int ret = cuda_reduce_local(in, inout, count, dtype, op_type);
    if (ret == 0) {
        cudaStreamSynchronize(reduce_stream);
    }
    return ret;
}

/*
 * Cleanup function to destroy stream
 */
extern "C" void cuda_reduce_cleanup(void)
{
    if (stream_initialized && reduce_stream != NULL) {
        cudaStreamDestroy(reduce_stream);
        reduce_stream = NULL;
        stream_initialized = 0;
    }
}
