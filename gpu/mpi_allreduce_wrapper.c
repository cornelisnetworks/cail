/*
 * Custom MPI_Allreduce Interposition Library
 * 
 * This library intercepts MPI_Allreduce calls to implement custom
 * GPU-aware collective operations that avoid staging through host memory.
 * 
 * Implements two algorithms:
 * - Recursive doubling: O(log P) steps, latency-optimized for small messages
 * - Ring allreduce: O(P) steps, bandwidth-optimized for large messages
 * 
 * Environment variables:
 *   CUSTOM_ALLREDUCE_ALGO   - Algorithm selection: auto|recursive|ring (default: auto)
 *   CUSTOM_ALLREDUCE_DEBUG  - Enable debug output: 0|1 (default: 0)
 * 
 * Usage: LD_PRELOAD=./libcustom_allreduce.so mpirun -np N ./your_program
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "common.h"
#include "recursive_doubling.h"

/* Debug flag */
static int debug = -1;
static int initialized = 0;

/* Algorithm selection */
typedef enum {
    ALGO_RECURSIVE = 1     /* Always use recursive doubling */
} AlgoType;

static AlgoType selected_algo = ALGO_RECURSIVE;
static int algo_initialized = 0;

/* Algorithm selection threshold */
#define RING_THRESHOLD (1024 * 1024)  /* 1MB: Use ring for larger messages */

/* Function pointer to the real MPI_Allreduce */
static int (*real_MPI_Allreduce)(const void *sendbuf, void *recvbuf, int count,
                                  MPI_Datatype datatype, MPI_Op op,
                                  MPI_Comm comm) = NULL;

/*
 * Get device ID for a pointer
 * Returns device ID or -1 if not device memory
 */
static int get_device_id(const void *ptr)
{
    if (ptr == NULL || ptr == MPI_IN_PLACE) return -1;
    
    struct cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    
    if (err != cudaSuccess) {
        cudaGetLastError();  /* Clear error */
        return -1;
    }
    
    if (attrs.type == cudaMemoryTypeDevice) {
        return attrs.device;
    }
    return -1;
}

/*
 * Check if a pointer is GPU device memory
 * Returns 1 if device memory, 0 otherwise
 */
int is_device_pointer(const void *ptr)
{
    return get_device_id(ptr) >= 0;
}

/*
 * Print GPU info and peer access status for each rank
 */
static void print_gpu_diagnostics(MPI_Comm comm)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    int device;
    cudaGetDevice(&device);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    fprintf(stderr, "[Rank %d] Using GPU %d: %s (UUID: ", rank, device, prop.name);
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, "%02x", (unsigned char)prop.uuid.bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) fprintf(stderr, "-");
    }
    fprintf(stderr, ")\n");
    
    /* Check peer access to all other GPUs */
    fprintf(stderr, "[Rank %d] Peer access from GPU %d:\n", rank, device);
    for (int peer = 0; peer < deviceCount; peer++) {
        if (peer == device) continue;
        
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, device, peer);
        
        /* Try to enable peer access if possible */
        if (canAccess) {
            cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
            if (err == cudaSuccess) {
                fprintf(stderr, "  -> GPU %d: CAN access, peer access ENABLED\n", peer);
            } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();  /* Clear error */
                fprintf(stderr, "  -> GPU %d: CAN access, peer access already enabled\n", peer);
            } else {
                cudaGetLastError();  /* Clear error */
                fprintf(stderr, "  -> GPU %d: CAN access, but enable failed: %s\n", 
                        peer, cudaGetErrorString(err));
            }
        } else {
            fprintf(stderr, "  -> GPU %d: CANNOT access (no NVLink/peer support)\n", peer);
        }
    }
    
    /* Gather all device IDs to check uniqueness */
    int *all_devices = (int *)malloc(nprocs * sizeof(int));
    MPI_Allgather(&device, 1, MPI_INT, all_devices, 1, MPI_INT, comm);
    
    if (rank == 0) {
        fprintf(stderr, "[Rank 0] GPU assignment across all ranks:\n");
        int duplicates = 0;
        for (int i = 0; i < nprocs; i++) {
            fprintf(stderr, "  Rank %d -> GPU %d\n", i, all_devices[i]);
            for (int j = 0; j < i; j++) {
                if (all_devices[i] == all_devices[j]) {
                    duplicates = 1;
                }
            }
        }
        if (duplicates) {
            fprintf(stderr, "  WARNING: Multiple ranks share the same GPU!\n");
        }
    }
    free(all_devices);
    
    MPI_Barrier(comm);
}

/* Initialize the real MPI_Allreduce function pointer */
static void init_real_allreduce(void)
{
    if (real_MPI_Allreduce == NULL) {
        real_MPI_Allreduce = dlsym(RTLD_NEXT, "MPI_Allreduce");
        if (real_MPI_Allreduce == NULL) {
            fprintf(stderr, "Error: Could not find real MPI_Allreduce: %s\n",
                    dlerror());
            exit(EXIT_FAILURE);
        }
    }
}

static void init_debug(void)
{
    if (debug == -1) {
        const char *env = getenv("CUSTOM_ALLREDUCE_DEBUG");
        debug = (env != NULL && atoi(env) > 0) ? 1 : 0;
    }
}

/*
 * Initialize algorithm selection from environment variable
 * CUSTOM_ALLREDUCE_ALGO: recursive (default)
 */
static void init_algo(void)
{
    /* Always recursive doubling in this version */
    selected_algo = ALGO_RECURSIVE;
    algo_initialized = 1;
}

/*
 * External CUDA reduction kernel (from gpu_reduce.cu)
 * op_type: 0=SUM, 1=PROD, 2=MAX, 3=MIN
 * dtype: 0=char, 1=int, 2=long, 3=float, 4=double, 5=long long,
 *        6=uchar, 7=uint, 8=ulong, 9=ulonglong, 10=short, 11=ushort
 */
extern int cuda_reduce_local_sync(const void *in, void *inout, size_t count,
                                   int dtype, int op_type);

/*
 * Map MPI datatype to our dtype enum
 */
static int mpi_type_to_dtype(MPI_Datatype datatype)
{
    if (datatype == MPI_CHAR || datatype == MPI_SIGNED_CHAR) return 0;
    if (datatype == MPI_INT) return 1;
    if (datatype == MPI_LONG) return 2;
    if (datatype == MPI_FLOAT) return 3;
    if (datatype == MPI_DOUBLE) return 4;
    if (datatype == MPI_LONG_LONG || datatype == MPI_LONG_LONG_INT) return 5;
    if (datatype == MPI_UNSIGNED_CHAR || datatype == MPI_BYTE) return 6;
    if (datatype == MPI_UNSIGNED) return 7;
    if (datatype == MPI_UNSIGNED_LONG) return 8;
    if (datatype == MPI_UNSIGNED_LONG_LONG) return 9;
    if (datatype == MPI_SHORT) return 10;
    if (datatype == MPI_UNSIGNED_SHORT) return 11;
    return -1;  /* Unsupported */
}

/*
 * Map MPI op to our op_type enum
 */
static int mpi_op_to_optype(MPI_Op op)
{
    if (op == MPI_SUM) return 0;
    if (op == MPI_PROD) return 1;
    if (op == MPI_MAX) return 2;
    if (op == MPI_MIN) return 3;
    return -1;  /* Unsupported */
}

/*
 * GPU-aware local reduction: inout = inout op in
 * 
 * For GPU buffers, uses CUDA kernels directly on device.
 * For unsupported types/ops or host buffers, falls back to MPI_Reduce_local.
 */
void gpu_reduce_local(void *in, void *inout, int count,
                      MPI_Datatype datatype, MPI_Op op,
                      int use_gpu, size_t bufsize)
{
    if (!use_gpu) {
        /* Host buffers - use MPI_Reduce_local directly */
        MPI_Reduce_local(in, inout, count, datatype, op);
        return;
    }
    
    /* Try CUDA kernel first */
    int dtype = mpi_type_to_dtype(datatype);
    int op_type = mpi_op_to_optype(op);
    
    if (dtype >= 0 && op_type >= 0) {
        int ret = cuda_reduce_local_sync(in, inout, (size_t)count, dtype, op_type);
        if (ret == 0) {
            return;  /* Success */
        }
    }
    
    /* Fallback: stage through host for unsupported types/ops */
    void *host_in = malloc(bufsize);
    void *host_inout = malloc(bufsize);
    
    if (!host_in || !host_inout) {
        fprintf(stderr, "Error: malloc failed in gpu_reduce_local\n");
        free(host_in);
        free(host_inout);
        return;
    }
    
    /* Copy GPU data to host */
    cudaMemcpy(host_in, in, bufsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_inout, inout, bufsize, cudaMemcpyDeviceToHost);
    
    /* Perform reduction on host */
    MPI_Reduce_local(host_in, host_inout, count, datatype, op);
    
    /* Copy result back to GPU */
    cudaMemcpy(inout, host_inout, bufsize, cudaMemcpyHostToDevice);
    
    free(host_in);
    free(host_inout);
}

/* 
 * Custom Allreduce implementation
 * 
 * Algorithm selection:
 * - recursive: Always use recursive doubling (latency-optimized)
 */
static int custom_allreduce(const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    init_debug();
    init_algo();
    
    /* Print GPU diagnostics once on first call */
    if (!initialized && debug) {
        print_gpu_diagnostics(comm);
        initialized = 1;
    }
    
    int type_size;
    MPI_Type_size(datatype, &type_size);
    size_t bufsize = (size_t)count * type_size;
    
    /* Determine which algorithm to use */
    int use_gpu = is_device_pointer(recvbuf);
    
    if (debug) {
        /* Get device IDs for buffers */
        int sendbuf_dev = get_device_id(sendbuf);
        int recvbuf_dev = get_device_id(recvbuf);
        
        fprintf(stderr, "[Rank %d] Allreduce: count=%d, bytes=%zu, algo=recursive_doubling, sendbuf on %s, recvbuf on %s\n",
                rank, count, bufsize,
                sendbuf == MPI_IN_PLACE ? "IN_PLACE" : (sendbuf_dev >= 0 ? "GPU" : "HOST"),
                recvbuf_dev >= 0 ? "GPU" : "HOST");
        
        if (sendbuf_dev >= 0) {
            fprintf(stderr, "[Rank %d]   sendbuf device: GPU %d\n", rank, sendbuf_dev);
        }
        if (recvbuf_dev >= 0) {
            fprintf(stderr, "[Rank %d]   recvbuf device: GPU %d\n", rank, recvbuf_dev);
        }
    }
    
    /* Execute selected algorithm */
    return recursive_doubling_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

/*
 * MPI_Allreduce interposition function
 * 
 * This function is called instead of the real MPI_Allreduce when the
 * library is preloaded using LD_PRELOAD.
 */
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    /* Ensure we have the real function pointer */
    init_real_allreduce();
    
    /* Call our custom implementation */
    return custom_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

/*
 * Also intercept PMPI_Allreduce for completeness
 * Some MPI implementations use PMPI prefix internally
 */
int PMPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    init_real_allreduce();
    return custom_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}
