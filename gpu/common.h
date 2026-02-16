/*
 * Common declarations for custom allreduce implementation
 */

#ifndef CUSTOM_ALLREDUCE_COMMON_H
#define CUSTOM_ALLREDUCE_COMMON_H

#include <mpi.h>
#include <stddef.h>

/* GPU detection */
int is_device_pointer(const void *ptr);

/* GPU-aware local reduction */
void gpu_reduce_local(void *in, void *inout, int count,
                     MPI_Datatype datatype, MPI_Op op,
                     int use_gpu, size_t bufsize);

#endif /* CUSTOM_ALLREDUCE_COMMON_H */
