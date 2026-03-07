/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#ifndef CAIL_TYPES_H
#define CAIL_TYPES_H

#include "cail.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Values match GPU kernel dispatch table in gpu/cuda/cail_cuda_reduce.cu */
typedef enum {
    CAIL_CHAR       = 0,
    CAIL_INT        = 1,
    CAIL_LONG       = 2,
    CAIL_FLOAT      = 3,
    CAIL_DOUBLE     = 4,
    CAIL_LONG_LONG  = 5,
    CAIL_UCHAR      = 6,
    CAIL_UINT       = 7,
    CAIL_ULONG      = 8,
    CAIL_ULONGLONG  = 9,
    CAIL_SHORT      = 10,
    CAIL_USHORT     = 11,
    CAIL_NUM_DTYPES,      /* must be last valid+1; used for kernel table dimensions */
    CAIL_INVALID    = -1
} cail_datatype_t;

typedef enum {
    CAIL_SUM        = 0,
    CAIL_PROD       = 1,
    CAIL_MAX        = 2,
    CAIL_MIN        = 3,
    CAIL_NUM_OPS,         /* must be last valid+1; used for kernel table dimensions */
    CAIL_OP_INVALID = -1
} cail_op_t;

typedef enum {
    CAIL_ALGO_AUTO                = -1,
    CAIL_ALGO_RECURSIVE_DOUBLING  = 0,
    CAIL_ALGO_RING                = 1,
    CAIL_ALGO_RABENSEIFNER        = 2
} cail_algo_t;

cail_datatype_t cail_mpi_type_to_dtype(MPI_Datatype dt);
cail_op_t       cail_mpi_op_to_optype(MPI_Op op);
MPI_Datatype cail_dtype_to_mpi_type(cail_datatype_t dtype);
MPI_Op       cail_optype_to_mpi_op(cail_op_t op);

int cail_type_supported(MPI_Datatype dt);
int cail_op_supported(MPI_Op op);

int cail_type_size(cail_datatype_t dt);

const char *cail_mpi_type_name(MPI_Datatype dt);
const char *cail_mpi_op_name(MPI_Op op);
const char *cail_dtype_name(cail_datatype_t dt);
const char *cail_op_name(cail_op_t op);

#ifdef __cplusplus
}
#endif

#endif /* CAIL_TYPES_H */
