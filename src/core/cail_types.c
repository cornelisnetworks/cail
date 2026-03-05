/* Copyright (c) 2026 Cornelis Networks. All rights reserved. */

#include "cail_types.h"

cail_datatype_t cail_mpi_type_to_dtype(MPI_Datatype dt)
{
    if (dt == MPI_CHAR || dt == MPI_SIGNED_CHAR) return CAIL_CHAR;
    if (dt == MPI_INT)                           return CAIL_INT;
    if (dt == MPI_LONG)                          return CAIL_LONG;
    if (dt == MPI_FLOAT)                         return CAIL_FLOAT;
    if (dt == MPI_DOUBLE)                        return CAIL_DOUBLE;
    if (dt == MPI_LONG_LONG || dt == MPI_LONG_LONG_INT) return CAIL_LONG_LONG;
    if (dt == MPI_UNSIGNED_CHAR || dt == MPI_BYTE)      return CAIL_UCHAR;
    if (dt == MPI_UNSIGNED)                      return CAIL_UINT;
    if (dt == MPI_UNSIGNED_LONG)                 return CAIL_ULONG;
    if (dt == MPI_UNSIGNED_LONG_LONG)            return CAIL_ULONGLONG;
    if (dt == MPI_SHORT)                         return CAIL_SHORT;
    if (dt == MPI_UNSIGNED_SHORT)                return CAIL_USHORT;

    /* C99 fixed-width type aliases */
    if (dt == MPI_INT8_T)                       return CAIL_CHAR;
    if (dt == MPI_UINT8_T)                      return CAIL_UCHAR;
    if (dt == MPI_INT16_T)                      return CAIL_SHORT;
    if (dt == MPI_UINT16_T)                     return CAIL_USHORT;
    if (dt == MPI_INT32_T)                      return CAIL_INT;
    if (dt == MPI_UINT32_T)                     return CAIL_UINT;
    if (dt == MPI_INT64_T)                      return CAIL_LONG_LONG;
    if (dt == MPI_UINT64_T)                     return CAIL_ULONGLONG;

    return CAIL_INVALID;
}

cail_op_t cail_mpi_op_to_optype(MPI_Op op)
{
    if (op == MPI_SUM)  return CAIL_SUM;
    if (op == MPI_PROD) return CAIL_PROD;
    if (op == MPI_MAX)  return CAIL_MAX;
    if (op == MPI_MIN)  return CAIL_MIN;
    return CAIL_OP_INVALID;
}

MPI_Datatype cail_dtype_to_mpi_type(cail_datatype_t dtype)
{
    switch (dtype) {
    case CAIL_CHAR:      return MPI_CHAR;
    case CAIL_SHORT:     return MPI_SHORT;
    case CAIL_INT:       return MPI_INT;
    case CAIL_LONG:      return MPI_LONG;
    case CAIL_LONG_LONG: return MPI_LONG_LONG;
    case CAIL_FLOAT:     return MPI_FLOAT;
    case CAIL_DOUBLE:    return MPI_DOUBLE;
    case CAIL_UCHAR:     return MPI_UNSIGNED_CHAR;
    case CAIL_USHORT:    return MPI_UNSIGNED_SHORT;
    case CAIL_UINT:      return MPI_UNSIGNED;
    case CAIL_ULONG:     return MPI_UNSIGNED_LONG;
    case CAIL_ULONGLONG: return MPI_UNSIGNED_LONG_LONG;
    default:              return MPI_DATATYPE_NULL;
    }
}

MPI_Op cail_optype_to_mpi_op(cail_op_t op)
{
    switch (op) {
    case CAIL_SUM:  return MPI_SUM;
    case CAIL_PROD: return MPI_PROD;
    case CAIL_MAX:  return MPI_MAX;
    case CAIL_MIN:  return MPI_MIN;
    default:         return MPI_OP_NULL;
    }
}

const char *cail_dtype_name(cail_datatype_t dt)
{
    switch (dt) {
    case CAIL_CHAR:      return "CHAR";
    case CAIL_INT:       return "INT";
    case CAIL_LONG:      return "LONG";
    case CAIL_FLOAT:     return "FLOAT";
    case CAIL_DOUBLE:    return "DOUBLE";
    case CAIL_LONG_LONG: return "LONG_LONG";
    case CAIL_UCHAR:     return "UCHAR";
    case CAIL_UINT:      return "UINT";
    case CAIL_ULONG:     return "ULONG";
    case CAIL_ULONGLONG: return "ULONGLONG";
    case CAIL_SHORT:     return "SHORT";
    case CAIL_USHORT:    return "USHORT";
    default:              return "UNKNOWN";
    }
}

const char *cail_op_name(cail_op_t op)
{
    switch (op) {
    case CAIL_SUM:  return "SUM";
    case CAIL_PROD: return "PROD";
    case CAIL_MAX:  return "MAX";
    case CAIL_MIN:  return "MIN";
    default:         return "UNKNOWN";
    }
}

int cail_type_supported(MPI_Datatype dt)
{
    return cail_mpi_type_to_dtype(dt) != CAIL_INVALID;
}

int cail_op_supported(MPI_Op op)
{
    return cail_mpi_op_to_optype(op) != CAIL_OP_INVALID;
}

const char *cail_mpi_type_name(MPI_Datatype dt)
{
    if (dt == MPI_CHAR || dt == MPI_SIGNED_CHAR) return "MPI_CHAR";
    if (dt == MPI_INT)                           return "MPI_INT";
    if (dt == MPI_LONG)                          return "MPI_LONG";
    if (dt == MPI_FLOAT)                         return "MPI_FLOAT";
    if (dt == MPI_DOUBLE)                        return "MPI_DOUBLE";
    if (dt == MPI_LONG_LONG || dt == MPI_LONG_LONG_INT) return "MPI_LONG_LONG";
    if (dt == MPI_UNSIGNED_CHAR || dt == MPI_BYTE)      return "MPI_UNSIGNED_CHAR";
    if (dt == MPI_UNSIGNED)                      return "MPI_UNSIGNED";
    if (dt == MPI_UNSIGNED_LONG)                 return "MPI_UNSIGNED_LONG";
    if (dt == MPI_UNSIGNED_LONG_LONG)            return "MPI_UNSIGNED_LONG_LONG";
    if (dt == MPI_SHORT)                         return "MPI_SHORT";
    if (dt == MPI_UNSIGNED_SHORT)                return "MPI_UNSIGNED_SHORT";
    if (dt == MPI_INT8_T)                        return "MPI_INT8_T";
    if (dt == MPI_UINT8_T)                       return "MPI_UINT8_T";
    if (dt == MPI_INT16_T)                       return "MPI_INT16_T";
    if (dt == MPI_UINT16_T)                      return "MPI_UINT16_T";
    if (dt == MPI_INT32_T)                       return "MPI_INT32_T";
    if (dt == MPI_UINT32_T)                      return "MPI_UINT32_T";
    if (dt == MPI_INT64_T)                       return "MPI_INT64_T";
    if (dt == MPI_UINT64_T)                      return "MPI_UINT64_T";
    return "UNKNOWN";
}

const char *cail_mpi_op_name(MPI_Op op)
{
    if (op == MPI_SUM)  return "MPI_SUM";
    if (op == MPI_PROD) return "MPI_PROD";
    if (op == MPI_MAX)  return "MPI_MAX";
    if (op == MPI_MIN)  return "MPI_MIN";
    return "UNKNOWN";
}

int cail_type_size(cail_datatype_t dt)
{
    switch (dt) {
    case CAIL_CHAR:      return (int)sizeof(char);
    case CAIL_UCHAR:     return (int)sizeof(unsigned char);
    case CAIL_SHORT:     return (int)sizeof(short);
    case CAIL_USHORT:    return (int)sizeof(unsigned short);
    case CAIL_INT:       return (int)sizeof(int);
    case CAIL_UINT:      return (int)sizeof(unsigned int);
    case CAIL_LONG:      return (int)sizeof(long);
    case CAIL_ULONG:     return (int)sizeof(unsigned long);
    case CAIL_LONG_LONG: return (int)sizeof(long long);
    case CAIL_ULONGLONG: return (int)sizeof(unsigned long long);
    case CAIL_FLOAT:     return (int)sizeof(float);
    case CAIL_DOUBLE:    return (int)sizeof(double);
    case CAIL_INVALID:   return 0;
    }
    return 0;
}
