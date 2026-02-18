dnl Copyright (c) 2026 Cornelis Networks. All rights reserved.

dnl ---------------------------------------------------------------------------
dnl cail — GPU-aware MPI_Allreduce interposition library
dnl ---------------------------------------------------------------------------

AC_PREREQ([2.69])
AC_INIT([cail], [0.1.0], [bug-cail@cornelisnetworks.com])
AM_INIT_AUTOMAKE([foreign subdir-objects -Wall -Werror])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_HEADERS([config.h])

dnl ---------------------------------------------------------------------------
dnl Libtool
dnl ---------------------------------------------------------------------------
AM_PROG_AR
LT_INIT([shared static])

dnl ---------------------------------------------------------------------------
dnl Compilers
dnl ---------------------------------------------------------------------------
AC_PROG_CC
AC_PROG_CXX

dnl ---------------------------------------------------------------------------
dnl MPI (macro provided by m4/ax_check_mpi.m4 — Task 2)
dnl ---------------------------------------------------------------------------
AX_CHECK_MPI

dnl ---------------------------------------------------------------------------
dnl --enable-host-path : compile host-only path, skip CUDA requirement
dnl ---------------------------------------------------------------------------
AC_ARG_ENABLE([host-path],
  [AS_HELP_STRING([--enable-host-path],
    [Build host-only path without GPU support @<:@default=no@:>@])],
  [enable_host_path=$enableval],
  [enable_host_path=no])

AM_CONDITIONAL([CAIL_HOST_PATH], [test "x$enable_host_path" = "xyes"])

AS_IF([test "x$enable_host_path" = "xyes"], [
  AC_DEFINE([CAIL_HOST_PATH], [1],
    [Define to 1 to build host-only path without GPU support])
])

dnl ---------------------------------------------------------------------------
dnl --enable-debug : debug build flags
dnl ---------------------------------------------------------------------------
AC_ARG_ENABLE([debug],
  [AS_HELP_STRING([--enable-debug],
    [Enable debug build with -g -O0 @<:@default=no@:>@])],
  [enable_debug=$enableval],
  [enable_debug=no])

AS_IF([test "x$enable_debug" = "xyes"], [
  AC_DEFINE([CAIL_DEBUG_BUILD], [1],
    [Define to 1 for debug builds])
  CFLAGS="$CFLAGS -g -O0"
  CXXFLAGS="$CXXFLAGS -g -O0"
])

dnl ---------------------------------------------------------------------------
dnl Algorithm selection
dnl ---------------------------------------------------------------------------
AC_ARG_ENABLE([recursive-doubling],
  [AS_HELP_STRING([--enable-recursive-doubling],
    [Enable recursive-doubling allreduce algorithm @<:@default=yes@:>@])],
  [enable_recursive_doubling=$enableval],
  [enable_recursive_doubling=yes])

AM_CONDITIONAL([CAIL_ENABLE_RECURSIVE_DOUBLING],
  [test "x$enable_recursive_doubling" = "xyes"])
AS_IF([test "x$enable_recursive_doubling" = "xyes"], [
  AC_DEFINE([CAIL_ENABLE_RECURSIVE_DOUBLING], [1],
    [Define to 1 to enable recursive-doubling algorithm])
])

AC_ARG_ENABLE([ring],
  [AS_HELP_STRING([--enable-ring],
    [Enable ring allreduce algorithm @<:@default=yes@:>@])],
  [enable_ring=$enableval],
  [enable_ring=yes])

AM_CONDITIONAL([CAIL_ENABLE_RING],
  [test "x$enable_ring" = "xyes"])
AS_IF([test "x$enable_ring" = "xyes"], [
  AC_DEFINE([CAIL_ENABLE_RING], [1],
    [Define to 1 to enable ring algorithm])
])

AC_ARG_ENABLE([rabenseifner],
  [AS_HELP_STRING([--enable-rabenseifner],
    [Enable Rabenseifner allreduce algorithm @<:@default=yes@:>@])],
  [enable_rabenseifner=$enableval],
  [enable_rabenseifner=yes])

AM_CONDITIONAL([CAIL_ENABLE_RABENSEIFNER],
  [test "x$enable_rabenseifner" = "xyes"])
AS_IF([test "x$enable_rabenseifner" = "xyes"], [
  AC_DEFINE([CAIL_ENABLE_RABENSEIFNER], [1],
    [Define to 1 to enable Rabenseifner algorithm])
])

AC_ARG_ENABLE([tree],
  [AS_HELP_STRING([--enable-tree],
    [Enable tree allreduce algorithm @<:@default=yes@:>@])],
  [enable_tree=$enableval],
  [enable_tree=yes])

AM_CONDITIONAL([CAIL_ENABLE_TREE],
  [test "x$enable_tree" = "xyes"])
AS_IF([test "x$enable_tree" = "xyes"], [
  AC_DEFINE([CAIL_ENABLE_TREE], [1],
    [Define to 1 to enable tree algorithm])
])

dnl ---------------------------------------------------------------------------
dnl --with-cuda-arch : NVCC architecture flag
dnl ---------------------------------------------------------------------------
AC_ARG_WITH([cuda-arch],
  [AS_HELP_STRING([--with-cuda-arch=SM],
    [Set CUDA architecture for nvcc, e.g. sm_70 @<:@default=sm_70@:>@])],
  [cuda_arch=$withval],
  [cuda_arch=sm_70])

AC_SUBST([CUDA_ARCH], [$cuda_arch])

dnl ---------------------------------------------------------------------------
dnl CUDA (macro provided by m4/ax_check_cuda.m4)
dnl Must come after --with-cuda-arch so $cuda_arch is set for NVCCFLAGS
dnl ---------------------------------------------------------------------------
AX_CHECK_CUDA

dnl ---------------------------------------------------------------------------
dnl Output
dnl ---------------------------------------------------------------------------
AC_CONFIG_FILES([
  Makefile
  src/Makefile
  src/core/Makefile
  src/coll/Makefile
  src/coll/allreduce/Makefile
  src/gpu/Makefile
  src/gpu/cuda/Makefile
  src/gpu/rocm/Makefile
  tests/Makefile
  cail.pc
])

AC_OUTPUT
