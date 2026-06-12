dnl ---------------------------------------------------------------------------
dnl AX_CHECK_CUDA — Detect CUDA toolkit, nvcc, and set build variables
dnl ---------------------------------------------------------------------------
AC_DEFUN([AX_CHECK_CUDA], [
  dnl --with-cuda=PATH
  AS_IF([test "x$with_cuda" = "x"], [
    AC_ARG_WITH([cuda],
      [AS_HELP_STRING([--with-cuda=PATH],
        [Path to CUDA toolkit @<:@default=auto@:>@])],
      [with_cuda=$withval],
      [with_cuda=auto])
  ])

  have_cuda=no

  dnl Skip CUDA if --without-cuda
  AS_IF([test "x$with_cuda" != "xno"], [
    dnl Find nvcc
    AS_IF([test "x$with_cuda" != "xauto"], [
      NVCC="$with_cuda/bin/nvcc"
      cuda_inc="$with_cuda/include"
      cuda_lib="$with_cuda/lib64"
      CUDA_HOME="$with_cuda"
    ], [
      AC_PATH_PROG([NVCC], [nvcc], [])
      AS_IF([test "x$NVCC" != "x"], [
        dnl Derive CUDA_HOME from nvcc location
        cuda_bin_dir=`AS_DIRNAME([$NVCC])`
        CUDA_HOME=`AS_DIRNAME([$cuda_bin_dir])`
        cuda_inc="$CUDA_HOME/include"
        cuda_lib="$CUDA_HOME/lib64"
      ])
    ])

    dnl Check nvcc exists
    AS_IF([test "x$NVCC" != "x" && test -x "$NVCC"], [
      dnl Check cuda_runtime.h
      AC_CHECK_FILE([$cuda_inc/cuda_runtime.h], [
        have_cuda=yes
        CUDA_CFLAGS="-I$cuda_inc"
        CUDA_LIBS="-L$cuda_lib -lcudart"
        NVCCFLAGS="-arch=$cuda_arch -Xcompiler -fPIC"
        AC_DEFINE([HAVE_CUDA], [1], [Define to 1 if CUDA is available])
        AC_MSG_NOTICE([CUDA found: $CUDA_HOME])
      ], [
        AC_MSG_WARN([cuda_runtime.h not found in $cuda_inc — disabling CUDA])
      ])
    ], [
      AS_IF([test "x$with_cuda" != "xauto"], [
        AC_MSG_ERROR([nvcc not found at $NVCC])
      ], [
        AC_MSG_NOTICE([nvcc not found — CUDA disabled])
      ])
    ])
  ])

  AC_SUBST([CUDA_HOME])
  AC_SUBST([NVCC])
  AC_SUBST([CUDA_CFLAGS])
  AC_SUBST([CUDA_LIBS])
  AC_SUBST([NVCCFLAGS])

  AM_CONDITIONAL([HAVE_CUDA], [test "x$have_cuda" = "xyes"])
])
