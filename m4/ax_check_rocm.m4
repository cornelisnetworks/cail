dnl ---------------------------------------------------------------------------
dnl AX_CHECK_ROCM — Detect ROCm toolkit, hipcc, and set build variables
dnl ---------------------------------------------------------------------------
AC_DEFUN([AX_CHECK_ROCM], [
  dnl --with-rocm=PATH
  AS_IF([test "x$with_rocm" = "x"], [
    AC_ARG_WITH([rocm],
      [AS_HELP_STRING([--with-rocm=PATH],
        [Path to ROCm toolkit @<:@default=auto@:>@])],
      [with_rocm=$withval],
      [with_rocm=auto])
  ])

  have_rocm=no
  ROCM_HOME=
  HIPCC=
  ROCM_CFLAGS=
  ROCM_LIBS=
  HIPCCFLAGS=

  dnl Skip ROCm if --without-rocm
  AS_IF([test "x$with_rocm" != "xno"], [
    dnl Find hipcc
    AS_IF([test "x$with_rocm" != "xauto"], [
      HIPCC="$with_rocm/bin/hipcc"
      ROCM_HOME="$with_rocm"
    ], [
      AC_PATH_PROG([HIPCC], [hipcc], [])
      AS_IF([test "x$HIPCC" != "x"], [
        dnl Derive ROCM_HOME from hipcc location
        rocm_bin_dir=`AS_DIRNAME([$HIPCC])`
        ROCM_HOME=`AS_DIRNAME([$rocm_bin_dir])`
      ], [
        AS_IF([test -x "/opt/rocm/bin/hipcc"], [
          HIPCC="/opt/rocm/bin/hipcc"
          ROCM_HOME="/opt/rocm"
        ])
      ])
    ])

    dnl Check hipcc exists
    AS_IF([test "x$HIPCC" != "x" && test -x "$HIPCC"], [
      rocm_inc="$ROCM_HOME/include"
      dnl Prefer lib64 if that is where libamdhip64.so lives (some distros)
      AS_IF([test -f "$ROCM_HOME/lib64/libamdhip64.so"],
            [rocm_lib="$ROCM_HOME/lib64"],
            [rocm_lib="$ROCM_HOME/lib"])

      dnl Validate HIP runtime header and ROCm runtime library
      AC_CHECK_FILE([$rocm_inc/hip/hip_runtime.h], [
        AC_CHECK_FILE([$rocm_lib/libamdhip64.so], [
          have_rocm=yes
          ROCM_CFLAGS="-I$rocm_inc -D__HIP_PLATFORM_AMD__"
          ROCM_LIBS="-L$rocm_lib -lamdhip64"
          HIPCCFLAGS="-fPIC"
          AS_IF([test "x$rocm_arch" != "x"], [
            HIPCCFLAGS="$HIPCCFLAGS --offload-arch=$rocm_arch"
          ])
          AC_DEFINE([HAVE_ROCM], [1], [Define to 1 if ROCm is available])
          AC_MSG_NOTICE([ROCm found: $ROCM_HOME])
        ], [
          AS_IF([test "x$with_rocm" != "xauto"], [
            AC_MSG_ERROR([libamdhip64.so not found in $rocm_lib])
          ], [
            AC_MSG_NOTICE([libamdhip64.so not found in $rocm_lib — ROCm disabled])
          ])
        ])
      ], [
        AS_IF([test "x$with_rocm" != "xauto"], [
          AC_MSG_ERROR([hip_runtime.h not found in $rocm_inc/hip])
        ], [
          AC_MSG_NOTICE([hip_runtime.h not found in $rocm_inc/hip — ROCm disabled])
        ])
      ])
    ], [
      AS_IF([test "x$with_rocm" != "xauto"], [
        AC_MSG_ERROR([hipcc not found at $HIPCC])
      ], [
        AC_MSG_NOTICE([hipcc not found — ROCm disabled])
      ])
    ])
  ])

  AC_SUBST([ROCM_HOME])
  AC_SUBST([HIPCC])
  AC_SUBST([ROCM_CFLAGS])
  AC_SUBST([ROCM_LIBS])
  AC_SUBST([HIPCCFLAGS])

  AM_CONDITIONAL([HAVE_ROCM], [test "x$have_rocm" = "xyes"])
])
