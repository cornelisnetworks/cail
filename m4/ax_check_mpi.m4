dnl ---------------------------------------------------------------------------
dnl AX_CHECK_MPI — Detect MPI compiler wrapper and extract flags
dnl ---------------------------------------------------------------------------
AC_DEFUN([AX_CHECK_MPI], [
  AC_ARG_VAR([MPICC], [MPI C compiler wrapper])
  AC_ARG_VAR([MPI_CFLAGS], [MPI C compiler flags])
  AC_ARG_VAR([MPI_LIBS], [MPI linker flags])

  AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi=PATH],
      [Path to MPI installation @<:@default=auto@:>@])],
    [with_mpi=$withval],
    [with_mpi=auto])

  dnl Determine MPICC: explicit MPICC > --with-mpi > CC if it looks like mpicc > PATH
  AS_IF([test "x$MPICC" = "x"], [
    AS_IF([test "x$with_mpi" != "xauto" && test "x$with_mpi" != "xyes"], [
      MPICC="$with_mpi/bin/mpicc"
    ], [
      dnl Check if CC is already an mpicc wrapper
      cc_base=`basename "$CC" 2>/dev/null`
      AS_CASE([$cc_base],
        [mpicc*], [MPICC="$CC"],
        [
          AC_PATH_PROG([MPICC], [mpicc], [])
        ])
    ])
  ])

  AS_IF([test "x$MPICC" = "x"], [
    AC_MSG_ERROR([mpicc not found. Install MPI or set MPICC or --with-mpi=PATH.])
  ])

  dnl Get MPI flags from mpicc
  AS_IF([test "x$MPI_CFLAGS" = "x"], [
    MPI_CFLAGS=`$MPICC --showme:compile 2>/dev/null || $MPICC -show 2>/dev/null | sed 's/ -l[[^ ]]*//g; s/mpicc//'`
  ])
  AS_IF([test "x$MPI_LIBS" = "x"], [
    MPI_LIBS=`$MPICC --showme:link 2>/dev/null || $MPICC -show 2>/dev/null | sed 's/ -I[[^ ]]*//g; s/ -D[[^ ]]*//g; s/mpicc//'`
  ])

  AC_SUBST([MPICC])
  AC_SUBST([MPI_CFLAGS])
  AC_SUBST([MPI_LIBS])

  dnl Set CC to mpicc so that all compilation uses MPI wrapper
  CC="$MPICC"

  AC_MSG_NOTICE([MPI C compiler: $MPICC])
  AC_MSG_NOTICE([MPI CFLAGS: $MPI_CFLAGS])
  AC_MSG_NOTICE([MPI LIBS: $MPI_LIBS])
])
