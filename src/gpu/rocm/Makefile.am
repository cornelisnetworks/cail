# Copyright (c) 2026 Cornelis Networks. All rights reserved.

if HAVE_ROCM

noinst_LTLIBRARIES = libcail_rocm.la

libcail_rocm_la_SOURCES = cail_rocm_mem.c cail_rocm_reduce.hip

SUFFIXES = .hip

.hip.lo:
	/bin/bash $(top_srcdir)/hip_lt.sh "$(LIBTOOL)" $@ \
	    $(HIPCC) $(HIPCCFLAGS) $(ROCM_CFLAGS) $(MPI_CFLAGS) \
	    -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu -c $<

AM_CFLAGS = $(ROCM_CFLAGS) $(MPI_CFLAGS) -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu

CLEANFILES = cail_rocm_reduce.lo

else

noinst_LTLIBRARIES = libcail_rocm.la
libcail_rocm_la_SOURCES = cail_rocm_stub.c
AM_CFLAGS = -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu

endif
