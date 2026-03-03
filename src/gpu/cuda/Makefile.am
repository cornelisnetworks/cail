# Copyright (c) 2026 Cornelis Networks. All rights reserved.

if HAVE_CUDA

noinst_LTLIBRARIES = libcail_cuda.la

libcail_cuda_la_SOURCES = cail_cuda_mem.c cail_cuda_reduce.cu

SUFFIXES = .cu

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ \
	    $(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(MPI_CFLAGS) \
	    -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu -c $<

AM_CFLAGS = $(CUDA_CFLAGS) $(MPI_CFLAGS) -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu

CLEANFILES = cail_cuda_reduce.lo

else

noinst_LTLIBRARIES = libcail_cuda.la
libcail_cuda_la_SOURCES = cail_cuda_stub.c

endif
