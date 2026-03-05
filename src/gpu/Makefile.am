# Copyright (c) 2026 Cornelis Networks. All rights reserved.

SUBDIRS = cuda rocm host

noinst_LTLIBRARIES = libcail_gpu.la

libcail_gpu_la_SOURCES = cail_gpu_stub.c
libcail_gpu_la_LIBADD =
noinst_HEADERS = cail_gpu.h

if CAIL_HOST_PATH
libcail_gpu_la_LIBADD += host/libcail_host.la
else
if HAVE_CUDA
libcail_gpu_la_LIBADD += cuda/libcail_cuda.la
endif
endif

AM_CFLAGS = $(MPI_CFLAGS) -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu
