# Copyright (c) 2026 Cornelis Networks. All rights reserved.

SUBDIRS = core coll gpu

lib_LTLIBRARIES = libcail.la

libcail_la_SOURCES =
libcail_la_LIBADD = \
	core/libcail_core.la \
	coll/allreduce/libcail_allreduce.la \
	gpu/libcail_gpu.la
libcail_la_LDFLAGS = -version-info 0:0:0 $(MPI_LIBS)

if HAVE_CUDA
libcail_la_LDFLAGS += $(CUDA_LIBS) -lstdc++
endif

include_HEADERS = core/cail.h

AM_CFLAGS = $(MPI_CFLAGS)
