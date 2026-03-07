# Copyright (c) 2026 Cornelis Networks. All rights reserved.

if CAIL_HOST_PATH

noinst_LTLIBRARIES = libcail_host.la

libcail_host_la_SOURCES = cail_host_reduce.c

AM_CFLAGS = $(MPI_CFLAGS) -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu

else

noinst_LTLIBRARIES = libcail_host.la
libcail_host_la_SOURCES =

endif
