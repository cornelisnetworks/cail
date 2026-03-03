# Copyright (c) 2026 Cornelis Networks. All rights reserved.

noinst_LTLIBRARIES = libcail_core.la

libcail_core_la_SOURCES = \
    cail_buf.c \
    cail_pmpi.c \
    cail_init.c \
    cail_types.c

noinst_HEADERS = cail_internal.h cail_types.h

AM_CFLAGS = $(MPI_CFLAGS) -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu
