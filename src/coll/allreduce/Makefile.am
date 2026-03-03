# Copyright (c) 2026 Cornelis Networks. All rights reserved.

noinst_LTLIBRARIES = libcail_allreduce.la

libcail_allreduce_la_SOURCES = \
	cail_allreduce.c \
	cail_allreduce_recursive_doubling.c \
	cail_allreduce_tree.c \
	cail_allreduce_rabenseifner.c \
	cail_allreduce_ring.c

noinst_HEADERS = cail_allreduce_impl.h

AM_CFLAGS = $(MPI_CFLAGS) -I$(top_srcdir)/src/core -I$(top_srcdir)/src/gpu
