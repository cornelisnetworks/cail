# Copyright (c) 2026 Cornelis Networks. All rights reserved.

check_PROGRAMS = \
    test_allreduce_basic \
    test_allreduce_inplace \
    test_allreduce_datatypes \
    test_allreduce_algorithms \
    test_allreduce_nonpof2 \
    test_allreduce_edge

noinst_PROGRAMS = bench_allreduce

TESTS = $(check_PROGRAMS)
LOG_COMPILER = $(srcdir)/test_runner.sh

test_allreduce_basic_SOURCES      = test_allreduce_basic.c
test_allreduce_inplace_SOURCES    = test_allreduce_inplace.c
test_allreduce_datatypes_SOURCES  = test_allreduce_datatypes.c
test_allreduce_algorithms_SOURCES = test_allreduce_algorithms.c
test_allreduce_nonpof2_SOURCES    = test_allreduce_nonpof2.c
test_allreduce_edge_SOURCES       = test_allreduce_edge.c
bench_allreduce_SOURCES           = bench_allreduce.c

AM_CFLAGS = $(MPI_CFLAGS) -I$(top_srcdir)/src/core
LDADD = $(top_builddir)/src/libcail.la $(MPI_LIBS) -lm

if HAVE_CUDA

SUFFIXES = .cu

noinst_PROGRAMS += test_allreduce_gpu bench_allreduce_gpu

test_allreduce_gpu_SOURCES =
test_allreduce_gpu_LDADD = test_allreduce_gpu.o $(top_builddir)/src/libcail.la $(MPI_LIBS) $(CUDA_LIBS) -lstdc++ -lm
test_allreduce_gpu_DEPENDENCIES = test_allreduce_gpu.o

bench_allreduce_gpu_SOURCES =
bench_allreduce_gpu_LDADD = bench_allreduce_gpu.o $(top_builddir)/src/libcail.la $(MPI_LIBS) $(CUDA_LIBS) -lstdc++ -lm
bench_allreduce_gpu_DEPENDENCIES = bench_allreduce_gpu.o


test_allreduce_gpu.o: $(srcdir)/test_allreduce_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(MPI_CFLAGS) -I$(top_srcdir)/src/core -c -o $@ $<

bench_allreduce_gpu.o: $(srcdir)/bench_allreduce_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(MPI_CFLAGS) -I$(top_srcdir)/src/core -c -o $@ $<

CLEANFILES = test_allreduce_gpu.o bench_allreduce_gpu.o

endif

EXTRA_DIST = test_runner.sh test_allreduce_gpu.cu bench_allreduce_gpu.cu
