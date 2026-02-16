# Makefile for Custom MPI_Allreduce Interposition Library (GPU version)
#
# Compiles a shared library that can be used with LD_PRELOAD to intercept
# MPI_Allreduce calls. Uses CUDA for GPU memory operations and reduction.

# MPI compiler wrapper
MPICC ?= mpicc

# CUDA path and compiler
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

# Compiler flags
CFLAGS = -Wall -Wextra -fPIC -O2 -I$(CUDA_HOME)/include
NVCCFLAGS = -Xcompiler -fPIC -O2
LDFLAGS = -shared -ldl -L$(CUDA_HOME)/lib64 -lcudart -lstdc++

# Target library
TARGET = libcustom_allreduce.so

# Source files
SRC = mpi_allreduce_wrapper.c recursive_doubling.c
CUDA_SRC = gpu_reduce.cu
OBJ = mpi_allreduce_wrapper.o recursive_doubling.o gpu_reduce.o

# Build the shared library
all: $(TARGET)

$(TARGET): $(OBJ)
	$(MPICC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

mpi_allreduce_wrapper.o: mpi_allreduce_wrapper.c common.h recursive_doubling.h
	$(MPICC) $(CFLAGS) -c -o $@ $<

recursive_doubling.o: recursive_doubling.c recursive_doubling.h common.h
	$(MPICC) $(CFLAGS) -c -o $@ $<

gpu_reduce.o: gpu_reduce.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up
clean:
	rm -f $(TARGET) $(OBJ)

# Install (optional - copies to a specified directory)
PREFIX ?= /usr/local/lib
install: $(TARGET)
	install -d $(PREFIX)
	install -m 755 $(TARGET) $(PREFIX)

.PHONY: all clean install
