# CAIL Build and Packaging

**Status**: Implemented (v0.1.0)
**Validated**: 2026-06-12 against `main` @ `91c5e4f`
**Source**: `configure.ac`, `Makefile.am`, `src/**/Makefile.am`, `cuda_lt.sh`, `m4/`

## 1. Toolchain

Autotools: autoconf ≥ 2.69, automake (`foreign subdir-objects -Wall
-Werror`), libtool (`LT_INIT([shared static])`). Bootstrap with
`./autogen.sh`, then the standard `configure && make && make install`.

```sh
./autogen.sh
mkdir build && cd build
../configure --with-cuda=/path/to/cuda --with-cuda-arch=sm_90
make -j$(nproc) && make install
```

## 2. Configure Options

| Option | Effect | Default |
|--------|--------|---------|
| `--with-mpi=PATH` | MPI detection via `m4/ax_check_mpi.m4` (probes compiler wrappers, extracts `MPI_CFLAGS`/`MPI_LIBS`) | auto |
| `--with-cuda=PATH` | CUDA toolkit root; probes `nvcc` + `cuda_runtime.h` (`m4/ax_check_cuda.m4`); sets `HAVE_CUDA` | auto |
| `--with-cuda-arch=SM` | Adds `-arch=$SM` to `NVCCFLAGS` | `sm_70` |
| `--enable-host-path` | `CAIL_HOST_PATH` — host-only backend, CUDA not required | no |
| `--enable-debug` | Appends `-g -O0` to `CFLAGS`/`CXXFLAGS` | no |
| `--enable-recursive-doubling` / `--enable-ring` / `--enable-rabenseifner` | `CAIL_ENABLE_*` macros gating each algorithm (see `interception-and-dispatch.md` §2–3) | all yes |

## 3. Layout and Artifacts

Convenience libraries per layer, assembled bottom-up:

```
src/core/           → libcail_core.la
src/coll/allreduce/ → libcail_allreduce.la     (all 4 .c files always compiled;
                                                gating is via CAIL_ENABLE_* macros)
src/gpu/            → libcail_gpu.la           (links exactly one backend:
  src/gpu/cuda/       → libcail_cuda.la          host XOR cuda; rocm never linked)
  src/gpu/host/       → libcail_host.la
  src/gpu/rocm/       → libcail_rocm.la          (built but orphaned — see gpu-layer.md §4)
src/                → libcail.la               (-version-info 0:0:0 → libcail.so.0.0.0)
```

pkg-config (`cail.pc.in`): `Libs: -L${libdir} -lcail`,
`Cflags: -I${includedir}`, version from `PACKAGE_VERSION` (0.1.0).

## 4. GPU Backend Selection Wiring

`src/gpu/Makefile.am`:

```make
if CAIL_HOST_PATH
libcail_gpu_la_LIBADD += host/libcail_host.la
else
if HAVE_CUDA
libcail_gpu_la_LIBADD += cuda/libcail_cuda.la
endif
endif
```

**KNOWN GAP — silent link failure**: with `--with-cuda=auto` (default), a
missing CUDA toolkit only emits a configure *warning* and leaves both
conditionals false. `libcail_gpu_la_LIBADD` ends up empty and the build dies
at link time with unresolved `cail_gpu_*` symbols. Correct behavior would be
`AC_MSG_ERROR` at configure time ("no GPU backend: install CUDA or pass
--enable-host-path"). Fold the fix into the ROCm enablement work
(`gpu-layer.md` §4 step 1), which has to restructure this block anyway.

## 5. Compiling `.cu` Under Libtool (`cuda_lt.sh`)

Libtool has no NVCC tag, so `.cu` files can't natively become `.lo` objects.
`cuda_lt.sh` bridges this; `src/gpu/cuda/Makefile.am` rule:

```make
/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(NVCC) $(NVCCFLAGS) ... -c $<
```

The script:

1. Compiles a PIC object under `.libs/` (`-Xcompiler -fPIC`).
2. Compiles a non-PIC object in place.
3. Hand-writes the `.lo` libtool metadata file pointing at both.

This is the same pattern used by other CUDA-autotools projects (e.g.
UCX/OSU). **Note**: it always builds both PIC and non-PIC objects regardless
of `--disable-static`, doubling `.cu` compile time. Minor inefficiency, not
a defect.

The GPU test programs (`tests/*.cu`) use the same shim; see `testing.md`.

## 6. Workspace Integration Notes (opx-ompi-osu-rocm)

CAIL is not currently wired into the workspace provisioner as a dep
(`load-dep.sh --list` does not know it). Manual build against the workspace
stack:

```sh
cd src/cail && ./autogen.sh && mkdir -p build && cd build
../configure --prefix=$WS_INSTALL_PREFIX \
             --with-mpi=$WS_INSTALL_PREFIX \
             --enable-host-path        # until ROCm backend exists
make -j && make install
```

Per workspace filesystem rules, if built with a different toolchain than the
provisioned stack, install into a named subdirectory of the install prefix
instead (e.g. `install/cail/`). Once a ROCm backend lands, a proper
`workspace/deps/cail/` definition (dep.conf + build.sh consuming
`{{ROCM_PATH}}` / `{{ROCM_GPU_TARGETS}}`) should be added to cn-ai-tools so
the `opx-ompi-osu-rocm` profile can build it natively.
