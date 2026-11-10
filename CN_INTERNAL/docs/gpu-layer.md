# CAIL GPU Abstraction Layer

**Status**: CUDA implemented; host path implemented; **ROCm stub only**
**Validated**: 2026-06-12 against `main` @ `91c5e4f`
**Source**: `src/gpu/cail_gpu.h`, `src/gpu/{cuda,host,rocm}/`

## 1. API Surface (`cail_gpu.h`)

A flat C API (wrapped `extern "C"`); exactly one backend provides these
symbols at link time:

| Function | Contract |
|----------|----------|
| `cail_gpu_init(void)` | Create stream + kernel dispatch table. 0/-1 |
| `cail_gpu_finalize(void)` | Destroy stream, release resources. Returns `void` |
| `cail_gpu_is_device_pointer(const void *ptr)` | 1 if device or managed memory, else 0 |
| `cail_gpu_malloc(void **ptr, size_t size)` | Device alloc. 0/-1 |
| `cail_gpu_free(void *ptr)` | Device free. 0/-1 (host backend: always 0) |
| `cail_gpu_memcpy(void *dst, const void *src, size_t size)` | Direction-agnostic copy (UVA). 0/-1 (host backend: always 0) |
| `cail_gpu_reduce_local(const void *in, void *inout, size_t count, cail_datatype_t, cail_op_t)` | `inout[i] = op(inout[i], in[i])`. Synchronous. 0/-1 |
| `cail_gpu_synchronize(void)` | Drain the internal stream. 0/-1 |
| `cail_gpu_flush_recv_buf(const void *recv_buf, size_t recv_bytes)` | PCIe read fence over a receive buffer (see §2). 0 on success or when not needed |

**Backend selection is compile-time only** — there is no runtime vtable or
dlopen. `src/gpu/Makefile.am` links exactly one convenience library into
`libcail_gpu.la`:

```make
if CAIL_HOST_PATH
libcail_gpu_la_LIBADD += host/libcail_host.la
else
if HAVE_CUDA
libcail_gpu_la_LIBADD += cuda/libcail_cuda.la
endif
endif
```

**KNOWN GAP**: if neither `CAIL_HOST_PATH` nor `HAVE_CUDA` is set (CUDA
auto-detect fails and `--enable-host-path` was not given), `LIBADD` stays
empty and the build fails at link with unresolved `cail_gpu_*` symbols
instead of a clear configure-time error. See `build-and-packaging.md` §4.

## 2. CUDA Backend (`src/gpu/cuda/`)

### Memory ops (`cail_cuda_mem.c`)

- `cail_gpu_is_device_pointer`: `cudaPointerGetAttributes`; returns 1 for
  `cudaMemoryTypeDevice` or `cudaMemoryTypeManaged`. On API error (untracked
  host pointer) it clears the sticky error with `cudaGetLastError()` and
  returns 0 — important: without the clear, a later unrelated CUDA call
  would observe the stale error.
- `cail_gpu_malloc`/`free` → `cudaMalloc`/`cudaFree`.
- `cail_gpu_memcpy` → `cudaMemcpy(..., cudaMemcpyDefault)` (UVA infers
  direction).

### Reduction kernels (`cail_cuda_reduce.cu`)

- Core kernel: grid-stride loop template
  `cail_reduce_kernel<T, Op>(const T *in, T *inout, size_t count)` with
  functors `OpSum/OpProd/OpMax/OpMin`.
- Launch geometry: `BLOCK_SIZE = 256`, `ELEMENTS_PER_THREAD = 4`,
  grid capped at `MAX_GRID = 65535`.
- Instantiation: `DECL_LAUNCH_{SUM,PROD,MAX,MIN}(type, name)` macros emit
  launch helpers for all 12 internal types × 4 ops; `init_kernel_table()`
  (called from `cail_gpu_init`) fills the function-pointer table
  `kernel_table[CAIL_NUM_OPS][CAIL_NUM_DTYPES]` consulted by
  `cail_gpu_reduce_local`.
- Vectorized fast paths: float SUM uses a `float4` kernel when `count % 4
  == 0` and both pointers are `float4`-aligned; double SUM similarly uses
  `double2`. Scalar tail handling covers remainders.
- Stream model: one static `cail_stream` created at init. Every
  `cail_gpu_reduce_local` launches on it and **immediately**
  `cudaStreamSynchronize`s — the API is synchronous. **KNOWN GAP**
  (performance): no kernel/communication overlap; the dedicated stream
  currently buys nothing over the default stream.

### PCIe read fence (`cail_gpu_flush_recv_buf`)

GPU-aware MPI receives may land in device memory via NIC RDMA or GDRCopy
BAR stores — *posted* PCIe writes with no ordering guarantee relative to a
subsequently launched kernel's reads. The flush:

1. At init, `cudaHostAlloc` one pinned host byte (`flush_host_byte`).
2. On flush, `cudaMemcpy(flush_host_byte, (char*)recv_buf + recv_bytes - 1,
   1, cudaMemcpyDeviceToHost)` — a device-to-host **read** of the buffer
   tail. PCIe ordering rules force all prior posted writes to that endpoint
   to complete before the read returns.

Every algorithm calls this after every receive into device memory
(`algorithms.md`). This is the same class of fence as
`gdr_flush`/`hfi-style` read fences used elsewhere in the Cornelis stack.

## 3. Host Path (`src/gpu/host/cail_host_reduce.c`)

Built with `--enable-host-path` (`CAIL_HOST_PATH`); replaces the GPU backend
for CPU-only clusters and development:

| API | Host implementation |
|-----|---------------------|
| `is_device_pointer` | always 0 (eligibility check is compiled out in `cail_pmpi.c` anyway) |
| `malloc`/`free` | libc `malloc`/`free` |
| `memcpy` | `memcpy` |
| `reduce_local` | `cail_dtype_to_mpi_type` + `cail_optype_to_mpi_op` → `PMPI_Reduce_local(in, inout, count, dt, op)` |
| `init`/`synchronize`/`flush_recv_buf` | no-ops returning 0 |
| `finalize` | no-op (`void`) |

This makes the algorithm layer testable on any node with MPI — the basis of
the CPU correctness tests (`testing.md`).

## 4. ROCm Status — Scaffold Only (STL-78196 relevance)

**There is no functional ROCm support.** Precise state of `src/gpu/rocm/`:

| File | Content | Built? |
|------|---------|--------|
| `cail_rocm_stub.c` | a single `typedef` (empty translation unit) | yes — sole member of `libcail_rocm_la_SOURCES` |
| `cail_rocm_reduce_stub.c` | full `cail_gpu.h` surface, but `init/malloc/free/memcpy/synchronize/reduce_local` all return `-1`; `is_device_pointer`/`flush_recv_buf` return 0; `finalize` no-op | **no — not listed in Makefile.am** |

Additionally:

- **No HIP code exists** — no `<hip/hip_runtime.h>`, no `hipcc` usage,
  nothing.
- `rocm/libcail_rocm.la` is **never linked** into `libcail_gpu.la` under any
  conditional in `src/gpu/Makefile.am`.
- `configure.ac` has no `--with-rocm`, no `HAVE_ROCM` conditional.

### Porting Checklist (ROCm enablement work)

The abstraction boundary is clean — porting is mechanical:

1. **configure.ac**: add `--with-rocm=PATH` (reuse the workspace convention:
   probe `$ROCM_PATH/include/hip/hip_runtime.h` and `lib/libamdhip64.so`),
   define `HAVE_ROCM`, make CUDA/ROCm/host mutually exclusive with a clear
   configure-time error when none is selected (fixes the silent-link-failure
   gap in §1 at the same time).
2. **Memory ops**: `hipPointerGetAttributes` (note: return semantics for
   unregistered host pointers differ across ROCm versions — 5.x returns
   `hipErrorInvalidValue` like CUDA; 6.x returns `hipSuccess` with
   `type == hipMemoryTypeUnregistered`; handle both), `hipMalloc`,
   `hipFree`, `hipMemcpy(..., hipMemcpyDefault)`.
3. **Kernels**: `cail_cuda_reduce.cu` is plain template + grid-stride code —
   hipify is near-1:1. `float4`/`double2` vector types exist in HIP. Wavefront
   size (64 on CDNA vs 32 on NVIDIA) does not affect correctness of the
   grid-stride pattern, only tuning of `BLOCK_SIZE`.
4. **Flush fence**: `hipHostMalloc` + 1-byte `hipMemcpyDeviceToHost` read —
   same PCIe ordering argument holds. For fabric correctness with OPX +
   ROCm-aware Open MPI this fence is load-bearing; verify against large-BAR
   vs non-large-BAR configurations.
5. **Build wiring**: ROCm kernels can be built either as HIP C++ via
   `hipcc` (needs a `hip_lt.sh` analogous to `cuda_lt.sh`) or — simpler — as
   a single `.cpp` compiled with `hipcc` acting as the C++ compiler for that
   convenience library.
6. **Tests**: `tests/` has CUDA-only `.cu` GPU tests; ROCm needs hipified
   equivalents (or a single-source HIP test compiled for both). See
   `testing.md` §5.
