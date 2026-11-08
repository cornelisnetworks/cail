# CAIL — Cornelis Allreduce Interposition Library

Design documentation for CAIL, a GPU-aware `MPI_Allreduce` interposition library.

## Quick Start

| Item | Value |
|------|-------|
| **Repo** | [cornelisnetworks/cail](https://github.com/cornelisnetworks/cail) |
| **Branch** | `main` |
| **Version** | 0.1.0 |
| **Key source files** | `src/core/{cail_pmpi.c, cail_init.c, cail_buf.c, cail_types.c}`, `src/coll/allreduce/`, `src/gpu/{cuda,host,rocm}/` |
| **Validated against** | `main` @ `91c5e4f` (2026-06-12) |
| **GPU backends** | CUDA (implemented), host-only (implemented), ROCm (stub only — see `gpu-layer.md`) |

## What Is CAIL?

CAIL is a drop-in `LD_PRELOAD` library that uses the MPI profiling interface
(PMPI) to transparently intercept `MPI_Allreduce` and route eligible calls
through optimized algorithms with native GPU reduction kernels. No application
changes are required. Ineligible calls (host buffers, unsupported types/ops,
intercommunicators, small messages) fall back to native `PMPI_Allreduce`.

## Files

### Core Architecture
| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | **START HERE** — System overview: layer map, PMPI interception, initialization lifecycle, global state, scratch-buffer cache, logging conventions |
| [interception-and-dispatch.md](interception-and-dispatch.md) | The eligibility gate (when CAIL intercepts vs falls back) and the auto-dispatch algorithm selection logic, including the non-power-of-two threshold-halving rule |

### Algorithms
| File | Description |
|------|-------------|
| [algorithms.md](algorithms.md) | The three allreduce algorithms: recursive doubling, ring, Rabenseifner — communication patterns, rank folding, buffer usage, GPU interaction points, complexity, error paths |

### GPU Layer
| File | Description |
|------|-------------|
| [gpu-layer.md](gpu-layer.md) | GPU backend abstraction (`cail_gpu.h`): compile-time backend selection, CUDA kernel design, PCIe read-fence flush, host path, **ROCm porting status and requirements** |

### Build and Test
| File | Description |
|------|-------------|
| [build-and-packaging.md](build-and-packaging.md) | Autotools build system: configure options, backend selection wiring, `cuda_lt.sh` libtool shim, pkg-config, known build-system gaps |
| [testing.md](testing.md) | Test programs, CLIs, validation methodology, output formats, `test_runner.sh` contract, coverage gaps, and guidance for cn-sw-test ReFrame integration |

## Scope

These documents cover the **implemented** CAIL v0.1.0 library as validated
against `main` @ `91c5e4f`. The CUDA and host-only paths are functional; the
ROCm path is a non-functional scaffold (documented as such in `gpu-layer.md`).
Known defects and gaps are flagged inline with **KNOWN GAP** markers rather
than omitted.

## Related

- [`cornelisnetworks/libfabric-ue`](https://github.com/cornelisnetworks/libfabric-ue/blob/main/CN_INTERNAL/docs/research/INDEX.md) — Cornelis middleware / Ultra Ethernet research notes (`CN_INTERNAL/docs/research/INDEX.md`)
- `src/cn-sw-test/test/opx/` — ReFrame test target for CAIL benchmark/correctness coverage (see `testing.md`)
