# CAIL Testing

**Status**: Implemented (v0.1.0)
**Validated**: 2026-06-12 against `main` @ `91c5e4f`
**Source**: `tests/`

## 1. Test Inventory

All test binaries link **directly** against `$(top_builddir)/src/libcail.la`
(no `LD_PRELOAD` needed under `make check`); production use is `LD_PRELOAD`,
so direct-link tests exercise the same interception path via normal symbol
resolution.

| Binary | Buffers | Coverage | Required args |
|--------|---------|----------|---------------|
| `test_allreduce_basic` | CPU | `MPI_FLOAT` × `MPI_SUM`, sweep {1, 10, 100, 1000, 4096, 100000, 200000} elements | none (`-c`, `-a` optional) |
| `test_allreduce_correctness` | CPU | 20 datatypes × 4 ops × {separate, `MPI_IN_PLACE`} = 160 sub-tests | `-c <count>` (required), `-a <algo>` optional |
| `test_allreduce_correctness_gpu` | GPU (CUDA) | same matrix as CPU correctness | `-c <count>` required; CUDA build only |
| `test_allreduce_edge` | CPU | 12 edge cases: count=0/1/200000, unsupported type (`MPI_LONG_DOUBLE`) and op (`MPI_BAND`) fallback, int MAX / double MIN, all duplicated for `MPI_IN_PLACE` | `-a` optional |
| `bench_allreduce` | CPU | latency/bandwidth sweep, CSV output | `[min_bytes] [max_bytes] [iterations] [warmup]`, defaults 4 B / 16 MB / 100 / 10 |
| `bench_allreduce_gpu` | GPU (CUDA) | same, with global min/max/avg via `MPI_Reduce` | same |

`-a <algo>` works by calling `setenv("CAIL_ALGO", ...)` **before**
`MPI_Init`, so the forced algorithm applies from the first interception.

## 2. Validation Methodology

- Send buffer per rank: `(rank + 1)` cast to the datatype.
- Expected values: SUM = `Σ(r+1)`; PROD = `nprocs!` (skipped for
  `nprocs > 4` — overflow); MAX = `nprocs`; MIN = `1`.
- Tolerances: float `|x−e| > |e|·1e-3 + 1e-6` fails; double
  `|x−e| > |e|·1e-9 + 1e-12` fails; **integers are bit-exact** under a
  width mask.
- Exit codes: `0` all pass, `1` any failure / bad CLI / alloc failure.
  GPU tests also exit `1` (error to stderr, `MPI_Finalize()`, `return 1`)
  when `cudaGetDeviceCount` finds no device.

### Output formats (for harness regex writers)

- CPU correctness: `PASS: algo=<a> <TYPE> × <OP> count=<n> np=<p> pof2=<yes|no>`
  — **UTF-8 multiplication sign `×`**.
- GPU correctness: `PASS: algo=<a> <TYPE> x <OP> count=<n> np=<p> pof2=<yes|no>`
  — **ASCII `x`**, plus a header block and a
  `=== Results: <N> passed, <M> failed ===` summary.
- Basic/edge: `PASS: count=<n> ...` per case.
- Benchmarks: CSV — `# size_bytes,min_us,max_us,avg_us,bw_gbps` then data
  rows. CPU bench times rank 0 only; GPU bench aggregates per-rank averages
  via three `MPI_Reduce` calls (MIN/MAX/SUM). Bandwidth formula:
  `bw = 2(P−1)/P × size / (avg_t × 1e9)` GB/s.

The `×` vs `x` inconsistency is **KNOWN GAP** (cosmetic, but any shared
parser must accept both).

## 3. GPU Test Specifics

- Rank→device binding: `cudaSetDevice(rank % dev_count)` (round-robin over
  local GPUs).
- Device buffers (`d_send`, `d_recv`, `d_inplace`) are **pre-allocated once**
  at max size and reused across all 160 sub-tests — deliberately avoids
  alloc/free churn that corrupts Open MPI's registration cache
  (`opal_interval_tree_traverse` segfaults observed with per-test
  `cudaMalloc`/`cudaFree` at power-of-two sizes). Preserve this pattern in
  any new GPU test.
- Fill via a generic byte-templated `fill_elements<<<grid,256>>>` kernel;
  verification copies results back with `cudaMemcpyDeviceToHost`.
- `bench_allreduce_gpu` auto-caps long runs: for message sizes > 8192
  bytes, `warmup` is reset to 10 and `iterations` capped at 100 regardless
  of CLI values.

## 4. `test_runner.sh` Contract

- Usage: `test_runner.sh <binary> [args...]`; runs the binary at
  `-np 4`, `-np 3`, `-np 1` (pof2, non-pof2, trivial).
- Prepends `-c 1000` **unconditionally** when the binary is
  `test_allreduce_correctness` (`set -- -c 1000 "$@"`); a user-supplied
  `-c` still wins because the binary's `getopt` loop is last-write-wins.
- Env: `MPIRUN` (default `mpirun`), `MPIRUN_FLAGS` (default empty).
- Prints `PASS: np=<n>` / `FAIL: np=<n>` per run and a final
  `Results: <P> passed, <F> failed`; exits 0 iff zero failures.

## 5. Coverage Gaps

| Gap | Impact |
|-----|--------|
| No multi-node runs (`test_runner.sh` is single-node loopback) | Fabric behavior (OPX, RDMA flush correctness) untested in-repo — this is what cn-sw-test must add |
| No ROCm/HIP tests (GPU tests are `.cu`/nvcc only) | Blocks ROCm enablement validation (`gpu-layer.md` §4 step 6) |
| Only `MPI_COMM_WORLD` | Sub-communicators (`MPI_Comm_split`) unexercised — relevant since eligibility checks run per-communicator |
| No multithreaded tests | Consistent with documented thread-unsafety (`architecture.md` §4) |
| No Fortran tests | Consistent with no Fortran bindings |

## 6. cn-sw-test Integration Guidance (planned PR)

Target subtree: `src/cn-sw-test/test/opx/` (ReFrame, Python) — the declared
gap area for OSU/NCCL/RCCL/CUDA/ROCm coverage. Reference:
`test/opx/MpiHelloCheckTest.py`.

Recommended initial ReFrame classes (`@rfm.simple_test`, inheriting the local
`RunControls`/`RegressionTest` conventions):

1. **CailCorrectnessTest** — build CAIL `--enable-host-path` against the
   workspace MPI, run `test_allreduce_correctness -c 4096` under
   `mpirun -np {4,3}` with `--mca pml cm --mca mtl ofi` (OPX provider);
   sanity pattern: zero `FAIL:` lines and exit 0. Parameterize over
   `CAIL_ALGO ∈ {auto, recursive_doubling, ring, rabenseifner}`.
2. **CailEdgeTest** — `test_allreduce_edge` at np=4 and np=3.
3. **CailOsuInterposeTest** — `LD_PRELOAD=libcail.so osu_allreduce -d cpu`
   (and `-d rocm D D` once the ROCm backend exists): validates the
   *production* interposition mechanism, which the direct-link unit tests do
   not.
4. **Two-node variants** of (1) and (3) once node pairs are available —
   closes the multi-node gap in §5.

Performance regression tracking can wrap `bench_allreduce` CSV output with
ReFrame `perf_patterns` later; correctness first.
