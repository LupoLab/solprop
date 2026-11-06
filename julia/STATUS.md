# The Julia experiment: status

An experiment. Draft PR, not for merge; no CI lane runs it. This page records what the
engine is, how fast it is against native on a CPU and on an NVIDIA A40, what it does not
do, and what is left. Measurements are short windows of the benchmark cases (`python -m
benchmarks ab`, ten to two hundred steps); [the ledger](../docs/dev/benchmarks.md#julia-backend-experiment)
has the full rows.

## What changed from PR #10

PR #10 was a CPU-only prototype written against kernel schema 35. This branch carries it
to current main and rewrites the engine:

- **Main's seam.** No schema number (ADR-068): a digest of the Julia sources, the adapter
  and the manifests, checked when the engine loads. `set_initial(resume=)`, the Adams
  engine protocol's carrier, hold, restore, observe and snapshot size, stage scalars from
  the driver's bundle cache, the general-leaf and carrier-frame refusals, SEP 21's split
  collar and SEP 44's transverse edge read through the engine.
- **One source for CPU and GPU, binary64 and binary32.** Every state-sized operation is a
  scalar body run by the device layer (`src/launch.jl`); arrays are `Array`s or
  `CuArray`s (the CUDA package extension) of `Float64` or `Float32`. The kernel bodies
  capture only values and arrays, which `SOLPROP_JULIA_CHECK` enforces on a CPU.
- **Julia 1.13.3**, CUDA.jl 6.5, FFTW.jl, Polyester, KernelAbstractions and
  AcceleratedKernels; numba is gone from the tooling, which is now solprop's own harness.

## What made it fast

Measured step by step on an Intel Xeon Gold 6240 (16 cores of one socket) unless said.

| Change | Effect |
|---|---|
| Measure every FFTW plan | a threaded estimated c2r (73728 x 17) took 6.6 ms, measured 1.49 ms |
| Raman: convolve the modal products, not the node drives, where fewer (as native does) | H2 Raman window 10.8 s -> 4.7 s |
| One CPU pool: Polyester tasks for every pass, FFTs as blocks of single-threaded plans, GEMMs as blocks of rows | rdw on four threads 1.10 s -> 0.22 s; Julia's tasks cost 10 us a pass where Polyester's cost 2.2 us |
| Contiguous column loops (`each!`), not a flat index | a flat index's divide made a Kerr pass ten times slower |
| Hold host BLAS to one thread while the engine lives (as native does) | NumPy's spinning OpenBLAS made a radial window 4.3 s against 1.1 s |
| Compile and run every path, and every bridge entry, at construction | 0.6-3.5 s of JIT left the timed window |
| Monitor and edge sums, state copies, in the engine | a Cartesian window's host copies and reductions 1.7 s -> 1.1 s |
| GPU: plasma and observation populations a sample a thread, scans column-wise | A40 rdw plasma 1.23 s -> 0.04 s, argon 4.9 s -> 0.14 s |
| Native's phi-function evaluation (quarter-ulp stop, reciprocals) | radial Adams predictor 0.16 s -> 0.10 s |
| Frequency-domain synthesis where the nodes match the modes (radial) | a radial window 0.86 s -> 0.63 s on 16 threads |
| Its own helpers record (SEP 29 section 6.6), an Adams trial's tables formed in the engine | `rdw` on 16 threads 0.273 s -> 0.214 s; equal to the Python forms bit for bit |
| Stage scalars read by the engine from its own tables (SEP 39's staged tables) | `rdw` stage preparation 17 ms -> 5 ms a window |
| The Poynting statistics' transforms and the populations in the engine, without allocating | `rdw` statistics 55 ms -> 40 ms; collector pauses (13 to 83 ms) gone |
| A plasma column's live samples gathered; vectorised `exp` and `log` (`src/vmath.jl`, 1 ulp) | argon's plasma on one thread 4.7 ms -> 2.9 ms a source |
| A source's forward transforms, and the quadrature alarm's, as one pool's work | `rdw` on 16 threads: commits 12.3 ms -> 5.8 ms, the source's transforms 8.7 ms -> 3.8 ms |

The vectorised functions paid only once the live samples were gathered and each pass
was a function of its own: in the column's branchy loop, and inside its closure, LLVM
kept them scalar, and a table lookup in a loop keeps LLVM from vectorising it, so the
rate is a vectorised `log`, the lookup and a vectorised `exp`, three passes.

## Where it stands

Julia's window time over native's, 2 % of each case, both arms in one interpreter, the
same steps and fields equal to rounding (1e-13 or better) unless said.

**Intel Xeon Gold 6240** (commit `3e2290c0`, socket 1, passive-bound threads):

| case | 1 thread | 16 threads |
|---|---:|---:|
| rdw | 1.54 | 1.34 |
| argon_blueshift | 2.43 | 2.41 |
| h2_raman_6ps | 1.19 | 1.08 |
| ctc_reduced | 2.67 | 1.42 |
| radial_filament_256 | 1.61 | 3.07 |
| kerr_beam_192 | 1.31 | 3.20 |
| line | 1.39 | 1.81 |

**NVIDIA A40** against native's CUDA build (commit `90840a43`, binary64):

| case | ratio |
|---|---:|
| rdw | 1.84 |
| argon_blueshift | 3.43 |
| h2_raman_6ps | 1.33 |
| ctc_reduced | 1.74 |
| radial_filament_256 | 4.35 |
| kerr_beam_192 | 3.62 |
| line | 2.98 |

The AMD EPYC 7763 and binary32 rows are in the ledger.

## What is left, measured

- **The march in Python.** The Adams controller's decisions and the bridge's calls
  between them: about 0.37 ms a step on the Xeon (`argon_blueshift` on 16 threads:
  0.08 s of a 0.40 s window, 213 steps; native's whole step is 0.86 ms). A Julia march
  (native's SEP 39, with its decision-trace tests as the oracle) would remove it; the
  tables and stage scalars it needs are in the engine already.
- **Host statistics per plane.** The spectra and the reductions over the engine's power
  are the shared host functions: about 1.4 ms a plane on `rdw`.
- **Small states on many threads.** With one mode and 1601 bins the passes are serial
  by size, and run 20 to 40 % slower with 16 cores busy than with one (the clock).
- **Serial plasma integrals.** The two running integrals are sequential by design.
- **Binary32 on x86.** It runs (fields 1e-6 from binary64), but slower than binary64:
  the scaled field's tails are subnormal, and flush-to-zero is not allowed. GPUs handle
  binary32 subnormals at full speed.
- **The A40's binary64.** It runs at 1/64 of its binary32 rate; the A40 says little about
  an H200 or an MI325X.

## Not run

Vector polarisation, batches, noise, envelope plasma, several plasma channels or Raman
terms, varying Raman, dispersive Kerr, TDSE, planar and general modal guides, tapers,
coupled flows, the interaction Adams formulation, the carrier frame and interpolated
planes are refused by name.

## Findings for main

- `parallel_rows` read the CPU topology from sysfs (a file per CPU) before deciding a
  pass was too small to thread: 60 ms of a 0.36 s rdw window. Fixed on this branch.
- `benchmarks.worker.timed_run` evaluated `simulation.metadata()` inside the timed
  window, and its first evaluation costs about 0.18 s on rdw (the plane bases' modal
  identity): every arm's first timed repeat, native's included, paid it. Fixed on this
  branch; ratios measured before it understate both arms' speed (native's `rdw` window
  on 16 threads is 0.145 s, not 0.314 s).
- Julia's LinearAlgebra gave wrong products for concurrent `mul!` calls on views of a
  `Complex{Float32}` matrix times a `Float32` one (relative error 1.5); the engine forms
  them as real GEMMs over the complex array's real view instead.
- `docs/dev/backends.md` still names `set_held_plane` and `held_observation`, which
  main removed.
