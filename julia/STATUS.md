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

What did not pay: branch-free vectorised `exp` and `log` (1 ulp, and 2x faster than
Base's in isolation) made the plasma's columns slower in place, where the table gather
keeps the loops scalar and Base's cached exponential is cheaper.

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

- **Host work per step.** The Adams march, its stage scalars and coefficient tables run in
  Python: about 1.2 ms a step on the Xeon (rdw at 16 threads: 0.09 s of engine work in a
  0.23 s window). Native runs the march compiled (SEP 39). A Julia march is the largest
  remaining CPU and GPU item.
- **Host statistics per plane.** The saved planes' statistics are the shared host
  functions: about 4.6 ms a plane on rdw (12 planes in the window). Native forms them in
  its own code.
- **Serial plasma.** The column scans' rate (scalar `log` and `exp` behind an exact
  zero-support test), fraction and two running integrals take 0.8 s of argon's 1.2 s
  window on one thread; native's vectorised SLEEF calls do the same in less.
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
- `benchmarks.worker.timed_run` evaluates `simulation.metadata()` inside the timed
  window, and its first evaluation costs about 0.18 s on rdw (the plane bases' modal
  identity): the first timed repeat of every arm, native's included, pays it.
