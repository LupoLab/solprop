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
  scalar body run by the device layer (`src/launch.jl`); arrays are `Array`s, `CuArray`s
  or `ROCArray`s (the CUDA and AMDGPU package extensions) of `Float64` or `Float32`. The kernel bodies
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
| The exponential Adams march in the engine (`src/march.jl`, native's SEP 39 march transcribed) | `argon_blueshift` on 16 threads 0.42 s -> 0.34 s; the Python march's decisions bit for bit |
| Radial and free planes' local statistics in the engine; the plane's admittance kept | radial on 16 threads 0.65 s -> 0.44 s, `line` 0.27 s -> 0.18 s (native's 0.20 s) |
| State-sized copies as the pool's tasks | a Cartesian 192² window on 16 threads 1.01 s -> 0.78 s (native's 0.85 s): one thread's copy of its 226 MB state took 45 ms, and a step made seven |
| A heavy pass splits from a few hundred elements | `argon_blueshift`'s Adams predictor on 16 threads 50 ms -> 5 ms (its 1601 elements ran serially) |
| Products with a thin side (at most eight columns or terms) as blocked loops, not BLAS | `argon_blueshift`'s plasma projections on 16 threads 54 ms -> 22 ms |
| The collar's half-step factor once a sample, not once an element | radial commits on 16 threads 31 ms -> 17 ms |
| A varying guide's power weight rebound alone, warmed at construction | `rdw`'s 12 planes 37 ms of binding -> none |
| GPU: heavy reductions a thread an element; batched transforms; one kernel a trapezoid increment | the A40's binary64 geometric mean 2.77 -> 2.32 |
| GPU: column and row sums, local sums and occupancy by device reductions | (from the MI325X's trace; measured in the ledger) |
| The split collar on the engine's march; a plasma integral's FIR in one pass | radial off the Python march; `ctc_reduced`'s integral 0.50 ms -> 0.27 ms |
| Nothing left to compile in a run's first window | radial on 16 threads 0.63 s -> 0.40 s, `line` 0.21 s -> 0.14 s (the worker's warm-up runs without a monitor) |

The vectorised functions paid only once the live samples were gathered and each pass
was a function of its own: in the column's branchy loop, and inside its closure, LLVM
kept them scalar, and a table lookup in a loop keeps LLVM from vectorising it, so the
rate is a vectorised `log`, the lookup and a vectorised `exp`, three passes.

## Where it stands

Julia's window time over native's, 2 % of each case, both arms in one interpreter, the
same steps and binary64 fields equal to rounding (1e-13 or better); the ledger has the
seconds, the machines and the earlier rows.

| case | Xeon, 1 thread | Xeon, 16 threads | A40, binary64 | A40, binary32 | H200 | MI325X |
|---|---:|---:|---:|---:|---:|---:|
| rdw | 1.52 | 1.04 | 2.86 | 3.05 | 2.02 | 2.03 |
| argon_blueshift | 1.51 | 1.21 | 2.80 | 2.24 | 2.56 | 2.33 |
| h2_raman_6ps | 1.24 | 0.89 | 1.02 | 0.66 | 1.96 | 1.71 |
| ctc_reduced | 1.56 | 0.90 | 1.47 | 1.49 | 1.98 | 2.27 |
| radial_filament_256 | 0.97 | 1.12 | 2.05 | 2.27 | 4.42 | 4.85 |
| kerr_beam_192 | 1.10 | 1.26 | 2.75 | 3.63 | 1.69 | 3.44 |
| line | 1.08 | 0.78 | 1.68 | 2.24 | 2.92 | 1.89 |

Intel Xeon Gold 6240 (16 cores of socket 1, quiet) and the NVIDIA A40 (binary64) at
commit `a6722029`, geometric means 1.13 and 1.97; the A40's binary32 and the NVIDIA H200
(binary64) at `50158fa6`, 1.99 and 2.39, before the first window stopped compiling, which
cost radial and the free-space windows most (radial 0.53 s -> 0.35 s on the A40); the AMD Instinct MI325X (binary64) against
native's HIP build at `196baa51`, before the GPU's column reductions, 2.48. On GPUs with
full-rate binary64 the windows are host-bound (the launches, copies and allocations a
step), which is what is left there. Binary32 is against native's binary64. Ratios measured before
the harness fix (#557) counted 0.18 s of metadata on both arms and are not comparable.

## What is left, measured

- **The march's host events.** The Adams march runs in the engine (`src/march.jl`)
  and returns at each plane, monitor check and statistics row; a split collar, a
  replayed schedule and interpolated planes keep the Python march.
- **Host statistics per plane.** The spectra, and the reductions over the engine's
  power and fluence, are the shared host functions: about 1.4 ms a plane on `rdw`.
- **Small states on many threads.** With one mode and 1601 bins the passes are serial
  by size, and run 20 to 40 % slower with 16 cores busy than with one (the clock).
- **Serial plasma integrals.** The two running integrals are sequential by design.
- **Binary32 on x86.** It runs (fields 1e-6 from binary64), but two to three times
  slower than binary64 on the Xeon. Subnormals are not the cause: flushing them
  (`SOLPROP_JULIA_FTZ=1`) left every window within its rounds' spread. Not yet found.
- **The A40's binary64.** It runs at 1/64 of its binary32 rate; the A40 says little about
  an H200 or an MI325X. Binary32 with the double-float wide type (below) takes it out of
  the operator and the phi functions, and the windows barely move (0.99): the host is
  what is left there too.
- **Metal.** The capillary windows are bound by waits on the device. On the M5 Max a
  queued kernel launch costs 4.8 us, but each value the host waits for costs about
  0.2 ms: a launch and `synchronize` 160 us, a reduction read back 210 us, a 40 kB copy
  to the host 190 us (CUDA's are tens of microseconds). The trial's norms, the
  quadrature alarm, the monitor and the statistics each wait; fewer of them a step (the
  sums fused and read once, or kept on the device until the march needs them) is the
  next lever. The section timers put `commit` (its alarm's transforms and reduction)
  at about 1.2 ms a step on `rdw` and `argon_blueshift`. `h2_raman_6ps`'s strided
  products are fixed (0.76); its Raman products and projections are still its largest
  share. Construction compiles the Metal kernels (about 28 s), with no precompiled
  trace for the device.

## Apple silicon and Metal (2026-11-07)

On the M5 Max the engine is faster than native on the CPU, and runs on its GPU. The
ledger has the rows.

- **CPU, binary64.** 2 % windows on 1, 6 and 18 threads: 0.83 over native (16 of 21
  rows faster); on 6 threads with 30 s idle between workers 0.79. Whole runs on 6 threads: 0.82 (`ctc_reduced` 0.71, `h2_raman_6ps` 0.82
  over 63 s, the Cartesian beam 1.02), the same steps, fields to 1e-11. On one thread
  the two are close (0.84 to 1.23); the gap opens with threads, as native's OpenMP
  regions slow on the 12 second-tier cores (eight threads already slower than six, 18
  at 1.4 to 1.5 times six's time) where Polyester's tasks lose less.
- **Accelerate.** Apple's BLAS for the products is the default on macOS: radial's
  window on one thread 1.02 s -> 0.60 s, no case slower.
- **Metal.** `SolpropJuliaMetalExt` (Metal.jl 1.12), binary32 only. Every binary32
  window passes against `numpy_ref`. Against native's binary64 on six CPU threads 1.95
  (the Cartesian beam 0.33, radial 0.92, the capillaries 3 to 6); against `mlx` 1.25
  (the Cartesian beam 0.71, radial and `ctc_reduced` 0.96, `h2_raman_6ps` 2.1).
  Metal.jl's FFTs (MPSGraph) cover every plan form the engine uses without MLX's
  Bluestein cliff; its float32 GEMM is full precision, not TF32.
- **Double-float.** An Apple GPU has no binary64, so a binary32 run there forms the
  operator's exponents, the propagators and the Adams phi functions in `DF32`, a value
  as two binary32s (about 48 bits; `src/dfloat.jl`, the mlx backend's primitives), each
  result rounded once to binary32. `SOLPROP_JULIA_WIDE=df32` takes it on any device;
  binary64 runs keep their arithmetic exactly. Its phi functions agree with binary64's
  to within 2^-36, its propagator pair to 8 binary32 ulps up to a 1e6 rad phase.
- **Measurement.** Absolute times on the Mac moved by up to 40 % over a day of
  benchmarks, with no thermal warning; the ratio did not (0.79 to 0.88 over every
  variant: windows, whole runs, 30 s idle between workers, single-backend passes back
  to back). Quote ratios from interleaved arms, not absolute times from different runs.

## Floating point (2026-11-07)

SEP 25 permits contraction and per-kernel reassociation and forbids umbrella fast-math,
approximations and flush-to-zero. The engine now contracts its ordinary kernels (the
stage combinations, the Adams sums, the phi series, the plasma rate's cubic, the FIR
sums); the protected helpers, the march's decisions, the defects and the norms stay as
written, and the CPU column scan's order is a choice, not policy. Contraction measured
0.99 on the M5 Max (`h2_raman_6ps` 0.93 to 0.96 the only steady gain): the engine is
bound by passes over memory, not by operations.

Flush-to-zero, as an experiment (`SOLPROP_JULIA_FTZ=1`, #561), helped nowhere: 1.00 on
the M5 Max in binary32, 0.99 on the Xeon in binary32. It is also hazardous embedded in
Python: Julia's first thread is the interpreter's own, and a flag set there flushed
NumPy's subnormals too, failing a `math.log` in solprop's translation. The switch flushes
only inside the engine's entry points.

## Toward a first-class backend

What these measurements say about Julia becoming a supported backend beside, or in
place of, native and `mlx`.

**For.**
- **Speed on a CPU.** Faster than native on the M5 Max (0.82 over whole runs), within
  13 % on the Xeon at 16 threads, faster in some rows there.
- **One source for every device.** About 3,500 lines of engine (plus the 1,600-line
  compile trace and a 2,400-line adapter), against about 31,000 of native's C++. The
  same scalar bodies run on CPUs, NVIDIA, AMD and now Apple GPUs; Kokkos has no Metal
  backend. Intel GPUs (oneAPI.jl) should take an extension like the others.
- **A replacement for `mlx`.** Within 25 % of the hand-written `mlx` backend on its own
  GPU on the first, unprofiled run, faster on the largest case.
- **Iteration.** Each device extension so far took an hour or two; the double-float
  path was a day's work and runs on every device.

**Against, or not yet.**
- **GPU speed.** About twice native on full-rate GPUs (H200 2.39, MI325X 2.48, not
  re-measured since), host-bound: launches (about 75 a trial), copies and allocations.
  CUDA graphs or fused trials are the next step, untried.
- **Coverage.** Julia refuses most of what native runs beyond the seven focus routes
  (the list below), and several things `mlx` runs (envelope plasma, tapers, batches,
  several plasma channels or Raman terms). Each is a feature to write and to test
  against `numpy_ref`.
- **Distribution and start.** A Julia runtime and a 1.8 GB depot beside the wheel;
  the first engine in a process compiles for 11 to 14 s on a CPU (after the
  precompiled trace) and about 28 s on Metal. A PackageCompiler system image, untried,
  would remove most of it. Native ships as a prebuilt wheel on PyPI.
- **Qualification.** No CI lane runs it. The double-float path has window tests and
  unit bounds, not `mlx`'s SEP 33 drift and envelope qualification.
- **The toolchain's own bugs.** Julia's LinearAlgebra gave wrong concurrent
  `Complex{Float32}` products (worked around); juliacall shares the interpreter's thread.

**The decisions a SEP would take.** Whether Julia replaces native, sits beside it, or
replaces only `mlx`; which features it must cover before that; how it ships (system
image, `juliapkg`, or a separate distribution); which CI lanes run it (a CPU lane, and a
GPU lane somewhere); and whether the double-float path takes `mlx`'s qualification.
The cheapest step with the most information is to replace `mlx` first: profile Metal
on the losing cases, add `mlx`'s features, run SEP 33's qualification on `df32`, and
compare again.

## Not run

Vector polarisation, batches, noise, envelope plasma, several plasma channels or Raman
terms, varying Raman, dispersive Kerr, TDSE, planar and general modal guides, tapers,
coupled flows, the interaction Adams formulation, the carrier frame and interpolated
planes are refused by name; so is binary64 on Metal.

## Findings for main

- `parallel_rows` read the CPU topology from sysfs (a file per CPU) before deciding a
  pass was too small to thread: 60 ms of a 0.36 s rdw window. Fixed on this branch;
  #556.
- `benchmarks.worker.timed_run` evaluated `simulation.metadata()` inside the timed
  window, and its first evaluation costs about 0.18 s on rdw (the plane bases' modal
  identity): every arm's first timed repeat, native's included, paid it. Fixed on this
  branch; ratios measured before it understate both arms' speed (native's `rdw` window
  on 16 threads is 0.145 s, not 0.314 s); #557.
- Julia's LinearAlgebra gave wrong products for concurrent `mul!` calls on views of a
  `Complex{Float32}` matrix times a `Float32` one (relative error 1.5); the engine forms
  them as real GEMMs over the complex array's real view instead.
- `docs/dev/backends.md` still names `set_held_plane` and `held_observation`, which
  main removed; #558.
- On the M5 Max native is slower on 8 and on 18 threads than on 6, on most cases: its
  parallel regions wait for the second-tier cores. OpenMP's `active` waiting does not
  change it. Not filed yet.
- On the Mac absolute times drift by up to 40 % over a day of sustained benchmarking
  (no thermal warning recorded); only interleaved ratios are comparable across runs.
  `benchmarks ab --cooldown` (this branch) idles between workers; 30 s did not undo it.
