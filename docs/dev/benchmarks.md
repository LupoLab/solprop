# Benchmarks

How solprop's performance is measured and judged, and what the measurements so far
indicate. A performance claim names what it measured, the build, the machine and the
result; the rules below say how to take it. The numbers in the indicative sections are
indicative: each was measured once, on one machine, for one change. Re-measure before
relying on any of them.

## Where things are

- **The benchmarks** are `benchmarks/`
  ([SEP 38](seps/archive/sep-0038-performance-tracking.md)), run with `uv run poe bench`
  (`python -m benchmarks`). `track` times complete runs of the tracking set (about a
  dozen representative workloads, with each interface's default integrator and
  tolerances) at one thread and the production width, in about an hour per machine;
  `ab` compares two builds while a change is made. [Running the benchmarks](benchmarking.md)
  is the manual.
- **The history** is `benchmarks/history/`: one CSV per hardware set, named by its CPU
  model and core count, with a few rows per release. `python -m benchmarks history`
  summarises it. Full run bundles stay in the ignored `.benchmarks/` directory.
- **[CPU performance: what was tried](cpu-performance.md)** records the compiled CPU
  backend's performance experiments: what was adopted, measured and rejected, and what
  remains open. Read it before repeating an experiment.
- **[backends.md](backends.md)** describes each backend's execution design and the
  performance policies it adopts.

(benchmark-rules)=
## Rules

- State thread counts and plan flags; never compare across different ones.
- A hot-path change needs a benchmark entry in its pull request or commit message.
- Estimates are labelled as estimates in the docs until a measurement exists.
- Do not tune for the benchmark: the cases are the target simulations at their real sizes.
- **A performance change is judged on the aggregate**, not case by case: the
  equal-weight geometric mean of its per-case timing ratios over the cases it can
  affect, declared before timing, taken separately for the production thread policy
  and for one thread, with the complete per-case table beside it and a whole-suite
  aggregate when the work closes. A change is adopted when the production saving is
  significant: more than 5 %, and the geometric mean of the conservative per-case ratios
  `(candidate median + 2 MAD) / (before median − 2 MAD)` below one, a non-positive
  denominator counting as unresolved. Small regressions on individual cases, or at one
  thread, are accepted when the aggregate improves; they stay visible in the table and
  do not veto the change. Numerical bounds are never relaxed to make a change pass.
  The A/A spread of the same comparison says how large a difference has to be to count
  ([comparing two builds](benchmarking.md#comparing-two-builds-while-iterating)).

(benchmark-windows)=
## What is timed

- **Release tracking times complete runs.** The tracking set's cases are complete
  guides at production sizes, with the library's defaults, so the history says how long
  the usual simulations take now. Construction, translation and backend construction
  are timed separately; an untimed warm-up builds plans first
  ([what is timed](benchmarking.md#what-is-timed)).
- **A change in progress may time less.** `ab --length` runs part of each guide from
  the entrance, which is fine for comparing two builds on the same work. Report the
  fraction run, and do not extrapolate an entrance window to a complete-run cost: the
  expensive part of a propagation (compression, ionisation, the Raman shift) often comes
  late.
- **A later state of a long propagation** is timed by replaying a saved witness into a
  short run, not by regenerating the propagation.
- **Compare two source trees with alternating pairs** (A/B/A/B…), and report the A/A
  spread beside the ratio; a difference inside the spread is not a result.

(benchmark-reporting)=
## Reporting a performance claim

**The machine record.** Every measurement carries the CPU (or device) model, core count,
memory, operating system, Python and dependency versions (including the Kokkos build and
its profile), thread counts, FFTW plan flags and wisdom state, the source revision and
whether the tree was dirty, and the load before and after. A tracking run's plan and
results record these; a hand-run measurement states them. Results are compared
only within one machine.

**A quiet machine.** Absolute numbers are taken on a dedicated workstation or an
exclusively allocated node. A busy laptop is fit only for same-process A/B comparisons,
labelled as such: read their ratios, not their times. Report the A/A spread of repeated
baseline runs beside every ratio; a difference inside it is not a result.

**What goes in a pull request.** The measured window and what it includes, the build and
machine record, the per-case table with medians and spreads, the declared aggregate and
its verdict, and any case that regressed. Concise evidence belongs in the change
description; raw runs stay local.

**Numba's on-disk cache.** Clear it
(`find src -name '__pycache__' -type d -prune -exec rm -rf {} +`) before timing in a
worktree that has run the suite many times: tens of thousands of cache files slow every
tool that walks the tree. Numba's startup includes compilation or cache loading and is
not a reproducible cold-compilation measurement.

(benchmark-results)=
## Indicative results

Indicative ratios, not absolute times. Each is a short window (see above) on one
machine; the rows' cases, grids and step lengths differ, so rows share no scale. They say
which backend or method to expect to win, and by roughly how much.

### Backends and builds

**Speed-up over `numpy_ref`, one thread**, ten-step windows on an 18-core AVX-512 server
(one socket, one NUMA node) with a local Kokkos build (vendor BLAS, native architecture).
The capillary cases have 3 or 4 channels on a 513-bin grid; the general-route cases are a
two-channel step-index pair and a graded-index guide at 2304 quadrature nodes.

| case | `numba_cpu` | Kokkos CPU |
|---|---:|---:|
| M1 soliton (GNLSE), reduced | 2.6× | 2.1× |
| GNLSE soliton with Raman | 2.6× | 2.6× |
| M2 (RDW capillary), reduced | 4.9× | 5.9× |
| M2 (RDW capillary), full | 7.1× | 8.6× |
| capillary, no wall | 2.5× | 3.9× |
| capillary, scalar coherent wall | 2.1× | 3.5× |
| capillary, TE/TM coherent wall | 4.2× | 8.8× |
| capillary taper with wall (coupled flow) | 2.4× | 2.8× |
| general route, step-index pair | 2.7× | 5.5× |
| general route, graded index | 2.9× | 4.7× |

**Threads.** `numpy_ref` does not thread. At four threads Kokkos CPU ran the general route
2 to 6 times faster than `numba_cpu` and the capillary cases 1.5 to 3 times faster. On
larger general-route and coherent-wall workloads (2 to 32 channels, up to 16 384 fine
samples) at 18 threads it was 1.2 to 4 times faster than `numba_cpu` at 18 threads and 6 to
40 times faster than single-threaded `numpy_ref`. Small capillary grids stop scaling beyond
a few threads on every backend.

**Kokkos build profiles**, same server, seven ten-step capillary and general-route cases.
The vendor BLAS and the instruction set are separate gains. Against the reference build
(no vendor BLAS, generic x86-64), linking OpenBLAS alone made the straight capillaries 2.6
to 3.8 times faster at one thread (the tapers 1.7) and left the general route unchanged;
compiling for `x86-64-v3` alone made the general route 1.4 to 1.5 times faster and left
the capillaries unchanged. A distribution-style build (`x86-64-v3` and the
`scipy-openblas32` OpenBLAS, ADR-034) ran within about 5 % of the local build
(`-march=native`, the system OpenBLAS) on every case at one and four threads and beat
`numba_cpu` on all seven; `x86-64-v4` added 7 to 14 % on the general route only. The
distribution wheels now target a lower baseline ([CPU performance](cpu-performance.md)).
At one thread, in milliseconds per ten steps:

| case | reference | distribution-style | local | `numba_cpu` |
|---|---:|---:|---:|---:|
| scalar wall | 61.4 | 21.8 | 22.8 | 35.0 |
| TE/TM wall | 1547 | 390 | 402 | 480 |
| taper wall | 90.3 | 54.6 | 56.6 | 64.7 |
| general pair | 188 | 118 | 120 | 241 |
| general GRIN | 358 | 245 | 244 | 454 |

On the same server MKL was 1.4 to 1.8 times faster than OpenBLAS on the TE/TM wall, where
the MKL build keeps the custom complex kernel, and level elsewhere.

**GPU against a whole CPU socket.** On the same seven cases, the Kokkos CUDA build on a
data-centre NVIDIA GPU and the HIP build on an AMD Instinct GPU ran 2.7 to 18 times faster
than the Kokkos OpenMP build on all 18 cores of the server, the graded-index general route
most. These grids are small for a device; the ratios say nothing about large grids.

### Kokkos CPU operation-size threading

The Kokkos CPU backend chooses each operation's thread width from its dimensions at
construction ([tuning](../guide/performance/tuning.md); the thresholds are under
[reference tables](#kokkos-cpu-sizing-thresholds)). It was calibrated on an 18-core arm64
laptop and a two-socket, 36-core AVX-512 server, against one thread on the small cases and
against the previous default build on the large windows (best of complete paired rounds):

| window | arm64 laptop | two-socket server, all 72 logical CPUs |
|---|---|---|
| small cases (eight case/track configurations), geometric mean | 2.00× faster | 2.06× faster |
| averaged helium RDW | same within noise | same within noise |
| resolved blueshift | 2.0–2.6× faster | 1.8–2.4× faster |
| reduced comb-to-continuum (revision 1, 1 ps) | 2.8–3.1× faster | 2.4–3.6× faster |
| RDW ten-step window | 1.81× faster | 2.33× faster |
| reduced comb-to-continuum ten-step window (revision 1, 1 ps) | 3.18× faster | 3.85× faster |
| full comb-to-continuum ten-step window | 7.9 % faster | 2.7 % faster, noisy |
| general modal | 4–11 % faster | same, noisy |
| Cartesian and radial filaments | same within noise | within 5 %, noisy |
| scalar / TE-TM walls | 44 % / 17 % faster | TE-TM same |

Separate thresholds for copies and arithmetic matter: an 8,192-element temporal copy is
slower threaded, while arithmetic stages of the same size gain from the full pool. A wide
response chunk followed by a full-width FFT needs both at full width, and a serialised
coupled-flow matrix function on the scalar wall took about 20 times its threaded kernel's
time; both were corrected before the comparison above. "Noisy" rows did not meet the
predeclared stability check; taper, short-line and planar controls were not requalified.

The exponential Adams trial without a coupled flow is two passes around its source
evaluation, a prediction and a correction that also reduces the defects' norms. An
element of them costs about ten stage entries, so they take a class of their own, its
work the field's elements times the history's rows (seven at the default fifth order).
Forced to 1-16 threads of a 16-thread process whose idle threads sleep, over
single-column capillaries of 401 to 10,241 elements, on 16 cores of a dual AMD EPYC
7643: four threads lost at 4,487 (0.8×) and gained 1.1-2.1× (median of two rounds)
from 5,607 to 71,687, while sixteen beat four on argon from 11,207 but lost to them on
helium up to 17,927 and lost to one thread at 8,967 (0.7×), a fork and join of sixteen
sleeping threads costing 100-200 µs. The first round had the node to itself; in the
second, with other jobs on it, the serial times held within 5 % while the threaded ones
slowed by up to 2.3×, the sixteen-thread ones most, yet four threads stayed at least as
fast as one from 5,607 up. On a loaded Apple M5 Max (indicative) four gained 1.0-1.6×
at 4,487-8,967 and sixteen beat four only at 71,687 (4.7× against 2.8×). The class
therefore threads at four from 8,192 and takes the pool from 65,536 on both platforms.

(stable-kernel-team)=
### A stable kernel team

GNU OpenMP keeps one pool of threads for the main thread's parallel regions. When a region
asks for fewer threads than the last, the surplus threads end; the next wider region
starts them again. A four-thread kernel between full-width ones therefore costs a thread
creation per surplus thread each time: alternating 16- and 4-thread regions took about
490 us a pair, against 40 us for two 16-thread regions and 23 us for 16 and 1 (one socket
of a Xeon Gold 6240, idle threads sleeping; twelve `clone3` calls a pair). FFTW's threaded
loops run on the runtime's whole team and only hand out fewer iterations, so a
four-thread FFT plan costs no resize. At 16 threads, `h2_raman_averaged` (an exponential
Adams trial pass and a response at four threads beside a full-width stage) re-created
twelve threads every source evaluation, and was slower than at one thread.

A kernel pool that already holds a full-width instance now hands it to medium-width
kernels too; a pool without one keeps four threads (averaged helium, whose other kernels
are serial, took 14 % longer on the Xeon with its Adams passes at full width). Workspaces ask for their stage
instance first. Interleaved `benchmarks ab` at 16 threads against the parent, 3 rounds,
`--length 0.1`, passive-bound placement, the same steps and fields to rounding (at most
1e-13 relative):

| case | Xeon Gold 6240 (socket 1) | AMD EPYC 7763 (16 cores) |
|---|---:|---:|
| `h2_raman_averaged` | 0.526 [0.521, 0.531] | 0.483 [0.483, 0.510] |
| `argon_blueshift` | 0.717 [0.700, 1.070] | 0.806 [0.800, 0.850] |
| `ctc` | 1.001 | 0.979 |
| `gnlse_raman_shock`, `gnlse_soliton`, `rdw`, `he_rdw_averaged`, `argon_blueshift_averaged`, `argon_blueshift_envelope`, `radial_filament_256`, `grin_nonlinear`, `kerr_beam_384` | 0.984-1.036 | 0.983-1.106 (node noise) |
| geometric mean (12 cases) | 0.924 | 0.938 |

The unchanged cases have no medium-width kernel or none beside a full-width one; their
spread is the A/A spread (about 2 % on the Xeon, up to 10 % on the shared EPYC nodes). On
the Xeon `h2_raman_averaged` now takes 0.79 s at 16 threads against 1.06-1.12 s at one.
Every medium width serial instead lost 2.8 times on averaged helium (EPYC geometric mean
1.08); every medium width full, the FFTs included, gained on resolved cases but lost on the
averaged ones (EPYC 0.94, Xeon 0.90 with averaged helium 1.22).

(host-blas-pool)=
### NumPy's BLAS beside a native run

NumPy's OpenBLAS is a pthreads library with a pool of its own, sized from
`OPENBLAS_NUM_THREADS` or every CPU (the benchmark suite sets the variable to the run's
width). During a native run the host calls it only for small products: the power
weights' expansion at each saved plane is a 3 × 2561 `dgemv` on averaged helium, 24 of
them in a `--length 0.5` window. After each call OpenBLAS's workers spin before they
sleep, and the planes come often enough that they never do: at 16 threads on the Xeon
Gold 6240, its 15 workers took 1.09 s of CPU each in a 1.24 s window, and the
four-thread Adams passes ran three times as slowly (0.74 s against 0.22 s, label timer).
The native backend now holds every BLAS but its own to one thread while it lives, as
`numba_cpu` holds its BLAS pool. Interleaved `benchmarks ab` against the parent, 3 rounds,
`--length 0.1`, passive-bound placement, the same steps and fields to rounding:

| case | Xeon Gold 6240, 1 thread | Xeon Gold 6240, 16 threads |
|---|---:|---:|
| `rdw` | 0.996 | 0.462 [0.456, 0.465] |
| `he_rdw_averaged` | 1.011 | 0.693 [0.670, 0.773] |
| `radial_filament_256` | 0.996 | 0.725 [0.718, 0.726] |
| the other nine tracking cases | 0.984-1.019 | 0.979-1.010 |
| geometric mean (12 cases) | 1.000 | 0.886 |

The four-mode capillary now runs 3.1 times faster at 16 threads than at one (0.77 s
against 2.40 s), where it was 1.4 times.

(smooth-size-cliffs)=
### Smooth grid sizes

`native/benchmarks/size_cliffs.py` times the four-mode RDW capillary per source
evaluation (short and long capped Runge-Kutta windows, their difference) on 2-3-5-smooth
and power-of-two grids, and flags a size whose cost per sample exceeds its nearest power
of two's by more than 25 %. Two size boundaries made cliffs: the stage threshold (8192
elements, just above n_t 4000's 8004-element stage, whose entry kernel then took 708 us
serially against 59 us at full width) with the FFT threshold (32,768 samples, above its
32,000-sample batch), and a node budget that left 33 nodes as a 32-node pass and a serial
one-node tail at n_t 8640. With the thresholds at 4096 and 16,384 and the nodes spread
evenly over the passes, on one 18-core socket of a Xeon Gold 6240 (GCC 16, FFTW 3.3.11,
ms per evaluation, two rounds each):

| n_t | 4000 | 4096 | 4500 | 5000 | 6000 | 7200 | 8000 | 8192 | 8640 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 2.45–2.47 | 1.24–1.36 | 1.49–1.56 | 1.54–1.57 | 1.86–1.87 | 2.21–2.26 | 2.38–2.39 | 2.29–2.30 | 3.52–3.55 |
| after | 1.32 | 1.27–1.36 | 1.53–1.54 | 1.57–1.61 | 1.83–1.87 | 2.17–2.26 | 2.38–2.40 | 2.28–2.31 | 2.54–2.58 |

At a stage threshold of 1024 a 1351-element SSFS stage regressed (1.2 to 1.7 ms); the
reduced comb-to-continuum and SSFS cases did not move at 4096. On a dual EPYC 7643 (Zen 3,
idle node, pinned) the same change made n_t 4000 24-27 % faster at 12 and 24 threads
(2.36-2.42 to 1.74-1.82 ms), and the even chunks made n_t 8640 4-13 % faster; other sizes
moved within 5 %. On an 18-core Apple arm64 laptop, in contrast, n_t 4000 went from 0.88
to 1.04 ms (level with 4096), and even 4096's threaded stage is dearer than 4000's serial
one: the crossover there lies above 8192. macOS therefore keeps the stage and FFT
starts at 8192 and 32,768; its own crossover still needs a sweep on a quiet machine. Its
node-chunk gain stands (n_t 8640: 2.35 to 1.73 ms).

(radial-chunk-dispatch)=
### Radial analytic chunk dispatch

Admitting partial analytic radial node chunks to the interleaved real GEMM, instead of a
scalar contraction that took about 94 % of the source time, made the radial filament's
adaptive Adams window (256 and 512 nodes, 12 accepted steps, construction excluded) 2.8
to 8.8 times faster on Kokkos CPU, on an exclusively allocated two-socket AVX-512 server.
At one thread Kokkos then took 0.91 (256 nodes) and 0.95 (512) times `numba_cpu`'s time,
at 36 threads 0.10 and 0.12 times.

(multimode-cpu-step)=
### Multimode CPU step costs

Sharing real cubic monomials across output components, planning general-modal tail
transforms over active rows only and seeding serial transforms with measured FFTW wisdom
brought Kokkos CPU to 1.00–1.07 times `numba_cpu`'s best time at one thread on an arm64
laptop, over 1–2 mm entrance windows of the general modal and reduced
comb-to-continuum cases with both adaptive methods. In ten-step profiles the general
route's tensor work fell from 0.44 to 0.15 s and its ordinary FFTs from 1.00 to 0.84 s.
Measured FFTW planning adds about 18 ms (general modal) and 120 ms (reduced
comb-to-continuum) of construction. A direct FFTW execution adapter, a pooled allocator
and vector transcendental functions were measured and not needed
([CPU performance](cpu-performance.md)).

(multimode-threaded-planning)=
### Threaded FFT plan selection

FFTW's measured planning sometimes chose a slower threaded plan: on a 32-core AVX-512
server, for 204 rows of 2048 complex samples at 32 threads, a 64 × 32 decomposition
instead of 32 × 64 made the warm general-modal RK window about 12 % slower, almost all of
it in complex transforms. Measured wisdom is now seeded only for serial transforms;
threaded ones default to estimate planning. Against the previous build, the general modal
case's window then took about 0.80 times as long at one thread and the same at 32
threads; reduced comb-to-continuum stayed within its repeatability.

### Low-storage interaction-picture RK

The low-storage march (four evaluations a step) against the IP-DOPRI5 replay (six, with
first-same-as-last) on a Cartesian envelope Kerr beam with no collar, fixed 0.5 µm steps,
ten timed steps after one untimed one, same-process comparisons on an 18-core arm64
workstation. Shape A is 192 × 192 × 300 (one state 177 MB), shape B 384 × 384 × 600
(1.42 GB). Ratios are how many times less time the march takes than the replay on the
same backend:

| backend | shape, threads | per step | per evaluation | field volumes held, march / replay |
|---|---|---:|---:|---:|
| `numba_cpu` | A, 1 | 1.83× | 1.22× | 4 / 16 |
| `numba_cpu` | A, all | 1.77× | 1.18× | 4 / 16 |
| `numba_cpu` | B, all | 1.73× | 1.15× | 4 / 16 |
| Kokkos CPU | A, 1 | 2.02× | 1.35× | 4 / 15 |
| Kokkos CPU | A, all | 1.81× | 1.21× | 4 / 15 |
| Kokkos CPU | B, all | 1.85× | 1.24× | 4 / 15 |

Recomputing the flow factors instead of tabulating them holds three volumes and costs
3–13 % more. Most of a march step is now the source evaluation's core (transforms, the
Kerr column, packing): 69 % on `numba_cpu` at one thread, the fused finish 27 %. At all
threads the Kokkos march was 1.6 (shape A) to 2.1 (shape B) times faster than
`numba_cpu`'s.

### Band-rule plasma integration

The band-exact plasma integrals
([SEP 18](seps/archive/sep-0018-band-exact-plasma-integrals.md)) against the trapezoid, on
the four-mode RDW window (33 nodes, 8192 fine samples, ten fixed steps from the launch),
for the band column as first built:

| backend | threads | band / trapezoid window |
|---|---|---:|
| `numba_cpu` (arm64 laptop) | default | 1.51–1.58 |
| `numba_cpu` (arm64 laptop, or the AVX-512 server, where this column ran serially) | 1 | 2.4–2.5 |
| Kokkos CPU (arm64 laptop) | 18 / 4 / 1 | 1.3 / 1.9 / 2.7 |
| Kokkos CUDA (data-centre NVIDIA GPU), direct halo reads | — | 1.10 |
| Kokkos HIP (AMD Instinct GPU), halo staged in team scratch | — | 1.25 |

On a CPU each integral is a nine-tap filter inside a serial traversal; on a device the
filters are parallel maps over the row, so the band rule costs about a tenth more than
the trapezoid there. Staging the halo in team scratch loses on the NVIDIA device (1.21)
and wins on the AMD one (1.36 with direct reads), so each build defaults to its winner.
`numba_cpu`'s column has since been rewritten as whole-row passes, 1.7 times faster than
the traversal measured here ([algorithms.md](algorithms.md)). A finer grid, the
alternative SEP 18 rejects, costs about twice the whole nonlinear term for a quarter of
the error.

### Dispersive electronic response

The dispersive Kerr response ([SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md),
[backends.md §3.11](backends.md#311-the-dispersive-electronic-response)) is judged by its
nonlinear stage time against the constant cubic's on the same route: the symmetric
evaluator's target is 1.3, the polynomial's 4. Same-process medians of ten-step windows
from the launch, on the RDW grid and on Cartesian, line, planar and general-route
problems:

| evaluator | `numba_cpu` | Kokkos CPU | GPUs (NVIDIA and AMD) |
|---|---|---|---|
| symmetric, one species | 0.92–1.23 (1.31–1.35 on one averaged node, analytic sector) | 0.81–1.21 at one thread | 0.86–1.30 |
| symmetric, mixture (one filter from the mixture's Kerr diagonal) | 0.90–1.11 | 0.97–1.21 at one thread | 0.86–1.30 |
| polynomial, one species or equal leg bounds | 1.16–3.72 | 1.52–3.65 at one thread; 4.2–5.0 on the general route | 1.01–3.53 |
| polynomial, mixture of distinct leg bounds | 4.1–7.6 (3.3–3.7 for helium and argon in the default pools) | 3.2–8.3 at one thread, at most 2.2 in the default pools | within 4 (air 1.5–2.1) |

Ten-step windows, which carry the integrator's own work, move less than the stage. A
rebuild of the mixture filter when the densities change costs about 2 % of a symmetric
stage. A polynomial mixture is one group per distinct leg bound, which is why mixtures
such as air miss its target on the CPUs.

### Far-field statistics pass

The banded far-field pass (six aperture statistics and the on-axis spectrum, 30 slots)
on the last state of a Cartesian envelope Kerr run (192 × 192 × 256 and 384 × 384 × 512),
same-process comparisons: about 2.3–2.6 % of one Runge–Kutta step on both `numba_cpu`
and Kokkos CPU, paid at save planes and periodic rows only. On `numba_cpu` it costs about
as much as the `energy` statistic and is 30 to 37 times faster than the host pass it
replaced (which cost 77–84 % of a step). On Kokkos CPU it is compute-bound (square roots
and saturated `tanh` weights), 15–16 times `energy`.

### Raman kernel rebuild

On a density gradient the Raman kernel is rebuilt at every stage, and its cost scales with
the response's rows. `numpy_ref`'s rebuild at $2^{14}$ fine samples (median of 35,
same-process, a loaded laptop) took 0.31 ms for a two-term N₂ model (rotation and
vibration), 0.47 ms for optmat's H₂ (4 rows), 6.0 ms for optmat's N₂ (72 rows) and
10.2 ms for dry air (126 rows), against 0.25 ms for one convolution. With tens of rows
the rebuild dominates the step; Kokkos's native rebuild is slower than the host's at one
thread and overtakes it at four. [SEP 10](seps/sep-0010-low-rank-raman.md)'s low-rank
representation is what removes it.

### General-route symmetric contraction

Summing the general route's real cubic over its ten symmetric monomials instead of the
dense 81 + 3 products ([algorithms.md](algorithms.md)), on a two-channel step-index pair
(288 nodes, 256 samples) at one thread: the respond kernel alone 2.5 times faster on
`numba_cpu`; a ten-step window 1.45 times faster on `numba_cpu` and 1.35 times on Kokkos
CPU. The gain shrinks as channels are added.

### Adams carrier frame per step

The scalar carrier frame ([SEP 35](seps/archive/sep-0035-carrier-frame-exponential-adams.md))
adds, per trial, a shifted argument and one complex product per element in the moments
pass and one complex product per history row read. Ten replayed steps of a 2 mm,
40 bar hydrogen capillary (16 µm, 515 nm, 2 ps window, 4097 bins, one mode, a weak
collar), the stepping alone timed, one thread, median ratio of nine same-process
interleaved pairs on an Apple M5 Max, a machine in ordinary use (labelled A/B, not an
absolute timing):

| source | `numba_cpu` off → scalar | ratio | native CPU off → scalar | ratio |
|---|---|---|---|---|
| Kerr only | 11.9 → 12.4 ms | 1.042 [1.019, 1.096] | 9.2 → 9.7 ms | 1.057 [1.036, 1.066] |
| Kerr and Raman | 121.5 → 121.9 ms | 1.007 [0.995, 1.017] | 95.4 → 95.8 ms | 1.006 [0.997, 1.009] |

Forming `exp(x - sigma)` as `exp(x) exp(-sigma)` instead of a second exponential took
native's Kerr-only ratio from 1.076 to 1.057. The per-step cost is a few per cent where
the source is cheap and negligible where it is not; the policy's case rests on the calls
it saves, which SEP 35's phase 4 measures at matched error.

(resident-adams-memory)=
### Resident Adams storage

At maximum order five the Kokkos resident Adams engine held 29 complex field volumes,
eleven of them dense-output slopes and saved fields allocated even when planes are landed
on. Allocating those only for `interpolate_planes=True` reduces it to 18
([backends.md §8](backends.md#8-memory-ledger)). Peak RSS of a reduced-resolution
`kerr_beam_384` (same box, 600 temporal bins, ten accepted steps, four threads, median of
three fresh processes on an 18-core arm64 laptop), in GiB; the 384² row is a projection
from the other two (constant plus transverse area), not a measurement:

| grid | `numba_cpu` | Kokkos, 29 volumes | Kokkos, 18 volumes | 18 volumes / `numba_cpu` |
|---|---:|---:|---:|---:|
| 128² | 4.31 | 7.53 | 5.92 | 1.37 |
| 256² | 15.7 | 28.2 | 21.8 | 1.38 |
| 384², projected | 34.8 | 62.7 | 48.2 | 1.39 |

Field export adds a further host copy (up to 1.43 times `numba_cpu`'s peak). No
field-sized workspace scales with the thread count. The allocations are itemised under
[reference tables](#resident-adams-allocations).

### Free-space cases at the package defaults

Per-step time of the free-space cases at the package's defaults, production thread
policy, on a 64-core AVX-512 cloud server (seconds per step, medians of two repeats):

| case | stepper | Kokkos CPU | `numba_cpu` |
|---|---|---:|---:|
| Kerr beam 384² | march | 0.348 | 0.796 |
| Kerr beam 384² | IP-DOPRI5 | 0.998 | 2.49 |
| Kerr beam 768² | march | 1.51 | 3.34 |
| Kerr beam 768² | IP-DOPRI5 | 4.09 | 10.1 |
| Cartesian filament 128² | default | 0.076 | 0.300 |
| Cartesian filament 256² | default | 0.296 | 1.19 |

The march scales from 384² to 768² as 4.33 (Kokkos) and 4.20 (`numba_cpu`) against
$N \log N$'s 4.30. Against a per-evaluation reference built from the cost of its parts
(transforms and response), the march's evaluations take 0.94–1.37 times the reference and
IP-DOPRI5's 1.21 (Kokkos) and 1.80 (`numba_cpu`)
([SEP 28](seps/archive/sep-0028-free-space-evaluation.md)).

### The boundary monitor's reading

Each backend reduces the boundary monitor's five numbers where its state lives, instead
of copying the field to the host to be transformed there (ADR-057). The benchmark
harness runs without a monitor, so the monitor's cost is measured by its own same-process
comparison: each case built by the harness's builder and run through `Simulation.run`
with the default check and with `boundary_check=None`, alternated, five times each
after a warm-up of each, the shortest of each arm compared.

**Monitored over unmonitored run time**, native CPU, one thread, on a two-socket AMD
EPYC 7763 node (GCC 16.2, FFTW 3.3.11, Kokkos 5.2.1 OpenMP); before is the parent
revision, after is this change:

| case, window | accepted steps | before | after |
|---|---:|---:|---:|
| Kerr beam 192², the whole 2 µm guide | 4 | +7.8 % | +4.0 % |
| radial filament 256, 3 mm | 68 | +0.9 % | +0.6 % |
| GNLSE soliton, a fifth of the guide | 81 | +5.5 % | +1.5 % |

The Kerr beam's guide is shorter than its steps times the monitor's 256 checks, so every
step is checked. A check now costs one inverse transform of the state where it lives
and a pass that sums it; it was three host copies and a single-threaded host transform.
**One check** of the launched state, the backend's reading against that host path, same
process, shortest of seven:

| case | EPYC 7763, one thread | NVIDIA A40 | NVIDIA L4 |
|---|---:|---:|---:|
| Kerr beam 192² (36,864 columns × 384) | 100 ms / 218 ms | 3.4 ms / 214 ms | 5.0 ms / 511 ms |
| radial filament 256 (256 × 864, field) | 1.08 ms / 1.22 ms | 0.13 ms / 1.03 ms | 0.14 ms / 4.3 ms |
| GNLSE soliton (one column × 3072) | 0.027 ms / 0.071 ms | 0.082 ms / 0.085 ms | 0.072 ms / 0.095 ms |

The GPU columns are CUDA 12.8 builds (an A40 on its node's host, an L4 on an 8-vCPU
cloud host, whose host path is the slower). On a field grid under the split scheme the
native reading uses the collar's inverse real transform, half the analytic signal's
work; through the analytic signal the radial filament's check took 1.83 ms against the
host path's 1.13 ms at one thread. The L4's column was measured before that change. On `mlx`
(Apple M5 Max, a busy laptop: ratios only, same process), monitored over unmonitored on
the spatial cases of 60 fixed steps: radial 256 1.36 → 1.01, Cartesian 64² 1.21 → 1.05,
Cartesian 128² 1.28 → 1.06. The unmonitored path is unchanged:
`python -m benchmarks ab` at one thread on the same node gives 1.000 (GNLSE soliton),
1.005 (radial filament) and 1.005 (Kerr beam) for the three cases above.

### Host vector maths

The native CPU kernels' transcendentals in vector passes (ADR-059) against the parent
revision, complete runs of the tracking cases from the entrance (the fraction of the
guide in brackets where it is not the whole), the default stepper and `RungeKutta()`,
three rounds of interleaved fresh processes, medians with their median absolute
deviations. Every arm took the same steps; final fields differ from the parent's by
4e-15 to 6e-11 relative, the run-to-run spread of FFTW's measured plans (a candidate
with this change's arithmetic on the libm route differs as much).

**Two-socket AMD EPYC 7763** (Zen 3, AVX2; 16 cores of a shared node, glibc 2.34, GCC
16.2, FFTW 3.3.11, OpenBLAS 0.3.34, Kokkos 5.2.1 OpenMP, a local build, so SLEEF's AVX2
functions), one thread, this change's final revision, two rounds:

| case | stepper | before (s) | vector maths |
|---|---|---:|---:|
| `gnlse_raman_shock` (0.3) | RK | 18.84 ± 0.02 | −19.4 % |
| `argon_blueshift_averaged` | RK | 2.947 ± 0.015 | −17.2 % |
| `argon_blueshift_averaged` | default | 8.235 ± 0.003 | −6.2 % |
| `he_rdw_averaged` | RK | 1.365 ± 0.003 | −12.4 % |
| `he_rdw_averaged` | default | 2.338 ± 0.009 | −10.5 % |
| `h2_raman_averaged` (0.5) | RK | 2.866 ± 0.002 | −10.1 % |
| `rdw` (0.3) | RK | 7.679 ± 0.012 | −14.4 % |
| `rdw` (0.3) | default | 5.313 ± 0.012 | −11.4 % |
| `radial_filament_256` (0.3) | RK | 8.915 ± 0.010 | −7.8 % |
| `radial_filament_256` (0.3) | default | 6.261 ± 0.025 | −5.9 % |
| `argon_blueshift_envelope` | RK | 1.665 ± 0.006 | −5.7 % |
| geometric mean | | | **−11.1 %** (conservative 0.896) |

An earlier revision, three rounds, with a third arm built with
`SOLPROP_NATIVE_VECTOR_MATH=SCALAR` (this change's arithmetic on the libm route, an
A/A-like control): that arm's geometric mean was +0.7 % and its cases −1.2 to +2.2 %.

The free-space routes, the same machine and protocol, one thread, with the low-storage
flows and the split factor: `spatial_absorber` (0.5) −12.2 %, `radial_filament_256`
(0.3) RK −5.3 % and default −1.8 %, `planar` (0.5) −1.5 %, `line` (0.5) −1.1 %,
`kerr_beam_192` (0.5) −0.9 % (inside its spread); geometric mean −3.9 %. Sixteen-thread
runs on that node, which other jobs shared, scattered by more than the effect (one
case's arms spread by 30 %), and are not reported.

**AMD EPYC 9B45** (Zen 5, AVX-512; a cloud instance of 16 cores without SMT, glibc
2.39, GCC 14.2, Ubuntu's FFTW 3.3.10 and OpenMP OpenBLAS 0.3.26), local builds at
`-march=native`; the AVX2 rows with `-mno-avx512f` too, so the AVX2 base is the
reference. One thread, three rounds:

| case | stepper | before, AVX2 (s) | vector maths, AVX2 | AVX-512F (opt-in) |
|---|---|---:|---:|---:|
| `gnlse_raman_shock` (0.3) | RK | 10.42 ± 0.03 | −21.3 % | −29.2 % |
| `argon_blueshift_averaged` | RK | 1.581 ± 0.003 | −10.9 % | −15.8 % |
| `argon_blueshift_averaged` | default | 4.424 ± 0.004 | −6.2 % | −13.1 % |
| `he_rdw_averaged` | RK | 0.765 ± 0.011 | −6.2 % | −8.7 % |
| `he_rdw_averaged` | default | 1.295 ± 0.014 | −10.8 % | −18.1 % |
| `h2_raman_averaged` (0.5) | RK | 1.546 ± 0.003 | −10.3 % | −14.7 % |
| `rdw` (0.3) | RK | 3.873 ± 0.012 | +4.9 % | +2.2 % |
| `rdw` (0.3) | default | 3.123 ± 0.001 | −2.5 % | −6.6 % |
| `radial_filament_256` (0.3) | RK | 5.026 ± 0.012 | −4.3 % | −6.1 % |
| `kerr_beam_192` (0.5) | default | 5.071 ± 0.036 | +2.0 % | +0.5 % |
| geometric mean | | | **−6.8 %** (conservative 0.945) | −11.4 % (0.900) |

The AVX-512 base was itself 2.0 % faster than the AVX2 base, so AVX-512F's functions
save 9.8 % against it. At sixteen threads, the AVX2 builds, two rounds, six cases:
geometric mean −5.3 % (conservative 0.983), `argon_blueshift_averaged` −10.9 % (RK) and
−11.3 % (default), `he_rdw_averaged` −6.9 % (RK) and −1.5 % (default), `rdw` −0.9 %
(RK) and +0.2 % (default). **An Intel Xeon of the Ice Lake generation** (a cloud instance of
16 cores without SMT, the same software), one to two rounds and noisier: the AVX2
functions −3.7 %, the AVX-512F functions +2.9 % (`he_rdw_averaged` with RK +21 %,
`rdw` with RK +12 %). The AVX2 functions are therefore the default, AVX-512F an opt-in.

`rdw` with RK lost 3 to 5 % on both cloud machines, against a 7 % gain on the EPYC
7763, whose glibc (2.34) is older and slower. The label timer put it in the band plasma
column (1.92 s → 2.21 s), whose block passes staged every sample's magnitude, rate and
fraction: most of the pulse's samples lie outside the rate table and cost about a
nanosecond each, as much as the staging. Writing the rows as the classifying loop goes
and gathering only the live samples for the vector calls took the Zen 5 machine's
`rdw` with RK to −4.7 % (from +4.4 %), `argon_blueshift_averaged` to −14.1 % (RK) and
−6.8 % (default) and `he_rdw_averaged` with RK to −11.4 %, three rounds; the tables
above predate that change.

**Apple M5 Max**, a machine in ordinary use (labelled A/B), one thread: this change's
default (the system libm, each element's own calls) against the parent, geometric mean
+0.4 % over eight of the cases above, every case within its spread; SLEEF's AdvSIMD
functions forced on (`SOLPROP_NATIVE_VECTOR_MATH=SLEEF`) +11.6 %, `gnlse_raman_shock`
+56 %.

### Exponential Adams host cost: the SEP 39 baseline

The host side of an exponential `Multistep()` trial on the native CPU backend, before
[SEP 39](seps/sep-0039-native-adams-march.md)'s native march, at the head of the vector
maths change (a500d2d). The SEP's §3.3 reads these figures.

**Method.**

- Complete runs of the tracking cases, with the fraction of the guide in brackets
  where it is not the whole, timed by the suite's worker after its warm-up.
- The default stepper and `RungeKutta()` interleaved over three rounds of fresh
  processes, the order reversed every round; medians reported.
- Host time per trial: `python -m benchmarks.adams_host CASE [--length F]` (SEP 39),
  the march's wall time less its native `adams_*` calls, with no other wrappers.
- The per-piece split comes from one run with `perf_counter` wrappers on about
  twenty-five host functions, which inflate the total by about half, so it gives the
  shares, not the sum.
- Steps and calls are counted.

**AMD EPYC 7763** (Zen 3, 16 cores of one NUMA domain of a shared node, GCC 16.2,
SLEEF's AVX2 functions, FFTW 3.3.11, OpenBLAS 0.3.34), one thread:

| case | `Multistep()` (s) | `RungeKutta()` (s) | host / native per trial (µs) | steps (acc/rej), calls: MS · RK |
|---|---:|---:|---:|---|
| `he_rdw_averaged` | 2.126 | 1.528 | 212 / 1 182 | 1 434/3, 1 438 · 480/1, 2 887 |
| `argon_blueshift_averaged` | 9.17 (7.83–9.47) | 2.460 | 177 / 632 | 11 166/2, 11 169 · 1 684/0, 10 105 |
| `h2_raman_averaged` (0.4) | 4.820 | 2.146 | 173 / 695 | 5 417/2, 5 420 · 1 308/0, 7 849 |
| `rdw` (0.25) | 4.157 | 5.440 | 372 / 9 245 | 393/1, 395 · 175/0, 1 051 |

The node's sixteen-thread runs, which other jobs shared, were slower than one thread on
several cases and are not reported.

**Apple M5 Max** (Homebrew Clang 23.1.3, FFTW 3.3.11 with NEON codelets, the system
libm, an otherwise idle machine):

| case | threads | `Multistep()` (s) | `RungeKutta()` (s) | host / native per trial (µs) |
|---|---:|---:|---:|---:|
| `he_rdw_averaged` | 1 | 0.815 | 0.650 | 51 / 448 |
| `argon_blueshift_averaged` | 1 | 2.920 | 1.419 | 38 / 213 |
| `h2_raman_averaged` (0.4) | 1 | 1.755 | 1.311 | 40 / 263 |
| `rdw` (0.25) | 1 | 1.711 | 2.736 | 129 / 3 730 |
| `he_rdw_averaged` | 16 | 0.636 | 0.650 | 53 / 282 |
| `argon_blueshift_averaged` | 16 | 2.474 | 1.381 | 36 / 175 |
| `h2_raman_averaged` (0.4) | 16 | 1.697 | 1.652 | 44 / 254 |
| `rdw` (0.25) | 16 | 0.862 | 1.256 | 205 / 1 555 |

The step and call counts are the EPYC's on both machines.

**Host pieces, helium, one thread** (µs per trial, instrumented):

| piece | EPYC 7763 | M5 Max |
|---|---:|---:|
| stage arrays (bundle, packing, sample) | 113 | 21 |
| … of which the bundle's Clenshaw reads | 73 | 14 |
| the decision, less the native accept | 48 | 7 |
| rate weights (uncached Clenshaw) | 26 | 4 |
| landing and approach | 30 | 5 |
| the tables call (one native helper) | 18 | 4 |
| order choice and stencil | 17 | 3 |
| sample, hoist, record and the rest | 22 | 4 |

On the four-mode `rdw` the step history's record costs 67 µs a trial on the EPYC: its
backend hook scores the quadrature alarm natively.

**Native time per trial** (µs, host-only Kokkos label timer, one thread):

| case | machine | predict pass | correct pass | the source and the rest |
|---|---|---:|---:|---:|
| `he_rdw_averaged` | EPYC 7763 | 557 | 287 | ≈290 |
| `argon_blueshift_averaged` | EPYC 7763 | 192 | 112 | ≈200 |
| `he_rdw_averaged` | M5 Max | 169 | 100 | ≈155 |
| `argon_blueshift_averaged` | M5 Max | 51 | 34 | ≈110 |

On the EPYC the two passes cost 1.5–2.9 source evaluations a trial, against 0.6–0.9 for
the host. Their per-element cost is 2.9–3.8 times the M5's, against about 1.9 times for
the source.

### The native Adams march

[SEP 39](seps/sep-0039-native-adams-march.md)'s native march against solprop's Python
march on the same build (`benchmarks.adams_host --host-march`), and against the march
with the host packing the stage rows, as before the SEP's phase 1 (`--host-stages`).
Interleaved fresh processes, the arm order reversed every round, medians; every arm
took the same steps and calls (the counts above). Host time per trial as in the
baseline above.

**Exponential formulation** (the default `Multistep()`):

| machine, threads | case | host µs/trial: packed → staged → native | wall, native vs staged | wall, native vs packed |
|---|---|---|---:|---:|
| M5 Max, 1 | `he_rdw_averaged` | 43 → 13 → 0.6 | −3.3 % | −7.6 % |
| | `argon_blueshift_averaged` | 30 → 11 → 0.4 | −6.0 % | −13.6 % |
| | `h2_raman_averaged` (0.4) | 37 → 11 → 0.5 | −3.9 % | −11.6 % |
| | `rdw` (0.25) | 91 → 47 → 2.7 | −0.6 % | +0.4 % |
| M5 Max, 16 | `he_rdw_averaged` | 47 → 13 → 0.6 | −4.2 % | −13.2 % |
| | `argon_blueshift_averaged` | 32 → 12 → 0.4 | −9.2 % | −17.2 % |
| | `h2_raman_averaged` (0.4) | 39 → 12 → 0.6 | −6.2 % | −16.2 % |
| | `rdw` (0.25) | 171 → 122 → 1.9 | −2.2 % | −4.5 % |
| EPYC 7763, 1 | `he_rdw_averaged` | 208 → 65 → 1.7 | −6.1 % | −14.0 % |
| | `argon_blueshift_averaged` | 148 → 58 → 1.1 | −11.5 % | −24.2 % |
| | `h2_raman_averaged` (0.4) | 168 → 58 → 1.2 | −9.5 % | −22.7 % |
| | `rdw` (0.25) | 168 → 8 (staged → native) | −1.7 % | — |

The EPYC's first three rows are from a shared node (load 80–97), whose one-thread
walls were consistent; its `rdw` row is from a quiet node, nine rounds, where the
sixteen-thread median read +2.7 % and the minimum −3.3 % inside a ±6 % spread.

**Interaction formulation** (`Multistep(formulation="interaction")`, M5 Max):

| threads | case | host µs/trial: packed → staged → native | wall, native vs staged | wall, native vs packed |
|---|---|---|---:|---:|
| 1 | `he_rdw_averaged` | 41 → 12 → 0.5 | −4.0 % | −11.8 % |
| | `argon_blueshift_averaged` | 33 → 11 → 0.5 | −5.4 % | −16.7 % |
| | `h2_raman_averaged` (0.4) | 40 → 11 → 0.5 | −5.6 % | −18.3 % |
| | `rdw` (0.25) | 85 → 43 → 1.5 | −1.3 % | −2.9 % |
| 16 | `he_rdw_averaged` | 41 → 12 → 0.5 | −3.9 % | −12.9 % |
| | `argon_blueshift_averaged` | 33 → 11 → 0.4 | −8.6 % | −18.2 % |
| | `h2_raman_averaged` (0.4) | 42 → 11 → 0.5 | −34.1 % | −15.0 % |
| | `rdw` (0.25) | 165 → 119 → 1.5 | −2.1 % | −4.7 % |

The staged Python march's sixteen-thread `h2_raman_averaged` (2.24 s, slower than the
packed march's 1.74 s, in every round) is unexplained; see SEP 39's phase 3 report.

**On an NVIDIA A40** (CUDA 12.8.1; exponential formulation, short windows; the same
steps in both arms): host time per trial 64 → 2.2 µs (`he_rdw_averaged`, 0.3), 61 → 1.6
(`argon_blueshift_averaged`, 0.1) and 83 → 5.0 (`rdw`, 0.1), walls −6.7 %, −10.6 % and
−1.8 %. A trial costs about a millisecond on this FP64-weak device.

### The Adams passes on x86-64 lanes

The exponential Adams trial's predict and correct passes on a host block of 32 elements
held as lanes (`include/solprop_native/adams_phi.hpp`, `adams_lanes_*` in
`src/general.cpp`) against the element-by-element form, at 740b24d. Before the change
the passes ran each element through latency-bound scalar strict code (the
phi-functions' series, Horner pass and squarings, the compensation and the defects),
GCC calling every strict helper out of line from the contracting kernels; per element
they cost 2.9–3.8 times the Apple M5 Max's on the AMD EPYC 7763 (§ "Exponential Adams
host cost" above). Now every lane loop vectorises on x86-64-v3, Zen 3 and Cascade Lake
(checked with GCC 16.2's vectoriser report): the selections are bitwise, the
comparisons quiet, and the corrector's endpoint rate is formed beside the step's
phase, in one pass over the operator's terms. The strict parts keep each element's
operations and their rounding (a native test holds the lanes' phi-functions to
`adams_phi`'s bits); the predictor's and corrector's sums may contract differently, so
the first step whose inputs differ differs by about 2e-16 in its norm. Every case
below took the same steps and trials in both arms; final fields differ by 2e-15 to
5e-12 relative (`he_rdw_averaged` at 0.3 of its length: 4.6e-12, the step sizes
apart by up to 5e-5 relative after the pulse's compression amplifies the rounding).

**AMD EPYC 7763** (16 cores of a shared node, GCC 16.2, SLEEF's AVX2 functions,
OpenMP OpenBLAS), one thread, three interleaved rounds, medians; the passes in µs per
call (host-only Kokkos label timer), the wall of the timed window:

| case (fraction) | predict | correct | wall |
|---|---:|---:|---:|
| `he_rdw_averaged` (0.3) | 562 → 273 | 293 → 109 | −32 % |
| `argon_blueshift_averaged` (0.1) | 190 → 82 | 112 → 45 | −30 % |
| `h2_raman_averaged` (0.2) | 269 → 115 | 164 → 65 | −37 % |
| `rdw` (0.1) | 2430 → 1373 | 1222 → 530 | −16 % |
| `radial_filament_256` (0.05) | 13 470 → 7 050 | 9 890 → 4 430 | −16 % |
| `argon_blueshift` (0.05) | 189 → 83 | 111 → 48 | −7 % |

Per element the predict pass now costs about 1.6 times the M5 Max's figure of § 3.3
of SEP 39, and the correct pass about 1.1 times. A stage clock (rdtsc around each lane
step, a profiling build) puts the rest of `he_rdw_averaged`'s predict pass at about a
third in SLEEF's six complex exponentials an element (the flow, the series argument and
four spans), a sixth in the 18-term phase and rate sum, which is memory-bound, an eighth
in the phi-functions and the rest in the history sums and stores. That node's
sixteen-thread runs scattered by more than the effect and are not reported.

**Intel Xeon Gold 6240** (18 physical cores of one socket, the other socket in use, GCC
16.2 `-march=cascadelake`, SLEEF's AVX2 functions, OpenMP OpenBLAS), complete windows at
0.1 of each case's length, `python -m benchmarks ab` with passive bound threads, three
rounds, medians of the paired ratios (range over rounds); every pair did the same work,
final fields within 6.2e-15:

| case | one thread | sixteen threads |
|---|---:|---:|
| `he_rdw_averaged` | 0.786 (0.779–0.809) | 0.879 (0.775–0.885) |
| `argon_blueshift_averaged` | 0.752 (0.747–0.759) | 0.946 (0.920–1.057) |
| `h2_raman_averaged` | 0.697 (0.696–0.710) | 0.936 (0.921–0.952) |
| `rdw` | 0.827 (0.816–0.829) | 0.974 (0.961–0.974) |
| `radial_filament_256` | 0.839 (0.836–0.849) | 0.937 (0.932–0.945) |
| `argon_blueshift` | 0.936 (0.934–0.963) | 0.953 (0.946–0.975) |

The geometric mean of the twelve ratios is 0.867. At sixteen threads the single-column
passes run on four threads (the size class above), so less of the gain shows.

SLEEF's AVX-512F functions (`SOLPROP_NATIVE_VECTOR_MATH_AVX512`, opt-in) took a further
5–8 % off these windows at one thread on the Xeon, and were 6 % slower on
`he_rdw_averaged` at sixteen threads (one round); they stay opt-in.

Tried and not adopted: blocks of 16 elements (5–15 % slower on the EPYC) and of 64
(within 2 % of 32 on single columns, 5 % faster on `rdw`).

### Device transfers and the free-space Kerr stage

On an NVIDIA A40 (CUDA 12.8.1, `AMPERE86`, binary64 at about a sixty-fourth of the
single-precision rate), `python -m benchmarks ab --device`, three rounds of interleaved
fresh processes, half of `kerr_beam_384` (two fixed RK4IP steps, 384² × 750):

| change | wall, s | ratio (range) | field difference |
|---|---|---:|---:|
| one copy at a load and an observation, the free-space split on the stored columns | 4.775 → 3.878 | 0.812 (0.812–0.815) | 1.0e-15 |
| the split factor per sample, the pinned single upload | 4.686 → 3.892 (with the above) | 0.830 (0.829–0.832) | 1.0e-15 |
| the rank-3 Kerr stage | 3.913 → 3.764 | 0.963 (0.957–0.964) | 1.3e-16 |
| all, against the baseline | 4.732 → 3.747 | 0.792 (0.789–0.798) | 1.0e-15 |

Kernel time fell from 827 to 651 ms a step: the split's transverse pair and transposes
(about 100 ms) and the chunks' pack, crop and pointwise passes (about 75 ms). An
observation of the 1.77 GB state took 0.85 s of host time through three fresh buffers.
On an Intel Xeon Gold 6240 at one thread `kerr_beam_192` (half) took 0.920 of the time
(0.917–0.922, fields to 6e-16). `radial_filament_256` is unchanged on the A40 (0.3 and 0.5
of its length, within ±2 %): its trial, 3.3 ms of kernels, is bound by binary64
arithmetic (the synthesis and projection GEMMs near the device's binary64 peak, the
φ-functions), so neither the single upload, the GEMM for its node chunks nor one chunk
in place of two (48 → 31 kernels a trial, not adopted) moved it. The retained-spectrum
synthesis does, by halving the synthesis's product: on the A40, half the run, three
rounds of three repeats, 0.569 → 0.512 s, ratio 0.902 (0.898–0.903), fields to 3.2e-15,
kernel time 238 → 200 ms over a 71-trial window.
The device prefactor with one reciprocal in place of two divisions took the separable
fold from 29.5 to 21.6 ms (`kerr_beam_384`, kernels 651 → 620 ms a step, wall 0.981,
0.936–0.984, fields to 2.4e-17).

### Retained radial synthesis and Raman modal products

Two work reductions from the CPU experiment's handoff, each measured with
`python -m benchmarks ab` (fresh interleaved processes, passive-bound threads,
`SOLPROP_FFTW_PLAN=measure`, three rounds, one repeat; medians of the per-round pairs,
range over rounds). Each case exercises one of them: the radial filament has no Raman
term, and the capillary is not a square radial map. Work counts are equal in every pair
and the final fields agree to 4–10 × 10⁻¹⁵ relative.

- **Retained-spectrum radial synthesis** (square radial analytic maps,
  [backends.md](backends.md)): the synthesis product covers `2 n_omega` real columns
  instead of `2 n_to`. `radial_filament_256`, complete guide: on an Intel Xeon Gold 6240
  (one socket, 18 cores) 0.928 [0.925, 0.945] at one thread and 0.954 [0.936, 0.968] at
  16; on 16 cores of an AMD EPYC 7763, 0.869 [0.868, 0.891] at one thread, and at
  `--length 0.05` 0.866 [0.850, 0.868] at one thread and 0.825 [0.653, 0.894] at 16. The
  EPYC's 16-thread ratios moved by about ±15 % between allocations on a shared node (a
  complete-guide repeat gave 0.999 [0.735, 1.020]); the Xeon's are the resolved ones.
  A 16-thread label profile on the EPYC put the synthesis GEMMs at 0.20 s against
  0.38 s, of which the gather and pad, as first written per chunk and over every fine
  bin, took back 0.14 s; they now gather once per evaluation and write only the retained
  bins.
- **Raman modal products** (three-mode `ctc`, 6 product rows against 33 node drives):
  `--length 0.05` at 16 threads 0.884 [0.881, 0.899] on the Xeon; on the EPYC 0.833
  [0.823, 0.837] at one thread and 0.921 [0.861, 0.928] at 16 (`--length 0.02`: 0.850
  and 0.884). The first reconstruction looped over the products per sample and cost ten
  times the node path's finish (0.46 s against 0.046 s in a one-thread window); it now
  accumulates each product over contiguous blocks of 512 samples.

Both add FFT plans (the node transforms; the product rows), measured at construction:
on the Xeon a CTC worker's wall time, construction included, rose from 16.3 s to 17.7 s.

Not adopted, on the same EPYC allocation:
- Routing every real product with a left matrix of at most 256 entries through
  `native_order_gemm` instead of the vendor BLAS: RDW 1.012 / 0.881, argon 1.057 /
  1.099, CTC 1.070 / 1.377 at one / 16 threads (`--length 0.05`). The vendor library
  wins on the serial transform spaces the operation-size policy gives these shapes.
- Plain row-block loops for one-mode synthesis and projection: argon 0.996 at one
  thread; at 16 the loops took 0.65–0.69 s of a short window where the vendor GEMMs took
  0.55 s. The vendor pool still costs 0.55 s against 0.23 s at one thread there.
- Planning the product rows at the full Raman chunk's width rather than their own:
  CTC 1.001 / 0.980 [0.952, 1.012].

## Reference tables

These tables back the indicative results above.

### Kokkos CPU sizing thresholds

Counts are elements or scalar product work, never allocated bytes. Stages and reductions
use the field's element count; batch updates use the fused batch's total. Packing uses
matrix entries and temporal copies use output samples, without multiplying by the dense
transform's inner dimension. Responses use active columns × components × fine-grid
samples. Dense products use output rows × inner dimension × samples; coupled matrix
functions instead use active spectral bins × modes³, and the exponential Adams trial's
passes the field's elements × history rows. FFT and Raman choices use each
plan's actual real-side batch dimensions, including a kernel transform and the last
partial chunk.

| Operation | Serial below | Full pool from | Intermediate rule |
|---|---:|---:|---|
| Stage/update/reduction | 8,192 | 8,192 | none |
| Packing/temporal copy | 65,536 | 65,536 | none |
| Response | 32,768 | 196,608 | up to four threads; at least 40 component columns or 65,536 fine samples per column also enables the full pool |
| Native/custom GEMM | 1,048,576 | 1,048,576 | none |
| General-modal synthesis/projection/maps | 65,536 | 65,536 | none |
| Coupled matrix functions | 32,768 | 32,768 | none |
| Exponential Adams trial passes | 8,192 | 65,536 | up to four threads |
| Modal/spatial FFT | 32,768 | 131,072 | up to four threads |
| Raman FFT | 65,536 | 262,144 | up to four threads |

A response column is one active node and one component. Every width is capped by the
global thread setting. A kernel pool that already holds a full-width instance gives it to
the medium-width kernels too ([a stable kernel team](#stable-kernel-team)); the FFT pools
keep their medium widths. Explicit per-pool requests bypass this table, including the
short-FFT rule. `build_info()` exposes the constants and override mask; backend metadata
exposes effective widths and sizing inputs. Non-OpenMP host builds report that
per-operation sizing is unsupported and keep their existing execution policy.

### Resident Adams allocations

`F = N² × 600 × 16` bytes is one complex128 field of the reduced `kerr_beam_384` above
and `C = 5592 × 600 × 16` bytes one node chunk (a 1 GiB node budget). Each pair is
**`numba_cpu` / Kokkos** capacity in GiB; arrows show the 29 → 18 volume change. The
384² capacities are calculated. These are owned allocations, not resident pages.

| Allocation | Field count, `numba_cpu` / Kokkos | 128² | 256² | 384² |
|---|---|---:|---:|---:|
| Adams history ring | 7 / 7 | 1.025 / 1.025 | 4.102 / 4.102 | 9.229 / 9.229 |
| Adams moments | 8 / 7 | 1.172 / 1.025 | 4.688 / 4.102 | 10.547 / 9.229 |
| Adams flow / propagator | 1 / 1 | 0.146 / 0.146 | 0.586 / 0.586 | 1.318 / 1.318 |
| Native Adams source, exponent, saved state | 0 / 3 | 0 / 0.439 | 0 / 1.758 | 0 / 3.955 |
| Dense slopes | 0 / 7 → 0 | 0 / 1.025 → 0 | 0 / 4.102 → 0 | 0 / 9.229 → 0 |
| Dense start, end, exponent, held observation | 0 / 4 → 0 | 0 / 0.586 → 0 | 0 / 2.344 → 0 | 0 / 5.273 → 0 |
| Current state | 1 / 1 | 0.146 / 0.146 | 0.586 / 0.586 | 1.318 / 1.318 |
| RK stages | 7 / 7 | 1.025 / 1.025 | 4.102 / 4.102 | 9.229 / 9.229 |
| Solution, error, advanced, recycled stage | 4 / 4 | 0.586 / 0.586 | 2.344 / 2.344 | 5.273 / 5.273 |
| RK exponent / weighted residual | 1 / 1 | 0.146 / 0.146 | 0.586 / 0.586 | 1.318 / 1.318 |
| RK propagators | 2 / 1 | 0.293 / 0.146 | 1.172 / 0.586 | 2.637 / 1.318 |
| Transverse FFT scratch | 1 / 1 | 0.146 / 0.146 | 0.586 / 0.586 | 1.318 / 1.318 |
| **Field-volume subtotal** | **32 / 44 → 33** | **4.688 / 6.445 → 4.834** | **18.750 / 25.781 → 19.336** | **42.188 / 58.008 → 43.506** |
| Temporal FFT buffers (`4C` each) | fixed chunk | 0.200 / 0.200 | 0.200 / 0.200 | 0.200 / 0.200 |
| Absorber scratch (`0 / 2C`) | fixed chunk | 0 / 0.100 | 0 / 0.100 | 0 / 0.100 |
| Statistics spectrum, time, power (`0 / 2.5C`) | fixed chunk | 0 / 0.125 | 0 / 0.125 | 0 / 0.125 |

The remaining ledger entries (reduction partials, axes, spatial coefficients, statistics
outputs, host problem tables, masks, weights and plan heap) total a few MiB. Outside the
ledgers, the input field adds one `F`; Kokkos keeps another `F` of host state, and field
export passes through two more `F` at the Python boundary. Host Adams trials form about
`4F` of temporary arrays. Capacities and transients do not all become resident at once,
so their sum overstates the measured peak.

## Native CPU FFT and diagnostic scheduling

This campaign uses the frozen `benchmarks ab` harness and unchanged registry recipes,
under an approved exception to standard-run and whole-suite performance reporting.
These are **lightweight measurements**, not a standard-run adoption verdict. Every
reported propagation window contains ten prescribed steps: `fixed-rk` for modal and
radial cases, `fixed-rk4ip` for Cartesian Kerr. Construction, first-window cost and
observations are recorded separately; no window is extrapolated to a full run.

The baseline is `9521a88`; the retained implementation is `b3bdd0f`. Each arm has a
separate source/environment/native wheel. On the Apple M5 Max (18 cores, 128 GiB,
macOS 27.0.1), Homebrew Clang 23.1.3 uses Accelerate BLAS and one LLVM OpenMP runtime.
The source-built FFTW 3.3.11 control and NEON library have identical M1 flags,
deployment target 14.0 and cycle-counter support, with both precisions and coherent
threading libraries. The two-socket Xeon Gold 6240 uses exclusive 72-logical-CPU
allocations and one hardware thread per selected physical core. Its independent
bundles are GCC 16.2/OpenBLAS 0.3.34/FFTW 3.3.11 with GNU OpenMP, and IntelLLVM
2026.1.1/MKL 2026.1 (including MKL FFT) with Intel OpenMP. Timed arms run sequentially,
without concurrent builds. Python is 3.14.5 on the Mac and 3.14.7 on the Xeon;
NumPy is 2.5.2, Kokkos 5.2.1, Kokkos Kernels 5.2.0 and Kokkos-FFT 2.0.0.

### NEON codelets and Raman plans

A same-source control with matched compiler flags changes only FFTW's explicit NEON
codelets. Its eight-case screen at budgets 1 and 18 has geometric mean ratio 0.933;
this one-round/two-repetition screen isolates the library effect but is not a
confirmation. Full and tail microchecks include H₂'s 65536×21/12 and CtC's 131072×3/1
rows at serial and parallel widths, raw FFT normalization and exact padding zeros.
ESTIMATE and MEASURE both retain short parallel shapes where vDSP is faster, so
complete vDSP removal is rejected.

With NEON in both arms, replacing the previous Raman choice with tuned Kokkos-FFT
has the following three-round/five-repetition warmed-window ratios. The rest of the
source is held fixed for this provider/planning comparison.

| Case | 1 thread | 18 threads |
|---|---:|---:|
| CtC | 0.698 [0.694, 0.699] | 0.966 [0.961, 0.971] |
| H₂ Raman | 0.480 [0.459, 0.485] | 0.788 [0.767, 0.795] |
| GNLSE Raman shock | 0.828 [0.817, 0.830] | 0.839 [0.839, 0.846] |

Measured planning has a real setup cost. In this ordered Raman comparison, median
serial CtC construction changes from 3.00 to 22.83 s, while a warmed ten-step window
changes from 6.00 to 4.18 s. H₂ construction changes from 1.09 to 1.81 s and its
window from 1.28 to 0.62 s. Wisdom is process-local and previous constructions affect
these costs; they are not universal constants. Explicit `SOLPROP_FFTW_PLAN=estimate`
trades this measured setup for cheaper construction. Parallel default planning stays
ESTIMATE. No full-propagation payback is inferred from these ten-step windows.

### Retained Mac combination

The fresh current-library baseline and retained NEON/provider/scheduling candidate
use three alternating rounds and five repetitions at each budget. Ratios are the
median of paired round medians, with the range across rounds in brackets.

| Case | 1 thread | 18 threads |
|---|---:|---:|
| RDW | 0.888 [0.877, 0.901] | 0.831 [0.810, 0.835] |
| CtC | 0.595 [0.577, 0.604] | 0.859 [0.858, 0.878] |
| Radial 256 | 0.893 [0.853, 0.912] | 0.962 [0.934, 0.967] |
| Resolved argon | 0.954 [0.950, 0.961] | 0.886 [0.879, 0.898] |
| H₂ Raman | 0.472 [0.457, 0.474] | 0.718 [0.712, 0.718] |
| Averaged helium RDW | 0.891 [0.880, 0.901] | 0.902 [0.891, 0.916] |
| Averaged argon | 0.918 [0.909, 0.922] | 0.915 [0.905, 0.920] |
| GNLSE soliton | 0.889 [0.882, 0.893] | 0.870 [0.867, 0.871] |

The geometric mean is 0.792 at one thread and 0.865 at the production budget of 18
(0.828 across both budgets). Using each fresh process's warmed median as one sample,
the conservative aggregate `(candidate median + 2 MAD) / (baseline median − 2 MAD)`
is 0.805 and 0.878 respectively. All denominators are positive. These spread checks
support the retained change within the lightweight scope. All fixed windows retain
ten accepted steps, no rejected steps and 61 source evaluations; field differences
are at most 2.1e-16 in relative L2 norm. A one-round/two-repetition guard screen at budget 18 gives radial
512 1.012, Cartesian plasma 128² 0.894, line 0.976, general modal 0.892, scalar wall
0.936 and vector wall 0.990. The 1.2% radial variation does not exceed the A/A spread.

The retained library also passes the Cartesian Kerr confirmation (three rounds,
two repetitions, ten `fixed-rk4ip` steps). No Cartesian traversal change is included:

| Case | 1 thread | 18 threads |
|---|---:|---:|
| Kerr 192² | 0.843 [0.832, 0.849] | 0.869 [0.718, 0.924] |
| Kerr 384² | 0.837 [0.834, 0.860] | 0.880 [0.880, 0.895] |

The geometric mean is 0.840 at one thread and 0.874 at 18 (0.857 combined);
the corresponding conservative aggregates are 0.860 and 0.919. Fields differ by at most 1.4e-18 and work
matches. The 192² parallel spread is wider, so its precise gain is less certain;
all three paired rounds still favour the candidate. These results measure the
library change with the original traversal.

### Diagnostic scheduling

The affected set is ordinary DOPRI5 modal CtC, RDW, resolved argon and H₂ Raman.
The implementation defers coarse projections to the final source and their FFTs and
held-current assembly to accepted scoring. General-modal, dispersive, low-storage,
Adams and envelope-plasma scheduling remains eager. A serial full-source profile
reduced RDW's forward real FFT calls from 427 to 274 per ten-step window; CtC's
fell from 1464 to 1311. Persistent field-buffer ledgers are unchanged. FFT plan heap
estimates can vary with planning and allocator state.

These confirmations use three alternating rounds and five repetitions. Each ratio
is the median of the per-round paired medians; brackets give the range over rounds.
A ratio below one favours the candidate.

| Case | M5 Max, 18 threads | GCC Xeon, 1 thread | GCC Xeon, 18 threads | Intel Xeon, 1 thread | Intel Xeon, 18 threads |
|---|---:|---:|---:|---:|---:|
| RDW | 0.803 [0.755, 0.845] | 0.930 [0.925, 0.946] | 0.823 [0.820, 0.843] | 0.900 [0.887, 0.916] | 0.822 [0.813, 0.828] |
| CtC | 0.906 [0.888, 0.917] | 0.897 [0.894, 0.901] | 0.946 [0.945, 0.949] | 0.904 [0.893, 0.910] | 0.937 [0.933, 0.937] |
| Resolved argon | — | 0.998 [0.984, 1.002] | 0.949 [0.941, 0.953] | 0.966 [0.963, 0.976] | 0.966 [0.956, 0.972] |
| H₂ Raman | 0.933 [0.914, 0.967] | 1.007 [0.986, 1.082] | 0.982 [0.981, 0.988] | 0.985 [0.976, 1.044] | 0.973 [0.957, 0.977] |

At one thread the GCC affected-set geometric mean is 0.957 and Intel's is 0.938;
at 18 threads they are 0.923 and 0.922. The corresponding conservative aggregates
from fresh-process medians are 0.972/0.931 for GCC and 0.951/0.929 for Intel, with
positive denominators. Both production-budget gains exceed 5% and satisfy the
conservative spread criterion. Mac's isolated 18-thread ratio is 0.879, with a
conservative aggregate of 0.950. Across both budgets, GCC is 0.940 and Intel 0.930.
The matched fixed windows retain ten accepted steps, no rejections and 61 source evaluations; their
field differences are at rounding level (at most 1.9e-16 in these confirmations).
The Mac's first 18-thread screen and one A/A round showed large host timing
variation and remain inconclusive. The runner's busy-at-launch flag does not establish its cause; load averages also include prior
owned work. A subsequent three-round A/A control gave RDW 1.010 [1.008, 1.054]
and H₂ 1.015 [0.995, 1.040].
Those controls do not justify interpreting a few-percent change as a gain.

### Retained Xeon combination

Each compiler bundle also compares the complete retained candidate with its frozen
baseline. Raman cases use three alternating rounds/five repetitions at budgets 1
and 18; other main cases and geometry guards use one round/two repetitions. The
Raman confirmations are:

| Case | GCC, 1 thread | GCC, 18 threads | Intel, 1 thread | Intel, 18 threads |
|---|---:|---:|---:|---:|
| CtC | 0.790 [0.789, 0.792] | 0.963 [0.953, 0.966] | 0.912 [0.908, 0.914] | 0.939 [0.934, 0.942] |
| H₂ Raman | 0.824 [0.817, 0.841] | 1.003 [0.988, 1.030] | 0.977 [0.966, 0.986] | 0.976 [0.974, 0.977] |

The geometric means at budgets 1/18 are 0.807/0.983 for GCC and 0.944/0.957 for
Intel (combined 0.890 and 0.951). GCC's parallel Raman-only conservative aggregate
is 1.004, so that subset does not establish a gain beyond spread; the confirmed
shared scheduling gain is assessed over its complete affected set above. The
serial GCC improvement includes measured FFTW planning; Intel uses MKL FFT, whose
FFTW compatibility layer ignores planning flags. No benefit is inferred for
unmeasured Linux Arm codelets.

Main-case guard screens, followed by A/A controls for the small averaged and GNLSE
windows, are reported separately from the confirmed affected-set gains:

| Case | GCC, 1 thread | GCC, 18 threads | Intel, 1 thread | Intel, 18 threads |
|---|---:|---:|---:|---:|
| RDW | 0.928 | 0.834 | 0.929 | 0.827 |
| Radial 256 | 1.002 | 0.962 | 1.012 | 0.991 |
| Resolved argon | 1.000 | 0.947 | 0.988 | 0.965 |
| Averaged helium RDW | 1.014 | 1.051 | 1.011 | 1.001 |
| Averaged argon | 1.004 | 1.025 | 0.994 | 1.017 |
| GNLSE soliton | 1.008 | 1.005 | 1.019 | 1.039 |

The initial GCC averaged-helium 18-thread ratio of 1.051 and Intel GNLSE ratio of
1.039 triggered separate three-round/five-repetition A/A and A/B controls:

| Control case | A/A, 1 thread | A/B, 1 thread | A/A, 18 threads | A/B, 18 threads |
|---|---:|---:|---:|---:|
| GCC averaged helium | 0.997 [0.989, 0.998] | 1.014 [1.005, 1.016] | 1.016 [0.998, 1.128] | 1.012 [1.012, 1.014] |
| GCC averaged argon | 0.987 [0.984, 0.994] | 1.014 [1.009, 1.031] | 1.001 [1.000, 1.001] | 1.010 [0.983, 1.017] |
| GCC GNLSE | 0.991 [0.818, 0.997] | 1.005 [1.003, 1.016] | 0.999 [0.995, 0.999] | 1.010 [0.992, 1.017] |
| Intel GNLSE | 0.981 [0.746, 0.993] | 1.005 [0.996, 1.008] | 1.018 [0.985, 1.061] | 0.999 [0.991, 1.018] |

The initial 4–5% slowdowns do not repeat. GCC's small-window confirmation aggregate
is 1.011, a roughly 1% cost that remains visible; this does not justify additional
specialisation. Large A/A outliers make those controls inconclusive at fine
resolution. No improvement is claimed for these unchanged small-window routes.

At budget 18, GCC's six geometry guards span 0.977–1.018 (aggregate 1.000), and
Intel's span 0.998–1.017 (aggregate 1.007). These cover radial 512, Cartesian plasma
128², line, general modal and scalar/vector walls. Cartesian Kerr 192²/384² guards
are 0.997/0.999 for GCC and 1.001/1.004 for Intel, with ten steps and matched work.
They support the absence of a large regression; they do not establish percent-level
improvements on routes whose implementation is unchanged.

### Verification scope

Focused checks cover raw FFT normalization, full/tail shapes, threaded execution,
provider selection, planning overrides, coherent library coexistence, repeated
construction/closure, source/step parity, held-plasma diagnostics after discarded
trials, FSAL, reset, accepted values and peaks, and band-filter boundaries and high
dynamic range. Native suites passed on both Xeon bundles: GCC has 137 C++ tests and
378 Python tests passing; Intel has 133 and 376. Provider/platform/device skips
remain explicit. Mac final focused Python checks have 118 passes plus 13 lifetime
and import checks. Independent bounded smooth-law adaptive fixtures pass on both
arms and all three builds against accuracy envelopes fixed from tighter NumPy
references; these small fixtures do not certify full production trajectories.

The macOS distribution helper was exercised with freshly built dependencies and
wheel repair. A clean installation discovers both bundled FFTW precisions as NEON,
one bundled OpenMP runtime and the required third-party licences, with the Apple M1
baseline and macOS 14 target. No workflow edit is needed: the existing macOS wheel
job calls this dependency recipe. Full gates, slow and coverage lanes are deferred
to CI at the maintainer's request; GPU/device checks were unavailable on these CPU
hosts. Focused formatting, lint, spelling and diff checks passed. Supplemental
CPU/wall, observation, load and memory receipts are collected outside A/B timing;
no additional speed claim is taken from those resource measurements.

### Rejected leads

| Lead | Bounded evidence | Decision |
|---|---|---|
| Remove vDSP completely | NEON FFTW still loses on short parallel rows, including 8192×1 and 16384×1, under both ESTIMATE and MEASURE | Keep vDSP for short parallel power-of-two rows; use FFTW for serial and long rows with NEON |
| Tap-first band filter, preserving each sample's tap order | Mac serial RDW 0.973 [0.970, 0.982], argon 1.079 [1.061, 1.090]; GCC/Intel RDW/argon aggregates 0.911/0.900 across 1/18 threads | Reject the shared traversal and avoid an architecture-specific fork; retain the new boundary/dynamic-range tests |
| Cartesian 8×32 tile plus unused-load elimination | 192² Kerr: serial 0.980 [0.939, 0.997], 18 threads 0.980 [0.959, 1.005], three rounds/two repetitions | Reject the approximately 2% gain; keep 32×16 and the existing loads |
| Force one Accelerate BLAS thread | Five-case confirmation at budget 18: geometric mean 1.000, three rounds/five repetitions | Keep the existing provider-managed startup policy |

The post-FFTW 192² serial profile assigns 4.62 s to the separable fold, 4.72 s to
complex FFTs and 2.11 s to temporal pack/crop within a 12.16 s instrumented window.
A separate non-IPO compiler diagnostic reports control-flow and call barriers in the
crossing and final reduction loops; the measured production binaries retain IPO.
The tile experiment changes locality without changing stored layouts or per-column
summation order, but its complete-window gain does not meet the greater-than-5%
threshold. Instrumented profiles are diagnostic evidence, not timings used for
retaining a change.

Independent startup-only pool screens at budget 18 varied kernel, ordinary FFT,
Raman full, Raman tail and BLAS widths separately to 1 or 4. Both Xeon bundles
showed aggregate slowdowns for every reduced-width screen (the smallest was 1.095
for Intel's four-thread Raman tail). Mac responses were mixed, including opposite
RDW and argon directions. No shared dimension threshold or pool policy is changed;
explicit overrides and global caps retain their existing semantics. A 36-core
sizing confirmation is unnecessary because no sizing change is retained.
