# Benchmarks

How solprop's performance is measured and judged, and what the measurements so far
indicate. A performance claim names its measured window, build, machine and result;
the rules below say how to take it. The numbers in the last two sections are indicative:
each was measured once, on one machine, for one change. Re-measure with the standard
suite before relying on any of them.

## Where things are

- **The standard benchmark suite** is `benchmarks/`
  ([SEP 30](seps/archive/sep-0030-standard-benchmark-suite.md)), run with
  `uv run poe bench` (`python -m benchmarks`). It measures bounded windows of
  full-resolution problems in fresh processes and reports warm time, startup time,
  peak memory and numerical check status. Its default is a quick Kokkos CPU run of the
  four core cases (four-mode RDW, six-mode comb-to-continuum, a 384² Kerr beam and a
  256-node radial filament); further sets cover modal capillaries, mode-averaged and
  GNLSE cases, free space, coherent walls, tapers, the general modal route and
  particular backend paths. Quick mode takes one sample and gives no verdict; standard
  mode takes paired, shuffled fresh-process samples at one thread and at the machine's
  declared production width and applies the adoption rule below. Cases are versioned
  by their workload, so a changed definition is a new revision and is never compared
  with the old one. [Running the benchmark suite](benchmarking.md) is the manual: case
  sets, timing boundaries, collection, numerical qualification, native coverage and
  export.
- **Results stay local.** Runs, machine profiles, references and coverage reports go to
  the ignored `.benchmarks/suite/` directory; no generated timing, reference or
  coverage report is committed. `poe bench export` writes a sanitised snapshot when
  one is wanted for a report.
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
  The standard suite's comparison adds its clean-pair, repeatability and qualified
  reference requirements
  ([collection and comparison](benchmarking.md#collection-recovery-and-comparison)).

(benchmark-windows)=
## Measurement windows

Timing is measured on short windows, not on complete propagations: a complete
comb-to-continuum run takes hours, and its cost is the window's times a step count the
window already predicts.

- A timing window is about ten accepted steps of the adaptive run, frozen from an
  untimed pilot so that every repeat takes the same prescribed steps, or a fibre
  short enough that a complete run takes seconds.
- Report the window that was measured: its z extent, accepted steps, what the timed
  region includes (propagation, observation and public statistics) and excludes
  (translation, backend construction, HDF5), and the A/A resolution indicator of the
  alternating pairs. Do not extrapolate a window to a complete-run cost.
- A later state of a long propagation is timed by replaying a saved witness into a
  short window, not by regenerating the propagation.
- Compare two source trees with alternating fresh-process pairs (A/B/A/B…), and report the
  A/A spread beside the ratio; a difference inside the spread is not a result.

The suite's windows and what its timers include are listed under
[timing boundaries and accounting](benchmarking.md#timing-boundaries-and-accounting).

(benchmark-reporting)=
## Reporting a performance claim

**The machine record.** Every measurement carries the CPU (or device) model, core count,
memory, operating system, Python and dependency versions (including the Kokkos build and
its profile), thread counts, FFTW plan flags and wisdom state, the source revision and
whether the tree was dirty, and the load before and after. The suite's machine profile
and run manifest record these; a hand-run measurement states them. Results are compared
only within one machine.

**A quiet machine.** Absolute numbers are taken on a dedicated workstation or an
exclusively allocated node. A busy laptop is fit only for same-process A/B comparisons,
labelled as such: read their ratios, not their times. Report the A/A spread of repeated
baseline runs beside every ratio; a difference inside it is not a result.

**What goes in a pull request.** The measured window and what it includes, the build and
machine record, the per-case table with medians and spreads, the declared aggregate and
its verdict, and any case that regressed. Concise evidence belongs in the change
description; raw runs stay local or go into an exported snapshot.

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
| reduced comb-to-continuum | 2.8–3.1× faster | 2.4–3.6× faster |
| RDW ten-step window | 1.81× faster | 2.33× faster |
| reduced comb-to-continuum ten-step window | 3.18× faster | 3.85× faster |
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

## Reference tables

These tables back the indicative results above.

### Kokkos CPU sizing thresholds

Counts are elements or scalar product work, never allocated bytes. Stages and reductions
use the field's element count; batch updates use the fused batch's total. Packing uses
matrix entries and temporal copies use output samples, without multiplying by the dense
transform's inner dimension. Responses use active columns × components × fine-grid
samples. Dense products use output rows × inner dimension × samples; coupled matrix
functions instead use active spectral bins × modes³. FFT and Raman choices use each
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
| Modal/spatial FFT | 32,768 | 131,072 | up to four threads |
| Raman FFT | 65,536 | 262,144 | up to four threads |

A response column is one active node and one component. Every width is capped by the
global thread setting. Explicit per-pool requests bypass this table, including the
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
