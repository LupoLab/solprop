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
