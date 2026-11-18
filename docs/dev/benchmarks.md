# Benchmarks

How solprop's performance is measured and judged, what the current release takes on the
recorded hardware, and the standing results a contributor needs before changing a hot
path. [Running the benchmarks](benchmarking.md) is the manual for the harness. Each
result below names its date (the day it was recorded) and the processor or device it ran
on; a few older records name only a class of machine, and say so. Every result was
measured once, on one machine, for one change: re-measure before relying on one. The
per-change campaign records kept until 0.3, each A/B with its full table, are in the
repository's history
([this page at commit `ec3eb8f`](https://github.com/LupoLab/solprop/blob/ec3eb8fc9353a30d9d7d43e82762c6ff8c18c70c/docs/dev/benchmarks.md));
much of that ledger measured code that no longer exists.

## Where things are

- **The benchmarks** are `benchmarks/`
  ([SEP 38](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0038-performance-tracking.md)): `track` times complete runs of the
  tracking set at one thread and the production width, `ab` compares two builds, and
  `benchmarks/history/` holds a few rows per release for each hardware set.
- **[CPU performance: what was tried](cpu-performance.md)** records what was adopted,
  measured and rejected on the compiled CPU backend. Read it before repeating an
  experiment. [backends.md](backends.md) describes each backend's execution design.

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
  do not automatically veto the change. Numerical bounds are never relaxed to make a change pass.
  The A/A spread of the same comparison says how large a difference has to be to count
  ([comparing two builds](benchmarking.md#comparing-two-builds-while-iterating)).

(benchmark-windows)=
## What is timed

- **Release tracking times complete runs** of guides at production sizes with the
  library's defaults; construction is timed separately, after an untimed warm-up
  ([what is timed](benchmarking.md#what-is-timed)).
- **A change in progress may time less.** `ab --length` runs part of each guide from the
  entrance. Report the fraction, and do not extrapolate it to a complete run: the
  expensive part of a propagation (compression, ionisation, the Raman shift) often comes
  late. A later state is timed by replaying a saved witness into a short run.
- **An event's cost is timed on a window that holds it.** The `sep47` set's windows hold
  the rows, snapshots, planes or flows they measure, and `--counters` reports the native
  run's returns to Python, crossings and bytes, so that two builds can be checked to do the
  same work ([SEP 47's matrix](benchmarking.md#sep-47s-matrix-and-the-structural-counters)).
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

(benchmark-results)=
## Current performance

### The tracking set at 0.2.0

The newest release rows of `benchmarks/history/`, recorded on 2026-11-07: the native
backend from the release wheel on the CPUs, a build of the tag on the GPUs
([recorded hardware](benchmarking.md#recorded-hardware)), passive threading, the median
of three complete runs in seconds. The production width is 18 threads on the Apple M5
Max, 36 on the two Intel Xeon Gold 6240, and 16 on the AMD EPYC 7763 node, the Google
Cloud C4D (AMD EPYC 9B45, Linux and Windows) and the Google Cloud C4A (Arm Neoverse V2).

| case, production width | M5 Max | Xeon Gold 6240 | EPYC 7763 | EPYC 9B45 | EPYC 9B45, Windows | Neoverse V2 |
|---|---:|---:|---:|---:|---:|---:|
| `gnlse_raman_shock` | 21.0 | 41.8 | 38.6 | 23.5 | 24.0 | 39.3 |
| `gnlse_soliton` | 0.291 | 0.627 | 0.574 | 0.270 | 0.328 | 0.717 |
| `rdw` | 3.19 | 7.53 | 4.78 | 3.26 | 3.73 | 4.31 |
| `he_rdw_averaged` | 0.423 | 1.16 | 0.731 | 0.407 | 0.581 | 0.894 |
| `argon_blueshift` | 6.21 | 14.7 | 9.75 | 7.07 | 7.20 | 9.24 |
| `argon_blueshift_averaged` | 1.94 | 3.60 | 3.38 | 1.86 | 2.30 | 3.90 |
| `argon_blueshift_envelope` | 0.340 | 1.65 | 1.16 | 0.351 | 0.554 | 1.76 |
| `h2_raman_averaged` | 3.00 | 5.74 | 5.18 | 2.85 | 3.22 | 6.19 |
| `ctc` | 65.5 | 176 | 112 | 65.6 | 80.2 | 104 |
| `radial_filament_256` | 0.957 | 1.61 | 1.76 | 0.846 | 1.09 | 1.17 |
| `grin_nonlinear` | 11.8 | 23.5 | 21.5 | 11.9 | 12.2 | 14.8 |
| `kerr_beam_384` | 7.28 | 11.0 | 23.5 | 7.86 | 9.17 | 7.93 |

The same rows' device runs, a whole case each, seconds:

| case | NVIDIA A40 | NVIDIA H200 | AMD Instinct MI325X |
|---|---:|---:|---:|
| `gnlse_raman_shock` | 18.1 | 9.77 | 5.00 |
| `rdw` | 1.18 | 0.537 | 0.476 |
| `argon_blueshift` | 4.85 | 2.38 | 2.23 |
| `h2_raman_averaged` | 6.13 | 2.11 | 1.83 |
| `ctc` | 34.1 | 9.41 | 7.88 |
| `radial_filament_256` | 0.552 | 0.162 | 0.152 |
| `grin_nonlinear` | 15.1 | 4.80 | 3.34 |
| `kerr_beam_384` | 3.30 | 0.421 | 0.461 |

Against the 0.1.0 rows of 2026-11-01 (geometric means of the per-case ratios), 0.2.0 took
1.05 to 1.11 of the time at one thread and 0.93 to 1.06 at the production width on the
CPUs, 0.72 on the H200 and 0.80 on the MI325X. The A40's 0.2.0 rows start a new series
(its job spanned fewer memory domains), so they do not compare with its 0.1.0 rows.

### Since 0.2.0

Changes after the 0.2.0 rows, each measured with `benchmarks ab` against its parent (the
sections below give each one): transforms threaded by their shape and a modal stage's
forward rows in one transform (2026-11-09; geometric mean 0.977 on an Intel Xeon Gold
6240 and 0.981 on an AMD EPYC 7763, `argon_blueshift` 0.80 to 0.87 at full width);
free space's separable propagators as vector maths (2026-11-08; `line` 0.893 at one
thread on a Xeon Gold 6240); a run's fixed host cost (2026-11-08; 0.3 s less to build
every case, and 1.8 s and 2.4 s less to build and translate `kerr_beam_384`); and SEP
52's native workspace seam (2026-11-16; every round within 0.98 to 1.04 on an AMD EPYC
7763). A machine calibration (SEP 46) moved the tracking set by 0.965 on a Google Cloud
C4A and 0.989 on an Intel Xeon Platinum 8581C, and chose the built-in table, or nearly, on
the AMD EPYC 7763, the AMD EPYC 9B45 and the Intel Xeon Gold 6240 (2026-11-08).

### Against `numpy_ref`

`numpy_ref` is the reference, not a production backend: it does not thread. Over ten-step
windows of capillary, coherent-wall and general-route cases on one 18-core socket of an
AVX-512 server (recorded by 2026-10-23; the record does not name the model), the native
backend at one thread ran 2.1 to 8.6 times faster than `numpy_ref`. At 18 threads, on
larger general-route and coherent-wall workloads (2 to 32 channels, up to 16 384 fine
samples), it ran 6 to 40 times faster than single-threaded `numpy_ref`.

## Where a run's time goes

- **A source evaluation's transforms, response and dense products dominate the steps.**
  In a one-thread `grin_nonlinear` window on an Intel Xeon Gold 6240 (2026-10-30), after
  the changes of [band column passes](#band-column-passes-and-the-general-routes-real-signal),
  the complex inverse transform took 1.21 s, the projection 1.17 s and the synthesis 1.31 s;
  the synthesis and projection each stream a 122 MB map per evaluation at 8–9 GB/s, about
  what one core draws from memory. In a one-thread `ctc` window the band plasma column
  took 1.00 s of the 1.99 s spent in native kernels before those changes, and 0.49 s
  after.
- **On free-space grids the separable fold and the transforms share the step.** In a
  one-thread `kerr_beam_384` window on an Intel Xeon Gold 6240 the separable fold took
  27.7 s of 69.8 s before its layout change, which took the window to 54.2 s; at sixteen
  threads the 384² fold is bound by memory bandwidth, about 52 GB/s (2026-10-29).
- **Host work outside the steps is a fixed cost per run.** On complete runs (Intel Xeon
  Gold 6240, 2026-10-30) it is 0.1–0.3 s on the capillary and radial cases and about 6 s on
  `kerr_beam_384`: 2–9 % of a one-thread capillary run, and up to a third of the complete
  `radial_filament_256` and `kerr_beam_384` runs at sixteen threads. The cost that repeats is a saved plane of a large Cartesian field: about
  3 s for `kerr_beam_384`'s 1.77 GB field (the saved plane's copy 1.1 s, the conversion to
  the optics convention 1.0 s, the store's copy 0.6 s, the native observation 0.35 s),
  against 7–25 ms for a capillary or radial plane.
- **Construction is mostly FFTW planning and imports.** Without stored wisdom, measured
  planning is most of a short run's backend construction (about 1.3 s for averaged
  helium's 6400-sample real pair; [persistent wisdom](#persistent-fftw-wisdom)). On an
  Intel Xeon Gold 6240 (2026-11-08) CoolProp's import, through optmat's transport model,
  is 2.3–2.7 s of every capillary build and the driver's own import 0.75 s.
- **The step loop runs in native code.** The native marches spend 0.4 to 8 µs of host
  time a trial where solprop's own march took 11 to 210 µs (AMD EPYC 7763 and Apple M5
  Max, 2026-10-29), so returns to Python are host events (planes, statistics rows, monitor
  checks), not steps. A statistics row at every step costs about 290 µs a step on the EPYC
  7763 and 75 µs on the M5 Max; a native call's return at its bound 29 to 91 µs, at most
  0.21 % of a run (Intel Xeon Gold 6240, 2026-11-04).

### Builds and libraries

On one 18-core socket of an AVX-512 server (recorded by 2026-10-23; model not recorded),
seven ten-step capillary and general-route cases: against a build with no vendor BLAS
for generic x86-64, linking OpenBLAS alone made the straight capillaries 2.6 to 3.8 times
faster at one thread (the tapers 1.7) and left the general route unchanged; compiling for
`x86-64-v3` alone made the general route 1.4 to 1.5 times faster. A distribution-style
build (`x86-64-v3` and `scipy-openblas32`, ADR-034) ran within about 5 % of a local
`-march=native` build. MKL was 1.4 to 1.8 times faster than OpenBLAS on the TE/TM wall and
level elsewhere. Later, with identical libraries (2026-11-08), `x86-64-v4` against v3 gave
a geometric mean of 0.984 over the tracking set on an Intel Xeon Gold 6240 and
`-march=znver5` 0.963 on an AMD EPYC 9B45, no case reaching the 10 % that run-time
dispatch would need ([#337](https://github.com/LupoLab/solprop/issues/337)).

### Host vector maths

The native CPU kernels' transcendentals through SLEEF's vector functions (ADR-059), complete
runs or long entrance windows of the tracking cases at one thread (2026-10-28): geometric
mean −11.1 % on an AMD EPYC 7763 (AVX2 functions; `gnlse_raman_shock` −19.4 %) and −6.8 %
on an AMD EPYC 9B45, where the opt-in AVX-512F functions gave −11.4 %. On an Intel Xeon of
the Ice Lake generation the AVX2 functions gained 3.7 % and the AVX-512F ones lost 2.9 %,
so AVX2 is the default. On an Apple M5 Max the system libm stays: SLEEF's AdvSIMD functions
forced on were 11.6 % slower (`gnlse_raman_shock` 56 %).

## Threads

### Kokkos CPU operation-size threading

The native CPU backend chooses each operation's thread width from its dimensions at
construction ([tuning](../guide/performance/tuning.md)); a machine calibration may replace
the table below (SEP 46). On an 18-core arm64 laptop and a two-socket, 36-core AVX-512
server (recorded by 2026-10-23; models not recorded) the sizing ran the small cases 2.00
and 2.06 times faster than one thread. Copies and arithmetic need separate thresholds: an
8,192-element temporal copy is slower threaded, an arithmetic stage of that size is not. The exponential Adams
trial's two passes, about ten stage entries an element, thread at four from 8,192 elements
and take the pool from 65,536: a fork and join of sixteen sleeping threads costs 100–200 µs
(16 cores of a dual AMD EPYC 7643, 2026-10-28).

The built-in table counts elements or product work, never bytes (the counts are defined
in `native/src/solprop_native/_execution.py`):

| Operation | Serial below | Full pool from | Between the two |
|---|---:|---:|---|
| Stage, update, reduction | 4,096 (macOS 8,192) | the same | none |
| Packing, temporal copy | 65,536 | 65,536 | none |
| Response | 32,768 | 196,608 | four threads; 40 columns or 65,536 fine samples a column also take the pool |
| Native or custom GEMM | 1,048,576 | 1,048,576 | none |
| General-modal synthesis, projection, maps | 65,536 | 65,536 | none |
| Coupled matrix functions | 32,768 | 32,768 | none |
| Exponential Adams trial passes | 8,192 | 65,536 | four threads |
| Modal and spatial FFT | 8,192 (macOS 32,768) | 4,194,304 | by shape ([transforms by shape](#transforms-by-shape-and-one-forward-pack)) |
| Raman FFT | 65,536 | 262,144 | four threads |

Every width is capped by the process's thread budget, and on Apple silicon's spinning top
tier every threshold is divided by sixteen (ADR-073). Backend metadata records the
effective widths.

### A stable kernel team

GNU OpenMP ends a team's surplus threads when a region asks for fewer and starts them
again for the next wider one: alternating 16- and 4-thread regions took about 490 µs a pair
against 40 µs for two 16-thread regions (one socket of an Intel Xeon Gold 6240,
2026-10-29). A pool that already holds a full-width instance therefore hands it to
medium-width kernels too. At 16 threads the tracking set's geometric mean was 0.924 on a
Xeon Gold 6240 and 0.938 on an AMD EPYC 7763, `h2_raman_averaged` 0.526 and 0.483. The
promotion applies everywhere but macOS, where the Apple M5 Max's two kinds of core made it
lose; on Windows (LLVM's libomp) it still gained (`h2_raman_averaged` 0.906 on eight cores
of an AMD EPYC 9B45).

### Compact CPU teams

A problem of one column on one node (the GNLSE, the mode-averaged capillary) has few
operations large enough to thread. At 16 threads `gnlse_raman_shock` ran its serial
kernels 1.2–2.5 times slower than in a one-thread process: waking fifteen cores each 0.9 ms
evaluation held the main thread at about 2.7 GHz instead of 3.5, and its loads found lines
modified in other cores' caches (Intel Xeon Gold 6240, 2026-10-29). Such a problem now runs
on a team of at most eight threads, its FFT plans on that team through FFTW's loop
callback, and its products on one OpenBLAS thread. `gnlse_raman_shock` then ran 1.25 times
faster at 16 threads than at one, where it had run at the same speed; on a Google Cloud C4A
(Neoverse V2) the single BLAS thread took it from 25.8 to 17.6 s at 16 threads.

### Stacked projection splits

A stacked fine-and-coarse projection's split ran on the temporal copy's serial instance,
about 90 µs a copy at 16 threads on `rdw` beside full-width kernels (Intel Xeon Gold 6240,
2026-10-29); on the stage's instance `ctc` took 0.977 of its time.

### Smooth grid sizes

`python -m benchmarks.size_cliffs` flags a 2-3-5-smooth grid size whose cost per sample
exceeds its nearest power of two's by more than 25 %. Two boundaries made cliffs: the
stage threshold just above n_t 4000's 8004-element stage, and a node budget that left a
serial one-node tail at n_t 8640. With the thresholds lowered to 4096 and 16,384 and the
nodes spread evenly, n_t 4000 went from 2.45 to 1.32 ms per evaluation and n_t 8640 from
3.52 to 2.54 ms on an 18-core socket of an Intel Xeon Gold 6240 (2026-10-25). On an 18-core
arm64 laptop (model not recorded) the threaded stage is dearer than the serial one up to
8192, so macOS keeps the higher thresholds.

### Thread scaling

One-thread time over production-width time in the 0.2.0 rows of the
[tracking set](#the-tracking-set-at-020) (2026-11-07):

| case | M5 Max, 18 | Xeon Gold 6240, 36 | EPYC 7763, 16 | EPYC 9B45, 16 | Neoverse V2, 16 |
|---|---:|---:|---:|---:|---:|
| `gnlse_soliton` | 0.99 | 0.98 | 1.01 | 1.00 | 1.01 |
| `gnlse_raman_shock` | 1.18 | 1.14 | 1.14 | 0.92 | 1.47 |
| `argon_blueshift_averaged` | 1.21 | 1.24 | 1.15 | 1.09 | 1.32 |
| `argon_blueshift_envelope` | 1.37 | 0.71 | 1.06 | 1.39 | 0.76 |
| `rdw` | 2.04 | 2.21 | 2.42 | 1.99 | 3.34 |
| `argon_blueshift` | 2.74 | 2.03 | 2.71 | 1.97 | 3.58 |
| `radial_filament_256` | 3.22 | 4.94 | 3.95 | 4.85 | 7.05 |

- **One column gains little.** A GNLSE or a mode-averaged capillary gains at most about
  1.5 times from a whole machine, and can lose; resolved multimode and radial problems gain
  2 to 7 times.
- **Keep a run in one memory domain.** Two runs of one wheel at 36 threads, unpinned across
  the two sockets of a Xeon Gold 6240, differed by 0.84 to 1.57 per configuration
  (2026-11-03). On an exclusive AMD EPYC 7763 allocation, binding (`OMP_PROC_BIND=close
  OMP_PLACES=cores`) was worth about 11 % on average and neutral on the Xeon Gold 6240, so
  it is not the default (2026-10-30).
- **Idle threads spin briefly.** Every machine's calibration refused passive waiting
  (2026-11-08): 1.10 to 1.94 times slower, from a Google Cloud C4A to an Intel Xeon Platinum
  8581C. Concurrent scan workers ran the same spinning or passive on an AMD EPYC 7763 node
  (2026-11-09).

## FFT

### Transforms by shape and one forward pack

A batched forward transform's best width depends on its rows (Intel Xeon Gold 6240, AMD
EPYC 7763 and Apple M5 Max, 2026-11-09): a few rows ran best at about a thread a row; many
rows of a few thousand samples wanted 8 to 18 threads, where four were two to three times
slower; one row gained from being split only from about 16384 samples on the two Linux
machines (4608 on the M5 Max's top tier). Hence the rule: from the FFT threshold a thread
per 4096 samples, at most one a row, or up to the medium width for rows of 16384 samples
(macOS 32768 and 65536). Estimate plans of a prime length at more than four threads ran 15
to 47 times slower than at four; measured ones did not. A modal stage's forward rows now go
through one plan (`rdw` transforms 28 rows of 10240 at once): `argon_blueshift` 0.88, `rdw`
0.90 and `ctc` 0.92 on 18 cores of a Xeon Gold 6240. One-column problems keep separate
transforms, which the pack made 1.35 times slower there.

### Threaded FFT plan selection

FFTW's measured planning sometimes chose a slower threaded plan: for 204 rows of 2048
complex samples at 32 threads on a 32-core AVX-512 server (recorded by 2026-10-23; model not
recorded), a 64 × 32 decomposition instead of 32 × 64 made a general-modal window about
12 % slower. Measured wisdom is seeded only for serial transforms; threaded ones default to
estimate planning.

### Persistent FFTW wisdom

At one thread the default plans every row transform with FFTW's measured search.
`SOLPROP_FFTW_WISDOM` keeps those plans for the next process
([tuning](../guide/performance/tuning.md)). On one core of an Intel Xeon Gold 6240
(2026-10-29) it took the backend's construction from 1.4 to 0.033 s on `he_rdw_averaged`,
from 4.0 to 0.029 s on `h2_raman_averaged` and from 5.0 to 0.056 s on `ctc`, with windows
as fast as cold ones; an AMD EPYC 7763 gave the same picture. What remains of a warm
construction is allocation and setup: tens of milliseconds, a few tenths of a second for a
free-space Kerr beam's 1.4 GB of state.

### Native CPU FFT and diagnostic scheduling

Ten-step windows on an Apple M5 Max and two Intel Xeon Gold 6240 (2026-10-24). FFTW's NEON
codelets made an eight-case screen 0.933 on the M5 Max, where vDSP stays for short parallel
power-of-two rows. Tuned Kokkos-FFT Raman plans took `ctc` to 0.698 and `h2_raman_averaged`
to 0.480 at one thread, for more measured planning (`ctc`'s serial construction 3.0 → 22.8 s).
Deferring a step's coarse projections to accepted scoring cut `rdw`'s forward transforms
from 427 to 274 a window (`rdw` 0.82 at 18 threads on the Xeon Gold 6240). The M5 Max's
retained combination: 0.792 at one thread, 0.865 at 18.

## BLAS

### NumPy's BLAS beside a native run

NumPy's OpenBLAS is a pthreads library with a pool of its own. During a native run the host
calls it only for small products, often enough that its workers never sleep: at 16 threads
on an Intel Xeon Gold 6240 its 15 workers took 1.09 s of CPU each in a 1.24 s window, and
the Adams passes ran three times as slowly (2026-10-29). The native backend holds every
BLAS but its own to one thread while it lives: `rdw` 0.462 at 16 threads on the Xeon Gold
6240 and 0.762 on an AMD EPYC 7763, geometric means 0.886 and 0.855. A fresh process pays
the same pools at import (0.75 s of wall time and 13.8 s of CPU unpinned on 72 logical CPUs
of the Xeon Gold 6240, 2026-11-08); processes solprop starts set `OPENBLAS_THREAD_TIMEOUT`.

### One-mode products on the response team

A one-mode capillary with a field-plasma column synthesises and projects its node fields
through products the sizing keeps serial on the engine's side, which the vendor GEMM then
threaded over its own pool (0.46 s of a short window at 16 threads against 0.29 s at one,
Intel Xeon Gold 6240, 2026-10-29). Two small kernels on the response team took `argon_blueshift` to
0.679 at 16 threads. They apply with OpenBLAS only: Apple's Accelerate runs these calls
well, and on an Apple M5 Max the kernels made the window 1.31 times slower. The vendor GEMM
also beat native GEMMs for products of at most 256 left-matrix entries and plain row loops
for one-mode products on an AMD EPYC 7763.

## Kernels

### The Adams passes on x86-64 lanes

The exponential Adams trial's predict and correct passes on blocks of 32 elements held as
lanes, against element-by-element strict code (2026-10-29): one thread on an AMD EPYC 7763,
−30 to −37 % on the averaged capillaries and −16 % on `rdw` and `radial_filament_256`; on
an Intel Xeon Gold 6240 a geometric mean of 0.867 over one and sixteen threads. The rest of
the predict pass is about a third SLEEF's complex exponentials, a sixth the memory-bound
phase and rate sum.

### Host stage entry, free-space crossings and statistics

On an Intel Xeon Gold 6240 at one thread (2026-10-29): the modal stage entry as 64-bin real
and imaginary lanes, `gnlse_raman_shock` 0.836; the separable low-storage fold in the
scratch's layout, `kerr_beam_384` 0.789; crossings without run-time divisions, the
monitor's runs and the radial statistics' GEMMs, `radial_filament_256` 0.834 and
`kerr_beam_384` 0.830 again. Together `kerr_beam_192` took 0.635 of its time at one thread.
On an AMD EPYC 7763, whose divisions and strided streams are far cheaper, the free-space
cases took about 0.95 at one thread.

### Band column passes and the general route's real signal

The band plasma column's passes as block loops, its filter a tap pair at a time on x86-64,
and a real-form cubic's field through a complex-to-real transform (Intel Xeon Gold 6240,
2026-10-30): `ctc` 0.703, `argon_blueshift` 0.641 and `grin_nonlinear` 0.701 at one thread,
0.80 to 0.85 at sixteen; on an AMD EPYC 7763 0.72 to 0.76 at one thread.

### Band-rule plasma integration

The band-exact plasma integrals (SEP 18) against the trapezoid on the four-mode RDW window
(recorded by 2026-10-23): 1.3, 1.9 and 2.7 times the time at 18, 4 and 1 threads on an
18-core arm64 laptop (model not recorded), before the block loops
[above](#band-column-passes-and-the-general-routes-real-signal) halved the column; 1.10
on CUDA (direct halo reads) and 1.25 on HIP (halo staged in team scratch).

### Retained radial synthesis and Raman modal products

On an Intel Xeon Gold 6240 (2026-10-29), a square radial map synthesising the retained
spectrum only: `radial_filament_256` 0.928 at one thread, 0.954 at 16; the Raman response
over three-mode `ctc`'s modal products rather than its node drives: 0.884 at 16, at the cost
of more FFT plans (a `ctc` worker's wall time, construction included, 16.3 → 17.7 s).

### Radial analytic chunk dispatch

Partial radial node chunks through the interleaved real GEMM, instead of a scalar
contraction (94 % of the source time), made the radial filament's Adams window 2.8 to 8.8
times faster on a two-socket AVX-512 server (recorded by 2026-10-23; model not recorded).

### Multimode CPU step costs

Real cubic monomials shared across output components, general-modal tail transforms planned
over active rows only and serial transforms seeded with measured wisdom: in ten-step
profiles on an 18-core arm64 laptop (recorded by 2026-10-23; model not recorded) the general
route's tensor work fell from 0.44 to 0.15 s and its FFTs from 1.00 to 0.84 s. Measured
planning adds about 18 ms (general modal) and 120 ms (reduced comb-to-continuum) of
construction.

### General-route symmetric contraction

The real cubic over its ten symmetric monomials instead of 81 + 3 dense products
([algorithms.md](algorithms.md)): a two-channel step-index window 1.35 times faster at one
thread (recorded by 2026-10-23; machine not recorded), less with more channels.

### Separable stage propagators as vector maths

Free space's separable propagators through one `vmath::sincos` pass per 32 × 16 tile instead
of a libm call per element (18 % of `line`): `line` 0.893 at one thread and 0.927 at 16 on
an Intel Xeon Gold 6240 (2026-11-08).

### Dispersive electronic response

The dispersive Kerr response (SEP 6) is judged by its nonlinear stage time against the
constant cubic's on the same route, the symmetric evaluator's target 1.3 and the
polynomial's 4 (recorded by 2026-10-23; machines not recorded). On the native CPU backend at
one thread the symmetric evaluator took 0.81–1.21 (one species) and 0.97–1.21 (a mixture);
the polynomial 1.52–3.65 for one species (4.2–5.0 on the general route) and 3.2–8.3 for a
mixture of distinct leg bounds, at most 2.2 in the default pools. On NVIDIA and AMD GPUs the
symmetric evaluator took 0.86–1.30 and the polynomial 1.01–3.53 for one species. A polynomial mixture is
one group per distinct leg bound, which is why mixtures such as air miss its target on CPUs.

### Raman kernel rebuild

On a density gradient the Raman kernel is rebuilt at every stage, and its cost scales with
the response's rows. `numpy_ref`'s rebuild at $2^{14}$ fine samples (a loaded laptop,
recorded by 2026-10-23, model not recorded) took 0.31 ms for a two-term N₂ model, 0.47 ms for
optmat's H₂ (4 rows), 6.0 ms for optmat's N₂ (72 rows) and 10.2 ms for dry air (126 rows),
against 0.25 ms for one convolution; the native rebuild is slower than the host's at one
thread and overtakes it at four. With tens of rows the rebuild dominates the step;
[SEP 10](seps/sep-0010-low-rank-raman.md)'s low-rank representation is what removes it.

### Adams carrier frame per step

The scalar carrier frame (SEP 35) on ten replayed steps of a hydrogen capillary at one
thread, Apple M5 Max (2026-10-25, a machine in use: ratios only): 1.057 with a Kerr-only
source and 1.006 with Raman. Its case rests on the calls it saves.

### Low-storage interaction-picture RK

The low-storage march (four evaluations a step) against the IP-DOPRI5 replay (six) on a
Cartesian envelope Kerr beam with fixed steps, on an 18-core arm64 workstation (recorded by
2026-10-23; model not recorded): the native CPU march took 2.02 times less time per step
at one thread on 192 × 192 × 300, and 1.81 and 1.85 times less at all threads on that grid
and on 384 × 384 × 600; per evaluation 1.35, 1.21 and 1.24 times. It holds 4 field volumes
against 15.

### Far-field statistics pass

The banded far-field pass costs about 2.3–2.6 % of one Runge–Kutta step on the native CPU
backend (192² and 384² Kerr runs, recorded by 2026-10-23; machine not recorded), paid at save
planes and periodic rows only; a whole-run check on an AMD EPYC 7763 moved
`far_field_stats` by 0.994. On an NVIDIA H200 and an AMD Instinct MI325X (2026-11-04) one
`aperture_spectrum` call at 768² × 600 took 15.4 and 14.4 ms with four circular holes and
33.3 and 33.8 ms with sixteen, about 1.5 ms a hole on top of the 9 ms `beam_spectrum` takes:
under 5 % of a saved plane at a few steps a plane.

## Memory

### Resident Adams storage

At maximum order five the native resident Adams engine held 29 complex field volumes,
eleven of them dense-output slopes and saved fields; allocating those only for
`interpolate_planes=True` reduces it to 18 ([backends.md §8](backends.md#8-memory-ledger)).
Peak resident memory of a reduced-resolution `kerr_beam_384` (600 temporal bins, four
threads, an 18-core arm64 laptop, recorded by 2026-10-23, model not recorded) fell from 7.53 to
5.92 GiB at 128² and from 28.2 to 21.8 GiB at 256², and is projected from 62.7 to 48.2 GiB
at 384². Field export adds a further host copy; no field-sized workspace scales with the
thread count.

## Boundaries

### The transverse edge

Each backend reduces the boundary monitor's numbers where its state lives: one check costs
an inverse transform of the state and a pass that sums it. After SEP 44's native read and
the growing backstop (`L/16` doubling to `L/4`), the default check added 2.6–3.6 % to
`line`, 9.1 % to `planar` and 3.9 % to `radial_filament_256` against no ramp and no reads,
at 18 threads of an Intel Xeon Gold 6240 (2026-11-08). On a device, a 768 × 768 × 600 Kerr
point with the check on took 1.30 times its time with it off on an NVIDIA H200 and 1.34 on
an AMD Instinct MI325X, an edge read 25 and 31 ms (2026-11-05). Bound threads overstate the
host's share: SciPy's transforms then take the main thread's single core (0.48 s with 16
workers against 0.11 s unbound on a 384² × 256 array, AMD EPYC 7763, 2026-11-04).

### A spectral walk-off absorber against the temporal collar

What the [spectral walk-off absorber](../guide/numerics/boundaries.md#a-spectral-walk-off-absorber)
saves is the temporal pass. Ten fixed `RungeKutta()` steps of an envelope Kerr launch in
fused silica on a field grid, eight threads of an Intel Xeon Gold 6240 on a shared machine
(2026-11-03, ratios of medians): the spectral absorber took 0.90–0.92 of the `split_step`
collar's time and 0.47–0.58 of the in-stage term's, on a 256-column line and a 64 × 64
Cartesian grid. A heavier response gives a smaller saving.

## Devices

The [tracking set](#the-tracking-set-at-020) shows where a GPU pays. Large Cartesian grids
gain most: `kerr_beam_384` takes 0.42–0.46 s on an NVIDIA H200 or an AMD Instinct MI325X
against 7.3–23.5 s on the CPUs. Modal problems are bound by kernel launches rather than
arithmetic: on an MI325X an Adams trial was 30 to 70 kernels of 2 to 9 µs launched one by
one, the device busy for a third of the run (2026-10-30;
[#357](https://github.com/LupoLab/solprop/issues/357)). The NVIDIA A40's binary64 rate is
a sixty-fourth of its binary32 rate, and its rows take two to eight times the H200's.

- **A Cartesian step.** A low-storage step at 768 × 768 × 600 took 239 ms on an H200 and
  224 ms on an MI325X, 112 to 144 ms of it in its eight rank-3 transforms (2026-11-04);
  changes to the passes beside them took it to 223 and 208 ms (2026-11-05).
- **Plans share one work area.** A device's plans of the scratch now share a work area: a
  768 × 768 × 600 run on an A40 fell from 38.5 to 22.6 GiB, the plans from 21.1 to 5.3 GiB,
  with the same time (2026-11-05).
- **Settings** (SEP 46, NVIDIA H200, 2026-11-08). Graph replay of an Adams trial's source
  pays (`ctc` 1.08 times slower without it); neither other team width does. An 8 GiB node
  chunk instead of 1 GiB takes 6 to 25 % off the spatial routes.
</content>
</invoke>
