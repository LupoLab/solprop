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

### Experimental Julia CPU prototype

Measured 2026-10-23 on an Apple M5 Max, 18 physical cores, 128 GiB RAM,
macOS 27.0.1 arm64, Python 3.14.5. Source: `1df3e4117d7f` plus the uncommitted
Julia prototype on `feature/julia-backend-prototype`. Julia 1.13.2,
PythonCall/JuliaCall 0.9.37, AcceleratedKernels 0.5.9, KernelAbstractions 0.10.1,
FFTW.jl 1.10.0, NumPy 2.5.2, SciPy 1.18.1, Numba 0.67.0, pyFFTW 0.15.1.
Julia's BLAS is ILP64 OpenBLAS 0.3.30. Both backends used FFTW `ESTIMATE`, no imported
wisdom, and the stated width for their numerical, FFT and BLAS pools. Julia's pool
counts were read back. Numba reported `blas_mechanism="uncontrolled"` on this Mac;
`VECLIB_MAXIMUM_THREADS` was set to the same ceiling, but its actual BLAS worker count
could not be read back. FFT, BLAS and elementwise work run in successive phases.

Two declared constant-argon scalar field-grid Kerr/THG capillaries: 125 µm radius,
1 bar, 800 nm carrier, 15 fs FWHM, 1 µJ, 400 fs window, 150 nm–4 µm grid limits,
4 or 16 modes. Both have 2049 spectral bins and 8192 fine time samples. The four-mode
case has 33 quadrature nodes; the sixteen-mode case has 129. The guide length is
0.01 m to define the collars; the measured window is only **ten prescribed 10 µm
steps**, ending at 0.1 mm. The same translated problem and initial field are used by
both arms. There are exactly 61 source evaluations per window, including FSAL reuse.

`julia/benchmark.py` takes three alternating fresh-process rounds, five warm repeats
per arm and round, after a compilation/plan warm-up. Each row below pools 15 samples;
spread is median absolute deviation (MAD). Propagation includes the Python shared
driver, trials, acceptance and quadrature diagnostics. Translation, construction,
initialisation, observation, statistics and output I/O are outside the timed window.
The last column is Numba time divided by Julia time; above one favours Julia.

| modes | threads | Numba median ± MAD (ms) | Julia median ± MAD (ms) | speed ratio |
|---:|---:|---:|---:|---:|
| 4 | 1 | 23.660 ± 0.195 | 32.150 ± 0.135 | 0.736 |
| 4 | 2 | 22.329 ± 0.162 | 23.573 ± 0.095 | 0.947 |
| 4 | 4 | 21.981 ± 0.444 | 22.601 ± 0.394 | 0.973 |
| 4 | 18 | 40.429 ± 0.333 | 70.004 ± 3.794 | 0.578 |
| 16 | 1 | 110.958 ± 1.615 | 184.714 ± 0.817 | 0.601 |
| 16 | 2 | 88.238 ± 1.342 | 101.876 ± 0.675 | 0.866 |
| 16 | 4 | 87.433 ± 1.357 | 76.508 ± 0.708 | 1.143 |
| 16 | 18 | 133.315 ± 1.226 | 507.968 ± 142.397 | 0.262 |

This is an indicative laptop comparison, not a standard-suite adoption verdict or an
exclusive-node measurement. The one-minute load ranged from 3.4 to 20.6, including
the benchmark itself. The 18-thread Julia sixteen-mode samples ranged from 315 to
983 ms and are particularly unstable; the regression is nevertheless much larger
than the observed spread. At four threads the equal-weight geometric-mean ratio is
1.055, but the conservative two-MAD ratio does not establish the required >5% saving.
At one thread the geometric-mean ratio is 0.665 and at 18 threads 0.389. **This initial prototype
established feasibility but did not meet the overall speed objective.** The CPU tuning
follow-up below supersedes this performance conclusion for the two declared cases. Four threads
are a useful starting point for these two cases, not a general sizing policy.

Across all pairs the largest relative final-field norm difference was
`2.763e-16`; source counts were exactly 61 in both arms. Independent NumPy-reference
qualification covers source evaluations, single/eight steps, both absorbers, THG on/off,
chunk tails, statistics and an adaptive accuracy envelope. Timing agreement alone is
not the accuracy qualification.

Median process-local construction costs were 2.70–3.04 s for Julia and 0.068–0.077 s
for Numba. First ten-step windows were 0.91–1.53 s and 0.20–0.33 s respectively;
these include compilation/cache loading and are not reproducible cold-compilation
measurements. Warm observation plus default host statistics cost about 0.62–0.79 ms
for four-mode Julia and 1.82–2.24 ms for sixteen-mode Julia, versus 0.61–0.68 ms and
1.62–1.71 ms for Numba. No full-run time is inferred.

The initial four-mode Julia serial pilot took about 45 ms. A Julia sampling profile
identified the elementwise Kerr loop and indexing as major costs. Removing redundant
bounds checks brought it to about 40 ms. A generic slab/SIMD traversal regressed to
51 ms and was discarded. The retained CPU specialisation traverses contiguous time
columns for the Kerr product, sharing its scalar arithmetic with the generic loop;
that brought the pilot to about 32 ms. No physics-specific approximation, fast-math
licence or numerical bound was changed. FFT/BLAS sizing and remaining elementwise
costs are still open optimisation work. Raw JSON, fields and profiles are local under
ignored `.benchmarks/`; the committed runner reproduces the experiment.

#### CPU tuning follow-up: providers, serial work and task sizing

Measured later on 2026-10-23, with the same two capillaries, ten-step schedule and
61 source evaluations. The Mac remained shared and contended. The Linux comparison
used an otherwise idle dual Xeon Gold 6240 (36 physical cores, 72 logical, 376 GiB),
Ubuntu 24.04.5, Python 3.14.7, under exclusive Slurm allocation of all 72 CPUs;
processes were restricted to the 18 physical cores of socket 0. Mac Python remained
3.14.5. Other Python versions are resolved by the same lockfile. Julia's private
runtime/depot were prepared independently on each host. Provider versions are
AppleAccelerate 0.7.0, MKL.jl 0.9.1 and MKL_jll 2025.2.0+0.

The retained changes are constant-table stage propagator reuse for exactly equal
step sizes; contiguous CPU sample traversal with shared scalar bodies; a 32768-sample
minimum task size for pointwise work and short reductions; and default FFT/BLAS
limits capped at four. Accelerate BLAS defaults to explicit serial mode. Changed
adaptive step sizes rebuild the propagators; explicit `recompute` disables reuse
between trials. Forward/inverse tables add twelve state-array equivalents compared
with the first prototype, counted in both memory admission and the owned ledger.

Julia requested the native CPU target (LLVM 20.1.8 reports `apple-m1` on this
M5 host), with inlining, `-O2`, automatic bounds checking, and local
`@inbounds` in validated kernels. LLVM inspection showed `<2 x double>` packed
arithmetic and no bounds-check calls in the Mac Kerr column kernel. No umbrella
fast-math, approximate functions or reassociation of protected reductions was added.
An alternating `-O3` pilot was indistinguishable from `-O2` (184.455 versus 184.288 ms
for the original 16-mode serial core), so it was discarded. A later chunk sweep
(129/64/32/16 nodes) measured 83.646/81.273/85.541/96.431 ms on the tuned Mac serial
case; the small 64-node gain did not justify a new default without broader evidence.

The remaining serial profile on the Mac put about 82% of samples in source evaluation,
including about 42% in Kerr traversal and 28% in FFT execution. Stage exponentials
ceased to be a leading cost with equal-step reuse. These are inclusive sampling
fractions, not independently additive timers. A matched cached/uncached pilot reduced
the 16-mode serial window from 334.310 to 299.541 ms during heavy contention. A
subsequent column/task-sizing pilot reduced a four-thread window from 253.680 to
215.398 ms. Absolute times from these separate contention periods must not be compared.

`julia/tune.py` alternates backend/provider order each repeat after a warm-up; setup,
initialisation, observations and provider switching are outside the propagation timer.
The Mac table pools eleven repeats per arm in one process. Julia and Numba numerical
widths are as stated; FFT widths are `min(width,4)`, and BLAS is one thread. Accelerate
was explicitly switched to serial mode, including the shared system BLAS used by
Numba. The one-minute load was 7.4–13.4. Values are median ± MAD in milliseconds.

| modes | numerical width | Numba | Julia/OpenBLAS | Julia/Accelerate | Numba / Accelerate |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 24.217 ± 0.141 | 25.543 ± 0.114 | 18.130 ± 0.062 | 1.336 |
| 4 | 4 | 22.409 ± 0.249 | 21.911 ± 0.441 | 14.362 ± 0.405 | 1.560 |
| 4 | 18 | 41.801 ± 0.387 | 32.493 ± 0.338 | 26.066 ± 0.321 | 1.604 |
| 16 | 1 | 114.200 ± 1.185 | 158.232 ± 1.576 | 86.795 ± 1.360 | 1.316 |
| 16 | 4 | 87.994 ± 1.545 | 132.588 ± 2.517 | 64.043 ± 3.825 | 1.374 |
| 16 | 18 | 123.200 ± 0.764 | 166.832 ± 1.038 | 90.209 ± 1.070 | 1.366 |

The Xeon sweep also kept numerical, FFT and BLAS work in successive phases. Using
18 workers everywhere was especially poor: the original sixteen-mode Julia median
was 937 ms, against 168 ms at four workers. With the tuned core, 18 Julia workers,
four FFT workers and four BLAS workers took 159 ms with MKL BLAS; four workers in
each pool took 142 ms. More workers did not improve either declared case.

MKL BLAS beat OpenBLAS in the controlled Xeon sweep: for four modes at one/four
workers, 56.276/36.947 ms versus 63.481/47.420 ms; for sixteen modes,
290.335/142.494 ms versus 309.104/158.918 ms. Each used seven alternating repeats.

The first mixed-process Linux provider experiment exposed an OpenMP affinity
interaction: loading MKL could bind the main thread before a later Numba constructor.
A subsequent Numba check warned about this binding. Its relative Numba timings are
not used as final speed evidence. The final comparison therefore uses fresh separate
backend processes, keeps `taskset` restricted to the same physical socket, and unsets
`OMP_PLACES` and `OMP_PROC_BIND`. Both backend order and FFT-provider order reverse
between the two rounds. Each arm has five warm repeats per round. All numerical,
FFT and BLAS widths equal the stated width; Julia uses MKL BLAS and Numba retains
its existing OpenBLAS/pyFFTW providers.

| modes | width | Julia FFT | Numba, ms | Julia/MKL BLAS, ms | speedup |
|---:|---:|:---|---:|---:|---:|
| 4 | 1 | FFTW | 91.079 ± 0.461 | 53.998 ± 0.091 | 1.687 |
| 4 | 1 | MKL | 92.048 ± 1.455 | 44.203 ± 0.703 | 2.082 |
| 4 | 4 | FFTW | 72.859 ± 0.399 | 35.752 ± 0.248 | 2.038 |
| 4 | 4 | MKL | 72.430 ± 0.390 | 29.459 ± 0.416 | 2.459 |
| 16 | 1 | FFTW | 524.158 ± 6.787 | 288.113 ± 0.709 | 1.819 |
| 16 | 1 | MKL | 502.271 ± 9.202 | 238.596 ± 0.450 | 2.105 |
| 16 | 4 | FFTW | 286.434 ± 3.386 | 139.598 ± 2.248 | 2.052 |
| 16 | 4 | MKL | 287.977 ± 1.808 | 119.454 ± 1.413 | 2.411 |

A separate nine-repeat Mac check added Numba's explicit `tabulate` option. Its
median changes relative to Numba's default were +1.8%, +1.2%, −0.04% and +1.8%
for four-mode serial/parallel and sixteen-mode serial/parallel respectively.
Julia remained faster in all four pairs. This does not establish that tabulation
is unhelpful for other Numba cases.


The largest final-field norm difference across these recorded comparisons was
`2.908e-16`; each arm performed exactly 61 source evaluations. On the Mac, sixteen
focused Julia-backend checks passed; two interop checks initially failed because
AppleAccelerate's AbstractFFTs extension intercepted dense-array FFT plans. Planning
from strided views retained FFTW explicitly, and both interop checks then passed.
Standalone Julia checks passed 22 assertions. On the Xeon, five short independent
NumPy-oracle cases passed with MKL BLAS and MKL FFT, covering THG on/off, both absorbers,
chunk tails, rejected trials and linear masks. The full suite was deliberately not
rerun for this tuning experiment.

The optional provider selector refuses unsupported platforms. A preliminary manual
MKL forwarding experiment omitted its ILP64 suffix and failed the numerical screen;
its timings are excluded. The corrected runner uses MKL's own forwarding routine.
The campaign restores its default FFTW preference after each run.

**The tuned configurations beat the existing Numba defaults in both declared cases,
serial and at four threads, on both hosts.** This is a result for these short windows,
not a general backend ranking, GPU qualification or adaptive full-run speed claim.
Accelerate is now the macOS default, with OpenBLAS retained as an explicit option
and the default on other prepared platforms. Linux MKL remains opt-in. Raw evidence is retained under ignored `.benchmarks/julia-tuning/`;
the source digests in its records identify the measured numerical snapshots. Final
metadata now labels the automatic Julia propagator policy as `tabulate`; earlier
Xeon records retained the inherited `recompute` label but explicitly recorded the
active `constant_table_exact_dz` cache.

#### Native comparison and isolated provider attribution

A subsequent rerun used `julia/compare.py`, with a fresh process per arm and
three alternating rounds of five warm windows. It compared the unchanged native
CPU backend from the same source snapshot as well as Numba and Julia. All arms
performed 61 source evaluations. The window, cases, FFT planning effort and
exclusions remain those above. All widths below are numerical and FFT widths;
BLAS was constrained to one on the Mac, and matched the width on Linux. Native's
kernel and FFT pool overrides were explicit in this primary comparison.

The Mac was reported quieter before this rerun, but activity monitoring still
showed other Python jobs and substantial security-scanner work. These are
shared-machine measurements, not idle-machine certification. The table includes
all fifteen repeats per arm, rather than selecting the fastest round; the wider
16-mode parallel spreads make its smaller differences less decisive.
Values are median ± MAD, milliseconds per ten-step window.

| modes | width | Numba/Accelerate | native/Accelerate | Julia/OpenBLAS | Julia/Accelerate |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 23.917 ± 0.634 | 27.728 ± 1.094 | 26.417 ± 0.886 | 18.865 ± 0.596 |
| 4 | 4 | 22.865 ± 0.457 | 18.574 ± 0.201 | 21.493 ± 1.712 | 13.178 ± 0.284 |
| 16 | 1 | 115.423 ± 2.509 | 134.059 ± 5.177 | 158.458 ± 2.904 | 90.461 ± 1.213 |
| 16 | 4 | 91.469 ± 5.430 | 71.677 ± 4.456 | 138.843 ± 5.739 | 65.328 ± 2.843 |

The Mac native build used Homebrew Clang 23.1.3, Release `-O3`, native architecture,
checked IPO, Kokkos 5.2.1 / Kernels 5.2.0 / FFT 2.0.0, Accelerate and FFTW 3.3.11.
It retained strict protected arithmetic and disabled fast-math. NumPy 2.5.2 and
SciPy 1.18.1 in this environment also link Accelerate; Numba therefore already had
that BLAS family. Julia's OpenBLAS is 0.3.30. Switching just Julia's BLAS to
Accelerate cut its median by 28.6%, 38.7%, 42.9% and 52.9% in table order.
The Accelerate Julia arm was 1.27–1.74 times faster than Numba and 1.10–1.48 times
faster than native's explicit-pool defaults in these measurements. The final-field
norm difference never exceeded `2.763e-16`. Julia/Accelerate construction took
3.35–3.59 s and the first ten-step call 1.05–1.13 s, versus native's 16–18 ms
construction and 21–146 ms first call. Shared host statistics/observation cost
0.66–1.86 ms for Julia and 0.30–0.86 ms for native, depending on case/width;
these costs are reported separately and excluded from the warm propagation table.

A separate two-round, seven-repeat native policy check tried `tabulate` and then
`tabulate` with automatic per-operation sizing under the same global cap. Source
inspection showed that native's resident DOPRI5 path does not consume this setting:
`_prepare_trial` passes the phase weights to the C++ trial, whose stage entry forms
the exponentials each time. Its low-storage integrator has a separate table option.
Thus the first comparison is a repeated baseline, not a native cache experiment;
the last comparison changes operation sizing. Its fields were identical to native's
default fields. Mac contention made its timing changes inconsistent: for example,
the 16-mode four-thread default varied from 122 to 95 ms by round, while automatic
sizing with tabulation gave 84 and 72 ms. It did not establish a faster native
configuration than the primary table. Linux repeats of the same policy check are
recorded below.

The final Xeon campaign used the same exclusive allocation and physical socket,
with `OMP_PROC_BIND=false` and no `OMP_PLACES`. It used three alternating rounds,
five warm windows each. Native was a local Release build with IntelLLVM 2026.1.1,
`-O3`, native architecture, IPO, precise protected arithmetic, and oneMKL 2026.1
for both BLAS and its FFTW-compatible FFT interface. Julia used its private
MKL 2025.2 library (GNU threading layer); these are the same library family,
not identical releases or OpenMP runtimes. Numba's modal `np.dot` uses NumPy's
OpenBLAS 0.3.34; SciPy also loads OpenBLAS 0.3.31.dev. Julia's default is
OpenBLAS 0.3.30. Loaded-library records confirm all stated thread limits.

| modes | width | Numba/OpenBLAS | native/MKL | Julia/OpenBLAS + FFTW | Julia/MKL + FFTW | Julia/MKL + MKL FFT |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 93.341 ± 0.359 | 65.599 ± 0.629 | 62.575 ± 0.385 | 55.246 ± 0.273 | 43.844 ± 0.580 |
| 4 | 4 | 73.597 ± 0.706 | 27.017 ± 0.185 | 42.438 ± 0.361 | 35.670 ± 0.361 | 29.443 ± 0.171 |
| 16 | 1 | 496.162 ± 2.912 | 405.264 ± 1.319 | 308.624 ± 1.197 | 290.550 ± 1.214 | 235.034 ± 3.543 |
| 16 | 4 | 293.951 ± 2.376 | 114.790 ± 0.221 | 155.835 ± 0.729 | 139.433 ± 0.405 | 118.514 ± 1.111 |

Switching **only Julia's BLAS** from OpenBLAS to MKL reduced the median by
11.7%, 15.9%, 5.9% and 10.5% in table order. Switching **its FFT** from FFTW to
MKL after that reduced it by a further 20.6%, 17.5%, 19.1% and 15.0%. Together,
these provider changes saved 29.9%, 30.6%, 23.8% and 23.9% relative to the same
Julia core with OpenBLAS/FFTW. The OpenBLAS/FFTW Julia arm was already 1.49–1.89
times faster than Numba. MKL accounts for about 21–38% of the total elapsed time
saved from Numba to the final Julia configuration in these four comparisons;
the remaining gain was present before switching providers. This is not a pure
language comparison: layouts, stage handling, loop scheduling and library builds
also differ.

With MKL BLAS and FFT, Julia was 1.50/1.72 times faster than native at one thread
for four/sixteen modes. Native took 8.2%/3.1% less time at four threads. Thus Julia
has not universally beaten native. All arms performed 61 source evaluations,
returned finite fields, and passed the same predeclared rounding screen; the
largest field norm difference was `2.952e-16`.

The native sizing check used two alternating rounds with seven warm windows per
arm. It omitted the redundant tabulation-only arm after the source inspection
above. Its automatic-sizing arm still records `tabulate`, which has no effect on
this DOPRI5 computation. Results are median ± MAD in milliseconds:

| modes | width | native, explicit pools | native, automatic sizing |
|---:|---:|---:|---:|
| 4 | 1 | 65.129 ± 0.266 | 65.462 ± 0.354 |
| 4 | 4 | 26.935 ± 0.285 | 29.174 ± 0.058 |
| 16 | 1 | 405.862 ± 1.152 | 402.272 ± 0.792 |
| 16 | 4 | 114.094 ± 0.473 | 114.565 ± 0.545 |

Automatic sizing did not improve the primary result on these two shapes: the
four-mode parallel case was slightly slower, and the other differences were small.
All fields were identical to the corresponding explicit-pool native field, with
61 source evaluations. Both exclusive jobs completed successfully and released
their allocations.


Both native builds were made from the unchanged native sources in this experiment's
checkout; no native optimisation was added. The full Python and native gates were
not run for this requested timing experiment. Focused numerical qualification from
the preceding tuning pass, static Python checks and the benchmark field screens are
reported separately. Raw worker JSON, field witnesses, build records and summaries
are retained under ignored `.benchmarks/julia-tuning/`. The checkout's default FFTW
preference was restored after the matrix.

#### Mode-averaged field extension

On 2026-10-23 the prototype enabled the existing mode-averaged transform: one
amplitude and one effective-area nonlinear node. Translation already supplies the
field conversion in synthesis and its reciprocal in the response prefactor, so
this extension uses the unchanged Julia numerical core. No frontend, schema,
physics coefficient or dependency changed. Envelopes/GNLSE remain unsupported.

The lightweight Mac comparison used `julia/compare.py --mode-averaged
--blas-threads 1 --arms numba_cpu native julia-accelerate-fftw`. It used the same
125 µm constant-radius, 1 bar argon capillary and 800 nm, 15 fs, 1 µJ pulse as the
multimode comparison: 150 nm–4 µm wavelength limits, 400 fs window, 2049 spectral
bins and 8192 fine time samples. The timed region was propagation alone over ten
prescribed 10 µm steps. Each arm performed exactly 61 source evaluations; setup,
first-window compilation, observation and statistics were timed separately.

Hardware was an Apple M5 Max with 18 physical cores, macOS 27.0.1, Python 3.14.5,
Julia 1.13.2, JuliaCall/PythonCall 0.9.37, NumPy 2.5.2, SciPy 1.18.1, Numba 0.67.0
and pyFFTW 0.15.1. Providers/builds were the Accelerate/FFTW configurations above.
Numerical and FFT widths were explicitly one or four, with serial BLAS requested
and FFTW `ESTIMATE`. Native reported folded synthesis/projection for this one-node
case. Julia used its constant-step propagator cache; Numba and native retained
their ordinary resident DOPRI5 policies. This compares backend implementations,
not just languages or BLAS providers.

Three alternating fresh-process rounds, each with five warm windows, gave the
following medians ± MAD in milliseconds. Other Python jobs and background system
work were active: these are indicative shared-machine measurements, not a quiet
machine performance qualification.

| numerical/FFT width | Numba | native | Julia |
|---:|---:|---:|---:|
| 1 | 7.900 ± 0.201 | 7.459 ± 0.684 | 3.484 ± 0.300 |
| 4 | 9.525 ± 0.407 | 8.471 ± 0.516 | 5.999 ± 0.459 |

The repeated round medians ranged over 7.881–7.967 / 6.496–8.067 /
3.184–4.321 ms for Numba/native/Julia at width one, and 9.108–11.151 /
8.004–9.318 / 5.551–7.100 ms at width four. All repeats, including one 51.4 ms
Numba outlier at width four, remain in the raw evidence. The median Julia speedups
were 2.27× over Numba and 2.14× over native at width one, and 1.59× / 1.41× at
width four. Every backend was faster with one thread for this small case; the
four-thread multimode setting should not be assumed optimal for one amplitude.
The largest relative final-field difference from Numba was `2.164e-16`.

Julia construction took 3.41–6.91 s and its first ten-step window 0.99–1.54 s.
Native construction took 16–47 ms and its first window 7.9–12.4 ms. These are
process-local startup measurements with existing caches, not cold installation
costs. Median observation plus default statistics took 0.435/0.346 ms for Julia
at widths one/four, 0.635/0.346 ms for Numba and 0.229/0.285 ms for native.

Focused NumPy-oracle checks cover Kerr with/without THG, both temporal absorber
schemes, source/single/eight-step agreement, default and on-axis statistics,
reinitialization, non-aliasing observations, memory accounting, rejected candidates,
linear-only propagation, exact hard-mask zeros, and adaptive/public output.
Mode averaging adds no Julia kernels; the resolved cases exercise the same core.
The full suite and production gate remain deferred for this experiment. Raw
worker JSON and field witnesses are retained under ignored
`.benchmarks/julia-mode-averaged/mac/`.

#### Radial field/Kerr extension

On 2026-10-23 the prototype added scalar radial Kerr propagation in constant media.
The translated QDHT synthesis/projection matrices use the existing Julia real
matrix pipeline. The shared `radial_absorber_matrix` composes the radial loss
once at construction; Julia applies it as an additive in-stage source independent
of the material prefactor under both temporal-absorber schemes. Default radial
statistics use `collect_spatial` on the host. There is no frontend, schema,
dependency or native-backend change, and no radial-specific performance tuning.

The two declared cases have 64 and 256 radial nodes out to 1 mm. They use a
constant 1 bar argon medium, an 800 nm, 30 fs Gaussian pulse with 100 µm waist
and peak power 0.9 times the critical power computed from the existing optmat
coefficient. The field grid covers 300 nm–2 µm with a 200 fs requested window,
513 retained spectral bins and 2048 fine time samples. Kerr includes THG; plasma
and Raman are absent. Spatial and temporal absorbers are enabled with the 1 cm
reference problem length, while each timed window propagates only ten prescribed
10 µm steps. All radial nodes fit in one chunk in both benchmark cases.

`julia/compare.py --radial --blas-threads 1 --arms numba_cpu native
julia-accelerate-fftw` runs three alternating fresh-process rounds with five warm
windows each, at one and four numerical/FFT threads. Providers are Accelerate
and FFTW `ESTIMATE`, with serial BLAS requested. Julia retains its constant-step
propagator cache; Numba and native use their resident DOPRI5 policies. Construction,
first-window compilation, observations and default spatial statistics are excluded
from propagation timings and measured separately. The machine is the same Apple
M5 Max, macOS 27.0.1, Python 3.14.5 and private Julia 1.13.2 environment above.
Other Python jobs and a compiler were active throughout parts of the campaign;
this is a shared-machine comparison, not a quiet-machine qualification.

All arms performed exactly 61 source evaluations per window and returned finite
fields. The maximum relative final-field norm difference from Numba was
`2.296e-16`. Values below are median ± MAD, milliseconds per ten-step window,
using all fifteen repeats per arm:

| radial nodes | numerical/FFT width | Numba | native | Julia |
|---:|---:|---:|---:|---:|
| 64 | 1 | 91.988 ± 4.752 | 103.560 ± 15.569 | 64.568 ± 4.629 |
| 64 | 4 | 72.849 ± 1.420 | 59.003 ± 1.265 | 55.476 ± 0.409 |
| 256 | 1 | 646.248 ± 34.545 | 782.724 ± 34.778 | 626.116 ± 23.168 |
| 256 | 4 | 415.985 ± 32.450 | 364.443 ± 28.814 | 309.763 ± 19.236 |

Repeated round-median ranges expose the machine drift beyond within-round noise:

| radial nodes | width | Numba round medians (ms) | native round medians (ms) | Julia round medians (ms) |
|---:|---:|---:|---:|---:|
| 64 | 1 | 73.330–96.740 | 87.991–122.582 | 59.939–72.737 |
| 64 | 4 | 71.429–76.898 | 53.688–59.579 | 55.277–55.668 |
| 256 | 1 | 540.503–658.746 | 720.760–800.047 | 524.763–678.375 |
| 256 | 4 | 383.535–534.371 | 347.728–482.840 | 295.323–484.305 |

Julia's median ratios over Numba are 1.42×, 1.31×, 1.03× and 1.34× in table
order, and over native 1.60×, 1.06×, 1.25× and 1.18×. The 256-node serial result
does not establish a Julia advantage over Numba; its ordering reverses between
rounds. The four-thread native margins are also provisional: native wins one
64-node round, and the final 256-node round is essentially tied. The larger
problem benefits from four threads, but these cases do not establish a universal
thread-sizing policy or an idle-machine speedup.

Julia construction took 3.96–5.34 s, and first-window execution 1.18–2.12 s.
Native construction took 19–35 ms, and first-window execution 53–793 ms. These
are process-local startup measurements with existing caches. Julia-owned arrays
total 25.2 MiB for 64 nodes and 102.2 MiB for 256 nodes, excluding the runtime/JIT,
opaque FFT plans and retained translated host tables as declared by its ledger.
Median observation plus default spatial statistics, also in milliseconds:

| radial nodes | width | Numba | native | Julia |
|---:|---:|---:|---:|---:|
| 64 | 1 | 1.519 | 6.148 | 1.606 |
| 64 | 4 | 1.823 | 2.161 | 1.954 |
| 256 | 1 | 6.885 | 109.633 | 9.686 |
| 256 | 4 | 5.998 | 33.058 | 8.450 |

The statistics implementations differ; their timing differences have not been
profiled, and no native code was changed. No complete-run cost is inferred.

Numerical checks include a broad 24-node beam that reaches the spatial collar,
so absorption is exercised at meaningful amplitude even though the benchmark's
beam is well contained. Nine focused radial cases cover source/single/eight-step
agreement with NumPy, Kerr with/without THG, both temporal absorber schemes,
chunk tails, default spatial statistics and optional peak intensity, adaptive
public output, linear-only diffraction, exact masks, reinitialization, memory
accounting and rejected-state preservation. The isolated radial-loss cases set
the material prefactor to zero and compare against independent node-space loss
followed by the forward Hankel map. The fourteen existing resolved/mode-averaged
cases also pass, as do 22 Julia assertions at one/four task widths. The full suite,
documentation build and production gates remain deferred for this experiment.
Raw evidence is retained under ignored `.benchmarks/julia-radial/mac/`.


#### Cartesian field/Kerr extension

On 2026-10-23 the prototype added scalar Cartesian Kerr propagation in constant
media through the existing `Simulation.from_free` frontend. The numerical engine
uses planned transverse complex FFTs and chunked real temporal transforms. The
shared scalar Kerr body handles THG and analytic-signal Kerr. Both temporal
absorber schemes and the independent spatial collar are included. Compact
frequency/transverse coefficient vectors supply the longitudinal root, evanescent
cap, frame, spectral/transverse attenuation and prefactor. The compensated square
subtraction is protected. A single reusable forward/inverse propagator pair
serves the current DOPRI5 stage; there is no dense Cartesian operator or seven-stage
propagator cache. The seven derivative arrays remain an explicit experimental
memory cost. There is no frontend, schema, dependency or native-source change.

The declared timing cases use 32×32 and 64×64 Cartesian grids spanning ±500 µm
in both axes, with the radial entry's constant argon, 800 nm/30 fs pulse,
100 µm waist and 0.9 critical power. They retain 513 spectral bins and 2048 fine
time samples. Each propagation window has ten prescribed 10 µm steps and 61
source evaluations, with construction, compilation, observations and default
spatial statistics timed separately. All arms use recomputed separable flow.
The Cartesian propagation path contains no BLAS contractions. On the Xeon,
the comparison sets **all** library limits to the numerical width, avoiding an
unrelated serial global MKL cap when requesting four-thread FFT plans. The arms
are Numba/FFTW, native/oneMKL FFT, Julia/OpenBLAS+FFTW and Julia/MKL+MKL FFT;
the Julia BLAS labels identify the loaded provider, not a Cartesian BLAS workload.

The first Mac campaign was interrupted after the maintainer reported heavy
contention and is retained only as diagnostic evidence. A replacement campaign
used three alternating rounds of five warm windows per arm, with one or four
numerical/FFT threads and one BLAS thread. The M5 Max has 18 physical cores;
Python is 3.14.5 and Julia is 1.13.2. Native uses Clang 23.1.3 Release/O3, IPO,
native architecture and FFTW 3.3.11. Julia uses FFTW.jl 1.10.0 and AppleAccelerate
0.7.0 (the latter does no BLAS work in this path). These **initial-implementation**
medians and median absolute deviations are milliseconds per ten-step window:

| Grid | Threads | Numba | Native | Julia |
|---|---:|---:|---:|---:|
| 32×32 | 1 | 1580.2 ± 9.2 | 2031.2 ± 9.4 | 2125.1 ± 9.3 |
| 32×32 | 4 | 912.5 ± 11.5 | 640.1 ± 6.5 | 651.1 ± 14.4 |
| 64×64 | 1 | 7236.0 ± 64.1 | 9107.5 ± 89.0 | 9513.7 ± 60.9 |
| 64×64 | 4 | 2655.1 ± 171.5 | 2730.3 ± 95.9 | 2966.2 ± 13.0 |

The replacement started with intermittent background load: the three 32×32,
four-thread Julia round medians were 745.8, 651.1 and 639.3 ms; corresponding
native medians were 840.0, 633.6 and 637.6 ms. Do not treat the whole campaign as
perfectly idle. All windows performed 61 source evaluations. The largest final
relative field difference from Numba was 2.58e-16. Julia owned 264.5/750.2 MiB
for the two grids, excluding opaque FFT plans; observation plus host statistics
cost approximately 20/78 ms separately.

A subsequent serial Julia sampling profile of three large-grid windows assigned
8632 of 20224 attributed trial/profile samples (about 43%) to stage propagator
construction. The retained optimisation skips the identity stage arithmetic when
FSAL supplies its derivative, fills an exact identity for the first zero-distance
stage, and reuses the current scratch pair for consecutive equal stage distances.
No dense operator cache or new numerical approximation is introduced. Alternating
three-repeat A/B screens of the large grid reduced serial time from
9568.9 ± 10.6 to 8448.9 ± 10.7 ms, and four-thread time from
2919.3 ± 17.8 to 2724.7 ± 10.0 ms. These are separate local A/B measurements,
not substitutions into the initial campaign's table. Tail-width FFT plans and
strided-view traversal were tried but not retained: one combined variant helped
serial execution further, but regressions in the parallel case erased the benefit.

The adapter now accepts `SOLPROP_FFTW_PLAN=estimate|measure|patient`, with
`estimate` retained as default. A fresh-process `measure` screen, one round of
three warm windows, used the retained propagator optimisation and requested the
same effort for all backends:

| Grid, four threads | Numba | Native | Julia |
|---|---:|---:|---:|
| 32×32 | 900.6 ± 0.5 | 619.2 ± 5.0 | 534.7 ± 0.9 |
| 64×64 | 2573.6 ± 14.9 | 2683.0 ± 2.2 | 2669.1 ± 20.9 |

Julia setup (including runtime loading/planning) was 4.7/5.1 s and its first
window 1.9/4.0 s. Its warm statistics cost was 19/77 ms, outside propagation.
The final relative field difference stayed below 2.61e-16. These short screens
do not show a universal Julia advantage, and their planning improvement is small
on the large grid.

A separate `estimate` screen (one round, three repeats, retained optimisation)
expanded the equal numerical/FFT caps to 2, 8 and 18. At 18 threads native reached
420.1 ± 7.3 ms on 32×32 and 1258.6 ± 11.9 ms on 64×64. Numba reached
968.9 ± 55.9 and 1677.5 ± 0.7 ms; Julia reached 1087.8 ± 80.7 and
5012.4 ± 431.4 ms. Thus the four-thread comparisons must not be presented as the
best achievable native performance. Julia's wider equal FFT/task pools regress
on this Mac. The subsequent large-grid screen separates numerical and FFT caps
rather than assuming that more FFT threads are better.

The subsequent 256×256 Mac screen retained the same 513 spectral bins, 2048
fine time samples, pulse and ten prescribed steps. It used `measure` plans and
one first window followed by **one warm window per arm**. These are exploratory
measurements, not a multi-round speed qualification. Numerical and FFT caps are
listed separately; BLAS was capped at one thread. The machine has 128 GiB RAM.

| Backend | Numerical / FFT cap | Warm window (s) | Peak process RSS (GiB) | Owned arrays (GiB) |
|---|---:|---:|---:|---:|
| Julia | 4 / 4 | 44.410 | 13.55 | 8.720 |
| Julia | 18 / 4 | 29.609 | 13.69 | 8.720 |
| Numba | 18 / 18 | 35.078 | 9.67 | 8.120 |
| Native | 18 / 18 | 27.513 | 9.58 | 7.743 |

All four completed with 61 source evaluations. The largest relative field
norm difference from Numba was 2.56e-16. Julia's 18/4 policy took 16% less time
than Numba and 8% more than native in this screen. Raising its numerical cap
while keeping four FFT threads reduced time by one third relative to Julia 4/4;
raising all pools together had failed on the smaller grid. This supports explicit
operation-level caps, not an assertion that Julia automatically tunes every pool.
The existing default FFT cap of `min(numerical_threads, 4)` already permits this
combination; the timings do not establish an optimum for other shapes.

Julia setup was 22.0–23.1 s and the first window 29.4/45.3 s for 18/4 and 4/4.
Numba setup/first window were 1.63/34.94 s; native 0.54/28.04 s. Observations and
host statistics cost 1.34–1.36 s for Julia, 1.10 s for Numba and 0.70 s for native,
separate from propagation. Peak RSS includes runtime, frontend, workspace and
transient allocations; owned bytes count only the declared backend arrays.
At this resolution 256×256 therefore fits comfortably on this 128 GiB Mac.
The measured final Julia source digest is
`07b7a624fa144b13e62d81c9da7aea45de6f5dc4e1e513a8d75fd146c3053ba6`
(SHA256 of the sorted concatenated `julia/src/*.jl` files).

An initial Xeon pilot used a serial global MKL cap; those records are excluded
from the final library comparison. The replacement uses exclusive Slurm access,
one physical socket and equal global/numerical/FFT thread limits. After the user
prioritised 256×256, the remaining small-grid rounds were stopped: one complete
matrix of five warm windows per arm is retained, with partial later rounds kept
only as diagnostics. A separate job qualifies the retained optimisation and
screens the large grid; it is queued behind other work and has not yet run.

The completed original-implementation Xeon matrix used one socket of a dual
Xeon Gold 6240 (18 physical cores available to each worker), Ubuntu 24.04.5,
Python 3.14.7 and Julia 1.13.2. Native was built with IntelLLVM 2026.1.1,
Release/O3, IPO and native architecture, using oneMKL 2026.1. Julia's MKL package
used MKL 2025.2. The first complete round's five-repeat medians ± MAD are
milliseconds per window:

| Grid | Threads | Numba/FFTW | Native/MKL FFT | Julia/FFTW | Julia/MKL FFT |
|---|---:|---:|---:|---:|---:|
| 32×32 | 1 | 5314.4 ± 1.0 | 9896.5 ± 5.5 | 5332.5 ± 9.0 | 5234.1 ± 7.1 |
| 32×32 | 4 | 3839.8 ± 3.0 | 2655.7 ± 1.7 | 1762.8 ± 1.2 | 1790.7 ± 3.5 |
| 64×64 | 1 | 26253.0 ± 5.9 | 42128.6 ± 26.4 | 25493.8 ± 7.1 | 23573.4 ± 28.6 |
| 64×64 | 4 | 11466.1 ± 2.7 | 11213.8 ± 3.1 | 8218.2 ± 4.2 | 7830.1 ± 19.5 |

All four complete cases passed the runner's final-field screen before round two
started, with 61 source evaluations per window. The exact maximum difference
will be reaggregated from the saved witnesses. MKL FFT improves Julia's large
case by about 5–8% here, while the small four-thread case slightly favours FFTW;
BLAS attribution is inapplicable to this FFT-only propagation path. The earlier
Julia implementation already outperformed Numba/native on these four-thread
Xeon cases. Its source digest is
`d64850facfc343b7ad87db6a83b27889ebbf28c92251119e8f679ae07467407f`.
The post-optimisation Xeon 256×256 screen remains pending; do not infer it from
Mac timings or these smaller-grid measurements.

All 44 focused optional-backend cases passed across the Mac verification runs,
with all ten Cartesian cases repeated after the retained optimisation using
four Julia/FFT threads and measured FFTW plans (10.72 s). The same ten Cartesian
cases passed on the Xeon with FFTW before the optimisation. They cover a rectangular
8×10 grid, chunk tails, Kerr with/without THG, both temporal absorbers, source and
single/eight-step comparisons against NumPy, adaptive public output, default
statistics, exact masks, rejected-state preservation, repeated initialization and
memory accounting. An independent two-dimensional FFT/loss/FFT construction tests
spatial absorption with the material prefactor disabled. Synthetic squared-wavevector
inputs exercise the protected cancellation, propagating/evanescent branches and
attenuation cap independently of the physical-grid validator. Numerical bounds were not widened. Ruff, targeted typing, spelling and whitespace checks pass.
The 22 standalone Julia assertions also pass with measured plans after the final
change. The full suite, documentation build and production gates remain deferred for this
experiment. Raw evidence is retained under ignored `.benchmarks/julia-cartesian/`.


Indicative ratios, not absolute times. Each is a short window (see above) on one
machine; the rows' cases, grids and step lengths differ, so rows share no scale. They say
which backend or method to expect to win, and by roughly how much.

#### Initial SEP 30 core extension

The experimental Julia backend now runs all ten declared core case/track pairs:
RDW, comb-to-continuum and the 256-node radial filament with exponential Adams,
DOPRI5 and the prescribed ten-step DOPRI5 schedule, plus the 384² envelope Kerr
beam with the prescribed ten-step RK4IP schedule. This adds execution of the
existing scalar plasma, Raman and longitudinal stage bundles, resident Adams
history and Cartesian complex temporal transforms. The frontend and schema 34
are unchanged. GPU execution remains unqualified.

Focused checks passed on both Apple Silicon and Linux x86-64. The Linux lane had
54 passing Python checks, covering source evaluations, eight-step comparisons,
rejected-state preservation, Adams reset/resume, protected exponential moments,
memory accounting and rectangular Cartesian envelopes. The standalone Julia
checks also passed (22 assertions). The experiment does not claim the full
Python or native quality gates.

For an initial accuracy screen, default-tolerance adaptive runs were compared
with the same NumPy problem at `rtol=1e-8`. Both Julia and native passed the
suite's existing field, in-band and weak-spectrum accuracy envelopes. The
largest relative field differences for Julia were:

| Case | Adams | DOPRI5 | Existing accuracy envelope |
| --- | ---: | ---: | ---: |
| RDW | 6.19e-8 | 2.23e-7 | 1e-4 |
| Comb-to-continuum | 5.91e-8 | 8.45e-8 | 1e-5 |
| Radial filament, 256 nodes | 1.31e-6 | 2.21e-6 | 1e-4 |

This is a non-exhaustive check against one tighter reference, not a reviewed
convergence receipt for production adoption. The focused Cartesian check uses
the same eight-step RK4IP history as NumPy on a small rectangular grid.

An initial Apple M5 Max screen used the actual SEP 30 worker with one first
window and one warm window per row, four numerical threads and a requested
serial BLAS. Julia used FFTW and Accelerate; the native build used its existing
CPU operation-size policy. Adaptive windows include their declared observations
and default statistics. The timings below are single warm samples in backend
blocks, **not alternating A/B evidence** and not final performance claims:

| Case / track | Julia / s | Native / s | Source evaluations per measured window |
| --- | ---: | ---: | ---: |
| RDW / Adams | 0.1075 | 0.0389 | 21 |
| RDW / DOPRI5 | 0.1316 | 0.0574 | 60 |
| Comb-to-continuum / Adams | 1.0771 | 0.9532 | 19 |
| Comb-to-continuum / DOPRI5 | 2.6628 | 2.8458 | 60 |
| Radial filament / Adams | 4.0032 | 2.2582 | 230 |
| Radial filament / DOPRI5 | 7.0176 | 2.8252 | 523 |
| 384² Kerr beam / RK4IP | 47.5716 | 26.9338 | 40 |

Work counts matched between the two backends in this screen. In particular, the
frozen adaptive radial window takes many more than ten steps; subsequent tuning
uses its existing ten-step fixed schedule to keep exploratory runs light. The
working core extension establishes feasibility, but these initial results do
not meet the performance objective for most core tracks.

`julia/core_compare.py` reuses `benchmarks.ab_worker` for the follow-up:
one construction and warm-up per case per worker, resets and lead-in outside the
timer, alternating arm order over rounds, actual work counters and retained-field
comparisons. It records compiler/provider identity and backend thread metadata.
`julia/core_profile.py` samples the warmed numerical window and records Python
controller costs separately. These are exploratory tools, not substitutes for
the suite's qualification and adoption process.

#### Xeon core provider screen

The 2026-10-24 follow-up uses an exclusive dual Xeon Gold 6240 allocation,
with each worker restricted to the 18 physical cores of one socket. The first
screen retains the unchanged SEP 30 **ten-step fixed DOPRI5 schedules** for
RDW, CtC and radial filament 256. All windows take 61 source evaluations,
10 accepted steps and no rejected steps. Construction, compilation, lead-in,
reset and final observations are outside the measured window. These are one
screening round with two warmed repetitions, not a completed adoption comparison.

The native arms use the same source revision: GCC 16.2.0, OpenMP OpenBLAS
0.3.34 and FFTW 3.3.11 in a distribution build targeting x86-64-v3; and
IntelLLVM 2026.1.1 with MKL 2026.1 in a local native-architecture build. They
compare **compiler/library/build bundles**, so differences cannot be attributed
to the compiler alone. Julia 1.13.2 uses OpenBLAS 0.3.30 or MKL 2025.2.0,
and FFTW 3.3.12 or the MKL FFT provider. Julia's library switches use fresh
processes; BLAS and FFT providers are varied separately.

Median propagation seconds before the new plasma optimizations:

| Backend and libraries | CPU budget | RDW | CtC | Radial 256 |
|---|---:|---:|---:|---:|
| Native GCC / OpenBLAS / FFTW | 1 | 0.3307 | 14.2389 | 2.2021 |
| Native Intel / MKL | 1 | 0.3669 | 12.6077 | 2.4805 |
| Julia OpenBLAS / FFTW | 1 | 0.6454 | 16.7483 | 3.4371 |
| Julia MKL / FFTW | 1 | 0.6284 | 16.5619 | 3.6948 |
| Julia MKL / MKL FFT | 1 | 0.6093 | 14.1921 | 3.4691 |
| Native GCC / OpenBLAS / FFTW | 4 | 0.1316 | 5.5256 | 0.7313 |
| Native Intel / MKL | 4 | 0.1388 | 4.8480 | 0.6859 |
| Julia OpenBLAS / FFTW | 4 | 0.2796 | 7.4851 | 1.4772 |
| Julia MKL / FFTW | 4 | 0.2739 | 7.5430 | 1.5278 |
| Julia MKL / MKL FFT | 4 | 0.2627 | 6.5852 | 1.3254 |

At these budgets Julia's numerical, BLAS and FFT pools have the stated cap;
native retains its recorded operation-size policy within the same budget.
Libraries execute in successive operations. Final field relative L2 differences
against the GCC native arm are at most `6.21e-16`. This matched-schedule screen
supplements the focused NumPy checks; it does not replace adaptive qualification.
The initial core implementation is slower than native in all these comparisons.
Changing only Julia's BLAS has little benefit for CtC; changing its FFT provider
accounts for most of the improvement in that case.

Warm serial profiles identify the next work: plasma takes about 74% of RDW,
with its two filtered integrations accounting for about 41% of the total.
Python stage preparation takes about 5%. CtC spends about 44% in plasma and
32% inside FFTW; radial has substantial dense-transform work as well. These
measurements motivate independent FIR traversal, exact-zero support pruning,
and separate thread limits for modal, Raman and transverse FFT plans. Each
candidate is checked and measured separately before it is retained.

The first retained change separates the CPU FIR pass from its dependent
trapezoidal prefix. Independent interior time samples use SIMD, preserving each
sample's tap-addition order, reflected endpoints and the strict prefix. The
destination doubles as filter scratch, so there is no new owned array. The
generic traversal remains available for other array types. Eight focused core
checks and 46 standalone Julia assertions pass. Three alternating before/after
rounds, each with two warmed repetitions, give the following six-sample medians
and observed minima/maxima, using OpenBLAS and FFTW throughout:

| Case | Budget | Before, s [min–max] | FIR traversal, s [min–max] |
|---|---:|---:|---:|
| RDW | 1 | 0.6477 [0.6397–0.6560] | 0.5876 [0.5841–0.5914] |
| RDW | 4 | 0.2792 [0.2786–0.2829] | 0.2698 [0.2670–0.2778] |
| CtC | 1 | 16.6374 [16.3032–16.8851] | 16.0861 [16.0067–16.4331] |
| CtC | 4 | 7.5081 [7.4867–7.5534] | 7.3036 [7.2585–7.3312] |
| Radial 256 | 1 | 3.4378 [3.4184–3.4477] | 3.3405 [3.3283–3.3540] |
| Radial 256 | 4 | 1.4684 [1.4498–1.4733] | 1.4630 [1.4327–1.4976] |

Work counts remain 61 sources per ten steps. All six retained matched fields
have zero measured difference; bitwise agreement is an observation, not the
qualification requirement. The serial RDW reduction is 9.3%; radial at four
threads has no clear improvement within the observed spread.

A second CPU change restricts filtering and the running prefix to exact nonzero
support, including the FIR halo, and fills the constant tail. Repeated identical
exposure values reuse the same exponential. There is no numerical cutoff or
fast-math approximation. Eight core checks and 106 standalone assertions pass,
including empty support, support at reflected endpoints and short overlapping
halos. An initial one-round/two-warm screen gives:

| Case | Budget | FIR traversal / s | Exact-support traversal / s |
|---|---:|---:|---:|
| RDW | 1 | 0.5863 | 0.3811 |
| RDW | 4 | 0.2751 | 0.2085 |
| CtC | 1 | 15.7050 | 14.3263 |
| CtC | 4 | 7.3416 | 6.5761 |
| Radial 256 | 1 | 3.3351 | 2.9454 |
| Radial 256 | 4 | 1.4728 | 1.3624 |

All six retained fields have zero measured difference. This candidate is retained
for paired confirmation; the screen is not a final speed estimate. Final observation
and default statistics are outside these fixed windows and recorded separately:
about 0.30–0.33 s for RDW, 0.31–0.50 s for CtC and 0.07–0.10 s for radial.
Their cost remains inside adaptive benchmark windows that request observations.

The same allocation also screened the full 384² × 600 envelope Kerr beam, using
the unchanged ten-step RK4IP window and 40 source evaluations. All workers have
an 18-core budget, Julia requests serial BLAS, and FFT width is explicit below.
One first and one warm window per arm give these **single warm samples**:

| Backend / transform provider | FFT width | Propagation / s |
|---|---:|---:|
| Native GCC / OpenBLAS / FFTW | operation-sized within 18 | 29.0938 |
| Native Intel / MKL | operation-sized within 18 | 25.0263 |
| Julia / FFTW | 4 | 127.0944 |
| Julia / FFTW | 18 | 78.3289 |
| Julia / MKL FFT | 4 | 92.7281 |
| Julia / MKL FFT | 18 | 66.4920 |

The maximum matched-field relative L2 difference is `6.57e-16`. Julia owns
11.762 GB of arrays, versus about 7.566 GB for native at this budget; opaque
library storage, runtime and observation copies are outside these ledgers.
Julia remains substantially slower on this large case. A subsequent threaded
Julia signal-sampling attempt terminated on SIGUSR1 after the timing screen
had finished. Explicit phase timers in a separate source snapshot instead
measured a 127.42 s FFTW-four-thread window: transverse FFTs took 63.60 s,
temporal FFTs 22.14 s, input copies 10.10 s, padding/cropping 10.94 s and
source finalization 6.62 s. These are diagnostic timings, motivating a test of
contiguous transverse planes before any layout change is adopted.

A subsequent one-round spatial-layout screen passed eight rectangular source,
ten-step and memory checks. At 18 numerical threads, explicitly transposing a
scratch volume around each spatial FFT took 122.21 s with four FFT threads,
83.93 s with 18 FFT threads and 76.18 s with MKL/18. The same-job strided/four
baseline was 126.69 s. Parallelizing large copies reduced the contiguous/18
FFTW arm to 74.43 s. The largest retained-field difference was `1.76e-18`.
This extra-buffer layout is not adopted: these results do not justify another
field volume, and it introduces repeated full-volume transposes. The copy
change remains a candidate for separate confirmation.

An additional one-round screen kept the strided layout and parallel copies,
then isolated the inverse-FFT normalization and exactly-real longitudinal-root
branches. The 29 focused source, propagation, statistics and memory checks pass.
The same arithmetic order and scale are retained; normalization distributes
independent samples, while the root fast path preserves signed zeros and the
compensated subtraction. With 18 numerical/FFT threads and serial BLAS, single
warm ten-step samples are:

| Julia configuration | 384² × 600 / s | 192² × 300 / s |
|---|---:|---:|
| FFTW, serial normalization | 67.5716 | 8.8957 |
| FFTW, parallel normalization | 61.0018 | 8.0559 |
| FFTW, parallel normalization and real root | 57.3138 | 7.4538 |
| MKL FFT, parallel normalization and real root | 45.0683 | 6.1457 |

The same-provider retained fields are exactly equal; changing to MKL has relative
L2 differences below `1.97e-18`. At four threads, parallel normalization changed
the three core fixed windows from 0.2082/6.5530/1.3530 s to
0.2091/6.3260/1.3061 s for RDW/CtC/radial. These remain screening results
pending paired confirmation. The previously measured native 384² window was
about 25–29 s, so these Julia improvements do not yet establish parity.

Keeping the physical-space volumes permanently transposed avoids the extra
volume, but its chunk gathers/scatters offset the spatial FFT benefit. Twenty
source/propagation/plasma checks pass. At eighteen workers, single warm
FFTW/estimate samples are 7.4395/57.2536 s for the existing strided 192²/384²
layouts and 7.9999/63.6858 s for the resident spatial layout. Resident MKL gives
6.8103/54.5987 s, slower than the preceding strided MKL screen; resident
FFTW/measure gives 7.3967/60.0648 s. The maximum retained-field relative L2
difference is `2.01e-18`. The resident layout is not adopted: there is no clear
benefit over the existing strided implementation, and the large FFTW case
regresses by about 11%.

The strided implementation then omitted the additive absorber transforms when
both physical-space and in-stage temporal loss are exactly absent. The guard
does not change split-step absorption or diagonal spectral loss. Twelve focused
field/envelope checks cover absent, temporal and physical absorbers, both schemes,
source evaluation, ten steps and chunk tails. All pass against NumPy.

Three alternating rounds with two warm windows per round confirm this saving
on the same eighteen-core allocation, FFTW/estimate with eighteen FFT workers,
serial BLAS, parallel copies and normalization, and the exact real-root branch:

| Case | Previous median [range] / s | Omit zero loss median [range] / s | Time reduction |
|---|---:|---:|---:|
| 192² × 300 | 7.4146 [7.3425, 7.4453] | 5.7188 [5.6989, 5.8551] | 22.9% |
| 384² × 600 | 56.8249 [56.6887, 57.0646] | 43.5377 [43.4297, 43.5950] | 23.4% |

Each measured window still takes ten RK4IP steps and forty source evaluations.
The retained fields are exactly equal. Allocation is unchanged for this A/B;
setup, first windows and observations are excluded and recorded separately.
Native already guards this absent source, so this finding requires no native
change. Confirmation against the fresh native builds and MKL FFT is separate.

The retained changes were then combined without either rejected spatial layout.
All 43 focused core, envelope, statistics, line, Cartesian-plasma and absorber
checks pass. Three alternating rounds with two warm 192² windows give
5.7692 s [5.7020, 5.8370] for the tested candidate and
5.7783 s [5.7100, 5.8208] for the combined source, with exactly equal retained
fields. The combined version is adopted in the source experiment, including
Cartesian field plasma, parallel copies and normalization, the real-root branch
and omission of exactly absent additive losses.

The final large-beam comparison repeats both grids with the fresh local native
builds and combined Julia source. All workers use an eighteen-core budget;
Julia uses OpenBLAS with one BLAS worker throughout. Three alternating rounds,
two warm repetitions per round, give six-sample medians and observed ranges:

| Backend / FFT configuration | 192² × 300, s [min, max] | 384² × 600, s [min, max] |
|---|---:|---:|
| Native GCC 16.2 / OpenBLAS / FFTW | 3.4723 [3.4605, 3.5089] | 31.2511 [31.0906, 31.4418] |
| Native Intel / MKL | 2.8504 [2.8250, 2.8726] | 25.0391 [24.9407, 25.0858] |
| Julia / FFTW, temporal 18, transverse 18 | 5.7781 [5.7493, 5.8222] | 43.4203 [43.3035, 43.6579] |
| Julia / MKL FFT, temporal 18, transverse 18 | 4.8136 [4.7597, 4.8517] | 35.3260 [35.0637, 35.7159] |
| Julia / FFTW, temporal 4, transverse 18 | 6.3860 [6.3620, 6.4219] | 49.1409 [48.9671, 49.4415] |

Every measured window contains ten RK4IP steps and forty source evaluations.
Setup, compilation, first windows, reset, observations and export are excluded
and recorded separately. The maximum retained-field relative L2 difference
against native GCC is `8.69e-16`. The Julia FFT-provider comparison changes no
BLAS provider or numerical kernel: MKL FFT reduces time by about 16.7% and
18.6% on the two grids. Restricting temporal FFTW to four workers increases
time by about 10.5% and 13.2%, despite retaining eighteen transverse workers.
Thus the conservative default temporal cap needs an explicit override for
these large measured cases.

The optimized Julia/MKL FFT implementation still takes about 1.69 times native
Intel/MKL's time at 192² and 1.41 times at 384²; it does not meet native parity.
On the large grid it takes about 1.13 times the local GCC/OpenBLAS/FFTW arm.
This comparison confirms a substantial reduction from the initial Julia screen,
but only the paired rows above isolate provider and thread-cap effects.

#### Repeated core comparison with local native builds

The refreshed comparison uses the same exclusive dual Xeon Gold 6240 allocation
and one-socket affinity. Both native builds now target the local architecture:
GCC 16.2/OpenBLAS 0.3.34/FFTW 3.3.11, and IntelLLVM 2026.1.1/MKL 2026.1.
Runtime receipts verify the requested one- and four-worker budgets, providers,
release builds and disabled umbrella fast-math. Julia 1.13.2 switches only BLAS
between OpenBLAS and MKL in this comparison; its FFT provider remains FFTW.
The compiler/library bundles remain distinct, so this is not a compiler-only
experiment.

Three alternating rounds with two warm repetitions per worker completed all
216 case/track/budget/arm rows. Each cell below is the median of six warm
windows, followed by the observed minimum and maximum. The fixed DOPRI5
windows take ten accepted steps, no rejections and 61 source evaluations:

| Case | Budget | Native GCC / OpenBLAS, s | Native Intel / MKL, s | Julia / OpenBLAS, s | Julia / MKL BLAS, s |
|---|---:|---:|---:|---:|---:|
| RDW | 1 | 0.3335 [0.3314, 0.3355] | 0.3655 [0.3650, 0.3852] | 0.3818 [0.3753, 0.3879] | 0.3640 [0.3622, 0.3741] |
| CtC | 1 | 14.1144 [14.0463, 14.1532] | 12.3483 [12.2717, 12.8170] | 13.9152 [13.6739, 13.9980] | 13.6781 [13.6209, 13.7934] |
| Radial 256 | 1 | 2.1859 [2.1490, 2.2112] | 2.5976 [2.5077, 2.6950] | 2.8831 [2.8745, 2.9070] | 3.2336 [3.1962, 3.2463] |
| RDW | 4 | 0.1247 [0.1236, 0.1258] | 0.1399 [0.1377, 0.1446] | 0.2107 [0.2087, 0.2121] | 0.1997 [0.1982, 0.2073] |
| CtC | 4 | 5.3776 [5.3453, 5.4088] | 4.8425 [4.7826, 4.8598] | 6.2736 [6.2129, 6.3571] | 6.2622 [6.2139, 6.3337] |
| Radial 256 | 4 | 0.7093 [0.6914, 0.7123] | 0.6799 [0.6727, 0.6975] | 1.3272 [1.2898, 1.3443] | 1.3475 [1.3330, 1.3926] |

The maximum retained-field relative L2 difference against native GCC is
`6.24e-16` for these matched fixed schedules. Construction, compilation,
preparation, first windows and final observations are recorded separately.
These fixed timings exclude observation and export.

The standard adaptive windows include their declared observations and default
statistics. Their six-sample median seconds are:

| Case / track | Budget | Native GCC / OpenBLAS | Native Intel / MKL | Julia / OpenBLAS | Julia / MKL BLAS |
|---|---:|---:|---:|---:|---:|
| RDW / Adams | 1 | 0.2326 | 0.2559 | 0.2799 | 0.2725 |
| CtC / Adams | 1 | 5.0560 | 4.6254 | 5.3742 | 5.2996 |
| Radial / Adams | 1 | 17.7779 | 19.5602 | 19.5978 | 20.6227 |
| RDW / Adams | 4 | 0.1184 | 0.1287 | 0.2274 | 0.2179 |
| CtC / Adams | 4 | 1.9722 | 1.8237 | 2.3719 | 2.4128 |
| Radial / Adams | 4 | 5.3908 | 5.3955 | 7.9901 | 8.1920 |
| RDW / DOPRI5 | 1 | 0.3355 | 0.3691 | 0.3857 | 0.3729 |
| CtC / DOPRI5 | 1 | 14.0624 | 12.2967 | 13.8781 | 13.7196 |
| Radial / DOPRI5 | 1 | 19.9316 | 22.6670 | 25.4182 | 28.3597 |
| RDW / DOPRI5 | 4 | 0.1288 | 0.1415 | 0.2141 | 0.2029 |
| CtC / DOPRI5 | 4 | 5.3002 | 4.8144 | 6.0748 | 6.2093 |
| Radial / DOPRI5 | 4 | 6.3442 | 6.1744 | 11.3662 | 11.7782 |

Across all cells, the largest observed maximum-minus-minimum spread is 7.7%
of that cell's median. Work counts match across arms: RDW/CtC Adams use
21/19 accepted steps and source evaluations; their DOPRI5 windows use ten
accepted steps and sixty source evaluations. The frozen radial adaptive window
is longer: Adams uses 227 accepted and two rejected steps, with 230 sources;
DOPRI5 uses 87 accepted steps and 523 sources. The maximum adaptive field
differences are `6.41e-11` for Adams and `5.82e-13` for DOPRI5. These are
cross-backend comparisons, supplementing the earlier tighter-NumPy accuracy
checks; matching work counts does not require identical adaptive histories.

There is no universal native bundle winner. GCC/OpenBLAS is faster for RDW
and serial radial propagation; Intel/MKL is faster for CtC. Switching Julia's
BLAS alone reduces the fixed RDW time by about 5%, changes CtC little, and
increases serial radial time by about 12%. Thus the accumulated Julia gains
cannot be explained by choosing MKL BLAS. Julia remains slower than the faster
native arm on every core window in this refreshed comparison. At four workers,
the best Julia fixed-window times are about 1.60, 1.29 and 1.95 times the best
native times for RDW, CtC and radial respectively.

#### Line free-space extension

The scalar `LineGrid` path reuses the Cartesian FFT implementation with a unit
second spatial axis. Four focused field/envelope checks passed against NumPy,
including source evaluation, ten-step propagation, chunk tails and memory
accounting. The existing standard `line:fixed-rk` window was compared in three
alternating rounds with two warm repetitions. Median seconds are:

| Budget | Native GCC / OpenBLAS | Native Intel / MKL | Julia / FFTW |
|---|---:|---:|---:|
| 1 | 0.24017 | 0.21207 | 0.31315 |
| 4 | 0.07564 | 0.07618 | 0.25060 |

All windows take ten DOPRI steps; the largest retained-field relative L2
difference is `2.35e-16`. Julia works on this case but is slower, especially at
four threads. These runs use the existing GCC distribution and Intel local builds;
the native local GCC comparison and further Julia spatial-layout tuning are
separate experiments.

#### Cartesian plasma and native local-build screen

The next candidate extends the scalar field current to Cartesian and line
physical-space chunks. Twelve focused checks passed: source evaluation, ten
DOPRI/RK4IP steps, zero and ionising inputs, both temporal absorber schemes,
chunk tails, population/diagnostic observations and memory accounting. A separate
twenty-check run also passed with the proposed resident spatial layout. These
checks retain the existing NumPy oracle and rounding bounds; increasing the
ionising test input made its positive-population witness meaningful without
relaxing a numerical tolerance.

On the same dual Xeon Gold 6240, a new native **local** build uses GCC 16.2,
native architecture, IPO and OpenBLAS 0.3.34 built for SkylakeX with OpenMP;
its FFT provider is FFTW 3.3.11/OpenMP. The comparison Intel local build retains
MKL. Both preserve protected arithmetic and disable umbrella fast-math. This
removes the earlier distribution-versus-local build distinction, but still
compares compiler/library bundles rather than isolating the compiler alone.

Preliminary fixed ten-step DOPRI windows, 61 source evaluations, seconds:

| Case | Budget | Native GCC local / OpenBLAS | Native Intel / MKL | Julia / OpenBLAS / FFTW |
|---|---:|---:|---:|---:|
| Spatial absorber | 4 | 0.03805 | 0.03962 | 0.08760 |
| Radial filament, 512 nodes | 4 | 1.89999 | 1.78428 | 3.21756 |
| Cartesian filament, 128² | 18 | 13.14425 | 13.77150 | 32.73220 |

The first two rows use two warm repetitions in one round; the Cartesian row
uses one. Julia uses four temporal FFT and BLAS workers, the stated numerical
budget, and eighteen transverse FFT workers on the Cartesian row. Its spatial
layout is still strided, with parallel copies, parallel inverse-FFT scaling and
the exact real-root specialization enabled. Setup, compilation, first windows,
observations and export are recorded separately. The largest retained-field
relative L2 difference is `1.05e-15`. Julia works on these additional cases but
is slower in this screen. The modest differences between native bundles need
repeated A/B confirmation; none establishes a general compiler winner.

The frozen `spatial_absorber` builder currently passes `absorber=None`: its grid
declares a collar, but translation gives it zero physical-space loss. Its row
above describes that unchanged recipe and does not measure active spatial
absorption. Focused correctness checks enable the physical absorber explicitly.
Correcting the benchmark recipe would require a separately identified baseline;
these measurements do not silently change it.

An additional shortcut skips the remaining plasma-current work in a column
only when its peak ionisation rate and initial electron fraction are both
exactly zero. It clears the current outputs used by projection; no threshold
or protected integration arithmetic changes. Nineteen focused checks pass,
including strong-to-zero-to-weak-to-strong buffer reuse and nonzero initial
ionisation. The original weak-field check also included a nearly absent
temporal absorber and failed its purely relative source bound in both the
baseline and candidate, which produced identical outputs. The reuse check now
isolates plasma by disabling that absorber, without changing its tolerance;
separate checks retain active-absorber coverage.

At eighteen numerical workers, four temporal FFT/BLAS workers and eighteen
transverse FFT workers, three alternating rounds with two warm ten-step DOPRI
windows give these medians and observed ranges, in seconds:

| Case | Existing current path | Skip zero-rate columns |
|---|---:|---:|
| RDW | 0.17485 [0.17384, 0.17508] | 0.17392 [0.17311, 0.17443] |
| CtC | 6.43419 [6.35023, 6.50762] | 6.37213 [6.30023, 6.43160] |
| Radial 256 | 1.43938 [1.35070, 1.47310] | 1.43987 [1.36869, 1.46461] |
| Cartesian 128² | 28.83709 [28.73086, 29.01319] | 28.14896 [28.03151, 28.21059] |

All windows use 61 source evaluations and retained fields are exactly equal.
The Cartesian improvement is about 2.4%, with separated observed ranges;
the other differences overlap or are negligible. This is a modest saving,
pending integration into the combined source, not a general speedup claim.

#### Mode-averaged envelope extension and additional cases

The next extension uses complex temporal arrays for the mode-averaged path,
including the existing GNLSE Kerr, Raman and self-steepening coefficients.
There is no new frontend or physical model. Eight envelope checks cover direct
NumPy source comparisons, short DOPRI/Adams propagation, both absorbers, hard
masks, reset, memory accounting and finite default statistics. They pass with
the eight core checks on the same Xeon allocation.

Four additional frozen ten-step cases were screened against both native builds
above. One first window and two warm repetitions per case/arm, one round,
give the following median propagation seconds; these are preliminary, without
paired-order confirmation. Julia uses OpenBLAS/FFTW, its FFT cap is the CPU
budget here, and native retains automatic operation sizing:

| Case | Budget | Native GCC / OpenBLAS | Native Intel / MKL | Julia |
|---|---:|---:|---:|---:|
| GNLSE soliton | 1 | 0.01615 | 0.01851 | 0.01964 |
| GNLSE Raman/shock | 1 | 0.07365 | 0.07473 | 0.09325 |
| Argon blueshift, averaged | 1 | 0.01281 | 0.01378 | 0.01278 |
| Hydrogen Raman 6 ps, averaged | 1 | 0.08157 | 0.07081 | 0.08995 |
| GNLSE soliton | 4 | 0.01659 | 0.01896 | 0.02306 |
| GNLSE Raman/shock | 4 | 0.05259 | 0.05365 | 0.07184 |
| Argon blueshift, averaged | 4 | 0.01300 | 0.01346 | 0.02122 |
| Hydrogen Raman 6 ps, averaged | 4 | 0.07533 | 0.05234 | 0.09190 |

Every fixed window uses 61 sources and 10 accepted steps; the largest retained
field relative L2 difference against GCC native is `3.90e-16`. These results
establish execution and matched-schedule agreement. They do not show a Julia
speed advantage beyond a serial argon tie within the observed spread. The
small one-node cases also demonstrate that a wider global pool can hurt;
operation-specific thread choices require separate measurements.

#### Operation sizing and adaptive profiles

Separate modal, Raman and transverse FFT plan widths pass the eight focused core
checks with deliberately mixed limits. A one-round/two-warm OpenBLAS/FFTW sweep
of the three fixed core cases retained exactly equal fields across its settings.
At a four-thread numerical budget, modal/Raman/BLAS widths of 4/4/4 took
0.2160/6.6366/1.3593 s for RDW/CtC/radial. Reducing only the modal FFT cap to
one increased these to 0.2344/7.0332/1.8255 s. Reducing Raman to one as well
increased CtC to 10.3674 s. A blanket serial-FFT policy is therefore rejected
for these cases. The 18-thread sweep also exposed a poor radial combination:
18 BLAS threads with four modal FFT threads took 10.7590 s. This was a forced
experimental override, not the backend's four-thread default BLAS cap.

A follow-up 18-thread screen restored BLAS to four. Modal/Raman/BLAS 4/4/4
took 0.1746/6.4182/1.4849 s for RDW/CtC/radial; 4/18/4 took
0.1735/8.0751/1.4887 s, and 18/18/4 took 0.3155/8.2114/1.3749 s.
BLAS one with modal/Raman four took 0.1862/7.3053/2.0169 s. All fields were
exactly equal in this screen. More FFT threads therefore do not uniformly help,
and serial BLAS is particularly costly for radial contractions.

For the four one-node cases at a four-thread budget, one round of three warm
repetitions isolated the opposite behavior at small transform sizes. Changing
modal/Raman/BLAS 4/4/4 to 1/1/1 reduced the soliton from 0.02328 to 0.01955 s
and averaged argon from 0.02088 to 0.01185 s, but increased GNLSE Raman/shock
from 0.06912 to 0.07215 s and averaged hydrogen from 0.09057 to 0.10241 s.
Keeping Raman four with modal/BLAS one restored hydrogen to 0.08815 s.
All 12 retained-field differences were zero. These are sizing screens, not
portable thresholds or confirmed final speedups.

The small-plan policy was subsequently qualified with seventeen core/envelope
checks and three alternating rounds of three warm repetitions. With a four-thread
budget, keeping transforms serial below 16384 real-side samples and capping larger
plans at four gave these medians (seconds):

| Case | Fixed four-worker pools | Operation-sized pools |
|---|---:|---:|
| GNLSE soliton | 0.02301 | 0.01940 |
| GNLSE Raman/shock | 0.06742 | 0.06821 |
| Argon blueshift, averaged | 0.02081 | 0.01181 |
| Hydrogen Raman 6 ps, averaged | 0.09101 | 0.09168 |

The soliton and argon ranges do not overlap: 0.02267–0.02332 versus
0.01921–0.01980, and 0.02031–0.02108 versus 0.01147–0.01250, respectively.
Their time reductions are 15.7% and 43.2%. The Raman differences are small and
their ranges overlap. All retained fields are exactly equal. This policy is
retained as an experimental default; Raman sizes its doubled grid independently,
mode-averaged BLAS defaults to one, and explicit pool overrides remain effective.

A separate four-thread adaptive Adams screen, one round and two warm repetitions,
retained the standard observation window. GCC native, Intel native and Julia
respectively took 0.1188/0.1276/0.2808 s for RDW and
2.0152/1.8194/2.7391 s for CtC. Each RDW window took 21 accepted steps and
21 sources; CtC took 19 of each. No rejected steps occurred. These timings
remain preliminary and use the explicit provider/build bundles above.

Python-only profiles were then collected at four Julia threads with serial
modal/Raman FFT plans, so they attribute costs but are not directly comparable
speed estimates for the preceding FFT-four timings. Over two RDW windows,
scalar stage preparation took 0.118 s of 0.600 s, while observations took
0.036 s. The original Adams adapter unnecessarily evaluated all seven DOPRI
stage bundles despite reading only the initial and endpoint rows. Over two
CtC windows, observations took 0.652 s of 8.155 s, including 0.612 s in
population work. The original path reevaluated the full source in Julia,
then separately repeats the integration diagnostics on the host. These are
isolated optimization candidates; profiling itself does not establish their
speed or correctness.

The stage-bundle change is now retained after eight core checks and three
alternating rounds with two warm repetitions each. It evaluates only the
initial/endpoint bundles for Adams, preserving the seven-column seam buffers;
DOPRI also reuses bundles at identical offsets. The RDW adaptive median fell
from 0.27261 s (range 0.27193–0.34009) to 0.24327 s (0.23358–0.24649),
an approximately 11% reduction. The fixed-DOPRI median was essentially unchanged:
0.20895 s versus 0.21029 s. Both matched-field differences were zero, with
unchanged source/step counts.

The accepted-field population pass is also retained. It reuses the source's
rate/exposure helper, synthesizes once, and computes populations and six raw
diagnostic reductions without the nonlinear current, Kerr/Raman responses or
source FFTs. Four observation checks against NumPy and the eight core checks
pass, including zero/ionizing fields, raw diagnostics, source-count preservation
and continued propagation after an intermediate observation. Three alternating
rounds with two warm repetitions give RDW 0.24337 s before versus 0.22378 s
after (ranges 0.24146–0.24775 and 0.22096–0.22742), an 8.1% reduction.
CtC is 2.72477 versus 2.51867 s (2.71207–2.74879 and 2.49589–2.52177),
a 7.6% reduction. Both retained fields are exactly equal and work counts are
unchanged. These gains apply to the declared adaptive windows, which include
observations; the fixed-step propagation timer excludes this work.

#### Additional modal cases and final qualification

On the same dual Xeon Gold 6240, the additional mode-averaged helium RDW,
resolved argon blueshift and six-picosecond hydrogen Raman cases passed a source
evaluation and ten prescribed DOPRI steps against `numpy_ref`. Relative source
errors were respectively 3.25e-13, 3.95e-14 and 4.75e-16; field errors were
2.21e-14, 2.54e-15 and 3.24e-16, within the operation-count bounds fixed before
comparison. Julia used 61 sources with FSAL; the reference used 70.

The repeated comparison used three alternating rounds and two warm repetitions
per round, six samples per cell. All arms used ten accepted steps and 61 sources;
setup, compilation and final observation are outside these windows. The largest
Julia/native relative field difference was 4.22e-16. Times are median seconds:

| Case | Threads | Native GCC 16.2/OpenBLAS | Native Intel/MKL | Julia OpenBLAS/FFTW | Julia MKL BLAS/FFTW |
|---|---:|---:|---:|---:|---:|
| Averaged helium RDW | 1 | 0.025347 | 0.026393 | 0.050307 | 0.050101 |
| Averaged helium RDW | 4 | 0.025014 | 0.026030 | 0.049961 | 0.051350 |
| Resolved argon blueshift | 1 | 0.155783 | 0.182656 | 0.169006 | 0.169880 |
| Resolved argon blueshift | 4 | 0.066027 | 0.071829 | 0.071599 | 0.075018 |
| Hydrogen Raman, 6 ps | 1 | 1.817832 | 1.426127 | 2.325660 | 2.383013 |
| Hydrogen Raman, 6 ps | 4 | 0.615658 | 0.487549 | 0.877124 | 0.909605 |

The fastest native configuration wins these cases too. Julia is closest on the
resolved argon case, about 8 % slower at four threads. The averaged helium case
has little available parallelism and remains about twice the native time.

The final combined exact-zero plasma-column implementation passed 45 focused
checks in 134.86 seconds and was adopted. The preceding repeated comparison
measured a 2.4 % Cartesian plasma improvement (28.837092 to 28.148965 seconds),
with exactly unchanged fields. Other plasma cases showed negligible or overlapping
changes, as recorded above. This is a zero-rate specialization, not a cutoff.

A separate modal/radial absent-absorber candidate passed 29 focused checks across
the original run and a corrected fixture retry. Its three-round, six-sample
comparison gave the following median seconds, baseline to candidate:

| Case | One thread | Four threads |
|---|---:|---:|
| Radial filament, 256 nodes | 2.908762 → 2.795097 | 1.334994 → 1.295635 |
| Spatial-absorber benchmark | 0.131055 → 0.120990 | 0.088636 → 0.081302 |
| GNLSE soliton | 0.019518 → 0.017325 | 0.019295 → 0.017528 |
| Averaged argon blueshift | 0.011579 → 0.011791 | 0.011583 → 0.011528 |

Every field was unchanged. The GNLSE serial reduction was 11.2 %, radial 3.9 %
serial and 2.9 % at four threads. The named spatial-absorber benchmark currently
translates absent losses; active absorbers were checked separately. The candidate
was then integrated with the final plasma shortcut: all 57 combined focused checks
passed in 147.19 seconds, and the modal/radial guard was adopted.

An updated instrumented 384² profile confirms remaining work in spatial FFTs,
padding/gathering and inverse normalization. Across one warm measured window per
provider, spatial transforms took 14.22 seconds with FFTW and 7.56 with MKL;
padding took about 7 seconds with either. Total inverse normalization took 2.26
and 4.21 seconds, respectively. These counters overlap other phase totals and
must not be summed. Instrumentation changed total time, especially for MKL, so
the repeated uninstrumented comparison above remains the performance evidence.
The subsequent candidate moved spatial inverse normalization into the padding
gather, with an optional parallel zero-fill. Four focused check processes
(FFTW/MKL, with/without parallel zero-fill) each passed 23 checks; the largest
retained-field difference across the performance arms was 1.97e-18. However,
the first bounded screen regressed substantially. Median seconds from two warm
ten-step, forty-source windows, eighteen workers and one OpenBLAS worker:

| Provider / grid | Baseline | Fused scale | Fused scale + parallel zero-fill |
|---|---:|---:|---:|
| FFTW / 192² | 5.730811 | 10.783111 | 10.200047 |
| FFTW / 384² | 44.099682 | 91.827302 | 88.616203 |
| MKL FFT / 192² | 4.860231 | 9.919637 | 9.340103 |
| MKL FFT / 384² | 35.471356 | 83.691516 | 81.295904 |

This implementation was rejected and not adopted. The negative screen was not
extended into another repeated campaign; the authorized Xeon work stopped after
these jobs completed. It does not establish that every possible fusion is slower.

#### Mac default-build comparison after core tuning

The Apple M5 Max comparison uses 18 physical cores (six Super and twelve
Performance), 128 GiB RAM, Python 3.14.5 and Julia 1.13.2. Native uses the
unchanged default Mac source build: Release Homebrew Clang 23.1.3, OpenMP,
Accelerate BLAS, FFTW OpenMP, and the default Accelerate Raman path. Native
architecture selection, IPO and loop vectorization are the ordinary local-build
defaults; no experimental compiler or provider override was added.

The one- and eighteen-thread budgets retain operation-size policies. Julia uses
Accelerate's serial BLAS mode, its current Mac default, and automatically sizes
modal/Raman FFTs up to four workers at the larger budget. Native retains its
default operation policy and library-managed Accelerate parallelism; its BLAS
worker count is not exposed. At one thread all numerical pools are requested
serially, including `VECLIB_MAXIMUM_THREADS=1`. The Mac run includes the qualified
zero-plasma and absent-modal-absorber changes, excluding the experimental spatial
normalization fusion.

Three alternating rounds each take two warm repetitions, after construction and
one warm-up window. These are ten prescribed DOPRI steps, 61 source evaluations,
no rejected steps; final observation/export and compilation are outside the timer.
All fields are finite, and the maximum Julia/native relative L2 difference is
1.65e-15. Median seconds from six samples per cell:

| Case | Native, 1 | Julia, 1 | Native, 18 | Julia, 18 |
|---|---:|---:|---:|---:|
| RDW | 0.128269 | 0.157036 | 0.070999 | 0.068320 |
| CtC | 6.288408 | 5.325555 | 2.685244 | 2.107181 |
| Radial filament, 256 nodes | 0.649903 | 0.794537 | 0.288600 | 0.855188 |
| Averaged helium RDW | 0.010500 | 0.017543 | 0.010590 | 0.017662 |
| Resolved argon blueshift | 0.056584 | 0.050703 | 0.022913 | 0.019836 |
| Hydrogen Raman, 6 ps | 1.194044 | 1.051791 | 0.243912 | 0.437328 |
| GNLSE soliton | 0.005278 | 0.006534 | 0.005524 | 0.006570 |
| Averaged argon blueshift | 0.005019 | 0.005053 | 0.005156 | 0.005084 |

Unlike the Xeon comparison, Julia wins CtC and resolved argon at both budgets,
and hydrogen Raman serially. Native's default Accelerate Raman path wins the
parallel hydrogen case. Averaged argon is effectively tied. The radial parallel
timings are noisy: Julia ranges from 0.772156 to 0.964948 seconds and native from
0.281087 to 0.419821 seconds. Their ranges remain separated, and Julia's current
eighteen-thread policy does not improve on its serial radial result. These results
compare measured configurations, not a language-wide performance limit.

The Cartesian comparison uses the standard 192² × 300 and 384² × 600
envelope-beam cases, ten RK4IP steps and forty source evaluations per measured
window. Julia's temporal and transverse FFT widths equal the selected budget
for these large cases, with Accelerate BLAS still serial; native retains its
default Mac settings. The same three alternating rounds give six warm samples:

| Grid | Native, 1 | Julia, 1 | Native, 18 | Julia, 18 |
|---|---:|---:|---:|---:|
| 192² × 300 | 9.334232 | 15.488092 | 1.217616 | 1.764022 |
| 384² × 600 | 89.175180 | 77.226190 | 10.626641 | 11.544229 |

Times are seconds. The 192² native/Julia ranges are 9.191217–9.654625 /
15.403656–16.302226 at one thread and 1.158469–1.260257 /
1.752230–1.947926 at eighteen. For 384² they are 88.926129–91.650891 /
76.819168–78.333928 and 9.992235–10.729157 / 11.251710–11.842016.
Julia is 1.15 times faster for serial 384², but native wins both parallel
comparisons and serial 192². Julia's parallel 384² time is 8.6 % above native.
The maximum relative field difference is 8.69e-16, with all fields finite and
matching work counts. Numerical source hashes remained unchanged throughout all
120 modal/radial/Cartesian timing rows. The run used normal macOS core placement,
not fixed affinity; spreads include the observed variation over the campaign.

The 384² owned-buffer ledgers are 11.762 GB for Julia and about 7.567 GB for
native (decimal GB). Both fit on the 128 GiB Mac; these figures exclude runtime
heaps, opaque FFT storage and transient exports. Setup, first-window and
observation costs were recorded separately. No further optimization or timing
campaign was started after this requested comparison finished.

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
