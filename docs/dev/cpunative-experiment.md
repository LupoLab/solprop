# Independent CPU backend experiment

This page records the implementation and qualification of the experimental
`solprop-cpunative` package. The [user manual](../guide/performance/cpunative.md)
describes installation, explicit selection, and admission limits.
The [comparison summary](cpunative-comparison.md) collects completed final measurements;
the [native handoff](cpunative-native-handoff.md) identifies measured changes worth porting.

**Status: experiment complete.** Capillary, radial, and the scalar Cartesian
Kerr route run through the ordinary drivers. The retained optimisations have focused
numerical and sanitizer checks, and complete trajectories pass independently fixed
accuracy envelopes. The final comparison contains 186 successful full-run workers,
three interleaved pairs per setting, plus the matched scalar-libm control, short
width sweep and observation measurements. CTC and Cartesian show the largest gains;
serial RDW loses by 11–12%. The earlier radial
accuracy failure came from using a reference envelope for the wrong tolerance;
the corrected standard-tolerance comparison passes. Short-window and preliminary
single-pair timings below remain distinct from final repeated measurements.

## Boundary and execution

The package consumes kernel schema 37 without changing it. `CPUNativeBackend` inherits
`StepDriver`, owns scalar stage preparation and observations, and supplies a resident
exponential Adams engine to the existing `AdamsMarch`. It does not inherit from
`NativeBackend`, call another backend to propagate, or link Kokkos.

C++ owns aligned frequency-contiguous complex state, stage and history arrays, real
node chunks, transform plans, and padded worker reductions. Dispatch and admission are
outside element loops. DOPRI5 executes a complete trial, FSAL reuse, and its error norm
in C++; exponential Adams executes prediction, correction, strict defects, varying
operator compensation, and history acceptance in the same workspace. Rejection leaves
the accepted state and history intact. Host observations are owning copies.

The source pipeline pads modal spectra, performs inverse transforms, synthesises nodes,
evaluates the translated responses, projects polarisation and the two held-current
streams, and applies the forward transform, crop, prefactor, window, and mask. The
plasma filter uses point reflection at its boundaries. Its interior is blocked across
independent samples for SIMD; its running integral remains sequential within each
column. Diagnostic-only population observations omit the unrelated Raman, projection,
and forward-transform work.

Plans are created and destroyed under a package planner mutex. No workspace destructor
calls FFTW's process-wide cleanup. New-array execution uses disjoint buffers with the
same layout and alignment as the plan. The small awkward-size tests exercise the
unaligned planning case as well as ordinary aligned dimensions.

## Frozen workloads and methods

The source baseline is revision `cc6d8e2b0092757de0cf9357c40e475961da54f3`.
Translated dimensions, rather than registry descriptions, determine the experiment:

| Case | State shape | Coarse/fine temporal samples | Quadrature nodes | Initial chunk |
|---|---|---|---|---|
| RDW | `(1, 4, 1, 2561)` | 5,120 / 10,240 | 33 | 33 |
| CTC | `(1, 3, 1, 5121)` | 10,240 / 20,480 | 33 | 11 |
| Argon blue shift | `(1, 1, 1, 1601)` | 3,200 / 6,400 | 33 | 33 |
| Radial filament | `(1, 256, 1, 433)` | 864 / 1,728 | 256 | 128 |
| Cartesian Kerr | `(1, 147456, 1, 750)` | 750 / 750 | 147,456 | Volume transforms |

RDW includes Kerr, band plasma, and varying pressure/linear/prefactor expansions. CTC
includes Kerr, band plasma, molecular Raman, and constant fill. Both default exponential
Adams and DOPRI5 are measured. Standard seeds, tolerances, observations, and diagnostics
are retained. A short window uses a fraction of the original guide length; it must not
be presented as a full-run result.

The argon blue-shift case is an additional comparison within the same admitted
capillary subset: one resolved mode, Kerr and band plasma, and constant operators.
Its recipe and diagnostics are unchanged.

The expanded experiment includes the unchanged `radial_filament_256` and
`kerr_beam_384` recipes. The radial case includes analytic-signal Kerr and plasma;
the Cartesian case uses a complex envelope and four fixed RK4IP steps. They are
not equivalent to enlarging the capillary node count. Radial execution now passes
focused source, diagnostic, thread/chunk and prescribed-step comparisons and runs
through the ordinary driver. Cartesian execution also runs through that driver,
using FFTW volume transforms and resident RK4IP stages, with the split transform
schedule described below.

## Initial evidence and decisions

The first Apple-silicon sweep used three alternating pairs for each case, integrator,
and width (1, 4, 8, and 18 threads), at length fraction 0.0025. All 96 workers completed;
accepted/rejected work matched and final field differences were between roughly
`4e-17` and `1.4e-16` in relative L2 norm. Forty-six launch guards marked the machine
busy. The initial implementation was slower at one thread, with candidate/baseline
ratios between 1.15 and 2.30; CTC at 18 threads was faster in that short window. These
results preceded the retained loop and observation changes and are not the final result.

Phase attribution identified projection as about one third of the initial RDW source
cost at one thread. A standalone comparison covered 3/4 modes, 1/11/22/33 node rows,
10,240/20,480 samples, synthesis and projection, and included ordinary Eigen packing.
Accelerate was faster than the blocked hand implementation and Eigen on the tested
Apple-silicon shapes. In source evaluations, it also improved on the blocked provider.
Eigen therefore remains an experiment executable, with no extension dependency.

The first bound Linux sweep is **invalid as scaling evidence**. Importing the extension
started libgomp, which bound the main thread to an SMT core; the adapter mistook the two
visible hardware threads for the process grant and capped its kernels at two threads.
The package now records affinity before and after OpenMP loading, and the shared budget
recognises that recorded binding while respecting other deliberate restrictions.
Benchmark receipts record effective kernel widths. Subsequent sweeps below use the
corrected budget capture.

The corrected capillary implementation keeps the OpenMP team size fixed between
small FFT batches and large flat loops. Repeatedly resizing GNU OpenMP teams was
expensive: at 36 Xeon cores a matched developed RDW source call fell from about
6.2 ms to 2.1 ms; at 128 EPYC cores it fell from roughly 55 ms to 3.5 ms. These
are implementation ablations, not speedups against the native backend.

The subsequent controlled Xeon sweep used three interleaved pairs, length fraction
0.0025, matching libraries, and close/core binding with passive waiting. All 216
workers completed, all work counts matched, and no launch guard reported load.
At one thread, candidate/native time ratios were 1.03/1.15 for RDW Adams/DOPRI5,
0.87/0.86 for CTC, and 1.03/1.05 for argon. At 36 cores they were 0.80/0.85,
0.73/0.73, and 0.92/0.95 respectively. Intermediate widths sometimes favoured
native more strongly. This policy is a controlled baseline, not a tuned optimum.
The 288-worker EPYC sweep also completed with matching work and no busy launch
guards, but some round ratios varied substantially. The later runtime/provider
study and complete-case repeats supersede these preliminary rankings.

The subsequent consolidated Xeon comparison gives native the same proposed GCC
math flags as the candidate. Both use GCC 16.2, the Cascade Lake target, LTO,
OpenMP FFTW and OpenMP OpenBLAS, with close/core binding and passive waiting.
The candidate uses SLEEF for rate interpolation and the retained radial spectral
contractions. Three interleaved pairs per setting produced 144 successful, quiet
workers. The measured window remains length fraction 0.0025. These are medians
in seconds from the ordinary driver, including its observations and metadata work:

| Case / method | Native, 1 thread | cpunative, 1 thread | Native, 36 cores | cpunative, 36 cores |
|---|---:|---:|---:|---:|
| RDW / Adams | 0.584 | 0.566 | 0.540 | 0.414 |
| RDW / DOPRI5 | 0.836 | 0.850 | 0.606 | 0.517 |
| CTC / Adams | 1.948 | 1.433 | 0.883 | 0.627 |
| CTC / DOPRI5 | 3.422 | 2.448 | 1.195 | 0.827 |
| Argon / Adams | 0.344 | 0.305 | 0.238 | 0.209 |
| Argon / DOPRI5 | 0.439 | 0.372 | 0.244 | 0.219 |
| Radial / Adams | 1.545 | 0.901 | 0.455 | 0.447 |
| Radial / DOPRI5 | 1.952 | 1.201 | 0.612 | 0.558 |

At eight cores, candidate/native ratios were 1.007/1.054 for RDW Adams/DOPRI5,
0.797/0.795 for CTC, 0.825/0.857 for argon, and 0.903/0.902 for radial propagation.
The radial serial gain therefore does not imply a similar gain at high widths,
and RDW remains roughly matched at one thread. A separate preliminary pair with
native alone found the extra math flags usually changed short-run time by a few
percent; it is not sufficient evidence to choose a production native flag policy.

## Runtime, compiler and OpenMP experiments

The earlier tuning record remains relevant: binding and active/passive waiting
can dominate arithmetic improvements, particularly when FFT and BLAS worker pools
interact. Policy experiments cover close, spread and unbound placement with passive,
runtime-default and active waiting. The selected practical comparison uses OpenMP
FFTW and OpenMP BLAS with close/core placement and passive waiting. Neither a
quiet launch guard nor an exclusive allocation guarantees identical timing.

Compiler comparisons separate compiler/runtime changes from FFT/BLAS provider
changes. A common serial BLAS and pthread FFTW library avoids introducing a
second OpenMP runtime when comparing GCC, Clang and Intel. Their practical tuned
configurations, including the requested Intel MKL experiments, remain a separate
comparison. With common SLEEF, serial BLAS, pthread FFTW, math flags and placement,
the controlled driver trial covered RDW, CTC and argon with both integrators at one
and 36 threads. All 48 workers met their requested widths. Clang/GCC time ratios
ranged approximately from 0.919 to 1.097 and Intel/GCC from 0.932 to 1.074: mixed
results, with no broad compiler win. These are one pair per configuration and do not
establish a compiler ranking. Eager OpenMP initialisation before affinity capture
was necessary for the LLVM runtimes; earlier incorrectly capped parallel samples
are excluded. The prior GCC 14 compatibility constraint is suspended for these
experiments; distribution qualification is not claimed.

Four math profiles retain the protected helpers while successively adding
`-fno-math-errno`, `-fno-trapping-math`, and `-fcx-fortran-rules` to the ordinary
`-O3`, CPU target, contraction and checked-LTO settings. The last flag retains
scaled division but changes complex exceptional-value recovery; it is not
equivalent to blanket fast-math. See the
[GCC option definitions](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html).
The full proposed set has passed the capillary numerical tests with GCC 16 on
Apple silicon. A separate executable checks protected cancellation, subnormal
survival and finiteness checks under each actual compiler/flag/LTO combination.
This establishes initial correctness, not a performance gain.

Host feature probes compile and execute exact integer-valued checks before a
feature is used in a numerical experiment. On GCC 16.2 and Clang 23.1.3,
`loop`, inclusive `scan`, SIMD scan, `tile`, `unroll`, strict taskloop grain sizes,
metadirectives, scope reductions and aligned allocators pass. GCC also accepts
`assume no_openmp_constructs`; Clang needs explicit OpenMP 6 mode for that clause
and the newer stripe transformation. In that mode the tested reverse, stripe
and interchange constructs pass on Clang; GCC rejects those three. These results
agree with the distinction between the
[GCC implementation table](https://gcc.gnu.org/onlinedocs/libgomp/OpenMP-6_002e0.html)
and [Clang's feature table](https://clang.llvm.org/docs/OpenMPSupport.html).
The numerical/performance experiment compares the current sequential-column scan,
explicit block offsets, SIMD scan within a column, and parallel scans across a
column. Rounding is checked against accumulated absolute work, including
oscillating inputs. Availability alone is not a reason to retain a feature.

The integrated SIMD-scan trial produced small mixed changes in full source time;
the parallel scan across individual columns was substantially slower. Neither is
selected. On the radial shapes, OpenMP BLAS beat the hand contraction and serial
BLAS controls, with source time already saturating around eight cores.

### Isolated rate-math trial

Rate interpolation now has a separate translation unit. An opt-in experiment
compares scalar libm with Intel SVML, SLEEF `u10`, and MKL VML HA for its logarithms
and exponentials only. Fraction, scan, and defect arithmetic retain their existing
math settings. The vector providers use approximately one-ULP accuracy contracts;
the four-ULP libmvec and AOCL vector contracts are outside this trial. See
[SLEEF's vector interfaces](https://sleef.org/x86.xhtml) and
[Intel's VML accuracy modes](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-vector-math-performance-accuracy-data/2021-1/overview.html).

Blocks of 256 samples retain scalar early exits when mostly inactive. Unconditional
vector evaluation nearly doubled RDW's serial source time, despite helping CTC.
The compiler report confirms vector `log`/`exp` calls in the isolated SVML variant.
All four providers pass the six initial source comparisons.

An initial matched-source trial on a Xeon Gold 6240 used Intel LLVM 2026.1.1,
common FFTW and serial BLAS, close/core binding and passive waiting. Each value is
the mean of 50 warmed source calls on the same stored developed CTC field:

| Rate provider | One thread, ms | 36 threads, ms |
|---|---:|---:|
| Scalar libm | 30.69 | 6.35 |
| SVML | 26.96 | 5.99 |
| SLEEF AVX2 | 25.65 | 5.87 |
| MKL HA | 24.90 | 6.46 |

These are exploratory source timings, not propagation speedups or a final provider
selection. MKL reduced rate time most, but increased surrounding Raman time in this
sample. In the follow-up, SLEEF AVX2 reduced developed CTC source time from 34.99 to
26.79 ms with GCC 16 and from 31.46 to 27.52 ms with Clang 23. Intel with SLEEF
AVX-512 reached 23.57 ms at one thread, but its wider vectors did not improve the
36-thread result. In the ordinary-driver comparison with GCC and matching libraries,
SLEEF reduced short CTC Adams/DOPRI5 time by 15%/19% at one thread and 5%/7% at
36 threads. Argon improved by 12%/13% at one thread; RDW changed by 0–3%.
Those are one interleaved pair per configuration with identical work, relative to
the scalar-libm cpunative control. All six complete capillary trajectories with
SLEEF pass the previously fixed convergence envelopes. SLEEF was selected for the
final performance comparison; the package default remains libm.

### Spatial execution and copies

The unchanged Cartesian case translates to 750 temporal samples and four fixed
RK4IP steps. Caching its temporal absorber factors reduced redundant exponential
work. Profiling then identified owning field copies as a parallel scaling cost.
The binding now allocates uninitialised storage for trivially copyable values and
copies large observations in contiguous worker partitions, retaining ownership.
Source, four-step, statistics, and retained-observation checks pass.

The updated four-step Xeon comparison at 36 physical cores took 11.41 s versus
native's 17.07 s with passive waiting, 11.30 versus 17.08 s with runtime-default
short spinning, and 11.55 versus 17.35 s with active waiting. Placement was close
with core places throughout. These are one preliminary pair per policy with
matching OpenMP FFTW/BLAS dependencies, not final repeated qualification.

For square real radial maps, the contractions commute with temporal transforms.
The retained optimisation synthesises before spectral padding and projects after
cropping, so both matrix products operate on the retained frequencies. The FFT count,
response, prefactors, and separate plasma streams remain the same. Existing buffers
hold the packed spectra, with no additional allocation. The ordinary time-domain
pipeline remains appropriate for skinny capillary maps, where moving the contraction
would multiply the FFT count.

On the developed radial field, this reduced source time from 31.51 to 20.51 ms at
one thread, 8.14 to 4.71 ms at eight cores, and 9.02 to 5.27 ms at 36 cores.
Three interleaved short-driver pairs per configuration gave candidate/control time
ratios of 0.858/0.750 for Adams/DOPRI5 at one thread, 0.900/0.815 at eight cores,
and 0.884/0.804 at 36 cores. All 36 workers were quiet. These compare two cpunative
implementations, not cpunative against native. Source/statistics, tail chunks,
prescribed steps and ordinary-driver checks pass.

The short radial profile also has a substantial shared-driver floor: at 36 cores,
opening and serialising run metadata took about 0.29 s, while propagation took
about 0.15 s in the earlier implementation. That overhead remains in the ordinary
driver comparisons. Separate chunk-size trials gave small, width-dependent changes;
the translated chunk limit remains unchanged.

## Generic Raman and large-volume FFT experiments

The Raman prototype expands the squared real node field into unordered modal
products, convolves each product once, then synthesises the convolved result at
the nodes. Cross terms carry a factor of two. This is the same linear causal
convolution, with a different contraction order. It reuses the existing Raman
scratch when `n_modes * (n_modes + 1) / 2` is smaller than the node count and fits
the allocated row count; otherwise it retains node convolution. No dispatch
depends on a case name or a fixed three-mode count.

On the Xeon Gold 6240, a one-through-eight-mode sweep used both launch fields and
matched developed inputs formed by embedding or truncating the stored CTC field.
Those embeddings are source inputs, not complete trajectories for the other mode
counts. At one thread, source time decreased by approximately 35%, 29%, 22%, and
13% for one, two, three, and four modes. Five through eight modes used the fallback
and were broadly unchanged. Three interleaved ordinary CTC driver pairs then gave
median candidate/control time ratios of 0.882/0.834 for Adams/DOPRI5 at one thread,
0.935/0.888 at eight cores, and 0.981/0.998 at 36 cores. All 36 workers were quiet
and work counts matched. The extra gain is substantial at one thread, smaller at
eight, and negligible or small at 36. Both complete CTC trajectories pass the
unchanged independently frozen envelopes for the field and all requested statistics.
Fourteen isolated Raman checks populate every mode appreciably and cover
one through eight modes, small/tail chunks, and both paths for the larger mode counts.

The available 96-core EPYC 7643 system exposed an FFTW planning problem in the
Cartesian direct-volume path. With passive waiting and close/core binding,
the standard transform pair took approximately 0.31 s at 32 threads but 27 s at
96. Plan dumps show deeply nested threaded subplans at the larger width. A bounded
MEASURE attempt selected the same slow structure. A standalone schedule using
serial two-dimensional plane plans and batched transforms along the remaining
axis, distributed by OpenMP, took approximately 0.23 s at 32 cores and 0.095 s at
96. Its round-trip error was about `1e-15`. The plans use FFTW's documented
[new-array execution](https://www.fftw.org/doc/New_002darray-Execute-Functions.html)
on disjoint matching layouts. The integrated schedule adds no state-sized buffer;
six small source/four-step/explicit-statistics checks, including tail batches,
pass. The complete standard four-step case then gave the following first quiet
interleaved pairs with matched GCC 16 flags and libraries:

| Physical cores | native, s | cpunative split FFT, s | native / cpunative |
|---:|---:|---:|---:|
| 1 | 129.58 | 55.85 | 2.32 |
| 8 | 33.24 | 20.99 | 1.58 |
| 32 | 12.81 | 9.25 | 1.38 |
| 96 | 8.05 | 5.47 | 1.47 |

All arms took four steps and 16 source evaluations. A separate 32-core ablation
against cpunative's direct FFT path reduced propagation time by 14.3%.
Planning took approximately 1.25--1.36 s for the split candidate, separately from
these warmed times. Candidate peak RSS was 17.9--18.0 GB versus native's 16.7--16.8 GB;
the smaller persistent numerical workspace does not imply a smaller process peak.
The split schedule is retained when the grid has at least as many planes as workers;
fewer planes retain internally threaded volume plans. The complete standard field
agrees with the independent Numba backend to a relative norm error of `1.16e-15`,
below the pre-existing dimension-based bound of `4.31e-11`. Small checks against
NumPy cover both schedules and all six supported spatial statistics; the standard
case itself requests no statistics. The final three-pair comparison confirms
2.30×, 1.37× and 1.49× median gains at one, 32 and 96 cores respectively;
the comparison summary reports the timing spread and higher candidate peak RSS.

Three interleaved short-driver pairs per contraction configuration compared the
handwritten products with OpenMP OpenBLAS (108 quiet workers). For the one-mode
argon case, median candidate/control ratios were 0.960/0.921 for Adams/DOPRI5 at
eight cores and 0.922/0.813 at 36 cores. Serial Adams was effectively unchanged;
the serial multimode cases generally preferred BLAS. Together with the matched
developed-source measurements, this supports a narrow default: handwritten
contractions for one mode at eight or more threads on Linux x86-64 with the ordinary
BLAS provider, vendor products otherwise. Explicit MKL builds retain MKL. The
decision uses dimensions and architecture, with no benchmark-name dispatch.

A separate controlled Intel compiler test, using unbound workers for both arms,
reduced the full four-step Cartesian run from 17.41 s with pthread FFTW to 13.75 s
with MKL. This is one pair and remains slower than the selected bound GCC/OpenMP
FFTW result of 11.26 s at 36 cores; the latter is a practical comparison with
different compiler and placement choices. An unrelated statistics-fusion
prototype was removed because the standard Cartesian case requests no statistics
and supplied no timing evidence for retaining it. Explicit statistics tests now
request all six admitted spatial statistics rather than relying on defaults.

## Numerical checks

Matched computations use rounding bounds based on transform depth, contraction length,
scan length, and the absolute magnitude of cancellation-prone integrals. In particular,
a held-current edge stream can be tiny after an oscillatory current cancels before the
taper: its absolute rounding budget is based on the integral's absolute work and filter
gain, not its tiny surviving tail. Adaptive trajectory accuracy requires a separate
converged-reference envelope and is not inferred from matching step counts.

Focused tests cover actual RDW/CTC source evaluations, eight prescribed steps with both
integrators, standard statistics, direct FFT scaling/padding/endpoint bins, and
independent array calculations of Kerr, Raman, rate interpolation, reflected band
integrals, exposure fractions, and held-current assembly. Other tests cover tail chunks,
thread counts, both contraction providers, both absorbers, reset, owning field retention,
FSAL counts, discarded trials, Adams snapshots, memory accounting, malformed inputs,
early admission refusals, ordinary memory/HDF5 output, and import-order/lifecycle checks.

The frozen capillary kernels pass complete RDW, CTC and argon trajectories with
both integrators against convergence envelopes fixed from three independent
reference tolerances. Prescribed eight-step windows from identical launch and
developed fields also agree with NumPy. DOPRI5 source counts differ from the
reference because the compiled engines reuse FSAL; the work actually performed
is recorded exactly.

The radial addition passes the expanded focused suite and the sanitizer subset
(79 tests with AddressSanitizer/UndefinedBehaviorSanitizer, excluding subprocess
import-order tests from the instrumented process). The first full radial comparison
incorrectly used an envelope whose coarsest reference had `rtol=1e-6`, while the
unchanged standard recipe uses `rtol=1e-5`. At the actual standard tolerance,
NumPy, Numba and cpunative all took 129 accepted and two rejected steps and exceeded
that old envelope by essentially identical amounts, including the 2.58 nm beam-radius
error. The old result is retained as an invalid tolerance comparison. A new envelope
uses the same pre-existing four-times-reference-discrepancy policy, with independent
Numba references at the standard tolerance, `1e-8`, and `1e-10`; the optimised
candidate passes it. Candidate differences did not set its budgets. Radial DOPRI5
also passes its independently frozen standard/`1e-8`/`1e-10` envelope, including
the field and all requested statistics; the candidate and standard reference each
take 55 accepted steps without rejection. Cartesian source, four-step, and explicit
statistics checks pass at one and four threads, and the complete field comparison
is described above. The final consolidated source additionally passes 35 capillary
parity checks on Xeon and 44 capillary/Cartesian checks on EPYC. The new Raman and
Cartesian paths pass 23 ASan/UBSan checks. Final strategy selection and repeated
timing are complete; the comparison summary separates full runs, short controls,
observation costs and remaining scaling limits. Full repository gates, coverage
lanes and distribution-wheel qualification were outside the agreed experiment.
