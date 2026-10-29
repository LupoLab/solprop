# cpunative comparison results

**Status: complete repeated comparison.**

The independent backend provides substantial gains on some workloads and loses on
others. These results measure the unchanged standard recipes through the ordinary
driver. They do not establish a general Kokkos abstraction penalty. See the
[native optimisation handoff](cpunative-native-handoff.md) for changes worth porting
and the [experiment record](cpunative-experiment.md) for the ablations and validation.

## Conclusions

The strongest complete-run gains are CTC and Cartesian Kerr. CTC is 1.80–1.93×
faster at one core and 1.72–2.07× at matched parallel widths. Cartesian is 2.30×
faster at one core and 1.49× at 96 cores. Argon wins at every sampled width;
its best-width gains are 1.82× for Adams and 1.58× for DOPRI5. Radial has more
modest gains: 1.18–1.25× at one core and 1.39×/1.22× at each backend's best
sampled width for Adams/DOPRI5.

**RDW is the clear loss.** cpunative takes 10.7% longer for serial Adams and
11.6% longer for serial DOPRI5. Parallel Adams wins, with 11.527 → 7.416 s at
eight cores, but DOPRI5 at the best sampled width still takes 3.7% longer.
These full trajectories supersede the earlier short-window rankings.

The experiment supports porting work reduction, data movement and scheduling
improvements to native. It does not show that Kokkos itself imposes these costs.
The strongest isolated ablations were retained-spectrum radial contractions,
Raman modal-product convolution, selective vector rate math, and large field-copy
and FFT scheduling changes. Their effects overlap. The
[handoff](cpunative-native-handoff.md) gives the measured ablations and source
entry points, including ideas already partly present in native.

Strong scaling remains limited. Capillary usually favours eight cores on the
tested Xeon; radial favours 16–32 on the tested EPYC. CTC Adams at eight and
36 cores is indistinguishable in these samples (124.110 versus 124.137 s).
Cartesian benefits from the full 96-core node, although its faster serial path
means lower relative scaling efficiency than native. The additional short width
sweep does not justify choosing full-node widths for small capillary workloads.

## Measurement scope

- Three interleaved fresh-process samples per backend and configuration, on exclusive
  physical-core allocations. Times are medians; speedup is native time / cpunative
  time. Values below one favour native. The paired range is the minimum and maximum
  of the three individual pair ratios, not a confidence interval.
- Capillary full runs use a dual Xeon Gold 6240 system, 36 physical cores. Spatial
  full runs use a dual EPYC 7643 system, 96 physical cores. Both use close/core
  placement and passive waiting. No Mac timing enters these conclusions.
- Both arms use GCC 16.2, matched CPU targets, LTO, FFTW 3.3.11 with OpenMP,
  OpenMP OpenBLAS, and the proposed errno-free/nontrapping/Fortran-complex flags.
  Protected arithmetic remains strict. The selected cpunative build additionally
  uses SLEEF u10 for rate interpolation. The scalar-libm control below separates
  that library change from the implementation changes.
- Times include propagation, ordinary observations, statistics and run metadata,
  after an untimed warm-up. Translation, backend construction and planning are
  reported separately. These are not cold-start timings. The package defaults
  remain the protected math profile and scalar libm.
- Native is frozen at source revision `cc6d8e2b0092757de0cf9357c40e475961da54f3`.
  Receipts record extension hashes, loaded libraries, effective thread counts,
  translated shapes, work counts and launch-load checks. The Cartesian candidate
  uses the same retained split FFT implementation as the final consolidated build.

All 186 successful full-run workers have finite outputs, requested effective
widths and matching accepted/rejected/source counts between backends. The CTC
allocation reached its wall-time limit after 32 successful workers. A continuation
preserved those receipts and completed the two missing pairs; the interrupted
attempt is excluded. No successful sample was discarded for timing variability.

Actual translated dimensions are recorded below. All cases have one batch member
and one polarisation. Cartesian has 384 × 384 transverse samples; its modes axis
is flattened at the kernel boundary.

| Case | Kernel state shape | Coarse / fine temporal samples | Quadrature nodes |
|---|---|---:|---:|
| RDW | `(1, 4, 1, 2561)` | 5120 / 10240 | 33 |
| CTC | `(1, 3, 1, 5121)` | 10240 / 20480 | 33 |
| Argon | `(1, 1, 1, 1601)` | 3200 / 6400 | 33 |
| Radial | `(1, 256, 1, 433)` | 864 / 1728 | 256 |
| Cartesian | `(1, 147456, 1, 750)` | 750 / 750 | 147456 |

## Full runs

### RDW

36 workers; 36 quiet launch checks. Complete standard length.

| Method | Cores | Native, s | cpunative, s | Speedup | Paired range |
|---|---:|---:|---:|---:|---:|
| Adams | 1 | 25.120 | 27.798 | 0.90× | 0.90–0.92× |
| Adams | 8 | 11.527 | 7.416 | 1.55× | 1.55–1.56× |
| Adams | 36 | 16.269 | 8.613 | 1.89× | 1.87–1.89× |
| DOPRI5 | 1 | 65.999 | 73.624 | 0.90× | 0.89–0.90× |
| DOPRI5 | 8 | 18.035 | 18.701 | 0.96× | 0.95–0.96× |
| DOPRI5 | 36 | 30.409 | 25.227 | 1.21× | 1.17–1.21× |

### CTC

36 workers; 36 quiet launch checks. Complete standard length.

| Method | Cores | Native, s | cpunative, s | Speedup | Paired range |
|---|---:|---:|---:|---:|---:|
| Adams | 1 | 743.220 | 412.951 | 1.80× | 1.78–1.81× |
| Adams | 8 | 225.454 | 124.110 | 1.82× | 1.81–1.82× |
| Adams | 36 | 256.905 | 124.137 | 2.07× | 2.06–2.08× |
| DOPRI5 | 1 | 980.396 | 506.678 | 1.93× | 1.91–1.94× |
| DOPRI5 | 8 | 251.430 | 145.811 | 1.72× | 1.70–1.78× |
| DOPRI5 | 36 | 266.956 | 135.383 | 1.97× | 1.87–1.99× |

### Argon blue shift

36 workers; 36 quiet launch checks. Complete standard length.

| Method | Cores | Native, s | cpunative, s | Speedup | Paired range |
|---|---:|---:|---:|---:|---:|
| Adams | 1 | 56.185 | 47.753 | 1.18× | 1.17–1.19× |
| Adams | 8 | 26.334 | 14.499 | 1.82× | 1.80–1.82× |
| Adams | 36 | 34.785 | 22.840 | 1.52× | 1.51–1.57× |
| DOPRI5 | 1 | 39.847 | 32.067 | 1.24× | 1.24–1.26× |
| DOPRI5 | 8 | 12.081 | 7.649 | 1.58× | 1.57–1.58× |
| DOPRI5 | 36 | 17.645 | 10.645 | 1.66× | 1.65–1.75× |

### Radial filament 256

60 workers; 60 quiet launch checks. Complete standard length.

| Method | Cores | Native, s | cpunative, s | Speedup | Paired range |
|---|---:|---:|---:|---:|---:|
| Adams | 1 | 8.917 | 7.538 | 1.18× | 1.18–1.21× |
| Adams | 8 | 2.166 | 1.755 | 1.23× | 0.62–1.23× |
| Adams | 16 | 1.606 | 1.312 | 1.22× | 1.18–1.23× |
| Adams | 32 | 1.639 | 1.156 | 1.42× | 1.34–1.46× |
| Adams | 96 | 2.986 | 2.236 | 1.34× | 1.24–2.25× |
| DOPRI5 | 1 | 11.467 | 9.154 | 1.25× | 1.24–1.26× |
| DOPRI5 | 8 | 2.872 | 2.128 | 1.35× | 1.35–1.39× |
| DOPRI5 | 16 | 2.286 | 1.880 | 1.22× | 1.20–1.22× |
| DOPRI5 | 32 | 2.387 | 1.973 | 1.21× | 1.18–1.36× |
| DOPRI5 | 96 | 9.161 | 7.122 | 1.29× | 0.58–1.38× |

### Cartesian Kerr beam 384

18 workers; 18 quiet launch checks. Complete standard length.

| Method | Cores | Native, s | cpunative, s | Speedup | Paired range |
|---|---:|---:|---:|---:|---:|
| RK4IP | 1 | 129.981 | 56.394 | 2.30× | 2.30–2.33× |
| RK4IP | 32 | 12.794 | 9.322 | 1.37× | 1.35–1.55× |
| RK4IP | 96 | 8.110 | 5.434 | 1.49× | 1.49–1.51× |

### Timing spread

Three pairs identify large effects but do not provide tight confidence intervals.
Quiet launch checks do not guarantee steady execution time. In particular,
radial Adams at eight cores has one cpunative sample at 3.552 s versus two near
1.755 s; its paired ratio crosses one. At 96 cores radial DOPRI5 also crosses
one, and native Adams ranges from 2.902 to 4.864 s. These full-node radial
ratios are noisy and are not evidence of a reliably sized win. All samples remain
in the medians and paired ranges above. The 16-core confirmation and 32-core
samples consistently favour cpunative, although native's 16-versus-32-core
minimum should not be treated as a precisely resolved optimum.

CTC DOPRI5 at 36 cores has a 6.1% cpunative range relative to its median;
Cartesian at 32 cores has an 11.6% native range. Both still favour cpunative in
every pair. Serial RDW loses in every pair for both integrators.

## Best measured widths

This compares the fastest sampled width for each backend; it does not assume the
full node is optimal. The full-run width sets are 1/8/36 for capillary,
1/8/16/32/96 for radial, and 1/32/96 for Cartesian. Unmeasured widths might improve
either backend.

| Case / method | Native seconds (cores) | cpunative seconds (cores) | Speedup |
|---|---:|---:|---:|
| RDW / Adams | 11.527 (8) | 7.416 (8) | 1.55× |
| RDW / DOPRI5 | 18.035 (8) | 18.701 (8) | 0.96× |
| CTC / Adams | 225.454 (8) | 124.110 (8) | 1.82× |
| CTC / DOPRI5 | 251.430 (8) | 135.383 (36) | 1.86× |
| Argon blue shift / Adams | 26.334 (8) | 14.499 (8) | 1.82× |
| Argon blue shift / DOPRI5 | 12.081 (8) | 7.649 (8) | 1.58× |
| Radial filament 256 / Adams | 1.606 (16) | 1.156 (32) | 1.39× |
| Radial filament 256 / DOPRI5 | 2.286 (16) | 1.880 (16) | 1.22× |
| Cartesian Kerr beam 384 / RK4IP | 8.110 (96) | 5.434 (96) | 1.49× |

## Work accounting

Each entry is accepted steps / rejected trials / source evaluations. Counts are
taken from the work actually performed; they are not inferred from the method.

| Case / method | Native | cpunative |
|---|---:|---:|
| RDW / Adams | 1817/4/1822 | 1817/4/1822 |
| RDW / DOPRI5 | 1760/0/10561 | 1760/0/10561 |
| CTC / Adams | 15062/2/15065 | 15062/2/15065 |
| CTC / DOPRI5 | 3921/0/23527 | 3921/0/23527 |
| Argon blue shift / Adams | 11082/2/11085 | 11082/2/11085 |
| Argon blue shift / DOPRI5 | 1631/0/9787 | 1631/0/9787 |
| Radial filament 256 / Adams | 129/2/132 | 129/2/132 |
| Radial filament 256 / DOPRI5 | 55/0/385 | 55/0/385 |
| Cartesian Kerr beam 384 / RK4IP | 4/0/16 | 4/0/16 |

## Construction and memory

The following values are median component costs and peak RSS at full sampled width.
Translation is shared Python work and is excluded from both warmed run times.
Backend construction includes its FFT planning. Peak RSS is the process peak,
not the persistent workspace ledger. GiB means 2^30 bytes.

| Case | Translation native / CP, s | Backend native / CP, s | FFT planning native / CP, s | Peak RSS native / CP, GiB |
|---|---:|---:|---:|---:|
| RDW | 3.717 / 3.701 | 1.369 / 1.794 | 1.331 / 1.769 | 0.272 / 0.250 |
| CTC | 3.504 / 3.487 | 11.481 / 5.578 | 11.451 / 5.562 | 0.271 / 0.265 |
| Argon | 2.949 / 2.961 | 1.308 / 1.685 | 1.291 / 1.675 | 0.228 / 0.227 |
| Radial | 2.816 / 2.799 | 2.463 / 1.970 | 2.422 / 1.933 | 0.717 / 0.604 |
| Cartesian | 10.755 / 10.645 | 1.350 / 1.626 | 1.120 / 1.261 | 15.620 / 16.722 |

Cartesian cpunative has a smaller persistent workspace (about 8.25 versus 8.80 GiB)
but a higher peak RSS (about 16.7 versus 15.6 GiB). Its planning also takes longer.
Adding construction to a one-off run reduces the practical gain, particularly
at high thread counts. The warmed speedups must not be presented as cold-start gains.

## Observation costs on matched stored fields

These measurements use the same 12 stored planes for both backends on the Xeon,
including launch and developed fields. Each operation has two warm calls and
seven measured calls per plane. Each table entry sums the 12 per-plane medians,
so it estimates one field export or one statistics request at every stored plane,
in milliseconds. It is not an instrumented share of the full-run times above.
The 18 subprocesses ran serially in an exclusive allocation, native before
cpunative at each setting; unlike the propagation timings, they are not three
fresh interleaved pairs. Treat small differences as descriptive microtimings.

| Case | Cores | Field export native / CP, ms | Statistics native / CP, ms |
|---|---:|---:|---:|
| RDW | 1 | 3.97 / 4.76 | 109.07 / 66.18 |
| RDW | 8 | 4.20 / 3.26 | 44.74 / 22.20 |
| RDW | 36 | 3.42 / 4.02 | 45.14 / 22.84 |
| CTC | 1 | 2.48 / 2.53 | 312.16 / 177.86 |
| CTC | 8 | 2.67 / 2.49 | 80.55 / 49.30 |
| CTC | 36 | 3.09 / 2.72 | 80.04 / 44.67 |
| Argon | 1 | 1.36 / 1.31 | 72.10 / 39.39 |
| Argon | 8 | 1.50 / 1.46 | 18.77 / 12.10 |
| Argon | 36 | 1.81 / 1.72 | 20.79 / 12.17 |

Statistics favour cpunative throughout these samples. Capillary field exports
are small and mixed; RDW export is slower at one and 36 cores. These absolute
costs cannot explain the multi-second propagation gains or serial RDW loss.
Large Cartesian field exports are a separate workload; the handoff records
their measured copy ablation. The standard Cartesian case requests no statistics.

## Matched scalar-libm control

These are short windows, length fraction 0.0025, with three pairs per setting.
Both backends use scalar libm as well as the matched compiler, flags, FFTW and
BLAS configuration. They isolate implementation gains from the SLEEF change;
they do not replace the full-run rankings. All 72 workers were quiet and work
counts matched.

| Case / method | Speedup, 1 core | Speedup, 36 cores |
|---|---:|---:|
| rdw / Adams | 1.04× | 1.31× |
| rdw / DOPRI5 | 0.95× | 1.14× |
| ctc / Adams | 1.27× | 1.35× |
| ctc / DOPRI5 | 1.35× | 1.35× |
| argon_blueshift / Adams | 1.07× | 1.36× |
| argon_blueshift / DOPRI5 | 1.06× | 1.34× |

## Scaling and saturation

The table uses complete runs: scaling is T(1) / T(P), and efficiency is that
scaling divided by P. Faster serial code can have lower scaling efficiency
while still finishing sooner. These are end-to-end driver measurements,
including observations and metadata, rather than isolated kernel throughput.

| Case / method | P | Native scaling / efficiency | cpunative scaling / efficiency |
|---|---:|---:|---:|
| RDW / Adams | 36 | 1.54× / 4.3% | 3.23× / 9.0% |
| RDW / DOPRI5 | 36 | 2.17× / 6.0% | 2.92× / 8.1% |
| CTC / Adams | 36 | 2.89× / 8.0% | 3.33× / 9.2% |
| CTC / DOPRI5 | 36 | 3.67× / 10.2% | 3.74× / 10.4% |
| Argon / Adams | 36 | 1.62× / 4.5% | 2.09× / 5.8% |
| Argon / DOPRI5 | 36 | 2.26× / 6.3% | 3.01× / 8.4% |
| Radial / Adams | 96 | 2.99× / 3.1% | 3.37× / 3.5% |
| Radial / DOPRI5 | 96 | 1.25× / 1.3% | 1.29× / 1.3% |
| Cartesian / RK4IP | 96 | 16.03× / 16.7% | 10.38× / 10.8% |

An additional EPYC sweep covers 1, 2, 4, 8, 16, 32, 64 and 96 cores, using
short windows and one interleaved pair per setting. All 128 workers were quiet,
all requested widths were met, and work counts matched. In those windows, the
lowest cpunative times occurred at 8 cores for RDW Adams, 16 for RDW DOPRI5,
16–32 for CTC, and 16 for argon and radial. The one-pair minima are exploratory;
they are not precise tuning thresholds or full-run rankings. At 96 cores several
capillary windows are slower than at one core. The large Cartesian case benefits
from 96 cores; these small capillary cases do not occupy a large node efficiently.


## Qualification and limits

Focused kernel, prescribed-step, lifecycle, thread-count and output checks pass.
The final consolidated source has 35 capillary parity checks on Xeon and 44
capillary/Cartesian checks on EPYC. The new Raman and Cartesian paths passed 23
ASan/UBSan checks in addition to earlier qualification. Complete capillary and
radial runs pass independently frozen convergence envelopes. The complete
Cartesian field agrees with Numba to about 1.16e-15 relative norm error, below
the fixed 4.31e-11 bound. CTC source checks cover 1–8 modes; repeated full timing
uses the unchanged three-mode standard case.

This qualifies the declared experimental subset, not the complete native kernel
contract. Release wheels, GCC 14 distribution qualification, full repository gates,
coverage lanes, and contended-Mac performance claims remain outside this experiment.
The original native history is preserved; these comparisons use separate bundles.
