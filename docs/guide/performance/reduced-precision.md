# Reduced precision

solprop computes in IEEE binary64 (double precision) by default. Two reduced modes trade
some of that arithmetic for speed and memory:

- `precision="single"` holds the propagated field and its nonlinear evaluation in
  binary32 (single precision)
  ([SEP 33](../../dev/seps/archive/sep-0033-single-precision-mlx.md)). It halves the
  memory of every field-sized buffer and is what graphics processors without fast
  binary64 arithmetic compute in.
- `precision="mixed"` keeps the propagated field, every stored stage and history row,
  the linear propagators, the boundary treatments and the error norms in binary64, and
  evaluates only the nonlinear source (its transforms, responses and in-stage collars)
  in binary32 ([SEP 36](../../dev/seps/sep-0036-native-reduced-precision.md)).

Whether a given problem keeps its accuracy in a reduced mode is a property of the
problem, so this page records which do.

```{note}
`numpy_ref` runs both modes as the reference implementation, which defines the
arithmetic and measures it but is not fast. The [native backend](native.md) runs both on
CPUs and GPUs, when its build compiled them (`SOLPROP_NATIVE_PRECISIONS`,
[building it](building-native.md#option-reference)). The `mlx` backend runs single
precision on the GPU of an Apple-silicon Mac
([choosing a backend](backends.md#the-apple-gpu-backend)); Apple GPUs have no binary64,
so it refuses mixed precision. `numba_cpu` computes in binary64 only.
```

## Asking for it

```python
from solprop.driver import prop_capillary

output = prop_capillary(
    125e-6,
    0.1,
    "He",
    0.8,
    lambda0=800e-9,
    fwhm=10e-15,
    energy=100e-6,
    wavelength_limits=(200e-9, 3e-6),
    trange=200e-15,
    precision="single",
    backend="numpy_ref",
)
```

`precision` is a keyword of `prop_gnlse`, `prop_capillary` and `prop_waveguide`, and a
field of {class}`~solprop.driver.simulation.SimulationSettings` for the free-space
interfaces; `precision="mixed"` is asked for the same way. Under single precision:

- **The tolerance floor.** The default `rtol` is `1e-6`
  ({data}`~solprop.translate.precision.SINGLE_RTOL_FLOOR`), and an explicit adaptive
  `rtol` below it is refused: a local error target below binary32's rounding per step
  buys steps without accuracy. A fixed schedule reads no tolerance.
- **The step's exponent.** Binary32 represents ``exp(x)`` only up to $x \approx 88.7$
  (binary64 to 709.8). The interaction picture multiplies by the linear propagator and
  its inverse, so a replayed step (or an adaptive run's floor `dz_min`) over which a
  strong collar or loss reaches that exponent is refused at translation, with the
  shortest step that fits, and one within a tenth of it is warned of. A 1 mm capillary's
  default collar on 0.25 mm steps is one such case.
- **The integrator.** The default is `RungeKutta()` (DOPRI5) and the low-storage march is
  admitted; the Adams integrator, `Multistep()`, is refused until its single-precision
  accuracy is established.
- **The backend.** `backend="auto"` resolves to `"mlx"` when MLX and a Metal GPU are
  present, otherwise to `"native"` when its build compiled single precision, and is
  refused otherwise; it never resolves to a binary64 backend, nor to the reference
  implementation, which is named explicitly, `backend="numpy_ref"`.
- **Outputs** are unchanged: saved fields, statistics and processing are binary64, and
  files are the same size. The precision is part of the run's configuration, so a run is
  never resumed in the other precision.

## Mixed precision

Under `precision="mixed"` each Runge–Kutta stage forms its input in binary64, narrows it
once to binary32 at the launch's power-of-two scale, evaluates the source in binary32,
and widens the result exactly before the binary64 interaction picture carries it back.
The linear part of the step is binary64, so the propagator's coherent drift and the
state's rounding (the two error terms that limit single precision's length) fall to
binary64 levels. What remains is the source's binary32 rounding, which enters the state
scaled by the step. Concretely:

- **The tolerance floor.** The default `rtol` is the larger of the interface's binary64
  default and `1e-8` ({data}`~solprop.translate.precision.MIXED_RTOL_FLOOR`); an explicit
  adaptive `rtol` below `1e-8` is refused. A step whose source rounding exceeds the
  target is rejected and shortened, so the controller cannot stall; at worst it takes
  more steps than binary64 would.
- **No damping limit.** The propagators are binary64, so mixed precision has no
  `exp(88.7)` limit on a step's exponent.
- **The integrator.** Each interface keeps its binary64 default: `Multistep()` for
  the capillary and the radial and planar routes, `RungeKutta()` for the GNLSE, the
  waveguide and free space. Both are admitted, as is the low-storage march. Against
  Runge–Kutta at one tolerance, mixed Adams is as accurate within every envelope, faster
  on a Kerr-driven capillary and slower on an ionising one, as in binary64; choose as
  [choosing a stepper](../numerics/choosing-a-stepper.md) says.
- **Noise.** Noise models are refused until the noise study (below) admits them.
- **The backend.** `backend="auto"` resolves to `"native"` when its build compiled mixed
  precision, and is refused otherwise.
- **Outputs** are binary64 and unchanged, as under single precision, and the precision is
  part of the configuration.

## When to use each mode

| | binary64 (default) | mixed | single |
|---|---|---|---|
| accuracy | the reference | the source's rounding only: close to binary64 on every case measured so far | the state's and propagator's rounding too, about $c\,M\,u$ after $M$ steps |
| tolerance floor | none | `1e-8` | `1e-6` |
| long runs (more than about $3\times10^4$ steps) | yes | yes | no: the propagator's drift reaches the field envelope |
| strong damping per step | yes | yes | only below `exp(88.7)` per step |
| memory of field-sized buffers | 1 | about 1 (the source's buffers halve) | about 1/2 |
| speed | the reference | gains where the source dominates a step, most on GPUs with slow binary64 | the largest gains, and on Apple GPUs the only option |
| noise-seeded runs | yes | refused until the noise study (phase 5) | refused |

Take **mixed precision** when a run is dominated by its nonlinear source (fine time
grids, plasma, Raman, many modes) on hardware whose binary32 arithmetic is much faster
than its binary64, and you want binary64's behaviour on long or strongly damped
propagations. Take **single precision** for the largest memory and speed gains, on
problems inside its survival record, and on Apple GPUs. Keep **binary64** when a result
is sensitive at the level of $10^{-7}$ per step (noise-free stimulated Raman scattering,
modulation instability from rounding) or when a run needs noise models.

## What it covers

Single precision:

| | `numpy_ref` | `native` CPU | `native` GPU | `mlx` |
|---|---|---|---|---|
| capillaries, solid-core and GNLSE fibres, mode-averaged and multimode, field and envelope grids | yes | yes | yes | yes |
| Kerr, the envelope grid's third-harmonic term, scalar and tensor Raman, gas Raman mixtures, the silica bundle | yes | yes | yes | yes |
| field-grid plasma | yes | yes | yes | yes |
| envelope-grid plasma | yes | yes | yes | yes, each sample's ellipse and cycle moments in one GPU kernel |
| radial and Cartesian free space, including a line, with their transverse collars | yes | yes | yes | yes |
| gradients, tapers, loss, both temporal absorber schemes, the polarisation taper | yes | yes | yes, where the operator is device-ready (not a z table) | yes, where the operator is device-ready (not a z table) |
| Runge–Kutta, the low-storage march, fixed-schedule batches | yes | yes | yes | yes |
| the Adams integrator | refused, pending its accuracy record | refused | refused | refused |
| noise models | refused (a known limitation, below) | refused | refused | refused |
| the general modal route, the hybrid planar kind, coupled flows, TDSE responses, the dispersive electronic response, the far-field statistics | refused | refused | refused | refused |

Mixed precision covers the same capabilities, on `numpy_ref` and `native` (CPU and GPU)
only, with no damping limit, and also admits the Adams integrator; `mlx` and `numba_cpu`
refuse it. Its noise models are refused until SEP 36's noise study admits them. The `native`
GPU column is qualified on an NVIDIA A40 (CUDA); an AMD Instinct GPU (HIP) runs the same
device code and is qualified separately.

## How the arithmetic is kept accurate

Binary32 carries about seven significant digits and represents magnitudes between about
$10^{-38}$ and $3\times10^{38}$. solprop keeps both from limiting a run:

- **Range.** Every table a run multiplies by is narrowed with its own power-of-two
  exponent, and the field is held at an exact power-of-two scale chosen from the launch.
  A nitrogen Raman kernel whose spectrum is near $10^{-50}$ — zero in binary32 as it
  stands — keeps every bin; a Cartesian energy sum that would reach $10^{44}$ is scaled
  before squaring.
- **The linear propagator** $e^{\Delta\Phi}$ is formed to binary64 accuracy and rounded
  once per step (under mixed precision it stays binary64). Its rounding still adds
  coherently from step to step, about $c\,M\,u$ after $M$ steps with $u = 2^{-24}$; the
  record below states $c$. The rounding must also be unbiased: a propagator multiplies
  the field once per step, so a mean bias of its modulus is an energy drift that Kerr
  phase turns into a field error growing faster than the step count. The binary32
  `sinf` and `cosf` of one GPU library, accurate to 2 ulp, biased the squared modulus by
  $+0.3\,u$ on average, so every backend rounds the binary64 functions instead.
- **Populations.** The ionised fraction and the neutral survival are formed in forms that
  keep relative accuracy at tiny and saturated exposures. Running integrals over time are
  summed in a balanced order on `numpy_ref` and accumulated in binary64 on `native`.
- **Reductions.** Error norms and statistics are binary64 sums of the physical field.
- **Matrix products and transforms** are binary32 library calls that solprop checks: a
  GPU build sets the matrix library to its strict binary32 mode and measures its
  products before a run (below), and the transforms are tested against their bounds.
- **The controller, landing and every scalar** stay in binary64.

## The survival record

Single precision's record, from SEP 33 on `numpy_ref`:

Each case below was run in single precision at `rtol = 1e-6` and `1e-5` and compared, at
the end of its window, with a binary64 run on the same grid at `rtol = 1e-10`, against
accuracy envelopes fixed before any comparison: the field to $10^{-3}$, the spectral
energy in each of 32 bands above −40 dB to 1 %, the total energy to $10^{-4}$, the peak
power or intensity, the final ionised fraction and the beam width to $10^{-3}$, all
relative. The table gives the largest of those relative errors at `rtol = 1e-6`, the
field's error, and binary64's own field error at the same tolerance for comparison.

| case | single: field error | largest other metric | binary64 at `1e-6`: field | verdict |
|---|---|---|---|---|
| self-phase modulation, 10π, no dispersion (GNLSE) | 7.9e-6 | 9.6e-5 (bands) | 1.9e-6 | survives |
| fundamental soliton, ten periods (GNLSE) | 2.7e-5 | 6.5e-6 (peak) | 1.1e-6 | survives |
| dispersive-wave emission, He capillary gradient, two modes, 1.5 m | 2.6e-6 | 4.9e-5 (bands) | 1.3e-6 | survives |
| comb-to-continuum, H₂ capillary at 40 bar, 0.1 m, noise-free | 2.6e-3 | 3.4e-4 (peak) | 3.0e-7 | survives over 0.05 m |
| plasma blue shift, Ar capillary, field grid, 0.4 m | 1.8e-4 | 2.1e-4 (bands) | 1.6e-4 | survives |
| plasma blue shift, Ar capillary, envelope grid, 0.4 m | 1.3e-5 | 2.1e-5 (ionised) | 1.3e-5 | survives |
| molecular Raman, N₂ capillary at 2 bar, 0.3 m | 1.1e-6 | 2.7e-5 (bands) | 1.6e-7 | survives |
| radial Kerr beam, 0.3 m | 8.5e-6 | 1.3e-4 (bands) | 8.6e-6 | survives |
| radial filament in argon with plasma, 3 cm | 2.3e-6 | 1.2e-5 (peak) | 2.2e-6 | survives |
| Cartesian Gaussian diffraction, 64 × 64 | 2.4e-7 | 2.4e-7 (bands) | 0 | survives |
| Cartesian Kerr beam, 64 × 64 | 2.7e-6 | 2.0e-5 (bands) | 4.2e-6 | survives |

At `rtol = 1e-5` every case but the comb-to-continuum stays inside its envelopes too.
The table is the reference implementation's. On `mlx`, whose envelope-grid plasma is
its own discretisation of the cycle moments, the envelope-grid blue shift over its full
0.4 m has field error 4.8e-5 (largest other 5.3e-5, the ionised fraction) and the
field-grid one 2.2e-4 (2.8e-4, bands): both survive.

- **The comb-to-continuum limit.** Noise-free stimulated Raman scattering grows from
  whatever seed the Stokes band holds; rounding seeds it at the level of the unit
  roundoff times the field. In binary64, adding a seed of $2^{-24}$ of the launch's peak
  spectral amplitude moves the result at 0.1 m by 5 %: the sensitivity is the
  physics, and binary32's own rounding is such a seed. Over 0.05 m the run is inside
  every envelope.
- **The tolerance floor** stays at `1e-6`: tightening from `1e-5` to `1e-6` still buys
  accuracy (by 92 % in the self-phase-modulation case's field).
- **The propagator's step limit.** With no nonlinear source the rounding of
  $e^{\Delta\Phi}$ accumulates as $c\,M\,u$ after $M$ steps, with $c \approx 0.5$
  measured on a capillary operator over $10^2$–$10^4$ steps (0.1–0.3 on a GNLSE one).
  The field envelope of $10^{-3}$ is reached after about $3\times10^4$ steps; a run
  that needs more takes binary64.

### Mixed precision's record

```{admonition} Phase 5
:class: note
SEP 36's mixed-precision survival study is in progress. Its record, with the floor it
fixes and the Adams integrator's measurement, will appear here. Native's confirmation
runs so far agree with `numpy_ref`'s mixed path and stay within binary64's accuracy
envelopes.
```

### Noise under mixed precision

```{admonition} Phase 5
:class: note
Noise models are refused under mixed precision until SEP 36's noise study qualifies
them. The study compares mixed and binary64 ensembles with the same seeds, band by band
({py:func}`~solprop.processing.coherence.band_coherence`), and replays each binary64 run's
accepted steps in mixed precision
({py:meth}`~solprop.stepper.schedule.FixedSchedule.from_history`) so that the two differ
by rounding alone. A noise-seeded mixed run records a heuristic **rounding margin** in
its execution metadata (`rounding_margin`): at each saved plane, the accumulated binary32
rounding $\sigma_r = c_s u \log_2 n \sqrt{\Sigma\phi^2\, n_\mathrm{occ}/n}$ from the
accepted steps' nonlinear phases $\phi$, beside the quantum seed's amplitude $\sigma_q$,
and their ratio in dB. It omits gain and coherent bias, and says so; $c_s$ is calibrated
by the study. The study's record and the noise models' admission will appear here.
```

## Known limitations

- **Noise.** A one-photon-per-mode seed is of order $10^{-5}$ of a microjoule pulse's peak
  spectral amplitude and smaller at higher energies, while binary32's accumulated
  rounding can reach $10^{-5}$ after $10^4$ steps. Noise-seeded runs are refused under
  single precision, and under mixed precision until its noise study (above) qualifies
  them.
- **Amplifying dynamics.** Rounding at the level of $10^{-7}$ per step is amplified like
  any perturbation by soliton fission, modulation instability or noise-free stimulated
  Raman scattering; such runs reach their accuracy over a shorter length, as the record
  states.
- **Performance.** On the reference implementation a single-precision run costs about as
  much as a double-precision one. On `mlx` the gain over `native` grows with the
  problem, and is largest for the envelope-grid plasma, whose cycle moments the GPU
  evaluates one sample per thread. On an otherwise idle MacBook Pro M5 Max (18 CPU cores, 40-core GPU), in the same
  process, the median time of a fixed step on `native` (binary64) over that on `mlx`
  (binary32) was:

  | case | `native` / `mlx` |
  |---|---|
  | GNLSE with Raman, one line, 2¹⁴ / 2¹⁶ / 2¹⁸ samples | 1.7 / 3.4 / 5.3 |
  | Ar capillary, field-grid plasma, 1 / 4 modes (300 fs) and 8 modes (1 ps) | 0.51 / 0.69 / 1.12 |
  | the same with the envelope-grid plasma | 8.1 / 8.1 / 3.5 |
  | radial Kerr beam, 64 / 256 nodes | 1.05 / 2.9 |
  | Cartesian Kerr beam, 64 × 64 / 128 × 128 | 1.9 / 1.9 |
  | radial argon filament, field grid, 64 / 128 / 256 nodes | 1.8 / 2.5 / 3.2 |

  Small field-plasma capillaries are slower on the GPU, bound by its per-operation
  latency. These are same-process comparisons on one machine, not the standard
  benchmark suite.
- **Performance on `native`.**

  ```{admonition} Phase 5
  :class: note
  The time per step of single and mixed precision against binary64 on `native`, on CPUs
  and GPUs, with the batch and concurrency throughput, is measured by SEP 36's
  benchmarks and will be tabulated here.
  ```

The [worked example](../../auto_examples/performance/plot_01_single_precision.rst) runs a
capillary with the envelope-grid plasma and a radial filament in both precisions and
compares them. The precision setting, its binary32 contract, the survival study and the
`mlx` backend were proposed and carried out by
[SEP 33](../../dev/seps/archive/sep-0033-single-precision-mlx.md); mixed precision and
both modes on `native` by [SEP 36](../../dev/seps/sep-0036-native-reduced-precision.md).
