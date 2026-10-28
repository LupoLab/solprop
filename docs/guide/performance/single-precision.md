# Single precision

solprop computes in IEEE binary64 (double precision) by default. A run can instead hold
its propagated field and its nonlinear evaluation in binary32 (single precision) with
`precision="single"` ([SEP 33](../../dev/seps/archive/sep-0033-single-precision-mlx.md)).
Single precision halves the memory of every field-sized buffer and is what graphics
processors without fast binary64 arithmetic compute in; whether a given problem keeps
its accuracy in binary32 is a property of the problem, so this page records which do.

```{note}
Single precision runs on the `mlx` backend, on the GPU of an Apple-silicon Mac
([choosing a backend](backends.md#the-apple-gpu-backend)), for capillary and fibre runs
and radial and Cartesian free space. `mlx` is **experimental** in this release: its
Metal kernels are not run in continuous integration, the survival record below was
measured on the reference implementation (on `mlx` for two cases only), and on the
default grids it can be slower than binary64 `native`; build single-precision grids
with `fft_size="pow2"` ([grid sizes on `mlx`](#grid-sizes-on-mlx)). `numpy_ref` runs
every route below as the reference implementation, which defines the arithmetic and
measures it but is not fast. `numba_cpu` and `native` compute in binary64 only.
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
interfaces. Under single precision:

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
  present, and is refused otherwise; it never resolves to a binary64 backend, nor to the
  reference implementation, which is named explicitly, `backend="numpy_ref"`.
- **Outputs** are unchanged: saved fields, statistics and processing are binary64, and
  files are the same size. The precision is part of the run's configuration, so a run is
  never resumed in the other precision.

## What it covers

| | `numpy_ref` | `mlx` |
|---|---|---|
| capillaries, solid-core and GNLSE fibres, mode-averaged and multimode, field and envelope grids | yes | yes |
| Kerr, the envelope grid's third-harmonic term, scalar and tensor Raman, gas Raman mixtures, the silica bundle | yes | yes |
| field-grid plasma | yes | yes |
| envelope-grid plasma | yes | yes, each sample's ellipse and cycle moments in one GPU kernel |
| radial and Cartesian free space, including a line, with their transverse collars | yes | yes |
| gradients, tapers, loss, both temporal absorber schemes, the polarisation taper | yes | yes, where the operator is device-ready (not a z table) |
| Runge–Kutta, the low-storage march, fixed-schedule batches | yes | yes |
| the Adams integrator | refused, pending its accuracy record | refused |
| noise models | refused (a known limitation, below) | refused |
| the general modal route, the hybrid planar kind, coupled flows, TDSE responses, the dispersive electronic response, the far-field statistics | refused | refused |

## Grid sizes on `mlx`

A grid's default sample count, `fft_size="auto"`, is the cheapest size with prime
factors 2, 3 and 5 by a cost model fitted to FFTW's transforms on CPUs
([grids](../setup/grids.md#coarse-and-fine-sampling)). MLX's GPU transform runs those
sizes four to eight times slower than the next power of two. On an Apple M5 Max one real
transform took:

| samples | µs | next power of two | µs |
|---|---|---|---|
| 12288 | 902 | 16384 | 218 |
| 98304 | 7629 | 131072 | 992 |

So on a default grid a single-precision run on `mlx` can be slower than a binary64 run
on `native`. For a single-precision run, build the grid with `fft_size="pow2"` and pass
it as `grid=`; the keyword interfaces build `"auto"` grids:

```python
from solprop.grid import FieldGrid

grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 3e-6), 200e-15, fft_size="pow2")
```

On the CPU backends `"auto"` stays the cheaper choice. Making powers of two the
single-precision default is planned, with a re-timing of `mlx` on such grids.

## How the arithmetic is kept accurate

Binary32 carries about seven significant digits and represents magnitudes between about
$10^{-38}$ and $3\times10^{38}$. solprop keeps both from limiting a run:

- **Range.** Every table a run multiplies by is narrowed with its own power-of-two
  exponent, and the field is held at an exact power-of-two scale chosen from the launch.
  A nitrogen Raman kernel whose spectrum is near $10^{-50}$ — zero in binary32 as it
  stands — keeps every bin; a Cartesian energy sum that would reach $10^{44}$ is scaled
  before squaring.
- **The linear propagator** $e^{\Delta\Phi}$ is formed to binary64 accuracy and rounded
  once per step. Its rounding still adds coherently from step to step, about
  $c\,M\,u$ after $M$ steps with $u = 2^{-24}$; the record below states $c$.
- **Populations.** The ionised fraction and the neutral survival are formed in forms that
  keep relative accuracy at tiny and saturated exposures, and running integrals over
  time are summed in a balanced order rather than one long sequential sum.
- **The controller, landing and every scalar** stay in binary64.

## The survival record

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
  every envelope. The reference's transforms now sum in binary32, as the backends' do
  (NumPy's float32 transforms are binary64 rounded once, which this record first used);
  on them the field error is 2.5×10⁻⁴ at 0.05 m and 1.6×10⁻³ at 0.1 m, so the limit
  stands ([SEP 36](../../dev/seps/archive/sep-0036-native-reduced-precision.md)).
- **The tolerance floor** stays at `1e-6`: tightening from `1e-5` to `1e-6` still buys
  accuracy (by 92 % in the self-phase-modulation case's field).
- **The propagator's step limit.** With no nonlinear source the rounding of
  $e^{\Delta\Phi}$ accumulates as $c\,M\,u$ after $M$ steps, with $c \approx 0.5$
  measured on a capillary operator over $10^2$–$10^4$ steps (0.1–0.3 on a GNLSE one).
  The field envelope of $10^{-3}$ is reached after about $3\times10^4$ steps; a run
  that needs more takes binary64.

## Known limitations

- **Noise.** A one-photon-per-mode seed is of order $10^{-5}$ of a microjoule pulse's peak
  spectral amplitude and smaller at higher energies, while binary32's accumulated
  rounding can reach $10^{-5}$ after $10^4$ steps. Noise-seeded runs are refused under
  single precision until that boundary is qualified.
- **Amplifying dynamics.** Rounding at the level of $10^{-7}$ per step is amplified like
  any perturbation by soliton fission, modulation instability or noise-free stimulated
  Raman scattering; such runs reach their accuracy over a shorter length, as the record
  states.
- **Performance.** On the reference implementation a single-precision run costs about as
  much as a double-precision one. On `mlx` the gain over `native` grows with the
  problem, and is largest for the envelope-grid plasma, whose cycle moments the GPU
  evaluates one sample per thread. On an otherwise idle MacBook Pro M5 Max (18 CPU
  cores, 40-core GPU), in the same process, on power-of-two grids and before the
  default grids and boundaries changed
  ([SEP 21](../../dev/seps/archive/sep-0021-absorbing-boundaries.md)), the median time
  of a fixed step on `native` (binary64) over that on `mlx` (binary32) was:

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
  benchmark suite, and they have not been repeated since: they indicate what a
  power-of-two grid gains, and do not hold on the default grids, where `mlx` can be
  slower than `native` ([grid sizes on `mlx`](#grid-sizes-on-mlx)).
- **`mlx` is experimental.** Its custom Metal kernels are tested on a Metal device by
  hand, not in continuous integration, whose macOS runners test the backend on MLX's
  CPU device; its coverage is not measured there. Its comb-to-continuum limit is
  unmeasured: the 0.05 m above is the reference implementation's, whose transforms are
  more accurate than MLX's. Power-of-two grids by default under single precision, a
  macOS `mlx` job in continuous integration and a re-timing on current grids come
  before it is fully supported.

The [worked example](../../auto_examples/performance/plot_01_single_precision.rst) runs a
capillary with the envelope-grid plasma and a radial filament in both precisions and
compares them. The precision setting, its binary32 contract, the survival study and the
`mlx` backend were proposed and carried out by
[SEP 33](../../dev/seps/archive/sep-0033-single-precision-mlx.md).
