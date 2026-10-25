# Single precision

solprop computes in IEEE binary64 (double precision) by default. A run can instead hold
its propagated field and its nonlinear evaluation in binary32 (single precision) with
`precision="single"` ([SEP 33](../../dev/seps/sep-0033-single-precision-mlx.md)).
Single precision halves the memory of every field-sized buffer and is what graphics
processors without fast binary64 arithmetic compute in; whether a given problem keeps
its accuracy in binary32 is a property of the problem, so this page records which do.

```{note}
Single precision runs on the `mlx` backend, on the GPU of an Apple-silicon Mac
([choosing a backend](backends.md#the-apple-gpu-backend)), for capillary and fibre runs
and radial and Cartesian free space. `numpy_ref` runs every route below as the
reference implementation, which defines the arithmetic and measures it but is not fast.
`numba_cpu` and `native` compute in binary64 only.
```

## Asking for it

```python
from solprop.driver import prop_capillary

output = prop_capillary(
    125e-6, 1.5, "He", (0.8, 0.0),
    lambda0=800e-9, fwhm=7.5e-15, energy=275e-6, modes=2,
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
| envelope-grid plasma | yes | refused, a later phase |
| radial and Cartesian free space, including a line, with their transverse collars | yes | yes |
| gradients, tapers, loss, both temporal absorber schemes, the polarisation taper | yes | yes, where the operator is device-ready (not a z table) |
| Runge–Kutta, the low-storage march, fixed-schedule batches | yes | yes |
| the Adams integrator | refused, pending its accuracy record | refused |
| noise models | refused (a known limitation, below) | refused |
| the general modal route, the hybrid planar kind, coupled flows, TDSE responses, the dispersive electronic response, the far-field statistics | refused | refused |

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
| plasma blue shift, Ar capillary, envelope grid, 0.1 m | 2.7e-6 | 4.2e-5 (bands) | 1.7e-6 | survives |
| molecular Raman, N₂ capillary at 2 bar, 0.3 m | 1.1e-6 | 2.7e-5 (bands) | 1.6e-7 | survives |
| radial Kerr beam, 0.3 m | 8.5e-6 | 1.3e-4 (bands) | 8.6e-6 | survives |
| radial filament in argon with plasma, 3 cm | 2.3e-6 | 1.2e-5 (peak) | 2.2e-6 | survives |
| Cartesian Gaussian diffraction, 64 × 64 | 2.4e-7 | 2.4e-7 (bands) | 0 | survives |
| Cartesian Kerr beam, 64 × 64 | 2.7e-6 | 2.0e-5 (bands) | 4.2e-6 | survives |

At `rtol = 1e-5` every case but the comb-to-continuum stays inside its envelopes too.

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
  problem: on one Apple-silicon laptop, in the same process, a GNLSE step on a single
  line took 0.65, 0.36 and 0.19 times `native`'s time at 2¹⁴, 2¹⁶ and 2¹⁸ samples,
  and an eight-mode capillary with ionisation on a 1 ps window 0.83 times. In free
  space, ten Kerr steps took 0.87 and 0.26 times `native`'s time on 64 and 256 radial
  nodes, about 0.5 times on 64 × 64 and 128 × 128 Cartesian grids, and a field-grid
  argon filament 0.68 and 0.43 times on 64 and 128 radial nodes. Small
  capillary runs (one to four modes on a 300 fs window) are slower on the GPU, 2–3
  times `native`'s time, bound by the GPU's per-operation latency. These are indicative
  figures from a loaded machine, not benchmarks.
