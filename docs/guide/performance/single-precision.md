# Single precision

solprop computes in IEEE binary64 (double precision) by default. A run can instead hold
its propagated field and its nonlinear evaluation in binary32 (single precision) with
`precision="single"` ([SEP 33](../../dev/seps/sep-0033-single-precision-mlx.md)).
Single precision halves the memory of every field-sized buffer and is what graphics
processors without fast binary64 arithmetic compute in; whether a given problem keeps
its accuracy in binary32 is a property of the problem, so this page records which do.

```{note}
Today single precision runs on `numpy_ref`, the reference implementation, which exists
to define the arithmetic and to measure it, not to be fast: a single-precision run on it
takes about as long as a double-precision one. A GPU backend that computes in binary32
is the next phase of SEP 33. `numba_cpu` and `native` compute in binary64 only.
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
- **The backend.** `backend="numpy_ref"` must be named. `backend="auto"` never resolves to
  a binary64 backend for a single-precision run, nor to the reference implementation;
  with no GPU backend installed it is refused.
- **Outputs** are unchanged: saved fields, statistics and processing are binary64, and
  files are the same size. The precision is part of the run's configuration, so a run is
  never resumed in the other precision.

## What it covers

| | single precision |
|---|---|
| capillaries, solid-core and GNLSE fibres, mode-averaged and multimode, field and envelope grids | yes |
| Kerr, the envelope grid's third-harmonic term, scalar and tensor Raman, gas Raman mixtures, the silica bundle | yes |
| field-grid and envelope-grid plasma | yes |
| radial and Cartesian free space, including a line | yes |
| gradients, tapers, loss, both temporal absorber schemes, the polarisation taper | yes |
| Runge–Kutta, the low-storage march, fixed-schedule batches | yes |
| the Adams integrator | refused, pending its accuracy record |
| noise models | refused (a known limitation, below) |
| the general modal route, the hybrid planar kind, coupled flows, TDSE responses, the dispersive electronic response, the far-field statistics | refused |

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

```{include} single-precision-record.md
```

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
  much as a double-precision one.
