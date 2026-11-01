# Input pulses

Every propagation starts from an initial spectrum. {py:mod}`solprop.pulses` builds one
from a physical description — a duration, a centre wavelength, an energy, a spectral
phase — and hands back the array the solver expects. The keyword drivers
(`prop_gnlse`, `prop_capillary`, `prop_waveguide`) build a Gaussian or sech pulse from
`fwhm`, `energy` or `peak_power` and `phases` themselves; the builders on this page are
what they use, and what to reach for when the input is anything else — a super-Gaussian,
a measured spectrum, a continuous wave, several pulses at once, or a transverse beam.

This page covers the six temporal builders, the two ways of fixing a pulse's size, the
sign and phase-origin conventions that decide where in the window a pulse lands and
which way it chirps, measured and continuous-wave inputs, multi-colour inputs, and the
transverse beams of the spatially resolved geometries. The normative statements are in
[conventions](../../dev/conventions.md), §2 and §4; this page derives them and shows them at
work.

## One method, two grids

A pulse in solprop is a *specification*, not an array. Each builder is a frozen dataclass
with a single method:

```{code-block} python
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.pulses import GaussianPulse

pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)

field = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 2e-12)
envelope = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 2e-12)

spectrum_field = pulse.build(field)        # (1, 1, n_omega), real field's spectrum
spectrum_envelope = pulse.build(envelope)  # (1, 1, n_omega), envelope's spectrum
```

The same specification builds against either grid and gives the same physical pulse. On a
field grid the time samples are $\sqrt{I(t)}\cos(\omega_p t)$, whose analytic signal is
the pulse; on an envelope grid they are $\sqrt{I(t)}\,e^{i(\omega_p-\omega_0)t}$, which
is that analytic signal with the grid's carrier removed. Because
[conventions](../../dev/conventions.md), §4, gives the two grids one energy functional,
the spectral energy density that comes out is the same function of $\omega$ either way,
and the test suite checks that it is.

The returned array is always `(n_modes, n_pol, n_omega)` with both leading axes equal to
one, which is the canonical kernel layout of [conventions](../../dev/conventions.md), §6.
Nothing squeezes it; a multimode input is a stack of these.

The six builders are {py:class}`~solprop.pulses.GaussianPulse`,
{py:class}`~solprop.pulses.SechPulse`, {py:class}`~solprop.pulses.CustomPulse`,
{py:class}`~solprop.pulses.DataPulse`, {py:class}`~solprop.pulses.TimeDataPulse` and
{py:class}`~solprop.pulses.CWPulse`. They all
satisfy the {py:class}`~solprop.pulses.Pulse` protocol, which is what downstream code
types against. The first three share {py:class}`~solprop.pulses.ShapedPulse`, which holds
the parameters they have in common and the whole of `build`; subclass it and supply an
`intensity(t)` to add a shape of your own. A shape function that returns the wrong
length, a negative value, a non-finite value or nothing but zeros is rejected with a
message that names the problem, so a bad envelope never reaches the initial state as a
`NaN`.

| builder | intensity $I(t)$ | duration given as |
|---|---|---|
| `GaussianPulse` | $\exp\!\left[-\tfrac12 (t/\sigma)^{2m}\right]$ | `fwhm` (power FWHM), order `m` |
| `SechPulse` | $\operatorname{sech}^2(t/\tau_w)$ | `fwhm` or the natural width `width` |
| `CustomPulse` | any callable `I(t)` | implicit in the callable |
| `DataPulse` | measured spectrum, interpolated | implicit in the data |
| `TimeDataPulse` | measured field, transformed exactly | implicit in the data |
| `CWPulse` | monochromatic or $\operatorname{sech}^2$ in $\omega$ | `bandwidth`, a wavelength FWHM |

The FWHM conversions are exact for every order. A super-Gaussian of order $m$ uses
$\sigma = \tau/\left[2(2\ln 2)^{1/2m}\right]$, so that $I(\tau/2) = \tfrac12$ whatever
$m$ is; a $\operatorname{sech}^2$ pulse uses $\tau_w = \tau/[2\ln(1+\sqrt2)]$.

## Energy or peak power, never both

A pulse is scaled by its `energy` or by its `peak_power`. Giving both, or neither, is an
error rather than a precedence rule. Both are realised *exactly* on the grid: the
spectrum is built, the
phase is applied, the realised quantity is measured with the grid's own functional, and
one multiplication by a square root fixes it. Nothing is left to a quadrature error, and
the round trip closes to a part in $10^{12}$ or better.

The order matters: **the peak power is the peak power after the spectral phase**.
For a transform-limited input, broadening it by chirp while holding its peak
power fixed therefore raises its energy. Holding the energy fixed is a different
experimental constraint. Scaling by energy is insensitive to the order,
because a spectral phase moves energy about in time but does not create or destroy it.

:::{warning}
A pulse whose spectrum falls outside the grid's window, or whose duration is longer than
the time window, is not silently truncated into something plausible: it is scaled so that
whatever the grid *does* hold carries the requested energy. Check the duration against
`trange` and the bandwidth against the wavelength limits before reading a result.
:::

## Signs: where a pulse lands and which way it chirps

A phase convention determines the sign of a delay or chirp and the time origin
of a measured spectrum. Use the same convention when specifying the input and
interpreting the output.

**The interface is the optics convention.** A spectral phase given to a builder (its
`phase` field, the `phases` keyword of the drivers) is
$\tilde E(\omega) = |\tilde E|\,e^{i\phi(\omega)}$ with

$$ \phi(\omega) = \omega t_0 + \sum_{n} \frac{\phi_n}{n!}\,(\omega - \omega_p)^n, $$

$\omega_p$ being the pulse's own centre frequency, not necessarily the grid's reference.
The group delay is $T_g(\omega) = \mathrm{d}\phi/\mathrm{d}\omega$, so

- a positive `t0` makes the pulse **arrive later** (it must lie strictly inside the
  window: a delay is a spectral phase, and one past the half-window would wrap the pulse
  round to the other end, so it is refused);
- `phase[1]`, the first-order coefficient, is a group delay and does exactly what `t0`
  does;
- a positive `phase[2]` (GDD) gives $T_g = \phi_2(\omega-\omega_p)$, so the blue arrives
  after the red: an **upchirp**, the sign of propagation through normally dispersive
  glass;
- `phase[0]` is the carrier-envelope phase at the pulse's own peak, on either grid and
  independent of the window length.

**The stored array is the FFT convention.** The conversion happens once per build, where
the phase factor is formed, through {py:func}`~solprop.fourier.to_internal`. The pulse
builders are one of only three places where the optics convention meets the internal one
(see the [design decisions](../../dev/decisions.md)); nothing else in
{py:mod}`solprop.pulses` conjugates anything.

### The phase origin, and why `(−1)^k` exists

A stored spectrum is the raw forward DFT of the sample array, so its phase origin is the
**first sample**, at $t = -n_t\,\delta t/2$, not the centre of the window. Referenced to
$t = 0$ the stored bin $k$ therefore carries an extra $e^{+i\omega_k n_t \delta t/2}$,
and since $\omega_k = k\,\delta\omega$ with $\delta\omega = 2\pi/(n_t\delta t)$,

$$ \omega_k \frac{n_t \delta t}{2} = \pi k \qquad\Longrightarrow\qquad
   e^{+i\omega_k n_t \delta t / 2} = (-1)^k . $$

On an envelope grid the transform's own variable is the *relative* frequency, whose bin
$k$ gives $\pi k$ as well because $n_t$ is even, so the factor is $(-1)^k$ on the storage
index for both layouts. Putting the two halves together,

$$ \tilde E_{\text{optics},k} = (-1)^k\,\overline{\tilde E_k}, $$

which is what {py:func}`~solprop.fourier.spectrum_to_optics` and
{py:func}`~solprop.fourier.spectrum_to_internal` do. Both are involutive and each is the
other's inverse.

The practical consequence: a transform-limited pulse sitting at $t = 0$ reads back a
**flat** optics phase, while its stored phase alternates by $\pi$ from bin to bin. That
alternation is not a bug and not a chirp; it is where the window's centre is.

A **phase increment**, such as a Taylor phase or an extra delay, multiplies an
existing spectrum. Its phasor needs the sign conversion, while the existing
spectrum already carries its time origin. A **complete measured spectrum**
specified about $t=0$ needs both the conjugation and the origin shift in the
spectrum helpers. {py:class}`~solprop.pulses.DataPulse` handles that conversion.
This distinction concerns whether an array is a complete spectrum or a phase
factor; multiplication of spectral phase factors itself is commutative.

The delay uses the absolute $\omega$ on both grids, so it moves the whole field rather
than only its envelope: the carrier-envelope phase slips by $\omega_0 t_0$ and the field
and envelope representations stay in step.

## Measured pulses

{py:class}`~solprop.pulses.DataPulse` stores a measured complex spectrum, because that
is what a spectrometer and a phase retrieval give.
{py:class}`~solprop.pulses.TimeDataPulse` stores a measured field in time, because that is
what a field measurement gives.

```{code-block} python
from solprop.pulses import DataPulse, TimeDataPulse

DataPulse(omega=omega, spectrum=complex_spectrum, energy=1e-6)
DataPulse.from_spectral_density(omega, intensity, spectral_phase, energy=1e-6)
TimeDataPulse(t=t, field=measured_field, energy=1e-6)
```

A `DataPulse` interpolates the magnitude and the phase separately with a cubic B-spline.
Interpolating the complex spectrum itself would be wrong: its real and imaginary parts
oscillate at the pulse's arrival time, and no smooth interpolant follows them.
`from_spectral_density` keeps the phase it is given, in the pulse's `spectral_phase`,
and that phase is interpolated as it stands. A complex `spectrum` holds its phase only
modulo $2\pi$, so it is unwrapped first, and unwrapping cannot follow a change of more
than $\pi$ between neighbouring samples: 20 000 fs² of group-delay dispersion on a 1.6 nm
axis at 800 nm is built quite wrongly, and with no warning. Give a measured phase through
`from_spectral_density`. Outside the data's own frequency range the built spectrum is
exactly zero, the grid's spectral collar is applied so that nothing is left sitting on the
window edge, and a warning is logged when the grid's active band reaches past the data.

A spline through a magnitude and a phase cannot follow a spectral fringe to its null, so a
`DataPulse` suits one compact pulse and not several far apart. If the measurement is a
field in time, use a `TimeDataPulse`, which interpolates nothing.

A `TimeDataPulse` evaluates the record's transform at the grid's own frequencies,

$$
\tilde{E}(\omega_k) = \sum_n E_n\, e^{+i\omega_k t_n},
$$

with the chirp-z transform, whose cost is that of a few FFTs of the record and the grid
together. This is exact whatever the record holds: any number of pulses, at any separation,
with any fringe contrast, and each lands where it sat wherever the record starts, because
$t_n$ is the record's own axis. It is what the record says rather than what the pulse was:

- The spectrum is zero at frequencies above the record's Nyquist frequency $\pi/\Delta t$,
  and a warning is logged when the grid's active band reaches past it.
- A pulse that reaches the ends of the record is cut there, and a constant baseline is a
  step at each end, whose spectrum falls with frequency. A baseline of 1 % of the peak on a
  20 fs Gaussian changes the built spectrum by 0.15 %, so remove a baseline first.
- A record longer than the grid's window folds back into it, as any spectrum sampled at the
  grid's spacing must, and a warning is logged when that takes more than a millionth of the
  energy.

A real array is the electric field; a complex one is the analytic signal in the optics
convention. The run record holds the record itself, which costs the same as the spectrum
at its own bins: two real numbers for each real sample.

## Continuous wave

{py:class}`~solprop.pulses.CWPulse` writes its spectrum straight onto the grid's own
bins, so the field is periodic in the time window by construction and needs no
apodisation. `bandwidth = 0` puts everything on the single nearest bin; a positive
`bandwidth`, given as a wavelength FWHM and converted with $\Delta\omega = 2\pi c\,
\Delta\lambda/\lambda^2$, gives a $\operatorname{sech}^2$ power spectrum with a uniform
random spectral phase. The conversion is first order at the centre wavelength: the
spectrum's **frequency** FWHM is $\Delta\omega$, the frequency equivalent of the
requested width there, and its wavelength FWHM equals `bandwidth` only in the
narrowband limit (a wide spectrum is asymmetric in wavelength). The phase is drawn from a generator of its own, never the global
one: give `seed` and the realisation is reproducible (the seed is a field of the
dataclass and appears in its description); leave it `None` and each build draws a fresh
phase. A `CWPulse` builds on either grid.

The `average_power` is realised in one pass, over the samples where the temporal collar
is exactly one: the spectrum is written onto the bins and scaled once so that the mean
power over that flat part of the window is the requested one. There is no iteration
between a fixed spectral magnitude and a windowed field; the single pass trades a
slightly perturbed spectral shape for an exact power and a bounded cost.

## Several pulses at once

Two-colour and multi-colour experiments — a pump and a seed for four-wave mixing, a
strong pump and a weak probe for cross-phase modulation, a pulse and a delayed replica —
launch the sum of independently specified pulses.
{class}`~solprop.pulses.PulseSum` holds such a set: each member is any pulse above, with
its own centre wavelength, duration, energy or peak power, spectral phase and delay
`t0`, and is built and scaled on the grid exactly as it would be alone; the input is the
sum of the members' fields. The energies of members that are separated in time or in
frequency therefore add, while members that overlap interfere, as fields do: two
identical pulses in phase carry four times the energy of one.

The keyword interfaces (`prop_gnlse`, `prop_capillary`, `prop_waveguide` and their
`Simulation` constructors) take the members as `pulses=[...]`, in place of the
single-pulse keywords
(`fwhm`, `width`, `shape`, `energy`, `peak_power`, `phases`, which must then be
left unset); `wavelength` still sets the grid's reference wavelength, and the grid's
`wavelength_limits` must cover every member:

```{code-block} python
from solprop.driver import prop_capillary
from solprop.pulses import GaussianPulse

pump = GaussianPulse(wavelength=267e-9, fwhm=35e-15, energy=95e-6)
seed = GaussianPulse(wavelength=800e-9, fwhm=35e-15, energy=45e-6, t0=20e-15)
run = prop_capillary(
    75e-6,
    1.5,
    "He",
    0.48,
    wavelength=267e-9,
    pulses=[pump, seed],
    wavelength_limits=(120e-9, 3e-6),
    trange=400e-15,
)
```

A positive `t0` makes a member arrive later, so here the seed trails the pump by 20 fs.
The members are recorded in the run's configuration, and a `Simulation` built with
`Simulation.from_capillary(..., pulses=...)` or from any `initial=PulseSum(...)` behaves
the same way. The worked examples on
[four-wave mixing to the vacuum ultraviolet](../../auto_examples/processes/plot_03_four_wave_mixing)
and on [cross-phase modulation in the deep ultraviolet](../../auto_examples/processes/plot_04_cross_phase_modulation)
use this.

## Transverse beams

The builders above describe a pulse in time. For the spatially resolved geometries a
temporal pulse is wrapped in a beam: {py:class}`~solprop.pulses.SpatialPulse` gives it a
(super-)Gaussian transverse profile with a 1/e² intensity radius `waist` and a
super-Gaussian `order`, a signed launch distance from the waist (`propz`, negative to
launch before it) and a Jones vector (`jones`, see [polarisation](polarisation.md)), for
the [free-space](../free-space/model.md) radial, Cartesian and line grids; {py:class}`~solprop.pulses.PlanarPulse` projects an astigmatic Gaussian beam
onto the slab modes of the hybrid planar grid (see
[spatial decompositions](spatial-decompositions.md)). In both, the temporal pulse's
`energy` or `peak_power` is the whole beam's physical energy or peak axial power through
the launch plane, in the medium there — per metre of y (J/m, W/m) on a line grid, which
is uniform in y. A beam built directly, `SpatialPulse.build(grid, space, medium=...,
z=..., side=...)`, names that medium and plane; there is no vacuum default. A medium of
index $n_0$ therefore carries a field $\sqrt{n_0}$ lower than vacuum for the same energy
([energy and flux in a medium](../free-space/flux.md)).

A beam launched into a guide's modes, or the output field of another run used as an
input, is a coupling rather than a pulse: the beam keeps its energy over its whole
incident plane and the modes receive only what they capture
([coupling simulations](../workflows/coupling.md)).

## Where the quantum floor sits

Building a pulse says nothing about the noise it sits on. The shot-noise background is a
separate object with its own page, [noise](../media/noise.md); the two are added by the driver,
never by a builder. {py:func}`~solprop.noise.photons_per_bin` is the bridge: it reports
how far above one photon per bin a built pulse sits, bin by bin.

Where solprop's choices differ from Luna.jl's, the differences are listed in
[Noteworthy differences from Luna](../workflows/luna.md).

## Related calculations

For a complete launch and analysis, follow the [RDW lesson](../../introduction/lessons/plot_01_rdw.rst) or [soliton lesson](../../introduction/lessons/plot_02_soliton.rst). [Polarisation](polarisation.md) extends the launch to vector fields; [beam setup](../free-space/beams.md) adds a transverse profile.
