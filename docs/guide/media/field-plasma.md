# Carrier-resolved plasma response

Ionisation creates free electrons that change the optical phase and removes the energy required to free them. A carrier-resolved calculation evaluates this history from the instantaneous electric field. Choose the [ionisation rate](ionisation.md) separately from the current model; an envelope calculation instead uses the [cycle-averaged response](envelope-plasma.md).

## The plasma response

A field strong enough to tear electrons off the gas leaves behind a plasma, and the
plasma answers the field in two ways at once. Both are carried by
{class}`solprop.nonlinear.responses.PlasmaResponse`, following Geissler *et al.*, *Phys.
Rev. Lett.* **83**, 2930 (1999).
The [envelope plasma chapter](envelope-plasma.md) derives the cycle-averaged
version and discusses why the ionisation rate and the current closure must be
chosen together.

## Two currents

Write `f(t)` for the fraction of atoms ionised by time `t` and `W(|E|)` for the
ionisation rate of [ionisation.md](ionisation.md). The currents below are per
initial atom; multiplying by the available density $\rho$ gives SI current
density. The **free-electron current** is the
current of electrons already freed and still being accelerated:

$$
J_\text{free}(t) = \frac{e^2}{m_e}\int_{-\infty}^{t} f(t')\,\mathbf{E}(t')\,\mathrm{d}t' .
$$

It is what lowers the refractive index — for a fixed `f` it is exactly the Drude
polarisation $P = -\varepsilon_0(\omega_p^2/\omega^2)E$ with
$\omega_p^2 = \rho f e^2/(\varepsilon_0 m_e)$, which the test suite checks — and hence what
blue-shifts a pulse as the plasma builds under it.

The **ionisation loss** is the energy the field pays to free each electron.
Requiring that work and choosing an isotropic current parallel to the real
electric field gives

$$
\mathbf{J}_\text{loss}(t) = I_p\,W\,(1-f)\,\frac{\mathbf{E}}{|\mathbf{E}|^2}
\qquad\Longrightarrow\qquad
\mathbf{E}\cdot\mathbf{J}_\text{loss} = I_p W (1-f) = I_p \frac{\mathrm{d}f}{\mathrm{d}t},
$$

so the work done per atom over a pulse is exactly `Ip` times the electrons it freed. It
is added pointwise rather than integrated, and the polarisation is the integral of the
sum, `P = ρ ∫ (J_free + J_loss) dt`. Three cumulative integrals in all, which is why the
response declares its domain as `memory`: the value at one sample depends on every
earlier sample. How the integrals are formed is the next section's subject; under the
plain trapezoid the value depends on none of the later samples, and under the band rule
on at most `3K` of them.

For one polarisation component the loss reduces to the familiar scalar form
`Ip W (1-f)/E`, sign and all. For two it is that magnitude directed along the field, so a linearly
polarised pulse gives the scalar answer rotated — which is what fixes the `|E|²` in the
denominator rather than an `|E|`, and what the vector tests assert at four angles.

## Holding the post-pulse current

The production field-grid source holds the current after the pulse. With $g$ the
analytic temporal Planck collar, $q=i\omega$ in the internal FFT convention, and
$J=\rho(J_\mathrm{free}+J_\mathrm{loss})$ summed over every channel, it is

$$
A_J=\frac{\operatorname{DFT}[gJ]}{q}
    -\frac{\operatorname{DFT}[g'J]}{q^2}.
$$

Both terms are evaluated only on the positive active bins. All other bins are
exactly zero. Integration by parts gives $q^2 A_J=\operatorname{DFT}[g\dot J]$
in the continuous, resolved limit: the collar does not artificially turn a steady
current off. This removes the artificial edge from a residual polarisation too.
The current is still built from the same population and free-current integrals;
propagation does not run a third integral to form the plasma polarisation.
Kerr and Raman retain their ordinary collared polarisation. The derivation,
finite-window terms and validation requirements are in
[SEP 27](../../dev/seps/archive/sep-0027-held-plasma-current.md).

Translation certifies the steady-current identity once per grid:
$\|\operatorname{DFT}[g]-\operatorname{DFT}[g']/q\|_A /
\|\operatorname{DFT}[g]\|_{\rm full}\le10^{-10}$.
The denominator includes DC and every fine-grid frequency. An unresolved collar
is refused with its leading/trailing sample counts; widen the temporal window or
tapers. The certificate, limit and counts are saved in `plasma_source`. This
certificate tests a steady current only. It does not bound arbitrary product
aliasing or justify a pulse overlapping a taper; compare wider windows and finer
sampling when those effects matter.

The source needs no option: `plasma=True` on the capillary interface, for example,
selects it. A direct
{meth}`~solprop.nonlinear.responses.PlasmaResponse.evaluate` call still returns the
physical, ungated $P=\int J\,dt$, including any residual drift. Gated-source
comparisons use a private NumPy benchmark hook with a distinct, non-resumable
configuration; it is not a propagation option.

## How the current integrals are formed

The exposure $\int W\,\mathrm{d}t$, the free current $\int fE\,\mathrm{d}t$ and the
physical-response polarisation $\int (J_\text{free}+J_\text{loss})\,\mathrm{d}t$ are running sums on the
fine time grid. The plain cumulative trapezoid integrates a Fourier component
$e^{i\omega t}$ with $\rho(\theta) = (\theta/2)\cot(\theta/2)$ times the exact
$-i/\omega$, where $\theta = \omega\,\delta t_\text{fine}$. The default grid puts the top
of the active band, $1.1\,\omega_\text{max}$, at $\theta_b = 1.15$. There one integral is
11 % short. The physical polarisation returned by `PlasmaResponse.evaluate` has two
current integrals, so its Drude multiplier is $\rho^2$, about 21 % short
([accuracy.md](../../dev/accuracy.md) §1). The production held source integrates the
free current once and divides by $i\omega$ in frequency; its interior Drude
multiplier is $\rho$. Collar and finite-window errors have separate budgets.
{class}`~solprop.nonlinear.responses.PlasmaResponse`'s
`integration` chooses between two rules
([SEP 18](../../dev/seps/archive/sep-0018-band-exact-plasma-integrals.md)).

**The band rule** (`integration="band"`) filters each integrand with symmetric taps
before the trapezoid:

$$
\tilde y_n = c_0 y_n + \sum_{k=1}^{K} c_k\,(y_{n-k} + y_{n+k}),
\qquad
H(\theta) = \Big(c_0 + 2\sum_{k} c_k\cos k\theta\Big)\,\rho(\theta).
$$

The taps belong to the grid.
{func}`~solprop.translate.plasma_integration.design_running_integral` fits them in
translation and certifies the result.

- **Design.** It fits by least squares against $1/\rho$ on $[0, \theta_b]$, with
  $C(0) = 1$, and takes the smallest $K \le 8$ for which
  $|H - 1| \le 10^{-5}$ across the band and $|H| \le 1.05$ above it. The default grid
  takes $K = 4$, nine taps.
- **Certification.** The maxima are found on points the fit never saw and refined by a
  bounded search.
- **Refusal.** Below an oversampling factor of about 3.55 no design reaches the
  tolerance, and translation refuses the rule by name. The message offers the two ways
  out: a finer `dt`, or `integration="trapezoid"`.

The filter reads $K$ samples past each end of the window. Those samples are supplied by
point reflection, $y_{-m} = 2y_0 - y_m$, which is exact for constant and linear
integrands, the residual free current after the pulse included. Its parts:

- **The interior.** Away from the ends, the filter's transfer function meets its certified
  tolerance for Fourier components within the design band. This is not a bound
  on arbitrary unresolved integrands. For a prescribed constant population, the
  held source's interior Drude multiplier is $H$, within $\varepsilon$ of one.
  The physical polarisation's multiplier is $H^2$, within
  $2\varepsilon + \varepsilon^2$ of one; the existing tests at 0.5, 1.0 and
  1.1 $\omega_\text{max}$ compare that polarisation with 0.95, 0.82 and 0.79
  of the susceptibility under the trapezoid. These interior statements exclude
  the collar terms and leading-edge truncation error.
- **The ends.** An abruptly truncated oscillatory integrand can introduce a
  boundary error that the interior multiplier tolerance does not describe.
  The finite-window tone tests use an error allowance equal to the tolerance
  plus $\kappa\lambda$. Here $\lambda$ is the integrand's level in the first
  $2K$ samples relative to its peak; $\kappa \approx 30$ was measured for the
  tested 4096-sample window at the band top. It grows with the window length in
  carrier periods and is not a universal bound for an arbitrary truncated pulse.
  A well-contained pulse should make the integrands negligible at the leading
  edge; inspect the recorded boundary level, including any free-current
  contribution in a pre-ionised gas. The test construction and measurements are
  retained in [SEP 18](../../dev/seps/archive/sep-0018-band-exact-plasma-integrals.md).
- **The exposure.** The filtered rate can dip below zero beside a spike sharper than a
  sample, so the exposure is floored at zero, pointwise. The fraction stays in
  $[f_0, 1)$ and is non-decreasing to within that undershoot. The floor acts on
  the accumulated exposure rather than clipping every local rate increment. It still modifies the computed population where
  active, so inspect the undershoot and refine the sampling.
- **The loss current.** It keeps the raw rate, so $E\cdot J_\text{loss} = I_p W(1-f)$
  holds sample by sample. The work it pays against the electrons it frees differs at
  second order in $\delta t$, by $10^{-8}$ of the electrons on resolved fields.
- **Causality.** Each filtered integral reads $K$ samples ahead. The physical
  current depends on samples up to $2K$ later; a direct physical-polarisation
  evaluation adds a third integral and reaches $3K$: about 600 as on the default
  90 nm grid. This is a band-limited approximation to the causal response. The
  production held source is then assembled from the whole finite window in
  frequency, with the continuation described above.

A run on the band rule records three statistics at each save plane:
`plasma_undershoot`, `plasma_work_defect` and `plasma_boundary_level`
([statistics](../results/statistics.md)). They are recorded for inspection, not acted on: on the
worked examples an undershoot of up to about 0.1, where the strongest node's rate rises
within a sample, left every printed result unchanged against a grid with four times the
oversampling, so no threshold on them separates an accurate run from an inaccurate one.

Two qualification studies against an independent reference, a Richardson extrapolation
of the trapezoid at oversampling factors 48 and 96, are committed as convergence
records. For the physical polarisation of five ionising fields, the largest
asserted-band error is about $6.5\times10^{-6}$. For the plasma-induced change
over ten propagation steps with the held source, it is about $3.9\times10^{-6}$,
against 0.98–6.9 % for the trapezoid in the same bands. These studies assert
bands holding at least $10^{-6}$ of the active response's spectral energy.
See [accuracy.md](../../dev/accuracy.md) §1 for their protocols and reference
uncertainties.

**The trapezoid** (`integration="trapezoid"`) is the plain rule, with no filter, no
reflection and no floor. Luna integrates the same way, and comparisons with Luna use it
together with the private gated benchmark hook ([luna.md](../workflows/luna.md)).

The band rule is the default. A propagation's drivers take `plasma_integration=` and
hand it to every field plasma response. The plasma channels
of one problem share their current integrals, so they must all use one rule.

A direct call to {meth}`~solprop.nonlinear.responses.PlasmaResponse.evaluate` or
`ionised_fraction` under the band rule is handed the grid's rule as `integral`. The
response never fits taps itself and never looks at the field's spectrum.

## Where the pre-ionised fraction goes

Each atom survives to time `t` with probability `exp(-∫W dt)`, so

$$
f = 1 - (1 - f_0)\,\mathrm{e}^{-\int W\,\mathrm{d}t},
$$

with the initial neutral fraction `1 − f₀` multiplying the survival exponential. This
is the only form that satisfies the population equation `df/dt = W(1-f)` and stays
bounded by one. The tempting alternative, adding the pre-ionised fraction outside the
exponential, `f₀ + 1 - exp(-∫W dt)`, starts at the same `f₀` but does three things wrong:
`f` exceeds one, `1 - f` goes negative, and — the reason it matters here — `df/dt` is no
longer `W(1-f)`, so the loss term above stops paying exactly one ionisation potential per
electron. Writing $H=\int W\,dt$, its excess population is $f_0(1-e^{-H})$, which
approaches $f_0$ at saturation.

The same `f` is used everywhere: the ionised fraction and electron density a run reports
({meth}`solprop.nonlinear.responses.PlasmaResponse.ionised_fraction`) are the ones the
polarisation sees, pre-ionised fraction included.

## A pre-ionised fraction along the fibre

`preionisation` is a number or a path along `z` ({class}`solprop.profiles.ZProfile`): a
path gives, at each position, the fraction ionised **before the pulse
arrives there** -- by an earlier pulse, a discharge, a pump beam -- and the pulse then
ionises the remaining neutrals exactly as above, with `f₀ = f₀(z)`. Nothing about how
those electrons got there, or what they are doing, is modelled: pre-ionisation is a
convenient initial condition, not a well-founded physical model of a pre-existing plasma,
and a path generalises the number without changing that. Concretely:

- **The electrons are cold, collisionless and at rest at the start of the window.** On
  a field grid the free current is integrated from zero at the first sample, so the
  pre-existing population starts with no velocity; on an envelope grid it is the
  frozen-cycle Drude drive of the same population. There is no heating, no collision
  or damping term and no recombination: the fraction only grows.
- **It is uniform across the mode at each `z`.** One fraction per species and plane is
  applied at every transverse quadrature node (or through the effective area of a
  mode-averaged run). There is no transverse plasma profile, although a plasma left by
  an earlier pulse is concentrated on the axis.
- **Its linear dispersion is not in the mode index.** The pre-existing plasma acts
  through the time-domain current of this response, not through the effective index
  the mode solver computes, so the design and phase-matching calculators of
  {mod}`solprop.tools` do not see it.
- **It is correct only when the field vanishes at the window edges**: the electrons are
  taken to be at rest where the window begins, and the envelope's Drude factor
  `-1/omega**2` is applied on a periodic transform.

A nonzero preionisation is accepted without a warning; these limitations are the
caller's to weigh.

The response, its components (`Component.preionisation`) and the interface
(`preionisation=` on `prop_capillary`, `prop_waveguide` and the `Simulation`
constructors, globally or per species label) all take the path; NumPy scalars and
zero-dimensional arrays are taken as the float they hold. The interface refuses, by
label, a preionisation (a path or a nonzero number) declared for a species whose
plasma is off, since no channel would carry it. A path is validated in `[0, 1]` at its
breakpoints and at evenly spaced probes when it is declared, and its compiled table is
checked again, interval by interval, at its endpoints and stationary points, so a table
that leaves the interval is refused before a run. Neither check sees a feature
narrower than the table resolves between its samples — a spike to 1.4 a fifth of a
micrometre wide in a `FromCallable` path is tabulated as the constant around it — so
such a feature must be declared as breakpoints of the path. The compiled problem
carries a per-channel table of the fraction, which every backend evaluates at each
stage. A constant path, including one with breakpoints whose values do not change,
compiles to the same table as the number, so the compiled problem is the same.

**Steps land on its kinks.** A step lands on the hard kinks of the cumulative linear
operator, and those are the mode set's breakpoints (see
[the linear operator](../numerics/linear-operator.md)). A pre-ionisation ramp
between two positions has kinks the modes know nothing about, and a step straddling
one integrates a right-hand side that is not smooth there, which costs the integrator
its order. The translation therefore adds a response path's breakpoints -- the
pre-ionisation of every plasma channel, and an explicit density path given to
{func}`solprop.translate.problem.assemble` -- to the hard kinks of `Phi`: a constant
operator is split there into exact degree-one pieces and keeps its constant `D`, so
its flow is unchanged to the last bit, and an expansion or table takes them as further
mandatory edges. The uncoupled modal route does this; a route that cannot (a coherent
wall's coupled flow, the radial and Cartesian free-space routes) refuses a path with
an interior kink by its name rather than stepping across it.

The reference evaluation sees one plane at a time, so
{meth}`~solprop.nonlinear.responses.PlasmaResponse.evaluate` refuses a response that
carries a path: the oracle for a plane is `dataclasses.replace` with that plane's
value, which rebuilds and revalidates the response.

## The guard that travels with the clipping

`J_loss` divides by the field. It is finite only because the rate is **exactly zero**
below threshold — which is why {mod}`solprop.nonlinear.ionisation` guarantees that, why
{class}`solprop.translate.ionisation.RateTable` keeps guaranteeing it below `field_min`,
and why the response refuses at construction any rate that is not zero at zero field. A
clipped-but-non-zero table would divide by a vanishing field at every carrier crossing.
The division is written with a `where` mask rather than a branch, so it never happens at
a sample where the field or the rate vanishes, and a pulse's exact zeros cost nothing.

## Field grid only

The response declares `grids=("field",)` and refuses anything else. The model is written
for the resolved carrier: `W(|E|)` is quasistatic and the loss term divides by the
instantaneous field, neither of which survives being handed a complex envelope: run on
complex data, the same arithmetic would return a quantity nobody has derived. The
separate
{py:class}`~solprop.nonlinear.envelope_plasma.EnvelopePlasmaResponse` implements the
derived cycle moments, depletion and frequency-domain current assembly described in
[the ionisation manual](ionisation.md#envelope-plasma-sources).

```{eval-rst}
.. plot::
   :filename-prefix: media-field-plasma
   :alt: Argon ionisation rises at carrier-field peaks; comparison of polarisation with zero and five per cent pre-ionisation.
   :caption: Argon at one bar in a 10 fs pulse at 5e10 V/m. Left: the field and the fraction it ionises, which rises in a staircase, one step per half-cycle where the field peaks, and then stays. Right: the polarisation with no pre-ionisation and with five per cent of the gas already ionised. The first is zero until the pulse is strong enough and then keeps oscillating after the pulse has gone -- free electrons the pulse left behind; the second responds from the first sample, which is the Drude polarisation of a plasma that was already there.

   import matplotlib.pyplot as plt
   import numpy as np

   from solprop.nonlinear.optmat_rates import OptmatRate
   from solprop.nonlinear.responses import PlasmaResponse
   from solprop.physdata import Gas, density

   t = np.linspace(-25e-15, 25e-15, 4097)
   dt = float(t[1] - t[0])
   field = 5e10 * np.exp(-((t / 10e-15) ** 2)) * np.cos(2.35e15 * t)

   rho = density(Gas.Ar, 1.0)
   rate = OptmatRate(Gas.Ar, 800e-9)
   # Bare samples carry no grid to design the band rule from; on samples this fine
   # the trapezoid's error is far below the line width.
   plasma = PlasmaResponse(Gas.Ar, rate, integration="trapezoid")
   fraction = plasma.ionised_fraction(field[None, :], dt=dt, kind="field")

   fig, (ax_field, ax_p) = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained")

   ax_field.plot(1e15 * t, 1e-10 * field, lw=0.7, color="0.6")
   ax_field.set_xlabel("time (fs)")
   ax_field.set_ylabel(r"field ($10^{10}$ V/m)", color="0.5")
   twin = ax_field.twinx()
   twin.plot(1e15 * t, 100 * fraction, lw=1.6, color="C3")
   twin.set_ylabel("ionised (%)", color="C3")

   for index, before in enumerate((0.0, 0.05)):
       response = PlasmaResponse(
           Gas.Ar, rate, preionisation=before, integration="trapezoid"
       )
       out = response.evaluate(field[None, :], rho, dt=dt, kind="field")[0]
       ax_p.plot(1e15 * t, 1e6 * out, lw=1.4, color=f"C{index}",
                 label=f"pre-ionised {100 * before:.0f}%")
   ax_p.set_xlabel("time (fs)")
   ax_p.set_ylabel(r"$P$  ($\mu$C/m$^2$)")
   ax_p.legend(frameon=False, loc="upper left")
```
