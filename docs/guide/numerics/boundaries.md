# Absorbing boundaries

Both axes of the grid are finite and periodic, because both are sampled for FFTs. Energy
that reaches the edge of the frequency window or the edge of the time window therefore does
not leave the simulation: it wraps around and aliases onto the rest of the field. Absorbing boundaries attenuate light near the edges to reduce these artefacts.
Their finite strength and width must be checked for the actual calculation.

They matter whenever the physics genuinely reaches the edge of a window — soliton
self-compression, modulation instability, a dispersive wave walking out of the time window
at a hundred femtoseconds per metre. The rest of the time they should do nothing at all,
and saying so is part of their job.

This page covers {py:mod}`solprop.boundaries`: why the absorber is a rate rather than a
mask, how its spectral and temporal parts are applied, what it costs, and how to tell when
it is too weak. It is for anyone whose pulse, or whose spectrum, reaches the edge of its
window, and for anyone who wants to know what the collars do to a run that never gets
there. The collars it works from are built by {py:mod}`solprop.grid` and described in
[grids and Fourier conventions](../setup/grids.md#the-active-band-and-absorbing-collars);
why the absorber is a rate, and where its temporal part is applied, are also recorded
among the [design decisions](../../dev/decisions.md). Which runs carry an absorber, and how
to change it, is [near the end](#which-runs-carry-one-and-how-to-change-it). Where solprop's absorbing boundaries differ from
Luna.jl's, the differences are listed in [Noteworthy differences from Luna](../workflows/luna.md).

## Rates, not masks

The simplest absorbing boundary multiplies the solution by a fixed collar profile `W`
after every accepted step. The cumulative absorption is then `Wⁿ`, where `n` is however
many steps the adaptive controller happened to take — a number derived from `rtol`. Two
things follow, and both are wrong:

- **the answer depends on the tolerance.** Tightening `rtol` does not approach a limit, it
  increases the absorption. Refining sixteen-fold erodes the spectral wings fifty-fold.
- **the taper is not a taper.** At realistic step counts `Wⁿ` is a brick wall at the edge
  of the flat region, and that brick wall is the limit the scheme approaches. It erodes
  genuine spectral wings.

An absorbing boundary must instead be an absorption **coefficient per unit propagation
distance**, so that traversing a distance `L` attenuates by a factor that depends only on
`L` and not on how the solver chose to subdivide it. solprop converts each collar profile
into one. The spectral collar becomes

$$ \alpha(\omega) = -\frac{2 \log W(\omega)}{\ell}, $$

the rate that reproduces one multiplication by the window over the reference length `ℓ`;
the temporal collar becomes the bounded profile

$$ \alpha(t) = \frac{A\,(1 - W(t))}{\ell}, \qquad A = 12.6, $$

the complement of its window ({py:data}`solprop.boundaries.TEMPORAL_ALPHA_LENGTH`). Why
the two differ is [below](#why-the-temporal-rate-is-not-the-logarithm).

## The factor of two, and where it goes

`α` is a **power** absorption coefficient, as everywhere else in solprop: power falls as
`exp(−αz)`, the field as `exp(−αz/2)`, and the linear operator carries `−α/2`
([the conventions](../../dev/conventions.md) §2). The factor of two in the spectral definition
is exactly what makes

$$ \mathrm{e}^{-\alpha \ell / 2} = W, $$

so that propagating through one reference length `ℓ` applies the collar profile to the
*field* precisely once. The reference length is
`ℓ = length / boundary_N` by default: "apply the window `boundary_N` times over the whole
propagation", with `boundary_N = 20`. The strength of the absorber is then a physical
distance rather than a step count.

Where each part goes is decided by what can be applied *exactly*:

| part | mechanism | why |
|---|---|---|
| `α(ω)` | folded into the linear operator as `−α/2` before the cumulative operator is integrated | diagonal in ω, so the propagator applies it exactly over any sub-interval, at no cost |
| `α(t)` | the term `−α(t)E(t)/2` added inside every stage evaluation, masked to the active band | diagonal in t, so it cannot ride the propagator, and it belongs in the derivative so that the scheme integrates it at its own order |
| hard band limit | a 0/1 mask outside the active band | idempotent, and part of the definition of the band rather than an absorber |

The hard limit sits outside the **active mask** `ω_min/2 < ω < 1.1 ω_max`, following
[the conventions](../../dev/conventions.md) §3. Putting it at the in-band
mask instead would erase the collar the graded absorber acts on.

## The two temporal schemes

The temporal rate cannot be folded into the cumulative operator, and it cannot share the
nonlinear polarisation's transform either, because the UPPE prefactor is applied after that
transform and the absorber is a rate on the *field*, not a polarisation. Two schemes
represent it as a rate:

- {py:func}`solprop.boundaries.nonlinear_absorber_term` — add `−α(t)E(t)/2` inside every
  stage, with its own modal transform, masked to the active band. **This is the default on
  the modal routes** (capillary, waveguide and GNLSE).
- {py:func}`solprop.boundaries.apply_split_step` — multiply the modal time-domain field by
  `exp(−α(t)h/2)` once per accepted step, followed by the hard band limit, because the
  product in time spreads content into every bin (on a field grid, chiefly the DC). One
  batched transform pair per step; the factors telescope, so the total is a rate
  whatever the step layout. Selectable as `absorber_scheme="split_step"`, and **the default
  on the spatial routes** (radial, free-space and planar) unless their planes are
  interpolated ([below](#which-runs-carry-one-and-how-to-change-it)).

**Why the term and not the factor.** The split factor is exact for the absorber alone, but
applying it once per step is a *Lie splitting* of a term that does not commute with the
linear flow — and a Lie splitting is first order. Measured on an N = 3 soliton launched
into its collar, at 100, 200 and 400 fixed steps against the same problem at 1600, with
the bounded temporal profile solprop uses:

| scheme | relative error | observed orders |
|---|---|---|
| `split_step` | 1.7e-3 · 8.6e-4 · 4.3e-4 | 1.00 · 1.00 |
| `nonlinear` | 3.5e-9 · 2.2e-10 · 7.1e-12 | 4.02 · 4.95 |

So the split factor caps a fifth-order scheme at first order wherever the collar removes
anything, and makes the absorbed energy a function of `rtol` — the very defect the rate
form exists to avoid, one level up. The conclusion does not depend on the temporal
profile. With the capped logarithm as the temporal rate ([below](#why-the-temporal-rate-is-not-the-logarithm))
the same ladder reads 7.657e-4 · 3.845e-4 · 1.912e-4 (orders 0.99 · 1.01) against
2.939e-5 · 1.278e-6 · 1.789e-8 (4.52 · 6.16), and quadrupling the steps moves the energy
the split factor removes by 6.5e-5 of itself and the in-stage term's by 9.0e-10. The
in-stage term's errors are four orders smaller under the bounded profile because the
term it integrates is not stiff there.

**What the term costs.** One extra modal transform per stage, and one thing that needs
watching: it is **conditionally stable**. The collar's largest rate becomes an eigenvalue of the linearised problem, so an
explicit method needs `h·α/2` inside its stability region — 3.3 for DOPRI5.
{py:func}`solprop.boundaries.stable_step` returns the limit, and translation *refuses*
a problem whose replayed schedule or step floor is already past it, with a
`TranslationError` naming the limit, the step and the remedies, rather than letting it
diverge. A replay cannot reject a step: one that diverges for another reason stops
with a `ConvergenceError` at the first step whose error norm is not finite, naming the
step and the position, rather than writing NaN planes.

```{code-block} python
from solprop.boundaries import absorber, stable_step

boundary = absorber(grid)                 # ell = length / 20 by default
stable_step(boundary.temporal_rate)       # the longest step that will not blow up
```

An adaptive run finds the limit for itself — the rate reaches its ceiling only at the very
edge of the window, where the field is negligible — so it costs steps and not correctness,
and with the bounded profile only on runs shorter than about `2 · boundary_N` steps.
With the capped logarithm as the temporal rate it would cost steps everywhere: a ceiling
of `60/ℓ` on the collar's last samples reaches the error estimate long before the
stability limit, and on a reduced four-mode resonant-dispersive-wave capillary case costs
40 % more steps than the split factor, where the bounded profile takes the split factor's
count, 456 against 459. The split factor's unconditional stability is the one thing it is
better at, and why it stays selectable.

The two agree to first order in the step and differ at second, by `(αh/2)²/2` — which is
what the tests assert, and what would be false if the factor of one half were dropped.

## Geometry, and the collar that is not one

The collar geometry comes from the grid: the spectral collar rises over `[ω_min/2, ω_min]`
and falls over `[ω_max, 1.1 ω_max]`; the temporal collar is symmetric about zero and lies
**outside** the requested window. The grid widens the realised window to at least
`trange / (1 − 2f)`, with `collar_fraction` `f = 0.1` by default, so the whole request is
flat and the collar takes a tenth of the realised window beyond it at each end
([grids](../setup/grids.md#the-active-band-and-absorbing-collars)). A collar inside the
request would cut into light the run is meant to keep; the SEP 21 campaign measured that
on a full-length Raman-soliton run, where it cost two orders of magnitude in accuracy.

Two settings change the geometry. `collar_fraction=0` leaves no room for a temporal collar
and realises the request alone, for a grid built by hand for a run without one;
`absorber=False` keeps the room (below).
`collar_placement="inside"` rebuilds the geometry solprop used before SEP 21: the window is
the request rounded up, and the collar takes at least `collar_fraction` of it, inside the
request when the rounding leaves too little slack. It exists to reproduce frozen
comparisons and stored witnesses bin for bin.

A collar made only of the slack between the requested and the realised window would
vanish whenever the transform size made the realised window equal to the requested one;
the minimum fraction is what prevents that.
{py:func}`solprop.boundaries.absorber` refuses a collar whose edge spans fewer than four
grid samples: below that the discrete profile steps rather than tapers, the rate jumps to
its cap in one sample, and the boundary is the brick wall the rate form exists to avoid.

The spectral cap, `α·ℓ ≤ 60`, has two jobs. The collar reaches exactly zero at the band edge
and `−2 log 0` is an infinity, which would put a NaN into the operator; and in the
interaction picture the right-hand side is back-propagated by `exp(+αΔz/2)`, which must
stay finite. At 60 the field is attenuated by `exp(−30)`, about `10⁻¹³`, over one reference
length — far below anything a propagation can represent. The temporal profile needs no cap:
it is bounded by construction at `12.6/ℓ`, and every rate is still validated against the
ceiling of 60.

## Why the temporal rate is not the logarithm

The obvious temporal rate is the same capped logarithm as the spectral one. That is the
right rate for a window that rides the linear operator for free, and the wrong one for a
term in every stage: the Planck window falls so steeply that `−2 log W` is above 12 over
the outer 14 % of each edge, which carries 73 % of the edge's integrated attenuation, and
the cap of 60 that keeps `−2 log 0` finite becomes the largest eigenvalue of the
linearised problem — the stability limit of the in-stage term and, through the error
estimate, its step count. Lowering the cap alone halves the attenuation, because so much
of it sits under the cap.

The profile `A(1 − W)/ℓ` keeps the integral — `A = 12.6` reproduces the capped logarithm's
6.3 per collar width per reference length to a few per cent — with a largest rate a fifth
of the logarithm's, every derivative continuous, and the attenuation spread over the edge
instead of piled at its end. Measured on the reduced resonant-dispersive-wave case above:
456 steps against 637, and the in-window field differs between the two profiles by 3e-3
in the fundamental and 2e-3 in the second mode, on a case that carries 6 % of its
fundamental in the collar. A case that holds a twentieth of its *input* field in the
collar — a reduced hydrogen-filled capillary supercontinuum case — changes in kind
between the two profiles: when a run's physics happens inside the absorber, the result
depends on the absorber, and the time window has to be widened.

## Walk-off

An absorber doing its job is silent, so {py:func}`solprop.boundaries.warn_walkoff` says
when it stops being strong enough. A packet whose group delay drifts by `Δτ` over the whole
propagation crosses the temporal collar at `Δτ/L` seconds per metre, so it spends
`dt/(Δτ/L)` metres at each sample and accumulates

$$ \frac{L}{\Delta\tau}\,\frac{1}{2}\int \alpha(t)\,\mathrm{d}t $$

nepers of field attenuation on the way through. Below `log(10³)` — the field down by a
thousand, the power by sixty decibels — the packet reaches the edge of the window still
alive, wraps around, and appears at the *front* of the time window as an alias.

The group-delay span is the caller's, deliberately. A prediction from the group delay
across the whole simulation band, which on a grid running to four microns is dominated by
band edges carrying nothing but shot noise, fires on almost every run — and a warning that
is usually wrong is one people learn to ignore. A caller who passes the span of the band
it cares about does not have that problem.

```{eval-rst}
.. plot::
   :caption: Left: the temporal collar of the RDW field grid and the absorption rate it becomes, for two absorber strengths. Right: the property the whole scheme exists for -- the surviving field after 0.3 m is the same however that distance is subdivided, while a per-step multiply gives a different answer for every step count.

   import numpy as np
   import matplotlib.pyplot as plt

   from solprop.boundaries import absorber, split_step_factor
   from solprop.grid import FieldGrid

   grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)

   fig, (ax_rate, ax_steps) = plt.subplots(1, 2, figsize=(9.5, 3.6))

   # Only the trailing collar, which runs from the edge of the 400 fs request at 200 fs
   # to the edge of the realised window near 256 fs: on the whole axis the flat request
   # would take most of the width.
   twin = ax_rate.twinx()
   ax_rate.plot(grid.t * 1e15, grid.t_window, color="0.5", lw=1.2)
   ax_rate.set_ylabel("collar profile", color="0.5")
   ax_rate.set_ylim(-0.02, 1.05)
   for index, strength in enumerate((20, 60)):
       rate = absorber(grid, boundary_N=strength).temporal_rate
       twin.semilogy(
           grid.t * 1e15, rate, color=f"C{index}", label=f"boundary_N = {strength}"
       )
   twin.set_ylabel("absorption rate (1/m)")
   twin.set_ylim(1e-2, 1e4)
   twin.legend(frameon=False, loc="center right", fontsize="small")
   ax_rate.set_xlabel("time (fs)")
   ax_rate.set_xlim(190, 260)

   # A point part way down the taper, where the collar profile is 0.9. As a rate the
   # surviving field is exp(-alpha L / 2) whatever the step layout; as a per-step
   # multiply it is 0.9^n, which is a different answer for every step count.
   boundary = absorber(grid)
   profile = 0.9
   alpha = -2.0 * np.log(profile) / boundary.reference_length
   distance = 0.3
   counts = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
   as_a_rate = [
       float(np.prod(split_step_factor(np.array([alpha]), distance / n) ** n))
       for n in counts
   ]
   ax_steps.loglog(counts, as_a_rate, "o-", label="as a rate")
   ax_steps.loglog(counts, profile**counts, "s--", label="per-step multiply")
   ax_steps.set_ylim(1e-20, 3.0)
   ax_steps.set_xlabel("steps taken over 0.3 m")
   ax_steps.set_ylabel("surviving field amplitude")
   ax_steps.legend(frameon=False)

   fig.tight_layout()
```

## Two lengths: the spectral rate and the in-stage collars

A single length would set three things: the spectral collar's strength, the ceiling
`60/ℓ` every rate is validated against, and the strength of the in-stage temporal collar,
`12.6 (1 − W(t))/ℓ`. (A free-space run's real-space collar is designed separately, from
a protected band and its width:
[transverse boundaries](../free-space/model.md#transverse-boundaries).)
The two kinds of rate are applied very differently. The spectral rate is diagonal in ω
and rides the linear operator, where it is exact and free whatever its strength; the
in-stage collars are explicit terms, and their largest rate bounds the step an explicit
method can take ([the stable step](#the-two-temporal-schemes)). A measurement that needs
a strong spectral absorber should not pay for it in steps.

So {py:class}`~solprop.boundaries.Absorber` carries two lengths
([SEP 22](../../dev/seps/archive/sep-0022-memory-lean-spatial-execution.md) §4.6):

- `reference_length`, `ℓ`, sets the spectral rate and its ceiling `60/ℓ`;
- `collar_length`, `ℓ_c`, sets the in-stage temporal collar — its largest rate is
  `12.6/ℓ_c` — and its ceiling `60/ℓ_c`. `None`, the default, means `reference_length`, and a value equal to the
  reference length is stored as `None`. {py:attr}`~solprop.boundaries.Absorber.in_stage_length` returns
  the length in force.

Take a 1 mm sample whose measurement calls for a spectral power rate of `10⁷ m⁻¹`. The
default `ℓ = 50 µm` caps rates at `1.2 × 10⁶ m⁻¹`, so a single length would have to fall
to 6 µm; the spatial collar's peak would rise from `2.5 × 10⁵` to `2.1 × 10⁶ m⁻¹` and its
DOPRI5 stable step fall from 26 µm to 3.1 µm. With two lengths the spectral rate moves
and the collars do not.

`collar_length` is whatever sets the temporal collar's strength — `boundary_N` or
`boundary_length` today. [SEP 21](../../dev/seps/archive/sep-0021-absorbing-boundaries.md), which
owns the temporal collar, may parameterise that strength by a protected walk-off; the
spatial collar follows the same length until its findings say otherwise.

### A custom spectral absorber

{py:func}`~solprop.boundaries.custom_absorber` adds a power rate of your own to a
collar's spectral rate:

```{code-block} python
from solprop.boundaries import absorber, custom_absorber

boundary = custom_absorber(grid, 1e7)                       # a scalar, 1/m
boundary = custom_absorber(grid, rates, collar=absorber(grid, boundary_N=60))
boundary = custom_absorber(grid, lambda omega: 1e6 * (omega > omega_cut))
```

The rate is a scalar, an array on the grid's frequency axis or a callable of `ω` in
rad/s; it must be finite and non-negative, and it is zeroed outside the collar's band
mask, where the operator is zero anyway. The collar is taken **by value** — `None` is
`absorber(grid)`, and any collar is accepted as it is, a spectral-only one included —
and its temporal rates, band mask and in-stage strength are kept: the result's
`collar_length` is the collar's own `in_stage_length`. Its `reference_length` stays the
collar's when the summed rate fits under the collar's ceiling, and otherwise becomes
`60` over the sum's largest value, the longest length that admits it. The default collar
already reaches its ceiling at the band edges, so any rate added there moves the
reference length; the explicit collars never move. A zero rate returns the collar field
for field, so its runs have the collar's results and configuration hash.

The user's rate is summed into the collar's, so a diagnostic of the spectral collar's
absorbed energy reports the two together.

### A spectral-only collar

`absorber(grid, temporal=False)` builds a collar with the spectral rate and the band
mask of a full collar and **no temporal part**: its temporal rates are zero arrays on both time
axes and its record says `temporal: False`. Any collar whose temporal rates are all zero,
however it was built, crosses the seam as no temporal absorber
({py:func}`~solprop.translate.masks.temporal_rates`), so the kernels compile no temporal
pass under either scheme: no in-stage term, no split pass and its transverse transform
pair, and the first-same-as-last stage recycled. `absorber_scheme` is still recorded and
applies to nothing, and its refusal of interpolated planes does not fire. The grid is
unchanged. Measured on a Cartesian envelope grid of 192 × 192 × 300 with a spectral-only
collar, the pass that compiling away removes cost 1.3–2.3 times a step on `numba_cpu` and
1.2–1.6 times on native CPU
([SEP 22](../../dev/seps/archive/sep-0022-memory-lean-spatial-execution.md) §3; the phase-1
re-measurement is in its
[implementation record](../../dev/seps/archive/sep-0022-memory-lean-spatial-execution.md#phase-1-report--driver-output-and-absorber)). Under the in-stage
scheme the zero rate only added zeros, so a run is unchanged; under the split it differs
by the transform pair's rounding.

A spectral-only collar is passed as `absorber=absorber(grid, temporal=False)`; a keyword
that turns both soft collars off on the fibre entry points is
[SEP 21](../../dev/seps/archive/sep-0021-absorbing-boundaries.md)'s.

### The transverse boundary

The collars above act on time and frequency. A free-space run's transverse boundary is
its own ([SEP 40](../../dev/seps/sep-0040-transverse-absorbing-boundaries.md)): a wall or a
wrap at the grid's outer size, a safety ramp in the transverse **wavenumber** on every
geometry, which rides the linear operator exactly like the spectral rate, the monitor's
transverse edge, and, on request, a designed real-space collar,
`spatial_absorber=SpatialAbsorber(...)`
({py:class}`~solprop.boundaries.SpatialAbsorber`). It is independent of the `Absorber` and
its lengths: [transverse boundaries](../free-space/model.md#transverse-boundaries) and
[grid sizing and the transverse safety ramp](../free-space/cartesian.md#grid-sizing-and-the-transverse-safety-ramp).

## Which runs carry one, and how to change it

| entry point | absorber by default |
|---|---|
| `prop_capillary`, `prop_waveguide`, `prop_gnlse` and their `Simulation.from_*` builders | the grid's default collar, `absorber(grid)`: spectral and temporal, `boundary_N = 20`, applied in every stage |
| `prop_radial`, `prop_free`, `prop_planar` and their builders, whose settings declare none | the same temporal and spectral collar, applied once per accepted step (`absorber_scheme="split_step"`) unless planes are interpolated; the transverse boundary is the safety ramp and the monitor's edge, with a real-space collar only on request (`spatial_absorber`, SEP 40) |
| a `Simulation` built directly with its own `SimulationSettings` | none unless `SimulationSettings(absorber=...)` is given |

Every `prop_*` entry point takes `absorber=` and `absorber_scheme=`. Pass
`absorber(grid, boundary_N=60)` for a stronger collar, `absorber(grid, boundary_length=...)`
to fix the reference length `ℓ` directly, and `absorber=False` for none: both soft collars,
temporal and spectral, are switched off and the hard band mask is kept
({py:func}`solprop.boundaries.no_absorber`). The built grid keeps the collar's room, so a
run without an absorber shares the grid of the run with one, and the Raman response's tail
taper and the held plasma current's gate, which read the collar's width, keep their
samples whatever the transform size rounds to. On the spatial
interfaces `absorber=False` leaves the transverse collars as they are. `absorber=None` means
"build the default", not "no absorber"; a scheme beside `absorber=False`, or a keyword
that contradicts what a spatial run's settings already declare, is refused.

The spatial routes apply the temporal collar by the split scheme unless the settings name
a scheme or interpolate their planes, which the split refuses: on the spatial cases of the
performance-tracking suite the in-stage collar took 1.2 to 4.6 times the steps of the
split one, at equal or worse accuracy inside the requested window
([SEP 21](../../dev/seps/archive/sep-0021-absorbing-boundaries.md)). The split is unconditionally
stable, so a fixed schedule on a spatial route keeps its collar.

On a spatial grid an absorber given explicitly also switches on the real-space transverse
collar ([free-space propagation](../free-space/model.md#spatial-collar-and-statistics)).

## Watching the window's edges

The tutorial [Is my time window big enough?](../../auto_tutorials/basics/plot_07_time_window.rst)
sizes a window with what follows, on a short fibre whose Raman soliton walks towards
the late edge.

Every run watches its time window ({py:class}`solprop.monitor.BoundaryCheck`, on by default
through `SimulationSettings.boundary_check`). At the launch and at every committed step it
measures the share of the field's energy in the outer 5 % of the realised window at each
edge, the guards, and in the collar:

- from the launch on, a guard occupancy whose square root exceeds the field-error budget
  (`1e-3` by default, a millionth of the energy) warns once
  ({py:class}`~solprop.exceptions.BoundaryWarning`), and ten times the budget (a ten
  thousandth) stops the run with {py:class}`~solprop.exceptions.BoundaryError`; at the
  launch that means the window clips the input, which can seed spurious radiation, and
  the run is refused before its first step. Light the launch puts in the collar alone is
  absorbed, and the end of the run reports it;
- at the end, the fraction of the launch energy the temporal collar removed is logged, and
  warns when its square root exceeds the budget. Weak spectral features can matter at far
  smaller fractions than that: the record is there to read.

**How often, and what it saves.** While some guard or the collar holds more than a tenth
of the budget (as a field amplitude), every committed step is checked. While the edges
are quiet, a step is checked every eighth step or after `length / 256` of the guide,
whichever comes first, and the state of every saved plane is checked regardless. The
collar itself stays in every stage.

**The quiet collar, on request.** `BoundaryCheck(quiet_collar=True)` also leaves the
in-stage temporal collar's term, whose transform is a stage's largest cost after the
response's, out of the stages while the collar is quiet: while it held less than `1e-12`
of the field's energy, the shot-noise floor included, at two checks in a row. A check
that finds light in it puts the term back at once. While the term is out no step is
longer than `length / 256`, so light cannot cross into the collar unabsorbed within one
long step, and the exponential Adams march no longer limits its order by the collar's
stability; the first step after the term returns is held to that limit. Under input
noise the floor alone fills the collar far above the threshold, so the option only acts
on noise-free runs; on a 2 cm noise-free GNLSE Raman/shock run whose pulse stays clear it
saved 6 to 9 % per step on `numba_cpu` and `native` (Apple M5 Max). It is off by default
because a run cannot know in advance whether its light stays clear, and an omission
judged too early is an error nothing reports (ADR-057). It applies to binary64 modal and
radial stages with the in-stage scheme; the split scheme, the low-storage march, the
general modal route, coupled flows, a real-space collar, single precision, and the
Cartesian, line and planar routes, whose stages keep the term regardless, always keep
the collar. The record's `collar_left_out_checks` counts the checks that found it out.

**Above the shot-noise floor.** Input noise, on by default, puts one photon per mode across
the whole window, guards and collar included. Light below that floor cannot be told from
the vacuum, so the monitor judges what lies above it: from each region's energy it
subtracts three times the floor's expected share (the floor is uniform in time on
average, and three times its expectation clears its fluctuation in a narrow guard), and
from the collar's removal the floor's expected removal, up to the floor itself: the
input noise puts one floor in at the launch and never adds to it, so the floor cannot
lose more than that, however long the run. A weak launch is therefore not stopped for its
own noise, while light well above the floor that reaches an edge still is. Light that
arrives at an edge, or leaves through the collar, weaker than the vacuum's own share of
it is not reported.

What the monitor saw is kept in the run's execution record, under `boundary_monitor`: the
launch occupancies, the largest guard and collar occupancies and where the guards peaked,
and the removed fraction, all above the floor; beside them the floor's share of the
launch energy (`noise_floor`) and the whole field's largest guard share and removed
fraction, floor included (`raw_guard_max`, `raw_removed_fraction`). Under `planes` it keeps
one row per saved plane: `z`, the collar occupancy there, each guard's largest occupancy
since the previous plane, and the removed fraction so far. The record is rewritten at every
saved plane without the rows, which the closing record (complete, failed or out of budget)
carries, so a file does not grow with the square of its planes. Beside them are the
thresholds the execution ran under (`guard_fraction`, `budget`, `stop_factor`,
`quiet_collar`) and the launch's energy in the monitor's own units (`launch_energy`).

Under the in-stage scheme the removed fraction is integrated along z by the trapezoid.
Under `absorber_scheme="split_step"` every backend adds up, in its own step finish, what
each accepted step's split takes of the power it finds, `P(1 − e^{−αh})`, so the removal
is exact on every step, checked or not. Read back from the state the split leaves, the
same removal would multiply whatever the band mask left deep in the collar by `e^{αh}`,
into removals larger than the launch and false warnings.

A check reads five numbers of the state, its energy and the four weighted energies above,
and each backend forms them where the state lives: on the GPU for `mlx` and a device build
of `native`, in compiled code for `numba_cpu` and `native` on the CPU, and with NumPy for
`numpy_ref`, the reference the others are tested against. The field is not copied to the
host to be watched.

A resumed run continues the record: the state it resumes from is checked by the first
step, not judged as a launch, so a run stopped at an edge can be resumed under a relaxed
check (the check is not part of the resume key; the quiet-collar opt-in is). The totals,
the launch occupancies, the quiet collar's state and the rows of a closing record are
restored, so the removed fraction and the guard maxima continue against the original
launch, and `resumed_at` lists where the run was resumed. A run killed without a closing
record restores its totals, and its rows start again at the resume. The record continues
from the latest execution that kept one: an execution killed before its monitor started,
or run with `boundary_check=None`, is passed over, and the stretch it ran unwatched is
listed under `unwatched` as `[start, end]` in metres, because what left the window there
is unknown rather than zero. The record also keeps the signed removal (`removed_signed`)
and the floor's expected removal (`floor_removed_fraction`), so that a resume continues
from the same numbers the whole run would hold.

On a free-space run the monitor also watches the **transverse edge**
([SEP 40](../../dev/seps/sep-0040-transverse-absorbing-boundaries.md)): the energy in the
outer 5 % of the grid's outer size above its launch value with no collar, and what
returns with a requested one, at `BoundaryCheck`'s levels or its own `transverse_budget`,
which relaxes the transverse edge without relaxing time. Its reads are spaced as a ceiling
on the step, and its record joins the monitor's under `transverse`, with a
`transverse_status` that only reads `clear` for a run that read its edge at the required
spacing and found nothing ([transverse boundaries](../free-space/model.md#transverse-boundaries)).

`SimulationSettings(boundary_check=None)`, or `boundary_check=None` on any `prop_*`
function or `Simulation.from_*` constructor, watches nothing; a `BoundaryCheck` with a
larger `budget` or `stop_factor` relaxes it.

## A worked example

```{code-block} python
import numpy as np
from solprop.boundaries import absorber, apply_band_mask, apply_split_step, warn_walkoff
from solprop.grid import FieldGrid

grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
boundary = absorber(grid, boundary_N=20)

# The spectral rate goes into the linear operator as -alpha/2. Here the temporal rate
# is applied to the time-domain field once per accepted step: the `split_step` scheme,
# which `absorber_scheme="split_step"` selects (the default applies it in every stage).
field = np.ones((1, 1, grid.n_t))
field = apply_split_step(field, boundary.temporal_rate, step=0.01)

# The hard band limit is a separate, idempotent multiply on the spectrum.
spectrum = apply_band_mask(np.ones((1, 1, grid.n_omega)), boundary.band_mask)

# And the run says so if the absorber is too weak for the walk-off it will produce.
warn_walkoff(grid, group_delay_span=1e-12)   # SolpropWarning: widen trange, or ...
```
