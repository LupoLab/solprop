# SEP 40 — Transverse absorbing boundaries for free-space propagation

| | |
|---|---|
| **SEP** | 40 |
| **Title** | Transverse absorbing boundaries for free-space propagation |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-10-31 |
| **Resolution** | — |
| **Requires** | [SEP 21](archive/sep-0021-absorbing-boundaries.md), [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md), [SEP 34](archive/sep-0034-medium-flux.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Rebuild the absorbing boundaries of the radial, Cartesian, line and planar free-space
routes so that light leaving the transverse window is removed, not reflected or wrapped,
at a cost that does not grow with the absorber. The real-space collar is moved outside the
requested aperture on every geometry. Its strength is set by the transverse wavenumbers
it has to absorb and by its width, not by the propagation length. It is applied once per
accepted step as an exact factor, not inside every stage. The transverse-wavenumber ramp
of SEP 22 is joined to it in one spatial absorber object, and is offered on the radial grid
too. The spatial edges get the same monitor that SEP 21 gave the time window. Success
means the following. A linear diffracting beam that sends most of its energy through the
boundary agrees with an unbounded reference to better than $10^{-3}$ over the requested
aperture, with no more steps than the same run with no absorber. A measured
reflection coefficient against transverse wavenumber agrees with the theory of §4 over
the band the design claims. The suite's self-focusing case reaches its large-radius
reference to the same level. Backward compatibility is not a constraint: no release
depends on the present interface, and every downstream script is updated with it.

### 2. Motivation and scope

**The use cases.** Three groups of free-space runs meet the transverse boundary.

- *Radial filamentation and self-focusing in gases and solids.* The beam diffracts,
  self-focuses, ionises and diffracts again. Conical emission and plasma defocusing send
  light to angles far beyond the launched divergence. The suite's `spatial_absorber`
  case (Kerr only, 0.9 of the critical power, argon at 1 bar, 8 cm, requested radius
  0.3 mm) is the standing witness.
- *Cartesian runs of thin nonlinear plates with non-collinear beams.* An example is
  pump–probe four-wave mixing in a thin silica plate, where the grid is sized by the
  one-half rule ([grid sizing](../../guide/free-space/cartesian.md#grid-sizing-and-the-transverse-safety-ramp))
  and the far-field statistics of SEP 23 are the observable. Products generated at large
  angles must leave the box, not fold back onto the beam.
- *The line and planar routes* (`LineGrid` and the hybrid slab × y geometry), which share
  the Cartesian collar along their periodic axis.

**The problem.** SEP 21's campaign (its finding R4, decision D9) measured the radial
real-space collar on the suite case with temporal absorption off. The collar as shipped
(the outer 10 % of the requested radius, peak rate $252/L$) left a field error of
$1.0\times10^{-1}$ over the requested radius and $2.3\times10^{-2}$ in the core. A hard
Dirichlet wall at the same radius gave $1.0\times10^{-1}$ and $2.8\times10^{-2}$. The error
was reflection from the collar, and it reached the core. A collar outside the requested
radius, 30 % wide and 25 times weaker, was 20 times more accurate over the aperture and
130 times in the core, and took 46 Runge–Kutta steps instead of 241. D9 deferred the
collar's geometry to a follow-up proposal. This is that proposal.

A linear probe on 2026-10-31 extended the finding to the Cartesian route and to a beam
that loses most of its energy through the boundary. A 30 µm Gaussian beam at 800 nm in
argon at 1 bar travelled 2 cm, about six Rayleigh lengths, with a requested radius of
150 µm, at an energy low enough that the Kerr term does nothing. The reference was the
same sampling on a grid of four (radial) or two (Cartesian) times the radius without a
collar. The integrator was `RungeKutta()` at `rtol = 1e-7` on `numba_cpu`. The table
gives the area-weighted field L2 error at $z = L$ over the requested aperture and over
its inner half, and the accepted steps. AN is the collar's peak power rate times $L$.

| route | boundary | aperture | core | steps |
|---|---|---|---|---|
| radial | none: Dirichlet wall at the requested radius (today's default) | 5.3e-1 | 3.3e-1 | 27 |
| radial | `absorber(grid)`: 10 % inside, AN 252 | 4.0e-1 | 2.1e-1 | 459 |
| radial | 10 % outside, AN 252 | 2.6e-1 | 1.2e-1 | 404 |
| radial | 30 % outside, AN 10 | 3.4e-2 | 1.1e-2 | 68 |
| radial | 50 % outside, AN 30 | 2.2e-3 | 1.3e-3 | 44 |
| Cartesian | none: periodic wrap (today's default) | 4.2e-1 | 1.5e-1 | 27 |
| Cartesian | `absorber(grid)`: 10 % outside, AN 252 | 2.2e-1 | 6.2e-2 | 544 |
| Cartesian | 30 % outside, AN 30 | 3.4e-2 | 2.4e-3 | 147 |
| Cartesian | 50 % outside, AN 30 | 1.3e-2 | 1.5e-3 | 90 |

Three things follow. The present collar removes only between a quarter and a half of the
boundary error, and costs 15 to 20 times the steps of a run without one. A wider, weaker
collar is two orders of magnitude more accurate *and* cheaper. And no setting tried reaches
$10^{-3}$ on both routes, so the profile itself, not only its width and strength, has to
be designed.

The causes are structural, and §3 sets them out: a strength tied to the run length, a
narrow and steep profile, an explicit in-stage rate, inconsistent placement, and a default
of no absorber with only a log message.

**In scope.**

- The real-space collar on the radial, Cartesian (`FreeGrid`), line (`LineGrid`) and planar
  (slab × y) grids: its placement, profile, strength, scheme and default.
- The transverse-wavenumber ($k_\perp$) ramp of SEP 22: its place beside the real-space
  collar, its default, and its extension to the radial grid.
- The spatial part of the boundary monitor: guard and collar occupancy, removal, and the
  warning and stop policy, on the model of SEP 21's temporal monitor.
- The interface: one spatial absorber object, the grid's collar fraction, and the keywords
  of `prop_radial`, `prop_free`, `prop_planar` and their `Simulation.from_*` builders.
- Requalifying the suite's spatial cases and the free-space galleries.

**Out of scope.**

- The temporal and spectral collars, settled by SEP 21.
- Perfectly matched layers by complex coordinate stretching (§10).
- Adaptive transverse windows: growing the radius during a run, remeshing, recentring.
- The escaped field: a record of the light that leaves through the transverse boundary,
  for far-field reconstruction. It is the spatial analogue of SEP 21's open escaped-field
  item and stays on the roadmap.
- The modal and general modal routes, whose transverse structure is a mode basis with no
  transverse window.
- An exact $\tan\theta$ far-field mapping, a separate roadmap item.

### 3. Context

**Today's real-space collar.** Every spatial grid carries a Planck-taper window `W`
([free-space model](../../guide/free-space/model.md#spatial-collar-and-statistics)):

- `RadialGrid(radius, n_r, window_factor=0.1)` puts the taper over the outer tenth
  *of the given radius*, from $0.9R$ to $R$, where the quasi-discrete Hankel transform
  [8] holds the field at zero. The collar eats into the radius the user asked for.
- `FreeGrid(radius_x, n_x, radius_y, n_y, window_factor=0.1)` adds the collar *outside*
  the requested half-widths: the periodic half-width is `radius * (1 + window_factor)`,
  and `n_x`, `n_y` count every sample.
- `LineGrid` and the slab geometry use the Cartesian construction along their periodic axis.

The rate is `12.6 * (1 - W) / ell_c` (`solprop.translate.radial`,
`solprop.translate.free`), the bounded profile that SEP 21 inherited for the temporal
collar. Here `ell_c` is `Absorber.in_stage_length`, which defaults to
`length / boundary_N` with `boundary_N = 20`. The rate enters every stage as
$F\,\mathrm{diag}(-\alpha_r/2)\,B\,E$. On the radial route $F$ and $B$ are the dense
Hankel matrices, composed once at construction (`solprop.backends.spatial`); on the
Cartesian route the same term goes through the transverse FFT pair. It is an explicit,
conditionally stable term: `KernelProblem._check_spatial_collar_stability` refuses a
fixed step beyond `2 * STABILITY_RADIUS / rate`. The temporal `split_step` scheme does
not split it.

**Default off.** `prop_radial`, `prop_free` and `prop_planar` build the temporal and
spectral collar with `Absorber.spatial=False`, so the real-space collar is inert
([which runs carry one](../../guide/numerics/boundaries.md#which-runs-carry-one-and-how-to-change-it)).
An absorber given explicitly switches it on. Without it the radial field reflects off
the Dirichlet wall and the Cartesian field wraps. At each saved plane the driver logs
a warning when the field norm in the collar region (`W < 1`) exceeds the run's `rtol`
(`solprop.backends.spatial.collar_field_fraction`). It does not stop the run, and a plane
that keeps no field is not checked.

**The $k_\perp$ ramp.** `TransverseCollar(strength, fraction=0.1)`, passed as
`SimulationSettings(transverse_absorber=...)` (SEP 22 §4.3), is a raised-cosine power
rate in $\max(|k_x|/k_{x,\max}, |k_y|/k_{y,\max})$. It is diagonal in the transverse
wavenumber, rides the linear operator exactly, and costs nothing per step. It is refused
on the radial grid, although the radial spectral basis is diagonal in $k_r$ too. Its
strength is an absolute rate with no default.

**Prior decisions this builds on.**

- ADR-007 and SEP 21: absorbers are rates, not per-step masks. Placement outside the
  requested window (SEP 21 Q4), and the split scheme on the spatial routes for the
  temporal collar (D14), because there the in-stage collar took 1.2 to 4.6 times the steps.
- SEP 22 §4.6: `collar_length` separates the in-stage collars' strength from the spectral
  ceiling. Today it still couples the temporal and the real-space strength.
- SEP 23: `absorbed_power` has spectral, transverse ($k_\perp$), evanescent and material
  channels. The real-space removal must be a separately named quantity, never relabelled
  as the transverse channel.
- SEP 34: the spatial energy is the admittance-weighted flux. A collar need not commute
  with the admittance, so its removal is measured, not inferred. The collar is passive in
  that metric except near transverse cutoff (SEP 21, passivity finding).

**The roadmap item this answers.** *The real-space collar's geometry* (Numerics). It is
covered in full and is removed from the informal list. *The boundary monitor's flux on
spatial routes* is covered in part (§5.4) and stays, linked.

### 4. Theory

Conventions are those of [conventions.md](../conventions.md). $\alpha$ is a **power**
absorption coefficient: power falls as $e^{-\alpha z}$, the field as $e^{-\alpha z/2}$,
and the linear operator carries $-\alpha/2$.

#### 4.1 A collar is a complex potential

Take one frequency $\omega$ in a homogeneous linear medium with wavenumber
$k = n(\omega)\omega/c$, and one transverse coordinate $x$ normal to the boundary. On the
radial grid $x$ is $r$, and the reduction is the large-argument limit of the Bessel
functions, valid where $\kappa r \gg 1$; at the collar, $r \gtrsim R_\mathrm{req}$, that
holds for every wavenumber that carries light there. In the paraxial limit of the free-space model [9] a
collar of power rate $\alpha(x)$ gives

$$
\partial_z E = \frac{i}{2k}\,\partial_x^2 E - \frac{\alpha(x)}{2}\,E .
$$

A component of transverse wavenumber $\kappa_0$ far from the collar evolves as
$E = u(x)\,e^{-i\kappa_0^2 z/2k}$. Substituting gives a stationary scattering problem,

$$
u'' + q(x)^2\,u = 0,\qquad q(x)^2 = \kappa_0^2 + i\,k\,\alpha(x).
$$

This is the time-independent Schrödinger equation with a negative imaginary potential, the
*complex absorbing potential* (CAP) of wave-packet dynamics [1–5]. The collar's quality
is the CAP's: the reflection $R(\kappa_0)$ from the region where $\alpha$ rises, and the
transmission $T(\kappa_0)$ through it to the wall or the periodic image and back. Away
from the paraxial limit the operator is $k_z(\kappa) = \sqrt{k^2-\kappa^2}$, and below
$k$ is replaced by $k_z$ where it multiplies $\alpha$. The ray angle is
$\tan\theta = \kappa/k_z$.

#### 4.2 Transmission: the strength

Where the absorption is weak, $k\alpha \ll \kappa_0^2$, WKB gives
$\operatorname{Im} q \approx k\alpha/(2\kappa_0)$. A component crossing the collar once is
attenuated in field by $\exp\!\big(-\tfrac{k}{2\kappa_0}\int\alpha\,dx\big)$. This is the
ray result: the transverse velocity is $dx/dz = \kappa_0/k = \tan\theta$, so the
crossing takes the propagation distance $w/\tan\theta$.

Light that crosses a radial collar meets the Dirichlet wall and crosses back. On a
periodic grid it crosses the far collar of the image instead. Either way the field
attenuation of what returns to the aperture is

$$
\Gamma(\kappa_0) = \frac{k}{\kappa_0}\int_\mathrm{collar}\alpha\,dx
= \frac{k}{\kappa_0}\,\alpha_0\,w\,\langle P\rangle ,
$$

for a profile $\alpha(x) = \alpha_0 P\big((x-x_0)/w\big)$ with $0\le P\le1$ over a width
$w$, and $\langle P\rangle = \int_0^1 P\,ds$. $\Gamma$ falls as $1/\kappa_0$: fast,
high-angle content is the hardest to absorb. Requiring $\Gamma \ge \Gamma_\mathrm{min}$
for every $\kappa_0 \le \kappa_\mathrm{hi}$ sets the strength:

$$
\alpha_0 = \frac{\Gamma_\mathrm{min}\,\kappa_\mathrm{hi}}{k_\mathrm{min}\,w\,\langle P\rangle},
$$

with $k_\mathrm{min}$ the smallest $k_z$ over the active band at $\kappa_\mathrm{hi}$, the
red edge. $\Gamma_\mathrm{min} = \log 10^3$, SEP 21's walk-off criterion, removes 60 dB of
power from the returning field.

**The propagation length does not appear.** Today's strength, $12.6/(L/20)$, halves when
the run doubles in length with the same beam and the same grid. A collar sized by
$\kappa_\mathrm{hi}$ and $w$ absorbs the same angles whatever $L$ is.

#### 4.3 Reflection: the width

WKB holds, and the reflection is small, where $q$ changes slowly on the scale of its own
wavelength, $|q'| \ll |q|^2$. At the onset of the collar $q \approx \kappa_0$ and
$q' \approx i k\alpha'/(2\kappa_0)$, so the condition is

$$
k\,|\alpha'| \ll 2\kappa_0^3
\quad\Longrightarrow\quad
\kappa_0 \gg \kappa_\mathrm{lo} \equiv
\left(\frac{k_\mathrm{max}\,\alpha_0\,\max|P'|}{2w}\right)^{1/3}.
$$

Here $k_\mathrm{max}$ is the blue edge, where the onset is steepest relative to
$\kappa_0$. Slow, low-angle content sees a steep wall and reflects. For a profile whose
derivatives all vanish at the onset (the Planck taper, or a smooth polynomial), the
reflection above $\kappa_\mathrm{lo}$ falls faster than any power of $\kappa_0 w$ [3, 5].
The constant in front of $\kappa_\mathrm{lo}$ depends on the profile and is measured in
phase 0.

#### 4.4 The absorbed band

A collar absorbs well on $\kappa_\mathrm{lo} \ll \kappa_0 \le \kappa_\mathrm{hi}$. With the
strength of §4.2,

$$
\frac{\kappa_\mathrm{hi}}{\kappa_\mathrm{lo}} \propto (k\,\alpha_0)^{2/3}\,w^{4/3}.
$$

The width is the stronger lever. This is SEP 21's empirical finding ("width matters
most; weaker is better down to an optimum"), now with a reason.

- **Today's collar** on the R4 case: $w = 30$ µm, $\alpha_0 = 3150$ m⁻¹, $k = 7.85\times10^6$ m⁻¹.
  This gives $\kappa_\mathrm{lo} \sim 10^5$ m⁻¹. The beam's diffracting content,
  $\kappa \sim 2/w_0 = 2\times10^4$ m⁻¹, lies below it and is reflected.
- **The 50 % collar at AN 30** in the probe of §2: $w = 150$ µm, $\alpha_0 = 1500$ m⁻¹.
  This gives $\kappa_\mathrm{lo} \sim 4\times10^4$ and $\kappa_\mathrm{hi} \approx 1.3\times10^5$ m⁻¹.
  The beam's $\kappa \approx 6.7\times10^4$ lies inside the band.

These are order-of-magnitude checks. The constants are phase 0's to measure.

**The real-space collar cannot cover the whole grid.** On the probe grid, the Nyquist
wavenumber is $\pi/\Delta x \approx 10^6$ m⁻¹. Absorbing everything up to $0.9$ of it with
$w = 150$ µm needs $\alpha_0 \approx 10^4$ m⁻¹, and that moves $\kappa_\mathrm{lo}$ above the
beam. The high wavenumbers are where the $k_\perp$ ramp is exact and free, because it is
diagonal there. **The two absorbers are complementary:**

- the real-space collar takes the slow content that reaches the edge, up to
  $\kappa_\mathrm{hi}$;
- the $k_\perp$ ramp takes everything above $\kappa_\mathrm{hi}$, wherever it is in $x$.

The design picks $\kappa_\mathrm{hi}$ where the ramp begins, or below it, so that no band is
left uncovered.

**Which slow content matters.** Content with $\kappa_0 < \kappa_\mathrm{lo}$ reflects, but
it moves slowly: over the run it travels $L\kappa_0/k$. Content launched or generated
well inside the aperture reaches the collar only if
$\kappa_0 \gtrsim k\,(R_\mathrm{req}-r_\mathrm{src})/L$. A long run therefore tolerates a
smaller $\kappa_\mathrm{lo}$ than a short one. Content already near the edge (diffraction
tails, conical emission born at large radius) arrives at any $\kappa_0$. This is where
the run length does enter: it sets a floor on the band that must be absorbed, not the
strength.

#### 4.5 Profiles

The candidates, all $0 \le P \le 1$ on $s \in [0, 1]$ and zero for $s \le 0$:

- the Planck taper's complement, $P = 1 - W$ (today's shape);
- the polynomial $P = s^n$, $n = 2, 3, 4$: the classic CAPs [2, 3], with a closed-form
  optimum for given $(\kappa_\mathrm{lo}, \kappa_\mathrm{hi}, w)$ [2];
- the transmission-free potential of Manolopoulos [4]. It diverges at the far end of the
  collar, so no light is transmitted to the wall, and is optimal for a given
  $\kappa_\mathrm{lo}$. It needs the cap of §4.7.

Phase 0 measures $R(\kappa_0)$ and $T(\kappa_0)$ for each on the discrete grids and chooses
one. The criterion is the largest $\kappa_\mathrm{hi}/\kappa_\mathrm{lo}$ at a fixed width and
a reflection floor of $10^{-6}$ in power.

#### 4.6 The radial grid

The Hankel basis is the Dirichlet eigenbasis of the disc, so a radial collar always has
a perfectly reflecting wall behind it, and $\Gamma$ counts the round trip. Near the axis
the large-$\kappa r$ reduction fails. The collar lies outside $R_\mathrm{req}$, where
$\kappa R_\mathrm{req} \gg 1$ for any content that travels, so the reduction is valid
where it is used. A $k_r$-diagonal ramp is exact on the radial grid, as on the Cartesian
one: the radial linear operator is diagonal in $k_r$. Its largest wavenumber is
$k_{r,\max} = j_{0,N}/R$ [8].

#### 4.7 The scheme

The collar rate does not depend on $z$ or on the field. Split from the rest of the step,
its exact flow over an accepted step $h$ is the real-space factor
$e^{-\alpha(x)h/2}$, unconditionally stable and exact for the absorber alone. The
splitting error is first order, as SEP 21 found for the temporal collar, and it is
confined to where $\alpha \ne 0$: outside the requested aperture. A stiff profile, such as
the transmission-free one, is admissible only under the split. In-stage it would set the
step.

On the Cartesian, line and planar routes the split temporal collar already visits the
physical $(x, y, t)$ field once per accepted step
(`solprop.backends.numpy_ref.backend`, the `split_step` pass). The spatial factor can be
multiplied in on that visit for nothing. On the radial route it costs one dense transform
pair per accepted step: about one stage's transverse work, against the stages the
in-stage collar adds.

The removal is measured in SEP 34's metric during the split pass, as for the temporal
collar (SEP 21 §4.6): the admittance-weighted flux before and after the factor.

### 5. Proposed interface

#### 5.1 The grid: placement

Every spatial grid takes `collar_fraction` in place of `window_factor`. The collar lies
**outside** the requested size on every geometry:

```text
RadialGrid(radius=300e-6, n_r=96, collar_fraction=0.3)
# outer Dirichlet radius = radius * (1 + collar_fraction); n_r counts every sample
FreeGrid(radius_x, n_x, radius_y, n_y, collar_fraction=0.3)
# periodic half-width = radius * (1 + collar_fraction), as today
```

`radius` is the aperture the user wants modelled: flat, with no absorption. The default
fraction is a phase 0 decision (Q2). The probe suggests 0.3 to 0.5. A Cartesian grid pays
the square of it in memory, so its default may be smaller than the radial one.
`collar_fraction=0` leaves no room for a collar. The window `W` keeps its role as the
monitor's region marker.

#### 5.2 The absorber: one object for the transverse boundary

```text
from solprop.boundaries import SpatialAbsorber, spatial_absorber

boundary = spatial_absorber(space, grid)                       # the default design
boundary = spatial_absorber(space, grid, kappa_hi=2e5)         # absorb up to 2e5 rad/m
boundary = spatial_absorber(space, grid, angle=25e-3)          # the same as a ray angle
boundary = spatial_absorber(space, grid, attenuation=math.log(1e4), kperp=False)

prop_radial(grid, space, medium, responses, pulse, settings=..., spatial_absorber=boundary)
prop_free(..., spatial_absorber=False)                         # wall or wrap, as today
```

`SpatialAbsorber` is a frozen dataclass:

- `rate`: the real-space power rate on the grid's nodes, m⁻¹, zero inside the requested
  size;
- `kperp_rate`: the $k_\perp$ ramp's power rate on the spectral columns, m⁻¹, or zeros;
- the design it records: `kappa_hi` (rad/m), `attenuation` ($\Gamma_\mathrm{min}$),
  `profile`, `kperp_fraction`, and the derived `kappa_lo`.

`spatial_absorber` builds both parts from §4.2–4.4:

- the real-space strength from `kappa_hi`, the width and the active band;
- the $k_\perp$ ramp starting at `kappa_hi` (or at `1 - kperp_fraction` of the grid's
  largest wavenumber, whichever is lower);
- a ramp strength large enough to remove $\Gamma_\mathrm{min}$ within a distance short
  against the run (the ramp is exact, so its strength costs nothing).

It refuses a design whose `kappa_lo` lies above `kappa_hi`. It warns when `kappa_lo` lies
above the launched beam's rms transverse wavenumber, since the beam's own diffraction
would then be reflected.

**The default `kappa_hi`** (Q1, Q5) is taken from the launched field: a multiple $c_b$ of
its rms transverse wavenumber, evaluated at translation, as SEP 21's walk-off default is
taken from the pulse. $c_b$ is a phase 0 constant, chosen to cover self-focusing and
conical emission on the witnesses. The $k_\perp$ ramp covers the rest of the grid.

**Removed:**

- `TransverseCollar` and `SimulationSettings.transverse_absorber`: the ramp is part of
  `SpatialAbsorber`;
- `Absorber.spatial`: the temporal `Absorber` no longer touches the transverse boundary;
- the real-space use of `Absorber.collar_length`, which then sets only the temporal
  in-stage collar;
- `window_factor`.

`absorber=False` on a spatial interface keeps its SEP 21 meaning (no temporal or spectral
collar) and does not touch the spatial one.

#### 5.3 Scheme

`SimulationSettings(spatial_absorber_scheme="split_step" | "nonlinear")`. The default is
`"split_step"` unless phase 0 finds otherwise (Q4). It follows the temporal scheme's rules:
the split refuses interpolated planes, which SEP 21 admits only in-stage. The in-stage
scheme keeps the stability check of §3 and refuses a stiff profile.

#### 5.4 Monitoring

The boundary monitor (`solprop.monitor.BoundaryCheck`, SEP 21) gains a transverse edge.
At its cadence it measures:

- the share of the field's energy in a guard band, the outer 5 % of the requested size,
  in SEP 34's flux on a paraxial beam and in the quadrature-weighted field norm otherwise
  (SEP 21 D16);
- the share inside the collar;
- the energy the collar has removed.

It warns and stops on the temporal monitor's policy and thresholds, and says which edge
fired and whether to widen the aperture or raise `kappa_hi`. The removed energy is
recorded as `spatial_removed`, separate from `absorbed_power`'s transverse ($k_\perp$)
channel (SEP 23). The end-of-plane log of `collar_field_fraction` is retired.

#### 5.5 Composition

- *Scans and coupling*: a `SpatialAbsorber` is plain data and is carried as part of the
  run's settings. A run relaunched from a saved plane keeps its design.
- *Single precision*: the factor is applied in the state's precision, as the temporal split is.
- *The low-storage march* (SEP 24): it already admits the split temporal collar and takes
  the spatial factor the same way.
- *The exponential Adams march* (SEP 16/39): it admits the split collar on the
  temporal-collar argument of SEP 21 (the state changes only where the splitting error
  already lies).
- *Refused*: a `SpatialAbsorber` on a modal route; a `kperp_rate` whose shape does not
  match the grid.

### 6. Implementation strategies

**Layers.** Design and validation in physics (`solprop.boundaries`, `solprop.grid`).
Rates onto the kernel problem in translation (`solprop.translate.radial`, `.free`,
`.planar`, `.spatial`). The split pass in each backend. The monitor in
`solprop.monitor` and the backends' reductions.

**The real-space rate: split (recommended) against in-stage.**

- *Split.* One factor per accepted step: free on the Cartesian, line and planar routes,
  one dense transform pair on the radial route. It is unconditionally stable and admits
  stiff profiles. It is first order, confined to the collar.
- *In-stage* (today). It adds the term to every stage: on the radial route a dense matrix
  product per stage; on the Cartesian route a transform pair per stage, unless fused into
  the nonlinear pair that is already there. It is conditionally stable and steers the error
  controller (459 against 27 steps in §2).

A good design is weaker, so the in-stage penalty shrinks: 44 steps in §2. Phase 0 measures
both on the witnesses. The default follows the measurement, as D14 did for the temporal collar.

**Folding the rate into the linear operator** is exact but not affordable. In the radial
basis $F\,\mathrm{diag}(\alpha)\,B$ is a dense $n_r\times n_r$ matrix that does not commute
with the diagonal linear operator. The cumulative operator would become a per-frequency
matrix exponential, which breaks the diagonal contract of
[kernel-contract.md](../kernel-contract.md) §3.

**The $k_\perp$ ramp on the radial grid** joins the existing per-column linear rate. It is
diagonal in $k_r$, as it is in $(k_x, k_y)$. The translation lifts today's refusal.

**Kernel contract.** The problem gains the spatial split rate, the scheme, and the radial
per-column ramp. `SCHEMA_VERSION` is bumped once, in phase 1. The output schema records
the `SpatialAbsorber` design and `spatial_removed`, also in phase 1.

**Backends, in order.**

- `numpy_ref`: the oracle.
- `numba_cpu`: the factor goes into the existing split kernels.
- The native CPU build: the factor goes into the split pass, which SEP 21 already compiled
  for the temporal collar. The radial transform pair reuses the stage's Hankel matrices.
- The MLX and native device backends follow the standing device policy
  ([backends.md](../backends.md)) and refuse the split spatial scheme by name until
  qualified.

**Cost.**

- The design is evaluated once at translation; the $\kappa$ reductions of the launched
  field are $O(n_\mathrm{nodes}\,n_\omega)$ once.
- The split factor is $O(n_\mathrm{nodes}\,n_t)$ per accepted step, inside an existing pass.
- The radial transform pair is $O(n_r^2\,n_\omega)$ per accepted step.
- A wider collar costs samples: $(1+f)$ on the radial grid, $(1+f)^2$ on the Cartesian one.
  This is the main cost of the proposal, and the reason for Q2.

**No new dependencies.**

### 7. Verification and validation

1. **Reflection and transmission curves (the core qualification).**
   - Cartesian: a tilted Gaussian of transverse wavenumber $\kappa_0$, linear, in vacuum
     or a dilute gas, aimed at the collar.
   - Radial: an outgoing annular wave packet (a ring of radius well inside the aperture,
     with a radial phase gradient $\kappa_0$).
   - Measure the returning energy against $\kappa_0$ for each profile, width and strength.
   - Compare with the WKB scalings of §4.2–4.3 and with a direct numerical solution of the
     1D scattering problem of §4.1 on a fine grid. The 1D solution is an independent
     reference, by a different method.
   - Tolerance: agreement of $\kappa_\mathrm{lo}$ and $\kappa_\mathrm{hi}$ within a factor
     the 1D solution's grid convergence sets. A measured reflection floor below
     $10^{-6}$ in power over the designed band.
2. **Linear diffraction against an analytic reference.** A Gaussian beam's angular
   spectrum is Gaussian. Propagated exactly with $e^{ik_z z}$ and transformed back by
   adaptive quadrature, independent of the QDHT and the FFT, it gives the field over the
   requested aperture at any $z$, non-paraxial included. The §2 probe becomes a test:
   field L2 error over the aperture below $10^{-3}$ at the default design, at no more steps
   than the run with no absorber, within the step controller's noise.
3. **The self-focusing witness.** The suite's `spatial_absorber` case against 2x and 4x
   radius references without a collar, extrapolated as SEP 21 R4 did. Target: the
   default design within the extrapolation's uncertainty, or within $10^{-3}$ if that is
   larger. Ten steps or a short length in tests. The full 8 cm only as a gallery or suite
   run.
4. **Passivity.** The generalised eigenvalues of the collar's quadratic form against
   SEP 34's flux, as SEP 21 measured: nonnegative to rounding away from transverse cutoff.
   Removal along $z$ positive on the witnesses.
5. **Invariants.**
   - A beam that stays inside the requested aperture is unchanged by the absorber to
     rounding.
   - Energy balance: the flux at $z$ plus `spatial_removed` plus the other channels equals
     the launched flux, within the step's error.
6. **Split against in-stage.** First-order convergence of the difference in $h$, confined
   to the collar region.
7. **Independence of $L$.** The same beam over $L$ and $2L$, saved at a common plane,
   gives the same field there to the controller's tolerance. Today it does not.
8. **Parity.** `numpy_ref`, `numba_cpu` and the native CPU build agree within the
   summation-bound tolerances of [testing.md](../testing.md), on short windows.
9. **Cost.** Accepted steps and wall time on the suite's spatial cases against today and
   against no absorber. A ledger entry in [benchmarks.md](../benchmarks.md).

### 8. Documentation

- [Free-space model](../../guide/free-space/model.md): *Spatial collar and statistics* is
  rewritten as *Transverse boundaries*. It covers the theory of §4 in its current form,
  the design parameters, the default, the monitor and a worked example.
- [Boundaries](../../guide/numerics/boundaries.md): the transverse boundary is linked from
  *Which runs carry one*, and the table's spatial row is updated.
- [Cartesian](../../guide/free-space/cartesian.md): *Grid sizing and the transverse safety
  ramp* is updated for the joined object and the complementary design.
- [Grids](../../guide/setup/grids.md) and
  [spatial decompositions](../../guide/setup/spatial-decompositions.md): `collar_fraction`
  and the placement outside the request.
- A short gallery tutorial: *Is my transverse window big enough?*, the spatial companion
  of the temporal one, built on the §2 probe.
- [Differences from Luna](../../guide/workflows/luna.md): Luna.jl's free-space routes apply
  no transverse absorber, so a radial run reflects from its outer radius and a Cartesian
  one wraps. solprop absorbs by default.

### 9. Compatibility and impact

Breaking, deliberately (no compatibility shims):

- `window_factor` becomes `collar_fraction`, and on `RadialGrid` the collar moves outside
  `radius`. A script that gave `RadialGrid(R, n_r)` now gets a larger outer radius and a
  coarser $\Delta r$ at the same `n_r`.
- `TransverseCollar`, `SimulationSettings.transverse_absorber` and `Absorber.spatial` are
  removed.
- Spatial runs absorb at the transverse edge by default (if Q1 is decided so). Results of
  runs whose light reaches the edge change, which is the purpose.

Kernel `SCHEMA_VERSION` and output `SCHEMA_VERSION` are bumped in phase 1.

Validated results to redo:

- the suite's `spatial_absorber` case and its baselines;
- the radial and Cartesian performance-tracking cases;
- the free-space galleries (`plot_03_line`, `plot_05_low_storage`,
  `plot_06_four_wave_mixing_scan`, `plot_07_far_field_statistics`) and the geometry figure.

Downstream Cartesian models that run with no real-space collar and a $k_\perp$ ramp take
the new object with `kperp` only, or the default design. Their numbers must be
re-qualified.

### 10. Alternatives considered

- **Keep the collar, retune its constants.** SEP 21's sweep shows that width and strength
  alone reach a few $10^{-3}$ at best, and that the right strength depends on the beam,
  not on $L$. Retuning would leave the length coupling and the in-stage cost.
- **Perfectly matched layers** [7]: complex coordinate stretching
  $x \to x + i\sigma(x)/\omega$. A PML is reflectionless in the continuum for every angle.
  On a Fourier or Hankel spectral grid, though, the stretched Laplacian is no longer
  diagonal in the transform basis, so the linear step loses its exact diagonal propagator.
  Discretised PMLs on spectral grids also reflect at the grid level. Out of reach of the
  kernel contract without a different transverse discretisation.
- **Transparent boundary conditions** [6]: exact for finite-difference beam propagation
  with an outgoing plane-wave ansatz at the last node. They do not exist for a global
  spectral transform, whose boundary is periodic or Dirichlet by construction.
- **Absorbing in $k_\perp$ alone.** It is exact and free, but it acts everywhere in $x$:
  slow content is not absorbed at the edge, and content at the beam's own wavenumbers
  cannot be touched. It is the complement of the real-space collar (§4.4), not a
  replacement.
- **A grid-only default** (cover the whole band below the $k_\perp$ ramp with the real-space
  collar). §4.4 shows that it moves $\kappa_\mathrm{lo}$ above the beam at practical widths.
  It is kept as a design the user can request, not as the default.

### 11. Phases

**Phase 0 — measurement** (`numpy_ref` and scripts; no public interface change).

- The $R(\kappa)$ and $T(\kappa)$ curves of §7.1 for the profiles of §4.5, on both routes.
- The 1D scattering reference.
- The §2 probe and the self-focusing witness across fraction, design and scheme.
- Measure the constants: $\kappa_\mathrm{lo}$'s prefactor, $c_b$, the default fraction, and
  split against in-stage.
- Report with recommendations for Q1–Q6. The maintainer decides before phase 1.

**Phase 1 — `numpy_ref`, interface and documentation.**

- `collar_fraction` and the outside placement.
- `SpatialAbsorber` and `spatial_absorber`, and the removals of §5.2.
- The split pass and the radial $k_\perp$ ramp.
- The monitor's transverse edge and `spatial_removed`.
- Both schema bumps.
- Tests §7.1–7.7 on `numpy_ref`.
- The manual pages and the gallery tutorial.
- The other backends refuse the new scheme by name.

**Phase 2 — `numba_cpu`.** The split factor, the radial ramp and the monitor reductions,
with parity tests.

**Phase 3 — native CPU, requalification, Final.**

- The compiled split factor and the radial transform pair.
- Parity.
- The suite's spatial cases, the galleries, the benchmark ledger and the Luna page.
- Device backends left to the standing policy, as a roadmap item.

### 12. Open questions

1. **Q1 — the default.** Do `prop_radial`, `prop_free` and `prop_planar` build the default
   `SpatialAbsorber` when none is given? Recommendation: yes, after phase 0, with
   `spatial_absorber=False` for a bare wall or wrap.
2. **Q2 — the default collar fraction**, per geometry. Radial costs $(1+f)$ in samples,
   Cartesian $(1+f)^2$. Phase 0 proposes from the witnesses.
3. **Q3 — the profile** (§4.5). Phase 0.
4. **Q4 — the default scheme.** Split unless the in-stage collar is as accurate at no more
   steps on the witnesses. Phase 0.
5. **Q5 — the default `kappa_hi`.** Launched beam times $c_b$, or a protected angle the
   user states, or both with a refusal when they conflict. Phase 0 and the maintainer.
6. **Q6 — the $k_\perp$ ramp's default strength and start**, and whether it is on by
   default on the radial grid. Phase 0.
7. **Q7 — the monitor's thresholds** on the transverse edge. They start with the temporal
   monitor's; whether stop or warn is the default is the maintainer's.

### 13. References

1. R. Kosloff and D. Kosloff, "Absorbing boundaries for wave propagation problems,"
   *J. Comput. Phys.* **63**, 363–376 (1986). doi:10.1016/0021-9991(86)90199-3
2. Á. Vibók and G. G. Balint-Kurti, "Parametrization of complex absorbing potentials for
   time-dependent quantum dynamics," *J. Phys. Chem.* **96**, 8712–8719 (1992).
   doi:10.1021/j100201a012
3. U. V. Riss and H.-D. Meyer, "Investigation on the reflection and transmission
   properties of complex absorbing potentials," *J. Chem. Phys.* **105**, 1409–1419
   (1996). doi:10.1063/1.472003
4. D. E. Manolopoulos, "Derivation and reflection properties of a transmission-free
   absorbing potential," *J. Chem. Phys.* **117**, 9552–9559 (2002).
   doi:10.1063/1.1517042
5. J. G. Muga, J. P. Palao, B. Navarro and I. L. Egusquiza, "Complex absorbing
   potentials," *Phys. Rep.* **395**, 357–426 (2004). doi:10.1016/j.physrep.2004.03.002
6. G. R. Hadley, "Transparent boundary condition for the beam propagation method,"
   *IEEE J. Quantum Electron.* **28**, 363–370 (1992). doi:10.1109/3.119536
7. J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic
   waves," *J. Comput. Phys.* **114**, 185–200 (1994). doi:10.1006/jcph.1994.1159
8. M. Guizar-Sicairos and J. C. Gutiérrez-Vega, "Computation of quasi-discrete Hankel
   transforms of integer order for propagating optical wave fields," *J. Opt. Soc. Am. A*
   **21**, 53–58 (2004). doi:10.1364/JOSAA.21.000053
9. A. Couairon, E. Brambilla, T. Corti, D. Majus, O. de J. Ramírez-Góngora and
   M. Kolesik, "Practitioner's guide to laser pulse propagation models and simulation,"
   *Eur. Phys. J. Spec. Top.* **199**, 5–76 (2011). doi:10.1140/epjst/e2011-01503-3

## Part II — Implementation record

## Part III — History

- 2026-10-31: Drafted, from SEP 21's finding R4 and decision D9, and a linear probe of the
  radial and Cartesian collars (§2).
