# SEP 40 — Transverse absorbing boundaries for free-space propagation

| | |
|---|---|
| **SEP** | 40 |
| **Title** | Transverse absorbing boundaries for free-space propagation |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 |
| **Status** | Draft |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-31 |
| **Resolution** | Accepted 2026-10-31 by John Travers; returned to Draft 2026-11-01 for re-acceptance, because phase 0 changed the design (Part II, Part III). |
| **Requires** | [SEP 21](archive/sep-0021-absorbing-boundaries.md), [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md), [SEP 34](archive/sep-0034-medium-flux.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers, branch `feature/sep-0040-transverse-boundaries`: phase 0 complete (Part II); phase 1 waits for re-acceptance |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Rebuild the transverse boundaries of the radial, Cartesian, line and planar free-space
routes, so that a run whose light reaches the boundary is never silently wrong, and a run
that needs an absorber gets one designed to a stated accuracy at a measured cost.

Phase 0 changed the design this SEP was first accepted with, a real-space collar on every
run. On a slow beam, no designed collar beat a bare wall at the same outer radius, and no
fixed fraction of the request met the budget (Part II). This revision follows that evidence:
distance and detection by default, and absorption for the fast light that needs it.

- **The default.** No real-space collar: a wall or a wrap at the requested size, as today.
  A $k_\perp$ ramp, SEP 22's, stays on as a *numerical safety band* at the grid's outer
  wavenumbers, and is offered on the radial and planar grids too.
- **The monitor.** The boundary monitor gains a transverse edge, on by default. With no
  absorber, whatever reaches the edge comes back, so it judges the energy *reaching* the edge
  and stops a run that exceeds the budget, naming a size to try.
- **A collar on request.** A `SpatialAbsorber`, outside the requested aperture, with a strength
  set by a *protected* band of transverse wavenumbers and by the collar's width, not by the
  propagation length.
- **The design check.** Translation predicts how much of the launch the collar would reflect
  back, and refuses a design above the budget, naming the fraction that meets it. With a
  collar, the monitor judges the light *returning*.
- **Scheme and step.** The collar is an in-stage rate seen by the error controller, under a
  step limit that stops a packet crossing it unseen within one step.

Success is judged against an allocated error budget (§7.1):

- the suite's self-focusing case as configured, 10 % wrong in field at 8 cm, is stopped by
  default; a run whose light stays away from the edge completes with no warning beyond INFO;
- a linear beam that sends most of its energy through a collar that passes the design check
  agrees with an independent analytic reference to $10^{-3}$ over the requested aperture;
- measured reflection and transmission against transverse wavenumber agree with the
  calibrated design over the protected band;
- a weak sideband generated at an angle leaves through a requested collar, agreeing with a
  larger-domain reference to $10^{-5}$ of its own field;
- the same self-focusing case with a requested collar reaches a converged large-radius
  reference to $10^{-3}$.

Cost is reported at matched accuracy. Backward compatibility is not a constraint: no
release depends on the present interface, and downstream scripts are updated with it.

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
  angles must leave the box, not fold back onto the beam. Weak legitimate sidebands
  inside the aperture must not be damped on the way there.
- *The line and planar routes* (`LineGrid` and the hybrid slab × y geometry), which have
  a periodic transverse axis of their own.

**The problem.** SEP 21's campaign (its finding R4, decision D9) measured the radial
real-space collar on the suite case with temporal absorption off. The collar as shipped
covers the outer 10 % of the requested radius, with peak rate $252/L$. It left a field
error of $1.0\times10^{-1}$ over the requested radius and $2.3\times10^{-2}$ in the core.
A hard Dirichlet wall at the same radius gave $1.0\times10^{-1}$ and $2.8\times10^{-2}$. The
error was reflection from the collar, and it reached the core. A collar outside the
requested radius, 30 % of the outer radius and 25 times weaker, was 20 times more accurate
over the aperture and 130 times in the core, and took 46 Runge–Kutta steps instead of 241.
D9 deferred the collar's geometry to a follow-up proposal. This is that proposal.

**A linear probe** on 2026-10-31 (main branch; the spatial collar code was unchanged since
SEP 21) extended the finding to the Cartesian route and to a beam that loses most of its
energy through the boundary.

- *Grid and launch.*
  - Envelope grid `EnvelopeGrid.from_limits(0.02, 800e-9, (700e-9, 920e-9), 400e-15, thg=False)`.
  - Launch `SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-9), 30e-6)`:
    a 30 µm beam, about six Rayleigh lengths over $L = 2$ cm.
  - Argon at 1 bar with its Kerr response, at an energy where the Kerr term is negligible.
  - The builders' default input noise.
- *Stepping.* `RungeKutta()`, `PIController(rtol=1e-7, atol=0)`, `dz_init = L·1e-4`,
  `dz_max = L`, one saved plane at $L$, backend `numba_cpu`.
- *The collar.* `absorber(grid, temporal=False)`, with `collar_length` set to give the
  stated peak rate. The profile is today's $12.6(1-W)$ with $W$ the grid's Planck window.
- *Requested size and sampling.* The requested radius or half-width is 150 µm. The
  transverse spacing is held fixed while the domain grows: $\Delta r = 3.125$ µm radially,
  $\Delta x = 3$ µm on the Cartesian grid.
- *Radial geometry.* The outer radius is $R = R_\mathrm{req}/(1-f)$, so a collar "outside"
  is $fR$ wide. The comparison evaluates each run's Fourier–Bessel series at 241 uniform
  radii on $[0, 150]$ µm.
- *Cartesian geometry.* `FreeGrid(150e-6, n, 150e-6, n, window_factor=f)` with
  $n = 100(1+f)$. The comparison uses the common 99 × 99 nodes with $|x|,|y|\le147$ µm.
- *References.* A grid of 600 µm radius ($n_r = 192$), or a 300 µm half-width box
  ($n = 200$), at the same spacing, without a collar, in which the field at $L$ does not
  reach the edge. Phase 0 later found the Cartesian reference itself wrong by
  $5.5\times10^{-4}$, and that the small radial core errors were mostly the default input
  noise (Part II); the table's aperture errors reproduce to two digits.
- *The error.* The area-weighted field L2 difference at $z = L$, over all frequencies and
  over the requested aperture, divided by the reference's norm over the same region at
  $L$. "Core" means $r\le75$ µm. AN is the collar's peak power rate times $L$.

| route | boundary | outer size (µm) | collar width (µm) | aperture | core | steps |
|---|---|---|---|---|---|---|
| radial | none: Dirichlet wall (today's default) | 150 | 0 | 5.3e-1 | 3.3e-1 | 27 |
| radial | `absorber(grid)`: inside, AN 252 | 150 | 15 | 4.0e-1 | 2.1e-1 | 459 |
| radial | outside, AN 252 | 167 | 17 | 2.6e-1 | 1.2e-1 | 404 |
| radial | outside, AN 10 | 214 | 64 | 3.4e-2 | 1.1e-2 | 68 |
| radial | outside, AN 30 | 300 | 150 | 2.2e-3 | 1.3e-3 | 44 |
| Cartesian | none: periodic wrap (today's default) | 150 | 0 | 4.2e-1 | 1.5e-1 | 27 |
| Cartesian | `absorber(grid)`: outside, AN 252 | 165 | 15 | 2.2e-1 | 6.2e-2 | 544 |
| Cartesian | outside, AN 10 | 195 | 45 | 5.6e-2 | 1.2e-2 | 118 |
| Cartesian | outside, AN 30 | 195 | 45 | 3.4e-2 | 2.4e-3 | 147 |
| Cartesian | outside, AN 30 | 225 | 75 | 1.3e-2 | 1.5e-3 | 90 |

The present collar removes only between a quarter and a half of the boundary error, and
costs 15 to 20 times the steps of a run without one. A wider, weaker collar is far better,
and the first version of this SEP took that to mean that every run should carry one.

**What phase 0 found** (Part II, every number reproducible from its harness).

- *Distance beats absorption for a slow beam.* On the suite case, at the same radial spacing:

  | boundary | error at 8 cm | Multistep steps |
  |---|---|---|
  | wall at the requested radius, as the case is configured | $1.0\times10^{-1}$ | 149 |
  | today's collar inside the request | $9.9\times10^{-2}$ | 470 |
  | a bare wall at twice the radius | $3.9\times10^{-5}$ | 68 |
  | a designed collar at the same outer radius | $2.1\times10^{-3}$ | 174 |
  | a designed collar, 3.6 times the radius | $2.9\times10^{-6}$ | 107 |
  | a bare wall, 3.6 times the radius | $6.3\times10^{-9}$ | 68 |

  No collar in a scan of fractions, band edges and strengths beat a wall at the same outer
  radius, and the probes above agree on all three geometries. The slow content that matters lies
  below the collar's reflecting threshold, however the design is chosen.
- *The collar's value is in fast content.* A weak four-wave-mixing sideband at
  $1.5\times10^5$ rad/m, $10^{-6}$ of the pump's energy, leaves through a designed collar to
  $4\times10^{-6}$ of its own field. With the collar inert it wraps back, wrong by 0.98.
- *So a default collar would be both costly and unsafe.* A fraction wide enough to pass is no
  better than a wall at the same radius, at up to 30 times the per-step cost.

**So the default is detection.** A run that reaches its boundary needs more room, and the
useful thing is to say so. A collar is for the run that has fast light it cannot give room
to, and which declares that light.

**In scope.**

- The monitor's transverse edge, with its warning and stop policy, on by default.
- The real-space collar on request, on the radial, Cartesian (`FreeGrid`), line (`LineGrid`)
  and planar (slab × y) grids: its placement, profile, strength, step limit and design check.
- The transverse-wavenumber ($k_\perp$) ramp of SEP 22, as a numerical safety band,
  on the Cartesian, line, radial and planar grids, on by default.
- The removal ledger: what a collar removes, measured, signed, and attributed alongside
  the temporal collar and the mask.
- The interface: one spatial absorber specification, its resolution at translation, the
  grid's collar fraction, and the keywords of `prop_radial`, `prop_free`, `prop_planar` and
  their `Simulation.from_*` builders.
- Requalifying the suite's spatial cases and the free-space galleries.

**Out of scope.**

- The temporal and spectral collars, settled by SEP 21.
- A split real-space factor applied once per step: measured in phase 0, and left to the
  roadmap (§10).
- Perfectly matched layers by complex coordinate stretching (§10).
- Adaptive transverse windows: growing the radius during a run, remeshing, recentring.
- The escaped field: a record of the light that leaves through the transverse boundary,
  for far-field reconstruction. It is the spatial analogue of SEP 21's open escaped-field
  item and stays on the roadmap.
- A flux-weighted (SEP 34) monitor and ledger on spatial routes. They stay with the
  roadmap item of SEP 21 D16, and this SEP uses the field-norm measure (§4.9).
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
- `LineGrid` uses the Cartesian construction along x.
- `PlanarGrid` (the slab × y geometry) builds its y axis the same way, as
  `PlanarGrid.window_y`, and applies a y collar, a function of y alone, row by row.

The rate is `12.6 * (1 - W) / ell_c` (`solprop.translate.radial`,
`solprop.translate.free`, `solprop.translate.planar`), the bounded profile that SEP 21
inherited for the temporal collar. Here `ell_c` is `Absorber.in_stage_length`, which
defaults to `length / boundary_N`, with `boundary_N = 20` a parameter of `absorber()`.
The rate is carried as `SpatialGeometry.absorber_rate` on the problem's transform. It
enters every stage as
$F\,\mathrm{diag}(-\alpha_r/2)\,B\,E$, independently of the material prefactor:

- on the radial route, $F$ and $B$ are the dense Hankel matrices, composed once at
  construction (`solprop.backends.spatial`), so each stage pays one dense product;
- on the Cartesian route each stage pays a transverse FFT pair of its own.

It is an explicit, conditionally stable term: `KernelProblem._check_spatial_collar_stability`
refuses a fixed step beyond $2r/\alpha_\mathrm{eff}$. Here $r$ is the integrator's
negative-real stability interval, `min(STABILITY_RADIUS, integrator.real_stability_interval)`:
3.3 for Runge–Kutta, 0.0877 for the default Adams `Multistep` at order 5. The effective rate
$\alpha_\mathrm{eff}$ is the collar's peak, times the radial transform's norm factor on the
radial grid, plus the temporal collar's rate under the in-stage temporal scheme. The temporal
`split_step` scheme (`ResponseSet.absorber_scheme`) does not split it.

**Default off.** `prop_radial`, `prop_free` and `prop_planar` build the temporal and
spectral collar with `Absorber.spatial=False`, so the real-space collar is inert
([which runs carry one](../../guide/numerics/boundaries.md#which-runs-carry-one-and-how-to-change-it)).
An absorber given explicitly switches it on. Without it the radial field reflects off the
Dirichlet wall and the Cartesian field wraps. At each saved plane the backend driver
(`solprop.backends.driver`) logs a warning when the field norm in the collar region
(`W < 1`, `solprop.backends.spatial.collar_field_fraction`) exceeds the run's `rtol`, floored
at a rounding level. It does not stop the run, and a plane that keeps no field is not checked.
The suite's `spatial_absorber` tracking case therefore runs with the real-space collar off:
its boundary is the Dirichlet wall.

**Other consumers of the window.** The coupling plane's radial resampling builds
`RadialGrid(2R, 2n_r, window_factor)`, which assumes the collar lies inside $R$
(`solprop.coupling.plane`). The coupling elements use `W < 1` as the boundary mask
(`solprop.coupling.elements`). The run record stores the window and the rate.

**The $k_\perp$ ramp.** `TransverseCollar(strength, fraction=0.1)`, passed as
`SimulationSettings(transverse_absorber=...)` (SEP 22 §4.3), is a raised-cosine power rate
in $\max(|k_x|/k_{x,\max}, |k_y|/k_{y,\max})$. It is diagonal in the transverse wavenumber,
rides the linear operator exactly, and costs nothing per step. Its strength is an absolute
rate with no default. It is refused on the radial grid (`transverse_collar_rate`) and on the
planar grid (SEP 22 Q5), although both spectral bases are diagonal in their transverse
wavenumber.

**The monitor.** `solprop.monitor.BoundaryCheck` (SEP 21) judges the *time* window's
guards. It warns when the square root of a guard's energy share exceeds its budget,
$10^{-3}$ by default, and stops above ten budgets, so an energy share above $10^{-4}$
stops the run. On spatial routes its shares and the split collar's removal
(`BoundaryMonitor.split_removal`) use the quadrature-weighted field norm, not SEP 34's
flux (SEP 21 D16).

**Integrator history.** A split that changes the accepted state after the stages forbids
first-same-as-last reuse; the backends already turn it off for the temporal split. The
Adams march keeps a history of sources (`solprop.backends.multistep`), which a pointwise
change of the state does not update. SEP 21 admitted the temporal split under Adams only
because the change lies where its splitting error already lies, and recorded the failure
of a history-transported Strang experiment (its §4.5).

**Prior decisions this builds on.**

- ADR-007 and SEP 21: absorbers are rates, not per-step masks. Placement outside the
  requested window (SEP 21 Q4). The split scheme on the spatial routes for the temporal
  collar (D14).
- SEP 22 §4.6: `collar_length` separates the in-stage collars' strength from the spectral
  ceiling. Today it still couples the temporal and the real-space strength.
- SEP 23: `absorbed_power` has spectral, transverse ($k_\perp$), evanescent and material
  channels, in J/m. Real-space removal must be a separately named quantity, never
  relabelled as the transverse channel.
- SEP 34: the spatial energy is the admittance-weighted flux. A collar need not commute
  with the admittance, so its removal is measured, not inferred. SEP 21's passivity
  finding records a worst generalised eigenvalue of $-2.9\times10^{-7}$ of the peak rate for
  the real-space collar on its radial witness: the collar is **not** passive in that metric
  to rounding (§4.9).

**The roadmap item this answers.** *The real-space collar's geometry* (Numerics). It is
covered in full and is removed from the informal list. *The boundary monitor's flux on
spatial routes* stays, linked.

### 4. Theory

Conventions are those of [conventions.md](../conventions.md). $\alpha$ is a **power**
absorption coefficient: power falls as $e^{-\alpha z}$, the field as $e^{-\alpha z/2}$, and
the linear operator carries $-\alpha/2$. Attenuations $\Gamma$ are in **field** nepers, and
reflection coefficients $r$ are field amplitudes; powers are their squares.

This section gives the design a collar uses. Its scalings come from the theory, and its
constants from phase 0's calibration on the discrete radial, Fourier and slab operators
themselves (§7.3, Part II).

#### 4.1 A collar as a complex potential, and the approximation's limits

Take one frequency $\omega$ in a homogeneous linear medium with wavenumber
$k = n(\omega)\omega/c$, and one transverse coordinate $x$ normal to the boundary. In the
paraxial limit of the free-space model [9], a collar of power rate $\alpha(x)$ gives

$$
\partial_z E = \frac{i}{2k}\,\partial_x^2 E - \frac{\alpha(x)}{2}\,E .
$$

A component of transverse wavenumber $\kappa_0$ far from the collar evolves as
$E = u(x)\,e^{-i\kappa_0^2 z/2k}$. Substituting gives a stationary scattering problem,

$$
u'' + q(x)^2\,u = 0,\qquad q(x)^2 = \kappa_0^2 + i\,k\,\alpha(x).
$$

This is the time-independent Schrödinger equation with a negative imaginary potential, the
*complex absorbing potential* (CAP) of wave-packet dynamics [1–5]. A collar's quality is
the CAP's:

- the reflection $r(\kappa_0)$ from the region where $\alpha$ rises;
- the transmission through the collar to the wall, or to the periodic image, and back.

**Limits.** The reduction holds under three approximations, and solprop's operators lie
outside each in places.

- *Paraxial, weak loss, slow variation.* solprop's operator is $k_z(\kappa)=\sqrt{k^2-\kappa^2}$.
  - With a constant rate, the exact local symbol is
    $q^2 = \kappa_0^2 + i k_{z0}\alpha + \alpha^2/4$. The $\alpha^2/4$ term is negligible only
    while $\alpha \ll k_{z0}$.
  - With a varying rate, the square-root operator and the multiplication by $\alpha(x)$ do
    not commute, so the local ODE holds only to leading order in $\alpha'/(\alpha\kappa_0)$.
  - Replacing $k$ by $k_z$ below is a weak-loss, slowly varying approximation, not an exact
    non-paraxial theory.
- *Planar geometry.* On the radial grid, writing $E = u/\sqrt r$ leaves a term $1/(4r^2)$ in
  the order-zero radial equation (§4.6).
- *Distance from cutoff.* As $\kappa_0 \to k$, $k_z \to 0$: the ray picture fails, and SEP 21
  found the collar's flux gain unbounded there. Designs whose protected band approaches
  cutoff are refused (§5.2).

#### 4.2 Transmission: the strength

Where $k\alpha \ll \kappa_0^2$, WKB gives $\operatorname{Im} q \approx k\alpha/(2\kappa_0)$. A
component crossing the collar once is attenuated in field by
$\exp\!\big(-\tfrac{k}{2\kappa_0}\int\alpha\,dx\big)$. This is the ray result: the
transverse velocity is $dx/dz = \kappa_0/k = \tan\theta$, so the crossing takes the
propagation distance $w/\tan\theta$.

Light that crosses a radial collar meets the Dirichlet wall and crosses back. On a periodic
grid it crosses the image's collar instead, so a face of width $w$ is crossed on the way
out and the opposite face on the way in. Either way, the field returning to the aperture
is attenuated by

$$
\Gamma(\kappa_0) \approx \frac{k}{\kappa_0}\int_\mathrm{collar}\alpha\,dx
= \frac{k}{\kappa_0}\,\alpha_0\,w\,\langle P\rangle ,
$$

for a profile $\alpha(x) = \alpha_0 P\big((x-x_0)/w\big)$ with $0\le P\le1$ over a width $w$,
and $\langle P\rangle = \int_0^1 P\,ds$. $\Gamma$ falls as $1/\kappa_0$: fast, high-angle
content is the hardest to absorb.

**The strength is a root, not this formula.** Measured on the discrete operators, the
weak-loss formula overstates the attenuation at practical widths: $\Gamma(\kappa_p)$ is 7.3
and 8.1 at $\kappa_p w = 10$ and 20, against the 8.5 targeted. The local WKB integral with
the exact symbol, with $q^2 = \kappa^2 + i k_z\alpha + \alpha^2/4$,

$$
\Gamma_\mathrm{wkb}(\kappa;\alpha_0) = 2\int_\mathrm{collar}\operatorname{Im}q\,dx,
$$

matches the measurement to between $-1.3\,\%$ and $+0.04\,\%$. Translation therefore solves
$\Gamma_\mathrm{wkb}(\kappa_p;\alpha_0) = \Gamma_\mathrm{min}$, a monotone scalar root, at
$k_{\min}$. It is 1.06 times the formula's value for Planck at $\kappa_p w = 20$.

$k_{\min}$ is the smallest $k_z$ at $\kappa_p$ over the *requested* band and over the run's
medium path (§5.2). It is computed, not assumed to be the red edge, because a material's
dispersion need not put it there. §7.1 sets $\Gamma_\mathrm{min} = \log(5\times10^3)$.

**The propagation length does not appear.** Today's strength, $12.6/(L/20)$, halves when
the run doubles in length with the same beam and the same grid. A collar sized by $\kappa_p$
and $w$ absorbs the same band whatever $L$ is. So does the safety band of §4.4. The whole
resolved operator is therefore independent of $L$.

#### 4.3 Reflection: the width

WKB holds, and the reflection is small, where $q$ changes slowly on the scale of its own
wavelength, $|q'| \ll |q|^2$. Slow, low-angle content sees a steep wall and reflects. The
simple estimate from that condition, a threshold scaling as $\alpha_0^{1/3}$, fails: the
measured threshold has the Born onset law's exponent, $\alpha_0^{1/(n+2)}$ for $s^n$. The
design uses the measured law, for the wavenumber above which $|r| \le 2\times10^{-4}$:

$$
\kappa_\mathrm{lo}\,w = A\,\big(k_{\max}\,\alpha_0\,w^2\big)^{\gamma},\qquad
(A, \gamma) = (7.36,\ 0.182)\quad\text{(Planck)}.
$$

$k_{\max}$ is the largest $k_z$ at $\kappa_\mathrm{lo}$ over the requested band and path. How
fast the reflection falls above $\kappa_\mathrm{lo}$ depends on the profile's smoothness at
its onset (§4.5). Reflection at the band edge itself falls below $2\times10^{-4}$ only for
$\kappa_p w \ge 22.3$ on Planck, which is a floor the design enforces.

#### 4.4 The bands, the launch, and what decides accuracy

A collar absorbs to $\Gamma_\mathrm{min}$ on $\kappa_\mathrm{lo} \lesssim \kappa_0 \le \kappa_p$.
The width is the stronger lever, which is SEP 21's empirical finding ("width matters most;
weaker is better down to an optimum"), now with a measured law.

**The band's opening does not decide accuracy.** The first version of this SEP took an open
band, $\kappa_\mathrm{lo} < \kappa_p$, to mean the collar works. Phase 0 refuted that. On the
nonlinear witness, the collar's excess error over a wall at the same radius falls on one
curve in $\kappa_\mathrm{lo}/\kappa_\mathrm{rms}$, whatever the fraction, band edge or
strength: 0 at 1.0, $10^{-5}$ at 3.2, $2\times10^{-4}$ at 4.7, $2\times10^{-3}$ at 6.7, and
$1.5\times10^{-2}$ at 12. A design that opens its band by raising $\kappa_p$ is the least
accurate, because the strength rises while the beam's own content stays near
$\kappa_\mathrm{rms}$.

**The design check.** What matters is how much of the launch reaches the collar and how much
of that reflects. Translation estimates the launch's own returned energy,

$$
U_\mathrm{ret} = \sum_\kappa S(\kappa)\,T(\kappa)\,|r(\kappa)|^2 ,
$$

from the noise-free launch's angular spectrum $S$, the share $T$ of each $\kappa$ that reaches
the collar within $L$ (a linear estimate), and the calibrated reflection $r(\kappa)$ for the
profile, which phase 1 tabulates from the phase 0 calibration. Above the reflection budget
of §7.1 the design is refused, naming the fraction that meets it. This is where the run length
enters: it decides how much slow content reaches the edge, not the collar's strength.

- For a beam that fills its collar this reduces to $\kappa_\mathrm{lo} \le \kappa_\mathrm{rms}$,
  which is $w \gtrsim 20\,w_0$ for Planck with $c_b = 3$.
- On the nonlinear witness, where 1 % of the energy reaches the collar,
  $\kappa_\mathrm{lo} \lesssim 3\text{–}4.7\,\kappa_\mathrm{rms}$ suffices.
- The check sees only the linear launch. Content generated later is the monitor's to judge
  (§5.3).

**The real-space collar cannot cover the whole grid.** On the probe grid the Nyquist
wavenumber is $\pi/\Delta x \approx 10^6$ m⁻¹. Absorbing everything up to $0.9$ of it at
$w = 150$ µm needs $\alpha_0 \approx 10^4$ m⁻¹, which moves $\kappa_\mathrm{lo}$ above the
beam. A $k_\perp$ rate is exact and free, but it acts **throughout the aperture**, so it
cannot be put below any wavenumber that physical light in the aperture may carry. The
design therefore orders three wavenumbers, $\kappa_\mathrm{lo} < \kappa_p \le \kappa_s$:

- **the protected band**, $\kappa \le \kappa_p$: never subject to a $k_\perp$ rate, and
  absorbed at the edge to $\Gamma_\mathrm{min}$ with reflection below the budget. What the
  run must keep, including radiation the user expects to be *generated*, belongs here. The
  user declares it, or accepts the default derived from the launch (§5.2), which covers
  only the launched content's spread. A weak launched or generated signal is protected only
  if its angle is declared: the launch-derived band does not cover a weak tilted seed.
- **the numerical safety band**, $\kappa \ge \kappa_s$: the outer fraction $f_s$ of the
  grid's largest wavenumber on each axis, $f_s = 0.1$ by default as today. Content there is
  removed by the $k_\perp$ ramp wherever it is. The grid's one-half sizing rule
  ([Cartesian](../../guide/free-space/cartesian.md#grid-sizing-and-the-transverse-safety-ramp))
  puts all physical content below it.
- **the intermediate band**, $\kappa_p < \kappa < \kappa_s$: kept in the aperture, and
  absorbed at the edge only to $\Gamma_\mathrm{wkb}(\kappa)$. The resolved design reports what
  would return (below), and the monitor's returned share (§5.3) detects it.

A user who knows that no physical light lies above some $\kappa_d < \kappa_s$ may lower the
ramp's start to $\kappa_d$ by an explicit declaration (`discard_above`, §5.2), and so close
the intermediate band. It is never lowered by default.

**The report.** The design prints the attenuation with the exact symbol per frequency,
weighted by the launch's or the declared signal's far-field spectrum. On the weak-signal
witness this predicts the returned energy to 1–4 % over four decades. The simpler
$\Gamma_\mathrm{min}\kappa_p/\kappa$ is a bound about 2× high in energy, stated as a bound and
not as the report.

**The safety ramp's strength.** It is exact wherever the content is, because it rides the
linear operator. It is sized so that content in its upper half is attenuated by
$\Gamma_\mathrm{min} + 2$ nepers over the distance in which it would come back:

$$
\alpha_s = \frac{2(\Gamma_\mathrm{min} + 2)\,\kappa_{s,1/2}}{k_{\min}\,\ell},
$$

where $\ell$ is the collar's width $w$ when one is present, and the requested size when none
is: with no collar, the distance over which the content crosses the aperture and returns.
The 2 nepers are measured: inside the ramp, the collar's own wavenumber broadening defeats
the additive estimate $\Gamma + \Gamma_s$ by up to 1.7 Np. The design reports the shortfall
band as measured, not as a crossing point of the two estimates. Like the collar's, this
strength does not depend on $L$.

Content generated *inside* the collar sees at most half the round-trip attenuation, and under
7 % of it from the collar's inner half. A collar is not a place to generate light.

**Rectangular grids.** Each face is designed on the component normal to it ($\kappa_x$ for
the x faces, with width $w_x$; $\kappa_y$ for the y faces, with $w_y$), so a face's band and
reflection depend on its own normal component. A packet heading for a corner crosses both
collars and is attenuated by both. The safety ramp keeps SEP 22's maximum of the
normalised components, so each axis keeps its own safety band.

**Slab modes.** A slab mode is the Cartesian case with $k \to \beta_p$. The extremes run over
modes: a design on one mode alone leaves another at 0.9 of $\Gamma_\mathrm{min}$.

#### 4.5 Profiles

All are $0 \le P \le 1$ on $s\in[0,1]$ and zero for $s\le0$.

- **The Planck taper's complement,** $P = 1 - W$ (today's shape), is the default (Q3). It has
  the lowest edge reflection for $\kappa_p w \ge 50$, the fastest tail, a stretched
  exponential, and no reflection floor.
- **The polynomials** $P = s^n$, $n = 2, 3, 4$, are the classic CAPs [2, 3]. Their reflection
  has a power-law tail. $s^3$ and $s^4$ open the band at narrower widths, with floors of
  $\kappa_p w \ge 14.5$ and 16.2 (25.3 for $s^2$), but reflect up to $2\times10^{-4}$ at the
  periodic grid's wrap point, where they meet 1 with a finite slope. They are offered only
  with a smooth top.

**A fair comparison** holds fixed the width, the grid, the protected band and the attained
$\Gamma(\kappa_p)$. It compares the reflection floor over the protected band, the
attenuation over the intermediate band, and the steps taken under the step limits of §4.7.

#### 4.6 The radial grid

The Hankel basis is the Dirichlet eigenbasis of the disc, so a radial collar always has a
perfectly reflecting wall behind it, and $\Gamma$ counts the round trip. Two features of
the radial operator lie outside §4.1's planar model:

- **the curvature term.** $u = \sqrt r E$ satisfies
  $u'' + (q^2 + 1/(4r^2))u = 0$. Measured, the corrections are below 1.2 % wherever the band
  is open, so the planar constants apply.
- **the lowest columns.** The lowest QDHT column has
  $\kappa R_\mathrm{req} = j_{0,1}/(1+f)$, near-standing modes of the disc. They are not
  attenuated by their overlap with the collar: they are *reflected* into modes trapped in the
  aperture, and decay at 0.2–7 % of the overlap estimate. The resolved design reports these
  columns as reflected, and the protected band may not include them.

A $k_r$-diagonal safety ramp is exact on the radial grid, because the radial linear operator
is diagonal in $k_r$. Its largest wavenumber is $k_{r,\max} = j_{0,N}/R$ [8].

#### 4.7 The scheme, and the collar-crossing step limit

**A step must not let a packet cross the collar unseen.** Over an accepted step $h$, a
component at $\kappa$ moves $h\,\kappa/k_z$ across. If that exceeds the collar's width, the
diffraction part of the step can carry it through the collar and the wall, or around the
period, and back into the aperture. A collar evaluated only at the step's end never sees it.
On a linear problem the source defect is zero, so the error controller gives no warning either.

An independent paraxial Fourier calculation shows this for a factor applied once per step:

- *Grid.* 512 samples, periodic half-width 300 µm, requested half-width 200 µm.
- *Launch.* $\exp[-(x/40\,\mu\mathrm m)^2 + i\kappa_0 x]$ with $\kappa_0 = 0.2k$, at 800 nm.
- *Collar.* $P = s^2$ over $w = 100$ µm, with
  $\alpha_0 = 3\log(10^3)(\kappa_0/k)/w$.
- *Run.* $L = 3$ mm: one transverse circuit of the box.

| method | energy left in the aperture |
|---|---|
| exact generator (matrix exponential of diffraction plus collar) | $1.03\times10^{-6}$ |
| exact diffraction then the factor, 1 step | 1.0 |
| 4 steps | $7.4\times10^{-5}$ |
| 8 steps | $1.67\times10^{-6}$ |
| 16 steps | $1.14\times10^{-6}$ |
| 64 steps | $1.03\times10^{-6}$ |

This is one reason the split scheme is not offered (§10). The collar is an **in-stage rate**,
a term $-\alpha E/2$ in every stage, as today. The stages sample the trajectory, the embedded
error estimate sees the collar's action, it changes no integrator's history, and it needs no
first-same-as-last exception. Measured against the exact generator, adaptive in-stage
Runge–Kutta is exact at every tolerance, and the crossing limit below never binds it: the
controller's growth cap of 1.1 per step keeps it from crossing a collar unseen.

The limit is kept as a guard, for adaptive runs through `dz_max` and for fixed schedules:

$$
h \le c_\times\,\frac{w\,k_{\min}}{\kappa_p},
$$

- $c_\times = 0.5$ for the adaptive `dz_max` clamp, which on the §2 probe never bound a step.
- A fixed in-stage schedule is checked at $c_\times = 0.15$, because it is bound first by damping
  accuracy, $\alpha_0 h/2 \lesssim 1.3$.
- Content above $\kappa_p$ crosses faster. In the safety band the ramp is exact at any step. In
  the intermediate band, content above about $(0.75/c_\times)\,\kappa_p$ could in principle cross
  within a clamped step. The resolved design records this *unguarded* band. Measured under
  adaptive stepping, tightening the clamp to cover it bought nothing, and the monitor's returned
  share is the check.
- A fixed schedule that exceeds the limit is refused at translation, and the monitor's check
  interval is held within it.

**The stability limit binds `Multistep` in proportion to $\alpha_0$.** It is
$2r/\alpha_\mathrm{eff}$ (§3), with $r = 3.3$ under Runge–Kutta and $r = 0.0877$ under `Multistep`
at order 5, about 38 times shorter. The default variable-order `Multistep` partly escapes by
lowering its order.

At the suite's tolerance, on the nonlinear witness:

- on the strong designs ($f = 1$), which the design check refuses, it takes 1.2–1.5 times
  Runge–Kutta's wall time;
- on the wide, weak designs the check admits, it takes at most 1.2–2 times its no-collar steps,
  and runs in 0.34–0.37 of Runge–Kutta's wall time.

So `Multistep` stays the radial and planar default under a collar (Q4). Per step, the collar adds
about 57 % to Runge–Kutta's cost radially at $n_r = 128$, and about 14 % to `Multistep`'s. A fixed
`Multistep` schedule beyond the limit is refused, as today.

**Cost.** One transverse transform pair per stage on the Cartesian, line and planar routes, or
one dense $n_r\times n_r$ product per stage on the radial route, plus a norm reduction for the
removal measurement (§4.9). On the free routes the in-stage pass also performs a temporal
transform pair even when no temporal rate is applied in-stage. A spatial-only pass needs only
the transverse pair, and this SEP removes the needless pair.

#### 4.8 The error budget behind the design

For the target of $10^{-3}$ in the aperture's field norm (§7.1), the design takes:

- $\Gamma_\mathrm{min} = \log(5\times10^3)$, a returned field below $2\times10^{-4}$ of the
  incident;
- a reflection floor $|r| \le 2\times10^{-4}$, $4\times10^{-8}$ in power, over
  $\kappa_\mathrm{lo} \le \kappa \le \kappa_p$;
- the launch's predicted own return (§4.4) within the same reflection budget.

These are added in amplitude, because they can arrive together.

#### 4.9 Removal, passivity and the ledger

**The collar is not passive in the flux metric.** For multiplication by
$A = \operatorname{diag}(\alpha)$ on the nodes, and a nonlocal flux metric $M$ (SEP 34), the
rate of change of the flux is governed by the form $(MA + AM)/2$. Positive $A$ and $M$ do
not make it positive semidefinite. A two-node block with an interior node ($\alpha_i = 0$)
and a collar node has a zero diagonal entry beside an off-diagonal $\alpha_j M_{ij}/2$, and
so a negative determinant whenever $M_{ij}\ne0$. This is not a cutoff effect:

- *An independent check.* 64 Fourier nodes on a 600 µm interval,
  $M = F^\dagger\operatorname{diag}\big(\sqrt{1-(k_x/k)^2}\big)F$ at 800 nm, and a
  unit-peak quadratic collar outside 200 µm. The minimum generalised eigenvalue is
  $-1.25\times10^{-10}$, with every $|k_x|/k\le0.043$ (reproduced 2026-10-31).
- *SEP 21's radial witness* recorded $-2.9\times10^{-7}$.

So:

- **Admission, not assertion.** A requested collar's worst generalised eigenvalue, relative to
  its peak rate, is checked at translation against $-\varepsilon_P$, with
  $\varepsilon_P = 1.3\times10^{-4}$ (Q6). In practice it is a guard near cutoff: the worst
  realistic value is $-1.3\times10^{-5}$.
  - The full problem is not affordable: on a $256^2$ Cartesian grid it is a dense eigenproblem
    of order 65 536 at each frequency. A one-axis surrogate does *not* bound it (short by 1.5–9
    times), and matrix-free Lanczos is unreliable.
  - The admission value is computed on a *corner* surrogate, the axis metric at
    $k\,\kappa_{\mathrm{axis},\max}/\kappa_\mathrm{corner}$, solved densely at the smallest-$k$
    bin, the value being monotone in frequency. It bounds every grid tested with a margin of
    2–7, at 6–11 ms per axis at 256 nodes; the radial problem costs about 30 ms at $n_r = 256$.
- **Removal is signed.** `spatial_removed` may go negative. It is called "removed field
  norm", never "absorbed physical energy". It may be read as energy only for a paraxial
  beam far from cutoff, where the field norm and the flux agree (SEP 21 D16).

**The measure.** This SEP's monitor and ledger use the quadrature-weighted field norm $U$,
as `BoundaryMonitor.split_removal` does today. The flux-weighted ledger is the roadmap item
of D16, not claimed here.

**Attribution.** On an accepted step the treatments act in a fixed, documented order:

1. the temporal factor, $U_0 \to U_1$, credited $U_0-U_1$;
2. the hard band mask, $U_1 \to U_2$, credited to a separate mask entry.

An in-stage collar's removal acts within the stages, and is credited as the integral of its
rate, $\int\!\!\int \alpha |E|^2$, sampled at the monitor's checks and integrated by the
trapezoid rule, as for the in-stage temporal collar.

- Only accepted steps are credited.
- Totals persist at checkpoints. A resume continues them; a relaunched child starts at zero
  (SEP 21 D10).
- A plane that keeps no field still accumulates, because the reduction runs on the backend.
- `spatial_removed` is reported in J/m on the line route and in J elsewhere, in the units of
  `absorbed_power`'s channels' integrals, and separately from them. It is reported signed,
  never clipped at zero as the temporal monitor's removed fraction is today. With no collar it
  is zero.
- The safety ramp's removal is SEP 23's transverse channel of `absorbed_power`. That
  channel exists today only where a separable operator does (Cartesian, line and planar).
  It is extended to the radial ($k_r$) ramp, so that every removal of the field norm has a
  named entry.

### 5. Proposed interface

#### 5.1 The grid: placement

Every spatial grid takes `collar_fraction` in place of `window_factor`. The default is 0: the
requested size is the grid's size, with a wall or a wrap at it. A collar lies **outside** the
requested size on every geometry:

```text
RadialGrid(radius=300e-6, n_r=64)                         # wall at 300 um
RadialGrid(radius=300e-6, n_r=230, collar_fraction=2.6)   # wall at 1.08 mm; n_r counts every sample
FreeGrid(radius_x, n_x, radius_y, n_y, collar_fraction=0.5)
# periodic half-width = radius * (1 + collar_fraction)
```

`radius` is the aperture the user wants modelled: flat, with no absorption.

- The grid counts its samples and is built before the absorber is known, so a fraction cannot
  be applied implicitly. A requested collar on a grid with `collar_fraction=0` is refused,
  naming the fraction that would meet the design check. The guide gives 1.0 (radial) and 0.5
  (Cartesian, line, planar) as starting points for fast content (Q2).
- A fraction $f$ costs $(1+f)$ in radial samples and $(1+f)$ to $(1+f)^2$ in radial work
  (measured below), and $(1+f)^2$ in Cartesian memory.
- A fraction with no `SpatialAbsorber` is a larger bare grid with the same requested aperture.
- The window `W` keeps its role as the marker of the region outside the request.
- The other consumers of the window (§3) follow the placement. The coupling plane's radial
  resampling keeps the requested radius and its own collar fraction, rather than doubling an
  outer radius that now includes the collar. The coupling elements' boundary mask is the
  region outside the requested size.

Measured at fixed spacing, on a dedicated Intel Xeon Gold 6240, $f = 1$ costs 1.9–2.3 times
per step for the grid alone and 2.5–2.9 times with the collar, below $(1+f)^2 = 4$.

#### 5.2 The absorber: a specification, resolved at translation

The user gives a specification; translation resolves it into an immutable design.

```text
from solprop.boundaries import SpatialAbsorber

prop_radial(grid, space, medium, responses, pulse)      # the default: safety ramp and monitor
prop_free(..., spatial_absorber=SpatialAbsorber(angle=25e-3))        # a collar, protecting sin(theta) <= 25e-3
prop_free(..., spatial_absorber=SpatialAbsorber(kappa=2e5))          # a collar, protecting |kappa| <= 2e5 rad/m
SpatialAbsorber()                                       # a collar, band from the launch
SpatialAbsorber(angle=25e-3, discard_above=4e5)         # and remove everything above 4e5 rad/m
SpatialAbsorber(attenuation=math.log(1e4), profile="planck")
prop_free(..., spatial_absorber=False)                  # no safety ramp either; the monitor still runs
```

**`SpatialAbsorber`, the specification.** A frozen dataclass:

- `kappa` (rad/m) or `angle` (rad): the protected band. Neither given means "from the launch".
- `discard_above` (rad/m or `None`).
- `attenuation` $\Gamma_\mathrm{min}$ (field nepers, default $\log(5\times10^3)$).
- `profile`.
- `safety_fraction` $f_s$ (default 0.1).

The angle follows SEP 23's convention, $\sin\theta = c\kappa/\omega$, the angle in vacuum
behind a planar exit face, so a protected angle is a frequency-dependent band
$\kappa_p(\omega) = \omega\sin\theta/c$. The design uses its largest value on the requested
band.

With no `spatial_absorber` argument, a run resolves a ramp-only design with the default
`safety_fraction`.

**`SpatialAbsorberDesign`, the resolution.** Translation resolves the specification
against the grid, the space, the medium and the launch, in a fixed order of precedence:
explicit `kappa`, then `angle`, then the launch.

- **Launch-derived band.** $\kappa_p = c_b\,\kappa_\mathrm{rms}$, $c_b = 3$, where
  $\kappa_\mathrm{rms}$ is the launch's energy-weighted rms transverse wavenumber over active
  bins, computed on the noise-free launch before noise is added. For a Gaussian launch
  $\kappa_\mathrm{rms} w_0 = \sqrt2$ on $|\kappa_\perp|$, whatever the waist. The band's edge
  says what the collar absorbs; it does not by itself protect the launch, which is the design
  check's job (§4.4).
- **Wavenumbers.** $k_{\min}$ and $k_{\max}$ (§4.2–4.3) are the extremes of $k_z$ at the
  relevant $\kappa$ over the *requested* band, `inband_mask & active_mask`, or a declared band.
  Envelope and field grids carry active bins well outside the request, where the spectral
  collar is already removing content. Taking $k_{\min}$ there doubles $\alpha_0$ and the error,
  halves the stability limit, and makes the report wrong by $10^4$ in returned energy. Bins
  evanescent at $\kappa_p$ are refused, as is any such bin with $\kappa_p > 0.9\,k$.
  - On a route whose medium varies along $z$ (SEP 34's radial adiabatic transport), they are
    taken over the medium at the path's nodes, so that the entrance alone does not set them.
  - On the planar route, $k_z$ is each slab mode's longitudinal wavenumber, and the
    extremes run over modes.
- **The resolved design holds:**
  - the real-space rate on the nodes, if a collar was requested;
  - the safety ramp's rate on the spectral columns;
  - $\kappa_\mathrm{lo}$, $\kappa_p$ and $\kappa_s$, with the intermediate band's report (§4.4);
  - the launch's predicted own return (§4.4);
  - the unguarded band of §4.7;
  - the QDHT columns reflected below the band (§4.6);
  - the crossing step limit;
  - the passivity admission value of §4.9.

  All are recorded in the output.
- **Refusals.** A requested collar is refused, by name, naming the fraction or width that would
  pass, when:
  - the launch's predicted own return exceeds the reflection budget (§4.4);
  - its band edge is below the profile's floor (Planck: $\kappa_p w < 22$, §4.3);
  - its passivity value fails §4.9;
  - the grid has no room for it (`collar_fraction=0`).
- **When it is frozen and rebuilt.** The design is resolved once per translation and is
  immutable. It is part of the translated problem, so it follows that problem's sharing.
  - `with_input` keeps the design. A derived run shares its parent's linear operator,
    transform and step limits, and that sharing is what lets derived runs and batches reuse
    them. The precedent is the optmat rate bindings, which also stay bound to the original
    input. When the new launch's $\kappa_\mathrm{rms}$ exceeds the band the design was
    resolved for, the run warns.
  - A scan point builds and translates its own simulation, so it resolves its own design.
    Points that differ only in energy keep $\kappa_\mathrm{rms}$ and the same design, and so
    still share leaves in a lockstep batch.
  - Coupling a field to another grid or medium (SEP 31) resolves the child's own
    specification on its own grid. A child whose band is launch-derived is translated
    against its actual incident launch, not against the zero placeholder launch that sizes
    its port.
  - A launch that is zero everywhere, with a requested collar and no declared `kappa` or
    `angle`, is refused.
  - A resume translates the simulation again and resolves the same design. The design is
    recorded in the run's configuration and is part of its resume key, so any difference
    refuses the resume.

**Removed:**

- `TransverseCollar` and `SimulationSettings.transverse_absorber`;
- `Absorber.spatial`;
- the real-space use of `Absorber.collar_length`;
- `window_factor`.

`absorber=False` on a spatial interface keeps its SEP 21 meaning (no temporal or spectral
collar) and does not touch the spatial absorber.

#### 5.3 Monitoring

The monitor gains a transverse edge per face: one for the radial grid, four for the
Cartesian box, two for the line and the slab. Every share is an energy share of the launch's
field norm $U_\mathrm{launch}$, and is judged at Q7's levels through `BoundaryCheck`, as for the
temporal guards: a warning above its budget, $10^{-3}$ in field ($10^{-6}$ in energy), and a stop
above ten budgets.

What it judges depends on whether a collar is present, because the physics differs.

| situation | measured quantity | judged |
|---|---|---|
| launch clipped by the aperture | the launch's share in the guard (the outer 5 % of the requested size) at $z=0$ | warn above budget |
| **no collar:** light reaching the wall or the wrap | the guard's share, its maximum over checks above its launch value | warn and stop |
| **collar:** inadequate absorption, or contamination returning | the returned share in the return strip, and the inward current at the collar's onset | warn and stop |
| **collar:** intended outward flow | guard and collar shares along $z$; `spatial_removed` | recorded at INFO, never a stop |
| a design's limits | the intermediate band's report and the unguarded band, at translation | recorded |

**With no collar, reaching the edge is the failure.** Everything that reaches the edge comes back
or wraps, so no direction is needed. The direction-resolving estimators below read nothing useful
here: on the suite case, 10 % wrong in field, the outer quarter's strip reads $1.4\times10^{-7}$
in energy, below the warning, and the inward current reads zero, because a slow return coincides
with outflow. The guard's share tracks the true error instead. On that case it reads:

| z (cm) | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| true aperture error, energy | $6.1\times10^{-7}$ | $2.4\times10^{-4}$ | $3.6\times10^{-3}$ | $1.1\times10^{-2}$ |
| guard above its launch value | $7.7\times10^{-7}$ | $9.5\times10^{-5}$ | $3.0\times10^{-4}$ | $4.1\times10^{-4}$ |

It stays within 3× at onset, and falls behind by up to 30× only once the error is already large.
So the warning comes near a true error of $10^{-6}$ and the stop near 1–5 × $10^{-4}$. The launch
baseline stops a converging launch being stopped falsely.

- **The noise allowance.** The builders' default input noise fills the guard. Its expectation is
  removed by the launch baseline, and a reading is judged above three standard deviations of the
  guard's noise reading, $3\,\eta f_g/\sqrt{M}$, for noise share $\eta$, guard area share $f_g$ and
  $M$ independent modes in the guard. This matches one-frequency radial measurements at
  $\eta = 3\times10^{-6}$ to $10^{-3}$; §7.6 qualifies it on full grids.
- **The stop message names a size to try,** by the ray estimate $R\,L/z_w$, with $z_w$ the plane
  at which the guard first passed the warning. It is deliberately generous: on the suite case it
  gives about 3.8 $R$, where 2 $R$ already meets the budget. A user who knows better raises the budget
  with `boundary_check`.

**With a collar, reaching the edge is intended, and what returns is the failure.** The returned
share is the field near the edge that travels *towards* the aperture's interior.

- **Its strip.** A strip of width $s$ resolves direction to $10^{-6}$ leakage only for
  wavenumbers $q \ge 16/s$, an uncertainty limit. The strip runs over the collar plus at most the
  request's outer quarter.
- **The estimator.** The inward energy at $q \le -16/s$ of a Kaiser(8)-windowed transform along
  the face's normal, on $\sqrt r\,E$ radially. Its transform is padded twice.
- **The current.** It is supplemented by the cumulative rectified inward current at the collar's
  onset, accumulated at every accepted step, judged above the launch's energy outside the
  request.
- **The floors.** The strip's maximum over checks is judged above its launch-time value and above
  three times the exact noise expectation; the current is judged above its own. When three times
  the strip's floor exceeds the warning, translation records that the run is insensitive below
  it, and warns if it exceeds the stop, as on a Cartesian grid with explicit input noise at 1 nJ.

Measured in two dimensions: clean escape, through a Cartesian corner included, reads at most
$2.4\times10^{-8}$; every wall, wrap and collar at one sixteenth of the design strength stops; the
margin to the weakest stop-level failure is about $6\times10^3$. The reflected spread of a slow
launch is under-read 7–100 times, so it warns rather than stops. The design check is what keeps
that case out, and the guide says so.

The end-of-plane log of `collar_field_fraction` is retired.

#### 5.4 Composition

- *Scans and coupling:* the specification is plain data, resolved per translation, with
  the sharing and coupling rules of §5.2.
- *Single precision:* the rate is applied in the state's precision.
- *The low-storage march* (SEP 24) and *the Adams march* (SEP 16, SEP 39): in-stage (§4.7).
- *The planar route:* the real-space collar acts on y alone, per $(p, y)$ row, as today's
  y collar does. The safety ramp acts on $k_y$ alone, per $(p, k_y)$ column, which is
  diagonal there. The design uses the modes' $k_z$ (§5.2).
- *Refused:* a spatial absorber on a modal route.

### 6. Implementation strategies

**Layers.**

- *Physics* (`solprop.boundaries`, `solprop.grid`): the specification, the design formulas,
  the tabulated reflection and the profile library.
- *Translation* (`solprop.translate.radial`, `.free`, `.planar`, `.spatial`, `.problem`):
  resolution against medium and launch, the design check, the step limits, the passivity
  admission, and the rates on the kernel problem.
- *Backends:* the in-stage term (existing), the guard reduction, the returned-share and removal
  reductions.
- *Monitor* (`solprop.monitor`): the transverse edge and its policy.

**The default's cost is the guard reading.** A norm reduction at the monitor's checks, beside the
temporal guards' own. The ramp rides the linear operator and costs nothing per step. A default
run's steps and wall time should not change, which §7.7 checks.

**Folding the rate into the linear operator** is exact but not affordable. In the radial
basis, $F\,\mathrm{diag}(\alpha)\,B$ is dense and does not commute with the diagonal linear
operator. The cumulative operator would become a per-frequency matrix exponential, breaking
the diagonal contract of [kernel-contract.md](../kernel-contract.md) §3.

**The safety ramp** is a per-column rate in the linear operator, existing on the Cartesian
route. It is lifted to the radial ($k_r$) and planar ($k_y$ per mode) columns, with its
strength from §4.4.

**Kernel contract.** The problem gains the crossing limit and the radial and planar ramp columns.
`SCHEMA_VERSION` is bumped once, in phase 1. The output schema records the design and the ledger,
also in phase 1.

**Backends, in order.**

- `numpy_ref`: the oracle.
- `numba_cpu`: the existing in-stage kernels, the ramp columns and the monitor reductions.
- The native CPU build: the same.
- The MLX and native device backends follow the standing device policy
  ([backends.md](../backends.md)) and refuse what they do not yet implement, by name.

**Cost to be measured, not assumed (§7.7).**

- The in-stage cost per stage.
- The monitor's windowed transforms and reductions, with their scratch.
- The added samples of a requested collar.
- The steps under the crossing and stability limits.

Scratch is reused from the stage buffers where the backend already holds a physical-space
volume. Any additional buffer is listed in the memory ledger
([benchmarks.md](../benchmarks.md)).

**No new dependencies.**

### 7. Verification and validation

#### 7.1 Norms and budget

- **Norms.** Errors are field L2 norms over the requested aperture (and over its inner half,
  "core"), area-weighted, over all active frequencies, at stated planes. They are
  normalised two ways, both reported:
  - by the launch's norm over the aperture: the *primary* criterion, which a field that has
    mostly left cannot inflate;
  - by the reference's norm over the same region at the same plane.

  A weak signal is judged on its own norm: a launch-normalised error cannot see it. A run that
  corrupted a sideband by 1 % of its own field read only $1.1\times10^{-4}$ of the launch norm.
- **Coefficients.** Reflection and transmission are field amplitudes. The incident amplitude
  is the packet's before it reaches the collar. A returned packet is measured at the time it
  re-enters the guard, separated by its inward travel (§5.3).
- **Budget** for a requested collar's $10^{-3}$ target, added in amplitude:

  | item | budget |
  |---|---|
  | returned transmission, $e^{-\Gamma_\mathrm{min}}$ | $2\times10^{-4}$ |
  | reflection, the launch's own return included | $2\times10^{-4}$ |
  | transverse and temporal discretisation | $2\times10^{-4}$ |
  | step controller and crossing limit | $2\times10^{-4}$ |
  | reference error | $\le 10^{-4}$ |

#### 7.2 References

Phase 0 built and used these; they are carried into the tests or kept with the harness.

1. **An analytic linear reference.** A Gaussian beam's angular spectrum is Gaussian.
   Propagated exactly with $e^{ik_z z}$ and transformed back by adaptive quadrature,
   independently of the QDHT and the FFT, it gives the field over the aperture at any $z$,
   non-paraxial included. Accurate to about $10^{-12}$.
2. **Converged large-domain references** for the nonlinear cases, at a fixed physical spacing,
   each run's Fourier–Bessel series evaluated on a common physical grid. Outer radii 1.2 to
   9.6 mm on the suite case, converged to about $10^{-9}$.
3. **The 1D scattering problem** of §4.1, for WKB trends only.

#### 7.3 Calibration

Done in phase 0 on the real discrete operators (the periodic Fourier operator, the QDHT and the
slab-mode operator), recorded in Part II: the WKB root, the threshold law, the band-edge floors,
the radial and slab corrections, the safety band's shortfall, the content generated in the collar,
and $c_\times$. Phase 1 turns the reflection into the table $r(\kappa)$ of §4.4 and checks it
against the 1-D measurements on `numpy_ref`.

#### 7.4 Scheme accuracy

The aperture error at three step sizes in-stage, under each integrator. The §4.7 crossing
counterexample, run in-stage, as a regression test: refused, or accurate, never silently wrong.

#### 7.5 Physics witnesses

These run in the tests at reduced size. Their phase 0 full-length values are the target.

- **The linear beam.** The probe of §2, on a collar that passes the design check, against reference
  7.2.1, within §7.1's budget. The same beam on a collar that fails it is refused.
- **A protected weak signal,** a four-wave-mixing sideband at an angle in fused silica, on the line
  route at 256 samples (seconds). Protected: within $10^{-5}$ of the sideband's own field. Declared
  outside the band: the report predicts its returned energy to ±10 %. Wrapped: wrong by 0.97.
- **A nonlinear field reaching the boundary,** the suite's `spatial_absorber` case scaled paraxially
  to 1 cm, which reproduces every 8 cm error to two digits in 3–16 s. Against a converged
  large-radius reference: as configured, stopped by the monitor; on a requested collar at $f = 2.6$, within $10^{-5}$. A
  ten-step witness is not possible, because the error appears only once the beam fills the
  collar's region. The full 8 cm runs only in the suite.

#### 7.6 Monitor

One or a few frequencies on `numpy_ref`, asserting phase 0's readings within a factor of 3:

- *No collar:*
  - a run that stays away from the edge completes with no warning beyond INFO;
  - a wall and a wrap, slow and fast, are caught by the guard's share;
  - converging launches read zero until they truly spread back;
  - a launch with tails on the edge is caught at $z = 0$, and not again;
  - the noise allowance holds under each route's noise, on full frequency grids, with the check
    interval the monitor really uses.
- *A collar:*
  - a strong escape, through a Cartesian corner included, completes with no warning beyond INFO;
  - a wall, a wrap, and collars at one quarter and one sixteenth of the design strength are caught;
  - the converging launch needs its baseline;
  - the slow launch on a refused design, built by hand, reads 7–100 times under, and is recorded as
    an expected shortfall.

#### 7.7 Invariants, ledger and cost

- **The default changes nothing short of the edge.** A run whose light stays away from the edge
  takes the same steps as today's, within the accuracy envelope, and its field agrees within the
  ramp's effect at $\kappa_s$.
- **Inside the protected band, nothing changes.** A beam whose content lies in the
  protected band and stays inside the aperture is unchanged to rounding.
- **The ledger closes.** With temporal and spatial collars both active, the ledger closes:
  $U(z)$ plus the credits plus the mask entry equals $U_\mathrm{launch}$, within
  the step's error. The test runs in a lossless, linear medium. Otherwise the safety ramp
  (the transverse channel), the evanescent and material channels and any ionisation loss
  enter the closure too. Checked across a checkpoint and resume, and with field-free planes.
- **Passivity admission.** The admission values of §4.9, on each geometry.
- **Independence of $L$.** The resolved operator is the same for $L$ and $2L$; this is
  checked directly. Propagations to a common plane are compared on a matched fixed schedule,
  where they must agree to rounding. Adaptive runs are compared within the accuracy envelope
  of [testing.md](../testing.md) §3, not against `rtol`.
- **Parity.** `numpy_ref`, `numba_cpu` and the native CPU build on matched schedules and
  histories, within the summation-bound tolerances of [testing.md](../testing.md).
- **Qualification matrix.**
  - *Routes:* radial, square and anisotropic Cartesian, line, and planar with one and two
    modes.
  - *Grids:* field and envelope.
  - *Precision:* double, and single where the route supports it.
  - *Integrators:* RK, Multistep and low-storage RK.
  - *Backends:* each CPU backend.

  A cell is either qualified or refused by name, with a refusal test. The design's
  serialisation and resume round trip is tested.
- **Cost.** Steps, wall time and memory on the suite's spatial cases against today, at matched
  accuracy, with and without a requested collar, at several collar widths at fixed physical
  resolution. A ledger entry in [benchmarks.md](../benchmarks.md).

### 8. Documentation

- [Free-space model](../../guide/free-space/model.md): *Spatial collar and statistics* is
  rewritten as *Transverse boundaries*. It covers:
  - room first: why a wall farther out beats a collar for a slow beam, and how to size the grid;
  - what the monitor judges, what a stop means, and how to act on its message;
  - when a collar helps: fast content, declared with `angle` or `kappa`;
  - the design check and its refusal;
  - the theory in its calibrated form, the protected, intermediate and safety bands;
  - that a weak generated or launched signal is protected only if its angle is declared, and that
    a slow reflected spread under a collar warns rather than stops;
  - the step limit and the ledger;
  - a worked example.
- [Boundaries](../../guide/numerics/boundaries.md): links the transverse boundary from
  *Which runs carry one*, and updates the spatial row.
- [Cartesian](../../guide/free-space/cartesian.md): *Grid sizing and the transverse safety
  ramp* is updated for the safety band and the anisotropic box.
- [Grids](../../guide/setup/grids.md) and
  [spatial decompositions](../../guide/setup/spatial-decompositions.md): `collar_fraction`,
  the placement, and the planar ramp.
- A gallery tutorial: *Is my transverse window big enough?*, the spatial companion of the
  temporal one, built on the suite case's stop and the protected-signal witness.
- [Differences from Luna](../../guide/workflows/luna.md): Luna.jl's free-space routes apply
  no transverse absorber, so a radial run reflects from its outer radius and a Cartesian one
  wraps. So does solprop by default, but it detects that and stops, and offers a designed
  absorber on request.

### 9. Compatibility and impact

Breaking, deliberately, with no compatibility shims:

- `window_factor` becomes `collar_fraction`, defaulting to 0. `RadialGrid(R, n_r)` keeps its wall
  and spacing. A script that gave `FreeGrid(rx, nx, ry, ny)` loses the inert 10 % margin, so its
  spacing at the same `n_x` changes.
- `TransverseCollar`, `SimulationSettings.transverse_absorber` and `Absorber.spatial` are
  removed. The safety ramp's strength is no longer a free rate, and it is on by default.
- **A run whose light reaches its boundary now warns or stops** where it used to log and carry on.
  Results that were silently wrong become refusals, which is the purpose.
- An explicit absorber, which used to enable today's collar, is replaced by a `SpatialAbsorber`
  that may be refused.

Kernel `SCHEMA_VERSION` and output `SCHEMA_VERSION` are bumped in phase 1.

Validated results to redo:

- the suite's `spatial_absorber` case, which as configured would now stop. It gets a requested
  collar at $f = 2.6$ on $n_r = 230$, with a measured error of $2.9\times10^{-6}$, so that the
  suite keeps tracking the collar's cost path. Its revision is bumped. The stop itself is tested
  in §7.5, because a tracking case that stops part-way gives no timing;
- the radial, Cartesian, line and planar performance-tracking cases, for the same check;
- the free-space galleries, all of them: those that set a collar or a ramp
  (`plot_03_line`, `plot_04_planar_hybrid`, `plot_05_low_storage`,
  `plot_06_four_wave_mixing_scan`, `plot_07_far_field_statistics`), and those whose run may now
  warn or stop (`plot_01_radial`, `plot_02_cartesian`, `plot_08_radial_capillary_coupling`);
- the introduction's filament and crossing-beam tutorials;
- the geometry figure.

Downstream Cartesian models that run with a $k_\perp$ ramp and no real-space collar take the
default and re-qualify their numbers.

The device backends (MLX and the native GPU builds) run today's in-stage collar now, with no
refusal. Under the new design they either carry the new parts (the radial and planar
safety ramps, the transverse monitor edges) or refuse those parts by name.

### 10. Alternatives considered

- **A real-space collar by default** (this SEP as first accepted), at fractions of 1.0 radially and
  0.5 otherwise. Phase 0: radial $f = 1$ gives $1.1$–$4.4\times10^{-3}$ against a bare wall's
  $3.9\times10^{-5}$ on the suite case, and Cartesian $f = 0.5$ gives $3.6\times10^{-2}$ on the
  probe. It costs 2.5–2.9 times per step radially and buys nothing on a slow beam.
- **Outside placement as distance only, by default:** a bare wall at $(1+f)R$. It is robust on the
  suite case, but not within budget on the §2 radial probe ($1.1\times10^{-3}$), and it charges
  every run 2.1 times for room most do not need. The monitor asks for the room when it is needed.
- **A default fraction that follows the design,** resolved at translation: $f$ of 8–10 on the suite
  case, with an outer size no longer fixed by the grid.
- **Protecting less,** lowering the band to what the width supports. The band opens only by
  *raising* $\kappa_p$, and that measured worst.
- **The direction-resolving monitor on the default.** It reads below its warning on the suite
  case (§5.3), with no collar to give the strip its width. Widening the strip to the outer half
  reads $7.9\times10^{-6}$ on a beam whose true return is $3.0\times10^{-2}$.
- **A split real-space factor,** applied once per step, on `RungeKutta`. It was the first version's
  option, and is not built. It gains at most 1.6 times at matched accuracy, on the Cartesian and
  line routes at $10^{-3}$, about 1.0 at $10^{-4}$, and nothing radially. Its error is invisible to
  the controller: at $c_\times = 0.5$ it reached $3\times10^{-3}$ where the beam dwells in the
  collar, and holding $10^{-4}$ needed $c_\times$ of 0.25 to 0.06, depending on the launch. It is
  incompatible with the Adams and low-storage histories. Most of its potential, about 2–2.7 times on
  Cartesian at a tight tolerance, is the in-stage pass's own overhead, of which this SEP removes
  the needless temporal pair. It goes to the roadmap at Final, with the accuracy cap it would need.
- **A step limit covering the intermediate band** (`cover="intermediate"` in the first version).
  It costs +1–3 % of steps, +11 % at most, with a floor of $L\kappa_s/(c_\times w k_{z,\min})$ that
  grows on fine grids and long runs. Under adaptive stepping it bought nothing measurable,
  including on a $10^{-3}$-amplitude packet under a strong beam. It is not offered.
- **The singular transmission-free profile of Manolopoulos** [4]. It needs the split scheme, because
  in-stage its peak rate would set the step, and it was not measured.
- **Keep the collar, retune its constants.** SEP 21's sweep shows that width and strength
  alone reach a few $10^{-3}$ at best, and that the right strength depends on the beam, not
  on $L$. Retuning would leave the length coupling.
- **Perfectly matched layers** [7]: complex coordinate stretching $x \to x + i\sigma(x)/\omega$.
  A PML is reflectionless in the continuum for every angle. On a Fourier or Hankel spectral
  grid, though, the stretched Laplacian is no longer diagonal in the transform basis, so the
  linear step loses its exact diagonal propagator. Discretised PMLs on spectral grids also
  reflect at the grid level. Out of reach of the kernel contract without a different
  transverse discretisation.
- **Transparent boundary conditions** [6]: exact for finite-difference beam propagation with
  an outgoing plane-wave ansatz at the last node. They do not exist for a global spectral
  transform, whose boundary is periodic or Dirichlet by construction.
- **A launch-derived $k_\perp$ ramp** (the first draft's default). It damps legitimately
  generated light throughout the aperture, before it reaches any edge. The ramp is confined
  to the numerical safety band unless the user declares otherwise.
- **A grid-only real-space design** (cover the whole band below the safety ramp with the
  collar). §4.4 shows it moves $\kappa_\mathrm{lo}$ above the beam at practical widths. It
  is available by declaring a large `kappa`, not as the default.

### 11. Phases

**Phase 0 — calibration and measurement.** Complete, 2026-10-31 and 2026-11-01, reported in
Part II. Its evidence led to this revision.

**Phase 1 — `numpy_ref`, interface and documentation.**

- `collar_fraction`, with the outside placement.
- `SpatialAbsorber` and its resolution, the design check with the tabulated reflection, and the
  removals of §5.2.
- The crossing limit.
- The radial and planar safety ramps, and the no-collar ramp strength.
- The monitor's transverse edges, in both regimes, and the ledger.
- Both schema bumps.
- The tests of §7 on `numpy_ref`, the qualification matrix's `numpy_ref` column, the manual
  pages and the tutorial.
- The other backends refuse what they lack, by name.

**Phase 2 — `numba_cpu`.** The same, with parity on matched schedules, and without the free routes'
needless temporal pair.

**Phase 3 — native CPU, requalification, Final.**

- The compiled paths, with parity.
- The suite's spatial cases, the galleries, the benchmark ledger and the Luna page.
- Roadmap items: the split scheme, with its measured potential and accuracy cap; device backends,
  left to the standing policy.

### 12. Open questions

All are decided (John Travers). Q1–Q7 were first decided on 2026-10-31. Phase 0 changed Q1, Q2 and
Q4, on 2026-10-31 and 2026-11-01, and this revision states them as they now stand. Part II
records what changed and why.

1. **Q1 — the default.** *Decided:* no real-space collar. `prop_radial`, `prop_free` and
   `prop_planar` keep a wall or a wrap at the requested size, with the safety ramp and the monitor.
   A `SpatialAbsorber` is on request. `spatial_absorber=False` also removes the ramp.
2. **Q2 — the collar fraction.** *Decided:* the grid's default is 0. The first version's 1.0
   (radial) and 0.5 (Cartesian, line, planar) become the guide's starting points for a requested
   collar. They are not applied implicitly, because a grid counts its samples and is built before
   the absorber (§5.1).
3. **Q3 — the profile.** *Decided:* the Planck taper's complement, confirmed by phase 0. Polynomials
   with a smooth top are offered. The singular profile is dropped.
4. **Q4 — the scheme and the integrator.** *Decided:* in-stage only, keeping `Multistep` as the
   radial and planar default under a collar. The split goes to the roadmap, and `cover` is dropped
   (§10). Whether SEP 21's split temporal collar needs the analogous crossing limit is already on
   the roadmap.
5. **Q5 — the default protected band.** *Decided:* the launch times $c_b = 3$, as the band's default
   upper edge. The design check, not the band, protects the launch. The guide tells users who expect
   generated radiation (four-wave mixing, conical emission) to declare an `angle`.
6. **Q6 — the safety band and passivity tolerance.** *Decided:* the outer 10 % of each axis's
   wavenumbers, on by default on every geometry, radial and planar included, with the strength
   rule of §4.4. $\varepsilon_P = 1.3\times10^{-4}$, phase 0's worst measured value with a margin
   of ten.
7. **Q7 — the monitor's policy.** *Decided:* warn above $10^{-3}$ of the launch's field norm and
   stop above $10^{-2}$, in both regimes of §5.3, through `BoundaryCheck`.

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

Only bibliographic identity has been checked for these references (2026-10-31). No profile,
coefficient or optimum is transcribed from them: the design's constants are this SEP's own
measurements on the discrete operators (§7.3).

## Part II — Implementation record

### Phase 0 — interim report (2026-10-31)

Phase 0 has run its operator calibration (§7.3), the linear probe against an analytic
reference (§7.2.1, §7.4), the integrator and crossing measurements (A1, §4.7) and the
monitor prototypes (§4.9, §5.4). Still to run:
- the protected weak-signal witness and the nonlinear collar-occupying witness (§7.5);
- the monitor's two-dimensional witnesses (§7.6);
- the step cost of `cover="intermediate"`;
- the split against in-stage at matched accuracy;
- timings on a quiet Linux machine (A11).

The campaign harness and its raw results are kept with the project's development records
(SEP 30's rule). Every number below is reproducible from them. Conventions are §4's: α is a
power rate, Γ is in field nepers, r is a field amplitude. P0-1 to P0-12 are the
recommendations for the maintainer's decision; Part I is not edited.

**Reference.**
- The analytic angular-spectrum reference (§7.2.1) uses the code's own exact longitudinal
  operator at continuous transverse wavenumbers and adaptive quadrature. It is accurate to
  about $10^{-12}$, and large collar-free runs agree with it to $2\times10^{-12}$ (radial)
  and $7\times10^{-12}$ (Cartesian, 600 µm).
- §2's Cartesian reference, 300 µm wide, was itself wrong by $5.5\times10^{-4}$.
- §2's aperture errors reproduce to two digits. Its small radial core errors were mostly the
  default input noise, $1.3\times10^{-3}$ of the launch norm.
- Runge–Kutta and `Multistep` give identical errors, so every error in the table is the
  boundary's.

**Calibration on the discrete operators.** Measured on the periodic Fourier operator (one
face; direction-resolved packets, Richardson-converged), the code's QDHT (steady state,
incoming and outgoing Hankel fit) and the slab-mode operator.

- **The strength (§4.2).** The $1/\kappa$ law and $k\to k_z$ hold. The weak-loss formula
  over-states the attenuation at practical widths: Γ(κp) is 7.3 and 8.1 at $\kappa_p w = 10$
  and 20, against the 8.5 targeted. The local WKB integral with the exact symbol
  $q^2 = \kappa^2 + i k_z\alpha + \alpha^2/4$ matches the measurement to between $-1.3\,\%$
  and $+0.04\,\%$.
  - **P0-1.** Resolve $\alpha_0$ by solving $\Gamma_\mathrm{wkb}(\kappa_p;\alpha_0) =
    \Gamma_\mathrm{min}$ with the exact symbol, a monotone scalar root. It is 1.06 times the
    formula's value for Planck at $\kappa_p w = 20$.
- **The reflection threshold (§4.3).** The $1/3$ law fails.
  - The $2\times10^{-4}$ threshold scales as $\alpha_0^{0.18}$ for the Planck complement, and
    as $\alpha_0^{1/(n+2)}$ for $s^n$, which is the Born onset law.
  - In §4.3's form, $C_P$ is 3.0 for Planck (2.9, 1.5 and 1.4 for $s^2$, $s^3$, $s^4$), not
    about 1.
  - Reflection at the band edge falls to $2\times10^{-4}$ only for $\kappa_p w \ge 22.3$
    (Planck), 25.3, 14.5 and 16.2.
  - **P0-2.** Replace the $\kappa_\mathrm{lo}$ estimate by the measured threshold law
    $\kappa_{2\times10^{-4}} w = A(k_{\max}\alpha_0 w^2)^\gamma$, with $(A, \gamma) =
    (7.36, 0.182)$ for Planck. Refuse a design whose band edge does not meet the floor
    (Planck: $\kappa_p w \ge 22$). With $C_P = 1$, §5.2's refusal admits failing designs.
- **The profile (Q3).**
  - Planck has the lowest edge reflection for $\kappa_p w \ge 50$, the fastest tail (a
    stretched exponential), and no floor.
  - $s^3$ and $s^4$ open the band at narrower widths but reflect up to $2\times10^{-4}$ at
    the periodic grid's wrap point, where they meet 1 with a finite slope.
  - **P0-3.** Q3 is confirmed: Planck by default, with a polynomial only with a smooth top.
    The singular profile was not measured.
- **The launch band (Q5).** For a Gaussian launch $\kappa_\mathrm{rms} w_0 = \sqrt2$ on
  $|\kappa_\perp|$, whatever the waist.
  - **P0-4.** $c_b = 3$. The beam's own returned field is $5.5\times10^{-6}$, and
    $1.2\times10^{-4}$ of its energy lies outside the band. With Planck, the band then opens
    for $w \ge 5.3\,w_0$.
- **The default fraction (Q2).** That condition is not met on the §2 probe:
  - radially ($w = 150$ µm, $\kappa_p w = 21$) the edge reflection is $2.5\times10^{-4}$,
    marginal;
  - on the Cartesian grid at $f = 0.5$ ($w = 75$ µm, $\kappa_p w = 10.6$) it is
    $2\times10^{-2}$.

  The monitor's one-dimensional runs independently found the designed $f = 0.5$ Cartesian
  collar returning $2\times10^{-4}$ to $5\times10^{-2}$ of the launch energy. On the radial
  probe, a bare wall at twice the requested radius (1.1e-3) was as accurate as the AN 30
  collar on the same grid (1.6e-3). See P0-5.
- **The radial grid (§4.6).**
  - Curvature corrections are below 1.2 % wherever the band is open, so the planar constants
    apply.
  - The lowest QDHT columns are *reflected* into modes trapped in the aperture. They decay
    at 0.2–7 % of the overlap estimate, not at it.
  - **P0-6.** The design reports these columns as reflected, not as attenuated by overlap.
- **The safety band (§4.4).**
  - The shortfall band is the whole intermediate band, with a weakest attenuation of
    $\Gamma_\mathrm{min}\kappa_p/\kappa_s$, as low as 0.6 Np.
  - Inside the ramp, the collar's own wavenumber broadening defeats additivity: up to 5.6
    times the predicted return, or 1.7 Np.
  - **P0-7.** Raise the ramp's strength by 2 Np over the §4.4 rule, and report the shortfall
    band as measured.
- **Content generated in the collar** sees at most half the round-trip attenuation, and under
  7 % of it from the collar's inner half.
- **Slab modes.** A slab mode is the Cartesian case with $k \to \beta_p$. Taking the extremes
  over modes is right; a design on one mode alone leaves another at 0.9 of
  $\Gamma_\mathrm{min}$.

**Integrators and the crossing limit.**

- **A1: collar stability binds `Multistep`.**
  - On the §2 rows its steps rise from 16 to 87–145. The median step is 0.9–1.6 times the
    order-5 limit $2r/\alpha_\mathrm{eff}$.
  - It is still 2.4–6.6 times cheaper in wall time than Runge–Kutta there (contended A/B).
  - On a strong collar at high angle (line route) it needs 830–3016 steps against Runge–Kutta's
    106–303, and 4–8 times the wall time.
  - **P0-8.** Q4 returns to the maintainer. Phase 0 recommends deciding after the nonlinear
    witness. The options are to keep `Multistep` (in-stage) as the radial and planar
    default, or to switch those routes to Runge–Kutta when a real-space collar is active.
- **c× (§4.7), line route against the exact generator.**
  - **Split.** Exact for a per-step displacement up to $0.5\,w$. It fails at $0.75\,w$
    (1.3–3 times the generator's return) and lands near 1 at $1.0\,w$, so the response is
    resonant.
  - **In-stage Runge–Kutta, adaptive.** Exact at every tolerance, and the crossing limit
    never binds.
  - **In-stage Runge–Kutta, fixed steps.** Bound first by damping accuracy,
    $\alpha_0 h/2 \lesssim 1.3$, which gives an effective $c_\times$ of 0.15–0.2.
  - **In-stage `Multistep`, fixed steps.** Refused by stability up to 256–512 steps, and
    inaccurate at its first admitted schedules.
  - **P0-9.** $c_\times = 0.5$ for the split and for the adaptive `dz_max` clamp. A fixed
    in-stage schedule is checked at 0.15.
- **A11: cost of $f = 1$ radially.** 2.0–2.9 times per step on a shared Apple M5 Max (contended),
  including the collar term, below the $(1+f)^2 = 4$ of the dense products alone. To be
  re-measured on Linux.

**Monitor.**

- **Passivity admission (§4.9, A5).**
  - The amended surrogate, one axis at the smallest $k_z$, does **not** bound the full
    two-dimensional value: it is short by 1.5–9 times.
  - Matrix-free Lanczos is unreliable: it fails to converge, or returns 2–5 times too small
    a value.
  - A *corner* surrogate does bound every grid tested, with a margin of 2–7: the axis
    metric at $k\,\kappa_{\mathrm{axis},\max}/\kappa_\mathrm{corner}$, solved densely. It
    costs 6–11 ms per axis at 256 nodes; the radial problem costs about 30 ms at
    $n_r = 256$.
  - The value is monotone in frequency, so only the smallest-$k$ bin is needed. The split
    map's gain is the generator value times $\alpha_\mathrm{peak}h$.
  - The worst realistic value is $-1.3\times10^{-5}$, near cutoff.
  - **P0-10.** Use the dense corner surrogate at one frequency. Set
    $\varepsilon_P = 1.3\times10^{-4}$ (Q6's rule). In practice it is a guard near cutoff.
- **The returned share (§5.4, A3).**
  - A strip of width $s$ resolves direction to $10^{-6}$ leakage only for wavenumbers
    $q \ge 16/s$. This is an uncertainty limit, so "resolution well below
    $\kappa_\mathrm{lo}$" cannot be met. The design-sized strip reaches about
    $1.7\,\kappa_\mathrm{lo}$.
  - The recommended estimator is the inward energy at $q \le -16/s$ of a Kaiser-windowed
    transform along the face's normal (on $\sqrt r\,E$ radially). The strip runs over the
    collar plus at most the request's outer quarter. Its maximum over checks is judged
    above its launch-time value and above three times the exact noise expectation.
  - It is supplemented by the cumulative inward current at the collar's onset, judged above
    the launch's energy outside the request.
  - Clean escape reads at most $2\times10^{-7}$ of the launch energy. Every wall, wrap and
    under-strength run whose true return exceeds $10^{-4}$ reads above it.
  - Slow content below the resolved band is under-read by 15–300 times: it warns, but does
    not stop.
  - **P0-11.** Adopt this estimator. Q7's thresholds as decided, a warning above $10^{-3}$
    and a stop above $10^{-2}$ in field, are consistent with these measurements, pending
    the two-dimensional witnesses.
- **P0-12.** A converging launch with tails near the edge is added to §7.6's monitor tests:
  without the launch baseline it would be stopped falsely.

**P0-5, the decision this report most needs.** A fixed collar fraction cannot guarantee that
the band opens, because the width the band needs, $w \gtrsim 22/\kappa_p$, scales with the
launch, not with the request. Three ways forward:
1. Keep Q2's fractions, and refuse a design whose band does not open, naming the fraction
   needed. This is Part I's rule with the corrected test, and it refuses the §2 Cartesian
   probe at $f = 0.5$.
2. Keep the fractions and protect less: lower $\kappa_p$ to what the width supports, and
   warn that the launch's own spread is then partly reflected.
3. Let the default fraction follow the design ($w = 22/\kappa_p$ for Planck), resolved at
   translation, so the outer size is no longer fixed by the grid alone.

Phase 0 leans to the first, consistent with a robust default that refuses rather than
degrades. The decision waits for the nonlinear witnesses.

### Phase 0 — final report (2026-10-31)

The remaining measurements have run:
- the protected weak-signal witness (§7.5);
- the nonlinear collar-occupying witness (§7.5), the suite's `spatial_absorber` case
  against a converged reference;
- the monitor's two-dimensional witnesses (§7.6);
- the step cost of `cover="intermediate"`;
- the split against in-stage at matched accuracy;
- the cost of $f = 1$ radially on a quiet Linux machine (A11).

Two of the interim report's recommendations change: P0-4 and P0-5. The other ten stand.
P0-13 to P0-20 are new. Together with Q1, Q4 and A10, they are for the maintainer's
decision before phase 1.

Machines:
- the nonlinear and cover sweeps ran under Slurm on shared AMD EPYC 7763 nodes, so their
  wall times are same-job ratios;
- the line-route and monitor runs ran on a shared Apple M5 Max (contended);
- A11 ran on a dedicated Intel Xeon Gold 6240.

As before, every number is reproducible from the campaign harness.

**The nonlinear witness: distance beats absorption for a slow beam.**
- *Set-up.* The suite's `spatial_absorber` case: radial, requested radius 300 µm, 8 cm, a
  100 µm Kerr beam at $0.9P_\mathrm{cr}$, field grid 500 nm–1.6 µm.
  - Every variant keeps the radial spacing fixed.
  - The reference is collar-free at outer radii 1.2 to 9.6 mm. It is converged to about
    $10^{-9}$, at the controller's own run-to-run noise.
  - The beam does reach the collar: $1.1\times10^{-2}$ of its energy lies beyond 300 µm by
    8 cm.
- *Aperture errors at 8 cm*, launch-normalised; reference-normalised values are within 1 %:

  | variant | error | Runge–Kutta steps | `Multistep` steps |
  |---|---|---|---|
  | the suite case as configured: wall at the requested radius, no collar | $1.0\times10^{-1}$ | 71 | 149 |
  | today's collar inside the request | $9.9\times10^{-2}$ | 416 | 470 |
  | bare wall at $2R$ | $3.9\times10^{-5}$ | 68 | 68 |
  | design, $f = 1$, $c_b = 3$ (band closed) | $2.1\times10^{-3}$ | 71 | 174 |
  | design, $f = 1$, $\kappa_p$ raised until the band opens | $4.4\times10^{-3}$ | 86 | 503 |
  | design, $f = 2.6$, band just open | $2.9\times10^{-6}$ | 67 | 107 |
  | bare wall at the same outer radius as $f = 2.6$ | $6.3\times10^{-9}$ | 67 | 68 |
  | design, $f = 10$, $\kappa_{2\times10^{-4}} = \kappa_\mathrm{rms}$ | $9.8\times10^{-10}$ | 66 | 68 |

  Steps are at $\mathrm{rtol} = 10^{-7}$; under `Multistep`, rejected steps are included.
  Runge–Kutta and `Multistep` agree to every digit above their controller floors, so every
  error is the boundary's.
- *No designed collar beat a bare wall at the same outer radius*, across a scan of
  fractions 0.5–10, $c_b$ 1–3 and strengths 0.01–1 times the design. The best was within
  10 % of its wall. The §2 probes agree in all three geometries:
  - radial: $2.8\times10^{-3}$ against a wall's $1.1\times10^{-3}$;
  - Cartesian, $f = 0.5$: $3.6\times10^{-2}$ against a wrap's $2.9\times10^{-2}$;
  - Cartesian, $f = 1.2$, band open: $6.4\times10^{-4}$ against $7.4\times10^{-5}$.
- *What decides accuracy is $\kappa_{2\times10^{-4}}/\kappa_\mathrm{rms}$, not whether the
  band opens.*
  - The collar's excess error over its wall falls on one curve in this ratio, whatever the
    fraction, $c_b$ or strength: 0 at 1.0, $10^{-5}$ at 3.2, $2\times10^{-4}$ at 4.7,
    $2\times10^{-3}$ at 6.7 and $1.5\times10^{-2}$ at 12.
  - At fixed width the excess grows in proportion to $\alpha_0$.
  - A design that opens its band by raising $\kappa_p$ is the least accurate: the
    strength rises while the beam's own content stays near $\kappa_\mathrm{rms}$.
- *The monitor's witnesses reproduce this independently.*
  - A Gaussian waist on a collar designed by P0-1, P0-2 and P0-4, with its band open,
    diffracts into the collar within $L$. It returns $10^{-4}$ to $6\times10^{-3}$ of its
    energy on both grids, because its bulk lies below $\kappa_{2\times10^{-4}}$.
  - P0-4's "returned field $5.5\times10^{-6}$" counted attenuation only, not this
    reflection.
  - A one-dimensional width scan returns $3.4\times10^{-3}$, $1.3\times10^{-4}$,
    $3.8\times10^{-9}$ and $3\times10^{-14}$ at $f = 0.6, 0.9, 2, 4$.

**The weak-signal witness: the collar's value is in fast content.**
- *Set-up.* A line-route sideband from four-wave mixing in fused silica, at
  $1.5\times10^5$ rad/m and $10^{-6}$ of the pump's energy, leaving a 500 µm aperture
  through a 350 µm collar.
  - Fused silica's normal dispersion phase-matches the angle. In argon, the sideband at the
    mirrored angle cannot be phase-matched.
  - The pump never reaches the collar, so returned strong light in the sideband's band is
    not tested here.
- *Results:*
  - *Protected design:* the sideband agrees with a 4× wider reference to $4\times10^{-6}$ of
    its own field. That is the onset reflection, within 2× of the calibrated operator's.
    Its far-field energy agrees to $10^{-6}$ while it crosses the aperture's edge.
  - *The same box with the collar inert (the wrap):* 0.98.
  - *Intermediate design (the sideband declared outside the band):* it returns
    $1.03\times10^{-4}$ of the sideband's energy.
- *The report a design should print.* The WKB attenuation with the exact symbol, at each
  bin's $k_z$ and weighted by the signal's far-field spectrum, predicts the returned energy
  to 1–4 % over four decades: $1.02\times10^{-4}$, $1.5\times10^{-8}$ and
  $1.3\times10^{-10}$. §4.4's $\Gamma_\mathrm{min}\kappa_p/\kappa$ is a bound about 2× high
  in energy, not a report.
- *Launch-normalised error cannot see a weak signal.* The intermediate run corrupts the
  sideband by 1 % of its own field, which is only $1.1\times10^{-4}$ of the launch norm.

**The wavenumber extremes (§5.2) are taken over the wrong bins.**
- Envelope and field grids carry active bins well outside the request. On the witnesses'
  grids these reached 1818 nm, 1877 nm and 3.1 µm, where the spectral collar is already
  removing content.
- Taking $k_\mathrm{min}$ there doubles $\alpha_0$. That doubles the error on every
  witness, halves the `Multistep` stability limit and the crossing clamps, and costs
  15–28 % more Runge–Kutta steps.
- It also closes a band that is open on the request, and makes the intermediate report
  wrong by $10^4$ in returned energy: 4.4 Np printed against 9.3 Np real.

**Q4: the default integrator.**
- On the nonlinear witness, collar stability binds `Multistep` in proportion to
  $\alpha_0$:
  - fixed-order `Multistep` sits exactly on $2r/\alpha_\mathrm{eff}$, with 155–1350 steps
    against 73 with no collar;
  - the default variable-order `Multistep` partly escapes by lowering its order;
  - Runge–Kutta, whose limit lies above its accuracy step, is not bound at all.
- At the suite's tolerance:
  - on the strong designs (the $f = 1$ rows), `Multistep` takes 1.2–1.5 times Runge–Kutta's
    wall time;
  - on the wide, weak designs that accuracy needs, it takes at most 1.2–2 times its
    no-collar steps and runs in 0.34–0.37 of Runge–Kutta's wall time.
- Per step, the collar adds about 57 % to Runge–Kutta's cost radially at $n_r = 128$, and
  about 14 % to `Multistep`'s.

**`cover="intermediate"` (A4).**
- *The clamps:*
  - the protected clamp ($\kappa_\times = \kappa_p$) never bound an adaptive step on the §2
    probes;
  - the intermediate clamp costs +1–3 % of steps at in-band designs, and +11 % at the most
    at the over-strong ones.
- *What it buys:* under adaptive in-stage Runge–Kutta and `Multistep` it bought nothing
  measurable on the line route, including a $10^{-3}$-amplitude packet under a strong beam.
  The controller's growth cap of 1.1 per step keeps it from crossing a collar unseen.
- *Its cost floor:* $N \ge L\kappa_s/(c_\times w k_{z,\min})$, which grows on fine grids
  and in long runs.

**The split against in-stage (A10).**
- *The split's error is invisible to the step controller.*
  - At $c_\times = 0.5$ on a variable schedule it reached $4\times10^{-4}$, and
    $3\times10^{-3}$ where the beam dwells in the collar.
  - Holding $10^{-4}$ needed $c_\times$ of 0.25 to 0.06, depending on the launch.
- *Its cost advantage at matched accuracy* is about 1.3–1.6 times on the Cartesian and line
  routes under Runge–Kutta at $10^{-3}$, and about 1.0 at $10^{-4}$ with FSAL on. It is
  none radially, and the split is refused under `Multistep`.
- *Transform counting.* The in-stage pass on the free routes always performs a temporal
  transform pair, even when no temporal rate is applied in-stage. A spatial-only pass needs
  only the transverse pair, which is a safe saving.

**The monitor in two dimensions (P0-11, P0-12).** One frequency, Cartesian and radial.
- *What it catches:*
  - clean escape, through a Cartesian corner included, reads at most $2.4\times10^{-8}$;
  - every wall, wrap and collar at one sixteenth of the design strength stops;
  - every quarter-strength collar whose true return exceeds $10^{-4}$ stops;
  - the margin to the weakest stop-level failure is about $6\times10^3$;
  - the launch baseline is necessary and sufficient for converging launches;
  - the guard catches clipped launches at $z = 0$.
- *What it does not:*
  - the reflected spread of a slow launch (above) is under-read 7–100 times, so it warns
    rather than stops;
  - with explicit input noise on a Cartesian grid at 1 nJ, the strip's floor
    ($1.5\times10^{-4}$) lies above the stop, and the rectified current gives a false
    warning on every clean run;
  - at the radial default noise the warning level is effectively 1–2 × 10⁻⁶ rather than
    $10^{-6}$.

**A11 on Linux.**
- *Set-up:* Intel Xeon Gold 6240, an exclusive job, fixed schedules, the median of five.
- *Results:* at fixed spacing, $f = 1$ costs 1.9–2.3 times per step for the grid alone,
  and 2.5–2.9 times with the collar. That is below $(1+f)^2 = 4$, as on the Apple M5 Max.
- *Threads:* the per-step times were the same at 1 and 18 threads at these sizes.

**Recommendations.**

- **P0-13. The wavenumber extremes.** Take $k_\mathrm{min}$ and $k_\mathrm{max}$ over the
  requested band, `inband_mask & active_mask` (or a declared band), not over every active
  bin. *Amends* §5.2.
- **P0-14. The design check.** Replace "the band opens" by an estimate, at translation, of
  the launch's own returned energy:
  - Inputs: the launch's angular spectrum (taken from the noise-free launch, as §5.2
    already does), the share of each $\kappa$ that reaches the collar within $L$, and the
    calibrated reflection $r(\kappa)$ for the profile. Phase 1 tabulates $r(\kappa)$ from
    the phase 0 calibration.
  - Refuse above the reflection budget, naming the fraction that meets it.
  - For a beam that fills its collar this reduces to
    $\kappa_{2\times10^{-4}} \le \kappa_\mathrm{rms}$, which is
    $w \gtrsim 20\,w_0$ for Planck with $c_b = 3$. On the nonlinear witness, where 1 % of
    the energy reaches the collar, $\kappa_{2\times10^{-4}} \lesssim 3\text{–}4.7\,
    \kappa_\mathrm{rms}$ suffices.
  - This *replaces* P0-4's claim that $c_b = 3$ protects the launch. $c_b = 3$ stays as the
    band's default upper edge.
- **P0-15. Q1 and P0-5 return to the maintainer: no default fraction meets §7.1's budget
  on a slow beam.**
  - Q1 makes the default collar conditional on the witnesses meeting the budget. At Q2's
    fractions they do not:
    - radial $f = 1$: $1.1\text{–}4.4\times10^{-3}$, against a bare wall's
      $3.9\times10^{-5}$;
    - Cartesian $f = 0.5$: $3.6\times10^{-2}$.
  - A collar wide enough to pass is no better than a wall at the same radius, at up to 30
    times the per-step cost: $n_r = 704$ against 64.
  - The P0-5 options:
    1. *Default to no real-space collar (recommended).* `prop_radial`, `prop_free` and
       `prop_planar` keep today's wall or wrap at the requested size, plus:
       - the safety ramp (Q6);
       - the returned-share monitor, on by default;
       - a `SpatialAbsorber` collar on request, designed for fast content with a declared
         `angle` and refused under P0-14 when it would reflect the launch.

       The monitor stops the suite's case as configured (its return is $10^{-1}$ in
       field); its message names the radius needed. *Corrected in the follow-up below: as
       specified, it would not have.*
    2. *Default to Q2's outside placement as distance only.* A bare wall at $(1+f)R$: 2.1
       times the radial step cost by default. This is robust on the nonlinear witness, but
       not within budget on the §2 radial probe ($1.1\times10^{-3}$).
    3. *Default to a collar whose fraction follows P0-14*, resolved at translation: $f$ of
       8–10 on the nonlinear witness.
  - The interim report's option 2 (protect less) is withdrawn: the band opens only by
    *raising* $\kappa_p$, and that measured worst.
- **P0-16. The intermediate-band report.** Print the WKB attenuation with the exact symbol
  per frequency, weighted by the launch's or the declared signal's far-field spectrum.
  Keep $\Gamma_\mathrm{min}\kappa_p/\kappa$ only as a stated bound. *Amends* §4.4 and §5.2.
- **P0-17. Weak signals.**
  - Judge §7.5's weak-signal witness on the signal's own norm.
  - Say in the guide that a weak generated or launched signal is protected only if its
    angle is declared.
  - The launch-derived band does not cover a weak tilted seed.
- **P0-18. The monitor.** Adopt P0-11 in two dimensions, with these changes:
  - subtract the inward current's noise expectation before judging it;
  - accumulate the current at every accepted step, not at checks;
  - record at translation when three times the strip's noise floor exceeds the warning,
    and warn when it exceeds the stop;
  - pad the strip transform twice, not eight times.

  Q7's thresholds stand. The guide says that a slow reflected spread warns rather than
  stops.
- **P0-19. Q4.** Keep in-stage `Multistep` as the radial and planar default. The designs on
  which it loses are the ones that fail accuracy. P0-14's designs are wide and weak, and
  under them `Multistep` keeps about a threefold advantage.
- **P0-20. `cover`.** Keep `"protected"` as the default (A4). Offer `"intermediate"` with
  its minimum step count recorded in the design.
- **P0-21. The split (A10).**
  - Drop it from phases 1–3 and put it on the roadmap, with:
    - its measured potential: about 2–2.7 times on Cartesian at a tight tolerance, mostly
      the in-stage pass's own overhead;
    - the accuracy cap it would need, which the controller cannot see.
  - First skip the free routes' needless temporal pair (above).
  - *Amends* §5.3; under the A10 ruling it would otherwise stay on `numpy_ref`.
- **P0-22. Requalification.**
  - The suite's `spatial_absorber` case is 10 % wrong at 8 cm as configured. Phase 3 resizes
    it, or recasts it as the monitor's stop witness under P0-15 option 1.
  - Short witnesses for phase 1 tests are recorded with the harness:
    - the four-wave-mixing case at 256 samples (seconds);
    - the nonlinear case scaled paraxially to 1 cm, which reproduces every 8 cm error to
      two digits in 3–16 s;
    - the monitor's two-dimensional set with expected readings.

    A ten-step nonlinear witness is not possible: the error appears only once the beam
    fills the collar.

### Maintainer decisions on phase 0 (2026-10-31)

**P0-15: option 1** (John Travers). This deviates from Q1 and Q2, and is recorded here as
SEP 0 requires; Part I is not edited.
- `prop_radial`, `prop_free` and `prop_planar` keep today's wall or wrap at the requested
  size by default: no real-space collar unless one is asked for.
- On by default:
  - the safety ramp (Q6);
  - the returned-share monitor, whose stop message names the size needed.
- A `SpatialAbsorber` collar is placed outside the request on request, at `collar_fraction`.
  Its design is checked at translation against the launch's predicted own return (P0-14),
  and refused with the fraction that would meet the budget.
- Q2's fractions become the defaults of a requested collar, not of every run.
- The interim report's option 2 is withdrawn.

### Phase 0 — follow-up: the monitor with no collar (2026-11-01)

**A correction.** P0-15 option 1 above says the monitor "stops the suite's case as configured".
No run had measured that, and for the monitor as specified it is wrong. Every two-dimensional
witness had a collar-sized strip; with no collar, the strip shrinks to the request's outer quarter.

*Set-up.* One frequency, exact in the box's modes, requested size 300 µm, spacing 4.69 µm, 8 cm,
checks every 1 mm. The suite row uses the nonlinear run's series at its five planes, every
frequency. Energy shares of the launch; warning $10^{-6}$, stop $10^{-4}$.

| case | true error | strip, outer quarter | strip, outer half | inward current | guard above launch |
|---|---|---|---|---|---|
| suite case, nonlinear | $1.1\times10^{-2}$ | $1.4\times10^{-7}$ | n/a | n/a | $4.1\times10^{-4}$ |
| radial, 100 µm waist | $3.0\times10^{-2}$ | $6.4\times10^{-7}$ | $7.9\times10^{-6}$ | 0 | $1.7\times10^{-3}$ |
| radial, 60 µm waist | $2.2\times10^{-1}$ | $6.3\times10^{-6}$ | $1.8\times10^{-4}$ | 0 | $7.0\times10^{-3}$ |
| radial ring, $10^5$ rad/m | 1.0 | $5.6\times10^{-3}$ | $1.4\times10^{-1}$ | 1.0 | $2.2\times10^{-1}$ |
| Cartesian, 100 µm waist | $1.6\times10^{-2}$ | $1.2\times10^{-7}$ | $8.8\times10^{-7}$ | 0 | $2.7\times10^{-2}$ |
| Cartesian, tilt $10^5$ rad/m | 1.0 | $3.9\times10^{-3}$ | $1.2\times10^{-1}$ | 1.6 | $2.2\times10^{-1}$ |

- *The strip and the current are blind on a slow beam at a wall.* The suite case, 10 % wrong in
  field, reads below the warning.
- *The guard's share tracks the error,* within 3× at onset and low by up to 30× once the error is
  large. It stops the suite case between 4 and 6 cm. The warning falls near a true error of
  $10^{-6}$ and the stop near 1–5 × $10^{-4}$.
- *It does not stop what it should not.* Converging launches (120 µm with focus 4 cm, 170 µm with
  focus 8 cm) read zero until they truly spread back to the edge. A launch with tails on the edge
  reads zero above its baseline, and the guard's check at $z = 0$ catches it.
- *Noise, radial, one frequency, white at share $\eta$:* above baseline, $10^{-7}$ to
  $7\times10^{-7}$ at $\eta = 3.16\times10^{-6}$; 3–8 × $10^{-6}$ at $10^{-4}$; 3–4 × $10^{-5}$ at
  $10^{-3}$. An allowance is needed; these match $3\,\eta f_g/\sqrt M$ (§5.3).
- *The outward current at the guard* tracks as well, but gathers noise much faster, and was not
  adopted.
- *Not measured:* the Cartesian noise row, a nonlinear run at the monitor's real check interval,
  and the reading on the backends' frequency grids. §7.6 qualifies them.

**P0-23.** With no real-space collar the monitor judges the guard's energy above its launch value,
at Q7's levels, with a noise allowance. The strip and the current remain for a requested collar.

### Maintainer decisions on phase 0, continued (2026-11-01)

John Travers, going through the rest of the recommendations:
- **P0-23: the guard's energy on the default.** The rest of P0-18 stands for a requested collar.
- **Q4, P0-19: keep in-stage `Multistep`** as the radial and planar default.
- **`cover`, P0-20: dropped.** This changes the final report's recommendation, on the evidence that
  `"intermediate"` bought nothing measurable. A deviation from A4.
- **The split, P0-21: to the roadmap,** after removing the free routes' needless temporal pair. This
  replaces the A10 ruling.
- **P0-1, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17 accepted as written.** P0-4 and P0-5 stand as the
  final report amended them, and P0-8 is settled by Q4. These now apply to requested collars.
- **P0-22: the suite's `spatial_absorber` case gets a requested collar,** at $f = 2.6$, so that the
  suite keeps tracking the collar's cost path. The stop is tested separately.
- **The SEP returns to `Draft`.** Part I's goal, default, monitor principle and success criteria no
  longer described the design, which SEP 0 treats as invalidating the proposal. Part I is rewritten
  from these decisions, for re-acceptance. Phase 0 stands, and no new measurements are required.

Two choices made while rewriting Part I, for the maintainer's re-acceptance:
- *Q2's fractions are not implicit defaults.* A grid counts its samples and is built before the
  absorber, so `collar_fraction` defaults to 0 and a requested collar on such a grid is refused,
  naming the fraction. The 2026-10-31 decision said Q2's fractions become the defaults of a
  requested collar; they become the guide's starting points instead (§5.1).
- *The safety ramp's strength with no collar.* §4.4's rule needed a collar width. With none, it
  uses the requested size, the distance over which content crosses the aperture and returns. The
  ramp is exact in its own basis, so this was derived and not measured.

## Part III — History

- 2026-10-31: Drafted, from SEP 21's finding R4 and decision D9, and a linear probe of the
  radial and Cartesian collars (§2).

- 2026-10-31: GPT-6 completed an AI review of Part I against the design documents,
  the current implementation and SEP 21's recorded evidence. Findings R1–R14 below;
  the proposal text is unchanged. Review status is `In review`, pending human review.

- 2026-10-31: Opus 5.5 (drafting agent) answered R1–R14 and revised Part I (response
  below): in-stage default under a collar-crossing step limit, protected and safety
  bands, a returned-light monitor, a specification resolved at translation, a signed
  ordered ledger, the planar route, and an allocated error budget.
- 2026-10-31: Maintainer decisions (John Travers): R1–R14 accepted as revised; Q1–Q7
  decided as recommended (§12). The §2 probe was re-run on main at b09d43b with
  unchanged step counts and errors.
- 2026-10-31: Opus 5.5 checked Part I against the code at 0fedea2 before acceptance
  (*Pre-acceptance check* below). The maintainer accepted findings A1–A9 and ruled on
  A10–A11, and Part I was amended accordingly. The amendments are listed below.
- 2026-10-31: Human review and acceptance (John Travers). Review status `Reviewed`; status
  `Accepted`, then `In progress` on `feature/sep-0040-transverse-boundaries`.
- 2026-10-31: Phase 0 interim report (Part II): operator calibration, the linear probe
  against an analytic reference, the integrator and crossing measurements, and the monitor
  prototypes. Recommendations P0-1 to P0-12 await the maintainer's decision; Q4 returns
  (A1).
- 2026-10-31: Phase 0 final report (Part II): the weak-signal and nonlinear witnesses, the
  monitor's two-dimensional witnesses, the cover cost, the split against in-stage, and A11
  on Linux. P0-13 to P0-22. Q1 and P0-5 return to the maintainer: no default fraction
  meets the budget on a slow beam.
- 2026-10-31: Maintainer decision (John Travers): P0-15 option 1, no real-space collar
  by default and the collar on request; a deviation from Q1 and Q2, recorded in Part II.
- 2026-11-01: Phase 0 follow-up (Part II): the monitor as specified is blind on the no-collar
  default, correcting the final report; P0-23. Maintainer decisions (John Travers) on P0-1 to
  P0-23, Q4 and A10.
- 2026-11-01: Returned to `Draft` (John Travers), because the decisions invalidate Part I as
  accepted. Opus 5.5 (drafting agent) rewrote Part I from them: §1, §2, §4.2–§12. §3 and §4.1 are
  unchanged. The reviewers recorded reviewed the earlier text; re-acceptance is pending.

### AI review — 2026-10-31 (GPT-6)

**Recommendation:** revise before acceptance. Moving the radial collar outside the
request and separating its strength from the temporal collar address the recorded
problem. The proposed default also needs a controlled splitting error, a protected
physical angular band and a monitor that permits the intended outgoing radiation.
Phase 0 should establish those contracts before selecting constants. P1 below denotes
a correctness or acceptance blocker; P2 denotes a specification or qualification gap
to address in the revision. All points await the maintainer's decision and the drafting
agent's response; none is an instruction to change the numerical model without that
decision.

1. **R1 — P1: an exact absorber substep does not control boundary crossing or the
   combined error (§§1, 4.7, 6, 7.2, 7.6).** A linear problem has zero source defect
   when diffraction is carried exactly. Its controller therefore need not reduce a
   step during which a packet crosses both collars and wraps into the aperture.
   Multiplying only at the endpoint cannot remove that returned packet. The leading
   Lie defect contains the diffraction–collar commutator; even when its differential
   form is supported on the ramp, subsequent propagation carries its error into the
   aperture. The non-paraxial spectral operator is nonlocal already. SEP 21 §4.5
   explicitly cautions against the claimed confinement.

   An independent, noiseless paraxial Fourier probe illustrates the failure: 512
   samples over a periodic interval with outer half-width 300 µm, requested half-width
   200 µm, wavelength 800 nm, and launch
   $E(x)=\exp[-(x/(40\,\mathrm{\mu m}))^2+i\kappa_0x]$ with
   $\kappa_0=0.2k=150\pi/(300\,\mathrm{\mu m})$. Use $w=100$ µm,
   $P(s)=s^2$, $\alpha_0=3\log(10^3)(\kappa_0/k)/w$, no spectral ramp, and
   $L=3$ mm, one transverse circuit. Exact Fourier diffraction followed by the
   spatial factor retains 1.0 of the launch energy inside the aperture with one
   step; 4, 8 and 16 steps retain respectively $7.43\times10^{-5}$,
   $1.67\times10^{-6}$ and $1.14\times10^{-6}$. The unbounded analytic Gaussian
   retains $3.34\times10^{-73}$ there. This probes the proposed map, not an
   implementation of SEP 40.

   **Requested resolution:** specify a crossing/splitting accuracy constraint or
   controlled substeps, including fixed schedules and the monitor's sampling, and
   qualify error *inside* the aperture at three step sizes. Compare Lie and symmetric
   splitting if useful. How can the unconditional “no more steps than no absorber”
   requirement survive this linear counterexample? Treat work counts as measured
   performance at equal accuracy, unless a restricted claim is justified.

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

2. **R2 — P1: the launch-derived spectral ramp can remove the intended signal
   (§§2, 4.4, 5.2, 7.5, 10).** A $k_\perp$-diagonal loss acts throughout the physical
   aperture. A multiple of the launch's rms wavenumber cannot protect arbitrary
   frequencies or angles subsequently generated by self-focusing, plasma or
   four-wave mixing. A weak, legitimate sideband above that threshold is damped
   before it reaches an edge, even when the whole interaction is well inside the
   requested aperture. This directly affects the far-field observable motivating
   the proposal. Increasing the multiplier on two witnesses does not establish a
   generally safe default, and the no-change invariant in §7.5 is then false.

   **Requested resolution:** distinguish an explicitly protected physical angular
   band from a numerical safety band; state what the default may discard and how a
   user protects generated radiation. Add a weak high-angle signal generated inside
   a wide aperture and compare its spectrum/port energy with a larger-domain reference.
   Should a launch-derived ramp require explicit opt-in or a declared physical band?

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

3. **R3 — P1: the inherited monitor policy stops successful escape (§§1, 5.4,
   7.2, 12 Q7).** `BoundaryCheck` and `BoundaryMonitor._judge` currently warn when
   the square root of guard occupancy exceeds $10^{-3}$ and stop above $10^{-2}$:
   an energy share above $10^{-4}$ stops the run. A beam intentionally losing most
   of its energy through a well-designed spatial collar can easily exceed this
   occupancy without significant reflection. Removal itself is also expected here.
   The temporal policy therefore cannot simultaneously be the default and qualify
   the proposed strong-escape witness without an explicit policy change.

   **Requested resolution:** decide separately how launch clipping, intended outward
   flow, inadequate absorption and returned contamination are judged. Name the
   denominator and per-edge regions, and give Q7 an owner and decision point. Test
   that accurate strong escape completes under the selected default, while an
   under-resolved or ineffective boundary is detected. What measured quantity can
   distinguish those outcomes, rather than merely detecting light at the edge?

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

4. **R4 — P1: the proposed builder lacks inputs required by its design (§§4.2–4.4,
   5.2, 5.5, 7.7).** `spatial_absorber(space, grid)` cannot evaluate the medium's
   $k_z(\omega)$, a slab mode's longitudinal wavenumber, or the launch's rms
   wavenumber. A temporal grid supplies neither the material nor the launch. Yet the
   return type is described as already containing resolved rate arrays. A requested
   angle also needs a reference frequency/material, since a fixed angle is not one
   fixed $\kappa$ across a broadband pulse. Radial media may vary along z under
   SEP 34, so entrance values need not bound the full path. The stated ramp strength
   “short against the run” can reintroduce the length dependence §7.7 forbids.

   **Requested resolution:** distinguish an unresolved specification from its
   translated, immutable design, or provide the missing inputs explicitly. Define
   the frequency/path extrema, treatment of inactive and nonpropagating bins,
   weighting and noise treatment of the rms, angle convention, and precedence of
   explicit and inferred values. State how zero collar width, ramp-only use and an
   infeasible design behave. At which operation is the design frozen, and when do
   `with_input`, scans, coupling to another grid/medium and resume rebuild or retain
   it? Make the length-independence claim cover the complete resolved operator.

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

5. **R5 — P1: complementary bands need a quantitative overlap (§§4.4, 5.2,
   7.1).** Section 4.4 allows $\kappa_\mathrm{hi}$ below the ramp's start, leaving
   a gap; §5.2 instead puts the start at or below $\kappa_\mathrm{hi}$. Even equality
   is insufficient to prove the target: a raised-cosine rate is zero at its start,
   whereas the real-space return attenuation is only just at its target there and
   decreases beyond it. “Everything above” is not immediately absorbed. A strong
   ramp can also hide real-space reflection in a combined measurement.

   **Requested resolution:** define the overlap and a bound or measured criterion
   for total returned amplitude across it, including packets generated close to the
   boundary. Measure the real-space collar separately with the ramp disabled, then
   the combination. On rectangular Cartesian grids, specify whether the transition
   is in $|k_\perp|$ or the existing normalised maximum of its components, and how
   the two widths and corner crossings enter the design. Which component normal to
   each face sets that face's absorption/reflection interval?

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

6. **R6 — P1: the scattering approximation needs explicit limits (§§4.1–4.3,
   4.6, 7.1).** Replacing $k$ by $k_z$ does not turn the paraxial local ODE into
   the exact non-paraxial scattering problem. Even the local constant-rate symbol
   gives $q^2=\kappa_0^2+i k_{z0}\alpha+\alpha^2/4$; for a varying rate, the
   square-root operator and multiplication do not commute. Dropping terms requires
   a stated weak-loss/slow-variation approximation. Similarly, putting the radial
   collar outside the aperture does not guarantee $\kappa R_\mathrm{req}\gg1$.
   The lowest QDHT column has
   $\kappa R_\mathrm{req}=j_{0,1}/(1+f)$, about 1.60–1.85 for the proposed
   fractions, and low-wavenumber light can reach the boundary on a long run or be
   launched near it. Writing a radial field as $u(r)/\sqrt r$ retains a
   $1/(4r^2)$ term in its order-zero radial equation.

   **Requested resolution:** delimit the WKB/planar approximation and calibrate the
   actual radial and non-paraxial operators outside that range. Do not identify the
   red/blue edges with extrema of $k_z$ without checking the material's dispersion
   and cutoff. What happens when the protected band approaches cutoff, or the
   required low-angle content falls outside the asymptotic regime? Include those
   cases in phase 0 or refuse them explicitly.

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

7. **R7 — P1: passivity to rounding contradicts both the metric and the cited
   evidence (§§3, 4.7, 7.4).** For physical-node multiplication
   $A=\operatorname{diag}(\alpha)$ and the nonlocal flux metric $M$, the removal
   quadratic form is $(MA+AM)/2$. Positive $A$ and $M$ do not make this form
   positive semidefinite. An interior/collar two-node principal block has diagonal
   entry zero at the interior node and off-diagonal $\alpha_j M_{ij}/2$; if this
   coupling is nonzero its determinant is negative. This is not restricted to
   cutoff. SEP 21's phase 0a finding 9 already records a real-space eigenvalue of
   $-2.9\times10^{-7}$ times the peak rate on its radial witness.

   An independent 64-node Fourier calculation on the same 600 µm interval, with
   $M=F^*\operatorname{diag}(\sqrt{1-(k_x/k)^2})F$, $k=2\pi/(800\,\mathrm{nm})$,
   and a unit-peak quadratic collar outside 200 µm, gives a minimum generalised
   eigenvalue $-1.25\times10^{-10}$; all columns have $|k_x|/k\le0.043$.
   This is well above rounding in magnitude despite being far from cutoff.

   **Requested resolution:** replace the universal passivity assertion with an
   explicit, justified gain bound/admission criterion, or propose a different
   dissipative operator for approval. Test the finite split map as well as its
   infinitesimal generator. Keep removal signed; under what conditions may it be
   called absorbed physical energy?

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

8. **R8 — P1: specify the removal ledger when collars are fused (§§4.7, 5.4,
   6, 7.5).** The current `BoundaryMonitor.split_removal` uses the field-norm proxy
   on spatial routes and excludes the following hard mask; SEP 21 D16 deferred
   admittance weighting. It does not already provide the claimed flux measurement.
   Although the temporal and spatial multipliers commute, their losses cannot each
   be computed from the same incoming state and added. Even for scalar factors
   $a,b$, $(1-a^2)U+(1-b^2)U$ double-counts $(1-a^2)(1-b^2)U$ compared with the
   combined loss. A nonlocal metric makes the attribution more consequential.

   **Requested resolution:** define an ordered, telescoping attribution or another
   explicit convention, the treatment of masks and signed gains, the in-stage
   removal estimator, and the units/integration of the other `absorbed_power`
   channels. Specify persistence at accepted steps, rejected trials, field-free
   planes, checkpoints/resume and child relaunches, consistent with SEP 21 D10.
   Test a run with both collars active and verify the complete ledger in one stated
   metric. How will flux evaluation fit the memory ledger and the advertised cost?

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

9. **R9 — P1: qualify the changed state against integrator history (§§5.3–5.5,
   6, 7.6, 11).** Runge–Kutta FSAL reuse must be disabled when the spatial factor
   changes the accepted state, even with temporal absorption disabled; the current
   `numba_cpu` eligibility check specifically tests the temporal split. Adams keeps
   predicted-state source history (`backends.multistep`, `AdamsMarch._decide`). A
   pointwise attenuation of a field and of its stored Kerr source requires different
   factors. SEP 21 §4.5 records the failure of a history-transported Strang experiment;
   it does not establish spatial support or accuracy merely because a temporal split
   is admitted. Real-space scattering couples all transverse spectral columns.

   **Requested resolution:** state the intended history treatment and its accuracy
   contract, then test spatial-only absorption with RK, low-storage RK and both Adams
   formulations as applicable, including the native march, rejections and resume.
   Refuse unsupported combinations by name. Can the claimed aperture accuracy be
   demonstrated with a nonlinear field occupying the collar, rather than only with
   a temporal collar that stays nearly empty?

   **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

10. **R10 — P2: make the candidate profiles mathematically implementable (§§4.3,
    4.5, 4.7, 11).** A nonzero polynomial cannot have all derivatives vanish at its
    onset: $s^n$ joined to zero has only finite smoothness. It therefore does not
    inherit the stated faster-than-any-power conclusion. The transmission-free
    potential has a second-order pole, as the author's
    [publication record](https://ora.ox.ac.uk/objects/uuid%3Aa1476780-fa60-4038-bc0d-2099b3332770)
    confirms. It cannot simultaneously satisfy $0\le P\le1$ and use the finite
    profile integral in §4.2. Section 4.7 supplies no cap despite the reference to it.

    **Requested resolution:** separate the finite-smoothness and smooth profiles,
    supply the actual potential and optical-unit mapping from the primary paper,
    and define its truncation/cap, endpoint sampling and finite-rate validation.
    Re-measure transmission and reflection after capping and discretising; do not
    carry over a continuum zero-transmission claim. What parameters are held fixed
    when comparing the singular and bounded candidates fairly?

    **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

11. **R11 — P2: resolve planar scope and qualify every advertised route (§§2–3,
    5–7, 11).** The current transverse ramp is refused on `PlanarGrid` as well as
    `RadialGrid` (`translate.problem` and `boundaries.transverse_collar_rate`;
    SEP 22 Q5). Only lifting the radial refusal leaves the planar default without
    the advertised companion ramp.
    Planar columns are $(p,k_y)$ with mode-dependent longitudinal dispersion and
    slab amplitudes; they are not Cartesian node fields.

    **Requested resolution:** say whether this SEP implements the planar ramp,
    with its per-mode design and y-only collar, or records an explicit exception.
    Add a qualification matrix for radial, rectangular Cartesian, line and planar
    routes on field/envelope grids, the supported precision and integrators, and
    each required CPU backend. Include a multi-mode slab and an anisotropic box;
    two radial/Cartesian probes alone cannot qualify those interfaces. Name refusal
    and serialization tests as well as numerical parity tests.

    **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

12. **R12 — P2: the “free” split cost has prerequisites (§§4.7, 6, 9).** In
    `numpy_ref.backend.NumpyRefBackend._finish`, the physical-space visit occurs when
    a nonzero temporal split is applied. `absorber=False` explicitly leaves the new
    spatial absorber enabled, so that supported case needs its own transverse
    transforms. In-stage temporal absorption has the same issue. Fusing a multiply
    also does not make the flux reductions or their scratch free. Radial dense
    transform work grows quadratically with the sample count, not merely with the
    collar's linear storage factor.

    **Requested resolution:** distinguish marginal multiply cost from complete
    transform, diagnostic, storage and history costs. Measure temporal split on/off,
    matched physical resolution at several collar widths, and accuracy-qualified
    schedules. Which scratch is reused, and what additional buffers are admitted?

    **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

13. **R13 — P2: repair the numerical audit trail (§§2, 4.4, 7.2–7.3).** The
    probe requests a radius of 150 µm, but §4.4 calls its 50 % outside collar
    $w=150$ µm. With §5.1's definition it is 75 µm; the stated band estimates must
    therefore be recomputed or the different geometry explained. The probe table
    also omits transverse counts, temporal band/window, launch duration, noise
    setting, save/step schedule, detailed profile and the reference error estimate.
    Without these it is not possible to reproduce the step counts or attribute the
    error to the absorber rather than resolution.

    **Requested resolution:** give a public, reproducible probe specification or
    repository script and record the exact error norm and reference convergence.
    Keep physical sampling fixed while extending the domain; increasing radius at
    fixed sample count also coarsens the beam. For QDHT comparisons, specify the
    common physical evaluation grid because changing the outer radius changes its
    nodes. Are three reference sizes available to establish an extrapolation order,
    rather than assuming one from the temporal campaign?

    **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

14. **R14 — P2: separate the acceptance error budgets (§§7, 11).** Grid convergence
    of the 1D ODE does not set a tolerance for its discrepancy from WKB, radial
    curvature or the non-paraxial operator. A reflection limit of $10^{-6}$ in power
    and return transmission $e^{-\Gamma}=10^{-3}$ in field can add coherently to
    $2\times10^{-3}$ in field before other errors. The meaning also changes if the
    aperture error is normalised by a small remaining field instead of the launch.
    Finally, `rtol` is a local controller setting, not a global bound for the
    length-independence comparison. `testing.md` §3 requires an accuracy envelope
    for differing adaptive trajectories and matched schedules/histories for a
    rounding-level backend comparison.

    **Requested resolution:** define amplitude versus power coefficients, incident
    normalisation, return-time measurement and aperture/core error norms. Allocate
    reflection, transmission, spatial discretisation, split and reference errors
    against the $10^{-3}$ target. Demonstrate three-level convergence separately
    from WKB trend checks, and use matched discrete methods for parity. Which
    protected observables, including weak far-field ports, must pass before a
    default is recommended?

    **Maintainer decision (John Travers, 2026-10-31):** accepted as revised.

**Evidence and limits of this review.** Part I was read in full. Repository checks
covered the architecture, sign/flux conventions, tolerance policy, SEP 21's findings
and decisions, the current monitor and split finish, transverse-ramp admission, the
QDHT definition, planar translation and integrator-history contracts. The two small
independent calculations above test specific claims; no full propagation campaign or
backend qualification was run. Publisher/author records were checked for the
[Vibók–Balint-Kurti paper](https://pubs.acs.org/doi/10.1021/j100201a012),
[Manolopoulos paper](https://manolopoulos.chem.ox.ac.uk/publications.html),
[Muga et al. review](https://jppalao.webs.ull.es/Publications/publications.htm) and
[QDHT paper](https://opg.optica.org/josaa/abstract.cfm?uri=JOSAA-21-1-53).
Those checks establish bibliographic identity and the reported singular-potential
property, not the detailed coefficient formulas or the claimed optimum. Full texts
and equation-level claims of references [1–9] were not comprehensively verified;
phase 0 must record those source checks before transcribing a profile or constant.

### Response to review — 2026-10-31 (Opus 5.5, drafting agent)

Every point was considered against the code and, where the review gave a calculation,
reproduced independently. All fourteen are accepted in substance. The principal changes
to Part I are four:

- the default scheme becomes the in-stage rate under a new collar-crossing step limit
  (R1, R9);
- the $k_\perp$ ramp is confined to a numerical safety band, beside a declared
  protected band (R2, R5);
- the monitor judges returning light, not arriving light (R3);
- the absorber becomes a specification resolved at translation (R4).

The maintainer's decisions on the points remain to be recorded beside them.

1. **R1 — accepted.** The counterexample was reproduced exactly: 1.0, $7.4\times10^{-5}$,
   $1.67\times10^{-6}$ and $1.14\times10^{-6}$ at 1, 4, 8 and 16 steps, against
   $1.03\times10^{-6}$ for the exact generator. Strang gives the same numbers as Lie, so a
   symmetric splitting does not cure it. Convergence sets in once the per-step transverse
   displacement falls below about three quarters of the collar width.
   - *Changes:*
     - §4.7 now states the failure and adds a collar-crossing step limit,
       $h \le c_\times w k_{\min}/\kappa_\times$, for **both** schemes, with `dz_max`
       clamped, fixed schedules refused and the monitor's interval held within it;
     - the confinement claim is withdrawn: the split's error is qualified inside the
       aperture at three step sizes (§7.4), and the counterexample becomes a regression
       test;
     - the "no more steps than no absorber" criterion is replaced by cost at matched
       accuracy (§1, §7.7);
     - the recommended default becomes in-stage (§4.7, §6, Q4), whose action the error
       controller sees.
   - *Residual:* the controller seeing an in-stage collar is argued, not yet shown, for a
     packet crossing between stage nodes. That is why the step limit binds in-stage too,
     with $c_\times$ calibrated per scheme and integrator in phase 0.

2. **R2 — accepted.** A $k_\perp$ rate acts throughout the aperture, so a launch-derived
   ramp would damp generated sidebands before any edge.
   - *Changes:*
     - §4.4 now orders a protected band (never under a $k_\perp$ rate), an intermediate
       band (kept, partially absorbed, reported) and a numerical safety band (the ramp, at
       the grid's outer $f_s$ of its wavenumbers, as SEP 22 has it);
     - lowering the ramp is an explicit declaration (`discard_above`, §5.2), never a
       default;
     - the launch-derived default protects only the launched spread, and the resolved
       design says so;
     - §7.5 adds the requested protected weak-sideband witness, compared through SEP 23's
       far-field ports with a larger domain;
     - §7.7's invariant is restricted to content inside the protected band;
     - Q5 asks whether a declared band should be mandatory;
     - §10 records the first draft's ramp as rejected.

3. **R3 — accepted.** The monitor's stop above an energy share of $10^{-4}$ was confirmed
   in `BoundaryMonitor._judge`. The temporal policy cannot be inherited.
   - *Changes:* §5.4 now separates four situations, each with its own quantity and the
     launch's norm as denominator, per face:
     - launch clipping, judged on the guard share at $z = 0$;
     - intended outward flow, recorded at INFO and never a stop;
     - returned light, measured by the inward-travelling share in the guard;
     - design gaps, reported at translation.

     The answer to "what can distinguish them" is the direction of travel: light leaving
     cleanly carries a small inward share however much reaches the edge. §7.6 requires a
     strong-escape run to complete and a wall, a wrap and an under-strength collar to be
     caught. Q7 is owned by the phase 0 report and decided by the maintainer before
     phase 1.
   - *Residual:* the inward-share measure, its window leakage floor, and the radial
     incoming-wave decomposition are proposed, not yet demonstrated. Phase 0 qualifies
     them.

4. **R4 — accepted.**
   - *Changes:* §5.2 now separates the user's `SpatialAbsorber` specification from the
     `SpatialAbsorberDesign` that translation resolves against grid, space, medium and
     launch. It defines:
     - **inputs:** $k_z$ extremes over active propagating bins, over the medium's path
       nodes, and over slab modes; refusals near cutoff;
     - **the rms:** energy-weighted, over active bins, on the noise-free launch;
     - **the angle:** SEP 23's $\sin\theta = c\kappa/\omega$, so a band that varies with
       frequency;
     - **precedence:** `kappa`, then `angle`, then the launch;
     - **edge cases:** zero width (real-space part refused, ramp-only allowed); infeasible
       designs refused, naming the width needed;
     - **lifetime:** resolved once per translation and immutable; re-resolved by
       `with_input` or a scan point only when launch-derived; resolved anew on a coupled
       child's grid; reused on resume, refusing a mismatched grid.
   - *Length independence:* the safety ramp's strength is now set by $w$, not by the run
     (§4.4), so the whole resolved operator is independent of $L$ (§4.2). §7.7 checks the
     operator itself and compares propagations on matched schedules.

5. **R5 — accepted.** The inconsistency between §4.4 and §5.2 is removed.
   - *Changes:*
     - §4.4 defines coverage by the combined attenuation
       $\Gamma_\mathrm{tot}(\kappa) = \Gamma(\kappa) + \Gamma_s(\kappa)$, not by where the
       ramp starts, and notes that a raised cosine adds nothing at its start;
     - the ramp strength rule targets $\Gamma_\mathrm{min}$ from its midpoint;
     - the design records the shortfall band;
     - content generated within $w$ of the wall is measured in phase 0;
     - §7.3 measures the real-space collar with the ramp off, then the combination;
     - on rectangular grids each face is designed on its normal component and width, the
       ramp keeps SEP 22's maximum-norm, and a corner packet crosses both collars.

6. **R6 — accepted.** The exact constant-rate symbol, including the $\alpha^2/4$ term, the
   non-commutation for a varying rate, and the radial $1/(4r^2)$ term are now stated (§4.1,
   §4.6). The lowest-column estimate is confirmed: $\kappa R_\mathrm{req} = j_{0,1}/(1+f)$,
   about 1.2–2.2 for $f$ between 1 and 0.1.
   - *Changes:*
     - §4 is declared a design scaling. Every constant is calibrated on the discrete QDHT,
       Fourier and slab operators (§7.3), with the WKB comparison a trend check only;
     - $k_{\min}$ and $k_{\max}$ are computed from the material, not assumed at the band
       edges;
     - protected bands approaching cutoff ($\kappa_p > 0.9k$ on any active bin) are refused
       until qualified;
     - the low radial columns are reported and may not be protected.

7. **R7 — accepted.** The 64-node minimum eigenvalue was reproduced:
   $-1.2529\times10^{-10}$ with $|k_x|/k \le 0.043$. The first draft's §3 also misread
   SEP 21 finding 9, whose $-2.9\times10^{-7}$ for the real-space collar it had overlooked.
   - *Changes:*
     - §3 corrects that misreading;
     - §4.9 replaces the passivity assertion with an admission criterion, computed at
       translation for the generator and the one-step split map, against a tolerance
       $\varepsilon_P$ that phase 0 proposes and the maintainer approves (Q6);
     - removal is signed, and is called "removed field norm". It may be read as energy only
       for paraxial beams far from cutoff.

     A different dissipative operator is not proposed. No candidate keeps the diagonal
     node rate's cost and also commutes with the flux metric.

8. **R8 — accepted.** `split_removal` does use the field-norm proxy on spatial routes; the
   first draft's claim of a flux measurement was wrong.
   - *Changes:*
     - §4.9 adopts the field-norm measure, with the flux ledger left on D16's roadmap item;
     - credits follow a fixed order — spatial, then temporal, then mask — on successive
       states, so they telescope without the double count the review shows;
     - the in-stage removal is the integrated rate at the monitor's checks;
     - only accepted steps are credited;
     - persistence follows SEP 21 D10 across checkpoint, resume, field-free planes and
       child relaunches;
     - units are stated alongside, and separate from, `absorbed_power`'s channels;
     - §7.7 adds a both-collars ledger-closure test across a resume.

9. **R9 — accepted.** The code confirms that `numba_cpu`'s first-same-as-last reuse is
   withdrawn for the temporal split, and that the Adams march carries source history.
   - *Changes:*
     - the default is in-stage, which changes no history;
     - the spatial split is admitted only with `RungeKutta` (first-same-as-last off), and
       refused by name under both Adams marches and the low-storage march until a later
       proposal qualifies a history treatment (§4.7, §5.3);
     - §7.5 adds the requested witness with a nonlinear field occupying the collar;
     - §7.7's qualification matrix covers RK, Multistep and low-storage RK in-stage, RK
       split, rejections and resume.

10. **R10 — accepted.** $s^n$ joined to zero has $n-1$ continuous derivatives, and the first
    draft's smoothness claim was wrong for it. The singular potential cannot satisfy
    $P\le1$.
    - *Changes:*
      - §4.5 separates smooth, finite-smoothness and singular classes;
      - for the singular profile it defines a cap, truncation, discrete measurement
        replacing the continuum's zero transmission, and split-only admission;
      - the optical-unit mapping is transcribed from the primary paper in phase 0 with a
        recorded source check;
      - a fair comparison is defined by fixed width, grid, protected band and attained
        $\Gamma(\kappa_p)$;
      - §13 states that only bibliographic identity has been checked so far.

11. **R11 — accepted.** The ramp is refused on `PlanarGrid` as well as `RadialGrid`.
    - *Changes:* this SEP implements the planar case (§5.5):
      - a y-only real-space collar per $(p,y)$ row;
      - a $k_y$ safety ramp per $(p,k_y)$ column, diagonal there;
      - the design taken over the slab modes' $k_z$.

      §7.7 adds a qualification matrix over radial, square and anisotropic Cartesian, line
      and planar (one and two modes) routes, field and envelope grids, precisions,
      integrators and CPU backends, each cell qualified or refused by name with a refusal
      test, plus a serialisation and resume round trip.

12. **R12 — accepted.** `absorber=False` with a spatial absorber does leave no temporal
    split to share a physical-space visit, and the dense radial work grows as the square of
    the sample count.
    - *Changes:*
      - §4.7 and §6 now list marginal multiply, transform, monitor and storage costs
        separately;
      - §5.1 states $(1+f)^2$ dense radial work;
      - §7.7 measures with the temporal split on and off, at several widths at fixed
        resolution, at matched accuracy;
      - scratch reuse is from existing physical-space stage buffers, and any new buffer is
        listed in the memory ledger.

13. **R13 — accepted.** The geometry mismatch was real. The probe used
    $R = R_\mathrm{req}/(1-f)$ on the radial grid (SEP 21's convention) but `FreeGrid`'s
    $R_\mathrm{req}(1+f)$ on the Cartesian one. So "50 %" meant a 150 µm collar radially
    and a 75 µm one in Cartesian, and the radial best corresponds to `collar_fraction = 1.0`
    under §5.1.
    - *Changes:*
      - §2's table now gives outer sizes and collar widths in micrometres, and the full
        probe specification;
      - §4.4's estimates are recomputed per row, including the Cartesian 75 µm collar,
        whose barely open band agrees with its larger error;
      - Q2 notes the cost of $f = 1$;
      - the probe used one reference size, and §2 now says its convergence was not
        established;
      - phase 0 commits the probe as a repository script with three reference sizes at
        fixed spacing and a common physical evaluation grid for the QDHT (§7.2).

14. **R14 — accepted.**
    - *Changes:* §7.1 now defines:
      - amplitude coefficients, the incident normalisation and the return-time measurement;
      - aperture and core norms, normalised primarily by the launch;
      - an amplitude-summed allocation of the $10^{-3}$ target over return transmission,
        reflection, discretisation, the controller and crossing, and the reference.

      $\Gamma_\mathrm{min}$ is raised to $\log(5\times10^3)$ and the reflection floor set
      to $2\times10^{-4}$ in amplitude (§4.8), so that the coherent sum the review computes
      fits the budget. Three-level convergence is separated from the WKB trend checks
      (§7.2–7.3). The length-independence check uses matched schedules or testing.md §3's
      envelope, not `rtol`. Parity uses matched schedules and histories. A default is
      recommended only when every budget item, including the protected weak-signal port,
      is met.

The probe's figures in §2 are unchanged, since they were measured; only their description
is corrected. No new propagation measurements were made for this revision.

### Pre-acceptance check — 2026-10-31 (Opus 5.5)

Part I was read against the code at 0fedea2: the grids, the three translation sites, the
problem's checks, the monitor, the steppers, the four backends, coupling, the tracking
cases, the galleries and a downstream package. Its scalings were also rechecked: §4.2–4.4,
the order-of-magnitude table and the probe geometry. Those hold. The maintainer's rulings
(John Travers, 2026-10-31) follow each finding. Part I was amended before acceptance as
noted.

1. **A1 — in-stage stability under the default integrator.** §4.7 quoted the stability
   limit for Runge–Kutta. The radial and planar routes default to the Adams `Multistep` at
   order 5, whose negative-real interval is 0.0877 against 3.3. On the probe that is about
   0.12 mm against 4 mm. *Ruling:* measure it in phase 0; Q4 returns if collar stability
   binds. *Amended:* §3, §4.7, §6, §11, §12 Q4.
2. **A2 — the launch-derived band's lifecycle.** `with_input` shares the parent's operator,
   transform and step limits by identity, and SEP 31 children are translated against a
   zero placeholder launch. Neither can re-resolve a band. *Ruling:* the design is frozen
   at translation; a coupled child is translated against its actual launch. *Amended:* §5.2,
   §5.5.
3. **A3 — the returned-share strip.** A 5 % guard cannot resolve the direction of the
   content that reflects, and the default input noise fills it. *Ruling:* accepted.
   *Amended:* §5.4, §11.
4. **A4 — the crossing limit and the intermediate band.** Above about
   $(0.75/c_\times)\kappa_p$, intermediate-band content may cross within one step, and §4.7
   referred to an option §5.2 did not define. *Ruling:* add `cover`, record the unguarded
   band, and measure the cost. *Amended:* §4.7, §5.2, §11.
5. **A5 — the cost of the passivity admission.** The full generalised eigenproblem is not
   affordable at translation on Cartesian grids. *Ruling:* use a reduced surrogate,
   validated in phase 0. *Amended:* §4.9, §11.
6. **A6 — the ledger and its channels.** The closure test ignored the safety ramp and the
   other channels. SEP 23's transverse channel exists only where an operator is separable.
   The temporal monitor clips removal at zero. *Ruling:* accepted. *Amended:* §4.9, §7.7.
7. **A7 — the reflection floor's range.** The floor was stated over a protected band that
   includes content reflected by design. *Ruling:* accepted. *Amended:* §4.8.
8. **A8 — other consumers of the window.** These are the coupling plane's radial
   resampling, the coupling elements' boundary mask and the run record. *Ruling:* in scope.
   *Amended:* §3, §5.1.
9. **A9 — factual corrections.** The corrections covered:
   - the planar rate site and `PlanarGrid.window_y`;
   - where `boundary_N` and the temporal scheme live;
   - the stability check's effective rate;
   - where the collar warning is raised;
   - the `spatial_absorber` tracking case, which today has no real-space collar;
   - the complete list of galleries to redo;
   - the device backends.

   *Amended:* §3, §9.
10. **A10 — the split's scope.** Its main reason, the singular profile, was dropped with Q3.
    *Ruling:* build it on `numpy_ref`, and carry it further only on phase 0's evidence.
    *Amended:* §5.3, §11, §12 Q4.
11. **A11 — the radial cost of $f = 1$.** The dense Hankel products dominate a radial stage.
    *Ruling:* phase 0 reports the factor before Q2 is confirmed. *Amended:* §11, §12 Q2.
