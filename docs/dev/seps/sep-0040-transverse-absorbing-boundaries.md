# SEP 40 — Transverse absorbing boundaries for free-space propagation

| | |
|---|---|
| **SEP** | 40 |
| **Title** | Transverse absorbing boundaries for free-space propagation |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 |
| **Status** | Draft |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI); Opus 5.5 (AI) |
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
  and stops a run that exceeds the budget, naming a size to try. It also judges what the
  safety ramp removes.
- **A collar on request.** A `SpatialAbsorber`, outside the requested aperture, with a strength
  set by a *protected* band of transverse wavenumbers and by the collar's width, not by the
  propagation length.
- **The design check.** Translation propagates the launch linearly through the collar it has
  designed, and refuses a design whose return exceeds the budget, naming a fraction that meets
  it. With a collar, the monitor judges the light *returning*.
- **Scheme and step.** The collar is an in-stage rate seen by the error controller, under a
  step limit that stops a packet crossing it unseen within one step.

Success is judged against an allocated error budget (§7.1):

- the suite's self-focusing case as configured, 10 % wrong in field at 8 cm, is stopped by
  default; a run whose light stays away from the edge completes with no warning beyond INFO;
- a linear beam that sends most of its energy through a collar that passes the design check
  agrees with an independent analytic reference to $10^{-3}$ over the requested aperture;
- measured reflection and transmission against transverse wavenumber agree with the
  calibrated design over the protected band;
- a weak sideband generated at an angle leaves through a requested collar, and returns what
  the design's report predicts, $1.3\times10^{-5}$ of its own field on the witness, against a
  larger-domain reference;
- the same self-focusing case with a requested collar reaches a converged large-radius
  reference to $10^{-3}$.

Cost is reported at matched accuracy. The interface changes below break 0.1.0's, which ADR-058
allows in the 0.x releases; each is listed under "Changed (breaking)" in the changelog.

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

- *At a designed strength, distance beats absorption for a slow beam.* On the suite case, at the
  same radial spacing, with collars designed by §4 on the requested band:

  | outer radius | wall, no collar | designed collar | `Multistep` steps, wall / collar |
  |---|---|---|---|
  | $R$, as the case is configured | $1.0\times10^{-1}$ | none: no room | 149 / – |
  | $R$, today's collar inside the request | – | $9.9\times10^{-2}$ | – / 470 |
  | $2R$ | $3.9\times10^{-5}$ | $1.1\times10^{-3}$ | 68 / 163 |
  | $2.5R$ | $9.1\times10^{-7}$ | $9.5\times10^{-5}$ | – / 125 |
  | $3.6R$ | $6.3\times10^{-9}$ | $1.7\times10^{-6}$ | 68 / 76 |

  In a scan of fractions, band edges and strengths on this case, no collar beat a wall at the same
  outer radius by more than 10 %, and that one was far below the design's strength. The §2 radial
  probe and the designed Cartesian collars agree. This is about designed strengths with room to
  spare: when the wall is close, an undesigned weak collar does help, as the table above shows
  (Cartesian AN 30 at $f = 0.5$ gives $1.18\times10^{-2}$ against its bare wrap's
  $2.93\times10^{-2}$, on the campaign's reference). The slow content that matters lies below the
  designed collar's reflecting threshold.
- *The collar's value is in fast content.* A weak four-wave-mixing sideband at
  $1.5\times10^5$ rad/m, $10^{-6}$ of the pump's energy, leaves through a designed collar to
  $1.1\times10^{-5}$ of its own field. With the collar inert it wraps back, wrong by 0.98.
- *So a default collar would buy little.* At the fractions first proposed it misses the budget: 1.0
  radially gives $1.1\times10^{-3}$ on the suite case, and 0.5 on the Cartesian probe gives
  $3.6\times10^{-2}$. A wider one passes, but a wall at the same radius is better by a further two
  orders, without the collar's cost.

**So the default is detection.** A run that reaches its boundary needs more room, and the
useful thing is to say so. A collar is for the run that has fast light it cannot give room
to, and which declares that light.

**In scope.**

- The monitor's transverse edge, with its warning and stop policy, on by default.
- The real-space collar on request, on the radial, Cartesian (`FreeGrid`), line (`LineGrid`)
  and planar (slab × y) grids: its placement, profile, strength, step limit and design check.
- The transverse-wavenumber ($k_\perp$) ramp of SEP 22, as a numerical safety band,
  on the Cartesian, line, radial and planar grids, on by default.
- The removal ledger: what a collar removes, measured and attributed alongside the temporal
  collar and the mask.
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

For a declared `kappa`, $k_{\min}$ is the smallest $k_z$ at $\kappa_p$ over the *requested* band
and over the run's medium path (§5.2). It is computed, not assumed to be the red edge, because a
material's dispersion need not put it there. §7.1 sets $\Gamma_\mathrm{min} = \log(5\times10^3)$.

A declared `angle` is different. With $\kappa_p(\omega) = \omega\sin\theta/c$, the attenuation
$\Gamma \approx n\,\alpha_0 w\langle P\rangle/\sin\theta$ hardly depends on frequency. Pairing the
bluest $\kappa_p$ with the reddest $k_z$, which never occur together, would over-strengthen the
collar by $\omega_{\max}/\omega_{\min}$: 1.3 on the §2 band and 3.2 on the suite's. The root is
therefore solved per frequency, $\min_\omega\Gamma_\mathrm{wkb}(\kappa_p(\omega),\omega;\alpha_0)
= \Gamma_\mathrm{min}$.

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
$1.5\times10^{-2}$ at 12. That curve is one beam on one geometry, so the design does not use it.
A design that opens its band by raising $\kappa_p$ is the least accurate, because the strength
rises while the beam's own content stays near $\kappa_\mathrm{rms}$.

**The design check.** What matters is how much of the launch reaches the collar and how much
comes back. A product of plane-wave reflection and a ray share cannot say: evaluated on the
witnesses it is wrong by up to nine decades (Part III, R15), and in the other direction once made
depth-aware, because the return lives in diffraction tails. Translation therefore computes it. It
propagates the launch linearly on the design's own operator, and compares it over the requested
aperture with the same launch on a bare grid large enough that the launch does not reach its edge,
taking the largest energy difference over planes up to $L$:

$$
U_\mathrm{ret} = \max_{z\le L}\ \frac{\big\|E_\mathrm{collar}(z) - E_\mathrm{large}(z)\big\|^2_\mathrm{aperture}}{U_\mathrm{launch}} .
$$

- *The launch* is the noise-free one, at its energy-bearing frequencies.
- *The operators.* Radially, the exact generator
  $\mathrm{diag}(i k_z) + F\,\mathrm{diag}(-\alpha/2)\,B$ by eigendecomposition: 16–66 ms per
  frequency at $n_r$ = 128–320. On the Cartesian grid, a one-frequency Strang march, 10–20 ms for 64
  steps at $150^2$–$220^2$, which reproduced five full §2 probes to three digits. On the line and
  planar routes, the same in one dimension, per mode, which has not been run. The large bare grid
  is diagonal, so it needs no march.
- *The verdict.* Above the reflection budget of §7.1, $4\times10^{-8}$ in energy, the design is
  refused. The check is rerun at trial fractions, at the grid's spacing, to name one that passes.
  It also computes the bare wall at the same outer size, and the message says when that returns
  less, which on a slow beam it does.

| suite case, designed on the requested band | $f = 1$ | 1.5 | 2 | 2.6 |
|---|---|---|---|---|
| direct estimate, one frequency, linear | $8.9\times10^{-7}$ | $6.9\times10^{-9}$ | $1.2\times10^{-10}$ | $2.7\times10^{-12}$ |
| measured, full nonlinear run | $1.14\times10^{-6}$ | $9.1\times10^{-9}$ | $1.7\times10^{-10}$ | $2.8\times10^{-12}$ |

It first passes near $f = 1.3$. It agrees on the linear monitor witnesses too: $5.25\times10^{-4}$
against $5.3\times10^{-4}$, and $6.70\times10^{-4}$ against $6.7\times10^{-4}$.

Its limits are stated, not assumed away. The nonlinear agreement is one case at 0.9 of the critical
power with noise off, and a collapsing beam will not follow it, which is the monitor's job, as is
any content generated later (§5.3). Several frequencies and the Cartesian cost at production sizes
are measured in phase 1. This is also where the run length enters: it decides how much slow content
reaches the edge, not the collar's strength.

**The real-space collar cannot cover the whole grid.** On the probe grid the Nyquist
wavenumber is $\pi/\Delta x \approx 10^6$ m⁻¹. Absorbing everything up to $0.9$ of it at
$w = 150$ µm needs $\alpha_0 \approx 10^4$ m⁻¹, which moves $\kappa_\mathrm{lo}$ above the
beam. A $k_\perp$ rate is exact and free, but it acts **throughout the aperture**, so it
cannot be put below any wavenumber that physical light in the aperture may carry. The
design therefore orders three wavenumbers, $\kappa_\mathrm{lo} < \kappa_p \le \kappa_s$:

- **the protected band**, $\kappa \le \kappa_p$: never subject to a $k_\perp$ rate, and above
  $\kappa_\mathrm{lo}$ absorbed at the edge to $\Gamma_\mathrm{min}$ with reflection below the
  budget. What the run must keep, including radiation the user expects to be *generated*, belongs
  here. The user declares it, or accepts the default derived from the launch (§5.2), which covers
  only the launched content's spread. A weak launched or generated signal is protected only
  if its angle is declared: the launch-derived band does not cover a weak tilted seed.
- **the numerical safety band**, $\kappa \ge \kappa_s$: the outer fraction $f_s$ of the
  grid's largest wavenumber on each axis, $f_s = 0.1$ by default as today. Content there is
  removed by the $k_\perp$ ramp wherever it is. The grid's one-half sizing rule
  ([Cartesian](../../guide/free-space/cartesian.md#grid-sizing-and-the-transverse-safety-ramp))
  keeps Kerr products below it. It says nothing of plasma and ionisation sources, whose products no
  multiple of the bandwidth bounds, so the monitor judges what the ramp removes (§5.3).
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
\alpha_s = \frac{2(\Gamma_\mathrm{min} + 2)\,\kappa_{s,1/2}}{k_s\,\ell}.
$$

- $\ell$ is the collar's width $w$ when one is present, and the requested size when none is: with
  no collar, the distance over which the content crosses the aperture and returns.
- $k_s$ is the smallest over the requested band of $\max\big(k_z(\kappa_{s,1/2},\omega),\,
  0.44\,k(\omega)\big)$. The floor is $k_z$ at $0.9\,k$, §5.2's limit: without it the rate diverges
  as $\kappa_{s,1/2}\to k$, which on a band reaching 1.6 µm happens at $\Delta x \approx 0.76$ µm.
  Columns beyond cutoff are already damped by the evanescent operator.
- The 2 nepers are measured: inside the ramp, the collar's own wavenumber broadening defeats
  the additive estimate $\Gamma + \Gamma_s$ by up to 1.7 Np. The design reports the shortfall
  band as measured, not as a crossing point of the two estimates.

Like the collar's, this strength does not depend on $L$.

Content generated *inside* the collar sees at most half the round-trip attenuation, and under
7 % of it from the collar's inner half. A collar is not a place to generate light.

**Rectangular grids.** The rate is one profile with one peak, $\alpha_0(1 - W_x W_y)$. Each face is
designed on the component normal to it, with its own $\kappa_{p,a}$ and width $w_a$, and $\alpha_0$
is the largest of the two roots. So a face's band and reflection depend on its own normal component,
and a packet heading for a corner crosses both collars and is attenuated by both. The launch-derived
band is per axis (§5.2), so an elliptical launch does not over-strengthen its slow axis. The safety
ramp keeps SEP 22's maximum of the normalised components, with the larger $\alpha_s$ of the two axes.
Every two-dimensional witness so far used a square box, so the anisotropic box is in §7.7's matrix.

**Slab modes.** A slab mode is the Cartesian case with $k \to \beta_p$. The extremes run over
modes: a design on one mode alone leaves another at 0.9 of $\Gamma_\mathrm{min}$.

#### 4.5 The profile

The profile is the Planck taper's complement, $P = 1 - W$ (today's shape), on $s\in[0,1]$ and zero
for $s\le0$ (Q3). It has the lowest edge reflection for $\kappa_p w \ge 50$, the fastest tail, a
stretched exponential, and no reflection floor. The classic polynomial CAPs [2, 3] were measured
and are not offered (§10).

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
  aperture, and decay at 0.2–7 % of the overlap estimate. Like all content below
  $\kappa_\mathrm{lo}$ they are not absorbed. The resolved design reports them as reflected, and
  the design check counts what they return.

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
h \le c_\times\,w\,\min_\omega\frac{k_z(\kappa_p(\omega),\omega)}{\kappa_p(\omega)},
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
- $c_\times = 0.15$ and the damping bound were measured under Runge–Kutta only. §7.4 measures them
  under `Multistep` and the low-storage march, which has no error estimate.

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
- the launch's computed own return (§4.4) within the same reflection budget, $4\times10^{-8}$
  in energy.

These are added in amplitude, because they can arrive together.

#### 4.9 Removal and the ledger

**The measure.** This SEP's monitor and ledger use the quadrature-weighted field norm $U$,
as `BoundaryMonitor.split_removal` does today. In that norm the rate $\alpha|E|^2$ is non-negative
at every node, so a collar's removal is too, within the trapezoid's error. The flux-weighted ledger
is the roadmap item of SEP 21 D16, not claimed here.

**Removal is not called absorbed energy.** The collar is not passive in the flux metric. For
multiplication by $A = \operatorname{diag}(\alpha)$ on the nodes, and a nonlocal flux metric $M$
(SEP 34), the rate of change of the flux is governed by $(MA + AM)/2$, which positive $A$ and $M$ do
not make positive semidefinite. SEP 21's radial witness recorded a worst generalised eigenvalue of
$-2.9\times10^{-7}$ of the peak rate. `spatial_removed` is therefore "removed field norm". It may be
read as energy only for a paraxial beam far from cutoff, where the field norm and the flux agree
(SEP 21 D16). This SEP does not add an admission on that eigenvalue, because the first version's was
only ever a guard near cutoff, which §5.2's refusal already provides (§10).

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
  `absorbed_power`'s channels' integrals, and separately from them. With no collar it is zero.
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

`radius` is the aperture the user wants modelled: flat, with no absorption. It changes meaning on
`RadialGrid`, from the outer radius to the requested one, and the grid gains `outer_radius`. The
transforms, nodes and `aeff` use the outer radius.

- The grid counts its samples and is built before the absorber is known, so a fraction cannot
  be applied implicitly. A requested collar on a grid with `collar_fraction=0` is refused,
  naming a fraction that passes the design check. The guide gives 1.0 (radial) and 0.5
  (Cartesian, line, planar) as starting points for fast content (Q2).
- A fraction $f$ costs $(1+f)$ in radial samples and $(1+f)$ to $(1+f)^2$ in radial work
  (measured below), and $(1+f)^2$ in Cartesian memory.
- A fraction with no `SpatialAbsorber` is a larger bare grid with the same requested aperture.
- The window `W` keeps its role as the marker of the region outside the request.
- The other consumers of the window (§3) follow the placement. The coupling plane's radial
  resampling keeps the requested radius and its own collar fraction, rather than doubling an
  outer radius that now includes the collar. The coupling elements' boundary mask is the
  region outside the requested size. Their edge diagnostic, `_edge_nodes`, keeps its outer-tenth
  fallback, so that it does not go blind at $f = 0$.
- **Statistics.** Field in a collar is being removed, so it is not an observable. With a
  `SpatialAbsorber`, energy, widths and peak are reported over the requested aperture, and SEP 23's
  far-field statistics transform the whole grid, as they do today; §7.5's weak-signal witness tests
  whether that suffices. The ledger sums the whole grid, so that it closes. With no absorber, a
  bare fraction is just a larger aperture and everything sums the whole grid.

Measured at fixed spacing, on a dedicated Intel Xeon Gold 6240, $f = 1$ costs 1.9–2.3 times
per step for the grid alone and 2.5–2.9 times with the collar, below $(1+f)^2 = 4$.

#### 5.2 The absorber: a specification, resolved at translation

The user gives a specification; translation resolves it into an immutable design.

```text
from solprop.boundaries import SpatialAbsorber

prop_radial(grid, space, medium, responses, pulse, settings=...)  # the default: safety ramp and monitor
prop_free(..., spatial_absorber=SpatialAbsorber(angle=25e-3))     # a collar, protecting sin(theta) <= 25e-3
prop_free(..., spatial_absorber=SpatialAbsorber(kappa=2e5))       # a collar, protecting |kappa| <= 2e5 rad/m
SpatialAbsorber()                                       # a collar, band from the launch
SpatialAbsorber(angle=25e-3, discard_above=4e5)         # and remove everything above 4e5 rad/m
SpatialAbsorber(attenuation=math.log(1e4))
SpatialAbsorber(real_space=False, safety_fraction=0.2)  # no collar, a wider ramp
prop_free(..., spatial_absorber=False)                  # no safety ramp either; the monitor still runs
```

**`SpatialAbsorber`, the specification.** A frozen dataclass:

- `kappa` (rad/m) or `angle` (rad): the protected band. Neither given means "from the launch".
- `discard_above` (rad/m or `None`).
- `attenuation` $\Gamma_\mathrm{min}$ (field nepers, default $\log(5\times10^3)$).
- `safety_fraction` $f_s$ (default 0.1).
- `real_space` (default `True`): `False` is a ramp-only specification, which is how the default
  run is resolved, and how a thin-plate Cartesian run sets its ramp without a collar.

The angle follows SEP 23's convention, $\sin\theta = c\kappa/\omega$, the angle in vacuum
behind a planar exit face, so a protected angle is a frequency-dependent band
$\kappa_p(\omega) = \omega\sin\theta/c$, handled per frequency (§4.2, §4.7).

**`SpatialAbsorberDesign`, the resolution.** Translation resolves the specification
against the grid, the space, the medium and the launch, in a fixed order of precedence:
explicit `kappa`, then `angle`, then the launch.

- **Launch-derived band.** Per axis, $\kappa_{p,a} = c_b\,\sqrt2\,\kappa_{\mathrm{rms},a}$, $c_b = 3$,
  where $\kappa_{\mathrm{rms},a}$ is the launch's energy-weighted rms wavenumber along that axis over
  active bins, computed on the noise-free launch before noise is added. Radially,
  $\kappa_p = c_b\,\kappa_\mathrm{rms}$ on $|\kappa_\perp|$. For a Gaussian launch
  $\kappa_\mathrm{rms} w_0 = \sqrt2$, whatever the waist. The band's edge says what the collar
  absorbs; it does not by itself protect the launch, which is the design check's job (§4.4).
- **Wavenumbers.** $k_{\min}$ and $k_{\max}$ (§4.2–4.3) are the extremes of $k_z$ at the
  relevant $\kappa$ over the *requested* band, `inband_mask & active_mask`, or a declared band.
  Envelope and field grids carry active bins well outside the request, where the spectral
  collar is already removing content. Taking $k_{\min}$ there doubles $\alpha_0$ and the error,
  halves the stability limit, and makes the report wrong by $10^4$ in returned energy.
  - On a route whose medium varies along $z$ (SEP 34's radial adiabatic transport), they are
    taken over the medium at the path's nodes, so that the entrance alone does not set them, and
    the design check runs at the path's extreme nodes, taking the larger return.
  - On the planar route, $k_z$ is each slab mode's longitudinal wavenumber, and the
    extremes run over modes.
- **The resolved design holds:**
  - the real-space rate on the nodes, if a collar was requested;
  - the safety ramp's rate on the spectral columns;
  - $\kappa_\mathrm{lo}$, $\kappa_p$ and $\kappa_s$, with the intermediate band's report (§4.4);
  - the launch's computed own return, and the bare wall's at the same size (§4.4);
  - the unguarded band of §4.7;
  - the QDHT columns reflected below the band (§4.6);
  - the crossing step limit.

  All are recorded in the output.
- **Refusals.** A requested collar is refused by name, naming a fraction or width that would pass
  where one exists, when:
  - the launch's computed own return exceeds the reflection budget (§4.4);
  - its band edge is below the profile's floor ($\kappa_p w < 22$, §4.3);
  - a requested-band bin is evanescent at $\kappa_p(\omega)$, or has $\kappa_p(\omega) > 0.9\,k(\omega)$;
  - the bands are out of order: $\kappa_p > \kappa_s$, or `discard_above` below $\kappa_p$;
  - the grid has no room for it (`collar_fraction=0`).
- **When it is frozen and rebuilt.** The design is resolved once per translation and is
  immutable. It is part of the translated problem, so it follows that problem's sharing.
  - `with_input` keeps the design. A derived run shares its parent's linear operator,
    transform and step limits, and that sharing is what lets derived runs reuse them. The
    precedent is the optmat rate bindings, which also stay bound to the original input. The design
    check is not rerun. When the new launch's $\kappa_\mathrm{rms}$ exceeds the band the design was
    resolved for, the run warns.
  - A scan point builds and translates its own simulation, so it resolves its own design.
  - Coupling a field to another grid or medium (SEP 31) resolves the child's own specification on
    its own grid. **A coupled child with a collar declares its band,** with `kappa` or `angle`: the
    port is translated before the launch exists, and a launch carries its source's realised noise,
    which would raise $\kappa_\mathrm{rms}$ by 1.45 at $\eta = 10^{-3}$ and 3.4 at $10^{-2}$ on the
    suite's grid. A launch-derived specification there is refused by name. So the design check on a
    child runs on the launch given to `with_input`, or is recorded as not run.
  - A launch that is zero everywhere, with a requested collar and no declared `kappa` or
    `angle`, is refused.
  - A resume translates the simulation again and resolves the same design. The run's resume key
    hashes the specification and the inputs it was resolved from. The derived values, which come
    from a root-finder and an eigensolver, are recorded beside it, not in it, so that a resume is
    not refused across BLAS builds.

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
above ten budgets. `BoundaryCheck` gains `transverse_budget`, defaulting to `budget`, so that
relaxing the transverse edge does not relax time.

What it judges depends on whether a collar is present, because the physics differs.

| situation | measured quantity | judged |
|---|---|---|
| launch clipped by the aperture | the launch's share in the outer 5 % of the *requested* size, at $z=0$ | warn above budget |
| **no collar:** light reaching the wall or the wrap | the *edge* share: the outer 5 % of the *outer* size, its maximum over checks above its launch value | warn and stop |
| **collar:** inadequate absorption, or contamination returning | the returned share in the return strip, and the inward current at the collar's onset | warn and stop |
| **collar:** intended outward flow | guard and collar shares along $z$; `spatial_removed` | recorded at INFO, never a stop |
| **any:** light removed by the safety ramp | the ramp's removal, cumulative, above the launch's share in the band | warn; a stop is settled in phase 1 (Q8) |
| a design's limits | the intermediate band's report and the unguarded band, at translation | recorded |

**With no collar, reaching the edge is the failure.** Everything that reaches the edge comes back
or wraps, so no direction is needed. The direction-resolving estimators below read nothing useful
here: on the suite case, 10 % wrong in field, the outer quarter's strip reads $1.4\times10^{-7}$
in energy, below the warning, and on a linear 100 µm beam the inward current reads zero, because a
slow return coincides with outflow. The edge share tracks the true error instead. On the suite case
at $f = 0$, at its five saved planes:

| z (cm) | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| true aperture error, energy | $6.1\times10^{-7}$ | $2.4\times10^{-4}$ | $3.6\times10^{-3}$ | $1.1\times10^{-2}$ |
| edge share above its launch value | $7.7\times10^{-7}$ | $9.5\times10^{-5}$ | $3.0\times10^{-4}$ | $4.1\times10^{-4}$ |

It stays within 3× at onset, and falls behind by up to 30× only once the error is already large.
So the warning comes near a true error of $10^{-6}$ and the stop near 1–5 × $10^{-4}$. It would stop
this case between 4 and 6 cm, as inferred from those planes; §7.6 tests it at the real check
interval. The launch baseline stops a converging launch being stopped falsely.

- **A larger bare grid reads conservatively.** The edge is at the outer size, not the request's,
  which would falsely stop a run with room to spare: on a reference grid the request's outer 5 %
  reads up to $3.4\times10^{-3}$ on this case. At the outer edge, one frequency, a bare wall reads
  $2.3\times10^{-4}$ at $f = 0.5$ (true error $8\times10^{-7}$), $2.9\times10^{-6}$ at $f = 1$ (below
  $5\times10^{-13}$) and $1.3\times10^{-7}$ at $f = 1.25$. Light at the wall has not yet come back
  within $L$, but it would in a slightly longer run, so the reading errs on the side of more room,
  deliberately.
- **The noise allowance.** The builders' default input noise fills the edge. Its expectation is
  removed by the launch baseline. The judged value is the difference of two noise readings,
  maximised over checks, so the allowance is several standard deviations of one reading:
  $8\,\eta f_g/\sqrt{M}$, for noise share $\eta$, edge area share $f_g$ and $M$ independent modes in
  it. On a one-dimensional probe, the judged value's 95th percentile is about $5\,\eta f_g/\sqrt M$
  and its largest about 7. §7.6 qualifies this on full grids. When the allowance exceeds the warning,
  translation records that the run is insensitive below it, and warns if it exceeds the stop.
- **The stop message names a size to try, and a sample count.** It gives at least $R\,L/z_w$, with
  $z_w$ the plane at which the edge first passed the warning, and says this is a lower bound. On the
  suite case it gives about 3.8 $R$, where 2 $R$ already meets the budget. Light generated at $z_f$
  needs $R(L-z_f)/(z_w-z_f)$, which can be more, so a filament may take a second try. A user who
  knows better raises `transverse_budget`.
- **Cadence.** The edge is read at a fixed check interval, staying out of the temporal guards'
  every-step `ACTIVE` rule, so a run with light near its edge does not slow down. Linear witnesses
  read the same within a few per cent at 1 mm and 0.1 mm.

**With a collar, reaching the edge is intended, and what returns is the failure.** The returned
share is the field near the edge that travels *towards* the aperture's interior.

- **Its strip.** A strip of width $s$ resolves direction to $10^{-6}$ leakage only for
  wavenumbers $q \ge 16/s$, an uncertainty limit. The strip runs over the collar plus
  $\min(w/2, R/4)$ of the request.
- **The estimator.** The inward energy at $q \le -16/s$ of a Kaiser(8)-windowed transform along
  the face's normal, on $\sqrt r\,E$ radially. Its transform is padded twice.
- **The current.** It is supplemented by the cumulative rectified inward current at the collar's
  onset, accumulated at every accepted step, judged above the launch's energy outside the
  request. In the native Adams march, which otherwise returns to the host only at checks, it is a
  native reduction.
- **The floors.** The strip's maximum over checks is judged above its launch-time value, above
  three times the exact noise expectation, and above $10^{-6}$ of its own windowed energy, the
  Kaiser window's leakage. The current is judged above its own noise expectation. When three times
  the strip's floor exceeds the warning, translation records that the run is insensitive below
  it, and warns if it exceeds the stop, as on a Cartesian grid with explicit input noise at 1 nJ.
- The launch baselines join `BoundaryMonitor.resume`, so that a resume judges against the original
  launch.

Measured in two dimensions, one frequency, linear: clean escape, through a Cartesian corner included,
reads at most $2.4\times10^{-8}$; every wall, wrap and collar at one sixteenth of the design strength
stops; the margin to the weakest stop-level failure is about $6\times10^3$. The reflected spread of a
slow launch is under-read 7–100 times, so it warns rather than stops. The design check is what keeps
that case out, and the guide says so. The strip and current have not yet read a nonlinear run, and
the prototype's noise fixes have not been re-run; §7.6 does both.

**The ramp's removal.** The default radial and planar ramps are new, and an under-resolved filament
could otherwise lose energy through them unseen. The removal is cheap, because the columns are
already spectral. It is judged cumulatively above the launch's own share in the band, which default
input noise fills.

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

- *Physics* (`solprop.boundaries`, `solprop.grid`): the specification and the design formulas.
- *Translation* (`solprop.translate.radial`, `.free`, `.planar`, `.spatial`, `.problem`):
  resolution against medium and launch, the design check, the step limits, and the rates on the
  kernel problem.
- *Backends:* the in-stage term (existing), the edge and ramp reductions, the returned-share and
  removal reductions.
- *Monitor* (`solprop.monitor`): the transverse edge and its policy.

**The default's cost.** The edge needs one transverse inverse transform per check: today's
reductions need none, since radially they use spectral weights and on the Cartesian grid Parseval's
theorem. At the fixed cadence of §5.3 this should be small, and §7.7 measures it. The campaign's
collar overhead under `Multistep`, +14 % radially and +22–27 % on the Cartesian grid, is what an
every-step reading would cost, which is why the edge stays out of that rule.

The ramp rides the linear operator and adds nothing per step, with one bound: it enters the exponent
check (`_check_exponent_budget`), so a fixed schedule is refused above $h\alpha_s/2 > 709$ (88.7 in
binary32), and an adaptive run is held to it at its floor. It does not bind on the suite case
($\alpha_s \approx 10^4$ m⁻¹). At $R = 50$ µm and $\Delta x = 1$ µm in silica ($\alpha_s$ about
$2\times10^5$ m⁻¹), the limit is about 7 mm in binary64 and 1 mm in binary32.

**Folding the rate into the linear operator** is exact but not affordable. In the radial
basis, $F\,\mathrm{diag}(\alpha)\,B$ is dense and does not commute with the diagonal linear
operator. The cumulative operator would become a per-frequency matrix exponential, breaking
the diagonal contract of [kernel-contract.md](../kernel-contract.md) §3.

**The safety ramp** is a per-column rate in the linear operator, existing on the Cartesian
route. It is lifted to the radial ($k_r$) and planar ($k_y$ per mode) columns, with its
strength from §4.4. The radial operator is a dense table with `radial_rate_bounds`, so this is
representable.

**Kernel contract.** The problem gains the crossing limit and the radial and planar ramp columns.
`KERNEL_SCHEMA_VERSION` is bumped once, in phase 1. The output schema records the design and the
ledger, also in phase 1.

**Backends, under ADR-061.** A default spatial run never refuses, because the ramp and the edge are
on for every run, and `backend="auto"` can resolve to `numba_cpu`.

- `numpy_ref` (Tier 1): the oracle, carrying everything.
- The native CPU build (Tier 1): everything, with parity.
- `numba_cpu` (Tier 2): the default parts only, the ramp columns and the edge and ramp reductions,
  with parity, because refusing them would break ordinary runs on machines below the native wheel's
  baseline and wherever the native backend is not installed. It refuses a requested collar by name,
  naming the native backend, because the return strip and current are not worth carrying there.
- The native device builds and `mlx` (Tier 3) carry the ramp, an operator table, and either read the
  edge on the host at a check or record the transverse edge as unwatched in the run's output. They
  may refuse a requested collar by name. Device qualification follows the standing policy
  ([backends.md](../backends.md)).

**Cost to be measured, not assumed (§7.7).**

- The in-stage cost per stage.
- The monitor's windowed transforms and reductions, with their scratch. The inward current cost 1.5
  FFT pairs per check in the prototype.
- The design check, per frequency, at production sizes.
- The added samples of a requested collar.
- The steps under the crossing and stability limits.

Scratch is reused from the stage buffers where the backend already holds a physical-space
volume. Any additional buffer is listed in the memory ledger
([backends.md](../backends.md) §8).

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
slab-mode operator), recorded in Part II: the WKB root, the threshold law, the band-edge floor,
the radial and slab corrections, the safety band's shortfall, the content generated in the collar,
and $c_\times$. Phase 1 adds the design check's cost at production sizes and with several frequencies,
and runs it on the line and planar routes.

#### 7.4 Scheme accuracy

The aperture error at three step sizes in-stage, under Runge–Kutta, `Multistep` and the low-storage
march, which sets the fixed-schedule limit and damping bound of §4.7 for each. Each is qualified or
refused by name. The §4.7 crossing counterexample, run in-stage, as a regression test: refused, or
accurate, never silently wrong.

#### 7.5 Physics witnesses

These run in the tests at reduced size. Part II tabulates each witness's definition and expected
reading, with the basis of its tolerance.

- **The design check.** Its estimate against a full run, on the four suite-case fractions of §4.4 and
  the two linear witnesses. The estimate is compared with a run, not with itself.
- **The linear beam.** The probe of §2, on a collar that passes the design check, against reference
  7.2.1, within §7.1's budget. The same beam on a collar that fails it is refused.
- **A protected weak signal,** a four-wave-mixing sideband at an angle in fused silica, on the line
  route at 256 samples (seconds). Protected: the returned energy matches the design's report within
  ±10 %, $1.6\times10^{-10}$ of the sideband's energy, on a design with the requested-band rule, and
  its far-field statistics agree with the larger-domain reference. Declared outside the band: the
  same, at $1.05\times10^{-4}$. Wrapped: wrong by 0.97.
- **A nonlinear field reaching the boundary,** the suite's `spatial_absorber` case scaled paraxially
  to 1 cm, which reproduces the 8 cm errors to two digits in 3–16 s ($1.07\times10^{-3}$ and
  $1.68\times10^{-6}$ on the $f = 1$ and 2.6 designs below), against
  a converged large-radius reference:
  - as configured, stopped by the monitor;
  - on a bare wall at $2R$, passing;
  - a hand-built collar at $f = 1$, true error $1.1\times10^{-3}$, which the design check refuses:
    the monitor's strip and current read it, for the first time on a nonlinear run;
  - on a requested collar at $f = 2.6$, within three times its measured $1.7\times10^{-6}$.

  A ten-step witness is not possible, because the error appears only once the beam reaches the
  edge. The full 8 cm runs only in the suite.
- **An under-resolved filament.** A self-focusing run with plasma on a coarse transverse grid, whose
  ramp removal the monitor reads, with a refined run as its reference. It settles Q8.

#### 7.6 Monitor

One or a few frequencies on `numpy_ref`, asserting Part II's readings within a factor of 3. Clean,
warning and stop classes lie decades apart, and the factor covers differences between the prototype's
transforms and the code's.

- *No collar:*
  - a run that stays away from the edge completes with no warning beyond INFO;
  - a wall and a wrap, slow and fast, are caught by the edge share;
  - a bare wall at $f = 1$ warns, and at $f = 1.5$ does not (§5.3);
  - converging launches read zero until they truly spread back;
  - a launch with tails on the edge is caught at $z = 0$, and not again;
  - the noise allowance holds under each route's noise, on full frequency grids, with the check
    interval the monitor really uses, on the nonlinear suite case as well as the linear ones.
- *A collar:*
  - a strong escape, through a Cartesian corner included, completes with no warning beyond INFO,
    under explicit noise too, with the current's noise subtraction and per-step accumulation;
  - a wall, a wrap, and collars at one quarter and one sixteenth of the design strength are caught;
  - the converging launch needs its baseline;
  - the slow launch on a refused design, built by hand, reads 7–100 times under, and is recorded as
    an expected shortfall.
- *The ramp:* its removal on a clean run stays below the warning under default noise.

#### 7.7 Invariants, ledger and cost

- **The default changes nothing short of the edge.** A run whose light stays away from the edge
  takes the same steps as today's, within the accuracy envelope of [testing.md](../testing.md) §3,
  and agrees with today's field within that envelope, on a launch with no content in the ramp's band.
- **Inside the protected band, nothing changes.** A beam whose content lies in the protected band,
  and whose tail at the collar's onset is below rounding, is unchanged to rounding.
- **The ledger closes.** With temporal and spatial collars both active, the ledger closes:
  $U(z)$ plus the credits plus the mask entry equals $U_\mathrm{launch}$. The test runs in a lossless,
  linear medium, checking at every step, so that its tolerance is the trapezoid's error at the step
  size; a coarser cadence widens it by the ratio squared. Otherwise the safety ramp
  (the transverse channel), the evanescent and material channels and any ionisation loss
  enter the closure too. Checked across a checkpoint and resume, and with field-free planes.
- **Independence of $L$.** The resolved operator is the same for $L$ and $2L$; this is
  checked directly. Propagations to a common plane are compared on a matched fixed schedule,
  where they must agree to rounding. Adaptive runs are compared within the accuracy envelope
  of [testing.md](../testing.md) §3, not against `rtol`.
- **Parity.** `numpy_ref` and the native CPU build on matched schedules and histories, and
  `numba_cpu` for the parts it carries, within the summation-bound tolerances of
  [testing.md](../testing.md).
- **Qualification matrix.**
  - *Routes:* radial, square and anisotropic Cartesian, line, and planar with one and two
    modes.
  - *Grids:* field and envelope.
  - *Precision:* double, and single where the route supports it.
  - *Integrators:* RK, Multistep and low-storage RK.
  - *Backends:* each CPU backend, and what §6 says of the others: the default runs on all of them.

  A cell is either qualified or refused by name, with a refusal test. The design's
  serialisation and resume round trip is tested, including a resume judged against the original
  launch's baselines.
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
  - what the monitor cannot judge: a filament generated late needs more room than the stop message
    names;
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
- `CHANGELOG.md`: each break of §9 under "Changed (breaking)" (ADR-058), citing this SEP.

### 9. Compatibility and impact

Breaking, deliberately, with no compatibility shims:

- `window_factor` becomes `collar_fraction`, defaulting to 0. `RadialGrid(R, n_r)` keeps its wall
  and spacing, and `radius` means the requested size. A script that gave `FreeGrid(rx, nx, ry, ny)`
  loses the inert 10 % margin, so its spacing at the same `n_x` changes.
- `TransverseCollar`, `SimulationSettings.transverse_absorber` and `Absorber.spatial` are
  removed. The safety ramp's strength is no longer a free rate, and it is on by default.
- **A run whose light reaches its boundary now warns or stops** where it used to log and carry on.
  Results that were silently wrong become refusals, which is the purpose.
- An explicit absorber, which used to enable today's collar, is replaced by a `SpatialAbsorber`
  that may be refused.

`KERNEL_SCHEMA_VERSION` and the output schema are bumped in phase 1.

Validated results to redo:

- the suite's `spatial_absorber` case, which as configured would now stop. It gets a requested
  collar at $f = 2.6$ on $n_r = 230$, with a measured error of $1.7\times10^{-6}$, so that the
  suite keeps tracking the collar's cost path. Its revision is bumped. The stop itself is tested
  in §7.5, because a tracking case that stops part-way gives no timing;
- the radial, Cartesian, line and planar performance-tracking cases, for the same check;
- the free-space galleries, all of them: those that set a collar or a ramp
  (`plot_03_line`, `plot_04_planar_hybrid`, `plot_05_low_storage`,
  `plot_06_four_wave_mixing_scan`, `plot_07_far_field_statistics`), and those whose run may now
  warn or stop (`plot_01_radial`, `plot_02_cartesian`, `plot_08_radial_capillary_coupling`);
- the introduction's filament and crossing-beam tutorials;
- the geometry figure.

Downstream Cartesian models that run with a $k_\perp$ ramp and no real-space collar take
`SpatialAbsorber(real_space=False)`, or the default, and re-qualify their numbers.

The device backends (MLX and the native GPU builds) run today's in-stage collar now, with no
refusal. Under the new design a default run keeps running on them, with the ramp and the edge read on
the host or recorded as unwatched, and they may refuse a requested collar by name (§6). `numba_cpu`
also comes to refuse a requested collar, which it runs today.

### 10. Alternatives considered

- **A real-space collar by default** (this SEP as first accepted), at fractions of 1.0 radially and
  0.5 otherwise. Phase 0: radial $f = 1$ gives $1.1\times10^{-3}$ against a bare wall's
  $3.9\times10^{-5}$ on the suite case, and Cartesian $f = 0.5$ gives $3.6\times10^{-2}$ on the
  probe. It costs 2.5–2.9 times per step radially and buys nothing on a slow beam.
- **Outside placement as distance only, by default:** a bare wall at $(1+f)R$. It is robust on the
  suite case, but not within budget on the §2 radial probe ($1.1\times10^{-3}$), and it charges
  every run 2.1 times for room most do not need. The monitor asks for the room when it is needed.
- **A default fraction that follows the design check,** resolved at translation: about $f = 1.3$ on
  the suite case, with an outer size no longer fixed by the grid. A wall at that size does better.
- **A design check from a tabulated reflection,** $\sum S\,T\,|r|^2$. Evaluated on the witnesses, it
  is wrong by up to nine decades (Part III, R15).
- **A passivity admission on the flux metric,** in the first version, on a dense corner surrogate.
  Its worst realistic value was a guard near cutoff, which the $0.9\,k$ refusal covers. It measures a
  metric the monitor and ledger do not use, and it would refuse any grid with an evanescent corner,
  whatever its fraction.
- **Polynomial profiles,** $s^n$, $n = 2, 3, 4$ [2, 3]. $s^3$ and $s^4$ open the band at narrower
  widths, with floors of $\kappa_p w \ge 14.5$ and 16.2 against Planck's 22.3, but reflect up to
  $2\times10^{-4}$ at the periodic grid's wrap point, where they meet 1 with a finite slope. They would
  need a smooth top, which was never calibrated, and the design check would have to be rerun on it.
  They go to the roadmap at Final.
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
- **Requiring a launch-derived band on a coupled child,** by translating the child again against its
  incident launch. It needs new machinery in SEP 31's port, and a band robust to the source's noise.
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

- `collar_fraction`, with the outside placement, `outer_radius` and the statistics of §5.1.
- `SpatialAbsorber` and its resolution, the design check by direct propagation, and the removals of
  §5.2.
- The crossing limit, and §7.4's measurements under each integrator.
- The radial and planar safety ramps, and the ramp's strength.
- The monitor's transverse edges, in both regimes, the ramp's removal, and the ledger.
- Q8, settled by the under-resolved witness.
- Both schema bumps.
- The tests of §7 on `numpy_ref`, the qualification matrix's `numpy_ref` column, the manual
  pages and the tutorial.
- The other backends: a default run still runs on them, with the edge recorded as unwatched until
  they carry it; they refuse a requested collar by name.

**Phase 2 — native CPU.** Everything, with parity, and without the free routes' needless temporal
pair. The native Adams march's inward current.

**Phase 3 — `numba_cpu`, requalification, Final.**

- `numba_cpu`'s default parts, with parity.
- The device backends' default path, compile-checked, with their qualification left to the standing
  policy as a roadmap item.
- The suite's spatial cases, the galleries, the benchmark ledger, the changelog and the Luna page.
- Roadmap items: the split scheme, with its measured potential and accuracy cap; the polynomial
  profiles.

### 12. Open questions

All but Q8 are decided (John Travers). Q1–Q7 were first decided on 2026-10-31. Phase 0 and the
2026-11-01 review changed Q1, Q2, Q3, Q4 and Q6, and this revision states them as they now stand.
Part II and Part III record what changed and why.

1. **Q1 — the default.** *Decided:* no real-space collar. `prop_radial`, `prop_free` and
   `prop_planar` keep a wall or a wrap at the requested size, with the safety ramp and the monitor.
   A `SpatialAbsorber` is on request. `spatial_absorber=False` also removes the ramp.
2. **Q2 — the collar fraction.** *Decided:* the grid's default is 0. The first version's 1.0
   (radial) and 0.5 (Cartesian, line, planar) become the guide's starting points for a requested
   collar. They are not applied implicitly, because a grid counts its samples and is built before
   the absorber (§5.1).
3. **Q3 — the profile.** *Decided:* the Planck taper's complement, confirmed by phase 0, and the only
   one offered. The polynomials go to the roadmap, and the singular profile is dropped.
4. **Q4 — the scheme and the integrator.** *Decided:* in-stage only, keeping `Multistep` as the
   radial and planar default under a collar. The split goes to the roadmap, and `cover` is dropped
   (§10). Whether SEP 21's split temporal collar needs the analogous crossing limit is already on
   the roadmap.
5. **Q5 — the default protected band.** *Decided:* the launch times $c_b = 3$, as the band's default
   upper edge. The design check, not the band, protects the launch. The guide tells users who expect
   generated radiation (four-wave mixing, conical emission) to declare an `angle`.
6. **Q6 — the safety band.** *Decided:* the outer 10 % of each axis's wavenumbers, on by default on
   every geometry, radial and planar included, with the strength rule of §4.4. The passivity
   tolerance is gone with the admission (§10).
7. **Q7 — the monitor's policy.** *Decided:* warn above $10^{-3}$ of the launch's field norm and
   stop above $10^{-2}$, in both regimes of §5.3, through `BoundaryCheck`.
8. **Q8 — whether the ramp's removal stops a run.** *Deferred to phase 1.* It warns at Q7's level from
   the start. The same level would stop an under-resolved filament, but its false-stop rate on the
   galleries is unknown, so phase 1's witness and the galleries' readings decide, and the question
   returns to the maintainer.

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

### Phase 0 — follow-up: the adopted rule and the direct check (2026-11-01)

The review (Part III, R18) found the headline numbers measured under the wavenumber rule P0-13
rejected. They are re-measured here on the requested band, at the same radial spacing, RK and
`Multistep` agreeing to two digits. Energy shares of the launch.

| suite case, at 8 cm | $f = 1$ | 1.5 | 2 | 2.6 |
|---|---|---|---|---|
| $n_r$; $\alpha_0$ (m⁻¹) | 128; 698 | 160; 436 | 192; 318 | 230; 242 |
| field error | $1.07\times10^{-3}$ | $9.53\times10^{-5}$ | $1.28\times10^{-5}$ | $1.68\times10^{-6}$ |
| energy | $1.14\times10^{-6}$ | $9.09\times10^{-9}$ | $1.65\times10^{-10}$ | $2.81\times10^{-12}$ |
| direct linear check, 800 nm | $8.91\times10^{-7}$ | $6.88\times10^{-9}$ | $1.24\times10^{-10}$ | $2.74\times10^{-12}$ |
| check over measured | 0.78 | 0.76 | 0.75 | 0.98 |
| RK / `Multistep` steps | 68 / 163 | 67 / 125 | 67 / 84 | 67 / 76 |

- *The check* first passes the budget near $f = 1.3$. Its floor in this form is about
  $5\times10^{-13}$, from the reference quadrature, so a larger bare-grid reference must meet the
  same.
- *The sideband* on the requested-band design reads $1.1\times10^{-5}$ of its own field at full size,
  and $1.26\times10^{-5}$ at 256 samples, where its report predicts $1.59\times10^{-10}$ in energy
  against $1.53\times10^{-10}$ measured.
- *A bare wall at the outer edge, one frequency,* the edge share against the true error: $f = 0$:
  $2.5\times10^{-3}$, $2.9\times10^{-2}$; 0.5: $2.3\times10^{-4}$, $8.0\times10^{-7}$; 0.75:
  $3.6\times10^{-5}$, $3.2\times10^{-10}$; 1: $2.9\times10^{-6}$, below $5\times10^{-13}$; 1.25:
  $1.3\times10^{-7}$; 1.5: $3.4\times10^{-9}$.

**The witnesses and their expected readings** (R24). Tests assert these. All are `numpy_ref` unless
stated. Noise off unless stated.

| witness | set-up | quantity | expected | tolerance and its basis |
|---|---|---|---|---|
| design check | suite case, 800 nm, the table above | the direct check over a full run | 0.75–0.98 | within a factor 2: the measured spread, plus the linear estimate's one case of evidence |
| design check | linear Gaussian, $R$ 150 µm, spacing 3 µm, 2 cm, $f$ 0.9, $\alpha_0$ 6880 m⁻¹, waist 10 and 15 µm | the direct check over the truth | $5.25/5.3$, $6.70/6.7$ ($\times10^{-4}$) | ±5 %: it is the same linear operator |
| nonlinear | suite scaled 1/8: $w_0$ 35.36 µm, $R$ 106.07 µm, 1 cm, $n_r = 64(1+f)$ | error at 1 cm: as configured; wall at $2R$; collar $f = 2.6$ | $1.0\times10^{-1}$; $4\times10^{-5}$; $1.7\times10^{-6}$ | a factor 3 on the last two; the first above $10^{-2}$; §7.1's $10^{-3}$ is the requirement |
| weak signal | line route, fused silica, §7.5, 256 samples, 1024 bins | returned energy over the report, protected and declared-outside | 0.96, 1.006 | ±10 %: measured 1–4 %, doubled for the grid |
| weak signal | as above, wrapped | error over the sideband's field | 0.97 | above 0.5: any wrap |
| no collar | radial wall, $R$ 300 µm, spacing 4.69 µm, 8 cm, one frequency | edge share at waist 100 µm; 60; 40 | $1.7\times10^{-3}$; $7.0\times10^{-3}$; $1.5\times10^{-2}$ | factor 3: §7.6 |
| no collar | Cartesian wrap, the same | edge share at waist 100 µm; 60; 40 | $2.7\times10^{-2}$; $1.2\times10^{-1}$; $1.9\times10^{-1}$ | factor 3 |
| no collar | bare wall on the same beam, $f$ 1 and 1.5 | edge share | $2.9\times10^{-6}$, $3.4\times10^{-9}$ | factor 3 |
| no collar | converging 120 µm, focus 4 cm; 170 µm, focus 8 cm | edge share above its launch before $z$ = 5 cm | 0 | exact: the baseline |
| collar | Cartesian, $R$ 150 µm, spacing 3 µm, $f$ 0.9, $\alpha_0$ 6880 m⁻¹, one frequency, 2 cm, checks every 1 mm | strip on clean corner escape; wrap; $\alpha_0/4$, tilt $3\times10^5$ | $\le 2\times10^{-8}$; 0.4; $1.7\times10^{-3}$ | factor 3 |
| collar | radial, the same | strip on clean ring $3\times10^5$; wall | $1.4\times10^{-8}$; 0.43 | factor 3 |
| collar | clipped launch, 80 µm | launch guard share, Cartesian and radial | $3.2\times10^{-4}$, $6.7\times10^{-4}$ | factor 3 |
| collar | radial waist 10 µm on that collar, refused by the check, built by hand | truth; strip | $5.3\times10^{-4}$; $7.1\times10^{-5}$ | an expected shortfall, recorded |

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
- 2026-11-01: Opus 5.5 completed an AI review of the rewritten Part I against main at 9d5d92f and
  the campaign's results. Findings R15–R35 below, one of them blocking; Part I is unchanged.
  The maintainer's decisions and the response are pending.
- 2026-11-01: Maintainer decisions on R15–R35 (John Travers). Opus 5.5 (drafting agent) re-measured
  the headline numbers on the requested band, revised Part I and answered every point (*Response to
  the 2026-11-01 review* below). Status stays `Draft`; re-acceptance is pending.

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

### AI review — 2026-11-01 (Opus 5.5)

An independent instance, with no part in the rewrite, given Part I, the code and the campaign's
results, not the drafter's conclusions. It is the same model as the drafting agent, so this review
is not independent of that model's habits. Its text follows as written.

**Verdict.** One blocker, R15: the design check has never been evaluated. As written it refuses the design §9 mandates, and a simpler exact replacement exists. Nine P1 points should change before re-acceptance. The monitor principle, the theory of §4.1–4.3 and the in-stage decision are sound.

"Verified" means I re-derived it, read it in the code, or probed it numerically. "Suspected" means I reasoned it but did not run it. All probes are one frequency, NumPy/SciPy, on a shared Apple M5 Max, so timings are orders of magnitude only.

#### P0

**R15. §4.4, §4.8, §5.2, §7.3, §7.5, §9: the design check, as written, refuses the design the SEP mandates. It is wrong by up to nine decades, and an exact, cheap estimate is available.**

*Problem.* $U_\mathrm{ret} = \sum S\,T\,|r|^2$ has never been computed. No campaign script implements it, and Part II only proposes it (P0-14). I evaluated it as follows:
- $r(\kappa)$ by Riccati integration of §4.1's ODE. This reproduces the calibration's $r(\kappa_p)$ at $\kappa_p w$ = 20, 25, 50, 100 to two digits (3.63e-4, 1.07e-4, 1.01e-6, 1.70e-9).
- $S$ the Gaussian angular spectrum.
- $T$ the Gaussian's ray share reaching the collar by $L$. This $T$ integrates to the correct linear share, 3.0 % on the suite case.

I read the budget as 4e-8 in energy (2e-4 in amplitude, §7.1).

| witness (field-norm energy returned) | §4.4, T = reaches the collar | T = reaches the turning point and returns | direct linear propagation | measured |
|---|---|---|---|---|
| suite, f=1, α0=1352 | 2.4e-3 | 2.7e-6 | 3.2e-6 | 4.3e-6 |
| suite, f=1, α0=699 | 1.4e-3 | 1.0e-7 | 8.9e-7 | 1.1e-6 |
| suite, f=1.5, α0=844 | 7.0e-4 | 1.7e-11 | 2.7e-8 | 3.6e-8 |
| suite, f=2, α0=616 | 2.1e-4 | 8e-17 | 4.6e-10 | 6.2e-10 |
| suite, f=2.625, α0=463 | 4.8e-5 | 2.6e-18 | 6.9e-12 | 8.4e-12 |
| linear, waist 10 µm, f=0.9 | 6.0e-3 | 4.1e-4 | 5.25e-4 | 5.3e-4 |
| linear, waist 15 µm, f=0.9 | 9.8e-3 | 5.1e-4 | 6.70e-4 | 6.7e-4 |
| linear, waist 15 µm, f=2 | 1.5e-5 | 1e-21 | 6.6e-9 | 6.7e-9 |

"Measured" is the excess over the wall twin, squared, on the suite rows, and the truth on the linear rows.

*Consequences of the §4.4 form.*
- It refuses §9's and §7.5's f = 2.6. With the in-band strength (α0 ≈ 240) it reads 1.5e-5, 370 times the budget.
- It would first admit about f = 8, which is $n_r \approx 580$, on a case the campaign shows passes at f = 1.5.
- Making T depth-aware swings the error to 13 decades low. The return lives in diffraction tails that a product of plane-wave reflection and a ray share cannot represent, so this is not a tuning problem.

*Also not supported.* The statement that for a beam filling its collar the check "reduces to … $w \gtrsim 20\,w_0$". On the §2 radial probe, w = 10 $w_0$ (f = 1, α0 = 1043) returns 6.8e-10 by direct propagation, 60 times inside the budget.

*Recommended change.* Replace the formula and the table with the direct linear propagation of the noise-free launch, with and without the collar, on the design's own operator:
- **Radial:** the exact generator $\mathrm{diag}(i k_z) + F\,\mathrm{diag}(-\alpha/2)\,B$ by eigendecomposition. It took 16–66 ms at $n_r$ = 128–320, comparable to the passivity admission.
- **Cartesian:** a one-frequency Strang march. 64 steps at 150²–220² took 0.01–0.02 s and reproduced five full solprop §2 probes to three digits (2.94e-2/2.93e-2, 3.62e-2/3.62e-2, 1.18e-2/1.18e-2, 6.42e-4/6.42e-4, 7.39e-5/7.40e-5).
- **Line and planar:** the same in 1-D, per mode. Suspected, not run.

This handles tilted, converging and arbitrary launches, and can be rerun at trial fractions to name one. A few frequencies should replace my single 800 nm; I did not test that.

*Scope of the evidence.*
- The nonlinear agreement (ratio 0.76–0.82) is one case at 0.9 $P_\mathrm{cr}$, noise off.
- A collapsing beam will not follow it, which is the monitor's job.
- I did not measure the Cartesian cost at production sizes.

#### P1

**R16. §5.1, §5.3: a grid with a fraction and no absorber is falsely stopped, or the guard is undefined.**
- §5.3 defines the guard as "the outer 5 % of the requested size", and its no-collar row judges "the guard's share". §5.1 allows "a larger bare grid with the same requested aperture".
- *Verified.* The campaign's nonlinear series reads the request's outer 5 % on a large-radius reference at 1.9e-6, 2.8e-4, 1.6e-3 and 3.4e-3 at 2, 4, 6 and 8 cm. That is 34 times above the stop level (1e-4), on a run whose bare wall at 2R has a true error of 3.9e-5 in field (1.5e-9 in energy).
- *Change.* Separate the two regions. The launch-clip check stays at the request. The no-collar edge reading sits at the outer 5 % of the *outer* size.

**R17. §6, §7.7, §9, §11 against ADR-061 and SEP 0 on main: backend order, and default-on features refused.**
- Main's SEP 0 orders `numpy_ref`, then native. `numba_cpu` takes a feature only "where the SEP says so", weighing who would run it. §6 and §11 still use the older order, with `numba_cpu` as phase 2 and no such statement.
- More serious: the ramp and the monitor edge are *on by default*, while §9 says device backends "either carry the new parts … or refuse those parts by name". Refusal then applies to every default spatial run. On GPUs that holds until the later device pass. On `numba_cpu` it means any machine where `backend="auto"` resolves there (ADR-061), and the documentation build.
- The only opt-out is `spatial_absorber=False` plus `boundary_check=None`, which also removes the temporal monitor.
- *Change.*
  - Reorder the phases.
  - State that `numba_cpu` carries at least the default parts.
  - Define a non-refusing default on devices: the ramp is an operator table, and the monitor has a host reference path (`StepDriver._boundary_energies`). A check could transfer the state, or the run could record "transverse edge unwatched".
  - The collar and return strip may refuse.
- *Suspected.* The native device build may inherit the CPU code and need only qualification. I did not check this.

**R18. §1, §2, §7.5, §9: the headline numbers were measured under the k_min rule that P0-13 rejected.**
- *Weak signal.* "4e-6" and §1's and §7.5's "$10^{-5}$ of its own field" come from design P (active-bin $k_\min$). The SEP's own rule is design Pr, which the campaign measured at **1.1e-5**, so it fails the stated criterion.
- The 1e-5 has no basis in the budget either. §7.1 allots $e^{-\Gamma_\min} = 2\times10^{-4}$ at $\kappa_p$. Pr's value simply reflects where this sideband sits ($\kappa = 0.75\,\kappa_p$).
- *Suite.* "2.9e-6" (§2, §9) is f = 2.625, $n_r$ = 232, α0 = 463 on active bins, not the stated f = 2.6, $n_r$ = 230 under §5.2's rule. "2.1e-3" is also active-bin (in-band 1.1e-3).
- *Change.* Re-measure under §5.2's rule, and derive the sideband test from $\Gamma_\mathrm{wkb}(\kappa_i,\omega_i)$ with a stated margin.

**R19. §4.2, §5.2, §4.7: an `angle` band over-strengthens the collar by $\omega_\max/\omega_\min$.**
- *Verified by derivation.* With $\kappa_p(\omega)=\omega\sin\theta/c$, $\Gamma = k_z\alpha_0 w\langle P\rangle/\kappa_p \approx n\,\alpha_0 w\langle P\rangle/\sin\theta$, which is independent of frequency.
- §5.2 takes $\kappa_p$ at the bluest bin, and §4.2 takes $k_\min$ at the reddest. That pair never occurs together. α0 comes out too large by 1.3 on the §2 band and 3.2 on the suite's band. The campaign found the error linear in α0 at fixed width.
- The same pairing shortens the crossing limit by the same factor, and can trigger the $\kappa_p > 0.9k$ refusal for a modest angle.
- *Change.* Solve $\min_\omega \Gamma_\mathrm{wkb}(\kappa_p(\omega),\omega;\alpha_0)=\Gamma_\min$ per frequency, and treat the limit and the refusal the same way.

**R20. §5.2: the coupling rule is not implementable as written.**
- *Verified.* `Simulation.input_port()` calls `self.translate()` first (`driver/simulation.py:2138`). The launch is computed for that port. `with_input` then shares `linear` and `transform` by identity (`_SHARED_LEAVES`).
- Three bullets therefore collide:
  - a launch-derived child is "translated against its actual incident launch";
  - `with_input` keeps the design;
  - a zero launch with a requested collar and no declared band is refused, which fires at `input_port()`.
- A launch also "carries the realised field of its source", so no noise-free launch exists. $\kappa_\mathrm{rms}^2$ gains $\approx\eta\kappa_\max^2/2$. On the suite grid (κ_max = 6.7e5) that raises $\kappa_\mathrm{rms}$ by 1.45 at η = 1e-3 and 3.4 at η = 1e-2, and α0 follows.
- *Change.* Either require a declared `kappa` or `angle` on a coupled child, or specify a re-translation step that leaves the port's identity intact. Consider an energy-quantile band, which the campaign's weak-signal findings already suggested and which also survives clipped launches. On a clipped launch $\kappa_\mathrm{rms}$ is grid-dependent (suspected).

**R21. §4.9, §5.2: an open passivity item was dropped from Part I.**
- The campaign recorded that an evanescent corner gives $-\infty$ and that "the prototype refuses". Part I says only "solved densely at the smallest-$k$ bin".
- *Derived.* $\kappa_\mathrm{corner}=\sqrt2\,\pi/\Delta x$ exceeds $k$ at 1.6 µm for $\Delta x<1.1$ µm, and at the active band's 3.1 µm for $\Delta x<2.2$ µm. Such a run is refused whatever its fraction, so the "naming the fraction" message cannot be honest.
- §4.9 also does not say whether the bin is taken over the active or the requested band, while §5.2 insists on the requested one.
- The admission measures the flux metric, but the monitor and ledger use the field norm, whose form is $A\ge0$. By the SEP's own account it is "a guard near cutoff", which §5.2's $0.9k$ refusal already covers.
- *Change.* Define it on the requested band's propagating subspace, or drop the admission.

**R22. §4.4: $\alpha_s$ is undefined on fine grids, on the default path.**
- $\alpha_s = 2(\Gamma_\min+2)\kappa_{s,1/2}/(k_\min\ell)$ is right at its midpoint (re-derived). But $k_\min$ is not defined here, and if it is $k_z$ at $\kappa_{s,1/2}$, it reaches zero when $\kappa_{s,1/2}\to k$ at the red edge. With a band to 1.6 µm that is $\Delta x\approx0.76$ µm.
- Every default run on a fine grid hits this formula.
- *Verified from code.* $\alpha_s$ enters the exponent check (`translate/problem.py`, `_check_exponent_budget`). An adaptive run is held to it only at its floor, shrinking through rejected trials. A fixed schedule is refused above $h\alpha_s/2 > 709$ (88.7 in binary32).
- On the suite ($\alpha_s \approx 1.1\times10^4$ m⁻¹) this does not bind. At R = 50 µm, Δx = 1 µm in silica ($\alpha_s\approx1.7\times10^5$), the limit is 8.5 mm in binary64 and 1 mm in binary32, so §6's "costs nothing per step" is not unconditional.
- *Change.* State the bin rule, and cap or exclude columns near cutoff. The evanescent operator already damps them.

**R23. §1, §5.3, §4.4: the ramp's removal is judged by nothing, contrary to §1's promise.**
- The default radial and planar ramps are new. Today's radial runs have none.
- §5.3's table has no row for the ramp. Its removal appears only in `absorbed_power`, a statistic the user must request (`OPERATOR_STATISTICS`).
- §4.4's "the one-half rule puts all physical content below it" is a Cartesian rule. The guide says plasma sources have no such bound and "only refinement judges the grid", and that physical products may lie between $K$ and $k_{x,\max}$.
- An under-resolved filament would then lose energy silently, in the first use case of §2 (suspected, not run).
- *Change.* Judge the removal or the band's share at Q7's levels. It is cheap, because the columns are already spectral. Add a witness.

**R24. §7.5, §7.6, §7.3, §7.1 against SEP 0 "A SEP stands alone" and "says what will set each tolerance".**
- §7.6 asserts "phase 0's readings within a factor of 3", but most of those readings and the witness definitions live only in the private harness. Part II gives ranges. The calibration behind the $r(\kappa)$ table is not in the SEP.
- Tolerances without a stated basis:
  - the factor 3;
  - the 1e-5 of R18;
  - the $\pm10$ % (measured 1–4 %, but the margin is not stated);
  - "within the step's error" for ledger closure, although §4.9's credit is a trapezoid at the check cadence (up to 8 steps), not per step;
  - "to rounding" in §7.7's protected-band test, which needs a beam whose tail at the collar is below rounding.
- §1 says the suite case with a collar reaches $10^{-3}$, and §7.5 says $10^{-5}$.
- *Change.* Tabulate the witnesses and expected readings in Part II, and state each basis.

#### P2

**R25. §7.5, §7.6: verification gaps.**
- *Verified from the findings.* The return strip and current have never read a nonlinear run. Every "measured in two dimensions" value is linear, one frequency, noise off for the headline numbers.
- A hand-built f = 1 collar on the nonlinear witness (true error 1.1–2.1e-3) would be the obvious test, and §7.5 has no monitor assertion on it.
- $c_\times = 0.15$ and $\alpha_0 h/2\lesssim1.3$ were measured under RK only. The low-storage march has no error estimate (ADR-050) and a different stability interval.
- The fixes adopted in P0-18 (subtracting the current's noise expectation, accumulating at each accepted step, padding of 2) were adopted without a re-run. The prototype's false warning on every clean run under explicit noise is therefore not shown to be gone.

**R26. §6, §5.3: the monitor's cost is understated.**
- §6 calls the guard "a norm reduction". Today's reduction needs no transverse transform: radial uses `spectral_weights` columns, Cartesian Parseval. A guard needs one transverse inverse per check.
- Under the existing `ACTIVE` rule a check runs at *every* step once a share exceeds 1e-8. The campaign's Multistep collar overhead (+14 % radial, +22–27 % Cartesian) is the right scale, so "wall time should not change" holds only if the guard stays out of the busy rule. Say which.
- The inward current "at every accepted step" cost 1.5 FFT pairs per check in the prototype. It must also run inside the native Adams march (ADR-060), which otherwise returns to the host only at checks.

**R27. §5.3: the noise allowance is not a 3σ bound, and a floor was dropped.**
- *Probe.* A Dirichlet line, flat noise, 256 checks over 8 cm. The judged quantity (maximum over checks minus launch) exceeds $3\eta f_g/\sqrt M$ in 80 %, 63 % and 43 % of draws at $n_\omega$ = 1, 16 and 256, by up to 2.35 times. The formula is one reading's σ, but the judged value is a difference of two readings, maximised.
- At 1 pJ on the harness's noise scaling (η ≈ 3e-3, 216 bins) the allowance is about 3.5e-5, above the warning. The guard regime needs the "insensitive below" record that §5.3 gives only to the strip.
- *Verified by reading.* The findings' floor included $10^{-6}$ of the strip's windowed energy (Kaiser leakage at $qs = 16$ is 1.1e-6). Part I dropped it. *Suspected:* false warnings when most of the beam dwells in the strip.

**R28. §5.3: the stop message's size is generous only for diffraction from z = 0.**
- Light generated at $z_f$ reaching the guard at $z_w$ needs $R(L-z_f)/(z_w-z_f)$, not $RL/z_w$.
- With $z_f$ = 5, $z_w$ = 5.5, L = 8 cm that is 6R against the named 1.45R, in the filament use case. The failure is an iteration, not a wrong result.
- *Change.* Word it as a lower bound, and name the sample count too.

**R29. §4.4, §5.2: rectangular grids are not fully defined.**
- Faces are "designed on the component normal" with $w_x\ne w_y$, but the composition of two $\alpha_0$ is unspecified. Every 2-D witness and the passivity value used a single $\alpha_0(1-W_xW_y)$ on a square box.
- $\kappa_\mathrm{rms}$ is on $|\kappa_\perp|$, one number, so an elliptical launch over-strengthens its slow axis.
- $\kappa_{s,1/2}$ and $\ell$ are singular in $\alpha_s$.
- Refusals for $\kappa_p>\kappa_s$ and $\kappa_d<\kappa_p$ are missing from §5.2's list, though §4.4 requires the order.

**R30. §5.2, §9: a ramp-only configuration cannot be expressed.**
- Removing `TransverseCollar` leaves no way to set `safety_fraction` or `discard_above` without a collar: `SpatialAbsorber` means a collar, and absent means the default.
- That is today's documented Cartesian recipe, and §2's thin-plate use case.
- *Change.* Add a ramp-only specification.

**R31. §5.1, §5: statistics and `radius` with f > 0 are unspecified.**
- `radius` changes meaning from outer to requested (for example `aeff` is $\pi R^2$ in `translate/radial.py`; the node and hankel consumers follow).
- Nothing says whether energy, widths, peak or SEP 23's far-field statistics cover the request or the whole grid. §2 names the far field as the observable. The witness used its own windowed aperture spectrum.
- Coupling's `_edge_nodes` must keep its outer-tenth fallback, or the coupling's edge diagnostic goes blind at f = 0.

**R32. §5.2, §5.3: resume.**
- `configuration_hash` is a SHA-256 of canonical JSON. Putting a LAPACK eigenvalue and root-found values into the key refuses a resume across BLAS builds. The batch code already quantises measured entries for that reason (`driver/batch.py`).
- *Change.* Hash the specification and inputs, and keep derived diagnostics outside or quantised.
- The transverse launch baselines must also join `BoundaryMonitor.resume`, which restores named fields only.

**R33. §1, §2, §4.4, §5.3: measured claims stated beyond their evidence.**
- "No collar … beat a wall … the probes above agree on all three geometries." §2's own Cartesian AN 30 at f = 0.5 reads 1.18e-2 against its bare wrap's 2.93e-2 (reproduced by my probe), so it beats the wrap by 2.5 times. On the radial scan a very weak collar beat its wall by 8 %. The claim holds for *designed* strengths. It suggests the design's strength is too high for slow beams, rather than that distance always wins.
- "On the suite case … the inward current reads zero." The campaign's suite row says n/a; the zero is from a linear 100 µm waist.
- The "one curve" in $\kappa_\mathrm{lo}/\kappa_\mathrm{rms}$ and "3–4.7" are one beam and one geometry.
- "Stops between 4 and 6 cm" and the full-frequency noise behaviour are inferred from five planes. §1's first criterion depends on them, though §5.3 does flag the noise part.

**R34. §4.5, §7.3: polynomials are "offered only with a smooth top", which was never calibrated.**
- The $(A,\gamma)$ fits and floors are for tops that hit the wrap kink.
- §7.3's "done in phase 0" does not cover them.
- *Change.* Calibrate in phase 1, or move them to the roadmap.

#### P3

**R35. Stale references and wording.**
- §1 "no release depends on the present interface": 0.1.0 is released. ADR-058 still permits the break, but each one must be listed under "Changed (breaking)", which §8 and §9 do not mention.
- `SCHEMA_VERSION` should be `KERNEL_SCHEMA_VERSION` (ADR-058).
- The memory ledger is `backends.md` §8, not `benchmarks.md`.
- §5.2 "share leaves in a lockstep batch": `driver/batch.py` refuses all spatial geometry ("Batch currently requires fibre geometry").
- `prop_radial(...)` in §5.2 omits the required `settings=`.
- §4.6 "the protected band may not include them" conflicts with "protected: $\kappa\le\kappa_p$".
- §4.9 "Removal is signed": in the field norm the rate $\alpha|E|^2$ is non-negative, and only floor subtraction makes it negative.
- §5.3 omits the strip's $\min(w/2,R/4)$.
- `BoundaryCheck.budget` is shared with the temporal guards, so raising it for the transverse edge also relaxes time.
- The header's `Reviewed` rests on reviews of the earlier text. A human re-read of the rewritten Part I is still owed before re-acceptance.

#### Checked and found sound
- §4.1's stationary form, the exact symbol, the WKB and two-face counting, $\Gamma_\min$, the Planck $\kappa_\mathrm{lo}$ arithmetic ($w\approx19.4\,w_0$), and $\alpha_s$ at its midpoint (all re-derived).
- §4.7's counterexample arithmetic. In-stage over split is well supported.
- The principle of §5.3's no-collar guard, and that the strip is blind there. The tables match the campaign's results.
- §2's and §5.1's phase-0 numbers match the harness, apart from R18 and R33.
- The radial ramp is representable: the operator is a dense table with `radial_rate_bounds`.
- The coupling plane's doubling logic follows the new placement.
- §3's code claims match main.
- `with_input` keeping the design looks tolerable. On the f = 2.625 design, waists of 34–150 µm return at most 1.7e-10 by direct propagation, inside the budget. That is one design, and the check is not re-run.
- Roadmap, index and draft status are consistent on the branch.

#### Not checked
- No solprop propagation was run, so nothing on the real monitor, the planar route, native or device kernels, or the full suite.
- The Cartesian direct estimate at production sizes, or with several frequencies.
- Line and planar direct estimates.
- The galleries' readings under the new monitor.
- The literature citations.

#### The drafting agent's verification of this review

Before recording it, the drafting agent re-ran R15's and R27's probes and checked the code and
evidence claims. The numbers below are from that re-run.

- *R15, reproduced.* The direct linear estimate gives $3.2\times10^{-6}$, $2.7\times10^{-8}$ and
  $6.9\times10^{-12}$ on the suite case at $f = 1$, 1.5 and 2.625, and $5.25\times10^{-4}$ and
  $6.70\times10^{-4}$ on the two monitor witnesses. The §4.4 product reads $1.5\times10^{-5}$ at
  $f = 2.6$. The probe's reference quadrature has a floor near $5\times10^{-13}$, so it cannot be
  read below that as written.
- *R15's consequence for Part I's motivation, not raised by the review.* $f \approx 1.5$ passes the
  budget on the suite case, not the $f$ of 8–10 behind P0-15's "up to 30 times the per-step cost"
  and §2's sentence repeating it. That sentence is wrong and needs correcting. The decision itself
  stands, because a bare wall at the same radius remains better ($9.1\times10^{-7}$ in field at
  $f = 1.5$, against $1.65\times10^{-4}$ for the collar).
- *R16, confirmed* by the follow-up's own series: the reference reads $1.9\times10^{-6}$ to
  $3.4\times10^{-3}$ in the request's outer 5 %. The drafting agent had that column and missed it.
- *R17, confirmed:* main's SEP 0 and ADR-061 read as the review says.
- *R18, confirmed:* design Pr reads $1.1\times10^{-5}$, and the suite row is $f = 2.625$, $n_r = 232$.
- *R19, confirmed* by derivation: the band ratios are 1.31 and 3.2.
- *R20, confirmed:* `input_port` translates first.
- *R21, confirmed:* the open item was in the campaign's findings and was dropped by the rewrite.
- *R22, partly confirmed:* the exponent check exists, and the example's order of magnitude holds.
- *R26, confirmed:* the `ACTIVE` rule checks at every step once a share passes its level.
- *R27, reproduced:* at 16 frequencies the judged value's median, $2.2\times10^{-2}\eta$, exceeds the
  allowance, $2.0\times10^{-2}\eta$. The probe is a 1-D Dirichlet line.
- *R30, confirmed:* the rewrite removed `real_space=False` without providing for it.
- *R32, partly confirmed:* the resume key is `RunMetadata.configuration_hash`; its digest was not
  traced.
- *R33, confirmed:* the §2 Cartesian row and the "inward current reads zero" claim are overstated.
- *R35, confirmed:* batches refuse spatial geometry, and the memory ledger is `backends.md` §8.
- *Not independently checked:* R23's and R25's suspected cases, R29, R31, R34.

#### The maintainer's decisions on this review (John Travers, 2026-11-01)

- **R15: direct propagation.** The design check propagates the noise-free launch linearly, with and
  without the collar, on the design's own operator, and refuses above the budget, naming a fraction
  that passes. Several frequencies and the Cartesian cost at production sizes are measured in phase 1.
- **R17: the default never refuses.** The order becomes `numpy_ref`, then native. `numba_cpu` carries
  the default parts. Device runs carry the ramp and check on the host or record the edge as unwatched.
  The collar and the return strip may refuse by name.
- **R20: a coupled child with a collar declares its band** (`kappa` or `angle`).
- **R21: the passivity admission is dropped,** with $\varepsilon_P$ and the corner surrogate. This
  changes Q6's tolerance.

**All the other points: as the review recommends** (John Travers), where the review offered a
choice taking the response's reading below.

### Response to the 2026-11-01 review (Opus 5.5, drafting agent)

- **R15, taken.** §4.4 now computes the launch's return by direct linear propagation against a larger
  bare grid, replacing the formula and the table in §4.8, §5.2, §6, §7.3 and §7.5. The "$w \gtrsim
  20\,w_0$" and "3–4.7" reductions are gone. The review compared with an analytic reference; §4.4 uses
  a larger bare grid, so that any launch works. §2's and §10's wrong "$f$ of 8–10" and "30 times" are
  corrected. A bare wall's return is computed too, and the refusal says when it is better.
- **R16, taken.** §5.3 separates the launch-clip region, at the request, from the edge, at the outer
  size. New measurements there show the reading conservative at $f > 0$, stated in §5.3 and kept
  deliberately. That goes beyond what the review asked, and is worth the maintainer's eye.
- **R17, decided.** §6, §9, §11. The order is `numpy_ref`, native, then `numba_cpu`'s default parts. The
  review left a requested collar on `numba_cpu` open; §6 refuses it by name, under ADR-061.
- **R18, taken.** Re-measured in Part II; §1, §2, §7.5 and §9 use the new numbers. The sideband test
  now follows the report, not a fixed $10^{-5}$.
- **R19, taken.** §4.2, §4.7 and §5.2 solve an `angle` band per frequency.
- **R20, decided.** §5.2. A launch-derived specification on a coupled child is refused. The design
  check runs on the launch given to `with_input`, or is recorded as not run.
- **R21, decided.** The admission and $\varepsilon_P$ are gone from §4.9, §5.2, §7.7 and Q6, with the
  reason in §10.
- **R22, taken.** §4.4 defines $k_s$ with a floor at $0.9\,k$. §6 states the exponent bound.
- **R23, taken.** §5.3 judges the ramp's removal, and §4.4 limits the one-half rule to Kerr products.
  The review offered a warning and a stop. Its false-stop rate is unmeasured, so it warns, and the stop
  is Q8, deferred to phase 1. §7.5 adds the witness, which has not yet been run.
- **R24, taken.** Part II tabulates the witnesses, readings and tolerance bases. §7.7 restates ledger
  closure at the trapezoid's error and the protected-band test with a tail below rounding. §1's
  $10^{-3}$ is the requirement and §7.5's bound the test.
- **R25, taken.** §7.4, §7.5, §7.6 add the nonlinear read, the other integrators and the re-run of the
  noise fixes.
- **R26, taken.** §5.3 keeps the edge at a fixed cadence, out of the every-step rule. §6 states the cost.
  §5.3 makes the current a native reduction in the Adams march.
- **R27, taken.** §5.3's allowance is $8\,\eta f_g/\sqrt M$, on the review's probe, with an "insensitive
  below" record, and the strip's leakage floor is restored. The 8 is one-dimensional, and §7.6 qualifies
  it.
- **R28, taken.** §5.3 words the size as a lower bound and names the sample count, and §8 documents it.
- **R29, taken.** §4.4 and §5.2: one profile with the larger root, per-axis launch band, the larger
  $\alpha_s$, and refusals for out-of-order bands. All of this is design, not measurement, so the
  anisotropic box stays in §7.7's matrix.
- **R30, taken.** §5.2 restores `real_space=False`, which the rewrite had dropped.
- **R31, taken.** §5.1 defines `radius` and `outer_radius`, the statistics and `_edge_nodes`. The review
  named this as unspecified; the statistics rule is the drafting agent's and is worth the maintainer's
  eye.
- **R32, taken.** §5.2 hashes the specification and inputs, with derived values outside the key. §5.3
  adds the baselines to the monitor's resume.
- **R33, taken.** §2, §4.4 and §5.3 restate the claims with their scope. The "reads zero" claim is now
  attributed to the linear witness.
- **R34, taken.** Of its two options, the polynomials go to the roadmap, which is the smaller change.
  §4.5, §5.2, §10 and Q3.
- **R35, taken.** Each stale reference is fixed. §4.9 now says that removal is non-negative in the
  field norm. `BoundaryCheck` gains `transverse_budget`. The maintainer's re-read of Part I remains
  owed before re-acceptance.
