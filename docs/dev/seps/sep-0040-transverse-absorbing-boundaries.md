# SEP 40 — Transverse absorbing boundaries for free-space propagation

| | |
|---|---|
| **SEP** | 40 |
| **Title** | Transverse absorbing boundaries for free-space propagation |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-31 |
| **Resolution** | Accepted 2026-10-31 by John Travers, after his human review, the GPT-6 AI review and the response to it, his decisions on R1–R14 and Q1–Q7, and the pre-acceptance check A1–A11 with the Part I amendments it led to (Part III). The proposal lands with its implementation's single pull request. |
| **Requires** | [SEP 21](archive/sep-0021-absorbing-boundaries.md), [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md), [SEP 34](archive/sep-0034-medium-flux.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers, branch `feature/sep-0040-transverse-boundaries`: started 2026-10-31; phase 0 in progress (interim report 2026-10-31) |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Rebuild the absorbing boundaries of the radial, Cartesian, line and planar free-space
routes, so that light leaving the transverse window is removed rather than reflected or
wrapped, to a stated accuracy at a measured cost.

- **Placement.** The real-space collar moves outside the requested aperture on every
  geometry.
- **Strength.** It is set by a declared *protected* band of transverse wavenumbers and by
  the collar's width, not by the propagation length.
- **Scheme and step.** It stays an in-stage rate seen by the error controller. A step
  limit stops a packet crossing the collar unseen within one step. An exact split factor
  is an option for the integrators that admit it.
- **One object.** The transverse-wavenumber ramp of SEP 22 is kept as a *numerical safety
  band* at the grid's outer wavenumbers, never inside the protected band. It joins the
  collar in one specification, resolved at translation, and is offered on the radial and
  planar grids too.
- **Monitor.** The boundary monitor gains a transverse edge. It judges light *returning*
  from the boundary, not light reaching it.

Success is judged against an allocated error budget (§7.1):

- a linear beam that sends most of its energy through the boundary agrees with an
  independent analytic reference to $10^{-3}$ over the requested aperture;
- measured reflection and transmission against transverse wavenumber agree with the
  calibrated design over the protected band;
- the suite's self-focusing case, whose nonlinear field occupies the collar, reaches a
  converged large-radius reference to the same level.

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
  reach the edge. A single reference size was used: the reference's own convergence was
  **not** established. Phase 0 repeats this with three sizes (§11).
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

What the probe shows:

- The present collar removes only between a quarter and a half of the boundary error, and
  costs 15 to 20 times the steps of a run without one.
- A wider, weaker collar is two orders of magnitude more accurate at a fraction of that
  cost.
- No setting tried reaches $10^{-3}$, and the best radial collar is as wide as the
  aperture. Width, strength and profile all have to be designed.

The table is evidence of the problem, not a qualification. Phase 0 turns it into a
repository script with converged references (§7.2).

The causes, set out in §3:

- a strength tied to the run length;
- a narrow, steep profile;
- placement inside the request on the radial grid;
- a default of no absorber, with only a log message.

**In scope.**

- The real-space collar on the radial, Cartesian (`FreeGrid`), line (`LineGrid`) and planar
  (slab × y) grids: its placement, profile, strength, scheme, step limit and default.
- The transverse-wavenumber ($k_\perp$) ramp of SEP 22, as a numerical safety band beside
  the collar, on the Cartesian, line, radial and planar grids.
- The spatial part of the boundary monitor: launch clipping, outward flow, removal, and
  returned light, with its warning and stop policy.
- The removal ledger: what the collar removes, measured, signed, and attributed alongside
  the temporal collar and the mask.
- The interface: one spatial absorber specification, its resolution at translation, the
  grid's collar fraction, and the keywords of `prop_radial`, `prop_free`, `prop_planar` and
  their `Simulation.from_*` builders.
- Requalifying the suite's spatial cases and the free-space galleries.

**Out of scope.**

- The temporal and spectral collars, settled by SEP 21.
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

This section gives scalings, used to design a collar and to choose what to measure. The
design constants are calibrated in phase 0 on the discrete radial, Fourier and slab
operators themselves (§4.6, §7.3). Nothing below is transcribed into code without that
calibration.

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
\Gamma(\kappa_0) = \frac{k}{\kappa_0}\int_\mathrm{collar}\alpha\,dx
= \frac{k}{\kappa_0}\,\alpha_0\,w\,\langle P\rangle ,
$$

for a profile $\alpha(x) = \alpha_0 P\big((x-x_0)/w\big)$ with $0\le P\le1$ over a width $w$,
and $\langle P\rangle = \int_0^1 P\,ds$. $\Gamma$ falls as $1/\kappa_0$: fast, high-angle
content is the hardest to absorb. Requiring $\Gamma \ge \Gamma_\mathrm{min}$ for every
$\kappa_0 \le \kappa_p$, the protected band's edge, sets the strength:

$$
\alpha_0 = \frac{\Gamma_\mathrm{min}\,\kappa_p}{k_{\min}\,w\,\langle P\rangle}.
$$

$k_{\min}$ is the smallest $k_z$ at $\kappa_p$ over the active band and over the run's
medium path (§5.2). It is *computed*, not assumed to be the red edge, because a material's
dispersion need not put it there. §7.1 sets $\Gamma_\mathrm{min} = \log(5\times10^3)$.

**The propagation length does not appear.** Today's strength, $12.6/(L/20)$, halves when
the run doubles in length with the same beam and the same grid. A collar sized by $\kappa_p$
and $w$ absorbs the same band whatever $L$ is. So does the safety band of §4.4. The whole
resolved operator is therefore independent of $L$.

#### 4.3 Reflection: the width

WKB holds, and the reflection is small, where $q$ changes slowly on the scale of its own
wavelength, $|q'| \ll |q|^2$. At the onset of the collar $q \approx \kappa_0$ and
$q' \approx i k\alpha'/(2\kappa_0)$, so the condition is

$$
k\,|\alpha'| \ll 2\kappa_0^3
\quad\Longrightarrow\quad
\kappa_0 \gg \kappa_\mathrm{lo} \equiv
C_P\left(\frac{k_{\max}\,\alpha_0\,\max|P'|}{2w}\right)^{1/3}.
$$

Here $k_{\max}$ is the largest $k_z$ at $\kappa_\mathrm{lo}$ over the band and path, and
$C_P$ is a profile constant measured in phase 0. Slow, low-angle content sees a steep wall
and reflects. How fast the reflection falls above $\kappa_\mathrm{lo}$ depends on the
profile's smoothness at its onset (§4.5):

- a profile with every derivative zero at its onset, such as the Planck taper's complement,
  gives a reflection that falls faster than any power of $\kappa_0 w$ [3, 5];
- a profile with a finite number of continuous derivatives gives a reflection with a
  power-law tail.

#### 4.4 The protected band, the safety band, and their overlap

A collar absorbs to $\Gamma_\mathrm{min}$ on $\kappa_\mathrm{lo} \ll \kappa_0 \le \kappa_p$.
With the strength of §4.2,

$$
\frac{\kappa_p}{\kappa_\mathrm{lo}} \propto (k\,\alpha_0)^{2/3}\,w^{4/3}.
$$

The width is the stronger lever. This is SEP 21's empirical finding ("width matters most;
weaker is better down to an optimum"), now with a reason.

**Order-of-magnitude checks** against the probe of §2, with $k = 7.85\times10^6$ m⁻¹,
$\langle P\rangle \approx 0.5$, $\max|P'|\approx 2$, $C_P = 1$ and
$\Gamma_\mathrm{min}=\log 10^3$:

| collar | $w$ | $\alpha_0$ (m⁻¹) | $\kappa_\mathrm{lo}$ (m⁻¹) | $\kappa$ up to which $\Gamma\ge\Gamma_\mathrm{min}$ (m⁻¹) | measured error |
|---|---|---|---|---|---|
| R4's shipped radial collar | 30 µm | 3150 | ~$1\times10^5$ | — | $1.0\times10^{-1}$ |
| §2 radial, AN 30 | 150 µm | 1500 | ~$4\times10^4$ | ~$1.3\times10^5$ | $2.2\times10^{-3}$ |
| §2 Cartesian, AN 30 | 75 µm | 1500 | ~$5\times10^4$ | ~$6\times10^4$ | $1.3\times10^{-2}$ |

- For the shipped collar, the beam's diffracting content ($\kappa\sim2/w_0=2\times10^4$ m⁻¹)
  lies below $\kappa_\mathrm{lo}$ and is reflected.
- For the radial AN 30 collar, the probe beam's $\kappa\approx6.7\times10^4$ lies inside the
  band.
- For the Cartesian AN 30 collar the band is barely open. That agrees with its larger error.

These are consistency checks, not calibration.

**The real-space collar cannot cover the whole grid.** On the probe grid the Nyquist
wavenumber is $\pi/\Delta x \approx 10^6$ m⁻¹. Absorbing everything up to $0.9$ of it at
$w = 150$ µm needs $\alpha_0 \approx 10^4$ m⁻¹, which moves $\kappa_\mathrm{lo}$ above the
beam. A $k_\perp$ rate is exact and free, but it acts **throughout the aperture**, so it
cannot be put below any wavenumber that physical light in the aperture may carry. The
design therefore orders three wavenumbers, $\kappa_\mathrm{lo} < \kappa_p \le \kappa_s$:

- **the protected band**, $\kappa \le \kappa_p$: never subject to a $k_\perp$ rate, and
  absorbed at the edge to $\Gamma_\mathrm{min}$ with reflection below the budget. What the
  run must keep, including radiation the user expects to be *generated*, belongs here. The
  user declares it, or accepts the default derived from the launch (§5.2), which protects
  only the launched content's spread.
- **the numerical safety band**, $\kappa \ge \kappa_s$: the outer fraction $f_s$ of the
  grid's largest wavenumber on each axis, $f_s = 0.1$ by default as today. Content there is
  removed by the $k_\perp$ ramp wherever it is. The grid's one-half sizing rule
  ([Cartesian](../../guide/free-space/cartesian.md#grid-sizing-and-the-transverse-safety-ramp))
  puts all physical content below it.
- **the intermediate band**, $\kappa_p < \kappa < \kappa_s$: kept in the aperture, and
  absorbed at the edge only to $\Gamma(\kappa) = \Gamma_\mathrm{min}\kappa_p/\kappa$. The
  resolved design records this band and its weakest attenuation. The monitor's return
  measure (§5.4) detects contamination from it.

A user who knows that no physical light lies above some $\kappa_d < \kappa_s$ may lower the
ramp's start to $\kappa_d$ by an explicit declaration (`discard_above`, §5.2), and so close
the intermediate band. It is never lowered by default.

**The overlap, quantitatively.** For a component at $\kappa$, the attenuation it carries
back into the aperture combines both mechanisms:

$$
\Gamma_\mathrm{tot}(\kappa) = \Gamma(\kappa) + \Gamma_s(\kappa),\qquad
\Gamma_s(\kappa) = \frac{\alpha_s\,P_s(\kappa)}{2}\cdot\frac{2 w\,k_{\min}}{\kappa}.
$$

$\Gamma_s$ is the ramp's attenuation over the distance in which the component crosses the
collar out and back. The ramp's profile $P_s$ is a raised cosine, zero at its start, so a
ramp starting at $\kappa_s$ adds nothing *at* $\kappa_s$. "Covered" is therefore defined by
$\Gamma_\mathrm{tot}(\kappa) \ge \Gamma_\mathrm{min}$, not by where the ramp begins:

- the resolved design evaluates $\Gamma_\mathrm{tot}$ on the grid's wavenumbers and records
  the band where it falls short;
- the ramp's strength is set so that $\Gamma_s \ge \Gamma_\mathrm{min}$ from the ramp's
  midpoint up, which gives $\alpha_s = 2\Gamma_\mathrm{min}\kappa_{s,1/2}/(k_{\min} w)$. This
  depends on $w$, not on $L$.

Content generated *inside* the collar, closer to the wall than $w$, sees less than these
attenuations. Phase 0 measures it (§7.3).

**Rectangular grids.** Each face is designed on the component normal to it ($\kappa_x$ for
the x faces, with width $w_x$; $\kappa_y$ for the y faces, with $w_y$), so a face's band and
reflection depend on its own normal component. A packet heading for a corner crosses both
collars and is attenuated by both. The safety ramp keeps SEP 22's maximum of the
normalised components, so each axis keeps its own safety band.

**Which slow content matters.** Content with $\kappa_0 < \kappa_\mathrm{lo}$ reflects, but
it moves slowly: over the run it travels $L\kappa_0/k$. Content launched or generated well
inside the aperture reaches the collar only if
$\kappa_0 \gtrsim k\,(R_\mathrm{req}-r_\mathrm{src})/L$. Content already near the edge
arrives at any $\kappa_0$. This is where the run length enters: it decides how much slow
content reaches the edge, not the collar's strength.

#### 4.5 Profiles

All are $0 \le P \le 1$ on $s\in[0,1]$ and zero for $s\le0$, in three classes:

- **Smooth at the onset (every derivative zero there):** the Planck taper's complement,
  $P = 1 - W$ (today's shape), and other smooth step functions.
- **Finite smoothness:** the polynomials $P = s^n$, $n = 2, 3, 4$, which join zero with $n-1$
  continuous derivatives. These are the classic CAPs [2, 3], with closed-form parameter
  optima for a given band and width [2]. Their reflection has a power-law tail, not an
  exponentially small one.
- **Singular:** the transmission-free potential of Manolopoulos [4]. It has a second-order
  pole at the far end of the collar, so it satisfies neither $P\le1$ nor a finite
  $\langle P\rangle$, and §4.2's strength formula does not apply to it.
  - Its use requires a cap, $\alpha \le \alpha_\mathrm{cap}$: the pole is truncated where
    the rate reaches the cap, and the cap is chosen so that $\alpha_\mathrm{cap}h/2$ stays
    within the step's representable attenuation.
  - The capped, discretely sampled profile is a different potential. Its reflection and
    transmission are *measured*; the continuum's zero transmission is not assumed.
  - Its optical-unit mapping — the paper's energy scale to $\kappa^2/2k$, its length scale
    to $w$ — is transcribed from the primary paper in phase 0, with a recorded source check.
  - It is admissible only under the split scheme (§4.7), because in-stage its peak rate
    would set the step.

The default is the Planck taper's complement (Q3). The polynomials are measured against
it, and the singular profile is kept only if phase 0 shows a large gain.

**A fair comparison** holds fixed the width, the grid, the protected band and the attained
$\Gamma(\kappa_p)$. It compares the reflection floor over the protected band, the
attenuation over the intermediate band, and the steps taken under the step limits of §4.7.

#### 4.6 The radial grid

The Hankel basis is the Dirichlet eigenbasis of the disc, so a radial collar always has a
perfectly reflecting wall behind it, and $\Gamma$ counts the round trip. Two features of
the radial operator lie outside §4.1's planar model:

- **the curvature term.** $u = \sqrt r E$ satisfies
  $u'' + (q^2 + 1/(4r^2))u = 0$. At the collar, $r \ge R_\mathrm{req}$, the term is small
  against $\kappa_0^2$ only when $\kappa_0 R_\mathrm{req} \gg 1/2$.
- **the lowest columns.** The lowest QDHT column has
  $\kappa R_\mathrm{req} = j_{0,1}/(1+f) \approx 1.2$–$2$ for the fractions considered.
  These columns are near-standing modes of the disc, not travelling packets, and the
  absorbed-band picture does not describe them. A collar attenuates them by its overlap
  with their profile. The resolved design reports the columns below the band, and the
  protected band may not include them.

The design constants for the radial grid are calibrated on the QDHT operator itself
(§7.3), not on the planar model. A $k_r$-diagonal safety ramp is exact on the radial grid,
because the radial linear operator is diagonal in $k_r$. Its largest wavenumber is
$k_{r,\max} = j_{0,N}/R$ [8].

#### 4.7 The scheme, and the collar-crossing step limit

**A step must not let a packet cross the collar unseen.** Over an accepted step $h$, a
component at $\kappa$ moves $h\,\kappa/k_z$ across. If that exceeds the collar's width, the
diffraction part of the step can carry it through the collar and the wall, or around the
period, and back into the aperture. A collar evaluated only at the step's end, or only at
stage nodes on either side of the crossing, never sees it. On a linear problem the source
defect is zero, so the error controller gives no warning either.

An independent paraxial Fourier calculation shows this:

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

Lie and Strang splittings give the same numbers (reproduced 2026-10-31). The split
converges to the generator once the displacement per step, $0.2h$, falls below about
three quarters of $w$ (8 steps: 75 µm).

Every scheme therefore takes a **collar-crossing step limit**:

$$
h \le c_\times\,\frac{w\,k_{\min}}{\kappa_\times},
$$

where $\kappa_\times$ is the largest wavenumber the collar is relied on to absorb. It is set by
`SpatialAbsorber(cover=...)` (§5.2): $\kappa_p$ under `cover="protected"`, the default, or
the safety ramp's midpoint $\kappa_{s,1/2}$ (or `discard_above`, if lower) under
`cover="intermediate"`.

- $c_\times \approx 0.5$ is calibrated in phase 0, for each scheme and integrator.
- Content above $\kappa_\times$ crosses faster. In the safety band the ramp is exact at any
  step. In the intermediate band it is not: under the default cover, content above about
  $(0.75/c_\times)\,\kappa_\times$ may cross the collar within one step, so the attenuation
  $\Gamma_\mathrm{tot}$ of §4.4 is not assured there. The resolved design records this
  *unguarded* band beside the intermediate band, and the monitor's returned share (§5.4) is
  the check on it. Phase 0 measures the step cost of `cover="intermediate"`.
- The limit joins the existing stability limit: `dz_max` is clamped to it, a fixed schedule
  that exceeds it is refused at translation, and the monitor's check interval is held
  within it.
- On the §2 radial probe ($w = 150$ µm, $k/\kappa_p \approx 60$, $c_\times = 0.5$) the limit
  is about 4.5 mm, at least five steps over 2 cm. The cost is small when the collar is wide,
  which is a further reason for width.

**The in-stage scheme (recommended default).** The rate stays a term $-\alpha E/2$ in every
stage, as today.

- Under the step limit, the stages sample the trajectory finely enough, and the embedded
  error estimate sees the collar's action, so the controller governs its error together
  with everything else.
- It changes no integrator's history and needs no first-same-as-last exception.
- Its stability limit, $2r/\alpha_\mathrm{eff}$ (§3), depends on the integrator. On the §2
  probe a peak rate of $1500$ m⁻¹ allows about 4 mm under Runge–Kutta ($r = 3.3$), but only
  about 0.12 mm under the default Adams `Multistep` at order 5 ($r = 0.0877$), at least 170
  steps over 2 cm. The radial and planar routes default to `Multistep`, and the §2 probe used
  Runge–Kutta. Phase 0 therefore measures the in-stage cost under the default `Multistep` on
  the radial and planar witnesses. If collar stability binds those runs, Q4 returns to the
  maintainer.
- It cannot carry a singular or very steep profile, whose peak rate would set the step.

**The split scheme (optional).** The real-space factor $e^{-\alpha(x)h/2}$ is applied once
per accepted step.

- It is exact for the absorber alone and unconditionally stable, so it admits the capped
  singular profile.
- It is first order, and its leading error contains the commutator of diffraction with the
  collar. That error is generated in the collar but **not confined there**: later
  propagation carries it into the aperture, as SEP 21 §4.5 warned. Its accuracy is
  qualified inside the aperture at three step sizes (§7.4); no confinement is assumed.
- It changes the state after the stages, so:
  - Runge–Kutta loses first-same-as-last reuse, as it already does for the temporal split;
  - the Adams march's source history and the low-storage march's registers are not updated
    by a pointwise change, which couples every transverse spectral column. The split is
    therefore **refused by name** under `Multistep` (Python and native marches) and
    `LowStorageRungeKutta` until a later proposal qualifies a history treatment.

**Cost.** The two schemes cost different things:

- *In-stage:* a transverse transform pair per stage on the Cartesian, line and planar routes
  (today's), or one dense $n_r\times n_r$ product per stage on the radial route.
- *Split:* the multiply is nearly free where the temporal split already visits the physical
  field. Otherwise — no temporal collar, or an in-stage temporal collar — it needs its own
  transform pair per accepted step: a dense pair on the radial route.
- *Measurement:* the removal measurement (§4.9) adds a norm reduction in either scheme.

#### 4.8 The error budget behind the design

For the target of $10^{-3}$ in the aperture's field norm (§7.1), the design takes:

- $\Gamma_\mathrm{min} = \log(5\times10^3)$, a returned field below $2\times10^{-4}$ of the
  incident;
- a reflection floor $|r| \le 2\times10^{-4}$, $4\times10^{-8}$ in power, over the
  protected band above its reflecting floor, $m\,\kappa_\mathrm{lo} \le \kappa \le \kappa_p$,
  with the margin $m$ measured in phase 0.

These are added in amplitude, because the two can arrive together. Below
$m\,\kappa_\mathrm{lo}$ the collar reflects by design (§4.3). That content is covered by the
travel argument of §4.4: it reaches the edge only if launched or generated near it, and the
monitor's returned share (§5.4) is the check.

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

- **Admission, not assertion.** A resolved design's worst generalised eigenvalue, relative
  to its peak rate, is computed at translation, for the generator and for the one-step split
  map. A design is admitted when that eigenvalue is bounded below by a tolerance
  $-\varepsilon_P$, which phase 0 sets from the measured values with a stated margin and the
  maintainer approves.
  - The full problem is not affordable at translation. On a $256^2$ Cartesian grid it is a
    dense eigenproblem of order 65 536 at each frequency, and on the radial grid it costs
    $n_r^3$ per frequency.
  - The admission value is therefore computed on a reduced surrogate: one transverse axis at
    a time, at the frequency with the smallest $k_z$, by a matrix-free Lanczos iteration
    for the minimum eigenvalue, with the metric applied by the grid's own transform.
  - Phase 0 validates the surrogate against the full computation on small grids of every
    geometry. A surrogate that does not bound the full value from below, within its margin,
    is not adopted.
- **Removal is signed.** `spatial_removed` may go negative. It is called "removed field
  norm", never "absorbed physical energy". It may be read as energy only for a paraxial
  beam far from cutoff, where the field norm and the flux agree (SEP 21 D16).

**The measure.** This SEP's monitor and ledger use the quadrature-weighted field norm $U$,
as `BoundaryMonitor.split_removal` does today. The flux-weighted ledger is the roadmap item
of D16, not claimed here.

**Attribution.** On an accepted step the treatments act in a fixed, documented order:

1. the spatial factor, $U_0 \to U_1$, credited $U_0-U_1$;
2. the temporal factor, $U_1 \to U_2$, credited $U_1-U_2$;
3. the hard band mask, $U_2 \to U_3$, credited to a separate mask entry.

The credits telescope to $U_0 - U_3$ and do not double-count, although the two factors
commute.

- For an in-stage collar the removal is the integral of its rate,
  $\int\!\!\int \alpha |E|^2$, sampled at the monitor's checks and integrated by the
  trapezoid rule, as for the in-stage temporal collar.
- Only accepted steps are credited.
- Totals persist at checkpoints. A resume continues them; a relaunched child starts at zero
  (SEP 21 D10).
- A plane that keeps no field still accumulates, because the reduction runs on the backend.
- `spatial_removed` is reported in J/m on the line route and in J elsewhere, in the units of
  `absorbed_power`'s channels' integrals, and separately from them. It is reported signed,
  never clipped at zero as the temporal monitor's removed fraction is today.
- The safety ramp's removal is SEP 23's transverse channel of `absorbed_power`. That
  channel exists today only where a separable operator does (Cartesian, line and planar).
  It is extended to the radial ($k_r$) ramp, so that every removal of the field norm has a
  named entry.

### 5. Proposed interface

#### 5.1 The grid: placement

Every spatial grid takes `collar_fraction` in place of `window_factor`, and the collar lies
**outside** the requested size on every geometry:

```text
RadialGrid(radius=150e-6, n_r=96, collar_fraction=1.0)
# outer Dirichlet radius = radius * (1 + collar_fraction); n_r counts every sample
FreeGrid(radius_x, n_x, radius_y, n_y, collar_fraction=0.5)
# periodic half-width = radius * (1 + collar_fraction), as today
```

`radius` is the aperture the user wants modelled: flat, with no absorption. Under this
definition the §2 radial AN 30 collar has `collar_fraction = 1.0`, and the Cartesian ones
0.3 and 0.5.

- The default fraction is 1.0 on the radial grid and 0.5 on the Cartesian, line and
  planar grids (Q2, to be confirmed in phase 0). A fraction $f$ costs $(1+f)$ in radial
  samples and $(1+f)^2$ in dense radial work, and $(1+f)^2$ in Cartesian memory.
- `collar_fraction=0` leaves no room for a collar. The real-space part is then refused, and
  a ramp-only absorber is allowed.
- The window `W` keeps its role as the monitor's region marker.
- The other consumers of the window (§3) follow the placement. The coupling plane's radial
  resampling keeps the requested radius and its own collar fraction, rather than doubling an
  outer radius that now includes the collar. The coupling elements' boundary mask is the
  region outside the requested size.

#### 5.2 The absorber: a specification, resolved at translation

The user gives a specification; translation resolves it into an immutable design.

```text
from solprop.boundaries import SpatialAbsorber

SpatialAbsorber()                                  # protected band from the launch
SpatialAbsorber(kappa=2e5)                         # protect |kappa| <= 2e5 rad/m
SpatialAbsorber(angle=25e-3)                       # protect sin(theta) <= 25e-3, all frequencies
SpatialAbsorber(angle=25e-3, discard_above=4e5)    # and remove everything above 4e5 rad/m
SpatialAbsorber(attenuation=math.log(1e4), profile="planck", scheme="in_stage")
SpatialAbsorber(real_space=False)                  # the safety ramp alone

prop_radial(grid, space, medium, responses, pulse, settings=..., spatial_absorber=SpatialAbsorber())
prop_free(..., spatial_absorber=False)             # wall or wrap, as today
```

**`SpatialAbsorber`, the specification.** A frozen dataclass:

- `kappa` (rad/m) or `angle` (rad): the protected band. Neither given means "from the
  launch".
- `discard_above` (rad/m or `None`).
- `attenuation` $\Gamma_\mathrm{min}$ (field nepers, default $\log(5\times10^3)$).
- `profile`.
- `safety_fraction` $f_s$ (default 0.1).
- `cover` (`"protected"`, the default, or `"intermediate"`): the band the crossing step limit
  of §4.7 guards.
- `scheme` (`"in_stage"` or `"split_step"`).
- `real_space` and `safety` switches.

The angle follows SEP 23's convention, $\sin\theta = c\kappa/\omega$, the angle in vacuum
behind a planar exit face, so a protected angle is a frequency-dependent band
$\kappa_p(\omega) = \omega\sin\theta/c$. The design uses its largest value on the active
band.

**`SpatialAbsorberDesign`, the resolution.** Translation resolves the specification
against the grid, the space, the medium and the launch, in a fixed order of precedence:
explicit `kappa`, then `angle`, then the launch.

- **Launch-derived band.** $\kappa_p = c_b\,\kappa_\mathrm{rms}$, where $\kappa_\mathrm{rms}$
  is the launch's energy-weighted rms transverse wavenumber over active bins, computed on
  the noise-free launch before noise is added. $c_b$ is a phase 0 constant. The design
  records that this protects the launched spread only: radiation generated beyond it lies in
  the intermediate band (§4.4), and the docs say how to protect it.
- **Wavenumbers.** $k_{\min}$ and $k_{\max}$ (§4.2–4.3) are the extremes of $k_z$ over
  active, propagating bins at the relevant $\kappa$. Inactive bins are excluded. Bins
  evanescent at $\kappa_p$ are refused, as is any active bin with $\kappa_p > 0.9\,k$, until
  phase 0 qualifies the approach to cutoff.
  - On a route whose medium varies along $z$ (SEP 34's radial adiabatic transport), they are
    taken over the medium at the path's nodes, so that the entrance alone does not set them.
  - On the planar route, $k_z$ is each slab mode's longitudinal wavenumber, and the
    extremes run over modes.
- **The resolved design holds:**
  - the real-space rate on the nodes;
  - the safety ramp's rate on the spectral columns;
  - $\kappa_\mathrm{lo}$, $\kappa_p$ and $\kappa_s$, with the intermediate band and its
    weakest $\Gamma_\mathrm{tot}$;
  - the unguarded band of §4.7, which a packet may cross within one step;
  - any QDHT columns below the band;
  - the crossing step limit;
  - the passivity admission value of §4.9.

  All are recorded in the output.
- **Refusals and warnings.**
  - A design with $\kappa_\mathrm{lo}\ge\kappa_p$ is refused, naming the width that would
    open the band.
  - A design whose passivity value fails §4.9 is refused.
  - A launch with $\kappa_\mathrm{rms} < \kappa_\mathrm{lo}$ is warned of, because the
    beam's own diffraction would then be reflected.
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
  - A launch that is zero everywhere, with no declared `kappa` or `angle`, is refused.
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

#### 5.3 Scheme

`SpatialAbsorber(scheme=...)`, `"in_stage"` by default (Q4). Both schemes take the crossing
limit of §4.7.

- `"split_step"` is admitted only with `RungeKutta`, and is refused by name under
  `Multistep` and `LowStorageRungeKutta`, under interpolated planes, and on backends not
  yet qualified. The refusal keys on the spatial scheme alone: the temporal split stays
  admitted under every integrator, as SEP 21 left it.
- `"split_step"` is built first on `numpy_ref`, for phase 0 and the phase 1 qualification. It
  is carried to the compiled backends only if phase 0 shows a cost advantage over in-stage
  at matched accuracy (§6). Otherwise those backends refuse it by name, and the option stays
  on `numpy_ref` with a roadmap item.
- `"in_stage"` keeps today's stability check, and refuses the singular profile.

#### 5.4 Monitoring: judge what returns, not what arrives

The monitor gains a transverse edge per face: one for the radial grid, four for the
Cartesian box, two for the line and the slab. It distinguishes four situations, each by its
own measured quantity. Every share is normalised by the launch's field norm $U_\mathrm{launch}$.

| situation | measured quantity | judged |
|---|---|---|
| launch clipped by the aperture | the launch's share in the guard (the outer 5 % of the requested size) at $z=0$ | warn above budget, as for time |
| intended outward flow | guard and collar shares along $z$; `spatial_removed` | recorded at INFO, never a stop |
| inadequate absorption, or contamination returning | the **inward-travelling** share in the return strip | warn above budget; stop above $10^{-2}$ of the launch until qualified (Q7) |
| a design gap | the intermediate band's weakest $\Gamma_\mathrm{tot}$ and the crossing limit, both at translation | a warning |

The returned share is the field near the edge that travels *towards* the aperture's
interior.

- **Its strip is not the guard.** A direction can be resolved only for content whose
  transverse wavelength fits in the strip. A 5 % guard of a 150 µm request is 7.5 µm wide, a
  wavenumber resolution near $8\times10^5$ m⁻¹. The content that reflects lies below
  $\kappa_\mathrm{lo}$, about $4\times10^4$ m⁻¹ on the §2 probe. The return strip, which
  may extend into the collar, is therefore sized from the design, with a wavenumber
  resolution well below $\kappa_\mathrm{lo}$. Its width and estimator are set in phase 0.
- On a Cartesian face it is the energy, in a windowed transform along the face's normal over
  the return strip, at wavenumbers pointing inward.
- On the radial grid it is the incoming cylindrical-wave part of the strip's field. A
  decomposition from the field and its radial derivative is defined only for one
  wavenumber, so the radial estimator is chosen in phase 0 from candidates measured
  against the radial witnesses.
- **The noise floor.** The builders' default input noise fills every strip. As for the
  temporal guards (SEP 21 D13), a share is judged only above the analytic expectation of
  the input noise in that region.

Light leaving cleanly gives a small returned share however much reaches the edge. A
reflecting or under-strength collar gives a large one. Phase 0 qualifies the measure's own
floor (the window's leakage) and the budgets, on a strong-escape run that must complete and
on reflecting and wrapping runs that must be caught (§7.6). The end-of-plane log of
`collar_field_fraction` is retired.

#### 5.5 Composition

- *Scans and coupling:* the specification is plain data, resolved per translation, with
  the sharing and coupling rules of §5.2.
- *Single precision:* the rate or the factor is applied in the state's precision.
- *The low-storage march* (SEP 24) and *the Adams march* (SEP 16, SEP 39): in-stage only
  (§4.7).
- *The planar route:* the real-space collar acts on y alone, per $(p, y)$ row, as today's
  y collar does. The safety ramp acts on $k_y$ alone, per $(p, k_y)$ column, which is
  diagonal there. The design uses the modes' $k_z$ (§5.2).
- *Refused:* a spatial absorber on a modal route.

### 6. Implementation strategies

**Layers.**

- *Physics* (`solprop.boundaries`, `solprop.grid`): the specification, the design formulas
  and the profile library.
- *Translation* (`solprop.translate.radial`, `.free`, `.planar`, `.spatial`, `.problem`):
  resolution against medium and launch, the step limits, the passivity admission, and the
  rates on the kernel problem.
- *Backends:* the in-stage term (existing), the split pass, the returned-share and removal
  reductions.
- *Monitor* (`solprop.monitor`): the transverse edge and its policy.

**In-stage (recommended) against split.** Phase 0 measures both, inside the aperture, at
three step sizes and at matched accuracy, on the linear probe, the self-focusing witness
and a run whose nonlinear field occupies the collar. The default follows the measurement.
The recommendation is in-stage because:

- the controller sees it;
- it disturbs no history;
- the step limit of §4.7 binds either scheme anyway;
- a wide, weak collar keeps its stability limit long under Runge–Kutta. Under the default
  Adams `Multistep` it is about 38 times shorter (§4.7), and phase 0 measures what that costs
  before the recommendation is confirmed for the radial and planar routes.

**Folding the rate into the linear operator** is exact but not affordable. In the radial
basis, $F\,\mathrm{diag}(\alpha)\,B$ is dense and does not commute with the diagonal linear
operator. The cumulative operator would become a per-frequency matrix exponential, breaking
the diagonal contract of [kernel-contract.md](../kernel-contract.md) §3.

**The safety ramp** is a per-column rate in the linear operator, existing on the Cartesian
route. It is lifted to the radial ($k_r$) and planar ($k_y$ per mode) columns, with its
strength from §4.4.

**Kernel contract.** The problem gains the scheme, the crossing limit, the split rate and
the radial and planar ramp columns. `SCHEMA_VERSION` is bumped once, in phase 1. The output
schema records the design and the ledger, also in phase 1.

**Backends, in order.**

- `numpy_ref`: the oracle.
- `numba_cpu`: the existing in-stage kernels, and the split factor in the split kernels.
- The native CPU build: the same, with the radial dense pair reusing the stage's Hankel
  matrices.
- The MLX and native device backends follow the standing device policy
  ([backends.md](../backends.md)) and refuse what they do not yet implement, by name.

**Cost to be measured, not assumed (§7.7).**

- The marginal multiply.
- The transforms a split needs when no temporal split visits physical space: dense
  $O(n_r^2 n_\omega)$ per accepted step on the radial grid.
- The in-stage cost per stage.
- The monitor's windowed transforms and reductions, with their scratch.
- The added samples: $(1+f)$ radially, which is $(1+f)^2$ in dense work; $(1+f)^2$ in
  Cartesian memory.
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
- **Coefficients.** Reflection and transmission are field amplitudes. The incident amplitude
  is the packet's before it reaches the collar. A returned packet is measured at the time it
  re-enters the guard, separated by its inward travel (§5.4).
- **Budget** for the $10^{-3}$ target, added in amplitude:

  | item | budget |
  |---|---|
  | returned transmission, $e^{-\Gamma_\mathrm{min}}$ | $2\times10^{-4}$ |
  | reflection | $2\times10^{-4}$ |
  | transverse and temporal discretisation | $2\times10^{-4}$ |
  | step controller and crossing limit | $2\times10^{-4}$ |
  | reference error | $\le 10^{-4}$ |

  A default is recommended only when every item is met on the witnesses.

#### 7.2 References

1. **An analytic linear reference.** A Gaussian beam's angular spectrum is Gaussian.
   Propagated exactly with $e^{ik_z z}$ and transformed back by adaptive quadrature,
   independently of the QDHT and the FFT, it gives the field over the aperture at any $z$,
   non-paraxial included.
2. **Converged large-domain references** for the nonlinear cases. Three domain sizes at a
   fixed physical spacing. For the QDHT, each run's Fourier–Bessel series is evaluated on a
   common physical grid, because the nodes move with the outer radius. The observed
   convergence order is reported, not assumed from the temporal campaign.
3. **The 1D scattering problem** of §4.1, solved directly on a fine grid. It is checked for
   grid convergence, and it is used for WKB trends only.

#### 7.3 Calibration (phase 0)

Reflection and transmission against $\kappa_0$ for each profile, width and strength, on the
real discrete operators, with the safety ramp **off**, and then with it on:

- Cartesian: a tilted Gaussian in a dilute linear medium;
- radial: an outgoing annular packet;
- planar: a single slab mode, then two.

From these come $C_P$, $c_b$, $c_\times$, the radial corrections, the low-column behaviour,
the cap of the singular profile, and the floor for content generated within $w$ of the wall.
The WKB scalings are compared as trends. Their agreement is not an acceptance criterion; the
discrete measurements are.

#### 7.4 Scheme accuracy

The aperture error at three step sizes for each scheme. The split's observed order, and
where its error lies. The §4.7 crossing counterexample as a regression test: refused, or
accurate, never silently wrong.

#### 7.5 Physics witnesses

- The linear probe of §2 as a repository script against reference 7.2.1, within §7.1's
  budget.
- **A protected weak signal.** A weak sideband generated by four-wave mixing at an angle
  inside the protected band, in a wide aperture. Its far-field port energy (SEP 23) agrees
  with a larger-domain reference. A second run declares it outside the band and shows the
  intermediate-band report.
- **A nonlinear field in the collar.** The suite's `spatial_absorber` case, whose
  self-focusing beam reaches the collar, against reference 7.2.2. Ten steps or a short
  length in tests; the full 8 cm only in the suite.

#### 7.6 Monitor

- A strong-escape linear run completes under the default policy with no warning beyond
  INFO.
- A wall, a wrap, and a deliberately under-strength collar are each caught by the returned
  share.
- A clipped launch is caught at $z=0$.

#### 7.7 Invariants, ledger and cost

- **Inside the protected band, nothing changes.** A beam whose content lies in the
  protected band and stays inside the aperture is unchanged to rounding.
- **The ledger closes.** With temporal and spatial collars both active, the ledger closes:
  $U(z)$ plus the ordered credits plus the mask entry equals $U_\mathrm{launch}$, within
  the step's error. The test runs in a lossless, linear medium. Otherwise the safety ramp
  (the transverse channel), the evanescent and material channels and any ionisation loss
  enter the closure too. Checked across a checkpoint and resume, and with field-free planes.
- **Passivity admission.** The admission values of §4.9, for generator and split map, on
  each geometry.
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
  - *Integrators:* RK, Multistep and low-storage RK, in-stage; RK, split.
  - *Backends:* each CPU backend.

  A cell is either qualified or refused by name, with a refusal test. The design's
  serialisation and resume round trip is tested.
- **Cost.** Steps, wall time and memory on the suite's spatial cases against today and
  against no absorber, at matched accuracy, with the temporal split on and off and at
  several collar widths at fixed physical resolution. A ledger entry in
  [benchmarks.md](../benchmarks.md).

### 8. Documentation

- [Free-space model](../../guide/free-space/model.md): *Spatial collar and statistics* is
  rewritten as *Transverse boundaries*. It covers:
  - the theory in its calibrated form;
  - the protected, intermediate and safety bands;
  - how to protect generated radiation;
  - the step limit, the monitor's returned share and the ledger;
  - a worked example.
- [Boundaries](../../guide/numerics/boundaries.md): links the transverse boundary from
  *Which runs carry one*, and updates the spatial row.
- [Cartesian](../../guide/free-space/cartesian.md): *Grid sizing and the transverse safety
  ramp* is updated for the safety band and the anisotropic box.
- [Grids](../../guide/setup/grids.md) and
  [spatial decompositions](../../guide/setup/spatial-decompositions.md): `collar_fraction`,
  the placement, and the planar ramp.
- A gallery tutorial: *Is my transverse window big enough?*, the spatial companion of the
  temporal one, built on the linear probe and the protected-signal witness.
- [Differences from Luna](../../guide/workflows/luna.md): Luna.jl's free-space routes apply
  no transverse absorber, so a radial run reflects from its outer radius and a Cartesian one
  wraps. solprop absorbs by default.

### 9. Compatibility and impact

Breaking, deliberately, with no compatibility shims:

- `window_factor` becomes `collar_fraction`, and on `RadialGrid` the collar moves outside
  `radius`. A script that gave `RadialGrid(R, n_r)` now gets a larger outer radius and a
  coarser $\Delta r$ at the same `n_r`.
- `TransverseCollar`, `SimulationSettings.transverse_absorber` and `Absorber.spatial` are
  removed. The safety ramp's strength is no longer a free rate.
- Spatial runs absorb at the transverse edge by default (if Q1 is decided so). Results of
  runs whose light reaches the edge change, which is the purpose.
- The crossing limit can shorten the steps of a run with a narrow collar.

Kernel `SCHEMA_VERSION` and output `SCHEMA_VERSION` are bumped in phase 1.

Validated results to redo:

- the suite's `spatial_absorber` case and its baselines, whose boundary changes from a wall
  to an absorber (§3);
- the radial, Cartesian, line and planar performance-tracking cases;
- the free-space galleries, all of them: those that set a collar or a ramp
  (`plot_03_line`, `plot_04_planar_hybrid`, `plot_05_low_storage`,
  `plot_06_four_wave_mixing_scan`, `plot_07_far_field_statistics`), and those whose radial
  sampling or aperture changes with the placement (`plot_01_radial`, `plot_02_cartesian`,
  `plot_08_radial_capillary_coupling`);
- the introduction's filament and crossing-beam tutorials;
- the geometry figure.

Downstream Cartesian models that run with a $k_\perp$ ramp and no real-space collar take a
specification with `real_space=False`, or the default, and re-qualify their numbers.

The device backends (MLX and the native GPU builds) run today's in-stage collar now, with no
refusal. Under the new design they either carry the new parts (the radial and planar
safety ramps, the transverse monitor edges) or refuse those parts by name.

### 10. Alternatives considered

- **Keep the collar, retune its constants.** SEP 21's sweep shows that width and strength
  alone reach a few $10^{-3}$ at best, and that the right strength depends on the beam, not
  on $L$. Retuning would leave the length coupling.
- **Make the split the default.** It is cheaper per step where the temporal split already
  visits physical space. But it needs the crossing limit all the same, hides its error from
  the controller, and is incompatible with the Adams and low-storage histories (§4.7). It is
  kept as an option.
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

**Phase 0 — calibration and measurement.** `numpy_ref` and repository scripts; no public
interface change.

- References 7.2.1–7.2.3, with the probe of §2 as a script with converged references.
- The calibration of §7.3, with each source check of [2–4] recorded before a constant or a
  profile is transcribed.
- Scheme accuracy (§7.4), including the crossing counterexample.
- The protected-signal and collar-occupying witnesses.
- A first returned-share measure, and its floor.
- Passivity values, and the reduced surrogate of §4.9 checked against the full
  computation on small grids.
- The in-stage cost under the default Adams `Multistep` on the radial and planar witnesses,
  beside Runge–Kutta (§4.7).
- The return strip's width and estimator, their resolution and floor, judged above the
  input-noise floor (§5.4).
- The step cost of `cover="intermediate"` (§4.7).
- Costs at matched accuracy, including the dense radial work at the default
  `collar_fraction` against a run of the same spacing without a collar, and the split
  against in-stage, which decides whether the split reaches the compiled backends (§5.3).
- A report that recommends answers to Q1–Q7, for the maintainer's decision before phase 1.

**Phase 1 — `numpy_ref`, interface and documentation.**

- `collar_fraction` and the outside placement.
- `SpatialAbsorber` and its resolution, and the removals of §5.2.
- The crossing limit.
- The in-stage default and the split option with its refusals.
- The radial and planar safety ramps.
- The monitor's transverse edges and the ledger.
- Both schema bumps.
- The tests of §7 on `numpy_ref`, the qualification matrix's `numpy_ref` column, the manual
  pages and the tutorial.
- The other backends refuse what they lack, by name.

**Phase 2 — `numba_cpu`.** The same, with parity on matched schedules.

**Phase 3 — native CPU, requalification, Final.**

- The compiled paths, with parity.
- The suite's spatial cases, the galleries, the benchmark ledger and the Luna page.
- Device backends are left to the standing policy, as a roadmap item.

### 12. Open questions

The maintainer decided Q1–Q7 on 2026-10-31 (John Travers), taking the drafting agent's
recommendations. Where a decision depends on a value phase 0 measures, phase 0 confirms
or reports against it, and a departure returns the question to the maintainer.

1. **Q1 — the default.** *Decided:* `prop_radial`, `prop_free` and `prop_planar` resolve
   `SpatialAbsorber()` when none is given, provided the phase 0 witnesses meet §7.1's
   budget; `spatial_absorber=False` gives a bare wall or wrap.
2. **Q2 — the default collar fraction.** *Decided:* `collar_fraction = 1.0` on the radial
   grid and 0.5 on the Cartesian, line and planar grids, to be confirmed in phase 0. The
   §2 probe's best radial collar was as wide as the aperture; the radial cost, $(1+f)^2$ in
   dense work, is small in absolute terms, while the Cartesian cost, $(1+f)^2$ in memory,
   is not. A run short of memory narrows the collar and declares its protected band.
   *Pre-acceptance note:* the dense Hankel products dominate a radial stage, so $f=1$ is
   four times that work at fixed spacing, plus the collar's own dense product. Phase 0
   reports the measured factor before Q2 is confirmed.
3. **Q3 — the profile.** *Decided:* the Planck taper's complement is the default. The
   polynomials are measured against it in phase 0. The singular profile is dropped unless
   phase 0 shows a large gain, because it would bring the split scheme and its history
   restrictions with it.
4. **Q4 — the default scheme.** *Decided:* in-stage by default; `"split_step"` optional for
   `RungeKutta` only, refused under `Multistep` and `LowStorageRungeKutta` (§4.7). Whether
   SEP 21's split temporal collar needs the analogous crossing limit goes to the roadmap.
   *Pre-acceptance note:* the radial and planar routes default to `Multistep`, under which
   the in-stage stability limit is about 38 times shorter than under Runge–Kutta (§4.7).
   Q4 returns to the maintainer if phase 0 finds those runs bound by it. The split is built
   on `numpy_ref` first, and reaches the compiled backends only on phase 0's evidence
   (§5.3).
5. **Q5 — the default protected band.** *Decided:* the launch times $c_b$, as a fallback
   that the resolved design records as protecting the launched spread only, with the
   monitor's returned share as the safety net. The guide tells users who expect generated
   radiation (four-wave mixing, conical emission) to declare an `angle`. $c_b$ is set in
   phase 0.
6. **Q6 — the safety band and passivity tolerance.** *Decided:* the safety band is the
   outer 10 % of each axis's wavenumbers, on by default on every geometry, radial and
   planar included, with the strength rule of §4.4. $\varepsilon_P$ is phase 0's worst
   measured admission value with a margin of ten.
7. **Q7 — the monitor's policy.** *Decided:* a returned share above budget warns; the run
   stops only on gross return, above $10^{-2}$ of the launch's field norm, until phase 0
   qualifies the measure. The policy is then revisited, to mirror the temporal monitor's if
   the measure's floor allows.

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

Only bibliographic identity has been checked for these references (2026-10-31). The
equation-level content of [2–5], in particular any profile, coefficient or optimum that is
transcribed, is checked against the full text in phase 0 and recorded in its report.

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
