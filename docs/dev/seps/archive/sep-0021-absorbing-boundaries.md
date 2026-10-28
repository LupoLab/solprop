# SEP 21 — Temporal boundaries: wider windows, absorption and the escaped field

| | |
|---|---|
| **SEP** | 21 |
| **Title** | Temporal boundaries: wider windows, absorption and the escaped field |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-16); GPT-6 (review revisions, 2026-10-16; window-strategy revision, 2026-10-20); Opus 5.5 (refresh against the integrated tree, 2026-10-24) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI); Fable 5.1 (AI); Opus 5.5 (AI) |
| **Type** | Change |
| **Created** | 2026-10-16 |
| **Resolution** | Accepted 2026-10-24 by John Travers after his review of the refreshed proposal and the Opus 5.5 AI review and response recorded in Part III (LupoLab/solprop pull request 14, closed into the implementation pull request). Final 2026-10-28 by John Travers after phases 0a and 2, his decisions D1–D17 and the deferral of phases 0b, 1, 3 and 4 (D4, D17; Part II, "Phase 2 — report") ([LupoLab/solprop#40](https://github.com/LupoLab/solprop/pull/40)). |
| **Requires** | [SEP 14](sep-0014-stepper-efficiency.md) (the collar study and the Adams limits); [SEP 16](sep-0016-unified-multistep.md) (the default multistep stepper and the collar ablation) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | John Travers with Opus 5.5 (AI), branch `feature/sep-0021-temporal-boundaries` ([LupoLab/solprop#40](https://github.com/LupoLab/solprop/pull/40)): phase 0a 2026-10-25, phase 2 2026-10-27; phases 0b, 1, 3 and 4 not built |
| **Manual pages** | [Boundaries](../../../guide/numerics/boundaries.md); [noise](../../../guide/media/noise.md); [fixed schedules](../../../guide/numerics/fixed-schedules.md); the tutorial *Is my time window big enough?* (`docs/tutorials/basics/plot_07_time_window.py`) |

## Part I — Proposal

### 1. Abstract

The temporal absorbing collar stops light that walks out of the time window from
wrapping around the periodic grid. Done as solprop does it, as a damping rate inside
every source evaluation, it can also be the stiffest term the default multistep stepper
integrates. Its peak-rate stability cap puts a conservative floor under the step count
for the default strength, independent of fibre length or tolerance: about 420 steps at
order three (PEC3/4) and 1440 at order five (PEC5/6). On short runs and on runs whose pulse stays clear of
the collar, that floor sets the cost. One alternative is the existing per-step split,
whose collar map has no explicit stability cap, but whose first-order error reaches the
physical window
wherever the collar holds field.

This SEP compares retaining the field on a larger time window with absorbing it:

- **whether to absorb at all:** a wider window without temporal absorption, with
  monitored guard regions, as a production candidate as well as a reference;
- **how to use the window:** fixed launch placement and prescribed frames, including
  an asymmetric allocation of time around the launch and a fastest-light frame;
- **its geometry and strength:** the collar's width, its rate profile and how its
  strength is set;
- **how it is integrated:** in-stage, split, or an implicit–explicit (IMEX) treatment;
- **what it reports:** diagnostics, and an optional record of the radiation it removes.

Every option is judged against a separately converged wide-window reference. The
campaign measures wall time and memory at matched physical accuracy, including the
cost and reliability of boundary checks. It explicitly considers a default without
temporal absorption that warns on boundary approach and stops before significant
wrapping. No default is selected in advance. The implicit-collar study proceeds only
for a case where a collar remains useful after this comparison. Only qualified
strategies are implemented; an escaped-field reconstruction is not a prerequisite
for a useful boundary alarm.

### 2. Motivation and scope

**The cost.** The in-stage collar's peak rate bounds the negative-real spectrum of its
projected operator (ADR-025). The bound need not be an attained eigenvalue. For the
default `Multistep` it sets a floor of $A N/(2r)$ steps over the run (§4.3): $A = 12.6$
is the profile depth, $N$ = `boundary_N` = 20, and $r$ is the method's real stability
interval. This is the constant-step scalar estimate used to interpret the cap; it is not
a stability theorem for variable steps, non-commuting dispersion and nonlinear sources.
SEP 14 found the floor holding the Adams march on a CtC slice at its order-three
stability limit "whatever the tolerance". SEP 16 found it setting the median step of the
full canonical run in the exponential formulation (1329 calls against 965 with the
per-step collar, which removes the explicit cap but still absorbs).
It also held a 2 cm gradient's Adams steps near 30 µm (503 attempts to 1 cm, against 27
with the per-step collar). And because each order's limit is $2r/\alpha_{\max}$, the
floor pushes the variable-order choice towards orders two and three.

**The two existing schemes each fail somewhere.** The per-step split is free of the
floor, but it is a Lie splitting: it is first order wherever the collar holds field, and
that error reaches the physical window. SEP 14 measured 2.3e-4 within ±300 fs on the RDW
capillary, against 3e-7 with the in-stage collar, and 6.9e-3 to 2.6e-3 on CtC between
`rtol` 1e-4 and 1e-6. The symmetric (Strang) split tested with unchanged Adams history did not fix this
for a nonlinear source (SEP 14 phase 4; qualified in §4.5). So today's guidance has a manual decision in it:
"take the per-step collar where the pulse stays clear of the collar"
([Adams](../../../guide/numerics/adams.md#stability-and-the-temporal-collar)). The
project's standing preference is a robust default over conditional speed-ups.

**The absorber is silent.** Nothing reports how much energy the temporal collar removed,
or how much of the field sat inside it. `warn_walkoff` is an a-priori estimate from a
span the caller supplies. The spatial routes already measure their collar's
electric-field-norm share at field-bearing saved planes (`collar_field_fraction`,
renamed and redefined by [SEP 34](sep-0034-medium-flux.md), which keeps it distinct
from physical flux); SEP 22 deliberately
skips that diagnostic at statistics-only planes. Neither that diagnostic nor sparse
output is a boundary-safety monitor. The temporal collar has no equivalent. A run
whose physics happens inside the absorber is answered differently
by the absorber, as the boundaries page warns. The Raman self-frequency-shift gallery
example plots its soliton out to 1.7 ps in a ±1.75 ps window whose temporal collar
begins at 1.64 ps.

**What the measurements so far did not compare against.** Every collar study to date
(ADR-025, SEP 14, SEP 16 Q4) measured a scheme against a converged run *with the
in-stage collar*. That establishes consistency, not correctness. The collar is meant to
imitate an unbounded window. Its own perturbation of the in-window field, as against a
run on a wider window, has never been measured, so it is not known whether the split's
2e-4 is large or small next to the in-stage collar's own error.

**A wider window may be the cheaper calculation.** At fixed temporal resolution,
extra samples cost roughly linearly for local responses and as $n_t\log n_t$ for
transforms. Removing the temporal collar can save both its transform pass and the
steps forced by its stability cap. It also removes the collar's modelling error,
provided the retained field stays clear of the periodic seam. The saving is not
universal: SEP 16's canonical call reduction is only a factor of 1.38, and weakening
the RDW collar barely reduced calls. Neither study measured an enlarged window
without temporal absorption. Sections 4.10–4.12 make that comparison mandatory,
including efficient placement, frame choice and the cost of detecting inadequate
windows. A monitored failure is preferable to silently returning wrapped light;
whether that policy makes a robust default is a campaign result, not an assumption.

**In scope:**

- production windows with no temporal absorber, retaining the spectral collar and
  hard band mask, compared at matched observable errors and measured resource limits;
- fixed launch offsets and prescribed propagation frames in that comparison; the
  supported settings, coordinate metadata and guidance needed for qualified choices;
- guard-region diagnostics independent of the absorber, their sampling cadence,
  warning/error policy, and evaluation of a monitored default without temporal absorption;
- the temporal collar on field and envelope grids, on every route that carries one;
- its geometry: the collar width (`collar_fraction`), the rate profile, and how strength
  is parameterised (`boundary_N`, `boundary_length`, or a protected walk-off);
- its integration: the in-stage term, the per-step split, and a new IMEX candidate for
  the multistep stepper in both formulations;
- diagnostics: absorbed energy, collar occupancy and guard occupancy, with in-run
  boundary alarms and an end-of-run report;
- an optional escaped-field record: the spectrum and energy of what left the window,
  estimated from crossings through a surface in the flat region before the collar,
  subject to the qualification of §4.7;
- the keyword switches the roadmap asks for (`absorber=False`, `absorber_scheme=`);
- the wide-window reference and the campaign that uses it.

**Out of scope:**

- **The spectral collar.** It is diagonal in ω and folded into the linear operator, so
  the propagator applies it exactly at no cost (ADR-007). It has no stability or
  splitting problem. Its strongly damped bins already take the anchor's value in dense
  output (ADR-039).
- **The transverse spatial collars.** They use the same bounded profile and the same
  in-stage form ([free-space model](../../../guide/free-space/model.md#spatial-collar-and-statistics)),
  so §4 motivates testing whether the geometry results transfer. Phase 0 measures this
  (R4); changing the real-space collars' geometry or strength is left to a later
  proposal. [SEP 22](sep-0022-memory-lean-spatial-execution.md) delivered the
  k⊥ collar and decoupled the in-stage collars' strength from the spectral rate's
  ceiling (its `collar_length`, Q16). Its temporal and real-space spatial strengths
  still share that length; this SEP must not alter spatial absorption inadvertently.
- **Adaptive window machinery:** recentering from the evolving solution, expanding
  the grid during propagation, remeshing, and automatic restart on a larger grid.
  Fixed launch placement and prescribed frames are in scope (§4.11); a general
  offset-grid API and independently asymmetric absorber ramps remain follow-up work.
- Perfectly matched layers, transparent boundary conditions and phase-space filters
  (§10). The phase-space filter prototype and the directional occupancy diagnostic
  were removed from this SEP on 2026-10-24 and kept as roadmap items (Q13, Q14).
- Implicit or IMEX treatment for the Runge–Kutta stepper. Its floor is about 38 steps
  (§4.3; about 45 for the low-storage RK4 march of
  [SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md), which must refuse
  an implicit scheme as the Runge–Kutta march does) and rarely binds.
- Reimplementing the spectral-only collar's keyword (`temporal=False`) or its
  compiled-away zero-rate pass: [SEP 22](sep-0022-memory-lean-spatial-execution.md)
  phase 1 supplied them (Q16). This SEP uses them in the comparison and retains them.

### 3. Context

**What exists.** {py:mod}`solprop.boundaries` and the manual's
[absorbing boundaries](../../../guide/numerics/boundaries.md) page:

- the spectral rate $\alpha(\omega) = -2\log W(\omega)/\ell$, folded into $D$;
- the temporal rate $\alpha(t) = A(1-W(t))/\ell$ with $A$ = `TEMPORAL_ALPHA_LENGTH`
  = 12.6 and $\ell = L/N$ (ADR-007);
- two temporal schemes behind `absorber_scheme`: `"nonlinear"`, the in-stage term
  $-\alpha E/2$ masked to the active band (the default), and `"split_step"`, the factor
  $e^{-\alpha h/2}$ once per accepted step followed by the hard band mask (ADR-025);
- `absorber(grid, temporal=False)`, with no temporal pass after translation, while
  the spectral rate and hard band mask remain (SEP 22);
- `linear.Frame`, including a fixed `beta1` or the reference mode's group delay
  evaluated along z, and pulse delays for fixed placement; these do not yet provide
  an automatic fastest-light frame or a qualified boundary alarm;
- `stable_step`, `warn_walkoff`, and the translation refusal of a replayed schedule or
  step floor past the limit;
- for the multistep stepper, the stability-capped order choice of the exponential
  formulation (SEP 16 phase 6). The per-step collar refuses interpolated planes;
- on the keyword interfaces, `prop_capillary` takes `absorber=` and
  `absorber_scheme=`, `prop_waveguide` takes `absorber=` only, and the other `prop_*`
  functions take neither. No interface accepts `absorber=False`; `None` means the
  default collar.

The grids build the temporal collar in `solprop.grid._temporal_collar`. Its flat region
is $\pm\min(T_{\mathrm{req}}/2,\,(1-2f)T/2)$, with $T$ the realised window and $f$ =
`collar_fraction` = 0.05, which must lie strictly between 0 and 0.5. Each edge is
therefore at least 5 % of the window, and is otherwise whatever slack the transform
size leaves.

**Prior studies this builds on:**

- ADR-007 and ADR-025: rates instead of masks, and the in-stage default with its order
  ladder.
- SEP 14 §4.5 and phase 4: the stability intervals, the Strang study and its failed exit
  criterion, and the CtC-slice diagnosis. SEP 14 also deferred "lower the collar's peak
  rate by default" as "a physical setting, measured separately". That is part of what
  this SEP measures.
- SEP 15 Q5: an implicit–explicit collar was deferred because the exponential
  formulation's endpoint weight does not commute with the collar.
- SEP 16 Q4: the ablation concluded that an implicit or inner collar would buy nothing
  beyond the per-step collar *where that is valid*, and so built none.

**Roadmap items.** This SEP covers two in part:

- *An absorber switch on the keyword interfaces.* The `absorber=False` and
  `absorber_scheme=` keywords are in scope. Whether spatial runs should default to an
  absorber is not.
- *The split-step collar on the general modal route, native against the reference.* The
  split scheme's out-of-band disagreement is unresolved, so phase 0a does not compare
  that scheme on a general modal guide. The item is linked and kept.

#### 3.1 Implemented contracts and sequencing

The campaign starts from the delivered contracts below, not the earlier drafting
costs or proposed interfaces. This cross-check was first made on 2026-10-20 and
refreshed on 2026-10-24 against the integrated tree (kernel schema 35, output
schema 24): SEPs 22–31 and 34 are Final. A campaign records the exact solprop
revision, including its in-repository native backend, that it actually uses.

| SEP | Contract this proposal must preserve | Consequence for SEP 21 |
|---|---|---|
| [22](sep-0022-memory-lean-spatial-execution.md) | All-zero temporal rates compile no temporal pass under either scheme; `collar_length` separates in-stage strength from the spectral ceiling; statistics-only observations avoid field transfers and may skip spatial collar diagnostics. | Use the real zero-pass baseline. Keep spatial strengths fixed when disabling temporal absorption. Monitor independently of output, with backend reductions rather than mandatory whole-field downloads (§4.12, Q16). |
| [23](sep-0023-far-field-aperture-spectra.md) | `absorbed_power` has four J/m channels: spectral, transverse, evanescent, material. The transverse channel is the k⊥ rate; temporal and real-space spatial collars are excluded. SEP 34 now weights every channel by the medium's axial admittance. | Preserve the four-channel meaning, order and flux weighting. Add temporal diagnostics separately; compare rate definitions and qualified integrals, not an unresolved endpoint trapezoid (§4.6, §7). |
| 24 ([implemented stepper](../../../guide/numerics/choosing-a-stepper.md)) | `LowStorageRungeKutta` replays a fixed schedule, has no dense output or embedded error estimate, and is calibrated by observable convergence. Its in-place register ownership and held checkpoint state are deliberate. | Include it in the campaign with independently calibrated schedules. Checks use sufficiently close committed endpoints; extra check steps require a newly calibrated schedule. Do not restore full-volume storage merely to monitor the boundary. |
| [27](sep-0027-held-plasma-current.md) | Field-grid rate plasma uses the held-current source, including two gated products and a sampled-gate certificate. The source gate remains when temporal field absorption is off. | Plasma candidates and references must all use this source. Converge the gate products and response preparation separately from the field absorber (§4.8); old gated-polarisation results do not settle the boundary choice. |
| [28](sep-0028-free-space-evaluation.md) | No seam change. Free-space chunks zero out-of-crop bins per chunk and alias the node field and polarisation onto the chunk's rows; CPU crossings are tiled; free-space, line and planar grids take a 1 GiB default node budget. | Guard reductions and accumulators must not assume a full-volume node field or reuse an aliased stage row without proving it holds the committed endpoint (§4.12). Measure monitor overhead on this execution path. |
| [29](sep-0029-compiled-backend-distribution.md) | The native backend lives in this repository as `solprop-native`, checked against the kernel schema and its contract tuples at import; native builds need GCC 14 or newer. | Native work lands in the same pull request as the Python backends, following the native contributor rules, and refuses unimplemented quantities by name. |
| [30](sep-0030-standard-benchmark-suite.md) | `benchmarks/` holds only the standard suite; campaign harnesses and raw results are kept with the project's development records. | Phase 0's harness follows that rule; cases that remain useful after the campaign, such as the wide-window references, become suite cases in phase 2 (§11). |
| [31](sep-0031-simulation-coupling.md) | Remapping reports the share of a record's energy outside the temporal collar's flat region (`one_low`/`one_high`), and pulse measures average where the collar is exactly one; persisted frame records rebuild absolute phase and delay; a relaunched run is a new run with an inherited origin. SEP 31 leaves wider windows and escaped-field models to this SEP. | Geometry changes keep the meaning of the collar's flat-region fields or update those consumers; a window without temporal absorption still needs a defined edge band, for which the guard band is the candidate. Reuse the frame records to align fields in §4.8 and §4.11. Accumulators start at zero in a relaunched run unless Q22 decides otherwise. |
| [34](sep-0034-medium-flux.md) | Spatial routes have a physical admittance metric, `SpatialFlux`, with endpoint weights and a radial endpoint term in the linear exponent. A collar need not commute with the admittance, so its quadratic-form change is measured, never inferred from a field-norm decrement. | The spatial metric $M$ of §4.6 is SEP 34's. Its radial endpoint term is joined to the cumulative linear exponent, so it belongs to $\mathcal L$, where it cancels $E^\dagger M'E$ for the retained forward field; the ledger must not count it twice. SEP 34's passivity rule answers part of Q15 on those routes. Its uniform-loss test (`tests/backends/test_flux_gradients.py`), which keeps the collar's field fraction negligible, must keep passing. |

**Execution order.** SEP 27 and SEP 28 are integrated, so the ordering this section
recorded on 2026-10-20 is satisfied: the plasma campaign uses the qualified held-current
source and the free-space costs are measured after SEP 28's changes, with no
remeasurement caveat for earlier baselines. Use the next free kernel and output schemas at landing, not numbers reserved
by an older draft: SEP 31's frame follow-up and [SEP 33](sep-0033-single-precision-mlx.md)
may take numbers first; output schema 24 is SEP 31's. If SEP 33 lands before phase 2, every new accumulator and
reduction states its accumulation precision and is covered in binary32. This is an
integration constraint, not a requirement to finish the escaped-field reconstruction
before choosing a useful window or alarm.

### 4. Theory

Conventions are those of [conventions.md](../../conventions.md). The equations below use
the internal FFT convention and internal amplitude $E$ (called $a$ in conventions), not
the saved true-power amplitude $b$. $\alpha$ is a nominal power coefficient: the field
falls as $e^{-\alpha z/2}$. $t$ is the retarded time in the frame of the grid. $P$ is
the projection onto the active band.

#### 4.1 The equation and the collar term

In the interaction picture the state $\tilde u(\omega,z) = e^{-\Phi(\omega,z)}E(\omega,z)$
obeys

$$
\partial_z \tilde u = e^{-\Phi}\,N\!\left(e^{\Phi}\tilde u\right)
  + e^{-\Phi}\,C\,e^{\Phi}\tilde u,
\qquad C = -\tfrac12\,P\,\mathcal F\,\mathrm{diag}\,\alpha(t)\,\mathcal F^{-1} P,
$$

where $\Phi$ is the cumulative diagonal linear operator, including the spectral collar.
$C$ is diagonal in neither ω nor (because of $P$) t. It is self-adjoint and negative
semidefinite in the unweighted Parseval metric on the represented band, with spectrum
in $[-\alpha_{\max}/2, 0]$; this does not assert passivity in every physical energy
metric (§4.6). On an oversampled grid the implementation is the corresponding
compression $J^* C_0 J$, where $C_0=-\mathrm{diag}(\alpha)/2$ and $J$ synthesises the
represented band on the fine grid with the Parseval scaling. Field grids require the
real-field transform and its weighted adjoint, with self-adjointness and CG understood
in the equivalent real vector space. Half-spectrum multiplication by a real temporal
profile need not be complex-linear. These pad/crop and endpoint weights
must be retained in prototypes. Coupled linear routes use their full evolution
operator in place of the scalar $e^\Phi$ (§4.7). The in-stage scheme evaluates the second term
explicitly with the source. The split scheme instead applies $e^{-\alpha h/2}$ in t, then
$P$, once per accepted step.

#### 4.2 What the collar must do: transmission and reflection

In the time domain the propagation equation is a Schrödinger equation in t, with z as
the evolution variable and $\beta(\omega)$ as the dispersion relation. The collar is
then a complex absorbing potential, of the kind that admits transmission and suits
periodic grids [1, §3, eqs 171–173]. A collar can fail in two ways.

- **Transmission.** A packet crosses the collar before it has been absorbed, and on a
  periodic grid re-enters from the other edge [1, p. 45].
- **Reflection.** A packet is scattered onto the branch that walks back into the window.

**Transmission.** For a weak, smooth potential the semiclassical (JWKB) transmission is
$|T|^2 = \exp(-2\,\mathrm{Im}\int k\,\mathrm{d}x)$, with no reflection [1, eqs 206–207].
In the weak-potential, narrow-packet, steady-drift approximation this gives
$|T|^2 \simeq e^{-2\Gamma}$, with $\Gamma$ derived below. It is not an exact
transmission law for a dispersing or nonlinear packet.

**Reflection.** For quadratic dispersion the wavenumber in this analogy is the packet's offset
$\Omega$ from the frame's zero-walk-off frequency, not the optical carrier. A general
dispersion relation can have several such frequencies, or no reflected branch in the
represented band. For a stationary medium and collar, elastic scattering conserves the real exterior
propagation constant in the frame,
$K(\omega) = \beta(\omega) - \beta_1^{\mathrm{f}}\omega$. It therefore reflects a packet
at $\omega$ onto the frequency $\omega'$ with $K(\omega') = K(\omega)$ and the opposite
walk-off. With $\beta_2$ alone, that partner is at $-\Omega$, a gap of $2\Omega$. A smooth
collar of width $w$ has little Fourier content at that gap when $\Omega w$ is large. The
literature's rule of thumb is $kL \gtrsim 10$ for smooth monomial potentials
[1, §3.2, p. 39], with the JWKB validity condition [1, eqs 208–209] as the local form.
It is more restrictive at low energy [1, p. 42]. For quadratic dispersion the local JWKB
parameter is $|k'|/k^2 \approx \alpha'/(2\beta_2\Omega^3)$ with $\alpha' \sim \alpha_{\max}/w$:
at fixed $\Omega w$ it is proportional to the collar's strength, so reflection is a
function of $(\Omega w, AN)$, not of $\Omega w$ alone.

So **fast walkers set the transmission requirement and slow walkers set the reflection
limit**. For quadratic dispersion, $\Omega w \lesssim 10$ identifies packets for a
reflection sweep, not a universal threshold or a predicted reflected fraction. With full
dispersion the gaps to all same-$K$ branches, the collar strength and its smoothness
also matter. On the RDW grid ($w$ = 20.5 fs), $\Omega w = 10$ means $\Omega \approx 0.5$
rad/fs. Moderate walkers may sit in the reflecting range, which phase 0 measures. The
carrier frequency plays no part: reflection is not small just because the collar spans
many optical cycles.

For a narrow packet with constant group-delay drift $v = |\partial_z\tau|$ (s/m),
neglecting dispersion, reflection and nonlinear evolution during the crossing, the
field attenuation accumulated crossing one edge is

$$
\Gamma = \frac{1}{2v}\int_{\mathrm{edge}}\alpha\,\mathrm{d}t
       = \frac{A}{2\ell v}\int_0^{w}\bigl(1-W\bigr)\,\mathrm{d}t
       = \frac{A\,N\,w}{4\,\Delta\tau},
$$

with $w$ the edge width and $\Delta\tau = vL$ the drift over the run. The last step uses
the Planck taper's symmetry, $W(x) + W(w-x) = 1$, so $\int_0^w (1-W) = w/2$ (this is
the "6.3 per collar width per reference length" of the boundaries page). $\Gamma$ is the
attenuation of **one** edge. `warn_walkoff` (`solprop.boundaries`) integrates the rate
over the whole time axis, both edges, and requires $2\Gamma \ge \Gamma_{\min} = \ln 10^3$;
a wrapping packet does cross both edges before it re-enters the flat region, so the
one-edge criterion $\Gamma \ge \Gamma_{\min}$ used below is conservative by a factor of two
in the exponent. Which count the design adopts is open (Q17); the tables of §4.3, the
strength formula of §6 and Q3's figures all count one edge. The **protected walk-off**,
the largest uniform drift meeting this model's one-edge attenuation target, is then

$$
\Delta\tau_p = \frac{A N w}{4\Gamma_{\min}} \approx 9.1\,w
\qquad (A N = 252),
$$

or $18.2\,w$ counting both edges as `warn_walkoff` does.

#### 4.3 The stiffness floor

Using the peak rate as a conservative scalar cap for an explicit method with
negative-real stability interval $r$ gives $h\,\alpha_{\max}/2 \le r$, and
$\alpha_{\max} = A/\ell = AN/L$. The number of steps over the run is therefore at least

$$
n_{\min} = \frac{A N}{2r} = \frac{2\,\Gamma_{\min}\,\Delta\tau_p}{r\,w}.
$$

The first form says the floor depends only on the absorber's dimensionless strength and
the method. The second says what it costs: protecting a walk-off of $\Delta\tau_p$ through
an edge of width $w$. With today's $A N = 252$ and the package's intervals
(`boundaries.STABILITY_RADIUS` for IP-DOPRI5, `LOW_STORAGE_REAL_INTERVAL` for the
low-storage march, and `PEC_REAL_INTERVAL` in `solprop.stepper.multistep`, which its
test derives from the recurrence's spectral limit), rounded:

| method | $r$ | $n_{\min}$ |
|---|---:|---:|
| IP-DOPRI5 | 3.3 | 38 |
| RK4 (SEP 24's implemented low-storage march) | 2.785 | 45 |
| PEC2/3 | 0.545 | 231 |
| PEC3/4 | 0.30 | 420 |
| PEC4/5 | 0.163 | 773 |
| PEC5/6 | 0.0877 | 1437 |

The same numbers come out of the default grids for the benchmark cases, which differ in
everything but $AN$:

| case | window | collar edge | $\alpha_{\max}$ (1/m) | protected walk-off (one edge) |
|---|---:|---:|---:|---:|
| RDW, He, 1.5 m | 410 fs | 20.5 fs (5.0 %) | 168 | 185 fs (9.0 $w$) |
| CtC reduced, H2, 0.1 m | 1.14 ps | 69 fs (6.1 %) | 2520 | 0.63 ps (9.0 $w$) |
| CtC, H2, 0.2 m | 6.19 ps | 1.10 ps (17.7 %) | 1260 | 10.0 ps (9.1 $w$) |
| SSFS example, H2, 2 m | 3.64 ps | 182 fs (5.0 %) | 126 | 1.65 ps (9.1 $w$) |

Four consequences follow.

1. **Strength is tied to the run length, not to the walk-off.** $\alpha_{\max} = AN/L$.
   Rebuilding the absorber from a short slice's length therefore gives a proportionally
   stronger collar for
   the same physics (SEP 14's 2 cm CtC slice: 12 600 against 2520 m⁻¹).
2. **Grid slack strengthens the collar without being asked.** The edge width is
   whatever the transform rounding leaves, from 5 % to tens of per cent. Absorption
   grows with $w$ while $\alpha_{\max}$, and so the floor, stay fixed. On the full CtC
   grid the default protects a 10 ps walk-off on a 6.2 ps window.
3. **The width is the lever.** At fixed protected walk-off, $n_{\min} \propto 1/w$:
   doubling the collar halves the floor. It costs either physical window or samples, and
   on a power-of-two grid the slack is sometimes already there.
4. **The profile's shape gains at most a factor of two.** The complement profile's peak
   is twice its edge mean. A flat profile, ratio one, is the lower bound, but it is
   discontinuous. A plateau that rises over half the edge has ratio about 1.33. A
   steeper rise also steepens $\alpha'(t)$, which raises the split scheme's error (§4.5)
   and can increase reflection (§4.2); the actual scattering must be measured. Some
   optimised potentials, such as Manolopoulos's double pole [1, eqs 216–220], have
   unbounded peaks, which is the wrong direction for this explicit rate cap. This
   does not rule out bounded optimised profiles for other error objectives.

#### 4.4 Explicit damping, PMLs and transparent boundaries

Whatever its form, an absorbing term in the equation that removes a packet crossing width
$w$ at speed $v$ must damp it at a z-rate averaging $\Gamma v/w$ while the packet is
inside. A bounded negative-semidefinite damping operator treated explicitly therefore
has a rate-dependent stability restriction. This is not a no-go theorem for every
possible nonlocal boundary construction or exponential treatment.

**A perfectly matched layer** [3, 4] damps per unit of *phase* travelled. Its modelling
error is $\exp(-2k\sin\gamma\,\sigma_{\max}d/(p+1))$ [4, eq. 4.7]: it depends on the
integral of the stretching, and decreases with increasing positive wavenumber. The
attenuation exponent grows with wavenumber. In both papers the layer is discretised by
finite differences and stepped by Crank–Nicolson, so the scalar dissipative stability
restriction is removed; accuracy and the full discrete operator still need checking. It
is implicit, which moves the stiffness rather than removing it. The optimised layer is
thinner than a third of the dominant wavelength [4, Table 2], so its z-rate, about
$\Gamma v/d$ with $d \ll w$, would be stiffer still if treated explicitly.

Three considerations argue against adopting those PMLs unchanged here:

- **It loses the exact linear propagator.** A stretched layer is equivalent to a
  modified kinetic operator plus a complex potential [4, eq. 2.9], so with full or
  tabulated $\beta(\omega)$ the operator is diagonal in neither domain.
- **A fixed stretch can amplify outgoing content.** The stretch damps according to the sign of
  the wavenumber, and both papers choose branches that align that sign with the outgoing
  direction [3, p. 4; 4, p. 6]. Here the walk-off direction is the sign of
  $\beta_1(\omega) - \beta_1^{\mathrm{f}}$, which need not have the sign of $\Omega$: higher-order dispersion can break the
  alignment. Where the signs disagree, that stretch would
  *amplify* outgoing content.
- **It still reflects when truncated.** A finite layer reflects at its outer wall
  [4, eq. 4.1; 2, p. 26], so it is reflectionless only when untruncated.

**Transparent boundary conditions** [2] are nonlocal in the evolution variable, z here
[2, eq. 2.1]. Sum-of-exponentials and rational approximations make them local [2, §§3.3–3.4],
so that is an approximation and a cost. Two further changes would be needed:

- the constant-coefficient constructions considered here assume a homogeneous exterior
  and need
  the roots of the dispersion relation continued off the real frequency axis [2, eqs
  2.5–2.7], which a table on the real axis does not supply;
- they need an explicit boundary/interface treatment in place of the present periodic
  FFT discretisation. This is a redesign, not a mathematical impossibility.

These papers do not supply a ready discrete boundary operator for solprop's
full-dispersion, band-limited propagation problem.

The stiffness of an absorbing *term* can therefore be moved (§4.5), reduced (§4.3), or
avoided by a different integration or boundary formulation. Diagnostics (§4.6, §4.7)
expose errors; they do not make a distorted solution accurate. An operation applied
*between* steps adds no eigenvalue. Examples are the per-step split and the phase-space
filter of Soffer and Stucchio [5], which removes only content that is outgoing (§4.9).
Such an operation trades stiffness for an error of a different kind.

#### 4.5 Integrating the collar

**The in-stage term** integrates $C$ at the method's own order, inside the method's
stability region (above).

**The per-step split** first advances the non-collar vector field
$G(E)=\mathcal L E+N(E)$, then applies the time multiplier and projection. Before
projection, with $C_0=-\alpha/2$, its leading local defect relative to the combined
flow is

$$
\frac{h^2}{2}\left(C_0G(E)-G'(E)C_0E\right)+O(h^3).
$$

Thus the nonlinear contribution involves $C_0N(E)-N'(E)C_0E$, not the operator
commutator $[C_0,N']E$. For quadratic dispersion,
$[\partial_t^2,\alpha]=\alpha''+2\alpha'\partial_t$; higher-order or tabulated
dispersion and band projection need not give a local error supported only on the ramp.
At fixed protection a geometrically rescaled collar has $\alpha'\propto w^{-2}$,
which motivates measuring, rather than assuming, a reduction of split error.

**The projection adds a separate defect.** On one common time grid, write $P$ in
the time representation for this paragraph. The implemented collar map is $S_h=P e^{hC_0}P$, restricted to $\operatorname{ran}P$. It is a
contraction in the Parseval norm, but

$$
S_h-e^{hPC_0P}=\frac{h^2}{2}PC_0(I-P)C_0P+O(h^3).
$$

The time multiplier is exact for its unprojected scalar subproblem; the complete map
is not the exact flow of the projected collar.
The in-stage and split implementations also use fine and coarse temporal rates,
respectively. Phase 0 separates that discretisation difference from splitting and
projection error before attributing an observed discrepancy to one mechanism.

**Strang** composition of exact subflows is second order. SEP 14's implementation
transported Adams history through the collar without recomputing nonlinear history:
a stored Kerr derivative then scales as $e^{-\alpha h/2}$ while the derivative of the
damped state scales as $e^{-3\alpha h/2}$. That implementation lost order. This is not
a general impossibility of second-order splitting with a multistep subsolver, but
repairing its history and projected subflow is outside this study.

**IMEX, interaction formulation.** Treat $C$ implicitly in the corrector and $N$
explicitly. A variable-step Adams–Moulton corrector for the collar part has the form
$\tilde u_{n+1} = r_n + h\beta_0\,e^{-\Phi_{n+1}}Ce^{\Phi_{n+1}}\tilde u_{n+1}$. With
$x = e^{\Phi_{n+1}}\tilde u_{n+1}$, the lab-frame spectrum at the new point, this becomes

$$
\bigl(I - h\beta_0\,C\bigr)\,x = P\,e^{\Phi_{n+1}} r_n .
$$

The conjugation by the linear flow drops out. What is left is symmetric positive
definite on the band, provided $\beta_0>0$, with condition number at most
$1+h\beta_0\alpha_{\max}/2$ in the Parseval metric.

- **Without $P$ it is a pointwise division in t**, one transform pair: the collar is
  diagonal on the modal time-domain field.
- **With $P$**, there are two choices: the pointwise division followed by $P$, which is generally not the inverse of the compressed operator and needs a measured
  residual and convergence test; or conjugate
  gradients, whose iterations each cost one transform pair and have the standard energy-norm bound
  $\|e_k\|_A/\|e_0\|_A \le 2[(\sqrt\kappa-1)/(\sqrt\kappa+1)]^k$.
  The stopping tolerance is tied to the step's error budget, not a fixed iteration count.

Either way the cost is transforms, not source calls. Within the AM intervals the
condition number is at most $1+\beta_0 r_{\rm AM}$: 3.5, 2.1, 1.6 and 1.4 for corrector
orders 3 to 6, so the CG bound below needs about 13, 9, 7 and 6 transform pairs per solve
to $10^{-6}$, in the predictor and in the corrector, against about one pair for a Kerr
source; the trapezoid at $x = 100$ needs about 50. Since `Multistep()` is the default for
every run, that is cheap only where the source is expensive (plasma, Raman, the TDSE
response), and phase 1 reports transforms beside source calls (§11).

The implicit part's stability depends on its order. The constant-step Adams–Moulton
methods' real intervals (independently recomputed from the characteristic polynomial;
see [9]) against the PEC intervals of today's march. The rows are keyed by the
corrector's order, which is one above solprop's `order`: PEC$q$/$(q+1)$, `order` $q$,
corrects with the Adams–Moulton formula of order $q+1$. So the abstract's "order
three" is PEC3/4, the corrector-order-4 row.

| corrector order | today's scheme (`order`) | AM, implicit collar | PEC, explicit collar |
|---|---|---:|---:|
| 2 (trapezoid) | — | unbounded (A-stable) | — |
| 3 | PEC2/3 (2) | 6.0 | 0.55 |
| 4 | PEC3/4 (3) | 3.0 | 0.30 |
| 5 | PEC4/5 (4) | 1.84 | 0.16 |
| 6 | PEC5/6 (5) | 1.18 | 0.087 |

For order $q$, integrate the Lagrange basis on nodes $1,0,\ldots,2-q$ over
$[0,1]$ to obtain $\beta_j$. The polynomial is
$\rho(\zeta)-h\lambda\sigma(\zeta)$, with
$\rho=\zeta^{q-1}-\zeta^{q-2}$ and
$\sigma=\sum_{j=0}^{q-1}\beta_j\zeta^{q-1-j}$. The negative-real endpoint
for $q>2$ is $-\rho(-1)/\sigma(-1)$. This table does not qualify variable-step
coefficients or the complete IMEX method; phase 1 checks the supported step ratios,
startup, order changes and non-commuting operators.

Five points qualify the table. With $x = h\alpha/2$:

- **An implicit collar at the corrector's own order moves the floor out 10–14 times, and
  no further.** Past its interval the AM collar is not merely inaccurate but strongly
  unstable: its largest root reaches 1.7–3.6 in modulus for large $x$. The order choice
  must therefore stay inside the implicit intervals, as it stays inside the explicit ones
  today (SEP 16 phase 6).
- **The trapezoid is A-stable but not L-stable.** Its collar factor per step is
  $(1-x/2)/(1+x/2)$, which tends to $-1$: $+0.33$, $-0.67$, $-0.96$ at $x$ = 1, 10, 100. The sign flips for $x>2$.
  For $x \gg 1$ it absorbs about $4/x$ nepers per step instead of $x$, so a *stronger*
  collar absorbs *less*. That is the transmission failure of §4.2, and the undamped,
  sign-flipping collar field also drives the explicit source. Ascher, Ruuth and Wetton
  found exactly this with Crank–Nicolson–Adams–Bashforth (CNAB): weak damping of stiff
  modes caused aliasing in spectral computations, and they advise against such schemes
  there [7, pp. 8, 22, 26]. The trapezoid therefore removes the *stability* floor but
  not the cost: where the collar holds field, its absorption is wrong unless $x \lesssim 1$.
  Its $O(h^3)$ local error in the collar part also holds only for $x \lesssim 1$. The factor tending to $-1$ already demonstrates the unresolved stiff transient;
  stiff order reduction of coupled IMEX schemes is an additional issue [8, pp. 25, 28].
- **Modified CNAB improves stiff decay but does not eliminate it.** In the
  two-step family of [7, eq. 15], the implicit weights are $9/16,3/8,1/16$.
  The stiff-limit polynomial is $9\zeta^2+6\zeta+1$, with a repeated root $-1/3$.
  Backward Euler has a zero stiff-limit factor but order one. These facts do not
  establish a universal optimal-decay bound for every possible Adams-like formula.
- **The predictor and estimator are part of the method.** An explicit predictor can
  feed a badly amplified collar state to the nonlinear source even when the implicit
  corrector is stable. Phase 1 derives and tests each complete predictor–corrector
  recurrence. Giving predictor and corrector the same implicit collar can cancel its
  defect in their difference, so an embedded difference alone may miss collar error.
  Each candidate needs an independent collar-error check, for example step doubling
  in the study. Occupancy diagnostics cannot replace integration-error control.
- **Semi-implicit BDF (SBDF) is the strongly damping alternative.** Its implicit part
  damps stiff modes to zero [7, pp. 8, 13], and it evaluates the source once per step,
  so its asymptotic source-call count can match today's, apart from startup, rejection
  and error-estimation costs. Its source-only imaginary-axis intervals must be
  recomputed from the chosen recurrence and compared with PEC3/4 and PEC4/5; the
  advection–diffusion CFL bounds in [7] are not interchangeable with those intervals. It
  was also the only IMEX family to integrate Kennedy and Carpenter's severe case at
  constant step [8, p. 23]. But adopting it means replacing the Adams march for the
  source as well, and BDF has a larger error constant [7, p. 9]. Phase 1 carries it as a
  reference, not a candidate default.
- **The history must hold the collar at the corrected state.** The intervals above
  assume the stored derivative's collar part is $C$ at the corrected state. Today's PEC
  march stores the *predicted*-state source (`backends/multistep.py`, the history entry
  marked `"predicted"`). An implicit collar at the new point beside predicted-state
  collar history recovers intervals indistinguishable from PEC's (a companion-matrix
  check at review: 1.24, 0.60, 0.31, 0.17 and 0.088 for `order` 1 to 5). Phase 1 states
  the storage rule. With an exact solve the corrected collar term is free,
  $Cx = (x - b)/(h\beta_0)$; with the projected pointwise solve it is not.
- **A hard step cap in both formulations.** Past $r_{\rm AM}$ the AM collar is unstable,
  and a collar shared by predictor and corrector can hide from their difference (above).
  Today `KernelProblem._check_collar_stability` returns for any scheme but
  `"nonlinear"`, and the order cap exists only in the exponential formulation and weighs
  the order, not the step. Phase 4 therefore refuses a replayed step or a `dz_min` past
  $2r_{\rm AM}/\alpha_{\max}$ under `"implicit"` in both formulations, and says what
  `"implicit"` does beside the always-explicit spatial collar
  (`_check_spatial_collar_stability`).

These are scalar results with the collar and the source commuting, so they are necessary
conditions only [7, p. 15]. The non-commuting coupling of the real problem has to be
measured.

**IMEX, exponential formulation.** The endpoint is
$u_{n+1} = e^{hD}u_n + h\sum_j w_j(hD)S_j$. An implicit collar there needs
$(I - h\,w_0(hD)\,C)\,u = b$, and $w_0(hD)$ does not commute with $C$ (SEP 15 Q5). It is
not pointwise. Two candidates:

- a Krylov solve (GMRES), whose iterations each cost a transform pair;
- giving the collar part a non-exponential weight in the lab frame, which reduces the
  solve to the interaction case's. The trapezoidal weight inherits the trapezoid's weak
  damping above; a backward-Euler or modified-CNAB weight damps, at order one or two.

**What no treatment changes.** The collar is an artificial term in the model. An
integrator must reproduce its effect on the requested observables to the specified
accuracy. This does not require resolving every discarded stiff transient, nor does
unconditional stability guarantee correct transmission. The accuracy and cost benefit of
an implicit method, particularly with nonlinear content in the collar, remains the phase
1 question.

#### 4.6 Diagnostics

**Occupancy and energy are different measurements.** On a conventional modal route,
use the nonnegative diagnostic

$$
\eta(z)=\frac{\sum_{m,p}\int (1-W(t))|b_{m,p}(t,z)|^2\,dt}
                   {\sum_{m,p}\int |b_{m,p}(t,z)|^2\,dt},
$$

where $b$ is the true-power analytic amplitude of conventions §4, with the field-grid
endpoint weights respected. This is an energy-density proxy, not the temporal Poynting
trace, which differs for a broadband dispersive index. Define zero occupancy for a
zero field. General modal and spatial routes must use their documented metric and
state the diagnostic's meaning; a sum of squared internal modal amplitudes is not a
universal physical energy. On spatial routes the metric is
[SEP 34](sep-0034-medium-flux.md)'s axial admittance, with its endpoint weights; the
spatial collar's `collar_field_fraction` is a field-norm share, not this proxy.

**The energy ledger uses the actual evolution operator.** Write the route's energy as
$U=E^\dagger M(z)E$, with $M$ including quadrature, DFT, real-field and power weights.
For $E'=\mathcal L E+N(E)+CE$,

$$
U'=E^\dagger M'E+2\operatorname{Re}(E^\dagger M\mathcal L E)
   +2\operatorname{Re}(E^\dagger MN(E))
   +2\operatorname{Re}(E^\dagger MCE).
$$

The temporal removal rate is $R_t=-2\operatorname{Re}(E^\dagger MCE)$ and
$U_t(z)=\int_0^zR_t(s)\,ds$; spectral removal uses the spectral-collar contribution
in exactly the same identity. The familiar $\int\alpha|E(t)|^2dt$ follows for an
unprojected scalar envelope with a constant power weight. For a frequency-dependent
metric, $M$ and $C$ need not commute: unweighted contractivity does not prove
$R_t\ge0$. SEP 34 §4.2 states the same rule for a spatial screen or collar against
the nonlocal admittance: measure the quadratic-form change, never call a field-norm
decrement physical absorption. On radial gradients SEP 34 joins its endpoint
amplitude term to the cumulative linear exponent: it is part of $\mathcal L$ and
cancels $E^\dagger M'E$ for the retained forward field, so it is counted once,
there. Do not clip a signed contribution to force a
positive ledger. Phase 0
checks passivity in the actual metrics and reports any failure before recommending a
default. Changing the collar operator to enforce another metric needs a recorded
maintainer decision (Q15), not an unnoticed normalisation change.

Under the split, record $U(E^-)-U(S_hE^-)$ using the same endpoint metric and the
actual map, including projection; distinguish the time multiplier's loss from the
hard mask's loss where they can be evaluated separately. This discrete difference
avoids a trapezoid approximation to an unresolved damping substep. A general energy
balance must also include material absorption, energy exchange with Raman/plasma or
other responses, changing metrics, and any further projections. The short ledger
$U_0=U+U_t+U_\omega+U_{\mathrm{loss}}$ is an acceptance identity only in a case where
all other contributions are proved zero or explicitly included.

SEP 23's existing `absorbed_power` is an instantaneous four-channel rate, not a
temporal escape counter. Preserve its spectral, transverse (k⊥), evanescent and
material channels, now weighted by SEP 34's admittance; material gain can make the
last one negative. On its supported
routes, a spectral integral uses the same actual summed `spectral_rate` and metric.
SEP 22's custom spectral rate is not separately attributable to the soft collar
(Q16). Temporal removal and any real-space spatial removal require separately named
quantities; neither may be relabelled as SEP 23's transverse channel.

**Sampling and accuracy.** Accumulate in-stage rates by trapezoidal quadrature at
actual accepted endpoints, with an independently measured second-order error bound;
the march's tolerance does not bound this quadrature error. The spectral channel is
diagonal and exact under the linear flow, and its rate reaches $60/\ell$, so $\alpha h/2$
can exceed one on its bins; there an endpoint trapezoid of $\alpha|E|^2$ has no
resolved-quadrature error bound at that step size. For an exact isolated linear
flow, use the per-step factor $1-e^{-\alpha h}$ (exponential weights). Coupled source
evolution needs a separately qualified exponential-weighted or refined rate integral;
that factor alone is not its exact total loss. Do not use the last trial
stage as if it were the corrected endpoint. In the current shared driver,
`_on_accepted_step` runs before state/position commitment on the adaptive
Runge–Kutta loop and after it on replay and on the multistep march; the quadrature
alarm deliberately consumes trial buffers. The new
accumulators therefore need a hook after state and position commitment, or explicit
accepted endpoint data, consistently on adaptive RK, both multistep formulations and
replay, including the low-storage march after its finish/commit block. The driver
already has one such point: the call to `StepDriver._take_step_stats`, made after
the state is finished and the position advanced in both step loops, the multistep
march, the low-storage replay and the batch driver. Its docstring records the
asymmetry of `_on_accepted_step`. That method returns at once unless periodic
statistics rows were requested and is not on the native backend's protected driver
surface, so the accumulators take a new call at the same sites rather than living
inside it; adding that call to the protected surface is the kernel schema change
named below. Rejections
contribute nothing. An in-place stage register is not retained endpoint data; any
extra buffer must respect the march's ownership contract and appear in its ledger.
Any change to that protected hook surface bumps the kernel schema. Extra transforms
needed to form endpoint rates are counted, even when a trial buffer can sometimes
be reused. Accumulator interpolation at dense save planes must be qualified separately;
until then recording these quantities requires step planes.

Warn when measured occupancy or removal relative to $U_0$ exceeds the phase 0
thresholds, with a defined zero-input case. Sample occupancy through `between_planes`
and report the maximum over the samples actually taken; it cannot certify that no
short excursion occurred between samples. Resume restores accumulators and their last
endpoint, so splitting a run or changing output cadence does not lose or double-count
removal. These are requirements for phase 2, not capabilities of the current alarm hook.

A directional occupancy, classifying collar content as outgoing or incoming by
windowed Fourier analysis [5], was also proposed; it was removed from this SEP on
2026-10-24 and is a roadmap item (Q14). One interpretive point remains:

- **Surface crossing.** The commutator of §4.7 produces a complex amplitude source,
  not by itself a positive directional energy flux. In a constant-metric, lossless
  quadratic-dispersion limit its quadratic-form balance gives the usual current.
  General dispersion requires the metric balance above and comparison against the
  wide window. Opposite-edge incoming content is evidence to investigate wrap-around,
  not an unconditional direct measurement of it.

#### 4.7 The escaped field

Two records are possible: the linearly propagated amplitude removed by the collar
(reactant–product decoupling [1, eq. 227]), and a reconstruction from a surface before
the collar (inspired by tSURFF [6]). Q11 prefers the latter, subject to phase 0b
establishing its accuracy for this nonlocal, band-limited problem. tSURFF's local
Hamiltonian result does not by itself prove absorber independence here.

**The surface record and its identity.** Choose smooth leading and trailing windows
$\theta$ ramping from zero to one in the flat region before their collars. On a common
represented band set $Q=P\mathcal F\theta\mathcal F^{-1}P$, or the corresponding
fine-grid compression. These are smooth windows, not idempotent projections.
Let $\mathcal U(z,s)$ be the collar-free *linear* evolution operator, including
material loss and the spectral collar. For a diagonal route it is
$\exp[\Phi(z)-\Phi(s)]$; a z-dependent coupled route uses its full linear flow.
The proposed record solves

$$
A'=\mathcal L(z)A+[Q,\mathcal L(z)]E+Q N(E,z),\qquad A(0)=Q E(0),
$$

or equivalently

$$
A(z)=\mathcal U(z,0)QE(0)
 +\int_0^z\mathcal U(z,s)\{[Q,\mathcal L(s)]E(s)+QN(E(s),s)\}\,ds.
$$

The initial term is required for pulse tails or initial noise outside the surface.
A surface moving relative to the chosen propagation coordinates would add $Q'E$
and is outside this record's scope; §4.11's prescribed frames keep these surfaces
fixed in their coordinates and change $\mathcal L$ consistently. For a collar-free
solution $F'=\mathcal L F+N(F)$ with the same initial field, subtraction gives

$$
A(z)-QF(z)=\int_0^z\mathcal U(z,s)\{
 [Q,\mathcal L(s)](../E-F)+Q[N(E,s)-N(F,s)]\}\,ds.
$$

This is the exact continuous identity of the proposed discrete operator model.
For a source-free problem ($N\equiv0$) the second term vanishes; for a *linear* source,
such as SEP 14's time-local witnesses, it does not, and the identity holds with
$\mathcal U$ the flow of $\mathcal L+N$ instead. Equality $A=QF$ requires the first
term to vanish as well: the absorbed and collar-free solutions must agree wherever
the commutator samples them. With a local differential operator that can be a narrow
surface layer; with full/tabulated dispersion, a hard band limit or spectral loss,
the commutator can have long tails. Its effective width and the resulting bias must
be measured (Q12). Even when $\theta=1$ on the damping region, $QC=C$ does not follow
after band compression. Reflected and wrapped content invalidate the reconstruction
by changing $E$ where the commutator acts. There is no identity valid for every
collar setting, and no finite buffer guarantees exact locality.

The $QN$ term includes nonlinear generation in the selected exterior of the *absorbed*
run. It is not a correction for the missing nonlinear evolution of already removed
radiation. The record is useful only where that exterior evolution is adequately
linear, or its error is independently small. A Raman soliton, plasma wake or persistent
rotational coherence can violate this assumption. Diagnostics may suggest a problem;
only the wide-window test qualifies the reconstruction.

**Integration and cost.** Evaluate the commutator with the actual discrete operator,
including the projected window products. For a fixed diagonal flow this takes roughly
two transform pairs per endpoint and edge without reusable intermediates; count
shared work, fine-grid padding and coupled-flow actions in the measured ledger.
Integrate with forward local propagators, never a global $e^{-\Phi(z)}$ accumulator:
inverse material/spectral damping can overflow. Use exponential weights or another
stable treatment for the linear part. A trapezoid is an initial second-order candidate
for the remaining integrand, not a guarantee of accuracy: phase mismatch can make it
oscillatory even when it contains no stiff collar factor. Compare three step levels,
set a record-specific error budget, and refuse unsupported dense planes. Keep a state
array per edge and the endpoint/work buffers that the chosen quadrature requires.

**The removal record.** With $B(0)=0$, solve

$$
B'=\mathcal L(z)B-CE.
$$

With $N\equiv0$, $E+B=\mathcal U(z,0)E(0)$ in the continuous discrete-space
model, and with a linear $N$ the same holds for $\mathcal U$ the flow of
$\mathcal L+N$, since $(E+B)'=\mathcal L(E+B)+N(E)$. This holds for arbitrary
non-commuting $C$ and $\mathcal L$; numerical integration still has to satisfy the
identity. For a split step, propagate the old
record by the same collar-free linear map and add the complete complement
$(I-S_h)E^-$ of the actual projected collar map. Then the linear identity telescopes
to rounding relative to that numerical linear flow. Recording only the unprojected
$(1-e^{-\alpha h/2})E^-$ misses the mask's removal. Under the in-stage scheme, a
trapezoid may underresolve $CE$; its error needs its own convergence bound. Storing
collar history at the march's order is an alternative with a larger memory cost.

This record is simpler and needs no surface buffer, so it is the fallback if the flux
record is unaffordable or fails its accuracy gate. It reconstructs what the numerical
collar removed, not necessarily the physical field outside the requested window.

**What may be reported.**

- Store complex amplitudes, with the output's optics convention, coordinate origin,
  route normalisation, surface definitions and reconstruction error recorded. Expose
  spectral densities using the existing physical weights and Jacobians, not raw
  $|A_k|^2$. Leading and trailing contributions may interfere at a given frequency;
  retain their phases even if plots normally show each separately.
- A periodised, adequately sampled field has Fourier-series coefficients equal to
  samples of its continuous spectrum, with the stated DFT factors. This does not make
  the proposed numerical record exact: discretisation, reconstruction, periodisation
  and unresolved fine spectral structure still need qualification. Zero-padding
  interpolates the sampled surface source; it does not recover a missed exterior or
  lost spectral information. Energy must come from an independently checked balance,
  not an unqualified sum over an undersampled escaped spectrum.
- Temporal separation alone does not justify adding spectral intensities. Coherent
  spectral sums are meaningful when the reconstruction, phase origin and resolution
  are qualified; for the linear removal identity the coherent sum is essential.
  An incoherent sum is a separate diagnostic, appropriate only after ensemble or
  instrument averaging suppresses the cross term. Any spectrometer prediction must
  average the coherent spectrum over its stated response.
- A smooth exterior window's norm and the interior remainder's norm do not add to
  the total without cross terms. The same issue, together with $M$, prevents treating
  this amplitude record as an automatically positive escaped-energy ledger.
- The record covers crossings up to the final propagation plane only. Slowly walking
  content still inside the surface has not escaped. Extending z for completeness
  would change the physical propagation length, unlike waiting after a TDSE pulse.

#### 4.8 The reference: a wide window

The reference and the production candidates of §4.10 have distinct roles. The
reference is widened and refined until its uncertainty is below the comparison
budget; a candidate is the least costly qualified window at a target accuracy.
The widest reference's cost is not assigned to every candidate without absorption.

The collar imitates an unbounded window, so the reference is the same problem on a
window $M$ times wider at the same sample spacing, compared on the requested analysis
region $|t| \le T_{\mathrm{req}}/2$. Today's narrow collar may enter that region; the
comparison must expose that intrusion. Start with $M=2$–4, but widen further
if needed. Convergence means the $M$-to-$2M$ difference is below a stated
absolute/relative budget for every reported observable, not literal equality.
Independently tighten the step tolerance and check frequency-grid interpolation/linear
tables, temporal resolution and response quadrature. Keep the physical launch,
dispersion, spectral collar and material parameters fixed; disable the temporal collar
in the reference or show its influence is below that budget. For comparison use the same
explicitly defined time aperture on both fields, and the Fourier transform of that
aperture for an in-window spectrum. Do not compare unequal frequency meshes without a
documented restriction/resampling and DFT scaling.

Use noise-free witnesses first. Equal seeds on different FFT grids do not give the same
physical noise field: noisy comparisons need a common prescribed realisation with
consistent spectral normalisation, or a separate ensemble uncertainty. Extended temporal
support also changes response memory and initial preparation, which must be converged.
The measured quantities are:

- **the absorber error:** the in-window field and spectrum, per mode, against the
  reference, plus the band observables each case is known for (the UV energy for RDW, the
  soliton centroid for SSFS);
- **the leak:** the reference's out-of-window field that wraps back in through the
  narrow window, and the narrow run's energy in the region that should be empty;
- **the escaped records' errors:** the flux record and the removal record, each against the
  reference's out-of-window spectrum;
- **the reflection:** packets of controlled walk-off launched into the collar, with
  $\Omega w$ from 1 to 30 and the strength $AN$ from 50 to 270 (the walk-off default's
  range on the cases of §4.3, more on slices), against the §4.2 criterion; reflection
  grows with the strength at fixed $\Omega w$ (§4.2), so a sweep in $\Omega w$ alone
  cannot support R1.

This separates each discrete absorber model's *intrinsic* error from numerical
integration error, which earlier studies measured relative to a fixed collar.
With different projected or sampled collar operators, their converged absorber models
can differ too; each scheme therefore needs its own step-refined limit.
When frames or launch positions differ, align the fields in one physical retarded
time and phase convention before applying the same analysis aperture. Preserve
physical pulse separations, carrier phase and material preparation. Changing window
size can also change a response gate or causal-memory truncation; record and converge
that change separately, including the atomic source gate where present. Disabling
the temporal field absorber must not silently disable a response's source policy.

**The SEP 27 plasma baseline.** Field-grid rate plasma forms its
polarisation-equivalent source on active bins as

$$
A_J=\frac{\operatorname{DFT}[g_sJ]}{i\omega}
    -\frac{\operatorname{DFT}[g_s'J]}{(i\omega)^2},
$$

where $g_s$ is the fine-grid source gate and $g_s'$ its analytic time derivative.
This is a source prescription, distinct from the damping rate $\alpha(t)$ and from
the monitor's guard weight. `temporal=False` removes neither product nor the gate
certificate. The same held-current source is required in every plasma candidate and
reference. A steady residual current at an edge is intentional; demanding that $J$
vanish there would reject the very case SEP 27 treats. Conversely, negligible optical
field in a guard does not certify that current variation or Raman memory is negligible.

Changing window size, placement or gate geometry must preserve the source preparation
and rebuild the derivative and sampled-gate certificate where their inputs change.
SEP 27's approved T2 correction requires independent sampling-error bounds for
$\operatorname{DFT}[g_sJ]$ and $\operatorname{DFT}[g_s'J]$: cancellation between
the two edges can make the steady-current certificate pass while a current confined
to one edge remains inaccurate. Include those one-edge witnesses and separate product
refinement; a passing certificate or field guard is not a full source-error bound.
Keep the source gate fixed when varying only damping strength/profile on a given
grid. If a geometry change also changes the source gate, identify and converge both
effects. Envelope plasma, Kerr/Raman and the TDSE source retain their own policies.

#### 4.9 The phase-space filter

Soffer and Stucchio's time-dependent phase-space filter [5] is designed for periodic
FFT interiors, as are the absorbing potentials already in use here. Its distinguishing
feature is selecting outgoing content in phase space.

**How it works.**

- **The frame.** At intervals, it projects the field in a buffer at each edge onto a
  Gaussian windowed-Fourier frame [5, eqs 2.1–2.5], computed by local transforms over
  the buffer only.
- **What it deletes.** It deletes the framelets that the *free* flow would carry out of
  the window before they could return [5, eqs 2.8–2.9]. Here that needs each framelet's
  walk-off rate and its dispersive spread, both cheap from the known $\beta(\omega)$.
- **What it keeps.** Slow framelets, and framelets moving along the boundary, are
  classed ambiguous. They are kept and reported, and if their norm exceeds a tolerance
  the run stops [5, Remark 2.3]. In their tests this flagged every failure, with a few
  false alarms.

**How it differs from both of today's schemes.**

- It is not a term in the equation, so it adds no eigenvalue and no floor.
- It removes only content that is leaving anyway. It avoids the same repeated time-multiplier split, but localisation, outgoing
  classification and finite-interval errors remain; these must be measured.
- Its boundary error falls exponentially with the buffer width, under the paper's free Schrödinger-flow assumptions, provided the interval stays
  below a cap: the buffer width divided by
  three times the fastest walk-off [5, p. 2–3]. The full proof is in their later work,
  not this paper. Extending that bound to full UPPE dispersion and the hard band mask
  is a hypothesis for the prototype, not an established error estimate.

**Where it fails.**

- Its assumption is that the field in the buffer evolves freely. A nonlinearity there
  breaks it, which is the remaining part of the split's error, through the source.
- Their success on fast NLS solitons is, in their words, coincidental [5, pp. 11–12].
- The interval cap may be short for broadband supercontinua.
- Slow content at the edges, including shot noise, is by construction ambiguous.
- As a discrete operation at fixed z it would restart the multistep history. Whether
  that is needed when the source is negligible in the buffer is untested.

It is not a candidate default. The exploratory prototype on the Runge–Kutta path and
the classification diagnostic this section once motivated were removed from this SEP
on 2026-10-24 (Q13, Q14) and remain roadmap items; the analysis is kept for them.

#### 4.10 Retaining the field: cost and accuracy without temporal absorption

At fixed fine and coarse sample spacings and represented frequency range, a window
$q$ times wider has about $q$ times as many samples and frequency spacing $1/q$ as
large. Local response work and state/history storage grow approximately as $q$;
FFT work grows as

$$
\frac{C_{\rm FFT}(q n_t)}{C_{\rm FFT}(n_t)}
\simeq q\frac{\log(q n_t)}{\log n_t}.
$$

This is an operation-count model, not a timing prediction. Actual coarse/fine sizes,
FFT rounding, modal/node work, response memory, cache/device capacity and any separate
atomic window determine cost. At $n_t=2^{14}$, the model gives factors 2.14 and 4.57
for doubling and quadrupling the window. Measure both power-of-two and smooth sizes;
report realised widths and sample counts, not only requested enlargement factors.

Write total measured cost as $K C+T_{\rm monitor}+T_{\rm other}$, where $K$ counts
physical-source evaluations including rejected attempts, $C$ is their mean cost
including the associated march work, and the remaining terms include boundary
checks, setup and output. Ignoring the latter terms only for an initial estimate,
the enlarged window wins when

$$
\frac{K_{\rm collar}}{K_{\rm wide}}
>
\frac{C_{\rm wide}}{C_{\rm collar}}.
$$

The absence of the temporal transform pass helps this inequality. Retaining the
default temporal rate on a wider grid does not remove its peak-rate cap (§4.3).
The comparison therefore uses `temporal=False`, leaving the spectral collar and hard
mask fixed. Extra response work over retained tails and extra monitoring are counted.

The hypothesis is strongest where the collar forces hundreds of steps but the physical
source needs far fewer. It is weaker where nonlinear accuracy or source-history
variation already limits the step. SEP 16's 1329-to-965 canonical call reduction came
from a per-step collar, not an enlarged grid without absorption; as a call ratio alone
it is below the 2.14 operation-count factor of doubling. But a retained window also
drops the temporal pass, which SEP 22 measured at 2.0–2.3 times the per-step cost on
`numba_cpu` and 1.6 on the native CPU build for a Cartesian Kerr case, so $C_{\rm wide}/
C_{\rm collar}$ can be well below the operation count; the ratio is backend- and
route-dependent and is measured, on every CPU backend, not estimated. Its RDW ablation
found little benefit from weakening the collar. These observations motivate distinct cases, not a universal speed-up.

Removing a collar removes an artificial term and its integration error. It does not
remove temporal aliasing, truncation of response memory, spectral discretisation error
or propagation error. Fewer steps are a cost benefit, not an accuracy argument.
For each strategy, sweep tolerance and width to find its measured wall-time/memory
tradeoff at the same predeclared observable-error targets against §4.8. Compare the
same physical analysis aperture and also declared weak-band or exterior observables;
a small error in total energy can conceal a large error in weak radiation. A candidate
with significant wrap-around is inadmissible regardless of its runtime.

Phase 0 includes the current realised width without temporal absorption, then wider
windows starting at factors two and four and intermediate smooth sizes where useful.
The no-absorption candidate must not be charged an unnecessary enlargement if the
original window is already adequate. If no affordable candidate meets the accuracy
and monitoring requirements, record the measured memory/runtime limit and retain a
collar for that case. Reference construction and optional escaped-field recording
are costed separately; comparisons enable equivalent requested outputs.

#### 4.11 Placement and frame: reducing unused time samples

For a linearly propagating packet born at $(z_b,\tau_b)$, stationary phase in the
conventions of this SEP gives

$$
\tau_{m,\omega}(z)=\tau_b+
\int_{z_b}^{z}\left[\beta_{1,m}(\omega,s)-\beta_1^{\rm f}(s)\right]ds.
$$

The fastest light has the smallest group delay per metre among the relevant modes
and frequencies, not necessarily the highest frequency. Choosing that delay as the
frame makes the linear walk-off one-sided; placing the launch towards the early
edge then uses the window efficiently. A midpoint frame is another candidate. In a
constant medium, for co-launched packets sharing an initial support of width $W_0$,
the extreme packet separation at length $L$ is

$$
\Delta\tau=L(\beta_{1,\max}-\beta_{1,\min}),\qquad
T_{\rm needed}\gtrsim W_0+\Delta\tau+\text{guard margins}.
$$

This is a packet-support estimate with dispersion and tail margins still to be added,
not a compact-support theorem for a band-limited field. A fastest-light frame with
asymmetric launch placement and a midpoint frame with symmetric placement can both
reach this estimate. A centred launch in the fastest-light frame instead requires
about $W_0+2\Delta\tau$. These are alternative ways of recovering unused space, not
independent factors to multiply. No frame reduces the separation of retained packets.

In a varying or nonlinear problem, use the envelope of trajectories over the entire
run, including late-born radiation, modal walk-off, spreading and frequency evolution.
A frame following the pulse can remove common drift but cannot remove its growing
temporal extent. A velocity range inferred only from the launch spectrum can miss
new radiation. Phase 0 states the range used for planning and a conservative range
for boundary monitoring; it reports the cost of either range rather than silently
discarding weak bands. A monitoring velocity bound does not set absorber strength
or reverse Q3's rejection of that strength prescription. Near-zero drift can still
carry dispersive spreading.

Compare the existing frame and centred launch with fixed asymmetric placement, a
midpoint frame and a fastest-light frame where their construction is supported.
Start with constant-frame witnesses using `linear.Frame`; a prescribed varying
frame is a prototype until its representation, smoothness and support are specified.
All modes share one frame. Keep the envelope carrier definition separate from the
frame's group delay, and preserve field-grid carrier-envelope phase. Evaluate cost
again after changing frame: exact linear translation does not imply frame-independent
errors in the exponential multistep approximation to the source history.

Fixed placement can use pulse delays on the existing centred lattice, with matching
output coordinate metadata and all drives/preparations shifted consistently. It does
not require a general offset-grid API. Phase 2 implements only the qualified setting
support and guidance selected by R6/R7. Solution-dependent recentering and expansion
remain out of scope: they require transport of multistep history, noise, response
memory and output coordinates, and cannot recover content that has already wrapped.

#### 4.12 Monitoring a window without an absorber

**An alarm, not an escaped-field reconstruction.** Reserve guard strips inside both
periodic edges, separately from any absorbing profile. On the conventional modal
route, use the nonnegative proxy of §4.6 with a guard weight $g(t)\in[0,1]$:

$$
\eta_g(z)=\frac{\sum_{m,p}\int g(t)|b_{m,p}(t,z)|^2\,dt}{U_{\rm ref}}.
$$

The declared analysis aperture lies inside the guards. The requested computational
window includes those margins explicitly; a zero temporal rate neither grows the
grid nor silently reduces the analysis aperture (Q4/Q16).
Report each edge separately, its peak field/power proxy and the maximum occupancy
over the samples actually checked. Fix and record $U_{\rm ref}$ and absolute as well
as relative thresholds; define zero-input behaviour without division by zero.
Other routes require their qualified metric (§4.6). Occupancy is accurately computable
for the sampled state but is not cumulative escaped energy or a universal bound on
boundary error. A fraction $\epsilon$ in energy has field-norm scale
$\sqrt{\epsilon}$; interference and nonlinear sensitivity can amplify its effect.
Weak-band observables, intentional shot noise and response tails need explicit
threshold treatment. A noise subtraction must not hide amplified radiation.

**Wrapping cannot be inferred from a final state alone.** On a periodic grid, pure
advection through one complete window multiplies every Fourier coefficient by
$e^{-2\pi i k}=1$. The initial and final fields, their energy and edge occupancies
are identical despite a complete crossing. Neither conservation of total energy nor
small final edge values certify containment. A last-trial stage is not the accepted
endpoint, and even endpoint checks can miss an excursion between them.

**Sampling is part of the numerical contract.** For guard width $w_g$ and a stated
bound $v_{\max}$ on relative group-delay drift, the packet estimate

$$
\Delta z_{\rm check}\le\theta\,w_g/v_{\max},\qquad 0<\theta<1,
$$

sets an initial check cadence. $v_{\max}$ is not the drift range of the whole
frequency grid: its edge bins carry only noise and would make the cadence arbitrarily
short, the reason Q3 rejects grid-predicted strength. It is taken over a declared
monitored band, the bins and modes whose content exceeds a stated spectral floor, or
the conservative monitoring range of §4.11, and content outside that band is covered
by the guard occupancy itself, not by the cadence. R7 defines the band. Phase 0a
calibrates $\theta$ and adds spreading and
nonlinear allowances; for negligible drift, spreading still sets a check interval.
This is not a finite-propagation-speed theorem for full dispersion and projection.
Reference [5]'s related buffer/velocity criterion assumes free exterior evolution;
it is not automatically an error bound here. Check cadence is independent of saved
planes. Qualified dense output may supply interior checks without physical-source
calls; otherwise land on check planes and include the resulting steps in the cost.
Interpolation has its own error budget and must detect the crossing witnesses below.
An adaptive march must reject or subdivide a step whose check coverage cannot be
established; an unexamined interval cannot be marked safe. A fixed replay schedule
must satisfy the required coverage before execution. In particular, SEP 24 supplies
no dense checks: reject an inadequate schedule or construct and calibrate a new one,
including any added check planes. Do not silently subdivide a replay and still claim
the original schedule's convergence result. Output thinning must not disable checks.

**Observation and memory contracts.** SEP 22's `observe(field=False)` deliberately
avoids downloading the field and skips the spatial collar diagnostic, recording
`collar_diagnostic_skipped`. Neither that observation nor periodic statistics supply
the independent safety coverage required here. Implement checks as backend-resident
transforms/reductions where supported, returning compact measurements. Do not require
a whole-field host copy or alter the accepted state to form them. Reusing source
scratch needs proof that it represents the committed endpoint and that every source
stream has finished with it; SEP 27's two current products and SEP 28's tiled buffers
make that ownership explicit. Count any additional scratch, transforms, device
synchronisation and host traffic in both the memory ledger and the campaign.

**What a flux measurement adds.** For the scalar constant-metric envelope witness
$\partial_z a=-v\partial_t a+i\beta_2\partial_t^2a/2$ with a local conservative Kerr
term allowed, $v$ is relative group delay in s/m and $\beta_2$ is in s²/m.
Multiplying by $a^*$ and adding the conjugate gives

$$
\partial_z|a|^2+\partial_t j=0,\qquad
j=v|a|^2+\beta_2\operatorname{Im}(a^*\partial_t a).
$$

This holds in the internal Fourier convention of this section; on saved arrays in the
optics convention the dispersive term of $j$ changes sign. Thus
$d\int_{t_l}^{t_r}|a|^2dt/dz=j(t_l)-j(t_r)$; pure advection is the
$\beta_2=0$ case. Boundary flux and its z-quadrature can be tested against this
identity. General dispersion, changing metrics, losses and nonlocal sources require
the operator balance of §4.6. Signed net flux does not count unique outward escape:
recrossings, coherent contributions and already wrapped content prevent that
interpretation without further assumptions. Directional analysis (a roadmap item,
Q14) and the
surface record (§4.7) are optional refinements, not requirements for the basic alarm.

**Policy and qualification.** The candidate default has an early warning threshold
and a larger stop threshold, both calibrated to act while significant field is still
inside the guards and before the declared error budget is exceeded.
An explicit warn-only policy may continue, but records the violation and does not
certify subsequent results as free of boundary contamination. A failure reports
edge, z, measured value, threshold, guard geometry and check coverage, and suggests
a larger window, better placement/frame or an explicitly chosen absorber. No silent
switch of boundary treatment or automatic grid expansion is introduced.

R7 qualifies the alarm against wider-window witnesses at declared accuracy budgets,
records false alarms and missed violations, and defines supported routes, noise
policies and plane modes. Passing finite witnesses establishes that tested scope,
not a theorem for every nonlinear propagation. If reliable checks are too costly or
the scope too narrow, R6 retains the existing default and may admit an opt-in mode.
Persist monitor geometry, coordinate origin/frame, thresholds, maxima, violation
status and next check position across resume. Only accepted states or qualified
dense states contribute; rejected trials do not create final alarms. Transform,
reduction, device synchronisation and extra-step costs belong in the comparison.
Reset this state for each fresh input to a reusable `Simulation.runner()`; resume
restores it instead. Specify which committed state a stop can checkpoint and retain
the violation that caused it. A low-storage march may have overwritten its previous
step; SEP 22/24's held observation plane does not imply an arbitrary last-safe-step
snapshot is available. If rollback is promised, its storage and work must be measured.

### 5. Proposed interface

The interface depends on the phase outcomes; the sketch shows the most it would add.

```text
grid = FieldGrid.from_limits(..., collar_fraction=0.05)   # existing; default per phase 0

absorber(grid, *, boundary_N=20, boundary_length=None,
         temporal=True,             # existing; False keeps the spectral collar
         walkoff=None,               # protected drift, s; overrides temporal strength only
         profile="complement")       # or "plateau", if phase 0 admits it

SimulationSettings(
    absorber_scheme="nonlinear" | "split_step" | "implicit",   # "implicit" after phase 4
    escaped_field=False,             # the record of §4.7, opt-in
)

prop_capillary(..., absorber=False)               # no absorber
prop_capillary(..., absorber_scheme="split_step") # roadmap keywords

boundary_check = BoundaryCheck(      # proposed; details and defaults fixed by R7
    guard_width=...,                # seconds at each edge, independent of collar width
    warn_threshold=..., stop_threshold=...,
    action="error",                 # explicit "warn" keeps a recorded violation
)
settings = SimulationSettings(..., absorber=absorber(grid, temporal=False),
                              boundary_check=boundary_check)  # proposed setting
Simulation(..., settings=settings)

run.statistics["collar_occupancy"]       # eta(z), every saved plane
run.statistics["guard_occupancy"]        # each edge; maxima over the actual check cadence
run.statistics["absorbed_energy"]        # U_t(z) and U_omega(z)
run.escaped_spectrum(z)                  # physically normalised spectral density
```

What is refused:

- `escaped_field=True` without nonzero temporal rates (a spectral-only collar has
  none);
- `"implicit"` on either Runge–Kutta stepper (outside this SEP);
- interpolated planes beside a nonzero split temporal collar, as today; an inert
  scheme setting with no temporal pass does not trigger this refusal;
- monitored configurations without qualified metric, check coverage or dense-state
  support; explicit check planes may be used where qualified (§4.12).

Basic diagnostics are intended to be on by default, subject to phase 0 measuring
their endpoint transforms and reductions. The escaped record is a separate opt-in. Unsupported route/stepper/plane combinations fail by name.
Guard checks are available with or without temporal absorption and are separate from
the removal ledger. The sketch does not fix a scalar threshold for every route:
R7 specifies absolute/relative normalisation, bands, noise handling, guard margins
and cadence before implementation. A monitored default without temporal absorption
requires R6/R7 and a maintainer decision; it must remain accessible without disabling
the spectral collar. Phase 2 supplies the corresponding convenience-interface path.
`absorber=False` disables both soft collars, temporal and spectral, retains the hard
representable-band mask, and rejects conflicting strength or scheme settings. On the
spatial interfaces (`prop_radial`, `prop_free`, `prop_planar`) it does not touch the
transverse collars, which keep their own settings; the docstrings say so, and a
switch for those collars stays out of scope (§3; John Travers, 2026-10-24). A spectral-only collar, its
temporal rates identically zero while the spectral rate stays, is built by
`absorber(grid, …, temporal=False)` and compiles no temporal pass under either scheme;
SEP 22's phase 1 supplied both, and this SEP keeps them (Q16). The proposed `walkoff`
option changes temporal strength while preserving the spectral rate. Phase 0a must
specify which explicit parameter combinations conflict and how the parent interval
is supplied (Q18). [SEP 22](sep-0022-memory-lean-spatial-execution.md)'s
`Absorber.collar_length` currently sets both temporal and real-space spatial strength.
Mapping `walkoff` into that shared field therefore needs R4's decision about spatial
coupling, or a separately approved strength parameter; it cannot silently change
the spatial boundary. Phase 0a settles the precise signatures, route support
and resume agreement of phase 2's settings, to a rounding bound rather than bit
identity ([SEP 25](sep-0025-rounding-level-reproducibility.md)), before phase 2
begins; phase 0b settles those of the escaped-field record before phase 3.

### 6. Implementation strategies

**Layers:**

- `solprop.boundaries` (the physics layer): profiles, strength and guard-policy
  definitions; `linear.Frame` and input placement retain coordinate ownership;
- `solprop.translate`: rates, scheme and qualified guard data reach the kernel problem;
- the backends: guard reductions, the implicit solve and the record;
- the driver and output: check scheduling, boundary failures, statistics, coordinate
  metadata and the escaped spectrum.

**Retained-window mode (phase 2, after R6/R7).** Reuse the spectral-only absorber;
verify that the temporal pass is absent on all backends. Add the qualified guard
monitor and convenience-interface path, with no automatic change to the requested
physics. Fixed launch placement and supported prescribed frames use existing
coordinate conventions; any added setting must have a concrete phase-0 specification
for phase, output alignment, response preparation, history and resume. If a frame
candidate requires a larger redesign, retain its measurements and defer that setting
explicitly rather than blocking the basic retained-window mode.

**Geometry and strength (phase 2).** These are physics-layer and grid changes. The
grid's temporal collar is shared state, not only absorber geometry: besides the
absorber's rates it sets the field-grid plasma source gate and its derivative, the
Raman kernel's tail taper, the TDSE descriptor's time collar, the recorded analysis
window that processing reads back, and the flat region SEP 31's remapping and pulse
measures use. A geometry change states its effect on each of these consumers and
keeps or deliberately updates it, with a test; changing the absorber must not move a
source gate silently (§4.8).

- **`walkoff`** sets the temporal peak from the realised profile area. For a shape
  $q(t)\in[0,1]$, set $\alpha_{\max}=2\Gamma_{\min}v_p/\int_{\rm edge}q\,dt$,
  where $v_p=\Delta\tau_p/L_{\rm ref}$. For the complement profile this is
  $AN=4\Gamma_{\min}\Delta\tau_p/w$. A plateau has a different integral.
  $L_{\rm ref}$ and the requested protected drift refer to the same parent interval;
  a resumed slice preserves the already constructed rates. Do not implement this by
  changing a shared `boundary_N`, which would also alter the spectral collar. A
  default tied to the requested window, such as $\Delta\tau_p = T_{\mathrm{req}}/2$,
  fixes consequence 2 of §4.3: on the tight grids of §4.3 it gives $AN$ of 199–270
  against today's 252, and on the full CtC grid 50, five times weaker. It does
  not fix consequence 1, because $\alpha_{\max} = AN/L$ still grows as a slice
  shortens. A slice keeps its parent run's strength with the existing
  `boundary_length`, which the manual should say. It changes every run's absorber, and
  is for the maintainers to decide (Q3).
- **A new `profile`** is a function of $W$, so its kernel data are the rates already
  carried. The grid must guarantee both $t_{\rm flat}\ge T_{\rm req}/2$ and the
  requested edge fraction, including its asymmetric last stored sample. Ignoring that
  one-sample offset, $T\ge T_{\rm req}/(1-2f)$. Round the fine grid and choose a
  compatible coarse divisor without breaking its pad/crop invariants. Test both FFT
  size policies just below, at and just above a rounding boundary.

Geometry/profile arrays alone need no `SCHEMA_VERSION` change. This does not exempt
the endpoint, diagnostic or record contract changes below; their actual seam changes
must be inventoried together before implementation.

**Diagnostics (phase 2).**

- Implement the metric-aware rates and the occupancy proxy of §4.6 on each route.
- Implement the independent guards and warning/error policy of §4.12. Guard checks
  do not depend on absorption being enabled or on an escaped-field record. Their
  sampling schedule and violation status survive output thinning and resume.
- Commit endpoint accumulators only after an accepted step is complete. Qualify
  adaptive, replay, batch, restart/resume and output-cadence agreement within §7's
  bounds.
- Record quadrature error separately from the propagation error. Use the exact
  before/after energy difference for the discrete split map.
- Bump the kernel schema if endpoint data or the protected driver hook changes;
  backend-private reductions alone do not require one. Bump the output schema for
  persisted accumulators, diagnostics and their definitions.

**Escaped field (phase 3, after R5).**

- Implement the qualified surface or removal record of §4.7, with forward local
  linear flow, endpoint quadrature and complex amplitudes per edge where meaningful.
- The measured commutator tail and record-error bound set the surface gap and route
  admission. If neither record meets the accuracy/memory gate, defer phase 3 and
  explicitly retain the missing feature on the roadmap; geometry work may proceed.
- Persist the complex accumulators, last quadrature endpoint, surface/metric identity
  and errors for resume. Dense observations require a separately qualified partial-step
  accumulator; initially require step planes for these records.
- Kernel and output schema bumps; device-resident work in the native backend. List all accumulators
  and work buffers in `memory_ledger()`, including per-batch-member storage.

**Implicit collar (phase 4, only if phase 1 passes).**

- **A new kernel operation:** the projected pointwise solve or CG for the interaction
  formulation, and GMRES or a damping lab-frame weight for the exponential one, in the
  predictor as well as the corrector (§4.5). A schema bump.
- **The order choice:** the exponential formulation's capped order weighing (SEP 16
  phase 6) moves to the implicit intervals.
- **The backends:**
  - `numpy_ref` first, as the oracle;
  - `numba_cpu` with the solve in compiled passes;
  - the native backend, where CG is a few batched FFT pairs and reductions per corrector, on device
    as on host.

**Cost:**

| piece | cost |
|---|---|
| retained wider window | roughly linear response/storage growth and $n_t\log n_t$ transforms, offset against actual calls saved; measure realised coarse/fine sizes and total wall time |
| guard monitor | temporal synthesis when not reusable, reductions, dense-state evaluation or extra check-plane steps, and any device synchronisation |
| diagnostics | endpoint transforms plus metric reductions per accepted step; plane/periodic occupancy |
| record | state and quadrature buffers per edge; commutator transforms or the split complement, measured per route |
| implicit solve | transforms in predictor and corrector; Krylov iterations depend on condition number and residual target; no required extra physical-source call |
| wider collar | grows $n_t$ when there is no slack: at most a factor of two on a power-of-two grid, less on 2-3-5-smooth sizes |

No new dependencies.

### 7. Verification and validation

**Analytic and unit:**

- **Profiles:** integrals against $\int(1-W) = w/2$; the `walkoff` parameterisation
  reproducing the continuum steady-drift model $\Gamma=\Gamma_{\min}$ at
  $\Delta\tau_p$; sampled profile quadrature converges under refinement, rather than
  being assumed exact to rounding;
  `stable_step` against $2r/\alpha_{\max}$.
- **The floor law:** fixed-step runs on a scalar collar witness diverge just past
  $n_{\min}$ and complete just inside it, for each PEC order, bracketing the table of
  §4.3 (as `stable_step`'s DOPRI5 bracket does today), with the bracket's width stated
  and divergence judged against the attained spectral bound of the discrete collar
  operator, not $\alpha_{\max}/2$ alone.
- **Window, frame and guards:**
  - Pure advection of a band-limited packet: analytic translation, one full-period
    crossing invisible at the endpoints, and a narrow packet crossing between output
    planes. The qualified check schedule must flag the approach before wrap exceeds
    its budget; include intentionally undersampled checks that must not claim coverage.
  - A spreading Gaussian with quadratic dispersion, including zero central drift,
    tests guard occupancy and the signed flux identity of §4.12 on three refinement
    levels. Bidirectional, recrossing and late-generated packets test the limits of
    interpreting flux as unique escape.
  - Frame/launch-shift equivalence on a common physical aperture, preserving energy,
    carrier phase, mode simultaneity, relative pulse delays and response preparation.
    Test the simple fastest-frame/asymmetric and midpoint-frame width estimates;
    measure source-call changes separately from field agreement.
  - Guards on field and envelope grids, zero input, intentional noise, weak bands,
    dispersive modal weights and each admitted spatial/general-modal metric. Include
    significant low-energy radiation beside a strong pump and nonlinear walkers.
  - Compare monitor warnings/stops with wide-reference errors over guard widths,
    thresholds and check intervals. Count false alarms and missed budget violations;
    demonstrate separately the accuracy of dense checks, where admitted. A matching
    formula in a unit test does not qualify a nonlinear alarm.
  - Adaptive/replay, batch, output-cadence and checkpoint/resume comparisons preserve
    coverage and accumulated violation status, with numerical diagnostic values
    judged by the applicable rounding or reference-accuracy budget.
  - Cover statistics-only observations, fresh inputs on a reusable runner and the
    low-storage fixed schedule. Verify that inadequate check coverage is refused,
    alarms do not depend on field downloads, and checkpoint claims match the state
    actually retained. Repeat affected buffer-ownership and cost checks after SEP 28.
- **The implicit collar:**
  - SEP 14's witnesses: the commuting scalar $u' = b\,u - a\,u$; a linear time-local source
    with and without dispersion; the NLS with Kerr in the window and in the collar.
  - Convergence orders on three step levels, against the in-stage collar's.
  - Stability measured past the PEC limit up to the AM interval, coupled with an
    imaginary-axis source eigenvalue, for each predictor variant.
  - The collar's per-step damping factor and nepers per crossing as functions of
    $h\alpha/2$, and whether its sign alternates.
  - Whether the step controller's error estimate sees collar error.
  - The projected division's residual and solution error against a dense solve on tiny
    grids and converged CG; CG agrees with the dense solve within its stopping tolerance
    plus rounding, and a matched (rounding-bound) comparison fixes the iteration count.
- **The records and diagnostics:**
  - Test the surface identity and its residual equation in §4.7, including nonzero
    $QE(0)$, tabulated dispersion, hard projection, material/spectral loss and varying
    linear flow. Compare against $QF$, not a differently windowed exterior spectrum.
    Sweep surface gap and collar strength to separate reconstruction bias from
    quadrature and propagation errors; include deliberate failure cases.
  - Test the removal identity for the complete split complement and exact linear
    witnesses within a rounding bound fixed before measurement, the step count times
    the transform bound ([testing.md](../../testing.md) §3); for in-stage quadrature use
    three refinement levels and its measured error budget, separately from the march's
    tolerance.
  - On [SEP 23](sep-0023-far-field-aperture-spectra.md)'s supported routes, compare
    `absorbed_power`'s spectral channel (J/m) with the spectral ledger's rate using
    the same operator and metric. Test the integrated energy against exact linear
    exponential weights or an independently converged rate integral (§4.6), including
    stiff bins; a raw endpoint trapezoid is not the reference. Preserve all four
    existing channels, including custom spectral rates and signed material gain.
  - Test occupancy at zero input, field/envelope Parseval factors, dispersive modal
    weights and each route's metric. Check temporal passivity instead of assuming it.
  - Check the restricted conservative energy ledger, then fixtures with material loss,
    Raman/plasma energy exchange, variable metrics and projection loss using the full
    balance. A diagnostic residual must be consistent with its own quadrature bound.
  - Verify coherent spectral reconstruction and resolution/averaging limits on
    separated pulses. Compare the escaped estimate with the identically windowed
    wide-reference field, with consistent Fourier scaling and route weights.
  - Verify rejected-step isolation and final-step accounting exactly (the accumulators
    count the steps actually accepted); the accumulators of a replayed schedule across
    backends, batch members and a checkpoint/resume within a rounding bound fixed before
    measurement; and adaptive against replayed runs, and runs of different save-plane
    cadence, within the case's accuracy envelope against a tighter-tolerance reference
    ([SEP 25](sep-0025-rounding-level-reproducibility.md)); including nonzero
    initial exterior field and a stiff spectral collar that would overflow a global
    inverse flow.

**Against the wide window (phase 0, then each implementation phase).** Every applicable
case includes the cost/accuracy comparison of §4.10 and the placement/frame candidates
of §4.11, with the monitored no-absorption mode and its check costs included:

- the N = 3 GNLSE soliton launched into its collar (ADR-025's witness);
- SEP 13's canonical capillary, whose pulse stays clear;
- the M2 RDW capillary and the reduced CtC capillary, whose broadened fields reach the
  collar;
- a short Raman-soliton case modelled on the SSFS gallery example, with a nonlinear
  walker;
- one free-space radial case, for transfer only.

Tolerances:

- In-window errors are reported against the reference's own $M$-to-$2M$ difference,
  which sets the floor of what can be resolved.
- For the collar-integration comparison, a scheme is "as accurate as the in-stage
  collar" when its in-window error is within a factor of two of the larger of the
  in-stage collar's own error and the measured reference uncertainty, at the same
  `rtol`, for every declared observable. A result
  below the reference uncertainty is unresolved, not a measured improvement.
- Production-strategy selection uses predeclared physical error budgets for every
  observable and sweeps each strategy's tolerance independently. Equal `rtol` is an
  attribution experiment, not a matched-accuracy timing comparison. Report reference
  uncertainty, spectral/time resolution, response-memory and monitoring errors
  separately. The alarm threshold alone does not establish the observable budget.

**Parity.** Every new kernel quantity agrees across `numpy_ref`, `numba_cpu` and the
native CPU build within the summation-bound tolerances of [testing.md](../../testing.md).

**Benchmarks.** Calls, steps, order histograms and the distribution of
$h\alpha_{\max}/(2r)$ over accepted steps, realised grid sizes, wall time, transforms,
peak resident/working memory and monitor overhead, on ten-step windows and short
fibres in tests. Count check-induced steps and distinguish startup, steady march,
setup and requested output. Do not extrapolate a ten-step timing to a whole run.
Full-length runs only in the campaign, as the maintainers authorise them (Q2), timed on
an otherwise idle machine.

### 8. Documentation

- **[Absorbing boundaries](../../../guide/numerics/boundaries.md)** gains:
  - the measured choice between retaining radiation and absorbing it, including the
    supported monitored mode and why an alarm is not an escaped-energy measurement;
  - the floor law and the protected walk-off;
  - geometry guidance (when to widen the collar);
  - the diagnostics and what to do when they fire;
  - the escaped-field record with its caveats;
  - the implicit scheme, if built.
- **[Adams](../../../guide/numerics/adams.md#stability-and-the-temporal-collar):** the
  stability section rewritten around the measured choice.
- **[Grids](../../../guide/setup/grids.md#the-active-band-and-absorbing-collars):** the
  collar width and its slack, fixed asymmetric placement, prescribed frame choice,
  guard margins and the distinction between requested, analysis and realised windows.
- **The statistics catalogue and output pages:** the new statistics and the record.
- **A focused tutorial, "Is my time window big enough?":** the diagnostics, the
  wide-window cost/accuracy comparison, frame/placement choices and a boundary-alarm
  failure, on a short Raman-soliton fibre; the escaped spectrum if phase 3 qualifies it.
- **The [differences from Luna](../../../guide/workflows/luna.md) page,** if a default
  changes.

### 9. Compatibility and impact

**Defaults** change only through the maintainers' decisions on phase 0's
recommendations:

- **No temporal absorber by default** is a candidate, not an adopted change. It needs
  a measured accuracy/resource advantage and a qualified boundary-error policy
  (R6/R7), with the spectral collar retained. If no policy qualifies broadly enough,
  retain the current default and document admitted opt-in choices. No automatic
  scheme switching is part of this proposal.
- **A strength or width default** changes every run whose field reaches the collar,
  and, through the floor, the step count of every multistep run.
- **Validated results that held field in the collar** must be re-run and compared: the
  gallery examples and the qualification references, listed in phase 0's report.
  Breaking changes are allowed before release, without aliases.
- **`absorber_scheme`** keeps its two values; `"implicit"` is added only after phase 4.

**Schemas:**

- the output schema for statistics, guard state and coordinate/check metadata
  (phase 2), and the record (phase 3);
- the kernel `SCHEMA_VERSION` for the record (phase 3) and the implicit solve
  (phase 4), and in phase 2 when guard data, check scheduling or the committed-endpoint
  hook changes the seam or protected driver surface (§6, §11). Each bump takes the
  next free version at landing.

The resume hash covers the new settings, actual rates, surface and metric definitions,
record scheme, guard geometry/thresholds/cadence, coordinate origin/frame and diagnostic
quadrature policy. Resume also restores accumulated state; a matching hash alone
does not reconstruct it.

### 10. Alternatives considered

| alternative | disposition |
|---|---|
| Perfectly matched layer [3, 4] | Reflectionless only when untruncated. It loses the exact linear propagator with full $\beta(\omega)$, a fixed stretch can *amplify* outgoing content where the walk-off direction is not aligned with the sign of the frequency offset, and implicitness only moves its stiffness (§4.4). |
| Transparent boundary conditions [2] | The constructions considered here need an exterior dispersion model beyond a real-axis table and an explicit boundary treatment replacing the periodic discretisation (§4.4). Approximating their nonlocality in z introduces another error budget. |
| Optimised absorbing-potential shapes [1] | They minimise reflection, often with unbounded peaks (Manolopoulos's double pole, [1, eqs 216–220]); that worsens the floor. At fixed width and prescribed steady-drift attenuation, changing a bounded complement profile can lower the peak by at most two (§4.3); this is not a universal bound on reflection or total error. The plateau profile tests that bound. |
| A real part in the collar, after Jolicard–Humbert and Hussain–Roberts [1, eq. 175] | A phase modulation ahead of the damping that speeds slow walkers, improving absorption at low $\Omega$. It adds imaginary-axis eigenvalues, so does not escape the floor, and it alters physics adjacent to the window. Not pursued; phase 0's reflection measurement says whether slow walkers need it. |
| Phase-space filter [5] as the default | Not a term in the equation, so not stiff, and its error bound depends on the buffer and free-flow assumptions, with a required interval cap (§4.9). But it assumes a linear buffer, leaves slow content ambiguous, and as a discrete operation would restart the multistep history. Not prototyped here; a roadmap item since 2026-10-24 (Q13). |
| A Strang split | SEP 14's unchanged-history implementation lost order with a nonlinear source; repairing multistep history and the projected subflow is outside this study. |
| IMEX Runge–Kutta (ARK) [8] | Removes RK's floor, which is about 38 steps and rarely binds. ARK4(3)6L is order four with six stages (DOPRI5 has seven stages and six new evaluations with FSAL), can suffer stiff order reduction despite the implicit component's stage order two, and needs a solve per stage [8, pp. 25, 28]. |
| Multirate inner sub-stepping of the collar | Studied in SEP 15's campaign; it addressed the limit but did not make the fastest candidate, and the source is frozen across inner steps. |
| Automatic switch to the split when the occupancy is low | A conditional default; solprop prefers robust defaults. The end-of-run message may *suggest* the split when the occupancy stayed negligible (Q9). |
| Strength from the group delay predicted across the frequency grid | The grid is always wide and its edges carry only shot noise, so the prediction warns or over-absorbs on almost every run; and a nonlinear run's populated frequencies cannot be known in advance. The protected walk-off is taken from the requested window instead (§6, Q3), and the diagnostics of §4.6 measure what was absorbed. |
| A wider window without temporal absorption | A production candidate in §4.10, with independent guard monitoring (§4.12), measured against the separately converged reference. R6/R7 decide whether it supports an opt-in mode or a default. |
| Fixed asymmetric placement and a prescribed frame | In the phase-0 comparison (§4.11), including fastest-light and midpoint frames. Qualified existing-setting support and guidance belong to phase 2; physical packet separation still sets a lower width limit. |
| Adaptive recentering or expansion | Deferred: history, response memory, noise and coordinates need consistent transport. A general offset-grid API and independently asymmetric absorber ramps also remain follow-up work (Q8). |
| The removal record as the escaped-field record | Simpler and needs no buffer, but it inherits the collar's integration error, which under the in-stage term is large. Kept as the fallback; the surface record of §4.7 is preferred only if its commutator-tail, exterior-model and quadrature errors pass R5. The absorber independence of the local tSURFF setting [6] is not automatic for a projected UPPE. |

### 11. Phases

**Phase 0: the campaign (no package change).** Two parts, each with its own report
in Part II. Phase 0a gates phases 1 and 2; phase 0b gates phase 3 only, and may run
after phase 2 has started.

- The harness and raw results are kept with the project's development records, as
  SEP 30 requires; cases that stay useful, such as the wide-window references and the
  retained-window comparison, become suite cases in phase 2. Accuracy runs use
  `numpy_ref` and `numba_cpu`; cost comparisons are timed on `numba_cpu` and the native
  CPU build, whose per-step ratios differ, on an otherwise idle machine, and record the solprop
  revision, source equation, gates and backend capabilities.
- SEP 27's held-current source is integrated and qualified, and SEP 28's free-space
  changes are in, so plasma and free-space cost comparisons use the current tree on
  both sides (§3.1).

**Phase 0a: windows, monitors and collar geometry.**

- The wide-window references of §4.8 for each case of §7.
- A production comparison at matched observable errors (§4.10): the present window
  with no temporal absorption, wider windows, and the in-stage and per-step collars.
  Keep spectral absorption fixed and compare the same requested outputs. Sweep each
  strategy's tolerance independently; include realised coarse/fine sizes, transforms,
  memory, rejected attempts and timing scatter, not only calls or equal-`rtol` rows.
- Include SEP 24's implemented low-storage RK4 march where supported, alongside
  adaptive RK and both multistep formulations. Refine and calibrate its fixed
  schedules against the same observables, including monitor-imposed check spacing;
  do not infer accuracy from `rtol` or assume its deferred adaptive pair exists.
  Use measured backend memory ledgers, including checkpoint and monitor storage.
- Fixed asymmetric placement and prescribed midpoint/fastest-light frames (§4.11),
  with coordinate/phase alignment (reusing SEP 31's frame records) and source-history
  costs measured. Distinguish settings already expressible from prototypes needing an
  implementation decision.
- Guard-monitor prototypes (§4.12) on committed endpoints and sufficiently frequent
  interior checks, including deliberate crossings between endpoints, spreading with
  zero central drift, nonlinear walkers, weak bands and noise. Measure the relation
  between thresholds and wide-reference observable errors, missed violations, false
  alarms and the extra transforms/steps required. Sparse saved planes are insufficient.
- The floor law: steps, order histograms and $h\alpha_{\max}/(2r)$ distributions.
- A geometry sweep with the existing in-stage scheme: collar fraction 5, 10 and 20 %;
  strength at constant protected walk-off and at today's $AN$, counting one edge and
  both (Q17); the complement and plateau profiles.
- The reflection of slow walkers against the criterion of §4.2, over $\Omega w$ and
  $AN$ (§4.8); R1 depends on it.
- At the best geometry, the in-stage scheme against the split, except on a general
  modal guide while the split scheme's native disagreement there is open (§3).
- An attribution of the split's in-window error: linear/nonlinear splitting, hard
  projection and the fine/coarse collar discretisation difference.
- The transfer of the geometry results to one radial case (R4).
- Passivity of the temporal collar in each route's actual metric (§4.6, Q15).
- The energy diagnostics prototyped on committed endpoints: occupancy and removal
  rates, the split's discrete energy difference, their trapezoid against exponential
  weights or refinement over three step levels, and the extra transforms they cost;
  then the thresholds at which occupancy and removal correlate with wide-reference
  errors (R3).
- **Exit:** a report with recommendations R1–R4, R6 and R7 below.

**Phase 0b: the escaped-field records.**

- The flux and removal records emulated from both narrow absorbed runs and the
  wide references, with enough accepted-endpoint data to converge their quadrature.
  References alone cannot measure absorber feedback. Measure commutator tails for
  each dispersion, projection and spectral collar, and test the initial-field and
  variable-flow terms (Q12).
- **Exit:** a report with recommendation R5.

The recommendations:

- **R1:** geometry and strength defaults;
- **R2:** whether the floor still binds a case that matters after R1 and the R6/R7
  retained-window comparison. Phase 1 requires a case where temporal absorption
  remains necessary or competitive within stated memory/runtime limits, and where
  removing its cap could improve measured total cost. A cap on an avoidable collar
  alone does not qualify;
- **R3:** diagnostic thresholds, and the energy diagnostics' route metrics, measured
  passivity decision (Q15), committed-endpoint contract and quadrature error budget;
- **R4:** whether the spatial collars transfer.
- **R5:** a qualified definition, error budget, supported-route matrix and measured
  memory/transform cost for the escaped record. R5 is a gate before phase 3.
- **R6:** the measured accuracy/runtime/memory tradeoff for each production boundary
  strategy, with and without placement/frame changes, and a recommendation on
  retaining the current default, admitting a monitored opt-in, or adopting a
  monitored default without temporal absorption. State counterexamples and resource
  limits; no single speed-up or unrun estimate settles the default.
- **R7:** guard geometry, metric, absolute/relative thresholds, noise/band policy,
  sampling coverage, supported routes/plane modes, warning/error behaviour and
  checkpoint contract. Qualify tested accuracy budgets and report failures openly.
  R7 gates a production monitor independently of the escaped-field reconstruction;
  if it fails, R6 cannot recommend the monitored default. Any adopted default or
  new frame/placement setting requires a recorded maintainer decision before phase 2.

**Phase 1: the implicit-collar study (prototypes only, after R2/R6/R7).** (Not built:
D4, 2026-10-25.)

- NumPy prototypes in the campaign harness, for both multistep formulations, with the collar
  implicit in predictor and corrector:
  - the storage rule, the collar part of the stored history at the corrected state,
    and the hard step cap at $2r_{\rm AM}/\alpha_{\max}$ (§4.5);
  - the AM collar at the corrector's order;
  - the trapezoidal collar;
  - a backward-Euler collar (L-stable, order one);
  - a modified-CNAB collar (decay one third, order two);
  - an SBDF3/4 march as a reference.
- The projected pointwise solve and CG; GMRES or a damping lab-frame weight in the
  exponential formulation.
- Each reported with the items of §7: coupled stability, damping per step, absorption
  per crossing against the wide window, and whether the error estimate sees the collar.
- Measured on the witnesses and the phase 0 cases against the wide window.
- **Exit criterion:** at the matched physical-error targets of §7, improve wall time
  beyond timing scatter over both existing collar schemes and every admissible
  retained-window candidate on each R2 case. Report calls, transforms, Krylov
  iterations, monitoring cost and memory on the same propagation windows. An
  alternative may be excluded only by a stated accuracy, supported-scope or measured
  resource limit, not simply because it uses more samples. On other admitted cases,
  regress neither calls nor wall time by more than 10 % against the best existing
  collar scheme at matched accuracy. Source-call savings alone do not pass. If the
  criterion fails, build no implicit scheme and report why; phase 2 may still proceed.

**Phase 2: qualified boundary strategies, diagnostics and keywords.**

- The keyword switches of §5 on every `prop_*` interface: `absorber=False` and
  `absorber_scheme=`, with conflicting settings refused by name.
- As decided on R1/R3 and R6/R7. Implement the retained-window mode, guards and
  warning/error behaviour if qualified, plus the selected fixed-placement/frame
  support and manual guidance. Adoption of a new default is an explicit decision.
- Absorbed-energy diagnostics require R3's metric, endpoint and quadrature contract;
  the basic guard monitor requires R7 and its shared metric definitions, not a
  successful escaped-field reconstruction. These parts may land separately. Any
  proposed physical-operator change returns to the maintainer under Q15.
- On all three backends in order: `numpy_ref`, `numba_cpu`, then the native CPU
  build, in the same pull request.
- The manual pages and the tutorial.
- The re-run of affected examples.
- The lasting campaign cases promoted to the SEP 30 suite.
- Geometry changes keep the meaning of the temporal collar's flat-region fields that
  SEP 31's remapping and pulse measures read, or update those consumers in the same
  pull request.
- Output and, where guard data or scheduling changes the seam or the endpoint hook
  changes its protected surface, kernel schema bumps.

**Phase 3: the escaped field.** Requires phase 0b's R5 and phase 2's endpoint/ledger contract;
implementation may overlap only after that shared contract is fixed.

- The qualified flux record (or the removal fallback, on Q11/R5) on all three
  backends, with §7's conditional identities and independent error bounds as its gates.
- Kernel and output schema bumps.

**Phase 4: the implicit collar.** Only if phase 1 passes. (Not built: D4, 2026-10-25.)

- On all three backends; the order choice moved to the implicit intervals; the
  replayed-step and `dz_min` refusal at $2r_{\rm AM}/\alpha_{\max}$ in both formulations
  (§4.5); a kernel schema bump.
- The Adams manual section rewritten.

**Closing.** The full gate, the slow and coverage lanes, and the native-selected slow lane,
once. Device qualification follows later, under the standing policy.

### 12. Open questions

1. **Q1: the reference.** Resolved 2026-10-16 (John Travers): the wide-window reference of
   §4.8 replaces the converged in-stage run as the acceptance reference for absorber
   work, and phase 0 re-measures ADR-025's ladder against it.
2. **Q2: full-length runs.** Resolved 2026-10-16 (John Travers):
   - the canonical and RDW capillaries at full length, as for SEP 16 Q9;
   - the reduced CtC capillary in full;
   - the Raman-soliton case shortened, with a narrower window, so the soliton reaches the
     collar early;
   - no full CtC run.
3. **Q3: how strength is parameterised.** Should strength follow the run length
   (`boundary_N`, today), a protected walk-off (`walkoff`), or remain as now with
   `walkoff` an option?
   - **The rejected alternative.** A protected walk-off is not to be predicted from the
     group delay across the frequency grid. That was considered and rejected: the grid
     is always wide, its edges carry only shot noise, so such a prediction warns or
     over-absorbs on almost every run, and the frequencies a nonlinear run will populate
     cannot be known in advance. This is why `warn_walkoff` takes its span from the
     caller.
   - **What is proposed instead.** $\Delta\tau_p = T_{\mathrm{req}}/2$ uses only the
     requested window, and on tight grids reproduces today's default, counting one
     edge (Q17); counting both, as `warn_walkoff` does, it gives half of today's
     strength.
   - **Its assumption.** It shares today's assumption of steady drift over the run. Walk-off
     that starts late, after fission or once a soliton begins to decelerate, crosses the
     collar faster than the mean and is absorbed less; phase 0 tests this on those cases.
     The measured diagnostics of §4.6, not a prediction, catch the runs where it fails.
   - **The trade.** A walk-off default ends the slack effect of §4.3, but changes every
     run's absorber. Slices keep their parent's strength with `boundary_length` either
     way.
   - **Status.** Deferred to phase 0a/R1 and the maintainer decision that follows it,
     before phase 2. John Travers supports the walk-off default subject to R1 and the
     test of the steady-drift assumption.
4. **Q4: the collar's placement and width.** Resolved in principle 2026-10-16 (John
   Travers):
   - The temporal collar is added *outside* the requested window: the flat region always
     covers $\pm T_{\mathrm{req}}/2$, and the grid grows to fit. Today the collar may
     eat into the requested window.
   - The growth need not be a power of two. The grids already offer
     `fft_size="smooth"`, the next even 2-3-5-smooth size.
   - Phase 0 chooses the fraction (5, 10 or 20 %) from the cost in $n_t$ against the
     steps saved, under both `fft_size` settings. It also says whether `"smooth"`
     should become the default.
   - A collar that absorbs nothing (`temporal=False`, SEP 22) adds no width: the grid
     stays at the requested window (Q16).
5. **Q5: where the absorbed energy is accumulated.** Resolved 2026-10-16 (John Travers):
   per accepted step, reported at the saved planes (§4.6, §6). Review found that the
   existing `quadrature_error` hook has different state timing on adaptive and replay
   paths, so a consistent committed-endpoint hook/data contract is required. The
   driver's per-step statistics call is already such a point on every march (§4.6).
6. **Q6: the escaped record under the in-stage multistep collar.** Resolved 2026-10-16
   (John Travers): accepted-step accumulation, with trapezoidal quadrature as the initial candidate.
   Review qualification: forward local linear propagation, endpoint storage and an
   independent quadrature-error test are required (§4.7).
   - The complete removal complement has a linear identity to rounding under the split.
   - The surface identity is conditional and includes the initial exterior field.
   - In-stage quadrature is tested to its own convergence bound, not the march's tolerance.
7. **Q7: the projection in the implicit solve.** Resolved 2026-10-16 (John Travers):
   phase 1 prototypes both the pointwise division followed by $P$ and conjugate
   gradients, with CG as the reference. The pointwise solve is adopted if its error stays
   below the integration error on every case.
8. **Q8: the time window.** The 2026-10-16 decision placed offset or walk-off-centred
   windows outside this SEP. Revised 2026-10-20 at John Travers's request: the wider
   production window without temporal absorption, fixed asymmetric launch placement
   and prescribed frame comparisons are now in phase 0 (§§4.10–4.12). R6/R7 select
   qualified setting support and guidance for phase 2. A general offset-grid API,
   independently asymmetric absorber ramps, solution-dependent recentering and grid
   expansion remain on the roadmap. No new frame or boundary default is preselected.
9. **Q9: the end-of-run message.** Resolved 2026-10-16 (John Travers):
   - A warning when the physics reached the absorber: occupancy or absorbed energy above
     the thresholds phase 0 sets.
   - An INFO-level note, not a warning, when the collar's stability limit bound a
     substantial share of the accepted steps *and* the occupancy stayed negligible. It
     reports the measured cap occupancy and suggests benchmarking the per-step collar.
     An unrun alternative does not support a quantitative claim of calls or time saved.
   - Revision 2026-10-20: suggestions also include a qualified wider window without
     temporal absorption and better placement/frame. The monitored mode's in-run
     warning/error policy (§4.12, R7) is independent of this end-of-run summary; it
     cannot wait until the final plane to detect a crossing.
   - Silence otherwise.
   - No automatic change of scheme.
10. **Q10: spatial collars.** Resolved 2026-10-16 (John Travers): phase 0 measures, on one
    radial case, whether the geometry results transfer (R4). Any change to the
    transverse collars goes into a follow-up SEP: SEP 22 for the k⊥ collar and the
    length decoupling (Q16); the real-space collars' geometry after R4.

11. **Q11: the form of the escaped-field record.** Resolved 2026-10-16 (John Travers): the
    surface-flux record of §4.7. The removal record is the fallback if phase 0b finds the
    flux buffer unaffordable or the reconstruction insufficiently accurate. Neither is
    built without R5; failed qualification is reported to the maintainer.
12. **Q12: the flux buffer.** How wide must the gap between the surfaces and the collar
    be to reduce the commutator's tails below the error budget, including the hard band
    projection and spectral collar as well as tabulated/modal dispersion? There may be
    no affordable gap. Measured in phase 0b and resolved by R5.
13. **Q13: a phase-space filter prototype.** Resolved 2026-10-16 (John Travers) as an
    exploratory phase 0 measurement on the Runge–Kutta path. Revised 2026-10-24 (John
    Travers): removed from this SEP to keep the campaign to what can change a default.
    §4.9 keeps the analysis, and a roadmap item keeps the prototype.
14. **Q14: the directional occupancy.** Resolved 2026-10-16 (John Travers) as an
    offline phase 0 evaluation. Revised 2026-10-24 (John Travers): removed from this SEP
    with Q13 and kept as a roadmap item.

15. **Q15: record and energy qualification (AI review).** Deferred to phase 0a/R3
    (specify each route's metric, measure collar passivity in it, choose the committed
    endpoint contract and qualify diagnostic quadrature) and phase 0b/R5 (qualify the
    surface reconstruction). On spatial routes the metric is SEP 34's admittance, and
    SEP 34 §4.2 already requires the quadratic-form change to be measured.
    Any change to the physical collar operator or narrowing of production scope needs
    a maintainer decision before implementation. This review does not select a new
    physics default or claim that the proposed flux record has already passed.
16. **Q16: the hand-off from SEP 22 (AI review, 2026-10-18; reconciled 2026-10-20).**
    [SEP 22](sep-0022-memory-lean-spatial-execution.md) is Final. Its delivered
    contracts and the remaining work here are:
    - **A spectral-only collar compiles no temporal pass (delivered).** Before
      SEP 22, an all-zero temporal rate still incurred a second source transform
      pipeline under the in-stage scheme, or the temporal pass and loss of
      first-same-as-last recycling under the split. SEP 22 §3 measured that cost on
      2026-10-18: a factor of 2.0–2.3 per step on `numba_cpu` and 1.6 on Kokkos CPU
      under the in-stage scheme, and 1.2–1.5 under the split. Translation
      (`translate/masks.py`, `temporal_rates`) returns no temporal rates for an all-zero
      collar, so the kernels compile no pass under either scheme. Handed here on
      2026-10-16 and taken back by SEP 22 on 2026-10-18, when this SEP was deferred.
    - **The keyword (delivered).** `absorber(grid, …, temporal=False)` retains the
      spectral collar and hard band mask with zero temporal rates. Its scheme setting
      is inert when no temporal pass exists; restrictions caused only by that pass
      must not apply. `absorber=False` on the keyword interfaces, both soft collars
      off, stays this SEP's (§5). Neither switch cancels a response's source gate.
    - **The grid must not grow for it.** Q4 places the collar outside the requested
      window and grows the grid to fit. A collar built with `temporal=False`, or whose
      rates are all zero, absorbs nothing and must add no width: the grid stays at the
      requested window, as it does today. Today, though, such a grid's flat region is
      still shrunk below $\pm T_{\mathrm{req}}/2$ on a tight grid, because the collar
      geometry ignores the rates (`collar_fraction` must lie in (0, 0.5), and the flat
      region is $\min(T_{\mathrm{req}}/2,\,(1-2f)T/2)$); phase 2 removes that
      shrinkage with Q4. Otherwise a
      spectral-only run on a grid matched to a measurement would gain 12–20 % in
      samples, memory and transform time for nothing.
    - **One effective length (delivered; R4 still measures its suitability).** SEP 22's
      `Absorber.collar_length` is the length that sets the temporal collar's peak,
      $A/\ell_c$, whichever of `boundary_N`, `boundary_length` or `walkoff` sets it,
      and the spatial collars read its `in_stage_length` fallback in place of the
      spectral ceiling's `reference_length`. `custom_absorber` preserves that strength
      when adding a spectral rate. Phase 2 maps any new temporal parameterisation
      into this existing contract. R4 asks whether spatial strength should follow a
      walk-off-derived temporal strength; changing or separating it needs an explicit
      design decision, not an assumed future change in the completed SEP 22.
    - **Attribution.** The spectral ledger $U_\omega$ counts the whole `spectral_rate`.
      SEP 22's `custom_absorber` sums a user rate into it, so the collar's share is not
      separable unless `Absorber` keeps the two apart. That separation was deferred
      (SEP 22 Q11/Q12); this SEP must not report it as already available.
17. **Q17: which edges count (AI review, 2026-10-18).** §4.2's $\Gamma$ is one edge's
    attenuation, and the floor tables, §6's strength formula and Q3's figures count one
    edge; `warn_walkoff` integrates the rate over both edges and so requires
    $2\Gamma \ge \Gamma_{\min}$. A wrapping packet crosses both, so the one-edge criterion is
    conservative by two in the exponent. The convention is chosen before R1:
    counting one edge, $\Delta\tau_p = T_{\mathrm{req}}/2$ reproduces today's default
    strength; counting both, it halves it ($AN$ about 100–135 on the tight grids of §6
    and 25 on the full CtC grid), and the floors of §4.3 halve with it. Deferred to
    phase 0a/R1: its wide-window reference and geometry sweep measure which
    attenuation the physics needs.
18. **Q18: review recommendations (AI review, 2026-10-18), for acceptance.**
    - **An early keyword phase.** `absorber=False` and `absorber_scheme=` on the keyword
      interfaces depend on no phase-0 result, so a phase 2a was recommended. Declined
      2026-10-24 (John Travers): the keywords land with phase 2 (§11). The
      spectral-only collar and its compiled-away pass do not wait on them: SEP 22's
      phase 1 delivered them (Q16).
    - **`walkoff` beside `boundary_N`.** §5 rejects "conflicting" `walkoff` and
      `boundary_N`, yet `boundary_N` keeps setting the spectral $\ell$ when `walkoff`
      sets the temporal strength, so no conflict is defined; state which combinations
      are refused. A slice built afresh with `walkoff` needs its parent's
      $L_{\rm ref}$, which no keyword carries; `boundary_length` is today's answer and
      the manual says so (§6). Deferred to phase 0a, which specifies the refused
      combinations with R1; phase 2 implements them.
19. **Q19: production strategy and default (2026-10-20).** Deferred to phase 0a/R6:
    which retained windows, placements and prescribed frames are cheaper at matched
    physical accuracy, and in which cases does absorption remain necessary or more
    efficient? Report memory limits, monitoring costs, counterexamples and uncertain
    comparisons. Recommend a default only after R7 qualifies its failure policy and
    the maintainer has reviewed the evidence. Compare with the cheapest admissible
    production window, not the cost of the most refined reference.
20. **Q20: guard reliability and cost (2026-10-20).** Deferred to phase 0a/R7: choose
    route metrics, guard widths, bands/noise treatment, absolute and relative warning
    and stop thresholds, and a check cadence that resolves crossing and spreading.
    Establish where dense checks avoid extra source calls and where check planes
    are required. Declare the tested scope and any missed accuracy-budget violations;
    neither endpoint occupancy nor an energy fraction alone certifies containment.
21. **Q21: integration and benchmark baseline (2026-10-20).** Satisfied 2026-10-24:
    SEP 27 and SEP 28 are integrated, so the campaign measures the qualified source and
    the optimised free-space path directly, and no earlier baseline needs remeasuring
    (§3.1). Native work lands with the Python backends in the same repository.
22. **Q22: accumulators across a relaunch (2026-10-24).** SEP 31 relaunches a saved
    plane as a new run with an inherited origin, and its transport report keeps a
    sequential energy ledger. Should a child run's temporal removal and guard history
    start at zero, or should the chain ledger carry the parent's? Proposed: start at
    zero, record the parent's totals as a chain-ledger item, and decide in phase 2
    alongside R3's output contract.

### 13. References

The drafting review checked references 1–8 against the published papers on 2026-10-16:
authors, title, venue, volume, pages and DOI. The Adams–Moulton intervals of §4.5 were
recomputed for this SEP by root locus; reference 9 is the textbook treatment of
multistep stability. Results that reference 1 reviews (Manolopoulos's transmission-free
potential, reactant–product decoupling, the Jolicard–Humbert and Hussain–Roberts real
parts) are cited through it.

1. J. G. Muga, J. P. Palao, B. Navarro and I. L. Egusquiza, "Complex absorbing
   potentials," *Physics Reports* **395**, 357–426 (2004).
   doi:10.1016/j.physrep.2004.03.002
2. X. Antoine, A. Arnold, C. Besse, M. Ehrhardt and A. Schädle, "A review of transparent
   and artificial boundary conditions techniques for linear and nonlinear Schrödinger
   equations," *Communications in Computational Physics* **4**, 729–796 (2008).
   doi:10.4208/cicp.2008.v4.p729
3. C. Zheng, "A perfectly matched layer approach to the nonlinear Schrödinger wave
   equations," *Journal of Computational Physics* **227**, 537–556 (2007).
   doi:10.1016/j.jcp.2007.08.004
4. A. Nissen and G. Kreiss, "An optimized perfectly matched layer for the Schrödinger
   equation," *Communications in Computational Physics* **9**, 147–179 (2011).
   doi:10.4208/cicp.010909.010410a
5. A. Soffer and C. Stucchio, "Open boundaries for the nonlinear Schrödinger equation,"
   *Journal of Computational Physics* **225**, 1218–1232 (2007).
   doi:10.1016/j.jcp.2007.01.020
6. L. Tao and A. Scrinzi, "Photo-electron momentum spectra from minimal volumes: the
   time-dependent surface flux method," *New Journal of Physics* **14**, 013021 (2012).
   doi:10.1088/1367-2630/14/1/013021
7. U. M. Ascher, S. J. Ruuth and B. T. R. Wetton, "Implicit-explicit methods for
   time-dependent partial differential equations," *SIAM Journal on Numerical
   Analysis* **32**, 797–823 (1995). doi:10.1137/0732037
8. C. A. Kennedy and M. H. Carpenter, "Additive Runge–Kutta schemes for
   convection–diffusion–reaction equations," *Applied Numerical Mathematics* **44**,
   139–181 (2003). doi:10.1016/S0168-9274(02)00138-1
9. E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
   Differential-Algebraic Problems*, 2nd ed. (Springer, 1996).

## Part II — Implementation record

### Phase 0a — decisions and interim findings

Recorded as they are made; the phase 0a report (§8 of SEP 0) follows at the end of the
phase. The campaign harness and its raw results are kept with the project's
development records (§11).

**D1 — smooth transform sizes (John Travers, 2026-10-24).** Phase 0a measures the cost
per sample of 2-3-5-smooth transform sizes against powers of two on `numba_cpu` and the
native CPU build. Phase 2 then makes `fft_size="smooth"` the default together with Q4's
collar outside the requested window: with smooth sizes the realised window is close to
the request, so a collar inside it would leave only 90 % of the requested window flat.
Tests that compare against Luna's stored runs pin `fft_size="pow2"`, since Luna's grids
are powers of two.

**D2 — the CtC cases (John Travers, 2026-10-24).** The reduced CtC case of §4.3 and §7
ran a 400 fs pump on a 1 ps request, realised as ±569 fs with its temporal collar from
±500 fs, where the pump still had 1.3 % of its peak intensity. Clipping the launched
pump seeded stimulated Raman scattering in noise-free runs of both solprop and Luna, so
the continuum and the cross-code differences recorded for that case were the boundary's.
On the same 2 ps grid, a launch clipped to the old collar alone brings the cascade back
(Stokes 0.112 of the energy against 8e-11). The reduced case is now the fundamental
alone on a 2 ps window over 0.1 m, and the full case runs 0.3 m on its 4 ps grid; the
full case's Luna comparison is retired. §4.3's row for the reduced case describes the
former case. The 1 ps launch is kept in the campaign as a guard-monitor counterexample:
the pump fills the guards from the start. Seeded with shot noise, the 0.1 m reduced case
forms the Raman cascade (Stokes 0.44–0.48 of the energy over three seeds); noise-free,
it grows only from rounding.

**Interim findings.**

- *Floor law.* Translation admits exactly ⌈n_min⌉ replayed steps and refuses fewer for
  every method in both multistep formulations; with the refusal bypassed, runs at
  0.8 n_min diverge or fail and the error departs from its converged trend within about
  5 % of n_min, so the scalar peak-rate bound is tight on ADR-025's witness.
- *Wrap.* Pure advection through one realised window returns the launch to 1.3e-13 with
  end-plane guard occupancy near 1e-28, so endpoints cannot detect a crossing (§4.12).
- *Flux.* The continuity current of §4.12 balances a region's energy at second order in
  plane spacing, with drift, spreading of either sign of β₂, or both.
- *References.* Wide-window references converge at first order in the window factor. The
  term is Kerr-generated content at the band's low-frequency edge, where the spectral
  collar and mask are sampled on bins that move with the bin spacing: widening the band
  from 1.6 to 3.2 µm cuts the edge energy from 2.5e-3 to 1.4e-6 and the term about
  threefold. It sets a floor of order 1e-5 on the canonical and RDW comparisons, so
  references are extrapolated in 1/M and absorber effects are compared with and without
  temporal absorption at the same window.
- *Step counts.* On the reduced CtC case solprop accepts 3163 steps to Luna's 205 at the
  witness tolerance; phase 0a explains the difference.

### Phase 0a report — 2026-10-25

**Scope.** Phase 0a of §11: windows, monitors and collar geometry; no package change.
Implementer Opus 5.5 (AI) for John Travers, 2026-10-24 to 2026-10-25, on
`feature/sep-0021-temporal-boundaries` (the phase's package commits are the reduced CtC
case of D2 only). The campaign harness, its raw records and run logs are kept with the
project's development records, as §11 requires. Accuracy runs used `numpy_ref` and
`numba_cpu` on the maintainer's Mac and on the university cluster; every timing ran on
one 18-core socket of a dual-socket AVX-512 workstation, one hardware thread per core, serially,
with `numba_cpu` and a native CPU build of this branch (GCC 16, FFTW, OpenBLAS). A
two-core job of another project shared the machine on separate cores.

**Delivered.** No package change beyond D2. Harness pieces: wide references extrapolated
in 1/M; the strategy sweep (§4.10) over five cases, two steppers and five tolerances;
the geometry and strength sweep; slow-walker reflection; placement and frames; guard
prototypes; the split's attribution; the radial transfer; passivity; endpoint energy
diagnostics; transform-size cost; full-length timings.

**Deviations.**

- The reduced CtC case changed (D2): the former 1 ps window clipped the pump, so its
  continuum and cross-code differences were the boundary's. It is kept as a
  guard-monitor counterexample.
- The geometry, reflection and attribution sweeps place the collar outside the requested
  window on smooth sizes (Q4, D1) rather than at today's geometry, because a power of
  two's slack makes every collar fraction alike.
- Low-storage RK4 schedules were not calibrated (§11 asks for them where supported). The
  adaptive pair and both multistep formulations were; RK4 is a gap for phase 2's monitor
  coverage rules.
- The general modal route was excluded, as the precondition of §11 allows; its split
  disagreement stays on the roadmap.

**Findings.**

1. *References.* Wide-window references converge at first order in the window factor
   through a band-edge mesh term (interim findings); extrapolated as `2F(2M) - F(M)`,
   their uncertainty is 4e-6 to 5e-5 per case. Every requested-window comparison is
   floored by that term (canonical 1.9e-5), so absorber effects are measured as collar
   against no temporal absorption at equal M.
2. *Where the field reaches the edge* (soliton witness, RDW, SSFS), the collar's own
   model error (1e-4 to 2e-3) or the wrap of the run without absorption is the floor of
   every requested-window strategy; only wider windows reach 1e-4 or below. *Where it
   stays clear* (canonical, reduced CtC) every strategy is equally accurate.
3. *Geometry and strength.* Strength set by a protected walk-off (Q17: counted on both
   edges) is never less accurate than today's AN = 252 and costs fewer Multistep steps
   (canonical 1120 to 890 at equal error, the floor no longer binding). The plateau
   profile is 10-30x worse on the soliton witness. Width matters only where light enters
   the collar (soliton 10 to 20 %: 2.7e-3 to 1.9e-5).
4. *Reflection of slow walkers* (§4.2). Reflection grows with AN and falls steeply with
   Ωw: at Ωw = 10, 6.4e-3 at AN 252, 1.2e-3 at AN 100, 5.9e-5 at AN 25; at Ωw = 2, 4-16 %
   at every strength. Transmission (wrap) is about 8 % at AN 25 and 0.8 % at AN 50, and
   below the measurement's 5e-3 resolution from AN 100. A collar weak enough not to
   reflect at Ωw of order 10 transmits; slow walkers with Ωw < 10 are reflected at the
   percent level by any collar that does not transmit.
5. *Split against in-stage.* At the recommended geometry the field barely enters the
   collar and the schemes differ by 1e-9 to 2e-7 on RDW and SSFS, against errors of 6e-5
   to 6e-4, with identical steps. Where the field sits in the collar (soliton witness)
   the difference is the Lie splitting: first order in h under Multistep (3.6e-6,
   1.8e-6, 7.4e-7 at 17, 29, 64 steps), 5.6e-7 under RK. The second mask contributes
   1e-8 to 5e-7 and partly cancels the splitting; the coarse against fine sampling of
   the rate contributes 1e-13 to 7e-12.
6. *Placement and frames* (§4.11). On RDW an early launch (0.3 of the window, away from
   the drift) cuts the error 5x at no cost; the fastest-light frame adds nothing (RDW's
   default frame is within 8 fs of it). A prescribed frame off the carrier costs
   Multistep steps (midpoint 5x, fastest 1.25x): the field-grid history rotation of the
   interim findings, at the frame's mismatch. On SSFS placement does not move the error
   (the grid's floor), and the midpoint frame is harmful: light leaves both edges. The
   planning group-delay range overstates the walk-off (SSFS 2.0 ps at a 1e-4 floor
   against a 1.8 ps window), because late radiation has not walked the whole length.
7. *Guards* (§4.12). Launch problems are flagged at z = 0 (the 1 ps CtC launch fills a
   5 % guard at 1.9e-3). The square root of the peak occupancy of a 5 % guard bounds the
   measured field error in every case, from about equal (RDW, 2x) to 25x above (SSFS).
   The peak is not always at the exit (RDW 2x at 1.2 m of 1.5 m, SSFS 2x at 0.13 m), and
   a full-period crossing is invisible at the endpoints (1.3e-13 return, end occupancy
   1e-28), so the check cadence matters. At a 1e-3 threshold a 50 fs packet is caught at
   cadences up to 3.5-7 guard crossing lengths.
8. *Energy diagnostics* (§4.6, R3). Occupancy, removal rate and the split's discrete
   difference were evaluated at every accepted endpoint. On the conservative soliton
   witness the in-stage trapezoid matches the true loss to 1e-3 of it at three step
   levels, and the split's discrete difference is exact to four digits. On the field-grid
   cases exponential weights and the split difference bracket the trapezoid within
   3-7 %. One inverse transform per accepted step is the cost.
   - The square root of the removed fraction bounds the error of the same window without
     absorption in every case but one, tightly on RDW (5.6e-4 against 2.2e-4).
   - The exception is the 1 ps CtC case. It removes 1.7e-3 and its error is 0.93: clipped
     launch content seeds stimulated Raman scattering. Removal cannot bound a nonlinear
     consequence; launch occupancy (8.8e-4 at z = 0, against 2e-5 or less elsewhere) does
     flag it.
9. *Passivity* (Q15). On constant-metric routes (ordinary modal and GNLSE) the collar is
   passive by construction; the measured generalised eigenvalues of the collar's
   quadratic form against the energy lie in [-2e-16, 1].
   - In SEP 34's admittance metric on the radial case the worst gain is -6.6e-11 of the
     peak rate for the temporal collar and -2.9e-7 for the real-space collar. Removal
     rates of actual fields along z are positive for both.
   - A column whose transverse cutoff falls inside the band reaches -6.7e-5. As the
     admittance approaches zero the ratio is unbounded: the collar scatters zero-flux
     content into propagating bins.
10. *The radial collars* (R4), on the suite's spatial-absorber case. The beam reaches a
    0.3 mm radius in 8 cm at 0.9 critical power; temporal absorption was off. Today's
    collar (10 % inside the radius, AN 252) is barely better than a hard wall: an error
    of 1.0e-1 over the requested radius and 2.3e-2 in the core, against 1.0e-1 and 2.8e-2
    for the wall. The error is reflection from the collar, and it reaches the core.
    - A collar outside the requested radius, wider and weaker, is 20x (aperture) and 130x
      (core) more accurate: 30 % at AN 10 gives 4.5e-3 and 1.7e-4, in 46 instead of 241
      RK steps.
    - The temporal trends transfer: width matters most, and weaker is better down to an
      optimum where transmission onto the outer wall takes over.
    - The walk-off formula is within 1.5x of the optimum at wide collars and too strong
      at narrow ones.
11. *Transform sizes* (D1). On `numba_cpu` the per-sample cost of smooth sizes is within
    -20 to +25 % of the nearest power of two, so any smooth size that saves samples is
    cheaper per evaluation: reduced CtC at 1.25x, 9000 samples cost 13.1 ms against
    30.2 ms for 16384.
    - The native build agrees in general, but some smooth sizes are slow. RDW at 4000
      costs 1.8x what 4096 does; 8640 costs 1.7x per sample what 8192 does.
    - The realised smooth size is not monotone in the request: RDW at 1.375x gives 8640
      samples, more than 2x's 8000.
12. *Timings at matched error.* These are full-length runs on that socket.
    - Where the requested window qualifies, the collar, split and no-absorber runs cost
      the same within 5 %. Smooth windows at 1.25-1.5x cost 1.0-1.6x as much, 2x windows
      1.5-2x and 4x windows 2.5-5x.
    - At 1e-4 the requested window does not qualify on RDW (RK) or SSFS. On RDW the
      cheapest run is smooth 1.25x (RK 37.6 s against 60.0 s for 2x on `numba_cpu`;
      native 9.1 s against 13.4 s). SSFS needs the 2x window: on `numba_cpu` 274 s with
      Multistep and 187 s with RK, 2.7x and 3.2x the 1e-3 cost; native 112 s and 50 s,
      2.6x and 3.1x.
    - Peak resident memory was 230-450 MB in every run, so memory binds none of these
      cases.
    - Independently of the boundary, RK is 1.7-6x faster than Multistep on the Raman
      cases (native reduced CtC: 4.4 s against 27 s). A carrier-frame history
      extrapolation addresses this outside this SEP.
13. *Long walk-off* (added 2026-10-25, after the maintainer asked whether any temporal
    absorber beats an alarm and a wider window). Two further cases were run against wide
    references without absorption, adding Luna.jl's released filter (the window `W`
    multiplied into the state once per accepted step) and the spectral error in dB
    where the reference is above -60 and -80 dB of its peak, which weak signals need.
    - The full-length SSFS gallery example (2 m, 3.5 ps) is not one: 3.8e-3 of the exit
      energy lies outside the request but only 2e-7 outside 1.25x. Today's collar,
      inside the requested window, cuts into that light (8e-3; 5 dB at -80 dB). Placing
      the collar outside the request, or a 1.25x window without one, gives 1-2e-4 and
      0.1 dB, at 22 % more samples.
    - A multimode RDW capillary (six HE1m modes, 4.5 m with a 3 m near-vacuum tail,
      400 fs) is. The light that walks off is weak and far-reaching: 2.4e-6 of the exit
      energy lies outside the request, 9e-8 outside 2x and 9e-9 outside 3x. At -80 dB
      today's collar on the requested window is within 2.5 dB (0.08 dB at -60 dB); without
      absorption windows of 1-2x are 10-19 dB wrong and 3x, at three times the samples,
      is needed for 0.04 dB. A weaker collar (walk-off strength, AN of about 30) is worse
      than today's at -80 dB (7.5 dB): weak light crosses it and wraps.
    - Luna.jl's released filter is the least accurate collar in both cases (SSFS 0.08-0.1
      and 13-27 dB at -80 dB) and changes with the stepper and tolerance; the rate
      collars (in-stage, and the split, which is the scheme of Luna.jl's rate-boundary
      pull request) agree with each other.
    - Steps and evaluations were the same for every strategy within a case, so cost
      follows the realised grid size.
    - Field errors near 1e-4 on smooth windows are the band-edge mesh term of
      finding 1, not the boundary; the decibel metrics separate the strategies.
14. *Transform sizes, continued.* An FFTW table of every even 2-3-5-smooth size and
    power of two from 1024 to 262144 (batched real transforms, both planner efforts):
    - The next smooth size costs a median 5-37 % less than the next power of two, but
      more for 5-20 % of requests; the slow sizes are 3-heavy (1458, 1944, 6250, 6750,
      up to 2x per sample). Estimated plans make it worse.
    - On Apple silicon FFTW's NEON codelets are 1.3-2.7x faster than a build without
      them, and the smooth choice is then within 14 % of the cheapest size for 90 % of
      requests. Local Mac builds had been linking an FFTW without NEON; Mac timings in
      this campaign are comparative only, and none of its conclusions rests on them.
    - The native backend's slow sizes of finding 11 were not FFTW: measured planning
      left them unchanged, and the same fine transforms are as fast as their powers of
      two. They came from size thresholds in native kernels, fixed separately
      (LupoLab/solprop pull request 24, open).

**Recommendations.**

- **R1 — geometry and strength defaults** (revised 2026-10-25 after finding 13).
  - Place the collar outside the requested window (Q4) on smooth sizes (D1). This is
    the change that matters: a collar inside the request cuts into light the run keeps
    (full SSFS, finding 13).
  - Keep today's strength (AN = 252) as the default; do not weaken it to the protected
    walk-off. Weak, far-walking light needs it (finding 13), and the transmission limit
    of finding 4 bounds weakening anyway. Its costs are reflection of slow walkers
    (finding 4) and, where light never enters the collar, extra Multistep steps
    (canonical +20 %). The walk-off parameterisation is offered as a setting for runs
    that know their walk-off, with Q17's both-edges count.
  - Keep the complement profile and reject the plateau. Use a 10 % collar per edge;
    the monitor suggests a wider one for launches into the edge.
  - Q3: no change to `boundary_N`'s meaning.
- **R2 — the floor.** At today's strength the floor costs Multistep steps where the
  collar does no work (canonical +20 %, RK unaffected) and about 8 % on the multimode
  RDW case, where the collar is needed; on RDW and SSFS it costs none. Removing the cap
  could therefore save at most 8 % on a case where absorption is necessary, and the
  canonical saving is on an avoidable collar, which §11 excludes. No case qualifies
  phase 1's implicit collar. Recommend recording phase 1 and phase 4 as not built.
- **R3 — diagnostics.**
  - Accumulate occupancy and temporal removal at committed endpoints (the
    `_take_step_stats` sites, §4.6) by trapezoid, with a 10 % quadrature budget. That is
    three times the measured spread, so the bound is defensible without per-run
    refinement.
  - Under the split, record the discrete energy difference instead, which is exact.
  - Warn at the end of a run when the square root of the removed fraction exceeds
    1e-3 (the 1e-3 budget). Warn at launch when collar or guard occupancy exceeds 1e-4.
  - These are field-error thresholds. Weak spectral features are not covered: on the
    multimode RDW case a removed fraction of a few 1e-6 governs the -80 dB spectrum.
    Report the removed fraction itself in the end-of-run summary so a user with a weak
    observable can judge it.
  - On spatial routes report signed rates in SEP 34's metric. They are passive except
    near transverse cutoff, so Q15 needs no change to the collar operator.
- **R4 — spatial collars.** The temporal results transfer, and the radial collar's error
  is larger than the temporal one: a collar outside the requested radius, 20-30 % wide
  at AN 10-50, is 20-130x more accurate and cheaper. They should not take the temporal
  walk-off strength unchanged. Recommend the follow-up SEP of Q10 for the real-space
  collars' geometry, with this case as its witness. Phase 2 must not move spatial
  strength when it changes the temporal default (SEP 22's shared `collar_length`).
- **R6 — production strategies** (revised 2026-10-25 after finding 13). Keep temporal
  absorption on by default, placed outside the requested window, with the guard
  monitor.
  - The collar costs nothing measurable in steps where light reaches it, and it is the
    only affordable way to keep weak spectral features: no-absorber windows needed 3x
    the samples to match it at -80 dB. Near the pulse, a wider window is the safer
    route to accuracy below the collar's own model error, and the cheapest is a smooth
    1.25-2x window, which no default can choose without knowing the budget.
  - Admit the monitored no-absorber mode as an opt-in for wide windows, and make the
    monitor's advice name the smooth window factor.
  - Make an early launch a documented setting, not a default.
  - Adopt no prescribed frame until the Multistep carrier-frame work removes its step
    cost.
  - Counterexamples: the 1 ps CtC launch, which no boundary saves, and slow walkers
    with Ωw < 10.
- **R7 — guard monitor.**
  - Guards of 5 % of the realised window per edge, measured in the energy-density proxy
    of §4.6 relative to the plane's energy.
  - Warn when the square root of the guard occupancy exceeds the run's error budget
    (1e-3 by default); error at 10x the budget.
  - Check at every accepted step on adaptive and multistep marches; a fixed low-storage
    schedule must refuse coverage it cannot give.
  - Always check at launch.
  - The bound held in every case measured, but it is 1-25x conservative, and a single
    nonlinear counterexample shows that occupancy below threshold after launch does not
    guarantee accuracy.
  - Not qualified: noise floors and weak bands, the general modal and spatial metrics
    (not measured), and the low-storage march.

**Decisions (John Travers, 2026-10-25).** The revised recommendations are accepted:

- **D3 — geometry and strength (R1, Q3, Q17).** The temporal collar sits outside the
  requested window, on smooth sizes, 10 % per edge, complement profile, at today's
  strength (AN = 252). A walk-off parameterisation, counted on both edges, is an
  optional setting. `boundary_N` keeps its meaning.
- **D4 — phases 1 and 4 are not built (R2).** No case qualifies the implicit collar.
- **D5 — diagnostics (R3, Q15).** Occupancy and temporal removal accumulate at
  committed endpoints by trapezoid (10 % quadrature budget); the split records its
  exact discrete difference. End-of-run warnings at a removed-fraction square root above
  1e-3 and launch occupancy above 1e-4, and the removed fraction is always reported.
  Spatial routes report signed rates in SEP 34's metric; the collar operator does not
  change.
- **D6 — default strategy (R6).** Temporal absorption stays on by default, with the
  guard monitor; the no-absorber mode is a monitored opt-in; early launch is a
  documented setting; no prescribed frame for now.
- **D7 — guard monitor (R7).** Guards of 5 % per edge in the energy-density proxy; warn
  when the square root of guard occupancy exceeds the error budget (1e-3 by default),
  error at ten times it; checks at every accepted step and at launch; a fixed
  low-storage schedule refuses coverage it cannot give.
- **D8 — the size chooser (Q23).** Choose the cheaper of the next smooth size and the
  next power of two from a measured per-platform cost model, and revisit the divisor
  rule.
- **D9 — spatial collars (R4).** Their geometry goes to the follow-up SEP of Q10; phase
  2 leaves spatial strength unchanged.
- **D10 — accumulators across a relaunch (Q22; maintainer, 2026-10-27).** A run
  relaunched from a saved plane starts its removal and guard totals at zero; the
  parent's totals are an item of the chain's energy ledger.
- **D11 — the divisor rule (Q23, second half; maintainer, 2026-10-27).** The rule that
  the coarse size divides the fine one stays; the non-monotone sizes it causes go to
  the roadmap.
- **D12 — the floor-subtracted density under a collar (maintainer, 2026-10-27).**
  Documented, not corrected: the subtraction reads low by up to the absorbed fraction
  of one photon per bin where the collar took part of the vacuum floor.
- **D13 — the monitor and the shot-noise floor (Q24; maintainer, 2026-10-27).** The
  monitor judges above the floor: light that walks off below the vacuum level does not
  matter. Implemented: the driver hands the monitor the input draw it added to the
  launch; each region's energy less three times the floor's expected share
  (`monitor.FLOOR_MARGIN`; the expectation alone for the removal rate, integrated with
  its sign) is taken as a share of the whole field, so a noiseless run is judged as
  before. The raw values are recorded beside. Measured: the N=3 soliton and pulses of
  1 kW to 10 mW peak no longer warn; a walking pulse well above the floor is still
  stopped; the removed fraction of a walking pulse agrees with the noiseless run's to
  2 %. Light reaching a guard below the floor's equilibrium share of it is not judged,
  even where the collar has depleted the floor there. The floor is taken from the
  launch, so after loss the monitor over-subtracts and is less sensitive; the collar,
  like any loss, also removes vacuum noise the input model never restores. Both go to
  the roadmap item "Vacuum noise under loss" (maintainer, 2026-10-27).
- **D14 — the split scheme on the spatial routes (maintainer, 2026-10-27).** The
  radial, free-space and planar routes apply the default temporal collar by the split
  scheme unless the settings name a scheme or interpolate their planes;
  `SimulationSettings.absorber_scheme` defaults to `None`, the route's choice, which
  the other routes resolve to the in-stage scheme. Evidence: the spatial collar-scheme
  study below. A fixed schedule on a spatial route now keeps its collar (the split is
  unconditionally stable), so the tracking suite's `kerr_beam` cases run the default
  collar on main's builder.
- **D15 — the monitor's cadence and the quiet collar (maintainer, 2026-10-27; amends
  D7's "every accepted step").** Every committed step is checked while a guard or the
  collar holds more than (budget / 10)^2 above the floor; otherwise every eighth step
  or every `length / 256`, whichever comes first, and every saved plane's state.
  While the collar holds less than 1e-12 of the field's energy above the floor, the
  modal stage leaves the in-stage temporal term out (`numpy_ref`, `numba_cpu` and both
  native engines); the split scheme, the low-storage march, the general modal route,
  coupled flows and a real-space collar keep it. The switch happens only below the
  threshold, so neither the Adams history nor the reused Runge-Kutta stage needs a
  restart. GNLSE Raman/shock (one thread, warm): native 4.25 ms per step with the
  collar always on, 3.39 without a collar, 3.37 to 3.43 with both changes; numba 2.81,
  2.27 and 2.23 to 2.41. The monitor's own cost fell from 2.3 s to 0.3 s of a 40 s
  run (1339 checks of 10682 steps). Evidence for the cause: the in-stage term costs
  one forward transform of the fine time grid per stage, and the split scheme one
  extra source evaluation per step (7 against 6, the reused stage lost).
  *Amended 2026-10-28 by ADR-057 ([decisions](../../decisions.md)):
  the quiet collar is opt-in (`BoundaryCheck(quiet_collar=True)`), judged on the whole
  field with the floor included, and no step is longer than the check interval while
  the collar rests.*
- **D16 — the monitor's energy measure on spatial routes (maintainer, 2026-10-27).**
  The quadrature-weighted field norm stays; SEP 34's admittance-weighted flux goes to
  the roadmap. For paraxial beams the admittance is about the index, which the shares
  cancel, and weighting the field in time would hold another half of the state (about
  0.45 GB on a 256 by 256 Cartesian beam) and add a product per check.
- **D17 — phases 0b and 3 are deferred (maintainer, 2026-10-27).** The escaped-field
  record (§4.7, phase 3) and the measurement that qualifies it (phase 0b, R5, Q11, Q12)
  move to the roadmap, and the SEP closes on phase 2. Nothing in phase 2 depends on
  them; the monitor's record (how much left, where and when) and a wider window cover
  what they would add for most runs, and the record would now also have to separate
  the vacuum floor that the collar absorbs (D13, "Vacuum noise under loss").

**Limitations.**

- One radial case, one gas, Kerr only.
- No general modal or planar route.
- Low-storage RK4 not calibrated.
- Guard and energy thresholds come from six cases.
- Reflection was measured at one dispersion (β₂ only).
- One long walk-off case with weak far-walking light; the strength recommendation rests
  on it and on the transmission limit of finding 4.
- Timings are for one socket of one machine.

**Open questions.**

23. **Q23: the smooth size chooser.** The next 2-3-5-smooth size is dearer than the next
    power of two for 5-20 % of requests (finding 14), and the realised size is not
    monotone in the request because the coarse size must divide the fine one. Recommend
    choosing the cheaper of the next smooth size and the next power of two from a small
    measured per-platform cost model, and revisiting the divisor rule. The native slow
    sizes were a separate kernel threshold problem (finding 14). Destination: phase 2,
    with D1.
24. **Q24: the guard monitor and the shot-noise floor (2026-10-27).** SEP 37 put one
    photon per mode into every launch, and the monitor reads that floor as edge
    occupancy and removal. Recommend judging every occupancy and the removed fraction
    above the floor: subtract the expected floor energy of each region, which the noise
    model gives exactly, from the measured energy, and the floor from the launch
    energy, recording the raw values beside them. Light that reaches an edge above the
    vacuum level is still caught.

**Handoff.** The harness, raw records, tables and logs are with the development records
under the SEP's number. Phase 2 starts from the decisions above; the witnesses of §7
(wrap, flux, floor law) are ready to become tests.

### Phase 2 — progress (in progress)

Recorded as slices land; the phase report follows at the end of the phase.

- **Slice 1, keywords.** `absorber=False` and `absorber_scheme=` on every `prop_*`
  interface; `boundaries.no_absorber` builds the hard mask alone. A scheme beside
  `absorber=False`, or a keyword that contradicts a spatial run's settings, is refused.
- **Slice 2, geometry and defaults (D1, D3, D6, D8, Q4, Q16).** Grids widen to
  `trange / (1 - 2f)`, `f = 0.1`, so the collar lies outside the request;
  `collar_placement="inside"` keeps the former geometry for frozen witnesses.
  `fft_size="auto"` is the default: the cheapest 2-3-5-smooth fine size up to the next
  power of two by a cost model fitted to the FFTW tables of finding 14 (each factor of
  3 costs 1.05, each of 5 1.04; within 13-19 % of the best size for 90 % of requests,
  against 1.53 at worst for the next smooth size). The coarse-divides-fine rule is kept
  (Q23's second half stays open). GNLSE and spatial runs get the default collar (the
  maintainer's decision of 2026-10-26); on spatial runs `Absorber.spatial=False` keeps
  the real-space collar off, as it was without an absorber (D9). `absorber=False` builds
  a grid with no collar room (Q16). Output schema 27 (26 went to SEP 37). Short fixed schedules on the
  newly collared routes are refused, as on capillary runs (maintainer's decision). The
  benchmark suite's cases take the new defaults at new revisions; their accuracy
  ceilings are carried over and must be requalified on the new grids.
- **Slice 3, the boundary monitor (D5, D7).** `solprop.monitor`, on by default through
  `SimulationSettings.boundary_check`.
  - **Deviation: no kernel change.** The monitor reads the committed state through the
    driver's existing host observation at the committed-endpoint point of §4.6
    (`_take_step_stats`), on every march and backend, and is computed in NumPy. No
    kernel schema bump; the cost is one inverse transform of the state per committed
    step, which a backend-resident reduction could later remove.
  - **Removal by scheme (D5, 2026-10-27).** Under the in-stage scheme the removed
    fraction is the trapezoid of the removal rate over committed steps: on a walking
    pulse fully absorbed it matches the run's own energy deficit to 5 % in the tests,
    and to 2e-4 relative on a clear-but-dispersing pulse. Under the split scheme each
    step's removal is read exactly from its committed state,
    `P(t) (exp(alpha(t) h) - 1)`, and matches the deficit to 1e-3; the band mask that
    follows the split is not counted.
  - **Deviation: per-plane rows in the execution record (2026-10-27).** §6 sketches
    the occupancies as plane statistics (`run.statistics[...]`). The statistics
    catalogue is chosen by the caller and computed by the backends, while the monitor
    runs on the host whenever the check is on, so its rows stay in its own record:
    `boundary_monitor["planes"]` holds, per saved plane from the launch on, `z`, the
    collar occupancy there, each guard's largest occupancy since the previous plane
    and the removed fraction so far. No output schema change. A resumed run's rows
    start at the resumed plane. The spectral collar's removal (`U_omega`) is not
    recorded.
  - **Limitation (D16):** the energy-density proxy on spatial routes is the
    quadrature-weighted field norm, not SEP 34's admittance-weighted flux.
- **Requalification (2026-10-26).** All 34 cases of the SEP 30 suite on both adaptive
  tracks, `numba_cpu`, one thread: every candidate error was within its carried-over
  ceiling (largest 9.8e-7 against 1e-4; CtC 2.4e-7 against 1e-5), except
  `grin_nonlinear`, whose tight run was refused by a harness that changed settings after
  translation. SEP 38 then replaced the suite and removed reference qualification, so
  the result is a record only; validation continues on SEP 38's tracking cases.
- **Integration with SEPs 37 and 38 and the run summary (2026-10-27).** Output schema
  27. The run-start summary's absorber row says whether the edges are watched. All
  tracking cases build with the default collar except the fixed-step `kerr_beam` cases,
  which run without collars and without collar room. The transform sizes of the
  tracking cases change both ways: 4096 to 5120 points for RDW, 8192 to 10240 for CtC
  and 16384 to 18432 for the 6 ps hydrogen Raman, but 4096 to 3072 for the GNLSE
  soliton, 4096 to 3200 for the argon blueshift and 1024 to 864 for the spatial cases.
  With SEP 37's Runge-Kutta defaults on GNLSE runs (rtol 1e-7) the in-stage collar does
  not cap the step: the soliton and Raman/shock cases take the same accepted steps as
  under the split scheme.
  - **Conflict with default shot noise.** One photon per mode fills the whole window,
    guards included. The N=3 soliton (about 100 W peak) warns twice from the noise
    alone: guard occupancy 2e-6 and a removed fraction of 3e-5, against 2e-19 and 1e-16
    noise-free. A 1 pJ pulse warns at launch and a 10 mW one is stopped. Three coupled
    input-noise tests that launch the seed alone are stopped. Resolved by D13 (Q24).
  - **The floor-subtracted density** assumes the vacuum floor is conserved; the default
    collar absorbs part of it, so the subtraction is biased low by up to the absorbed
    fraction of one photon per bin. The identity test runs without a collar.
- **A/B against main (2026-10-27).** The tracking cases with their defaults (noise
  on), native backend, on 16 cores of one memory domain of an AMD EPYC 7763 node:
  main against this branch before D14, three rounds of two repeats, at each case's
  declared widths; the `kerr_beam` cases were left out (their builders then
  differed). Time ratio, this branch over main, median over rounds (all rounds within
  1 % at one thread unless noted):

  | case | points (coarse) | accepted steps | 1 thread | 16 threads |
  |---|---|---|---|---|
  | GNLSE soliton | 4096 to 3072 | 8065 to 1893 | 0.20 | 0.29 |
  | GRIN nonlinear | 4096 to 3072 | 198 | | 0.62 |
  | argon blueshift, resolved / averaged / envelope | 4096 to 3200 (envelope 2048 to 1728) | unchanged | 0.81 / 0.83 / 0.84 | 0.76 / 0.83 / 0.84 |
  | CtC | 8192 to 10240 | 15104 to 15062 | | 0.86 |
  | hydrogen Raman, averaged | 4096 to 4608 | unchanged | 1.08 | 1.09 |
  | helium RDW, averaged | 4096 to 5120 | 1445 to 1434 | 1.16 | 1.16 |
  | GNLSE Raman/shock | 16384 | 10913 to 10677 | 1.16 | 1.19 |
  | RDW | 4096 to 5120 | 1801 to 1817 | 1.23 | 1.23 |
  | radial filament, 256 | 1024 to 864 | 132 to 265 | 1.32 | 1.11 (0.96 to 1.62) |

  The wider windows cost what their sizes say (RDW, helium RDW, hydrogen Raman); the
  smooth sizes save elsewhere (argon, GRIN). The radial filament's steps were the
  in-stage collar's, which D14 removes. The GNLSE soliton's are the shot noise's: the
  collar absorbs the vacuum noise that otherwise walks through the soliton and drives
  the step size (without noise it takes 691), and the tracking case is noiseless from
  revision 2. On Raman/shock, which main ran without a collar, the in-stage collar
  costs about 9 % per step and the monitor's host transform about 6 % (native backend).
  CtC's steps cost 17.5 ms against 20.4 ms on a grid 25 % larger, which the sizes do
  not explain; not yet investigated. Records with the development records
  (`results/ab/summary-sep21.json`).
- **The collar scheme on the spatial routes (2026-10-27).** The A/B against main
  (above) found the radial filament taking twice the steps (132 to 265) under the
  default collar that phase 2 gave the spatial routes; the in-stage collar caused it.
  On six spatial cases of the tracking registry, built by their own builders, each
  scheme was scored against a no-collar window 2.5 times the request at a tenth of the
  case's `rtol`, on one shared lattice (power-of-two sizes), noise off; the error is the
  relative L2 inside the requested window. Records with the development records
  (`results/spatial_split/table.json`).

  | case (integrator) | reference at the case's `rtol` | in-stage | split | no collar | steps in-stage / split |
  |---|---|---|---|---|---|
  | radial filament, 256 (Multistep) | 5.2e-6 | 3.2e-5 | 3.6e-5 | 5.2e-6 | 265 / 129 |
  | radial filament, 512 (Multistep) | 4.9e-6 | 3.1e-5 | 3.8e-5 | 4.9e-6 | 265 / 130 |
  | radial with a real-space collar (Multistep) | 7.4e-6 | 2.3e-5 | 8.0e-6 | 7.7e-6 | 220 / 48 |
  | planar (Multistep) | 1.6e-6 | 1.5e-5 | 1.6e-6 | 1.6e-6 | 187 / 41 |
  | line (Runge-Kutta) | 0 | 3.7e-7 | 3e-11 | 3e-11 | 41 / 28 |
  | Cartesian filament, 128 (Runge-Kutta) | 8.9e-5 | 1.3e-4 | 1.6e-4 | 9.1e-5 | 61 / 51 |

  The split is as accurate as the in-stage collar or better (on the planar case the
  largest spectral error above -60 dB falls from 2.2e-3 to 4.6e-5 dB), with 1.2 to 4.6
  times fewer steps and 20 to 38 % less wall time on the radial and Cartesian cases
  (warm; the small cases' first runs are dominated by compilation). On the radial
  filaments both schemes leave 3e-5, the collar's own effect. Decision D14.
- **Observed:** on the capability capillary route an adaptive exponential Adams run
  limited by the in-stage collar chooses steps that differ between backends by up to
  2 % at rounding level, with fields equal to 6e-12; the test was pinned to the former
  grid. A controller question near the collar's stability limit, recorded here.

### Phase 2 — report (2026-10-27)

**Built.** The keywords (`absorber=False`, `absorber_scheme=` on every `prop_*`); the
collar outside the requested window on cost-chosen smooth sizes (D1, D3, D8, Q16); the
default collar on the GNLSE and spatial routes (D6, D9), applied by the split scheme
on the spatial routes (D14); the boundary monitor (`solprop.monitor`, on by default),
judging above the shot-noise floor (D13), with a row per saved plane and the split's
exact removal (D5), checked every step only near an edge and leaving a quiet collar
out of the stages (D15); the run summary's edge-watch row; the manual pages, the
tutorial "Is my time window big enough?" and the changelog. Output schema 27; the
kernel schema is unchanged.

**Departures from Part I.**

- No backend-resident reductions and no kernel hook: the monitor reads the committed
  state on the host at the committed-endpoint point of §4.6, every march and backend.
  Its cost is bounded by the cadence of D15 instead.
- The occupancies are rows of the monitor's execution record, not plane statistics
  (§6's sketch): the statistics catalogue is the caller's and the backends'.
- The spatial routes measure with the quadrature-weighted field norm, not SEP 34's
  flux (D16, roadmap).
- The removal of the spectral collar (`U_omega`) is not recorded.
- The quiet collar (D15) is new: Part I did not foresee a switch of the in-stage
  term; it is exact below a stated threshold and bounded above it.

**Measured.** Against main on SEP 38's tracking cases (the A/B above): the wider
windows cost what their sizes say (RDW +23 %, helium RDW +16 %, hydrogen Raman +8 %),
the smooth sizes save elsewhere (argon -16 to -24 %, GRIN -38 %), and the two
regressions it found, the radial filament's doubled steps and GNLSE Raman/shock's
collar and monitor, are removed by D14 and D15. A targeted A/B on the final branch
(the same node type and settings, two rounds) confirms it: the radial filament runs at
0.76 of main's time at one thread and 0.86 at sixteen, argon at 0.84 to 0.91, RDW at
1.12 to 1.27 (its window). The fixed-step Kerr beam takes 1.5 times main's time: the
collar is applied on it again (split, 28 % per step on an Apple M5 Max) and its window is 28 %
larger; the roadmap's "The split collar while quiet" would recover the first half. Its
GNLSE Raman/shock rounds (1.01 to 1.40) shared their nodes with other work and are not
quoted; on one machine, warm, the default costs what no collar does (D15). The spatial
study (D14) and the GNLSE measurements (D15) give the
accuracy evidence: the split scheme is as accurate as the in-stage collar or better
on every spatial case, and the quiet collar leaves a clear run's field unchanged to
within 3e-9 on every backend, far below the threshold's amplitude of 1e-6.

**Not built (Part I's phases).** Phases 1 and 4, the implicit collar (D4); phases 0b
and 3, the escaped field and its qualification (D17, the roadmap's "The escaped field").
A record of the absorbed field will have to separate the vacuum floor, as the monitor
does (D13), since the collar takes floor energy as well as light (the roadmap's "Vacuum
noise under loss"); a relaunched run starts its accumulators at zero (D10).

**Open.** Why CtC's step became cheaper on a larger grid (the A/B above) is not
investigated. The worked examples were re-run on the final branch. The monitor
warned on six that light left their windows, and five were widened until it no longer
did (self-phase modulation 0.7 to 3 ps, Raman self-frequency shift 3.5 to 5 ps,
ionisation blue shift 0.4 to 0.6 ps, the graded-index fibre 1.2 to 1.6 ps, the moving
solid guide's linear runs 0.4 to 0.8 ps). On the sixth, the Raman comb, the collar
removes 6e-6 of the launch energy on every window from 6 to 8 ps, with or without input
noise; that is not investigated, and its committed output predates the monitor.

## Part III — History

- 2026-10-16: Drafted. The same day, John Travers resolved Q1–Q10 with the drafting agent,
  and the walk-off parameterisation was corrected: it does not fix short slices, and must
  not be predicted from the frequency grid.
- 2026-10-16: Revised against references 1–8, which were read in full:
  - §4.2: reflection depends on the offset from the frame's zero-walk-off frequency, not
    the carrier, so slow walkers reflect;
  - §4.4: the PML and transparent-boundary arguments were corrected, and the PML's
    amplification of misaligned outgoing content added;
  - §4.5: the trapezoidal collar is not L-stable and absorbs less as it stiffens; the
    predictor must be implicit too; backward-Euler, modified-CNAB and SBDF references
    were added;
  - §4.6: directional occupancy and surface flux were added as diagnostics;
  - §4.7: the surface-flux record, after Tao and Scrinzi, is now recommended;
  - §4.9: the phase-space filter was added;
  - Q11–Q14 were opened and resolved the same day; Q12 is a phase 0 measurement.

- 2026-10-16: Reviewed by John Travers (human, recorded at his request) and GPT-6 (AI).
  Review status is **Reviewed**; lifecycle status remains **Draft**, pending acceptance.
  The AI review checked all of Part I against the architecture, conventions, absorber
  kernels, accepted-step hooks, SEP process and the relevant passages of references
  1–8. It corrected:
  - the projected split's separate defect and the nonlinear Lie bracket;
  - the trapezoidal factor at unit stiffness, the scope of fixed-step stability claims,
    predictor/estimator requirements and unsupported universal IMEX assertions;
  - the physical energy metric, material/source exchange, projection loss and the
    timing of accepted-endpoint diagnostics;
  - the surface record's initial term, conditional identity, nonlocal tails and
    exterior-model limitation, and the exact split removal complement;
  - stable local record propagation, independent quadrature convergence, coherent
    spectral reconstruction, resume state and cost accounting;
  - reference convergence, launch/noise matching, profile-area strength and the
    requirement to preserve the spectral collar when changing temporal strength.
  R5/Q15 now gate production records and metric decisions; phase 0 retains the
  existing maintainer choices on geometry, exploratory filtering and campaign cases.
  Independent scalar checks reproduced AM orders 3–6 endpoints 6, 3, 90/49 and
  45/38, and the trapezoidal factors at $x=1,10,100$. A 24-point Fourier compression
  witness verified the split defect's $O(h^2)$ scaling and $O(h^3)$ remainder; a
  linear surface witness showed a nonzero reconstruction residual while the complete
  removal identity held at approximately $6\times10^{-16}$ in the split calculation.
  Reference 9 was not consulted directly; its cited stability intervals were derived
  independently as specified in §4.5.
  No full propagation campaign or new absorber implementation was run for this review.
- 2026-10-18: Reviewed by Fable 5.1 (AI), against
  [SEP 25](sep-0025-rounding-level-reproducibility.md) Final (reproducibility
  to the error bound), SEPs 22–24 as drafted and the current tree (kernel schema 28,
  output schema 18). Resolved in the text: the SEP 22 hand-offs, recorded as Q16 (a
  spectral-only collar compiles no temporal pass; its keyword; one effective collar
  length, and R4), with §2, §5 and Q10 pointing to SEP 22; the floor of SEP 24's RK4
  march (§2, §4.3); SEP 25's restatement of the equivalence tests (§5, §6, §7: exact
  accounting, rounding bounds on a replayed schedule, accuracy envelopes for adaptive and
  cadence comparisons); the cross-check with SEP 23's `absorbed_power` (§7); phase 2's
  conditional kernel bump and the versions at landing (§9). The derivations were
  checked independently by Opus 5.5 (AI) against the code; verified at review and
  resolved in the text: `warn_walkoff` integrates both edges, so §4.2's one-edge
  $\Gamma$ is half its count (§4.2, §4.3, Q3; the convention is Q17); the §4.3 floors
  from `PEC_REAL_INTERVAL` are 231, 420, 773 and 1437 against the table's 2-digit
  figures, left as approximations; the AM intervals of §4.5 assume corrected-state
  collar history where PEC stores the predicted-state source, and phase 4 needs a hard
  step cap in both formulations (§4.5, §11); the solve's transform count quantified
  (§4.5); the surface and removal identities hold for $N\equiv0$, or for a linear $N$
  inside $\mathcal U$ (§4.7); the JWKB parameter grows with strength, so the reflection
  sweep spans $AN$ (§4.2, §4.8); the spectral ledger's quadrature takes exponential
  weights (§4.6); class-R bounds named for the removal identity, CG parity and the floor
  law (§7); the `escaped_field` refusal keys on nonzero temporal rates (§5); the early
  keyword phase, the `walkoff`–`boundary_N` conflict and the slice's parent length
  recorded as Q18. No propagation campaign was run for this review.
- 2026-10-18 (later): SEPs 22–24 are taken up first and this SEP waits (John Travers).
  The zero-rate pass and the spectral-only keyword return to SEP 22's phase 1 with
  their measured cost (§2, §5, Q16, Q18); Q16 gains the requirement that a collar with
  `temporal=False` keeps the requested window under Q4.
- 2026-10-20: At John Travers's request, revised Part I to compare wider production
  windows without temporal absorption, fixed asymmetric placement and prescribed
  frames with the collar strategies. Added the cost model, frame-width derivation,
  independent guard monitor, crossing/flux witnesses and matched-accuracy campaign
  (§§4.10–4.12). R6/R7 gate the production policy; R2 and the implicit-study exit now
  include the best admissible retained-window candidate. Updated Q8, added Q19/Q20,
  and reconciled the title, scope, interface, phases, documentation and roadmap.
  The existing review header records reviews of the earlier draft; this substantial
  revision needs review before acceptance. No propagation campaign was run and no
  physics default was selected. Cost factors and the scalar continuity current are
  derived here; reference [5]'s buffer/velocity and exterior-evolution assumptions
  were checked against the authors' paper for the discussion, not qualified for UPPE.
- 2026-10-20 (dependency review): At John Travers's request, checked SEPs 22–24's
  delivered code and implementation records, SEP 24's Final integration record,
  SEP 27's pending implementation and approved T2 correction, and SEP 28's accepted
  design. Added §3.1 and Q21: the established SEP 27 → SEP 28 order, independent
  SEP 21 studies, and source/performance gates on its final campaign conclusions.
  Corrected stale Q16 hand-offs and the low-storage stepper's status; preserved
  SEP 23's rate channels and corrected the spectral-integral cross-check. Specified
  statistics-only monitoring, fixed-schedule coverage, scratch/checkpoint ownership
  and reusable-runner resets. Plasma comparisons preserve the held-current source
  gate and qualify its products independently of the absorber. No numerical campaign
  was run, no default was changed and no frozen Part I of another SEP was edited.
- 2026-10-24: Refreshed by Opus 5.5 (drafting agent) at John Travers's request against
  the integrated tree (kernel schema 35, output schema 23), before review and
  acceptance. §3.1 records SEPs 22–30 as Final and adds SEPs 29, 30, 31 and 34; the
  execution order and Q21 are satisfied. §4.6 takes SEP 34's admittance metric and
  collar passivity rule on spatial routes and names the driver's existing
  committed-endpoint statistics call for the accumulators. Stale references were
  corrected: the stability intervals now live in the package, `collar_field_fraction`
  replaced `spatial_collar_fraction`, the native backend is in this repository, and
  the keyword interfaces' present absorber arguments are stated. At John Travers's
  request phase 0 is split into 0a (windows, monitors, geometry, reflection; R1–R4,
  R6, R7) and 0b (escaped-field records; R5); the energy diagnostics' metric and
  endpoint contract moved from R5 to R3 so phase 2 does not wait for 0b; the campaign
  harness follows SEP 30's rule on development records; and the phase-space filter
  prototype and directional occupancy were removed to the roadmap (Q13, Q14). Q22
  added for accumulators across a SEP 31 relaunch. No campaign was run and no default
  was changed.
- 2026-10-24 (later): Reviewed by Opus 5.5 (AI), covering the unreviewed 2026-10-20
  revision and the refresh (AR1–AR12 below). The drafting agent's response revises
  Part I; no finding remains open. Next: human review and acceptance.

- 2026-10-24: Accepted by John Travers after his review of the refreshed proposal and
  the AI review and response; Part I is frozen. The same day set `In progress`, with
  every phase on one branch, `feature/sep-0021-temporal-boundaries`, started on top of
  the open native CPU optimisation pull request; the earlier draft pull request was
  closed into it. On that base SEPs 29, 31 and 34 are Final, so §3.1's links and status
  were updated before freezing.
- 2026-10-25: Phase 0a reported in Part II, revised the same day after long walk-off
  cases the maintainer asked for; its recommendations accepted by John Travers as
  decisions D3–D9. Phases 1 and 4 are not built. Phase 2 started.
- 2026-10-28: Final, after phases 0a and 2 (Part II); phases 0b, 1, 3 and 4 were not
  built (D4, D17). Archived.

### AI review — 2026-10-24 (Opus 5.5)

Scope: the unreviewed 2026-10-20 revision (§3.1, §§4.10–4.12, the retained-window
mode, the guard monitor, placement and frames, R6/R7, Q19–Q21) and the 2026-10-24
refresh (the §3.1 table, SEP 34 in §4.6, the committed-endpoint hook, the phase 0a/0b
split with the energy-diagnostic contract moved to R3, the removal of the filter and
directional occupancy, Q22). Checked against the code on this branch, SEPs 22–34 and,
read-only, SEP 31's implementation branch. Earlier reviews used no numbered findings in
Part III, and R1–R7 already name the phase 0 recommendations, so these findings use
the prefix **AR**. Part I was not edited. No campaign or propagation was run; the
derivations below were redone by hand and the stated numbers recomputed.

**Verified without a finding.** §4.10: the FFT factors 2.14 and 4.57 at
$n_t=2^{14}$, and the break-even inequality $K_{\rm collar}/K_{\rm wide} >
C_{\rm wide}/C_{\rm collar}$. §4.11: the stationary-phase delay, and the widths
$W_0+\Delta\tau$ (fastest-light frame with an early-edge launch, or a midpoint frame
with a centred launch) against $W_0+2\Delta\tau$ (centred launch in the fastest-light
frame). §4.12: one full period of advection multiplies every stored coefficient by
$e^{-2\pi ik}=1$ on both grids (the field grid's frame term runs through the origin,
the envelope grid's acts on offset frequencies), so the final state cannot reveal it.
The cadence bound is dimensionally right and conservative for a packet whose leading
edge must be seen inside a guard before it reaches the seam. The continuity current is
right in sign and factor: for $\partial_z a=-v\partial_t a+i\beta_2\partial_t^2a/2$,
$a^*\partial_z a+{\rm c.c.}=-v\partial_t|a|^2-\beta_2\partial_t\operatorname{Im}(a^*\partial_t a)$,
and a local Kerr term drops out. A component $a\propto e^{i\Omega t}$ gives
$j=(v+\beta_2\Omega)|a|^2$, the group-delay rate $\beta_1+\beta_2\Omega-\beta_1^{\rm f}$,
which is consistent with the internal convention's plane wave $e^{i(\omega t-\beta z)}$
(conventions §2). The code claims also check out. `StepDriver._take_step_stats` runs after the commit at every call site: in
`backends/driver.py`, the adaptive loop commits at 1575–1578 and calls the hook at
1593, and replay commits at 1765–1768 and calls it at 1771. `backends/multistep.py`'s
adaptive march commits in `_attempt` before the call at 2968, and its replay calls it
at 3018. The low-storage replay commits at 638–646 and calls it at 590. The batch
replay calls it at 183 and fans it out to members at 237–240.
`STABILITY_RADIUS` = 3.3, `LOW_STORAGE_REAL_INTERVAL` = 2.785 and
`PEC_REAL_INTERVAL` = (1.0, 0.545, 0.3, 0.163, 0.0877, 0.0465), indexed by `order`.
`collar_fraction` must lie strictly inside (0, 0.5), and the flat region is
$\min(T_{\rm req}/2,(1-2f)T/2)$ (`grid.py` 1764, 1798). `collar_field_fraction` is a
field-norm share (`backends/spatial.py` 62). On the keyword interfaces, `absorber=` and
`absorber_scheme=` are as §3 states (`driver/interface.py` 724–725, 1198). Kernel
schema 35 and output schema 23 are correct. SEP 31's `edge_share` reads
`t_collar.one_low/one_high` (`coupling/remap.py` 474–476).

**AR1 — blocking. Q18's early keyword phase is neither adopted nor declined (§11,
Q18, §5).** Q18 recommends a phase 2a for `absorber=False` and `absorber_scheme=`,
"independently mergeable before phase 0 closes", and marks itself "for acceptance".
§11 still puts the keywords in phase 2, which needs 0a's R1/R3/R6/R7. Its second item,
which `walkoff`/`boundary_N` combinations are refused and how a slice gets
$L_{\rm ref}$, is deferred only by §5's "Phase 0 must specify". SEP 0 requires every
open question to be resolved or deferred to a named phase. *Fix:* record the
maintainer's decision on phase 2a. If it is adopted, add it to §11 with its own exit,
and note that it needs §5's definition of `absorber=False` (AR10). If it is declined,
say so in Q18. Defer the second item explicitly to phase 0a/R1.

**AR2 — blocking. Several open questions and references were not re-targeted after the
split (Q3, Q11, Q17, Q19, Q20, §4.7, §5).** Q3's status ("supports the walk-off
default subject to phase 0's R1 and this review") names no phase or decision point.
Q17 says the convention is "chosen before R1", but phase 0a's geometry sweep is what
measures it. Q19 and Q20 defer to "phase 0/R6" and "phase 0/R7"; they belong to 0a.
§4.7 (line 749) and Q11 say "phase 0" where the record's qualification is now 0b. §5's
"Phase 0 settles the precise signatures, route support and resume agreement … before
phase 2 begins" cannot hold for the escaped-field signatures if 0b may finish after
phase 2 starts. *Fix:* give Q3 and Q17 an explicit deferral ("deferred to phase 0a/R1;
maintainer decision before phase 2"). Change Q19/Q20 to 0a and §4.7/Q11 to 0b. Split
§5's sentence: 0a settles the keywords, `walkoff`, diagnostics and `BoundaryCheck`,
and 0b settles `escaped_field` and `escaped_spectrum` before phase 3.

**AR3 — should-fix. Removed work is still promised, and the roadmap items that the text
cites do not exist (§7, §4.6, §2, §4.9, §10, Q13, Q14).** §7, "The records and
diagnostics", still requires that tests "compare incoming and outgoing classification
against controlled reflected, wrapped and genuinely returning packets". That is the
removed directional occupancy. §4.6's "An optional diagnostic helps interpret failures"
now introduces only the removed item. §2, §4.9, §10, Q13 and Q14 say the filter
prototype and the directional occupancy are "kept as roadmap items", but
`docs/dev/roadmap.md` has neither; the only "phase-space" in `docs/` outside this SEP is
in SEP 8. SEP 0's rule for a reduced scope is to restore still-relevant informal work on
the roadmap. *Fix:* delete the §7 bullet, reword the §4.6 sentence, and add the two
roadmap items in the same pull request, linking §4.9.

**AR4 — should-fix. The §4.3 floor table no longer follows from the constants it now
cites (§1, §4.3).** The refresh says the table uses `PEC_REAL_INTERVAL`, rounded. With
$AN=252$, $AN/(2r)$ from those values gives 231, 420, 773 and 1437 for PEC2/3 to
PEC5/6. The table shows 229, 420, 788 and 1448, which are computed from the rounded
$r$ of 0.55, 0.30, 0.16 and 0.087. The abstract's "1450 at order five" inherits the
error. The 2026-10-18 review accepted the old figures as approximations only because
the source of $r$ was unstated. *Fix:* show $r$ = 0.545, 0.3, 0.163 and 0.0877 and
$n_{\min}$ = 231, 420, 773 and 1437, and say "about 1440" in §1.

**AR5 — should-fix. SEP 34's radial endpoint term is put in the wrong part of the
ledger (§3.1 SEP 34 row, §4.6).** The text says the endpoint term "is part of the
ledger's $E^\dagger M'E$" and "belongs to $E^\dagger M'E$". In the code the term
$-\tfrac12\ln(Y_z(z)/Y_z(0))$ is joined to $\Phi$ (`translate/linop.py`
`endpoint_phi`, 1743–1777), so it belongs to $\mathcal L$, and the propagated variable
is the field with $M\propto Y_z(z)$. Its contribution is
$2\operatorname{Re}(E^\dagger M\mathcal L_{\rm end}E)=-E^\dagger M'E$, which cancels the
metric-change term exactly; that is SEP 34 §4.4's invariance of $Y_z|E|^2$. A ledger
written as the text says would count the gradient's metric change twice and show
spurious removal on a lossless radial gradient. *Fix:* say that the endpoint term is
part of $\mathcal L$ and cancels $E^\dagger M'E$ for the lossless forward model. Add a
§7 witness: on a lossless radial gradient with zero collar, the ledger residual is
within its quadrature bound.

**AR6 — should-fix. The cost argument of §4.10 leaves out the removed temporal pass,
and the campaign's timing backends leave out the native build (§4.10, §11 phase 0).**
§4.10 says that SEP 16's 1329-to-965 call reduction "by itself … would not pay the FFT
cost of doubling" (1.38 against 2.14). The only measured cost of the in-stage temporal
pass, SEP 22 §3, shows it alone multiplying the step cost by 2.02–2.26 on `numba_cpu`
and 1.60–1.63 on the native CPU build. That case is a Cartesian Kerr-only grid, so it
does not transfer to modal routes, but it shows that $C_{\rm wide}/C_{\rm collar}$ at
$q=2$ can be near or below one. In that case doubling wins even at a 1.38 call ratio.
The ratio also differs by backend, while phase 0 times only `numpy_ref` and
`numba_cpu`, and R6 chooses a default on measured wall time. *Fix:* replace the
sentence with the decomposition: per route and per backend, measure the temporal pass's
share of $C_{\rm collar}$ beside the $n_t\log n_t$ growth. Include the native CPU build
in phase 0a's timing comparison, or state why numba timings stand in for it.

**AR7 — should-fix. The grid's temporal collar is shared state, not only the absorber's
geometry (§3.1, §4.8, §4.12, §6, Q4, Q16).** `grid.t_collar` also sets:

- the plasma source gate and its derivative (`translate/masks.py` 327–328,
  `t_window_fine`, `window_derivative`);
- the Raman response's tail taper, whose width is `t_collar.width_high`, which sets the
  response-memory truncation (`translate/raman.py` 131, 301, 747);
- the TDSE descriptor's `time_collar` (`translate/tdse.py` 782);
- the recorded analysis window `t_window_min/max` (`output/schema.py` 1016–1017);
- SEP 31's `edge_share`.

§4.8 deals with the plasma gate only. The R1 width default, Q4's "collar outside the
requested window" and the retained mode's guard band would therefore also change Raman
memory truncation, the TDSE descriptor and the output metadata. §4.12 also requires
that "a zero temporal rate neither grows the grid nor silently reduces the analysis
aperture". With `temporal=False` today, the flat region on a tight grid is still
$(1-2f)T/2<T_{\rm req}/2$, and that is what the gate, the Raman taper and
`t_window_min/max` use. Meeting the requirement therefore needs a grid change that
§6's inventory does not list. *Fix:* inventory these consumers in §3.1 and §6. State
whether the guard band, the analysis aperture, the source gate and the Raman taper stay
tied to `t_collar` or are decoupled, and convergence-test the Raman taper in §4.8
beside the gate. List any output or descriptor schema consequence in §9.

**AR8 — should-fix. R3 has no campaign item in phase 0a (§11).** Phase 0a's exit
includes R3, which covers the diagnostic thresholds, the route metrics, passivity, the
committed-endpoint contract and the quadrature error budget. Of these, 0a's bullets
cover only passivity. Nothing in 0a calibrates the occupancy and removal thresholds
that Q9 and R3 need. Nothing qualifies the in-stage removal quadrature at accepted
endpoints, or the spectral channel's exponential weights on stiff bins (§4.6). Before
the split, 0b's "emulated from … accepted-endpoint data" did this work. *Fix:* add a 0a
bullet. Emulate $U_t$, $U_\omega$ and $\eta$ from accepted-endpoint data on the narrow
runs at three refinement levels against the wide references and the exact split
difference, and set the occupancy and removal thresholds and the committed-endpoint
attachment point.

**AR9 — should-fix. The cadence bound's $v_{\max}$ has no qualified source (§4.11,
§4.12, Q20).** $\Delta z_{\rm check}\le\theta w_g/v_{\max}$ needs a drift bound that
covers late-born radiation. The group-delay range over the represented band is the
quantity that Q3 rejects for setting strength, because the band edges carry only noise
and dominate it. Used as $v_{\max}$, it would make the cadence, and with check planes
the step count, arbitrarily expensive. Used on the populated band only, it can miss new
radiation, as §4.11 notes. *Fix:* require R7 to define $v_{\max}$, for example from bins
above a stated spectral-energy floor at the latest check, refreshed at each check with
a growth allowance, and to report the check-induced steps it causes. Include in 0a's
guard-monitor bullet a witness in which a noise-floor band would otherwise set the
cadence.

**AR10 — should-fix. `absorber=False` is undefined on the spatial interfaces (§5, §11
phase 2).** Phase 2 adds `absorber=False` "on every `prop_*` interface", which now
includes `prop_radial`, `prop_free` and `prop_planar`. §5 defines it only as "disables
both soft collars", meaning spectral and temporal. It says nothing about the real-space
spatial collar or the k⊥ collar, which share `collar_length`. §2 forbids altering
spatial absorption inadvertently, and §3 puts a spatial absorber default out of scope.
*Fix:* say whether `absorber=False` leaves the spatial collars to their own settings
(recommended), or limit the keyword to the modal and GNLSE interfaces, and add a §7
test that the spatial strengths are unchanged.

**AR11 — note. The convention of the current and of the hook asymmetry (§4.6, §4.12).**

- (a) $j$ is derived for the internal amplitude, while $\eta_g$ is written in the saved
  amplitude $b$. The optics boundary conjugates the envelope, so
  $\operatorname{Im}(b^*\partial_tb)=-\operatorname{Im}(a^*\partial_ta)$ for a
  constant power scaling, and the dispersive term changes sign. *Fix:* label $j$ as
  internal-convention, and give the optics form or forbid evaluating it on saved
  arrays.
- (b) §4.6 says that `_on_accepted_step` runs before commitment "on an adaptive run".
  That holds for the Runge–Kutta loop (`driver.py` 1570 before 1575–1578), but the Adams
  march commits inside `_attempt` before `_record` (`multistep.py` 2964–2966). The
  asymmetry is between adaptive Runge–Kutta and everything else.
- (c) `_take_step_stats` returns at once unless periodic rows were requested
  (`driver.py` 1846). It is also not on the frozen protected surface
  (`tests/backends/test_driver.py`), whereas `_on_accepted_step` is. Accumulators must
  run before that return. What bumps the kernel schema is a new backend hook called
  from there, not the attachment itself.

*Fix:* reword those sentences.

**AR12 — note. Currency against the tree (§3.1).**

- This branch predates main's 5e8f688. Here SEP 29's header is "In progress", which is
  the cause of the three known `test_seps.py` failures, so "SEPs 22–30 are Final" holds
  only after a rebase.
- SEP 31 is now Final on its implementation branch. On merge the §3.1 link to
  `sep-0031-simulation-coupling.md` and the phrase "awaiting merge" go stale.
- I could not find the "uniform-loss test that keeps the collar's field fraction
  negligible" cited in the SEP 34 row. The tests that use `collar_field_fraction` are
  `test_radial_absorber.py`, `test_collar_diagnostic.py` and `test_free_absorber.py`,
  and SEP 34 §7's absorption row names no collar check.

*Fix:* rebase, update the SEP 31 link and status when it lands, and name the test or
drop the clause.

### Response to review — 2026-10-24 (drafting agent, Opus 5.5)

All twelve findings were checked against the tree and accepted; Part I is revised as
follows.

- **AR1.** Q18's early keyword phase is declined (John Travers, 2026-10-24): the
  keywords land with phase 2, and §11 lists them there. The `walkoff`/`boundary_N`
  combinations are deferred to phase 0a with R1 and implemented in phase 2.
- **AR2.** Q3 and Q17 are deferred to phase 0a/R1 and the maintainer decision before
  phase 2; Q19 and Q20 to phase 0a; §4.7, Q11 and Q12 to phase 0b. §5 now has phase 0a
  settle phase 2's signatures and phase 0b the record's, before phase 3.
- **AR3.** §7's classification test and §4.6's lead-in are removed or reworded, and the
  roadmap gains the two items that §2, §4.9, §10, Q13 and Q14 point to.
- **AR4.** The §4.3 table and the abstract now follow the package's intervals: 231,
  420, 773 and 1437 steps, and "about 1440" at order five.
- **AR5.** Confirmed in `translate/linop.py`: SEP 34's radial endpoint term is joined to
  the cumulative exponent. §3.1 and §4.6 now place it in $\mathcal L$, where it cancels
  $E^\dagger M'E$ for the retained forward field, counted once.
- **AR6.** §4.10 now weighs the removed temporal pass with SEP 22's measured per-step
  ratios, and phase 0 times cost on `numba_cpu` and the native CPU build.
- **AR7.** Confirmed: the temporal collar also feeds the plasma source gate and its
  derivative, the Raman tail taper, the TDSE descriptor, the recorded analysis window
  and SEP 31's flat-region reads. §6 requires each geometry change to state and test
  its effect on them. Q16 now records that today a zero-rate collar still shrinks the
  flat region on a tight grid, which phase 2 removes with Q4.
- **AR8.** Phase 0a gains the energy-diagnostic prototypes, their quadrature check
  and the threshold calibration that R3 needs.
- **AR9.** §4.12 takes $v_{\max}$ over a declared monitored band above a stated
  spectral floor, or §4.11's conservative monitoring range. The guard occupancy covers
  content outside that band, and R7 defines the band.
- **AR10.** §5: `absorber=False` on the spatial interfaces disables the temporal and
  spectral collars only. The transverse collars keep their own settings, and a switch
  for them stays out of scope. Confirmed by John Travers, 2026-10-24.
- **AR11.** §4.6 now restricts the pre-commit asymmetry to the adaptive Runge–Kutta
  loop. It also says the accumulators need a new call at the statistics call sites,
  which is a protected-surface change with a kernel schema bump. §4.12 notes the
  current's sign in the optics convention.
- **AR12.**
  - The SEP 34 row now attributes the uniform-loss test to SEP 34's pending phase 3,
    where it is defined.
  - SEP 29's folder and status, and SEP 31's link and "awaiting merge", are updated
    when this branch is rebased for acceptance, after those land.
