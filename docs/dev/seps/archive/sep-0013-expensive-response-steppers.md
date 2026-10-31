# SEP 13 — Low-call integration for expensive nonlinear responses

| | |
|---|---|
| **SEP** | 13 |
| **Title** | Low-call integration for expensive nonlinear responses |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-04 |
| **Resolution** | 2026-10-04: accepted by John Travers after the human and AI reviews |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI), for John Travers. Phases 1–4 on `feature/sep-0013-expensive-response-steppers-1`; solprop-kokkos `feature/sep-0013-expensive-response-steppers` (`d9fcc31`) |
| **Manual pages** | [Low-call stepping for expensive responses](../../../guide/numerics/adams.md); [Output files](../../../guide/results/output.md) (schema 9) |

## Part I — Proposal

### 1. Abstract

Compare and implement low-call interaction-picture integrators for propagation
whose nonlinear evaluation dominates its cost. Start with fixed-order Adams
methods, then variable step and order, comparing one-call PEC, two-call PECE,
explicit Adams–Bashforth and a bounded multirate correction experiment. Select
the method with the lowest measured or explicitly cost-weighted work at useful
physical accuracy, including loose tolerances around `rtol=1e-2`. Implement the
selected method and its lifecycle in `numpy_ref` only, with explicit Numba and
Kokkos exceptions. This is a standalone prerequisite for
[SEP 12](sep-0012-tdse-nonlinear-response.md): it uses existing responses and
manufactured problems, adds no TDSE dependency, and supplies a tested method,
call ledger and qualification protocol which SEP 12 must then check with the
actual atomic response. Existing Runge–Kutta behaviour remains the reference
and the default for existing runs.

### 2. Motivation and scope

A nonlinear function can be expensive even when the optical state is small.
One TDSE call evolves complete temporal waveforms at all radial nodes. Its
internal time steps are distinct from the macroscopic z steps, and an RK
stage at a trial field requires another complete response. Batching radial
points helps; it does not batch causally dependent z stages.

The updated solprop-tdse public performance report [1] measures the complete
host response call, including third-order shadow and plasma observables, over
1,000 internal atomic steps. The H200 medians, reported per atomic step for the
whole batch, are 0.515, 0.548 and 0.810 ms for 1, 8 and 30 rows. Thus the
30-row/one-row ratio is approximately 1.57. These are measurements for the
specified production basis and short sample lattice, not a latency guarantee
for a long, strongly driven waveform or the newer strong-field basis.

The motivating cost scenario is **120 s or more per complete nonlinear
evaluation**, supplied as a planning estimate. At 120 s, a day permits at most
720 such calls before setup and other work. In steady operation that is at
most 720 one-call steps, 360 two-call steps, or about 102 seven-call RK trials,
before startup, rejection and diagnostic costs. Across 4.2 m those idealised
mean steps are about 5.8, 11.7 and 41 mm respectively. This is budget arithmetic,
not evidence that any of those steps is stable or accurate. A more expensive
configuration reduces the call allowance proportionally.

Use this canonical specification supplied by the maintainer, shown as a
non-executing scientific example:

```text
import solprop
from solprop.driver import prop_capillary
from solprop.gasflow import steady_fill

fill = steady_fill("He", [0.0, 4.2 / 2, 4.2], [0.0, 0.2, 0.0], radius=256e-6)
run = prop_capillary(
    256e-6, 4.2, "He", fill,
    lambda0=800e-9,
    fwhm=6e-15,
    energy=1e-3,
    modes=1,
    trange=60e-15,
    wavelength_limits=(90e-9, 1.6e-6),
    rtol=1e-2,
    n_save=50,
    noise=None,
    backend="numpy_ref",
)
```

Pressures are bar at the interface: the middle port is 0.2 bar, or 20,000 Pa.
Keep `steady_fill`'s resolved flow, endpoint treatment, temperature and density;
do not substitute a different analytic pressure curve. The interior port at
2.1 m must be checked as a physical profile breakpoint. Record the realised
window and grids: `trange=60e-15` is a request, not a promise that every atomic
waveform lasts exactly 60 fs after grid construction and collars.

Use `n_save=50` and `noise=None` for the main comparison and intended
production configuration. The count includes the entrance plane, so there
are 49 output intervals; physical breakpoints can require additional landings.
Compare fewer exact save planes when measuring the output-related cost floor.
No random launch noise is added to this canonical case.

The existing `prop_capillary(modes=1)` is radially integrated `HE_11`, not an
effective-area single-node closure. At the inspected revision its default
radial request is 32, rounded to 33 Kronrod nodes. Hold the quadrature fixed
while comparing z methods. SEP 12 separately qualifies whether fewer nodes
are sufficient with TDSE.

The 90 nm lower grid limit is retained for this **numerical** study using
existing declared responses. It extends beyond SEP 12's initial approximately
200 nm physical qualification scope; neither this grid nor a converged stepper
validates its material or TDSE response at 90 nm. The maintainer's observation
that `rtol=1e-2` looks adequate is a useful starting point. It is not converted
into an unstated high-precision requirement or a claim of 1% global error.

Scope is deterministic propagation with the existing diagonal cumulative
linear flow, initially scalar field-grid capillaries with smooth density
segments and declared hard breakpoints. Include a small envelope/GNLSE control
case to establish general usefulness. Coupled noncommuting flows, tapers,
stochastic increments, a general stiff solver, vector-response qualification
and compiled implementations are outside the initial qualification. Multirate
work is a bounded comparator, not a requirement to build a general integrator
framework. No atomic physics is added here.

### 3. Context

The inspected solprop revision is `2c7471a`. See the
[stepper design](../../stepper.md), [conventions](../../conventions.md),
[architecture](../../architecture.md) and [kernel contract](../../kernel-contract.md).

| Current component | Relevance |
|---|---|
| `stepper/tableau.py`, `controller.py`, `schedule.py` | RK tableaux, a fixed estimator exponent, PI control and exact landing targets; an Adams method is not another RK tableau |
| `backends/driver.py:StepDriver` | Owns trials, acceptance, rejections, save-plane propagation and history of taken steps; persistent derivative history needs an explicit accept/reject lifecycle |
| `backends/numpy_ref/backend.py:_trial_step` | Seven fresh nonlinear evaluations for IP-DOPRI5/IP-Tsit5; readable existing oracle |
| `backends/numba_cpu/backend.py` | Normally six fresh evaluations after FSAL startup; a relevant cost comparator even though this SEP adds no Numba implementation |
| `translate/stage.py` and cumulative linear tables | Provide accurate density-dependent propagators and source factors at the actual z position |
| `driver/simulation.py`, `output/` | Default 201 saved planes including the entrance, exact landing and resume semantics; potentially substantial work independent of tolerance |

Module paths above are relative to `src/solprop/`. The current driver caps a
step at the width of the Chebyshev linear-table interval containing its start
(a step may cross an interval edge), and lands exactly on hard kinks, save
planes and the end (`stepper/schedule.py:landing`). The capillary constructor
starts at `length * 1e-4`, and the normal PI growth cap is 1.1. For this
application, audit whether startup, collars, table-interval widths or output
landings set the step count before choosing a more sophisticated formula.
Raising `rtol` cannot remove those limits. At the inspected revision the
driver computes the limiting landing target of each step but does not record
it, and no backend counts nonlinear source evaluations; both are new work here.

`steady_fill` returns a `PressureGradient` whose stage boundaries are its
breakpoints, so the interior port at 2.1 m is already a hard kink and a forced
landing. `prop_capillary` has no tableau or integrator keyword yet. Saved
fields are exact step endpoints: [ADR-009](../../decisions.md) chose landing over
dense output, and §6.4 revisits that choice for the new methods only.

SEP 12 originally contained this investigation inside its TDSE implementation.
This SEP takes ownership of integrator mathematics, selection and lifecycle.
It can finish with existing responses; SEP 12 follows it and owns validation
against actual TDSE current, including any change in the winning method.

### 4. Theory

#### 4.1 Transporting the history

Let $a(z)$ be the internal complex spectral state, with shape
`(n_batch, n_modes, n_pol, n_omega)`, in solprop's raw-FFT convention. Positions
and steps are in metres. Write

$$a'=L(z)a+N(a,z),\qquad U(z_2,z_1)=e^{\Phi(z_2)-\Phi(z_1)}.$$

Here $\Phi$ is the existing cumulative operator, including phase, loss and the
power-normalisation term $-\tfrac12\ln[w_m(z)/w_m(0)]$, with
$w_m=\mathrm{Re}\,n_{\mathrm{eff},m}$. Do not replace it by a frozen local
dispersion operator or omit the normalisation term on a gradient.

At anchor $z_n$, $v(z)=U(z_n,z)a(z)$ satisfies

$$v'=g^{(n)}(z,v)=U(z_n,z)N(U(z,z_n)v,z).$$

Store physical derivatives $F_j=N(a_j,z_j)$ and positions, so the history at
this anchor is $g_j=U(z_n,z_j)F_j$. Each historical source retains its own
density, field and response identity. A predicted-state derivative additionally
records that fact. The transformed history is not a sequence of derivatives
already evaluated at the current density. Old derivatives are transported
forward to the anchor, which attenuates them under loss; only the predicted
endpoint source is transported back, over one step. Amplification of roundoff
therefore needs gain or a long backward span (a lagging anchor, dense output
far from the anchor), and that conditioning belongs to admission and restart
policy.

Transport does not make every source smooth in $z$. A phase-mismatched
contribution at frequency $\omega$, for example harmonic or far-UV generation,
appears in the transported history as $g\propto e^{i\Delta\beta z}$ with the
mismatch $\Delta\beta$ of the driving process. This is oscillatory forcing,
not an eigenvalue of the source Jacobian. An Adams method interpolates $g$
across its whole stencil, about $qh$, whereas an RK step samples it within
$h$, so mismatched forcing can limit a multistep step size more than an RK
step. Its accuracy limit in $\Delta\beta\,h$ is measured (§7).

#### 4.2 Variable-step Adams formulas

For order $q$, interpolate the old $g_j$ on $q$ history positions to obtain a
degree-$q-1$ polynomial $p_B(s)$. The predictor is

$$v_P=a_n+\int_{z_n}^{z_{n+1}}p_B(s)\,ds.$$

Evaluate $g_P=U(z_n,z_{n+1})N(U(z_{n+1},z_n)v_P,z_{n+1})$. Interpolate
$g_P$ and the latest $q-1$ old derivatives to obtain $p_M(s)$; a single
Adams–Moulton correction is

$$v_C=a_n+\int_{z_n}^{z_{n+1}}p_M(s)\,ds,\qquad
a_{n+1}=U(z_{n+1},z_n)v_C.$$

Compute weights by integrating Lagrange/Newton polynomials on the **actual**
history positions, scaled to the current step. Do not use constant-step
weights after a step-size change. First qualify order four, whose constant-step
weights are

$$
v_P=a_n+\frac h{24}(55g_n-59g_{n-1}+37g_{n-2}-9g_{n-3}),
$$
$$
v_C=a_n+\frac h{24}(9g_P+19g_n-5g_{n-1}+g_{n-2}).
$$

The execution choices have different contracts:

| Method | Fresh expensive evaluations in a mature accepted step | Stored derivative and consequence |
|---|---|---|
| AB | 1 | Evaluate at the provisionally accepted predictor for future history; a divided-difference estimator controls the update and is rechecked with the new derivative (§4.3) |
| ABM PEC | 1 | Store the derivative at the predictor while accepting the corrected field; no accepted-field source is available automatically |
| ABM PECE | 2 | Accept or reject on $v_C-v_P$ before the second evaluation, so a rejection costs one call; then reevaluate at the corrected accepted field |
| Iterated AM | Variable | Listed for contrast, not implemented here: solve the implicit equation to a declared nonlinear tolerance, charging every iteration/Jacobian action, including failures |

One correction is an explicit predictor–corrector algorithm, even though its
corrector coefficients come from an implicit formula. It does not inherit the
fully solved AM stability region. PECE does not perform a second correction;
its final evaluation supplies consistent endpoint history. The one-call PEC
history is a numerical approximation, not exact memoisation of an unchanged
field. Its order, stability and diagnostics must be tested as an augmented
state method. The second-order PEC in Richter et al. [2] is a separate cheap
comparator; its predicted derivative is likewise not an FSAL derivative.

#### 4.3 Derived error estimates

With smooth exact history, let the leading exact-minus-formula defects be
$C_Bh^{q+1}g^{(q)}$ and $C_Mh^{q+1}g^{(q)}$. In normalised coordinates
$\theta=(s-z_n)/h$ they follow directly from interpolation remainders:

$$
C_B=\frac1{q!}\int_0^1\prod_{j=0}^{q-1}(\theta-\theta_{n-j})\,d\theta,
\quad
C_M=\frac1{q!}\int_0^1(\theta-1)
                 \prod_{j=0}^{q-2}(\theta-\theta_{n-j})\,d\theta.
$$

For constant-step AB4/AM4, $C_B=251/720$ and $C_M=-19/720$.
The predicted endpoint's source error changes the correction at higher order
under these smooth-history assumptions. Consequently

$$e_C\simeq\frac{C_M}{C_B-C_M}(v_C-v_P)
=-\frac{19}{270}(v_C-v_P)$$

estimates the corrector's local defect. Derive the coefficients again for the
actual step ratios; check conditioning when their difference is small. Carry
$e_C$ forward with $U(z_{n+1},z_n)$ before taking the state-error norm.
This is an asymptotic estimator, not a global-error bound. In PEC, predicted
history errors add another contribution; separately measure their effect and
compare against PECE and refined references. A small predictor/corrector
difference cannot certify an unresolved source or a resonant instability.

For explicit AB4, retaining a fifth old derivative gives an estimator without
another evaluation per mature step. Subtracting the integrated cubic from the
integrated quartic interpolant gives

$$\delta v=\frac{251h}{720}
(g_n-4g_{n-1}+6g_{n-2}-4g_{n-3}+g_{n-4})
=\frac{251}{720}h^5g^{(4)}(z_n)+O(h^6).$$

On irregular nodes use the fourth divided difference times the integral of
the four-node product instead. An AB4/AB3 difference is only $O(h^4)$ and
requires a fourth-root controller, not the fifth-root controller of these
order-four local-defect estimates.

This estimator is retrospective: it uses only derivatives already evaluated
and cannot see an onset, such as an ionisation threshold, inside the step it
controls. AB therefore treats its predicted state as provisional. Before the
call it accepts or rejects on the history estimate, at no cost. After the call
it recomputes the estimate over the stencil ending at the new derivative
$g_{n+1}$; if that exceeds the tolerance it discards the step and its
derivative and retries from $z_n$ with a smaller step. Such a rollback costs
one call and keeps the accepted history free of a step that was never
controlled.

#### 4.4 Variable order, stability and useful tolerance

After fixed-order qualification, investigate orders two through six, with
explicit low-order startup. Higher orders are not automatically cheaper when
the response changes rapidly near compression. Derive neighbouring-order
error estimates from the same transported history where enough data exist.
Choose among eligible $q-1,q,q+1$ using predicted accepted distance per new
expensive call, with hysteresis, a minimum history length and no order increase
immediately after rejection or a history reset. If the data cannot estimate a
candidate's error, keep the current order or use a qualified lower order; do
not evaluate the expensive source merely to try every order.

The controller is method-specific: its local exponent is $q+1$ for the above
estimator, not a fixed DOPRI constant. Start without asynchronous control lag.
Compare conservative step-ratio policies with the existing 1.1 growth cap;
select the allowed ratios through order, stability and conditioning tests.
When a landing clips a step, recompute the variable-step error coefficients.
The RK shortcut of rescaling a norm by $(h_{\rm proposed}/h_{\rm taken})^{q+1}$
is not generally valid with an otherwise unchanged irregular history stencil.
A clipped sliver before a forced landing, followed by a full step after it,
also produces a large step ratio, which variable-step Adams stability does
not tolerate as RK does. When the distance to the next forced landing is less
than about two proposals, divide it into equal steps instead; the existing
1.1 stretch only absorbs the final sliver.

Measure stability on imaginary and negative-real source eigenvalues, including
parasitic history modes and products of update matrices for changing step
ratios. The temporal absorbing collar is in $N$, so it can impose an explicit
stability limit even for a weak optical field. Relaxing tolerance must not
silently bypass this limit. Use the method's actual stability domain rather
than DOPRI5's collar bound. A cheap rate-based collar treatment inside a
multirate solve is one alternative; an unqualified per-step split can reduce
the whole method to first order.

Sweep `rtol` through at least $10^{-1},3\times10^{-2},10^{-2},3\times10^{-3},
10^{-3}$, against a converged full-length reference (§7). Compare
physical errors at common output positions; equal `rtol` is not equal accuracy
across methods. Keep an absolute scale for weak bands and empty modes instead
of unstable independent relative normalisation at spectral zeros. Report a
Pareto curve of calls versus error, so the maintainer can choose an adequate
solution without paying for unnecessary digits.

#### 4.5 Bounded multirate correction study

Compare a split $N=N_c+R$, where $N_c$ is cheap and
$R=N_{\rm target}-N_c$ is obtained only at expensive sample states. In the
current anchor the auxiliary problem is

$$v'=U(z_n,z)N_c(U(z,z_n)v,z)+\widehat R^{(n)}(z),$$

where $\widehat R^{(n)}$ interpolates **transported** historical remainders.
Cheap substeps resolve Kerr, a simple plasma surrogate or the temporal collar.
An expensive endpoint call audits/updates the remainder; an endpoint correction
changes the field and has the same PEC/PECE distinction as above.

Implement at most one explicitly derived low-order correction prototype for
the initial comparison, or a suitable published MRI-GARK pair [5, 6] whose
coupling conditions and embedded estimate have been checked. Specify slow
interpolation order, fast-solve tolerance, endpoint state, correction weights
and the combined error estimate. A polynomial forcing sketch alone does not
establish the coupled method's order. Expensive and slowly varying are separate
properties: near ionisation, $R$ may vary faster than $N_c$. An adaptive fast
solver cannot observe an inaccurate frozen remainder. Require separate slow
audits, and count their full cost; reject the approach if they erase the saving.
There is no arbitrary multiplier on a TDSE correction every few steps.

### 5. Proposed interface

Add a frozen `Multistep` specification in `solprop.stepper`, passed explicitly
through `SimulationSettings` and the propagation convenience interfaces. A
proposed, non-executable sketch is:

```text
method = Multistep(
    family="adams",
    corrector="pec",       # or "pece"; the winning choice is not yet known
    order=4,               # fixed; order=None enables qualified order selection
    max_order=6,
)
run = prop_capillary(..., backend="numpy_ref", integrator=method, rtol=1e-2,
                     n_save=50, noise=None, max_response_calls=budget)
```

`integrator=None` retains the existing tableau/controller path. Reject a new
integrator together with a conflicting explicit RK tableau. Scientific step
settings are serialisable; derivative buffers and runtime handles are not part
of that specification. Do not decide the algorithm from a backend name or an
implicit TDSE detection. Require `backend="numpy_ref"` initially; unsupported
backend selections refuse the feature before expensive preparation, without
altering `auto` for other runs.

Propose an optional positive integer `max_response_calls` covering all charged
target-source evaluations in a run, including startup, rejected attempts,
diagnostics, audits and restarts. Reserve enough calls before starting a trial
to leave a valid accepted state. Exhaustion returns an explicitly incomplete
run/checkpoint or a named controlled-stop exception carrying that checkpoint;
it never marks an unfinished propagation successful or accepts an inaccurate
step to fit the budget. Define this convention with the existing output API
before implementation. Ordinary unbudgeted propagation stays unchanged.

Record method, actual order per trial, proposed/taken step, defect norm,
accept/reject decision, history resets and limiter reason. Counters distinguish
target source calls, cheap-source calls, atomic chunks when a later provider
uses them, startup, diagnostics and audits. Record the state identity behind
each response: accepted, predicted or dense-output field. No source statistic
may be silently attached to a different field.

### 6. Implementation strategies

#### 6.1 Recommended sequence and layer boundary

Build the readable order-four AB/PEC/PECE comparison first; add variable order
only after those schemes and their estimators pass. Share source assembly,
linear-flow evaluation and physical diagnostics with existing NumPy code.
Keep the unchanged RK trial path as an independent reference. Persist physical
derivatives plus z/state identities, initially in a bounded history ring with
weights formed on normalised nodes. This is easier to inspect than immediately
introducing a Nordsieck representation. A Nordsieck form is a later option if
conditioning or dense output justifies it, with independent parity tests.

The public specification belongs in `stepper/`; translation validates it and
produces the numerical method/controller descriptor. Persistent work arrays
and numerical execution belong in `backends/numpy_ref/`. The multistep loop,
with its explicit begin-trial, commit and discard semantics so rejected steps
cannot mutate accepted history, lives in the NumPy backend as its own
`_march_to`, reusing the driver's existing stage scalars, cumulative
$\Delta\Phi$ and source evaluation. The shared `StepDriver` and its protected
surface stay unchanged, so the compiled backends are untouched by a method
they refuse. The method descriptor is a new field of the kernel's stepper
settings, which bumps the kernel schema once; solprop-kokkos follows with its
version literal only. Settings resolution refuses the new integrator with any
backend other than `numpy_ref` before a backend is constructed, so no
out-of-tree backend can silently ignore the field and run RK. Do not construct
a fake RK tableau with history masquerading as stages.

Native solver wrappers such as CVODE/VODE provide useful design references
[3, 4], not a shortcut to the required one-call contract. Their variable-order
Adams methods solve an implicit corrector and can perform multiple RHS
evaluations/iterations. Blindly wrapping them around a full TDSE response gives
neither a fixed PECE count nor a cheap Jacobian. Add no dependency here.

#### 6.2 Lifecycle and artificial step limits

| Event | Required behaviour |
|---|---|
| Startup | Compare a cheap low-order ramp with RK startup at matched error; count all calls and qualify starting error before higher order |
| Rejection | Discard trial history, retain accepted history, recompute weights for the new step, and reuse only genuinely identical source evaluations |
| Corrected PEC field | Keep predicted-source identity; a requested exact accepted-state atomic diagnostic is a new charged call |
| Pure observation/save | Preserve valid multistep history across a save plane (the NumPy RK path already keeps its controller state there; only Numba's FSAL cache resets, ADR-019) |
| Hard physical breakpoint | Land exactly, use the closing side for the arriving step, discard history spanning the discontinuity and restart on the opening side; `steady_fill` ports are such breakpoints |
| Smooth table boundary | Steps already cross Chebyshev interval edges under the interval-width cap; retain that cap initially and qualify each source/flow query at its own position |
| Changed field, grid, modes, radial rule, source, operator or noise identity | Invalidate affected history explicitly; refuse unsupported state-changing effects rather than continuing stale history |
| Checkpoint/resume | Fresh-start resume only: restart the history from the saved exact-landing field, charge and label the startup calls; no method history is persisted in the output |

Do not treat every smooth pressure/table knot as a physical discontinuity.
Conversely do not extrapolate a history across the central fill port merely
because pressure is continuous there. Inspect the resolved profile's smoothness
and registered events. Capillary vacuum-end handling remains the fill model's
responsibility and must be represented consistently in the operator tables.

The canonical 50 output planes enforce at least 49 accepted landing steps.
At the illustrative two-minute call cost, that corresponds to 1 h 38 min
for one-call methods, or 3 h 16 min for two-call methods, before startup,
physical breakpoints and other costs. The central 2.1 m port is not one of
those uniformly spaced output planes; preserve its required event landing
independently. Compare this configuration with fewer exact save planes and,
secondarily, the unchanged default of 201 planes (at least 200 landing steps,
6 h 40 min or 13 h 20 min at the same illustrative costs). Optical spectra
and durations from a saved field need no new source solve. Existing plasma
statistics that recompute the response must be identified and charged.

Optional dense observation, which removes the output landings, is analysed
in §6.4. It is an addition, not a reason to delay the method comparison.

#### 6.3 Selection by total work

Report $N_{\rm target}$, $N_c$, rejected work and actual host time separately.
For proxy experiments, display the labelled sensitivity model
$T_{\rm model}=N_{\rm target}C_{\rm target}+T_{\rm host}$ for several assumed
call costs, including 120 s. Do not insert sleeps or label this a measured TDSE
runtime. For real TDSE later, use measured complete calls with actual row
count, basis, waveform length and outputs. Startup and audits can dominate a
short run; report cold-start and mature-history windows separately.

Select one primary production method after the comparison, with only useful
controls retained in the public API. The leading hypothesis is variable-step,
variable-order PEC when its predicted-source history is stable, with PECE as
the stronger endpoint-consistency comparator. Explicit AB and the multirate
prototype must have a fair chance to win. One-call PEC loses to PECE if it needs
more than twice as many accepted steps or equivalent extra audits. Any candidate
loses to existing RK if its stability-limited step count outweighs its call
reduction. No winner or speedup is declared by this Draft.

#### 6.4 Interpolation and dense observation

Dense observation returns the field at an observation position $z_o$ inside
an accepted step $[z_n,z_{n+1}]$ without landing there, removing the output
floor of §6.2. It never changes the method selection: §6.3 ranks methods by
the cost of the full propagation with exact landings, and dense observation is
reported separately as a reduction of that cost. [ADR-009](../../decisions.md)
(land on save planes, no dense output) stays the default; a new ADR records
the opt-in exception for `numpy_ref`, and the compiled backends refuse it by
name.

**Frame.** Interpolate in the interaction picture. The internal state carries
the residual phase of $\Phi$, many radians per step at the edges of a wide
field grid, so a polynomial in $a(z)$ itself is useless. Every option below
interpolates $v(z)=U(z_*,z)a(z)$ about an anchor $z_*$ in or next to the step
and applies the exact linear flow $U(z_o,z_*)$ afterwards.

| Option | Data used | Local error at $z_o$ | Cost and contract |
|---|---|---|---|
| (a1) Lagrange on accepted states | $k$ accepted states transported to $z_*$, centred on the step where available | $O(h^k)$ | No calls; $k$ stored states; any method, RK included; output delayed until the bracketing steps are accepted |
| (a2) Cubic Hermite | $v_n$, $v_{n+1}$ and endpoint derivatives $g_n$, $g_{n+1}$ | $O(h^4)$ | No calls: AB and PECE have exact endpoint derivatives, PEC its labelled predicted one, NumPy IP-DOPRI5/Tsit5 their evaluated last stage; any method |
| (b) Adams-native | The accepted step's derivative polynomial: $p_M$ for PEC/PECE, $p_B$ for AB | $O(h^{q+1})$, the step's own order | $v(z_o)=a_n+\int_{z_n}^{z_o}p(s)\,ds$ with fractional-step weights; reproduces the accepted endpoint exactly; PEC's interpolant carries the predicted-source identity |
| (c) RK continuous extension | The seven stages of IP-DOPRI5/Tsit5 | $O(h^5)$, one below the solution | Free, but changes the existing RK path; outside this SEP, noted for a later ADR-009 revision |

Options (a1) and (a2) need no knowledge of the stepper and are the reference
against which (b) is checked. Option (b) is what the stepper adds: at no call
and no extra storage it matches the local order of the step, where cubic
Hermite is one order lower than an order-four Adams step and two lower than
DOPRI5. Implement (a1), (a2) and (b) behind one opt-in observation setting.

The interpolation error is not controlled by the step's error norm. That norm
is dominated by the strongest spectral components, so a weak band, the far-UV
here, can carry a larger relative error at a dense position and scatter
between otherwise similar runs; this is why exact landing remains the default.
Measure each option's field, weak-band and UV errors against exact landing on
the canonical run (§7). Dense positions never become history points and
trigger no source call. Source-derived statistics at a dense position are
unavailable, or interpolated from the bracketing source records and labelled
so, unless a charged call is requested. Stencils never cross a hard
breakpoint, and an observation is produced only once its bracketing step is
accepted, never by extrapolation. A resumable checkpoint is still an exact
landing.

### 7. Verification and validation

**Separate numerical roles.** An independent tightly integrated reference
establishes z error with the same response and grid. Basis, temporal sampling,
radial quadrature and material-model uncertainty are separate. Cheap response
proxies can rank algorithms and expose defects; only SEP 12 can establish that
the ranking transfers to a full coherent atomic response.

**Full-length evidence.** With the existing responses the canonical 4.2 m
case runs in seconds, so the primary evidence is the full propagation, as the
maintainer requires. This is an explicit exception, for this cheap canonical
case only, to the repository's rule against full-length propagation in tests
and timings; SEP 12's TDSE runs keep their own budget. Build a full-length RK
reference at tight tolerance and show its own convergence with a three-level
`rtol` ladder and an order gate. Compare every candidate against it at the 50
common output planes over the whole length, and report its calls-versus-error
Pareto curve. A slow-marked full-length qualification test of the selected
method against that reference is part of delivery.

Short windows of about ten production-scale accepted steps, started from
saved reference fields at the entrance, compression, UV generation and the two
sides of the 2.1 m port, are diagnostics: they locate where a method's error
or work arises and carry the stability and lifecycle tests. Store the starting
field, common grid, resolved fill/material identities and verified history
for warm tests; a manufactured warm history cannot substitute for testing real
startup. Mismatched oscillatory forcing (§4.1) is tested on a manufactured
problem with known $\Delta\beta$, measuring each method's accuracy limit in
$\Delta\beta\,h$ against RK.

The canonical comparison fixes the field and flow specification in §2 and
records all defaults explicitly: Gaussian launch and CEP, material/rate model,
Kerr/THG/plasma switches, temperature, loss, collars and radial rule, with
`noise=None` and 50 saved planes for the main comparison.
Compare Kerr-only, Kerr plus the existing plasma model, and a deterministic
whole-waveform memory-response fixture. For multirate tests define the target
and surrogate explicitly. Include threshold-sensitive and oscillatory
remainders; a smooth Kerr-only win is insufficient.

| Test family | Required evidence |
|---|---|
| Algebra and transport | Exact polynomial moments, constant and variable step weights, zero-source exact linear flow, constant forcing and independently known nonautonomous gradient solution |
| Fixed-order methods | Three-level global convergence for every admitted order; leading defect constants, PEC augmented-history order and PECE endpoint consistency |
| Adaptive methods | Tolerance ladders, step/order changes, alternating ratios, order hysteresis, rejection and source-noise floor; actual error alongside estimator values |
| Stability | Imaginary/dissipative spectra, parasitic roots, changing ratios, temporal-collar eigenvalues and nonlinear resonance/threshold cases |
| Lifecycle | Startup, hard breakpoint, smooth table boundary, clipped save step, pure observation, checkpoint round trip, incompatible identity, controlled budget exhaustion and failed source call |
| Canonical run | Full length against the converged reference, and short diagnostic windows: fundamental complex field without time/phase alignment, peak power, declared pulse duration, pulse energy, spectral shift and UV energy/phase/delay above declared floors |
| General usefulness | A small deterministic envelope/GNLSE control and a short multimode capillary case, within the admitted diagonal-flow scope |
| Multirate prototype | Derived coupled order, independent micro/macro refinements, endpoint-correction defects and cases where the residual is not slow |
| Work accounting | Instrumented calls for every path; no hidden solve at a save, diagnostic, order probe or restart; cold/warm and exact/dense-output costs distinguished |
| Dense observation | Interpolation order of options (a1), (a2) and (b) on a smooth manufactured problem; endpoint agreement; field, weak-band and UV error against exact landing on the canonical run |

Use approximately 1% errors in fundamental field, peak power and duration as
an initial comparison point inherited from SEP 12, alongside looser 3% and
5% bands. Those bands are proposed reporting levels, not newly accepted
physical tolerances. Show where the current `rtol=1e-2` result lies. Also
report field phase and delay without alignment, compressed-pulse satellites
and a weak-UV absolute error, so visually similar pump spectra cannot hide a
different waveform. Numerical acceptance levels and the chosen duration/UV
metric are fixed with the maintainer at the selection gate. No phase requires
an unnecessarily tight z tolerance once the selected observables converge.

Verification uses analytic/polynomial solutions, independently refined RK and
the published numerical-analysis framework [3–6]. NumPy is the only new
implementation; compare common RK results with existing Numba where useful,
but do not require compiled parity for a method those backends refuse.

### 8. Documentation

Add a manual page describing low-call methods, their cost and accuracy tradeoff,
PEC versus PECE, order/step control, source diagnostics and restart semantics.
Update the stepper design, kernel contract, output guide, backend capability
table and benchmark ledger at implementation. Add a short deterministic
tutorial with existing responses. The canonical 4.2 m scientific run belongs
in a worked example with its measured accuracy/call ledger. Update the changelog for delivered
functionality and any deliberate Luna differences.

### 9. Compatibility and impact

Existing RK defaults, tableaux, cached schedules and backend selection remain
unchanged. New method selection, trial/history lifecycle and call accounting
require one kernel schema bump (the new stepper-settings field) and its
literal seam tests; the shared driver surface is unchanged (§6.1). Resume is
fresh-start (§6.2), so no method history is persisted. Method record,
counters, limiter reasons and dense-observation identity require output-schema
changes under the normal compatibility policy. A list of z steps alone is not
enough to replay a variable-order/PEC trajectory: record method, orders,
resets, source-state policy and relevant controller state.

**Backend exception:** only `numpy_ref` implements the new methods in this SEP.
Numba CPU and solprop-kokkos reject them by name; their support is not required
for `Final`. This exception is requested explicitly for acceptance. It leaves
room for later ports if general-purpose benchmarks justify them. SEP 12 uses
the selected NumPy method with its independently accelerated atomic engine.

This proposal moves the stepper investigation out of SEP 12 without accepting
either proposal or selecting a new global default. No dependency on SEP 12 is
introduced here, so there is no circular prerequisite.

### 10. Alternatives considered

| Alternative | Decision |
|---|---|
| Relax RK tolerances only | Required baseline; useful if error-limited, ineffective against stage count, output/collar limits or source stiffness |
| Optimise Numba/Kokkos host kernels first | Outside scope; count reduction is the priority at minute-scale target calls |
| Adopt variable-order AM through a solver library | Mature option, but implicit iterations and history/restart/norm conventions need adaptation and can spend many target calls; no new library dependency initially |
| Fixed-order AB alone | Small and transparent, one call per step; retain as a comparator because stability may force smaller steps than PEC/PECE |
| Treat PEC as a fully implicit AM method | Rejected: finite correction count gives a different method and stability problem |
| High-order methods up to order twelve immediately | Defer; initial orders two through six bound complexity and may suit loose tolerances better |
| Multirate correction as the assumed winner | Defer selection to evidence; target cost does not establish slow residual variation |
| Implement every candidate permanently | Keep the comparison reproducible, but expose the winning method and justified controls rather than an unqualified collection |

### 11. Phases

1. **Measure the numerical problem.** Freeze the canonical configuration,
   build the converged full-length RK reference and the short diagnostic
   windows, instrument all source calls and limiter reasons, compare existing
   RK methods/tolerances at full length, and define physical reporting metrics.
   Publish the baseline including startup, save planes and collars. No TDSE
   installation is required.
2. **Fixed-order NumPy comparison.** Implement derived AB4, ABM4 PEC and PECE,
   and the second-order PEC comparator, with the algebra and convergence-order
   tests needed to trust the comparison. Add the persistent-history lifecycle
   and explicit backend refusals, and compare at full length.
3. **Adaptive candidates.** Add qualified variable steps/orders, budget and
   fresh-start resume handling, the dense-observation options of §6.4 and one
   bounded multirate prototype. Report negative results as well as savings.
4. **Select and deliver.** Choose the lowest-cost qualified method at the
   agreed physical accuracy, finish its public interface/manual/short example,
   and record reproducible selection evidence. If none beats RK, retain RK and
   record the useful limits instead of declaring an unsupported improvement.
   Hand SEP 12 the method, permitted options, restart/source contracts, fixture
   bundle and mandatory actual-TDSE requalification protocol.

Phases 1–3 are exploratory: they carry only the tests needed to trust the
comparison (weights and moments, convergence order, the call ledger, refusals)
and run the fast checks on the changed code. The full verification table of
§7, the gate, the slow lane and the coverage lane apply in phase 4, to the
selected method, before `Final`, with the explicit compiled-backend exception
above.

### 12. Open questions

1. **Q1 — Useful error (phases 1 and 4).** Which fundamental and weak-UV
   observables determine adequacy, and where does the supplied `rtol=1e-2`
   example lie on the measured cost/error curve?
2. **Q2 — Method and order (phases 2–4).** Does PEC's saved call survive its
   stability and audit costs, and is variable order useful at these tolerances?
3. **Q3 — Controller and startup (phase 3).** Which step-ratio limits,
   starting rule and order hysteresis are reliable without excessive regrowth?
4. **Q4 — Nonphysical limits (phases 1 and 3).** How much work comes from
   collars, table intervals and observation planes, and which compatible
   remedies materially reduce it?
5. **Q5 — Multirate value (phase 3).** Is an inexpensive response surrogate's
   residual sufficiently smooth in z near compression to justify sparse calls?
6. **Q6 — Transfer to TDSE (SEP 12).** Does the chosen method retain its order,
   stability and cost advantage with converged atomic sources and their
   cancellation/noise floor? This question does not block standalone stepper
   delivery, but it does block a TDSE production recommendation.
7. **Q7 — Dense observation (phase 3).** Does the Adams-native interpolant
   keep the weak-band and UV errors of exact landing at the canonical output
   planes, and how much work does removing the output landings save?

### 13. References

1. **solprop-tdse public documentation**, revision
   `0d41f529960f17352180732d1928614d6da081c0`,
   `docs/dev/kernel-mappings.md`, *The solprop substep evaluation*;
   `docs/dev/cpu-performance.md`; `docs/model/accuracy.md`; and
   `docs/guide/standalone-solver.md`. Performance table's tested code is
   `5024293`, subsequently integrated. These package-relative public manual
   identities locate the inspected evidence without relying on private notes.
   The short response-call measurements are not full-run forecasts.
2. M. Richter, F. Morales, S. Patchkovskii and A. Husakou, “Simulation of
   spatiotemporal light dynamics based on the time-dependent Schrödinger
   equation,” *Optics Express* **31**, 39941–39952 (2023).
   [doi:10.1364/OE.499406](https://doi.org/10.1364/OE.499406).
   Eq. (4) motivates the second-order PEC comparator; its printed forward-source
   correction is not adopted. SEP 12 retains the physical stability analysis.
3. P. N. Brown, G. D. Byrne and A. C. Hindmarsh, “VODE: A Variable-Coefficient
   ODE Solver,” *SIAM Journal on Scientific and Statistical Computing* **10**,
   1038–1051 (1989).
   [doi:10.1137/0910062](https://doi.org/10.1137/0910062).
   Variable-coefficient/order and Nordsieck-method context; no source code is
   transcribed and no VODE controller is claimed to be implemented here.
4. SUNDIALS, [CVODE mathematical considerations](https://sundials.readthedocs.io/en/latest/cvode/Mathematics_link.html),
   inspected 4 October 2026. Documents variable-order Adams–Moulton and
   nonlinear solution/control; it does not specify a one-call PEC contract.
5. A. Sandu, “A Class of Multirate Infinitesimal GARK Methods,”
   *SIAM Journal on Numerical Analysis* **57**, 2300–2327 (2019).
   [doi:10.1137/18M1205492](https://doi.org/10.1137/18M1205492).
   Order/coupling/stability framework for the optional multirate comparator.
6. SUNDIALS, [ARKODE mathematical considerations, version 7.7.0](https://sundials.readthedocs.io/en/v7.7.0/arkode/Mathematics_link.html),
   MRIStep and multirate error-control sections. Coupling conditions and both
   fast and slow errors must be qualified before adopting a published pair.

## Part II — Implementation record

### Phase 1 — Measure the numerical problem

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0013-expensive-response-steppers-1`; commits `847743b` (acceptance) and
`4460e49`. Not merged.

**Delivered.**

- `benchmarks/sep13/canonical.py`: the canonical run of §2, its instrumentation and its
  metrics.
  - Source calls are counted by wrapping the backend's single source entry point
    (`NumpyRefBackend._nonlinear`).
  - Step limiters are counted by wrapping `stepper.schedule.landing`.
  - The metrics compare all 50 planes with a converged reference run:
    - the complex field with no time or phase alignment, over the whole grid and
      within the ±30 fs physical window;
    - peak power, FWHM duration and energy;
    - UV (90–200 nm) energy and UV-band field.
- `benchmarks/sep13/baseline.py`: the reference ladder and the RK tolerance ladder.
- `benchmarks/sep13/limits.py`: removes the artificial limits one at a time.
- `benchmarks/sep13/perplane.py`: the physics and RK's error, plane by plane.

**Realised configuration.**

- Field grid: 1024 samples at 0.100 fs, a 102 fs window.
- 513 frequency bins and 2048 fine time samples.
- Radially integrated `HE_11` on 33 Kronrod nodes.
- Kerr and the existing plasma response, with no noise.
- The in-stage temporal collar (ADR-025) with a peak power rate of 60 m⁻¹.
- 123 Chebyshev intervals for Φ; 99 of them lie within 1 mm of the two vacuum ends, with
  a median width of 4 µm.
- Hard kinks at 2.1 m (the port, already a forced landing) and 4.2 m.

The pulse compresses from 6 fs to 1.25 fs by the exit. The UV fraction reaches
5×10⁻³ of the energy.

**The reference.** RK (`numpy_ref` IP-DOPRI5) at `rtol` 1e-6, 1e-7 and 1e-8 agrees with
the 1e-8 run to 1.3e-6 and 9.0e-9 in the field: the reference converges. Each full run
takes 3–8 s, so SEP 13 qualifies at full length (§7).

**RK baseline** (full length, 50 planes, 7 calls per attempt; errors are the worst
plane):

| `rtol` | steps | calls | field | peak power | duration | UV energy |
|---|---|---|---|---|---|---|
| 1e-1 | 160 | 1120 | 1.6e-2 | 1.3e-2 | 1.8e-2 | 0.58 |
| 1e-2 | 161 | 1127 | 1.6e-2 | 1.3e-2 | 1.8e-2 | 0.56 |
| 3e-3 | 180 | 1260 | 5.4e-3 | 8.2e-3 | 1.0e-2 | 0.29 |
| 1e-3 | 192 | 1344 | 1.4e-3 | 8.7e-4 | 1.4e-3 | 6.1e-2 |
| 3e-4 | 211 | 1477 | 3.7e-4 | 3.5e-4 | 4.7e-4 | 2.0e-3 |
| 1e-4 | 263 | 1841 | 1.5e-4 | 1.9e-4 | 1.5e-4 | 1.5e-4 |

**What sets the step count (Q4).** At `rtol ≥ 1e-2` the error estimate sets almost
none of RK's steps. Of the 161 steps:

- 65 are capped by the Chebyshev interval width, almost all of them µm-scale steps at
  the choked exit;
- about 45 are the startup ramp, from `dz_init = L·1e-4` growing ×1.1 per step;
- 49 are the save-plane landings, which also cap the mid-fibre steps at 86 mm against
  error-limited proposals of 80–130 mm.

Removing the limits from RK changes the work but not the error at `rtol=1e-2`:

| Change to RK | steps | calls |
|---|---|---|
| none | 161 | 1127 |
| interval-width cap removed | 100 | 700 |
| + `dz_init` of 1 cm | 68 | 476 |
| + only the final plane saved | 62 | 434 |

The field error was 1.57e-2 in each of the 50-plane runs; the last run cannot be scored
at 50 planes.

By the maintainer's decision these RK limits are reported, not changed: the shared
driver and the compiled backends are untouched.

**The useful tolerance (Q1).** `rtol=1e-2` gives about 1.5% in the field, peak power and
duration, but up to 56% in the UV energy, which is mostly accumulated phase error in a
weak band. The field and UV errors fall to 0.14% and 6% at 1e-3, and to 0.04% and 0.2%
at 3e-4. The comparison is reported as calls-versus-error curves (Pareto), with no
target fixed, as the maintainer chose; the target is set at the selection gate.

**Deviations.** Full-length runs are the primary evidence, as accepted. For RK the call
ledger is an instrument wrapped around the backend, not an in-tree counter: the shared
driver is unchanged. The Adams march has its own in-tree ledger (phase 2).

**Verification.** Scripts only. No tests in this phase.

**Limitations.** One machine (the maintainer's Mac). The wall times are indicative only.

**Open questions.**

- Q1 (useful accuracy) moves to the selection gate.
- Q4 (artificial limits for RK) is recorded for the roadmap.

**Handoff.** Every candidate is scored by `benchmarks/sep13/compare.py` against the
`rtol=1e-8` reference, at full length.

### Phase 2 — Fixed-order NumPy comparison

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0013-expensive-response-steppers-1`; commits `4460e49`, `d9c66d7` and
`87d2607`. Not merged.

**Delivered.**

- `src/solprop/stepper/multistep.py`:
  - the frozen `Multistep` specification, covering corrector `none` / `pec` / `pece`,
    order 1–6, a corrector matched to the predictor's order or one order higher, the
    step-ratio cap, the interval cap, equal approach, variable order and dense
    observation;
  - the variable-step algebra: `integrated_weights` (integrated Lagrange weights on the
    actual nodes), `defect_integral` and `milne_factor`.
- `src/solprop/backends/numpy_ref/multistep.py`: `AdamsMarch` and `CallLedger`:
  - the transported derivative history and its re-anchoring by each accepted step's
    propagator;
  - the P, E, C and E stages, with commit or discard, and the AB recheck after the call;
  - equal steps before forced landings, and a history restart at hard kinks and after
    three consecutive rejections;
  - variable order and dense observation (phase 3 items, built here so the comparison
    could use them).
- `NumpyRefBackend`: `_march_to` and `set_initial` delegate to the march; a by-name
  refusal of coupled flows and the `modified` noise model.
- Wiring:
  - `StepperSettings.integrator`, `assemble(integrator=)`, `SimulationSettings.integrator`
    and `integrator=` on `prop_capillary`, `prop_gnlse`, `from_capillary` and
    `from_gnlse`;
  - settings resolution refuses any backend other than `numpy_ref` and an explicit RK
    tableau, and `auto` resolves to `numpy_ref`;
  - the stepper record carries the method in full.
- `docs/dev/kernel-contract.md`: the new field.
- Tests:
  - `tests/stepper/test_multistep.py`: constant-step AB/AM weights, exact moments on
    irregular nodes, 251/720, −19/720, −19/270, the fifth-difference estimator, names
    and refusals;
  - `tests/backends/test_numpy_ref_multistep.py`: convergence order, the exact linear
    flow, the call ledger, refusals, dense observation and variable order.
- `benchmarks/sep13/compare.py`: the method matrix.

**Deviations from the proposal.**

1. **A corrector one order higher.** The proposal compared matched AB*q*/AM*q* pairs
   with Milne's estimate (§4.3). The comparison added the family whose corrector uses
   one more derivative (AB*q* predictor; AM corrector through the new source and *q*
   history derivatives). It accepts the corrected value and controls on the
   predictor–corrector difference, which is the predictor's defect: conservative, with
   exponent 1/(*q* + 1).
   - Richter et al.'s scheme [2] is the *q* = 1 member, `pec1/2`.
   - This family won (below). The matched pairs lost twice over:
     - Their Milne estimate (−19/270 of the difference for order four) badly
       underestimates a defect far from asymptotic. PEC4 at `rtol=1e-1` diverged with
       every scaled norm below 0.6.
     - Their small stability regions met the collar first.
2. **The weak norm.** The Adams norm is scaled by the accepted state before the step.
   Scaling by the larger of the states before and after, as RK does, let a diverging
   state enlarge its own tolerance and pass as accurate.
3. **The split collar.** It is admitted with the Adams march (the proposal did not
   address it). See *Collar* below.
4. **Variable order and dense observation** were built in this phase, to complete the
   comparison.

**Decisions.**

- Accept or reject before PECE's second call.
- AB rechecks its step with the stencil that ends at the new derivative.
- Restart the history after three consecutive rejections.
- Order-change hysteresis of 1.2.
- A step-ratio cap of 2.
- Two equal steps when a forced landing lies between one and two proposals ahead.
- The schema bump is deferred to delivery (the maintainer's decision); a `TODO(SEP 13)`
  marks the field.

**Verification.**

- On the N = 1 soliton over one period, against IP-DOPRI5 at `rtol=1e-12`, the
  observed orders are:
  - AB4 3.7–3.9;
  - PEC4 and PECE4 about 4, down to 2–3 only at n = 160, where the errors near a 1e-7
    floor;
  - PECE3 3; PEC2 2; Richter 2.
- A zero source gives the exact linear flow to round-off, with dense observation too.
- The ledger is 1 call per accepted step for AB and PEC and 2 for PECE, plus one
  startup call.
- Dense observation matches landing to 2.0e-7 at `rtol=1e-7`, which is the two runs'
  own difference.
- Checks run: ruff, basedpyright and the two new test modules (58 pass), plus
  `tests/backends/test_driver.py` and `tests/translate/test_problem.py`. No full gate,
  slow lane or coverage (deferred to phase 4, as agreed).

**Results at full length** (canonical case, 50 planes, worst plane, against RK at
`rtol=1e-8`; "window" is the field within ±30 fs, clear of the collar; "UV" is the
90–200 nm band's complex field; "split" is the per-step collar):

| Method (interval cap off unless RK) | `rtol` | calls | window field | peak power | UV field |
|---|---|---|---|---|---|
| RK, IP-DOPRI5 | 1e-2 | 1127 | 1.5e-2 | 1.3e-2 | 0.75 |
| RK, IP-DOPRI5 | 1e-3 | 1344 | 1.4e-3 | 8.7e-4 | 0.25 |
| RK, IP-DOPRI5 | 3e-4 | 1477 | 3.5e-4 | 3.5e-4 | 4.7e-2 |
| RK, IP-DOPRI5 | 1e-4 | 1841 | 1.5e-4 | 1.9e-4 | 1.3e-2 |
| Richter PEC1/2 | 1e-2 | 140 | 1.3e-2 | 1.9e-2 | 0.32 |
| PEC2/3 | 1e-3 | 297 | 1.9e-3 | 2.3e-3 | 3.1e-2 |
| PEC3/4 | 1e-3 | 384 | 5.4e-4 | 5.8e-4 | 2.5e-2 |
| PEC3/4 | 3e-4 | 440 | 2.1e-4 | 2.0e-4 | 1.1e-2 |
| PEC3/4, split | 1e-2 | 161 | 7.9e-3 | 1.0e-2 | 0.36 |
| PEC3/4, split | 3e-3 | 214 | 2.7e-3 | 3.4e-3 | 0.13 |
| PEC3/4, split | 1e-3 | 291 | 9.4e-4 | 7.4e-4 | 4.3e-2 |
| PEC3/4, split | 1e-4 | 537 | 7.7e-5 | 6.4e-5 | 4.3e-3 |
| PEC4/5, split | 3e-3 | 240 | 1.4e-3 | 1.1e-3 | 0.12 |
| PECE3/4 | 1e-3 | 588 | 6.4e-4 (whole field) | 8.1e-4 | — |
| PECE4 (matched) | 1e-3 | 435 | 2.3e-3 (whole field) | 2.4e-3 | — |
| AB4, split | 1e-3 | 314 | 2.3e-3 (whole field) | 2.4e-3 | — |
| PEC4 (matched) | any | 700–770 | collar-limited | — | — |

At matched accuracy the one-order-higher PEC family needs **4.5–7 times fewer source
calls** than RK:

| Matched accuracy | Adams | RK |
|---|---|---|
| ~1e-3 window field | 291 calls (PEC3/4, split) | 1344 calls |
| ~4–5% UV field | 291 calls | 1477 calls |
| ~1% field | 140–161 calls | 1127 calls |

PECE does not repay its second call. AB is cheaper per step but less accurate.

**The collar (review point and §4.4).**

- The in-stage temporal collar has a peak power rate of 60 m⁻¹. That is a −30 m⁻¹ field
  eigenvalue, inside every stage.
- Matched PEC4 is pinned at a uniform 5.3 mm step (*h*·30 ≈ 0.16) from 0.4 m to the
  end, whatever the density or tolerance.
- PEC3/4, PEC4/5 and PEC5/6 are collar-limited at about 370, 700 and 1340 steps.
- PECE4 is limited near 43 mm. DOPRI5's bound of about 110 mm is never reached.
- The split collar removes the limit. Its first-order splitting error stays in the
  collar: RK's whole-grid field error at `rtol=3e-4` is 1.3e-3 with it, but 3.6e-4 within
  the window, the same as with the in-stage collar. Peak power and duration are
  unaffected.

**Dense observation (§6.4).** On this case it saves almost nothing:

- PEC3/4 with the split collar goes from 161 to 157 calls at `rtol=1e-2` and from 291 to
  288 at 1e-3.
- The steps (~26 mm) are already well below the 86 mm plane spacing.
- The native Adams and the generic Hermite interpolants agree with landing within every
  metric's precision.
- Its value lies with many planes or a tight `n_save`, not with this configuration.

**Variable order (Q2).** It settles on orders 2–3 and never takes four:

- Split collar: 292 calls at `rtol=1e-3`, against 291 for fixed PEC3/4.
- In-stage collar: it falls back to low orders, which tolerate the collar (299 calls
  against fixed PEC3/4's 384); fixed PEC2/3 does as well (297).
- At these tolerances variable order adds little to a fixed order of 2–3. The cause is
  stability, not mismatched forcing. Phase 3's scalar study (`benchmarks/sep13/stability.py`)
  shows the imaginary-axis interval collapsing from 0.43 at PEC3/4 to 0.04 at PEC4/5.
  Meanwhile the Δβh allowed for a 1e-3 local error grows with order, from 0.11 for
  PEC1/2 through 0.44 for PEC3/4 to 0.56 for PEC4/5.

**Limitations.**

- `rtol=1e-1` is unreliable for every Adams method. A 10% error allowed per step
  compounds, and several runs diverged or stopped at `dz_min`. RK at 1e-1 survives only
  because its artificial limits keep its steps short.
- The UV is the weakest observable for every method at loose tolerances.
- The manufactured phase-mismatch test (§7) and the stability sweeps on model spectra
  are not yet written; the collar evidence is from the canonical run.
- Only Kerr and plasma were tried, not a memory response. One machine.

**Open questions.**

- Q1 (useful accuracy): for the maintainer, at selection.
- Q2: answered above, provisionally.
- Q3: startup is 8–10 steps from `dz_init` with growth 2; not yet optimised.
- Q4: the collar is the new limiter for Adams; the interval cap matters for RK too.
- Q5: multirate is not started.
- Q7: answered above for this configuration.

**Handoff.**

- Recommended candidate: one-call PEC with a one-order-higher corrector, fixed order 3
  (`Multistep(corrector="pec", order=3, corrector_order=4, interval_cap=False)`), with
  the split collar.
- Order 2 is the alternative if the in-stage collar is kept.
- Phase 3 still needs:
  - the call budget;
  - a resume test;
  - the manufactured mismatch and stability tests;
  - an `absorber_scheme` keyword on the capillary interface;
  - a decision on multirate.

### Phase 3 — Adaptive candidates (interim)

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0013-expensive-response-steppers-1`; commits `87d2607`, `52de90b`,
`cdb20e1`, `7072028` and `57eb0cc`. Not merged. Interim: the multirate prototype is not
started (Q5 below), and the selection gate (phase 4) needs the maintainer.

**Delivered.**

- **Variable order** (`Multistep(variable_order=True)`) for the one-order-higher
  corrector family. The neighbouring orders' defects come from the same history and
  predicted-state source, at no extra call. Hysteresis is 1.2, and the order never rises
  straight after a rejection.
- **The call budget.** `Multistep(max_calls=...)` raises `ResponseBudgetError`, a
  `ConvergenceError`, before any trial the remaining budget cannot pay for. The error
  carries the last accepted position, its field and the calls spent.
- **Fresh-start resume, tested.** `set_initial` at a positive `z` restarts the history
  at order one and charges its startup call. It agrees with the uninterrupted run to
  1.2e-8 at `rtol=1e-7`.
- **The capillary interface takes `absorber_scheme`.** `prop_capillary` and
  `Simulation.from_capillary` accept it.
- **The interval cap is off by default.** `Multistep.interval_cap` now defaults to
  `False`.
- **Benchmarks.**
  - `benchmarks/sep13/stability.py`: constant-step amplification matrices and the
    scalar mismatched-forcing study.
  - `benchmarks/sep13/limits_adams.py`: the remaining limits on PEC3/4.
  - `benchmarks/sep13/gnlse.py`: the envelope GNLSE control.
  - The canonical benchmarks take the pulse energy.
- **Documentation.**
  - The draft manual page `docs/guide/numerics/adams.md`.
  - A design note in `docs/dev/stepper.md`.
  - A changelog entry.
- **Tests.**
  - The AB(q+1) identity of the weights.
  - Variable order on a soliton: 6.9e-8 against the reference at `rtol=1e-7`.
  - The budget and the resume.
  - The interface keywords.
  - 66 tests pass in the two new modules.

**Findings.**

1. **Stability, quantitatively.** The constant-step intervals reproduce the collar
   limits measured on the full run:

   | Method | interval, negative real axis | predicted (collar field rate 30 m⁻¹) | measured |
   |---|---|---|---|
   | IP-DOPRI5 | 3.31 | — | — |
   | PEC2/3 | 0.55 | — | — |
   | PEC3/4 | 0.30 | about 420 steps | about 370 |
   | PEC4/5 | 0.16 | about 770 steps | about 700 |
   | Matched PEC4 | 0.157 | 5.2 mm | 5.3 mm |
   | Matched PECE4 | 1.28 | 43 mm | 43 mm |

   On the imaginary axis the intervals are 1.00 (IP-DOPRI5), 0.72 (PEC2/3),
   0.43 (PEC3/4), 0.04 (PEC4/5) and 1.18 (PECE3/4).
2. **PEC with a one-order-higher corrector is AB(q+1) in disguise.** At constant step
   its predicted states advance exactly by Adams–Bashforth of order *q* + 1; the
   corrector is an Adams–Moulton output correction. So it has AB(*q* + 1)'s stability
   and AM accuracy at the output, for one call. Tested as an identity of the weights.
3. **Mismatched forcing does not penalise order.** The Δβ*h* allowed for a 1e-3 local
   error per step grows with order: 0.05 (AB2), 0.11 (PEC1/2), 0.29 (PEC2/3),
   0.44 (PEC3/4), 0.56 (PEC4/5). §4.1's concern holds only for Δβ*h* of order one. The
   low orders that variable order settles on are set by the imaginary-axis interval.
4. **Remaining limits on PEC3/4 with the split collar.**
   - The interval cap costs 20–40% of the calls with no change in accuracy. At
     `rtol` 1e-2, 3e-3 and 1e-3 the calls are 195 → 161, 292 → 214 and 355 → 291.
   - A 1 cm first step saves 2–4 calls; dense observation saves 3–4 calls.
   - What remains is error-limited.
5. **2 mJ (maintainer's steer).** The RK reference at 2 mJ converges (1.1e-6 at
   `rtol=1e-6`, 5.0e-8 at 1e-7, against 1e-8). Every run below completed stably under
   the stronger ionisation:

   | Method | `rtol` | calls | window field | peak power | UV-band field |
   |---|---|---|---|---|---|
   | RK | 1e-2 | 1316 | 4.1e-2 | 1.8e-2 | 0.70 |
   | RK | 3e-3 | 1449 | 1.3e-2 | 7.1e-3 | 0.20 |
   | RK | 1e-3 | 1575 | 2.5e-3 | 1.3e-3 | 2.8e-2 |
   | RK | 3e-4 | 1323 | 5.6e-4 | 4.9e-4 | 1.2e-2 |
   | PEC3/4, split | 1e-2 | 232 | 1.9e-2 | 8.7e-3 | 0.25 |
   | PEC3/4, split | 3e-3 | 323 | 8.3e-3 | 1.8e-3 | 5.8e-2 |
   | PEC3/4, split | 1e-3 | 443 | 3.3e-3 | 5.1e-4 | 2.8e-2 |
   | PEC4/5, split | 3e-4 | 594 | 4.6e-4 | 3.5e-4 | 9.8e-3 |
   | PEC3/4, in-stage collar | 1e-3 | 473 | 3.0e-3 | 4.7e-4 | 2.5e-2 |
   | PEC2/3, in-stage collar | 1e-3 | 456 | 4.2e-3 | 7.9e-4 | 1.4e-2 |

   The saving is 3.5–5.5× at loose and medium accuracy, down to 2.2× at the tightest,
   against 4.5–7× at 1 mJ. At matched `rtol` the Adams field error grows more than RK's
   from 1 mJ to 2 mJ; compare errors, not tolerances.
6. **The GNLSE control** (Dudley et al.'s 835 nm supercontinuum, 15 cm, Raman and
   shock, no collar). Field error at 50 planes against RK at `rtol=1e-10`:

   | Method | `rtol` | calls | field error |
   |---|---|---|---|
   | RK | 1e-5 | 19278 | 1.4e-2 |
   | RK | 1e-6 | 30709 | 1.6e-3 |
   | PEC4/5 | 1e-6 | 22643 | 5.5e-3 |
   | PEC3/4 | 1e-6 | 27596 | 1.1e-2 |
   | PECE3/4 | 1e-6 | 55189 | 3.6e-3 |

   Here the Adams methods need about five times RK's steps and **gain nothing** at
   matched error. A chaotic soliton fission demands tolerances near 1e-6, where
   DOPRI5's fifth order and the Adams control's conservative predictor-defect estimate
   both tell.
   - The low-call methods are a tool for an expensive source at moderate accuracy, not
     a replacement for RK. RK remains the default.
   - This is the general-usefulness evidence §7 asked for, and it is negative for this
     case.

**Deviations.**

- The call budget is a `Multistep` field (`max_calls`), not a run keyword
  `max_response_calls`. Only the Adams march charges calls in-tree, so the budget
  belongs to the method that keeps the ledger.
- The interval cap defaults to off for the Adams march only. RK is unchanged.

**Not done.**

- The multirate prototype (Q5). The existing responses have no expensive part to
  isolate, so a proxy (Kerr cheap, plasma "expensive") would measure the proxy's
  residual, not a TDSE residual's smoothness, which is what Q5 asks. Proposed: defer
  Q5 to SEP 12, with the real atomic residual, and record the deferral at the selection
  gate.
- Manufactured stability tests on a model spectrum inside the test suite. The scalar
  study is a benchmark, not a test.
- The tutorial and the worked example.

**Open questions for the selection gate (phase 4).**

- Q1: the accuracy target.
- The selected method: PEC3/4 with the split collar recommended; PEC4/5 for the tightest
  targets at 2 mJ.
- Whether to keep the alternatives (PECE, AB, the matched pairs, variable order) public
  or prune them.
- The Q5 deferral.
- The schema bump and its solprop-kokkos follow-up.

**Handoff.** Every result above is reproducible with `benchmarks/sep13/` (scripts
write JSON lines; full-length runs take 2–8 s each on `numpy_ref`).

### Phase 3 — Adaptive candidates (closing)

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0013-expensive-response-steppers-1`. Closes the interim report above; no
further phase 3 code.

**Multirate (Q5): deferred to SEP 12, by the maintainer's decision at the selection
gate.** The built-in responses have no expensive part to isolate. A proxy split (Kerr
cheap, plasma "expensive") would measure the proxy's residual, not the smoothness in *z*
of a converged atomic residual, which is what Q5 asks. SEP 12 inherits Q5 with the real
residual.

**Correction to the interim GNLSE finding.** The interim table compared the fixed
orders only. With order 5 and a sixth-order corrector, or with variable order (which
settles there), the Adams method reaches a field error of 2.0e-3 in 24701 calls, where
RK needs about 29400 (1.2× fewer). It is neutral to slightly better in calls, not
worse. At the production tolerance `rtol=1e-6`, matched in tolerance rather than error,
Adams makes 1.5× fewer calls but its error is 3.4× larger, so compare errors, not
tolerances.

**Further evidence gathered for the selection** (in source calls, measured for the
SEP 14 survey):

| Case | Adams' saving at matched error |
|---|---|
| RDW in a capillary, `rtol` near 1e-6 | 2.4× |
| Pure *N* = 4 soliton, tight tolerances | 1.4–2.5×, growing as `rtol` tightens |
| GNLSE supercontinuum | up to 1.2× |
| CtC slice | about 1× |

**Dense observation (Q7), at many planes.** On the canonical case with 501 save planes
at `rtol=1e-3`:
- the Adams march costs 272 calls observing densely, against 506 landing on every plane;
- the window field error is 8.9e-4, the level the tolerance buys and the same as a
  landed 50-plane run;
- the native and Hermite interpolants agree to every printed digit there, since the
  strong field dominates both. The Hermite interpolant is still uncontrolled in general:
  its error is one order below the step's estimate (the manual page gives the witness);
- RK at 501 planes costs 3675 calls at both `rtol` 1e-2 and 1e-3, because landing,
  not accuracy, sets its step. RK dense output is SEP 14's.

At 50 planes, dense observation saves only 3–4 calls, because the Adams steps are
already shorter than the plane spacing.

**Wall time is not calls.** Timed interleaved on an Apple M5 Max, with the load sampled (4.8–6.6
on 18 cores; spreads 2–9%):

| Case | Backend | Source evaluation | Linear work per RK step |
|---|---|---|---|
| RDW | `numba_cpu` | 0.52 ms | 1.31 ms |
| RDW | Kokkos CPU | 0.70 ms | 2.63 ms |
| CtC, reduced | `numba_cpu` | 1.93 ms | 0.34 ms |

On the compiled backends a step's fixed work is comparable to, or larger than, a source
evaluation. A method taking three to seven times more steps can therefore lose part or
all of its call saving: the estimates are 1.2–1.4× faster on RDW and about 0.85× on
the CtC slice. They are estimates until the compiled ports exist. SEP 14 builds the
ports and measures run time against precision.

### Phase 4 — Select and deliver

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0013-expensive-response-steppers-1`; commits `ea99afd`, `a5f9dcf` (merge
of `main`), `9fb8fb0`, `70a8596`, `e21e4e4`, `14444df`, `fc29247`, `5d2e4a3` and
`778b515`. solprop-kokkos: branch
`feature/sep-0013-expensive-response-steppers`, commit `d9fcc31`. Neither is merged:
the merge is the maintainer's.

**Selection-gate decisions** (taken with the maintainer, 2026-10-04):

1. **D1, accuracy (Q1).** No single target. The evidence is reported as Pareto curves of
   calls against error; for expensive runs `rtol=1e-3` is the recommended start.
2. **D2, method (Q2).** `Multistep()`: one-call PEC with a corrector one order higher,
   variable order up to five (corrector six). It matched the best fixed order in every
   case. PEC3/4 remains available as `Multistep(order=3, variable_order=False)`.
3. **D3, pruning.** PEC, `variable_order`, `dense`, `max_calls`, `interval_cap` and
   `equal_approach` are public. PECE, explicit Adams–Bashforth and the matched pairs lost
   everywhere; they survive only as `ComparisonMultistep`, for the benchmarks.
4. **D4, collar.** The in-stage collar stays the default with Adams. The per-step
   collar is allowed, and documented as safe only when the collar holds no field.
5. **D5, multirate (Q5).** Deferred to SEP 12 (phase 3, closing).
6. **D6, budget.** `max_calls` lives on `Multistep`, a recorded deviation from the run
   keyword `max_response_calls` of §5.
7. **D7, norm.** The Adams error norm is scaled by the accepted state before the step, a
   deliberate difference from RK's norm (ADR-037).
8. **D8, dense output.** ADR-036 amends ADR-009 for the Adams march only; SEP 14 extends
   dense output to RK.
9. **D9, stepper record.** The method has its own field, `StepperRecord.method`, with
   output schema 9, rather than a stopgap inside `tableau`.
10. **D10, versions and merge.** Kernel contract version 16 with the one-line
    solprop-kokkos change; the full gate, slow lane and coverage; then the merge and
    `Final`, under the compiled-backend exception of §9.
11. **D11, handoff.** SEP 12 receives variable-order PEC, the call ledger and the
    budget, with the requalification protocol below.

**Delivered.**

- **The public method.** `Multistep(order=5, variable_order=True, dense="off",
  max_calls=None, interval_cap=False, equal_approach=True, max_growth=2.0,
  min_factor=0.2, safety=0.8)`, named for instance `adams-pec5/6-variable`.
- **The kernel contract, version 16.** `StepperSettings.integrator`. The driver refuses an
  integrator on any backend other than `numpy_ref` (`auto` resolves to it) and together
  with a tableau other than DOPRI5. `numba_cpu` and solprop-kokkos refuse it by name.
- **Output schema 9.** `StepperRecord.method` records the stepping method. It enters
  the configuration hash from schema 9 only, so schema-8 files keep their hashes; they
  read and verify as before and are refused for resume.
- **Interfaces.** `integrator=` on `prop_gnlse`, `prop_capillary` and the simulation
  constructors; `absorber_scheme=` on the capillary ones.
- **Records.** ADR-036 (dense observation) and ADR-037 (the method).
- **Docs.**
  - The manual page `docs/guide/numerics/adams.md`, with the Hermite and per-step
    collar caveats of dense observation.
  - The tutorial *Fewer source calls* (`plot_10_low_call_stepping.py`, 8 s): 1344
    against 292 calls, field difference 1.5e-3.
  - The stepper, kernel-contract, backend-status and output pages, the structure
    figures and the changelog.
- **Tests.**
  - The weights, constants and AB(*q* + 1) identity.
  - Convergence order, exact linear flow, the call ledger, refusals, dense observation,
    variable order, the budget and resume.
  - A history restart at an interior hard kink.
  - The slow full-length qualification `tests/driver/test_low_call_canonical.py` (29 s).
    It requires a window field error below 3e-3 against IP-DOPRI5 at `rtol=1e-6`
    (1.3e-3 measured), fewer than 400 calls (292 measured) and at least 3.5 times fewer
    calls than IP-DOPRI5 at the same `rtol`.

**Qualification of the selected method.** The canonical case, 4.2 m, at full length,
against the `rtol=1e-8` reference (worst plane):

| Run | calls | window field | peak power | UV-band field |
|---|---|---|---|---|
| IP-DOPRI5, `rtol=1e-3` | 1344 | 1.4e-3 | 8.7e-4 | 0.25 |
| `Multistep()`, per-step collar, `rtol=1e-3` | 292 | 1.3e-3 | 1.7e-3 | 5.4e-2 |
| `Multistep()`, per-step collar, `rtol=1e-2` | 140 | 1.3e-2 | 1.9e-2 | 0.32 |
| IP-DOPRI5, `rtol=1e-2` | 1127 | 1.5e-2 | 1.3e-2 | 0.75 |

At matched field error that is 4.6–8× fewer calls.

At 2 mJ the saving is 2.2–5.5× (phase 3, interim); GNLSE, RDW, soliton and CtC above.

**Deviations.**

- `max_calls` on the method (D6).
- No `poe bench` ledger entry. The RK hot path is unchanged, and the Adams march runs
  on `numpy_ref` only, where the ledger has no stepping entry; the measurements are in
  phases 1–3.
- No parity tests: the compiled backends refuse the method, under §9's exception.

**Gate** (the maintainer's Mac, load 4–10 on 18 cores):

- `poe check` at `778b515`: 12157 passed, 28 skipped.
- `poe cov` at `5d2e4a3`: 100% line and branch coverage. The first run reached 99.94%;
  the missing refusals, failure restart, order-one and schema-8 resume tests were
  added, and one Literal check, which the test suite's type checking always reaches
  first, is excluded with its reason.
- `poe test-slow`: 137 passed, 1 skipped (the Kokkos comparison, run in the next lane).
- `poe test-slow` with `SOLPROP_TEST_BACKEND=kokkos`: 139 passed.
- The full-length qualification test, re-run after the last change to the march:
  passed in 18 s.
- The strict docs build: clean.

**Open questions, moved on.**

- **To SEP 14** (accepted): the `numba_cpu` and Kokkos ports with optimisation, and the
  wall-time study against precision; whether Adams should become a default; RK's
  artificial step limits (interval cap, growth cap, first step); RK dense output; the
  per-step collar's order; the choked-end tables.
- **To SEP 12**: Q5 (multirate) and Q6 (transfer to the TDSE).
- **Roadmap**: the compiled ports, as SEP 14's item.

**Handoff to SEP 12.** Use `Multistep()` with `max_calls` and the call ledger. Before a
production recommendation, requalify on the actual TDSE source: the convergence order
on a short window, the call count against IP-DOPRI5 at matched error on the canonical
case, and the stability of the history under the source's noise floor. Fresh-start
resume charges one startup call. The reproducible evidence is in `benchmarks/sep13/`.

## Part III — History

- **2026-10-04:** Draft created as the standalone numerical prerequisite to
  SEP 12, following updated atomic-response timings and the maintainer's
  canonical helium example. Draft / Not reviewed; no implementation begun.
- **2026-10-04:** Reviewed by John Travers (human) and Opus 5.5 (AI); review
  status `Reviewed`. The AI review re-derived the Adams weights, error
  constants, Milne factor, fifth-difference estimator and budget arithmetic,
  and corrected against revision `2c7471a`: table-interval caps (a width, not
  an edge), the save-plane reset (Numba FSAL only), the `steady_fill` port as
  an existing hard kink, ADR-009, and the missing call counter, limiter record
  and keywords. It added AB's a-posteriori recheck, phase-mismatched
  oscillatory forcing, equal steps before a forced landing, PECE rejection
  before its second call, the transport-conditioning wording and a
  NumPy-local multistep loop leaving the shared driver unchanged. At the
  maintainer's request: §6.4 analyses interpolation options (generic and
  stepper-native, both to be implemented, selection unchanged); resume is
  fresh-start only; the primary evidence is full-length with convergence
  checks (the canonical case runs in seconds); phases 1–3 carry only the tests
  needed to trust the comparison. Q7 added.
- **2026-10-04:** Accepted by John Travers. Part I is frozen.
- **2026-10-04:** Implementation started: phase 1 by Opus 5.5 (AI) on
  `feature/sep-0013-expensive-response-steppers-1`.
- **2026-10-04:** Phases 1 and 2 reported; phase 3 reported as interim (multirate not
  started; selection gate pending), including the maintainer's 2 mJ qualification and
  the GNLSE control.
- **2026-10-04:** Phase 3 closed (multirate deferred to SEP 12) and phase 4 reported:
  the selection-gate decisions D1–D11 with the maintainer, `Multistep()` delivered with
  kernel contract 16, output schema 9, ADR-036 and ADR-037, the manual page, the
  tutorial and the full-length qualification test. Final, under the compiled-backend
  exception of §9, whose ports SEP 14 takes up. Archived.
