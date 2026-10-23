# SEP 14 — Step efficiency across routes: limits, collars, tables and dense output

| | |
|---|---|
| **SEP** | 14 |
| **Title** | Step efficiency across routes: limits, collars, tables and dense output |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-04) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-10-04 |
| **Resolution** | 2026-10-04: accepted by John Travers after the GPT-6 (AI) and human reviews and the author's responses |
| **Requires** | [SEP 13](sep-0013-expensive-response-steppers.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI), for John Travers. Phases 1–6 on `feature/sep-0014-stepper-efficiency-1`; solprop-kokkos `feature/sep-0014-stepper-efficiency-1` (`ec4b67b`) |
| **Manual pages** | [Low-call stepping for expensive responses](../../../guide/numerics/adams.md); [Performance tuning](../../../guide/performance/tuning.md) (which stepper); [The solver](../../../guide/numerics/solver.md) (save planes); [Output](../../../guide/results/output.md) (schemas 10 and 11) |

## Part I — Proposal

### 1. Abstract

SEP 13's comparisons ran every candidate stepper at full length on several capillary
cases. That exposed limits on solprop's existing Runge–Kutta march that have nothing to do
with accuracy:

- a step cap at the operator table's interval widths;
- a slow startup ramp under the 1.1 growth cap;
- a save-plane landing per output plane;
- a table builder that tiles choked fibre ends at its width floor;
- a temporal collar whose two schemes each fail somewhere.

It also showed where SEP 13's low-call Adams methods help a moderately expensive
response and where they do not. This SEP proposes, measures and qualifies the
changes that follow, on every backend:

- removing the interval cap and giving the start-up a defined first-step rule and a
  growth cap that do not throttle the first steps, on the diagonal-flow routes;
- an opt-in RK dense output, DOPRI5's own continuous extension;
- a diagnosed and repaired table at choked ends;
- a bounded study of an Adams-consistent treatment of the temporal collar;
- porting SEP 13's Adams method to the compiled backends, optimised and measured as
  wall-clock runtime against precision. That measurement decides, per problem class,
  where Adams is recommended, and whether any route's default should change.

Success is a measured accuracy that meets the criterion of §7 at fewer source calls
and, on the compiled backends, less wall time. It is shown on the canonical, RDW, CtC,
GNLSE and soliton cases. Every existing default trajectory either stays unchanged or
meets that criterion, and the change is documented.

### 2. Motivation and scope

**What the measurements show.** Every number below is a full-length run (or the stated
slice) on `numpy_ref`, scored at every saved plane against the same case at a tight
tolerance. The scripts are `benchmarks/sep13/` and `benchmarks/sep14/`; SEP 13's Part II
records the canonical case in full.

*The canonical case* (SEP 13 §2: 4.2 m helium gas-flow capillary, 6 fs pulse, 50 planes).
Here the existing limits, not the error estimate, set RK's steps:

| IP-DOPRI5 variant | calls at `rtol` 1e-2 | calls at 1e-3 | worst-plane field error at 1e-2 / 1e-3 |
|---|---|---|---|
| as today (interval cap, growth 1.1) | 1127 | 1344 | 1.57e-2 / 1.36e-3 |
| interval cap removed | 700 | 917 | 1.57e-2 / 1.36e-3 |
| cap removed, growth cap 2 | 413 | 630 | 1.57e-2 / 1.36e-3 |
| cap removed, growth cap 5 | 392 | 616 | 1.57e-2 / 1.36e-3 |
| cap removed, growth 2, two planes (dense-output bound) | 357 | 511 | end plane 1.6e-2 / 9.4e-4 |
| IP-Tsit5 | 1239 | 1358 | 1.80e-2 / 1.78e-3 |

- At 2 mJ the same change takes RK from 1316 to 602 calls at `rtol=1e-2`, with no change
  in error.
- At `rtol=1e-3` it takes RK from 1575 to 861 calls; the error moves from 2.5e-3 to
  3.4e-3.

*M2 RDW* (1.5 m helium gradient, four modes, `rtol` 1e-5/1e-6, full length). This case
is error-limited:

| Method | calls at `rtol` 1e-5 | calls at 1e-6 | field error at 1e-5 / 1e-6 |
|---|---|---|---|
| IP-DOPRI5 | 8176 | 13188 | 1.5e-4 / 2.2e-5 |
| IP-DOPRI5, cap removed, growth cap 2 | 8036 | 13055 | 1.5e-4 / 2.2e-5 |
| IP-DOPRI5, two planes | 7861 | 12943 | end plane only |
| IP-Tsit5 | 9422 | 14924 | 1.6e-4 / 1.7e-5 |
| Adams PEC3/4 (SEP 13) | 6425 | 11552 | 2.8e-5 / 2.2e-6 |
| Adams PEC4/5 (SEP 13) | 5709 | 9202 | 2.1e-5 / 2.0e-6 |
| Adams PEC5/6 | 5385 | 8059 | 2.2e-5 / 5.0e-6 |
| Adams, variable order up to 5 (SEP 13) | 5405 | 8328 | 2.3e-5 / 3.4e-6 |
| Adams PEC4/5, split collar | 5743 | 9244 | 7.5e-4 / 4.6e-4 |

- The step-limit changes are neutral here, and nothing is lost.
- At matched error PEC4/5 needs about **2.3× fewer calls**: 5709 for 2.1e-5 against RK's
  13188 for 2.2e-5. Its wall time on the reference backend is also halved (31 s against
  62 s).
- Variable order climbs to the high orders here, matching the best fixed order. On the
  canonical case it settled at orders 2–3, also the best there. That makes it the
  robust choice when the problem class is not known in advance.

*CtC reduced* (16 µm hydrogen capillary at 40 bar, Raman, two modes; 1 cm slice, so
startup weighs more than in the full 10 cm):

| Method | calls at `rtol` 1e-5 | calls at 1e-6 | field error at 1e-5 / 1e-6 |
|---|---|---|---|
| IP-DOPRI5 | 658 | 819 | 4.5e-6 / 2.7e-7 |
| IP-DOPRI5, growth cap 2 | 441 | 651 | 4.9e-6 / 2.8e-7 |
| IP-Tsit5 | 700 | 875 | 5.3e-6 / 3.4e-7 |
| Adams PEC3/4 | 425 | 557 | 6.2e-6 / 1.8e-6 |
| Adams PEC4/5 | 777 | 784 | 3.2e-7 / 4.1e-8 |
| Adams PEC3/4, split collar | 307 | 540 | 4.1e-4 / 2.7e-4 |

- Adams gives at most a factor of about 1.5 at the tightest accuracy (PEC4/5 at 1e-6:
  784 calls for 4.1e-8).
- Here the split collar damages the *physical* window: 2.3e-4 within ±300 fs. RK with
  the split collar gives 4e-5 at the same tolerance; with the in-stage collar, 3e-7.

*The GNLSE control* (Dudley et al.'s 835 nm supercontinuum [6], 15 cm; RK reference at
`rtol=1e-11`). The Adams methods take about five times RK's steps:

| Method | calls (field error) at the `rtol`s run |
|---|---|
| IP-DOPRI5 | 19278 (1.4e-2) · 24647 (4.6e-3) · 30709 (1.6e-3) · 38976 (5.3e-4) · 48314 (1.8e-4) |
| Adams PEC3/4 | 27596 (1.1e-2) at 1e-6 |
| Adams PEC4/5 | 22643 (5.5e-3) · 28872 (1.6e-3) · 36012 (5.3e-4) |
| Adams PEC5/6 | 20174 (5.3e-3) · 24701 (2.0e-3) · 29691 (8.5e-4) |
| Adams, variable order up to 5 | 20155 (5.5e-3) · 24692 (2.2e-3) · 29686 (8.9e-4) |

- **At the production tolerance**, `rtol=1e-6` for both, the variable-order Adams method
  costs 20155 calls for a 5.5e-3 field error. IP-DOPRI5 costs 30709 calls for 1.6e-3.
  Adams is 1.5 times cheaper in calls there, but 3.4 times less accurate: equal `rtol`
  is not equal accuracy across methods.
- In calls at matched error, the high orders are neutral to slightly ahead. PEC5/6
  reaches 2.0e-3 in 24701 calls, where IP-DOPRI5 needs about 29400 (1.2×).
- PEC3/4 is clearly behind.
- On the compiled backends, where an Adams step's fixed work is not negligible (below),
  the wall time is expected to favour RK.

*A high-precision case* (an N = 4 soliton over one period; Kerr only, envelope grid;
reference IP-DOPRI5 at `rtol=1e-13`):

| target error | IP-DOPRI5 calls | Adams PEC5/6 calls | variable order up to 5 | saving |
|---|---|---|---|---|
| ~1e-5 | 6160 | 3201 | 3192 | 1.9× |
| ~7e-8 | 9646 | 6951 | 6947 | 1.4× |
| ~5e-9 | 23898 | 10213 | 10209 | 2.3× |
| ~5e-10 | 37779 | 14995 | — | 2.5× |

The Adams advantage grows as the tolerance tightens on a smooth problem.

*Many save planes* (canonical case, 1 mJ):

- **Landing makes the plane spacing cap the step.**
  - IP-DOPRI5 at 201 and 501 planes gives identical runs at `rtol` 1e-2 and 1e-3: 2268
    and 3675 calls.
  - That is 4.5–7 times the two-plane cost, for accuracy nobody asked for (window field
    error 7.5e-6 and 1.6e-5).
- **Adams with dense observation decouples the cost from the plane count.**
  - At 501 planes it takes 272 calls against 506 landing (`rtol=1e-3`).
  - Its window error is 8.9e-4, the level the tolerance buys, the same as a landed
    50-plane run.
  - The native Adams and the generic Hermite interpolants agree to every printed
    digit.
- RK dense output would turn the 3675 calls into roughly the two-plane count (511).

*Source calls are not run time.* Interleaved ten-step windows on 18 cores, with the
one-minute load average sampled before every window (4.8–6.6 throughout; spreads 2–9%,
up to 16% on one Kokkos variant), split a step into source evaluations and the step's
linear work. "One evaluation" is (split-collar step − linear-only step)/7. The linear-only
step drops the responses, its cost in milliseconds and in evaluations:

| Case | Backend | one evaluation | linear-only step | production RK step |
|---|---|---|---|---|
| RDW | `numpy_ref` | 4.5 ms | 2.0 ms (0.44) | 34 ms |
| RDW | `numba_cpu` | 0.52 ms | 1.31 ms (2.5) | 4.8 ms |
| RDW | Kokkos CPU | 0.70 ms | 2.63 ms (3.8) | 7.4 ms |
| CtC reduced | `numba_cpu` | 1.93 ms | 0.34 ms (0.18) | 12.0 ms |
| CtC reduced | Kokkos CPU | 1.71 ms | 1.64 ms (0.96) | 12.5 ms |
| CtC (full size) | `numba_cpu` | 26.7 ms | 6.6 ms (0.25) | 176 ms |
| CtC (full size) | Kokkos CPU | 15.7 ms | 4.5 ms (0.28) | 100 ms |

On `numpy_ref` the Adams march's own work outside the source is about 1.1 ms a step on
RDW, about half of RK's linear step. On `numpy_ref` the measured full RDW runs at matched
error (about 2e-5) took 62 s with IP-DOPRI5 and 31 s with Adams PEC5/6.

These window costs suggest that on the compiled backends the call advantage shrinks in
wall time, and could vanish on CtC. That is a hypothesis, not a prediction. A ten-step
window is not extrapolated to a complete run ([benchmarks](../../benchmarks.md) §5), and
an Adams/RK overhead ratio measured on NumPy is not a measured ratio on a compiled
backend. Phase 2 replaces it with measurements of an optimised port.

**What this means.**

1. Three of RK's limits are artificial wherever they bind: the interval cap, the slow
   ramp and the plane landings. Removing them cost nothing in any case measured, and
   halved or better the calls on the gas-flow case.
2. The operator table resolves a choked exit with about 100 intervals at its 4 µm width
   floor, within the last 0.4 mm. The interval cap turns these into about 60 RK
   micro-steps. The entrance, also choked, needs only about 12 graded intervals.
3. The temporal collar has two schemes, and each fails somewhere:
   - the in-stage term's rate is a stability limit, which binds the Adams methods;
   - the per-step split is first order, which damages the physical field whenever the
     pulse reaches the collar.
4. The low-call methods win decisively when the source is expensive (SEP 13's target).
   In source calls they are also ahead on a moderate source at Luna's tolerance (RDW,
   2.4×) and on a smooth high-precision problem (up to 2.5×). On a chaotic supercontinuum
   they are about even. Today they run only on `numpy_ref`.
5. On the compiled backends a step's fixed linear work is comparable to a source
   evaluation for the smaller grids, so the call advantage may shrink in wall time.
   Whether Adams should be recommended there, or become a route's default, is a runtime
   question. Phase 2 answers it by measuring an optimised port.
6. With many save planes, landing caps the step at the plane spacing. Dense output
   returns the cost to what the tolerance needs: up to sevenfold here.

**In scope.**

- The RK march's step limits (interval cap, start-up, growth cap) on every backend, for
  the diagonal-flow routes (§4.1).
- An opt-in RK dense output.
- The Φ-table builder at choked ends.
- A bounded study of an Adams-consistent collar treatment.
- Numba and Kokkos ports of SEP 13's selected Adams method, optimised, with a measured
  runtime-against-precision study. The study settles the problem classes Adams is
  recommended for and whether a route's default should change. Any default change is
  decided with the maintainer at phase 2's gate.
- Two bounded studies: a higher-order RK pair for tight tolerances, and the weighted norm
  for weak bands.

**Out of scope.**

- Coupled-flow sections (constant Schur, frozen residual, CF4). They keep today's
  interval cap and growth cap, and fixed-schedule semantics are unchanged.
- GPU qualification beyond the standing policy.

**Full-length runs (maintainer's scope decision, 2026-10-04).** The canonical, RDW,
GNLSE and soliton cases complete in about a minute or less on `numpy_ref`. They may run
at full length in this SEP's qualification scripts and slow-marked tests, as SEP 13's
canonical case did. CtC runs only as slices and ten-step windows. Full-size scientific
examples stay in the manually run gallery. Raw samples, run parameters, code and build
identities, and error arrays are committed beside each summary
(`benchmarks/sep14/results/`).

### 3. Context

- [The stepper design](../../stepper.md) §§2–3 and §6; [algorithms](../../algorithms.md).
- [ADR-004](../../decisions.md): IP-DOPRI5 with a PI controller; growth cap 1.1; step capped
  by the containing Chebyshev interval.
- ADR-005: the cumulative Φ as piecewise Chebyshev in z, an exact linear flow over any
  step.
- ADR-009: steps land on save planes; no dense output.
- ADR-019: fixed schedules.
- ADR-025: the in-stage temporal collar.
- [SEP 13](sep-0013-expensive-response-steppers.md) and its Part II.
- Modules (paths relative to `src/solprop/`):
  - `stepper/schedule.py` (`landing`, `StepTargets`) and `stepper/controller.py`
    (`PIController`, `limit_step`);
  - `backends/driver.py` (`StepDriver._march_to`);
  - `translate/chebyshev.py` (`build`, `MIN_WIDTH_FRACTION`) and `translate/linop.py`;
  - `boundaries.py` (the collar rates);
  - `backends/numpy_ref/multistep.py` (the Adams march).

**Tolerances are comparable in kind with Luna's, not in achieved accuracy.**

The shared form:

- solprop's Runge–Kutta norm has the form of Luna.jl's default weak norm [7]: a global
  2-norm of the error over `rtol` times the larger of the two states' norms.
- `rtol=1e-6` is the default in both.

The differences:

- solprop adds `atol` where Luna takes it inside the maximum.
- solprop reduces over the in-band bins only, which excludes populated bins in the
  spectral collar.
- SEP 13's Adams norm is scaled by the state before the step, not by the larger of the
  two.
- Luna's safety factor is 0.9 against solprop's 0.8, and Luna has no 1.1 growth cap.

Equal `rtol` therefore means a comparable error criterion, not equal achieved accuracy.
Only measured errors compare configurations.

**Why the existing limits exist.**

- **The 1.1 growth cap.** It comes from the reject-free, lagged controller designed for
  devices (ADR-004). That controller cannot throw a step away, so it must never propose
  one it would have to.
- **The interval cap.** Its rationale is not recorded beyond ADR-004's decision.
  Measured here, crossing intervals changes no error, because the cumulative Φ is
  evaluated at each stage's own position, interval by interval.
- **Landing.** It was chosen over dense output for device memory and for Luna's FSAL
  ordering trap (ADR-009).

None of these reasons applies to a CPU run with rejection allowed. Each still applies to
some configuration, which is why the proposals below keep each limit as an option.

### 4. Theory

#### 4.1 Interval cap

With ADR-005's cumulative table, a stage at $z_n + c_i h$ reads
$\Phi(z_n + c_i h) - \Phi(z_n)$ from whichever intervals contain the two positions. The
flow is exact for any $h$, whatever it spans. What a step across intervals still
integrates numerically is the *source*'s $z$-dependence: the density, the prefactor and
the ionisation tables. The embedded estimate sees that dependence as it sees any other.

The only remaining reasons to stop at an interval edge are therefore:

- a genuine kink, which is already a landing (`hard_kinks`) and stays mandatory;
- a device kernel that needs both stage positions in one interval. No backend does
  today, and Kokkos is checked in phase 1.

**This argument covers only the diagonal cumulative flow.** On a coupled-flow section
the flow's own error lies outside the embedded estimate, as for CF4 (stepper §2), so a
longer step there could change an error the controller cannot see. Coupled-flow
sections, reject-free runs and device configurations keep today's interval cap and
growth cap.

**Exact tables do not mean resolved sources.** A tabulated linear flow evaluated
exactly does not prove that the RK nodes resolve every smooth source feature: both
embedded formulas can miss a narrow forcing between nodes. The interval-cap gate
therefore includes:

- a localised smooth source witness;
- an independent table refinement;
- endpoint and kink-side tests.

The cap is removed only for the configurations that evidence covers.

#### 4.2 Start-up and growth

With the first step `dz_init = L·1e-4` and growth capped at 1.1, a run needs
$\ln(h^\star/h_0)/\ln 1.1$ accepted steps to reach its natural step $h^\star$. That is
about 45 on the canonical case, before error control matters.

The standard alternative is the starting-step algorithm of [1, §II.4]. Here it is
applied to the **interaction-picture** derivative, never to the lab-frame
$La + N$, whose large linear phase is already integrated exactly.

1. **Scales.** Take $d_0 = \lVert a_0\rVert$ and $d_1 = \lVert g_0\rVert$, where
   $g_0 = N(a_0, z_0)$ is the source at the anchor. Both are in-band weak norms scaled
   by $\mathrm{rtol}$.
2. **Probe step.** $h_0 = 0.01\,d_0/d_1$. The probe is capped at the distance to the next
   landing, so it stays on the arriving side of a hard kink.
3. **Probe.** One explicit Euler probe in the interaction picture, at
   $z_0 + h_0$ with the source's explicit $z$-dependence evaluated there, gives
   $d_2 = \lVert g(z_0+h_0) - g_0\rVert / h_0$.
4. **Step.** $h_1 = (0.01/\max(d_1, d_2))^{1/(p+1)}$, and the first step is
   $\min(100\,h_0, h_1)$, bounded by `dz_min` and `dz_max`.

The edge cases:

- **Zero source.** A zero state or zero source ($d_1 = 0$) falls back to `dz_init`.
- **Failed probe.** A non-finite probe also falls back to `dz_init`, with a warning.
- **Resume.** A resumed run applies the same rule at its resumed position.
- **Explicit settings.** An explicit `dz_init`, or an explicit controller setting, is
  honoured as today.
- **Charged calls.** The rule costs two source calls, both charged to SEP 13's ledger and
  budget when it starts an Adams march. $g_0$ is reused as the first stage, or as the
  first history derivative.

The PI controller then grows by up to its cap while the norm is far below one; the cap
rises to 2 for runs with rejection allowed (Q1). The landing `stretch` stays 1.1 and is
not tied to the growth cap: stretching by 2 could multiply an order-five defect by 32.
`reject_free=True` keeps the 1.1 growth cap, because an overshoot cannot be undone.
The start-up rule and the growth cap are measured separately.

#### 4.3 Dense output for RK

**What it saves.** Landing on every plane costs at most one extra step per plane. The
benefit is largest when planes are dense relative to the natural step: about 120 calls
of 630 on the canonical case at `rtol=1e-3`, and nothing measurable on RDW.

**The options** (SEP 13 §6.4 for the Adams case):

- **DOPRI5's own fourth-order continuous extension** [3]. Its local error is
  $O(h^5)$, the order of the embedded estimate that controls the step. The accepted
  step's stages are contracted into the extension's polynomial coefficients when the
  step is accepted, so no raw stage needs keeping until the observation. **This is the
  RK dense output.**
- **A generic cubic Hermite interpolant in the interaction picture.** It uses the state
  and the stage derivative at both ends. Its $O(h^4)$ local error is one order below
  what the step's estimate controls, so it can miss the requested accuracy in the
  *strong* field too, not only in a weak band.

  The independent review's witness shows this: $u' = x^3$ on $[0, h]$. Both DOPRI5
  quadratures integrate the endpoint exactly, so the embedded defect is zero, while the
  Hermite midpoint value is 0 against an exact $h^4/64$. Hermite therefore stays only as
  an explicitly uncontrolled reference observation, labelled so. SEP 13's
  `dense="hermite"` carries the same caveat.

Either one interpolates $v = U(z_n, z)a(z)$ and applies $U(z_o, z_n)$. Plane positions
stay out of the controller's history, and restartable checkpoints still land exactly.

Even the native extension leaves a weak band at an interpolated plane uncontrolled by
the step's norm. Dense output therefore stays opt-in, and interpolated planes are
labelled. The combinations admitted, labelled and refused are fixed in §5.

**Many planes make it more than marginal.** At 501 planes on the canonical case landing
costs IP-DOPRI5 3675 calls, where the tolerance alone needs about 511 (§2). The same
measurement on the Adams march shows the interpolated planes as accurate as the landed
ones at that tolerance. That is evidence for the Adams interpolant only. The RK
extension is qualified on its own: at matched interior-plane error, with nonzero
sources and a varying linear flow.

#### 4.4 Tables at a choked end

The steady-flow pressure near a choked exit varies steeply, and the density integral
behind Φ inherits that. A bisecting Chebyshev builder resolves it only by halving down to
its floor. It then tiles the remainder of the region at the floor: on the canonical case,
about 100 intervals of $1\times10^{-6}$ of the 4.41 m span.

`translate.chebyshev._split` already bisects geometrically towards an isolated
difficult endpoint. A band of about 100 unresolved floor-width intervals therefore has a
cause that must be found before any remedy is chosen. At the floor `_split` accepts an
unresolved interval, so fewer intervals alone is not an accuracy result.

Phase 5 first diagnoses:

- which sampled function fails;
- whether it fails on the coefficient tail or on off-grid residuals;
- how sensitive it is to sampling precision, tolerance and width floor;
- a reflected entrance/exit witness.

Only then does it choose between two remedies:

- **Repair the refinement**, so that widths shrink geometrically at the exit as they do
  at the entrance.
- **Change variables**, $u = \sqrt{z_e - z}$. This is used only to *construct* a
  certified representation in the existing affine-$z$ Chebyshev basis, so that the seam
  is unchanged. Evaluating in a mapped coordinate would need a Jacobian and a seam
  change, which this SEP does not propose.

Acceptance requires continuity and bounded errors in both $\Phi$ and
$\Phi(z_2) - \Phi(z_1)$, including the power-normalisation contribution, before any
golden data are updated.

#### 4.5 The temporal collar

**The in-stage term (ADR-025).** It adds $-\alpha(t)E(t)/2$ to every source evaluation.
Its peak rate is a negative-real eigenvalue, which bounds every explicit method's step by
$r/(\alpha_{\max}/2)$, with $r$ the method's real stability interval.

The measured intervals (`benchmarks/sep13/stability.py`):

| Method | $r$ |
|---|---|
| IP-DOPRI5 | 3.31 |
| PEC2/3 | 0.55 |
| PEC3/4 | 0.30 |
| PEC4/5 | 0.16 |
| PECE3/4 | 1.93 |

The peak power rates are 60 m⁻¹ (canonical), 168 m⁻¹ (RDW) and 2520 m⁻¹ (CtC).

**The per-step split** $e^{-\alpha h/2}$ is first order. Its attenuation substep is
unconditionally stable, but the composed explicit method is not thereby
unconditionally stable. Its error is harmless only where the collar holds no field.
That holds for the canonical case, and fails for RDW and CtC, whose broadened fields
reach the collar.

**A symmetric split** is not a drop-in fix for the Adams march, as the independent
review showed (Q7, Q8):

- **Q7: the history.** Applying half factors around the existing recurrence, with
  history stored from half-collared states, leaves a second-order *local* defect even
  for commuting scalar flows. The witness is $u' = b\,u - a\,u$ with exact history: the
  one-step error over $h^2$ is 0.061–0.062 as $h$ halves. A derivative stored at a
  half-collared state is not a derivative on the collar-free subproblem's trajectory,
  and the half factor changes with each trial's step.
- **Q8: the splitting error.** Strang's error involves the collar's commutators with the
  whole remaining field: the linear flow and the nonlinear source. The witness is
  $u' = -au + i\gamma|u|^2u$: an exact inner Kerr solve between two half attenuations
  still has splitting error. No embedded estimate measures it.
- **Q8: the cost.** Time-domain damping needs a transform pair per step, and a changed
  endpoint invalidates FSAL.

For the Runge–Kutta march none of this is needed: the in-stage collar is well inside
DOPRI5's stability interval on every case measured. The symmetric split is therefore
**a bounded study** (phase 4) with an exit criterion. It must:

- derive an Adams history transformation that is consistent with a split collar: its
  source identities and its variable-step recurrence;
- pass the review's witnesses, a three-level convergence test with non-commuting
  operators, and rejection, order-change, hard-kink and restart tests that preserve
  accepted history;
- give the splitting error an explicit control, either a collar step cap
  $h\,\alpha_{\max}/2 \le \theta$ with $\theta$ a documented accuracy setting, or a
  separate estimate, with its calls, transforms and buffers in the cost and admission
  contract.

If it does not pass, the in-stage collar remains the Adams default, and the per-step
split remains admitted only for runs whose collar holds no field.

**The rate profile itself** (`α = 12.6(1 − W)/ℓ`) could be widened and lowered for the
same integrated attenuation. That trades window size for stability, and it is a physical
setting rather than a stepper one.

#### 4.6 Where the low-call methods pay

A step of SEP 13's PEC*q*/(*q*+1) costs one source call plus the transport of *q* + 1
stored derivatives, which are elementwise products. IP-DOPRI5 costs six calls with FSAL,
or seven on `numpy_ref`.

PEC*q*/(*q*+1) advances its predicted states by Adams–Bashforth of order *q* + 1, and its
stability region is that formula's (SEP 13 Part II). It wins when two things hold
together:

- the source dominates the per-step cost;
- the tolerance is moderate enough that Adams' larger step count, typically 4–5× RK's,
  is outweighed by the six- to sevenfold saving per step.

**The measurements place the cases:**

- **SEP 13's expensive TDSE target:** wins.
- **The canonical case at moderate accuracy:** 2.2× fewer calls than the *improved* RK
  (291 against 630 at `rtol=1e-3`).
- **RDW at Luna's tolerance:** 2.3×.
- **CtC:** up to 1.5×.
- **Chaotic supercontinuum generation at `rtol` ≈ 1e-6:** at best 1.2×, and only with
  order 5.
- **A smooth high-precision soliton:** 1.4–2.5×, growing with precision.

**In wall time the per-step work also counts.** With $C_N$ one evaluation, $C_L$ RK's
linear work per step and $C_A$ the Adams step's own work, the costs are

$$T_{\rm RK} \approx n_{\rm RK}\,(6C_N + C_L), \qquad
T_{\rm A} \approx n_{\rm A}\,(C_N + C_A),$$

with $n_{\rm A}/n_{\rm RK}$ typically 3–7. Adams therefore wins only where $C_N$ dominates
$C_A$ by enough.

The measured split (§2) has $C_L/C_N$ from 0.2 to 3.8 on the compiled backends. $C_A$
on `numpy_ref` is about $C_L/2$.

An optimised Adams step can reduce $C_A$. The dependency order is fixed: the predicted
field must exist before its source does, and acceptance needs a completed norm. So a
legal schedule fuses only within these stages:

1. **Predictor pass.** Form $v_P$ from the history, transported to the anchor, in one
   pass over the field, and the predicted lab field $U\,v_P$ with the step's single
   propagator exponential.
2. **Source evaluation.** The FFT, the response and the projection, as today.
3. **Corrector pass.** Form $v_C$, the defect and its deterministic norm partials (and
   the neighbouring-order estimates of variable order) in one pass.
4. **Reduction and decision.** Reduce the norm partials and accept or reject.
5. **Conditional commit.** On acceptance only, write the accepted state, rotate the
   ring by index and transport the stored derivatives. A rejection leaves the accepted
   state and every history entry untouched.

The memory ledger counts everything this needs: the ring, the predictor, corrector and
source scratch, the dense-output and rollback storage. The legal schedule is benchmarked
against the same arithmetic on `numpy_ref`.

#### 4.7 Higher-order RK and weak-band norms (studies)

**Higher-order RK.** At the tight tolerances that the chaotic and multimode cases need,
Hairer's DOP853, an 8(5,3) pair with a seventh-order continuous extension [1, 8], or
Verner's efficient 6(5)/7(6) pairs [5] may reduce the call count further. If phase 6
selects one, it adds that method's primary reference. IP-Tsit5, measured here, never
beat IP-DOPRI5.

**Weak-band norms.** The weak UV band was the least accurate observable for every method
at loose tolerances. The weighted norm (stepper §3) exists, but no weighting has been
qualified. A bounded study should measure the cost of a band-weighted norm that holds
the UV-band field to a target, against tightening `rtol`.

### 5. Proposed interface

Defaults change only where the measurements meet §7's criterion. The changes:

- **Interval cap off.** `SimulationSettings(interval_cap=None)` resolves to off on the
  diagonal-flow routes and on for coupled-flow sections; `True` or `False` overrides it.
  The Adams march already defaults to it off.
- **Start-up.** `dz_init=None` selects the first-step rule of §4.2; an explicit value is
  honoured as today.
- **Growth cap.** `PIController.max_factor` defaults to 2 for runs with rejection
  allowed, and stays 1.1 for `reject_free=True` and for coupled-flow sections. The
  landing `stretch` stays 1.1.
- **RK dense output, opt-in.** `SimulationSettings(dense_output="off" | "native")`, `"off"`
  by default; `"native"` is DOPRI5's continuous extension (or Tsit5's own, if
  qualified). The admission matrix:

  | Combination | Dense output |
  |---|---|
  | Diagonal-flow routes with the in-stage collar | admitted |
  | `split_step` collar | refused: the pre-collar endpoint's derivative is not that of the saved, collared field; until derived |
  | `modified` noise | refused |
  | Fixed-schedule batches | refused |
  | Coupled flows | refused |

  Source statistics at an interpolated plane are computed from the interpolated field
  and labelled. The end of the run and any requested checkpoint are landed on exactly.
  The same matrix applies to SEP 13's `Multistep(dense=...)` from phase 3, which adds
  the `split_step` refusal there.
- **The collar study.** No new public value unless phase 4's study succeeds; then
  `absorber_scheme="strang"` joins `"nonlinear"` and `"split_step"`, with its
  accuracy setting.
- **No new keyword for the tables.** The repaired choked-end table is internal.
- **Adams on the compiled backends.** `numba_cpu` and solprop-kokkos execute
  `Multistep` with SEP 13's selected configuration: one-call PEC, variable order up to 5,
  optional dense observation and call budget. The SEP 13 refusal is lifted as each
  backend qualifies.

### 6. Implementation strategies

**Shared driver.** The interval cap, the first-step rule and the growth cap live in the
shared `StepDriver` and the stepper scalars, so every backend changes together. The
recommended form reads two new `StepperSettings` fields, `interval_cap` and
`auto_init`; no protected hook is added.

**Versioning (Q15).** Every incompatible merged seam bumps its version, phase by phase:

| Phase | Kernel contract | Output schema and resume key |
|---|---|---|
| 1 | bumped for `StepperSettings.interval_cap` and `auto_init` | the effective interval cap, first-step rule and growth cap enter the stepper record (bumped) |
| 2 | unchanged; Kokkos admission changes | unchanged |
| 3 | bumped for `dense_output` | interpolated planes labelled (bumped), reconciled with SEP 13's method record |
| 4 | bumped only if `"strang"` is admitted | records the collar setting |
| 5 | unchanged: table values only | unchanged |

Each bump comes with its literals, its Kokkos admission check, and a test that the
effective defaults, not just their keywords, are in the configuration identity.

**RK dense output.**

- DOPRI5's continuous extension contracts the accepted step's stages into its
  polynomial coefficients when the step is accepted. That needs five retained
  field-sized buffers per member, against Hermite's two.
- The contraction is done on every CPU backend: on `numpy_ref` and `numba_cpu` in host
  memory, and on Kokkos in device memory, admitted only where the memory ledger allows
  it.
- If compiled support is omitted after all, the explicit backend-parity exception is
  written into this SEP before Final.

**The Φ-table builder.** The repair is a translation change, made after phase 5's
diagnosis (§4.4). Every backend sees the new intervals. Golden data are re-recorded only
after §4.4's continuity and error criteria are met.

**The collar study.** It is derivation and NumPy prototypes only, until its exit
criterion is met (§4.5).

**Adams ports, with optimisation.**

- `numba_cpu` reuses its existing source kernels. It needs a history ring of field-sized
  buffers (*q* + 1 of them) and the scalar weights from `stepper.multistep`.
- Kokkos holds the history on the device and computes the weights on the host.
- Both take the accept/commit/discard lifecycle of SEP 13 §6.2.
- Parity is to `numpy_ref` within summation bounds.
- **The step's own work is the optimisation target**, with the legal schedule of
  §4.6: predictor pass, source, corrector and norm pass, decision, conditional commit.
  Each optimisation is kept only with a benchmark-ledger entry showing its gain.

**Cost.**

- All the RK changes reduce or keep the step count.
- RK dense output adds five buffers.
- Adams adds its ring of *q* + 1 buffers and its scratch, each counted in
  `memory_ledger()`.

### 7. Verification and validation

- **Defaults, and the acceptance criterion (Q17).** For every changed default, compare
  the trajectories at full length:
  - the canonical case at 1 and 2 mJ, and RDW;
  - a CtC slice, since full CtC is too long for a lane;
  - the GNLSE control;
  - the existing worked examples (run by hand).

  Errors are scored against tight references at every plane, and each reference's own
  convergence is shown on its `rtol` ladder.

  **The criterion.** At every `rtol` of the ladder, the new default's worst-plane
  window field error and UV-band field error may not exceed 1.25 times the old
  default's, or it must reach the old default's errors at fewer calls.

  A local `rtol` does not bound the difference between two adaptive runs, so the
  criterion is on measured errors, not on tolerances. The Luna-witness regressions on
  fixed schedules are unaffected by construction. Adaptive golden arrays are re-recorded
  only after the criterion is met, with the measured change stated.
- **Controller.** Rejection rates and error proportionality on the existing tolerance
  ladders, with the new growth cap and first step.
- **Interval cap.** On the diagonal-flow routes only (§4.1):
  - steps crossing intervals on a manufactured density with a known integral;
  - a localised smooth source witness;
  - an independent table refinement;
  - endpoint and kink-side tests;
  - a choked end;
  - the coupled-flow sections, checked to keep today's policy.
- **First-step rule.** The zero-state, zero-source, failed-probe, resumed and
  kink-adjacent cases, and its calls in the ledger. Its cost is measured separately from
  the growth cap.
- **Dense output.**
  - The native extension's interpolation order with nonzero sources and a varying
    linear flow.
  - The endpoint identity.
  - Matched interior-plane error against landing.
  - The review's polynomial witness, showing why Hermite is not offered.
  - The admission matrix of §5 and the plane labels.
  - Statistics positions, resume refusal, and the observation position against the
    accepted endpoint.
  - UV-band and weak-band errors reported.
- **Choked-end table.** §4.4's diagnosis, then the interval count, $\Phi$ and
  $\Delta\Phi$ errors against a fine reference, and translation time and memory before
  and after.
- **Collar study.** §4.5's witnesses and exit criterion.
- **Adams ports.**
  - Kernel parity: one step, eight steps, and a short run.
  - The SEP 13 ledger and refusals, on every backend.
- **Runtime against precision** (the deciding measurement, Q16).
  - **Cases:** the canonical case at 1 and 2 mJ, RDW, the GNLSE control and the N = 4
    soliton at full length, within §2's scope decision; CtC on a slice and on its
    ten-step windows.
  - **Tolerances:** four or five values per method, IP-DOPRI5 against Adams, on
    `numba_cpu` and Kokkos CPU, including `rtol=1e-6` for every case.
  - **Metric:** wall time against measured error at the planes.
  - **Hardware:** the Linux workstation, otherwise idle, for the recorded numbers; the
    maintainer's Mac only as a labelled cross-check.
  - **Protocol:**
    - a warm-up run, with thread counts fixed;
    - alternating samples, method by method, with the load average sampled before each;
    - an A/A comparison of each method against itself, giving the uncertainty;
    - a bounded stopping rule: at most ten rounds per point, reported as median and
      interquartile range, and marked inconclusive if the two methods' ranges overlap.
  - **Records:** raw samples, parameters, code and build identities and error arrays,
    committed beside the summaries.
  - **The gate:**
    - A problem class gets the Adams recommendation on a backend only where its curve
      lies below RK's at the accuracies of interest, beyond the A/A uncertainty.
    - Whether any route's default should become Adams is decided there with the
      maintainer, weighing the curves against Adams' refusals (fixed schedules, coupled
      flows, `modified` noise).

### 8. Documentation

- Update the solver manual page (`docs/guide/numerics/solver.md`): caps, start-up, dense output and
  collar schemes.
- Update the low-call stepping page with the compiled backends and the problem-class
  guidance.
- Update [the stepper design](../../stepper.md) §§3 and 6, the kernel contract, backend
  status and the benchmark ledger.
- New ADRs superseding the relevant parts of ADR-004 (growth cap, interval cap) and
  ADR-009 (opt-in dense output).
- Add a decision table to the manual, "which stepper for which problem":

| Problem | Recommended |
|---|---|
| Cheap source, tight accuracy, chaotic dynamics (supercontinuum) | IP-DOPRI5 |
| Moderate source, Luna-level tolerance (RDW, CtC) | Decided by phase 2's runtime curves; until then IP-DOPRI5 |
| Smooth high-precision problems | Adams with variable order, if phase 2's curves confirm the call saving in time |
| Expensive source, moderate accuracy (TDSE) | Adams with variable order; the per-step collar only when the collar holds no field, otherwise the in-stage collar |

### 9. Compatibility and impact

**Changing defaults changes adaptive trajectories.** So far the measured changes are
small. On the canonical case the worst-plane field error was unchanged to three
digits at `rtol` 1e-2 and 1e-3. At 2 mJ and `rtol=1e-3` it rose from 2.5e-3 to 3.4e-3
(1.36 times, at 55% of the calls). On RDW it was within 2%. Each change must meet §7's
criterion before it becomes a default. Results saved by earlier versions are unaffected.

**Records and keys.** Versioning follows §6's table, one bump per incompatible merged
seam. The effective defaults enter the stepper record, so a resume across a change of
default is refused, and a test checks that it is.

**Luna.** Luna.jl evaluates its operator without a table, so it has no interval cap, and
its controller lets a step grow by up to a factor of five [7]. The changes move solprop
closer to it, and the Luna differences page is updated accordingly.

### 10. Alternatives considered

| Alternative | Decision |
|---|---|
| Keep the RK limits and only use Adams | Rejected: the limits cost RK a factor of 2–2.7 on gas-flow cases at no gain, and Adams is not the right scheme everywhere |
| Make dense output the default | Rejected: weak-band accuracy at interpolated planes is not controlled; landing stays the default |
| Split collar as the default for Adams | Rejected: first-order damage to the field on RDW and CtC |
| A symmetric split around the existing Adams recurrence | Rejected: second-order local defect even for commuting flows (review Q7); only a derived history transformation is studied |
| Hermite as the RK dense output | Rejected: one order below the step's error control; misses strong-field accuracy on a polynomial witness (review Q9) |
| Remove the cap on coupled-flow sections too | Deferred: their flow error is outside the embedded estimate; needs its own qualification |
| Lower the collar's peak rate by default | Deferred: a physical setting, measured separately |
| IP-Tsit5 as default | Rejected: never better than IP-DOPRI5 here |
| Port every Adams variant | Rejected: only SEP 13's selected method, variable-order PEC, is ported |
| Make Adams the general default now | Deferred to phase 2's gate: ahead in calls nearly everywhere, but not yet shown in wall time on the compiled backends, and it refuses fixed schedules, coupled flows and modified noise |
| Judge the ports by call counts | Rejected: calls overstate the gain where the step's fixed work is comparable to an evaluation |

### 11. Phases

1. **Step limits (all backends, diagonal-flow routes).**
   - Interval cap off by default.
   - The first-step rule of §4.2 and the growth cap of 2 for rejecting runs.
   - The evidence and criterion of §7.
   - Its version bumps (§6) with the Kokkos follow-up.
2. **Adams on the compiled backends, optimised, and the runtime study.**
   - SEP 13's selected method, with variable order up to 5, on `numba_cpu` and then
     Kokkos CPU, with parity.
   - The fused-step optimisations of §6, each with a ledger entry.
   - The runtime-against-precision curves of §7.
   - The problem-class recommendations, and a decision on whether any route should
     default to Adams; any such default change would be a separate, small Change SEP.
3. **RK dense output.**
   - DOPRI5's continuous extension, contracted at acceptance, on every CPU backend.
   - The admission matrix of §5, also applied to SEP 13's dense observation.
   - A new ADR amending ADR-009.
   - Qualified with many save planes.
4. **The collar study.** The Adams-consistent collar of §4.5, to its exit criterion.
   Implementation follows only if the criterion is met.
5. **Choked-end tables.** The diagnosis of §4.4, then the repair; interval count,
   $\Phi$ and $\Delta\Phi$ errors, translation cost.
6. **Studies.** A higher-order RK pair at tight tolerances, and a band-weighted norm for
   weak bands. Each ends in a recommendation, not necessarily in code.

Phases 1, 3, 4 and 5 are independent of each other. Phase 2 follows SEP 13's delivery
and benefits from phase 1's controller changes.

### 12. Open questions

1. **Q1 — Growth cap.** Is 2 the right default, or should the cap depend on the
   controller's lag and the backend? Settled in phase 1.
2. **Q2 — Interval cap.** Is any backend kernel relying on a stage staying in one
   interval? To be checked on Kokkos in phase 1.
3. **Q3 — Choked ends.** Grading or a change of variable (phase 5)?
4. **Q4 — Collar default per scheme.** Should the Adams march default to an
   Adams-consistent split collar, if phase 4's study yields one (phase 4)?
5. **Q5 — Problem-class guidance.** Where, as wall time against precision on
   `numba_cpu` and Kokkos CPU, does an optimised Adams step beat IP-DOPRI5, and should any
   route default to it (phase 2)?
6. **Q6 — Tight-tolerance RK.** Does a higher-order pair change the supercontinuum and
   CtC costs (phase 6)?

#### Independent review — GPT-6, 2026-10-04

**Scope and recommendation.** Reviewed Part I at `781db1e` against the stepper,
translation, backend and output contracts, SEP 13's implementation at that commit,
and the benchmark scripts. The separation of call count from wall time, retention of
IP-DOPRI5 as the default, and explicit opt-in observation are good choices. **Request
revisions before acceptance:** the collar/history construction and the error contracts
below are not yet sufficiently specified. The questions are additions to Q1–Q6, not
changes to the proposed numerical choices. P1 denotes an accuracy or scope issue to
resolve before acceptance; P2 denotes a design or evidence clarification. Each remains
open until answered by the author and reflected in the proposal.

7. **Q7 — P1: Derive the split Adams history, rather than applying two half factors
   around the existing recurrence (§§4.5, 6).** Owner: author before acceptance;
   phase 4. A stored derivative at a previous half-collared state is not automatically
   a derivative on the current collar-free subproblem's trajectory. The half factor
   also changes with the next trial's step size, including after rejection.

   A small counterexample is already available without dispersion or a nonlinear
   response. In dimensionless distance $x=z/\ell$, split $u'=bu-au$ with $a=1$ and
   $b=1/2$. Give PEC3/4 exact total-ODE history, store each old source as
   $b e^{-ah/2}u_j$, and apply the proposed half damping before prediction and after
   correction. Its one-step error divided by $h^2$ is 0.06077, 0.06167 and 0.06210
   for $h=0.1,0.05,0.025$: a second-order **local** defect, incompatible with a
   second-order global claim. The two scalar subflows even commute. This tests the
   literal half-state construction, not a yet-to-be-derived history transformation.
   Please specify that transformation, its source identities and its variable-step
   recurrence, or defer the Adams combination and qualify Strang for RK first.
   Require three-level convergence with noncommuting operators too, and rejection,
   order-change, hard-kink and restart tests that preserve accepted history.

8. **Q8 — P1: Control or explicitly bound the complete collar-splitting error
   (§§4.5, 7–8).** Owner: author before acceptance; phase 4. Strang's commutators
   involve the remaining **linear and nonlinear** vector field. They do not vanish
   merely because dispersion vanishes. For the dimensionless witness
   $u'=-au+i\gamma|u|^2u$, $u(0)=1$, the exact nonlinear phase after a step is
   $\gamma(1-e^{-2ah})/(2a)$; exact Kerr between two half attenuations gives
   $\gamma h e^{-ah}$. Thus even an exact inner solve, with zero inner error, has
   splitting error. The existing embedded estimate does not measure it, and global
   order is at most two when it contributes.

   What estimator, step cap or explicitly separate user accuracy setting will govern
   this error? An order test alone does not show that tightening `rtol` controls the
   complete solution. Also qualify “unconditionally stable” as a property of the
   attenuation substep, not the composed explicit method. Revisit the claim of only
   one extra multiplication: the spectral state needs transforms for time-domain
   damping, and a changed endpoint invalidates FSAL (`NumbaCpuBackend._fsal` already
   excludes `split_step`). Include those calls, buffers and the endpoint derivative
   needed by dense output in the cost and admission contract.

9. **Q9 — P1: State that Hermite can miss the requested accuracy in the strong field,
   not just in a weak band (§§4.3, 7, 9).** Owner: author before acceptance; phase 3.
   Consider dimensionless $u'=x^3$, $u(0)=0$ on $[0,h]$. Both DOPRI5 quadratures
   integrate the endpoint exactly, so the embedded defect is zero in exact arithmetic.
   Endpoint cubic Hermite gives zero at $h/2$, while the exact value is $h^4/64$.
   The current tableau reproduces this: at $h=1$ the embedded defect is about
   $2.1\times10^{-17}$ and the midpoint error is 0.015625.

   Please make this accuracy limitation explicit for **all** interpolated fields and
   statistics. Choose whether Hermite is an explicitly uncontrolled observation, has
   an independent interpolation check, or is subordinate to native DOPRI5 output.
   Compare the two at matched interior-plane error, rather than using the two-plane
   call count as an accuracy-qualified prediction. Add this polynomial witness and
   nonzero-source, varying-linear-flow interior order tests; Adams agreement on one
   case does not qualify the RK interpolant.

10. **Q10 — P1: Bound the scope of the shared default changes (§§2, 4.1, 5–7).**
    Owner: author before acceptance; phase 1. The exact-flow argument is for diagonal
    cumulative propagation, but `StepDriver` also serves coupled flows and all spatial
    routes. In particular, the stepper design §2 states that CF4 flow error is outside
    the embedded pair; changing the caps or startup can change that error without a
    corresponding controller signal. “Coupled-flow semantics out of scope” does not
    isolate those runs from edits in the shared march. Specify which configurations
    retain their old policy, or extend qualification to the affected routes, coupled
    sections and reject-free/device configurations.

    Even on the diagonal route, exact evaluation of the **tabulated** linear flow does
    not prove that sparse RK nodes resolve every smooth source feature. Both embedded
    formulas can miss a narrow forcing feature between nodes. Keep genuine kinks as
    mandatory landings, and include a localised smooth source witness, independent
    table refinement, and endpoint/kink-side tests in the interval-cap gate. This is
    a request for evidence and a declared accuracy scope, not an argument to retain
    the cap universally.

11. **Q11 — P2: Define the automatic first-step algorithm and its effective settings
    (§§4.2, 5–6).** Owner: phase 1. Which state, derivative and norm enter the estimate:
    lab-frame $La+N$, or the interaction-picture derivative $g$? The former can make
    startup resolve a large linear phase already integrated exactly. The second
    probe must include explicit position dependence and remain on the correct side
    of a hard kink. Specify the tolerance scaling, zero-state/zero-source cases,
    resumed start, bounds and handling of a failed probe. Charge extra source calls
    to the ledger/budget when used with Adams, and state whether an initial derivative
    can be reused. Preserve explicit `dz_init` and controller overrides. Keep landing
    `stretch` separate from `max_factor`: raising a stretch to 2 can multiply an
    order-five defect by 32. The current measurements change growth, not this new
    startup rule; measure its cost separately.

12. **Q12 — P2: Diagnose the choked-end refinement before choosing its replacement
    (§§4.4, 6–7).** Owner: phase 5. `translate.chebyshev._split` already bisects
    geometrically toward an isolated difficult endpoint. A band of roughly 100
    unresolved minimum-width intervals therefore needs a cause, not just a new
    grading rule. Record which sampled function fails, coefficient-tail versus
    off-grid residuals, and sensitivity to sampling precision, tolerance and width
    floor. Check a reflected entrance/exit witness. At present `_split` accepts an
    unresolved interval when it reaches the floor; fewer intervals alone cannot be
    an accuracy result.

    A mapped-coordinate representation also is not automatically translation-only:
    `PiecewiseChebyshev` and the seam evaluate polynomials in affine **z**. Either
    use the change of variable only to construct a certified representation in the
    existing basis, or specify mapped evaluation, the integration Jacobian and the
    corresponding seam change. Require continuity and errors in both $\Phi$ and
    $\Phi(z_2)-\Phi(z_1)$, including the power-normalisation contribution, before
    updating any golden data.

13. **Q13 — P2: Replace the single-pass Adams promise with an executable dependency
    order (§§4.6, 6).** Owner: phase 2. The predicted field must exist before its FFT,
    response and projection produce the new source. Only then can the corrector and
    defect be formed, and acceptance needs the completed deterministic norm reduction.
    A single field pass cannot form the predictor and its source-dependent corrected
    state while also committing history after that global decision. Describe a
    predictor/transport pass, source evaluation, corrector/norm pass and conditional
    commit, fusing only across valid dependencies. Rejections must leave the accepted
    state and every history entry intact. Count the actual scratch, ring, dense and
    rollback storage in `memory_ledger()` rather than reporting only the history's
    $q+1$ fields. Benchmark this legal schedule against the same arithmetic in NumPy.

14. **Q14 — P2: Make the dense-output combination and checkpoint matrix explicit
    (§§5–6, 9, 11).** Owner: phase 3. State the default `"off"`, supported tableaux,
    backend refusals, and interactions with `split_step`, `strang`, noise, source
    statistics, batches and coupled flows. A derivative of the pre-collar endpoint is
    not the derivative of the saved post-collar field. State which combinations are
    refused until derived, and how exact checkpoints force a landing while ordinary
    observations do not. Test the saved-plane labels, statistics positions, resume
    rejection and the distinction between the observation position and the backend's
    accepted endpoint.

    DOPRI5 need not retain every raw stage **until observation**: its accepted stages
    can be contracted into continuous-extension coefficients. Compare that storage
    with Hermite before deciding native dense output is NumPy-only. If compiled
    support is intentionally omitted, write the explicit backend-parity exception
    required for this SEP to become Final.

15. **Q15 — P2: Version each incompatible merged seam, not each release (§§6, 9).**
    Owner: all phase handoffs. Combining the SEP 13 and SEP 14 bumps is safe only if no
    independently usable merged state exposes different contracts under the same
    version. Phases 1, 3 and 4 may change settings, protected hooks or stage/output
    records separately. Name the affected contracts per phase and bump on each
    incompatible transition, with updated literals and Kokkos admission checks.
    Reconcile dense labels with SEP 13's delivered output contract instead of assuming
    a new output-schema change is always needed. Verify the effective defaults are
    serialised into the configuration identity: the existence of a configuration hash
    alone does not prove older resumes will be refused when implicit defaults change.

16. **Q16 — P2: Bring the timing evidence and planned gate into the documented
    qualification scope (§§2, 7, 11).** Owner: author before acceptance; phases 1–2.
    The extrapolated 9.1/6.7 s and 13.9/11.6 s full-RDW timings conflict with
    [benchmarks §5](../../benchmarks.md): a ten-step window is not extrapolated to a
    complete-run cost. An Adams/RK overhead ratio measured on NumPy is not a measured
    ratio on either compiled backend. Retain the cost equation as a hypothesis;
    replace those numerical predictions with actual short-window measurements after
    the port. Use an idle-machine protocol, warmup, controlled threads, alternating
    samples and A/A uncertainty; repeat-until-spread-is-small needs a bounded stopping
    rule, not a success threshold that hides noisy trials.

    Reconcile the full-length requirements with the repository's short-run policy.
    SEP 13 explicitly records its accepted canonical-case exception; do not silently
    extend it to all of SEP 14's RDW/GNLSE/soliton timing ladders. Use short qualified
    problems or saved-state windows, with full-size scientific examples in the
    existing manually run gallery, or seek an explicit scope decision before running
    the proposed campaigns. Publish the benchmark parameters, code/build identities,
    raw samples and error arrays alongside the summaries so these tables are auditable.

17. **Q17 — P2: Separate measured endpoint accuracy from tolerance guarantees and
    qualify the citations (§§2–3, 4.7, 7, 9).** Owner: author before acceptance.
    A local `rtol` does not bound global field differences between two adaptive runs.
    Equal rounded worst-plane errors against a reference also do not mean equal
    trajectories. Replace “a rerun differs by its own tolerance” with the measured
    errors and a quantitative acceptance criterion, including weak bands, reference
    convergence and the uncertainty of the timing comparison. Do not automatically
    re-record adaptive golden arrays before showing that criterion is met.

    The Luna comparison needs a precise public documentation citation and a narrower
    claim: solprop's in-band mask does exclude populated collar bins, and Adams uses
    the pre-step norm rather than RK's maximum of the two norms
    (`AdamsMarch`'s norm, SEP 13 Part II). Equal `rtol` therefore is not equal achieved
    accuracy across these configurations. Also correct DOP853's attribution to
    reference [2]: the 1980 5(4) paper does not identify its 8(5,3) implementation.
    [Hairer's solver documentation](https://www.unige.ch/~hairer/software.html)
    distinguishes DOPRI5 5(4) from DOP853 8(5,3), with continuous extensions of orders
    four and seven respectively; add the appropriate primary method reference if
    phase 6 selects it.

**Review evidence and limits.** The Hermite witness was evaluated with the checked-out
DOPRI5 tableau; the scalar split-history witness used solprop's integrated Adams
weights and exact initial history. The Kerr/splitting comparison above is an independent
analytic derivation, also evaluated numerically. These are short algebraic probes,
not new propagation benchmarks. Reviewed the Kokkos adapter's per-stage scalar bundles
as well as the in-tree implementation; no same-interval requirement was apparent in
that adapter, but no Kokkos execution was performed and Q2 remains a qualification task.
Checked the public solver documentation for the RK order claims and the publisher's
record for Strang [4]; the full Strang/Shampine texts and the draft's full-run timings
were not independently reproduced. Human review and the author's responses remain due.

#### Author's responses — 2026-10-04

Each review question is answered here, and the answer is reflected in Part I.

- **Q7 (accepted).** The half-factor construction is withdrawn. The symmetric collar
  becomes phase 4's bounded study: it must derive an Adams-consistent history
  transformation, its source identities and its variable-step recurrence, and pass the
  review's witness, a non-commuting three-level test and the lifecycle tests (§4.5).
  The Runge–Kutta march does not need it, because the in-stage collar is well inside
  DOPRI5's stability interval.
- **Q8 (accepted).** The study must control the splitting error explicitly, by a collar
  step cap with a documented accuracy setting or by a separate estimate (§4.5).
  "Unconditionally stable" now describes the attenuation substep only. The transform pair
  per step and the loss of FSAL are in the cost and admission contract.
- **Q9 (accepted).** The RK dense output is DOPRI5's own continuous extension. Hermite
  is an explicitly uncontrolled reference only (§4.3), and the same caveat is recorded
  for SEP 13's `dense="hermite"`. The polynomial witness and interior-order tests with
  nonzero sources and a varying linear flow are in §7.
- **Q10 (accepted).** The new interval cap and growth cap apply to the diagonal-flow
  routes only. Coupled-flow sections, reject-free runs and device configurations keep
  today's policy until separately qualified. Kinks stay mandatory landings, and the gate
  includes the localised source witness, the independent table refinement and the
  endpoint and kink-side tests (§4.1, §7).
- **Q11 (accepted).** The first-step rule is now defined (§4.2):
  - it uses the interaction-picture derivative, not the lab-frame one;
  - its probe is capped at the next landing;
  - it specifies its fallbacks, its resume behaviour and the reuse of the first
    derivative;
  - its two calls are charged to the ledger and budget, and explicit settings are
    honoured;
  - the landing stretch stays 1.1, separate from the growth cap;
  - it is measured separately from the growth cap.
- **Q12 (accepted).** Phase 5 begins with the diagnosis the review asks for. Any change
  of variable only constructs a certified representation in the existing affine-$z$
  basis, with $\Phi$ and $\Delta\Phi$ continuity and error criteria before any golden
  update (§4.4).
- **Q13 (accepted).** §4.6 gives the legal schedule: a predictor pass, the source, a
  corrector and norm pass, the reduction and decision, and a conditional commit.
  Rejections leave the accepted state and history untouched. The memory ledger counts
  all scratch, ring, dense and rollback storage, and the schedule is benchmarked
  against the same arithmetic in NumPy.
- **Q14 (accepted).** §5 now has the admission matrix (`dense_output` defaults to
  `"off"`):
  - refused for the per-step collar, the `modified` noise, fixed-schedule batches and
    coupled flows;
  - statistics labelled, and checkpoints landed on;
  - the matrix applied to SEP 13's dense observation from phase 3.

  The DOPRI5 extension is contracted at acceptance, five buffers against Hermite's two,
  and is done on every CPU backend. Any omission becomes a written parity exception.
- **Q15 (accepted).** §6 lists the contract changes per phase. Each incompatible merged
  seam bumps its version, with literals, Kokkos admission checks and a test that the
  effective defaults are in the configuration identity.
- **Q16 (accepted).** The extrapolated full-run timings are removed. The cost equation
  stays as a hypothesis, and §7 has the measurement protocol: idle machine, warm-up,
  fixed threads, alternating samples, A/A uncertainty and a bounded stopping rule. §2
  records the maintainer's scope decision for full-length runs; CtC uses slices and
  windows. Raw samples and error arrays are committed with the summaries.
- **Q17 (accepted).** §7 states a quantitative acceptance criterion on measured errors,
  replacing "a rerun differs by its own tolerance". §3 narrows the Luna comparison to
  the norm's form, cites Luna's public repository and says that equal `rtol` is not
  equal accuracy. DOP853 is attributed to Hairer's codes [8] and [1].

**The maintainer's review (John Travers, 2026-10-04).** It asked for two things:

- A comparison at the production tolerance `rtol=1e-6`, the one a GNLSE run would use.
  It is in §2: at `rtol=1e-6`, Adams is 1.5 times cheaper in calls and 3.4 times less
  accurate.
- To leave open whether Adams should become a default. That is decided at phase 2's
  gate on the compiled backends' runtime curves (§7, §2 scope).

### 13. References

1. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I*,
   2nd ed., Springer (1993), §II.4 (starting step size, step-size control) and §II.6
   (dense output).
2. J. R. Dormand and P. J. Prince, "A family of embedded Runge–Kutta formulae,"
   *Journal of Computational and Applied Mathematics* **6**, 19–26 (1980).
   [doi:10.1016/0771-050X(80)90013-3](https://doi.org/10.1016/0771-050X(80)90013-3).
3. L. F. Shampine, "Some practical Runge–Kutta formulas," *Mathematics of Computation*
   **46**, 135–150 (1986). [doi:10.2307/2008219](https://doi.org/10.2307/2008219).
4. G. Strang, "On the construction and comparison of difference schemes," *SIAM Journal
   on Numerical Analysis* **5**, 506–517 (1968).
   [doi:10.1137/0705041](https://doi.org/10.1137/0705041).
5. J. H. Verner, "Numerically optimal Runge–Kutta pairs with interpolants," *Numerical
   Algorithms* **53**, 383–396 (2010).
   [doi:10.1007/s11075-009-9290-3](https://doi.org/10.1007/s11075-009-9290-3).
6. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Reviews of Modern Physics* **78**, 1135–1184 (2006).
   [doi:10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135).
7. Luna.jl, the predecessor code's public repository and documentation,
   <https://github.com/LupoLab/Luna.jl>: its default `RK45` integrator's weak error
   norm, `rtol=1e-6`, safety factor 0.9 and growth factors. See also solprop's
   [differences from Luna](../../../guide/workflows/luna.md).
8. E. Hairer, [Fortran and Matlab codes](https://www.unige.ch/~hairer/software.html):
   DOPRI5, a 5(4) pair with a fourth-order continuous extension, and DOP853, an
   8(5,3) pair with a seventh-order continuous extension.

## Part II — Implementation record

**Status, 2026-10-05.** Every phase is implemented or reported, and merged to `main` on
2026-10-05 from `feature/sep-0014-stepper-efficiency-1` (solprop-kokkos: the branch of
the same name, merged to its `main`). The phase gate ran on the Linux workstation on
2026-10-05, after its fixes at `942f5f3` and `248be52`, and again at the merge of the
two follow-ups and `main` (`305c797`), where every lane passed (slow 138, default
12 225, coverage 100 %):

- the default lane passes but for two Linux-only tests that fail on `main` too (an
  ionisation-knots record and a one-ulp equality in the optmat cycle average), raised
  separately;
- the slow lane passes, the M2 tolerance witness re-recorded after phase 1's limits
  (1.1e-6 to 1.6e-5 of its gap, the same on macOS and Linux);
- coverage is 100 %;
- the Kokkos suite fails only on the SEP 3 drift (ADK import, H2's Raman carrier).

Decided by the maintainer, 2026-10-05:

1. **Phase 2.** No route's default changes: IP-DOPRI5 stays the default, and the
   low-call Adams method is documented where it pays (the performance guide's "Which
   stepper" section, the low-call stepping page and the `integrator` docstrings). A
   device-resident Kokkos Adams step is left for a later change.
2. **Phase 3.** The consolidated interface (`interpolate_planes`, `step_planes`,
   `resume="interpolated"`) is accepted, and the bitwise replay resume on `numba_cpu`
   and Kokkos is dropped.
3. **Phase 5.** ADR-040 is accepted; an absolute tolerance on the density paths stays
   open as a complement (the ADR's "Why").
4. **Phase 6.** Neither DOP853 nor a band-weighted norm is adopted.
5. **The benchmark ledger** names a machine by `SOLPROP_BENCH_MACHINE` or by its
   operating system, architecture and core count, never its host name; the phase 2
   entry is in `benchmarks/results/linux-x86-64-72c/`.
6. **SEP 15** is reviewed separately and merged as a draft with this SEP.
7. **Phase 4.** The Strang split is not offered: it is first order wherever the collar
   holds field, and its splitting error is invisible to the error estimate. An
   implicit treatment of the collar is pursued in SEP 15 instead (its Q5).

Set `Final` by the maintainer on 2026-10-05. The SEP 3 drift in solprop-kokkos's tests
was fixed on its `main` by the port of those tests to optmat's materials, merged with
this SEP's work. Left open and moved to the [roadmap](../../roadmap.md):

- a device-resident Kokkos Adams step (Performance and backends);
- an absolute tolerance on the density paths, beside ADR-040's floor (Numerics);
- CUDA rows for the spatial routes' convergence records (Performance and backends);
- an implicit treatment of the temporal collar, which is SEP 15's Q5.

### Phase 1 — Step limits

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0014-stepper-efficiency-1`; solprop-kokkos branch of the same name. Merged
to `main` on 2026-10-05.

**Delivered.**

- **Interval cap by route.** `StepperSettings.interval_cap` (kernel contract 17). The
  shared march caps a step at its interval's width only when it is set. `assemble`
  resolves it:
  - off on the diagonal-flow routes;
  - on for a coupled flow or a reject-free controller;
  - `SimulationSettings(interval_cap=...)` overrides either way.
- **Start-up rule.** `StepperSettings.auto_init` (contract 17) and
  `PIController.start_factor`. The first step is `dz_init`. The step after it is the
  deadbeat step its own scaled norm implies, at most 100 times as long. The history
  then holds that step's equivalent norm, or zero when the factor hits the cap of 100:
  such a norm is round-off and differs between backends (below).
  - The convenience interfaces set it when no `dz_init` is given.
  - An explicit `dz_init`, a coupled flow or a reject-free controller turns it off.
  - A resumed run takes it again.
- **Growth cap: unchanged at 1.1** (Q1, below).
- **Output schema 10.** `StepLimits.interval_cap` and `auto_init` are recorded, and are
  in the resume key from schema 10 only.
  - A schema-9 file is refused for resume.
  - Taken back to schema 9, a schema-10 record reproduces the schema-9 digest exactly
    (tested).
- **Records.** ADR-038, which supersedes ADR-004's interval cap on the diagonal-flow
  routes.
- **Docs.** Step control in `stepper.md` §3, the kernel contract, the output page's
  resume key, the changelog, and the low-call page's IP-DOPRI5 numbers.
- **Benchmarks.** `benchmarks/sep14/limits.py`; its results are in
  `benchmarks/sep14/data/limits.jsonl`.
- **solprop-kokkos.** Only the version literal moves to 17 (Q2, below).

**The measurements** (full length on `numpy_ref`, worst plane against a tight reference;
"weak" is the far-UV band on the canonical case and a 1e-9 to 1e-3 band elsewhere).
`v16` is the old march. The new default is the cap off plus the start-up rule.

*Canonical case, 1 mJ* (calls; window field; weak):

| `rtol` | v16 | new |
|---|---|---|
| 1e-2 | 1127; 1.5e-2; 0.75 | 497; 1.5e-2; 0.75 |
| 3e-3 | 1260; 5.3e-3; 0.54 | 644; 5.3e-3; 0.54 |
| 1e-3 | 1344; 1.4e-3; 0.25 | 742; 1.4e-3; 0.25 |
| 3e-4 | 1477; 3.5e-4; 4.7e-2 | 896; 3.5e-4; 5.7e-2 |
| 1e-4 | 1841; 1.5e-4; 1.3e-2 | 1015; 1.5e-4; 1.3e-2 |
| 1e-5 | 2121; 8.8e-6; 1.6e-3 | 1484; 1.7e-5; 1.6e-3 |
| 1e-6 | 2583; 1.3e-6; 7.9e-5 | 2184; 2.7e-6; 8.0e-5 |
| 5e-7 | — | 2422; 1.0e-6; 1.0e-5 |
| 1e-7 | 3731; 2.8e-8; 2.4e-6 | 3255; 2.2e-8; 2.1e-6 |

*Canonical case, 2 mJ:*

| `rtol` | v16 | new |
|---|---|---|
| 1e-2 | 1316; 4.1e-2; 0.70 | 700; 4.2e-2; 0.69 |
| 1e-3 | 1575; 2.5e-3; 2.8e-2 | 1001; 2.5e-3; 2.8e-2 |
| 3e-4 | 1323; 5.6e-4; 1.2e-2 | 1176; 5.6e-4; 1.2e-2 |
| 1e-4 | 1526; 2.4e-4; 3.1e-3 | 1400; 2.4e-4; 2.6e-3 |
| 1e-5 | 2149; 5.0e-6; 2.9e-4 | 2065; 8.0e-6; 2.9e-4 |
| 5e-6 | — | 2303; 2.8e-6; 7.6e-5 |
| 3e-6 | — | 2527; 1.3e-5; 4.7e-5 |
| 1e-6 | 3542; 1.2e-6; 4.4e-6 | 3108; 7.8e-7; 4.2e-6 |

*The other cases:*

| Case, `rtol` | v16 | new |
|---|---|---|
| RDW, 1e-5 | 8176; 1.5e-4; 3.2e-3 | 7987; 1.5e-4; 3.2e-3 |
| RDW, 1e-6 | 13188; 2.1e-5; 6.1e-4 | 13013; 2.1e-5; 6.1e-4 |
| CtC slice (2 cm), 1e-5 | 889; 1.5e-5; 2.2e-4 | 658; 2.7e-5; 7.9e-4 |
| CtC slice, 3e-6 | — | 819; 5.3e-6; 1.4e-4 |
| CtC slice, 1e-6 | 1218; 8.5e-7; 1.4e-5 | 1022; 1.2e-6; 3.1e-5 |
| CtC slice, 4e-7 | — | 1225; 3.5e-7; 8.8e-6 |
| GNLSE control, 1e-5 | 19278; 1.4e-2; 9.9e-2 | 19173; 1.3e-2; 0.10 |
| GNLSE control, 1e-6 | 30709; 1.6e-3; 1.3e-2 | 30625; 1.6e-3; 1.3e-2 |

**Against the criterion of §7** (at every `rtol`, worst-plane errors within 1.25 times
v16's, or v16's errors reached at fewer calls):

- **Met within 1.25 times** on the canonical case from `rtol` 1e-2 to 1e-4 at both
  energies, at 1e-6 and 1e-7, on RDW and on the GNLSE control. The canonical case at
  1 mJ and 3e-4 is 1.22 times in the weak band. These runs take 1.1 to 2.3 times fewer
  calls.
- **Met at fewer calls** on the canonical case at 1 mJ and 1e-6, and on the CtC slice
  at 1e-5. On the CtC slice at 1e-6 it is met at equal calls (1225 against 1218) with
  errors 1.5 to 2.5 times smaller.
- **Not met: the canonical case at 2 mJ, `rtol` 1e-5.** The field error is 1.6 times
  v16's, and no tighter tolerance reaches 5.0e-6 in fewer calls. The global error is
  not monotone in `rtol` here: 2.8e-6 at 5e-6, 1.3e-5 at 3e-6. Both v16 and the new
  default sit within a few times `rtol`.
- **The 1 mJ difference at `rtol` 1e-5 and 1e-6 is all at the exit plane.** The cap
  forced 50 steps (down to 0.5 µm) into the last centimetre, against the choked end's
  square-root density; the controller takes 3. Every interior plane is equal or better.
- **The CtC slice difference is the start-up rule's.** The old ramp kept the first 30
  steps far inside the tolerance, which on a 2 cm slice is a large part of the run.

**Q1, growth cap: 1.1 stays.** With the start-up rule on, a cap of 1.5 or 2 saves a
further 15–20% on the canonical case and under 2% elsewhere. It costs accuracy that the
weak norm does not see:

- 11 times the field error at `rtol=1e-7` (1 mJ);
- 3 to 5 times in the far-UV band at 1e-4 (1 mJ) and at 3e-4 and 1e-3 (2 mJ).

Most of the saving the review expected from the cap was the start-up ramp, which the
rule removes. A looser cap waits for phase 6's weak-band norm.

**Q2, does any kernel need a stage inside one interval?** No. The stage scalars, ΔΦ and
the prefactor, are computed on the host per stage on every backend. The Kokkos C++
uses no table interval to limit a step. solprop's suite, with its Kokkos
parametrisations, passes with the cap off; the solprop-kokkos suite passes 176 tests.
Its 9 failures and 2 collection errors come from SEP 3's material API changes, which
its tests predate, not from the step limits.

**The localised-source witness (§4.1).**

- **The setup.** A narrow smooth bump was put in two places:
  - in a GNLSE fibre's effective area (10 µm wide, area halved, analytic phase);
  - in a capillary's pressure (0.1–1 mm wide, five times the density).
- **The interval cap never bound in either.** The operator table represents the
  cumulative Φ, which stays smooth, so it does not refine around such a bump. The old
  and new limits gave identical runs.
- **The GNLSE witness shows an existing limitation.** At 87 µm steps, both embedded
  formulas skip a 10 µm bump: the phase is wrong by 1.4e-2, the bump's whole
  contribution, under both the old and the new limits. A source feature narrower than
  the step must be declared as a kink or bounded by `dz_max`. This is now said in
  `stepper.md` §3.

**Deviations.**

- **The first-step rule is the error-scaled jump above,** not the two-call Hairer–Nørsett–
  Wanner probe of §4.2.
  - It costs no extra call, since the first step is a real step, and needs no new
    backend hook. Numba and Kokkos have no standalone source call before phase 2.
  - It meets each of Q11's points: an interaction-picture estimate, a landing-capped
    probe, the fallbacks, resume, the call ledger and explicit settings.
  - It is not applied to SEP 13's Adams march, whose start is unchanged.
- **The growth cap is unchanged** (Q1).
- **Device configurations take the same defaults** (§4.1 had them keep the old
  policy).
  - The limits are host-side, and no kernel depends on them (Q2).
  - The standing GPU requalification of `backends.md` covers them.
  - A reject-free GPU run keeps the old limits.

**Not done.**

- **A manufactured density with a known integral** among §7's interval-cap witnesses
  (added on 2026-10-05: `test_a_manufactured_density_integrates_exactly_across_intervals`,
  a density `sqrt(a + b z)` near choking whose operator integrates analytically; the
  table gives ΔΦ to 8.4e-14 over pairs spanning up to 9 of its 18 intervals).
  The analytic-phase witness `tests/driver/test_general_gnlse.py` integrates a known
  path, but through the prefactor rather than the operator table; the kink-side
  witness is the same test across a kink of the area path. Its tolerance moved from
  1e-9 to 2e-9: the start-up rule takes 27 steps where the old ramp took 74.
  (The independent table refinement was added after this report:
  `test_a_coarser_table_moves_an_uncapped_run_far_less_than_its_tolerance`. A table
  built to 1e-9 instead of 1e-12 moves an uncapped canonical run by 3.3e-10 at
  `rtol=1e-6`.)
- **Coverage and the slow lane,** which belong to the phase gate.
- **Linux timings.** Phase 1 changes calls only, and those are platform-independent.

**Accepted by the maintainer (2026-10-04):** the one strict miss of the criterion (the
canonical case at 2 mJ, `rtol` 1e-5) and the deviations above. Phase 2 builds on this
branch; the remaining witnesses, coverage and the slow lanes run at the phase gate.

### Phase 2 — Adams on the compiled backends, and the runtime study

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0014-stepper-efficiency-1`, solprop-kokkos branch of the same name. Merged
to `main` on 2026-10-05. Reported after phase 3 because the Linux campaign was still running.

**Delivered.**

- **One march for every backend.** SEP 13's Adams march moved from `numpy_ref` into
  `solprop.backends.multistep.AdamsMarch`, which steps on host arrays through five
  `StepDriver` hooks (`_adams_source`, `_adams_propagator`, `_adams_state`,
  `_adams_adopt`, `_adams_restore`; kernel contract 17). A backend supplies one source
  evaluation and the propagator; the history, the weights, the order choice and the
  decisions are shared.
- **The fused step (§4.6).** For the selected family, one-order-higher PEC, the step is
  three passes over a ring store of history rows: `AdamsKernels.predict`, `correct` (the
  corrected field and the defect norms of the current and neighbouring orders in one
  pass) and `transport`. `numba_cpu` compiles them (`NumbaAdamsKernels`), with weights
  bitwise those of NumPy. A rejection leaves the accepted state and the ring untouched.
- **`numba_cpu`** evaluates the source through its fused stage pipeline with identity
  combination; **Kokkos** through its general engine's stage 0 at the evaluation
  position (`State.adams_source`), adopting the step through its native finish
  (`State.adopt`). Both follow `numpy_ref`'s trajectory: the same steps, orders and
  calls, fields to 1e-13 of the peak (`tests/backends/test_compiled_multistep.py`).
- **Optimisations kept** (Mac, same-process A/B on the RDW slice): the fused passes and
  the held propagator took a `numba_cpu` Adams step from 1.93 to 1.25 ms.
- **Refusals lifted** on both compiled backends for the fibre and capillary mode sets;
  the `modified` noise model stays refused there.

**The runtime study** (`benchmarks/sep14/runtime.py`, raw samples in
`benchmarks/sep14/data/runtime/`). A two-socket Intel Gold 6240 Linux workstation,
eight threads pinned to socket 0 (`taskset 0-7`), three interleaved rounds after a
warm-up, median and interquartile range; references two decades below the ladder on the
same backend. The `numba_cpu` rows were measured again on 2026-10-05, after the Linux
fix described under "Found on the way" (merge `305c797`, optmat 0.3.0); the rows from
before the fix are kept in `data/runtime/before-openmp-fix/`. The Kokkos rows are the
first campaign's (optmat 0.2.2), which the fix cannot touch. The CtC rows of both
backends were measured again with the whole case's collar strength (below, "Why the
Adams march wins little on CtC"); the slice-collar rows are kept in
`data/runtime/slice-collar/`. The interquartile ranges are
below 1 % on Kokkos and below 10 % on `numba_cpu`. The table gives the
Adams wall time over IP-DOPRI5's at **matched worst-plane field error**, interpolated in
log-log between the measured tolerances (below 1, Adams is faster):

| case | backend | error 1e-2 to 4e-3 | 8e-4 to 2e-4 | 1e-4 to 2e-5 | 6e-6 to 2e-6 |
|---|---|---|---|---|---|
| canonical, 1 mJ | `numba_cpu` | 0.58 | 0.76 | 0.79 | 0.70 |
| canonical, 1 mJ | Kokkos | 0.59 | 0.88 | 0.93 | 0.88 |
| canonical, 2 mJ | `numba_cpu` | 0.55 | 0.79 | 0.89 | 0.94 |
| canonical, 2 mJ | Kokkos | 0.58 | 0.93 | 1.10 | 1.20 |
| CtC slice | `numba_cpu` | — | 0.96 (4e-4) | 0.87–0.89 | 0.91 (6e-6) |
| CtC slice | Kokkos | — | 1.00 (4e-4) | 0.91–0.94 | 0.96 (6e-6) |
| RDW | `numba_cpu` | 0.77 | 0.99 | 0.96 | 0.89 |
| RDW | Kokkos | 1.03 | 1.38 | 1.27 | 1.04 |
| GNLSE | `numba_cpu` | 2.3 (3e-1) | 2.7 (8e-2) | 2.6 (2e-2) | 2.3 (6e-3) |
| GNLSE | Kokkos | 3.0 (3e-1) | 3.5 (8e-2) | 3.2 (2e-2) | 2.8 (6e-3) |

The GNLSE column headings do not apply: its errors at the measured tolerances span
3e-1 to 6e-3 and are given in brackets. The weak-band errors tell the same story, a
little more favourably to Adams: on `numba_cpu` 0.47–0.82 on the canonical case and
0.63–0.78 on RDW.

- **Per step**, an Adams step costs a third to a fifth of an IP-DOPRI5 step: on the
  canonical case 2.8 against 13.3 ms on `numba_cpu` and 1.9 against 7.2 ms on Kokkos;
  on RDW 4.0 against 15.3 and 3.4 against 9.8 ms; on the GNLSE 3.1 against 8.0 and 4.2
  against 9.1 ms.
- **Steps.** The Adams march takes 2–5 times as many steps for the same error, so it
  wins only where a source evaluation dominates a step. That is the gas-filled
  capillary with ionisation; the GNLSE's Kerr and Raman source is cheap beside the
  linear work.
- **Kokkos executes IP-DOPRI5 natively and the Adams march on the host,** with one
  native source evaluation per step. On the capillary cases its Runge-Kutta step is
  1.4–1.8 times cheaper than `numba_cpu`'s and its Adams step 1.2–1.5 times, so the
  margin shrinks: Adams wins on Kokkos only on the canonical case and at loose
  tolerances. (On the GNLSE `numba_cpu` is the faster of the two for both methods.)
- **`split_step` Adams** is 2–10 % faster than the in-stage collar on Kokkos at tight
  tolerances, with errors up to 17 times larger at `rtol=1e-6`; not recommended.

**On macOS** (Apple silicon, six threads on the top-tier cores, the same script, cases
and rounds; `data/runtime-macos/`, CtC with the whole case's collar), the ratios agree
with the workstation's within a few per cent, though every run is three to four times
faster; the GNLSE's loss is smaller because an Adams step is relatively cheaper there:

| case | `numba_cpu` | Kokkos |
|---|---|---|
| canonical, 1 mJ | 0.58–0.77 | 0.58–0.88 |
| canonical, 2 mJ | 0.55–0.93 | 0.56–1.11 |
| CtC slice | 0.83–0.93 | 0.84–0.96 |
| RDW | 0.74–0.93 | 0.95–1.26 |
| GNLSE | 1.6–2.0 | 2.2–3.0 |

The macOS spreads were within 4 % on `numba_cpu` (one point 11.5 %) and 8.5 % on Kokkos,
at loads of 3–6 and 5–10.

**Recommendations (problem classes; no default changed, the maintainer decides).**

| problem class | `numba_cpu` | Kokkos CPU |
|---|---|---|
| gas-filled capillary with ionisation (canonical, RDW) | Adams: 0.55–0.8 of the time on the canonical case, 0.75–1.0 on RDW, at matched error | Adams on the canonical case at loose tolerances (0.6); otherwise even |
| CtC-type Raman-active fibre slice | Adams, marginally: 0.87–0.96 | about even (0.91–1.0) |
| GNLSE (Kerr and Raman only) | IP-DOPRI5: Adams takes 2.3–2.7 times as long | IP-DOPRI5 |
| expensive responses (SEP 12's TDSE) | Adams: calls dominate, SEP 13's 2.5–7 times fewer calls apply | Adams |

**Deviations.**

- **No A/A comparison** (§7): three rounds instead, with interquartile ranges reported.
  Ratios within 0.9–1.1 (the CtC slice, Kokkos's RDW at the loosest tolerance) are read
  as even.
- **The benchmark-ledger entry** (`benchmarks/bench_adams.py`) was not recorded with
  each optimisation; one entry was recorded on the Linux workstation after the phase
  gate, eight pinned threads, and is in `benchmarks/results/linux-x86-64-72c/` (filed
  under the ledger's new machine name, its record otherwise as written; its dirty flag
  marks only untracked log files). Its Adams windows agree with an earlier unrecorded run to
  5 %. It predates the Linux fix below: on RDW `numba_cpu`'s window (1.21 s) is slower
  than `numpy_ref`'s (0.79 s).
- **Native Kokkos Adams step** (§6 "Kokkos holds the history on the device") is not
  built: the march is on the host and only the source is native. A profile says what
  it would buy: on macOS (Apple silicon), a Kokkos Adams attempt on the canonical case takes 0.67 ms,
  of which the native source evaluation is 0.38 ms and the host march's passes the
  other 0.29 ms (43 %). Removing most of that would take Kokkos's Adams time to about
  0.6–0.7 of IP-DOPRI5's on the canonical case. It is left for a later change.

**Why the Adams march wins little on CtC and loses on the GNLSE** (diagnosis,
2026-10-05, `benchmarks/sep14/adams_limits.py`, data in `data/adams_limits-*.jsonl`).
On every case an Adams step costs a third to a fifth of an IP-DOPRI5 step, close to the
one call in six on CtC, whose source is the most expensive. What decides the comparison
is how many more steps the Adams march takes, and that has two causes.

- **The temporal collar's real-axis stability, on the CtC slice.** The collar's rate
  scales as `boundary_N / length`, so the 2 cm slice of the 10 cm case absorbed at
  12 600 m⁻¹ against the case's 2 520. The Adams march then ran 418 of its 423 steps at
  order three with a median step of 4.8e-5 m, the order-three stability limit
  `2 r / alpha_max` (r = 0.30), whatever the tolerance; IP-DOPRI5's limit is eleven times
  longer. At the whole case's collar strength the march runs at order five with 1.5
  times longer steps and needs 1.25–1.5 times fewer calls than IP-DOPRI5 at matched field
  error (278 against 348 at 2.9e-4). Splitting the collar out of the source removes the
  limit but not the error: the field reaches CtC's collar, and the per-step split's
  first-order error (6.9e-3 to 2.6e-3 between `rtol` 1e-4 and 1e-6) does not converge to
  the tolerance. The runtime study's CtC slice now keeps the whole case's collar
  strength (`runtime.py`), and its rows above are measured again: the Adams march
  takes 0.87–0.96 of IP-DOPRI5's time on `numba_cpu` and 0.91–1.00 on Kokkos, where it
  was even or behind with the slice's collar.
- **Phase-mismatched forcing, on every dispersive case.** In the interaction picture the
  source of a populated bin oscillates along the fibre at that bin's linear phase rate,
  and a polynomial fitted over several past steps tolerates about 0.6 radian of it per
  step (SEP 13's scalar study); IP-DOPRI5's stages follow several radians within one.
  With the largest `|Im D|` over the bins within 1e-6 of the peak power:

  | case | phase rate | `Delta beta h`, Adams | `Delta beta h`, IP-DOPRI5 | Adams orders |
  |---|---|---|---|---|
  | canonical | 44 m⁻¹ | 0.28 | 1.3 | 2–3 (collar-bounded) |
  | CtC slice, whole-case collar | 1.0e4 m⁻¹ | 0.75 | 3.8 | 5 |
  | RDW | 5.3e3 m⁻¹ | 1.4 | 7.7 | 2–3 |
  | GNLSE | 2.0e5 m⁻¹ | 2.1 | 11 | 4 |

  Only the canonical case stays inside the limit. The GNLSE has no collar, and capping
  the order to gain imaginary-axis stability made the steps shorter (order ≤ 3:
  9.5e-6 m, ≤ 2: 7.9e-6 m, against 1.0e-5 m), as did PECE3/4; the nonlinear phase
  `gamma P h` at the Adams steps is 0.009 (median). Narrowing its window from
  400–2000 nm to 600–1300 nm, which removes the most detuned bins, cuts the Adams step
  penalty against IP-DOPRI5 from 5.2 to 2.9 times. On the GNLSE the source is also cheap,
  so an Adams step is only 2.6 times cheaper than an IP-DOPRI5 step, and the chaotic
  supercontinuum turns the Adams march's larger per-step error into a seven times larger
  exit error at the same `rtol`.

What follows: the low-call methods suit weakly dispersive problems, where SEP 12's TDSE
in a gas-filled capillary lies. For dispersive, broadband problems a method that fits the
source in the laboratory frame and integrates the linear phase exactly is proposed in
[SEP 15](sep-0015-exponential-adams.md).

**Found on the way.** `numba_cpu`'s Runge-Kutta step on RDW (four modes, a pressure
gradient) takes 120 ms against Kokkos's 10 ms, where the other cases differ by 1.7
times. On macOS (Apple silicon), after compilation, the two are within 10 % on a 5 cm RDW slice (8.4
against 9–11 ms a step). On the GPU droplet's 8-core Linux host `numba_cpu` took 44 ms a
step on the same slice once compiled, against 66 ms for `numpy_ref`.

*Resolved 2026-10-05 (outside this SEP).* The cause was the campaign's environment
combined with pyFFTW's Linux wheel, not the stepper. `SOLPROP_FFTW_THREADS=8` overrides
the backend's size rule, which gives RDW's 32 768-element modal transforms one thread.
pyFFTW's Linux wheel threads FFTW through its own GNU OpenMP runtime, and that
runtime's idle threads spin. They then hold the cores that the Numba kernels' TBB pool
needs. Timing each pass of the 5 cm slice on 8 cores of the workstation, every planned
transform took 3.5–5.9 ms in place of 0.1 ms, and a step 129 ms in place of 25 ms. Two
or four FFTW threads were as bad. `OMP_WAIT_POLICY=passive` brought the step back to
25 ms. The Numba threading layer, OpenBLAS and denormals made no difference.
`numba_cpu` now imports pyFFTW with that policy on Linux unless the user sets one
(`backends/numba_cpu/openmp.py`). The backend's own size rule hit this too. In the
benchmark ledger's ten-step windows at the default pools, the full CtC window, where
FFTW gets four threads, went from 6.92 to 4.85 s and the reduced one from 0.40 to
0.32 s. RDW, whose transforms get one thread, did not move (0.147 against 0.144 s).
In the campaign's environment the RDW window went from 1.18 to 0.14 s. The
`numpy_ref` and Kokkos rows, which the fix cannot touch, stayed within 4 %.

**Consequence for the phase 2 runtime study.** Every `numba_cpu` row above was measured
in that environment (`benchmarks/sep14/data/runtime/campaign*.sh`), so the linear work
of each step was inflated, and the inflation was larger for IP-DOPRI5's steps than for
the Adams march's. On the 5 cm RDW slice at `rtol=1e-3`, in the campaign's environment,
the Adams march took 0.59–0.72 of IP-DOPRI5's wall time before the fix and 0.81–0.82
after it. The `numba_cpu` rows were therefore measured again with the fix (above): the
RDW campaign took 11 minutes instead of 74, and Adams' margin narrowed from 0.62–0.83 to
0.77–0.99 on RDW, from 0.56–0.73 to 0.58–0.79 on the canonical case and from 0.90–0.99
to 1.00–1.08 (even) on the CtC slice. The Kokkos rows are unaffected.

### Phase 3 — Runge-Kutta dense output

**Scope.** 2026-10-04; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0014-stepper-efficiency-1`, solprop-kokkos branch of the same name. Taken
ahead of phase 2's report while its Linux runtime campaign runs. Merged to `main`
on 2026-10-05.

**Delivered.**

- **The option** (named as consolidated below). `interpolate_planes` on
  `SimulationSettings`, `prop_gnlse`, `prop_capillary` and their `Simulation.from_*`
  constructors; `False` is the default. For the Runge-Kutta march it is DOPRI5's
  continuous extension, contracted from the accepted
  step's seven stages into five coefficients in the anchor's interaction picture
  (`stepper.tableau.extension_coefficients`, `extension_value`), and carried to the
  plane by the exact linear flow.
- **The march.** The shared `StepDriver` drops the save planes from the landing targets,
  contracts any accepted step that crosses the requested plane, and evaluates the plane
  while holding the march's own state ahead. The end of the run and the hard kinks are
  landed on. One hook, `_dense_stages`, returns the accepted step's stage derivatives on
  the host: `numpy_ref` keeps the trial's list, `numba_cpu` copies its stage buffers,
  and Kokkos copies them from the engine (`State.stage_to_host`).
- **The stiff-bin guard.** A bin the linear flow attenuates by more than an e-fold over
  the step takes the anchor's value. The first version returned errors up to 48 times the
  field there: the spectral collar's bins lose about 9 e-folds per step, and the
  interaction-picture derivative grows exponentially across it. The same guard now applies
  to the Adams interpolant.
- **The admission matrix (§5)**, at translation, for both steppers: refused beside the
  per-step collar, a coupled flow and the `modified` noise; the settings refuse a
  schedule and a Runge-Kutta tableau without an extension (Tsit5).
- **Resume (the maintainer's decision, below).** A file whose last plane was
  interpolated is refused for `resume=True`, for both kinds of dense observation;
  `resume="interpolated"` continues afresh from it and appends the position to the
  annotation `interpolated_resumes`. A plane is interpolated when no accepted step of
  the run ends on it.
- **Versions.** Kernel contract 18 (`StepperSettings.interpolate_planes`, the
  `_dense_stages` hook) and output schema 11 (`StepLimits.interpolate_planes`, in the
  resume key from schema 11
  only; a schema-10 file is refused for resume; a schema-11 record taken back to schema
  10 reproduces the schema-10 digest, tested).
- **Records.** ADR-039, amending ADR-009; ADR-009's status names it and ADR-036.
- **Docs.** The solver page (the extension's formula, the matrix, the accuracy caveat),
  `stepper.md` §3, the kernel contract, the output page's resume key, the low-call page's
  matrix, the changelog.
- **Tests.** `tests/backends/test_dense_output.py`: the extension's local order five and
  its exact ends; dense against landed planes on an N = 2 soliton at `rtol=1e-6` (at most
  2.3e-7 of the peak apart, 125 attempts against 130); `numba_cpu` and Kokkos equal to
  `numpy_ref` in attempts and to 1e-12 of the peak; each refusal; the record and the
  resume key. The frozen driver surface now lists the `_adams_*` hooks and
  `_dense_stages`.

**The measurement** (`benchmarks/sep14/dense_planes.py`, `data/dense_planes.jsonl`;
canonical case at 1 mJ, `rtol=1e-3`, `numpy_ref`; every plane against an RK reference at
1e-8 saved at the same planes; worst window field error and worst UV-band error):

| planes | run | accepted | calls | field | UV |
|---|---|---|---|---|---|
| 50 | RK landing | 106 | 742 | 1.35e-3 | 0.247 |
| 50 | RK dense | 91 | 637 | 2.31e-3 | 0.164 |
| 50 | Adams landing | 295 | 299 | 1.82e-3 | 0.055 |
| 50 | Adams dense | 291 | 295 | 1.75e-3 | 0.041 |
| 501 | RK landing | 504 | 3528 | 1.56e-5 | 3.2e-4 |
| 501 | RK dense | 91 | 637 | 2.45e-3 | 0.189 |
| 501 | Adams landing | 505 | 506 | 4.26e-4 | 6.0e-3 |
| 501 | Adams dense | 273 | 275 | 2.05e-3 | 0.025 |

- **The cost no longer grows with the plane count:** 637 calls at 50 and at 501 planes,
  against 3528 landed. The landed run at 501 planes is 150 times more accurate only
  because its steps are capped at the spacing; the dense run delivers what the tolerance
  asks for, as the 50-plane runs do.
- **Accuracy at the tolerance.** At 50 planes the interpolated planes' worst field error
  is 1.7 times the landed run's, inside the factor-of-two scatter a tolerance buys; the
  UV band is better (0.16 against 0.25).
- **The three backends agree:** `numba_cpu` and Kokkos took the same 91 attempts at both
  plane counts, with the same errors to the digits printed.
- **Adams with dense observation** stays the low-call choice at many planes: 275 calls
  against RK's 637, at comparable accuracy. Its interpolants, native and Hermite, agree
  to every printed digit, as in SEP 13.

**Deviations.**

- **Resume from an interpolated plane is refused by default,** where §5 asked only that
  checkpoints be landed on: no checkpoint other than the save planes exists. This
  changes SEP 13's Adams `dense=`, which resumed afresh from any plane (ADR-036).
- **Statistics are not labelled per plane.** The run record says the planes were
  interpolated, and a plane is interpolated exactly when no accepted step ends on it.
  Per-step statistics come from the accepted steps as before.
- **Device memory.** The coefficients live on the host. Kokkos copies the seven stages
  to the host once per step that crosses a plane; with few planes per step that is
  cheaper than five resident device buffers, and the GPU requalification covers it.

**Not done.** The coverage and slow lanes, at the phase gate.

**The maintainer's decision (2026-10-04).** A resume from an interpolated plane is
refused by default, with an override; one rule for both kinds of dense observation.
Implemented as `resume="interpolated"`.

**Addendum: saving the march's own steps (the maintainer's proposal, 2026-10-04).**

- **The option.** `step_planes=k` also saves the end of every `k`-th accepted step since
  the load, beside the requested planes, on every backend, with or without dense output
  and with the Adams march (kernel contract 18 `StepperSettings.step_planes`, output
  schema 11 `StepLimits.step_planes`, ADR-039).
- **How.** After each accepted step the shared march counts it; on the `k`-th it returns
  from `propagate_to` short of the target, and the run loop saves the plane and asks for
  the same target again. A step that crosses a densely observed plane is saved after the
  plane. A resume continues with the first requested plane beyond the one it starts from.
- **Verified** (`tests/driver/test_step_planes.py`): on `numpy_ref`, `numba_cpu` and
  Kokkos the accepted steps are those of the run without it and the requested planes
  are bitwise equal; thinning keeps every `k`-th; dense RK and Adams combine with it; a
  run stopped on a saved step resumes without the override.
- **Bitwise replay resume dropped (the maintainer's decision).** `numba_cpu` and Kokkos
  kept the reused last stage only within a plane, which would have cost about one
  evaluation in six with a plane at every step. They now keep it across planes, and a
  resumed replay agrees to round-off on them.
- **Refused or limited:** scans (a scan saves the same planes at every point) and the
  mixture processing that reads the requested planes' positions.
- **A guard added on the way.** `numba_cpu` reads plasma statistics from the last
  stage's scan when the observed state is the adopted one; it now forgets that after an
  interpolated observation. The shortcut did not fire on the plasma capillary tried, so
  the guard is defensive. Statistics at interpolated planes match `numpy_ref` to 1e-9 on
  both compiled backends (tested), but for the quadrature diagnostic, which the compiled
  backends score per call and so report as `nan` at a plane reached with no step.

**Interface consolidation (the maintainer's decision, 2026-10-04).** Reviewing the save
options together showed one idea behind two switches: `dense_output="native"` for the
Runge-Kutta march and SEP 13's `Multistep(dense="adams")` for the Adams march, named for
the interpolant rather than the choice. They are now one run setting,
`interpolate_planes: bool`, whose interpolant follows from the stepper;
`Multistep.dense` is removed and the Hermite reference moves to
`ComparisonMultistep(interpolant="hermite")` for the benchmarks. The natural-step
option is named `step_planes=k`. The model is three questions: *where*
(`save_planes`/`n_save`), *how* (landed, or `interpolate_planes`) and *what else*
(`step_planes`), set out on the solver page.

### Phase 5 — Choked-end tables

**Scope.** 2026-10-05; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0014-stepper-efficiency-1`. Merged to `main` on 2026-10-05.

**Diagnosis (§4.4).**

- **Which function fails.** The scalar pressure path itself, and the operator built
  on its edges after it. On the canonical capillary the Φ table had 123 intervals, 99
  of them at the 4 µm width floor in the last 0.51 mm before the choked exit at 4.2 m,
  at degrees 32–64; the path alone had a band of 20 floor intervals from 40 µm.
- **Tail or residual.** The coefficient tail: the builder tests nothing else, and the
  tail never fell below the tolerance there.
- **Sampling precision.** The path at the floating-point positions differs from the
  same formula in the exact distance from the exit by 5.6e-14 at 1 mm, 5.2e-13 at
  0.1 mm and 3.1e-12 at 10 µm. Node placement in absolute z adds `spacing(4.2)/(2d)`,
  4.4e-13 at 1 mm and 4.4e-12 at 0.1 mm. That noise crosses the 1e-12 tolerance about
  half a millimetre from the exit, which is where the floor band begins.
- **Tolerance and floor.** At tolerance 1e-11 the path needs 14 intervals and no floor
  band. Each halving of the floor doubles the band's interval count.
- **The reflected witness.** The entrance has the identical cusp near z = 0, where z is
  fine-grained, and resolves in eight intervals. Written in the exact distance from
  the exit, the exit stage resolves in 15; read through a floating-point reflection,
  `p(4.2 - z)`, it bands again.

**The remedy.** Neither a refinement change nor a change of variable was needed. The
builder now also accepts an interval whose tail is at the sampling-noise floor of its
own positions, `SAMPLING_NOISE_FACTOR` (4) times `spacing(z) max|f'| / |f|`, with
`|f'|` bounded from the coefficients (ADR-040). Below that floor the samples carry no
information about the function's variation, and the table is evaluated at the same
floating-point z.

**Acceptance (§4.4).**

| | before | after |
|---|---|---|
| Φ intervals, canonical capillary | 123 | 30 |
| in the last centimetre | 99 | 6 |
| narrowest interval | 4.0e-6 m | 1.3e-4 m |
| translation | 0.8 s | 0.3 s |

- **Φ and ΔΦ.** Φ(0, z) at 101 positions, 60 of them within 1 cm of the exit, moved
  by at most 3.6e-13 rad of 850: round-off. Against `quad_vec` both tables deviate
  identically pair by pair, by at most 2.4e-7 rad over a centimetre, in one bin's loss.
  That is the operator's own round-off floor, which sets the table's tolerance (and is
  1.8e-8 rad per centimetre mid-fibre), not the exit.
- **Continuity.** The cumulative table is continuous by construction (the constant is
  fixed per interval); the exit now refines geometrically like the entrance.
- **The power normalisation** is an expansion on this case, which the change does not
  touch.
- **Golden data.** One recorded array set moved across the default lane: the M2
  witness's density path lost one floor-width interval at its near-vacuum end at
  1.0 m, its values moving by 3.3e-14. It was re-recorded with its provenance; setting
  the factor to zero reproduces the former bytes.
- **Tests.** `tests/translate/test_choked_tables.py`: the exit refines as the entrance
  does (14 against 15 intervals; 37 before), the table is accurate to its sampling
  noise against the exact-distance formula from 0.1 µm to 2 m, and the canonical Φ
  table has at most 40 intervals and 10 in the last centimetre. Both fail without the
  change (37 and 123).

**Not done.** The coupled-flow and reject-free routes keep the interval cap. They now
take longer steps near a choked end, but no route with both a coupled flow and a choked
end was measured.

### GPU qualification (phases 1–3)

**Scope.** 2026-10-05; Opus 5.5 (AI) for John Travers. A rented RTX 4000 Ada
(Ubuntu 22.04, CUDA 13.1, GCC 13, `SOLPROP_KOKKOS_CUDA_ARCH=ADA89`), solprop
`e9ecfd6` and solprop-kokkos `ec4b67b`; 37 minutes, about US$0.50. A correctness
qualification only: the card's double-precision rate is a sixty-fourth of its single,
so no timing was taken.

- **The build** reports `execution_space='Cuda'`, cuFFT and cuBLAS, kernel contract 18.
- **SEP 14's own tests on the CUDA build** (`test_compiled_multistep`,
  `test_dense_output`, `test_step_planes`, `test_step_limits` and the Kokkos coupled-flow
  and moving-guide drivers): 64 of 66 pass. The Adams march, interpolated planes and
  step planes follow `numpy_ref`'s trajectory on the device. The two failures were the
  statistics-at-interpolated-planes test's 1e-9 bound: round-off flips a step decision
  on the device, as on Linux `numba_cpu`, and the planes then differ at the
  interpolant's error (4.3e-7). The bound is now 1e-5, still far below the percent a
  statistic read off the march's state would show.
- **The Kokkos suite** (fast lane) fails only where solprop-kokkos's tests predate
  SEP 3 (the removed `ADK` import, H2's Raman carrier); that drift is being repaired
  separately.
- **Every Kokkos-parametrised solprop test** (`-k kokkos` over the backends and
  drivers): 187 of 586 fail, and none on a number. 185 find no `kokkos_cuda` row in the
  convergence records (`spatial_accuracy`, and `backend_equivalence` for the polarised
  and free-space routes), which have never been recorded on a CUDA build; one expects
  four host threads where a CUDA build reports one; one is the statistics bound above.
  Recording those rows belongs to the standing GPU requalification of the spatial
  routes.

### Phase 4 — The collar study

**Scope.** 2026-10-05; Opus 5.5 (AI) for John Travers. Derivation and NumPy prototypes
only (`benchmarks/sep14/collar.py`, results in `data/collar.jsonl`), as §4.5 bounds it.

**The candidate.** A Strang split for the Adams march: half a collar, the collar-free
Adams step, half a collar, with the history of derivatives carried by the collar over
each step, `exp(-alpha (z_n - z_j) / 2)`, the way the march already carries it by the
linear flow. The prototype is a fixed-step PEC4/5 in the interaction picture.

**The witnesses** (worst error against a converged in-stage reference; observed order
as the step halves):

| witness | in-stage | per step (`split_step`) | Strang, history carried |
|---|---|---|---|
| Q7, `u' = b u - a u` | 3.6e-4 → 6.2e-6, order 3 | 1.7e-2 → 4.4e-3, order 1 | 2.1e-4 → 2.9e-6, order 3 |
| linear time-local source, no dispersion | — | order 1 | order 3 (to 8.3e-9) |
| linear time-local source, dispersion | — | order 1 | order 2.2–3.5 (to 1.7e-8) |
| Q8, NLS with Kerr, window | unstable at 64 and 128 steps; order 3 from 256 | order 1 from 256 steps (to 2.0e-6 at 2048) | **order 1** from 256 steps (to 1.3e-6 at 2048) |
| Q8, NLS with Kerr, collar | as the window | order 1 (to 8.6e-5) | order 1 (to 7.8e-6) |

(The orders are capped at three by the start-up at order one.)

- **Q7 is answered:** carrying the history by the collar removes the review's
  second-order local defect when the collar commutes with the source.
- **Dispersion is not the obstacle.** With a linear, time-local source the scheme
  keeps order 2–3 although the collar and the dispersion do not commute.
- **The nonlinear source is.** A Kerr derivative stored at a state and carried by the
  collar is scaled by `exp(-alpha h / 2)`, but the Kerr source of the damped state
  scales by `exp(-3 alpha h / 2)`. Every history entry carries a defect of order
  `alpha h`, and the method is first order wherever the collar holds field. No linear
  transport of the history follows a nonlinear source; only re-evaluating the source,
  at a call per history entry, would.
- **What the split still buys.** It is stable where the in-stage collar is not (64
  steps, `alpha h / 2 = 0.19` above PEC4/5's 0.16), and its error in the collar is ten
  times smaller than the per-step split's, while its window error is the same.

**Exit criterion (§4.5): not met.** The candidate fails the non-commuting witness
with a nonlinear source, which is every production case. No implementation follows.
The in-stage collar stays the Adams default; the per-step split stays admitted only for
runs whose collar holds no field, as SEP 13 left it. The Strang variant could be
offered later as a stability-first option for collars that hold field, with its
first-order error documented, but nothing measured here asks for it.

### Phase 6 — Studies

**Scope.** 2026-10-05; Opus 5.5 (AI) for John Travers. Each study ends in a
recommendation; neither adds code to the package.

**A higher-order pair: IP-DOP853** (`benchmarks/sep14/dop853.py`, `data/dop853-gnlse.jsonl`).
Hairer's 8(5,3) pair from SciPy's coefficients, its combined estimator
`e5**2 / sqrt(e5**2 + 0.01 e3**2)` at estimator order seven, patched into
`numpy_ref`'s trial step. Twelve calls a step on a first-same-as-last backend against
IP-DOPRI5's six. On the GNLSE, the case Adams loses, against a DOP853 reference at
`rtol=1e-10`:

| `rtol` | DOPRI5 calls; field | DOP853 calls; field |
|---|---|---|
| 1e-5 | 16 434; 1.3e-2 | 30 828; 1.4e-2 |
| 1e-6 | 26 250; 1.6e-3 | 41 064; 1.3e-3 |
| 1e-7 | 41 364; 1.9e-4 | 54 432; 1.6e-4 |
| 1e-8 | 64 884; 1.5e-5 | 72 012; 2.1e-5 |

- Its steps grow 1.32 times per decade of `rtol` against DOPRI5's 1.57, so it gains
  19 % a decade. At matched field error it costs 1.55 times DOPRI5's calls at 1e-3, 1.3
  times at 1.5e-4 and 1.15 at 1.5e-5; the crossover is near 5e-6, tighter than any
  production run.
- On a stiff interaction picture both pairs lose order (a soliton under fixed steps
  gives 4 for DOPRI5 and 4–6 for DOP853), which narrows DOP853's advantage.
- **Recommendation: not adopted.** Verner's pairs were not tried; with the same order
  reduction they would need their own measurement.

**A band-weighted norm** (`benchmarks/sep14/band_norm.py`, `data/band_norm.jsonl`). On
the canonical capillary the far-UV band (90–200 nm) is held to a target either by
tightening `rtol` with the weak norm or by weighting its bins by `w` in the weighted
norm at a fixed `rtol`:

| run | calls | field | UV field |
|---|---|---|---|
| weak, `rtol` 1e-3 | 742 | 1.35e-3 | 0.25 |
| weak, `rtol` 1e-4 | 1015 | 1.45e-4 | 1.3e-2 |
| weak, `rtol` 1e-5 | 1484 | 1.67e-5 | 1.6e-3 |
| `rtol` 1e-2, `w` 1e-2 | 959 | 1.45e-4 | 1.6e-2 |
| `rtol` 1e-2, `w` 1e-3 | 1393 | 1.44e-5 | 1.5e-3 |
| `rtol` 1e-3, `w` 1e-2 | 1414 | 1.43e-5 | 1.5e-3 |

- Weighting the UV by `w` behaves as tightening `rtol` by `w`: the same field error,
  the same UV error, and 5–8 % fewer calls. The step is shared by every band, and the
  UV is made where the strong field evolves, so holding it costs the same steps either
  way.
- **Recommendation: no band-weighted default.** Tighten `rtol` to hold a weak band;
  the weighted norm stays available, and a growth cap above 1.1 (phase 1) is not
  unlocked by it.

## Part III — History

- **2026-10-04:** Draft created from SEP 13's comparisons and a survey of the RDW, CtC
  and GNLSE cases (`benchmarks/sep14/`). Draft / Not reviewed; no implementation begun.
- **2026-10-04:** Revised after the follow-up measurements (`benchmarks/sep14/`):
  - many save planes and dense output;
  - the GNLSE calls-against-error curve, which corrects "no gain" to "neutral to 1.2× in
    calls";
  - an N = 4 soliton at tight tolerance;
  - an interleaved, load-sampled split of a step into source evaluations and linear work
    on every backend, which shows that calls overstate the gain on compiled backends;
  - tolerances shown comparable with Luna's.

  The Adams port moved to phase 2 as an optimisation and runtime-against-precision
  study with a wall-clock gate.

- **2026-10-04:** GPT-6 (AI) independently reviewed `781db1e`; review status changed
  to In review. Added Q7–Q17 with numerical counterexamples, scope and compatibility
  questions, and validation suggestions. Requests revisions before acceptance; no
  implementation or numerical default changed.
- **2026-10-04:** Revised for the review:
  - Q7–Q17 answered and reflected in Part I;
  - the collar becomes a bounded study;
  - native DOPRI5 dense output, with an admission matrix;
  - the step-limit changes scoped to diagonal flows;
  - a defined first-step rule;
  - a diagnosis-first table repair;
  - a legal fused Adams schedule;
  - per-phase versioning;
  - a measurement protocol without extrapolated timings, and a quantitative acceptance
    criterion;
  - the GNLSE comparison at `rtol=1e-6` (maintainer's review);
  - the default question left to phase 2's gate.
- **2026-10-04:** Reviewed by John Travers (human); review status `Reviewed`. Accepted by
  John Travers. Part I is frozen.
- **2026-10-04:** Implementation started by Opus 5.5 (AI) on
  `feature/sep-0014-stepper-efficiency-1`, after SEP 13 was set Final.
- **2026-10-04:** Phase 1 (step limits) reported: interval cap off and a start-up rule
  on the diagonal-flow routes, growth cap kept at 1.1; kernel contract 17, output
  schema 10, ADR-038. Not merged.
- **2026-10-04:** Phase 3 (RK dense output) reported ahead of phase 2's report: DOPRI5's
  continuous extension on every CPU backend, the admission matrix, kernel contract 18,
  output schema 11, ADR-039. Not merged.
- **2026-10-04:** Phase 3 extended at the maintainer's request: resume from an
  interpolated plane only with `resume="interpolated"`; `step_planes` saves the march's
  own steps; `numba_cpu` and Kokkos keep the reused stage across save planes.
- **2026-10-04:** Phase 2 (Adams on `numba_cpu` and Kokkos) reported with the Linux
  runtime study: Adams 0.55–0.9 of IP-DOPRI5's time on gas-filled capillaries on
  `numba_cpu`, even on Kokkos, 2.5–3.5 times slower on the GNLSE; no default changed.
- **2026-10-05:** Phase 5 (choked-end tables) reported: the floor band at a choked exit
  is the sampling noise of floating-point positions near the exit; the builder accepts a
  tail at that floor (ADR-040), 30 Φ intervals instead of 123 with Φ unchanged to
  round-off.
- **2026-10-05:** Phase 4 (the collar study) reported: the exit criterion is not met, so
  no implementation follows; phase 6 (DOP853, a band-weighted norm) reported: neither
  adopted.
- **2026-10-05:** GPU qualification of phases 1–3 on a CUDA build (RTX 4000 Ada): SEP
  14's tests pass on the device; the other failures are records never made on CUDA and
  the separate SEP 3 test drift.
- **2026-10-05:** The two follow-ups merged (`numba_cpu`'s pyFFTW with passive OpenMP
  waiting on Linux; two results made platform-stable), with `main` at optmat 0.3.0. The
  phase 2 `numba_cpu` rows re-measured after the fix: Adams 0.55–0.8 of IP-DOPRI5's time
  on the canonical case and 0.75–1.0 on RDW, even on the CtC slice, 2.3–2.7 times as
  long on the GNLSE.
- **2026-10-05:** Diagnosis of the Adams march's limits (`benchmarks/sep14/adams_limits.py`):
  the CtC slice's collar, five times the whole case's, held the march on its stability
  limit; phase-mismatched forcing limits it on every dispersive case. SEP 15 drafted.
- **2026-10-05:** The CtC slice re-timed with the whole case's collar (Adams 0.87–0.96 of
  IP-DOPRI5's time on `numba_cpu`, 0.91–1.00 on Kokkos), and the runtime study repeated on
  macOS, agreeing with the workstation. The gate passed at the merge of the two follow-ups
  and `main` (`305c797`): slow lane, default lane and 100 % coverage.
- **2026-10-05:** The maintainer's decisions recorded (Part II status): no default
  changed, the save interface accepted, ADR-040 accepted, the Strang split not offered
  (SEP 15's Q5 instead), the ledger's machine names without the host. Merged to `main`
  with SEP 15 as a draft.
- **2026-10-05:** Set `Final` by John Travers and archived; the open items moved to the
  roadmap.
