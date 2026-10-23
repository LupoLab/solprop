# SEP 16 — Unified multistep integration and work–precision qualification

| | |
|---|---|
| **SEP** | 16 |
| **Title** | Unified multistep integration and work–precision qualification |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra (2026-10-05); Opus 5.5 (review revision, 2026-10-05) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | Opus 5.5 (AI); John Travers (human) |
| **Type** | Feature |
| **Created** | 2026-10-05 |
| **Resolution** | Accepted by John Travers, 2026-10-05, in the SEP 16 review discussion |
| **Requires** | [SEP 12](sep-0012-tdse-nonlinear-response.md), [SEP 13](sep-0013-expensive-response-steppers.md), [SEP 14](sep-0014-stepper-efficiency.md) |
| **Supersedes** | [SEP 15](sep-0015-exponential-adams.md) |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (coding agent): phases 1–4 on `feature/sep-0016-unified-multistep-1` to `-4` (solprop-kokkos on `-2` and `-3`), merged 2026-10-06; phases 3b and 5 on `-3b` and `-5`, phase 6 and the GPU campaign on their own branches (solprop-kokkos on the same names), merged 2026-10-06 and 2026-10-07; the default integrator, the active-band norm, resume with history and the interrupt handling merged 2026-10-07; closed on `feature/sep-0016-final` |
| **Manual pages** | [The Adams methods](../../../guide/numerics/adams.md); [Choosing a stepper](../../../guide/numerics/choosing-a-stepper.md); [The solver](../../../guide/numerics/solver.md) (the two integrators, the error norm's active band); [Output](../../../guide/results/output.md) (schemas 13–15, the march's state); [Running](../../../guide/setup/running.md) (resume and SIGTERM) |

## Part I — Proposal

### 1. Abstract

Deliver one production multistep family, alongside the existing interaction-picture
Runge–Kutta (RK) default, with two formulations that share one history, controller and
lifecycle: the existing transported-source (interaction-picture) Adams method and an
exponential Adams method. Deliver it in gated slices. First, exponential Adams for
constant linear operators — the GNLSE, constant-profile fibres and capillaries, constant
general modal guides and the free-space routes — with NumPy, Numba and Kokkos CPU support
for both formulations on those routes, which also completes the compiled routes of the
existing method. Then varying diagonal operators, whose measured gate passed during
review, through a trial-local constant reference with an exact residual (§4.2). Then
coupled sections and absorber treatments,
if the earlier slices show that the method earns them. Optimise source calls,
coefficient construction, memory traffic and total propagation time at matched measured
accuracy, and deliver a reproducible work–precision gallery and evidence-based solver
guidance. Success is judged by correctness and backend parity on a published support
matrix, and by source calls and wall time against RK and the existing method at a common
measured error. The target is strongly dispersive broadband propagation with costly
sources; SEP 12's time-dependent Schrödinger equation (TDSE) response must work with
whatever is delivered but is not a performance target. CF4 coupled sections and other
general varying noncommuting coupled generators remain roadmap work; neither a universal
speedup nor a change of the default solver is proposed.

### 2. Motivation and scope

SEP 15 investigated why interaction-picture Adams can lose its source-call advantage in
broadband propagation. Its initial exponential prototype saved calls but lost time in
weight construction. Subsequent compiled weights and a compensated constant reference
improved both calls and time on the tested canonical, developed resonant dispersive wave
(RDW) and short continuum-to-comb (CtC) cases. These are experimental results, not a
qualified public integrator or evidence for all geometries, tolerances and responses.
The [benchmark ledger](../../benchmarks.md)
records the matched error ceilings, references, windows and limitations.

**What the evidence supports.** The study's results are end-to-end errors against
refined references, valid for the windows measured; what differs between them is how
far they generalise.

| Case | Operator | Measured extent | Exponential formulation at a common 1e-4 error |
|---|---|---|---|
| GNLSE supercontinuum | constant | 0.2–1 mm windows | about five times fewer calls than RK; on Numba 26–32 % less time than RK and 22 % less than current Multistep; no separable gain over Multistep on NumPy |
| CtC launch, full grid | constant | 0.17 mm | 46 % fewer calls and 42 % less time than current Multistep; 54 % less time than RK |
| Canonical capillary | varying | full 4.2 m | 28 % fewer calls and 11–13 % less time than current Multistep; 16–21 % less time than RK |
| Developed M2/RDW | varying | 2 mm, about five steps | 74 % fewer calls and 53 % less time than current Multistep |
| M2/RDW, review gate | varying | full 1.5 m | 5.5 times fewer calls and 3.4 times less time than current Multistep; 11 times fewer calls than RK (5.8 on field and UV errors alone) and about 5.5 times less time |

The constant-operator gains need no compensation. The full canonical run shows a modest
gain on a cheap source. The short RDW window could not show accumulated error, so the
full M2/RDW propagation was run during review as the varying-operator gate (§7.2, Q8):
from launch to the choked exit its gain is larger than the window's, the largest
measured, on a cheap source. At the same requested tolerance the exponential
formulation's steps there are as long as RK's (about 1 mm at `rtol=3e-6`) but cost one
source call instead of six, and about five times longer than current Multistep's, whose
transported history limits the linear phase per step to about one in the populated UV
bins (SEP 14's diagnosis). RK's errors in the dim spectral bins are large beside its
field error, so a common error that includes them pushes it to a tighter tolerance;
that doubles its ratio. At the loose
accuracy at which SEP 12 runs its TDSE response on the canonical capillary (`rtol` of
1e-2 to 3e-3 in its tolerance check), the study found no call advantage at a 1 %
error (134 against 139 calls at the historical thresholds; 403 against 412 at a common
1 % field/UV error), which is why the TDSE response is a compatibility requirement here
rather than a target. Selections between 1 % and 1e-4 error were not reported.

**Who it is for.** Strongly dispersive, broadband propagation, especially where source
calls are costly: gas-filled capillaries generating resonant dispersive waves, multimode
Raman-active fibres and capillaries, fibre supercontinuum and the free-space routes. On
the last, the operator is constant, diffraction gives very large $|\operatorname{Im}D|$
at high transverse wavenumber, where a driven bin favours the exponential frame (SEP 15
§4), and each call is a full spatial transform plus the response, so fewer calls convert
almost directly into time. That expectation is untested, so a spatial pilot comes early
(phase 2). Large spatial states additionally require bounded storage and
memory-efficient execution.

**Implementation baseline.** Implementation starts from the new `main` that includes
SEP 12's integration and its reconciliation with preceding mainline changes. Record
that exact solprop commit and the corresponding solprop-tdse and solprop-kokkos revisions
in the first implementation report. Re-audit the interfaces, capability matrix and
qualification exceptions against that baseline before coding. The study branch is not
that baseline. SEP 12 is final and merged; its record admits the TDSE response on
`numpy_ref` only, with Numba and Kokkos propagation refusing it under an accepted
backend exception, and its native-current response adds an opt-in temporal source
filter that is part of the source assembly $F$ here. Its dependency on SEP 12 is for the
integrated interface and a compatibility check, not a demand to finish deferred atomic
physics, compare solvers on the TDSE response or perform a full-length TDSE run.

**Required coverage.** “All problem classes” means the existing physical models on that
baseline, with the numerical scope below; it does not implement physics from other
unaccepted proposals, and it does not admit a combination that the RK route refuses on
physical grounds. A published matrix must distinguish implemented, qualified,
physically unavailable, gated and deliberately deferred combinations, per formulation
and backend. The constant-operator slice is unconditional; the extensions follow the
gates of §11.

| Dimension | Constant-operator slice (phases 2–3) | Gated extensions (phases 4–5) |
|---|---|---|
| Routes | GNLSE; mode-averaged and multimode fibres/capillaries with constant profiles; constant general modal guides; radial, Cartesian, line and hybrid planar spatial routes | The same waveguide routes with profiles varying along z (phase 4); constant-Schur and frozen-residual coupled sections (phase 5) |
| State | Field and envelope representations and each route's admitted polarisation and batch layouts (a lockstep batch only with a prescribed schedule, as for RK) | The same |
| Linear representation | The `constant` and `separable` Φ representations | `expansion` and `table` (phase 4); coupled sections (phase 5, Q3) |
| Profiles | — | Pressure/material gradients, changing effective areas and source maps, and basic adiabatic tapers with declared kinks (phase 4) |
| Noise | None, input and modified noise wherever the RK route admits them | Modified noise along gradients (phase 4); a route or flow that refuses a noise model under RK (the hybrid planar route, every coupled flow) keeps refusing it |
| Sources | Existing supported Kerr, Raman, plasma and spectral-response combinations; SEP 12's response as a compatibility check | The same |
| Boundaries | The spectral collar (part of $D$, already exact), explicit temporal and spatial absorbers, including appreciable field in the absorbing region | Implicit or inner absorber treatments, only if Q4 shows them needed (phase 5) |
| Lifecycle | Adaptive steps and orders, dense observations within SEP 14's admission matrix, source diagnostics, budgets, fresh resume, batching and prescribed/replayed schedules (§5) | The same with varying operators |
| Execution | NumPy oracle, Numba CPU and Kokkos CPU, for both formulations; separately declared GPU qualification | The same |

**Adiabatic means an explicit physical approximation.** Varying diagonal propagation,
normalisation, effective area and source maps must work. A requested mode-coupling
connection must never be silently omitted to admit a taper. Two adiabatic routes exist:
the capillary's `taper_coupling=False`, an explicit opt-out of the taper coupling that a
multimode capillary taper carries by default, and the mode-set route's moving multimode
guides, which carry no connection and warn by name (`AdiabaticTaperWarning`). The
general modal route's moving maps always carry a frozen-residual coupled flow and refuse
to run without one ([general modal propagation](../../../guide/waveguides/general-modal.md)). No
adiabatic construction exists there; building one is new route physics, outside this
proposal, so moving general guides enter the scope only with the coupled sections of
phase 5. Conversion between modes remains physical content, not a solver option to
discard.

**Coupled operators.** Existing coupled flows have three section kinds
(`solprop.translate.coupled.SectionMethod`): `constant_schur`, a constant coherent block
integrated by Schur matrix-function actions; `frozen_residual`, a block frozen within
each section whose variation the explicit lab-frame source already carries; and `cf4`, a
commutator-free fourth-order flow of the whole section generator with no explicit
residual. The first two have exactly the structure of the exponential formulation: a
constant reference within a section plus an explicit residual (§4.5). They carry the
coherent capillary wall, the default multimode capillary taper coupling, and a split
block of a declared basis or a moving guide on the general route. They are in scope for
the exponential formulation only, in the gated phase 5 (Q3); matrix dimensions,
conditioning, action error, loss and inverse-flow safety must be specified before
admission. The transported formulation keeps refusing coupled flows. New coupling
physics is not included. A case that cannot meet these requirements is named for review
rather than silently removed from the coverage matrix.

**Outside scope.** CF4 sections and other general varying noncommuting coupled flows,
and moving-basis conversion requiring them; noise models on routes or flows that refuse
them under RK (their derivations are roadmap items); new atomic/material models; the
TDSE response as a performance target, a TDSE work–precision campaign or a full-length
TDSE run; delivery of exponential RK, nonlinear-Jacobian methods or automatic
RK/Multistep switching; unqualified learned phase frames. Existing spatial/atomic
resolution and model-validity limitations are not cured by a smaller propagation error.
Additional paid computation needs its own agreed budget; none is proposed here.

### 3. Context

The shared march is `solprop.backends.multistep.AdamsMarch`; the public configuration is
`solprop.stepper.multistep.Multistep`. SEP 13 introduced the one-source-call PEC march;
SEP 14 qualified compiled implementations and studied stability, absorber, table and
observation costs. SEP 15's benchmark implementations are evidence and derivation
witnesses, not a production interface to copy unchanged.

Today the reference multistep route admits more geometries than the compiled routes.
`numpy_ref` steps `Multistep` on every route except a coupled flow and the modified
noise model; `numba_cpu` and Kokkos also refuse the free-space, planar and general modal
routes, because their compiled source hooks are tied to the fibre/capillary pipeline.
`backend="auto"` resolves to `numpy_ref` whenever an integrator is given. A prescribed
schedule beside an integrator is refused at translation. Dense observation is refused
beside the split-step temporal collar, a coupled flow and the modified noise, for RK and
Adams alike (SEP 14). Every coupled flow refuses every noise model, under RK too. These
restrictions mix absent backend wiring, absent numerical qualification and physical
refusals; the successor must separate them.

The linear operator has four Φ representations ([stepper design](../../stepper.md) §4):
`constant`; `expansion`, spectral arrays weighted by exact integrals of resolved scalar
parameter paths; `table`, a piecewise Chebyshev table whose coefficients resolve $D$
itself and are integrated exactly; and `separable`, the constant free-space operator
assembled per element from one-dimensional arrays without coefficient volumes. The
spectral collar is part of $D$ in all four. The SEP 15 constant-reference study handled
only the first two.

The governing sources are [architecture](../../architecture.md),
[conventions](../../conventions.md), [kernel contract](../../kernel-contract.md),
[stepper design](../../stepper.md), [low-call stepping](../../../guide/numerics/adams.md),
[noise](../../../guide/media/noise.md), [spatial decompositions](../../../guide/setup/spatial-decompositions.md)
and [general modal propagation](../../../guide/waveguides/general-modal.md).
`solprop.backends.driver.StepDriver` and each backend define source, noise and
observation semantics. SEP 12 defines the expensive-response adapter and its budget,
accuracy and output obligations; as its implementation record describes, the adapter
reserves the atomic budget before each trial, labels every evaluation as startup,
predictor or corrected work, and restores logical call ledgers, never numerical history,
on resume. Both formulations must preserve those semantics. Review the companion
backend's actual contract on the implementation baseline; a method name alone does not
establish compatibility.

This proposal inherits SEP 15's remaining delivery obligations: a qualified numerical
method, varying-operator support, public interface, backend parity, documentation and
matched-error performance evidence. SEP 15 is superseded, not certified as fully delivered.

### 4. Theory

#### 4.1 Equation and source identity

Use SI units and the internal Fourier convention of conventions §2. Write

$$u'(z)=D(z)u(z)+F(z,u(z);\eta(z)),\qquad \Phi'(z)=D(z).$$

Here $z,h$ are in metres, $D$ in m⁻¹, $\Phi$ dimensionless, and $F$ has units of the
internal field amplitude per metre. $F$ denotes the existing complete source assembly,
including the chosen absorber semantics, not a new physical model. For the diagonal
case, $U(b,a)=\exp(\Phi(b)-\Phi(a))$ is the exact homogeneous flow relative to the supplied
table. Its accuracy relative to the physical operator is a separate translation error.

Each source sample has a position, the actual field at which it was evaluated, its
source value and its accepted/predicted identity. A PEC history retains the source at
the predictor even though the accepted field is corrected. Changing reference operators,
rejecting trials or producing observations must not change that identity.

#### 4.2 Two formulations in one multistep family

The transported-source formulation interpolates
$U(z_n,s)F(s,u(s);\eta(s))$, as in SEP 13, which applies Lawson's interaction-picture
transformation [7] to an Adams history. It remains available with its established
semantics. Exponential Adams instead interpolates in the untransported frame and
integrates the polynomial exactly against the linear exponential [1–3]. For constant
$D$, distinct normalised nodes $\theta_j=(z_j-z_n)/h$ and Lagrange basis
$\ell_j(\theta)=\sum_k c_{jk}\theta^k$ give

$$u_{n+1}=e^{hD}u_n+h\sum_j w_j(hD)\,F_j,\qquad
w_j(x)=\int_0^1 e^{(1-\theta)x}\ell_j(\theta)\,d\theta
       =\sum_k c_{jk}k!\varphi_{k+1}(x),$$

$$\varphi_r(x)=\sum_{m=0}^{\infty}\frac{x^m}{(m+r)!}.$$

At $x=0$ the weights reduce to the ordinary variable-step Adams weights
$a_j=\int_0^1\ell_j(\theta)\,d\theta$ [8, §III.5]. With $q$ previous samples
($\theta_j\le0$), the predictor uses those $q$ nodes; one new source is evaluated at the
predicted endpoint, and the corrector uses that sample at $\theta=1$ with the same $q$
old samples.

For varying diagonal $D$, choose a constant reference within each attempted step:

$$B_n=\frac{\Phi(z_n+h)-\Phi(z_n)}{h},\qquad
R_n(s,u)=F(s,u;\eta(s))+(D(s)-B_n)u.$$

Then $u'=B_nu+R_n(s,u)$ is an exact rearrangement for any constant $B_n$; the chord
additionally makes the endpoint homogeneous flow exact, $e^{hB_n}=U(z_n+h,z_n)$. The
reference changes after a step-size change, including rejection, so reconstruct
residuals from retained raw samples and their actual states. Do not reevaluate the
expensive physical source just to change $B_n$. For constant $D$, $R_n=F$: ordinary
exponential Adams is a storage and execution specialisation. For varying $D$ with no
physical source, an interpolated compensation need not cancel exactly even at the
endpoint, whereas the uncompensated chord $e^{hB_n}u_n$ is exact there. Require that
exact linear-only path when the complete source, including absorbers, is structurally
absent. Do not infer absence from a source sample happening to vanish, or claim exact
nonlinear-bin evolution from the homogeneous factor alone.

**The rate is the derivative of the same representation.** The rearrangement relies on
$\int_{z_n}^{z_n+h}(D(s)-B_n)\,ds=0$, which holds only if the rate $D(s)$ in the residual
is the derivative of the cumulative representation that defines the flow. The tables
provide it exactly: a z table stores the exact Chebyshev integral of a resolved $D$, so
the analytic derivative of its coefficients is that resolved $D$, and an expansion's
scalar integral tables likewise differentiate to its resolved integrands. Evaluate $D$
that way rather than from an independent physical evaluation, so that table error
remains a translation error shared by every method rather than an inconsistency between
flow and residual.
A table's per-interval fits are continuous in $\Phi$ but, beyond its resolution
tolerance, not in $D$: the residual has small jumps at every adaptive breakpoint, not
only at declared kinks. Bound them against the step tolerance; declared hard kinks,
where $D$ itself jumps or kinks, restart the history as today. Form $D(s_j)-B_n$ before
multiplying by the stored state: reconstructing $R$ as a stored full right-hand side
minus $B_nu_j$ cancels catastrophically in bins where $|Du|\gg|F|$.

**Where the compensation is smooth.** The compensation $C_{n,j}=(D(z_j)-B_n)u_j$ is
linear in the propagated state, so it carries each bin's own free phase: across the
history it rotates like $e^{i\operatorname{Im}D\,(z_j-z_n)}$, while its amplitude grows
with the distance from the step, $|D(z_j)-B_n|\approx|D'|\,|z_j-z_n-h/2|$. Interpolating
it in the untransported frame, as the SEP 15 study did, is therefore the case SEP 15 §4
found to favour the transported frame, however slowly $D$ varies: a rotating sample
sequence extrapolated by a polynomial. Once $|\operatorname{Im}D|h\gtrsim1$ its error is
$O(|D'|h^2|u|)$ per step, lower order than the $O(|D'|h^3|F|)$ error of omitting the
compensation, and proportional to the field rather than the source. A scalar check made
in review (Part III) confirms this at every order from PEC2/3 to PEC5/6: with 1 %
variation of $D$ over $10^3$–$10^4$ steps and $\omega_0h=0.6$–5, the untransported
compensation's final relative error was $6\times10^{-5}$–$6\times10^{-2}$, against at
most $2\times10^{-5}$ for the uncompensated chord.

Integrate the linear compensation instead in the frame of the reference flow, where it
is smooth. Interpolating the integrand $e^{(1-\theta)hB_n}C_n(\theta)$ with the ordinary
Adams basis on the same nodes gives

$$u_{n+1}=e^{hB_n}u_n+h\sum_j w_j(hB_n)F_j
 +h\sum_j a_j\,e^{(1-\theta_j)hB_n}\bigl(D(z_j)-B_n\bigr)u_j ,$$

with the same samples, states and identities. Every factor $e^{(1-\theta_j)hB_n}$ is a
forward flow over at least $h$, so a damped bin needs no inverse flow, and the material
source keeps the untransported frame in which a driven bin is smooth. In the scalar
check this transported compensation converged at the formula's order when
$\omega_0h\ll1$, stayed within about an order of magnitude of the uncompensated chord
when $\omega_0h\gtrsim1$, and for source-free propagation with 50 % variation of $D$
reached $10^{-10}$ at $\omega_0h=0.16$, where the untransported compensation's error was
still $10^{-5}$.
It costs one exponential per sample and element per trial unless node spacings repeat.

**What the gate found.** On the full M2/RDW and canonical propagations (§7.2, run during
review) the two treatments agree to two significant digits at every observation plane
with the same calls: there the compensation is negligible beside the material source's
interpolation error. The transported one costs 6–14 % more time on these cheap sources
once each sample's factor is formed once per trial in a compiled loop: on RDW at 1e-4,
3.50 s against 3.17 s untransported and 11.84 s for current Multistep; on canonical, 560
against 518 and 577 ms. That remainder is the per-sample exponential. The compensation
cannot simply be dropped: in adaptive scalar runs the uncompensated chord's error is
invisible to the predictor–corrector estimate (14 steps and a 1.2e-3 error even at
`rtol=1e-8`), and on the full RDW run it needs 27 % more calls at a 3e-5 error and
misses 1e-5 within its ladder, where both compensations meet it. Where the compensation
dominates, the scalar runs show the untransported treatment's error is seen by the
estimate, at 1.3–10 times the transported treatment's steps, yet ends 5–60 times above
its tolerance against 1–9 times transported. The cases where that matters, tapers and
steep gradients, are what phase 4 adds, so the transported compensation is delivered
(Q8). Its exponentials can probably be avoided. Since
$e^{(z_n-z_j)B_n}=U(z_n,z_j)\,e^{\psi_n(z_j)}$ with
$\psi_n(z_j)=\Phi(z_j)-\Phi(z_n)-B_n(z_j-z_n)=O(|D'|(z_n-z_j)^2)$, accumulate each
retained sample's flow $U(z_n,z_j)$ and phase $\Phi(z_n)-\Phi(z_j)$ with one
multiplication and one addition per accepted step, and evaluate $e^{\psi}$ by a short
series. Phase 4 measures that against the untransported witness.

**What the literature establishes.** The cheap reference plus exact residual is
motivated by [4], whose stability results are for a negative-real, diffusive spectrum
and do not transfer to predominantly imaginary dispersion. Hochbruck and Ostermann prove
fixed-step convergence of order $k$ for the explicit $k$-step exponential Adams method
under an analytic semigroup [3, Theorem 4.3]; their Remark 4.4(c) extends the argument
to strongly continuous semigroups, which include the unitary group of lossless
dispersion, when the nonlinearity is Lipschitz on the state space. That result is for
fixed steps, the explicit form and a constant operator, and leaves variable steps and
orders outside its scope [3, §5]. It does not cover the PEC recurrence, the
trial-dependent reference or the transported compensation, which need their own
analysis; the constant in its bound grows with the nonlinearity's Lipschitz constant,
which is large for a developed continuum.

Candidate predictor orders are 1–5 (the existing `Multistep` admits up to six), but
every shipped order and variable-order transition needs convergence and stability
evidence. Do not label the corrected global order from the predictor order alone. The
difference between predictor and corrector estimates a predictor defect, not a
guaranteed bound on the accepted state. Controller exponents and neighbouring-order
estimates must follow the actual recurrence, including its stored predictor samples and
both parts of the residual.

#### 4.3 Dense output, weights and stability

For fractional position $0\leq\xi\leq1$, the same interpolants give

$$u_d(z_n+\xi h)=e^{\xi h B_n}u_n+
 h\sum_{j,k}c_{jk}k!\xi^{k+1}\varphi_{k+1}(\xi hB_n)F_{j}
 +h\sum_j A_j(\xi)\,e^{(\xi-\theta_j)hB_n}C_{n,j},\qquad
 A_j(\xi)=\int_0^\xi\ell_j(\theta)\,d\theta,$$

with the last sum replaced by the first form applied to $C_{n,j}$ if the untransported
treatment is delivered. Retain that accepted trial's reference and samples for its
observations. At an interior point $e^{\xi hB_n}$ is not generally the true varying
homogeneous flow: the compensation supplies the correction. In particular, the endpoint
identity alone does not prove linear-only dense accuracy. For the endpoint sample and
$\xi<1$ the factor $e^{(\xi-1)hB_n}$ is an inverse flow, which a strongly damped bin
amplifies; evaluate it stably or guard it with a separate argument, as SEP 14 required
for the transported interpolant. Qualify endpoint and interior errors separately; a
specialised exact linear-only observation path must be derived and tested if adopted.
Observations are not new history samples and cannot conceal fresh source calls.

Use stable series/recurrence or equivalent bounded-error moment evaluation; independent
contour/quadrature or augmented matrix exponentials validate it [1, 5]. Test complete
weights as well as individual moments. Node conditioning and cancellation amplify
moment error. Bound the induced state uncertainty $h\sum_j\delta w_j R_{n,j}$ in the
same norm used by the stepper, with separately controlled operator/action and inner-solve
errors. An embedded difference cannot detect a shared coefficient error.

Sweep the split problem $u'=(D+\lambda)u$, strongly damped forced equilibria, and forcing
with independent source and linear phase rates, with constant and varying $D$. Exact
handling of $D$ does not eliminate explicit residual stability limits: the compensation
is an explicit linear term, and the transported formulation's PEC4/5 imaginary-axis
interval is only 0.04 ([low-call stepping](../../../guide/numerics/adams.md)). Variable
steps/orders introduce history-root and conditioning issues beyond a fixed-step scalar
stability plot [8, §III.5]. Declared source features, step limits and restart
safeguards remain necessary: no estimator detects an event missed by every sample.

#### 4.4 Modified noise

Preserve solprop's existing model, defined in the noise manual and `solprop.noise`:

$$\eta(z)=\eta_0\exp(i\operatorname{Im}\Phi(z)).$$

One draw is fixed for the run. Conditional on it, $F(z,u;\eta(z))$ is a deterministic,
nonautonomous source, so source interpolation is mathematically admissible; this is not
a fresh stochastic increment at every step. Evaluate the phase at each sample's actual
position, without loss/gain amplification of $\eta$. Rejection does not redraw it;
resume preserves the realisation and absolute phase origin. Store $F$ with the exact
predictor state and position used. The compensation $(D-B_n)u$ acts on the propagated
state, not on $u+\eta$.

The draw rotates with each bin's own free phase. Wherever a source term is linear in
$\eta$ — the Kerr terms linear in the draw in a weak bin, for instance — the
untransported source carries that rotation, which is the case SEP 15 §4 found to favour
the transported formulation. In noise-dominated weak bins the exponential formulation's
advantage may therefore shrink or reverse; measure it rather than assume the noise-free
result.

Audit material injection, absorber inputs, physical observables and cached response
outputs separately against RK. A compact notation $F(z,u;\eta)$ must not become a blanket
substitution of $u+\eta$ into terms where current physics uses $u$. Noise changes source
smoothness and may reduce the speed advantage; correctness and performance are separate
gates. Spatial/modal draw normalisation, masks, polarisation and basis transformations
need independent checks on each route connected to a multistep method. Preserve the
existing approximation to noise transport along gradients; improving that physical
normalisation is separate roadmap work, not an unannounced integrator change.

The displayed phase formula applies to the diagonal representation. Every coupled flow
refuses every noise model under RK, because a draw's transport through a coupled flow is
not derived; both formulations keep that refusal, and its derivation stays on the
roadmap. SEP 14 refuses dense observation beside the modified noise for RK and Adams
alike; lifting that refusal for either formulation needs its own argument and test.

#### 4.5 Absorbers and coupled operators

The spectral collar is part of $\operatorname{Re}D$ and both formulations integrate it
exactly. Temporal and spatial absorbers are diagonal in time or space, not in the
representation where $D$ is diagonal. Their explicit source treatment remains the
correctness baseline. If it dominates the cost frontier, qualify an implicit cheap term
or cheap inner integration with interpolated expensive material source [6]. For an
implicit endpoint term $C$, the solve has the form

$$(I-hW_{\rm end}(hB_n)C)u=b.$$

$W_{\rm end}$ and $C$ generally do not commute. The field-grid operator may be
real-linear; tests and solver packing must respect that. Bound solve/inner-integration
error against the outer error budget and handle failure through explicit rejection or
failure status. Retain the actual sample identities of cheap and expensive terms.
Do not silently reduce expensive-source evaluations by adding an uncontrolled boundary
split. Existing explicit split-step settings retain their documented error semantics.

For a constant coupled section the same moment formula uses matrix-function actions
[2, §2]. A nonnormal matrix may require a Schur/action method and conditioning
estimates rather than diagonalisation. The rearrangement
$u'=B\,u+[F+(A(z)-B)u]$ is exact for any constant $B$ and generator $A(z)$, so the
exponential formulation needs no exponential-of-integral identity for a coupled family.
A frozen-residual section already carries $(A(z)-B_s)u$ in its explicit source, with
$B_s$ the section's frozen block; its scalar part, a multiple of the channel identity,
commutes with $B_s$, so its chord adds to the reference without changing the section's
Schur or eigenbasis. Its residual is linear in the state, so it is integrated in the
reference flow's frame as in §4.2, by forward matrix actions only. A section change
rebuilds the residual history with the new block, as a step-size change does, or
restarts it; a declared kink restarts it. The compiled backends already execute
constant-Schur and frozen-residual sections for RK, and refuse CF4. The
transported formulation would also transport every history sample through the coupled
flow, including inverse actions whose conditioning grows with loss, which the
exponential formulation avoids; Q3 therefore admits coupled sections for the
exponential formulation only. A CF4 section has no explicit residual and
exponentiates the generator at Gauss nodes inside each stage; extending it needs a
separate propagator, residual and error analysis and is excluded from delivery here.

### 5. Proposed interface

Keep the existing RK default and the existing meaning of `Multistep()`. Add an explicit
formulation option (Q1):

```text
integrator=Multistep()                           # Existing transported-source method.
integrator=Multistep(formulation="interaction")  # The same, spelled out.
integrator=Multistep(formulation="exponential")  # Constant or compensated varying D.
```

`"interaction"` matches the interaction-picture vocabulary of IP-DOPRI5 and the manual.
No new public class per geometry, frame or coefficient strategy. Constant/varying
specialisation follows operator structure, not case names or trial timing; until
phase 4 delivers varying operators, the exponential formulation refuses a varying one
by name. Record the resolved formulation, order/controller settings, coefficient
accuracy policy, compensation treatment and any absorber strategy in run metadata. The
method's record name distinguishes the formulations (for example
`exponential-adams-pec5/6-variable`). The default formulation is kept out of the
recorded method string, which `StepperRecord.method` hashes from output schema 9, so
existing records and resumes are unchanged. Do not switch formulations mid-run without
a defined history conversion/restart; automatic switching is outside this proposal.

Reuse call budgets, order bounds, error norms and observation controls where their
meaning is unchanged. Units/ranges and supported order combinations must be explicit,
with each formulation's own qualified maximum order. Only expose an additional absorber
control if the qualified implementation needs one; name its mathematical treatment and
failure behaviour.

Prescribed schedules are a required supported mode with declared-kink checks and
truthful error diagnostics, not an assertion that every supplied step is safe. For a
multistep method a schedule fixes the accepted positions and, for exact replay, the
order sequence and restart points; the predictor–corrector estimate is still computed,
recorded and warned on, as for RK's fixed mode ([stepper design](../../stepper.md) §8).
Lockstep batches follow from schedules (ADR-013).

A capability check names the unsupported physical feature, representation or backend
before an expensive response is invoked. No silent RK fallback, backend fallback or
omission of coupling/noise is permitted. Admission describes correctness support;
guidance separately describes demonstrated efficiency. Existing physics incompatibilities
remain explicit rather than being mistaken for integrator omissions. Once the compiled
backends admit a problem, `backend="auto"` resolves as it does for RK, per problem,
instead of always to `numpy_ref`; phase 3 makes that change with its admission tests.

### 6. Implementation strategies

Use three responsibilities: shared propagation services (source, linear actions, norms,
observations and budgets); a common Adams history/controller lifecycle; and the chosen
history integration formulation. RK shares the services, not the Adams recurrence.
Keep backend hot loops free of physics-specific decisions. Source evaluation must not
require manufacturing RK stages or duplicating each geometry's response implementation;
the study's untransported source paid an identity propagator and repeated anchor work
through the RK stage path, which the shared service must avoid. The compiled source
services built for the free-space, planar and general modal routes in phase 3 serve
both formulations, which is what completes the existing method's compiled routes.

NumPy provides inspectable reference operations. Numba then fuses weight application,
residual reconstruction and history combinations with existing deterministic arithmetic.
Kokkos CPU follows the same contract with resident history and bounded scratch. Audit
`KernelProblem`, protected driver hooks and stage bundles together. Review changes
against SEP 12's actual response adapter. A response unavailable on a backend must remain
explicitly unavailable; this SEP does not assume an atomic GPU response is a Kokkos
propagation implementation.

**The seam.** `StepperSettings.integrator` carries the formulation from phase 2, which
bumps `SCHEMA_VERSION`, updates the literal compatibility tests and changes the output
schema for the solver metadata. Phases 3, 4 and 5 bump `SCHEMA_VERSION` again for any
new hook, stage-bundle field or kernel argument they need, and solprop-kokkos follows the
version on the same delivered revisions. Phase 3's resident history partly covers the
roadmap's device-resident Kokkos Adams step, for both formulations.

**Constant operators.** A separable operator stores no coefficient volume, so its
moments must be formed per element inside the fused application and never cached as
volumes. That is $O(M)$ transcendental work per trial against the transported
formulation's per-element propagator, to be measured on large Cartesian states.
Constant operators need no state history for compensation.

**What the varying slice adds.** Given the constant-operator slice on all three
backends, phase 4 adds: rates by representation (an expansion's scalar integrands at
each sample, `n_terms` scalars; a z table's analytic derivative, one Clenshaw pass over
its coefficient volumes per sample, the cost of a $\Phi$ evaluation, stored or
recomputed by measured cost); sample states retained beside the source samples; residual
reconstruction when the reference changes; the transported compensation kernel of §4.2,
with its exponential-free form as a cost target; the dense-output guard of §4.3; and
breakpoint and kink handling. The free-space work does not carry over, because those
operators are constant. The extension is moderate in code, a fraction of the
constant-operator slice, and mostly qualification; its gate, which passed in review
(Q8), was about whether it pays, not about effort. Coupled sections (phase 5) are a
larger step: per-section matrix moment actions on all three backends, section edges and
conditioning certificates.

For $M$ complex state elements and $q$ history samples, retained history costs $O(qM)$;
spatial transform scratch remains bounded by the existing chunking policies. Varying
operators retain sample states, and for a z table possibly rates, beside the source
samples; an expansion retains scalar table weights rather than full rate volumes.
Compare storage/recomputation choices against a measured memory ledger. Avoid a full
$q\times M$ weight tensor when streamed application can replace it. Cache moments only
when their actual arguments agree; similar step ratios are insufficient. Account for
dense-output storage, batches, rejection rollback and device copies.

Optimise through component profiles and controlled ablations: source transforms,
response work, moment evaluation, compensation transport, history traffic, norms,
observations and setup. Persistent FFT plans, fusion and buffer reuse are candidates; no
relaxed arithmetic or reduced physical grid solely to obtain a speedup. Retain
deterministic reductions and thread invariance. Each accepted optimisation has a
numerical witness and a benchmark ledger entry; negative experiments remain evidence
rather than public options.

No new dependency is proposed. Stop an optimisation line when repeatable measurements
show no material gain over timing spread or when memory/accuracy costs make it dominated.
Maintain the measured accuracy–cost frontier instead of endlessly tuning one case.

### 7. Verification and validation

#### 7.1 Numerical contract

- Independently check moments/weights at zero, near evaluator boundaries, large imaginary
  arguments, strong damping, resonance, unequal nodes and ill-conditioned histories.
  Recover ordinary Adams and exact polynomial forcing where applicable.
- Establish at least three-level convergence for every shipped order with prescribed
  forcing and nonlinear feedback, separately for endpoint and dense observations.
  Vary step sequences and operator representations; include hard-kink landings.
- For the varying slice, march manufactured varying-operator problems with the actual
  PEC recurrence for at least $10^3$ steps: source-free and driven bins,
  $|\operatorname{Im}D|h$ from 0.1 to 10, relative variation of $D$ over the run from
  $10^{-3}$ to order one, and lossy bins. Compare both compensation treatments of §4.2
  with the uncompensated chord as a control, and include table breakpoints with
  resolution-level rate jumps.
- Test exact linear endpoint flow and interior convergence separately, source-free and
  forced dissipative cases, conservation/symmetry where the physical problem has it,
  and manufactured noise-driven nonautonomous problems, including weak bins whose
  noise-linear source rotates with their own free phase, for both formulations.
- Exercise startup, order changes, multiple rejections, rollback, budgets, resume,
  fixed/replayed schedules, masks, source-statistic identity and independent batch members.
  Count every source call, including startup, rejected and diagnostic calls.
- Check scalar and noncommuting cheap-absorber reference problems against independent
  matrix exponentials or refined ODE solutions. If a cheap solve is shipped, exercise
  convergence failure, residual certification and its effect on outer rejection.
- For coupled sections (phase 5), check their matrix moments against augmented matrix
  exponentials at and beyond the declared conditioning limits, and the residual history
  across section edges and kinks.
- Compare NumPy/Numba/Kokkos CPU weights/actions, individual source evaluations, steps,
  short adaptive runs, dense values, ledgers and memory. Check thread invariance and
  the interpreted-kernel coverage lane. GPU results are a separate marked qualification;
  unavailable hardware is recorded, never presented as a pass.

Tolerances follow round-off/conditioning bounds or demonstrated reference refinement,
not fitted differences from one run. The NumPy oracle is not the sole accuracy witness.
Different adaptive trajectories are assessed through physical errors and accounted calls;
bitwise trajectories are required only where the existing deterministic contract demands them.

#### 7.2 Physical coverage matrix

The implementation report enumerates the actual tests by route, representation, source,
noise, absorber and backend. Use deliberate interaction coverage instead of an enormous
blind Cartesian product: every newly connected source pipeline and every high-risk
combination gets a dedicated case. Small unit cases may share physical limits, but one
successful capillary is not qualification of free space.

**The varying-operator gate (phase 1).** Before phase 4, using the SEP 15 study
implementation extended with the transported compensation: the full 1.5 m M2/RDW
propagation with the untransported compensation, the transported compensation and the
uncompensated chord, each against a refined reference, with the study's field, weak-band
and 100–150 nm metrics at every observation plane; and the cost of the transported
compensation measured on the full canonical run against the study's untransported
candidate and current Multistep. Phase 4 proceeds if a treatment keeps a call and time
advantage over current Multistep on the full RDW run at two adjacent resolved targets
and does not lose the canonical advantage; otherwise varying operators stay with the
existing interaction-picture formulation and the result is reported (Q8). The gate was
run during review and passed (Part III).

Campaign cases for the constant-operator slice (phase 3, after the phase 2 pilot):

1. Broadband GNLSE supercontinuum windows, including a longer mature window.
2. CtC launch and a developed comb window with modified noise, matched realisations,
   and Raman-sideband/weak-band observables. Launch-only evidence is insufficient; the
   developed input comes from an authorised worked-example parent propagation (Q9).
3. Radial and Cartesian spatial propagation in smooth, focusing/plasma and
   active-boundary regimes; line and planar controls exercising their distinct
   transforms, and the line route's noise map (the planar route carries no noise).
4. Constant general modal guides, with loss and polarisation where admitted.
5. A TDSE compatibility check, not a performance row: both formulations drive SEP 12's
   adapter on its admitted backend for a few steps in a reduced CPU configuration, like
   SEP 12's integration witness (three atomic calls took 54 s with two threads),
   exercising budget reservation, evaluation labels, rejection and fresh resume, and
   reconciling the call ledgers. No physical claim and no paid compute.

Campaign cases for the gated extensions:

6. Full canonical propagation (phase 4) with the established observation planes and both
   legacy tapered-UV and sharply bounded UV metrics identified distinctly.
7. The full M2/RDW propagation and the developed RDW window (phase 4).
8. Adiabatic tapers and gradients with changing mode maps or areas, loss and profile
   kinks (phase 4).
9. Constant-Schur and frozen-residual coupled cases (phase 5), with independent action
   references.

Prepare expensive developed inputs through documented worked examples and retain their
provenance. Benchmarks use short windows, normally about ten steps or seconds of work.
The authorised exceptions are the full canonical propagation, the full M2/RDW
propagation and the CtC parent propagation that develops the comb (Q9); each is manual
and untimed preparation or a campaign run, never a test or CI workload. Long preparation
is never hidden inside a performance claim. Full CtC and full TDSE propagation are not
required.

#### 7.3 Work–precision gallery and acceptance

For every campaign case except the TDSE check, publish paired logarithmic plots of
measured error against (a) total expensive source/RHS calls and (b) wall time. The error
axis and observation planes are common across methods. Sweep requested tolerances and
plot achieved errors; equal `rtol` is not equal accuracy. Include RK, the existing
transported formulation and the new exponential formulation, using the same physics,
grids, masks, boundaries and draws. Method-specific settings are permitted only when
disclosed and reproducible. Show failed, budget-limited and unstable points rather than
deleting them from the comparison.

Measure worst-plane complex-field error and case-relevant observables: weak-band energy,
Raman sidebands, temporal peaks, spatial fluence/width, plasma or modal content as
appropriate. Define bands and near-zero normalisation before the sweep. Reference
refinement must bound uncertainty below the smallest claimed resolved error; plot its
floor and do not rank methods below it. Separate propagation discretisation from spatial,
atomic and source-model errors. For modified noise use paired fixed draws for numerical
convergence and several seeds for robustness; this is not a new validation of the
physical noise model or a full coherence ensemble.

Record source calls, accepted/rejected attempts, startup and mature work, orders,
coefficient/linear/absorber actions, solver iterations, observations, peak memory and
component time. Define an RHS call at the physical response boundary. Cheap inner actions
are separate counts and are all included in wall time. Additional diagnostic evaluations
are never free.

Use warmed, repeated, interleaved timings on a quiet machine with fixed thread policies;
report median and spread and preserve raw results, environment, seeds, grids, reference
identities and revisions. Distinguish translation, setup/JIT, march and output costs.
The primary complete-propagation comparison includes required solver setup, coefficients,
source work and observations; a separate steady-march panel may diagnose them. Include
both a representative workstation CPU and a second CPU architecture where available;
never extrapolate CPU speedups to GPU or a short window to a full expensive run.

Completion requires all correctness/support rows and backend obligations of the
delivered slices to be satisfied, or an explicit reviewed scope amendment. Performance
acceptance requires reproducible call and elapsed-time improvements over RK at at least
two adjacent resolved accuracy targets in each claimed winning regime, with the
improvement exceeding timing spread. Compare against the existing formulation too and
explain regressions. Repeat the SEP 15 witnesses on the new baseline rather than
importing their speedups. A regime where RK wins remains a documented result, not a
reason to loosen its error metric. Fewer than 140 canonical calls is an investigated
frontier at a stated accuracy, not a correctness criterion or a universal production
promise.

**The delivery decision (end of phase 3).** The exponential formulation stays a public
option only if the constant-operator campaign shows it winning at least one regime
against both RK and the existing formulation by the criterion above. Otherwise it is
kept as a comparison method, like `ComparisonMultistep`, the compiled routes of the
existing method are still delivered, and phases 4 and 5 return to review (Q10).
Phase 5 starts only after review of the phase 3 and phase 4 campaigns.

### 8. Documentation

Extend the low-call manual with the two formulations, source/sample identities, the
reference and its compensation, error control, stability, noise, boundaries and the
support matrix. Update solver, noise, spatial and general-modal pages,
stepper/backend/contract design and public docstrings. Record the formulation design as
an ADR in phase 2 and the compensation treatment in phase 4. Record functionality in
`CHANGELOG.md`; record deliberate behavioural differences on `docs/guide/workflows/luna.md` without
presenting a solver change as changed physics.

Deliver a worked-example gallery of the §7 comparisons for the delivered slices, with a
common plotting/data format and downloadable machine-readable results. Scientific
figures use standard plotting tools and exportable vector/raster output. Keep expensive
campaign execution manual and separate from figure regeneration; never hand-edit
generated gallery files. A small tutorial demonstrates ordinary selection without
running a campaign on every build. The benchmark ledger links the evidence and records
optimisation ablations and the gate results.

Guidance must distinguish supported from recommended: what to use for cheap or expensive
responses, constant or varying operators, modified noise, rapid nonlinear transitions,
large spatial states and active absorbers. Explain how to recognise stability-limited
history, reference-limited error and source-limited accuracy. Give measured ranges and
counterexamples, not a universal phase-rate threshold or a claim that higher order wins.
State that the canonical capillary at loose accuracy, where SEP 12's TDSE response runs,
is not a regime in which the exponential formulation was found to save calls. No runtime
autotuner or automatic method switch is needed to provide this guidance.

### 9. Compatibility and impact

RK remains the default; existing `Multistep()` keeps the transported-source behaviour
and its recorded method string. The new formulation is opt-in. Existing output fields
retain their meaning; new solver metadata is versioned where required.
`backend="auto"` with an integrator moves from always `numpy_ref` to the RK resolution
rule once the compiled backends admit the problem. Fresh resume reconstructs history
under documented startup semantics rather than claiming continuation of unavailable
samples. Existing physics/noise approximations are preserved and named. Partial or
gated delivery refuses unsupported combinations explicitly. Any new protected seam
surface is versioned in solprop and checked in solprop-kokkos on the same delivered
revisions (§6).

The work is implemented on the post-SEP-12 mainline, not by substituting the SEP 15 study
checkout for it. Preserve SEP 12's adapter, calibration, budget and qualification findings;
new numerical work must not erase physical limitations or silently broaden atomic scope.

### 10. Alternatives considered

- **Separate solver classes for each experiment:** multiplies interfaces, settings and
  qualification paths despite a shared recurrence. Use one family and explicit formulations.
- **One separate class for the exponential formulation:** keeps `repr(Multistep())`
  unchanged but duplicates every shared setting and its validation. A formulation field
  whose default is kept out of the record achieves the same compatibility.
- **Deliver every slice unconditionally:** the original draft's plan. It commits the
  route breadth and the varying-operator work before the compiled constant-operator
  results show where the method pays. The gates of §11 keep that work but order it
  behind the evidence.
- **Stop at constant-D exponential Adams:** a possible outcome of the gates, not a plan:
  it leaves the canonical and RDW gains, tapers and gradients unclaimed.
- **Retain variable-D quadrature as the universal hot path:** transparent oracle but the
  measured weight cost motivates the compensated reference for production.
- **The study's untransported compensation:** as accurate as the transported one on both
  full physical runs and 6–14 % cheaper on these cheap sources, but where the
  compensation dominates its error is under-reported to the controller (§4.2). Kept as
  a test witness for phase 4's cost and accuracy comparisons (Q8).
- **The uncompensated chord:** endpoint-exact homogeneous flow, but second order with
  varying $D$, and its error is invisible to the predictor–corrector estimate. A
  diagnostic control only.
- **A reference fixed per table interval or section,** as the frozen-residual coupled
  flow does: no residual reconstruction within an interval, but no endpoint-exact
  homogeneous flow and a larger residual. Worth measuring only if reconstruction cost
  dominates.
- **Target the TDSE response:** at its loose operating accuracy on the canonical
  capillary the study found no call advantage; compatibility, not performance, is
  required.
- **Replace RK or auto-select a method now:** unsupported by universal performance evidence;
  selection trials are especially costly with expensive sources. Publish guidance first.
- **Deliver every phase frame and history experiment:** adds choices without a robust
  selection rule. Keep these as research witnesses unless new evidence changes the scope.
- **Implicit full nonlinear solves or exponential Rosenbrock methods:** source Jacobian
  actions can consume the expensive calls this project seeks to save. Cheap absorber
  treatment is a narrower, measurable option.
- **Only lift refusal checks:** does not establish noise history, geometry pipelines,
  moving-map consistency or backend storage correctness. Derivation and tests precede admission.

### 11. Phases

Each slice includes its relevant tests, docstrings, manual changes and a report. Apply
`AGENTS.md` phased-SEP gates during development; run the full gate, slow/coverage lanes
and Kokkos-selected CPU qualification on the final integrated tree before `Final`.
Do not mark complete while required checks fail. GPU qualification uses its own marked
lane and device evidence; any final exception must be explicit in the accepted scope.
Phases 4 and 5 are gated; if a gate closes, the SEP can become `Final` on the delivered
slices with the gated work recorded as a reviewed scope amendment and moved to the
roadmap.

1. **Baseline, contract and the varying-operator gate.** Record the post-SEP-12
   revisions, audit capabilities and specify history/source/action interfaces. Fix the
   performance baseline (Q2). The varying-operator gate of §7.2 ran during review with
   the study implementation; repeat it on the baseline only if the march it measured has
   changed. Deliver the capability matrix and independent
   numerical witnesses. No public method is promised merely by adopting an experimental
   class.
2. **Constant-operator exponential Adams on NumPy.** The formulation field, its record and
   the ADR, with the `SCHEMA_VERSION` and output-schema bumps; constant and separable
   representations; qualified orders, controller and dense output; the modified-noise
   lifecycle; schedules; explicit absorbers; all constant-operator routes, including the
   free-space routes and constant general modal guides, with shared source services for
   both formulations. Exit through analytic/convergence, rejection and same-draw RK
   comparisons, a short CtC case, the TDSE compatibility check, and an early spatial
   pilot: short radial and Cartesian windows comparing the calls of RK and both
   formulations at common measured error.
3. **Compiled delivery of the constant-operator slice.** `numba_cpu`, then Kokkos CPU
   (two merges if convenient), for both formulations on every route of phase 2, which
   completes the existing method's compiled routes; resident Kokkos history;
   `backend="auto"` per problem; optimisation and memory ledgers; the constant-operator
   campaign of §7.2 and its work–precision plots. GPU qualification where a compatible
   device/build is available. Ends with the delivery decision of §7.3 (Q10).
4. **Varying diagonal operators — gated by phase 3; the phase 1 gate passed in review.**
   The compensated reference with the transported compensation (Q8), on all three
   backends: expansion and table rates, gradients, changing areas and source maps,
   adiabatic tapers, modified noise along gradients, kinks and breakpoints; the
   exponential-free transport of §4.2 as a cost target against the untransported
   witness; the full canonical and full M2/RDW campaigns.
5. **Coupled sections and absorber treatments — gated by review of phases 3 and 4.**
   Constant-Schur and frozen-residual sections for the exponential formulation (Q3), and
   an implicit or inner absorber treatment only if Q4's ablation shows it needed.
6. **Gallery, guidance and final integration.** Complete the gallery for the delivered
   slices, independent reference refinements, repeated timings and negative results;
   resolve or move the remaining open questions; run the final gates against the
   integrated mainline and publish supported/recommended matrices and reproduction
   commands.

### 12. Open questions

- **Q1 — resolved in review, 2026-10-05:** `formulation="interaction"` (the default) and
  `"exponential"`; RK and `Multistep()` unchanged; the default formulation kept out of
  the recorded method string.
- **Q2 — phase 1/2:** qualify fixed and variable orders, controller/ratio limits and
  coefficient/table/action error budgets. Specify a baseline before performance tuning;
  do not infer it from the best single benchmark point.
- **Q3 — scope resolved in review, 2026-10-05; qualified in phase 5:** constant-Schur and
  frozen-residual coupled sections for the exponential formulation only; CF4 sections
  and every noise model beside a coupled flow stay refused, and the transported
  formulation keeps refusing coupled flows. Define conditioning/action limits. If a
  required case needs general noncommuting machinery, bring the specific boundary to
  review; no silent narrowing of the coverage matrix.
- **Q4 — phases 3/5:** choose whether absorber-active performance requires an implicit
  cheap term or inner integration, using a bounded ablation. Correct explicit support is
  mandatory. Any added treatment must meet independent solve, stability and cost gates.
- **Q5 — resolved in review, 2026-10-05:** the TDSE response is not a performance target;
  both formulations must work with it, shown by the compatibility check of §7.2 in a
  reduced CPU configuration, with no paid compute.
- **Q6 — phase 1/3:** audit backend/response exceptions inherited from SEP 12 and available
  GPU builds. CPU parity is mandatory for the new integrator on admitted source pipelines;
  an unavailable atomic backend or GPU device must have an explicit reviewed exception,
  not an inferred pass or an unbounded port of the atomic solver.
- **Q7 — before each campaign:** freeze error observables, resolved target ranges, seed
  ensemble and representative windows before selecting winning configurations. Preserve
  both canonical UV definitions and independently quantify reference uncertainty.
- **Q8 — resolved in review, 2026-10-05:** the gate passed and phase 4 proceeds with the
  transported compensation. Both compensations meet the criterion of §7.2 with equal
  calls and accuracy on the full runs; the transported one is robust where the
  compensation dominates and costs 6–14 % more time on cheap sources, which phase 4
  aims to remove with the exponential-free transport of §4.2. The untransported
  treatment remains a test witness; the uncompensated chord is excluded because its
  error is invisible to the controller.
- **Q9 — resolved in review, 2026-10-05:** the full canonical propagation, the full
  1.5 m M2/RDW propagation (about a minute per run on `numba_cpu`, judging by the 45 s
  preparation of its 1.2 m input) and a worked-example parent propagation that develops
  the CtC comb are authorised; all manual and untimed, like the existing RDW input
  preparation.
- **Q10 — resolved in review, 2026-10-05:** the constant-operator slice goes through all
  three CPU backends first, with the delivery decision of §7.3 at the end of phase 3;
  varying operators follow the phase 1 gate, and the remaining breadth follows review of
  the phase 3 and 4 campaigns.

### 13. References

1. S. M. Cox and P. C. Matthews, “Exponential time differencing for stiff systems,”
   *Journal of Computational Physics* **176**, 430–455 (2002).
   [doi:10.1006/jcph.2002.6995](https://doi.org/10.1006/jcph.2002.6995).
2. M. Hochbruck and A. Ostermann, “Exponential integrators,” *Acta Numerica* **19**,
   209–286 (2010). [doi:10.1017/S0962492910000048](https://doi.org/10.1017/S0962492910000048).
3. M. Hochbruck and A. Ostermann, “Exponential multistep methods of Adams-type,”
   *BIT Numerical Mathematics* **51**, 889–908 (2011).
   [doi:10.1007/s10543-011-0332-6](https://doi.org/10.1007/s10543-011-0332-6).
4. M. Caliari, F. Cassini, L. Einkemmer and A. Ostermann, “Accelerating exponential
   integrators to efficiently solve semilinear advection-diffusion-reaction equations,”
   *SIAM Journal on Scientific Computing* **46**, A906–A928 (2024).
   [doi:10.1137/23M1562056](https://doi.org/10.1137/23M1562056).
5. A.-K. Kassam and L. N. Trefethen, “Fourth-order time-stepping for stiff PDEs,”
   *SIAM Journal on Scientific Computing* **26**, 1214–1233 (2005).
   [doi:10.1137/S1064827502410633](https://doi.org/10.1137/S1064827502410633).
6. A. Demirel, J. Niegemann, K. Busch and M. Hochbruck, “Efficient multiple time-stepping
   algorithms of higher order,” *Journal of Computational Physics* **285**, 133–148
   (2015). [doi:10.1016/j.jcp.2015.01.018](https://doi.org/10.1016/j.jcp.2015.01.018).
7. J. D. Lawson, “Generalized Runge–Kutta processes for stable systems with large
   Lipschitz constants,” *SIAM Journal on Numerical Analysis* **4**, 372–380 (1967).
   [doi:10.1137/0704033](https://doi.org/10.1137/0704033).
8. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
   Nonstiff Problems*, 2nd ed., Springer (1993), §III.5.
   [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1).

References [1–3, 5] support exponential integration and weight evaluation; [4] motivates
reference/residual separation; [6] motivates cheap inner integration; [7] is the
interaction-picture transformation of the existing formulation; [8] covers
variable-step Adams weights and stability. Their fixed-step, diffusive or nonstiff
analyses do not certify solprop's adaptive dispersive PEC scheme. The modified-noise
convention is the existing solprop model linked in §3, not a new stochastic model
derived here. No published performance factor substitutes for the proposed campaign.

## Part II — Implementation record

SEP 15's benchmark code, independent numerical witnesses and raw evidence are inherited
research inputs; they do not satisfy this proposal's coverage or final qualification
gates. The implementation began from the post-SEP-12 mainline; one report per phase
follows the review's handoff.

### Handoff from the review — before phase 1

Not a phase report: what the review learned that phase 1 should start from. Part I
governs where this note and Part I differ.

**Evidence and reusable witnesses.** `benchmarks/sep16/` at revision `efe657e` on
`bench/sep-0016-gate` extends the SEP 15 study (`benchmarks/sep15/` on the same branch).
`compensation.py` holds the three compensation treatments as subclasses of the study's
constant-reference march, including the fused transported kernel; `gate.py` the
full-run campaign and its full-RDW score; `fused_check.py` the treatment timings;
`scalar/` the fixed-step and adaptive PEC scalar checks of §4.2, a starting point for
§7.1's long-march tests; its README the reproduction commands and thread settings.
These are research witnesses to re-derive from, not code to promote.

**The baseline march.** On main after SEP 12, `solprop.backends.multistep.AdamsMarch`
passes a `purpose` (`"startup"`, `"predictor"`, `"corrected"`) to
`StepDriver._adams_source`, and `_afford` calls `StepDriver._reserve_source_calls`
before every trial; SEP 12's atomic budget and labels depend on both, so the
exponential formulation must use the same lifecycle. The compiled PEC family runs a
fused path (`_attempt_fused` with `AdamsKernels`) whose transport pass the exponential
formulation does not have; it needs its own kernels. `_adams_source` returns the source
in the anchor's frame with `U(position, anchor)`: the study obtained an untransported
source by passing equal anchor and position, which built an identity propagator and
repeated anchor work until a study-local mixin bypassed it on Numba. Phase 2 should
add an explicit lab-frame source hook rather than reuse that route. The existing dense
observation already guards strongly damped bins (`|Re ΔΦ| > 1` keeps the anchor's flow);
§4.3's endpoint inverse flow needs the analogous guard.

**Places that change together.** Admission: `solprop.backends.numpy_ref.backend._refuse_adams`,
`NumbaCpuBackend._refuse_adams`, solprop-kokkos `backend._refuse_adams`,
`solprop.driver.simulation._admit_integrator` (which resolves `backend="auto"` to
`numpy_ref` for any integrator), `StepperSettings` (which refuses an integrator with a
schedule) and `solprop.translate.problem._admit_dense`. Record: `_stepper_record` stores
`repr(integrator)` in `StepperRecord.method`; Q1's rule (the default formulation kept
out of the record) needs a custom representation, because `field(repr=False)` would hide
a non-default formulation too.

**Rates.** `LinearOperator.phi_coefficients` are the z table's Chebyshev coefficients of
$\Phi$, so $D$ is their analytic Chebyshev derivative; an expansion's scalar integral
tables differentiate the same way (the study's `FrozenLinearMarch.rate` rescales by
`2/Δedge`); `phi_constant` is $D$ itself. The study handled only constant and expansion
operators.

**Measurement lessons.**

- Short windows are dominated by startup and understated the RDW gain; gate-level
  claims need the authorised full runs (Q9).
- Full-run band scores need a floor (early planes hold almost nothing in the UV or weak
  bands); the canonical score's `weak` entry is its sharp 90–200 nm band. Report each
  comparison with and without the weak-bin score: IP-DOPRI5's weak-bin error is large
  beside its field error and doubles its call ratio on RDW.
- References: the full RDW reference at `rtol=1e-10` takes about 72 000 calls; the
  canonical sharp-UV refinement against `1e-9` is only 4.1e-6, so canonical targets
  below 1e-4 need a `1e-11` reference.
- Tolerance ladders in factors of about three give about ±30 % granularity in the
  selected calls; use finer ladders near any target that carries a claim.
- Only fixed order four was measured on full runs with the exponential formulation;
  variable order and its controller are Q2's work.
- The fused transported compensation's remaining 6–14 % cost over the untransported one
  is the per-sample exponential; §4.2's exponential-free form is the phase 4 target.
- The uncompensated chord's error is invisible to the predictor–corrector estimate; do
  not offer it even as a cheap mode.
- Time on a quiet machine with one thread per pool, interleaved and repeated; the
  review's timings were same-process comparisons on a machine under background load.

### Phase 1 report — baseline, contract and witnesses

**Scope.** Phase 1 of §11, 2026-10-06, implemented by Opus 5.5 (coding agent) working
autonomously at John Travers's request, on `feature/sep-0016-unified-multistep-1`
(branched from `main` at `01fb731`). Not merged: the branch, and the phase 2 and 3
branches built on it, await review. No `solprop-kokkos` change.

**Baseline revisions.** solprop `main` at `01fb731` (kernel contract `SCHEMA_VERSION`
19, output schema 12); solprop-kokkos `main` at `dc4e7f4`, which implements contract 19;
solprop-tdse at `3e99e319`, the revision solprop's optional `tdse` dependency pins
(that extra is not part of the default environment). The varying-operator gate's march
(`AdamsMarch` at the gate's base, `62316d7`) differs from this baseline only by SEP 12's
lifecycle — evaluation labels, the source-call reservation before each trial and its
release, and the restart bookkeeping at a natural stop on a kink — and not in any
arithmetic of a step, so the gate was not repeated (§11, phase 1).

**Delivered.**

- `src/solprop/backends/exponential.py`: the reference evaluation of
  $\varphi_0\ldots\varphi_8$ for complex arrays and the exponential Adams weights of
  §4.2–4.3 on distinct nodes, for a whole step or a fraction of one.
- `src/solprop/stepper/multistep.py`: `lagrange_coefficients`, the monomial
  coefficients $c_{jk}$ of the Lagrange basis the weights are built from.
- `tests/backends/test_exponential_weights.py`: moments against augmented matrix
  exponentials and against weighted quadrature of their defining integrals; complete
  weights against an augmented exponential that integrates an arbitrary polynomial
  independently of $c_{jk}$ and of the $\varphi$ evaluator; ordinary Adams recovered at
  $x=0$; refusals.
- `tests/exponential_scalar.py` and `tests/stepper/test_exponential_scalar_march.py`:
  a scalar fixed-step PEC march in both formulations with the three compensation
  treatments, and the long-march witnesses of §7.1.
- `benchmarks/sep16/capability_audit.py` and `data/capability-audit.json`: a probe
  that builds every route and variant and records what each backend and method
  admits, by construction and a short run, not by reading the refusal code.
- `benchmarks/sep16/campaign.py` and `data/baseline/`: the work–precision harness and
  the baseline measurement of Q2 (below); `data/inputs/gnlse-input.*`, the developed
  GNLSE state the SEP 15 study prepared, with its fingerprint and checksum.

**The capability matrix on the baseline.** Probed on `numpy_ref` and `numba_cpu` for
RK and `Multistep()` (Kokkos was not installed when the probe ran; its column is read
from `solprop_kokkos.backend._refuse_adams`). Every combination below was constructed
and run a short distance, or refused by name at the layer stated.

| Route (Φ representation) | RK, all backends | `Multistep()`, `numpy_ref` | `Multistep()`, `numba_cpu` | `Multistep()`, Kokkos (from code) |
|---|---|---|---|---|
| GNLSE (constant) | admitted | admitted | admitted | admitted |
| mode-averaged, envelope and multimode capillaries, constant pressure (constant) | admitted | admitted | admitted | admitted |
| capillary with a pressure gradient (expansion) | admitted | admitted | admitted | admitted |
| constant general modal guide (constant, modal tables) | admitted | admitted | refused, backend | refused, backend |
| radial free space (constant) | admitted | admitted | **admitted** | admitted by its Adams refusal; route support not probed |
| Cartesian and line free space (separable) | admitted | admitted | refused, backend | refused, backend |
| hybrid planar (separable) | admitted | admitted | refused, backend | refused, backend |
| coherent wall, taper coupling, moving general guide (coupled flow) | admitted | refused, backend | refused, backend | refused, backend |

| Variant | RK | `Multistep()` |
|---|---|---|
| `input` noise | admitted wherever the route admits noise | the same |
| `modified` noise | admitted wherever the route admits noise | refused at the backend, every backend |
| any noise beside a coupled flow | refused at translation (physics: no derived transport) | the same |
| noise on the planar route | not offered by `from_planar` | the same |
| prescribed schedule | admitted | refused at translation |
| dense observation | admitted except beside a split-step collar, a coupled flow or `modified` noise (translation) | the same |
| `backend="auto"` | resolves to the preferred backend | resolves to `numpy_ref` |

So the refusals fall into three kinds: **physical** (noise beside a coupled flow,
whose transport is not derived; RK refuses it too); **numerical, not yet qualified**
(Adams with `modified` noise, a schedule, a coupled flow); and **backend wiring**
(the compiled Adams source hook serves the modal pipeline only). Phases 2 and 3 move
the second and third kinds for the constant-operator slice; the first stays.

Two findings differ from §3. `numba_cpu` already steps `Multistep()` on the radial
route: its refusal keys on a separable operator, and the radial operator is constant,
not separable; it runs, but has no parity test yet (phase 3 adds one). And the
split-step refusal of dense observation binds only where a temporal collar exists; the
GNLSE, free-space and planar problems carry none, so dense observation is admitted
with `absorber_scheme="split_step"` there.

**Interfaces specified for phases 2–4.**

1. *Lab-frame source hook.* `StepDriver._adams_lab_source(lab, position, *, purpose)`
   returns the complete source assembly $F$ at a lab-frame field and position — the
   noise draw at that position's absolute phase, the in-stage absorbers, SEP 12's
   adapter with its labels and reservations — untransported. The default refuses,
   naming the backend; `numpy_ref` implements it in phase 2 through its reference
   passes, and the compiled backends in phase 3. It replaces the study's route of
   equal anchor and position through `_adams_source`, which built an identity
   propagator and repeated the anchor's work.
2. *Per-sample store.* A history row holds a sample's position, its lab-frame source
   and its identity (predicted for PEC); for a varying operator (phase 4) also the
   state it was evaluated at and, for a z table, its rate if measurement favours
   storing it. A constant operator stores no state: $R_n=F$. The rows are a store of
   `order + 2` slots like the transported formulation's fused path, but nothing is
   re-anchored on acceptance.
3. *Moment action.* Per trial, $x=hB_n$ is the backend's own `_delta_phi` over the
   step, so that $e^x$ is exactly the propagator the step applies, and
   $\varphi_1\ldots\varphi_m(x)$ are formed once. Every combination the trial needs —
   predictor, corrector, the defects of the neighbouring orders, a dense observation —
   is applied in the monomial basis,
   $h\sum_k k!\,\varphi_{k+1}(x)\odot\bigl(\sum_j c_{jk}F_j\bigr)$, which never forms a
   $q\times M$ weight tensor. A separable operator forms its $x$, and so its moments,
   per element in the same pass. A NumPy `ExponentialKernels` mirrors `AdamsKernels`
   (no transport pass) and is what the compiled kernels are checked against.
4. *The formulation field and its record.* `Multistep.formulation`,
   `"interaction"` (default) or `"exponential"`, with a representation that omits the
   default so that `repr(Multistep())`, hashed into `StepperRecord.method`, is
   unchanged; the record name `exponential-adams-pec<q>/<q+1>[-variable]`. Phase 2
   bumps `SCHEMA_VERSION` to 20 for the field and the new protected hook, and the
   output schema to 13 because the method string then names a formulation; phase 3
   bumps the contract again only for compiled hooks or kernel arguments it adds, and
   solprop-kokkos follows on the same revisions; phase 4 for the varying-operator
   rows.
5. *Admission.* By formulation: the exponential one refuses a varying operator
   (expansion or table) by name until phase 4 and a coupled flow until phase 5; both
   refuse CF4; the compiled backends refuse the exponential formulation by name until
   phase 3.

**Performance baseline (Q2).** Fixed before any tuning:

- *Methods.* IP-DOPRI5 with its defaults; `Multistep()`; and the exponential
  formulation with exactly `Multistep()`'s controller — order up to five, variable,
  growth at most 2, safety 0.8, smallest factor 0.2, order hysteresis 1.2, a restart
  after three rejections, equal approach to landings, no interval cap, the norm scaled
  by the state before the step. Nothing is tuned before phase 3's campaign has
  measured these settings.
- *Coefficient budget.* The witnesses bound each moment within 256 ulp of its natural
  scale and each complete weight within 64 ulp of $\sum_k|c_{jk}|k!$ times that scale,
  so the induced state error is of order $10^{-13}h|F|$, five or more orders below any
  admitted `rtol`. No coefficient-uncertainty term enters the controller's norm for a
  constant operator (unlike the study's quadrature weights for varying operators).
- *Protocol.* `benchmarks/sep16/campaign.py`: per case and backend, a ladder of
  `rtol` in factors of about three, every point scored at every plane against an
  IP-DOPRI5 reference at `1e-11` refined against `3e-12`, the fewest-call point
  meeting each common target timed in warmed, randomly interleaved rounds (five), one
  thread in every pool; median and IQR with the raw samples and the load before and
  after each sample. Calls are logical: an Adams run's ledger total, and six per
  attempted IP-DOPRI5 step plus one.
- *Measurement.* Same-machine interleaved timings on Darwin/arm64, one thread in
  every pool, five rounds, median (IQR within 3 % of every median); 1-minute load
  average 2.9–5.1 throughout from macOS background services (no other user work
  ran). Fewest calls meeting both the worst-plane field error and the weak-band
  error at each target; GNLSE references refine to 6e-12/1e-10, CtC to 1e-13/5e-12.

  | case, backend | target | IP-DOPRI5 calls / s | `Multistep()` calls / s |
  |---|---|---:|---:|
  | GNLSE 1 mm, `numba_cpu` | 1e-3 | 133 / 0.099 | 78 / 0.082 |
  | | 1e-4 | 211 / 0.152 | 136 / 0.155 |
  | | 1e-5 | 259 / 0.186 | 174 / 0.190 |
  | GNLSE 1 mm, Kokkos CPU | 1e-3 | 133 / 0.104 | 78 / 0.111 |
  | | 1e-4 | 211 / 0.163 | 136 / 0.211 |
  | | 1e-5 | 259 / 0.203 | 174 / 0.257 |
  | CtC launch 0.17 mm, `numba_cpu` | 1e-3 | 37 / 2.56 | 27 / 1.97 |
  | | 1e-4 | 49 / 3.38 | 28 / 2.04 |
  | | 1e-5 | 55 / 3.80 | 41 / 2.97 |
  | CtC launch 0.17 mm, Kokkos CPU | 1e-3 | 37 / 2.84 | 27 / 2.17 |
  | | 1e-4 | 49 / 3.79 | 28 / 2.25 |
  | | 1e-5 | 55 / 4.22 | 41 / 3.29 |

  These are the numbers the exponential formulation is first compared with (phase 3);
  they are short windows, so startup weighs on them, and they are not extrapolated to
  complete runs. The CtC Kokkos row was retaken from a clean tree at the phase 1
  commit after the first attempt failed (a schema change landed in the working tree
  while that process started); the other rows ran before any phase 2 edit.

**Numerical witnesses.**

- *Moments and weights.* 168 checks pass: $\varphi_0\ldots\varphi_8$ at zero, $10^{-12}$
  imaginary, both sides of the join at $|x|=8$ in three directions, $|\operatorname{Im}x|$
  up to 1000, damping to $\operatorname{Re}x=-700$, gain to $\operatorname{Re}x=5$ and
  the resonances $x=2\pi i n$ (where $\varphi_1=0$), within 256 ulp of the natural scale
  of the augmented exponential and $10^{-13}$ of quadrature; complete weights on equal,
  unequal, ratio-two and clustered nodes ($10^{-2}$ spacing beside $2$), whole and
  fractional steps, exact for polynomials to within 64 ulp of their conditioning sum;
  ordinary Adams recovered at $x=0$.
- *Orders.* PEC $q/(q+1)$ converges at order $q+1$ for $q=1\ldots5$ in both formulations
  on a lossy Kerr bin with its self-rotation resolved (measured 1.95–6.14 over three
  levels).
- *A driven bin at large linear phase* (constant $D$, $|\operatorname{Im}D|h$ from 10
  to 0.6): the exponential formulation is accurate to $2\times10^{-5}$ at order one and
  to $10^{-11}$ at order five, while the interaction formulation's error is $O(1)$ —
  SEP 15 §4's mechanism, reproduced.
- *Varying operators over 1000–4000 steps* (relative variation $10^{-3}$, $10^{-2}$ and
  one half; lossless and lossy). Source-free: the chord is exact to round-off; the
  transported compensation converges below $10^{-6}$; the untransported one is $10^2$
  or more times worse. Driven, with the state following a smooth lab-frame drive: the
  untransported compensation is accurate to round-off ($<10^{-11}$), because there
  $(D-B_n)u_j$ is smooth in the lab frame, and the transported one stays within ten
  times the chord's error and beats it once $|\operatorname{Im}D|h\lesssim0.5$ —
  consistent with §4.2's claim, but see Q11.
- *A source rotating with its bin's free phase* (a noise-linear term): the interaction
  formulation is exact to round-off; the exponential one converges at fifth order once
  $|\operatorname{Im}D|h\lesssim1$ and is $10^3$ times worse or more before (§4.4).

**Deviations from the proposal.** None in Part I's scope. Two facts of §3 are
corrected above (the radial route on `numba_cpu`, and where the split-step refusal
binds).

**Decisions** (taken autonomously, for review):

- *D1, the moment evaluator.* Scaled Taylor series and doubling for $|x|\le8$, upward
  recurrence beyond, as SEP 15's study did, re-derived; a fixed 24-term series (a
  tail of $10^{-31}$) instead of a data-dependent stopping test, so the compiled
  kernels can mirror it branch for branch. Alternatives: contour integrals
  (Kassam–Trefethen), costlier per element; Padé with scaling and squaring of the
  augmented matrix, kept as the test oracle.
- *D2, the lab-frame hook.* A separate protected hook rather than a flag on
  `_adams_source`, so the transported formulation's hook, which SEP 12's adapter and
  Kokkos implement, is untouched. Alternative: an identity anchor, which the study
  showed repeats anchor work.
- *D3, monomial-basis application.* As interface 3: one moment set per trial serves
  every formula; per-node weight volumes are never formed. Alternative: per-node
  weights (the study), $q$ volumes per formula.
- *D4, no coefficient term in the norm* for constant operators, by the budget above.
- *D5, the developed GNLSE input* is the SEP 15 study's state at 0.12 m, reused with its
  fingerprint and checksum rather than regenerated: regenerating it is a long fibre
  propagation that Q9 does not authorise.
- *D6, logical call counts* for IP-DOPRI5 (six per attempt plus one), so a count does
  not depend on whether a backend recycles the last stage.

**Verification run.** `ruff check`, `ruff format`, `basedpyright` (0 errors) on every
changed file; `pytest -n auto` on `tests/backends/test_exponential_weights.py`,
`tests/stepper/test_exponential_scalar_march.py` and `tests/stepper/test_multistep.py`:
247 passed; the doctests of the two changed modules; `tests/test_seps.py`. The full
gate is not run in phase 1 (§11 and `AGENTS.md`: once, at the end).

**Sources checked.** The doubling identity and the augmented-exponential
representation of $\varphi_k$ against Hochbruck and Ostermann (2010) §2 and Cox and
Matthews (2002) §2, as cited in the module; verified numerically by the witnesses
rather than transcribed.

**Limitations.** The Kokkos column of the matrix is read from the code, not probed.
The baseline is on one machine (Darwin/arm64) under background system load.

**Open questions.**

- **Q11 — phase 4:** in a driven bin whose state follows a smooth lab-frame drive, the
  untransported compensation is accurate to round-off while the transported one, which
  Q8 chose, rotates with the reference flow and is only within ten times the chord's
  error at $|\operatorname{Im}D|h\gtrsim1$. Physical bins mix both behaviours, and the
  gate found the treatments equal on the full runs; phase 4 should measure whether a
  split of the compensation by bin character, or the transport of only the state's
  free part, is worth its cost. Q8 is unchanged.

**Handoff.** Phase 2 implements interfaces 1–5 on `numpy_ref`: the formulation field,
its representation and record, the schema bumps, the lab-frame hook, the exponential
march for constant and separable operators sharing `AdamsMarch`'s lifecycle (labels,
reservations, budgets, kinks, restarts, rejection, resume), dense observation, the
modified noise and schedules for both formulations, and the routes of the matrix.
`benchmarks/sep16/campaign.py` takes `--methods rk,interaction,exponential` once the
formulation exists.

### Phase 2 report — constant-operator exponential Adams on NumPy

**Scope.** Phase 2 of §11, 2026-10-06, Opus 5.5 (coding agent), autonomously, on
`feature/sep-0016-unified-multistep-2` (branched from phase 1's
`feature/sep-0016-unified-multistep-1`); solprop-kokkos on its branch of the same name,
from `main` at `dc4e7f4`. Neither is merged.

**Delivered.**

- `src/solprop/stepper/multistep.py`: `Formulation`, `Multistep.formulation` with a
  representation that omits the default, the record name
  `exponential-adams-pec<q>/<q+1>[-variable]`, the refusal of the formulation on the
  comparison variants, and `PEC_REAL_INTERVAL` with
  `Multistep.real_stability_interval`.
- `src/solprop/backends/driver.py`: the protected hook `_adams_lab_source`, refusing by
  default; a scheduled run with an Adams march replays through the march.
- `src/solprop/backends/exponential.py`: `EXPONENTIAL_REPRESENTATIONS` and
  `refuse_formulation` beside phase 1's moments and weights.
- `src/solprop/backends/multistep.py`: `ExponentialKernels` (moments once per trial,
  every formula applied in the monomial basis), `AdamsMarch._attempt_exponential`,
  `_observe_exponential` (dense observation), `_replay_to` and `replay_step`
  (prescribed schedules for both formulations), the `fixed` path of every attempt.
- `src/solprop/backends/numpy_ref/backend.py`: `_adams_lab_source` through the
  reference passes (an exact multiplication by one in place of the inverse
  propagator); the `modified` noise admitted for both formulations.
- `src/solprop/backends/batch.py`: lockstep replay for Adams members, each with its
  own history and ledger; members must share the integrator.
- `src/solprop/backends/numba_cpu/backend.py` and solprop-kokkos `backend.py`: the
  exponential formulation refused by name at construction.
- `src/solprop/translate/problem.py`: kernel schema 20; the collar stability warning
  uses the stepper's own real interval (DOPRI5's 3.3, or PEC's).
- `src/solprop/output/schema.py`: output schema 13.
- Tests: `tests/backends/test_exponential_march.py` (record, exact linear flow and
  its dense observation, order on schedules, agreement with RK, the call ledger,
  budget, rejection, dense against landing, refusals, `modified` noise with the same
  draw, schedule replay, lockstep, the collar warning),
  `tests/backends/test_exponential_routes.py` (every constant route),
  `tests/backends/test_tdse_march.py` and
  `tests/driver/test_sep12_sep14_integration.py` parametrized over both formulations,
  the PEC interval test in `tests/stepper/test_multistep.py`; updated literals for the
  two schemas.
- `docs/guide/numerics/adams.md` (the exponential formulation, where it suits,
  schedules, noise, limitations), `docs/dev/stepper.md`, `docs/dev/kernel-contract.md`
  (version 20), ADR-041, `CHANGELOG.md`, the layers figure.
- Evidence: `benchmarks/sep16/data/capability-audit-phase2.json`,
  `data/pilot/` (spatial pilot, short CtC), `data/tdse-compatibility.json`,
  `benchmarks/sep16/tdse_compatibility.py`.

**Verification.**

- The capability audit, re-run with the exponential formulation: on `numpy_ref` it
  runs every constant route and variant RK runs there — GNLSE, mode-averaged, envelope
  and multimode capillaries, the constant general guide, radial, Cartesian, line and
  planar free space; `input` and `modified` noise; dense observation; a prescribed
  schedule; the split-step collar — and refuses the pressure gradient (expansion) and
  every coupled flow by name. `numba_cpu` and Kokkos refuse it by name.
- *Exactness and order.* Source-free runs, landed and observed, equal the exact flow
  to round-off. On uniform prescribed steps after a halving start-up ramp, a
  second-order soliton converges in its populated bins at 3.12/3.07 (`q = 2`) and
  5.00/5.01 (`q = 4`), 30–200 times more accurately than the interaction formulation
  at the same steps (3.97/3.01 and 4.18/5.27 for it).
- *A finding in the weak bins.* Below $10^{-8}$ of the peak power the same runs
  converge at about order 0.6 in the exponential formulation (relative weak-bin
  errors $1.3\times10^{-2}$ to $3.6\times10^{-3}$ from 100 to 800 steps) and at three to
  four in the interaction one ($3.9$ to $3.2\times10^{-4}$). The worst bins sit where
  $|\operatorname{Im}D|h\approx\pi$: weak free radiation under the soliton's
  cross-phase modulation, a source linear in the bin's own field that rotates with its
  free phase — §4.4's mechanism, here without noise. The step-size norm does not see
  these bins; the weak-band observable of the campaigns does (Q12).
- *Agreement with RK.* An N = 3 soliton at `rtol=1e-8`, both formulations, within
  $10^{-5}$ of RK at `1e-11`; the same with the `modified` noise, both formulations,
  with the same draw (a different draw moves the field far more than the error); on
  every constant route a short run at `rtol=1e-9` agrees with RK at `1e-11` within
  $4.4\times10^{-7}$; on every route the lab-frame sample equals the transported
  source at equal anchor and position bit for bit.
- *Lifecycle.* One call per accepted step after one startup, rejected trials charged,
  the budget stopping before overspending, forced rejections leaving the history,
  dense observation within $10^{-6}$ of landing at `rtol=1e-9`; a schedule's steps
  taken as given, bitwise repeatable, one call each; a lockstep batch equal to its
  members run alone, bit for bit, for both formulations.
- *TDSE compatibility (§7.2, item 5).* With manufactured bindings, every test of the
  adapter's lifecycle passes for both formulations: whole-trial reservation before
  any chunk, evaluation labels, a failed chunk charged and the unused reservation
  released, budget stops at the accepted state, ledger restoration and the interior
  natural save resumed without resetting the atomic budget. With the real CPU atomic
  engine (solprop-tdse at the pinned revision, two OpenMP threads) on SEP 12's
  bounded fixture: both formulations complete the 2 µm run on three atomic calls (one
  startup, two predictors, orders one and one; three logical evaluations, none reserved
  or failed after the run; 73 s and 77 s), and on a two-call budget each stops at its
  first natural save (`budget_exhausted` at 1 µm) and resumes afresh to the end,
  charging its startup again (two startups, two predictors, four atomic calls in all,
  the ledgers carried across the resume). The fields of the two formulations agree to
  $3\times10^{-21}$ and their resumed runs to $1.3\times10^{-16}$: at this fixture's
  energy the response is negligible beside the linear flow, so this is a lifecycle
  check, not an accuracy comparison (Q5).
- *Short CtC case* (the hydrogen launch, 0.17 mm, `numpy_ref`, calls only):
  at common worst-plane field and weak-band errors of 1e-3, 1e-4
  and 1e-5, IP-DOPRI5 needs 37, 49 and 55 calls, the interaction formulation 27, 28
  and 41, the exponential one 10, 17 and 22 — two to three times fewer than either.
  This is the six-mode launch window with Raman, constant pressure (a constant
  operator); its reference refines to $1.1\times10^{-13}$.
- *Spatial pilot* (§11; `numpy_ref`, calls only, a 30 fs beam at 0.9 of the critical
  power in one bar of argon over 2 cm from its waist; radial 64 nodes, Cartesian
  32 × 32). At common worst-plane field and weak-band errors:

  | target | radial RK / interaction / exponential | Cartesian RK / interaction / exponential |
  |---|---|---|
  | 1e-3 | 115 / 19 / 16 | 115 / 22 / 16 |
  | 1e-4 | 115 / 31 / 18 | 115 / 30 / 18 |
  | 1e-5 | 115 / 44 / 22 | 115 / 40 / 22 |

  The exponential formulation takes 1.4–2 times fewer calls than the interaction
  formulation, most at tight targets: diffraction gives the high transverse
  wavenumbers the largest $|\operatorname{Im}D|$, the driven regime. The RK column is
  not an accuracy-limited count: from launch its steps are held by the growth cap of
  1.1 after the start-up rule, so it reaches its 1e-7 error at every tolerance with
  the same 115 calls. This short window therefore shows the formulations' difference
  clearly and RK's only as a startup bound; phase 3's campaign takes developed or
  longer windows before ranking RK. On `numpy_ref` the exponential formulation's
  moments make each call dearer (the radial window took 0.10 s against 0.08 s at
  `rtol=1e-6`); time is phase 3's question, on the compiled backends.
- Checks run: `ruff check`, `ruff format`, `basedpyright` (0 errors) on every changed
  file; `pytest -n auto` on the changed and directly exercised modules (listed in
  the commits) — all pass, with the Kokkos-selected cases after the contract-20
  wheel; solprop-kokkos's own suite with that wheel (229 passed, 4 skipped);
  `tests/test_structure_figures.py`, `tests/test_seps.py`; `poe docs` (strict, 2 min 12 s): build succeeded.
  Not run: the full gate, the slow lane and coverage (once, at the end of the SEP).

**Deviations from the proposal.** None in Part I's scope. One reading is recorded:
§6 asks that source evaluation not manufacture RK stages; on `numpy_ref` the lab-frame
hook calls the same reference passes with a unit array in place of the inverse
propagator — an exact multiplication, no identity propagator formed from $\Phi$ and
no repeated anchor work beyond the position's own scalars.

**Decisions** (taken autonomously, for review):

- *D7, schedules for both formulations, accepted whatever their estimate, with the
  order chosen by the adaptive rule.* Replaying an adaptive run's steps reproduces its
  positions but, where that run rejected, may choose a different order, since a
  replay never has to wait a step after a rejection; a replay of a replay is bitwise.
  Recording the order sequence for exact replay of an adaptive run would extend the
  saved schedule; not done. Alternative: fix the order on a schedule (`variable_order`
  off), rejected as a silent change of method.
- *D8, the `modified` noise admitted on `numpy_ref` for both formulations*, lifting
  SEP 13's backend refusal, whose stated reason (a state change between steps) does
  not hold: the draw enters each sample's field at its own position, never the state.
  The compiled backends keep refusing it until phase 3; dense observation beside it
  stays refused everywhere.
- *D9, the collar warning uses the method's own interval*, taking the highest order's
  (variable order may reach it). The comparison variants are warned at the PEC
  interval, conservative for PECE.
- *D10, Kokkos follows contract 20 now* (schema bump and a construction-time refusal)
  rather than in phase 3, so that `backend="auto"` and the Kokkos tests keep working
  on this branch.
- *D11, the moments come from the backend's own `_delta_phi`* over the step, so
  that $e^x$ is exactly the propagator the interaction formulation applies and a
  separable operator is assembled per element as for any stage.

**Sources checked.** The AB(q+1) negative-real stability intervals (1, 6/11, 0.3,
0.1636, 0.0877, 0.0465) were computed from the PEC recurrence's spectral radius and
agree with the standard values (Hairer and Wanner II §V.1 tabulate the Adams regions;
not reopened here). Nothing else was transcribed.

**Limitations.** `numpy_ref` only; the compiled backends refuse the formulation and,
with an Adams method, the `modified` noise. Varying operators and coupled flows are
refused. The weak-bin behaviour above is measured on one soliton problem only. The
real-engine TDSE check is a compatibility check, with no accuracy claim.

**Open questions.**

- **Q12 — phase 3 campaign:** weak bins whose source is linear in their own field and
  rotates with their free phase — free radiation under cross-phase modulation, the
  `modified` noise's linear term — converge slowly in the exponential formulation once
  $|\operatorname{Im}D|h\gtrsim1$, unseen by the step-size norm. The GNLSE and CtC
  campaigns must report the weak-band observable beside the field (as Q7 requires)
  and say whether this bounds the exponential formulation's advantage there. A
  per-bin hybrid frame is outside this SEP's scope.

**Handoff.** Phase 3 implements `_adams_lab_source` on `numba_cpu` and Kokkos for
every route, compiled `ExponentialKernels`, the transported formulation's compiled
free-space, planar and general modal routes, the `modified` noise there, and
`backend="auto"` per problem; then the constant-operator campaign with timings. The
NumPy kernels here are the oracle for the compiled ones.

### Phase 3 report — compiled delivery of the constant-operator slice

**Scope.** Phase 3 of §11, 2026-10-06, Opus 5.5 (coding agent), autonomously, on
`feature/sep-0016-unified-multistep-3` (branched from phase 2's) and solprop-kokkos
`feature/sep-0016-unified-multistep-3` (from its phase 2 branch). Neither is merged.
GPU qualification was not attempted: no device was used (paid or remote compute is
outside this work's authority).

**Delivered.**

- `src/solprop/backends/numba_cpu/backend.py`: `_adams_lab_source` for every route,
  through the route's own stage pipeline with a zero-length first stage whose
  propagator is exactly one; the transported source of the non-modal routes formed
  from it; the Adams admission reduced to the coupled flow (both formulations) and a
  varying operator (exponential formulation).
- `src/solprop/backends/numba_cpu/multistep.py`: `NumbaExponentialKernels` — one
  compiled pass forming every element's moments and flow per trial into a reused
  scratch, a compiled predictor and corrector; after the campaign below, each defect factored into a
  divided difference of the samples and one moment combination of its node
  polynomial, and the moment join moved from $|x|=8$ to 3, where the upward
  recurrence still holds within 32 ulp of the natural scale at order eight.
- solprop-kokkos: `_adams_lab_source` through the native stage-zero source with the
  anchor at the position; the native Adams source no longer refuses a separable
  geometry; route, `modified`-noise and formulation refusals lifted (a coupled flow
  and a varying operator for the exponential formulation stay refused). Kokkos runs
  both formulations' step arithmetic in host NumPy, as it already did for the
  interaction formulation.
- `src/solprop/driver/simulation.py`: `backend="auto"` with an integrator resolves as
  for the Runge-Kutta march.
- Tests: compiled-against-reference parity for both formulations on every constant
  route and on both compiled backends (`tests/backends/test_exponential_routes.py`),
  the `modified` noise on both compiled backends
  (`tests/backends/test_compiled_multistep.py`), the `auto` resolution.
- Docs: the manual's support statements, `docs/guide/performance/backends.md`, the stepper
  design, the interface docstrings.
- Evidence: `benchmarks/sep16/data/campaign/` (timed work–precision records and
  `work-precision.svg`).

**Campaign** (§7.2 cases measurable tonight; the observables frozen before the sweep as
in phase 1: worst-plane complex-field error and the weak-band error, bins between
$10^{-9}$ and $10^{-3}$ of the final reference's peak power, the larger of the two
meeting each target). Same-machine interleaved timings on Darwin/arm64, one thread in
every pool, five warmed rounds in random order, median with IQR (at most 6.7 % of any
median); the 1-minute load average stayed between 1.9 and 4.0 from macOS background
services. Calls / median seconds at 1e-3, 1e-4 and 1e-5, first as measured at
`00965ad`, then for `numba_cpu` after the kernel optimisation (`54e0866`):

| case, backend | IP-DOPRI5 | interaction | exponential |
|---|---|---|---|
| CtC launch 0.17 mm, `numba_cpu` | 37/2.51, 49/3.33, 55/3.73 | 27/1.93, 28/2.00, 41/2.93 | 10/0.78, 17/1.35, 22/1.74 |
| CtC launch, Kokkos CPU | 37/2.79, 49/3.68, 55/4.14 | 27/2.13, 28/2.20, 41/3.23 | 10/0.90, 17/1.55, 22/2.01 |
| GNLSE 0.2 mm, `numba_cpu` | 31/0.026, 49/0.039, 55/0.043 | 17/0.020, 30/0.035, 39/0.045 | 8/0.023, 10/0.028, 17/0.045 |
| GNLSE 1 mm, `numba_cpu` | 133/0.098, 211/0.152, 259/0.184 | 78/0.081, 136/0.148, 174/0.187 | 33/0.103, 50/0.152, 65/0.185 |
| GNLSE 1 mm, Kokkos CPU | 133/0.103, 211/0.162, 259/0.198 | 78/0.111, 136/0.211, 174/0.256 | 33/0.162, 50/0.239, 65/0.295 |
| radial 2 cm, `numba_cpu` | 115/0.069 (all) | 19/0.020, 31/0.031, 44/0.043 | 16/0.035, 18/0.041, 22/0.052 |
| radial 2 cm, Kokkos CPU | 115/0.427 (all) | 19/0.084, 31/0.137, 44/0.195 | 16/0.107, 18/0.124, 22/0.156 |
| Cartesian 2 cm, `numba_cpu` | 115/1.18 (all) | 22/0.46, 30/0.63, 40/0.84 | 16/0.55, 18/0.65, 22/0.81 |
| Cartesian 2 cm, Kokkos CPU | 115/1.44 (all) | 22/0.53, 30/0.73, 40/0.96 | 16/0.94, 18/1.10, 22/1.38 |
| *after the optimisation, `numba_cpu`:* | | | |
| CtC launch | 37/2.51, 49/3.33, 55/3.73 | 27/1.93, 28/1.99, 41/2.92 | 10/0.77, 17/1.32, 22/1.70 |
| GNLSE 0.2 mm | 31/0.026, 49/0.038, 55/0.042 | 17/0.020, 30/0.035, 39/0.044 | 8/0.019, 10/0.024, 17/0.040 |
| GNLSE 1 mm | 133/0.096, 211/0.149, 259/0.182 | 78/0.079, 136/0.145, 174/0.185 | 33/0.083, 50/0.128, 65/0.166 |
| radial 2 cm | 115/0.069 (all) | 19/0.020, 31/0.031, 44/0.043 | 16/0.031, 18/0.036, 22/0.046 |
| Cartesian 2 cm | 115/1.17 (all) | 22/0.46, 30/0.63, 40/0.85 | 16/0.50, 18/0.60, 22/0.74 |

With IQRs separated (the data's `selected` rows), after the optimisation the
exponential formulation is faster than **both** RK and the interaction formulation
on `numba_cpu` at every target from 3e-4 to 1e-5 on both GNLSE windows and at all five
targets on the CtC launch (2.2–3.3 times RK, 1.5–2.5 times the interaction
formulation there), and on the Cartesian pilot from 1e-4 to 1e-5; on the radial pilot
it takes fewer calls but more time than the interaction formulation (a cheap source
on a small state, where the step's arithmetic is the cost). On Kokkos, whose Adams
step arithmetic is host NumPy (D14), it wins the CtC launch at every target and loses
in time where the source is cheap. The RK column of the spatial pilots is not an
accuracy-limited count: from launch its steps are held by the 1.1 growth cap after the
start-up rule, reaching $10^{-7}$ at every tolerance with 115 calls, so those rows rank
the two Adams formulations and bound RK only. The windows are short (startup weighs
on them) and are not extrapolated to complete runs; no second CPU architecture was
measured. Plots: `benchmarks/sep16/data/campaign*/work-precision.svg`.

**The delivery decision (§7.3, Q10).** The criterion — reproducible call and
elapsed-time improvements over RK at two or more adjacent resolved targets in a
regime, exceeding the timing spread, and compared with the existing formulation — is
met clearly in the expensive-source regime: the CtC launch on both compiled backends
(every target), and on `numba_cpu` also the developed GNLSE windows (3e-4 to 1e-5),
each against both RK and the interaction formulation. **The exponential formulation
stays a public option.** Where the source is cheap and the state small (the radial
pilot) the interaction formulation is faster, and on Kokkos only the expensive-source
regime is won; the guidance must say so. This is the decision as the criterion reads
it on the cases measured tonight; the campaign is not complete (below), and the
person among the maintainers who owns SEP 16 may prefer to defer it until the CtC
comb, plasma and boundary regimes are measured.

**Not delivered in phase 3.**

- *Resident Kokkos history.* The Kokkos Adams march keeps its history and step
  arithmetic on the host for both formulations, with one native source evaluation
  and one native adoption a step, as SEP 14 left it. A device-resident history is a
  native-engine change not attempted overnight; it stays on the roadmap item it
  partly covers.
- *The full constant-operator campaign of §7.2.* Measured here: the developed GNLSE
  windows, the CtC launch, the radial and Cartesian smooth-focusing pilots. Not
  measured: a developed CtC comb window with `modified` noise (it needs the
  authorised CtC parent propagation, a long worked-example run not made tonight),
  focusing with plasma and active-boundary spatial regimes, the line and planar
  controls, constant general modal guides, and a second CPU architecture.
- *GPU qualification*: no device.
- *Memory ledgers*: the exponential march's scratch (``q + 2`` state-sized moment
  rows and a flow row on `numba_cpu`) is not yet listed in `memory_ledger()`.

**Verification.** `ruff check`, `ruff format`, `basedpyright` (0 errors) on every
changed file; `pytest -n auto` on the changed and directly exercised modules,
including both compiled backends' parity on every constant route for both
formulations, the noise parity, the existing compiled Adams, dense, resume, driver,
free-space and planar suites (all pass; 245 + 90 + 65 tests in the slices' last
runs); solprop-kokkos's own suite with the phase 3 wheel (229 passed, 4 skipped).
Parity bounds: fields within $10^{-9}$ of the peak (measured $10^{-18}$ to
$5\times10^{-14}$ on the GNLSE, free-space and general routes, up to $10^{-10}$ on the
gas-filled capillaries, where RK's own compiled difference is $1.3\times10^{-11}$),
step sizes within $10^{-3}$ (measured $1.8\times10^{-4}$ on Kokkos over 690 steps of
the general guide), every call equal. Not run: the full gate, the slow lane,
coverage, GPU.

**Decisions** (taken autonomously, for review):

- *D12, the compiled lab-frame source through each route's stage pipeline* with an
  identity slot, rather than a separate source service per route: one code path per
  route serves both formulations and RK, at the cost of the stage's unit
  multiplications (exact). Alternative: route-specific source kernels, more code and
  more parity surface.
- *D13, moments as a per-trial scratch on `numba_cpu`*: forming them inside both the
  predictor and the corrector pass cost twice the moment work, measured at about
  80 ms a step against an 8 ms source on the Cartesian pilot; the scratch is
  ``(q + 2)`` state-sized rows reused from trial to trial, not a cached coefficient
  volume.
- *D14, Kokkos's step arithmetic stays in host NumPy* for the exponential
  formulation, as for the interaction one: compiling it would make the external
  backend depend on solprop's Numba kernels, and the native resident history is the
  right home for it.
- *D15, `backend="auto"`* follows the Runge-Kutta rule for every integrator, with no
  per-problem exception: every backend now executes both formulations on every
  route the reference does.

**Open questions.**

- **Q13 — phase 6 or a follow-up:** the exponential formulation's per-element step
  arithmetic (moments, predictor, corrector with three defects) is several times a
  cheap source's cost on large states; where a source is cheap, its call advantage
  does not become time. A Kokkos-native or threaded implementation, and the Newton
  form of the corrector itself, are the remaining levers.

**Handoff.** Phase 4 starts from the constant-operator march: the reference's
rate, sample states and the transported compensation go into `AdamsMarch` beside the
moments, with the untransported treatment as a test-only witness; the compiled
backends take the same hooks. The full canonical and M2/RDW campaigns are the
authorised long runs (Q9).

### Phase 4 report — varying diagonal operators

**Scope.** Phase 4 of §11, 2026-10-06, Opus 5.5 (coding agent), autonomously, on
`feature/sep-0016-unified-multistep-4` (branched from phase 3's), after the §7.3
criterion was met in phase 3. No solprop-kokkos change was needed: Kokkos runs the
shared march, whose admission it imports. Not merged.

**Delivered.**

- `src/solprop/translate/linop.py`: `LinearOperator.rate`, the derivative of the
  representation's own table (an expansion's scalar weights or the z table), and
  `phi_rate(table, z, *, before=False)`, one-sided at an edge.
- `src/solprop/backends/multistep.py`: the compensated reference in the exponential
  march — per-slot state and rate rows beside the sources, the trial's chord
  `B_n = ΔΦ/h`, `_Compensation` (the transported treatment, Q8; the untransported one
  selectable only as `AdamsMarch(compensation="untransported")`, a test witness),
  added to the predictor, the corrector, every scored defect and dense observation
  (with the damped-bin guard), `source_absent` for the exact chord; the kink fix
  below.
- `src/solprop/backends/exponential.py`: the varying representations admitted.
- Compiled: `ExponentialKernels.carry` and `NumbaExponentialKernels.carry` (one
  traversal forming every sample's carried compensation), the corrector passes'
  compensation terms (`extra`).
- Tests: `tests/backends/test_exponential_varying.py` (the rate against the flow,
  the exact chord without a source, agreement with RK for both formulations, the
  untransported witness, order on a schedule, dense observation, a z table against the
  expansion it tabulates and on `numba_cpu`, an interior kink, the `modified` noise
  along a gradient, an adiabatic taper, a kink on a save plane, both compiled
  backends' parity).
- Docs: the manual's varying-operator section, ADR-042, the stepper design,
  `CHANGELOG.md`.
- Evidence: `benchmarks/sep16/varying_campaign.py` and `data/varying/`.

**Campaign** (the two authorised full-length propagations, Q9; §7.2 cases 6 and 7 on
the production code). From launch, `numba_cpu`, one thread in every pool, warmed
rounds in random order (five canonical, three RDW), median with IQR (at most 1.3 % of
any median), same-machine interleaved timings on Darwin/arm64 with the 1-minute load
between 1.4 and 3.7. Scores and planes are the review gate's, frozen there; references
at `rtol=1e-10` refine against `1e-9` to $2.5\times10^{-8}$ (RDW) and
$4.1\times10^{-6}$ (canonical), so canonical targets stop at 1e-4. The exponential
formulation runs its production default (variable order up to five); the gate's
numbers were fixed order four. Calls / median seconds at a common error on every
score:

| case, target | IP-DOPRI5 | interaction | exponential | untransported witness |
|---|---:|---:|---:|---:|
| full M2/RDW, 1e-3 | 11041 / 12.80 | 5683 / 8.49 | 974 / 2.76 | 973 / 3.62 |
| full M2/RDW, 3e-4 | 13987 / 16.23 | 7142 / 10.62 | 1473 / 4.05 | 1456 / 5.34 |
| full M2/RDW, 1e-4 | 17443 / 20.22 | 8389 / 12.38 | 1782 / 4.84 | 1801 / 6.49 |
| full canonical, 1e-3 | 1543 / 0.656 | 1090 / 0.570 | 814 / 0.639 | 790 / 0.740 |
| full canonical, 3e-4 | 1675 / 0.709 | 1145 / 0.597 | 1009 / 0.778 | 1004 / 0.926 |
| full canonical, 1e-4 | 1873 / 0.792 | 1415 / 0.732 | 1329 / 1.003 | 1179 / 1.073 |

On the full M2/RDW propagation the exponential formulation takes 4.7–5.8 times fewer
calls and 2.6–3.1 times less time than the interaction formulation, and 9.8–11 times
fewer calls and 4.2–4.6 times less time than IP-DOPRI5, at every target from 1e-2 to
1e-4 — the gate's result reproduced on the production code. The transported and
untransported compensations take the same calls to within 1 % with errors agreeing
to two digits, as in the gate; the witness's time is higher because it runs the
reference NumPy path. On the full canonical propagation, a single-mode case whose
source is cheap, the exponential formulation takes fewer calls than both (6–30 %
fewer than the interaction formulation) but is slower than the interaction
formulation at every target and than RK below 1e-3: its per-trial coefficient
construction is Python-level work comparable with the source itself here (Q14). The
RDW rows were timed at `bd5fd71`, before the scalar reuse of `94be7d3`, which removed
a fifth of the canonical step's time; the canonical rows are after it. The ablation
before the compensation was compiled (`data/varying-host-compensation/`, at
`993dc28`): RDW at 1e-4 5.11 s against 4.84 s after, canonical unchanged, the
remaining cost there being elsewhere.

**A defect found and fixed (existing behaviour).** When a hard kink coincided with a
save plane, the shared march classified the landing as the plane and did not restart
the history, so either formulation interpolated across the jump in `D`, rejected
three trials and failure-restarted (reproduced on `main`'s interaction formulation).
The march now restarts there; the run's own end is not counted as a restart.

**Findings.**

- *Collar stability binds the Adams step on short gradients.* With the in-stage
  temporal collar a 2 cm gradient's Adams steps were held near 30 µm (503 attempts to
  1 cm) against 27 attempts with the per-step collar, and near that limit
  rounding-level differences between backends moved the trajectories. The
  method-aware warning of phase 2 covers schedules; adaptive runs pay it in steps.
- *Order near a steepening fill.* On uniform steps the compensated march converges
  at its order in the interior (the midpoint plane: order 3 at `q = 2`), but over the
  last millimetres of a steady-flow fill, whose density steepens towards the exit and
  whose rate's higher derivatives grow forty-fold, any interpolated compensation loses
  order, while the chord stays exact for a source-free bin. In an adaptive run the
  estimate includes the compensation and sees it.

**Decisions** (taken autonomously, for review):

- *D16, the rate's side at a step's end.* `phi_rate(..., before=True)` for the
  endpoint sample, the interval the step came from; a startup sample after a kink
  reads the far side. Alternative: always the interval starting at an edge, which
  would put a kink's jump into the step ending there.
- *D17, states and rates stored per slot* (two more state-sized rows per history
  slot) rather than recomputing the rate per trial from the table: one evaluation per
  accepted sample. A z table's rate costs a Clenshaw pass over its coefficient
  volumes; the stored row avoids repeating it each trial.
- *D18, the carried rows formed once per trial* and every formula a weighted sum over
  them, rather than the study's fused per-formula kernel: the same arithmetic with
  one exponential per sample and element per trial, as Q8 accepted; the
  exponential-free form remains the cost target.
- *D19, the kink-on-a-plane fix* in the shared march, changing existing behaviour of
  the interaction formulation where a kink coincides with a save plane (no
  physics or default changes; three wasted trials and a failure restart removed).

**Verification.** `ruff check`, `ruff format`, `basedpyright` (0 errors) on every
changed file; `pytest -n auto` on the changed and directly exercised modules — the
phase 4 tests, the constant-operator and compiled parity suites, the Adams march,
kink, dense, step-plane, resume and TDSE suites (all pass; 105 and 158 tests in the
last two runs); `numpydoc lint` on the changed modules and the strict `poe docs`
build (clean).

*The full gate, run once for phases 1–4* (§11), on this branch with the solprop-kokkos
phase 3 wheel installed. It found stale literals of kernel contract 19 and output
schema 12 in tests, and the SEP 12 parent harness refusing an output-schema-13 parent
(schema 13 changes no field of the default Adams method's record, so a schema-12
parent now maps onto 13 directly), and branches without witnesses: a replay's
natural stop and run-out, its non-finite refusal (single and lockstep), the lockstep
integrator check, the driver's default lab-frame refusal, the shared coupled-flow
refusal and a constant operator's rate. These were fixed and witnessed, and an
unreachable representation refusal was removed. After that: `poe check` passes every
step but `test`, whose only failures are seven bitwise digests on `numba_cpu`
(`test_flow_collar.py`, five; `test_coherent_capillary.py`, two). They fail
identically at the phase 1 commit, which changes no existing march code, so they are
a property of this machine's build, not of this SEP, and are left alone. `poe docs`
(strict) passes; `poe cov` reaches 100 % of the CPU source, with the same seven
failures; `poe test-slow` passes (138 passed, 1 skipped); the Kokkos-selected slow
lane passes (139 passed); solprop-kokkos's own suite passes (229 passed, 4 skipped).

**Limitations.** Coupled flows (phase 5) are refused. The exponential-free transport
of §4.2 is not implemented; the carried rows cost one exponential per sample and
element per trial. Kokkos's compensation is host NumPy. Only single-mode tapers were
tested; multimode tapers with `taper_coupling=False` and the mode-set route's moving
guides are admitted but not separately qualified. No second CPU architecture.

**Open questions.**

- **Q14 — phase 6:** on cheap sources and small states (the canonical capillary, the
  radial pilot) the exponential formulation's per-trial work — moments, the
  coefficient tables and the compensation, partly Python-level — exceeds the calls
  it saves. Compiling the coefficient construction and, on Kokkos, moving the step
  arithmetic native are the levers; the guidance should say where it pays.

**Handoff.** Phase 5 (coupled sections, gated by review of phases 3 and 4) is not
started, as instructed.

### Review of phases 1–4

**Scope.** 2026-10-06, John Travers (human), on the phase reports above and the
implementing agent's list of the decisions they left open, each with a recommendation.
Every recommendation was accepted; this section records them, the corrections the
review made to that list, and the evidence gathered to close Q7.

**Accepted as recorded.** Decisions D1–D19 of the phase reports, including D8 (the
`modified` noise admitted with either formulation), D15 (`backend="auto"` with an
integrator resolves as for the Runge-Kutta march) and D19 (a hard kink on a save plane
restarts the history, a change to existing behaviour). The §7.3 delivery decision of
phase 3 is confirmed: the exponential formulation stays public and the default stays
`"interaction"`. The SEP 12 parent harness's admission of output schema 13, made when
the end-of-SEP gate ran, is accepted.

**Open questions.**

- **Q2 — resolved in review, 2026-10-06:** the baseline of phase 1, `Multistep()`'s own
  controller with variable order up to five, is the delivered configuration; nothing
  was tuned, and the full-length runs of phase 4 reproduced the gate's RDW result with
  it. Any tuning belongs to phase 6, with its own frozen observables.
- **Q4 — guidance now, ablation in phase 5:** the manual states where the in-stage
  collar's stability sets the Adams step (phase 4's short gradients); the bounded
  ablation of an implicit or inner collar treatment is phase 5's, as §11 has it.
- **Q6 — explicit exception, 2026-10-06:** phases 1–4 are qualified on CPU only
  (`numpy_ref`, `numba_cpu`, Kokkos CPU); no device was used. GPU qualification is a
  separately funded segment before the SEP goes `Final`.
- **Q7 — completed for phase 4, 2026-10-06:** below.
- **Q11 — resolved in review, 2026-10-06:** the transported compensation stays (Q8).
  The full runs show the two treatments within 1 % in calls with errors agreeing to two
  digits; a split by bin character is not worth its complexity on that evidence.
- **Q12 — to phase 6:** the weak-band score is reported beside the field in every
  campaign; the developed CtC comb window with the `modified` noise measures it; no
  per-bin hybrid frame. Until then the guidance does not recommend the exponential
  formulation beside the `modified` noise.
- **Q13, Q14 — to phase 3b and phase 6:** the Kokkos step arithmetic moves native
  with the resident history (below); compiling the coefficient construction on
  `numba_cpu` is phase 6's.

**Deviations owed, found in review.** §11 assigns three items that the phase reports
listed as not delivered; they are owed, not optional, and are done before phase 5:

1. *Phase 3: the resident Kokkos history* — the device-resident Adams step for both
   formulations, history, step arithmetic and norms in the engine, with only scalars
   crossing to the host (the roadmap's device-resident Kokkos Adams step, which §6 says
   phase 3 partly covers).
2. *Phase 3: memory ledgers* for both formulations' history and scratch on every
   backend.
3. *Phase 4: the exponential-free transport of §4.2*, measured against the
   untransported witness (Q8's cost target).

They are taken as phase 3b, before phase 5.

**Corrections to the reports.** Rechecked against the saved data
(`benchmarks/sep16/data/campaign-optimised/`): the phase 3 report's CtC launch saving
over IP-DOPRI5 on `numba_cpu` is 2.0–3.3 times, not 2.2–3.3 (2.05 at 3e-5); the
manual's GNLSE range ("7 to 25 % less time than both") was wrong and now reads 3–36 %
against IP-DOPRI5 and 8–31 % against the interaction formulation from 3e-4 to 1e-5,
with the interaction formulation as fast at 1e-3. D1's join at $|x|=8$ is superseded
by phase 3's join at 3, which the manual now states.

**The pre-existing failures of the gate.** The seven `numba_cpu` bitwise digests the
gate reported failing are an environment difference, not a change of arithmetic: their
baseline (`tests/data/fftw-witness-baseline.json`) was recorded with pyFFTW built
against Homebrew FFTW 3.3.11, and the environment the gate ran in had the PyPI wheel's
bundled FFTW 3.3.10. Rebuilt against the recorded FFTW, all seven pass on this SEP's
branch (117 tests in the three modules), so its changes leave every recorded
Runge-Kutta field bit for bit.

**Q7, the canonical case.** The full canonical campaign re-run at `f43bdea`
(`benchmarks/sep16/data/varying-q7/`; same protocol as phase 4, five rounds, IQR at
most 1.5 % of any median, 1-minute load 2.2–2.9), selected under three frozen score
sets: with the sharp-band weak score (phase 4's), with SEP 13's Planck-tapered
90–200 nm observable in its place (`solprop.processing.spectra.bandpass`, formed on the
spectrum: by Parseval the time-domain norm of SEP 13's `uv_field` differs by one
constant, which the ratio cancels), and on the field alone. Ladders were extended
coarser so that no tabulated selection sits at a ladder's end; references
refine to $1.7\times10^{-7}$ (field), $4.1\times10^{-6}$ (sharp band) and
$2.0\times10^{-6}$ (tapered). Calls / median seconds:

| score set, target | IP-DOPRI5 | interaction | exponential | untransported witness |
|---|---:|---:|---:|---:|
| sharp band, 1e-3 | 1543 / 0.644 | 1090 / 0.569 | 814 / 0.638 | 790 / 0.736 |
| sharp band, 1e-4 | 1873 / 0.779 | 1415 / 0.730 | 1329 / 0.999 | 1179 / 1.054 |
| tapered band, 1e-3 | 1543 / 0.644 | 890 / 0.463 | 726 / 0.569 | 718 / 0.664 |
| tapered band, 1e-4 | 1873 / 0.779 | 1145 / 0.591 | 1009 / 0.773 | 1004 / 0.909 |
| tapered band, 3e-5 | 1957 / 0.811 | 1343 / 0.692 | 1283 / 0.962 | 1269 / 1.126 |
| field only, 1e-4 | 1069 / 0.451 | 729 / 0.383 | 726 / 0.569 | 718 / 0.664 |
| field only, 3e-5 | 1273 / 0.532 | 890 / 0.463 | 814 / 0.638 | 790 / 0.736 |

The ranking does not depend on the UV definition: on this cheap single-mode source
the interaction formulation is fastest under every score set, the exponential one
takes the fewest calls (0.4–25 % fewer than the interaction formulation) but more
time (Q14). The tapered observable is the
looser of the two bands here, so every method needs fewer calls under it; RK's
selection is unchanged because its field error binds. Field-only selections at 1e-3
and 3e-4 sit at the RK and interaction ladders' coarsest point and are not tabulated.

**Q7, the M2/RDW case.** Re-run the same way (three rounds, IQR at most 1.6 %,
load 2.5–3.8), with the RK ladder extended to 1e-4, and selected with and without the
weak-bin score (the 100–150 nm band kept in both); references refine to
$1.3\times10^{-9}$, $2.5\times10^{-8}$ and $6.3\times10^{-9}$:

| score set, target | IP-DOPRI5 | interaction | exponential | untransported witness |
|---|---:|---:|---:|---:|
| with weak bins, 1e-3 | 11041 / 12.98 | 5683 / 8.61 | 974 / 2.62 | 973 / 3.55 |
| with weak bins, 3e-4 | 13987 / 16.40 | 7142 / 10.78 | 1473 / 3.85 | 1456 / 5.18 |
| with weak bins, 1e-4 | 17443 / 20.46 | 8389 / 12.61 | 1782 / 4.59 | 1801 / 6.31 |
| without, 1e-3 | 6715 / 7.89 | 5683 / 8.61 | 974 / 2.62 | 973 / 3.55 |
| without, 3e-4 | 6715 / 7.89 | 7142 / 10.78 | 1473 / 3.85 | 1456 / 5.18 |
| without, 1e-4 | 8815 / 10.36 | 8389 / 12.61 | 1782 / 4.59 | 1801 / 6.31 |

Dropping the weak-bin score changes only IP-DOPRI5's selections, as the gate found: its
weak-bin error is large beside its field error. Without it the exponential formulation
still takes 4.6–6.9 times fewer calls than IP-DOPRI5 at 1e-3 to 1e-4 (2.0–3.0 times less
time), and 4.7–5.8 times fewer calls and 2.7–3.3 times less time than the interaction
formulation. The interaction formulation's coarsest ladder point, 1e-5, already meets
1e-3, so its selections there are an upper bound. Phase 4's numbers are reproduced
exactly in calls and within 6 % in time.

### Phase 3b report — the resident Kokkos history, memory ledgers and the exponential-free transport

**Scope.** The three §11 items the review found owed (above), 2026-10-06, Opus 5.5
(coding agent) at John Travers's direction, on `feature/sep-0016-unified-multistep-3b`
(from `main` at `2f83634`, the merge of phases 1–4) and solprop-kokkos
`feature/sep-0016-unified-multistep-3b` (from its `main` at `873c10f`). Not merged.

**Delivered.**

- *The Adams engine* (ADR-043, kernel schema 21). `AdamsMarch` keeps the controller
  and hands every state-sized operation of its fused and exponential paths to an
  `AdamsEngine` (`src/solprop/backends/multistep.py`): the startup sample, a trial's
  predictor, source, corrector and scored norms, held until the decision, the
  adoption and transport of an accepted step, the accepted state held across a dense
  observation, and the observation. `HostHistory` carries the previous arithmetic
  unchanged (the whole suite passed through the refactor, bitwise records included);
  a backend returns its own engine from the protected hook `StepDriver._adams_engine`.
  Public helpers give a resident engine the host's coefficient tables:
  `fused_tables`, `exponential_tables` (with `factored_defect_rows`, which
  `numba_cpu` now shares), `compensation_tables` and `translate.linop.rate_weights`.
- *The resident history on Kokkos* (solprop-kokkos `src/general.cpp`,
  `src/bindings.cpp`, `backend.py`): for both formulations on every route but a
  coupled flow -- the modal fibre and capillary routes, the general modal route, the
  radial, Cartesian, line and planar routes -- the history's rows, the step's flow,
  the exponential formulation's moments (the reference's φ algorithm, join at 3), the
  predictor, the stage-zero source at the predicted field, the corrector with the
  defect norms (factored for the exponential formulation), adoption, transport and
  dense observation (the Adams and the Hermite interpolants) stay in the engine. A
  varying operator's transported compensation is resident too: each sample's state
  and rate (the rate formed from the operator's fixed terms), the carried rows, the
  endpoint term and the guarded dense interpolant. Only the trial's scalars, its
  coefficient tables (by value), the norm's per-bin scale and the norms cross.
- *Memory ledgers*: an `adams_history` group (`backends/memory.py`, `adams_memory`):
  the host engine's store, a varying operator's state and rate rows and `numba_cpu`'s
  moment scratch on `numpy_ref` and `numba_cpu`, Kokkos's resident engine natively;
  `estimate_memory` equals each backend's ledger to the byte.
- *The exponential-free transport of §4.2*, built opt-in on `numba_cpu` with a NumPy
  reference, measured below and removed at review (D26).
- Tests: `tests/backends/test_resident_adams.py` (which engine each route and
  formulation takes; the resident engine against the host engine over the same
  backend's hooks, so that only the engine differs: both dense interpolants, both
  formulations, a budget stop, a replayed schedule, the separable routes and the
  compensated reference landed and dense), the memory test of every backend, the
  transport's equivalence; the contract literals at schema 21.
- Docs: ADR-043, the stepper design's engine paragraph, the kernel contract (version
  21), the manual's run-time and limitations notes, the backends support table (whose
  Adams row still read "modal fibre route (SEP 14)"), the running and backends guides'
  memory groups, both changelogs.

**Measurements** (same-process A/B, interleaved warmed rounds, one thread in every
pool, Darwin/arm64; the machine shared with another project's builds, so each
sample's load is recorded with it).

| case (Kokkos CPU, the constant-operator windows) | resident / host engine, s | speed-up | load, IQR |
|---|---|---|---|
| GNLSE 1 mm, interaction, 1e-3 / 1e-4 / 1e-5 | 0.083/0.118, 0.150/0.224, 0.197/0.272 | 1.38–1.54 | 3.5–5.5, ≤ 3.6 % |
| CtC launch, interaction, the same | 2.178/2.260, 2.269/2.314, 3.322/3.413 | 1.02–1.04 | 3.5–5.5, ≤ 2.4 % |
| GNLSE 1 mm, exponential | 0.085/0.150, 0.131/0.243, 0.173/0.319 | 1.72–1.86 | 7–13, ≤ 12 % |
| CtC launch, exponential | 0.866/0.991, 1.476/1.627, 1.888/2.109 | 1.10–1.15 | 7–13, ≤ 15 % |

The two engines take the same calls, and their fields agree to 1e-14 of the peak. The
GNLSE gains are far outside the spread; the exponential CtC gain was measured during
another project's build and is within its spread (Q15). Kokkos has not been re-ranked
against IP-DOPRI5 with the resident engine.

The exponential-free transport against the exponential one, `numba_cpu`, the
authorised full runs (Q9) at the exponential formulation's selected points of Q7's
campaign (three rounds, load 3.4–5.3, IQR at most 1.2 %): the same calls and fields to
2e-12, but **slower** -- 4 % on the full canonical run (0.675 s against 0.649 s at
1e-3, 1.063 against 1.018 at 1e-4) and 13–15 % on the full M2/RDW run (5.38 s against
4.60 s at 1e-4). Reading each sample's accumulated flow and phase, two more
state-sized rows a sample, costs more than the exponential it saves on this CPU. The
exponential form stays the delivered transport (`benchmarks/sep16/data/transport/`; the
harness and the opt-in were removed at review and are at commit `ab4660f`).

**Decisions** (taken under the review's direction, for review):

- *D20, one coarse engine* (ADR-043): a trial is one call returning its norms, an
  acceptance one call; a hook per pass would leave the state crossing between passes.
- *D21, the Kokkos engine receives its native state and three bound operations*
  (the trial's arrays, the hoist, the adoption) rather than reaching into the
  backend, as the repository's strict typing requires.
- *D22, the resident predictor is formed in the anchor's frame*, without the host
  engine's multiplication and division by the propagator: the engines then agree to
  rounding, not bit for bit -- step sizes to 2e-8 relative, fields to 1e-14 of the
  peak (4.5e-12 densely beside the in-stage collar's many short steps). The tests
  bound the resident engine against the host engine over the same backend.
- *D23, the norm's scale*: the state's in-band magnitude is reduced in the engine and
  the controller's scale formed on the host, one scalar out and one per-bin row in a
  trial.
- *D24, the rate from the operator's fixed terms* (`rate_weights`, a handful of
  scalars per sample) rather than an uploaded state-sized row; Kokkos holds no z table.
- *D25, what keeps the host engine on Kokkos*: the general path (PECE and the
  comparison variants) and the untransported test witness.
- *D26, the exponential-free transport* was measured and removed (review, 2026-10-06:
  an unused option); the delivered transport is unchanged. The GPU campaign may
  re-measure it on a device, where the balance of exponentials and memory traffic
  differs.
- *D27, the history's memory* is counted from construction on every backend, from
  formulas the estimate shares; Kokkos allocates its norm scale and rate weights with
  the history so that its native count equals the estimate.

**Verification.** `ruff`, `basedpyright` and numpydoc on every changed solprop file;
in solprop-kokkos `ruff`, `pyright` (strict, 0 errors), `clang-format`, and the wheel
built with `-Werror`. Tests: every solprop module that touches Kokkos and the Adams
march (350 in the last run), the resident and memory suites, the varying-operator
suite, the driver contract, and solprop-kokkos's own suite (230 passed, 4 skipped).
Not run: solprop's full gate, slow lane and coverage, and solprop-kokkos's
`scripts/check.sh` (sanitisers, distribution wheel), all due at the end of the SEP;
any GPU.

**Limitations.** No device was used: Kokkos's CPU build shares host memory, so a host
access to device memory in the new passes would not show here (Q16). The general path
and the untransported witness keep host arrays on Kokkos. The host still forms each
trial's stage scalars and coefficient tables (a few hundred bytes) and copies the
state when a budget stop reports it.

**Open questions.**

- **Q15 — resolved, 2026-10-06:** Kokkos re-ranked with the resident engine in one
  session (`benchmarks/sep16/data/campaign-resident/`, five interleaved rounds, load
  2.2–4.7, IQR at most 8.6 %), calls / median seconds at 1e-3, 1e-4 and 1e-5:

  | case, Kokkos CPU | IP-DOPRI5 | interaction | exponential |
  |---|---|---|---|
  | GNLSE 1 mm | 133/0.102, 211/0.159, 259/0.204 | 78/0.077, 136/0.140, 174/0.181 | 33/0.078, 50/0.128, 65/0.163 |
  | CtC launch | 37/2.929, 49/3.881, 55/4.353 | 27/2.209, 28/2.260, 41/3.309 | 10/0.872, 17/1.481, 22/1.922 |

  The exponential formulation is now the fastest on Kokkos at every target of both
  (1.2–1.3 times IP-DOPRI5 on the GNLSE window, where phase 3's host engine made it the
  slowest; 2.1–3.4 times on the CtC launch, 1.5–2.5 times the interaction formulation).
  The CtC launch's gain over the host engine is a few per cent: its source dominates.
- **Q16 — GPU qualification (Q6's segment):** the resident engine has run on Kokkos's
  CPU build only; its device qualification must exercise every resident pass,
  including the compensated reference and dense observation.

**Handoff.** Phase 5 (coupled sections) starts from the engine: a coupled flow's
trial needs its own resident pass (its flow is not a diagonal exponential), behind the
same `AdamsEngine` calls.

### Phase 5 report — coupled sections and the absorber ablation

**Scope.** Phase 5 of §11, 2026-10-06, Opus 5.5 (coding agent) at John Travers's
direction, on `feature/sep-0016-unified-multistep-5` (from phase 3b's branch) and
solprop-kokkos `feature/sep-0016-unified-multistep-5` (from its phase 3b branch). Not
merged.

**Delivered.**

- *Coupled sections in the exponential formulation* (Q3, ADR-044). On a coupled
  section the step's reference is `B = (ΔΦ/h) I + G_s`, the frozen block of the
  section the step is anchored in beside the cumulative diagonal, which is a multiple
  of the identity in the channel axis. `backends/multistep.py`: `CoupledBlock`,
  `coupled_moments` (the flow and moments `k! φ_{k+1}(hB)` on the block's bins, from
  the exponential of an augmented matrix carried back by the block's Schur vectors,
  elementwise elsewhere), `CoupledExponentialKernels` (the reference passes with those
  products) and its `carry_coupled`. A frozen-residual section's residual `R_s(z) u`
  joins the transported compensation of ADR-042: each sample keeps its flow weights,
  and a trial forms `(D(z_j) − ΔΦ/h) u_j + R_s(z_j) u_j` against its own section's
  block -- so a step entering a new section rebuilds the history's residual -- carried
  by forward matrix flows only; the endpoint term gains `R_s(z + h)` at the predicted
  field. The residual acts on the in-band active bins, as the Runge-Kutta stage
  applies it.
- *The backends.* The protected hook `StepDriver._adams_block` (kernel schema 22)
  returns a step's block: `numpy_ref` and `numba_cpu` from the reference's own
  decomposition, Kokkos from its host-side one. The lab-frame sample is the material
  source alone: `numba_cpu` runs its stage without the flow's pipeline, and Kokkos's
  native stage-zero source sets the flow aside for the evaluation. The matrix passes
  are the host engine's on every backend.
- *Admission.* `backends.exponential.COUPLED_SECTIONS`; the interaction formulation and
  CF4 sections are refused by name on every backend; the untransported compensation
  witness is refused beside a coupled flow.
- *Q4's ablation* (`benchmarks/sep16/collar_ablation.py`, below), the coupled calls
  (`benchmarks/sep16/coupled_calls.py`).
- Tests: `tests/backends/test_exponential_coupled.py` (the matrix moments against
  quadrature of their defining integrals on a nonnormal block; the constant-Schur
  walls and the frozen-residual gradient and taper against IP-DOPRI5; exactness
  without a source; both compiled backends against the reference; the refusals).
- Docs: the manual's coupled-sections paragraph, collar guidance and limitations, the
  stepper design, ADR-044, the kernel contract (version 22), both changelogs.

**Verification.** On a nonnormal three-channel block the matrix moments equal
quadrature of `∫₀¹ e^{(1−s)X} s^k ds` to 1e-11 and the flow `expm(X)` to 1e-13. Against
IP-DOPRI5 at `rtol=1e-11` on the qualification cases (0.2 mm, a pulse whose source
changes the field by 8e-3): the constant-Schur scalar and TE/TM walls to 2.6e-12 and
5.7e-13 at `rtol=1e-9`; the frozen-residual gradient to 1.2e-11, and the taper, which
crosses section edges, to 1.7e-6, 1.4e-8 and 1.4e-10 at `rtol` 1e-5, 1e-7 and 1e-9 --
the error follows the tolerance. Without a source the coupled flow is exact to 1e-12 of
the peak. `numba_cpu` and Kokkos CPU follow `numpy_ref` on the scalar, TE/TM, gradient
and taper cases with the same calls and fields within 1e-11. `ruff`, `basedpyright`,
numpydoc; solprop-kokkos `ruff`, strict `pyright`, `clang-format` and a `-Werror` build;
the Adams, coupled-flow and Kokkos suites (328 and 231 passed in the last runs). Not
run: the full gates (at the end of the SEP), any GPU.

**Calls beside a coupled flow** (`numpy_ref`, calls only: 0.2 mm windows on the
reference backend). The scalar wall's IP-DOPRI5 steps do not move with its tolerance
(49 calls from 1e-4 to 1e-9, the same field as its reference), while the exponential
formulation reaches 1e-8 to 2.6e-12 in 5 to 15 calls. On the taper, at matched error the
exponential formulation takes two to three times fewer calls: 51 against 109 near 1e-9,
67 against 151 near 1e-10.

**Q4, the absorber ablation.** On the two authorised full-length propagations (Q9), at
Q7's selected points for 1e-4, every accepted step's ratio to the in-stage collar's
stability limit (`h max α / 2` over PEC's negative-real interval at the step's order),
and the calls of the same run with the per-step collar and with the in-stage collar at
half and a quarter of its strength, whose limit moves out by two and four:

| full run, formulation | median ratio | steps above 0.9 | calls: in-stage | per-step | half | quarter |
|---|---:|---:|---:|---:|---:|---:|
| canonical, interaction | 0.82 | 25 % | 1415 | 1312 | 1312 | 1312 |
| canonical, exponential | 1.00 | 87 % | 1329 | 965 | 965 | 965 |
| RDW, interaction | 0.11 | 0 % | 8389 | 8433 | 8297 | 8266 |
| RDW, exponential | 0.83 | 28 % | 1782 | 2060 | 1790 | 1765 |

The limit binds the exponential formulation on the canonical run, whose pulse stays
clear of the collar: removing it saves 27 % of the calls, and the per-step collar already
removes it there. On RDW, whose field reaches the collar, weakening the collar changes
nothing although the steps sit near the limit: accuracy sets them, and the per-step
collar's splitting error costs calls. An implicit or inner collar treatment would buy
nothing beyond the per-step collar where that is valid, so by §11's condition none is
built. The manual now says to take the per-step collar with the exponential formulation
where the pulse stays clear of the collar.

**Decisions** (taken under the review's direction, for review):

- *D28, the reference's matrix functions from an augmented exponential on the Schur
  factors* rather than the eigenbasis the compiled flow actions use: correct for a
  nonnormal or nearly degenerate block without a conditioning gate. The compiled
  backends use the same host passes for now.
- *D29, the residual through each sample's flow weights*, a few scalars, rather than a
  stored matrix per sample and bin; it is formed against the trial's own section, which
  is what rebuilds the history at a section change.
- *D30, the lab-frame sample without the flow's pipeline* on every backend: the residual
  is in the compensation and the flow in the moments, so leaving either in the sample
  would count it twice.
- *D31, the matrix passes on the host on Kokkos*, its resident engine declining a
  coupled flow; resident coupled passes are for the GPU campaign to weigh.
- *D32, the untransported witness refused beside a coupled flow*: it is a witness for
  the diagonal compensation only.
- *D33 (Q4), no implicit or inner collar treatment*, by the ablation above; guidance
  instead.

**Limitations.** The coupled matrix passes run in NumPy on every backend and have not
been optimised; the coupled exponential formulation's run time has not been measured
against IP-DOPRI5 on a compiled backend. The coupled windows measured are short.
Dense observation stays refused beside a coupled flow, as for every stepper. CF4 is
excluded. No GPU.

**Open questions.**

- **Q3 — qualified, 2026-10-06:** constant-Schur and frozen-residual sections are
  delivered for the exponential formulation on every CPU backend; CF4 and every noise
  model beside a coupled flow stay refused, as decided.
- **Q4 — resolved, 2026-10-06:** by the ablation above; no implicit treatment.
- **Q17 — GPU campaign and phase 6:** the coupled exponential formulation's compiled or
  resident matrix passes, and its run time against IP-DOPRI5 on a developed coupled
  window.

**Handoff.** Phase 6 (the gallery, guidance, final integration and the final gates) and
the GPU optimisation campaign the review set after phase 5, on one AMD and one NVIDIA
device.

### Review of phase 5

**2026-10-06, John Travers**, with explanations of each decision from Opus 5.5:

- **D28–D33 accepted.** The reference's augmented-exponential matrix functions stay the
  oracle that any faster path is checked against (D28); the coupled matrix passes stay
  on the host on Kokkos until the GPU campaign has measured them (D31).
- **The collar default is unchanged.** The per-step collar saves calls with the
  exponential formulation only where the pulse stays clear of the collar (27 % on the
  full canonical run), and where field reaches the collar it leaves an error floor of
  2e-4 to 7e-4 in the physical window that no tolerance removes. Whether a run stays
  clear is not known before it runs, so the in-stage collar stays the default and the
  per-step collar a documented opt-in.
- **Q17 split.** On CPU, in phase 6: compile the coupled passes on `numba_cpu` (the
  flow actions' eigenbasis where it is well conditioned and the Schur route otherwise,
  checked against D28's reference), and time IP-DOPRI5 against the exponential
  formulation on a developed coupled window, not a full-length run. On device, in the
  GPU campaign: whether resident coupled passes on Kokkos pay for themselves. Until
  then the manual says that the coupled path is qualified for accuracy, not yet for
  speed.

### GPU optimisation campaign — plan

Set by the review of phase 3b (after phase 5) and written for the go that any device
use needs. It is an optimisation campaign, not only Q6's qualification: it closes Q16,
the device half of Q17, and Q6's exception before `Final`.

**Devices.** One AMD CDNA 3 accelerator (Kokkos `AMD_GFX942`, the `hip` preset) and
one NVIDIA Hopper accelerator (`HOPPER90`, the `cuda` preset), rented by the hour within
a fixed budget the maintainers set; new work stops at nine tenths of it. Each session
creates its device, pulls its evidence off often, and destroys the device at its end;
the device-hours and cost of every session are recorded in the report.

**G0 — preparation, CPU only, no cost.** The campaign harness: a qualification
selection of the Kokkos-touching suites, per-pass timers through Kokkos Tools regions,
and the work–precision harnesses of Q7 and Q15 extended to a device. The phase 6 CPU
half of Q17 is done first, so that the device sees the coupled passes in their final
host form.

**G1 — qualification, on both devices.** A `-Werror` build; solprop-kokkos's suite and
solprop's resident, exponential, varying-operator, coupled and memory suites on the
device; every resident pass exercised, the compensated reference and dense observation
included (Q16); parity with Kokkos CPU (the same calls, fields within the CPU tests'
bounds); the memory ledger against the device's own allocation; the vendor's race and
memory checkers on a few targeted tests, not the whole suite.

**G2 — profile.** A time breakdown by pass, and the host–device synchronisations per
trial, on the GNLSE window, the CtC launch, the full canonical and full RDW runs (the
authorised full-length runs of Q9) and a coupled taper on the host engine: what share
of a step is the source, the Adams passes, the controller's scalars and the transfers.

**G3 — optimise.** Candidates, taken in the order G2's profile ranks them: one transfer
of a trial's scalars and norms per trial; fusing passes (the predictor with the
stage-zero source's prologue, the corrector with the defect norms and the accepted
transport); hierarchical reductions for the defect norms; forming the moments in
registers instead of reading stored arrays, where the passes are bound by memory
bandwidth; launch bounds and layouts per architecture; resident coupled passes, if the
host's coupled passes dominate a coupled run; and §4.2's exponential-free transport
re-measured only if the exponentials are a measurable share. Each candidate is
developed and verified on Kokkos CPU between sessions, and kept only if it holds parity
on both devices, wins on a device by more than the rounds' spread and costs Kokkos CPU
nothing beyond it.

**G4 — work–precision on device.** IP-DOPRI5 against both formulations on the GNLSE
window, the CtC launch and the full canonical and RDW runs, with Q7's score sets, at
least five interleaved warmed rounds per point, median and IQR; recorded in the
benchmark ledger with its machine records.

**Exit.** The campaign's report in Part II: Q16 and Q6's exception closed or stated
open, Q17's device half answered, the kept optimisations with their measurements and
the rejected ones with theirs, and the spend.

### Phase 6 report, part 1 — Q17 on CPU and the collar in the order choice

**Scope.** The CPU half of Q17 as the review of phase 5 set it, 2026-10-06, Opus 5.5
(coding agent) at John Travers's direction, on `feature/sep-0016-unified-multistep-6`
and solprop-kokkos `feature/sep-0016-unified-multistep-6`. Not merged.

**Delivered.**

- *The coupled matrix functions from the flow actions' eigenbasis* (D35).
  `CoupledBlock.basis` carries a section's `(W, Y, λ, route)`, which `numba_cpu` and
  Kokkos already hold for their flow actions; on the route's bins every matrix
  function is `W f(x + h λ) Y`, the off-route bins keep the augmented exponential, and
  `numpy_ref` passes no basis, so the reference is unchanged (D28). On the bins of a
  real wall they agree with the augmented exponential to 1e-15 relative; a test checks
  a nonnormal block and an off-route bin.
- *Compiled coupled passes on `numba_cpu`.* `NumbaCoupledExponentialKernels` forms the
  route's flow and moments in one pass, applies them in the predictor and the corrector
  (the defects factored as on the diagonal path) and carries every sample's
  compensation through the block's flow in one pass; `ExponentialKernels.coupled()`
  is how the host engine reaches a backend's coupled passes. A frozen-residual sample's
  residual action is formed once per section and kept (`BlockResidual`, D36).
- *The exponential formulation's order choice weighs the in-stage collar* (D34),
  below.
- Benchmarks: `coupled_window.py` (two developed coupled windows, IP-DOPRI5 against the
  exponential formulation, ladders, selection and timing) and `order_choice.py`.

**Q17 on CPU.** Two 2 cm argon capillaries at 1 bar with a coherent wall, three modes,
30 fs and 300 µJ at 800 nm, Kerr only (the cheapest source a coupled run has): `wall`,
constant-Schur; `taper`, 60 to 45 µm, frozen-residual. Fewest calls meeting each target
(worst plane against IP-DOPRI5 at `rtol=1e-10`), five interleaved warmed rounds, one
thread, Darwin/arm64 under load 2–5 (same-process comparisons; interquartile ranges at
most 3 %); calls / median seconds at 1e-4, 1e-5 and 1e-6:

| window | IP-DOPRI5, `numba_cpu` | exponential, `numba_cpu` | IP-DOPRI5, Kokkos | exponential, Kokkos |
|---|---|---|---|---|
| wall | 1285/0.83, 1873/1.21, 2827/1.79 | 952/1.44, 1344/1.99, 1556/2.29 | 0.86, 1.24, 1.85 | 3.67, 4.89, 5.61 |
| taper | 1909/1.81, 2863/2.69, 4405/4.13 | 746/1.93, 1263/3.32, 1555/4.05 | 1.88, 2.80, 4.29 | 4.66, 8.09, 9.84 |

Before the compiled passes a coupled step cost about 50 ms; it now costs 1.5 ms (wall)
and 2.6 ms (taper), against 0.96 ms for the diagonal formulation on the same window
without the wall and 0.63 to 0.94 ms per IP-DOPRI5 call. The exponential formulation
takes 1.3 to 2.8 times fewer calls; on `numba_cpu` the taper is level with IP-DOPRI5 at
1e-6 and up to 1.23 times slower at looser targets, the wall 1.3 to 1.7 times slower.
On Kokkos the coupled passes are the host's (D31): two to four times slower. What
remains per step on `numba_cpu` is the per-trial formation of the block's matrices
(0.4 ms) and the compensation's NumPy sums (0.3 ms, shared with the diagonal varying
path): candidates for the GPU campaign, which redesigns these passes for the device.
The source here is the cheapest a coupled run has; ionisation, Raman or more modes
make each saved call worth more. The translation's inverse-conditioning refusal, which
forced `dz_max = 5 mm` on these windows for IP-DOPRI5, does not bind the exponential
formulation at these accuracies: lifted, its calls and errors were unchanged (steps of
25–60 µm).

**The order choice and the in-stage collar** (D34). On the wall window the exponential
formulation's runs a rounding apart -- a 1e-14 perturbation of the launch, or another
FFT plan -- settled in different order regimes: 930 to 936 calls in one session, 1313
in another, fields up to 2.7e-5 apart at `rtol=1e-6`. With the per-step collar every run
took 598 calls: the in-stage collar's stability limit, which shrinks with the order
(PEC's negative-real interval: 0.3 at order three, 0.088 at five), was being met by the
higher orders, whose defects then carry the collar's growing mode, and the choice on
those defects was decided by rounding. Each order is now weighed by the step its defect
allows, capped at `COLLAR_STABILITY` (one) times its interval over `h max(α)/2`; the
step itself is unchanged. `benchmarks/sep16/data/order-choice/`, `numba_cpu`:

| case, exponential formulation | unweighted | weighted |
|---|---|---|
| wall, seven runs (four perturbed by 1e-14) | 930–936 calls, fields 4.4e-7 apart | 688 calls each, fields 5e-14 apart |
| general modal guide, `numpy_ref` / `numba_cpu` | 734 / 734 | 749 / 749 |
| full canonical at Q7's 1e-4 point | 1329, scores ≤ 6.3e-5 | 1219, scores ≤ 3.0e-5 |
| full RDW at Q7's 1e-4 point | 1782, scores ≤ 4.5e-5 | 1704, scores ≤ 6.6e-5 |

Measured and not taken: capping the step as well at each order's limit (the wall took
737 calls, the general guide's interaction run left `numpy_ref` and `numba_cpu` 24
calls apart); a share of 0.5 to 0.7 (no cure the full interval lacks, and up to 19 %
more calls); and the weighting in the interaction formulation, whose history is carried
through the propagators, so that the collar alone does not describe its stiffness:
there it cost the canonical run up to 18 % at the smaller shares and broke the general
guide's agreement between backends. Two parity bounds are recalibrated from
measurement: the gas-filled capillaries' compiled-against-reference fields (1.1e-9 of
the peak, the bound now 3e-9; one step more on the multimode capillary) and the
resident engine's dense compensated reference (1.5e-10, the bound now 1e-9; the collar
holds that case at its limit, where the step controller passes rounding on).

**Decisions** (accepted in review, 2026-10-06, John Travers):

- *D34, the exponential formulation's order choice weighs the in-stage collar*, the
  choice only and at the full interval; the interaction formulation's is unchanged.
- *D35, the compiled backends take the coupled matrix functions from the flow actions'
  eigenbasis*, with its route and gate; the reference keeps D28's augmented
  exponential.
- *D36, residual actions kept per (section, weights)*, the least recently used of 64
  dropped.

**Verification.** The coupled suite (17, both compiled backends against the reference:
the same calls, fields within 1e-11), the exponential, resident and coupled-flow suites
(502 passed), the Adams suites (716 passed); `ruff`, `basedpyright`, numpydoc. Kokkos's
side is the basis it passes (its coupled passes stay the host's).

**Open.** Q17's device half stays with the GPU campaign: resident coupled passes, the
block's matrices formed in the device's passes rather than per trial, and the
compensation's sums fused into the corrector.

### Phase 6 report, part 2 — resident coupled passes, the GPU campaign's preparation and free space

**Scope.** 2026-10-06, Opus 5.5 (coding agent) at John Travers's direction, on
`feature/sep-0016-unified-multistep-6b` and solprop-kokkos
`feature/sep-0016-unified-multistep-6b`: the preparation on CPU the review of part 1
asked for before any device session -- Kokkos's resident coupled passes (the device half
of Q17, built and qualified on Kokkos CPU), the G2 profile harness -- and, at John
Travers's request, a free-space qualification on CPU. Not merged.

**Resident coupled passes on Kokkos** (D37). Beside a coupled flow whose every bin of
every section is on its eigenbasis route, with at most `ADAMS_MAX_CHANNELS` (16)
channels and one batch row, the exponential formulation's trial runs in the resident
engine: per bin, the block's flow and moments `W f(x + h λ) Y` from the section's
resident eigenbasis; the coupled predictor and corrector; the compensation's carry
through the block's flow; and a coupled defect reducer. The source is evaluated with
the flow set aside (`FlowAside`), the Adams scalars carry the step's section and a
frozen-residual flow's weights, and the elementwise moments skip the block's bins. A
flow with a bin off the route keeps solprop's host engine (D31's path), as does a block
of more than 16 channels. This reverses D31 under the review's direction: on a device
the host path would copy the state off the device every trial.

**A frozen-residual flow's residual, natively** (D38). A sample's residual against a
trial's section is `G(z_j) u_j - G_s u_j`, and only the second part depends on the
trial: the engine applies each sample's generator once, when the sample is made (at the
start and on acceptance, from the trial's end), keeps `G(z_j) u_j` beside the history,
and each trial applies only its section's frozen block. Forming the generator per
sample per trial instead, from all of the flow's Chebyshev terms, cost the taper 3.8 ms
a trial; this form, 2.6 ms.

**Verification.** All four coupled cases on Kokkos run resident (the parity test now
asserts it) and follow `numpy_ref` with the same calls and fields within 1e-11 of the
peak, the taper crossing sections with its compensation; `adams_memory` counts the new
rows to the byte on every backend (a coupled case added to the estimate test); a
`-Werror` build, solprop-kokkos's suite (231 passed, 4 skipped) and solprop's
exponential, resident, memory, Adams and coupled-flow suites (629 passed).

**Timing, Kokkos CPU** (`benchmarks/sep16/data/coupled/window-kokkos.json`, at solprop
`f65aa74` and solprop-kokkos `0182a00`, before the moments skip; one thread, load 4.7–6.1,
same-process comparisons), calls / median seconds at 1e-4, 1e-5, 1e-6:

| window, Kokkos CPU | IP-DOPRI5 | exponential, resident | (host engine, part 1) |
|---|---|---|---|
| wall | 1285/0.88, 1873/1.27, 2827/1.91 | 952/1.43, 1345/1.99, 1556/2.29 | 3.67, 4.89, 5.61 |
| taper | 1909/1.95, 2863/2.94, 4405/4.48 | 746/1.89, 1263/3.28, 1555/4.12 | 4.66, 8.09, 9.84 |

The taper is level with IP-DOPRI5 at 1e-4 and faster at 1e-6; the wall, the cheapest
source, stays 1.2–1.6 times slower, as on `numba_cpu`.

**The G2 profile harness** (`benchmarks/sep16/device_profile.py`). For one case,
method and tolerance on Kokkos: the march's wall time split into the Kokkos backend's
own frames (the native calls, each ending at a fence, and their glue) and the host's,
and, with Kokkos Tools' simple kernel timer, every `solprop_kokkos.*` label's time
grouped into the Adams passes, the coupled flow's, the FFTs and the rest of the source.
On Kokkos CPU it already shows the shape of the device question. On the wall, 1.89 s
of the march is native and 0.57 s host: a quarter of a coupled step (about 0.37 ms) is
the controller's and the tables' Python, which a device's faster kernels will not
shrink. On the taper the Adams passes take 4.9 s of the two fenced marches against
1.0 s for the source and 0.5 s for the FFTs; the block's functions (0.37 ms a trial)
and the carry (0.34 ms) lead.

**Free space on CPU** (§7.2 item 3; `benchmarks/sep16/free_space.py`,
`free_campaign.py`, `data/free/`). The phase 2 pilot was a 2 cm window from the waist,
where IP-DOPRI5's steps were held by its start-up growth. These cases run long enough
for the Adams methods to reach their accuracy-limited steps, in argon at one bar with a
30 fs pulse of 100 µm waist: Kerr self-focusing at 0.9 of the critical power over 8 cm
(radial, 128 nodes; Cartesian, 48 × 48), a filament at three times the critical power
with ionisation over 3 cm (radial, 256 nodes), the focusing beam on a 0.3 mm aperture
(radial, 64 nodes), and line and planar controls. Each ladder is scored by its worst
plane's relative field error over the band the error norm scores, against IP-DOPRI5 at
`rtol=1e-10`; fewest calls meeting each target, three interleaved warmed rounds, one
thread, Darwin/arm64 under load 2–6 (same-process comparisons). Calls / median seconds
at 1e-4 and 1e-6:

| case | IP-DOPRI5 | interaction | exponential |
|---|---|---|---|
| radial focus, `numba_cpu` | 181/0.28, 277/0.42 | 108/0.25, 167/0.38 | 44/0.18, 64/0.28 |
| radial focus, Kokkos CPU | 181/0.26, 277/0.39 | 108/0.22, 167/0.33 | 44/0.15, 64/0.23 |
| Cartesian focus, `numba_cpu` | 193/4.70, 343/8.30 | 86/3.55, 172/6.92 | 36/2.93, 66/5.37 |
| Cartesian focus, Kokkos CPU | 193/5.47, 343/9.65 | 86/3.35, 172/6.62 | 36/2.63, 66/4.82 |
| radial filament, `numba_cpu` | 325/3.82, 775/8.87 | 122/1.78, 387/5.30 | 84/1.97, 231/5.57 |
| radial filament, Kokkos CPU | 325/2.76, 775/6.46 | 122/1.29, 387/3.99 | 84/1.40, 231/4.08 |
| radial aperture, Kokkos CPU | 181/0.11, 283/0.18 | 110/0.10, 172/0.16 | 36/0.06, 151/0.26 |
| line, Kokkos CPU | 169/0.24, 169/0.24 | 44/0.09, 64/0.13 | 28/0.10, 45/0.18 |
| planar, Kokkos CPU | 169/0.34, 175/0.36 | 46/0.13, 89/0.24 | 37/0.18, 55/0.30 |

The exponential formulation takes the fewest calls everywhere and is the fastest on
the focusing beams (1.5–2.1 times IP-DOPRI5, 1.2–1.5 times the interaction formulation);
on the filament the two formulations are level in time with the exponential one taking
fewer calls; on the aperture the interaction formulation wins at 1e-6. Every Adams run
repeated to the call under 1e-14 perturbations of its launch: no case showed the
rounding-decided order regimes of the coupled wall. IP-DOPRI5's rows on the smooth
cases are held by the default controller's growth limit (`PIController(max_factor=1.1)`),
which caps each step at 1.1 times the last; they rank the default and bound the method.

**The unscored corner** (Q18). Scored over every bin, the filament's exponential runs
looked a hundred times miscalibrated (9.0e-5 at `rtol=1e-6`); their error is in a
corner the error norm does not score -- low frequency (3.5–4 µm) and the highest
transverse wavenumbers, where the field is 1e-10 of the peak -- about eighty times that
corner's content, present by the first plane (7.5 mm) and not growing; in band it is
8.8e-7.
The interaction formulation leaves 2.6e-7 there. A fixed order three leaves none (4e-6,
all in band, at the same calls): the high orders' lab-frame history integrates the
stiff, fast-rotating corner badly, and nothing scores it.

**Kokkos's GEMM dispatch** (merged as `perf/kokkos-gemm-dispatch`). Kokkos CPU ran the
radial route sixteen times slower than `numba_cpu`: 97 % of a march was the transform's
synthesis in the custom kernel. The `HYBRID` policy's time-grid guard (`n_to >= 1024`)
was a placeholder of the first native phase never replaced by a measured rule, and an
analytic field's or an envelope's transform -- a real matrix times complex samples --
never reached BLAS on macOS. Host builds now send every eligible transform to the
vendor GEMM, the complex ones as one real GEMM over the samples' interleaved real view
(as `numba_cpu` already did); measured with a build forcing BLAS everywhere, it won at
every shape tried: radial 1.1–35 times by size, a one-mode field capillary 2.4–3.1,
envelope capillaries 1.3–2.7, multimode field capillaries unchanged. Device builds
keep the earlier rule until their crossover is measured (G2).

**G0.** `device_profile.py` covers every case; the filaments have device-sized
variants (512 nodes; 128 × 128); `free_campaign.py` takes `--reference-backend`; the
session scripts (private: they name the providers) pack the exact heads as git bundles,
build for the device, and run G1, G2 (with the GEMM crossover, solprop-kokkos's
`benchmarks/gemm_shapes.py`) and G4.

**Decisions** (accepted in review, 2026-10-06, John Travers):

- *D37, resident coupled passes on Kokkos*, reversing D31 under the review's direction;
- *D38, a frozen-residual sample's generator applied once*, one state row a slot more;
- *the free-space campaign scored in band*, as the gate's scores are, the whole-field
  error kept beside it;
- *IP-DOPRI5 reported both at the default growth limit and at `max_factor=2`*, the
  default unchanged;
- *the spatial absorber not added to D34's weighting* until a case shows it needed.

**Open.** Q18 (above): documented; whether to mask the source in the unscored corner
for the exponential formulation is phase 6's, with a test that the corner stays bounded.

### GPU campaign, session 1 — the AMD device

**Scope.** 2026-10-06, Opus 5.5 (coding agent) at John Travers's direction: one
unattended session on one AMD CDNA 3 accelerator (Kokkos `AMD_GFX942`, ROCm, rocBLAS
and rocFFT through Kokkos-Kernels and KokkosFFT, the `HYBRID` GEMM policy), at
solprop `30d03b4` and solprop-kokkos `821fbd9` (kernel schema 22), before the
Runge-Kutta rejection fix. G1, G2 and G4 ran; G3's first candidates were built on CPU
afterwards and wait for the next session. Receipts: `benchmarks/sep16/data/device/`.

**Spend.** 2.65 device-hours, about US$10, of which about 0.8 h idled after G4 had
finished: the session's completion check matched its own command line. The device was
destroyed at the end of the session.

**G1 — qualification.** solprop-kokkos's suite: 235 tests, none failed (13 skipped).
solprop's Kokkos selection: 215 tests; 160 passed and 55 failed, every one for want of
an accuracy record for a device backend (`kokkos_hip`), not for a deviation. The
records' own writers regenerated them on the device, each asserting its derived bound,
and the 55 then passed: 83 rows, the largest deviation 2.2 % of its bound, the median
deviation equal to the CPU build's (`tests/data/convergence/spatial_accuracy.json`).
The selection ran the resident, exponential, varying-operator, coupled and memory
suites, the compensated reference and dense observation among them. Not done: a
`-Werror` device build, and the vendor's race and memory checkers. **Q16 is closed on
this device**; Q6's exception stands for the Hopper device.

**G2 — profile.** Each case's march on the device, split by timed native calls, and
every Kokkos kernel timed through Kokkos Tools' kernel timer (which fences each launch,
so small kernels read high). Per trial or source call:

- *the coupled windows*: the six coupled Adams kernels take about 0.6 ms a trial on the
  taper, each 70–150 µs, far above a launch for a few modes per bin, so badly spread
  over the device; a trial launches about 45 kernels;
- *the full canonical run*: about 49 launches a trial at a mean of 13 µs, bound by
  launch latency; a trial's native work takes 0.8 ms on the device against 0.32 ms on
  a laptop's CPU core;
- *the Cartesian filament (128 × 128)*: the Adams predictor and corrector take 7.5 ms a
  trial where the state's size would allow well under 1 ms; the source's plasma pass
  runs in 161 chunks of 102 columns (the default 64 MiB node budget), 161 launches a
  call, for every method;
- *the radial filament (512 nodes)*: the analytic field's synthesis ran the custom
  kernel at 2.1 ms a call, against 0.46 ms for the projection's GEMM at the same shape:
  the device kept the custom synthesis until its crossover was measured, and this
  measures it.

**G4 — work–precision on the device**, calls / median seconds at a target error (the
free-space cases in band, five interleaved rounds; the full runs at three; Kokkos HIP;
IP-DOPRI5 at the default growth limit):

| case, target | IP-DOPRI5 | interaction | exponential |
|---|---|---|---|
| radial focus, 1e-6 | 277/0.07 | 167/0.10 | **64/0.06** |
| Cartesian focus, 1e-6 | 343/0.31 | 172/0.28 | **66/0.16** |
| radial filament (512), 1e-6 | 775/1.88 | 388/1.15 | **235/0.80** |
| Cartesian filament (128 × 128), 1e-6 | 931/17.5 | 444/10.4 | 332/10.5 |
| full RDW, with the weak band, 1e-4 | 17443/10.1 | 8389/8.67 | **1704/2.84** |
| full canonical, with the weak band, 1e-4 | **1873/0.80** | 1415/1.22 | 993/1.48 |
| coupled wall, 1e-6 | **2827/0.71** | — | 1557/1.55 |
| coupled taper, 1e-6 | **4405/1.67** | — | 1555/2.36 |

The exponential formulation takes the fewest calls everywhere and is the fastest on every
free-space case but the Cartesian filament, where it ties the interaction formulation,
and on the RDW run by 3.6 times; it is slower than IP-DOPRI5 on the canonical run and
the coupled windows. On a device a source call is cheap enough that what a trial costs
beyond its source -- the host's scalars, the launches, the Adams kernels -- decides
those two. The GNLSE window and the CtC launch ran IP-DOPRI5 and the interaction
formulation only (their harness's methods), which the next session completes.

**G3 — the first candidates, built on CPU.** Ranked by G2; the first three are built:

1. *The host's work per trial.* Measured cleanly (timed native calls, no profiler) on
   Kokkos CPU, the canonical run's exponential trial spent 0.50 ms on the host, 60 % of
   it evaluating the operator's Chebyshev tables: each position is evaluated three
   times (a trial's end, its sample, the next anchor). A table now keeps its last few
   positions' values, read-only: 0.50 → 0.32 ms a trial, and IP-DOPRI5's step 0.96 →
   0.51 ms, bit for bit (the same calls and fields). What remains is the Adams
   coefficient tables (about 0.14 ms) and the five distinct evaluations (0.12 ms); the
   second would need numba in the translation layer and is left.
2. *The device synthesis* (solprop-kokkos). An analytic field's synthesis, and an
   envelope's synthesis and projection, follow the device's shape guard into the
   interleaved real GEMM, as the projection already did.
3. *The plasma chunks.* The free-space harness takes `SOLPROP_BENCH_NODE_BUDGET`; a
   chunk of every column gives the same fields bit for bit on Kokkos CPU. Whether the
   default budget should follow the device is for the measurement.
4. *Launches and the coupled kernels* (next, tuned on the device): fusing the Adams
   passes (the predictor with the source's prologue, the corrector with the defect
   norms), and a team per bin for the coupled block's matrix work.
5. *The spatial Adams passes*: the Cartesian predictor and corrector's memory traffic.

**Open.** Q6's exception for the Hopper device; G3's candidates measured on a device;
G4's GNLSE and CtC rows with the exponential formulation.

### GPU campaign, sessions 2 and 3 — G3's optimisations on the AMD device

**Scope.** 2026-10-06/07, Opus 5.5 (coding agent), overnight under the review's
authority (US$40 for the night): G3's candidates built and verified on Kokkos CPU, then
measured on the same AMD CDNA 3 device in two sessions, each built from the exact
heads, qualified (G1) and profiled (G2) before its work–precision runs (G4). Every
change is bit for bit on the host, or within the CPU suites' bounds where it reorders a
device reduction; Kokkos CPU's time is unchanged or better throughout. Receipts:
`benchmarks/sep16/data/device/session2/` and `session3/`.

**Spend.** Session 2: 1.54 device-hours, US$5.84; session 3: 0.79 h,
US$3.01. The campaign has spent US$18.92 of its US$100.

**What changed** (solprop-kokkos and solprop, `perf/sep16-device-overheads`):

- *The iteration order.* Kokkos's CUDA and HIP default for a two-dimensional range runs
  the first index fastest, which put adjacent device threads a row apart in every
  `LayoutRight` view; all 71 two-dimensional passes now run their last index fastest
  (`Range2`). The 128 × 128 filament's predictor and corrector became 2.5 and 2.8 times
  faster.
- *The device's interleaved GEMM.* An analytic field's synthesis took the custom kernel
  on a device even where the projection took rocBLAS: now both follow the device's
  shape guard. IP-DOPRI5 on the 512-node radial filament became 6.5 times faster, the
  exponential formulation 2.4.
- *Coupled passes a thread per (bin, channel)* on a device, where a thread per bin held
  sixteen-channel vectors; host builds keep the per-bin passes. The taper's coupled
  kernels became 1.6 times faster.
- *One upload of a trial's scalars* (twenty small copies before), *fused exponential
  passes* (the moments, the carry and the accepted state's copy in the predictor, the
  endpoint's rate in the corrector: 49 launches a canonical trial became 26), *the
  quadrature score and the Adams norm scale kept on the device* (two synchronisations
  a step fewer).
- *The host's work*: the z tables keep their last positions' values and evaluate by a
  compiled Clenshaw pass, and the Lagrange tables are compiled, all bit for bit. The
  canonical case's host work per exponential trial fell from 0.50 to 0.13 ms on the
  workstation, IP-DOPRI5's per step from 0.96 to 0.15 ms.
- *The free-space node budget* (`SOLPROP_BENCH_NODE_BUDGET` in the harness): at the
  default 64 MiB the 128 × 128 filament evaluates its source in 161 chunks a call; with
  every column in one chunk IP-DOPRI5 ran 4 times faster and the exponential formulation
  2.4. The default is unchanged (a recommendation below).
- *Rejected*: reading each history sample once into per-thread arrays in the
  elementwise passes. On the device the arrays doubled the defect norms (0.97 → 2.04
  ms on the filament); reverted.

**G1.** Both sessions, every build: solprop-kokkos's suite (222 passed, 13 skipped)
and solprop's Kokkos selection (215 passed), the per-channel coupled passes included.

**G4 after the optimisations**, calls / median seconds at two targets (Kokkos HIP; the
host-sensitive rows after the compiled tables):

| case | target | IP-DOPRI5 | interaction | exponential |
|---|---|---|---|---|
| GNLSE window | 0.0001 | 211/0.04 | 136/0.06 | 50/0.04 |
| GNLSE window | 1e-05 | – | 174/0.07 | 65/0.04 |
| CtC launch | 0.0001 | 49/0.42 | 28/0.26 | 17/0.17 |
| CtC launch | 1e-05 | 55/0.47 | 41/0.37 | 22/0.21 |
| full canonical | 0.001 | 1543/0.32 | 1090/0.45 | 721/0.40 |
| full canonical | 0.0001 | 1873/0.38 | 1415/0.57 | 993/0.56 |
| full RDW | 0.001 | 11041/3.84 | 5683/3.28 | 974/0.80 |
| full RDW | 0.0001 | 17443/6.05 | 8389/4.79 | 1704/1.36 |
| coupled wall | 0.0001 | 1285/0.30 | — | 953/0.57 |
| coupled wall | 1e-06 | 2827/0.66 | — | 1557/0.88 |
| coupled taper | 0.0001 | 1909/0.56 | — | 746/0.60 |
| coupled taper | 1e-06 | 4405/1.28 | — | 1555/1.23 |
| radial focus | 0.0001 | 181/0.03 | 108/0.05 | 44/0.03 |
| radial focus | 1e-06 | 277/0.04 | 167/0.07 | 64/0.04 |
| Cartesian focus | 0.0001 | 193/0.15 | 86/0.12 | 36/0.08 |
| Cartesian focus | 1e-06 | 343/0.24 | 172/0.20 | 66/0.12 |
| radial filament, 512 nodes | 0.0001 | 325/0.14 | 122/0.12 | 84/0.11 |
| radial filament, 512 nodes | 1e-06 | 775/0.25 | 388/0.25 | 235/0.22 |
| Cartesian filament, 128 × 128, 16 GiB | 0.0001 | 397/2.31 | 169/1.89 | 205/2.62 |
| Cartesian filament, 128 × 128, 16 GiB | 1e-06 | 931/4.14 | 444/3.39 | 332/3.67 |

The exponential formulation now wins on the device where it wins on the CPU, by as
much or more: the RDW run 4.4–4.8 times IP-DOPRI5 (3.5–4 times the interaction formulation),
the CtC launch 2.2–2.5, the Cartesian focus 1.9–2.0, the 512-node filament 1.1–1.3; it
ties on the GNLSE window and the radial focus. IP-DOPRI5 keeps the canonical capillary
(1.25–1.5 times) and the coupled wall (1.3–1.9 times); the taper is level, the
exponential formulation ahead at 1e-6; the interaction formulation keeps the
128 × 128 filament. On these, the exponential trial's fixed cost per call -- about
26 launches and 0.2 ms of host work on the device's host -- is what a cheap source
cannot hide, while IP-DOPRI5 spreads its own over six stage calls. The guidance and the
whole gallery are on the manual's new page, *Choosing a stepper*.

**Remaining candidates**, by what the profiles show (session 5 tested and rejected a
fourth, the custom kernel for skinny transforms): the spatial Adams passes, still about three times their memory traffic on the large
Cartesian state (the φ functions' evaluation); fusing the last few Adams launches (the
phase into the predictor, the scale into the norms); and a device-sized default node
budget, which would serve every method on the Cartesian route.

**Recommendations for decision.** (1) Merge `perf/sep16-device-overheads` in both
repositories: every change is measured, qualified on the device and neutral or better
on the CPU. (2) A device build's default node budget sized to the device (the
measured 2.4–4 times on the Cartesian filament), as its own change with its own test.
(3) The Hopper session (Q6's exception) with these heads.

### GPU campaign, session 4 — the NVIDIA Hopper device (Q6's exception)

**Scope.** 2026-10-07, Opus 5.5 (coding agent): one session on one NVIDIA Hopper
accelerator (80 GB; Kokkos `HOPPER90`, CUDA, cuBLAS and cuFFT, the `HYBRID`
policy), rented by the hour, at the heads of sessions 2 and 3's report (solprop-kokkos
`74fb556`, solprop `5577d09`). 0.60 device-hours, US$2.11; the campaign has spent
US$21.03 of its US$100. Receipts: `benchmarks/sep16/data/device/session4/`.

**G1.** The device build and its native tests passed; solprop-kokkos's suite passed (222,
13 skipped). solprop's Kokkos selection: 158 passed and 57 failed. Fifty-five wanted
accuracy records for the new backend (`kokkos_cuda`); the suite's own writers wrote
them on the device, 83 rows, the largest deviation 2.2 % of its bound, and the 55 then
passed. Two are real: `test_a_compiled_backend_follows_the_reference` for the
exponential formulation on the mode-averaged and the multimode gas-filled capillaries,
the same calls as `numpy_ref`, the field 1.18 and 1.25 times outside the test's bound
(3.5e-9 and 3.75e-9 of the peak, against 3e-9, a measured margin over the CPU's 1.1e-9,
where the ionisation rate amplifies rounding at the peak). The interaction formulation
and every other route pass. It is not the contraction of multiply–adds, which nvcc
does by default: Kokkos CPU built with `-ffp-contract=fast` differs from `numpy_ref` by
8.5–8.9e-10 of the peak on these two rows, as without it. The device's own exponential
and power functions in the ionisation rate, whose rounding differs from the host
library's and which the rate amplifies, are the likely source; the AMD device stayed
inside the bound. **Q6's exception: closed on both devices but for these two rows,
which are for decision** (below).

**G4 on Hopper**, calls / median seconds (the AMD device's table above for comparison):

| case | target | IP-DOPRI5 | interaction | exponential |
|---|---|---|---|---|
| CtC launch | 0.0001 | 49/0.25 | 28/0.16 | 17/0.11 |
| CtC launch | 1e-05 | 55/0.28 | 41/0.23 | 22/0.14 |
| full canonical | 0.001 | 1543/0.25 | 1090/0.42 | 721/0.42 |
| full canonical | 0.0001 | 1873/0.30 | 1415/0.54 | 993/0.58 |
| coupled wall | 0.0001 | 1285/0.25 | — | 952/0.60 |
| coupled wall | 1e-06 | 2827/0.55 | — | 1556/0.94 |
| coupled taper | 0.0001 | 1909/0.51 | — | 746/0.67 |
| coupled taper | 1e-06 | 4405/1.19 | — | 1555/1.36 |
| Cartesian focus | 0.0001 | 193/0.15 | 86/0.13 | 36/0.10 |
| Cartesian focus | 1e-06 | 343/0.23 | 172/0.21 | 66/0.14 |
| radial filament, 512 nodes | 0.0001 | 325/0.21 | 122/0.17 | 84/0.17 |
| radial filament, 512 nodes | 1e-06 | 775/0.35 | 388/0.32 | 236/0.29 |

The ranking is the AMD device's. On this host the Python of a trial took 0.36 ms
against 0.19 ms of device work for the canonical case (0.28 and 0.32 on the AMD
device's host): the slower host shifts the cheap-source cases further towards
IP-DOPRI5, whose host work is spread over six stage calls.

**For decision.** (1) The two capillary rows: a bound for device builds (5e-9 of the
peak, over the measured 3.75e-9), keeping the CPU's 3e-9, or a closer look at the
device's rate first. Recommended: the device bound, with this measurement cited. (2) The host work per trial is now
the larger share on NVIDIA systems; the next candidate is the factored defect rows'
`polyfromroots` (about 30 µs a trial), whose rounding follows NumPy's BLAS dot product.

### GPU campaign, session 5 — the device's transform dispatch, confirmed

**Scope.** 2026-10-07, Opus 5.5 (coding agent): the AMD device, the same heads built
twice, with the `HYBRID` policy and with the custom kernel forced
(`SOLPROP_KOKKOS_GEMM_PROVIDER=CUSTOM`), to test session 3's reading that rocBLAS spent
66–116 µs on a three-mode taper's transforms. 0.34 device-hours, US$1.28; the campaign
has spent US$22.31 of its US$100. Receipts: `benchmarks/sep16/data/device/session5/`.

**Result: the reading was the kernel timer's.** Per source call, the custom kernel
against the vendor GEMM: few-mode capillaries 0.94–1.13 times (field and envelope, 2–6
modes), radial routes 0.91–0.98 at 8–16 nodes and 1.10, 1.40, 1.70 and 2.98 times
slower at 32, 64, 128 and 256 nodes. The coupled windows and the CtC launch take the
same calls and times either way (the taper's IP-DOPRI5 at 1e-6: 1.27 and 1.31 s). The
timer fences every launch, which inflated a small GEMM's apparent cost. The device's
shape guard stays; the candidate is rejected.

### Q18 investigated — the exponential formulation's unscored corner

**Scope.** 2026-10-07, Opus 5.5 (coding agent), on `numba_cpu` (one thread): the radial
filament of the free-space campaign (256 nodes, three times the critical power, with
ionisation, 3 cm) at `rtol=1e-6` against IP-DOPRI5 at `rtol=1e-10`, and every campaign
with the exponential formulation's highest order set to four. Data:
`benchmarks/sep16/data/q18/`.

**The corner's content is not negligible in norm.** Outside the scored band the
reference holds 8.7e-5 of the field's norm at 3 cm (2.4e-4 at 6 cm). Part 2's "1e-10 of
the peak" is the field per bin there; summed over the corner it is not small.

**The error is the highest order's.** The corner's error as a fraction of the field's
norm, with the in-band error and the calls:

| highest order | calls | in band | corner error | corner amplitude, shape |
|---|---|---|---|---|
| 5 (the default) | 231 | 8.8e-7 | 9.0e-5 | 1.44 times, overlap 0.69 |
| 4 | 180 | 1.1e-6 | 5.3e-7 | — |
| 3 | 242 | 3.8e-6 | 8.0e-8 | — |
| interaction formulation | 387 | 1.0e-7 | 2.4e-7 | — |

At order 5 the corner is wrong by as much as it holds: 44 % too strong and only 69 %
of its shape. The error is made by the first plane (7.5 mm) and carried unchanged
after it; at twice the length it is still 8.8e-5, against a corner grown to 2.4e-4
(amplitude 1.06 times, overlap 0.94), while order 4 reproduces the corner exactly
(overlap 1.000) at both lengths, in fewer calls. The in-band error meets the tolerance
either way: the controller scores the band, and the corner never reaches it.

**Two remedies measured and rejected.**

- *Masking the source outside the band* (this question's original proposal) removes
  the corner's content altogether and makes the in-band error 30 times worse (2.5e-5 at
  order 5 or 4): the corner's content feeds back into the band.
- *A highest order of four for the exponential formulation* is free or cheaper at loose
  targets and doubles the calls at tight ones. The calls meeting each target, order 5 →
  order 4: radial focus 69 → 135 and Cartesian focus 78 → 158 at 1e-7; the coupled wall
  1555 → 2373 and taper 1555 → 2383 at 1e-6; line, planar and the GNLSE window 23–28 %
  more at their tightest; the full RDW run 1704 → 1530 at 1e-4, the radial filament
  231 → 276 at 1e-6 and 454 → 434 at 1e-7; canonical and CtC unchanged.

**The cause, partly.** The filament has no in-stage absorber (its in-stage damping rate
is zero), so it is not the term D34's order choice already weighs. The corner lies at
low frequency (3.5–4 µm), where the plasma term in the source is strongest against the
field (it scales as 1/ω): a stiff term the explicit history integrates, as the in-stage
collar is, against which order 5's stability interval is the shortest. The case cannot
be run without ionisation (the beam would collapse), so this is not yet shown.

**For decision.** (a) Keep order 5 and document the corner, with a test that bounds it;
(b) give the formulation's order choice a stability weight for the plasma term, as D34
does for the collar (development and a measurement first); or (c) offer a whole-field
error norm for runs whose out-of-band content matters. Recommended: (a) now, (b) as a
follow-up if a case shows the corner mattering.

### Phase 6 report, part 3 — the default, resume, signals and closing

**Scope.** 2026-10-07, Opus 5.5 (coding agent), at John Travers's request after the
GPU campaign's sessions. Merged to main one at a time on his review: the error norm over
the active band, `Multistep()` as the default integrator, the gallery measured again
under that norm, the manual, a resume that continues the history, the interrupt handling
and the test grouping. Then, on `feature/sep-0016-final`, the test suite moved onto the
default and this SEP closed.

**Q18 resolved: the norm scores the active band.** `PIController.region` selects the
bins the step controller's norm scores: `"active"`, the default, is every bin the
propagation evolves, the requested band and the margin around it in which generated
light propagates and the spectral collar absorbs it; `"band"` is the norm before.
The kernel contract carries the choice as `grids.norm_mask` (kernel schema 23), which
every backend's norm, the Adams engines' state magnitude and the batches read; the
output records it (output schema 14), and older records hash without it. This is
Q18's option (c), made the default: the light outside the requested band is no longer
integrated without control, so the filament's corner (Q18 investigated, above) is
scored rather than masked, and order 5 stays. A frozen-residual coupled flow keeps its
explicit residual, and the advisory step its bound, on the requested band, where the
wall model is valid. What remains unscored in effect are a soliton's wings below 1e-10
of its peak power: free radiation under cross-phase modulation, which rotates with each
bin's own free phase. There the exponential formulation's error is bounded by that
light's own content (1 to 8 % of it at `rtol` from 1e-7 to 1e-9) while the populated
bins follow the tolerance. The manual says so ([the Adams methods](../../../guide/numerics/adams.md#limitations)),
and `test_the_default_leaves_its_excess_error_in_the_dark_wings` bounds it.

**The device bound.** The two capillary parity rows that NVIDIA Hopper put outside the
CPU bound (3.5 to 3.8e-9 of the peak against 3e-9) are not multiply-add contraction: a
Kokkos CPU build contracting every multiply-add stays within 9e-10. The device's own
transcendental functions in the ionisation rate are the likely source. A device
build's parity with the reference is bounded at 5e-9, with this measurement cited
beside the bound.

**The default (ADR-045).** `Multistep()`, the exponential formulation, is the default
integrator on every entry point; `RungeKutta(tableau=DOPRI5)` names the Runge–Kutta
march, and `SimulationSettings.tableau` is gone. The kernel contract is unchanged:
translation passes the Runge–Kutta tableau with no Adams integrator, or the Adams
integrator. The method's representation always names its formulation (output schema
15). A run that fits in memory only without its history is refused before it allocates,
naming `RungeKutta()`; an adaptive run at `rtol` of 5e-2 or more is warned on; the
collar stability warnings name `RungeKutta()` where an Adams interval is the limit.

**The gallery and the manual.** Both CPU backends' campaigns were run again under the
active-band norm and the gallery redrawn from them; the device rows keep the norm they
were measured with and are labelled so (regenerating them is on the roadmap). The
manual sets the two integrators side by side: [the solver](../../../guide/numerics/solver.md)
introduces both and gives the Runge–Kutta one, [the Adams methods](../../../guide/numerics/adams.md)
describes the default, its options and its limits, and
[choosing a stepper](../../../guide/numerics/choosing-a-stepper.md) gives the short answer, what
decides, the measured ranking on both CPU backends, large problems and memory, and the
gallery.

**A resume continues the history (ADR-046).** The output keeps the march's state at
every save plane and terminal checkpoint (`MarchSnapshot`): the accepted state as the
backend held it, the engine's rows, each history entry's position, kind and row, the
order, the rejection count and the proposed step, rewritten in place so that the file
does not grow. A resume at the snapshot's position with the same method continues from
it, with no startup evaluation and no climb from order one: on `numpy_ref` and
`numba_cpu` the resumed run is the uninterrupted one bit for bit — adaptive or replayed,
compensated, beside a frozen-residual coupled flow, from a terminal checkpoint between
planes. `SimulationSettings.resume_history` keeps the snapshot while it is at most
256 MiB. Kokkos keeps its history resident and resumes afresh (deferred: a TDSE source,
the use that needs it, runs on `numpy_ref`, and a snapshot would copy the history off
the device at every save).

**A run stops between its commits (ADR-047).** An accepted step's state, its position
and a restart at a hard kink now change in one commit; before, an exception between
them left a checkpoint holding the next step's field at the previous position, and a
resume took that step twice (a 0.1 % difference, measured). During a run SIGTERM raises
`RunTerminated` and Ctrl-C `KeyboardInterrupt`, at once outside a commit and as the
commit ends inside one; the run writes its checkpoint and the march's state, so a job
killed at a scheduler's limit loses only the step in flight and `resume=True` continues
it.

**Test grouping.** A test is grouped with its module on one worker only when it shares
a module-, class- or package-scoped fixture, and with the tests of the same parameter
when that fixture is parametrised; others go to whichever worker is free. On a 14-core
workstation the default lane went from 628 s to 562 s.

**The suite under the default.** Tests exercise the default integrator; `RungeKutta()`
is named only where the march is the test's subject — FSAL and the seven stages, a
moving map combined per stage, CF4 sections, the conditioning certificate (which guards
the Runge–Kutta march's inverse flow action; ADR-044's formulation acts forward only),
frozen-split references, and checks of the march's own fifth-order convergence on
fixed schedules. Each of the last has a default-integrator variant beside it, adaptive
at a tight tolerance against the same reference, with a bound from measurement.
Bitwise records written under the Runge–Kutta march were recorded again under the
default. What the move found:

- *Fixed schedules beside the in-stage collar.* The collar's rate scales as the inverse
  of the run's length, so $h\alpha/2 < 0.0877$ asks for a number of replayed steps
  independent of the length: about 1440 for the default collar against about 40 for
  DOPRI5. The translation warns, naming the limit. Tests not about the collar split
  it off (`absorber_scheme="split_step"`); the tests about it step under the limit.
- *A replay beyond a stability limit can stay finite.* Growth confined to the collar
  reached the exit finite at ten steps and overflowed at twenty. The Adams replay's
  divergence error now names the collar's and a coupled flow's limits as the
  Runge–Kutta replay's does.
- *One refusal list.* `numba_cpu` refused an Adams integrator beside a coupled flow
  only when it built the march, after its own route checks, so a refused flow drew a
  different message from the reference's; the check now stands where the reference's
  does, after `numba_cpu`'s own CF4 refusal (no integrator runs CF4 there).
- *A sourceless run beside a frozen-residual coupled flow is stepped by its frozen
  block alone.* The residual joins the compensation, which is sampled with the source
  (ADR-042, ADR-044); with no source there is no sample, and the block's chord carries
  the step at an error the controller does not see. A diagonal operator's chord is its
  exact flow, so this is the coupled flows' case only. A linear graded-index taper at
  one bin is 6.4e-4 from the converged answer whatever `rtol`, falling at second order
  with `dz_max` (4.0e-5 at a quarter, 2.5e-6 at a sixteenth); with its Kerr term the
  same taper conserves energy to 5e-11 at `rtol=1e-8`. The manual states it; the
  remedy is on the roadmap.
- *Per-step plasma statistics cost a scan each.* The Runge–Kutta march reads them off
  its last stage, which is the accepted state (2.5 % of the reduced M2 run at a period
  of one); a default step's one evaluation is at its predicted state, so a row scans the
  columns afresh and the channel added 80 % to the same run. John Travers kept the
  statistics as they are (the channel is off unless asked for, and the planes' are
  exact); the manual states the cost, and scanning only the peak-field node is on the
  roadmap.
- *Kokkos's Adams engine reduced its norms in thread order.* On host builds the
  magnitude and the defect norms were a `parallel_reduce`, so an adaptive run forked with
  the thread count (a split block's steps from the eighth on), against the
  deterministic-reduction rule every backend owes. They now sum fixed chunks in index
  order and combine the partials serially; a device build's fixed-order form is left
  to the next device session. The `adams_source` binding also dropped a moving guide's
  map weights, so a moving guide whose Adams source runs on the host path failed.
- *An Adams store's unused rows were uninitialised*, and went into a resume snapshot as
  they were, so two runs' files differed there; they are zeroed.
- *The guide's CF4 example* now names `RungeKutta()`, which CF4 needs; Kokkos, like
  `numba_cpu`, refuses a CF4 section in its own words before the integrator's.
- *The split-step collar on the general modal route* treats a launch's content
  outside the active band differently on Kokkos and `numpy_ref` (zeroed against
  carried); found when Kokkos's general modal parity cases left the in-stage collar,
  which now run without one. On the roadmap, untraced.
- *Replayed resumes are bitwise on the host engines* (they differed by one unit in the
  last place before ADR-046); on Kokkos a replay resumed afresh is a step error away
  (1.4e-10 to 5.2e-7 of the peak on the cases measured).

**Gates.** On `feature/sep-0016-final`, on Darwin/arm64 with the Kokkos build of the
same branch installed: the default lane 13,779 passed and 28 skipped, with two failures
each passing alone and neither the branch's (the plasma-scan prototype's intermittent
NaN, and the distribution test's nested 45 s run, which takes 2 s alone, timing out
under the lane's load); the slow lane, numpy and numba, passed once the M2 witness
record was measured under the default; the slow lane with the Kokkos backend selected,
139 passed; the coverage lane 13,781 passed at 100 % of lines and branches; the
Kokkos repository's quality gate (`scripts/check.sh` against this branch: formatting,
lint and types, the tidy and sanitizer builds with their native tests, 231 Python
tests passed and 5 skipped, and the distribution wheel's isolated smoke test) passed
on `feature/sep-0016-final` there. The slow lane found the Kokkos and store defects
above and the statistics' cost; the coverage lane, a guard that could not be false and
a batch test passing for another reason than its own; the Kokkos gate, a type stub
missing two methods, a throwing accessor declared `noexcept` and its own tests' place
on the new default (all from the Kokkos commits since its last gate run).

**Open questions, moved.** Q1–Q11 and Q15–Q18 are resolved above or in the reviews.
Q13 and Q14 are closed: Kokkos's step arithmetic is native with the resident history
(phase 3b), the Adams coefficient tables are compiled (phase 6), and where a cheap
source's call saving does not become time is measured and stated in
[choosing a stepper](../../../guide/numerics/choosing-a-stepper.md). To the
[roadmap](../../roadmap.md): Q12 (the exponential formulation beside the `modified`
noise on a developed noisy window, until which the manual prefers the interaction
formulation there), a sourceless run beside a frozen-residual coupled flow, a Kokkos resume that continues the history,
the device gallery under the active-band norm, a device-sized node chunk, a
fixed-order form of the Adams engine's norms on devices, plasma statistics between
planes from the peak-field node, and the split-step collar's out-of-band content on
the general modal route. CF4 and
other general varying coupled flows stay with the Runge–Kutta march, as the roadmap
already records.

## Part III — History

- **2026-10-05:** Drafted by GPT-6 Astra at John Travers's request following the SEP 15
  study and production-scope discussion. John Travers requested supersession of SEP 15,
  a separate SEP 16 review, broad route/noise/taper qualification, optimisation and a
  work–precision gallery. Implementation is to start on main after SEP 12 integration.
  Status is Draft; no human or independent AI review of this proposal is recorded.

- **2026-10-05 — AI review, Opus 5.5:** reviewed all of Part I against SEP 0, the
  archived SEP 13–15 records, the benchmark ledger, the low-call manual, the current
  `Multistep`/`AdamsMarch` admission and refusal code on all three backends, the linear
  operator and coupled-flow representations, the noise refusals, SEP 12's implementation
  record and the SEP 15 study's constant-reference march. Revised the proposal directly:

  - §4.2: displayed the full update; required the residual's rate to be the derivative
    of the flow's own representation (which the tables supply exactly), bounded the
    rate jumps at adaptive breakpoints, and named the cancellation hazard; identified
    that the study's untransported compensation carries each bin's free phase and loses
    accuracy once $|\operatorname{Im}D|h\gtrsim1$, and proposed the transported
    compensation (Q8); refined the literature claims.
  - §4.3: dense formula for the transported compensation and its endpoint inverse flow.
  - §4.4: the modified draw's own-phase rotation as a likely limit on the exponential
    advantage; coupled-flow noise and dense-observation refusals kept, since RK refuses
    them too.
  - §2 and §4.5: the spectral collar is already exact in $D$; the three existing
    coupled-section kinds, of which constant-Schur and frozen-residual already have the
    reference-plus-residual structure and need no exponential-of-integral identity
    (Q3); the adiabatic routes that actually exist; noise only where RK admits it
    (extending noise to the planar route would be new physics).
  - §3, §5, §6, §9, §11: current admission facts, schedules for a multistep method,
    the record-name/hash consequence of a new field, `backend="auto"`, rates by
    representation, the separable route's per-element moments, and the phase of each
    `SCHEMA_VERSION` bump.
  - §7: manufactured long-march varying-operator and noise-rotation tests; long runs and
    the TDSE configuration put to decision (Q5, Q9); a delivery decision (Q10).
  - Added references [7] and [8].

  **Sources checked:** Cox–Matthews §2, Hochbruck–Ostermann (2010) §2,
  Hochbruck–Ostermann (2011) §§2, 4 (Assumption 4.1, Theorem 4.3, Remark 4.4) and 5,
  Caliari et al. §§1–2 (accepted manuscript), Kassam–Trefethen §2, Demirel et al.
  §§2–4 and Lawson's bibliographic record, from the extracted article texts with
  validated identities; all six original references' bibliographic data agree with the
  validated records. Hairer–Nørsett–Wanner was not available as an extracted text and is
  cited for its standard variable-step Adams material only; its DOI was checked against
  the registry record.

  **Numerical cross-check:** a scalar model $u'=i(\omega_0+az)u+F$, $u(0)=1$, with the
  exact solution by adaptive quadrature, exact starting values and fixed steps, using
  the PEC recurrence with predictor-state samples at orders PEC2/3 to PEC5/6. With
  $\omega_0=2000$, $a=20$, $F=500$ (1 % variation of $D$ over unit length) and 400–6400
  steps, the untransported compensation's final relative error was
  $5.7\times10^{-5}$–$5.5\times10^{-2}$ for $\omega_0h=0.62$–5, the uncompensated chord's
  $5.6\times10^{-8}$–$1.3\times10^{-5}$, and the transported compensation's
  $1.4\times10^{-8}$–$1.7\times10^{-5}$, falling to $1.9\times10^{-10}$ at
  $\omega_0h=0.31$ (PEC5/6). Source-free with $\omega_0=200$, $a=100$ (PEC4/5, 20–1280
  steps): untransported $1.1\times10^{-5}$–$0.69$, transported
  $8.2\times10^{-11}$–$0.56$, chord exact to round-off. This tests the algebra of one
  scalar bin, not an adaptive implementation, a physical case or the study's measured
  windows, which remain valid at their stated accuracy.

- **2026-10-05 — Decisions in the review discussion, John Travers:** asked whether the
  proposal has value given the review's finding. The review's assessment, recorded in §2:
  the study's measurements stand for their windows; the constant-operator gains are the
  largest and are unaffected; the full canonical gain is real but modest; the RDW window
  does not establish a full-run gain; and the TDSE capillary at its operating accuracy
  shows no call advantage. John Travers accepted the review's recommendations and
  resolved Q1, Q3 (scope), Q5, Q9 and Q10; re-scoped the proposal into a constant-operator
  slice that also completes the existing method's compiled routes, a varying-operator
  slice behind the phase 1 gate (Q8), and gated breadth; and made the TDSE response a
  compatibility requirement rather than a target. Part I was rewritten accordingly.
  Status remains Draft; review status In review, with no human review of Part I recorded.

- **2026-10-05 — Varying-operator gate, run during review by Opus 5.5 at John Travers's
  request:** the SEP 15 study implementation, extended with the transported compensation
  and the uncompensated chord (all fixed order four), propagated the full 1.5 m M2/RDW
  case (15 planes) and the full 4.2 m canonical case (50 planes) from launch, against
  IP-DOPRI5 references at `rtol=1e-10` whose refinement against `1e-9` was 2.5e-8 and
  4.1e-6. Each method swept a tolerance ladder; the fewest calls meeting every scored
  error at a common target were timed in interleaved warmed rounds, one thread, Numba,
  on Darwin/arm64 under background load (same-process comparisons). At a common 1e-4
  error (calls / median time):

  | Case | IP-DOPRI5 | current Multistep | untransported | transported | chord |
  |---|---:|---:|---:|---:|---:|
  | full M2/RDW | 17443 / 19.3 s | 8389 / 12.1 s | 1533 / 3.30 s | 1530 / 3.91 s | 1528 / 3.09 s |
  | full canonical | 1873 / 623 ms | 1415 / 579 ms | 1051 / 521 ms | 1049 / 574 ms | 1139 / 556 ms |

  The RDW advantage holds at 3e-4 and 1e-3 and at the tighter 3e-5 and 1e-5 targets,
  which IP-DOPRI5 does not reach within its ladder. Per-plane errors show no
  accumulation peculiar to any treatment. Adaptive scalar checks were added to the
  review's evidence (§4.2). The gate passes; the treatment is Q8's decision. Evidence:
  `benchmarks/sep16/` at revision `2f93df8` on `bench/sep-0016-gate`, and the
  [benchmark ledger](../../benchmarks.md).

- **2026-10-05 — Q8 decided, John Travers:** asked why the more robust transported
  compensation should not be chosen. A fused implementation, forming each sample's
  factor once per trial, reproduces the transported fields to 1.3e-13 with identical
  calls and costs 6–14 % more time than the untransported one (RDW at 1e-4: 3.50
  against 3.17 s, current Multistep 11.84 s; canonical: 560 against 518 and 577 ms).
  The review's earlier recommendation of the untransported treatment had weighed an
  unoptimised implementation's cost over a structural robustness advantage. John
  Travers chose the transported compensation, with its exponential-free form as a
  phase 4 cost target and the untransported one kept as a test witness. Evidence:
  `benchmarks/sep16/` at revision `efe657e` on `bench/sep-0016-gate`.

- **2026-10-05 — Human review and acceptance, John Travers:** read the whole of Part I
  and approved it, completing the required human and AI reviews. Status changed to
  Accepted; Part I is frozen. Open questions Q2, Q4, Q6 and Q7 are deferred to their
  named phases; the others are resolved. Implementation has not started.

- **2026-10-06 — Implementation started, Opus 5.5:** phase 1 begun on
  `feature/sep-0016-unified-multistep-1`, working autonomously at John Travers's request,
  with phases 2 and 3 to follow on branches of their own for his review. Status changed
  to In progress.
- **2026-10-06 — Phase 1 reported, Opus 5.5:** baseline revisions, the capability
  audit, the interfaces of phases 2–4, the performance baseline (Q2) and the numerical
  witnesses; Q11 opened for phase 4. Not merged; for review.
- **2026-10-06 — Phase 2 reported, Opus 5.5:** the exponential formulation on
  `numpy_ref` for every constant-operator route, kernel schema 20 and output schema 13,
  ADR-041, schedules and the `modified` noise for both formulations, the TDSE
  compatibility check, the spatial pilot and a short CtC case; Q12 opened. Not
  merged; for review.
- **2026-10-06 — Phase 3 reported, Opus 5.5:** both formulations on `numba_cpu` and
  Kokkos CPU for every constant-operator route, `backend="auto"` as for RK, the
  constant-operator timing campaign and the §7.3 delivery decision (the exponential
  formulation stays public; criterion met in the expensive-source regime); Q13 opened.
  Not merged; for review.
- **2026-10-06 — Phase 4 reported, Opus 5.5:** varying diagonal operators with the
  transported compensation on all three CPU backends (ADR-042), the full canonical and
  M2/RDW campaigns on the production code, and a fix to the restart at a kink that is
  also a save plane; Q14 opened. Phase 5 not started. Not merged; for review.

- **2026-10-06 — Review of phases 1–4, John Travers:** accepted D1–D19, the §7.3
  decision and the SEP 12 parent conversion; resolved Q2 and Q11, completed Q7 for
  phase 4 with re-run full campaigns, recorded the Q6 GPU exception, sent Q12–Q14 to
  phases 3b and 6, and found three §11 items owed (the resident Kokkos history, memory
  ledgers and the exponential-free transport), taken as phase 3b before phase 5.
- **2026-10-06 — Phase 3b reported, Opus 5.5:** the Adams engine (ADR-043, kernel
  schema 21) with Kokkos's resident history for both formulations on every route but
  a coupled flow, the compensated reference included; the `adams_history` memory
  group on every backend; §4.2's exponential-free transport measured slower than the
  exponential one and left opt-in. Q15 and Q16 opened. Not merged; for review.
- **2026-10-06 — Phase 5 reported, Opus 5.5:** coupled constant-Schur and
  frozen-residual sections in the exponential formulation on every CPU backend
  (ADR-044, kernel schema 22); Q3 qualified, Q4 resolved by the collar ablation (no
  implicit treatment), Q17 opened. Not merged; for review.
- **2026-10-06 — Review of phases 3b and 5, John Travers:** for phase 3b, accepted its
  decisions, removed the exponential-free transport (D26) and resolved Q15; set a GPU
  optimisation campaign after phase 5. For phase 5, accepted D28–D33, kept the in-stage
  collar as the default and split Q17 between phase 6 and the campaign, whose plan is
  in Part II for the go that device use needs.
- **2026-10-06 — Phase 6 part 1 reported, Opus 5.5:** Q17 on CPU (the coupled matrix
  functions from the flow actions' eigenbasis, compiled coupled passes on `numba_cpu`,
  coupled windows timed) and the exponential formulation's order choice weighing the
  in-stage collar; D34–D36 for review. Not merged.
- **2026-10-06 — Review of phase 6 part 1, John Travers:** accepted D34–D36; asked for
  as much of the GPU campaign's preparation as is reasonable on CPU before any device
  session.
- **2026-10-06 — Phase 6 part 2 reported and reviewed, Opus 5.5 and John Travers:**
  Kokkos's resident coupled passes (D37, D38), free-space qualification on CPU (Q18
  opened), Kokkos's host GEMM dispatch (merged separately), the GPU campaign's G0; the
  review accepted every recommendation and authorised the first device session.
- **2026-10-06 — GPU campaign session 1 reported, Opus 5.5:** the AMD device; G1
  (Q16 closed there), G2 and G4; G3's first three candidates built on CPU. Not merged.
- **2026-10-07 — GPU campaign sessions 2 and 3, Q18 and the gallery, Opus 5.5:** G3's
  optimisations measured and qualified on the AMD device (US$18.92 spent in all); Q18
  investigated (for decision); the measured gallery and guidance page. Not merged.
- **2026-10-07 — GPU campaign session 4 reported, Opus 5.5:** the NVIDIA Hopper device;
  G1 passes but for two capillary parity rows (for decision), its accuracy records
  written; G4 ranks the methods as the AMD device does. US$21.03 spent in all.
- **2026-10-07 — GPU campaign session 5, Opus 5.5:** the device's transform dispatch
  measured against the custom kernel and confirmed. US$22.31 spent in all.
- **2026-10-07 — Closing report, Opus 5.5:** the active-band norm (Q18), the device
  bound, `Multistep()` as the default (ADR-045), the gallery and the manual, a resume
  that continues the history (ADR-046), the interrupt handling (ADR-047) and the test
  grouping, merged on John Travers's review; the test suite moved onto the default
  integrator; the open questions moved to the roadmap.
- **2026-10-07 — Final, John Travers:** approved the Final status with the coupled flow
  without a source stated in the manual and its remedy put on the roadmap. Status
  changed to Final; the SEP is archived.
