# SEP 15 — Exponential Adams stepping for dispersive, broadband problems

| | |
|---|---|
| **SEP** | 15 |
| **Title** | Exponential Adams stepping for dispersive, broadband problems |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-05); GPT-6 Astra (revision, 2026-10-05) |
| **Status** | Superseded |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 Astra (AI); John Travers (human) |
| **Type** | Feature |
| **Created** | 2026-10-05 |
| **Resolution** | Accepted by John Travers, 2026-10-05, in the SEP 15 review discussion; superseded at his request, 2026-10-05 |
| **Requires** | [SEP 13](sep-0013-expensive-response-steppers.md), [SEP 14](sep-0014-stepper-efficiency.md) |
| **Supersedes** | — |
| **Superseded-by** | [SEP 16](sep-0016-unified-multistep.md) |
| **Implementation** | GPT-6 Astra; `feature/sep-0015-exponential-adams-study` (phase 1) |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

SEP 13's interaction-picture Adams march needs one source evaluation per step, and on a
weakly dispersive gas-filled capillary it reaches IP-DOPRI5's accuracy with 2.5–3.5
times fewer evaluations. SEP 14 found that the advantage shrinks to about 1.3 times on a
Raman-active fibre and turns into a loss on a fibre supercontinuum. The diagnosis suggests
phase-mismatched forcing: a source that varies slowly in the propagation frame can
oscillate rapidly after interaction-picture transport, making polynomial interpolation
over several past steps expensive. The scalar phase limit depends on the order,
tolerance and source; it is not a universal bound for populated spectral bins. This
SEP proposes a bounded study of the *exponential* Adams family, which fits the
polynomial to the untransported source in the existing propagation frame and
integrates the linear phase exactly, and, if the study succeeds, a method beside SEP 13's. Success is judged on calls
and wall time against IP-DOPRI5 and SEP 13's method at matched error, on the cases where
SEP 13's method loses.

### 2. Motivation and scope

**The measurement.** SEP 14's runtime study (phase 2) and its diagnosis
(`benchmarks/sep14/adams_limits.py`) give, for each case, a proxy $\Omega_D$ for the
largest linear phase rate. The script estimates $D$
near the entrance, selects frequency bins whose saved spectral power reaches
$10^{-6}$ of the peak, and maximises $|\operatorname{Im}D|$ across the selected
frequencies and modes. The table reports $\Omega_D h$ at each method's median step;
it does not measure the phase of $N$ or its mismatch with the linear flow:

| case | phase rate | $\Omega_D h$, SEP 13's Adams | $\Omega_D h$, IP-DOPRI5 | Adams wall time over IP-DOPRI5 |
|---|---|---|---|---|
| canonical gas-filled capillary | 44 m⁻¹ | 0.28 | 1.3 | 0.55–0.8 |
| CtC slice (Raman sidebands) | 1.0e4 m⁻¹ | 0.75 | 3.8 | about even |
| RDW (UV dispersive wave) | 5.3e3 m⁻¹ | 1.4 | 7.7 | 0.75–1.0 |
| fibre supercontinuum (GNLSE) | 2.0e5 m⁻¹ | 2.1 | 11 | 2.3–2.7 |

SEP 13 measured $\Delta\beta h \approx 0.56$ for PEC4/5 in its scalar
oscillatory test, with error relative to the exact forcing increment below
$10^{-3}$. That test-specific value cannot
be applied as a stability or accuracy boundary to this table: the source amplitudes,
orders, error norms and actual phase mismatches differ.
Narrowing the supercontinuum's window from 400–2000 nm to 600–1300 nm removes its most
detuned bins and cuts SEP 13's step penalty against IP-DOPRI5 from 5.2 to 2.9 times,
which supports a role for far-detuned bins without isolating the mechanism:
changing the window also changes the represented problem. The supercontinuum has no
temporal collar, and capping the order to gain imaginary-axis stability made its
steps shorter. These checks do not exclude nonlinear stability or estimator effects
in general. Phase 1 must test the forcing explanation directly.

**Who needs it.** Low-call stepping matters where a source evaluation is expensive. SEP
12's TDSE response is in a weakly dispersive gas-filled capillary, where SEP 13's method
already wins; a TDSE or similarly expensive response in a solid-core fibre, or a
broadband multimode capillary run with expensive Raman, could encounter the source-interpolation limitation described above.

**In scope.** The exponential Adams family (§4) in a NumPy study on manufactured problems
and on short windows of the constant-$D$ cases; if the exit criterion of §7 is met,
its implementation for varying $D$ and comparison on all four cases, in the
shared Adams march on `numpy_ref` and then the compiled backends.

**Not in scope.** Delivery of exponential Runge–Kutta methods; changing any default.
IP-DOPRI5 and ETD Runge–Kutta are different families; resolving transported forcing
with several RK stages does not make their error or stability properties equivalent.
The temporal collar's stability
limit, which SEP 14 phase 4 studied, is a separate problem: this proposed method treats
exactly the operator diagonal in frequency, and the collar is diagonal in time, so it
stays a stiff term of the source in either picture. An implicit treatment of the collar
is raised as Q5.

### 3. Context

- [Low-call stepping](../../../guide/numerics/adams.md) and SEP 13's selected method,
  `Multistep()`: one-call PEC, variable order up to five, history transported by the
  linear flow (`solprop.backends.multistep.AdamsMarch`).
- [The solver](../../../guide/numerics/solver.md) and [stepper design](../../stepper.md) §§2–4: the
  interaction picture, the cumulative linear operator $\Phi(z)$ and its tables.
- SEP 14 phase 2 (the runtime study and the diagnosis) and phase 4 (the collar study).

### 4. Theory

The propagation equation is $\partial_z a = D(z)\,a + N(a, z)$, with $D$ diagonal in the
frequency bins and $\Phi(z) = \int_0^z D(s)\,ds$. Here $z$ and $h$ are in metres,
$D$ is in m⁻¹, $\Phi$ is dimensionless, and $N$ has units of $a$ per metre. The
state is the internal, vacuum-normalised amplitude with the FFT sign of
[conventions](../../conventions.md): $\operatorname{Im}D$ is the negative residual
propagation constant. "Untransported" refers to this existing co-moving propagation
frame, not a change to a laboratory time coordinate or to the optics convention.
Over a step from $z_n$ to $z_{n+1} = z_n + h$,

$$
a(z_{n+1}) = e^{\Phi(z_{n+1}) - \Phi(z_n)} a(z_n)
  + \int_{z_n}^{z_{n+1}} e^{\Phi(z_{n+1}) - \Phi(s)}\, N(a(s), s)\, ds .
$$

**The interaction-picture (Lawson) Adams method**, SEP 13's, interpolates the
*transported* source $g(s) = e^{\Phi(z_n) - \Phi(s)} N(s)$ by a polynomial through the
history and integrates it. Lawson [1, §2, (2.2)–(2.6)] and Hult
[2, §II.D, (9)–(12)] justify the interaction-picture transformation and RK
construction; SEP 13 supplies the Adams algorithm, not either of those papers.
For a bin driven by a pump whose phase is near the
frame's, $N(s)$ varies slowly but $g(s)$ oscillates as $e^{-i \operatorname{Im} D\,(s -
z_n)}$, which the polynomial must follow over the whole history span.

**The exponential (ETD) Adams method** interpolates $N(s)$ itself and integrates it
exactly against the exponential [3, §2.1.1; 4, §2.5]. For constant $D$ over the step and a
polynomial $p(\theta) = \sum_j N_j \ell_j(\theta)$ on normalised nodes $\theta_j$,

$$
a_{n+1} = e^{hD} a_n + h \sum_j w_j(hD)\, N_j,\qquad
w_j(x) = \int_0^1 e^{x(1 - \theta)} \ell_j(\theta)\, d\theta
       = \sum_k c_{jk}\, k!\, \varphi_{k+1}(x),
$$

with $\ell_j(\theta) = \sum_k c_{jk}\theta^k$, distinct nodes
$\theta_j=(z_j-z_n)/h$, and $N_j=N(a_j,z_j)$ (or a labelled predicted-field
sample in PEC). The identity follows by integrating each monomial:

$$
\varphi_r(x)=\sum_{m=0}^{\infty}\frac{x^m}{(m+r)!},\qquad
\int_0^1 e^{x(1-\theta)}\theta^k\,d\theta=k!\varphi_{k+1}(x).
$$

Thus $\varphi_0=e^x$, $\varphi_r(0)=1/r!$, and
$\varphi_{r+1}(x)=(\varphi_r(x)-1/r!)/x$ away from zero. The last recurrence is
an algebraic identity, not a stable evaluator near zero. Use a converged series in
that region and validate the switch to recurrence over complex $x$, including
large imaginary parts and negative real parts. Kassam–Trefethen [5, §2] provide
an independent contour strategy and explain cancellation; their real-part,
half-contour shortcut assumes a real argument and must not be applied to an
individual complex dispersive bin. Validate complete weights as well as individual
$\varphi$ values, since the polynomial coefficients can amplify round-off.

For example, nodes $0,-1$ give $w_0=\varphi_1+\varphi_2$ and
$w_1=-\varphi_2$, agreeing with [3, (6); 4, (2.71)]. At $D=0$ all weights reduce
to the ordinary variable-step Adams weights. Constant forcing is integrated exactly
for any $hD$, including a stationary balance $Da+N=0$.

**What the literature establishes.** Hochbruck–Ostermann §2.5 derives the
one-interval exponential Adams formula and gives a fixed-step convergence theorem
under smoothness and semigroup assumptions. Its error depends on derivatives of
$N(a(z),z)$; exact treatment of $D$ does not make a rapidly oscillating source
smooth or establish uniform accuracy for dispersive spectra. Calvo–Palencia
[6, §2, (6)–(8)] instead advance from $z_n$ to $z_{n+k}$ over $kh$. Their method
is a related alternative, not this Adams formula; their theorem is not a proof for
our variable-step, variable-order PEC history. Phase 1 must establish the order
and stability of the actual stored-history algorithm.

**Where each frame is smooth.** Write a scalar forcing as $F e^{i\kappa z}$
and $D=i\omega_D$ for this lossless illustration. ETD interpolates a source with
phase rate $\kappa$; Lawson interpolates one with rate $\kappa-\omega_D$.
Both rates have units m⁻¹. For constant $D$, including loss, its exact solution is

$$
u(z_n+h)=e^{hD}u_n+
 F e^{i\kappa z_n}h e^{hD}\varphi_1\bigl(h(i\kappa-D)\bigr).
$$

The resonant limit $i\kappa=D$ is regular. In particular:

- A bin driven by a slowly varying pump source ($\kappa\approx0$ in the chosen
  frame) favours ETD, even when the receiving bin's linear phase is large.
- An established bin with $N=i c(z)u$, where $c$ is real and varies slowly, can
  favour Lawson: the source carries the bin's own free phase. Its exact solution
  is $u(z)=u(0)\exp(Dz+i\int_0^z c(s)\,ds)$.

A supercontinuum can have both behaviours, even within one bin. A source phase
cannot safely be inferred from field power alone, or from the phase of a source
near zero. The study compares interpolation residuals of recorded sources in both
frames, together with field and weak-band errors, before claiming a cause.

A **secondary exploratory option**, only after plain ETD is measured, is a fixed
per-bin $\alpha\in[0,1]$ and
$g_\alpha(s)=e^{-\alpha(\Phi(s)-\Phi(z_n))}N(s)$. For constant $D$ the corresponding
weights are

$$
w_j^{(\alpha)}(x)=e^{\alpha x}
 \sum_k c_{jk}k!\varphi_{k+1}\bigl((1-\alpha)x\bigr),\qquad
 a_{n+1}=e^x a_n+h\sum_j w_j^{(\alpha)}(x)g_{\alpha,j}.
$$

This recovers ETD at $\alpha=0$ and Lawson at $\alpha=1$. It is a different
interpolation choice, not a free improvement: changing $\alpha$ requires all
retained samples to be represented in the new frame, or a restart. Tests must
cover frame changes, losses, conditioning and rejection rollback. A population-based
adaptive rule is not specified or approved by this SEP; Q2 owns that decision.

**A z-dependent operator.** Since the admitted $D(z)$ commute, the exact flow is
already $e^{\Phi(b)-\Phi(a)}$: all Magnus commutators vanish. The unresolved work
is the weighted source integral, not a noncommuting linear propagator. The proposed
reference strategy is quadrature of

$$
w_j(z_n,h)=\int_0^1
 e^{\Phi(z_n+h)-\Phi(z_n+\theta h)}\ell_j(\theta)\,d\theta.
$$

Bound the induced state error $h\sum_j\delta w_j N_j$ in the same norm as the
step defect, account for the cumulative table's error, and check quadrature by
independent refinement, including oscillatory cases where two rules can alias.
The coefficient-error budget is separate from the interpolation-error estimate;
a small predictor–corrector difference does not detect shared weight error.
Retain the exact endpoint flow even when weights are approximated. Replacing
$\Phi$ inside the integral by its endpoint chord needs its own error control:
for smooth $D$, the exponent error is generally $O(h^2)$ and the source-integral
error $O(h^3)$, which can limit global order to two. Quadrature is the proposed
oracle; phase 2 compares its cost with a controlled expansion about constant $D$.

**Predictor, correction and error control.** Let the predictor use $q$ history
nodes and the corrector use those same nodes plus $\theta=1$. Form the predicted
field, evaluate $N^P_{n+1}$ once there, and use that sample in the corrector.
PEC accepts the corrected field while storing the predicted-field source. This
sample identity must remain explicit. A single correction is an explicit method,
not an implicit solve and not an implicit method's stability guarantee.

For exact smooth history at fixed $D$, define

$$
 C_q(x)=\frac{1}{q!}\int_0^1 e^{x(1-\theta)}
       \prod_{j=0}^{q-1}(\theta-\theta_j)\,d\theta.
$$

The predictor's local defect is
$h^{q+1}C_q(hD)N^{(q)}(z_n)+O(h^{q+2})$. With an exact endpoint source the
corrector has one higher formal order; evaluating at the predictor introduces
another error $h w_{\mathrm{end}}(hD)(N^P_{n+1}-N(a(z_{n+1}),z_{n+1}))$.
Thus the predictor–corrector difference estimates the leading **predictor**
defect asymptotically. It is not a guaranteed bound for the accepted PEC field,
particularly at large $|hD|$, near coefficient cancellations, or with inexact
stored sources. The actual augmented recurrence needs convergence and estimator
checks. Do not transfer a scalar Milne factor or SEP 13's stability intervals.

Use the existing physical endpoint error norm (including configured weak bands),
without an extra Lawson transport of an ETD defect. Derive neighbouring-order
estimates using the same weights and sample identities. The usual step exponent
$1/(q+1)$ is only a small-step guide: $C_q(hD)$ itself changes with $h$. Sweep the
split test equation $u'=(D+\lambda)u$ over both $hD$ and $h\lambda$, and compare
PEC with accepted-field PECE and refined solutions before adopting the controller.
Retain bounded step ratios, startup and restart accounting, hard-kink landings and
history resets, and rollback on a rejected trial. No additional source evaluation
used for validation may be hidden in the delivered one-call claim.

**Dense observations.** The endpoint weights alone do not provide dense output.
For $0\leq\eta\leq1$, the same source polynomial gives

$$
 a_\mathrm{dense}(z_n+\eta h)=e^{\eta hD}a_n+h\sum_j w_j(x;\eta)N_j,\qquad
 w_j(x;\eta)=\sum_k c_{jk}k!\eta^{k+1}\varphi_{k+1}(\eta x).
$$

For varying $D$, replace the exponential kernel and upper integration limit in the
quadrature accordingly. This supplies an observation formula to qualify, not an
additional accepted history point. Check both endpoints, interior errors, strongly
damped forced bins and source-statistic identity; do not reuse the Lawson fallback
that drops a source increment in strongly damped bins without a separate argument.

### 5. Proposed interface

Only if the study's exit criterion is met. A method beside `Multistep`, selected the same
way, for example

```text
prop_gnlse(..., integrator=ExponentialMultistep())
```

with the same budget and observation semantics, independently qualified order and
dense formulas (§4), and the same refusals
(coupled flows, the `modified` noise model, fixed schedules). The name and whether it is a
separate class or a `Multistep` option are open (Q1).

### 6. Implementation strategies

- **The study (NumPy only).** Derive fixed-order ETD predictors first, then the
  proposed one-call PEC variant and its actual history recurrence. Use the scalar
  families in §4, and short constant-$D$ physical windows. The existing RDW case
  has a pressure gradient and belongs to the varying-$D$ extension, not a silently
  frozen approximation counted as the original benchmark.
- **The weights.** Compare a series/recurrence evaluator with independently refined
  contour or quadrature values. Both predictor and corrector weights, neighbouring
  orders and dense observations have costs. Repeated step ratios reuse polynomial
  coefficients; they do not reuse $\varphi(hD)$ unless $hD$ also repeats. Benchmark
  weight construction, memory traffic and peak storage rather than assuming their
  cost is comparable to a transform pair. Stream or reuse bounded buffers; declare
  every retained array in `memory_ledger()`.
- **The shared march.** Reuse the scheduler and source-evaluation hook where their
  semantics agree. Plain ETD stores untransported source samples and does not
  transport history. Per-bin weights replace scalar Adams weights, and ETD defects
  already live at the endpoint. Keep the current `Multistep` path unchanged.
- **Backends and contract.** `numpy_ref` is the oracle, then `numba_cpu`, then Kokkos
  CPU. A backend without the method refuses it by name. Stepper settings, protected
  driver hooks and stage bundles are all parts of the versioned seam: audit all
  three, bump `SCHEMA_VERSION` for each incompatible delivered change, and update
  the literal contract tests and changelog. The existing one-evaluation hook does
  not by itself establish compatibility of new weight arrays or fused passes.

### 7. Verification and validation

- **Weights:** recover ordinary Adams at $D=0$ and the two-node formula in §4;
  integrate polynomial forcing of every admitted degree exactly up to round-off.
  Cover complex arguments near zero, evaluator switches, large imaginary arguments,
  damping, unequal node spacings and coefficient cancellation. Tolerances follow
  floating-point summation/conditioning bounds and independent reference refinement.
- **Manufactured forcing:** use §4's exact forced and cross-phase solutions,
  including resonance, constant forcing, loss and mixtures of source phase rates.
  Separate exact-history one-step defects from propagated-history global errors.
  Establish three-level convergence at fixed $D$ for each delivered order, then
  sweep $hD$ independently to expose nonuniform accuracy. Report both field error
  and the estimated defect; a scalar phase limit must name its normalisation.
- **Variable $D$:** manufacture a known $a_*(z)$ and define
  $N(z)=a_*'(z)-D(z)a_*(z)$ for smooth diagonal $D(z)$ with analytic cumulative
  phase. Refine weights independently of step size, and test hard kinks and history
  restarts. Zero source must reproduce the cumulative linear flow to round-off
  relative to the supplied table; table error against the analytic phase is separate.
- **Lifecycle and stability:** startup, order and step changes, repeated rejection,
  fresh resume, exhausted budgets, masks, dense observations and batched members.
  Check the split stability problem in §4 and actual PEC/PECE histories; unchanged
  temporal-collar treatment still requires stability qualification.
- **Physical comparisons:** short windows representative of all four cases, using
  identical physics, masks, noise seeds, collar rates, output planes and initial
  fields for each method. Use about ten accepted steps or a fibre short enough to
  finish in seconds; if a developed broadband state is needed, use a documented
  saved input with provenance. Do not rerun full-length SEP 14 campaigns. A window
  must actually populate the weak band used in its score. Retain SEP 14's error
  metrics, but independently refine the reference for the new window. Report
  worst-plane complex-field and weak-band errors, rather than comparing at equal
  `rtol`. Record startup and mature calls separately, all rejected calls and
  diagnostic calls, attempts, orders, weight time, total wall time and memory.
- **Accepted study gate (Q6):** on the constant-$D$ supercontinuum window, at least
  1.5 times fewer total source calls than IP-DOPRI5 at matched field **and** weak-band
  error over at least two adjacent resolved accuracy targets; no increase in calls
  against SEP 13 on the canonical window, and no timing regression beyond measured
  timing spread.
  Require a total wall-time improvement against IP-DOPRI5 on the winning case,
  including weight construction and startup. Warm-up and repeated timings use the
  same backend/thread count and report median and spread. This makes the abstract's
  call-and-time claim testable; the maintainer confirmed the gate in Q6.
- **Conditional RDW study:** the original varying-pressure RDW cannot pass a
  constant-$D$ gate. If phase 1 fails but gives evidence for benefit there, report
  that evidence and obtain a scope decision before a bounded varying-$D$ study;
  the same accuracy, call and wall-time gate applies. Failure otherwise ends in a
  study report without a public integrator.
- **Delivery:** NumPy/Numba/Kokkos CPU parity for weights, one step, short runs,
  observations and call ledgers; deterministic reductions and declared memory.
  Follow the documentation and phased-SEP gates in `AGENTS.md`, including final
  full qualification before `Final`. Record the measured windows without
  extrapolating to complete-run cost.

### 8. Documentation

Extend [low-call stepping](../../../guide/numerics/adams.md) with the two source
frames, the qualified method and its refusals, error-control limitations and a short
worked example. Treat $\Omega_Dh$ as a diagnostic proxy, not a selection rule.
Update solver and stepper design documentation, public docstrings, the decision
table, the benchmark ledger and `CHANGELOG.md` when delivering the method.

### 9. Compatibility and impact

None until a method is added; it would be opt-in, with no default changed.

### 10. Alternatives considered

- **Higher-order Runge–Kutta** (DOP853): SEP 14 phase 6 found it 1.15–1.55 times more
  expensive than IP-DOPRI5 on the supercontinuum at practical accuracies.
- **Narrower windows:** can change the represented physics and therefore cannot
  establish a same-problem algorithmic speedup.
- **ETD Runge–Kutta:** a relevant independent ETD baseline if the study needs one;
  more source calls than a mature Adams step, but different stability and startup
  behaviour. Its delivery remains outside scope.
- **Calvo–Palencia multistep:** a distinct $kh$-interval scheme [6], with a
  different history recurrence and parasitic roots [4, §2.5]. The proposed first
  candidate remains the one-interval Adams family to reuse the existing march.
- **Multirate splitting of the detuned bins:** more machinery than a change of frame, and
  the bins are coupled through the source.

### 11. Phases

1. **Study:** the manufactured family and the exponential march on `numpy_ref` for
   constant $D$; report the plain method before any fixed per-bin-frame experiment,
   and assess the accepted study gate. No public method is promised by a negative result.
2. **Delivery on `numpy_ref`** (only if phase 1 passes): z-dependent operators, the
   interface, tests and the manual page.
3. **`numba_cpu`, then Kokkos CPU:** qualify each against the NumPy oracle and
   report the final cross-backend gate.

### 12. Open questions

- **Q1 — phase 2.** A separate class or a `Multistep` option, and its name; decide
  after the study establishes the supported order and refusal set.
- **Q2 — phase 1 report.** Is plain ETD sufficient? A fixed per-bin frame is an
  exploratory comparison only. An adaptive selection rule needs its own derivation,
  conditioning evidence and maintainer agreement before delivery.
- **Q3 — phase 2.** Qualify cumulative-table quadrature as the oracle and decide
  whether a controlled expansion is needed for cost. Purely diagonal $D(z)$ has
  no Magnus commutator error to remove.
- **Q4 — resolved in AI review.** Checked all six bibliographic records and the
  relevant derivations. Corrected reference 6's DOI to end in `-0`, not `-z`;
  distinguish online publication in 2005 from the 2006 journal volume.
- **Q5 — deferred, outside delivery scope.** An implicit–explicit temporal collar
  needs a separate coupled derivation. If $C=\mathcal F\,\mathrm{diag}(-\alpha_t/2)
  \mathcal F^{-1}$ is the collar on the spectral grid, an implicit ETD endpoint
  generally requires solving $(I-h W_{\mathrm{end}}(hD)C)a=b$. The frequency-diagonal
  weight and time-diagonal collar do not generally commute, so this is not a
  pointwise division in time. The displayed $C$ is the same-grid form; actual
  oversampling and projection must also enter any proposed solve. Even at $D=0$, high-order Adams–Moulton formulas are
  not unconditionally stable, and one PEC correction is not an implicit solve.
  No stability or order benefit is claimed until a specific pairing is analysed.
- **Q6 — resolved at acceptance, 2026-10-05.** John Travers confirmed §7's
  measurable call-and-time gate and the constant-$D$ supercontinuum study first.
  A varying-$D$ RDW study remains conditional on the findings and a subsequent
  scope decision, as specified in §7.

### 13. References

1. J. D. Lawson, "Generalized Runge–Kutta processes for stable systems with large
   Lipschitz constants," *SIAM Journal on Numerical Analysis* **4**, 372–380 (1967).
   [doi:10.1137/0704033](https://doi.org/10.1137/0704033)
2. J. Hult, "A fourth-order Runge–Kutta in the interaction picture method for simulating
   supercontinuum generation in optical fibers," *Journal of Lightwave Technology* **25**,
   3770–3775 (2007). [doi:10.1109/JLT.2007.909373](https://doi.org/10.1109/JLT.2007.909373)
3. S. M. Cox and P. C. Matthews, "Exponential time differencing for stiff systems,"
   *Journal of Computational Physics* **176**, 430–455 (2002). [doi:10.1006/jcph.2002.6995](https://doi.org/10.1006/jcph.2002.6995)
4. M. Hochbruck and A. Ostermann, "Exponential integrators," *Acta Numerica* **19**,
   209–286 (2010). [doi:10.1017/S0962492910000048](https://doi.org/10.1017/S0962492910000048)
5. A.-K. Kassam and L. N. Trefethen, "Fourth-order time-stepping for stiff PDEs,"
   *SIAM Journal on Scientific Computing* **26**, 1214–1233 (2005). [doi:10.1137/S1064827502410633](https://doi.org/10.1137/S1064827502410633)
6. M. P. Calvo and C. Palencia, "A class of explicit multistep exponential integrators
   for semilinear problems," *Numerische Mathematik* **102**, 367–381 (2006).
   [doi:10.1007/s00211-005-0627-0](https://doi.org/10.1007/s00211-005-0627-0).
   Published online 5 December 2005; journal volume dated 2006, also listed in
   [Calvo's publication record](https://hermite.uva.es/maripaz/RAhd.html)

## Part II — Implementation record

### Phase 1 — study and bounded optimisation completed, awaiting maintainer review

**Initial outcome, 2026-10-05: the tested prototype fails the accepted gate.** Its source-call
saving on the developed GNLSE continuum exceeds the required factor of 1.5, but
weight construction makes the complete NumPy run slower than IP-DOPRI5. The canonical
control also regresses in wall time. No public integrator, compiled backend or seam
change was delivered. The SEP remains **In progress** pending the requested human
review of this study; it is not marked Final or Rejected by this report.

**Approved scope decisions.** The maintainer requested a phase-1 report before any
public API or compiled-backend delivery. Inspection found that the canonical control
also uses varying pressure. The maintainer approved bounded cumulative-table
quadrature for that control during phase 1, while general varying-operator delivery
remains phase 2. The controls evaluate the original parameter expansion through its
cumulative scalar tables; their Chebyshev coefficients provide the quadrature's phase
variation bound. No new approximation to the pressure profile was introduced. A worked
example prepares the broadband inputs. These decisions supplement the accepted Part I.

#### Delivered work and derivations

The study lives under `benchmarks/sep15/`: `weights.py`, `march.py`, `cases.py`,
`harness.py`, `manufactured.py`, `diagnostics.py`, `interpolation.py`, `gate.py`,
`plot.py` and `verify.py`, with a
reproduction README and raw JSON/NPZ evidence in `data/`. The tests are
`tests/backends/test_sep15_study.py`. The worked example is
`docs/examples/supercontinuum/plot_05_stepper_study_inputs.py`; its generated page,
figure and downloads are committed under `docs/auto_examples/supercontinuum/`.

The study subclass reuses the Adams scheduler's landing, budget, rejection and
restart rules. It carries untransported source history, q/q+1 predictor/corrector
weights and a separately tagged predicted endpoint derivative for PEC. PECE explicitly
pays its second source call. The Adams--Bashforth comparator has fixed maximum order
five and no history-only free rejection. Variable-order PEC/PECE use the existing
hysteresis; no new order-selection rule is inferred from these experiments.

Moment coefficients are re-derived from the integral in Part I. Near zero, a Taylor
series with a next-term tail test feeds the half-interval doubling identity; for
argument magnitude above eight, division recurrence avoids repeated squaring. Orders
within one step reuse a single moment array. Independent augmented matrix exponentials
and adaptive quadrature check both sides of the join, unequal nodes, dense positions,
zero arguments, oscillation and damping. Strong damping never requires inverse flow
in the ETD source call. Varying-operator weights split at table edges, bound phase
variation per panel and compare Gauss rules before accepting coefficients. Their
estimated absolute uncertainty, including a roundoff allowance, enters the trial norm.
This is a quadrature estimate, not a proof of accepted-step error or translation error.

Prescribed forcing gives observed global orders 2 through 6 for q=1 through 5,
on three levels. Feedback tests with the actual predicted or corrected history also
recover the expected AB order q and PEC/PECE order q+1. The local estimator diagnostic
records the true accepted error separately from the predictor/corrector difference.
Fixed-frame sweeps confirm that slowly varying laboratory forcing favours ETD, while
forcing that rotates with the linear phase favours Lawson; an intermediate fixed frame
can help an intermediate source phase. This establishes no adaptive selection rule.
The stability sweep includes history modes of the explicit PEC recurrence and does
not confer implicit Adams--Moulton stability. The varying-rate regression recovers
only order two when endpoint-exact flow is paired with chord-based source weights.

#### Problems, windows and reference qualification

All timed runs use `numpy_ref`, configured for one host thread, on Darwin/arm64.
They preserve the full problem's grid, pressure, masks, noise setting and temporal
collar. Every method starts from the same field at the same physical position.

| case | full problem | measured interval (m) | input |
|---|---|---|---|
| GNLSE | 0.15 m silica benchmark, 400--2000 nm, 12.5 ps | 0.12--0.1202 | developed continuum |
| canonical | 4.2 m He capillary, 1 mJ, 90--1600 nm | 3.5--3.52 | developed continuum |
| canonical2 | same capillary, 2 mJ | 3.5--3.505 | developed continuum |
| reduced CtC | original 0.1 m H2 problem, two modes | 0--0.0002 | original launch |

Fresh parent inputs were prepared by the worked example with IP-DOPRI5 on
`numba_cpu`, `rtol=1e-8`, no noise. They are archived with checksums, grid/launch
signatures, factory and version provenance, and converted from saved true-power
amplitude back to the internal vacuum amplitude. The archived field defines the
window's initial-value problem. Parent accuracy is not substituted for window-reference
accuracy. The reduced CtC score selects populated weak spectral bins near the launch;
it is not evidence for a developed Raman comb or a full CtC run.

Each window reference was recomputed at `rtol=1e-11` and `3e-12` with IP-DOPRI5 on
NumPy. The worst discrepancies (complex-field, weak-band) were respectively:
GNLSE `(6.11e-12, 1.11e-10)`, canonical `(3.43e-13, 3.43e-12)`, canonical2
`(1.51e-13, 1.55e-13)` and reduced CtC `(1.17e-14, 3.48e-13)`, rounded upwards.
These are well below one tenth of the tightest gate target. This qualifies integration
relative to the fixed translated problem, not material-model or grid accuracy.

The harness retains SEP 14's complex-field and weak-bin metrics. Canonical fields
are scored in the analytic time-domain field inside +/-30 fs and in the 90--200 nm
UV band. No phase or time alignment is fitted. Other weak-bin masks use reference
power between 1e-9 and 1e-3 of the last-plane peak and remain fixed at every plane.
The weak-band amplitude fractions at the final plane are 1.16%, 3.49%, 17.62% and
3.11%, respectively: these scores are not on empty bands.

#### Gate and costs

For each target epsilon in `1e-3, 3e-4, 1e-4, 3e-5, 1e-5`, both errors must be
at most epsilon. Among measured eligible points, choose the fewest source calls,
breaking ties by median time. No curve interpolation or extrapolation is used.
The ladder is `rtol=1e-3` through `1e-6`, with intermediate factors of about three;
GNLSE extends to `1e-7` to resolve tighter comparison targets. Every point has one
warm-up and five timings, alternating forward/reverse method order. Translation and
fixture preparation are excluded; backend setup, startup, rejected trials, weights
and field collection are included. The table shows two adjacent resolved targets;
the other three fail the same timing conditions.

| target for both errors | GNLSE calls, RK / ETD | GNLSE median ms, RK / ETD | canonical calls, current / ETD | canonical median ms, current / ETD |
|---|---|---|---|---|
| 1e-4 | 56 / 10 | 56.9 / 110.0 | 8 / 8 | 19.0 / 132.1 |
| 3e-5 | 56 / 12 | 56.9 / 132.4 | 9 / 8 | 20.4 / 132.1 |

At 1e-4 the GNLSE timing interquartile ranges are `[56.5, 57.5]` ms for RK and
`[109.5, 110.2]` ms for ETD. The canonical ranges are `[18.8, 19.1]` ms for the
current method and `[131.6, 133.3]` ms for ETD. The regressions exceed the measured
spread. Thus **calls pass; time fails**. The numerical tolerances selected for each
method differ; the gate is evaluated at common error limits, never at common `rtol`.

The short windows include startup and may settle below order five. A separate GNLSE
sensitivity run over 0.12--0.121 m uses the same input and reference-refinement test.
At a common 1e-4 error limit, ETD uses 50 calls against RK's 301, but takes about
610 ms against about 307 ms. It remains slower after a longer mature march. Neither
window is extrapolated to a complete simulation.

At `rtol=1e-5` on the primary GNLSE window, about 121 ms of the 153 ms diagnostic
warm-up is coefficient work. Untimed traced allocation peaks are about 14.0 MB for
ETD, 10.3 MB for RK and 6.4 MB for current Multistep, excluding the prebuilt translation
and native-library allocations. ETD retains about 3.93 MB in history/dense/moment
arrays; its largest coefficient-plus-uncertainty result is 2.36 MB. These are distinct
measurements, not quantities to add. The raw records include every attempted step,
accepted order, ledger partition, coefficient time, memory result and timing sample.

AB and PECE comparisons, the doubled-energy control and the reduced CtC launch are
also recorded. They do not overturn the gate. In particular, on the reduced CtC
window at a 1e-4 error limit, ETD uses 8 calls and about 100 ms, against current
Multistep's 7 calls and about 70 ms. More expensive physics can make ETD faster than
RK without making it the best low-call method.

Separate tiny-window diagnostics charge seven reference-propagation calls and two
source samples per case. On the GNLSE sample, the globally weighted untransported
source changes by 0.0112 and the transported source by 0.00279. This first-difference
probe does **not** establish the proposed phase-mismatch explanation for the whole
problem: polynomial approximation depends on higher derivatives and on the weighted
weak bins. The measured call saving is evidence about this method/controller pair;
Omega_D alone is not a selection rule or a causal explanation.

The required recorded-source interpolation comparison is in `interpolation.json`.
It fits a cubic through four equally spaced past source samples and compares
midpoint/endpoint residuals in the physical source frame, using the same fields for
both fits. On the GNLSE endpoint, ETD/Lawson residuals are 0.0565/0.242 in-band and
0.249/3.30 in the weak band. The corresponding reduced-CtC in-band residuals are
0.00339/0.000447, favouring Lawson there. These higher-order residuals support the
forcing explanation on the selected GNLSE stencil even though its first difference
does not. They are source-fit errors, not integrated state errors or a universal
step-size limit. The four reference sampling runs cost 504, 168, 63 and 154 source
calls respectively; each pays nine further diagnostic source evaluations. All are
recorded separately and excluded from the timed gate.

#### Sources, checks, limitations and handoff

The derivations use the six primary papers checked in Part III, particularly the
phi-function integral, cancellation issue and distinction between one-interval Adams
and the Calvo--Palencia family. No coefficients or performance numbers were copied
from another implementation. No additional paper was needed for this phase.

Qualification includes independent weights, manufactured forcing/feedback order,
resonance and amplitude scaling, frozen-step stability, rejection rollback and
failure restart, hard-kink restart, resume, dense observation, exact linear decay,
source-call budgets and simultaneous-error gate selection. The report's recorded
validation is: `ruff check --fix .`, `ruff format .`, `ruff check .` (clean);
`basedpyright` (zero errors or warnings); targeted spelling and `git diff --check`
(clean); `pytest tests/backends/test_sep15_study.py tests/test_seps.py -n auto`
(205 passed, one expected process-SEP skip); and `poe docs` (clean). The worked
example ran in 76.15 s on `numba_cpu`, and its generated artifacts were refreshed.
`verify.py` reproduced all four cases' recorded ETD call counts and errors from the
regenerated inputs, linking their checksums to the final study sources.

The initial shared-environment docs run encountered an already stale Kokkos wheel
(contract 16 versus the repository's 18). Validation then used a separate environment
created offline from the unchanged lockfile, with NumPy/Numba and no Kokkos wheel.
The ordinary docs build used Numba's normal thread ceiling; the one-thread timing
setting would prevent its existing spatial tutorials from requesting larger pools.
The shared environment was left untouched. The final-delivery
full gate, slow/coverage lanes and Kokkos parity are not claimed: no production
backend was changed and this phase stops at the study review.

**Limitations:** frozen-step stability is not variable-step stability; the PEC defect
is not a certified bound on accepted error; table refinement checks integration of
the existing cumulative operator, not a tighter physical translation; fixed-frame
results are scalar only. There is no qualified coupled/separable flow, modified-noise,
fixed-schedule, split-collar or arbitrary-corrector implementation. Timing results
apply to these NumPy windows and include the prototype's weight cost.

**Q7 — resolved.** The maintainer authorised a bounded coefficient-cost optimisation
and a study with Numba, compared with the existing Numba RK scheme. The supplement
below records the result. This approval did not authorise public delivery.

**Q8 — evidence before further physics scope.** A developed varying-pressure RDW
study, general varying-operator delivery, adaptive per-bin frames and implicit
collars remain unqualified. No evidence here authorises extending to them.

**Handoff:** review this report and the stored work-precision data. Part I remains
frozen; production defaults, public APIs and schema are unchanged. Stop here as
requested, before phases 2 and 3.

### Phase 1 supplement — coefficient optimisation and Numba study

**Outcome, 2026-10-05:** optimisation recovers a measured GNLSE speedup over RK on
both backends, but the accepted gate still fails because the canonical control
regresses against current Multistep. The result qualifies the implementation cost
of the initial prototype; it does not support abandoning exponential Adams on the
basis of that cost alone. Part I and the error targets are unchanged.

The maintainer approved this bounded follow-up after the initial review and asked
that a compiled comparison use the existing Numba RK scheme. The study now selects
`numpy_ref` or `numba_cpu` explicitly. The compiled variant uses Numba's existing
source/FFT pipeline and a study-local compiled moment evaluator; its scheduler and
field combinations still use the Python/NumPy study march. It is not a public or
fully fused Numba ETD backend. Current Numba RK and Multistep retain their existing
compiled kernels, including RK's actual FSAL savings. No production module, public
API, dependency or kernel-contract field changed.

#### Algebra and implementation

`benchmarks/sep15/optimized.py` implements the following equivalent calculations:

- At the scaled argument $|y|\le1/2$, evaluate only the highest required Taylor
  series and use $\varphi_k(y)=1/k!+y\varphi_{k+1}(y)$ downwards. In exact
  arithmetic the highest moment's error is multiplied by $y$ at each recurrence;
  no small-argument subtraction or division is introduced. The original doubling
  identity and large-argument division branch remain. Independent matrix
  exponentials check both branches and all supported orders.
- Compute only the moments needed by the current order and eligible neighbouring
  orders. The compiled kernel scales each frequency independently, uses a scalar
  scratch vector, avoids complex division by real denominators, and has no
  `fastmath`, allocation, exception or parallel reduction inside it. Its output
  and scratch are allocated by the Python wrapper. Its undecorated body is also
  tested with JIT disabled.
- Form the small Lagrange polynomials with scalar products and combine them with
  the shared moments. Constant-operator coefficient uncertainty is identically
  zero in the existing study, so its zero arrays and reductions are omitted.
- For a varying operator, share monomial moments
  $M_k=\int_0^\eta\theta^k U(z_n+h\eta,z_n+h\theta)\,d\theta$ across all
  formulas in a trial. For polynomial coefficient matrix $C$, weights are $CM$.
  Batched Clenshaw evaluations use the original tables, grouped by interval, and
  cache the common endpoint. An expansion subtracts its scalar table values
  before combining its spectral terms, as the original operator does.
- Carry quadrature differences through $|C|\,|M^{fine}-M^{coarse}|$ and add the
  original coefficient-conditioning roundoff floor. Each requested formula must
  pass the same floor criterion. The phase-resolution bound, table-edge splits,
  8/16 Gauss comparison, bounded refinement and trial-norm allowance remain;
  there is no frozen/chord approximation to the varying operator. Shared moments
  are replaced on every attempted step and cleared on reset. Dense positions
  construct moments for their own upper limit.

`weights.py` remains the original implementation. `--optimized` selects the new
study path; `etd-original` retains the original comparator in the same NumPy timing
campaign. `optimization_report.py` regenerates the gates, original-field parity,
source/input/timing hashes and the work-precision figure. The evidence is under
`benchmarks/sep15/data/optimized/`; the original campaign is retained separately.

#### Backend-matched results

The windows, archived inputs, error metrics, tolerance ladders and five alternating
timing rounds are those of the initial study. All references remain refined NumPy
RK solutions. Each comparison below uses the same backend and a common simultaneous
field/weak error limit of **1e-4**, with measured-point selection as before.

| backend | GNLSE calls, RK / ETD | GNLSE median ms, RK / ETD | canonical calls, current / ETD | canonical median ms, current / ETD |
|---|---|---|---|---|
| NumPy | 56 / 10 | 60.51 / 41.81 | 8 / 8 | 19.75 / 25.03 |
| Numba | 49 / 10 | 41.00 / 27.97 | 8 / 8 | 14.38 / 20.25 |

The GNLSE IQRs are `[60.31, 61.39]` / `[41.24, 41.86]` ms on NumPy and
`[40.29, 41.23]` / `[27.63, 28.21]` ms on Numba. Canonical IQRs are
`[19.55, 19.93]` / `[24.95, 25.65]` ms on NumPy and `[14.31, 14.65]` /
`[20.16, 20.55]` ms on Numba. Both the speedup and the regression exceed the spread.
GNLSE's call and time conditions pass at four of five targets on NumPy and all five
on Numba. Canonical's time condition fails at every target on both. Applying the
same complete gate therefore returns **fail** on each backend.

In the same NumPy campaign the original ETD takes 113.55 ms on GNLSE and 128.64 ms
on canonical at that target. Optimisation reduces these totals by factors of 2.72
and 5.14 respectively. In a separate diagnostic warm-up at `rtol=1e-5`, NumPy GNLSE
coefficient time falls from 125.7 to 38.3 ms; the Numba evaluator takes 14.9 ms.
Canonical coefficient time falls from 113.2 to 4.5 ms on NumPy. These warm-up
partitions are diagnostic, not substituted for the repeated end-to-end medians.

The longer 0.12--0.121 m GNLSE window retains the improvement at the same error
limit: NumPy RK/ETD take 319.6/243.2 ms with 301/50 calls, and Numba RK/ETD take
198.8/147.6 ms with 259/50 calls. Current Multistep takes 245.3 ms on NumPy and
188.0 ms on Numba; the NumPy ETD/Multistep IQRs overlap. This remains a short-window
measurement, not a full-run prediction.

The doubled-energy control and reduced CtC launch also remain slower than current
Multistep. At 1e-4, canonical2 takes 14.21/17.19 ms (current/ETD) on NumPy and
12.99/15.56 ms on Numba; both use five calls. Reduced CtC takes 69.49/84.30 ms on
NumPy and 51.28/62.28 ms on Numba, using seven/eight calls. Expensive sources alone
do not guarantee that ETD beats the existing low-call method.

Allocation measurements remain separate from timings. At `rtol=1e-5`, NumPy GNLSE
traced peaks are 14.06 MB original and 13.99 MB optimised; Numba ETD is 16.44 MB.
Sharing canonical quadrature increases retained study arrays from 65.7 to 164.4 kB.
The optimisation primarily reduces repeated computation, not peak memory. Native
FFT/library allocations and prebuilt translation tables remain excluded.

#### Verification and remaining decision

At `rtol=1e-5`, all four physical windows preserve the original ETD source-call
counts and accepted-order sequences on both backends. The largest field/weak metric
difference against the original NumPy ETD fields is `1.35e-14`, rounded upwards.
The tests add independent moment checks at orders 1 through 8, argument-distribution
and invalid-input cases, unequal-node dense polynomial integration, shared
quadrature across a table knot, reuse accounting, NumPy/Numba resume parity and
three-level feedback convergence. Existing exact-flow, budget, predicted-history,
rejection, restart and hard-kink tests now also exercise the optimised march.

Validation: lint and formatting are clean; basedpyright reports zero errors and
warnings; the study and SEP tests pass (287 passed, one expected process-SEP skip);
all 74 optimisation tests also pass with `NUMBA_DISABLE_JIT=1`. Spelling, whitespace
and the ordinary documentation build are checked for this supplement. The gallery
source and input archives are unchanged, so no parent propagation was repeated.
The full production gate, slow/coverage lanes and Kokkos qualification remain outside
this study slice.

The first multimode Numba attempt exposed automatic pool selection exceeding the
one-thread environment ceiling. The harness now explicitly fixes Numba, FFTW,
transverse FFTW and BLAS pools to one; all compiled timings were repeated with that
setting. JIT warm-up is excluded; subsequent backend construction and plan setup
remain inside each timer. The isolated validation environment lacks the optional
Kokkos module, so its first type check reported missing imports in existing Kokkos
tests. The clean type check resolves imports with the shared development Python;
runtime tests and timings retain the isolated locked environment. No missing-import
check was suppressed and no shared environment was changed.

The six-paper audit and original derivations remain the source basis; the changes
above are algebraic rearrangements checked against independent references. No new
paper, physics approximation or controller rule was introduced. Further compilation
of the study's field arithmetic and optimisation of the remaining varying-table cost
are possible avenues, with no measured claim that they would eliminate the control
regression.

**Q9 — resolved as a study authorisation.** The maintainer authorised the detailed
canonical-cost analysis and bounded RDW/CtC comparisons below, including full
canonical propagation and worked-example preparation of a developed RDW snapshot.
This does not change the accepted delivery gate or authorise a public integrator.

### Phase 1 supplement — canonical cost, developed RDW and full-grid CtC

The full canonical comparison confirms that the regression persists after startup.
The developed M2/RDW window saves substantially more source calls but still loses
time constructing varying-operator weights. The following comparisons retain the
original campaigns; they do not replace or relax the accepted gate.

#### Scope and qualification

The full canonical problem is measured from 0 to 4.2 m, with observations at
2.1 and 4.2 m, by explicit maintainer authorisation. The RDW parent uses the
existing M2 parameters: four modes in a 1.5 m helium capillary, a 125 micrometre
radius, 0.8 bar entrance and choked vacuum exit, an 800 nm, 7.5 fs, 275 microjoule
pulse, a 400 fs window and 90 nm--4 micrometre spectral limits. The worked example
`docs/examples/processes/plot_07_stepper_rdw_input.py` runs the parent with Numba
RK at `rtol=1e-8`, archives the vacuum-normalised field at 1.2 m, and finishes in
45.03 s. The timed window is only 1.200--1.202 m. At its input, 100--150 nm contains
2.59% of the spectral norm squared, so the dispersive wave is appreciably populated.

CtC retains the original six-mode grid and all original responses, with a 4 ps
window and 32768/65536 coarse/fine time samples. It is only a launch window,
0--0.00017 m of the original 0.2 m hydrogen problem. This qualifies short-window
integration on the full grid, not the developed Raman comb. Neither a full CtC
propagation nor a new full GNLSE comparison is part of this campaign.

All candidates use one-thread Numba and its existing source/FFT kernels; two tighter
NumPy RK solves provide reference and refinement checks. Five alternating rounds
exclude translation, reference solves, parent preparation and compilation warm-up;
backend construction and FFT-plan setup remain timed. The selector chooses the
fewest actual calls satisfying all scored errors, with median time breaking ties.
RDW additionally requires the complex 100--150 nm field error below the target,
alongside its whole-field and weak-bin errors. Reference refinement must resolve
every scored metric to one tenth of the target. No timing curve is interpolated.
The frozen RDW parent defines the window initial-value problem; the errors below
do not certify propagation error accumulated before 1.2 m.

The raw reports, attempt ledgers, diagnostic profiles and provenance hashes live in
`benchmarks/sep15/data/canonical-analysis/`. `analysis_report.py` generates the
selections and work-precision plot; the study README gives reproduction commands.

#### Measured comparison

At simultaneous error limits of 1e-4 (including the additional RDW UV score):

| problem and measured extent | RK calls / median | current Multistep calls / median | ETD calls / median |
|---|---:|---:|---:|
| canonical, full 0--4.2 m | 1447 / 685.2 ms | 1334 / 774.3 ms | 1056 / 1663.0 ms |
| developed RDW, 1.200--1.202 m | 37 / 61.37 ms | 23 / 51.83 ms | 7 / 97.51 ms |
| CtC full grid, 0--0.170 mm | 49 / 3764.3 ms | 28 / 2227.9 ms | 17 / 1489.1 ms |

The corresponding RK/current/ETD IQRs in milliseconds are
[675.7, 688.7]/[773.7, 775.9]/[1639.9, 1665.2] for canonical,
[61.24, 61.79]/[50.04, 51.87]/[96.49, 97.62] for RDW, and
[3732.9, 3827.7]/[2209.3, 2263.2]/[1482.2, 1562.6] for CtC.
Every reported advantage or regression exceeds the interquartile spread.
Selected tolerances RK/current/ETD are 3e-6/3e-6/1e-5, 1e-6/3e-6/3e-5 and
3e-6/3e-5/3e-5 respectively; equal requested tolerances are not assumed to give
equal physical error.

Reference refinement discrepancies in the worst scored metric are 1.48e-9,
2.78e-11 and 4.88e-12 respectively. The selected RDW ETD point has whole-field
error 6.41e-6, weak-bin error 5.39e-5 and UV error 1.69e-6. The 100--150 nm band
contains 2.65% of the final spectral norm squared. The RDW ladder does not resolve
RK/current Multistep at the 1e-5 target; comparisons at that target are left open.

CtC ETD wins across all five sampled targets, from 1e-3 to 1e-5, against both
comparators, with separated IQRs. At 1e-4 it saves 39% of current Multistep's calls
and 33% of its time, and takes 60% less time than RK. Its linear operator is
constant, so it uses the compiled moment evaluator. Its result is consistent with
the earlier constant-operator GNLSE improvement, while its expensive full-grid
source makes call savings more valuable than in the reduced launch control.
This does not qualify a developed CtC spectrum or justify a full-run extrapolation.

#### Why canonical remains slower

At simultaneous field/weak error limits of 1e-4, the full-run ETD candidate takes
1056 calls and 1.663 s, compared with current Multistep's 1334 calls and 0.774 s.
The 21% call saving does not pay for the extra work in each step. ETD accepts
1053 steps: 652 at order five and 386 at order four. This is not a startup-only
measurement. There are only two rejected steps, and the maximum normalised
coefficient-uncertainty contribution is 7.6e-10; neither rejection churn nor the
coefficient-error term explains the slowdown at this point.

The original 3.5--3.52 m window permits controlled ablations at `rtol=1e-4`.
All variants use eight actual source calls. Seven alternating uninstrumented rounds
produce these medians:

| backend | current Multistep | current, fusion disabled | ETD | ETD, exact weights replayed |
|---|---:|---:|---:|---:|
| NumPy | 19.010 ms | 19.054 ms | 24.392 ms | 19.990 ms |
| Numba | 13.986 ms | 14.262 ms | 19.633 ms | 15.375 ms |

Replay supplies previously computed weights from precisely the same trajectory,
checks the resulting field and calls, and retains every real source evaluation.
It is a diagnostic counterfactual, not a usable integrator or a gate candidate.
On Numba it removes about 4.26 ms of the 5.65 ms gap; a residual 1.39 ms remains
with free weights. Disabling current Multistep's fusion adds only 0.28 ms, so
compiling the study's field combinations alone cannot explain away the regression.

The remaining source overhead has a concrete cause. Current Numba Multistep forms
its propagator and holds its endpoint stage bundle for the source call. ETD obtains
an untransported source by calling the same backend with equal anchor and endpoint.
That misses the held-bundle path, repeats anchor work and constructs an identity
propagator. The short diagnostic records sixteen anchor calls for ETD versus nine
for current Multistep, despite both making eight material-source evaluations.
The corresponding instrumented source times are 4.77 versus 3.53 ms. This is extra
work around the same response pipeline, not more nonlinear physics per evaluation.

The earlier Numba acceleration evaluates constant-operator moments in compiled
loops. These canonical and RDW operators vary with position, so their moments still
use Python/NumPy quadrature over the original cumulative tables. Every trial shares
moments across predictor/corrector/orders, but still evaluates complex flows at
coarse and fine quadrature nodes. A whole-grid rate bound chooses the panels.
This explains why the constant-operator GNLSE improvement does not automatically
carry over to these cases.

#### Separate profiles and practical headroom

Three separately instrumented runs at each selected 1e-4 point give the following
median partitions. They are diagnostic timings, not replacements for the
uninstrumented campaign. Medians of individual partitions need not sum to the
median total.

| case / method | setup | source path | weight preparation + combination | other | total |
|---|---:|---:|---:|---:|---:|
| full canonical / current | 10 ms | 428 ms | 0 | 336 ms | 775 ms |
| full canonical / ETD | 11 ms | 573 ms | 847 ms | 405 ms | 1842 ms |
| RDW / current | 7.2 ms | 29.9 ms | 0 | 14.9 ms | 52.0 ms |
| RDW / ETD | 7.4 ms | 11.5 ms | 70.2 ms | 6.7 ms | 95.2 ms |
| CtC / current | 6 ms | 2088 ms | 0 | 135 ms | 2229 ms |
| CtC / ETD | 6 ms | 1269 ms | 117 ms | 92 ms | 1483 ms |

Full canonical spends about 46% of its instrumented ETD total on weights.
Even subtracting that entire cost leaves roughly 1.0 s versus current's 0.78 s:
a varying-moment kernel alone is unlikely to remove its regression. The ETD source
path takes about 0.54 ms per evaluation versus current's 0.32 ms, so eliminating
redundant stage work is material at this scale. Its 1055 moment preparations visit
2936 quadrature panels across coarse/fine/refinement passes, with at most sixteen
panels in any one pass. Flow-table evaluation and exponentiation account for
432 ms within the 498 ms quadrature total; these nested timings are already
included in the weight column.

RDW spends about 74% of its ETD total on weights. Its six preparations visit
64 panels over nineteen quadrature passes, with at most eight panels per pass.
About 58 ms of the 62 ms quadrature time is in flow-table evaluation/exponentiation.
Subsequent refinement inside weight combinations accounts for much of that work;
the preparation timer alone understates the coefficient cost. Exact weight replay
at the same seven-call trajectory takes 24.28 ms median (IQR 23.69--24.56 ms),
with bitwise-identical fields. Compared with the current 51.83 ms median, this
counterfactual leaves about 27.6 ms for real weight construction: an approximately
2.6-fold reduction of the measured 70.2 ms cost could make this window competitive,
if the other costs stay similar. That is a measured cost budget, not a prediction
that compilation will achieve it.

CtC has no table-quadrature passes. Its source path costs about 75 ms per evaluation
for both methods and dominates their totals. Saving eleven such evaluations easily
pays for ETD's 117 ms of moment/weight work. This explains its improvement without
attributing it merely to having more modes or a larger grid.

#### Remaining optimisation path

The measurements justify a focused implementation sequence: compile and fuse the
varying-table flow/moment loop while preserving the actual cumulative operator and
refinement check; reuse endpoint scalar tables and avoid identity-propagator work
for laboratory-frame sources; then fuse the field combinations and error norms.
The short-window replay quantifies removable weight cost but is not evidence that
any real implementation has removed it. Full canonical still needs a measured
combined improvement; no blanket speedup is claimed. Freezing the operator or
changing the controller would be a numerical change requiring separate review.

#### Verification and handoff

The phase-slice suite passes 295 tests with one expected process-SEP skip. It
covers the original study, optimised weights, lifecycle behaviour, new UV selection,
weight replay and proposal consistency. After extracting argument/window setup
from the enlarged harness, all eight analysis tests pass again. Ruff lint and
format checks, the pinned type checker, changed-file spelling, whitespace checks
and the warning-as-error documentation build pass. The type checker resolves the
optional Kokkos imports using the existing development Python; runtime checks and
timings use the isolated locked NumPy/Numba environment. No Kokkos, slow-lane or
full-delivery qualification is claimed for this research-only slice.

The first full-run attempt exposed a duplicate final save plane in the harness;
endpoint deduplication fixes it and is regression-tested with all three methods.
The interrupted CtC timing attempt was discarded and repeated as a complete
five-round campaign. The RDW worked example and its generated output were executed
and retained together; the final plot was visually checked. Source, input and
raw-evidence hashes are recorded separately from the historical campaigns.

The six-paper audit and previously verified derivations remain the source basis.
This follow-up adds measured diagnostics, observation metrics and case coverage;
it introduces no new physical coefficient, integration formula or approximation.
The full implementation gate remains reserved for any later production delivery.

**Q10 — delivery remains conditional.** Review the remaining varying-operator and
source-path optimisations against these measurements. The accepted canonical
no-regression condition remains unmet. The developed RDW window supplies the
bounded evidence requested in Q8, but general varying-operator delivery, adaptive
per-bin frames and implicit temporal collars remain unqualified. Part I, production
defaults, the public API and kernel schema are unchanged.

### Phase 1 supplement — compiled tables and hybrid varying-flow quadrature

The maintainer requested further optimisation, with particular attention to the
full canonical control and to expensive source evaluations in SEP 12. This remains
a research implementation under `benchmarks/sep15/`; no public selection, backend
contract, physical default or production driver changes.

#### Implementation and independent checks

`accelerated.py` compiles scalar Chebyshev evaluation, interpolation coefficients
and candidate combinations without fastmath. The stored scalar tables are shared
and their coefficients are unchanged. The harness applies this table optimisation
to RK and current Multistep too. ETD additionally bypasses construction of identity
transport for a laboratory-frame source, while retaining the production response
pipeline. A per-march plan reuses the scalar-expansion layout and derivative bounds.

The cumulative scalar phase difference is evaluated through a difference form of
Clenshaw recursion within an interval, rather than by subtracting two almost equal
cumulative values. With evaluation coordinates $a,b$, recurrence differences obey
$d_0^{new}=-d_1$ and
$d_1^{new}=d_0+2a d_1+2(a-b)c_1(b)$; the final difference is
$d_0+a d_1+(a-b)c_1(b)$. The code forms $a-b$ from physical distances. Across
intervals it retains the actual stored endpoint offset, including a possible jump
at a knot. A manufactured cubic with a large additive constant independently
checks preservation of tiny increments; an affine two-interval test checks the
stored jump. This changes rounding, not the represented cumulative operator.

An entirely compiled quadrature loop was measured first and retained as raw
ablation evidence. Scalar complex exponentials in that loop were slower on the
large RDW grid. The retained hybrid evaluates the scalar phase differences in
compiled code, then uses panel-sized vector exponentials and matrix products.
It retains the same Gauss rules, panel refinement criterion, coefficient error
floor and interpolation/controller policy. It also lists retained coefficient
error storage in the study memory ledger. The independent quadrature tests cover
loss, oscillation, interior polynomial knots and partial-step weights.

#### Mac results and what is matched

Five interleaved warmed rounds, one thread, with separate allocation probes give:

| Case and window | Method | Actual source calls | Median time |
|---|---|---:|---:|
| Canonical, full 0--4.2 m | RK with compiled scalar tables | 1447 | 0.4980 s |
| Same | Current Multistep with compiled scalar tables | 1334 | 0.5552 s |
| Same | Previous ETD weights/source path, compiled scalar tables | 1056 | 1.2236 s |
| Same | Hybrid ETD | 1056 | 0.8097 s |
| Developed RDW, 1.200--1.202 m | RK with compiled scalar tables | 37 | 51.26 ms |
| Same | Current Multistep with compiled scalar tables | 23 | 40.50 ms |
| Same | Previous ETD weights/source path, compiled scalar tables | 7 | 76.25 ms |
| Same | Hybrid ETD | 7 | 69.10 ms |

Every selected row meets the same $10^{-4}$ ceiling on every required measured
error; these are not equal requested tolerances or exactly equal attained errors.
The sampled ladder selects the fewest actual calls satisfying the ceiling, with
median time breaking ties. Canonical uses requested tolerances $3\times10^{-6}$
for RK/Multistep and $10^{-5}$ for ETD; RDW uses $10^{-6}$, $3\times10^{-6}$ and
$3\times10^{-5}$ respectively. The raw files retain all attained errors,
reference refinements, five samples, IQRs and attempts. No interpolation between
sampled tolerances is used. The hybrid cuts canonical time by 33.8% relative to
the ETD control with the shared table improvement, but remains 45.8% slower than
current Multistep. The canonical no-regression delivery condition remains unmet.

Separate interleaved same-tolerance controls compare the full pre-existing
optimised path with all new acceleration enabled. They give canonical ETD
1.476 versus 0.792 s, RDW ETD 76.75 versus 69.93 ms, and full-grid CtC launch
ETD 1.365 versus 1.402 s. The CtC window is only 0--0.170 mm; its constant
operator already uses compiled moments and its source dominates. The new path
provides no measured CtC speedup. It preserves 17 ETD calls versus 28 for current
Multistep at the checked tolerance. The canonical field/UV differences against
the old path are $6.81\times10^{-13}$/$1.15\times10^{-11}$; RDW differences are
below $1.1\times10^{-13}$ on those metrics. CtC and current Multistep outputs are
bitwise identical in these controls. They test implementation parity, separately
from comparisons against the refined RK references.

A separate five-round canonical comparison on one pinned Xeon Gold 6240 core
confirms the direction of the improvement. At the same $10^{-4}$ joint ceiling,
RK/current Multistep/hybrid ETD take 1.757/2.029/2.758 s, using the same
1447/1334/1056 source calls. The previous ETD path with compiled scalar tables
takes 4.155 s. Hybrid ETD is 33.6% faster than that control and 35.9% slower than
current Multistep. The final Xeon campaign uses the two tolerances needed for
this comparison; the earlier compiled-quadrature ablation has the full ladder.
The core's simultaneous-multithreading sibling was idle during the recorded
sample; other cores had unrelated work. Both hosts use SciPy 1.18.1 and Numba
0.67.0, but NumPy is 2.5.2 on an Apple M5 Max and 2.5.3 on a Xeon Gold 6240. These are within-host
comparisons, not a controlled comparison of processor architectures.

#### Verification and handoff

The phase-slice suite passes 321 tests with one expected process-SEP skip;
26 new numerical/behaviour checks also pass with JIT disabled. Focused tests
are repeated after typing changes. Ruff, the pinned type checker, changed-file
spelling, whitespace and warning-as-error documentation checks qualify this
research slice. The manufactured test fixture initially violated immutable
array and common-span contracts; fixing the fixture preserved those production
checks. A documentation anchor was corrected after the strict build caught it.
The type checker uses the existing development environment to resolve optional
Kokkos imports; all runtime checks and Mac timings use the isolated environment.
No production-backend, Kokkos or full-delivery qualification is claimed.

**Q11 — next numerical experiments.** The maintainer authorised this continued
optimisation in response to Q10. This slice improves implementation cost but
does not satisfy the canonical delivery gate. The literature assessment below
identifies further options requiring a concrete numerical plan and review;
it does not authorise replacing SEP 12's independently selected stepper.

#### Why SEP 12 recorded about 140 calls

[SEP 12 section 6.4](sep-0012-tdse-nonlinear-response.md)
records **140 source calls**, not 140 accepted steps, for an earlier Kerr/plasma
proxy with `rtol=1e-2`, a split temporal collar, a 90 nm grid edge and 50 saved
planes. Its measured main-field error was 0.013 and its UV error 0.32. It is not
an actual TDSE result. The present canonical control uses the in-stage collar,
observations at 2.1 and 4.2 m, and imposes much tighter simultaneous field/UV
accuracy. The current TDSE configuration also has a 120 nm grid edge. Tolerance
is a major difference, but it is not the only one.

The new canonical ladder spans `rtol=1e-2` through `1e-7`. At the same requested
`rtol=1e-2`, current Multistep uses 123 source calls and ETD 157. At a common
1% **main-field-only** ceiling, the sampled selections are 210 and 334 calls;
at a common 1% **joint field/UV** ceiling they are 412 and 403. At the stricter
$10^{-4}$ joint ceiling they are 1334 and 1056, a 20.8% reduction. The RDW
reduction at that joint ceiling is 69.6%. ETD does not reduce calls universally,
and the loose main-field selections do not satisfy the joint delivery gate.
The precise frontier between ladder points remains unsampled.

For SEP 12, extra inexpensive coefficient work can be acceptable if it saves
expensive material evaluations at the required physical accuracy. These proxy
measurements establish that such savings occur in some regimes; they do not
establish the TDSE trajectory, its call reduction or its run time. No actual
TDSE method comparison, full propagation or change to SEP 12 was performed.

Raw campaigns, the original compiled-quadrature ablation, same-tolerance parity
controls, source hashes and a work/precision plot are under
`benchmarks/sep15/data/accelerated/`. `acceleration_report.py` regenerates the
joint and main-field-only selection tables. The original phase-1 and previous
cost-analysis campaigns remain separate historical evidence.

### Literature follow-up — additional optimisation avenues

The maintainer supplied six further papers during the optimisation study. Their
relevant derivations, implementation sections and stated limitations were read;
this assessment distinguishes the papers' results from proposed applications to
solprop. It does not amend Part I or qualify a new numerical method.

**Hochbruck and Ostermann, “Exponential multistep methods of Adams-type” (2011),
[DOI 10.1007/s10543-011-0332-6](https://doi.org/10.1007/s10543-011-0332-6): highest
priority for the current Adams formulation.** Section 2, equations (2.3)--(2.5),
writes the update as exponential Euler plus backward-difference corrections.
For smooth sources the higher differences decrease with powers of the step.
Section 5.2 uses this fact to reduce the cost of matrix-function actions; section
5.3 interprets the same method as a cheap inner differential equation driven by
an interpolated expensive source, with one fresh source evaluation per outer
step. Section 5.1 explicitly favours diagonalisation when available. Thus Krylov
iteration is not automatically advantageous for our already diagonal dispersion.
A useful adaptation is a Newton divided-difference source representation on our
nonuniform history, followed by quadrature whose contribution is bounded in the
existing solution-error norm. This could avoid resolving every coefficient to a
source-independent roundoff threshold. It needs an independently checked error
bound and variable-step tests: the paper explicitly leaves variable step/order
implementation outside its scope. Its constant-operator analysis does not itself
prove our varying-operator PEC march.

**Demirel, Niegemann, Busch and Hochbruck, “Efficient multiple time-stepping
algorithms of higher order” (2015),
[DOI 10.1016/j.jcp.2015.01.018](https://doi.org/10.1016/j.jcp.2015.01.018): highest
priority for expensive SEP 12 sources.** Sections 2--4 separate a cheap stiff
part from an expensive non-stiff part, interpolate the latter on an outer
multistep grid, and integrate the cheap part on inner steps. This directly
motivates advancing dispersion, known linear loss and possibly the temporal
collar without another TDSE evaluation at every inner stage. It may also bypass
explicit construction of the varying exponential weights. These are proposed
solprop applications, not results established by the paper. The corrector
algorithm shown is PECE, with two fresh expensive evaluations per outer step;
it must not be described as a one-call replacement for our PEC march. The paper
also varies history length independently of formal order to enlarge stability
regions, which suggests another route to fewer outer steps if stability is the
active limitation. Its concluding section leaves adaptive error estimation for
future work. Neither its fixed-step stability contours nor its measured speedups
establish our variable-step, dispersive, TDSE result.

**Caliari, Cassini, Einkemmer and Ostermann, “Accelerating exponential integrators
to efficiently solve semilinear advection-diffusion-reaction equations” (2024),
[DOI 10.1137/23M1562056](https://doi.org/10.1137/23M1562056): a direct alternative
to expensive varying-flow weights.** The supplied accepted manuscript's
introduction and sections 2--3 choose a cheaply exponentiated operator and move
the exact residual into the explicit term. For solprop a candidate identity is

```{math}
\partial_z u = D_0 u + \left[N(z,u)+(D(z)-D_0)u\right].
```

Keeping the residual preserves the differential equation; it is different from
simply discarding variation in the operator. Constant-operator moments would
then suffice, but the remainder can reintroduce fast variation, stiffness and
source-history error. Choosing or refreshing the reference operator also needs
consistent treatment of stored history. The paper proves stability statements
for a diffusion model with negative-real spectrum; its remark 2.9 explicitly
notes that the new stabilised Lawson schemes are not A-stable. No general
stability conclusion follows for our predominantly imaginary dispersion or for
the noncommuting temporal collar. This is an experimental numerical alternative,
requiring its own matched-error and source-count comparison.

**Carroll and O'Callaghan, “Exponential almost Runge-Kutta methods for semilinear
problems” (2013),
[DOI 10.1007/s10543-012-0416-y](https://doi.org/10.1007/s10543-012-0416-y): useful
second-line method design.** Sections 4--7 combine a small number of new stages
with past nonlinear values to approximate derivatives and organise combinations
of phi-function actions. It offers alternatives if our Adams predictor or
stability region limits useful step size. Its cost comparisons emphasise
matrix-function actions, whereas SEP 12 is dominated by fresh source calls.
A two-stage method therefore needs enough improvement in admissible step size
to compensate for its source cost. The published numerical experiments use
fixed steps; the conclusion describes adaptive implementations and cheap error
estimation as subsequent work. This paper is not a ready-made adaptive driver.

**Ju, Zhang, Zhu and Du, “Fast Explicit Integration Factor Methods for Semilinear
Parabolic Equations” (2015),
[DOI 10.1007/s10915-014-9862-9](https://doi.org/10.1007/s10915-014-9862-9): useful
supporting implementation and splitting ideas.** Sections 2 and 4 combine
analytic exponential integration of interpolated sources, transform-based
diagonalisation and an added/subtracted damping term. Solprop already has
spectral diagonalisation and analytic constant-operator moments. Their reusable
weights assume uniform steps and a constant operator, so they do not remove our
adaptive varying-pressure cost directly. The scalar damping construction is
worth considering only after establishing the relevant dispersive stability
conditions; Allen--Cahn energy stability is not a propagation guarantee.

**Mukam and Tambue, “A note on exponential Rosenbrock--Euler method for the finite
element discretization of a semilinear parabolic partial differential equation”
(2018), [DOI 10.1016/j.camwa.2018.07.025](https://doi.org/10.1016/j.camwa.2018.07.025):
useful background, lower immediate priority.** Its main contribution is a
space/time convergence analysis, including nonsmooth data, of a locally
linearised Rosenbrock method. The smooth-data time order is two. Incorporating
the nonlinear Jacobian could reduce explicit stiffness, but a full material or
TDSE Jacobian action has not been shown cheap; finite-difference actions could
add costly source evaluations and destroy the diagonal-operator advantage.
The paper supplies neither a cheaper varying-D coefficient construction nor
a demonstrated low-call TDSE method.

The immediate research priorities suggested by this reading are: derive and
validate source-weighted quadrature error control in a divided-difference basis;
compare cheap inner stepping with exact weight construction; then test an
explicitly compensated reference operator. These proposals need review of the
numerical choices before implementation. Kernel acceleration alone does not
exhaust the optimisation paths. All six supplied documents were readable;
no additional paper is required to make this assessment.

### Supersession report — production delivery transferred to SEP 16

**2026-10-05, at the maintainer's request.** This SEP is **Superseded**, not Final.
The dated reports above retain the sequence of the study, including the initial
negative result; statements of then-current status are historical. No public
integrator or complete NumPy/Numba/Kokkos production qualification was delivered.

The subsequent compensated-reference experiments improved calls and time at a common
measured field/weak-band error ceiling in the full canonical, developed M2/RDW and
short CtC launch comparisons. The retained measurements, reference refinements,
limitations and experimental checks are in the
[benchmark ledger](../../benchmarks.md)
and the method discussion (recorded in the development archive).
These findings do not establish modified-noise, developed-CtC, general-spatial or
actual-TDSE qualification. The loose canonical sub-140-call result is a distinct
accuracy comparison, not the strict-error result.

[SEP 16](sep-0016-unified-multistep.md) inherits the remaining production obligations:
a qualified shared formulation and public interface, varying-operator support,
backend parity, lifecycle/dense-output/error-control qualification and documentation.
It broadens delivery to the route/noise/adiabatic-taper matrix, memory and runtime
optimisation, work–precision gallery and solver guidance. General varying noncommuting
coupled flows remain explicit roadmap work. The replacement is **Draft** and requires
its own review; this supersession is not acceptance of it.

The study implementations, tests, generated inputs and raw results remain available
as research witnesses at Git revision `63a5e04` on
`feature/sep-0015-exponential-adams-study`. Paths to study artifacts in this record
refer to that revision; the documentation merge does not deliver the experimental code.
Implementation of SEP 16 starts from main after SEP 12 integration, auditing that
baseline rather than substituting this experimental branch for it. No new propagation
or qualification campaign was run to change this status.

## Part III — History

- **2026-10-05:** Drafted from SEP 14 phase 2's diagnosis of where the interaction-picture
  Adams march loses.

- **2026-10-05 — AI review, GPT-6 Astra:** reviewed all of Part I against SEP 13,
  SEP 14's stored diagnosis and the current Adams march. Revised the source-frame
  argument, derived the ETD moments, fixed-frame and dense weights, distinguished
  predictor error from accepted PEC error, corrected the commuting-$D$ and implicit
  collar arguments, and made the study/delivery gates and seam obligations explicit.
  Review recommendation: proceed as a bounded study after human review and Q6;
  delivery remains conditional on evidence. Status remains Draft.

  **Sources checked:** Lawson §2, Hult §II.D, Cox–Matthews §2.1.1,
  Hochbruck–Ostermann §2.5 and its distinction between smooth and oscillatory
  problems, Kassam–Trefethen §2, and Calvo–Palencia §§2–5, using the available
  article texts and validated bibliographic records. The extraction reports have
  complete page coverage. Lawson's DOI was registry-matched because its PDF prints
  no usable DOI. Cox–Matthews has extraction warnings on PDF pages 17 and 23,
  Kassam–Trefethen an unmatched table caption on page 3, and Calvo–Palencia an
  unmapped glyph on page 10; no disputed table or glyph was used for coefficients.
  No paper's performance numbers were adopted as solprop evidence. The variable-step
  PEC controller, adaptive per-bin frames and varying-$D$ performance remain to be
  established by the proposed studies; this review did not run those studies or
  repeat SEP 14's propagation campaign.

  **Numerical cross-check of the revision:** an independent matrix-exponential
  evaluation of the scalar $\varphi$ moments agreed with adaptive quadrature in
  756 endpoint, dense and fixed-frame weight checks (up to six nodes, unequal
  spacings, zero/small complex arguments, oscillation and damping). The largest
  dense-weight discrepancy divided by the larger of one and the polynomial
  coefficient 1-norm was $2.9\times10^{-16}$ (rounded up). The forced solution
  agreed with direct quadrature, including resonance. For a dimensionless scalar state with
  $D(z)=z/L^2$, $L=1$ m and constant source $1/L$ on $[0,L]$, exact endpoint
  flow with chord-based source weights gave observed global orders 1.9996 and 1.9999 on 10/20/40 steps.
  These are algebra checks, not qualification of an adaptive ETD implementation.

- **2026-10-05 — Human review and acceptance, John Travers:** read and approved
  the revised proposal in the SEP 15 review discussion, completing the required
  human and AI reviews. Accepted the bounded study and its conditional delivery
  phases, resolving Q6 with the constant-$D$ study first and the call-and-time
  gate in §7. Status changed to Accepted; Part I is frozen. Implementation has
  not started.

- **2026-10-05 — Superseded, John Travers:** requested a separate SEP 16 draft and
  review for the broader production scope. Archived this study without claiming Final;
  transferred outstanding delivery to SEP 16 and retained its research evidence.
