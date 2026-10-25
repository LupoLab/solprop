# SEP 35 — A carrier frame for the exponential Adams history

| | |
|---|---|
| **SEP** | 35 |
| **Title** | A carrier frame for the exponential Adams history |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-25) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-25 |
| **Resolution** | — |
| **Requires** | [SEP 16](archive/sep-0016-unified-multistep.md) (the default multistep stepper and its exponential formulation) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

The default `Multistep()` integrates the nonlinear source in SEP 16's exponential
formulation: the linear flow is exact and a polynomial through the untransported source
samples is integrated against it. On a grid whose operator keeps $\beta_0$ — every field
grid — the source rotates along $z$ with the carrier at
$\kappa = \beta(\omega_0) - \beta_1\omega_0$, and the polynomial has to follow that
rotation. In dense molecular gases $\kappa$ reaches thousands of radians per metre and the
default stepper takes several times more source evaluations than IP-DOPRI5 or the
interaction formulation (2669 against 798 and 1401 on a reduced hydrogen
comb-to-continuum case). This SEP proposes to interpolate, inside each step, the source
with a scalar carrier phase removed and to integrate the remainder against the
correspondingly shifted $\varphi$-functions. The step stays exact in its linear part for
any shift, the state, the stored history, the carrier-envelope phase and resume are
untouched, and the error estimate and order choice work unchanged. On the reference
backend it brings the reduced comb-to-continuum case from 2669 to 477 evaluations, and to
the cost of the same physics on an envelope grid, while costing nothing where the
exponential formulation already led. Success is judged on work–precision against
converged references on the benchmark suite's field-grid cases, with no regression on the
helium capillaries that motivated SEP 16's default.

### 2. Motivation and scope

**The cost.** SEP 16 chose the exponential formulation as the default on campaigns
dominated by low-pressure helium capillaries, where it is best. A later campaign on dense
molecular gases found it 4 to 10 times more expensive than the alternatives on field
grids. The cause is not stability — lowering the order makes it worse (order 3 and 2
took about 2 and 4 times more steps than order 5) — and not any particular response:
Kerr alone reproduces it, and the same physics on an envelope grid, where $\beta_0$ is
subtracted, takes a tenth of the steps. The source history the predictor extrapolates
rotates at $\kappa$, and at order five the controller has to keep $h\kappa$ small enough
for a polynomial to follow it.

**In scope.**

- The exponential formulation's predictor, corrector, defect estimates and interpolated
  planes, on every grid whose operator keeps a carrier phase (field grids; envelope grids
  with a third-harmonic term, §9).
- The diagonal and expansion/table operators, the varying-operator compensation, the
  in-stage collar and the coupled sections SEP 16 supports.
- `numpy_ref`, `numba_cpu` and the native backend, CPU first, devices after.
- The default: whether the frame is on by default, and with which choice of shift (§12).

**Out of scope.**

- The interaction formulation and the Runge–Kutta marches, which already transport the
  linear phase.
- The weak-band blind spot of the step-size norm in general (the manual's
  [limitations](../../guide/numerics/adams.md#limitations)); §7 records where this
  proposal meets it.
- The physics of the envelope grid's third-harmonic route. While this SEP was drafted, a
  comparison against the field grid indicated that the envelope route generates far less
  third harmonic than the field grid for the same short propagation; that is a separate
  question, to be settled on its own, and nothing here depends on its answer.
- Raising the order or changing the controller.

### 3. Context

The exponential formulation is described in
[the Adams methods](../../guide/numerics/adams.md#the-exponential-formulation); its
arithmetic is `ExponentialKernels` and `HostHistory.exponential_trial` in
`solprop/backends/multistep.py`, with compiled counterparts in
`solprop/backends/numba_cpu/multistep.py` and the native backend's Adams engine. The
operator and which frame terms each grid removes are set out in
[the linear operator](../../guide/numerics/linear-operator.md#which-terms-the-frame-removes):
a field grid keeps $\beta_0$ because a frequency-independent phase on its half-spectrum
is a change of carrier-envelope phase, not a global phase.
[SEP 15](archive/sep-0015-exponential-adams.md) introduced exponential Adams stepping and
SEP 16 unified it with the interaction formulation and made it the default.

### 4. Theory

**The step.** With a diagonal operator $D$ and source $F$, the exact step from $z_n$ is

$$u_{n+1} = e^{hD}u_n + \int_0^h e^{(h-s)D}F(z_n+s)\,ds .$$

The exponential formulation replaces $F$ by its interpolating polynomial through the
samples $F_j$ at the normalised nodes $\theta_j = (z_j - z_n)/h$ and integrates exactly,
giving $h\sum_k k!\,\varphi_{k+1}(hD)\sum_j c_{jk}F_j$ with $c_{jk}$ the monomial
coefficients of the Lagrange basis.

**The frame.** For any $\sigma$ — a complex scalar, or one value per element when $D$ is
diagonal — write $F(z_n+s) = e^{-\sigma(1-s/h)}G(s)$. Then
$e^{(h-s)D}F = e^{(h-s)(D - \sigma/h)}G$, so

$$u_{n+1} = e^{hD}u_n + h\sum_k k!\,\varphi_{k+1}(hD-\sigma)\sum_j c_{jk}\,
e^{\sigma(1-\theta_j)}F_j .$$

Nothing is approximated by the change of variable; only the function interpolated
changes, from $F$ to $G$. $\sigma = 0$ is the present formulation; $\sigma = hD$
element by element is the interaction formulation (the moments become the ordinary
Adams weights and the samples are transported by the linear flow). This proposal takes
the one scalar between them that removes the carrier: $\sigma = \mathrm{i}\,
\mathrm{Im}\,(hD)$ at the reference frequency of the reference mode. On a field grid the
operator is the envelope grid's plus $-\mathrm{i}\kappa$ at every frequency (§3), so this
choice turns the field-grid step into the envelope-grid step for every part of the
source that rotates with the carrier.

Rotating the history alone, without shifting the moments, is not equivalent: the
moments would then integrate $G$ against the wrong flow, an error of order $h\kappa$ per
step that the controller cannot remove.

**What each part of the march sees.**

- *The flow on the state* is $e^{hD}$ as before; the state, its carrier-envelope phase and
  the stored samples (untransported, laboratory frame) are unchanged.
- *The shift* is formed per trial from the step's own exponent $\Phi(z+h) - \Phi(z)$,
  so it follows a gradient's $\kappa(z)$ without a stored phase, and a resumed history
  needs nothing new.
- *The defect estimates* of the step's order and its neighbours are formed from the same
  rotated samples and shifted moments; the order choice and the controller are unchanged.
- *The transported compensation* of a varying operator carries each sample's
  $(D(z_j) - B_n)u_j$ by its own flow and integrates it with the ordinary Adams weights;
  it does not use the moments and is unchanged. The untransported witness integrates its
  terms with the moments and must rotate them alike.
- *An interpolated plane* at the fraction $\xi$ becomes
  $e^{\xi hD}u_n + e^{-\sigma(1-\xi)}h\sum_k k!\,\xi^{k+1}\varphi_{k+1}(\xi(hD-\sigma))
  \sum_j c_{jk}G_j$.
- *A coupled section* needs $\varphi_k(hA - \sigma I)$ of its frozen block, a shift of the
  block's diagonal.
- *The in-stage collar's stability limit* (SEP 16) concerns the negative-real part of the
  operator; a purely imaginary shift leaves it where it is.

**Which rotation a source has.** A cubic source on a real field has parts near
$\omega_0$, which rotate with the carrier at $\kappa$, and near $3\omega_0$, which rotate
at $3\kappa$; a multimode source in a higher mode is driven by the fundamental and
rotates at the fundamental's $\kappa$, not its own mode's. A scalar shift therefore
removes the dominant rotation exactly and leaves the third-harmonic band with a residual
$2\kappa$. A shift of $3\sigma$ over the harmonic band removes that too, at the price of
a band boundary (§6, §12).

### 5. Proposed interface

None by default: the frame is a property of the exponential formulation's arithmetic,
on wherever the operator keeps a carrier phase. For comparison and qualification a
switch on the method,

```text
Multistep(carrier_frame="scalar")    # proposed default
Multistep(carrier_frame="harmonic")  # 3 sigma above 2 omega0 (§12, Q1)
Multistep(carrier_frame="off")       # SEP 16's arithmetic
```

refused with the interaction formulation. The name and whether the switch is public are
open (Q4).

### 6. Implementation strategies

**Layer.** Backends only: the march's trial arithmetic and its interpolated planes. The
kernel contract, the translation and the output schema are unchanged; no
`SCHEMA_VERSION` change.

**Where the shift comes from.** The reference element's imaginary part of the trial's own
exponent, so it is the backend's own $\Delta\Phi$ and needs no new table. On the native
backend the exponent is formed in C++, on a device for device builds: the host can form
the single reference element's chord itself, or the engine can read it back (one
element).

**Applying the rotation.** Two equivalent ways:

1. *Rotated samples* — multiply each history row by $e^{\sigma(1-\theta_j)}$ as it is read
   into the predictor and corrector sums. One complex multiply per element per sample read,
   no new storage when fused into the existing sums; works for a per-element shift.
2. *Complex coefficients* — fold the scalar factor into $c_{jk}$, making the coefficient
   tables complex. No extra arithmetic per element, but every compiled kernel's coefficient
   argument changes type, and it cannot express a per-element (harmonic) shift.

The recommendation is (1) behind a small per-sample factor array, which serves both the
scalar and the harmonic shift; the moments take the shifted exponent, a scalar subtraction
before the existing $\varphi$ evaluation.

**Backends in turn.** `numpy_ref`: the trial, the accepted step's interpolant and the
untransported witness (prototyped for the scalar and harmonic shifts on diagonal
operators). `numba_cpu`: its exponential kernels form their own moments and read the
store by slot; they take the shifted exponent and the per-sample factors. Native: the
same two arguments to its trial and accept entry points, and the shift's source as above.
Interpolated planes are evaluated by the shared reference passes on every backend, so
they change once.

**Cost.** Per trial, one scalar and a factor per sample; per element, at most one complex
multiply per sample read. Memory is unchanged.

### 7. Verification and validation

**Exactness and order.**

- A scalar witness with $F = e^{\mathrm{i}\kappa z}g(z)$ for a polynomial $g$ of degree below
  the order: with the frame the step is exact to rounding at any $h\kappa$; without it,
  it is not.
- The convergence order of the populated bins on the existing second-order soliton test,
  unchanged, with the frame on a field-grid variant.
- Field/envelope parity: Kerr-only propagation on the field grid with the frame takes the
  envelope grid's step count to within a few per cent and agrees with it in the
  fundamental band.

**Work–precision.** Against converged references (IP-DOPRI5 at a tolerance three orders
tighter, checked against a second method), error at the last plane against source
evaluations and time, on the four cases below and on the benchmark suite's field-grid
cases, on `numba_cpu` and native CPU. The frame must not lose on any case where the
exponential formulation led, at matched error.

**The weak band.** A weak-band observable beside the field error, as the manual already
recommends: a harmonic boundary must not leave a band below the norm's sight wrong by
more than the unshifted formulation leaves it.

**Backend parity, resume and kinks.** The compiled backends against `numpy_ref` to the
existing bounds; a resumed run against an uninterrupted one to rounding; a hard kink's
restart unchanged.

**Measured so far** (prototype, `numpy_ref`, `rtol = 1e-6`, final-field relative $L^2$
error against IP-DOPRI5 at `rtol = 1e-9`; the cases are built from
`Simulation.from_capillary` with the parameters given):

| case | $\kappa$, rad/m | today | scalar frame | interaction | IP-DOPRI5 |
|---|---|---|---|---|---|
| He 256 µm, steady fill to 0.2 bar, 4.2 m, 6 fs 1 mJ 800 nm | −12 | 1128 / 1.4e-6 | **932 / 1.1e-6** | 1644 / 7.5e-7 | 2268 / 4.4e-6 |
| He 125 µm, 0.8→0 bar, 1.5 m, 7.5 fs 275 µJ, four modes | −48 | 1805 / 1.6e-6 | **1742 / 2.5e-6** | 10475 / 3.1e-7 | 12397 / 2.5e-6 |
| H₂ 16 µm, 40 bar, 0.1 m, 400 fs 10 µJ 515 nm, Raman, plasma | −5470 | 2669 / 5.8e-6 | **477 / 1.4e-5** | 1401 / 2.3e-6 | 798 / 3.0e-7 |
| H₂ 20 µm, 5 bar, 1 m, 40 fs 4 µJ 1030 nm, Raman, no loss or plasma | −2420 | 26060 / 1.0e-4 | **18910 / 7.6e-5** | 17032 / 6.6e-5 | 27979 / 3.2e-5 |

Source evaluations / error. On the hydrogen comb-to-continuum case the frame at
`rtol = 1e-7` takes 582 evaluations to an error of 1.2e-6: 4.6 times fewer than today's
default and 5 times more accurate. With Kerr alone it takes 395 evaluations, against 387
for the same physics on an envelope grid and 2485 without the frame. In the self-frequency
shift case the frame closes only a third of the gap: a soliton shifting away from
$\omega_0$ and a third harmonic phase-mismatched from $3\kappa$ add rotations no scalar
removes.

Shifts tried and not proposed: a shift per mode from each mode's own $\beta_0$ (four-mode
helium: 2522, worse than the scalar's 1742, because the higher modes' sources are driven
by the fundamental); a shift per element estimated from the two newest samples (four-mode
helium: 3009 with 260 rejections, the estimate being noisy in weak bins). The harmonic
shift (§12, Q1) took 757, 1210 and 13895 evaluations on the first, second and fourth cases
but 584 on the third, whose blue self-phase-modulated content above $2\omega_0$ rotates at
$\kappa$, not $3\kappa$.

### 8. Documentation

[The Adams methods](../../guide/numerics/adams.md): the frame in *The exponential
formulation*, with the formula for interpolated planes and the paragraph *Which
formulation suits which source* revised for the carrier. [Choosing a
stepper](../../guide/numerics/choosing-a-stepper.md): re-measured ranks on field-grid
cases. [The linear operator](../../guide/numerics/linear-operator.md): a pointer from the
grid table to the stepper's treatment of the kept carrier phase. Docstrings of
`ExponentialKernels` and the method.

### 9. Compatibility and impact

The default stepper's trajectory changes on every grid that keeps a carrier phase: field
grids, and envelope grids with the third-harmonic term. Results change within the
tolerance; the reproducibility contract (rounding-level bounds, not bitwise) is unaffected.
Envelope grids without the third harmonic subtract $\beta_0$, so their shift is at most
the reference mode's residual frame phase; whether to apply it there or leave those runs
exactly as today is Q3. Recorded call counts and timings on field-grid cases in the
manual, the benchmark suite and tests that pin call budgets change and are re-measured.
No output schema or `SCHEMA_VERSION` change. Luna.jl's exponential steppers are not
affected; no difference to record.

### 10. Alternatives considered

- **Make the interaction formulation the default on field grids.** It wins on the dense
  molecular cases but loses on the helium capillaries, by 1.5 times on the canonical
  case and 5.8 times on the four-mode one; the frame keeps
  the exponential formulation's advantage on driven bins and removes its weakness.
- **Choose the formulation per run from $\kappa$.** A heuristic switch between two
  numerics, with a threshold to qualify; the frame needs no threshold.
- **Subtract $\beta_0$ on field grids.** Wrong: on a real field it changes the
  carrier-envelope phase and so the physics.
- **Store the history in a rotating frame.** Equivalent per step, but the stored rows would
  carry an absolute phase $\int\kappa\,dz$ that a resume and a gradient must track; forming
  the shift per trial keeps the store untransported.
- **Lower order on such cases.** Measured worse (§2).

### 11. Phases

1. **`numpy_ref`.** The scalar frame in the trial, the interpolated planes, the
   untransported witness and the coupled sections; the method switch; the scalar witness,
   field/envelope parity, order, resume and kink tests; the manual pages. Work–precision on
   the four cases above.
2. **`numba_cpu`.** The shifted exponent and per-sample factors in its exponential
   kernels; parity with `numpy_ref` to the existing bounds; timings on the four cases.
3. **Native.** The same on CPU, then device builds; parity and timings.
4. **Qualification and default.** The benchmark suite's field-grid cases on `numba_cpu`
   and native CPU, at matched error, against today's default, the interaction formulation
   and IP-DOPRI5; the decision on the default and on the harmonic shift (Q1); re-measured
   manual ranks and pinned budgets.

### 12. Open questions

- **Q1 — the harmonic shift.** $3\sigma$ over the third-harmonic band helps the helium
  and self-frequency-shift cases and hurts the comb-to-continuum case, and with a fixed
  boundary at $2\omega_0$ a weak band beside it was left wrong below the norm's sight.
  A boundary placed in a spectral gap, a boundary that moves with the spectrum, or the
  scalar shift alone; to settle in phase 4.
- **Q2 — loss in the shift.** Whether $\sigma$ should carry the reference element's
  real part as well. Not measured; the imaginary part alone is a pure phase and cannot
  amplify a sample.
- **Q3 — envelope grids without the third harmonic.** Apply the (near-zero) shift
  everywhere for one code path, or skip it so those trajectories are unchanged.
- **Q4 — the switch.** Public (`carrier_frame=`) for comparisons, or internal and always
  on once qualified.
- **Q5 — the reference mode on multimode and coupled routes.** The fundamental's shift
  measured best on four helium modes; whether that holds where the launched energy is not
  in the reference mode.

### 13. References

1. S. M. Cox and P. C. Matthews, "Exponential time differencing for stiff systems,"
   *J. Comput. Phys.* **176**, 430 (2002). doi:10.1006/jcph.2002.6995
2. M. Hochbruck and A. Ostermann, "Exponential multistep methods of Adams-type,"
   *BIT Numerical Mathematics* **51**, 889 (2011). doi:10.1007/s10543-011-0332-6

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).*

## Part III — History

- 2026-10-25: Drafted, with a `numpy_ref` prototype's measurements.

- 2026-10-25: GPT-6 completed the AI review of Part I at `2a4a7f9`, recorded
  below. Review status is `In review`; proposal status remains `Draft`. No human
  review or maintainer decision is implied.

### AI review — 2026-10-25 — GPT-6

Reviewed all of Part I against the conventions, current Adams implementations,
SEPs 16 and 25, and SEP 0's acceptance criteria. **Recommendation: revise before
acceptance.** The constant-operator change of variable is sound, including the
dense-output prefactor. It provides a useful intermediate choice between the
unshifted exponential and interaction formulations. It does not, however, establish
unchanged stability or reliable error detection for the residual source, and the
proposed default covers more regimes than the evidence establishes.

Only the review header and Part III are changed. P1 points concern numerical
correctness, qualification or execution contracts that should be settled before
acceptance; P2 points require explicit design decisions or corrections. All
maintainer decisions and responses to R1–R10 are pending. Numerical probes below
evaluate the proposed formula independently; they are not qualification of an
implemented feature or a reproduction of the prototype's propagation results.

1. **R1 (P1), §§1, 4, 7, 9 — An unchanged predictor–corrector defect can miss a
   large error introduced by the shift.** Consider a populated scalar bin with
   $h=1$ m, $D=2\pi\mathrm{i}$ m⁻¹, $F=1$ amplitude-unit/m, and $u_n=1$,
   with exact history on equally spaced nodes. The proposed rule gives
   $\sigma=2\pi\mathrm{i}$ and $hD-\sigma=0$. Every rotated sample is one,
   so both predictor and corrector add one. The exact source integral is zero:
   $\int_0^1 e^{2\pi\mathrm{i}(1-s)}\,ds=0$. Independent moment calculations
   at predictor orders 1, 3 and 5 gave absolute endpoint error 1, with defects
   between zero and $4.5\times10^{-16}$; the unshifted formula was exact to
   rounding. At $h\operatorname{Im}D=2\pi\pm0.001$, order 3 still gives
   error about one and defect $3.75\times10^{-10}$, so the failure is not
   confined to an exactly resonant phase. This also affects mixed rotations:
   with $D=\mathrm{i}$ m⁻¹,
   $F(z)=e^{\mathrm{i}z}+0.1e^{3\mathrm{i}z}$ in consistent units and
   $h=\pi$ m, the scalar frame aliases the residual harmonic. A third-order
   predictor gave error $0.1\pi$ and a defect below $10^{-30}$.

   This is finite-sampling aliasing, not a flaw in the integral identity, and
   does not prove that a normal adaptive startup reaches these stencils. It does
   show that neither a small defect nor a populated-bin norm establishes the
   blanket claims; the risk is not confined to an invisible weak band or a
   harmonic boundary. **Recommended change/question:** state the smoothness and
   resolution assumptions for the *rotated* source. Add mixed-frequency and
   non-carrier-driven witnesses, including near-resonant phases, variable steps,
   adaptive startup and replay. What qualification or admission rule prevents
   the new default from losing the unshifted method's accuracy on these sources?
   Do not infer global accuracy or universal improvement from exactness of the
   homogeneous flow.

2. **R2 (P1), §4, “The in-stage collar's stability limit” — A purely imaginary
   shift does change the explicit recurrence's stability.** For the scalar test
   equation $u'=Du+\lambda u$, a constant shift makes the recurrence, after a
   unit-modulus change of coordinates, the unshifted exponential PEC recurrence
   with $x'=hD-\sigma$ and the same $h\lambda$. Its stability therefore depends
   on the residual imaginary phase as well as the real damping. For PEC3/4,
   $hD=0$ and $h\lambda=-0.25$, the recurrence's spectral radius is
   $0.8879010401$. Applying $\sigma=-4\mathrm{i}$, as a second bin's reference
   phase can require, changes it to $1.0245058313$: stable becomes unstable,
   despite $0.25$ lying inside the quoted negative-real interval $0.30$.
   Both quadrature-built weights and production `exponential_weights` give
   these values. The recurrence retains the predicted-state source, as the
   actual PEC method does.

   This commuting scalar damping example already disproves the stated reason;
   the temporal collar additionally couples spectral bins. **Recommended
   change/question:** derive or measure the applicable stability envelope in
   residual phase and damping, and qualify the order-selection weight against
   it. Include a resolved collar case and a coupled case. Under precisely what
   restrictions can `PEC_REAL_INTERVAL` remain an adequate order-selection
   heuristic? Keep that question distinct from a rigorous stability guarantee.

3. **R3 (P2), §§1, 4, 7 — Make the carrier sign and the field/envelope equivalence
   explicit.** With the defined $\kappa=\beta(\omega_0)-\beta_1\omega_0$,
   the internal reference operator is $D_{\rm ref}=-\mathrm{i}\kappa$ and
   $\sigma=-\mathrm{i}h\kappa$. The matching witness is consequently
   $F=e^{-\mathrm{i}\kappa z}g(z)$. Section 7 instead writes the plus sign;
   using the proposed default shift on that witness leaves a rotation at
   $2\kappa$, so its promised polynomial exactness does not follow. Either
   change the witness's sign or explicitly define a different signed rotation
   rate and supply its shift.

   Also, $D_{\rm field}=D_{\rm envelope}-\mathrm{i}\kappa$ refers to the
   envelope operator **without** THG and with the same physical frame. For the
   documented envelope operator **with** THG, the difference is instead
   $+\mathrm{i}\beta_1\omega_0$, and its carrier shift is based on
   $-\mathrm{i}\beta_0$. **Recommended change/question:** derive the two
   cases separately and state the phase-equivariant response assumptions under
   which the fundamental source transforms as claimed. “Kerr-only” must specify
   whether the real-field cubic's THG part is retained. Restrict field/envelope
   parity to matched physics and grids; similar call counts are a performance
   observation, not a correctness criterion. The already suspected envelope-THG
   discrepancy cannot certify that route's accuracy.

4. **R4 (P2), §§5–6, 11–12, Q3 and Q5 — Specify how every supported route obtains
   its reference shift before phase 1.** “At the reference frequency of the
   reference mode” is not yet an executable rule. A field lattice need not
   contain that frequency exactly; a custom `Frame` can use a reference different
   from the grid's; free-space columns are transverse wavevectors rather than
   guided modes; and a coupled route's cumulative diagonal can be a common
   carrier rather than the full first channel's operator. These choices can
   change the residual phase that drives R1 and R2. The statement that an
   envelope without THG necessarily has a near-zero shift also needs the
   co-moving-frame assumption; custom frame coefficients are supported.

   **Recommended change/question:** name the selected frequency/bin, tie rule,
   channel or transverse column, and behaviour when that reference is inactive
   or absent, for each admitted route. Say whether the shift represents the
   common diagonal or an eigenvalue of a coupled block, and specify its batch
   semantics. Test a shifted launch, a non-fundamental launch, a custom frame
   and free-space propagation. Assign Q3–Q5 to a decision point before they are
   needed by the implementation; retain an explicit refusal or `off` path for
   unqualified choices rather than relying on a backend-specific convention.

5. **R5 (P1), §§4, 6–7, 11 — Complete the varying-operator and coupled recurrence,
   rather than extending the constant-diagonal identity by assertion.** The
   transported compensation can indeed retain its existing *formula*, but its
   endpoint state is the new predictor, so its values and its combined defect
   change. A chord shift also removes only the chord phase: for a source
   proportional to $e^{\int D_{\rm ref}\,dz}$, a varying carrier leaves a
   residual phase across the history. The constant-polynomial witness cannot
   establish accuracy or estimator behaviour there.

   For a coupled block, distinguish the homogeneous flow $e^{hB}$ from the
   moments $\varphi_k(hB-\sigma I)$. Currently `coupled_moments` returns both
   in one `_MatrixMoments` object and `HostHistory.exponential_trial` uses that
   object as the flow too. Merely shifting its input shifts the propagated
   state, contrary to this SEP's central contract. **Recommended
   change/question:** write the complete predictor, corrector, combined defect
   and dense formula with transported compensation, and identify the separate
   unshifted flow in both Schur and eigenbasis paths. Require manufactured
   varying-carrier and nonnormal-block witnesses, three-level convergence,
   changing step sizes/rejections, dense endpoints and interiors, section
   transitions, hard kinks and the structurally source-free exact chord. Include
   cancellation between the source and compensation defects. No extra source
   call should be required, but that does not by itself validate the estimate.

6. **R6 (P2), §§5, 7, 11–12, Q1–Q2 — Separate the scalar proposal from extensions
   whose safety criterion is not yet defined.** A frequency threshold does not
   identify a nonlinear source's phase: fundamental broadening and harmonic
   generation can contribute to the same bin, as the prototype already found.
   Moving the threshold into a gap does not prove that it remains in one. The
   requirement that a weak band be no worse than the unshifted result permits
   both methods to be wrong. Give the band an independent reference and an
   absolute or relative observable error budget fixed before comparison,
   including motion and filling of the gap and any boundary change after a
   rejected trial. Otherwise defer `harmonic` explicitly and retain scalar-only
   acceptance criteria.

   Including loss in the shift has a separate range problem. An evaluated bin
   with $hD=0$ and a lossy reference $\sigma=-1000$ needs moments at $+1000$
   and exponentially small sample factors: the proposed separate evaluation
   overflows/underflows although the original integral is finite.
   `phi_functions` deliberately refuses real arguments above 700. **Recommended
   change/question:** keep Q2 outside this implementation unless a stable
   combined evaluation, its admissible range and its tests are specified; the
   algebra's allowance for complex shifts is not a floating-point guarantee.

7. **R7 (P1), §§5–6, 9, 11 — Reconcile the method switch and phased rollout with
   the versioned seam.** A `carrier_frame` field on `Multistep` is carried inside
   `StepperSettings.integrator`, which is part of `KernelProblem`; it is not a
   backend-only private flag. The version-20 record in `translate.problem`
   explicitly includes the earlier addition of `formulation` to that method.
   The unconditional promise of no `SCHEMA_VERSION` change is therefore not
   justified by leaving the outer dataclass fields alone. In particular, an
   older native adapter must not silently accept a scalar/harmonic request and
   run unshifted arithmetic.

   **Recommended change/question:** settle the interface and capability
   contract, state which phase bumps the schema when required, and require
   unsupported backends to refuse the selected frame by name. Keep the old
   default until phase 4's qualification or state how earlier phases remain
   independently usable. Distinguish kernel schema compatibility from whether
   the existing output fields can already record the setting; the latter need
   not automatically require a new output layout.

8. **R8 (P1), §6 — Native dense observation is a separate implementation, and
   the cost claim omits changed moment work.** `_ResidentAdams.observe` in
   `native/src/solprop_native/backend.py` calls the native
   `adams_exponential_observe`; `GeneralWorkspace::adams_exponential_observe`
   in `native/src/general.cpp` evaluates the moments and sums in a Kokkos
   kernel. It does not use the shared reference pass. Updating trial and accept
   alone leaves saved interior planes inconsistent with the new step.

   **Recommended change/question:** include native dense state and arithmetic
   explicitly, preserve the accepted trial's shift across later trial/rejection
   activity, and test interior planes against an analytic shifted witness and
   NumPy on CPU and qualified devices. For coupled host paths, likewise ensure
   the shift reaches their actual interpolant. Account for any retained factors
   in the memory ledger. The timing model must also include moment evaluation:
   `phi_functions` switches algorithms at $|x|=3$, so making $x-\sigma$ small
   changes the cost beyond one multiply per history sample. A device readback
   also introduces a dependency/synchronisation. Measure wall time, memory and
   transfers before claiming that the frame costs nothing where the old method
   already led.

9. **R9 (P2), §§1, 7, 9 — Unchanged history arrays do not mean unchanged resume
   compatibility or rounding-level agreement.** `AdamsMarch._refusal` compares
   `snapshot.method` with `repr(self._method)`, and the run record stores the
   method representation. Adding the proposed field changes that identity even
   for `off`; an entirely hidden switch instead risks leaving different
   arithmetic under one identity. Moreover, a history's stored source was
   evaluated at a predictor produced by the previous method. Keeping its array
   representation is not proof that a cross-policy continuation is the same
   discrete computation.

   **Recommended change/question:** state the treatment of old checkpoints and
   of `off`/`scalar`/`harmonic` changes: explicit refusal, a documented restart,
   or a qualified conversion. Record the effective policy and deterministic
   selection rule, and test saved and in-memory continuation. Follow SEP 25:
   rounding bounds apply to matched methods, schedules, orders and histories;
   general adaptive/resumed comparisons need accuracy envelopes against a
   converged reference. Include rotation, phase evaluation, subtraction and
   interpolation conditioning in new rounding budgets rather than promising
   the existing bounds unchanged. “Results change within the tolerance” is
   not a global-error guarantee supplied by `rtol`.

10. **R10 (P2), §§1–2, 7, 10–11 — Make the performance evidence reproducible and
    the qualification compliant with the short-window policy.** The table names
    physical parameters but no complete executable fixture, prototype revision,
    realised grids, time window, wavelength limits, response selections, collar
    scheme or timing setup. These affect the comparison materially. At common
    `rtol`, 477 calls with error $1.4\times10^{-5}$ versus 2669 with
    $5.8\times10^{-6}$ is promising but not matched accuracy; the additional
    582-call point strengthens the motivation without establishing all claimed
    nonregressions. An IP-DOPRI5 run at a tighter tolerance is a candidate
    reference until its convergence and agreement with the named second method
    are shown.

    The proposed phases also call for repeated work–precision measurements on
    fibres of 0.1–4.2 m with hundreds to tens of thousands of source calls.
    `AGENTS.md` and SEP 0 §7 require about ten steps or a problem short enough
    to finish in seconds; full problems belong in the gallery exceptions.
    **Recommended change/question:** retain these figures as preliminary
    motivation, publish reproducible case builders and a ledger, and define
    short launch and developed-spectrum windows that retain the difficult
    physics. State reference convergence checks, worst-plane field and named
    weak-band error budgets, the error range for matched-work comparisons, and
    timing variability/nonregression margins in advance. Cover cheap-source and
    currently favoured unshifted cases as well as dense gases. Do not extrapolate
    short-window timing to a full run; put any full-size demonstration in an
    appropriate worked example.

**Evidence and limits of this review.** The shifted integral and dense formula
were independently re-derived. A polynomial rotating source on irregular history
nodes, with $hD=-0.4+7\mathrm{i}$ and $\sigma=2.3\mathrm{i}$, agreed with
independent quadrature at $\xi=0.2,0.63,1$ to below $5\times10^{-17}$ absolute
error in the chosen unit-amplitude witness. R1 used augmented matrix exponentials
for moments; R2 was cross-checked with direct quadrature and production weights.
These small scalar probes do not establish propagation accuracy, performance or
device support. The prototype's four campaigns were not rerun.

The bibliographic details of both §13 references were checked against the
[Cox–Matthews publisher record](https://doi.org/10.1006/jcph.2002.6995) and
[Hochbruck–Ostermann institutional record](https://publikationen.bibliothek.kit.edu/1000041812).
The latter's [author manuscript](https://na.math.kit.edu/download/papers/multistep.pdf),
§2, supplies the variation-of-constants/interpolation foundation; it does not by
itself qualify this variable-step, carrier-shifted PEC controller. The full
Cox–Matthews text was not checked. No physical constants or material coefficients
were transcribed or changed.
