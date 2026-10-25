# SEP 35 — A carrier frame for the exponential Adams history

| | |
|---|---|
| **SEP** | 35 |
| **Title** | A carrier frame for the exponential Adams history |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-25) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
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
