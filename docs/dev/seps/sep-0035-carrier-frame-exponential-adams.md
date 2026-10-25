# SEP 35 — A carrier frame for the exponential Adams history

| | |
|---|---|
| **SEP** | 35 |
| **Title** | A carrier frame for the exponential Adams history |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-25; revision after review, 2026-10-25) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-10-25 |
| **Resolution** | Accepted 2026-10-25 by John Travers after reading the revised proposal, with the maintainer decisions recorded in Part III. No pull request; the proposal lands with its implementation's single pull request. |
| **Requires** | [SEP 16](archive/sep-0016-unified-multistep.md) (the default multistep stepper and its exponential formulation); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (the rounding-level reproducibility contract) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Branch `feature/multistep-carrier-frame`: phases 1–4 complete (`numpy_ref`, `numba_cpu`, native CPU; opt-in; the default unchanged; `"harmonic"` removed); one pull request at Final |
| **Manual pages** | [The Adams integrator: the carrier frame](../../guide/numerics/adams.md#the-carrier-frame); [choosing a stepper](../../guide/numerics/choosing-a-stepper.md); [the linear operator](../../guide/numerics/linear-operator.md#which-terms-the-frame-removes) |

## Part I — Proposal

### 1. Abstract

The default `Multistep()` integrates the nonlinear source in SEP 16's exponential
formulation: the linear flow is exact and a polynomial through the untransported source
samples is integrated against it. On a field grid, and on an envelope grid with the third
harmonic, the operator keeps $\beta_0$, so the part of the source that follows the carrier
rotates along $z$ at
$\kappa = \beta(\omega_0) - \beta_1\omega_0$, and the polynomial has to follow that
rotation. In dense molecular gases $|\kappa|$ reaches thousands of radians per metre and the
default takes several times more source evaluations than IP-DOPRI5 or the interaction
formulation (2669 against 798 and 1401 on a reduced hydrogen comb-to-continuum case); at
a loose tolerance it can also be badly wrong (a final-field error of 0.97 on an argon
capillary at `rtol = 1e-3`, against 0.07 for IP-DOPRI5). This SEP proposes an opt-in policy, `Multistep(carrier_frame="scalar")`, that
interpolates, inside each step, the source with the carrier's phase removed, integrates
the remainder against correspondingly shifted $\varphi$-functions, and caps the carrier
phase a step may cover. The change of variable is exact for any shift. The state, the
stored history, the carrier-envelope phase and the linear flow are untouched; the
error estimate, the order choice and their stability assumptions are re-examined (§4)
rather than assumed. In a reference-backend prototype the policy reaches the comb-to-continuum
case's present accuracy with 4.6 to 5 times fewer evaluations and leaves three helium and
hydrogen cases where the present default already led unchanged or slightly better; on the
argon case it brings the error at `rtol = 1e-3` to 0.05 with fewer evaluations than
IP-DOPRI5. A per-band variant, `carrier_frame="harmonic"`, that also removes the third
harmonic's rotation is developed alongside on the reference backend and promoted only if it
passes a gate fixed in §7.4. Whether either becomes the default is decided only after short-window qualification on the
benchmark suite, against error and timing criteria fixed in §7 before measurement.

### 2. Motivation and scope

**The cost.** SEP 16 chose the exponential formulation as the default on campaigns
dominated by low-pressure helium capillaries, where it is best. A later campaign on dense
molecular gases found it 4 to 10 times more expensive than the alternatives on field
grids. Lowering the order made it worse (orders 3 and 2 took about 2 and 4 times more
steps than order 5), so accuracy, not stability, sets the step. Kerr alone reproduces it,
and the same propagation on an envelope grid without the third harmonic, where $\beta_0$ is
subtracted, takes about a tenth of the steps. The source history the predictor
extrapolates rotates at $\kappa$, and at order five the controller keeps $h|\kappa|$ small
enough for a polynomial to follow it.

**In scope.**

- The exponential formulation's predictor, corrector, defect estimates and interpolated
  planes on every grid whose operator keeps the carrier phase — **field grids and envelope
  grids with the third-harmonic term**, whose operators are the same (§4.2) — for the
  operators SEP 16 steps: constant, separable and varying
  diagonal operators with their compensation, the in-stage collar, and the constant-Schur
  and frozen-residual coupled sections.
- A step cap on the carrier phase, part of the policy (§4.3).
- The per-band ("harmonic") shift of §4.6, on `numpy_ref` first and on the compiled
  backends only once it passes the gate of §7.4.
- An interim note in the manual (phase 1) on the present default's loose-tolerance
  failure on field grids in dense gases (§7.2).
- `numpy_ref`, `numba_cpu` and the native backend, CPU first, devices after.
- The decision on the default (phase 4), against criteria fixed in §7.

**Out of scope.**

- *Envelope grids without the third harmonic.* The operator subtracts $\beta_0$ and the
  policy is the present arithmetic exactly (§6.1).
- *Shifts with a real part.* Not proposed (§10).
- The interaction formulation and the Runge–Kutta marches, which already transport the
  linear phase.
- The weak-band blind spot of the step-size norm in general
  ([limitations](../../guide/numerics/adams.md#limitations)); §7 gives the weak bands
  this policy is judged on their own error budget.

### 3. Context

The exponential formulation is described in
[the Adams methods](../../guide/numerics/adams.md#the-exponential-formulation); its
arithmetic is `ExponentialKernels` and `HostHistory.exponential_trial` in
`solprop/backends/multistep.py`, with compiled counterparts in
`solprop/backends/numba_cpu/multistep.py` and the native backend's resident Adams engine
(`native/src/solprop_native/backend.py`, `native/src/general.cpp`). The operator and which
frame terms each grid removes are set out in
[the linear operator](../../guide/numerics/linear-operator.md#which-terms-the-frame-removes)
and in `solprop.linear.Frame`: the frame is fixed on the first mode at its own reference
frequency, and a field grid keeps $\beta_0$ because a frequency-independent phase on its
half-spectrum changes the carrier-envelope phase. The order choice caps each order's step
for an in-stage collar with `PEC_REAL_INTERVAL` (`solprop/stepper/multistep.py`), the
negative-real stability interval of PEC $q/(q+1)$ for $u' = \lambda u$.
[SEP 15](archive/sep-0015-exponential-adams.md) introduced exponential Adams stepping;
SEP 16 unified it with the interaction formulation and made it the default; SEP 25 sets
the reproducibility contract the parity tests use.

### 4. Theory

#### 4.1 The shifted step

With a diagonal operator $D$ (internal FFT convention, so propagation is
$e^{\int D\,dz}$ and $D = -\mathrm{i}(\dots)$, [conventions](../conventions.md)) and a
source $F$, the exact step from $z_n$ is

$$u_{n+1} = e^{hD}u_n + \int_0^h e^{(h-s)D}F(z_n+s)\,ds .$$

The exponential formulation replaces $F$ by its interpolating polynomial through the
samples $F_j$ at the normalised nodes $\theta_j = (z_j - z_n)/h$ and integrates exactly:
$h\sum_k k!\,\varphi_{k+1}(hD)\sum_j c_{jk}F_j$, with $c_{jk}$ the monomial coefficients
of the Lagrange basis. For any $\sigma$ — a complex scalar, or one value per element when
$D$ is diagonal — write $F(z_n+s) = e^{-\sigma(1-s/h)}G(s)$. Then
$e^{(h-s)D}F = e^{(h-s)D-\sigma(1-s/h)}G$, and interpolating $G$ instead of $F$ gives

$$u_{n+1} = e^{hD}u_n + h\sum_k k!\,\varphi_{k+1}(hD-\sigma)\sum_j c_{jk}\,G_j,
\qquad G_j = e^{\sigma(1-\theta_j)}F_j .$$

The identity is exact; only the function interpolated changes. $\sigma = 0$ is the present
formulation; $\sigma = hD$ element by element is the interaction formulation. Rotating
the samples without shifting the moments is not equivalent: it integrates $G$ against the
wrong flow, an error of order $|\sigma|$ per step that no step-size choice removes.

An interpolated plane at the fraction $\xi$ of an accepted step is

$$u(z_n+\xi h) = e^{\xi hD}u_n + e^{-\sigma(1-\xi)}\,h\sum_k k!\,\xi^{k+1}
\varphi_{k+1}\bigl(\xi(hD-\sigma)\bigr)\sum_j c_{jk}G_j ,$$

with the accepted step's own $\sigma$; every factor is bounded for imaginary $\sigma$.

#### 4.2 Which shift removes the carrier, and from which source

**Field grid.** The operator is $D_\text{field} = -\mathrm{i}(\beta - \beta_1\omega) - \alpha/2$,
and the envelope grid without the third harmonic, with the same frame, has
$D_\text{env} = -\mathrm{i}(\beta - \beta_0 - \beta_1(\omega-\omega_0)) - \alpha/2$, so

$$D_\text{field} = D_\text{env} - \mathrm{i}\kappa ,\qquad
\kappa = \beta_0 - \beta_1\omega_0 = \beta(\omega_0) - \beta_1\omega_0 ,$$

at every frequency. The field grid's state is the envelope grid's times
$e^{-\mathrm{i}\int\kappa\,dz}$, and the proposed shift is the chord of that phase over
the step,

$$\sigma = -\mathrm{i}\int_{z_n}^{z_n+h}\kappa\,dz \;\approx\; -\mathrm{i}\kappa h ,$$

the imaginary part of $hD$ at the frame's reference (§6.1). A carrier-following source
sample then behaves as $F_j \propto e^{-\mathrm{i}\kappa z_j}g(z_j)$ with $g$ smooth, and
$G_j = e^{\sigma(1-\theta_j)}F_j \propto g(z_j)$ up to the constant
$e^{-\mathrm{i}\kappa z_{n+1}}$.

**Which part of the source follows the carrier.** Exactly the part that is equivariant
under a change of carrier phase, $S(e^{\mathrm{i}\phi}u) = e^{\mathrm{i}\phi}S(u)$ near
$\omega_0$: the fundamental-band part of the instantaneous cubic response, of Raman
responses driven by the cycle-averaged intensity, and of any response that depends on the
field only through its envelope. Three parts do not:

- the third-harmonic part of the real-field cubic, near $3\omega_0$, which rotates at
  $3\kappa$ and is left with a residual $2\kappa$ (plus its phase mismatch);
- slowly varying parts near zero frequency (optical rectification, the low-frequency
  part of a Raman or plasma polarisation) and any driven bin not tied to the carrier,
  which do not rotate and are given a residual $\kappa$ by the shift;
- parts that depend on the carrier-envelope phase itself, such as field-grid ionisation
  driven by the instantaneous real field, whose sub-cycle structure adds components at
  other rotation rates.

Every one of these is still integrated exactly in its linear part; what changes is how
fast the interpolated function rotates. The unshifted formulation has the mirror image:
the carrier-following part rotates at $\kappa$ and the stationary part is smooth.

**Envelope grid with the third harmonic.** Its frame keeps the carrier phase whole and
measures the delay from zero frequency
([which terms the frame removes](../../guide/numerics/linear-operator.md#which-terms-the-frame-removes),
`solprop.linear.frame_origin`), so its operator is the field grid's,
$-\mathrm{i}(\beta - \beta_1\omega) - \alpha/2$, the same $\kappa$ and the same shift apply,
and the harmonic term's carrier factor $e^{2\mathrm{i}\omega_0 t}$ is fixed in time, so the
harmonic band's source rotates at $3\kappa$ as on the field grid.

#### 4.3 Resolution and aliasing

The polynomial must resolve the rotated function $G$, as the present formulation must
resolve $F$. A component of $G$ with residual rotation rate $\nu$ turns by $|\nu| h$ per
step. When $|\nu|h$ approaches a multiple of $2\pi$ the samples alias: they look smooth,
both the endpoint and the defect are computed from the aliased samples, and the defect
does not see the error. This is not specific to the shift. For the present formulation,
$hD = 2\pi\mathrm{i}$ with a carrier-following source $F = e^{2\pi\mathrm{i}z}$ (unit
amplitude, $h = 1$) gives an endpoint error of 1 and defects of $4\times10^{-17}$ to
$5\times10^{-16}$ at orders 1, 3 and 5; the shifted step has the same failure for a
stationary source. Which components are smooth is what the shift changes.

On the mixed witness $D = \mathrm{i}$, $F = e^{\mathrm{i}z} + 0.1e^{3\mathrm{i}z}$ (a
carrier with a residual $2\kappa$ component, $\kappa = 1$), one step on exact history, the
shifted step's endpoint error and its order-$q$ defect (corrector through the exact
endpoint sample less the predictor) are:

| $h\vert\kappa\vert$ | $q$ | error, shifted | defect, shifted | error, unshifted | defect, unshifted |
|---|---|---|---|---|---|
| 0.25 | 3 | 4.0e-5 | 1.1e-3 | 2.2e-4 | 5.1e-3 |
| 0.5 | 5 | 5.9e-4 | 1.3e-2 | 5.3e-3 | 7.8e-2 |
| 1 | 3 | 2.8e-2 | 1.8e-1 | 1.1e-1 | 5.4e-1 |
| 1 | 5 | 4.2e-2 | 4.5e-1 | 1.6e-1 | 7.6e-1 |
| 2 | 5 | 3.7e-1 | 1.3 | 9.7e-1 | 8.2 |
| $\pi$ | 5 | 3.1e-1 | 4.4e-16 | 5.1 | 30 |

Up to $h|\kappa| = 2$ the defect exceeds the error; at $h|\kappa| = \pi$ the residual
$2\kappa h = 2\pi$ aliases and the defect vanishes while the error does not.

**The cap.** The policy therefore limits every step to $h|\kappa| \le c$, $c = 1$ rad
(§4.4). Then the residual phase per step of the components named in §4.2 is at most $c$
(stationary) and $2c$ (third harmonic, plus its mismatch), below the aliasing condition
$\pi$ with a margin, and on the witness the defect over-estimates the error. The cap is
an assumption-bounding device for the components the theory names, not a guarantee for an
arbitrary source: a populated component with a residual rate far above $2|\kappa|$ is
resolved, as now, only if the controller's gradual step growth sees its defect before
the step reaches an alias. The present formulation relies on the same mechanism for every
rotating component. The verification (§7) adds witnesses for near-resonant phases,
variable steps, startup and replay.

**What it costs.** Without the cap the controller's steps averaged $h|\kappa| \approx 1.1$ on the
comb-to-continuum case; with it, that case takes 582 rather than 477
evaluations and its error falls from $1.4\times10^{-5}$ to $6.4\times10^{-6}$, consistent
with the uncapped run's under-resolved residual rotation. On the three helium and
hydrogen cases of §7 the cap never binds and the step sequences are identical.

**What the scalar shift leaves.** The third-harmonic band's residual $2\kappa$ is resolved,
not removed, and at a loose tolerance it can set the order. On the argon case of §7.2 the
present default runs at order 2 throughout; with the scalar shift the run sits at order 1,
where the order-2 defect exceeds the order-1 one, with or without the plasma. §4.6 removes
that residual.

#### 4.4 Stability

For the test equation $u' = Du + \lambda u$ with $\lambda u$ in the source (the in-stage
collar's damping is such a term), the shifted recurrence is, after the unit-modulus change
of coordinates $v_j = e^{-\sigma\theta_j}u_j$, the unshifted exponential PEC recurrence
with $x' = hD - \sigma$ and the same $h\lambda$. Its stability depends on the residual
phase $\operatorname{Im}x'$ as well as on $h\lambda$: for PEC3/4 with $hD = 0$ and
$h\lambda = -0.25$ the spectral radius is 0.888, and with $\sigma = -4\mathrm{i}$ it is
1.025. `PEC_REAL_INTERVAL` is the interval at $x' = 0$. Computed from the same recurrence
(production $\varphi$-functions and Lagrange coefficients), the largest $|\operatorname{Im}x'|$
up to which every $h|\lambda| \le$ `PEC_REAL_INTERVAL[q-1]` stays stable is:

| $q$ | interval | $\operatorname{Re}x' = 0$ | $\operatorname{Re}x' = -0.1$ | $\operatorname{Re}x' = -0.5$ |
|---|---|---|---|---|
| 1 | 1.0 | > 5.9 | > 5.9 | > 5.9 |
| 2 | 0.545 | 5.1 | > 5.9 | > 5.9 |
| 3 | 0.3 | 3.5 | > 5.9 | > 5.9 |
| 4 | 0.163 | 3.2 | 3.4 | > 5.9 |
| 5 | 0.0877 | 3.0 | 3.1 | > 5.9 |

So `PEC_REAL_INTERVAL` remains a valid bound in a damped bin whose residual phase is at
most about 3 rad per step at orders up to five. The present formulation needs the same
with $x' = hD$, and neither formulation enforces it: the temporal collar damps whatever
frequencies reach the window's edges, and its time-domain multiplication couples bins, so
the per-bin bound is a heuristic in both. The shift moves each damped bin's residual phase
by $-\sigma$, at most $c = 1$ rad under the cap, which can move a bin across the
threshold in either direction. The order weight is kept, and its adequacy under the policy
is qualified on a resolved collar case and a coupled case (§7), not asserted.

#### 4.5 Varying operators and coupled sections

**Compensation.** With a varying diagonal operator the present formulation steps with the
chord $B_n = \Delta\Phi/h$ and adds the transported compensation of
$C_j = (D(z_j) - B_n)u_j$, carried by $e^{(1-\theta_j)hB_n}$ and integrated with the
ordinary Adams weights $a_j$. With the shift the predictor is

$$u^{P}_{n+1} = e^{hB_n}u_n + h\sum_k k!\,\varphi_{k+1}(hB_n-\sigma)\sum_{j} c^{P}_{jk}G_j
+ h\sum_j a^{P}_j\,e^{(1-\theta_j)hB_n}C_j ,$$

the corrector replaces $c^P, a^P$ by the corrector's coefficients over the endpoint sample
$G_{n+1} = F(z_{n+1}, u^P_{n+1})$ and the endpoint compensation $(D(z_{n+1}) - B_n)u^P_{n+1}$,
and each defect is the difference of the two, both parts together. The compensation's
*formula* is unchanged, but its endpoint is the new predictor, so its values change and
the source and compensation parts of a defect can cancel differently; witnesses for that
cancellation are in §7. A dense plane adds the existing compensation term to the shifted
source term of §4.1.

**A varying carrier.** $\sigma$ is the chord of the carrier phase over the step being
taken. A history sample at $\theta_j$ carries the phase $\int^{z_j}\kappa\,dz$, so after
the rotation a residual phase $\int_{z_j}^{z_{n+1}}\kappa\,dz - (1-\theta_j)\int_{z_n}^{z_{n+1}}\kappa\,dz$
remains; for $\kappa$ linear in $z$ it is exactly $\tfrac12\kappa' h^2\theta_j(1-\theta_j)$,
quadratic in $\theta$ and so within the polynomial; a manufactured varying-carrier witness checks it (§7). The
exactness of §4.1 holds for any $\sigma$.

**Coupled sections.** With a frozen block $B$, the flow is $e^{hB}$ and the moments are
$\varphi_k(hB - \sigma I)$. Because $\sigma I$ commutes with $B$,
$e^{hB} = e^{\sigma}e^{hB-\sigma I}$, so a single evaluation of the shifted block gives
both, the flow by the scalar factor $e^{\sigma}$; in the eigenbasis path every eigenvalue
moves by $-\sigma$ and the eigenvectors are unchanged. The present code returns the flow
and the moments in one object (`coupled_moments`, used as both by
`HostHistory.exponential_trial`): the shifted object's moments must not be used as the
flow. The structurally source-free exact chord has no source and is unchanged.

#### 4.6 The harmonic shift

Because $D$ is diagonal, §4.1 holds with one shift per element. The harmonic policy takes
$\sigma(\omega) = m(\omega)\,\sigma$ with $m = 3$ on the third-harmonic band and $m = 1$
elsewhere, so that the harmonic source, rotating at $3\kappa$, is smooth as well; the
moments are evaluated at $hD - m\sigma$ element by element and the samples rotated by
$e^{m\sigma(1-\theta_j)}$, as in §6.2's first form, which already allows a per-element
shift. Every statement of §4.1–4.5 holds element by element, including the exactness of the
step for any $m$; the cap is unchanged.

The difficulty is $m(\omega)$. A frequency boundary does not identify a source's phase:
self-phase-modulated broadening of the fundamental and harmonic generation reach the same
bins, so a bin given $m = 3$ may hold fundamental-band content with a residual $-2\kappa$,
and the reverse. A boundary placed in a spectral gap does not stay in one as the spectrum
develops, and a boundary that moves would have to stay fixed within an accepted step, its
history and its dense output, and be reproducible after a rejection and on resume. The
candidate rules, to be compared under §7.4, are a fixed boundary at $2\omega_0$ (the
prototype's), a fixed boundary at the run's declared band edge between the fundamental and
harmonic bands, and a boundary fixed per run from the launched spectrum's support. The
harmonic policy is admitted only by the gate of §7.4; until it passes, `"harmonic"` exists on
`numpy_ref` only and refuses elsewhere by name.

### 5. Proposed interface

```text
Multistep(carrier_frame="off")     # SEP 16's arithmetic; the default until phase 4
Multistep(carrier_frame="scalar")    # §4.1–4.5, with the step cap
Multistep(carrier_frame="harmonic")  # §4.6; numpy_ref only until it passes §7.4
```

A field of `Multistep`, validated like `formulation`: either shift is refused with the
interaction formulation and equivalent to `"off"` on an envelope grid without the
third-harmonic term (§6.1). The cap $c$ is a module
constant of the policy, not a separate knob. A backend or route that does not yet
implement the policy refuses `"scalar"` by name, as `refuse_formulation` does for
undelivered exponential cases. Whether `"scalar"` becomes the default is phase 4's
decision (§7.3); until then nothing a user does not ask for changes.

### 6. Implementation strategies

#### 6.1 Where the shift comes from

**Rule.** For batch member $b$, $\sigma_b = \mathrm{i}\,\operatorname{Im}x_{b,0,0,k}$,
where $x = \Phi(z_n + h) - \Phi(z_n)$ is the trial's own exponent, column 0 is the
frame's mode (`build_phi` fixes the frame on the first mode; on a coupled route whose
carrier is the reference channel replicated, every column is that carrier, a multiple of
the identity), polarisation 0, and $k$ is the active lattice bin nearest the frame's
reference frequency, ties going to the lower index. The lattice need not contain
$\omega_0$: the offset changes $\sigma$ by about
$\tfrac12|\beta_2|\,\delta\omega^2 h$ (in a co-moving frame its $\beta_1$ cancels the linear
term), which only changes efficiency, since §4.1 is exact for any $\sigma$. If that bin
is inactive the policy falls back to $\sigma = 0$ for the run and records why. Free-space
routes take column 0, the lowest transverse wavenumber; they are admitted only once a
free-space case is qualified (§7) and refuse `"scalar"` until then. A custom `Frame`
changes $\kappa$ and with it $\sigma$; nothing else depends on it.

**Per batch member.** Members may carry different operators, so $\sigma$, the sample
factors and the cap are per member; the step is capped by the largest $|\kappa_b|$.

**Harmonic band.** For `"harmonic"`, $m(\omega)$ is fixed for the run when the problem is
translated, by the rule §7.4 selects, and recorded with the run; it never changes within a
run, so a rejected trial, a dense plane and a resumed history see the same $m$.

**Where the policy is inert.** On an envelope grid without the third harmonic the frame
subtracts $\beta_0$, $\sigma$ would be the table's rounding, and the policy sets
$\sigma \equiv 0$ there, so those runs are the present arithmetic exactly.

**On native.** The exponent is formed in C++, on a device in a device build. The host
forms the one reference element's chord from the operator's own representation (an
expansion or table column, one complex number per member), so no device read-back is
needed; a test checks it against the engine's exponent.

#### 6.2 Applying the rotation

Two equivalent forms:

1. *Rotated samples* — multiply each history row by $e^{\sigma_b(1-\theta_j)}$ as it is
   read into the predictor and corrector sums: one complex multiply per element per
   sample read, fused into the existing sums, with a small per-sample factor array; works
   per batch member and for any later per-element shift.
2. *Complex coefficients* — fold the factor into $c_{jk}$. No per-element work, but every
   compiled kernel's coefficient argument changes type, and it cannot express a
   per-member shift without per-member tables.

The recommendation is (1). The moments take $x - \sigma_b$, one subtraction before the
existing $\varphi$ evaluation.

#### 6.3 Backends

- `numpy_ref` and `numba_cpu` share `HostHistory`, so the trial, the accepted step's dense
  record and its interpolated planes (`_observe_exponential`, shared reference passes)
  change once; `NumbaExponentialKernels` forms its own moments and reads the store by slot,
  so its predict and correct take the shifted exponent and the sample factors.
- Native: the trial and accept entry points take $\sigma_b$ and the factors, and its
  dense observation is a separate implementation (`_ResidentAdams.observe` calling
  `GeneralWorkspace::adams_exponential_observe` in `native/src/general.cpp`) that must
  apply the shifted moments and the prefactor of §4.1 with the **accepted** step's
  $\sigma$, held across later trials and rejections.
- Coupled host paths: the shifted block with the flow by $e^{\sigma}$ (§4.5), in both
  the Schur and eigenbasis paths, and the same in their interpolant.

#### 6.4 The cap

The step proposal is shortened, before the trial, so that
$\max_b|\operatorname{Im}x_{b,0,0,k}| \le c$; a shortened step is not a landing. It costs
one scalar chord per proposal on the host.

#### 6.5 The kernel contract, resume and identity

`carrier_frame` is a field of the integrator carried in `StepperSettings` inside
`KernelProblem`, as `formulation` was (kernel version 20), so phase 1 bumps
`SCHEMA_VERSION`, and a backend built against the older contract refuses the problem
rather than running unshifted arithmetic under the new request. The output layout needs
no new field: the method representation already recorded in the run and in a march
snapshot names the policy. `AdamsMarch._refusal` compares that representation, so a
snapshot taken under another policy, or before this SEP, is refused and the run restarts
its history, as for any method change; there is no conversion.

#### 6.6 Cost

Per trial, one scalar per member and one factor per sample; per element, at most one
complex multiply per sample read; per proposal, one scalar chord. Memory: the factor array
and one complex scalar per member in the dense record. The moment evaluation sees $x -
\sigma$ instead of $x$; `phi_functions` changes algorithm at $|x| = 3$, so its cost can
change. On the two cases measured it did not (no element crossed the threshold either way;
within 5 %), but timing is measured per phase on short windows (§7), and no claim of zero
cost is made before that.

### 7. Verification and validation

#### 7.1 Witnesses and tests

- **Exactness**: a source $F = e^{-\mathrm{i}\kappa z}g(z)$ with $g$ a polynomial of degree
  below the order, $hD$ with imaginary part $-\kappa h$ at the reference: with the policy
  the step is exact to rounding at any $h\kappa$ below the cap; without it, it is not.
  The dense formula against quadrature at interior $\xi$ and at the endpoint.
- **Aliasing and resolution** (§4.3): the mixed witness of §4.3 and a stationary-source
  witness, at fixed steps across $h|\kappa|$ including near-resonant phases, and under the
  adaptive controller from startup, with rejections and variable step ratios, and replayed
  on a fixed schedule; the defect must bound the error wherever the cap allows.
- **Stability** (§4.4): the threshold table reproduced from the recurrence; a resolved
  in-stage collar case and a coupled case run under the policy without the order weight's
  bound being violated in practice (no growth beyond the reference's).
- **Varying operators and coupled sections** (§4.5): a manufactured varying-carrier
  witness with three-level convergence; a non-normal frozen block; changing steps and
  rejections; dense endpoints and interiors; section transitions; hard kinks; the
  source-free chord unchanged; a witness where source and compensation defects cancel.
- **Envelope grids with the third harmonic**: the policy on that grid against the field
  grid for the same physics, both against converged references (not against each other).
- **Harmonic band** (§4.6): exactness with a per-element shift; $m$ unchanged across
  rejections, dense planes and resume; a witness whose fundamental broadening crosses the
  boundary.
- **Routes and the reference rule** (§6.1): a shifted launch (pulse away from the frame
  frequency), a non-fundamental launch on a multimode route, a custom frame, and a
  free-space case before free space is admitted; an inactive reference bin falls back to
  $\sigma = 0$ with its reason recorded.
- **Contract**: older-schema refusal; refusal of `"scalar"` where undelivered, with the
  interaction formulation, and of `"harmonic"` off `numpy_ref` before the gate; a snapshot under
  another policy refused; saved and in-memory continuation under the same policy agreeing
  within SEP 25's bounds for matched methods, schedules, orders and histories.
- **Backend parity**: `numba_cpu` and native against `numpy_ref` under SEP 25's rounding
  bounds, with a rounding budget that includes the rotation factors, the phase evaluation,
  the subtraction $x - \sigma$ and the interpolation conditioning, derived in phase 1
  rather than assumed equal to the present bounds.

#### 7.2 Preliminary measurements (motivation, not qualification)

A `numpy_ref` prototype of §4 (the scalar shift and the cap) on the development tree at
`174be2a`, on one contended Darwin/arm64 workstation (wall times not compared). Each case is
`Simulation.from_capillary(radius, length, gas, pressure, grid=FieldGrid.from_limits(length,
λ0, limits, window), lambda0=λ0, fwhm, energy, atol=1e-10, noise=None, rtol=1e-6)`, other
settings at their defaults (temporal collar on, its default scheme):

| case | radius, gas, pressure, length | λ0, fwhm, energy, modes | limits, window | responses |
|---|---|---|---|---|
| A | 256 µm He, steady fill 0 → 0.2 → 0 bar, 4.2 m | 800 nm, 6 fs, 1 mJ, 1 | 90 nm–1.6 µm, 60 fs | defaults |
| B | 125 µm He, 0.8 → 0 bar, 1.5 m | 800 nm, 7.5 fs, 275 µJ, 4 | 90 nm–4 µm, 400 fs | defaults |
| C | 16 µm H₂, 40 bar, 0.1 m | 515 nm, 400 fs, 10 µJ, 1 | 250 nm–2 µm, 2 ps | Raman (294 K), plasma |
| D | 20 µm H₂, 5 bar, 1 m | 1030 nm, 40 fs, 4 µJ, 1 | 400 nm–4 µm, 1.2 ps | Raman, no loss, no plasma |

Source evaluations and final-field relative $L^2$ error against IP-DOPRI5 at
`rtol = 1e-9` on `numba_cpu` (for C, that reference agrees with the interaction
formulation at `rtol = 1e-8` to $7.9\times10^{-9}$; for the others its convergence is not
yet shown):

| case | $\kappa$, rad/m | `off` | `scalar` (capped) | interaction | IP-DOPRI5 |
|---|---|---|---|---|---|
| A | −12 | 1128 / 1.4e-6 | 932 / 1.1e-6 | 1644 / 7.5e-7 | 2268 / 4.4e-6 |
| B | −48 | 1805 / 1.6e-6 | 1742 / 2.5e-6 | 10475 / 3.1e-7 | 12397 / 2.5e-6 |
| C | −5470 | 2669 / 5.8e-6 | 582 / 6.4e-6 | 1401 / 2.3e-6 | 798 / 3.0e-7 |
| D | −2420 | 26060 / 1.0e-4 | 18910 / 7.6e-5 | 17032 / 6.6e-5 | 27979 / 3.2e-5 |

On C at matched error: 582 against 2669 evaluations near $6\times10^{-6}$; at `rtol = 1e-7`
the policy takes 649 evaluations to $1.1\times10^{-6}$, where `off` needs between 2669
($5.8\times10^{-6}$) and 3918 ($3.1\times10^{-7}$), about 3300 by log interpolation. On A,
B and D the cap never binds. On D the policy closes only a third of the gap to the
envelope grid; the soliton's shift away from $\omega_0$ adds rotation no scalar removes.
On B the error rose from $1.6$ to $2.5\times10^{-6}$ at fewer calls; matched-error
comparison there is part of phase 4. These figures are full-length runs made to
understand the problem; qualification uses the short windows of §7.3.

**A loose-tolerance failure.** Case E is the suite's `argon_blueshift`: 13 µm Ar at
1.7 bar, 0.4 m, 800 nm, 30 fs, 3 µJ, one mode, no loss, 300 nm–4 µm, 400 fs window, with
the default responses including the plasma; $\kappa = -4412$ rad/m. Error against
IP-DOPRI5 at `rtol = 1e-8` on `numba_cpu`:

| `rtol` | `off` | `scalar` | `scalar` (capped) | `harmonic` | `harmonic` (capped) | IP-DOPRI5 |
|---|---|---|---|---|---|---|
| 1e-3 | 3909 / 0.97 | 2092 / 5.9e-2 | 2312 / 4.7e-2 | 2857 / 0.21 | 3100 / 5.1e-2 | 2625 / 7.2e-2 |
| 1e-4 | 6741 / 9.5e-3 | 4596 / 8.8e-3 | 4596 / 8.8e-3 | 5162 / 2.2e-2 | 5413 / 1.7e-3 | 4949 / 9.9e-3 |

`off` gives the same 0.97 on `numba_cpu` and native. It is not aliasing ($h|\kappa|$ is at
most 0.56) and not an under-reading estimate: the order stays at 2 (3 at `1e-4`), the defects
are level from the input on, nothing is rejected, and about 3900 honestly estimated steps of
a few parts in $10^4$ each accumulate; a self-compression near 8 cm amplifies the
accumulated error of every method about 4.5 times. With the scalar shift the order sits at 1
(§4.3), also without the plasma (at `rtol = 1e-3`: 4931 steps and 0.18 against a
plasma-free reference, where IP-DOPRI5 takes 688 steps to 0.25). The uncapped harmonic shift is worse than the scalar
one; capped, it is the most accurate at `1e-4` but takes about 20 % more evaluations.

**Envelope grid with the third harmonic.** Case D's physics with Kerr only (no Raman) over
4 cm on that grid, against IP-DOPRI5 at `rtol = 1e-9`, the shifts with the cap:

| `rtol` | `off` | `scalar` | `harmonic` | IP-DOPRI5 |
|---|---|---|---|---|
| 1e-4 | 230 / 7.7e-5 | 117 / 7.1e-6 | 103 / 5.1e-7 | 203 / 4.3e-5 |
| 1e-6 | 500 / 8.0e-7 | 383 / 4.5e-8 | 103 / 1.0e-8 | 385 / 1.8e-8 |

The harmonic run's 103 evaluations are the cap's ($4\,\text{cm}\times|\kappa| \approx 97$ rad).

Shifts measured and not proposed: one per mode from each mode's own $\beta_0$ (B: 2522
evaluations, against 1742, because the higher modes' sources are driven by the
fundamental); one per element estimated from the two newest samples (B: 3009 with 260
rejections).

#### 7.3 Qualification and the default decision (phase 4)

On short windows as `AGENTS.md` requires — about ten steps, or a propagation that
finishes in seconds — never full-length runs:

- **Cases**: the benchmark suite's `ctc_reduced`, `h2_raman_6ps`, `rdw`,
  `he_rdw_averaged` and `argon_blueshift` capillaries, a case like A, and a cheap-source
  GNLSE field-grid case where the exponential formulation currently leads or loses.
- **Windows**: a launch window from the input plane, and a developed-spectrum window
  started from a plane written once by a converged reference run and propagated a short
  distance, so that the difficult physics (broadened spectra, harmonic and Raman bands) is
  in the window.
- **References**: IP-DOPRI5 three orders tighter than the tightest tolerance compared,
  agreeing with the interaction formulation at a tolerance two orders tighter to within a
  tenth of the smallest error compared; a reference that fails this is replaced, not used.
- **Error measures**: worst-plane relative field error, and named weak-band errors (each
  case's declared observable bands, such as the ultraviolet and Raman sidebands), each with
  a budget fixed before measurement: the policy's weak-band error at matched worst-plane
  error must not exceed `off`'s by more than a factor of 2, and must be below $10^{-2}$
  relative in each named band at the tolerances compared.
- **Loose tolerance**: case E's failure mode at `rtol = 1e-3` on a developed-spectrum
  window that includes its self-compression, where accumulated error is visible in a short
  propagation.
- **Matched work**: work–precision curves over `rtol` from $10^{-3}$ to $10^{-8}$; compare
  calls and time at worst-plane errors from $10^{-3}$ to $10^{-6}$ by interpolation.
- **Default criterion**: `"scalar"` becomes the default only if, on every case and window,
  it needs no more than 1.1 times `off`'s calls and time at matched worst-plane error and
  meets the weak-band budget; timings on `numba_cpu` and native CPU, one thread, median of
  five samples, with a 10 % non-regression margin. Otherwise the policy stays opt-in and
  the manual says where it helps.

Short-window timings are not extrapolated to full runs; a full-size demonstration, if
wanted, is a worked example.

#### 7.4 The harmonic gate

`"harmonic"` is promoted to the compiled backends, and considered for the default, only if,
on the cases and windows of §7.3 including a self-phase-modulation-dominated one (case C)
and one with a developed harmonic (case D on the envelope grid with the third harmonic):

- each band named in §7.3 has its own converged reference and an error budget fixed before
  measurement: at matched worst-plane error the harmonic policy's band error is at most the
  scalar policy's, and below $10^{-2}$ relative;
- at matched worst-plane error it needs no more than 1.1 times the scalar policy's calls on
  any case, and fewer on at least one;
- the boundary rule chosen (§4.6) passes the crossing-broadening witness, and a run whose
  spectrum moves across the boundary meets the band budget;
- with the cap, which the measurements of §7.2 show it needs.

If it fails, `"harmonic"` is removed and the scalar policy stands alone.

### 8. Documentation

[The Adams methods](../../guide/numerics/adams.md): the policy in *The exponential
formulation*, with the shifted step, the dense formula, the cap, the resolution and
stability conditions of §4.3–4.4, and *Which formulation suits which source* revised.
[Choosing a stepper](../../guide/numerics/choosing-a-stepper.md): the phase 4 results.
[The linear operator](../../guide/numerics/linear-operator.md): a pointer from the grid
table to the policy. Docstrings of `Multistep`, `ExponentialKernels` and
`PEC_REAL_INTERVAL` (its residual-phase condition).

### 9. Compatibility and impact

Phases 1–3 change nothing unless `carrier_frame="scalar"` is asked for. The kernel
contract's `SCHEMA_VERSION` is bumped in phase 1; older compiled adapters refuse rather
than run the old arithmetic. The method's representation changes for every `Multistep`,
so a march snapshot taken before this SEP is refused on resume and the history restarts;
no output layout changes. Envelope grids without the third harmonic are unchanged under
any policy; envelope grids with it are treated as field grids. If phase 4 makes a shift the
default, field-grid trajectories change: results stay within the accuracy envelopes
qualified against converged references (SEP 25), which `rtol` alone does not guarantee,
and recorded call counts, timings and pinned budgets are re-measured. Luna.jl's steppers
are not affected; no difference to record.

### 10. Alternatives considered

- **The interaction formulation as the field-grid default.** It wins on the dense
  molecular cases but loses on the helium capillaries, by 1.5 times on case A and 5.8
  times on case B.
- **Choose the formulation per run from $\kappa$.** A heuristic switch between two
  numerics with a threshold to qualify; the policy needs no threshold beyond its cap.
- **Subtract $\beta_0$ on field grids.** Changes the carrier-envelope phase and so the
  physics.
- **Store the history in a rotating frame.** Equivalent per step, but the stored rows
  would carry an absolute phase $\int\kappa\,dz$ that resume and gradients must track.
- **Lower order.** Measured worse (§2).
- **The harmonic shift as the only policy.** Uncapped with a boundary at $2\omega_0$ it
  took 757, 1210 and 13895 evaluations on A, B and D but 584 on C, whose
  self-phase-modulated content above $2\omega_0$ follows the carrier, and on E it was worse
  than the scalar shift unless capped. It is developed beside the scalar policy and gated
  (§4.6, §7.4), not adopted alone.
- **A shift with a real part (loss).** With a lossy reference, a bin with $hD = 0$ needs
  moments at $-\sigma$ with a large positive real part and exponentially small sample
  factors; evaluated separately they overflow and underflow although the integral is
  finite, and `phi_functions` refuses real arguments above 700. Not proposed.
- **No cap.** 477 rather than 582 evaluations on C, but twice the error and the aliasing
  exposure of §4.3.

### 11. Phases

1. **`numpy_ref`, opt-in.** The scalar and harmonic policies on field grids and envelope
   grids with the third harmonic, for the diagonal, separable and varying operators with
   compensation, the coupled constant-Schur and frozen-residual sections, the cap and the
   dense formula; the `carrier_frame` field, its refusals and the `SCHEMA_VERSION` bump;
   every witness and test of §7.1 on `numpy_ref`; the rounding budget derived; the manual
   pages. Free space is refused until qualified here or later. **Interim manual note**, in
   [the Adams methods](../../guide/numerics/adams.md#limitations) and
   [choosing a stepper](../../guide/numerics/choosing-a-stepper.md): on field grids (and
   envelope grids with the third harmonic) in dense gases, the default `Multistep()` at a
   loose tolerance can accumulate a large error over many steps (case E); use
   `RungeKutta()` or `rtol <= 1e-4` there, or `carrier_frame="scalar"` once delivered on the
   backend in use.
2. **`numba_cpu`.** Shifted exponent and sample factors in its exponential kernels, for
   the scalar policy; parity to the phase 1 budget; short-window timings.
3. **Native.** Trial, accept and dense observation on CPU with the host-formed shift;
   parity and timings; then device builds under the standing GPU policy.
4. **Qualification and the default.** §7.3 on `numba_cpu` and native CPU, the decision on
   the default against its criterion, re-measured manual ranks and pinned budgets; the
   harmonic gate (§7.4) on `numpy_ref`, and, if it passes, the harmonic policy on
   `numba_cpu` and native in this phase, with the same parity and timing requirements, or
   its removal if it fails.

Each phase is independently mergeable: the default is `"off"` until phase 4, and a backend
without the policy refuses it by name.

### 12. Open questions

- **Q1 — per-band shifts.** In scope behind a gate (§4.6, §7.4): which boundary rule,
  settled in phase 4; the policy is removed if no rule passes.
- **Q2 — loss in the shift.** Not proposed (§10).
- **Q3 — envelope grids.** Resolved: without the third harmonic the policy is inert
  ($\sigma \equiv 0$); with it, the grid's operator is the field grid's and the policy
  applies unchanged (§4.2).
- **Q4 — the switch.** Resolved: a public `carrier_frame` field, `"off"` by default until
  phase 4 (§5).
- **Q5 — the reference on multimode, coupled and free-space routes.** The rule is fixed
  (§6.1); whether the fundamental's shift serves a run whose energy is not in the frame's
  mode is measured in phase 1 (non-fundamental launch) and phase 4. Free space waits for
  its own qualification.
- **Q6 — the cap's value.** 1 rad measured on one case; whether it should depend on the
  order, settled in phase 1 from the witnesses and in phase 4 from the suite.
- **Q7 — the stability heuristic.** Whether the order weight needs a residual-phase
  condition (§4.4) beyond the present one, settled by phase 1's collar and coupled cases.

### 13. References

1. S. M. Cox and P. C. Matthews, "Exponential time differencing for stiff systems,"
   *J. Comput. Phys.* **176**, 430 (2002). doi:10.1006/jcph.2002.6995
2. M. Hochbruck and A. Ostermann, "Exponential multistep methods of Adams-type,"
   *BIT Numerical Mathematics* **51**, 889 (2011). doi:10.1007/s10543-011-0332-6

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).*

### Phase 1 — `numpy_ref`, opt-in (2026-10-25)

**Delivered.**

- `solprop/stepper/multistep.py`: `CarrierFrame` and the `Multistep.carrier_frame`
  field (`"off"` by default), validated like `formulation` (a shift with the
  interaction formulation is a `StepperError`, so `ComparisonMultistep` can only be
  `"off"`); a `-scalar-frame` / `-harmonic-frame` suffix on the record name;
  `CARRIER_PHASE_CAP = 1.0`; the residual-phase condition of §4.4 in the docstring of
  `PEC_REAL_INTERVAL`.
- `solprop/translate/problem.py`: `SCHEMA_VERSION = 36` with its version note; the
  native binding reports 36 (`native/src/bindings.cpp`) and the frozen literals are
  rewritten.
- `solprop/backends/exponential.py`: `CarrierReference` (the run's reference bin, its
  frequency, the harmonic multiple and the reason), `carrier_reference` (the rule of
  §6.1), `HARMONIC_BOUNDARY` and `refuse_carrier_frame`, called beside
  `refuse_formulation` on all three backends: `numpy_ref` delivers both policies,
  `numba_cpu` and native refuse either by name, and every spatial geometry is refused.
- `solprop/backends/multistep.py`: the shifted trial in `HostHistory` (moments at
  `x - sigma`, the rows read rotated into a compact buffer by `_rotated`, the store
  untouched); `coupled_moments(..., carrier=sigma)`, whose flow is `exp(sigma)` times
  the shifted block's `phi_0` in both the augmented and eigenbasis paths;
  `_Compensation.rotate` for the untransported witness; the dense formula of §4.1 with
  the accepted step's `sigma`; the cap in `AdamsMarch._capped`. The prototype's
  environment switches are gone.
- Tests: `tests/backends/test_carrier_frame.py` (73 cases), the shifted reference march
  in `tests/exponential_scalar.py` with two witnesses in
  `tests/stepper/test_exponential_scalar_march.py`, and `carrier_increment_budget` in
  `tests/bounds.py`.
- Manual: [the carrier frame](../../guide/numerics/adams.md#the-carrier-frame), the
  carrier-following source in *Which formulation suits which source*, the interim note
  in [limitations](../../guide/numerics/adams.md#limitations) and in
  [choosing a stepper](../../guide/numerics/choosing-a-stepper.md), and a pointer from the
  grid table of [the linear operator](../../guide/numerics/linear-operator.md#which-terms-the-frame-removes).
  Design: `docs/dev/stepper.md`, the capability table of `docs/dev/backends.md`, the
  integrator row of `docs/dev/kernel-contract.md`. `CHANGELOG.md`.

**Deviations from Part I, and why.**

1. *The reference frequency* is the grids' `omega0`, not the `Frame`'s own: the kernel
   problem does not carry the frame's reference. The two coincide for every co-moving
   frame; a custom frame with another reference changes only how much rotation the
   shift removes (§4.1 is exact for any `sigma`), which a test runs (a frame 5 % above
   the grid's: 294 calls, as with the co-moving frame).
2. *Recording.* The effective policy (the reference bin and its frequency, the reason
   for an inert run, the harmonic rule) is logged at INFO when the march is built and
   held on the engine (`HostHistory.carrier`); no output field is added, as §6.5
   foresaw. The harmonic boundary is the module constant `HARMONIC_BOUNDARY`, so the
   code version and the method's representation fix it.
3. *Separable operators* exist only on the spatial routes, which phase 1 refuses with
   free space (radial, Cartesian and line, planar). The separable case of §11 therefore
   waits for the free-space qualification.
4. *The coupled interpolant.* Interpolated planes are refused beside a coupled flow for
   every stepper, so §6.3's "and the same in their interpolant" has nothing to change.
5. *The cap* is one chord of the reference bin's exponent over the part of the proposal
   inside the run, scaled once; a replayed schedule is not capped (its steps are given).
   On a varying carrier a capped step can exceed one radian by the chord's error.
6. *The native schema literal.* Bumping the kernel schema moves `bindings.cpp`'s
   reported version; no native arithmetic changes, and the adapter refuses either shift
   by name. The native extension was not built locally; CI builds it.
7. *The crossing-broadening witness* (§7.1, harmonic band) is a manufactured source
   whose whole content follows the carrier, so every bin at or above `2 omega_0` holds
   fundamental-rate content under `m = 3`; a propagation whose spectrum moves across the
   boundary is left to the phase 4 gate (§7.4), which needs one with per-band references.

**Decisions.**

- *Q6, the cap's value.* One cap for every order. On the mixed witness the defect
  exceeds the error by 2.8 (q = 1) to 13 (q = 6) at `h|kappa| = 1`, and by 5.9 to 27 on
  the stationary one; the mixed defect first falls below its error at `h|kappa|` = 2.05
  (q = 1) to 2.5 (q = 6). An order-dependent cap would buy nothing measurable.
- *Q7, the stability heuristic* (partly). On a 3 mm window whose in-stage collar binds
  the step, the policy shows no growth: its field meets the accuracy envelope and its
  energy does not exceed the reference's. It took 842 calls against 754 unshifted there,
  so the order weight's interaction with the shifted residual phase is a cost question
  for phase 4, not a stability failure. No residual-phase condition was added.
- *Q5, a non-fundamental launch* (partly). All of a two-mode run's energy launched in the
  second mode: the policy meets its envelope in 757 calls against 849 unshifted.

**The parity rounding budget** (for phases 2 and 3; `bounds.carrier_increment_budget`).
Elementwise, for the increment `sum_k M_k S_k`, `S_k = sum_j c_jk G_j`, with `u` the unit
roundoff and `n` the nodes: a rotated sample `G_j = exp(sigma (1 - theta_j)) F_j` is
within `|F_j| r_j`, `r_j = (u |sigma| + d) |1 - theta_j| + 6 u` (the argument's product,
three correctly rounded real functions, the multiply, and the phase uncertainty `d` with
which a backend forms `sigma`); `S_k` within `sum_j |c_jk| |F_j| (r_j + gamma_n + n^2 u)`,
the last term the Lagrange coefficients' own formation; a moment within 256 ulp of its
natural scale `E / (k + 1)`, `E = exp(max(Re x', 0))`, plus the argument's
`u |x'| + d` carried by `|M_k'| <= E / (k + 1)`; and the outer sum within
`gamma_n sum_k |M_k| |S_k|`. The rotation has unit modulus, so `|G_j| = |F_j|` and the
interpolation's conditioning is the unshifted formulation's. Two implementations of a
matched step differ by at most twice the bound; a run's budget adds the per-step bounds
carried by the lossy flow. Against a 40-digit evaluation of one increment (five irregular
nodes, `|Im x|` to 40, `sigma = 0.9i`) every element is inside it. A compiled backend
whose `sigma` is formed from its own exponent passes that exponent's rounding as `d`.

**Verification run.**

- `uv run pytest -n 8 tests/backends/test_carrier_frame.py
  tests/stepper/test_exponential_scalar_march.py`: 101 passed, 1 skipped (native is not
  installed here), 27 s. The existing Adams and exponential modules
  (`tests/stepper/test_multistep.py`, `tests/backends/test_exponential_{march,varying,coupled,weights}.py`,
  `tests/backends/test_multistep_march.py`, `tests/backends/test_driver.py`,
  `tests/translate/test_problem.py`): 525 passed, 1 skipped, after the two pinned
  representations were updated. `tests/test_seps.py` passed.
- `ruff check`, `ruff format`, `basedpyright` and `poe docstrings` on the changed files;
  doctests of the two changed modules; `codespell`; `poe docs`; the public-tree scan.
- Not run locally, by the working agreement: the full gate, the slow and coverage lanes
  and the native gate, which run in CI on the SEP's pull request. No GPU and no
  `numba_cuda`.

**What the tests establish** (all on `numpy_ref`).

- *Exactness through the march itself*: a manufactured source `p exp(i rho z)` at the
  frame's own rate (and, for `"harmonic"`, at `m(omega)` times it) is stepped to
  3e-16 – 8e-16 at every plane, the interpolated ones included, against 2e-10 – 1e-9
  unshifted at the same `rtol`; the exact steps grow until the cap holds them at
  `h|kappa| = 1.000`, with no restart, in 19 calls against 37.
- §4.3's table reproduced to its two figures; aliasing blinds the defect in both
  formulations; under the cap the defect bounds the error at every order (Q6).
- §4.4's threshold table and the review's two spectral radii, from the recurrence with
  production weights.
- Adaptive marches from startup on a third-harmonic remainder and on a stationary part,
  with variable steps, replayed on their own accepted steps, within the envelope; a run
  that rejects and recovers.
- The inert envelope grid bit for bit equal to `"off"`; the inactive-bin fallback with
  its reason.
- Short physical windows (3 mm of 40 bar hydrogen, 2 uJ, 30 fs, Kerr only, a weak
  collar; Runge–Kutta at `rtol = 1e-11` as each window's reference), at `rtol = 1e-8`:

  | window | `"off"` calls / error | `"scalar"` calls / error |
  |---|---|---|
  | field grid | 376 / 1.4e-9 | 294 / 6.7e-9 |
  | envelope grid with the third harmonic | 375 / 4.8e-10 | 268 / 2.9e-9 |
  | field grid, the window's own in-stage collar | 754 / 1.1e-9 | 842 / 3.3e-9 |
  | launch 40 bins red of the reference | 374 / 8.9e-10 | 297 / 6.1e-10 |
  | launch 40 bins blue of the reference | 653 / 3.1e-9 | 701 / 2.3e-9 |
  | two modes, launched in the second | 849 / 1.6e-9 | 757 / 1.5e-9 |
  | custom frame 5 % above the grid's reference | 377 / 1.4e-9 | 294 / 6.6e-9 |
  | gas gradient 40 → 20 bar (compensated) | 754 / 9.1e-10 | 804 / 2.8e-9 |

  Coupled sections at `rtol = 1e-9`, against Runge–Kutta at 1e-11: the scalar wall 13
  and 12 calls (2.6e-12, 8.9e-13), the taper 65 and 63 (1.4e-10 both). These are equal
  tolerances, not matched errors, and one machine: they are not the phase 4 comparison.

**Sources.** Nothing was transcribed: the shifted step, the dense formula, the coupled
identity and the budget were re-derived and checked against quadrature, augmented
exponentials and a 40-digit evaluation. The two references of §13 were checked by the
AI review; they were not re-read for this phase.

**Limitations.**

- `numpy_ref` only; the free-space routes, and so the separable representation, are
  refused.
- The harmonic boundary is the prototype's fixed `2 omega_0`, pending §7.4.
- At equal `rtol` the policy's error on the short windows is up to five times the
  unshifted one's, and it takes more calls under a binding in-stage collar, on a blue
  launch and on the gradient window; whether it is cheaper at matched error is phase 4's
  measurement.

**Open questions.**

1. Q7 remains open for the cost side: why the in-stage collar's window takes 12 % more
   calls under the policy (the order weight against the shifted residual phase, or the
   orders chosen), measured in phase 4.
2. Q5 for a launch far from the reference: the blue launch took 7 % more calls; whether
   a reference that follows the launch's own carrier is worth a deviation from §6.1.
3. Whether the cap should follow the varying carrier to more than one chord (deviation 5).

**Handoff.** Phase 2 (`numba_cpu`): give `NumbaExponentialKernels`' predict and correct
the shifted exponent and a per-slot factor array (the rotated-sample form, fused into
the row reads); the dense observation is the shared host code and needs nothing; the
coupled path already takes `shift=` (its signature exists, the backend refuses until
then). Lift the refusal for `"scalar"` only, check parity against `numpy_ref` with
`carrier_increment_budget` (one step, eight steps, a short window) and record ten-step
timings. Phase 3 (native): the host forms `sigma_b` from the operator's representation,
the trial and accept entry points take it and the factors, `adams_exponential_observe`
takes the accepted step's `sigma`, with parity, timings and then devices. Phase 4: §7.3
and §7.4 on the suite's short windows.

### Phase 2 — `numba_cpu` (2026-10-25)

**Delivered.**

- `solprop/backends/numba_cpu/multistep.py`: the compiled moments pass takes `shifts`
  per flattened row and forms `phi(x - sigma)` with `exp(x)` beside it; the
  exponential predict and correct passes take one rotation factor per slot and row and
  multiply each history row by it as they read it (§6.2's rotated-sample form, fused);
  the coupled compiled passes rotate their few rows beforehand (`_rotated_rows`), and
  `coupled_moments` takes the shift on the compiled eigenbasis route too.
- `solprop/backends/multistep.py`: the reference passes take the same interface
  (`moments(..., shift=)`, `predict(..., factors=)`, `correct(..., factors=)`), so
  `HostHistory` has one path for every backend; phase 1's pre-rotated copy of the rows
  (`_rotated`) is replaced by `_carrier_factors` and the passes' own row reads.
- `solprop/backends/numba_cpu/backend.py`: `numba_cpu` delivers `"scalar"` and refuses
  `"harmonic"` by name, pending §7.4.
- Tests in `tests/backends/test_carrier_frame.py`, parametrised over the compiled
  backends: one trial's moments, flow and predictor against the reference within the
  phase 1 budget (`carrier_increment_budget`), with a check that the factors are
  applied; exactness of the manufactured source through the compiled march; a matched
  window and a matched gas gradient (the reference's own accepted steps, fixed order);
  dense planes; and coupled sections.

**A defect found and fixed before it shipped.** `NumbaExponentialKernels.flow` returns
the exponential its moments pass formed, whatever argument it is given. Phase 1's host
path called `moments(x - sigma)` and then `flow(x)`, which on `numba_cpu` would have
been the shifted flow `exp(x - sigma)`: harmless while `numba_cpu` refused the policy,
wrong once delivered. The moments pass now receives `x` and the shift separately, so the
flow is always `exp(x)`.

**Deviation.** Both compiled backends form `exp(x - sigma)` as `exp(x) exp(-sigma)`, one
complex product per element, where the reference takes a second exponential. The
difference is a few ulp, inside the moments' 256-ulp share of the budget; it took
native's Kerr-only per-step ratio from 1.076 to 1.057.

**Verification.** Matched comparisons against `numpy_ref` (relative, the largest plane):
a 3 mm dense-hydrogen window with interpolated planes 3.3e-15 in 294 calls on both; the
40 → 20 bar gradient 1.1e-14 in 275; the coupled scalar wall 5.6e-16 in 9 and the taper
2.3e-15 in 48. The Adams, exponential, coupled, resume and compiled-pass modules pass
(207 passed, 3 skipped). Ten-step timings in
[benchmarks](../benchmarks.md#adams-carrier-frame-per-step): 1.04 per step on a
Kerr-only source, 1.01 with Raman.

### Phase 3 — native CPU (2026-10-25)

**Delivered.**

- `native/include/solprop_native/core.hpp`: `ExponentialWeights` carries a scalar
  carrier frame by value (`shifted`, `shift`, `exp(-sigma)`, and the factors' real and
  imaginary parts, one per history row); a native state holds one batch member, so one
  `sigma` a trial.
- `native/src/general.cpp`: `adams_moment_phi` (moments at `x - sigma`, flow `exp(x)`)
  and `adams_history` (a history row rotated as it is read) at every exponential site:
  the elementwise moments beside a coupled flow, the block functions on host and device
  (the flow `exp(h B)` kept, each eigenvalue shifted for the moments), the predictor and
  corrector, both coupled passes, and both defect-norm reducers. Dense observation takes
  the accepted step's `sigma` and factors: the moments at `xi (x - sigma)`, the rotated
  samples and `exp(-sigma (1 - xi))`.
- `native/src/bindings.cpp`, `_core.pyi`: the trial and the observation take a packed
  `carrier` array (empty for none).
- `native/src/solprop_native/backend.py`: `_ResidentAdams` holds the run's
  `CarrierReference`, forms `sigma` from its chord and the factors on the host, and holds
  the accepted step's across later trials and rejections; CPU builds deliver `"scalar"`,
  device builds and `"harmonic"` are refused by name.
- `solprop/backends/exponential.py`: `CarrierChord`, the operator's own representation
  (constant, expansion or z table) at the reference element alone, for every batch
  member. The step cap now reads it on every backend instead of the full exponent, one
  scalar a proposal (§6.4's cost); a test checks it against each backend's own exponent
  for a constant operator and a gradient's expansion.

**Deviations.**

1. *The source of native's `sigma`.* §6.1 asks for the trial's own exponent; native forms
   that in C++, so the host takes the same element from the chord (§6.1's "On native"
   paragraph). The two agree to rounding (tested within 64 ulp of the exponent).
2. *Device builds are refused*, not qualified: the kernels are written for both host and
   device variants, but no device build was run (that needs a rental, under the standing
   GPU policy and a budget agreed first).

**Verification.** Matched comparisons against `numpy_ref`: the window 1.4e-15, the
gradient 3.5e-15, the scalar wall 5.6e-16 and the taper 2.1e-15, each in the reference's
calls; dense planes inside steps agree to rounding. The native package's own suite passes
(384 passed, 11 skipped) with its schema literals updated; the solprop modules that run
native (resident Adams, coupled, resume history, the carrier frame) pass. The native
extension was built from this tree (`uv sync --extra native --reinstall-package
solprop-native`); its full gate (`poe check-native`: clang-tidy, sanitisers, the wheel)
was not run locally and runs in CI. Ten-step timings: 1.06 per step on a Kerr-only
source, 1.01 with Raman.

**Open questions.**

1. Device qualification: a short GPU segment (one NVIDIA, one AMD device) running the
   compiled parity tests and the ten-step timing, under the standing policy.
2. The per-step overhead on cheap sources (4–6 %) counts against the policy in phase
   4's 1.1× criterion wherever it saves few calls; fusing the rotation into the
   divided-difference rows of the defect reducers is the remaining lever.

**Handoff.** Phase 4: §7.3's qualification on the suite's short windows on `numba_cpu`
and native CPU, the default decision, and §7.4's harmonic gate on `numpy_ref`.

### Phase 4 — qualification and the default (2026-10-25)

**Decision.** `"off"` stays the default and `"scalar"` stays opt-in: it fails §7.3's
criterion on the weak-band budget and, on native CPU, on time. `"harmonic"` fails the
gate of §7.4 and is removed.

**What was run** (fixed before measurement, as §7.3 and §7.4 require).

- *Cases*, every one a field grid: the suite's `ctc_reduced` (two modes, Raman),
  `h2_raman_6ps`, `rdw` (four modes), `he_rdw_averaged` and `argon_blueshift`; a case-A
  capillary (§7.2's A, helium steady fill, plasma on); and, for the cheap-source case,
  case A with Kerr alone (deviation 2). For the gate: §7.2's case C on its field grid,
  and case D's physics with Kerr alone on the envelope grid with the third harmonic.
- *Windows*: the launch window, from the input plane to the end of the suite's entrance
  schedule (1.3 mm to 39 mm; 40 mm for case A, 2 and 40 mm for the gate's C and D),
  and a developed window started from a plane written once and propagated 3 to 40 mm:
  `ctc_reduced` 50–55 mm, argon 60–100 mm (through its self-compression),
  `h2_raman_6ps` 0.50–0.52 m, both RDW cases 0.60–0.64 m, case A 2.00–2.04 m, C
  50–53 mm, D 0.30–0.34 m. Five planes a window.
- *References*: IP-DOPRI5 at `rtol = 1e-11` on `numba_cpu`, three orders tighter than
  the tightest tolerance compared, each checked against the interaction formulation at
  `1e-10`. Every worst-plane check passed: agreement from 1.5e-12 to 6.9e-9 (argon's
  developed window), below a tenth of every error compared at the matched errors used.
- *Runs*: `"off"` and `"scalar"` at `rtol` 1e-3 to 1e-8 on `numba_cpu` and native CPU,
  one thread, five interleaved samples each (median); the gate on `numpy_ref` with the
  three boundary rules of §4.6: 2 omega_0, the run's requested band edge (2.06 omega_0
  for C, 2.58 for D), and the launched spectrum's support (its upper edge at 1e-6 of
  the peak power, reflected about 2 omega_0: 2.99 and 2.75).
- *Errors*: worst-plane relative field error; each window's weak bins (1e-9 to 1e-3 of
  the reference's final peak power) and its named band (the blue side, 250–400 nm, for
  the comb-to-continuum; 300–600 nm for argon; the red-shifted soliton band,
  1.3–2.5 µm, for `h2_raman_6ps`; the ultraviolet, 100–150 nm, for both RDW cases and
  100–200 nm for case A), as the band-limited difference over the band's largest
  reference norm (floor 1e-3), worst plane.

**Results: the default criterion.** Calls and time at matched worst-plane error as a
ratio to `"off"` (the range over the matched errors and both windows), and the largest
ratio of the weak or named band error at matched worst-plane error:

| case | calls | time, `numba_cpu` | time, native | band error ratio |
|---|---|---|---|---|
| `ctc_reduced` | 0.48–0.69 | 0.49–0.70 | 0.49–0.71 | 394 (blue, launch); 5.4 (developed) |
| `h2_raman_6ps` | 0.34–0.73 | 0.35–0.75 | 0.38–0.77 | 484 (red, launch); 58 (weak) |
| `argon_blueshift` | 0.57–1.08 | 0.58–1.09 | 0.59–1.13 | 5.3 (blue, launch) |
| `he_rdw_averaged` | 0.82–1.04 | 0.88–1.13 | 0.96–1.29 | 3.3 (ultraviolet, launch) |
| `rdw` | 0.87–1.01 | 0.88–1.02 | 0.97–1.15 | 1.7 |
| case A | 0.93–1.03 | 0.98–1.08 | 1.04–1.16 | 1.2 |
| case A, Kerr only | 0.93–1.03 | 1.02–1.13 | 1.20–1.33 | 1.2 |

The criterion asked for at most 1.1× in calls and time and at most 2× and below 1e-2 in
every band: the calls pass everywhere (argon's developed window reaches 1.08 at 1e-6);
the time fails on native for both helium RDW cases and both case-A variants and on
`numba_cpu` for the averaged RDW and the Kerr-only case; and the band budget fails on
five windows. The weak-band loss is the finding that matters: at the worst-plane error
at which the default resolves the comb-to-continuum launch's blue side to 3.1e-5 the
shift leaves it at 4.3e-3, and the self-frequency-shift launch's red band at 0.11
against 9.5e-4. The source of a weakly populated, driven bin does not follow the
carrier, so the shift gives it a residual rotation, and the step-size norm, which the
populated bins dominate, does not see it. The loose-tolerance failure of case E is
mended on its developed window: at `rtol = 1e-3`, 3.9e-3 in 353 calls against the
default's 2.0e-2 in 538, on both backends.

**Results: the harmonic gate.** Case C's windows are cap-bound (21 and 26 calls for
every policy): its only bins above 2 omega_0 are the 250–257 nm edge of its grid, where
self-phase-modulated fundamental light lies, and the 2 omega_0 and band-edge rules left
that band 5 to 8 times worse than the scalar shift (0.5 against 0.06; that band's
reference agreed only to 7e-3, so the comparison is indicative). On D's developed window
those two rules took 0.72–0.96 of the scalar shift's calls but left the fundamental band
2 to 5 times less accurate. The support rule is the scalar shift on C (its boundary
lies beyond the grid) and on D's developed window (1.00–1.02 of the calls, band errors
within interpolation noise of the scalar's), and took 0.43–0.52 of the calls on D's
launch, with a better harmonic band and a fundamental band up to 1.3 times worse at one
matched error. §7.4 asks for every band at most the scalar's and below 1e-2: the first
two rules fail clearly, the support rule narrowly, and its one advantage is a single
launch window. `"harmonic"` is removed (§4.6, §7.4).

**Delivered.**

- `solprop/stepper/multistep.py`, `solprop/backends/exponential.py` and the three
  backends: `CarrierFrame` is `"off" | "scalar"`; `HARMONIC_BOUNDARY`, the per-bin
  multiple of `CarrierReference` and the harmonic refusals are gone (the kernel contract
  needs no bump to remove an admitted value). Tests in
  `tests/backends/test_carrier_frame.py` follow; the per-band manufactured source stays
  as the witness that the scalar shift resolves a developed third harmonic without being
  exact for it.
- Manual: [the carrier frame](../../guide/numerics/adams.md#the-carrier-frame) with the
  phase 4 results and the advice that follows from them (use `"scalar"` where the
  populated band is the observable; check a weak-band observable against a tighter run),
  the limitations entry, and [choosing a stepper](../../guide/numerics/choosing-a-stepper.md),
  whose ranking, measured with the default, stands. The changelog, design notes,
  capability table and contract row lose `"harmonic"`.

**Deviations.**

1. *The matched range.* §7.3 compares at worst-plane errors 1e-3 to 1e-6; the short
   windows rarely reach 1e-3 (most stay below 1e-5 even at `rtol = 1e-3`), so the
   comparison was made at every decade from 1e-3 to 1e-9 that both curves span. That is
   stricter, not looser.
2. *The cheap-source case.* The GNLSE route exists only on envelope grids, where the
   policy is inert; case A with Kerr alone stands in for it.
3. *The developed windows' start planes* were written by IP-DOPRI5 at `rtol = 1e-8`, not
   a converged run: every compared run starts from the same plane, so it has to be
   physical, not exact.
4. *Timing.* Nine single-thread processes at once on separate physical cores of one
   socket of a two-socket Intel Xeon Gold 6240 workstation, each pair of policies
   interleaved sample by sample, so that a change of load reaches both alike; not one
   process on an idle machine.
5. *Case C's band above 2 omega_0* has no converged reference (its two references agree
   to 7e-3 there); its gate comparison is indicative only.

**Verification.** The carrier-frame, multistep and exponential-march modules pass after
the removal (228 passed). The qualification's harness, fixtures, raw records and
references are kept with the project's development records; the figures here are its
analysis.

**Open questions** (for the roadmap at Final).

1. A weak-band-aware estimate under the shift: scoring a driven weak band in the
   step-size norm, or a per-bin shift chosen by the source's own rotation rather than a
   frequency boundary, might keep the calls and recover the bands.
2. Device qualification of native's kernels (refused by name until then).
3. Native's per-step overhead on cheap sources (1.2–1.3× on helium Kerr alone).

**Handoff.** All four phases are complete. Final needs the SEP archived, its open
questions on the roadmap and the single pull request opened.

### Phase 4, supplement — the four methods at matched band error (2026-10-25)

§7.3 compares the two policies at matched worst-plane error, where `"scalar"`'s bands are
worse. A user who needs a band tightens the tolerance until that band converges, so the
question that decides whether the policy is worth keeping is the cost at matched error
*in each band*, against every method a user could choose. IP-DOPRI5 and the interaction
formulation were therefore run through the same windows, tolerances and references, on
both backends, one thread, five interleaved samples (the setup of phase 4).

**Method.** For each case, window, backend and error measure (worst-plane field, weak
bins, named band), each method's calls and time at matched error, interpolated log-log on
its work-precision front at every decade from 1e-2 to 1e-7 that all four reach: 105
comparisons a backend. IP-DOPRI5's calls are its physical source evaluations, six a step
and its rejections; the Adams methods' are their logical calls.

**Calls.** `"scalar"` needs the fewest at 64 of the 105 points, the interaction
formulation at 28, `"off"` at 13, IP-DOPRI5 at none.

**Time**, the cost a user pays: the method that was fastest at most of a window's
matched errors, and the median ratio of `"scalar"`'s time to the fastest of the other
three:

| case | fastest, `numba_cpu` | `"scalar"` / best other | fastest, native | `"scalar"` / best other |
|---|---|---|---|---|
| `ctc_reduced`, launch | `"scalar"` | 0.88 | `"scalar"` | 0.88 |
| `ctc_reduced`, developed | `"scalar"` | 0.69 | `"scalar"` | 0.70 |
| `rdw`, launch | `"scalar"` | 0.90 | IP-DOPRI5 | 1.08 |
| `rdw`, developed | `"scalar"` | 0.94 | `"off"` | 1.06 |
| `argon_blueshift`, launch | interaction | 1.31 | interaction | 1.37 |
| `argon_blueshift`, developed | interaction | 1.17 | interaction | 1.20 |
| `h2_raman_6ps`, launch | interaction | 1.30 | interaction | 1.19 |
| `h2_raman_6ps`, developed | interaction | 1.16 | interaction | 1.25 |
| `he_rdw_averaged`, launch | IP-DOPRI5 | 2.06 | IP-DOPRI5 | 2.32 |
| `he_rdw_averaged`, developed | IP-DOPRI5 | 2.28 | IP-DOPRI5 | 2.60 |
| case A, launch | interaction | 1.30 | interaction | 1.39 |
| case A, Kerr only, launch | interaction | 1.97 | interaction | 2.28 |

(Case A's developed windows reach only one common decade: 1.01–1.66 on `numba_cpu`,
1.10–2.48 on native.) Over all points the fastest method is the interaction formulation
at 41 (`numba_cpu`) and 44 (native), `"scalar"` at 39 and 22, IP-DOPRI5 at 20 and 28,
`"off"` at 5 and 11.

**Reading.** The carrier frame is the fastest method on one case of seven: the
two-mode comb-to-continuum capillary with Raman, where it takes 0.69–0.88 of the time of
the next best. On the four-mode helium RDW it is level with the default (0.90–0.94 on
`numba_cpu`, 1.06–1.08 on native). Everywhere else a method that already exists is
faster: the interaction formulation in dense argon and in the 5 bar hydrogen soliton
(by 16–37 %), and IP-DOPRI5 or the interaction formulation on the single-mode helium
cases, where a source evaluation is cheap (by a factor of 1.3–2.6). Where the shift wins
calls it often loses time, because its per-step cost (§7.2's ledger entry) and the
Adams arithmetic outweigh the calls a cheap source saves.

**What this means for the policy** (a reading for the maintainer, not a decision): the
default stays `"off"` as decided above. Keeping `"scalar"` buys 12–30 % on expensive,
dense, multimode Raman problems at the cost of carrier-specific arithmetic in three
backends, a step cap, and a tolerance that means something different in weak bands. The
alternative is to withdraw it, with the manual advising the interaction formulation for
dense gases on field grids and `RungeKutta()` where the source is cheap, which this
comparison supports in five of the seven cases.

### Phase 4, supplement — the comb-to-continuum case on realistic windows (2026-10-25)

The suite's `ctc_reduced` case uses a 1 ps window, which crowds its 400 fs pulse; a
production run of this physics takes 2 ps or more. The same case (16 µm, 40 bar H₂, two
modes, Raman at 294 K, plasma, 250 nm–2 µm, 515 nm, 400 fs, 10 µJ) was therefore run
with 2 ps and 4 ps windows (4097 and 8193 bins against 2049), on the same launch and
developed windows, with fresh start planes and references, on native CPU, one thread,
five interleaved samples, IP-DOPRI5, `"off"` and `"scalar"`.

**The 2 ps window is enough:** every method takes exactly the same calls at every
tolerance on the 2 ps and 4 ps grids; only the cost of a call scales.

**Time at matched error**, as a ratio to `"scalar"` (`"off"` / `"scalar"`, IP-DOPRI5 /
`"scalar"`), at the decades all three reach:

| window | worst-plane field | weak bins |
|---|---|---|
| 2 ps, launch | 2.28, 2.08 | 1.8–2.4, 1.9–2.4 |
| 2 ps, developed | 2.5–2.9, 1.6–2.0 | 1.9–2.1, 1.6–2.1 |
| 4 ps, launch | 2.26, 2.07 | 1.8–2.3, 1.9–2.4 |
| 4 ps, developed | 2.8–2.9, 1.8–1.9 | 2.0–2.2, 1.8–2.0 |

In calls the ratios are about the same. So on this physics with a realistic window
the carrier frame takes about half the time of both the default and IP-DOPRI5 at matched
error, in the populated bands and in the weak bins alike: a much larger margin than on
the 1 ps window (0.69–0.88 of the next best), and without the weak-bin loss measured
there.

**The blue band (250–400 nm)** cannot be judged on these windows: its two references
(IP-DOPRI5 at `1e-11`, the interaction formulation at `1e-10`) disagree by 2e-6 to 8e-4
there, and every method, IP-DOPRI5 at `rtol = 1e-8` included, leaves it with errors of
1e-2 and more. So early in the propagation that light is weak enough to be seeded by
the noise floor and is highly sensitive; it does not discriminate between methods, and
it qualifies the large blue-band losses recorded above for the 1 ps window.

The Adams interaction formulation (`Multistep(formulation="interaction")`, second on
the 1 ps window and fastest on the argon and hydrogen-soliton cases) was then run on the
2 ps windows in the same setup. It is the second-fastest method there, and the carrier
frame stays the fastest. Time at matched error as a ratio to `"scalar"`:

| 2 ps window | measure | `"off"` | IP-DOPRI5 | interaction |
|---|---|---|---|---|
| launch | worst-plane field (1e-7) | 2.28 | 2.08 | 1.63 |
| launch | weak bins (1e-5 to 1e-7) | 1.81–2.44 | 1.95–2.40 | 1.37–1.73 |
| developed | worst-plane field (1e-5 to 1e-7) | 2.51–2.92 | 1.64–2.02 | 1.37–1.42 |
| developed | weak bins (1e-3 to 1e-6) | 1.93–2.11 | 1.64–2.06 | 1.06–1.28 |

In calls the interaction formulation takes 1.21–1.69 times the carrier frame's. So on
this case with a realistic window the carrier frame takes 27–42 % less time than the
best method that exists without it, narrowing to 6–22 % in the developed window's weak
bins. The suite's `ctc_reduced` case moves to the 2 ps window (revision 2).

## Part III — History

- 2026-10-25: Drafted, with a `numpy_ref` prototype's measurements.

- 2026-10-25: GPT-6 completed the AI review of Part I at `2a4a7f9`, recorded
  below. Review status is `In review`; proposal status remains `Draft`. No human
  review or maintainer decision is implied.

- 2026-10-25: Part I revised in response to the AI review (response recorded below):
  an opt-in `carrier_frame` policy with a step cap, field grids only, stability and
  aliasing conditions derived, the reference rule and contract specified, the harmonic and
  lossy shifts deferred, qualification criteria fixed before measurement.

- 2026-10-25: Revised after maintainer decisions. Folded in a loose-tolerance failure of
  the present default (argon, `rtol = 1e-3`, error 0.97) as evidence and a phase 4 case;
  brought the harmonic shift back as a gated item (§4.6, §7.4); admitted envelope grids with
  the third harmonic, whose frame now measures its delay from zero frequency so that their
  operator is the field grid's; added an interim manual note to phase 1.

- 2026-10-25: Maintainer decisions recorded: responses to R1–R10 approved; default
  `"off"` until phase 4; phase 4 thresholds accepted; pre-SEP snapshots refused; fixed cap.

- 2026-10-25: Reviewed by John Travers (human) and accepted. Implementation starts with
  phase 1 (`numpy_ref`, opt-in) on the same branch.

- 2026-10-25: Phase 1 (`numpy_ref`, opt-in) implemented on
  `feature/multistep-carrier-frame`; status `In progress`; report in Part II.

- 2026-10-25: Phases 2 (`numba_cpu`) and 3 (native CPU) implemented on the same
  branch; reports in Part II.

- 2026-10-25: Phase 4: the default stays `"off"` (the weak-band budget and native's
  time fail §7.3's criterion); `"harmonic"` fails §7.4's gate and is removed. Report
  in Part II.

- 2026-10-25: Phase 4 supplement: IP-DOPRI5 and the interaction formulation through the
  same windows; the four methods compared at matched band error (Part II).

- 2026-10-25: Phase 4 supplement: the comb-to-continuum case on 2 ps and 4 ps windows
  (Part II); the suite's `ctc_reduced` moves to a 2 ps window.

- 2026-10-25: The Adams interaction formulation added to the 2 ps comparison (Part II).

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

### Response to review — 2026-10-25 — Opus 5.5

Every point was checked against the code and, where it made a numerical claim, reproduced
before Part I was revised. The proposal is now an opt-in policy with a step cap; the
default decision moves to phase 4 against criteria fixed in advance.

*Maintainer decisions (John Travers, 2026-10-25):* the responses to R1–R10 below are
approved as recorded (R6 with the further decision noted under it). In addition: the
default stays `"off"` until the phase 4 gate, with the interim manual note in phase 1; the
phase 4 thresholds of §7.3 and §7.4 are accepted as written; march snapshots taken before
this SEP or under another policy are refused and restart their history (no compatibility
path for `"off"`); and the step cap is a fixed constant of the policy, its value re-checked
in phase 1 (Q6), not a user setting.

1. **R1 — accepted; the cap added.** The counterexamples reproduce, and the failure has an
   exact mirror in the present formulation: a carrier-following source at
   $hD = 2\pi\mathrm{i}$ gives endpoint error 1 with defects below $5\times10^{-16}$ at
   orders 1, 3 and 5. Aliasing is a property of any history-only estimate; the shift
   changes which components are smooth. §4.2 now states which parts of the source follow
   the carrier (the phase-equivariant ones) and which are given a residual rotation; §4.3
   states the resolution assumption for the rotated source, tabulates the reviewer's mixed
   witness (defect above error up to $h|\kappa| = 2$, blind at $\pi$), and adds a cap
   $h|\kappa| \le 1$ that keeps the named components' residual phase per step at or below
   2 rad. Measured: the cap costs nothing on cases A, B and D (identical step sequences)
   and on C trades 477 for 582 evaluations while halving the error. The global claims of
   §1 and §9 are withdrawn; §7.1 adds the requested near-resonant, variable-step,
   startup and replay witnesses, and the default now waits for §7.3.

2. **R2 — accepted.** The stated reason was wrong; the spectral radii 0.888 and 1.025
   reproduce with production $\varphi$-functions and Lagrange coefficients. §4.4 now
   derives the residual-phase dependence and tabulates, per order, the residual phase up
   to which `PEC_REAL_INTERVAL` remains valid (about 3 rad at orders 4–5 without loss,
   more with loss), notes that the present formulation needs the same condition with
   $x' = hD$ and enforces neither, and keeps the order weight as a heuristic to be
   qualified on a resolved collar case and a coupled case (§7.1, Q7) rather than asserted.

3. **R3 — accepted.** The sign is fixed: $\sigma = -\mathrm{i}\kappa h$ and the witness is
   $F = e^{-\mathrm{i}\kappa z}g$ (§4.2, §7.1). The field/envelope relation is derived
   separately for the envelope grid without ($D_\text{field} = D_\text{env} - \mathrm{i}\kappa$)
   and with the third harmonic ($+\mathrm{i}\beta_1\omega_0$, shift $-\mathrm{i}\beta_0h$);
   the latter route is no longer admitted (§2) because its correctness is in question.
   The equivariance assumption is stated (§4.2). The field/envelope step-count comparison
   is removed as a criterion; the Kerr-only comparison is not repeated in Part I.

4. **R4 — accepted.** §6.1 gives an executable rule: the trial's own exponent, batch
   member by member, column 0 (the frame's mode; a replicated coupled carrier is a
   multiple of the identity), polarisation 0, the active bin nearest the frame's reference
   frequency with ties to the lower index; an inactive bin falls back to $\sigma = 0$ with
   the reason recorded; free space is refused until qualified; a custom frame only changes
   $\sigma$, which §4.1 makes exact for any value. The envelope grid without the third
   harmonic is inert by rule rather than by assumption. Q3–Q5 are resolved or assigned to
   phases 1 and 4 (§12), and every unqualified route keeps an explicit refusal.

5. **R5 — accepted.** §4.5 writes the predictor, corrector, combined defect and dense
   formula with the transported compensation, states that its values change through the
   new endpoint, derives the varying carrier's residual phase
   ($\tfrac12\kappa'h^2\theta_j(1-\theta_j)$ for a linear $\kappa$), and separates the
   coupled flow from its moments: since $\sigma I$ commutes with the block,
   $e^{hB} = e^{\sigma}e^{hB-\sigma I}$ in both the Schur and eigenbasis paths, and the
   present `coupled_moments` object must not be shifted and reused as the flow. §7.1 lists
   the requested witnesses, including source/compensation cancellation.

6. **R6 — accepted.** The harmonic shift is deferred out of this SEP with the reasons
   given (§10, Q1), and a real part in the shift is not proposed (§10, Q2), for the range
   reason the review gives. The weak-band criterion is now an absolute budget against an
   independent reference, fixed before measurement (§7.3).
   *Maintainer decision (John Travers, 2026-10-25):* bring the harmonic shift back into
   this SEP as a gated item. Done in §4.6 and §7.4, with R6's criteria as the gate:
   per-band references and budgets fixed in advance, boundary crossing and fixed-per-run
   $m(\omega)$ across rejections and resume, and removal if it fails.

7. **R7 — accepted.** §5 and §6.5 make `carrier_frame` a field of `Multistep`, carried in
   `KernelProblem`, with a `SCHEMA_VERSION` bump in phase 1, refusals by name on every
   backend and route that does not deliver it, and the default `"off"` until phase 4, so
   each phase is independently usable (§11). The output layout needs no new field: the
   method representation already records the policy.

8. **R8 — accepted.** The claim that dense observation is shared on every backend was
   wrong for native: §6.3 now names `GeneralWorkspace::adams_exponential_observe` and the
   need to hold the accepted step's $\sigma$ across later trials and rejections, and the
   coupled interpolant. The shift is formed on the host from one element of the operator's
   representation, so no device read-back is needed (§6.1). Cost: measured `phi_functions`
   on the shifted and unshifted exponents of cases A and C, equal within 5 % (no element
   crossed $|x| = 3$); §6.6 keeps the timing as a per-phase measurement and drops the
   zero-cost claim.

9. **R9 — accepted.** §6.5 and §9: the method representation changes, so snapshots from
   before this SEP or under another policy are refused and the history restarts; there is
   no conversion. §7.1 adds saved and in-memory continuation tests under the same policy,
   and a parity rounding budget that includes the rotation, the phase, the subtraction and
   the interpolation, derived in phase 1. "Within the tolerance" is replaced by accuracy
   envelopes against converged references (§9).

10. **R10 — accepted.** §7.2 now gives the complete fixtures (radius, gas, pressure,
    length, wavelength, duration, energy, modes, wavelength limits, window, responses,
    tolerances), the prototype's base commit and the machine, labels the figures as
    preliminary full-length motivation, adds a matched-error comparison on C (582 against
    2669 near $6\times10^{-6}$; 649 against about 3300 near $1.1\times10^{-6}$), and shows
    C's reference agreeing with a second method to $7.9\times10^{-9}$ while saying the
    others' is not yet shown. §7.3 defines short launch and developed-spectrum windows on
    the suite's cases, reference convergence checks, worst-plane and named weak-band
    budgets, the matched-work range and timing margins, all before measurement, and
    includes cheap-source and currently favoured cases.
