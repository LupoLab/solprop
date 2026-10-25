# The Adams integrator

`Multistep()`, the default integrator, reuses recent nonlinear source evaluations
to advance the field. Away from startup and rejected trials, it usually needs one
new source evaluation per accepted step. This can save time when a response is
expensive, while requiring more state history in memory.

The default exponential formulation integrates an interpolating source polynomial
against the represented linear flow. The alternative interaction formulation
transports samples into a moving phase frame before interpolation. Both still
require error control and convergence checks; a cheaper source or a memory limit
can favour Runge–Kutta. [Choosing a stepper](choosing-a-stepper.md) gives the
comparison, and [accuracy](accuracy.md) explains what to refine.

This chapter gives configuration, equations, interpolation and restart behaviour,
and limitations. Its design record is
[SEP 16](../../dev/seps/archive/sep-0016-unified-multistep.md).

## Using it

Every run steps with `Multistep()` unless told otherwise:

```{code-block} python
from solprop.driver import prop_capillary
from solprop.gasflow import steady_fill
from solprop.stepper import Multistep, RungeKutta

fill = steady_fill("He", [0.0, 2.1, 4.2], [0.0, 0.2, 0.0], radius=256e-6)
run = prop_capillary(
    256e-6,
    4.2,
    "He",
    fill,
    lambda0=800e-9,
    fwhm=6e-15,
    energy=1e-3,
    trange=60e-15,
    wavelength_limits=(90e-9, 1.6e-6),
    rtol=1e-3,
    n_save=50,
    absorber_scheme="split_step",
)
```

`integrator=Multistep(order=3)` caps the order, `integrator=Multistep(formulation="interaction")`
takes the interaction picture and `integrator=RungeKutta()` the Runge–Kutta march, on
every entry point and on `SimulationSettings`. Both formulations run on `numpy_ref`,
`numba_cpu` and `native`, CPU and device, on the GNLSE, fibre and capillary mode sets,
the general modal route and the radial, Cartesian, line and planar free-space routes,
with constant and varying operators and every noise model the route admits. Beside a
coupled flow (a coherent capillary wall, a split block, a moving guide) only the
exponential formulation runs, on its constant-Schur and frozen-residual sections; the
interaction formulation and CF4 sections are refused by name. Another backend refuses
`Multistep()` by name and runs `RungeKutta()`. `Multistep(carrier_frame="scalar")`
removes the carrier's rotation from the exponential history on field grids
([the carrier frame](#the-carrier-frame)); it is opt-in and runs on `numpy_ref` only
for now.

`Multistep()` is SEP 13's selected method, in SEP 16's exponential formulation:

- one source call per step;
- an Adams–Bashforth predictor with an Adams–Moulton corrector one order higher;
- the order chosen between one and five as the run goes;
- no cap at the operator table's interval edges.

The per-step collar in the example suits this capillary, whose pulse never reaches the
collar; keep the default in-stage collar otherwise (below). The sections below explain
each choice ([ADR-037](../../dev/decisions.md)).

## The method

The recurrence is set out first in the interaction picture, where SEP 13 developed it;
[the exponential formulation](#the-exponential-formulation) keeps it and changes only
what the polynomial interpolates.

### Transported derivatives

Write the propagation equation as $a' = L(z)a + N(a, z)$, with the exact linear flow
$U(z_2, z_1) = e^{\Phi(z_2) - \Phi(z_1)}$ of the cumulative operator $\Phi$. Anchored
at the start of a step, $z_n$, the interaction variable $v(z) = U(z_n, z)\,a(z)$ obeys

$$v' = g(z) = U(z_n, z)\,N\bigl(U(z, z_n)\,v, z\bigr).$$

An Adams method integrates a polynomial through earlier values of $g$. The method
stores each source evaluation with its position, carried into the current anchor's
frame. Accepting a step re-anchors every stored derivative by that step's own
propagator, so each keeps the density, field and response it was evaluated with.
No operator table is queried backwards.

### Variable-step weights

With the history positions normalised by the step, $\theta_j = (z_j - z_n)/h$, the
increment is $h\sum_j w_j g_j$, where $w_j$ integrates the $j$-th Lagrange basis
polynomial on the **actual** nodes over $[0, 1]$:
{py:func}`solprop.stepper.multistep.integrated_weights`. On equal steps these are the
classical Adams–Bashforth and Adams–Moulton coefficients; after a step-size change
they are recomputed, never reused.

### Predict, evaluate, correct

1. **Predict.** An Adams–Bashforth formula of order $q$ extrapolates the stored
   derivatives over the step: $v_P = a_n + h\sum_{j<q} w^{B}_j g_{n-j}$.
2. **Evaluate.** The source is evaluated once, at the predicted field
   $U(z_{n+1}, z_n)v_P$. This is the step's only call.
3. **Correct.** An Adams–Moulton formula through the new derivative and $q$ stored
   ones gives $v_C$, one order higher than the predictor. The step accepts
   $a_{n+1} = U(z_{n+1}, z_n)v_C$.
4. **Control.** The difference $v_C - v_P$ is the predictor's local defect. Carried
   to the end of the step, it is scaled by the accepted state the step started from
   and fed to an elementary controller with exponent $1/(q + 1)$.

The derivative stored for the next step is the one evaluated at the *predicted*
field. That is the meaning of PEC: predict, evaluate, correct, with no second
evaluation.

On equal steps this scheme has an exact identity. The predicted states advance by
Adams–Bashforth of order $q + 1$, and the corrector adds an Adams–Moulton output
correction. It therefore has Adams–Bashforth $(q+1)$'s stability region, and the
accuracy of the corrected value.

### Order

`order` is the highest predictor order, five by default. `variable_order=True`, the
default, chooses the next step's order among the current one and its neighbours from
their defect estimates. All of them come from the same history and source, so the
choice costs no call. It settled at the best fixed order on every case measured: orders
two and three on the canonical capillary, and up to five on the RDW capillary and on a
high-order soliton. `Multistep(order=3, variable_order=False)` fixes the order.

In the exponential formulation the choice also weighs the in-stage collar (SEP 16
phase 6). Its linear flow is exact, so the collar's damping is the one stiff term the
explicit history integrates, and each order is stable only up to a step its
negative-real interval sets. Each order is weighed by the step its defect allows, capped
at that stable step; the step itself is still the chosen order's defect's. Without it,
near the limit the higher orders' defects carried the collar's growing mode and the
choice was decided by rounding: on a 2 cm coherent-wall capillary, runs a rounding
apart took 930 or 1313 calls, their fields 2.7e-5 apart. With it every run took 688,
and the full canonical and RDW runs 8 and 4 per cent fewer calls. The interaction formulation's choice is
unchanged: its history is carried through the propagators, whose stiffness the collar
alone does not describe, and the cap cost it calls.

### What was compared

SEP 13 also ran these variants. They remain executable, as
`solprop.stepper.multistep.ComparisonMultistep`, for the benchmarks only:

- **PECE**, which spends a second call at the corrected field;
- **plain Adams–Bashforth**, with an a-posteriori recheck;
- **matched predictor–corrector pairs**, controlled by Milne's estimate.

None of them used fewer calls at the same accuracy on any case:

- PECE does not repay its second call.
- The matched pairs underestimate their error at loose tolerances, and their small
  stability regions meet the temporal collar first.
- Plain Adams–Bashforth takes about as many steps as PEC, but is less accurate.

## The exponential formulation

`Multistep(formulation="exponential")` keeps the predict–evaluate–correct recurrence,
the variable order and the controller above, and changes what is interpolated. With a
constant operator $D$ the equation is $u' = Du + F(z, u)$, and over a step $h$ from
$z_n$

$$u_{n+1} = e^{hD}u_n + h\sum_j w_j(hD)\,F_j,\qquad
w_j(x) = \int_0^1 e^{(1-\theta)x}\,\ell_j(\theta)\,d\theta,$$

where the $F_j$ are the **untransported** source samples at their own positions and
$\ell_j$ the Lagrange basis on the normalised nodes $\theta_j = (z_j - z_n)/h$. Writing
$\ell_j(\theta) = \sum_k c_{jk}\theta^k$,

$$w_j(x) = \sum_k c_{jk}\,k!\,\varphi_{k+1}(x),\qquad
\varphi_r(x) = \sum_{m\ge0}\frac{x^m}{(m+r)!},$$

so the linear flow is exact and only the source's polynomial is approximated
(Cox and Matthews 2002, [doi:10.1006/jcph.2002.6995](https://doi.org/10.1006/jcph.2002.6995);
Hochbruck and Ostermann 2011, [doi:10.1007/s10543-011-0332-6](https://doi.org/10.1007/s10543-011-0332-6)).
At $x = 0$ the weights are the ordinary Adams ones. The predictor uses the $q$ newest
samples, the one call is made at the predicted endpoint, and the corrector uses that
sample with the same $q$; the sample stored is the predicted one, as before.

**Which formulation suits which source.** In the interaction picture each sample is
carried to the step's anchor by the linear flow, $e^{-\Delta\Phi}F$; in the
exponential formulation it is not. Whichever of the two sequences is smooth across the
history is the one a polynomial follows:

- *A driven bin*, whose source is set by other frequencies — the four-wave mixing,
  Raman and plasma sources that build a dispersive wave or a continuum — is smooth in
  the laboratory frame; transported, it rotates by $|\operatorname{Im}D|h$ per step.
  The exponential formulation integrates it at any linear phase; the interaction
  picture's history limits its steps to a linear phase of about one in the populated
  bins.
- *A source that rotates with its own bin's free phase* — a term linear in the field of
  that bin, such as the part of the `modified` noise's source linear in the draw in a
  weak bin — is smooth only once transported, and favours the interaction picture.
- *A source that follows the carrier* on a field grid — the fundamental band of the
  Kerr, Raman and envelope-driven responses — rotates at the carrier's rate $\kappa$
  in the laboratory frame, thousands of radians per metre in a dense gas. Neither
  formulation makes it smooth; [the carrier frame](#the-carrier-frame) does.

On the scalar witnesses of `tests/stepper/test_exponential_scalar_march.py` the first
case gives the exponential formulation errors of $10^{-11}$ where the interaction
picture's are of order one, at $|\operatorname{Im}D|h$ up to ten; the second gives the
opposite. Real problems mix both, which is why the choice is measured per regime
rather than assumed. On a second-order soliton (`tests/backends/test_exponential_march.py`)
the populated bins converge at order $q+1$ in both formulations, the exponential one
30 to 200 times more accurately; but the weak wings, more than $10^8$ below the peak
power, hold free radiation under the soliton's cross-phase modulation, a source linear
in each bin's own field, and there the exponential formulation converges only at about
order 0.6 once $|\operatorname{Im}D|h$ exceeds one, where the interaction picture
converges at order three to four. The step-size norm does not see bins that weak; a
weak-band observable is scored separately for that reason.

**The moments.** $\varphi_0\ldots\varphi_{q+2}$ are formed once per trial for every
element, by a scaled Taylor series and doubling for $|x| \le 3$ and an upward
recurrence beyond, and every formula of the trial — predictor, corrector, the
neighbouring orders' estimates, an interpolated plane — is applied in the monomial
basis of its interpolant, so no per-node weight array is formed. Checked against
augmented matrix exponentials and quadrature, each moment is within a few hundred ulp
of its scale, so the coefficients add nothing measurable to a step's error and no term
for them enters the error norm.

**What it measured** (short windows, one thread, at common worst-plane field and
weak-band errors from 1e-3 to 1e-5). On the six-mode hydrogen comb-to-continuum launch, a
Raman-active source that is costly to evaluate, it took 2.0 to 3.3 times less time
than IP-DOPRI5 and 1.5 to 2.5 times less than the interaction formulation on
`numba_cpu`, and won on native CPU too. On developed GNLSE supercontinuum windows it
took two to five times fewer calls than both and, from 3e-4 to 1e-5, 3 to 36 % less
time than IP-DOPRI5 and 8 to 31 % less than the interaction formulation on
`numba_cpu`; at 1e-3 the interaction formulation was as fast.
Where the source is cheap and the state small — a radial beam with Kerr alone — its
per-step arithmetic outweighs the calls it saves and the interaction formulation is
faster. [Choosing a stepper](choosing-a-stepper.md#measured) ranks the methods on full
runs, on both compiled backends and on devices. Beside the
`modified` noise, whose source is partly linear in each weak bin's own field (above),
prefer the interaction formulation until the exponential one is measured on a
developed noisy window.

**Interpolated planes** integrate the accepted corrector's polynomial to the fraction
$\xi$ of the step: $e^{\xi hD}u_n + h\sum_{j,k}c_{jk}k!\,\xi^{k+1}\varphi_{k+1}(\xi hD)F_j$.
Every factor is a forward flow, so a strongly damped bin needs no special treatment.

**Varying operators.** Along a gas gradient or a taper, $D$ depends on $z$. Each trial
then takes the chord over its own step as a constant reference,

$$B_n = \frac{\Phi(z_n+h)-\Phi(z_n)}{h},\qquad
u' = B_nu + \bigl[F + (D(z)-B_n)\,u\bigr],$$

an exact rearrangement whose homogeneous flow $e^{hB_n}$ is exactly the step's. The
source samples are integrated against it as above, and the linear compensation
$C_j = (D(z_j)-B_n)u_j$ of each sample — its state is kept beside its source — is
integrated in the reference flow's frame, where it is smooth for a freely propagating
bin:

$$u_{n+1} = e^{hB_n}u_n + h\sum_j w_j(hB_n)\,F_j
 + h\sum_j a_j\,e^{(1-\theta_j)hB_n}\,C_j ,$$

with the ordinary Adams weights $a_j$. Every factor is a forward flow over at least the
step. The rate $D(z_j)$ is the derivative of the representation that defines the flow
— an expansion's scalar integral tables or a z table, differentiated exactly
(`solprop.translate.linop.phi_rate`) — so that the residual integrates to zero against
the chord exactly, and the table's own error stays a translation error shared by every
stepper. When the reference changes, after a step-size change or a rejection, the
compensation is formed again from the same samples; no source is re-evaluated. A
problem whose source is structurally absent takes the chord alone, which is then exact.
A hard kink restarts the history, as for the interaction formulation. Interpolated
planes add the same compensation integrated to the fraction $\xi$; a bin that the step
damps by more than an e-fold takes the anchor's flow.

The compensation is an explicit linear term: it inherits the method's stability
limits, and where a fill steepens towards its exit, its higher derivatives grow and a
fixed step loses order there (the step-size estimate, which includes the compensation,
sees it in an adaptive run). On a short, steep gradient the in-stage collar's stability
limit (below) can set the step instead: with it, a 2 cm gradient held the Adams step
near 30 µm, where the per-step collar needed 18 times fewer attempts. The per-step
collar is the remedy only when the collar holds no field. Treating the compensation in the untransported frame
instead is kept only as a test witness: where the compensation dominates, its error is
under-reported to the controller ([SEP 16](../../dev/seps/archive/sep-0016-unified-multistep.md)
§4.2, Q8).

**Where it runs.** On `numpy_ref`, `numba_cpu` and native CPU, for the constant
operators — the GNLSE, fibres and capillaries with constant profiles, constant general
modal guides and the radial, Cartesian, line and planar free-space routes — and for
varying diagonal operators: gas gradients, steady fills with ports, tapers. The compiled
backends form each element's moments once per trial in their own pass and store no
coefficient volume.

**Coupled sections.** Beside a coupled flow the cumulative diagonal is
a multiple of the identity in the channel axis, so a step's reference is
$B = (\Delta\Phi/h)\,I + G_s$, with $G_s$ the frozen block of the section the step is
anchored in, and the two commute. On the bins that carry a block the step's flow and
moments are matrix functions of $hB$ in the channel axis
(`backends.multistep.coupled_moments`, from the exponential of an augmented matrix
carried back by the block's Schur vectors), applied to each sample's channel vector; off
them they are the diagonal ones. A frozen-residual section's residual $R_s(z)u$ is
linear in the state and joins the transported compensation above: each sample keeps its
flow weights, and a trial forms $(D(z_j)-\Delta\Phi/h)\,u_j + R_s(z_j)\,u_j$ against its
own section's block -- rebuilding the history's residual when a step enters a new
section -- and carries it by the reference's forward flow, never an inverse one. The
residual acts on the in-band bins, as the Runge-Kutta stage applies it. The lab-frame
sample is the material source alone. The interaction formulation stays refused, since it
would carry its history through inverse actions whose conditioning grows with loss; a
CF4 section, which exponentiates the generator inside each stage, is excluded. On the
scalar wall and the taper (0.2 mm windows) it takes
two to three times fewer calls than IP-DOPRI5 at matched error. On `numba_cpu` and
the native backend, the bins on the flow actions' eigenbasis route take their matrix functions as
`W f(x + h λ) Y` from that basis; `numba_cpu` forms them, applies them and carries the
compensation in compiled passes.

### The carrier frame

On a field grid the operator keeps the propagation constant at the reference frequency,
so it differs from the envelope grid's (same frame, no third harmonic) by a constant
phase rate at every frequency,

$$D_\text{field} = D_\text{env} - \mathrm{i}\kappa,\qquad
\kappa = \beta(\omega_0) - \beta_1\omega_0 ,$$

and an envelope grid with the third harmonic has the field grid's operator. The part of
the source that follows the carrier then rotates at $\kappa$ along $z$: 12 rad/m in a
low-pressure helium capillary, $-5.5\times10^{3}$ rad/m in 40 bar of hydrogen. At order
five the controller keeps $h|\kappa|$ small enough for a polynomial to follow that
rotation, which costs steps: on a dense molecular gas the exponential formulation took
several times more source evaluations than IP-DOPRI5.

`Multistep(carrier_frame="scalar")` removes the rotation inside each step. For any
$\sigma$, writing $F(z_n+s) = e^{-\sigma(1-s/h)}G(s)$ gives exactly

$$u_{n+1} = e^{hD}u_n + h\sum_k k!\,\varphi_{k+1}(hD-\sigma)\sum_j c_{jk}\,G_j,
\qquad G_j = e^{\sigma(1-\theta_j)}F_j ,$$

so the step interpolates the rotated samples $G_j$ and integrates them against the
moments at $hD - \sigma$. The shift is the step's own carrier phase,
$\sigma = \mathrm{i}\operatorname{Im}(hD)$ at the active bin nearest the reference
frequency, on the frame's mode and first polarisation, one per batch member. The
state, the stored samples and the linear flow $e^{hD}$ are unchanged; only the function
the polynomial follows is. Rotating the samples without shifting the moments would not
be equivalent: it integrates $G$ against the wrong flow, an error of order
$|\sigma|$ per step. An interpolated plane at the fraction $\xi$ is

$$u(z_n+\xi h) = e^{\xi hD}u_n + e^{-\sigma(1-\xi)}\,h\sum_k k!\,\xi^{k+1}
\varphi_{k+1}\bigl(\xi(hD-\sigma)\bigr)\sum_j c_{jk}G_j ,$$

with the accepted step's $\sigma$. A varying operator keeps its transported
compensation unchanged (its endpoint is now the shifted predictor's state); beside a
coupled flow the moments are those of the shifted block $hB - \sigma I$ and the flow is
$e^{\sigma}$ times its $\varphi_0$, the true $e^{hB}$, since $\sigma I$ commutes with
$B$.

**What the shift does not remove.** Three parts of a source do not follow the carrier:
the third-harmonic part of the real-field cubic, which rotates at $3\kappa$ and keeps a
residual $2\kappa$; parts near zero frequency and driven bins not tied to the carrier
(optical rectification, a Raman or plasma polarisation's slow part), which the shift
gives a residual $\kappa$; and responses to the carrier-envelope phase itself, such as
field-grid ionisation. They are still integrated exactly in their linear part, and the
polynomial resolves them as it resolves any rotating component, while the residual phase
per step stays well below $\pi$. At $\pi$ the samples alias: they look smooth and the
predictor–corrector defect stops seeing the error. The same failure exists without the
shift for a carrier-following source at $h|\kappa| = 2\pi$.

**The cap.** Each step therefore covers at most one radian of carrier phase,
$h|\kappa| \le 1$ (`solprop.stepper.multistep.CARRIER_PHASE_CAP`, part of the policy
rather than a setting): the residual phase of a stationary part is then at most 1 rad
and of the third harmonic 2 rad. On a mixed witness, a carrier with a residual
$2\kappa$ component of a tenth of its amplitude, the defect exceeds the error by 2.8 to
13 at the cap and orders one to six, and first falls below it at $h|\kappa|$ between 2.05
and 2.5, so one cap serves every order. The step proposal is shortened before the trial;
a shortened step is not a landing and does not restart the history. On the
comb-to-continuum case the cap cost 582 rather than 477 evaluations and halved the error;
on the helium cases it never bound.

**Stability.** For a damping term in the source, the in-stage collar's, the shifted
recurrence is the unshifted one at $x' = hD - \sigma$, and its stability depends on the
residual phase $\operatorname{Im}x'$ as well as on the damping. Every damping up to
`PEC_REAL_INTERVAL` stays stable while $|\operatorname{Im}x'|$ is at most

| order | interval | $\operatorname{Re}x' = 0$ | $\operatorname{Re}x' = -0.1$ | $\operatorname{Re}x' = -0.5$ |
|---|---|---|---|---|
| 1 | 1.0 | > 5.9 | > 5.9 | > 5.9 |
| 2 | 0.545 | 5.1 | > 5.9 | > 5.9 |
| 3 | 0.3 | 3.5 | > 5.9 | > 5.9 |
| 4 | 0.163 | 3.2 | 3.4 | > 5.9 |
| 5 | 0.0877 | 3.0 | 3.1 | > 5.9 |

The unshifted formulation needs the same with $x' = hD$, and neither enforces it: the
collar couples bins, so the per-bin bound is a heuristic in both. The shift moves each
damped bin's residual phase by at most the cap's one radian.

**Where it applies.** On field grids and on envelope grids with the third harmonic. On
an envelope grid without it the frame removes $\beta_0$ already, the shift is zero, and
the run is the unshifted arithmetic exactly. If the reference bin is inactive the run
takes $\sigma = 0$ and its log says why. The shift is formed from the grid's reference
frequency; a launch away from it, or a custom frame, leaves part of the rotation, which
changes the cost and never the exactness. Either shift needs the exponential
formulation, and the free-space routes refuse it until it is qualified there.

**What it measured** (preliminary, full-length `numpy_ref` runs at `rtol = 1e-6`,
source evaluations and final-field error against a tighter IP-DOPRI5 reference;
[SEP 35](../../dev/seps/sep-0035-carrier-frame-exponential-adams.md) §7.2):

| case | $\kappa$, rad/m | `"off"` | `"scalar"` | interaction | IP-DOPRI5 |
|---|---|---|---|---|---|
| He capillary, 0.2 bar fill, 4.2 m | −12 | 1128 / 1.4e-6 | 932 / 1.1e-6 | 1644 / 7.5e-7 | 2268 / 4.4e-6 |
| He capillary, four modes, 1.5 m | −48 | 1805 / 1.6e-6 | 1742 / 2.5e-6 | 10475 / 3.1e-7 | 12397 / 2.5e-6 |
| H₂ 40 bar comb-to-continuum, 0.1 m | −5470 | 2669 / 5.8e-6 | 582 / 6.4e-6 | 1401 / 2.3e-6 | 798 / 3.0e-7 |
| H₂ 5 bar self-frequency shift, 1 m | −2420 | 26060 / 1.0e-4 | 18910 / 7.6e-5 | 17032 / 6.6e-5 | 27979 / 3.2e-5 |

On the argon capillary of the limitations below, at `rtol = 1e-3`, it brought the error
from 0.97 to 0.05 in fewer evaluations than IP-DOPRI5. These are motivation, not
qualification; whether it becomes the default is decided on the benchmark suite's short
windows against criteria fixed in advance.

**The harmonic frame.** `carrier_frame="harmonic"` also gives the third-harmonic band,
the bins at or above $2\omega_0$, three times the shift, so that the harmonic source is
smooth too; the moments then take $hD - m(\omega)\sigma$ element by element, with
$m(\omega)$ fixed for the run. A frequency boundary does not identify a source's
phase — self-phase-modulated fundamental light and harmonic generation reach the same
bins — so it is experimental: `numpy_ref` only, until it passes the gate of SEP 35 §7.4,
and removed if it does not.

## Stability, and the temporal collar

A multistep method is only conditionally stable. SEP 13's scalar study gives the
constant-step stability intervals:

| Method | negative real axis | imaginary axis |
|---|---|---|
| IP-DOPRI5 | 3.31 | 1.00 |
| PEC2/3 | 0.55 | 0.72 |
| PEC3/4 | 0.30 | 0.43 |
| PEC4/5 | 0.16 | 0.04 |
| PEC5/6 | 0.087 | 0.077 |
| PECE3/4 | 1.93 | 1.18 |

The temporal absorbing collar ([absorbing boundaries](boundaries.md)) is, by default,
a damping term inside every source evaluation. Its peak rate is a negative-real
eigenvalue: on the canonical capillary $\alpha/2 = 30\ \mathrm{m^{-1}}$. That limits
PEC3/4 to steps of about $0.30/30 = 10$ mm, whatever the tolerance; the full run
confirms the bound, step for step.

`absorber_scheme="split_step"` applies the collar once per accepted step instead,
which is unconditionally stable. Its splitting error is first order and lies wherever
the collar holds field.

- **When the pulse stays clear of the collar,** as on the canonical capillary, the
  field inside the physical window, the peak power and the duration are unchanged. The
  Adams step is then set by accuracy alone.
- **When broadened field reaches the collar, the error reaches the physical window.**
  This was measured on the M2 RDW capillary and on a comb-to-continuum hydrogen
  capillary: a field error of 2e-4 to 7e-4 whatever the tolerance, against 2e-6 with the
  in-stage collar.
- **With the exponential formulation the in-stage collar's limit can set the step.**
  Its steps are longer, and on the full canonical capillary the median step sat at the
  collar's limit; the per-step collar, or a collar of half the strength, took 965 calls
  against 1329. On the full RDW capillary the steps sat near the limit as well, but
  weakening the collar changed nothing (1782, 1790 and 1765 calls at full, half and a
  quarter of its strength): there accuracy sets the step. So where the pulse stays clear
  of the collar, take the per-step collar with the exponential formulation; an implicit
  collar treatment would buy nothing beyond it (SEP 16 Q4).
- **For such runs keep the default in-stage collar.** With it, the interaction
  formulation needed about 2.4 times fewer calls than IP-DOPRI5 on the RDW case at
  Luna's tolerance. [SEP 14](../../dev/seps/archive/sep-0014-stepper-efficiency.md)
  studied a second-order split and did not adopt it: with a Kerr source it is first
  order wherever the collar holds field. An implicit collar is an open question of
  [SEP 15](../../dev/seps/archive/sep-0015-exponential-adams.md).

The higher orders' tiny imaginary-axis intervals are why variable order settles at
orders two and three on the canonical capillary. On the measured runs the Kerr phase a
step accrues stayed below 0.005 rad, far inside every order's imaginary interval: a
real problem meets the negative real axis first, through the in-stage collar, and the
order choice weighs that limit.

## Landing, interpolation and restart

**Steps land on forced positions.** Steps land on save planes, hard kinks (such as a
gas-fill port) and the end of the run.

- Between one and two proposals from such a landing, the remaining distance is split
  into two equal steps rather than a full step and a sliver: variable-step Adams
  tolerates large step ratios poorly.
- A save plane keeps the history.
- A hard kink restarts it, at order one, on its far side.

**Interpolated planes.** `interpolate_planes=True`, the run's setting for either
stepper ([solver](solver.md)), steps across save planes and observes them instead. The accepted step's own derivative polynomial is integrated to the plane in
the interaction picture, and the exact linear flow carries the result there. This
has the step's own order and costs no call. Observed planes never enter the history,
and the end of the run is still landed on.

Dense output also costs memory. On the native backend, enabling it retains `q + 2` source
volumes and four saved fields, where `q` is the maximum predictor order; a varying
operator adds `q + 2` compensation volumes. These buffers are omitted when
`interpolate_planes=False`, the default. At order five the ordinary saving is eleven
complex field volumes. The [memory investigation](../../dev/benchmarks.md#resident-adams-storage)
gives measured peaks and an allocation breakdown for a large Cartesian beam.

For the comparison benchmarks, `ComparisonMultistep(interpolant="hermite")` uses the
generic cubic Hermite interpolant of the step's end states and derivatives instead. It
is a reference, not a production option: its error is one order
below what the step's estimate controls, so it can miss the requested accuracy even in
the strong field. The witness is $u' = x^3$ on $[0, h]$, where the step is exact and
the Hermite midpoint value is 0 against $h^4/64$.

Interpolation is refused with the per-step collar (`absorber_scheme="split_step"`),
whose collared field is not the interpolant's, a coupled flow and the `modified` noise
model (SEP 14 §5). A bin the linear flow attenuates by more than an e-fold over the
step takes the anchor's value carried by the flow, as in the Runge-Kutta march's dense
output ([ADR-039](../../dev/decisions.md)).

The step-size norm is dominated by the strongest spectral components, so a weak band
can be less accurate at an interpolated plane than at a landed one; landing remains
the default ([ADR-036](../../dev/decisions.md)).

What dense observation saves depends on the plane spacing:

- **With 50 planes on the canonical capillary,** the steps are already shorter than the
  spacing, so it saves only a few calls.
- **With 501 planes,** landing caps every step at the spacing. Dense observation then
  takes 272 calls against 506 at `rtol=1e-3`. Its window error, 8.9e-4, is what that
  tolerance buys, the same as a landed run with 50 planes. The native and Hermite
  interpolants agree to every printed digit.

**A resume continues the history.** At every save plane and terminal checkpoint the
output keeps the march's state beside the field: the accepted state exactly as the
backend held it, the history's samples, the order and the proposed step
(`SimulationSettings.resume_history`, on by default while the snapshot is at most
256 MiB). A resume at that position carries on from it, with no startup call and
no climb from order one, so on every backend a resumed run continues the
uninterrupted run's computation, between planes too when it stopped at a terminal
checkpoint (a spent call budget, a failed source), and agrees with it to rounding;
solprop does not promise identical bits ([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)), and a rounding
difference can occasionally change an adaptive step. That is what makes a run
with a minutes-long source cheap to continue ([ADR-046](../../dev/decisions.md)). The
native backend keeps its history resident and copies it to the host only for the
snapshot. Without a snapshot -- `resume_history=False`, a file written before, or one
the resuming backend cannot continue (a frozen-residual coupled flow's history crosses
only between `numpy_ref` and `numba_cpu`) -- the history restarts at order one and the
startup call is charged again, and the resumed run agrees with the uninterrupted one to
its tolerance. A method whose history cannot be kept at all (the comparison methods'
general path) warns when its resume restarts it.

**Repeated rejections.** After three consecutive rejections the history is restarted
from the accepted state, which removes a parasitic mode a smaller step cannot.

## The call budget

`Multistep(max_calls=...)` bounds the charged source calls of a run: startup,
accepted and rejected trials, and restarts. A trial the remaining budget cannot pay
for is not started. The run stops with {py:class}`solprop.exceptions.ResponseBudgetError`,
which carries the last accepted position, its field and the calls spent. A file
output holds every plane reached, and the run resumes from the last one.

## What it costs and what it buys

SEP 13 counted the calls of the interaction formulation on the canonical helium
capillary (4.2 m, steady fill peaking at 0.2 bar, 6 fs and 1 mJ at 800 nm, compressing
to 1.25 fs, 50 save planes), in full-length runs scored at every plane against a
converged reference. They are the method's call economy; run time, the exponential
formulation and the compiled backends are in [choosing a stepper](choosing-a-stepper.md).

| Method | `rtol` | source calls | field error in the window | peak power | UV-band field |
|---|---|---|---|---|---|
| IP-DOPRI5 | 1e-2 | 497 | 1.5e-2 | 1.4e-2 | 0.75 |
| IP-DOPRI5 | 1e-3 | 742 | 1.4e-3 | 1.3e-3 | 0.25 |
| IP-DOPRI5 | 3e-4 | 896 | 3.5e-4 | 5.8e-4 | 5.7e-2 |
| PEC3/4, split collar | 1e-2 | 161 | 7.9e-3 | 1.0e-2 | 0.36 |
| PEC3/4, split collar | 3e-3 | 214 | 2.7e-3 | 3.4e-3 | 0.13 |
| PEC3/4, split collar | 1e-3 | 291 | 9.4e-4 | 7.4e-4 | 4.3e-2 |
| `Multistep()`, split collar | 1e-2 | 140 | 1.3e-2 | 1.9e-2 | 0.32 |
| `Multistep()`, split collar | 3e-3 | 215 | 2.7e-3 | 4.8e-3 | 0.12 |
| `Multistep()`, split collar | 1e-3 | 292 | 1.3e-3 | 1.7e-3 | 5.4e-2 |

At matched accuracy that is 2.5 to 3.5 times fewer evaluations. Before SEP 14 removed
the Runge–Kutta march's interval cap and slow start on these routes, IP-DOPRI5 needed
1127–1477 calls here and the saving was five to seven times. `rtol` means
something different to each method, so compare errors, not tolerances. Loose
tolerances near `1e-1` are unreliable for every Adams variant: an error of ten per
cent allowed per step compounds.

The weak UV band is the most sensitive observable at every tolerance.

At 2 mJ, with stronger ionisation, the same comparison gives a saving of about three
times at loose and medium accuracy, falling to about two at the tightest:

- 443 calls (PEC3/4) against 1001 (IP-DOPRI5) for a window error near 3e-3;
- 594 calls (PEC4/5) against 1176 for 5e-4.

On a cheap source, calls are not run time. On the compiled backends one source
evaluation can cost less than a Runge–Kutta step's own linear work, and the Adams march
takes more steps, so the call saving shrinks in wall time or disappears: a
single-mode capillary, a coupled wall. [Choosing a stepper](choosing-a-stepper.md)
gives the measured ranking by regime, on both compiled backends and on devices.

## Limitations

- `numpy_ref`, `numba_cpu` and `native` on every route, for both formulations, but
  beside a coupled flow, where the exponential formulation alone steps its
  constant-Schur and frozen-residual sections and CF4 sections are refused. On `native`
  the history and the step's arithmetic are resident in the engine, on the CPU and on
  the AMD CDNA 3 and NVIDIA Hopper devices SEP 16 qualified, beside a coupled flow too
  when every bin of its sections is on the flow's eigenbasis route and it has at most
  16 channels (otherwise its matrix passes run on the host).
- **Resume across backends.** A frozen-residual coupled flow's history is kept in a
  native form on `native` and as per-sample weights on the host engines, so its snapshot
  continues only on a backend of the same kind; a resume elsewhere starts the history
  afresh and says why. Every other history crosses between backends.
- **A coupled flow with no source.** Beside a frozen-residual coupled flow (a moving
  guide, a coherent wall) the explicit residual is sampled with the source, where the
  controller scores it. A run with no nonlinear source has no sample, so the step is
  the frozen block's alone and its error is not seen: a linear taper is as accurate
  as its `dz_max` allows, whatever `rtol` (second order in the step; 6e-4 on a
  graded-index taper at ten steps). Set `dz_max` from a refinement, or step a linear
  coupled run with `RungeKutta()`, whose embedded pair integrates the residual. A
  diagonal operator is not affected: its chord is its exact flow.
- **Memory.** The history holds from seven (`numpy_ref`, constant operator) to about
  sixty (`native`, a varying operator) further copies of the state. On a capillary the
  operator tables dominate and that is 8 to 55 % more; on a Cartesian free-space grid
  it is two to three times the Runge–Kutta march's. A run that fits only without the
  history is refused before it allocates, naming `RungeKutta()`
  ([choosing a stepper](choosing-a-stepper.md#large-problems-memory)).
- **Loose tolerances.** An error of five per cent or more allowed per step compounds
  through the history, and the predictor–corrector estimate is unreliable there: an
  adaptive run at `rtol` of 5e-2 or more is warned on.
- **Loose tolerances in dense gases, on field grids.** On field grids, and envelope
  grids with the third harmonic, in a dense gas the default `Multistep()` at a loose
  tolerance can accumulate a large error over many honestly estimated steps: on a 13 µm
  argon capillary at 1.7 bar (0.4 m, 30 fs, 3 µJ, plasma on), `rtol = 1e-3` gave a
  final-field error of 0.97 in 3909 evaluations, against 0.07 for `RungeKutta()` in
  2625. The order stays at two, nothing is rejected and each step's estimate is a few
  parts in $10^4$; a self-compression amplifies the accumulated error of every method.
  At `rtol = 1e-4` all methods agree to about 1 %. There, use `RungeKutta()` or
  `rtol <= 1e-4`, or `carrier_frame="scalar"` once it is delivered on the backend in
  use (0.05 on `numpy_ref`, [the carrier frame](#the-carrier-frame)).
- **The carrier frame** runs on `numpy_ref` only; `numba_cpu`, `native` and the
  free-space routes refuse it by name. A march snapshot taken under another policy, or
  before the policy existed, restarts the history.
- **Weak wings.** The error norm scores the whole active band, so the light generated
  outside the requested band is integrated under control. Bins far below the peak —
  under 1e-10 of its power — can still hold free radiation under the pulse's
  cross-phase modulation, which rotates with each bin's own free phase and which the
  exponential formulation interpolates in the laboratory frame: there its error,
  bounded by that light's own content (1 to 8 % of it on a soliton at `rtol` from
  1e-7 to 1e-9), is what a total-field error measures at tight tolerances, while the
  populated bins follow the tolerance (SEP 16 Q18).
- The `modified` noise model runs on every backend with either formulation. Its draw is
  fixed for the run and added to each sample's field at that sample's own position,
  never to the propagating state, so conditional on the draw the source is a
  deterministic function of position, which the history interpolates
  ([SEP 16](../../dev/seps/archive/sep-0016-unified-multistep.md) §4.4), on every backend.
  Interpolated planes are refused beside it for every stepper.
- **A prescribed schedule** is replayed with either formulation: its steps are taken
  as given and accepted, the predictor–corrector estimate is still computed and
  recorded every step, the order is chosen by the same rule as an adaptive run's, and
  a hard kink restarts the history. A replay meets the run's accuracy target rather
  than reproducing it bit for bit: replaying an adaptive run's steps reproduces its
  positions, but the orders are chosen afresh and rejected trials are not replayed, so
  where that run rejected a step the replay may choose a different order. Lockstep batches follow (each member keeps
  its own history). The in-stage collar is a damping term integrated explicitly, so a
  replayed step must keep $h\alpha/2$ inside the method's real stability interval —
  0.0877 for PEC5/6 against DOPRI5's 3.3 (the table above, extended to all orders in
  `solprop.stepper.multistep.PEC_REAL_INTERVAL`); the translation refuses a
  schedule whose longest step does not. A few fixed steps measure a one-step method, not
  this one: the history starts at order one.
- The ranking was measured with the built-in Kerr and plasma responses. An expensive
  response must be requalified with its own sources; SEP 12 does this for the
  time-dependent Schrödinger equation.
