# The algorithms

What the solver computes, algorithm by algorithm: what each one is, where it lives, why it
was chosen, the order and accuracy theory gives it, what it costs and where it bites. This
page is a companion to the design notes, not a replacement. [stepper.md](stepper.md) argues
the solver's design and records the experiments behind it, [decisions.md](decisions.md) is
the decision log, [kernel-contract.md](kernel-contract.md) is what crosses into a backend,
[backends.md](backends.md) is how a backend executes it, and [accuracy.md](accuracy.md) is
what has been *measured*.

**What this page states is theory.** An order here is what the method promises; where
solprop has measured it the entry links `accuracy.md` rather than repeating a number that
lives in a test. Each algorithm comes in the same parts — *what it is*, *where*, *why this
one*, *order and accuracy*, *cost*, *traps*. Costs are in transforms, GEMMs and passes over
the state, never in milliseconds; those are in [benchmarks.md](benchmarks.md). Where
solprop deliberately departs from Luna's choice the entry says so in a sentence; the
user-facing list is [Luna and solprop](../guide/workflows/luna.md). Equations are set as MyST maths here, where the other dev pages use fenced Unicode.

## 1. The equation and the interaction picture

**What it is.** solprop propagates a spectral amplitude $\hat{A}(\omega, z)$ obeying

$$\partial_z \hat{A} = D(\omega, z)\,\hat{A} + N[\hat{A}](\omega, z),$$

the $z$-propagated unidirectional pulse propagation equation of Kolesik and Moloney (2004,
Eq. 28; the original equation is Kolesik, Moloney and Mlejnek 2002). The linear operator is
diagonal in $\omega$; on an envelope grid without a third harmonic it is

$$D = -i\big(\beta - \beta_0 - \beta_1(\omega - \omega_{\rm ref})\big) - \alpha/2,
\qquad \beta = \tfrac{\omega}{c}\operatorname{Re} n_{\rm eff},
\qquad \alpha = \tfrac{2\omega}{c}\operatorname{Im} n_{\rm eff},$$

in the internal FFT convention of [conventions.md §2](conventions.md). A field grid, and an
envelope grid with a third harmonic, keep $\beta_0$ (a field grid measures the frame term as
$\beta_1\omega$); every grid adds the spectral absorber rate $-\alpha_{\rm abs}/2$, and a modal
route the true-power term $-\tfrac12\partial_z\ln\operatorname{Re}n_{\rm eff}$
([the linear operator](../guide/numerics/linear-operator.md)). The nonlinear term is the
polarisation, transformed and multiplied by
$-i\omega/(2\varepsilon_0 c\operatorname{Re}n_{\rm eff})$.
Because $D$ is diagonal it commutes with itself along $z$, so with the **cumulative
operator** $\Phi(\omega, z) = \int_0^z D\,\mathrm{d}z'$ the factor
$\exp[\Phi(z_2) - \Phi(z_1)]$ is the *exact* linear flow between any two positions, however
far apart (ADR-005). That is the interaction picture: a stage evaluates $N$ once and applies
$\exp(\pm\Delta\Phi_i)$ either side of it, and the picture is re-anchored at the start of
every step, so no exponent grows with the run.

**Where.** {py:func}`solprop.linear.dispersion_operator` and
{py:class}`solprop.linear.Frame` assemble $D$; {py:func}`solprop.translate.linop.delta_phi`
returns the exponent; {py:func}`solprop.backends.numpy_ref.passes.stage_input` and
{py:func}`solprop.backends.numpy_ref.passes.stage_output` are the two applications.

**Why this one.** The alternative is an operator splitting, which is what Luna uses.
An earlier prototype study measured both at matched error
([stepper.md §2](stepper.md)): with a shock term the interaction picture leads
by 4.4 times and with plasma by 6.7, and the gap is the evaluation count — six of $N$ per
step against twenty-eight (ADR-004). It integrates the represented diagonal dispersion in the propagator. Rapidly
varying interaction-picture sources, absorbers and coupled-flow residuals can
still restrict the step; exponential treatment does not remove every stability
or accuracy constraint.

**Order and accuracy.** The interaction picture contributes **no truncation error of its
own**: the linear flow is exact for any step, so the scheme's order is the tableau's. What
is approximate is $\Phi$, and only through the interpolation of $D$ (§4). The mode-averaged
GNLSE is the same equation with one mode, a dispersion-specified index and $n_{\rm eff} = 1$
in the prefactor (ADR-026), so it is not a second solver path and inherits all of this —
{py:func}`solprop.nonlinear.gnlse.problem_from` performs the mapping.

**Cost.** Two complex exponentials per element per stage and no transform; on a compiled
backend both signs come from one real `exp` and one `sincos`
([backends.md §3.6](backends.md)). Where each pass of a stage sits is
[stepper.md §1](stepper.md#one-step-pass-by-pass).

**Traps.** Luna applies its $z$-dependent operator by a right-endpoint rectangle rule,
globally first order in the step *and invisible to the error estimate*, because the outer
propagator cancels out of the embedded difference. A zero error estimate is
not a converged answer.

## 2. The Runge–Kutta pairs

This section is the tableau theory of `RungeKutta()`, the explicit march beside the
default integrator. The default, `Multistep()`, is the low-call exponential Adams method:
its formulation and history management are in the [solver guide](../guide/numerics/adams.md)
and the [stepper design](stepper.md#low-call-adams-methods).

**What it is.** An explicit Runge–Kutta pair with an embedded companion: stage derivatives
$k_i = f(z + c_i h,\, y + h\sum_j a_{ij}k_j)$, step $y_{n+1} = y_n + h\sum_i b_i k_i$, local
error estimate $e = h\sum_i (b_i - \hat b_i)k_i$. The *difference* of the weights is what is
stored, because forming it once avoids a cancellation between two nearly equal vectors every
step.

| | DOPRI5 (RungeKutta default) | Tsit5 |
|---|---|---|
| source | Dormand and Prince (1980) | Tsitouras (2011) |
| order, embedded order | 5, 4 | 5, 4 |
| stages, evaluations per step | 7, 6 (FSAL) | 7, 6 (FSAL) |
| coefficients | exact rationals, evaluated once at import | published decimals, no rational form |
| worst order-condition residual | below 1e-15 | 8.0e-16 |
| stability reach, negative real axis | 3.3 | 3.3 |

**Where.** {py:class}`solprop.stepper.tableau.Tableau` with
{py:data}`solprop.stepper.tableau.DOPRI5` and {py:data}`solprop.stepper.tableau.TSIT5`;
{py:func}`solprop.stepper.tableau.check_order_conditions` measures the order the
coefficients attain, tree by tree.

**Why this one.** Explicit rather than implicit or exponential-RK because the interaction
picture has already removed the stiff linear part; what remains is not stiff except through
an absorber (§8). Order 5(4) because the embedded estimate is then free and the evaluation
count is the whole cost. Two schemes because the prototype study ranked them differently on
different rungs — Tsit5 first with plasma, DOPRI5 first with a shock — by no large margin (ADR-004).

**FSAL.** Both tableaux are first-same-as-last: the last stage's input is the accepted
state, so its derivative serves as the first stage of the next step once **rebased** by the
whole step's propagator, $k_1^{n+1} = \exp(\Delta\Phi^n)k_7^n$, because the picture is
re-anchored every step. That is one evaluation in seven. The identity holds only if nothing
changes the state after the last stage: the hard band mask does not, and ADR-025's
in-stage absorber does not, which is part of why it is the default. Which backends take
the saving, and the measurements, are [stepper.md §2](stepper.md).

**Order and accuracy.** Order 5 on the solution and 4 on the companion, by construction: the
17 rooted trees through order 5 are checked against $\sum_i b_i\Phi_i(t) = 1/\gamma(t)$ for
**both** weight vectors, so the labelling of the pair is verified rather than assumed.
Measured orders are in [accuracy.md §1](accuracy.md).

**Cost.** Six evaluations of $N$ per step, each a modal transform pair plus the response
traversal; twelve applications of the propagator.

**Traps.** Luna's `dopri.jl` had `b` and `b_hat` swapped until upstream PR #436, so it
propagated the fourth-order solution and nothing noticed; its `tsit.jl` still stores an
error vector in the slot labelled as fifth-order weights. Only the constants are ported, and
two sum rules — an error vector does not sum to one, a weight vector does not sum to zero —
catch both. The order check uses `math.fsum`: a BLAS dot product reassociates and costs
about 1e-15 on the Tsit5 order-5 trees, enough to read a genuine order 5 as order 4.

## 3. Step control

**What it is.** The backend reduces the embedded estimate to one scalar; the rest is scalar
logic. The **norm scale** is
$\lVert e\rVert_2/(\mathrm{rtol}\max(\lVert y\rVert_2, \lVert y'\rVert_2) + \mathrm{atol})$
over the active band by default (`region="band"` scores the requested band alone) — one
*global* magnitude, so a scaled norm of 1 means the step sat exactly at the requested
tolerance. The **controller** is Söderlind's PI.4.2 rule,

$$\text{factor} = \operatorname{clip}\!\big(S\,e_n^{-\beta_1}e_{n-1}^{-\beta_2},\,
f_{\min}, f_{\max}\big),$$

with $\beta_1 = 3/25$ and $\beta_2 = -1/25$ — Söderlind's $0.6$ and $-0.2$ divided by
$k = q+1 = 5$, the **estimator's** order, not the scheme's — safety $S = 0.8$, growth cap
$f_{\max} = 1.1$, and a lag of one step by default.

**Where.** {py:func}`solprop.stepper.controller.weak_norm_scale` and
{py:func}`solprop.stepper.controller.weighted_norm_scale`;
{py:class}`solprop.stepper.controller.PIController`;
{py:func}`solprop.stepper.controller.limit_step`;
{py:func}`solprop.stepper.schedule.landing` with
{py:class}`solprop.stepper.schedule.StepTargets`. The loop is `backends/driver.py`.

**Why a global magnitude.** Per-element scaling is catastrophic here: the in-band dynamic
range of a hollow-fibre spectrum reaches $10^{17}$ and about a third of in-band bins carry
no light, so a 5e-15 error in an empty bin scores several thousand and the step collapses
before one is accepted ([stepper.md §10](stepper.md), trap 1). The weighted option
redistributes the tolerance across the band with per-bin weights while keeping the magnitude
global, which is what an observable orders below the pump needs; which weighting serves it
best is open.

**Landing, and the equivalent norm.** By default there is no dense output (ADR-009): the
step is clipped so it ends exactly on the next save plane, hard kink or the end of the run,
and the driver *snaps* to the target rather than accumulating $z + h$. `interpolate_planes`
(ADR-039) instead observes save planes inside the steps that cross them, through DOPRI5's
continuous extension. The cap is `dz_max`; the width of the Chebyshev interval *containing*
$z$ is a cap only for a coupled flow or a reject-free controller (ADR-038), and is never the
narrowest interval anywhere, which on a real fibre is four orders below the median. A landing just out of reach is stretched by up
to `stretch`, default 1.1, the controller's own growth cap, so the local error rises by at
most $1.1^5$. Because a clipped step is not the step the controller chose, the controller
is fed the **equivalent norm** $\lVert e\rVert(h_{\rm proposed}/h_{\rm taken})^{q+1}$ —
what the proposal would have scored — while *acceptance* is decided, and the history
written, on the executed step's own norm. Without it every plane restarts the
controller from a sliver that regrows at 1.1 per step.

**Fixed schedules.** `rtol` is a *local* tolerance, so ADR-019 makes fixed stepping first
class: {py:func}`solprop.stepper.schedule.discover` records $h(z)$ from one adaptive run,
{py:func}`solprop.stepper.schedule.calibrate` replays it at $h$, $h/r$ and $h/r^2$ and
measures the observed order by Richardson extrapolation, and
{py:func}`solprop.stepper.schedule.replay` rescales it to a target error. The schedule is
reproducible, because {py:class}`solprop.stepper.controller.PIController` is frozen and its
`propose` is a pure function of the norm history; the replayed run is required to meet the
overall accuracy target, not to reproduce the original run's bits or every adaptive
decision ([stepper.md §8](stepper.md);
[SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)).

**Order and accuracy.** The controller chooses $h$; it does not change the order. The
calibration gate refuses an observed order above $p + 3$
({py:data}`solprop.stepper.schedule.ORDER_MARGIN`): a legitimate excursion reads about half
an order high and a schedule off its convergence curve reads nine orders high, and a
two-level extrapolant cannot tell them apart. A difference within
{py:data}`solprop.stepper.schedule.CONCLUSIVE_FACTOR` of the round-off floor is reported
inconclusive rather than believed. `rtol` does not fix the global error at the output, so
every run records what it achieved ({py:class}`solprop.stepper.controller.ControlRecord`);
how far the two can differ is measured in [stepper.md §3](stepper.md).

**Cost.** A few scalar operations per step, plus one reduction over the in-band mask that
the backend fuses into its accept pass. A weighted norm costs a second traversal, and only a
weighted run pays it.

**Traps.** A proposal below `dz_min` raises at once rather than being clamped;
{py:data}`solprop.stepper.controller.DZ_MIN_RELATIVE` is 1e-9 of the run length, five orders
below the finest legitimate local scale. Luna clamps to `min_dt` *and marks the step
accepted*, so a run that cannot meet its tolerance abandons it with no warning and no
counter. The floor applies to the controller's proposal only; a landing may be as short as
the geometry demands.

## 4. The cumulative operator Φ

**What it is.** The exponent the interaction picture applies, built once per run and
evaluated per stage. It has **three** representations, and
{py:func}`solprop.translate.linop.delta_phi` takes any of them, so the choice can change
without a backend noticing.

*Constant.* A $z$-independent operator gives $\Phi = D z$, degree one: one interval, exact,
and the increment is returned as $D(z_2 - z_1)$ to the last bit rather than as a difference
of two Chebyshev sums.

*The $z$ table.* Adaptive piecewise Chebyshev in $z$ with array-valued coefficients.
Sampling is at Chebyshev–Lobatto points — second kind, so the sets nest and a degree
doubling costs only the new points. An interval is accepted when twice the relative
coefficient tail is at most `tol`, the factor two being the bound
$\lVert f - p_n\rVert \le 2\sum_{k>n}|a_k|$ on the interpolant (Trefethen, theorem 4.2), so
`tol` estimates the uniform error from the tail of the computed coefficients (an estimate,
not a rigorous bound for an unsampled function: [varying operators](../guide/numerics/linear-tables.md)); otherwise the degree doubles to `n_max`, otherwise the interval
bisects. Integration is a coefficient recurrence in the Chebyshev basis — exact for the
polynomial it is given — with the constant fixed from the left so the antiderivative is
continuous by construction. Evaluation is Clenshaw, not Horner on monomials: the same
operation count, far better conditioned at the degrees the builder reaches. Which backends
run the table is [backends.md §1](backends.md).

*The parameter expansion.* When a mode set's whole $z$ dependence runs through declared
scalar paths — a core radius $a(z)$, a number density $\rho(z)$ — the operator is analytic
in those while the paths carry the cusps, so the two are separated:

$$D(\omega; \rho, a) \approx \sum_{jk} d_{jk}(\omega)T_j(\hat\rho)T_k(\hat a),
\qquad
\Phi(\omega, z) = \sum_{jk} d_{jk}(\omega)\,I_{jk}(z),$$

with $I_{jk}(z) = \int_0^z T_j(\hat\rho)T_k(\hat a)\,\mathrm{d}z'$ a table of *scalars*. One
global box per parameter and no splitting in a parameter: a quantity that kinks in one is
not analytic there, and the $z$ table is the representation for it.
Shared mixture fills declare total pressure; independent populations declare labelled
densities. A total-degree expansion for three varying densities meets the
accuracy bounds but does not improve both memory and build cost on both acceptance cases.
It remains an experiment; independent profiles beyond the existing tensor expansion use
the host $z$ table. See the [mixture manual](../guide/media/mixtures.md).

**Where.** {py:func}`solprop.translate.chebyshev.build` and
{py:func}`solprop.translate.chebyshev.build_from_paths` with
{py:class}`solprop.translate.chebyshev.PiecewiseChebyshev`;
{py:func}`solprop.translate.expansion.build_expansion` with
{py:class}`solprop.translate.expansion.ParameterExpansion`;
{py:func}`solprop.translate.linop.build_phi` and
{py:func}`solprop.translate.linop.verify_phi`; the per-stage bundle is
{py:func}`solprop.translate.stage.terms`, {py:func}`solprop.translate.stage.scalars` and
{py:func}`solprop.translate.stage.combine`.

**Why a cumulative integral.** Because a right-endpoint rule is first order in $h$ whatever
the tableau says. A cumulative integral is exact over any step; the only
approximation left anywhere in $\Phi$ is the interpolation of $D$, and
{py:func}`solprop.translate.linop.verify_phi` proves it against `scipy.integrate.quad_vec`
at random pairs, which is the acceptance test of either representation (ADR-005).

**The co-moving frame.** $\beta_0$ and $\beta_1$ are frozen on a reference mode for the
whole set: a $\beta_1\omega$ term is a change of the time variable, so modes given their own
would no longer be simultaneous and the walk-off between them would be silently lost.
$\beta_1$ is **analytic**, from the mode's own `dneff_domega` — a differenced group delay
has a roughness floor near 4.75e-8 *independent of the differencing step*, which would cap
the whole operator at 1e-8 (ADR-006). {py:func}`solprop.modes.base.beta2` is a
Richardson-extrapolated difference of that analytic $\beta_1$, reaching about 1e-10
relative; nothing in the solver consumes it.

**Order and accuracy.** Spectral in the interval width for an analytic integrand: a
degree-$n$ interval converges at order $n+1$, and the build reproduces exactly, to
round-off, any polynomial of degree at most $n_{\max} - 3$ per declared segment, any
piecewise constant, and the round trip `build(f).integral().derivative()`. Agreement with
`quad_vec` is in [accuracy.md §1](accuracy.md). The limit is round-off, not the tolerance:
the frame subtracts $\beta_0 + \beta_1(\omega - \omega_{\rm ref})$ from a $\beta$ reaching
$8\times10^7\, \mathrm{m^{-1}}$, about eight digits of cancellation, so one evaluation of
$D$ carries a relative round-off of order 1e-11 and far more with a collar. No interpolant
can resolve a function below the noise of its own evaluation, so `build_phi` measures that
floor by a second difference before building and raises the request to
{py:data}`solprop.translate.linop.ROUNDOFF_SAFETY` times it, in the acceptance test's own
norm so the two are comparable.

**Cost.** One build per run. Per stage, either one linear combination over `n_terms` host
scalars (the expansion, and the constant as its one-term case) or one Clenshaw pass over an
array of the state's size times the degree (the $z$ table). Memory is why the expansion
exists: $n_{\rm terms}\times$ state against $n_{\rm intervals}n_{\rm deg}\times$ state.
{py:attr}`solprop.translate.problem.KernelProblem.device_ready` is the predicate
that a problem is in the first form; a GPU build refuses what fails it, and the native
CPU build evaluates the table at a trial's positions into terms of its own (SEP 29
section 6.6).

**Traps.** Endpoint samples are inset by one ulp of the interval's own scale: a profile
written `z >= z_stage` returns the post-jump value *at* the boundary, and without the inset
the interval ending there samples the wrong side of its own discontinuity and bisects to the
width floor chasing it. Hard kinks are tracked apart from the far more numerous split edges
— steps land on the first and must not be capped by the second. Both representations cover
five per cent beyond the fibre end, so the last step and the noise phase are defined there.

## 5. Fourier conventions and grids

**What it is.** Two grids implement one protocol. A field grid carries the real $E(t)$ and
stores its one-sided spectrum ($n_\omega = n_t/2 + 1$, `rfft`); an envelope grid carries
$A(t) = E_a(t)e^{-i\omega_0 t}$ and stores the full spectrum in FFT order (`fft`). Both
carry an oversampled **fine** grid on which responses are evaluated: a cubic response
reaches $3\omega_{\max}$, so the fine grid is sampled at `ffac` times the highest frequency
of the window — 6 with a third harmonic, 4 without — the state is zero-padded onto it, the
response evaluated, and the result cropped back. What the crop discards is exactly the
aliasing the fine grid existed to keep out of the coarse one. The **analytic signal** keeps
the zero bin, doubles the positive bins, *keeps the Nyquist bin* and zeroes the rest, so
$\operatorname{Re}E_a$ reproduces $E$ identically. Two masks matter: the **in-band** mask,
which the error norm uses, and the **active** mask,
$\omega_{\min}/2 < \omega < 1.1\omega_{\max}$, outside which the operator and the prefactor
are zero and the hard mask removes the state.

**Where.** {py:func}`solprop.fourier.forward`, {py:func}`solprop.fourier.analytic_signal`,
{py:func}`solprop.fourier.oversample`, {py:func}`solprop.fourier.crop`,
{py:func}`solprop.fourier.energy_omega` and {py:func}`solprop.fourier.spectral_density`;
{py:class}`solprop.grid.FieldGrid`, {py:class}`solprop.grid.EnvelopeGrid` and
{py:func}`solprop.grid.planck_taper`; the kernel's copies are
{py:func}`solprop.translate.masks.build_masks`.

**Why these conventions.** Optics at the interface, FFT inside, the boundary a conjugation
in three named places (ADR-017), so nothing in between has to ask which convention an array
is in. The analytic signal's Nyquist convention is `scipy.signal.hilbert`'s and not Luna's,
because only it makes the real part an identity and the two energy functionals an exact
discrete Parseval pair.

**Order and accuracy.** Parseval is **exact**, not accurate to a quadrature order:
`energy_t` is the trapezoid rule, which on a periodic band-limited grid is spectrally
accurate and reduces to $\delta t$ times a plain sum, and `energy_omega` is its exact
discrete counterpart. Luna uses Simpson's rule, whose fourth-order error is what makes its
two energies disagree with each other. Oversampling preserves energy on a field grid
provided the coarse Nyquist bin is zero, which every grid solprop builds apodises to zero.

**Cost.** One batched inverse transform of $n_{\rm modes}n_{\rm pol}$ columns to the fine
time grid and one forward transform back, per stage.

**Traps.** Neither dtype nor length identifies the two layouts — a real-valued envelope is
an ordinary unchirped pulse — so every function that needs to know takes an explicit `kind`
and checks. `ffac` changes the *whole* grid, not only the fine one, so two values give two
different grids that both satisfy the request; compare them as physical spectra on their
common band, never array to array.

## 6. Transverse quadrature

**What it is.** One fixed product rule over the cross-section, chosen once per problem, with
two matrices built on it: **synthesis** takes modal amplitudes to the field at the nodes and
**projection** takes the polarisation back, with the quadrature weights and the basis
normalisation $\varepsilon_0 c/2$ folded in, so the pair inverts to the identity on an
orthonormal set.

| rule | where | nodes | exact to | error estimate |
|---|---|---|---|---|
| Gauss–Legendre | radius, $\xi = r/a \in (0,1)$ | $n_r$, strictly interior | degree $2n_r - 1$ | none |
| Gauss–Kronrod | radius, when asked | $2n + 1$ | degree $3n + 1$ | the embedded Gauss subset |
| periodic trapezoid | azimuth, offset half a step | $n_\theta$ | trigonometric degree $n_\theta - 1$ | none |
| Gauss–Legendre | both Cartesian axes | $n_x, n_y$ | degree $2n - 1$ | none |

**Where.** {py:func}`solprop.modes.quadrature.transverse_quadrature` builds the rule,
{py:func}`solprop.modes.quadrature.gauss_kronrod` its extension,
{py:func}`solprop.modes.quadrature.mode_matrix` the matrices, and
{py:func}`solprop.translate.modal.multimode_transform` the transform the seam carries.

**Why fixed rather than adaptive.** The Kerr polarisation is cubic, so its projection is a
*quartic* in the transverse profiles — for Marcatili modes, Bessel functions of $r/a$ times
$\cos h\theta$: analytic, and integrated to rounding by a Gauss rule of modest order
(ADR-010). An adaptive cubature buys nothing, costs a data-dependent time per step, and
brings an error estimate that can be fooled — Luna's `pcubature` reported convergence on a
wrong value because on-grid azimuthal nodes at $0, \pi/2, \pi, 3\pi/2$ annihilate exactly
the products the rule exists to integrate. The half-step offset removes that case and leaves
no convergence decision to fool. The Kronrod extension is *computed* by Laurie's algorithm
rather than tabulated, so any order is available.

**Order and accuracy.** Spectral convergence in the node count for analytic integrands,
which is what products of Bessel modes are. A quartic product of modes of azimuthal order up
to $h$ reaches trigonometric degree $4h$, so the azimuthal rule needs $n_\theta \ge 4h + 1$
and `transverse_quadrature` warns below it. The Kronrod difference is an **alarm, never a
controller**: it is reported as a statistic and the rule is never adapted from it. The
measured ladder is in [accuracy.md §2](accuracy.md).

**Cost.** Two GEMMs per stage, skinny in the mode count and wide in time. The native backend
therefore uses ordinary KokkosBlas GEMM rather than a batch of tiny GEMMs, and stacks the fine
and coarse projection matrices when an embedded rule is present. Its local hybrid build uses
vendor BLAS only for measured winning whole-transform shapes; chunks and two-component state
retain the explicit native kernel. The alarm costs one extra projection but no second
nonlinear response. A taper costs nothing: for a scale-invariant mode the normalised field
goes as $1/a$ at nodes placed at $r = a\xi$ and the weights as $a^2$, so rescaling both
matrices by $a(0)/a(z)$ is **exact**.

**Traps.** A rule with no embedded pair reports an error estimate of identically zero, which
must not be read as convergence —
{py:attr}`solprop.modes.quadrature.TransverseQuadrature.has_embedded_rule` is the predicate.
The mode normalisation is taken from each mode's closed form and not from the rule the
matrices use, or the orthonormality test would be a tautology.

## 7. The nonlinear responses

**What it is.** A response says what polarisation a medium produces from the field at a
point; the translation layer compiles the declared set into one fused time-domain traversal
(ADR-011), and the prefactor $-i\omega/(2\varepsilon_0 c\operatorname{Re}n_{\rm eff})$ turns
the result into a rate of change of the amplitude. That $\omega$ dependence **is** the
optical shock term, so switching the shock off is not a different equation but the same
prefactor with $\omega$ frozen at the carrier
({py:func}`solprop.nonlinear.normalisation.uppe_prefactor`).

**Kerr.** Instantaneous and pointwise, $P = \varepsilon_0\chi^{(3)}\rho\times$ (a cubic form
of the field). On a field grid the full cubic already carries the third harmonic and the
counter-rotating terms; on an envelope grid the fundamental part is
$P_p = \tfrac12(\sum_q|a_q|^2)a_p + \tfrac14(\sum_q a_q^2)\bar a_p$, and a harmonic term
needs an explicit carrier $e^{2i\omega_0 t}$ because the state has the carrier removed
({py:class}`solprop.nonlinear.responses.KerrResponse`,
{py:func}`solprop.nonlinear.responses.thg_carrier`).

**The general modal route's tensor cubic.** At each quadrature node the route contracts
the local three-component field with the region's rank-four tensor,
$P_i = \varepsilon_0 A_{ijkl}E_jE_k\overline{E_l}$. On the real field (the default form)
the product $E_jE_kE_l$ is symmetric in $(j, k, l)$, so the sum is evaluated as
$\sum_m C_{im}\,(E_jE_k)E_l$ over the ten monomials $j \le k \le l$, with $C_{im}$ the
sum of $A_{ijkl}$ over the orderings of monomial $m$ — an identity for any tensor —
formed once per node and evaluation: 3 + 10 products a component a sample in place of
the dense 81 + 3, 2.5 times faster in `numba_cpu`'s kernel
([benchmarks.md](benchmarks.md#general-route-symmetric-contraction)). The
order is fixed: the orderings in lexicographic order, the monomials likewise, every sum
from zero; `numpy_ref` defines it and both compiled backends run it. Its rounding is
within $\gamma_{17}$ of $\sum_{jkl}|A_{ijkl}||E_j||E_k||E_l|$, against $\gamma_{14}$ for
the dense order. The analytic form, symmetric in $(j, k)$ only, and the delayed tensor
contraction against a convolved history keep the dense order.

**Ionisation and the plasma current.** Two contributions form the total current: the
free-electron current $J_{\rm free} = (e^2/m_e)\int fE\,\mathrm{d}t$ and the loss current
$J_{\rm loss} = I_p W(1-f)E/|E|^2$, whose form is chosen so that
$E\cdot J_{\rm loss} = I_p\,\mathrm{d}f/\mathrm{d}t$ **exactly** — the field pays one
ionisation potential per electron freed. The ionised fraction is
$f = 1 - (1-f_0)\exp(-\!\int W\mathrm{d}t)$, with the pre-ionised fraction *inside* the
exponential; Luna adds it outside, which lets $f$ exceed one and breaks that identity.

For a mixture, each channel retains that independent exposure recurrence, its rate table,
potential and pre-ionisation. The common current and work integrals use
$\sum_c\rho_c f_c E$ and $\sum_c\rho_c I_{p,c}W_c(1-f_c)$ respectively. Numba and
native specialise the one-channel path at compile time and preserve its operation order;
the multiple-channel loop adds weighted contributions in mixture order. The new node
statistics retain every component, including zero rows for non-ionising populations.
Final ionised inventory includes pre-ionisation; work done during the pulse counts
$f_c-f_{0,c}$.

**The rate.** solprop keeps no rate formula of its own
([SEP 3](seps/archive/sep-0003-optmat-materials.md), D1): every built-in rate is one of optmat's
curated models — short-range PPT forms with Tong and Lin's barrier-suppression
correction for the atoms, orientation-averaged molecular models for the molecules —
adapted to the rate protocol by {py:class}`solprop.nonlinear.optmat_rates.OptmatRate`,
which makes the rate exactly zero below the field where it falls to a declared
negligible rate and holds it above the model's maximum field. A caller's own rate meets
{py:class}`solprop.nonlinear.ionisation.IonisationRate`; the Keldysh parameter
({py:func}`solprop.nonlinear.ionisation.keldysh`) and the barrier-suppression field are
diagnostics, not rates.

**The rate table.** Evaluating a rate model directly costs microseconds per sample and a propagation
wants it once per fine sample per column per stage, so it is tabulated: a cubic spline of
$\log W$ against $\log E$ on **uniform log-spaced knots**, so locating a knot is arithmetic
rather than a search. The count doubles until the interpolant meets its requested relative
tolerance against direct evaluation, and the table is exactly zero below `field_min` and
holds its last value above `field_max` rather than extrapolating
({py:func}`solprop.translate.ionisation.build_rate_table`).

**Raman.** A convolution, $P(t) = \varepsilon_0\rho E(t)\int h(t-t')E^2(t')\,\mathrm{d}t'$,
over a damped kernel: a damped oscillator, a molecular vibration, the non-rigid-rotor
rotational ladder with its Boltzmann populations, optmat's sampled shape of fused silica,
or a caller's intermediate-broadening line set. It is evaluated as a product of transforms on a **doubled, zero-padded** grid,
because a transform product is a *circular* convolution and hydrogen's rotational coherence
lives about 2.9 ns against a window of tens of picoseconds — the wrapped tail would be
almost the whole response. The expensive part, the undamped model responses, is sampled once
per grid; the dephasing is one exponential per term at a decay rate that is a scalar path
along $z$, and the damped kernel spectrum is **memoised** by the exact ordered tuple of
decay rates. Density alone is not a cache key: constant-rate glass and constant-pressure gas
transform once, while a pressure-dependent dephasing path rebuilds only when its stage tuple
actually changes ({py:func}`solprop.translate.raman.build_raman_kernels`,
{py:func}`solprop.translate.raman.kernel_spectrum`).

With several Raman components, the effective kernel is the sum of their kernels weighted
by their own partial densities. All changing amplitude densities and decay rates enter
the cache key, and the summed kernel still requires one convolution. The default gas
decay law receives total density as the documented mixture linewidth approximation;
explicit component dephasing overrides it. Selection uses available models and an
explicit unavailable selection raises an error naming the species.

**Order and accuracy.** The Kerr term is pointwise and exact. The exposure and
free current use a running integral along $t$. Under `integration="trapezoid"` it is
second order for a slowly varying integrand. Its Fourier multiplier is
$-i(\delta t/2)\cot(\omega\,\delta t/2)$, compared with the exact $-i/\omega$:
the ratio is $(\theta/2)\cot(\theta/2)$ with $\theta=\omega\,\delta t_{\rm fine}$.
The default field grid reaches $\theta=\pi/3$ at $\omega_{\max}$, giving 0.907.
The production source does not integrate the current into a polarisation in time
([SEP 27](seps/archive/sep-0027-held-plasma-current.md)): it uses
$A_J=\operatorname{DFT}[gJ]/(i\omega)-\operatorname{DFT}[g'J]/(i\omega)^2$ on active
bins, with exact zeros elsewhere, so the free-current dispersion has one running-integral
multiplier and converting the loss current to its source has none. The physical
`PlasmaResponse.evaluate` oracle integrates the current to an ungated polarisation.

The band rule (`integration="band"`,
[SEP 18](seps/archive/sep-0018-band-exact-plasma-integrals.md)) keeps the running sums and filters
each integrand first. The filter has $2K+1$ symmetric taps designed in translation for
the grid, so that the composite multiplier is within a certified $10^{-5}$ of one up to
$1.1\,\omega_{\max}$. It uses $K = 4$ on the default grid and is refused below
`ffac` ≈ 3.55.

- **Ends.** The window's ends are closed by point reflection.
- **Exposure.** It is floored at zero, pointwise.
- **Loss current.** It keeps the raw rate.
- **Cost.** No transform, and per filtered scalar $K$ pair additions, $K+1$
  multiplications and $K$ accumulations, in a fixed order: centre first, then each
  pair summed before it is weighted. On `numba_cpu` it is a column of whole-row passes
  of its own ([backends.md §3.12](backends.md#312-the-plasma-columns-band-rule)); its
  measured cost is in [benchmarks.md](benchmarks.md#band-rule-plasma-integration).
- **Look-ahead.** Each integral reads $K$ samples ahead. Production needs two
  complete scans: exposure and free current. The physical-polarisation oracle
  retains the third scan. Whole-row compiled passes finish each integrand before
  its filter reads it.

The field route's held source, the envelope plasma and the TDSE current stream
convert their currents with exact inverse-frequency factors on the active bins.
Only the field route's exposure and free-current recurrences use the running rule. The measured effect, the multipliers and the band rule's records are in
[accuracy.md §1](accuracy.md). The density's values are
ordered recurrences, but a GPU may evaluate them with a deterministic fixed-tree inclusive
scan after parallel rate evaluation; the native CUDA path does so with one team per node.
The CPU path retains serial traversal of $t$. A cubic interpolant has error
$h^4|f''''|/384$, so the rate table
is **fourth order in the knot spacing**. Measured orders for both are in [accuracy.md
§1](accuracy.md). The Raman convolution on the doubled grid is exact to the transform's own
round-off.

**Cost.** One fused traversal of $t$ per node column for the whole nonlinear term. The Raman
term adds one real transform pair on a grid of twice the length, plus one kernel transform
per distinct decay-rate tuple — against $n_{\rm modes}n_{\rm pol}$ transforms per stage if
it were recomputed per column. How each backend plans these transforms is
[backends.md](backends.md).

**Traps.** Clip the rate: $(I/I_{\rm ref})^{16}$ overflows on a diverged coarse step and one
NaN poisons the ladder ([stepper.md §10](stepper.md), trap 9).
{py:data}`solprop.nonlinear.ionisation.MAX_RATE` is one ionisation per atomic unit of time,
at which nothing physical is discarded. The loss current divides by the field and is finite
*only* because the rate is exactly zero below threshold, so a clipped-but-non-zero table
would be unusable. A corner in $\log W$ inside the range a table spans — a clamped
barrier-suppression correction, or the hold at an optmat model's maximum field, which is
why the table stops there — is where a cubic converges at **first** order: 257 knots
for a smooth rate against 5e5 for argon with Zhang's clamped correction.

## 8. Absorbing boundaries

**What it is.** Both axes are finite and periodic, so energy reaching an edge wraps round
rather than leaving. The collar profiles of {py:class}`solprop.grid.Collar` mark where
propagation is absorbed instead of modelled, and this becomes a **rate** with the reference
length $\ell = \text{length}/\texttt{boundary\_N}$: for the spectral collar

$$\alpha(\omega) = -\frac{2\log W(\omega)}{\ell},$$

so that traversing one reference length attenuates the field by exactly $W$, and for the
temporal collar the bounded profile

$$\alpha(t) = \frac{A\,(1 - W(t))}{\ell}, \qquad A = 12.6,$$

which carries the capped logarithm's integrated attenuation with a fifth of its largest
rate. Three mechanisms: the spectral collar is diagonal in $\omega$ and folded
into $D$ before $\Phi$ is integrated, so it is exact and **free**; the temporal collar is
diagonal in $t$ and cannot ride the propagator; the hard band limit outside the active
mask stays a 0/1 mask, idempotent, and never had the problem.

**Where.** {py:func}`solprop.boundaries.collar_rate`,
{py:func}`solprop.boundaries.temporal_collar_rate` and
{py:func}`solprop.boundaries.absorber`;
{py:func}`solprop.boundaries.nonlinear_absorber_term` and
{py:func}`solprop.boundaries.split_step_factor` are the two temporal applications;
{py:func}`solprop.boundaries.stable_step`; {py:func}`solprop.boundaries.apply_band_mask`.

**Why a rate.** A per-step multiplicative window gives total absorption $W^n$ for however
many steps the controller happened to take, so the physics becomes a function of `rtol`:
refining sixteen-fold erodes the spectral wings fifty-fold, and the limit is not the taper
but a brick wall at the edge of the flat region (ADR-007). As a rate it is exact,
because traversing a distance attenuates by a factor that depends only on the distance.

**Which temporal application, and its order.** The default (ADR-025) is the term
$-\alpha(t)E(t)/2$ added inside every *stage*. The alternative, a factor
$\exp(-\alpha(t)h/2)$ once per accepted step, is a **Lie splitting** of a term that does not
commute with the linear flow: its splitting error is $O(h^2)$ local and $O(h)$ global, so it
caps the whole scheme at **first** order wherever the collar does any work, and the energy
it removes drifts with the step count at the same order. The in-stage term keeps the
scheme's own order. The two agree to first order in the step and differ at second — one
Euler step of the term is the first two terms of the factor. The in-stage term is
**conditionally stable**: the collar's ceiling rate becomes an eigenvalue of the linearised
problem, so an explicit method needs $h\alpha/2$ inside its stability region, and
{py:func}`solprop.boundaries.stable_step` returns $2R/\max\alpha$ with $R = {}$
{py:data}`solprop.boundaries.STABILITY_RADIUS`. The split factor integrates the same term
exactly and is unconditionally stable, which is the one thing it is better at; both stay
implemented behind `absorber_scheme`, and the measured orders are in ADR-025.

**Cost.** The spectral rate is free. The in-stage term costs one extra modal transform per
stage, on the fine grid because $\alpha(t)E(t)$ has more bandwidth than $E(t)$ — and, being
the last thing that does *not* change the state after the last stage, it is what buys FSAL.

**Traps.** The stability limit binds only what cannot discover it. An adaptive run keeps
the physics safe for itself, because the rate reaches its ceiling only where the field is
negligible; a replayed schedule and a floor `dz_min` above the limit cannot, so the kernel
problem's validation refuses them before the run rather than letting them diverge. Safe physics is not bounded rounding:
while the collar's content is below the error tolerance, the controller may take steps
past the limit, and the unstable collar then amplifies rounding-level differences, by about
4.5 × 10⁴ over ten steps three times the limit on a reduced capillary window. The answer is
unchanged, but two backends, or two thread counts, can then differ far above rounding on a
short window; a comparison of that kind steps below the limit. {py:data}`solprop.boundaries.MAX_ALPHA_LENGTH` caps
the spectral rate where the collar reaches exactly zero and $-2\log 0$ would be infinite;
the temporal rate is bounded by construction at
{py:data}`solprop.boundaries.TEMPORAL_ALPHA_LENGTH`, a fifth of the cap, because as the
capped logarithm it cost the scheme 40 % of its steps through the error estimate before it
ever cost stability.

## 9. Noise

**What it is.** One realisation of the vacuum field: one photon per mode per frequency bin,
$|E_k| = \sqrt{\hbar\omega_k/\delta\omega}\,\sqrt{2\pi}/(2\delta t)$ on a field grid and the
same without the two on an envelope grid, with a phase uniform on $[0, 2\pi)$ as the only
random quantity. The amplitudes pair with the prefactors of
{py:func}`solprop.fourier.spectral_density`, so $S(\omega_k)\delta\omega = \hbar\omega_k$ in
every active bin. Two models share the draw: the **input** model adds it to the field once
at $z = 0$; the **modified** model keeps it as a source injected at every stage
({py:func}`solprop.noise.input_noise`, {py:func}`solprop.noise.modified_noise`).

**Why.** The modified model seeds spontaneous processes without the artificial
four-wave-mixing phase matching a propagated background produces. solprop advances it by the
**unitary** part of the linear flow, $N(z) = N_0\exp(i\operatorname{Im}\Phi(\omega, z))$
({py:func}`solprop.translate.linop.unitary_phase`): it disperses, but neither grows nor
decays (ADR-008). Luna holds it constant in $z$, exact only in the dispersion-free gain
model it came from — 1.4 to 6.6 times per frequency for spontaneous modulation instability.

**Cost.** No extra transform: the same table serves the field and the noise, and the
addition happens in the frequency domain inside the stage's own propagator pass — one
complex multiply-add per element per stage.

**Traps.** The propagating field never contains the modified noise; it is restored on
output. Expect roughly eight times as many steps as the input model, because the source is
broadband and never decays.

## 10. Statistics

**What it is.** The reductions taken at a saved plane. Four need care. **Peak power** sums
the polarisation components of one mode before taking the maximum, because they are two
components of one field, and `peak_power_total` sums the *modes* first too — the peak of the
total power is not the largest of the per-mode peaks, which arrive at different times.
**Full width** is by crossing: the half-maximum level comes from the global peak with no
baseline subtraction, and the crossings are found by walking *outwards from the peak* to the
first sample below the level, so a dispersive wave or a noise pedestal does not widen the
answer; `fwhm_t_outer` is the other convention. The **spectral centroid**
$\omega_0 = \sum_k\omega_k S_k/\sum_k S_k$ is an energy-weighted mean, so it is per column
and the whole field's centroid is the energy-weighted mean of those. The **zero-dispersion
wavelength** is a root search on $\beta_2$ per mode, off by default.
{py:func}`solprop.stats.collect`, {py:func}`solprop.stats.peak_power_total`,
{py:func}`solprop.stats.fwhm_t`, {py:func}`solprop.stats.omega0`,
{py:func}`solprop.stats.zdw`.

**Why at planes, and why two-stage.** Luna evaluates every statistic on every accepted
step, with a full inverse transform each time and the root search among them, at about a
fifth of a step (ADR-012). Here they are evaluated where a field is saved, and between planes a named subset may be
evaluated every $N$ accepted steps through the same `stats` a plane uses, so a row is
the plane's own number to the statistic's rounding bound ([backends.md §1](backends.md)).
**The reductions are two-stage**: a kernel writes one partial per column and the driver
combines them. Their order is free
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): repeats and thread counts are
asserted against the reduction's rounding bound, not for identical bits, and a norm that
moves in its last bit may fork an adaptive trajectory, which the accuracy envelopes of
adaptive comparisons allow for.

**Cost.** One pass per plane; the backend statistics — electron density, peak rate, on-axis
intensity, the quadrature alarm — reuse the response column's own first half rather than
re-synthesising every node.

**Traps.** Undefined is not an error: a width with no crossing returns `nan`, while a
*missing dependency* raises. Crossings use the stable form
$x_1 + (\text{level} - y_1)(x_2 - x_1)/(y_2 - y_1)$; forming a slope and intercept instead
loses most of the significand on a time axis of order 1e-12.

## 11. Modes and materials

**What it is.** Marcatili and Schmeltzer's asymptotic expansion for a hollow dielectric
waveguide whose core is large compared with the wavelength,

$$n_{\rm eff} = \sqrt{\varepsilon_{\rm co}
- \Big(\frac{u_{nm}}{ka}\Big)^{2}\Big(1 - \frac{i v_n}{ka}\Big)^{2}},$$

with a `"reduced"` variant expanded to leading order in $1/(ka)$; a positive imaginary part
is loss. Materials are per-particle susceptibilities $\gamma(\omega)$ in $\mathrm{m^3}$, so
$\chi^{(1)} = \gamma\rho$ and a pressure gradient is a scalar multiplier on one fixed
spectral profile. The number density comes from a Helmholtz-energy equation of state, which
departs from ideal by about 0.1 % at 2 bar and 2.5 % at 40 bar for argon at room
temperature.

**Where.** {py:class}`solprop.modes.capillary.MarcatiliMode` with
{py:class}`solprop.modes.capillary.MarcatiliTerms`; {py:func}`solprop.modes.base.beta1` and
{py:func}`solprop.modes.base.zdw`; {py:func}`solprop.physdata.sellmeier.gamma`;
{py:func}`solprop.physdata.equation_of_state.density`;
{py:class}`solprop.profiles.PressureGradient`;
{py:func}`solprop.physdata.species.ionisation_potential`.

**Why analytic derivatives.** optmat returns each gas law's exact frequency derivative,
and every formula type a caller may supply is a rational function of $1/\lambda^2$, so
its derivative is closed form — a *requirement*, not a convenience, because the co-moving frame needs $\beta_1$ to about 1e-10 and a differenced
one has a roughness floor near 5e-8 (ADR-006). The capillary chains through those and forms
$n_{\rm eff} - 1$ directly rather than building $1 + \chi/2$ and subtracting one, which
loses about eight digits at the 1e-4 susceptibility of a gas fill. The mode also
declares that $z$ reaches it only through $a(z)$ and $\rho(z)$, which is what makes the
parameter expansion of §4 possible.

**Order and accuracy.** Analytic where it matters; the differenced quantities state their
floors ({py:data}`solprop.modes.base.BETA2_RELATIVE_STEP` is chosen where truncation and
rounding cross). Density is tabulated over the pressure range once and interpolated by a
cubic spline whose derivative is exact, so $\mathrm{d}\rho/\mathrm{d}z$ stays analytic
wherever the pressure path is.

**Traps.** A steady-flow fill has $p(z) = \sqrt{p_0^2 + (z/L)(p_1^2 - p_0^2)}$, whose slope
diverges as $p \to 0$: a fill reaching a true vacuum degrades every integration scheme
however small the step ([stepper.md §10](stepper.md), trap 5), so
{py:class}`solprop.profiles.PressureGradient` refuses an exactly-zero stage end and
substitutes a floor, logging what it did.

## 12. The compiled backend

How the compiled backends execute these algorithms — the fused traversal, why stages are
never materialised, FFTW plans and wisdom, the thread pools and the regime rule — is
[backends.md §3](backends.md#3-numba_cpu-the-compiled-in-tree-backend).

## References

- J. R. Dormand and P. J. Prince, *J. Comput. Appl. Math.* **6**, 19 (1980),
  doi:10.1016/0771-050X(80)90013-3 — the DOPRI5 pair.
- Ch. Tsitouras, *Comput. Math. Appl.* **62**, 770 (2011), doi:10.1016/j.camwa.2011.06.002 —
  the Tsit5 pair.
- E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I*, 2nd
  ed., Springer (1993), §II.4 — embedded pairs, order conditions and stability regions.
- G. Söderlind and L. Wang, *J. Comput. Appl. Math.* **185**, 225 (2006),
  doi:10.1016/j.cam.2005.03.008 — the PI.4.2 step-size controller (their Eq. 4 and Table 1),
  from G. Söderlind, *Numerical Algorithms* **31**, 281 (2002).
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (2013), theorem 4.2
  — the Chebyshev aliasing bound the build's tolerance rests on.
- D. P. Laurie, *Math. Comp.* **66**, 1133 (1997), doi:10.1090/S0025-5718-97-00861-2 — the
  Gauss–Kronrod construction; the rules themselves are Kronrod's (1965).
- E. A. J. Marcatili and R. A. Schmeltzer, *Bell Syst. Tech. J.* **43**, 1783 (1964) — the
  capillary modes.
- M. V. Ammosov, N. B. Delone and V. P. Krainov, *Sov. Phys. JETP* **64**, 1191 (1986) — the
  ADK tunnelling rate, behind optmat's quasistatic models.
- A. M. Perelomov, V. S. Popov and M. V. Terent'ev, *Sov. Phys. JETP* **23**, 924 (1966) —
  PPT, behind optmat's atomic models; the Ilkov–Decker–Chin form is *J. Phys. B* **25**,
  4005 (1992).
- X. M. Tong and C. D. Lin, *J. Phys. B* **38**, 2593 (2005), doi:10.1088/0953-4075/38/15/001
  — the barrier-suppression correction optmat's atomic models carry; Zhang, Lan and Lu,
  *Phys. Rev. A* **90**, 043410 (2014), is the alternative.
- M. Geissler et al., *Phys. Rev. Lett.* **83**, 2930 (1999) — the plasma current.
- D. Hollenbeck and C. D. Cantrell, *J. Opt. Soc. Am. B* **19**, 2886 (2002),
  doi:10.1364/JOSAB.19.002886 — the silica Raman shape of optmat's default bundle; the molecular models follow Gao et
  al., *Laser Photonics Rev.* **16**, 2100426 (2022), doi:10.1002/lpor.202100426.
- D. J. A. McKechan, C. Robinson and B. S. Sathyaprakash, *Class. Quantum Grav.* **27**,
  084020 (2010), eq. (7) — the Planck taper the collars use.
- M. Kolesik and J. V. Moloney, *Phys. Rev. E* **70**, 036604 (2004),
  doi:10.1103/PhysRevE.70.036604, Eq. (28) — the propagation equation this page marches;
  M. Kolesik, J. V. Moloney and M. Mlejnek, *Phys. Rev. Lett.* **89**, 283902 (2002),
  doi:10.1103/PhysRevLett.89.283902 — the original unidirectional equation.
