(primer-stepping)=
# 12. Marching in $z$

The UPPE has a particular structure, which every good method for it exploits. Its linear
part is large, oscillatory and exactly solvable: a phase and a decay for each frequency and
mode. Its nonlinear part is comparatively small and slowly varying, but expensive to evaluate,
and it couples everything. Over half a century, propagation codes have moved from splitting
the two parts, to integrating the linear part exactly and the nonlinear part with a Runge–Kutta
method, to adaptive control of the step, and most recently to exponential multistep methods
that reuse past evaluations of the nonlinear term.

This chapter follows that history, and at each stage says what solprop does and why. It covers
the split-step method, the interaction picture, solprop's exact treatment of a linear operator
that varies along $z$, the embedded Runge–Kutta pair and its error control, the
exponential Adams method, absorbers expressed as rates, linear coupling between modes, and fixed
step schedules. Throughout, $\partial_z\hat A=D\hat A+N[\hat A]$, with $D$ the linear operator of
§2.9 and $N$ the nonlinear term.

## 12.1 Split-step methods

The split-step Fourier method alternates between two problems, each easy on its own: the linear
part, solved exactly by a phase in the frequency domain, and the nonlinear part, solved in the time
domain. Hardin and Tappert (1973) described it in an abstract for the SIAM 1972 Fall Meeting. They
advanced "the solution using only the nonlinear or variable coefficient term by means of an implicit
finite difference approximation", then advanced it "exactly using only the linear, constant coefficient
term by means of the FFT". They reported the method "second order accurate and unconditionally
stable". Tappert and Judice (1972) had already used it for the Korteweg–de Vries equation. Fisher and
Bischel (1973) independently developed the same alternation for optical self-phase modulation with
dispersion, cutting the medium into thin segments, each traversed "once as a nonlinear segment and
once as a dispersive segment". In 1975 they tested convergence by changing the order of the
alternation and by refining the segments, an early example of checking a result by an independent
change of scheme.

The symmetric version, $e^{hD/2}e^{hN}e^{hD/2}$, has a local error of $O(h^3)$ and a global error of
$O(h^2)$ (Sinkin et al. 2003, Eq. 3). Sinkin et al. compared ways of choosing the step and found
the best to be a local-error method. That method takes one step of $2h$ and two of $h$, estimates the
error from their difference, and extrapolates, at a cost of 50% more Fourier transforms. Hult (2007)
made the decisive observation for what follows: "none of the aforementioned Fourier split-step schemes
can possess a global accuracy exceeding that of the numerical scheme used to integrate the nonlinear
term". The order of a scheme is limited by how well it integrates the nonlinearity.

## 12.2 The interaction picture

The interaction picture removes the linear part entirely by a change of variable. Lawson (1967) wrote
it for general stiff systems. With $z(t)=e^{-tA}y(t)$ the equation for $z$ has only the transformed
nonlinear term, and his Eq. 2.6 is the exact step

$$
y(t+h)=e^{hA}y(t)+\int_0^he^{(h-\theta)A}\big\{f[t+\theta,y(t+\theta)]-Ay(t+\theta)\big\}\,\mathrm d\theta .
$$

Lawson then applied a classical Runge–Kutta method to the transformed equation. In his words the aim
was to make the eigenvalues of the transformed problem "small enough that accuracy rather than stability
should dictate the step-size". Hult (2007) brought the same idea to the GNLSE as the fourth-order
Runge–Kutta in the interaction picture (RK4IP). He anchored the transformation at the step midpoint, which
halves the number of transforms to eight per step, four evaluations of the nonlinear term and four of the
linear exponential, and gives global fourth order (his Eqs. 12a–f). Couairon et al. (2011, Eqs. 180–182) state
the same method for the UPPE, in "native variables", re-aligned after every step. In Kolesik and Moloney's
(2014) phrase, the step is then "controlled solely by the strength of the nonlinear wavepacket
reshaping".

The interaction picture has one weakness, which matters for what follows. In the transformed variable,
a nonlinear source that is smooth in the laboratory frame rotates with the linear phase of its own
frequency. Cox and Matthews (2002, Eq. 32) show that the error constant of an integrating-factor method
is larger than that of an exponential time-differencing method by a factor of order $c^2$, where $c$ is
the linear rate. Hochbruck and Ostermann (2010, Example 2.32) show that Lawson's method applied to a
second-order Runge–Kutta scheme has stiff order one. In practice, a bin driven by other frequencies, as the
source of a dispersive wave is, limits the step to about one radian of linear phase. §12.6 returns to this.

## 12.3 A linear operator that changes along $z$

When the guide or the gas changes along the fibre, so does $D$. The exact linear flow from $z_1$ to $z_2$
is then $\exp[\Phi(z_2)-\Phi(z_1)]$, with the **cumulative operator**

$$
\Phi(\omega,z)=\int_0^zD(\omega,z')\,\mathrm dz' ,
$$

with $D$ the whole bracket of §2.9. Because that bracket includes the true-power term
$-\tfrac12\partial_z\ln w$, its integral is the dispersion-and-loss part plus
$-\tfrac12\ln[w(\omega,z)/w(\omega,0)]$, which is how the guide writes it; the logarithm
appears once. The flow is exact for any step length because $D$ is diagonal and its values commute
([the linear operator](../guide/numerics/linear-operator.md)). The tempting shortcut evaluates $D$ at the end
of each step. That right-endpoint rule is first-order accurate in the step, and the embedded error
estimate of a Runge–Kutta pair cannot see the error, because it lives in the linear flow the estimate treats as
exact. Luna uses a right-endpoint rule. solprop instead computes $\Phi$ to a controlled tolerance, as a piecewise
Chebyshev expansion in $z$
([varying operators](../guide/numerics/linear-tables.md#cumulative-operator)). The table is built adaptively,
in the manner of Chebfun. On each interval the coefficients come from samples at Lobatto points. An interval is
accepted when the tail of its coefficients is negligible, and otherwise its degree is doubled or it is bisected.
Declared breakpoints of the profile, such as the joins of a two-stage fill or the corners of a taper, are
mandatory interval edges. The table is integrated exactly with the Chebyshev antiderivative and evaluated by
Clenshaw's recurrence. A typical fibre needs a few tens of intervals.

For a fill or taper that depends only on a few scalars, the density and the radius for example, solprop can
instead expand $D$ in Chebyshev polynomials of those parameters and integrate only scalar paths along $z$. On a
resonant-dispersive-wave run this reduces the table from 10.3 MB to 0.31 MB.

## 12.4 An embedded Runge–Kutta pair

An adaptive method needs an estimate of its own error. An embedded pair computes two solutions of different
order from the same stages, and their difference estimates the error of the lower-order one. The pair of Dormand
and Prince (1980), RK5(4)7M, is the standard. It has seven stages, and the last stage of one step is the first of
the next, "first same as last" or FSAL, so a step costs six evaluations. It propagates the fifth-order solution, in
"local extrapolation" mode. Their notation is a trap: the *hatted* weights in their table are the fifth-order ones.
Luna's implementation had the two weight columns swapped, which made it propagate the fourth-order solution, and nothing noticed. solprop checks its tableaux against the order conditions, tree by tree, to catch exactly
that ([the Runge–Kutta integrator](../guide/numerics/solver.md#the-rungekutta-integrator)). The pair is also the
basis of MATLAB's `ode45` (Shampine and Reichelt 1997), and Dormand and Prince (1986) gave its continuous extension,
for output between steps at no extra cost.

Balac and Mahé (2013) built an embedded pair for the interaction picture that keeps RK4IP's midpoint structure; an earlier Fehlberg pair used with the interaction picture, which they cite, had lost it. Their ERK4(3)-IP and "coincides with Dormand and Prince Runge–Kutta 4(3) T method", and
they called for "a ERK5(4) method well suited to be used in conjunction with the IP method" as future work. solprop's `RungeKutta()` integrator
is the Dormand–Prince 5(4) pair in the interaction picture, with Tsitouras's 5(4) pair as an alternative. Each step
evaluates the source once per stage, the linear flow is applied by the exact $\Phi$ between stages, and the error
estimate is the error of the propagated state:

$$
y\leftarrow e^{\Delta\Phi}\Big(y+h\sum_ib_ik_i\Big),\qquad e=e^{\Delta\Phi}\,h\sum_i(b_i-\hat b_i)\,k_i .
$$

## 12.5 Measuring the error and choosing the step

A step is accepted if a scalar measure of its error is below one. solprop uses one global scale:

$$
\mathrm{err}=\frac{\|e\|_2}{\mathtt{rtol}\cdot\max(\|y\|_2,\|y_{\rm new}\|_2)+\mathtt{atol}}
$$

over the active band ([the error norm](../guide/numerics/solver.md#the-error-norm)). This deliberately differs from the
per-component test of general-purpose ODE codes, $|e_i|\le r|y_i|+a_i$ (Shampine and Reichelt 1997). The spectrum of a
hollow-fibre run spans seventeen orders of magnitude, and a third of its bins may carry no light at all. Under
per-component scaling, a rounding error of $5\times10^{-15}$ in an empty bin scores several thousand, and every step is
rejected. The global norm has a cost, which the guide states plainly. A weak observable, such as a dispersive wave five
to eight orders of magnitude below the pump, gets no error control of its own. An optional weighting exists for such
cases, and Chapter 13 explains how to check them.

The step is then chosen by a proportional–integral controller,

$$
h_{n+1}=h_n\,\mathrm{clip}\big(0.8\,\mathrm{err}_n^{-3/25}\,\mathrm{err}_{n-1}^{+1/25},\;0.2,\;1.1\big)
$$

([the controller](../guide/numerics/solver.md#the-controller)). The exponents are Söderlind's PI.4.2 gains, $3/5$ and
$-1/5$, divided by $k=5$, one more than the order of the error estimator (Söderlind and Wang 2006, Eq. 4 and Table 1).
Söderlind and Wang argue that such controllers give "computational stability", a smooth relation between tolerance and
accuracy. The safety factor and the tight growth limit of 1.1 are solprop's own choices. Larger growth limits save 15–20%
of steps but cost up to eleven times more field error at tight tolerances. Steps land exactly on output planes and on hard
kinks of the guide. A step that falls below $10^{-9}$ of the length raises an error rather than being accepted silently.

A different approach controls the step by a conserved quantity. Heidt (2009) used the photon number of the GNLSE, which
works well with RK4IP. Heidt himself notes that the per-bin errors "can assume positive and negative values ... which can
cancel each other when integrated", so a small conservation error does not guarantee a small field error. Rieznik et al.
(2012) found the method most efficient with the nonlinear term integrated in the frequency domain, but with a floor set by
round-off. solprop's problems have loss, ionisation, absorbers and Raman, so there is no conserved quantity to control by.
It uses the embedded estimate, and treats conservation as a diagnostic (Chapter 13).

## 12.6 Exponential Adams

A Runge–Kutta step costs six evaluations of the nonlinear term. When an evaluation is expensive, as it is with many modes,
ionisation, Raman, a spatial grid or a Schrödinger response, the alternative is to reuse evaluations from previous steps. That
is a multistep method. solprop's `Multistep()` is an exponential Adams method, and it usually needs one
new evaluation per accepted step ([the Adams integrator](../guide/numerics/adams.md)). It is the default integrator of the
capillary, radial and planar interfaces. The GNLSE and waveguide interfaces default to `RungeKutta()`, which was faster on
them at matched accuracy, and so does Cartesian free space, where the Adams history would hold too many copies of a large
transverse state ([the defaults](../guide/numerics/choosing-a-stepper.md#the-defaults)).

Exponential multistep methods go back, as Hochbruck and Ostermann (2010, §6) record, to Certaine
(1960) and Nørsett (1969). Cox and Matthews (2002) studied them for stiff
systems as exponential time differencing, and Hochbruck and Ostermann (2011) analysed the Adams type. For a constant operator
the method integrates a polynomial through past source samples against the exact linear flow:

$$
u_{n+1}=e^{hD}u_n+h\sum_jw_j(hD)\,F_j,\qquad
w_j(x)=\int_0^1e^{(1-\theta)x}\ell_j(\theta)\,\mathrm d\theta=\sum_kc_{jk}\,k!\,\varphi_{k+1}(x),
$$

with $\ell_j$ the Lagrange basis on the actual step nodes and $\varphi_r(x)=\sum_{m\ge0}x^m/(m+r)!$ (Hochbruck and Ostermann
2010, Eq. 2.10). The linear flow is exact. Only the source is approximated by its polynomial. solprop's construction around
these weights is its own: variable step and variable order from one to five, and a predict–evaluate–correct cycle. An
Adams–Bashforth predictor of order $q$ is followed by one evaluation, then by an Adams–Moulton corrector of order $q+1$, with
the error estimated from the difference. Hochbruck and Ostermann's analysis is for fixed steps, and their paper defers
"variable step sizes and variable order implementations". For a linear operator that changes along $z$, the step uses the
chord of $\Phi$ over the step, with explicit compensation terms. The $\varphi$-functions suffer catastrophic cancellation for
small arguments (Kassam and Trefethen 2005, Table 2.1). solprop computes the moments it needs by a scaled Taylor series with
doubling, and by a recurrence for larger arguments.

The exponential formulation integrates a bin driven by other frequencies at any linear phase per step, where the interaction
picture would limit it to about one radian (§12.2). The opposite holds for a source that rotates with its own bin's phase, and
`Multistep(formulation="interaction")` exists for that case. On the canonical helium-capillary benchmark, the Adams method needs
two and a half to three and a half times fewer evaluations than the Runge–Kutta march at matched accuracy. On the full
resonant-dispersive-wave case it needs twelve to fourteen times fewer, and takes five to six times less time
([choosing a stepper](../guide/numerics/choosing-a-stepper.md#the-short-answer)).

There are costs. The method keeps a history of past evaluations in memory, between 7 and 60 extra copies of the state depending
on the backend. Its stability interval on the negative real axis is much shorter than DOPRI5's, between 0.09 and 1 depending on
order against 3.3, which matters when an explicitly integrated damping term, such as a strong temporal collar, limits the step.
For a cheap source and a small state, a single-mode GNLSE for example, the Runge–Kutta march is 1.2 to 1.4 times faster despite
taking more evaluations, because each of its evaluations is cheap compared with the fixed work of a step.

## 12.7 Absorbers as rates, not masks

A collar (§11.5) could be applied as a mask: multiply the spectrum or the field by the window after every step. Then the total
absorption depends on how many steps were taken, which is set by the tolerance. In solprop's measurement, refining a run sixteen
times erodes the spectral wings fifty times more ([rates, not masks](../guide/numerics/boundaries.md#rates-not-masks)). solprop
instead turns each collar into an absorption rate, so that the absorption is a property of the equation, not of the step
sequence. The spectral collar becomes $\alpha(\omega)=-2\ln W(\omega)/\ell$, added to the linear operator, where it is integrated
exactly and costs nothing. The temporal collar becomes $\alpha(t)=A(1-W(t))/\ell$, with $A=12.6$ and $\ell$ the length divided by
`boundary_N` (20 by default). On the modal routes it is applied inside every stage of the step; on the radial, Cartesian and
planar routes it is applied once per accepted step, unless planes are interpolated.

The in-stage absorber keeps the integrator's full order. A Lie splitting, applying it as a separate sub-step, is first order.
On an N = 3 soliton driven into its collar, the in-stage scheme converges at fourth to fifth order and the split
scheme at first ([the two temporal schemes](../guide/numerics/boundaries.md#the-two-temporal-schemes)). The in-stage rate is
integrated explicitly, so it limits the step to the integrator's stability interval, which binds only on very short runs. The
split integrates the rate exactly and is unconditionally stable; on the spatial routes the in-stage collar took 1.2 to 4.6
times its steps at equal or worse accuracy, so the spatial routes take the split by default
([SEP 21](../dev/seps/archive/sep-0021-absorbing-boundaries.md)).

## 12.8 Linear coupling between modes

The coherent wall and the taper coupling of Chapter 5 make the linear operator non-diagonal in the mode index. The exact
exponential is then a matrix function. solprop keeps the diagonal operator common to all modes, and carries each mode's departure
from it, and the coupling, as a separate coupled linear flow. That flow is applied by a matrix function computed from a Schur or
eigenvector basis. Along a varying guide it is applied section by section, with the variation inside each section treated as a
residual. That keeps the linear part accurate, but the flow is no longer exact in the way the diagonal $\Phi$ is. The guide states
the consequences: a fourth-order Magnus-type section limits such a run to fourth order, and a purely linear coupled run, with no
nonlinear source to drive the error estimate, should use the Runge–Kutta integrator or a bounded step.

## 12.9 Fixed schedules

Adaptive stepping is the right default, but an ensemble of runs, for noise statistics for example, is better served by one fixed
sequence of steps shared by every member. solprop's fixed schedules first discover a step sequence adaptively. They then calibrate
it at $h$, $h/2$ and $h/4$, measuring the observed order and a Richardson estimate of the error. Finally they replay it refined
along the measured exponent to reach a requested error ([fixed schedules](../guide/numerics/fixed-schedules.md)). An observed
order far above the method's own, which is how a failure usually shows itself, is refused. A calibration within a thousand times
of the round-off floor is reported as inconclusive.

:::{admonition} What to remember
:class: tip

- Split-step methods (Hardin and Tappert; Fisher and Bischel) alternate exact linear and approximate nonlinear sub-steps. Symmetric
  splitting is second order. No composite scheme beats the order of its nonlinear integrator.
- The interaction picture (Lawson 1967; Hult's RK4IP) integrates the linear part exactly. Its error constant grows with the linear
  rate for a driven source.
- solprop integrates a $z$-varying operator exactly, $\exp[\Phi(z_2)-\Phi(z_1)]$, with $\Phi$ in adaptive Chebyshev tables. A right-endpoint
  rule per step is first order, and the error of any approximate linear flow is invisible to the error estimate.
- `RungeKutta()` is Dormand–Prince 5(4) in the interaction picture, with FSAL and order-condition checks. The norm is one global scale
  over the active band. The controller is Söderlind's PI.4.2 with a growth limit of 1.1.
- `Multistep()` is a variable-step, variable-order exponential Adams PEC method: one evaluation per step, exact linear flow,
  a memory history, and a short negative-real stability interval. It is the capillary, radial and planar default; the GNLSE,
  waveguide and Cartesian interfaces default to `RungeKutta()`.
- Collars are rates. The spectral rate is exact in $D$. The temporal rate is integrated in-stage at full order on the modal
  routes, and applied once per step, as an unconditionally stable first-order split, on the radial, Cartesian and planar
  routes.
- Coupled linear flows are matrix functions applied by section. Fixed schedules are discovered, calibrated and replayed.
:::

The precise statements are in [the solver](../guide/numerics/solver.md), [the Adams integrator](../guide/numerics/adams.md),
[choosing a stepper](../guide/numerics/choosing-a-stepper.md), [varying operators](../guide/numerics/linear-tables.md),
[absorbing boundaries](../guide/numerics/boundaries.md) and, for the design record, [the stepper notes](../dev/stepper.md).
[Chapter 13](accuracy.md) asks when the answer can be trusted.
