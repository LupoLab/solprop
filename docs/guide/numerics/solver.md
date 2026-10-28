# The solver

The solver advances the field along the propagation direction. Choose its error
tolerance and step controls alongside the temporal and spatial resolution of the
problem. A small local error estimate does not establish that the full simulation
or a weak generated band is accurate.

Start with the interface's default integrator, `Multistep()` on `prop_capillary`,
`prop_radial` and `prop_planar` and `RungeKutta()` on `prop_gnlse`, `prop_waveguide` and
`prop_free` ([the defaults](choosing-a-stepper.md#the-defaults)), then use
[stepper selection](choosing-a-stepper.md) when source cost, memory or dynamics
motivate a comparison. This chapter explains
`rtol`, `atol`, `dz_init` and `dz_max`, the interaction picture and the error norm.
[Fixed schedules](fixed-schedules.md) describe prescribed steps;
[assessing accuracy](accuracy.md) puts those controls into a convergence study.

## The equation and the interaction picture

Propagation is written in the frequency domain as

$$\partial_z \hat{A}(\omega, z) = D(\omega, z)\,\hat{A} + N[\hat{A}](\omega, z),$$

for the diagonal linear part, with $D$ complex (dispersion and loss) and $N$ the nonlinear
term, which costs a Fourier round trip and, in a multimode problem, a pair of transforms
over the mode basis as well. Because $D(\omega, z_1)$ and $D(\omega, z_2)$ commute,
the cumulative operator $\Phi(\omega, z) = \int_0^z D(\omega, z')\,\mathrm{d}z'$ gives
the *exact* linear flow $\exp(\Phi(z_2) - \Phi(z_1))$ over a step of any length. Working
in the interaction picture anchored at $z_n$, with
$\hat{y} = \exp(-(\Phi(z) - \Phi(z_n)))\,\hat{A}$, leaves

$$\partial_z \hat{y} = e^{-\Delta\Phi(z)}\, N\!\left[e^{\Delta\Phi(z)}\hat{y}\right],$$

which an explicit method integrates through source evaluations. The diagonal
homogeneous flow is treated by the exponential, with the accuracy of the
[operator representation](linear-tables.md). The transformed source can still
vary rapidly, and temporal absorption or coupled linear terms can restrict the
step. Removing the homogeneous linear oscillation is not a universal step-size
or accuracy guarantee.

## Stepping and control

## Three integrators

Three integrators take the steps, and everything else on this page — the linear flow,
the source definitions, caps and save-plane conventions — is used by all of them.
`integrator=` on every entry point and on
{py:class}`~solprop.driver.simulation.SimulationSettings` chooses between `Multistep()`,
the exponential Adams method ([the Adams integrator](adams.md)), `RungeKutta()`,
the interaction-picture Dormand–Prince pair below, and `LowStorageRungeKutta()`, a
fixed-schedule RK4; [choosing a stepper](choosing-a-stepper.md) describes each, gives
each interface's default and sets out the measurements and the guidance. Their error estimates and histories differ, so
equal requested tolerances do not guarantee equal measured error.

`LowStorageRungeKutta()` replays a prescribed
[schedule](fixed-schedules.md#the-low-storage-march) and has no error estimate: `rtol`
sets only the recorded tolerance and the spatial collar's diagnostic threshold, every
step's recorded norm is 0.0 ("not estimated"), `dz_max` must be at least the schedule's
longest step and `dz_init` is informational.

## The Runge–Kutta integrator

`RungeKutta()` steps with an interaction-picture Dormand–Prince 5(4) pair, `DOPRI5`;
`RungeKutta(TSIT5)` takes the Tsitouras pair instead. Both are order 5 with an embedded
order-4 companion and both are first-same-as-last, so a step costs **six** evaluations of
$N$ where a backend reuses the last stage as the next step's first, and seven where it
does not (`numpy_ref`, a coupled flow, the `"split_step"` collar). The local error
estimate comes for free as the difference between the two solutions. Both the update and
that difference are carried to the end of the step by the same exponential,

$$
y \leftarrow e^{\Delta\Phi}\Bigl(y + h\sum_i b_i k_i\Bigr),
\qquad
e = e^{\Delta\Phi}\,h\sum_i (b_i - \hat b_i)\,k_i,
$$

so the local error is that of the *propagated* state, not of the interaction variable,
and it is this error that [the error norm](#the-error-norm) scales. In a multimode
problem, where $N$ is an order of magnitude more expensive than everything else, that
evaluation count is the whole argument for this family: an operator-splitting scheme of
comparable order needs several times as many, and loses its closed-form advantage as
soon as self-steepening or plasma is present.

A {py:obj}`~solprop.stepper.tableau.Tableau` carries the nodes, the Butcher matrix, the
propagated weights $b$ and the *difference* $b - \hat{b}$ that forms the error estimate.
Storing the difference rather than $\hat{b}$ avoids a cancellation between two nearly
equal vectors on every step. {py:obj}`~solprop.stepper.tableau.check_order_conditions` measures
what the coefficients actually attain, tree by tree:

```{eval-rst}
.. code-block:: pycon

    >>> from solprop.stepper import DOPRI5, check_order_conditions
    >>> report = check_order_conditions(DOPRI5)
    >>> report.solution_order, report.embedded_order
    (5, 4)
    >>> report.max_residual < 1e-15
    True
```

A tableau whose two weight vectors are labelled the wrong way
round propagates the 4th-order solution while believing it is 5th order: nothing else
about the tableau reveals the mistake, and it costs an order of global accuracy
silently. Tabulated coefficients copied from elsewhere are therefore checked here, tree
by tree, rather than trusted.

## The error norm

The controller consumes one scalar per step: the error estimate reduced over the bins the
norm scores and divided by a scale,

$$\mathrm{err} = \frac{\lVert e \rVert_2}
{\texttt{rtol}\cdot\max\left(\lVert y\rVert_2, \lVert y_\mathrm{new}\rVert_2\right)
+ \texttt{atol}},$$

so that $\mathrm{err} = 1$ means the step sat exactly at the requested tolerance.

The norm scores the whole **active band** ([grids](../setup/grids.md#the-active-band-and-absorbing-collars)):
the requested band and the spectral collars around it, where light generated in the run
propagates before the collar absorbs it. That light feeds back into the band through the
nonlinearity, so it is integrated under control like the rest: scoring the requested band
alone left it unchecked, and a high-order method could get it wrong by as much as the
light the margin held while the band met its tolerance. The controller's
`region="band"` scores the requested band alone.

The
scale is one **global** number, from
{py:obj}`~solprop.stepper.controller.weak_norm_scale`. This matters more than it looks. Scaling per
element instead — the obvious choice, and what most general-purpose integrators do — is
catastrophic on this problem: the in-band dynamic range of a hollow-fibre spectrum
reaches $10^{17}$ and roughly a third of in-band bins carry no light at all, so a
$5\times10^{-15}$ error in an empty bin scores several thousand, every step is rejected,
and the step size collapses to zero before a single step is accepted.

The cost of a global scale is that a weak observable — a dispersive wave five to eight
orders below the pump — gets no error control of its own.
{py:obj}`~solprop.stepper.controller.weighted_norm_scale`, reached through the
controller's `weights=`, is the option for that: it applies per-bin
weights while keeping the magnitude global, so a chosen band can be held to a tighter
tolerance without reopening the collapse. Which weighting actually serves weak
observables best is still an open experiment, so this form is provisional.

## The controller

{py:obj}`~solprop.stepper.controller.PIController` is a proportional–integral controller, Söderlind's PI.4.2
([Söderlind and Wang 2006](https://doi.org/10.1016/j.cam.2005.03.008), Table 1),

$$\text{factor} = \mathrm{clip}\!\left(
\texttt{safety}\cdot \mathrm{err}_n^{-\beta_1}\,\mathrm{err}_{n-1}^{-\beta_2},\;
\texttt{min\_factor},\ \texttt{max\_factor}\right),$$

with $\beta_1 = 3/25$ and $\beta_2 = -1/25$, safety 0.8, growth capped at 1.1
(`max_factor`) and shrinkage at 0.2 (`min_factor`) per step. The
integral term damps the response to fluctuations in the estimate, which is what keeps
the step size smooth enough for a recorded step history to be worth replaying. The gains
are Söderlind's $(0.6, -0.2)$ divided by $k = q + 1$, where $q$ is the order of the
**error estimate** — 4 for an embedded 5(4) pair, not 5. Use
{py:obj}`~solprop.stepper.controller.PIController.for_estimator_order` for any other estimator.

Two controller options are available on every backend. They are designed for device
execution, where waiting for each step's error norm to come back to the host is a
synchronisation; the device backends still return each trial's norm to the shared host
loop, so they change the step sequence and not the synchronisation. The costs
below were measured on the test problems the controller was designed on:

`lag`
: The factor computed from step $n$ is applied to step $n + 1 + \texttt{lag}$, so the
  error norm has a whole step in which to come back from a device asynchronously. Lag 1,
  the default, was measured to cost nothing: error ×1.00–1.05, cost ×0.99–1.00. Lag 2
  was free on most configurations but drifted to ×2.45 on one, so it wants checking per
  problem.

`reject_free`
: Never reject a step whose error norm is finite, letting the lagged update absorb the
  overshoot. Rejections happen 0.0–0.4 % of the time in practice, so this abolishes
  something that was barely happening; it does not remove the current norm transfer.
  With the growth cap and the safety factor it takes about half as many steps again and
  buys a 2.4 to 23 times smaller error, so it is neutral to favourable on efficiency;
  whether it wins at a matched *global* error, and on which problems, is not
  established. It is an explicit option, never a default a backend chooses. A
  **non-finite** norm is still rejected whatever the mode: no later update can absorb a
  poisoned state.

`propose` is a pure function of the norm history, so the same norms always produce the
same step sizes. Norms that differ by rounding — another backend, thread count or build
— can produce slightly different steps, or flip an accept/reject decision, so two
adaptive runs agree to their accuracy, not bit for bit. A replayed schedule takes the
recorded steps and meets the run's accuracy target ([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).

Asking for a step below `dz_min` raises
{py:obj}`~solprop.exceptions.ConvergenceError` rather than clamping and carrying on. A
run that cannot meet its tolerance must say so. The default floor,
{py:obj}`~solprop.stepper.controller.default_dz_min`, is a billionth of the run length:
a run that wants steps that small needs over a billion of them and would never finish,
while the finest legitimate scale on a real gradient fibre sits five orders above it. The
floor applies to the controller's *proposal* only; a step that landing clips onto a save
plane may be as short as the geometry demands.

The proposal, not the executed step, is the controller's state. After a step that landing
clipped or stretched, the driver bases the next proposal on the previous proposal, and
the controller learns from the norm the proposal would have scored,
{py:obj}`~solprop.stepper.controller.PIController.equivalent_norm`, which rescales the
measured norm by the length ratio to the power $q + 1$; a rejection is scaled by the same
equivalent norm. Without this, every save plane would restart the controller from a
sliver and it would regrow at 1.1 per step: a thousandfold recovery takes over seventy
steps. Whether the step is accepted, and what the step history in the output records,
is the norm the executed step actually scored, so a short landing step that was accurate
is neither thrown away nor written up as a failure.

## Caps and landing

{py:obj}`~solprop.stepper.schedule.landing` clips a trial step. Two kinds of bound apply:

| bound | kind | effect |
|---|---|---|
| `dz_max` | cap | shortens the step |
| width of the Chebyshev interval **containing** $z$ | cap, only for a coupled flow or a reject-free controller | shortens the step |
| next hard kink | landing | the step ends exactly there |
| next save plane | landing | the step ends exactly there |
| `z_end` | landing | the run stops on it, not past it |

Since [ADR-038](../../dev/decisions.md) the interval cap is off on the diagonal-flow routes;
`SimulationSettings(interval_cap=...)` overrides the choice. Where it applies it is the
*containing* interval, never the narrowest one anywhere: on a
real gradient fibre the finest local scale is four orders below the median width, and
capping globally by it would turn a metre that is smooth almost everywhere into over a
million steps. Equally, only *genuine* kinks — where $D$ itself jumps or kinks, such as
a pressure-stage boundary or a taper corner — are landed on, never the edges the
adaptive Chebyshev split introduces to resolve smooth variation; a typical fibre has
thirty intervals and two kinks.

Landing on save planes is what replaces dense output by default: every saved field is
an exact step endpoint, so no interpolant and no extra field-sized buffers are needed,
and no interpolation error enters a stored plane.

**Choosing the saved planes.** Three settings answer three questions:

- *where*: `save_planes=[...]` or `n_save=N`, the requested planes;
- *how*: landed on (the default) or `interpolate_planes=True`, interpolated inside the
  steps that cross them;
- *what else*: `step_planes=k`, the end of every `k`-th accepted step as well.

**Interpolated planes, opt-in.** With many planes, landing caps every step at the plane
spacing. `interpolate_planes=True` (on `prop_gnlse`, `prop_capillary`, their
`Simulation.from_*` constructors and `SimulationSettings`) instead steps across the
save planes and observes each inside its accepted step through DOPRI5's fourth-order
continuous extension ([ADR-039](../../dev/decisions.md)); with an Adams `integrator` it
is the march's own polynomial instead ([low-call stepping](adams.md)). With $v_0$ the state at the
step's anchor $z_n$, $k_i$ its stage derivatives in the interaction picture there and
$v_1 = v_0 + h\sum_i b_i k_i$, the plane at $z_n + \theta h$ is

$$
v(\theta) = r_1 + \theta\bigl(r_2 + (1-\theta)(r_3 + \theta(r_4 + (1-\theta)r_5))\bigr),
$$

with $r_1 = v_0$, $r_2 = v_1 - v_0$, $r_3 = h k_1 - r_2$, $r_4 = r_2 - h k_7 - r_3$ and
$r_5 = h\sum_i d_i k_i$ (Hairer, Nørsett and Wanner II.6), carried to the plane by the
exact linear flow. It costs no source call. A bin the flow attenuates by more than an
e-fold over the step, the spectral collar's, takes the anchor's value instead: its
interaction-picture derivative grows exponentially across the step.

- **Still landed on:** the end of the run and every hard kink.
- **Refused:** another tableau, a fixed schedule, the per-step temporal collar
  (`absorber_scheme="split_step"`), a coupled flow and the `modified` noise model. A
  run stopped on an interpolated plane resumes only with `resume="interpolated"`, afresh
  from the interpolated field ([output](../results/output.md)).
- **Accuracy.** The step's norm is set by the strongest components, so a weak band at
  an interpolated plane is less controlled than at a landed one. On SEP 13's canonical
  capillary at `rtol=1e-3` with 50 planes, the worst plane's field error against the
  converged run is 2.3e-3 interpolated and 1.4e-3 landed, and the UV band's 0.16 and
  0.25: the same tolerance's accuracy, not the same digits. With 501 planes the run
  keeps its 91 natural attempts where landing takes 504.

Landing stays the default.

**Saving the steps themselves.** `step_planes=k` (on the same interfaces) also saves the
end of every `k`-th accepted step, beside the requested planes, which are landed on or
interpolated as without it. The march stops there and carries on, so the run takes exactly
the steps it would without them and every such plane is a state it passed through: the
saved positions follow the problem's own scales, densest where the field changes
fastest, at no extra step, and a run stopped on one resumes as from any landed plane.
`k = 1` saves every step; a larger `k` thins the file.

- **The file grows with the step count:** about a hundred steps on SEP 13's canonical
  capillary at `rtol=1e-3`, thousands at `rtol=1e-6` or over a long fibre, each a full
  field and its statistics.
- **Two runs do not share planes,** so comparing two runs (an `rtol` ladder, two
  methods) needs interpolation onto common positions.
- **Material processing** that reads the requested planes' positions
  (`processing.mixtures`) refuses the extra planes, and **scans** refuse `step_planes`,
  since a scan saves the same planes at every point.
- On `numba_cpu` and `native` the reused last stage of a step is kept across saved planes
  (ADR-039), so saving costs no evaluation either.

A target just beyond the capped step is *stretched* to rather than approached in two
steps: when it lies within `stretch` times the capped step, and no further than `dz_max`,
the step is lengthened to end on it, and the returned
{py:obj}`~solprop.stepper.schedule.Landing` says so in `stretched`. The default stretch of
1.1 is the controller's growth cap, which it already permits every step and which raises
the local error by at most $1.1^5$; a stretch of 1 switches this off.

```{eval-rst}
.. code-block:: pycon

    >>> import numpy as np
    >>> from solprop.stepper import StepTargets, landing
    >>> targets = StepTargets(
    ...     save_planes=np.array([0.25, 0.5, 0.75, 1.0]),
    ...     hard_kinks=np.array([0.4]),
    ...     breakpoints=np.array([0.0, 1.0]),
    ...     dz_max=0.3,
    ...     z_end=1.0,
    ... )
    >>> stop = landing(0.25, 1.0, targets)
    >>> stop.target.value, stop.z_target
    ('hard_kink', 0.4)
```

The driver snaps its position to `z_target` after such a step rather than accumulating
`z + dz`, so a plane is hit exactly however many steps have gone before.

## Achieved versus requested

`rtol` is a *local* tolerance. The global error it buys at the output is problem
dependent and misses the request by one to three orders in either direction — in both
directions, not just the pessimistic one. So every run records what it achieved.
{py:obj}`~solprop.stepper.controller.ControlRecord` carries the local side: the largest and mean
scaled norm over the accepted steps, which are the achieved local error in units of the
requested tolerance, along with the rejection rate. The global side comes from
calibration, below. Neither is a substitute for the other, and neither may be assumed.


For backend execution details and the stage-by-stage formulas, see the
[solver design](../../dev/stepper.md#one-step-pass-by-pass). The guide's
[nonlinear projection](../waveguides/modal-projection.md) and [response models](../media/index.md)
define the physical source independently of that implementation.
