# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar transform: its rows of the case catalogue.

**What is compared.** Native executes schema 15's ``"slab"`` kind -- the y FFT around
the slab rule's synthesis and projection in x, each column's operator with its
channel's ``kappa_sq``, the y collar in both absorber schemes, every response family
and the spatial and plasma statistics -- as ``numpy_ref`` does. On native against
``numpy_ref``:

- the plates (``test_planar_native.planar_case``): vacuum-filled plates of constant
  index, three slab orders with the beam off centre in x and a node chunk of four y
  rows over eleven, so that the last chunk is partial, in eight families (the
  envelope, analytic-signal and real-field Kerr columns, the field and envelope
  plasma, Raman, and the y collar in both schemes). ``derivative``: one nonlinear
  evaluation at the launch; ``launch``: the spatial (and plasma) statistics the session
  observes at the launch; ``one_step``: one step over the whole plate; ``steps``: ten
  steps over it, with every statistic at their end;
- the hybrid cases (``planar_cases.py``): the hybrid families launched as a Gaussian
  in y on an argon- or nitrogen-filled 127 um gap, the analytic-signal field family and
  the collar alone, on Runge-Kutta and on the Adams method pinned to its top order
  (``Multistep(variable_order=False)``, so the two runs share every decision).
  ``one_step``: the family's own step; ``steps``: ten of them (two for the envelope
  plasma, whose schedule has two);
- the Kerr case with every response removed and the collar split off (no term enters
  the stages: the linear flow alone carries the state), and the collar case with its
  safety ramp, each over its own schedule (``steps``);
- an argon-filled gap through the planar driver (``Simulation.from_planar``), ten Kerr
  steps over 10 cm with every default free-space statistic (``run``).

Every field comparison is peak-normalised (``metric="peak"``): the bounds are relative
to the largest reference amplitude.

**The bounds are the old tests', fixed before native runs.** The plates take
``test_planar_native.amplitude_bound``: per stage evaluation ``32 eps`` per level of
the temporal and y transforms and per term of the slab rule's two x contractions, a
plasma scan's ``gamma(12 n_to)``, seven stages a step, and over a propagation the
operator's square root, ``8 eps beta_max L``. Their statistics take twice that for
squared fields, plus the column, node and temporal reductions,
``gamma(128 (n_columns + n_nodes) + 32 n_to)``. The hybrid cases take
:func:`ladder_bound`: the principal root's ``2 eps k_max L`` over the length (times
``2 (order + 1)`` on the Adams method, which forms each step's propagator and phi
functions afresh), plus each replayed step's evaluations' ``32 eps`` per transform
level, ``2 n_p gamma(n_x)`` for the x GEMMs and a plasma scan. The planar driver's
run takes the plates' bound over its 10 cm.

**The guard.** Each row's term is switched off on the translated problem: every
response (:func:`without_responses`) or the plasma's density
(``optmat_rows.without_plasma``), the y collar (:func:`without_collar`), the safety ramp
(:func:`without_ramp`), or the slab channels' own transverse term in the linear flow
(:func:`without_slab_term`). Each guarded rung's reference must then move by more than
``parity_ladder.DETECTION`` (a thousand) of that rung's bounds. These replace the old
modules' own non-vacuity checks: native's nonlinear run away from the reference's
linear one by a thousand bounds, the reference's run away from its launch by a
thousand bounds, and the safety ramp moving native by one bound. The statistics at the
launch read the launch itself, which no term has touched, so a ``launch`` rung is
unguarded; at the launch a collar plate's y collar reads only the beam's wings, so its
``derivative`` rung is unguarded too (:data:`COLLAR_AT_THE_LAUNCH`).

**How to read a row.** Every row is a :func:`planar_row` call, for example::

    planar_row(
        name="hybrid kerr runge-kutta",
        stepper="rk",
        response="kerr",
        features="hybrid, Gaussian in y",
        why="On runge kutta the hybrid Kerr gap is the reference's over one ...",
        build=partial(hybrid_built, "kerr", "runge_kutta"),
        switch_off=without_responses,
        rungs={
            "one_step": partial(hybrid_bound, "kerr", "runge_kutta", 1),
            "steps": partial(hybrid_bound, "kerr", "runge_kutta", None),
        },
    )

``name`` makes the test id (``planar hybrid kerr runge-kutta``); ``route``,
``stepper``, ``response`` and ``features`` are the axes the row covers. ``build``
makes the problem: here the hybrid Kerr case on ten of its own steps. ``rungs`` names
what is compared, each with its bound: here the field after the first step and after
all ten, within that many replayed steps' budget. ``switch_off`` removes the row's
term for the guard. ``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends import planar_cases
from backends import test_planar_native as plates
from backends.optmat_rows import without_plasma
from backends.test_planar_hybrid import FAMILIES
from bounds import gamma, step_evaluations
from parity_ladder import Built, Row, Statistic
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import flux_of

EPS = float(np.finfo(float).eps)
"""The machine epsilon."""

QUIET = (
    # The argon-filled cases' FFT padding reaches beyond the material's fit.
    "ignore::solprop.exceptions.MaterialRangeWarning",
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning",
)
"""The warnings the old planar modules ignored, which these rows' runs raise."""

# --- the switch-offs ------------------------------------------------------------------


def without_responses(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with every response removed: a linear problem."""
    return dataclasses.replace(
        problem, responses=planar_cases.linear_responses(problem.responses)
    )


def _with_separable(problem: PropagationProblem, **changes: Any) -> PropagationProblem:
    """Return the problem with its separable operator changed everywhere it is held.

    The linear operator, the responses' prefactors and the flux plan each carry it.
    """
    separable = problem.linear.separable
    flux = flux_of(problem.route)
    assert separable is not None and flux is not None
    changed = dataclasses.replace(separable, **changes)
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=changed),
        responses=dataclasses.replace(problem.responses, separable=changed),
        route=dataclasses.replace(
            problem.route, flux=dataclasses.replace(flux, base=changed)
        ),
    )


def without_ramp(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its safety ramp: every column's rate zeroed."""
    separable = problem.linear.separable
    assert separable is not None and separable.kperp_rate is not None
    return _with_separable(problem, kperp_rate=np.zeros_like(separable.kperp_rate))


def without_slab_term(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its slab channels' own transverse term.

    Each channel's ``kappa_sq``, ``k^2 - beta_p^2``, which every column of the hybrid
    kind subtracts from its square, zeroed: the channels propagate as plane waves in
    x.
    """
    separable = problem.linear.separable
    assert separable is not None and separable.kappa_sq is not None
    return _with_separable(problem, kappa_sq=np.zeros_like(separable.kappa_sq))


def without_collar(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its y collar: the in-stage rate zeroed everywhere."""
    transform = problem.transform
    geometry = transform.geometry
    assert geometry is not None and np.any(geometry.absorber_rate > 0)
    quiet = dataclasses.replace(
        geometry, absorber_rate=np.zeros_like(geometry.absorber_rate)
    )
    return dataclasses.replace(
        problem,
        route=dataclasses.replace(
            problem.route,
            transform=dataclasses.replace(transform, geometry=quiet),
        ),
    )


def has_a_ramp(problem: PropagationProblem) -> bool:
    """Return whether the problem's safety ramp has a nonzero rate."""
    separable = problem.linear.separable
    return (
        separable is not None
        and separable.kperp_rate is not None
        and float(np.max(np.abs(separable.kperp_rate))) > 0.0
    )


SWITCH_OFFS: dict[str, Callable[[PropagationProblem], PropagationProblem]] = {
    "responses": without_responses,
    "plasma": without_plasma,
    "collar": without_collar,
}
"""Each guard's switch-off, by the term it removes."""

LAUNCH_IS_UNTOUCHED = (
    "The launch rung compares statistics of the launch itself, before any step: the "
    "spatial ones read a field no term has touched yet, and the row's field rungs "
    "carry its guard."
)
"""Why a row's ``launch`` rung is unguarded."""

# --- the plates (test_planar_native.py) -----------------------------------------------


@functools.cache
def _plate(family: str) -> tuple[Any, Any, PropagationProblem, Any]:
    """Return one plate family's ``(grid, space, problem, run_request)``, built once."""
    return plates.planar_case(family)


def plate_names(family: str) -> tuple[str, ...]:
    """Return the statistics a plate family's rows read: spatial, and plasma's too."""
    plasma = plates.PLASMA_NAMES if family.endswith("plasma") else ()
    return plates.SPATIAL_NAMES + plasma


def plate_built(family: str, steps: int) -> Built:
    """Return a plate family on `steps` uniform steps, its statistics requested."""
    grid, space, problem, run_request = _plate(family)
    request = plates.with_steps(run_request, steps).for_run(
        stats=StatsSpec(names=plate_names(family))
    )
    return Built(problem, request, grid, options=plates.chunked(space))


def plate_bound(family: str, steps: int, built: Built) -> float:
    """Bound a plate's field over `steps` steps (zero: one evaluation)."""
    grid, space, _, _ = _plate(family)
    return plates.amplitude_bound(grid, space, family, steps)


def plate_statistics_bound(family: str, steps: int, built: Built) -> float:
    """Bound a plate's statistics: twice the field's, plus the reductions."""
    grid, space, _, _ = _plate(family)
    bound = 2 * plates.amplitude_bound(grid, space, family, steps)
    bound += gamma(128 * (space.n_columns + space.n_nodes) + 32 * grid.n_to)
    return bound


def held_at_the_launch(built: Built, name: str, field: float) -> Statistic:
    """Hold a statistic at the launch to the launch rung's own bound, positive."""
    return Statistic(field, positive=True)


def held_after_ten_steps(
    family: str, built: Built, name: str, field: float
) -> Statistic:
    """Hold a statistic after ten steps to the plates' statistics bound, positive."""
    return Statistic(plate_statistics_bound(family, 10, built), positive=True)


COLLAR_AT_THE_LAUNCH = {
    "collar_in_stage": "At the launch the beam lies inside the aperture and the y "
    "collar's in-stage term reads only its wings: switching the collar off moves the "
    "derivative by 0.12 of its bounds (measured). The collar is guarded over the step, "
    "which it moves by 1.0e9 bounds.",
    "collar_split_step": "Split off the stages, the y collar is applied between steps "
    "and is not in one evaluation: switching it off moves the derivative by 0.08 of "
    "its bounds (measured, rounding). The collar is guarded over the step, which it "
    "moves by 9.8e8 bounds.",
}
"""Why a collar plate's ``derivative`` rung is unguarded."""

PLATES = {
    # family: (features, response axis, the claim's subject, term switched off)
    "envelope_kerr": ("envelope grid", "kerr", "the envelope Kerr column", "responses"),
    "field_analytic_kerr": (
        "field grid, analytic signal",
        "kerr",
        "the analytic-signal Kerr column",
        "responses",
    ),
    "field_real_kerr": (
        "field grid, real field, third harmonic",
        "kerr",
        "the real-field Kerr column with its harmonic",
        "responses",
    ),
    "field_plasma": (
        "field grid, held current",
        "plasma",
        "the field plasma column",
        "plasma",
    ),
    "envelope_raman": ("envelope grid", "raman", "the Raman response", "responses"),
    "envelope_plasma": (
        "envelope grid",
        "envelope plasma",
        "the envelope plasma",
        "plasma",
    ),
    "collar_in_stage": (
        "envelope grid, y collar in stage",
        "kerr",
        "the y collar in the stages",
        "collar",
    ),
    "collar_split_step": (
        "field grid, y collar split step",
        "kerr",
        "the y collar split off the stages",
        "collar",
    ),
}
"""Each plate family's axes, its claim's subject and the term its guard removes."""


SLOW_BY_COST = {
    "envelope_plasma": "slow by cost (#792): 90 s serially on the Mac, 156 s under "
    "-n 4 (the two old tests it merges took 81 s and 68 s), most of it the "
    "reference's envelope-plasma cycle moments; a parity check, not a witness",
}
"""The plates whose ten-step row is slow by cost, and why."""


def plate_rows() -> list[Row]:
    """Return the plates' rows: one step with the launch, and ten steps."""
    rows = []
    for family, (features, response, subject, term) in PLATES.items():
        rows += [
            planar_row(
                name=f"plate {family} one step",
                stepper="rk",
                response=response,
                features=f"plates, {features}, launch statistics",
                why=f"On the plates {subject} is the reference's in one evaluation "
                "at the launch and over one step of the whole plate, and the "
                "statistics observed at the launch are the reference's.",
                build=partial(plate_built, family, 1),
                switch_off=SWITCH_OFFS[term],
                rungs={
                    "derivative": partial(plate_bound, family, 0),
                    "launch": partial(plate_statistics_bound, family, 0),
                    "one_step": partial(plate_bound, family, 1),
                },
                every_statistic=held_at_the_launch,
                unguarded={"launch": LAUNCH_IS_UNTOUCHED}
                | (
                    {"derivative": COLLAR_AT_THE_LAUNCH[family]}
                    if family in COLLAR_AT_THE_LAUNCH
                    else {}
                ),
            ),
            planar_row(
                name=f"plate {family} ten steps",
                stepper="rk",
                response=response,
                features=f"plates, {features}, every statistic",
                why=f"On the plates {subject} is the reference's over ten steps, and "
                "so is every spatial and plasma statistic at their end.",
                build=partial(plate_built, family, 10),
                switch_off=SWITCH_OFFS[term],
                rungs={"steps": partial(plate_bound, family, 10)},
                every_statistic=partial(held_after_ten_steps, family),
                slow=SLOW_BY_COST.get(family, ""),
            ),
        ]
    return rows


# --- the hybrid cases -----------------------------------------------------------------

PLASMA_CASES = ("field-plasma", "envelope-plasma")
"""The hybrid cases whose evaluation adds a plasma scan."""

RUN_STEPS = 10
"""The longest fixed-step run here (``AGENTS.md``: about ten steps)."""

MARCHES = {"runge_kutta": "rk", "adams": "adams"}
"""Each march a hybrid case runs on, and its stepper axis."""


@functools.cache
def _hybrid(name: str) -> tuple[Any, Any, PropagationProblem, Any]:
    """Return one hybrid case's ``(grid, space, problem, run_request)``, built once."""
    return planar_cases.planar_case(name)


def operator_bound(problem: PropagationProblem, length: float) -> float:
    """Return ``2 eps k_max L``, the principal roots' one ulp each over ``L``."""
    separable = problem.linear.separable
    assert separable is not None
    square = separable.k_squared[separable.active]
    return 2 * EPS * math.sqrt(float(np.max(np.abs(square)))) * length


def evaluation_bound(name: str, grid: Any, space: Any) -> float:
    """Return one evaluation's relative bound: the transforms, the x GEMMs, a scan."""
    depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(space.n_y))
    bound = 32 * EPS * depth + 2 * space.n_p * gamma(space.x.size)
    if name in PLASMA_CASES:
        bound += gamma(12 * grid.n_to)
    return bound


def ladder_bound(
    name: str,
    grid: Any,
    space: Any,
    problem: PropagationProblem,
    run_request: Any,
    steps: int,
    length: float,
) -> float:
    """Return the bound of ``steps`` replayed steps over ``length``.

    The two backends perform the same operations in the same order everywhere but
    three places. The operator: both form the column's square in the same order and
    differ chiefly in the principal square root, each within one ulp, so the phase
    over ``L`` differs by ``2 eps k_max L`` (``k_max`` the largest
    ``sqrt |k^2 eps_co|`` on the active band); the exponential formulation forms each
    step's propagator and phi functions afresh, up to ``order + 1`` of them in the
    predictor and again in the corrector. One evaluation (:func:`evaluation_bound`).
    And a replayed schedule keeps the two trajectories on the same steps, so the
    rounding of a run's evaluations adds rather than compounds.
    """
    evaluations = max(1.0, step_evaluations(run_request) * steps)
    operator = operator_bound(problem, length)
    method = run_request.stepper.integrator
    if method is not None:
        operator *= 2 * (method.order + 1)
    return operator + evaluations * evaluation_bound(name, grid, space)


def on_march(run_request: Any, march: str) -> Any:
    """Return the request on Runge-Kutta, or on the fixed-order Adams method."""
    if march == "runge_kutta":
        assert run_request.stepper.integrator is None
        return run_request
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper, integrator=Multistep(variable_order=False)
        )
    )


def with_steps(run_request: Any, steps: int, length: float) -> Any:
    """Return the request on ``steps`` equal replayed steps, saved at ``length``."""
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, length, length / steps),
            save_planes=np.array([length]),
        )
    )


def hybrid_count(name: str, steps: int | None) -> int:
    """Return a hybrid run's steps: `steps`, or the family's own up to ten.

    Read from the case's family (``test_planar_hybrid.FAMILIES``), so that counting
    builds nothing.
    """
    family = planar_cases.CASES[name][0]
    own = FAMILIES[family][8]
    return min(own, RUN_STEPS) if steps is None else steps


def _hybrid_request(name: str, march: str, count: int) -> tuple[Any, float]:
    """Return a hybrid case's request on `count` of its own steps, and their length."""
    grid, _, _, run_request = _hybrid(name)
    run_request = on_march(run_request, march)
    schedule = run_request.stepper.schedule
    assert schedule is not None
    length = grid.length * count / schedule.steps.size
    return with_steps(run_request, count, length), length


def hybrid_built(name: str, march: str, steps: int | None) -> Built:
    """Return a hybrid case on `steps` of its own steps (``None``: up to ten)."""
    grid, _, problem, _ = _hybrid(name)
    request, _ = _hybrid_request(name, march, hybrid_count(name, steps))
    return Built(problem, request, grid)


def hybrid_bound(name: str, march: str, steps: int | None, built: Built) -> float:
    """Bound a hybrid case over `steps` of its own steps (``None``: up to ten)."""
    grid, space, problem, _ = _hybrid(name)
    count = hybrid_count(name, steps)
    request, length = _hybrid_request(name, march, count)
    return ladder_bound(name, grid, space, problem, request, count, length)


HYBRIDS = {
    # case: (response axis, features, the claim's subject, term switched off)
    "kerr": ("kerr", "hybrid, Gaussian in y", "the hybrid Kerr gap", "responses"),
    "kerr-collar": (
        "kerr",
        "hybrid, y collar in stage",
        "the hybrid Kerr gap with its y collar",
        "collar",
    ),
    "kerr-split": (
        "kerr",
        "hybrid, y collar split step",
        "the hybrid Kerr gap with its collar split off",
        "collar",
    ),
    "raman": ("raman", "hybrid, nitrogen", "the hybrid Raman gap", "responses"),
    "field-plasma": (
        "plasma",
        "hybrid, field grid",
        "the hybrid field plasma",
        "responses",
    ),
    "field-kerr": (
        "kerr",
        "hybrid, field grid, analytic signal",
        "the analytic-signal Kerr column on the hybrid gap",
        "responses",
    ),
    "envelope-plasma": (
        "envelope plasma",
        "hybrid, envelope grid",
        "the hybrid envelope plasma",
        "responses",
    ),
    "collar-only": (
        "none",
        "hybrid, y collar alone",
        "the y collar with no response",
        "collar",
    ),
}
"""Each hybrid case's axes, its claim's subject and the term its guard removes."""


def hybrid_rows() -> list[Row]:
    """Return the hybrid cases' rows: one step and their run, on either march."""
    rows = []
    for name, (response, features, subject, term) in HYBRIDS.items():
        count = hybrid_count(name, None)
        for march, stepper in MARCHES.items():
            words = march.replace("_", "-")
            rows.append(
                planar_row(
                    name=f"hybrid {name} {words}",
                    stepper=stepper,
                    response=response,
                    features=features,
                    why=f"On {march.replace('_', ' ')} {subject} is the reference's "
                    f"over one of its steps and over {count} of them.",
                    build=partial(hybrid_built, name, march, None),
                    switch_off=SWITCH_OFFS[term],
                    rungs={
                        "one_step": partial(hybrid_bound, name, march, 1),
                        "steps": partial(hybrid_bound, name, march, None),
                    },
                )
            )
    return rows


# --- the linear run and the safety ramp -----------------------------------------------


def linear_built() -> Built:
    """Return the hybrid Kerr case with every response removed and its collar split off.

    The case has no y collar; with the Kerr term's accumulation alone compiled and
    ``absorber_scheme="split_step"`` no term enters the stages, and the linear flow
    alone carries the state over the case's own ten steps.
    """
    grid, _, problem, run_request = _hybrid("kerr")
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, kerr_coefficients=None, absorber_scheme="split_step"
        ),
    )
    return Built(linear, run_request, grid)


def own_schedule_bound(name: str, built: Built) -> float:
    """Bound a hybrid case over its own schedule, on the built problem."""
    grid, space, _, run_request = _hybrid(name)
    schedule = run_request.stepper.schedule
    assert schedule is not None
    return ladder_bound(
        name, grid, space, built.problem, run_request, schedule.steps.size, grid.length
    )


OWN_STEPS = 10
"""A hybrid case's own schedule, whole: its ten steps sum to one ulp short of the plane
it saves, so the march ends at the schedule's last edge."""


def ramp_built() -> Built:
    """Return the collar case, its safety ramp nonzero, over its own ten steps."""
    grid, _, problem, run_request = _hybrid("collar-only")
    return Built(problem, run_request, grid)


# --- the planar driver ----------------------------------------------------------------

DRIVER_LENGTH = 0.1
"""The planar driver's run, m: ten steps of a centimetre."""


@functools.cache
def _driver_grids() -> tuple[Any, Any]:
    """Return the planar driver run's grid and planar grid."""
    from legacy_geometry import LEGACY_GRID  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(
        DRIVER_LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15, **LEGACY_GRID
    )
    modes = [SlabMode(plates.HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3, 5)]
    return grid, PlanarGrid(modes, 1.5e-3, 32)


def driver_built() -> Built:
    """Return ten Kerr steps of an argon-filled gap, every default statistic saved."""
    from legacy_geometry import bare_absorber  # noqa: PLC0415

    grid, space = _driver_grids()
    launch = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * plates.HALF_GAP,
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[DRIVER_LENGTH],
        dz_max=0.01,
        dz_init=0.01,
        schedule=FixedSchedule(np.full(10, 0.01)),
        backend="numpy_ref",
        stats=default_spec("free"),
        absorber=bare_absorber(grid),
        # The coarse case launches 4e-4 into the safety ramp's band, which would
        # refuse it (SEP 40 section 5.3); the subject is backend parity.
        spatial_absorber=False,
    )
    simulation = Simulation.from_planar(
        grid,
        space,
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        launch,
        settings=settings,
    )
    return Built(
        simulation.translate(), simulation._run_request(), grid, simulation=simulation
    )


def driver_bound(built: Built) -> float:
    """Bound the run: the plates' ten-step amplitude bound over its 10 cm."""
    grid, space = _driver_grids()
    return plates.amplitude_bound(
        grid, space, "envelope_kerr", 10, length=DRIVER_LENGTH
    )


def driver_statistic(built: Built, name: str, field: float) -> Statistic:
    """Hold a statistic: twice the field's bound, plus the reductions."""
    grid, space = _driver_grids()
    return Statistic(
        2 * field + gamma(128 * (space.n_columns + space.n_nodes) + 32 * grid.n_to)
    )


# --- the rows -------------------------------------------------------------------------


def planar_row(
    *,
    name: str,
    route: str = "planar",
    stepper: str,
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a planar row, medium when it marches, unless it is slow by cost.

    Its id is ``planar {name}``; `fields` are the rest of :class:`parity_ladder.Row`.
    Every row quiets the old modules' warnings (:data:`QUIET`) and compares
    peak-normalised.
    """
    return Row(
        id=f"planar {name}",
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=build,
        medium=not fields.get("slow")
        and bool({"one_step", "steps", "run"} & set(fields["rungs"])),
        metric="peak",
        quiet=QUIET,
        **fields,
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every planar row."""
    return (
        *plate_rows(),
        *hybrid_rows(),
        planar_row(
            name="hybrid kerr linear, collar split off",
            stepper="rk",
            response="none",
            features="hybrid, no response, collar split step",
            why="With every response removed and the collar split off, no term "
            "enters the stages and native's linear flow alone carries the Kerr gap "
            "over its ten steps as the reference's does.",
            build=linear_built,
            switch_off=without_slab_term,
            rungs={"steps": partial(own_schedule_bound, "kerr")},
            window=OWN_STEPS,
        ),
        planar_row(
            name="hybrid collar-only safety ramp",
            stepper="rk",
            response="none",
            features="hybrid, y collar, safety ramp",
            why="The safety ramp's transverse rate reaches native's slab columns: "
            "the collar case is the reference's over its ten steps.",
            build=ramp_built,
            switch_off=without_ramp,
            rungs={"steps": partial(own_schedule_bound, "collar-only")},
            window=OWN_STEPS,
            expect={"problem": has_a_ramp},
        ),
        planar_row(
            name="driver argon",
            stepper="rk",
            response="kerr",
            features="envelope grid, argon, driver, every free-space statistic",
            why="Ten Kerr steps of an argon-filled gap through the planar driver are "
            "the reference's, every saved plane and every default free-space "
            "statistic.",
            bound_note="The plates' ten-step amplitude bound over the run's 10 cm "
            "(the operator's term dominates it); the statistics twice that plus "
            "the reductions.",
            build=driver_built,
            switch_off=without_responses,
            rungs={"run": driver_bound},
            every_statistic=driver_statistic,
        ),
    )
