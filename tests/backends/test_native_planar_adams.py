# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar transform on native, on both marches, against the oracle.

``test_planar_native.py`` compares its own vacuum-filled plates on Runge-Kutta; this is
every case of ``tests/backends/planar_cases.py`` (the hybrid families launched as a
Gaussian in y, the analytic-signal field family, and the collar alone): one step and the
family's fixed schedule, with the state moved far beyond the bound so that the parity is
not of two copies of the launch. These cases translate with Runge-Kutta, the modal
route's default; the Adams method is pinned to its top order so that the two runs share
every decision (``backends.matched.fixed_order``).

**The bounds, stated before measuring.** The two backends perform the same operations
in the same order everywhere but three places, each of which is bounded:

- **The operator.** Both form the column's square ``q`` in the same order and differ
  chiefly in the principal square root, each within one ulp of the exact root, so
  ``Re k_z`` differs by at most ``2 eps k_max`` and the phase over a length ``L`` by
  ``2 eps k_max L``. ``k_max`` is the largest ``sqrt |k^2 eps_co|`` on the active band,
  which bounds ``|k_z|`` since the wall term and ``k_y^2`` only reduce it. The
  exponential formulation forms each step's propagator and phi functions afresh from
  ``h D``: up to ``order + 1`` of them in the predictor and again in the corrector.
- **One evaluation.** Each transform's two libraries differ by ``32 eps`` per level
  (``ceil log2 n_to`` in time, ``ceil log2 n_y`` in y); the x synthesis and projection
  are real GEMMs of lengths ``n_p`` and ``n_x`` whose dot products each round within
  ``gamma(n)`` of ``|S||a|``, and ``|P||S|`` is at most ``n_p`` times ``PS``, so each
  adds ``2 n_p gamma(n_x)``; a plasma scan adds ``gamma(12 n_to)``.
- **A run** of ``s`` steps is the operator term plus its evaluations: a replayed
  schedule keeps the two trajectories on the same steps, so the rounding does not
  compound as an independent integration's would.
"""

import dataclasses
import math

import numpy as np
import pytest

from backends.planar_cases import CASES, planar_case
from bounds import gamma, step_evaluations
from sessions.executors import committed_steps
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule

native = pytest.importorskip("solprop_native")

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

EPS = np.finfo(float).eps
PLASMA_CASES = ("field-plasma", "envelope-plasma")
RUN_STEPS = 10
"""The longest fixed-step run here (``AGENTS.md``: about ten steps)."""


def operator_bound(problem, length):
    """Return ``2 eps k_max L``, the principal roots' one ulp each over ``L``."""
    separable = problem.linear.separable
    assert separable is not None
    square = separable.k_squared[separable.active]
    return 2 * EPS * math.sqrt(float(np.max(np.abs(square)))) * length


def evaluation_bound(name, grid, space):
    """Return one evaluation's relative bound: the transforms, the x GEMMs, a scan."""
    depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(space.n_y))
    bound = 32 * EPS * depth + 2 * space.n_p * gamma(space.x.size)
    if name in PLASMA_CASES:
        bound += gamma(12 * grid.n_to)
    return bound


def ladder_bound(name, grid, space, problem, run_request, steps, length):  # noqa: PLR0913, PLR0917 - the case, its run and the window
    """Return the bound of ``steps`` replayed steps over ``length``."""
    evaluations = max(1.0, step_evaluations(run_request) * steps)
    operator = operator_bound(problem, length)
    method = run_request.stepper.integrator
    if method is not None:
        operator *= 2 * (method.order + 1)
    return operator + evaluations * evaluation_bound(name, grid, space)


def on_march(run_request, march):
    """Return the request on Runge-Kutta, or on the fixed-order Adams method."""
    if march == "runge_kutta":
        assert run_request.stepper.integrator is None
        return run_request
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper, integrator=Multistep(variable_order=False)
        )
    )


def with_steps(run_request, steps, length):
    """Return the request on ``steps`` equal replayed steps, saved at ``length``."""
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, length, length / steps),
            save_planes=np.array([length]),
        )
    )


def run(backend_class, problem, run_request, grid, length):
    """Run the problem's launch to `length`, its last plane, in the backend's session.

    Returns the state there and the accepted steps.
    """
    assert float(run_request.stepper.save_planes[-1]) == length
    backend = backend_class(problem, run_request, grid=grid)
    try:
        landed = session_run(backend, problem, run_request)
        return landed.fields[-1], committed_steps(landed)
    finally:
        backend.close()


def relative(actual, expected):
    """Return the largest departure relative to the largest reference magnitude."""
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    return float(np.max(np.abs(actual - expected))) / scale


@pytest.fixture(scope="module")
def cases():
    """Build each case once per module; construction only."""
    built = {}

    def build(name):
        if name not in built:
            built[name] = planar_case(name)
        return built[name]

    return build


@pytest.mark.medium
@pytest.mark.parametrize("march", ["runge_kutta", "adams"])
@pytest.mark.parametrize("steps", [1, None])
@pytest.mark.parametrize("name", list(CASES))
def test_a_step_and_a_run_are_the_reference_s(
    cases, name, steps, march, record_property
):
    """One step, and a run of ten steps of the family's (two for the envelope plasma).

    The field families' whole schedule of 80 steps became ten of the same length, for
    the limit on a test's length: the bound is restated from those ten.
    """
    grid, space, problem, run_request = cases(name)
    run_request = on_march(run_request, march)
    schedule = run_request.stepper.schedule
    assert schedule is not None
    count = min(schedule.steps.size, RUN_STEPS) if steps is None else steps
    # One step is the family's own step.
    length = grid.length * count / schedule.steps.size
    run_request = with_steps(run_request, count, length)
    bound = ladder_bound(name, grid, space, problem, run_request, count, length)
    reference, summary = run(NumpyRefBackend, problem, run_request, grid, length)
    actual, ours = run(native.NativeBackend, problem, run_request, grid, length)
    assert ours == summary == count
    measured = relative(actual, reference)
    record_property("measured", measured)
    record_property("bound", bound)
    assert measured <= bound
    assert relative(problem.initial.field0, reference) > 1e3 * bound


@pytest.mark.medium
@pytest.mark.parametrize("name", ["kerr", "raman", "field-plasma", "field-kerr"])
def test_the_responses_are_exercised(cases, name):
    """The runs above are nonlinear: removing the responses moves the state."""
    grid, space, problem, run_request = cases(name)
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, response_order=problem.responses.response_order[:1]
        ),
    )
    schedule = run_request.stepper.schedule
    assert schedule is not None
    count = min(schedule.steps.size, RUN_STEPS)
    length = grid.length * count / schedule.steps.size
    bound = ladder_bound(name, grid, space, problem, run_request, count, length)
    # On the oracle, which reads ``response_order``: a run within the bound of it is
    # then the nonlinear run, and not a linear one.
    stepped = with_steps(run_request, count, length)
    nonlinear, _ = run(NumpyRefBackend, problem, stepped, grid, length)
    plain, _ = run(NumpyRefBackend, linear, stepped, grid, length)
    assert relative(plain, nonlinear) > 1e3 * bound
