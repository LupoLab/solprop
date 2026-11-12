# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The planar route's non-vacuity and its bare flow on ``native`` (SEP 43).

From the retired ``test_numba_planar.py``, on its cases and its ladder bound. Without
these the parity in ``test_planar_native.py`` could be a comparison of two linear runs.
"""

import dataclasses
import importlib.util
import math

import numpy as np
import pytest

from backends.planar_cases import planar_case
from bounds import gamma, step_evaluations
from sessions.runs import run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.noise import NoiseRealisation
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import Initial
from solprop.translate.routes import flux_of

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("solprop_native") is None,
        reason="solprop_native is not installed",
    ),
    pytest.mark.medium,
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

EPS = np.finfo(float).eps
PLASMA_CASES = ("field-plasma", "envelope-plasma")


def native_class():
    """Return the optional wheel's backend class."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


def operator_bound(problem, length):
    """Return ``2 eps k_max L``, the principal roots' one ulp each over ``L``."""
    separable = problem.linear.separable
    assert separable is not None
    square = separable.k_squared[separable.active]
    return 2 * EPS * math.sqrt(float(np.max(np.abs(square)))) * length


def ladder_bound(name, grid, space, problem, run_request, steps, length):
    """Return the retired ladder's bound of ``steps`` replayed steps over ``length``.

    The operator's ``2 eps k_max L``, and per evaluation ``32 eps`` per FFT level with
    ``2 n_p gamma(n_x)`` for the slab rule's two real contractions, plus a plasma scan.
    """
    depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(space.n_y))
    evaluation = 32 * EPS * depth + 2 * space.n_p * gamma(space.x.size)
    if name in PLASMA_CASES:
        evaluation += gamma(12 * grid.n_to)
    evaluations = max(1.0, step_evaluations(run_request) * steps)
    return operator_bound(problem, length) + evaluations * evaluation


def propagate(backend_class, problem, run_request, grid, length, noise=None):
    """Return the state a backend's session reaches at ``length``, its last plane."""
    assert float(run_request.stepper.save_planes[-1]) == length
    backend = backend_class(problem, run_request, grid=grid)
    try:
        start = Start.entrance(problem, noise=noise)
        return run(backend, problem, run_request, start).fields[-1]
    finally:
        backend.close()


def relative(actual, expected):
    """Return the largest departure relative to the largest reference magnitude."""
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    return float(np.max(np.abs(actual - expected))) / scale


@pytest.mark.parametrize("name", ["kerr", "raman", "field-plasma", "field-kerr"])
def test_the_responses_are_exercised(name):
    """The parity runs are nonlinear: native departs from the reference's linear run.

    The retired test removed the responses from ``response_order`` and ran the same
    compiled backend. Native does not read that order for its Kerr term (its payload
    takes ``kerr_coefficients``, and its other responses their own leaves), so the
    linear run here is the reference's, which does.
    """
    grid, space, problem, run_request = planar_case(name)
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, response_order=problem.responses.response_order[:1]
        ),
    )
    steps = run_request.stepper.schedule.steps.size  # pyright: ignore[reportOptionalMemberAccess]
    bound = ladder_bound(name, grid, space, problem, run_request, steps, grid.length)
    nonlinear = propagate(native_class(), problem, run_request, grid, grid.length)
    plain = propagate(NumpyRefBackend, linear, run_request, grid, grid.length)
    assert relative(plain, nonlinear) > 1e3 * bound


def test_a_linear_run_with_the_collar_split_off_is_the_reference_s():
    """No response and no in-stage absorber: every stage derivative is zero.

    The Kerr case with only its accumulation compiled and ``absorber_scheme=
    "split_step"``, so no term enters the stages and the linear flow alone carries the
    state (the case has no y collar); the bound is the ladder's.
    """
    grid, space, problem, run_request = planar_case("kerr")
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            response_order=problem.responses.response_order[-1:],
            # Native's payload reads the coefficients, not the order.
            kerr_coefficients=None,
            absorber_scheme="split_step",
        ),
    )
    steps = run_request.stepper.schedule.steps.size  # pyright: ignore[reportOptionalMemberAccess]
    bound = ladder_bound("kerr", grid, space, linear, run_request, steps, grid.length)
    expected = propagate(NumpyRefBackend, linear, run_request, grid, grid.length)
    actual = propagate(native_class(), linear, run_request, grid, grid.length)
    assert relative(actual, expected) < bound


def test_a_modified_noise_draw_rides_the_slab_phase():
    """The noise is carried by each column's own unitary phase, wall term included.

    A realisation the translation would refuse beside a planar grid, set on the
    contract directly: both backends add ``N0 exp(i z Im D)`` at every stage, and one
    step with it agrees within the one-step bound. The phase itself is private to each
    backend, so the retired test's separate hold of it at ``2 eps k_max z`` is reached
    through the state alone; the draw has to move that state by more than the bound.
    """
    grid, space, problem, run_request = planar_case("kerr")
    length = grid.length / 10
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, length, length),
            save_planes=np.array([length]),
        )
    )
    quiet = propagate(NumpyRefBackend, problem, run_request, grid, length)
    generator = np.random.default_rng(7)
    shape = problem.initial.field0.shape
    noise = (
        generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    ) * (1e-3 * np.max(np.abs(problem.initial.field0)))
    noise[..., ~grid.active_mask] = 0
    problem = dataclasses.replace(
        problem,
        initial=Initial(problem.initial.field0, noise, "modified"),
    )
    realisation = NoiseRealisation(noise, None, "modified")
    reference = propagate(
        NumpyRefBackend, problem, run_request, grid, length, realisation
    )
    actual = propagate(native_class(), problem, run_request, grid, length, realisation)
    bound = ladder_bound("kerr", grid, space, problem, run_request, 1, length)
    assert relative(actual, reference) <= bound
    assert relative(reference, quiet) > bound


def test_the_safety_ramp_joins_the_slab_on_either_backend():
    """A transverse rate on the slab: the same answer on both, and a different one off.

    ``numba_cpu``'s ``test_the_safety_ramp_joins_the_slab_element...`` read the element.
    It asked whether a nonzero rate reaches native's slab, which the collar's parity
    could not say on its own: ``collar-only`` has one, the case's collar with no
    response, and removing it moves the answer by more than the ladder's bound on both.
    """
    grid, space, problem, run_request = planar_case("collar-only")
    separable = problem.linear.separable
    assert float(np.max(np.abs(separable.kperp_rate))) > 0.0  # pyright: ignore[reportArgumentType, reportCallIssue, reportOptionalMemberAccess]
    bare = dataclasses.replace(
        separable,  # pyright: ignore[reportArgumentType]
        kperp_rate=np.zeros_like(separable.kperp_rate),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
    )
    without = dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=bare),
        responses=dataclasses.replace(problem.responses, separable=bare),
        route=dataclasses.replace(
            problem.route, flux=dataclasses.replace(_flux(problem), base=bare)
        ),
    )
    steps = run_request.stepper.schedule.steps.size  # pyright: ignore[reportOptionalMemberAccess]
    bound = ladder_bound(
        "collar-only", grid, space, problem, run_request, steps, grid.length
    )
    expected = propagate(NumpyRefBackend, problem, run_request, grid, grid.length)
    actual = propagate(native_class(), problem, run_request, grid, grid.length)
    assert relative(actual, expected) < bound
    moved = propagate(native_class(), without, run_request, grid, grid.length)
    assert relative(moved, actual) > bound


def _flux(problem):
    """Return a spatial problem's flux plan."""
    flux = flux_of(problem.route)
    assert flux is not None
    return flux
