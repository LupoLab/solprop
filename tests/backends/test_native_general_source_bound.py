# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route on native against the reference, by the tight bound.

``tests/general_parity.py``'s bound, written from operation counts and the projection's
measured conditioning before anything is measured, on its six cases' scaled launches
with random phases in every channel: one stage's source and ten fixed Runge-Kutta
steps. The source is read through a zero-length trial's stage 0, which is the source
times ``crop_scale``: one more product, inside the count's ``+ 2``.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from general_parity import (
    CASES,
    LENGTH,
    conditioning,
    nonlinear_scale,
    problem_for,
    reference_source,
    scaled_case,
    source_bound,
    ten_step_bound,
)
from sessions.runs import native_trial
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.exceptions import GeneralModalBasisWarning

native = pytest.importorskip("solprop_native")

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.GeneralModalBasisWarning"),
    # The launches are noise-free in windows kept tight for speed: the boundary monitor
    # warns, and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

HEAVY = {"grin", "raman"}
"""Cases whose guide or window makes a run seconds rather than milliseconds."""


def marks(name):
    """Mark the heavier parity cases medium."""
    return pytest.param(name, marks=pytest.mark.medium) if name in HEAVY else name


@pytest.fixture(scope="module")
def cases():
    """Build each parity case once for the module, on demand."""
    built = {}

    def get(name):
        if name not in built:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", GeneralModalBasisWarning)
                built[name] = scaled_case(name)
        return built[name]

    return get


def run(backend_class, problem, run_request, grid):
    """Return the final state of ten fixed steps on one backend, in its session."""
    backend = backend_class(problem, run_request, grid=grid)
    try:
        landed = session_run(backend, problem, run_request)
    finally:
        backend.close()
    assert landed.observations[-1].z == pytest.approx(LENGTH, rel=1e-12)
    return landed.fields[-1]


def departure(got, expected):
    """Return the relative L2 departure of two states."""
    return float(np.linalg.norm(got - expected) / np.linalg.norm(expected))


@pytest.mark.parametrize("name", [marks(one) for one in CASES])
def test_one_stage_of_the_source_is_the_reference_within_the_bound(
    cases, name, record_property
):
    """Native's source on the launch state, against the reference's."""
    grid, guide, route, state = cases(name)
    problem, run_request = problem_for(grid, guide, route, state)
    maps = _general_maps(problem)
    assert maps is not None
    bound = source_bound(maps, grid.n_to, conditioning(maps, state, grid))
    expected = reference_source(maps, state, grid) * grid.crop_scale
    backend = native.NativeBackend(problem, run_request, grid=grid)
    try:
        native_trial(backend, 0.0, 0.0, field=problem.initial.field0)
        got = np.asarray(backend._native_state.stage_to_host(0)).reshape(expected.shape)
    finally:
        backend.close()
    measured = departure(got, expected)
    record_property("source_bound", bound)
    record_property("source_departure", measured)
    assert measured <= bound


@pytest.mark.parametrize("name", [marks(one) for one in CASES])
def test_ten_steps_are_the_reference_within_the_bound(cases, name, record_property):
    """Ten fixed steps on each backend, from the same launch.

    ``ten_step_bound`` carries the source's bound through the step at the launch's
    nonlinear scale and adds both backends' rounding of the linear half; the schedule
    is fixed, so the two step through the same positions. The nonlinear half is really
    exercised: the launch's own linear flow alone is far from the answer.
    """
    grid, guide, route, state = cases(name)
    problem, run_request = problem_for(grid, guide, route, state)
    maps = _general_maps(problem)
    assert maps is not None
    bound = ten_step_bound(
        nonlinear_scale(maps, state, grid),
        source_bound(maps, grid.n_to, conditioning(maps, state, grid)),
    )
    expected = run(NumpyRefBackend, problem, run_request, grid)
    got = run(native.NativeBackend, problem, run_request, grid)
    measured = departure(got, expected)
    record_property("ten_step_bound", bound)
    record_property("ten_step_departure", measured)
    assert measured <= bound
    linear = dataclasses.replace(route, responses=None)
    alone = run(NumpyRefBackend, *problem_for(grid, guide, linear, state), grid)
    assert departure(alone, expected) > 1e3 * bound
