# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A modified noise draw on the planar route's slab phase, on ``native`` (SEP 43).

From the retired ``test_numba_planar.py``, on its cases and its ladder bound
(``planar_rows.ladder_bound``). Its other claims -- that the parity runs are nonlinear,
that a linear run with the collar split off is the reference's, and that the safety
ramp reaches the slab -- are the case catalogue's planar rows
(``tests/backends/planar_rows.py``) and their guards.
"""

import dataclasses
import importlib.util

import numpy as np
import pytest

from backends.planar_cases import planar_case
from backends.planar_rows import ladder_bound
from sessions.runs import run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.noise import NoiseRealisation
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import Initial

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


def native_class():
    """Return the optional wheel's backend class."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


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
