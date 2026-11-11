# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Envelope plasma sources at a subnormal field, against numpy_ref (SEP 43, item 21).

The retired test handed Numba's envelope column a field of 1e-320 everywhere: every
source finite, the loss source exactly zero, no reciprocal of a subnormal field entering
either. Native's column reads a field it synthesises, so this launches the smallest
state whose synthesis is not zero, one constant spectral bin, in a guide wide enough
that its mode's weight takes the node field to about 1e-320 V/m on the reference, which
the test checks. The rate there is exactly zero.

**Weaker than the original.** The node field is what the transform leaves, not exactly
1e-320. The comparison is of the whole stage, after a forward transform and the
prefactor, not of the column's two sources, so the loss source's exact zero is read
through a whole stage that is exactly zero there. At larger fields, where that stage
underflows gradually, the bound counts the transforms' levels and their quanta.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import dataclasses
import math
import warnings

import solprop_native

import numpy as np
import numpy.typing as npt
import pytest
from session_runs import launch, loaded_state, native_trial
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.translate.problem import PropagationProblem, assemble
from solprop.translate.run_request import RunRequest

from solprop_native import NativeBackend
from bounds import fft_bound

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

assert solprop_native is not None

QUANTUM = float(np.nextafter(0.0, 1.0))
"""The smallest subnormal."""
EPS = float(np.finfo(float).eps)
NAMES = ("electron_density", "peak_ionisation_rate")
WIDE_RADIUS = 1.1e-2
"""A core radius, m, whose mode's weight makes one quantum a node field of 1e-320."""


def host_case(
    radius: float = 125e-6,
) -> tuple[PropagationProblem, RunRequest, EnvelopeGrid]:
    """Return the original's case: argon, two components, 10 % preionised."""
    grid = EnvelopeGrid.from_limits(0.001, 800e-9, (500e-9, 1.2e-6), 80e-15)
    mode = MarcatiliMode(radius, "Ar", 1.0, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-5).build(grid)
    field = pulse * np.array([1.0, 0.35j])[None, :, None] / np.sqrt(1 + 0.35**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, run_request = assemble(
            grid,
            mode,
            [EnvelopePlasmaResponse("Ar", OptmatRate("Ar", 800e-9), preionisation=0.1)],
            field,
            controller=PIController(rtol=1e-7, atol=1e-14),
            save_planes=[1e-6, 8e-6],
            dz_max=1e-6,
            dz_init=1e-6,
            ionisation_field_range=(1e9, 1e11),
        )
    return problem, run_request.for_run(stats=StatsSpec(names=NAMES)), grid


def constant_launch(
    problem: PropagationProblem, value: float
) -> tuple[PropagationProblem, npt.NDArray[np.complex128]]:
    """Return `problem` launching one bin of `value` in both components, a constant."""
    field = np.zeros(problem.shape, complex)
    field[..., 0] = value
    return (
        dataclasses.replace(
            problem, initial=dataclasses.replace(problem.initial, field0=field)
        ),
        field,
    )


def evaluate(
    implementation: type[NumpyRefBackend] | type[NativeBackend],
    problem: PropagationProblem,
    run_request: RunRequest,
    grid: EnvelopeGrid,
) -> tuple[npt.NDArray[np.complex128], dict[str, np.ndarray]]:
    """Return the first stage on the launch and the launch's statistics.

    The statistics are the launch plane's, from the backend's session; the stage is
    native's zero-length trial on its state, or the reference's evaluation.
    """
    statistics = launch(
        implementation(problem, run_request, grid=grid), problem, run_request
    )["stats"]
    backend = implementation(problem, run_request, grid=grid)
    try:
        if isinstance(backend, NativeBackend):
            # A zero-length trial exposes the first stage, N(E, z = 0).
            state = loaded_state(backend, problem.initial.field0)
            native_trial(backend, state, 0.0, 0.0)
            stage = np.asarray(state.stage_to_host(0)).reshape(problem.shape)
        else:
            backend.set_initial(problem.initial.field0, None)
            backend._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
            stage = backend._nonlinear(  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
                backend._require_state(),  # pyright: ignore[reportArgumentType, reportPrivateUsage]
                backend._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
                np.ones((1, *problem.shape), complex),
            )[0]
        return stage, statistics  # pyright: ignore[reportUnknownVariableType]
    finally:
        backend.close()


def node_magnitude(
    problem: PropagationProblem, run_request: RunRequest, grid: EnvelopeGrid
) -> float:
    """Return the largest node field the reference synthesises, V/m."""
    reference = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        state = reference._require_state()  # pyright: ignore[reportPrivateUsage]
        nodes = np.concatenate(
            list(reference._plasma_nodes(state[0]))  # pyright: ignore[reportPrivateUsage]
        )
        return float(np.max(np.abs(nodes)))
    finally:
        reference.close()


def test_a_subnormal_zero_rate_field_has_finite_sources() -> None:
    """A node field of about 1e-320: no rate, an exact zero of a source, no NaN.

    The original's bound: four roundings and two subnormal quanta for a result that
    underflows. Its value here is an exact zero on both, and its rate an exact zero.
    """
    # The node field is the state over ``n_to``, in quanta, times the mode's weight.
    base, run_request, grid = host_case(WIDE_RADIUS)
    problem, _ = constant_launch(base, grid.n_to * QUANTUM)
    field = node_magnitude(problem, run_request, grid)
    assert 0.0 < field < 2e-320
    expected, expected_stats = evaluate(NumpyRefBackend, problem, run_request, grid)
    actual, actual_stats = evaluate(NativeBackend, problem, run_request, grid)
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual - expected) <= 4 * EPS * np.abs(expected) + 2 * QUANTUM)
    np.testing.assert_array_equal(expected_stats["peak_ionisation_rate"], 0.0)
    np.testing.assert_array_equal(actual_stats["peak_ionisation_rate"], 0.0)
    # The prescribed share alone, through a few products from one density.
    np.testing.assert_allclose(
        actual_stats["electron_density"],
        expected_stats["electron_density"],
        rtol=4 * EPS,
        atol=0.0,
    )


@pytest.mark.parametrize("value", [1e-300, 1e-305, 1e-308, 1e-310, 1e-312])
def test_a_source_going_through_gradual_underflow_is_the_reference_s(
    value: float,
) -> None:
    """Below 1e-308 the state is subnormal, and the stage's quanta decide.

    Bound, stated first: a transform pair's relative ``fft_bound`` while the values are
    normal, the column's four roundings, and, for subnormal values, one quantum at each
    of the pair's levels and the two product's, carried by the stage's gain over the
    field. No reciprocal of the field may enter, which a NaN or a jump would show.
    """
    base, run_request, grid = host_case()
    problem, _ = constant_launch(base, value)
    field = node_magnitude(problem, run_request, grid)
    assert field > 0.0
    expected, _ = evaluate(NumpyRefBackend, problem, run_request, grid)
    actual, _ = evaluate(NativeBackend, problem, run_request, grid)
    assert np.all(np.isfinite(actual))
    unit_problem, _ = constant_launch(base, 1.0)
    unit, _ = evaluate(NumpyRefBackend, unit_problem, run_request, grid)
    gain = float(np.max(np.abs(unit))) / node_magnitude(unit_problem, run_request, grid)
    levels = 2 * math.ceil(math.log2(grid.n_to)) + 2
    bound = (fft_bound(grid.n_to) + 4 * EPS) * np.abs(expected) + (
        levels * QUANTUM * max(1.0, gain)
    )
    assert np.all(np.abs(actual - expected) <= bound)
