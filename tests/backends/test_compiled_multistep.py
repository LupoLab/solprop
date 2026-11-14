# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 2: the Adams march on the compiled backends, against the reference.

The march is shared (``solprop.backends.numpy_ref.multistep``); only the one source
evaluation and the propagator differ: Kokkos evaluates through its general engine's
stage pipeline. Parity is the reference's trajectory to round-off: the same steps,
orders and calls, and fields that agree to the accumulated rounding of the evaluation
orders.
"""

import dataclasses
import math

import numpy as np
import pytest

from backends.adams_source_cases import (
    ROUTES,
    SOURCE_AT,
    SOURCE_STEPS,
    matched_run,
    one_evaluation,
)
from bounds import EPS, assert_within
from sessions.runs import adams_ledger
from sessions.runs import run as session_run
from solprop.backends import available
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.boundaries import absorber
from solprop.driver.settings import SimulationSettings
from solprop.exceptions import BackendError
from solprop.noise import modified_noise
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import delta_phi

COMPILED: list[type] = []
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED.append(NativeBackend)

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27


def run(backend_class, problem, run_request, grid, start=None):
    """Run the session to the last plane; return its field, ledger and step sizes.

    The step sizes are every attempt's, rejected ones included.
    """
    backend = backend_class(problem, run_request, grid=grid)
    record: dict[str, object] = {}
    try:
        result = session_run(backend, problem, start=start, record=record)
        ledger = adams_ledger(backend, record)
    finally:
        backend.close()
    return result.fields[-1], ledger, np.asarray(result.joined[0].dz)


@pytest.mark.parametrize(
    "method",
    [
        Multistep(formulation="interaction"),
        Multistep(order=3, variable_order=False, formulation="interaction"),
        # The comparison variants' general path is the reference's alone (SEP 47).
        "interpolated",
    ],
    ids=["selected", "pec3/4", "dense"],
)
@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_follows_the_reference_trajectory(
    narrow_grid, soliton_problem, method, backend_class
):
    interpolate = method == "interpolated"
    problem, run_request = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 8, SOLITON_PERIOD / 4],
        controller=PIController(rtol=1e-6, atol=1e-18),
        integrator=Multistep(formulation="interaction") if interpolate else method,
        interpolate_planes=interpolate,
    )
    reference, reference_ledger, reference_steps = run(
        NumpyRefBackend, problem, run_request, narrow_grid
    )
    field, ledger, steps = run(backend_class, problem, run_request, narrow_grid)
    # The same decisions: every step, order and call.
    assert ledger.orders == reference_ledger.orders
    assert ledger.total == reference_ledger.total
    # The evaluation orders round differently and the controller passes that on to
    # the step sizes: measured at most 1.5e-10 relative over 47-99 steps (Kokkos's
    # PECE4).
    np.testing.assert_allclose(steps, reference_steps, rtol=1e-9)
    # Measured 8e-16 to 1.8e-15 of the peak over the quarter period.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_steps_the_modified_noise_as_the_reference(
    narrow_grid, soliton_problem, formulation, backend_class
):
    # SEP 16 phase 3: the draw enters each sample at its own position, through the
    # same stage pipeline the Runge-Kutta march uses.
    noise = modified_noise(narrow_grid, np.random.default_rng(3))
    problem, run_request = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 8],
        noise=noise,
        controller=PIController(rtol=1e-7, atol=1e-18),
        integrator=Multistep(formulation=formulation),  # pyright: ignore[reportArgumentType]
    )

    start = Start.entrance(problem, noise=noise)
    reference, reference_ledger, _ = run(
        NumpyRefBackend, problem, run_request, narrow_grid, start
    )
    start = Start.entrance(problem, noise=noise)
    field, ledger, _ = run(backend_class, problem, run_request, narrow_grid, start)
    assert ledger.total == reference_ledger.total
    # As for the noiseless trajectory (above): rounding of the evaluation orders.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-12 * scale)


def test_the_settings_admit_the_compiled_backends():
    control = PIController(rtol=1e-6, atol=1e-12)
    common = {"save_planes": [1.0], "dz_max": 0.1, "dz_init": 1e-3}
    settings = SimulationSettings(
        control,
        backend="native",
        integrator=Multistep(formulation="interaction"),
        **common,
    )
    assert settings.backend == "native"
    with pytest.raises(BackendError, match="'cuda' is not a backend"):
        SimulationSettings(
            control,
            backend="cuda",
            integrator=Multistep(formulation="interaction"),
            **common,
        )


def assert_marches_without_responses(
    backend_class, narrow_grid, linear_problem, collar
):
    """Assert a problem with no response marches as on the reference, to round-off."""
    # No nonlinear response: the source is zero, or the in-stage collar's term alone.
    extra = (
        {"absorber": absorber(narrow_grid), "absorber_scheme": "nonlinear"}
        if collar
        else {}
    )
    problem, run_request = linear_problem(
        narrow_grid,
        save_planes=[SOLITON_PERIOD / 8],
        integrator=Multistep(formulation="interaction"),
        **extra,
    )
    reference, _, _ = run(NumpyRefBackend, problem, run_request, narrow_grid)
    field, _, _ = run(backend_class, problem, run_request, narrow_grid)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
def test_native_marches_a_problem_without_responses(
    narrow_grid, linear_problem, collar
):
    native = pytest.importorskip("solprop_native")
    assert_marches_without_responses(
        native.NativeBackend, narrow_grid, linear_problem, collar
    )


SOURCE_SHARE_BOUND = 1e-10
"""The matched runs' source contributions' departure, relative to the reference's.

Measured 5.4e-12 (analytic field) and 7.5e-12 (envelope plasma) on native. The floor is
the fields' own rounding over the source's share of them (1.1e-5): an eps of the field
is 2e-11 of the source, and the linear runs alone part by 2e-17 to 5e-17 of their
field. 1e-10 is about five eps of the field, 13 times the worst measured.
"""


def assert_propagates_as_the_reference(backend_class, route):
    """Assert one propagator, ``exp(Phi(position) - Phi(anchor))``, is the reference's.

    The native stage kernels' phase sum over its ``Phi`` terms for the step's first
    stage row, both read from the native state (``adams_stage_scalars``,
    ``phase_sum_to_host``), against the reference's ``_adams_propagator``, a step from
    the anchor at the launch. An exponent rounded within a few eps of its size and an
    exponential's few eps, so 16 eps (1 + max |delta Phi|) relative; measured equal to
    the bit. The source evaluated there is the catalogue's row ``ladder adams source
    ...`` (``backends/ladder_rows.py``).
    """
    built = one_evaluation(route)
    problem, run_request = built.problem, built.run_request
    simulation = built.simulation
    assert simulation is not None
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    reference = NumpyRefBackend(problem, run_request, **context)
    try:
        reference_back = np.array(reference._adams_propagator(0.0, SOURCE_AT))
    finally:
        reference.close()
    backend = backend_class(problem, run_request, **context)
    try:
        backend._staged_rows()
        state = backend._native_state
        stages = run_request.stepper.tableau.n_stages
        rows = np.asarray(
            state.adams_stage_scalars(0.0, SOURCE_AT)["phi_delta"], dtype=np.float64
        ).reshape(stages + 1, -1)
        exponent = np.array(state.phase_sum_to_host(np.ascontiguousarray(rows[0])))
    finally:
        backend.close()
    back = np.exp(exponent).reshape(reference_back.shape)
    size = np.abs(delta_phi(problem.linear, SOURCE_AT, 0.0))
    bound = 16 * EPS * (1 + float(np.max(size)))
    assert_within(back, reference_back, bound, what="propagator")


def assert_sources_the_reference(backend_class, route):
    """Assert `backend_class`'s Adams march sources a column family as the reference.

    The Adams source through the stage pipeline's two other column families: the
    analytic signal of a field without THG, and the envelope with its plasma assembly,
    at a launch that reaches into the temporal collar. The propagator is compared as
    before SEP 47 (:func:`assert_propagates_as_the_reference`); then the march's use of
    the source: both backends replay :data:`SOURCE_STEPS` fixed steps at the method's
    top order through their sessions, the same discrete method and history
    (``backends/matched.py``), so they part by rounding alone. Each backend's source
    contribution is its run's field less its own run of the launch scaled to the linear
    limit (``faint``), where the source's share falls a millionfold; the two
    contributions are held to :data:`SOURCE_SHARE_BOUND` of the reference's. The source
    moves the field by 1.1e-5 of it over the run, seven decades above the fields'
    matched-window bound (512 eps a step); the fields themselves are the catalogue's
    row ``ladder adams matched run ...``.
    """
    assert_propagates_as_the_reference(backend_class, route)
    built = matched_run(route)
    problem, run_request = built.problem, built.run_request
    simulation = built.simulation
    assert simulation is not None
    faint = 1e-3
    linear = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial, field0=faint * problem.initial.field0
        ),
    )
    found = []
    for evaluating, launched in (
        (NumpyRefBackend, problem),
        (NumpyRefBackend, linear),
        (backend_class, problem),
        (backend_class, linear),
    ):
        backend = evaluating(
            launched,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            found.append(session_run(backend, launched).fields[-1])
        finally:
            backend.close()
    reference, unsourced, field, unsourced_here = found
    bound = SOURCE_STEPS * 512 * EPS
    share = reference - unsourced / faint
    assert np.linalg.norm(share) / np.linalg.norm(reference) > 1e6 * bound
    assert_within(
        field - unsourced_here / faint,
        share,
        SOURCE_SHARE_BOUND,
        what="the run's source contribution",
    )


@pytest.mark.parametrize(
    "route",
    [
        pytest.param(route, marks=pytest.mark.medium)
        if route == "envelope plasma"
        else route
        for route in ROUTES
    ],
)
def test_native_sources_the_analytic_field_and_the_envelope_plasma(route):
    native = pytest.importorskip("solprop_native")
    assert_sources_the_reference(native.NativeBackend, route)
