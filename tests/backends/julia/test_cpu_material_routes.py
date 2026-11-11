# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify combined scalar materials on every admitted route through actual sessions."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.material_routes import material_route, refined_request
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.tableau import TSIT5
from solprop.translate.run_request import MethodKind

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _run(case, grid, modes, *, julia, steps, method="dopri5"):
    request = refined_request(case.request, steps)
    integration = request.integration
    if method == "tsit5":
        integration = replace(integration, tableau=TSIT5)
    elif method == "adams":
        integration = replace(integration, kind=MethodKind.ADAMS, method=Multistep())
    elif method == "low_storage":
        integration = replace(
            integration, kind=MethodKind.LOW_STORAGE, method=LowStorageRungeKutta()
        )
    request = replace(request, integration=integration)
    if julia:
        owner = JuliaBackend(
            case.problem,
            request,
            grid=grid,
            modes=modes,
            options=ExecutionOptions(
                node_chunk=3,
                backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
            ),
        )
    else:
        owner = select("numpy_ref", case.problem, request, grid=grid, modes=modes)
    try:
        with owner.prepare(case.problem, request).start(
            Start.entrance(case.problem)
        ) as session:
            result = drive(Started(session), export=IDENTITY, work_limit=2)
            assert result.status.outcome is Outcome.COMPLETED
            assert result.status.attempts == steps
            fields = result.fields[-1]
            stats = next(
                event.stats
                for event in reversed(result.observations)
                if isinstance(event, ObservationReady)
            )
            return fields, stats
    finally:
        owner.close()


def _assert_reference_accuracy(case, grid, modes, *, method="dopri5"):
    # Ordinary replay (including variable-order Adams) has its own discrete history.
    # Fix a relative L2 field envelope of 1e-7 before either backend runs, and require
    # an independently refined reference to settle two orders more tightly. Positive
    # population observables use their own 1e-6 envelope with 1e-8 reference settlement;
    # none is accepted on raw rtol or on a posteriori backend discrepancies.
    actual, stats = _run(case, grid, modes, julia=True, steps=8, method=method)
    coarse, coarse_stats = _run(case, grid, modes, julia=False, steps=16)
    fine, fine_stats = _run(case, grid, modes, julia=False, steps=32)
    scale = float(np.linalg.norm(fine.ravel()))
    assert scale > 0.0 and np.all(np.isfinite(actual))
    assert np.linalg.norm((fine - coarse).ravel()) <= 1e-9 * scale
    assert np.linalg.norm((actual - fine).ravel()) <= 1e-7 * scale
    for name, expected in fine_stats.items():
        reference = np.asarray(expected)
        magnitude = float(np.max(np.abs(reference)))
        assert magnitude > 0.0, name
        assert np.all(np.isfinite(stats[name])), name
        assert np.max(np.abs(coarse_stats[name] - reference)) <= 1e-8 * magnitude, name
        assert np.max(np.abs(stats[name] - reference)) <= 1e-6 * magnitude, name
    # Summing two positive independent populations adds at most two rounded additions
    # on either side; this checks catalogue composition separately from rate accuracy.
    for values in (stats, coarse_stats, fine_stats):
        species = values["species_electron_density"]
        np.testing.assert_allclose(
            values["electron_density"],
            np.sum(species),
            rtol=4 * np.finfo(float).eps,
            atol=0.0,
        )


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize(
    "geometry", ["averaged", "multimode", "radial", "cartesian", "line"]
)
def test_two_material_populations_and_raman_components_on_each_route(kind, geometry):
    case, grid, modes = material_route(kind, geometry)
    responses = case.problem.responses
    if kind == "field":
        channels = responses.plasma_channels
    else:
        assert responses.envelope_plasma is not None
        channels = responses.envelope_plasma.channels
    assert len(channels) == 2
    assert len(case.problem.responses.raman_terms) == 2
    _assert_reference_accuracy(case, grid, modes)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["averaged", "multimode"])
def test_independent_material_gradients_and_uncoupled_tapers(kind, geometry):
    case, grid, modes = material_route(kind, geometry, gradient=True, taper=True)
    assert case.problem.linear.coupled_flow is None
    _assert_reference_accuracy(case, grid, modes)


@pytest.mark.parametrize("method", ["tsit5", "adams", "low_storage"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_material_composition_across_all_march_families(kind, method):
    case, grid, modes = material_route(kind, "cartesian")
    _assert_reference_accuracy(case, grid, modes, method=method)
