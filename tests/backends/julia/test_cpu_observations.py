# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify complete Julia observation delivery on short binary64 sessions."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import adams_case, cpu_case, low_storage_case, material_case
from backends.julia.observation_bounds import scalar_envelope_bounds
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import ObservationReady, Outcome, join_attempts
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.profiles import Constant
from solprop.stats import StatsSpec
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _owner(case, grid, mode, *, julia, profiles=None):
    if julia:
        setup_id = os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]
        return JuliaBackend(
            case.problem,
            case.request,
            grid=grid,
            modes=(mode,),
            profiles=profiles,
            options=ExecutionOptions(backend_options={"setup_id": setup_id}),
        )
    return select(
        "numpy_ref",
        case.problem,
        case.request,
        grid=grid,
        modes=(mode,),
        profiles=profiles,
    )


def _statistics_case(case, names, *, periodic=(), period=1, windows=()):
    spec = StatsSpec(
        names=names,
        periodic=periodic,
        period=period,
        energy_windows=np.asarray(windows, dtype=float).reshape(-1, 2),
    )
    return Translation(
        case.problem,
        replace(
            case.request, observation=replace(case.request.observation, stats=spec)
        ),
    )


def _run(case, *, julia, work_limit):
    periodic_fields = {}

    def retain_periodic(session, batch):
        if not julia or work_limit != 1:
            return
        for event in batch.events:
            rows = getattr(event, "step_stats", None)
            if rows is not None and rows.z.size:
                ticket = session.pending_ticket()
                assert ticket is not None and rows.z.tolist() == [ticket.z]
                periodic_fields[ticket.z] = session.export(ticket, IDENTITY)

    with (
        _owner(case, window(), fibre().mode, julia=julia)
        .prepare(case.problem, case.request)
        .start(Start.entrance(case.problem)) as session
    ):
        result = drive(
            Started(session),
            export=IDENTITY,
            work_limit=work_limit,
            deadline=60.0,
            between=retain_periodic,
        )
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        return result, periodic_fields


@pytest.mark.parametrize("factory", [cpu_case, adams_case, low_storage_case])
@pytest.mark.parametrize("work_limit", [1, 256])
def test_periodic_cadence_and_plane_shapes_across_integrators(factory, work_limit):
    names = (
        "omega0",
        "energy",
        "energy_window",
        "peak_power",
        "peak_power_window",
        "peak_power_total",
        "fwhm_t",
        "fwhm_t_outer",
        "quadrature_error",
    )
    periodic = ("peak_power_window", "energy", "quadrature_error")
    case = _statistics_case(
        factory(),
        names,
        periodic=periodic,
        period=3,
        windows=((-np.inf, window().omega0), (window().omega0, np.inf)),
    )
    actual, periodic_fields = _run(case, julia=True, work_limit=work_limit)
    reference, _ = _run(case, julia=False, work_limit=work_limit)
    actual_planes = [
        event
        for batch in actual.batches
        for event in batch.events
        if isinstance(event, ObservationReady)
    ]
    reference_planes = [
        event
        for batch in reference.batches
        for event in batch.events
        if isinstance(event, ObservationReady)
    ]
    assert len(actual_planes) == len(reference_planes) == 3
    for got, expected in zip(actual_planes, reference_planes, strict=True):
        assert got.z == expected.z
        assert tuple(got.stats) == names
        for name in names:
            value, oracle = got.stats[name], expected.stats[name]
            assert value.shape == oracle.shape
            assert not value.flags.writeable
            if name == "quadrature_error":
                assert np.isnan(value) and np.isnan(oracle)

    def joined(result):
        return join_attempts(
            [
                (getattr(event, "history", None), getattr(event, "step_stats", None))
                for batch in result.batches
                for event in batch.events
            ]
        )

    history, rows = joined(actual)
    _, expected = joined(reference)
    assert history.n_steps == 8
    assert set(rows.values) == set(periodic)
    assert rows.step.tolist() == expected.step.tolist() == [2, 5]
    np.testing.assert_array_equal(rows.z, expected.z)
    assert np.all(np.isnan(rows.values["quadrature_error"]))
    # Numerical comparisons use the identical Julia field, independently of its
    # already-qualified propagation. The separate helper admits the transform and
    # width conditioning before inspecting either backend's statistic outputs.
    assert len(periodic_fields) == (2 if work_limit == 1 else 0)
    oracle = select(
        "numpy_ref", case.problem, case.request, grid=window(), modes=(fibre().mode,)
    )
    spec = case.request.observation.stats
    try:
        for event, field in zip(actual_planes, actual.fields, strict=True):
            bounds = scalar_envelope_bounds(case.problem, window(), spec, field)
            oracle.set_initial(field, None, z=event.z, resume=True)
            expected_values = oracle.stats(spec)
            for name, bound in bounds.items():
                assert np.all(
                    np.abs(event.stats[name] - expected_values[name]) <= bound
                )
        for index, z in enumerate(rows.z):
            if work_limit != 1:
                break
            field = periodic_fields[z]
            bounds = scalar_envelope_bounds(case.problem, window(), spec, field)
            oracle.set_initial(field, None, z=float(z), resume=True)
            expected_values = oracle.stats(spec)
            for name in set(periodic) - {"quadrature_error"}:
                assert np.all(
                    np.abs(rows.values[name][index] - expected_values[name])
                    <= bounds[name]
                )
    finally:
        oracle.close()


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_material_catalogue_launch_transport_and_population_invariants(kind):
    original, grid, mode = material_case(kind, "plasma")
    names = (
        "omega0",
        "energy",
        "energy_window",
        "peak_power",
        "peak_power_window",
        "peak_power_total",
        "fwhm_t",
        "fwhm_t_outer",
        "density",
        "pressure",
        "core_radius",
        "species_ionisation",
        "species_electron_density",
        "electron_density",
        "peak_ionisation_rate",
        "quadrature_error",
    )
    case = _statistics_case(
        original,
        names,
        windows=((-np.inf, grid.omega0), (grid.omega0, np.inf)),
    )
    profiles = {
        "density": Constant(1e24),
        "pressure": Constant(1.7),
        "core_radius": Constant(30e-6),
    }
    observations = []
    for julia in (True, False):
        owner = _owner(case, grid, mode, julia=julia, profiles=profiles)
        with owner.prepare(case.problem, case.request).start(
            Start.entrance(case.problem)
        ) as session:
            batch = session.advance()
            (event,) = batch.events
            assert isinstance(event, ObservationReady)
            observations.append(event.stats)
            if julia:
                tables = session.execution_record()["statistics_tables"]
                assert isinstance(tables, dict) and set(tables) == set(profiles)
                assert any("observation" in name for name in owner.memory_ledger())
            session.acknowledge(batch.receipt)
    # This witness qualifies catalogue transport and physical invariants. The
    # independent Julia population/cycle tests carry their analytic numerical
    # bounds; a generic relative tolerance cannot bound rate-spline sensitivity.
    for name in names:
        actual, reference = observations[0][name], observations[1][name]
        assert actual.shape == reference.shape
        assert not actual.flags.writeable
        if name == "quadrature_error":
            assert np.isnan(actual) and np.isnan(reference)
        else:
            assert np.all(np.isfinite(actual))
    for values in observations:
        for name, profile in profiles.items():
            # A constant profile is copied unchanged, with its own catalogue unit.
            assert float(values[name]) == float(profile(np.asarray(0.0)))
        fractions = values["species_ionisation"]
        assert np.all((fractions >= 0.02) & (fractions <= 1.0))
        assert np.all(values["species_electron_density"] >= 0.0)
        assert np.all(values["peak_ionisation_rate"] > 0.0)
        # One species contributes one multiply and one reduction. Bound these
        # positive operations rather than a rate evaluation or cumulative scan.
        np.testing.assert_allclose(
            values["species_electron_density"],
            fractions.reshape(-1) * 1e24,
            rtol=4 * np.finfo(float).eps,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            values["electron_density"], values["species_electron_density"].sum()
        )
