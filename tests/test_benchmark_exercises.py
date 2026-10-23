# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise batch reset and observation boundaries on short CPU problems."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest
from benchmarks.batching import construct
from benchmarks.builders import build, fixed_schedule
from benchmarks.exercises import exercise, plasma
from benchmarks.planning import Selection, matrix
from benchmarks.records import SuiteError
from benchmarks.registry import ADAPTIVE, CASES, FIXED
from benchmarks.system import Profile
from benchmarks.worker import measure, prepare, reset_columns, retained_fields

from solprop.backends import select
from solprop.stats import StatsSpec


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_batch_reuses_workspace_and_resets_each_member(backend):
    if backend == "native" and importlib.util.find_spec("solprop_native") is None:
        pytest.skip("optional native CPU backend is not installed")
    configuration = CASES["batch"].measurement(FIXED) | {"backend": backend}
    case = construct(configuration)
    try:
        ledger = case.engine.memory_ledger()
        first = case.measure(float(configuration["stop"]))
        original = case.observe()
        case.reset()
        assert case.engine.memory_ledger() == ledger
        second = case.measure(float(configuration["stop"]))
        repeated = case.observe()
        assert first["accepted"] == second["accepted"] == 10
        assert first["member_source_evaluations"] == second["member_source_evaluations"]
        assert first["source_evaluations"] == sum(first["member_source_evaluations"])
        for before, after in zip(original, repeated, strict=True):
            # Matched ten-step computations: 32 epsilon per step covers stage
            # reductions and field conversions; reset need not be bitwise.
            np.testing.assert_allclose(
                after.fields(convention="fft"),
                before.fields(convention="fft"),
                rtol=320 * np.finfo(float).eps,
                atol=0,
            )
            assert after.step_stat_names == before.step_stat_names == ("energy",)
            np.testing.assert_array_equal(
                after["step_stats/step"], before["step_stats/step"]
            )
            np.testing.assert_allclose(
                after["stats/energy"],
                before["stats/energy"],
                rtol=320 * np.finfo(float).eps,
                atol=0,
            )
        assert original[1]["stats/energy"][0, 0] > original[0]["stats/energy"][0, 0]
    finally:
        case.engine.close()


@pytest.mark.medium
def test_statistics_only_planes_avoid_field_saves_but_keep_periodic_reset_evidence():
    configuration = CASES["smoke"].measurement(ADAPTIVE[1]) | {
        "backend": "numpy_ref",
        "fields": "planes",
        "observations": 4,
    }
    simulation = build(configuration)
    simulation.settings = replace(
        simulation.settings,
        stats=StatsSpec(names=("energy",), periodic=("energy",), period=1),
    )
    problem = simulation.translate()
    engine = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    try:
        output = prepare(simulation, problem, engine, configuration)
        result = measure(engine, problem, configuration, output)
        assert result["endpoint"] == configuration["stop"]
        assert len(output["z"]) == 4 and len(output["stats/energy"]) == 4
        assert len(retained_fields(output)) == 1
        columns = reset_columns(output)
        assert len(columns["step_stats/energy"]) > 0
        assert len(columns["step_stats/step"]) == len(columns["step_stats/energy"])
        output.close()
    finally:
        engine.close()


def test_fixed_only_batch_is_visible_in_quick_and_unsupported_tracks_are_explicit():
    profile = Profile("test", 1, 1, (0,), "test")
    plan = matrix((CASES["batch"],), profile, Selection(backend="numpy_ref"))
    assert len(plan["rows"]) == 1
    assert plan["rows"][0]["configuration"]["track"] == FIXED
    assert plan["rows"][0]["status"] == "ready"
    other = matrix((CASES["batch"],), profile, Selection(tracks=ADAPTIVE))
    assert all(row["status"] == "excluded" for row in other["rows"])
    for builder in (exercise, plasma):
        with pytest.raises(SuiteError, match="not implemented"):
            builder("absent")


def test_short_plasma_schedule_stops_at_the_original_guide_exit():
    case = CASES["envelope_plasma"]
    configuration = case.measurement(FIXED)
    schedule = fixed_schedule(case, configuration)
    assert len(schedule.steps) == 4
    # Four subtraction/addition operations reconstruct the frozen pilot edges.
    np.testing.assert_allclose(
        np.cumsum(schedule.steps),
        np.array([2, 4, 6, 8]) * 1e-6,
        rtol=4 * np.finfo(float).eps,
        atol=0,
    )
    with pytest.raises(SuiteError, match="4 entrance steps"):
        fixed_schedule(case, configuration | {"schedule": [0, 8e-6]})


@pytest.mark.medium
def test_dense_observations_keep_the_integrator_trajectory_between_planes():
    configuration = CASES["dense_output"].measurement(ADAPTIVE[1]) | {
        "backend": "numpy_ref"
    }
    simulation = build(configuration)
    assert simulation.settings.interpolate_planes
    assert simulation.grid.length == 1.0
    problem = simulation.translate()
    engine = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    try:
        output = prepare(simulation, problem, engine, configuration)
        measure(engine, problem, configuration, output)
        planes = output["z"]
        assert len(planes) == 8
        # Dense observations need not be accepted step endpoints. At least one
        # plane lies inside a step, proving this exercises interpolation.
        endpoints = output["step_history/z"] + output["step_history/h"]
        assert any(
            not np.any(
                np.isclose(plane, endpoints, rtol=8 * np.finfo(float).eps, atol=0)
            )
            for plane in planes[:-1]
        )
        calls = engine.metadata()["source_evaluations"]
        first = retained_fields(output)
        one_plane = configuration | {"observations": 1}
        repeated = prepare(simulation, problem, engine, one_plane)
        measure(engine, problem, one_plane, repeated)
        assert engine.metadata()["source_evaluations"] == calls
        # Same discrete march with two observation itineraries: 32 epsilon for
        # each of the ten steps plus their amplitude/output conversions.
        np.testing.assert_allclose(
            retained_fields(repeated)[-1],
            first[-1],
            rtol=320 * np.finfo(float).eps,
            atol=0,
        )
        output.close()
        repeated.close()
    finally:
        engine.close()
