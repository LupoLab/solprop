# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check extracted workload definitions without running benchmark campaigns."""

import numpy as np
import pytest
from benchmarks.builders import build, capillary, gnlse
from benchmarks.planning import Selection, matrix
from benchmarks.records import SuiteError
from benchmarks.registry import ADAPTIVE, CASES, FIXED, select
from benchmarks.schedules import ENTRANCE_SCHEDULES
from benchmarks.system import Profile

from solprop.nonlinear.responses import PlasmaResponse
from solprop.stepper import Multistep, RungeKutta


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in ENTRANCE_SCHEDULES
        if CASES[name].builder not in ("advanced", "exercise")
    ],
)
def test_windows_preserve_full_guides_and_frozen_fixed_steps(name):
    case = CASES[name]
    adaptive = case.measurement(ADAPTIVE[1])
    assert 0 == adaptive["start"] < adaptive["stop"] < case.length
    configuration = case.measurement(FIXED) | {"backend": "numpy_ref"}
    simulation = build(configuration)
    assert simulation.grid.length == case.length
    assert isinstance(simulation.settings.integrator, RungeKutta)
    assert simulation.settings.fields == "none"
    schedule = simulation.settings.schedule
    assert schedule is not None
    expected = (
        np.linspace(0.0, 0.0012, 11)
        if name == "ctc_reduced"
        else np.array(ENTRANCE_SCHEDULES[name])
    )
    # Schedule construction subtracts stored edges, and cumulative addition can
    # round once per edge. This tests their frozen itinerary, not a solver bound.
    np.testing.assert_allclose(
        np.cumsum(schedule.steps[:10]),
        expected[1:],
        rtol=10 * np.finfo(float).eps,
        atol=0,
    )
    assert simulation.settings.save_planes[-1] == case.length
    assert simulation.grid.n_t == case.dimensions[-2]
    assert simulation.grid.n_to == case.dimensions[-1]


@pytest.mark.parametrize(
    ("resolved", "averaged"),
    [
        ("rdw_one_mode", "he_rdw_averaged"),
        ("argon_blueshift", "argon_blueshift_averaged"),
        ("h2_raman_6ps", "h2_raman_6ps_averaged"),
    ],
)
def test_averaging_is_a_distinct_route_with_the_original_grid(resolved, averaged):
    first, second = capillary(resolved), capillary(averaged)
    assert first.settings.quadrature is not None
    assert second.settings.quadrature is None
    assert second.settings.components is None
    # A route change must not change the launch, guide or temporal axes.
    assert first.grid.length == second.grid.length
    np.testing.assert_array_equal(first.grid.omega, second.grid.omega)
    assert first.settings.stats != second.settings.stats
    if resolved == "h2_raman_6ps":
        assert not any(
            isinstance(response, PlasmaResponse) for response in first.responses
        )


def test_new_sets_are_ready_for_timing_without_invented_references():
    cases = select(sets=("modal-capillary", "mode-averaged"))
    plan = matrix(
        cases, Profile("test", 1, 1, (0,), "topology"), Selection(backend="numba_cpu")
    )
    assert len(cases) == 11
    assert len(plan["rows"]) == 22
    assert all(row["status"] == "ready" for row in plan["rows"])
    assert all(row["reference"] == "not-qualified" for row in plan["rows"])
    for name in ENTRANCE_SCHEDULES:
        if CASES[name].builder in ("advanced", "exercise"):
            continue
        simulation = build(
            CASES[name].measurement(ADAPTIVE[0]) | {"backend": "numpy_ref"}
        )
        assert isinstance(simulation.settings.integrator, Multistep)
        assert simulation.settings.fields == "all"


def test_named_builders_refuse_unknown_workloads():
    with pytest.raises(SuiteError, match="not implemented"):
        capillary("absent")
    with pytest.raises(SuiteError, match="not implemented"):
        gnlse("absent")
