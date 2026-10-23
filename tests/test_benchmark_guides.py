# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify bounded advanced definitions without running a benchmark campaign."""

import numpy as np
import pytest
from benchmarks.builders import build, fixed_schedule
from benchmarks.guides import ScaledPermittivity, advanced, coupled
from benchmarks.records import SuiteError
from benchmarks.registry import ADAPTIVE, CASES, FIXED, select
from benchmarks.worker import Diagnostics

from solprop.modes import IndexMaterial
from solprop.stepper import Multistep, RungeKutta


@pytest.mark.medium
@pytest.mark.parametrize("name", ["scalar_wall", "vector_wall", "taper", "taper_wall"])
def test_wall_cases_keep_full_profiles_and_the_original_fixed_itinerary(name):
    case = CASES[name]
    diagnostics = Diagnostics()
    diagnostics.case, diagnostics.phase = name, "construction"
    with diagnostics.capture():
        simulation = build(case.measurement(FIXED) | {"backend": "numpy_ref"})
        problem = simulation.translate()
    assert simulation.grid.length == 2e-4
    assert problem.shape[1] == (4 if name == "vector_wall" else 3)
    assert (simulation.wall is None) == (name == "taper")
    assert simulation.taper_coupling
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert len(schedule.steps) == 10
    assert diagnostics.records() and all(
        row["expected"] for row in diagnostics.records()
    )
    if name.startswith("taper"):
        # The fixed schedule must land on both original profile corners even
        # though the adaptive diagnostic ends during the taper.
        edges = schedule.edges
        for corner in (6e-5, 1.4e-4):
            assert np.min(abs(edges - corner)) <= 10 * np.finfo(float).eps * corner


@pytest.mark.medium
@pytest.mark.parametrize("track", [*ADAPTIVE, FIXED])
def test_linear_grin_launch_is_configurable_before_translation(track):
    case = CASES["grin_linear"]
    simulation = build(case.measurement(track) | {"backend": "numpy_ref"})
    assert simulation.grid.length == 1e-3
    waveguide = simulation.configuration["waveguide"]
    assert isinstance(waveguide, dict)
    assert waveguide["length"] == case.length
    assert isinstance(
        simulation.settings.integrator,
        Multistep if track == ADAPTIVE[0] else RungeKutta,
    )
    centre = int(np.argmin(abs(simulation.grid.omega - simulation.grid.omega0)))
    launch = simulation.initial
    assert isinstance(launch, np.ndarray)
    assert np.count_nonzero(launch) == 2
    np.testing.assert_array_equal(launch[:, 0, centre], np.ones(2))
    simulation.translate()


def test_advanced_selection_keeps_seeds_windows_and_unexecuted_profile_corners():
    cases = select(sets=("advanced",))
    assert len(cases) == 8
    assert all(case.status == "implemented" for case in cases)
    assert all(0 < case.stop <= case.length for case in cases)
    assert CASES["grin_nonlinear"].measurement(ADAPTIVE[0])["noise_seed"] == 41
    case = CASES["moving_solid_guide"]
    schedule = fixed_schedule(case, case.measurement(FIXED))
    # Schedule arithmetic may round once per edge; its unexecuted tail still
    # has to land on the original taper/waist corner for translation to admit it.
    assert (
        np.min(abs(schedule.edges - 5e-3))
        <= len(schedule.steps) * np.finfo(float).eps * case.length
    )
    assert schedule.edges[10] < 5e-3
    for builder in (advanced, coupled):
        with pytest.raises(SuiteError, match="not implemented"):
            builder("absent")


def test_scaled_cladding_preserves_the_material_identity_and_domain():
    law = IndexMaterial(1.45)
    scaled = ScaledPermittivity(law, 0.981)
    omega = np.array([1e15, 2e15])
    # One multiplication by the declared scale; no fitted reference constants.
    np.testing.assert_allclose(
        scaled.permittivity(omega),
        law.permittivity(omega) * 0.981,
        rtol=np.finfo(float).eps,
        atol=0,
    )
    assert scaled.omega_domain == law.omega_domain
    assert scaled.identity == {"law": dict(law.identity), "factor": 0.981}
