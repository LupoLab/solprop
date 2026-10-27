# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the advanced guide cases without running them."""

import warnings

import numpy as np
import pytest
from benchmarks.builders import build
from benchmarks.guides import ScaledPermittivity, advanced, coupled
from benchmarks.records import SuiteError
from benchmarks.registry import CASES, select

from solprop.exceptions import GrazingIncidenceWarning, SolpropWarning
from solprop.modes import IndexMaterial
from solprop.stepper import Multistep


def configuration(name, stepper="default"):
    return CASES[name].configuration(stepper, 1, "numpy_ref")


@pytest.mark.medium
@pytest.mark.parametrize("name", ["scalar_wall", "vector_wall", "taper", "taper_wall"])
def test_wall_cases_keep_full_profiles(name):
    # The tutorials' short guides under-absorb walk-off, and the walls are near
    # the paraxial limit: the warnings SEP 30 reviewed for these cases.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = build(configuration(name))
        problem = simulation.translate()
    assert {type(item.message) for item in caught} == (
        {SolpropWarning}
        if name == "taper"
        else {SolpropWarning, GrazingIncidenceWarning}
    )
    assert simulation.grid.length == 2e-4
    assert problem.shape[1] == (4 if name == "vector_wall" else 3)
    assert (simulation.wall is None) == (name == "taper")
    assert simulation.taper_coupling


@pytest.mark.medium
@pytest.mark.parametrize("stepper", ["default", "multistep"])
def test_linear_grin_launch_is_configurable_before_translation(stepper):
    case = CASES["grin_linear"]
    simulation = build(configuration("grin_linear", stepper))
    assert simulation.grid.length == 1e-3
    waveguide = simulation.configuration["waveguide"]
    assert isinstance(waveguide, dict)
    assert waveguide["length"] == case.length
    assert isinstance(simulation.settings.integrator, Multistep) == (
        stepper == "multistep"
    )
    centre = int(np.argmin(abs(simulation.grid.omega - simulation.grid.omega0)))
    launch = simulation.initial
    assert isinstance(launch, np.ndarray)
    assert np.count_nonzero(launch) == 2
    np.testing.assert_array_equal(launch[:, 0, centre], np.ones(2))
    simulation.translate()


def test_advanced_selection_and_unknown_guides():
    cases = select(sets=("advanced",))
    assert len(cases) == 8
    assert CASES["grin_nonlinear"].length == 0.02
    with pytest.raises(SuiteError, match="not implemented"):
        advanced("absent", 1.0)
    with pytest.raises(SuiteError, match="not implemented"):
        coupled("absent")


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


@pytest.mark.slow
def test_the_nonlinear_grin_guide_takes_its_planes_and_stepper_at_construction():
    # Its launch is coupled at construction, which translates the run, so the
    # builder passes the planes and integrator to the constructor.
    case = CASES["grin_nonlinear"]
    simulation = build(configuration("grin_nonlinear", "multistep"))
    problem = simulation.translate()
    np.testing.assert_allclose(
        problem.stepper.save_planes, np.linspace(0, case.length, case.planes + 1)[1:]
    )
    assert isinstance(problem.stepper.integrator, Multistep)
    assert simulation.settings.backend == "numpy_ref"
