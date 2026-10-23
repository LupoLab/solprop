# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preionisation along z through the public interface."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from solprop.driver import Simulation, prop_capillary
from solprop.exceptions import DriverError, MaterialError, MaterialRangeWarning
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import PlasmaResponse
from solprop.output import open_file
from solprop.profiles import Constant, Linear, PiecewiseLinear
from solprop.stats import StatsSpec

LENGTH = 1e-3
RAMP = PiecewiseLinear(
    np.array([0.0, 0.3e-3, 0.6e-3, LENGTH]), np.array([0.02, 0.02, 0.2, 0.2])
)
"""Argon ionised 2 % at the entrance and 20 % beyond 0.6 mm, with two kinks."""
HELIUM = Linear(0.05, 0.01, LENGTH)
PLANES = np.array([0.15e-3, 0.3e-3, 0.45e-3, 0.8e-3, LENGTH])
NAMES = ("species_ionisation", "species_electron_density", "species_density")


@pytest.fixture(autouse=True)
def _quiet_gas_index():
    """Silence the gas indices' extrapolation notice, as the capillary tests do."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        yield


def keywords(**overrides):
    """Return a cheap two-species capillary whose pulse is far too weak to ionise."""
    return {
        "lambda0": 800e-9,
        "wavelength_limits": (200e-9, 2e-6),
        "trange": 100e-15,
        "fwhm": 10e-15,
        "energy": 1e-15,
        "n_r": 4,
        "plasma": True,
        # The rate tables start where the rate stops being exactly zero; the pulse
        # never reaches that field, so it frees no electron at all.
        "ionisation_field_range": (1e9, 1e11),
        "preionisation": {"Ar": RAMP, "He": HELIUM},
        "save_planes": PLANES,
        "stats": StatsSpec(names=NAMES),
        "rtol": 1e-6,
    } | overrides


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_prop_capillary_runs_a_path_per_species_and_the_density_follows_it(
    backend, tmp_path
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    prop_capillary(
        125e-6,
        LENGTH,
        {"Ar": 0.5, "He": 0.5},
        1.0,
        backend=backend,
        progress="none",
        output=tmp_path / "paths.h5",
        **keywords(),
    )
    with open_file(tmp_path / "paths.h5") as output:
        check_the_run(output)


def check_the_run(output):
    """Assert the saved planes, the analytic densities and the recorded paths."""
    np.testing.assert_allclose(output["z"][1:], PLANES)
    expected = np.stack([RAMP(PLANES), HELIUM(PLANES)], axis=-1)
    # A 1 fJ pulse stays below the tables' first field, where the rate is exactly
    # zero, so the final fraction is the initial one: the preionisation table
    # reproduces a linear piece to rounding, a few ulp; 1e-12 relative leaves room
    # for the per-node reduction.
    fractions = output["stats/species_ionisation"][1:]
    np.testing.assert_allclose(fractions.max(axis=-1), expected, rtol=1e-12)
    np.testing.assert_allclose(fractions.min(axis=-1), expected, rtol=1e-12)
    np.testing.assert_allclose(
        output["stats/species_electron_density"][1:],
        expected * output["stats/species_density"][1:],
        rtol=1e-12,
    )
    assert output.metadata is not None
    components = output.metadata.configuration["medium"]["components"]
    argon = next(one for one in components if one["label"] == "Ar")
    assert argon["preionisation"]["kind"] == "PiecewiseLinear"
    assert argon["preionisation"]["breakpoints_m"] == RAMP.breakpoints.tolist()


def test_one_path_for_every_species_and_a_single_gas_route():
    simulation = Simulation.from_capillary(
        125e-6, LENGTH, "Ar", 1.0, **keywords(preionisation=RAMP)
    )
    (plasma,) = [one for one in simulation.responses if isinstance(one, PlasmaResponse)]
    assert plasma.preionisation is RAMP
    problem = simulation.translate()
    np.testing.assert_array_equal(problem.linear.hard_kinks, [0.3e-3, 0.6e-3, LENGTH])
    mixed = Simulation.from_capillary(
        125e-6, LENGTH, {"Ar": 0.5, "He": 0.5}, 1.0, **keywords(preionisation=RAMP)
    )
    assert {
        one.preionisation for one in mixed.responses if isinstance(one, PlasmaResponse)
    } == {RAMP}


def test_with_mixture_carries_a_path_into_the_responses():
    simulation = Simulation.from_capillary(
        125e-6, LENGTH, {"Ar": 0.5, "He": 0.5}, 1.0, **keywords()
    )
    mode = simulation.modes[0]
    assert isinstance(mode, MarcatiliMode)
    mixture = mode.mixture
    assert mixture is not None
    changed = type(mixture)(
        tuple(
            component
            if component.label == "He"
            else type(component)(
                component.species,
                component.density,
                component.label,
                Constant(0.3),
                component.dephasing,
                component.raman,
            )
            for component in mixture.components
        ),
        mixture.temperature,
        mixture.shared,
        mixture.fractions,
    )
    updated = simulation.with_mixture(changed)
    by_label = {
        one.species: one.preionisation
        for one in updated.responses
        if isinstance(one, PlasmaResponse)
    }
    assert isinstance(by_label["Ar"], Constant)
    assert by_label["He"] is HELIUM


def test_the_interface_refuses_a_path_outside_the_unit_interval_by_name():
    with pytest.raises(
        MaterialError, match=r"preionisation must be in \[0, 1\] for Ar"
    ):
        Simulation.from_capillary(
            125e-6,
            LENGTH,
            {"Ar": 0.5, "He": 0.5},
            1.0,
            **keywords(preionisation={"Ar": Linear(0.5, 1.5, LENGTH)}),
        )
    with pytest.raises(DriverError, match="unknown preionisation component labels"):
        Simulation.from_capillary(
            125e-6, LENGTH, "Ar", 1.0, **keywords(preionisation={"Xe": RAMP})
        )


@pytest.mark.parametrize("declared", [RAMP, 0.1])
def test_a_single_gas_without_plasma_refuses_its_preionisation(declared):
    with pytest.raises(DriverError, match="declared for 'Ar', which has no plasma"):
        Simulation.from_capillary(
            125e-6,
            LENGTH,
            "Ar",
            1.0,
            **keywords(plasma=False, preionisation=declared),
        )
    # Zero declares nothing, and a run without plasma is still a run.
    assert Simulation.from_capillary(
        125e-6, LENGTH, "Ar", 1.0, **keywords(plasma=False, preionisation=0.0)
    ).responses


def test_a_mixture_refuses_a_preionisation_on_a_species_without_plasma():
    with pytest.raises(DriverError, match="declared for 'He', which has no plasma"):
        Simulation.from_capillary(
            125e-6,
            LENGTH,
            {"Ar": 0.5, "He": 0.5},
            1.0,
            **keywords(plasma={"Ar": True}, preionisation={"He": RAMP}),
        )


@pytest.mark.parametrize(
    ("plasma", "preionisation"),
    [(False, RAMP), ({"Ar": True}, {"He": 0.05})],
    ids=["plasma-off", "one-species-off"],
)
def test_an_envelope_run_refuses_a_preionisation_without_plasma(plasma, preionisation):
    options = keywords(plasma=plasma, preionisation=preionisation, envelope=True)
    options["wavelength_limits"] = (500e-9, 1.2e-6)
    del options["stats"]
    with pytest.raises(DriverError, match="which has no plasma response"):
        Simulation.from_capillary(
            125e-6, LENGTH, {"Ar": 0.5, "He": 0.5}, 1.0, **options
        )


def test_with_mixture_refuses_a_path_on_a_species_without_plasma():
    simulation = Simulation.from_capillary(
        125e-6,
        LENGTH,
        {"Ar": 0.5, "He": 0.5},
        1.0,
        **keywords(plasma={"Ar": True}, preionisation={"Ar": RAMP}),
    )
    mode = simulation.modes[0]
    assert isinstance(mode, MarcatiliMode)
    mixture = mode.mixture
    assert mixture is not None
    changed = type(mixture)(
        tuple(
            replace(component, preionisation=HELIUM)
            if component.label == "He"
            else component
            for component in mixture.components
        ),
        mixture.temperature,
        mixture.shared,
        mixture.fractions,
    )
    with pytest.raises(DriverError, match="declared for 'He', which has no plasma"):
        simulation.with_mixture(changed)
