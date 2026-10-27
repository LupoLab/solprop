# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Each interface's default integrator, tolerances and TDSE noise (SEP 37 phase 1)."""

import pytest

from driver.test_tdse_entry import _capillary, _sae
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.simulation import (
    CAPILLARY_RTOL,
    DEFAULT_RTOL,
    SPATIAL_ATOL,
    SPATIAL_RTOL,
    TDSE_NOISE,
    WAVEGUIDE_RTOL,
)
from solprop.exceptions import DriverError, SolpropWarning
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.output.schema import NoiseResolution
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import Multistep, PIController, RungeKutta

GNLSE = {
    "lambda0": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "pulseshape": "sech",
    "n_save": 3,
    "noise": None,
}
PULSE = {"lambda0": 800e-9, "fwhm": 30e-15, "energy": 1e-6, "noise": None}
WINDOW = {"wavelength_limits": (400e-9, 2e-6), "trange": 200e-15, "n_save": 2}


def test_the_gnlse_takes_the_runge_kutta_march_and_its_tolerance():
    simulation = Simulation.from_gnlse(0.11, 0.01, [0.0, 0.0, -1.183e-26], **GNLSE)
    assert isinstance(simulation.settings.integrator, RungeKutta)
    controller = simulation.settings.controller
    assert controller is not None
    assert controller.rtol == DEFAULT_RTOL == 1e-7
    named = Simulation.from_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], integrator=Multistep(), **GNLSE
    )
    assert isinstance(named.settings.integrator, Multistep)


def test_the_capillary_keeps_the_adams_march_and_its_tolerance():
    simulation = Simulation.from_capillary(
        100e-6, 0.01, "He", 1.0, mode_average=True, **PULSE, **WINDOW
    )
    assert isinstance(simulation.settings.integrator, Multistep)
    controller = simulation.settings.controller
    assert controller is not None
    assert controller.rtol == CAPILLARY_RTOL == 1e-6


def test_the_waveguide_takes_the_runge_kutta_march_and_its_tolerance():
    simulation = Simulation.from_waveguide(
        [MarcatiliMode(100e-6)], 0.01, "He", 1.0, **PULSE, **WINDOW
    )
    assert isinstance(simulation.settings.integrator, RungeKutta)
    controller = simulation.settings.controller
    assert controller is not None
    assert controller.rtol == WAVEGUIDE_RTOL == 1e-5


def _spatial(space, controller=None):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    settings = SimulationSettings(
        controller,
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    build = (
        Simulation.from_radial
        if isinstance(space, RadialGrid)
        else Simulation.from_free
    )
    return build(
        grid,
        space,
        SpatialMedium(),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6),
        settings=settings,
        noise=None,
    )


def test_a_radial_run_without_a_controller_takes_the_spatial_default():
    controller = _spatial(RadialGrid(6e-4, 16)).settings.controller
    assert controller is not None
    assert (controller.rtol, controller.atol) == (SPATIAL_RTOL, SPATIAL_ATOL)
    explicit = PIController(rtol=1e-7, atol=1e-12)
    assert _spatial(RadialGrid(6e-4, 16), explicit).settings.controller is explicit


def test_a_free_space_run_without_a_controller_is_refused():
    with pytest.raises(DriverError, match="no default tolerance yet"):
        _spatial(FreeGrid(3e-4, 8, 3e-4, 8))


def test_a_simulation_built_directly_needs_a_controller():
    built = Simulation.from_gnlse(0.11, 0.01, [0.0, 0.0, -1.183e-26], **GNLSE)
    settings = SimulationSettings(
        save_planes=built.settings.save_planes,
        dz_max=built.settings.dz_max,
        dz_init=built.settings.dz_init,
    )
    assert "interface default tolerances" in repr(settings)
    with pytest.raises(DriverError, match="needs a PIController"):
        Simulation(
            built.grid, built.modes, built.responses, built.initial, settings=settings
        )


def test_a_tdse_run_takes_no_noise_by_default():
    simulation = _capillary(tdse=_sae(), noise="default")
    assert simulation.noise is None
    assert simulation.noise_resolution == NoiseResolution("default", None, TDSE_NOISE)
    with pytest.raises(DriverError, match="takes no noise by default"):
        _capillary(tdse=_sae(), noise="default", seed=3)


def test_noise_beside_the_tdse_is_warned_on():
    with pytest.warns(SolpropWarning, match="TDSE"):
        simulation = _capillary(tdse=_sae(), noise="input", seed=3)
    assert simulation.noise is not None
