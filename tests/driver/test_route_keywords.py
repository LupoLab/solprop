# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The keyword routes' gaps, filled (SEP 41, S14).

``interpolate_planes`` and ``step_planes`` reach the waveguide route, both its
gas-filled and its general half; ``polarisation_taper`` reaches the capillary and the
waveguide, whose grids all carry a spectral collar; and the GNLSE's ``step_planes``
defaults to ``None``, which behaves as its ``0`` did. Nothing is renamed and no default
changes behaviour.
"""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop.driver import (
    Simulation,
    prop_capillary,
    prop_gnlse,
    prop_waveguide,
)
from solprop.modes import MarcatiliMode

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

WINDOW = {
    "wavelength": 800e-9,
    "wavelength_limits": (450e-9, 1200e-9),
    "trange": 100e-15,
    "fwhm": 30e-15,
    "energy": 1e-8,
    "envelope": True,
    "plasma": False,
    "raman": False,
    "n_save": 2,
    "backend": "numpy_ref",
    "noise": None,
}
PULSE = {"wavelength": 835e-9, "fwhm": 20e-15, "energy": 1e-12, "shape": "gauss"}


def capillary(**overrides):
    return Simulation.from_capillary(125e-6, 1e-3, "Ar", 1.0, **WINDOW | overrides)


def waveguide(**overrides):
    return Simulation.from_waveguide(
        [MarcatiliMode(125e-6, "Ar", 1.0)], 1e-3, "Ar", 1.0, **WINDOW | overrides
    )


@pytest.fixture(scope="module")
def grid():
    return general_grid()


@pytest.fixture(scope="module")
def guide(grid):
    return general_guide(grid=grid)


def general(guide, grid, **overrides):
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        noise=None,
        save_planes=[GENERAL_Z_END / 10],
        backend="numpy_ref",
        dz_max=GENERAL_Z_END / 40,
        **PULSE | overrides,
    )


# ------------------------------------------------------------- the signatures


@pytest.mark.parametrize(
    ("interface", "constructor", "names"),
    [
        (
            prop_waveguide,
            Simulation.from_waveguide,
            ("launch", "interpolate_planes", "step_planes", "polarisation_taper"),
        ),
        (prop_capillary, Simulation.from_capillary, ("launch", "polarisation_taper")),
        (prop_gnlse, Simulation.from_gnlse, ("step_planes",)),
    ],
)
def test_each_interface_takes_the_keywords_with_its_constructor_s_defaults(
    interface, constructor, names
):
    run = inspect.signature(interface).parameters
    build = inspect.signature(constructor).parameters
    for name in names:
        assert run[name].default == build[name].default, name
    defaults = {"launch": None, "step_planes": None, "polarisation_taper": False}
    for name in names:
        assert build[name].default == defaults.get(name, False), name


# ------------------------------------------------------------------ the planes


def test_the_gas_filled_waveguide_takes_the_plane_keywords():
    default = waveguide()
    assert default.settings.step_planes == 0
    assert default.settings.interpolate_planes is False
    chosen = waveguide(interpolate_planes=True, step_planes=2)
    assert chosen.settings.interpolate_planes is True
    assert chosen.settings.step_planes == 2


def test_the_general_route_takes_the_plane_keywords_and_saves_its_steps(guide, grid):
    assert general(guide, grid).settings.step_planes == 0
    assert general(guide, grid, interpolate_planes=True).settings.interpolate_planes
    simulation = general(guide, grid, step_planes=1)
    output = simulation.run(progress="none")
    # The entrance and the end of every accepted step, the last of which lands on
    # the requested plane: the step is capped at a quarter of its distance.
    assert output.n_saved >= 5


def test_the_gnlse_takes_step_planes_none_as_zero():
    keywords = {
        "wavelength": 835e-9,
        "wavelength_limits": (500e-9, 2e-6),
        "trange": 1e-12,
        "width": 100e-15,
        "peak_power": 9.8,
        "shape": "sech",
        "n_save": 3,
        "noise": None,
    }
    default = Simulation.from_gnlse(0.11, 0.01, [0.0, 0.0, -1.183e-26], **keywords)
    zero = Simulation.from_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], step_planes=0, **keywords
    )
    assert default.settings.step_planes == 0
    assert (
        default.metadata().configuration_hash() == zero.metadata().configuration_hash()
    )


# -------------------------------------------------------------- the collar taper


@pytest.mark.parametrize("route", ["capillary", "waveguide", "general"])
def test_polarisation_taper_reaches_the_collar_of_every_modal_route(route, guide, grid):
    def build(**keywords):
        if route == "general":
            return general(guide, grid, **keywords)
        return (capillary if route == "capillary" else waveguide)(**keywords)

    plain = build()
    assert plain.settings.polarisation_taper is False
    assert np.all(plain.translate().grids.omega_window == 1.0)
    tapered = build(polarisation_taper=True)
    assert tapered.settings.polarisation_taper is True
    # The window the kernel applies is the grid's own collar, element for element.
    assert_array_equal(
        tapered.translate().grids.omega_window, tapered.grid.omega_window
    )
    assert np.any(tapered.grid.omega_window < 1.0)
