# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A small run on each route, for what must hold of a record whichever built it.

Run as a script, this prints each route's configuration hash as JSON, so that a test can
compare the hashes of processes with different environments.
"""

import json
import sys
import warnings
from collections.abc import Callable

from modal_provider_cases import general_grid, general_guide
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController


def gnlse() -> Simulation:
    """Return a soliton fibre, with its input noise drawn from a seed."""
    return Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        n_save=2,
        backend="numpy_ref",
        noise="input",
        seed=7,
    )


def capillary() -> Simulation:
    """Return an argon capillary with plasma: optmat, a quadrature and a band rule."""
    grid = FieldGrid.from_limits(4e-4, 800e-9, (300e-9, 3e-6), 100e-15)
    return Simulation.from_capillary(
        50e-6,
        4e-4,
        "Ar",
        1.0,
        wavelength=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0),
        fwhm=30e-15,
        energy=1e-6,
        n_r=8,
        n_save=3,
        backend="numpy_ref",
        plasma=True,
        noise=None,
        rtol=1e-6,
    )


def _settings(grid: EnvelopeGrid) -> SimulationSettings:
    """Return the settings of a small spatial run on `grid`."""
    return SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[5e-4, 1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
        absorber=absorber(grid),
        backend="numpy_ref",
    )


def _spatial_grid() -> EnvelopeGrid:
    """Return the temporal grid of a small spatial run."""
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)


def _pulse() -> SpatialPulse:
    """Return the launch of a small spatial run."""
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 60e-6
    )


def radial() -> Simulation:
    """Return a small radial run."""
    grid = _spatial_grid()
    return Simulation.from_radial(
        grid,
        RadialGrid(250e-6, 16, collar_fraction=0.2),
        SpatialMedium(),
        [KerrResponse(1e-22, thg=False)],
        _pulse(),
        settings=_settings(grid),
        noise=None,
    )


def cartesian() -> Simulation:
    """Return a small Cartesian run."""
    grid = _spatial_grid()
    return Simulation.from_free(
        grid,
        FreeGrid(3e-4, 8, 4e-4, 6, collar_fraction=0.2),
        SpatialMedium(),
        [KerrResponse(1e-22, thg=False)],
        _pulse(),
        settings=_settings(grid),
        noise=None,
    )


def general() -> Simulation:
    """Return a general modal run on a provider's guide."""
    grid = general_grid()
    return Simulation.from_waveguide(
        (),
        1e-4,
        provider=general_guide(grid=grid),
        grid=grid,
        fwhm=30e-15,
        energy=1e-9,
        wavelength=800e-9,
        n_save=2,
        backend="numpy_ref",
        noise=None,
    )


ROUTES: dict[str, Callable[[], Simulation]] = {
    "gnlse": gnlse,
    "capillary": capillary,
    "radial": radial,
    "cartesian": cartesian,
    "general": general,
}
"""Each route's small run, by name."""


if __name__ == "__main__":
    # The runs are built and not stepped, and their warnings are not what is asked.
    warnings.simplefilter("ignore")
    hashes = {
        name: build().metadata().configuration_hash() for name, build in ROUTES.items()
    }
    sys.stdout.write(json.dumps(hashes))
