# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tiny runs with a requested collar outside the aperture, on every spatial route.

A linear (or padded Kerr) beam on the radial, Cartesian, line and planar routes with
SEP 40's collar in a room as wide as the aperture, run on native by default
(:func:`collar_case`). ``test_native_edge_collar.py`` holds native's edge reading on
them; the case catalogue's geometry rows (``geometry_rows.py``) compare native's
matched runs with the reference's.
"""

from typing import Any, cast

from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

LENGTH = 2e-3
"""The tiny cases' length, m."""

ROUTES = ("radial", "cartesian", "line", "planar")
"""Every route a requested collar runs on."""

STEPS = 8
"""Fixed steps over the tiny cases: short enough for the collar's stability limit."""


INWARD: dict[str, Any] = {
    "waist": 60e-6,
    "propz": -2.4e-2,
    "boundary_check": BoundaryCheck(budget=1.0, stop_factor=1.0, transverse_budget=1.0),
}
"""A converging beam reaching the collar, whose current is inward and above rounding.

A narrow beam's current is 1e-30 and a comparison of it passes whatever it reads.
"""


def collar_settings(*, planar: bool = False, **overrides: Any) -> SimulationSettings:
    """Return settings for a linear run with a requested collar on native."""
    keywords: dict[str, Any] = {
        "controller": PIController(rtol=1e-7, atol=0.0),
        "save_planes": [LENGTH / 2, LENGTH],
        "dz_max": LENGTH / 8,
        "dz_init": LENGTH * 1e-3,
        "backend": "native",
        "spatial_absorber": SpatialAbsorber(kappa=2.5e5 if planar else 8e4),
    }
    return SimulationSettings(**(keywords | overrides))


def collar_case(
    route: str,
    *,
    kerr: bool = False,
    waist: float = 40e-6,
    propz: float = 0.0,
    noise: str | None = None,
    **overrides: Any,
) -> Simulation:
    """Return a run with a requested collar outside its aperture.

    Parameters
    ----------
    route : str
        The route.
    kerr : bool, optional
        Whether the medium has a Kerr response, on a field grid, whose time is padded.
    waist : float, optional
        The beam's waist on a 300 um aperture, m; the planar case's scales with its own.
    propz : float, optional
        The launch's signed distance from the waist, m: negative is converging. A
        planar launch is at the guide's entrance, so it ignores this.
    noise : str or None, optional
        The noise model; the planar route takes none.
    **overrides : Any
        Settings to change.

    Returns
    -------
    solprop.driver.Simulation
        The run.
    """
    grid = (
        FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
        if kerr
        else EnvelopeGrid.from_limits(LENGTH, 800e-9, (700e-9, 900e-9), 200e-15)
    )
    temporal = GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=1e-9)
    settings = collar_settings(planar=route == "planar", **overrides)
    responses: list[Any] = (
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)] if kerr else []
    )
    medium = SpatialMedium("Ar", 1.0) if kerr else SpatialMedium(1.0)
    if route == "planar":
        modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
        return Simulation.from_planar(
            EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15),
            PlanarGrid(modes, 1e-4, 40, collar_fraction=1.0),
            responses,
            PlanarPulse(
                GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
                waist / 2.0,
                waist_x=0.735 * 63.5e-6,
            ),
            settings=settings,
        )
    pulse = SpatialPulse(temporal, waist, propz=propz)
    if route == "radial":
        return Simulation.from_free(
            grid,
            RadialGrid(300e-6, 48, collar_fraction=1.0),
            medium,
            responses,
            pulse,
            settings=settings,
            noise=cast("Any", noise),
        )
    space = (
        LineGrid(300e-6, 96, collar_fraction=1.0)
        if route == "line"
        else FreeGrid(300e-6, 64, 360e-6, 80, collar_fraction=1.0)
    )
    return Simulation.from_free(
        grid,
        space,
        medium,
        responses,
        pulse,
        settings=settings,
        noise=cast("Any", noise),
    )
