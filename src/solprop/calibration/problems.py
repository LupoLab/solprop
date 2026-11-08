# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The calibration's synthetic problems (SEP 46 section 6.3).

Five problems that solprop ships, so that a calibration never needs a user's
simulation: a GNLSE with Raman response and self-steepening, a four-mode helium
capillary with plasma, a mode-averaged hydrogen capillary with Raman response, a
radial argon filament with plasma and a one-dimensional free-space line. Together they
exercise every CPU operation class: stage passes, responses with short and long
columns, modal and dense products, Raman histories, plasma columns, radial and
temporal transforms. Each is adapted from the benchmark tracking set's case of the
same physics (``benchmarks/builders.py``), with no noise so that every candidate does
the same work, and with a *scale* that sizes its grid: the time window for the
one-dimensional problems and the transverse count for the spatial ones. A run times
their trial steps (``StepDriver.timed_step``), never a propagation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solprop.driver import Simulation

__all__ = ["PROBLEMS", "build"]


def _gnlse(scale: float) -> Simulation:
    """Build Dudley, Genty and Coen's soliton fission on a `scale` x 3 ps window.

    The dispersion coefficients follow J. M. Dudley, G. Genty and S. Coen, Rev. Mod.
    Phys. 78, 1135 (2006), doi:10.1103/RevModPhys.78.1135, section III, as the
    benchmark case does.

    Parameters
    ----------
    scale : float
        The window's size relative to the standard one.

    Returns
    -------
    solprop.driver.Simulation
        The simulation.
    """
    from solprop.driver import Simulation  # noqa: PLC0415 - imports the physics

    return Simulation.from_gnlse(
        0.11,
        0.15,
        [
            0.0,
            0.0,
            -11.830e-27,
            8.1038e-41,
            -9.5205e-56,
            2.0737e-70,
            -5.3943e-85,
            1.3486e-99,
            -2.5495e-114,
            3.0524e-129,
            -1.7140e-144,
        ],
        wavelength=835e-9,
        fwhm=50e-15,
        shape="sech",
        peak_power=10e3,
        raman=True,
        shock=True,
        noise=None,
        backend="native",
        wavelength_limits=(400e-9, 2e-6),
        trange=3e-12 * scale,
    )


def _capillary(scale: float) -> Simulation:
    """Build four modes of a helium capillary with plasma, on `scale` x 200 fs.

    Parameters
    ----------
    scale : float
        The window's size relative to the standard one.

    Returns
    -------
    solprop.driver.Simulation
        The simulation.
    """
    from solprop.driver import Simulation  # noqa: PLC0415 - imports the physics

    return Simulation.from_capillary(
        125e-6,
        1.5,
        "He",
        (0.8, 0.0),
        wavelength=800e-9,
        fwhm=7.5e-15,
        energy=275e-6,
        modes=4,
        noise=None,
        backend="native",
        wavelength_limits=(90e-9, 4e-6),
        trange=200e-15 * scale,
    )


def _averaged(scale: float) -> Simulation:
    """Build a mode-averaged hydrogen capillary with Raman, on `scale` x 800 fs.

    Parameters
    ----------
    scale : float
        The window's size relative to the standard one.

    Returns
    -------
    solprop.driver.Simulation
        The simulation.
    """
    from solprop.driver import Simulation  # noqa: PLC0415 - imports the physics

    return Simulation.from_capillary(
        20e-6,
        0.4,
        "H2",
        10.0,
        wavelength=1030e-9,
        fwhm=40e-15,
        energy=3e-6,
        loss=False,
        plasma=False,
        modes=1,
        mode_average=True,
        noise=None,
        backend="native",
        wavelength_limits=(200e-9, 4e-6),
        trange=800e-15 * scale,
    )


def _spatial(kind: str, scale: float) -> Simulation:
    """Build an argon filament on a radial grid, or Kerr focusing on a line.

    Parameters
    ----------
    kind : str
        ``"radial"`` or ``"line"``.
    scale : float
        The transverse count relative to the standard one.

    Returns
    -------
    solprop.driver.Simulation
        The simulation.
    """
    import numpy as np  # noqa: PLC0415

    from solprop.constants import C  # noqa: PLC0415
    from solprop.driver import Simulation, SimulationSettings  # noqa: PLC0415
    from solprop.grid import FieldGrid, LineGrid, RadialGrid  # noqa: PLC0415
    from solprop.nonlinear.optmat_rates import OptmatRate  # noqa: PLC0415
    from solprop.nonlinear.responses import (  # noqa: PLC0415
        KerrResponse,
        PlasmaResponse,
    )
    from solprop.physdata import gamma3  # noqa: PLC0415
    from solprop.pulses import GaussianPulse, SpatialPulse  # noqa: PLC0415
    from solprop.spatial import SpatialMedium, critical_power  # noqa: PLC0415

    radial = kind == "radial"
    length = 0.01
    grid = FieldGrid.from_limits(
        length,
        800e-9,
        (350e-9, 2e-6) if radial else (500e-9, 1.6e-6),
        150e-15,
    )
    medium = SpatialMedium("Ar", 1.0)
    chi3 = gamma3("Ar", wavelength=800e-9)
    density = float(medium.density(np.array([0.0]))[0])
    carrier = 2 * math.pi * C / 800e-9
    index = float(np.real(medium.index(np.array([carrier]))[0]))
    power = (3.0 if radial else 3000.0) * critical_power(
        800e-9, chi3 * density, index=index
    )
    energy = power * 30e-15 * math.sqrt(math.pi / (4 * math.log(2)))
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), 100e-6
    )
    responses: list[KerrResponse | PlasmaResponse] = [KerrResponse(chi3, thg=False)]
    settings = SimulationSettings(
        save_planes=[length],
        dz_max=length,
        dz_init=length * 1e-4,
        auto_init=True,
        backend="native",
    )
    if radial:
        responses.append(PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)))
        settings = dataclasses.replace(settings, ionisation_field_range=(1e7, 1e11))
        return Simulation.from_radial(
            grid,
            RadialGrid(1.5e-3, max(16, 2 * round(32 * scale))),
            medium,
            responses,
            pulse,
            settings=settings,
            noise=None,
        )
    return Simulation.from_free(
        grid,
        LineGrid(1e-3, max(32, round(128 * scale))),
        medium,
        responses,
        pulse,
        settings=settings,
    )


PROBLEMS: dict[str, Callable[[float], Simulation]] = {
    "gnlse": _gnlse,
    "capillary": _capillary,
    "averaged": _averaged,
    "radial": lambda scale: _spatial("radial", scale),
    "line": lambda scale: _spatial("line", scale),
}
"""The calibration problems, each built at a scale (1 is the standard size)."""


def build(name: str, scale: float = 1.0) -> Simulation:
    """Build one calibration problem at a scale of its standard size.

    Parameters
    ----------
    name : str
        One of :data:`PROBLEMS`.
    scale : float, optional
        Its grid's size relative to the standard one: the time window of the
        one-dimensional problems, the transverse count of the spatial ones.

    Returns
    -------
    solprop.driver.Simulation
        The simulation, on the native backend, without noise.

    Raises
    ------
    KeyError
        For an unknown name.
    """
    return PROBLEMS[name](scale)
