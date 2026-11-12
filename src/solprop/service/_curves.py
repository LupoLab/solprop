# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The dispersion curves the design sheets draw: their window, axis and readings.

The capillary, fibre and waveguide sheets (:mod:`solprop.service.sheet`,
:mod:`solprop.service.waveguide`) draw a guide's dispersion over the run's own window
and read the zero and the dispersive wave from it.
"""

import math

import numpy as np

from solprop.constants import C
from solprop.grid import EnvelopeGrid
from solprop.recipe import Recipe
from solprop.tools import lambda_rdw_all

__all__ = [
    "CURVE_POINTS",
    "curve_range",
    "reach_of",
    "shortest_wave",
    "wave_window",
    "zero_near",
]

CURVE_POINTS = 240
"""The wavelengths of a sheet's dispersion curve."""

_NEAR_PUMP = 0.35
"""How far from the pump, as a share of the frequency, a curve's axis is judged."""

_NEAR_REACH = 3.0
"""How far past the pump's neighbourhood a curve's axis may reach, as a multiple."""

_CURVE_WINDOW = (0.15, 2.0)
"""The curve's wavelengths with no window in the recipe, as fractions of the pump's."""


def curve_range(recipe: Recipe, pump: float) -> tuple[float, float]:
    """Return the wavelengths a dispersion curve covers: the run's own window.

    - ``pump``: the pump's wavelength, m.
    - Returns: the two ends, m.
    """
    keywords = recipe.keywords()
    limits = keywords.get("wavelength_limits")
    if isinstance(limits, tuple | list) and len(limits) == 2:
        return float(limits[0]), float(limits[1])  # pyright: ignore[reportArgumentType]
    grid = keywords.get("grid")
    if isinstance(grid, EnvelopeGrid):
        return 2.0 * math.pi * C / grid.omega_max, 2.0 * math.pi * C / grid.omega_min
    return pump * _CURVE_WINDOW[0], pump * _CURVE_WINDOW[1]


def reach_of(values: np.ndarray, wavelength: np.ndarray, pump: float) -> float:
    """Return how far a dispersion curve's axis should reach from zero.

    The curve runs off near a resonance, and a fibre's resonance dwarfs the pump's
    neighbourhood, so the axis follows what the eye reads there.

    - ``wavelength``: its wavelengths, m, ascending.
    - ``pump``: the pump's wavelength, m.
    """
    reach = float(np.nanmax(np.abs(values[wavelength > 0.25 * wavelength[-1]])))
    omega, centre = 1.0 / wavelength, 1.0 / pump
    near = np.abs(omega - centre) < _NEAR_PUMP * omega
    around = float(np.nanmax(np.abs(values[near]))) if near.any() else reach
    return max(min(reach, _NEAR_REACH * around), 1.0) * 1.05


def zero_near(wavelength: np.ndarray, beta2: np.ndarray, pump: float) -> float | None:
    """Return the zero of a dispersion curve nearest a pump, read from the curve.

    - ``wavelength``: the curve's wavelengths, m, ascending.
    - ``pump``: the pump's wavelength, m.
    - Returns: the wavelength, m; ``None`` where the curve does not cross zero.
    """
    crossings = np.flatnonzero(np.sign(beta2[:-1]) * np.sign(beta2[1:]) < 0)
    zeros = [
        float(
            wavelength[i]
            + (wavelength[i + 1] - wavelength[i]) * beta2[i] / (beta2[i] - beta2[i + 1])
        )
        for i in crossings
    ]
    return min(zeros, key=lambda at: abs(at - pump), default=None)


def wave_window(recipe: Recipe, pump: float) -> tuple[float, float] | None:
    """Return where to look for a dispersive wave in an antiresonant fibre.

    The tool's own range reaches where this fibre's fits go non-finite (#538), so the
    search keeps to the run's window; a capillary keeps the tool's.

    - ``pump``: the pump's wavelength, m.
    - Returns: the range, m; ``None`` for the tool's own.
    """
    if recipe.keywords().get("lattice") is None:
        return None
    return curve_range(recipe, pump)[0], 0.9 * pump


def shortest_wave(
    mode: object,
    wavelength: float,
    *,
    beta2: float,
    peak_power: float,
    gamma: float,
    fwhm: float,
    shape: str,
    limits: tuple[float, float] | None = None,
) -> float | None:
    """Return the shortest wavelength phase-matched to the launch soliton.

    - ``wavelength``: the pump's wavelength, m.
    - ``beta2``: the dispersion at the pump, s^2/m.
    - ``peak_power``: the peak power, W.
    - ``gamma``: the nonlinear coefficient, 1/(W m).
    - ``fwhm``: the duration, s.
    - ``limits``: the range to search, m; ``None`` takes the tool's.
    - Returns: the wavelength, m; ``None`` when none is, as a pump in normal dispersion
      launches no soliton.
    """
    if beta2 >= 0:
        return None
    roots = (
        lambda_rdw_all(
            mode,  # pyright: ignore[reportArgumentType]
            wavelength,
            peak_power=peak_power,
            gamma=gamma,
            fwhm=fwhm,
            shape=shape,  # pyright: ignore[reportArgumentType]
        )
        if limits is None
        else lambda_rdw_all(
            mode,  # pyright: ignore[reportArgumentType]
            wavelength,
            peak_power=peak_power,
            gamma=gamma,
            fwhm=fwhm,
            shape=shape,  # pyright: ignore[reportArgumentType]
            wavelength_limits=limits,
        )
    )
    return float(roots[0]) if len(roots) else None
