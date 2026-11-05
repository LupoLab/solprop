# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a capillary's sheet shows as you type (design 6.2), in the kernel.

Three requests, each its own latest-wins key so a slow one never holds up a quick one:
the numbers, including any pin solved (:func:`capillary_sheet`); the dispersion curve
and the dispersive wave, from the recipe those numbers resolved to
(:func:`capillary_curve`); and the output's size (:func:`output_size`). None of them
runs anything: the numbers call ``tools.capillary_params`` on the recipe's keywords,
which is why they meet the sheet's targets where a whole summary does not.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from solprop.constants import C
from solprop.modes import base
from solprop.physdata.gases import Gas
from solprop.recipe import Recipe
from solprop.recipe.metadata import parameters
from solprop.tools import SolitonParams, capillary_params, lambda_rdw_all

__all__ = [
    "SHEET_HANDLERS",
    "capillary_curve",
    "capillary_numbers",
    "capillary_sheet",
    "design_numbers",
    "gases",
    "output_size",
]

Payload = Mapping[str, object]

_SHARED = ("shape", "temperature", "model", "loss", "cladding", "materials")
"""The capillary's keywords that ``tools.capillary_params`` also takes, by name."""

CURVE_POINTS = 240
"""The wavelengths of a sheet's dispersion curve."""

_CURVE_WINDOW = (0.15, 2.0)
"""The curve's wavelengths with no window in the recipe, as fractions of the pump's."""


def _recipe(payload: Payload) -> Recipe:
    """Read the recipe a payload carries.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    solprop.recipe.Recipe
        The recipe.
    """
    base_dir = payload.get("base")
    return Recipe.loads(
        str(payload["text"]), base=None if base_dir is None else Path(str(base_dir))
    )


def capillary_numbers(recipe: Recipe, energy: float) -> SolitonParams:
    """Return ``tools.capillary_params`` at a capillary's entrance.

    The keywords the tool shares with the route are passed on, with the route's default
    where the recipe gives none (the pulse shape's differs: Gaussian).

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        A resolved capillary recipe.
    energy : float
        The pulse energy launched, J.

    Returns
    -------
    solprop.tools.SolitonParams
        The numbers.

    Raises
    ------
    KeyError
        If the recipe gives no ``fwhm``: the numbers need it.
    """
    keywords = recipe.keywords()
    defaults = parameters(recipe.route)
    shared = {
        name: keywords.get(name, defaults[name].default)
        for name in _SHARED
        if keywords.get(name, defaults[name].default) is not None
    }
    pressure = keywords.get("pressure")
    if isinstance(pressure, tuple | list):
        # A gradient: the numbers at the entrance.
        first = pressure[0]
        pressure = first[1] if isinstance(first, tuple | list) else first
    return capillary_params(
        float(keywords["radius"]),  # pyright: ignore[reportArgumentType]
        keywords["gas"],  # pyright: ignore[reportArgumentType]
        None if pressure is None else float(pressure),  # pyright: ignore[reportArgumentType]
        energy=energy,
        fwhm=float(keywords["fwhm"]),  # pyright: ignore[reportArgumentType]
        wavelength=float(keywords["wavelength"]),  # pyright: ignore[reportArgumentType]
        **shared,  # pyright: ignore[reportArgumentType]
    )


def _as_numbers(params: SolitonParams) -> dict[str, float]:
    """Return solprop's numbers as plain floats.

    Parameters
    ----------
    params : solprop.tools.SolitonParams
        The numbers.

    Returns
    -------
    dict
        ``soliton_order``, ``beta2`` (s^2/m), ``gamma`` (1/(W m)), ``n2`` (m^2/W),
        ``peak_intensity`` (W/m^2), ``peak_power`` (W), ``pressure`` (bar), ``zdw``
        (m), ``fission_length`` (m) and ``critical_power`` (W).
    """
    return {
        "soliton_order": float(params.soliton_order),
        "beta2": float(params.beta2),
        "gamma": float(params.gamma),
        "n2": float(params.n2),
        "peak_intensity": float(params.peak_intensity),
        "peak_power": float(params.peak_power),
        "pressure": float(params.pressure),
        "zdw": float(params.zdw),
        "fission_length": float(params.fission_length),
        "critical_power": float(params.critical_power),
    }


def _shortest_wave(  # noqa: PLR0913 - what the tool reads
    mode: object,
    wavelength: float,
    *,
    beta2: float,
    peak_power: float,
    gamma: float,
    fwhm: float,
    shape: str,
) -> float | None:
    """Return the shortest wavelength phase-matched to the launch soliton.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The launch mode.
    wavelength : float
        The pump's wavelength, m.
    beta2 : float
        The dispersion at the pump, s^2/m.
    peak_power : float
        The peak power, W.
    gamma : float
        The nonlinear coefficient, 1/(W m).
    fwhm : float
        The duration, s.
    shape : str
        The pulse's shape.

    Returns
    -------
    float or None
        The wavelength, m; ``None`` when none is, as a pump in normal dispersion
        launches no soliton.
    """
    if beta2 >= 0:
        return None
    roots = lambda_rdw_all(
        mode,  # pyright: ignore[reportArgumentType]
        wavelength,
        peak_power=peak_power,
        gamma=gamma,
        fwhm=fwhm,
        shape=shape,  # pyright: ignore[reportArgumentType]
    )
    return float(roots[0]) if len(roots) else None


def design_numbers(recipe: Recipe, energy: float) -> dict[str, float]:
    """Return a capillary recipe's design numbers, with its dispersive wave.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        A resolved capillary recipe.
    energy : float
        The pulse energy launched, J (given, or from the peak power).

    Returns
    -------
    dict
        :func:`_as_numbers`, and ``rdw`` (m) where a wave is phase-matched.
    """
    params = capillary_numbers(recipe, energy)
    numbers = _as_numbers(params)
    wave = _shortest_wave(
        params.mode,
        params.wavelength,
        beta2=params.beta2,
        peak_power=params.peak_power,
        gamma=params.gamma,
        fwhm=params.fwhm,
        shape=params.shape,
    )
    if wave is not None:
        numbers["rdw"] = wave
    return numbers


def capillary_sheet(payload: Payload) -> dict[str, object]:
    """Return a capillary sheet's numbers, any pin solved.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        :func:`_as_numbers`; ``energy`` (J), ``fwhm`` (s) and ``shape``, which the
        curve's request reads back; ``pinned``, by pinned path, the solved value in its
        keyword's unit; and ``resolved``, the recipe's text with its pins solved.
    """
    authored = _recipe(payload)
    recipe = authored.resolve()
    keywords = recipe.keywords()
    energy = keywords.get("energy")
    if energy is None:
        energy = recipe.preview_input().energy
    params = capillary_numbers(recipe, float(energy))  # pyright: ignore[reportArgumentType]
    return {
        **_as_numbers(params),
        "energy": float(energy),  # pyright: ignore[reportArgumentType]
        "fwhm": float(params.fwhm),
        "shape": str(params.shape),
        "wavelength": float(params.wavelength),
        "pinned": {
            path: float(recipe.get(path))  # pyright: ignore[reportArgumentType]
            for path in authored.to_dict().get("pins", {})
        },
        "resolved": recipe.to_text(),
    }


def capillary_curve(payload: Payload) -> dict[str, object]:
    """Return a capillary sheet's dispersion curve, loss and dispersive wave.

    Parameters
    ----------
    payload : Mapping
        ``text``, a recipe with no pin left to solve, optionally ``base``, and what
        :func:`capillary_sheet` found: ``beta2``, ``peak_power``, ``gamma``, ``fwhm``,
        ``shape`` and ``wavelength``.

    Returns
    -------
    dict
        ``wavelength`` (m), ``beta2`` (s^2/m) along it, ``loss`` (the power loss
        coefficient at the pump, 1/m) and ``rdw`` (m, or ``None``).
    """
    recipe = _recipe(payload)
    mode = recipe.build().modes[0]
    pump = float(payload["wavelength"])  # pyright: ignore[reportArgumentType]
    limits = recipe.keywords().get("wavelength_limits")
    low, high = (
        limits
        if isinstance(limits, tuple | list)
        else (pump * _CURVE_WINDOW[0], pump * _CURVE_WINDOW[1])
    )
    wavelength = np.geomspace(float(low), float(high), CURVE_POINTS)
    omega = 2.0 * math.pi * C / wavelength
    return {
        "wavelength": wavelength,
        "beta2": np.asarray(base.beta2(mode, omega)),
        "loss": float(
            np.asarray(base.alpha(mode, np.array([2 * math.pi * C / pump])))[0]
        ),
        "rdw": _shortest_wave(
            mode,
            pump,
            beta2=float(payload["beta2"]),  # pyright: ignore[reportArgumentType]
            peak_power=float(payload["peak_power"]),  # pyright: ignore[reportArgumentType]
            gamma=float(payload["gamma"]),  # pyright: ignore[reportArgumentType]
            fwhm=float(payload["fwhm"]),  # pyright: ignore[reportArgumentType]
            shape=str(payload["shape"]),
        ),
    }


def output_size(payload: Payload) -> dict[str, object]:
    """Return the size of the run a recipe makes (``Simulation.output_bytes``).

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``output_bytes``.
    """
    return {"output_bytes": _recipe(payload).resolve().build().output_bytes()}


def gases(payload: Payload) -> list[str]:  # noqa: ARG001 - none
    """List the gases solprop knows.

    Parameters
    ----------
    payload : Mapping
        Unused.

    Returns
    -------
    list of str
        Their names, as a recipe writes them.
    """
    return [gas.value for gas in Gas]


SHEET_HANDLERS = {
    "capillary_curve": capillary_curve,
    "capillary_sheet": capillary_sheet,
    "gases": gases,
    "output_size": output_size,
}
"""The sheet's handlers, by request kind."""
