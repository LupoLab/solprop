# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design a waveguide: its curves and numbers, with no pulse and no run (design 7.4).

Everything here starts from a class's example with the design's values set, so a
design is a recipe at every moment: what it shows is what the recipe builds, and
*Simulate this* carries that recipe on. A pulse the design does not give stays the
example's, and nothing is said of a pulse until one is given.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from solprop.constants import C
from solprop.exceptions import SolpropError
from solprop.modes import base
from solprop.recipe import Recipe
from solprop.service.design import _CHART_KEYWORDS
from solprop.service.explore import QUANTITIES, quantity_values
from solprop.service.sheet import (
    CURVE_POINTS,
    _curve_range,
    _reach_of,
    _shortest_wave,
    _wave_window,
    _zero_near,
    capillary_sheet,
)
from solprop.tools import capillary_design, phase_mismatch, pressure_zdw

__all__ = [
    "WAVEGUIDE_HANDLERS",
    "design_recipe",
    "waveguide_curves",
    "waveguide_numbers",
    "waveguide_solve",
]

Payload = Mapping[str, object]

_EXAMPLES: Final = {"capillary": "rdw", "antiresonant": "antiresonant_fibre"}
"""The example each guide starts from."""

_SOURCES: Final = {quantity.key: quantity for quantity in QUANTITIES}
"""The quantities a curve can be of, by key."""

_DB_PER_NEPER: Final = 10.0 / math.log(10.0)
"""What a power loss coefficient in 1/m is multiplied by to read in dB/m."""


def design_recipe(payload: Payload) -> Recipe:
    """Return the recipe a design stands for.

    Parameters
    ----------
    payload : Mapping
        ``guide`` (``"capillary"`` or ``"antiresonant"``) and ``values``, by parameter
        path, as a recipe writes them. A pump moved from the example's takes the
        wavelength window with it.

    Returns
    -------
    solprop.recipe.Recipe
        The recipe, with one mode.

    Raises
    ------
    solprop.recipe.errors.RecipeError
        If a value does not read.
    """
    recipe = Recipe.example(_EXAMPLES[str(payload["guide"])])
    recipe.set("modes", 1)
    before = float(recipe.keywords()["wavelength"])  # pyright: ignore[reportArgumentType]
    limits = recipe.keywords()["wavelength_limits"]
    values: Mapping[str, object] = payload["values"]  # pyright: ignore[reportAssignmentType]
    for path, value in values.items():
        recipe.set(path, value)
    pump = float(recipe.keywords()["wavelength"])  # pyright: ignore[reportArgumentType]
    if pump != before:
        low, high = limits  # pyright: ignore[reportGeneralTypeIssues]
        recipe.set(
            "wavelength_limits",
            [float(low) * pump / before, float(high) * pump / before],
        )
    return recipe


def _curve_of(
    recipe: Recipe, key: str, numbers: Mapping[str, object] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return one recipe's curve of a quantity against wavelength.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.
    key : str
        The quantity's key; of :data:`~solprop.service.explore.QUANTITIES`.
    numbers : Mapping or None
        The pulse's numbers, for a phase mismatch; ``None`` for none.

    Returns
    -------
    tuple of numpy.ndarray
        The wavelengths, m, and the values in the quantity's unit.
    """
    mode = recipe.build().modes[0]
    pump = float(recipe.keywords()["wavelength"])  # pyright: ignore[reportArgumentType]
    wavelength = np.geomspace(*_curve_range(recipe, pump), CURVE_POINTS)
    omega = 2.0 * math.pi * C / wavelength
    quantity = _SOURCES[key]
    match quantity.source:
        case "neff_minus_one":
            values = np.real(mode.neff_minus_one(omega))
        case "beta1":
            values = base.beta1(mode, omega)
        case "beta3":
            values = base.beta3(mode, omega)
        case "alpha":
            values = base.alpha(mode, omega)
        case "phase_mismatch":
            pulse = (
                {}
                if numbers is None
                else {
                    "peak_power": float(numbers["peak_power"]),  # pyright: ignore[reportArgumentType]
                    "gamma": float(numbers["gamma"]),  # pyright: ignore[reportArgumentType]
                    "fwhm": float(numbers["fwhm"]),  # pyright: ignore[reportArgumentType]
                    "shape": str(numbers["shape"]),
                }
            )
            values = phase_mismatch(mode, omega, pump, **pulse)  # pyright: ignore[reportArgumentType]
        case _:
            values = base.beta2(mode, omega)
    drawn = quantity_values({quantity.source: values}, omega, quantity)
    assert drawn is not None  # the mode was just given it
    return wavelength, drawn


def _varied(payload: Payload) -> list[tuple[str | None, Payload]]:
    """Return a design for each value of its sweep, or the design alone.

    Parameters
    ----------
    payload : Mapping
        The design, with optionally ``sweep``: ``name`` (a parameter path) and
        ``values``.

    Returns
    -------
    list of tuple
        The value (``None`` without a sweep) and its design.
    """
    sweep = payload.get("sweep")
    if not isinstance(sweep, Mapping) or not sweep.get("values"):
        return [(None, payload)]
    name = str(sweep["name"])
    values: Sequence[str] = sweep["values"]
    held: Mapping[str, object] = payload["values"]  # pyright: ignore[reportAssignmentType]
    return [
        (str(value), {**payload, "values": {**held, name: value}}) for value in values
    ]


def waveguide_curves(payload: Payload) -> dict[str, object]:
    """Return a design's curves of one quantity, one per value of its sweep.

    Parameters
    ----------
    payload : Mapping
        ``guide``, ``values``, optionally ``sweep`` (:func:`design_recipe`,
        :func:`_varied`), ``quantity`` and ``pulse``, whether the design gives one.

    Returns
    -------
    dict
        ``x`` (nm), ``pump`` (nm), ``key`` (the quantity), ``limit`` (its axis, or
        ``None`` for its whole range), and ``curves``, each ``value`` with ``y``,
        ``zero`` (the zero of its dispersion nearest the pump, nm, or ``None``) or the
        ``error`` that value met.
    """
    key = str(payload.get("quantity") or "beta2")
    pulse = bool(payload.get("pulse"))
    x = np.zeros(0)
    pump = math.nan
    curves: list[dict[str, object]] = []
    for value, design in _varied(payload):
        try:
            recipe = design_recipe(design)
            numbers = None
            if pulse and _SOURCES[key].source == "phase_mismatch":
                numbers = capillary_sheet({"text": recipe.to_text()})
            wavelength, y = _curve_of(recipe, key, numbers)
            dispersion = y if key == "beta2" else _curve_of(recipe, "beta2", None)[1]
        except (SolpropError, TypeError, ValueError) as error:
            curves.append({"value": value, "error": str(error)})
            continue
        x = wavelength * 1e9
        pump = float(recipe.keywords()["wavelength"]) * 1e9  # pyright: ignore[reportArgumentType]
        zero = _zero_near(wavelength, dispersion, pump * 1e-9)
        curves.append(
            {
                "value": value,
                "y": y,
                "zero": None if zero is None else zero * 1e9,
            }
        )
    limit = None
    drawn = [np.asarray(curve["y"], dtype=float) for curve in curves if "y" in curve]
    if drawn and key in ("beta2", "dispersion", "beta3", "mismatch"):
        limit = _reach_of(np.max(np.abs(drawn), axis=0), x * 1e-9, pump * 1e-9)
    return {"x": x, "pump": pump, "key": key, "limit": limit, "curves": curves}


def waveguide_numbers(payload: Payload) -> dict[str, object]:
    """Return a design's numbers at its pump, and its recipe.

    Parameters
    ----------
    payload : Mapping
        ``guide``, ``values`` and ``pulse``, whether the design gives one.

    Returns
    -------
    dict
        ``beta2``, ``beta3`` (SI), ``loss`` (dB/m), ``aeff`` (m^2), ``gamma`` (1/(W m))
        and ``zdw`` (m, NaN for none); with a pulse also ``soliton_order``,
        ``peak_power`` (W), ``peak_intensity`` (W/m^2), ``dispersion_length``,
        ``nonlinear_length``, ``fission_length`` (m) and ``rdw`` (m, or ``None``); and
        ``resolved``, the recipe's text.
    """
    recipe = design_recipe(payload)
    text = recipe.to_text()
    sheet = capillary_sheet({"text": text})
    mode = recipe.build().modes[0]
    pump = float(sheet["wavelength"])  # pyright: ignore[reportArgumentType]
    omega = np.array([2.0 * math.pi * C / pump])
    reply: dict[str, object] = {
        "beta2": sheet["beta2"],
        "beta3": float(base.beta3(mode, omega)[0]),
        "loss": float(base.alpha(mode, omega)[0]) * _DB_PER_NEPER,
        "aeff": float(base.aeff(mode)),
        "gamma": sheet["gamma"],
        "zdw": sheet["zdw"],
        "wavelength": pump,
        "resolved": text,
    }
    if payload.get("pulse"):
        for key in (
            "soliton_order",
            "peak_power",
            "peak_intensity",
            "dispersion_length",
            "nonlinear_length",
            "fission_length",
        ):
            reply[key] = sheet[key]
        reply["rdw"] = _shortest_wave(
            mode,
            pump,
            beta2=float(sheet["beta2"]),  # pyright: ignore[reportArgumentType]
            peak_power=float(sheet["peak_power"]),  # pyright: ignore[reportArgumentType]
            gamma=float(sheet["gamma"]),  # pyright: ignore[reportArgumentType]
            fwhm=float(sheet["fwhm"]),  # pyright: ignore[reportArgumentType]
            shape=str(sheet["shape"]),
            limits=_wave_window(recipe, pump),
        )
    return reply


def waveguide_solve(payload: Payload) -> dict[str, object]:
    """Solve a capillary's pressure from a ZDW, and its energy from a soliton order.

    Parameters
    ----------
    payload : Mapping
        ``guide`` and ``values``, ``zdw`` (m) and optionally ``soliton_order``.

    Returns
    -------
    dict
        ``pressure`` (bar), and with an order ``energy`` (J).
    """
    keywords = design_recipe(payload).keywords()
    zdw = float(payload["zdw"])  # pyright: ignore[reportArgumentType]
    radius = float(keywords["radius"])  # pyright: ignore[reportArgumentType]
    gas: object = keywords["gas"]
    fill = {
        name: keywords[name]
        for name in _CHART_KEYWORDS
        if keywords.get(name) is not None
    }
    order = payload.get("soliton_order")
    if order is None:
        return {"pressure": pressure_zdw(radius, gas, zdw, **fill)}  # pyright: ignore[reportArgumentType]
    design = capillary_design(
        radius,
        gas,  # pyright: ignore[reportArgumentType]
        zdw_wavelength=zdw,
        fwhm=float(keywords["fwhm"]),  # pyright: ignore[reportArgumentType]
        wavelength=float(keywords["wavelength"]),  # pyright: ignore[reportArgumentType]
        soliton_order=float(order),  # pyright: ignore[reportArgumentType]
        # The recipe's Gaussian, as the numbers the design then shows read it.
        shape=keywords.get("shape", "gauss"),  # pyright: ignore[reportArgumentType]
        **fill,  # pyright: ignore[reportArgumentType]
    )
    return {"pressure": float(design.pressure), "energy": float(design.energy)}


WAVEGUIDE_HANDLERS: Final = {
    "waveguide_curves": waveguide_curves,
    "waveguide_numbers": waveguide_numbers,
    "waveguide_solve": waveguide_solve,
}
"""The design stage's requests, by kind."""
