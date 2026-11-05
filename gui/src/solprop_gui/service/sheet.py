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

import contextlib
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from solprop.backends import ThreadBudget, available, thread_budget
from solprop.backends.registry import metal_available
from solprop.constants import C
from solprop.exceptions import SolpropError
from solprop.grid import EnvelopeGrid
from solprop.modes import base
from solprop.physdata.gases import Gas
from solprop.plotting.geometry import LIGHT, Disc, Ring, outline
from solprop.recipe import Recipe
from solprop.recipe.errors import RecipeError
from solprop.recipe.metadata import parameters
from solprop.recipe.quantity import Unit, parse_quantity
from solprop.tools import (
    SolitonParams,
    beam_params,
    beam_through_lens,
    capillary_params,
    dispersion_length,
    fission_length,
    lambda_rdw_all,
    nonlinear_coefficient,
    nonlinear_length,
    pulse_peak_power,
    pulse_width,
    soliton_order,
    soliton_params,
)

from solprop_gui.service.kernel import RUN_THREADS

__all__ = [
    "SHEET_HANDLERS",
    "beam_curve",
    "beam_sheet",
    "capillary_curve",
    "capillary_numbers",
    "capillary_sheet",
    "design_numbers",
    "fibre_curve",
    "fibre_sheet",
    "gases",
    "grin_curve",
    "grin_sheet",
    "lens_beam",
    "output_size",
    "run_options",
]

Payload = Mapping[str, object]

_SHARED = ("shape", "temperature", "model", "loss", "cladding", "materials")
"""The capillary's keywords that ``tools.capillary_params`` also takes, by name."""

CURVE_POINTS = 240
"""The wavelengths of a sheet's dispersion curve."""

_DB_PER_NEPER = 10.0 / math.log(10.0)
"""What a power loss coefficient in 1/m is multiplied by to read in dB/m."""

_NEAR_PUMP = 0.35
"""How far from the pump, as a share of the frequency, a curve's axis is judged."""

_NEAR_REACH = 3.0
"""How far past the pump's neighbourhood a curve's axis may reach, as a multiple."""

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
    gas: object = keywords["gas"]
    fill = None if pressure is None else float(pressure)  # pyright: ignore[reportArgumentType]
    if keywords.get("lattice") is not None:
        # An antiresonant fibre's dispersion is its own, so its mode gives the numbers.
        return soliton_params(
            recipe.build().modes[0],
            gas,  # pyright: ignore[reportArgumentType]
            fill,
            energy=energy,
            fwhm=float(keywords["fwhm"]),  # pyright: ignore[reportArgumentType]
            wavelength=float(keywords["wavelength"]),  # pyright: ignore[reportArgumentType]
            shape=shared.get("shape", "sech"),  # pyright: ignore[reportArgumentType]
        )
    return capillary_params(
        float(keywords["radius"]),  # pyright: ignore[reportArgumentType]
        gas,  # pyright: ignore[reportArgumentType]
        fill,
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
        "dispersion_length": params.dispersion_length,
        "nonlinear_length": params.nonlinear_length,
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
    limits: tuple[float, float] | None = None,
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
    limits : tuple of float or None, optional
        The range to search, m; ``None`` takes the tool's.

    Returns
    -------
    float or None
        The wavelength, m; ``None`` when none is, as a pump in normal dispersion
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


def _wave_window(recipe: Recipe, pump: float) -> tuple[float, float] | None:
    """Return where to look for a dispersive wave in an antiresonant fibre.

    The tool's own range reaches where this fibre's fits go non-finite (#538), so the
    search keeps to the run's window; a capillary keeps the tool's.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.
    pump : float
        The pump's wavelength, m.

    Returns
    -------
    tuple of float or None
        The range, m; ``None`` for the tool's own.
    """
    if recipe.keywords().get("lattice") is None:
        return None
    return _curve_range(recipe, pump)[0], 0.9 * pump


def _zero_near(wavelength: np.ndarray, beta2: np.ndarray, pump: float) -> float | None:
    """Return the zero of a dispersion curve nearest a pump, read from the curve.

    Parameters
    ----------
    wavelength : numpy.ndarray
        The curve's wavelengths, m, ascending.
    beta2 : numpy.ndarray
        Its group-velocity dispersion there.
    pump : float
        The pump's wavelength, m.

    Returns
    -------
    float or None
        The wavelength, m; ``None`` where the curve does not cross zero.
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


def _lattice_zero(recipe: Recipe, mode: object, pump: float) -> float:
    """Return the zero an antiresonant fibre's run can reach.

    The tool reports the zero nearest in frequency to the pump, which may lie far
    outside the run's window, beyond the curve the sheet draws: 15 um with a 180 nm
    wall has one at 1919 nm as well as 485 nm for an 800 nm pump, with the window
    ending at 1600 nm. The sheet's number is the one on its curve.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.
    mode : solprop.modes.protocol.Mode
        The fibre's mode.
    pump : float
        The pump's wavelength, m.

    Returns
    -------
    float
        The wavelength, m; NaN where the window holds no zero.
    """
    low, high = _curve_range(recipe, pump)
    wavelength = np.geomspace(low, high, CURVE_POINTS)
    beta2 = np.asarray(base.beta2(mode, 2.0 * math.pi * C / wavelength))  # pyright: ignore[reportArgumentType]
    zero = _zero_near(wavelength, beta2, pump)
    return math.nan if zero is None else zero


def _curve_range(recipe: Recipe, pump: float) -> tuple[float, float]:
    """Return the wavelengths a dispersion curve covers: the run's own window.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.
    pump : float
        The pump's wavelength, m.

    Returns
    -------
    tuple of float
        The two ends, m.
    """
    keywords = recipe.keywords()
    limits = keywords.get("wavelength_limits")
    if isinstance(limits, tuple | list) and len(limits) == 2:
        return float(limits[0]), float(limits[1])  # pyright: ignore[reportArgumentType]
    grid = keywords.get("grid")
    if isinstance(grid, EnvelopeGrid):
        return 2.0 * math.pi * C / grid.omega_max, 2.0 * math.pi * C / grid.omega_min
    return pump * _CURVE_WINDOW[0], pump * _CURVE_WINDOW[1]


def _reach_of(values: np.ndarray, wavelength: np.ndarray, pump: float) -> float:
    """Return how far a dispersion curve's axis should reach from zero.

    The curve runs off near a resonance, and a fibre's resonance dwarfs the pump's
    neighbourhood, so the axis follows what the eye reads there.

    Parameters
    ----------
    values : numpy.ndarray
        The curve.
    wavelength : numpy.ndarray
        Its wavelengths, m, ascending.
    pump : float
        The pump's wavelength, m.

    Returns
    -------
    float
        The reach, in the curve's unit.
    """
    reach = float(np.nanmax(np.abs(values[wavelength > 0.25 * wavelength[-1]])))
    omega, centre = 1.0 / wavelength, 1.0 / pump
    near = np.abs(omega - centre) < _NEAR_PUMP * omega
    around = float(np.nanmax(np.abs(values[near]))) if near.any() else reach
    return max(min(reach, _NEAR_REACH * around), 1.0) * 1.05


def _shapes_of(mode: object) -> list[dict[str, object]]:
    """Return a guide's cross-section as plain shapes, to scale.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The guide's mode.

    Returns
    -------
    list of dict
        In drawing order, each ``kind`` and its numbers in metres, with ``fill`` and
        ``edge``, solprop's colours of its region.
    """
    plain: list[dict[str, object]] = []
    for shape in outline(mode):
        fill = str(getattr(LIGHT, shape.region))
        edge = str(getattr(LIGHT, f"{shape.region}_edge", fill))
        if isinstance(shape, Disc):
            numbers: dict[str, object] = {
                "kind": "disc",
                "centre": list(shape.centre),
                "radius": shape.radius,
            }
        elif isinstance(shape, Ring):
            numbers = {
                "kind": "ring",
                "centre": list(shape.centre),
                "inner": shape.inner,
                "outer": shape.outer,
            }
        else:
            numbers = {
                "kind": "box",
                "lower": list(shape.lower),
                "upper": list(shape.upper),
            }
        plain.append({**numbers, "fill": fill, "edge": edge})
    return plain


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
        limits=_wave_window(recipe, params.wavelength),
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
    numbers = _as_numbers(params)
    if recipe.keywords().get("lattice") is not None:
        numbers["zdw"] = _lattice_zero(recipe, params.mode, params.wavelength)
    return {
        **numbers,
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
        ``text``, a recipe with no pin left to solve, optionally ``base``, and
        ``numbers``, what :func:`capillary_sheet` found.

    Returns
    -------
    dict
        ``plot`` (what the sheet draws: ``x``, ``y``, their labels, ``unit``, ``limit``
        and ``marks``), and ``numbers``, by key the dispersive wave ``rdw`` (m, or
        ``None``) and the ``loss`` at the pump (dB/m). With a lattice, also ``loss``
        (``x``, ``y`` in dB/m and ``pump``) and ``shapes``, its cross-section.
    """
    recipe = _recipe(payload)
    numbers: Mapping[str, object] = payload["numbers"]  # pyright: ignore[reportAssignmentType]
    mode = recipe.build().modes[0]
    pump = float(numbers["wavelength"])  # pyright: ignore[reportArgumentType]
    low, high = _curve_range(recipe, pump)
    wavelength = np.geomspace(low, high, CURVE_POINTS)
    omega = 2.0 * math.pi * C / wavelength
    beta2 = np.asarray(base.beta2(mode, omega)) * 1e30
    loss = float(base.alpha(mode, np.array([2 * math.pi * C / pump]))[0])
    wave = _shortest_wave(
        mode,
        pump,
        beta2=float(numbers["beta2"]),  # pyright: ignore[reportArgumentType]
        peak_power=float(numbers["peak_power"]),  # pyright: ignore[reportArgumentType]
        gamma=float(numbers["gamma"]),  # pyright: ignore[reportArgumentType]
        fwhm=float(numbers["fwhm"]),  # pyright: ignore[reportArgumentType]
        shape=str(numbers["shape"]),
        limits=_wave_window(recipe, pump),
    )
    nanometres = wavelength * 1e9
    top = _reach_of(beta2, wavelength, pump)
    zdw = float(numbers["zdw"])  # pyright: ignore[reportArgumentType]
    reply: dict[str, object] = {
        "plot": {
            "x": nanometres,
            "y": beta2,
            "xlabel": "Wavelength (nm)",
            "ylabel": "β₂ (fs²/m)",
            "unit": "fs²/m",
            "limit": [-top, top],
            "zero": True,
            "marks": [
                ["Pump", pump * 1e9],
                ["ZDW", zdw * 1e9 if math.isfinite(zdw) else None],
                ["Dispersive wave", None if wave is None else wave * 1e9],
            ],
        },
        "numbers": {"rdw": wave, "loss": loss * _DB_PER_NEPER},
    }
    if recipe.keywords().get("lattice") is not None:
        reply["loss"] = {
            "x": nanometres,
            "y": np.asarray(base.alpha(mode, omega)) * _DB_PER_NEPER,
            "pump": pump * 1e9,
        }
        reply["shapes"] = _shapes_of(mode)
    return reply


def fibre_sheet(payload: Payload) -> dict[str, object]:
    """Return a solid-core fibre's numbers, from its recipe's mode and pulse.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``peak_power`` (W), ``energy`` (J), ``beta2`` (s^2/m), ``wavelength`` (m),
        ``gamma`` (1/(W m), the recipe's, or a solved fibre's from its mode and
        ``n2``), ``dispersion_length`` and ``nonlinear_length`` (m), and where a
        soliton forms,
        ``soliton_order`` and ``fission_length`` (m); ``pinned`` (empty: this route has
        no pins) and ``resolved``, the recipe's text. What the recipe does not give
        is left out.
    """
    recipe = _recipe(payload).resolve()
    keywords = recipe.keywords()
    mode = recipe.build().modes[0]
    pump = float(keywords["wavelength"])  # pyright: ignore[reportArgumentType]
    preview = recipe.preview_input()
    power = float(preview.peak_power)
    beta2 = float(base.beta2(mode, np.array([2.0 * math.pi * C / pump]))[0])
    reply: dict[str, object] = {
        "peak_power": power,
        "energy": float(preview.energy),
        "beta2": beta2,
        "wavelength": pump,
        "pinned": {},
        "resolved": recipe.to_text(),
    }
    shape = str(keywords.get("shape") or "sech")
    width = keywords.get("width")
    fwhm = keywords.get("fwhm")
    if fwhm is None and width is not None:
        fwhm = float(width) / pulse_width(1.0, shape)  # pyright: ignore[reportArgumentType, arg-type]
    gamma = keywords.get("gamma")
    fibre, n2 = keywords.get("fibre"), keywords.get("n2")
    if not isinstance(gamma, int | float) and fibre is not None and n2 is not None:
        # A solved fibre's coefficient is its mode's, with the glass's n2.
        gamma = nonlinear_coefficient(fibre, float(n2), pump)  # pyright: ignore[reportArgumentType]
    if not isinstance(gamma, int | float):
        return reply
    reply["gamma"] = float(gamma)
    if fwhm is None or beta2 == 0:
        return reply
    fwhm, gamma = float(fwhm), float(gamma)  # pyright: ignore[reportArgumentType]
    reply["dispersion_length"] = dispersion_length(fwhm, beta2, shape)  # pyright: ignore[reportArgumentType]
    reply["nonlinear_length"] = nonlinear_length(power, gamma)
    if beta2 < 0:
        reply["soliton_order"] = soliton_order(power, fwhm, gamma, beta2, shape)  # pyright: ignore[reportArgumentType]
        reply["fission_length"] = fission_length(power, fwhm, gamma, beta2, shape)  # pyright: ignore[reportArgumentType]
    return reply


def fibre_curve(payload: Payload) -> dict[str, object]:
    """Return a solid-core fibre's dispersion curve.

    The zero is read from the curve, to what the sheet can show.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``, and ``numbers``, what :func:`fibre_sheet`
        found.

    Returns
    -------
    dict
        ``plot`` (as :func:`capillary_curve` gives it) and ``numbers``, by key the
        ``zdw`` nearest the pump (m, or ``None``); a solved fibre's ``shapes``, its
        cross-section.
    """
    recipe = _recipe(payload)
    numbers: Mapping[str, object] = payload["numbers"]  # pyright: ignore[reportAssignmentType]
    pump = float(numbers["wavelength"])  # pyright: ignore[reportArgumentType]
    mode = recipe.build().modes[0]
    wavelength = np.geomspace(*_curve_range(recipe, pump), CURVE_POINTS)
    beta2 = np.asarray(base.beta2(mode, 2.0 * math.pi * C / wavelength)) * 1e30
    zdw = _zero_near(wavelength, beta2, pump)
    reach = float(np.nanmax(np.abs(beta2))) or 1.0
    reply: dict[str, object] = {
        "plot": {
            "x": wavelength * 1e9,
            "y": beta2,
            "xlabel": "Wavelength (nm)",
            "ylabel": "β₂ (fs²/m)",
            "unit": "fs²/m",
            "limit": [-reach * 1.05, reach * 1.05],
            "zero": True,
            "marks": [
                ["Pump", pump * 1e9],
                ["ZDW", None if zdw is None else zdw * 1e9],
            ],
        },
        "numbers": {"zdw": zdw},
    }
    fibre = recipe.keywords().get("fibre")
    if fibre is not None:
        reply["shapes"] = _shapes_of(fibre)
    return reply


def beam_sheet(payload: Payload) -> dict[str, object]:
    """Return a filament sheet's numbers.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``power_ratio``, ``critical_power`` (W), ``peak_power`` (W), ``rayleigh_range``
        (m), ``collapse_distance`` (m, or ``None``), ``waist``, ``focus`` and ``length``
        (m), ``pinned`` (empty: this route has no pins) and ``resolved``, the recipe's
        text. A medium that is no gas has only its ``peak_power``, with ``refusal``
        saying why.
    """
    recipe = _recipe(payload)
    keywords = recipe.keywords()
    beam = keywords["initial"]
    medium = keywords["medium"]
    power = float(recipe.preview_input().peak_power)
    reply: dict[str, object] = {
        "peak_power": power,
        "waist": float(beam.waist),  # pyright: ignore[reportAttributeAccessIssue]
        "focus": -float(beam.propz),  # pyright: ignore[reportAttributeAccessIssue]
        "length": float(keywords["grid"].length),  # pyright: ignore[reportAttributeAccessIssue]
        "wavelength": float(beam.temporal.wavelength),  # pyright: ignore[reportAttributeAccessIssue]
        "pinned": {},
        "resolved": recipe.to_text(),
    }
    try:
        found = beam_params(
            float(reply["waist"]),  # pyright: ignore[reportArgumentType]
            float(reply["wavelength"]),  # pyright: ignore[reportArgumentType]
            medium.material,  # pyright: ignore[reportAttributeAccessIssue]
            float(medium.pressure),  # pyright: ignore[reportAttributeAccessIssue]
            power,
            focus=float(reply["focus"]),  # pyright: ignore[reportArgumentType]
        )
    except (SolpropError, TypeError, ValueError) as error:
        reply["refusal"] = str(error)
        return reply
    reply.update(
        power_ratio=found.power_ratio,
        critical_power=found.critical_power,
        rayleigh_range=found.rayleigh_range,
        collapse_distance=found.collapse_distance,
    )
    return reply


def beam_curve(payload: Payload) -> dict[str, object]:
    """Return a filament sheet's linear beam radius along the propagation.

    Parameters
    ----------
    payload : Mapping
        ``numbers``, what :func:`beam_sheet` found.

    Returns
    -------
    dict
        ``plot`` (as :func:`capillary_curve` gives it) and ``numbers``, empty.
    """
    numbers: Mapping[str, object] = payload["numbers"]  # pyright: ignore[reportAssignmentType]
    length = float(numbers["length"])  # pyright: ignore[reportArgumentType]
    z = np.linspace(0.0, length, CURVE_POINTS)
    marks: list[list[object]] = []
    focus = float(numbers["focus"])  # pyright: ignore[reportArgumentType]
    collapse = numbers.get("collapse_distance")
    if 0.0 < focus <= length:
        marks.append(["Focus", focus * 100.0])
    if isinstance(collapse, float) and collapse <= length:
        marks.append(["Collapse (Marburger)", collapse * 100.0])
    rayleigh = numbers.get("rayleigh_range")
    if not isinstance(rayleigh, float):
        return {"plot": None, "numbers": {}}
    offset = (z - focus) / rayleigh
    # BeamParams.radius, from the numbers already at hand.
    radius = float(numbers["waist"]) * np.sqrt(1.0 + offset**2) * 1e6  # pyright: ignore[reportArgumentType]
    return {
        "plot": {
            "x": z * 100.0,
            "y": radius,
            "xlabel": "z (cm)",
            "ylabel": "Beam radius (µm)",
            "unit": "µm",
            "limit": [0.0, float(radius.max()) * 1.1],
            "zero": False,
            "marks": marks,
        },
        "numbers": {},
    }


def lens_beam(payload: Payload) -> dict[str, float]:
    """Return the waist a lens makes (``tools.beam_through_lens``).

    Parameters
    ----------
    payload : Mapping
        ``focal_length``, ``radius`` (quantity strings or SI numbers) and ``text``, the
        recipe whose ``wavelength`` path says the light's wavelength.

    Returns
    -------
    dict
        ``waist`` and ``distance``, m.
    """
    wavelength = _recipe(payload).keywords()["initial"].temporal.wavelength  # pyright: ignore[reportAttributeAccessIssue]
    waist, distance = beam_through_lens(
        float(payload["focal_length"]),  # pyright: ignore[reportArgumentType]
        float(payload["radius"]),  # pyright: ignore[reportArgumentType]
        float(wavelength),
    )
    return {"waist": waist, "distance": distance}


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


def _run_budget() -> ThreadBudget:
    """Return the thread budget a run gets, not the kernel's own, which is smaller.

    Returns
    -------
    solprop.backends.ThreadBudget
        The budget under the ``SOLPROP_THREADS`` the window started with.
    """
    held = os.environ.get("SOLPROP_THREADS")
    started = os.environ.get(RUN_THREADS)
    if started is None:
        return thread_budget()
    try:
        if started:
            os.environ["SOLPROP_THREADS"] = started
        else:
            os.environ.pop("SOLPROP_THREADS", None)
        return thread_budget()
    finally:
        if held is None:
            os.environ.pop("SOLPROP_THREADS", None)
        else:
            os.environ["SOLPROP_THREADS"] = held


def _graded(recipe: Recipe) -> dict[str, float]:
    """Return a graded-index fibre's profile, as its recipe writes it.

    Read from the recipe's text alone: building the fibre solves its modes, which is
    seconds, not the moment an edit takes.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe, its provider a ``GradedIndexProvider`` inside ``provider``.

    Returns
    -------
    dict
        ``radius`` (m), ``contrast``, ``exponent``, and where they are plain indices,
        ``axis`` and ``exterior``.
    """
    inner = "provider.provider"
    profile: dict[str, float] = {
        "radius": float(
            parse_quantity(recipe.get(f"{inner}.radius"), Unit("m"), path="radius")
        ),
        "contrast": float(recipe.get(f"{inner}.permittivity.contrast")),  # pyright: ignore[reportArgumentType]
        "exponent": 2.0,
    }
    with contextlib.suppress(RecipeError):
        profile["exponent"] = float(recipe.get(f"{inner}.permittivity.exponent"))  # pyright: ignore[reportArgumentType]
    for key, path in (
        ("axis", f"{inner}.permittivity.centre.index"),
        ("exterior", f"{inner}.exterior.index"),
    ):
        with contextlib.suppress(RecipeError):
            held = recipe.get(path)
            if isinstance(held, int | float):
                profile[key] = float(held)
    return profile


def grin_sheet(payload: Payload) -> dict[str, object]:
    """Return a graded-index fibre's numbers, from its profile and its pulse.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``peak_power`` (W), ``energy`` (J), ``wavelength`` (m), and where the indices
        are plain numbers its numerical aperture ``na``, ``v_number`` and
        ``mode_count``, the guided modes a power-law core holds, about
        V² p / (2 (p + 2)); ``pinned`` (empty) and ``resolved``, the recipe's text.
    """
    recipe = _recipe(payload)
    metre, joule, second = Unit("m"), Unit("J"), Unit("s")
    pump = float(parse_quantity(recipe.get("wavelength"), metre, path="wavelength"))
    energy = float(parse_quantity(recipe.get("energy"), joule, path="energy"))
    fwhm = float(parse_quantity(recipe.get("fwhm"), second, path="fwhm"))
    shape = str(recipe.get("shape"))
    reply: dict[str, object] = {
        "energy": energy,
        "peak_power": pulse_peak_power(energy, fwhm, shape),  # pyright: ignore[reportArgumentType]
        "wavelength": pump,
        "pinned": {},
        "resolved": recipe.to_text(),
    }
    profile = _graded(recipe)
    if "axis" in profile and "exterior" in profile:
        aperture = math.sqrt(max(profile["axis"] ** 2 - profile["exterior"] ** 2, 0.0))
        v_number = 2.0 * math.pi * profile["radius"] * aperture / pump
        power = profile["exponent"]
        reply["na"] = aperture
        reply["v_number"] = v_number
        reply["mode_count"] = v_number**2 * power / (2.0 * (power + 2.0))
    return reply


def grin_curve(payload: Payload) -> dict[str, object]:
    """Return a graded-index fibre's index across its core and into its cladding.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``plot``, the index against radius in µm with the core's edge marked, as
        :func:`capillary_curve` gives one; none where the indices are not plain
        numbers. ``numbers`` is empty.
    """
    profile = _graded(_recipe(payload))
    if "axis" not in profile or "exterior" not in profile:
        return {"numbers": {}}
    radius = profile["radius"]
    r = np.linspace(0.0, 1.6 * radius, CURVE_POINTS)
    inside = profile["axis"] * np.sqrt(
        np.maximum(
            1.0 - 2.0 * profile["contrast"] * (r / radius) ** profile["exponent"], 0.0
        )
    )
    index = np.where(r <= radius, inside, profile["exterior"])
    low, high = float(index.min()), float(index.max())
    pad = 0.08 * (high - low or 1e-3)
    return {
        "plot": {
            "x": r * 1e6,
            "y": index,
            "xlabel": "Radius (µm)",
            "ylabel": "Refractive index",
            "unit": "",
            "limit": [low - pad, high + pad],
            "zero": False,
            "marks": [["Core edge", radius * 1e6]],
        },
        "numbers": {},
    }


def run_options(payload: Payload) -> dict[str, object]:  # noqa: ARG001 - none
    """Say what a run here can use: its threads and its backends.

    Parameters
    ----------
    payload : Mapping
        Unused.

    Returns
    -------
    dict
        ``threads`` (the budget a run gets by default), ``limit`` (what sets it),
        ``double`` and ``single`` (the backends installed for each precision) and
        ``metal`` (whether the MLX backend has a Metal GPU here).
    """
    budget = _run_budget()
    return {
        "threads": budget.threads,
        "limit": budget.source,
        "double": list(available("double")),
        "single": list(available("single")),
        "metal": metal_available(),
    }


SHEET_HANDLERS = {
    "beam_curve": beam_curve,
    "beam_sheet": beam_sheet,
    "capillary_curve": capillary_curve,
    "capillary_sheet": capillary_sheet,
    "fibre_curve": fibre_curve,
    "fibre_sheet": fibre_sheet,
    "grin_curve": grin_curve,
    "grin_sheet": grin_sheet,
    "gases": gases,
    "lens_beam": lens_beam,
    "output_size": output_size,
    "run_options": run_options,
}
"""The sheet's handlers, by request kind."""
