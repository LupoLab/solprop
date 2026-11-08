# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's design requests: the modes a recipe builds, and its pins (design 7).

The waveguide explorer draws what ``Recipe.modes`` gives, per mode, as plain arrays,
for the recipe as it is or for each value of one keyword swept; the pins' editors ask
what a route can pin and, for a pinned value, every root solprop finds. Like
:mod:`solprop.service.handlers`, this module runs only in the kernel process and
computes nothing of its own.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from solprop.exceptions import SolpropError
from solprop.recipe import Recipe
from solprop.recipe.pins import GUIDE_PINS, PULSE_PINS, TARGETS
from solprop.recipe.quantity import Unit, parse_quantity
from solprop.tools import pressure_zdw

__all__ = ["DESIGN_HANDLERS", "modes", "pin_roots", "pin_targets", "zdw_chart"]

Payload = Mapping[str, object]


def _base(payload: Payload) -> Path | None:
    """Return the directory a payload's recipe reads its data files from.

    Parameters
    ----------
    payload : Mapping
        Optionally ``base``.

    Returns
    -------
    pathlib.Path or None
        The directory, or ``None`` for none.
    """
    base = payload.get("base")
    return Path(str(base)) if base else None


def _recipe(payload: Payload) -> Recipe:
    """Read the recipe a payload carries.

    Parameters
    ----------
    payload : Mapping
        ``text``, the recipe's TOML, and optionally ``base``.

    Returns
    -------
    solprop.recipe.Recipe
        The recipe.
    """
    return Recipe.loads(str(payload["text"]), base=_base(payload))


def _with(recipe: Recipe, name: str, value: object, base: Path | None) -> Recipe:
    """Return a copy of a recipe with one keyword set, and any pin on it removed.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.
    name : str
        The keyword.
    value : object
        Its value, as a recipe writes it.
    base : pathlib.Path or None
        The directory its data files resolve against.

    Returns
    -------
    solprop.recipe.Recipe
        The copy.
    """
    document = recipe.to_dict()
    pins = dict(document.get("pins", {}))
    if name in pins:
        del pins[name]
        if pins:
            document["pins"] = pins
        else:
            del document["pins"]
    copy = Recipe.from_dict(document, base=base)
    copy.set(name, value)
    return copy


def _mode_set(recipe: Recipe, z: float | None) -> dict[str, object]:
    """Return one recipe's mode set as plain data.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        A recipe on a guided route.
    z : float or None
        The position, m; ``None`` for the launch.

    Returns
    -------
    dict
        ``z`` and ``omega``, and per mode its ``label``, ``neff_minus_one``,
        ``beta1``, ``beta2``, ``beta3``, ``alpha``, ``gamma`` (or ``None``), ``aeff``,
        ``zdw`` (or ``None``), ``phase_mismatch`` to the launch soliton (or
        ``None``) and ``rdw``, its phase-matched wavelengths, in SI.
    """
    preview = recipe.modes(z)
    return {
        "z": float(preview.z),
        "omega": np.asarray(preview.omega),
        "modes": [
            {
                "label": mode.label,
                "neff_minus_one": np.asarray(mode.neff_minus_one),
                "beta1": np.asarray(mode.beta1),
                "beta2": np.asarray(mode.beta2),
                "beta3": np.asarray(mode.beta3),
                "alpha": np.asarray(mode.alpha),
                "gamma": None if mode.gamma is None else np.asarray(mode.gamma),
                "aeff": float(mode.aeff),
                "zdw": None if mode.zdw is None else float(mode.zdw),
                "phase_mismatch": (
                    None
                    if mode.phase_mismatch is None
                    else np.asarray(mode.phase_mismatch)
                ),
                "rdw": [float(root) for root in mode.rdw],
            }
            for mode in preview.modes
        ],
    }


def _pump(recipe: Recipe) -> float | None:
    """Return the recipe's pump wavelength, if it gives one.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.

    Returns
    -------
    float or None
        The ``wavelength`` keyword in m, or ``None`` where it is not a number (a
        recipe of ``pulses``, or one that leaves it to a rule).
    """
    value = recipe.keywords().get("wavelength")
    return float(value) if isinstance(value, int | float) else None


def modes(payload: Payload) -> dict[str, object]:
    """Return the mode set a recipe builds, or one per value of a swept keyword.

    Parameters
    ----------
    payload : Mapping
        ``text``, optionally ``base``, ``z`` (a length, in m or as a quantity;
        ``None`` for the launch) and
        ``sweep``: ``name``, a keyword, and ``values``, its values as a recipe
        writes them.

    Returns
    -------
    dict
        ``pump``, the pump wavelength in m or ``None``; ``sweep``, the swept
        keyword or ``None``; and ``sets``, one per value (one for no sweep), each
        ``value`` (the swept value, or ``None``) with the mode set
        (:func:`_mode_set`) or the ``error`` that value met.
    """
    recipe = _recipe(payload)
    z = payload.get("z")
    position = None if z is None else parse_quantity(z, Unit("m"), path="z")
    sweep = payload.get("sweep")
    name: str | None = None
    variants: list[tuple[object, Recipe | SolpropError]] = [(None, recipe)]
    if isinstance(sweep, Mapping) and sweep.get("values"):
        name = str(sweep["name"])
        values: Sequence[object] = sweep["values"]  # pyright: ignore[reportAssignmentType]
        variants = []
        for value in values:
            try:
                variants.append((value, _with(recipe, name, value, _base(payload))))
            except SolpropError as error:
                variants.append((value, error))
    sets: list[dict[str, object]] = []
    for value, variant in variants:
        entry: dict[str, object] = {"value": value}
        if isinstance(variant, SolpropError):
            entry["error"] = str(variant)
        else:
            try:
                entry.update(_mode_set(variant, position))
            except (SolpropError, TypeError, ValueError) as error:
                # A keyword missing from an unchecked sweep value reaches the
                # route's constructor as a TypeError.
                entry["error"] = str(error)
        sets.append(entry)
    pump: float | None = None
    try:
        pump = _pump(recipe.resolve())
    except SolpropError:
        pump = None
    return {"pump": pump, "sweep": name, "sets": sets}


def pin_targets(payload: Payload) -> dict[str, list[dict[str, object]]]:
    """Return the keywords a route can pin, and the targets each takes.

    Parameters
    ----------
    payload : Mapping
        ``route``.

    Returns
    -------
    dict
        By keyword, its targets: each ``target``, its ``stage`` (``"guide"`` or
        ``"pulse"``) and its ``required`` keys. Empty on a route without pins.

    Examples
    --------
    >>> sorted(pin_targets({"route": "capillary"}))
    ['energy', 'peak_power', 'pressure', 'radius']
    >>> pin_targets({"route": "free"})
    {}
    """
    route = str(payload["route"])
    if route not in GUIDE_PINS:
        return {}
    names = {"guide": tuple(GUIDE_PINS[route]), "pulse": PULSE_PINS}
    found: dict[str, list[dict[str, object]]] = {}
    for target, (stage, required, _optional) in TARGETS.items():
        for name in names[stage]:
            found.setdefault(name, []).append(
                {"target": target, "stage": stage, "required": list(required)}
            )
    return found


def pin_roots(payload: Payload) -> dict[str, object]:
    """Return every root of one pin's target (``Recipe.pin_roots``).

    Parameters
    ----------
    payload : Mapping
        ``text``, optionally ``base``, and ``path``, the pinned path.

    Returns
    -------
    dict
        ``roots`` (ascending, in the keyword's unit), ``search``, ``quantity`` (what
        ``attainable`` measures) and ``attainable`` (the range of it reached when no
        root is found, or ``None``).
    """
    found = _recipe(payload).pin_roots(str(payload["path"]))
    return {
        "roots": [float(root) for root in found.roots],
        "search": [float(end) for end in found.search],
        "quantity": found.quantity,
        "attainable": (
            None if found.attainable is None else [float(v) for v in found.attainable]
        ),
    }


_CHART_KEYWORDS = ("temperature", "model", "cladding", "materials")
"""The capillary's keywords that ``tools.pressure_zdw`` also takes, by name."""

CHART_POINTS = 400
"""The most pressures one chart solves (radii times wavelengths), about 15 s."""


def zdw_chart(payload: Payload) -> dict[str, object]:
    """Return the pressure that puts the zero-dispersion point at each wavelength.

    The design chart of a gas-filled capillary (design 7.4): for each wavelength,
    the pressure ``tools.pressure_zdw`` solves for, against the core radius, with
    the setup's fill (its gas, temperature, Marcatili model, cladding and
    materials). It is the zero-dispersion wavelength over the pressure and radius
    plane, drawn as its contours.

    Parameters
    ----------
    payload : Mapping
        ``text``, optionally ``base``, ``radii`` and ``wavelengths`` (lengths, in m
        or as quantities) and ``pressure_max`` (bar, default 100).

    Returns
    -------
    dict
        ``radius`` (m); ``curves``, per wavelength its ``zdw`` (m) and ``pressure``
        (bar per radius, ``None`` where no pressure up to the maximum gives it);
        ``setup``, the setup's ``radius`` (m) and ``pressure`` (bar, ``None`` for a
        gradient); and ``error``, the first reason a pressure was not found.

    Raises
    ------
    solprop.exceptions.SolpropError
        If the recipe does not resolve, or a radius or wavelength does not read.
    ValueError
        If the chart asks for more than :data:`CHART_POINTS` pressures.
    """
    keywords = _recipe(payload).resolve().keywords()
    length = Unit("m")
    radii = [parse_quantity(r, length, path="radii") for r in payload["radii"]]  # pyright: ignore[reportGeneralTypeIssues]
    wavelengths = [
        parse_quantity(w, length, path="wavelengths")
        for w in payload["wavelengths"]  # pyright: ignore[reportGeneralTypeIssues]
    ]
    if len(radii) * len(wavelengths) > CHART_POINTS:
        msg = f"a chart solves at most {CHART_POINTS} pressures"
        raise ValueError(msg)
    shared = {
        name: keywords[name]
        for name in _CHART_KEYWORDS
        if keywords.get(name) is not None
    }
    pressure_max = float(payload.get("pressure_max") or 100.0)  # pyright: ignore[reportArgumentType]
    error: str | None = None
    curves: list[dict[str, object]] = []
    for wavelength in wavelengths:
        pressures: list[float | None] = []
        for radius in radii:
            try:
                pressures.append(
                    pressure_zdw(
                        radius,
                        keywords["gas"],  # pyright: ignore[reportArgumentType]
                        wavelength,
                        pressure_max=pressure_max,
                        **shared,  # pyright: ignore[reportArgumentType]
                    )
                )
            except SolpropError as failure:
                pressures.append(None)
                error = error or str(failure)
        curves.append({"zdw": wavelength, "pressure": pressures})
    pressure = keywords.get("pressure")
    return {
        "radius": radii,
        "curves": curves,
        "setup": {
            "radius": float(keywords["radius"]),  # pyright: ignore[reportArgumentType]
            "pressure": float(pressure) if isinstance(pressure, int | float) else None,
        },
        "error": error,
    }


DESIGN_HANDLERS = {
    "modes": modes,
    "pin_roots": pin_roots,
    "pin_targets": pin_targets,
    "zdw_chart": zdw_chart,
}
"""The design requests, by kind."""
