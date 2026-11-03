# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's handlers: each answers one request with a public solprop call.

This module imports solprop and runs only in the kernel process (it is imported by
:func:`solprop_gui.service.kernel.serve`, never by the window). Each handler takes the
request's payload and returns plain data: numbers, strings, lists, dicts and NumPy
arrays. None computes physics of its own; where solprop offers no call for a number,
the number is not shown.
"""

import contextlib
import math
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from solprop import capabilities
from solprop.exceptions import SolpropError
from solprop.output import Output, open_file, resumability
from solprop.processing import channel_labels
from solprop.processing.fields import get_power, get_spectral_density
from solprop.recipe import Parameter, Recipe, parameters
from solprop.recipe.metadata import REQUIRED, per_unit_width
from solprop.recipe.quantity import Unit, format_quantity
from solprop.recipe.registry import REGISTRY
from solprop.tools import capillary_params, lambda_rdw_all

from solprop_gui.service.design import DESIGN_HANDLERS, pin_targets
from solprop_gui.service.figures import FIGURE_HANDLERS
from solprop_gui.service.reduce import Display, Reducer, stats_row
from solprop_gui.service.results import RESULTS_HANDLERS, band_of, spatial_plane

__all__ = ["HANDLERS"]

Payload = Mapping[str, object]


def _recipe(payload: Payload) -> Recipe:
    """Read the recipe a payload carries.

    Parameters
    ----------
    payload : Mapping
        ``text``, the recipe's TOML, and optionally ``base``, the directory its data
        files resolve against.

    Returns
    -------
    solprop.recipe.Recipe
        The recipe.
    """
    base = payload.get("base")
    return Recipe.loads(str(payload["text"]), base=Path(str(base)) if base else None)


def _plain(value: object) -> object:
    """Return a value as plain data, or its text where it is not.

    Parameters
    ----------
    value : object
        A default or a resolved value.

    Returns
    -------
    object
        Numbers, strings, booleans and ``None`` as they are; tuples and lists
        element by element; anything else as its ``repr``.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return repr(value)


def _default_text(entry: Parameter) -> str | None:
    """Return a keyword's default as a front end writes it.

    Parameters
    ----------
    entry : solprop.recipe.Parameter
        The keyword's metadata.

    Returns
    -------
    str or None
        The default as a quantity string in the display unit where it is a number
        with a unit, its text otherwise; ``None`` when the keyword is required or
        its default is ``None``.
    """
    default = entry.default
    if default is REQUIRED or default is None:
        return None
    unit = entry.forms[0].unit if entry.forms else None
    if (
        isinstance(default, int | float)
        and not isinstance(default, bool)
        and unit is not None
        and entry.display is not None
        and math.isfinite(default)
    ):
        return format_quantity(float(default), Unit(unit), entry.display)
    return str(_plain(default))


def _parameter(entry: Parameter) -> dict[str, object]:
    """Return one keyword's metadata as plain data.

    Parameters
    ----------
    entry : solprop.recipe.Parameter
        The keyword's metadata.

    Returns
    -------
    dict
        Its fields, the forms as a list of dicts, and the default as written
        (``default``) and as plain data (``default_value``, in the keyword's unit).
    """
    return {
        "name": entry.name,
        "group": entry.group,
        "level": entry.level,
        "summary": entry.summary,
        "anchor": entry.anchor,
        "display": entry.display,
        "none": entry.none,
        "required": entry.default is REQUIRED,
        "default": _default_text(entry),
        "default_value": None if entry.default is REQUIRED else _plain(entry.default),
        "resolved_by_rule": entry.resolver is not None,
        "forms": [
            {
                "kind": form.kind,
                "unit": form.unit,
                "elements": form.elements,
                "minimum": form.minimum,
                "maximum": form.maximum,
                "exclusive": form.exclusive,
                "choices": list(form.choices),
                "types": list(form.types),
                "length": form.length,
                "recipe": form.recipe,
            }
            for form in entry.forms
        ],
    }


def examples(payload: Payload) -> list[dict[str, object]]:  # noqa: ARG001 - none
    """List the example recipes solprop ships.

    Parameters
    ----------
    payload : Mapping
        Unused.

    Returns
    -------
    list of dict
        ``name``, ``title``, ``route`` and ``build`` (its ``[build]`` table, as the
        recipe writes it, for the window to tell its geometry class) of each.
    """
    out: list[dict[str, object]] = []
    for name in Recipe.examples():
        recipe = Recipe.example(name)
        out.append(
            {
                "name": name,
                "title": recipe.title,
                "route": recipe.route,
                "build": recipe.to_dict().get("build", {}),
            }
        )
    return out


def example(payload: Payload) -> str:
    """Return one example recipe's text.

    Parameters
    ----------
    payload : Mapping
        ``name``.

    Returns
    -------
    str
        Its TOML, comments included.
    """
    return Recipe.example(str(payload["name"])).to_text()


def describe(payload: Payload) -> dict[str, object]:
    """Describe a route: its keywords' metadata, choices and combination rules.

    Parameters
    ----------
    payload : Mapping
        ``route``.

    Returns
    -------
    dict
        ``parameters`` (the route's keywords in order), ``execution`` (the
        ``[execution]`` table's), ``choices``, ``rules`` and ``types``, the keywords
        of every registered type the route's keywords can name, nested ones included
        (:func:`_type_catalog`); each keyword carries ``pins``, the design targets it
        can be pinned to (empty for most).
    """
    route = str(payload["route"])
    caps = capabilities.route(route)
    targets = pin_targets({"route": route})
    keywords = [_parameter(entry) for entry in parameters(route).values()]
    for keyword in keywords:
        keyword["pins"] = targets.get(str(keyword["name"]), [])
    return {
        "route": route,
        "types": _type_catalog(keywords),
        "parameters": keywords,
        "execution": [_parameter(entry) for entry in parameters("execution").values()],
        "choices": {name: list(values) for name, values in caps.choices.items()},
        "rules": [
            {"id": rule.id, "keywords": list(rule.keywords), "reason": rule.reason}
            for rule in caps.rules
        ],
    }


def _type_catalog(
    keywords: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Return the keywords of every registered type a set of keywords can name.

    Parameters
    ----------
    keywords : list of dict
        Keywords as :func:`_parameter` describes them.

    Returns
    -------
    dict
        By type, its keywords described alike, for every type a recipe can write
        that the keywords or those types' own keywords name.
    """
    catalog: dict[str, list[dict[str, object]]] = {}
    waiting = [keywords]
    while waiting:
        for keyword in waiting.pop():
            for form in keyword["forms"]:  # pyright: ignore[reportGeneralTypeIssues]
                if form["kind"] != "object" or not form["recipe"]:
                    continue
                for name in form["types"]:
                    if name in catalog or name not in REGISTRY:
                        continue
                    entries = [_parameter(e) for e in parameters(name).values()]
                    catalog[name] = entries
                    waiting.append(entries)
    return catalog


def check(payload: Payload) -> list[dict[str, object]]:
    """Check a recipe (``Recipe.check``).

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    list of dict
        Each issue's ``message``, ``parameter`` (a path, a list of paths or
        ``None``), ``severity`` and ``rule``; a recipe that cannot be read gives one
        error at no parameter.
    """
    try:
        issues = _recipe(payload).check()
    except SolpropError as error:
        return [
            {
                "message": str(error),
                "parameter": getattr(error, "parameter", None),
                "severity": "error",
                "rule": getattr(error, "rule", None),
            }
        ]
    return [
        {
            "message": issue.message,
            "parameter": (
                list(issue.parameter)
                if isinstance(issue.parameter, tuple)
                else issue.parameter
            ),
            "severity": issue.severity,
            "rule": issue.rule,
        }
        for issue in issues
    ]


def summarise(payload: Payload) -> dict[str, object]:
    """Return the summary dock's numbers for a recipe.

    The input pulse's energy and peak power (``Recipe.preview_input``), the defaults
    resolved by rule, and on the capillary route the design numbers at the entrance
    (``solprop.tools.capillary_params``) where the recipe gives what they need.

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``energy`` and ``peak_power`` (J, W), ``resolved`` (by keyword: the value
        and the rule's sentence), ``design`` (by name, SI; empty when not
        available) with ``design_error`` saying why it is not, ``pinned`` (by
        pinned path, the solved value in its keyword's unit), and ``per_width``,
        whether the energy and peak power are per metre of y (a slab mode set).
    """
    authored = _recipe(payload)
    recipe = authored.resolve()
    pinned = {
        path: _plain(recipe.get(path)) for path in authored.to_dict().get("pins", {})
    }
    preview = recipe.preview_input()
    build = recipe.to_dict()["build"]
    resolved: dict[str, object] = {}
    for name, entry in parameters(recipe.route).items():
        if entry.resolver is not None and name not in build:
            resolution = entry.resolver(build)
            resolved[name] = [_plain(resolution.value), resolution.rule]
    design: dict[str, float] = {}
    design_error: str | None = None
    if recipe.route == "capillary":
        try:
            design = _capillary_design(recipe, float(preview.energy))
        except (SolpropError, KeyError, TypeError, ValueError) as error:
            design_error = str(error)
    return {
        "energy": float(preview.energy),
        "peak_power": float(preview.peak_power),
        "resolved": resolved,
        "design": design,
        "design_error": design_error,
        "pinned": pinned,
        "per_width": per_unit_width(recipe.route, build),
    }


_SHARED = ("shape", "temperature", "model", "loss", "cladding", "materials")
"""The capillary's keywords that ``tools.capillary_params`` also takes, by name."""


def _capillary_design(recipe: Recipe, energy: float) -> dict[str, float]:
    """Return a capillary recipe's design numbers at the entrance.

    The keywords the tool shares with the route are passed on, with the route's
    default where the recipe gives none (the pulse shape's differs: Gaussian).

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        A resolved capillary recipe.
    energy : float
        The pulse energy launched, J (given, or from the peak power).

    Returns
    -------
    dict
        ``soliton_order``, ``beta2`` (s^2/m), ``gamma`` (1/(W m)), ``n2`` (m^2/W),
        ``peak_intensity`` (W/m^2), ``pressure`` (bar), ``zdw`` (m),
        ``fission_length`` (m), ``critical_power`` (W) and, where the pump is
        anomalous and a wave is phase-matched, ``rdw`` (m, the shortest such
        wavelength, ``tools.lambda_rdw_all``).

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
    params = capillary_params(
        float(keywords["radius"]),  # pyright: ignore[reportArgumentType]
        keywords["gas"],  # pyright: ignore[reportArgumentType]
        None if pressure is None else float(pressure),  # pyright: ignore[reportArgumentType]
        energy=energy,
        fwhm=float(keywords["fwhm"]),  # pyright: ignore[reportArgumentType]
        wavelength=float(keywords["wavelength"]),  # pyright: ignore[reportArgumentType]
        **shared,  # pyright: ignore[reportArgumentType]
    )
    numbers = {
        "soliton_order": float(params.soliton_order),
        "beta2": float(params.beta2),
        "gamma": float(params.gamma),
        "n2": float(params.n2),
        "peak_intensity": float(params.peak_intensity),
        "pressure": float(params.pressure),
        "zdw": float(params.zdw),
        "fission_length": float(params.fission_length),
        "critical_power": float(params.critical_power),
    }
    # A pump in normal dispersion launches no soliton, so no wave is phase-matched.
    roots = (
        lambda_rdw_all(
            params.mode,
            params.wavelength,
            peak_power=params.peak_power,
            gamma=params.gamma,
            fwhm=params.fwhm,
            shape=params.shape,
        )
        if params.beta2 < 0
        else ()
    )
    for root in roots[:1]:  # the shortest
        numbers["rdw"] = float(root)
    return numbers


def design(payload: Payload) -> dict[str, object]:
    """Solve a design that sets several values at once, without keeping its pins.

    The design tool's request (design 7.5): the recipe is pinned as asked (each
    pinned keyword's value replaced by its pin; a guide pin and a pulse pin solve in
    solprop's two stages, as ``tools.capillary_design`` does) and resolved, and the
    solved values come back for the tool to write as literals.

    Parameters
    ----------
    payload : Mapping
        ``text``, optionally ``base``, and ``pins``: by keyword, the ``[pins]`` entry
        to solve it from.

    Returns
    -------
    dict
        ``values``, by keyword, the solved value in its unit; and the summary's
        ``design`` numbers at the solved values, with ``design_error`` saying why
        they are missing.

    Raises
    ------
    solprop.exceptions.SolpropError
        If a pin has no root, or the pinned recipe does not read.
    """
    document = _recipe(payload).to_dict()
    pins: Mapping[str, object] = payload["pins"]  # pyright: ignore[reportAssignmentType]
    build = document["build"]
    for name in pins:
        build.pop(name, None)
    document["pins"] = {**document.get("pins", {}), **pins}
    base = payload.get("base")
    pinned = Recipe.from_dict(document, base=Path(str(base)) if base else None)
    recipe = pinned.resolve()
    values = {name: _plain(recipe.get(name)) for name in pins}
    numbers: dict[str, float] = {}
    error: str | None = None
    if recipe.route == "capillary":
        try:
            numbers = _capillary_design(recipe, float(recipe.preview_input().energy))
        except (SolpropError, KeyError, TypeError, ValueError) as failure:
            error = str(failure)
    return {"values": values, "design": numbers, "design_error": error}


def preview_input(payload: Payload) -> dict[str, object]:
    """Return the input pulse as launched (``Recipe.preview_input``).

    Parameters
    ----------
    payload : Mapping
        ``text`` and optionally ``base``.

    Returns
    -------
    dict
        ``time`` (s), ``power`` (W), ``omega`` (rad/s), ``spectral_power`` (the
        spectrum's squared magnitude summed over modes), ``energy`` (J) and
        ``peak_power`` (W).
    """
    preview = _recipe(payload).preview_input()
    spectrum = np.abs(np.asarray(preview.spectrum)) ** 2
    if spectrum.ndim > 1:
        spectrum = spectrum.reshape(-1, spectrum.shape[-1]).sum(axis=0)
    return {
        "time": np.asarray(preview.time),
        "power": np.asarray(preview.power),
        "omega": np.asarray(preview.omega),
        "spectral_power": spectrum,
        "energy": float(preview.energy),
        "peak_power": float(preview.peak_power),
    }


def estimate(payload: Payload) -> dict[str, object]:
    """Estimate a run's cost (``Simulation.estimate``).

    It builds the backend and times trial steps: seconds, so it is asked for, not
    run on every edit.

    Parameters
    ----------
    payload : Mapping
        ``text``, optionally ``base`` and ``memory_budget``.

    Returns
    -------
    dict
        The estimate's fields.
    """
    budget = payload.get("memory_budget")
    result = (
        _recipe(payload)
        .build()
        .estimate(
            memory_budget=budget  # pyright: ignore[reportArgumentType]
        )
    )
    return {
        "shape": list(result.shape),
        "n_planes": result.n_planes,
        "memory": result.memory,
        "admitted": result.admitted,
        "refusal": result.refusal,
        "output_bytes": result.output_bytes,
        "backend": result.backend,
        "threads": _plain(result.threads),
        "setup": result.setup,
        "rhs": result.rhs,
        "calls_per_step": result.calls_per_step,
        "total_calls": result.total_calls,
    }


def to_python(payload: Payload) -> str:
    """Return the plain Python script that makes the same run (``Recipe.to_python``).

    Parameters
    ----------
    payload : Mapping
        ``text``, optionally ``base`` and ``output``, the file the script writes.

    Returns
    -------
    str
        The script.
    """
    output = payload.get("output")
    return _recipe(payload).to_python(None if output is None else str(output))


def check_resume(payload: Payload) -> dict[str, object]:
    """Say whether a results file can be continued (``solprop.output.resumability``).

    Parameters
    ----------
    payload : Mapping
        ``path``, and ``gate``: whether to rebuild the run and compare its record.

    Returns
    -------
    dict
        ``resumable``, ``z``, ``mode``, ``reason``, ``recipe``, ``rebuildable``,
        ``differences`` (the gate's comparison as text, when asked) and
        ``gate_error``.
    """
    found = resumability(str(payload["path"]), gate=bool(payload.get("gate", False)))
    return {
        "resumable": found.resumable,
        "z": found.z,
        "mode": found.mode,
        "reason": found.reason,
        "recipe": found.recipe,
        "rebuildable": found.rebuildable,
        "differences": None if found.comparison is None else str(found.comparison),
        "gate_error": found.gate_error,
    }


def describe_output(payload: Payload) -> dict[str, object]:
    """Say what a results file can give (``solprop.capabilities.output``).

    Parameters
    ----------
    payload : Mapping
        ``path``.

    Returns
    -------
    dict
        ``basis``, ``simulation_kind``, ``field_planes``, ``statistics_planes``,
        ``refused``, ``recipe``, ``outcome`` (the run's status, position and
        message), ``z`` (every plane's position) and ``labels`` (the modes).
    """
    path = str(payload["path"])
    caps = capabilities.output(path)
    with open_file(path) as handler:
        outcome = handler.outcome
        z = [float(v) for v in np.asarray(handler["z"])]
        try:
            labels = [label.display_label for label in channel_labels(handler)]
        except SolpropError:
            labels = []
    return {
        "z": z,
        "labels": labels,
        "basis": caps.basis,
        "simulation_kind": caps.simulation_kind,
        "field_planes": list(caps.field_planes),
        "statistics_planes": list(caps.statistics_planes),
        "refused": dict(caps.refused),
        "recipe": caps.recipe,
        "outcome": None
        if outcome is None
        else {"status": outcome.status, "z": outcome.z, "message": outcome.message},
    }


def recipe_of(payload: Payload) -> str:
    """Return the recipe a results file records (``Recipe.from_output``).

    Parameters
    ----------
    payload : Mapping
        ``path``, and ``resolved``: the resolved recipe rather than the authored.

    Returns
    -------
    str
        Its TOML.
    """
    recipe = Recipe.from_output(
        str(payload["path"]), resolved=bool(payload.get("resolved", False))
    )
    return recipe.to_text()


def overview(payload: Payload) -> dict[str, object]:
    """Reduce a results file's field planes for the Overview layout.

    The planes' spectral density and power come from :mod:`solprop.processing`, the
    statistics from the file; the reduction is :class:`~.reduce.Reducer`'s.

    Parameters
    ----------
    payload : Mapping
        ``path``, and optionally ``wavelength_range`` and ``time_range`` (pairs,
        SI), ``axis`` (``"wavelength"`` or ``"f"``) and ``bandpass`` (a wavelength
        band, m, the power is computed within).

    Returns
    -------
    dict
        ``axes`` (the ``spectral`` and ``time`` bin centres, and the spectral
        ``axis``) and ``rows``, one per field plane: ``index``, ``z``, ``density``,
        ``power`` and ``stats``.
    """
    path = str(payload["path"])
    display = Display(
        wavelength_range=_pair(payload.get("wavelength_range")),
        time_range=_pair(payload.get("time_range")),
        axis=str(payload.get("axis") or "wavelength"),
        bandpass=band_of(payload),
    )
    caps = capabilities.output(path)
    if caps.recipe:
        with contextlib.suppress(SolpropError):  # else the file's own band
            display = display.within(Recipe.from_output(path).keywords())
    reducer = Reducer(display)
    rows: list[dict[str, object]] = []
    planes = caps.field_planes
    with open_file(path) as handler:
        z = np.asarray(handler["z"])
        for index in planes:
            if caps.basis == "spatial":
                # A beam's columns are field samples: solprop's spatial readings.
                beam = spatial_plane(
                    path, handler, index, axis=display.axis, bandpass=display.bandpass
                )
                spectrum, flux = beam["spectrum"], beam["power"]
                if "error" in spectrum:  # pyright: ignore[reportOperatorIssue]
                    raise SolpropError(str(spectrum["error"]))  # pyright: ignore[reportIndexIssue]
                x, density = spectrum["x"], spectrum["y"]  # pyright: ignore[reportIndexIssue]
                t, power = flux["t"], flux["y"]  # pyright: ignore[reportIndexIssue]
            else:
                x, density = get_spectral_density(
                    handler,
                    display.axis,  # pyright: ignore[reportArgumentType]
                    plane=index,
                    modes="sum",
                )
                t, power = get_power(
                    handler, plane=index, modes="sum", bandpass=display.bandpass
                )
            reduced_density, reduced_power = reducer.reduce(x, density, t, power)
            rows.append(
                {
                    "index": int(index),
                    "z": float(z[index]),
                    "density": reduced_density,
                    "power": reduced_power,
                    "stats": stats_row(_stats_at(handler, index)),
                }
            )
    return {"axes": reducer.axes() if rows else {}, "rows": rows}


def _pair(value: object) -> tuple[float, float] | None:
    """Return a two-element sequence as a pair of floats.

    Parameters
    ----------
    value : object
        A pair, or ``None``.

    Returns
    -------
    tuple of float or None
        The pair.
    """
    if value is None:
        return None
    low, high = value  # pyright: ignore[reportGeneralTypeIssues]
    return float(low), float(high)


def _stats_at(handler: Output, index: int) -> dict[str, object]:
    """Return a file's statistics row at one plane.

    Parameters
    ----------
    handler : solprop.output.Output
        The open file.
    index : int
        The plane.

    Returns
    -------
    dict
        Each recorded statistic at that plane, by name.
    """
    return {name: handler[f"stats/{name}"][index] for name in handler.stat_names}


HANDLERS: dict[str, Callable[[Payload], object]] = {
    "check": check,
    "check_resume": check_resume,
    "describe": describe,
    "describe_output": describe_output,
    "design": design,
    "estimate": estimate,
    "example": example,
    "examples": examples,
    "overview": overview,
    "preview_input": preview_input,
    "recipe_of": recipe_of,
    "summarise": summarise,
    "to_python": to_python,
    **DESIGN_HANDLERS,
    **FIGURE_HANDLERS,
    **RESULTS_HANDLERS,
}
"""The handlers, by request kind."""
