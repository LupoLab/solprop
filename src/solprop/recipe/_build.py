# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""From recipe values to the objects a route takes: the registry and the decoder.

A recipe names an object by a ``type`` key, which must be an entry of the fixed
:data:`REGISTRY` (SEP 41 section 5.1): a **named factory**, a constructor, class
method or module function, with the units of its keywords declared and any **build
context** it receives declared too. Format 1 defines one context, the route's time
grid, so ``absorber = {type = "absorber"}`` builds
``solprop.boundaries.absorber(grid)``.
A recipe never imports a module or evaluates code: a type outside the registry is
refused, and the Python export is the way out.

:func:`decode` turns one value of a recipe into the Python value a keyword takes:
quantities into numbers in the keyword's unit, ``{re, im}`` into a complex number,
``{file, sha256}`` into an array read from a checked file, a table with a ``type`` into
an object, and ``"none"`` into ``None`` where the keyword has that spelling.
"""

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np

from solprop.boundaries import absorber
from solprop.grid import EnvelopeGrid, FieldGrid, TimeGrid
from solprop.modes.antiresonant import TubeLattice
from solprop.modes.capillary import CoherentWall
from solprop.monitor import BoundaryCheck
from solprop.physdata.mixtures import Mixture
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import Constant, Linear, PiecewiseLinear, PressureGradient
from solprop.pulses import GaussianPulse, SechPulse, load_spectrum
from solprop.recipe._errors import RecipeError
from solprop.recipe._format import (
    MAX_ARRAY_LENGTH,
    NONE,
    decode_complex,
    decode_data_file,
    load_array,
    read_data_file,
    unknown_key,
)
from solprop.recipe._metadata import Form, Parameter
from solprop.recipe._quantity import Unit, beta_unit, parse_quantity, phase_unit
from solprop.stats import StatsSpec
from solprop.stepper import LowStorageRungeKutta, Multistep, RungeKutta
from solprop.stepper.schedule import FixedSchedule

__all__ = [
    "REGISTRY",
    "BuildContext",
    "Factory",
    "decode",
    "decode_keyword",
    "unit_of",
]

Context = Literal["grid"]
"""What a factory may be given besides its keywords: format 1 defines the grid."""

UnitSpec = str | Callable[[int], Unit] | None
"""A keyword's unit: one unit string, a rule by element index, or none."""


@dataclass(frozen=True)
class BuildContext:
    """What decoding a recipe may read besides the recipe's own values.

    Attributes
    ----------
    base : pathlib.Path or None
        The recipe's directory, against which data files resolve; ``None`` for a
        recipe from text, which then may hold no data file.
    allow_outside : bool
        Whether a data file may lie outside `base`.
    grid : solprop.grid.TimeGrid or None
        The route's time grid, for a factory that declares it as its context;
        ``None`` until the route has built one.

    Examples
    --------
    >>> from solprop.recipe._build import BuildContext
    >>> BuildContext().grid is None
    True
    """

    base: Path | None = None
    allow_outside: bool = False
    grid: TimeGrid | None = None


@dataclass(frozen=True)
class Factory:
    """A registered type: what a ``type`` names, and how its keywords are read.

    Attributes
    ----------
    name : str
        The ``type`` a recipe writes, as ``"GaussianPulse"`` or
        ``"EnvelopeGrid.from_limits"``.
    target : callable
        The constructor, class method or function it calls.
    units : collections.abc.Mapping
        The unit of each keyword that has one (:data:`UnitSpec`); a keyword not
        listed is read as it stands.
    context : tuple of str
        What it receives besides its keywords, passed first and positionally.
    data_file : str or None
        The keyword a ``{file, sha256}`` reference fills with the file's text, for a
        loader such as ``load_spectrum``; ``None`` otherwise.

    Examples
    --------
    >>> from solprop.recipe._build import REGISTRY
    >>> REGISTRY["GaussianPulse"].units["fwhm"]
    's'
    """

    name: str
    target: Callable[..., object]
    units: Mapping[str, UnitSpec] = field(default_factory=dict)
    context: tuple[Context, ...] = ()
    data_file: str | None = None


def _pressure_pair(index: int) -> Unit:
    """Return the unit of an element of a ``(z, p)`` pair: metres, then bar.

    Parameters
    ----------
    index : int
        The element's position in its pair.

    Returns
    -------
    Unit
        ``m`` for the position, ``bar`` for the pressure.
    """
    return Unit("m") if index == 0 else Unit("bar")


_PULSE_UNITS: Final[Mapping[str, UnitSpec]] = {
    "wavelength": "m",
    "energy": "J",
    "peak_power": "W",
    "phase": phase_unit,
    "t0": "s",
    "fwhm": "s",
    "width": "s",
}
_GRID_UNITS: Final[Mapping[str, UnitSpec]] = {
    "length": "m",
    "reference_wavelength": "m",
    "wavelength_limits": "m",
    "trange": "s",
    "dt": "s",
}
_GRADIENT_UNITS: Final[Mapping[str, UnitSpec]] = {
    "length": "m",
    "start_bar": "bar",
    "end_bar": "bar",
    "floor": "bar",
    "radius": "m",
    "positions": "m",
    "pressures": "bar",
}
_FILL_UNITS: Final[Mapping[str, UnitSpec]] = {
    "pressure": "bar",
    "temperature": "K",
    "floor": "bar",
    "length": "m",
    "radius": "m",
}

REGISTRY: Final[Mapping[str, Factory]] = {
    factory.name: factory
    for factory in (
        Factory("GaussianPulse", GaussianPulse, _PULSE_UNITS),
        Factory("SechPulse", SechPulse, _PULSE_UNITS),
        Factory("load_spectrum", load_spectrum, {"energy": "J"}, data_file="path"),
        Factory(
            "PressureGradient.two_point", PressureGradient.two_point, _GRADIENT_UNITS
        ),
        Factory(
            "PressureGradient.from_stages",
            PressureGradient.from_stages,
            _GRADIENT_UNITS,
        ),
        Factory("Constant", Constant, {"span": "m"}),
        Factory("Linear", Linear, {"length": "m"}),
        Factory("PiecewiseLinear", PiecewiseLinear, {"positions": "m"}),
        Factory("Mixture.fill", Mixture.fill, _FILL_UNITS),
        Factory(
            "Mixture.partial_pressures",
            Mixture.partial_pressures,
            {"temperature": "K"},
        ),
        Factory("Mixture.air", Mixture.air, _FILL_UNITS),
        Factory("EnvelopeGrid.from_limits", EnvelopeGrid.from_limits, _GRID_UNITS),
        Factory("FieldGrid.from_limits", FieldGrid.from_limits, _GRID_UNITS),
        Factory("StatsSpec", StatsSpec),
        Factory("BoundaryCheck", BoundaryCheck),
        Factory(
            "FixedSchedule.uniform",
            FixedSchedule.uniform,
            {
                "z_start": "m",
                "z_end": "m",
                "dz": "m",
                "save_planes": "m",
                "hard_kinks": "m",
            },
        ),
        Factory("Multistep", Multistep),
        Factory("RungeKutta", RungeKutta),
        Factory("LowStorageRungeKutta", LowStorageRungeKutta),
        Factory("absorber", absorber, {"boundary_length": "s"}, context=("grid",)),
        Factory(
            "TubeLattice", TubeLattice, {"wall_thickness": "m", "tube_radius": "m"}
        ),
        Factory("CoherentWall", CoherentWall),
        Factory("MaterialSelection", MaterialSelection, {"carrier_wavelength": "m"}),
    )
}
"""The types a recipe may name, keyed by the ``type`` it writes (SEP 41 section 5.1).

The documented user-facing types of the ``gnlse`` and ``capillary`` routes. Every
keyword of an entry is part of recipe format 1, so the registry starts small: adding an
entry later is additive, removing one is not. ``RungeKutta`` takes no ``tableau`` in a
recipe, whose default is the only one the interfaces document.
"""

_ELEMENTS: Final[Mapping[str, Callable[[int], Unit]]] = {
    "phases": phase_unit,
    "betas": beta_unit,
    "z_pressure": _pressure_pair,
}


def _number(value: object, unit: UnitSpec, index: int, path: str) -> float | object:
    """Read one number in its unit, or return a non-numeric value as it stands.

    Parameters
    ----------
    value : object
        The recipe value.
    unit : str, callable or None
        The unit, or the rule giving it by element.
    index : int
        The element's position, for a rule.
    path : str
        The value's path, for errors.

    Returns
    -------
    float or object
        The number in the unit, or `value` unchanged when it has no unit or is not a
        number or quantity string.
    """
    if unit is None or isinstance(value, bool):
        return value
    if not isinstance(value, int | float | str):
        return value
    resolved = unit(index) if callable(unit) else Unit(unit)
    return parse_quantity(value, resolved, path=path)


def _array(
    value: Sequence[object], unit: UnitSpec, context: BuildContext, path: str
) -> list[object]:
    """Read a list: each element in its unit, a nested list as a row.

    Parameters
    ----------
    value : collections.abc.Sequence
        The list.
    unit : str, callable or None
        The unit of each element, or the rule by element index.
    context : BuildContext
        For tables inside the list.
    path : str
        The list's path.

    Returns
    -------
    list
        The decoded elements.

    Raises
    ------
    RecipeError
        If the list is longer than :data:`~solprop.recipe._format.MAX_ARRAY_LENGTH`.
    """
    if len(value) > MAX_ARRAY_LENGTH:
        msg = (
            f"a list of {len(value)} values is longer than the {MAX_ARRAY_LENGTH} a "
            "recipe holds inline; refer to a data file: {file = ..., sha256 = ...}"
        )
        raise RecipeError(msg, path=path)
    out: list[object] = []
    for index, element in enumerate(value):
        where = f"{path}[{index}]"
        if isinstance(element, Mapping):
            out.append(decode(element, unit, context, where))
        elif isinstance(element, list):
            # A row of a pair form: its elements take the rule by position.
            out.append(_array(element, unit, context, where))
        else:
            out.append(_number(element, unit, index, where))
    return out


def decode(value: object, unit: UnitSpec, context: BuildContext, path: str) -> object:
    """Turn one recipe value into the Python value a keyword takes.

    Parameters
    ----------
    value : object
        The value as TOML read it.
    unit : str, callable or None
        The keyword's unit (:data:`UnitSpec`).
    context : BuildContext
        The recipe's directory and the route's grid.
    path : str
        The value's parameter path, for errors.

    Returns
    -------
    object
        A number in the keyword's unit, a complex number, an array, an object, a
        list of these, or the value as it stands.

    Examples
    --------
    >>> from solprop.recipe._build import BuildContext, decode
    >>> decode("800 nm", "m", BuildContext(), "wavelength")
    8e-07
    >>> decode({"re": 1.45, "im": 0.0}, None, BuildContext(), "cladding")
    (1.45+0j)
    """
    if isinstance(value, Mapping):
        if "type" in value:
            return build_object(value, context, path)
        if set(value) == {"re", "im"}:
            return decode_complex(value, path=path)
        if set(value) == {"file", "sha256"}:
            return _data_array(value, context, path)
        return {
            key: decode(item, None, context, f'{path}["{key}"]')
            for key, item in value.items()
        }
    if isinstance(value, list):
        return _array(value, unit, context, path)
    return _number(value, unit, 0, path)


def _data_array(
    value: Mapping[str, object], context: BuildContext, path: str
) -> np.ndarray:
    """Read an array from a data file a recipe refers to.

    Parameters
    ----------
    value : collections.abc.Mapping
        The ``{file, sha256}`` reference.
    context : BuildContext
        The recipe's directory.
    path : str
        The value's path.

    Returns
    -------
    numpy.ndarray
        The array.

    Raises
    ------
    RecipeError
        If the recipe has no directory to resolve the file against.
    """
    reference = decode_data_file(value, path=path)
    if context.base is None:
        msg = "a recipe read from text holds no data file; load it from its file"
        raise RecipeError(msg, path=path)
    return load_array(
        reference, context.base, allow_outside=context.allow_outside, path=path
    )


def build_object(
    table: Mapping[str, object], context: BuildContext, path: str
) -> object:
    """Build the object a table with a ``type`` key names.

    Parameters
    ----------
    table : collections.abc.Mapping
        The table: ``type``, the factory's keywords and, for a loader, ``file`` and
        ``sha256``.
    context : BuildContext
        The recipe's directory and the route's grid.
    path : str
        The table's path.

    Returns
    -------
    object
        What the factory returns.

    Raises
    ------
    RecipeError
        If the type is not registered, a keyword is not the factory's, or the factory
        needs a grid that is not yet built.

    Examples
    --------
    >>> from solprop.recipe._build import BuildContext, build_object
    >>> pulse = build_object(
    ...     {"type": "GaussianPulse", "wavelength": "800 nm", "fwhm": "30 fs",
    ...      "energy": "1 mJ"},
    ...     BuildContext(),
    ...     "pulses[0]",
    ... )
    >>> pulse.fwhm
    3e-14
    """
    name = table["type"]
    factory = REGISTRY.get(name) if isinstance(name, str) else None
    if factory is None:
        raise unknown_key(
            str(name), sorted(REGISTRY), where="the registered types", path=path
        )
    keywords = {key: item for key, item in table.items() if key != "type"}
    positional: list[object] = []
    if "grid" in factory.context:
        if context.grid is None:
            msg = f"{factory.name} is built on the route's grid, not built yet"
            raise RecipeError(msg, path=path)
        positional.append(context.grid)
    if factory.data_file is not None:
        reference = decode_data_file(
            {
                "file": keywords.pop("file", None),
                "sha256": keywords.pop("sha256", None),
            },
            path=path,
        )
        if context.base is None:
            msg = "a recipe read from text holds no data file; load it from its file"
            raise RecipeError(msg, path=path)
        data = read_data_file(
            reference, context.base, allow_outside=context.allow_outside, path=path
        )
        keywords[factory.data_file] = io.StringIO(data.decode("utf-8-sig"))
    accepted = _keywords(factory)
    arguments: dict[str, object] = {}
    for key, item in keywords.items():
        if key == factory.data_file:
            arguments[key] = item
            continue
        if key not in accepted:
            raise unknown_key(
                key,
                sorted(accepted),
                where=f"the keywords of {factory.name}",
                path=path,
            )
        arguments[key] = decode(item, factory.units.get(key), context, f"{path}.{key}")
    try:
        return factory.target(*positional, **arguments)
    except TypeError as error:
        raise RecipeError(f"{factory.name}: {error}", path=path) from error


def _keywords(factory: Factory) -> frozenset[str]:
    """Return the keywords a factory takes from a recipe.

    Parameters
    ----------
    factory : Factory
        The entry.

    Returns
    -------
    frozenset of str
        Its signature's names, less its context and its data-file keyword.
    """
    import inspect  # noqa: PLC0415 - only here

    names = set(inspect.signature(factory.target).parameters)
    names -= set(factory.context)
    names.discard("self")
    if factory.data_file is not None:
        names.discard(factory.data_file)
    if factory.name == "RungeKutta":
        names.discard("tableau")
    return frozenset(names)


def unit_of(entry: Parameter, value: object) -> UnitSpec:
    """Return the unit of the form a keyword's value takes.

    Parameters
    ----------
    entry : Parameter
        The keyword's metadata.
    value : object
        The value as TOML read it.

    Returns
    -------
    str, callable or None
        The unit of the matching form, or the element rule of an array form.
    """
    forms: Sequence[Form] = entry.forms
    if isinstance(value, list):
        nested = bool(value) and isinstance(value[0], list)
        for form in forms:
            if form.kind in ("pair", "array"):
                if form.elements is not None:
                    if nested == (form.elements == "z_pressure"):
                        return _ELEMENTS[form.elements]
                    continue
                if not nested:
                    return form.unit
        return None
    for form in forms:
        if form.kind == "number" and form.unit is not None:
            return form.unit
    return None


def decode_keyword(entry: Parameter, value: object, context: BuildContext) -> object:
    """Turn the value of one build keyword into what the constructor takes.

    Parameters
    ----------
    entry : Parameter
        The keyword's metadata.
    value : object
        The value as TOML read it.
    context : BuildContext
        The recipe's directory and the route's grid.

    Returns
    -------
    object
        The decoded value; ``None`` for ``"none"`` where the keyword has the spelling.

    Raises
    ------
    RecipeError
        If the keyword's forms cannot hold the value in a recipe.

    Examples
    --------
    >>> from solprop.recipe._build import BuildContext, decode_keyword
    >>> from solprop.recipe._metadata import parameters
    >>> table = parameters("capillary")
    >>> decode_keyword(table["pressure"], ["0 bar", "1.2 bar"], BuildContext())
    [0.0, 1.2]
    >>> decode_keyword(table["noise"], "none", BuildContext()) is None
    True
    """
    if entry.none and value == NONE:
        return None
    if isinstance(value, str) and any(
        form.kind == "choice" and value in form.choices for form in entry.forms
    ):
        return value
    kinds = {form.kind for form in entry.forms}
    if isinstance(value, str) and "string" in kinds and "number" not in kinds:
        return value
    if not any(form.recipe for form in entry.forms):
        msg = (
            f"{entry.name} takes a Python object or function that a recipe cannot "
            "hold; use the Python export (Recipe.to_python)"
        )
        raise RecipeError(msg, path=entry.name)
    return decode(value, unit_of(entry, value), context, entry.name)
