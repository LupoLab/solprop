# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""From recipe values to the objects a route takes: the registry and the decoder.

A recipe names an object by a ``type`` key, which must be an entry of the fixed
:data:`REGISTRY` (SEP 41 section 5.1): a **named factory**, a constructor, class
method or module function, with the units of its keywords declared and any **build
context** it receives declared too. Format 1 defines one context, the route's time
grid, so ``absorber = {type = "absorber"}`` builds
``solprop.boundaries.absorber(grid)``; on the free-space routes it is the ``grid``
keyword itself, and the absorber is a keyword of the ``settings`` table.
A recipe never imports a module or evaluates code: a type outside the registry is
refused, and the Python export is the way out.

:func:`decode` turns one value of a recipe into the Python value a keyword takes:
quantities into numbers in the keyword's unit, ``{re, im}`` into a complex number,
``{file, sha256}`` into an array read from a checked file, a table with a ``type`` into
an object, and ``"none"`` into ``None`` where the keyword has that spelling.
"""

import functools
import io
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np

from solprop.boundaries import SpatialAbsorber, absorber
from solprop.driver.simulation import SimulationSettings
from solprop.grid import (
    EnvelopeGrid,
    FieldGrid,
    FreeGrid,
    LineGrid,
    RadialGrid,
    TimeGrid,
)
from solprop.modes.antiresonant import TubeLattice
from solprop.modes.capillary import CoherentWall, MarcatiliMode
from solprop.modes.collection import ModeCollection
from solprop.modes.graded_index import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial, MaterialKerr
from solprop.modes.rectangular import RectMode
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.modes.stepindex import StepIndexMode
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata.glasses import glass
from solprop.physdata.mixtures import Mixture
from solprop.physdata.nonlinear_data import gamma3
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import Constant, Linear, PiecewiseLinear, PressureGradient
from solprop.pulses import (
    GaussianPulse,
    PlanarPulse,
    SechPulse,
    SpatialPulse,
    SpatialPulseSum,
    load_spectrum,
)
from solprop.recipe.document import (
    MAX_ARRAY_LENGTH,
    NONE,
    DataFile,
    array_from_bytes,
    checked_bytes,
    decode_complex,
    decode_data_file,
    decode_integer,
    read_data_file,
    unknown_key,
)
from solprop.recipe.errors import RecipeError
from solprop.recipe.metadata import Form, Parameter
from solprop.recipe.quantity import Unit, beta_unit, parse_quantity, phase_unit
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import LowStorageRungeKutta, Multistep, PIController, RungeKutta
from solprop.stepper.schedule import FixedSchedule

__all__ = [
    "REGISTRY",
    "BuildContext",
    "Context",
    "Factory",
    "UnitSpec",
    "build_object",
    "coerce",
    "decode",
    "decode_keyword",
    "unit_of",
]

Context = Literal["grid"]
"""What a factory may be given besides its keywords: format 1 defines the time grid."""

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
    embedded : collections.abc.Mapping
        Data files held in memory by the name the recipe refers to them by, as a
        run's output embeds them (SEP 41, S2); read before the directory.
    mode_fill : bool
        Whether a supplied mode may name its own fill (:attr:`Factory.fill`): on the
        planar route, whose slab channels carry the run's fill. The guided routes
        rebind every supplied mode to their own fill, so there it may not.
    stand_in : bool
        Whether `grid` is a stand-in for checking values before the route has one: a
        factory that solves on the grid (:attr:`Factory.solves`) has its keywords
        decoded and is not called.

    Examples
    --------
    >>> from solprop.recipe.registry import BuildContext
    >>> BuildContext().grid is None
    True
    """

    base: Path | None = None
    allow_outside: bool = False
    grid: TimeGrid | None = None
    embedded: Mapping[str, bytes] = field(default_factory=dict)
    mode_fill: bool = False
    stand_in: bool = False


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
    omitted : tuple of str
        Keywords of the target that a recipe may not give, so that they are not part
        of the format: ``RungeKutta``'s ``tableau``, and a mode's own fill, which the
        route replaces with its own.
    fill : tuple of str
        A mode's own fill keywords, which a recipe gives only where the context
        allows a mode its fill (:attr:`BuildContext.mode_fill`): the slab channels of
        the planar route, whose fill is the run's.
    solves : bool
        Whether calling the target solves on the grid, so that it is not called on a
        stand-in grid (:attr:`BuildContext.stand_in`).

    Examples
    --------
    >>> from solprop.recipe.registry import REGISTRY
    >>> REGISTRY["GaussianPulse"].units["fwhm"]
    's'
    """

    name: str
    target: Callable[..., object]
    units: Mapping[str, UnitSpec] = field(default_factory=dict)
    context: tuple[Context, ...] = ()
    data_file: str | None = None
    omitted: tuple[str, ...] = ()
    fill: tuple[str, ...] = ()
    solves: bool = False


def _pressure_pair(index: int) -> Unit:
    """Return the unit of a row of the ``(z, p)`` form: metres, then bar.

    The form is two rows, the positions and the pressures there, ``[[z...], [p...]]``,
    and every element of a row takes the row's unit.

    Parameters
    ----------
    index : int
        The row.

    Returns
    -------
    Unit
        ``m`` for the positions, ``bar`` for the pressures.
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
_MODE_FILL: Final = ("gas", "pressure", "temperature", "susceptibility", "mixture")
"""A mode's own fill, which a recipe does not give.

The waveguide and capillary routes rebind every supplied mode to the route's fill
(``with_mixture``), keeping only its geometry, cladding and loss, so the fill a mode
was built with never reaches the run: in a recipe the fill is the route's keywords.
"""

_SLAB_FILL: Final = ("gas", "pressure", "temperature")
"""A slab mode's own fill: the run's fill on the planar route, the route's elsewhere.

The planar route takes its fill from its slab channels, so there a slab mode names its
gas, pressure and temperature; the waveguide route rebinds every mode to its own fill
and refuses them. A supplied susceptibility has no table form, and the planar route
takes no mixture.
"""

_SLAB_UNITS: Final[Mapping[str, UnitSpec]] = {
    "half_gap": "m",
    "pressure": "bar",
    "temperature": "K",
}
_BEAM_UNITS: Final[Mapping[str, UnitSpec]] = {
    "waist": "m",
    "propz": "m",
    "displacement": "m",
    "tilt": "rad",
}
_EXECUTION_FIELDS: Final = ("fields", "field_planes", "resume_history")
"""The settings a recipe's ``[execution]`` table holds, and so no ``settings`` table."""

_SETTINGS_OMITTED: Final = (
    *_EXECUTION_FIELDS,
    "auto_init",
    "density",
    "prefactor_index",
    "quadrature",
    "components",
    "adiabatic_accuracy",
    "tdse_execution",
    "_default_integrator",
    "_route",
    "_default_controller",
)
"""The settings fields a ``SimulationSettings`` table does not take.

The execution settings are ``[execution]``'s. The rest are not the free-space routes'
to give: the start-up rule follows ``dz_init``, the
density, prefactor index, transverse quadrature and components are set by a route's
builder, an adiabatic budget and a TDSE's resources have no registered type, and the
underscored fields are private.
"""

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
        Factory("FixedSchedule", FixedSchedule, {"steps": "m", "z_start": "m"}),
        Factory("Multistep", Multistep),
        Factory("RungeKutta", RungeKutta, omitted=("tableau",)),
        Factory("LowStorageRungeKutta", LowStorageRungeKutta),
        Factory(
            "absorber",
            absorber,
            {"boundary_length": "m", "walkoff": "s"},
            context=("grid",),
        ),
        Factory(
            "TubeLattice", TubeLattice, {"wall_thickness": "m", "tube_radius": "m"}
        ),
        Factory("CoherentWall", CoherentWall),
        Factory(
            "MarcatiliMode",
            MarcatiliMode,
            {"radius": "m", "phi": "rad"},
            omitted=_MODE_FILL,
        ),
        Factory(
            "RectMode",
            RectMode,
            {"half_width_x": "m", "half_width_y": "m"},
            omitted=_MODE_FILL,
        ),
        Factory(
            "StepIndexMode",
            StepIndexMode,
            {"radius": "m", "profile_wavelength": "m", "loss": "m^-1"},
        ),
        Factory("glass", glass),
        Factory("IndexMaterial", IndexMaterial),
        Factory("MaterialKerr", MaterialKerr, {"n2": "m^2/W", "at": "m"}),
        Factory("PowerLawPermittivity", PowerLawPermittivity),
        Factory("GradedIndexProvider", GradedIndexProvider, {"radius": "m"}),
        Factory(
            "ModalGuide.on_grid", ModalGuide.on_grid, context=("grid",), solves=True
        ),
        Factory(
            "SlabMode",
            SlabMode,
            _SLAB_UNITS,
            omitted=("susceptibility", "mixture"),
            fill=_SLAB_FILL,
        ),
        Factory(
            "ModeCollection.slab",
            ModeCollection.slab,
            _SLAB_UNITS,
            omitted=("susceptibility", "mixture"),
            fill=_SLAB_FILL,
        ),
        Factory("MaterialSelection", MaterialSelection, {"carrier_wavelength": "m"}),
        # The free-space routes: grids, medium, responses, beams and settings.
        Factory("RadialGrid", RadialGrid, {"radius": "m"}),
        Factory("FreeGrid", FreeGrid, {"radius_x": "m", "radius_y": "m"}),
        Factory("LineGrid", LineGrid, {"radius_x": "m"}),
        Factory("PlanarGrid", PlanarGrid, {"radius_y": "m"}),
        Factory(
            "SpatialMedium",
            SpatialMedium,
            {"pressure": "bar", "temperature": "K"},
            omitted=("susceptibility",),
        ),
        Factory("KerrResponse", KerrResponse),
        Factory("gamma3", gamma3, {"wavelength": "m"}),
        Factory("EnvelopePlasmaResponse", EnvelopePlasmaResponse),
        Factory("PlasmaResponse", PlasmaResponse),
        Factory(
            "OptmatRate",
            OptmatRate,
            {"wavelength": "m", "negligible_rate": "s^-1", "maximum": "s^-1"},
        ),
        Factory("SpatialPulse", SpatialPulse, _BEAM_UNITS),
        Factory("SpatialPulseSum", SpatialPulseSum, {}),
        Factory(
            "PlanarPulse",
            PlanarPulse,
            {**_BEAM_UNITS, "waist_x": "m", "offset_x": "m"},
        ),
        Factory(
            "SimulationSettings",
            SimulationSettings,
            {
                "save_planes": "m",
                "dz_max": "m",
                "dz_init": "m",
                "dz_min": "m",
                "evanescent_rate": "m^-1",
            },
            omitted=_SETTINGS_OMITTED,
        ),
        Factory(
            "PIController",
            PIController,
            omitted=("weights", "lag", "estimator_order"),
        ),
        Factory(
            "SpatialAbsorber",
            SpatialAbsorber,
            {"kappa": "rad/m", "angle": "rad", "discard_above": "rad/m"},
        ),
        Factory("default_spec", default_spec),
    )
}
"""The types a recipe may name, keyed by the ``type`` it writes (SEP 41 section 5.1).

The documented user-facing types of the ``gnlse``, ``capillary``, ``waveguide``,
``radial``, ``free`` and ``planar`` routes. Every keyword of an entry is part of
recipe format 1, so the registry starts small: adding an entry later is additive,
removing one is not. ``RungeKutta`` takes no ``tableau`` in a recipe, whose default is
the only one the interfaces document. The supplied modes (``MarcatiliMode``,
``RectMode``, ``SlabMode`` and ``ModeCollection.slab``) take their geometry, cladding
and loss and not their fill, which the route sets; on the planar route a slab mode's
fill is the run's, so there it names one (:attr:`Factory.fill`). ``StepIndexMode``, with
``glass`` for its materials, is the GNLSE route's ``fibre``. ``ModalGuide.on_grid``
is a graded-index guide on the waveguide route, with its knots and node rule taken from
the run's grid and the provider's own method; a guide built any other way has no table
form.

The free-space routes register their grids (``RadialGrid``, ``FreeGrid``,
``LineGrid``, ``PlanarGrid``), the medium (``SpatialMedium``, without a supplied
susceptibility), the responses the free-space guide uses (``KerrResponse``, with
``gamma3`` for optmat's coefficient of a gas, ``EnvelopePlasmaResponse``,
``PlasmaResponse`` and ``OptmatRate``), the beams (``SpatialPulse``,
``PlanarPulse``), their settings (``SimulationSettings`` without the execution
settings, which are ``[execution]``'s, nor the fields a route's builder sets;
``PIController`` without its per-bin weights, history lag and estimator order;
``default_spec``) and the transverse boundary (``SpatialAbsorber``).
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
        If the list is longer than :data:`~solprop.recipe.document.MAX_ARRAY_LENGTH`.
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
            # A row of a two-row form: every element takes the row's unit.
            row = unit(index).text if callable(unit) else unit
            out.append(_array(element, row, context, where))
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
    >>> from solprop.recipe.registry import BuildContext, decode
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


def _data_bytes(reference: DataFile, context: BuildContext, path: str) -> bytes:
    """Return the checked bytes of a data file, embedded or on disk.

    Parameters
    ----------
    reference : DataFile
        The reference.
    context : BuildContext
        The embedded files and the recipe's directory.
    path : str
        The value's path.

    Returns
    -------
    bytes
        The contents, whose digest is the one recorded.

    Raises
    ------
    RecipeError
        If the file is neither embedded nor reachable from a directory.
    """
    if reference.file in context.embedded:
        return checked_bytes(context.embedded[reference.file], reference, path=path)
    if context.base is None:
        msg = "a recipe read from text holds no data file; load it from its file"
        raise RecipeError(msg, path=path)
    return read_data_file(
        reference, context.base, allow_outside=context.allow_outside, path=path
    )


def _data_array(
    value: Mapping[str, object], context: BuildContext, path: str
) -> np.ndarray:
    """Read an array from a data file a recipe refers to.

    Parameters
    ----------
    value : collections.abc.Mapping
        The ``{file, sha256}`` reference.
    context : BuildContext
        The recipe's directory and embedded files.
    path : str
        The value's path.

    Returns
    -------
    numpy.ndarray
        The array.
    """
    reference = decode_data_file(value, path=path)
    return array_from_bytes(_data_bytes(reference, context, path), reference, path=path)


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
    >>> from solprop.recipe.registry import BuildContext, build_object
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
        data = _data_bytes(reference, context, path)
        keywords[factory.data_file] = io.StringIO(data.decode("utf-8-sig"))
    accepted = _keywords(factory)
    arguments: dict[str, object] = {}
    for key, item in keywords.items():
        if key == factory.data_file:
            arguments[key] = item
            continue
        if factory.target is SimulationSettings and key in _EXECUTION_FIELDS:
            msg = f"{key} is set in the recipe's [execution] table, not in settings"
            raise RecipeError(msg, path=f"{path}.{key}")
        if key not in accepted:
            raise unknown_key(
                key,
                sorted(accepted),
                where=f"the keywords of {factory.name}",
                path=path,
            )
        if key in factory.fill and not context.mode_fill:
            msg = (
                f"{key!r} is the route's: every supplied mode is rebound to the "
                "route's gas, pressure and temperature, so a mode table names no fill"
            )
            raise RecipeError(msg, path=path)
        decoded = decode(item, factory.units.get(key), context, f"{path}.{key}")
        arguments[key] = coerce(factory.target, key, decoded)
    if factory.solves and context.stand_in:
        # A solve on a band that is not the run's would refuse what the run accepts.
        return None
    try:
        return factory.target(*positional, **arguments)
    except TypeError as error:
        raise RecipeError(f"{factory.name}: {error}", path=path) from error


@functools.cache
def _hints(target: Callable[..., object]) -> Mapping[str, object]:
    """Return the annotations of a callable's parameters.

    Parameters
    ----------
    target : callable
        A function, class method or class.

    Returns
    -------
    collections.abc.Mapping
        The annotation of each parameter, read once.
    """
    function = target.__init__ if isinstance(target, type) else target
    try:
        return typing.get_type_hints(function)
    except NameError, TypeError:  # pragma: no cover - every entry's hints resolve
        return {}


def _wants(annotation: object, kind: type) -> bool:
    """Return whether an annotation admits a container of a kind, and not a list.

    Parameters
    ----------
    annotation : object
        The parameter's annotation.
    kind : type
        ``tuple`` or :class:`numpy.ndarray`.

    Returns
    -------
    bool
        True when the annotation names `kind` and admits no list or sequence.
    """
    text = str(annotation)
    if "Sequence" in text or "list[" in text:
        return False
    if kind is tuple:
        return "tuple[" in text
    return "ndarray" in text


def coerce(target: Callable[..., object], name: str, value: object) -> object:
    """Give a decoded list the container its parameter's annotation asks for.

    TOML has lists only; a parameter typed as a tuple takes a tuple and one typed as
    an array takes an array, so a value reaches the constructor as a script passes it.

    Parameters
    ----------
    target : callable
        The constructor or factory.
    name : str
        The parameter.
    value : object
        The decoded value.

    Returns
    -------
    object
        `value`, as a tuple or a float array where the annotation asks for one.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.recipe.registry import coerce
    >>> coerce(EnvelopeGrid.from_limits, "wavelength_limits", [4e-7, 2e-6])
    (4e-07, 2e-06)
    """
    if not isinstance(value, list):
        return value
    annotation = _hints(target).get(name)
    if _wants(annotation, tuple):
        return tuple(value)
    if _wants(annotation, np.ndarray):
        return np.asarray(value, dtype=float)
    return value


def _keywords(factory: Factory) -> frozenset[str]:
    """Return the keywords a factory takes from a recipe.

    Parameters
    ----------
    factory : Factory
        The entry.

    Returns
    -------
    frozenset of str
        Its signature's names, less its context, its data-file keyword and what it
        omits.
    """
    import inspect  # noqa: PLC0415 - only here

    names = set(inspect.signature(factory.target).parameters)
    names -= set(factory.context)
    names -= set(factory.omitted)
    names.discard("self")
    if factory.data_file is not None:
        names.discard(factory.data_file)
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
    >>> from solprop.recipe.registry import BuildContext, decode_keyword
    >>> from solprop.recipe.metadata import parameters
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
    if isinstance(value, str) and "integer" in kinds and "number" not in kinds:
        # An integer beyond 64 bits, such as a drawn seed, is written as a string.
        return decode_integer(value, path=entry.name)
    if (
        not any(form.recipe for form in entry.forms)
        or _python_array(entry, value)
        or _python_type(entry, value)
    ):
        msg = (
            f"{entry.name} takes a Python object or function that a recipe cannot "
            "hold; use the Python export (Recipe.to_python)"
        )
        raise RecipeError(msg, path=entry.name)
    unit = unit_of(entry, value)
    if unit is None and _numeric_rows(entry, value):
        msg = f"{entry.name} takes one row of numbers, not a list of rows"
        raise RecipeError(msg, path=entry.name)
    return decode(value, unit, context, entry.name)


def _python_type(entry: Parameter, value: object) -> bool:
    """Return whether a value names a type of a keyword that only Python holds.

    Parameters
    ----------
    entry : Parameter
        The keyword.
    value : object
        The value as written.

    Returns
    -------
    bool
        True for a table whose ``type`` a form with ``recipe=False`` lists.
    """
    return isinstance(value, Mapping) and any(
        not form.recipe and value.get("type") in form.types for form in entry.forms
    )


def _python_array(entry: Parameter, value: object) -> bool:
    """Return whether a value is an array for a keyword whose array only Python holds.

    An object route's ``initial`` takes a built spectrum as well as a beam: its array
    form is marked as not a recipe's, so a list or a data file given there is refused
    rather than passed on.

    Parameters
    ----------
    entry : Parameter
        The keyword's metadata.
    value : object
        The value as TOML read it.

    Returns
    -------
    bool
        True when the value is a list or a data-file reference and every array form
        of the keyword is one a recipe cannot hold.
    """
    arrays = [form for form in entry.forms if form.kind in ("pair", "array")]
    reference = isinstance(value, Mapping) and set(value) == {"file", "sha256"}
    return (
        bool(arrays)
        and not any(form.recipe for form in arrays)
        and (isinstance(value, list) or reference)
    )


def _numeric_rows(entry: Parameter, value: object) -> bool:
    """Return whether a value is a list of rows for a keyword of numeric arrays.

    Parameters
    ----------
    entry : Parameter
        The keyword's metadata.
    value : object
        The value as TOML read it.

    Returns
    -------
    bool
        True when the value is a list of lists and the keyword has an array form
        whose elements carry units; such a value matched none of its forms.
    """
    rows = isinstance(value, list) and bool(value) and isinstance(value[0], list)
    return rows and any(
        form.kind in ("pair", "array")
        and (form.unit is not None or form.elements is not None)
        for form in entry.forms
    )
