# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pins: a recipe value set from a design target instead of a literal (SEP 41, S1).

A recipe's ``[pins]`` table sets a whole value from a target of
:func:`solprop.tools.solve_parameter`::

    [pins]
    pressure = {target = "zdw", wavelength = "800 nm", search = ["0.1 bar", "10 bar"]}
    energy = {target = "soliton_order", order = 4}

Pins resolve in two fixed stages after the literals: at most one pin on a parameter
of the guide or fill (``zdw`` or ``rdw``), solved on the launch mode at the launch
position; then the pins on pulses (``soliton_order``), which read the mode the first
stage settled. So no order has to be declared and no cycle can be written.
:func:`solve_pin` is the solver call on a route's own keywords, which the Python
export writes before the route's call.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np

from solprop.exceptions import SolpropError
from solprop.recipe.errors import RecipeError
from solprop.recipe.paths import Attribute, Index, format_path, parse_path
from solprop.recipe.quantity import Unit, parse_quantity

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation
    from solprop.modes.base import Mode
    from solprop.recipe.core import Recipe
    from solprop.tools import DesignRoots, DesignTarget, RootChoice

__all__ = [
    "GUIDE_PINS",
    "PULSE_PINS",
    "TARGETS",
    "Pin",
    "PinStage",
    "design_target",
    "read_pins",
    "solve_pin",
    "solve_pulse",
]

type PinStage = Literal["guide", "pulse"]
"""The stage a pin resolves in: a parameter of the guide or fill, or a pulse's."""

GUIDE_PINS: Final[Mapping[str, Mapping[str, str]]] = {
    "capillary": {"radius": "core_radius", "pressure": "pressure"},
}
"""The guide and fill keywords each route can pin, and the mode parameter each sets."""

PULSE_PINS: Final = ("energy", "peak_power")
"""The pulse keywords a pin can set, on the route or on one of its ``pulses``."""

TARGETS: Final[Mapping[str, tuple[PinStage, tuple[str, ...], tuple[str, ...]]]] = {
    "zdw": ("guide", ("wavelength",), ()),
    "rdw": (
        "guide",
        ("rdw",),
        ("peak_power", "fwhm", "shape", "nonlinear_phase"),
    ),
    "soliton_order": ("pulse", ("order",), ()),
}
"""Each target: its stage, its required keys and its optional ones, beside
``target``, ``search`` and ``root``."""

_UNITS: Final[Mapping[str, str | None]] = {
    "wavelength": "m",
    "rdw": "m",
    "peak_power": "W",
    "fwhm": "s",
    "shape": None,
    "nonlinear_phase": None,
    "order": None,
}
"""The unit each target key is read in."""

# The placeholder a pulse pin stands at while the guide is solved: the mode does not
# depend on it, and the route needs some energy or peak power to build.
_PLACEHOLDER: Final[Mapping[str, float]] = {"energy": 1e-6, "peak_power": 1e6}


@dataclass(frozen=True)
class Pin:
    """One entry of a recipe's ``[pins]`` table, checked.

    Attributes
    ----------
    path : str
        The canonical parameter path the pin sets.
    stage : {'guide', 'pulse'}
        Whether it sets a parameter of the guide or fill, or a pulse's energy or
        peak power.
    target : {'zdw', 'rdw', 'soliton_order'}
        The design target.
    keys : collections.abc.Mapping
        The target's keys, numbers in SI.
    search : tuple of float or None
        The search range in the pinned keyword's unit; ``None`` for a pulse pin
        without one.
    root : {'lowest', 'highest'} or int
        Which root, where there are several.
    """

    path: str
    stage: PinStage
    target: str
    keys: Mapping[str, object]
    search: tuple[float, float] | None
    root: "RootChoice"  # noqa: UP037 - TYPE_CHECKING only


def read_pins(recipe: "Recipe") -> tuple[Pin, ...]:  # noqa: UP037 - run time
    """Return a recipe's pins, checked, guide pin first.

    Parameters
    ----------
    recipe : Recipe
        The recipe.

    Returns
    -------
    tuple of Pin
        The pins, the one on the guide or fill (if any) first.

    Raises
    ------
    RecipeError
        On the pinned path, if a pin sets a value the route cannot pin, an element
        of an array, a path also set literally, or a second guide or fill
        parameter; or if its target, keys, search range or root do not read.
    """
    table = recipe.to_dict().get("pins", {})
    pins = [_read_pin(recipe, path, entry) for path, entry in table.items()]
    guide = [pin for pin in pins if pin.stage == "guide"]
    if len(guide) > 1:
        names = ", ".join(pin.path for pin in guide)
        msg = (
            f"at most one parameter of the guide or fill can be pinned, and {names} "
            "are: solve one and write the other as a literal"
        )
        raise RecipeError(msg, path=guide[1].path)
    return (*guide, *(pin for pin in pins if pin.stage == "pulse"))


def _read_pin(recipe: "Recipe", path: str, entry: object) -> Pin:  # noqa: UP037 - one check per rule
    """Return one ``[pins]`` entry, checked.

    Parameters
    ----------
    recipe : Recipe
        The recipe.
    path : str
        The entry's key, a parameter path.
    entry : object
        Its value, a table.

    Returns
    -------
    Pin
        The pin.

    Raises
    ------
    RecipeError
        On `path`, naming the rule broken.
    """
    steps = parse_path(path)
    path = format_path(steps)
    if isinstance(steps[-1], Index):
        msg = (
            "an element of an array cannot be pinned: pin the whole value (a pressure "
            "gradient cannot be pinned at all, because a target needs one mode)"
        )
        raise RecipeError(msg, path=path)
    if _is_set(recipe, path):
        msg = "a pinned value cannot also be set in [build]: remove one of the two"
        raise RecipeError(msg, path=path)
    if not isinstance(entry, Mapping):
        msg = "a pin is a table: {target = ..., search = [low, high], ...}"
        raise RecipeError(msg, path=path)
    entry = cast("Mapping[str, object]", entry)
    target = entry.get("target")
    if target not in TARGETS:
        msg = f"a pin's target is one of {', '.join(TARGETS)}, not {target!r}"
        raise RecipeError(msg, path=path)
    target = cast("str", target)
    stage, required, optional = TARGETS[target]
    keyword = (
        cast("Attribute", steps[0]).name if isinstance(steps[0], Attribute) else ""
    )
    guide = GUIDE_PINS.get(recipe.route, {})
    if stage == "guide" and (len(steps) != 1 or keyword not in guide):
        pinnable = ", ".join(guide) or "nothing"
        msg = (
            f"a {target} pin sets a parameter of the guide or fill, which on the "
            f"{recipe.route} route is one of: {pinnable}"
        )
        raise RecipeError(msg, path=path)
    if stage == "pulse" and not _is_pulse_path(steps):
        msg = (
            f"a {target} pin sets a pulse's energy or peak_power (as energy or "
            "pulses[i].energy)"
        )
        raise RecipeError(msg, path=path)
    allowed = {"target", "search", "root", *required, *optional}
    for key in entry:
        if key not in allowed:
            takes = ", ".join(sorted(allowed))
            msg = f"a {target} pin has no key {key!r}; it takes {takes}"
            raise RecipeError(msg, path=path)
    for key in required:
        if key not in entry:
            msg = f"a {target} pin needs {key!r}"
            raise RecipeError(msg, path=path)
    if target == "rdw" and "peak_power" not in entry and "fwhm" in entry:
        msg = "an rdw pin's fwhm goes with its peak_power"
        raise RecipeError(msg, path=path)
    keys = {
        key: _read_key(entry[key], key, path)
        for key in (*required, *optional)
        if key in entry
    }
    unit = recipe._unit_at(path)
    search = entry.get("search")
    if search is None and stage == "guide":
        msg = "a pin on the guide or fill needs search = [low, high]"
        raise RecipeError(msg, path=path)
    return Pin(
        path=path,
        stage=stage,
        target=target,
        keys=keys,
        search=None if search is None else _read_search(search, unit, path),
        root=_read_root(entry.get("root", "lowest"), path),
    )


def _is_set(recipe: "Recipe", path: str) -> bool:  # noqa: UP037 - run time
    """Return whether a path is also set in ``[build]``.

    Parameters
    ----------
    recipe : Recipe
        The recipe.
    path : str
        The path.

    Returns
    -------
    bool
        Whether ``recipe.get(path)`` finds a value.
    """
    try:
        recipe.get(path)
    except RecipeError, KeyError, IndexError, TypeError:
        return False
    return True


def _is_pulse_path(steps: tuple[object, ...]) -> bool:
    """Return whether a path names a pulse's energy or peak power.

    Parameters
    ----------
    steps : tuple
        The parsed path.

    Returns
    -------
    bool
        For ``energy``, ``peak_power`` or ``pulses[i].energy`` and the like.
    """
    names = [step.name for step in steps if isinstance(step, Attribute)]
    if len(steps) == 1:
        return names == [names[0]] and names[0] in PULSE_PINS
    return (
        len(steps) == 3
        and names[0] == "pulses"
        and isinstance(steps[1], Index)
        and names[-1] in PULSE_PINS
    )


def _read_key(value: object, key: str, path: str) -> object:
    """Return a target key's value, a quantity in SI.

    Parameters
    ----------
    value : object
        The value as written.
    key : str
        The key.
    path : str
        The pin's path, for errors.

    Returns
    -------
    object
        A float in SI, or a string for a choice.

    Raises
    ------
    RecipeError
        If the value does not read in the key's unit.
    """
    unit = _UNITS[key]
    if unit is None:
        if key == "order":
            return _number(value, None, path)
        if not isinstance(value, str):
            msg = f"the pin's {key} is a string, not {value!r}"
            raise RecipeError(msg, path=path)
        return value
    return _number(value, unit, path)


def _number(value: object, unit: str | None, path: str) -> float:
    """Return a number written plain or as a quantity.

    Parameters
    ----------
    value : object
        A number, or a quantity string such as ``"800 nm"``.
    unit : str or None
        The unit it is read in; ``None`` for a plain number.
    path : str
        The pin's path, for errors.

    Returns
    -------
    float
        The number in `unit`.

    Raises
    ------
    RecipeError
        If it does not read, or is not finite.
    """
    if isinstance(value, bool):
        msg = f"expected a number, got {value!r}"
        raise RecipeError(msg, path=path)
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str) and unit is not None:
        number = float(parse_quantity(value, Unit(unit), path=path))
    else:
        kind = "a number" if unit is None else "a number or a quantity"
        msg = f"expected {kind}, got {value!r}"
        raise RecipeError(msg, path=path)
    if not math.isfinite(number):
        msg = f"expected a finite number, got {value!r}"
        raise RecipeError(msg, path=path)
    return number


def _read_search(value: object, unit: str | None, path: str) -> tuple[float, float]:
    """Return a search range, read in the pinned keyword's unit.

    Parameters
    ----------
    value : object
        ``[low, high]``.
    unit : str or None
        The pinned keyword's unit.
    path : str
        The pin's path.

    Returns
    -------
    tuple of float
        ``(low, high)``, ordered.

    Raises
    ------
    RecipeError
        If it is not two ordered numbers.
    """
    if not isinstance(value, list) or len(value) != 2:
        msg = "a pin's search is [low, high]"
        raise RecipeError(msg, path=path)
    low, high = (_number(item, unit, path) for item in value)
    if not low < high:
        msg = f"a pin's search is [low, high] with low < high, got {value!r}"
        raise RecipeError(msg, path=path)
    return low, high


def _read_root(value: object, path: str) -> "RootChoice":  # noqa: UP037 - run time
    """Return a root choice.

    Parameters
    ----------
    value : object
        ``"lowest"``, ``"highest"`` or an integer index.
    path : str
        The pin's path.

    Returns
    -------
    {'lowest', 'highest'} or int
        The choice.

    Raises
    ------
    RecipeError
        If it is none of these.
    """
    if value in ("lowest", "highest"):
        return cast("Literal['lowest', 'highest']", value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"a pin's root is 'lowest', 'highest' or an index, not {value!r}"
    raise RecipeError(msg, path=path)


# ---------------------------------------------------------------------- solving


def design_target(pin: Pin, keywords: Mapping[str, object]) -> "DesignTarget":  # noqa: UP037 - run time
    """Return the solver's target for a guide pin.

    Parameters
    ----------
    pin : Pin
        A ``zdw`` or ``rdw`` pin.
    keywords : collections.abc.Mapping
        The route's keywords, for the pump ``wavelength``.

    Returns
    -------
    ZDW or RDW
        The target.
    """
    from solprop.tools import RDW, ZDW  # noqa: PLC0415 - layer above recipes' import

    keys = pin.keys
    if pin.target == "zdw":
        return ZDW(cast("float", keys["wavelength"]))
    optional: dict[str, object] = {
        key: keys[key]
        for key in ("peak_power", "fwhm", "shape", "nonlinear_phase")
        if key in keys
    }
    return RDW(
        cast("float", keywords["wavelength"]),
        cast("float", keys["rdw"]),
        **optional,  # type: ignore[arg-type]
    )


def solve_pin(
    route: str,
    keywords: Mapping[str, object],
    path: str,
    target: "DesignTarget",  # noqa: UP037 - run time
    search: tuple[float, float],
) -> "DesignRoots":  # noqa: UP037 - run time
    """Solve a parameter of a route's guide or fill for a design target.

    The route is built once from `keywords`, with the parameter at the middle of
    `search`, and its launch mode is re-evaluated at each trial value (no further
    builds), as :func:`solprop.tools.pressure_zdw` does; a target that needs the
    fill's Kerr coefficient is given it at each trial value. This is the call the
    Python export of a pinned recipe writes before the route's call.

    Parameters
    ----------
    route : str
        The route, such as ``"capillary"``.
    keywords : collections.abc.Mapping
        The route's keywords, without the one solved for.
    path : str
        The keyword solved for, one of :data:`GUIDE_PINS` for the route.
    target : ZDW or RDW
        The target.
    search : tuple of float
        The search range, in the keyword's unit (bar for a pressure, m for a radius).

    Returns
    -------
    solprop.tools.DesignRoots
        Every root in the range, or none with the attainable range.

    Raises
    ------
    RecipeError
        If the keyword cannot be pinned on the route, or the route does not build
        with it at the middle of the range.

    Examples
    --------
    >>> from solprop.recipe.pins import solve_pin
    >>> from solprop.tools import ZDW
    >>> keywords = dict(radius=125e-6, gas="He", length=1.0, wavelength=800e-9,
    ...                 fwhm=10e-15, trange=400e-15, wavelength_limits=(130e-9, 4e-6))
    >>> roots = solve_pin("capillary", keywords, "pressure", ZDW(500e-9), (0.01, 10.0))
    >>> f"{roots.pick('lowest'):.3f} bar"
    '1.278 bar'
    """
    from solprop.tools import solve_parameter  # noqa: PLC0415 - layer above

    parameter = GUIDE_PINS.get(route, {}).get(path)
    if parameter is None:
        msg = f"the {route} route cannot pin {path}"
        raise RecipeError(msg, path=path)
    low, high = search
    middle = math.sqrt(low * high) if low > 0.0 else 0.5 * (low + high)
    simulation = _build(route, {**keywords, path: middle}, path)
    mode = _launch_mode(simulation)
    wavelength = cast("float", keywords["wavelength"])
    nonlinear = _needs_gamma(target)

    def build(value: float) -> "Mode | tuple[Mode, float]":  # noqa: UP037
        """Return the launch mode at a trial value, with its Kerr coefficient.

        Parameters
        ----------
        value : float
            The trial value of the pinned parameter.

        Returns
        -------
        Mode or tuple
            The mode, or the mode and ``gamma`` where the target needs it.
        """
        trial = _at(mode, parameter, value)
        return (trial, _gamma(trial, wavelength)) if nonlinear else trial

    return solve_parameter(build, search, target)


def solve_pulse(
    route: str,
    keywords: Mapping[str, object],
    path: str,
    order: float,
    search: tuple[float, float] | None = None,
) -> "DesignRoots":  # noqa: UP037 - run time
    """Return the energy or peak power that gives a pulse a soliton order.

    Closed form on the launch mode at the launch position, with the fill's Kerr
    coefficient (:func:`solprop.tools.soliton_peak_power`). The route is built once,
    the pulse at a placeholder, for its mode. This is the call the Python export of a
    recipe with a ``soliton_order`` pin writes before the route's call.

    Parameters
    ----------
    route : str
        The route, such as ``"capillary"``.
    keywords : collections.abc.Mapping
        The route's keywords. For a route-level `path` (``"energy"`` or
        ``"peak_power"``) without it; for ``"pulses[i].energy"`` with the pulse built
        at any energy, which is not read.
    path : str
        The pulse value solved for.
    order : float
        The conventional soliton order, > 0.
    search : tuple of float or None, optional
        A range the value must lie in, in its unit (J or W); ``None`` for any.

    Returns
    -------
    solprop.tools.DesignRoots
        The one value, or none with it as the attainable range when it lies outside
        `search`.

    Raises
    ------
    RecipeError
        If the pulse is not a sech or ordinary Gaussian pulse, the dispersion at its
        wavelength is not anomalous, or the route does not build.

    Examples
    --------
    >>> from solprop.recipe.pins import solve_pulse
    >>> keywords = dict(radius=125e-6, gas="He", pressure=0.5, length=1.0,
    ...                 wavelength=800e-9, fwhm=10e-15, trange=400e-15,
    ...                 wavelength_limits=(130e-9, 4e-6))
    >>> roots = solve_pulse("capillary", keywords, "energy", 3.0)
    >>> f"{roots.pick() * 1e6:.1f} uJ"
    '348.3 uJ'
    """
    from solprop.modes.base import beta2  # noqa: PLC0415 - layer above
    from solprop.pulses import GaussianPulse, SechPulse  # noqa: PLC0415
    from solprop.tools import (  # noqa: PLC0415
        DesignRoots,
        SolitonOrder,
        pulse_energy,
        soliton_peak_power,
    )

    steps = parse_path(path)
    name = cast("Attribute", steps[-1]).name
    built = dict(keywords)
    if len(steps) == 1:
        built[name] = _PLACEHOLDER[name]
    try:
        if len(steps) == 1:
            pulse = _route_pulse(built)
        else:
            pulses = cast("list[object]", built["pulses"])
            pulse = pulses[cast("Index", steps[1]).position]
        if not isinstance(pulse, GaussianPulse | SechPulse):
            msg = "a soliton_order pin needs a sech or Gaussian pulse"
            raise RecipeError(msg, path=path)
        target = SolitonOrder(order, pulse)
        mode = _launch_mode(_build(route, built, path))
        wavelength = pulse.wavelength
        omega = np.array([2 * math.pi * _C / wavelength])
        dispersion = float(beta2(mode, omega)[0])
        if not dispersion < 0.0:
            msg = (
                f"the dispersion at {wavelength * 1e9:.4g} nm is normal "
                f"(beta2 = {dispersion:.3e} s^2/m): no soliton order to pin"
            )
            raise RecipeError(msg, path=path)
        power = soliton_peak_power(
            target.order,
            target.fwhm,
            _gamma(mode, wavelength),
            dispersion,
            target.shape,
        )
    except SolpropError as error:
        if isinstance(error, RecipeError):
            raise
        raise RecipeError(str(error), path=path) from error
    value = (
        pulse_energy(power, target.fwhm, target.shape) if name == "energy" else power
    )
    inside = search is None or search[0] <= value <= search[1]
    return DesignRoots(
        roots=(value,) if inside else (),
        search=(value, value) if search is None else search,
        target=target,
        builds=1,
        quantity=name,
        attainable=None if inside else (value, value),
    )


_C: Final = 299_792_458.0


def _route_pulse(keywords: Mapping[str, object]) -> object:
    """Return the pulse a route's own pulse keywords describe.

    Parameters
    ----------
    keywords : collections.abc.Mapping
        The route's keywords: ``wavelength``, ``fwhm`` or ``width``, ``shape`` and
        ``energy`` or ``peak_power``.

    Returns
    -------
    GaussianPulse or SechPulse
        The pulse.
    """
    from solprop.pulses import GaussianPulse, SechPulse  # noqa: PLC0415

    shape = keywords.get("shape", "gauss")
    power = {
        name: keywords[name]
        for name in ("energy", "peak_power")
        if keywords.get(name) is not None
    }
    wavelength = cast("float", keywords["wavelength"])
    if shape == "sech":
        widths = {
            name: keywords[name]
            for name in ("fwhm", "width")
            if keywords.get(name) is not None
        }
        return SechPulse(wavelength=wavelength, **widths, **power)  # type: ignore[arg-type]
    return GaussianPulse(
        wavelength=wavelength,
        fwhm=cast("float", keywords["fwhm"]),
        **power,  # type: ignore[arg-type]
    )


def _build(route: str, keywords: Mapping[str, object], path: str) -> "Simulation":  # noqa: UP037 - run time
    """Build a route for its mode, a refusal named on the pinned path.

    Parameters
    ----------
    route : str
        The route.
    keywords : collections.abc.Mapping
        Its keywords; a pulse without an energy or peak power (one a later pin
        solves) is built at a placeholder.
    path : str
        The pinned path.

    Returns
    -------
    solprop.driver.simulation.Simulation
        The simulation, not translated.

    Raises
    ------
    RecipeError
        If the route refuses the keywords.
    """
    from solprop.recipe.metadata import surface  # noqa: PLC0415 - cycle

    if keywords.get("pulses") is None and all(
        keywords.get(name) is None for name in PULSE_PINS
    ):
        # A pulse pin not yet solved: the mode does not depend on the pulse.
        keywords = {**keywords, "energy": _PLACEHOLDER["energy"]}
    try:
        return surface(route)(**keywords)
    except SolpropError as error:
        if isinstance(error, RecipeError):
            raise
        msg = f"the route does not build while the pin is solved: {error}"
        raise RecipeError(msg, path=path) from error


def _launch_mode(simulation: "Simulation") -> "Mode":  # noqa: UP037 - run time
    """Return the mode a pin solves on: the launch mode, the first of the set.

    Parameters
    ----------
    simulation : solprop.driver.simulation.Simulation
        The built route.

    Returns
    -------
    solprop.modes.base.Mode
        Its first mode.
    """
    return simulation.modes[0]


def _at(mode: "Mode", parameter: str, value: float) -> "Mode":  # noqa: UP037
    """Return the launch mode at a trial value of the pinned parameter.

    Parameters
    ----------
    mode : solprop.modes.base.Mode
        The mode as built.
    parameter : str
        ``"pressure"`` (bar) or a mode parameter such as ``"core_radius"`` (m).
    value : float
        The trial value.

    Returns
    -------
    solprop.modes.base.Mode
        The mode there.
    """
    from solprop.tools import _capillary_at  # noqa: PLC0415 - the tools' own re-fill

    if parameter == "pressure":
        return _capillary_at(cast("object", mode), value)  # type: ignore[arg-type]
    return mode.at_parameters({parameter: value})  # type: ignore[attr-defined]


def _gamma(mode: "Mode", wavelength: float) -> float:  # noqa: UP037 - run time
    """Return the fill's nonlinear coefficient on a mode at the launch position.

    Parameters
    ----------
    mode : solprop.modes.base.Mode
        The mode, filled.
    wavelength : float
        The pump wavelength, m.

    Returns
    -------
    float
        ``gamma`` in rad/(W m).

    Raises
    ------
    RecipeError
        If the mode carries no fill.
    """
    from solprop.tools import _fill_n2, nonlinear_coefficient  # noqa: PLC0415

    mixture = getattr(mode, "mixture", None)
    if mixture is None:
        msg = "a nonlinear target needs a filled mode"
        raise RecipeError(msg)
    return nonlinear_coefficient(mode, _fill_n2(mixture, wavelength)[0], wavelength)


def _needs_gamma(target: "DesignTarget") -> bool:  # noqa: UP037 - run time
    """Return whether a target needs the fill's Kerr coefficient.

    Parameters
    ----------
    target : ZDW, RDW or SolitonOrder
        The target.

    Returns
    -------
    bool
        For an RDW with a peak power, and a soliton order.
    """
    from solprop.tools import RDW, SolitonOrder  # noqa: PLC0415

    return isinstance(target, SolitonOrder) or (
        isinstance(target, RDW) and target.peak_power is not None
    )


def placeholder(path: str) -> float:
    """Return the value a pulse pin stands at while the guide is solved.

    Parameters
    ----------
    path : str
        The pulse pin's path.

    Returns
    -------
    float
        An energy of 1 uJ or a peak power of 1 MW, in SI.
    """
    name = cast("Attribute", parse_path(path)[-1]).name
    return _PLACEHOLDER[name]
