# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Machine-readable metadata of every keyword a recipe can set (SEP 41, S9).

One :class:`Parameter` per keyword of a route's ``Simulation.from_*`` constructor, of
its ``prop_*`` function and of the execution settings: its forms (a keyword often takes
several, as ``pressure`` takes a number, a gradient or a profile), their units and
ranges, whether it needs the ``"none"`` spelling, the unit a front end shows it in,
whether the resolved recipe writes it back, the rule that resolves a ``None`` default,
and where it is documented. The function's signature stays the single source of the
default: :func:`parameters` reads it, and a test holds the table and the signature to
the same names.

A front end's tooltips and field validation come from this table; its labels and
wording are its own.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from solprop.driver.defaults import (
    default_integrator,
    default_rtol,
    default_scheme,
    resolve_noise,
)
from solprop.exceptions import DriverError

__all__ = [
    "REQUIRED",
    "Form",
    "Parameter",
    "Resolution",
    "parameters",
    "surface",
]

FormKind = Literal[
    "number",
    "integer",
    "boolean",
    "choice",
    "string",
    "pair",
    "array",
    "mapping",
    "object",
    "callable",
]
"""What a form of a keyword holds.

``"number"`` a real number or quantity string; ``"integer"`` an integer (a decimal
string for one beyond 64 bits); ``"boolean"``; ``"choice"`` one of fixed strings;
``"string"`` free text, such as a gas name; ``"pair"`` two numbers; ``"array"`` a
one-dimensional list of numbers or a data file; ``"mapping"`` a table keyed by name;
``"object"`` a registered type, written as a table with a ``type`` key; ``"callable"``
a Python function, which no recipe holds (the Python export is the way out).
"""

Group = Literal[
    "geometry", "medium", "nonlinear", "input", "grid", "numerics", "output"
]
"""Where a front end shows a keyword."""

Level = Literal["basic", "advanced"]
"""Whether a newcomer sees a keyword, or only an expert."""

ElementUnits = Literal["phases", "betas"]
"""Keywords whose elements carry different units (SEP 41 section 4).

``phases`` is ``[rad, s, s^2, ...]``; ``betas`` is ``s^n/m`` for order ``n``.
"""


class _Required:
    """The default of a keyword that has none: the caller must give it."""

    def __repr__(self) -> str:
        """Return the sentinel's name.

        Returns
        -------
        str
            ``"REQUIRED"``.
        """
        return "REQUIRED"


REQUIRED: Final = _Required()
"""The default :func:`parameters` records for a keyword the caller must give."""


@dataclass(frozen=True)
class Form:
    """One form a keyword's value may take.

    Attributes
    ----------
    kind : str
        A :data:`FormKind`.
    unit : str or None
        The keyword's own unit, in which a bare number is read: SI, except that
        pressures are in bar (``docs/dev/conventions.md``). ``None`` for a number with
        no unit, and for the non-numeric kinds.
    elements : {'phases', 'betas'} or None
        For an array whose elements carry different units, which rule gives them.
    minimum, maximum : float or None
        The allowed range, inclusive unless `exclusive` says otherwise.
    exclusive : bool
        Whether the range's ends are excluded.
    choices : tuple of str
        The allowed values of a ``"choice"``.
    types : tuple of str
        The registered types an ``"object"`` may name.
    length : int or None
        The number of elements of a ``"pair"`` or fixed-length array.
    recipe : bool
        Whether a recipe can hold this form; ``False`` for a callable or an object
        type that is not registered, which only the Python export carries.

    Examples
    --------
    >>> from solprop.recipe._metadata import Form
    >>> Form("number", unit="m", minimum=0.0, exclusive=True).recipe
    True
    """

    kind: FormKind
    unit: str | None = None
    elements: ElementUnits | None = None
    minimum: float | None = None
    maximum: float | None = None
    exclusive: bool = False
    choices: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    length: int | None = None
    recipe: bool = True


@dataclass(frozen=True)
class Resolution:
    """The value a ``None`` default will take, and the rule that chose it.

    Attributes
    ----------
    value : object
        The value the build will use.
    rule : str
        The rule, in a sentence a front end can show.

    Examples
    --------
    >>> from solprop.recipe._metadata import Resolution
    >>> Resolution(1e-6, "the capillary's default").value
    1e-06
    """

    value: object
    rule: str


Resolver = Callable[[Mapping[str, object]], Resolution]
"""Given the values a recipe sets, return what a ``None`` default resolves to."""


@dataclass(frozen=True)
class Parameter:
    """Everything a front end needs to show and check one keyword.

    Attributes
    ----------
    name : str
        The keyword.
    forms : tuple of Form
        The forms its value may take; ``None`` is a form wherever the default is
        ``None`` or `none` is set.
    group : str
        A :data:`Group`.
    level : str
        A :data:`Level`.
    summary : str
        One line.
    anchor : str
        The manual section, as a path under ``docs/`` with an optional ``#anchor``.
    display : str or None
        The unit a front end shows and a canonical recipe writes, in ASCII
        (``"um"``, ``"fs"``); ``None`` for no unit.
    none : bool
        Whether the keyword's default is not ``None`` yet it accepts ``None``, so a
        recipe spells ``None`` as the string ``"none"`` (SEP 41 section 5.1).
    resolvable : bool
        Whether the resolved recipe writes the value that was used (SEP 41 section
        5.2): only where writing it back gives the same record.
    resolver : callable or None
        For a ``None`` default resolved by a rule, the function that says what it
        resolves to (:data:`Resolver`).
    default : object
        The function's default, filled in by :func:`parameters` from the signature;
        :data:`REQUIRED` for a keyword the caller must give.

    Examples
    --------
    >>> from solprop.recipe import _metadata
    >>> table = _metadata.parameters("capillary")
    >>> table["pressure"].display, table["pressure"].forms[0].unit
    ('bar', 'bar')
    """

    name: str
    forms: tuple[Form, ...]
    group: Group
    level: Level
    summary: str
    anchor: str
    display: str | None = None
    none: bool = False
    resolvable: bool = False
    resolver: Resolver | None = None
    default: object = REQUIRED


# ------------------------------------------------------------------- the resolvers


def _precision(values: Mapping[str, object]) -> str:
    """Return the precision a recipe asks for.

    Parameters
    ----------
    values : collections.abc.Mapping
        The recipe's values.

    Returns
    -------
    str
        ``"double"`` unless the recipe sets ``"single"``.
    """
    return "single" if values.get("precision") == "single" else "double"


def _capillary_row(values: Mapping[str, object]) -> str:
    """Return the defaults-table row a capillary recipe reads.

    Parameters
    ----------
    values : collections.abc.Mapping
        The recipe's values.

    Returns
    -------
    str
        ``"capillary tdse"`` beside a TDSE response, else ``"capillary"``.
    """
    return "capillary" if values.get("tdse") is None else "capillary tdse"


def _rtol(route: str) -> Resolver:
    """Return the resolver of a route's ``rtol``.

    Parameters
    ----------
    route : str
        The keyword route.

    Returns
    -------
    callable
        The resolver.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return the route's default in the recipe's precision.

        Parameters
        ----------
        values : collections.abc.Mapping
            The recipe's values.

        Returns
        -------
        Resolution
            The tolerance and its rule.
        """
        row = _capillary_row(values) if route == "capillary" else route
        precision = _precision(values)
        return Resolution(
            default_rtol(row, precision),
            f"the {row} route's default in {precision} precision "
            "(solprop.driver.defaults.ROUTE_DEFAULTS)",
        )

    return resolve


def _integrator(route: str) -> Resolver:
    """Return the resolver of a route's ``integrator``.

    Parameters
    ----------
    route : str
        The keyword route.

    Returns
    -------
    callable
        The resolver.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return the route's default method in the recipe's precision.

        Parameters
        ----------
        values : collections.abc.Mapping
            The recipe's values.

        Returns
        -------
        Resolution
            The integrator and its rule.
        """
        row = _capillary_row(values) if route == "capillary" else route
        precision = _precision(values)
        rule = (
            "single precision steps with RungeKutta()"
            if precision == "single"
            else f"the {row} route's default"
        )
        return Resolution(default_integrator(row, precision), rule)

    return resolve


def _noise(route: str) -> Resolver:
    """Return the resolver of a route's ``noise="default"``.

    Parameters
    ----------
    route : str
        The keyword route.

    Returns
    -------
    callable
        The resolver.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return the model the default draws, and why.

        Parameters
        ----------
        values : collections.abc.Mapping
            The recipe's values.

        Returns
        -------
        Resolution
            The model (``None`` for none) and the reason.
        """
        row = _capillary_row(values) if route == "capillary" else route
        model, resolution = resolve_noise(row, "default", None, _precision(values))
        return Resolution(
            model, resolution.reason or "one photon per mode on the input"
        )

    return resolve


def _scheme(values: Mapping[str, object]) -> Resolution:
    """Return the collar scheme a keyword route takes when none is named.

    Parameters
    ----------
    values : collections.abc.Mapping
        The recipe's values.

    Returns
    -------
    Resolution
        The scheme and its rule.
    """
    del values
    return Resolution(
        default_scheme("gnlse", interpolate_planes=False),
        "the in-stage collar on the guided routes (SEP 21)",
    )


def _by_envelope(name: str) -> Resolver:
    """Return the resolver of a keyword that defaults to the opposite of ``envelope``.

    Parameters
    ----------
    name : str
        ``"thg"`` or ``"plasma"``.

    Returns
    -------
    callable
        The resolver.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return on for a field grid, off for an envelope grid.

        Parameters
        ----------
        values : collections.abc.Mapping
            The recipe's values.

        Returns
        -------
        Resolution
            The switch and its rule.
        """
        envelope = bool(values.get("envelope", False))
        return Resolution(
            not envelope,
            f"{name} is on for a field grid and off for an envelope grid",
        )

    return resolve


def _step_planes(values: Mapping[str, object]) -> Resolution:
    """Return the capillary's ``step_planes`` default.

    Parameters
    ----------
    values : collections.abc.Mapping
        The recipe's values.

    Returns
    -------
    Resolution
        1 beside a TDSE response, else 0.
    """
    tdse = values.get("tdse") is not None
    return Resolution(
        1 if tdse else 0,
        "every step is saved beside a TDSE response, none otherwise",
    )


def _raman(values: Mapping[str, object]) -> Resolution:
    """Return the capillary's ``raman`` default, which depends on the species.

    Parameters
    ----------
    values : collections.abc.Mapping
        The recipe's values.

    Returns
    -------
    Resolution
        The rule; the value is ``None`` here, because it is decided by building the
        fill: on for a molecular gas whose Raman data are curated, off otherwise.
    """
    del values
    return Resolution(
        None,
        "on for a molecular gas with curated Raman data, beside the Kerr response; "
        "off for an atomic gas",
    )


def _pressure(values: Mapping[str, object]) -> Resolution:
    """Return the capillary's default pressure.

    Parameters
    ----------
    values : collections.abc.Mapping
        The recipe's values.

    Returns
    -------
    Resolution
        One bar.
    """
    del values
    return Resolution(1.0, "one bar when no pressure is given")


def _length_fraction(fraction: float, text: str) -> Resolver:
    """Return the resolver of a step length that defaults to a fraction of the length.

    Parameters
    ----------
    fraction : float
        The fraction of ``length``.
    text : str
        The rule.

    Returns
    -------
    callable
        The resolver.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return the fraction of the recipe's length.

        Parameters
        ----------
        values : collections.abc.Mapping
            The recipe's values.

        Returns
        -------
        Resolution
            The step, m, or ``None`` without a length, and the rule.
        """
        length = values.get("length")
        value = None if not isinstance(length, int | float) else fraction * length
        return Resolution(value, text)

    return resolve


# ------------------------------------------------------------------- the forms

_NONE: Final = Form("choice", choices=("none",))
"""The ``"none"`` spelling, as a form."""

_POSITIVE_LENGTH: Final = Form("number", unit="m", minimum=0.0, exclusive=True)
_LIMITS: Final = Form("pair", unit="m", minimum=0.0, exclusive=True, length=2)
_DURATION: Final = Form("number", unit="s", minimum=0.0, exclusive=True)
_ENERGY: Final = Form("number", unit="J", minimum=0.0, exclusive=True)
_POWER: Final = Form("number", unit="W", minimum=0.0, exclusive=True)
_BOOLEAN: Final = Form("boolean")
_PULSES: Final = Form(
    "object",
    types=("GaussianPulse", "SechPulse", "DataPulse", "TimeDataPulse", "load_spectrum"),
)
_GRID: Final = Form("object", types=("EnvelopeGrid.from_limits",))
_STATS: Final = Form("object", types=("StatsSpec",))
_SCHEDULE: Final = Form("object", types=("FixedSchedule.uniform", "FixedSchedule"))
_INTEGRATOR: Final = Form(
    "object", types=("Multistep", "RungeKutta", "LowStorageRungeKutta")
)
_ABSORBER: Final = Form("object", types=("absorber",))
_CHECK: Final = Form("object", types=("BoundaryCheck",))
_MATERIALS: Final = Form("object", types=("MaterialSelection",))
_SCHEMES: Final = Form("choice", choices=("nonlinear", "split_step"))
_SAVE_PLANES: Final = Form("array", unit="m", minimum=0.0, exclusive=True)
_PHASES: Final = Form("array", elements="phases")
_SHAPES: Final = Form("choice", choices=("gauss", "sech"))
_PRECISIONS: Final = Form("choice", choices=("double", "single"))
_BACKENDS: Final = Form("string")
_NOISE: Final = Form("choice", choices=("default", "input", "modified"))
_SEED: Final = Form("integer", minimum=0)

_PULSE_PAGE: Final = "guide/setup/pulses.md"
_GRID_PAGE: Final = "guide/setup/grids.md#choose-the-window-and-wavelength-range"
_STEPPER_PAGE: Final = "guide/numerics/choosing-a-stepper.md#the-defaults"
_BOUNDARY_PAGE: Final = "guide/numerics/boundaries.md"
_NOISE_PAGE: Final = "guide/media/noise.md"
_OUTPUT_PAGE: Final = "guide/results/output.md"
_BACKEND_PAGE: Final = "guide/performance/backends.md"
_PRECISION_PAGE: Final = "guide/performance/single-precision.md"
_MATERIALS_PAGE: Final = "guide/media/material-selection.md"
_PLANES_PAGE: Final = "guide/numerics/solver.md"


def _shared(route: str) -> tuple[Parameter, ...]:
    """Return the keywords the GNLSE and capillary routes share, with the route's rules.

    Parameters
    ----------
    route : {'gnlse', 'capillary'}
        The route.

    Returns
    -------
    tuple of Parameter
        The shared entries.
    """
    return (
        Parameter(
            "length",
            (_POSITIVE_LENGTH,),
            "geometry",
            "basic",
            "Propagation length.",
            "guide/setup/running.md#one-call",
            display="m",
        ),
        Parameter(
            "wavelength",
            (_POSITIVE_LENGTH,),
            "input",
            "basic",
            "Reference wavelength: the grid's centre and the pulse's carrier.",
            _PULSE_PAGE,
            display="nm",
        ),
        Parameter(
            "wavelength_limits",
            (_LIMITS,),
            "grid",
            "basic",
            "Shortest and longest wavelength the grid resolves.",
            _GRID_PAGE,
            display="nm",
        ),
        Parameter(
            "trange",
            (_DURATION,),
            "grid",
            "basic",
            "Length of the time window.",
            _GRID_PAGE,
            display="fs",
        ),
        Parameter(
            "grid",
            (_GRID,),
            "grid",
            "advanced",
            "A grid built in full, in place of wavelength_limits and trange.",
            "guide/setup/grids.md",
        ),
        Parameter(
            "fwhm",
            (_DURATION,),
            "input",
            "basic",
            "Full width at half maximum of the intensity.",
            _PULSE_PAGE,
            display="fs",
        ),
        Parameter(
            "width",
            (_DURATION,),
            "input",
            "advanced",
            "The shape's natural width, in place of fwhm.",
            _PULSE_PAGE,
            display="fs",
        ),
        Parameter(
            "shape",
            (_SHAPES,),
            "input",
            "basic",
            "Temporal shape of the pulse.",
            _PULSE_PAGE,
        ),
        Parameter(
            "energy",
            (_ENERGY,),
            "input",
            "basic",
            "Pulse energy; give it or peak_power.",
            "guide/setup/pulses.md#energy-or-peak-power-never-both",
            display="uJ",
        ),
        Parameter(
            "peak_power",
            (_POWER,),
            "input",
            "basic",
            "Peak power; give it or energy.",
            "guide/setup/pulses.md#energy-or-peak-power-never-both",
            display="W",
        ),
        Parameter(
            "phases",
            (_PHASES,),
            "input",
            "advanced",
            "Spectral phase coefficients: carrier phase, delay, GDD, TOD and on.",
            "guide/setup/pulses.md#signs-where-a-pulse-lands-and-which-way-it-chirps",
        ),
        Parameter(
            "pulses",
            (_PULSES,),
            "input",
            "advanced",
            "Several pulses, in place of the single-pulse keywords.",
            "guide/setup/pulses.md#several-pulses-at-once",
        ),
        Parameter(
            "noise",
            (_NOISE, _NONE),
            "input",
            "advanced",
            "Shot noise added to the input: a model, or none.",
            _NOISE_PAGE,
            none=True,
            resolver=_noise(route),
        ),
        Parameter(
            "seed",
            (_SEED,),
            "input",
            "advanced",
            "Seed of the noise draw; none draws one and records it.",
            "guide/media/noise.md#reproducibility",
        ),
        Parameter(
            "rtol",
            (Form("number", minimum=0.0, exclusive=True),),
            "numerics",
            "advanced",
            "Relative tolerance of the adaptive step.",
            _STEPPER_PAGE,
            resolvable=True,
            resolver=_rtol(route),
        ),
        Parameter(
            "atol",
            (Form("number", minimum=0.0),),
            "numerics",
            "advanced",
            "Absolute tolerance of the adaptive step, in the square root of watts.",
            _STEPPER_PAGE,
            resolvable=True,
        ),
        Parameter(
            "dz_max",
            (_POSITIVE_LENGTH,),
            "numerics",
            "advanced",
            "Largest step the controller may take.",
            _PLANES_PAGE,
            display="mm",
            resolvable=True,
            resolver=_length_fraction(0.5, "half the length"),
        ),
        Parameter(
            "dz_init",
            (_POSITIVE_LENGTH,),
            "numerics",
            "advanced",
            "First step; none takes it from the start-up rule.",
            _PLANES_PAGE,
            display="mm",
            resolver=_length_fraction(1e-4, "a ten-thousandth of the length"),
        ),
        Parameter(
            "schedule",
            (_SCHEDULE,),
            "numerics",
            "advanced",
            "A fixed schedule of steps to replay instead of adapting.",
            "guide/numerics/fixed-schedules.md",
        ),
        Parameter(
            "integrator",
            (_INTEGRATOR,),
            "numerics",
            "advanced",
            "The method that steps the run.",
            _STEPPER_PAGE,
            resolvable=True,
            resolver=_integrator(route),
        ),
        Parameter(
            "interpolate_planes",
            (_BOOLEAN,),
            "output",
            "advanced",
            "Observe the save planes inside steps instead of landing on them.",
            _PLANES_PAGE,
        ),
        Parameter(
            "save_planes",
            (_SAVE_PLANES,),
            "output",
            "basic",
            "Positions to save the field at, after the entrance.",
            _OUTPUT_PAGE,
            display="mm",
            resolvable=True,
        ),
        Parameter(
            "n_save",
            (Form("integer", minimum=2),),
            "output",
            "basic",
            "Number of evenly spaced planes to save, counting the entrance.",
            _OUTPUT_PAGE,
        ),
        Parameter(
            "stats",
            (_STATS,),
            "output",
            "advanced",
            "Which statistics to record at every saved plane.",
            "guide/results/processing.md",
            resolvable=True,
        ),
        Parameter(
            "absorber",
            (_ABSORBER, Form("boolean")),
            "numerics",
            "advanced",
            "The absorbing collars; false switches the soft collars off.",
            _BOUNDARY_PAGE,
        ),
        Parameter(
            "absorber_scheme",
            (_SCHEMES,),
            "numerics",
            "advanced",
            "How the temporal collar is applied.",
            "guide/numerics/boundaries.md#the-two-temporal-schemes",
            resolver=_scheme,
        ),
        Parameter(
            "boundary_check",
            (_CHECK, _NONE),
            "numerics",
            "advanced",
            "How the run watches its time window's edges, or none.",
            "guide/numerics/boundaries.md#watching-the-windows-edges",
            none=True,
        ),
        Parameter(
            "backend",
            (_BACKENDS,),
            "numerics",
            "advanced",
            "The backend that executes the run; auto picks the fastest installed.",
            _BACKEND_PAGE,
        ),
        Parameter(
            "precision",
            (_PRECISIONS,),
            "numerics",
            "advanced",
            "The arithmetic the run is held in.",
            _PRECISION_PAGE,
        ),
        Parameter(
            "materials",
            (_MATERIALS,),
            "medium",
            "advanced",
            "Which optmat models the run's materials take.",
            _MATERIALS_PAGE,
        ),
    )


def _gnlse() -> tuple[Parameter, ...]:
    """Return the GNLSE route's own keywords.

    Returns
    -------
    tuple of Parameter
        The entries not shared with the capillary.
    """
    page = "guide/waveguides/gnlse.md"
    return (
        Parameter(
            "gamma",
            (Form("number", unit="W^-1/m", minimum=0.0),),
            "nonlinear",
            "basic",
            "Nonlinear coefficient; none derives it from n2 and the fibre's area.",
            page,
            display="W^-1/m",
        ),
        Parameter(
            "betas",
            (Form("array", elements="betas"),),
            "medium",
            "basic",
            "Taylor coefficients of the propagation constant, from order zero.",
            "guide/waveguides/gnlse.md#the-fibre-is-a-mode",
        ),
        Parameter(
            "beta",
            (Form("callable", recipe=False),),
            "medium",
            "advanced",
            "The propagation constant as a function, in place of betas.",
            "guide/waveguides/gnlse.md#the-fibre-is-a-mode",
        ),
        Parameter(
            "fibre",
            (Form("object", types=("Mode", "IndexTable"), recipe=False),),
            "medium",
            "advanced",
            "A mode or a tabulated index as the fibre.",
            "guide/setup/running.md#a-table-or-mode-as-the-gnlse-fibre",
        ),
        Parameter(
            "n2",
            (Form("number", unit="m^2/W"), Form("string")),
            "nonlinear",
            "advanced",
            "Nonlinear index, or a material name, with the fibre's area for gamma.",
            page,
            display="m^2/W",
        ),
        Parameter(
            "loss_db_per_m",
            (Form("number", minimum=0.0),),
            "medium",
            "basic",
            "Loss, in dB per metre.",
            page,
        ),
        Parameter(
            "shock",
            (_BOOLEAN,),
            "nonlinear",
            "advanced",
            "Keep the optical shock term.",
            "guide/waveguides/gnlse.md#the-shock-term",
        ),
        Parameter(
            "raman",
            (_BOOLEAN,),
            "nonlinear",
            "basic",
            "Add fused silica's Raman response.",
            "guide/waveguides/gnlse.md#the-raman-fraction-is-not-silently-taken",
            resolvable=True,
        ),
        Parameter(
            "raman_fraction",
            (Form("number", minimum=0.0, maximum=1.0),),
            "nonlinear",
            "advanced",
            "Fraction of the nonlinearity that is Raman; none takes the rule's.",
            "guide/waveguides/gnlse.md#the-raman-fraction-is-not-silently-taken",
        ),
        Parameter(
            "step_planes",
            (Form("integer", minimum=0),),
            "output",
            "advanced",
            "Also save every k-th accepted step; 0 saves the requested planes only.",
            _PLANES_PAGE,
            resolvable=True,
        ),
        Parameter(
            "polarisation_taper",
            (_BOOLEAN,),
            "numerics",
            "advanced",
            "Taper the nonlinear polarisation in the spectral collar too.",
            _BOUNDARY_PAGE,
        ),
    )


def _capillary() -> tuple[Parameter, ...]:
    """Return the capillary route's own keywords.

    Returns
    -------
    tuple of Parameter
        The entries not shared with the GNLSE.
    """
    page = "guide/waveguides/capillaries.md"
    ionisation = "guide/media/ionisation.md"
    return (
        Parameter(
            "radius",
            (_POSITIVE_LENGTH, Form("object", types=("profile",))),
            "geometry",
            "basic",
            "Core radius, or a radius profile along z.",
            page,
            display="um",
        ),
        Parameter(
            "gas",
            (Form("string"), Form("mapping"), Form("object", types=("Mixture",))),
            "medium",
            "basic",
            "The fill: a species, a mapping of species to fractions, or a mixture.",
            "guide/media/mixtures.md",
        ),
        Parameter(
            "pressure",
            (
                Form("number", unit="bar", minimum=0.0),
                Form("pair", unit="bar", minimum=0.0, length=2),
                Form("array", unit="bar", minimum=0.0),
                Form("object", types=("PressureGradient.two_point", "profile")),
            ),
            "medium",
            "basic",
            "Fill pressure: a number, an end-to-end gradient, (z, p) pairs or a "
            "profile.",
            "guide/media/mixtures.md",
            display="bar",
            resolver=_pressure,
        ),
        Parameter(
            "pressure_floor",
            (Form("number", unit="bar", minimum=0.0),),
            "medium",
            "advanced",
            "Lowest pressure of a gradient's evacuated end.",
            "guide/media/mixtures.md",
            display="mbar",
        ),
        Parameter(
            "vacuum_end",
            (Form("choice", choices=("choked", "floor")),),
            "medium",
            "advanced",
            "How a gradient treats an end below its choked pressure.",
            "guide/media/mixtures.md",
        ),
        Parameter(
            "envelope",
            (_BOOLEAN,),
            "grid",
            "basic",
            "Propagate the complex envelope instead of the real field.",
            "guide/setup/grids.md#choose-a-field-or-an-envelope",
        ),
        Parameter(
            "modes",
            (
                Form("integer", minimum=1),
                Form("object", types=("MarcatiliMode",), recipe=False),
            ),
            "geometry",
            "basic",
            "Number of HE1m modes to propagate, or the modes themselves.",
            "guide/waveguides/mode-sets.md",
        ),
        Parameter(
            "mode_average",
            (_BOOLEAN,),
            "geometry",
            "basic",
            "Propagate one mode with its nonlinearity averaged over the profile.",
            "guide/waveguides/mode-averaging.md",
        ),
        Parameter(
            "lattice",
            (Form("object", types=("TubeLattice",)),),
            "geometry",
            "advanced",
            "An antiresonant tube lattice around the core.",
            "guide/waveguides/antiresonant.md",
        ),
        Parameter(
            "wall",
            (Form("object", types=("CoherentWall",)),),
            "geometry",
            "advanced",
            "A coherent wall model in place of the capillary's bulk cladding.",
            "guide/waveguides/coherent-wall.md",
        ),
        Parameter(
            "taper_coupling",
            (_BOOLEAN,),
            "geometry",
            "advanced",
            "Couple the modes of a tapered core.",
            "guide/waveguides/taper-coupling.md",
        ),
        Parameter(
            "model",
            (Form("choice", choices=("full", "reduced")),),
            "geometry",
            "advanced",
            "Which Marcatili model of the propagation constant.",
            "guide/waveguides/capillary-model.md",
            resolvable=True,
        ),
        Parameter(
            "loss",
            (_BOOLEAN, Form("number"), Form("callable", recipe=False)),
            "medium",
            "basic",
            "Include the capillary's loss: on or off, a scale, or a curve.",
            "guide/waveguides/capillary-model.md",
            resolvable=True,
        ),
        Parameter(
            "cladding",
            (
                Form("object", types=("GlassIndex",), recipe=False),
                Form("number"),
            ),
            "medium",
            "advanced",
            "Cladding index; none is fused silica.",
            "guide/waveguides/capillary-model.md",
        ),
        Parameter(
            "susceptibility",
            (Form("object", types=("GasSusceptibility",), recipe=False),),
            "medium",
            "advanced",
            "A gas susceptibility in place of the fill's own.",
            "guide/media/materials.md",
        ),
        Parameter(
            "temperature",
            (
                Form("number", unit="K", minimum=0.0, exclusive=True),
                Form("object", types=("profile",)),
            ),
            "medium",
            "advanced",
            "Gas temperature, or a profile along z.",
            "guide/media/mixtures.md",
            display="K",
            resolvable=True,
        ),
        Parameter(
            "kerr",
            (_BOOLEAN,),
            "nonlinear",
            "basic",
            "Include the Kerr response.",
            "guide/media/nonlinear.md",
        ),
        Parameter(
            "thg",
            (_BOOLEAN,),
            "nonlinear",
            "advanced",
            "Keep third-harmonic generation in the Kerr response.",
            "guide/media/nonlinear.md",
            resolvable=True,
            resolver=_by_envelope("thg"),
        ),
        Parameter(
            "plasma",
            (
                _BOOLEAN,
                Form("object", types=("OptmatRate",), recipe=False),
                Form("mapping"),
            ),
            "nonlinear",
            "basic",
            "Include photoionisation and the plasma: on or off, a rate, or per "
            "species.",
            ionisation,
            resolvable=True,
            resolver=_by_envelope("plasma"),
        ),
        Parameter(
            "preionisation",
            (
                Form("number", minimum=0.0, maximum=1.0),
                Form("object", types=("profile",)),
                Form("mapping"),
            ),
            "nonlinear",
            "advanced",
            "Ionisation fraction already present, or per species.",
            ionisation,
        ),
        Parameter(
            "plasma_integration",
            (Form("choice", choices=("band", "trapezoid")),),
            "nonlinear",
            "advanced",
            "How the plasma current is integrated over time.",
            "guide/media/field-plasma.md",
        ),
        Parameter(
            "raman",
            (_BOOLEAN, Form("array"), Form("mapping")),
            "nonlinear",
            "basic",
            "Include the molecular Raman response: on or off, or which species.",
            "guide/media/raman.md",
            resolvable=True,
            resolver=_raman,
        ),
        Parameter(
            "n_r",
            (Form("integer", minimum=1),),
            "numerics",
            "advanced",
            "Radial quadrature nodes of a multimode run.",
            "guide/waveguides/radial-quadrature.md",
        ),
        Parameter(
            "n_theta",
            (Form("integer", minimum=1),),
            "numerics",
            "advanced",
            "Azimuthal quadrature nodes of a multimode run.",
            "guide/waveguides/radial-quadrature.md",
        ),
        Parameter(
            "radial_rule",
            (
                Form(
                    "choice",
                    choices=(
                        "gauss_kronrod",
                        "gauss",
                        "field_scale",
                        "field_scale_radau",
                    ),
                ),
            ),
            "numerics",
            "advanced",
            "The radial quadrature rule.",
            "guide/waveguides/radial-quadrature.md",
        ),
        Parameter(
            "ionisation_field_range",
            (Form("pair", minimum=0.0, exclusive=True, length=2),),
            "numerics",
            "advanced",
            "Lowest and highest field, V/m, the ionisation-rate table covers.",
            ionisation,
        ),
        Parameter(
            "step_planes",
            (Form("integer", minimum=0),),
            "output",
            "advanced",
            "Also save every k-th accepted step; none takes 1 beside a TDSE response, "
            "else 0.",
            _PLANES_PAGE,
            resolvable=True,
            resolver=_step_planes,
        ),
        Parameter(
            "node_budget",
            (Form("integer", minimum=1),),
            "numerics",
            "advanced",
            "Bytes of node buffers a response pass may hold.",
            "guide/performance/resources.md",
        ),
        Parameter(
            "polarisation",
            (Form("string"), Form("number")),
            "input",
            "advanced",
            "Refused: the capillary route is scalar.",
            "guide/setup/polarisation.md",
        ),
        Parameter(
            "tdse",
            (Form("object", types=("TDSEGasResponse",), recipe=False),),
            "nonlinear",
            "advanced",
            "An atomic TDSE response in place of the rate model.",
            "guide/media/tdse.md",
        ),
        Parameter(
            "tdse_execution",
            (Form("object", types=("TDSEExecution",), recipe=False),),
            "numerics",
            "advanced",
            "Explicit runtime resources of a TDSE response.",
            "guide/media/tdse.md",
        ),
    )


_EXECUTION: Final = (
    Parameter(
        "fields",
        (Form("choice", choices=("all", "planes", "none")),),
        "output",
        "advanced",
        "Which saved planes keep their field.",
        _OUTPUT_PAGE,
    ),
    Parameter(
        "field_planes",
        (Form("array"),),
        "output",
        "advanced",
        "Saved-plane indices that keep their field, with fields='planes'.",
        _OUTPUT_PAGE,
    ),
    Parameter(
        "resume_history",
        (_BOOLEAN,),
        "output",
        "advanced",
        "Keep the Adams method's history at each plane, for an exact resume.",
        "guide/results/output.md",
    ),
    Parameter(
        "memory_budget",
        (Form("integer", minimum=0), Form("string"), _NONE),
        "numerics",
        "advanced",
        "Memory the run may hold, in bytes or as a size such as '8 GiB'.",
        "guide/performance/resources.md",
        none=True,
    ),
)
"""The ``[execution]`` table's keys: settings a run reads when it executes."""


def _route_entries(route: str) -> tuple[Parameter, ...]:
    """Return a route's entries, the shared ones overridden by the route's own.

    Parameters
    ----------
    route : {'gnlse', 'capillary'}
        The route, already checked by :func:`surface`.

    Returns
    -------
    tuple of Parameter
        The entries, without defaults.
    """
    own = _gnlse() if route == "gnlse" else _capillary()
    names = {entry.name for entry in own}
    return (*own, *(entry for entry in _shared(route) if entry.name not in names))


def surface(route: str) -> Callable[..., object]:
    """Return the constructor whose keywords a route's ``[build]`` table holds.

    Parameters
    ----------
    route : {'gnlse', 'capillary'}
        The route.

    Returns
    -------
    callable
        ``Simulation.from_gnlse`` or ``Simulation.from_capillary``.

    Raises
    ------
    DriverError
        If the route has no recipe yet.

    Examples
    --------
    >>> from solprop.recipe._metadata import surface
    >>> surface("gnlse").__name__
    'from_gnlse'
    """
    from solprop.driver.simulation import Simulation  # noqa: PLC0415 - layer D

    builders = {"gnlse": Simulation.from_gnlse, "capillary": Simulation.from_capillary}
    try:
        return builders[route]
    except KeyError:
        msg = f"no recipe surface for route {route!r}; the routes are gnlse, capillary"
        raise DriverError(msg) from None


def parameters(route: str) -> Mapping[str, Parameter]:
    """Return the metadata of a route's build keywords, with their defaults.

    Parameters
    ----------
    route : {'gnlse', 'capillary', 'execution'}
        A keyword route, or ``"execution"`` for the ``[execution]`` table's keys.

    Returns
    -------
    collections.abc.Mapping
        :class:`Parameter` by keyword, in the constructor's order, each with the
        default its signature declares (``REQUIRED`` where it has none).

    Examples
    --------
    >>> from solprop.recipe._metadata import parameters
    >>> table = parameters("gnlse")
    >>> table["rtol"].default is None, table["rtol"].resolver({}).value
    (True, 1e-07)
    >>> parameters("execution")["fields"].default
    'all'
    """
    if route == "execution":
        from solprop.driver.simulation import (  # noqa: PLC0415 - layer D
            SimulationSettings,
        )

        fields = {field.name: field for field in dataclasses.fields(SimulationSettings)}
        out = {}
        for entry in _EXECUTION:
            field = fields.get(entry.name)
            default = (
                None
                if field is None or field.default is dataclasses.MISSING
                else field.default
            )
            out[entry.name] = dataclasses.replace(entry, default=default)
        return MappingProxyType(out)
    signature = inspect.signature(surface(route))
    entries = {entry.name: entry for entry in _route_entries(route)}
    ordered = {}
    for name, parameter in signature.parameters.items():
        if name in entries:
            default = (
                REQUIRED
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            ordered[name] = dataclasses.replace(entries[name], default=default)
    return MappingProxyType(ordered)
