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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from solprop.driver.defaults import (
    default_scheme,
    describe_default,
    route_row,
)
from solprop.exceptions import DriverError
from solprop.modes.slab import SlabMode

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = [
    "OBJECT_ROUTES",
    "PER_WIDTH_KEYWORDS",
    "REQUIRED",
    "SETTINGS_KEYWORDS",
    "ElementUnits",
    "Form",
    "FormKind",
    "Group",
    "Level",
    "Parameter",
    "Resolution",
    "Resolver",
    "parameters",
    "per_unit_width",
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

ElementUnits = Literal["phases", "betas", "z_pressure"]
"""Keywords whose elements carry different units (SEP 41 section 4).

``phases`` is ``[rad, s, s^2, ...]``; ``betas`` is ``s^n/m`` for order ``n``; the
``(z, p)`` form of ``pressure`` is two rows, positions in metres and pressures in bar.
"""


class _Required:
    """The default of a keyword that has none: the caller must give it."""

    def __repr__(self) -> str:
        """Return the sentinel's name."""
        return "REQUIRED"


REQUIRED: Final = _Required()
"""The default :func:`parameters` records for a keyword the caller must give."""

OBJECT_ROUTES: Final = ("radial", "free", "planar")
"""The routes whose ``[build]`` keywords are objects: grids, medium, beam and settings.

Their ``Simulation.from_*`` takes the time grid, the transverse grid, the medium, the
responses and the beam positionally, and the stepping in a ``settings`` object; a
recipe writes each as a table under its keyword's name.
"""

SETTINGS_KEYWORDS: Final = ("absorber", "absorber_scheme")
"""Keywords of an object route's constructor that a recipe writes under ``settings``.

``prop_radial``, ``prop_free`` and ``prop_planar`` apply them to their settings; a
recipe holds the ``from_*`` form, so they are keywords of its ``settings`` table
(SEP 41 section 5.1), and the route's table does not list them.
"""


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
    >>> from solprop.recipe.metadata import Form
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
    >>> from solprop.recipe.metadata import Resolution
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
    >>> from solprop.recipe import metadata
    >>> table = metadata.parameters("capillary")
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


_SLAB_TYPES: Final = ("SlabMode", "ModeCollection.slab")
"""The registered types that make a waveguide's mode set a slab set."""


def _is_slab(value: object) -> bool:
    """Return whether a ``modes`` value, as a recipe or a script gives it, holds a slab.

    - Returns: True when any element is a slab mode or a collection of them.
    """
    if isinstance(value, Mapping):
        return value.get("type") in _SLAB_TYPES
    if isinstance(value, SlabMode):
        return True
    if isinstance(value, Sequence) and not isinstance(value, str):
        return any(_is_slab(item) for item in value)
    return False


PER_WIDTH_KEYWORDS: Final = ("energy", "peak_power")
"""The keywords that are per metre of y on a slab mode set (:func:`per_unit_width`)."""


def per_unit_width(route: str, values: Mapping[str, object]) -> bool:
    """Return whether a recipe's pulse energy and peak power are per metre of y.

    A slab mode set is uniform in y, a plane wave in the free direction, so on the
    waveguide route with slab modes the :data:`PER_WIDTH_KEYWORDS` are read in J/m
    and W/m though written, as everywhere, in J and W: the value is the same number,
    and a front end shows the unit per metre.

    Parameters
    ----------
    route : str
        The route.
    values : collections.abc.Mapping
        The recipe's ``[build]`` values, as it gives them (``modes`` as tables).

    Returns
    -------
    bool
        Whether the route is the waveguide's and its mode set a slab set.

    Examples
    --------
    >>> from solprop.recipe.metadata import per_unit_width
    >>> per_unit_width("waveguide", {"modes": {"type": "ModeCollection.slab"}})
    True
    >>> per_unit_width("waveguide", {"modes": [{"type": "RectMode"}]})
    False
    """
    return route == "waveguide" and _is_slab(values.get("modes"))


def _slab_note(route: str) -> str:
    """Return what a route's energy summary adds for slab sets.

    - Returns: `` (per metre of y on a slab set)`` on the waveguide route, else nothing.
    """
    return " (per metre of y on a slab set)" if route == "waveguide" else ""


def _row(route: str, values: Mapping[str, object]) -> str:
    """Return the defaults-table row a recipe reads.

    - Returns: ``"capillary tdse"`` for a capillary beside a TDSE response, and
      ``"waveguide slab"`` for a waveguide of slab modes; the route otherwise.
    """
    return route_row(
        route,
        tdse=route == "capillary" and values.get("tdse") is not None,
        slab=per_unit_width(route, values),
    )


def _route_default(
    route: str, keyword: Literal["rtol", "integrator", "noise", "dz_max", "dz_init"]
) -> Resolver:
    """Return the resolver of a keyword whose default is the route's own.

    The value and the rule are :func:`solprop.driver.defaults.describe_default`'s, for
    the defaults-table row the recipe's values select.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return the row's default for the keyword, and its rule."""
        length = values.get("length")
        value, rule = describe_default(
            _row(route, values),
            keyword,
            length if isinstance(length, int | float) else None,
        )
        return Resolution(value, rule)

    return resolve


def _scheme(values: Mapping[str, object]) -> Resolution:
    """Return the collar scheme a keyword route takes when none is named."""
    del values
    return Resolution(
        default_scheme("gnlse", interpolate_planes=False),
        "the in-stage collar on the guided routes (SEP 21)",
    )


def _by_envelope(name: str) -> Resolver:
    """Return the resolver of a keyword that defaults to the opposite of ``envelope``.

    - ``name``: ``"thg"`` or ``"plasma"``.
    """

    def resolve(values: Mapping[str, object]) -> Resolution:
        """Return on for a field grid, off for an envelope grid."""
        envelope = bool(values.get("envelope", False))
        return Resolution(
            not envelope,
            f"{name} is on for a field grid and off for an envelope grid",
        )

    return resolve


def _step_planes(values: Mapping[str, object]) -> Resolution:
    """Return the capillary's ``step_planes`` default.

    - Returns: 1 beside a TDSE response, else 0.
    """
    tdse = values.get("tdse") is not None
    return Resolution(
        1 if tdse else 0,
        "every step is saved beside a TDSE response, none otherwise",
    )


def _requested_planes(values: Mapping[str, object]) -> Resolution:
    """Return the ``step_planes`` default of the GNLSE and the waveguide.

    - Returns: 0: the requested planes only.
    """
    del values
    return Resolution(0, "the requested planes only")


def _raman(values: Mapping[str, object]) -> Resolution:
    """Return the capillary's ``raman`` default, which depends on the species.

    - Returns: the rule; the value is ``None`` here, because it is decided by building
      the fill: on for a molecular gas whose Raman data are curated, off otherwise.
    """
    del values
    return Resolution(
        None,
        "on for a molecular gas with curated Raman data, beside the Kerr response; "
        "off for an atomic gas",
    )


def _pressure(values: Mapping[str, object]) -> Resolution:
    """Return the capillary's default pressure.

    - Returns: one bar.
    """
    del values
    return Resolution(1.0, "one bar when no pressure is given")


# ------------------------------------------------------------------- the forms

_PROFILES: Final = ("Constant", "Linear", "PiecewiseLinear")
"""The registered profiles along z a keyword may take."""

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
    types=("GaussianPulse", "SechPulse", "load_spectrum"),
)
_GRID: Final = Form(
    "object", types=("EnvelopeGrid.from_limits", "FieldGrid.from_limits")
)
_STATS: Final = Form("object", types=("StatsSpec",))
_SCHEDULE: Final = Form("object", types=("FixedSchedule.uniform",))
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
_MATERIALS_PAGE: Final = "guide/media/material-selection.md"
_PLANES_PAGE: Final = "guide/numerics/solver.md"


def _shared(route: str) -> tuple[Parameter, ...]:
    """Return the keywords the keyword routes share, with the route's rules."""
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
            f"Pulse energy; give it or peak_power{_slab_note(route)}.",
            "guide/setup/pulses.md#energy-or-peak-power-never-both",
            display="uJ",
        ),
        Parameter(
            "peak_power",
            (_POWER,),
            "input",
            "basic",
            f"Peak power; give it or energy{_slab_note(route)}.",
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
            "launch",
            (Form("mapping"), Form("array")),
            "input",
            "advanced",
            "The modes the pulse is launched into: a share (and a phase) per mode "
            "label, or one such table per pulse.",
            "guide/setup/pulses.md#launching-into-chosen-modes",
        ),
        Parameter(
            "noise",
            (_NOISE, _NONE),
            "input",
            "advanced",
            "Shot noise added to the input: a model, or none.",
            _NOISE_PAGE,
            none=True,
            resolver=_route_default(route, "noise"),
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
            resolver=_route_default(route, "rtol"),
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
            resolver=_route_default(route, "dz_max"),
        ),
        Parameter(
            "dz_init",
            (_POSITIVE_LENGTH,),
            "numerics",
            "advanced",
            "First step; none takes it from the start-up rule.",
            _PLANES_PAGE,
            display="mm",
            resolver=_route_default(route, "dz_init"),
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
            resolver=_route_default(route, "integrator"),
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
            "materials",
            (_MATERIALS,),
            "medium",
            "advanced",
            "Which optmat models the run's materials take.",
            _MATERIALS_PAGE,
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


def _gnlse() -> tuple[Parameter, ...]:
    """Return the GNLSE route's own keywords."""
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
            none=True,
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
            (
                Form("object", types=("StepIndexMode",)),
                Form("object", types=("Mode", "IndexTable"), recipe=False),
            ),
            "medium",
            "advanced",
            "A mode or a tabulated index as the fibre.",
            "guide/setup/running.md#a-table-or-mode-as-the-gnlse-fibre",
        ),
        Parameter(
            "n2",
            (Form("number", unit="m^2/W"), Form("choice", choices=("recommended",))),
            "nonlinear",
            "advanced",
            "Nonlinear index, or recommended for optmat's silica, with the fibre's "
            "area for gamma.",
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
            "Also save every k-th accepted step; none and 0 save the requested planes "
            "only.",
            _PLANES_PAGE,
            resolvable=True,
            resolver=_requested_planes,
        ),
    )


def _capillary() -> tuple[Parameter, ...]:
    """Return the capillary route's own keywords."""
    page = "guide/waveguides/capillaries.md"
    ionisation = "guide/media/ionisation.md"
    return (
        Parameter(
            "radius",
            (_POSITIVE_LENGTH, Form("object", types=_PROFILES)),
            "geometry",
            "basic",
            "Core radius, or a radius profile along z.",
            page,
            display="um",
        ),
        Parameter(
            "gas",
            (
                Form("string"),
                Form("mapping"),
                Form(
                    "object",
                    types=("Mixture.fill", "Mixture.partial_pressures", "Mixture.air"),
                ),
            ),
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
                Form("array", elements="z_pressure"),
                Form(
                    "object",
                    types=(
                        "PressureGradient.two_point",
                        "PressureGradient.from_stages",
                        *_PROFILES,
                    ),
                ),
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
                Form("object", types=("MarcatiliMode",)),
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
                Form("object", types=_PROFILES),
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
                Form("object", types=("OptmatRate",)),
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
                Form("object", types=_PROFILES),
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


_MODE_TYPES: Final = ("MarcatiliMode", "RectMode", "SlabMode", "ModeCollection.slab")
"""The registered supplied modes: geometry, cladding and loss, rebound to the fill."""


def _waveguide() -> tuple[Parameter, ...]:
    """Return the waveguide route's own keywords.

    The fill, the responses and the transverse rule are the capillary's entries
    (:func:`_route_entries`); these are the ones the waveguide has alone or reads
    differently.
    """
    supplied = "guide/waveguides/supplied-guides.md"
    provider = f"{supplied}#a-guide-instead-of-a-fill"
    general = "guide/waveguides/general-modal.md"
    beam = "guide/workflows/coupling.md#a-beam-into-a-general-modal-guide"
    rule = "guide/waveguides/mode-sets.md#the-transverse-quadrature"
    return (
        Parameter(
            "modes",
            (
                Form("object", types=_MODE_TYPES),
                Form("object", types=("Mode",), recipe=False),
            ),
            "geometry",
            "basic",
            "The supplied gas-filled modes, or a slab collection; empty beside a "
            "provider.",
            f"{supplied}#supplied-gas-filled-waveguides",
        ),
        Parameter(
            "provider",
            (
                Form("object", types=("ModalGuide.on_grid",)),
                Form("object", types=("ModalGuide",), recipe=False),
            ),
            "geometry",
            "advanced",
            "A guide that solves its own fields, in place of modes and a fill.",
            provider,
        ),
        Parameter(
            "channel",
            (Form("integer", minimum=0), Form("string")),
            "input",
            "advanced",
            "The channel a provider's pulse enters, by index or display label.",
            provider,
        ),
        Parameter(
            "incident",
            (Form("object", types=("SpatialPulse",)),),
            "input",
            "advanced",
            "An incident beam coupled into a provider's channels.",
            beam,
        ),
        Parameter(
            "coupling",
            (Form("object", types=("CouplingModel",), recipe=False),),
            "input",
            "advanced",
            "The model an incident beam is coupled under.",
            beam,
        ),
        Parameter(
            "kerr_form",
            (Form("choice", choices=("real", "analytic")),),
            "nonlinear",
            "advanced",
            "The cubic a provider's material regions respond with.",
            provider,
        ),
        Parameter(
            "observable_rtol",
            (Form("number", minimum=0.0),),
            "numerics",
            "advanced",
            "Accuracy of a provider run's observables; only the solver's own, none.",
            f"{general}#declared-refused-recorded",
        ),
        Parameter(
            "observable_atol",
            (Form("number", minimum=0.0),),
            "numerics",
            "advanced",
            "Accuracy of a provider run's observables; only the solver's own, none.",
            f"{general}#declared-refused-recorded",
        ),
        Parameter(
            "n_r",
            (Form("integer", minimum=1),),
            "numerics",
            "advanced",
            "Radial nodes, or the x-axis nodes of a Cartesian or slab rule.",
            rule,
        ),
        Parameter(
            "n_theta",
            (Form("integer", minimum=1),),
            "numerics",
            "advanced",
            "Azimuthal nodes, or the y-axis nodes of a Cartesian rule.",
            rule,
        ),
        Parameter(
            "polarisation",
            (Form("string"), Form("number")),
            "input",
            "advanced",
            "Refused: the waveguide route is scalar.",
            "guide/setup/polarisation.md",
        ),
        Parameter(
            "step_planes",
            (Form("integer", minimum=0),),
            "output",
            "advanced",
            "Also save every k-th accepted step; none and 0 save the requested planes "
            "only.",
            _PLANES_PAGE,
            resolvable=True,
            resolver=_requested_planes,
        ),
    )


_SPACES: Final[Mapping[str, tuple[str, ...]]] = {
    "radial": ("RadialGrid",),
    "free": ("FreeGrid", "LineGrid"),
    "planar": ("PlanarGrid",),
}
"""The transverse grids each object route takes."""

_RESPONSES: Final = (
    "KerrResponse",
    "EnvelopePlasmaResponse",
    "PlasmaResponse",
)
"""The registered responses an object route's ``responses`` may list."""


def _free_space(route: str) -> tuple[Parameter, ...]:
    """Return an object route's own keywords.

    The noise, seed, material selection and boundary check are the shared entries
    (:func:`_route_entries`); the absorber and its scheme are keywords of the
    ``settings`` table (:data:`SETTINGS_KEYWORDS`).
    """
    beams = "guide/free-space/beams.md"
    planar = "guide/waveguides/planar.md#resolving-y"
    space_page = planar if route == "planar" else f"{beams}#choosing-a-grid"
    beam = (
        Form("object", types=("PlanarPulse",))
        if route == "planar"
        else Form("object", types=("SpatialPulse", "SpatialPulseSum"))
    )
    return (
        Parameter(
            "grid",
            (_GRID,),
            "grid",
            "basic",
            "The time grid: a field grid or an envelope grid.",
            "guide/setup/grids.md",
        ),
        Parameter(
            "space",
            (Form("object", types=_SPACES[route]),),
            "geometry",
            "basic",
            "The transverse grid, with the room outside it for a collar."
            if route != "planar"
            else "The slab modes in x, with their fill, and the y axis.",
            space_page,
        ),
        Parameter(
            "medium",
            (Form("object", types=("SpatialMedium",)),),
            "medium",
            "basic",
            "The bulk medium: a gas at a pressure, a constant index, or vacuum.",
            f"{beams}#a-beam-in-argon",
        ),
        Parameter(
            "responses",
            (Form("object", types=_RESPONSES),),
            "nonlinear",
            "basic",
            "The nonlinear responses; none is linear propagation.",
            f"{beams}#ionisation-plasma-and-filamentation",
        ),
        Parameter(
            "initial",
            (beam, Form("array", recipe=False)),
            "input",
            "basic",
            "The launched beam, or a built spectrum, which only Python carries.",
            planar if route == "planar" else f"{beams}#a-beam-in-argon",
        ),
        Parameter(
            "settings",
            (Form("object", types=("SimulationSettings",)),),
            "numerics",
            "basic",
            "Stepping, saved planes, statistics, absorber and backend.",
            "guide/free-space/model.md#construct-and-run-a-beam",
        ),
        Parameter(
            "frame",
            (Form("object", types=("Frame",), recipe=False),),
            "numerics",
            "advanced",
            "The moving frame; none follows the responses' choice.",
            "guide/free-space/model.md#exact-longitudinal-operator",
        ),
        Parameter(
            "spatial_absorber",
            (Form("object", types=("SpatialAbsorber",)), _BOOLEAN),
            "numerics",
            "advanced",
            "The transverse boundary: a designed collar, or false for no ramp.",
            "guide/free-space/model.md#transverse-boundaries",
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
)
"""The ``[execution]`` table's keys: settings a run reads when it executes."""


def _route_entries(route: str) -> tuple[Parameter, ...]:
    """Return a route's entries, the shared ones overridden by the route's own.

    The waveguide takes the capillary's entries for the fill, the responses and the
    rule, under its own; an entry the route's constructor does not name is dropped by
    :func:`parameters`.
    """
    layers = {
        "gnlse": (_gnlse(),),
        "capillary": (_capillary(),),
        "waveguide": (_waveguide(), _capillary()),
        "radial": (_free_space("radial"),),
        "free": (_free_space("free"),),
        "planar": (_free_space("planar"),),
    }[route]
    entries: dict[str, Parameter] = {}
    for layer in (*layers, _shared(route)):
        for entry in layer:
            entries.setdefault(entry.name, entry)
    return tuple(entries.values())


def surface(route: str) -> "Callable[..., Simulation]":
    """Return the constructor whose keywords a route's ``[build]`` table holds.

    Parameters
    ----------
    route : str
        The route, one of :data:`solprop.recipe.core.ROUTES`.

    Returns
    -------
    callable
        The route's ``Simulation.from_*``: ``from_gnlse``, ``from_capillary``,
        ``from_waveguide``, ``from_radial``, ``from_free`` or ``from_planar``.

    Raises
    ------
    DriverError
        If the route has no recipe yet.

    Examples
    --------
    >>> from solprop.recipe.metadata import surface
    >>> surface("gnlse").__name__
    'from_gnlse'
    """
    from solprop.driver.simulation import Simulation  # noqa: PLC0415 - layer D

    builders = {
        "gnlse": Simulation.from_gnlse,
        "capillary": Simulation.from_capillary,
        "waveguide": Simulation.from_waveguide,
        "radial": Simulation.from_radial,
        "free": Simulation.from_free,
        "planar": Simulation.from_planar,
    }
    try:
        return builders[route]
    except KeyError:
        msg = (
            f"no recipe surface for route {route!r}; the routes are "
            f"{', '.join(builders)}"
        )
        raise DriverError(msg) from None


# ---------------------------------------------------------- the registered types
#
# A recipe's tables name registered types (solprop.recipe.registry.REGISTRY): a second
# pulse, a pressure gradient, a grid, a mode, the whole [build] of a free-space route.
# Each type's keywords get the entry a route keyword has, so that a front end builds
# every table's form from solprop's own description. A test holds every form's unit to
# the registry's, every keyword to the target's signature and every anchor to the
# manual.


def _number(
    unit: str | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive: bool = False,
) -> Form:
    """Return a number's form."""
    return Form(
        "number", unit=unit, minimum=minimum, maximum=maximum, exclusive=exclusive
    )


def _integer(minimum: float | None = None, maximum: float | None = None) -> Form:
    """Return an integer's form."""
    return Form("integer", minimum=minimum, maximum=maximum)


def _choice(*choices: str) -> Form:
    """Return the form of one of fixed strings."""
    return Form("choice", choices=choices)


def _object(*types: str) -> Form:
    """Return the form of a registered type."""
    return Form("object", types=types)


_POSITIVE: Final = _number(minimum=0.0, exclusive=True)
_FRACTION: Final = _number(minimum=0.0, maximum=1.0)
_LENGTH: Final = _number("m", minimum=0.0, exclusive=True)
_POSITION: Final = _number("m")
_TYPE_DURATION: Final = _number("s", minimum=0.0, exclusive=True)
_PRESSURE: Final = _number("bar", minimum=0.0)
_TEMPERATURE: Final = _number("K", minimum=0.0, exclusive=True)
_TYPE_BOOLEAN: Final = Form("boolean")
_STRING: Final = Form("string")
_MAPPING: Final = Form("mapping")
_TYPE_PROFILES: Final = ("Constant", "Linear", "PiecewiseLinear")
_PROFILE: Final = _object(*_TYPE_PROFILES)
_CALLABLE: Final = Form("callable", recipe=False)
_UNREGISTERED: Final = Form("object", recipe=False)
_TYPE_PULSES: Final = ("GaussianPulse", "SechPulse", "load_spectrum")
_LOSS: Final = (_TYPE_BOOLEAN, _number(minimum=0.0), _CALLABLE)
_CLADDING: Final = (_UNREGISTERED, Form("mapping"))
_GAS: Final = (_STRING,)
_FIBRE_MATERIAL: Final = (_number(minimum=0.0, exclusive=True), _object("glass"))


def _keyword(
    name: str,
    forms: tuple[Form, ...] | Form,
    level: Level,
    summary: str,
    *,
    display: str | None = None,
) -> tuple[str, tuple[Form, ...], Level, str, str | None]:
    """Return one keyword's row, its group and anchor left to its type."""
    return (
        name,
        forms if isinstance(forms, tuple) else (forms,),
        level,
        summary,
        display,
    )


Row = tuple[str, tuple[Form, ...], Level, str, str | None]


def _type(group: Group, anchor: str, *rows: Row) -> tuple[Parameter, ...]:
    """Return a type's entries, each with the type's group and anchor."""
    return tuple(
        Parameter(name, forms, group, level, summary, anchor, display=display)
        for name, forms, level, summary, display in rows
    )


# ------------------------------------------------------------------------ the types

_TYPE_PULSE_PAGE: Final = "guide/setup/pulses.md"
_PROFILE_PAGE: Final = "guide/waveguides/profiles.md"
_MIXTURE_PAGE: Final = "guide/media/mixtures.md"
_TYPE_GRID_PAGE: Final = "guide/setup/grids.md"
_TYPE_STEPPER_PAGE: Final = "guide/numerics/choosing-a-stepper.md"
_TYPE_BOUNDARY_PAGE: Final = "guide/numerics/boundaries.md"
_FREE_PAGE: Final = "guide/free-space/model.md"
_PLANAR_PAGE: Final = "guide/waveguides/planar.md"
_IONISATION_PAGE: Final = "guide/media/ionisation.md"


def _pulse(*extra: Row) -> tuple[Parameter, ...]:
    """Return a shaped pulse's entries: the shared keywords and its own."""
    return _type(
        "input",
        _TYPE_PULSE_PAGE,
        _keyword("wavelength", _LENGTH, "basic", "Centre wavelength.", display="nm"),
        _keyword(
            "energy",
            _number("J", minimum=0.0, exclusive=True),
            "basic",
            "Pulse energy; give it or the peak power.",
            display="uJ",
        ),
        _keyword(
            "peak_power",
            _number("W", minimum=0.0, exclusive=True),
            "basic",
            "Peak power after the spectral phase; give it or the energy.",
            display="GW",
        ),
        _keyword(
            "phase",
            Form("array", elements="phases"),
            "advanced",
            "Taylor coefficients of the spectral phase about the centre wavelength.",
        ),
        _keyword(
            "t0",
            _number("s"),
            "advanced",
            "Delay, positive for later arrival.",
            display="fs",
        ),
        *extra,
    )


def _gradient(*first: Row) -> tuple[Parameter, ...]:
    """Return a pressure gradient's entries: its stages and the flow keywords."""
    return _type(
        "medium",
        f"{_PROFILE_PAGE}#steady-flow-fills",
        *first,
        _keyword(
            "floor",
            _PRESSURE,
            "advanced",
            "Smallest pressure a stage end may take; the default is a fraction of the "
            "highest.",
            display="bar",
        ),
        _keyword(
            "radius",
            (_LENGTH, _PROFILE),
            "advanced",
            "Core radius the flow follows; the default is the uniform bore.",
            display="um",
        ),
        _keyword(
            "weight",
            (_POSITIVE, _PROFILE),
            "advanced",
            "Flow weight multiplying the radius to the minus fourth power.",
        ),
    )


def _fill_rows() -> tuple[Row, ...]:
    """Return the keywords a gas fill shares: its pressure path and the flow."""
    return (
        _keyword(
            "floor",
            _PRESSURE,
            "advanced",
            "Pressure-gradient floor; the default is the standard floor.",
            display="bar",
        ),
        _keyword(
            "length",
            _LENGTH,
            "advanced",
            "Length of the fill, needed beside a pair of end pressures.",
            display="m",
        ),
        _keyword(
            "radius",
            (_LENGTH, _PROFILE),
            "advanced",
            "Core radius the gas flows through, which makes end pressures a steady "
            "flow.",
            display="um",
        ),
        _keyword(
            "vacuum_end",
            _STRING,
            "advanced",
            "How an outer end below its choked pressure is treated.",
        ),
        _keyword(
            "selection",
            _object("MaterialSelection"),
            "advanced",
            "The material selection the species are read from optmat under.",
        ),
    )


_PRESSURE_PATH: Final = (
    _PRESSURE,
    Form("pair", unit="bar", minimum=0.0, length=2),
    _PROFILE,
    _object("PressureGradient.two_point", "PressureGradient.from_stages"),
)
_TEMPERATURE_PATH: Final = (_TEMPERATURE, _PROFILE)


def _grid(*rows: Row) -> tuple[Parameter, ...]:
    """Return a time grid's entries: the window, the band and the sampling.

    - ``*rows``: the rows between the sample spacing and the transform sizes.
    """
    return _type(
        "grid",
        f"{_TYPE_GRID_PAGE}#coarse-and-fine-sampling",
        _keyword("length", _LENGTH, "basic", "Propagation distance.", display="m"),
        _keyword(
            "reference_wavelength",
            _LENGTH,
            "basic",
            "Reference wavelength, usually the pump's.",
            display="nm",
        ),
        _keyword(
            "wavelength_limits",
            Form("pair", unit="m", minimum=0.0, exclusive=True, length=2),
            "basic",
            "Shortest and longest wavelength of the frequency window.",
            display="nm",
        ),
        _keyword(
            "trange",
            _TYPE_DURATION,
            "basic",
            "Extent of the time window.",
            display="fs",
        ),
        _keyword(
            "dt",
            _TYPE_DURATION,
            "advanced",
            "Fine sample spacing; the band's own where it is finer.",
            display="fs",
        ),
        *rows,
        _keyword(
            "fft_size",
            _choice("auto", "pow2", "smooth"),
            "advanced",
            "How the transform sizes are rounded up.",
        ),
        _keyword(
            "collar_fraction",
            _number(minimum=0.0, maximum=0.5),
            "advanced",
            "Width of each temporal collar as a fraction of the window.",
        ),
        _keyword(
            "collar_placement",
            _choice("outside", "inside"),
            "advanced",
            "Whether the collars lie outside the requested window or inside it.",
        ),
    )


def _hollow_rows(cladding: str) -> tuple[Row, ...]:
    """Return a hollow mode's loss and cladding keywords."""
    return (
        _keyword(
            "loss",
            _LOSS,
            "advanced",
            "The model's loss, none, a scale of it, or an absolute loss.",
        ),
        _keyword(
            "cladding",
            _CLADDING,
            "advanced",
            f"The {cladding}'s index; the default is fused silica.",
        ),
    )


def _slab_fill(temperature: Form | tuple[Form, ...]) -> tuple[Row, ...]:
    """Return a slab channel's own fill, which the planar route reads."""
    return (
        _keyword("gas", _GAS, "basic", "The filling gas; none for vacuum."),
        _keyword(
            "pressure",
            (_PRESSURE, _PROFILE),
            "basic",
            "Gas pressure, constant or along z.",
            display="bar",
        ),
        _keyword(
            "polarisation",
            _choice("te", "tm"),
            "advanced",
            "TE (field along y, parallel to the plates) or TM.",
        ),
        *_hollow_rows("plates"),
        _keyword(
            "temperature", temperature, "advanced", "Gas temperature.", display="K"
        ),
    )


def _beam(*extra: Row) -> tuple[Parameter, ...]:
    """Return a transverse beam's entries."""
    return _type(
        "input",
        f"{_TYPE_PULSE_PAGE}#transverse-beams",
        _keyword(
            "temporal",
            _object("GaussianPulse", "SechPulse"),
            "basic",
            "The pulse in time; its energy or peak power is the beam's.",
        ),
        _keyword(
            "waist",
            _LENGTH,
            "basic",
            "The 1/e^2 intensity radius.",
            display="um",
        ),
        _keyword("order", _integer(1), "advanced", "Super-Gaussian order, at least 1."),
        _keyword(
            "propz",
            _POSITION,
            "advanced",
            "Distance the beam is propagated back from its waist at the launch.",
            display="mm",
        ),
        _keyword(
            "jones",
            Form("array"),
            "advanced",
            "Polarisation: one scalar component or the x and y components.",
        ),
        _keyword(
            "displacement",
            Form("pair", unit="m", length=2),
            "advanced",
            "The beam centre's offset in x and y at the launch.",
            display="um",
        ),
        _keyword(
            "tilt",
            Form("pair", unit="rad", length=2),
            "advanced",
            "The beam's tilt towards x and y.",
            display="mrad",
        ),
        *extra,
    )


def _plasma(*extra: Row) -> tuple[Parameter, ...]:
    """Return a plasma response's entries."""
    return _type(
        "nonlinear",
        _IONISATION_PAGE,
        _keyword("gas", _GAS, "basic", "The species ionised."),
        _keyword(
            "rate",
            (_object("OptmatRate"), _UNREGISTERED),
            "basic",
            "The ionisation rate against the field amplitude.",
        ),
        _keyword(
            "preionisation",
            (_FRACTION, _PROFILE),
            "advanced",
            "Fraction already ionised before the pulse, constant or along z.",
        ),
        _keyword("species", _STRING, "advanced", "Component label in a mixture."),
        *extra,
    )


_ENTRIES: Final[dict[str, tuple[Parameter, ...]]] = {
    "GaussianPulse": _pulse(
        _keyword(
            "fwhm",
            _TYPE_DURATION,
            "basic",
            "Full width at half maximum of the power.",
            display="fs",
        ),
        _keyword("m", _integer(1), "advanced", "Super-Gaussian order, at least 1."),
    ),
    "SechPulse": _pulse(
        _keyword(
            "fwhm",
            _TYPE_DURATION,
            "basic",
            "Full width at half maximum of the power; give it or the natural width.",
            display="fs",
        ),
        _keyword(
            "width",
            _TYPE_DURATION,
            "advanced",
            "Natural width of sech^2(t/width); give it or the FWHM.",
            display="fs",
        ),
    ),
    "load_spectrum": _type(
        "input",
        f"{_TYPE_PULSE_PAGE}#measured-spectra-from-a-file",
        _keyword(
            "path",
            Form("mapping"),
            "basic",
            "The spectrum's file, as a checked data-file reference.",
        ),
        _keyword(
            "energy",
            _number("J", minimum=0.0, exclusive=True),
            "basic",
            "Pulse energy to scale to.",
            display="uJ",
        ),
        _keyword(
            "axis",
            _choice("frequency", "wavelength"),
            "basic",
            "What the axis column holds.",
        ),
        _keyword(
            "axis_unit",
            _STRING,
            "basic",
            "Unit of the axis column; the default is the axis's SI unit.",
        ),
        _keyword(
            "columns",
            Form("array"),
            "advanced",
            "Indices of the axis, power and phase columns.",
        ),
        _keyword(
            "phase_unit", _choice("rad", "deg"), "advanced", "Unit of the phase column."
        ),
        _keyword(
            "phase_convention",
            _choice("optics", "fft"),
            "advanced",
            "Sign convention of the phase column.",
        ),
        _keyword(
            "wrapped",
            _TYPE_BOOLEAN,
            "advanced",
            "Whether the phase is wrapped; the default decides from its range.",
        ),
        _keyword(
            "floor",
            _number(minimum=0.0, maximum=1.0),
            "advanced",
            "Share of the peak power below which a sample is noise for the unwrapping.",
        ),
    ),
    "PressureGradient.two_point": _gradient(
        _keyword("length", _LENGTH, "basic", "Length of the fill.", display="m"),
        _keyword(
            "start_bar", _PRESSURE, "basic", "Pressure at the entrance.", display="bar"
        ),
        _keyword("end_bar", _PRESSURE, "basic", "Pressure at the exit.", display="bar"),
    ),
    "PressureGradient.from_stages": _gradient(
        _keyword(
            "positions",
            Form("array", unit="m"),
            "basic",
            "Stage boundaries, strictly increasing.",
            display="m",
        ),
        _keyword(
            "pressures",
            Form("array", unit="bar", minimum=0.0),
            "basic",
            "Pressure at each boundary.",
            display="bar",
        ),
    ),
    "Constant": _type(
        "medium",
        f"{_PROFILE_PAGE}#the-paths",
        _keyword("value", _number(), "basic", "The value, in the consumer's unit."),
        _keyword(
            "span",
            Form("pair", unit="m", length=2),
            "advanced",
            "The interval the path is declared over; the default is any.",
            display="m",
        ),
    ),
    "Linear": _type(
        "medium",
        f"{_PROFILE_PAGE}#the-paths",
        _keyword("start", _number(), "basic", "Value at the entrance."),
        _keyword("end", _number(), "basic", "Value at the end of the section."),
        _keyword(
            "length", _LENGTH, "basic", "Length of the varying section.", display="m"
        ),
    ),
    "PiecewiseLinear": _type(
        "medium",
        f"{_PROFILE_PAGE}#the-paths",
        _keyword(
            "positions",
            Form("array", unit="m"),
            "basic",
            "Positions, strictly increasing.",
            display="m",
        ),
        _keyword("values", Form("array"), "basic", "The value at each position."),
    ),
    "Mixture.fill": _type(
        "medium",
        _MIXTURE_PAGE,
        _keyword(
            "fractions",
            _MAPPING,
            "basic",
            "Mole fraction of each species, summing to one.",
        ),
        _keyword(
            "pressure",
            _PRESSURE_PATH,
            "basic",
            "Total pressure: constant, end pressures, positions and pressures, or a "
            "profile.",
            display="bar",
        ),
        _keyword(
            "temperature",
            _TEMPERATURE_PATH,
            "advanced",
            "Temperature, constant or along z.",
            display="K",
        ),
        *_fill_rows(),
    ),
    "Mixture.partial_pressures": _type(
        "medium",
        _MIXTURE_PAGE,
        _keyword(
            "pressures",
            _MAPPING,
            "basic",
            "Partial pressure of each species in bar, constant or a profile.",
        ),
        _keyword(
            "temperature",
            _TEMPERATURE_PATH,
            "advanced",
            "Temperature, constant or along z.",
            display="K",
        ),
        _keyword(
            "selection",
            _object("MaterialSelection"),
            "advanced",
            "The material selection the species are read from optmat under.",
        ),
    ),
    "Mixture.air": _type(
        "medium",
        f"{_MIXTURE_PAGE}#air",
        _keyword(
            "pressure",
            _PRESSURE_PATH,
            "basic",
            "Total pressure, as a fill's.",
            display="bar",
        ),
        _keyword(
            "temperature",
            _TEMPERATURE_PATH,
            "advanced",
            "Temperature, constant or along z.",
            display="K",
        ),
        _keyword(
            "fractions",
            _MAPPING,
            "advanced",
            "Mole fractions; the default is the selection's named composition.",
        ),
        _keyword(
            "argon",
            _TYPE_BOOLEAN,
            "advanced",
            "Whether the default composition carries argon.",
        ),
        *(row for row in _fill_rows() if row[0] != "floor"),
        _keyword(
            "floor",
            _PRESSURE,
            "advanced",
            "Pressure-gradient floor; the default is the standard floor.",
            display="bar",
        ),
    ),
    "EnvelopeGrid.from_limits": _grid(
        _keyword(
            "thg",
            _TYPE_BOOLEAN,
            "advanced",
            "Leave room for third-harmonic generation.",
        ),
        _keyword(
            "ffac",
            _number(minimum=2.0),
            "advanced",
            "Fine-grid oversampling factor; 6 with third harmonics, else 2.",
        ),
    ),
    "FieldGrid.from_limits": _grid(
        _keyword(
            "ffac",
            _number(minimum=2.0),
            "advanced",
            "Fine-grid oversampling factor; 6 resolves a cubic response.",
        ),
    ),
    "StatsSpec": _type(
        "output",
        "guide/results/statistics.md",
        _keyword(
            "names",
            Form("array"),
            "basic",
            "The statistics evaluated at every saved plane.",
        ),
        _keyword(
            "energy_windows",
            Form("array"),
            "advanced",
            "Spectral windows in rad/s, one (low, high) row each.",
        ),
        _keyword(
            "periodic",
            Form("array"),
            "advanced",
            "The statistics also evaluated between planes.",
        ),
        _keyword(
            "period",
            _integer(1),
            "advanced",
            "Accepted steps between the periodic evaluations.",
        ),
        _keyword(
            "apertures",
            _UNREGISTERED,
            "advanced",
            "Far-field apertures, for the aperture spectra.",
        ),
        _keyword(
            "spectral_regions",
            Form("array"),
            "advanced",
            "Transverse wavevector boxes in rad/m, one row each.",
        ),
    ),
    "BoundaryCheck": _type(
        "numerics",
        f"{_TYPE_BOUNDARY_PAGE}#watching-the-windows-edges",
        _keyword(
            "guard_fraction",
            _number(minimum=0.0, maximum=0.5, exclusive=True),
            "advanced",
            "Width of each edge's guard as a fraction of the window.",
        ),
        _keyword(
            "budget",
            _POSITIVE,
            "advanced",
            "The field-error budget above which the check warns.",
        ),
        _keyword(
            "stop_factor",
            _POSITIVE,
            "advanced",
            "How many budgets stop the run.",
        ),
        _keyword(
            "quiet_collar",
            _TYPE_BOOLEAN,
            "advanced",
            "Leave the in-stage collar out while it is quiet, a speed-up.",
        ),
        _keyword(
            "transverse_budget",
            _POSITIVE,
            "advanced",
            "The budget of a free-space run's transverse edge; the default is the "
            "budget.",
        ),
    ),
    "FixedSchedule.uniform": _type(
        "numerics",
        "guide/numerics/fixed-schedules.md#plain-fixed-steps",
        _keyword("z_start", _POSITION, "basic", "Where the run starts.", display="m"),
        _keyword("z_end", _POSITION, "basic", "Where it ends.", display="m"),
        _keyword("dz", _LENGTH, "basic", "Largest step.", display="mm"),
        _keyword(
            "save_planes",
            Form("array", unit="m"),
            "advanced",
            "Positions the run will save at.",
            display="m",
        ),
        _keyword(
            "hard_kinks",
            Form("array", unit="m"),
            "advanced",
            "Positions where the linear operator jumps or kinks.",
            display="m",
        ),
    ),
    "FixedSchedule": _type(
        "numerics",
        "guide/numerics/fixed-schedules.md",
        _keyword(
            "steps",
            Form("array", unit="m", minimum=0.0, exclusive=True),
            "basic",
            "The step lengths.",
            display="mm",
        ),
        _keyword(
            "z_start",
            _POSITION,
            "advanced",
            "Position the first step starts from.",
            display="m",
        ),
    ),
    "Multistep": _type(
        "numerics",
        "guide/numerics/adams.md",
        _keyword("order", _integer(1, 6), "advanced", "Highest predictor order."),
        _keyword(
            "variable_order",
            _TYPE_BOOLEAN,
            "advanced",
            "Choose the order step by step.",
        ),
        _keyword(
            "max_calls",
            _integer(1),
            "advanced",
            "Budget of source calls for the run; the default is none.",
        ),
        _keyword(
            "interval_cap",
            _TYPE_BOOLEAN,
            "advanced",
            "Cap each step at its operator-table interval.",
        ),
        _keyword(
            "equal_approach",
            _TYPE_BOOLEAN,
            "advanced",
            "Approach a forced landing in two equal steps.",
        ),
        _keyword(
            "max_growth",
            _number(minimum=1.0, exclusive=True),
            "advanced",
            "Largest ratio of one step to the one before.",
        ),
        _keyword(
            "min_factor",
            _number(minimum=0.0, maximum=1.0, exclusive=True),
            "advanced",
            "Smallest step factor after a rejection.",
        ),
        _keyword(
            "safety",
            _number(minimum=0.0, maximum=1.0),
            "advanced",
            "Safety factor on a proposed step.",
        ),
        _keyword(
            "formulation",
            _choice("interaction", "exponential"),
            "advanced",
            "How the source history is integrated.",
        ),
        _keyword(
            "carrier_frame",
            _choice("off", "scalar"),
            "advanced",
            "The carrier phase removed from the history inside each step.",
        ),
    ),
    "RungeKutta": (),
    "LowStorageRungeKutta": _type(
        "numerics",
        f"{_TYPE_STEPPER_PAGE}#the-defaults",
        _keyword("scheme", _choice("rk4ip"), "advanced", "The register scheme."),
    ),
    "absorber": _type(
        "numerics",
        _TYPE_BOUNDARY_PAGE,
        _keyword(
            "boundary_N",
            _POSITIVE,
            "advanced",
            "Absorber strength: how often the collar is applied over the run.",
        ),
        _keyword(
            "boundary_length",
            _LENGTH,
            "advanced",
            "Reference length, in place of the strength.",
            display="m",
        ),
        _keyword(
            "temporal",
            _TYPE_BOOLEAN,
            "advanced",
            "Whether the collar absorbs in time; off is a spectral-only collar.",
        ),
        _keyword(
            "walkoff",
            _TYPE_DURATION,
            "advanced",
            "Group-delay drift the temporal collar must stop, over the grid's length.",
            display="fs",
        ),
    ),
    "TubeLattice": _type(
        "geometry",
        "guide/waveguides/antiresonant.md",
        _keyword(
            "wall_thickness",
            _LENGTH,
            "basic",
            "Wall thickness of a tube.",
            display="nm",
        ),
        _keyword("tube_radius", _LENGTH, "basic", "Outer tube radius.", display="um"),
        _keyword("n_tubes", _integer(3), "basic", "Number of tubes, at least three."),
        _keyword(
            "n_terms",
            _integer(1),
            "advanced",
            "Number of radial and azimuthal fit terms.",
        ),
    ),
    "CoherentWall": _type(
        "geometry",
        "guide/waveguides/coherent-wall.md",
        _keyword(
            "condition_limit",
            _number(minimum=1.0),
            "advanced",
            "Largest admitted condition number of one step's flow action.",
        ),
        _keyword(
            "validity_limit",
            _POSITIVE,
            "advanced",
            "Largest admitted validity ratio, above which the block is refused.",
        ),
        _keyword(
            "sections",
            _integer(1),
            "advanced",
            "Equal sections each operator-table interval is divided into.",
        ),
        _keyword(
            "impedance",
            _choice("mean", "te_tm"),
            "advanced",
            "The impedance-averaged block or the coupled azimuthal one.",
        ),
    ),
    "MarcatiliMode": _type(
        "geometry",
        "guide/waveguides/capillary-model.md",
        _keyword("radius", (_LENGTH, _PROFILE), "basic", "Core radius.", display="um"),
        _keyword("kind", _choice("HE", "TE", "TM"), "basic", "The mode family."),
        _keyword("n", _integer(), "basic", "Azimuthal index."),
        _keyword("m", _integer(1), "basic", "Radial index."),
        _keyword(
            "phi",
            _number("rad"),
            "advanced",
            "Rotation of the mode about the axis.",
            display="rad",
        ),
        _keyword(
            "model",
            _choice("full", "reduced"),
            "advanced",
            "Marcatili's full square root or its leading-order expansion.",
        ),
        *_hollow_rows("cladding"),
    ),
    "RectMode": _type(
        "geometry",
        "guide/waveguides/rectangular.md",
        _keyword(
            "half_width_x",
            (_LENGTH, _PROFILE),
            "basic",
            "Half-width of the core in x.",
            display="um",
        ),
        _keyword(
            "half_width_y",
            (_LENGTH, _PROFILE),
            "basic",
            "Half-width of the core in y.",
            display="um",
        ),
        _keyword("n", _integer(1), "basic", "Mode index along y."),
        _keyword("m", _integer(1), "basic", "Mode index along x."),
        _keyword(
            "polarisation",
            _choice("x", "y"),
            "basic",
            "The transverse electric-field component.",
        ),
        *_hollow_rows("wall"),
    ),
    "SlabMode": _type(
        "geometry",
        _PLANAR_PAGE,
        _keyword(
            "half_gap",
            (_LENGTH, _PROFILE),
            "basic",
            "Half the plate separation.",
            display="um",
        ),
        *_slab_fill((_TEMPERATURE, _PROFILE))[:2],
        _keyword(
            "order", _integer(1), "basic", "Mode order; odd orders are symmetric."
        ),
        *_slab_fill((_TEMPERATURE, _PROFILE))[2:],
    ),
    "ModeCollection.slab": _type(
        "geometry",
        _PLANAR_PAGE,
        _keyword(
            "half_gap",
            (_LENGTH, _PROFILE),
            "basic",
            "Half the plate separation.",
            display="um",
        ),
        _keyword(
            "orders",
            Form("array", minimum=1.0),
            "basic",
            "The mode orders, without repeats, in state order.",
        ),
        *_slab_fill(_TEMPERATURE),
    ),
    "MaterialSelection": _type(
        "medium",
        "guide/media/material-selection.md",
        _keyword(
            "linear",
            (_STRING, _UNREGISTERED),
            "advanced",
            "The linear dispersion model of a gas.",
        ),
        _keyword(
            "electronic",
            _STRING,
            "advanced",
            "The electronic third-order policy.",
        ),
        _keyword("raman", (_STRING, _UNREGISTERED), "advanced", "The Raman response."),
        _keyword(
            "ionisation",
            (_STRING, _UNREGISTERED),
            "advanced",
            "The ionisation rate model.",
        ),
        _keyword(
            "carrier_wavelength",
            _LENGTH,
            "advanced",
            "Carrier wavelength the carrier coefficients are evaluated at.",
            display="nm",
        ),
        _keyword(
            "energy_loss",
            _choice("peak-weighted"),
            "advanced",
            "The plasma energy-loss convention.",
        ),
        _keyword(
            "raman_shift",
            _choice("reference-density"),
            "advanced",
            "Hold a density-dependent Raman shift at a reference density.",
        ),
    ),
    "RadialGrid": _type(
        "grid",
        f"{_FREE_PAGE}",
        _keyword("radius", _LENGTH, "basic", "The aperture modelled.", display="mm"),
        _keyword("n_r", _integer(2), "basic", "Number of radial samples."),
        _keyword(
            "collar_fraction",
            _number(minimum=0.0),
            "advanced",
            "Room outside the aperture as a fraction of it.",
        ),
    ),
    "FreeGrid": _type(
        "grid",
        _FREE_PAGE,
        _keyword("radius_x", _LENGTH, "basic", "Half-width in x.", display="mm"),
        _keyword("n_x", _integer(2), "basic", "Samples in x."),
        _keyword("radius_y", _LENGTH, "basic", "Half-width in y.", display="mm"),
        _keyword("n_y", _integer(2), "basic", "Samples in y."),
        _keyword(
            "collar_fraction",
            _number(minimum=0.0),
            "advanced",
            "Room outside each half-width as a fraction of it.",
        ),
    ),
    "LineGrid": _type(
        "grid",
        _FREE_PAGE,
        _keyword("radius_x", _LENGTH, "basic", "Half-width in x.", display="mm"),
        _keyword("n_x", _integer(2), "basic", "Samples in x."),
        _keyword(
            "collar_fraction",
            _number(minimum=0.0),
            "advanced",
            "Room outside the half-width as a fraction of it.",
        ),
    ),
    "PlanarGrid": _type(
        "grid",
        _PLANAR_PAGE,
        _keyword(
            "modes",
            _object("ModeCollection.slab", "SlabMode"),
            "basic",
            "The slab channels, in state order.",
        ),
        _keyword("radius_y", _LENGTH, "basic", "Half-width in y.", display="mm"),
        _keyword("n_y", _integer(2), "basic", "Samples in y."),
        _keyword(
            "collar_fraction",
            _number(minimum=0.0),
            "advanced",
            "Room outside the half-width in y as a fraction of it.",
        ),
        _keyword(
            "n_x",
            _integer(1),
            "advanced",
            "Nodes of the x rule; the default follows the orders.",
        ),
        _keyword(
            "x_rule",
            _choice("gauss_kronrod", "gauss"),
            "advanced",
            "The x rule's family.",
        ),
    ),
    "SpatialMedium": _type(
        "medium",
        _FREE_PAGE,
        _keyword(
            "material",
            (
                _STRING,
                _number(),
                _object("Mixture.fill", "Mixture.partial_pressures", "Mixture.air"),
                _UNREGISTERED,
            ),
            "basic",
            "A gas name, a glass, a constant index or a mixture.",
        ),
        _keyword(
            "pressure",
            (_PRESSURE, _PROFILE),
            "basic",
            "Gas pressure, constant or along z.",
            display="bar",
        ),
        _keyword(
            "temperature",
            _TEMPERATURE,
            "advanced",
            "Gas temperature; the default is room temperature.",
            display="K",
        ),
    ),
    "KerrResponse": _type(
        "nonlinear",
        "guide/media/nonlinear.md",
        _keyword(
            "chi3",
            (_number(), _object("gamma3")),
            "basic",
            "Third-order susceptibility per unit density.",
        ),
        _keyword(
            "thg",
            _TYPE_BOOLEAN,
            "advanced",
            "Whether the response radiates the third harmonic.",
        ),
        _keyword("species", _STRING, "advanced", "Component label in a mixture."),
    ),
    "StepIndexMode": _type(
        "geometry",
        "guide/waveguides/step-index.md",
        _keyword(
            "radius",
            (_LENGTH, _PROFILE),
            "basic",
            "Core radius.",
            display="um",
        ),
        _keyword(
            "core",
            _FIBRE_MATERIAL,
            "basic",
            "Core index, or a glass; none is fused silica.",
        ),
        _keyword(
            "cladding",
            _FIBRE_MATERIAL,
            "basic",
            "Cladding index, or a glass; none is index 1.",
        ),
        _keyword("kind", _choice("HE", "EH", "TE", "TM"), "basic", "Mode family."),
        _keyword("n", _integer(0), "basic", "Azimuthal order."),
        _keyword("m", _integer(1), "basic", "Radial order."),
        _keyword("parity", _choice("even", "odd"), "advanced", "Hybrid orientation."),
        _keyword(
            "profile_wavelength",
            _LENGTH,
            "basic",
            "Wavelength the transverse profile is frozen at.",
            display="nm",
        ),
        _keyword(
            "loss",
            (_number("m^-1", minimum=0.0), _CALLABLE),
            "advanced",
            "Power attenuation.",
        ),
    ),
    "glass": _type(
        "medium",
        "guide/media/material-selection.md",
        _keyword("name", _STRING, "basic", "A built-in glass, such as SiO2 or BK7."),
        _keyword(
            "model", (_STRING, _UNREGISTERED), "advanced", "An optmat model name."
        ),
        _keyword(
            "axis",
            _choice("o", "e"),
            "advanced",
            "The ray of a uniaxial crystal.",
        ),
    ),
    "IndexMaterial": _type(
        "medium",
        "guide/waveguides/graded-index.md",
        _keyword("index", _FIBRE_MATERIAL, "basic", "Index, or a glass."),
    ),
    "MaterialKerr": _type(
        "nonlinear",
        "guide/waveguides/general-modal.md",
        _keyword("chi3", _number(), "advanced", "Third-order susceptibility, m^2/V^2."),
        _keyword("n2", _number("m^2/W"), "basic", "Nonlinear index."),
        _keyword(
            "at",
            _LENGTH,
            "basic",
            "Wavelength n2 was measured at.",
            display="nm",
        ),
    ),
    "PowerLawPermittivity": _type(
        "geometry",
        "guide/waveguides/graded-index.md",
        _keyword("centre", _object("IndexMaterial"), "basic", "Material on the axis."),
        _keyword(
            "contrast",
            _number(minimum=0.0, maximum=0.5, exclusive=True),
            "basic",
            "Contrast Delta: eps = eps_axis (1 - 2 Delta (r/a)^p).",
        ),
        _keyword("exponent", _number(minimum=2.0), "basic", "Shape exponent p."),
    ),
    "GradedIndexProvider": _type(
        "geometry",
        "guide/waveguides/graded-index.md",
        _keyword(
            "permittivity",
            _object("PowerLawPermittivity"),
            "basic",
            "The core's radial law.",
        ),
        _keyword("exterior", _object("IndexMaterial"), "basic", "Exterior material."),
        _keyword(
            "radius",
            (_LENGTH, _PROFILE),
            "basic",
            "Core radius.",
            display="um",
        ),
        _keyword(
            "family",
            _choice("TE", "TM", "LP"),
            "basic",
            "Sector: TE0, TM0 or the weak-guidance LP approximation.",
        ),
        _keyword("count", _integer(1), "basic", "Consecutive radial channels."),
        _keyword(
            "elements", _integer(8), "advanced", "Radial elements per core radius."
        ),
        _keyword(
            "extent",
            _number(minimum=1.0, exclusive=True),
            "advanced",
            "Matching radius in core radii.",
        ),
        _keyword(
            "azimuthal_order", _integer(0), "basic", "Angular order of an LP sector."
        ),
        _keyword(
            "core_responses",
            _object("MaterialKerr"),
            "basic",
            "The core's responses; empty is a linear guide.",
        ),
    ),
    "ModalGuide.on_grid": _type(
        "geometry",
        "guide/waveguides/graded-index.md",
        _keyword(
            "provider",
            _object("GradedIndexProvider"),
            "basic",
            "The provider that solves the guide's fields.",
        ),
        _keyword("knots", _integer(4), "advanced", "Frequency knots across the band."),
        _keyword("n_r", _integer(1), "advanced", "Radial nodes per element."),
    ),
    "gamma3": _type(
        "nonlinear",
        "guide/media/material-selection.md",
        _keyword("gas", _GAS, "basic", "A built-in gas other than air."),
        _keyword(
            "wavelength",
            _LENGTH,
            "basic",
            "Carrier wavelength the coefficient is taken at.",
            display="nm",
        ),
    ),
    "EnvelopePlasmaResponse": _plasma(),
    "PlasmaResponse": _plasma(
        _keyword(
            "integration",
            _choice("band", "trapezoid"),
            "advanced",
            "The running-integral rule of the electron density.",
        ),
    ),
    "OptmatRate": _type(
        "nonlinear",
        _IONISATION_PAGE,
        _keyword("source", _GAS, "basic", "The species optmat curates a rate for."),
        _keyword(
            "wavelength",
            _LENGTH,
            "basic",
            "Carrier wavelength the rate is taken at.",
            display="nm",
        ),
        _keyword(
            "model",
            _STRING,
            "advanced",
            "The rate model; the default is optmat's recommended one.",
        ),
        _keyword(
            "reference_field",
            _POSITIVE,
            "advanced",
            "Field amplitude in V/m the energy per event is taken at.",
        ),
        _keyword(
            "field_range",
            Form("pair", minimum=0.0, length=2),
            "advanced",
            "Field amplitudes in V/m the rate is tabulated over.",
        ),
        _keyword(
            "negligible_rate",
            _number("s^-1", minimum=0.0),
            "advanced",
            "Rate below which the rate is zero.",
        ),
        _keyword(
            "maximum",
            _number("s^-1", minimum=0.0, exclusive=True),
            "advanced",
            "Ceiling on the rate.",
        ),
    ),
    "SpatialPulse": _beam(),
    "SpatialPulseSum": _type(
        "input",
        "guide/free-space/beams.md#launching-several-beams",
        _keyword(
            "beams",
            _object("SpatialPulse"),
            "basic",
            "The beams launched together, sharing one Jones vector.",
        ),
    ),
    "PlanarPulse": _beam(
        _keyword(
            "waist_x",
            _LENGTH,
            "basic",
            "The 1/e field radius in x.",
            display="um",
        ),
        _keyword(
            "offset_x",
            _POSITION,
            "advanced",
            "The beam centre's offset from the mid-plane in x.",
            display="um",
        ),
    ),
    "SimulationSettings": _type(
        "numerics",
        "guide/setup/running.md",
        _keyword(
            "controller",
            _object("PIController"),
            "advanced",
            "The step-size controller: tolerances, safety and gains.",
        ),
        _keyword(
            "save_planes",
            (Form("array", unit="m", minimum=0.0, exclusive=True), _integer(1)),
            "basic",
            "Positions to save at, or how many equally spaced planes.",
            display="m",
        ),
        _keyword("dz_max", _LENGTH, "advanced", "Largest step.", display="mm"),
        _keyword("dz_init", _LENGTH, "advanced", "First step.", display="mm"),
        _keyword(
            "dz_min",
            _LENGTH,
            "advanced",
            "Smallest step; the default follows the run length.",
            display="mm",
        ),
        _keyword("stretch", _POSITIVE, "advanced", "Landing stretch factor."),
        _keyword(
            "interval_cap",
            _TYPE_BOOLEAN,
            "advanced",
            "Cap each step at its operator-table interval.",
        ),
        _keyword(
            "interpolate_planes",
            _TYPE_BOOLEAN,
            "advanced",
            "Observe the save planes inside accepted steps.",
        ),
        _keyword(
            "step_planes",
            _integer(0),
            "advanced",
            "Also save the end of every k-th accepted step.",
        ),
        _keyword(
            "schedule",
            _object("FixedSchedule.uniform", "FixedSchedule"),
            "advanced",
            "A fixed schedule to replay instead of stepping adaptively.",
        ),
        _keyword(
            "propagator_mode",
            _choice("auto", "tabulate", "recompute"),
            "advanced",
            "Whether the backend tabulates the stage propagators.",
        ),
        _keyword(
            "shock",
            _TYPE_BOOLEAN,
            "advanced",
            "Keep the optical shock term.",
        ),
        _keyword(
            "absorber",
            _object("absorber"),
            "advanced",
            "The temporal and spectral collar.",
        ),
        _keyword(
            "absorber_scheme",
            _choice("split_step", "nonlinear"),
            "advanced",
            "How the temporal collar is applied.",
        ),
        _keyword(
            "ionisation_field_range",
            Form("pair", minimum=0.0, length=2),
            "advanced",
            "Field amplitudes in V/m the ionisation-rate table covers.",
        ),
        _keyword(
            "node_budget",
            _integer(1),
            "advanced",
            "Bytes of node buffers a response pass may hold.",
        ),
        _keyword(
            "stats",
            _object("StatsSpec", "default_spec"),
            "advanced",
            "The statistics collected at every saved plane.",
        ),
        _keyword(
            "polarisation_taper",
            _TYPE_BOOLEAN,
            "advanced",
            "Whether the spectral collar also tapers the nonlinear polarisation.",
        ),
        _keyword(
            "backend",
            _STRING,
            "advanced",
            "The backend that executes the run.",
        ),
        _keyword(
            "integrator",
            _object("Multistep", "RungeKutta", "LowStorageRungeKutta"),
            "advanced",
            "The method that steps the run.",
        ),
        _keyword(
            "evanescent_rate",
            _number("m^-1", minimum=0.0, exclusive=True),
            "advanced",
            "Cap on the evanescent field attenuation.",
        ),
        _keyword(
            "spatial_absorber",
            (_object("SpatialAbsorber"), _TYPE_BOOLEAN),
            "advanced",
            "The transverse boundary; false for none.",
        ),
        _keyword(
            "boundary_check",
            _object("BoundaryCheck"),
            "advanced",
            "How the run watches its time window's edges.",
        ),
    ),
    "PIController": _type(
        "numerics",
        "guide/numerics/solver.md",
        _keyword("rtol", _POSITIVE, "basic", "Relative tolerance."),
        _keyword(
            "atol", _number(minimum=0.0), "basic", "Absolute tolerance, in state units."
        ),
        _keyword(
            "safety",
            _number(minimum=0.0, maximum=1.0),
            "advanced",
            "Factor applied to every proposal.",
        ),
        _keyword(
            "max_factor",
            _number(minimum=1.0, exclusive=True),
            "advanced",
            "Largest growth per step.",
        ),
        _keyword(
            "min_factor",
            _number(minimum=0.0, maximum=1.0, exclusive=True),
            "advanced",
            "Smallest shrink per step.",
        ),
        _keyword("beta1", _POSITIVE, "advanced", "Proportional gain."),
        _keyword("beta2", _number(), "advanced", "Integral gain."),
        _keyword(
            "reject_free",
            _TYPE_BOOLEAN,
            "advanced",
            "Never reject a step whose error is finite.",
        ),
        _keyword(
            "region",
            _choice("active", "band"),
            "advanced",
            "The spectral bins the error norm scores.",
        ),
    ),
    "SpatialAbsorber": _type(
        "numerics",
        f"{_TYPE_BOUNDARY_PAGE}#the-transverse-boundary",
        _keyword(
            "kappa",
            _number("rad/m", minimum=0.0, exclusive=True),
            "advanced",
            "The protected band's edge in transverse wavenumber.",
        ),
        _keyword(
            "angle",
            _number("rad", minimum=0.0, exclusive=True),
            "advanced",
            "The protected band as an angle in vacuum.",
            display="mrad",
        ),
        _keyword(
            "discard_above",
            _number("rad/m", minimum=0.0, exclusive=True),
            "advanced",
            "Wavenumber above which no physical light is declared.",
        ),
        _keyword(
            "attenuation",
            _POSITIVE,
            "advanced",
            "The collar's attenuation at the band's edge, in field nepers.",
        ),
        _keyword(
            "safety_fraction",
            _number(minimum=0.0, maximum=1.0, exclusive=True),
            "advanced",
            "The safety ramp's width as a fraction of each axis's largest wavenumber.",
        ),
        _keyword(
            "real_space",
            _TYPE_BOOLEAN,
            "advanced",
            "Whether a real-space collar is asked for.",
        ),
    ),
    "default_spec": _type(
        "output",
        "guide/results/statistics.md",
        _keyword(
            "kind",
            _choice("mode_averaged", "multimode", "radial", "free", "slab"),
            "basic",
            "Which transform the run uses.",
        ),
        _keyword(
            "with_profiles",
            _TYPE_BOOLEAN,
            "advanced",
            "Also record the gas density, pressure and core radius.",
        ),
        _keyword(
            "with_zdw",
            _TYPE_BOOLEAN,
            "advanced",
            "Also record each mode's zero-dispersion wavelength.",
        ),
    ),
}

_TYPES: Final[Mapping[str, tuple[Parameter, ...]]] = MappingProxyType(_ENTRIES)
"""Each registered type's keyword entries, by the ``type`` a recipe writes (S9 for the
tables): the units are the registry's, the group and anchor the type's. The defaults are
left to :func:`parameters`, which reads them from the target's signature;
``RungeKutta`` has no keyword a recipe gives.
"""


def parameters(route: str) -> Mapping[str, Parameter]:
    """Return the metadata of a route's build keywords, with their defaults.

    Parameters
    ----------
    route : str
        A route of :data:`solprop.recipe.core.ROUTES`, ``"execution"`` for the
        ``[execution]`` table's keys, or a registered type's name
        (:data:`solprop.recipe.registry.REGISTRY`) for the keywords of its table.

    Returns
    -------
    collections.abc.Mapping
        :class:`Parameter` by keyword, in the constructor's order, each with the
        default its signature declares (``REQUIRED`` where it has none). An object
        route's table leaves out :data:`SETTINGS_KEYWORDS`, which a recipe writes
        under ``settings``; a type's leaves out what it receives as its build context
        and the keywords a recipe may not give
        (:attr:`~solprop.recipe.registry.Factory.omitted`).

    Examples
    --------
    >>> from solprop.recipe.metadata import parameters
    >>> table = parameters("gnlse")
    >>> table["rtol"].default is None, table["rtol"].resolver({}).value
    (True, 1e-07)
    >>> parameters("execution")["fields"].default
    'all'
    >>> pulse = parameters("GaussianPulse")
    >>> pulse["fwhm"].display, pulse["m"].default
    ('fs', 1)
    """
    if route in _TYPES:
        return _type_parameters(route, _TYPES[route])
    if route == "execution":
        from solprop.driver.settings import (  # noqa: PLC0415 - layer D
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
    skipped = SETTINGS_KEYWORDS if route in OBJECT_ROUTES else ()
    # Every keyword of the constructor has an entry, as the metadata tests check.
    ordered = {
        name: dataclasses.replace(
            entries[name],
            default=REQUIRED
            if parameter.default is inspect.Parameter.empty
            else parameter.default,
        )
        for name, parameter in signature.parameters.items()
        if name not in skipped
    }
    return MappingProxyType(ordered)


def _type_parameters(
    name: str, entries: Sequence[Parameter]
) -> Mapping[str, Parameter]:
    """Return a registered type's keyword metadata, with its target's defaults."""
    from solprop.recipe.registry import REGISTRY  # noqa: PLC0415 - a cycle

    factory = REGISTRY[name]
    by_name = {entry.name: entry for entry in entries}
    signature = inspect.signature(factory.target)
    target: object = factory.target
    fields = (
        {field.name: field for field in dataclasses.fields(target)}
        if isinstance(target, type) and dataclasses.is_dataclass(target)
        else {}
    )
    keywords = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and parameter.name not in factory.omitted
    ][len(factory.context) :]
    out = {}
    for parameter in keywords:
        field = fields.get(parameter.name)
        if field is not None and field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        elif parameter.default is inspect.Parameter.empty:
            default = REQUIRED
        else:
            default = parameter.default
        # Every keyword of the target has an entry, as the metadata tests check.
        out[parameter.name] = dataclasses.replace(
            by_name[parameter.name], default=default
        )
    return MappingProxyType(out)
