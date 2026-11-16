# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Machine-readable metadata of every keyword a recipe can set (SEP 41, S9).

One :class:`Parameter` per keyword of a route's ``Simulation.from_*`` constructor, of
its ``prop_*`` function and of the execution settings: its :class:`Form` values (a
keyword often takes several, as ``pressure`` takes a number, a gradient or a profile)
with their units and ranges, the unit a front end shows it in, its group, whether the
resolved recipe writes it back, the rule that resolves a ``None`` default (a
:class:`Resolution`), and where it is documented. :func:`parameters` is the way in: it
reads each default from the signature :func:`surface` names, which stays its single
source, and also answers for a registered type's table, described in
:mod:`solprop.recipe.type_metadata` (imported only then, to break the import cycle
through the registry). :func:`per_unit_width` says when a slab set's energy and power
are per metre of y. A test holds the table and the signatures to the same names. A
front end's tooltips and field validation come from this table; its labels and wording
are its own.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from solprop.driver.defaults import (
    default_scheme,
    describe_default,
    route_row,
)
from solprop.driver.settings import SimulationSettings
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError
from solprop.modes.slab import SlabMode

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

``prop_free`` and ``prop_planar`` apply them to their settings; a
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
_SAVE_PLANES: Final = Form("array", unit="m", minimum=0.0)
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
            "Propagation length, the run's end; none, or left out, beside a grid, "
            "which supplies it.",
            "guide/setup/running.md#one-call",
            display="m",
            none=True,
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
            "Positions to record the run at; the entrance and the end are added.",
            _OUTPUT_PAGE,
            display="mm",
            resolvable=True,
        ),
        Parameter(
            "n_save",
            (Form("integer", minimum=2),),
            "output",
            "basic",
            "Evenly spaced observations over the run, both ends included.",
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
            "alpha",
            (_number("m^-1", minimum=0.0), Form("callable", recipe=False)),
            "medium",
            "advanced",
            "An absolute power attenuation in place of the capillary's own loss.",
            "guide/waveguides/capillary-model.md",
        ),
        Parameter(
            "loss_scale",
            (_number(minimum=0.0),),
            "medium",
            "basic",
            "Factor on the capillary's own loss; zero is lossless.",
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
        ``from_waveguide``, ``from_free`` (the radial and the free routes) or
        ``from_planar``.

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
    builders = {
        "gnlse": Simulation.from_gnlse,
        "capillary": Simulation.from_capillary,
        "waveguide": Simulation.from_waveguide,
        "radial": Simulation.from_free,
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
    # type_metadata imports the registry, which imports this module.
    from solprop.recipe import type_metadata  # noqa: PLC0415 - cycle

    if route in type_metadata._TYPES:
        return type_metadata._type_parameters(route, type_metadata._TYPES[route])
    if route == "execution":
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
