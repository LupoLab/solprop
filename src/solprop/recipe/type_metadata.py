# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Machine-readable metadata of every keyword of a registered type's table (SEP 41, S9).

A recipe's tables name registered types (:data:`solprop.recipe.registry.REGISTRY`): a
second pulse, a pressure gradient, a grid, a mode, the whole ``[build]`` of a
free-space route. Each type's keywords get the
:class:`~solprop.recipe.metadata.Parameter` a route keyword has, so that a front end
builds every table's form from solprop's own description. The rows are ``_TYPES``,
built by ``_type`` and its family helpers (``_pulse``, ``_gradient``, ``_grid``,
``_beam``, ...); :func:`solprop.recipe.metadata.parameters` is the public way in, and
adds each keyword's default from the target's signature (``_type_parameters``). This
module imports the registry, which imports :mod:`solprop.recipe.metadata`, so the
latter imports this one only when a type is asked for. A test holds every form's unit to
the registry's, every keyword to the target's signature and every anchor to the manual.
"""

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from solprop.recipe.metadata import (
    _BOOLEAN,
    _BOUNDARY_PAGE,
    _DURATION,
    _PROFILES,
    _PULSE_PAGE,
    REQUIRED,
    Form,
    Group,
    Level,
    Parameter,
    _number,
)
from solprop.recipe.registry import REGISTRY

__all__: list[str] = []


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
_PRESSURE: Final = _number("bar", minimum=0.0)
_TEMPERATURE: Final = _number("K", minimum=0.0, exclusive=True)
_STRING: Final = Form("string")
_MAPPING: Final = Form("mapping")
_PROFILE: Final = _object(*_PROFILES)
_CALLABLE: Final = Form("callable", recipe=False)
_UNREGISTERED: Final = Form("object", recipe=False)
_ALPHA: Final = (_number("m^-1", minimum=0.0), _CALLABLE)
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

_PROFILE_PAGE: Final = "guide/waveguides/profiles.md"
_MIXTURE_PAGE: Final = "guide/media/mixtures.md"
_TYPE_GRID_PAGE: Final = "guide/setup/grids.md"
_TYPE_STEPPER_PAGE: Final = "guide/numerics/choosing-a-stepper.md"
_FREE_PAGE: Final = "guide/free-space/model.md"
_PLANAR_PAGE: Final = "guide/waveguides/planar.md"
_IONISATION_PAGE: Final = "guide/media/ionisation.md"


def _pulse(*extra: Row) -> tuple[Parameter, ...]:
    """Return a shaped pulse's entries: the shared keywords and its own."""
    return _type(
        "input",
        _PULSE_PAGE,
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
            "phases",
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
            _DURATION,
            "basic",
            "Extent of the time window.",
            display="fs",
        ),
        _keyword(
            "dt",
            _DURATION,
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
            "alpha",
            _ALPHA,
            "advanced",
            "An absolute power attenuation in place of the model's loss.",
        ),
        _keyword(
            "loss_scale",
            _number(minimum=0.0),
            "advanced",
            "Factor on the model's own loss; zero is lossless.",
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
        f"{_PULSE_PAGE}#transverse-beams",
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
        _keyword("m", _integer(1), "advanced", "Super-Gaussian order, at least 1."),
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
            _DURATION,
            "basic",
            "Full width at half maximum of the power.",
            display="fs",
        ),
        _keyword("m", _integer(1), "advanced", "Super-Gaussian order, at least 1."),
    ),
    "SechPulse": _pulse(
        _keyword(
            "fwhm",
            _DURATION,
            "basic",
            "Full width at half maximum of the power; give it or the natural width.",
            display="fs",
        ),
        _keyword(
            "width",
            _DURATION,
            "advanced",
            "Natural width of sech^2(t/width); give it or the FWHM.",
            display="fs",
        ),
    ),
    "load_spectrum": _type(
        "input",
        f"{_PULSE_PAGE}#measured-spectra-from-a-file",
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
            _BOOLEAN,
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
            "start_pressure",
            _PRESSURE,
            "basic",
            "Pressure at the entrance.",
            display="bar",
        ),
        _keyword(
            "end_pressure", _PRESSURE, "basic", "Pressure at the exit.", display="bar"
        ),
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
            _BOOLEAN,
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
            _BOOLEAN,
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
        f"{_BOUNDARY_PAGE}#watching-the-windows-edges",
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
            _BOOLEAN,
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
        _keyword(
            "interval",
            _LENGTH,
            "advanced",
            "The longest distance a quiet window goes unchecked; the default is the "
            "length over 256.",
            display="mm",
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
            _BOOLEAN,
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
            _BOOLEAN,
            "advanced",
            "Cap each step at its operator-table interval.",
        ),
        _keyword(
            "equal_approach",
            _BOOLEAN,
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
        _BOUNDARY_PAGE,
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
            _BOOLEAN,
            "advanced",
            "Whether the collar absorbs in time; off is a spectral-only collar.",
        ),
        _keyword(
            "walkoff",
            _DURATION,
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
            _BOOLEAN,
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
            "alpha",
            (_number("m^-1", minimum=0.0), _CALLABLE),
            "advanced",
            "Power attenuation coefficient.",
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
            Form("array", unit="m", minimum=0.0),
            "basic",
            "Positions to record the run at; the entrance and the end are added.",
            display="m",
        ),
        _keyword(
            "n_save",
            _integer(2),
            "basic",
            "Evenly spaced observations over the run, both ends included.",
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
            _BOOLEAN,
            "advanced",
            "Cap each step at its operator-table interval.",
        ),
        _keyword(
            "interpolate_planes",
            _BOOLEAN,
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
            _BOOLEAN,
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
            _BOOLEAN,
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
            (_object("SpatialAbsorber"), _BOOLEAN),
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
            _BOOLEAN,
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
        f"{_BOUNDARY_PAGE}#the-transverse-boundary",
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
            _BOOLEAN,
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
            _BOOLEAN,
            "advanced",
            "Also record the gas density, pressure and core radius.",
        ),
        _keyword(
            "with_zdw",
            _BOOLEAN,
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


def _type_parameters(
    name: str, entries: Sequence[Parameter]
) -> Mapping[str, Parameter]:
    """Return a registered type's keyword metadata, with its target's defaults."""
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
