# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The first four problem classes (design 6.1), and how a recipe finds its class.

Every other route stays under Other geometry on the generic form. A test holds each
sheet's paths to its class's example recipe, each statistic to solprop's catalogue and
each derived number to the kernel's summary.
"""

from collections.abc import Mapping
from typing import Final

from solprop_gui.classes.problem import (
    AnalysisPage,
    Derived,
    Field,
    ProblemClass,
    Section,
)

__all__ = [
    "ANTIRESONANT",
    "CAPILLARY",
    "CLASSES",
    "FILAMENT",
    "GENERIC_PAGES",
    "SOLID_CORE",
    "class_named",
    "class_of",
]

_OVERVIEW = AnalysisPage("overview", "Overview", ("fields",))
_PLANE = AnalysisPage("plane", "At this plane", ("fields",))
_SPECTROGRAM = AnalysisPage("spectrogram", "Spectrogram", ("fields",))
_ALONG = AnalysisPage("along", "Along z", ("statistics",))
_MODES = AnalysisPage("modes", "Modes", ("fields", "modes"))
_COMPRESS = AnalysisPage("compress", "Compress", ("fields",), "compression")
_NOISE = AnalysisPage("noise", "Noise and coherence", ("fields", "ensemble"))
_HEALTH = AnalysisPage("health", "Run health")

_PULSE = Section(
    "Pulse",
    (
        Field("wavelength", "Wavelength"),
        Field("fwhm", "Duration"),
        Field("energy"),
    ),
)
_GAS = Section("Gas", (Field("gas"), Field("pressure")))
_PHYSICS = Section(
    "Physics",
    (Field("kerr", "Kerr"), Field("plasma", "Plasma"), Field("raman", "Raman")),
)
_WINDOW = Section(
    "Simulation window",
    (
        Field("wavelength_limits", "Wavelengths"),
        Field("trange", "Time window"),
        Field("n_save", "Saved planes"),
    ),
    folded=True,
)
_GUIDE_NUMBERS = (
    Derived("soliton_order", "Soliton order", prefixed=False),
    Derived("zdw", "Zero-dispersion wavelength", "m"),
    Derived("rdw", "Dispersive wave", "m"),
    Derived("peak_power", "Peak power", "W"),
    Derived("peak_intensity", "Peak intensity", "W/cm²", 1e-4, prefixed=False),
    Derived("fission_length", "Fission length", "m"),
)

CAPILLARY: Final = ProblemClass(
    key="capillary",
    title="Capillary",
    summary="A gas-filled hollow capillary, mode-averaged or with several modes.",
    route="capillary",
    example="rdw",
    sections=(
        Section(
            "Capillary",
            (Field("radius", "Core radius"), Field("length"), Field("modes")),
        ),
        _GAS,
        _PULSE,
        _PHYSICS,
        _WINDOW,
    ),
    derived=_GUIDE_NUMBERS,
    preview="Group-velocity dispersion against wavelength",
    pages=(_OVERVIEW, _PLANE, _SPECTROGRAM, _ALONG, _MODES, _COMPRESS, _HEALTH),
    statistics=("energy", "peak_intensity", "electron_density"),
)

ANTIRESONANT: Final = ProblemClass(
    key="antiresonant",
    title="Antiresonant fibre",
    summary="A gas-filled hollow fibre with a tube lattice, and its wall loss.",
    route="capillary",
    example="antiresonant_fibre",
    marker=("lattice",),
    sections=(
        Section(
            "Fibre",
            (
                Field("radius", "Core radius"),
                Field("length"),
                Field("modes"),
                Field("lattice.n_tubes", "Tubes"),
                Field("lattice.tube_radius", "Tube radius"),
                Field("lattice.wall_thickness", "Wall thickness"),
            ),
        ),
        _GAS,
        _PULSE,
        _PHYSICS,
        _WINDOW,
    ),
    derived=(Derived("peak_power", "Peak power", "W"),),
    preview="Group-velocity dispersion and loss against wavelength",
    pages=(_OVERVIEW, _PLANE, _SPECTROGRAM, _ALONG, _MODES, _COMPRESS, _HEALTH),
    statistics=("energy", "peak_intensity"),
)

FILAMENT: Final = ProblemClass(
    key="filament",
    title="Radial filament",
    summary="A beam in free space on a radial grid: self-focusing and plasma.",
    route="radial",
    example="radial_kerr",
    sections=(
        Section(
            "Medium",
            (
                Field("medium.material", "Gas"),
                Field("medium.pressure", "Pressure"),
                Field("grid.length", "Propagate"),
            ),
        ),
        Section(
            "Beam",
            (
                Field("initial.waist", "Waist"),
                Field("initial.propz", "To focus"),
            ),
        ),
        Section(
            "Pulse",
            (
                Field("initial.temporal.wavelength", "Wavelength"),
                Field("initial.temporal.fwhm", "Duration"),
                Field("initial.temporal.energy", "Energy"),
            ),
        ),
        Section(
            "Resolution",
            (
                Field("space.radius", "Grid radius"),
                Field("space.n_r", "Radial nodes"),
                Field("grid.wavelength_limits", "Wavelengths"),
                Field("grid.trange", "Time window"),
            ),
            folded=True,
        ),
    ),
    derived=(Derived("peak_power", "Peak power", "W"),),
    preview="Linear beam radius along z",
    pages=(
        _OVERVIEW,
        _PLANE,
        _ALONG,
        AnalysisPage("axis", "On axis", ("fields",)),
        AnalysisPage("farfield", "Far field", ("fields",)),
        _HEALTH,
    ),
    statistics=("energy", "beam_radius", "peak_intensity", "electron_density"),
)

SOLID_CORE: Final = ProblemClass(
    key="solid-core",
    title="Solid-core fibre",
    summary="A fibre set by its γ and Taylor coefficients, in the GNLSE.",  # noqa: RUF001 - gamma
    route="gnlse",
    example="soliton",
    sections=(
        Section(
            "Fibre",
            (
                Field("gamma", "Nonlinear coefficient"),
                Field("betas", "Dispersion"),
                Field("length"),
            ),
        ),
        Section(
            "Pulse",
            (
                Field("wavelength", "Wavelength"),
                Field("width", "Width"),
                Field("peak_power"),
                Field("shape"),
            ),
        ),
        Section(
            "Physics",
            (Field("raman", "Raman"), Field("shock", "Self-steepening")),
        ),
        Section(
            "Simulation window",
            (
                Field("grid.wavelength_limits", "Wavelengths"),
                Field("grid.trange", "Time window"),
                Field("n_save", "Saved planes"),
            ),
            folded=True,
        ),
    ),
    derived=(Derived("peak_power", "Peak power", "W"),),
    preview="Group-velocity dispersion against wavelength",
    pages=(_OVERVIEW, _PLANE, _SPECTROGRAM, _ALONG, _COMPRESS, _NOISE, _HEALTH),
    statistics=("energy",),
)

CLASSES: Final = (CAPILLARY, FILAMENT, SOLID_CORE, ANTIRESONANT)
"""The classes Home offers, in its order."""

GENERIC_PAGES: Final = (_OVERVIEW, _PLANE, _ALONG, _HEALTH)
"""The rail of a file whose route is no class's."""


def class_named(key: str) -> ProblemClass:
    """Return a class by its key.

    Parameters
    ----------
    key : str
        The key.

    Returns
    -------
    ProblemClass
        The class.

    Raises
    ------
    KeyError
        If no class has that key.

    Examples
    --------
    >>> class_named("filament").route
    'radial'
    """
    for problem in CLASSES:
        if problem.key == key:
            return problem
    raise KeyError(key)


def class_of(route: str, build: Mapping[str, object]) -> ProblemClass | None:
    """Return the class a recipe belongs to.

    A class with a marker wins over one of the same route without, so a capillary with
    a lattice is an antiresonant fibre.

    Parameters
    ----------
    route : str
        The recipe's route.
    build : Mapping
        Its ``[build]`` table.

    Returns
    -------
    ProblemClass or None
        The class, or ``None`` for a route under Other geometry.

    Examples
    --------
    >>> class_of("capillary", {"radius": "125 um"}).key
    'capillary'
    >>> class_of("capillary", {"lattice": {"type": "TubeLattice"}}).key
    'antiresonant'
    >>> class_of("free", {}) is None
    True
    """
    fits = [problem for problem in CLASSES if problem.route == route]
    marked = [p for p in fits if p.marker and all(k in build for k in p.marker)]
    if marked:
        return marked[0]
    return next((problem for problem in fits if not problem.marker), None)
