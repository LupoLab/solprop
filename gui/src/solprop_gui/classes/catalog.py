# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The first four problem classes (design 6.1), and how a recipe finds its class.

Every other route stays under Other geometry on the generic form. A test holds each
sheet's paths to its class's example recipe, each statistic to solprop's catalogue and
each derived number to the kernel's summary.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from solprop_gui.classes.problem import (
    AnalysisPage,
    Derived,
    Field,
    ProblemClass,
    Section,
    Target,
)

__all__ = [
    "ANTIRESONANT",
    "CAPILLARY",
    "CARTESIAN",
    "CLASSES",
    "FILAMENT",
    "GENERIC_PAGES",
    "LINE",
    "OTHER_KEY",
    "SOLID_CORE",
    "class_named",
    "class_of",
    "other_class",
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
_GAS = Section("Gas", (Field("gas", choices="gases"), Field("pressure")))
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
    Derived(
        "soliton_order",
        "Soliton order",
        prefixed=False,
        target=Target("soliton_order", "order", None, ("energy", "peak_power")),
    ),
    Derived(
        "zdw",
        "Zero-dispersion wavelength",
        "m",
        target=Target("zdw", "wavelength", "nm", ("pressure", "radius")),
    ),
    Derived(
        "rdw",
        "Dispersive wave",
        "m",
        target=Target("rdw", "rdw", "nm", ("pressure", "radius")),
    ),
    Derived("peak_power", "Peak power", "W"),
    Derived("peak_intensity", "Peak intensity", "W/cm²", 1e-4, prefixed=False),
    Derived("fission_length", "Fission length", "m"),
    Derived("loss", "Loss at the pump", "dB/m", prefixed=False),
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
    numbers="capillary_sheet",
    curve="capillary_curve",
    preview="Group-velocity dispersion at the entrance",
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
    # A fibre has many zeros and resonances, so only the order is offered as a target.
    derived=(
        _GUIDE_NUMBERS[0],
        Derived("zdw", "Zero-dispersion wavelength", "m"),
        Derived("rdw", "Dispersive wave", "m"),
        *_GUIDE_NUMBERS[3:],
    ),
    numbers="capillary_sheet",
    curve="capillary_curve",
    preview="Group-velocity dispersion at the entrance",
    pages=(_OVERVIEW, _PLANE, _SPECTROGRAM, _ALONG, _MODES, _COMPRESS, _HEALTH),
    statistics=("energy", "peak_intensity"),
)

FILAMENT: Final = ProblemClass(
    key="filament",
    title="Radial filament",
    summary="A beam in free space on a radial grid: self-focusing and plasma.",
    route="radial",
    example="filament",
    sections=(
        Section(
            "Medium",
            (
                Field(
                    "medium.material",
                    "Gas",
                    choices="gases",
                    also=(
                        "responses[*].chi3.gas",
                        "responses[*].gas",
                        "responses[*].rate.source",
                    ),
                ),
                Field("medium.pressure", "Pressure"),
                Field("grid.length", "Propagate", moves_planes="settings.save_planes"),
            ),
        ),
        Section(
            "Beam",
            (
                Field("initial.waist", "Waist"),
                Field("initial.propz", "To focus", negated=True),
            ),
        ),
        Section(
            "Pulse",
            (
                Field(
                    "initial.temporal.wavelength",
                    "Wavelength",
                    also=(
                        "grid.reference_wavelength",
                        "responses[*].chi3.wavelength",
                        "responses[*].rate.wavelength",
                    ),
                ),
                Field("initial.temporal.fwhm", "Duration"),
                Field("initial.temporal.energy", "Energy"),
            ),
        ),
        Section(
            "Physics",
            (
                Field("responses", "Kerr", member="KerrResponse"),
                Field(
                    "responses",
                    "Plasma",
                    member="EnvelopePlasmaResponse",
                    goes_with=("settings.ionisation_field_range",),
                ),
            ),
        ),
        Section(
            "Resolution",
            (
                Field("space.radius", "Grid radius"),
                Field("space.n_r", "Radial nodes"),
                Field("grid.wavelength_limits", "Wavelengths"),
                Field("grid.trange", "Time window"),
                Field(
                    "settings.save_planes", "Saved planes", planes_over="grid.length"
                ),
            ),
            folded=True,
        ),
    ),
    derived=(
        Derived("power_ratio", "P / P_cr", prefixed=False),
        Derived("critical_power", "Critical power", "W"),
        Derived("peak_power", "Peak power", "W"),
        Derived("rayleigh_range", "Rayleigh range", "m"),
        Derived("collapse_distance", "Collapse (Marburger)", "m"),
    ),
    numbers="beam_sheet",
    curve="beam_curve",
    lens=("initial.waist", "initial.propz", "initial.temporal.wavelength"),
    preview="Linear beam radius along z",
    pages=(
        AnalysisPage("overview", "Overview", ("fields",), view="beam_overview"),
        AnalysisPage("plane", "At this plane", ("fields",), view="beam_plane"),
        _ALONG,
        AnalysisPage("axis", "On axis", ("fields",), view="on_axis"),
        AnalysisPage("farfield", "Far field", ("fields",), view="far_field"),
        _HEALTH,
    ),
    statistics=(
        "peak_intensity",
        "electron_density",
        "beam_radius",
        "energy",
    ),
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
                Field("length", also=("grid.length",)),
            ),
        ),
        Section(
            "Pulse",
            (
                Field("wavelength", "Wavelength", also=("grid.reference_wavelength",)),
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
    derived=(
        Derived("soliton_order", "Soliton order", prefixed=False),
        Derived("zdw", "Zero-dispersion wavelength", "m"),
        Derived("energy", "Energy", "J"),
        Derived("dispersion_length", "Dispersion length", "m"),
        Derived("nonlinear_length", "Nonlinear length", "m"),
        Derived("fission_length", "Fission length", "m"),
    ),
    numbers="fibre_sheet",
    curve="fibre_curve",
    preview="Group-velocity dispersion against wavelength",
    pages=(_OVERVIEW, _PLANE, _SPECTROGRAM, _ALONG, _COMPRESS, _NOISE, _HEALTH),
    statistics=("energy",),
)

CARTESIAN: Final = ProblemClass(
    key="cartesian",
    title="Cartesian beam",
    summary="A beam in free space on an x-y grid: a shape of any symmetry.",
    route="free",
    example="cartesian_beam",
    marker=("space.type=FreeGrid",),
    sections=(
        Section(
            "Medium",
            (
                Field(
                    "medium.material",
                    "Gas",
                    choices="gases",
                    also=("responses[*].chi3.gas",),
                ),
                Field("medium.pressure", "Pressure"),
                Field("grid.length", "Propagate", moves_planes="settings.save_planes"),
            ),
        ),
        Section(
            "Beam",
            (
                Field("initial.waist", "Waist"),
                Field("initial.propz", "To focus", negated=True),
            ),
        ),
        Section(
            "Pulse",
            (
                Field(
                    "initial.temporal.wavelength",
                    "Wavelength",
                    also=("grid.reference_wavelength", "responses[*].chi3.wavelength"),
                ),
                Field("initial.temporal.fwhm", "Duration"),
                Field("initial.temporal.energy", "Energy"),
            ),
        ),
        Section(
            "Physics",
            (Field("responses", "Kerr", member="KerrResponse"),),
        ),
        Section(
            "Grid",
            (
                Field("space.radius_x", "Half-width in x"),
                Field("space.n_x", "Nodes in x"),
                Field("space.radius_y", "Half-width in y"),
                Field("space.n_y", "Nodes in y"),
                Field("grid.wavelength_limits", "Wavelengths"),
                Field("grid.trange", "Time window"),
                Field(
                    "settings.save_planes", "Saved planes", planes_over="grid.length"
                ),
            ),
            folded=True,
        ),
    ),
    derived=(
        Derived("power_ratio", "P / P_cr", prefixed=False),
        Derived("critical_power", "Critical power", "W"),
        Derived("peak_power", "Peak power", "W"),
        Derived("rayleigh_range", "Rayleigh range", "m"),
        Derived("collapse_distance", "Collapse (Marburger)", "m"),
    ),
    numbers="beam_sheet",
    curve="beam_curve",
    lens=("initial.waist", "initial.propz", "initial.temporal.wavelength"),
    preview="Linear beam radius along z",
    pages=(
        _OVERVIEW,
        AnalysisPage("across", "Across the beam", ("fields",), view="fluence_xy"),
        _PLANE,
        _ALONG,
        _HEALTH,
    ),
    statistics=("energy",),
)

LINE: Final = ProblemClass(
    key="line",
    title="Line beam",
    summary="A beam in free space resolved in x alone, per metre of y.",
    route="free",
    example="line_beam",
    marker=("space.type=LineGrid",),
    sections=(
        CARTESIAN.sections[0],
        Section(
            "Beam",
            (
                Field("initial.waist", "Waist"),
                Field("initial.propz", "To focus", negated=True),
            ),
        ),
        CARTESIAN.sections[2],
        CARTESIAN.sections[3],
        Section(
            "Grid",
            (
                Field("space.radius_x", "Half-width in x"),
                Field("space.n_x", "Nodes in x"),
                Field("grid.wavelength_limits", "Wavelengths"),
                Field("grid.trange", "Time window"),
                Field(
                    "settings.save_planes", "Saved planes", planes_over="grid.length"
                ),
            ),
            folded=True,
        ),
    ),
    derived=(
        Derived("energy", "Energy per metre of y", "J/m"),
        Derived("peak_power", "Peak power per metre of y", "W/m"),
    ),
    preview="",
    pages=(
        _OVERVIEW,
        AnalysisPage("across", "Across the beam", ("fields",), view="fluence_xz"),
        _PLANE,
        _ALONG,
        _HEALTH,
    ),
    statistics=("energy", "beam_radius"),
)

CLASSES: Final = (CAPILLARY, FILAMENT, SOLID_CORE, ANTIRESONANT, CARTESIAN, LINE)
"""The classes Home offers, in its order."""

GENERIC_PAGES: Final = (_OVERVIEW, _PLANE, _ALONG, _HEALTH)
"""The rail of a file whose route is no class's."""

OTHER_KEY: Final = "other"
"""The key of Other geometry's class, made for each route it holds."""

_OTHER_SECTIONS: Final = (
    ("Geometry", ("geometry",), False),
    ("Medium", ("medium",), False),
    ("Nonlinearity", ("nonlinear",), False),
    ("Input", ("input",), False),
    ("Grid", ("grid",), True),
    ("Output", ("output",), True),
)
"""The generic form's sections: title, metadata groups and whether folded."""


def other_class(
    route: str,
    title: str,
    build: Mapping[str, object],
    parameters: Sequence[Mapping[str, object]] = (),
) -> ProblemClass:
    """Return a route's class under Other geometry, its form from solprop's metadata.

    A section holds the keywords of its groups that the recipe sets; the rest are under
    Advanced, as on every sheet. With no metadata yet the form is empty.

    Parameters
    ----------
    route : str
        The route.
    title : str
        What to call it: the recipe's title, or the route's.
    build : Mapping
        The recipe's ``[build]`` table.
    parameters : Sequence of Mapping, optional
        The route's keywords, as the kernel describes them, each with its ``group``.

    Returns
    -------
    ProblemClass
        The class, with the generic pages and solprop's summary for its numbers.

    Examples
    --------
    >>> keywords = [{"name": "width", "group": "geometry"},
    ...             {"name": "rtol", "group": "numerics"},
    ...             {"name": "length", "group": "geometry"}]
    >>> made = other_class("free", "A beam", {"width": 1, "rtol": 1e-6}, keywords)
    >>> [(s.title, [f.path for f in s.fields]) for s in made.sections]
    [('Geometry', ['width'])]
    >>> made.key, made.title
    ('other', 'A beam')
    >>> [page.key for page in other_class("planar", "P", {}).pages]
    ['plane', 'along', 'health']
    """
    sections = []
    for name, groups, folded in _OTHER_SECTIONS:
        fields = tuple(
            Field(str(entry["name"]))
            for entry in parameters
            if entry.get("group") in groups and entry.get("name") in build
        )
        if fields:
            sections.append(Section(name, fields, folded=folded))
    return ProblemClass(
        key=OTHER_KEY,
        title=title,
        summary="Every other route, on the form solprop's metadata gives.",
        route=route,
        example="",
        sections=tuple(sections),
        derived=(
            Derived("energy", "Energy", "J"),
            Derived("peak_power", "Peak power", "W"),
        ),
        preview="",
        # A hybrid planar run's planes are slab amplitudes, which no map reads yet.
        pages=GENERIC_PAGES[1:] if route == "planar" else GENERIC_PAGES,
    )


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


def _has(build: Mapping[str, object], mark: str) -> bool:
    """Return whether a recipe's ``[build]`` table carries a class's mark.

    Parameters
    ----------
    build : Mapping
        The table.
    mark : str
        A keyword it sets, or ``"keyword.type=Name"``: an object of that type.

    Returns
    -------
    bool
        Whether it does.

    Examples
    --------
    >>> _has({"lattice": {}}, "lattice"), _has({}, "lattice")
    (True, False)
    >>> space = {"space": {"type": "FreeGrid"}}
    >>> _has(space, "space.type=FreeGrid"), _has(space, "space.type=LineGrid")
    (True, False)
    >>> _has({"space": "x"}, "space.type=FreeGrid")
    False
    """
    path, _, wanted = mark.partition("=")
    if not wanted:
        return path in build
    keyword, _, field = path.partition(".")
    held = build.get(keyword)
    return isinstance(held, Mapping) and held.get(field) == wanted


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
    marked = [p for p in fits if p.marker and all(_has(build, k) for k in p.marker)]
    if marked:
        return marked[0]
    return next((problem for problem in fits if not problem.marker), None)
