# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The setup pages, and the presentation the front end adds to solprop's metadata.

solprop's parameter metadata says each keyword's group, level, unit, display unit,
range, summary and manual anchor (SEP 41, S9). The front end adds only what is
presentation (design 12.1): which page a group is shown on, and a label where the
keyword's own name does not read well. A test holds every label to a real keyword.
"""

from dataclasses import dataclass
from typing import Final

__all__ = [
    "GAS_TITLES",
    "GROUP_TITLES",
    "LABELS",
    "PAGES",
    "ROUTE_GROUP_TITLES",
    "ROUTE_HELP",
    "TYPE_LABELS",
    "Page",
    "label",
    "page_of",
    "type_label",
]


@dataclass(frozen=True)
class Page:
    """One step of the setup.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        Its title in the step bar.
    groups : tuple of str
        The metadata groups it shows.
    help : str
        What the page sets, in a sentence or two.
    short : str
        Its name under its icon in the step bar, where the title is long.
    """

    key: str
    title: str
    groups: tuple[str, ...]
    help: str
    short: str = ""


PAGES: Final = (
    Page(
        "geometry",
        "Geometry",
        ("geometry",),
        "The waveguide: its kind, its size and its length, and the modes it carries.",
    ),
    Page(
        "medium",
        "Medium",
        ("medium",),
        "What fills the guide: the gas or mixture, its pressure and temperature, and "
        "the guide's loss and cladding.",
    ),
    Page(
        "nonlinear",
        "Nonlinearity",
        ("nonlinear",),
        "The responses the run includes: the Kerr effect, photoionisation and the "
        "plasma, and the molecular Raman response.",
    ),
    Page(
        "input",
        "Input and grid",
        ("input", "grid"),
        "The input pulse, its noise, and the time-frequency grid the run is held on.",
        "Input",
    ),
    Page(
        "propagate",
        "Propagate",
        ("output", "numerics"),
        "What the run saves, how it steps, and the run itself.",
    ),
    Page(
        "results",
        "Results",
        (),
        "The run's results: the spectrum and the power along the guide, a plane, "
        "its spectrogram, and the statistics.",
    ),
    Page(
        "postprocess",
        "Post-process",
        (),
        "Compress the pulse after the run: a chain of linear optics, from a Taylor "
        "phase to chirped mirrors, prisms and gratings, some elements searched by "
        "solprop.",
    ),
)
"""The steps, in order."""

GROUP_TITLES: Final = {
    "geometry": "Waveguide",
    "medium": "Fill",
    "nonlinear": "Responses",
    "input": "Input pulse",
    "grid": "Grid",
    "output": "Output",
    "numerics": "Numerics",
}
"""The section titles of the metadata groups, where a page shows several."""

LABELS: Final = {
    "radius": "Core radius",
    "length": "Length",
    "gas": "Gas",
    "pressure": "Pressure",
    "temperature": "Temperature",
    "mode_average": "Mode-averaged",
    "modes": "Modes",
    "wavelength": "Central wavelength",
    "wavelength_limits": "Wavelength window",
    "trange": "Time window",
    "fwhm": "Duration (FWHM)",
    "energy": "Energy",
    "peak_power": "Peak power",
    "shape": "Shape",
    "kerr": "Kerr effect",
    "plasma": "Photoionisation and plasma",
    "raman": "Raman response",
    "loss": "Loss",
    "n_save": "Saved planes",
    "save_planes": "Save positions",
    "rtol": "Relative tolerance",
    "seed": "Noise seed",
    "betas": "Dispersion (Taylor coefficients)",
    "loss_db_per_m": "Loss (dB/m)",
    "gamma": "Nonlinear coefficient γ",  # noqa: RUF001 - gamma
    "lattice": "Tube lattice",
    "half_width_x": "Half-width in x",
    "half_width_y": "Half-width in y",
    "half_gap": "Half gap",
}
"""Labels for keywords whose names do not read well on their own."""

GAS_TITLES: Final = {
    "Air": "Air",
    "He": "Helium",
    "Ne": "Neon",
    "Ar": "Argon",
    "Kr": "Krypton",
    "Xe": "Xenon",
    "N2": "Nitrogen",
    "H2": "Hydrogen",
    "D2": "Deuterium",
    "O2": "Oxygen",
    "CH4": "Methane",
    "SF6": "Sulphur hexafluoride",
}
"""How the sheets name a gas; one with no entry goes by what a recipe writes."""

TYPE_LABELS: Final = {
    ("FreeGrid", "radius_x"): "Half-width in x",
    ("FreeGrid", "radius_y"): "Half-width in y",
    ("FreeGrid", "n_x"): "Nodes in x",
    ("FreeGrid", "n_y"): "Nodes in y",
    ("LineGrid", "radius_x"): "Half-width in x",
    ("LineGrid", "n_x"): "Nodes in x",
    ("RadialGrid", "radius"): "Grid radius",
    ("RadialGrid", "n_r"): "Radial nodes",
}
"""Labels for keywords of registered types, where the type changes what a name means
(a radial grid's ``radius`` is its extent, a capillary's its core)."""

ROUTE_HELP: Final = {
    ("geometry", "free"): (
        "The beam's transverse grid -- its half-widths and nodes -- and how far the "
        "beam goes."
    ),
    ("geometry", "radial"): (
        "The beam's radial grid -- its radius and nodes -- and how far the beam goes."
    ),
    ("medium", "free"): "What the beam travels through: a gas, a glass or vacuum.",
    ("medium", "radial"): "What the beam travels through: a gas, a glass or vacuum.",
}
"""A step's help where a route makes it read differently, by step and route."""

ROUTE_GROUP_TITLES: Final = {
    ("free", "geometry"): "Beam and grid",
    ("radial", "geometry"): "Beam and grid",
    ("free", "medium"): "Medium",
    ("radial", "medium"): "Medium",
}
"""A section's title where a route makes it read differently, by route and group."""


def label(name: str) -> str:
    """Return a keyword's label.

    Parameters
    ----------
    name : str
        The keyword.

    Returns
    -------
    str
        Its label from :data:`LABELS`, or its name made readable.

    Examples
    --------
    >>> label("fwhm"), label("dz_max")
    ('Duration (FWHM)', 'Dz max')
    """
    if name in LABELS:
        return LABELS[name]
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def type_label(type_name: str, name: str) -> str:
    """Return a registered type's keyword's label.

    Parameters
    ----------
    type_name : str
        The registered type.
    name : str
        The keyword.

    Returns
    -------
    str
        Its label from :data:`TYPE_LABELS`, or :func:`label`'s.

    Examples
    --------
    >>> type_label("FreeGrid", "radius_x"), type_label("RadialGrid", "radius")
    ('Half-width in x', 'Grid radius')
    >>> type_label("GaussianPulse", "fwhm")
    'Duration (FWHM)'
    """
    return TYPE_LABELS.get((type_name, name)) or label(name)


def page_of(group: str) -> Page | None:
    """Return the page a metadata group is shown on.

    Parameters
    ----------
    group : str
        The group.

    Returns
    -------
    Page or None
        The page, or ``None`` for a group no page shows.

    Examples
    --------
    >>> page_of("grid").key
    'input'
    """
    for page in PAGES:
        if group in page.groups:
            return page
    return None
