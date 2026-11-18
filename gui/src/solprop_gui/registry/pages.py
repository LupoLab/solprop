# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The labels the front end adds to solprop's metadata.

solprop's parameter metadata says each keyword's group, level, unit, display unit,
range, summary and manual anchor (SEP 41, S9). The front end adds only what is
presentation (design 12.1): a label where the keyword's own name does not read well,
and how the sheets name a gas. A test holds every label to a real keyword.
"""

from typing import Final

__all__ = ["GAS_TITLES", "LABELS", "TYPE_LABELS", "label", "type_label"]


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
    "alpha": "Attenuation (replaces the loss)",
    "loss_scale": "Loss scale",
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
