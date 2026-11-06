# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The geometry classes' pictures, for the start page's gallery.

Each class has a simple cross-section drawn for the gallery (``figures/<key>.svg``),
after the user guide's geometry figures but reduced to what tells the classes apart.
They are styled with CSS custom properties, a light palette and a dark one under
``prefers-color-scheme``, which Qt's SVG renderer does not read: :func:`resolved` writes
the chosen palette's colours in their place, so the gallery follows the window's theme.
"""

import re
from importlib import resources
from typing import Final

__all__ = ["FIGURES", "figure_text", "resolved"]

FIGURES: Final = (
    "step-index",
    "grin",
    "capillary-averaged",
    "capillary-modal",
    "antiresonant",
    "gnlse",
    "rectangular",
    "slab",
    "cartesian",
    "line",
    "radial",
)
"""The geometry classes with a picture, by their cards' keys."""

_PALETTE = re.compile(r"svg\s*\{([^}]*)\}")
_PROPERTY = re.compile(r"--([\w-]+):\s*([^;]+);")
_DARK = re.compile(
    r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{\s*svg\s*\{[^}]*\}\s*\}"
)
_VARIABLE = re.compile(r"var\(--([\w-]+)\)")


def resolved(text: str, *, dark: bool = False) -> str:
    """Return a figure's SVG with its palette's colours written in for Qt.

    Parameters
    ----------
    text : str
        The figure, as the guide's script draws it.
    dark : bool, optional
        Whether to take the dark palette.

    Returns
    -------
    str
        The SVG without custom properties: each ``var(--name)`` replaced by the
        colour, and the palettes removed.

    Examples
    --------
    >>> svg = (
    ...     "<style>svg { --ink: #111; } @media (prefers-color-scheme: dark) "
    ...     "{ svg { --ink: #eee; } } .o { stroke: var(--ink); }</style>"
    ... )
    >>> resolved(svg).split("{")[-1].strip()
    'stroke: #111; }</style>'
    >>> "#eee" in resolved(svg, dark=True), "var(" in resolved(svg)
    (True, False)
    >>> resolved("<style>svg { --a: #123; } .o { fill: var(--a); }</style>", dark=True)
    '<style> .o { fill: #123; }</style>'
    """
    palettes = _PALETTE.findall(text)
    colours = dict(_PROPERTY.findall(palettes[0])) if palettes else {}
    if dark:
        dark_block = _DARK.search(text)
        if dark_block is not None:
            colours.update(_PROPERTY.findall(dark_block.group(0)))
    text = _DARK.sub("", text)
    text = _PALETTE.sub("", text, count=1)
    return _VARIABLE.sub(lambda match: colours[match.group(1)].strip(), text)


def figure_text(key: str, *, dark: bool = False) -> str | None:
    """Return a geometry class's figure, resolved for Qt.

    Parameters
    ----------
    key : str
        The card's key.
    dark : bool, optional
        Whether to take the dark palette.

    Returns
    -------
    str or None
        The SVG; ``None`` for a class without a figure.

    Examples
    --------
    >>> "var(" in figure_text("antiresonant"), figure_text("nope") is None
    (False, True)
    """
    if key not in FIGURES:
        return None
    source = resources.files("solprop_gui.help") / "figures" / f"{key}.svg"
    return resolved(source.read_text(encoding="utf-8"), dark=dark)
