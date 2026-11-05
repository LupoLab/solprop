# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Home's tile drawings: each geometry as a simple cross-section or beam, two-toned.

The drawings are small SVGs kept here as text, so the front end ships no image files,
in the accent colour over a pale tint of it. Qt's SVG renderer reads presentation
attributes, not CSS, so the colours are written into each drawing.
"""

from typing import Final

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from solprop_gui.ui.theme import ACCENT

__all__ = ["ART", "TINT", "tile_pixmap"]

TINT: Final = "#dcebf3"
"""The pale fill the drawings put behind the accent."""

_RING = "fill:{tint};stroke:{accent};stroke-width:2"


def _tubes() -> str:
    """Return the six tubes of an antiresonant fibre, around its core.

    Returns
    -------
    str
        The SVG circles.
    """
    places = ((24.0, 11.0), (35.3, 17.5), (35.3, 30.5), (24.0, 37.0), (12.7, 30.5),
              (12.7, 17.5))  # fmt: skip
    return "".join(
        f'<circle cx="{x}" cy="{y}" r="5.2" fill="#fff" stroke="{{accent}}" '
        'stroke-width="1.6"/>'
        for x, y in places
    )


ART: Final = {
    "capillary": (
        '<circle cx="24" cy="24" r="19" fill="{tint}" stroke="{accent}" '
        'stroke-width="2"/>'
        '<circle cx="24" cy="24" r="12" fill="#fff" stroke="{accent}" '
        'stroke-width="1.6"/>'
        '<circle cx="20" cy="21" r="1.6" fill="{accent}" opacity="0.45"/>'
        '<circle cx="27" cy="25" r="1.6" fill="{accent}" opacity="0.45"/>'
        '<circle cx="22" cy="28" r="1.6" fill="{accent}" opacity="0.45"/>'
    ),
    "antiresonant": (
        '<circle cx="24" cy="24" r="21" fill="{tint}" stroke="{accent}" '
        'stroke-width="2"/>'
        '<circle cx="24" cy="24" r="17.5" fill="#fff" stroke="{accent}" '
        'stroke-width="1"/>' + _tubes()
    ),
    "solid-core": (
        '<circle cx="24" cy="24" r="19" fill="{tint}" stroke="{accent}" '
        'stroke-width="2"/>'
        '<circle cx="24" cy="24" r="6" fill="{accent}"/>'
    ),
    "filament": (
        '<path d="M3 9 C 14 12, 22 20, 28 22.5 L 28 25.5 C 22 28, 14 36, 3 39 Z" '
        'fill="{tint}" stroke="{accent}" stroke-width="1.6" stroke-linejoin="round"/>'
        '<path d="M27 24 H 45" stroke="{accent}" stroke-width="3" '
        'stroke-linecap="round"/>'
        '<path d="M33 18 L 45 13 M33 30 L 45 35" stroke="{accent}" '
        'stroke-width="1.2" stroke-dasharray="2 2" opacity="0.6"/>'
    ),
    "cartesian": (
        '<rect x="6" y="6" width="36" height="36" fill="{tint}" stroke="{accent}" '
        'stroke-width="2"/>'
        '<path d="M18 6 V 42 M30 6 V 42 M6 18 H 42 M6 30 H 42" stroke="{accent}" '
        'stroke-width="0.8" opacity="0.45"/>'
        '<ellipse cx="24" cy="24" rx="10" ry="7" fill="{accent}" opacity="0.35"/>'
        '<ellipse cx="24" cy="24" rx="5" ry="3.5" fill="{accent}"/>'
    ),
    "line": (
        '<path d="M4 24 H 44" stroke="{accent}" stroke-width="1" opacity="0.4"/>'
        '<path d="M4 14 C 14 14, 18 24, 24 24 C 30 24, 34 14, 44 14" fill="none" '
        'stroke="{accent}" stroke-width="1" opacity="0.4"/>'
        '<rect x="4" y="20" width="40" height="8" fill="{tint}" stroke="{accent}" '
        'stroke-width="1.6"/>'
        '<path d="M8 24 C 16 18, 32 18, 40 24 C 32 30, 16 30, 8 24 Z" '
        'fill="{accent}" opacity="0.7"/>'
    ),
    "other": (
        '<rect x="6" y="11" width="36" height="26" rx="2" fill="{tint}" '
        'stroke="{accent}" stroke-width="2"/>'
        '<rect x="12" y="17" width="24" height="14" fill="#fff" stroke="{accent}" '
        'stroke-width="1.4"/>'
        '<ellipse cx="24" cy="24" rx="7" ry="4" fill="{accent}" opacity="0.55"/>'
    ),
    "design": (
        '<path d="M6 6 V 42 H 44" fill="none" stroke="{accent}" stroke-width="1.6" '
        'opacity="0.55"/>'
        '<path d="M6 24 H 44" fill="none" stroke="{accent}" stroke-width="1" '
        'stroke-dasharray="2 2" opacity="0.55"/>'
        '<path d="M9 8 C 14 22, 18 26, 26 26.5 S 38 30, 43 40" fill="none" '
        'stroke="{accent}" stroke-width="2.4" stroke-linecap="round"/>'
        '<circle cx="21" cy="24" r="3" fill="{tint}" stroke="{accent}" '
        'stroke-width="1.6"/>'
    ),
}
"""The drawings on a 48-unit square, by the tile they stand on."""


def tile_pixmap(name: str, size: int = 44) -> QPixmap:
    """Draw a tile's drawing.

    Parameters
    ----------
    name : str
        Its key in :data:`ART`.
    size : int, optional
        The pixmap's side, in pixels.

    Returns
    -------
    PySide6.QtGui.QPixmap
        The drawing on a transparent square, sharp on a high-density screen.
    """
    body = ART[name].replace("{tint}", TINT).replace("{accent}", ACCENT)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">{body}</svg>'
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap
