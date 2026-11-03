# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The step bar's icons: line drawings in the text colour, with a status badge.

The drawings are small SVGs kept here as text, so the front end ships no image files.
Each is drawn in the palette's text colour, so it reads on a light or a dark theme, and
can carry a badge in its corner: solprop's check of the step (accepted, a warning, an
error).
"""

from typing import Final

from PySide6.QtCore import QByteArray, QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer

__all__ = ["BADGE_COLOURS", "DRAWINGS", "icon"]

_HEAD = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
    'stroke-linejoin="round">'
)

DRAWINGS: Final = {
    "start": (
        '<path d="M5 15 16 6l11 9"/><path d="M8 13v13h16V13"/><path d="M14 26v-7h4v7"/>'
    ),
    "geometry": (
        '<ellipse cx="8" cy="16" rx="4" ry="8"/><path d="M8 8h18M8 24h18"/>'
        '<path d="M26 8a4 8 0 0 1 0 16" stroke-dasharray="2 2"/>'
        '<ellipse cx="8" cy="16" rx="1.6" ry="3.2"/>'
    ),
    "medium": (
        '<circle cx="9" cy="10" r="2.5"/><circle cx="21" cy="8" r="2"/>'
        '<circle cx="15" cy="17" r="2.5"/><circle cx="24" cy="20" r="2.5"/>'
        '<circle cx="8" cy="23" r="2"/><circle cx="17" cy="26" r="1.5"/>'
    ),
    "nonlinear": (
        '<path d="M3 22c3 0 3-12 6-12s3 12 6 12 3-12 6-12 3 12 6 12"/>'
        '<path d="M22 4l2 3 3-1-1 3 3 2-3 1 1 3-3-1-2 3-1-3-3 1 1-3-3-2 3-1-1-3 3 1z"'
        ' stroke-width="1.2"/>'
    ),
    "input": (
        '<path d="M3 26h26"/><path d="M5 26c5 0 6-20 11-20s6 20 11 20"/>'
        '<path d="M10 22v4M13 13v13M16 7v19M19 13v13M22 22v4" stroke-width="1"/>'
    ),
    "propagate": (
        '<path d="M3 11h17M3 21h17"/><path d="M20 8l9 8-9 8z"/>'
        '<path d="M6 16c2-3 3 3 5 0s3 3 5 0" stroke-width="1.4"/>'
    ),
    "results": (
        '<path d="M5 4v23h24"/><path d="M8 23c4 0 4-14 8-14s4 10 7 10 3-6 5-6"/>'
        '<path d="M8 19h3M8 15h2" stroke-width="1"/>'
    ),
    "postprocess": (
        '<path d="M4 25 10 9l6 16z"/>'
        '<path d="M17 16c2 0 2-7 4-7s2 14 4 14 2-7 4-7" stroke-width="1.4"/>'
    ),
    "panel-left": (
        '<rect x="4" y="6" width="24" height="20" rx="2"/>'
        '<rect x="4" y="6" width="9" height="20" rx="2" fill="currentColor"/>'
    ),
    "panel-bottom": (
        '<rect x="4" y="6" width="24" height="20" rx="2"/>'
        '<rect x="4" y="19" width="24" height="7" rx="2" fill="currentColor"/>'
    ),
    "back": '<path d="M20 7 11 16l9 9"/>',
    "forward": '<path d="M12 7l9 9-9 9"/>',
    "run": '<path d="M10 6v20l16-10z" fill="currentColor"/>',
    "stop": '<rect x="8" y="8" width="16" height="16" rx="2" fill="currentColor"/>',
    "kill": '<path d="M8 8l16 16M24 8 8 24" stroke-width="3"/>',
    "resume": (
        '<path d="M6 6v20" stroke-width="3"/>'
        '<path d="M12 6v20l14-10z" fill="currentColor"/>'
    ),
    "estimate": (
        '<circle cx="16" cy="17" r="11"/><path d="M16 17l5-6"/><path d="M13 3h6"/>'
    ),
}
"""The drawings, by name: the inside of a 32 by 32 SVG drawn in ``currentColor``."""

BADGE_COLOURS: Final = {
    "accepted": "#2e7d32",
    "warning": "#ef8f00",
    "error": "#c62828",
}
"""The badge colours, by solprop's verdict on the step."""

_BADGE_MARKS = {"accepted": "check", "warning": "bang", "error": "cross"}


def _drawing(name: str, colour: QColor, size: int) -> QPixmap:
    """Draw one drawing in a colour.

    Parameters
    ----------
    name : str
        The drawing's name in :data:`DRAWINGS`.
    colour : PySide6.QtGui.QColor
        Its colour.
    size : int
        The pixmap's side, pixels.

    Returns
    -------
    PySide6.QtGui.QPixmap
        The drawing on a transparent square.
    """
    svg = (_HEAD + DRAWINGS[name] + "</svg>").replace("currentColor", colour.name())
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    return pixmap


def _badge(pixmap: QPixmap, verdict: str) -> None:
    """Paint a verdict's badge in the drawing's lower right corner.

    Parameters
    ----------
    pixmap : PySide6.QtGui.QPixmap
        The drawing, painted over.
    verdict : str
        ``"accepted"``, ``"warning"`` or ``"error"``.
    """
    side = pixmap.width()
    radius = side * 0.22
    centre = QPointF(side - radius - 0.5, side - radius - 0.5)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("white"), side * 0.04))
    painter.setBrush(QColor(BADGE_COLOURS[verdict]))
    painter.drawEllipse(centre, radius, radius)
    pen = QPen(QColor("white"), side * 0.06)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    x, y, r = centre.x(), centre.y(), radius * 0.5
    mark = _BADGE_MARKS[verdict]
    if mark == "check":
        painter.drawPolyline(
            [
                QPointF(x - r, y),
                QPointF(x - r / 3, y + r * 0.7),
                QPointF(x + r, y - r * 0.6),
            ]
        )
    elif mark == "cross":
        painter.drawLine(QPointF(x - r, y - r), QPointF(x + r, y + r))
        painter.drawLine(QPointF(x - r, y + r), QPointF(x + r, y - r))
    else:
        painter.drawLine(QPointF(x, y - r), QPointF(x, y + r * 0.2))
        painter.drawPoint(QPointF(x, y + r * 0.85))
    painter.end()


def icon(
    name: str, colour: QColor, verdict: str | None = None, size: int = 64
) -> QIcon:
    """Return a drawing as an icon, optionally with a verdict's badge.

    Parameters
    ----------
    name : str
        The drawing's name in :data:`DRAWINGS`.
    colour : PySide6.QtGui.QColor
        The colour to draw it in: the palette's text colour.
    verdict : str or None, optional
        ``"accepted"``, ``"warning"`` or ``"error"`` for a badge; ``None`` for none.
    size : int, optional
        The side it is drawn at, pixels; Qt scales it down.

    Returns
    -------
    PySide6.QtGui.QIcon
        The icon, with a paler copy for the disabled state.
    """
    normal = _drawing(name, colour, size)
    if verdict is not None:
        _badge(normal, verdict)
    faded = QColor(colour)
    faded.setAlphaF(0.35)
    result = QIcon(normal)
    result.addPixmap(_drawing(name, faded, size), QIcon.Mode.Disabled)
    return result
