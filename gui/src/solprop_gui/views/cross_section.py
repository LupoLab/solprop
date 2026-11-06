# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A guide's cross-section, painted to scale from the shapes solprop gives."""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["CrossSection"]

_MARGIN = 8.0
"""The space around the drawing, in pixels."""


def _extent(shape: Mapping[str, object]) -> float:
    """Return how far a shape reaches from the axis.

    Parameters
    ----------
    shape : Mapping
        The shape.

    Returns
    -------
    float
        The distance, m.
    """
    if shape["kind"] == "box":
        corners = [*shape["lower"], *shape["upper"]]  # pyright: ignore[reportGeneralTypeIssues]
        return max(abs(float(v)) for v in corners)
    centre: Sequence[float] = shape["centre"]  # pyright: ignore[reportAssignmentType]
    radius = shape.get("radius", shape.get("outer"))
    return max(abs(float(centre[0])), abs(float(centre[1]))) + float(radius)  # pyright: ignore[reportArgumentType]


class CrossSection(QWidget):
    """Paint a guide's cross-section.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    shapes : list of Mapping
        What it paints, back to front, as the kernel gives them.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the widget, with nothing to paint."""
        super().__init__(parent)
        self.shapes: list[Mapping[str, object]] = []
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def set_shapes(self, shapes: Sequence[Mapping[str, object]]) -> None:
        """Take what to paint.

        Parameters
        ----------
        shapes : Sequence of Mapping
            The kernel's shapes.
        """
        self.shapes = list(shapes)
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt's name
        """Paint the shapes, the largest fitting the widget.

        Parameters
        ----------
        _event : PySide6.QtGui.QPaintEvent
            The event.
        """
        if not self.shapes:
            return
        reach = max(_extent(shape) for shape in self.shapes) or 1.0
        scale = (min(self.width(), self.height()) / 2.0 - _MARGIN) / reach
        middle = QPointF(self.width() / 2.0, self.height() / 2.0)

        def at(x: float, y: float) -> QPointF:
            """Return a point of the guide on the widget.

            Parameters
            ----------
            x : float
                Its x, m.
            y : float
                Its y, m.

            Returns
            -------
            PySide6.QtCore.QPointF
                The point, y upwards.
            """
            return QPointF(middle.x() + x * scale, middle.y() - y * scale)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for shape in self.shapes:
            painter.setPen(QPen(QColor(str(shape["edge"])), 1.0))
            painter.setBrush(QColor(str(shape["fill"])))
            if shape["kind"] == "box":
                low: Sequence[float] = shape["lower"]  # pyright: ignore[reportAssignmentType]
                high: Sequence[float] = shape["upper"]  # pyright: ignore[reportAssignmentType]
                painter.drawRect(QRectF(at(low[0], high[1]), at(high[0], low[1])))
                continue
            centre = at(float(shape["centre"][0]), float(shape["centre"][1]))  # pyright: ignore[reportIndexIssue, reportArgumentType]
            if shape["kind"] == "disc":
                radius = float(shape["radius"]) * scale  # pyright: ignore[reportArgumentType]
                painter.drawEllipse(centre, radius, radius)
                continue
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for edge in (shape["outer"], shape["inner"]):
                radius = float(edge) * scale  # pyright: ignore[reportArgumentType]
                path.addEllipse(centre, radius, radius)
            painter.drawPath(path)
        painter.end()
