# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's step size along the guide, as a small line on the progress strip."""

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from solprop_gui.ui.theme import ACCENT, AXIS

__all__ = ["StepSparkline"]

_MARGIN = 3.0
"""The space around the line, in pixels."""


class StepSparkline(QWidget):
    """The accepted step length against z, on a log scale, as a run reports it.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    points : list of tuple of float
        Each report's ``(z, dz)``, m, in the order they came.
    z_end : float
        Where the run ends, m; the line's right-hand edge.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make an empty line."""
        super().__init__(parent)
        self.points: list[tuple[float, float]] = []
        self.z_end = 0.0
        self.setFixedSize(120, 22)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolTip("The step size along z, on a log scale.")

    def start(self, z_end: float) -> None:
        """Forget the last run's steps.

        Parameters
        ----------
        z_end : float
            Where the new run ends, m.
        """
        self.points.clear()
        self.z_end = z_end
        self.setToolTip("The step size along z, on a log scale.")
        self.update()

    def add(self, z: float, dz: float) -> None:
        """Take one report of the step size.

        Parameters
        ----------
        z : float
            Where the step ended, m.
        dz : float
            Its length, m; one that is not positive is left out.
        """
        if not (dz > 0 and math.isfinite(dz) and math.isfinite(z)):
            return
        self.points.append((z, dz))
        sizes = [size for _, size in self.points]
        self.setToolTip(
            "The step size along z, on a log scale: "
            f"now {dz:.3g} m, from {min(sizes):.3g} to {max(sizes):.3g} m."
        )
        self.update()

    def polygon(self) -> QPolygonF:
        """Return the line in the widget's pixels.

        Returns
        -------
        PySide6.QtGui.QPolygonF
            One point per report: z across, log dz up; empty with fewer than two.
        """
        if len(self.points) < 2:
            return QPolygonF()
        zs = [z for z, _ in self.points]
        logs = [math.log10(dz) for _, dz in self.points]
        left = min(zs)
        right = max(self.z_end, *zs)
        low, high = min(logs), max(logs)
        if high - low < 1e-3:
            low, high = low - 0.5, high + 0.5
        width = self.width() - 2 * _MARGIN
        height = self.height() - 2 * _MARGIN
        span = right - left or 1.0
        return QPolygonF(
            [
                QPointF(
                    _MARGIN + width * (z - left) / span,
                    _MARGIN + height * (high - value) / (high - low),
                )
                for z, value in zip(zs, logs, strict=True)
            ]
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Draw the line on a faint baseline.

        Parameters
        ----------
        event : PySide6.QtGui.QPaintEvent
            The paint event.
        """
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(AXIS), 1.0))
        bottom = self.height() - _MARGIN
        painter.drawLine(
            QPointF(_MARGIN, bottom), QPointF(self.width() - _MARGIN, bottom)
        )
        line = self.polygon()
        if not line.isEmpty():
            pen = QPen(QColor(ACCENT), 1.4)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(line)
        painter.end()
