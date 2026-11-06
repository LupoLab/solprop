# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A splitter whose handles can be seen: a grip of dots, lit under the mouse.

Qt's own handles are a few pixels of the window's colour, with nothing to say they
can be dragged. These are wider, carry a grip in their middle, and darken under the
mouse; double-clicking one asks for its panel to be hidden (the window says how to
bring it back). Panels cannot be dragged shut, so a panel is never lost by a drag.
"""

from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QEnterEvent, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QSplitter, QSplitterHandle, QWidget

__all__ = ["GRIP_WIDTH", "GripHandle", "GripSplitter"]

GRIP_WIDTH = 8
"""The handles' thickness, pixels."""


class GripHandle(QSplitterHandle):
    """A handle drawn with a grip, lit while the mouse is over it.

    Parameters
    ----------
    orientation : PySide6.QtCore.Qt.Orientation
        The splitter's orientation.
    parent : GripSplitter
        The splitter.
    """

    def __init__(self, orientation: Qt.Orientation, parent: GripSplitter) -> None:
        """Make the handle, unlit."""
        super().__init__(orientation, parent)
        self.lit = False
        self.setToolTip("Drag to resize; double-click to hide the panel.")

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802 - Qt's name
        """Light the handle.

        Parameters
        ----------
        event : PySide6.QtGui.QEnterEvent
            The event.
        """
        self.lit = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt's name
        """Put the handle's light out.

        Parameters
        ----------
        event : PySide6.QtCore.QEvent
            The event.
        """
        self.lit = False
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Ask for the splitter's panel to be hidden.

        Parameters
        ----------
        event : PySide6.QtGui.QMouseEvent
            The event.
        """
        splitter = self.splitter()
        if isinstance(splitter, GripSplitter):
            splitter.hide_asked.emit()
        event.accept()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt's name
        """Draw the handle: a band, darker when lit, with five dots in its middle.

        Parameters
        ----------
        _event : PySide6.QtGui.QPaintEvent
            The event.
        """
        palette = self.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        band = palette.mid().color() if self.lit else palette.window().color()
        if self.lit:
            band.setAlphaF(0.45)
        painter.fillRect(self.rect(), band)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(palette.dark().color())
        centre = QPointF(self.rect().center())
        vertical = self.orientation() == Qt.Orientation.Horizontal
        for offset in (-12, -6, 0, 6, 12):
            if vertical:
                point = QPointF(centre.x() + 0.5, centre.y() + offset)
            else:
                point = QPointF(centre.x() + offset, centre.y() + 0.5)
            painter.drawEllipse(point, 1.4, 1.4)
        painter.end()


class GripSplitter(QSplitter):
    """A splitter with :class:`GripHandle` handles, whose panels cannot be shut.

    Parameters
    ----------
    orientation : PySide6.QtCore.Qt.Orientation, optional
        Side by side (horizontal) or one above the other (vertical).
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    hide_asked : Signal
        Emitted when a handle is double-clicked.
    """

    hide_asked = Signal()

    def __init__(
        self,
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        parent: QWidget | None = None,
    ) -> None:
        """Make the splitter, with wide handles and no collapsing."""
        super().__init__(orientation, parent)
        self.setHandleWidth(GRIP_WIDTH)
        self.setChildrenCollapsible(False)

    def createHandle(self) -> QSplitterHandle:  # noqa: N802 - Qt's name
        """Make a handle with a grip.

        Returns
        -------
        PySide6.QtWidgets.QSplitterHandle
            The handle.
        """
        return GripHandle(self.orientation(), self)
