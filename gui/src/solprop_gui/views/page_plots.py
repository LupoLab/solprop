# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The analysis pages' plots: white, with a crosshair and a readout, in pyqtgraph.

A plot knows nothing of a run. It draws what it is given, says what is under the
pointer through a reader it is handed, and, where one of its axes is z, says which
position was picked by dragging on it (:attr:`Plot.plane_at`).
"""

from collections.abc import Callable, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent, QPainter

from solprop_gui.model.display import steady_limits
from solprop_gui.ui.theme import ACCENT, AXIS, FONT, GREY, INK

__all__ = ["BLUE", "COLOURS", "ORANGE", "LinePlot", "MapPlot", "Plot"]

BLUE = "#1f6fb4"
"""The first curve's colour."""

ORANGE = "#d2670b"
"""The second curve's colour."""

COLOURS = (
    BLUE,
    ORANGE,
    "#2a9d6a",
    "#8c5bb5",
    "#c2445c",
    "#5a6675",
    "#b08a16",
    "#2b8fb3",
)
"""The curves' colours, in order."""

_LABEL = {"color": GREY, "font-size": "11px", "font-family": FONT}


def _small_font(size: int = 11) -> QFont:
    """Return the font of a plot's ticks.

    Parameters
    ----------
    size : int, optional
        Its size, in pixels.

    Returns
    -------
    PySide6.QtGui.QFont
        The font.
    """
    font = QFont(FONT)
    font.setPixelSize(size)
    return font


class _SmoothImage(pg.ImageItem):
    """An image drawn with bilinear smoothing, since a map's bins are not its data."""

    def paint(self, painter: QPainter, *args: object) -> None:
        """Paint the image, smoothed.

        Parameters
        ----------
        painter : PySide6.QtGui.QPainter
            The painter.
        *args : object
            The item's other paint arguments.
        """
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        super().paint(painter, *args)


class Plot(pg.PlotWidget):
    """A plot with a crosshair and a readout on hover.

    Parameters
    ----------
    xlabel : str
        The lower axis's label.
    ylabel : str
        The left axis's label.
    left : int, optional
        The left axis's width, in pixels.

    Attributes
    ----------
    plane_at : Signal
        Emitted with the position, in the unit of the axis that is z, when it is picked.
    reader : callable or None
        Says what is at a point of the plot, as text.
    along : {'x', 'y'} or None
        Which axis is z.
    """

    plane_at = Signal(float)

    def __init__(self, *, xlabel: str, ylabel: str, left: int = 54) -> None:
        """Make the plot, empty, with its view fixed until it is told otherwise."""
        super().__init__(background="w")
        self.setMinimumSize(120, 90)
        self.along: str | None = None
        self.reader: Callable[[float, float], str] | None = None
        self._dragging = False
        item = self.getPlotItem()
        assert item is not None  # a PlotWidget always has one
        self.item = item
        item.hideButtons()
        item.setMenuEnabled(False)
        item.layout.setContentsMargins(0, 8, 8, 0)  # pyright: ignore[reportAttributeAccessIssue]
        self.box = item.getViewBox()
        self.box.setMouseEnabled(False, False)
        self.box.disableAutoRange()
        font = _small_font()
        for side, text in (("bottom", xlabel), ("left", ylabel)):
            axis = item.getAxis(side)
            axis.setPen(pg.mkPen(AXIS))
            axis.setTextPen(pg.mkPen(GREY))
            axis.setTickFont(font)
            axis.setStyle(tickLength=-4, tickTextOffset=3)
            axis.setLabel(text, **_LABEL)
            # A label says its own unit; no "(x0.001)" beside it.
            axis.enableAutoSIPrefix(False)
        item.getAxis("left").setWidth(left)
        item.getAxis("bottom").setHeight(38)
        for side in ("top", "right"):
            item.showAxis(side)
            axis = item.getAxis(side)
            axis.setPen(pg.mkPen(AXIS))
            axis.setStyle(showValues=False, tickLength=0)
        self.cross_x = pg.InfiniteLine(angle=90, pen=pg.mkPen((120, 130, 140, 190)))
        self.cross_y = pg.InfiniteLine(angle=0, pen=pg.mkPen((120, 130, 140, 190)))
        self.note = pg.TextItem("", color="#fff", fill=(27, 36, 48, 235), anchor=(0, 1))
        self.note.setFont(_small_font(12))
        for extra in (self.cross_x, self.cross_y, self.note):
            self.addItem(extra, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
            extra.setZValue(50)
            extra.hide()
        self.scene().sigMouseMoved.connect(self._hover)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    def set_xlabel(self, text: str) -> None:
        """Change the lower axis's label.

        Parameters
        ----------
        text : str
            The label.
        """
        self.item.getAxis("bottom").setLabel(text, **_LABEL)

    def set_ylabel(self, text: str) -> None:
        """Change the left axis's label.

        Parameters
        ----------
        text : str
            The label.
        """
        self.item.getAxis("left").setLabel(text, **_LABEL)

    def _to_view(self, position: QPointF) -> QPointF | None:
        """Return a scene point in the plot's coordinates, or ``None`` outside it.

        Parameters
        ----------
        position : PySide6.QtCore.QPointF
            The scene point.

        Returns
        -------
        PySide6.QtCore.QPointF or None
            The point.
        """
        if not self.box.sceneBoundingRect().contains(position):
            return None
        return self.box.mapSceneToView(position)

    def _hover(self, position: QPointF) -> None:
        """Follow the pointer.

        Parameters
        ----------
        position : PySide6.QtCore.QPointF
            Where it is, in the scene.
        """
        point = self._to_view(position)
        if point is None or self.reader is None:
            self.clear_hover()
            return
        if self._dragging:
            self.plane_at.emit(point.y() if self.along == "y" else point.x())
        text = self.reader(point.x(), point.y())
        if not text:
            self.clear_hover()
            return
        self.cross_x.setPos(point.x())
        self.cross_y.setPos(point.y())
        self.cross_x.show()
        self.cross_y.show()
        self.note.setText(f" {text} ")
        (x0, x1), (y0, y1) = self.box.viewRange()
        right = (point.x() - x0) / ((x1 - x0) or 1.0) > 0.62
        top = (point.y() - y0) / ((y1 - y0) or 1.0) > 0.9
        self.note.setAnchor((1.0 if right else 0.0, 0.0 if top else 1.0))
        self.note.setPos(point.x(), point.y())
        self.note.show()

    def clear_hover(self) -> None:
        """Take the crosshair away."""
        for extra in (self.cross_x, self.cross_y, self.note):
            extra.hide()

    def leaveEvent(self, event: object) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Take the crosshair away as the pointer leaves.

        Parameters
        ----------
        event : object
            The event.
        """
        self.clear_hover()
        super().leaveEvent(event)  # pyright: ignore[reportArgumentType]

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Pick a plane, where one of the axes is z.

        Parameters
        ----------
        event : PySide6.QtGui.QMouseEvent
            The press.
        """
        if self.along:
            point = self._to_view(self.mapToScene(event.position().toPoint()))
            if point is not None:
                self._dragging = True
                self.plane_at.emit(point.y() if self.along == "y" else point.x())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """End a drag.

        Parameters
        ----------
        event : PySide6.QtGui.QMouseEvent
            The release.
        """
        self._dragging = False
        super().mouseReleaseEvent(event)


class MapPlot(Plot):
    """A false-colour map with a colour bar and a marker of the plane.

    Parameters
    ----------
    xlabel : str
        The lower axis's label.
    ylabel : str
        The left axis's label.
    colours : str
        The colour map's name.
    unit : str
        What the colour bar reads.
    """

    def __init__(self, *, xlabel: str, ylabel: str, colours: str, unit: str) -> None:
        """Make the map, empty."""
        super().__init__(xlabel=xlabel, ylabel=ylabel)
        self.extent: tuple[float, float] | None = None
        self.image = _SmoothImage(axisOrder="row-major")
        self.addItem(self.image)
        self.bar = pg.ColorBarItem(
            colorMap=pg.colormap.get(colours),
            interactive=False,
            width=10,
            colorMapMenu=False,
        )
        self.bar.setImageItem(self.image, insert_in=self.item)
        axis = self.bar.axis
        axis.setPen(pg.mkPen(AXIS))
        axis.setTextPen(pg.mkPen(GREY))
        axis.setTickFont(_small_font())
        axis.setStyle(tickLength=-3, tickTextOffset=3)
        axis.setLabel(unit, **_LABEL)
        self.marker = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen((255, 255, 255, 245), width=1.4, style=Qt.PenStyle.DashLine),
        )
        self.addItem(self.marker, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.marker.setZValue(40)
        self.marker.hide()
        self.message = pg.TextItem("", color=GREY, anchor=(0.5, 0.5))
        self.addItem(self.message, ignoreBounds=True)  # pyright: ignore[reportCallIssue]

    def set_along(self, along: str | None) -> None:
        """Say which axis is z.

        Parameters
        ----------
        along : {'x', 'y'} or None
            The axis.
        """
        self.along = along
        if along:
            self.marker.setAngle(0 if along == "y" else 90)
        self.setCursor(
            Qt.CursorShape.SizeVerCursor
            if along == "y"
            else Qt.CursorShape.SizeHorCursor
            if along
            else Qt.CursorShape.CrossCursor
        )

    def show_map(
        self,
        data: np.ndarray,
        rect: tuple[float, float, float, float],
        view: tuple[tuple[float, float], tuple[float, float]],
        levels: tuple[float, float],
    ) -> None:
        """Show an image.

        Parameters
        ----------
        data : numpy.ndarray
            The image, rows along y.
        rect : tuple of float
            Where it lies: x, y, width and height.
        view : tuple of tuple of float
            The x and y ranges shown.
        levels : tuple of float
            What the colour bar spans.
        """
        self.message.hide()
        self.image.setImage(data, autoLevels=False)
        self.image.setRect(QRectF(*rect))
        self.bar.setLevels(values=levels)
        self.box.setRange(xRange=view[0], yRange=view[1], padding=0)

    def set_levels(self, levels: tuple[float, float]) -> None:
        """Change what the colour bar spans, at no cost in the data.

        Parameters
        ----------
        levels : tuple of float
            The levels.
        """
        self.bar.setLevels(values=levels)

    def set_mark(self, value: float | None) -> None:
        """Move the marker.

        Parameters
        ----------
        value : float or None
            Where it goes; ``None`` hides it.
        """
        if value is None:
            self.marker.hide()
            return
        self.marker.setPos(value)
        self.marker.show()

    def say(self, text: str) -> None:
        """Say something in place of the map.

        Parameters
        ----------
        text : str
            What to say.
        """
        self.image.clear()
        self.marker.hide()
        self.box.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        self.message.setText(text)
        self.message.setPos(0.5, 0.5)
        self.message.show()


class LinePlot(Plot):
    """A plot of one or several curves.

    Parameters
    ----------
    xlabel : str
        The lower axis's label.
    ylabel : str
        The left axis's label.
    names : Sequence of str or None, optional
        The curves' names; several get a legend.
    unit : str, optional
        What the readout says the values are in.
    """

    def __init__(
        self,
        *,
        xlabel: str,
        ylabel: str,
        names: Sequence[str] | None = None,
        unit: str = "",
    ) -> None:
        """Make the plot, with no data."""
        super().__init__(xlabel=xlabel, ylabel=ylabel)
        self.unit = unit
        self.log = False
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.showGrid(x=True, y=True, alpha=0.12)
        self.names: list[str] = []
        self.curves: list[pg.PlotDataItem] = []
        self.legend: pg.LegendItem | None = None
        self.xs = np.zeros(0)
        self.ys: list[np.ndarray] = []
        self.marker = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(ACCENT, style=Qt.PenStyle.DashLine)
        )
        self.addItem(self.marker, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.marker.hide()
        self.reader = self.read
        self.set_names(names or [""])

    def set_names(self, names: Sequence[str]) -> None:
        """Say which curves there are.

        Parameters
        ----------
        names : Sequence of str
            Their names; several get a legend.
        """
        if list(names) == self.names:
            return
        for curve in self.curves:
            self.removeItem(curve)
        if self.legend is not None:
            self.legend.clear()
        elif len(names) > 1:
            self.legend = self.addLegend(
                offset=(-8, 8), labelTextColor=INK, labelTextSize="8pt"
            )
        self.names = list(names)
        self.curves = [
            self.plot(
                [],
                [],
                pen=pg.mkPen(COLOURS[index % len(COLOURS)], width=1.7),
                name=name or None,
                antialias=True,
            )
            for index, name in enumerate(self.names)
        ]

    def set_log(self, log: bool) -> None:
        """Choose a logarithmic y axis.

        Parameters
        ----------
        log : bool
            Whether it is.
        """
        self.log = log
        self.item.setLogMode(False, log)
        if self.ys:
            self.set_data(self.xs, self.ys)
            self.fit()

    def read(self, x: float, _y: float) -> str:
        """Return the curves' values at a position.

        Parameters
        ----------
        x : float
            The position.
        _y : float
            Not read.

        Returns
        -------
        str
            What the readout says.
        """
        if not self.xs.size or not self.ys:
            return ""
        at = int(np.clip(np.searchsorted(self.xs, x), 0, self.xs.size - 1))
        values = ", ".join(f"{y[at]:.3g}" for y in self.ys if at < y.size)
        return f"{x:.4g}: {values} {self.unit}".rstrip()

    def set_data(self, x: np.ndarray, ys: Sequence[np.ndarray]) -> None:
        """Replace the curves.

        Parameters
        ----------
        x : numpy.ndarray
            The positions.
        ys : Sequence of numpy.ndarray
            One array of values per curve.
        """
        self.xs = np.asarray(x, dtype=float)
        self.ys = [np.asarray(y, dtype=float) for y in ys]
        for curve, y in zip(self.curves, self.ys, strict=False):
            if self.log:
                keep = y > 0
                curve.setData(self.xs[keep], y[keep])
            else:
                curve.setData(self.xs, y)

    def set_view(
        self, x_range: tuple[float, float], y_range: tuple[float, float]
    ) -> None:
        """Fix the view.

        Parameters
        ----------
        x_range : tuple of float
            The x range.
        y_range : tuple of float
            The y range, in the data's units.
        """
        low, high = y_range
        if self.log:
            low, high = np.log10(max(low, 1e-300)), np.log10(max(high, 1e-300))
        self.box.setRange(xRange=x_range, yRange=(low, high), padding=0)

    def fit(self) -> None:
        """Fit the view to the curves, magnifying no noise."""
        if not self.xs.size or not self.ys:
            return
        values = np.concatenate(self.ys)
        x_range = (float(self.xs[0]), float(self.xs[-1]))
        if self.log:
            positive = values[np.isfinite(values) & (values > 0)]
            if positive.size:
                low, high = float(positive.min()), float(positive.max())
                self.set_view(x_range, (low / 1.3, high * 1.3))
            return
        self.set_view(x_range, steady_limits(values))
