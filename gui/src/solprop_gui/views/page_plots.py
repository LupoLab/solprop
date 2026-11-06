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
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent, QPainter
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu

from solprop_gui.model.display import half_width, steady_limits
from solprop_gui.ui.theme import ACCENT, AXIS, FONT, GREY, INK

__all__ = ["BLUE", "COLOURS", "MARK_COLOURS", "ORANGE", "LinePlot", "MapPlot", "Plot"]

BLUE = "#1f6fb4"
"""The first curve's colour."""

ORANGE = "#d2670b"
"""The second curve's colour."""

MARK_COLOURS = (ORANGE, GREY, "#0b6e99", "#8c5bb5")
"""The colours of a plot's marks, in order."""

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

SNAP_PIXELS = 14
"""How near a line plot's data point the pointer must come, in pixels, to snap to it."""


def _unit_of(label: str) -> str:
    """Return the unit an axis label gives in brackets.

    Parameters
    ----------
    label : str
        The label, as ``"Time (fs)"``.

    Returns
    -------
    str
        The unit, or nothing.

    Examples
    --------
    >>> _unit_of("Time (fs)"), _unit_of("z")
    ('fs', '')
    """
    if "(" not in label or not label.endswith(")"):
        return ""
    return label.rsplit("(", 1)[1][:-1]


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
    left : int or None, optional
        The left axis's width, in pixels; ``None`` follows its tick labels.

    Attributes
    ----------
    plane_at : Signal
        Emitted with the position, in the unit of the axis that is z, when it is picked.
    picked : Signal
        Emitted with the position, as drawn, when the plot is clicked.
    reader : callable or None
        Says what is at a point of the plot, as text.
    more_actions : callable or None
        What the plot's owner offers at a point, added to the plot's own menu.
    along : {'x', 'y'} or None
        Which axis is z.
    """

    plane_at = Signal(float)
    picked = Signal(float, float)

    def __init__(self, *, xlabel: str, ylabel: str, left: int | None = 54) -> None:
        """Make the plot, empty, with its view fixed until it is told otherwise."""
        super().__init__(background="w")
        self.setMinimumSize(120, 90)
        self.along: str | None = None
        self.reader: Callable[[float, float], str] | None = None
        self.more_actions: (
            Callable[[QPointF], list[tuple[str, Callable[[], None]]]] | None
        ) = None
        self._dragging = False
        item = self.getPlotItem()
        assert item is not None  # a PlotWidget always has one
        self.item = item
        item.hideButtons()
        item.setMenuEnabled(False)
        item.layout.setContentsMargins(0, 8, 8, 0)  # pyright: ignore[reportAttributeAccessIssue]
        self.box = item.getViewBox()
        # The wheel zooms, over the plot or over one axis; a double click goes back.
        self.box.setMouseEnabled(True, True)
        self.box.disableAutoRange()
        self.box.sigRangeChangedManually.connect(self._zoomed_by_hand)
        self.home_view: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.zoomed = False
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
        # None lets the axis take the width of its tick labels.
        if left is not None:
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
        self.dot = pg.ScatterPlotItem(size=9, pen=pg.mkPen("w", width=1.5))
        for extra in (self.cross_x, self.cross_y, self.note, self.dot):
            self.addItem(extra, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
            extra.setZValue(50)
            extra.hide()
        self.dot.setZValue(51)
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

    def snap(self, _x: float, _y: float) -> tuple[float, float, str] | None:
        """Return the data point the crosshair jumps to near a position.

        Parameters
        ----------
        _x, _y : float
            The position, as drawn.

        Returns
        -------
        tuple or None
            The point, as drawn, and the colour of its dot; ``None`` here: a plot
            snaps only where it has points to snap to.
        """
        return None

    def _hover(self, position: QPointF) -> None:
        """Follow the pointer, onto a data point when it comes near one.

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
        snapped = self.snap(point.x(), point.y())
        if snapped is not None:
            point = QPointF(snapped[0], snapped[1])
        text = self.reader(point.x(), point.y())
        if not text:
            self.clear_hover()
            return
        if snapped is None:
            self.dot.hide()
        else:
            self.dot.setData([point.x()], [point.y()], brush=pg.mkBrush(snapped[2]))
            self.dot.show()
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

    def show_view(
        self, x_range: tuple[float, float], y_range: tuple[float, float]
    ) -> None:
        """Set the view a page asks for, unless the person has zoomed it.

        Parameters
        ----------
        x_range : tuple of float
            The x range.
        y_range : tuple of float
            The y range, as the view box holds it.
        """
        self.home_view = (x_range, y_range)
        if not self.zoomed:
            self.box.setRange(xRange=x_range, yRange=y_range, padding=0)

    def reset_view(self) -> None:
        """Go back to the page's view."""
        self.zoomed = False
        if self.home_view is not None:
            self.box.setRange(
                xRange=self.home_view[0], yRange=self.home_view[1], padding=0
            )

    def _zoomed_by_hand(self, *_args: object) -> None:
        """Keep a view the person chose while the page redraws.

        Parameters
        ----------
        *_args : object
            What the signal carries, unused.
        """
        self.zoomed = True

    def menu_actions(self, _point: QPointF) -> list[tuple[str, Callable[[], None]]]:
        """Return what a plot of this kind offers at a point, beyond saving it.

        Parameters
        ----------
        _point : PySide6.QtCore.QPointF
            Where the menu was asked for, in the plot's coordinates.

        Returns
        -------
        list of tuple
            Each action's text and what it does.
        """
        return []

    def show_menu(self, point: QPointF, where: QPoint) -> QMenu:
        """Offer what can be done with the plot at a point.

        Parameters
        ----------
        point : PySide6.QtCore.QPointF
            The point, in the plot's coordinates.
        where : PySide6.QtCore.QPoint
            Where on the screen the menu opens.

        Returns
        -------
        PySide6.QtWidgets.QMenu
            The menu, shown.
        """
        menu = QMenu(self)
        actions = self.menu_actions(point)
        if self.more_actions is not None:
            actions += self.more_actions(point)
        for text, action in actions:
            menu.addAction(text, action)
        if menu.actions():
            menu.addSeparator()
        reset = menu.addAction("Reset the zoom", self.reset_view)
        reset.setEnabled(self.zoomed)
        menu.addAction("Save this plot…", self.choose_save)
        menu.addAction("Copy this plot as an image", self.copy_image)
        menu.popup(where)
        return menu

    def choose_save(self) -> None:  # pragma: no cover - a modal dialog
        """Ask where to save the plot, and save it."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save this plot", "plot.png", "PNG (*.png);;SVG (*.svg)"
        )
        if path:
            self.save(path)

    def save(self, path: str) -> None:
        """Save the plot as it is drawn: a PNG, or an SVG by its extension.

        Parameters
        ----------
        path : str
            The file.
        """
        self.clear_hover()
        if not path.lower().endswith(".svg"):
            self.grab().save(path)
            return
        # Qt's own SVG writer: pyqtgraph's exporter fails on this plot's legend.
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(self.size())
        generator.setViewBox(self.rect())
        painter = QPainter(generator)
        self.render(painter)
        painter.end()

    def copy_image(self) -> None:
        """Put the plot, as it is drawn, on the clipboard."""
        self.clear_hover()
        QApplication.clipboard().setPixmap(self.grab())

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Go back to the page's view.

        Parameters
        ----------
        event : PySide6.QtGui.QMouseEvent
            The double click.
        """
        self.reset_view()
        event.accept()

    def clear_hover(self) -> None:
        """Take the crosshair away."""
        for extra in (self.cross_x, self.cross_y, self.note, self.dot):
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
        point = self._to_view(self.mapToScene(event.position().toPoint()))
        if event.button() == Qt.MouseButton.RightButton:
            if point is not None:
                self.show_menu(point, event.globalPosition().toPoint())
            event.accept()
            return
        if point is not None:
            self.picked.emit(point.x(), point.y())
        if self.along and point is not None:
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
        self.data = np.zeros((0, 0))
        self.map_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self.marker = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen((255, 255, 255, 245), width=1.4, style=Qt.PenStyle.DashLine),
        )
        self.addItem(self.marker, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.marker.setZValue(40)
        self.marker.hide()
        self.message = pg.TextItem("", color=GREY, anchor=(0.5, 0.5))
        self.addItem(self.message, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.reader = self.read

    def read(self, x: float, y: float) -> str:
        """Say what the map holds at a point: its position and its value.

        A page with more to say gives the map a reader of its own.

        Parameters
        ----------
        x : float
            The position across the map.
        y : float
            The position up it.

        Returns
        -------
        str
            The readout, in the axes' and the colour bar's units; nothing off the
            image.
        """
        if not self.data.size:
            return ""
        x0, y0, width, height = self.map_rect
        rows, columns = self.data.shape
        column = int(np.floor((x - x0) / (width or 1.0) * columns))
        row = int(np.floor((y - y0) / (height or 1.0) * rows))
        if not (0 <= row < rows and 0 <= column < columns):
            return ""
        value = float(self.data[row, column])
        if not np.isfinite(value):
            return ""
        xunit = _unit_of(str(self.item.getAxis("bottom").labelText))
        yunit = _unit_of(str(self.item.getAxis("left").labelText))
        unit = str(self.bar.axis.labelText)
        return (
            f"{x:.4g} {xunit}".rstrip()
            + f", {y:.4g} {yunit}".rstrip()
            + f": {value:.3g} {unit}".rstrip()
        )

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

    def set_unit(self, unit: str) -> None:
        """Say what the colour bar's values are in.

        Parameters
        ----------
        unit : str
            The unit.
        """
        self.bar.axis.setLabel(unit, **_LABEL)

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
        self.data = np.asarray(data)
        self.map_rect = rect
        self.show_view(view[0], view[1])

    def menu_actions(self, _point: QPointF) -> list[tuple[str, Callable[[], None]]]:
        """Offer to go to the map's peak, where one axis picks a plane.

        Parameters
        ----------
        _point : PySide6.QtCore.QPointF
            Where the menu was asked for.

        Returns
        -------
        list of tuple
            The action, where the map has data and an axis is z.
        """
        if self.along is None or not self.data.size:
            return []
        return [("Go to the peak", self.go_to_peak)]

    def peak(self) -> tuple[float, float] | None:
        """Return where the map is highest, in its coordinates.

        Returns
        -------
        tuple of float or None
            The x and y of the brightest pixel's middle; ``None`` with no data.
        """
        if not self.data.size or not np.isfinite(self.data).any():
            return None
        row, column = np.unravel_index(int(np.nanargmax(self.data)), self.data.shape)
        x0, y0, width, height = self.map_rect
        rows, columns = self.data.shape
        return (
            x0 + (column + 0.5) * width / columns,
            y0 + (row + 0.5) * height / rows,
        )

    def go_to_peak(self) -> None:
        """Move the plane to where the map is highest."""
        found = self.peak()
        if found is not None:
            self.plane_at.emit(found[1] if self.along == "y" else found[0])

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
        # A long reason wraps within the map rather than running off it.
        self.message.setTextWidth(max(self.width() - 90, 120))
        self.message.setPos(0.5, 0.5)
        self.message.show()


class PlotMarks:
    """Named vertical lines on a plot.

    Parameters
    ----------
    plot : Plot
        The plot.

    Attributes
    ----------
    lines : dict
        The lines made so far, by name.
    """

    def __init__(self, plot: Plot) -> None:
        """Hold the plot; a line is made as it is first asked for."""
        self.plot = plot
        self.lines: dict[str, pg.InfiniteLine] = {}

    def show(self, marks: Sequence[Sequence[object]]) -> None:
        """Mark positions, and hide every other line.

        Parameters
        ----------
        marks : Sequence of Sequence
            Each mark's name and position, ``None`` where it has none.
        """
        for line in self.lines.values():
            line.hide()
        for place, (name, at) in enumerate(marks):
            if at is None:
                continue
            line = self.lines.get(str(name))
            if line is None:
                colour = MARK_COLOURS[place % len(MARK_COLOURS)]
                line = pg.InfiniteLine(
                    angle=90, pen=pg.mkPen(colour, style=Qt.PenStyle.DashLine)
                )
                pg.InfLineLabel(
                    line,
                    str(name),
                    position=0.96 - 0.05 * place,
                    anchors=[(0, 0), (0, 0)],
                    color=colour,
                )
                self.plot.addItem(line, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
                self.lines[str(name)] = line
            line.setPos(float(at))  # pyright: ignore[reportArgumentType]
            line.show()


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
    left : int or None, optional
        The left axis's width, in pixels; ``None`` follows its tick labels.
    """

    def __init__(
        self,
        *,
        xlabel: str,
        ylabel: str,
        names: Sequence[str] | None = None,
        unit: str = "",
        left: int | None = 54,
    ) -> None:
        """Make the plot, with no data."""
        super().__init__(xlabel=xlabel, ylabel=ylabel, left=left)
        self.unit = unit
        self.measures: list[pg.GraphicsObject] = []
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

    def read(self, x: float, y: float) -> str:
        """Return the value of the curve nearest a position.

        Parameters
        ----------
        x : float
            The position along the plot.
        y : float
            The position up it, as drawn: it picks the curve.

        Returns
        -------
        str
            What the readout says: the curve's name, where it has one, and value.
        """
        nearest = self.nearest_curve(x, y)
        if nearest is None:
            return ""
        index, at = nearest
        value = self.ys[index][at]
        name = self.names[index] if index < len(self.names) else ""
        named = f"{name}: " if name and len(self.ys) > 1 else ""
        return f"{named}{self.xs[at]:.4g}, {value:.3g} {self.unit}".rstrip()

    def snap(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Return the data point the crosshair jumps to near a position.

        Parameters
        ----------
        x, y : float
            The position, as drawn (a log axis's exponent).

        Returns
        -------
        tuple or None
            The nearest point of any curve, as drawn, and its curve's colour, when it
            is within :data:`SNAP_PIXELS` of the position on the screen.
        """
        found = self.nearest_point(x, y)
        if found is None or found[2] > SNAP_PIXELS:
            return None
        index, at, _gap = found
        value = float(self.ys[index][at])
        drawn = np.log10(value) if self.log else value
        return float(self.xs[at]), float(drawn), COLOURS[index % len(COLOURS)]

    def nearest_point(self, x: float, y: float) -> tuple[int, int, float] | None:
        """Return the data point nearest a position on the screen.

        Parameters
        ----------
        x : float
            The position along the plot.
        y : float
            The position up it, as drawn (a log axis's exponent).

        Returns
        -------
        tuple or None
            The curve's index, the sample's, and how far it is, in pixels; ``None``
            with no curve, or before the plot has a size.
        """
        width, height = self.box.viewPixelSize()
        if not self.xs.size or not self.ys or not (width > 0 and height > 0):
            return None
        best: tuple[int, int, float] | None = None
        for index, curve in enumerate(self.ys):
            values = curve[: self.xs.size]
            with np.errstate(divide="ignore", invalid="ignore"):
                drawn = np.log10(values) if self.log else values
            gaps = np.hypot((self.xs[: values.size] - x) / width, (drawn - y) / height)
            gaps[~np.isfinite(gaps)] = np.inf
            if not gaps.size:
                continue
            at = int(np.argmin(gaps))
            if best is None or gaps[at] < best[2]:
                best = (index, at, float(gaps[at]))
        return None if best is None or not np.isfinite(best[2]) else best

    def nearest_curve(self, x: float, y: float) -> tuple[int, int] | None:
        """Return the curve nearest a position, and its sample there.

        Parameters
        ----------
        x : float
            The position along the plot.
        y : float
            The position up it, as drawn (a log axis's exponent).

        Returns
        -------
        tuple of int or None
            The curve's index and the sample's; ``None`` with no curve.
        """
        if not self.xs.size or not self.ys:
            return None
        at = int(np.argmin(np.abs(self.xs - x)))
        best, gap = None, np.inf
        for index, curve in enumerate(self.ys):
            if at >= curve.size or not np.isfinite(curve[at]):
                continue
            drawn = curve[at]
            if self.log:
                drawn = np.log10(drawn) if drawn > 0 else -np.inf
            if abs(drawn - y) < gap:
                best, gap = index, abs(drawn - y)
        return None if best is None else (best, at)

    def menu_actions(self, point: QPointF) -> list[tuple[str, Callable[[], None]]]:
        """Offer to measure the width of the nearest curve's peak, and its data.

        Parameters
        ----------
        point : PySide6.QtCore.QPointF
            Where the menu was asked for, in the plot's coordinates.

        Returns
        -------
        list of tuple
            Each action's text and what it does.
        """
        actions: list[tuple[str, Callable[[], None]]] = []

        def measure() -> None:
            """Measure at the point the menu was opened at."""
            self.measure_width(point.x(), point.y())

        if self.xs.size and self.ys:
            actions.append(("Width at half maximum here", measure))
            actions.append(("Copy the data as CSV", self.copy_data))
        if self.measures:
            actions.append(("Clear the widths", self.clear_measures))
        return actions

    def measure_width(self, x: float, y: float) -> str:
        """Mark and label the width at half maximum of the peak nearest a point.

        Parameters
        ----------
        x : float
            The position along the plot.
        y : float
            The position up it, as drawn: it picks the curve.

        Returns
        -------
        str
            The label, or why there is none.
        """
        nearest = self.nearest_curve(x, y)
        if nearest is None:
            return ""
        curve = self.ys[nearest[0]]
        found = half_width(self.xs, curve, x, decibels=self.unit == "dB")
        if found is None:
            text = "No half maximum on both sides here"
            self.note.setText(f" {text} ")
            self.note.setPos(x, y)
            self.note.show()
            return text
        left, right, level = found
        unit = _unit_of(self.item.getAxis("bottom").labelText)
        text = f"FWHM {right - left:.4g} {unit}".rstrip()
        height = np.log10(level) if self.log and level > 0 else level
        line = pg.PlotDataItem(
            [left, right],
            [level, level],
            pen=pg.mkPen(ORANGE, width=2),
            symbol="|",
            symbolPen=pg.mkPen(ORANGE, width=2),
            symbolSize=10,
        )
        label = pg.TextItem(text, color=ORANGE, anchor=(0.5, 1.2))
        label.setFont(_small_font(12))
        label.setPos((left + right) / 2.0, height)
        for item in (line, label):
            self.addItem(item, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
            self.measures.append(item)
        return text

    def clear_measures(self) -> None:
        """Take the width marks away."""
        for item in self.measures:
            self.removeItem(item)
        self.measures = []

    def copy_data(self) -> None:
        """Put the curves on the clipboard as CSV: one column of x, one per curve."""
        names = [name or f"curve {i + 1}" for i, name in enumerate(self.names)]
        names += [f"curve {i + 1}" for i in range(len(names), len(self.ys))]
        lines = [",".join(["x", *names[: len(self.ys)]])]
        for at, x in enumerate(self.xs):
            row = [f"{x:.10g}"] + [
                f"{curve[at]:.10g}" if at < curve.size else "" for curve in self.ys
            ]
            lines.append(",".join(row))
        QApplication.clipboard().setText("\n".join(lines) + "\n")

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
        self.clear_measures()
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
        self.show_view(x_range, (low, high))

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
