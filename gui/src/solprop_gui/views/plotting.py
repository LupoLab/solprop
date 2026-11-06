# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plots' common style and their hover readouts, after the Qt prototype.

White plots with black axes; a curve in one blue; a readout that follows the mouse,
with a dot on the point of a curve it reads, opening towards the middle of the view
so it never leaves it, and gone when the mouse leaves the plots.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QFileDialog, QWidget

__all__ = [
    "BLUE",
    "PALETTE",
    "HoverLayout",
    "Lookup",
    "Readout",
    "choose_figure",
    "colormap",
    "curve_lookup",
    "figure_spec",
]

FIGURE_FILTER = "PNG image (*.png);;SVG drawing (*.svg);;PDF document (*.pdf)"
"""The figure formats offered, for a file dialog."""

pg.setConfigOptions(antialias=True, background="w", foreground="k")

BLUE = "#1f77b4"
"""The colour of a single curve."""

PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
"""The colours of several curves, in turn (Matplotlib's first six)."""

Lookup = Callable[[float, float], tuple[float, float, str] | None]
"""Given a point in a plot's coordinates, the point to mark and its label, or None."""


def colormap(name: str) -> pg.ColorMap:
    """Return one of pyqtgraph's colour maps.

    Parameters
    ----------
    name : str
        The map's name.

    Returns
    -------
    pyqtgraph.ColorMap
        The map.

    Raises
    ------
    LookupError
        If pyqtgraph has no map of that name.
    """
    found = pg.colormap.get(name)
    if found is None:
        raise LookupError(name)
    return found


class Readout:
    """A hover label on one plot, cross-hairs through the point it reads, a dot.

    The cross-hairs are dotted lines across the plot at the point read: the mouse
    on a map, the nearest point of a curve. They are marked ``crosshair`` and left
    out of an exported figure (:func:`figure_spec`).

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        The plot.
    dot : bool, optional
        Whether to mark the point read.
    """

    def __init__(self, plot: pg.PlotItem, *, dot: bool = False) -> None:
        """Make the label and the dot, hidden."""
        self.plot = plot
        self.text = pg.TextItem(
            color="w", fill=pg.mkBrush(40, 40, 40, 210), anchor=(0, 1)
        )
        self.text.setZValue(100)
        plot.addItem(self.text, ignoreBounds=True)
        pen = pg.mkPen((150, 150, 150, 200), width=1, style=Qt.PenStyle.DotLine)
        self.cross = [pg.InfiniteLine(angle=angle, pen=pen) for angle in (90, 0)]
        for line in self.cross:
            line.crosshair = True  # pyright: ignore[reportAttributeAccessIssue]
            line.setZValue(98)
            plot.addItem(line, ignoreBounds=True)
        self.dot: pg.ScatterPlotItem | None = None
        if dot:
            self.dot = pg.ScatterPlotItem(
                size=9, pen=pg.mkPen("w", width=1.5), brush=pg.mkBrush(BLUE)
            )
            self.dot.setZValue(99)
            plot.addItem(self.dot, ignoreBounds=True)
        self.hide()

    def hide(self) -> None:
        """Hide the label, the cross-hairs and the dot."""
        self.text.hide()
        for line in self.cross:
            line.hide()
        if self.dot is not None:
            self.dot.hide()

    def show(self, x: float, y: float, label: str) -> None:
        """Show a label at a point, opening towards the middle of the view.

        Parameters
        ----------
        x, y : float
            The point, in the plot's coordinates.
        label : str
            The text.
        """
        (x_low, x_high), (y_low, y_high) = self.plot.getViewBox().viewRange()
        self.text.setAnchor(
            (1 if x > (x_low + x_high) / 2 else 0, 0 if y > (y_low + y_high) / 2 else 1)
        )
        self.text.setText(label)
        self.text.setPos(x, y)
        self.text.show()
        for line, value in zip(self.cross, (x, y), strict=True):
            line.setPos(value)
            line.show()
        if self.dot is not None:
            self.dot.setData([x], [y])
            self.dot.show()


def curve_lookup(curve: pg.PlotDataItem, label: str) -> Lookup:
    """Return a lookup that reads a curve's point nearest in x.

    Parameters
    ----------
    curve : pyqtgraph.PlotDataItem
        The curve.
    label : str
        The label's format, given the point's x and y.

    Returns
    -------
    callable
        The :data:`Lookup`.
    """

    def lookup(x: float, _y: float) -> tuple[float, float, str] | None:
        """Read the curve's point nearest in x.

        Parameters
        ----------
        x, _y : float
            The mouse, in the plot's coordinates.

        Returns
        -------
        tuple or None
            The point and its label; ``None`` for an empty curve.
        """
        xs, ys = curve.getData()
        if xs is None or ys is None or len(xs) == 0:
            return None
        index = (
            int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1)) if len(xs) > 1 else 0
        )
        if len(xs) > 1 and x - xs[index - 1] < xs[index] - x:
            index -= 1
        return float(xs[index]), float(ys[index]), label.format(xs[index], ys[index])

    return lookup


class HoverLayout(pg.GraphicsLayoutWidget):
    """A grid of plots whose readouts follow the mouse and vanish when it leaves."""

    def __init__(self) -> None:
        """Make the grid, with no readouts."""
        super().__init__()
        self.targets: list[tuple[pg.PlotItem, Lookup, Readout]] = []
        self.scene().sigMouseMoved.connect(self.moved)  # pyright: ignore[reportAttributeAccessIssue]

    def add_plot(self, row: int, col: int, title: str, colspan: int = 1) -> pg.PlotItem:
        """Add a plot to the grid.

        Parameters
        ----------
        row, col : int
            Its place.
        title : str
            Its title.
        colspan : int, optional
            The columns it spans.

        Returns
        -------
        pyqtgraph.PlotItem
            The plot.
        """
        return self.ci.addPlot(row=row, col=col, title=title, colspan=colspan)

    def add_readout(
        self, plot: pg.PlotItem, lookup: Lookup, *, dot: bool = False
    ) -> None:
        """Give a plot a readout.

        Parameters
        ----------
        plot : pyqtgraph.PlotItem
            The plot.
        lookup : callable
            What the readout reads (:data:`Lookup`).
        dot : bool, optional
            Whether to mark the point read.
        """
        self.targets.append((plot, lookup, Readout(plot, dot=dot)))

    def moved(self, position: QPointF) -> None:
        """Show the readout of the plot under the mouse, and hide the others.

        Parameters
        ----------
        position : PySide6.QtCore.QPointF
            The mouse, in scene coordinates.
        """
        for plot, lookup, readout in self.targets:
            hit = None
            box = plot.getViewBox()
            if box.sceneBoundingRect().contains(position):
                point = box.mapSceneToView(position)
                hit = lookup(point.x(), point.y())
            if hit:
                readout.show(*hit)
            else:
                readout.hide()

    def leaveEvent(self, ev: QEvent) -> None:  # noqa: N802 - Qt's name
        """Hide every readout when the mouse leaves.

        Parameters
        ----------
        ev : PySide6.QtCore.QEvent
            The event.
        """
        for _, _, readout in self.targets:
            readout.hide()
        super().leaveEvent(ev)


_DASHES: Final = {
    Qt.PenStyle.SolidLine: "-",
    Qt.PenStyle.DashLine: "--",
    Qt.PenStyle.DotLine: ":",
    Qt.PenStyle.DashDotLine: "-.",
}
"""Qt's pen styles as Matplotlib's line styles."""


def _pen(pen: object) -> dict[str, object]:
    """Return a pen as Matplotlib's line keywords.

    Parameters
    ----------
    pen : object
        The pen, or what :func:`pyqtgraph.mkPen` makes one from.

    Returns
    -------
    dict
        ``colour`` (``#rrggbb``), ``style`` and ``width`` (points).
    """
    qpen = pen if isinstance(pen, QPen) else pg.mkPen(pen)
    return {
        "colour": qpen.color().name(),
        "style": _DASHES.get(qpen.style(), "-"),
        "width": max(qpen.widthF(), 1.0) * 0.75,
    }


def figure_spec(plot: pg.PlotItem) -> dict[str, object]:
    """Describe what a plot shows, as plain data a figure can be drawn from.

    The curves (as given, before a log axis maps them), the marker lines and the
    marked points, the axis titles and the log axes: what the kernel's ``figure``
    request draws with Matplotlib, so an exported figure is the view's.

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        The plot.

    Returns
    -------
    dict
        ``xlabel``, ``ylabel``, ``logx``, ``logy``, ``curves`` (each ``x``, ``y``,
        ``label`` and its pen), ``lines`` (each ``x`` or ``y``, ``label`` and pen)
        and ``points`` (each ``x``, ``y``, ``label``, ``colour``).
    """
    logx = bool(plot.getAxis("bottom").logMode)
    logy = bool(plot.getAxis("left").logMode)
    curves: list[dict[str, object]] = []
    lines: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for item in plot.items:
        if isinstance(item, pg.PlotDataItem):
            x, y = item.getOriginalDataset()
            if x is None or y is None:
                continue
            curves.append(
                {
                    "x": np.asarray(x, dtype=float),
                    "y": np.asarray(y, dtype=float),
                    "label": item.name() or "",
                    **_pen(item.opts["pen"]),
                }
            )
        elif isinstance(item, pg.InfiniteLine):
            if getattr(item, "crosshair", False):
                continue  # the readout's, not the view's
            vertical = abs(item.angle % 180 - 90) < 1e-9
            value = float(item.value())  # pyright: ignore[reportArgumentType]
            if logx if vertical else logy:
                value = 10.0**value
            marker = getattr(item, "label", None)  # set only for a labelled line
            label = marker.format if marker is not None else ""
            line = {"label": label, **_pen(item.pen)}
            line["x" if vertical else "y"] = value
            lines.append(line)
        elif isinstance(item, pg.ScatterPlotItem):
            for spot in item.points():
                position = spot.pos()
                x, y = float(position.x()), float(position.y())
                points.append(
                    {
                        "x": 10.0**x if logx else x,
                        "y": 10.0**y if logy else y,
                        "label": item.name() or "",
                        "colour": spot.pen().color().name(),
                    }
                )
    return {
        "xlabel": plot.getAxis("bottom").labelText,
        "ylabel": plot.getAxis("left").labelText,
        "logx": logx,
        "logy": logy,
        "curves": curves,
        "lines": lines,
        "points": points,
    }


def choose_figure(parent: QWidget, name: str) -> str | None:  # pragma: no cover - modal
    """Ask where to write a figure.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget
        The dialog's owner.
    name : str
        The file name offered.

    Returns
    -------
    str or None
        The file chosen, or ``None`` when the dialog is cancelled.
    """
    chosen, _ = QFileDialog.getSaveFileName(
        parent, "Export figure", name, FIGURE_FILTER
    )
    return chosen or None
