# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scans: a mode of their own, started from a sheet (design 12.7).

The setup's ``[scan]`` table is edited here, beside the scan's progress and its results.
A run from the sheet leaves the table out and runs the setup once; the scan runs only
from here, each point a run of the setup with its variables set and pins solved again.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.display import scaled_unit
from solprop_gui.model.scans import result_title, scan_lines, shown_results
from solprop_gui.registry.pages import label
from solprop_gui.registry.units import pretty
from solprop_gui.ui.scan_editor import ScanEditor
from solprop_gui.ui.theme import named_label
from solprop_gui.views.page_plots import ORANGE, LinePlot

__all__ = ["ScanResults", "ScanStage", "point_text"]


def point_text(data: Mapping[str, object]) -> str:
    """Say what a scan's ``point`` event means.

    Parameters
    ----------
    data : Mapping
        The event's data: ``index``, and ``values`` as a point starts, or ``done``,
        ``points`` and ``status`` as it ends.

    Returns
    -------
    str
        A line for the person watching.

    Examples
    --------
    >>> point_text({"index": 0, "values": {"pressure": 0.5}, "status": None})
    'Point 1: pressure = 0.5'
    >>> point_text({"index": 0, "done": 1, "points": 4, "status": "done"})
    '1 of 4 points done'
    >>> point_text({"index": 2, "done": 3, "points": 4, "status": "failed"})
    '3 of 4 points done; point 3 failed'
    """
    index = int(data.get("index") or 0)  # pyright: ignore[reportArgumentType]
    if data.get("status") is None:
        values: Mapping[str, object] = data.get("values") or {}  # pyright: ignore[reportAssignmentType]
        shown = ", ".join(f"{name} = {value:.4g}" for name, value in values.items())  # pyright: ignore[reportUnknownArgumentType]
        return f"Point {index + 1}: {shown}"
    text = f"{data.get('done')} of {data.get('points')} points done"
    if data["status"] == "failed":
        text += f"; point {index + 1} failed"
    return text


class ScanResults(QWidget):
    """A scan's results: one recorded quantity over its variables.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    result : PySide6.QtWidgets.QComboBox
        Which result is drawn.
    status : PySide6.QtWidgets.QLabel
        What the scan holds, or why it is not shown.
    plot : solprop_gui.views.page_plots.LinePlot
        The result over the first variable, a line for each value of the second.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the choice and the plot, hidden until a scan is shown."""
        super().__init__(parent)
        self.setObjectName("page")
        self.scan: Mapping[str, object] = {}
        self.result = QComboBox()
        self.result.setObjectName("scan-result")
        self.result.currentIndexChanged.connect(lambda _index: self.redraw())
        self.status = named_label("No scan run yet.", "pageNote")
        self.status.setWordWrap(True)
        self.plot = LinePlot(xlabel="", ylabel="", left=None)
        self.failed = pg.ScatterPlotItem(
            symbol="x", size=11, pen=pg.mkPen(ORANGE, width=2)
        )
        self.plot.addItem(self.failed)  # pyright: ignore[reportCallIssue]
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self.result)
        column.addWidget(self.status)
        column.addWidget(self.plot, 1)
        self.result.hide()
        self.plot.hide()

    def set_scan(self, scan: Mapping[str, object]) -> None:
        """Show a scan.

        Parameters
        ----------
        scan : Mapping
            The kernel's ``scan_overview`` reply.
        """
        self.scan = scan
        results: Mapping[str, Mapping[str, object]] = scan.get("results") or {}  # pyright: ignore[reportAssignmentType]
        self.result.blockSignals(True)
        self.result.clear()
        for name in shown_results(results):
            self.result.addItem(result_title(name), name)
        self.result.blockSignals(False)
        status = np.asarray(scan.get("status") or [], dtype=object).ravel()
        failed = int(np.count_nonzero(status == "failed"))
        variables: Sequence[Mapping[str, object]] = scan.get("variables") or []  # pyright: ignore[reportAssignmentType]
        over = " × ".join(label(str(v["name"])).lower() for v in variables)  # noqa: RUF001 - times
        said = f"{status.size} points over {over or 'nothing'}"
        said += f", {failed} failed (marked ×)." if failed else "."  # noqa: RUF001 - times
        self.status.setText(f"{said} Each point is read at its last plane.")
        self.result.setVisible(self.result.count() > 0)
        self.plot.setVisible(self.result.count() > 0 and bool(variables))
        self.redraw()

    def redraw(self) -> None:
        """Draw the result chosen."""
        name = self.result.currentData()
        variables = self.scan.get("variables") or []
        if name is None or not variables:
            return
        drawn = scan_lines(
            self.scan,
            str(name),
            {str(v["name"]): label(str(v["name"])) for v in variables},  # pyright: ignore[reportGeneralTypeIssues]
        )
        ys = [y for _title, y in drawn.lines]
        unit, scale = scaled_unit(np.concatenate(ys), pretty(drawn.unit))
        xunit, xscale = scaled_unit(drawn.x, drawn.xunit) if drawn.xunit else ("", 1.0)
        title = result_title(str(name))
        self.plot.set_xlabel(f"{drawn.xlabel} ({xunit})" if xunit else drawn.xlabel)
        self.plot.set_ylabel(f"{title} ({unit})" if unit else title)
        self.plot.unit = unit
        self.plot.set_names([line for line, _y in drawn.lines])
        self.plot.set_data(drawn.x * xscale, [y * scale for y in ys])
        for curve in self.plot.curves:
            colour = curve.opts["pen"].color()
            curve.setSymbol("o")
            curve.setSymbolSize(6)
            curve.setSymbolBrush(colour)
            curve.setSymbolPen(colour)
        self.plot.fit()
        # The points sit at the ends of the range, so the view reaches past them.
        low, high = float(drawn.x.min()) * xscale, float(drawn.x.max()) * xscale
        pad = 0.04 * (high - low or abs(high) or 1.0)
        self.plot.box.setXRange(low - pad, high + pad, padding=0)
        # A failed point has no value: it is marked along the foot of the plot.
        bottom, top = self.plot.box.viewRange()[1]
        foot = bottom + 0.04 * (top - bottom)
        self.failed.setData(drawn.failed * xscale, np.full(drawn.failed.size, foot))

    def show_error(self, message: str) -> None:
        """Say why the scan is not shown.

        Parameters
        ----------
        message : str
            The reason.
        """
        self.scan = {}
        self.result.clear()
        self.result.hide()
        self.plot.hide()
        self.status.setText(message)


class ScanStage(QWidget):
    """The scan's variables, its progress, and its results.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    table_changed : Signal
        Emitted with the ``[scan]`` table as it is edited, or ``None`` for none.
    editor : solprop_gui.ui.scan_editor.ScanEditor
        The table.
    progress : PySide6.QtWidgets.QLabel
        Where a running scan says how far it has got.
    browser : ScanResults
        The scan's results.
    """

    table_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the editor on the left and the results on the right."""
        super().__init__(parent)
        self.setObjectName("page")
        self.editor = ScanEditor()
        self.editor.enabled.setText("Scan this setup")
        self.editor.changed.connect(self.table_changed)
        inner = QWidget()
        inner.setObjectName("page")
        column = QVBoxLayout(inner)
        column.setContentsMargins(20, 16, 16, 16)
        column.setSpacing(8)
        column.addWidget(named_label("Scan", "h3"))
        about = named_label(
            "Each point is a run of this setup with its variables set, its targets "
            "solved again. Run from the sheet, the setup runs once. Right-click a "
            "number on the sheet to scan it.",
            "note",
        )
        about.setWordWrap(True)
        column.addWidget(about)
        column.addWidget(self.editor)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName("sheetScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)
        scroll.setFixedWidth(380)
        right = QWidget()
        right.setObjectName("page")
        side = QVBoxLayout(right)
        side.setContentsMargins(20, 16, 20, 12)
        side.setSpacing(8)
        side.addWidget(named_label("Results", "h3"))
        self.progress = named_label("", "pageNote")
        self.progress.setWordWrap(True)
        side.addWidget(self.progress)
        self.browser = ScanResults()
        side.addWidget(self.browser, 1)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(scroll)
        row.addWidget(right, 1)

    def set_paths(self, paths: Sequence[str]) -> None:
        """Offer a route's numeric keywords as the scan's variables.

        Parameters
        ----------
        paths : Sequence of str
            The keywords.
        """
        self.editor.set_paths(paths)

    def show_table(self, table: object) -> None:
        """Show the setup's ``[scan]`` table.

        Parameters
        ----------
        table : object
            The table, or ``None``.
        """
        self.editor.set_value(table)

    def has_points(self) -> bool:
        """Return whether the scan has a point to run.

        Returns
        -------
        bool
            Whether it is on with a variable holding a value.
        """
        return not self.editor.count.text().startswith("No") and (
            self.editor.table() is not None
        )

    def show_point(self, data: Mapping[str, object]) -> None:
        """Follow a running scan.

        Parameters
        ----------
        data : Mapping
            A ``scan_started`` or ``point`` event's data.
        """
        if "index" not in data:
            self.started(int(data.get("points") or 0))  # pyright: ignore[reportArgumentType]
            return
        self.progress.setText(point_text(data))

    def started(self, points: int) -> None:
        """Say a scan has started.

        Parameters
        ----------
        points : int
            How many points it runs.
        """
        self.progress.setText(f"Running {points} points…")

    def show_scan(self, scan: Mapping[str, object], path: str) -> None:
        """Show a scan's results.

        Parameters
        ----------
        scan : Mapping
            The kernel's ``scan_overview`` reply.
        path : str
            Its file.
        """
        self.browser.set_scan(scan)
        folder = Path(path).parent.name
        self.progress.setText(f"The scan in {folder}.")

    def show_error(self, message: str) -> None:
        """Say why a scan's results cannot be shown.

        Parameters
        ----------
        message : str
            The reason solprop gave.
        """
        self.browser.show_error(message)
