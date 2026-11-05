# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scans: a mode of their own, started from a sheet (design 12.7).

The setup's ``[scan]`` table is edited here, beside the scan's progress and its results.
A run from the sheet leaves the table out and runs the setup once; the scan runs only
from here, each point a run of the setup with its variables set and pins solved again.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from solprop.recipe.quantity import Unit, parse_quantity

from solprop_gui.model.display import decibels, even_rows, pulse_window, scaled_unit
from solprop_gui.model.scans import (
    result_title,
    scan_lines,
    shown_results,
    variable_unit,
)
from solprop_gui.registry.pages import label
from solprop_gui.registry.units import pretty
from solprop_gui.ui.quantity import QuantityField
from solprop_gui.ui.scan_editor import ScanEditor
from solprop_gui.ui.theme import named_label
from solprop_gui.views.page_plots import ORANGE, LinePlot, MapPlot

__all__ = ["ScanProgress", "ScanRecord", "ScanResults", "ScanStage", "point_text"]


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
        shown = ", ".join(
            f"{name} = {_value_text(value)}" for name, value in values.items()
        )
        return f"Point {index + 1}: {shown}"
    text = f"{data.get('done')} of {data.get('points')} points done"
    if data["status"] == "failed":
        text += f"; point {index + 1} failed"
    return text


def _value_text(value: object) -> str:
    """Write a point's value: a number to four figures, a quantity as it is written.

    Parameters
    ----------
    value : object
        The value.

    Returns
    -------
    str
        Its text.

    Examples
    --------
    >>> _value_text(0.123456), _value_text("0.8 bar")
    ('0.1235', '0.8 bar')
    """
    return f"{value:.4g}" if isinstance(value, int | float) else str(value)


class ScanProgress(QWidget):
    """How far a scan has got: a bar for each variable, outermost first, and in all.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    bars : list of PySide6.QtWidgets.QProgressBar
        One per variable, the last the innermost, then one for the whole scan.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the place the bars go, empty until a scan starts."""
        super().__init__(parent)
        self.setObjectName("page")
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(4)
        self.bars: list[QProgressBar] = []
        self._names: list[str] = []
        self._values: list[list[str]] = []
        self.hide()

    def start(self, data: Mapping[str, object]) -> None:
        """Lay out a bar per variable for a scan starting.

        Parameters
        ----------
        data : Mapping
            The ``scan_started`` event's data: ``points``, and its ``shape``, the
            variables' ``names`` and each one's ``values``.
        """
        while self.form.rowCount():
            self.form.removeRow(0)
        self._names = [str(n) for n in data.get("names") or []]  # pyright: ignore[reportGeneralTypeIssues]
        self._values = [list(v) for v in data.get("values") or []]  # pyright: ignore[reportGeneralTypeIssues]
        shape: Sequence[int] = data.get("shape") or []  # pyright: ignore[reportAssignmentType]
        self.bars = []
        for name, size in zip(self._names, shape, strict=False):
            bar = QProgressBar()
            bar.setRange(0, int(size))
            bar.setFormat("%v of %m")
            self.form.addRow(label(name.split("[", 1)[0]) + _index_of(name), bar)
            self.bars.append(bar)
        whole = QProgressBar()
        whole.setRange(0, int(data.get("points") or 0))  # pyright: ignore[reportArgumentType]
        whole.setFormat("%v of %m points done")
        self.form.addRow("In all", whole)
        self.bars.append(whole)
        self.show()

    def point(self, data: Mapping[str, object]) -> None:
        """Follow a point: where it lies as it starts, the total as it ends.

        Parameters
        ----------
        data : Mapping
            A ``point`` event's data: its ``place`` as it starts, or ``done``.
        """
        if not self.bars:
            return
        if data.get("status") is None:
            place: Sequence[int] = data.get("place") or []  # pyright: ignore[reportAssignmentType]
            for axis, (bar, at) in enumerate(zip(self.bars[:-1], place, strict=False)):
                bar.setValue(int(at) + 1)
                values = self._values[axis] if axis < len(self._values) else []
                shown = values[int(at)] if int(at) < len(values) else ""
                bar.setFormat(f"%v of %m: {shown}")
            return
        self.bars[-1].setValue(int(data.get("done") or 0))  # pyright: ignore[reportArgumentType]


def _index_of(name: str) -> str:
    """Return the index part of a nested path, to tell its bar apart.

    Parameters
    ----------
    name : str
        The path, as ``pressure[0]``.

    Returns
    -------
    str
        `` [0]``, or nothing for a plain keyword.

    Examples
    --------
    >>> _index_of("pressure[0]"), _index_of("energy")
    (' [0]', '')
    """
    return f" [{name.split('[', 1)[1]}" if "[" in name else ""


class ScanRecord(QWidget):
    """What a scan records at each point beyond its statistics.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    band : PySide6.QtWidgets.QCheckBox
        Whether to record the energy and the peak power within a band.
    spectrum : PySide6.QtWidgets.QCheckBox
        Whether to record each point's spectrum.
    pulse : PySide6.QtWidgets.QCheckBox
        Whether to record each point's pulse.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the choices, all off."""
        super().__init__(parent)
        self.setObjectName("page")
        self.band = QCheckBox("Energy and peak power within a band")
        self.low = QuantityField("m", "nm")
        self.high = QuantityField("m", "nm")
        self.low.number.setPlaceholderText("from")
        self.high.number.setPlaceholderText("to")
        self.spectrum = QCheckBox("The spectrum")
        self.pulse = QCheckBox("The pulse")
        self.at = QuantityField("m", "mm")
        self.at.number.setPlaceholderText("the last plane")
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        form.addRow(self.band)
        form.addRow("From", self.low)
        form.addRow("To", self.high)
        form.addRow(self.spectrum)
        form.addRow(self.pulse)
        form.addRow("At z", self.at)
        self.allow_curves(one=True)

    def allow_curves(self, *, one: bool) -> None:
        """Offer a spectrum and a pulse only over one variable, where they draw a map.

        Parameters
        ----------
        one : bool
            Whether the scan has one variable.
        """
        for box in (self.spectrum, self.pulse):
            box.setEnabled(one)
            box.setToolTip(
                "" if one else "A spectrum or a pulse is drawn over one variable."
            )
            if not one:
                box.setChecked(False)

    def specs(self) -> list[dict[str, object]]:
        """Return the reductions chosen, as a run's spec carries them.

        Returns
        -------
        list of dict
            Each ``kind``, with its ``band`` (m) and ``z`` (m) where it has them.
        """
        z = self.at.read()
        where: dict[str, object] = {}
        if z is not None:
            where["z"] = float(parse_quantity(z, Unit("m"), path="z"))
        chosen: list[dict[str, object]] = []
        low, high = self.low.read(), self.high.read()
        if self.band.isChecked() and low is not None and high is not None:
            band = [
                float(parse_quantity(value, Unit("m"), path="band"))
                for value in (low, high)
            ]
            for kind in ("energy_in_band", "peak_power_in_band"):
                chosen.append({"kind": kind, "band": band, **where})
        for kind, box in (("spectrum", self.spectrum), ("pulse", self.pulse)):
            if box.isChecked() and box.isEnabled():
                chosen.append({"kind": kind, **where})
        return chosen


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
    point_chosen : Signal
        Emitted with a point's flat index when it is to be opened as a setup.
    """

    point_chosen = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the choice and the plot, hidden until a scan is shown."""
        super().__init__(parent)
        self.setObjectName("page")
        self.scan: Mapping[str, object] = {}
        self._xscale = 1.0
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
        self.map = MapPlot(xlabel="", ylabel="", colours="viridis", unit="")
        self.plot.more_actions = self._point_actions
        self.map.more_actions = self._point_actions
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self.result)
        column.addWidget(self.status)
        column.addWidget(self.plot, 1)
        column.addWidget(self.map, 1)
        self.result.hide()
        self.plot.hide()
        self.map.hide()

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
        self.redraw()

    def redraw(self) -> None:
        """Draw the result chosen."""
        name = self.result.currentData()
        variables = self.scan.get("variables") or []
        results: Mapping[str, Mapping[str, object]] = self.scan.get("results") or {}  # pyright: ignore[reportAssignmentType]
        if name is None or not variables:
            return
        held = results[str(name)]
        self.plot.setVisible("x" not in held)
        self.map.setVisible("x" in held)
        if "x" in held:
            self._draw_map(str(name), held, variables[0])  # pyright: ignore[reportArgumentType, reportIndexIssue]
            return
        drawn = scan_lines(
            self.scan,
            str(name),
            {str(v["name"]): label(str(v["name"])) for v in variables},  # pyright: ignore[reportGeneralTypeIssues]
        )
        ys = [y for _title, y in drawn.lines]
        unit, scale = scaled_unit(np.concatenate(ys), pretty(drawn.unit))
        xunit, xscale = scaled_unit(drawn.x, drawn.xunit) if drawn.xunit else ("", 1.0)
        self._xscale = xscale
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

    def point_at(self, x: float, y: float) -> int | None:
        """Return the scan point drawn nearest a position.

        Parameters
        ----------
        x : float
            The position across the plot shown.
        y : float
            The position up it.

        Returns
        -------
        int or None
            The point's flat index, the third variable on at its first value;
            ``None`` with no scan.
        """
        variables: Sequence[Mapping[str, object]] = self.scan.get("variables") or []  # pyright: ignore[reportAssignmentType]
        if not variables:
            return None
        shape = [len(v["values"]) for v in variables]  # pyright: ignore[reportArgumentType]
        first = np.asarray(variables[0]["values"], dtype=float)
        place = [0] * len(shape)
        if self.map.isVisibleTo(self):
            place[0] = int(np.argmin(np.abs(first - y)))
        else:
            place[0] = int(np.argmin(np.abs(first * self._xscale - x)))
            near = self.plot.nearest_curve(x, y)
            if len(shape) > 1 and near is not None:
                place[1] = near[0]
        return int(np.ravel_multi_index(place, shape))

    def _point_actions(self, point: QPointF) -> list[tuple[str, Callable[[], None]]]:
        """Offer to open the point nearest the pointer as a setup.

        Parameters
        ----------
        point : PySide6.QtCore.QPointF
            Where the menu was asked for, in the plot's coordinates.

        Returns
        -------
        list of tuple
            The action, where a scan is shown.
        """
        index = self.point_at(point.x(), point.y())
        if index is None:
            return []

        def choose() -> None:
            """Ask for the point's setup."""
            self.point_chosen.emit(index)

        return [("Open this point as a setup", choose)]

    def _draw_map(
        self, name: str, held: Mapping[str, object], variable: Mapping[str, object]
    ) -> None:
        """Draw a spectrum or a pulse at each value of the scan's first variable.

        Parameters
        ----------
        name : str
            The result.
        held : Mapping
            Its ``values`` (a row per point) and ``x`` (m or s).
        variable : Mapping
            The variable: ``name``, ``values`` and ``unit``.
        """
        spectrum = name.endswith("spectrum")
        x = np.asarray(held["x"], dtype=float) * (1e9 if spectrum else 1e15)
        values = np.asarray(held["values"], dtype=float)
        rows = values.reshape(values.shape[0], -1, x.size)[:, 0, :]
        along = np.asarray(variable["values"], dtype=float)
        unit = variable_unit(variable)
        order = np.argsort(along)
        along, rows = along[order], rows[order]
        picked, first, step = even_rows(along)
        if spectrum:
            shown = decibels(np.nan_to_num(rows))[picked]
            levels = (-40.0, 0.0)
            self.map.set_unit("dB")
        else:
            power_unit, scale = scaled_unit(np.nan_to_num(rows), "W")
            shown = np.nan_to_num(rows)[picked] * scale
            levels = (0.0, float(shown.max()) or 1.0)
            self.map.set_unit(power_unit)
        self.map.set_xlabel("Wavelength (nm)" if spectrum else "Time (fs)")
        title = label(str(variable["name"]).split("[", 1)[0])
        self.map.set_ylabel(f"{title} ({unit})" if unit else title)
        span = (float(x[0]), float(x[-1]))
        if not spectrum:
            half = pulse_window(x, np.nan_to_num(rows).max(axis=0))
            span = (-half, half)
        self.map.show_map(
            shown,
            (float(x[0]), first - step / 2, float(x[-1] - x[0]), step * picked.size),
            (span, (first - step / 2, first + step * (picked.size - 0.5))),
            levels,
        )

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
        self.map.hide()
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
        column.addSpacing(8)
        column.addWidget(named_label("Also record at each point", "h3"))
        self.record = ScanRecord()
        column.addWidget(self.record)
        column.addStretch(1)
        self.editor.changed.connect(self._variables_changed)
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
        self.bars = ScanProgress()
        side.addWidget(self.bars)
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

    def _variables_changed(self, table: object) -> None:
        """Offer a spectrum and a pulse only while the scan has one variable.

        Parameters
        ----------
        table : object
            The scan's table, or ``None``.
        """
        count = (
            0
            if not isinstance(table, Mapping)
            else len([key for key in table if key not in ("name", "base_seed")])
        )
        self.record.allow_curves(one=count <= 1)

    def show_table(self, table: object) -> None:
        """Show the setup's ``[scan]`` table.

        Parameters
        ----------
        table : object
            The table, or ``None``.
        """
        self.editor.set_value(table)
        self._variables_changed(table)

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
            self.bars.start(data)
            return
        self.progress.setText(point_text(data))
        self.bars.point(data)

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
