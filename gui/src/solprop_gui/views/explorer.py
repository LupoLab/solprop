# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The waveguide explorer: the modes' dispersion, loss, area and nonlinearity (7.4).

The view draws what the kernel's ``modes`` request gives (``Recipe.modes``): one curve
per mode, or per mode and value of a swept keyword, of the chosen quantity against
wavelength, frequency or photon energy, with each mode's zero-dispersion wavelength and
the pump marked, and a table of every quantity at the pump. It asks for nothing itself:
the window sends the request when the setup or the sweep changes.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.registry.explore import (
    AXES,
    QUANTITIES,
    SWEEPS,
    axis_values,
    quantity_values,
    sweep_values,
    table_rows,
    value_at,
)
from solprop_gui.registry.pages import label
from solprop_gui.views.plotting import PALETTE, HoverLayout, colormap
from solprop_gui.views.splitter import GripSplitter

__all__ = ["Curve", "ExplorerView"]

_STYLES = (
    Qt.PenStyle.SolidLine,
    Qt.PenStyle.DashLine,
    Qt.PenStyle.DotLine,
    Qt.PenStyle.DashDotLine,
)

Curve = tuple[str, np.ndarray, np.ndarray]
"""One drawn curve: its name, its abscissae and its values."""


class ExplorerView(QWidget):
    """The modes' quantities against frequency, for the setup or a sweep of it.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    sweep_changed : Signal
        Emitted when the user applies a sweep, or clears it: the window asks again.
    """

    sweep_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a layout
        """Make the controls and the empty plot."""
        super().__init__(parent)
        self.sets: list[Mapping[str, object]] = []
        self.pump: float | None = None
        self.swept: str | None = None
        self.hidden: set[str] = set()
        self.curves: list[Curve] = []

        self.quantity = QComboBox()
        self.quantity.setObjectName("explorer-quantity")
        for quantity in QUANTITIES:
            self.quantity.addItem(quantity.title, quantity.key)
        self.quantity.setCurrentIndex(2)  # beta2
        self.axis = QComboBox()
        self.axis.setObjectName("explorer-axis")
        for axis in AXES:
            self.axis.addItem(axis.title, axis.key)
        self.modes_button = QToolButton()
        self.modes_button.setText("Modes")
        self.modes_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.modes_menu = QMenu(self)
        self.modes_button.setMenu(self.modes_menu)
        self.sweep_name = QComboBox()
        self.sweep_name.setObjectName("sweep-name")
        self.sweep_name.addItem("no sweep", None)
        self.sweep_name.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.sweep_text = QLineEdit()
        self.sweep_text.setObjectName("sweep-values")
        self.sweep_text.setPlaceholderText("values, e.g. 0.5, 1, 2 bar")
        self.sweep_text.returnPressed.connect(self.sweep_changed)
        self.apply = QPushButton("Sweep")
        self.apply.setObjectName("sweep-apply")
        self.apply.clicked.connect(self.sweep_changed)
        self.export_button = QToolButton()
        self.export_button.setText("Export CSV…")
        self.export_button.clicked.connect(self.choose_export)
        for box in (self.quantity, self.axis):
            box.currentIndexChanged.connect(self.redraw)
        self.status = QLabel("Open a setup on a guided route to see its modes.")
        self.status.setObjectName("explorer-status")
        self.status.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(self.quantity)
        controls.addWidget(QLabel("against"))
        controls.addWidget(self.axis)
        controls.addWidget(self.modes_button)
        controls.addSpacing(12)
        controls.addWidget(self.sweep_name)
        controls.addWidget(self.sweep_text, 1)
        controls.addWidget(self.apply)
        controls.addWidget(self.export_button)

        self.plots = HoverLayout()
        self.plot = self.plots.add_plot(0, 0, "")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plots.add_readout(self.plot, self.lookup, dot=True)
        self.table = QTableWidget()
        self.table.setObjectName("explorer-table")
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        split = GripSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.plots)
        split.addWidget(self.table)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)

    # ------------------------------------------------------------------ input

    def set_route(self, route: str) -> None:
        """Offer the keywords a route's sweep can vary.

        Parameters
        ----------
        route : str
            The route.
        """
        current = self.sweep_name.currentData()
        self.sweep_name.clear()
        self.sweep_name.addItem("no sweep", None)
        for name in SWEEPS.get(route, ()):
            self.sweep_name.addItem(f"sweep {label(name).lower()}", name)
        index = self.sweep_name.findData(current)
        self.sweep_name.setCurrentIndex(max(index, 0))

    def sweep(self) -> dict[str, object] | None:
        """Return the sweep asked for, as the kernel's ``modes`` request takes it.

        Returns
        -------
        dict or None
            ``name`` and ``values``; ``None`` for no sweep.
        """
        name = self.sweep_name.currentData()
        values = sweep_values(self.sweep_text.text())
        if name is None or not values:
            return None
        return {"name": name, "values": values}

    def busy(self, text: str) -> None:
        """Say what the view waits for.

        Parameters
        ----------
        text : str
            The message.
        """
        self.status.setText(text)

    def set_sets(self, reply: Mapping[str, object]) -> None:
        """Show the kernel's mode sets.

        Parameters
        ----------
        reply : Mapping
            The ``modes`` request's reply: ``pump``, ``sweep`` and ``sets``.
        """
        self.sets = list(reply.get("sets") or [])  # pyright: ignore[reportArgumentType]
        pump = reply.get("pump")
        self.pump = None if pump is None else float(pump)  # pyright: ignore[reportArgumentType]
        swept = reply.get("sweep")
        self.swept = None if swept is None else str(swept)
        errors = [
            f"{entry.get('value')}: {entry['error']}"
            for entry in self.sets
            if "error" in entry
        ]
        self.status.setText(
            "; ".join(errors)
            if errors
            else "From solprop's mode preview (Recipe.modes) at the launch position."
        )
        self._fill_modes_menu()
        self.redraw()

    def show_error(self, message: str) -> None:
        """Show why there are no modes to draw.

        Parameters
        ----------
        message : str
            The reason.
        """
        self.sets = []
        self.status.setText(message)
        self.redraw()

    # ------------------------------------------------------------------ drawing

    def labels(self) -> list[str]:
        """Return the mode labels of the first set that has modes.

        Returns
        -------
        list of str
            The labels.
        """
        for entry in self.sets:
            if "modes" in entry:
                return [str(m["label"]) for m in entry["modes"]]  # pyright: ignore[reportGeneralTypeIssues]
        return []

    def _fill_modes_menu(self) -> None:
        """List the modes, each a check to show or hide it."""
        self.modes_menu.clear()
        for name in self.labels():
            action = self.modes_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name not in self.hidden)
            action.toggled.connect(
                lambda shown, name=name: self._show_mode(name, shown)
            )

    def _show_mode(self, name: str, shown: bool) -> None:
        """Show or hide one mode's curves.

        Parameters
        ----------
        name : str
            The mode's label.
        shown : bool
            Whether to show it.
        """
        if shown:
            self.hidden.discard(name)
        else:
            self.hidden.add(name)
        self.redraw()

    def redraw(self) -> None:
        """Draw the curves, the markers and the table for the chosen quantity."""
        self.plot.clear()
        self.legend.clear()
        self.curves = []
        quantity = QUANTITIES[max(self.quantity.currentIndex(), 0)]
        axis = AXES[max(self.axis.currentIndex(), 0)].key
        self.plot.setLabel("bottom", AXES[max(self.axis.currentIndex(), 0)].title)
        self.plot.setLabel("left", quantity.title)
        drawn = [entry for entry in self.sets if "modes" in entry]
        ramp = colormap("viridis")
        for index, entry in enumerate(drawn):
            omega = np.asarray(entry["omega"], dtype=float)
            x = axis_values(omega, axis)
            for number, mode in enumerate(entry["modes"]):  # pyright: ignore[reportArgumentType]
                name = str(mode["label"])
                if name in self.hidden:
                    continue
                values = quantity_values(mode, omega, quantity)
                if values is None:
                    continue
                if len(drawn) > 1:
                    # Short of viridis's pale end, which a white plot loses.
                    colour = ramp.mapToQColor(0.85 * index / (len(drawn) - 1))
                    style = _STYLES[number % len(_STYLES)]
                    title = f"{name}, {entry['value']}"
                else:
                    colour = PALETTE[number % len(PALETTE)]
                    style = Qt.PenStyle.SolidLine
                    title = name
                pen = pg.mkPen(colour, width=1.6, style=style)
                self.plot.plot(x, values, pen=pen, name=title)
                self.curves.append((title, x, values))
                zero = mode.get("zdw")
                if zero is not None and index == 0 and quantity.source == "beta2":
                    first = not any(
                        isinstance(item, pg.InfiniteLine) for item in self.plot.items
                    )
                    position = axis_values(
                        np.array([2 * np.pi * 299_792_458.0 / float(zero)]), axis
                    )[0]
                    self.plot.addItem(
                        pg.InfiniteLine(
                            float(position),
                            angle=90,
                            pen=pg.mkPen(colour, style=Qt.PenStyle.DotLine),
                            label="ZDW" if first else None,
                            labelOpts={"position": 0.95, "color": colour},
                        )
                    )
        if quantity.source in ("beta2", "beta3"):
            self.plot.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen("#888888")))
        if self.pump is not None:
            position = axis_values(
                np.array([2 * np.pi * 299_792_458.0 / self.pump]), axis
            )[0]
            self.plot.addItem(
                pg.InfiniteLine(
                    float(position),
                    angle=90,
                    pen=pg.mkPen("#c62828", width=1.2),
                    label="pump",
                    labelOpts={"position": 0.88, "color": "#c62828"},
                )
            )
        self._fill_table()

    def _fill_table(self) -> None:
        """Write every quantity at the pump, one row per mode."""
        self.table.clear()
        if self.pump is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        rows = table_rows(self.sets, self.pump)
        self.table.setColumnCount(1 + len(QUANTITIES))
        self.table.setHorizontalHeaderLabels(
            [f"At {self.pump * 1e9:.4g} nm"] + [q.title for q in QUANTITIES]
        )
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(str(values["label"])))
            for column, quantity in enumerate(QUANTITIES, start=1):
                value = values[quantity.key]
                digits = 8 if quantity.key == "group_index" else 5
                text = "" if value is None else f"{value:.{digits}g}"
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def lookup(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Read the curve nearest the mouse.

        Parameters
        ----------
        x, y : float
            The mouse, in the plot's coordinates.

        Returns
        -------
        tuple or None
            The point on the nearest curve and its label; ``None`` with no curve
            under the mouse.
        """
        best: tuple[float, float, str] | None = None
        distance = np.inf
        for title, xs, ys in self.curves:
            value = value_at(xs, ys, x)
            if value is not None and abs(value - y) < distance:
                distance = abs(value - y)
                best = (x, value, f"{title}\n{x:.5g}\n{value:.5g}")
        return best

    # ------------------------------------------------------------------ export

    def choose_export(self) -> None:
        """Ask where to write the curves, and write them."""
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export curves", "modes.csv", "CSV (*.csv)"
        )
        if chosen:
            self.export(chosen)

    def export(self, path: str | Path) -> None:
        """Write the drawn curves as CSV: one abscissa and value column per curve.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        """
        quantity = QUANTITIES[max(self.quantity.currentIndex(), 0)].title
        axis = AXES[max(self.axis.currentIndex(), 0)].title
        columns: list[Sequence[object]] = []
        header: list[str] = []
        for title, xs, ys in self.curves:
            header.extend([f"{title}: {axis}", f"{title}: {quantity}"])
            columns.extend([list(xs), list(ys)])
        length = max((len(column) for column in columns), default=0)
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in range(length):
                writer.writerow(
                    [column[row] if row < len(column) else "" for column in columns]
                )
