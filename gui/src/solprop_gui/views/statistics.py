# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Statistics view: every recorded statistic against z, and the step history.

The statistics a run recorded (``stats/*``) are listed with solprop's own summary and
unit (``solprop.stats.CATALOGUE``); the one picked is drawn against z, a column per
mode and component where it has several. Below, the step size along the run, the
rejected steps marked, which is where a run's cost and its difficult stretches show.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.views.plotting import PALETTE, HoverLayout, choose_figure, curve_lookup

__all__ = ["StatisticsView"]


class StatisticsView(QWidget):
    """A results file's statistics against z.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    figure_wanted : Signal
        Emitted with a figure's ``file`` and ``plot`` (solprop's ``plot_stats``).
    """

    figure_wanted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the list and the empty plots."""
        super().__init__(parent)
        self.stats: list[Mapping[str, object]] = []
        self.z = np.zeros(0)
        self.names = QListWidget()
        self.names.setObjectName("stats-names")
        self.names.currentRowChanged.connect(self.redraw)
        self.summary = QLabel("No results file open.")
        self.summary.setObjectName("stats-summary")
        self.summary.setWordWrap(True)
        self.figure_button = QPushButton("Export figures…")
        self.figure_button.setToolTip("Every statistic, drawn by solprop (plot_stats).")
        self.figure_button.clicked.connect(self._choose_figure)
        self.plots = HoverLayout()
        self.plot = self.plots.add_plot(0, 0, "")
        self.steps = self.plots.add_plot(1, 0, "Step size")
        for plot in (self.plot, self.steps):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("bottom", "z (cm)")
        self.steps.setLabel("left", "Step", units="m")
        self.steps.setLogMode(x=False, y=True)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.step_curve = self.steps.plot(
            pen=pg.mkPen(PALETTE[0], width=1.0), name="accepted"
        )
        self.rejected = pg.ScatterPlotItem(
            symbol="x", size=7, pen=pg.mkPen("#c62828"), name="rejected"
        )
        self.steps.addItem(self.rejected)
        self.curves: list[pg.PlotDataItem] = []
        self.plots.add_readout(self.plot, self.lookup, dot=True)

        side = QWidget()
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(QLabel("Statistics recorded"))
        column.addWidget(self.names, 1)
        column.addWidget(self.figure_button)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(side)
        split.addWidget(self.plots)
        split.setStretchFactor(1, 4)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        line = QHBoxLayout()
        line.addWidget(self.summary, 1)
        layout.addLayout(line)
        layout.addWidget(split, 1)

    def set_statistics(self, reply: Mapping[str, object]) -> None:
        """Show a file's statistics and step history.

        Parameters
        ----------
        reply : Mapping
            The kernel's ``statistics`` reply.
        """
        self.z = np.asarray(reply["z"], dtype=float)
        self.stats = list(reply["stats"])  # pyright: ignore[reportArgumentType]
        self.names.blockSignals(True)
        self.names.clear()
        for entry in self.stats:
            item = QListWidgetItem(str(entry["name"]))
            item.setToolTip(f"{entry['summary']} ({entry['units'] or 'no unit'})")
            self.names.addItem(item)
        self.names.blockSignals(False)
        steps: Mapping[str, object] = reply["steps"]  # pyright: ignore[reportAssignmentType]
        z = np.asarray(steps["z"], dtype=float) * 100
        h = np.asarray(steps["h"], dtype=float)
        self.step_curve.setData(z, h, connect="finite")
        rejected_z = np.asarray(steps["rejected_z"], dtype=float) * 100
        rejected_h = np.asarray(steps["rejected_h"], dtype=float)
        self.rejected.setData(rejected_z, np.log10(np.maximum(rejected_h, 1e-30)))
        self.figure_button.setEnabled(bool(self.stats))
        if self.stats:
            self.names.setCurrentRow(0)
        else:
            self.redraw(-1)
        self.summary.setText(
            f"{len(self.stats)} statistics at {self.z.size} planes; "
            f"{z.size} accepted steps, {rejected_z.size} rejected."
        )

    def redraw(self, row: int) -> None:
        """Draw the statistic picked.

        Parameters
        ----------
        row : int
            Its row; -1 for none.
        """
        for curve in self.curves:
            self.plot.removeItem(curve)
        self.curves = []
        self.legend.clear()
        if row < 0 or row >= len(self.stats):
            self.plot.setTitle("")
            return
        entry = self.stats[row]
        units = str(entry["units"])
        self.plot.setTitle(str(entry["summary"]))
        # A plain unit is pyqtgraph's to prefix (GW, fs); any other goes in the label.
        plain = units in ("J", "W", "s", "m", "rad")
        name = str(entry["name"])
        self.plot.setLabel(
            "left",
            name if plain or not units else f"{name} ({units})",
            units=units if plain else None,
        )
        columns: Sequence[Mapping[str, object]] = entry["columns"]  # pyright: ignore[reportAssignmentType]
        for number, column in enumerate(columns):
            curve = self.plot.plot(
                self.z * 100,
                np.asarray(column["y"], dtype=float),
                pen=pg.mkPen(PALETTE[number % len(PALETTE)], width=1.4),
                name=str(column["label"]) if len(columns) > 1 else None,
                symbol="o",
                symbolSize=4,
            )
            self.curves.append(curve)

    def lookup(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Read the statistic's curve nearest the mouse.

        Parameters
        ----------
        x, y : float
            The mouse, in the plot's coordinates.

        Returns
        -------
        tuple or None
            The point and its label; ``None`` with nothing drawn.
        """
        best: tuple[float, float, str] | None = None
        distance = np.inf
        for curve in self.curves:
            hit = curve_lookup(curve, "z = {0:.4g} cm: {1:.4g}")(x, y)
            if hit is not None and abs(hit[1] - y) < distance:
                distance = abs(hit[1] - y)
                best = hit
        return best

    def _choose_figure(self) -> None:  # pragma: no cover - a modal dialog
        """Ask where to write the figures, and ask for them."""
        path = choose_figure(self, "statistics.pdf")
        if path:
            self.export_figure(path)

    def export_figure(self, path: str) -> None:
        """Ask for solprop's figures of every statistic.

        Parameters
        ----------
        path : str
            The file (numbered for several).
        """
        self.figure_wanted.emit({"file": path, "plot": "plot_stats"})

    def clear(self) -> None:
        """Forget the file shown."""
        self.set_statistics(
            {
                "z": [],
                "stats": [],
                "steps": {"z": [], "h": [], "rejected_z": [], "rejected_h": []},
            }
        )
        self.summary.setText("No results file open.")
