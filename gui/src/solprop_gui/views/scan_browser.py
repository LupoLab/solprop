# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scan browser: one result of a scan over its variables (design 12.7).

A scan's results (each point's statistics and reductions, at its last plane) are drawn
over the scan's variables: a line over one variable, a map over the first two (the
others held at their first value). Failed points are marked; points not run are gaps.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from solprop_gui.registry.pages import label
from solprop_gui.views.plotting import PALETTE, HoverLayout, colormap, curve_lookup

__all__ = ["ScanView"]


class ScanView(QWidget):
    """A scan's results over its variables.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the result picker and the empty plot."""
        super().__init__(parent)
        self.scan: Mapping[str, object] = {}
        self.result = QComboBox()
        self.result.setObjectName("scan-result")
        self.result.currentIndexChanged.connect(self.redraw)
        self.status = QLabel("No scan open.")
        self.status.setObjectName("scan-status")
        self.status.setWordWrap(True)
        self.plots = HoverLayout()
        self.plot = self.plots.add_plot(0, 0, "")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.curve = self.plot.plot(
            pen=pg.mkPen(PALETTE[0], width=1.4), symbol="o", symbolSize=6
        )
        self.failed = pg.ScatterPlotItem(symbol="x", size=10, pen=pg.mkPen("#c62828"))
        self.plot.addItem(self.failed)
        self.image = pg.ImageItem()
        self.image.setColorMap(colormap("viridis"))
        self.plot.addItem(self.image)
        self.plots.add_readout(
            self.plot, curve_lookup(self.curve, "{0:.4g}: {1:.4g}"), dot=True
        )
        line = QHBoxLayout()
        line.addWidget(QLabel("Result"))
        line.addWidget(self.result, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addLayout(line)
        layout.addWidget(self.status)
        layout.addWidget(self.plots, 1)

    def set_scan(self, scan: Mapping[str, object]) -> None:
        """Show a scan's overview.

        Parameters
        ----------
        scan : Mapping
            The kernel's ``scan_overview`` reply.
        """
        self.scan = scan
        results: Mapping[str, object] = scan.get("results") or {}  # pyright: ignore[reportAssignmentType]
        self.result.blockSignals(True)
        self.result.clear()
        for name in results:
            self.result.addItem(name, name)
        self.result.blockSignals(False)
        status = np.asarray(scan.get("status") or [], dtype=object).ravel()
        done = int(np.count_nonzero(status == "done"))
        failed = int(np.count_nonzero(status == "failed"))
        variables: Sequence[Mapping[str, object]] = scan.get("variables") or []  # pyright: ignore[reportAssignmentType]
        over = " × ".join(label(str(v["name"])).lower() for v in variables)  # noqa: RUF001 - times
        self.status.setText(
            f"{status.size} points over {over or 'nothing'}: {done} done, "
            f"{failed} failed. Each point's value is at its last plane, summed over "
            "its modes."
        )
        self.redraw()

    def redraw(self, *_args: object) -> None:
        """Draw the result picked over the variables.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        self.curve.setData([], [])
        self.failed.setData([], [])
        self.image.hide()
        name = self.result.currentData()
        results: Mapping[str, Mapping[str, object]] = self.scan.get("results") or {}  # pyright: ignore[reportAssignmentType]
        variables: Sequence[Mapping[str, object]] = self.scan.get("variables") or []  # pyright: ignore[reportAssignmentType]
        if name is None or name not in results or not variables:
            return
        entry = results[name]
        values = np.asarray(entry["values"], dtype=float)
        status = np.asarray(self.scan.get("status"), dtype=object)
        units = str(entry["units"])
        self.plot.setTitle(str(entry["summary"]))
        first = variables[0]
        x = np.asarray(first["values"], dtype=float)
        self.plot.setLabel("bottom", label(str(first["name"])))
        if values.ndim == 1 or len(variables) == 1:
            line = values.reshape(x.size, -1)[:, 0]
            self.curve.setData(x, line, connect="finite")
            bad = status.reshape(x.size, -1)[:, 0] == "failed"
            self.failed.setData(x[bad], np.zeros(int(bad.sum())))
            self.plot.setLabel("left", name, units=units or None)
            return
        # Over the first two variables, the rest at their first value.
        second = variables[1]
        y = np.asarray(second["values"], dtype=float)
        grid = values.reshape(x.size, y.size, -1)[:, :, 0]
        self.image.setImage(grid)
        self.image.setRect(
            float(x.min()),
            float(y.min()),
            float(np.ptp(x)) or 1.0,
            float(np.ptp(y)) or 1.0,
        )
        self.image.show()
        self.plot.setLabel("left", label(str(second["name"])))

    def show_error(self, message: str) -> None:
        """Say why the scan is not shown.

        Parameters
        ----------
        message : str
            The reason.
        """
        self.scan = {}
        self.result.clear()
        self.redraw()
        self.status.setText(message)
