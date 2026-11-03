# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Compression view: the pulse before, after the compressor, and at its limit.

The Post-process step's result (:func:`solprop_gui.service.postprocess.compress_chain`):
the power before the compressor, after it and at the transform limit, with solprop's
width and peak of each, and what each searched element found.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.compressor import KINDS
from solprop_gui.registry.units import si
from solprop_gui.views.live import set_range
from solprop_gui.views.plotting import PALETTE, HoverLayout, curve_lookup

__all__ = ["CompressionView"]

_TRACES = (
    ("before", "before", "#888888", Qt.PenStyle.SolidLine),
    ("after", "after the compressor", PALETTE[0], Qt.PenStyle.SolidLine),
    ("limit", "transform limit", PALETTE[2], Qt.PenStyle.DashLine),
)


class CompressionView(QWidget):
    """A compressor's effect on the pulse, and what its search found.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the empty plot and table."""
        super().__init__(parent)
        self.summary = QLabel("Build a compressor on the Post-process step.")
        self.summary.setObjectName("compression-summary")
        self.summary.setWordWrap(True)
        self.plots = HoverLayout()
        self.power = self.plots.add_plot(0, 0, "Power")
        self.power.setLabel("bottom", "Time (fs)")
        self.power.setLabel("left", "Power", units="W")
        self.power.showGrid(x=True, y=True, alpha=0.15)
        self.power.addLegend(offset=(-10, 10))
        self.curves = {
            key: self.power.plot(
                pen=pg.mkPen(colour, width=1.5, style=style), name=name
            )
            for key, name, colour, style in _TRACES
        }
        self.plots.add_readout(
            self.power, curve_lookup(self.curves["after"], "{0:.1f} fs, {1:.4g} W")
        )
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("compression-stages")
        self.table.setHorizontalHeaderLabels(["Element", "Found", "Quality", "Search"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.plots)
        split.addWidget(self.table)
        split.setSizes([480, 160])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addWidget(self.summary)
        layout.addWidget(split, 1)

    def show_compression(
        self,
        found: Mapping[str, object],
        limits: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """Draw a compressor's result.

        Parameters
        ----------
        found : Mapping
            The kernel's ``compress_chain`` reply.
        limits : tuple, optional
            The plot options' time range, fs; where none is typed, the view spans
            the light (each trace above a hundredth of its peak), widened by half.
        """
        t = np.asarray(found["t"], dtype=float) * 1e15
        for key, curve in self.curves.items():
            curve.setData(t, np.asarray(found[key], dtype=float))
        pulses: Mapping[str, Mapping[str, float]] = found["pulses"]  # pyright: ignore[reportAssignmentType]
        parts = []
        for key, name, _, _ in _TRACES:
            pulse = pulses[key]
            parts.append(
                f"{name}: {pulse['fwhm'] * 1e15:.3g} fs, {si(pulse['peak'], 'W')}"
            )
        share = pulses["after"]["peak"] / pulses["limit"]["peak"]
        self.summary.setText(
            "; ".join(parts) + f". Peak power {share:.0%} of the transform limit's."
        )
        stages: Sequence[Mapping[str, object]] = found["stages"]  # pyright: ignore[reportAssignmentType]
        self.table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            auto = bool(stage.get("auto"))
            cells = (
                f"{KINDS.get(str(stage['kind']), stage['kind'])}"
                + (" (auto)" if auto else ""),
                str(stage.get("text", "fixed, as given")),
                f"{float(stage['quality']):.3g}" if auto else "",  # pyright: ignore[reportArgumentType]
                ("converged" if stage.get("success") else str(stage.get("message")))
                if auto
                else "",
            )
            for column, text in enumerate(cells):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.power.getViewBox().enableAutoRange(axis="y")
        if limits == (None, None):
            lit = np.zeros(t.size, dtype=bool)
            for key, _, _, _ in _TRACES:
                trace = np.asarray(found[key], dtype=float)
                lit |= trace > 1e-2 * trace.max()
            low, high = t[lit].min(), t[lit].max()
            pad = 0.5 * max(high - low, 1.0)
            limits = (low - pad, high + pad)
        set_range(self.power, "x", limits)

    def clear(self) -> None:
        """Forget the result shown."""
        for curve in self.curves.values():
            curve.setData([], [])
        self.table.setRowCount(0)
        self.summary.setText("Build a compressor on the Post-process step.")
