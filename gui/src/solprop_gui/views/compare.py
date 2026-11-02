# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare runs: their last planes' spectra and power, overlaid (design 11.2, Q10).

The spectra share **one** dB reference, the highest first-plane peak among the runs
shown, so a difference in energy stays visible; the view says so. Each run is reduced
on its own display bins, as its overview is.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from solprop_gui.views.live import decibels
from solprop_gui.views.plotting import HoverLayout

__all__ = ["CompareView"]

_PENS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


class CompareView(QWidget):
    """Overlaid lines of several runs.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the two plots, empty."""
        super().__init__(parent)
        self.runs: dict[
            str, tuple[Mapping[str, object], list[Mapping[str, object]]]
        ] = {}
        self.plots = HoverLayout()
        self.spectra = self.plots.ci.addPlot(row=0, col=0)
        self.spectra.setTitle("Last plane: spectral energy density (dB, one reference)")
        self.spectra.setLabel("bottom", "Wavelength (nm)")
        self.spectra.setLabel("left", "dB")
        self.spectra_legend = self.spectra.addLegend()
        self.power = self.plots.ci.addPlot(row=1, col=0)
        self.power.setTitle("Last plane: power")
        self.power.setLabel("bottom", "Time (fs)")
        self.power.setLabel("left", "Power (GW)")
        self.power_legend = self.power.addLegend()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plots)

    def add_run(
        self,
        name: str,
        axes: Mapping[str, object],
        rows: Sequence[Mapping[str, object]],
    ) -> None:
        """Add a run, or replace one of the same name.

        Parameters
        ----------
        name : str
            The run's name in the legend.
        axes : Mapping
            Its display bins' centres.
        rows : Sequence of Mapping
            Its overview rows.
        """
        if rows:
            self.runs[name] = (axes, sorted(rows, key=lambda r: int(r["index"])))  # pyright: ignore[reportArgumentType]
            self._redraw()

    def clear(self) -> None:
        """Remove every run."""
        self.runs = {}
        self._redraw()

    def reference(self) -> float:
        """Return the shared dB reference.

        Returns
        -------
        float
            The largest first-plane density peak among the runs; 0 with none.
        """
        peaks = [
            float(np.max(np.asarray(rows[0]["density"])))
            for _, rows in self.runs.values()
        ]
        return max(peaks, default=0.0)

    def _redraw(self) -> None:
        """Redraw every run's lines."""
        self.spectra.clear()
        self.power.clear()
        self.spectra_legend.clear()
        self.power_legend.clear()
        reference = self.reference()
        for number, (name, (axes, rows)) in enumerate(self.runs.items()):
            pen = _PENS[number % len(_PENS)]
            last = rows[-1]
            wavelength = np.asarray(axes["wavelength"], dtype=float) * 1e9
            time = np.asarray(axes["time"], dtype=float) * 1e15
            db = np.maximum(decibels(np.asarray(last["density"]), reference), -80.0)
            self.spectra.plot(wavelength, db, pen=pg.mkPen(pen, width=1.5), name=name)
            self.power.plot(
                time,
                np.asarray(last["power"]) / 1e9,
                pen=pg.mkPen(pen, width=1.5),
                name=name,
            )
