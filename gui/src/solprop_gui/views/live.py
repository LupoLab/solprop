# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Overview: spectrum and power along the guide, and the plane at a position.

One view serves the live run (rows arrive as the planes are saved) and the results
of a file (all rows at once). Its specification (design 12.8) is fixed and stated on
the view:

- the spectral energy density summed over modes, on a wavelength axis, in dB
  relative to the peak of the first plane shown;
- the power summed over modes, in time, normalised to the largest value shown;
- both reduced for display by keeping each bin's largest value, which the axis
  labels say.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

__all__ = ["DB_FLOOR", "OverviewView", "decibels"]

DB_FLOOR = -60.0
"""The default bottom of the dB scale."""


def decibels(density: np.ndarray, reference: float) -> np.ndarray:
    """Return a density in dB relative to a reference.

    Parameters
    ----------
    density : numpy.ndarray
        The density.
    reference : float
        The value that is 0 dB.

    Returns
    -------
    numpy.ndarray
        ``10 log10(density / reference)``, with zero and negative values at -300 dB.
    """
    ratio = np.asarray(density, dtype=float) / (reference if reference > 0 else 1.0)
    return 10.0 * np.log10(np.maximum(ratio, 1e-30))


def _colormap(name: str) -> pg.ColorMap:
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
    colormap = pg.colormap.get(name)
    if colormap is None:
        raise LookupError(name)
    return colormap


class OverviewView(QWidget):
    """Spectrum and power maps along z, and the selected plane's lines.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a layout
        """Make the plots, empty."""
        super().__init__(parent)
        self.wavelength = np.zeros(0)
        self.time = np.zeros(0)
        self.rows: list[Mapping[str, object]] = []
        self.reference = 0.0

        self.plots = pg.GraphicsLayoutWidget()
        self.spectrum_map = self.plots.ci.addPlot(row=0, col=0)
        self.spectrum_map.setLabel("bottom", "Wavelength (nm), display-binned")
        self.spectrum_map.setLabel("left", "z (m)")
        self.spectrum_map.setTitle("Spectral energy density (dB)")
        self.spectrum_image = pg.ImageItem(axisOrder="row-major")
        self.spectrum_image.setColorMap(_colormap("viridis"))
        self.spectrum_map.addItem(self.spectrum_image)
        self.spectrum_marker = pg.InfiniteLine(angle=0, movable=False, pen="w")
        self.spectrum_map.addItem(self.spectrum_marker)

        self.power_map = self.plots.ci.addPlot(row=0, col=1)
        self.power_map.setLabel("bottom", "Time (fs), display-binned")
        self.power_map.setLabel("left", "z (m)")
        self.power_map.setTitle("Power (normalised)")
        self.power_image = pg.ImageItem(axisOrder="row-major")
        self.power_image.setColorMap(_colormap("inferno"))
        self.power_map.addItem(self.power_image)
        self.power_marker = pg.InfiniteLine(angle=0, movable=False, pen="w")
        self.power_map.addItem(self.power_marker)

        self.spectrum_line = self.plots.ci.addPlot(row=1, col=0)
        self.spectrum_line.setLabel("bottom", "Wavelength (nm)")
        self.spectrum_line.setLabel("left", "dB")
        self.spectrum_curve = self.spectrum_line.plot(pen="c")
        self.spectrum_line.getViewBox().setXLink(self.spectrum_map.getViewBox())

        self.power_line = self.plots.ci.addPlot(row=1, col=1)
        self.power_line.setLabel("bottom", "Time (fs)")
        self.power_line.setLabel("left", "Power (GW)")
        self.power_curve = self.power_line.plot(pen="y")
        self.power_line.getViewBox().setXLink(self.power_map.getViewBox())

        self.energy_line = self.plots.ci.addPlot(row=2, col=0, colspan=2)
        self.energy_line.setLabel("bottom", "z (m)")
        self.energy_line.setLabel("left", "Energy (µJ)")
        self.energy_curve = self.energy_line.plot(pen="m", symbol="o", symbolSize=4)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("plane-slider")
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_plane)
        self.follow = QCheckBox("Follow the newest plane")
        self.follow.setChecked(True)
        self.position = QLabel("no planes yet")
        self.floor = QDoubleSpinBox()
        self.floor.setRange(-200.0, -5.0)
        self.floor.setValue(DB_FLOOR)
        self.floor.setSuffix(" dB")
        self.floor.setToolTip("The bottom of the dB scale.")
        self.floor.valueChanged.connect(self._redraw)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plane"))
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.position)
        controls.addWidget(self.follow)
        controls.addWidget(QLabel("Floor"))
        controls.addWidget(self.floor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plots, 1)
        layout.addLayout(controls)

    # ------------------------------------------------------------------ data

    def clear(self) -> None:
        """Forget every row and the axes."""
        self.rows = []
        self.wavelength = np.zeros(0)
        self.time = np.zeros(0)
        self.reference = 0.0
        self.spectrum_image.clear()
        self.power_image.clear()
        self.spectrum_curve.setData([], [])
        self.power_curve.setData([], [])
        self.energy_curve.setData([], [])
        self.slider.setEnabled(False)
        self.position.setText("no planes yet")

    def set_axes(self, axes: Mapping[str, object]) -> None:
        """Take the display bins' centres.

        Parameters
        ----------
        axes : Mapping
            ``wavelength`` (m) and ``time`` (s).
        """
        self.wavelength = np.asarray(axes["wavelength"], dtype=float)
        self.time = np.asarray(axes["time"], dtype=float)

    def add_row(self, row: Mapping[str, object]) -> None:
        """Add one plane's row, replacing a row of the same plane.

        Parameters
        ----------
        row : Mapping
            ``index``, ``z``, ``density``, ``power`` and ``stats``.
        """
        index = int(row["index"])  # pyright: ignore[reportArgumentType]
        self.rows = [r for r in self.rows if int(r["index"]) != index]  # pyright: ignore[reportArgumentType]
        self.rows.append(row)
        self.rows.sort(key=lambda r: int(r["index"]))  # pyright: ignore[reportArgumentType]
        if self.reference <= 0.0:
            self.reference = float(np.max(np.asarray(self.rows[0]["density"])))
        self._redraw()

    def set_rows(
        self, axes: Mapping[str, object], rows: Sequence[Mapping[str, object]]
    ) -> None:
        """Show a whole file's rows.

        Parameters
        ----------
        axes : Mapping
            The bins' centres.
        rows : Sequence of Mapping
            The rows, one per plane.
        """
        self.clear()
        if not rows:
            return
        self.set_axes(axes)
        self.rows = sorted(rows, key=lambda r: int(r["index"]))  # pyright: ignore[reportArgumentType]
        self.reference = float(np.max(np.asarray(self.rows[0]["density"])))
        self._redraw()

    # ------------------------------------------------------------------ drawing

    def _redraw(self) -> None:
        """Redraw the maps and the selected plane."""
        if not self.rows or self.wavelength.size == 0:
            return
        z = np.array([float(r["z"]) for r in self.rows])  # pyright: ignore[reportArgumentType]
        density = np.array([np.asarray(r["density"]) for r in self.rows])
        power = np.array([np.asarray(r["power"]) for r in self.rows])
        floor = self.floor.value()
        db = np.clip(decibels(density, self.reference), floor, 0.0)
        lam_nm = self.wavelength * 1e9
        t_fs = self.time * 1e15
        z_low, z_high = float(z.min()), float(z.max())
        height = (z_high - z_low) or 1e-9
        self.spectrum_image.setImage(db, levels=(floor, 0.0), autoLevels=False)
        self.spectrum_image.setRect(
            QRectF(lam_nm[0], z_low, lam_nm[-1] - lam_nm[0], height)
        )
        peak = float(power.max()) or 1.0
        self.power_image.setImage(power / peak, levels=(0.0, 1.0), autoLevels=False)
        self.power_image.setRect(QRectF(t_fs[0], z_low, t_fs[-1] - t_fs[0], height))
        energy = [
            float(r["stats"].get("energy", np.nan)) * 1e6  # pyright: ignore[reportAttributeAccessIssue]
            for r in self.rows
        ]
        self.energy_curve.setData(z, energy)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.rows) - 1)
        if self.follow.isChecked():
            self.slider.setValue(len(self.rows) - 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(len(self.rows) > 1)
        self._show_plane(self.slider.value())

    def _show_plane(self, position: int) -> None:
        """Draw the lines of one plane.

        Parameters
        ----------
        position : int
            The plane's place among the rows.
        """
        if not self.rows:
            return
        position = min(max(position, 0), len(self.rows) - 1)
        if position != len(self.rows) - 1:
            self.follow.setChecked(False)
        row = self.rows[position]
        z = float(row["z"])  # pyright: ignore[reportArgumentType]
        db = decibels(np.asarray(row["density"]), self.reference)
        self.spectrum_curve.setData(
            self.wavelength * 1e9, np.maximum(db, self.floor.value())
        )
        self.power_curve.setData(self.time * 1e15, np.asarray(row["power"]) / 1e9)
        self.spectrum_marker.setValue(z)
        self.power_marker.setValue(z)
        self.position.setText(f"plane {int(row['index'])}, z = {z:.4g} m")  # pyright: ignore[reportArgumentType]
