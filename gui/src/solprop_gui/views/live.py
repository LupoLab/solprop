# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Overview: spectrum and power along the guide, and the plane at a position.

One view serves the live run (rows arrive as the planes are saved) and the results
of a file (all rows at once), laid out as the Qt prototype was: the two maps above,
the selected plane's spectrum, power and the energy below. Its specification (design
12.8) is fixed and stated on the view:

- the spectral energy density summed over modes, on a wavelength axis, in dB
  relative to the peak of the first plane shown;
- the power summed over modes, in time, in GW;
- both reduced for display by keeping each bin's largest value, which the maps'
  titles say.

Hovering reads a map's value or a curve's point; clicking a map, or dragging its z
marker, selects the plane.
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

from solprop_gui.views.plotting import BLUE, HoverLayout, colormap, curve_lookup
from solprop_gui.views.splitter import GripSplitter

__all__ = ["DB_FLOOR", "MapPanel", "OverviewView", "decibels"]

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


class MapPanel:
    """A false-colour map of one quantity against z, with a colour bar and a z marker.

    Parameters
    ----------
    layout : solprop_gui.views.plotting.HoverLayout
        The grid it is drawn in.
    column : int
        Its column in the grid.
    title : str
        Its title.
    xlabel : str
        The abscissa's label.
    cmap : str
        The colour map.
    unit : str
        The colour bar's label.
    label : str
        The readout's format, given x, z and the value.
    """

    def __init__(  # noqa: PLR0913 - a map's labels
        self,
        layout: HoverLayout,
        column: int,
        *,
        title: str,
        xlabel: str,
        cmap: str,
        unit: str,
        label: str,
    ) -> None:
        """Make the plot, its image, colour bar and marker."""
        self.label = label
        self.x = np.zeros(0)
        self.z = np.zeros(0)
        self.data = np.zeros((0, 0))
        self.plot = layout.add_plot(0, column, title)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", "z", units="m")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.bar = pg.ColorBarItem(
            colorMap=colormap(cmap), label=unit, interactive=True, width=12
        )
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self.marker = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("c", width=2),
        )
        self.plot.addItem(self.marker)
        self.marker.hide()
        layout.add_readout(self.plot, self.lookup)

    def show(
        self,
        x: np.ndarray,
        z: np.ndarray,
        data: np.ndarray,
        levels: tuple[float, float],
    ) -> None:
        """Draw the map.

        Parameters
        ----------
        x : numpy.ndarray
            The abscissa, one value per column.
        z : numpy.ndarray
            The planes' positions, m, one per row.
        data : numpy.ndarray
            The values, one row per plane.
        levels : tuple of float
            The colour bar's range.
        """
        self.x, self.z, self.data = x, z, data
        step = float(np.median(np.diff(z))) if z.size > 1 else 1e-3
        self.image.setImage(data, autoLevels=False)
        self.image.setRect(QRectF(x[0], z[0] - step / 2, x[-1] - x[0], step * len(z)))
        self.bar.setLevels(levels)
        self.plot.getViewBox().autoRange(items=[self.image])

    def clear(self) -> None:
        """Remove the map."""
        self.x, self.z, self.data = np.zeros(0), np.zeros(0), np.zeros((0, 0))
        self.image.clear()
        self.marker.hide()

    def lookup(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Return the map's value under a point, labelled, or None off the map.

        Parameters
        ----------
        x, y : float
            The point.

        Returns
        -------
        tuple or None
            The point and its label.
        """
        if self.z.size == 0 or self.x.size < 2 or not self.x[0] <= x <= self.x[-1]:
            return None
        row = int(np.argmin(np.abs(self.z - y)))
        spacing = (self.z[-1] - self.z[0]) / max(self.z.size - 1, 1)
        if self.z.size > 1 and abs(self.z[row] - y) > spacing:
            return None
        column = round((x - self.x[0]) / (self.x[-1] - self.x[0]) * (self.x.size - 1))
        return (
            x,
            y,
            self.label.format(self.x[column], self.z[row], self.data[row, column]),
        )


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

        self.maps = HoverLayout()
        self.spectrum_map = MapPanel(
            self.maps,
            0,
            title="Spectral energy density (display-binned)",
            xlabel="Wavelength (nm)",
            cmap="viridis",
            unit="dB",
            label="λ {0:.1f} nm\nz {1:.4g} m\n{2:.1f} dB",
        )
        self.power_map = MapPanel(
            self.maps,
            1,
            title="Power (display-binned)",
            xlabel="Time (fs)",
            cmap="inferno",
            unit="GW",
            label="t {0:.2f} fs\nz {1:.4g} m\n{2:.3g} GW",
        )
        self.power_map.plot.getViewBox().setYLink(self.spectrum_map.plot.getViewBox())
        for panel in (self.spectrum_map, self.power_map):
            panel.marker.sigPositionChanged.connect(self._marker_moved)
        self.maps.scene().sigMouseClicked.connect(self._map_clicked)  # pyright: ignore[reportAttributeAccessIssue]

        self.slices = HoverLayout()
        pen = pg.mkPen(BLUE, width=1.5)
        self.spectrum_line = self.slices.add_plot(0, 0, "Spectrum at z")
        self.spectrum_line.setLabels(bottom="Wavelength (nm)", left="dB")
        self.spectrum_curve = self.spectrum_line.plot(pen=pen)
        self.power_line = self.slices.add_plot(0, 1, "Power at z")
        self.power_line.setLabels(bottom="Time (fs)", left="GW")
        self.power_curve = self.power_line.plot(pen=pen)
        self.energy_line = self.slices.add_plot(0, 2, "Energy")
        self.energy_line.setLabel("bottom", "z", units="m")
        self.energy_line.setLabel("left", "µJ")
        self.energy_curve = self.energy_line.plot(pen=pen, symbol="o", symbolSize=4)
        self.energy_marker = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("gray", style=Qt.PenStyle.DashLine)
        )
        self.energy_line.addItem(self.energy_marker)
        self.slices.add_readout(
            self.spectrum_line,
            curve_lookup(self.spectrum_curve, "λ {0:.1f} nm\n{1:.1f} dB"),
            dot=True,
        )
        self.slices.add_readout(
            self.power_line,
            curve_lookup(self.power_curve, "t {0:.2f} fs\n{1:.3g} GW"),
            dot=True,
        )
        self.slices.add_readout(
            self.energy_line,
            curve_lookup(self.energy_curve, "z {0:.4g} m\n{1:.4g} µJ"),
            dot=True,
        )

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("plane-slider")
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_plane)
        self.follow = QCheckBox("Follow the newest plane")
        self.follow.setChecked(True)
        self.position = QLabel("no planes yet")
        self.position.setMinimumWidth(160)
        self.floor = QDoubleSpinBox()
        self.floor.setRange(-200.0, -5.0)
        self.floor.setValue(DB_FLOOR)
        self.floor.setSuffix(" dB")
        self.floor.setToolTip("The bottom of the dB scale.")
        self.floor.valueChanged.connect(self._redraw)

        controls = QHBoxLayout()
        controls.addWidget(self.position)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.follow)
        controls.addWidget(QLabel("Floor"))
        controls.addWidget(self.floor)
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.addLayout(controls)
        middle_layout.addWidget(self.slices, 1)
        split = GripSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.maps)
        split.addWidget(middle)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)

    # ------------------------------------------------------------------ data

    def set_spatial(self, spatial: bool) -> None:
        """Name what the maps show: a beam's energy per bin and axial power, or not.

        Parameters
        ----------
        spatial : bool
            Whether the file is a free-space run's, read through solprop's spatial
            functions (the plane's energy per frequency bin, the axial flux).
        """
        spectrum = "Energy per bin" if spatial else "Spectral energy density"
        power = "Axial power" if spatial else "Power"
        self.spectrum_map.plot.setTitle(f"{spectrum} (display-binned)")
        self.power_map.plot.setTitle(f"{power} (display-binned)")
        self.power_line.setTitle(f"{power} at z")

    def clear(self) -> None:
        """Forget every row and the axes."""
        self.rows = []
        self.wavelength = np.zeros(0)
        self.time = np.zeros(0)
        self.reference = 0.0
        self.spectrum_map.clear()
        self.power_map.clear()
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

    def _z(self) -> np.ndarray:
        """Return the rows' positions.

        Returns
        -------
        numpy.ndarray
            The position of each row, m.
        """
        return np.array([float(r["z"]) for r in self.rows])  # pyright: ignore[reportArgumentType]

    def _redraw(self) -> None:
        """Redraw the maps and the selected plane."""
        if not self.rows or self.wavelength.size == 0:
            return
        z = self._z()
        density = np.array([np.asarray(r["density"]) for r in self.rows])
        power = np.array([np.asarray(r["power"]) for r in self.rows]) / 1e9
        floor = self.floor.value()
        db = np.clip(decibels(density, self.reference), floor, 0.0)
        self.spectrum_map.show(self.wavelength * 1e9, z, db, (floor, 0.0))
        self.power_map.show(
            self.time * 1e15, z, power, (0.0, float(power.max()) or 1.0)
        )
        self.spectrum_line.getViewBox().setYRange(floor, 3)
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
        """Draw the lines of one plane, and move the markers to it.

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
        self.energy_marker.setPos(z)
        for panel in (self.spectrum_map, self.power_map):
            panel.marker.blockSignals(True)
            panel.marker.setPos(z)
            panel.marker.show()
            panel.marker.blockSignals(False)
        self.position.setText(f"plane {int(row['index'])}, z = {z:.4g} m")  # pyright: ignore[reportArgumentType]

    def nearest(self, z: float) -> int:
        """Return the row nearest a position.

        Parameters
        ----------
        z : float
            The position, m.

        Returns
        -------
        int
            The row's place.
        """
        return int(np.argmin(np.abs(self._z() - z)))

    def _marker_moved(self, line: pg.InfiniteLine) -> None:
        """Select the plane nearest a dragged marker.

        Parameters
        ----------
        line : pyqtgraph.InfiniteLine
            The marker.
        """
        if self.rows:
            self.slider.setValue(self.nearest(float(line.value())))  # pyright: ignore[reportArgumentType]

    def _map_clicked(self, event: object) -> None:
        """Select the plane nearest a click on either map.

        Parameters
        ----------
        event : pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
            The click.
        """
        if not self.rows:
            return
        position = event.scenePos()  # pyright: ignore[reportAttributeAccessIssue]
        for panel in (self.spectrum_map, self.power_map):
            box = panel.plot.getViewBox()
            if box.sceneBoundingRect().contains(position):
                self.slider.setValue(self.nearest(box.mapSceneToView(position).y()))
