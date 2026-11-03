# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Overview: spectrum and power along the guide, and the plane at a position.

One view serves the live run (rows arrive as the planes are saved) and the results
of a file (all rows at once), laid out as the Qt prototype was: the two maps above,
the selected plane's spectrum, power and the energy below. Its specification (design
12.8) is stated on the view:

- the spectral energy density summed over modes, on the wavelength or frequency
  axis the plot options choose, in dB relative to the peak of the first plane shown
  or linear;
- the power summed over modes, in time, in GW, within the options' bandpass;
- both reduced for display by keeping each bin's largest value, which the maps'
  titles say.

The plot options' typed limits set the axes' ranges; a live run's maps span the
whole length from its start, so the rows filling them show its progress.

Hovering reads a map's value or a curve's point; clicking a map, or dragging its z
marker, selects the plane.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.plot_options import UNITS, PlotOptions
from solprop_gui.views.plotting import BLUE, HoverLayout, colormap, curve_lookup
from solprop_gui.views.splitter import GripSplitter

__all__ = ["DB_FLOOR", "MapPanel", "OverviewView", "decibels", "set_range"]

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


def set_range(
    plot: pg.PlotItem, axis: str, limits: tuple[float | None, float | None]
) -> None:
    """Set one axis of a plot to typed limits, automatic where a limit is blank.

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        The plot.
    axis : {'x', 'y'}
        Which axis.
    limits : tuple
        The low and high limits, in the plot's units; ``None`` for automatic.
    """
    box = plot.getViewBox()
    low, high = limits
    if low is None and high is None:
        box.enableAutoRange(axis=axis)
        return
    current = box.viewRange()[0 if axis == "x" else 1]
    low = current[0] if low is None else low
    high = current[1] if high is None else high
    if axis == "x":
        box.setXRange(low, high, padding=0)
    else:
        box.setYRange(low, high, padding=0)


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
        self.extent: tuple[float, float] | None = None
        self.limits: tuple[float | None, float | None] = (None, None)
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
        self.frame()

    def frame(self) -> None:
        """Set the view: the typed limits across, the run's whole length along z."""
        box = self.plot.getViewBox()
        if self.x.size:
            box.autoRange(items=[self.image], padding=0)
        if self.extent is not None:
            box.setYRange(*self.extent, padding=0)
        if self.limits != (None, None):
            set_range(self.plot, "x", self.limits)

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
        self.options = PlotOptions()
        self.spectra: dict[str, np.ndarray] = {}
        self.band: tuple[float, float] | None = None
        self.time = np.zeros(0)
        self.rows: list[Mapping[str, object]] = []

        self.maps = HoverLayout()
        self.spectrum_map = MapPanel(
            self.maps,
            0,
            title="Spectral energy density (display-binned)",
            xlabel="Wavelength (nm)",
            cmap="viridis",
            unit="dB",
            label="{0:.1f}\nz {1:.4g} m\n{2:.3g}",
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
        self.profile_map = MapPanel(
            self.maps,
            2,
            title="Beam profile",
            xlabel="x (mm)",
            cmap="magma",
            unit="J/m²",
            label="{0:.3g} mm\nz {1:.4g} m\n{2:.3g}",
        )
        for panel in (self.power_map, self.profile_map):
            panel.plot.getViewBox().setYLink(self.spectrum_map.plot.getViewBox())
        for panel in (self.spectrum_map, self.power_map, self.profile_map):
            panel.marker.sigPositionChanged.connect(self._marker_moved)
        self.maps.scene().sigMouseClicked.connect(self._map_clicked)  # pyright: ignore[reportAttributeAccessIssue]

        self.slices = HoverLayout()
        pen = pg.mkPen(BLUE, width=1.5)
        self.spectrum_line = self.slices.add_plot(0, 0, "Spectrum at z")
        self.spectrum_line.setLabels(bottom="Wavelength (nm)", left="dB")
        self.spectrum_note = QLabel("")
        self.spectrum_note.setObjectName("overview-note")
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
        self.profile_line = self.slices.add_plot(0, 3, "Beam at z")
        self.profile_line.setLabels(bottom="x (mm)", left="J/m²")
        self.profile_curve = self.profile_line.plot(pen=pen)
        self.slices.add_readout(
            self.profile_line,
            curve_lookup(self.profile_curve, "{0:.3g} mm\n{1:.3g}"),
            dot=True,
        )
        self.spatial = True
        self.set_spatial(False)
        self.slices.add_readout(
            self.spectrum_line,
            curve_lookup(self.spectrum_curve, "{0:.1f}\n{1:.3g}"),
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

        controls = QHBoxLayout()
        controls.addWidget(self.position)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.follow)
        controls.addWidget(self.spectrum_note)
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
        if spatial == self.spatial:
            return
        self.spatial = spatial
        spectrum = "Energy per bin" if spatial else "Spectral energy density"
        power = "Axial power" if spatial else "Power"
        self.spectrum_map.plot.setTitle(f"{spectrum} (display-binned)")
        self.power_map.plot.setTitle(f"{power} (display-binned)")
        self.power_line.setTitle(f"{power} at z")
        # A beam's profile across its centre, mapped along z, only for a beam.
        for layout, plot, column in (
            (self.maps, self.profile_map.plot, 2),
            (self.slices, self.profile_line, 3),
        ):
            present = plot in layout.ci.items
            if spatial and not present:
                layout.ci.addItem(plot, row=0, col=column)
            elif not spatial and present:
                layout.ci.removeItem(plot)
            plot.setVisible(spatial)

    def _panels(self) -> tuple[MapPanel, ...]:
        """Return the maps shown.

        Returns
        -------
        tuple of MapPanel
            The spectrum and power maps, and a beam's profile map.
        """
        maps = (self.spectrum_map, self.power_map)
        return (*maps, self.profile_map) if self.spatial else maps

    def clear(self) -> None:
        """Forget every row, the axes and the run's extent."""
        self.rows = []
        self.spectra = {}
        self.band = None
        self.time = np.zeros(0)
        for panel in self._panels():
            panel.extent = None
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
            ``wavelength`` (m), ``f`` (Hz) and ``time`` (s), and the ``bandpass``
            the rows' power is computed within.
        """
        self.spectra = {
            axis: np.asarray(axes[axis], dtype=float)
            for axis in ("wavelength", "f")
            if axis in axes
        }
        self.time = np.asarray(axes["time"], dtype=float)
        band = axes.get("bandpass")
        self.band = None if band is None else tuple(band)  # pyright: ignore[reportArgumentType]
        self._note()

    def set_extent(self, z_start: float, z_end: float) -> None:
        """Span the maps over a run's whole length, so its rows show its progress.

        Parameters
        ----------
        z_start, z_end : float
            Where the run starts and ends, m.
        """
        extent = (float(z_start), float(z_end))
        for panel in self._panels():
            panel.extent = extent
            panel.plot.getViewBox().setYRange(*extent, padding=0)
        self.energy_line.getViewBox().setXRange(*extent, padding=0.02)

    def apply_options(self, options: PlotOptions) -> None:
        """Draw with new plot options: axis, scale, floor and the typed limits.

        The rows hold both spectral axes, so the axis changes at once; the bandpass
        is solprop's to compute, and a view whose rows were computed with another
        says so until it is given new rows.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        self.options = options
        self.spectrum_map.plot.setLabel("bottom", options.label())
        self.spectrum_line.setLabel("bottom", options.label())
        unit = "dB" if options.scale == "dB" else "relative"
        self.spectrum_line.setLabel("left", unit)
        self.spectrum_map.bar.getAxis("left").setLabel(unit)  # where its label is
        self.spectrum_map.limits = options.shown_limits("spectral")
        self.power_map.limits = options.shown_limits("time")
        self._redraw()
        set_range(self.spectrum_line, "x", self.spectrum_map.limits)
        set_range(self.power_line, "x", self.power_map.limits)

    def _note(self) -> None:
        """Say where the rows shown differ from the options asked for."""
        described = self.options.describe()
        if self.rows and self.band != self.options.bandpass:
            shown = PlotOptions(bandpass=self.band).describe() or "whole spectrum"
            described = f"power shown: {shown}, until re-read"
        self.spectrum_note.setText(described)

    def reference(self) -> float:
        """Return the 0 dB reference: the first plane's peak on the axis shown.

        Returns
        -------
        float
            The peak density, or 0 with no rows.
        """
        if not self.rows:
            return 0.0
        return float(np.max(self._density(self.rows[0])))

    def _density(self, row: Mapping[str, object]) -> np.ndarray:
        """Return a row's density on the axis shown.

        Parameters
        ----------
        row : Mapping
            The row.

        Returns
        -------
        numpy.ndarray
            Its density.
        """
        density: Mapping[str, object] = row["density"]  # pyright: ignore[reportAssignmentType]
        return np.asarray(density[self.options.axis], dtype=float)

    def _x(self) -> np.ndarray:
        """Return the spectral bins' centres on the axis shown, in its unit.

        Returns
        -------
        numpy.ndarray
            In nm or THz; empty before the axes come.
        """
        axis = self.options.axis
        return self.spectra.get(axis, np.zeros(0)) / UNITS[axis][1]

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
        self._note()
        x = self._x()
        if not self.rows or x.size == 0:
            return
        z = self._z()
        density = np.array([self._density(r) for r in self.rows])
        power = np.array([np.asarray(r["power"]) for r in self.rows]) / 1e9
        shown, levels = self._scaled(density)
        self.spectrum_map.show(x, z, shown, levels)
        self.power_map.show(
            self.time * 1e15, z, power, (0.0, float(power.max()) or 1.0)
        )
        self.set_spatial("profile" in self.rows[0])
        if self.spatial:
            across, profiles = self._profiles()
            self.profile_map.show(
                across, z, profiles, (0.0, float(profiles.max()) or 1.0)
            )
        if self.options.scale == "dB":
            self.spectrum_line.getViewBox().setYRange(levels[0], 3)
        else:
            self.spectrum_line.getViewBox().enableAutoRange(axis="y")
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

    def _profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a beam's profiles across its centre, one row per plane.

        Returns
        -------
        tuple of numpy.ndarray
            The coordinate across the centre, mm, and the fluence there, one row
            per plane; the axes are labelled by the coordinate's name.
        """
        first: Mapping[str, object] = self.rows[0]["profile"]  # pyright: ignore[reportAssignmentType]
        name = str(first["axis"])
        for plot in (self.profile_map.plot, self.profile_line):
            plot.setLabel("bottom", f"{name} (mm)")
        profiles = np.array(
            [np.asarray(r["profile"]["y"], dtype=float) for r in self.rows]  # pyright: ignore[reportIndexIssue]
        )
        return np.asarray(first["x"], dtype=float) * 1e3, profiles

    def _scaled(self, density: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Return densities on the options' scale, and the colour bar's levels.

        Parameters
        ----------
        density : numpy.ndarray
            The densities, one row per plane (or one plane).

        Returns
        -------
        tuple
            The values to draw, and the levels: dB of the reference clipped to the
            floor, or the density relative to the reference.
        """
        reference = self.reference()
        if self.options.scale == "dB":
            floor = self.options.floor
            return np.clip(decibels(density, reference), floor, 0.0), (floor, 0.0)
        reference = reference if reference > 0 else 1.0
        return np.asarray(density, dtype=float) / reference, (0.0, 1.0)

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
        shown, _levels = self._scaled(self._density(row))
        self.spectrum_curve.setData(self._x(), shown)
        self.power_curve.setData(self.time * 1e15, np.asarray(row["power"]) / 1e9)
        if self.spatial:
            profile: Mapping[str, object] = row["profile"]  # pyright: ignore[reportAssignmentType]
            self.profile_curve.setData(
                np.asarray(profile["x"], dtype=float) * 1e3,
                np.asarray(profile["y"], dtype=float),
            )
        self.energy_marker.setPos(z)
        for panel in self._panels():
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
        for panel in self._panels():
            box = panel.plot.getViewBox()
            if box.sceneBoundingRect().contains(position):
                self.slider.setValue(self.nearest(box.mapSceneToView(position).y()))
