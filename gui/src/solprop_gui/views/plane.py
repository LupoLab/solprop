# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Plane view: one saved plane's spectrum, power, phase and spectrogram.

A results file's field planes are stepped through with a slider (design 11.2); for the
plane shown the kernel gives the spectral density, the power and the spectral phase
from :mod:`solprop.processing`, and the spectrogram (``get_spectrogram``) drawn here
as an image to zoom and read, with each plane. Compression is the Post-process step's.
A mode sum has a power and a spectrum but no phase or field, so the phase and the
spectrogram take the first mode then, and say so.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.plot_options import UNITS, PlotOptions
from solprop_gui.registry.units import si
from solprop_gui.views.live import set_range
from solprop_gui.views.plotting import (
    PALETTE,
    HoverLayout,
    choose_figure,
    colormap,
    curve_lookup,
)

__all__ = ["PlaneView"]

_C = 299_792_458.0

PHASE_FLOOR = 1e-3
"""The share of the spectrum's peak below which the phase is not drawn: noise there."""


class SpectrogramPlot:
    """A plane's spectrogram as an image: zoom, pan and a readout, in dB or linear.

    Parameters
    ----------
    layout : solprop_gui.views.plotting.HoverLayout
        The layout it is drawn in.
    """

    def __init__(self, layout: HoverLayout) -> None:
        """Make the plot, its image and colour bar."""
        self.plot = layout.add_plot(0, 0, "Spectrogram")
        self.plot.setLabel("bottom", "Time (fs)")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.bar = pg.ColorBarItem(
            colorMap=colormap("viridis"), label="dB", interactive=True, width=12
        )
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self.data: Mapping[str, object] | None = None
        self.shown = np.zeros((0, 0))
        self.unit = 1.0
        layout.add_readout(self.plot, self.lookup)

    def show(self, data: Mapping[str, object], options: PlotOptions) -> None:
        """Draw a spectrogram, or a beam's intensity across its centre, as scaled.

        Parameters
        ----------
        data : Mapping
            The kernel's ``spectrogram`` reply (``t``, ``x``, ``values`` and a
            spectral ``axis``) or its ``beam`` reply (an ``axis`` of ``"x"`` or
            ``"r"``, in m).
        options : PlotOptions
            The scale, floor and ranges.
        """
        self.data = data
        self.redraw(options)

    def redraw(self, options: PlotOptions) -> None:
        """Draw the spectrogram held with new options.

        Parameters
        ----------
        options : PlotOptions
            The scale, floor and ranges.
        """
        if self.data is None:
            return
        data = self.data
        axis = str(data["axis"])
        spectral = axis in UNITS
        self.unit = UNITS[axis][1] if spectral else 1e-3
        t = np.asarray(data["t"], dtype=float) * 1e15
        x = np.asarray(data["x"], dtype=float) / self.unit
        values = np.asarray(data["values"], dtype=float)
        peak = float(values.max()) if values.size and values.max() > 0 else 1.0
        if options.scale == "dB":
            floor = options.floor
            self.shown = np.clip(
                10 * np.log10(np.maximum(values / peak, 1e-30)), floor, 0.0
            )
            levels, unit = (floor, 0.0), "dB"
        else:
            self.shown = values / peak
            levels, unit = (0.0, 1.0), "relative"
        self.image.setImage(self.shown, autoLevels=False)
        self.image.setRect(QRectF(t[0], x[0], float(t[-1] - t[0]), float(x[-1] - x[0])))
        self.bar.setLevels(levels)
        self.bar.getAxis("left").setLabel(unit)
        if spectral:
            name = "Wavelength" if axis == "wavelength" else "Frequency"
            self.plot.setLabel("left", f"{name} ({UNITS[axis][0]})")
            mode = data.get("mode")
            gate = float(data.get("gate_fwhm") or 0.0) * 1e15  # pyright: ignore[reportArgumentType]
            self.plot.setTitle(f"Spectrogram, mode {mode}, gate {gate:.3g} fs")
        else:  # a beam's intensity across its centre, against time
            self.plot.setLabel("left", f"{axis} (mm)")
            self.plot.setTitle("Intensity across the beam's centre")
        self.plot.getViewBox().autoRange(items=[self.image], padding=0)
        set_range(self.plot, "x", options.shown_limits("time"))
        if axis == options.axis:
            set_range(self.plot, "y", options.shown_limits("spectral"))

    def lookup(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Read the map's value under the mouse.

        Parameters
        ----------
        x, y : float
            The mouse, in fs and the axis's unit.

        Returns
        -------
        tuple or None
            The point and its label; ``None`` off the map.
        """
        if self.data is None or self.shown.size == 0:
            return None
        t = np.asarray(self.data["t"], dtype=float) * 1e15
        rows = np.asarray(self.data["x"], dtype=float) / self.unit
        if not (t[0] <= x <= t[-1] and rows[0] <= y <= rows[-1]):
            return None
        column = int(np.argmin(np.abs(t - x)))
        row = int(np.argmin(np.abs(rows - y)))
        return x, y, f"{x:.1f} fs, {y:.4g}\n{self.shown[row, column]:.3g}"

    def clear(self) -> None:
        """Forget the spectrogram."""
        self.data = None
        self.shown = np.zeros((0, 0))
        self.image.clear()
        self.plot.setTitle("Spectrogram")


class PlaneView(QWidget):
    """One plane of a results file.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    plane_wanted : Signal
        Emitted with the plane's index and the mode selection.
    spectrogram_wanted : Signal
        Emitted with the plane's index and the mode selection, with each plane.
    figure_wanted : Signal
        Emitted with a figure's ``file``, ``plot``, ``z`` and ``modes``.
    """

    plane_wanted = Signal(int, object)
    spectrogram_wanted = Signal(int, object)
    figure_wanted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a layout
        """Make the controls and the empty plots."""
        super().__init__(parent)
        self.planes: list[int] = []
        self.z: list[float] = []
        self.current_z: float | None = None
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("plane-slider")
        self.slider.valueChanged.connect(self._moved)
        self.where = QLabel("No results file open.")
        self.where.setObjectName("plane-where")
        self.modes = QComboBox()
        self.modes.setObjectName("plane-modes")
        self.modes.addItem("mode sum", "sum")
        self.modes.currentIndexChanged.connect(self._ask)
        self.options = PlotOptions()
        self.axis = "wavelength"
        self.figure = QComboBox()
        self.figure.setObjectName("plane-figure")
        self.figure.addItem("Export figure…", None)
        for title, kind in (
            ("spectrum at this plane", "spec_1d"),
            ("power at this plane", "time_1d"),
            ("spectrogram at this plane", "spectrogram"),
            ("the maps along z", "prop_2d"),
        ):
            self.figure.addItem(title, kind)
        self.figure.addItem("this plane's curves as CSV", "csv")
        self.figure.activated.connect(self._figure_picked)
        self.status = QLabel("")
        self.status.setObjectName("plane-status")
        self.status.setWordWrap(True)

        self.plots = HoverLayout()
        self.spectrum_plot = self.plots.add_plot(0, 0, "Spectrum")
        self.power_plot = self.plots.add_plot(0, 1, "Power")
        self.phase_plot = self.plots.add_plot(1, 0, "Spectral phase")
        self.power_plot.addLegend(offset=(-10, 10))
        for plot in (self.spectrum_plot, self.power_plot, self.phase_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.spectrum_plot.setLabel("bottom", "Wavelength (nm)")
        self.power_plot.setLabel("bottom", "Time (fs)")
        self.power_plot.setLabel("left", "Power", units="W")
        self.phase_plot.setLabel("bottom", "Wavelength (nm)")
        self.phase_plot.setLabel("left", "Phase (rad)")
        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(PALETTE[0], width=1.4)
        )
        self.power_curve = self.power_plot.plot(
            pen=pg.mkPen(PALETTE[0], width=1.4), name="at the plane"
        )
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen(PALETTE[1], width=1.4))
        self.plots.add_readout(
            self.spectrum_plot, curve_lookup(self.spectrum_curve, "{0:.1f}, {1:.4g}")
        )
        self.plots.add_readout(
            self.power_plot, curve_lookup(self.power_curve, "{0:.1f} fs, {1:.4g} W")
        )
        self.plots.add_readout(
            self.phase_plot, curve_lookup(self.phase_curve, "{0:.1f}, {1:.3g} rad")
        )
        self.image = pg.ImageItem()
        self.image.setColorMap(colormap("viridis"))
        self.phase_plot.addItem(self.image)
        self.image.hide()
        self.spectrogram_area = HoverLayout()
        self.spectrogram = SpectrogramPlot(self.spectrogram_area)
        self._spectrum: tuple[np.ndarray, np.ndarray] | None = None
        self._quantity = ""

        controls = QHBoxLayout()
        self.plane_label = QLabel("Plane")
        controls.addWidget(self.plane_label)
        controls.addWidget(self.slider, 2)
        controls.addWidget(self.where)
        controls.addWidget(self.modes)
        controls.addSpacing(12)
        controls.addWidget(self.figure)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.plots)
        split.addWidget(self.spectrogram_area)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 420])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)
        self.set_planes([], [])

    # ------------------------------------------------------------------ input

    def set_planes(
        self, planes: Sequence[int], z: Sequence[float], labels: Sequence[str] = ()
    ) -> None:
        """Know a file's field planes and modes, and ask for the last plane.

        Parameters
        ----------
        planes : Sequence of int
            The indices of the planes that kept a field.
        z : Sequence of float
            Their positions, m.
        labels : Sequence of str, optional
            The file's modes.
        """
        self.planes = [int(p) for p in planes]
        self.z = [float(v) for v in z]
        self.modes.blockSignals(True)
        self.modes.clear()
        self.modes.addItem("mode sum", "sum")
        for index, name in enumerate(labels):
            self.modes.addItem(name, index)
        self.modes.setVisible(bool(labels))  # a beam has no modes to choose
        self.modes.blockSignals(False)
        enabled = bool(self.planes)
        for widget in (
            self.slider,
            self.modes,
            self.figure,
        ):
            widget.setEnabled(enabled)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self.planes) - 1, 0))
        self.slider.setValue(max(len(self.planes) - 1, 0))
        self.slider.blockSignals(False)
        if enabled:
            self._moved(self.slider.value())
        else:
            self.where.setText("No plane kept a field.")

    def selection(self) -> object:
        """Return the mode selection picked.

        Returns
        -------
        object
            ``"sum"`` or a mode's index.
        """
        return self.modes.currentData()

    def index(self) -> int:
        """Return the file's index of the plane shown.

        Returns
        -------
        int
            The plane.
        """
        return self.planes[self.slider.value()]

    def _moved(self, position: int) -> None:
        """Show where the slider is, and ask for that plane.

        Parameters
        ----------
        position : int
            The slider's position among the field planes.
        """
        if not self.planes:
            return
        self.current_z = self.z[position]
        self.where.setText(f"z = {self.current_z * 100:.4g} cm")
        self._ask()

    def _ask(self, *_args: object) -> None:
        """Ask for the plane shown.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if self.planes:
            self.status.setText("Reading the plane…")
            self.plane_wanted.emit(self.index(), self.selection())
            self.ask_spectrogram()

    def ask_spectrogram(self) -> None:
        """Ask for the shown plane's spectrogram."""
        if self.planes:
            self.spectrogram_wanted.emit(self.index(), self.selection())

    def _figure_picked(self, index: int) -> None:  # pragma: no cover - a modal dialog
        """Ask where to write the figure picked, and ask for it.

        Parameters
        ----------
        index : int
            The item picked.
        """
        kind = self.figure.itemData(index)
        self.figure.setCurrentIndex(0)
        if kind is None:
            return
        if kind == "csv":
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Export the plane", "plane.csv", "CSV (*.csv)"
            )
            if chosen:
                self.export_data(chosen)
            return
        path = choose_figure(self, f"{kind}.pdf")
        if path:
            self.export_figure(path, str(kind))

    def export_data(self, path: str | Path) -> None:
        """Write the plane's curves as CSV, as the kernel gave them (full resolution).

        Parameters
        ----------
        path : str or pathlib.Path
            The file: a column per abscissa and value, the shorter ones padded.
        """
        columns: list[tuple[str, np.ndarray]] = []
        for name, curve, x_title, y_title in (
            ("spectrum", self.spectrum_curve, self._axis_title(), "spectral density"),
            ("power", self.power_curve, "time (fs)", "power (W)"),
            ("phase", self.phase_curve, self._axis_title(), "phase (rad)"),
        ):
            x, y = curve.getData()
            if x is None or y is None:
                continue
            if name == "spectrum" and self._spectrum is not None:
                order = np.argsort(self._spectrum[0])
                x, y = self._spectrum[0][order], self._spectrum[1][order]
            columns.append((f"{name}: {x_title}", np.asarray(x, dtype=float)))
            columns.append((f"{name}: {y_title}", np.asarray(y, dtype=float)))
        length = max((len(values) for _, values in columns), default=0)
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([title for title, _ in columns])
            for row in range(length):
                writer.writerow(
                    [repr(float(v[row])) if row < len(v) else "" for _, v in columns]
                )

    def export_figure(self, path: str, kind: str) -> None:
        """Ask for one of solprop's plots of the plane as a figure.

        Parameters
        ----------
        path : str
            The file.
        kind : str
            The plot: ``"spec_1d"``, ``"time_1d"``, ``"spectrogram"`` or
            ``"prop_2d"``.
        """
        self.figure_wanted.emit(
            {"file": path, "plot": kind, "z": self.current_z, "modes": self.selection()}
        )

    # ------------------------------------------------------------------ showing

    def show_plane(self, plane: Mapping[str, object]) -> None:
        """Draw the kernel's plane.

        Parameters
        ----------
        plane : Mapping
            The ``plane`` reply.
        """
        spectrum: Mapping[str, object] = plane["spectrum"]  # pyright: ignore[reportAssignmentType]
        power: Mapping[str, object] = plane["power"]  # pyright: ignore[reportAssignmentType]
        phase: Mapping[str, object] = plane["phase"]  # pyright: ignore[reportAssignmentType]
        notes = []
        self._quantity = str(spectrum.get("unit") or "")
        self.axis = str(plane.get("axis") or "wavelength")
        unit = UNITS[self.axis][1]
        self.spectrum_plot.setLabel("bottom", self._axis_title(capital=True))
        if "error" in spectrum:
            self._spectrum = None
            self.spectrum_curve.setData([], [])
            notes.append(f"No spectrum: {spectrum['error']}")
        else:
            x = np.asarray(spectrum["x"], dtype=float) / unit
            self._spectrum = (x, np.asarray(spectrum["y"], dtype=float))
            self._redraw_spectrum()
        t = np.asarray(power["t"], dtype=float) * 1e15
        self.power_curve.setData(t, np.asarray(power["y"], dtype=float))
        fluence = plane.get("fluence")
        if isinstance(fluence, Mapping):
            self._show_fluence(fluence, notes)
        elif "error" in phase:
            self.phase_curve.setData([], [])
            notes.append(f"No phase: {phase['error']}")
        else:
            self._phase_axes()
            frequency = np.asarray(phase["omega"], dtype=float) / (2 * np.pi)
            along = _C / frequency if self.axis == "wavelength" else frequency
            along = along / unit
            values = np.asarray(phase["y"], dtype=float)
            x, y = self._spectrum or (np.zeros(0), np.zeros(0))
            if x.size and y.size and np.max(y) > 0:
                # Where there is no light the phase is noise: drawn as gaps.
                order = np.argsort(x)
                light = np.interp(along, x[order], y[order]) / np.max(y)
                values = np.where(light >= PHASE_FLOOR, values, np.nan)
            order = np.argsort(along)
            self.phase_curve.setData(along[order], values[order], connect="finite")
            if self.selection() == "sum" and self.modes.count() > 2:
                notes.append("The phase is the first mode's: a sum of modes has none.")
        values = np.asarray(power["y"], dtype=float)
        peak = float(values.max()) if values.size else 0.0
        notes.insert(0, f"Peak power {si(peak, 'W')}.")
        if self.options.bandpass is not None:
            notes.insert(1, f"{self.options.describe().capitalize()}.")
        self.status.setText(" ".join(notes))
        self._frame()

    def _phase_axes(self) -> None:
        """Show the third plot as the spectral phase."""
        self.image.hide()
        self.phase_plot.setTitle("Spectral phase")
        self.phase_plot.setLabel("bottom", self._axis_title(capital=True))
        self.phase_plot.setLabel("left", "Phase (rad)")

    def _show_fluence(self, fluence: Mapping[str, object], notes: list[str]) -> None:
        """Show a spatial plane's fluence in the third plot, a map or a profile.

        Parameters
        ----------
        fluence : Mapping
            The plane's ``fluence``: ``axes``, ``values`` and ``unit``, or
            ``error``.
        notes : list of str
            The status line's notes, added to.
        """
        self.phase_curve.setData([], [])
        if "error" in fluence:
            self._phase_axes()
            notes.append(f"No fluence: {fluence['error']}")
            return
        axes: Mapping[str, object] = fluence["axes"]  # pyright: ignore[reportAssignmentType]
        values = np.asarray(fluence["values"], dtype=float)
        unit = str(fluence["unit"])
        self.phase_plot.setTitle(f"Fluence ({unit})")
        if values.ndim == 2:
            x = np.asarray(axes["x"], dtype=float) * 1e3
            y = np.asarray(axes["y"], dtype=float) * 1e3
            self.image.setImage(values.T)
            self.image.setRect(
                float(x.min()), float(y.min()), float(np.ptp(x)), float(np.ptp(y))
            )
            self.image.show()
            self.phase_plot.setLabel("bottom", "x (mm)")
            self.phase_plot.setLabel("left", "y (mm)")
            self.phase_plot.getViewBox().autoRange(items=[self.image])
            return
        self.image.hide()
        name = "r" if "r" in axes else "x"
        self.phase_curve.setData(np.asarray(axes[name], dtype=float) * 1e3, values)
        self.phase_plot.setLabel("bottom", f"{name} (mm)")
        self.phase_plot.setLabel("left", f"Fluence ({unit})")

    def _axis_title(self, *, capital: bool = False) -> str:
        """Return the spectral axis's title, with its unit.

        Parameters
        ----------
        capital : bool, optional
            Whether to capitalise it, for a plot's label.

        Returns
        -------
        str
            ``"wavelength (nm)"`` or ``"frequency (THz)"``.
        """
        name = "wavelength" if self.axis == "wavelength" else "frequency"
        if capital:
            name = name.capitalize()
        return f"{name} ({UNITS[self.axis][0]})"

    def apply_options(self, options: PlotOptions) -> None:
        """Draw with new plot options: scale, floor and the typed limits.

        The axis and the bandpass are solprop's to compute: they take effect with
        the next plane asked for.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        self.options = options
        self._redraw_spectrum()
        self._frame()
        self.spectrogram.redraw(options)

    def _frame(self) -> None:
        """Set the plots' ranges to the typed limits, automatic where blank."""
        spectral = self.options.shown_limits("spectral")
        if self.axis != self.options.axis:
            spectral = (None, None)  # limits in another axis's unit
        set_range(self.spectrum_plot, "x", spectral)
        if self.image.isVisible():
            return  # the third plot is a fluence map
        set_range(self.phase_plot, "x", spectral)
        set_range(self.power_plot, "x", self.options.shown_limits("time"))

    def _spectrum_title(self) -> str:
        """Return what the spectrum plot shows.

        Returns
        -------
        str
            The spectral density, or a beam's energy per frequency bin.
        """
        return "Energy per bin" if self._quantity == "J per bin" else "Spectral density"

    def _redraw_spectrum(self, *_args: object) -> None:
        """Draw the spectrum, in dB of its peak or linear.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if self._spectrum is None:
            return
        x, y = self._spectrum
        if self.options.scale == "dB":
            peak = float(np.max(y)) if y.size and np.max(y) > 0 else 1.0
            shown = 10 * np.log10(np.maximum(y / peak, 1e-30))
            self.spectrum_plot.setLabel("left", f"{self._spectrum_title()} (dB)")
            self.spectrum_plot.getViewBox().setYRange(self.options.floor, 3)
        else:
            shown = y
            self.spectrum_plot.setLabel("left", self._spectrum_title())
            self.spectrum_plot.getViewBox().enableAutoRange(axis="y")
        order = np.argsort(x)
        self.spectrum_curve.setData(x[order], shown[order])

    def show_spectrogram(self, data: Mapping[str, object]) -> None:
        """Show a plane's spectrogram.

        Parameters
        ----------
        data : Mapping
            The kernel's ``spectrogram`` reply.
        """
        self.spectrogram.show(data, self.options)

    def show_error(self, message: str) -> None:
        """Say why the plane or its spectrogram is not shown.

        Parameters
        ----------
        message : str
            The reason.
        """
        self.status.setText(message)

    def clear(self) -> None:
        """Forget the file shown."""
        self._spectrum = None
        for curve in (self.spectrum_curve, self.power_curve, self.phase_curve):
            curve.setData([], [])
        self.spectrogram.clear()
        self.set_planes([], [])
