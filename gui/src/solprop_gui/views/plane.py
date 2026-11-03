# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Plane view: one saved plane's spectrum, power, phase, spectrogram, compression.

A results file's field planes are stepped through with a slider (design 11.2); for the
plane shown the kernel gives the spectral density, the power and the spectral phase
from :mod:`solprop.processing`, solprop draws the spectrogram, and
``compress_taylor`` finds the best Taylor phase, drawn over the power with the transform
limit. A mode sum has a power and a spectrum but no phase or field, so the phase and the
spectrogram take the first mode then, and say so.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.registry.units import si
from solprop_gui.views.plotting import PALETTE, HoverLayout, choose_figure, curve_lookup

__all__ = ["PlaneView"]

_C = 299_792_458.0

PHASE_FLOOR = 1e-3
"""The share of the spectrum's peak below which the phase is not drawn: noise there."""


class _Image(QLabel):
    """A picture that keeps its aspect and fills its space as that changes.

    Parameters
    ----------
    text : str
        What it says before it has a picture.
    """

    def __init__(self, text: str) -> None:
        """Hold no picture yet."""
        super().__init__(text)
        self.original: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 160)

    def set_image(self, pixmap: QPixmap) -> None:
        """Show a picture, scaled to the space.

        Parameters
        ----------
        pixmap : PySide6.QtGui.QPixmap
            The picture.
        """
        self.original = pixmap
        self._fit()

    def _fit(self) -> None:
        """Scale the picture to the label's size."""
        if self.original is not None and not self.original.isNull():
            self.setPixmap(
                self.original.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt's name
        """Refit the picture to the new size.

        Parameters
        ----------
        event : PySide6.QtGui.QResizeEvent
            The resize.
        """
        super().resizeEvent(event)
        self._fit()

    def forget(self, text: str) -> None:
        """Drop the picture and show a text.

        Parameters
        ----------
        text : str
            The text.
        """
        self.original = None
        self.clear()
        self.setText(text)


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
        Emitted with the plane's z and the mode selection.
    compression_wanted : Signal
        Emitted with the plane's index, the highest Taylor order and the selection.
    figure_wanted : Signal
        Emitted with a figure's ``file``, ``plot``, ``z`` and ``modes``.
    """

    plane_wanted = Signal(int, object)
    spectrogram_wanted = Signal(float, object)
    compression_wanted = Signal(int, int, object)
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
        self.log = QCheckBox("dB")
        self.log.setChecked(True)
        self.log.toggled.connect(self._redraw_spectrum)
        self.spectrogram_button = QPushButton("Spectrogram")
        self.spectrogram_button.setObjectName("plane-spectrogram")
        self.spectrogram_button.clicked.connect(self.ask_spectrogram)
        self.order = QSpinBox()
        self.order.setObjectName("plane-order")
        self.order.setRange(2, 6)
        self.order.setPrefix("to order ")
        self.compress_button = QPushButton("Compress")
        self.compress_button.setObjectName("plane-compress")
        self.compress_button.setToolTip(
            "The Taylor phase that best compresses this plane (compress_taylor), and "
            "the transform limit."
        )
        self.compress_button.clicked.connect(self.ask_compression)
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
            self.spectrum_plot, curve_lookup(self.spectrum_curve, "{0:.1f} nm, {1:.4g}")
        )
        self.plots.add_readout(
            self.power_plot, curve_lookup(self.power_curve, "{0:.1f} fs, {1:.4g} W")
        )
        self.plots.add_readout(
            self.phase_plot, curve_lookup(self.phase_curve, "{0:.1f} nm, {1:.3g} rad")
        )
        self.compressed: list[pg.PlotDataItem] = []
        self.spectrogram = _Image("Spectrogram: press the button.")
        self.spectrogram.setObjectName("plane-spectrogram-image")
        self._spectrum: tuple[np.ndarray, np.ndarray] | None = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plane"))
        controls.addWidget(self.slider, 2)
        controls.addWidget(self.where)
        controls.addWidget(self.modes)
        controls.addWidget(self.log)
        controls.addSpacing(12)
        controls.addWidget(self.spectrogram_button)
        controls.addWidget(self.order)
        controls.addWidget(self.compress_button)
        controls.addWidget(self.figure)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.plots)
        split.addWidget(self.spectrogram)
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
        self.modes.blockSignals(False)
        enabled = bool(self.planes)
        for widget in (
            self.slider,
            self.modes,
            self.spectrogram_button,
            self.compress_button,
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

    def ask_spectrogram(self) -> None:
        """Ask for the plane's spectrogram."""
        if self.current_z is not None:
            self.spectrogram.forget("Drawing the spectrogram…")
            self.spectrogram_wanted.emit(self.current_z, self.selection())

    def ask_compression(self) -> None:
        """Ask for the plane's best compression."""
        if self.planes:
            self.status.setText("Compressing…")
            self.compression_wanted.emit(
                self.index(), self.order.value(), self.selection()
            )

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
            ("spectrum", self.spectrum_curve, "wavelength (nm)", "spectral density"),
            ("power", self.power_curve, "time (fs)", "power (W)"),
            ("phase", self.phase_curve, "wavelength (nm)", "phase (rad)"),
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
        x = np.asarray(spectrum["x"], dtype=float) * 1e9
        self._spectrum = (x, np.asarray(spectrum["y"], dtype=float))
        self._redraw_spectrum()
        t = np.asarray(power["t"], dtype=float) * 1e15
        self.power_curve.setData(t, np.asarray(power["y"], dtype=float))
        self._clear_compression()
        notes = []
        if "error" in phase:
            self.phase_curve.setData([], [])
            notes.append(f"No phase: {phase['error']}")
        else:
            omega = np.asarray(phase["omega"], dtype=float)
            wavelength = 2 * np.pi * _C / omega * 1e9
            values = np.asarray(phase["y"], dtype=float)
            x, y = self._spectrum
            if x.size and y.size and np.max(y) > 0:
                # Where there is no light the phase is noise: drawn as gaps.
                order = np.argsort(x)
                light = np.interp(wavelength, x[order], y[order]) / np.max(y)
                values = np.where(light >= PHASE_FLOOR, values, np.nan)
            order = np.argsort(wavelength)
            self.phase_curve.setData(wavelength[order], values[order], connect="finite")
            if self.selection() == "sum" and self.modes.count() > 2:
                notes.append("The phase is the first mode's: a sum of modes has none.")
        values = np.asarray(power["y"], dtype=float)
        peak = float(values.max()) if values.size else 0.0
        notes.insert(0, f"Peak power {si(peak, 'W')}.")
        self.status.setText(" ".join(notes))

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
        if self.log.isChecked():
            peak = float(np.max(y)) if y.size and np.max(y) > 0 else 1.0
            shown = 10 * np.log10(np.maximum(y / peak, 1e-12))
            self.spectrum_plot.setLabel("left", "Spectral density (dB)")
            self.spectrum_plot.getViewBox().setYRange(-60, 3)
        else:
            shown = y
            self.spectrum_plot.setLabel("left", "Spectral density")
            self.spectrum_plot.getViewBox().enableAutoRange(axis="y")
        order = np.argsort(x)
        self.spectrum_curve.setData(x[order], shown[order])

    def show_spectrogram(self, image: bytes) -> None:
        """Show the spectrogram solprop drew.

        Parameters
        ----------
        image : bytes
            The PNG.
        """
        pixmap = QPixmap()
        pixmap.loadFromData(image, "PNG")
        self.spectrogram.set_image(pixmap)

    def _clear_compression(self) -> None:
        """Remove the compressed and transform-limited curves."""
        for curve in self.compressed:
            self.power_plot.removeItem(curve)
        self.compressed = []

    def show_compression(self, found: Mapping[str, object]) -> None:
        """Draw the compressed pulse and the transform limit over the power.

        Parameters
        ----------
        found : Mapping
            The ``compression`` reply.
        """
        self._clear_compression()
        t = np.asarray(found["t"], dtype=float) * 1e15
        for key, name, colour in (
            ("after", "compressed", PALETTE[2]),
            ("limit", "transform limit", "#888888"),
        ):
            curve = self.power_plot.plot(
                t,
                np.asarray(found[key], dtype=float),
                pen=pg.mkPen(colour, width=1.2, style=Qt.PenStyle.DashLine),
                name=name,
            )
            self.compressed.append(curve)
        before = float(found["fwhm_before"]) * 1e15  # pyright: ignore[reportArgumentType]
        after = float(found["fwhm_after"]) * 1e15  # pyright: ignore[reportArgumentType]
        limit = float(found["fwhm_limit"]) * 1e15  # pyright: ignore[reportArgumentType]
        coefficients: Sequence[float] = found["coefficients"]  # pyright: ignore[reportAssignmentType]
        phase = ", ".join(
            f"φ{order} = {value * 1e15**order:.4g} fs^{order}"
            for order, value in enumerate(coefficients)
            if order >= 2
        )
        self.status.setText(
            f"FWHM {before:.3g} fs → {after:.3g} fs (limit {limit:.3g} fs), "
            f"peak {si(float(found['peak_power_after']), 'W')}"  # pyright: ignore[reportArgumentType]
            f" (limit {si(float(found['peak_power_limit']), 'W')}); {phase}."  # pyright: ignore[reportArgumentType]
        )

    def show_error(self, message: str) -> None:
        """Say why the plane or its compression is not shown.

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
        self._clear_compression()
        self.spectrogram.forget("Spectrogram: press the button.")
        self.set_planes([], [])
