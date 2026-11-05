# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The pages of the Analyse rail (design 11.3).

A page draws one thing large and keeps its controls beside it. It asks the session for
what it shows, only while it is on screen: a hidden page is marked stale, and catches up
when it is next shown, so scrubbing costs only what can be seen.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.display import (
    decibels,
    duration_text,
    even_rows,
    length_unit,
    pulse_window,
    scaled_unit,
    size_text,
    wants_log,
)
from solprop_gui.registry.units import pretty, si
from solprop_gui.service.reduce import C
from solprop_gui.ui.analysis import AnalysisSession
from solprop_gui.ui.postprocess import PostprocessPage
from solprop_gui.ui.theme import clear_layout, named_label
from solprop_gui.views.controls import LabelledSlider, Segments, control_panel
from solprop_gui.views.page_plots import ORANGE, LinePlot, MapPlot

__all__ = [
    "PAGE_VIEWS",
    "AlongPage",
    "CompressPage",
    "HealthPage",
    "ModesPage",
    "OverviewPage",
    "PageView",
    "PlanePage",
    "SpectrogramPage",
]

_AXES = (("wavelength", "Wavelength"), ("f", "Frequency"))
_AXIS_LABEL = {"wavelength": "Wavelength (nm)", "f": "Frequency (THz)"}
_AXIS_UNIT = {"wavelength": "nm", "f": "THz"}
_AXIS_SCALE = {"wavelength": 1e9, "f": 1e-12}
_FS = 1e15
_LIVE_MS = 250
"""The live maps redraw at most this often, whatever the run's speed."""

_PHASE_WITHIN_DB = 30.0
"""A spectral phase is drawn only where the spectrum is within this of its peak."""


def _plotted(title: str, plot: QWidget) -> QVBoxLayout:
    """Return a plot under its plain-language title.

    Parameters
    ----------
    title : str
        The title.
    plot : PySide6.QtWidgets.QWidget
        The plot.

    Returns
    -------
    PySide6.QtWidgets.QVBoxLayout
        The two together.
    """
    box = QVBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(2)
    box.addWidget(named_label(title, "plotTitle"))
    box.addWidget(plot, 1)
    return box


class PageView(QWidget):
    """What every page shares.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.

    Attributes
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    controls : PySide6.QtWidgets.QWidget or None
        What sits beside the plots.
    note : PySide6.QtWidgets.QLabel
        A line above the plots, for what they show or why they do not.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Mark the page stale until it is first shown."""
        super().__init__()
        self.setObjectName("page")
        self.session = session
        self.controls: QWidget | None = None
        self.note = named_label("", "pageNote")
        self._stale = True
        session.described.connect(lambda _value: self.stale())
        session.plane_moved.connect(self.plane_moved)
        session.options_changed.connect(self.stale)
        session.mode_changed.connect(self.stale)

    def stale(self) -> None:
        """Bring the page up to date now if it is on screen, or else when it is next."""
        self._stale = True
        if self.isVisible():
            self._stale = False
            self.refresh()

    def plane_moved(self) -> None:
        """Follow the scrubber."""
        self.stale()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt's name
        """Catch up as the page comes on screen.

        Parameters
        ----------
        event : PySide6.QtGui.QShowEvent
            The event.
        """
        super().showEvent(event)
        if self._stale:
            self._stale = False
            self.refresh()

    def refresh(self) -> None:
        """Ask for, or draw, what the page shows."""

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return what the page shows as columns, for a CSV file.

        Returns
        -------
        tuple or None
            The columns' names and their values, a column to each; ``None`` for a page
            with nothing to export that way.
        """
        return None

    def figure(self) -> dict[str, object] | None:
        """Return what asks solprop to draw the page as a publication figure.

        Returns
        -------
        dict or None
            The kernel's ``figure`` keys; ``None`` for a page with no such figure.
        """
        return None

    def python(self) -> str | None:
        """Return the page's name in :func:`~solprop_gui.model.snippets.snippet`.

        Returns
        -------
        str or None
            The key; ``None`` for a page with no snippet.
        """
        return None


def _shown_range(
    value: Mapping[str, Any], axis: str, x: np.ndarray
) -> tuple[float, float]:
    """Return the range of a plane's spectral axis to show: the run's own window.

    Parameters
    ----------
    value : Mapping
        The kernel's ``plane`` reply.
    axis : str
        The spectral axis.
    x : numpy.ndarray
        The spectrum's positions on it, ascending.

    Returns
    -------
    tuple of float
        The range, in the axis's unit.
    """
    window = value.get("window")
    if window is None:
        return float(x[0]), float(x[-1])
    return float(window[0]) * _AXIS_SCALE[axis], float(window[1]) * _AXIS_SCALE[axis]


def _mode_box(session: AnalysisSession) -> QComboBox:
    """Return a combo box of the file's modes, kept in step with the session.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.

    Returns
    -------
    PySide6.QtWidgets.QComboBox
        The box.
    """
    box = QComboBox()
    box.setObjectName("mode-choice")

    def fill(_value: object = None) -> None:
        """List the file's modes.

        Parameters
        ----------
        _value : object, optional
            Not read.
        """
        box.blockSignals(True)
        box.clear()
        box.addItem("Every mode, summed", "sum")
        for index, label in enumerate(session.labels):
            box.addItem(label, index)
        box.setCurrentIndex(max(box.findData(session.mode), 0))
        box.setVisible(len(session.labels) > 1)
        box.blockSignals(False)

    session.described.connect(fill)
    box.activated.connect(lambda place: session.set_mode(box.itemData(place)))
    fill()
    return box


class OverviewPage(PageView):
    """The spectrum and pulse along z, and at the plane selected.

    One page serves a run as it goes and a file read back, so what you watch grow is
    what you then analyse. It takes the calls
    :class:`~solprop_gui.ui.run_panel.RunPanel` makes of a view.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.

    Attributes
    ----------
    rows_grew : Signal
        Emitted with the planes' positions as a run saves another.
    rows : list of Mapping
        The planes' reduced data.
    band : tuple or None
        The band the rows' power is computed within.
    """

    rows_grew = Signal(object)

    def __init__(self, session: AnalysisSession) -> None:
        """Make the maps and the plane's lines."""
        super().__init__(session)
        self.rows: list[Mapping[str, Any]] = []
        self.band: tuple[float, float] | None = None
        self.spectra: dict[str, np.ndarray] = {}
        self.time = np.zeros(0)
        self.extent: tuple[float, float] | None = None
        self.shown = 0
        self.half_range = 0.0
        self.following = True
        self._live = False
        self._db = np.zeros((0, 0))
        self._power = np.zeros((0, 0))
        self._map_rows = np.zeros(0, dtype=int)
        self._unit = ("m", 1.0)

        self.spectrum_map = MapPlot(
            xlabel=_AXIS_LABEL["wavelength"],
            ylabel="z",
            colours="viridis",
            unit="dB",
        )
        self.spectrum_map.set_along("y")
        self.power_map = MapPlot(
            xlabel="Time (fs)", ylabel="z", colours="inferno", unit="GW"
        )
        self.power_map.set_along("y")
        self.spectrum_line = LinePlot(
            xlabel=_AXIS_LABEL["wavelength"], ylabel="dB", unit="dB"
        )
        self.power_line = LinePlot(xlabel="Time (fs)", ylabel="Power (GW)", unit="GW")
        for plot in (self.spectrum_map, self.power_map):
            plot.plane_at.connect(self._picked)
        self.spectrum_map.reader = self._read_spectrum
        self.power_map.reader = self._read_power
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addLayout(_plotted("Spectrum along z", self.spectrum_map), 0, 0)
        grid.addLayout(_plotted("Pulse along z", self.power_map), 0, 1)
        grid.addLayout(_plotted("Spectrum at this plane", self.spectrum_line), 1, 0)
        grid.addLayout(_plotted("Pulse at this plane", self.power_line), 1, 1)

        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.floor = LabelledSlider(
            "Floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(self._floor_moved)
        self.range = LabelledSlider("Time range", 5, 500, 40, " fs")
        self.range.moved.connect(self._range_moved)
        self.controls = control_panel(self.axis_choice, self.floor, self.range)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_LIVE_MS)
        self._timer.timeout.connect(self._draw_maps)

    # -------------------------------------------------- what a run's panel calls

    def clear(self) -> None:
        """Start empty, for a new run."""
        self.rows = []
        self.extent = None
        self.shown = 0
        self.half_range = 0.0
        self.following = True
        self._live = True
        self._db = np.zeros((0, 0))
        for plot in (self.spectrum_map, self.power_map):
            plot.say("Waiting for the first plane…")
        for line in (self.spectrum_line, self.power_line):
            line.set_data(np.zeros(0), [np.zeros(0)])

    def set_axes(self, axes: Mapping[str, object]) -> None:
        """Take the display bins' centres.

        Parameters
        ----------
        axes : Mapping
            ``wavelength`` (m), ``f`` (Hz) and ``time`` (s), and the ``bandpass`` the
            rows' power is computed within.
        """
        self.spectra = {
            axis: np.asarray(axes[axis], dtype=float)
            for axis in ("wavelength", "f")
            if axis in axes
        }
        self.time = np.asarray(axes["time"], dtype=float)
        band = axes.get("bandpass")
        self.band = None if band is None else tuple(band)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

    def set_extent(self, z_start: float, z_end: float) -> None:
        """Take how far the run will go, for the maps' axes.

        Parameters
        ----------
        z_start : float
            Where it starts, m.
        z_end : float
            Where it ends, m.
        """
        self.extent = (z_start, z_end)

    def add_row(self, row: Mapping[str, Any]) -> None:
        """Take a plane of a run as it is saved.

        Parameters
        ----------
        row : Mapping
            ``index``, ``z``, ``density`` by axis, ``power`` and ``stats``.
        """
        self.rows.append(row)
        self.rows_grew.emit([float(r["z"]) for r in self.rows])
        if self.following:
            self.shown = len(self.rows) - 1
        if not self._timer.isActive():
            self._timer.start()

    def set_rows(
        self, axes: Mapping[str, object], rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Take every plane at once.

        Parameters
        ----------
        axes : Mapping
            As :meth:`set_axes` takes.
        rows : Sequence of Mapping
            As :meth:`add_row` takes.
        """
        if axes:
            self.set_axes(axes)
        self.rows = list(rows)
        self._live = False
        self._draw_maps()

    def apply_options(self, _options: object) -> None:
        """Take new options, which the session already holds.

        Parameters
        ----------
        _options : object
            The options.
        """
        self._draw_maps()

    def end_of_run(self) -> None:
        """Stop following: what is on screen is the whole file."""
        self._live = False

    # ------------------------------------------------------------------ the session

    def refresh(self) -> None:
        """Read the file back, unless a run has just drawn all of it."""
        session = self.session
        if session.path is None or self._live:
            self._draw_maps()
            return
        if len(self.rows) == len(session.planes) > 0:
            self._draw_maps()
            return
        self.spectrum_map.say("Reading…")
        self.power_map.say("Reading…")
        session.ask(
            "overview",
            {"path": session.path},
            self._arrived,
            key="page-overview",
            failed=self._refused,
        )

    def _arrived(self, value: Mapping[str, Any]) -> None:
        """Draw a file read back.

        Parameters
        ----------
        value : Mapping
            The kernel's ``overview`` reply.
        """
        self.extent = None
        self.set_rows(value.get("axes", {}), value.get("rows", []))
        self.half_range = 0.0
        self._draw_maps()

    def _refused(self, reason: str) -> None:
        """Say why there is nothing to draw.

        Parameters
        ----------
        reason : str
            The reason solprop gave.
        """
        self.spectrum_map.say(reason)
        self.power_map.say("")

    def plane_moved(self) -> None:
        """Follow the scrubber, at no cost in the maps."""
        self.shown = self.session.place
        self._draw_plane()

    def show_place(self, place: int) -> None:
        """Select a plane of a run going, whose planes the session does not hold yet.

        Parameters
        ----------
        place : int
            Which row.
        """
        self.shown = place
        self.following = place >= len(self.rows) - 1
        self._draw_plane()

    def _picked(self, value: float) -> None:
        """Take a plane picked on a map.

        Parameters
        ----------
        value : float
            The position, in the map's unit.
        """
        z = value / self._unit[1]
        if self.session.positions and not self._live:
            self.session.move_near(z)
        elif self.rows:
            positions = np.array([float(r["z"]) for r in self.rows])
            self.show_place(int(np.argmin(np.abs(positions - z))))

    # ------------------------------------------------------------------ drawing

    def _time_window(self) -> float:
        """Return the half-width of the time range, in fs, set from the light at first.

        Returns
        -------
        float
            The half-width.
        """
        if self.half_range <= 0 and self.time.size and self.rows:
            found = pulse_window(self.time * _FS, self._power)
            whole = float(np.abs(self.time).max()) * _FS
            self.half_range = max(min(found, whole), 1.0)
            self.range.set_range(1, max(round(whole), 2))
            self.range.set_value(round(self.half_range))
        return self.half_range or 1.0

    def _draw_maps(self) -> None:
        """Draw both maps from the rows."""
        options = self.session.options
        axis = options.axis
        if not self.rows or axis not in self.spectra:
            return
        x = self.spectra[axis] * _AXIS_SCALE[axis]
        density = np.array([np.asarray(r["density"][axis], float) for r in self.rows])
        self._db = decibels(density)
        self._power = (
            np.array([np.asarray(r["power"], float) for r in self.rows]) * 1e-9
        )
        z = np.array([float(r["z"]) for r in self.rows])
        self._map_rows, z_first, step = even_rows(z)
        reach = self.extent[1] if self.extent else float(z[-1])
        name, factor = length_unit(reach or 1.0)
        self._unit = (name, factor)
        window = self._time_window()
        t = self.time * _FS
        bottom = (z_first - step / 2) * factor
        height = step * self._map_rows.size * factor
        z_view = (float(z[0]) * factor, max(reach * factor, float(z[-1]) * factor))
        label = _AXIS_LABEL[axis]
        for plot in (self.spectrum_map, self.spectrum_line):
            plot.set_xlabel(label)
        for plot in (self.spectrum_map, self.power_map):
            plot.set_ylabel(f"z ({name})")
        self.spectrum_map.show_map(
            self._db[self._map_rows],
            (float(x[0]), bottom, float(x[-1] - x[0]), height),
            ((float(x[0]), float(x[-1])), z_view),
            (options.floor, 0.0),
        )
        self.power_map.show_map(
            self._power[self._map_rows],
            (float(t[0]), bottom, float(t[-1] - t[0]), height),
            ((-window, window), z_view),
            (0.0, float(self._power.max()) or 1.0),
        )
        self.spectrum_line.set_view((float(x[0]), float(x[-1])), (options.floor, 3.0))
        self.power_line.set_view(
            (-window, window), (0.0, float(self._power.max()) * 1.05 or 1.0)
        )
        self._draw_plane()

    def _draw_plane(self) -> None:
        """Draw the plane selected, and mark it on the maps."""
        if self._db.size == 0 or not self.rows:
            return
        at = min(self.shown, self._db.shape[0] - 1)
        z = float(self.rows[at]["z"]) * self._unit[1]
        axis = self.session.options.axis
        self.spectrum_map.set_mark(z)
        self.power_map.set_mark(z)
        x = self.spectra[axis] * _AXIS_SCALE[axis]
        self.spectrum_line.set_data(x, [self._db[at]])
        self.power_line.set_data(self.time * _FS, [self._power[at]])

    def _floor_moved(self, value: int) -> None:
        """Move the dB floor, at no cost in the data.

        Parameters
        ----------
        value : int
            The floor, dB.
        """
        self.session.options = replace(self.session.options, floor=float(value))
        self.spectrum_map.set_levels((float(value), 0.0))
        x_range = self.spectrum_line.box.viewRange()[0]
        self.spectrum_line.set_view((x_range[0], x_range[1]), (float(value), 3.0))

    def _range_moved(self, value: int) -> None:
        """Move the time range, at no cost in the data.

        Parameters
        ----------
        value : int
            The half-width, fs.
        """
        self.half_range = float(value)
        self.power_map.box.setXRange(-value, value, padding=0)
        self.power_line.box.setXRange(-value, value, padding=0)

    def _read_spectrum(self, x: float, y: float) -> str:
        """Say what is at a point of the spectrum's map.

        Parameters
        ----------
        x : float
            The position on the spectral axis.
        y : float
            The position along z, in the map's unit.

        Returns
        -------
        str
            The readout.
        """
        if self._db.size == 0:
            return ""
        axis = self.session.options.axis
        bins = self.spectra[axis] * _AXIS_SCALE[axis]
        row = self._nearest_row(y)
        at = int(np.clip(np.searchsorted(bins, x), 0, bins.size - 1))
        text = f"{x:.4g} {_AXIS_UNIT[axis]}, {y:.3g} {self._unit[0]}"
        return f"{text}: {self._db[row, at]:.1f} dB"

    def _read_power(self, x: float, y: float) -> str:
        """Say what is at a point of the pulse's map.

        Parameters
        ----------
        x : float
            The time, fs.
        y : float
            The position along z, in the map's unit.

        Returns
        -------
        str
            The readout.
        """
        if self._power.size == 0:
            return ""
        t = self.time * _FS
        row = self._nearest_row(y)
        at = int(np.clip(np.searchsorted(t, x), 0, t.size - 1))
        text = f"{x:.3g} fs, {y:.3g} {self._unit[0]}"
        return f"{text}: {self._power[row, at]:.3g} GW"

    def _nearest_row(self, y: float) -> int:
        """Return the row nearest a position on a map.

        Parameters
        ----------
        y : float
            The position, in the map's unit.

        Returns
        -------
        int
            The row.
        """
        z = np.array([float(r["z"]) for r in self.rows]) * self._unit[1]
        return int(np.argmin(np.abs(z - y)))

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the plane selected, as columns.

        Returns
        -------
        tuple or None
            Time and power in the columns' units; ``None`` with no planes.
        """
        if self._power.size == 0:
            return None
        at = min(self.shown, self._power.shape[0] - 1)
        return ["time (fs)", "power (GW)"], np.column_stack(
            (self.time * _FS, self._power[at])
        )

    def figure(self) -> dict[str, object] | None:
        """Return the page as solprop draws it.

        Returns
        -------
        dict
            Its maps, one figure each.
        """
        return {"plot": "prop_2d"}

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "overview"


class PlanePage(PageView):
    """One plane: its spectrum with a band and the energy in it, its pulse and phase.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the plots and the band's controls."""
        super().__init__(session)
        self.spectrum = LinePlot(
            xlabel=_AXIS_LABEL["wavelength"], ylabel="dB", unit="dB"
        )
        self.power = LinePlot(xlabel="Time (fs)", ylabel="Power (GW)", unit="GW")
        self.phase = LinePlot(
            xlabel=_AXIS_LABEL["wavelength"], ylabel="Phase (rad)", unit="rad"
        )
        self.region = pg.LinearRegionItem(
            brush=(11, 110, 153, 40),
            pen=pg.mkPen("#0b6e99"),
            movable=True,
        )
        self.region.hide()
        self.spectrum.addItem(self.region, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        self._last: Mapping[str, Any] | None = None
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.addLayout(_plotted("Spectrum", self.spectrum), 0, 0, 1, 2)
        grid.addLayout(_plotted("Pulse", self.power), 1, 0)
        grid.addLayout(_plotted("Spectral phase", self.phase), 1, 1)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.note)
        column.addLayout(grid, 1)

        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.floor = LabelledSlider(
            "Floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(lambda v: session.set_options(floor=float(v)))
        self.band_low = QLineEdit()
        self.band_low.setObjectName("band-low")
        self.band_low.setPlaceholderText("from")
        self.band_high = QLineEdit()
        self.band_high.setObjectName("band-high")
        self.band_high.setPlaceholderText("to")
        for box in (self.band_low, self.band_high):
            box.editingFinished.connect(self._band_typed)
        band_clear = QPushButton("Whole spectrum")
        band_clear.setObjectName("plain")
        band_clear.clicked.connect(lambda: session.set_options(bandpass=None))
        band_box = QWidget()
        band_box.setObjectName("page")
        band_grid = QGridLayout(band_box)
        band_grid.setContentsMargins(0, 0, 0, 0)
        band_grid.addWidget(named_label("Band (nm)", "control"), 0, 0, 1, 2)
        band_grid.addWidget(self.band_low, 1, 0)
        band_grid.addWidget(self.band_high, 1, 1)
        band_grid.addWidget(band_clear, 2, 0, 1, 2)
        self.modes = _mode_box(session)
        self.controls = control_panel(
            self.axis_choice, self.floor, band_box, self.modes
        )

    def refresh(self) -> None:
        """Ask for the plane selected."""
        session = self.session
        if session.path is None or not session.planes:
            return
        session.ask(
            "plane",
            {
                "path": session.path,
                "index": session.index,
                "modes": session.mode,
                "axis": session.options.axis,
                "bandpass": session.options.bandpass,
            },
            self._show,
            key="page-plane",
            failed=self.note.setText,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw a plane.

        Parameters
        ----------
        value : Mapping
            The kernel's ``plane`` reply.
        """
        self._last = value
        options = self.session.options
        axis = str(value["axis"])
        spectrum = value["spectrum"]
        x = np.asarray(spectrum["x"], float) * _AXIS_SCALE[axis]
        order = np.argsort(x)
        x = x[order]
        db = decibels(np.asarray(spectrum["y"], float))[order]
        self.spectrum.set_data(x, [db])
        self.spectrum.set_xlabel(_AXIS_LABEL[axis])
        view = _shown_range(value, axis, x)
        self.spectrum.set_view(view, (options.floor, 3.0))
        self._show_band(axis)
        self._show_pulse(value)
        self._show_phase(value, axis, x, db)
        self._say(value)

    def _show_band(self, axis: str) -> None:
        """Mark the band on the spectrum, and in its boxes.

        Parameters
        ----------
        axis : str
            The spectral axis.
        """
        band = self.session.options.bandpass
        self.region.blockSignals(True)
        self.band_low.setText("" if band is None else f"{band[0] * 1e9:.4g}")
        self.band_high.setText("" if band is None else f"{band[1] * 1e9:.4g}")
        if band is None:
            view = self.spectrum.box.viewRange()[0]
            span = view[1] - view[0]
            self.region.setRegion((view[0] + 0.4 * span, view[0] + 0.6 * span))
            self.region.setOpacity(0.35)
        elif axis == "wavelength":
            self.region.setRegion((band[0] * 1e9, band[1] * 1e9))  # pyright: ignore[reportIndexIssue]
            self.region.setOpacity(1.0)
        else:
            self.region.setRegion((C / band[1] * 1e-12, C / band[0] * 1e-12))  # pyright: ignore[reportIndexIssue]
            self.region.setOpacity(1.0)
        self.region.show()
        self.region.blockSignals(False)

    def _show_pulse(self, value: Mapping[str, Any]) -> None:
        """Draw the pulse, and within the band if there is one.

        Parameters
        ----------
        value : Mapping
            The kernel's ``plane`` reply.
        """
        t = np.asarray(value["power"]["t"], float) * _FS
        power = np.asarray(value["power"]["y"], float) * 1e-9
        full = value.get("full_power")
        if full is None:
            self.power.set_names([""])
            self.power.set_data(t, [power])
            trace = power
        else:
            whole = np.asarray(full, float) * 1e-9
            self.power.set_names(["whole spectrum", "in the band"])
            self.power.set_data(t, [whole, power])
            trace = whole
        window = pulse_window(t, trace)
        self.power.set_view((-window, window), (0.0, float(trace.max()) * 1.05 or 1.0))

    def _show_phase(
        self, value: Mapping[str, Any], axis: str, x: np.ndarray, db: np.ndarray
    ) -> None:
        """Draw the spectral phase where the spectrum is strong enough to have one.

        Parameters
        ----------
        value : Mapping
            The kernel's ``plane`` reply.
        axis : str
            The spectral axis.
        x : numpy.ndarray
            The spectrum's positions, ascending.
        db : numpy.ndarray
            The spectrum, dB.
        """
        phase = value["phase"]
        self.phase.set_xlabel(_AXIS_LABEL[axis])
        if "omega" not in phase:
            self.phase.set_data(np.zeros(0), [np.zeros(0)])
            return
        omega = np.asarray(phase["omega"], float)
        f = omega / (2.0 * np.pi)
        on_axis = C / f * 1e9 if axis == "wavelength" else f * 1e-12
        order = np.argsort(on_axis)
        on_axis = on_axis[order]
        y = np.asarray(phase["y"], float)[order]
        keep = np.interp(on_axis, x, db) > -_PHASE_WITHIN_DB
        y = np.where(keep, y, np.nan)
        self.phase.set_data(on_axis, [y])
        self.phase.set_view(
            _shown_range(value, axis, x),
            (float(np.nanmin(y)), float(np.nanmax(y))) if keep.any() else (-1.0, 1.0),
        )

    def _say(self, value: Mapping[str, Any]) -> None:
        """Say the numbers of the plane.

        Parameters
        ----------
        value : Mapping
            The kernel's ``plane`` reply.
        """
        total = float(value["energy"])
        parts = [f"Energy {si(total, 'J', digits=3)}"]
        band_energy = value.get("band_energy")
        if band_energy is not None:
            share = 100.0 * float(band_energy) / total if total else 0.0
            parts.append(
                f"in the band {si(float(band_energy), 'J', 3)} ({share:.2g} %)"
            )
        width = float(value["fwhm"])
        if np.isfinite(width):
            parts.append(f"duration {width * _FS:.3g} fs")
        if "error" in value["phase"]:
            parts.append("phase of the first mode only")
        self.note.setText(" · ".join(parts))

    def _region_moved(self) -> None:
        """Take a band dragged on the spectrum."""
        low, high = self.region.getRegion()  # pyright: ignore[reportGeneralTypeIssues]
        axis = self.session.options.axis
        if axis == "wavelength":
            band = (float(low) * 1e-9, float(high) * 1e-9)  # pyright: ignore[reportArgumentType]
        else:
            band = (C / (float(high) * 1e12), C / (float(low) * 1e12))  # pyright: ignore[reportArgumentType]
        self.session.set_options(bandpass=band)

    def _band_typed(self) -> None:
        """Take a band typed in its boxes."""
        try:
            low, high = float(self.band_low.text()), float(self.band_high.text())
        except ValueError:
            return
        if 0 < low < high:
            self.session.set_options(bandpass=(low * 1e-9, high * 1e-9))

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the plane's spectrum, as columns.

        Returns
        -------
        tuple or None
            Its axis and dB, in the columns' units.
        """
        if self.spectrum.xs.size == 0:
            return None
        axis = self.session.options.axis
        return [_AXIS_LABEL[axis], "spectral density (dB)"], np.column_stack(
            (self.spectrum.xs, self.spectrum.ys[0])
        )

    def figure(self) -> dict[str, object] | None:
        """Return the page as solprop draws it.

        Returns
        -------
        dict
            The spectrum at the plane.
        """
        return {"plot": "spec_1d", "z": self.session.z, "modes": self.session.mode}

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "plane"


class SpectrogramPage(PageView):
    """A plane's spectrogram.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the map and its gate."""
        super().__init__(session)
        self.map = MapPlot(
            xlabel="Time (fs)",
            ylabel=_AXIS_LABEL["wavelength"],
            colours="viridis",
            unit="dB",
        )
        self._values = np.zeros((0, 0))
        self._t = np.zeros(0)
        self._x = np.zeros(0)
        self.map.reader = self._read
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.note)
        column.addWidget(self.map, 1)
        self.gate = LabelledSlider("Gate", 1, 200, 20, " fs")
        self.gate.moved.connect(
            lambda v: session.set_options(gate_fwhm=float(v) * 1e-15)
        )
        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.floor = LabelledSlider(
            "Floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(lambda v: self.map.set_levels((float(v), 0.0)))
        self.floor.slider.sliderReleased.connect(
            lambda: session.set_options(floor=float(self.floor.slider.value()))
        )
        self.modes = _mode_box(session)
        self.controls = control_panel(
            self.axis_choice, self.gate, self.floor, self.modes
        )

    def refresh(self) -> None:
        """Ask for the plane selected's spectrogram."""
        session = self.session
        if session.path is None or not session.planes:
            return
        options = session.options
        session.ask(
            "spectrogram",
            {
                "path": session.path,
                "index": session.index,
                "modes": session.mode,
                "axis": options.axis,
                "gate_fwhm": options.gate_fwhm,
                "positions": options.positions,
            },
            self._show,
            key="page-spectrogram",
            failed=self._refused,
        )

    def _refused(self, reason: str) -> None:
        """Say why there is no spectrogram.

        Parameters
        ----------
        reason : str
            The reason solprop gave.
        """
        self.map.say(reason)

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw a spectrogram.

        Parameters
        ----------
        value : Mapping
            The kernel's ``spectrogram`` reply.
        """
        options = self.session.options
        axis = str(value["axis"])
        t = np.asarray(value["t"], float) * _FS
        x = np.asarray(value["x"], float) * _AXIS_SCALE[axis]
        values = np.asarray(value["values"], float)
        self._values = decibels(values)
        self._t, self._x = t, x
        self.map.set_ylabel(_AXIS_LABEL[axis])
        self.map.show_map(
            self._values,
            (float(t[0]), float(x[0]), float(t[-1] - t[0]), float(x[-1] - x[0])),
            ((float(t[0]), float(t[-1])), (float(x[0]), float(x[-1]))),
            (options.floor, 0.0),
        )
        gate = float(value["gate_fwhm"])
        self.gate.set_value(round(gate * _FS))
        labels = self.session.labels
        mode = int(value["mode"])
        name = labels[mode] if mode < len(labels) else f"mode {mode}"
        self.note.setText(f"{name}, gate {gate * _FS:.3g} fs")

    def _read(self, x: float, y: float) -> str:
        """Say what is at a point of the map.

        Parameters
        ----------
        x : float
            The time, fs.
        y : float
            The position on the spectral axis.

        Returns
        -------
        str
            The readout.
        """
        if self._values.size == 0:
            return ""
        column = int(np.clip(np.searchsorted(self._t, x), 0, self._t.size - 1))
        row = int(np.clip(np.searchsorted(self._x, y), 0, self._x.size - 1))
        unit = _AXIS_UNIT[self.session.options.axis]
        return f"{x:.3g} fs, {y:.4g} {unit}: {self._values[row, column]:.1f} dB"

    def figure(self) -> dict[str, object] | None:
        """Return the page as solprop draws it.

        Returns
        -------
        dict
            The spectrogram at the plane.
        """
        return {"plot": "spectrogram", "z": self.session.z, "modes": self.session.mode}

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "spectrogram"


class AlongPage(PageView):
    """What was recorded along z, up to four quantities at once.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.

    Attributes
    ----------
    wanted : tuple of str
        What the class shows first, where it was recorded.
    """

    _SHOWN = 4

    def __init__(self, session: AnalysisSession) -> None:
        """Make the empty grid and the chooser."""
        super().__init__(session)
        self.wanted: tuple[str, ...] = ()
        self._stats: Mapping[str, Mapping[str, Any]] = {}
        self._z = np.zeros(0)
        self._plots: list[LinePlot] = []
        self._log = "auto"
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(10)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.note)
        column.addLayout(self.grid, 1)
        self.choice = QListWidget()
        self.choice.setObjectName("statistics-choice")
        self.choice.setFrameShape(QFrame.Shape.NoFrame)
        self.choice.itemChanged.connect(self._chosen)
        self.log = Segments(
            "Log axes",
            (("auto", "Auto"), ("on", "On"), ("off", "Off")),
            self._log,
        )
        self.log.chosen.connect(self._log_chosen)
        box = QWidget()
        box.setObjectName("page")
        boxed = QVBoxLayout(box)
        boxed.setContentsMargins(0, 0, 0, 0)
        boxed.addWidget(named_label("Quantities", "control"))
        boxed.addWidget(self.choice, 1)
        panel = QWidget()
        panel.setObjectName("page")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self.log)
        layout.addWidget(box, 1)
        self.controls = panel

    def refresh(self) -> None:
        """Ask for what was recorded."""
        session = self.session
        if session.path is None:
            return
        if self._stats:
            self._mark()
            return
        session.ask(
            "statistics",
            {"path": session.path},
            self._show,
            key="page-along",
            failed=self.note.setText,
        )

    def forget(self) -> None:
        """Drop what was read, as another file opens."""
        self._stats = {}

    def _show(self, value: Mapping[str, Any]) -> None:
        """Take what was recorded.

        Parameters
        ----------
        value : Mapping
            The kernel's ``statistics`` reply.
        """
        self._stats = {str(entry["name"]): entry for entry in value["stats"]}
        self._z = np.asarray(value["z"], float)
        accepted = np.asarray(value["steps"]["z"]).size
        rejected = np.asarray(value["steps"]["rejected_z"]).size
        self.note.setText(
            f"{len(self._stats)} quantities at {self._z.size} planes; "
            f"{accepted} steps, {rejected} rejected."
        )
        first = [name for name in self.wanted if name in self._stats]
        for fallback in ("energy", "peak_power", "peak_intensity", "fwhm_t"):
            if fallback in self._stats and fallback not in first:
                first.append(fallback)
        first = first[: self._SHOWN]
        self.choice.blockSignals(True)
        self.choice.clear()
        for name, entry in self._stats.items():
            item = QListWidgetItem(_title(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(str(entry.get("summary") or ""))
            item.setCheckState(
                Qt.CheckState.Checked if name in first else Qt.CheckState.Unchecked
            )
            self.choice.addItem(item)
        self.choice.blockSignals(False)
        self._draw()

    def _chosen(self, _item: QListWidgetItem) -> None:
        """Redraw for the quantities ticked.

        Parameters
        ----------
        _item : PySide6.QtWidgets.QListWidgetItem
            The item changed.
        """
        self._draw()

    def _log_chosen(self, value: object) -> None:
        """Take the choice of log axes.

        Parameters
        ----------
        value : object
            ``"auto"``, ``"on"`` or ``"off"``.
        """
        self._log = str(value)
        self._draw()

    def _names(self) -> list[str]:
        """Return the quantities ticked.

        Returns
        -------
        list of str
            At most four: a page shows what it can show large.
        """
        items = (self.choice.item(i) for i in range(self.choice.count()))
        names = [
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in items
            if item is not None and item.checkState() == Qt.CheckState.Checked
        ]
        return names[: self._SHOWN]

    def _draw(self) -> None:
        """Lay out a plot per quantity ticked."""
        clear_layout(self.grid)
        self._plots = []
        name, factor = length_unit(float(self._z[-1]) if self._z.size else 1.0)
        names = self._names()
        for index, stat in enumerate(names):
            entry = self._stats[stat]
            columns = entry["columns"]
            ys = [np.asarray(column["y"], float) for column in columns]
            unit = pretty(str(entry.get("units") or ""))
            shown_unit, scale = scaled_unit(np.concatenate(ys), unit)
            label = f"{_title(stat)} ({shown_unit})" if shown_unit else _title(stat)
            plot = LinePlot(
                xlabel=f"z ({name})",
                ylabel=label,
                names=[str(c["label"]) for c in columns],
                unit=shown_unit,
            )
            plot.along = "x"
            plot.setCursor(Qt.CursorShape.SizeHorCursor)
            plot.plane_at.connect(lambda v, f=factor: self.session.move_near(v / f))
            logged = self._log == "on" or (
                self._log == "auto" and wants_log(np.concatenate(ys))
            )
            plot.set_log(logged)
            plot.set_data(self._z * factor, [y * scale for y in ys])
            plot.fit()
            plot.setProperty("z_factor", factor)
            self._plots.append(plot)
            row, col = (index // 2, index % 2) if len(names) > 1 else (0, 0)
            self.grid.addLayout(_plotted(_title(stat), plot), row, col)
        self._mark()

    def _mark(self) -> None:
        """Mark the plane selected on the plots."""
        z = self.session.z
        for plot in self._plots:
            if z is None:
                plot.marker.hide()
            else:
                plot.marker.setPos(z * float(plot.property("z_factor")))
                plot.marker.show()

    def plane_moved(self) -> None:
        """Follow the scrubber, at no cost."""
        self._mark()

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the quantities shown, as columns.

        Returns
        -------
        tuple or None
            The position and each column, in SI; ``None`` with none shown.
        """
        names = self._names()
        if not names or self._z.size == 0:
            return None
        headers = ["z (m)"]
        columns = [self._z]
        for stat in names:
            for column in self._stats[stat]["columns"]:
                headers.append(str(column["label"]))
                columns.append(np.asarray(column["y"], float))
        return headers, np.column_stack(columns)

    def figure(self) -> dict[str, object] | None:
        """Return the page as solprop draws it.

        Returns
        -------
        dict
            One figure per statistic.
        """
        return {"plot": "plot_stats"}

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "along"


def _title(name: str) -> str:
    """Return a statistic's plain name.

    Parameters
    ----------
    name : str
        The name solprop gives it.

    Returns
    -------
    str
        What the page calls it.

    Examples
    --------
    >>> _title("peak_intensity"), _title("fwhm_t"), _title("energy")
    ('Peak intensity', 'Duration', 'Energy')
    """
    return {"fwhm_t": "Duration", "fwhm_f": "Bandwidth"}.get(
        name, name.replace("_", " ").capitalize()
    )


class ModesPage(PageView):
    """The energy in each mode along z, and each mode's spectrum at the plane.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the two plots and the chooser."""
        super().__init__(session)
        self.energy = LinePlot(xlabel="z", ylabel="Energy", unit="")
        self.spectra = LinePlot(
            xlabel=_AXIS_LABEL["wavelength"], ylabel="dB", unit="dB"
        )
        self.energy.along = "x"
        self._spectra: dict[int, np.ndarray] = {}
        self._x = np.zeros(0)
        self._read_stats = False
        self._factor = 1.0
        self.energy.plane_at.connect(lambda v: session.move_near(v / self._factor))
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        column.addLayout(_plotted("Energy in each mode along z", self.energy), 1)
        column.addLayout(
            _plotted("Each mode's spectrum at this plane", self.spectra), 1
        )
        self.choice = QListWidget()
        self.choice.setObjectName("modes-choice")
        self.choice.setFrameShape(QFrame.Shape.NoFrame)
        self.choice.itemChanged.connect(lambda _item: self._ask_spectra())
        self.controls = control_panel(named_label("Modes", "control"), self.choice)
        session.described.connect(self._fill)

    def forget(self) -> None:
        """Drop what was read, as another file opens."""
        self._read_stats = False
        self._spectra = {}

    def _fill(self, _value: object = None) -> None:
        """List the file's modes, all ticked.

        Parameters
        ----------
        _value : object, optional
            Not read.
        """
        self.choice.blockSignals(True)
        self.choice.clear()
        for index, label in enumerate(self.session.labels):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setCheckState(Qt.CheckState.Checked)
            self.choice.addItem(item)
        self.choice.blockSignals(False)

    def _selected(self) -> list[int]:
        """Return the modes ticked.

        Returns
        -------
        list of int
            Their indices.
        """
        items = (self.choice.item(i) for i in range(self.choice.count()))
        return [
            int(item.data(Qt.ItemDataRole.UserRole))
            for item in items
            if item is not None and item.checkState() == Qt.CheckState.Checked
        ]

    def refresh(self) -> None:
        """Ask for what the page shows."""
        session = self.session
        if session.path is None:
            return
        if not self._read_stats:
            session.ask(
                "statistics",
                {"path": session.path},
                self._show_energy,
                key="page-modes",
            )
        self._ask_spectra()

    def _show_energy(self, value: Mapping[str, Any]) -> None:
        """Draw the energy per mode.

        Parameters
        ----------
        value : Mapping
            The kernel's ``statistics`` reply.
        """
        self._read_stats = True
        z = np.asarray(value["z"], float)
        name, self._factor = length_unit(float(z[-1]) if z.size else 1.0)
        entry = next((e for e in value["stats"] if e["name"] == "energy"), None)
        if entry is None:
            return
        ys = [np.asarray(column["y"], float) for column in entry["columns"]]
        unit, scale = scaled_unit(np.concatenate(ys), "J")
        self.energy.set_names([str(column["label"]) for column in entry["columns"]])
        self.energy.set_xlabel(f"z ({name})")
        self.energy.set_ylabel(f"Energy ({unit})")
        self.energy.unit = unit
        self.energy.set_log(wants_log(np.concatenate(ys)))
        self.energy.set_data(z * self._factor, [y * scale for y in ys])
        self.energy.fit()
        self._mark()

    def _ask_spectra(self) -> None:
        """Ask for each mode ticked's spectrum at the plane selected."""
        session = self.session
        if session.path is None or not session.planes:
            return
        self._spectra = {}
        selected = self._selected()
        self.spectra.set_names([session.labels[index] for index in selected])
        for index in selected:
            session.ask(
                "plane",
                {
                    "path": session.path,
                    "index": session.index,
                    "modes": index,
                    "axis": session.options.axis,
                },
                lambda value, i=index: self._take_spectrum(i, value),
                key=f"page-modes-{index}",
            )
        self._mark()

    def _take_spectrum(self, index: int, value: Mapping[str, Any]) -> None:
        """Draw the spectra once each mode's has come.

        Parameters
        ----------
        index : int
            The mode.
        value : Mapping
            The kernel's ``plane`` reply.
        """
        axis = str(value["axis"])
        self._x = np.asarray(value["spectrum"]["x"], float) * _AXIS_SCALE[axis]
        self._spectra[index] = np.asarray(value["spectrum"]["y"], float)
        selected = self._selected()
        if all(i in self._spectra for i in selected) and selected:
            order = np.argsort(self._x)
            x = self._x[order]
            top = max(float(self._spectra[i].max()) for i in selected) or 1.0
            lines = [
                decibels(np.append(self._spectra[i][order], top))[:-1] for i in selected
            ]
            self.spectra.set_xlabel(_AXIS_LABEL[axis])
            self.spectra.set_data(x, lines)
            floor = self.session.options.floor
            self.spectra.set_view(_shown_range(value, axis, x), (floor, 3.0))

    def _mark(self) -> None:
        """Mark the plane selected."""
        z = self.session.z
        if z is not None:
            self.energy.marker.setPos(z * self._factor)
            self.energy.marker.show()

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "modes"


class CompressPage(PageView):
    """The pulse before, after and at its transform limit, through a compressor chain.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Host the chain beside its plots."""
        super().__init__(session)
        self.chain = PostprocessPage(session.bridge, session)
        self.chain.heading.hide()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.chain.view, 1)
        self.controls = self.chain

    def refresh(self) -> None:
        """Say what will be compressed."""
        self.chain.refresh()

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "compress"


class HealthPage(PageView):
    """How the run went: its steps, its boundary, its record and its log.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the plots and the log."""
        super().__init__(session)
        self.steps = LinePlot(xlabel="z", ylabel="Step size", names=["accepted"])
        self.steps.setProperty("rejected", None)
        self.rejected = pg.ScatterPlotItem(
            size=6, brush=pg.mkBrush(ORANGE), pen=None, name="rejected"
        )
        self.steps.addItem(self.rejected)
        self.boundary = LinePlot(xlabel="z", ylabel="Share removed at the edges")
        self.boundary.set_log(True)
        self.budget = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(ORANGE, style=Qt.PenStyle.DashLine)
        )
        self.budget.hide()
        self.boundary.addItem(self.budget, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.log = QPlainTextEdit()
        self.log.setObjectName("drawerText")
        self.log.setReadOnly(True)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.addLayout(_plotted("Step size along z", self.steps), 0, 0)
        grid.addLayout(
            _plotted("Light removed at the time window's edges", self.boundary), 0, 1
        )
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.note)
        column.addLayout(grid, 2)
        column.addLayout(_plotted("Log", self.log), 1)
        self.facts = named_label("", "note")
        self.facts.setWordWrap(True)
        self.facts.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.controls = control_panel(named_label("This run", "control"), self.facts)

    def refresh(self) -> None:
        """Ask how the run went."""
        session = self.session
        if session.path is None:
            return
        session.ask(
            "run_health",
            {"path": session.path},
            self._show,
            key="page-health",
            failed=self.note.setText,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw how the run went.

        Parameters
        ----------
        value : Mapping
            The kernel's ``run_health`` reply.
        """
        steps = value["steps"]
        z = np.asarray(steps["z"], float)
        h = np.asarray(steps["h"], float)
        rejected_z = np.asarray(steps["rejected_z"], float)
        rejected_h = np.asarray(steps["rejected_h"], float)
        name, factor = length_unit(float(z.max()) if z.size else 1.0)
        unit, scale = scaled_unit(h, "m")
        self.steps.set_xlabel(f"z ({name})")
        self.steps.set_ylabel(f"Step size ({unit})")
        self.steps.unit = unit
        self.steps.set_data(z * factor, [h * scale])
        self.steps.fit()
        self.rejected.setData(rejected_z * factor, rejected_h * scale)
        self._show_boundary(value.get("boundary"), name, factor)
        status = str(value.get("status") or "no record of how it ended")
        message = value.get("message")
        self.note.setText(
            f"{status.capitalize()}; {z.size} steps, {rejected_z.size} rejected"
            + (f". {message}" if message else ".")
        )
        self.facts.setText(_facts_text(value.get("facts") or {}))
        log = value.get("log")
        self.log.setPlainText("" if log is None else str(log))

    def _show_boundary(self, boundary: object, name: str, factor: float) -> None:
        """Draw what the boundary monitor recorded.

        Parameters
        ----------
        boundary : object
            The reply's ``boundary``.
        name : str
            The unit along z.
        factor : float
            What metres are multiplied by to read in it.
        """
        self.boundary.set_xlabel(f"z ({name})")
        if not isinstance(boundary, Mapping) or not len(boundary["z"]):
            self.boundary.set_data(np.zeros(0), [np.zeros(0)])
            self.budget.hide()
            return
        z = np.asarray(boundary["z"], float) * factor
        removed = np.asarray(boundary["removed"], float)
        self.boundary.set_data(z, [removed])
        budget = boundary.get("budget")
        high = float(removed.max())
        if isinstance(budget, int | float):
            self.budget.setPos(np.log10(budget))
            self.budget.show()
            high = max(high, float(budget))
        positive = removed[removed > 0]
        low = float(positive.min()) if positive.size else high * 1e-6
        self.boundary.set_view((float(z[0]), float(z[-1])), (low / 3.0, high * 3.0))

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "health"


def _facts_text(facts: Mapping[str, Any]) -> str:
    r"""Write what the run's record says.

    Parameters
    ----------
    facts : Mapping
        The reply's ``facts``.

    Returns
    -------
    str
        One line each, of those it has.

    Examples
    --------
    >>> _facts_text({"backend": "native", "wall_time": 42.0})
    'Backend: native\nTook 0:42'
    >>> _facts_text({"peak_memory": 2e8, "plasma_source": [1e-16, 1e-10]})
    'Peak memory 200 MB\nPlasma source check 1e-16 (limit 1e-10)'
    """
    lines = []
    if facts.get("backend"):
        lines.append(f"Backend: {facts['backend']}")
    if facts.get("precision"):
        lines.append(f"Precision: {facts['precision']}")
    if facts.get("threads"):
        lines.append(f"Threads: {facts['threads']}")
    if facts.get("wall_time") is not None:
        lines.append(f"Took {duration_text(float(facts['wall_time']))}")
    if facts.get("peak_memory") is not None:
        lines.append(f"Peak memory {size_text(float(facts['peak_memory']))}")
    plasma = facts.get("plasma_source")
    if plasma:
        lines.append(f"Plasma source check {plasma[0]:.2g} (limit {plasma[1]:.2g})")
    return "\n".join(lines)


PAGE_VIEWS: dict[str, Callable[[AnalysisSession], PageView]] = {
    "overview": OverviewPage,
    "plane": PlanePage,
    "spectrogram": SpectrogramPage,
    "along": AlongPage,
    "modes": ModesPage,
    "compress": CompressPage,
    "health": HealthPage,
}
"""The page each key of a class's rail shows."""
