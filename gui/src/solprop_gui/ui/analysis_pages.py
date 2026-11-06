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
    beam_reach,
    decibels,
    duration_text,
    even_rows,
    length_unit,
    light_range,
    mirrored_rows,
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
    "BeamOverviewPage",
    "BeamPlanePage",
    "CompressPage",
    "FarFieldPage",
    "HealthPage",
    "ModesPage",
    "OnAxisPage",
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

    def _failed(self, reason: str) -> None:
        """Say why a read failed, and leave nothing of an earlier file on the page.

        Parameters
        ----------
        reason : str
            The reason.
        """
        self.note.setText(reason)
        for plot in self.findChildren(LinePlot):
            plot.set_data(np.zeros(0), [np.zeros(0) for _ in plot.curves])

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


_FLAT_PHASE = 1e-3
"""A spectral phase moving less than this, in radians, is drawn as flat."""

_STEADY = 1e-12
"""A statistic moving less than this share of its size does not vary."""

_STAT_TITLES = {
    "density": "Gas density",
    "fwhm_f": "Bandwidth",
    "fwhm_t": "Duration",
    "fwhm_t_outer": "Outer duration",
    "omega0": "Centre frequency",
    "peak_power_total": "Peak power, all modes",
    "species_density": "Density by species",
    "species_electron_density": "Electron density by species",
    "species_ionisation": "Ionised fraction by species",
    "species_ionisation_energy": "Ionisation energy by species",
    "species_peak_ionisation_rate": "Peak ionisation rate by species",
}
"""What a page calls a statistic whose name does not read as it stands."""


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
        self._power_unit = "GW"
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

        self._build()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_LIVE_MS)
        self._timer.timeout.connect(self._draw_maps)

    def _build(self) -> None:
        """Make the plots and their controls."""
        session = self.session
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
            plot.say("")  # the stage covers them while the run starts
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
        power = np.array([np.asarray(r["power"], float) for r in self.rows])
        self._power_unit, scale = scaled_unit(power, "W")
        self._power = power * scale
        self.power_map.set_unit(self._power_unit)
        self.power_line.set_ylabel(f"Power ({self._power_unit})")
        self.power_line.unit = self._power_unit
        view = light_range(x, density, margin=0.1)
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
            (view, z_view),
            (options.floor, 0.0),
        )
        self.power_map.show_map(
            self._power[self._map_rows],
            (float(t[0]), bottom, float(t[-1] - t[0]), height),
            ((-window, window), z_view),
            (0.0, float(self._power.max()) or 1.0),
        )
        self.spectrum_line.set_view(view, (options.floor, 3.0))
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
        row = self._nearest_row(y, self._db.shape[0])
        if row is None:
            return ""
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
        row = self._nearest_row(y, self._power.shape[0])
        if row is None:
            return ""
        at = int(np.clip(np.searchsorted(t, x), 0, t.size - 1))
        text = f"{x:.3g} fs, {y:.3g} {self._unit[0]}"
        return f"{text}: {self._power[row, at]:.3g} {self._power_unit}"

    def _nearest_row(self, y: float, drawn: int) -> int | None:
        """Return the row nearest a position on a map.

        Parameters
        ----------
        y : float
            The position, in the map's unit.
        drawn : int
            The rows the map being read holds: during a run, planes arrive between
            drawings, and one map may be drawn before the other.

        Returns
        -------
        int or None
            The row; ``None`` where the map holds none yet.
        """
        rows = self.rows[:drawn]
        if not rows:
            return None
        z = np.array([float(r["z"]) for r in rows]) * self._unit[1]
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
        return ["time (fs)", f"power ({self._power_unit})"], np.column_stack(
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
            failed=self._failed,
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
        self._show_pulse(value)
        if "x" not in spectrum:
            # A planar run's plane has a power and no spectrum to read.
            self.spectrum.set_data(np.zeros(0), [np.zeros(0)])
            self.phase.set_data(np.zeros(0), [np.zeros(0)])
            self.note.setText(str(spectrum.get("error", "")))
            return
        x = np.asarray(spectrum["x"], float) * _AXIS_SCALE[axis]
        order = np.argsort(x)
        x = x[order]
        db = decibels(np.asarray(spectrum["y"], float))[order]
        self.spectrum.set_data(x, [db])
        self.spectrum.set_xlabel(_AXIS_LABEL[axis])
        view = _shown_range(value, axis, x)
        self.spectrum.set_view(view, (options.floor, 3.0))
        self._show_band(axis)
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
        power = np.asarray(value["power"]["y"], float)
        full = value.get("full_power")
        unit, scale = scaled_unit(power if full is None else np.asarray(full), "W")
        self.power.set_ylabel(f"Power ({unit})")
        self.power.unit = unit
        power = power * scale
        if full is None:
            self.power.set_names([""])
            self.power.set_data(t, [power])
            trace = power
        else:
            whole = np.asarray(full, float) * scale
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
        limits = (-1.0, 1.0)
        if keep.any():
            low, high = float(np.nanmin(y)), float(np.nanmax(y))
            # A flat phase is flat: its rounding is not to fill the axis.
            middle = (low + high) / 2.0
            flat = high - low < _FLAT_PHASE
            limits = (middle - 1.0, middle + 1.0) if flat else (low, high)
        self.phase.set_view(_shown_range(value, axis, x), limits)

    def _say(self, value: Mapping[str, Any]) -> None:
        """Say the numbers of the plane.

        Parameters
        ----------
        value : Mapping
            The kernel's ``plane`` reply.
        """
        if "energy" not in value:
            # A beam's plane is a profile across it; its numbers are its maps'.
            self.note.setText("The power and spectrum summed across the beam.")
            return
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
        shown = f"{name}, gate" if len(labels) > 1 else "Gate"
        self.note.setText(f"{shown} {gate * _FS:.3g} fs")

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
        self.reference: float | None = None
        self._said = ""
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
            failed=self._failed,
        )

    def forget(self) -> None:
        """Drop what was read, as another file opens."""
        self._stats = {}

    def set_reference(self, z: float | None) -> None:
        """Say where to mark a reference distance, as Marburger's is for a filament.

        Parameters
        ----------
        z : float or None
            The distance, m; ``None`` for none.
        """
        if z != self.reference:
            self.reference = z
            if self._stats:
                self._draw()

    def _show(self, value: Mapping[str, Any]) -> None:
        """Take what was recorded.

        Parameters
        ----------
        value : Mapping
            The kernel's ``statistics`` reply.
        """
        self._stats = {
            str(entry["name"]): entry for entry in value["stats"] if _varies(entry)
        }
        self._z = np.asarray(value["z"], float)
        accepted = np.asarray(value["steps"]["z"]).size
        rejected = np.asarray(value["steps"]["rejected_z"]).size
        self._said = f"{self._z.size} planes; {accepted} steps, {rejected} rejected."
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
        marked = "" if self.reference is None else " Dotted: Marburger's distance."
        self.note.setText(self._said + marked)
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
                names=_column_names(stat, columns),
                unit=shown_unit,
                left=None,
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
            if self.reference is not None:
                line = pg.InfiniteLine(
                    self.reference * factor,
                    angle=90,
                    pen=pg.mkPen(ORANGE, style=Qt.PenStyle.DotLine),
                )
                plot.addItem(line, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
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


_COLUMNS = {"beam_radius": ("second moment", "1/e²")}
"""What the columns of a statistic mean, where solprop numbers them."""


def _column_names(stat: str, columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return what a statistic's curves are called.

    Parameters
    ----------
    stat : str
        The statistic.
    columns : Sequence of Mapping
        Its columns, as the kernel gives them.

    Returns
    -------
    list of str
        Their names.

    Examples
    --------
    >>> _column_names("beam_radius", [{"label": "a"}, {"label": "b"}])
    ['second moment', '1/e²']
    >>> _column_names("energy", [{"label": "a"}])
    ['a']
    """
    named = _COLUMNS.get(stat)
    if named is not None and len(named) == len(columns):
        return list(named)
    return [str(column["label"]) for column in columns]


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
    return _STAT_TITLES.get(name, name.replace("_", " ").capitalize())


def _varies(entry: Mapping[str, Any]) -> bool:
    """Return whether a statistic changes along z, and so has anything to plot.

    The gas's pressure in a straight fibre, or an ionised fraction with plasma off, is
    recorded all the same, and would only lengthen the list.

    Parameters
    ----------
    entry : Mapping
        The statistic, from the kernel's ``statistics`` reply.

    Returns
    -------
    bool
        Whether any of its columns moves by more than rounding.

    Examples
    --------
    >>> _varies({"columns": [{"y": [1.0, 1.0]}, {"y": [1.0, 2.0]}]})
    True
    >>> _varies({"columns": [{"y": [0.0, 0.0]}]})
    False
    """
    for column in entry["columns"]:
        y = np.asarray(column["y"], float)
        if y.size and np.ptp(y) > _STEADY * float(np.abs(y).max()):
            return True
    return False


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
        self.energy_title = named_label("Energy in each mode along z", "plotTitle")
        column.addWidget(self.energy_title)
        column.addWidget(self.energy, 1)
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
        self.energy.setVisible(entry is not None)
        self.energy_title.setVisible(entry is not None)
        if entry is None:
            # A guide whose modes are not orthogonal records only their total.
            self.note.setText(
                "This run's modes are not orthogonal, so their energies do not add: "
                "it records their total (Along z), and each mode's spectrum below."
            )
            return
        self.note.setText("")
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
            failed=self._failed,
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


class BeamOverviewPage(OverviewPage):
    """A filament: its fluence along z, on-axis spectrum, and the plane across the beam.

    While a run goes it shows the fluence and the beam's spectrum, which each plane
    reports; the on-axis spectrum and the plane's map are read from the file after.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the maps and the plane's profile."""
        self._data: Mapping[str, Any] | None = None
        self._scale = "dB"
        self._fluence_floor = -30.0
        super().__init__(session)

    def _build(self) -> None:
        """Make the plots and their controls."""
        session = self.session
        self.fluence_map = MapPlot(
            xlabel="z", ylabel="r (µm)", colours="inferno", unit="dB"
        )
        self.fluence_map.set_along("x")
        self.spectrum_map = MapPlot(
            xlabel=_AXIS_LABEL["wavelength"], ylabel="z", colours="viridis", unit="dB"
        )
        self.spectrum_map.set_along("y")
        self.across_map = MapPlot(
            xlabel=_AXIS_LABEL["wavelength"],
            ylabel="r (µm)",
            colours="viridis",
            unit="dB",
        )
        self.profile = LinePlot(xlabel="r (µm)", ylabel="Fluence (J/cm²)", unit="J/cm²")
        for plot in (self.fluence_map, self.spectrum_map):
            plot.plane_at.connect(self._picked)
        self.fluence_map.reader = self._read_fluence
        self.spectrum_title = named_label("Beam spectrum along z", "plotTitle")
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addLayout(_plotted("Fluence along z", self.fluence_map), 0, 0)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        right.addWidget(self.spectrum_title)
        right.addWidget(self.spectrum_map, 1)
        grid.addLayout(right, 0, 1)
        grid.addLayout(_plotted("Across the beam at this plane", self.across_map), 1, 0)
        grid.addLayout(_plotted("Fluence at this plane", self.profile), 1, 1)
        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.scale_choice = Segments(
            "Fluence scale", (("dB", "dB"), ("linear", "Linear")), self._scale
        )
        self.scale_choice.chosen.connect(self._scale_chosen)
        self.fluence_floor = LabelledSlider("Fluence floor", -50, -5, -30, " dB")
        self.fluence_floor.moved.connect(self._fluence_floor_moved)
        self.floor = LabelledSlider(
            "Spectrum floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(self._floor_moved)
        self.controls = control_panel(
            self.axis_choice, self.scale_choice, self.fluence_floor, self.floor
        )
        self._z = np.zeros(0)
        self._r = np.zeros(0)
        self._flu = np.zeros((0, 0))
        self._spec = np.zeros((0, 0))
        self._x = np.zeros(0)
        self._reach = 0.0
        self._spectral_view = (0.0, 1.0)

    def forget(self) -> None:
        """Drop what was read, as another file opens."""
        self._data = None

    def clear(self) -> None:
        """Start empty, for a new run."""
        self._data = None
        self.rows = []
        self.extent = None
        self.shown = 0
        self.following = True
        self._live = True
        self._flu = np.zeros((0, 0))
        for plot in (self.fluence_map, self.spectrum_map, self.across_map):
            plot.say("")  # the stage covers them while the run starts
        self.profile.set_data(np.zeros(0), [np.zeros(0)])

    def refresh(self) -> None:
        """Read the file back; a run going draws from its rows."""
        session = self.session
        if session.path is None or self._live:
            self._draw_maps()
            return
        if self._data is None or self._data["axis"] != session.options.axis:
            if not self.rows:
                self.fluence_map.say("Reading…")
                self.spectrum_map.say("Reading…")
            session.ask(
                "beam_overview",
                {"path": session.path, "axis": session.options.axis},
                self._read_back,
                key="page-beam-overview",
                failed=self._refused,
            )
            return
        self._draw_maps()

    def _read_back(self, value: Mapping[str, Any]) -> None:
        """Take what the file holds.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_overview`` reply.
        """
        self._data = value
        self._draw_maps()

    def _refused(self, reason: str) -> None:
        """Say why there is nothing to draw.

        Parameters
        ----------
        reason : str
            The reason solprop gave.
        """
        self.fluence_map.say(reason)

    def _gather(self) -> bool:
        """Take the maps' data from the file, or from a run's rows so far.

        Returns
        -------
        bool
            Whether there is anything to draw.
        """
        axis = self.session.options.axis
        if self._data is not None and not self._live:
            self._z = np.asarray(self._data["z"], float)
            self._r = np.asarray(self._data["r"], float) * 1e6
            self._flu = np.asarray(self._data["fluence"], float) * 1e-4
            self._x = np.asarray(self._data["x"], float) * _AXIS_SCALE[axis]
            self._spec = decibels(np.maximum(self._data["on_axis"], 0.0))
            self.spectrum_title.setText("On-axis spectrum along z")
            return self._z.size > 0
        rows = [row for row in self.rows if "profile" in row]
        if not rows or axis not in self.spectra:
            return False
        self._z = np.array([float(row["z"]) for row in rows])
        self._r = np.asarray(rows[0]["profile"]["x"], float) * 1e6
        self._flu = (
            np.array([np.asarray(r["profile"]["y"], float) for r in rows]) * 1e-4
        )
        self._x = self.spectra[axis] * _AXIS_SCALE[axis]
        self._spec = decibels(
            np.array([np.asarray(r["density"][axis], float) for r in rows])
        )
        self.spectrum_title.setText("Beam spectrum along z")
        return True

    def _draw_maps(self) -> None:
        """Draw the fluence and the spectrum along z."""
        if not self._gather():
            return
        options = self.session.options
        z = self._z
        reach = self.extent[1] if self.extent and self._live else float(z[-1])
        name, factor = length_unit(reach or 1.0)
        self._unit = (name, factor)
        self._reach = beam_reach(self._r, self._flu)
        rows, first, step = even_rows(z)
        y, mirrored = mirrored_rows(self._r, self._flu[rows], self._reach)
        x_low = (first - step / 2) * factor
        width = step * rows.size * factor
        view = (float(z[0]) * factor, max(reach, float(z[-1])) * factor)
        self.fluence_map.set_xlabel(f"z ({name})")
        self.spectrum_map.set_ylabel(f"z ({name})")
        self.spectrum_map.set_xlabel(_AXIS_LABEL[options.axis])
        self.across_map.set_xlabel(_AXIS_LABEL[options.axis])
        self._mirrored = mirrored.T
        self._y = y
        self._fluence_rect = (x_low, float(y[0]), width, float(y[-1] - y[0]))
        self._fluence_view = (view, (float(y[0]), float(y[-1])))
        self._show_fluence()
        x = self._x
        lit = x[(self._spec > options.floor).any(axis=0)]
        span = (
            (float(lit[0]), float(lit[-1])) if lit.size else (float(x[0]), float(x[-1]))
        )
        margin = 0.05 * (span[1] - span[0])
        self._spectral_view = (span[0] - margin, span[1] + margin)
        self.spectrum_map.show_map(
            self._spec[rows],
            (float(x[0]), x_low, float(x[-1] - x[0]), width),
            (self._spectral_view, view),
            (options.floor, 0.0),
        )
        self.profile.set_view(
            (0.0, self._reach), (0.0, float(np.nanmax(self._flu)) * 1.05 or 1.0)
        )
        self._draw_plane()

    def _show_fluence(self) -> None:
        """Draw the fluence map on the scale chosen."""
        data = self._mirrored
        top = float(np.nanmax(data)) or 1.0
        linear = self._scale == "linear"
        self.fluence_map.show_map(
            data if linear else decibels(np.maximum(data, 0.0)),
            self._fluence_rect,
            self._fluence_view,
            (0.0, top) if linear else (self._fluence_floor, 0.0),
        )

    def _draw_plane(self) -> None:
        """Draw the plane selected, and mark it on the maps."""
        if self._flu.size == 0:
            return
        at = min(self.shown, self._flu.shape[0] - 1)
        z = float(self._z[at]) * self._unit[1]
        self.fluence_map.set_mark(z)
        self.spectrum_map.set_mark(z)
        self.profile.set_data(self._r, [self._flu[at]])
        session = self.session
        if self._live or session.path is None or not session.planes:
            self.across_map.say("Shown when the run has finished.")
            return
        session.ask(
            "beam_plane",
            {
                "path": session.path,
                "index": session.index,
                "axis": session.options.axis,
            },
            self._show_across,
            key="page-beam-across",
            failed=self.across_map.say,
        )

    def _show_across(self, value: Mapping[str, Any]) -> None:
        """Draw the plane's spectrum across the beam.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_plane`` reply.
        """
        axis = str(value["axis"])
        x = np.asarray(value["x"], float) * _AXIS_SCALE[axis]
        r = np.asarray(value["r"], float) * 1e6
        image = np.maximum(np.asarray(value["spectrum"], float), 0.0)
        reach = self._reach or float(r[-1])
        y, rows = mirrored_rows(r, image.T, reach)
        half = y >= 0
        self.across_map.show_map(
            decibels(rows[:, half].T),
            (float(x[0]), 0.0, float(x[-1] - x[0]), reach),
            (self._spectral_view, (0.0, reach)),
            (self.session.options.floor, 0.0),
        )

    def _scale_chosen(self, value: object) -> None:
        """Take the fluence's scale.

        Parameters
        ----------
        value : object
            ``"dB"`` or ``"linear"``.
        """
        self._scale = str(value)
        unit = "dB" if self._scale == "dB" else "J/cm²"
        self.fluence_map.bar.axis.setLabel(unit)
        if self._flu.size:
            self._show_fluence()

    def _fluence_floor_moved(self, value: int) -> None:
        """Move the fluence's floor, at no cost in the data.

        Parameters
        ----------
        value : int
            The floor, dB.
        """
        self._fluence_floor = float(value)
        if self._scale == "dB":
            self.fluence_map.set_levels((float(value), 0.0))

    def _floor_moved(self, value: int) -> None:
        """Move the spectrum's floor, at no cost in the data.

        Parameters
        ----------
        value : int
            The floor, dB.
        """
        self.session.options = replace(self.session.options, floor=float(value))
        self.spectrum_map.set_levels((float(value), 0.0))
        self.across_map.set_levels((float(value), 0.0))

    def _read_fluence(self, x: float, y: float) -> str:
        """Say what is at a point of the fluence map.

        Parameters
        ----------
        x : float
            The position along z, in the map's unit.
        y : float
            The radius, µm.

        Returns
        -------
        str
            The readout.
        """
        if self._flu.size == 0:
            return ""
        z = self._z * self._unit[1]
        row = int(np.argmin(np.abs(z - x)))
        at = int(np.argmin(np.abs(self._r - abs(y))))
        text = f"{x:.3g} {self._unit[0]}, {abs(y):.3g} µm"
        return f"{text}: {self._flu[row, at]:.3g} J/cm²"

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the plane's fluence, as columns.

        Returns
        -------
        tuple or None
            The radius and the fluence, in the columns' units.
        """
        if self._flu.size == 0:
            return None
        at = min(self.shown, self._flu.shape[0] - 1)
        return ["r (um)", "fluence (J/cm2)"], np.column_stack((self._r, self._flu[at]))

    def figure(self) -> dict[str, object] | None:
        """Say that solprop's plots do not draw this page.

        Returns
        -------
        None
            No figure: solprop has no plot of a beam's maps.
        """
        return None

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "beam_overview"


class BeamPlanePage(PageView):
    """One plane across the beam: in time, in its spectrum, and its fluence.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the maps and the profile."""
        super().__init__(session)
        self.in_time = MapPlot(
            xlabel="Time (fs)", ylabel="r (µm)", colours="inferno", unit="10¹² W/cm²"
        )
        self.in_spectrum = MapPlot(
            xlabel=_AXIS_LABEL["wavelength"],
            ylabel="r (µm)",
            colours="viridis",
            unit="dB",
        )
        self.profile = LinePlot(xlabel="r (µm)", ylabel="Fluence (J/cm²)", unit="J/cm²")
        self._last: Mapping[str, Any] | None = None
        self._reach = 0.0
        self._half_time = 0.0
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        grid.addLayout(_plotted("Across the beam, in time", self.in_time), 0, 0)
        grid.addLayout(
            _plotted("Across the beam, in its spectrum", self.in_spectrum), 0, 1
        )
        grid.addLayout(_plotted("Fluence", self.profile), 1, 0, 1, 2)
        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.floor = LabelledSlider(
            "Floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(
            lambda value: self.in_spectrum.set_levels((float(value), 0.0))
        )
        self.reach = LabelledSlider("Radius shown", 10, 5000, 500, " µm")
        self.reach.moved.connect(self._reach_moved)
        self.time_range = LabelledSlider("Time range", 5, 500, 60, " fs")
        self.time_range.moved.connect(self._window_moved)
        self.controls = control_panel(
            self.axis_choice, self.floor, self.reach, self.time_range
        )

    def refresh(self) -> None:
        """Ask for the plane selected."""
        session = self.session
        if session.path is None or not session.planes:
            return
        session.ask(
            "beam_plane",
            {
                "path": session.path,
                "index": session.index,
                "axis": session.options.axis,
            },
            self._show,
            key="page-beam-plane",
            failed=self.in_time.say,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw a plane.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_plane`` reply.
        """
        self._last = value
        axis = str(value["axis"])
        r = np.asarray(value["r"], float) * 1e6
        t = np.asarray(value["t"], float) * _FS
        fluence = np.asarray(value["fluence"], float) * 1e-4
        if self._reach <= 0:
            self._reach = beam_reach(r, fluence[None, :])
            self.reach.set_range(10, max(round(float(r[-1])), 11))
            self.reach.set_value(round(self._reach))
        intensity = np.maximum(np.asarray(value["intensity"], float), 0.0) * 1e-16
        if self._half_time <= 0:
            self._half_time = pulse_window(t, intensity.max(axis=0))
            self.time_range.set_range(1, max(round(float(np.abs(t).max())), 2))
            self.time_range.set_value(round(self._half_time))
        reach = self._reach
        y, image = mirrored_rows(r, intensity.T, reach)
        self.in_time.show_map(
            image.T,
            (float(t[0]), float(y[0]), float(t[-1] - t[0]), float(y[-1] - y[0])),
            ((-self._half_time, self._half_time), (float(y[0]), float(y[-1]))),
            (0.0, float(image.max()) or 1.0),
        )
        x = np.asarray(value["x"], float) * _AXIS_SCALE[axis]
        spectrum = np.maximum(np.asarray(value["spectrum"], float), 0.0)
        y, rows = mirrored_rows(r, spectrum.T, reach)
        self.in_spectrum.set_xlabel(_AXIS_LABEL[axis])
        self.in_spectrum.show_map(
            decibels(rows.T),
            (float(x[0]), float(y[0]), float(x[-1] - x[0]), float(y[-1] - y[0])),
            ((float(x[0]), float(x[-1])), (float(y[0]), float(y[-1]))),
            (float(self.floor.slider.value()), 0.0),
        )
        self.profile.set_data(r, [fluence])
        self.profile.set_view((0.0, reach), (0.0, float(fluence.max()) * 1.05 or 1.0))

    def _reach_moved(self, value: int) -> None:
        """Show another radius.

        Parameters
        ----------
        value : int
            The radius, µm.
        """
        self._reach = float(value)
        if self._last is not None:
            self._show(self._last)

    def _window_moved(self, value: int) -> None:
        """Show another time range, at no cost in the data.

        Parameters
        ----------
        value : int
            The half-width, fs.
        """
        self._half_time = float(value)
        self.in_time.box.setXRange(-value, value, padding=0)

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the plane's fluence, as columns.

        Returns
        -------
        tuple or None
            The radius and the fluence, in the columns' units.
        """
        if self.profile.xs.size == 0:
            return None
        return ["r (um)", "fluence (J/cm2)"], np.column_stack(
            (self.profile.xs, self.profile.ys[0])
        )

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "beam_plane"


class OnAxisPage(PageView):
    """One plane on the beam's axis: its pulse, its spectrum and its spectrogram.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the plots and the gate."""
        super().__init__(session)
        self.pulse = LinePlot(xlabel="Time (fs)", ylabel="Intensity", unit="")
        self.spectrum = LinePlot(
            xlabel=_AXIS_LABEL["wavelength"], ylabel="dB", unit="dB"
        )
        self.map = MapPlot(
            xlabel="Time (fs)",
            ylabel=_AXIS_LABEL["wavelength"],
            colours="viridis",
            unit="dB",
        )
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addLayout(_plotted("Pulse on the axis", self.pulse), 0, 0)
        grid.addLayout(_plotted("Spectrum on the axis", self.spectrum), 1, 0)
        grid.addLayout(_plotted("Spectrogram on the axis", self.map), 0, 1, 2, 1)
        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.floor = LabelledSlider(
            "Floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(lambda v: self.map.set_levels((float(v), 0.0)))
        self.gate = LabelledSlider("Gate", 1, 200, 20, " fs")
        self.gate.moved.connect(
            lambda v: session.set_options(gate_fwhm=float(v) * 1e-15)
        )
        self.controls = control_panel(self.axis_choice, self.floor, self.gate)

    def refresh(self) -> None:
        """Ask for the plane selected."""
        session = self.session
        if session.path is None or not session.planes:
            return
        session.ask(
            "beam_on_axis",
            {
                "path": session.path,
                "index": session.index,
                "axis": session.options.axis,
                "gate_fwhm": session.options.gate_fwhm,
                "positions": session.options.positions,
            },
            self._show,
            key="page-beam-axis",
            failed=self.map.say,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw a plane on the axis.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_on_axis`` reply.
        """
        axis = str(value["axis"])
        t = np.asarray(value["t"], float) * _FS
        power = np.asarray(value["power"], float)
        unit, scale = scaled_unit(power, "W/m²")
        self.pulse.set_ylabel(f"Intensity ({unit})")
        self.pulse.unit = unit
        self.pulse.set_data(t, [power * scale])
        half = pulse_window(t, np.maximum(power, 0.0))
        self.pulse.set_view(
            (-half, half), (0.0, float(power.max()) * scale * 1.05 or 1.0)
        )
        x = np.asarray(value["x"], float) * _AXIS_SCALE[axis]
        floor = float(self.floor.slider.value())
        self.spectrum.set_xlabel(_AXIS_LABEL[axis])
        self.spectrum.set_data(x, [decibels(np.maximum(value["spectrum"], 0.0))])
        drawn = value["spectrogram"]
        map_x = np.asarray(drawn["x"], float) * _AXIS_SCALE[axis]
        map_t = np.asarray(drawn["t"], float) * _FS
        self.spectrum.set_view((float(map_x[0]), float(map_x[-1])), (floor, 3.0))
        self.map.set_ylabel(_AXIS_LABEL[axis])
        self.map.show_map(
            decibels(np.asarray(drawn["values"], float)),
            (
                float(map_t[0]),
                float(map_x[0]),
                float(map_t[-1] - map_t[0]),
                float(map_x[-1] - map_x[0]),
            ),
            ((-half * 2, half * 2), (float(map_x[0]), float(map_x[-1]))),
            (floor, 0.0),
        )
        self.gate.set_value(round(float(drawn["gate_fwhm"]) * _FS))
        self.note.setText(f"Gate {float(drawn['gate_fwhm']) * _FS:.3g} fs")

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "on_axis"


class FarFieldPage(PageView):
    """One plane's far field: angle against wavelength, where conical emission shows.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the map and its controls."""
        super().__init__(session)
        self.map = MapPlot(
            xlabel=_AXIS_LABEL["wavelength"],
            ylabel="Angle (mrad)",
            colours="viridis",
            unit="dB",
        )
        self._angles = np.zeros(0)
        self._x = np.zeros(0)
        self._db = np.zeros((0, 0))
        self.map.reader = self._read
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.note)
        column.addWidget(self.map, 1)
        self.axis_choice = Segments("Spectral axis", _AXES, session.options.axis)
        self.axis_choice.chosen.connect(lambda axis: session.set_options(axis=axis))
        self.floor = LabelledSlider(
            "Floor", -80, -10, int(session.options.floor), " dB"
        )
        self.floor.moved.connect(lambda v: self.map.set_levels((float(v), 0.0)))
        self.range = LabelledSlider("Angle shown", 1, 500, 50, " mrad")
        self.range.moved.connect(lambda v: self.map.box.setYRange(0, v, padding=0))
        self.controls = control_panel(self.axis_choice, self.floor, self.range)
        self.note.setText("Energy per solid angle, the angle being outside the medium.")

    def refresh(self) -> None:
        """Ask for the plane selected's far field."""
        session = self.session
        if session.path is None or not session.planes:
            return
        session.ask(
            "beam_far_field",
            {
                "path": session.path,
                "index": session.index,
                "axis": session.options.axis,
            },
            self._show,
            key="page-beam-far",
            failed=self.map.say,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw a far field.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_far_field`` reply.
        """
        axis = str(value["axis"])
        self._angles = np.asarray(value["theta"], float) * 1e3
        self._x = np.asarray(value["x"], float) * _AXIS_SCALE[axis]
        self._db = decibels(np.asarray(value["values"], float))
        x, theta = self._x, self._angles
        floor = float(self.floor.slider.value())
        lit = (self._db > floor).any(axis=1)
        reach = float(theta[lit].max()) * 1.1 if lit.any() else float(theta[-1])
        reach = min(max(reach, float(theta[1])), float(theta[-1]))
        self.range.set_range(1, max(round(float(theta[-1])), 2))
        self.range.set_value(round(reach))
        self.map.set_xlabel(_AXIS_LABEL[axis])
        self.map.show_map(
            self._db,
            (float(x[0]), 0.0, float(x[-1] - x[0]), float(theta[-1])),
            ((float(x[0]), float(x[-1])), (0.0, reach)),
            (floor, 0.0),
        )

    def _read(self, x: float, y: float) -> str:
        """Say what is at a point of the map.

        Parameters
        ----------
        x : float
            The position on the spectral axis.
        y : float
            The angle, mrad.

        Returns
        -------
        str
            The readout.
        """
        if self._db.size == 0:
            return ""
        column = int(np.clip(np.searchsorted(self._x, x), 0, self._x.size - 1))
        row = int(np.clip(np.searchsorted(self._angles, y), 0, self._angles.size - 1))
        unit = _AXIS_UNIT[self.session.options.axis]
        return f"{x:.4g} {unit}, {y:.3g} mrad: {self._db[row, column]:.1f} dB"

    def python(self) -> str | None:
        """Return the page's snippet.

        Returns
        -------
        str
            Its key.
        """
        return "far_field"


class FluencePage(PageView):
    """A Cartesian beam's fluence across x and y at the plane, and through its middle.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the map and the two profiles."""
        super().__init__(session)
        self.map = MapPlot(
            xlabel="x (mm)", ylabel="y (mm)", colours="inferno", unit="J/cm²"
        )
        # A round beam is drawn round.
        self.map.box.setAspectLocked(lock=True, ratio=1)
        self.profiles = LinePlot(
            xlabel="Position (mm)",
            ylabel="Fluence (J/cm²)",
            names=["along x, through y = 0", "along y, through x = 0"],
            unit="J/cm²",
        )
        self._last: Mapping[str, Any] | None = None
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addLayout(_plotted("Fluence across the beam", self.map), 0, 0)
        grid.addLayout(_plotted("Through its middle", self.profiles), 0, 1)

    def refresh(self) -> None:
        """Ask for the plane selected."""
        session = self.session
        if session.path is None or not session.planes:
            return
        session.ask(
            "beam_fluence",
            {"path": session.path, "index": session.index},
            self._show,
            key="page-beam-fluence",
            failed=self.map.say,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw a plane's fluence.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_fluence`` reply.
        """
        self._last = value
        x = np.asarray(value["x"], float) * 1e3
        y = np.asarray(value["y"], float) * 1e3
        fluence = np.asarray(value["fluence"], float)
        unit, scale = scaled_unit(fluence, "J/m²")
        shown = fluence * scale
        self.map.set_unit(unit)
        self.profiles.set_ylabel(f"Fluence ({unit})")
        self.profiles.unit = unit
        dx, dy = float(np.diff(x[:2])[0]), float(np.diff(y[:2])[0])
        self.map.show_map(
            shown,
            (float(x[0]) - dx / 2, float(y[0]) - dy / 2, x.size * dx, y.size * dy),
            ((float(x[0]), float(x[-1])), (float(y[0]), float(y[-1]))),
            (0.0, float(shown.max()) or 1.0),
        )
        middle_x = shown[int(np.argmin(np.abs(y))), :]
        middle_y = shown[:, int(np.argmin(np.abs(x)))]
        # Two axes on one: x and y are each drawn over their own positions.
        positions = np.union1d(x, y)
        self.profiles.set_data(
            positions,
            [
                np.interp(positions, x, middle_x, left=np.nan, right=np.nan),
                np.interp(positions, y, middle_y, left=np.nan, right=np.nan),
            ],
        )
        self.profiles.fit()
        self.note.setText(
            f"Energy {si(float(fluence.sum()) * dx * dy * 1e-6, 'J', digits=3)} "
            f"across the grid at z = {float(value['z']) * 1e3:.4g} mm."
        )

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the plane's fluence, a row per y.

        Returns
        -------
        tuple or None
            ``y (mm)`` and a column per x, in J/m²; ``None`` before a plane is read.
        """
        if self._last is None:
            return None
        x = np.asarray(self._last["x"], float) * 1e3
        y = np.asarray(self._last["y"], float) * 1e3
        fluence = np.asarray(self._last["fluence"], float)
        names = ["y (mm)"] + [f"x = {value:.4g} mm (J/m²)" for value in x]
        return names, np.column_stack((y, fluence))


class LineFluencePage(PageView):
    """A line beam's fluence across x along z, and at the plane.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Make the map and the profile."""
        super().__init__(session)
        self.map = MapPlot(xlabel="x (mm)", ylabel="z", colours="inferno", unit="J/cm²")
        self.map.set_along("y")
        self.map.plane_at.connect(self._picked)
        self.profile = LinePlot(xlabel="x (mm)", ylabel="Fluence", unit="")
        self._data: Mapping[str, Any] | None = None
        self._unit = ("m", 1.0)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addLayout(_plotted("Fluence across the beam, along z", self.map), 0, 0)
        grid.addLayout(_plotted("At this plane", self.profile), 0, 1)

    def refresh(self) -> None:
        """Ask for every plane's fluence, once a file is open; then draw the plane."""
        session = self.session
        if session.path is None or not session.planes:
            return
        if self._data is not None and self._data.get("path") == session.path:
            self._draw_plane()
            return
        session.ask(
            "beam_line",
            {"path": session.path},
            self._show,
            key="page-beam-line",
            failed=self.map.say,
        )

    def _show(self, value: Mapping[str, Any]) -> None:
        """Draw the map, and the plane.

        Parameters
        ----------
        value : Mapping
            The kernel's ``beam_line`` reply.
        """
        self._data = {**value, "path": self.session.path}
        x = np.asarray(value["x"], float) * 1e3
        z = np.asarray(value["z"], float)
        fluence = np.asarray(value["fluence"], float)
        unit, scale = scaled_unit(fluence, "J/m²")
        name, factor = length_unit(float(z[-1]) or 1.0)
        self._unit = (name, factor)
        rows, first, step = even_rows(z)
        shown = fluence[rows] * scale
        dx = float(np.diff(x[:2])[0])
        self.map.set_unit(unit)
        self.map.set_ylabel(f"z ({name})")
        self.map.show_map(
            shown,
            (
                float(x[0]) - dx / 2,
                (first - step / 2) * factor,
                x.size * dx,
                step * rows.size * factor,
            ),
            (
                (float(x[0]), float(x[-1])),
                (float(z[0]) * factor, float(z[-1]) * factor),
            ),
            (0.0, float(shown.max()) or 1.0),
        )
        self.profile.set_ylabel(f"Fluence ({unit})")
        self.profile.unit = unit
        self._draw_plane()

    def forget(self) -> None:
        """Drop what was read, as another file opens."""
        self._data = None

    def _draw_plane(self) -> None:
        """Draw the plane the scrubber is on, and mark it on the map."""
        if self._data is None:
            return
        x = np.asarray(self._data["x"], float) * 1e3
        z = np.asarray(self._data["z"], float)
        fluence = np.asarray(self._data["fluence"], float)
        _unit, scale = scaled_unit(fluence, "J/m²")
        at = min(self.session.index, fluence.shape[0] - 1)
        self.map.set_mark(float(z[at]) * self._unit[1])
        self.profile.set_data(x, [fluence[at] * scale])
        self.profile.set_view(
            (float(x[0]), float(x[-1])),
            (0.0, float(fluence.max()) * scale * 1.05 or 1.0),
        )
        self.note.setText(f"{fluence.shape[0]} planes; drawn at z = {z[at]:.4g} m.")

    def _picked(self, value: float) -> None:
        """Take a plane picked on the map.

        Parameters
        ----------
        value : float
            The position, in the map's unit.
        """
        self.session.move_near(value / self._unit[1])

    def data(self) -> tuple[list[str], np.ndarray] | None:
        """Return the fluence at the plane, as columns.

        Returns
        -------
        tuple or None
            ``x (mm)`` and the fluence in J/m²; ``None`` before the file is read.
        """
        if self._data is None:
            return None
        fluence = np.asarray(self._data["fluence"], float)
        at = min(self.session.index, fluence.shape[0] - 1)
        x = np.asarray(self._data["x"], float) * 1e3
        return ["x (mm)", "fluence (J/m²)"], np.column_stack((x, fluence[at]))


PAGE_VIEWS: dict[str, Callable[[AnalysisSession], PageView]] = {
    "overview": OverviewPage,
    "plane": PlanePage,
    "spectrogram": SpectrogramPage,
    "along": AlongPage,
    "modes": ModesPage,
    "compress": CompressPage,
    "health": HealthPage,
    "beam_overview": BeamOverviewPage,
    "beam_plane": BeamPlanePage,
    "on_axis": OnAxisPage,
    "far_field": FarFieldPage,
    "fluence_xy": FluencePage,
    "fluence_xz": LineFluencePage,
}
"""The page each key of a class's rail shows."""
