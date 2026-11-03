# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plot options panel, at the bottom left of the Propagate and Results steps.

It edits one :class:`~solprop_gui.model.plot_options.PlotOptions`: the spectral axis,
typed limits for the spectral and time axes (blank for automatic), the spectral scale
and floor, and a bandpass for the time-domain views. The window keeps one set of
options and shows it in each step's panel.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from solprop_gui.model.plot_options import (
    AXES,
    SCALES,
    UNITS,
    PlotOptions,
    from_display,
    parse_limit,
    to_display,
)

__all__ = ["PlotOptionsPanel", "SpectrogramOptions"]

_BAD = "QLineEdit { border: 1px solid #c62828; }"


def _text(value: float | None) -> str:
    """Return a limit as the panel shows it.

    Parameters
    ----------
    value : float or None
        The limit in its display unit; ``None`` for automatic.

    Returns
    -------
    str
        The number, compactly, or blank.
    """
    return "" if value is None else f"{value:.6g}"


class _Range(QWidget):
    """Two typed limits, low and high, with a unit.

    Parameters
    ----------
    name : str
        The fields' object-name prefix.
    unit : str
        The unit shown after them.
    """

    edited = Signal()

    def __init__(self, name: str, unit: str) -> None:
        """Make the two fields."""
        super().__init__()
        self.low = QLineEdit()
        self.high = QLineEdit()
        for field, end in ((self.low, "low"), (self.high, "high")):
            field.setObjectName(f"{name}-{end}")
            field.setPlaceholderText("auto")
            field.setMaximumWidth(80)
            field.editingFinished.connect(self.edited)
        self.unit = QLabel(unit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.low)
        layout.addWidget(QLabel("to"))
        layout.addWidget(self.high)
        layout.addWidget(self.unit)
        layout.addStretch(1)

    def values(self) -> tuple[float | None, float | None] | None:
        """Return the typed limits, ascending, marking a field that is not a number.

        Returns
        -------
        tuple or None
            The limits in the display unit, ``None`` where blank; ``None`` when
            either is not a number.
        """
        found: list[float | None] = []
        good = True
        for field in (self.low, self.high):
            try:
                found.append(parse_limit(field.text()))
                field.setStyleSheet("")
            except ValueError:
                field.setStyleSheet(_BAD)
                good = False
        if not good:
            return None
        low, high = found
        if low is not None and high is not None and low > high:
            low, high = high, low
        return low, high

    def show_values(self, low: float | None, high: float | None) -> None:
        """Show two limits.

        Parameters
        ----------
        low, high : float or None
            The limits in the display unit; ``None`` for automatic.
        """
        self.low.setText(_text(low))
        self.high.setText(_text(high))
        for field in (self.low, self.high):
            field.setStyleSheet("")


class PlotOptionsPanel(QWidget):
    """Edit the plot options.

    Parameters
    ----------
    name : str
        A prefix for its widgets' object names, to tell two panels apart.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the new :class:`PlotOptions` when an option is changed.
    """

    changed = Signal(object)

    def __init__(self, name: str = "plot", parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a form
        """Make the fields, showing the default options."""
        super().__init__(parent)
        self.setObjectName(f"{name}-options")
        self._options = PlotOptions()
        self.axis = QComboBox()
        self.axis.setObjectName(f"{name}-axis")
        for key, label in AXES.items():
            self.axis.addItem(f"{label} ({UNITS[key][0]})", key)
        self.axis.setToolTip(
            "The abscissa of spectra, maps and spectrograms. solprop gives the "
            "density per unit of the axis chosen."
        )
        self.axis.currentIndexChanged.connect(self._axis_changed)
        self.spectral = _Range(f"{name}-spectral", UNITS["wavelength"][0])
        self.time = _Range(f"{name}-time", UNITS["time"][0])
        self.scale = QComboBox()
        self.scale.setObjectName(f"{name}-scale")
        self.scale.addItems(list(SCALES))
        self.floor = QDoubleSpinBox()
        self.floor.setObjectName(f"{name}-floor")
        self.floor.setRange(-200.0, -5.0)
        self.floor.setSuffix(" dB")
        self.floor.setToolTip("The bottom of the dB scale.")
        scale = QWidget()
        scale_row = QHBoxLayout(scale)
        scale_row.setContentsMargins(0, 0, 0, 0)
        scale_row.addWidget(self.scale)
        scale_row.addWidget(QLabel("floor"))
        scale_row.addWidget(self.floor)
        scale_row.addStretch(1)
        self.band_on = QCheckBox("within")
        self.band_on.setObjectName(f"{name}-band-on")
        self.band_on.setToolTip(
            "Compute power, the time-domain views and the spectrogram from the light "
            "in this wavelength band only (solprop's Planck-tapered bandpass)."
        )
        self.band = _Range(f"{name}-band", UNITS["wavelength"][0])
        band = QWidget()
        band_row = QHBoxLayout(band)
        band_row.setContentsMargins(0, 0, 0, 0)
        band_row.addWidget(self.band_on)
        band_row.addWidget(self.band, 1)
        self.note = QLabel("")
        self.note.setObjectName(f"{name}-options-note")
        self.note.setWordWrap(True)
        self.reset = QPushButton("Reset")
        self.reset.setObjectName(f"{name}-options-reset")
        self.reset.clicked.connect(lambda: self._emit(PlotOptions()))

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Spectral axis", self.axis)
        form.addRow("Spectral range", self.spectral)
        form.addRow("Time range", self.time)
        form.addRow("Spectra", scale)
        form.addRow("Bandpass", band)
        form.addRow(self.note)
        form.addRow(self.reset)
        for signal in (
            self.spectral.edited,
            self.time.edited,
            self.band.edited,
            self.scale.currentIndexChanged,
            self.floor.valueChanged,
            self.band_on.toggled,
        ):
            signal.connect(self._edited)
        self.set_options(self._options)

    @property
    def options(self) -> PlotOptions:
        """Return the options shown.

        Returns
        -------
        PlotOptions
            The options.
        """
        return self._options

    def set_options(self, options: PlotOptions) -> None:
        """Show options, without announcing them.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        self._options = options
        widgets = (self.axis, self.scale, self.floor, self.band_on)
        for widget in widgets:
            widget.blockSignals(True)
        self.axis.setCurrentIndex(self.axis.findData(options.axis))
        self.spectral.unit.setText(UNITS[options.axis][0])
        self.spectral.show_values(*options.shown_limits("spectral"))
        self.time.show_values(*options.shown_limits("time"))
        self.scale.setCurrentText(options.scale)
        self.floor.setValue(options.floor)
        self.floor.setEnabled(options.scale == "dB")
        self.band_on.setChecked(options.bandpass is not None)
        if options.bandpass is not None:
            low, high = (to_display("wavelength", v) for v in options.bandpass)
            self.band.show_values(low, high)
        for widget in widgets:
            widget.blockSignals(False)

    def _limits(self, field: _Range, axis: str) -> tuple[float | None, float | None]:
        """Return a range's typed limits in SI, or raise for a bad one.

        Parameters
        ----------
        field : _Range
            The range.
        axis : str
            Its axis.

        Returns
        -------
        tuple
            The limits in SI, ascending, ``None`` where automatic.

        Raises
        ------
        ValueError
            If either is not a number.
        """
        values = field.values()
        if values is None:
            msg = "not a number"
            raise ValueError(msg)
        low, high = (None if v is None else from_display(axis, v) for v in values)
        return low, high

    def _read(self) -> PlotOptions | None:
        """Return the options the fields show, or ``None`` while one is invalid.

        Returns
        -------
        PlotOptions or None
            The options.
        """
        axis = str(self.axis.currentData())
        try:
            spectral = self._limits(self.spectral, axis)
            time = self._limits(self.time, "time")
            band = self._limits(self.band, "wavelength")
        except ValueError:
            self.note.setText("A limit is not a number: type a number, or leave blank.")
            return None
        bandpass = None
        if self.band_on.isChecked():
            if band[0] is None or band[1] is None or band[0] <= 0:
                self.note.setText("A bandpass needs two positive wavelengths.")
                return None
            bandpass = (band[0], band[1])
        self.note.setText("")
        return PlotOptions(
            axis=axis,  # pyright: ignore[reportArgumentType]
            spectral_limits=spectral,
            time_limits=time,
            scale=self.scale.currentText(),  # pyright: ignore[reportArgumentType]
            floor=self.floor.value(),
            bandpass=bandpass,
            gate_fwhm=self._options.gate_fwhm,  # the spectrogram's, set elsewhere
            positions=self._options.positions,
        )

    def _edited(self, *_args: object) -> None:
        """Announce the options after a field changed, if they are valid.

        Parameters
        ----------
        *_args : object
            What the signal carries, unused.
        """
        self.floor.setEnabled(self.scale.currentText() == "dB")
        options = self._read()
        if options is not None and options != self._options:
            self._options = options
            self.changed.emit(options)

    def _axis_changed(self, _index: int) -> None:
        """Change the spectral axis: its typed limits, in the old unit, are cleared.

        Parameters
        ----------
        _index : int
            The axis's place, unused.
        """
        axis = str(self.axis.currentData())
        self.spectral.unit.setText(UNITS[axis][0])
        self.spectral.show_values(None, None)
        self._edited()

    def _emit(self, options: PlotOptions) -> None:
        """Show options and announce them.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        self.set_options(options)
        self.band.show_values(None, None)
        self.note.setText("")
        self.changed.emit(options)


class SpectrogramOptions(QWidget):
    """The spectrogram's gate: its width and the number of its positions.

    The time range is the plot options' time range; the mode is the Plane view's
    (a sum of modes takes the first: a spectrogram is of one field).

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the gate's width (s, or ``None`` for solprop's default) and
        the positions.
    """

    changed = Signal(object, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the two fields."""
        super().__init__(parent)
        self.gate = QLineEdit()
        self.gate.setObjectName("spectrogram-gate")
        self.gate.setPlaceholderText("auto")
        self.gate.setMaximumWidth(80)
        self.gate.setToolTip(
            "The Gaussian gate's full width at half maximum. A short gate shows when, "
            "a long one shows what; solprop's default is a twentieth of the time range."
        )
        self.gate.editingFinished.connect(self._edited)
        self.positions = QSpinBox()
        self.positions.setObjectName("spectrogram-positions")
        self.positions.setRange(8, 4096)
        self.positions.setValue(256)
        self.positions.setToolTip("How many gate positions across the time range.")
        self.positions.editingFinished.connect(self._edited)
        gate = QWidget()
        row = QHBoxLayout(gate)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.gate)
        row.addWidget(QLabel(UNITS["time"][0]))
        row.addStretch(1)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Gate width", gate)
        form.addRow("Positions", self.positions)
        form.addRow(QLabel("Its time range is the plot options'."))
        self._shown: tuple[float | None, int] = (None, 256)

    def set_options(self, options: PlotOptions) -> None:
        """Show the gate the options hold, without announcing it.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        gate = options.gate_fwhm
        self.gate.setText("" if gate is None else _text(to_display("time", gate)))
        self.gate.setStyleSheet("")
        self.positions.blockSignals(True)
        self.positions.setValue(options.positions)
        self.positions.blockSignals(False)
        self._shown = (gate, options.positions)

    def _edited(self) -> None:
        """Announce the gate, if it is blank or a positive number."""
        try:
            value = parse_limit(self.gate.text())
        except ValueError:
            value = -1.0
        if value is not None and value <= 0:
            self.gate.setStyleSheet(_BAD)
            return
        self.gate.setStyleSheet("")
        chosen = (
            None if value is None else from_display("time", value),
            self.positions.value(),
        )
        if chosen != self._shown:
            self._shown = chosen
            self.changed.emit(*chosen)
