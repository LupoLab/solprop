# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design a waveguide: a guide's curves and numbers, no pulse or run (design 7.4).

The design never touches an open setup. Every edit asks the kernel for what it needs,
latest wins, and every reply carries the recipe the design stands for, which
*Simulate this* opens as a new sheet.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from solprop.recipe.quantity import Unit, parse_quantity

from solprop_gui.classes.problem import Derived, derived_text
from solprop_gui.registry.explore import design_sweep, sweep_values
from solprop_gui.registry.pages import GAS_TITLES
from solprop_gui.service.messages import Reply
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.quantity import QuantityField
from solprop_gui.ui.theme import named_label
from solprop_gui.views.controls import Segments
from solprop_gui.views.page_plots import ORANGE, LinePlot, PlotMarks

__all__ = ["DesignStage"]

_DEBOUNCE_MS: Final = 150
"""How long typing rests before the kernel is asked."""

_PAGES: Final = (
    ("curves", "Curves"),
    ("map", "ZDW map"),
    ("phase", "Phase matching"),
    ("loss", "Loss"),
)
"""The pages of the large plot."""

_CURVES: Final = (
    ("beta2", "Group-velocity dispersion β₂", "β₂ (fs²/m)", "fs²/m"),
    ("dispersion", "Dispersion parameter D", "D (ps/(nm km))", "ps/(nm km)"),
    ("beta3", "Third-order dispersion β₃", "β₃ (fs³/m)", "fs³/m"),
    ("neff", "Effective index - 1", "n_eff - 1", ""),
    ("group_index", "Group index", "Group index", ""),
)
"""What the Curves page can plot: key, name, axis and unit."""

_FIXED: Final = {
    "phase": ("mismatch", "Phase mismatch Δβ (1/cm)", "1/cm"),
    "loss": ("loss", "Loss (dB/m)", "dB/m"),
}
"""What the other pages of curves plot: key, axis and unit."""

_SHARED: Final = (
    ("radius", "Core radius", "m", "um", "core"),
    ("pressure", "Pressure", "bar", "bar", "core"),
    ("wavelength", "Pump wavelength", "m", "nm", "core"),
)
"""The rows of every guide: path, label, keyword unit, unit shown, section."""

_LATTICE: Final = (
    ("lattice.n_tubes", "Tubes", None, None, "lattice"),
    ("lattice.tube_radius", "Tube radius", "m", "um", "lattice"),
    ("lattice.wall_thickness", "Wall thickness", "m", "nm", "lattice"),
)
"""The rows of an antiresonant fibre."""

_PULSE: Final = (
    ("fwhm", "Duration", "s", "fs", "pulse"),
    ("energy", "Energy", "J", "uJ", "pulse"),
)
"""The rows of the optional pulse."""

_STARTS: Final = {
    "capillary": {
        "radius": "125 um",
        "gas": "He",
        "pressure": "1 bar",
        "wavelength": "800 nm",
        "fwhm": "10 fs",
        "energy": "200 uJ",
    },
    "antiresonant": {
        "radius": "15 um",
        "gas": "Ar",
        "pressure": "1 bar",
        "wavelength": "800 nm",
        "lattice.n_tubes": 6,
        "lattice.tube_radius": "9 um",
        "lattice.wall_thickness": "180 nm",
        "fwhm": "30 fs",
        "energy": "2 uJ",
    },
}
"""Where each guide's design starts."""

_SWEEPS: Final = (
    (None, "No sweep", ""),
    ("pressure", "Pressure", "0.5, 1, 2, 5"),
    ("radius", "Core radius", "50, 100, 150, 250"),
    ("gas", "Gas", "He, Ne, Ar"),
)
"""What can be swept: path, name and the values offered at first."""

_NUMBERS: Final = (
    Derived("beta2_fs", "β₂ at the pump", "fs²/m", prefixed=False),
    Derived("beta3_fs", "β₃ at the pump", "fs³/m", prefixed=False),
    Derived("loss", "Loss at the pump", "dB/m", prefixed=False),
    Derived("aeff_um", "Effective area", "µm²", prefixed=False),
    Derived("gamma_km", "γ", "1/(W km)", prefixed=False),  # noqa: RUF001 - gamma
    Derived("zdw", "Zero-dispersion wavelength", "m"),
)
"""The numbers every design has."""

_PULSE_NUMBERS: Final = (
    Derived("soliton_order", "Soliton order", prefixed=False),
    Derived("peak_power", "Peak power", "W"),
    Derived("dispersion_length", "Dispersion length", "m"),
    Derived("nonlinear_length", "Nonlinear length", "m"),
    Derived("fission_length", "Fission length", "m"),
    Derived("rdw", "Dispersive wave", "m"),
)
"""The numbers a design with a pulse also has."""

_MAP_ZEROS: Final = (250, 300, 350, 400, 450, 500, 600, 700)
"""The zero-dispersion wavelengths a map draws, nm, those below the pump."""

_MAP_RADII: Final = 9
"""How many radii a map solves for each zero."""


class DesignStage(QWidget):
    """Design a guide: its inputs on one panel, one large plot and its numbers.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The kernel.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    text : str or None
        The recipe the design stands for, as last read; ``None`` before any.
    guide : str
        ``"capillary"`` or ``"antiresonant"``.
    page : str
        The large plot's page.
    fields : dict
        The number boxes, by recipe path.
    numbers : dict
        The numbers' labels, by key.
    plot : solprop_gui.views.page_plots.LinePlot
        The large plot.
    note : PySide6.QtWidgets.QLabel
        A line above the plot, for what it shows or why it does not.
    """

    def __init__(self, bridge: ServiceBridge, parent: QWidget | None = None) -> None:
        """Make the panel and the plot, with a capillary to design."""
        super().__init__(parent)
        self.setObjectName("page")
        self.bridge = bridge
        self.text: str | None = None
        self.guide = "capillary"
        self.page = "curves"
        self.fields: dict[str, QuantityField] = {}
        self.numbers: dict[str, QPushButton] = {}
        self._rows: dict[str, QWidget] = {}
        self._pending: dict[int, str] = {}
        self._curves: list[tuple[str, np.ndarray]] = []
        self._x = np.zeros(0)
        self._solved_note = ""
        self._pump: float | None = None
        self._zdw: float | None = None
        self._rdw: float | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_DEBOUNCE_MS)
        self._timer.timeout.connect(self.ask)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._make_panel())
        row.addWidget(self._make_right(), 1)
        bridge.replied.connect(self._on_reply)
        self._load_start()

    # ------------------------------------------------------------------ the layout

    def _make_panel(self) -> QWidget:  # noqa: PLR0915 - a layout
        """Make the panel of inputs.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The panel, scrolling when it must.
        """
        inner = QWidget()
        inner.setObjectName("page")
        column = QVBoxLayout(inner)
        column.setContentsMargins(20, 16, 16, 16)
        column.setSpacing(8)
        self.guide_choice = Segments(
            "Guide",
            (("capillary", "Capillary"), ("antiresonant", "Antiresonant")),
            "capillary",
        )
        self.guide_choice.chosen.connect(self._guide_chosen)
        column.addWidget(self.guide_choice)
        form = QFormLayout()
        form.setSpacing(6)
        self.form = form
        for spec in (*_SHARED[:1], *_LATTICE):
            self._add_row(form, spec)
        self.gas = QComboBox()
        self.gas.setObjectName("design-gas")
        self.gas.currentIndexChanged.connect(lambda _index: self._edited())
        form.addRow("Gas", self.gas)
        for spec in _SHARED[1:]:
            self._add_row(form, spec)
        column.addLayout(form)
        column.addSpacing(8)
        column.addWidget(named_label("Sweep", "h3"))
        self.sweep_name = QComboBox()
        self.sweep_name.setObjectName("design-sweep")
        for path, name, _values in _SWEEPS:
            self.sweep_name.addItem(name, path)
        self.sweep_name.currentIndexChanged.connect(self._sweep_chosen)
        self.sweep_text = QLineEdit()
        self.sweep_text.setObjectName("design-sweep-values")
        self.sweep_text.setPlaceholderText("values, as 0.5, 1, 2")
        self.sweep_text.setEnabled(False)
        self.sweep_text.textEdited.connect(lambda _text: self._edited())
        column.addWidget(self.sweep_name)
        column.addWidget(self.sweep_text)
        column.addSpacing(8)
        column.addWidget(named_label("Solve", "h3"))
        solve = QFormLayout()
        solve.setSpacing(6)
        self.zdw_target = QuantityField("m", "nm")
        self.order_target = QuantityField(None)
        self.solve_button = QPushButton("Solve the pressure")
        self.solve_button.setObjectName("design-solve")
        self.solve_button.clicked.connect(self.solve)
        solve.addRow("ZDW at", self.zdw_target)
        solve.addRow("Soliton order", self.order_target)
        solve.addRow("", self.solve_button)
        column.addLayout(solve)
        column.addSpacing(8)
        self.pulse = QCheckBox("With a pulse")
        self.pulse.setObjectName("design-pulse")
        self.pulse.toggled.connect(self._pulse_toggled)
        column.addWidget(self.pulse)
        self.pulse_form = QFormLayout()
        self.pulse_form.setSpacing(6)
        for spec in _PULSE:
            self._add_row(self.pulse_form, spec)
        column.addLayout(self.pulse_form)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName("sheetScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)
        scroll.setFixedWidth(340)
        return scroll

    def _add_row(self, form: QFormLayout, spec: tuple[object, ...]) -> None:
        """Add one number's row to a form.

        Parameters
        ----------
        form : PySide6.QtWidgets.QFormLayout
            The form.
        spec : tuple
            Its path, label, keyword unit and unit shown.
        """
        path, label, keyword, shown = spec[:4]
        field = QuantityField(
            None if keyword is None else str(keyword),
            None if shown is None else str(shown),
            integer=keyword is None,
            live=True,
        )
        field.changed.connect(lambda _value: self._edited())
        field.refused.connect(self._say)
        form.addRow(str(label), field)
        self.fields[str(path)] = field

    def _make_right(self) -> QWidget:  # noqa: PLR0915 - a layout
        """Make the large plot, its pages and the numbers below it.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The side.
        """
        right = QWidget()
        right.setObjectName("page")
        column = QVBoxLayout(right)
        column.setContentsMargins(20, 16, 20, 12)
        column.setSpacing(8)
        bar = QHBoxLayout()
        self.pages = Segments("", _PAGES, "curves")
        self.pages.chosen.connect(self.show_page)
        self.pages.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        bar.addWidget(self.pages)
        self.quantity = QComboBox()
        self.quantity.setObjectName("design-quantity")
        for key, name, _axis, _unit in _CURVES:
            self.quantity.addItem(name, key)
        self.quantity.currentIndexChanged.connect(lambda _index: self.ask_plot())
        bar.addWidget(self.quantity)
        bar.addStretch(1)
        self.export = QToolButton()
        self.export.setObjectName("plain")
        self.export.setText("Export CSV")
        self.export.clicked.connect(self.choose_export)
        bar.addWidget(self.export)
        column.addLayout(bar)
        self.note = named_label("", "pageNote")
        self.note.setWordWrap(True)
        column.addWidget(self.note)
        self.plot = LinePlot(xlabel="Wavelength (nm)", ylabel="", left=None)
        self.plot.picked.connect(self._picked)
        self.zero = pg.InfiniteLine(angle=0, pen=pg.mkPen((150, 158, 170)))
        self.zero.hide()
        self.plot.addItem(self.zero, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.marks = PlotMarks(self.plot)
        self.design_dot = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(ORANGE), pen=pg.mkPen("w")
        )
        self.design_dot.setZValue(30)
        self.plot.addItem(self.design_dot, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        column.addWidget(self.plot, 1)
        tiles = QGridLayout()
        tiles.setHorizontalSpacing(24)
        tiles.setVerticalSpacing(8)
        self._tiles: dict[str, QWidget] = {}
        for index, number in enumerate((*_NUMBERS, *_PULSE_NUMBERS)):
            tile = QWidget()
            tile.setObjectName("page")
            stack = QVBoxLayout(tile)
            stack.setContentsMargins(0, 0, 0, 0)
            stack.setSpacing(0)
            stack.addWidget(named_label(number.label, "derivedLabel"))
            value = named_label("…", "derivedValue")
            stack.addWidget(value)
            self.numbers[number.key] = value  # pyright: ignore[reportArgumentType]
            self._tiles[number.key] = tile
            tiles.addWidget(tile, index // 6, index % 6)
        column.addLayout(tiles)
        return right

    # ------------------------------------------------------------------ the inputs

    def _load_start(self) -> None:
        """Put the guide's starting design in the boxes."""
        start: Mapping[str, object] = _STARTS[self.guide]
        for path, field in self.fields.items():
            if path in start:
                field.blockSignals(True)
                field.set_value(start[path])
                field.blockSignals(False)
        for spec in _LATTICE:
            self._show_row(str(spec[0]), self.guide == "antiresonant")
        for spec in _PULSE:
            self._show_row(str(spec[0]), self.pulse.isChecked())
        self._show_gas(str(start["gas"]))
        capillary = self.guide == "capillary"
        self.solve_button.setEnabled(capillary)
        self.zdw_target.setEnabled(capillary)
        self.order_target.setEnabled(capillary)
        self.solve_button.setToolTip(
            ""
            if capillary
            else "A fibre has many zeros, so this solve is for a capillary."
        )
        self._show_tiles()

    def _show_row(self, path: str, shown: bool) -> None:
        """Show or hide a row of a form.

        Parameters
        ----------
        path : str
            The row's path.
        shown : bool
            Whether it shows.
        """
        field = self.fields[path]
        form = self.pulse_form if path in ("fwhm", "energy") else self.form
        form.setRowVisible(form.getWidgetPosition(field)[0], shown)  # pyright: ignore[reportIndexIssue]

    def _show_gas(self, gas: str) -> None:
        """Select a gas.

        Parameters
        ----------
        gas : str
            Its name.
        """
        self.gas.blockSignals(True)
        self.gas.setCurrentIndex(self.gas.findData(gas))
        self.gas.blockSignals(False)

    def _show_tiles(self) -> None:
        """Show the numbers the design has."""
        for number in _PULSE_NUMBERS:
            self._tiles[number.key].setVisible(self.pulse.isChecked())

    def set_gases(self, gases: Sequence[str]) -> None:
        """Take the gases solprop knows.

        Parameters
        ----------
        gases : Sequence of str
            Their names.
        """
        self.gas.blockSignals(True)
        self.gas.clear()
        for gas in gases:
            self.gas.addItem(GAS_TITLES.get(gas, gas), gas)
        self.gas.blockSignals(False)
        self._show_gas(str(_STARTS[self.guide]["gas"]))  # pyright: ignore[reportIndexIssue]

    def _guide_chosen(self, guide: object) -> None:
        """Design another guide, from its starting design.

        Parameters
        ----------
        guide : object
            The guide.
        """
        self.guide = str(guide)
        self._load_start()
        if self.guide != "capillary" and self.page == "map":
            self.pages.set_current("curves")
            self.page = "curves"
        self._edited()

    def _sweep_chosen(self, _index: int) -> None:
        """Offer values for what is swept.

        Parameters
        ----------
        _index : int
            The choice, which is read from the box.
        """
        chosen = self.sweep_name.currentIndex()
        self.sweep_text.setEnabled(chosen > 0)
        self.sweep_text.setText(_SWEEPS[chosen][2])
        self._edited()

    def _pulse_toggled(self, on: bool) -> None:
        """Show the pulse's rows and numbers.

        Parameters
        ----------
        on : bool
            Whether the design gives a pulse.
        """
        for spec in _PULSE:
            self._show_row(str(spec[0]), on)
        self._show_tiles()
        self._edited()

    def _say(self, text: str) -> None:
        """Say something above the plot.

        Parameters
        ----------
        text : str
            What.
        """
        self.note.setText(text)

    def _edited(self) -> None:
        """Ask again once the typing rests."""
        self._solved_note = ""
        self._timer.start()

    def design(self) -> dict[str, object]:
        """Return the design as the kernel's requests take it.

        Returns
        -------
        dict
            ``guide``, ``values`` and ``pulse``.
        """
        values: dict[str, object] = {"gas": self.gas.currentData() or "He"}
        pulse = self.pulse.isChecked()
        for path, field in self.fields.items():
            if path.startswith("lattice.") and self.guide != "antiresonant":
                continue
            if path in ("fwhm", "energy") and not pulse:
                continue
            value = field.read()
            if value is not None:
                values[path] = value
        return {"guide": self.guide, "values": values, "pulse": pulse}

    def _sweep(self) -> dict[str, object] | None:
        """Return the sweep, if one is asked for.

        Returns
        -------
        dict or None
            ``name`` and ``values``.
        """
        name = self.sweep_name.currentData()
        if name is None:
            return None
        return {
            "name": name,
            "values": design_sweep(str(name), sweep_values(self.sweep_text.text())),
        }

    # ------------------------------------------------------------------ the kernel

    def _request(self, kind: str, payload: Mapping[str, object], key: str) -> None:
        """Ask the kernel something, latest wins.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping
            Its arguments.
        key : str
            What the reply is for.
        """
        self._pending[self.bridge.request(kind, payload, key=f"design-{key}")] = key

    def ask(self) -> None:
        """Ask for the numbers and the page shown."""
        if not self.gas.count():
            self._timer.start()
            return
        self._request("waveguide_numbers", self.design(), "numbers")
        self.ask_plot()

    def ask_plot(self) -> None:
        """Ask for what the page shown draws."""
        self.quantity.setVisible(self.page == "curves")
        if self.page == "map":
            self._ask_map()
            return
        key = (
            str(self.quantity.currentData())
            if self.page == "curves"
            else _FIXED[self.page][0]
        )
        payload = {**self.design(), "quantity": key}
        sweep = self._sweep()
        if sweep is not None:
            payload["sweep"] = sweep
        self._request("waveguide_curves", payload, "curves")

    def _ask_map(self) -> None:
        """Ask for the chart of pressure against radius, for each zero."""
        if self.text is None:
            return
        radius = self.fields["radius"].read()
        pump = self.fields["wavelength"].read()
        centre = _in_metres(radius)
        limit = _in_metres(pump)
        self._say("Solving the pressure for each radius…")
        self._request(
            "zdw_chart",
            {
                "text": self.text,
                "radii": [
                    float(r)
                    for r in np.geomspace(centre / 2.0, centre * 2.0, _MAP_RADII)
                ],
                "wavelengths": [
                    z * 1e-9 for z in _MAP_ZEROS if z * 1e-9 < 0.95 * limit
                ],
            },
            "map",
        )

    def solve(self) -> None:
        """Ask for the pressure, and the energy with an order, that give the targets."""
        zero = self.zdw_target.read()
        if zero is None:
            self._say("Give the zero-dispersion wavelength to solve for.")
            return
        order = self.order_target.read()
        payload = {**self.design(), "zdw": _in_metres(zero)}
        if order is not None:
            payload["soliton_order"] = order
        self._request("waveguide_solve", payload, "solve")

    def _on_reply(self, reply: Reply) -> None:
        """Take a reply, if it is the stage's.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply.
        """
        purpose = self._pending.pop(reply.request_id, None)
        if purpose is None:
            return
        if not reply.ok:
            self._say(str(reply.error))
        elif purpose == "numbers":
            self._show_numbers(reply.value)  # pyright: ignore[reportArgumentType]
        elif purpose == "curves":
            self._show_curves(reply.value)  # pyright: ignore[reportArgumentType]
        elif purpose == "map":
            self._show_map(reply.value)  # pyright: ignore[reportArgumentType]
        else:
            self._solved(reply.value)  # pyright: ignore[reportArgumentType]

    # ------------------------------------------------------------------ the replies

    def _show_numbers(self, value: Mapping[str, object]) -> None:
        """Show the numbers at the pump.

        Parameters
        ----------
        value : Mapping
            The kernel's ``waveguide_numbers`` reply.
        """
        self.text = str(value["resolved"])
        shown = {
            **value,
            "beta2_fs": float(value["beta2"]) * 1e30,  # pyright: ignore[reportArgumentType]
            "beta3_fs": float(value["beta3"]) * 1e45,  # pyright: ignore[reportArgumentType]
            "aeff_um": float(value["aeff"]) * 1e12,  # pyright: ignore[reportArgumentType]
            "gamma_km": float(value["gamma"]) * 1e3,  # pyright: ignore[reportArgumentType]
        }
        for number in (*_NUMBERS, *_PULSE_NUMBERS):
            if number.key in shown:
                held = shown[number.key]
                reading = None if held is None else float(held)  # pyright: ignore[reportArgumentType]
                if reading is not None and np.isnan(reading):
                    reading = None
                self.numbers[number.key].setText(derived_text(number, reading))
        zero = shown.get("zdw")
        self._zdw = None if zero is None or np.isnan(float(zero)) else float(zero)  # pyright: ignore[reportArgumentType]
        rdw = shown.get("rdw")
        self._rdw = None if rdw is None else float(rdw)  # pyright: ignore[reportArgumentType]
        self._mark()
        if self.page == "map" and not self._pending_map():
            self._ask_map()

    def _pending_map(self) -> bool:
        """Return whether a map is being solved.

        Returns
        -------
        bool
            Whether one was asked for and has not come.
        """
        return "map" in self._pending.values()

    def _mark(self) -> None:
        """Mark the pump, the zero and the wave on a plot against wavelength."""
        if self.page == "map":
            self.marks.show(())
            return
        self.design_dot.setData([], [])
        self.marks.show(
            (
                ("Pump", self._pump),
                ("ZDW", None if self._zdw is None else self._zdw * 1e9),
                ("Dispersive wave", None if self._rdw is None else self._rdw * 1e9),
            )
        )

    def _axis(self, key: str) -> tuple[str, str]:
        """Return a quantity's axis title and unit.

        Parameters
        ----------
        key : str
            The quantity.

        Returns
        -------
        tuple of str
            The title and the unit.
        """
        for fixed, axis, unit in _FIXED.values():
            if fixed == key:
                return axis, unit
        return next((axis, unit) for k, _n, axis, unit in _CURVES if k == key)

    def _show_curves(self, value: Mapping[str, object]) -> None:
        """Draw a design's curves.

        Parameters
        ----------
        value : Mapping
            The kernel's ``waveguide_curves`` reply.
        """
        key = str(value["key"])
        entries: Sequence[Mapping[str, object]] = value["curves"]  # pyright: ignore[reportAssignmentType]
        errors = [str(e["error"]) for e in entries if "error" in e]
        drawn = [e for e in entries if "y" in e]
        self._say("; ".join(errors) or self._solved_note or self._about(len(drawn)))
        x = np.asarray(value["x"], dtype=float)
        self._x = x
        self._curves = [
            (str(e["value"] or ""), np.asarray(e["y"], dtype=float)) for e in drawn
        ]
        title, unit = self._axis(key)
        self.plot.set_xlabel("Wavelength (nm)")
        self.plot.set_ylabel(title)
        self.plot.unit = unit
        self.plot.set_names([name for name, _y in self._curves] or [""])
        self.plot.set_log(key == "loss")
        self.plot.set_data(x, [y for _name, y in self._curves])
        pump = value.get("pump")
        self._pump = None if pump is None or np.isnan(float(pump)) else float(pump)  # pyright: ignore[reportArgumentType]
        self.zero.setVisible(value.get("limit") is not None)
        limit = value.get("limit")
        if x.size and limit is not None:
            top = float(limit)  # pyright: ignore[reportArgumentType]
            self.plot.set_view((float(x[0]), float(x[-1])), (-top, top))
        elif x.size:
            self.plot.fit()
        self._mark()

    def _about(self, count: int) -> str:
        """Say what the plot shows.

        Parameters
        ----------
        count : int
            How many curves.

        Returns
        -------
        str
            The line above the plot.
        """
        if self.page == "phase":
            return (
                "Phase mismatch to a soliton at the pump: a zero is a matched wave."
                if self.pulse.isChecked()
                else "Linear phase mismatch; give a pulse for its soliton's."
            )
        if count > 1:
            return "Click a curve to take its value."
        return ""

    def _show_map(self, value: Mapping[str, object]) -> None:
        """Draw the chart: the pressure against the radius, one curve to a zero.

        Parameters
        ----------
        value : Mapping
            The kernel's ``zdw_chart`` reply.
        """
        radii = np.asarray(value["radius"], dtype=float) * 1e6
        curves: Sequence[Mapping[str, object]] = value["curves"]  # pyright: ignore[reportAssignmentType]
        self._x = radii
        self._curves = [
            (
                f"ZDW {float(curve['zdw']) * 1e9:.0f} nm",  # pyright: ignore[reportArgumentType]
                np.array(
                    [np.nan if p is None else float(p) for p in curve["pressure"]],  # pyright: ignore[reportGeneralTypeIssues]
                    dtype=float,
                ),
            )
            for curve in curves
        ]
        self.plot.set_xlabel("Core radius (µm)")
        self.plot.set_ylabel("Pressure (bar)")
        self.plot.unit = "bar"
        self.plot.set_names([name for name, _y in self._curves] or [""])
        self.plot.set_log(True)
        self.plot.set_data(radii, [y for _name, y in self._curves])
        self.plot.fit()
        self.zero.hide()
        self._mark()
        setup: Mapping[str, object] = value["setup"]  # pyright: ignore[reportAssignmentType]
        pressure = setup.get("pressure")
        if pressure is not None:
            self.design_dot.setData(
                [float(setup["radius"]) * 1e6],  # pyright: ignore[reportArgumentType]
                [np.log10(float(pressure))],  # pyright: ignore[reportArgumentType]
            )
        self._say(
            str(value.get("error") or "")
            or "Click a point to take its radius and pressure."
        )

    def _solved(self, value: Mapping[str, object]) -> None:
        """Take a solved design.

        Parameters
        ----------
        value : Mapping
            The kernel's ``waveguide_solve`` reply.
        """
        pressure = float(value["pressure"])  # pyright: ignore[reportArgumentType]
        self.fields["pressure"].set_value(f"{pressure:.6g} bar")
        said = f"Solved: {pressure:.4g} bar"
        energy = value.get("energy")
        if energy is not None:
            if not self.pulse.isChecked():
                self.pulse.setChecked(True)
            self.fields["energy"].set_value(f"{float(energy) * 1e6:.6g} uJ")  # pyright: ignore[reportArgumentType]
            said += f" and {float(energy) * 1e6:.4g} µJ"  # pyright: ignore[reportArgumentType]
        self._timer.stop()
        self.ask()
        self._solved_note = f"{said}."
        self._say(self._solved_note)

    # ------------------------------------------------------------------ picking

    def _picked(self, x: float, y: float) -> None:
        """Take a point of the plot as the design.

        On the map, the radius and pressure there; on a sweep's curves, the value of
        the nearest curve.

        Parameters
        ----------
        x : float
            The position along the plot.
        y : float
            The position up it, as drawn.
        """
        if self.page == "map":
            self.fields["radius"].set_value(f"{x:.4g} um")
            self.fields["pressure"].set_value(
                f"{10.0**y if self.plot.log else y:.4g} bar"
            )
            self._edited()
            return
        name = self.sweep_name.currentData()
        if name is None or not self._curves or not self._x.size:
            return
        at = int(np.argmin(np.abs(self._x - x)))
        heights = np.array(
            [
                np.log10(max(c[at], 1e-300)) if self.plot.log else c[at]
                for _n, c in self._curves
            ]
        )
        chosen = self._curves[int(np.nanargmin(np.abs(heights - y)))][0]
        if name == "gas":
            self._show_gas(chosen)
        else:
            self.fields[str(name)].set_value(chosen)
        self._edited()
        self._solved_note = f"Took {chosen}."

    def show_page(self, page: object) -> None:
        """Show a page of the large plot.

        Parameters
        ----------
        page : object
            The page.
        """
        if str(page) == "map" and self.guide != "capillary":
            self.pages.set_current(self.page)
            self._say("The map solves a capillary's pressure, and this is a fibre.")
            return
        self.page = str(page)
        self.ask_plot()

    # ------------------------------------------------------------------ export

    def choose_export(self) -> None:  # pragma: no cover - a modal dialog
        """Ask where to write the curves, and write them."""
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export curves", "waveguide.csv", "CSV (*.csv)"
        )
        if chosen:
            self.write_csv(chosen)

    def write_csv(self, path: str | Path) -> None:
        """Write what the plot shows, a column to a curve.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        """
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    self.plot.item.getAxis("bottom").labelText,
                    *[
                        name or self.plot.item.getAxis("left").labelText
                        for name, _y in self._curves
                    ],
                ]
            )
            writer.writerows(
                np.column_stack((self._x, *[y for _name, y in self._curves])).tolist()
            )

    def showEvent(self, event: object) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Ask for what to show as the stage comes up.

        Parameters
        ----------
        event : object
            The event.
        """
        super().showEvent(event)  # pyright: ignore[reportArgumentType]
        if self.text is None:
            self._timer.start()


def _in_metres(value: object) -> float:
    """Return a box's value in metres.

    Parameters
    ----------
    value : object
        A quantity string, or a number of metres.

    Returns
    -------
    float
        The length, m.
    """
    return float(parse_quantity(value, Unit("m"), path="design"))
