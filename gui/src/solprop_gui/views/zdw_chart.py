# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The design chart: the pressure that puts the zero-dispersion point where wanted.

The zero-dispersion wavelength of a gas-filled capillary over the plane of core
radius and pressure, drawn as its contours: one curve per wavelength, the pressure
``tools.pressure_zdw`` solves for against the radius, with the setup's own radius and
pressure marked. Read across at a radius for the pressure a ZDW needs; the curves
fall as the radius grows, since a wider core has less waveguide dispersion to
balance.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from solprop.recipe.errors import RecipeError
from solprop.recipe.quantity import Unit, parse_quantity

from solprop_gui.registry.explore import sweep_values
from solprop_gui.ui.quantity import RangeField
from solprop_gui.views.plotting import HoverLayout, choose_figure, colormap, figure_spec

__all__ = ["ZdwChartView"]

_RADII = ["25 um", "250 um"]
"""The radii charted at first."""

_WAVELENGTHS = "300, 400, 500, 600, 800 nm"
"""The zero-dispersion wavelengths charted at first."""


class ZdwChartView(QWidget):
    """Pressure against radius, one curve per zero-dispersion wavelength.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    chart_wanted : Signal
        Emitted with the request's ``radii`` and ``wavelengths`` when the user asks
        for the chart.
    figure_wanted : Signal
        Emitted with a figure's ``path`` and ``spec`` when the user exports one.
    """

    chart_wanted = Signal(object)
    figure_wanted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the controls and the empty plot."""
        super().__init__(parent)
        self.curves: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.radii = RangeField("m", "um")
        self.radii.set_value(_RADII)
        self.count = QSpinBox()
        self.count.setObjectName("chart-count")
        self.count.setRange(3, 60)
        self.count.setValue(16)
        self.count.setSuffix(" radii")
        self.wavelengths = QLineEdit(_WAVELENGTHS)
        self.wavelengths.setObjectName("chart-wavelengths")
        self.wavelengths.setPlaceholderText(
            "zero-dispersion wavelengths, e.g. 400, 600 nm"
        )
        self.wavelengths.returnPressed.connect(self.ask)
        self.figure_button = QToolButton()
        self.figure_button.setText("Export figure…")
        self.figure_button.clicked.connect(self.choose_figure)
        self.draw_button = QPushButton("Chart")
        self.draw_button.setObjectName("chart-draw")
        self.draw_button.clicked.connect(self.ask)
        self.status = QLabel(
            "The pressure that puts the zero-dispersion wavelength at each value, for "
            "the setup's gas: press Chart."
        )
        self.status.setObjectName("chart-status")
        self.status.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Radius"))
        controls.addWidget(self.radii)
        controls.addWidget(self.count)
        controls.addSpacing(12)
        controls.addWidget(QLabel("ZDW"))
        controls.addWidget(self.wavelengths, 1)
        controls.addWidget(self.draw_button)
        controls.addWidget(self.figure_button)

        self.plots = HoverLayout()
        self.plot = self.plots.add_plot(0, 0, "")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLogMode(x=False, y=True)
        self.plot.setLabel("bottom", "Core radius (µm)")
        self.plot.setLabel("left", "Pressure (bar)")
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plots.add_readout(self.plot, self.lookup, dot=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(self.plots, 1)

    def choose_figure(self) -> None:  # pragma: no cover - a modal dialog
        """Ask where to write the figure, and ask for it."""
        path = choose_figure(self, "zdw-chart.pdf")
        if path:
            self.export_figure(path)

    def export_figure(self, path: str) -> None:
        """Ask for the chart as a figure.

        Parameters
        ----------
        path : str
            The file.
        """
        self.figure_wanted.emit({"path": path, "spec": figure_spec(self.plot)})

    def request(self) -> dict[str, object] | None:
        """Return the chart's request, or say why it cannot be made.

        Returns
        -------
        dict or None
            ``radii`` (m) and ``wavelengths`` (as written); ``None`` when a field
            does not read.
        """
        try:
            ends = self.radii.read()
            if not isinstance(ends, Sequence) or len(ends) != 2:
                self.status.setText("Give the radii's range.")
                return None
            low, high = (parse_quantity(end, Unit("m")) for end in ends)
            wavelengths = sweep_values(self.wavelengths.text())
            for wavelength in wavelengths:
                parse_quantity(wavelength, Unit("m"))
        except (RecipeError, ValueError) as error:
            self.status.setText(str(error))
            return None
        if not wavelengths:
            self.status.setText("Give the zero-dispersion wavelengths to chart.")
            return None
        radii = np.linspace(low, high, self.count.value())
        return {"radii": [float(r) for r in radii], "wavelengths": wavelengths}

    def ask(self) -> None:
        """Ask for the chart."""
        request = self.request()
        if request is None:
            return
        points = len(request["radii"]) * len(request["wavelengths"])  # pyright: ignore[reportArgumentType]
        self.status.setText(f"Solving {points} pressures…")
        self.chart_wanted.emit(request)

    def show_chart(self, chart: Mapping[str, object]) -> None:
        """Draw the kernel's chart.

        Parameters
        ----------
        chart : Mapping
            The ``zdw_chart`` reply.
        """
        self.plot.clear()
        self.legend.clear()
        self.curves = []
        radius = np.asarray(chart["radius"], dtype=float) * 1e6
        curves: Sequence[Mapping[str, object]] = chart["curves"]  # pyright: ignore[reportAssignmentType]
        ramp = colormap("viridis")
        for index, curve in enumerate(curves):
            pressure = np.array(
                [np.nan if p is None else p for p in curve["pressure"]],  # pyright: ignore[reportGeneralTypeIssues]
                dtype=float,
            )
            zdw = float(curve["zdw"]) * 1e9  # pyright: ignore[reportArgumentType]
            colour = ramp.mapToQColor(0.85 * index / max(len(curves) - 1, 1))
            self.plot.plot(
                radius,
                pressure,
                pen=pg.mkPen(colour, width=1.6),
                name=f"ZDW {zdw:.4g} nm",
                connect="finite",
            )
            self.curves.append((zdw, radius, pressure))
        setup: Mapping[str, object] = chart.get("setup") or {}  # pyright: ignore[reportAssignmentType]
        if setup.get("pressure") and setup.get("radius"):
            point = pg.ScatterPlotItem(
                [float(setup["radius"]) * 1e6],  # pyright: ignore[reportArgumentType]
                [float(np.log10(float(setup["pressure"])))],  # pyright: ignore[reportArgumentType]
                symbol="+",
                size=14,
                pen=pg.mkPen("#c62828", width=2),
                name="setup",
            )
            self.plot.addItem(point)
        missing = chart.get("error")
        self.status.setText(
            "From tools.pressure_zdw for the setup's fill."
            + (f" Gaps: {missing}" if missing else "")
        )

    def show_error(self, message: str) -> None:
        """Say why there is no chart.

        Parameters
        ----------
        message : str
            The reason.
        """
        self.plot.clear()
        self.curves = []
        self.status.setText(message)

    def lookup(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Read the curve nearest the mouse, at the radius nearest it.

        Parameters
        ----------
        x, y : float
            The mouse: radius in µm and log10 of the pressure in bar.

        Returns
        -------
        tuple or None
            The point (in the plot's coordinates) and its label; ``None`` with
            nothing drawn.
        """
        best: tuple[float, float, str] | None = None
        distance = np.inf
        for zdw, radius, pressure in self.curves:
            index = int(np.argmin(np.abs(radius - x)))
            value = pressure[index]
            if not np.isfinite(value) or value <= 0:
                continue
            gap = abs(np.log10(value) - y)
            if gap < distance:
                distance = gap
                label = f"ZDW {zdw:.4g} nm: {value:.4g} bar at {radius[index]:.4g} µm"
                best = (float(radius[index]), float(np.log10(value)), label)
        return best
