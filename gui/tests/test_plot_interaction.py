# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a plot does under the mouse: zoom, readouts, widths, peaks and saving."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from solprop_gui.views.page_plots import LinePlot, MapPlot


def two_pulses(qtbot: QtBot) -> LinePlot:
    plot = LinePlot(xlabel="Time (fs)", ylabel="Power (W)", names=["a", "b"], unit="W")
    qtbot.addWidget(plot)
    plot.resize(400, 300)
    t = np.linspace(-100.0, 100.0, 2001)
    plot.set_data(t, [np.exp(-4 * np.log(2) * (t / 20.0) ** 2), 0.2 + 0 * t])
    plot.set_view((-100.0, 100.0), (0.0, 1.1))
    return plot


def test_the_readout_is_the_nearest_curve_s(qtbot: QtBot) -> None:
    plot = two_pulses(qtbot)
    assert plot.read(0.0, 0.95).startswith("a: 0, 1 W")
    assert plot.read(0.0, 0.25).startswith("b: 0, 0.2 W")
    plot.set_log(True)
    plot.set_data(plot.xs, plot.ys)
    assert plot.read(0.0, np.log10(0.21)).startswith("b:")
    empty = LinePlot(xlabel="", ylabel="")
    qtbot.addWidget(empty)
    assert empty.read(0.0, 0.0) == ""


def test_a_width_at_half_maximum_is_marked_and_cleared(qtbot: QtBot) -> None:
    plot = two_pulses(qtbot)
    assert plot.measure_width(5.0, 0.9) == "FWHM 20 fs"
    assert len(plot.measures) == 2
    # A flat curve has no half maximum to find.
    assert plot.measure_width(0.0, 0.2).startswith("No half maximum")
    labels = [text for text, _action in plot.menu_actions(QPointF(0.0, 0.9))]
    assert labels == [
        "Width at half maximum here",
        "Copy the data as CSV",
        "Clear the widths",
    ]
    plot.clear_measures()
    assert plot.measures == []
    # New data take the old marks away.
    plot.measure_width(5.0, 0.9)
    plot.set_data(plot.xs, plot.ys)
    assert plot.measures == []


def test_a_width_in_decibels_is_three_below_the_peak(qtbot: QtBot) -> None:
    plot = LinePlot(xlabel="Wavelength (nm)", ylabel="dB", unit="dB")
    qtbot.addWidget(plot)
    x = np.linspace(700.0, 900.0, 2001)
    gaussian = np.exp(-4 * np.log(2) * ((x - 800.0) / 10.0) ** 2)
    plot.set_data(x, [10 * np.log10(gaussian)])
    assert plot.measure_width(800.0, 0.0) == "FWHM 10 nm"


def test_a_plot_s_data_and_image_are_copied_and_saved(
    qtbot: QtBot, tmp_path: Path
) -> None:
    plot = two_pulses(qtbot)
    plot.copy_data()
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0] == "x,a,b"
    assert len(lines) == 2002
    plot.copy_image()
    assert not QApplication.clipboard().pixmap().isNull()
    plot.save(str(tmp_path / "plot.png"))
    plot.save(str(tmp_path / "plot.svg"))
    assert (tmp_path / "plot.png").stat().st_size > 0
    assert "<svg" in (tmp_path / "plot.svg").read_text()[:400]


def test_a_zoom_is_kept_through_redraws_until_reset(qtbot: QtBot) -> None:
    plot = two_pulses(qtbot)
    plot.box.scaleBy((0.5, 0.5))
    plot.box.sigRangeChangedManually.emit(plot.box.state["mouseEnabled"])
    assert plot.zoomed
    zoomed = plot.box.viewRange()[0]
    plot.set_view((-100.0, 100.0), (0.0, 1.1))
    assert plot.box.viewRange()[0] == pytest.approx(zoomed)
    menu = plot.show_menu(QPointF(0.0, 0.5), QPoint(0, 0))
    reset = next(a for a in menu.actions() if a.text() == "Reset the zoom")
    assert reset.isEnabled()
    menu.close()
    qtbot.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton)
    assert not plot.zoomed
    assert plot.box.viewRange()[0] == pytest.approx((-100.0, 100.0))


def test_a_right_click_opens_the_menu_and_a_map_goes_to_its_peak(
    qtbot: QtBot,
) -> None:
    plot = MapPlot(xlabel="x", ylabel="z", colours="viridis", unit="")
    qtbot.addWidget(plot)
    plot.resize(300, 300)
    plot.show()
    assert plot.menu_actions(QPointF(0, 0)) == []
    data = np.zeros((10, 20))
    data[7, 4] = 1.0
    plot.show_map(data, (0.0, 0.0, 20.0, 10.0), ((0.0, 20.0), (0.0, 10.0)), (0, 1))
    plot.set_along("y")
    assert plot.peak() == pytest.approx((4.5, 7.5))
    [(text, _action)] = plot.menu_actions(QPointF(0, 0))
    assert text == "Go to the peak"
    with qtbot.waitSignal(plot.plane_at) as moved:
        plot.go_to_peak()
    assert moved.args == [pytest.approx(7.5)]
    plot.set_along("x")
    with qtbot.waitSignal(plot.plane_at) as moved:
        plot.go_to_peak()
    assert moved.args == [pytest.approx(4.5)]
    plot.data = np.full((2, 2), np.nan)
    assert plot.peak() is None
    plot.go_to_peak()
    qtbot.mouseClick(plot.viewport(), Qt.MouseButton.RightButton, pos=QPoint(150, 150))
