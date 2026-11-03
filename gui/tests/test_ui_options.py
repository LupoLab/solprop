# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plot options panel, and the window keeping one set of options everywhere."""

import json
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.ui.main_window import MainWindow, app_settings
from solprop_gui.ui.plot_options import PlotOptionsPanel
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture


def type_into(field, text: str) -> None:
    field.setText(text)
    field.editingFinished.emit()


def test_the_panel_reads_limits_scale_and_band(qtbot: QtBot, tmp_path: Path):
    panel = PlotOptionsPanel("t")
    qtbot.addWidget(panel)
    seen: list[PlotOptions] = []
    panel.changed.connect(seen.append)
    type_into(panel.spectral.low, "1200")
    type_into(panel.spectral.high, "400")  # either order
    assert seen[-1].spectral_limits == (pytest.approx(4e-7), pytest.approx(1.2e-6))
    type_into(panel.time.low, "-100")
    assert seen[-1].time_limits == (pytest.approx(-1e-13), None)
    type_into(panel.time.high, "soon")  # not a number: marked, nothing sent
    assert "not a number" in panel.note.text() and len(seen) == 3
    assert panel.time.high.styleSheet()
    type_into(panel.time.high, "")
    panel.scale.setCurrentText("linear")
    assert seen[-1].scale == "linear" and not panel.floor.isEnabled()
    panel.band_on.setChecked(True)  # no band typed yet
    assert "two positive wavelengths" in panel.note.text()
    type_into(panel.band.low, "700")
    type_into(panel.band.high, "900")
    assert seen[-1].bandpass == (pytest.approx(7e-7), pytest.approx(9e-7))
    count = len(seen)
    type_into(panel.band.high, "900")  # unchanged: not sent again
    assert len(seen) == count
    # A new axis clears the limits typed in the old one's unit.
    panel.axis.setCurrentIndex(panel.axis.findData("f"))
    assert seen[-1].axis == "f" and seen[-1].spectral_limits == (None, None)
    assert panel.spectral.unit.text() == "THz"
    shot(panel, tmp_path, "plot-options")
    panel.reset.click()
    assert seen[-1] == PlotOptions() and not panel.band_on.isChecked()
    assert panel.options == PlotOptions()


def test_the_window_keeps_one_set_of_options(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "runs")
    qtbot.addWidget(window)
    options = PlotOptions(axis="f", floor=-40.0, bandpass=(7e-7, 9e-7))
    window.results.plot_options.changed.emit(options)  # set on the Results step
    assert window.plot_options.options == options  # shown on Propagate too
    assert window.run_panel.display().axis == "f"
    assert window.run_panel.display().bandpass == (7e-7, 9e-7)
    assert window.run_panel.view.options == options
    assert window.results.overview.options == options
    saved = json.loads(str(app_settings().value("plot_options")))
    assert saved["axis"] == "f" and saved["floor"] == -40.0
    assert app_settings().fileName().startswith(str(tmp_path))  # not the user's
    window.close()
    again = MainWindow(workspace=tmp_path / "runs")
    qtbot.addWidget(again)
    assert again.plot_options.options == options  # kept for the next session
    app_settings().setValue("plot_options", "{")
    third = MainWindow(workspace=tmp_path / "runs")
    qtbot.addWidget(third)
    assert third.plot_options.options == PlotOptions()  # unreadable: the defaults
