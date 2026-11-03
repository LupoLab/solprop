# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The second review's details: a failed run's dialog, a route's words, cross-hairs."""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from solprop_gui.registry.pages import PAGES
from solprop_gui.service.messages import Event
from solprop_gui.service.states import ExecutionState
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.pages import ParameterPage
from solprop_gui.views.plotting import HoverLayout, curve_lookup, figure_spec
from test_ui import private_settings  # noqa: F401 - a fixture


def test_a_failed_run_says_why_in_a_dialog(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "runs")
    qtbot.addWidget(window)
    panel = window.run_panel
    reported: list[tuple[str, str, str]] = []
    panel.report = lambda *args: reported.append(args)
    panel.run_id = "r1"
    panel.folder = tmp_path / "r1"
    panel.state = ExecutionState.RUNNING
    message = (
        "at z = 0.36 m the field puts 1e-4 of its energy in the guards: widen trange"
    )
    panel.on_event(Event("r1", "failed", {"message": message, "detail": "Traceback"}))
    assert reported == [("The run r1 failed", message, "Traceback")]
    assert "failed: at z = 0.36 m" in panel.state_label.text()
    panel._report("The run failed", message, "Traceback")  # the dialog itself
    box = panel.findChild(QMessageBox, "run-failed")
    assert box is not None and box.text() == message
    assert box.detailedText() == "Traceback"
    box.close()
    window.close()


def test_a_free_space_route_words_the_geometry_step(qtbot: QtBot):
    geometry = next(page for page in PAGES if page.key == "geometry")
    page = ParameterPage(geometry)
    qtbot.addWidget(page)
    page.set_route("free")
    assert "transverse grid" in page.help.text()
    assert page.sections["geometry"].header.text() == "Beam and grid"
    page.set_route("capillary")
    assert page.help.text() == geometry.help
    assert page.sections["geometry"].header.text() == "Waveguide"


def test_a_readout_has_cross_hairs_left_out_of_a_figure(qtbot: QtBot):
    layout = HoverLayout()
    qtbot.addWidget(layout)
    layout.resize(400, 300)
    layout.show()
    plot = layout.add_plot(0, 0, "x")
    curve = plot.plot(np.arange(5.0), np.arange(5.0) ** 2)
    layout.add_readout(plot, curve_lookup(curve, "{0} {1}"))
    (_, _, readout) = layout.targets[0]
    readout.show(2.0, 4.0, "2 4")
    assert [line.isVisible() for line in readout.cross] == [True, True]
    assert [line.value() for line in readout.cross] == [2.0, 4.0]
    plot.addItem(pg.InfiniteLine(pos=1.0, angle=90))  # a marker of the view's own
    lines: list[dict[str, object]] = figure_spec(plot)["lines"]  # pyright: ignore[reportAssignmentType]
    assert len(lines) == 1 and lines[0]["x"] == 1.0  # the cross-hairs left out
    readout.hide()
    assert not any(line.isVisible() for line in readout.cross)
