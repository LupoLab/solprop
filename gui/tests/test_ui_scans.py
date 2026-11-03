# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scan editor, the scan browser and a scan run from the window, offscreen."""

from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from solprop_gui.model.recipe import RecipeModel
from solprop_gui.service.messages import Reply
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.scan_editor import ScanEditor
from solprop_gui.views.scan_browser import ScanView
from test_design import SCAN
from test_ui import WAIT_MS, leave, private_settings, shot  # noqa: F401 - a fixture


def test_the_scan_editor_writes_the_scan_table(qtbot: QtBot):
    editor = ScanEditor()
    qtbot.addWidget(editor)
    editor.set_paths(["pressure", "energy"])
    seen = []
    editor.changed.connect(seen.append)
    editor.enabled.setChecked(True)
    assert seen[-1] == {"name": "scan"} and len(editor.variables) == 1
    variable = editor.variables[0]
    variable.values.setText("0.5, 1 bar")
    variable.values.editingFinished.emit()
    assert seen[-1] == {"name": "scan", "pressure": ["0.5 bar", "1 bar"]}
    assert editor.count.text() == "2 points"
    editor.add_variable()
    second = editor.variables[1]
    second.path.setCurrentText("energy")
    second.form.setCurrentIndex(1)
    second.start.setText("50 uJ")
    second.stop.setText("100 uJ")
    second.num.setValue(3)
    assert seen[-1]["energy"] == {
        "start": "50 uJ",
        "stop": "100 uJ",
        "num": 3,
        "spacing": "linear",
    }
    assert editor.count.text() == "6 points"
    editor.name.setText("my-scan")
    editor.name.editingFinished.emit()
    assert seen[-1]["name"] == "my-scan"
    second.start.setText("")  # incomplete: left out
    second.start.editingFinished.emit()
    assert "energy" not in seen[-1]
    editor.remove_variable(second)
    variable.path.setCurrentText("")  # no path: left out
    assert seen[-1] == {"name": "my-scan"} and "No variable" in editor.count.text()
    editor.set_value(
        {"name": "s", "base_seed": 3, "gas": "He", "energy": {"start": 1, "stop": 2}}
    )
    assert [v.path.currentText() for v in editor.variables] == ["gas", "energy"]
    assert editor.variables[1].form.currentData() == "range"
    editor.set_value(None)
    assert not editor.enabled.isChecked() and editor.table() is None
    editor.enabled.setChecked(False)


def test_the_scan_browser_draws_a_line_or_a_map(qtbot: QtBot, tmp_path: Path):
    view = ScanView()
    qtbot.addWidget(view)
    view.resize(700, 450)
    line = {
        "variables": [{"name": "pressure", "values": [0.5, 1.0, 1.5]}],
        "shape": [3],
        "status": ["done", "failed", "done"],
        "results": {
            "stats/energy": {
                "summary": "Energy.",
                "units": "J",
                "values": np.array([1e-4, np.nan, 0.9e-4]),
            }
        },
    }
    view.set_scan(line)
    assert "1 failed" in view.status.text()
    xs = np.asarray(view.failed.getData()[0], dtype=float)
    assert list(xs) == [1.0]
    shot(view, tmp_path, "scan-line")
    grid = {
        "variables": [
            {"name": "pressure", "values": [0.5, 1.0]},
            {"name": "energy", "values": [1e-4, 2e-4, 3e-4]},
        ],
        "shape": [2, 3],
        "status": [["done"] * 3] * 2,
        "results": {
            "stats/energy": {"summary": "", "units": "J", "values": np.ones((2, 3))}
        },
    }
    view.set_scan(grid)
    assert view.image.isVisible()
    view.show_error("no scan")
    assert view.status.text() == "no scan" and view.result.count() == 0


def test_a_scan_set_up_runs_and_opens_in_the_browser(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.confirm = lambda _title, _text: "discard"
    window.set_model(RecipeModel(SCAN), base=None)
    qtbot.waitUntil(lambda: window._built_route == "capillary", timeout=WAIT_MS)
    editor = window.scan_editor
    assert editor.enabled.isChecked() and editor.name.text() == "pressure"
    assert "pressure" in editor.paths
    # An edit is one undo step.
    editor.variables[0].values.setText("0.5, 1, 2 bar")
    editor.variables[0].values.editingFinished.emit()
    assert window.model is not None
    assert window.model.document["scan"]["pressure"] == ["0.5 bar", "1 bar", "2 bar"]
    window.undo.undo()
    assert window.model.document["scan"]["pressure"] == ["0.5 bar", "1 bar"]
    window.show_step("propagate")
    shot(window, tmp_path, "scan-setup")
    # Run it: point by point, then the browser.
    qtbot.waitUntil(lambda: not window.issues, timeout=WAIT_MS)
    window.run_panel.start()
    qtbot.waitUntil(lambda: window.results.scan.result.count() > 0, timeout=WAIT_MS * 2)
    assert "2 points" in window.results.scan.status.text()
    assert window.run_panel.progress.value() == 1000
    window.show_view("scan")
    shot(window, tmp_path, "scan-browser")
    # A scan that cannot be read says so.
    window.results._pending[-9] = ("scan", "x.h5")
    window.results.on_reply(Reply("scan_overview", -9, 0, error="unreadable"))
    assert "unreadable" in window.results.scan.status.text()
    window.close()
