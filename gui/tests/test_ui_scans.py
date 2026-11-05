# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scan editor, the scan browser and a scan run from its stage, offscreen."""

from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from solprop_gui.classes.workflow import Stage
from solprop_gui.model.recipe import RecipeModel
from solprop_gui.service.messages import Reply
from solprop_gui.service.worker import SCAN_OUTPUT
from solprop_gui.ui.scan_editor import ScanEditor
from solprop_gui.ui.scan_stage import ScanStage
from solprop_gui.ui.workflow_window import WorkflowWindow
from solprop_gui.views.scan_browser import ScanView
from test_design import SCAN
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture


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


def test_a_scan_set_up_on_its_stage_runs_and_shows_its_results(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = WorkflowWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.document.confirm = lambda _title, _text: "discard"
    window.document.set_model(RecipeModel(SCAN), base=None)
    qtbot.waitUntil(lambda: window.stage is Stage.SETUP, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: "pressure" in window.scans.editor.paths, timeout=WAIT_MS)
    editor = window.scans.editor
    assert editor.enabled.isChecked() and editor.name.text() == "pressure"
    # From the sheet the setup runs once: the scan is left out.
    text, _base, _title = window._run_recipe()
    assert "[scan]" not in text
    qtbot.waitUntil(lambda: window.progress().can_run, timeout=WAIT_MS)
    window.show_stage(Stage.SCAN)
    assert window.primary.text() == "Run scan"
    assert window.primary.isEnabled()
    # An edit is one undo step.
    editor.variables[0].values.setText("0.5, 1, 2 bar")
    editor.variables[0].values.editingFinished.emit()
    model = window.document.model
    assert model is not None
    assert model.document["scan"]["pressure"] == ["0.5 bar", "1 bar", "2 bar"]
    window.document.undo.undo()
    assert model.document["scan"]["pressure"] == ["0.5 bar", "1 bar"]
    window.primary.click()
    browser = window.scans.browser
    qtbot.waitUntil(lambda: browser.result.count() > 0, timeout=WAIT_MS * 2)
    assert window.stage is Stage.SCAN
    assert browser.status.text().startswith("2 points over pressure")
    assert window.scans.progress.text().startswith("The scan in")
    assert browser.plot.isVisible()
    shot(window, tmp_path, "scan-stage")
    # A scan file opened from Home comes to this stage too.
    found = next((tmp_path / "workspace").rglob(SCAN_OUTPUT))
    window.show_stage(Stage.HOME)
    window.open_results(str(found))
    assert window.stage is Stage.SCAN
    qtbot.waitUntil(lambda: browser.result.count() > 0, timeout=WAIT_MS)
    window._scan_path = "x.h5"
    window._pending[-9] = "scan"
    window._on_reply(Reply("scan_overview", -9, 0, error="unreadable"))
    assert "unreadable" in browser.status.text()
    window.close()


def test_a_number_s_menu_on_the_sheet_scans_it(qtbot: QtBot, tmp_path: Path) -> None:
    window = WorkflowWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.document.confirm = lambda _title, _text: "discard"
    window.begin_class("capillary")
    sheet = window.sheet
    qtbot.waitUntil(lambda: ("radius", "build") in sheet.rows, timeout=WAIT_MS)
    (scan,) = sheet.rows["radius", "build"].actions()
    scan.trigger()
    assert window.stage is Stage.SCAN
    model = window.document.model
    assert model is not None
    assert model.document["scan"] == {"name": "scan", "radius": ["125 um"]}
    assert window.scans.has_points()
    window.close()


def test_the_stage_follows_a_scan_s_points(qtbot: QtBot) -> None:
    stage = ScanStage()
    qtbot.addWidget(stage)
    stage.show_point({"points": 4})
    assert stage.progress.text() == "Running 4 points…"
    stage.show_point({"index": 1, "values": {"pressure": 1.0}, "status": None})
    assert stage.progress.text() == "Point 2: pressure = 1"
    assert not stage.has_points()
