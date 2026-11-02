# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The window, its pages, docks and views, offscreen (smoke and screenshot tests)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget
from pytestqt.qtbot import QtBot

from conftest import TINY
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.service.states import ExecutionState
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.views.compare import CompareView
from solprop_gui.views.live import OverviewView, decibels

WAIT_MS = 120_000


@pytest.fixture(autouse=True)
def private_settings(tmp_path: Path) -> Iterator[None]:
    """Keep the window's settings and recovery file out of the user's own."""
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )
    yield
    QStandardPaths.setTestModeEnabled(False)


def shot(widget: QWidget, folder: Path, name: str) -> Path:
    """Render a widget offscreen and check the picture is not empty."""
    path = folder / f"{name}.png"
    image = widget.grab().toImage()
    assert not image.isNull() and image.width() > 100 and image.height() > 100
    assert image.save(str(path))
    return path


def row_data(index: int, scale: float = 1.0) -> dict[str, object]:
    return {
        "index": index,
        "z": 0.01 * index,
        "density": scale * np.exp(-(np.linspace(-3, 3, 16) ** 2)).astype(np.float32),
        "power": scale * np.exp(-(np.linspace(-3, 3, 8) ** 2)).astype(np.float32) * 1e9,
        "stats": {"energy": 1e-4 * scale},
    }


AXES = {
    "wavelength": np.linspace(2e-7, 2e-6, 16),
    "time": np.linspace(-1e-13, 1e-13, 8),
}


# ---------------------------------------------------------------------- rows


def meta(name: str, *forms: dict[str, object], **more: object) -> dict[str, object]:
    return {
        "name": name,
        "forms": list(forms),
        "summary": "S.",
        "anchor": "guide/a.md",
        "level": "basic",
        **more,
    }


def test_a_boolean_row_picks_default_on_or_off(qtbot: QtBot):
    row = ParameterRow(meta("kerr", {"kind": "boolean"}, default="True"))
    qtbot.addWidget(row)
    assert isinstance(row.editor, QComboBox)
    seen = []
    row.edited.connect(lambda *args: seen.append(args))
    row.editor.setCurrentIndex(2)
    row.editor.setCurrentIndex(0)
    assert seen == [("kerr", "build", False), ("kerr", "build", MISSING)]
    row.set_value(True)
    assert row.editor.currentIndex() == 1 and row.is_set
    row.set_resolved([True, "the rule"])
    assert "the rule" in row.editor.itemText(0)


def test_a_choice_row_lists_solprops_choices(qtbot: QtBot):
    row = ParameterRow(meta("shape", {"kind": "choice", "choices": ["gauss", "sech"]}))
    qtbot.addWidget(row)
    assert isinstance(row.editor, QComboBox)
    row.set_value("sech")
    assert row.editor.currentIndex() == 2
    row.set_value("other")
    assert row.editor.currentIndex() == 0
    seen = []
    row.edited.connect(lambda *args: seen.append(args[2]))
    row.editor.setCurrentIndex(1)
    assert seen == ["gauss"]


def test_a_text_row_reads_and_refuses_what_is_typed(qtbot: QtBot):
    row = ParameterRow(
        meta("energy", {"kind": "number", "unit": "J"}, display="uJ", required=True)
    )
    qtbot.addWidget(row)
    assert isinstance(row.editor, QLineEdit)
    assert row.editor.placeholderText() == "required"
    seen = []
    row.edited.connect(lambda *args: seen.append(args[2]))
    row.editor.setText("10 fs")
    row.editor.editingFinished.emit()
    assert not row.issue.isHidden() and seen == []
    row.editor.setText("10 uJ")
    row.editor.editingFinished.emit()
    row.editor.editingFinished.emit()  # unchanged: announced once
    row.editor.setText("")
    row.editor.editingFinished.emit()
    assert seen == ["10 uJ", MISSING] and row.issue.isHidden()
    row.set_issue("solprop's word", "warning")
    assert row.issue.text() == "solprop's word"


def test_a_row_says_where_its_default_comes_from(qtbot: QtBot):
    row = ParameterRow(meta("rtol", {"kind": "number", "unit": None}, default=None))
    qtbot.addWidget(row)
    assert "solprop chooses" in row.editor.placeholderText()
    row.set_resolved([1e-6, "the route's tolerance"])
    assert "1e-06" in row.editor.placeholderText()
    row.set_resolved(None)
    row2 = ParameterRow(meta("n", {"kind": "integer"}, default="3"))
    qtbot.addWidget(row2)
    assert row2.editor.placeholderText() == "default: 3"


# ---------------------------------------------------------------------- views


def test_the_overview_draws_rows_and_follows_the_newest(qtbot: QtBot, tmp_path: Path):
    view = OverviewView()
    qtbot.addWidget(view)
    view.resize(900, 700)
    view.add_row(row_data(0))  # no axes yet: kept, not drawn
    view.set_axes(AXES)
    for index in range(1, 4):
        view.add_row(row_data(index, scale=1.0 - 0.1 * index))
    view.add_row(row_data(2, scale=0.5))  # a replayed plane replaces its row
    assert len(view.rows) == 4 and view.slider.value() == 3
    view.slider.setValue(1)
    assert not view.follow.isChecked() and "plane 1" in view.position.text()
    view.floor.setValue(-30.0)
    shot(view, tmp_path, "overview")
    view.set_rows(AXES, [])
    assert view.rows == []
    view.set_rows(AXES, [row_data(1), row_data(0)])
    assert [r["index"] for r in view.rows] == [0, 1]
    view._show_plane(0)
    view.clear()
    view._show_plane(0)


def test_decibels_are_relative_to_the_reference():
    assert decibels(np.array([1.0, 0.1]), 1.0).tolist() == pytest.approx([0.0, -10.0])
    assert decibels(np.array([0.0]), 0.0)[0] == pytest.approx(-300.0)


def test_compare_shares_one_reference(qtbot: QtBot, tmp_path: Path):
    view = CompareView()
    qtbot.addWidget(view)
    view.add_run("a", AXES, [row_data(1, 0.5), row_data(0, 2.0)])
    view.add_run("b", AXES, [row_data(0, 1.0)])
    view.add_run("empty", AXES, [])
    assert list(view.runs) == ["a", "b"]
    peak = float(np.max(np.asarray(row_data(0, 2.0)["density"])))
    assert view.reference() == pytest.approx(peak)  # the highest first plane
    shot(view, tmp_path, "compare")
    view.clear()
    assert view.reference() == 0.0


# ---------------------------------------------------------------------- the window


def test_the_window_runs_the_whole_lifecycle(qtbot: QtBot, tmp_path: Path):  # noqa: PLR0915
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.start.examples.count() > 0, timeout=WAIT_MS)
    shot(window, tmp_path, "start")

    # A new setup from the card: solprop's check says what is required.
    window.new_from_card("capillary-averaged")
    qtbot.waitUntil(lambda: "required" in _issues(window), timeout=WAIT_MS)
    assert window.steps.item(1).text().startswith("✗")
    assert not window.run_panel.run_button.isEnabled()

    # Open a setup: every page is built from solprop's metadata and accepted.
    window.confirm = lambda _title, _text: "discard"
    window.set_model(RecipeModel(TINY), base=str(tmp_path))
    qtbot.waitUntil(lambda: "accepts" in _issues(window), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: "Soliton order" in window.summary.text(), timeout=WAIT_MS)
    for row in range(1, 6):
        window.steps.setCurrentRow(row)
        shot(window, tmp_path, f"page-{row}")
    assert all(window.steps.item(r).text().startswith("✓") for r in range(1, 6))

    # Edit, refuse, undo and redo.
    energy = window.pages["input"].rows["energy", "build"]
    assert isinstance(energy.editor, QLineEdit)
    energy.editor.setText("150 uJ")
    energy.editor.editingFinished.emit()
    assert window.model is not None and window.model.get("energy") == "150 uJ"
    assert window.isWindowModified()
    window.undo.undo()
    assert window.model.get("energy") == "100 uJ"
    window.undo.redo()
    window.edit("energy", "build", "150 uJ")  # unchanged: no command
    window.edit("energy", "build", "1 fs")
    qtbot.waitUntil(lambda: "✗" in window.steps.item(4).text(), timeout=WAIT_MS)
    window.issue_list.itemActivated.emit(window.issue_list.item(0))
    assert window.steps.currentRow() == 4
    window.undo.undo()
    qtbot.waitUntil(window.run_panel.run_button.isEnabled, timeout=WAIT_MS)

    # The code dock, the export and saving.
    window.code_dock.show()
    qtbot.waitUntil(lambda: "import" in window.code_view.toPlainText(), timeout=WAIT_MS)
    assert window.save_as(tmp_path / "setup.toml")
    assert not window.isWindowModified()
    assert window.save()
    window.write_recovery()  # saved: nothing to recover
    window.edit("energy", "build", "120 uJ")
    window.write_recovery()
    assert window.recovery_path().exists()

    # Estimate, run, stop and resume.
    panel = window.run_panel
    window.steps.setCurrentRow(5)
    panel.estimate()
    qtbot.waitUntil(lambda: "memory" in panel.estimate_text.text(), timeout=WAIT_MS)
    panel.start()
    qtbot.waitUntil(lambda: len(panel.view.rows) >= 2, timeout=WAIT_MS)
    panel.start()  # a run is going: ignored
    panel.stop()
    assert panel.state in (ExecutionState.STOPPING, ExecutionState.COMPLETE)
    qtbot.waitUntil(lambda: not panel.active, timeout=WAIT_MS)
    if panel.state == ExecutionState.STOPPED:
        qtbot.waitUntil(panel.resume_button.isEnabled, timeout=WAIT_MS)
        panel.resume()
        qtbot.waitUntil(lambda: panel.state == ExecutionState.COMPLETE, timeout=WAIT_MS)
    shot(window, tmp_path, "run")
    qtbot.waitUntil(lambda: not panel.resume_button.isEnabled(), timeout=WAIT_MS)
    first = window.results.path

    # Results, compare and the setup of a run.
    qtbot.waitUntil(lambda: len(window.results.overview.rows) == 6, timeout=WAIT_MS)
    window.results.add_to_compare()
    panel.start()
    qtbot.waitUntil(lambda: window.results.path != first, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: window.results.capabilities is not None, timeout=WAIT_MS)
    window.results.add_to_compare()
    qtbot.waitUntil(lambda: len(window.results.compare.runs) == 2, timeout=WAIT_MS)
    window.open_results(str(first))
    assert window.stack.currentWidget() is window.results
    shot(window, tmp_path, "results")
    window.results.open_setup()
    qtbot.waitUntil(
        lambda: window.model is not None and window.model.path is None, timeout=WAIT_MS
    )
    assert window.start.runs.count() == 2
    window.close()


def test_closing_over_a_run_or_unsaved_changes_asks(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.set_model(RecipeModel(TINY), base=None)
    window.edit("energy", "build", "120 uJ")
    answers = ["cancel", "discard"]
    window.confirm = lambda _title, _text: answers.pop(0) if answers else "discard"
    assert not window.close()
    assert window.close()


def test_opening_a_recipe_file_and_an_example(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    recipe = tmp_path / "a.toml"
    recipe.write_text(TINY, encoding="utf-8")
    window.open_recipe(recipe)
    assert window.model is not None and window.base == str(tmp_path)
    window.open_recipe(tmp_path / "missing.toml")  # refused, said in the status bar
    assert window.model.path == recipe
    window.open_example("soliton")  # a route the window does not offer yet
    qtbot.waitUntil(
        lambda: window.model is not None and window.model.route != "capillary",
        timeout=WAIT_MS,
    )
    window.close()


def _issues(window: MainWindow) -> str:
    return " ".join(
        window.issue_list.item(i).text() for i in range(window.issue_list.count())
    )
