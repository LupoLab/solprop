# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The window, its pages, docks and views, offscreen (smoke and screenshot tests)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QLineEdit, QWidget
from pytestqt.qtbot import QtBot

from conftest import TINY
from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.service.states import ExecutionState
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.icons import DRAWINGS, icon
from solprop_gui.ui.main_window import STEPS, MainWindow
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
        "density": {
            axis: scale * np.exp(-(np.linspace(-3, 3, 16) ** 2)).astype(np.float32)
            for axis in ("wavelength", "f")
        },
        "power": scale * np.exp(-(np.linspace(-3, 3, 8) ** 2)).astype(np.float32) * 1e9,
        "stats": {"energy": 1e-4 * scale},
    }


AXES = {
    "wavelength": np.linspace(2e-7, 2e-6, 16),
    "f": np.linspace(1.5e14, 1.5e15, 16),
    "time": np.linspace(-1e-13, 1e-13, 8),
    "bandpass": None,
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


def leave(field: QLineEdit, text: str) -> None:
    field.setText(text)
    field.editingFinished.emit()


def test_a_boolean_row_shows_solprops_default_selected(qtbot: QtBot):
    row = ParameterRow(
        meta("kerr", {"kind": "boolean"}, default="True", default_value=True)
    )
    qtbot.addWidget(row)
    assert row.choice.currentText() == "on (default)"
    seen = []
    row.edited.connect(lambda *args: seen.append(args))
    row.choice.setCurrentIndex(1)  # off
    row.choice.setCurrentIndex(0)  # the default: unset
    assert seen == [("kerr", "build", False), ("kerr", "build", MISSING)]
    row.set_value(False)
    assert row.choice.currentIndex() == 1 and row.is_set
    row.reset.click()
    assert seen[-1] == ("kerr", "build", MISSING)


def test_a_choice_without_a_known_default_says_solprop_chooses(qtbot: QtBot):
    row = ParameterRow(
        meta("integrator", {"kind": "choice", "choices": ["rk", "adams"]})
    )
    qtbot.addWidget(row)
    assert row.choice.currentText() == "automatic"
    row.set_resolved(["adams", "the route's rule"])
    assert row.choice.currentText() == "adams (default)"
    row.set_resolved(["adams", "the route's rule"])  # unchanged: nothing redone
    row.set_value("rk")
    assert row.choice.currentText() == "rk"
    row.set_resolved(["other", "a rule"])
    assert row.choice.itemText(0) == "automatic (a rule)"
    row.set_value("unknown")
    assert row.choice.currentIndex() == 0


def test_a_quantity_row_converts_between_units(qtbot: QtBot):
    row = ParameterRow(
        meta(
            "radius",
            {"kind": "number", "unit": "m"},
            {"kind": "object", "recipe": True},
            display="um",
            required=True,
        )
    )
    qtbot.addWidget(row)
    assert row.mode.isVisibleTo(row) and row.modes == ["number", "text"]
    assert row.number.number.placeholderText() == "required"
    seen = []
    row.edited.connect(lambda *args: seen.append(args[2]))
    leave(row.number.number, "10 fs")
    assert not row.issue.isHidden() and seen == []
    leave(row.number.number, "125")
    assert seen == ["125 um"] and row.issue.isHidden()
    assert row.number.units is not None
    row.number.units.setCurrentIndex(row.number.units.findData("mm"))
    assert row.number.number.text() == "0.125" and seen[-1] == "0.125 mm"
    leave(row.number.number, "0.2 m")  # a whole quantity in another unit
    assert seen[-1] == "0.2 m" and row.number.units.currentData() == "m"
    leave(row.number.number, "")
    assert seen[-1] is MISSING
    row.set_value({"type": "profile"})  # a table: shown as text
    assert row.stack.currentWidget() is row.text
    row.mode.setCurrentIndex(0)
    assert row.stack.currentWidget() is row.number


def test_a_range_row_is_two_numbers_with_one_unit(qtbot: QtBot):
    row = ParameterRow(
        meta(
            "wavelength_limits",
            {"kind": "pair", "unit": "m"},
            display="nm",
            default_value=None,
        )
    )
    qtbot.addWidget(row)
    seen = []
    row.edited.connect(lambda *args: seen.append(args[2]))
    row.set_value(["200 nm", "2 um"])
    assert row.range.low.number.text() == "200"
    assert row.range.high.number.text() == "2000"
    leave(row.range.low.number, "")
    assert "both ends" in row.issue.text()
    leave(row.range.low.number, "100")
    assert seen[-1] == ["100 nm", "2000 nm"]
    row.range.units.setCurrentIndex(row.range.units.findData("um"))
    assert seen[-1] == ["0.1 um", "2 um"]
    leave(row.range.low.number, "")
    leave(row.range.high.number, "")
    assert seen[-1] is MISSING


def test_numbers_without_units_and_text_rows(qtbot: QtBot):
    rtol = ParameterRow(meta("rtol", {"kind": "number", "unit": None}, default=None))
    qtbot.addWidget(rtol)
    assert rtol.number.units is None
    seen = []
    rtol.edited.connect(lambda *args: seen.append(args[2]))
    leave(rtol.number.number, "1e-6")
    leave(rtol.number.number, "1 m")
    assert seen == [1e-6] and "not a number" in rtol.issue.text()
    count = ParameterRow(meta("n", {"kind": "integer"}, default="3", default_value=3))
    qtbot.addWidget(count)
    assert count.number.number.placeholderText() == "3"
    count.edited.connect(lambda *args: seen.append(args[2]))
    leave(count.number.number, "4")
    leave(count.number.number, "4.5")
    assert seen[-1] == 4 and "whole" in count.issue.text()
    table = ParameterRow(meta("fill", {"kind": "mapping"}, required=True))
    qtbot.addWidget(table)
    assert table.text.placeholderText() == "required"
    leave(table.text, "{ He = 1")
    assert not table.issue.isHidden()
    gas = ParameterRow(meta("gas", {"kind": "string"}, {"kind": "mapping"}))
    qtbot.addWidget(gas)
    gas.edited.connect(lambda *args: seen.append(args[2]))
    leave(gas.text, "Ar")
    leave(gas.text, "")
    assert seen[-2:] == ["Ar", MISSING]
    gas.set_resolved(["He", "a rule"])
    assert "a rule" in gas.text.placeholderText()
    gas.set_issue("solprop's word", "warning")
    assert gas.issue.text() == "solprop's word"
    plain = ParameterRow(meta("x", {"kind": "array"}, default="[1]"))
    qtbot.addWidget(plain)
    assert plain.text.placeholderText() == "default: [1]"
    other = ParameterRow(meta("y", {"kind": "array"}))
    qtbot.addWidget(other)
    assert "automatic" in other.text.placeholderText()


def test_a_number_box_steps_with_keys_and_wheel(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: PLC0415 - only here
    from PySide6.QtGui import QWheelEvent  # noqa: PLC0415

    from solprop_gui.ui.quantity import NumberEdit, step  # noqa: PLC0415

    assert step("x", 1) is None and step("0", 1) == "0.1"
    box = NumberEdit()
    qtbot.addWidget(box)
    box.setPlaceholderText("125")
    qtbot.keyClick(box, Qt.Key.Key_Up)
    assert box.text() == "135"  # a tenth of the leading digit's place
    qtbot.keyClick(box, Qt.Key.Key_Down)
    qtbot.keyClick(box, Qt.Key.Key_A)
    assert box.text() == "125a"
    wheel = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    box.setText("1")
    box.wheelEvent(wheel)  # no focus: left to the page's scroll
    assert box.text() == "1"
    monkeypatch.setattr(box, "hasFocus", lambda: True)  # no window to focus offscreen
    box.wheelEvent(wheel)
    assert box.text() == "1.1"


# ---------------------------------------------------------------------- views


def test_the_overview_draws_rows_and_follows_the_newest(qtbot: QtBot, tmp_path: Path):
    view = OverviewView()
    qtbot.addWidget(view)
    view.resize(900, 700)
    view.show()  # the maps' z axes are linked on screen
    qtbot.waitExposed(view)
    view.add_row(row_data(0))  # no axes yet: kept, not drawn
    view.set_axes(AXES)
    for index in range(1, 4):
        view.add_row(row_data(index, scale=1.0 - 0.1 * index))
    view.add_row(row_data(2, scale=0.5))  # a replayed plane replaces its row
    assert len(view.rows) == 4 and view.slider.value() == 3
    view.slider.setValue(1)
    assert not view.follow.isChecked() and "plane 1" in view.position.text()
    view.set_extent(0.0, 1.0)  # a run's whole length: the rows fill it
    box = view.spectrum_map.plot.getViewBox()
    assert box.viewRange()[1] == pytest.approx([0.0, 1.0])
    view.apply_options(PlotOptions(floor=-30.0, time_limits=(-2e-14, None)))
    assert view.power_map.plot.getViewBox().viewRange()[0][0] == pytest.approx(-20)
    assert box.viewRange()[1] == pytest.approx([0.0, 1.0])
    shot(view, tmp_path, "overview")
    # The frequency axis at once, from the rows' own; a new band waits for new rows.
    view.apply_options(PlotOptions(axis="f", scale="linear", bandpass=(7e-7, 9e-7)))
    assert view.spectrum_map.plot.getAxis("bottom").labelText == "Frequency (THz)"
    assert view.spectrum_map.x[0] == pytest.approx(150.0)  # THz
    assert "whole spectrum, until re-read" in view.spectrum_note.text()
    view.set_axes({**AXES, "bandpass": (7e-7, 9e-7)})
    assert view.spectrum_note.text() == "time domain within 700-900 nm"
    view.apply_options(PlotOptions())
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
    peak = float(np.max(np.asarray(row_data(0, 2.0)["density"]["wavelength"])))  # pyright: ignore[reportIndexIssue]
    assert view.reference() == pytest.approx(peak)  # the highest first plane
    shot(view, tmp_path, "compare")
    view.clear()
    assert view.reference() == 0.0


def test_every_drawing_makes_an_icon_with_each_badge(qtbot: QtBot):
    from PySide6.QtGui import QColor  # noqa: PLC0415 - only here

    for name in DRAWINGS:
        for verdict in (None, "accepted", "warning", "error"):
            made = icon(name, QColor("black"), verdict, size=32)
            assert not made.pixmap(32, 32).toImage().isNull()


# ---------------------------------------------------------------------- the window


def test_the_window_runs_the_whole_lifecycle(qtbot: QtBot, tmp_path: Path):  # noqa: PLR0915
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(
        lambda: window.start.examples.topLevelItemCount() > 0, timeout=WAIT_MS
    )
    shot(window, tmp_path, "start")

    # A new setup from the card: solprop's check says what is required.
    window.new_from_card("capillary-averaged")
    qtbot.waitUntil(lambda: "required" in _issues(window), timeout=WAIT_MS)
    assert window.verdicts["geometry"] == "error"
    assert not window.run_panel.run_button.isEnabled()

    # Open a setup: every page is built from solprop's metadata and accepted.
    window.confirm = lambda _title, _text: "discard"
    window.set_model(RecipeModel(TINY), base=str(tmp_path))
    qtbot.waitUntil(lambda: "accepts" in _issues(window), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: "Soliton order" in window.summary.text(), timeout=WAIT_MS)
    for key in STEPS[1:6]:
        window.show_step(key)
        shot(window, tmp_path, f"page-{key}")
    assert set(window.verdicts.values()) == {"accepted"}

    # Back and forward through the steps, by the toolbar's actions.
    window.show_step("start")
    assert not window.back_action.isEnabled()
    window.forward_action.trigger()
    assert window.step == "geometry"
    window.step_actions["postprocess"].trigger()
    assert not window.forward_action.isEnabled()
    assert window.views.currentWidget() is window.postprocess.view
    window.step_by(5)  # past the end: stays
    window.back_action.trigger()
    assert window.step == "results"
    window.back_action.trigger()
    assert window.step == "propagate"

    # Edit, refuse, undo and redo.
    energy = window.pages["input"].rows["energy", "build"]
    leave(energy.number.number, "150")
    assert window.model is not None and window.model.get("energy") == "150 uJ"
    assert window.isWindowModified()
    window.undo.undo()
    assert window.model.get("energy") == "100 uJ"
    window.undo.redo()
    window.edit("energy", "build", "150 uJ")  # unchanged: no command
    window.edit("energy", "build", "1 fs")
    qtbot.waitUntil(lambda: window.verdicts["input"] == "error", timeout=WAIT_MS)
    window.issue_list.itemActivated.emit(window.issue_list.item(0))
    assert window.step == "input"
    window.undo.undo()
    qtbot.waitUntil(window.run_panel.run_button.isEnabled, timeout=WAIT_MS)

    # The code dock, the export and saving.
    window.console.setCurrentWidget(window.code_view)
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
    window.show_step("propagate")
    panel.estimate()
    qtbot.waitUntil(lambda: "memory" in panel.estimate_text.text(), timeout=WAIT_MS)
    panel.start()
    assert window.views.currentWidget() is panel.view
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
    assert window.panels.currentWidget() is window.results
    assert window.views.currentWidget() is window.results.overview
    shot(window, tmp_path, "results")
    window.results.open_setup()
    qtbot.waitUntil(
        lambda: window.model is not None and window.model.path is None, timeout=WAIT_MS
    )
    assert window.results.runs.count() == 2
    window.results.runs.itemActivated.emit(window.results.runs.item(0))
    assert window.results.path is not None
    window.close()


def test_panels_hide_and_come_back(qtbot: QtBot, tmp_path: Path):
    from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: PLC0415 - only here
    from PySide6.QtGui import QEnterEvent, QMouseEvent  # noqa: PLC0415

    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    assert not window.right.isVisible()  # the welcome screen has the window
    window.show_step("geometry")
    assert window.right.isVisible()
    handle = window.split.handle(1)
    point = QPointF(2, 2)
    handle.enterEvent(QEnterEvent(point, point, point))
    shot(window, tmp_path, "lit-handle")
    handle.leaveEvent(QEvent(QEvent.Type.Leave))
    click = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        point,
        point,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    handle.mouseDoubleClickEvent(click)
    assert not window.panels.isVisible() and not window.settings_action.isChecked()
    window.right.handle(1).mouseDoubleClickEvent(click)
    assert not window.console.isVisible()
    window.settings_action.trigger()
    assert window.panels.isVisible()
    window.reset_layout()
    assert window.console.isVisible() and window.messages_action.isChecked()
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
    assert window.bridge.request("examples") == -1  # closed: no kernel is started


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
