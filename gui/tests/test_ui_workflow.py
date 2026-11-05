# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workflow window, offscreen: Home, Setup, Run and Analyse, for every class."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QBoxLayout, QLineEdit, QWidget
from pytestqt.qtbot import QtBot

from conftest import TINY
from solprop_gui.classes.catalog import CLASSES, GENERIC_PAGES
from solprop_gui.classes.problem import AnalysisPage, ProblemClass
from solprop_gui.classes.workflow import Stage
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.ui import theme as ui_theme
from solprop_gui.ui.analyse import AnalyseStage
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.theme import FONT
from solprop_gui.ui.workflow_window import WorkflowWindow

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


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path) -> Iterator[WorkflowWindow]:
    """Return a window whose questions answer themselves, closed at the end."""
    made = WorkflowWindow(workspace=tmp_path / "runs")
    made.confirm = lambda _title, _text: "discard"
    qtbot.addWidget(made)
    made.show()
    yield made
    made.close()


def shot(widget: QWidget, folder: Path, name: str) -> None:
    """Render a widget offscreen and check the picture is not empty."""
    image = widget.grab().toImage()
    assert not image.isNull() and image.width() > 100 and image.height() > 100
    assert image.save(str(folder / f"{name}.png"))


def open_tiny(window: WorkflowWindow, qtbot: QtBot) -> None:
    """Open the quick capillary, and wait until solprop has accepted it."""
    window.document.set_model(RecipeModel(TINY), base=None)
    qtbot.waitUntil(lambda: bool(window.sheet.rows), timeout=WAIT_MS)
    qtbot.waitUntil(window.primary.isEnabled, timeout=WAIT_MS)


def test_the_window_uses_the_carried_font_and_its_licence_travels_with_it(
    window: WorkflowWindow,
) -> None:
    assert FONT in QFontDatabase.families()
    assert window.font().family() == FONT
    folder = Path(ui_theme.__file__).parent / "assets" / "fonts"
    assert len(list(folder.glob("*.ttf"))) == 3
    assert "SIL OPEN FONT LICENSE" in (folder / "LICENSE.txt").read_text("utf-8")


def test_home_offers_each_class_and_asks_for_nothing_open(
    window: WorkflowWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    assert window.stage is Stage.HOME
    assert not window.top.isVisible()
    assert {problem.key for problem in CLASSES} < set(window.home.tiles)
    assert not window.home.tiles["design"].isEnabled()
    with qtbot.waitSignal(window.home.class_chosen) as chosen:
        window.home.tiles["capillary"].clicked.emit()
    assert chosen.args == ["capillary"]
    window.document.undo.clear()
    shot(window, tmp_path, "home")


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_each_class_opens_its_sheet_ready_to_run(
    window: WorkflowWindow, qtbot: QtBot, tmp_path: Path, problem: ProblemClass
) -> None:
    window.begin_class(problem.key)
    qtbot.waitUntil(lambda: window.problem is problem, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: window.sheet.problem is problem, timeout=WAIT_MS)
    qtbot.waitUntil(window.primary.isEnabled, timeout=WAIT_MS)
    assert window.stage is Stage.SETUP
    assert window.primary.text() == "Run"
    assert problem.title in window.context.text()
    model = window.document.model
    assert model is not None
    # Every row shows what the example holds; none of it is the front end's.
    shown = [
        path
        for section in problem.sections
        for path in (row.path for row in section.fields)
    ]
    for path in shown:
        assert (path, "build") in window.sheet.rows
    assert window.sheet.ready.text().startswith("Ready")
    assert window.sheet.examples.currentText() == model.title
    if problem.derived:
        first = problem.derived[0].key
        qtbot.waitUntil(
            lambda: window.sheet.derived[first].text() != "…", timeout=WAIT_MS
        )
    shot(window, tmp_path, f"sheet-{problem.key}")


def test_a_row_edits_the_recipe_and_is_undone(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    open_tiny(window, qtbot)
    model = window.document.model
    assert model is not None
    window.sheet.rows["length", "build"].edited.emit("length", "build", "6 cm")
    assert model.get("length") == "6 cm"
    assert model.dirty
    assert window.isWindowModified()
    window.document.undo.undo()
    assert model.get("length") == "5 cm"
    window.sheet.rows["length", "build"].edited.emit("length", "build", MISSING)
    assert model.get("length") is MISSING
    qtbot.waitUntil(lambda: not window.primary.isEnabled(), timeout=WAIT_MS)
    assert window.sheet.ready.text().startswith("Not ready")
    assert window.sheet.ready.property("error")


def test_the_sheet_helps_finds_and_folds(window: WorkflowWindow, qtbot: QtBot) -> None:
    open_tiny(window, qtbot)
    sheet = window.sheet
    row = sheet.rows["radius", "build"]
    sheet._focus_moved(None, row.editor)
    assert sheet.help.text() == row.meta["summary"]
    assert not sheet.advanced.body.isVisible()
    sheet.advanced.header.setChecked(True)
    find = sheet.advanced.findChild(QLineEdit, "find-setting")
    assert isinstance(find, QLineEdit)
    find.setText("tolerance")
    rtol = sheet.rows["rtol", "build"]
    other = sheet.rows["seed", "build"]
    assert rtol.isVisibleTo(sheet.advanced.body)
    assert not other.isVisibleTo(sheet.advanced.body)
    find.clear()
    assert other.isVisibleTo(sheet.advanced.body)
    sheet.resize(800, 700)
    qtbot.waitUntil(lambda: sheet.split.direction() is QBoxLayout.Direction.TopToBottom)


def test_a_tall_editor_takes_the_width_below_its_label(qtbot: QtBot) -> None:
    catalog = {"Linear": [{"name": "start", "forms": [{"kind": "number"}]}]}
    meta = {
        "name": "pressure",
        "forms": [
            {"kind": "number", "unit": "bar"},
            {"kind": "object", "types": ["Linear"]},
        ],
    }
    row = ParameterRow(meta, catalog=catalog, one_line=True)  # pyright: ignore[reportArgumentType]
    qtbot.addWidget(row)
    assert row._line.indexOf(row.stack) >= 0
    row.mode.setCurrentIndex(row.modes.index("object"))
    assert row._line.indexOf(row.stack) < 0
    assert row._column.indexOf(row.stack) >= 0


def test_a_run_goes_from_setup_to_analyse_with_one_scrubber(
    window: WorkflowWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    open_tiny(window, qtbot)
    window.primary.click()
    qtbot.waitUntil(lambda: window.stage is Stage.RUN, timeout=WAIT_MS)
    assert window.primary.text() in ("Stop", "Rerun")
    shot(window, tmp_path, "run")
    qtbot.waitUntil(lambda: window.stage is Stage.ANALYSE, timeout=WAIT_MS)
    assert window.primary.text() == "Rerun"
    scrubber = window.analyse.scrubber
    qtbot.waitUntil(lambda: bool(scrubber.positions), timeout=WAIT_MS)
    overview = window.results.overview
    qtbot.waitUntil(lambda: bool(overview.rows), timeout=WAIT_MS)
    assert window.analyse.current == "overview"
    assert not window.analyse.buttons["modes"].isEnabled()
    assert window.analyse.buttons["along"].isEnabled()
    last = len(scrubber.positions) - 1
    assert scrubber.slider.value() == last
    assert f"plane {last + 1} of {last + 1}" in scrubber.text.text()
    scrubber.step(-2)
    assert window.results.plane.slider.value() == last - 2
    assert overview.slider.value() == overview.nearest(scrubber.positions[last - 2])
    # A plane picked on a map moves the scrubber.
    overview.slider.setValue(1)
    assert scrubber.slider.value() == 1
    # Moving between pages keeps the place, and the keys stop at the rail's ends.
    window.analyse.show_page("along")
    assert scrubber.slider.value() == 1
    window.analyse.step_page(-1)
    assert window.analyse.current == "spectrogram"
    assert "not built" in window.analyse.unbuilt.text()
    window.analyse.step_page(-99)
    assert window.analyse.current == "overview"
    shot(window, tmp_path, "analyse")
    window.show_stage(Stage.SETUP)
    assert window.top.isVisible()
    window.show_stage(Stage.HOME)
    assert window.top.isVisible()  # the stages lead back to what is open
    assert window.home.runs.count() == 1


def test_results_from_home_find_their_class_and_restore_the_setup(
    window: WorkflowWindow, qtbot: QtBot, finished_run: Path
) -> None:
    window.open_results(str(finished_run))
    assert window.stage is Stage.ANALYSE
    qtbot.waitUntil(lambda: window.problem is not None, timeout=WAIT_MS)
    assert window.problem is not None
    assert window.problem.key == "capillary"
    assert window.stage is Stage.ANALYSE  # restoring the sheet does not move on
    assert window.has_results
    assert window.primary.text() == "Run"
    assert set(window.analyse.buttons) == {p.key for p in window.problem.pages}


def test_a_file_without_a_recipe_gets_the_generic_rail(
    window: WorkflowWindow,
) -> None:
    window._file_described({"z": [0.0, 1.0], "field_planes": [0, 1], "recipe": False})
    assert set(window.analyse.buttons) == {page.key for page in GENERIC_PAGES}
    assert window.analyse.scrubber.positions == [0.0, 1.0]


def test_a_failed_run_stays_with_its_advice(
    window: WorkflowWindow, qtbot: QtBot, finished_run: Path
) -> None:
    open_tiny(window, qtbot)
    window.has_run = True
    window.show_stage(Stage.RUN)
    window._report_failure("It failed", "The grid is too small.", "")
    window._run_ended(str(finished_run))
    assert window.stage is Stage.RUN
    assert not window.run_stage.advice.isHidden()
    assert window.has_results  # the planes reached stay open for analysis
    window._run_state("starting")
    assert window.run_stage.advice.isHidden()


def test_a_page_that_a_file_cannot_show_says_why(qtbot: QtBot) -> None:
    stage = AnalyseStage({})
    qtbot.addWidget(stage)
    pages = [
        AnalysisPage("a", "A", ("fields",)),
        AnalysisPage("b", "B", (), "compression"),
        AnalysisPage("c", "C"),
    ]
    stage.set_pages(pages, {"refused": {"compression": "Spatial runs cannot."}})
    assert not stage.buttons["a"].isEnabled()
    assert "fields" in stage.buttons["a"].toolTip()
    assert not stage.buttons["b"].isEnabled()
    assert stage.buttons["b"].toolTip() == "Spatial runs cannot."
    assert stage.current == "c"


def test_the_drawer_shows_the_recipe_and_its_python(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    open_tiny(window, qtbot)
    window.show_recipe()
    assert window.drawer.isVisible()
    assert "Tiny capillary" in window.drawer_text.toPlainText()
    window.show_python()
    qtbot.waitUntil(
        lambda: "prop_capillary" in window.drawer_text.toPlainText(), timeout=WAIT_MS
    )


def test_recent_setups_are_listed(
    window: WorkflowWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "setup.toml"
    path.write_text(TINY, encoding="utf-8")
    window.open_setup(str(path))
    qtbot.waitUntil(lambda: window.stage is Stage.SETUP, timeout=WAIT_MS)
    window.show_stage(Stage.HOME)
    assert window.home.setups.count() == 1
    window.open_setup(str(tmp_path / "nothing.toml"))
    assert "Cannot open" in window.sheet.ready.text()


def test_closing_asks_before_losing_a_setup(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    open_tiny(window, qtbot)
    window.document.edit("length", "build", "6 cm")
    asked: list[str] = []

    def answer(title: str, _text: str) -> str:
        asked.append(title)
        return "cancel"

    window.confirm = answer
    window.close()
    assert asked == ["Unsaved setup"]
    assert window.isVisible()
    window.confirm = lambda _title, _text: "discard"
