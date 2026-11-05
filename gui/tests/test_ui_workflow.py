# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workflow window, offscreen: Home, Setup, Run and Analyse, for every class."""

import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, QStandardPaths, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QBoxLayout, QLabel, QLineEdit, QWidget
from pytestqt.qtbot import QtBot

from conftest import TINY
from solprop_gui.classes.catalog import CLASSES, GENERIC_PAGES
from solprop_gui.classes.problem import AnalysisPage, ProblemClass
from solprop_gui.classes.workflow import Stage
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.service.messages import Event
from solprop_gui.service.states import ExecutionState
from solprop_gui.ui import theme as ui_theme
from solprop_gui.ui.analyse import AnalyseStage
from solprop_gui.ui.analysis_pages import (
    AlongPage,
    CompressPage,
    HealthPage,
    ModesPage,
    PlanePage,
    SpectrogramPage,
)
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


def test_numbers_follow_typing_and_one_undo_takes_back_the_number(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    open_tiny(window, qtbot)
    sheet = window.sheet
    order = sheet.derived["soliton_order"]
    qtbot.waitUntil(lambda: order.text() != "…", timeout=WAIT_MS)
    before = order.text()
    model = window.document.model
    assert model is not None
    box = sheet.rows["energy", "build"].number.number
    for text in ("2", "20", "200"):
        box.setText(text)
        box.textEdited.emit(text)
    assert model.get("energy") == "200 uJ"
    qtbot.waitUntil(lambda: order.text() != before, timeout=WAIT_MS)
    window.document.undo.undo()
    assert model.get("energy") == "100 uJ"
    assert box.text() == "100"


def test_a_target_typed_into_a_number_writes_a_pin_and_can_be_released(  # noqa: PLR0915 - one story
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    open_tiny(window, qtbot)
    sheet = window.sheet
    model = window.document.model
    assert model is not None
    numbers = {number.key: number for number in sheet.problem.derived}  # pyright: ignore[reportOptionalMemberAccess]
    order = sheet.derived["soliton_order"]
    qtbot.waitUntil(lambda: order.text() != "…", timeout=WAIT_MS)

    sheet.open_target(numbers["soliton_order"])
    assert sheet.target_editor.isVisible()
    assert sheet.target_editor.moves.count() == 1  # the energy: no peak power is set
    sheet.target_editor.value.setText("not a number")
    sheet.target_editor.accept()
    assert sheet.target_editor.problem.text()
    assert model.get("energy", "pins") is MISSING
    sheet.target_editor.value.setText("3")
    sheet.target_editor.accept()
    assert not sheet.target_editor.isVisible()
    assert model.get("energy", "pins") == {"target": "soliton_order", "order": 3.0}
    assert model.get("energy") is MISSING
    qtbot.waitUntil(lambda: order.text() == "3", timeout=WAIT_MS)
    row = sheet.rows["energy", "build"]
    qtbot.waitUntil(lambda: row.number.number.text() != "", timeout=WAIT_MS)
    assert not row.stack.isEnabled()
    said = sheet._notes["energy"].findChild(QLabel, "solved")
    assert isinstance(said, QLabel)
    assert "solved for soliton order 3" in said.text()
    window.document.undo.undo()
    assert model.get("energy") == "100 uJ"
    window.document.undo.redo()

    # A second target on the guide replaces the first, which keeps what it solved.
    zdw = sheet.derived["zdw"]
    qtbot.waitUntil(lambda: zdw.text() != "…", timeout=WAIT_MS)
    sheet.open_target(numbers["zdw"])
    sheet.target_editor.value.setText("450")
    sheet.target_editor.accept()
    assert isinstance(model.get("pressure", "pins"), dict)
    qtbot.waitUntil(lambda: zdw.text() == "450 nm", timeout=WAIT_MS)
    qtbot.waitUntil(lambda: "pressure" in sheet._solved, timeout=WAIT_MS)
    sheet.open_target(numbers["rdw"])
    sheet.target_editor.moves.setCurrentIndex(1)  # the radius
    sheet.target_editor.value.setText("200")
    sheet.target_editor.accept()
    assert model.get("pressure", "pins") is MISSING
    assert str(model.get("pressure")).endswith("bar")
    assert isinstance(model.get("radius", "pins"), dict)

    # Releasing a target leaves the value it solved.
    qtbot.waitUntil(lambda: "radius" in sheet._solved, timeout=WAIT_MS)
    sheet._release("radius")
    assert model.get("radius", "pins") is MISSING
    assert str(model.get("radius")).endswith(("um", "mm", "m"))

    # A target out of reach says what is attainable.
    sheet.open_target(numbers["zdw"])
    sheet.target_editor.value.setText("100")
    sheet.target_editor.accept()
    qtbot.waitUntil(lambda: "spans" in sheet.ready.text(), timeout=WAIT_MS)
    assert sheet.ready.property("error")


def test_the_curve_is_drawn_and_marked(window: WorkflowWindow, qtbot: QtBot) -> None:
    open_tiny(window, qtbot)
    sheet = window.sheet
    qtbot.waitUntil(lambda: sheet.curve.xs.size > 0, timeout=WAIT_MS)
    assert sheet.marks["pump"].isVisible()
    assert sheet.marks["pump"].value() == pytest.approx(800.0)
    assert sheet.marks["zdw"].isVisible()
    qtbot.waitUntil(lambda: "of output" in sheet.ready.text(), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: sheet.derived["loss"].text() != "…", timeout=WAIT_MS)


def test_the_gas_is_a_list_with_plain_names(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    open_tiny(window, qtbot)
    row = window.sheet.rows["gas", "build"]
    qtbot.waitUntil(
        lambda: (
            row.modes[0] == "choice"
            or window.sheet.rows["gas", "build"].modes[0] == "choice"
        ),
        timeout=WAIT_MS,
    )
    row = window.sheet.rows["gas", "build"]
    assert row.choice.currentText().startswith("Helium")


def test_a_lost_worker_explains_itself(window: WorkflowWindow, qtbot: QtBot) -> None:
    open_tiny(window, qtbot)
    panel = window.run_panel
    panel.run_id = "gone"
    panel.state = ExecutionState.RUNNING
    panel.on_event(Event("gone", "lost", {"exitcode": -9}))
    assert not window.analyse.advice.isHidden()
    assert "out of memory" in window.analyse.advice_text.text()
    assert "LowStorageRungeKutta" in window.analyse.advice_text.text()


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
    assert not window.has_results  # Analyse holds no older results while it goes
    shot(window, tmp_path, "run")
    qtbot.waitUntil(lambda: window.stage is Stage.ANALYSE, timeout=WAIT_MS)
    assert window.primary.text() == "Rerun"
    assert not window.analyse.strip.isVisible()
    assert window.analyse.rail.isVisible()
    scrubber = window.analyse.scrubber
    session = window.session
    qtbot.waitUntil(lambda: bool(scrubber.positions), timeout=WAIT_MS)
    overview = window.overview
    qtbot.waitUntil(lambda: bool(overview.rows), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: bool(window.analyse.buttons), timeout=WAIT_MS)
    assert window.analyse.current == "overview"
    assert not window.analyse.buttons["modes"].isEnabled()
    assert window.analyse.buttons["along"].isEnabled()
    last = len(scrubber.positions) - 1
    assert scrubber.slider.value() == last
    assert f"plane {last + 1} of {last + 1}" in scrubber.text.text()
    scrubber.step(-2)
    assert session.place == last - 2
    assert overview.shown == last - 2
    # Picking on a map moves the scrubber, in the map's unit.
    overview._picked(session.positions[1] * overview._unit[1])
    assert scrubber.slider.value() == 1
    assert session.place == 1
    # Moving between pages keeps the place, and the keys stop at the rail's ends.
    window.analyse.show_page("along")
    assert scrubber.slider.value() == 1
    window.analyse.step_page(-99)
    assert window.analyse.current == "overview"
    shot(window, tmp_path, "analyse")
    window.show_stage(Stage.RUN)
    assert window.analyse.strip.isVisible()
    assert not window.analyse.rail.isVisible()
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
    qtbot.waitUntil(lambda: bool(window.analyse.buttons), timeout=WAIT_MS)
    assert set(window.analyse.buttons) == {p.key for p in window.problem.pages}
    qtbot.waitUntil(lambda: bool(window.overview.rows), timeout=WAIT_MS)


def walk_pages(window: WorkflowWindow, qtbot: QtBot, folder: Path) -> None:
    """Show every page of a file, each drawing what it should."""
    analyse = window.analyse
    views = window.page_views
    qtbot.waitUntil(lambda: bool(analyse.buttons), timeout=WAIT_MS)
    plane = views["plane"]
    analyse.show_page("plane")
    assert isinstance(plane, PlanePage)
    qtbot.waitUntil(lambda: plane.note.text().startswith("Energy"), timeout=WAIT_MS)
    assert plane.spectrum.xs.size > 0
    assert plane.data() is not None
    # The band gives the energy in it, and the pulse with and without it.
    window.session.set_options(bandpass=(600e-9, 900e-9))
    qtbot.waitUntil(lambda: "in the band" in plane.note.text(), timeout=WAIT_MS)
    assert plane.power.names == ["whole spectrum", "in the band"]
    plane.band_low.setText("500")
    plane.band_high.setText("1000")
    plane._band_typed()
    assert window.session.options.bandpass == pytest.approx((500e-9, 1000e-9))
    window.session.set_options(axis="f")
    qtbot.waitUntil(
        lambda: "THz" in plane.spectrum.item.getAxis("bottom").labelText,
        timeout=WAIT_MS,
    )
    plane._region_moved()
    window.session.set_options(axis="wavelength", bandpass=None)
    qtbot.waitUntil(lambda: "in the band" not in plane.note.text(), timeout=WAIT_MS)
    shot(window, folder, "plane")

    spectrogram = views["spectrogram"]
    analyse.show_page("spectrogram")
    assert isinstance(spectrogram, SpectrogramPage)
    qtbot.waitUntil(lambda: "gate" in spectrogram.note.text(), timeout=WAIT_MS)
    assert spectrogram.map.image.image is not None
    assert spectrogram._read(0.0, 800.0)
    shot(window, folder, "spectrogram")

    along = views["along"]
    analyse.show_page("along")
    assert isinstance(along, AlongPage)
    qtbot.waitUntil(lambda: bool(along._plots), timeout=WAIT_MS)
    assert along.data() is not None
    along._log_chosen("on")
    along._log_chosen("off")
    shot(window, folder, "along")

    health = views["health"]
    analyse.show_page("health")
    assert isinstance(health, HealthPage)
    qtbot.waitUntil(lambda: "steps" in health.note.text(), timeout=WAIT_MS)
    assert "Backend" in health.facts.text()
    shot(window, folder, "health")


def test_every_page_of_a_mode_averaged_run_shows_it(
    window: WorkflowWindow, qtbot: QtBot, finished_run: Path, tmp_path: Path
) -> None:
    window.open_results(str(finished_run))
    walk_pages(window, qtbot, tmp_path)
    assert not window.analyse.buttons["modes"].isEnabled()


@pytest.fixture(scope="module")
def two_mode_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a finished run of two modes."""
    from solprop.recipe import Recipe  # noqa: PLC0415 - only the tests that need it

    text = TINY.replace("mode_average = true", "modes = 2")
    path = tmp_path_factory.mktemp("modes") / "run.h5"
    Recipe.loads(text).run(path)
    return path


def test_the_modes_and_compress_pages_on_two_modes(
    window: WorkflowWindow, qtbot: QtBot, two_mode_run: Path, tmp_path: Path
) -> None:
    window.open_results(str(two_mode_run))
    analyse = window.analyse
    qtbot.waitUntil(lambda: bool(analyse.buttons), timeout=WAIT_MS)
    assert analyse.buttons["modes"].isEnabled()
    modes = window.page_views["modes"]
    assert isinstance(modes, ModesPage)
    analyse.show_page("modes")
    qtbot.waitUntil(lambda: bool(modes._read_stats), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: modes.spectra.xs.size > 0, timeout=WAIT_MS)
    assert len(modes.spectra.ys) == 2
    item = modes.choice.item(1)
    assert item is not None
    item.setCheckState(Qt.CheckState.Unchecked)
    assert modes.spectra.names == [window.session.labels[0]]
    shot(window, tmp_path, "modes")

    compress = window.page_views["compress"]
    assert isinstance(compress, CompressPage)
    analyse.show_page("compress")
    chain = compress.chain
    assert "plane" in chain.target.text()
    chain.add("taylor")
    chain.compress()
    qtbot.waitUntil(lambda: chain.status.text().startswith("Done"), timeout=WAIT_MS)
    shot(window, tmp_path, "compress")


def test_copy_as_python_runs(
    window: WorkflowWindow, qtbot: QtBot, two_mode_run: Path, tmp_path: Path
) -> None:
    window.open_results(str(two_mode_run))
    analyse = window.analyse
    qtbot.waitUntil(lambda: bool(analyse.buttons), timeout=WAIT_MS)
    for key in (
        "overview",
        "plane",
        "spectrogram",
        "along",
        "modes",
        "compress",
        "health",
    ):
        analyse.show_page(key)
        text = analyse.python_text()
        assert text is not None, key
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(text, key, "exec"), {})
    matplotlib.pyplot.close("all")
    analyse.copy_python()
    assert "open_file" in analyse.said.text() or "Copied" in analyse.said.text()
    path = tmp_path / "data.csv"
    analyse.show_page("along")
    along = window.page_views["along"]
    assert isinstance(along, AlongPage)
    qtbot.waitUntil(lambda: bool(along._plots), timeout=WAIT_MS)
    analyse.export_data(str(path))
    assert path.read_text("utf-8").startswith("# z (m)")
    analyse.show_page("overview")
    analyse.export_figure(str(tmp_path / "maps.png"))
    qtbot.waitUntil(lambda: "maps" in analyse.said.text(), timeout=WAIT_MS)
    assert list(tmp_path.glob("maps*.png"))


def test_a_file_without_a_recipe_gets_the_generic_rail(
    window: WorkflowWindow,
) -> None:
    window.session._describe({"z": [0.0, 1.0], "field_planes": [0, 1], "recipe": False})
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
    assert not window.analyse.advice.isHidden()
    assert window.has_results  # the planes reached stay open for analysis
    window._run_state("starting")
    assert window.analyse.advice.isHidden()


def test_a_page_that_a_file_cannot_show_says_why(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    stage = AnalyseStage(window.session, None, {})
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
