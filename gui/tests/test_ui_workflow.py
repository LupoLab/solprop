# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workflow window, offscreen: Home, Setup, Run and Analyse, for every class."""

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QStandardPaths, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QBoxLayout, QLabel, QLineEdit, QWidget
from pytestqt.qtbot import QtBot
from solprop.constants import C
from solprop.output import open_file

from conftest import TINY
from solprop_gui.classes.catalog import (
    ANTIRESONANT,
    CLASSES,
    FILAMENT,
    GENERIC_PAGES,
    SOLID_CORE,
)
from solprop_gui.classes.problem import AnalysisPage, ProblemClass
from solprop_gui.classes.workflow import Stage
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.service.messages import Event
from solprop_gui.service.states import ExecutionState
from solprop_gui.ui import theme as ui_theme
from solprop_gui.ui.analyse import AnalyseStage
from solprop_gui.ui.analysis_pages import (
    AlongPage,
    BeamOverviewPage,
    BeamPlanePage,
    CompressPage,
    FarFieldPage,
    HealthPage,
    ModesPage,
    OnAxisPage,
    OverviewPage,
    PlanePage,
    SpectrogramPage,
)
from solprop_gui.ui.design_stage import DesignStage
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.quantity import QuantityField, _readable
from solprop_gui.ui.theme import FONT
from solprop_gui.ui.workflow_window import WorkflowWindow
from solprop_gui.views.cross_section import CrossSection

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
    assert window.home.tiles["design"].isEnabled()
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
        row.path
        for section in problem.sections
        for row in section.fields
        if row.member is None
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
    assert sheet.marks["Pump"].isVisible()
    assert sheet.marks["Pump"].value() == pytest.approx(800.0)
    assert sheet.marks["ZDW"].isVisible()
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
    qtbot.waitUntil(lambda: "gate" in spectrogram.note.text().lower(), timeout=WAIT_MS)
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


@pytest.fixture(scope="module")
def filament_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a finished short radial run with a plasma, with the filament's recipe."""
    from solprop.recipe import Recipe  # noqa: PLC0415 - only the tests that need it

    text = Recipe.example("filament").to_text()
    # Far fewer planes and nodes, over a short distance: the pages, not a filament.
    build: Any = Recipe.loads(text).to_dict()
    build["build"]["grid"]["length"] = "1 cm"
    build["build"]["space"]["n_r"] = 32
    build["build"]["settings"]["save_planes"] = ["0.5 cm", "1 cm"]
    build["build"]["settings"]["dz_init"] = "1 mm"
    recipe = Recipe.from_dict(build)
    path = tmp_path_factory.mktemp("filament") / "run.h5"
    recipe.run(path)
    return path


def test_the_filament_s_sheet_links_its_rows(  # noqa: PLR0915 - one story
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    window.begin_class("filament")
    sheet = window.sheet
    qtbot.waitUntil(lambda: sheet.problem is FILAMENT, timeout=WAIT_MS)
    ratio = sheet.derived["power_ratio"]
    qtbot.waitUntil(lambda: ratio.text() == "3", timeout=WAIT_MS)
    qtbot.waitUntil(lambda: sheet.marks != {}, timeout=WAIT_MS)
    model = window.document.model
    assert model is not None
    # A template arrives for the switches.
    qtbot.waitUntil(lambda: bool(sheet._example), timeout=WAIT_MS)
    gas = sheet.rows["medium.material", "build"]
    assert gas.choice.currentText().startswith("Argon")
    gas.choice.setCurrentIndex(gas.choice.findData("Kr"))
    assert model.get("responses[1].rate.source") == "Kr"
    qtbot.waitUntil(lambda: ratio.text() != "3", timeout=WAIT_MS)
    window.document.undo.undo()
    assert model.get("responses[1].rate.source") == "Ar"
    box = sheet.rows["initial.propz", "build"].number.number
    box.setText("10")
    box.textEdited.emit("10")
    assert str(model.get("initial.propz")).startswith("-10")
    assert box.text() == "10"
    window.document.undo.undo()
    switches = {field.member: check for field, check in sheet._switches}
    plasma = switches["EnvelopePlasmaResponse"]
    assert plasma.isChecked()
    plasma.click()
    assert len(model.get("responses")) == 1  # pyright: ignore[reportArgumentType]
    assert model.get("settings.ionisation_field_range") is MISSING
    plasma.click()
    assert len(model.get("responses")) == 2  # pyright: ignore[reportArgumentType]
    planes = sheet.rows["settings.save_planes", "build"].number.number
    planes.setText("10")
    planes.textEdited.emit("10")
    held = model.get("settings.save_planes")
    assert isinstance(held, list)
    assert len(held) == 10
    sheet.apply_lens(50e-6, 0.1)
    assert str(model.get("initial.waist")).endswith("um")
    assert str(model.get("initial.propz")).startswith("-")
    sheet.lens_editor.focal.number.setText("x")
    sheet.lens_editor.accept()
    assert sheet.lens_editor.problem.text()
    sheet.lens_editor.focal.number.setText("50")
    sheet.lens_editor.radius.number.setText("0")
    sheet.lens_editor.accept()
    assert "positive" in sheet.lens_editor.problem.text()
    sheet.lens_editor.radius.number.setText("2")
    with qtbot.waitSignal(sheet.lens_wanted) as lens:
        sheet.lens_editor.accept()
    assert lens.args == [pytest.approx(0.5), pytest.approx(2e-3)]
    window._ask_lens(0.5, 2e-3)
    window.document.undo.undo()


def test_every_page_of_a_filament_shows_it(  # noqa: PLR0915 - one story
    window: WorkflowWindow, qtbot: QtBot, filament_run: Path, tmp_path: Path
) -> None:
    window.open_results(str(filament_run))
    analyse = window.analyse
    qtbot.waitUntil(lambda: bool(analyse.buttons), timeout=WAIT_MS)
    assert window.problem is FILAMENT
    assert isinstance(window.overview, BeamOverviewPage)
    views = window.page_views
    overview = views["beam_overview"]
    assert isinstance(overview, BeamOverviewPage)
    qtbot.waitUntil(lambda: overview._data is not None, timeout=WAIT_MS)
    qtbot.waitUntil(
        lambda: overview.across_map.image.image is not None, timeout=WAIT_MS
    )
    assert overview.spectrum_title.text() == "On-axis spectrum along z"
    assert overview.data() is not None
    assert overview.figure() is None
    assert overview._read_fluence(0.0, 10.0)
    overview._scale_chosen("linear")
    overview._fluence_floor_moved(-20)
    overview._scale_chosen("dB")
    overview._fluence_floor_moved(-25)
    overview._floor_moved(-50)
    shot(window, tmp_path, "beam-overview")
    analyse.scrubber.step(-1)
    assert overview.shown == window.session.place
    # On frequency, it asks again.
    window.session.set_options(axis="f")
    qtbot.waitUntil(lambda: (overview._data or {}).get("axis") == "f", timeout=WAIT_MS)
    window.session.set_options(axis="wavelength")
    qtbot.waitUntil(
        lambda: (overview._data or {}).get("axis") == "wavelength", timeout=WAIT_MS
    )

    plane = views["beam_plane"]
    assert isinstance(plane, BeamPlanePage)
    analyse.show_page("plane")
    qtbot.waitUntil(lambda: plane.profile.xs.size > 0, timeout=WAIT_MS)
    assert plane.data() is not None
    plane._reach_moved(300)
    plane._window_moved(40)
    shot(window, tmp_path, "beam-plane")

    axis = views["on_axis"]
    assert isinstance(axis, OnAxisPage)
    analyse.show_page("axis")
    qtbot.waitUntil(lambda: axis.pulse.xs.size > 0, timeout=WAIT_MS)
    assert axis.map.image.image is not None
    shot(window, tmp_path, "beam-axis")

    far = views["far_field"]
    assert isinstance(far, FarFieldPage)
    analyse.show_page("farfield")
    qtbot.waitUntil(lambda: far._db.size > 0, timeout=WAIT_MS)
    assert far._read(800.0, 1.0)
    shot(window, tmp_path, "beam-far")

    along = views["along"]
    assert isinstance(along, AlongPage)
    analyse.show_page("along")
    qtbot.waitUntil(lambda: bool(along._plots), timeout=WAIT_MS)
    analyse.set_reference(0.005)
    assert "Marburger" in along.note.text()
    analyse.set_reference(None)

    # Each page can say the Python that reproduces it.
    for key in ("overview", "plane", "axis", "farfield", "health"):
        analyse.show_page(key)
        text = analyse.python_text()
        assert text is not None, key
        exec(compile(text, key, "exec"), {})


def test_a_filament_runs_on_its_own_overview(
    window: WorkflowWindow, qtbot: QtBot, tmp_path: Path, filament_run: Path
) -> None:
    from solprop.recipe import Recipe  # noqa: PLC0415 - only the tests that need it

    with open_file(str(filament_run)) as output:
        assert output is not None
    text = Recipe.from_output(str(filament_run)).to_text()
    window.document.set_model(RecipeModel(text), base=None)
    sheet = window.sheet
    qtbot.waitUntil(lambda: sheet.problem is FILAMENT, timeout=WAIT_MS)
    qtbot.waitUntil(window.primary.isEnabled, timeout=WAIT_MS)
    assert window.run_panel.view is window.overview
    window.primary.click()
    qtbot.waitUntil(lambda: window.stage is Stage.RUN, timeout=WAIT_MS)
    overview = window.overview
    assert isinstance(overview, BeamOverviewPage)
    qtbot.waitUntil(lambda: window.stage is Stage.ANALYSE, timeout=WAIT_MS)
    assert overview.rows
    qtbot.waitUntil(lambda: overview._data is not None, timeout=WAIT_MS)
    window.run_panel.reread()
    # What a run draws on before the file is read: its rows.
    overview.clear()
    assert overview.fluence_map.image.image is None
    assert not overview._gather()
    overview.end_of_run()


def test_an_antiresonant_fibre_draws_its_loss_and_cross_section(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    window.begin_class("antiresonant")
    sheet = window.sheet
    qtbot.waitUntil(lambda: sheet.problem is ANTIRESONANT, timeout=WAIT_MS)
    qtbot.waitUntil(sheet.below.isVisible, timeout=WAIT_MS)
    assert sheet.loss_plot.xs.size > 0
    assert sheet.loss_mark.value() == pytest.approx(800.0)
    assert any(shape["kind"] == "ring" for shape in sheet.cross_section.shapes)
    sheet.cross_section.grab()
    # Only the order is offered as a target: a fibre has many zeros.
    numbers = {derived.key: derived for derived in ANTIRESONANT.derived}
    assert numbers["soliton_order"].target is not None
    assert numbers["zdw"].target is None
    assert numbers["rdw"].target is None


def test_a_cross_section_paints_each_kind_of_shape(qtbot: QtBot) -> None:
    drawing = CrossSection()
    qtbot.addWidget(drawing)
    drawing.resize(200, 200)
    drawing.grab()
    colours = {"fill": "#eeeeee", "edge": "#333333"}
    drawing.set_shapes(
        [
            {"kind": "box", "lower": [-2e-6, -1e-6], "upper": [2e-6, 1e-6], **colours},
            {"kind": "disc", "centre": [0.0, 0.0], "radius": 1e-6, **colours},
            {
                "kind": "ring",
                "centre": [1e-6, 0.0],
                "inner": 2e-7,
                "outer": 3e-7,
                **colours,
            },
        ]
    )
    image = drawing.grab().toImage()
    assert image.pixelColor(100, 100).name() == "#eeeeee"


def test_a_solid_core_fibre_runs_and_every_page_reads_it(  # noqa: PLR0915 - one story
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    window.begin_class("solid-core")
    sheet = window.sheet
    qtbot.waitUntil(lambda: sheet.problem is SOLID_CORE, timeout=WAIT_MS)
    order = sheet.derived["soliton_order"]
    qtbot.waitUntil(lambda: order.text() == "3", timeout=WAIT_MS)
    qtbot.waitUntil(lambda: sheet.marks != {}, timeout=WAIT_MS)
    assert not sheet.below.isVisible()
    # The sheet's rows hold the two places the recipe writes a length.
    model = window.document.model
    assert model is not None
    length = sheet.rows["length", "build"].number.number
    assert length.text() == "1.327808"
    assert model.get("length") == model.get("grid.length")
    length.setText("0.1")
    length.textEdited.emit("0.1")
    assert str(model.get("grid.length")).startswith("0.1")
    window.document.undo.undo()
    qtbot.waitUntil(lambda: order.text() == "3", timeout=WAIT_MS)
    qtbot.waitUntil(lambda: window.progress().can_run, timeout=WAIT_MS)
    window.analyse.current = "health"
    window._use_class()
    assert window.analyse.current is None
    window.primary.click()
    qtbot.waitUntil(lambda: window.stage is Stage.ANALYSE, timeout=WAIT_MS)
    analyse = window.analyse
    qtbot.waitUntil(lambda: analyse.current == "overview", timeout=WAIT_MS)
    overview = window.page_views["overview"]
    assert isinstance(overview, OverviewPage)
    qtbot.waitUntil(lambda: overview._power.size > 0, timeout=WAIT_MS)
    assert overview._power_unit == "W"
    assert "power (W)" in overview.data()[0]  # pyright: ignore[reportOptionalSubscript]
    # The map keeps to where the light is, and that is well inside the grid.
    low, high = overview.spectrum_map.box.viewRange()[0]
    assert low > 500.0
    assert high < 2000.0
    assert not analyse.buttons["noise"].isEnabled()
    analyse.show_page("plane")
    plane = window.page_views["plane"]
    assert isinstance(plane, PlanePage)
    qtbot.waitUntil(lambda: plane.note.text().startswith("Energy"), timeout=WAIT_MS)
    assert plane.power.unit == "W"
    # A failed read leaves nothing of the last file behind it.
    plane._failed("It could not be read")
    assert plane.note.text() == "It could not be read"
    assert plane.power.xs.size == 0
    analyse.show_page("spectrogram")
    spectrogram = window.page_views["spectrogram"]
    qtbot.waitUntil(lambda: spectrogram.note.text().startswith("Gate"), timeout=WAIT_MS)
    analyse.show_page("along")
    along = window.page_views["along"]
    assert isinstance(along, AlongPage)
    qtbot.waitUntil(lambda: bool(along._stats), timeout=WAIT_MS)
    assert "energy" in along._stats
    assert "peak_power" in along._stats
    assert along.note.text().endswith("rejected.")


def test_a_flat_phase_is_drawn_flat(window: WorkflowWindow, qtbot: QtBot) -> None:
    window.begin_class("capillary")
    plane = window.page_views["plane"]
    assert isinstance(plane, PlanePage)
    omega = np.array([2.0, 3.0]) * 1e15
    x = np.sort(C / (omega / (2.0 * np.pi)) * 1e9)
    plane._show_phase(
        {"phase": {"omega": omega, "y": np.array([5.0, 5.0 + 1e-9])}, "window": None},
        "wavelength",
        x,
        np.zeros(2),
    )
    assert plane.phase.box.viewRange()[1] == pytest.approx([4.0, 6.0], abs=0.1)


def test_a_box_shows_a_long_number_short_and_keeps_it_exact(qtbot: QtBot) -> None:
    field = QuantityField("m", "m")
    qtbot.addWidget(field)
    edits: list[object] = []
    field.changed.connect(edits.append)
    exact = "1.3278075458959395 m"
    field.set_value(exact)
    assert field.number.text() == "1.327808"
    # Reading and leaving it writes nothing, and rounds nothing.
    assert field.read() == exact
    field._finished()
    field._typed("1.327808")
    assert edits == []
    # A unit keeps the exact value, not what the box shows.
    units = field.units
    assert units is not None
    units.setCurrentIndex(units.findData("mm"))
    assert field.number.text() == "1327.808"
    assert str(edits[-1]).startswith("1327.8075458959")
    # What is typed is taken as typed.
    field.number.setText("2.5")
    field._finished()
    assert str(edits[-1]).startswith("2.5")
    assert _readable("not a number at all") == "not a number at all"


def design_stage(window: WorkflowWindow, qtbot: QtBot) -> DesignStage:
    window.home.tiles["design"].clicked.emit()
    stage = window.design
    qtbot.waitUntil(lambda: stage.text is not None, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: bool(stage._curves), timeout=WAIT_MS)
    return stage


def test_a_design_answers_with_no_pulse_and_touches_no_setup(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    window.begin_class("capillary")
    qtbot.waitUntil(lambda: window.stage is Stage.SETUP, timeout=WAIT_MS)
    model = window.document.model
    stage = design_stage(window, qtbot)
    assert window.stage is Stage.DESIGN
    assert window.document.model is model
    assert window.primary.text() == "Simulate this"
    assert window.context.text() == "Design a waveguide"
    assert not window.stage_buttons[Stage.SETUP].isVisible()
    assert not window.overflow.isVisible()
    qtbot.waitUntil(lambda: stage.numbers["zdw"].text() == "471 nm", timeout=WAIT_MS)
    # Rows of what the design does not hold stay out of the way.
    assert not stage.fields["energy"].isVisible()
    assert not stage.fields["lattice.n_tubes"].isVisible()
    assert not stage.numbers["soliton_order"].isVisible()
    window.show_stage(Stage.HOME)
    assert window.stage_buttons[Stage.SETUP].isVisible()


def test_the_zero_is_solved_with_the_order_and_the_energy(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    stage = design_stage(window, qtbot)
    stage.solve()
    assert "Give the zero" in stage.note.text()
    stage.zdw_target.number.setText("450")
    stage.solve()
    pressure = stage.fields["pressure"].number
    qtbot.waitUntil(lambda: pressure.text().startswith("0.83"), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: stage.numbers["zdw"].text() == "450 nm", timeout=WAIT_MS)
    assert "Solved" in stage.note.text()
    stage.order_target.number.setText("3")
    stage.solve()
    qtbot.waitUntil(stage.pulse.isChecked, timeout=WAIT_MS)
    order = stage.numbers["soliton_order"]
    qtbot.waitUntil(lambda: order.text().startswith("3"), timeout=WAIT_MS)
    assert stage.fields["energy"].isVisible()


def test_a_sweep_draws_a_curve_each_and_a_click_takes_one(
    window: WorkflowWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    stage = design_stage(window, qtbot)
    # A click with nothing swept changes nothing.
    stage._picked(600.0, 0.0)
    assert stage.fields["pressure"].number.text() == "1"
    stage.sweep_name.setCurrentIndex(stage.sweep_name.findData("pressure"))
    qtbot.waitUntil(lambda: len(stage._curves) == 4, timeout=WAIT_MS)
    assert stage.note.text().startswith("Click")
    at = int(np.argmin(np.abs(stage._x - 600.0)))
    stage._picked(600.0, float(stage._curves[3][1][at]))
    assert stage.fields["pressure"].read() == "5 bar"
    path = tmp_path / "curves.csv"
    stage.write_csv(path)
    assert path.read_text().splitlines()[0].endswith("5 bar")
    stage.sweep_name.setCurrentIndex(stage.sweep_name.findData("gas"))
    qtbot.waitUntil(lambda: len(stage._curves) == 3, timeout=WAIT_MS)
    at = int(np.argmin(np.abs(stage._x - 300.0)))
    stage._picked(300.0, float(stage._curves[2][1][at]))
    assert stage.gas.currentData() == "Ar"
    assert stage.gas.currentText() == "Argon"


def test_the_other_pages_draw_and_a_map_point_sets_the_design(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    stage = design_stage(window, qtbot)
    stage.show_page("loss")
    qtbot.waitUntil(lambda: stage.plot.unit == "dB/m", timeout=WAIT_MS)
    assert stage.plot.log
    assert not stage.quantity.isVisible()
    stage.show_page("phase")
    qtbot.waitUntil(lambda: stage.plot.unit == "1/cm", timeout=WAIT_MS)
    assert stage.note.text().startswith("Linear")
    stage.pulse.setChecked(True)
    qtbot.waitUntil(lambda: stage.note.text().startswith("Phase"), timeout=WAIT_MS)
    wave = stage.numbers["rdw"]
    qtbot.waitUntil(lambda: wave.text() not in ("…", "—"), timeout=WAIT_MS)
    assert stage.marks.lines["Dispersive wave"].isVisible()
    stage.show_page("map")
    qtbot.waitUntil(lambda: stage.plot.unit == "bar", timeout=WAIT_MS)
    assert stage.note.text().startswith("Click")
    assert stage._curves[0][0].startswith("ZDW")
    assert np.size(stage.design_dot.getData()[0]) == 1
    stage._picked(100.0, 0.0)
    assert stage.fields["radius"].read() == "100 um"
    assert stage.fields["pressure"].read() == "1 bar"
    # New numbers ask for the map again, at the new design.
    qtbot.waitUntil(lambda: abs(stage._x[0] - 50.0) < 1e-6, timeout=WAIT_MS)
    stage.quantity.setCurrentIndex(stage.quantity.findData("beta3"))
    stage.show_page("curves")
    qtbot.waitUntil(lambda: stage.plot.unit == "fs³/m", timeout=WAIT_MS)
    assert np.size(stage.design_dot.getData()[0]) == 0


def test_a_fibre_has_its_own_rows_and_no_map(
    window: WorkflowWindow, qtbot: QtBot
) -> None:
    stage = design_stage(window, qtbot)
    stage.show_page("map")
    qtbot.waitUntil(lambda: stage.plot.unit == "bar", timeout=WAIT_MS)
    stage.guide_choice._pick("antiresonant")
    assert stage.page == "curves"
    assert stage.fields["lattice.n_tubes"].isVisible()
    assert not stage.solve_button.isEnabled()
    zero = stage.numbers["zdw"]
    qtbot.waitUntil(lambda: zero.text() == "483 nm", timeout=WAIT_MS)
    stage.show_page("map")
    assert stage.page == "curves"
    assert "capillary" in stage.note.text()
    window.primary.click()
    qtbot.waitUntil(lambda: window.stage is Stage.SETUP, timeout=WAIT_MS)
    assert window.problem is ANTIRESONANT
    model = window.document.model
    assert model is not None
    assert model.get("radius") == "15 um"


def test_a_failed_request_says_why(window: WorkflowWindow, qtbot: QtBot) -> None:
    stage = design_stage(window, qtbot)
    box = stage.fields["pressure"].number
    box.setText("-1")
    box.textEdited.emit("-1")
    stage.show_page("curves")
    qtbot.waitUntil(lambda: bool(stage.note.text()), timeout=WAIT_MS)


def test_a_design_with_no_recipe_yet_simulates_nothing(
    window: WorkflowWindow,
) -> None:
    window.design.text = None
    window.show_stage(Stage.DESIGN)
    window.simulate_design()
    assert window.document.model is None
