# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workflow window: Home, then Setup, Run and Analyse (design 5.1).

A problem class decides what is on screen, so nothing here asks which class it is
drawing. The window holds a :class:`~solprop_gui.ui.document.SetupDocument` for the open
setup and moves between the stages; each stage only shows. It is built beside the step
window, which stays runnable until every route has moved here.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.catalog import GENERIC_PAGES, class_named, class_of
from solprop_gui.classes.problem import ProblemClass
from solprop_gui.classes.workflow import Progress, Stage, primary_action, stage_open
from solprop_gui.model.recipe import RecipeModel
from solprop_gui.model.workspace import DEFAULT_WORKSPACE, list_runs
from solprop_gui.service.messages import Reply
from solprop_gui.ui.analyse import AnalyseStage
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.document import Confirm, SetupDocument
from solprop_gui.ui.home import HomePage
from solprop_gui.ui.main_window import MainWindow, app_settings
from solprop_gui.ui.results import ResultsPage
from solprop_gui.ui.run_panel import RunPanel
from solprop_gui.ui.run_stage import RunStage
from solprop_gui.ui.sheet import SheetStage
from solprop_gui.ui.theme import STYLE, load_font, named_label, repolish

__all__ = ["WorkflowWindow"]

_RECENT_SETUPS = "recent-setups"
"""The setting that lists recent setups."""

_CURVE_KEYS = ("beta2", "peak_power", "gamma", "fwhm", "shape", "wavelength")
"""What the curve's request reads back from the numbers'."""

_STAGE_TEXT = ((Stage.SETUP, "Setup"), (Stage.RUN, "Run"), (Stage.ANALYSE, "Analyse"))


class WorkflowWindow(QMainWindow):
    """The application's window, in its workflow form.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge or None, optional
        The service; ``None`` makes one, which starts the kernel on the first request.
    workspace : pathlib.Path or None, optional
        The run workspace; ``None`` takes the saved one, or the default.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owning widget; None for a top-level window.

    Attributes
    ----------
    confirm : callable
        How the window asks before losing work; replaceable, so that tests answer
        without a dialog.
    stage : Stage
        Where the window is.
    problem : ProblemClass or None
        The open setup's class.
    has_run : bool
        Whether a run has been started from the open setup.
    has_results : bool
        Whether a results file is open.
    """

    def __init__(  # noqa: PLR0915 - the window's parts
        self,
        bridge: ServiceBridge | None = None,
        workspace: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Make the window, on Home."""
        super().__init__(parent)
        load_font()
        self.setStyleSheet(STYLE)
        self.settings = app_settings()
        self.bridge = bridge or ServiceBridge(self)
        self.confirm: Confirm = self._ask
        self.stage = Stage.HOME
        self.problem: ProblemClass | None = None
        self.has_run = False
        self.has_results = False
        self._capabilities: Mapping[str, object] = {}
        self._quiet = False
        self._own_run = False
        self._failed = False
        self._pending: dict[int, str] = {}
        self.step_window: MainWindow | None = None
        saved = self.settings.value("workspace")
        self.workspace = workspace or (Path(str(saved)) if saved else DEFAULT_WORKSPACE)

        self.document = SetupDocument(
            self.bridge,
            lambda title, text: self.confirm(title, text),  # noqa: PLW0108 - replaceable
            self,
        )
        self.run_panel = RunPanel(self.bridge, self.document.snapshot, self.workspace)
        self.results = ResultsPage(self.bridge)
        self.home = HomePage()
        self.sheet = SheetStage()
        self.run_stage = RunStage(self.run_panel)
        self.analyse = AnalyseStage(
            {
                "overview": self.results.overview,
                "plane": self.results.plane,
                "along": self.results.statistics,
                "health": self.run_panel.log,
            }
        )
        # One scrubber: the views' own sliders follow it, out of sight.
        self.results.overview.slider.hide()
        self.results.overview.position.hide()
        self.results.overview.follow.hide()
        self.results.plane.slider.hide()
        self.results.plane.plane_label.hide()
        self.analyse.scrubber.moved.connect(self._plane_moved)
        self.results.overview.slider.valueChanged.connect(self._overview_moved)

        self.stack = QStackedWidget()
        self.screens: dict[Stage, QWidget] = {
            Stage.HOME: self.home,
            Stage.SETUP: self.sheet,
            Stage.RUN: self.run_stage,
            Stage.ANALYSE: self.analyse,
        }
        for screen in self.screens.values():
            self.stack.addWidget(screen)
        self.drawer = self._make_drawer()
        self.top = self._make_top()
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        middle.addWidget(self.stack, 1)
        middle.addWidget(self.drawer)
        root = QWidget()
        root.setObjectName("page")
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.top)
        column.addLayout(middle, 1)
        self.setCentralWidget(root)
        self._make_menus()

        self.home.class_chosen.connect(self.begin_class)
        self.home.other_chosen.connect(self.open_step_window)
        self.home.open_results_wanted.connect(self.choose_results)
        self.home.open_setup_wanted.connect(self.choose_setup)
        self.home.results_chosen.connect(self.open_results)
        self.home.setup_chosen.connect(self.open_setup)
        self.sheet.edited.connect(
            lambda name, table, value: self.document.edit(
                name, table, value, merging=True
            )
        )
        self.sheet.pins_changed.connect(self.document.replace_pins)
        self.sheet.example_chosen.connect(self.document.open_example)
        self.document.opened.connect(self._opened)
        self.document.described.connect(lambda _route: self._build_sheet())
        self.document.loaded.connect(self._loaded)
        self.document.values_changed.connect(self._show_values)
        self.document.state_changed.connect(self._refresh)
        self.document.checked.connect(self._checked)
        self.document.summarised.connect(self._summarised)
        self.document.message.connect(self._say)
        self.run_panel.report = self._report_failure
        self.run_panel.state_changed.connect(self._run_state)
        self.run_panel.ended.connect(self._run_ended)
        self.results.described.connect(self._file_described)
        self.results.setup_opened.connect(self._file_has_setup)
        self.bridge.replied.connect(self._on_reply)

        self.setWindowTitle("solprop")
        self.resize(1280, 800)
        self._numbers: Mapping[str, object] = {}
        self._ask_kernel("examples", {}, "examples")
        self._ask_kernel("gases", {}, "gases")
        self.refresh_home()
        self.show_stage(Stage.HOME)

    # ------------------------------------------------------------------ the frame

    def _make_top(self) -> QFrame:
        """Make the bar at the top: where you are, the stages, what comes next.

        Returns
        -------
        PySide6.QtWidgets.QFrame
            The bar.
        """
        top = QFrame()
        top.setObjectName("top")
        top.setFixedHeight(52)
        row = QHBoxLayout(top)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(12)
        brand = QPushButton("solprop")
        brand.setObjectName("brand")
        brand.setCursor(Qt.CursorShape.PointingHandCursor)
        brand.setToolTip("Home")
        brand.clicked.connect(lambda: self.show_stage(Stage.HOME))
        self.context = named_label("", "context")
        row.addWidget(brand)
        row.addWidget(self.context)
        row.addStretch(1)
        self.stage_buttons: dict[Stage, QPushButton] = {}
        for stage, text in _STAGE_TEXT:
            button = QPushButton(text)
            button.setObjectName("stage")
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _on=False, s=stage: self.show_stage(s))
            self.stage_buttons[stage] = button
            row.addWidget(button)
        row.addStretch(1)
        self.primary = QPushButton("Run")
        self.primary.setObjectName("primary")
        self.primary.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.primary.clicked.connect(self.primary_clicked)
        row.addWidget(self.primary)
        self.overflow = QToolButton()
        self.overflow.setObjectName("overflow")
        self.overflow.setText("⋯")
        self.overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.overflow)
        menu.addAction("Save", self.document.save)
        # Through a lambda: `triggered` would pass its checked state as the path.
        menu.addAction("Save as…", lambda: self.document.save_as())  # noqa: PLW0108
        menu.addSeparator()
        menu.addAction("Show the recipe", self.show_recipe)
        menu.addAction("Show as Python", self.show_python)
        self.overflow.setMenu(menu)
        row.addWidget(self.overflow)
        return top

    def _make_drawer(self) -> QFrame:
        """Make the drawer that shows the recipe or its Python on request.

        Returns
        -------
        PySide6.QtWidgets.QFrame
            The drawer, hidden.
        """
        drawer = QFrame()
        drawer.setObjectName("drawer")
        drawer.setFixedWidth(440)
        self.drawer_title = named_label("", "h3")
        close = QPushButton("Close")
        close.setObjectName("plain")
        close.clicked.connect(drawer.hide)
        head = QHBoxLayout()
        head.addWidget(self.drawer_title, 1)
        head.addWidget(close)
        self.drawer_text = QPlainTextEdit()
        self.drawer_text.setObjectName("drawerText")
        self.drawer_text.setReadOnly(True)
        column = QVBoxLayout(drawer)
        column.setContentsMargins(14, 10, 10, 10)
        column.addLayout(head)
        column.addWidget(self.drawer_text, 1)
        drawer.hide()
        return drawer

    def _make_menus(self) -> None:
        """Repeat the actions, with their shortcuts, in the platform's menu bar."""
        file_menu = self.menuBar().addMenu("&File")
        home = QAction("&Home", self)
        home.triggered.connect(lambda: self.show_stage(Stage.HOME))
        file_menu.addAction(home)
        open_setup = QAction("&Open a setup…", self)
        open_setup.setShortcut(QKeySequence.StandardKey.Open)
        open_setup.triggered.connect(self.choose_setup)
        file_menu.addAction(open_setup)
        open_results = QAction("Open &results…", self)
        open_results.triggered.connect(self.choose_results)
        file_menu.addAction(open_results)
        save = QAction("&Save", self)
        save.setShortcut(QKeySequence.StandardKey.Save)
        save.triggered.connect(self.document.save)
        file_menu.addAction(save)
        save_as = QAction("Save &as…", self)
        save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        # Through a lambda: `triggered` would pass its checked state as the path.
        save_as.triggered.connect(lambda: self.document.save_as())  # noqa: PLW0108
        file_menu.addAction(save_as)
        edit_menu = self.menuBar().addMenu("&Edit")
        undo = self.document.undo.createUndoAction(self, "&Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(undo)
        redo = self.document.undo.createRedoAction(self, "&Redo")
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(redo)

    def progress(self) -> Progress:
        """Return how far the simulation has got.

        Returns
        -------
        Progress
            What the stages and the primary action follow.
        """
        return Progress(
            has_setup=self.document.model is not None and self.problem is not None,
            can_run=self.document.can_run,
            running=self.run_panel.active,
            has_run=self.has_run,
            has_results=self.has_results,
        )

    def _refresh(self) -> None:
        """Bring the bar, the stage buttons and the title up to date."""
        progress = self.progress()
        # From Home, the stages lead back to what is open.
        self.top.setVisible(
            self.stage is not Stage.HOME or progress.has_setup or progress.has_results
        )
        for stage, button in self.stage_buttons.items():
            button.setEnabled(stage_open(stage, progress))
            button.setProperty("current", stage is self.stage)
            repolish(button)
        primary = primary_action(progress)
        self.primary.setText(primary.text)
        self.primary.setEnabled(primary.enabled)
        model = self.document.model
        name = None if model is None else model.title
        title = None if self.problem is None else self.problem.title
        self.context.setText(" · ".join(part for part in (title, name) if part))
        if model is None:
            self.setWindowTitle("solprop")
            return
        self.setWindowTitle(f"{name or 'Untitled setup'}[*] — solprop")
        self.setWindowModified(model.dirty)

    def show_stage(self, stage: Stage) -> None:
        """Go to a stage.

        Parameters
        ----------
        stage : Stage
            Where to go.
        """
        self.stage = stage
        if stage is Stage.HOME:
            self.refresh_home()
        self.stack.setCurrentWidget(self.screens[stage])
        self._refresh()

    def primary_clicked(self) -> None:
        """Do what the primary button says."""
        action = primary_action(self.progress()).action
        if action == "stop":
            self.run_panel.stop()
        else:
            self.run_panel.set_can_run(self.document.can_run)
            self.run_panel.start()

    def _say(self, text: str) -> None:
        """Tell the user something, on the Ready line.

        Parameters
        ----------
        text : str
            What to say.
        """
        self.sheet.ready.setText(text)

    def _ask(self, title: str, text: str) -> str:
        """Ask before losing work.

        Parameters
        ----------
        title : str
            The dialog's title.
        text : str
            The question.

        Returns
        -------
        str
            ``"save"``, ``"discard"``, ``"stop"`` or ``"cancel"``.
        """
        box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=self)
        stop = "run is going" in text
        answers = {
            box.addButton(
                "Stop it" if stop else "Save", QMessageBox.ButtonRole.AcceptRole
            ): ("stop" if stop else "save"),
            box.addButton(QMessageBox.StandardButton.Cancel): "cancel",
        }
        if not stop:
            answers[box.addButton(QMessageBox.StandardButton.Discard)] = "discard"
        box.exec()
        return answers.get(box.clickedButton(), "cancel")  # pyright: ignore[reportArgumentType, reportCallIssue]

    # ------------------------------------------------------------------ the kernel

    def _ask_kernel(
        self, kind: str, payload: Mapping[str, object], purpose: str
    ) -> None:
        """Send a request, remembering what its reply is for.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping
            Its arguments.
        purpose : str
            What the reply is for; also its latest-wins key.
        """
        self._pending[self.bridge.request(kind, payload, key=purpose)] = purpose

    def _on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the window's requests.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        purpose = self._pending.pop(reply.request_id, None)
        if purpose == "examples" and reply.ok:
            examples: Sequence[Mapping[str, object]] = reply.value  # pyright: ignore[reportAssignmentType]
            self.sheet.set_examples(examples)
        elif purpose == "gases" and reply.ok:
            self.sheet.set_list("gases", reply.value)  # pyright: ignore[reportArgumentType]
            self._build_sheet()
        elif purpose == "python":
            self.drawer_text.setPlainText(
                str(reply.value) if reply.ok else f"# {reply.error}"
            )
        elif purpose == "numbers":
            self._numbers_arrived(reply)
        elif purpose == "curve" and reply.ok:
            self.sheet.show_curve(reply.value, self._numbers)  # pyright: ignore[reportArgumentType]
        elif purpose == "size":
            value = reply.value if reply.ok else None
            self.sheet.show_size(
                value["output_bytes"] if isinstance(value, Mapping) else None  # pyright: ignore[reportArgumentType]
            )

    def _ask_numbers(self) -> None:
        """Ask for the sheet's numbers, where its class can have them at each edit."""
        model = self.document.model
        problem = self.problem
        if model is None or problem is None or problem.numbers == "summarise":
            return
        payload = {"text": model.to_text(), "base": self.document.base}
        self._ask_kernel(problem.numbers, payload, "numbers")

    def _numbers_arrived(self, reply: Reply) -> None:
        """Show the numbers, and ask for the curve of what they resolved.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The kernel's reply.
        """
        value = reply.value
        if not reply.ok or not isinstance(value, Mapping):
            self.sheet.show_error(str(reply.error))
            return
        self._numbers = value
        self.sheet.show_summary(value)
        problem = self.problem
        if problem is not None and problem.curve is not None:
            payload = {
                "text": value["resolved"],
                "base": self.document.base,
                **{k: value[k] for k in _CURVE_KEYS},
            }
            self._ask_kernel(problem.curve, payload, "curve")

    def _summarised(self, summary: object, error: object) -> None:
        """Show the check's summary, on a class that has no quicker numbers.

        Parameters
        ----------
        summary : object
            The kernel's summary, or ``None``.
        error : object
            Why there is none.
        """
        if self.problem is not None and self.problem.numbers == "summarise":
            self.sheet.show_summary(summary, error)  # pyright: ignore[reportArgumentType]

    # ------------------------------------------------------------------ opening

    def begin_class(self, key: str) -> None:
        """Open a new sheet of a class, from its example.

        Parameters
        ----------
        key : str
            The class.
        """
        self.document.open_example(class_named(key).example)

    def open_step_window(self) -> None:
        """Open the step window, which holds every route not yet a class."""
        self.step_window = MainWindow()
        self.step_window.show()

    def choose_setup(self) -> None:
        """Pick a setup file and open it."""
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open a setup", "", "solprop recipes (*.toml);;All files (*)"
        )
        if chosen:
            self.open_setup(chosen)

    def open_setup(self, path: str) -> None:
        """Open a setup file; a route that is no class's goes to the step window.

        Parameters
        ----------
        path : str
            The file.
        """
        try:
            model = RecipeModel.load(path)
        except (OSError, ValueError) as error:
            self._say(f"Cannot open {path}: {error}")
            return
        if class_of(model.route, model.document.get("build", {})) is None:
            self.open_step_window()
            if self.step_window is not None:
                self.step_window.open_recipe(path)
            return
        self.document.open_recipe(path)

    def choose_results(self) -> None:
        """Pick a results file and open it."""
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open results", "", "solprop results (*.h5 *.hdf5);;All files (*)"
        )
        if chosen:
            self.open_results(chosen)

    def open_results(self, path: str) -> None:
        """Open a results file in Analyse.

        Parameters
        ----------
        path : str
            The file.
        """
        self._own_run = False
        self._open_file(path)
        self.show_stage(Stage.ANALYSE)

    def _open_file(self, path: str) -> None:
        """Read a results file into the views, which start at its last plane.

        Parameters
        ----------
        path : str
            The file.
        """
        self.results.overview.follow.setChecked(True)
        self.results.open(path, show=False)
        self.has_results = True

    def _opened(self) -> None:
        """Take a recipe becoming the open setup."""
        model = self.document.model
        if model is None:
            return
        self.problem = class_of(model.route, model.document.get("build", {}))
        self._remember_setup()
        if not self._quiet and not self.run_panel.active:
            self.has_run = False
            self.has_results = False

    def _build_sheet(self) -> None:
        """Make the sheet of the open setup's class, once its route is described."""
        model = self.document.model
        if model is None or self.problem is None:
            return
        description = self.document.descriptions.get(model.route)
        if description is None:
            return
        self.sheet.build(
            self.problem, description, model.document.get("build", {}), model.title
        )
        self._show_values()
        self._ask_numbers()

    def _loaded(self) -> None:
        """Bring a setup forward, unless it came with a results file."""
        if not self._quiet:
            self.show_stage(Stage.SETUP)
        self._quiet = False

    def _show_values(self) -> None:
        """Show the recipe's values on the sheet, and ask what they now give."""
        if self.document.model is not None:
            self.sheet.show_values(self.document.model)
            self._ask_numbers()

    def _checked(self, issues: list[Mapping[str, object]]) -> None:
        """Show solprop's check.

        Parameters
        ----------
        issues : list of Mapping
            The issues.
        """
        self.sheet.show_issues(issues)
        self.run_panel.set_can_run(self.document.can_run)
        if self.document.can_run and self.document.model is not None:
            payload = {
                "text": self.document.model.to_text(),
                "base": self.document.base,
            }
            self._ask_kernel("output_size", payload, "size")
        self._refresh()

    def _remember_setup(self) -> None:
        """Add the open setup's file to the recent ones."""
        model = self.document.model
        if model is None or model.path is None:
            return
        saved = self.settings.value(_RECENT_SETUPS) or []
        names = [saved] if isinstance(saved, str) else list(saved)  # pyright: ignore[reportArgumentType, reportCallIssue]
        path = str(model.path)
        self.settings.setValue(_RECENT_SETUPS, [path, *(n for n in names if n != path)])

    def refresh_home(self) -> None:
        """List the recent runs and setups on Home."""
        self.home.show_runs(list_runs(self.run_panel.workspace))
        saved = self.settings.value(_RECENT_SETUPS) or []
        self.home.show_setups([saved] if isinstance(saved, str) else list(saved))  # pyright: ignore[reportArgumentType, reportCallIssue]

    # ------------------------------------------------------------------ runs

    def _run_state(self, state: str) -> None:
        """Follow a run's state.

        Parameters
        ----------
        state : str
            Its new execution state.
        """
        if state == "starting":
            self.has_run = True
            self._failed = False
            self.run_stage.clear_advice()
            self.show_stage(Stage.RUN)
        self._refresh()

    def _report_failure(self, title: str, message: str, _detail: str) -> None:
        """Keep a failure on the Run stage, in place of a dialog.

        Parameters
        ----------
        title : str
            What happened.
        message : str
            The message solprop gave.
        _detail : str
            The traceback, which the log holds.
        """
        self._failed = True
        self.run_stage.show_advice(title, message)

    def _run_ended(self, path: str) -> None:
        """Open what the run left, and move on to Analyse unless it failed.

        Parameters
        ----------
        path : str
            Its results file.
        """
        self._own_run = True
        self._open_file(path)
        self.refresh_home()
        if not self._failed:
            self.show_stage(Stage.ANALYSE)
        self._refresh()

    # ------------------------------------------------------------------ analysing

    def _file_described(self, value: Mapping[str, object]) -> None:
        """Offer the rail on a file that says what it can give.

        Parameters
        ----------
        value : Mapping
            The kernel's ``describe_output`` reply.
        """
        self._capabilities = value
        z: list[float] = list(value.get("z") or [])  # pyright: ignore[reportArgumentType]
        planes = [int(p) for p in value.get("field_planes") or []]  # pyright: ignore[reportGeneralTypeIssues]
        self.analyse.scrubber.set_positions([z[p] for p in planes if p < len(z)])
        if self._own_run and self.problem is not None:
            self.analyse.set_pages(self.problem.pages, value)
        elif value.get("recipe"):
            self.results.open_setup()
        else:
            self.analyse.set_pages(GENERIC_PAGES, value)

    def _file_has_setup(self, text: str, path: str) -> None:
        """Find a file's class from the recipe it records, and restore its sheet.

        Parameters
        ----------
        text : str
            The recipe.
        path : str
            The results file.
        """
        model = RecipeModel(text)
        problem = class_of(model.route, model.document.get("build", {}))
        pages = GENERIC_PAGES if problem is None else problem.pages
        self.analyse.set_pages(pages, self._capabilities)
        if problem is not None and self.document.may_discard():
            self._quiet = True
            self.document.set_model(model, base=str(Path(path).parent))
            self.has_results = True

    def _plane_moved(self, place: int) -> None:
        """Move the views to the scrubber's plane.

        Parameters
        ----------
        place : int
            Its place among the planes that kept a field.
        """
        self.results.plane.slider.setValue(place)
        positions = self.analyse.scrubber.positions
        if self.results.overview.rows and place < len(positions):
            overview = self.results.overview
            overview.slider.setValue(overview.nearest(positions[place]))

    def _overview_moved(self, row: int) -> None:
        """Move the scrubber to a plane picked on the Overview's maps.

        Parameters
        ----------
        row : int
            The Overview's row.
        """
        overview = self.results.overview
        positions = self.analyse.scrubber.positions
        if not positions or not overview.rows:
            return
        z = float(overview.rows[row]["z"])  # pyright: ignore[reportArgumentType]
        place = min(range(len(positions)), key=lambda i: abs(positions[i] - z))
        self.analyse.scrubber.go_to(place)
        self.results.plane.slider.setValue(place)

    # ------------------------------------------------------------------ the drawer

    def show_recipe(self) -> None:
        """Show the open recipe in the drawer."""
        model = self.document.model
        if model is not None:
            self.drawer_title.setText("Recipe")
            self.drawer_text.setPlainText(model.to_text())
            self.drawer.show()

    def show_python(self) -> None:
        """Show the plain Python script of the open recipe in the drawer."""
        model = self.document.model
        if model is not None:
            self.drawer_title.setText("Python")
            self.drawer_text.setPlainText("Asking solprop…")
            self.drawer.show()
            payload = {"text": model.to_text(), "base": self.document.base}
            self._ask_kernel("to_python", payload, "python")

    # ------------------------------------------------------------------ closing

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's name
        """Ask before closing over a running run or unsaved changes.

        Parameters
        ----------
        event : PySide6.QtGui.QCloseEvent
            The close.
        """
        if self.run_panel.active and not self.bridge.service:
            answer = self.confirm("Run going", "A run is going. Stop it and close?")
            if answer != "stop":
                event.ignore()
                return
            self.run_panel.stop()
            if self.run_panel.client is not None:
                self.run_panel.client.join(30.0)
        if not self.document.may_discard():
            event.ignore()
            return
        self.settings.setValue("workspace", str(self.run_panel.workspace))
        self.document.close()
        self.bridge.close()
        event.accept()
