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

from solprop_gui.classes.catalog import (
    CLASSES,
    GENERIC_PAGES,
    OTHER_KEY,
    class_named,
    class_of,
    other_class,
)
from solprop_gui.classes.problem import ProblemClass
from solprop_gui.classes.workflow import Progress, Stage, primary_action, stage_open
from solprop_gui.model.recipe import MISSING, RecipeModel, without_table
from solprop_gui.model.scans import profile_points
from solprop_gui.model.workspace import DEFAULT_WORKSPACE, list_runs
from solprop_gui.registry.pages import label
from solprop_gui.registry.routes import card_of
from solprop_gui.service.messages import Reply
from solprop_gui.service.worker import SCAN_OUTPUT
from solprop_gui.ui.analyse import AnalyseStage
from solprop_gui.ui.analysis import AnalysisSession
from solprop_gui.ui.analysis_pages import PAGE_VIEWS, OverviewPage
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.design_stage import DesignStage
from solprop_gui.ui.document import Confirm, SetupDocument
from solprop_gui.ui.home import HomePage
from solprop_gui.ui.run_panel import RunPanel
from solprop_gui.ui.scan_stage import ScanStage
from solprop_gui.ui.sheet import SheetStage
from solprop_gui.ui.theme import STYLE, app_settings, load_font, named_label, repolish

__all__ = ["WorkflowWindow"]

_RECENT_SETUPS = "recent-setups"
"""The setting that lists recent setups."""

_STAGE_TEXT = (
    (Stage.HOME, "Home"),
    (Stage.SETUP, "Setup"),
    (Stage.RUN, "Run"),
    (Stage.ANALYSE, "Analyse"),
    (Stage.SCAN, "Scan"),
)


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
        # A bar for passing notices, shown only while it has one.
        bar = self.statusBar()
        bar.messageChanged.connect(lambda text: bar.setVisible(bool(text)))
        bar.hide()
        # Before any request: the first may find the service out of reach.
        self.bridge.service_failed.connect(
            lambda reason: self.notify(
                f"The solprop service is not available ({reason}): runs end with "
                "the window."
            )
        )
        self.confirm: Confirm = self._ask
        self.stage = Stage.HOME
        self.problem: ProblemClass | None = None
        self.has_run = False
        self.has_results = False
        self._quiet = False
        self._own_run = False
        self._failed = False
        self._pending: dict[int, str] = {}
        saved = self.settings.value("workspace")
        self.workspace = workspace or (Path(str(saved)) if saved else DEFAULT_WORKSPACE)

        self.document = SetupDocument(
            self.bridge,
            lambda title, text: self.confirm(title, text),  # noqa: PLW0108 - replaceable
            self,
        )
        self.document.guard = self._may_leave_run
        self._leaving_run = False
        self.session = AnalysisSession(self.bridge, self)
        self.page_views = {key: make(self.session) for key, make in PAGE_VIEWS.items()}
        overview = self.page_views["overview"]
        assert isinstance(overview, OverviewPage)  # what the registry holds
        self.overview = overview
        self.run_panel = RunPanel(
            self.bridge, self._run_recipe, self.workspace, view=self.overview
        )
        self._scanning = False
        self._scan_path = ""
        self._scan_paths: list[str] = []
        kept = self.settings.value("threads")
        self.run_panel.threads = int(str(kept)) if kept else None
        self.home = HomePage()
        self.sheet = SheetStage()
        self.analyse = AnalyseStage(self.session, self.run_panel, self.page_views)
        self.design = DesignStage(self.bridge)
        self.scans = ScanStage()

        self.stack = QStackedWidget()
        self.screens: dict[Stage, QWidget] = {
            Stage.HOME: self.home,
            Stage.DESIGN: self.design,
            Stage.SETUP: self.sheet,
            Stage.RUN: self.analyse,
            Stage.ANALYSE: self.analyse,
            Stage.SCAN: self.scans,
        }
        for screen in (self.home, self.design, self.sheet, self.analyse, self.scans):
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
        self.home.example_chosen.connect(self.document.open_example)
        self.home.design_chosen.connect(lambda: self.show_stage(Stage.DESIGN))
        self.home.open_results_wanted.connect(self.choose_results)
        self.home.open_setup_wanted.connect(self.choose_setup)
        self.home.results_chosen.connect(self.open_results)
        self.home.setup_chosen.connect(self.open_setup)
        self.home.workspace_wanted.connect(self.run_panel.choose_workspace)
        self.run_panel.workspace_changed.connect(self._workspace_changed)
        self.home.show_workspace(self.run_panel.workspace)
        self.sheet.edited.connect(
            lambda name, table, value: self.document.edit(
                name, table, value, merging=True
            )
        )
        self.sheet.pins_changed.connect(self.document.replace_pins)
        self.sheet.values_edited.connect(
            lambda values: self.document.edit_values(values, merging=True)
        )
        self.sheet.lens_wanted.connect(self._ask_lens)
        self.sheet.scan_wanted.connect(self.scan_keyword)
        self.sheet.run_options.backend_chosen.connect(self._backend_chosen)
        self.sheet.run_options.threads_chosen.connect(self._threads_chosen)
        self.scans.table_changed.connect(
            lambda table: self.document.edit_table("scan", table)
        )
        self.run_panel.point_seen.connect(self.scans.show_point)
        self.scans.browser.point_chosen.connect(self.open_scan_point)
        self.sheet.example_chosen.connect(self.document.open_example)
        self.document.opened.connect(self._opened)
        self.document.described.connect(self._described)
        self.document.loaded.connect(self._loaded)
        self.document.values_changed.connect(self._show_values)
        self.run_panel.resume_action.changed.connect(self._refresh)
        self.document.state_changed.connect(self._refresh)
        self.document.checked.connect(self._checked)
        self.document.summarised.connect(self._summarised)
        self.document.message.connect(self._say)
        self.run_panel.report = self._report_failure
        self.run_panel.state_changed.connect(self._run_state)
        self.run_panel.ended.connect(self._run_ended)
        self.session.opened.connect(self._file_opened)
        self.session.described.connect(self._file_described)
        self.session.setup_found.connect(self._file_has_setup)
        self.bridge.kernel_lost.connect(self._kernel_lost)
        self.bridge.runs_listed.connect(self._runs_listed)
        self.bridge.replied.connect(self._on_reply)

        self.setWindowTitle("solprop")
        self.resize(1280, 800)
        self._numbers: Mapping[str, object] = {}
        self._run_revision: int | None = None
        self._ask_kernel("examples", {}, "examples")
        self._ask_kernel("gases", {}, "gases")
        self._ask_kernel("run_options", {}, "run options")
        self.refresh_home()
        self.show_stage(Stage.HOME)
        # With the service, a run the last window left going is taken over.
        self.bridge.list_runs()

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
            can_resume=(
                self.document.model is not None
                and self.document.model.revision == self._run_revision
                and self.run_panel.resume_action.isEnabled()
            ),
        )

    def _refresh(self) -> None:
        """Bring the bar, the stage buttons and the title up to date."""
        progress = self.progress()
        # From Home, the stages lead back to what is open.
        self.top.setVisible(
            self.stage is not Stage.HOME or progress.has_setup or progress.has_results
        )
        designing = self.stage is Stage.DESIGN
        for stage, button in self.stage_buttons.items():
            button.setVisible(not designing or stage is Stage.HOME)
            button.setEnabled(stage_open(stage, progress))
            button.setProperty("current", stage is self.stage)
            repolish(button)
        primary = primary_action(progress)
        self.primary.setText("Simulate this" if designing else primary.text)
        self.primary.setEnabled(designing or primary.enabled)
        if self.stage is Stage.SCAN and not progress.running:
            self.primary.setText("Run scan")
            self.primary.setEnabled(progress.can_run and self.scans.has_points())
        self.overflow.setVisible(not designing)
        if designing:
            self.context.setText("Design a waveguide")
            self.setWindowTitle("Design a waveguide — solprop")
            return
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
        if stage in (Stage.RUN, Stage.ANALYSE):
            self.analyse.show_running(stage is Stage.RUN)
        self._refresh()

    def primary_clicked(self) -> None:
        """Do what the primary button says."""
        if self.stage is Stage.DESIGN:
            self.simulate_design()
            return
        if self.stage is Stage.SCAN and not self.run_panel.active:
            self._scanning = True
            self.run_panel.set_can_run(self.document.can_run)
            self.run_panel.reductions = self.scans.record.specs()
            self.run_panel.start()
            self.run_panel.reductions = []
            return
        action = primary_action(self.progress()).action
        if action == "stop":
            self.run_panel.stop()
        elif action == "resume":
            self.run_panel.resume()
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
            self._examples_arrived(reply.value)  # pyright: ignore[reportArgumentType]
        elif purpose is not None and purpose.startswith("warm-") and reply.ok:
            self._ask_kernel("check", {"text": str(reply.value)}, f"warmed-{purpose}")
        elif purpose == "gases" and reply.ok:
            self.design.set_gases(reply.value)  # pyright: ignore[reportArgumentType]
            self.sheet.set_list("gases", reply.value)  # pyright: ignore[reportArgumentType]
            self._build_sheet()
        elif purpose == "run options" and reply.ok:
            self.sheet.run_options.set_options(reply.value, self.run_panel.threads)  # pyright: ignore[reportArgumentType]
        elif purpose == "scan":
            self._scan_arrived(reply)
        elif purpose == "scan point":
            self._scan_point_arrived(reply)
        elif purpose == "python":
            self.drawer_text.setPlainText(
                str(reply.value) if reply.ok else f"# {reply.error}"
            )
        elif purpose == "numbers":
            self._numbers_arrived(reply)
        elif purpose == "curve" and reply.ok:
            self.sheet.show_curve(reply.value)  # pyright: ignore[reportArgumentType]
        elif purpose == "template" and reply.ok:
            self.sheet.set_example(
                RecipeModel(str(reply.value)).document.get("build", {})
            )
        elif purpose == "lens":
            self._lens_arrived(reply)
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
        self.analyse.set_reference(value.get("collapse_distance"))  # pyright: ignore[reportArgumentType]
        if problem is not None and problem.curve is not None:
            payload = {
                "text": value["resolved"],
                "base": self.document.base,
                "numbers": {k: v for k, v in value.items() if k != "resolved"},
            }
            self._ask_kernel(problem.curve, payload, "curve")

    def _ask_lens(self, focal_length: float, radius: float) -> None:
        """Ask what a lens makes of the beam.

        Parameters
        ----------
        focal_length : float
            The focal length, m.
        radius : float
            The beam's radius on it, m.
        """
        model = self.document.model
        if model is not None:
            payload = {
                "text": model.to_text(),
                "base": self.document.base,
                "focal_length": focal_length,
                "radius": radius,
            }
            self._ask_kernel("lens_beam", payload, "lens")

    def _lens_arrived(self, reply: Reply) -> None:
        """Set the beam a lens makes.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The kernel's reply.
        """
        value = reply.value
        if reply.ok and isinstance(value, Mapping):
            self.sheet.apply_lens(float(value["waist"]), float(value["distance"]))  # pyright: ignore[reportArgumentType]
        else:
            self._say(f"No lens: {reply.error}")

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

    def _may_leave_run(self) -> bool:
        """Ask before opening anything else while a run is going, and stop it if told.

        The window follows one run: another setup or file open beside it would leave
        the run's stages showing the wrong thing.

        Returns
        -------
        bool
            Whether to go on.
        """
        if not self.run_panel.active or self._leaving_run:
            return True
        answer = self.confirm(
            "Run going", "A run is going. Stop it and open something else?"
        )
        if answer != "stop":
            return False
        self._leaving_run = True
        self.run_panel.stop()
        return True

    def _backend_chosen(self, name: str) -> None:
        """Write the backend picked into the recipe, with the precision it needs.

        Parameters
        ----------
        name : str
            The backend; MLX runs in single precision only.
        """
        model = self.document.model
        if model is None:
            return
        values: list[tuple[str, object]] = [("backend", name)]
        said: list[str] = []
        if name == "mlx":
            values.append(("precision", "single"))
            # solprop refuses in single precision what binary32 cannot hold: a noise
            # seed (rounding swamps a one-photon seed) and a tolerance below 1e-6.
            if model.get("seed") is not MISSING and model.get("noise") is MISSING:
                values.append(("seed", MISSING))
                said.append("the noise seed was dropped")
            rtol = model.get("rtol")
            if isinstance(rtol, int | float) and rtol < _SINGLE_RTOL:
                values.append(("rtol", _SINGLE_RTOL))
                said.append(f"rtol was raised from {rtol:g} to {_SINGLE_RTOL:g}")
        elif model.get("precision") == "single" and model.get("backend") == "mlx":
            values.append(("precision", "double"))
        self.document.edit_values(values)
        if said:
            changed = " and ".join(said)
            self.notify(
                f"MLX runs in single precision: {changed}. Undo puts them back."
            )

    def _threads_chosen(self, threads: object) -> None:
        """Keep the thread count picked, for this run and the next session's.

        Parameters
        ----------
        threads : object
            The count, or ``None`` for solprop's default.
        """
        self.run_panel.threads = None if threads is None else int(threads)  # pyright: ignore[reportArgumentType]
        if threads is None:
            self.settings.remove("threads")
        else:
            self.settings.setValue("threads", int(threads))  # pyright: ignore[reportArgumentType]

    def _run_recipe(self) -> tuple[str, str | None, str | None]:
        """Return what a run runs: the setup once, or its scan from the Scan stage.

        Returns
        -------
        tuple
            The recipe's text, its base directory and its title.
        """
        text, base, title = self.document.snapshot()
        return (text if self._scanning else without_table(text, "scan")), base, title

    def scan_keyword(self, name: str, value: object) -> None:
        """Make a keyword a variable of the setup's scan, and go to the scan.

        Parameters
        ----------
        name : str
            The keyword.
        value : object
            Its value now, or ``None``.
        """
        self.scans.editor.scan(name, value)
        self.show_stage(Stage.SCAN)

    def open_scan(self, path: str) -> None:
        """Show a scan's results on the Scan stage.

        Parameters
        ----------
        path : str
            The scan's file.
        """
        self.scans.progress.setText("Reading the scan…")
        self._scan_path = path
        self._ask_kernel("scan_overview", {"path": path}, "scan")
        self.show_stage(Stage.SCAN)

    def simulate_design(self) -> None:
        """Carry the design to a new sheet, asking first about unsaved changes."""
        if self.design.text is not None:
            self.document.open_text(self.design.text, None)

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
        del model
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
        if not self._may_leave_run():
            return
        if Path(path).name == SCAN_OUTPUT:
            self.open_scan(path)
            return
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
        self.session.open(path)
        self.has_results = True

    def _opened(self) -> None:
        """Take a recipe becoming the open setup."""
        model = self.document.model
        if model is None:
            return
        self.problem = _class_of(model)
        self._use_class()
        self._remember_setup()
        if not self._quiet and not self.run_panel.active:
            self.has_run = False
            self.has_results = False

    def _use_class(self) -> None:
        """Fetch what the open class needs, and draw a run on its own Overview."""
        problem = self.problem
        if problem is None:
            return
        if problem.example:
            self._ask_kernel("example", {"name": problem.example}, "template")
        # A new class starts at its Overview, not at the page the last one left.
        self.analyse.current = None
        if not self.run_panel.active:
            first = problem.pages[0]
            view = self.page_views[first.view or first.key]
            if isinstance(view, OverviewPage):
                self.overview = view
                self.run_panel.view = view  # pyright: ignore[reportAttributeAccessIssue]
                self.analyse.overview = view

    def _scan_arrived(self, reply: Reply) -> None:
        """Show a scan's results, or why they cannot be read.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The ``scan_overview`` reply.
        """
        if reply.ok:
            self.scans.show_scan(reply.value, self._scan_path)  # pyright: ignore[reportArgumentType]
        else:
            self.scans.show_error(str(reply.error))

    def open_scan_point(self, index: int) -> None:
        """Ask for one point of the scan shown, to open it as a setup.

        Parameters
        ----------
        index : int
            The point's flat index.
        """
        self._ask_kernel(
            "scan_point", {"path": self._scan_path, "index": index}, "scan point"
        )

    def _scan_point_arrived(self, reply: Reply) -> None:
        """Open a scan point's setup as a new sheet, or say why it cannot be.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The ``scan_point`` reply.
        """
        if not reply.ok:
            self.notify(f"The point cannot be opened: {reply.error}")
            return
        value: Mapping[str, object] = reply.value  # pyright: ignore[reportAssignmentType]
        before = self.document.model
        self.document.open_text(str(value["text"]), self.document.base)
        model = self.document.model
        if model is not None and model is not before:
            values: Mapping[str, object] = value["values"]  # pyright: ignore[reportAssignmentType]
            said = ", ".join(
                f"{label(path).lower()} {model.get(path)}" for path in values
            )
            self.notify(f"The scan's point at {said}, as a new setup.")
            self.show_stage(Stage.SETUP)

    def _examples_arrived(self, examples: Sequence[Mapping[str, object]]) -> None:
        """Offer solprop's examples, and ask ahead for what a first class needs.

        Parameters
        ----------
        examples : Sequence of Mapping
            The kernel's ``examples`` reply.
        """
        self.sheet.set_examples(examples)
        self.home.set_other_examples(
            [
                (str(entry["name"]), str(entry["title"]))
                for entry in examples
                if class_of(str(entry["route"]), entry["build"]) is None  # pyright: ignore[reportArgumentType]
            ]
        )
        self._warm_up()

    def _warm_up(self) -> None:
        """Ask ahead of the first click for what opening a class needs.

        The classes' route descriptions, so a sheet is built as its recipe arrives,
        and a first check of the slow routes, the first costing most.
        """
        self.document.prefetch([problem.route for problem in CLASSES])
        for name in _WARM_EXAMPLES:
            self._ask_kernel("example", {"name": name}, f"warm-{name}")

    def _offer_scan_paths(self) -> None:
        """Offer the route's numbers as scan variables, and each point of a profile.

        A pressure given as a gradient, ``["1.2 bar", "0 bar"]``, offers
        ``pressure[0]`` and ``pressure[1]``: one point of it can be scanned.
        """
        model = self.document.model
        build = {} if model is None else model.document.get("build", {})
        self.scans.set_paths([*self._scan_paths, *profile_points(build)])

    def _described(self, route: str) -> None:
        """Build the sheet when the open setup's route has been described.

        Parameters
        ----------
        route : str
            The route described; one fetched ahead for a later setup changes nothing.
        """
        model = self.document.model
        if model is not None and model.route == route:
            self._build_sheet()

    def _build_sheet(self) -> None:
        """Make the sheet of the open setup's class, once its route is described."""
        model = self.document.model
        if model is None or self.problem is None:
            return
        description = self.document.descriptions.get(model.route)
        if description is None:
            return
        if self.problem.key == OTHER_KEY:
            # Other geometry's form comes from the route's metadata, now it is here.
            self.problem = other_class(
                model.route,
                self.problem.title,
                model.document.get("build", {}),
                description["parameters"],  # pyright: ignore[reportArgumentType]
            )
        self.sheet.build(
            self.problem, description, model.document.get("build", {}), model.title
        )
        keywords: Sequence[Mapping[str, object]] = description["parameters"]  # pyright: ignore[reportAssignmentType]
        self._scan_paths = [
            str(entry["name"])
            for entry in keywords
            if any(
                form.get("kind") in ("number", "integer")
                for form in entry.get("forms", [])  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
            )
        ]
        self._offer_scan_paths()
        self._show_values()
        self._ask_numbers()

    def _loaded(self) -> None:
        """Bring a setup forward, unless it came with a results file."""
        if not self._quiet:
            self.show_stage(Stage.SETUP)
        self._quiet = False

    def _show_values(self) -> None:
        """Show the recipe's values on the sheet, and ask what they now give."""
        model = self.document.model
        if model is not None:
            self.sheet.show_values(model)
            self.scans.show_table(model.document.get("scan"))
            self._offer_scan_paths()
            # Run scan is offered once the scan has a point to run.
            self._refresh()
            described = self.document.descriptions.get(model.route, {})
            parameters: Sequence[Mapping[str, object]] = described.get("parameters", [])  # pyright: ignore[reportAssignmentType]
            self.sheet.run_options.show_backend(
                model.get("backend"),
                offered=any(entry.get("name") == "backend" for entry in parameters),
            )
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

    def _workspace_changed(self, folder: str) -> None:
        """Make runs in another folder, keep it, and list its runs.

        Parameters
        ----------
        folder : str
            The folder.
        """
        self.settings.setValue("workspace", folder)
        self.home.show_workspace(Path(folder))
        self.refresh_home()

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
        if state == "starting" and self._scanning:
            # A scan's points are watched on its own stage, not on Run.
            self.scans.progress.setText("Starting the scan…")
        elif state == "starting":
            self.has_run = True
            self.has_results = False
            self._failed = False
            model = self.document.model
            self._run_revision = None if model is None else model.revision
            self.analyse.run_started()
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
        self.analyse.show_advice(title, message)

    def _run_ended(self, path: str) -> None:
        """Open what the run left, and move on to Analyse unless it failed.

        Parameters
        ----------
        path : str
            Its results file.
        """
        if self._leaving_run:
            # A run stopped to open something else: it is listed, and not shown.
            self._leaving_run = False
            self._scanning = False
            self.refresh_home()
            self._refresh()
            return
        if Path(path).name == SCAN_OUTPUT:
            self._scanning = False
            self.open_scan(path)
            self.refresh_home()
            self._refresh()
            return
        self._own_run = True
        self.overview.end_of_run()
        self._open_file(path)
        self.refresh_home()
        if not self._failed:
            self.show_stage(Stage.ANALYSE)
        self._refresh()

    # ------------------------------------------------------------------ analysing

    def _file_opened(self) -> None:
        """Forget what the pages read of the last file."""
        for view in self.page_views.values():
            forget = getattr(view, "forget", None)
            if forget is not None:
                forget()

    def _file_described(self, value: Mapping[str, object]) -> None:
        """Offer the rail on a file that says what it can give.

        Parameters
        ----------
        value : Mapping
            The kernel's ``describe_output`` reply.
        """
        if self._own_run and self.problem is not None:
            self.analyse.set_wanted(self.problem.statistics)
            self.analyse.set_pages(self.problem.pages, value)
        elif value.get("recipe"):
            self.session.find_setup()
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
        problem = _class_of(model)
        self.analyse.set_wanted(problem.statistics)
        self.analyse.set_pages(problem.pages, self.session.capabilities)
        if self.document.may_discard():
            self._quiet = True
            self.document.set_model(model, base=str(Path(path).parent))
            self.has_results = True

    def _runs_listed(self, runs: Sequence[Mapping[str, object]]) -> None:
        """Take over the latest run the service holds that is still going.

        Parameters
        ----------
        runs : Sequence of Mapping
            The service's runs, oldest first.
        """
        going = [run for run in runs if run.get("alive")]
        if going and self.run_panel.attach(going[-1]):
            self.has_run = True
            self.show_stage(Stage.RUN)
            self.notify(f"Re-attached to {going[-1]['run_id']}.")

    def notify(self, text: str) -> None:
        """Show a passing notice at the foot of the window.

        Parameters
        ----------
        text : str
            What to say.
        """
        self.statusBar().showMessage(text, _NOTICE_MS)

    def _kernel_lost(self) -> None:
        """Say that the kernel ended; the next request starts it again."""
        self._say(
            "solprop's kernel process ended, which a request too large for memory can "
            "do. It starts again with the next request."
        )

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
            self._leaving_run = True
            self.run_panel.stop()
            client = self.run_panel.client
            if client is not None:
                # A run stops at its next plane, which may be far: a window being
                # closed waits a moment, then ends the worker.
                client.join(_STOP_WAIT_S)
                if not client.ended.is_set():
                    client.kill()
                    client.join(_STOP_WAIT_S)
        # Closing has dealt with the run: kept by the service, or stopped above.
        guard, self.document.guard = self.document.guard, lambda: True
        try:
            discard = self.document.may_discard()
        finally:
            self.document.guard = guard
        if not discard:
            event.ignore()
            return
        self.settings.setValue("workspace", str(self.run_panel.workspace))
        self.document.close()
        self.bridge.close()
        event.accept()


_WARM_EXAMPLES = ("rdw", "filament")
"""Examples checked once as the window opens: a route's first check costs the most
(about 0.9 s on the capillary and 0.5 s on the radial route), and is then the
kernel's, not the first click's."""

_STOP_WAIT_S = 3.0
"""How long a closing window waits for a stopped run to end before ending it."""

_SINGLE_RTOL = 1e-6
"""The smallest ``rtol`` solprop accepts in single precision."""

_NOTICE_MS = 20_000
"""How long a passing notice stays, in milliseconds."""


def _class_of(model: RecipeModel) -> ProblemClass:
    """Return a recipe's class: one of the four, or Other geometry's for its route.

    Parameters
    ----------
    model : solprop_gui.model.recipe.RecipeModel
        The recipe.

    Returns
    -------
    ProblemClass
        The class; Other geometry's form waits for the route's metadata.
    """
    build = model.document.get("build", {})
    found = class_of(model.route, build)
    if found is not None:
        return found
    card = card_of(model.route, build, enabled_only=False)
    return other_class(model.route, model.route if card is None else card.title, build)
