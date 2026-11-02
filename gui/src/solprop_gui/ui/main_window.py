# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The main window: the step bar, the settings, the views and the messages (design 5).

The layout follows COMSOL's, simplified. The steps are icons across the top, with
back and forward; the step's settings are a narrow panel at the left; the views (the
live run, the results, the compare) are tabs over the rest; and the summary with
solprop's check, the run's log and the Python script are tabs below the views.

The window holds one recipe at a time (:class:`~solprop_gui.model.recipe.RecipeModel`).
Every edit is an undoable command; after a pause in typing the window asks the kernel
to check the recipe and summarise it, and shows solprop's issues at their fields and
as badges on the steps. A step's ✓ means solprop accepts what it sets.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from PySide6.QtCore import QSettings, QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QKeySequence,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.model.workspace import DEFAULT_WORKSPACE, list_runs
from solprop_gui.registry.pages import PAGES
from solprop_gui.registry.routes import CARDS, card, card_of
from solprop_gui.service.messages import Reply
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.icons import icon
from solprop_gui.ui.pages import ParameterPage, StartPage, issue_keyword
from solprop_gui.ui.results import ResultsPage
from solprop_gui.ui.run_panel import RunPanel

__all__ = ["CHECK_DELAY_MS", "ROUTES", "STEPS", "MainWindow"]

CHECK_DELAY_MS = 400
"""The pause in typing after which the recipe is checked, ms."""

ROUTES = frozenset(entry.route for entry in CARDS if entry.enabled)
"""The routes the window offers in this version."""

_BADGES = {None: "✓ ", "warning": "⚠ ", "error": "✗ "}

_VERDICTS = {None: "accepted", "warning": "warning", "error": "error"}

_SAID = {
    "accepted": "\n✓ solprop accepts this step.",
    "warning": "\n⚠ solprop warns about this step.",
    "error": "\n✗ solprop refuses this step as it is.",
}

STEPS: Final = ("start", *(page.key for page in PAGES))
"""The steps' keys, in order: the start, then the pages."""

Confirm = Callable[[str, str], str]
"""Asks the user a question; returns ``"save"``, ``"discard"``, ``"stop"`` or
``"cancel"``."""


class _SetValue(QUndoCommand):
    """Set, or unset, one value of the recipe; undone by setting the old one back.

    Parameters
    ----------
    window : MainWindow
        The window whose recipe it changes.
    name : str
        The keyword.
    table : str
        Its table.
    values : tuple
        The new value and the old one.
    """

    def __init__(
        self,
        window: MainWindow,
        name: str,
        table: str,
        values: tuple[object, object],
    ) -> None:
        """Hold the change: the new value and the old one."""
        super().__init__(f"Set {name}")
        self.window = window
        self.name = name
        self.table = table
        self.new, self.old = values

    def redo(self) -> None:
        """Set the new value."""
        self.window.apply_value(self.name, self.table, self.new)

    def undo(self) -> None:
        """Set the old value back."""
        self.window.apply_value(self.name, self.table, self.old)


class MainWindow(QMainWindow):
    """The application's one main window.

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
        How the window asks before losing work (:data:`Confirm`); replaceable, so
        that tests answer without a dialog.
    """

    def __init__(  # noqa: PLR0915 - the window's layout
        self,
        bridge: ServiceBridge | None = None,
        workspace: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Make the window, with no setup open."""
        super().__init__(parent)
        self.settings = QSettings("solprop", "solprop-gui")
        self.bridge = bridge or ServiceBridge(self)
        self.model: RecipeModel | None = None
        self.base: str | None = None
        self.descriptions: dict[str, Mapping[str, object]] = {}
        self.issues: list[Mapping[str, object]] = []
        self.code = ""
        self.confirm: Confirm = self._ask
        self._pending: dict[int, str] = {}
        self._built_route: str | None = None
        self.verdicts: dict[str, str | None] = {}
        self.undo = QUndoStack(self)
        saved = self.settings.value("workspace")
        self.workspace = workspace or (Path(str(saved)) if saved else DEFAULT_WORKSPACE)

        self.start = StartPage()
        self.start.card_chosen.connect(self.new_from_card)
        self.start.example_chosen.connect(self.open_example)
        self.start.open_button.clicked.connect(self.choose_recipe)
        self.results = ResultsPage(self.bridge)
        self.results.setup_opened.connect(self.open_setup_text)
        self.results.view_wanted.connect(self.show_view)
        self.start.results_button.clicked.connect(self.results.choose)
        self.run_panel = RunPanel(self.bridge, self.snapshot, self.workspace)
        self.run_panel.ended.connect(self._run_ended)
        self.run_panel.state_changed.connect(self._run_state)
        self.run_panel.workspace_changed.connect(lambda _path: self.refresh_runs())

        # The settings panel at the left: one panel per step.
        self.pages: dict[str, ParameterPage] = {}
        self.panels = QStackedWidget()
        self.panels.setObjectName("settings")
        self.panels.setMinimumWidth(280)
        self.panels.addWidget(self.start)
        for page in PAGES:
            if page.key == "results":
                self.panels.addWidget(self.results)
                continue
            widget = ParameterPage(page)
            widget.edited.connect(self.edit)
            if page.key == "propagate":
                widget.add_section("Run", self.run_panel)
            self.pages[page.key] = widget
            self.panels.addWidget(widget)

        # The views at the right, in tabs: the plots now, the geometry and the
        # materials later.
        self.views = QTabWidget()
        self.views.setObjectName("views")
        self.views.setDocumentMode(True)
        self.views.addTab(self.run_panel.view, "Live run")
        self.views.addTab(self.results.overview, "Results")
        self.views.addTab(self.results.compare, "Compare")

        # Below the views: the summary and solprop's check, the log, the code.
        self.summary = QLabel("No setup open.")
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.issue_list = QListWidget()
        self.issue_list.setObjectName("issues")
        self.issue_list.itemActivated.connect(self._go_to_issue)
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        summary_scroll.setWidget(self.summary)
        checks = QWidget()
        checks_layout = QVBoxLayout(checks)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.setSpacing(2)
        checks_layout.addWidget(QLabel("solprop's check"))
        checks_layout.addWidget(self.issue_list, 1)
        summary_body = QSplitter()
        summary_body.setObjectName("summary-panel")
        summary_body.addWidget(summary_scroll)
        summary_body.addWidget(checks)
        summary_body.setSizes([380, 540])
        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setObjectName("code")
        self.console = QTabWidget()
        self.console.setObjectName("console")
        self.console.setDocumentMode(True)
        self.console.setTabPosition(QTabWidget.TabPosition.South)
        self.console.addTab(summary_body, "Summary")
        self.console.addTab(self.run_panel.log, "Log")
        self.console.addTab(self.code_view, "Python")
        self.console.currentChanged.connect(self._console_changed)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.views)
        right.addWidget(self.console)
        right.setStretchFactor(0, 1)
        right.setSizes([560, 180])
        split = QSplitter()
        split.addWidget(self.panels)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 920])
        self.setCentralWidget(split)

        status = self.statusBar()
        status.addPermanentWidget(self.run_panel.steps)
        status.addPermanentWidget(self.run_panel.lamp)
        status.addPermanentWidget(self.run_panel.progress)

        self._make_toolbar()
        self._make_actions()
        self.check_timer = QTimer(self)
        self.check_timer.setSingleShot(True)
        self.check_timer.timeout.connect(self.check)
        self.autosave = QTimer(self)
        self.autosave.timeout.connect(self.write_recovery)
        self.autosave.start(30_000)
        self.bridge.replied.connect(self.on_reply)

        self.show_step("start")
        self.resize(1280, 800)
        self._update_title()
        self._update_steps()
        self.refresh_runs()
        self._ask_kernel("examples", {}, "examples")

    # ------------------------------------------------------------------ set-up

    def _add_action(
        self,
        menu: QMenu,
        text: str,
        slot: Callable[[], object],
        shortcut: QKeySequence.StandardKey | None = None,
    ) -> QAction:
        """Add an action to a menu.

        Parameters
        ----------
        menu : PySide6.QtWidgets.QMenu
            The menu.
        text : str
            Its text.
        slot : callable
            What it does.
        shortcut : PySide6.QtGui.QKeySequence.StandardKey or None, optional
            Its shortcut.

        Returns
        -------
        PySide6.QtGui.QAction
            The action.
        """
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _make_actions(self) -> None:
        """Make the menus and their shortcuts."""
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        new_menu = file_menu.addMenu("&New setup")
        for entry in CARDS:
            action = self._add_action(
                new_menu, entry.title, lambda key=entry.key: self.new_from_card(key)
            )
            action.setEnabled(entry.enabled)
        keys = QKeySequence.StandardKey
        self._add_action(file_menu, "&Open setup…", self.choose_recipe, keys.Open)
        self._add_action(file_menu, "&Save setup", self.save, keys.Save)
        self._add_action(file_menu, "Save setup &as…", self.save_as, keys.SaveAs)
        self._add_action(file_menu, "&Export as Python…", self.export_python)
        file_menu.addSeparator()
        self._add_action(file_menu, "Open &results…", self.results.choose)
        self._add_action(file_menu, "Run &workspace…", self.run_panel.choose_workspace)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Quit", self.close, keys.Quit)
        edit_menu = menu.addMenu("&Edit")
        undo = self.undo.createUndoAction(self, "&Undo")
        undo.setShortcut(keys.Undo)
        redo = self.undo.createRedoAction(self, "&Redo")
        redo.setShortcut(keys.Redo)
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)
        go_menu = menu.addMenu("&Go")
        go_menu.addAction(self.back_action)
        go_menu.addAction(self.forward_action)
        go_menu.addSeparator()
        for action in self.step_actions.values():
            go_menu.addAction(action)
        run_menu = menu.addMenu("&Run")
        panel = self.run_panel
        for action in (
            panel.run_action,
            panel.stop_action,
            panel.kill_action,
            panel.resume_action,
            panel.estimate_action,
        ):
            run_menu.addAction(action)
        run_menu.addSeparator()
        self._add_action(run_menu, "Run &workspace…", panel.choose_workspace)
        view_menu = menu.addMenu("&View")
        for widget, text in ((self.panels, "&Settings"), (self.console, "&Messages")):
            toggle = QAction(text, self)
            toggle.setCheckable(True)
            toggle.setChecked(True)
            toggle.toggled.connect(widget.setVisible)
            view_menu.addAction(toggle)
        view_menu.addSeparator()
        for index, name in enumerate(("Summary", "Log", "Python")):
            self._add_action(
                view_menu,
                name,
                lambda index=index: self._show_console(index),
            )

    def _make_toolbar(self) -> None:
        """Make the step bar: back and forward, the steps, and the run's actions."""
        colour = self.palette().windowText().color()
        keys = QKeySequence.StandardKey
        self.back_action = QAction(icon("back", colour), "Back", self)
        self.back_action.setShortcut(keys.Back)
        self.back_action.setToolTip("The previous step")
        self.back_action.triggered.connect(lambda: self.step_by(-1))
        self.forward_action = QAction(icon("forward", colour), "Next", self)
        self.forward_action.setShortcut(keys.Forward)
        self.forward_action.setToolTip("The next step")
        self.forward_action.triggered.connect(lambda: self.step_by(1))
        group = QActionGroup(self)
        group.setExclusive(True)
        self.step_actions: dict[str, QAction] = {}
        for number, key in enumerate(STEPS):
            page = next((p for p in PAGES if p.key == key), None)
            text = "Start" if page is None else (page.short or page.title)
            action = QAction(icon(key, colour), text, self)
            action.setCheckable(True)
            action.setToolTip(page.help if page else "New setups, examples and files.")
            action.setShortcut(QKeySequence(f"Ctrl+{number + 1}"))
            action.triggered.connect(
                lambda _checked=False, key=key: self.show_step(key)
            )
            group.addAction(action)
            self.step_actions[key] = action
        bar = QToolBar("Steps", self)
        bar.setObjectName("step-bar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setIconSize(QSize(30, 30))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        bar.addAction(self.back_action)
        bar.addAction(self.forward_action)
        bar.addSeparator()
        for action in self.step_actions.values():
            bar.addAction(action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        panel = self.run_panel
        for action in (
            panel.estimate_action,
            panel.run_action,
            panel.stop_action,
            panel.resume_action,
        ):
            bar.addAction(action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)
        self.step_bar = bar

    # ------------------------------------------------------------------ steps

    @property
    def step(self) -> str:
        """Return the step shown.

        Returns
        -------
        str
            Its key: ``"start"`` or a page's.
        """
        return STEPS[self.panels.currentIndex()]

    def show_step(self, key: str) -> None:
        """Show a step's settings, and the view that goes with it.

        Parameters
        ----------
        key : str
            The step: ``"start"`` or a page's key.
        """
        index = STEPS.index(key)
        self.panels.setCurrentIndex(index)
        self.step_actions[key].setChecked(True)
        self.back_action.setEnabled(index > 0)
        self.forward_action.setEnabled(index < len(STEPS) - 1)
        if key == "results":
            self.show_view("results")

    def step_by(self, steps: int) -> None:
        """Go back or forward through the steps.

        Parameters
        ----------
        steps : int
            How many, signed.
        """
        index = min(max(STEPS.index(self.step) + steps, 0), len(STEPS) - 1)
        self.show_step(STEPS[index])

    def show_view(self, name: str) -> None:
        """Bring a view forward.

        Parameters
        ----------
        name : str
            ``"live"``, ``"results"`` or ``"compare"``.
        """
        widget = {
            "live": self.run_panel.view,
            "results": self.results.overview,
            "compare": self.results.compare,
        }[name]
        self.views.setCurrentWidget(widget)

    def _show_console(self, index: int) -> None:
        """Show the bottom panel at one of its tabs.

        Parameters
        ----------
        index : int
            The tab.
        """
        self.console.setVisible(True)
        self.console.setCurrentIndex(index)

    def _ask(self, title: str, text: str) -> str:
        """Ask the user before work is lost.

        Parameters
        ----------
        title : str
            The question's title: ``"Unsaved setup"`` or ``"Run going"``.
        text : str
            The question.

        Returns
        -------
        str
            ``"save"``, ``"discard"``, ``"stop"`` or ``"cancel"``.
        """
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        if title == "Run going":
            stop = box.addButton("Stop the run", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            return "stop" if box.clickedButton() is stop else "cancel"
        save = box.addButton(QMessageBox.StandardButton.Save)
        discard = box.addButton(QMessageBox.StandardButton.Discard)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        return "discard" if clicked is discard else "cancel"

    # ------------------------------------------------------------------ kernel

    def _ask_kernel(
        self,
        kind: str,
        payload: Mapping[str, object],
        purpose: str,
        *,
        revision: int = 0,
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
        revision : int, optional
            The recipe revision it is for.
        """
        request = self.bridge.request(kind, payload, revision=revision, key=purpose)
        self._pending[request] = purpose

    def snapshot(self) -> tuple[str, str | None, str | None]:
        """Return the recipe to run: its text, data directory and title.

        Returns
        -------
        tuple
            The text, the directory its data files resolve against, and its title.

        Raises
        ------
        RuntimeError
            If no recipe is open.
        """
        if self.model is None:
            msg = "no setup is open"
            raise RuntimeError(msg)
        return self.model.to_text(), self.base, self.model.title

    def on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the window's requests.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        purpose = self._pending.pop(reply.request_id, None)
        if purpose is None:
            return
        value = reply.value
        if purpose == "examples" and reply.ok:
            self.start.show_examples(value, ROUTES)  # pyright: ignore[reportArgumentType]
        elif purpose.startswith("describe-") and reply.ok and isinstance(value, dict):
            self.descriptions[str(value["route"])] = value
            self._build_pages()
        elif purpose == "example" and reply.ok and isinstance(value, str):
            self.set_model(RecipeModel(value), base=None)
        elif purpose == "check":
            self._show_issues(value if reply.ok else _failure(reply))  # pyright: ignore[reportArgumentType]
        elif purpose == "summarise":
            self._show_summary(value if reply.ok else None, reply.error)  # pyright: ignore[reportArgumentType]
        elif purpose in ("to_python", "export"):
            self.code = str(value) if reply.ok else f"# {reply.error}"
            self.code_view.setPlainText(self.code)
            if purpose == "export" and reply.ok:
                self._save_code()
        if not reply.ok and reply.detail:
            self.statusBar().showMessage(f"{purpose}: {reply.error}", 10_000)

    # ------------------------------------------------------------------ the recipe

    def new_from_card(self, key: str) -> None:
        """Start a new setup from a route card.

        Parameters
        ----------
        key : str
            The card's key.
        """
        if self._may_discard():
            self.set_model(RecipeModel(card(key).blank("New setup")), base=None)

    def open_example(self, name: str) -> None:
        """Start a setup from one of solprop's examples.

        Parameters
        ----------
        name : str
            The example's name.
        """
        if self._may_discard():
            self._ask_kernel("example", {"name": name}, "example")

    def choose_recipe(self) -> None:
        """Pick a recipe file and open it."""
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open setup", "", "solprop recipes (*.toml);;All files (*)"
        )
        if chosen:
            self.open_recipe(chosen)

    def open_recipe(self, path: str | Path) -> None:
        """Open a recipe file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        """
        if not self._may_discard():
            return
        try:
            model = RecipeModel.load(path)
        except (OSError, ValueError) as error:
            self.statusBar().showMessage(f"Cannot open {path}: {error}", 10_000)
            return
        self.set_model(model, base=str(Path(path).parent))

    def open_setup_text(self, text: str, results: str) -> None:
        """Open the setup a results file records, as a new, unsaved setup.

        Parameters
        ----------
        text : str
            The recipe's text.
        results : str
            The results file it came from.
        """
        if self._may_discard():
            self.set_model(RecipeModel(text), base=str(Path(results).parent))
            self.statusBar().showMessage(f"Setup of {results}", 10_000)

    def open_results(self, path: str) -> None:
        """Show a results file on the Results page.

        Parameters
        ----------
        path : str
            The file.
        """
        self.results.open(path)
        self.show_step("results")

    def set_model(self, model: RecipeModel, *, base: str | None) -> None:
        """Make a recipe the window's.

        Parameters
        ----------
        model : solprop_gui.model.recipe.RecipeModel
            The recipe.
        base : str or None
            The directory its data files resolve against.
        """
        self.model = model
        self.base = base
        self.undo.clear()
        self.issues = []
        route = model.route
        if route not in ROUTES:
            self.statusBar().showMessage(
                f"The {route} route is not yet offered by this version of the front "
                "end: its pages show solprop's keywords, unchecked by the window.",
                15_000,
            )
        if route in self.descriptions:
            self._build_pages()
        else:
            self._ask_kernel("describe", {"route": route}, f"describe-{route}")
        self._update_title()
        self.schedule_check(now=True)
        self.show_step(PAGES[0].key)

    def _build_pages(self) -> None:
        """Make the pages' rows for the open recipe's route, and show its values."""
        if self.model is None or self.model.route not in self.descriptions:
            return
        route = self.model.route
        if self._built_route != route:
            description = self.descriptions[route]
            for page in self.pages.values():
                page.clear()
                page.build(description["parameters"])  # pyright: ignore[reportArgumentType]
                page.build(description["execution"], table="execution")  # pyright: ignore[reportArgumentType]
            self._built_route = route
        self.show_values()
        self._update_steps()

    def show_values(self) -> None:
        """Show the recipe's values on every page."""
        if self.model is not None:
            for page in self.pages.values():
                page.show_values(self.model)

    def edit(self, name: str, table: str, value: object) -> None:
        """Change one value, as an undoable command.

        Parameters
        ----------
        name : str
            The keyword.
        table : str
            Its table.
        value : object
            The new value, or :data:`~solprop_gui.model.recipe.MISSING`.
        """
        if self.model is None:
            return
        old = self.model.get(name, table)
        if old != value:
            self.undo.push(_SetValue(self, name, table, (value, old)))

    def apply_value(self, name: str, table: str, value: object) -> None:
        """Set a value in the recipe and show the change (the commands' action).

        Parameters
        ----------
        name : str
            The keyword.
        table : str
            Its table.
        value : object
            The value, or :data:`~solprop_gui.model.recipe.MISSING`.
        """
        if self.model is None:
            return
        try:
            if value is MISSING:
                self.model.unset(name, table)
            else:
                self.model.set(name, value, table)
        except ValueError as error:
            self.statusBar().showMessage(f"{name}: {error}", 10_000)
        self.show_values()
        self._update_title()
        self.schedule_check()

    # ------------------------------------------------------------------ checking

    def schedule_check(self, *, now: bool = False) -> None:
        """Check the recipe after a pause in typing.

        Parameters
        ----------
        now : bool, optional
            Check at once.
        """
        self.check_timer.start(0 if now else CHECK_DELAY_MS)

    def check(self) -> None:
        """Ask the kernel to check and summarise the current revision."""
        if self.model is None:
            return
        revision = self.model.revision
        self.bridge.set_revision(revision)
        payload = {"text": self.model.to_text(), "base": self.base}
        self._ask_kernel("check", payload, "check", revision=revision)
        self._ask_kernel("summarise", payload, "summarise", revision=revision)

    def _show_issues(self, issues: Sequence[Mapping[str, object]]) -> None:
        """Show solprop's issues at their fields, in the summary and on the steps.

        Parameters
        ----------
        issues : Sequence of Mapping
            The issues.
        """
        self.issues = list(issues)
        self.issue_list.clear()
        for issue in self.issues:
            names = issue_keyword(issue.get("parameter"))
            where = f"{', '.join(names)}: " if names else ""
            badge = _BADGES.get(str(issue.get("severity")), "")
            item = QListWidgetItem(f"{badge}{where}{issue.get('message')}")
            item.setData(Qt.ItemDataRole.UserRole, names[0] if names else None)
            self.issue_list.addItem(item)
        if not self.issues:
            self.issue_list.addItem("✓ solprop accepts this setup.")
        self._update_steps()
        errors = any(issue.get("severity") == "error" for issue in self.issues)
        self.run_panel.set_can_run(self.model is not None and not errors)
        if not errors and self.console.currentWidget() is self.code_view:
            self._request_code()

    def _update_steps(self) -> None:
        """Badge each step with solprop's verdict on its page."""
        colour = self.palette().windowText().color()
        for page in PAGES:
            widget = self.pages.get(page.key)
            if widget is None:
                continue
            verdict = None
            if self.model is not None:
                verdict = _VERDICTS[widget.show_issues(self.issues)]
            self.verdicts[page.key] = verdict
            action = self.step_actions[page.key]
            action.setIcon(icon(page.key, colour, verdict))
            said = _SAID[verdict] if verdict is not None else ""
            action.setToolTip(f"{page.help}{said}")

    def _show_summary(
        self, summary: Mapping[str, object] | None, error: str | None
    ) -> None:
        """Show the summary's numbers.

        Parameters
        ----------
        summary : Mapping or None
            The kernel's summary; ``None`` when it failed.
        error : str or None
            Why it failed.
        """
        if self.model is None:
            return
        lines = [f"<b>{self.model.title or 'Untitled setup'}</b>"]
        entry = card_of(self.model.route, self.model.document.get("build", {}))
        lines.append(entry.title if entry else f"route: {self.model.route}")
        if summary is None:
            lines.append(f"<i>No summary: {error}</i>")
        else:
            energy = float(summary["energy"]) * 1e6  # pyright: ignore[reportArgumentType]
            peak = float(summary["peak_power"]) / 1e9  # pyright: ignore[reportArgumentType]
            lines.append(f"Energy {energy:.4g} µJ")
            lines.append(f"Peak power {peak:.4g} GW")
            design = summary.get("design") or {}
            if isinstance(design, dict) and design:
                lines.append(f"Soliton order {design['soliton_order']:.3g}")
                lines.append(f"β₂ {design['beta2'] * 1e30:.4g} fs²/m")
                lines.append(f"γ {design['gamma'] * 1e3:.4g} /(W km)")  # noqa: RUF001
                intensity = design["peak_intensity"] * 1e-4
                lines.append(f"Peak intensity {intensity:.3g} W/cm²")
                lines.append("<i>at the entrance (solprop.tools.capillary_params)</i>")
            resolved = summary.get("resolved") or {}
            for page in self.pages.values():
                page.show_resolved(resolved)  # pyright: ignore[reportArgumentType]
        self.summary.setText("<br>".join(lines))

    def _go_to_issue(self, item: QListWidgetItem) -> None:
        """Show the step and the field of an issue.

        Parameters
        ----------
        item : PySide6.QtWidgets.QListWidgetItem
            The issue's item.
        """
        name = item.data(Qt.ItemDataRole.UserRole)
        for key, widget in self.pages.items():
            if any(row_name == name for row_name, _ in widget.rows):
                self.show_step(key)
                widget.reveal(name)
                return

    # ------------------------------------------------------------------ code

    def _console_changed(self, _index: int) -> None:
        """Fetch the script when the Python tab is shown.

        Parameters
        ----------
        _index : int
            The tab now shown.
        """
        if self.console.currentWidget() is self.code_view:
            self._request_code()

    def _request_code(self, purpose: str = "to_python") -> None:
        """Ask for the Python script of the current revision.

        Parameters
        ----------
        purpose : str, optional
            ``"to_python"`` to show it, ``"export"`` to save it.
        """
        if self.model is None:
            return
        self._ask_kernel(
            "to_python",
            {"text": self.model.to_text(), "base": self.base},
            purpose,
            revision=self.model.revision,
        )

    def export_python(self) -> None:
        """Save the plain Python script that makes the same run."""
        self._request_code("export")

    def _save_code(self) -> None:
        """Ask where to save the script, and save it."""
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export as Python", "run.py", "Python (*.py)"
        )
        if chosen:
            Path(chosen).write_text(self.code, encoding="utf-8")

    # ------------------------------------------------------------------ saving

    def save(self) -> bool:
        """Save the setup to its file, asking for one if it has none.

        Returns
        -------
        bool
            Whether it was saved.
        """
        if self.model is None:
            return False
        if self.model.path is None:
            return self.save_as()
        self.model.save()
        self._update_title()
        return True

    def save_as(self, path: str | Path | None = None) -> bool:
        """Save the setup to a file the user picks.

        Parameters
        ----------
        path : str or pathlib.Path or None, optional
            The file; ``None`` asks for it.

        Returns
        -------
        bool
            Whether it was saved.
        """
        if self.model is None:
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save setup", "recipe.toml", "solprop recipes (*.toml)"
            )
        if not path:
            return False
        self.model.save(path)
        self.base = str(Path(path).parent)
        self._update_title()
        return True

    def recovery_path(self) -> Path:
        """Return where the recovery copy of an unsaved setup is written.

        Returns
        -------
        pathlib.Path
            The file.
        """
        folder = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        return Path(folder or Path.home()) / "recovery.toml"

    def write_recovery(self) -> None:
        """Write a recovery copy of the setup if it has unsaved changes."""
        if self.model is not None and self.model.dirty:
            self.model.write_recovery(self.recovery_path())

    def _update_title(self) -> None:
        """Show the setup's name, and whether it has unsaved changes."""
        if self.model is None:
            self.setWindowTitle("solprop")
            return
        name = self.model.title or (
            self.model.path.name if self.model.path else "Untitled setup"
        )
        self.setWindowTitle(f"{name}[*] — solprop")
        self.setWindowModified(self.model.dirty)

    def _may_discard(self) -> bool:
        """Ask before replacing a setup with unsaved changes.

        Returns
        -------
        bool
            Whether to go on.
        """
        if self.model is None or not self.model.dirty:
            return True
        answer = self.confirm(
            "Unsaved setup", "The setup has changes that are not saved. Save them?"
        )
        if answer == "save":
            return self.save()
        return answer == "discard"

    # ------------------------------------------------------------------ runs

    def refresh_runs(self) -> None:
        """List the workspace's runs on the start screen."""
        self.results.show_runs(list_runs(self.run_panel.workspace))

    def _run_state(self, state: str) -> None:
        """Bring the live view forward when a run starts.

        Parameters
        ----------
        state : str
            The run's new execution state.
        """
        if state == "starting":
            self.show_view("live")

    def _run_ended(self, path: str) -> None:
        """Show a finished run's results, and list it.

        Parameters
        ----------
        path : str
            Its results file.
        """
        self.results.open(path, show=False)
        self.refresh_runs()
        self.statusBar().showMessage(
            f"Run ended: {Path(path).parent.name} (see the Results view)", 10_000
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's name
        """Ask before closing over a running run or unsaved changes.

        Parameters
        ----------
        event : PySide6.QtGui.QCloseEvent
            The close.
        """
        if self.run_panel.active:
            answer = self.confirm("Run going", "A run is going. Stop it and close?")
            if answer != "stop":
                event.ignore()
                return
            self.run_panel.stop()
            if self.run_panel.client is not None:
                self.run_panel.client.join(30.0)
        if not self._may_discard():
            event.ignore()
            return
        self.settings.setValue("workspace", str(self.run_panel.workspace))
        self.bridge.close()
        event.accept()


def _failure(reply: Reply) -> list[dict[str, object]]:
    """Return a failed check as one issue.

    Parameters
    ----------
    reply : solprop_gui.service.messages.Reply
        The failed reply.

    Returns
    -------
    list of dict
        One error at no parameter.
    """
    return [{"message": reply.error, "parameter": None, "severity": "error"}]
