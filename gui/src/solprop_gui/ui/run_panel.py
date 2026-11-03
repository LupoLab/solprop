# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run: pre-flight, run, stop, resume, and what the live view is fed.

The panel runs the recipe revision current when Run is pressed, in a new folder of the
workspace, in a worker of its own (design 10). The panel's controls sit on the
Propagate step, its actions also on the window's toolbar; its live view, log and
progress are shown by the window, among its views, in its bottom panel and in its
status bar. Its execution state is the window's
(:mod:`solprop_gui.service.states`); whether the file can be continued is asked of
solprop after every ending, and Resume follows that answer alone.
"""

from collections.abc import Callable, Mapping
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.model.workspace import new_run_folder
from solprop_gui.service.client import RunClient
from solprop_gui.service.messages import Event, Reply
from solprop_gui.service.reduce import Display
from solprop_gui.service.remote import RemoteRun
from solprop_gui.service.states import ACTIVE, ExecutionState, advance
from solprop_gui.service.worker import OUTPUT, RunSpec
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.icons import icon
from solprop_gui.views.live import OverviewView

__all__ = ["STOP_GRACE_MS", "RecipeSnapshot", "RunPanel"]

STOP_GRACE_MS = 15_000
"""How long a stop may take before the panel offers to kill the worker, ms."""

RecipeSnapshot = Callable[[], tuple[str, str | None, str | None]]
"""Returns the current recipe's text, its data files' directory and its title."""


def _bytes(count: object) -> str:
    """Write a byte count for a person.

    Parameters
    ----------
    count : object
        Bytes.

    Returns
    -------
    str
        The count in kB, MB or GB.
    """
    value = float(count)  # pyright: ignore[reportArgumentType]
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.3g} {unit}"
        value /= 1000
    return f"{value:.3g} TB"  # pragma: no cover - the loop returns


class RunPanel(QWidget):
    """Start, watch, stop and resume runs of the current recipe.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The service.
    recipe : callable
        Returns the recipe to run (:data:`RecipeSnapshot`).
    workspace : pathlib.Path
        The folder new runs are made in.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    ended : Signal
        Emitted with the results file when a run ends, however it ends.
    state_changed : Signal
        Emitted with the new execution state's value.
    workspace_changed : Signal
        Emitted with the workspace's path when the user picks another.
    """

    ended = Signal(str)
    state_changed = Signal(str)
    workspace_changed = Signal(str)

    def __init__(  # noqa: PLR0915 - the panel's layout
        self,
        bridge: ServiceBridge,
        recipe: RecipeSnapshot,
        workspace: Path,
        parent: QWidget | None = None,
    ) -> None:
        """Make the panel, with no run."""
        super().__init__(parent)
        self.bridge = bridge
        self.recipe = recipe
        self.state = ExecutionState.QUEUED
        self.run_id: str | None = None
        self.folder: Path | None = None
        self.client: RunClient | RemoteRun | None = None
        self.resume_mode: object = None
        self.can_run = False
        self._pending: dict[int, str] = {}
        self._last_index = -1
        self._z_end = 0.0
        self.options = PlotOptions()

        self.workspace = Path(workspace)
        text = self.palette().windowText().color()
        self.run_action = self._action("Run", "run", self.start, QColor("#2e7d32"))
        self.run_action.setToolTip("Run the setup as it is now, in a new run folder.")
        self.stop_action = self._action("Stop", "stop", self.stop, QColor("#c62828"))
        self.stop_action.setToolTip(
            "Stop at the next step; the file keeps what was done, and can be resumed."
        )
        self.kill_action = self._action("Kill", "kill", self.kill, text)
        self.kill_action.setToolTip(
            "End the worker at once. The file may then hold no state to resume from."
        )
        self.resume_action = self._action("Resume", "resume", self.resume, text)
        self.estimate_action = self._action("Estimate", "estimate", self.estimate, text)
        self.estimate_action.setToolTip(
            "Ask solprop what the run will cost: it builds the run and times a few "
            "trial steps, which takes seconds."
        )
        self.run_button = self._button(self.run_action)
        self.stop_button = self._button(self.stop_action)
        self.kill_button = self._button(self.kill_action)
        self.resume_button = self._button(self.resume_action)
        self.estimate_button = self._button(self.estimate_action)
        self.estimate_text = QLabel("")
        self.estimate_text.setObjectName("estimate-text")
        self.estimate_text.setWordWrap(True)
        self.state_label = QLabel("No run yet.")
        self.state_label.setObjectName("run-state")
        self.state_label.setWordWrap(True)
        self.workspace_label = QLabel()
        self.workspace_label.setWordWrap(True)
        self.workspace_button = QToolButton()
        self.workspace_button.setObjectName("workspace")
        self.workspace_button.setText("Change…")
        self.workspace_button.clicked.connect(self.choose_workspace)
        self._show_workspace()

        # Shown by the window: the progress in its status bar, the live view among
        # its views, the log in its bottom panel.
        self.lamp = QLabel("boundary: ok")
        self.lamp.setToolTip("The time window's edges, as solprop's monitor reports.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumWidth(200)
        self.progress.setMaximumHeight(16)
        self.steps = QLabel("")
        self.bar = QProgressBar()
        self.bar.setObjectName("run-progress")
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p %")
        self.bar.setToolTip("How far along the guide the run has gone.")
        self.view = OverviewView()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setObjectName("run-log")

        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        for button in (
            self.run_button,
            self.stop_button,
            self.kill_button,
            self.resume_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        estimate = QHBoxLayout()
        estimate.addWidget(self.estimate_button)
        estimate.addStretch(1)
        where = QHBoxLayout()
        where.addWidget(self.workspace_label, 1)
        where.addWidget(self.workspace_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(buttons)
        layout.addWidget(self.bar)
        layout.addWidget(self.state_label)
        layout.addLayout(estimate)
        layout.addWidget(self.estimate_text)
        layout.addLayout(where)

        self._grace = QTimer(self)
        self._grace.setSingleShot(True)
        self._grace.timeout.connect(self._offer_kill)
        bridge.run_event.connect(self.on_event)
        bridge.replied.connect(self.on_reply)
        self._update_buttons()

    # ------------------------------------------------------------------ state

    @property
    def active(self) -> bool:
        """Return whether a run is going.

        Returns
        -------
        bool
            True from Run until its ending.
        """
        return self.run_id is not None and self.state in ACTIVE

    def set_can_run(self, can_run: bool) -> None:
        """Say whether the recipe can be run: solprop's check found no error.

        Parameters
        ----------
        can_run : bool
            Whether it can.
        """
        self.can_run = can_run
        self._update_buttons()

    def _set_state(self, state: ExecutionState) -> None:
        """Move to a state, and show it.

        Parameters
        ----------
        state : ExecutionState
            The state.
        """
        if state != self.state:
            self.state = state
            self.state_changed.emit(state.value)
        self._update_buttons()

    def _action(
        self, text: str, drawing: str, slot: Callable[[], object], colour: QColor
    ) -> QAction:
        """Make one of the panel's actions, shared with the window's toolbar.

        Parameters
        ----------
        text : str
            Its text.
        drawing : str
            Its icon's drawing (:data:`~solprop_gui.ui.icons.DRAWINGS`).
        slot : callable
            What it does.
        colour : PySide6.QtGui.QColor
            Its icon's colour.

        Returns
        -------
        PySide6.QtGui.QAction
            The action.
        """
        action = QAction(icon(drawing, colour), text, self)
        action.setObjectName(drawing)
        action.triggered.connect(lambda _checked=False: slot())
        return action

    @staticmethod
    def _button(action: QAction) -> QToolButton:
        """Make a button for an action.

        Parameters
        ----------
        action : PySide6.QtGui.QAction
            The action; the button follows its text, icon and state.

        Returns
        -------
        PySide6.QtWidgets.QToolButton
            The button.
        """
        button = QToolButton()
        button.setDefaultAction(action)
        button.setObjectName(action.objectName())
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        return button

    def _update_buttons(self) -> None:
        """Enable the actions the state allows."""
        active = self.active
        self.run_action.setEnabled(self.can_run and not active)
        self.estimate_action.setEnabled(self.can_run)
        self.stop_action.setEnabled(active and self.state != ExecutionState.STOPPING)
        self.kill_action.setEnabled(
            active
            and self.state == ExecutionState.STOPPING
            and not self._grace.isActive()
        )
        self.resume_action.setEnabled(not active and self.resume_mode is not None)

    def _show_workspace(self) -> None:
        """Say where new runs go."""
        self.workspace_label.setText(f"Runs go in the workspace {self.workspace.name}")
        self.workspace_label.setToolTip(str(self.workspace))

    def choose_workspace(self) -> None:
        """Pick the folder new runs are made in."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Workspace", str(self.workspace)
        )
        if chosen:
            self.set_workspace(Path(chosen))

    def set_workspace(self, folder: Path) -> None:
        """Make new runs in a folder.

        Parameters
        ----------
        folder : pathlib.Path
            The workspace.
        """
        self.workspace = folder
        self._show_workspace()
        self.workspace_changed.emit(str(folder))

    # ------------------------------------------------------------------ actions

    def estimate(self) -> None:
        """Ask solprop for the run's cost."""
        text, base, _ = self.recipe()
        self.estimate_text.setText("Estimating: building the run and timing steps…")
        request = self.bridge.request(
            "estimate", {"text": text, "base": base}, key="estimate"
        )
        self._pending[request] = "estimate"

    def start(self) -> None:
        """Run the current recipe in a new folder of the workspace."""
        if self.active:
            return
        text, base, title = self.recipe()
        folder = new_run_folder(self.workspace.expanduser(), title)
        self.view.clear()
        self.log.clear()
        self._launch(
            RunSpec(
                folder.name,
                str(folder),
                recipe=text,
                base=base,
                display=self.display(),
            ),
            fresh=True,
        )

    def resume(self) -> None:
        """Continue the last run from its file, as solprop said it can."""
        if (
            self.active
            or self.folder is None
            or self.run_id is None
            or self.resume_mode is None
        ):
            return
        mode = self.resume_mode
        self.resume_mode = None
        self._launch(
            RunSpec(
                self.run_id,
                str(self.folder),
                resume=mode,  # pyright: ignore[reportArgumentType]
                display=self.display(),
            ),
            fresh=False,
        )

    def display(self) -> Display:
        """Return the bins and readings a new run's live views are reduced with.

        Returns
        -------
        solprop_gui.service.reduce.Display
            The plot options' bandpass.
        """
        return Display(bandpass=self.options.bandpass)

    def apply_options(self, options: PlotOptions) -> None:
        """Take new plot options, for the live view and the next run.

        The live view redraws with them at once, on either axis. A new band is the
        next run's, and the last run's file is read again with it once it has
        ended.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        self.options = options
        self.view.apply_options(options)
        self.reread()

    def reread(self) -> None:
        """Read the ended run's file again when its rows have another band."""
        if (
            self.active
            or self.folder is None
            or not self.view.rows
            or self.view.band == self.options.bandpass
            or not (self.folder / OUTPUT).exists()
        ):
            return
        payload = {"path": str(self.folder / OUTPUT), "bandpass": self.options.bandpass}
        request = self.bridge.request("overview", payload, key="live-overview")
        self._pending[request] = "overview"

    def _progress(self, value: int, *, reset: bool = False) -> None:
        """Show how far the run has gone, never going back but on a reset.

        Parameters
        ----------
        value : int
            The share done, in thousandths.
        reset : bool, optional
            Whether to start again from the value.
        """
        if not reset:
            value = max(value, self.bar.value())
        for bar in (self.progress, self.bar):
            bar.setValue(min(max(value, 0), 1000))

    def _launch(self, spec: RunSpec, *, fresh: bool) -> None:
        """Start a worker.

        Parameters
        ----------
        spec : solprop_gui.service.worker.RunSpec
            What it runs.
        fresh : bool
            Whether it is a new run, rather than a continued one.
        """
        self.run_id = spec.run_id
        self.folder = Path(spec.folder)
        if fresh:
            self._last_index = -1
        self._z_end = 0.0
        self.lamp.setText("boundary: ok")
        self.lamp.setStyleSheet("")
        self._progress(0, reset=True)
        self.state = ExecutionState.QUEUED
        self._set_state(advance(self.state, "launch"))
        self.state_label.setText(f"{spec.run_id}: starting")
        self.client = self.bridge.start_run(spec)

    def attach(self, run: Mapping[str, object]) -> bool:
        """Take over a run the service holds: its events, replayed from its start.

        Parameters
        ----------
        run : Mapping
            The run, as the service lists it: ``run_id`` and ``folder``.

        Returns
        -------
        bool
            Whether it was attached (only the service holds runs to attach to).
        """
        if self.active:
            return False
        spec = RunSpec(str(run["run_id"]), str(run["folder"]))
        client = self.bridge.attach(spec)
        if client is None:
            return False
        self.run_id = spec.run_id
        self.folder = Path(spec.folder)
        self._last_index = -1
        self.view.clear()
        self.log.clear()
        self.client = client
        self.state = ExecutionState.RUNNING
        self._set_state(ExecutionState.RUNNING)
        self.state_label.setText(f"{spec.run_id}: re-attached")
        self._update_buttons()
        return True

    def stop(self) -> None:
        """Ask the run to stop at its next step, through solprop's stop flag."""
        if self.client is None or not self.active:
            return
        self.client.stop()
        self._set_state(advance(self.state, "stop"))
        self.state_label.setText(f"{self.run_id}: stopping")
        self._grace.start(STOP_GRACE_MS)
        self._update_buttons()

    def _offer_kill(self) -> None:
        """Offer to kill a worker that has not stopped in the grace period."""
        self._update_buttons()
        if self.state == ExecutionState.STOPPING:
            self.state_label.setText(
                f"{self.run_id}: still stopping; Kill ends it at once"
            )

    def kill(self) -> None:
        """End the worker at once."""
        if self.client is not None:
            self.client.kill()

    # ------------------------------------------------------------------ events

    def on_event(self, event: Event) -> None:
        """Take one event of the panel's run.

        Parameters
        ----------
        event : solprop_gui.service.messages.Event
            The event; events of other runs are ignored.
        """
        if event.run_id != self.run_id:
            return
        data = event.data
        if event.kind == "axes":
            self.view.set_axes(data)
        elif event.kind == "row":
            index = int(data["index"])  # pyright: ignore[reportArgumentType]
            if index > self._last_index + 1 and self.client is not None:
                # Rows were dropped on the way: the worker replays them.
                self.client.subscribe(self._last_index + 1)
            self._last_index = max(self._last_index, index)
            self.view.add_row(data)
            z = float(data["z"])  # pyright: ignore[reportArgumentType]
            self.steps.setText(
                f"z {z:.4g} / {self._z_end:.4g} m   steps {data['steps']} "
                f"({data['rejected']} rejected)"
            )
            if self._z_end > 0:
                self._progress(int(1000 * z / self._z_end))
        elif event.kind == "progress":
            z, z_end = float(data["z"]), float(data["z_end"])  # pyright: ignore[reportArgumentType]
            if z_end > 0:
                self._progress(int(1000 * z / z_end))
            self.steps.setText(
                f"z {z:.4g} / {z_end:.4g} m   steps {data['steps']} "
                f"({data['rejected']} rejected)"
            )
        elif event.kind == "log":
            self._show_log(data)
        elif event.kind == "started":
            self._z_end = float(data["z_end"])  # pyright: ignore[reportArgumentType]
            self.view.set_extent(float(data.get("z_start", 0.0)), self._z_end)  # pyright: ignore[reportArgumentType]
            self.state_label.setText(f"{self.run_id}: running")
        elif event.kind == "scan_started":
            self.state_label.setText(f"{self.run_id}: scanning {data['points']} points")
        elif event.kind == "point":
            self._show_point(data)
        # A scan's start moves the run on as a run's does.
        kind = "started" if event.kind == "scan_started" else event.kind
        self._set_state(advance(self.state, kind))
        if event.terminal:
            self._ended(event)

    def _show_log(self, data: Mapping[str, object]) -> None:
        """Add a log line, and light the boundary lamp for a boundary warning.

        Parameters
        ----------
        data : Mapping
            The ``log`` event's data: ``level`` and ``message``.
        """
        message = f"{data.get('level')}: {data.get('message')}"
        self.log.appendPlainText(message)
        if "Boundary" in str(data.get("message")):
            error = "BoundaryError" in str(data.get("message"))
            self.lamp.setText("boundary: stopped" if error else "boundary: warning")
            self.lamp.setStyleSheet("color: #c62828;" if error else "color: #b26a00;")

    def _show_point(self, data: Mapping[str, object]) -> None:
        """Show a scan point's start, with its values, or its end.

        Parameters
        ----------
        data : Mapping
            The ``point`` event's data.
        """
        if data.get("status") is None:
            values: Mapping[str, float] = data.get("values") or {}  # pyright: ignore[reportAssignmentType]
            shown = ", ".join(f"{k} = {v:.4g}" for k, v in values.items())
            self.steps.setText(f"point {data['index']}: {shown}")
            return
        points = int(data.get("points") or 0)  # pyright: ignore[reportArgumentType]
        if points:
            done = int(data["done"])  # pyright: ignore[reportArgumentType]
            self._progress(int(1000 * done / points))
        if data["status"] == "failed":
            self.log.appendPlainText(f"point {data['index']} failed")

    def _ended(self, event: Event) -> None:
        """Show how the run ended, and ask solprop whether it can be continued.

        Parameters
        ----------
        event : solprop_gui.service.messages.Event
            The terminal event.
        """
        self._grace.stop()
        message = str(event.data.get("message") or "")
        words = {
            "finished": "complete",
            "stopped": "stopped",
            "failed": "failed",
            "lost": "the worker was lost",
        }[event.kind]
        text = f"{self.run_id}: {words}"
        if event.kind == "failed" and message:
            text += f": {message.splitlines()[0]}"
            detail = event.data.get("detail")
            if detail:
                self.log.appendPlainText(str(detail))
        self.state_label.setText(text)
        if event.kind == "finished":
            self._progress(1000)
        if self.folder is not None and event.data.get("scan"):
            # A scan's points are run again with retry_failed, not resumed.
            self.ended.emit(str(event.data["path"]))
        elif self.folder is not None:
            path = str(self.folder / OUTPUT)
            request = self.bridge.request(
                "check_resume", {"path": path}, key="panel-resume"
            )
            self._pending[request] = "check_resume"
            self.ended.emit(path)
            self.reread()  # a band changed while it ran
        self._update_buttons()

    def on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the panel's requests.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        kind = self._pending.pop(reply.request_id, None)
        if kind == "overview" and reply.ok and isinstance(reply.value, dict):
            value = reply.value
            extent = self.view.spectrum_map.extent
            self.view.set_rows(value.get("axes", {}), value.get("rows", []))  # pyright: ignore[reportArgumentType]
            if extent is not None:
                self.view.set_extent(*extent)
        elif kind == "estimate":
            self._show_estimate(reply)
        elif kind == "check_resume":
            value = reply.value if reply.ok else None
            if isinstance(value, dict) and value.get("resumable"):
                self.resume_mode = value.get("mode")
                self.resume_action.setToolTip(
                    f"Continue from z = {value.get('z'):.4g} m."
                )
            else:
                self.resume_mode = None
                reason = value.get("reason") if isinstance(value, dict) else reply.error
                self.resume_action.setToolTip(f"Cannot resume: {reason}")
            self._update_buttons()

    def _show_estimate(self, reply: Reply) -> None:
        """Show solprop's estimate of the run.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The kernel's reply.
        """
        if not reply.ok or not isinstance(reply.value, dict):
            self.estimate_text.setText(f"No estimate: {reply.error}")
            return
        value = reply.value
        if not value.get("admitted"):
            self.estimate_text.setText(f"Refused: {value.get('refusal')}")
            return
        parts = [
            f"state {tuple(value['shape'])}",
            f"{value['n_planes']} planes",
            f"memory {_bytes(value['memory'])}",
            f"output {_bytes(value['output_bytes'])}",
            f"backend {value['backend']}",
        ]
        if value.get("rhs") is not None:
            parts.append(
                f"{float(value['rhs']) * 1e3:.3g} ms per evaluation, "
                f"{value['calls_per_step']} per step"
            )
        self.estimate_text.setText("; ".join(parts))
