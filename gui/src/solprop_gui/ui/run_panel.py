# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run on the Propagate page: pre-flight, run, stop, resume and the live view.

The panel runs the recipe revision current when Run is pressed, in a new folder of the
workspace, in a worker of its own (design 10). Its execution state is the window's
(:mod:`solprop_gui.service.states`); whether the file can be continued is asked of
solprop after every ending, and Resume follows that answer alone.
"""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.workspace import new_run_folder
from solprop_gui.service.client import RunClient
from solprop_gui.service.messages import Event, Reply
from solprop_gui.service.states import ACTIVE, ExecutionState, advance
from solprop_gui.service.worker import OUTPUT, RunSpec
from solprop_gui.ui.bridge import ServiceBridge
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
        self.client: RunClient | None = None
        self.resume_mode: object = None
        self.can_run = False
        self._pending: dict[int, str] = {}
        self._last_index = -1
        self._z_end = 0.0

        self.workspace = Path(workspace)
        self.workspace_button = QToolButton()
        self.workspace_button.setObjectName("workspace")
        self.workspace_button.setText("Workspace…")
        self.workspace_button.clicked.connect(self.choose_workspace)
        self._show_workspace()
        self.estimate_button = QToolButton()
        self.estimate_button.setObjectName("estimate")
        self.estimate_button.setText("Estimate")
        self.estimate_button.setToolTip(
            "Ask solprop what the run will cost: it builds the run and times a few "
            "trial steps, which takes seconds."
        )
        self.estimate_button.clicked.connect(self.estimate)
        self.estimate_text = QLabel("")
        self.estimate_text.setObjectName("estimate-text")

        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("run")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self.stop)
        self.kill_button = QPushButton("Kill")
        self.kill_button.setObjectName("kill")
        self.kill_button.setToolTip(
            "End the worker at once. The file may then hold no state to resume from."
        )
        self.kill_button.clicked.connect(self.kill)
        self.resume_button = QPushButton("Resume")
        self.resume_button.setObjectName("resume")
        self.resume_button.clicked.connect(self.resume)
        self.state_label = QLabel("No run yet.")
        self.state_label.setObjectName("run-state")
        self.lamp = QLabel("boundary: ok")
        self.lamp.setToolTip("The time window's edges, as solprop's monitor reports.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumWidth(240)
        self.steps = QLabel("")
        self.log_button = QToolButton()
        self.log_button.setText("Log")
        self.log_button.setCheckable(True)
        self.log_button.setToolTip("Show solprop's log of the run.")

        bar = QHBoxLayout()
        for widget in (
            self.run_button,
            self.stop_button,
            self.kill_button,
            self.resume_button,
        ):
            bar.addWidget(widget)
        bar.addWidget(self.progress)
        bar.addWidget(self.steps)
        bar.addStretch(1)
        bar.addWidget(self.lamp)
        bar.addWidget(self.estimate_button)
        bar.addWidget(self.workspace_button)
        bar.addWidget(self.log_button)
        status = QHBoxLayout()
        status.addWidget(self.state_label)
        status.addStretch(1)
        status.addWidget(self.estimate_text)

        self.view = OverviewView()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setObjectName("run-log")
        self.log.setMaximumHeight(160)
        self.log.hide()
        self.log_button.toggled.connect(self.log.setVisible)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(bar)
        layout.addLayout(status)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.log)

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

    def _update_buttons(self) -> None:
        """Enable the buttons the state allows."""
        active = self.active
        self.run_button.setEnabled(self.can_run and not active)
        self.estimate_button.setEnabled(self.can_run)
        self.stop_button.setEnabled(active and self.state != ExecutionState.STOPPING)
        self.kill_button.setEnabled(
            active
            and self.state == ExecutionState.STOPPING
            and not self._grace.isActive()
        )
        self.resume_button.setEnabled(not active and self.resume_mode is not None)

    def _show_workspace(self) -> None:
        """Say on the workspace button where runs go."""
        self.workspace_button.setToolTip(f"New runs go in {self.workspace}")

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
            RunSpec(folder.name, str(folder), recipe=text, base=base), fresh=True
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
            RunSpec(self.run_id, str(self.folder), resume=mode),  # pyright: ignore[reportArgumentType]
            fresh=False,
        )

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
        self.progress.setValue(0)
        self.state = ExecutionState.QUEUED
        self._set_state(advance(self.state, "launch"))
        self.state_label.setText(f"{spec.run_id}: starting")
        self.client = self.bridge.start_run(spec)

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
                self.progress.setValue(
                    max(self.progress.value(), int(1000 * z / self._z_end))
                )
        elif event.kind == "progress":
            z, z_end = float(data["z"]), float(data["z_end"])  # pyright: ignore[reportArgumentType]
            if z_end > 0:
                self.progress.setValue(int(1000 * z / z_end))
            self.steps.setText(
                f"z {z:.4g} / {z_end:.4g} m   steps {data['steps']} "
                f"({data['rejected']} rejected)"
            )
        elif event.kind == "log":
            message = f"{data.get('level')}: {data.get('message')}"
            self.log.appendPlainText(message)
            if "Boundary" in str(data.get("message")):
                error = "BoundaryError" in str(data.get("message"))
                self.lamp.setText("boundary: stopped" if error else "boundary: warning")
                self.lamp.setStyleSheet(
                    "color: #c62828;" if error else "color: #b26a00;"
                )
        elif event.kind == "started":
            self._z_end = float(data["z_end"])  # pyright: ignore[reportArgumentType]
            self.state_label.setText(f"{self.run_id}: running")
        self._set_state(advance(self.state, event.kind))
        if event.terminal:
            self._ended(event)

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
            self.progress.setValue(1000)
        if self.folder is not None:
            path = str(self.folder / OUTPUT)
            request = self.bridge.request(
                "check_resume", {"path": path}, key="panel-resume"
            )
            self._pending[request] = "check_resume"
            self.ended.emit(path)
        self._update_buttons()

    def on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the panel's requests.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        kind = self._pending.pop(reply.request_id, None)
        if kind == "estimate":
            self._show_estimate(reply)
        elif kind == "check_resume":
            value = reply.value if reply.ok else None
            if isinstance(value, dict) and value.get("resumable"):
                self.resume_mode = value.get("mode")
                self.resume_button.setToolTip(
                    f"Continue from z = {value.get('z'):.4g} m."
                )
            else:
                self.resume_mode = None
                reason = value.get("reason") if isinstance(value, dict) else reply.error
                self.resume_button.setToolTip(f"Cannot resume: {reason}")
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
