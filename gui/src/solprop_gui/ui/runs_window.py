# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Runs window: every run on every machine, and what can be done with each (13.4).

Each machine lists its own runs from their folders (on this computer here, elsewhere
through its helper), so the list is the same whoever started a run and whenever: a
run started from another window, or left going overnight, is there with its state.
A run can be opened (followed live, or its results read on its machine), stopped,
cancelled, copied here, or removed. A machine that cannot be reached says so and is
tried again at the next refresh.
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from solprop.service.machines import Machine

from solprop_gui.ui.machines_hub import MachineHub

__all__ = ["REFRESH_MS", "RunsWindow", "progress_text", "state_text"]

REFRESH_MS: Final = 5000
"""How often the list is read again while it is shown, ms."""

_GOING: Final = frozenset({"queued", "starting", "running"})

_COLUMNS: Final = ("Machine", "Run", "State", "Progress", "Started")


def state_text(run: Mapping[str, Any]) -> str:
    """Return a run's state as the list says it.

    Parameters
    ----------
    run : Mapping
        Its summary (:func:`solprop.service.runs.summarise`).

    Returns
    -------
    str
        The state, with why it waits, why it ended or that it is stopping.

    Examples
    --------
    >>> state_text({"state": "queued", "detail": "Priority"})
    'queued (Priority)'
    >>> state_text({"state": "running", "detail": "", "stopping": True})
    'stopping'
    """
    state = str(run.get("state", ""))
    if state in _GOING and run.get("stopping"):
        return "stopping"
    detail = str(run.get("detail") or "")
    return f"{state} ({detail.splitlines()[0]})" if detail else state


def progress_text(run: Mapping[str, Any]) -> str:
    """Return how far a run has gone.

    Parameters
    ----------
    run : Mapping
        Its summary.

    Returns
    -------
    str
        A scan's points done, or a run's share of its length; empty when unknown.

    Examples
    --------
    >>> progress_text({"points": 8, "done": 2})
    '2 of 8 points'
    >>> progress_text({"progress": {"z": 0.05, "z_end": 0.2}})
    '25 %'
    """
    if run.get("points"):
        return f"{run.get('done') or 0} of {run['points']} points"
    progress = run.get("progress")
    if isinstance(progress, Mapping) and progress.get("z_end"):
        share = float(progress.get("z", 0)) / float(progress["z_end"])
        return f"{100 * share:.0f} %"
    return ""


def _started(run: Mapping[str, Any]) -> str:
    """Return when a run started, or was submitted.

    Parameters
    ----------
    run : Mapping
        Its summary.

    Returns
    -------
    str
        The local date and time; empty when unknown.
    """
    import time  # noqa: PLC0415 - light

    when = run.get("started") or run.get("submitted")
    if not isinstance(when, int | float):
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(when))


class RunsWindow(QWidget):
    """The runs of every machine the window remembers.

    Parameters
    ----------
    machines : Callable
        Returns the machines, this computer first.
    hub : solprop_gui.ui.machines_hub.MachineHub
        The window's connections.
    workspace : Callable
        Returns this computer's workspace.
    cache : Callable
        Returns where copies of other machines' runs go, for a machine and a run.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner; the window is a window of its own.

    Attributes
    ----------
    open_wanted : Signal
        Emitted with a machine and a run's summary when the run is to be opened.
    fetched : Signal
        Emitted with a copy's path once a run's results are copied here.
    """

    open_wanted = Signal(object, object)
    fetched = Signal(str)

    def __init__(
        self,
        machines: Callable[[], Sequence[Machine]],
        hub: MachineHub,
        workspace: Callable[[], Path],
        cache: Callable[[Machine, Mapping[str, Any]], Path],
        parent: QWidget | None = None,
    ) -> None:
        """Lay out the list and its actions."""
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Runs")
        self.resize(820, 420)
        self._machines = machines
        self.hub = hub
        self._workspace = workspace
        self._cache = cache
        self._rows: list[tuple[Machine, dict[str, Any]]] = []
        self._listed: dict[str, list[dict[str, Any]]] = {}
        self._notes: dict[str, str] = {}
        self._local = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="solprop-runs"
        )

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setObjectName("runs-table")
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(lambda *_: self._open())

        self.note = QLabel()
        self.note.setObjectName("note")
        self.note.setWordWrap(True)
        self.buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout()
        for name, action in (
            ("Open", self._open),
            ("Stop", self._stop),
            ("Cancel job", self._cancel),
            ("Fetch a copy", self._fetch),
            ("Remove", self._remove),
        ):
            button = QPushButton(name)
            button.clicked.connect(action)
            self.buttons[name] = button
            row.addWidget(button)
        row.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.note)
        layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        hub.state_changed.connect(self._machine_state)
        self._update_buttons()

    # ------------------------------------------------------------------ listing

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt's name
        """Read the list as the window appears, and keep it current while shown.

        Parameters
        ----------
        event : PySide6.QtGui.QShowEvent
            The event.
        """
        super().showEvent(event)  # pyright: ignore[reportArgumentType]
        self.refresh()
        self._timer.start()

    def hideEvent(self, event: object) -> None:  # noqa: N802 - Qt's name
        """Stop reading the list while hidden.

        Parameters
        ----------
        event : PySide6.QtGui.QHideEvent
            The event.
        """
        self._timer.stop()
        super().hideEvent(event)  # pyright: ignore[reportArgumentType]

    def refresh(self) -> None:
        """Ask every machine for its runs."""
        for machine in self._machines():
            if machine.local:
                self._list_here(machine)
            else:
                self.hub.ensure(machine)
                if self.hub.states.get(machine.name) == "up":
                    self.hub.call(
                        machine.name,
                        "list_runs",
                        {"workspace": machine.workspace},
                        lambda value, error, m=machine: self._arrived(m, value, error),
                    )

    def _list_here(self, machine: Machine) -> None:
        """List this computer's runs on a thread, and show them when read.

        Parameters
        ----------
        machine : solprop.service.machines.Machine
            This computer.
        """
        from solprop.service.runs import list_runs  # noqa: PLC0415 - light

        workspace = self._workspace().expanduser()
        future = self._local.submit(list_runs, workspace, None)

        def read() -> None:
            """Show the listing once it is read, looking again shortly if not."""
            if future.done():
                error = future.exception()
                value = None if error is not None else future.result()
                self._arrived(machine, value, error)
            else:
                QTimer.singleShot(50, read)

        QTimer.singleShot(50, read)

    def _arrived(self, machine: Machine, value: object, error: object) -> None:
        """Show a machine's runs.

        Parameters
        ----------
        machine : solprop.service.machines.Machine
            The machine.
        value : object
            Its runs' summaries.
        error : object
            Why they could not be listed, or ``None``.
        """
        if error is not None:
            self._notes[machine.name] = f"{machine.name}: {error}"
        else:
            self._notes.pop(machine.name, None)
            self._listed[machine.name] = list(value)  # pyright: ignore[reportArgumentType, reportCallIssue]
        self._show()

    def _machine_state(self, name: str, state: str, detail: object) -> None:
        """Note a machine that cannot be reached, and list one that comes back.

        Parameters
        ----------
        name : str
            The machine.
        state : str
            Its connection's state.
        detail : object
            Why it is down or refused.
        """
        if state in {"down", "refused"}:
            self._notes[name] = f"{name} is {state}: {detail}"
            self._show()
        elif state == "up" and self.isVisible():
            self._notes.pop(name, None)
            self.refresh()

    def _show(self) -> None:
        """Draw the list, keeping the selection."""
        selected = self.selected()
        key = None if selected is None else (selected[0].name, selected[1]["folder"])
        machines = {m.name: m for m in self._machines()}
        self._rows = [
            (machines[name], run)
            for name, runs in self._listed.items()
            if name in machines
            for run in runs
        ]
        self.table.setRowCount(len(self._rows))
        for index, (machine, run) in enumerate(self._rows):
            where = "This computer" if machine.local else machine.name
            cells = (
                where,
                str(run.get("title") or run["run_id"]),
                state_text(run),
                progress_text(run),
                _started(run),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(str(run.get("folder")))
                self.table.setItem(index, column, item)
            if key == (machine.name, run["folder"]):
                self.table.selectRow(index)
        self.note.setText("\n".join(self._notes.values()))
        self._update_buttons()

    # ------------------------------------------------------------------ actions

    def selected(self) -> tuple[Machine, dict[str, Any]] | None:
        """Return the selected run and its machine.

        Returns
        -------
        tuple or None
            The machine and the run's summary; ``None`` with nothing selected.
        """
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self._rows):
            return None
        return self._rows[rows[0].row()]

    def _update_buttons(self) -> None:
        """Offer what the selected run allows."""
        chosen = self.selected()
        run = {} if chosen is None else chosen[1]
        going = run.get("state") in _GOING
        remote = chosen is not None and not chosen[0].local
        self.buttons["Open"].setEnabled(chosen is not None)
        self.buttons["Stop"].setEnabled(going and run.get("record", False))
        self.buttons["Cancel job"].setEnabled(going and run.get("record", False))
        self.buttons["Fetch a copy"].setEnabled(remote and bool(run.get("output")))
        self.buttons["Remove"].setEnabled(chosen is not None and not going)

    def _open(self) -> None:
        """Open the selected run: follow it while it goes, read its results after."""
        chosen = self.selected()
        if chosen is not None:
            self.open_wanted.emit(*chosen)

    def _act(self, kind: str) -> None:
        """Send a request about the selected run to its machine.

        Parameters
        ----------
        kind : str
            ``"stop"`` or ``"cancel"``.
        """
        chosen = self.selected()
        if chosen is None:
            return
        machine, run = chosen

        def done(value: object, error: object) -> None:
            """Say what came of the request, and list again.

            Parameters
            ----------
            value : object
                The call's value.
            error : object
                Why it failed, or ``None``.
            """
            del value
            if error is not None:
                self.note.setText(f"{run['run_id']}: {error}")
            self.refresh()

        if machine.local:
            from solprop.service.record import request_stop  # noqa: PLC0415 - light

            if kind == "stop":
                request_stop(Path(run["folder"]))
                self.refresh()
                return
        self.hub.ensure(machine)
        self.hub.call(machine.name, kind, {"folder": run["folder"]}, done)

    def _stop(self) -> None:
        """Ask the selected run to stop; it can be resumed."""
        self._act("stop")

    def _cancel(self) -> None:
        """End the selected run's job; its file can still be resumed."""
        self._act("cancel")

    def _remove(self) -> None:
        """Delete the selected run's folder, after asking."""
        chosen = self.selected()
        if chosen is None:
            return
        machine, run = chosen
        where = "this computer" if machine.local else machine.name
        answer = QMessageBox.question(
            self,
            "Remove a run",
            f"Delete {run['folder']} on {where}, with its results? "
            "This cannot be undone.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.hub.ensure(machine)
        self.hub.call(
            machine.name,
            "remove",
            {"folder": run["folder"]},
            lambda _value, error: self._removed(run, error),
        )

    def _removed(self, run: Mapping[str, Any], error: object) -> None:
        """Say whether a run was removed, and list again.

        Parameters
        ----------
        run : Mapping
            The run.
        error : object
            Why it was not, or ``None``.
        """
        if error is not None:
            self.note.setText(f"{run['run_id']}: {error}")
        self.refresh()

    def _fetch(self) -> None:
        """Copy the selected run's results here, showing how far the copy has got."""
        chosen = self.selected()
        if chosen is None or not chosen[1].get("output"):
            return
        machine, run = chosen
        target = self._cache(machine, run)
        dialog = QProgressDialog(f"Copying {run['run_id']}…", "Cancel", 0, 1000, self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(300)

        def progress(have: int, total: int) -> None:
            """Show how far the copy has got, on the window's thread.

            Parameters
            ----------
            have : int
                The bytes copied.
            total : int
                The file's size, bytes.
            """
            share = int(1000 * have / total) if total else 1000
            QTimer.singleShot(0, lambda: dialog.setValue(share))

        def done(value: object, error: object) -> None:
            """Say what came of the request, and list again.

            Parameters
            ----------
            value : object
                The call's value.
            error : object
                Why it failed, or ``None``.
            """
            dialog.reset()
            if error is not None:
                self.note.setText(f"{run['run_id']}: {error}")
                return
            self.fetched.emit(str(value))

        cancelled = self.hub.fetch(
            machine.name, str(run["output"]), target, progress, done
        )
        dialog.canceled.connect(cancelled.set)
