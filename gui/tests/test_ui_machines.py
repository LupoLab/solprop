# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Machines in the window: the Runs on line, the machines dialog, the Runs window.

The machine used is this computer's own helper (``Machine("local")``), which is how
a remote one is reached short of ``ssh`` itself; solprop's service tests drive the
helper and the connection in depth.
"""

import shutil
from pathlib import Path
from typing import Any

from pytestqt.qtbot import QtBot
from solprop.service.machines import LOCAL, Machine
from solprop.service.messages import Reply
from solprop.service.worker import RunSpec

from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.machines_dialog import MachinesDialog
from solprop_gui.ui.machines_hub import MachineHub, MachineStore
from solprop_gui.ui.run_options import MACHINES_ITEM, RunOptions
from solprop_gui.ui.runs_window import RunsWindow
from solprop_gui.ui.theme import app_settings
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture

SLURM = Machine(
    "cluster",
    "me@cluster",
    launcher="slurm",
    resources={"partition": "gpu", "cpus": 8, "gres": "gpu:1", "time": "12:00:00"},
)


def test_machines_are_remembered_with_this_computer_first():
    store = MachineStore(app_settings())
    assert [m.name for m in store.machines()] == [LOCAL]
    store.save([Machine(LOCAL), SLURM])
    assert store.get("cluster") == SLURM and store.get("nowhere") is None
    app_settings().setValue("machines", "not json")
    assert [m.name for m in store.machines()] == [LOCAL]


def test_the_runs_on_line_offers_the_machines(qtbot: QtBot, tmp_path: Path):
    options = RunOptions()
    qtbot.addWidget(options)
    options.set_machines([Machine(LOCAL), SLURM], "cluster")
    assert options.machine.currentData() == "cluster"
    options.show_machine(SLURM, "down", "no route to host")
    assert options.connection.text() == "not reachable; retrying"
    assert (
        options.resources.text() == "Slurm: gpu · 8 CPUs · gpu:1 · 12:00:00 · Change…"
    )
    hello = {"host": "login1", "solprop": "0.2.0", "warning": "another build"}
    options.show_machine(SLURM, "up", hello)
    assert options.connection.text() == "another build"
    assert options.connection.toolTip() == "login1, solprop 0.2.0"
    options.show_machine(Machine(LOCAL))
    assert options.resources.isHidden() and options.connection.text() == ""
    with qtbot.waitSignal(options.machine_chosen) as chosen:
        options.machine.activated.emit(0)
    assert chosen.args == [LOCAL]
    with qtbot.waitSignal(options.machines_wanted):
        options.machine.activated.emit(options.machine.findData(MACHINES_ITEM))


def test_the_machines_dialog_adds_edits_and_removes(qtbot: QtBot):
    hub = MachineHub()
    dialog = MachinesDialog([Machine(LOCAL), SLURM], hub, current="cluster")
    qtbot.addWidget(dialog)
    assert dialog.list.count() == 1 and dialog.slurm.isVisibleTo(dialog)
    assert dialog.partition.currentText() == "gpu" and dialog.cpus.value() == 8
    dialog.cpus.setValue(16)
    dialog.extra.setPlainText("--account=abc\n\n")
    dialog._add("cluster", "")  # a typed host, named apart from the first
    assert [m.name for m in dialog.machines] == [LOCAL, "cluster", "cluster 2"]
    assert dialog.machines[1].resources["cpus"] == 16
    assert dialog.machines[1].resources["extra"] == ("--account=abc",)
    dialog.destination.setText("")
    dialog._test()
    assert "SSH host first" in dialog.status.text()
    dialog._state("cluster 2", "down", "timed out")
    assert dialog.status.text() == "down: timed out"
    dialog._state("cluster 2", "up", {"host": "h", "solprop": "0.2.0", "slurm": False})
    assert "without Slurm" in dialog.status.text()
    dialog._state("elsewhere", "up", {})
    dialog._remove()
    assert [m.name for m in dialog.machines] == [LOCAL, "cluster"]
    dialog.accept()
    assert dialog.machines[1].resources["cpus"] == 16
    hub.close()


def test_a_request_about_a_machine_s_run_goes_to_its_kernel(qtbot: QtBot):
    bridge = ServiceBridge()
    bridge.know_folder("cluster", "/runs/a")
    assert bridge.machine_of({"path": "/runs/a/run.h5"}) == "cluster"
    assert bridge.machine_of({"path": "/runs/ab/run.h5"}) is None
    assert bridge.machine_of({"base": "/runs/a"}) == "cluster"
    assert bridge.machine_of({"machine": "cluster"}) is None  # not connected
    assert bridge.machine_of(None) is None and bridge.machine_of({"path": 3}) is None
    sent: list[Any] = []
    bridge.hub.kernel = lambda name, request: sent.append((name, request))  # type: ignore[method-assign]
    identity = bridge.request("describe_output", {"path": "/runs/a/run.h5"}, key="k")
    ((name, request),) = sent
    assert name == "cluster" and request.request_id == identity
    with qtbot.waitSignal(bridge.replied) as got:
        bridge._machine_replied(Reply("describe_output", identity, 0, value={}))
    replies: Any = got.args
    assert replies[0].request_id == identity
    bridge.close()


def test_the_runs_window_lists_and_opens_this_computer_s_runs(
    qtbot: QtBot, tmp_path: Path, run: str
):
    workspace = tmp_path / "runs"
    folder = workspace / "2026-11-07-two-modes"
    folder.mkdir(parents=True)
    shutil.copy(run, folder / "run.h5")
    (folder / "recipe.toml").write_text('[recipe]\ntitle = "Two modes"\n', "utf-8")
    hub = MachineHub()
    window = RunsWindow(
        lambda: [Machine(LOCAL)],
        hub,
        lambda: workspace,
        lambda machine, summary: tmp_path / "copies" / "run.h5",
    )
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.table.rowCount() == 1, timeout=WAIT_MS)
    title: Any = window.table.item(0, 1)
    assert title.text() == "Two modes"
    window.table.selectRow(0)
    assert window.buttons["Open"].isEnabled() and window.buttons["Remove"].isEnabled()
    assert not window.buttons["Stop"].isEnabled()
    assert not window.buttons["Fetch a copy"].isEnabled()  # this computer's own
    with qtbot.waitSignal(window.open_wanted) as wanted:
        window._open()
    opened: Any = wanted.args
    machine, summary = opened
    assert machine.local and summary["output"].endswith("run.h5")
    window._machine_state("cluster", "down", "no route")
    assert "cluster is down: no route" in window.note.text()
    shot(window, tmp_path, "runs-window")
    window.hide()
    hub.close()


def test_a_run_is_started_and_followed_through_a_machine(qtbot: QtBot, tmp_path: Path):
    from conftest import TINY  # noqa: PLC0415 - only here

    bridge = ServiceBridge()
    events: list[object] = []
    bridge.run_event.connect(events.append)
    spec = RunSpec("r1", str(tmp_path / "r1"), recipe=TINY)
    handle = bridge.start_on(Machine(LOCAL), spec)
    qtbot.waitUntil(handle.ended.is_set, timeout=WAIT_MS)
    # The ending is flagged on the connection's thread; its event reaches this one
    # through Qt's queue.
    qtbot.waitUntil(
        lambda: any(getattr(e, "kind", "") == "finished" for e in events),
        timeout=WAIT_MS,
    )
    kinds = [getattr(event, "kind", "") for event in events]
    assert kinds.count("row") == 6 and kinds[-1] == "finished"
    again = bridge.attach_on(Machine(LOCAL), "r1", str(tmp_path / "r1"))
    qtbot.waitUntil(again.ended.is_set, timeout=WAIT_MS)
    handle.subscribe(3)  # a record drops nothing: no replay needed
    bridge.close()


def test_the_setup_dialog_shows_its_script_before_running_it(
    qtbot: QtBot, tmp_path: Path
):
    from solprop_gui.ui.setup_dialog import SetupDialog  # noqa: PLC0415 - only here

    dialog = SetupDialog(SLURM, "0.2.1.dev3+gabc")
    qtbot.addWidget(dialog)
    assert "development build" in dialog.note.text()
    assert not dialog.run_button.isEnabled()  # PyPI has no such build
    assert '"solprop==0.2.1.dev3+gabc"' in dialog.script.toPlainText()
    dialog.root.setText("/scratch/envs")
    dialog.add_wheels([tmp_path / "solprop-0.2.1-py3-none-any.whl"])
    assert "--find-links /scratch/envs/wheels" in dialog.script.toPlainText()
    assert "copied first" in dialog.wheel_label.text()
    assert dialog.run_button.isEnabled()
    dialog._finished(None, "no route to host")
    assert dialog.python is None and "Not set up: no route" in dialog.log.toPlainText()
    dialog._finished("/scratch/envs/x/bin/python", None)
    assert dialog.python == "/scratch/envs/x/bin/python"
    released = SetupDialog(SLURM, "0.3.0")
    qtbot.addWidget(released)
    assert released.note.text() == "" and released.run_button.isEnabled()


def test_a_copy_is_fetched_in_pieces_and_resumed(
    qtbot: QtBot, tmp_path: Path, monkeypatch
):
    from solprop_gui.ui import machines_hub  # noqa: PLC0415 - only here

    monkeypatch.setattr(machines_hub, "CHUNK", 1000)
    source = tmp_path / "remote" / "run.h5"
    source.parent.mkdir()
    source.write_bytes(bytes(range(256)) * 20)  # 5120 bytes: six pieces
    target = tmp_path / "copies" / "run.h5"
    target.parent.mkdir()
    target.with_name("run.h5.part").write_bytes(source.read_bytes()[:1500])
    hub = MachineHub()
    hub.ensure(Machine(LOCAL))
    seen: list[tuple[int, int]] = []
    done: list[tuple[object, object]] = []
    hub.fetch(
        LOCAL,
        str(source),
        target,
        lambda have, total: seen.append((have, total)),
        lambda value, error: done.append((value, error)),
    )
    qtbot.waitUntil(lambda: bool(done), timeout=WAIT_MS)
    assert done == [(str(target), None)]
    assert target.read_bytes() == source.read_bytes()
    assert seen[0] == (2500, 5120) and seen[-1] == (5120, 5120)  # from the partial
    missing: list[tuple[object, object]] = []
    hub.fetch(
        LOCAL,
        str(tmp_path / "none.h5"),
        tmp_path / "none.h5",
        lambda have, total: None,
        lambda value, error: missing.append((value, error)),
    )
    qtbot.waitUntil(lambda: bool(missing), timeout=WAIT_MS)
    assert "is not on local" in str(missing[0][1])
    hub.close()


class FakeRun:
    """A run's handle that remembers how it was asked to end."""

    def __init__(self) -> None:
        self.ended: list[bool] = []

    def end_now(self, *, kill: bool = False) -> None:
        self.ended.append(kill)


def test_closing_over_a_run_asks_whether_to_keep_stop_or_kill_it(
    qtbot: QtBot, tmp_path: Path
):
    from solprop.service.states import ExecutionState  # noqa: PLC0415 - only here

    from solprop_gui.ui.workflow_window import WorkflowWindow  # noqa: PLC0415

    for answer, expected, closes in (
        ("cancel", [], False),
        ("keep", [], True),
        ("stop", [False], True),
        ("kill", [True], True),
    ):
        window = WorkflowWindow(workspace=tmp_path / "runs")
        qtbot.addWidget(window)
        asked: list[str] = []

        def confirm(
            title: str, text: str, answer: str = answer, asked: list[str] = asked
        ) -> str:
            asked.append(title)
            return answer if title.startswith("Closing") else "discard"

        window.confirm = confirm
        run = FakeRun()
        window.run_panel.run_id = "r1"
        window.run_panel.state = ExecutionState.RUNNING
        window.run_panel.client = run  # type: ignore[assignment]
        window.show()
        assert window.close() is closes
        assert asked[0] == "Closing with a run going" and run.ended == expected
        if not closes:
            window.run_panel.state = ExecutionState.COMPLETE
            window.close()


def test_runs_can_be_tied_to_the_window(qtbot: QtBot, tmp_path: Path):
    import os  # noqa: PLC0415 - only here

    from solprop_gui.ui.workflow_window import WorkflowWindow  # noqa: PLC0415

    window = WorkflowWindow(workspace=tmp_path / "runs")
    qtbot.addWidget(window)
    window.confirm = lambda _title, _text: "discard"
    assert not window.run_panel.tie_to_window and window.run_panel._owner() is None
    window._tie_runs(True)
    owner = window.run_panel._owner()
    assert owner is not None and owner[0] == os.getpid()
    assert app_settings().value("tie-runs") == "true"
    window.run_panel.run_machine = SLURM  # a run elsewhere cannot see this window
    assert window.run_panel._owner() is None
    window.run_panel.run_machine = None
    window._tie_runs(False)
    window.close()
