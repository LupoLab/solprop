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
    bridge = ServiceBridge(service=False)
    bridge.know_folder("cluster", "/runs/a")
    assert bridge.machine_of({"path": "/runs/a/run.h5"}) == "cluster"
    assert bridge.machine_of({"path": "/runs/ab/run.h5"}) is None
    assert bridge.machine_of({"base": "/runs/a"}) == "cluster"
    assert bridge.machine_of({"machine": "cluster"}) is None  # not connected
    assert bridge.machine_of(None) is None and bridge.machine_of({"path": 3}) is None
    sent: list[object] = []
    bridge.hub.kernel = lambda name, request: sent.append((name, request))  # type: ignore[method-assign]
    identity = bridge.request("describe_output", {"path": "/runs/a/run.h5"}, key="k")
    ((name, request),) = sent
    assert name == "cluster" and request.request_id == identity
    with qtbot.waitSignal(bridge.replied) as got:
        bridge._machine_replied(Reply("describe_output", identity, 0, value={}))
    assert got.args[0].request_id == identity
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
    assert window.table.item(0, 1).text() == "Two modes"
    window.table.selectRow(0)
    assert window.buttons["Open"].isEnabled() and window.buttons["Remove"].isEnabled()
    assert not window.buttons["Stop"].isEnabled()
    assert not window.buttons["Fetch a copy"].isEnabled()  # this computer's own
    with qtbot.waitSignal(window.open_wanted) as wanted:
        window._open()
    machine, summary = wanted.args
    assert machine.local and summary["output"].endswith("run.h5")
    window._machine_state("cluster", "down", "no route")
    assert "cluster is down: no route" in window.note.text()
    shot(window, tmp_path, "runs-window")
    window.hide()
    hub.close()


def test_a_run_is_started_and_followed_through_a_machine(qtbot: QtBot, tmp_path: Path):
    from conftest import TINY  # noqa: PLC0415 - only here

    bridge = ServiceBridge(service=False)
    events: list[object] = []
    bridge.run_event.connect(events.append)
    spec = RunSpec("r1", str(tmp_path / "r1"), recipe=TINY)
    handle = bridge.start_on(Machine(LOCAL), spec)
    qtbot.waitUntil(handle.ended.is_set, timeout=WAIT_MS)
    kinds = [getattr(event, "kind", "") for event in events]
    assert kinds.count("row") == 6 and kinds[-1] == "finished"
    again = bridge.attach_on(Machine(LOCAL), "r1", str(tmp_path / "r1"))
    qtbot.waitUntil(again.ended.is_set, timeout=WAIT_MS)
    handle.subscribe(3)  # a record drops nothing: no replay needed
    bridge.close()
