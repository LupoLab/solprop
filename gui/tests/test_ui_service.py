# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run outlives its window with the service (option C), and a new window takes it."""

import threading
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from solprop_gui.classes.workflow import Stage
from solprop_gui.model.recipe import RecipeModel
from solprop_gui.service import server
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.workflow_window import WorkflowWindow
from test_server import SLOW, wait_for
from test_ui import WAIT_MS, private_settings  # noqa: F401 - a fixture


def test_a_new_window_re_attaches_to_the_run_the_last_one_left(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    folder = tmp_path / "state"
    monkeypatch.setenv("SOLPROP_GUI_STATE", str(folder))
    monkeypatch.setattr(server, "IDLE_SECONDS", 1.0)
    thread = threading.Thread(target=server.main, kwargs={"poll": 0.05}, daemon=True)
    thread.start()
    wait_for((folder / "service.json").exists)

    first = WorkflowWindow(
        bridge=ServiceBridge(service=True), workspace=tmp_path / "ws"
    )
    qtbot.addWidget(first)
    first.document.confirm = lambda _title, _text: "discard"
    first.document.set_model(RecipeModel(SLOW), base=None)
    qtbot.waitUntil(lambda: first.document.can_run, timeout=WAIT_MS)
    first.primary.click()
    qtbot.waitUntil(lambda: first.run_panel._last_index >= 0, timeout=WAIT_MS)
    run_id = first.run_panel.run_id
    first.close()  # no question: the run goes on in the service

    second = WorkflowWindow(
        bridge=ServiceBridge(service=True), workspace=tmp_path / "ws"
    )
    qtbot.addWidget(second)
    second.document.confirm = lambda _title, _text: "discard"
    qtbot.waitUntil(lambda: second.run_panel.run_id == run_id, timeout=WAIT_MS)
    assert second.stage is Stage.RUN
    assert "Re-attached" in second.statusBar().currentMessage()
    qtbot.waitUntil(lambda: second.run_panel._last_index >= 0, timeout=WAIT_MS)
    second.run_panel.stop()
    qtbot.waitUntil(lambda: not second.run_panel.active, timeout=WAIT_MS)
    second._runs_listed([])  # nothing going: nothing taken over
    assert second.stage is Stage.ANALYSE
    second.close()
    thread.join(60)


def test_a_service_out_of_reach_falls_back_to_the_window_s_own(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from solprop_gui.service import remote  # noqa: PLC0415 - only here

    monkeypatch.setenv("SOLPROP_GUI_STATE", str(tmp_path / "none"))
    monkeypatch.setattr(remote, "start_service", lambda _state: None)
    monkeypatch.setattr(remote, "START_SECONDS", 0.2)
    bridge = ServiceBridge(service=True)
    failed: list[str] = []
    bridge.service_failed.connect(failed.append)
    window = WorkflowWindow(bridge=bridge, workspace=tmp_path / "ws")
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: bool(failed), timeout=WAIT_MS)
    assert not bridge.service and "did not start" in failed[0]
    assert "not available" in window.statusBar().currentMessage()
    # The window's own kernel answers instead.
    qtbot.waitUntil(lambda: bool(window.sheet._examples), timeout=WAIT_MS)
    window.close()
