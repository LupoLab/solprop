# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run outlives its window with the service (option C), and a new window takes it."""

import threading
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from solprop_gui.model.recipe import RecipeModel
from solprop_gui.service import server
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.main_window import MainWindow
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

    first = MainWindow(bridge=ServiceBridge(service=True), workspace=tmp_path / "ws")
    qtbot.addWidget(first)
    first.confirm = lambda _title, _text: "discard"
    first.set_model(RecipeModel(SLOW), base=None)
    qtbot.waitUntil(lambda: first._built_route == "capillary", timeout=WAIT_MS)
    qtbot.waitUntil(lambda: not first.issues, timeout=WAIT_MS)
    first.run_panel.start()
    qtbot.waitUntil(lambda: first.run_panel._last_index >= 0, timeout=WAIT_MS)
    run_id = first.run_panel.run_id
    first.close()  # no question: the run goes on in the service

    second = MainWindow(bridge=ServiceBridge(service=True), workspace=tmp_path / "ws")
    qtbot.addWidget(second)
    second.confirm = lambda _title, _text: "discard"
    qtbot.waitUntil(lambda: second.run_panel.run_id == run_id, timeout=WAIT_MS)
    assert "re-attached" in second.run_panel.state_label.text()
    qtbot.waitUntil(lambda: second.run_panel._last_index >= 0, timeout=WAIT_MS)
    second.run_panel.stop()
    qtbot.waitUntil(lambda: not second.run_panel.active, timeout=WAIT_MS)
    second._runs_listed([])  # nothing going: says so
    assert "No run is going" in second.statusBar().currentMessage()
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
    window = MainWindow(bridge=bridge, workspace=tmp_path / "ws")
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: bool(failed), timeout=WAIT_MS)
    assert not bridge.service and "did not start" in failed[0]
    qtbot.waitUntil(lambda: window.start.examples.count() > 0, timeout=WAIT_MS)
    window.close()
