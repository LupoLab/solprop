# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run outlives its window with the service (option C), and a new window takes it."""

import threading
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot
from solprop.service import server

from conftest import SLOW, wait_for
from solprop_gui.classes.workflow import Stage
from solprop_gui.model.recipe import RecipeModel
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.workflow_window import WorkflowWindow
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
    from solprop.service import remote  # noqa: PLC0415 - only here

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


def test_a_request_between_a_failure_and_its_news_is_not_lost(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The connection fails on its own thread and its news is queued to the window's.
    # A request sent before the news arrives once opened another service connection,
    # which the fallback then closed with the request held in it.
    from solprop.service import remote  # noqa: PLC0415 - only here
    from solprop.service.client import KernelClient  # noqa: PLC0415 - only here

    gate = threading.Event()

    def refuse(_self: remote.ServiceConnection) -> object:
        gate.wait(WAIT_MS / 1000)  # refused once the first request is held
        msg = "the solprop service did not start"
        raise ConnectionError(msg)

    monkeypatch.setattr(remote.ServiceConnection, "_connect", refuse)
    bridge = ServiceBridge(service=True)
    failed: list[str] = []
    replies: list[int] = []
    bridge.service_failed.connect(failed.append)
    bridge.replied.connect(lambda reply: replies.append(reply.request_id))
    held = bridge.request("examples", key="held")
    connection = bridge._kernel
    assert isinstance(connection, remote.ServiceConnection)
    gate.set()
    connection._reader.join(WAIT_MS / 1000)  # failed; the news not yet delivered
    assert connection.failed and failed == []
    later = bridge.request("examples", key="later")
    assert isinstance(bridge._kernel, KernelClient)  # not the service again
    qtbot.waitUntil(lambda: {held, later} <= set(replies), timeout=WAIT_MS)
    assert len(failed) == 1 and "did not start" in failed[0]
    bridge.close()


def test_a_first_request_that_finds_the_connection_failed_is_not_lost(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The reader thread starts in the connection's constructor, and may fail before the
    # window's first request reaches it (seen on macOS): that request is given back.
    from solprop.service import remote  # noqa: PLC0415 - only here

    monkeypatch.setenv("SOLPROP_GUI_STATE", str(tmp_path / "none"))
    monkeypatch.setattr(remote, "start_service", lambda _state: None)
    monkeypatch.setattr(remote, "START_SECONDS", 0.0)  # refused at once
    send = remote.ServiceConnection.request

    def after_the_failure(self: remote.ServiceConnection, *args, **kwargs) -> int:
        self._reader.join(WAIT_MS / 1000)
        return send(self, *args, **kwargs)

    monkeypatch.setattr(remote.ServiceConnection, "request", after_the_failure)
    bridge = ServiceBridge(service=True)
    failed: list[str] = []
    replies: list[int] = []
    bridge.service_failed.connect(failed.append)
    bridge.replied.connect(lambda reply: replies.append(reply.request_id))
    first = bridge.request("examples", key="first")
    qtbot.waitUntil(lambda: first in replies, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: bool(failed), timeout=WAIT_MS)
    assert len(failed) == 1 and "did not start" in failed[0]
    assert not bridge.service
    bridge.close()
