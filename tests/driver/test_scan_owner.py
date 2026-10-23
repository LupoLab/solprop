# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify local-owner identity checks without relying on a running scheduler."""

import socket
import subprocess
import sys

import pytest

from solprop.driver.executors import _queue_state as state
from solprop.exceptions import DriverError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the Queue executor is POSIX only and refuses Windows",
)


def test_queue_record_requires_a_json_object(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("[]")
    with pytest.raises(DriverError, match="JSON object"):
        state.read_json(path)


def test_linux_process_stamp_handles_parentheses_in_process_name(tmp_path, monkeypatch):
    path = tmp_path / "stat"
    path.write_text(
        "123 (command (with) brackets) " + " ".join(str(i) for i in range(30))
    )
    monkeypatch.setattr(state, "Path", lambda name: path)
    assert state.process_stamp(123) == "19"


@pytest.mark.parametrize(
    "printed,stamp",
    [("Tue Sep 23 13:44:28 2026\n", "Tue Sep 23 13:44:28 2026"), ("", None)],
)
def test_process_stamp_asks_ps_where_there_is_no_proc(
    tmp_path, monkeypatch, printed, stamp
):
    """Without `/proc` (macOS) the stamp is `ps`'s start time, or None if it has none.

    `/proc` is taken away on every platform, so Linux exercises the branch only
    macOS takes in production.
    """
    monkeypatch.setattr(state, "Path", lambda name: tmp_path / "absent")
    calls = []

    def ps(arguments, **options):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout=printed, stderr="")

    monkeypatch.setattr(state.subprocess, "run", ps)
    assert state.process_stamp(123) == stamp
    assert calls == [["ps", "-p", "123", "-o", "lstart="]]


def test_owner_with_invalid_pid_can_be_reclaimed():
    assert not state.live_owner(
        {"owner": {"host": socket.gethostname(), "pid": 0}}, 60.0
    )


def test_permission_denied_does_not_license_owner_takeover(monkeypatch):
    def denied(*args):
        raise PermissionError("other account")

    monkeypatch.setattr(state.os, "kill", denied)
    assert state.live_owner({"owner": {"host": socket.gethostname(), "pid": 123}}, 60.0)


@pytest.mark.parametrize(
    "previous,current,live",
    [("before", "after", False), (None, "now", True), ("before", None, True)],
)
def test_owner_process_creation_identity_prevents_pid_reuse(
    monkeypatch, previous, current, live
):
    monkeypatch.setattr(state.os, "kill", lambda *args: None)
    monkeypatch.setattr(state, "process_stamp", lambda pid: current)
    row = {
        "owner": {"host": socket.gethostname(), "pid": 123, "process_start": previous},
        "updated": 0,
    }
    assert state.live_owner(row, 60.0) is live
