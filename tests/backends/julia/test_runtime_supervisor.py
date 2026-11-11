# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the external termination deadline strict before and after child entry."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backends.julia import test_runtime_subprocess as runtime_tests

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process supervision")


class Clock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()


class Process:
    pid = 12345

    def __init__(self, clock):
        self.clock = clock
        self.returncode = None
        self.waits = []
        self.outcomes = [(0.1, -signal.SIGTERM)]

    def poll(self):
        return self.returncode

    def wait(self, *, timeout):
        self.waits.append(timeout)
        elapsed, code = self.outcomes.pop(0)
        self.clock.now += min(elapsed, timeout)
        if elapsed > timeout:
            if code == "race":
                self.returncode = -signal.SIGTERM
            raise subprocess.TimeoutExpired("owned probe", timeout)
        self.returncode = code
        return code


def _ready(output, process, **changes):
    result = {
        "case": "sigterm",
        "pid": process.pid,
        "requested_threads": 2,
        "actual_threads": 2,
        "initial_handler": "ignored",
        "phase": "work",
        "stage": "ready-for-accepted-termination",
        **changes,
    }
    output.write_text(json.dumps(result), encoding="utf-8")
    output.with_suffix(".term-entered").write_text("entered", encoding="utf-8")


@pytest.fixture
def supervision(tmp_path, monkeypatch):
    clock = Clock()
    process = Process(clock)
    output = tmp_path / "probe.json"
    signals = []

    def send(pid, number):
        assert pid == process.pid
        assert output.with_suffix(".term-entered").read_text(encoding="utf-8") == (
            "entered"
        )
        signals.append((number, clock.now))

    monkeypatch.setattr(runtime_tests.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime_tests.time, "sleep", clock.sleep)
    monkeypatch.setattr(runtime_tests.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(runtime_tests.os, "killpg", send)
    return SimpleNamespace(clock=clock, process=process, output=output, signals=signals)


def _run(fixture, **options):
    return runtime_tests._supervise_sigterm(
        fixture.process, fixture.output, "ignored:work", 2, **options
    )["supervision"]


@pytest.mark.parametrize("code", [-signal.SIGTERM, 128 + signal.SIGTERM])
def test_prompt_term_exit_records_actual_status_without_escalation(supervision, code):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(0.1, code)]
    result = _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0)]
    assert fixture.process.waits == [2.0]
    assert result["returncode"] == code and result["reaped"]
    assert (
        result["forced_escalation"] is False and result["term_grace_expired"] is False
    )
    assert result["kill_seconds"] is None
    assert result["exit_signal"] == (signal.SIGTERM if code < 0 else None)
    assert result["exit_status"] == (code if code >= 0 else None)
    assert result["elapsed_seconds"] == 0.1
    assert (
        json.loads(fixture.output.read_text(encoding="utf-8"))["supervision"] == result
    )


@pytest.mark.parametrize("code", [-signal.SIGKILL, -signal.SIGTERM])
def test_expired_term_grace_records_own_kill_and_reaps(supervision, code):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(3.0, None), (0.2, code)]
    result = _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0), (signal.SIGKILL, 2.0)]
    assert fixture.process.waits == [2.0, 15.0]
    assert result["returncode"] == code and result["reaped"]
    assert result["forced_escalation"] and result["term_grace_expired"]
    assert result["kill_seconds"] == 2.0 and result["elapsed_seconds"] == 2.2
    assert result["exit_signal"] == -code and result["exit_status"] is None


def test_supervisor_waits_for_actual_entry_before_term(supervision):
    fixture = supervision

    def enter():
        if fixture.clock.now >= 0.01:
            _ready(fixture.output, fixture.process)

    fixture.clock.on_sleep = enter
    result = _run(fixture)
    assert result["entered_seconds"] >= 0.01
    assert result["term_seconds"] >= result["entered_seconds"]
    assert fixture.signals == [(signal.SIGTERM, result["term_seconds"])]


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": 54321},
        {"case": "wrong"},
        {"stage": "not-entered"},
        {"requested_threads": 1},
        {"actual_threads": 1},
        {"initial_handler": "default"},
        {"phase": "idle"},
    ],
)
def test_mismatched_ready_identity_never_authorizes_term(supervision, changes):
    fixture = supervision
    _ready(fixture.output, fixture.process, **changes)
    with pytest.raises(AssertionError):
        _run(fixture)
    assert fixture.signals == [] and fixture.process.waits == []


@pytest.mark.parametrize("marker", ["", "not entered"])
def test_incomplete_entered_marker_never_authorizes_term(supervision, marker):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.output.with_suffix(".term-entered").write_text(marker, encoding="utf-8")
    with pytest.raises(AssertionError):
        _run(fixture)
    assert fixture.signals == []


def test_timeout_before_entry_is_failure_without_term(supervision):
    fixture = supervision
    with pytest.raises(AssertionError, match="never verified entry"):
        _run(fixture, ready_timeout=0.01)
    assert fixture.clock.now >= 0.01 and fixture.signals == []
    assert fixture.process.waits == []


@pytest.mark.parametrize("code", [0, 1, -signal.SIGKILL])
def test_exit_before_entry_is_failure_without_term(supervision, code):
    fixture = supervision
    fixture.process.returncode = code
    _ready(fixture.output, fixture.process)
    with pytest.raises(AssertionError, match="exited before verified entry"):
        _run(fixture)
    assert fixture.signals == []


def test_unowned_process_group_is_never_signalled(supervision, monkeypatch):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    monkeypatch.setattr(runtime_tests.os, "getpgid", lambda _: 98765)
    with pytest.raises(AssertionError, match="private group"):
        _run(fixture)
    assert fixture.signals == []


@pytest.mark.parametrize("code", [0, 1, -signal.SIGSEGV, -signal.SIGKILL, 137])
def test_unexpected_exit_is_not_accepted_as_supervised_termination(supervision, code):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(0.1, code)]
    with pytest.raises(AssertionError, match="unexpected supervised SIGTERM exit"):
        _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0)]
    report = json.loads(fixture.output.read_text(encoding="utf-8"))["supervision"]
    assert report["returncode"] == code and report["forced_escalation"] is False


@pytest.mark.parametrize("code", [0, 1, -signal.SIGSEGV, 137])
def test_deadline_escalation_still_refuses_an_unrelated_exit(supervision, code):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(3.0, None), (0.1, code)]
    with pytest.raises(AssertionError, match="unexpected supervised SIGTERM exit"):
        _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0), (signal.SIGKILL, 2.0)]
    report = json.loads(fixture.output.read_text(encoding="utf-8"))["supervision"]
    assert report["returncode"] == code and report["forced_escalation"]


def test_entered_marker_without_ready_record_does_not_authorize_term(supervision):
    fixture = supervision
    fixture.output.with_suffix(".term-entered").write_text("entered", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _run(fixture)
    assert fixture.signals == []


def test_stale_readiness_is_refused_before_spawning(supervision, monkeypatch, tmp_path):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    called = []
    monkeypatch.setattr(
        runtime_tests.subprocess, "Popen", lambda *_, **__: called.append(1)
    )
    with pytest.raises(AssertionError):
        runtime_tests._run_terminating(
            None, tmp_path, [], fixture.output, "ignored:work", 2
        )
    assert called == [] and fixture.signals == []


def test_term_completion_race_does_not_claim_unsent_kill(supervision):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(3.0, "race"), (0.0, -signal.SIGTERM)]
    result = _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0)]
    assert result["term_grace_expired"] and not result["forced_escalation"]


def test_term_completion_during_kill_does_not_claim_unsent_kill(
    supervision, monkeypatch
):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(3.0, None), (0.0, -signal.SIGTERM)]
    original = runtime_tests.os.killpg

    def send(pid, number):
        if number == signal.SIGKILL:
            raise ProcessLookupError
        original(pid, number)

    monkeypatch.setattr(runtime_tests.os, "killpg", send)
    result = _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0)]
    assert result["term_grace_expired"] and not result["forced_escalation"]


def test_unreaped_killed_child_remains_a_failure(supervision):
    fixture = supervision
    _ready(fixture.output, fixture.process)
    fixture.process.outcomes = [(3.0, None), (16.0, None)]
    with pytest.raises(subprocess.TimeoutExpired):
        _run(fixture)
    assert fixture.signals == [(signal.SIGTERM, 0.0), (signal.SIGKILL, 2.0)]
    assert fixture.process.returncode is None
    assert "supervision" not in json.loads(fixture.output.read_text(encoding="utf-8"))


def test_failed_readiness_cleans_up_child_without_becoming_success(
    supervision, monkeypatch, tmp_path
):
    fixture = supervision
    fixture.process.outcomes = [(0.1, -signal.SIGKILL)]
    failure = AssertionError("missing actual Julia entry")

    def refuse(*_):
        raise failure

    monkeypatch.setattr(runtime_tests, "_environment", lambda *_: {})
    monkeypatch.setattr(
        runtime_tests.subprocess, "Popen", lambda *_, **__: fixture.process
    )
    monkeypatch.setattr(runtime_tests, "_supervise_sigterm", refuse)
    monkeypatch.setattr(
        runtime_tests.os,
        "killpg",
        lambda pid, number: fixture.signals.append((pid, number)),
    )
    with pytest.raises(AssertionError, match="missing actual Julia entry") as caught:
        runtime_tests._run_terminating(
            None, tmp_path, [], fixture.output, "ignored:work", 2
        )
    assert caught.value is failure
    assert fixture.signals == [(fixture.process.pid, signal.SIGKILL)]
    assert fixture.process.returncode == -signal.SIGKILL and fixture.process.waits == [
        15
    ]
    assert not fixture.output.exists()


_CHILD = """\
import json, os, signal, sys
from pathlib import Path
output = Path(sys.argv[1])
ignored = sys.argv[2] == 'ignored'
signal.signal(signal.SIGTERM, signal.SIG_IGN if ignored else signal.SIG_DFL)
output.write_text(json.dumps(dict(case='sigterm', pid=os.getpid(), requested_threads=2,
    actual_threads=2, initial_handler=sys.argv[2], phase='idle',
    stage='ready-for-accepted-termination')), encoding='utf-8')
pending = output.with_suffix('.pending')
pending.write_text('entered', encoding='utf-8')
pending.replace(output.with_suffix('.term-entered'))
while True:
    signal.pause()
"""


@pytest.mark.parametrize("mode", ["default", "ignored"])
def test_real_posix_supervisor_distinguishes_prompt_and_forced_exit(tmp_path, mode):
    output = tmp_path / "real.json"
    process = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(output), mode],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        result = runtime_tests._supervise_sigterm(
            process, output, f"{mode}:idle", 2, ready_timeout=10, grace=0.05
        )["supervision"]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    expected = signal.SIGKILL if mode == "ignored" else signal.SIGTERM
    assert process.returncode == -expected and result["reaped"]
    assert result["forced_escalation"] == (mode == "ignored")
    assert (
        result["entered_seconds"] <= result["term_seconds"] <= result["elapsed_seconds"]
    )
    if result["forced_escalation"]:
        assert result["kill_seconds"] >= result["term_seconds"] + 0.05
