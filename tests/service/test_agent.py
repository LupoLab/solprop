# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The helper on a machine and a window's connection to it.

The helper runs as it does over SSH, a process speaking frames on its standard input
and output, here on this computer (``Machine("local")``), so the whole path short of
``ssh`` itself is exercised: hello, requests, a real run followed to its end, files,
the hosted kernel, a dropped session and its reconnection, and refusals.
"""

import io
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from service.cases import TINY
from solprop.service import agent, machines
from solprop.service.agent import Agent, hello, serve
from solprop.service.frames import encode, split_frames
from solprop.service.machines import (
    LOCAL,
    Connection,
    Machine,
    agent_command,
    ssh_hosts,
)
from solprop.service.messages import ServiceError
from solprop.service.worker import RunSpec, spec_value

WAIT = 120.0


class Window:
    """What a window gets from a connection: its states and its unasked frames."""

    def __init__(self) -> None:
        self.states: list[tuple[str, object]] = []
        self.frames: list[dict[str, Any]] = []
        self.changed = threading.Condition()

    def on_state(self, state: str, detail: object) -> None:
        with self.changed:
            self.states.append((state, detail))
            self.changed.notify_all()

    def on_frame(self, frame: dict[str, Any]) -> None:
        with self.changed:
            self.frames.append(frame)
            self.changed.notify_all()

    def wait(self, check, timeout: float = WAIT) -> None:
        with self.changed:
            assert self.changed.wait_for(check, timeout), "timed out"

    def kinds(self) -> list[str]:
        return [state for state, _ in self.states]


@pytest.fixture
def window() -> Window:
    return Window()


@pytest.fixture
def connection(window: Window):
    connection = Connection(Machine(LOCAL), window.on_frame, window.on_state)
    connection.open()
    window.wait(lambda: "up" in window.kinds())
    yield connection
    connection.close()


def test_the_helper_says_who_it_is(connection: Connection):
    assert connection.hello is not None
    assert connection.hello["python"] == sys.executable
    assert connection.hello["slurm"] in {True, False}


def test_a_run_is_started_followed_to_its_end_and_removed(
    connection: Connection, window: Window, tmp_path: Path
):
    workspace = tmp_path / "runs"
    assert connection.call("list_runs", {"workspace": str(workspace)}) == []
    spec = RunSpec("r1", str(workspace / "r1"), recipe=TINY)
    summary = connection.call("start", {"spec": spec_value(spec), "launcher": "direct"})
    assert summary["run_id"] == "r1" and summary["state"] in {"starting", "running"}
    assert (
        connection.call("follow", {"folder": str(workspace / "r1"), "offset": 0}) == 0
    )

    def ended() -> bool:
        events = [
            e for f in window.frames if f["kind"] == "events" for e in f["events"]
        ]
        return any(event["kind"] == "finished" for event in events)

    window.wait(ended)
    rows = [
        event
        for frame in window.frames
        if frame["kind"] == "events"
        for event in frame["events"]
        if event["kind"] == "row"
    ]
    assert len(rows) == 6 and isinstance(
        rows[0]["data"]["density"]["wavelength"], np.ndarray
    )
    listed = connection.call("list_runs", {"workspace": str(workspace)})
    assert [(run["run_id"], run["state"]) for run in listed] == [("r1", "finished")]
    connection.call("unfollow", {"folder": str(workspace / "r1")})
    connection.call("remove", {"folder": str(workspace / "r1")})
    assert not (workspace / "r1").exists()


def test_files_go_both_ways_in_pieces(connection: Connection, tmp_path: Path):
    path = tmp_path / "data" / "spectrum.txt"
    data = np.frombuffer(b"0 1\n1 2\n", dtype=np.uint8)
    assert connection.call("put", {"path": str(path), "data": data}) == 8
    assert (
        connection.call("put", {"path": str(path), "data": data, "append": True}) == 16
    )
    assert connection.call("stat", {"path": str(path)})["size"] == 16
    assert connection.call("stat", {"path": str(tmp_path / "none")}) is None
    piece = connection.call("fetch", {"path": str(path), "offset": 4, "size": 6})
    assert piece.tobytes() == b"1 2\n0 "


def test_the_hosted_kernel_answers_as_the_window_s_does(
    connection: Connection, window: Window
):
    request = {"kind": "examples", "payload": {}, "request_id": 7, "revision": 0}
    assert connection.call("kernel", {"request": request}) == 7
    window.wait(lambda: any(f["kind"] == "kernel_reply" for f in window.frames))
    reply = next(f["reply"] for f in window.frames if f["kind"] == "kernel_reply")
    assert reply["request_id"] == 7 and reply["error"] is None
    assert any(entry["name"] == "rdw" for entry in reply["value"])
    later = {"kind": "examples", "payload": {}, "request_id": 8, "revision": 3}
    assert connection.call("kernel", {"request": later}) == 8


def test_requests_that_fail_say_why(connection: Connection, tmp_path: Path):
    with pytest.raises(ServiceError, match="no request 'dance'"):
        connection.call("dance")
    with pytest.raises(ServiceError, match="no launcher 'teleport'"):
        spec = RunSpec("r2", str(tmp_path / "r2"), recipe=TINY)
        connection.call("start", {"spec": spec_value(spec), "launcher": "teleport"})
    with pytest.raises(ServiceError, match="no process or job"):
        connection.call("cancel", {"folder": str(tmp_path / "r2")})
    with pytest.raises(ServiceError, match="holds no run"):
        (tmp_path / "empty").mkdir()
        connection.call("remove", {"folder": str(tmp_path / "empty")})


def test_a_running_run_is_neither_removed_nor_lost_with_its_helper(
    window: Window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(machines, "RECONNECT_PAUSES", (0.05,))
    connection = Connection(Machine(LOCAL), window.on_frame, window.on_state)
    connection.open()
    window.wait(lambda: "up" in window.kinds())
    slow = TINY.replace('length = "5 cm"', 'length = "40 cm"').replace(
        "n_save = 6", "n_save = 40"
    )
    folder = tmp_path / "slow"
    spec = RunSpec("slow", str(folder), recipe=slow)
    connection.call("start", {"spec": spec_value(spec), "launcher": "direct"})
    deadline = time.monotonic() + WAIT
    while connection.call("list_runs", {"workspace": str(tmp_path)})[0]["state"] != (
        "running"
    ):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    with pytest.raises(ServiceError, match="is running"):
        connection.call("remove", {"folder": str(folder)})

    # The session drops (as a network does): the helper goes, the run does not.
    process = connection._process
    assert process is not None
    process.kill()
    window.wait(lambda: window.kinds().count("up") == 2)
    assert "down" in window.kinds()
    listed = connection.call("list_runs", {"workspace": str(tmp_path)})
    assert listed[0]["state"] in {"running", "finished"}
    connection.call("cancel", {"folder": str(folder)})
    deadline = time.monotonic() + WAIT
    while connection.call("list_runs", {"workspace": str(tmp_path)})[0]["state"] in {
        "running",
        "starting",
    }:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    connection.close()
    assert window.kinds()[-1] == "closed"


def test_another_solprop_or_none_is_refused_and_not_retried(window: Window):
    other = Connection(
        Machine(LOCAL),
        window.on_frame,
        window.on_state,
        version="0.0.0",
        command=fake("release"),
    )
    other.open()
    window.wait(lambda: "refused" in window.kinds())
    reason = dict(window.states)["refused"]
    assert "has solprop 0.1.0" in str(reason) and "install solprop 0.0.0" in str(reason)
    other.close()

    # A development build on either side: allowed, with a warning.
    dev = Window()
    near = Connection(
        Machine(LOCAL), dev.on_frame, dev.on_state, version="0.2.1.dev3+gabc"
    )
    near.open()
    dev.wait(lambda: "up" in dev.kinds())
    assert (
        near.hello is not None
        and "this window 0.2.1.dev3+gabc" in near.hello["warning"]
    )
    near.close()

    missing = Window()
    command = [sys.executable, "-c", "import sys; sys.exit('No module named solprop')"]
    none = Connection(
        Machine("lab", "me@lab", python="/opt/py"),
        missing.on_frame,
        missing.on_state,
        command=lambda machine: (command, dict(os.environ)),
    )
    none.open()
    missing.wait(lambda: "refused" in missing.kinds())
    assert "has no solprop at /opt/py" in str(dict(missing.states)["refused"])
    none.close()


def test_a_session_that_cannot_start_is_retried(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(machines, "RECONNECT_PAUSES", (0.01,))
    window = Window()
    broken = Connection(
        Machine("lab", "me@lab"),
        window.on_frame,
        window.on_state,
        command=lambda machine: (["/no/such/ssh"], dict(os.environ)),
    )
    broken.open()
    window.wait(lambda: window.kinds().count("down") >= 2)
    assert "could not start" in str(dict(window.states)["down"])
    with pytest.raises(ServiceError, match="is not connected"):
        broken.call("list_runs", {"workspace": "/tmp"}, timeout=0.2)
    broken.close()


def test_a_helper_that_says_something_else_is_refused():
    window = Window()
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(" + repr(encode({"kind": "hi"})) + ")",
    ]
    odd = Connection(
        Machine(LOCAL),
        window.on_frame,
        window.on_state,
        command=lambda machine: (command, dict(os.environ)),
    )
    odd.open()
    window.wait(lambda: "refused" in window.kinds())
    assert "did not say hello" in str(dict(window.states)["refused"])
    odd.close()


def test_the_helper_serves_a_stream_and_ignores_what_is_not_a_request():
    requests = io.BytesIO(
        encode([1])
        + encode({"id": 1, "kind": "stat", "body": {"path": "/no/such"}})
        + encode({"kind": "bye"})
    )
    output = io.BytesIO()
    serve(requests, output)
    time.sleep(0.2)  # the request's own thread
    frames: list[Any] = split_frames(output.getvalue())[0]
    assert frames[0]["kind"] == "hello"
    assert {"kind": "reply", "id": 1, "value": None} in frames
    ended = io.BytesIO()
    serve(io.BytesIO(b""), ended)
    first: Any = split_frames(ended.getvalue())[0][0]
    assert first["kind"] == "hello"


def test_sinfo_s_partitions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    if sys.platform == "win32":
        pytest.skip("the fake sinfo is a shell script")
    fake = tmp_path / "sinfo"
    fake.write_text(
        "#!/bin/sh\n"
        'echo "nodes*|up|7-00:00:00|64|1000000|(null)"\n'
        'echo "gpu|up|7-00:00:00|64|1000000|gpu:a40:2"\n'
        'echo "gpu|up|7-00:00:00|128|1000000|gpu:a40:2"\n'
        "echo garbled\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    found = Agent(io.BytesIO()).slurm_info({})
    assert [p["name"] for p in found] == ["nodes", "gpu"]
    assert found[0]["default"] and found[1]["gres"] == "gpu:a40:2"
    fake.write_text("#!/bin/sh\necho down >&2\nexit 1\n", encoding="utf-8")
    with pytest.raises(ServiceError, match="sinfo failed: down"):
        Agent(io.BytesIO()).slurm_info({})


def test_a_followed_record_that_is_not_one_is_said(tmp_path: Path):
    (tmp_path / "events.bin").write_bytes(encode([1]))
    output = io.BytesIO()
    helper = Agent(output)
    helper.follow({"folder": str(tmp_path), "offset": 0})
    deadline = time.monotonic() + 10
    while b"follow_failed" not in output.getvalue():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    helper.follow({"folder": str(tmp_path), "offset": 0})  # replaces the first
    helper.close()


def test_the_ssh_command_and_its_environment():
    command, environment = agent_command(
        Machine("lab", "me@lab", port=2222, python="~/env/bin/python"),
        askpass="/opt/askpass",
    )
    assert command[0] == "ssh" and command[command.index("-p") :][:2] == ["-p", "2222"]
    assert command[-1] == "exec ~/env/bin/python -m solprop.service.agent"
    assert environment["SSH_ASKPASS"] == "/opt/askpass"
    assert environment["SSH_ASKPASS_REQUIRE"] == "force"
    local, _ = agent_command(Machine(LOCAL))
    assert local == [sys.executable, "-m", "solprop.service.agent"]
    assert Machine(LOCAL).local and not Machine("lab", "me@lab").local
    assert not Machine("typed").local  # a machine whose host is still to come
    with pytest.raises(ServiceError, match="typed has no SSH host"):
        agent_command(Machine("typed"))
    machine = Machine("lab", "me@lab", resources={"cpus": 4})
    value = {**machine.value(), "future": 1}
    assert Machine.from_value(value) == machine


def test_hosts_are_read_from_the_ssh_configuration_and_its_includes(tmp_path: Path):
    (tmp_path / "lab.conf").write_text("Host lab-1 lab-2\n", encoding="utf-8")
    config = tmp_path / "config"
    config.write_text(
        "# hosts\n"
        "Host workstation\n"
        "  HostName 10.0.0.2\n"
        "Host=cluster\n"
        f"Include {tmp_path}/*.conf\n"
        f"Include {config}\n"
        "Host *.example !bad\n"
        "Host workstation\n",
        encoding="utf-8",
    )
    assert ssh_hosts(config) == ["workstation", "cluster", "lab-1", "lab-2"]
    assert ssh_hosts(tmp_path / "missing") == []


def test_hello_names_this_machine():
    greeting = hello()
    assert greeting["pid"] == os.getpid() and greeting["home"] == str(Path.home())
    assert agent.CHUNK >= 1 << 20


# ------------------------------------------------------- the helper, in-process


def answers(output: io.BytesIO) -> list[Any]:
    """Return the frames a helper wrote."""
    return split_frames(output.getvalue())[0]


def test_each_request_in_this_process(tmp_path: Path, monkeypatch):
    output = io.BytesIO()
    helper = Agent(output)
    workspace = tmp_path / "runs"
    assert helper.list_runs({"workspace": str(workspace)}) == []

    folder = workspace / "r1"
    spec = spec_value(RunSpec("r1", str(folder), recipe=TINY))
    summary = helper.start({"spec": spec, "launcher": "direct"})
    assert summary["state"] in {"starting", "running", "finished"}
    deadline = time.monotonic() + WAIT
    while helper.list_runs({"workspace": str(workspace)})[0]["state"] != "finished":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    helper.cancel({"folder": str(folder)})  # ended: its process has gone, a no-op
    helper.stop({"folder": str(folder)})
    assert (folder / "stop").exists()
    helper.follow({"folder": str(folder), "offset": 0})
    deadline = time.monotonic() + WAIT
    while b"finished" not in output.getvalue():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    helper.unfollow({"folder": str(folder)})
    helper.unfollow({"folder": str(folder)})  # twice: nothing to stop
    helper.remove({"folder": str(folder)})
    assert not folder.exists()
    helper.close()


def test_a_direct_run_is_cancelled_by_its_signal(tmp_path: Path):
    import subprocess  # noqa: PLC0415 - only here

    from solprop.service.record import Job, process_stamp, write_job  # noqa: PLC0415

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        job = Job("r1", "direct", socket.gethostname(), sleeper.pid, None, None, 0.0)
        write_job(tmp_path, job)
        Agent(io.BytesIO()).cancel({"folder": str(tmp_path)})
        assert sleeper.wait(timeout=WAIT) != 0
        stamped = Job(
            "r1",
            "direct",
            socket.gethostname(),
            sleeper.pid,
            process_stamp(os.getpid()),
            None,
            0.0,
        )
        write_job(tmp_path, stamped)  # its process has gone: nothing to send
        Agent(io.BytesIO()).cancel({"folder": str(tmp_path)})
    finally:
        sleeper.kill()
        sleeper.wait()


def test_slurm_runs_through_the_helper(tmp_path: Path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("the fake commands are shell scripts")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, script in (
        ("sbatch", 'echo "77"\n'),
        ("squeue", 'echo "77|PENDING|Resources"\n'),
        ("scancel", "exit 0\n"),
    ):
        (bin_ / name).write_text("#!/bin/sh\n" + script, encoding="utf-8")
        (bin_ / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    helper = Agent(io.BytesIO())
    folder = tmp_path / "runs" / "s1"
    spec = spec_value(RunSpec("s1", str(folder), recipe=TINY))
    summary = helper.start(
        {"spec": spec, "launcher": "slurm", "resources": {"cpus": 2}}
    )
    assert summary["slurm_job"] == "77" and summary["state"] == "queued"
    listed = helper.list_runs({"workspace": str(tmp_path / "runs")})
    assert listed[0]["detail"] == "Resources"
    helper.cancel({"folder": str(folder)})
    with pytest.raises(ServiceError, match="is queued"):
        helper.remove({"folder": str(folder)})
    with pytest.raises(ServiceError, match="no launcher"):
        helper.start({"spec": spec, "launcher": "nowhere"})


def test_the_hosted_kernel_in_this_process(monkeypatch):
    output = io.BytesIO()
    helper = Agent(output)
    request = {"kind": "examples", "payload": {}, "request_id": 3, "revision": 2}
    assert helper.kernel({"request": request}) == 3
    deadline = time.monotonic() + WAIT
    while not any(frame["kind"] == "kernel_reply" for frame in answers(output)):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    reply = next(f for f in answers(output) if f["kind"] == "kernel_reply")["reply"]
    assert reply["request_id"] == 3 and reply["error"] is None
    plain = {"kind": "examples", "payload": {}, "request_id": 4}
    assert helper.kernel({"request": plain}) == 4  # the same kernel, no revision
    kernel = helper._kernel
    helper._kernel_lost()
    assert helper._kernel is None
    assert answers(output)[-1] == {"kind": "kernel_lost"}
    helper._kernel = kernel
    helper.close()  # ends the kernel


def test_a_request_s_failure_is_answered_not_raised():
    output = io.BytesIO()
    helper = Agent(output)
    helper.handle({"id": 4, "kind": "nothing"})
    helper.handle({"id": 5, "kind": "stat", "body": {"path": "/no/such/file"}})
    assert answers(output) == [
        {"kind": "error", "id": 4, "message": "no request 'nothing'"},
        {"kind": "reply", "id": 5, "value": None},
    ]


# ------------------------------------------------------- a misbehaving helper


def fake(mode: str):
    """Return a command factory running the misbehaving helper in a mode."""
    script = str(Path(__file__).with_name("fake_helper.py"))
    return lambda machine: ([sys.executable, script, mode], dict(os.environ))


def connected(mode: str, window: Window) -> Connection:
    connection = Connection(
        Machine("lab", "me@lab"), window.on_frame, window.on_state, command=fake(mode)
    )
    connection.open()
    connection.open()  # once is enough: a second open does nothing
    return connection


def test_another_protocol_is_refused(window: Window):
    connection = connected("protocol", window)
    window.wait(lambda: "refused" in window.kinds())
    assert "speaks protocol" in str(dict(window.states)["refused"])
    connection.close()


def test_frames_that_are_not_for_a_request_go_to_the_window(window: Window):
    connection = connected("chatty", window)
    window.wait(lambda: "up" in window.kinds())
    assert connection.call("stat", {"path": "/"}) == "ok"
    window.wait(lambda: any(f.get("kind") == "note" for f in window.frames))
    assert [f["kind"] for f in window.frames] == ["note"]
    connection.close()


def test_a_silent_helper_times_out_and_a_dropped_one_fails_its_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(machines, "RECONNECT_PAUSES", (30.0,))
    silent_window = Window()
    silent = connected("silent", silent_window)
    silent_window.wait(lambda: "up" in silent_window.kinds())
    with pytest.raises(ServiceError, match=r"did not answer stat within 0\.2 s"):
        silent.call("stat", {"path": "/"}, timeout=0.2)
    silent.close()

    window = Window()
    dropped = connected("drop", window)
    window.wait(lambda: "up" in window.kinds())
    with pytest.raises(ServiceError, match="the connection dropped"):
        dropped.call("stat", {"path": "/"})
    window.wait(lambda: "down" in window.kinds())
    dropped.close()


def test_a_helper_that_will_not_end_is_killed(
    window: Window, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(machines, "STOP_SECONDS", 0.1)
    connection = connected("stubborn", window)
    window.wait(lambda: "up" in window.kinds())
    process = connection._process
    connection.close()
    assert process is not None and process.returncode is not None


def test_a_request_that_cannot_be_written_says_so(
    window: Window, monkeypatch: pytest.MonkeyPatch
):
    connection = connected("silent", window)
    window.wait(lambda: "up" in window.kinds())

    def broken(stream, value):
        raise BrokenPipeError("gone")

    monkeypatch.setattr(machines, "write_frame", broken)
    with pytest.raises(ServiceError, match="is not connected: gone"):
        connection.call("stat", {"path": "/"})
    monkeypatch.undo()
    connection.close()


def test_a_session_ssh_cannot_make_is_retried_with_what_ssh_said(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(machines, "RECONNECT_PAUSES", (0.01,))
    window = Window()
    command = [sys.executable, "-c", "import sys; sys.exit('ssh: Connection refused')"]
    connection = Connection(
        Machine("lab", "me@lab"),
        window.on_frame,
        window.on_state,
        command=lambda machine: (command, dict(os.environ)),
    )
    connection.open()
    window.wait(lambda: window.kinds().count("down") >= 2)
    assert dict(window.states)["down"] == "ssh: Connection refused"
    connection.close()


def test_a_closed_connection_never_starts(window: Window):
    connection = Connection(Machine("lab", "me@lab"), window.on_frame, window.on_state)
    connection.close()  # never opened: nothing to end
    connection._closing.set()
    connection.open()
    window.wait(lambda: window.kinds() == ["closed"])


def test_windows_has_no_master_connections(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(machines.sys, "platform", "win32")
    command, _ = agent_command(Machine("lab", "me@lab"))
    assert not any("ControlMaster" in part for part in command)


def test_a_relative_include_is_read_from_the_ssh_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "lab.conf").write_text("Host lab\n", encoding="utf-8")
    (tmp_path / ".ssh" / "config").write_text("Include lab.conf\n", encoding="utf-8")
    assert ssh_hosts() == ["lab"]


def test_the_helper_s_files_and_refusals_in_this_process(tmp_path: Path):
    output = io.BytesIO()
    helper = Agent(output)
    path = tmp_path / "a" / "b.txt"
    data = np.frombuffer(b"abcdef", dtype=np.uint8)
    assert helper.put({"path": str(path), "data": data}) == 6
    assert helper.put({"path": str(path), "data": data, "append": True}) == 12
    found = helper.stat({"path": str(path)})
    assert found is not None and found["size"] == 12
    assert (
        helper.fetch({"path": str(path), "offset": 3, "size": 4}).tobytes() == b"defa"
    )
    (tmp_path / "empty").mkdir()
    with pytest.raises(ServiceError, match="no process or job"):
        helper.cancel({"folder": str(tmp_path / "empty")})
    with pytest.raises(ServiceError, match="holds no run"):
        helper.remove({"folder": str(tmp_path / "empty")})
    # A run still going: its events come, and the watch goes on.
    from solprop.service.messages import Event  # noqa: PLC0415 - only here
    from solprop.service.record import EventLog  # noqa: PLC0415 - only here

    going = tmp_path / "going"
    going.mkdir()
    log = EventLog(going, "going")
    log.put(Event("going", "row", {"index": 0}))
    log.close()
    helper.follow({"folder": str(going), "offset": 0})
    deadline = time.monotonic() + WAIT
    while not any(f.get("folder") == str(going) for f in answers(output)):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    helper.unfollow({"folder": str(going)})
    # Followed, and unfollowed before anything is recorded: the watch ends.
    helper.follow({"folder": str(tmp_path / "empty"), "offset": 0})
    time.sleep(0.1)
    helper.unfollow({"folder": str(tmp_path / "empty")})
    time.sleep(1.2 * agent.FOLLOW_PERIOD)
    helper.close()


def test_a_machine_with_no_host_is_refused(window: Window):
    connection = Connection(Machine("typed"), window.on_frame, window.on_state)
    connection.open()
    window.wait(lambda: "refused" in window.kinds())
    assert dict(window.states)["refused"] == "typed has no SSH host"
    connection.close()
