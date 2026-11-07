# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solprop service (option C) and a window's connection to it, over a socket."""

import contextlib
import json
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from multiprocessing import Pipe
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path

import pytest

from service.cases import TINY
from solprop.service import remote, server
from solprop.service.messages import Envelope, Event, Reply, Request
from solprop.service.worker import RunSpec

WAIT = 60.0

SLOW = TINY.replace('length = "5 cm"', 'length = "40 cm"').replace(
    "n_save = 6", "n_save = 40"
)
"""The tiny capillary, long enough to be stopped or re-attached to while it runs."""


def wait_for(check: Callable[[], bool], timeout: float = WAIT) -> None:
    """Wait until a check passes."""
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run a service on a thread, quick to idle out, and give its state file."""
    monkeypatch.setattr(server, "IDLE_SECONDS", 1.0)
    path = tmp_path / "state" / "service.json"
    thread = threading.Thread(
        target=server.main, args=(path,), kwargs={"poll": 0.05}, daemon=True
    )
    thread.start()
    wait_for(path.exists)
    yield path
    thread.join(WAIT)
    assert not thread.is_alive() and not path.exists()  # idled out, and tidied


class Window:
    """What a window's callbacks receive."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.events: list[Event] = []
        self.runs: list[list[dict[str, object]]] = []
        self.lost = threading.Event()
        self.failed: list[tuple[str, list[Request], list[RunSpec]]] = []

    def connect(self, state: Path) -> remote.ServiceConnection:
        return remote.ServiceConnection(
            self.replies.append,
            self.events.append,
            on_runs=self.runs.append,
            on_lost=self.lost.set,
            state=state,
            on_failed=lambda reason, requests, runs: self.failed.append(
                (reason, requests, runs)
            ),
        )

    def kinds(self, run_id: str) -> list[str]:
        return [event.kind for event in self.events if event.run_id == run_id]


def test_a_window_asks_the_service_and_runs_in_it(state: Path, tmp_path: Path):
    window = Window()
    connection = window.connect(state)
    assert connection.alive
    first = connection.request("examples")
    wait_for(lambda: any(r.request_id == first for r in window.replies))
    assert window.replies[-1].ok and window.replies[-1].value
    # A reply for an old revision is dropped, as the in-window kernel's are.
    connection.revision = 2
    assert connection.revision == 2
    stale = connection.request("examples", revision=1)
    current = connection.request("examples", revision=2)
    wait_for(lambda: any(r.request_id == current for r in window.replies))
    assert all(r.request_id != stale for r in window.replies)
    # A run in the service streams its events to the window.
    spec = RunSpec("r1", str(tmp_path / "r1"), recipe=TINY, base=str(tmp_path))
    run = connection.start_run(spec)
    run.join(WAIT)
    assert not run.alive and window.kinds("r1")[-1] == "finished"
    connection.list_runs()
    wait_for(lambda: bool(window.runs))
    (listed,) = window.runs[-1]
    assert listed["run_id"] == "r1" and listed["last"] == "finished"
    connection.close()
    assert not connection.alive


def test_a_run_outlives_its_window_and_is_re_attached(state: Path, tmp_path: Path):
    first, second = Window(), Window()
    connection = first.connect(state)
    spec = RunSpec("r2", str(tmp_path / "r2"), recipe=SLOW, base=str(tmp_path))
    connection.start_run(spec)
    wait_for(lambda: "row" in first.kinds("r2"))
    connection.close()  # the window goes; the run does not
    again = second.connect(state)
    again.list_runs()
    wait_for(lambda: bool(second.runs))
    assert second.runs[-1][0]["alive"]
    run = again.attach(spec)
    wait_for(lambda: "row" in second.kinds("r2"))  # replayed from the start
    assert run.subscribe(0)
    run.stop()
    run.join(WAIT)
    assert second.kinds("r2")[-1] in ("stopped", "finished")
    again.control("nobody", "kill")  # an unknown run: ignored
    again.close()


def test_a_killed_run_is_lost_and_strangers_are_refused(state: Path, tmp_path: Path):
    window = Window()
    connection = window.connect(state)
    spec = RunSpec("r3", str(tmp_path / "r3"), recipe=SLOW, base=str(tmp_path))
    run = connection.start_run(spec)
    wait_for(lambda: "row" in window.kinds("r3"))
    run.kill()
    run.join(WAIT)
    assert window.kinds("r3")[-1] == "lost"
    recorded = json.loads(state.read_text(encoding="utf-8"))
    # Without the token, no message is read; the service goes on.
    with pytest.raises(Exception):  # noqa: B017 - multiprocessing's own error
        Client(recorded["address"], family=recorded["family"], authkey=b"wrong")
    # Messages of another shape are ignored.
    raw = Client(
        recorded["address"],
        family=recorded["family"],
        authkey=bytes.fromhex(recorded["authkey"]),
    )
    raw.send("not an envelope")
    raw.send(server.Envelope("what"))
    raw.send(server.Envelope("attach", "nobody"))
    raw.close()
    assert connection.request("examples") > 0
    wait_for(lambda: bool(window.replies))
    connection.close()


def test_a_hang_up_wakes_its_reader_and_an_acceptor_ends_with_its_listener():
    near, _far = Pipe()
    woken = threading.Event()

    def read() -> None:
        with contextlib.suppress(EOFError, OSError):
            near.recv()
        woken.set()

    threading.Thread(target=read, daemon=True).start()
    time.sleep(0.1)  # blocked in its read: on Linux, a bare close would not wake it
    server.hang_up(near)
    assert woken.wait(WAIT)
    server.hang_up(near)  # already closed: nothing to do
    closed: list[bool] = []

    class Other:
        def close(self) -> None:
            closed.append(True)

    server.hang_up(Other())
    assert closed == [True]

    class Gone(Listener):
        def __init__(self) -> None:  # a closed listener: no socket at all
            pass

        def accept(self) -> Connection:
            raise OSError("closed")

    closing = threading.Event()
    closing.set()
    server._accept(Gone(), server.Service(), closing)


def test_a_service_in_its_own_process_answers(tmp_path: Path, monkeypatch):
    # Service mode runs `python -m solprop.service.server`, where that module is
    # __main__: a message class defined there was not the one a window pickles, and
    # every message was ignored. The thread-run services above could not show it.
    started: list[subprocess.Popen[str]] = []
    real = remote.subprocess.Popen

    def keep(*arguments, **keywords):
        started.append(real(*arguments, **keywords))
        return started[-1]

    monkeypatch.setattr(remote.subprocess, "Popen", keep)
    state = tmp_path / "state" / "service.json"
    window = Window()
    connection = window.connect(state)  # none recorded: one is started
    try:
        connection.list_runs()
        wait_for(lambda: bool(window.runs))
        assert window.runs[-1] == []
        asked = connection.request("examples")
        wait_for(lambda: any(r.request_id == asked for r in window.replies))
    finally:
        connection.close()
        for process in started:
            process.terminate()
            process.wait(WAIT)


def test_the_state_file_and_starting_a_service(tmp_path: Path, monkeypatch):
    missing = tmp_path / "none" / "service.json"
    assert server.read_state(missing) is None
    junk = tmp_path / "junk.json"
    junk.write_text("[1, 2]", encoding="utf-8")
    assert server.read_state(junk) is None
    junk.write_text("{", encoding="utf-8")
    assert server.read_state(junk) is None
    monkeypatch.setenv("SOLPROP_GUI_STATE", str(tmp_path / "here"))
    assert server.state_file() == tmp_path / "here" / "service.json"
    started = []
    monkeypatch.setattr(
        remote.subprocess, "Popen", lambda command, **kw: started.append((command, kw))
    )
    remote.start_service(tmp_path / "s" / "service.json")
    command, options = started[-1]
    assert command[-1] == "solprop.service.server"
    assert options["env"]["SOLPROP_GUI_STATE"] == str(tmp_path / "s")
    remote.start_service()
    assert started[-1][1]["env"] is None
    # A recorded service that does not answer, and none that starts: refused.
    dead = tmp_path / "dead.json"
    dead.write_text(
        json.dumps(
            {"address": str(tmp_path / "x.sock"), "family": "AF_UNIX", "authkey": "00"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote, "START_SECONDS", 0.3)
    window = Window()
    window.connect(dead)  # sought on its own thread: the window goes on
    wait_for(lambda: bool(window.failed))
    assert "did not start" in window.failed[0][0]


def test_what_a_window_sends_while_connecting_is_held_then_sent(
    tmp_path: Path, monkeypatch
):
    near, far = Pipe()
    gate = threading.Event()

    def connect(_self: remote.ServiceConnection) -> object:
        gate.wait(WAIT)
        return near

    monkeypatch.setattr(remote.ServiceConnection, "_connect", connect)
    window = Window()
    connection = window.connect(tmp_path / "service.json")
    assert connection.alive  # still connecting
    asked = connection.request("examples")
    assert connection.list_runs()
    gate.set()  # connected: the held messages go, in order
    first, second = far.recv(), far.recv()
    assert first.kind == "request" and first.body.request_id == asked
    assert second.kind == "list_runs"
    connection.close()


def test_a_service_out_of_reach_gives_back_what_it_held(tmp_path: Path, monkeypatch):
    gate = threading.Event()

    def refuse(_self: remote.ServiceConnection) -> object:
        gate.wait(WAIT)
        msg = "the solprop service did not start"
        raise ConnectionError(msg)

    monkeypatch.setattr(remote.ServiceConnection, "_connect", refuse)
    window = Window()
    connection = window.connect(tmp_path / "service.json")
    asked = connection.request("examples")
    connection.start_run(RunSpec("r9", str(tmp_path / "r9"), recipe=TINY))
    gate.set()
    wait_for(lambda: bool(window.failed))
    reason, requests, runs = window.failed[0]
    assert "did not start" in reason
    assert [r.request_id for r in requests] == [asked]
    assert [spec.run_id for spec in runs] == ["r9"]
    assert not connection._send(Envelope("list_runs"))  # refused: nothing is held
    connection.close()
    # A window that went before the refusal hears nothing of it.
    gate.clear()
    gone = Window()
    closed = gone.connect(tmp_path / "service.json")
    closed.close(timeout=0.1)
    gate.set()
    closed._reader.join(WAIT)
    assert gone.failed == []


def test_what_is_sent_after_a_refusal_is_given_back_too(tmp_path: Path, monkeypatch):
    # The window's first request can reach the connection after its reader has failed:
    # the reader thread starts in the constructor and may fail before it returns.
    def refuse(_self: remote.ServiceConnection) -> object:
        msg = "the solprop service did not start"
        raise ConnectionError(msg)

    monkeypatch.setattr(remote.ServiceConnection, "_connect", refuse)
    window = Window()
    connection = window.connect(tmp_path / "service.json")
    connection._reader.join(WAIT)
    assert connection.failed and len(window.failed) == 1
    assert window.failed[0][1:] == ([], [])
    late = connection.request("examples")
    connection.start_run(RunSpec("r9", str(tmp_path / "r9"), recipe=TINY))
    assert not connection.list_runs()  # nothing to give back for this
    reason, requests, _runs = window.failed[1]
    assert "did not start" in reason
    assert [r.request_id for r in requests] == [late]
    assert window.failed[2][2][0].run_id == "r9"
    assert len(window.failed) == 3
    connection.close()
    assert connection.request("examples") > late  # closed: nothing told
    assert len(window.failed) == 3


def test_a_late_send_waits_for_the_held_batch_to_be_given_back(
    tmp_path: Path, monkeypatch
):
    # The reader has taken what was held but not yet given it back when a late request
    # arrives. A window that closes the connection on hearing of the late one must not
    # make the reader drop the held batch.
    def refuse(_self: remote.ServiceConnection) -> object:
        gate.wait(WAIT)
        msg = "the solprop service did not start"
        raise ConnectionError(msg)

    give_back = remote.ServiceConnection._give_back
    taken = threading.Event()

    def slow(self: remote.ServiceConnection, envelopes, *, news: bool) -> None:
        if news:
            taken.set()
            time.sleep(0.5)  # the late request arrives meanwhile
        give_back(self, envelopes, news=news)

    gate = threading.Event()
    monkeypatch.setattr(remote.ServiceConnection, "_connect", refuse)
    monkeypatch.setattr(remote.ServiceConnection, "_give_back", slow)
    told: list[list[int]] = []

    def on_failed(_reason: str, requests: list[Request], _runs: list[RunSpec]) -> None:
        told.append([r.request_id for r in requests])
        if threading.current_thread() is threading.main_thread():
            # The window's fallback runs here at once; the reader's news is queued.
            connection.close(timeout=0.1)

    connection = remote.ServiceConnection(
        lambda _reply: None,
        lambda _event: None,
        state=tmp_path / "service.json",
        on_failed=on_failed,
    )
    held = connection.request("examples")
    gate.set()
    assert taken.wait(WAIT)
    late = connection.request("examples")
    assert told == [[held], [late]]


def test_a_send_from_the_failure_news_itself_does_not_wait_for_it(
    tmp_path: Path, monkeypatch
):
    # A window whose news handler sends again, on the reader's thread, is given that
    # back too, rather than the reader waiting for itself.
    gate = threading.Event()

    def refuse(_self: remote.ServiceConnection) -> object:
        gate.wait(WAIT)
        msg = "the solprop service did not start"
        raise ConnectionError(msg)

    monkeypatch.setattr(remote.ServiceConnection, "_connect", refuse)
    told: list[list[int]] = []

    def on_failed(_reason: str, requests: list[Request], _runs: list[RunSpec]) -> None:
        told.append([r.request_id for r in requests])
        if len(told) == 1:
            connection.request("examples")

    connection = remote.ServiceConnection(
        lambda _reply: None,
        lambda _event: None,
        state=tmp_path / "service.json",
        on_failed=on_failed,
    )
    gate.set()
    connection._reader.join(WAIT)
    assert not connection._reader.is_alive()
    assert told == [[], [1]]
    connection.close()


def test_a_connection_closed_while_connecting_hangs_up_when_it_connects(
    tmp_path: Path, monkeypatch
):
    near, far = Pipe()
    gate = threading.Event()

    def connect(_self: remote.ServiceConnection) -> object:
        gate.wait(WAIT)
        return near

    monkeypatch.setattr(remote.ServiceConnection, "_connect", connect)
    connection = Window().connect(tmp_path / "service.json")
    connection.close(timeout=0.1)  # the window went before the service answered
    gate.set()
    connection._reader.join(WAIT)
    with pytest.raises(EOFError):
        far.recv()  # hung up, unused


def test_a_dropped_connection_and_a_closed_session(tmp_path: Path, monkeypatch):
    near, far = Pipe()
    monkeypatch.setattr(remote.ServiceConnection, "_connect", lambda _self: near)
    window = Window()
    connection = window.connect(tmp_path / "service.json")
    far.send("not an envelope")
    far.send(server.Envelope("runs", []))
    far.send(server.Envelope("event", Event("other", "row", {})))
    far.close()  # the service has gone
    assert window.lost.wait(WAIT)
    assert window.runs == [[]] and window.events[-1].run_id == "other"
    assert not connection._send(server.Envelope("list_runs"))
    connection.close()
    # A session whose window has gone is closed when the service next sends to it.
    mine, theirs = Pipe()
    theirs.close()
    session = server._Session(mine)
    session.send(server.Envelope("event"))
    assert not session.open
    session.send(server.Envelope("event"))  # closed: nothing sent


def test_an_idle_service_has_no_window_and_no_run(monkeypatch):
    monkeypatch.setattr(server, "IDLE_SECONDS", 0.0)
    service = server.Service()
    service.idle_since -= 1.0
    assert service.idle
    service.sessions.append(server._Session(Pipe()[0]))
    assert not service.idle
    service.close()  # no kernel started: nothing to end


def test_a_started_service_is_found_and_a_missing_one_refused(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(server, "IDLE_SECONDS", 0.5)
    monkeypatch.setattr(remote, "START_SECONDS", 0.5)
    monkeypatch.setattr(remote, "start_service", lambda _state: None)
    refused = Window()
    refused.connect(tmp_path / "nothing" / "service.json")  # none recorded
    wait_for(lambda: bool(refused.failed))
    path = tmp_path / "late" / "service.json"
    threads: list[threading.Thread] = []

    def start(state: Path | None) -> None:
        thread = threading.Thread(
            target=server.main, args=(state,), kwargs={"poll": 0.05}, daemon=True
        )
        thread.start()
        threads.append(thread)

    monkeypatch.setattr(remote, "START_SECONDS", WAIT)
    monkeypatch.setattr(remote, "start_service", start)
    connection = Window().connect(path)  # started, then found
    connection.close()
    # Another service has taken the state file: this one leaves it.
    path.write_text("{}", encoding="utf-8")
    threads[0].join(WAIT)
    assert path.read_text(encoding="utf-8") == "{}"


def test_the_connection_s_edges(tmp_path: Path, monkeypatch):
    near, far = Pipe()
    monkeypatch.setattr(remote.ServiceConnection, "_connect", lambda _self: near)
    replies: list[Reply] = []
    holder: list[remote.ServiceConnection] = []
    connection = remote.ServiceConnection(
        replies.append,
        lambda _event: None,
        on_lost=lambda: holder[0].close(),  # noqa: PLW0108 - made after it
        state=tmp_path / "s.json",
    )
    holder.append(connection)
    far.send(server.Envelope("runs", []))  # nobody listens for runs
    far.close()
    connection._reader.join(WAIT)
    near, far = Pipe()
    connection = remote.ServiceConnection(
        replies.append, lambda _event: None, state=tmp_path / "s.json"
    )
    connection._closing = True
    far.send(server.Envelope("reply", Reply("x", 1, 0)))
    connection._reader.join(WAIT)
    assert replies == []
    connection.close()


def test_the_service_s_edges():
    service = server.Service()
    service._replied(Reply("x", 999, 0))
    calls: list[str] = []

    class Run:
        alive = False

        def stop(self) -> None:
            calls.append("stop")

        def subscribe(self, _from_plane: int) -> None:
            calls.append("subscribe")

    service.runs["r"] = Run()  # pyright: ignore[reportArgumentType]
    service.control("r", "neither")
    assert calls == []
    closed = server._Session(Pipe()[0])
    closed.open = False
    service.serve(closed)  # already closed: served at once
    assert service.sessions == []


def test_an_event_goes_only_to_the_windows_attached_to_its_run():
    service = server.Service()
    mine, theirs = Pipe()
    attached = server._Session(mine)
    attached.runs.add("r")
    other = server._Session(Pipe()[0])
    service.sessions.extend([other, attached])
    service._event(Event("r", "row", {}))
    assert theirs.recv().body.run_id == "r" and service.last["r"] == "row"
