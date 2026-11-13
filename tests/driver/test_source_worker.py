# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The provider worker's contract, in real worker processes (SEP 49 section 7.2).

Manufactured adapters (``tests/source_adapters.py``) stand in for the provider: a
provider-agnostic one whose requests misbehave on purpose, and Pyroraptor's own adapter
around the manufactured haCC provider of ``hacc_cases``. Deadlines here are the test
harness's, to detect a broken cancellation; the production code has none.
"""

import dataclasses
import gc
import importlib.util
import io
import json
import os
import queue
import runpy
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil
import pytest

from solprop._interrupts import committing, handled, interruptible
from solprop.driver import _source_worker
from solprop.driver._source_worker import (
    CONTROL_LIMIT,
    POLL_INTERVAL,
    Opaque,
    Record,
    RemoteError,
    Worker,
    WorkerError,
    decode,
    encode,
    worker_environment,
)
from solprop.exceptions import RunTerminated, StopRequested

# How long a cancellation may take before the test calls it broken: the poll interval
# and a kill, with room for a loaded machine.
DEADLINE = 10.0
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and groups")


def _worker(adapter="source_adapters", **options):
    worker = Worker(adapter, environment=worker_environment(os.environ, {}), **options)
    worker.start()
    return worker


@pytest.fixture
def worker():
    started = _worker()
    yield started
    started.kill()


def _assert_ended(worker):
    """The process is reaped and both pipe ends closed."""
    process = worker._process
    assert process is not None and process.returncode is not None
    assert process.stdin is not None and process.stdin.closed
    assert process.stdout is not None and process.stdout.closed
    assert not worker.alive


@dataclasses.dataclass(frozen=True)
class _Pair:
    left: object
    right: object


@pytest.mark.medium
def test_the_codec_round_trips_values_and_records_without_pickle():
    value = {
        "arrays": [
            np.arange(6, dtype=">f8").reshape(2, 3),
            np.array([1 + 2j], dtype=np.complex128),
            np.array([True, False]),
            np.zeros((0, 4), dtype=np.int64),
        ],
        "tuple": (1, 2.5, None, "text"),
        "nonfinite": [float("nan"), float("inf"), -float("inf")],
        "record": _Pair((1,), np.ones(2)),
        "opaque": object(),
    }
    tree, arrays = encode(value)
    json.dumps(tree, allow_nan=False)
    back: Any = decode(json.loads(json.dumps(tree)), arrays)
    for sent, received in zip(value["arrays"], back["arrays"], strict=True):
        assert received.dtype == sent.dtype and received.shape == sent.shape
        np.testing.assert_array_equal(received, sent)
    assert back["tuple"] == value["tuple"]
    assert np.isnan(back["nonfinite"][0]) and back["nonfinite"][1:] == [
        float("inf"),
        -float("inf"),
    ]
    record = back["record"]
    assert isinstance(record, Record) and record.record_type.endswith("_Pair")
    assert record.left == (1,)
    assert back["opaque"].type_name == "builtins.object"


@pytest.mark.medium
def test_the_codec_carries_numpy_scalars_and_sends_callables_as_opaque():
    tree, arrays = encode({"step": np.int32(3), "z": np.float64(0.5), "f": len})
    back: Any = decode(json.loads(json.dumps(tree)), arrays)
    assert back["step"] == 3 and type(back["step"]) is int
    assert back["z"] == 0.5 and type(back["z"]) is float
    assert back["f"] == Opaque("builtins.builtin_function_or_method", repr(len))


@pytest.mark.medium
def test_a_received_record_re_encodes_with_its_type_and_computed_values():
    """A record sent on keeps its type; the sender's computed values win."""
    tree, arrays = encode(_Pair(1, 2), lambda value: {"ok": True, "kept": 1})
    received: Any = decode(tree, arrays)
    again: Any = decode(*encode(received, lambda value: {"ok": False}))
    assert again.record_type == received.record_type
    assert vars(again) == {"left": 1, "right": 2}
    assert again.computed == {"ok": False, "kept": 1}


@pytest.mark.medium
def test_a_record_s_computed_values_read_as_attributes_but_are_not_fields():
    tree, arrays = encode(_Pair(1, 2), lambda value: {"ok": True})
    record: Any = decode(tree, arrays)
    assert record.ok is True
    assert vars(record) == {"left": 1, "right": 2}
    with pytest.raises(AttributeError, match="no field 'missing'"):
        record.missing  # noqa: B018 - the attribute access is the test


@pytest.mark.medium
def test_the_worker_environment_drops_native_s_own_wait_settings_only():
    native = {
        "SOLPROP_NATIVE_WAIT": "spin",
        "GOMP_SPINCOUNT": "10000",
        "KMP_BLOCKTIME": "0",
    }
    environment = worker_environment(native, {"PYRORAPTOR_THREADS": "3"})
    assert "SOLPROP_NATIVE_WAIT" not in environment
    assert "GOMP_SPINCOUNT" not in environment and "KMP_BLOCKTIME" not in environment
    assert environment["PYRORAPTOR_THREADS"] == "3"
    user = {"OMP_WAIT_POLICY": "active", "KMP_BLOCKTIME": "200"}
    assert worker_environment(user, {}).items() >= user.items()
    edited = {
        "SOLPROP_NATIVE_WAIT": "spin",
        "GOMP_SPINCOUNT": "5",
        "KMP_BLOCKTIME": "0",
    }
    kept = worker_environment(edited, {})
    assert kept["GOMP_SPINCOUNT"] == "5" and "KMP_BLOCKTIME" not in kept


@pytest.mark.medium
def test_the_native_wait_settings_are_solprop_native_s_own(monkeypatch):
    """The worker removes exactly what solprop-native sets (ADR-071)."""
    spec = importlib.util.find_spec("solprop_native")
    if spec is None or spec.origin is None:
        pytest.skip("solprop-native is not installed")
    # Loaded by path: importing the package would start the native runtime.
    path = Path(spec.origin).parent / "_wait.py"
    loader = importlib.util.spec_from_file_location("native_wait_probe", path)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    # Its dataclasses look their module up by name.
    monkeypatch.setitem(sys.modules, "native_wait_probe", module)
    loader.loader.exec_module(module)
    assert _source_worker._NATIVE_WAIT_MARKER == module.MARKER
    assert _source_worker._NATIVE_WAIT_SETTINGS == module._SETTINGS


@pytest.mark.medium
def test_the_worker_environment_keeps_blas_idle():
    environment = worker_environment({}, {})
    assert "OPENBLAS_THREAD_TIMEOUT" in environment


@pytest.mark.medium
def test_values_cross_a_real_worker_bit_for_bit(worker):
    rng = np.random.default_rng(49)
    sent = {"field": rng.standard_normal((3, 257)), "labels": (1, 3), "z": 0.25}
    received = worker.request("echo", sent)
    np.testing.assert_array_equal(received["field"], sent["field"])
    assert received["field"].dtype == np.float64
    assert received["labels"] == (1, 3) and received["z"] == 0.25
    assert worker.identity["adapter"] == "manufactured"
    assert worker.pid == worker.identity["pid"] != os.getpid()


@pytest.mark.medium
def test_the_provider_s_printing_never_corrupts_a_frame(worker):
    assert worker.request("print") == "printed"
    assert worker.request("echo", [1, 2]) == [1, 2]


@pytest.mark.medium
def test_a_provider_exception_is_reported_by_name_and_ends_the_worker(worker):
    with pytest.raises(RemoteError) as raised:
        worker.request("raise")
    error = raised.value
    assert error.type_name == "builtins.ValueError"
    assert error.detail == "manufactured provider failure"
    assert error.classification == "manufactured"
    assert "manufactured provider failure" in error.remote_traceback
    # Never used again after a failure (section 6.5): its owner starts a fresh one.
    _assert_ended(worker)


@pytest.mark.medium
def test_a_worker_that_exits_is_a_named_failure_with_its_status_and_stderr(worker):
    with pytest.raises(WorkerError, match="exit status 7") as raised:
        worker.request("exit")
    assert "gives up" in "\n".join(raised.value.__notes__)
    _assert_ended(worker)
    with pytest.raises(WorkerError, match="not running"):
        worker.request("echo", 1)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("close_pipe", "ended"),
        ("malformed", "malformed worker message"),
    ],
)
@pytest.mark.medium
def test_a_closed_pipe_or_a_malformed_frame_ends_the_worker(worker, kind, message):
    with pytest.raises(WorkerError, match=message):
        worker.request(kind)
    _assert_ended(worker)


@pytest.mark.medium
def test_an_oversized_frame_is_refused_before_it_is_read():
    worker = _worker(limit=4096)
    with pytest.raises(WorkerError, match="over its bound of 4096"):
        worker.request("oversized", {"bytes": 10_000})
    _assert_ended(worker)


@pytest.mark.medium
def test_a_failure_while_starting_is_reported_by_name():
    worker = Worker(
        "source_adapters:failing_adapter",
        environment=worker_environment(os.environ, {}),
    )
    with pytest.raises(RemoteError, match="cannot start"):
        worker.start()
    _assert_ended(worker)


@pytest.mark.medium
def test_a_protocol_mismatch_is_refused_before_any_request(monkeypatch):
    monkeypatch.setattr(_source_worker, "PROTOCOL_VERSION", 99)
    worker = Worker("source_adapters", environment=worker_environment(os.environ, {}))
    with pytest.raises(WorkerError, match="protocol 1, not 99"):
        worker.start()
    _assert_ended(worker)


@pytest.mark.parametrize(
    "body",
    ['{"$d": []}', "[" * 100_000 + "]" * 100_000],
    ids=["bad tag", "too deep"],
)
@pytest.mark.medium
def test_a_message_this_side_cannot_decode_is_a_protocol_error(body):
    control = f'{{"kind": "reply", "body": {body}, "arrays": []}}'.encode()
    stream = io.BytesIO(_source_worker._HEADER.pack(len(control)) + control)
    with pytest.raises(WorkerError, match="malformed worker message"):
        _source_worker.read_message(stream.read, len(control))


def _frame(payload):
    return _source_worker._HEADER.pack(len(payload)) + payload


def _control(*arrays, kind: object = "reply"):
    message = {"kind": kind, "body": None, "arrays": list(arrays)}
    return _frame(json.dumps(message).encode())


_LIMIT = 4096
_HEADER = _source_worker._HEADER
_FOUR = {"dtype": "<f8", "shape": [4]}


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (_frame(b"{}")[:3], "ended inside a frame"),
        (_HEADER.pack(10), "ended inside a frame"),
        (_HEADER.pack(10) + b"abc", "ended inside a frame"),
        (_HEADER.pack(_LIMIT + 1), f"declares {_LIMIT + 1} bytes, over its bound"),
        (_control(kind=3), "malformed worker message: TypeError"),
        (_frame(b'{"kind": "reply", "body": null, "arrays": {}}'), "TypeError"),
        (_control({"dtype": "|O", "shape": [1]}), "TypeError: object"),
        (_control({"dtype": "<M8[s]", "shape": [1]}), "TypeError: datetime64"),
        (_control({"dtype": "<f8", "shape": [2, -1]}), "TypeError: float64"),
        (_control({"dtype": "<f8", "shape": [1024]}), "8192 bytes is over its bound"),
        (_control(_FOUR) + _frame(bytes(16)), "does not match its declared shape"),
        (_control(_FOUR), "does not match its declared shape"),
    ],
    ids=[
        "header cut",
        "payload missing",
        "payload cut",
        "frame over the bound",
        "kind not a string",
        "arrays not a list",
        "object dtype",
        "datetime dtype",
        "negative extent",
        "array over the bound",
        "short array frame",
        "missing array frame",
    ],
)
@pytest.mark.medium
def test_a_truncated_oversized_or_mistyped_message_is_refused(stream, message):
    """Every length and shape is checked before anything is allocated (P5)."""
    with pytest.raises(WorkerError, match=message):
        _source_worker.read_message(io.BytesIO(stream).read, _LIMIT)


@pytest.mark.medium
def test_end_of_file_between_messages_is_none_and_a_whole_message_reads():
    stream = io.BytesIO(_control(_FOUR) + _frame(np.arange(4.0).tobytes()))
    kind, body, arrays = cast("Any", _source_worker.read_message(stream.read, _LIMIT))
    assert (kind, body) == ("reply", None)
    np.testing.assert_array_equal(arrays[0], np.arange(4.0))
    assert _source_worker.read_message(stream.read, _LIMIT) is None


class _Unreadable(io.BufferedReader):
    """A reply pipe whose read fails with `error`."""

    def __init__(self, error):
        read_end, write_end = os.pipe()
        os.close(write_end)
        super().__init__(io.FileIO(read_end, "r"))
        self.error = error

    def read(self, size: int | None = -1):
        raise self.error


@pytest.mark.parametrize(
    ("error", "queued"),
    [
        (OSError("Bad file descriptor"), None),
        (ValueError("I/O operation on closed file"), None),
        (RuntimeError("reentrant call"), "could not be read: RuntimeError"),
    ],
    ids=["closed descriptor", "closed file", "anything else"],
)
@pytest.mark.medium
def test_the_reply_reader_always_queues_an_ending(error, queued):
    """A pipe the parent closed while killing is an exit; anything else a failure."""
    replies = queue.Queue()
    with _Unreadable(error) as stream:
        _source_worker._reader(stream, _LIMIT, replies)
    item = replies.get_nowait()
    if queued is None:
        assert item is None
    else:
        assert isinstance(item, WorkerError)
        assert queued in str(item)


@pytest.mark.medium
@POSIX
def test_a_worker_that_died_between_calls_is_a_named_failure(worker):
    assert worker._process is not None
    os.killpg(worker._process.pid, signal.SIGKILL)
    worker._process.wait(timeout=DEADLINE)
    with pytest.raises(WorkerError, match="exit status -9"):
        worker.request("echo", 1)
    _assert_ended(worker)


@pytest.mark.medium
def test_an_interruption_during_close_still_ends_the_worker(worker, monkeypatch):
    process = worker._process
    assert process is not None
    wait = process.wait

    def interrupted(timeout=None):
        if timeout is not None:
            raise KeyboardInterrupt
        return wait()

    monkeypatch.setattr(process, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        worker.close()
    assert process.returncode is not None and not worker.alive


@pytest.mark.medium
def test_a_worker_its_owner_forgets_is_ended_when_collected():
    worker = _worker()
    process = worker._process
    assert process is not None
    del worker
    gc.collect()
    assert process.returncode is not None


@pytest.mark.medium
def test_close_between_calls_ends_the_worker_in_order(worker):
    worker.close()
    _assert_ended(worker)
    assert worker._process is not None and worker._process.returncode == 0


@pytest.mark.medium
def test_a_worker_never_started_closes_and_kills_as_no_ops():
    worker = Worker("source_adapters", environment={})
    worker.close()
    worker.kill()
    assert worker._process is None and not worker.alive
    with pytest.raises(WorkerError, match="not running"):
        worker.request("echo", 1)


@pytest.mark.medium
def test_a_worker_that_answers_before_its_handshake_is_refused():
    worker = Worker(
        "source_adapters:early_reply_adapter",
        environment=worker_environment(os.environ, {}),
    )
    with pytest.raises(WorkerError, match="opened with 'reply', not a handshake"):
        worker.start()
    _assert_ended(worker)


@pytest.mark.medium
def test_a_message_out_of_turn_ends_the_worker(worker):
    with pytest.raises(WorkerError, match="answered 'hello_again' with 'hello'"):
        worker.request("hello_again")
    _assert_ended(worker)


@pytest.mark.medium
@POSIX
def test_a_worker_whose_close_overruns_the_grace_is_killed(worker, monkeypatch):
    monkeypatch.setattr(_source_worker, "CLOSE_GRACE", 0.2)
    assert worker.request("slow_close") is None
    started = time.monotonic()
    worker.close()
    assert time.monotonic() - started < DEADLINE
    _assert_ended(worker)
    assert worker._process is not None
    assert worker._process.returncode == -signal.SIGKILL


@pytest.mark.medium
@POSIX
def test_a_grandchild_holding_standard_error_does_not_hold_up_a_kill(
    worker, monkeypatch
):
    """The stream its reader is still inside is left open, never waited on."""
    monkeypatch.setattr(_source_worker, "CLOSE_GRACE", 0.2)
    grandchild = cast("int", worker.request("spawn"))
    process = worker._process
    assert process is not None and process.stderr is not None
    stderr = process.stderr
    try:
        started = time.monotonic()
        worker.kill()
        assert time.monotonic() - started < DEADLINE
        assert process.returncode == -signal.SIGKILL
        assert process.stdout is not None and process.stdout.closed
        assert not stderr.closed
    finally:
        os.kill(grandchild, signal.SIGKILL)
        assert _gone(grandchild)
        worker._threads[stderr].join(DEADLINE)
        stderr.close()


def _timed(worker, kind, expected):
    started = time.monotonic()
    with pytest.raises(expected):
        worker.request(kind)
    elapsed = time.monotonic() - started
    _assert_ended(worker)
    return elapsed


@pytest.mark.medium
def test_a_stop_flag_cancels_a_hanging_call_at_the_next_poll(worker):
    flag = threading.Event()
    worker.stop = flag
    threading.Timer(0.3, flag.set).start()
    assert _timed(worker, "hang", StopRequested) < DEADLINE


@POSIX
@pytest.mark.parametrize(
    ("number", "expected"),
    [(signal.SIGINT, KeyboardInterrupt), (signal.SIGTERM, RunTerminated)],
)
@pytest.mark.medium
def test_a_signal_cancels_a_hanging_call(worker, number, expected):
    defaults = {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGINT: signal.default_int_handler,
    }
    if signal.getsignal(number) is not defaults[number]:
        pytest.skip("an application handler owns the signal here")
    with handled():
        threading.Timer(0.3, os.kill, (os.getpid(), number)).start()
        assert _timed(worker, "hang", expected) < DEADLINE


@pytest.mark.medium
@POSIX
def test_a_signal_held_by_a_commit_is_taken_by_an_interruptible_wait(worker):
    """Native's ``advance`` waits inside ``committing``: the loop raises the signal."""
    if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        pytest.skip("an application handler owns SIGINT here")
    with handled():
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt), committing(), interruptible():
            worker.request("hang")
        assert time.monotonic() - started < DEADLINE
    _assert_ended(worker)


@pytest.mark.medium
@POSIX
def test_outside_an_interruptible_scope_the_commit_keeps_the_signal(worker):
    """The stop flag ends the wait; the signal stays held, dropped with the stop."""
    if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        pytest.skip("an application handler owns SIGINT here")
    flag = threading.Event()
    worker.stop = flag
    with handled():
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
        threading.Timer(1.0, flag.set).start()
        with pytest.raises(StopRequested), committing():
            worker.request("hang")
    _assert_ended(worker)


@pytest.mark.medium
def test_the_worker_imports_neither_backend_nor_the_parent_s_main(worker):
    loaded = worker.request("modules")
    names = loaded["modules"]
    assert not [name for name in names if name.split(".")[0] == "solprop_native"]
    assert not [name for name in names if "numpy_ref" in name]
    assert "pyroraptor" not in names
    assert loaded["main"] == "solprop.driver._source_worker"
    assert POLL_INTERVAL <= 0.2


# ----------------------------------------------- the worker's side, in this process
#
# The worker's own code runs in a process of its own above, where this process's
# coverage cannot see it. Here its entry point runs in this process instead, on pipes
# standing in for descriptors 0 and 1, with ``os._exit`` recorded rather than called.


class _Exits:
    """Record ``os._exit``'s statuses instead of ending this process."""

    def __init__(self):
        self.statuses = []
        self._called = threading.Event()

    def __call__(self, status):
        self.statuses.append(status)
        self._called.set()

    def wait(self):
        assert self._called.wait(DEADLINE)
        return self.statuses


@pytest.fixture
def exits(monkeypatch):
    recorder = _Exits()
    monkeypatch.setattr(os, "_exit", recorder)
    return recorder


def _serve(monkeypatch, adapter, requests):
    """Run the worker's ``main`` here; return its status, messages and signal handlers.

    The requests are written first; the parent's end of their pipe is closed after
    ``main`` returns, which is the watchdog's cue. Descriptors 0 and 1, standard
    output and the signal handlers are restored.
    """
    requests_read, requests_write = os.pipe()
    replies_read, replies_write = os.pipe()
    for kind, value in requests:
        _source_worker.write_message(
            lambda data: _source_worker._write_all(requests_write, data), kind, value
        )
    messages = []

    def _collect():
        while True:
            message = _source_worker.read_message(
                lambda size: os.read(replies_read, size), CONTROL_LIMIT
            )
            if message is None:
                return
            messages.append(message[:2])

    collector = threading.Thread(target=_collect, daemon=True)
    collector.start()
    monkeypatch.setattr(_source_worker, "PROTOCOL_DESCRIPTORS", [])
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    numbers = (signal.SIGINT, signal.SIGTERM)
    handlers = [signal.getsignal(number) for number in numbers]
    saved = os.dup(0), os.dup(1)
    try:
        os.dup2(requests_read, 0)
        os.dup2(replies_write, 1)
        os.close(requests_read)
        os.close(replies_write)
        status = _source_worker.main([adapter, str(CONTROL_LIMIT)])
        serving = [signal.getsignal(number) for number in numbers]
    finally:
        for descriptor, original in enumerate(saved):
            os.dup2(original, descriptor)
            os.close(original)
        for number, handler in zip(numbers, handlers, strict=True):
            signal.signal(number, handler)
    os.close(requests_write)
    inbound, outbound = _source_worker.PROTOCOL_DESCRIPTORS
    os.close(outbound)
    collector.join(DEADLINE)
    os.close(replies_read)
    return status, messages, serving, inbound


@pytest.mark.medium
@POSIX
def test_the_worker_serves_requests_in_order_until_close(monkeypatch, exits):
    status, messages, serving, inbound = _serve(
        monkeypatch,
        "source_adapters",
        [
            ("echo", {"field": np.arange(3.0)}),
            ("print", None),
            ("raise", None),
            ("unencodable", None),
            ("echo", "after"),
            ("close", None),
        ],
    )
    assert status == 0
    # The parent's pipe ended after the close: the watchdog's exit.
    assert exits.wait() == [_source_worker._ORPHANED]
    os.close(inbound)
    assert [kind for kind, _ in messages] == [
        "hello",
        "reply",
        "reply",
        "error",
        "error",
        "reply",
    ]
    hello: Any = messages[0][1]
    assert hello["protocol"] == _source_worker.PROTOCOL_VERSION
    assert hello["pid"] == os.getpid()
    assert hello["identity"]["adapter"] == "manufactured"
    np.testing.assert_array_equal(cast("Any", messages[1][1])["field"], np.arange(3.0))
    # What the provider printed went to standard error, not into a frame.
    assert messages[2][1] == "printed"
    raised: Any = messages[3][1]
    assert raised["type"] == "builtins.ValueError"
    assert raised["message"] == "manufactured provider failure"
    assert raised["classification"] == "manufactured"
    # A reply the codec refuses is reported, not half sent: the next one reads.
    assert cast("Any", messages[4][1])["type"] == "builtins.RecursionError"
    assert messages[5][1] == "after"
    assert serving == [signal.SIG_IGN, signal.SIG_IGN]


@pytest.mark.parametrize(
    ("adapter", "type_name", "detail"),
    [
        ("source_adapters:failing_adapter", "ValueError", "cannot start"),
        ("source_adapters_absent", "ModuleNotFoundError", "source_adapters_absent"),
    ],
)
@pytest.mark.medium
@POSIX
def test_an_adapter_that_cannot_start_is_reported_in_place_of_the_handshake(
    monkeypatch, exits, adapter, type_name, detail
):
    status, messages, _, inbound = _serve(monkeypatch, adapter, [])
    assert status == 1
    assert exits.wait() == [_source_worker._ORPHANED]
    os.close(inbound)
    assert [kind for kind, _ in messages] == ["error"]
    report: Any = messages[0][1]
    assert report["type"] == f"builtins.{type_name}"
    assert detail in report["message"]
    assert report["classification"] == "unexpected"


@pytest.mark.parametrize("ending", ["end of file", "garbage"])
@pytest.mark.medium
@POSIX
def test_the_watchdog_exits_when_the_parent_s_pipe_ends_or_breaks(exits, ending):
    """P8: the worker exits at once, whatever it is computing."""
    read_end, write_end = os.pipe()

    def write(data):
        _source_worker._write_all(write_end, data)

    _source_worker.write_message(write, "echo", 1)
    if ending == "garbage":
        write(_frame(b"{not json"))
    os.close(write_end)
    inbox = queue.Queue()
    _source_worker._watch(read_end, CONTROL_LIMIT, inbox)
    os.close(read_end)
    assert exits.statuses == [_source_worker._ORPHANED]
    assert inbox.get_nowait() == ("echo", 1)
    assert inbox.empty()


class _Exited(BaseException):
    """Raised by a recorded ``os._exit``, to leave the module's entry block."""


@pytest.mark.parametrize("fails", [False, True], ids=["returns", "raises"])
@pytest.mark.medium
def test_the_module_entry_runs_the_imported_main_and_never_finalises(
    monkeypatch, capsys, fails
):
    seen = []

    def main(argv):
        seen.append(list(argv))
        if fails:
            raise RuntimeError("the provider's runtime aborted")
        return 0

    def exit_now(status):
        raise _Exited(status)

    monkeypatch.setattr(_source_worker, "main", main)
    monkeypatch.setattr(os, "_exit", exit_now)
    monkeypatch.setattr(sys, "argv", ["worker", "source_adapters", "4096"])
    with warnings.catch_warnings(), pytest.raises(_Exited) as exited:
        # The worker's own command line ignores this runpy warning too.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
        runpy.run_module("solprop.driver._source_worker", run_name="__main__")
    assert seen == [["source_adapters", "4096"]]
    assert exited.value.args == (1 if fails else 0,)
    assert ("the provider's runtime aborted" in capsys.readouterr().err) is fails


_CHILD = """
import json, os, sys
from solprop.driver._source_worker import Worker, worker_environment
worker = Worker("source_adapters", environment=worker_environment(os.environ, {}))
worker.start()
loaded = worker.request("modules")
print(json.dumps({"pid": worker.pid, "main": loaded["main"],
                  "native": [m for m in loaded["modules"]
                             if m.split(".")[0] == "solprop_native"
                             or "numpy_ref" in m]}), flush=True)
"""


@pytest.mark.slow
def test_a_worker_started_from_a_script_or_python_c_never_re_imports_it(tmp_path):
    script = tmp_path / "user_script.py"
    script.write_text(
        "import sys\n"
        "if __name__ != '__main__':\n"
        "    raise SystemExit('the user script was imported again')\n" + _CHILD,
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    for command in ([sys.executable, str(script)], [sys.executable, "-c", _CHILD]):
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
            env=environment,
            cwd=tmp_path,
        )
        report = json.loads(done.stdout.strip().splitlines()[-1])
        assert report["main"] == "solprop.driver._source_worker"
        assert report["native"] == []


_HANGING_PARENT = """
import json, os, sys
from solprop._interrupts import handled
from solprop.driver._source_worker import Worker, WorkerError, worker_environment
worker = Worker("source_adapters", environment=worker_environment(os.environ, {}))
worker.start()
print(json.dumps({"worker": worker.pid}), flush=True)
try:
    with handled():
        worker.request("hang")
except KeyboardInterrupt:
    print(json.dumps({"ended": "interrupted"}), flush=True)
except WorkerError as error:
    print(json.dumps({"ended": "provider failure"}), flush=True)
"""


def _gone(pid):
    """Wait, within the harness deadline, for a process to end.

    A zombie has ended: in a container nothing may reap an orphan.
    """
    started = time.monotonic()
    while time.monotonic() - started < DEADLINE:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def _hanging_parent():
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    parent = subprocess.Popen(
        [sys.executable, "-c", _HANGING_PARENT],
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    assert parent.stdout is not None
    worker = json.loads(parent.stdout.readline())["worker"]
    time.sleep(2 * POLL_INTERVAL)
    return parent, worker


@POSIX
@pytest.mark.slow
def test_a_group_wide_ctrl_c_interrupts_the_run_and_spares_the_worker():
    parent, worker = _hanging_parent()
    os.killpg(parent.pid, signal.SIGINT)
    out, _ = parent.communicate(timeout=DEADLINE)
    assert json.loads(out.strip().splitlines()[-1]) == {"ended": "interrupted"}
    assert _gone(worker)


@POSIX
@pytest.mark.slow
def test_a_parent_killed_outright_leaves_no_worker_behind():
    parent, worker = _hanging_parent()
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait(timeout=DEADLINE)
    assert parent.stdout is not None
    parent.stdout.close()
    assert _gone(worker)
