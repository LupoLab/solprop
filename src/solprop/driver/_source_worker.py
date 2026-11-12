# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a source provider in a worker process of its own (SEP 49 section 6.5).

Provider-agnostic: the start, framing, handshake, lifetime, signals and cancellation of
a worker in which an *adapter* module runs a provider. Nothing here knows TDSE or
Pyroraptor; ``solprop.driver._pyroraptor`` is one adapter, and the tests run a
manufactured one.

The parent starts ``sys.executable -m solprop.driver._source_worker <adapter> <limit>``
with :mod:`subprocess` -- never ``multiprocessing``, whose spawn re-imports the parent's
``__main__`` and whose fork inherits its runtimes -- in a session of its own, so that a
terminal's Ctrl-C or a scheduler's signal to the job reaches the parent alone, which
then ends the worker. The worker moves the protocol off its standard input and output
before importing the adapter, points standard output at standard error, ignores SIGINT
and SIGTERM, and exits at once when its pipe from the parent reaches end of file, even
during a call.

**Frames.** An 8-byte little-endian length, then the payload. A message is one control
frame, a UTF-8 JSON object ``{"kind", "body", "arrays"}``, followed by one raw frame per
entry of ``arrays``, each the array's bytes in the dtype (byte order included) and shape
its entry names. Nothing is pickled. Every frame's declared length is checked against
the channel's limit, and an array frame's against its declared shape, before anything is
allocated.

**Values.** A body is JSON with tagged objects: ``{"$a": i}`` the i-th array, ``{"$t":
[...]}`` a tuple, ``{"$d": {...}}`` a mapping, ``{"$f": "nan"}`` a non-finite float and
``{"$r": type, "f": fields, "c": computed}`` a record -- a dataclass or a plain object's
attributes, received as a :class:`Record`; any other object travels as its type and
``repr`` only (:class:`Opaque`).
"""

import contextlib
import dataclasses
import importlib
import json
import math
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import IO, Any, Protocol, cast, runtime_checkable

import numpy as np

from solprop._interrupts import poll, raise_held
from solprop.exceptions import BackendError
from solprop.log import StopFlag

__all__ = [
    "CONTROL_LIMIT",
    "POLL_INTERVAL",
    "PROTOCOL_VERSION",
    "Adapter",
    "Opaque",
    "Record",
    "RemoteError",
    "Worker",
    "WorkerError",
    "decode",
    "encode",
    "worker_environment",
]

PROTOCOL_VERSION = 1
"""The frame and message protocol both sides speak; a mismatch is refused."""

CONTROL_LIMIT = 1 << 26
"""The default bound on one frame, 64 MiB, raised by a caller to its prepared shapes."""

POLL_INTERVAL = 0.2
"""Seconds between the wait loop's checks for a signal or a stop (section 6.7)."""

CLOSE_GRACE = 5.0
"""Seconds an orderly close between calls waits for the worker before killing it."""

_HEADER = struct.Struct("<Q")
# The worker's private protocol descriptors, inbound and outbound, once it has moved
# them (tests write malformed frames through the outbound one).
PROTOCOL_DESCRIPTORS: list[int] = []
_TAIL_LINES = 40
# A worker that loses its parent exits with this status (the watchdog, P8).
_ORPHANED = 3
# How solprop-native marks the OpenMP wait variables it set itself, and what each of its
# choices sets (solprop_native._wait, ADR-071). A user's own settings carry no marker.
_NATIVE_WAIT_MARKER = "SOLPROP_NATIVE_WAIT"
_NATIVE_WAIT_SETTINGS = {
    "spin": {"GOMP_SPINCOUNT": "10000", "KMP_BLOCKTIME": "0"},
    "passive": {"OMP_WAIT_POLICY": "passive"},
}


class WorkerError(BackendError):
    """The worker exited, closed its pipe or broke the protocol."""


class RemoteError(BackendError):
    """An exception raised by the adapter inside the worker, as it reported it.

    Attributes
    ----------
    type_name : str
        The exception's ``module.qualname``.
    detail : str
        Its message.
    classification : str
        The adapter's classification of it.
    remote_traceback : str
        The worker's traceback text.
    """

    def __init__(
        self, type_name: str, detail: str, classification: str, remote_traceback: str
    ) -> None:
        super().__init__(f"{type_name.rpartition('.')[2]}: {detail}")
        self.type_name = type_name
        self.detail = detail
        self.classification = classification
        self.remote_traceback = remote_traceback


class Record(SimpleNamespace):
    """A provider record received from the worker: its fields as attributes.

    ``record_type`` names the original type; ``computed`` holds values the adapter
    computed from it (a property, a method's result) beside its fields, readable as
    attributes but not among ``vars()``, so a record's fields stay what was sent.
    """

    __slots__ = ("computed", "record_type")

    def __init__(
        self,
        record_type: str,
        fields: Mapping[str, object],
        computed: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(**fields)
        self.record_type = record_type
        self.computed = dict(computed or {})

    def __getattr__(self, name: str) -> object:
        """Return a computed value of that name; called only for absent fields."""
        try:
            return object.__getattribute__(self, "computed")[name]
        except KeyError:
            msg = f"{self.record_type} record has no field {name!r}"
            raise AttributeError(msg) from None


@dataclasses.dataclass(frozen=True)
class Opaque:
    """An object the codec does not carry: its type and ``repr`` only.

    Attributes
    ----------
    type_name : str
        Its ``module.qualname``.
    text : str
        Its ``repr``.
    """

    type_name: str
    text: str


def _name(kind: type) -> str:
    """Return a type's ``module.qualname``."""
    return f"{kind.__module__}.{kind.__qualname__}"


def encode(
    value: object, computed: Callable[[object], Mapping[str, object]] | None = None
) -> tuple[object, list[np.ndarray]]:
    """Encode a value as tagged JSON and the arrays it refers to.

    - ``computed``: for a record, the values to send beside its fields (an adapter's
      choice), or None for none.
    """
    arrays: list[np.ndarray] = []

    def _walk(item: object) -> object:  # noqa: PLR0911 - one return per value kind
        if item is None or isinstance(item, bool | str):
            return item
        if isinstance(item, int):
            return int(item)
        if isinstance(item, float):
            return item if math.isfinite(item) else {"$f": repr(item)}
        if isinstance(item, np.ndarray) and not item.dtype.hasobject:
            arrays.append(item)
            return {"$a": len(arrays) - 1}
        if isinstance(item, np.generic):
            return _walk(item.item())
        if isinstance(item, tuple):
            return {"$t": [_walk(entry) for entry in cast("tuple[object, ...]", item)]}
        if isinstance(item, list):
            return [_walk(entry) for entry in cast("list[object]", item)]
        if isinstance(item, Mapping):
            mapping = cast("Mapping[object, object]", item)
            return {"$d": {str(key): _walk(entry) for key, entry in mapping.items()}}
        fields = _fields(item)
        if fields is None:
            return {"$o": _name(type(item)), "repr": repr(item)}
        extra = {} if computed is None else computed(item)
        if isinstance(item, Record):
            extra = {**item.computed, **extra}
        return {
            "$r": item.record_type if isinstance(item, Record) else _name(type(item)),
            "f": {name: _walk(entry) for name, entry in fields.items()},
            "c": {name: _walk(entry) for name, entry in extra.items()},
        }

    return _walk(value), arrays


def _fields(value: object) -> dict[str, object] | None:
    """Return a record's fields: a dataclass's, or a plain object's attributes."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: getattr(value, item.name) for item in dataclasses.fields(value)
        }
    if isinstance(value, Opaque | type) or callable(value):
        return None
    attributes = getattr(value, "__dict__", None)
    return None if attributes is None else dict(attributes)


def decode(tree: object, arrays: Sequence[np.ndarray]) -> object:  # noqa: PLR0911
    """Rebuild a value from tagged JSON and its arrays, inverting :func:`encode`."""
    if isinstance(tree, list):
        return [decode(item, arrays) for item in cast("list[object]", tree)]
    if not isinstance(tree, dict):
        return tree
    node = cast("dict[str, Any]", tree)
    if "$a" in node:
        return arrays[node["$a"]]
    if "$t" in node:
        return tuple(decode(item, arrays) for item in node["$t"])
    if "$d" in node:
        return {key: decode(item, arrays) for key, item in node["$d"].items()}
    if "$f" in node:
        return float(node["$f"])
    if "$o" in node:
        return Opaque(node["$o"], node["repr"])
    return Record(
        node["$r"],
        {key: decode(item, arrays) for key, item in node["f"].items()},
        {key: decode(item, arrays) for key, item in node["c"].items()},
    )


# ------------------------------------------------------------------------- frames


def _read_exact(read: Callable[[int], bytes], size: int) -> bytearray | None:
    """Read exactly `size` bytes; None at end of file before the first byte."""
    buffer = bytearray()
    while len(buffer) < size:
        chunk = read(size - len(buffer))
        if not chunk:
            if buffer:
                msg = "the worker pipe ended inside a frame"
                raise WorkerError(msg)
            return None
        buffer += chunk
    return buffer


def _read_frame(read: Callable[[int], bytes], limit: int) -> bytearray | None:
    """Read one frame, refusing a declared length over `limit` before allocating."""
    header = _read_exact(read, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > limit:
        msg = f"a worker frame declares {length} bytes, over its bound of {limit}"
        raise WorkerError(msg)
    payload = _read_exact(read, length) if length else bytearray()
    if payload is None:
        msg = "the worker pipe ended inside a frame"
        raise WorkerError(msg)
    return payload


def read_message(
    read: Callable[[int], bytes], limit: int
) -> tuple[str, object, list[np.ndarray]] | None:
    """Read one message: its kind, its decoded body and its arrays; None at EOF."""
    control = _read_frame(read, limit)
    if control is None:
        return None
    try:
        message = json.loads(control.decode("utf-8"))
        kind, body, specs = message["kind"], message["body"], message["arrays"]
        if not isinstance(kind, str) or not isinstance(specs, list):
            raise TypeError(type(message).__name__)
        arrays: list[np.ndarray] = []
        for spec in cast("list[dict[str, Any]]", specs):
            dtype = np.dtype(spec["dtype"])
            shape = tuple(int(extent) for extent in spec["shape"])
            if (
                dtype.hasobject
                or dtype.kind not in "biufc"
                or min(shape, default=0) < 0
            ):
                raise TypeError(str(dtype))
            expected = math.prod(shape) * dtype.itemsize
            if expected > limit:
                msg = f"a worker array of {expected} bytes is over its bound of {limit}"
                raise WorkerError(msg)
            payload = _read_frame(read, expected)
            if payload is None or len(payload) != expected:
                msg = "a worker array frame does not match its declared shape"
                raise WorkerError(msg)
            arrays.append(np.frombuffer(payload, dtype=dtype).reshape(shape))
        return kind, decode(body, arrays), arrays
    except WorkerError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, UnicodeDecodeError) as error:
        msg = f"a malformed worker message: {type(error).__name__}: {error}"
        raise WorkerError(msg) from error


def write_message(
    write: Callable[[bytes], object],
    kind: str,
    value: object,
    computed: Callable[[object], Mapping[str, object]] | None = None,
) -> None:
    """Write one message of `kind` carrying `value`."""
    tree, arrays = encode(value, computed)
    contiguous = [np.ascontiguousarray(array) for array in arrays]
    control = json.dumps(
        {
            "kind": kind,
            "body": tree,
            "arrays": [
                {"dtype": array.dtype.str, "shape": list(array.shape)}
                for array in contiguous
            ],
        },
        allow_nan=False,
    ).encode("utf-8")
    parts = [_HEADER.pack(len(control)), control]
    for array in contiguous:
        data = array.tobytes()
        parts += [_HEADER.pack(len(data)), data]
    write(b"".join(parts))


# ---------------------------------------------------------------------- the parent


def worker_environment(
    environ: Mapping[str, str], explicit: Mapping[str, str]
) -> dict[str, str]:
    """Return the worker's environment: the parent's, less native's own wait variables.

    The OpenMP wait variables solprop-native set for itself (ADR-071), which its marker
    names, are removed so that the provider's runtime keeps its own defaults; a user's
    own settings pass through. The worker imports from this process's import path.
    ``explicit`` (the thread count, the device) wins.
    """
    environment = dict(environ)
    environment["PYTHONPATH"] = os.pathsep.join(entry for entry in sys.path if entry)
    settings = _NATIVE_WAIT_SETTINGS.get(environment.get(_NATIVE_WAIT_MARKER, ""), {})
    if settings:
        for name, value in settings.items():
            if environment.get(name, "").strip() == value:
                del environment[name]
        del environment[_NATIVE_WAIT_MARKER]
    environment.update(explicit)
    return environment


class Worker:
    """Own one worker process, its two pipes and the threads that read them.

    Started by :meth:`start`, which waits for the handshake; ended by :meth:`close`
    between calls (a shutdown request, a short grace, then a kill) or :meth:`kill`
    (at once). Every path reaps the process and closes both pipe ends. A worker is
    never used again after a failure of any kind or a cancellation.

    - ``adapter``: the adapter module's dotted name, imported in the worker.
    - ``environment``: the worker's whole environment (:func:`worker_environment`).
    - ``limit``: the bound on any frame either way, bytes.
    - ``stop``: the run's flag, polled while waiting.
    """

    def __init__(
        self,
        adapter: str,
        *,
        environment: Mapping[str, str],
        limit: int = CONTROL_LIMIT,
        stop: StopFlag | None = None,
    ) -> None:
        self.adapter = adapter
        self.limit = limit
        self.stop = stop
        self.identity: Mapping[str, object] = {}
        self.pid: int | None = None
        self._environment = dict(environment)
        self._process: subprocess.Popen[bytes] | None = None
        self._replies: queue.Queue[tuple[str, object] | BaseException | None] = (
            queue.Queue()
        )
        self._tail: deque[str] = deque(maxlen=_TAIL_LINES)
        self._threads: list[threading.Thread] = []
        self._usable = False

    # -- lifetime -------------------------------------------------------------------

    def start(self) -> Mapping[str, object]:
        """Start the worker and return its handshake's provider identity.

        A :class:`RemoteError` if the adapter could not start (a missing provider, say);
        a :class:`WorkerError` if the worker exits first or speaks another protocol.
        """
        creation = 0
        if sys.platform == "win32":  # pragma: no cover - Windows only
            creation = subprocess.CREATE_NEW_PROCESS_GROUP
        self._process = subprocess.Popen(
            [
                sys.executable,
                # The package imports this module before ``-m`` runs it, which runpy
                # warns of; ``main`` runs the imported copy, so nothing is duplicated.
                "-W",
                "ignore::RuntimeWarning:runpy",
                "-m",
                "solprop.driver._source_worker",
                self.adapter,
                str(self.limit),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._environment,
            start_new_session=sys.platform != "win32",
            creationflags=creation,
        )
        process = self._process
        assert process.stdout is not None and process.stderr is not None
        for target, stream in (
            (self._read, process.stdout),
            (self._drain, process.stderr),
        ):
            thread = threading.Thread(target=target, args=(stream,), daemon=True)
            thread.start()
            self._threads.append(thread)
        try:
            kind, body = self._wait()
            if kind != "hello":
                msg = f"the source worker opened with {kind!r}, not a handshake"
                raise WorkerError(msg)
            hello = cast("Mapping[str, object]", body)
            if hello.get("protocol") != PROTOCOL_VERSION:
                msg = (
                    f"the source worker speaks protocol {hello.get('protocol')!r}, "
                    f"not {PROTOCOL_VERSION}"
                )
                raise WorkerError(msg)
        except BaseException:
            self.kill()
            raise
        self.pid = cast("int", hello.get("pid"))
        self.identity = cast("Mapping[str, object]", hello.get("identity", {}))
        self._usable = True
        return self.identity

    def request(self, kind: str, value: object = None) -> object:
        """Send one request and return the adapter's reply.

        Waits in :data:`POLL_INTERVAL` steps; between them signal handlers run, a
        signal a commit holds is raised (:func:`solprop._interrupts.raise_held`) and the
        run's stop flag is polled. Any interruption kills the worker before it
        propagates. An exception the adapter reported is a :class:`RemoteError`; a
        worker that exits, closes its pipe or sends a malformed message is a
        :class:`WorkerError`. Either way the worker is ended: it is never used again
        after a failure (section 6.5), and its owner starts a fresh one.
        """
        if not self._usable or self._process is None:
            msg = "the source worker is not running"
            raise WorkerError(msg)
        process = self._process
        assert process.stdin is not None
        stdin = process.stdin
        try:
            write_message(stdin.write, kind, value)
            stdin.flush()
            reply, body = self._wait()
        except BaseException:
            self.kill()
            raise
        if reply != "reply":
            self.kill()
            msg = f"the source worker answered {kind!r} with {reply!r}"
            raise WorkerError(msg)
        return body

    def close(self) -> None:
        """End the worker between calls: ask it to exit, then kill it after a grace."""
        process = self._process
        if process is None:
            return
        if self._usable and process.poll() is None:
            try:
                assert process.stdin is not None
                write_message(process.stdin.write, "close", None)
                process.stdin.flush()
                process.wait(timeout=CLOSE_GRACE)
            except OSError, subprocess.TimeoutExpired, ValueError:
                pass
        self.kill()

    def kill(self) -> None:
        """End the worker at once, reap it and close both pipe ends."""
        self._usable = False
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                if sys.platform == "win32":  # pragma: no cover - Windows only
                    process.kill()
                else:
                    # Its own session: the group holds anything the provider started.
                    os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if process.stdin is not None:
            with contextlib.suppress(OSError):
                process.stdin.close()
        # The worker's ends are closed now, so its readers see the end of their pipes.
        for thread in self._threads:
            thread.join(timeout=CLOSE_GRACE)
        self._threads = []
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    @property
    def alive(self) -> bool:
        """Whether the worker is running and may take a request."""
        return (
            self._usable and self._process is not None and self._process.poll() is None
        )

    # -- waiting --------------------------------------------------------------------

    def _wait(self) -> tuple[str, object]:
        """Return the next message, polling signals and the stop flag meanwhile."""
        while True:
            try:
                item = self._replies.get(timeout=POLL_INTERVAL)
            except queue.Empty:
                raise_held()
                poll(self.stop)
                continue
            if isinstance(item, BaseException):
                raise item
            if item is None:
                raise self._exited()
            kind, body = item
            if kind == "error":
                report = cast("Mapping[str, str]", body)
                raise RemoteError(
                    report["type"],
                    report["message"],
                    report["classification"],
                    report["traceback"],
                )
            return kind, body

    def _exited(self) -> WorkerError:
        """Return the failure of a worker whose pipe ended: its status, its stderr."""
        process = self._process
        self._usable = False
        status: int | None = None
        if process is not None:
            try:
                status = process.wait(timeout=CLOSE_GRACE)
            except subprocess.TimeoutExpired:
                status = None
        tail = "\n".join(self._tail)
        error = WorkerError(
            f"the source worker ended (exit status {status}) without replying"
        )
        if tail:
            error.add_note(f"its last standard error:\n{tail}")
        return error

    def _read(self, stream: IO[bytes]) -> None:
        """Move the worker's messages to the reply queue until its pipe ends."""
        try:
            while True:
                message = read_message(stream.read, self.limit)
                if message is None:
                    self._replies.put(None)
                    return
                kind, body, _ = message
                self._replies.put((kind, body))
        except WorkerError as error:
            self._replies.put(error)
        except OSError, ValueError:
            # The parent closed the pipe while killing the worker.
            self._replies.put(None)

    def _drain(self, stream: IO[bytes]) -> None:
        """Pass the worker's standard error on, keeping its last lines for a failure."""
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            self._tail.append(text)
            with contextlib.suppress(OSError, ValueError):
                sys.stderr.write(text + "\n")


# ----------------------------------------------------------------------- the worker


@runtime_checkable
class Adapter(Protocol):
    """What an adapter module's ``make_adapter()`` returns, inside the worker."""

    def identity(self) -> Mapping[str, object]:
        """Return the provider's identity for the handshake."""
        ...

    def handle(self, kind: str, value: object) -> object:
        """Serve one request."""
        ...

    def classify(self, error: BaseException) -> str:
        """Classify an exception for the failure record."""
        ...

    def computed(self, value: object) -> Mapping[str, object]:
        """Return values to send beside a record's fields."""
        ...

    def close(self) -> None:
        """Release everything the adapter holds."""
        ...


@contextlib.contextmanager
def _private_streams() -> Iterator[tuple[int, int]]:
    """Move the protocol off fds 0 and 1, pointing 1 at stderr and 0 at the null device.

    Whatever the provider or its runtime prints then goes to standard error and cannot
    corrupt a frame.
    """
    inbound, outbound = os.dup(0), os.dup(1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    yield inbound, outbound


def _watch(inbound: int, limit: int, inbox: queue.Queue[tuple[str, object]]) -> None:
    """Read the parent's messages; exit the process at once when its pipe ends (P8)."""
    try:
        while True:
            message = read_message(lambda size: os.read(inbound, size), limit)
            if message is None:
                break
            kind, body, _ = message
            inbox.put((kind, body))
    except WorkerError, OSError:
        pass
    # The parent is gone, or wrote garbage: nothing the worker computes can be used.
    os._exit(_ORPHANED)


def _error_report(error: BaseException, adapter: Adapter | None) -> dict[str, str]:
    """Return an exception as type, message, classification and traceback text."""
    return {
        "type": _name(type(error)),
        "message": str(error),
        "classification": "unexpected" if adapter is None else adapter.classify(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def main(argv: Sequence[str]) -> int:
    """Serve one parent: handshake, requests until ``close``, then exit.

    - ``argv``: the adapter, ``module`` or ``module:factory`` (``make_adapter`` by
      default), and the frame bound in bytes.
    """
    adapter_name, limit = argv[0], int(argv[1])
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, signal.SIG_IGN)
    with _private_streams() as (inbound, outbound):
        PROTOCOL_DESCRIPTORS[:] = [inbound, outbound]

        def _send(kind: str, value: object, adapter: Adapter | None = None) -> None:
            write_message(
                lambda data: _write_all(outbound, data),
                kind,
                value,
                None if adapter is None else adapter.computed,
            )

        inbox: queue.Queue[tuple[str, object]] = queue.Queue()
        threading.Thread(
            target=_watch, args=(inbound, limit, inbox), daemon=True
        ).start()
        try:
            module_name, _, factory = adapter_name.partition(":")
            module = importlib.import_module(module_name)
            adapter = cast("Adapter", getattr(module, factory or "make_adapter")())
            identity = adapter.identity()
        except Exception as error:
            _send("error", _error_report(error, None))
            return 1
        _send(
            "hello",
            {"protocol": PROTOCOL_VERSION, "pid": os.getpid(), "identity": identity},
        )
        while True:
            kind, body = inbox.get()
            if kind == "close":
                adapter.close()
                return 0
            try:
                reply = adapter.handle(kind, body)
            except Exception as error:
                _send("error", _error_report(error, adapter))
                continue
            _send("reply", reply, adapter)


def _write_all(descriptor: int, data: bytes) -> None:
    """Write every byte to a descriptor, across partial writes."""
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view) :]


if __name__ == "__main__":
    # Run the imported module's main, not this __main__ copy's, so that the records
    # an adapter receives are the classes it imports.
    from solprop.driver._source_worker import main as _main

    status = _main(sys.argv[1:])
    # Skip interpreter finalisation: the provider's runtime is already closed by the
    # adapter, and nothing else of the worker's needs flushing.
    sys.stderr.flush()
    os._exit(status)
