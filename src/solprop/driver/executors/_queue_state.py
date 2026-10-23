# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Coordinate atomic filesystem claims with leases and persistent advisory locks."""

import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from solprop.exceptions import DriverError
from solprop.output._file_lock import lock, unlock
from solprop.output.scan import _json


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold a kernel-owned lock whose ownership ends when the process dies.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.

    Yields
    ------
    None
        Control while the exclusive advisory lock is held.
    """
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock(descriptor)
        yield
    finally:
        unlock(descriptor)
        os.close(descriptor)


def sync_directory(path: Path) -> None:
    """Make filename publication durable on a filesystem supporting directory fsync.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    """Replace a JSON record only after its complete content has been fsynced.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.
    value : object
        JSON-serialisable record to publish atomically.
    """
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    """Read a JSON object while refusing malformed state containers.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.

    Returns
    -------
    dict
        Validated JSON object.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DriverError(f"queue record must be a JSON object: {path}")
    return value


def process_stamp(pid: int) -> str | None:
    """Read process creation identity where the operating system exposes it.

    Parameters
    ----------
    pid : int
        Positive operating-system process identifier.

    Returns
    -------
    str or None
        Process creation identity, or None if unavailable.
    """
    proc = Path(f"/proc/{pid}/stat")
    if proc.exists():
        return proc.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def owner() -> dict:
    """Identify this controller independently of each point's ownership token.

    Returns
    -------
    dict
        Host, PID, process creation identity and controller UUID.
    """
    return {
        "controller": uuid.uuid4().hex,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_start": process_stamp(os.getpid()),
    }


def live_owner(row: Mapping, age: float) -> bool:
    """Protect live local processes and remotely refreshed leases from takeover.

    Parameters
    ----------
    row : Mapping
        Point state containing owner identity and last refresh time.
    age : float
        Positive lease duration in seconds.

    Returns
    -------
    bool
        Whether takeover is forbidden by local liveness or a fresh lease.
    """
    holder = row.get("owner", {})
    if holder.get("host") == socket.gethostname():
        pid = int(holder.get("pid", -1))
        if pid < 1:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        previous = holder.get("process_start")
        current = process_stamp(pid)
        return previous is None or current is None or previous == current
    return time.time() - float(row.get("updated", 0)) < age


def state(path: Path) -> dict:
    """Read active state or the completion marker published before its removal.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.

    Returns
    -------
    dict
        Active state or the durable completion record.
    """
    active = path / "queue.json"
    return read_json(active if active.exists() else path / ".done")


def finish(path: Path, record: dict) -> None:
    """Publish completion before removing active queue state.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.
    record : dict
        Complete mutable queue state under its advisory lock.
    """
    atomic_json(path / "queue.json", record)
    if all(row["status"] == "done" for row in record["rows"]):
        atomic_json(path / ".done", record)
        (path / "queue.json").unlink()
        (path / ".nowork").unlink(missing_ok=True)
        sync_directory(path)
