# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The helper on a machine: ``python -m solprop.service.agent`` (design 13.4).

A window reaches a machine through one helper, started over SSH (or as a plain child
on the window's own computer), and speaks to it in frames
(:mod:`solprop.service.frames`) on the helper's standard input and output. The helper
keeps nothing the window needs kept: runs are files in their folders
(:mod:`solprop.service.record`), their states are read from those
(:mod:`solprop.service.runs`), and a helper that is killed, or whose connection drops,
is replaced by the next one started with nothing lost.

The first frame the helper writes is its ``hello``: the protocol and solprop versions,
the machine and whether Slurm is on it. Then each request is a frame
``{"id", "kind", "body"}``, answered by ``{"kind": "reply", "id", "value"}`` or
``{"kind": "error", "id", "message"}``. Two kinds of frame come unasked:
``{"kind": "events", "folder", "events", "offset"}`` for a run being followed, and
``{"kind": "kernel_reply", "reply"}`` for a kernel request.

The requests (:data:`REQUESTS`):

``list_runs`` ``{workspace}``
    Every run in a workspace with its state (:func:`~solprop.service.runs.list_runs`).
``start`` ``{spec, launcher, python?, resources?}``
    Prepare a run's folder and start it, directly or through Slurm; its summary.
``follow`` ``{folder, offset}`` and ``unfollow`` ``{folder}``
    Send a run's recorded events from a place on, as they are written, until it ends.
``stop`` ``{folder}``, ``cancel`` ``{folder}``
    Ask a run to stop (its stop file), or end its job (``scancel``, or ``SIGTERM``).
``remove`` ``{folder}``
    Delete an ended run's folder.
``stat`` ``{path}``, ``fetch`` ``{path, offset, size}``,
``put`` ``{path, data, append}``
    A file's size, a piece of it, or a piece written to it: a copy of a run's results,
    or a data file a recipe reads.
``slurm_info``
    The partitions Slurm offers here.
``kernel`` ``{request}``
    A request for the kernel the helper hosts on this machine, answered as a
    ``kernel_reply``: checks, previews and readings of this machine's results files.
"""

import contextlib
import os
import shutil
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from solprop.service.frames import Readable, Writable, read_frame, write_frame
from solprop.service.machines import CHUNK
from solprop.service.messages import PROTOCOL, Reply, Request, ServiceError
from solprop.service.record import read_events, request_stop
from solprop.service.worker import spec_from_value

__all__ = ["CHUNK", "FOLLOW_PERIOD", "REQUESTS", "Agent", "hello", "main"]

FOLLOW_PERIOD: Final = 0.5
"""How often a followed run's record is read for new events, s."""


def hello() -> dict[str, object]:
    """Return what the helper says first: who it is and what this machine has.

    Returns
    -------
    dict
        ``kind`` (``"hello"``), ``protocol``, ``solprop`` (its version), ``python``
        (this interpreter), ``host``, ``platform``, ``home``, ``pid`` and ``slurm``
        (whether ``sbatch`` is on the path).

    Examples
    --------
    >>> greeting = hello()
    >>> greeting["kind"], greeting["protocol"] == PROTOCOL
    ('hello', True)
    """
    from solprop import __version__  # noqa: PLC0415 - the package's own

    return {
        "kind": "hello",
        "protocol": PROTOCOL,
        "solprop": __version__,
        "python": sys.executable,
        "host": socket.gethostname(),
        "platform": sys.platform,
        "home": str(Path.home()),
        "pid": os.getpid(),
        "slurm": shutil.which("sbatch") is not None,
    }


def _folder(body: Mapping[str, Any]) -> Path:
    """Return the run folder a request names, with ``~`` expanded.

    Parameters
    ----------
    body : Mapping
        The request's body.

    Returns
    -------
    pathlib.Path
        The folder.
    """
    return Path(str(body["folder"])).expanduser()


class Agent:
    """Answer one window's requests on this machine.

    Parameters
    ----------
    output : solprop.service.frames.Writable
        Where its frames go: the real standard output.
    """

    def __init__(self, output: Writable) -> None:
        """Hold the output; no kernel until one is asked for."""
        self._output = output
        self._lock = threading.Lock()
        self._following: dict[str, threading.Event] = {}
        self._kernel: Any = None

    # ----------------------------------------------------------------- sending

    def send(self, value: Mapping[str, object]) -> None:
        """Write one frame, whole, from any thread.

        Parameters
        ----------
        value : Mapping
            The frame's value.
        """
        with self._lock:
            write_frame(self._output, value)

    def handle(self, message: Mapping[str, Any]) -> None:
        """Answer one request, with its reply or the error it raised.

        Parameters
        ----------
        message : Mapping
            ``id``, ``kind`` and ``body``.
        """
        identity = message.get("id")
        kind = str(message.get("kind"))
        handler = REQUESTS.get(kind)
        try:
            if handler is None:
                msg = f"no request {kind!r}"
                raise ServiceError(msg)
            value = handler(self, dict(message.get("body") or {}))
        except Exception as error:
            self.send({"kind": "error", "id": identity, "message": str(error)})
            return
        self.send({"kind": "reply", "id": identity, "value": value})

    # ---------------------------------------------------------------- requests

    def list_runs(self, body: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return every run in a workspace (``list_runs``).

        Parameters
        ----------
        body : Mapping
            ``workspace``.

        Returns
        -------
        list of dict
            Each run's summary.
        """
        from solprop.service.launch import slurm_states  # noqa: PLC0415 - light
        from solprop.service.runs import list_runs  # noqa: PLC0415 - light

        workspace = Path(str(body["workspace"])).expanduser()
        slurm = slurm_states if shutil.which("squeue") else None
        return list_runs(workspace, slurm)

    def start(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare a run's folder and start it (``start``).

        Parameters
        ----------
        body : Mapping
            ``spec`` (:func:`~solprop.service.worker.spec_value`), ``launcher``
            (``"direct"`` or ``"slurm"``), and optionally ``python`` (the job's;
            this helper's own interpreter by default, with its environment, which
            a job inherits) and ``resources`` (for Slurm).

        Returns
        -------
        dict
            The run's summary.

        Raises
        ------
        solprop.service.messages.ServiceError
            For a launcher that is not known.
        """
        from dataclasses import replace  # noqa: PLC0415 - light

        from solprop.service.launch import (  # noqa: PLC0415 - light
            SlurmResources,
            prepare,
            start_direct,
            submit_slurm,
        )
        from solprop.service.runs import summarise  # noqa: PLC0415 - light

        spec = spec_from_value(body["spec"])
        spec = replace(spec, folder=str(Path(spec.folder).expanduser()), record=True)
        folder = prepare(spec)
        python = str(Path(str(body.get("python") or sys.executable)).expanduser())
        launcher = body.get("launcher", "direct")
        if launcher == "direct":
            start_direct(folder, python)
        elif launcher == "slurm":
            resources = SlurmResources.from_value(body.get("resources") or {})
            submit_slurm(folder, python, resources, threads=spec.threads)
        else:
            msg = f"no launcher {launcher!r}: 'direct' or 'slurm'"
            raise ServiceError(msg)
        return summarise(folder)

    def follow(self, body: Mapping[str, Any]) -> int:
        """Send a run's events from a place on, as they come (``follow``).

        Parameters
        ----------
        body : Mapping
            ``folder``, and ``offset`` (bytes into its record; 0 for all).

        Returns
        -------
        int
            The offset the events start from.
        """
        folder = _folder(body)
        offset = int(body.get("offset", 0))
        key = str(folder)
        previous = self._following.pop(key, None)
        if previous is not None:
            previous.set()
        done = threading.Event()
        self._following[key] = done
        threading.Thread(
            target=self._follow, args=(folder, offset, done), daemon=True
        ).start()
        return offset

    def _follow(self, folder: Path, offset: int, done: threading.Event) -> None:
        """Read a run's record and send what is new, until it ends or is unfollowed.

        Parameters
        ----------
        folder : pathlib.Path
            The run's folder.
        offset : int
            Where to start, bytes.
        done : threading.Event
            Set to stop following.
        """
        while not done.is_set():
            try:
                events, offset = read_events(folder, offset)
            except ServiceError as error:
                self.send(
                    {
                        "kind": "follow_failed",
                        "folder": str(folder),
                        "message": str(error),
                    }
                )
                return
            if events:
                self.send(
                    {
                        "kind": "events",
                        "folder": str(folder),
                        "offset": offset,
                        "events": [
                            {"run_id": e.run_id, "kind": e.kind, "data": e.data}
                            for e in events
                        ],
                    }
                )
                if any(event.terminal for event in events):
                    return
            done.wait(FOLLOW_PERIOD)

    def unfollow(self, body: Mapping[str, Any]) -> None:
        """Stop sending a run's events (``unfollow``).

        Parameters
        ----------
        body : Mapping
            ``folder``.
        """
        done = self._following.pop(str(_folder(body)), None)
        if done is not None:
            done.set()

    def stop(self, body: Mapping[str, Any]) -> None:
        """Ask a run to stop, by its stop file (``stop``).

        Parameters
        ----------
        body : Mapping
            ``folder``.
        """
        request_stop(_folder(body))

    def cancel(self, body: Mapping[str, Any]) -> None:
        """End a run's job: ``scancel``, or ``SIGTERM`` to its process (``cancel``).

        Either way solprop writes where the run stood, so it can be resumed.

        Parameters
        ----------
        body : Mapping
            ``folder``.

        Raises
        ------
        solprop.service.messages.ServiceError
            If the run has no job this machine can end.
        """
        from solprop.service.launch import cancel_slurm  # noqa: PLC0415 - light
        from solprop.service.record import alive, read_job  # noqa: PLC0415 - light
        from solprop.service.runs import summarise  # noqa: PLC0415 - light

        folder = _folder(body)
        summary = summarise(folder)
        if summary["slurm_job"]:
            cancel_slurm(str(summary["slurm_job"]))
            return
        job = read_job(folder)
        if job is not None and job.host == socket.gethostname():
            if alive(job.pid, job.stamp):
                os.kill(job.pid, signal.SIGTERM)
            return
        msg = f"{folder.name} has no process or job on this machine to end"
        raise ServiceError(msg)

    def remove(self, body: Mapping[str, Any]) -> None:
        """Delete an ended run's folder (``remove``).

        Parameters
        ----------
        body : Mapping
            ``folder``.

        Raises
        ------
        solprop.service.messages.ServiceError
            If the run has not ended, or the folder holds no run.
        """
        from solprop.service.runs import summarise  # noqa: PLC0415 - light

        folder = _folder(body)
        summary = summarise(folder)
        if summary["state"] in {"queued", "starting", "running"}:
            msg = f"{folder.name} is {summary['state']}: stop or cancel it first"
            raise ServiceError(msg)
        if not any((folder / name).exists() for name in ("recipe.toml", "spec.json")):
            msg = f"{folder} holds no run"
            raise ServiceError(msg)
        shutil.rmtree(folder)

    def stat(self, body: Mapping[str, Any]) -> dict[str, object] | None:
        """Return a file's size and modification time (``stat``).

        Parameters
        ----------
        body : Mapping
            ``path``.

        Returns
        -------
        dict or None
            ``size`` (bytes) and ``mtime``; ``None`` for no such file.
        """
        path = Path(str(body["path"])).expanduser()
        try:
            found = path.stat()
        except FileNotFoundError:
            return None
        return {"size": found.st_size, "mtime": found.st_mtime}

    def fetch(self, body: Mapping[str, Any]) -> np.ndarray:
        """Return a piece of a file (``fetch``).

        Parameters
        ----------
        body : Mapping
            ``path``, ``offset`` (bytes) and ``size`` (bytes, at most :data:`CHUNK`).

        Returns
        -------
        numpy.ndarray
            The bytes, as ``uint8``; fewer than asked at the file's end.
        """
        path = Path(str(body["path"])).expanduser()
        size = min(int(body.get("size", CHUNK)), CHUNK)
        with path.open("rb") as stream:
            stream.seek(int(body.get("offset", 0)))
            data = stream.read(size)
        return np.frombuffer(data, dtype=np.uint8)

    def put(self, body: Mapping[str, Any]) -> int:
        """Write a piece of a file (``put``).

        Parameters
        ----------
        body : Mapping
            ``path``, ``data`` (``uint8``) and ``append`` (False starts the file).

        Returns
        -------
        int
            The file's size after the write, bytes.
        """
        path = Path(str(body["path"])).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.asarray(body["data"], dtype=np.uint8).tobytes()
        with path.open("ab" if body.get("append") else "wb") as stream:
            stream.write(data)
        return path.stat().st_size

    def slurm_info(self, body: Mapping[str, Any]) -> list[dict[str, object]]:
        """Return the partitions Slurm offers here (``slurm_info``).

        Parameters
        ----------
        body : Mapping
            Nothing.

        Returns
        -------
        list of dict
            Each partition's ``name``, ``default`` (whether it is the default),
            ``available``, ``time_limit``, ``cpus`` (per node), ``memory`` (per node,
            MB) and ``gres``.

        Raises
        ------
        solprop.service.messages.ServiceError
            If ``sinfo`` fails.
        """
        del body
        from solprop.service.launch import _run  # noqa: PLC0415 - light

        result = _run(["sinfo", "--noheader", "--format=%P|%a|%l|%c|%m|%G"])
        if result.returncode != 0:
            msg = f"sinfo failed: {result.stderr.strip()}"
            raise ServiceError(msg)
        partitions: dict[str, dict[str, object]] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 6:
                continue
            name = parts[0].rstrip("*")
            partitions.setdefault(
                name,
                {
                    "name": name,
                    "default": parts[0].endswith("*"),
                    "available": parts[1] == "up",
                    "time_limit": parts[2],
                    "cpus": parts[3],
                    "memory": parts[4],
                    "gres": "" if parts[5] == "(null)" else parts[5],
                },
            )
        return list(partitions.values())

    def kernel(self, body: Mapping[str, Any]) -> int:
        """Hand a request to this machine's kernel (``kernel``).

        Parameters
        ----------
        body : Mapping
            ``request``: ``kind``, ``payload``, ``request_id``, ``revision`` and
            ``key``.

        Returns
        -------
        int
            The request's id; its answer comes as a ``kernel_reply``.
        """
        given = dict(body["request"])
        request = Request(
            str(given["kind"]),
            dict(given.get("payload") or {}),
            int(given["request_id"]),
            revision=int(given.get("revision", 0)),
            key=str(given.get("key", "")),
        )
        client = self._kernel_client()
        if request.revision:
            client.revision = request.revision
        return client.submit(request)

    def _kernel_client(self) -> Any:
        """Return this machine's kernel, started on first use.

        Returns
        -------
        solprop.service.client.KernelClient
            The kernel.
        """
        if self._kernel is None:
            from solprop.service.client import KernelClient  # noqa: PLC0415 - light

            self._kernel = KernelClient(self._kernel_reply, on_lost=self._kernel_lost)
        return self._kernel

    def _kernel_reply(self, reply: Reply) -> None:
        """Send a kernel reply to the window.

        Parameters
        ----------
        reply : Reply
            The reply.
        """
        self.send({"kind": "kernel_reply", "reply": _reply_value(reply)})

    def _kernel_lost(self) -> None:
        """Say the kernel ended, and start another on the next request."""
        self._kernel = None
        self.send({"kind": "kernel_lost"})

    def close(self) -> None:
        """End every follow, and the kernel."""
        for done in self._following.values():
            done.set()
        if self._kernel is not None:
            with contextlib.suppress(Exception):
                self._kernel.close()


def _reply_value(reply: Reply) -> dict[str, object]:
    """Return a kernel reply as a frame carries it.

    Parameters
    ----------
    reply : Reply
        The reply.

    Returns
    -------
    dict
        Its fields.
    """
    return {
        "kind": reply.kind,
        "request_id": reply.request_id,
        "revision": reply.revision,
        "value": reply.value,
        "error": reply.error,
        "detail": reply.detail,
        "superseded": reply.superseded,
    }


REQUESTS: Final[dict[str, Callable[[Agent, Mapping[str, Any]], object]]] = {
    "cancel": Agent.cancel,
    "fetch": Agent.fetch,
    "follow": Agent.follow,
    "kernel": Agent.kernel,
    "list_runs": Agent.list_runs,
    "put": Agent.put,
    "remove": Agent.remove,
    "slurm_info": Agent.slurm_info,
    "start": Agent.start,
    "stat": Agent.stat,
    "stop": Agent.stop,
    "unfollow": Agent.unfollow,
}
"""The requests a helper answers, by kind."""


def serve(stream_in: Readable, stream_out: Writable) -> None:
    """Answer requests from a stream until it ends.

    Parameters
    ----------
    stream_in : solprop.service.frames.Readable
        Where requests come from.
    stream_out : solprop.service.frames.Writable
        Where frames go.
    """
    agent = Agent(stream_out)
    agent.send(hello())
    try:
        while True:
            try:
                message = read_frame(stream_in)
            except EOFError:
                return
            if not isinstance(message, dict):
                continue
            if message.get("kind") == "bye":
                return
            # Each request on a thread of its own: a slow one (a long sinfo, a big
            # listing) never holds up the window's others.
            threading.Thread(target=agent.handle, args=(message,), daemon=True).start()
    finally:
        agent.close()


def main() -> int:  # pragma: no cover - the helper's own process; test_agent drives it
    """Run the helper on this process's standard input and output.

    Anything else that writes to standard output (a library's print) would break the
    frames, so the frames go to a copy of it and standard output itself is pointed at
    standard error.

    Returns
    -------
    int
        0 when the window said goodbye or went away.
    """
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        serve(sys.stdin.buffer, output)
    finally:
        with contextlib.suppress(OSError):
            output.close()
    # Give the threads' last frames a moment, then leave without waiting for them.
    time.sleep(0.05)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command itself
    raise SystemExit(main())
