# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Machines a window runs on, and its connection to each (design 13.4).

A machine is this computer or an SSH host. The connection is one long-lived helper
(:mod:`solprop.service.agent`) on that machine, started through the system's own
OpenSSH client, so the user's keys, agent, ``~/.ssh/config`` and jump hosts apply as
they do in a terminal. Nothing here stores a password: a host that asks for one is
answered through ``SSH_ASKPASS``, a program the window gives.

A :class:`Connection` reconnects by itself when its session drops, with a growing
pause, and calls back when it is up again; a window then lists the machine's runs and
follows the ones it shows from where it had read to, so a dropped network loses
nothing. Runs never depend on the connection: they are files on the machine.

This module is light (no simulation code), so the window's process uses it.
"""

import contextlib
import itertools
import os
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

from solprop.service.frames import read_frame, write_frame
from solprop.service.messages import PROTOCOL, ServiceError

__all__ = [
    "CHUNK",
    "LOCAL",
    "RECONNECT_PAUSES",
    "Connection",
    "Machine",
    "agent_command",
    "ssh_command",
    "ssh_hosts",
]

LOCAL: Final = "local"
"""The name of this computer, as a machine."""

RECONNECT_PAUSES: Final = (1.0, 2.0, 5.0, 10.0, 30.0)
"""The pauses before each try to reconnect, s; the last repeats."""

STOP_SECONDS: Final = 3.0
"""How long a session's process may take to end once told, before it is killed, s."""

CHUNK: Final = 4 << 20
"""The largest piece of a file one ``fetch`` or ``put`` carries, bytes."""


def ssh_hosts(config: Path | None = None) -> list[str]:
    r"""Return the hosts an SSH configuration names, for a window to offer.

    Patterns (``*``, ``?``, ``!``) are left out, since they name no one host; included
    files are read, as ``ssh`` reads them.

    Parameters
    ----------
    config : pathlib.Path or None, optional
        The configuration; the user's ``~/.ssh/config`` by default.

    Returns
    -------
    list of str
        The host names, in the order the files give them, each once.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> _ = (folder / "config").write_text("Host a b\nHost *.lab !c\nHost d\n")
    >>> ssh_hosts(folder / "config")
    ['a', 'b', 'd']
    """
    path = Path.home() / ".ssh" / "config" if config is None else config
    found: list[str] = []
    _read_hosts(path, found, set())
    return list(dict.fromkeys(found))


def _read_hosts(path: Path, found: list[str], seen: set[Path]) -> None:
    """Add the hosts one configuration file names, following its includes.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    found : list of str
        Where the hosts go.
    seen : set of pathlib.Path
        The files already read, so an include loop ends.
    """
    try:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        words = shlex.split(line, comments=True) if line.strip() else []
        if not words:
            continue
        keyword, *values = words
        if "=" in keyword:
            keyword, _, first = keyword.partition("=")
            values = [first, *values] if first else values
        keyword = keyword.lower()
        if keyword == "host":
            found.extend(v for v in values if not any(c in v for c in "*?!"))
        elif keyword == "include":
            for value in values:
                pattern = Path(value).expanduser()
                if not pattern.is_absolute():
                    pattern = Path.home() / ".ssh" / pattern
                anchor = Path(pattern.anchor)
                for name in sorted(anchor.glob(str(pattern.relative_to(anchor)))):
                    _read_hosts(Path(name), found, seen)


@dataclass(frozen=True)
class Machine:
    """A machine a window runs on, as it remembers it.

    Attributes
    ----------
    name : str
        What the window calls it; :data:`LOCAL` for this computer.
    destination : str or None
        What ``ssh`` is given: a host of the SSH configuration, or
        ``user@host``; ``None`` for this computer.
    port : int or None
        A port other than the configuration's.
    python : str
        The Python with solprop on the machine, as its shell finds it (``~`` is
        expanded there).
    workspace : str
        The folder of its runs, on the machine.
    launcher : str
        How runs start there: ``"direct"`` or ``"slurm"``.
    resources : dict
        Its Slurm template (:class:`~solprop.service.launch.SlurmResources`), as
        plain data.

    Examples
    --------
    >>> Machine.from_value(Machine("lab", "me@lab").value()) == Machine("lab", "me@lab")
    True
    """

    name: str
    destination: str | None = None
    port: int | None = None
    python: str = "python3"
    workspace: str = "~/solprop-runs"
    launcher: str = "direct"
    resources: dict[str, object] = field(default_factory=dict)

    @property
    def local(self) -> bool:
        """Return whether this is this computer.

        Returns
        -------
        bool
            True for :data:`LOCAL`; any other machine is reached over SSH.
        """
        return self.name == LOCAL

    def value(self) -> dict[str, object]:
        """Return the machine as plain data, as a window's settings keep it.

        Returns
        -------
        dict
            Its fields.
        """
        return asdict(self)

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> "Machine":  # noqa: UP037 - the runtime type hook resolves it in the class body
        """Return a machine from its plain data, ignoring fields it does not know.

        Parameters
        ----------
        value : Mapping
            The fields.

        Returns
        -------
        Machine
            The machine.
        """
        known = {
            name: value[name] for name in cls.__dataclass_fields__ if name in value
        }
        return cls(**known)  # pyright: ignore[reportArgumentType]


def ssh_command(
    machine: Machine, remote: str, *, askpass: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Return an ``ssh`` command that runs a shell command on a machine.

    Parameters
    ----------
    machine : Machine
        The machine, reached over SSH.
    remote : str
        The command, as the machine's shell reads it.
    askpass : str or None, optional
        A program ``ssh`` asks a password or code of, where it needs one.

    Returns
    -------
    tuple
        The command, and the environment to start it in.

    Raises
    ------
    solprop.service.messages.ServiceError
        For a machine with no SSH host.

    Examples
    --------
    >>> command, _ = ssh_command(Machine("lab", "me@lab", port=2222), "true")
    >>> command[0], command[-2:], "2222" in command
    ('ssh', ['me@lab', 'true'], True)
    """
    if not machine.destination:
        msg = f"{machine.name} has no SSH host"
        raise ServiceError(msg)
    environment = dict(os.environ)
    options = [
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        "-o",
        "ConnectTimeout=20",
        "-T",
    ]
    if sys.platform != "win32":
        # One master connection per machine: a password is typed once, and the copies
        # a window makes reuse it.
        control = Path.home() / ".ssh" / "solprop-%C"
        options += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control}",
            "-o",
            "ControlPersist=600",
        ]
    if machine.port is not None:
        options += ["-p", str(int(machine.port))]
    if askpass is not None:
        environment["SSH_ASKPASS"] = askpass
        environment["SSH_ASKPASS_REQUIRE"] = "force"
        environment.setdefault("DISPLAY", ":0")
    return ["ssh", *options, str(machine.destination), remote], environment


def agent_command(
    machine: Machine, *, askpass: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Return the command that starts a machine's helper, and its environment.

    Parameters
    ----------
    machine : Machine
        The machine.
    askpass : str or None, optional
        A program ``ssh`` asks a password or code of, where it needs one.

    Returns
    -------
    tuple
        The command, and the environment to start it in.

    Raises
    ------
    solprop.service.messages.ServiceError
        For a machine other than this computer with no SSH host.

    Examples
    --------
    >>> command, _ = agent_command(Machine("lab", "me@lab", python="~/env/bin/python"))
    >>> command[0], command[-2:]
    ('ssh', ['me@lab', 'exec ~/env/bin/python -m solprop.service.agent'])
    """
    if machine.local:
        return [sys.executable, "-m", "solprop.service.agent"], dict(os.environ)
    remote = f"exec {machine.python} -m solprop.service.agent"
    return ssh_command(machine, remote, askpass=askpass)


def _drain(stream: BinaryIO, into: list[bytes]) -> None:
    """Read a stream to its end, keeping what it said, and close it.

    Parameters
    ----------
    stream : BinaryIO
        The stream: a session's error output.
    into : list of bytes
        Where what it said goes.
    """
    with contextlib.suppress(OSError, ValueError):
        into.append(stream.read())
    with contextlib.suppress(OSError):
        stream.close()


def _said(drainer: threading.Thread, errors: list[bytes]) -> str:
    """Return what an ended session wrote to its error stream.

    Parameters
    ----------
    drainer : threading.Thread
        The thread reading that stream, which ends with it.
    errors : list of bytes
        Where it read the stream to.

    Returns
    -------
    str
        The text.
    """
    drainer.join(timeout=5)
    return b"".join(errors).decode("utf-8", "replace").strip()


def _development(version: str) -> bool:
    """Return whether a solprop version is a development build's.

    Parameters
    ----------
    version : str
        The version.

    Returns
    -------
    bool
        True for a ``.dev`` release or one with a local part.

    Examples
    --------
    >>> _development("0.2.0"), _development("0.2.1.dev17+g8077f26c3.d20261107")
    (False, True)
    """
    return ".dev" in version or "+" in version


class Connection:
    """A window's session with one machine's helper, which reconnects by itself.

    Parameters
    ----------
    machine : Machine
        The machine.
    on_frame : Callable
        Called, from the reader thread, with each frame the helper sends unasked
        (``events``, ``kernel_reply``, ...).
    on_state : Callable
        Called with the connection's state as it changes: ``"connecting"``,
        ``"up"`` (with the helper's ``hello``), ``"down"`` (with why) and
        ``"refused"`` (with why: another solprop, or none, on the machine), from
        the reader thread.
    askpass : str or None, optional
        A program ``ssh`` asks a password or code of.
    version : str or None, optional
        The solprop version the machine must have; this process's by default.
    command : Callable or None, optional
        What gives the command and environment; :func:`agent_command` by default.
    """

    def __init__(
        self,
        machine: Machine,
        on_frame: Callable[[dict[str, Any]], None],
        on_state: Callable[[str, object], None],
        *,
        askpass: str | None = None,
        version: str | None = None,
        command: Callable[[Machine], tuple[list[str], dict[str, str]]] | None = None,
    ) -> None:
        """Hold the machine; :meth:`open` starts the session."""
        from solprop import __version__  # noqa: PLC0415 - the package's own

        self.machine = machine
        self.state = "closed"
        self.hello: dict[str, Any] | None = None
        self._on_frame = on_frame
        self._on_state = on_state
        self._command = command or (lambda m: agent_command(m, askpass=askpass))
        self._version = __version__ if version is None else version
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[threading.Event, list[object]]] = {}
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._was_up = False
        self._up = threading.Event()

    def open(self) -> None:
        """Start the session, and keep it up until :meth:`close`."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"solprop-machine-{self.machine.name}", daemon=True
        )
        self._thread.start()

    def _set(self, state: str, detail: object = None) -> None:
        """Change the state and say so.

        Parameters
        ----------
        state : str
            The new state.
        detail : object, optional
            What goes with it.
        """
        self.state = state
        if state == "up":
            self._up.set()
        else:
            self._up.clear()
        self._on_state(state, detail)

    def _run(self) -> None:
        """Connect, read until the session drops, and reconnect, until closed."""
        tries = 0
        while not self._closing.is_set():
            self._set("connecting")
            self._was_up = False
            reason = self._session()
            if self._closing.is_set():
                break
            if self.state == "refused":
                return
            self._set("down", reason)
            self._fail_pending(reason)
            # A session that came up starts the pauses again from the shortest.
            tries = 0 if self._was_up else tries + 1
            pause = RECONNECT_PAUSES[min(tries, len(RECONNECT_PAUSES) - 1)]
            if self._closing.wait(pause):
                break
        self._set("closed")

    def _session(self) -> str:
        """Run one session to its end.

        Returns
        -------
        str
            Why it ended.
        """
        try:
            command, environment = self._command(self.machine)
        except ServiceError as error:
            self._set("refused", str(error))
            return str(error)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        except OSError as error:
            return f"could not start {command[0]}: {error}"
        self._process = process
        errors: list[bytes] = []
        drainer = threading.Thread(
            target=_drain, args=(cast("BinaryIO", process.stderr), errors), daemon=True
        )
        drainer.start()
        stream = cast("BinaryIO", process.stdout)
        try:
            return self._read(stream, lambda: _said(drainer, errors))
        finally:
            stream.close()

    def _read(self, stream: BinaryIO, said: Callable[[], str]) -> str:
        """Read a session's frames until it ends.

        Parameters
        ----------
        stream : BinaryIO
            The session's output.
        said : Callable
            Returns what the session wrote to its error stream, once it has ended.

        Returns
        -------
        str
            Why it ended.
        """
        try:
            greeting = read_frame(stream, skip_preface=True)
        except (EOFError, ServiceError) as error:
            self._stop_process()
            return self._no_helper(said() or str(error) or "the session ended")
        refusal = self._refusal(greeting)
        if refusal is not None:
            self._set("refused", refusal)
            self._stop_process()
            return refusal
        self.hello = cast("dict[str, Any]", greeting)
        self._was_up = True
        self._set("up", self.hello)
        while True:
            try:
                frame = read_frame(stream)
            except (EOFError, ServiceError) as error:
                self._stop_process()
                return said() or str(error) or "the session ended"
            if isinstance(frame, dict):
                self._deliver(cast("dict[str, Any]", frame))

    def _no_helper(self, said: str) -> str:
        """Return why a session gave no hello, as the user should read it.

        Parameters
        ----------
        said : str
            What ``ssh`` or the machine's shell wrote.

        Returns
        -------
        str
            The reason; a machine without solprop is refused, not retried.
        """
        missing = (
            "No module named" in said or "not found" in said or "No such file" in said
        )
        if missing:
            reason = (
                f"{self.machine.name} has no solprop at {self.machine.python}: "
                f"set the machine up, or name its Python ({said.splitlines()[-1]})"
            )
            self._set("refused", reason)
            return reason
        return said

    def _refusal(self, greeting: object) -> str | None:
        """Return why a helper's hello is refused, or ``None`` to accept it.

        Parameters
        ----------
        greeting : object
            The first frame.

        Returns
        -------
        str or None
            The reason.
        """
        if not isinstance(greeting, dict) or greeting.get("kind") != "hello":
            return "the machine's helper did not say hello"
        hello = cast("dict[str, Any]", greeting)
        if hello.get("protocol") != PROTOCOL:
            return (
                f"{self.machine.name} speaks protocol {hello.get('protocol')}, "
                f"this window {PROTOCOL}: install solprop {self._version} there"
            )
        theirs = str(hello.get("solprop"))
        if theirs != self._version:
            if _development(theirs) or _development(self._version):
                # A development build's version names its commit and checkout, so two
                # builds of one change rarely agree: allowed, and said.
                hello["warning"] = (
                    f"{self.machine.name} has solprop {theirs}, this window "
                    f"{self._version}"
                )
                return None
            return (
                f"{self.machine.name} has solprop {theirs}, this window "
                f"{self._version}: install solprop {self._version} there"
            )
        return None

    def _deliver(self, frame: dict[str, Any]) -> None:
        """Hand a frame to whoever waits for it, or to the window.

        Parameters
        ----------
        frame : dict
            The frame.
        """
        if frame.get("kind") in {"reply", "error"}:
            with self._lock:
                waiting = self._pending.pop(int(frame.get("id", -1)), None)
            if waiting is not None:
                event, box = waiting
                box.append(frame)
                event.set()
            return
        self._on_frame(frame)

    def _fail_pending(self, reason: str) -> None:
        """Answer every request still waiting with the session's end.

        Parameters
        ----------
        reason : str
            Why the session ended.
        """
        with self._lock:
            pending, self._pending = self._pending, {}
        for event, box in pending.values():
            box.append(
                {"kind": "error", "message": f"the connection dropped: {reason}"}
            )
            event.set()

    def call(
        self,
        kind: str,
        body: Mapping[str, object] | None = None,
        timeout: float = 120.0,
    ) -> Any:
        """Send a request and wait for its answer; never from the window's thread.

        A machine still connecting, or reconnecting, is waited for within the time.

        Parameters
        ----------
        kind : str
            The request (:data:`~solprop.service.agent.REQUESTS`).
        body : Mapping or None, optional
            Its body.
        timeout : float, optional
            The most seconds to wait.

        Returns
        -------
        object
            Its value.

        Raises
        ------
        solprop.service.messages.ServiceError
            If the machine is not connected within the time, the request fails, or
            no answer comes.
        """
        if self.state in {"connecting", "down"}:
            # Still coming up, or coming back: wait for it, within the same time.
            self._up.wait(timeout)
        identity, event, box = self.send(kind, body)
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(identity, None)
            msg = f"{self.machine.name} did not answer {kind} within {timeout:g} s"
            raise ServiceError(msg)
        answer = cast("dict[str, Any]", box[0])
        if answer.get("kind") == "error":
            raise ServiceError(str(answer.get("message")))
        return answer.get("value")

    def send(
        self, kind: str, body: Mapping[str, object] | None = None
    ) -> tuple[int, threading.Event, list[object]]:
        """Send a request without waiting.

        Parameters
        ----------
        kind : str
            The request.
        body : Mapping or None, optional
            Its body.

        Returns
        -------
        tuple
            Its id, the event set when it is answered, and the list its answer is
            put in.

        Raises
        ------
        solprop.service.messages.ServiceError
            If the machine is not connected.
        """
        process = self._process
        if self.state != "up" or process is None or process.stdin is None:
            msg = f"{self.machine.name} is not connected ({self.state})"
            raise ServiceError(msg)
        identity = next(self._ids)
        event = threading.Event()
        box: list[object] = []
        with self._lock:
            self._pending[identity] = (event, box)
            try:
                write_frame(
                    cast("BinaryIO", process.stdin),
                    {"id": identity, "kind": kind, "body": dict(body or {})},
                )
            except OSError as error:
                self._pending.pop(identity, None)
                msg = f"{self.machine.name} is not connected: {error}"
                raise ServiceError(msg) from error
        return identity, event, box

    def _stop_process(self) -> None:
        """End the session's process."""
        process = self._process
        self._process = None
        if process is None:
            return
        with contextlib.suppress(OSError):
            cast("BinaryIO", process.stdin).close()
        try:
            process.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def close(self) -> None:
        """End the session for good; the machine's runs go on."""
        self._closing.set()
        process = self._process
        if process is not None and process.stdin is not None and self.state == "up":
            with contextlib.suppress(OSError, ValueError):
                write_frame(cast("BinaryIO", process.stdin), {"kind": "bye"})
        self._stop_process()
        if self._thread is not None:
            self._thread.join(timeout=5)
