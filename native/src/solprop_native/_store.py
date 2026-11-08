# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The per-user calibration store (solprop SEP 46 sections 5 and 6.8).

One directory per user, the platform's user cache directory unless
``SOLPROP_CACHE_DIR`` moves it::

    <store>/
      machines/<fingerprint>/
        current                  # names the published generation
        <generation>.json        # profile rows, their evidence and their FFT wisdom
        claim-<allocation>       # held while one process calibrates that class
        attempts.json            # inconclusive attempts, so that auto waits a day
      problems/<fingerprint>/<descriptor>.json   # stage-2 refinements

A calibration writes one complete generation file, then atomically replaces
``current`` to name it, so a reader that reads ``current`` once and then that
generation never sees a partial profile, and a failed or interrupted upgrade leaves
the previous generation current. ``SOLPROP_CALIBRATION_PATH`` lists read-only
directories of the same layout searched after the store, for example a cluster's
profiles for each node type.

This module is standard library only and loads nothing of the package, so solprop's
calibration controller can read it by path without starting the native runtime.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import secrets
import socket
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

__all__ = [
    "FORMAT",
    "MODES",
    "STALE_CLAIM_SECONDS",
    "Claim",
    "Generation",
    "Mode",
    "Row",
    "Store",
    "StoreWarning",
    "cache_directory",
    "generation_name",
    "lookup",
    "mode",
    "refinement",
    "search_path",
    "stores",
]

FORMAT = 1
"""The generation file's format; a file of another format is not read."""

type Mode = Literal["auto", "use", "off"]
MODES: tuple[Mode, ...] = ("auto", "use", "off")
"""``SOLPROP_CALIBRATION``'s values: calibrate on first use, use only, or neither."""

STALE_CLAIM_SECONDS = 60.0
"""How long past its deadline a claim is held before another process removes it."""

type Level = Literal["quick", "standard", "thorough"]
LEVELS: tuple[Level, ...] = ("quick", "standard", "thorough")


class StoreWarning(UserWarning):
    """A store that cannot be read or written; runs keep the built-in table."""


def mode(environ: Mapping[str, str] = os.environ) -> Mode:
    """Return ``SOLPROP_CALIBRATION``, ``"auto"`` when unset.

    Raises
    ------
    ValueError
        For a value other than ``auto``, ``use`` or ``off``.
    """
    value = environ.get("SOLPROP_CALIBRATION", "").strip().lower() or "auto"
    for known in MODES:
        if value == known:
            return known
    raise ValueError(
        f"SOLPROP_CALIBRATION must be one of {', '.join(MODES)}, got {value!r}"
    )


def cache_directory(
    environ: Mapping[str, str] = os.environ, system: str | None = None
) -> Path:
    """Return the store's directory: the platform's user cache directory.

    ``SOLPROP_CACHE_DIR`` when set; otherwise ``$XDG_CACHE_HOME/solprop`` or
    ``~/.cache/solprop`` on Linux and other POSIX systems,
    ``~/Library/Caches/solprop`` on macOS and ``%LOCALAPPDATA%\\solprop\\Cache`` on
    Windows.
    """
    explicit = environ.get("SOLPROP_CACHE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    system = system or platform.system()
    home = Path(environ.get("HOME") or Path.home())
    if system == "Darwin":
        return home / "Library" / "Caches" / "solprop"
    if system == "Windows":
        local = environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else home / "AppData" / "Local"
        return base / "solprop" / "Cache"
    xdg = environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg) if xdg else home / ".cache") / "solprop"


def search_path(environ: Mapping[str, str] = os.environ) -> tuple[Path, ...]:
    """Return ``SOLPROP_CALIBRATION_PATH``'s read-only directories, in order."""
    value = environ.get("SOLPROP_CALIBRATION_PATH", "")
    return tuple(Path(part) for part in value.split(os.pathsep) if part.strip())


@dataclass(frozen=True)
class Row:
    """One allocation class's measured settings.

    Attributes
    ----------
    allocation : Mapping[str, object]
        The allocation class it was measured in (``_allocation.AllocationClass``'s
        record).
    level : str
        The calibration level that measured it.
    process : Mapping[str, object]
        Its process settings: ``threads`` and ``wait`` (``"spin"`` or
        ``"passive"``), and any other the level chose.
    workspace : Mapping[str, object]
        Its workspace settings, a partial ``_execution.ExecutionRecord``.
    fitted_under : Mapping[str, object]
        The process settings the workspace settings were validated under; they apply
        only to a process running on the same.
    evidence : Mapping[str, object]
        What was measured: ratios, noise and why each setting was kept.
    planning : Mapping[str, object]
        Its FFT planning: ``efforts``, each plan group's effort; the plans' wisdom is
        the generation's.
    """

    allocation: Mapping[str, object]
    level: str
    process: Mapping[str, object] = field(default_factory=dict[str, object])
    workspace: Mapping[str, object] = field(default_factory=dict[str, object])
    fitted_under: Mapping[str, object] = field(default_factory=dict[str, object])
    evidence: Mapping[str, object] = field(default_factory=dict[str, object])
    planning: Mapping[str, object] = field(default_factory=dict[str, object])

    def record(self) -> dict[str, object]:
        """Return the row as JSON data."""
        return {
            "allocation": dict(self.allocation),
            "level": self.level,
            "process": dict(self.process),
            "workspace": dict(self.workspace),
            "fitted_under": dict(self.fitted_under),
            "evidence": dict(self.evidence),
            "planning": dict(self.planning),
        }

    @classmethod
    def read(cls, data: object) -> Row:
        """Read a row written by :meth:`record`.

        Raises
        ------
        ValueError
            When `data` is not a row.
        """
        values = _mapping(data, "row")
        level = values.get("level")
        if not isinstance(level, str) or level not in LEVELS:
            raise ValueError(f"a row's level must be one of {LEVELS}, got {level!r}")
        return cls(
            allocation=_mapping(values.get("allocation"), "allocation"),
            level=level,
            process=_mapping(values.get("process", {}), "process"),
            workspace=_mapping(values.get("workspace", {}), "workspace"),
            fitted_under=_mapping(values.get("fitted_under", {}), "fitted_under"),
            evidence=_mapping(values.get("evidence", {}), "evidence"),
            planning=_mapping(values.get("planning", {}), "planning"),
        )


@dataclass(frozen=True)
class Generation:
    """One published calibration: its rows, evidence and FFT wisdom together.

    Attributes
    ----------
    fingerprint : str
        The machine and build it measured (``_fingerprint``).
    name : str
        Its generation name, unique within the fingerprint.
    created : str
        When it was published, ISO 8601 in UTC.
    rows : tuple of Row
        One per allocation class measured.
    wisdom : Mapping[str, str]
        Each FFT shape class's winning wisdom, by plan key.
    source : Path or None
        The file it was read from; ``None`` before publication.
    """

    fingerprint: str
    name: str
    created: str
    rows: tuple[Row, ...]
    wisdom: Mapping[str, str] = field(default_factory=dict[str, str])
    source: Path | None = None

    def record(self) -> dict[str, object]:
        """Return the generation as JSON data."""
        return {
            "format": FORMAT,
            "fingerprint": self.fingerprint,
            "name": self.name,
            "created": self.created,
            "rows": [row.record() for row in self.rows],
            "wisdom": dict(self.wisdom),
        }

    @classmethod
    def read(cls, data: object, source: Path | None = None) -> Generation:
        """Read a generation written by :meth:`record`.

        Raises
        ------
        ValueError
            When `data` is not a generation of this format.
        """
        values = _mapping(data, "generation")
        if values.get("format") != FORMAT:
            raise ValueError(
                f"generation format {values.get('format')!r}, expected {FORMAT}"
            )
        fingerprint, name, created = (
            values.get("fingerprint"),
            values.get("name"),
            values.get("created"),
        )
        if not (
            isinstance(fingerprint, str)
            and isinstance(name, str)
            and isinstance(created, str)
        ):
            raise ValueError("a generation names its fingerprint, name and date")
        rows = values.get("rows")
        if not isinstance(rows, list):
            raise ValueError("a generation's rows must be a list")
        wisdom = _mapping(values.get("wisdom", {}), "wisdom")
        if not all(isinstance(text, str) for text in wisdom.values()):
            raise ValueError("a generation's wisdom must be text")
        return cls(
            fingerprint=fingerprint,
            name=name,
            created=created,
            rows=tuple(Row.read(row) for row in cast(list[object], rows)),
            wisdom=cast(Mapping[str, str], wisdom),
            source=source,
        )


def _mapping(data: object, what: str) -> dict[str, object]:
    """Return `data` as a string-keyed mapping, or raise naming `what`."""
    if not isinstance(data, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], data)
    ):
        raise ValueError(f"a {what} must be a mapping with string keys")
    return dict(cast(dict[str, object], data))


@dataclass(frozen=True)
class Claim:
    """A held calibration claim on one allocation class; release it when done.

    Attributes
    ----------
    path : Path
        The claim file, created exclusively.
    host, pid : str, int
        Who holds it.
    deadline : float
        When its calibration must end, in seconds since the epoch.
    """

    path: Path
    host: str
    pid: int
    deadline: float

    def release(self) -> None:
        """Remove the claim, if it is still this one's."""
        with contextlib.suppress(OSError, ValueError):
            held = json.loads(self.path.read_text(encoding="utf-8"))
            if held.get("host") == self.host and held.get("pid") == self.pid:
                self.path.unlink()


class Store:
    """A calibration store rooted at `root`; read-only unless writes are asked for.

    Parameters
    ----------
    root : Path
        The store's directory (:func:`cache_directory`, or a read-only one).
    """

    def __init__(self, root: Path):
        self.root = root

    def directory(self, fingerprint: str) -> Path:
        """Return one fingerprint's directory."""
        return self.root / "machines" / fingerprint

    def current(self, fingerprint: str) -> Generation | None:
        """Return the published generation, or ``None`` when there is none.

        A corrupt, mismatched or unreadable file is skipped with one
        :class:`StoreWarning`.
        """
        directory = self.directory(fingerprint)
        try:
            name = (directory / "current").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as error:
            _warn(f"cannot read the calibration store {directory}: {error}")
            return None
        path = directory / f"{name}.json"
        try:
            generation = Generation.read(
                json.loads(path.read_text(encoding="utf-8")), source=path
            )
        except (OSError, ValueError) as error:
            _warn(f"ignoring the calibration profile {path}: {error}")
            return None
        if generation.fingerprint != fingerprint or generation.name != name:
            _warn(f"ignoring the calibration profile {path}: it names another machine")
            return None
        return generation

    def publish(self, generation: Generation) -> Path:
        """Write `generation` completely, then make it current; return its file.

        Raises
        ------
        OSError
            When the store cannot be written; the previous generation stays current.
        """
        directory = self.directory(generation.fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{generation.name}.json"
        _write_atomically(path, json.dumps(generation.record(), indent=1) + "\n")
        _write_atomically(directory / "current", generation.name + "\n")
        return path

    def clear(self, fingerprint: str | None = None) -> int:
        """Remove ``current`` and every generation, never a live claim; return a count.

        `fingerprint` limits it to one machine and build; ``None`` clears them all.
        """
        removed = 0
        base = self.root / "machines"
        directories = (
            [self.directory(fingerprint)]
            if fingerprint is not None
            else [path for path in base.glob("*") if path.is_dir()]
        )
        for directory in directories:
            for path in (directory / "current", *directory.glob("*.json")):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                    removed += 1
        return removed

    def claim(self, fingerprint: str, allocation: str, deadline: float) -> Claim | None:
        """Take the calibration claim on one allocation class, or ``None`` if held.

        The claim file is created exclusively (``O_CREAT | O_EXCL``, on Windows too)
        and names the host, process and deadline. A claim whose deadline passed
        :data:`STALE_CLAIM_SECONDS` ago, or whose process is gone on this host, is
        stale and replaced.
        """
        directory = self.directory(fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"claim-{allocation}"
        held = Claim(path, socket.gethostname(), os.getpid(), deadline)
        payload = json.dumps(
            {
                "host": held.host,
                "pid": held.pid,
                "start": time.time(),
                "deadline": deadline,
            }
        )
        for _ in range(2):
            try:
                handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if not self._stale(path):
                    return None
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                continue
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            return held
        return None

    def holder(self, fingerprint: str, allocation: str) -> Claim | None:
        """Return the live claim on one allocation class, or ``None``."""
        path = self.directory(fingerprint) / f"claim-{allocation}"
        if self._stale(path):
            return None
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
            return Claim(
                path, str(held["host"]), int(held["pid"]), float(held["deadline"])
            )
        except OSError, ValueError, KeyError, TypeError:
            return None

    @staticmethod
    def _stale(path: Path) -> bool:
        """Whether a claim file is gone, unreadable, past its deadline or orphaned."""
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
            host, pid, deadline = (
                str(held["host"]),
                int(held["pid"]),
                float(held["deadline"]),
            )
        except FileNotFoundError:
            return True
        except OSError, ValueError, KeyError, TypeError:
            # Being written: an exclusive create comes before its contents.
            try:
                return time.time() - path.stat().st_mtime > STALE_CLAIM_SECONDS
            except FileNotFoundError:
                return True
        if time.time() > deadline + STALE_CLAIM_SECONDS:
            return True
        return host == socket.gethostname() and not _alive(pid)

    def refinement(self, fingerprint: str, descriptor: str) -> dict[str, object] | None:
        """Return one simulation's stage-2 refinement, or ``None``.

        Kept under ``problems/<fingerprint>/<descriptor>.json``; an unreadable file is
        skipped with one :class:`StoreWarning`.
        """
        path = self.root / "problems" / fingerprint / f"{descriptor}.json"
        try:
            found = _mapping(json.loads(path.read_text(encoding="utf-8")), "refinement")
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            _warn(f"ignoring the refinement {path}: {error}")
            return None
        if found.get("descriptor") != descriptor:
            _warn(f"ignoring the refinement {path}: it names another descriptor")
            return None
        return found

    def publish_refinement(
        self, fingerprint: str, descriptor: str, record: Mapping[str, object]
    ) -> Path:
        """Write one simulation's refinement whole; return its file.

        Raises
        ------
        OSError
            When the store cannot be written.
        """
        directory = self.root / "problems" / fingerprint
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{descriptor}.json"
        _write_atomically(
            path, json.dumps(dict(record) | {"descriptor": descriptor}, indent=1) + "\n"
        )
        return path

    def inconclusive_since(self, fingerprint: str, allocation: str) -> float | None:
        """Return when this class's last calibration was inconclusive, or ``None``."""
        try:
            attempts = json.loads(
                (self.directory(fingerprint) / "attempts.json").read_text(
                    encoding="utf-8"
                )
            )
            value = attempts.get(allocation)
            return float(value) if value is not None else None
        except OSError, ValueError, AttributeError, TypeError:
            return None

    def record_inconclusive(
        self, fingerprint: str, allocation: str, when: float | None = None
    ) -> None:
        """Note an inconclusive calibration of one class; nothing is published."""
        directory = self.directory(fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "attempts.json"
        try:
            attempts = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(attempts, dict):
                attempts = {}
        except OSError, ValueError:
            attempts = {}
        attempts = cast(dict[str, object], attempts)
        attempts[allocation] = time.time() if when is None else when
        _write_atomically(path, json.dumps(attempts) + "\n")


def lookup(
    fingerprint: str,
    stores: Sequence[Store],
) -> Generation | None:
    """Return the first published generation of `fingerprint` among `stores`.

    The user's store comes first, then each read-only directory; one with no
    generation, or an unreadable one, is passed over.
    """
    for store in stores:
        generation = store.current(fingerprint)
        if generation is not None:
            return generation
    return None


def refinement(
    fingerprint: str, descriptor: str, stores: Sequence[Store]
) -> dict[str, object] | None:
    """Return the first refinement of `descriptor` among `stores`, or ``None``."""
    for store in stores:
        found = store.refinement(fingerprint, descriptor)
        if found is not None:
            return found
    return None


def generation_name() -> str:
    """Return a new generation's name: its UTC time, process and a random suffix."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"g{stamp}-{os.getpid()}-{secrets.token_hex(3)}"


def _write_atomically(path: Path, text: str) -> None:
    """Write `text` beside `path`, then replace `path` with it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _alive(pid: int) -> bool:
    """Whether process `pid` exists on this host."""
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(
            0x1000, False, pid
        )  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_WARNED: set[str] = set()


def _warn(message: str) -> None:
    """Issue `message` once per process as a :class:`StoreWarning`."""
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(message, StoreWarning, stacklevel=3)


def stores(environ: Mapping[str, str] = os.environ) -> tuple[Store, ...]:
    """Return the user's store, then each read-only directory's."""
    return (
        Store(cache_directory(environ)),
        *(Store(root) for root in search_path(environ)),
    )
