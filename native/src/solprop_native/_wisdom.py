# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the FFTW plans the native backend's runs measure (solprop ADR-076).

Measured planning is most of a short run's construction: on an AMD EPYC 7763 a
6400-sample real transform alone takes about 1.3 s to plan, and the threaded rows of a
multimode stage up to 12 s, against a run window of a few seconds. FFTW's wisdom records
the plans a measurement chose, and a later process that imports it plans the same
transforms at once. Wisdom only chooses among FFTW's exact algorithms, so it never
changes what a transform computes; which algorithm was chosen can change the rounding,
as a new measurement can.

The plans are kept by transform, not as one blob of wisdom: each plan key (the
transform's kind, sizes, strides, placement, alignment and threads, as a calibration
keys its winners) holds the wisdom of that transform alone, measured in place by the
run that first planned it. A workspace is offered the plans it may use; a transform the
store holds plans from its own wisdom only, and one it lacks is measured alone and
added (``FftPlanning`` in ``fft_plan.hpp``). A calibration profile's winners keep
precedence: a key the profile holds is never read from or written to this store.

By default the file is ``wisdom.json`` in the calibration store's directory of this
machine and build (``<store>/machines/<fingerprint>/``), so different machines and
builds never share plans and ``solprop calibrate --clear`` removes them with the
profile. ``SOLPROP_FFTW_WISDOM`` names another file, beside which this build's is kept
as ``<name>.native-<fingerprint>``; ``SOLPROP_CALIBRATION=off`` keeps no default file.
Every process also keeps the plans it measured in memory, for its later workspaces.

Many processes of a scan may share one file. A reader never takes a lock: the file is
only ever replaced whole, by an atomic rename. A writer takes an exclusive lock on
``<file>.lock`` (on POSIX systems), merges the file's current plans with its own, and
replaces the file only when that adds a plan, so a run whose plans were all known
writes nothing. A file that cannot be named, read or written costs only the
measurement: it is reported in the log and the run goes on.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import threading
from collections.abc import Collection, Generator, Mapping
from pathlib import Path
from typing import cast

from . import _core
from ._store import cache_directory, mode

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: the atomic replace alone applies
    fcntl = None

__all__ = ["FILE", "FORMAT", "digest", "plans", "remember", "store", "wisdom_file"]

logger = logging.getLogger(__name__)

ENVIRONMENT = "SOLPROP_FFTW_WISDOM"
"""The variable naming another wisdom file."""

FILE = "wisdom.json"
"""The default file's name, in the store's directory of this machine and build."""

FORMAT = 1
"""The file's format; a file of another format is not read."""

_LOCK = threading.Lock()
# Per file: the (mtime, size) of the version this process read last, and its plans.
_read: dict[Path, tuple[tuple[int, int], dict[str, str]]] = {}
# The plans this process measured, for its later workspaces.
_measured: dict[str, str] = {}


@functools.cache
def _keeps_wisdom() -> bool:
    """Whether the build keeps FFTW wisdom: not a device build or a provider without."""
    return _core.export_fftw_wisdom() is not None


@functools.cache
def machine_key() -> str:
    """Return this machine and build's fingerprint digest, the profile's (SEP 46 §5)."""
    import solprop_native as package

    found = package.PROFILE.fingerprint
    if found is not None:
        return found.digest
    # SOLPROP_CALIBRATION=off reads no profile, and so took no fingerprint.
    from ._fingerprint import build_identity, fingerprint
    from ._topology import read_topology

    return fingerprint(
        read_topology(),
        build_identity(
            [Path(entry) for entry in package.__path__],
            package._STEM,  # pyright: ignore[reportPrivateUsage] - same package
        ),
    ).digest


def wisdom_file(environ: Mapping[str, str] = os.environ) -> Path | None:
    """Return this machine and build's plan file, or ``None`` without one.

    - Returns: ``<name>.native-<fingerprint>`` beside the file ``SOLPROP_FFTW_WISDOM``
      names; otherwise ``wisdom.json`` in the calibration store's directory of this
      machine and build. ``None`` under ``SOLPROP_CALIBRATION=off`` without the
      variable, on a build that keeps no FFTW wisdom (a device build, or a provider
      without it), and, with a logged warning, when the name cannot be used.
    """
    named = environ.get(ENVIRONMENT, "").strip()
    try:
        if not named and mode(environ) == "off":
            return None
        if not _keeps_wisdom():
            return None
        key = machine_key()
        if named:
            base = Path(named).expanduser()
            if not base.name or base.name in {".", ".."}:
                raise ValueError(f"{named!r} names no file")
            return base.with_name(f"{base.name}.native-{key}")
        return cache_directory(environ) / "machines" / key / FILE
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(
            "solprop_native: cannot keep FFTW plans (%s); planning without them", error
        )
        return None


def _version(path: Path) -> tuple[int, int] | None:
    """Return the file's (mtime, size), or ``None`` when it does not exist."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _parse(path: Path, text: str) -> dict[str, str]:
    """Return the plans of a file's `text`; report and refuse anything else."""
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not a mapping")
        values = cast(dict[str, object], data)
        if values.get("format") != FORMAT:
            raise ValueError(f"format {values.get('format')!r}, expected {FORMAT}")
        found = values.get("plans")
        if not isinstance(found, dict) or not all(
            isinstance(key, str) and isinstance(plan, str)
            for key, plan in cast(dict[object, object], found).items()
        ):
            raise ValueError("its plans are not text by key")
    except ValueError as error:
        logger.warning("solprop_native: ignoring the FFTW plans in %s: %s", path, error)
        return {}
    return dict(cast(dict[str, str], found))


def _load(path: Path) -> dict[str, str]:
    """Return the file's plans, read again only when it changed; hold `_LOCK`."""
    try:
        version = _version(path)
        if version is None:
            return {}
        cached = _read.get(path)
        if cached is not None and cached[0] == version:
            return cached[1]
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("solprop_native: cannot read FFTW plans: %s", error)
        return {}
    found = _parse(path, text)
    _read[path] = (version, found)
    return found


def plans(path: Path | None) -> tuple[dict[str, str], bool]:
    """Return the plans a workspace is offered, and whether the file supplied any.

    - ``path``: this build's plan file (:func:`wisdom_file`); ``None`` offers this
      process's own plans alone.
    """
    with _LOCK:
        found = _load(path) if path is not None else {}
        return found | _measured, bool(found)


def remember(measured: Mapping[str, str]) -> None:
    """Keep plans this process measured for its later workspaces."""
    with _LOCK:
        _measured.update(measured)


@contextlib.contextmanager
def _exclusive(path: Path) -> Generator[None]:
    """Hold the file's writer lock, beside it, for one merge and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f"{path.name}.lock").open("a", encoding="ascii") as lock:
        if fcntl is not None:  # pragma: no branch - Windows has no fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield
        # Closing the file releases the lock.


def _replace(path: Path, text: str) -> None:
    """Write `text` beside `path`, then rename it over `path` in one step."""
    partial = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}"
    try:
        partial.write_text(text, encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def store(
    path: Path | None, measured: Mapping[str, str], profiled: Collection[str] = ()
) -> bool:
    """Merge plans a run measured into the file.

    - ``path``: this build's plan file (:func:`wisdom_file`); ``None`` does nothing.
    """
    new = {key: plan for key, plan in measured.items() if key not in profiled}
    if path is None or not new:
        return False
    with _LOCK:
        try:
            with _exclusive(path):
                version = _version(path)
                current = (
                    _parse(path, path.read_text(encoding="utf-8"))
                    if version is not None
                    else {}
                )
                merged = current | new
                changed = merged != current
                if changed:
                    record = {"format": FORMAT, "plans": merged}
                    _replace(path, json.dumps(record, indent=0, sort_keys=True))
                    version = _version(path)
        except (OSError, UnicodeDecodeError) as error:
            logger.warning("solprop_native: cannot store FFTW plans: %s", error)
            return False
        if version is not None:
            _read[path] = (version, merged)
        return changed


def digest(applied: Mapping[str, str]) -> str | None:
    """Return a short digest of a workspace's plans, for its run's metadata.

    - Returns: the first sixteen hex characters of a BLAKE2b over the plans; ``None`` on
      a build without FFTW wisdom.
    """
    if not _keeps_wisdom():
        return None
    text = json.dumps(dict(applied), sort_keys=True)
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
