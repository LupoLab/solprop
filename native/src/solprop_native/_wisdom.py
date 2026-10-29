# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the native backend's FFTW wisdom in the file ``SOLPROP_FFTW_WISDOM`` names.

Measured planning is most of a short one-thread run's construction: on an AMD EPYC 7763
a 6400-sample real transform alone takes about 1.3 s to plan, against a run window of
one or two seconds. FFTW's wisdom records the plans a measurement chose, and a later
process that imports it plans the same transforms at once. Wisdom only chooses among
FFTW's exact algorithms, so it never changes what a transform computes; which algorithm
was chosen can change the rounding, as a new measurement can.

The variable is the one ``numba_cpu`` reads (``solprop.backends.numba_cpu.fft``), and
nothing is read or written without it. The native backend links its own FFTW, whose
wisdom pyFFTW's cannot read, so it keeps a file of its own beside the named one,
``<name>.native-<key>``. The key digests the FFTW library's wisdom preamble (its version
and the signature of its planner's configuration) and the processor's model, so
machines of different kinds sharing a home directory keep separate plans, and a file
from another FFTW is never offered to this one; FFTW refuses foreign wisdom by its
signature in any case.

Many processes of a scan may share one file. A reader never takes a lock: the file is
only ever replaced whole, by an atomic rename. A writer takes an exclusive lock on
``<file>.lock`` (on POSIX systems), merges the file's current wisdom into its own, and
replaces the file only when that adds something, so a run whose plans were all known
writes nothing. A file that cannot be read or written costs only the measurement: it is
reported in the log and the run goes on.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import os
import platform
import threading
from collections.abc import Generator
from pathlib import Path

from . import _core

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: the atomic replace alone applies
    fcntl = None

__all__ = ["digest", "load", "store", "wisdom_file"]

logger = logging.getLogger(__name__)

ENVIRONMENT = "SOLPROP_FFTW_WISDOM"
"""The variable naming the wisdom file, shared with ``numba_cpu``."""

_LOCK = threading.Lock()
# Per wisdom file: the (mtime, size) of the version this process imported last, and the
# wisdom the process held right after it last loaded or stored that file.
_imported: dict[Path, tuple[int, int] | None] = {}
_known: dict[Path, str] = {}


def _processor(text: str | None = None) -> str:
    """Name the processor model from `text`, by default Linux's ``/proc/cpuinfo``."""
    if text is None:
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - not Linux
            return platform.processor()
    wanted = ("model name", "cpu model", "CPU implementer", "CPU part")
    fields = [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()[:64]
        if ":" in line and line.split(":", 1)[0].strip() in wanted
    ]
    return " ".join(dict.fromkeys(fields)) or platform.processor()


@functools.cache
def _key() -> str | None:
    """Digest what decides whether stored plans suit this process, once.

    ``None`` when the build keeps no FFTW wisdom.
    """
    wisdom = _core.export_fftw_wisdom()
    if wisdom is None:
        return None
    text = "\n".join((wisdom.split("\n", 1)[0], platform.machine(), _processor()))
    return hashlib.blake2b(text.encode(), digest_size=6).hexdigest()


def wisdom_file(named: str | None = None) -> Path | None:
    """Return this build's wisdom file for the named one, or ``None`` without one.

    Parameters
    ----------
    named : str, optional
        The file ``SOLPROP_FFTW_WISDOM`` names; ``None`` reads the variable.

    Returns
    -------
    pathlib.Path or None
        ``<name>.native-<key>`` beside the named file; ``None`` when nothing is named or
        the build keeps no FFTW wisdom (a device build, or a provider without it).
    """
    named = os.environ.get(ENVIRONMENT) if named is None else named
    if not named:
        return None
    key = _key()
    if key is None:
        return None
    base = Path(named).expanduser()
    return base.with_name(f"{base.name}.native-{key}")


def _version(path: Path) -> tuple[int, int] | None:
    """Return the file's (mtime, size), or ``None`` when it does not exist."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _import(path: Path, text: str) -> bool:
    """Merge `text` into FFTW's wisdom; report and refuse what FFTW refuses."""
    if _core.import_fftw_wisdom(text):
        return True
    logger.warning("solprop_native: %s is not this FFTW's wisdom; ignoring it", path)
    return False


def load(path: Path | None) -> bool:
    """Import the wisdom file before planning, unless this process already holds it.

    Parameters
    ----------
    path : pathlib.Path or None
        This build's wisdom file (:func:`wisdom_file`); ``None`` does nothing.

    Returns
    -------
    bool
        Whether the file's wisdom is in force.
    """
    if path is None:
        return False
    with _LOCK:
        try:
            version = _version(path)
            if version is None:
                return False
            if _imported.get(path) == version:
                return True
            text = path.read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError) as error:
            logger.warning("solprop_native: cannot read FFTW wisdom: %s", error)
            return False
        if not _import(path, text):
            return False
        _imported[path] = version
        _known[path] = _core.export_fftw_wisdom() or ""
        return True


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
        partial.write_text(text, encoding="ascii")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def store(path: Path | None) -> bool:
    """Merge the plans this process measured into the wisdom file.

    Parameters
    ----------
    path : pathlib.Path or None
        This build's wisdom file (:func:`wisdom_file`); ``None`` does nothing.

    Returns
    -------
    bool
        Whether the file was replaced.
    """
    if path is None:
        return False
    with _LOCK:
        wisdom = _core.export_fftw_wisdom() or ""
        if wisdom == _known.get(path):
            return False
        try:
            with _exclusive(path):
                stored = path.read_text(encoding="ascii") if path.exists() else ""
                if stored:
                    _import(path, stored)
                merged = _core.export_fftw_wisdom() or ""
                changed = merged != stored
                if changed:
                    _replace(path, merged)
                version = _version(path)
        except (OSError, UnicodeDecodeError) as error:
            logger.warning("solprop_native: cannot store FFTW wisdom: %s", error)
            return False
        _known[path] = merged
        _imported[path] = version
        return changed


def digest() -> str | None:
    """Return a short digest of the wisdom in force, for the run's metadata.

    Returns
    -------
    str or None
        The first sixteen hex characters of a BLAKE2b over FFTW's exported wisdom, as
        ``numba_cpu`` records; ``None`` on a build without FFTW wisdom.
    """
    wisdom = _core.export_fftw_wisdom()
    if wisdom is None:
        return None
    return hashlib.blake2b(wisdom.encode(), digest_size=8).hexdigest()
