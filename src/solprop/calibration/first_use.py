# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calibrating automatically on first use (SEP 46 sections 5 and 6.10).

Under ``SOLPROP_CALIBRATION=auto``, the default, the first native workspace built in
an allocation with no row in its machine's profile first runs the quick calibration,
about 20 s, after a :class:`~solprop.exceptions.CalibrationNotice`. A process that
had not loaded the native extension yet then takes the whole new profile when it does;
one that had applies the new workspace settings that hold under the process settings
it started with, and the rest from its next process. Importing the native backend
never calibrates.

A scan calibrates its workers' allocation at the standard level before it starts
them (:func:`for_workers`), unless a standard or thorough row already matches; a
worker never calibrates. Concurrent first runs on one host wait for the one that holds
the calibration claim, up to its deadline, then use its profile. A busy machine's
inconclusive calibration publishes nothing, and ``auto`` tries again after a day.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from solprop import _process
from solprop.calibration import _native
from solprop.exceptions import BackendError, CalibrationNotice

if TYPE_CHECKING:
    from solprop_native._allocation import (  # pyright: ignore[reportMissingImports]
        AllocationClass,
    )
    from solprop_native._profile import (  # pyright: ignore[reportMissingImports]
        Choice,
    )
    from solprop_native._store import (  # pyright: ignore[reportMissingImports]
        Store,
    )

    from solprop.calibration.controller import CalibrationReport

__all__ = ["RETRY_SECONDS", "ROLE", "ensure", "for_workers"]

logger = logging.getLogger(__name__)

ROLE = "SOLPROP_CALIBRATION_ROLE"
"""Set to ``worker`` in a scan's workers, which never calibrate."""

RETRY_SECONDS = 86400.0
"""How long ``auto`` waits after an inconclusive calibration before trying again."""

_DONE: set[tuple[str, ...]] = set()


def _notice(level: str, seconds: float, store: object, cores: int) -> None:
    """Issue the first-use notice as a warning and a log record.

    Parameters
    ----------
    level : str
        The level about to run.
    seconds : float
        Its budget.
    store : object
        The store's directory.
    cores : int
        The allocation's cores.
    """
    message = (
        f"solprop is calibrating the native backend to this machine ({level}, about "
        f"{seconds:.0f} s on {cores} cores; once per machine, allocation and release). "
        f"The profile is kept in {store}. Run `solprop calibrate` for a fuller "
        "calibration, or set SOLPROP_CALIBRATION=use to never measure, or =off to "
        "ignore profiles."
    )
    logger.info(message)
    warnings.warn(message, CalibrationNotice, stacklevel=4)


def _device_build() -> bool:
    """Return whether the installed native build runs on a device (phase 6's).

    Returns
    -------
    bool
        Whether its build record names a device backend.
    """
    import json  # noqa: PLC0415

    for directory in _native.packages():
        try:
            record = json.loads((directory / "_build.json").read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue
        return isinstance(record, dict) and record.get("device_backend") not in (
            None,
            "none",
        )
    return False


def _wait_for(store: Store, digest: str, key: str) -> bool:
    """Wait for another process on this host calibrating this class; say if one was.

    Parameters
    ----------
    store : solprop_native._store.Store
        The user's store.
    digest : str
        The machine and build's key.
    key : str
        The allocation class's key.

    Returns
    -------
    bool
        Whether a claim was held, here or elsewhere: then this process does not
        calibrate.
    """
    import socket  # noqa: PLC0415

    holder = store.holder(digest, key)
    if holder is None:
        return False
    if holder.host != socket.gethostname():
        return True
    while store.holder(digest, key) is not None:
        time.sleep(0.25)
    return True


def _allowed(environ: Mapping[str, str]) -> bool:
    """Return whether this process may measure implicitly.

    Parameters
    ----------
    environ : Mapping[str, str]
        The environment.

    Returns
    -------
    bool
        ``SOLPROP_CALIBRATION=auto``, not a scan's worker, and a CPU build.
    """
    return (
        _native.store.mode(environ) == "auto"
        and environ.get(ROLE) != "worker"
        and not _device_build()
    )


def _covered(found: Choice, allocation: AllocationClass, need: str) -> bool:
    """Return whether the profile already has the row an allocation needs.

    Parameters
    ----------
    found : solprop_native._profile.Choice
        What the stores hold for this machine and build.
    allocation : solprop_native._allocation.AllocationClass
        The allocation's class.
    need : str
        ``"any"``: any row it takes, its own or a nearer one's; ``"standard"``: its own
        at the standard or thorough level.

    Returns
    -------
    bool
        Whether nothing needs measuring.
    """
    from solprop.calibration.controller import ORDER  # noqa: PLC0415

    if need == "any" and found.found is not None:
        return True
    for row in () if found.generation is None else found.generation.rows:
        try:
            same = _native.allocation.AllocationClass.read(row.allocation) == allocation
        except ValueError:
            continue
        if same and (need == "any" or ORDER.get(row.level, -1) >= ORDER["standard"]):
            return True
    return False


def _calibrate(
    level: str, cpus: Sequence[int], environ: Mapping[str, str], *, need: str
) -> CalibrationReport | None:
    """Calibrate `cpus`' allocation at `level` unless a row it needs exists.

    Parameters
    ----------
    level : str
        ``"quick"`` or ``"standard"``.
    cpus : sequence of int
        The allocation's CPUs.
    environ : Mapping[str, str]
        The environment.
    need : str
        ``"any"`` or ``"standard"`` (:func:`_covered`).

    Returns
    -------
    CalibrationReport or None
        The report, or ``None`` when nothing was measured.
    """
    if len(cpus) < 2:
        # One CPU has no team, waiting or grain to choose: a single-threaded scan's
        # workers run on the built-in table without measuring (SEP 46 Part II).
        return None
    try:
        _native.packages()
        _native.store.mode(environ)
    except BackendError, ImportError:
        # No native backend, or one without the calibration's modules: selecting it
        # fails, or runs on its own table, on its own terms.
        return None
    if not _allowed(environ):
        return None
    from solprop import calibration  # noqa: PLC0415 - imports this module's package
    from solprop.calibration.controller import LEVELS, calibrate  # noqa: PLC0415

    machine = _native.topology.read_topology()
    allocation = _native.allocation.classify(machine, cpus, _process.cgroup_cpus())
    found = calibration.choice(environ)
    assert found.fingerprint is not None  # "auto" always fingerprints
    digest = found.fingerprint.digest
    store = _native.store.Store(calibration.store_directory(environ))
    key = (digest, allocation.key(), level)
    try:
        since = store.inconclusive_since(digest, allocation.key())
        if (
            key in _DONE
            or _covered(found, allocation, need)
            or (since is not None and time.time() - since < RETRY_SECONDS)
            or _wait_for(store, digest, allocation.key())
        ):
            return None
        _DONE.add(key)
        _notice(level, LEVELS[level].budget, store.root, allocation.cores)
        return calibrate(level, environ=environ, cpus=cpus)
    except OSError as error:
        # The calibration is optional: a store that cannot be written (read-only,
        # missing, full) leaves runs on the built-in table, with one warning.
        _DONE.add(key)
        message = (
            f"solprop could not use its calibration store at {store.root} ({error}); "
            "the native backend runs on its built-in settings"
        )
        logger.warning(message)
        warnings.warn(message, CalibrationNotice, stacklevel=4)
        return None


def ensure(environ: Mapping[str, str] | None = None) -> CalibrationReport | None:
    """Calibrate quickly before a first native workspace, if this allocation needs it.

    A no-op unless ``SOLPROP_CALIBRATION=auto``, in a process that is not a scan's
    worker, on a CPU build, with no row for its allocation, no inconclusive attempt in
    the last day and nobody else calibrating it; once per process. A process that has
    already loaded the native extension then refreshes its profile.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    solprop.calibration.controller.CalibrationReport or None
        The quick calibration's report, or ``None`` when none ran.
    """
    environ = os.environ if environ is None else environ
    report = _calibrate("quick", _process.allowed_cpus(), environ, need="any")
    if report is not None and "solprop_native" in sys.modules:
        sys.modules["solprop_native"].refresh_profile()
    return report


def for_workers(
    cpus: Sequence[int], environ: Mapping[str, str] | None = None
) -> CalibrationReport | None:
    """Calibrate a scan's workers' allocation at the standard level, if it needs it.

    Run by the scan's parent before its workers start, in children pinned like one
    worker; the other partitions stay idle meanwhile, so caches and memory bandwidth
    are less loaded than in the scan, which the row's evidence records.

    Parameters
    ----------
    cpus : sequence of int
        One worker's CPUs.
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    solprop.calibration.controller.CalibrationReport or None
        The report, or ``None`` when a standard or thorough row already matches or
        nothing may be measured.
    """
    environ = os.environ if environ is None else environ
    return _calibrate("standard", cpus, environ, need="standard")
