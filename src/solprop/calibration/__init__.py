# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calibrating the native backend to its machine (SEP 46).

The native backend's CPU execution settings (how each operation threads, how idle
threads wait, how wide the team is) are fitted to the machine by a bounded, one-off
measurement, kept in a per-user store in the platform's cache directory and applied
by every later run, which records the settings it used in its metadata
(``calibration``). Without a profile the native backend runs its built-in table.

This package reads, shows, exports, imports and clears profiles without loading the
native extension. The manual page is ``docs/guide/performance/calibration.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from solprop import _process
from solprop.calibration import _native
from solprop.exceptions import CalibrationError

if TYPE_CHECKING:
    from solprop_native._profile import (  # pyright: ignore[reportMissingImports]
        Choice,
    )

    from solprop.calibration.controller import CalibrationReport

__all__ = [
    "MachineProfile",
    "calibrate",
    "clear",
    "export_profile",
    "import_profile",
    "profile",
    "store_directory",
]


@dataclass(frozen=True)
class MachineProfile:
    """The calibration this process would apply, and its evidence.

    Attributes
    ----------
    fingerprint : str
        The machine and build's store key.
    generation : str
        The published generation's name.
    created : str
        When it was published, ISO 8601 in UTC.
    level : str
        The level that measured the row: ``"quick"``, ``"standard"`` or
        ``"thorough"``.
    allocation : Mapping[str, object]
        The allocation class the row was measured in.
    exact : bool
        Whether that is this process's own class. A row of the nearest smaller class
        supplies workspace settings only.
    process : Mapping[str, object]
        The row's process settings (pool width, waiting policy).
    workspace : Mapping[str, object]
        The row's workspace settings, each a field of the native execution record.
    fitted_under : Mapping[str, object]
        The process settings the workspace settings were validated under.
    evidence : Mapping[str, object]
        What was measured, and why each setting was kept.
    source : pathlib.Path or None
        The generation file read.
    """

    fingerprint: str
    generation: str
    created: str
    level: str
    allocation: Mapping[str, object]
    exact: bool
    process: Mapping[str, object]
    workspace: Mapping[str, object]
    fitted_under: Mapping[str, object]
    evidence: Mapping[str, object]
    source: Path | None


def calibrate(
    level: str = "standard", budget: float | None = None
) -> CalibrationReport:
    """Calibrate this machine's native backend for the process's allocation.

    Child processes measure the candidate settings under one deadline; the result is
    published in the user's store only if it was validated (SEP 46 section 6.5), and
    later runs apply it. ``solprop calibrate`` is the same from the command line.

    Parameters
    ----------
    level : str, optional
        ``"quick"`` (about 30 s), ``"standard"`` (about two minutes) or
        ``"thorough"`` (about seven minutes).
    budget : float or None, optional
        The deadline in seconds, the level's when ``None``.

    Returns
    -------
    solprop.calibration.controller.CalibrationReport
        Each setting chosen beside its built-in value, the evidence, and whether it
        was published.
    """
    from solprop.calibration.controller import calibrate as run  # noqa: PLC0415

    return run(level, budget)


def store_directory(environ: Mapping[str, str] | None = None) -> Path:
    r"""Return the user's calibration store: the platform's user cache directory.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    pathlib.Path
        ``SOLPROP_CACHE_DIR`` when set; otherwise ``$XDG_CACHE_HOME/solprop`` or
        ``~/.cache/solprop`` on Linux, ``~/Library/Caches/solprop`` on macOS and
        ``%LOCALAPPDATA%\solprop\Cache`` on Windows.

    Examples
    --------
    >>> store_directory({"SOLPROP_CACHE_DIR": "/tmp/solprop"}).as_posix()
    '/tmp/solprop'
    """
    return _native.store.cache_directory(os.environ if environ is None else environ)


def choice(environ: Mapping[str, str] | None = None) -> Choice:
    """Return what this process's native import would find in the stores.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    solprop_native._profile.Choice
        The fingerprint, the allocation class, the generation and its row.
    """
    return _native.profile.choose(
        os.environ if environ is None else environ,
        packages=_native.packages(),
        stem=_native.variant(),
        cpus=_process.allowed_cpus(),
        quota=_process.cgroup_cpus(),
    )


def profile(environ: Mapping[str, str] | None = None) -> MachineProfile | None:
    """Return the calibration this process would apply, or ``None``.

    ``None`` means the native backend runs its built-in table: no profile was
    published for this machine and build, none has a row for this allocation, or
    ``SOLPROP_CALIBRATION=off``.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    MachineProfile or None
        The profile, its row and evidence.
    """
    found = choice(environ)
    if found.generation is None or found.found is None:
        return None
    row = found.found.row
    return MachineProfile(
        fingerprint=found.generation.fingerprint,
        generation=found.generation.name,
        created=found.generation.created,
        level=row.level,
        allocation=row.allocation,
        exact=found.found.exact,
        process=row.process,
        workspace=row.workspace,
        fitted_under=row.fitted_under,
        evidence=row.evidence,
        source=found.generation.source,
    )


def export_profile(path: Path, environ: Mapping[str, str] | None = None) -> Path:
    """Write this machine and build's published generation to `path`.

    The file can be imported elsewhere (:func:`import_profile`) or placed in a
    directory listed in ``SOLPROP_CALIBRATION_PATH``, for example a cluster's
    profiles for each node type.

    Parameters
    ----------
    path : pathlib.Path
        The JSON file to write.
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    pathlib.Path
        The generation file that was exported.

    Raises
    ------
    CalibrationError
        When no generation was published for this machine and build.
    """
    found = choice(environ)
    if found.generation is None or found.generation.source is None:
        raise CalibrationError(
            "no calibration profile for this machine and build to export; run "
            "`solprop calibrate` first"
        )
    path.write_text(
        json.dumps(found.generation.record(), indent=1) + "\n", encoding="utf-8"
    )
    return found.generation.source


def import_profile(path: Path, environ: Mapping[str, str] | None = None) -> bool:
    """Publish an exported generation in the user's store.

    Parameters
    ----------
    path : pathlib.Path
        A file written by :func:`export_profile`.
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    bool
        Whether the generation is for this machine and build, so that it applies
        here; another fingerprint's is kept for the store's other machines.

    Raises
    ------
    CalibrationError
        When the file is not a generation of this format.
    """
    store = _native.store
    try:
        generation = store.Generation.read(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        raise CalibrationError(f"cannot import {path}: {error}") from error
    store.Store(store_directory(environ)).publish(generation)
    found = choice(environ)
    return found.fingerprint is not None and (
        found.fingerprint.digest == generation.fingerprint
    )


def clear(environ: Mapping[str, str] | None = None) -> int:
    """Forget this machine and build's profiles and stored FFT plans.

    The user's store loses the profiles, their refinements and the plans runs measured
    (``wisdom.json``, solprop ADR-076), and the file ``SOLPROP_FFTW_WISDOM`` keeps for
    this build, when the variable is set, goes too. A calibration in progress keeps
    its claim. Read-only directories are not touched.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    int
        The files removed.
    """
    found = choice(environ)
    removed = 0
    if found.fingerprint is not None:
        digest = found.fingerprint.digest
        removed = _native.store.Store(store_directory(environ)).clear(digest)
    named = (os.environ if environ is None else environ).get("SOLPROP_FFTW_WISDOM", "")
    if not named.strip():
        return removed
    try:
        if found.fingerprint is None:
            # SOLPROP_CALIBRATION=off reads no profile, but runs still keep the named
            # file, under the same fingerprint (solprop_native._wisdom.machine_key).
            build = _native.fingerprint.build_identity(
                _native.packages(), _native.variant()
            )
            digest = _native.fingerprint.fingerprint(
                _native.topology.read_topology(), build
            ).digest
        base = Path(named.strip()).expanduser()
        base.with_name(f"{base.name}.native-{digest}").unlink()
        removed += 1
    except OSError, ValueError, RuntimeError:
        pass
    return removed
