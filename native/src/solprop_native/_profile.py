# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which calibration a process applies, and what of it (solprop SEP 46 section 6.4).

At import the native package finds this machine's profile (:func:`choose`): the
machine and build's fingerprint, the published generation in the user's store or a
read-only directory, and the row for the process's allocation class. An exact row
supplies *process settings* (the pool width and how idle threads wait), which the
import applies before the runtime starts, unless the user set them. Each workspace
then takes the row's *workspace settings* over the built-in table
(:meth:`Choice.workspace`), but only where they were validated under the process
settings actually in force: a row fitted under spinning threads gives way to the
table in a process whose threads sleep.

This module is standard library only and loads nothing of the package's extension.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ._allocation import AllocationClass, Match, classify, match
from ._execution import ExecutionRecord
from ._fingerprint import Fingerprint, build_identity, fingerprint
from ._store import Generation, Mode, Row, Store, StoreWarning, lookup, mode, stores
from ._topology import Topology, read_topology

__all__ = ["Choice", "Process", "choose", "descriptor"]

WAITS = ("spin", "passive")
"""The waiting policies a row may choose (``_wait``'s settings)."""


@dataclass(frozen=True)
class Process:
    """The process settings in force once the runtime started.

    Attributes
    ----------
    threads : int
        The runtime's width.
    wait : str
        ``"spin"`` or ``"passive"`` as the import chose, or ``"environment: ..."``
        for the user's choice (``_wait.WaitPolicy.description``).
    """

    threads: int
    wait: str

    def record(self) -> dict[str, object]:
        """Return the settings as JSON data."""
        return {"threads": self.threads, "wait": self.wait}


@dataclass(frozen=True)
class Choice:
    """The profile a process found, and the row for its allocation.

    Attributes
    ----------
    mode : str
        ``SOLPROP_CALIBRATION``: ``"auto"``, ``"use"`` or ``"off"``.
    fingerprint : Fingerprint or None
        The machine and build's; ``None`` under ``off``.
    allocation : AllocationClass or None
        The process's class; ``None`` under ``off``.
    generation : Generation or None
        The published generation found, or ``None``.
    found : Match or None
        Its row for the allocation, and whether the match is exact.
    note : str
        Why no row applies, when none does.
    """

    mode: Mode
    fingerprint: Fingerprint | None = None
    allocation: AllocationClass | None = None
    generation: Generation | None = None
    found: Match[Row] | None = None
    note: str = ""

    @property
    def row(self) -> Row | None:
        """The row the process takes, or ``None``."""
        return None if self.found is None else self.found.row

    def threads(self) -> int | None:
        """Return the row's pool width, or ``None``; only an exact row supplies it."""
        if self.found is None or not self.found.exact:
            return None
        value = self.found.row.process.get("threads")
        return value if type(value) is int and value > 0 else None

    def wait(self) -> str | None:
        """Return the row's waiting policy, or ``None``; only an exact row has one."""
        if self.found is None or not self.found.exact:
            return None
        value = self.found.row.process.get("wait")
        return value if isinstance(value, str) and value in WAITS else None

    def workspace(
        self, process: Process, builtin: ExecutionRecord
    ) -> tuple[ExecutionRecord, str]:
        """Return a workspace's record under `process`, and why.

        The row's workspace settings over `builtin`, if they were validated under the
        waiting policy in force and, for an exact row, the width in force too;
        otherwise `builtin`.
        """
        row = self.row
        if row is None:
            return builtin, self.note or "no profile"
        if not row.workspace:
            return builtin, "the row holds no workspace settings"
        fitted = row.fitted_under
        if fitted.get("wait") != process.wait:
            return builtin, (
                f"fitted under {fitted.get('wait')!r} waiting, the process runs "
                f"{process.wait!r}"
            )
        assert self.found is not None
        if self.found.exact and fitted.get("threads") not in (None, process.threads):
            return builtin, (
                f"fitted on {fitted.get('threads')} threads, the process runs "
                f"{process.threads}"
            )
        try:
            return builtin.updated(row.workspace), "profile"
        except ValueError as error:
            _warn(f"ignoring a calibration row's workspace settings: {error}")
            return builtin, f"the row's workspace settings are invalid: {error}"

    def planning(
        self, process: Process, environ: Mapping[str, str] = os.environ
    ) -> dict[str, object] | None:
        """Return a workspace's FFT planning under `process`, or ``None``.

        The row's plan efforts and the generation's wisdom (SEP 46 section 6.7), where
        the row's workspace settings hold (:meth:`workspace`); ``None`` keeps the plan
        policy: no profile, a row without planning, other process settings, or the
        user's own ``SOLPROP_FFTW_PLAN`` or ``SOLPROP_FFTW_WISDOM``, which take
        precedence.
        """
        row = self.row
        if row is None or self.generation is None or not row.planning:
            return None
        if any(
            environ.get(name, "").strip()
            for name in ("SOLPROP_FFTW_PLAN", "SOLPROP_FFTW_WISDOM")
        ):
            return None
        fitted = row.fitted_under
        assert self.found is not None
        if fitted.get("wait") != process.wait or (
            self.found.exact and fitted.get("threads") not in (None, process.threads)
        ):
            return None
        efforts = row.planning.get("efforts", {})
        if not isinstance(efforts, dict):
            return None
        return {
            "mode": "profile",
            "efforts": dict(cast(dict[str, str], efforts)),
            "wisdom": dict(self.generation.wisdom),
        }

    def record(
        self,
        process: Process,
        applied: ExecutionRecord,
        builtin: ExecutionRecord,
        reason: str,
    ) -> dict[str, object]:
        """Return the ``calibration`` metadata of a run that applied `applied`."""
        settings = {
            name: {"applied": value, "built_in": built}
            for (name, value), built in zip(
                applied.native().items(), builtin.native().values(), strict=True
            )
        }
        base: dict[str, object] = {
            "mode": self.mode,
            "process": process.record(),
            "workspace": settings,
            "workspace_source": reason,
        }
        if self.generation is None or self.found is None:
            return base | {"profile": "built-in", "note": self.note or "no profile"}
        row = self.found.row
        return base | {
            "profile": {
                "level": row.level,
                "fingerprint": self.generation.fingerprint,
                "generation": self.generation.name,
                "created": self.generation.created,
                "allocation": dict(row.allocation),
                "exact": self.found.exact,
                "source": str(self.generation.source)
                if self.generation.source is not None
                else None,
            },
        }


def descriptor(
    *,
    shape: Sequence[int],
    grid_kind: str,
    integrator: str,
    sizing: Mapping[str, object],
    process: Process,
    fingerprint: str,
) -> str:
    """Return a simulation's execution descriptor (SEP 46 section 6.6).

    A stage-2 refinement is reused only where the execution is the same: the state's
    shape, the field or envelope grid, the integrator, every operation its workspace
    runs with its work (``execution_sizing``'s keys: the dimensions, the node chunk,
    the response buffers and the transforms' sizes, which set its plans), the process
    settings and the machine and build. Equal descriptors reuse; anything else falls
    back to the machine profile.

    Parameters
    ----------
    shape : sequence of int
        The state's shape.
    grid_kind : str
        ``"field"`` or ``"envelope"``.
    integrator : str
        The integrator's name.
    sizing : Mapping[str, object]
        The workspace's ``execution_sizing``; only its keys are read.
    process : Process
        The process settings.
    fingerprint : str
        The machine and build's store key.

    Returns
    -------
    str
        24 hexadecimal digits.
    """
    text = json.dumps(
        {
            "shape": list(shape),
            "grid_kind": grid_kind,
            "integrator": integrator,
            "operations": sorted(sizing),
            "process": process.record(),
            "fingerprint": fingerprint,
        },
        sort_keys=True,
    )
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def choose(
    environ: Mapping[str, str] = os.environ,
    *,
    packages: Sequence[Path],
    stem: str = "_core",
    cpus: Sequence[int] = (),
    quota: int | None = None,
    topology: Topology | None = None,
    searched: Sequence[Store] | None = None,
) -> Choice:
    """Find this process's profile and row (SEP 46 sections 5 and 6.4).

    Under ``SOLPROP_CALIBRATION=off`` nothing is read. An unreadable store or a
    malformed row leaves the built-in table, with one :class:`StoreWarning`.

    Parameters
    ----------
    environ : Mapping[str, str]
        The environment.
    packages : Sequence[Path]
        The ``solprop_native`` package's directories, for the build's identity.
    stem : str
        The chosen extension's stem.
    cpus : Sequence[int]
        The process's affinity mask; empty for the whole machine.
    quota : int or None
        Its CPU quota.
    topology : Topology or None
        The machine's, read when ``None``.
    searched : Sequence[Store] or None
        The stores to search, the user's and ``SOLPROP_CALIBRATION_PATH``'s when
        ``None``.

    Returns
    -------
    Choice
        What was found.
    """
    chosen = mode(environ)
    if chosen == "off":
        return Choice(mode=chosen, note="SOLPROP_CALIBRATION=off")
    try:
        machine = topology or read_topology()
        allocation = classify(machine, cpus, quota)
        identity = fingerprint(machine, build_identity(packages, stem))
        generation = lookup(
            identity.digest, stores(environ) if searched is None else searched
        )
    except OSError as error:
        _warn(f"cannot read the calibration store: {error}")
        return Choice(mode=chosen, note=f"the store is unreadable: {error}")
    if generation is None:
        return Choice(
            mode=chosen,
            fingerprint=identity,
            allocation=allocation,
            note="no profile for this machine and build",
        )
    rows: list[tuple[AllocationClass, Row]] = []
    for row in generation.rows:
        try:
            rows.append((AllocationClass.read(row.allocation), row))
        except ValueError as error:
            _warn(f"ignoring a calibration row: {error}")
    found = match(allocation, rows)
    return Choice(
        mode=chosen,
        fingerprint=identity,
        allocation=allocation,
        generation=generation,
        found=found,
        note="" if found is not None else "no row for this allocation",
    )


_WARNED: set[str] = set()


def _warn(message: str) -> None:
    """Issue `message` once per process as a :class:`StoreWarning`."""
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(message, StoreWarning, stacklevel=3)
