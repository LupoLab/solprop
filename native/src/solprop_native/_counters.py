# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Count the adapter's returns to Python and host crossings, for benchmarks only.

solprop SEP 47 section 7.3 judges a change by structural counters beside its timings:
why each native call came back to Python, how many attempts it carried, and what
crossed to the host. Counting is off unless the benchmark worker turns it on: every
hook reads :data:`active` once and does nothing while it is ``None``. A run that is
not counted pays one attribute test per native march call, plane, statistics call,
edge read, field crossing and low-storage or group step, and nothing per attempt:
the two per-attempt hooks (solprop's own loop's attempts and periodic rows) are
installed on ``NativeBackend`` by :func:`enable` and removed by :func:`disable`. Its
numbers are unchanged either way.

What is counted, and where:

- **Native march calls** (``adams_march``, ``rk_march``): one per call, with the
  reason it returned (``march::Reason``). A ``host_step`` return is split by the
  driver's work that fell due on its last accepted step: ``statistics`` (a periodic
  row), ``step_planes`` (a natural stop, which a snapshot needs today), ``monitor``
  (the boundary monitor's check by steps) and ``transverse`` (a collar's per-step
  current, or a read of the transverse edge that fell due by distance,
  ``transverse_read``). Several causes due together are joined with ``+``.
- **Python-stepped attempts**: every attempt the driver's own loop records (a host
  march, a replay, the low-storage march and a lockstep group's members), each one or
  more native calls the counters do not split. The low-storage engine's steps and a
  native group's lockstep steps are also counted as such.
- **Planes** (calls of ``propagate_to``), **statistics rows** (periodic rows) and
  **statistics evaluations** (every ``stats`` call, rows and planes), **transverse
  edge reads**, and **fields crossing to the host** with their bytes, by where they
  cross: ``plane`` (an observation), ``step`` (a host Adams engine's copy of each
  accepted state) and ``load`` (a host engine's copy at a load).

Private: the benchmark harness's hook, with no stability promise. A build that moves
these returns keeps the record's keys and says in its own counting what it removed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from solprop.backends.driver import StepDriver

__all__ = ["Counters", "active", "disable", "enable"]


@dataclass
class Counters:
    """What the adapter returned to Python for and moved to the host while enabled.

    Attributes
    ----------
    march_calls : collections.Counter
        Native march calls, by march.
    returns : collections.Counter
        Native march returns, by reason; ``host_step`` by the work due on its step.
    march_attempts : int
        Attempts the native marches made, rejected ones included.
    march_accepted : int
        Accepted steps among them.
    host_attempts : int
        Attempts recorded by solprop's own loop, each a return to Python.
    host_accepted : int
        Accepted steps among them.
    low_storage_steps : int
        Low-storage engine steps, among the host attempts.
    group_steps : int
        Lockstep steps of a native group, each covering every member's attempt.
    group_attempts : int
        Member attempts those steps covered, among the host attempts.
    planes : int
        ``propagate_to`` calls: one per saved plane after the launch.
    statistics_rows : int
        Periodic statistics rows.
    statistics_evaluations : int
        ``stats`` calls, rows and planes together.
    transverse_reads : int
        Native reads of the transverse edge: its faces, or a collar's current.
    field_crossings : collections.Counter
        Fields copied to the host, by where they cross.
    field_bytes : int
        Bytes of those fields.
    """

    march_calls: Counter[str] = field(default_factory=Counter[str])
    returns: Counter[str] = field(default_factory=Counter[str])
    march_attempts: int = 0
    march_accepted: int = 0
    host_attempts: int = 0
    host_accepted: int = 0
    low_storage_steps: int = 0
    group_steps: int = 0
    group_attempts: int = 0
    planes: int = 0
    statistics_rows: int = 0
    statistics_evaluations: int = 0
    transverse_reads: int = 0
    field_crossings: Counter[str] = field(default_factory=Counter[str])
    field_bytes: int = 0

    def march(
        self,
        kind: str,
        reason: str,
        rows: np.ndarray[Any, Any],
        due: dict[str, int],
        read_at: float | None,
    ) -> None:
        """Count one native march call from its returned rows.

        Parameters
        ----------
        kind : str
            The march, ``adams_march`` or ``rk_march``.
        reason : str
            The call's ``march::Reason``, by name.
        rows : numpy.ndarray
            The call's attempt rows, the accepted flag in column 3.
        due : dict
            Accepted steps from the call's start to each piece of the driver's work,
            by name, as the call's ``host_step`` bound was formed.
        read_at : float or None
            The transverse edge's next read, if the call was bounded by one.
        """
        accepted = int(np.count_nonzero(rows[:, 3])) if rows.size else 0
        self.march_calls[kind] += 1
        self.march_attempts += len(rows)
        self.march_accepted += accepted
        name = reason
        if reason == "host_step":
            causes = sorted(key for key, steps in due.items() if steps == accepted)
            if not causes and read_at is not None:
                causes = ["transverse_read"]
            name += ":" + ("+".join(causes) or "unattributed")
        self.returns[name] += 1

    def group_step(self, members: int) -> None:
        """Count one lockstep step of a native group of `members`."""
        self.group_steps += 1
        self.group_attempts += members

    def crossing(self, where: str, nbytes: int) -> None:
        """Count one field copied to the host."""
        self.field_crossings[where] += 1
        self.field_bytes += int(nbytes)

    def record(self) -> dict[str, object]:
        """Return the counts as a JSON-ready record.

        ``native_returns`` is every return to Python by cause: the marches' reasons,
        then ``host_attempt`` for solprop's own loop, less the low-storage steps and
        group members it also counts, and ``low_storage_step`` and ``group_step``.
        """
        host = self.host_attempts - self.low_storage_steps - self.group_attempts
        loops = {
            "host_attempt": host,
            "low_storage_step": self.low_storage_steps,
            "group_step": self.group_steps,
        }
        returns = dict(sorted(self.returns.items())) | {
            name: count for name, count in loops.items() if count
        }
        return {
            "march_calls": dict(sorted(self.march_calls.items())),
            "native_returns": returns,
            "attempts": self.march_attempts + self.host_attempts,
            "accepted": self.march_accepted + self.host_accepted,
            "march_attempts": self.march_attempts,
            "host_attempts": self.host_attempts,
            "low_storage_steps": self.low_storage_steps,
            "group_steps": self.group_steps,
            "planes": self.planes,
            "statistics_rows": self.statistics_rows,
            "statistics_evaluations": self.statistics_evaluations,
            "transverse_reads": self.transverse_reads,
            "field_crossings": dict(sorted(self.field_crossings.items())),
            "field_bytes": self.field_bytes,
        }


active: Counters | None = None
"""The counters the hooks add to, or ``None`` when counting is off."""


def _record(self: Any, z: float, dz: float, norm: float, accepted: bool) -> None:
    """``StepDriver._record``, counting the attempt: installed while counting."""
    StepDriver._record(self, z, dz, norm, accepted)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType] - the hook replaces it
    if active is not None:
        active.host_attempts += 1
        active.host_accepted += int(accepted)


def _record_step_stats(self: Any, spec: Any, z: float, step: int) -> None:
    """``StepDriver._record_step_stats``, counting the row: installed while counting."""
    StepDriver._record_step_stats(self, spec, z, step)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType] - the hook replaces it
    if active is not None:
        active.statistics_rows += 1


_HOOKS = {"_record": _record, "_record_step_stats": _record_step_stats}
"""The per-attempt hooks :func:`enable` installs on ``NativeBackend``."""


def enable() -> Counters:
    """Start counting into fresh counters and return them."""
    global active
    from .backend import NativeBackend

    for name, hook in _HOOKS.items():
        setattr(NativeBackend, name, hook)
    active = Counters()
    return active


def disable() -> Counters | None:
    """Stop counting and remove the hooks; return the counters that were active."""
    global active
    from .backend import NativeBackend

    for name, hook in _HOOKS.items():
        if NativeBackend.__dict__.get(name) is hook:
            delattr(NativeBackend, name)
    found, active = active, None
    return found
