# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Count the adapter's returns to Python and host crossings, for benchmarks only.

solprop SEP 47 section 7.3 judges a change by structural counters beside its timings:
why each native call came back to Python, how many attempts it carried, and what
crossed to the host. Counting is off unless the benchmark worker turns it on: every
hook reads :data:`active` once and does nothing while it is ``None``. A run that is
not counted pays one attribute test per native march call, plane, statistics call,
edge read, field crossing or load, low-storage or group step and native session
``advance`` or export, and nothing per attempt: the two per-attempt hooks (solprop's
own loop's attempts and periodic rows) are installed on ``NativeBackend`` by
:func:`enable` and removed by :func:`disable`. Its numbers are unchanged either way.

The same record covers both ways a native run is served -- the transitional bridge
(:class:`~solprop.backends.driver_session.DriverSession` over ``NativeBackend`` and its
native marches) and the native execution session
(:class:`~solprop_native.session.NativeSession`) -- so that the two can be compared
key for key. A key keeps one meaning on both; where the two differ in kind, the
session's own key starts ``session_``.

What is counted, and where:

- **Native march calls** made from Python (``adams_march``, ``rk_march``): one per
  call, with the reason it returned (``march::Reason``). A ``host_step`` return is
  split by the driver's work that fell due on its last accepted step: ``statistics``
  (a periodic row), ``step_planes`` (a natural stop, which a snapshot needs on the
  bridge), ``monitor`` (the boundary monitor's check by steps) and ``transverse`` (a
  collar's per-step current, or a read of the transverse edge that fell due by
  distance, ``transverse_read``). Several causes due together are joined with ``+``.
- **Python-stepped attempts**: every attempt the driver's own loop records (a host
  march, a replay, the low-storage march and a lockstep group's members), each one or
  more native calls the counters do not split. The low-storage engine's steps and a
  native group's lockstep steps are also counted as such.
- **Native session batches**: every ``advance`` of a native session, each one return
  to Python, by its event -- ``session_observation``, ``session_snapshot`` (a
  requested snapshot serviced), ``session_warning``, ``session_terminal``, or
  ``session_progress`` split by its cause: ``buffer`` (the attempt or row buffer
  filled), ``work_limit`` (the caller's attempt bound) or ``deadline`` (its time
  bound). Each batch's chunk adds its attempts and accepted steps to the native
  attempts, its periodic rows to the statistics rows and, with its plane statistics,
  to the statistics evaluations, and a non-launch observation to the planes; the
  session's own march calls inside the ``advance`` are counted from its status as
  ``session_adams``, ``session_runge_kutta`` or ``session_low_storage``. Members of
  the native group session (``solprop_native.group``) are native sessions and are
  counted so; the group's common steps are counted among the group steps.
- **Planes** (calls of ``propagate_to`` on the bridge; non-launch observations of a
  session), **statistics rows** (periodic rows, taken by solprop's loop on the bridge
  and natively in a session) and **statistics evaluations** (rows and plane
  statistics together), **transverse edge reads** made through Python (a native
  session reads its edge natively: none), and **fields crossing to the host** with
  their bytes, by where they cross: ``plane`` (the bridge's observation), ``step`` (a
  host Adams engine's copy of each accepted state), ``load`` (a host engine's copy at
  a load), ``export`` (a native session's export of an observation's field, one
  native pass into the destination; a recipe asked again for the same batch is
  served from the first and not counted) and ``snapshot`` (a native session's copy of
  the Adams state and history).
- **Fields loaded into the native state** with their bytes, by where: ``start``
  (``NativeBackend.set_initial``: a run's start or resume, on either path) and
  ``restore`` (the Adams march's reload after a dense observation or at a method
  history's restore). solprop's host Adams engine reloads its masked state at each
  step, which is not counted: its ``step`` crossings mark that route.

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
        Native march calls, by march; a native session's, by ``session_`` and its
        march's kind.
    returns : collections.Counter
        Native march returns, by reason; ``host_step`` by the work due on its step;
        and native session batches, by ``session_`` and their event.
    march_attempts : int
        Attempts the native marches and native sessions made, rejected ones
        included.
    march_accepted : int
        Accepted steps among them.
    host_attempts : int
        Attempts recorded by solprop's own loop, each a return to Python.
    host_accepted : int
        Accepted steps among them.
    low_storage_steps : int
        Low-storage engine steps, among the host attempts.
    group_steps : int
        Lockstep steps of a native group, each covering every member's attempt and
        each a return to Python.
    group_attempts : int
        Member attempts those steps covered, among the host attempts.
    session_group_steps : int
        Common steps of the native group session, which returns per window, not per
        step; the record's ``group_steps`` counts both kinds.
    planes : int
        ``propagate_to`` calls, or a native session's observations: one per saved
        plane after the launch.
    statistics_rows : int
        Periodic statistics rows, wherever taken.
    statistics_evaluations : int
        Statistics evaluations, rows and planes together: ``stats`` calls, or a
        native session's rows and plane statistics.
    transverse_reads : int
        Native reads of the transverse edge made through Python: its faces, or a
        collar's current.
    field_crossings : collections.Counter
        Fields copied to the host, by where they cross.
    field_bytes : int
        Bytes of those fields.
    field_loads : collections.Counter
        Host fields loaded into the native state, by where.
    load_bytes : int
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
    session_group_steps: int = 0
    planes: int = 0
    statistics_rows: int = 0
    statistics_evaluations: int = 0
    transverse_reads: int = 0
    field_crossings: Counter[str] = field(default_factory=Counter[str])
    field_bytes: int = 0
    field_loads: Counter[str] = field(default_factory=Counter[str])
    load_bytes: int = 0

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

    def session_batch(
        self,
        event: str,
        attempts: int,
        accepted: int,
        *,
        rows: int = 0,
        evaluations: int = 0,
        plane: bool = False,
        march: str = "",
        calls: int = 0,
    ) -> None:
        """Count one batch a native session returned from ``advance``.

        Parameters
        ----------
        event : str
            Its event's kind: ``observation``, ``snapshot``, ``warning``,
            ``terminal``, or ``progress:`` and its cause.
        attempts : int
            The attempts its chunk carries.
        accepted : int
            The accepted steps among them.
        rows : int, optional
            The periodic rows its chunk carries.
        evaluations : int, optional
            The statistics evaluations it carries: its rows and its plane statistics.
        plane : bool, optional
            Whether it is an observation after the launch.
        march : str, optional
            The session's march, for its native march calls.
        calls : int, optional
            The native march calls the ``advance`` made.
        """
        self.returns[f"session_{event}"] += 1
        self.march_attempts += attempts
        self.march_accepted += accepted
        self.statistics_rows += rows
        self.statistics_evaluations += evaluations
        self.planes += plane
        if calls:
            self.march_calls[march] += calls

    def group_window(self, steps: int) -> None:
        """Count the common steps of one window of the native group session."""
        self.session_group_steps += steps

    def crossing(self, where: str, nbytes: int) -> None:
        """Count one field copied to the host."""
        self.field_crossings[where] += 1
        self.field_bytes += int(nbytes)

    def load(self, where: str, nbytes: int) -> None:
        """Count one host field loaded into the native state."""
        self.field_loads[where] += 1
        self.load_bytes += int(nbytes)

    def record(self) -> dict[str, object]:
        """Return the counts as a JSON-ready record.

        ``native_returns`` is every return to Python by cause: the marches' reasons
        and the native sessions' batches, then ``host_attempt`` for solprop's own
        loop, less the low-storage steps and group members it also counts, and
        ``low_storage_step`` and ``group_step``.
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
            "group_steps": self.group_steps + self.session_group_steps,
            "planes": self.planes,
            "statistics_rows": self.statistics_rows,
            "statistics_evaluations": self.statistics_evaluations,
            "transverse_reads": self.transverse_reads,
            "field_crossings": dict(sorted(self.field_crossings.items())),
            "field_bytes": self.field_bytes,
            "field_loads": dict(sorted(self.field_loads.items())),
            "load_bytes": self.load_bytes,
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
