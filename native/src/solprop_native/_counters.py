# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Count the adapter's returns to Python and host crossings, for benchmarks only.

solprop SEP 47 section 7.3 judges a change by structural counters beside its timings:
why each native call came back to Python, how many attempts it carried, and what
crossed to the host. Counting is off unless the benchmark worker turns it on: every
hook reads :data:`active` once and does nothing while it is ``None``, so a run that is
not counted pays one attribute test per native session ``advance``, export, snapshot
or load, and nothing per attempt. Its numbers are unchanged either way.

Every native run is served by the native execution session
(:class:`~solprop_native.session.NativeSession`), which steps, observes and reads its
edges natively; its keys start ``session_``. What is counted, and where:

- **Native session batches**: every ``advance`` of a native session, each one return
  to Python, by its event -- ``session_observation``, ``session_snapshot`` (a
  requested snapshot serviced), ``session_warning``, ``session_terminal``, or
  ``session_progress`` split by its cause: ``buffer`` (the attempt or row buffer
  filled), ``work_limit`` (the caller's attempt bound) or ``deadline`` (its time
  bound). Each batch's chunk adds its attempts and accepted steps, its periodic rows
  to the statistics rows and, with its plane statistics, to the statistics
  evaluations, and a non-launch observation to the planes; the session's own march
  calls inside the ``advance`` are counted from its status as ``session_adams``,
  ``session_runge_kutta`` or ``session_low_storage``.
- **Fields crossing to the host** with their bytes, by where they cross: ``export``
  (a native session's export of an observation's field, one native pass into the
  destination; a recipe asked again for the same batch is served from the first and
  not counted) and ``snapshot`` (a native session's copy of the Adams state and
  history).
- **Fields loaded into the native state** with their bytes, by where: ``start``
  (``NativeBackend._load``: a run's start or resume, or a timing probe's) and
  ``restore`` (the Adams march's history restored at a resume).

Private: the benchmark harness's hook, with no stability promise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

__all__ = ["Counters", "active", "disable", "enable"]


@dataclass
class Counters:
    """What the adapter returned to Python for and moved to the host while enabled.

    Attributes
    ----------
    march_calls : collections.Counter
        The native session's march calls, by ``session_`` and its march's kind.
    returns : collections.Counter
        Native session batches, by ``session_`` and their event.
    attempts : int
        Attempts the native sessions made, rejected ones included.
    accepted : int
        Accepted steps among them.
    planes : int
        A native session's observations: one per saved plane after the launch.
    statistics_rows : int
        Periodic statistics rows.
    statistics_evaluations : int
        Statistics evaluations, a native session's rows and plane statistics
        together.
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
    attempts: int = 0
    accepted: int = 0
    planes: int = 0
    statistics_rows: int = 0
    statistics_evaluations: int = 0
    field_crossings: Counter[str] = field(default_factory=Counter[str])
    field_bytes: int = 0
    field_loads: Counter[str] = field(default_factory=Counter[str])
    load_bytes: int = 0

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

        - ``event``: its event's kind: ``observation``, ``snapshot``, ``warning``,
          ``terminal``, or ``progress:`` and its cause.
        """
        self.returns[f"session_{event}"] += 1
        self.attempts += attempts
        self.accepted += accepted
        self.statistics_rows += rows
        self.statistics_evaluations += evaluations
        self.planes += plane
        if calls:
            self.march_calls[march] += calls

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

        ``native_returns`` is every return to Python, by the native sessions' batch
        events.
        """
        return {
            "march_calls": dict(sorted(self.march_calls.items())),
            "native_returns": dict(sorted(self.returns.items())),
            "attempts": self.attempts,
            "accepted": self.accepted,
            "planes": self.planes,
            "statistics_rows": self.statistics_rows,
            "statistics_evaluations": self.statistics_evaluations,
            "field_crossings": dict(sorted(self.field_crossings.items())),
            "field_bytes": self.field_bytes,
            "field_loads": dict(sorted(self.field_loads.items())),
            "load_bytes": self.load_bytes,
        }


active: Counters | None = None
"""The counters the hooks add to, or ``None`` when counting is off."""


def enable() -> Counters:
    """Start counting into fresh counters and return them."""
    global active
    active = Counters()
    return active


def disable() -> Counters | None:
    """Stop counting; return the counters that were active."""
    global active
    found, active = active, None
    return found
