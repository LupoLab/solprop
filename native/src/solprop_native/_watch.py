# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's boundary monitor's record and verdicts, as solprop reads them.

solprop SEP 47 section 6.4 and the maintainer's decision 4 at its acceptance: the
native session owns the boundary monitor's every decision and record
(``watch::Monitor``, ``native/include/solprop_native/watch.hpp``). The monitor's
configuration and a resume's continuation are solprop's run records
(:mod:`solprop.backends.records`, SEP 52), which :mod:`solprop_native._records`
converts. This module only translates what comes back: it formats the native record
as :meth:`~solprop.monitor.BoundaryMonitor.summary` lays it out, and a native verdict
as solprop's warning or exception. It decides nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from solprop.backends.records import PLANE_COLUMNS, Notice, NoticeKind
from solprop.exceptions import BackendError, BoundaryError
from solprop.monitor import (
    _boundary_warning,  # pyright: ignore[reportPrivateUsage] - one warning
    _guard_message,  # pyright: ignore[reportPrivateUsage] - one text
    _removal_message,  # pyright: ignore[reportPrivateUsage] - one text
)

if TYPE_CHECKING:
    from solprop.monitor import BoundaryCheck

__all__ = [
    "notice_error",
    "notice_of",
    "notice_warning",
    "record_of",
]

_UNMEASURED = (
    "launch_energy",
    "launch_guard_low",
    "launch_guard_high",
    "launch_collar",
    "guard_low_at",
    "guard_high_at",
)
"""The record's fields that read ``None`` where nothing was measured."""


def notice_of(native: Mapping[str, object]) -> Notice:
    """Return the notice ``_core`` reports (``SessionEvent.notice``)."""
    return Notice(
        NoticeKind(int(cast("int", native["kind"]))),
        bool(native["launch"]),
        float(cast("float", native["z"])),
        float(cast("float", native["value"])),
        float(cast("float", native["cap"])),
        float(cast("float", native["step"])),
    )


def record_of(check: BoundaryCheck, native: Mapping[str, object]) -> dict[str, object]:
    """Format the native monitor's record as :meth:`BoundaryMonitor.summary` does.

    - ``native``: ``_core``'s record: every number, NaN where nothing was measured.
    - Returns: the record, with ``None`` for what was never measured.
    """

    def finite(value: object) -> float | None:
        """Return a number, or ``None`` when it was never measured (not finite)."""
        number = float(cast("float", value))
        return number if math.isfinite(number) else None

    record: dict[str, object] = check.thresholds()
    for name in (
        "launch_energy",
        "resumed_at",
        "launch_guard_low",
        "launch_guard_high",
        "launch_collar",
        "guard_low_max",
        "guard_high_max",
        "guard_low_at",
        "guard_high_at",
        "collar_max",
        "removed_fraction",
        "removed_signed",
        "floor_removed_fraction",
        "quiet_checks",
        "collar_quiet",
        "unwatched",
        "steps",
        "skipped",
        "noise_floor",
        "raw_guard_max",
        "raw_removed_fraction",
        "collar_left_out_checks",
    ):
        value = native[name]
        if name in _UNMEASURED:
            value = finite(value)
        elif name == "resumed_at":
            value = [float(one) for one in cast("list[float]", value)]
        elif name == "unwatched":
            value = [
                [float(one) for one in stretch]
                for stretch in cast("list[tuple[float, float]]", value)
            ]
        record[name] = value
    rows = cast("Mapping[str, list[float]]", native["planes"])
    record["planes"] = {
        name: [finite(value) for value in rows[name]] for name in PLANE_COLUMNS
    }
    return record


def notice_warning(check: BoundaryCheck, notice: Notice) -> Warning:
    """Return a native warning, as solprop's monitor issues it."""
    if notice.kind is NoticeKind.GUARD:
        z = None if notice.launch else notice.z
        return _boundary_warning(_guard_message(check, notice.value, z, stop=False))
    if notice.kind is NoticeKind.REMOVAL:
        return _boundary_warning(_removal_message(check, notice.value))
    msg = f"the native session warned with a {notice.kind.name.lower()} notice"
    raise BackendError(msg)


def _zero_division() -> ZeroDivisionError:
    """Return the error solprop's monitor raises dividing by a zero launch energy.

    Raised as the monitor's own float division raises it, so that its message is this
    interpreter's.
    """
    zero = 0.0
    try:
        _ = 1.0 / zero
    except ZeroDivisionError as error:
        return error
    msg = "a float division by zero did not raise"  # pragma: no cover - IEEE Python
    raise AssertionError(msg)


def notice_error(check: BoundaryCheck, notice: Notice) -> BaseException:
    """Return the exception a native monitor's stop reports, as solprop's raises it.

    - Returns: a :class:`~solprop.exceptions.BoundaryError`; a
      :class:`~solprop.exceptions.BackendError` for a reading the split left empty; the
      :class:`ZeroDivisionError` solprop's monitor raises on light after a launch that
      held none.
    """
    kind = notice.kind
    if kind is NoticeKind.GUARD:
        where = None if notice.launch else notice.z
        return BoundaryError(_guard_message(check, notice.value, where, stop=True))
    if kind is NoticeKind.CROSSING:
        return BoundaryError(_split_crossing_message(notice.z, notice.cap, notice.step))
    if kind is NoticeKind.READING:
        return BackendError(
            "under the split_step scheme a committed step's reading must carry what "
            "the step's split found, and that must hold light where the state does; "
            f"the split found none at z = {notice.z:.6e} m"
        )
    if kind is NoticeKind.DIVISION:
        return _zero_division()
    return BackendError(f"the native session stopped on a {kind.name.lower()} notice")


def _split_crossing_message(z: float, cap: float, step: float) -> str:
    """Return the message of a fixed schedule refused by the split collar's cap.

    The reference's text (``HostSession._refuse_scheduled_crossing``), for the cap
    the native session holds (SEP 47 section 6.4).

    - ``z``: where the cap was read, m.
    - ``cap``: the split temporal collar's crossing limit, m.
    - ``step``: the schedule's longest step before the next check, m.
    """
    return (
        f"at z = {z:.6e} m the split temporal collar's crossing limit "
        f"fell to {cap:.3e} m, below the fixed schedule's step {step:.3e} m "
        "before the next check: light has spread into bins that walk faster "
        "than the launch's, and a step that long could carry it through the "
        "collar and round the window undamped. Take shorter steps, step "
        'adaptively, or use absorber_scheme="nonlinear"'
    )
