# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's boundary monitor, as solprop configures and reads it.

solprop SEP 47 section 6.4 and the maintainer's decision 4 at its acceptance: the
native session owns the boundary monitor's every decision and record
(``watch::Monitor``, ``native/include/solprop_native/watch.hpp``). This module only
translates: it forms the monitor's configuration from solprop's
:class:`~solprop.monitor.BoundaryMonitor` and the problem, reads an earlier
execution's record into what a resume restores, formats the native record as
:meth:`~solprop.monitor.BoundaryMonitor.summary` lays it out, and formats a native
verdict as solprop's warning or exception. It decides nothing.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from solprop.backends.driver import (
    _split_crossing_message,  # pyright: ignore[reportPrivateUsage] - one text
)
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.monitor import (
    _FLOOR,  # pyright: ignore[reportPrivateUsage] - the module's constant
    FLOOR_MARGIN,
    _guard_message,  # pyright: ignore[reportPrivateUsage] - one text
    _removal_message,  # pyright: ignore[reportPrivateUsage] - one text
)

if TYPE_CHECKING:
    from solprop.boundaries import SplitCrossing
    from solprop.monitor import BoundaryCheck, BoundaryMonitor

__all__ = [
    "NOTICE_KINDS",
    "PLANE_COLUMNS",
    "monitor_config",
    "notice_error",
    "notice_warning",
    "record_of",
    "resumption",
    "unwatched_of",
]

NONE, GUARD, REMOVAL, CROSSING, READING = range(5)
"""A native notice's kinds (``session::Notice::Kind``)."""

NOTICE_KINDS = ("none", "guard", "removal", "crossing", "reading")
"""Their names, in that order."""

PLANE_COLUMNS = ("z", "collar_occupancy", "guard_low", "guard_high", "removed_fraction")
"""The monitor record's plane rows, in the native order (``watch::plane_columns``)."""

_NUMBERS = (
    "launch_energy",
    "launch_guard_low",
    "launch_guard_high",
    "launch_collar",
    "guard_low_max",
    "guard_high_max",
    "guard_low_at",
    "guard_high_at",
    "collar_max",
    "raw_removed_fraction",
    "removed_signed",
    "removed_fraction",
    "floor_removed_fraction",
    "quiet_checks",
    "raw_guard_max",
    "steps",
    "skipped",
    "collar_left_out_checks",
)
"""The numbers a resume reads from an earlier record (``watch::Resumption``)."""

_UNMEASURED = (
    "launch_energy",
    "launch_guard_low",
    "launch_guard_high",
    "launch_collar",
    "guard_low_at",
    "guard_high_at",
)
"""The record's fields that read ``None`` where nothing was measured."""


def monitor_config(
    monitor: BoundaryMonitor, crossing: SplitCrossing | None, *, can_rest: bool
) -> dict[str, object]:
    """Return the native monitor's configuration for a run's fresh monitor.

    The predictions' policy is the monitor's own (:meth:`BoundaryMonitor.watch_policy`)
    with the split collar's crossing limit the driver would give it
    (``StepDriver.attach_monitor``); its floor regions and launch floor are formed
    natively at the launch.

    Parameters
    ----------
    monitor : solprop.monitor.BoundaryMonitor
        The run's fresh monitor; it is not changed.
    crossing : solprop.boundaries.SplitCrossing or None
        The split collar's crossing limit (the problem's, under the split scheme).
    can_rest : bool
        Whether the route's stages may leave a quiet collar out
        (``StepDriver._collar_can_rest``).

    Returns
    -------
    dict
        The configuration ``_core`` reads.
    """
    configured = dataclasses.replace(monitor, crossing=crossing)
    policy = configured.watch_policy()
    policy["floor_regions"] = [0.0, 0.0, 0.0, 0.0]
    policy["launch_floor"] = 0.0
    return {
        "policy": policy,
        "floor_share": _FLOOR,
        "floor_margin": FLOOR_MARGIN,
        "budget": float(monitor.check.budget),
        "rate": np.ascontiguousarray(monitor.weights[3], dtype=np.float64),
        "can_rest": bool(can_rest),
    }


def _number(record: Mapping[str, object], name: str) -> float | None:
    """Return a record's number, or ``None`` when it is absent or not a number.

    As :func:`solprop.monitor._number` reads it, without its default, which the native
    reconstruction applies.

    Parameters
    ----------
    record : collections.abc.Mapping
        A monitor record.
    name : str
        The field.

    Returns
    -------
    float or None
        The value.
    """
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def resumption(
    record: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return what the native resume reads of an earlier execution's record.

    Only the record's types are read here, as :meth:`BoundaryMonitor.resume` reads
    them: each number present, the lists, and the plane rows when they have the five
    columns. Which are restored, their defaults and the fallbacks are native.

    Parameters
    ----------
    record : collections.abc.Mapping or None
        The ``boundary_monitor`` record of the latest earlier execution, or ``None``.

    Returns
    -------
    dict
        The resumption ``_core`` reads.
    """
    if record is None:
        return {"record": False, "collar_quiet": False}
    out: dict[str, object] = {
        "record": True,
        "collar_quiet": bool(record.get("collar_quiet", False)),
    }
    for name in _NUMBERS:
        value = _number(record, name)
        if value is not None:
            out[name] = value
    listed = record.get("unwatched")
    if isinstance(listed, list):
        out["unwatched"] = [
            [float(one) for one in cast("list[float]", stretch)]
            for stretch in cast("list[object]", listed)
        ]
    resumed = record.get("resumed_at")
    if isinstance(resumed, list):
        out["resumed_at"] = [float(one) for one in cast("list[float]", resumed)]
    rows = record.get("planes")
    if isinstance(rows, Mapping) and set(cast("Mapping[str, object]", rows)) == set(
        PLANE_COLUMNS
    ):
        columns = cast("Mapping[str, list[float | None]]", rows)
        out["planes"] = [
            [math.nan if one is None else float(one) for one in columns[name]]
            for name in PLANE_COLUMNS
        ]
    return out


def unwatched_of(stretches: Sequence[tuple[float, float]]) -> list[list[float]]:
    """Return the stretches a resumed run ran without a monitor, as lists.

    Parameters
    ----------
    stretches : collections.abc.Sequence of tuple of float
        ``(start, end)`` in m.

    Returns
    -------
    list
        ``[start, end]`` pairs.
    """
    return [[float(start), float(end)] for start, end in stretches]


def record_of(check: BoundaryCheck, native: Mapping[str, object]) -> dict[str, object]:
    """Format the native monitor's record as :meth:`BoundaryMonitor.summary` does.

    Parameters
    ----------
    check : solprop.monitor.BoundaryCheck
        The run's thresholds, which head the record.
    native : collections.abc.Mapping
        ``_core``'s record: every number, NaN where nothing was measured.

    Returns
    -------
    dict
        The record, with ``None`` for what was never measured.
    """

    def finite(value: object) -> float | None:
        """Return a number, or ``None`` when it was never measured (not finite).

        Parameters
        ----------
        value : object
            A recorded number.

        Returns
        -------
        float or None
            The value, or ``None``.
        """
        number = float(cast("float", value))
        return number if math.isfinite(number) else None

    record: dict[str, object] = dict(dataclasses.asdict(check))
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


def notice_warning(
    check: BoundaryCheck, notice: Mapping[str, object]
) -> tuple[str, type[Warning]]:
    """Return a native warning's message and category.

    Parameters
    ----------
    check : solprop.monitor.BoundaryCheck
        The run's thresholds.
    notice : collections.abc.Mapping
        The warning's verdict (``session::Notice``).

    Returns
    -------
    tuple
        The message and :class:`~solprop.exceptions.BoundaryWarning`.

    Raises
    ------
    solprop.exceptions.BackendError
        For a kind that does not warn.
    """
    kind = int(cast("int", notice["kind"]))
    value = float(cast("float", notice["value"]))
    if kind == GUARD:
        z = None if notice["launch"] else float(cast("float", notice["z"]))
        return _guard_message(check, value, z, stop=False), BoundaryWarning
    if kind == REMOVAL:
        return _removal_message(check, value), BoundaryWarning
    msg = f"the native session warned with a {_kind_name(kind)} notice"
    raise BackendError(msg)


def notice_error(check: BoundaryCheck, notice: Mapping[str, object]) -> BaseException:
    """Return the exception a native monitor's stop reports.

    Parameters
    ----------
    check : solprop.monitor.BoundaryCheck
        The run's thresholds.
    notice : collections.abc.Mapping
        The stop's verdict (``session::Notice``).

    Returns
    -------
    BaseException
        A :class:`~solprop.exceptions.BoundaryError`, or a
        :class:`~solprop.exceptions.BackendError` for a reading the split left empty.
    """
    kind = int(cast("int", notice["kind"]))
    z = float(cast("float", notice["z"]))
    if kind == GUARD:
        where = None if notice["launch"] else z
        worst = float(cast("float", notice["value"]))
        return BoundaryError(_guard_message(check, worst, where, stop=True))
    if kind == CROSSING:
        cap = float(cast("float", notice["cap"]))
        step = float(cast("float", notice["step"]))
        return BoundaryError(_split_crossing_message(z, cap, step))
    if kind == READING:
        return BackendError(
            "under the split_step scheme a committed step's reading must carry what "
            "the step's split found, and that must hold light where the state does; "
            f"the split found none at z = {z:.6e} m"
        )
    return BackendError(f"the native session stopped on a {_kind_name(kind)} notice")


def _kind_name(kind: int) -> str:
    """Name a native notice's kind.

    Parameters
    ----------
    kind : int
        ``session::Notice::Kind``.

    Returns
    -------
    str
        Its name.
    """
    return NOTICE_KINDS[kind] if 0 <= kind < len(NOTICE_KINDS) else f"kind {kind}"
