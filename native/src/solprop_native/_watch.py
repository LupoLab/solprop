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
import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from solprop.exceptions import BackendError, BoundaryError
from solprop.monitor import (
    _FLOOR,  # pyright: ignore[reportPrivateUsage] - the module's constant
    FLOOR_MARGIN,
    _boundary_warning,  # pyright: ignore[reportPrivateUsage] - one warning
    _count,  # pyright: ignore[reportPrivateUsage] - one refusal
    _guard_message,  # pyright: ignore[reportPrivateUsage] - one text
    _removal_message,  # pyright: ignore[reportPrivateUsage] - one text
)

if TYPE_CHECKING:
    from solprop.boundaries import SplitCrossing
    from solprop.monitor import BoundaryCheck, BoundaryMonitor

__all__ = [
    "PLANE_COLUMNS",
    "MonitorConfig",
    "Notice",
    "NoticeKind",
    "Resumption",
    "monitor_config",
    "notice_error",
    "notice_warning",
    "record_of",
    "resumption",
    "unwatched_of",
]


class NoticeKind(enum.IntEnum):
    """A native notice's kind (``session::Notice::Kind``).

    Attributes
    ----------
    NONE : int
        No notice.
    GUARD : int
        A guard occupancy, at the launch or at a check.
    REMOVAL : int
        The collar's removal at the end of a run.
    CROSSING : int
        A replay's step passes the split collar's cap.
    READING : int
        The split found nothing where the state holds light.
    DIVISION : int
        Light after a launch that held none, which solprop's monitor divides by.
    TRANSVERSE : int
        The transverse edge's verdict, which :mod:`solprop_native._transverse`
        formats.
    """

    NONE = 0
    GUARD = 1
    REMOVAL = 2
    CROSSING = 3
    READING = 4
    DIVISION = 5
    TRANSVERSE = 6


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
    "raw_guard_max",
)
"""The numbers a resume reads from an earlier record (``watch::Resumption``)."""

_COUNTS = ("steps", "skipped", "collar_left_out_checks")
"""The counts it reads, each whole or refused (:func:`solprop.monitor._count`)."""

_UNMEASURED = (
    "launch_energy",
    "launch_guard_low",
    "launch_guard_high",
    "launch_collar",
    "guard_low_at",
    "guard_high_at",
)
"""The record's fields that read ``None`` where nothing was measured."""


@dataclass(frozen=True)
class MonitorConfig:
    """The native monitor's configuration (``watch::Config``).

    Attributes
    ----------
    policy : collections.abc.Mapping
        The predictions' policy (:meth:`BoundaryMonitor.watch_policy`'s fields); its
        floor regions and launch floor are formed natively at the launch.
    floor_share : float
        ``_FLOOR``: no light is judged below this share of the launch energy.
    floor_margin : float
        :data:`solprop.monitor.FLOOR_MARGIN`.
    budget : float
        The budget the finish's removal is judged against.
    rate : numpy.ndarray
        The collar's power rate on the coarse axis, 1/m.
    can_rest : bool
        Whether the route's stages may leave a quiet collar out.
    """

    policy: Mapping[str, object]
    floor_share: float
    floor_margin: float
    budget: float
    rate: np.ndarray
    can_rest: bool

    def native(self) -> dict[str, object]:
        """Return the configuration as ``_core`` reads it.

        Returns
        -------
        dict
            Its fields.
        """
        return {
            "policy": dict(self.policy),
            "floor_share": self.floor_share,
            "floor_margin": self.floor_margin,
            "budget": self.budget,
            "rate": self.rate,
            "can_rest": self.can_rest,
        }


@dataclass(frozen=True)
class Resumption:
    """What a native resume reads of an earlier record (``watch::Resumption``).

    Attributes
    ----------
    record : bool
        Whether there is a record at all.
    collar_quiet : bool
        ``bool(record.get("collar_quiet", False))``.
    numbers : collections.abc.Mapping
        Each number present in the record, by field; the counts already whole.
    unwatched : tuple or None
        The record's ``unwatched`` stretches, when it lists them.
    resumed_at : tuple or None
        The record's ``resumed_at``, when it lists them.
    planes : tuple or None
        The record's plane rows by :data:`PLANE_COLUMNS` (NaN for ``None``), when they
        have those five columns.
    """

    record: bool
    collar_quiet: bool = False
    numbers: Mapping[str, float] = field(default_factory=dict[str, float])
    unwatched: tuple[tuple[float, ...], ...] | None = None
    resumed_at: tuple[float, ...] | None = None
    planes: tuple[tuple[float, ...], ...] | None = None

    def native(self) -> dict[str, object]:
        """Return the resumption as ``_core`` reads it.

        Returns
        -------
        dict
            Its fields, the numbers flattened in and absent lists left out.
        """
        out: dict[str, object] = {
            "record": self.record,
            "collar_quiet": self.collar_quiet,
            **self.numbers,
        }
        if self.unwatched is not None:
            out["unwatched"] = [list(stretch) for stretch in self.unwatched]
        if self.resumed_at is not None:
            out["resumed_at"] = list(self.resumed_at)
        if self.planes is not None:
            out["planes"] = [list(column) for column in self.planes]
        return out


@dataclass(frozen=True)
class Notice:
    """A native monitor's verdict (``session::Notice``): a warning or a stop.

    Attributes
    ----------
    kind : NoticeKind
        What it is.
    launch : bool
        For a guard: whether it is the launch's judgement.
    z : float
        Where, m.
    value : float
        A guard's amplitude, or the removed fraction.
    cap : float
        A crossing's cap, m.
    step : float
        A crossing's longest step before the next check, m.
    """

    kind: NoticeKind
    launch: bool
    z: float
    value: float
    cap: float
    step: float

    @classmethod
    def of(cls, native: Mapping[str, object]) -> Notice:
        """Return the notice ``_core`` reports.

        Parameters
        ----------
        native : collections.abc.Mapping
            ``SessionEvent.notice``.

        Returns
        -------
        Notice
            The notice.
        """
        return cls(
            NoticeKind(int(cast("int", native["kind"]))),
            bool(native["launch"]),
            float(cast("float", native["z"])),
            float(cast("float", native["value"])),
            float(cast("float", native["cap"])),
            float(cast("float", native["step"])),
        )


def monitor_config(
    monitor: BoundaryMonitor, crossing: SplitCrossing | None, *, can_rest: bool
) -> MonitorConfig:
    """Return the native monitor's configuration for a run's fresh monitor.

    The predictions' policy is the monitor's own (:meth:`BoundaryMonitor.watch_policy`)
    with the split collar's crossing limit the driver would give it
    (``HostDriver.attach_monitor``).

    Parameters
    ----------
    monitor : solprop.monitor.BoundaryMonitor
        The run's fresh monitor; it is not changed.
    crossing : solprop.boundaries.SplitCrossing or None
        The split collar's crossing limit (the problem's, under the split scheme).
    can_rest : bool
        Whether the route's stages may leave a quiet collar out
        (``HostDriver._collar_can_rest``).

    Returns
    -------
    MonitorConfig
        The configuration.
    """
    configured = dataclasses.replace(monitor, crossing=crossing)
    policy = configured.watch_policy()
    policy["floor_regions"] = [0.0, 0.0, 0.0, 0.0]
    policy["launch_floor"] = 0.0
    return MonitorConfig(
        policy=policy,
        floor_share=_FLOOR,
        floor_margin=FLOOR_MARGIN,
        budget=float(monitor.check.budget),
        rate=np.ascontiguousarray(monitor.weights[3], dtype=np.float64),
        can_rest=bool(can_rest),
    )


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


def resumption(record: Mapping[str, object] | None) -> Resumption:
    """Return what the native resume reads of an earlier execution's record.

    Only the record's types are read here, as :meth:`BoundaryMonitor.resume` reads
    them: each number present, the lists, and the plane rows when they have the five
    columns. A count that is not whole is refused as that resume refuses it, once the
    record's launch energy makes it read the counts. Which numbers are restored,
    their defaults and the fallbacks are native.

    Parameters
    ----------
    record : collections.abc.Mapping or None
        The ``boundary_monitor`` record of the latest earlier execution, or ``None``.

    Returns
    -------
    Resumption
        The resumption.

    Raises
    ------
    solprop.exceptions.BoundaryError
        For a count that is not finite or is negative.
    """
    if record is None:
        return Resumption(record=False)
    numbers: dict[str, float] = {}
    for name in _NUMBERS:
        value = _number(record, name)
        if value is not None:
            numbers[name] = value
    launch = numbers.get("launch_energy", math.nan)
    if launch > 0.0:
        # `BoundaryMonitor.resume` reads the counts only past its launch test.
        quiet = _number(record, "quiet_checks")
        if quiet is not None and math.isfinite(quiet):
            numbers["quiet_checks"] = float(_count(quiet, "quiet_checks"))
        for name in _COUNTS:
            value = _number(record, name)
            numbers[name] = float(_count(0.0 if value is None else value, name))
    listed = record.get("unwatched")
    resumed = record.get("resumed_at")
    rows = record.get("planes")
    planes = None
    if isinstance(rows, Mapping) and set(cast("Mapping[str, object]", rows)) == set(
        PLANE_COLUMNS
    ):
        columns = cast("Mapping[str, list[float | None]]", rows)
        planes = tuple(
            tuple(math.nan if one is None else float(one) for one in columns[name])
            for name in PLANE_COLUMNS
        )
    return Resumption(
        record=True,
        collar_quiet=bool(record.get("collar_quiet", False)),
        numbers=numbers,
        unwatched=tuple(
            tuple(float(one) for one in cast("list[float]", stretch))
            for stretch in cast("list[object]", listed)
        )
        if isinstance(listed, list)
        else None,
        resumed_at=tuple(float(one) for one in cast("list[float]", resumed))
        if isinstance(resumed, list)
        else None,
        planes=planes,
    )


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


def notice_warning(check: BoundaryCheck, notice: Notice) -> Warning:
    """Return a native warning, as solprop's monitor issues it.

    Parameters
    ----------
    check : solprop.monitor.BoundaryCheck
        The run's thresholds.
    notice : Notice
        The warning's verdict.

    Returns
    -------
    Warning
        The :class:`~solprop.exceptions.BoundaryWarning`, with its attributes.

    Raises
    ------
    solprop.exceptions.BackendError
        For a kind that does not warn.
    """
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

    Returns
    -------
    ZeroDivisionError
        The error.
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

    Parameters
    ----------
    check : solprop.monitor.BoundaryCheck
        The run's thresholds.
    notice : Notice
        The stop's verdict.

    Returns
    -------
    BaseException
        A :class:`~solprop.exceptions.BoundaryError`; a
        :class:`~solprop.exceptions.BackendError` for a reading the split left empty;
        the :class:`ZeroDivisionError` solprop's monitor raises on light after a launch
        that held none.
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

    The reference's text (``HostDriver._refuse_scheduled_crossing``), for the cap
    the native session holds (SEP 47 section 6.4).

    Parameters
    ----------
    z : float
        Where the cap was read, m.
    cap : float
        The split temporal collar's crossing limit, m.
    step : float
        The schedule's longest step before the next check, m.

    Returns
    -------
    str
        The message.
    """
    return (
        f"at z = {z:.6e} m the split temporal collar's crossing limit "
        f"fell to {cap:.3e} m, below the fixed schedule's step {step:.3e} m "
        "before the next check: light has spread into bins that walk faster "
        "than the launch's, and a step that long could carry it through the "
        "collar and round the window undamped. Take shorter steps, step "
        'adaptively, or use absorber_scheme="nonlinear"'
    )
