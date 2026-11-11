# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate Julia monitor records without making numerical monitoring decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import cast

import numpy as np
from jaxtyping import Float

from solprop.backends.driver import (
    _split_crossing_message,  # pyright: ignore[reportPrivateUsage] - shared message
)
from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._ownership import JuliaHandle
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.exceptions import BackendError, BoundaryError
from solprop.monitor import (
    BoundaryCheck,
    _boundary_warning,  # pyright: ignore[reportPrivateUsage] - shared warning
    _count,  # pyright: ignore[reportPrivateUsage] - shared legacy count policy
    _guard_message,  # pyright: ignore[reportPrivateUsage] - shared message
    _log_finish,  # pyright: ignore[reportPrivateUsage] - shared completion logging
    _number,  # pyright: ignore[reportPrivateUsage] - shared legacy record policy
    _removal_message,  # pyright: ignore[reportPrivateUsage] - shared message
)

__all__ = [
    "MonitorContinuation",
    "MonitorLog",
    "MonitorNotice",
    "MonitorPayload",
    "MonitorResume",
    "monitor_error",
    "monitor_finished",
    "monitor_warning",
    "parse_resume",
    "read_monitor",
]

_COLUMNS = ("z", "collar_occupancy", "guard_low", "guard_high", "removed_fraction")
_UNMEASURED = frozenset(
    (
        "launch_energy",
        "launch_guard_low",
        "launch_guard_high",
        "launch_collar",
        "guard_low_at",
        "guard_high_at",
    )
)


@dataclass(frozen=True)
class MonitorContinuation:
    """Own the parsed numerical fields of a temporal monitor continuation.

    Attributes
    ----------
    launch : float
        Original launch proxy energy, positive when its totals can be restored.
    launch_occupancy, guard_max, guard_at : tuple of float
        Three launch shares, two guard maxima and their two coordinates in metres.
    collar_max, raw_removed, floor_removed : float
        Maximum collar share and cumulative raw/expected-floor removal shares.
    quiet_run : int
        Nonnegative parsed quiet count, used only when ``has_quiet_run`` is true.
    quiet : bool
        Previous collar-quiet flag, interpreted by the current Julia policy.
    raw_guard_max : float
        Maximum guard share before subtracting the noise floor.
    steps, skipped, left_out_checks : int
        Nonnegative counts parsed with the shared monitor's truncation policy.
    last_plane_z : float
        Last restored plane in metres, or NaN without one.
    has_quiet_run : bool
        Whether a finite legacy quiet count was present, including explicit zero.
    """

    launch: float
    launch_occupancy: tuple[float, float, float]
    guard_max: tuple[float, float]
    guard_at: tuple[float, float]
    collar_max: float
    raw_removed: float
    floor_removed: float
    quiet_run: int
    quiet: bool
    raw_guard_max: float
    steps: int
    skipped: int
    left_out_checks: int
    last_plane_z: float
    has_quiet_run: bool


@dataclass(frozen=True)
class MonitorResume:
    """Retain a parsed resume and independently owned record history.

    Attributes
    ----------
    numerical : MonitorContinuation or None
        Numerical totals, absent when there was no usable positive launch record.
        Absence still denotes a resumed run whose launch must not be judged again.
    plane : bool
        Whether this resume owes a monitor row at its saved coordinate.
    planes : numpy.ndarray
        Owned read-only rows, shape ``(n_rows,5)``, in the shared monitor column order.
    resumed_at : tuple of float
        Earlier resumption coordinates, in metres.
    unwatched : tuple of tuple of float
        Earlier unmonitored stretches and new positive-length stretches, in metres.
    """

    numerical: MonitorContinuation | None
    plane: bool
    planes: Float[np.ndarray, "n_rows 5"]
    resumed_at: tuple[float, ...]
    unwatched: tuple[tuple[float, ...], ...]


def _plane_history(record: Mapping[str, object]) -> Float[np.ndarray, "n_rows 5"]:
    """Copy equal-length saved monitor columns into owned rows.

    Parameters
    ----------
    record : mapping
        Legacy public monitor record, whose optional planes are column lists.

    Returns
    -------
    numpy.ndarray
        Read-only binary64 rows in the shared five-column order.

    Raises
    ------
    BackendError
        If the saved column lengths disagree.
    """
    rows = record.get("planes")
    if not isinstance(rows, Mapping) or set(rows) != set(_COLUMNS):
        result = np.empty((0, 5), dtype=np.float64)
    else:
        columns = cast("Mapping[str, Sequence[float | None]]", rows)
        parsed = [
            [math.nan if value is None else float(value) for value in columns[name]]
            for name in _COLUMNS
        ]
        if len({len(column) for column in parsed}) != 1:
            raise BackendError("Julia monitor continuation has unequal plane columns")
        result = np.array(parsed, dtype=np.float64).T.copy()
    result.setflags(write=False)
    return result


def parse_resume(continuation: Mapping[str, object]) -> MonitorResume:
    """Parse the shared monitor's legacy continuation before entering Julia.

    Parameters
    ----------
    continuation : mapping
        ``Start.monitor`` or its later fallback, with ``record``, ``plane`` and
        ``unwatched`` entries. Missing totals do not imply a fresh launch.

    Returns
    -------
    MonitorResume
        Detached scalar restoration and owned history, with no live monitor retained.

    Raises
    ------
    BoundaryError
        For a count refused by the shared monitor's legacy parsing policy.
    BackendError
        For inconsistent saved plane-column lengths.
    """
    raw = continuation.get("record")
    record = cast("Mapping[str, object]", raw) if isinstance(raw, Mapping) else {}
    listed = record.get("unwatched")
    unwatched = (
        tuple(tuple(float(value) for value in stretch) for stretch in listed)
        if isinstance(listed, list)
        else ()
    )
    extra = cast("Sequence[tuple[float, float]]", continuation.get("unwatched", ()))
    unwatched += tuple(
        (float(start), float(end)) for start, end in extra if end > start
    )
    launch = _number(record, "launch_energy")
    planes = _plane_history(record if launch > 0.0 else {})
    resumed = record.get("resumed_at")
    resumed_at = (
        tuple(float(value) for value in resumed)
        if launch > 0.0 and isinstance(resumed, list)
        else ()
    )
    numerical = None
    if launch > 0.0:
        raw_removed = _number(record, "raw_removed_fraction", 0.0)
        judged = _number(record, "removed_signed")
        if not math.isfinite(judged):
            judged = _number(record, "removed_fraction", 0.0)
        quiet = _number(record, "quiet_checks")
        has_quiet = math.isfinite(quiet)
        # This is the shared legacy record's fallback, not a new live reduction.
        floor_removed = _number(record, "floor_removed_fraction", raw_removed - judged)
        numerical = MonitorContinuation(
            launch,
            (
                _number(record, "launch_guard_low"),
                _number(record, "launch_guard_high"),
                _number(record, "launch_collar"),
            ),
            (
                _number(record, "guard_low_max", 0.0),
                _number(record, "guard_high_max", 0.0),
            ),
            (_number(record, "guard_low_at"), _number(record, "guard_high_at")),
            _number(record, "collar_max", 0.0),
            raw_removed,
            floor_removed,
            _count(quiet, "quiet_checks") if has_quiet else 0,
            bool(record.get("collar_quiet", False)),
            _number(record, "raw_guard_max", 0.0),
            _count(_number(record, "steps", 0.0), "steps"),
            _count(_number(record, "skipped", 0.0), "skipped"),
            _count(
                _number(record, "collar_left_out_checks", 0.0), "collar_left_out_checks"
            ),
            float(planes[-1, 0]) if planes.size else math.nan,
            has_quiet,
        )
    return MonitorResume(
        numerical, bool(continuation.get("plane", False)), planes, resumed_at, unwatched
    )


@dataclass(frozen=True)
class MonitorNotice:
    """Detach a Julia-decided temporal warning or boundary refusal.

    Attributes
    ----------
    code : int
        Julia temporal notice: one for guard warning, two for stop, three for removal.
    z : float
        Coordinate in metres; NaN identifies the launch or closing removal notice.
    value : float
        Already computed guard amplitude, or removed energy share for code three.
    """

    code: int
    z: float
    value: float


@dataclass(frozen=True)
class MonitorPayload:
    """Own one complete generation's numerical monitor record and bounded rows.

    Attributes
    ----------
    generation : int
        Bound numerical event generation, zero only for the initial state.
    record : mapping
        Detached numerical summary fields; Python only formats unavailable values.
    rows : numpy.ndarray
        Read-only rows, shape ``(n_rows,5)``, with at most two new rows.
    attached, finished, resuming : bool
        Julia's lifecycle facts, including whether a pending resume is unconsumed.
    committed : float
        Julia monitor's current committed coordinate, in metres or NaN before launch.
    notice : MonitorNotice or None
        An already decided temporal notice, absent when none is pending.
    crossing : tuple of float or None
        Fixed-itinerary refusal's position, cap and step in metres.
    """

    generation: int
    record: Mapping[str, object]
    rows: Float[np.ndarray, "n_rows 5"]
    attached: bool
    finished: bool
    resuming: bool
    committed: float
    notice: MonitorNotice | None
    crossing: tuple[float, float, float] | None


@dataclass(frozen=True)
class MonitorLog:
    """Publish detached monitor data once per generation, with owned row history.

    Attributes
    ----------
    check : solprop.monitor.BoundaryCheck
        Immutable public policy used only for record and message formatting.
    start_z : float
        This execution's starting coordinate, in metres, including resumes.
    resume : MonitorResume or None
        The first supplied continuation; absent for a fresh execution.
    payload : MonitorPayload or None
        Last complete copied payload; absent before initial publication.
    chunks : tuple of numpy.ndarray
        Retained independent row chunks, including restored history.
    resumed_at : tuple of float
        Committed resumption coordinates, in metres.
    resumed : bool
        Whether Julia has consumed this execution's continuation.
    """

    check: BoundaryCheck
    start_z: float
    resume: MonitorResume | None = None
    payload: MonitorPayload | None = None
    chunks: tuple[Float[np.ndarray, "_ 5"], ...] = ()
    resumed_at: tuple[float, ...] = ()
    resumed: bool = False

    def continued(self, resume: MonitorResume) -> MonitorLog:
        """Retain the first continuation and invalidate its prior scalar publication.

        Parameters
        ----------
        resume : MonitorResume
            Continuation already accepted by the fresh numerical monitor.

        Returns
        -------
        MonitorLog
            New immutable record, or this record if a previous continuation wins.
        """
        if self.resume is not None:
            return self
        return replace(
            self,
            resume=resume,
            payload=None,
            chunks=(resume.planes,) if resume.planes.size else (),
            resumed_at=resume.resumed_at,
        )

    def captured(self, payload: MonitorPayload) -> MonitorLog:
        """Publish a complete payload without duplicating rows or resume metadata.

        Parameters
        ----------
        payload : MonitorPayload
            Independently copied, non-draining numerical payload.

        Returns
        -------
        MonitorLog
            New immutable publication, or this record for the same generation.
        """
        if self.payload is not None and self.payload.generation == payload.generation:
            return self
        resumed = self.resume is not None and not payload.resuming and not self.resumed
        return replace(
            self,
            payload=payload,
            chunks=self.chunks + ((payload.rows,) if payload.rows.size else ()),
            resumed_at=self.resumed_at + ((self.start_z,) if resumed else ()),
            resumed=self.resumed or resumed,
        )

    def summary(self) -> dict[str, object] | None:
        """Format the last numerical record as an independently owned public summary.

        Returns
        -------
        dict or None
            Shared monitor summary, absent only before its initial numerical read.
            No reduction or monitoring decision is evaluated here.
        """
        if self.payload is None:
            return None
        record: dict[str, object] = {**asdict(self.check), **self.payload.record}
        for name in _UNMEASURED:
            value = cast("float", record[name])
            record[name] = value if math.isfinite(value) else None
        record["resumed_at"] = list(self.resumed_at)
        record["unwatched"] = (
            [] if self.resume is None else [list(one) for one in self.resume.unwatched]
        )
        rows = np.concatenate(self.chunks) if self.chunks else np.empty((0, 5))
        record["planes"] = {
            name: [
                float(value) if math.isfinite(value) else None for value in rows[:, i]
            ]
            for i, name in enumerate(_COLUMNS)
        }
        return record


def _read_monitor(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> MonitorPayload | None:
    """Copy monitor scalars and rows while every Julia wrapper is guarded.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized runtime inside main-thread guarded entry.
    handle : JuliaHandle
        Owned numerical session.
    generation : int
        Pending generation, or zero for the initial state.

    Returns
    -------
    MonitorPayload or None
        Fully detached numerical monitor data, absent without temporal monitoring.
    """
    session = handle.borrow()
    result = runtime.engine.session_monitor_payload(session, generation)
    if result is None:
        return None
    state, continuation = result.record, result.record.continuation
    count = int(result.row_count)
    if count not in (0, 1, 2):
        raise BackendError("Julia temporal monitor exceeded its bounded row capacity")
    rows = np.empty((count, 5), dtype=np.float64)
    getattr(runtime.engine, "copy_session_monitor_rows!")(rows, session, generation)
    rows.setflags(write=False)
    record = {
        "launch_energy": float(continuation.launch),
        "launch_guard_low": float(continuation.launch_occupancy[0]),
        "launch_guard_high": float(continuation.launch_occupancy[1]),
        "launch_collar": float(continuation.launch_occupancy[2]),
        "guard_low_max": float(continuation.guard_max[0]),
        "guard_high_max": float(continuation.guard_max[1]),
        "guard_low_at": float(continuation.guard_at[0]),
        "guard_high_at": float(continuation.guard_at[1]),
        "collar_max": float(continuation.collar_max),
        "removed_signed": float(state.removed_signed),
        "removed_fraction": float(state.removed_fraction),
        "floor_removed_fraction": float(continuation.floor_removed),
        "quiet_checks": int(continuation.quiet_run),
        "collar_quiet": bool(continuation.quiet),
        "steps": int(continuation.steps),
        "skipped": int(continuation.skipped),
        "noise_floor": float(state.noise_floor),
        "raw_guard_max": float(continuation.raw_guard_max),
        "raw_removed_fraction": float(continuation.raw_removed),
        "collar_left_out_checks": int(continuation.left_out_checks),
    }
    notice = result.notice
    detached_notice = (
        None
        if notice is None
        else MonitorNotice(
            int(runtime.engine.Int(notice.code)), float(notice.z), float(notice.value)
        )
    )
    crossing = result.crossing
    return MonitorPayload(
        generation,
        record,
        rows,
        bool(result.attached),
        bool(result.finished),
        bool(state.resuming),
        float(state.committed),
        detached_notice,
        None
        if crossing is None
        else (float(crossing.z), float(crossing.cap), float(crossing.step)),
    )


def read_monitor(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> MonitorPayload | None:
    """Detach a complete numerical monitor payload inside guarded main-thread entry.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized process runtime.
    handle : JuliaHandle
        Owned session with the requested generation.
    generation : int
        Pending generation, or zero for the never-advanced initial transport state.

    Returns
    -------
    MonitorPayload or None
        Detached scalars and rows, or None for an unmonitored session.
    """
    runtime.check()
    return invoke(
        lambda: _read_monitor(runtime, handle, generation),
        "the julia monitor copy failed",
    )


def monitor_warning(check: BoundaryCheck, notice: MonitorNotice) -> Warning:
    """Format an already decided Julia warning with the shared monitor's message.

    Parameters
    ----------
    check : BoundaryCheck
        Immutable public formatting policy.
    notice : MonitorNotice
        Julia's guard or removal warning.

    Returns
    -------
    Warning
        Shared boundary warning with its parameter metadata.

    Raises
    ------
    BackendError
        For a notice code that cannot be a temporal warning.
    """
    if notice.code == 1:
        z = None if math.isnan(notice.z) else notice.z
        message = _guard_message(check, notice.value, z, stop=False)
    elif notice.code == 3:
        message = _removal_message(check, notice.value)
    else:
        raise BackendError(f"Julia temporal warning has unsupported code {notice.code}")
    return _boundary_warning(message)


def monitor_finished(payload: MonitorPayload) -> None:
    """Log Julia's completed numerical record through the shared monitor logger.

    Parameters
    ----------
    payload : MonitorPayload
        The first publication whose numerical services are finished.
    """
    record = payload.record
    _log_finish(
        cast("float", record["removed_fraction"]),
        cast("float", record["guard_low_max"]),
        cast("float", record["guard_high_max"]),
    )


def monitor_error(check: BoundaryCheck, payload: MonitorPayload) -> BoundaryError:
    """Format a Julia-decided temporal guard or fixed-crossing refusal.

    Parameters
    ----------
    check : BoundaryCheck
        Immutable public formatting policy.
    payload : MonitorPayload
        Closed numerical barrier with its notice or crossing facts.

    Returns
    -------
    BoundaryError
        Shared boundary error without another guard or step-size decision.

    Raises
    ------
    BackendError
        For a boundary barrier without its expected numerical notice.
    """
    if payload.crossing is not None:
        return BoundaryError(_split_crossing_message(*payload.crossing))
    notice = payload.notice
    if notice is None or notice.code != 2:
        raise BackendError("Julia boundary stop has no temporal refusal notice")
    z = None if math.isnan(notice.z) else notice.z
    return BoundaryError(_guard_message(check, notice.value, z, stop=True))
