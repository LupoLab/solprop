# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parse and format detached Julia edge records without numerical decisions."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from solprop.backends.julia._transverse import TransverseFormat
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.transverse_monitor import (
    _UNWATCHED_MESSAGE,  # pyright: ignore[reportPrivateUsage] - shared messages
    _UNWATCHED_REASON,  # pyright: ignore[reportPrivateUsage]
    TransverseStatus,
    _clip_message,  # pyright: ignore[reportPrivateUsage]
    _coarse_message,  # pyright: ignore[reportPrivateUsage]
    _face_message,  # pyright: ignore[reportPrivateUsage]
    _insensitive_message,  # pyright: ignore[reportPrivateUsage]
    _insensitive_reason,  # pyright: ignore[reportPrivateUsage]
    _log_declared,  # pyright: ignore[reportPrivateUsage]
    _log_finish,  # pyright: ignore[reportPrivateUsage]
    _ramp_band_message,  # pyright: ignore[reportPrivateUsage]
    _ramp_removal_message,  # pyright: ignore[reportPrivateUsage]
    _return_message,  # pyright: ignore[reportPrivateUsage]
    _stop_message,  # pyright: ignore[reportPrivateUsage]
)

__all__ = [
    "TransverseContinuation",
    "TransversePayload",
    "parse_transverse_resume",
    "transverse_error",
    "transverse_warning",
]

_STATUSES: tuple[TransverseStatus, ...] = (
    "clear",
    "warned",
    "stopped",
    "insensitive",
    "coarse",
    "unwatched",
    "off",
)


@dataclass(frozen=True)
class TransverseContinuation:
    """Own the shared edge's numerical continuation in the Julia field order.

    Attributes
    ----------
    launch : float
        Positive original launch proxy energy.
    baseline, peak, peak_raw : tuple of float
        Per-face launch, maximum judged and maximum raw energy shares.
    allowance : float
        Original input-noise allowance.
    ramp_raw, ramp_noise, noise_share_in_ramp, declared_share : float
        Raw/expected noise removal and original noise/declared ramp shares.
    spatial_removed, current, current_baseline : float
        Collar removal, cumulative inward current and its launch baseline.
    strip_baseline, strip_floor, strip_peak, returned_peak : float
        Strip launch/floor and maximum strip/returned energy shares.
    reads : int
        Saved read count, parsed with the shared legacy integer conversion.
    largest_gap, warned_at : float
        Largest read spacing and first warning coordinate, in metres.
    status : int
        Shared status index; only clear, warned, insensitive and coarse restore.
    """

    launch: float
    baseline: tuple[float, ...]
    peak: tuple[float, ...]
    peak_raw: tuple[float, ...]
    allowance: float
    ramp_raw: float
    ramp_noise: float
    noise_share_in_ramp: float
    declared_share: float
    spatial_removed: float
    current: float
    current_baseline: float
    strip_baseline: float
    strip_floor: float
    strip_peak: float
    returned_peak: float
    reads: int
    largest_gap: float
    warned_at: float
    status: int


def _number(record: Mapping[str, object], name: str, default: float = 0.0) -> float:
    value = record.get(name)
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else default
    )


def _faces(
    record: Mapping[str, object], name: str, count: int
) -> tuple[float, ...] | None:
    value = record.get(name)
    if not isinstance(value, list) or len(value) != count:
        return None
    return tuple(float(one) for one in cast("list[float]", value))


def parse_transverse_resume(
    record: object, count: int
) -> TransverseContinuation | None:
    """Parse a legacy edge record with the shared resume's conditional read order.

    Parameters
    ----------
    record : object
        Earlier monitor's ``transverse`` entry, absent or malformed when unusable.
    count : int
        Actual number of faces, from immutable preparation.

    Returns
    -------
    TransverseContinuation or None
        Numerical totals only after a positive launch and correctly sized baseline.

    Raises
    ------
    ValueError
        For a usable record whose per-face entries or read count cannot be converted.
    OverflowError
        For an infinite read count, as in the shared legacy resume.
    """
    if not isinstance(record, Mapping):
        return None
    mapping = cast("Mapping[str, object]", record)
    launch = _number(mapping, "launch_energy", math.nan)
    baseline = _faces(mapping, "edge_baseline", count)
    if not launch > 0.0 or baseline is None:
        return None
    status = mapping.get("transverse_status")
    restored_status = (
        _STATUSES.index(cast("TransverseStatus", status))
        if status in ("clear", "warned", "insensitive", "coarse")
        else 0
    )
    return TransverseContinuation(
        launch,
        baseline,
        _faces(mapping, "edge_max", count) or (0.0,) * count,
        _faces(mapping, "edge_max_raw", count) or (0.0,) * count,
        _number(mapping, "noise_allowance"),
        _number(mapping, "raw_ramp_removed"),
        _number(mapping, "noise_ramp_removed"),
        _number(mapping, "noise_share_in_ramp"),
        _number(mapping, "declared_share_in_ramp"),
        _number(mapping, "spatial_removed"),
        _number(mapping, "current"),
        _number(mapping, "current_baseline"),
        _number(mapping, "strip_baseline"),
        _number(mapping, "strip_floor"),
        _number(mapping, "strip_max"),
        _number(mapping, "returned_max"),
        int(_number(mapping, "reads")),
        _number(mapping, "largest_gap"),
        _number(mapping, "warned_at", math.nan),
        restored_status,
    )


@dataclass(frozen=True)
class TransversePayload:
    """Own Julia's edge summary and finish verdict for one transport generation.

    Attributes
    ----------
    continuation : TransverseContinuation
        Detached numerical totals; launch can be NaN before attachment.
    reason : int
        Zero for no reason, one unwatched, two insensitive.
    removed_signed : float
        Julia's signed judged ramp removal, also retained in the finish verdict.
    resumed : bool
        Whether positive earlier launch totals were restored.
    finish : tuple
        Numerical status index, largest face share, judged ramp removal and read count.
    """

    continuation: TransverseContinuation
    reason: int
    removed_signed: float
    resumed: bool
    finish: tuple[int, float, float, int]

    def summary(self, context: TransverseFormat) -> dict[str, object]:
        """Format the numerical record with shared public names and unavailable values.

        Parameters
        ----------
        context : TransverseFormat
            Declared face names, used without re-evaluating any numerical decision.

        Returns
        -------
        dict
            Fresh public ``transverse`` record with independent face lists.
        """
        state = self.continuation
        reasons = ("", _UNWATCHED_REASON, _insensitive_reason(state.allowance))
        return {
            "launch_energy": state.launch if math.isfinite(state.launch) else None,
            "noise_ramp_removed": state.ramp_noise,
            "noise_share_in_ramp": state.noise_share_in_ramp,
            "declared_share_in_ramp": state.declared_share,
            "transverse_status": _STATUSES[state.status],
            "transverse_reason": reasons[self.reason],
            "faces": list(context.faces),
            "edge_baseline": list(state.baseline),
            "edge_max": list(state.peak),
            "edge_max_raw": list(state.peak_raw),
            "noise_allowance": state.allowance,
            "ramp_removed": self.finish[2],
            "raw_ramp_removed": state.ramp_raw,
            "spatial_removed": state.spatial_removed,
            "reads": state.reads,
            "returned_max": state.returned_peak,
            "strip_max": state.strip_peak,
            "strip_baseline": state.strip_baseline,
            "strip_floor": state.strip_floor,
            "current": state.current,
            "current_baseline": state.current_baseline,
            "largest_gap": state.largest_gap,
            "warned_at": None if math.isnan(state.warned_at) else state.warned_at,
        }

    def log_declared(self) -> bool:
        """Log one declared launch after Julia has completed its launch judgement.

        Returns
        -------
        bool
            Whether launch is settled, so the caller must not attempt logging again.
            The caller applies the immutable design/declaration gate first.
        """
        state = self.continuation
        if math.isnan(state.launch) and not self.resumed:
            return False
        if state.launch > 0.0 and not self.resumed:
            _log_declared(state.declared_share)
        return True

    def log_finish(self) -> None:
        """Log Julia's completed verdict using the shared transverse logger."""
        status, largest, removed, reads = self.finish
        _log_finish(_STATUSES[status], largest, removed, reads)


def transverse_warning(
    context: TransverseFormat, code: int, values: tuple[float, float, int, float]
) -> BoundaryWarning:
    """Format an already decided Julia transverse warning.

    Parameters
    ----------
    context : TransverseFormat
        Host-only descriptive context.
    code : int
        Julia monitor notice code.
    values : tuple
        Position, energy share or step, one-based face and auxiliary scalar.

    Returns
    -------
    solprop.exceptions.BoundaryWarning
        Shared warning text without re-testing its threshold.

    Raises
    ------
    BackendError
        For an unsupported warning code or invalid face.
    """
    z, value, face, auxiliary = values
    if code in (4, 7) and not 1 <= face <= len(context.faces):
        raise BackendError("Julia transverse notice has an invalid face")
    if code == 4:
        message = _face_message(context.faces[face - 1], z, value)
    elif code == 6:
        message = _ramp_band_message(value, stop=False)
    elif code == 7:
        message = _clip_message(value, context.faces[face - 1])
    elif code == 8:
        message = _coarse_message(value, auxiliary)
    elif code == 9:
        message = _return_message(value, z, None)
    elif code == 11:
        message = _ramp_removal_message(value, None)
    elif code == 14:
        message = _insensitive_message(value)
    elif code == 15:
        message = _UNWATCHED_MESSAGE
    else:
        raise BackendError(f"Julia transverse warning has unsupported code {code}")
    return BoundaryWarning(message)


def transverse_error(
    context: TransverseFormat, code: int, values: tuple[float, float, int, float]
) -> BoundaryError:
    """Format an already decided Julia transverse stop.

    Parameters
    ----------
    context : TransverseFormat
        Host-only descriptive context.
    code : int
        Julia monitor notice code.
    values : tuple
        Position, energy share, one-based face and auxiliary first-warning coordinate.

    Returns
    -------
    solprop.exceptions.BoundaryError
        Shared stop text, including the existing grid-size recommendation.

    Raises
    ------
    BackendError
        For an unsupported stop code or invalid face.
    """
    z, value, face, auxiliary = values
    if code == 5:
        if not 1 <= face <= len(context.faces):
            raise BackendError("Julia transverse stop has an invalid face")
        message = _stop_message(
            context.space,
            context.length,
            context.stop,
            context.faces[face - 1],
            value,
            z=z,
            warned_at=auxiliary,
        )
    elif code == 10:
        message = _return_message(value, z, context.stop)
    elif code == 12:
        message = _ramp_removal_message(value, context.stop)
    elif code == 13:
        message = _ramp_band_message(value, stop=True)
    else:
        raise BackendError(f"Julia transverse stop has unsupported code {code}")
    return BoundaryError(message)
