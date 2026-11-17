# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native transverse edge's launch readings, record and verdicts, for solprop.

solprop SEP 47 section 6.4 and the maintainer's decision 4 at its acceptance: the
native session owns a free-space run's transverse edge (SEP 40 section 5.3) with every
decision and record (``transverse::Edge``,
``native/include/solprop_native/session/transverse.hpp``). The edge's configuration and
a resume's continuation are solprop's run records (:mod:`solprop.backends.records`, SEP
52), which the extension's record converter (``native/src/bindings/records.hpp``) reads.
This module only translates the rest: it forms the input noise's readings a launch
carries, formats the native record as
:meth:`~solprop.transverse_monitor.TransverseEdge.summary` lays it out, and formats a
native verdict as the edge's warning or exception with the edge's own texts. It decides
nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
from solprop.backends.records import STATUSES, EdgeNotice, EdgeNoticeKind
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.transverse_monitor import (
    _clip_message,  # pyright: ignore[reportPrivateUsage] - the edge's texts
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

if TYPE_CHECKING:
    from solprop.transverse_monitor import TransverseEdge

__all__ = [
    "edge_notice_of",
    "log_declared",
    "log_finish",
    "noise_readings",
    "notice_error",
    "notice_warning",
    "record_of",
]


def noise_readings(
    edge: TransverseEdge, noise: np.ndarray, clean: np.ndarray | None
) -> dict[str, object]:
    """Return the input noise's readings a native launch carries, formed as solprop's.

    :meth:`~solprop.transverse_monitor.TransverseEdge.launch` reads its input noise on
    the host with its own reductions: each column's energy, the weighted node energy
    its allowance takes, the weighted columns' energy in the safety ramp's band and a
    requested collar's strip. They are formed here from the same draw, the same way,
    once (``transverse::Noise``), so the native launch judges the same numbers.

    - ``noise``: the input noise's draw, shape ``(n_modes, n_pol, n_omega)``, a host
      array.
    - ``clean``: each column's unweighted launch energy without that noise
      (``NativeBackend._launch_clean_columns``), or ``None``.
    - Returns: the readings, as ``transverse::Noise`` takes them.
    """
    reduction = edge.reduction
    columns = reduction.column_power(noise)
    weights = edge._weights  # pyright: ignore[reportPrivateUsage]
    design = edge.design
    ramp = 0.0
    if design is not None:
        power = edge._column_power(columns)  # pyright: ignore[reportPrivateUsage]
        ramp = float(np.sum(power[design.ramp_rate > 0.0]))
    strip = np.empty(0, dtype=np.float64)
    if edge.every_step:
        strip = np.ascontiguousarray(reduction.strip(noise), dtype=np.float64).reshape(
            -1
        )
    return {
        "columns": np.ascontiguousarray(columns, dtype=np.float64),
        "energy": float(np.sum(weights * reduction.node_power(noise))),
        "ramp": ramp,
        "strip": strip,
        "clean": None
        if clean is None
        else np.ascontiguousarray(clean, dtype=np.float64),
    }


def record_of(edge: TransverseEdge, native: Mapping[str, object]) -> dict[str, object]:
    """Format the native edge's record as :meth:`TransverseEdge.summary` does.

    - ``native``: ``_core``'s record: every number, NaN where nothing was measured.
    - Returns: the record, with ``None`` for what was never measured.
    """

    def number(name: str) -> float:
        """Return a recorded number."""
        return float(cast("float", native[name]))

    def numbers(name: str) -> list[float]:
        """Return a recorded per-face list."""
        return [float(one) for one in cast("list[float]", native[name])]

    launch = number("launch_energy")
    warned_at = number("warned_at")
    allowance = number("noise_allowance")
    reasons = ("", _insensitive_reason(allowance))
    return {
        "launch_energy": launch if math.isfinite(launch) else None,
        "noise_ramp_removed": number("noise_ramp_removed"),
        "noise_share_in_ramp": number("noise_share_in_ramp"),
        "declared_share_in_ramp": number("declared_share_in_ramp"),
        "transverse_status": STATUSES[int(cast("int", native["transverse_status"]))],
        "transverse_reason": reasons[int(cast("int", native["transverse_reason"]))],
        "faces": [
            face.name
            for face in edge._faces  # pyright: ignore[reportPrivateUsage]
        ],
        "edge_baseline": numbers("edge_baseline"),
        "edge_max": numbers("edge_max"),
        "edge_max_raw": numbers("edge_max_raw"),
        "noise_allowance": allowance,
        "ramp_removed": number("ramp_removed"),
        "raw_ramp_removed": number("raw_ramp_removed"),
        "spatial_removed": number("spatial_removed"),
        "reads": int(cast("int", native["reads"])),
        "returned_max": number("returned_max"),
        "strip_max": number("strip_max"),
        "strip_baseline": number("strip_baseline"),
        "strip_floor": number("strip_floor"),
        "current": number("current"),
        "current_baseline": number("current_baseline"),
        "largest_gap": number("largest_gap"),
        "warned_at": None if math.isnan(warned_at) else warned_at,
    }


def edge_notice_of(native: Mapping[str, object]) -> EdgeNotice:
    """Return the edge notice ``_core`` reports (``SessionEvent.notice["edge"]``)."""
    return EdgeNotice(
        EdgeNoticeKind(int(cast("int", native["kind"]))),
        bool(native["stop"]),
        int(cast("int", native["face"])),
        float(cast("float", native["value"])),
        float(cast("float", native["z"])),
        float(cast("float", native["spacing"])),
        float(cast("float", native["warned_at"])),
    )


def _face(edge: TransverseEdge, notice: EdgeNotice) -> str:
    """Return the name of the face a notice names."""
    return edge._faces[notice.face].name  # pyright: ignore[reportPrivateUsage]


def notice_warning(edge: TransverseEdge, notice: EdgeNotice) -> BoundaryWarning:
    """Return a native edge's warning, as the edge issues it."""
    kind = notice.kind
    if notice.stop or kind is EdgeNoticeKind.NONE:
        msg = f"the native transverse edge warned with a {kind.name.lower()} notice"
        raise BackendError(msg)
    if kind is EdgeNoticeKind.RAMP_BAND:
        text = _ramp_band_message(notice.value, stop=False)
    elif kind is EdgeNoticeKind.CLIP:
        text = _clip_message(notice.value, _face(edge, notice))
    elif kind is EdgeNoticeKind.INSENSITIVE:
        text = _insensitive_message(notice.value)
    elif kind is EdgeNoticeKind.COARSE:
        text = _coarse_message(notice.value, notice.spacing)
    elif kind is EdgeNoticeKind.FACE:
        text = _face_message(_face(edge, notice), notice.z, notice.value)
    elif kind is EdgeNoticeKind.RETURNED:
        text = _return_message(notice.value, notice.z, None)
    else:
        text = _ramp_removal_message(notice.value, None)
    return BoundaryWarning(text)


def notice_error(edge: TransverseEdge, notice: EdgeNotice) -> BaseException:
    """Return the exception a native edge's stop reports, as the edge raises it."""
    stop = (edge.stop_factor * edge.budget) ** 2
    kind = notice.kind
    if not notice.stop:
        return BackendError(
            f"the native transverse edge stopped on a {kind.name.lower()} warning"
        )
    if kind is EdgeNoticeKind.RAMP_BAND:
        return BoundaryError(_ramp_band_message(notice.value, stop=True))
    if kind is EdgeNoticeKind.FACE:
        return BoundaryError(
            _stop_message(
                edge.space,
                edge.length,
                stop,
                _face(edge, notice),
                notice.value,
                z=notice.z,
                warned_at=notice.warned_at,
            )
        )
    if kind is EdgeNoticeKind.RETURNED:
        return BoundaryError(_return_message(notice.value, notice.z, stop))
    if kind is EdgeNoticeKind.RAMP_REMOVAL:
        return BoundaryError(_ramp_removal_message(notice.value, stop))
    return BackendError(
        f"the native transverse edge stopped on a {kind.name.lower()} notice"
    )


def log_finish(native: Mapping[str, object]) -> None:
    """Log the native edge's verdict at the end of a run, as the edge logs it.

    - ``native``: ``_core``'s record, with its ``finish``.
    """
    finish = cast("Mapping[str, object]", native["finish"])
    _log_finish(
        STATUSES[int(cast("int", finish["status"]))],
        float(cast("float", finish["largest"])),
        float(cast("float", finish["ramp_removed"])),
        int(cast("int", finish["reads"])),
    )


def log_declared(native: Mapping[str, object]) -> bool:
    """Log a declared launch's share in the ramp's band once it was launched.

    As :meth:`TransverseEdge.launch` logs it, for a launch it judged: one with energy,
    not resumed.

    - ``native``: ``_core``'s record.
    """
    launch = float(cast("float", native["launch_energy"]))
    if math.isnan(launch) and not native["resumed"]:
        return False
    if launch > 0.0 and not native["resumed"]:
        _log_declared(float(cast("float", native["declared_share_in_ramp"])))
    return True
