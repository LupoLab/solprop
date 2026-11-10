# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's transverse edge, as solprop configures and reads it.

solprop SEP 47 section 6.4 and the maintainer's decision 4 at its acceptance: the
native session owns a free-space run's transverse edge (SEP 40 section 5.3) with every
decision and record (``transverse::Edge``,
``native/include/solprop_native/transverse.hpp``). This module only translates: it
forms the edge's configuration from solprop's
:class:`~solprop.transverse_monitor.TransverseEdge` and the problem, reads an earlier
execution's ``transverse`` record into what a resume restores, formats the native
record as :meth:`~solprop.transverse_monitor.TransverseEdge.summary` lays it out, and
formats a native verdict as the edge's warning or exception with the edge's own texts.
It decides nothing.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from solprop.backends.driver import (
    _declared_discard,  # pyright: ignore[reportPrivateUsage] - one predicate
)
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.spatial_absorber import RAMP_K_FLOOR
from solprop.transverse_monitor import (
    _UNWATCHED_MESSAGE,  # pyright: ignore[reportPrivateUsage] - the edge's texts
    _UNWATCHED_REASON,  # pyright: ignore[reportPrivateUsage]
    BACKSTOP_READS,
    NOISE_FACTOR,
    REACH_GROWTH,
    READS,
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

from ._watch import _number  # pyright: ignore[reportPrivateUsage] - one reading

if TYPE_CHECKING:
    from solprop.translate.problem import PropagationProblem, StepperSettings
    from solprop.transverse_monitor import TransverseEdge, TransverseStatus

__all__ = [
    "STATUSES",
    "EdgeConfig",
    "EdgeNotice",
    "NoticeKind",
    "config_of",
    "declared",
    "edge_config",
    "log_declared",
    "log_finish",
    "noise_readings",
    "notice_error",
    "notice_warning",
    "record_of",
    "resumption",
]

STATUSES: tuple[TransverseStatus, ...] = (
    "clear",
    "warned",
    "stopped",
    "insensitive",
    "coarse",
    "unwatched",
)
"""The transverse statuses in the native order (``transverse::Status``)."""


class NoticeKind(enum.IntEnum):
    """A native edge notice's kind (``transverse::Notice::Kind``).

    Attributes
    ----------
    NONE : int
        No notice.
    RAMP_BAND : int
        The launch's share in the safety ramp's band.
    UNWATCHED : int
        The backend does not read the edge.
    CLIP : int
        A face's clip of the launch.
    INSENSITIVE : int
        The input noise's allowance exceeds the warning.
    COARSE : int
        A fixed schedule's longest step outruns the read spacing.
    FACE : int
        A face's judged share.
    RETURNED : int
        A requested collar's returned share.
    RAMP_REMOVAL : int
        The safety ramp's removal.
    """

    NONE = 0
    RAMP_BAND = 1
    UNWATCHED = 2
    CLIP = 3
    INSENSITIVE = 4
    COARSE = 5
    FACE = 6
    RETURNED = 7
    RAMP_REMOVAL = 8


_NUMBERS = (
    "launch_energy",
    "noise_allowance",
    "raw_ramp_removed",
    "noise_ramp_removed",
    "noise_share_in_ramp",
    "declared_share_in_ramp",
    "spatial_removed",
    "current",
    "current_baseline",
    "strip_baseline",
    "strip_floor",
    "strip_max",
    "returned_max",
    "largest_gap",
    "warned_at",
)
"""The numbers a resume reads from an earlier record (``transverse::Resumption``)."""


@dataclass(frozen=True, eq=False)
class EdgeConfig:
    """The native edge's configuration (``transverse::Config``).

    Attributes
    ----------
    values : collections.abc.Mapping
        Each field, as ``_core`` reads it.
    """

    values: Mapping[str, object]

    def native(self) -> dict[str, object]:
        """Return the configuration as ``_core`` reads it.

        Returns
        -------
        dict
            Its fields.
        """
        return dict(self.values)


def declared(problem: PropagationProblem) -> bool:
    """Return whether a run declares content above ``discard_above``.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.

    Returns
    -------
    bool
        True when its transverse specification sets ``discard_above``.
    """
    return _declared_discard(problem)


def edge_config(
    edge: TransverseEdge, problem: PropagationProblem, stepper: StepperSettings
) -> EdgeConfig:
    """Return the native edge's configuration for a run's fresh edge.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's fresh edge; it is not changed.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem: its declared content, as ``StepDriver.attach_monitor``
        hands it to the edge's launch.
    stepper : solprop.translate.problem.StepperSettings
        The run's stepper section: its fixed schedule, if any.

    Returns
    -------
    EdgeConfig
        The configuration.
    """
    schedule = stepper.schedule
    steps = None if schedule is None else np.asarray(schedule.steps)
    return config_of(edge, steps=steps, declared=declared(problem))


def config_of(
    edge: TransverseEdge,
    *,
    steps: np.ndarray | None,
    declared: bool,
    reads: bool = True,
) -> EdgeConfig:
    """Return the native edge's configuration for an edge and its launch's arguments.

    Every threshold is formed here as the edge forms it, so that the native edge
    compares the same numbers: ``budget ** 2``, ``(stop_factor * budget) ** 2``, the
    longest fixed step, and the input noise's share of the weights and count of modes.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        A fresh edge; it is not changed.
    steps : numpy.ndarray or None
        A fixed schedule's steps, as :meth:`TransverseEdge.launch` takes them.
    declared : bool
        Whether content above ``discard_above`` was declared.
    reads : bool, optional
        Whether the backend reads the edge; every native session does.

    Returns
    -------
    EdgeConfig
        The configuration.
    """
    design = edge.design
    faces = edge._faces  # pyright: ignore[reportPrivateUsage]
    weights = edge._weights  # pyright: ignore[reportPrivateUsage]
    first = faces[0].edge
    modes = max(int(np.count_nonzero(first)), 1) * max(int(edge.band.size), 1)
    return EdgeConfig(
        {
            "faces": [face.name for face in faces],
            "warn": edge.budget**2,
            "stop": (edge.stop_factor * edge.budget) ** 2,
            "length": float(edge.length),
            "guard": float(edge._guard),  # pyright: ignore[reportPrivateUsage]
            "k_min": float(edge.k_min),
            "k_floor": float(RAMP_K_FLOOR),
            "reads": float(READS),
            "backstop_reads": float(BACKSTOP_READS),
            "reach_growth": float(REACH_GROWTH),
            "noise_factor": float(NOISE_FACTOR),
            "columns": np.ascontiguousarray(
                edge._columns,  # pyright: ignore[reportPrivateUsage]
                dtype=np.float64,
            ),
            "kappa": np.ascontiguousarray(
                edge._kappa,  # pyright: ignore[reportPrivateUsage]
                dtype=np.float64,
            ),
            "kappa_order": np.ascontiguousarray(
                edge._kappa_order,  # pyright: ignore[reportPrivateUsage]
                dtype=np.uint64,
            ),
            "design": design is not None,
            "ramp_rate": np.empty(0, dtype=np.float64)
            if design is None
            else np.ascontiguousarray(design.ramp_rate, dtype=np.float64),
            "collar": design is not None and design.has_collar,
            "collar_rate": design is not None and design.collar_rate is not None,
            "declared": bool(declared),
            "reads_edge": bool(reads),
            "longest_step": float(np.max(steps))
            if steps is not None and steps.size
            else None,
            "noise_share": float(np.sum(weights[first])) / float(np.sum(weights)),
            "noise_modes": float(modes),
        }
    )


def noise_readings(
    edge: TransverseEdge, noise: np.ndarray, clean: np.ndarray | None
) -> dict[str, object]:
    """Return the input noise's readings a native launch carries, formed as solprop's.

    :meth:`~solprop.transverse_monitor.TransverseEdge.launch` reads its input noise on
    the host with its own reductions: each column's energy, the weighted node energy
    its allowance takes, the weighted columns' energy in the safety ramp's band and a
    requested collar's strip. They are formed here from the same draw, the same way,
    once (``transverse::Noise``), so the native launch judges the same numbers.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's fresh edge; it is not changed.
    noise : numpy.ndarray
        The input noise's draw, shape ``(n_modes, n_pol, n_omega)``, a host array.
    clean : numpy.ndarray or None
        Each column's unweighted launch energy without that noise
        (``StepDriver._edge_clean_columns``), or ``None``.

    Returns
    -------
    dict
        The readings, as ``transverse::Noise`` takes them.
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


def resumption(record: object, faces: int) -> dict[str, object]:
    """Return what the native resume reads of an earlier execution's edge record.

    Only the record's types are read here, as :meth:`TransverseEdge.resume` reads
    them: each number present, and, once the record's launch energy and baseline make
    that resume read further, each per-face maximum that has a value per face and the
    reads, made whole as it makes them and refused as it refuses them. Which fields are
    restored, their defaults and the status a resume keeps are native.

    Parameters
    ----------
    record : object
        The ``transverse`` entry of the earlier monitor record, or anything else when
        there is none.
    faces : int
        The edge's faces.

    Returns
    -------
    dict
        The resumption, as ``_core`` reads it.

    Raises
    ------
    ValueError
        For reads that are not a number Python's ``int`` takes, as solprop's resume.
    OverflowError
        Likewise, for infinite reads.
    """
    if not isinstance(record, Mapping):
        return {"record": False}
    mapping = cast("Mapping[str, object]", record)
    out: dict[str, object] = {"record": True}
    for name in _NUMBERS:
        value = _number(mapping, name)
        if value is not None:
            out[name] = value
    baseline = _per_face(mapping, "edge_baseline", faces)
    launch = cast("float", out.get("launch_energy", math.nan))
    if launch > 0.0 and baseline is not None:
        # `TransverseEdge.resume` reads the maxima and the count only past its test
        # of the launch and the baseline: a record it ignores is not read further.
        out["edge_baseline"] = baseline
        for name in ("edge_max", "edge_max_raw"):
            value = _per_face(mapping, name, faces)
            if value is not None:
                out[name] = value
        reads = _number(mapping, "reads")
        out["reads"] = int(0.0 if reads is None else reads)
    status = mapping.get("transverse_status")
    if isinstance(status, str):
        out["transverse_status"] = status
    return out


def _per_face(
    record: Mapping[str, object], name: str, faces: int
) -> list[float] | None:
    """Return a record's per-face list, or ``None`` when it has no value per face.

    As ``TransverseEdge.resume``'s ``numbers`` reads it.

    Parameters
    ----------
    record : collections.abc.Mapping
        The edge's record.
    name : str
        The field.
    faces : int
        The edge's faces.

    Returns
    -------
    list of float or None
        One value per face.
    """
    value = record.get(name)
    if not isinstance(value, list) or len(cast("list[object]", value)) != faces:
        return None
    return [float(cast("float", one)) for one in cast("list[object]", value)]


def record_of(edge: TransverseEdge, native: Mapping[str, object]) -> dict[str, object]:
    """Format the native edge's record as :meth:`TransverseEdge.summary` does.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's edge, which names the faces.
    native : collections.abc.Mapping
        ``_core``'s record: every number, NaN where nothing was measured.

    Returns
    -------
    dict
        The record, with ``None`` for what was never measured.
    """

    def number(name: str) -> float:
        """Return a recorded number.

        Parameters
        ----------
        name : str
            The field.

        Returns
        -------
        float
            The value.
        """
        return float(cast("float", native[name]))

    def numbers(name: str) -> list[float]:
        """Return a recorded per-face list.

        Parameters
        ----------
        name : str
            The field.

        Returns
        -------
        list of float
            One value per face.
        """
        return [float(one) for one in cast("list[float]", native[name])]

    launch = number("launch_energy")
    warned_at = number("warned_at")
    allowance = number("noise_allowance")
    reasons = ("", _UNWATCHED_REASON, _insensitive_reason(allowance))
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


@dataclass(frozen=True)
class EdgeNotice:
    """A native edge's verdict (``transverse::Notice``): a warning or a stop.

    Attributes
    ----------
    kind : NoticeKind
        What it is.
    stop : bool
        Whether it stops the run.
    face : int
        The face it names, by index.
    value : float
        The share, allowance or step it names.
    z : float
        Where, m.
    spacing : float
        A coarse schedule's read spacing, m.
    warned_at : float
        Where the edge first warned, for a face's stop, m; NaN where it did not.
    """

    kind: NoticeKind
    stop: bool
    face: int
    value: float
    z: float
    spacing: float
    warned_at: float

    @classmethod
    def of(cls, native: Mapping[str, object]) -> EdgeNotice:
        """Return the notice ``_core`` reports.

        Parameters
        ----------
        native : collections.abc.Mapping
            A notice's dictionary.

        Returns
        -------
        EdgeNotice
            The notice.
        """
        return cls(
            NoticeKind(int(cast("int", native["kind"]))),
            bool(native["stop"]),
            int(cast("int", native["face"])),
            float(cast("float", native["value"])),
            float(cast("float", native["z"])),
            float(cast("float", native["spacing"])),
            float(cast("float", native["warned_at"])),
        )


def _face(edge: TransverseEdge, notice: EdgeNotice) -> str:
    """Return the name of the face a notice names.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's edge.
    notice : EdgeNotice
        The notice.

    Returns
    -------
    str
        The face's name.
    """
    return edge._faces[notice.face].name  # pyright: ignore[reportPrivateUsage]


def notice_warning(edge: TransverseEdge, notice: EdgeNotice) -> BoundaryWarning:
    """Return a native edge's warning, as the edge issues it.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's edge.
    notice : EdgeNotice
        The warning's verdict.

    Returns
    -------
    solprop.exceptions.BoundaryWarning
        The warning.

    Raises
    ------
    solprop.exceptions.BackendError
        For a stop, or a kind that does not warn.
    """
    kind = notice.kind
    if notice.stop or kind is NoticeKind.NONE:
        msg = f"the native transverse edge warned with a {kind.name.lower()} notice"
        raise BackendError(msg)
    if kind is NoticeKind.RAMP_BAND:
        text = _ramp_band_message(notice.value, stop=False)
    elif kind is NoticeKind.UNWATCHED:
        text = _UNWATCHED_MESSAGE
    elif kind is NoticeKind.CLIP:
        text = _clip_message(notice.value, _face(edge, notice))
    elif kind is NoticeKind.INSENSITIVE:
        text = _insensitive_message(notice.value)
    elif kind is NoticeKind.COARSE:
        text = _coarse_message(notice.value, notice.spacing)
    elif kind is NoticeKind.FACE:
        text = _face_message(_face(edge, notice), notice.z, notice.value)
    elif kind is NoticeKind.RETURNED:
        text = _return_message(notice.value, notice.z, None)
    else:
        text = _ramp_removal_message(notice.value, None)
    return BoundaryWarning(text)


def notice_error(edge: TransverseEdge, notice: EdgeNotice) -> BaseException:
    """Return the exception a native edge's stop reports, as the edge raises it.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's edge.
    notice : EdgeNotice
        The stop's verdict.

    Returns
    -------
    BaseException
        The :class:`~solprop.exceptions.BoundaryError`; a
        :class:`~solprop.exceptions.BackendError` for a notice that does not stop.
    """
    stop = (edge.stop_factor * edge.budget) ** 2
    kind = notice.kind
    if not notice.stop:
        return BackendError(
            f"the native transverse edge stopped on a {kind.name.lower()} warning"
        )
    if kind is NoticeKind.RAMP_BAND:
        return BoundaryError(_ramp_band_message(notice.value, stop=True))
    if kind is NoticeKind.FACE:
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
    if kind is NoticeKind.RETURNED:
        return BoundaryError(_return_message(notice.value, notice.z, stop))
    if kind is NoticeKind.RAMP_REMOVAL:
        return BoundaryError(_ramp_removal_message(notice.value, stop))
    return BackendError(
        f"the native transverse edge stopped on a {kind.name.lower()} notice"
    )


def log_finish(native: Mapping[str, object]) -> None:
    """Log the native edge's verdict at the end of a run, as the edge logs it.

    Parameters
    ----------
    native : collections.abc.Mapping
        ``_core``'s record, with its ``finish``.
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

    Parameters
    ----------
    native : collections.abc.Mapping
        ``_core``'s record.

    Returns
    -------
    bool
        Whether the edge has launched, so that there is nothing left to log.
    """
    launch = float(cast("float", native["launch_energy"]))
    if math.isnan(launch) and not native["resumed"]:
        return False
    if launch > 0.0 and not native["resumed"]:
        _log_declared(float(cast("float", native["declared_share_in_ramp"])))
    return True
