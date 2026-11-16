# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""solprop's run records as ``_core`` reads them (solprop SEP 52 section 5.1).

solprop forms each run's records once (:func:`solprop.backends.records.run_records`):
the method's policy, the low-storage settings, the schedule, the plane plan and the
boundary monitor's configuration and continuation. This module only converts them into
the dictionaries and arrays the extension's configuration calls take; it derives
nothing.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from solprop.backends.records import (
        EdgeConfig,
        EdgeResumption,
        MonitorConfig,
        MonitorContinuation,
        PlanePlan,
        Resumption,
        RungeKuttaPolicy,
    )
    from solprop.stepper.multistep import MarchPolicy
    from solprop.stepper.schedule import FixedSchedule

__all__ = [
    "NO_SCHEDULE",
    "adams_policy",
    "continuation",
    "edge_config",
    "edge_resumption",
    "monitor_config",
    "resumption",
    "rk_policy",
    "schedule_arrays",
    "session_plan",
]

NO_SCHEDULE = np.empty(0, dtype=np.float64)
"""The empty schedule of an adaptive march's configuration."""


def adams_policy(policy: MarchPolicy) -> dict[str, object]:
    """Return the Adams march's policy as ``adams_march_configure`` reads it."""
    return dataclasses.asdict(policy)


def rk_policy(policy: RungeKuttaPolicy) -> dict[str, object]:
    """Return the Runge-Kutta march's policy as ``rk_march_configure`` reads it.

    The controller's fields flattened in: its gains, factors, lag, acceptance rule and
    the equivalent norm's power and exponent (the kernel contract's section 12).
    """
    controller = policy.controller
    return {
        "dz_min": policy.dz_min,
        "dz_max": policy.dz_max,
        "stretch": policy.stretch,
        "z_end": policy.z_end,
        "save_planes": list(policy.save_planes),
        "interpolate": policy.interpolate,
        "hard_kinks": list(policy.hard_kinks),
        "breakpoints": list(policy.breakpoints),
        "landing_ulps": policy.landing_ulps,
        "quiet_interval": policy.quiet_interval,
        "safety": float(controller.safety),
        "beta1": float(controller.beta1),
        "beta2": float(controller.beta2),
        "min_factor": float(controller.min_factor),
        "max_factor": float(controller.max_factor),
        "lag": int(controller.lag),
        "reject_free": bool(controller.reject_free),
        "power": float(controller.estimator_order + 1),
        "exponent": 1.0 / (controller.estimator_order + 1),
        "start_growth": policy.start_growth,
        "auto_init": policy.auto_init,
    }


def schedule_arrays(schedule: FixedSchedule | None) -> tuple[np.ndarray, np.ndarray]:
    """Return a replay's steps and edges, m, or an adaptive run's empty schedule."""
    if schedule is None:
        return NO_SCHEDULE, NO_SCHEDULE
    return (
        np.ascontiguousarray(schedule.steps, dtype=np.float64),
        np.ascontiguousarray(schedule.edges, dtype=np.float64),
    )


def session_plan(
    planes: PlanePlan,
    *,
    plane_statistics: object,
    row_statistics: object,
    monitor: dict[str, object] | None,
    dense: dict[str, object] | None,
) -> dict[str, object]:
    """Return the observation request and itinerary ``start_session`` reads.

    - ``plane_statistics``, ``row_statistics``: the lowered statistics plans
      (``_core.StatisticsPlan``), each ``None`` for none.
    - ``monitor``: the monitor's plan, or ``None`` without a monitor.
    - ``dense``: the dense output's weights, or ``None`` without dense output.
    """
    return {
        "planes": planes.planes,
        "next_plane": planes.next_plane,
        "launch_pending": planes.launch_pending,
        "step_planes": planes.step_planes,
        "plane_statistics": plane_statistics,
        "row_statistics": row_statistics,
        "period": planes.period,
        "attempt_capacity": planes.attempt_capacity,
        "row_capacity": planes.row_capacity,
        "replay": planes.replay_steps > 0,
        "replay_steps": planes.replay_steps,
        "replay_length": planes.replay_length,
        "call_attempts": planes.call_attempts,
        "call_seconds": planes.call_seconds,
        "monitor": monitor,
        "dense": dense,
    }


def monitor_config(config: MonitorConfig) -> dict[str, object]:
    """Return the monitor's configuration as ``watch::Config`` reads it."""
    return {
        "policy": dict(config.policy),
        "floor_share": config.floor_share,
        "floor_margin": config.floor_margin,
        "budget": config.budget,
        "rate": config.rate,
        "can_rest": config.can_rest,
    }


def edge_config(config: EdgeConfig) -> dict[str, object]:
    """Return the edge's configuration as ``transverse::Config`` reads it.

    Its fields by name, the arrays as they are (``dataclasses.asdict`` would copy them).
    """
    values = {one.name: getattr(config, one.name) for one in dataclasses.fields(config)}
    values["faces"] = list(config.faces)
    return values


def resumption(held: Resumption) -> dict[str, object]:
    """Return a monitor record's resumption: numbers flattened in, no absent list."""
    out: dict[str, object] = {
        "record": held.record,
        "collar_quiet": held.collar_quiet,
        **held.numbers,
    }
    if held.unwatched is not None:
        out["unwatched"] = [list(stretch) for stretch in held.unwatched]
    if held.resumed_at is not None:
        out["resumed_at"] = list(held.resumed_at)
    if held.planes is not None:
        out["planes"] = [list(column) for column in held.planes]
    return out


def edge_resumption(held: EdgeResumption) -> dict[str, object]:
    """Return an edge record's resumption: the numbers flattened in, no absent field."""
    out: dict[str, object] = {"record": held.record}
    if not held.record:
        return out
    out.update(held.numbers)
    for name in ("edge_baseline", "edge_max", "edge_max_raw"):
        values = getattr(held, name)
        if values is not None:
            out[name] = list(values)
    if held.reads is not None:
        out["reads"] = held.reads
    if held.transverse_status is not None:
        out["transverse_status"] = held.transverse_status
    return out


def continuation(held: MonitorContinuation) -> dict[str, object]:
    """Return a monitored resume's continuation as the native session reads it.

    - Returns: the earlier record's resumption and its transverse edge's, whether the
      run resumes at a saved plane, and the stretches run since without a monitor.
    """
    return {
        "edge_record": edge_resumption(held.edge),
        "record": resumption(held.record),
        "plane": held.plane,
        "unwatched": [[start, end] for start, end in held.unwatched],
    }
