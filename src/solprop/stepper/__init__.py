# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The integrators, Runge-Kutta tableaux, the step controller and step schedules.

Backend-agnostic scalar logic of the solver (``docs/dev/stepper.md``).
"""

from solprop.stepper.controller import (
    DZ_MIN_RELATIVE,
    ControlRecord,
    PIController,
    default_dz_min,
    limit_step,
    weak_norm_scale,
    weighted_norm_scale,
)
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import (
    CONCLUSIVE_FACTOR,
    ORDER_MARGIN,
    Calibration,
    FixedSchedule,
    Landing,
    LandingTarget,
    StepTargets,
    calibrate,
    discover,
    landing,
    observed_order,
    replay,
)
from solprop.stepper.tableau import (
    DOPRI5,
    MAX_CHECKED_ORDER,
    TSIT5,
    OrderConditionReport,
    Tableau,
    check_order_conditions,
)

__all__ = [
    "CONCLUSIVE_FACTOR",
    "DOPRI5",
    "DZ_MIN_RELATIVE",
    "MAX_CHECKED_ORDER",
    "ORDER_MARGIN",
    "TSIT5",
    "Calibration",
    "ControlRecord",
    "FixedSchedule",
    "Landing",
    "LandingTarget",
    "LowStorageRungeKutta",
    "Multistep",
    "OrderConditionReport",
    "PIController",
    "RungeKutta",
    "StepTargets",
    "Tableau",
    "calibrate",
    "check_order_conditions",
    "default_dz_min",
    "discover",
    "landing",
    "limit_step",
    "observed_order",
    "replay",
    "weak_norm_scale",
    "weighted_norm_scale",
]
