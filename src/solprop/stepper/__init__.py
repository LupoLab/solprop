# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The step specifications a run carries, and the scalar algebra of every step.

The frozen specifications a :class:`~solprop.translate.run_request.RunRequest` carries,
plus the backend-agnostic scalar logic every backend runs per step
(``docs/dev/stepper.md``); applying a method to a state is a backend kernel. The
package imports only :mod:`solprop.exceptions` and :mod:`solprop._helpers`.

- Integrators: :class:`RungeKutta` (:mod:`~solprop.stepper.runge_kutta`, an embedded
  interaction-picture pair), :class:`Multistep` (:mod:`~solprop.stepper.multistep`,
  the low-call Adams family and its variable-step weights) and
  :class:`LowStorageRungeKutta` (:mod:`~solprop.stepper.low_storage`, the fixed-schedule
  RK4IP replay).
- :mod:`~solprop.stepper.tableau`: :class:`Tableau`, the ``DOPRI5`` and ``TSIT5``
  pairs and :func:`check_order_conditions`.
- :mod:`~solprop.stepper.controller`: the error-norm scales and :class:`PIController`.
- :mod:`~solprop.stepper.schedule`: :class:`FixedSchedule` (:func:`discover`,
  :func:`calibrate`, :func:`replay`) and :func:`landing` on save planes and kinks.
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
