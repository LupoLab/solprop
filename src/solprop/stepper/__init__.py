# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The step specifications a run carries, and the scalar algebra of every step.

The frozen specifications a :class:`~solprop.translate.run_request.RunRequest` carries,
plus the backend-agnostic scalar logic every backend runs per step
(``docs/dev/stepper.md``); applying a method to a state is a backend kernel. The
package imports only :mod:`solprop.exceptions` and the private ``solprop._helpers`` and
``solprop._ownership``.

- Integrators: :class:`~solprop.stepper.runge_kutta.RungeKutta`
  (:mod:`~solprop.stepper.runge_kutta`, an embedded interaction-picture pair),
  :class:`~solprop.stepper.multistep.Multistep` (:mod:`~solprop.stepper.multistep`, the
  low-call Adams family and its variable-step weights) and
  :class:`~solprop.stepper.low_storage.LowStorageRungeKutta`
  (:mod:`~solprop.stepper.low_storage`, the fixed-schedule RK4IP replay).
- :mod:`~solprop.stepper.tableau`: :class:`~solprop.stepper.tableau.Tableau`, the
  ``DOPRI5`` and ``TSIT5`` pairs and
  :func:`~solprop.stepper.tableau.check_order_conditions`.
- :mod:`~solprop.stepper.controller`: the error-norm scales and
  :class:`~solprop.stepper.controller.PIController`.
- :mod:`~solprop.stepper.schedule`: :class:`~solprop.stepper.schedule.FixedSchedule`
  (:func:`~solprop.stepper.schedule.discover`,
  :func:`~solprop.stepper.schedule.calibrate`, :func:`~solprop.stepper.schedule.replay`)
  and :func:`~solprop.stepper.schedule.landing` on save planes and kinks.
- :mod:`~solprop.stepper.settings`: :class:`~solprop.stepper.settings.StepperSettings`,
  the stepper section a run request carries (method, controller, step limits, schedule
  and planes), and :data:`~solprop.stepper.settings.PropagatorMode`.
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
from solprop.stepper.settings import StepperSettings
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
    "StepperSettings",
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
