# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The catalogue's routes on the Runge-Kutta march, adaptive or replayed.

``tests/driver/test_native_rk_march.py`` runs each route of
``capability_routes.ROUTES`` on the native Runge-Kutta march; :func:`replayed` puts one
on a uniform fixed schedule landing on its save planes, which the catalogue's ``rk
replay ...`` rows (``tests/backends/driver_rows.py``) run on native against
``numpy_ref``.
"""

import dataclasses
import warnings
from typing import Any

import numpy as np

from capability_routes import ROUTES
from solprop.driver import Simulation
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule

UNCHECKED = {"general guide (moving)"}
"""Routes whose window the launch check refuses when run whole: built to translate."""


def runge_kutta_route(route: str, **options: Any) -> Simulation:
    """Return a catalogue route on the Runge-Kutta march, on native.

    `options` replace any of its settings; the moving guide runs without its boundary
    check (:data:`UNCHECKED`).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
    settings = dataclasses.replace(
        simulation.settings, integrator=RungeKutta(), backend="native", **options
    )
    if route in UNCHECKED:
        settings = dataclasses.replace(settings, boundary_check=None)
    simulation.settings = settings
    return simulation


def replayed(route: str, steps: int) -> Simulation:
    """Return a route on the Runge-Kutta march, replaying `steps` uniform steps.

    The schedule lands on the route's own save planes.
    """
    stepper = runge_kutta_route(route)._run_request().stepper
    end = float(stepper.z_end)
    schedule = FixedSchedule.uniform(
        0.0, end, end / steps, save_planes=np.asarray(stepper.save_planes).tolist()
    )
    return runge_kutta_route(route, schedule=schedule)
