# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which loop steps a native Runge-Kutta run (solprop SEP 45 phase 3).

The native march steps an adaptive run on staged rows with ``PIController``'s norm, and
the run's metadata records which loop stepped it; no switch selects solprop's loop any
more (solprop SEP 47 section 6.3 retired the test interface with the transitional
bridge).
"""

import dataclasses
import warnings

import pytest

from capability_routes import ROUTES
from sessions.executors import committed_steps
from sessions.runs import run
from solprop.backends import available
from solprop.backends.events import Outcome
from solprop.exceptions import HostMarchWarning
from solprop.stepper.runge_kutta import RungeKutta

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")


ROUTES_NATIVE = (
    "gnlse",
    "capillary (multimode)",
    "radial",
    "capillary (coherent wall)",
)
FLOWS = ("capillary (coherent wall)",)


def launched(route: str, **settings):
    """Return a route's backend on the Runge-Kutta march, its problem and request."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=RungeKutta(), **settings
        )
        problem = simulation.translate()
        run_request = simulation._run_request()
    backend = native.NativeBackend(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    return backend, problem, run_request


@pytest.mark.parametrize(
    ("route", "dense"),
    [
        *((route, False) for route in ROUTES_NATIVE),
        # Translation refuses interpolated planes beside a coupled flow.
        *((route, True) for route in ROUTES_NATIVE if route not in FLOWS),
    ],
)
def test_the_default_loop_is_native_without_a_warning(route, dense):
    """The backend's session steps the whole run natively, warning of no host march."""
    backend, problem, run_request = launched(route, interpolate_planes=dense)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", HostMarchWarning)
            landed = run(backend, problem, run_request)
        assert landed.terminal.outcome is Outcome.COMPLETED
        assert committed_steps(landed) > 0
        assert backend.metadata()["rk_march"] == "native"
    finally:
        backend.close()
