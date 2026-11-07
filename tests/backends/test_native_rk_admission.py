# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which loop steps a native Runge-Kutta run (solprop SEP 45 phase 3).

The native march steps an adaptive run on staged rows with ``PIController``'s weak norm;
any other configuration runs solprop's loop with a ``HostMarchWarning`` that names the
reason, and the run's metadata records which loop stepped it. The private switch the
benchmark harness uses selects solprop's loop without a warning.
"""

import dataclasses
import warnings

import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.exceptions import HostMarchWarning
from solprop.stepper.controller import PIController
from solprop.stepper.runge_kutta import RungeKutta

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")


class _Scaled(PIController):
    """A controller whose scale is its own."""

    def scale(self, norm_y, norm_y_new):
        return 2.0 * super().scale(norm_y, norm_y_new)


def launched(route: str, **settings):
    """Return a route's backend on the Runge-Kutta march, launched."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=RungeKutta(), **settings
        )
        problem = simulation.translate()
    backend = native.NativeBackend(
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
    return backend


@pytest.mark.parametrize("route", ["gnlse", "capillary (multimode)", "radial"])
def test_the_default_loop_is_native_without_a_warning(route):
    backend = launched(route)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", HostMarchWarning)
            assert backend._rk_march() is not None
        assert backend.metadata()["rk_march"] == "native"
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("route", "settings", "reason"),
    [
        ("capillary (coherent wall)", {}, "coupled flow"),
        (
            "gnlse",
            {"controller": _Scaled(rtol=1e-6, atol=0.0)},
            "controller with its own scale",
        ),
        ("gnlse", {"interpolate_planes": True}, "interpolated planes"),
    ],
    ids=["coupled flow", "own scale", "interpolated planes"],
)
def test_an_uncovered_run_takes_the_host_loop_and_says_why(route, settings, reason):
    backend = launched(route, **settings)
    try:
        with pytest.warns(HostMarchWarning, match=reason):
            assert backend._rk_march() is None
        assert reason in backend.metadata()["rk_march"]
    finally:
        backend.close()


def test_the_private_switch_is_silent(monkeypatch):
    monkeypatch.setattr(backend_module, "_RK_MARCH", "host")
    backend = launched("gnlse")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", HostMarchWarning)
            assert backend._rk_march() is None
        assert backend.metadata()["rk_march"].startswith("host")
    finally:
        backend.close()
