# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Runge-Kutta march takes the host loop's steps (SEP 45 sections 4.9, 7.3).

On one build the native march and solprop's loop read the same staged rows and the same
native norms and apply the same decisions in binary64 without contraction, so they take
the same steps and produce the same fields, and the boundary monitor writes the same
record -- measured bit for bit here, a diagnostic SEP 25 does not promise across builds.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.exceptions import BoundaryWarning, HostMarchWarning
from solprop.monitor import BoundaryCheck
from solprop.stepper.runge_kutta import RungeKutta
from test_monitor import TRANGE, walker

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
backend_module = pytest.importorskip("solprop_native.backend")

ROUTED = (
    "gnlse",
    "capillary (mode-averaged)",
    "capillary (envelope)",
    "capillary (multimode)",
    "capillary (pressure gradient)",
    "general guide (constant)",
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""Routes the native march takes: staged rows, with or without a coupled flow (phase 4),
and no transverse edge."""


UNCHECKED = {"general guide (moving)"}
"""Routes whose window the launch check refuses when run whole: built to translate."""


def _route(route: str, **options: Any) -> Any:
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


def _run(build, march: str, monkeypatch) -> dict[str, Any]:
    with monkeypatch.context() as patch:
        patch.setattr(backend_module, "_RK_MARCH", march)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            warnings.simplefilter("error", HostMarchWarning)
            run = build().run(progress="none")
    outcome = run.outcome
    assert outcome is not None and outcome.execution is not None
    return {
        "march": outcome.execution["rk_march"],
        "record": json.dumps(outcome.execution.get("boundary_monitor"), sort_keys=True),
        "z": np.asarray(run["step_history/z"]),
        "h": np.asarray(run["step_history/h"]),
        "norm": np.asarray(run["step_history/norm"]),
        "accepted": np.asarray(run["step_history/accepted"]),
        "field": run.plane_field(run.n_saved - 1, "fft"),
        "fields": np.stack([run.plane_field(i, "fft") for i in range(run.n_saved)]),
    }


def _same(native: dict[str, Any], host: dict[str, Any]) -> None:
    assert native["march"] == "native"
    assert host["march"].startswith("host")
    for name in ("z", "h", "norm", "accepted", "field", "fields"):
        np.testing.assert_array_equal(native[name], host[name], name)
    assert native["record"] == host["record"]


@pytest.mark.parametrize("route", ROUTED)
def test_the_native_march_takes_the_host_loop_s_steps(route, monkeypatch):
    build = lambda: _route(route)  # noqa: E731
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))


@pytest.mark.parametrize(
    "route", ["gnlse", "capillary (mode-averaged)", "capillary (coherent wall)"]
)
def test_interpolated_planes_are_the_host_loop_s(route, monkeypatch):
    """The step across a plane returns to solprop, which holds its extension."""
    build = lambda: _route(route, interpolate_planes=True)  # noqa: E731
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))


@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_busy_monitor_s_checks_stay_in_the_native_rk_march(scheme, monkeypatch):
    build = lambda: walker(  # noqa: E731
        0.65 * TRANGE,
        check=BoundaryCheck(stop_factor=1000.0),
        backend="native",
        integrator=RungeKutta(),
        absorber_scheme=scheme,
        n_save=6,
    )
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))
