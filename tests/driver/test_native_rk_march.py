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
from tests.driver.test_native_z_table import walk_off

from capability_routes import ROUTES
from solprop.backends import available, sessions
from solprop.exceptions import BoundaryWarning, HostMarchWarning
from solprop.monitor import BoundaryCheck
from solprop.stepper.controller import PIController
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
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
    "radial",
    "cartesian",
    "line",
    "planar",
)
"""Routes the native march takes: staged rows, beside a coupled flow or not, and the
spatial routes with their transverse edge (phase 4)."""


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
    "route", ["gnlse", "capillary (mode-averaged)", "capillary (multimode)"]
)
def test_interpolated_planes_are_the_host_loop_s(route, monkeypatch):
    """The step across a plane returns to solprop, which holds its extension."""
    build = lambda: _route(route, interpolate_planes=True)  # noqa: E731
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))


def test_a_z_table_s_staged_trial_is_the_host_packed_trial(monkeypatch):
    """The z table's positions load into other slots, holding the same values."""
    simulation = walk_off("native", RungeKutta())
    problem = simulation.translate()
    assert problem.linear.phi_representation == "table"
    backend = backend_module.NativeBackend(
        problem, grid=simulation.grid, modes=simulation.modes
    )
    try:
        backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
        assert backend._staged_rows()
        end = float(problem.stepper.z_end)
        for z in (0.0, 0.37 * end, 0.8 * end):
            dz = 0.01 * end
            staged = np.asarray(backend._prepare_trial(z, dz))
            with monkeypatch.context() as patch:
                patch.setattr(backend_module, "_RK_ROWS", "host")
                packed = np.asarray(backend._prepare_trial(z, dz))
            assert np.array_equal(staged, packed), z
    finally:
        backend.close()


@pytest.mark.parametrize("dense", [False, True], ids=["landed", "interpolated"])
def test_a_z_table_s_march_is_the_host_loop_s(dense, monkeypatch):
    """The moving-dispersion fibre, whose operator is a z table.

    Interpolated, it is held on the backend's own march (the bridge, which still
    serves what the native session refuses): the native session forms a z table's
    dense plane where the state lives, which agrees with the host loop's NumPy
    extension to rounding, not bitwise (``tests/sessions`` holds it to the driver).
    """
    if dense:
        monkeypatch.setattr(sessions, "_native_session", lambda *_, **__: None)
    build = lambda: walk_off(  # noqa: E731
        "native", RungeKutta(), n_save=7, interpolate_planes=dense
    )
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))


def _replayed(route: str, steps: int) -> Any:
    simulation = _route(route)
    stepper = simulation.translate().stepper
    end = float(stepper.z_end)
    schedule = FixedSchedule.uniform(
        0.0, end, end / steps, save_planes=np.asarray(stepper.save_planes).tolist()
    )
    return _route(route, schedule=schedule)


@pytest.mark.parametrize("route", ["gnlse", "capillary (coherent wall)"])
def test_a_replay_is_the_host_loop_s(route, monkeypatch):
    """A fixed schedule's steps, taken as given (``StepDriver._replay_to``)."""
    build = lambda: _replayed(route, 400)  # noqa: E731
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))


def test_a_weighted_norm_s_march_is_the_host_loop_s(monkeypatch):
    """Tightening part of the band (``weighted_norm_scale``), scored natively."""
    n_omega = _route("gnlse").translate().grids.n_omega
    weights = np.ones(n_omega)
    weights[: n_omega // 3] = 0.1
    controller = PIController(rtol=1e-6, atol=1e-12, weights=weights)
    build = lambda: _route("gnlse", controller=controller)  # noqa: E731
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


@pytest.mark.parametrize("route", ["radial", "planar"])
def test_a_check_due_by_distance_beside_the_edge_is_the_host_loop_s(route, monkeypatch):
    # Beside a transverse edge solprop takes the monitor's checks itself. Steps shorter
    # than the edge's read spacing but longer than the monitor's interval make a check
    # due by distance between reads, so the march returns to the host to take it.
    build = lambda: _route(route, dz_max=3e-6)  # noqa: E731
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))


@pytest.mark.gpu
@pytest.mark.parametrize("route", ["gnlse", "capillary (multimode)"])
def test_a_device_build_s_march_is_the_host_loop_s(route, monkeypatch):
    """On a device the march drives the resident engine from the host (SEP 45)."""
    if not backend_module._on_device():
        pytest.skip("the installed solprop-native build has no accelerator backend")
    build = lambda: _route(route)  # noqa: E731
    _same(_run(build, "native", monkeypatch), _run(build, "host", monkeypatch))
