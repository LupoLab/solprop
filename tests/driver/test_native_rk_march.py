# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Runge-Kutta march on real fields (SEP 45 sections 4.9, 7.3; SEP 47).

Until SEP 47's phase 5 these runs were held bit for bit to solprop's Python loop on the
same build, selected by a private switch that phase retired with the loop's forcing
hooks (SEP 47 section 6.3). That comparison's diagnostic purpose is kept by two
independent checks: the march's decisions reproduce the recorded identical-input traces
exactly (``tests/backends/test_native_rk_traces.py``, ``tests/backends/rk_traces.py``),
and here each route's native session is the same run twice -- its steps, norms, fields
and boundary-monitor record bit for bit on one thread, where the reductions' order is
fixed -- and is held to ``numpy_ref``'s run of the same problem within a hundred times
the controller's relative tolerance of the field's peak. The two adaptive runs take
their own steps (SEP 25 class T), and each keeps its local error within the tolerance;
a hundred covers the error's growth over these short runs (measured at most 4.2e-8 of
the peak, against tolerances of 1e-6 and looser).
"""

from __future__ import annotations

import dataclasses
import json
import os
import warnings
from typing import Any

import numpy as np
import pytest
from tests.driver.test_native_z_table import walk_off

from capability_routes import ROUTES
from solprop.backends import available
from solprop.exceptions import BoundaryWarning, HostMarchWarning
from solprop.monitor import BoundaryCheck
from solprop.stepper.controller import PIController
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.stage import anchor, scalars_at
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


def _run(build, backend: str = "native") -> dict[str, Any]:
    simulation = build()
    simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
    rtol = simulation._run_request().stepper.controller.rtol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        warnings.simplefilter("error", HostMarchWarning)
        run = simulation.run(progress="none")
    outcome = run.outcome
    assert outcome is not None and outcome.execution is not None
    return {
        "march": outcome.execution.get("rk_march"),
        "rtol": float(rtol),
        "record": json.dumps(outcome.execution.get("boundary_monitor"), sort_keys=True),
        "z": np.asarray(run["step_history/z"]),
        "h": np.asarray(run["step_history/h"]),
        "norm": np.asarray(run["step_history/norm"]),
        "accepted": np.asarray(run["step_history/accepted"]),
        "field": run.plane_field(run.n_saved - 1, "fft"),
        "fields": np.stack([run.plane_field(i, "fft") for i in range(run.n_saved)]),
    }


def _one_thread() -> bool:
    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


def _held(build) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run natively twice and on ``numpy_ref``; hold them as the module says.

    Returns the native run and the reference's.
    """
    native, again = _run(build), _run(build)
    assert native["march"] == again["march"] == "native"
    if _one_thread():
        for name in ("z", "h", "norm", "accepted", "field", "fields"):
            np.testing.assert_array_equal(native[name], again[name], name)
        assert native["record"] == again["record"]
    reference = _run(build, "numpy_ref")
    expected = reference["fields"]
    assert native["fields"].shape == expected.shape
    peak = float(np.max(np.abs(expected)))
    bound = 100.0 * native["rtol"] * peak
    assert float(np.max(np.abs(native["fields"] - expected))) <= bound
    return native, reference


@pytest.mark.parametrize(
    "route",
    [
        # The reference's moving guide takes most of a minute.
        pytest.param(route, marks=pytest.mark.medium)
        if route == "general guide (moving)"
        else route
        for route in ROUTED
    ],
)
def test_the_native_march_runs_each_route(route):
    _held(lambda: _route(route))


@pytest.mark.parametrize(
    "route", ["gnlse", "capillary (mode-averaged)", "capillary (multimode)"]
)
def test_interpolated_planes(route):
    """The session interpolates each plane inside the step that crossed it."""
    _held(lambda: _route(route, interpolate_planes=True))


def test_a_z_table_s_staged_trial_is_the_host_packed_trial():
    """The z table's positions load into other slots, holding the same values.

    The native state forms a staged trial's stage rows itself (``rk_trial``), loading
    the z table's positions stage by stage; the host packs the same trial's rows from
    solprop's stage scalars (``translate.stage.scalars_at`` into the backend's
    ``_trial_arrays``) for the state's unstaged trial (``trial_step``), loading them
    together. On one state the two trials are the same to the bit: their scored
    partials and every stage. Both trials are called on the backend's native state
    directly, loaded with the launch: SEP 47 phase 5 retired the private switch that
    made the backend's own trial pack its rows on the host.
    """
    simulation = walk_off("native", RungeKutta())
    problem = simulation.translate()
    run_request = simulation._run_request()
    assert problem.linear.phi_representation == "table"
    backend = backend_module.NativeBackend(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    try:
        backend._staged_rows()
        assert backend._rk_staged
        state = backend._native_state
        field = np.ascontiguousarray(problem.initial.field0, dtype=np.complex128)
        state.set_initial(field.reshape(-1), np.empty(0, dtype=np.complex128))
        nodes = [float(node) for node in run_request.stepper.tableau.nodes]
        end = float(run_request.stepper.z_end)
        for z in (0.0, 0.37 * end, 0.8 * end):
            dz = 0.01 * end
            staged = np.asarray(state.rk_trial(z, dz, nodes, prepare_only=False))
            stages = [np.array(state.stage_to_host(i)) for i in range(len(nodes))]
            held = anchor(problem, z, helpers=backend.helpers)
            *bundles, whole = scalars_at(
                problem,
                z,
                [*(z + node * dz for node in nodes), z + dz],
                at=held,
                helpers=backend.helpers,
            )
            arrays, material = backend._trial_arrays(bundles, whole)
            packed = np.asarray(state.trial_step(dz, *arrays, material))
            assert staged.tobytes() == packed.tobytes(), z
            for index, stage in enumerate(stages):
                assert np.array(state.stage_to_host(index)).tobytes() == (
                    stage.tobytes()
                ), (z, index)
            assert np.max(np.abs(stages[-1])) > 0.0
    finally:
        backend.close()


@pytest.mark.parametrize("dense", [False, True], ids=["landed", "interpolated"])
def test_a_z_table_s_march(dense):
    """The moving-dispersion fibre, whose operator is a z table, landed or dense."""
    _held(lambda: walk_off("native", RungeKutta(), n_save=7, interpolate_planes=dense))


def _replayed(route: str, steps: int) -> Any:
    simulation = _route(route)
    stepper = simulation._run_request().stepper
    end = float(stepper.z_end)
    schedule = FixedSchedule.uniform(
        0.0, end, end / steps, save_planes=np.asarray(stepper.save_planes).tolist()
    )
    return _route(route, schedule=schedule)


@pytest.mark.parametrize(
    "route",
    ["gnlse", pytest.param("capillary (coherent wall)", marks=pytest.mark.medium)],
)
def test_a_replay(route):
    """A fixed schedule's steps, taken as given: the reference's steps exactly."""
    native, reference = _held(lambda: _replayed(route, 400))
    np.testing.assert_array_equal(native["h"], reference["h"])
    np.testing.assert_array_equal(native["accepted"], reference["accepted"])


def test_a_weighted_norm_s_march():
    """Tightening part of the band (``weighted_norm_scale``), scored natively."""
    n_omega = _route("gnlse").translate().grids.n_omega
    weights = np.ones(n_omega)
    weights[: n_omega // 3] = 0.1
    controller = PIController(rtol=1e-6, atol=1e-12, weights=weights)
    _held(lambda: _route("gnlse", controller=controller))


@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_busy_monitor_s_checks_stay_in_the_native_rk_march(scheme):
    _held(
        lambda: walker(
            0.65 * TRANGE,
            check=BoundaryCheck(stop_factor=1000.0),
            backend="native",
            integrator=RungeKutta(),
            absorber_scheme=scheme,
            n_save=6,
        )
    )


@pytest.mark.parametrize("route", ["radial", "planar"])
def test_a_check_due_by_distance_beside_the_edge(route):
    # Steps shorter than the edge's read spacing but longer than the monitor's
    # interval make a check due by distance between reads.
    _held(lambda: _route(route, dz_max=3e-6))


@pytest.mark.gpu
@pytest.mark.parametrize("route", ["gnlse", "capillary (multimode)"])
def test_a_device_build_s_march(route):
    """On a device the march drives the resident engine from the host (SEP 45)."""
    if not backend_module._on_device():
        pytest.skip("the installed solprop-native build has no accelerator backend")
    _held(lambda: _route(route))
