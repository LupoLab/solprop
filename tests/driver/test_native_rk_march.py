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
the peak, against tolerances of 1e-6 and looser). The replays on a fixed schedule are
held to the reference by the catalogue's rows ``rk replay ...``
(``backends/driver_rows.py``); here they take the schedule's steps exactly.
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

from rk_route_cases import replayed, runge_kutta_route
from sessions.runs import load_native, native_trial
from solprop.backends import available
from solprop.exceptions import BoundaryWarning, HostMarchWarning
from solprop.monitor import BoundaryCheck
from solprop.stepper.controller import PIController
from solprop.stepper.runge_kutta import RungeKutta
from solprop.translate.stage import anchor, scalars_at
from test_monitor import TRANGE, walker

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
backend_module = pytest.importorskip("solprop_native.backend")
_testing = pytest.importorskip("solprop_native._core")._testing

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


def _repeated(build) -> dict[str, Any]:
    """Run natively twice, on the native march, the same run bit for bit on one thread.

    Returns the native run.
    """
    native, again = _run(build), _run(build)
    assert native["march"] == again["march"] == "native"
    if _one_thread():
        for name in ("z", "h", "norm", "accepted", "field", "fields"):
            np.testing.assert_array_equal(native[name], again[name], name)
        assert native["record"] == again["record"]
    return native


def _held(build) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run natively twice and on ``numpy_ref``; hold them as the module says.

    Returns the native run and the reference's.
    """
    native = _repeated(build)
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
        if route in {"general guide (constant)", "general guide (moving)"}
        else route
        for route in ROUTED
    ],
)
def test_the_native_march_runs_each_route(route):
    _held(lambda: runge_kutta_route(route))


@pytest.mark.parametrize(
    "route", ["gnlse", "capillary (mode-averaged)", "capillary (multimode)"]
)
def test_interpolated_planes(route):
    """The session interpolates each plane inside the step that crossed it."""
    _held(lambda: runge_kutta_route(route, interpolate_planes=True))


def _host_packed(backend, workspace, bundles, whole):
    """Pack a z table's trial rows on the host, as the backend packed them once.

    The ``Phi`` table's positions are loaded into the workspace's slots in one call,
    each stage's increment and value then the +-1 weights of its slots; the prefactor
    a table's one-hot slot or the bundles' own weights; the other rows the bundles'.
    """
    tables = backend._z_tables
    stages = [*bundles, whole]
    positions = [float(one.z0) for one in stages] + [float(one.z1) for one in stages]
    wanted = list(dict.fromkeys(positions))
    slot_of = dict(zip(wanted, _testing.table_load(workspace, wanted), strict=True))

    def one_hot(z, count):
        row = np.zeros(count)
        row[slot_of[float(z)]] = 1.0
        return row

    phi_delta = np.stack(
        [
            one_hot(one.z1, tables.n_phi) - one_hot(one.z0, tables.n_phi)
            for one in stages
        ]
    )
    phi_at = np.stack([one_hot(one.z1, tables.n_phi) for one in bundles])
    prefactor = (
        np.stack([one_hot(one.z1, tables.slots) for one in bundles])
        if tables.prefactor
        else np.stack([one.prefactor for one in bundles])
    )
    arrays = tuple(
        np.ascontiguousarray(values, dtype=np.float64)
        for values in (
            phi_delta,
            phi_at,
            prefactor,
            [one.density for one in bundles],
            [one.taper_scale for one in bundles],
            [one.preionisation for one in bundles],
            [one.raman_rates for one in bundles],
        )
    )
    material = {
        "kerr_density": np.asarray([one.kerr_density for one in bundles]),
        "plasma_densities": np.asarray([one.plasma_densities for one in bundles]),
        "plasma_preionisation": np.asarray(
            [one.plasma_preionisation for one in bundles]
        ),
        "raman_densities": np.asarray([one.raman_densities for one in bundles]),
    }
    return arrays, material


def test_a_z_table_s_staged_trial_is_the_host_packed_trial():
    """The z table's positions load into other slots, holding the same values.

    The native state forms a staged trial's stage rows itself (``rk_trial``), loading
    the z table's positions stage by stage; the host packs the same trial's rows from
    solprop's stage scalars (``translate.stage.scalars_at``), with the z table's
    positions loaded together into the workspace's slots (``table_load``) and selected
    by their +-1 weights, for the state's unstaged trial (``trial_step``). On one state
    the two trials are the same to the bit: their scored partials and every stage.
    Both trials are called on the backend's workspace directly, loaded with the launch.
    """
    simulation = walk_off("native", RungeKutta())
    problem = simulation.translate()
    run_request = simulation._run_request()
    assert problem.linear.phi_representation == "table"
    backend = backend_module.NativeBackend(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    try:
        state = load_native(backend, problem.initial.field0)
        nodes = [float(node) for node in run_request.stepper.tableau.nodes]
        end = float(run_request.stepper.z_end)
        for z in (0.0, 0.37 * end, 0.8 * end):
            dz = 0.01 * end
            staged = native_trial(backend, z, dz)
            stages = [
                np.array(_testing.stage_to_host(state, i)) for i in range(len(nodes))
            ]
            held = anchor(problem, z, helpers=backend.helpers)
            *bundles, whole = scalars_at(
                problem,
                z,
                [*(z + node * dz for node in nodes), z + dz],
                at=held,
                helpers=backend.helpers,
            )
            arrays, material = _host_packed(backend, state, bundles, whole)
            packed = np.asarray(_testing.trial_step(state, dz, *arrays, material))
            assert staged.tobytes() == packed.tobytes(), z
            for index, stage in enumerate(stages):
                assert np.array(_testing.stage_to_host(state, index)).tobytes() == (
                    stage.tobytes()
                ), (z, index)
            assert np.max(np.abs(stages[-1])) > 0.0
    finally:
        backend.close()


@pytest.mark.parametrize("dense", [False, True], ids=["landed", "interpolated"])
def test_a_z_table_s_march(dense):
    """The moving-dispersion fibre, whose operator is a z table, landed or dense."""
    _held(lambda: walk_off("native", RungeKutta(), n_save=7, interpolate_planes=dense))


@pytest.mark.parametrize(
    "route",
    ["gnlse", pytest.param("capillary (coherent wall)", marks=pytest.mark.medium)],
)
def test_a_replay_takes_the_schedule_s_steps(route):
    """A fixed schedule's steps, taken as given: the schedule's steps exactly.

    The reference takes the same steps; the planes against its run are the
    catalogue's rows ``rk replay ...`` (``backends/driver_rows.py``).
    """
    native = _repeated(lambda: replayed(route, 400))
    schedule = replayed(route, 400)._run_request().stepper.schedule
    assert schedule is not None
    np.testing.assert_array_equal(native["h"], schedule.steps)
    np.testing.assert_array_equal(native["accepted"], np.ones_like(native["accepted"]))


def test_a_weighted_norm_s_march():
    """Tightening part of the band (``weighted_norm_scale``), scored natively."""
    n_omega = runge_kutta_route("gnlse").translate().grids.n_omega
    weights = np.ones(n_omega)
    weights[: n_omega // 3] = 0.1
    controller = PIController(rtol=1e-6, atol=1e-12, weights=weights)
    _held(lambda: runge_kutta_route("gnlse", controller=controller))


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
    _held(lambda: runge_kutta_route(route, dz_max=3e-6))


@pytest.mark.gpu
@pytest.mark.parametrize("route", ["gnlse", "capillary (multimode)"])
def test_a_device_build_s_march(route):
    """On a device the march drives the resident engine from the host (SEP 45)."""
    if not backend_module._on_device():
        pytest.skip("the installed solprop-native build has no accelerator backend")
    _held(lambda: runge_kutta_route(route))
