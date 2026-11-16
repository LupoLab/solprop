# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run records, formed once per run from the problem, the request and the start.

SEP 52 section 5.1: both backends read the same records, built by shared code
(``solprop.backends.records``) and importable without the native package. These tests
hold each record to the facts it is built from, for each march -- Adams, the adaptive
and the replayed Runge-Kutta march, the low-storage replay -- for a resumed start and a
monitored and an edged run; hold the reference's own derivations to them; and, where
the native backend is installed, hold its conversion to what its extension reads.
"""

import dataclasses
import math
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

from backends.test_transverse_backends import radial_collar
from solprop.backends.memory import low_storage_flows
from solprop.backends.records import (
    ATTEMPT_CAPACITY,
    CALL_ATTEMPTS,
    CALL_SECONDS,
    PLANE_COLUMNS,
    ROW_CAPACITY,
    EdgeResumption,
    LowStorageSettings,
    PlanePlan,
    Resumption,
    RungeKuttaPolicy,
    RunRecords,
    adams_policy,
    collar_rests,
    declared,
    edge_resumption,
    monitor_continuation,
    resumption,
    run_records,
    runge_kutta_policy,
)
from solprop.backends.registry import available, select
from solprop.backends.session import Start
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError
from solprop.grid import EnvelopeGrid
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import GaussianPulse, SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.stepper.controller import START_GROWTH
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import MarchPolicy, march_policy
from solprop.stepper.schedule import LANDING_ULPS
from solprop.translate.assembly import assemble
from solprop.translate.problem import Translation

LENGTH = 0.02
"""The fibre's length, m: a few steps."""

PLANES = (0.01, 0.02)
"""Its save planes, m."""

native_available = "native" in available()
needs_native = pytest.mark.skipif(
    not native_available, reason="the native backend is not installed"
)


def _gnlse(**options: Any) -> Translation:
    """Return a short GNLSE soliton's problem and request, with `options` assembled."""
    grid = EnvelopeGrid.from_limits(LENGTH, 835e-9, (500e-9, 2e-6), 1e-12)
    gnlse = problem_from(
        [0.0, 0.0, -1.183e-26], omega_ref=grid.omega0, gamma=0.11, shock=False
    )
    launch = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8)
    settings: dict[str, Any] = {
        "controller": PIController(rtol=1e-8, atol=1e-16),
        "save_planes": list(PLANES),
        "dz_max": LENGTH / 4,
        "dz_init": 1e-4,
        "density": 1.0,
        "shock": False,
        "prefactor_index": 1.0,
    }
    settings.update(options)
    return assemble(grid, gnlse.mode, [gnlse.kerr], launch.build(grid), **settings)


def _uniform(step: float = LENGTH / 8) -> FixedSchedule:
    """Return a uniform schedule landing on the planes."""
    return FixedSchedule.uniform(0.0, LENGTH, step, save_planes=list(PLANES))


def _monitored(**options: Any) -> Simulation:
    """Return a linear pulse walking towards its window's edge, monitored."""
    grid = EnvelopeGrid.from_limits(1.0, 1e-6, (0.8e-6, 1.25e-6), 1e-12)
    omega0 = 2 * np.pi * C / 1e-6
    simulation = Simulation.from_gnlse(
        1e-30, 1.0, [0.0, 0.0, -2e-26], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(
            wavelength=2 * np.pi * C / (omega0 + 1e12), fwhm=200e-15, peak_power=1.0,
        )],
        noise=None, raman=False, shock=False, backend="numpy_ref", n_save=3,
        **options,
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings, boundary_check=BoundaryCheck()
    )
    return simulation


def _entrance(problem: Any) -> Start:
    return Start.entrance(problem)


def test_the_records_import_and_build_without_the_native_package():
    # The reference reads the records with no native package installed (SEP 52 D4):
    # importing them, and forming a run's, loads nothing of it.
    code = textwrap.dedent(
        """
        import sys

        class Refused:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] == "solprop_native":
                    raise ImportError(name)

        sys.meta_path.insert(0, Refused())
        from solprop.backends import engines, records
        from solprop.backends.session import Start
        from solprop.grid import EnvelopeGrid
        from solprop.nonlinear.gnlse import problem_from
        from solprop.pulses import SechPulse
        from solprop.stepper import PIController
        from solprop.translate.assembly import assemble

        grid = EnvelopeGrid.from_limits(0.02, 835e-9, (500e-9, 2e-6), 1e-12)
        gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
                             gamma=0.11, shock=False)
        launch = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8)
        problem, request = assemble(grid, gnlse.mode, [gnlse.kerr],
                                    launch.build(grid), save_planes=[0.02],
                                    controller=PIController(rtol=1e-8, atol=1e-16),
                                    dz_max=0.005, dz_init=1e-4, density=1.0,
                                    shock=False, prefactor_index=1.0)
        built = records.run_records(problem, request, Start.entrance(problem))
        assert isinstance(built.policy, records.RungeKuttaPolicy)
        assert engines.AcceptRecord().removed == 0.0
        assert not [name for name in sys.modules if name.startswith("solprop_native")]
        print("built")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "built"


# -- the method's policy ------------------------------------------------------------


def test_an_adams_run_s_policy_is_the_march_policy_of_its_settings():
    problem, request = _gnlse(integrator=Multistep())
    records = run_records(problem, request, _entrance(problem))
    stepper = request.stepper
    method = stepper.integrator
    assert isinstance(method, Multistep)
    # An envelope grid without the third harmonic takes no carrier shift, and the
    # problem has no collar: the policy is the method's with the settings' landings.
    expected = march_policy(
        method,
        dz_min=stepper.dz_min,
        dz_max=stepper.dz_max,
        stretch=stepper.stretch,
        z_end=stepper.z_end,
        save_planes=list(PLANES),
        hard_kinks=list(problem.linear.hard_kinks),
        breakpoints=list(problem.linear.phi_breakpoints),
        interpolate=False,
    )
    assert records.policy == expected
    assert records.policy.quiet_interval is None
    assert (records.low_storage, records.schedule, records.monitor) == (
        None,
        None,
        None,
    )


def test_the_reference_s_adams_march_reads_the_shared_policy():
    problem, request = _gnlse(integrator=Multistep())
    records = run_records(problem, request, _entrance(problem))
    backend: Any = select("numpy_ref", problem, request)
    try:
        assert backend._adams._policy == records.policy
    finally:
        backend.close()


def test_an_adams_policy_takes_its_carrier_and_collar_from_its_arguments():
    problem, request = _gnlse(integrator=Multistep())
    stepper = request.stepper
    method = stepper.integrator
    assert isinstance(method, Multistep)
    policy = adams_policy(problem, stepper, method, carrier=True)
    assert policy.carrier
    assert policy.collar_rate == 0.0  # no in-stage temporal collar


def test_an_adaptive_runge_kutta_run_s_policy_holds_its_controller_and_landings():
    problem, request = _gnlse()
    stepper = request.stepper
    records = run_records(problem, request, _entrance(problem))
    breakpoints = np.asarray(problem.linear.phi_breakpoints)
    assert records.policy == RungeKuttaPolicy(
        controller=stepper.controller,
        dz_min=stepper.dz_min,
        dz_max=LENGTH / 4,
        auto_init=stepper.auto_init,
        stretch=stepper.stretch,
        z_end=LENGTH,
        save_planes=PLANES,
        interpolate=False,
        hard_kinks=tuple(float(z) for z in problem.linear.hard_kinks),
        # Without the interval cap only the operator table's ends bound a step.
        breakpoints=(float(breakpoints[0]), float(breakpoints[-1])),
        landing_ulps=LANDING_ULPS,
        start_growth=START_GROWTH,
        quiet_interval=None,
    )
    assert isinstance(records.policy, RungeKuttaPolicy)
    assert records.policy.controller is stepper.controller
    assert (records.low_storage, records.schedule) == (None, None)


def test_the_reference_s_runge_kutta_march_reads_the_shared_policy():
    problem, request = _gnlse()
    backend: Any = select("numpy_ref", problem, request)
    try:
        assert (
            backend._march._policy
            == run_records(problem, request, _entrance(problem)).policy
        )
        assert backend._collar_can_rest() is collar_rests(problem, request.stepper)
    finally:
        backend.close()


def test_interpolated_planes_are_stepped_across():
    problem, request = _gnlse(interpolate_planes=True)
    policy = runge_kutta_policy(problem, request.stepper)
    assert (policy.save_planes, policy.interpolate) == ((), True)
    records = run_records(problem, request, _entrance(problem))
    assert records.planes.dense
    assert records.planes.fixed_positions is None


def test_a_replayed_run_carries_its_schedule_and_its_length():
    schedule = _uniform()
    problem, request = _gnlse(schedule=schedule)
    records = run_records(problem, request, _entrance(problem))
    assert isinstance(records.policy, RungeKuttaPolicy)
    assert records.schedule is request.stepper.schedule
    plan = records.planes
    assert (plan.replay_steps, plan.replay_length) == (
        schedule.n_steps,
        float(schedule.total_length),
    )


def test_a_low_storage_replay_carries_its_flows_and_its_operator_s_origin():
    problem, request = _gnlse(integrator=LowStorageRungeKutta(), schedule=_uniform())
    stepper = request.stepper
    records = run_records(problem, request, _entrance(problem))
    assert isinstance(records.policy, RungeKuttaPolicy)
    assert records.low_storage == LowStorageSettings(
        flows=low_storage_flows(problem, stepper, "native"),
        separable=False,
        origin=float(problem.linear.z_span[0]),
    )
    # The GNLSE's operator is constant: the replay tabulates one flow under "auto".
    assert records.low_storage is not None and records.low_storage.flows == 1
    # The low-storage march keeps its collar in every pass (SEP 21 D15).
    assert not collar_rests(problem, stepper)


# -- the plane plan -----------------------------------------------------------------


def test_a_fresh_start_owes_its_launch_plane_and_fixes_every_position():
    problem, request = _gnlse()
    plan = run_records(problem, request, _entrance(problem)).planes
    assert isinstance(plan, PlanePlan)
    assert plan.planes.tolist() == list(PLANES)
    assert (plan.z, plan.next_plane, plan.launch_pending) == (0.0, 0, True)
    assert (plan.step_planes, plan.dense, plan.period) == (0, False, 0)
    assert (plan.replay_steps, plan.replay_length) == (0, 0.0)
    assert plan.fixed_positions == (0.0, *PLANES)
    assert (plan.attempt_capacity, plan.row_capacity) == (
        ATTEMPT_CAPACITY,
        ROW_CAPACITY,
    )
    assert (plan.call_attempts, plan.call_seconds) == (CALL_ATTEMPTS, CALL_SECONDS)


def test_a_resumed_start_continues_past_its_saved_plane():
    problem, request = _gnlse()
    resumed = Start(
        field=problem.initial.field0,
        z=PLANES[0],
        plane_index=1,
        plane_pending=False,
        resume=True,
    )
    plan = run_records(problem, request, resumed).planes
    assert (plan.z, plan.next_plane, plan.launch_pending) == (PLANES[0], 1, False)
    assert plan.fixed_positions == (PLANES[0], *PLANES)
    # The start's field is not read: a session takes it afterwards.
    assert resumed.field is not None


def test_periodic_rows_and_step_planes_set_the_cadence():
    problem, request = _gnlse(
        step_planes=3,
        stats_spec=StatsSpec(names=("energy",), periodic=("energy",), period=4),
    )
    plan = run_records(problem, request, _entrance(problem)).planes
    assert (plan.step_planes, plan.period) == (3, 4)
    # A step plane's position is the march's.
    assert plan.fixed_positions is None


# -- the boundary monitor -----------------------------------------------------------


def _monitored_records(start: Start | None = None) -> tuple[Any, Any, RunRecords]:
    simulation = _monitored()
    problem = simulation.translate()
    request = simulation._run_request().for_run(monitor=simulation._boundary_monitor())
    return problem, request, run_records(problem, request, start or _entrance(problem))


def test_a_monitored_run_carries_its_monitor_s_configuration():
    problem, request, records = _monitored_records()
    monitor = request.boundary.monitor
    held = records.monitor
    assert held is not None
    config = held.config
    expected = dataclasses.replace(
        monitor, crossing=problem.split_crossing if monitor.split else None
    ).watch_policy()
    # The launch's floor is formed natively: the record carries zeros for it.
    expected["floor_regions"] = [0.0, 0.0, 0.0, 0.0]
    expected["launch_floor"] = 0.0
    assert config.policy == expected
    assert config.budget == monitor.check.budget
    assert config.can_rest is collar_rests(problem, request.stepper)
    np.testing.assert_array_equal(config.rate, monitor.weights[3])
    assert config.rate.flags.c_contiguous
    assert (held.edge, held.continuation) == (None, None)


def _earlier_record() -> dict[str, Any]:
    """Return an earlier execution's monitor record, as the output stores it."""
    return {
        "launch_energy": 2.0,
        "guard_low_max": 1e-6,
        "removed_fraction": 1e-4,
        "steps": 12,
        "skipped": 3.0,
        "quiet_checks": 2,
        "collar_quiet": True,
        "unwatched": [[0.1, 0.2]],
        "resumed_at": [0.1],
        "planes": {name: [0.5, None] for name in PLANE_COLUMNS},
        "transverse": {"launch_energy": 1.0, "edge_baseline": [0.1, 0.2]},
        "noise_floor": "not a number",
    }


def test_a_monitored_resume_carries_its_continuation():
    record = _earlier_record()
    simulation = _monitored()
    problem = simulation.translate()
    start = Start(
        field=problem.initial.field0,
        z=0.5,
        plane_index=1,
        plane_pending=False,
        resume=True,
        monitor={"record": record, "plane": True, "unwatched": [(0.2, 0.5)]},
    )
    _, _, records = _monitored_records(start)
    assert records.monitor is not None
    continuation = records.monitor.continuation
    assert continuation is not None
    assert continuation.plane
    assert continuation.unwatched == ((0.2, 0.5),)
    assert continuation.record == resumption(record)
    # Without a transverse edge, no face reads a per-face list.
    assert continuation.edge == edge_resumption(record["transverse"], 0)


def test_a_resumption_reads_only_the_record_s_types():
    record = _earlier_record()
    held = resumption(record)
    assert held.record and held.collar_quiet
    # The counts are whole past the launch test; text is not a number.
    assert held.numbers == {
        "launch_energy": 2.0,
        "guard_low_max": 1e-6,
        "removed_fraction": 1e-4,
        "quiet_checks": 2.0,
        "steps": 12.0,
        "skipped": 3.0,
        "collar_left_out_checks": 0.0,
    }
    assert held.unwatched == ((0.1, 0.2),)
    assert held.resumed_at == (0.1,)
    assert held.planes is not None
    assert held.planes[0][0] == 0.5
    assert all(math.isnan(column[1]) for column in held.planes)
    assert resumption(None) == Resumption(record=False)


def test_a_record_without_a_launch_reads_no_counts():
    record = {"steps": 2.5, "launch_energy": 0.0}
    assert resumption(record).numbers == {"launch_energy": 0.0}


def test_a_count_that_is_not_whole_is_refused_as_the_monitor_refuses_it():
    record = {"launch_energy": 1.0, "steps": -1.0}
    with pytest.raises(BoundaryError, match="steps"):
        resumption(record)


def test_an_edge_resumption_reads_further_only_past_its_launch_and_baseline():
    full = {
        "launch_energy": 1.0,
        "warned_at": 0.3,
        "edge_baseline": [0.1, 0.2],
        "edge_max": [0.3, 0.4],
        "edge_max_raw": [0.5],
        "reads": 7.0,
        "transverse_status": "warned",
    }
    assert edge_resumption(full, 2) == EdgeResumption(
        record=True,
        numbers={"launch_energy": 1.0, "warned_at": 0.3},
        edge_baseline=(0.1, 0.2),
        edge_max=(0.3, 0.4),
        # One value for two faces is not a value per face.
        edge_max_raw=None,
        reads=7,
        transverse_status="warned",
    )
    dark = dict(full, launch_energy=0.0)
    assert edge_resumption(dark, 2) == EdgeResumption(
        record=True,
        numbers={"launch_energy": 0.0, "warned_at": 0.3},
        transverse_status="warned",
    )
    assert edge_resumption(None, 2) == EdgeResumption(record=False)


def test_an_edged_run_carries_its_edge_s_configuration():
    simulation = radial_collar("numpy_ref", RungeKutta())
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None and monitor.transverse is not None
    request = simulation._run_request().for_run(stats=StatsSpec(), monitor=monitor)
    records = run_records(problem, request, _entrance(problem))
    assert records.monitor is not None
    edge = records.monitor.edge
    assert edge is not None
    transverse = monitor.transverse
    assert edge.faces == tuple(face.name for face in transverse._faces)
    assert edge.warn == transverse.budget**2
    assert edge.stop == (transverse.stop_factor * transverse.budget) ** 2
    assert edge.length == transverse.length
    # An adaptive run has no longest fixed step; the edge is read.
    assert (edge.longest_step, edge.reads_edge) == (None, True)
    assert edge.declared is declared(problem)
    assert edge.design is (transverse.design is not None)
    assert edge.kappa_order.dtype == np.uint64
    np.testing.assert_array_equal(edge.kappa_order, np.argsort(transverse._kappa))
    # The edge's faces decide whether a list holds a value per face.
    faces = len(edge.faces)
    for count, read in ((faces, True), (faces + 1, False)):
        transverse_record = {"launch_energy": 1.0, "edge_baseline": [0.1] * count}
        continuation = monitor_continuation(
            monitor, {"record": {"transverse": transverse_record}}
        )
        assert (continuation.edge.edge_baseline is not None) is read
        assert continuation.unwatched == ()
        assert not continuation.plane


# -- the native conversion ----------------------------------------------------------


@needs_native
def test_native_reads_the_runge_kutta_policy_with_the_controller_s_fields():
    native_records = pytest.importorskip("solprop_native._records")
    problem, request = _gnlse()
    stepper = request.stepper
    controller = stepper.controller
    policy = runge_kutta_policy(problem, stepper)
    breakpoints = np.asarray(problem.linear.phi_breakpoints)
    assert native_records.rk_policy(policy) == {
        "dz_min": stepper.dz_min,
        "dz_max": stepper.dz_max,
        "stretch": stepper.stretch,
        "z_end": stepper.z_end,
        "save_planes": list(PLANES),
        "interpolate": False,
        "hard_kinks": [float(z) for z in problem.linear.hard_kinks],
        "breakpoints": [float(breakpoints[0]), float(breakpoints[-1])],
        "landing_ulps": LANDING_ULPS,
        "quiet_interval": None,
        "safety": controller.safety,
        "beta1": controller.beta1,
        "beta2": controller.beta2,
        "min_factor": controller.min_factor,
        "max_factor": controller.max_factor,
        "lag": controller.lag,
        "reject_free": controller.reject_free,
        "power": float(controller.estimator_order + 1),
        "exponent": 1.0 / (controller.estimator_order + 1),
        "start_growth": START_GROWTH,
        "auto_init": stepper.auto_init,
    }


@needs_native
def test_native_s_adams_march_holds_the_shared_policy():
    native_backend = pytest.importorskip("solprop_native.backend")
    problem, request = _gnlse(integrator=Multistep())
    records = run_records(problem, request, _entrance(problem))
    backend = native_backend.NativeBackend(problem, request)
    try:
        policy: MarchPolicy = backend._adams._policy  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
        assert policy == records.policy
    finally:
        backend.close()


@needs_native
def test_native_reads_the_plane_plan_and_the_continuation():
    native_records = pytest.importorskip("solprop_native._records")
    record = _earlier_record()
    problem, request = _gnlse(schedule=_uniform())
    start = Start(field=None, z=PLANES[0], plane_index=1, plane_pending=False)
    plan = native_records.session_plan(
        run_records(problem, request, start).planes,
        plane_statistics=None,
        row_statistics=None,
        monitor=None,
        dense=None,
    )
    assert plan["replay"] is True
    assert (plan["next_plane"], plan["launch_pending"]) == (1, False)
    assert plan["planes"].tolist() == list(PLANES)
    simulation = _monitored()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    held = native_records.continuation(
        monitor_continuation(
            monitor, {"record": record, "plane": True, "unwatched": [(0.2, 0.5)]}
        )
    )
    assert held["plane"] is True
    assert held["unwatched"] == [[0.2, 0.5]]
    assert held["edge_record"] == {"record": True, "launch_energy": 1.0}
    flat = held["record"]
    assert flat["record"] and flat["collar_quiet"] and flat["steps"] == 12.0
    assert flat["unwatched"] == [[0.1, 0.2]] and flat["resumed_at"] == [0.1]
    assert len(flat["planes"]) == len(PLANE_COLUMNS)


@needs_native
def test_native_reads_the_edge_s_configuration_by_field():
    native_records = pytest.importorskip("solprop_native._records")
    simulation = radial_collar("numpy_ref", RungeKutta())
    problem = simulation.translate()
    request = simulation._run_request().for_run(
        stats=StatsSpec(), monitor=simulation._boundary_monitor()
    )
    held = run_records(problem, request, _entrance(problem)).monitor
    assert held is not None and held.edge is not None
    values = native_records.edge_config(held.edge)
    assert set(values) == {field.name for field in dataclasses.fields(held.edge)}
    assert values["faces"] == list(held.edge.faces)
    # The arrays cross as they are, not copied.
    assert values["kappa"] is held.edge.kappa
    config = native_records.monitor_config(held.config)
    assert config["rate"] is held.config.rate
    assert config["policy"] == held.config.policy
