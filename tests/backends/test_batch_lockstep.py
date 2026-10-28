# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the member-workspace lockstep baseline against independent fixed runs.

A batch member and its solo run take one method on one fixed schedule, so they agree
within a rounding budget (SEP 25), not to the bit.
"""

import dataclasses
from collections.abc import Sequence
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest

from bounds import assert_within, gamma, ladder_bound
from solprop.backends import Backend, BatchBackend, select, select_batch
from solprop.backends.batch import LockstepRunner
from solprop.backends.driver import StepDriver
from solprop.backends.numba_cpu.batch import NumbaCpuBatchBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.driver import Simulation
from solprop.exceptions import BackendError, BoundaryError
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule


def simulation(index, backend, noise=None):
    # The Runge-Kutta march's fused batch passes; an Adams batch is in
    # tests/backends/test_exponential_march.py.
    return Simulation.from_gnlse(
        0.11 * (index + 1),
        0.001,
        [0.0, 0.0, -1.183e-26 * (index + 1)],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=float(index + 1),
        pulseshape="sech",
        save_planes=[0.0005, 0.001],
        schedule=FixedSchedule.uniform(
            0.0, 0.001, 0.00025, save_planes=[0.0005, 0.001]
        ),
        stats=StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2),
        noise=noise,
        seed=index + 13 if noise is not None else None,
        backend=backend,
        integrator=RungeKutta(),
        absorber=False,
    )


def loaded(simulation, cleanup):
    problem = simulation.translate()
    owner = select(
        simulation.settings.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    cleanup.callback(owner.close)
    owner.set_initial(problem.initial.field0, simulation.noise)
    assert isinstance(owner, StepDriver)
    return cast(StepDriver[object], owner)


STEPS = 4
"""The fixed Runge-Kutta steps to the last plane."""

FIELD_BOUND = STEPS * 16 * np.finfo(float).eps
"""The singleton ladder's 16 eps a step: a member and its solo run take one method on
one schedule and differ by the order of their reductions alone."""


def assert_observations_agree(actual, expected, rtol, bound=FIELD_BOUND):
    """A member's observation against its solo run's: exact where it is structural.

    The position, the schedule and the acceptances are prescribed, so exact; the field
    and the modified noise's phase are held to the run's relative `bound`; the error
    norm, a defect over ``rtol``, to seven stages of that over ``rtol``; the statistics
    rows, squares and their sums, to twice the bound and the sum's reduction.
    """
    assert actual.z == expected.z
    assert_within(actual.field, expected.field, bound, what="field")
    if expected.noise_phase is None:
        assert actual.noise_phase is None
    else:
        np.testing.assert_allclose(
            actual.noise_phase,
            expected.noise_phase,
            rtol=0,
            atol=bound * float(np.max(np.abs(expected.noise_phase), initial=1.0)),
        )
    for name in ("z", "dz", "accepted"):
        np.testing.assert_array_equal(
            getattr(actual.step_history, name), getattr(expected.step_history, name)
        )
    np.testing.assert_allclose(
        actual.step_history.error_norm,
        expected.step_history.error_norm,
        rtol=0,
        atol=7 * bound / rtol,
    )
    np.testing.assert_array_equal(actual.step_stats.z, expected.step_stats.z)
    np.testing.assert_array_equal(actual.step_stats.step, expected.step_stats.step)
    assert actual.step_stats.values.keys() == expected.step_stats.values.keys()
    for name in expected.step_stats.values:
        np.testing.assert_allclose(
            actual.step_stats.values[name],
            expected.step_stats.values[name],
            rtol=statistics_bound(actual.field, bound),
            err_msg=name,
        )


def statistics_bound(field, bound=FIELD_BOUND):
    """Twice the field's budget, for squares, and a reduction over the samples."""
    return 2 * bound + gamma(np.size(field))


@pytest.mark.parametrize(
    ("backend", "runner_factory"),
    [
        ("numpy_ref", LockstepRunner),
        ("numpy_ref", NumpyBatchBackend),
        ("numba_cpu", LockstepRunner),
        ("numba_cpu", NumbaCpuBatchBackend),
    ],
)
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("noise", [None, "input", "modified"])
@pytest.mark.parametrize("weighted", [False, True])
def test_lockstep_preserves_member_physics_noise_and_statistics(
    backend, runner_factory, count, noise, weighted
):
    with ExitStack() as cleanup:
        simulations = [simulation(index, backend, noise) for index in range(count)]
        if weighted:
            for one in simulations:
                one.settings = dataclasses.replace(
                    one.settings,
                    controller=dataclasses.replace(
                        cast("PIController", one.settings.controller),
                        weights=np.linspace(0.5, 1.5, one.grid.n_omega),
                    ),
                )
        members = tuple(loaded(one, cleanup) for one in simulations)
        independent = tuple(loaded(one, cleanup) for one in simulations)
        runner = runner_factory(members)
        cleanup.callback(runner.close)
        spec = simulations[0].translate().stats_spec
        rtol = cast("PIController", simulations[0].settings.controller).rtol
        for target in (0.0, 0.0005, 0.001):
            runner.propagate_to(target)
            for owner in independent:
                owner.propagate_to(target)
            batched_stats = runner.stats(spec)
            for index, (actual, owner) in enumerate(
                zip(runner.observe_members(), independent, strict=True)
            ):
                expected = owner.observe()
                assert_observations_agree(actual, expected, rtol)
                for name, value in owner.stats(spec).items():
                    np.testing.assert_allclose(
                        batched_stats[name][index],
                        value[0],
                        rtol=statistics_bound(expected.field),
                        err_msg=name,
                    )
        assert runner.metadata()["backend"] == backend
        assert runner.metadata()["fused_batch_kernels"] is False
        assert len(runner.member_metadata()) == count
        assert sum(runner.memory_ledger().values()) > 0
        runner.close()
        assert runner.memory_ledger() == {}


def test_lockstep_interleaves_members_at_each_fixed_step(monkeypatch):
    visited = []
    with ExitStack() as cleanup:
        members = tuple(loaded(simulation(i, "numpy_ref"), cleanup) for i in range(3))
        for index, owner in enumerate(members):
            original = owner._trial_step

            def traced(z, dz, *, index=index, original=original):
                visited.append((index, z, dz))
                return original(z, dz)

            monkeypatch.setattr(owner, "_trial_step", traced)
        runner = LockstepRunner(members)
        cleanup.callback(runner.close)
        runner.propagate_to(0.001)
    assert [index for index, _, _ in visited] == [0, 1, 2] * 4
    for start in range(0, len(visited), 3):
        assert len({(z, dz) for _, z, dz in visited[start : start + 3]}) == 1


@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
@pytest.mark.parametrize("noise", [None, "input", "modified"])
def test_batch_agrees_with_numpy_oracle(backend, noise):
    if backend == "native":
        pytest.importorskip("solprop_native")
    with ExitStack() as cleanup:
        candidates = [simulation(index, backend, noise) for index in range(3)]
        # The established singleton ladder allows 16 eps per step on the host,
        # with seven FFT round trips per step in the predeclared device bound.
        bound = ladder_bound(
            backend, 4 * 16 * np.finfo(float).eps, candidates[0].grid.n_to, 4 * 7
        )
        assert bound > 0
        actual = select_batch(
            cast(Sequence[Backend], tuple(loaded(one, cleanup) for one in candidates))
        )
        reference = select_batch(
            cast(
                Sequence[Backend],
                tuple(
                    loaded(simulation(i, "numpy_ref", noise), cleanup) for i in range(3)
                ),
            )
        )
        cleanup.callback(actual.close)
        cleanup.callback(reference.close)
        actual.propagate_to(0.001)
        reference.propagate_to(0.001)
        for ours, oracle in zip(
            actual.observe_members(), reference.observe_members(), strict=True
        ):
            relative = np.linalg.norm(
                ours.require_field() - oracle.require_field()
            ) / np.linalg.norm(oracle.require_field())
            assert relative < bound


def test_lockstep_refuses_mixed_backends():
    with ExitStack() as cleanup:
        members = tuple(
            loaded(simulation(0, name), cleanup) for name in ("numpy_ref", "numba_cpu")
        )
        with pytest.raises(BackendError, match="mixed backends"):
            LockstepRunner(members)


def test_lockstep_refuses_adaptive_schedule():
    one = simulation(0, "numpy_ref")
    one.settings = dataclasses.replace(one.settings, schedule=None)
    with ExitStack() as cleanup:
        owner = loaded(one, cleanup)
        with pytest.raises(BackendError, match="explicit fixed schedule"):
            LockstepRunner([owner])


def test_lockstep_refuses_shared_history_observation_and_closes_members():
    with ExitStack() as cleanup:
        owner = loaded(simulation(0, "numpy_ref"), cleanup)
        runner = LockstepRunner([owner])
        with pytest.raises(BackendError, match="individual error histories"):
            runner.observe()
        runner.close()
        runner.close()
        with pytest.raises(BackendError, match="closed"):
            owner.observe()


def test_lockstep_refuses_empty_members():
    with pytest.raises(BackendError, match="at least one member"):
        LockstepRunner([])


def test_batch_registry_adopts_members_and_exposes_independent_observations():
    with ExitStack() as cleanup:
        members = tuple(loaded(simulation(i, "numpy_ref"), cleanup) for i in range(2))
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        assert isinstance(runner, BatchBackend)
        assert runner.max_batch == 2
        assert len(runner.observe_members()) == 2
        assert len(runner.member_metadata()) == 2


def test_lockstep_gives_every_member_its_own_boundary_monitor():
    with ExitStack() as cleanup:
        simulations = [simulation(index, "numpy_ref") for index in range(2)]
        members = tuple(loaded(one, cleanup) for one in simulations)
        runner = LockstepRunner(members)
        cleanup.callback(runner.close)
        grid = simulations[0].grid
        # The cheap members' window holds about 1e-5 of their energy in its guards,
        # a field of 3e-3: a check at that budget watches them without a warning.
        relaxed = BoundaryCheck(budget=1e-2)
        runner.attach_monitor(BoundaryMonitor.build(grid, None, relaxed))
        runner.propagate_to(0.001)
        monitors = [member._monitor for member in members]
        assert monitors[0] is not monitors[1]
        for monitor in monitors:
            assert monitor is not None
            planes = monitor.summary()["planes"]
            assert isinstance(planes, dict)
            assert planes["z"][-1] == pytest.approx(0.001)


def test_a_member_s_stop_ends_its_watch_alone():
    # A check strict enough to refuse any launch stops the first member before its
    # first step; the second, unwatched, and the batch go on.
    with ExitStack() as cleanup:
        simulations = [simulation(index, "numpy_ref") for index in range(2)]
        members = tuple(loaded(one, cleanup) for one in simulations)
        runner = LockstepRunner(members)
        cleanup.callback(runner.close)
        strict = BoundaryCheck(budget=1e-300, stop_factor=1.0)
        runner.attach_monitors(
            [BoundaryMonitor.build(simulations[0].grid, None, strict), None]
        )
        runner.propagate_to(0.001)
        runner.finish_monitors()
        stopped, unwatched = runner.member_stops
        assert isinstance(stopped, BoundaryError)
        assert unwatched is None
        assert members[0]._monitor is None
        assert runner._z == pytest.approx(0.001)


def test_lockstep_refuses_a_monitor_per_member_miscount():
    with ExitStack() as cleanup:
        members = tuple(loaded(simulation(i, "numpy_ref"), cleanup) for i in range(2))
        runner = LockstepRunner(members)
        cleanup.callback(runner.close)
        with pytest.raises(BackendError, match="1 boundary monitors for 2 members"):
            runner.attach_monitors([None])


def test_batch_registry_refuses_empty_and_mixed_owners():
    with pytest.raises(BackendError, match="at least one"):
        select_batch([])
    with ExitStack() as cleanup:
        members = tuple(
            loaded(simulation(0, name), cleanup) for name in ("numpy_ref", "numba_cpu")
        )
        with pytest.raises(BackendError, match="mixed backends"):
            select_batch(cast(Sequence[Backend], members))
        for member in members:
            assert member.observe().z == 0.0
