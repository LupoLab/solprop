# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reject invalid batch contracts and isolate member failures at the shared boundary."""

import dataclasses
import inspect
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from backends import test_batch_lockstep as fixtures
from backends.test_driver import ScriptedDriver
from solprop.backends import Backend, select_batch
from solprop.backends.batch import LockstepRunner
from solprop.backends.numba_cpu.batch import NumbaCpuBatchBackend
from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.stepper.tableau import TSIT5


def test_batch_registry_dispatches_optional_native_without_installation(monkeypatch):
    package = ModuleType("solprop_native")
    batch_module = ModuleType("solprop_native.batch")
    dispatched = []

    def constructor(members):
        dispatched.extend(members)
        raise BackendError("native allocation intentionally unavailable")

    monkeypatch.setattr(package, "NativeBackend", object, raising=False)
    monkeypatch.setattr(batch_module, "NativeBatchBackend", constructor, raising=False)
    monkeypatch.setitem(sys.modules, "solprop_native", package)
    monkeypatch.setitem(sys.modules, "solprop_native.batch", batch_module)
    with ExitStack() as cleanup:
        owner = fixtures.loaded(fixtures.simulation(0, "numpy_ref"), cleanup)
        monkeypatch.setattr(owner, "name", "native")
        with pytest.raises(
            BackendError, match="native allocation intentionally unavailable"
        ):
            select_batch(cast(Sequence[Backend], [owner]))
        assert dispatched == [owner]


@pytest.mark.parametrize("contract", ["tableau", "schedule", "planes", "statistics"])
def test_batch_rejects_different_member_execution_contract(contract):
    first = fixtures.simulation(0, "numpy_ref")
    second = fixtures.simulation(1, "numpy_ref")
    changes = {
        "tableau": {"integrator": RungeKutta(TSIT5)},
        "schedule": {"schedule": FixedSchedule.uniform(0.0, 0.001, 0.000125)},
        "planes": {"save_planes": [0.001]},
        "statistics": {"stats": StatsSpec(names=("energy",))},
    }
    second.settings = dataclasses.replace(second.settings, **changes[contract])
    with ExitStack() as cleanup:
        members = [fixtures.loaded(one, cleanup) for one in (first, second)]
        with pytest.raises(BackendError, match="same"):
            LockstepRunner(members)
        assert all(member.observe().z == 0.0 for member in members)


@pytest.mark.parametrize("changed", ["mask", "shape", "advanced"])
def test_batch_rejects_incompatible_or_advanced_members(changed):
    with ExitStack() as cleanup:
        first, second = [
            fixtures.loaded(fixtures.simulation(i, "numpy_ref"), cleanup)
            for i in range(2)
        ]
        problem = second._problem
        if changed == "mask":
            mask = problem.grids.norm_mask.copy()
            mask[0] = not mask[0]
            second._problem = dataclasses.replace(
                problem, grids=dataclasses.replace(problem.grids, norm_mask=mask)
            )
        elif changed == "shape":
            second._problem = dataclasses.replace(
                problem,
                shape=(2, *problem.shape[1:]),
                initial=dataclasses.replace(
                    problem.initial, field0=np.repeat(problem.initial.field0, 2, axis=0)
                ),
            )
        else:
            second.propagate_to(0.0005)
        with pytest.raises(BackendError, match=r"mask|shapes|entrance"):
            LockstepRunner([first, second])


def test_batch_refuses_reloading_a_shared_field_and_mismatched_statistics(monkeypatch):
    with ExitStack() as cleanup:
        members = [
            fixtures.loaded(fixtures.simulation(i, "numpy_ref"), cleanup)
            for i in range(2)
        ]
        runner = LockstepRunner(members)
        cleanup.callback(runner.close)
        with pytest.raises(BackendError, match="independently loaded"):
            runner.set_initial(members[0]._problem.initial.field0, None)
        monkeypatch.setattr(members[1], "stats", lambda spec: {"wrong": np.zeros(1)})
        with pytest.raises(BackendError, match="different statistic names"):
            runner.stats(StatsSpec(names=("energy",)))
        assert runner.available_device_memory() is None


@pytest.mark.parametrize("kind", ["common", "numpy", "numba"])
@pytest.mark.parametrize("norm", [float("nan"), float("inf"), -1.0])
def test_batch_names_the_member_behind_an_invalid_norm(monkeypatch, kind, norm):
    # The same value refuses the member's own StepHistory/StepSummary, so a
    # singleton does not survive it either: what the batch owes its caller is the
    # index of the point that produced it, not an anonymous history error.
    with ExitStack() as cleanup:
        name = "numba_cpu" if kind == "numba" else "numpy_ref"
        healthy = fixtures.loaded(fixtures.simulation(1, name), cleanup)
        member = fixtures.loaded(fixtures.simulation(0, name), cleanup)
        solo = fixtures.loaded(fixtures.simulation(0, name), cleanup)
        if kind == "common":
            for one in (member, solo):
                monkeypatch.setattr(
                    one, "_trial_step", lambda z, dz, one=one: (norm, one._state)
                )
            runner = LockstepRunner([healthy, member])
        elif kind == "numpy":
            monkeypatch.setattr(passes, "error_norm", lambda *args: norm)
            runner = select_batch(cast(Sequence[Backend], [healthy, member]))
        else:
            for one in (member, solo):
                monkeypatch.setattr(one, "_score_trial", lambda whole, dz: norm)
            runner = select_batch(cast(Sequence[Backend], [healthy, member]))
        cleanup.callback(runner.close)
        offender = 0 if kind == "numpy" else 1
        with pytest.raises(
            BackendError, match=f"member {offender} returned an invalid"
        ):
            runner.propagate_to(0.0005)
        # The same member run alone is refused too: a non-finite norm at the step that
        # produced it, by the replay that cannot reject it, and a negative one at the
        # save plane, through the step record that cannot carry the value either.
        refusal = (
            pytest.raises(BackendError, match="must be non-negative and finite")
            if np.isfinite(norm)
            else pytest.raises(ConvergenceError, match="non-finite error norm")
        )
        with refusal:
            solo.propagate_to(0.0005)


def test_reference_batch_refuses_mixed_noise_models():
    with ExitStack() as cleanup:
        members = [
            fixtures.loaded(fixtures.simulation(i, "numpy_ref", noise), cleanup)
            for i, noise in enumerate((None, "modified"))
        ]
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        with pytest.raises(BackendError, match="compatible noise models"):
            runner.propagate_to(0.0005)


def test_registry_refuses_unknown_backend_without_taking_ownership(monkeypatch):
    with ExitStack() as cleanup:
        owner = fixtures.loaded(fixtures.simulation(0, "numpy_ref"), cleanup)
        monkeypatch.setattr(owner, "name", "future_backend")
        with pytest.raises(BackendError, match="future_backend"):
            select_batch(cast(Sequence[Backend], [owner]))
        assert owner.observe().z == 0.0


def test_batch_requires_complete_backend_protocol():
    simulation = fixtures.simulation(0, "numpy_ref")
    owner = ScriptedDriver(simulation.translate())
    try:
        owner.set_initial(simulation.translate().initial.field0, None)
        with pytest.raises(BackendError, match="backend protocol"):
            LockstepRunner([owner])
    finally:
        owner.close()


@pytest.mark.parametrize(
    ("factory", "wrong_backend"),
    [(NumpyBatchBackend, "numba_cpu"), (NumbaCpuBatchBackend, "numpy_ref")],
)
def test_concrete_batch_checks_family_without_the_session_type_hook(
    factory, wrong_backend
):
    with ExitStack() as cleanup:
        owner = fixtures.loaded(fixtures.simulation(0, wrong_backend), cleanup)
        instance = factory.__new__(factory)
        # Production does not install the test session's annotation hook. Exercise
        # its explicit family refusal, which that hook would otherwise preempt.
        initialize = inspect.unwrap(factory.__init__)
        with pytest.raises(BackendError, match="requires"):
            initialize(instance, [owner])
        assert owner.observe().z == 0.0


def test_batch_close_releases_other_members_when_one_close_fails(monkeypatch):
    with ExitStack() as cleanup:
        members = [
            fixtures.loaded(fixtures.simulation(i, "numpy_ref"), cleanup)
            for i in range(2)
        ]
        runner = LockstepRunner(members)
        close = cast(Backend, members[1]).close

        def failing_close():
            close()
            raise BackendError("member cleanup failed")

        monkeypatch.setattr(members[1], "close", failing_close)
        with pytest.raises(BackendError, match="cleanup failed"):
            runner.close()
        assert runner.memory_ledger() == {}
        for member in members:
            with pytest.raises(BackendError, match="closed"):
                member.observe()
        runner.close()
