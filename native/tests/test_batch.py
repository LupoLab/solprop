# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare native batch updates with independently owned Kokkos trajectories."""

from __future__ import annotations

import dataclasses
import math
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest
from bounds import assert_within, gamma, step_bound
from conftest import PhaseACase
from solprop.backends.protocol import BatchBackend
from solprop.exceptions import HostMarchWarning
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep

from solprop_native import (
    NativeBackend,
    NativeBatchBackend,
    _core,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_batch_matches_member_fields_histories_and_statistics(
    phase_a_case: PhaseACase, count: int, weighted: bool
) -> None:
    """Each batch member agrees with the same member run alone.

    Both replay one prescribed schedule with the explicit pair: the same discrete
    computation, which the batch's lockstep reductions may order differently, so the
    two agree to rounding (solprop SEP 25), not bits. Bounds, fixed beforehand:

    - the field, the transforms' budget of four steps (:func:`bounds.step_bound`);
    - the step positions, lengths, acceptances and row positions exactly, as the
      schedule prescribes them;
    - each error norm, one step's budget over ``rtol`` (the embedded difference's
      rounding is relative to the field, scored against ``rtol |E|``), twice for the
      weighted controller's bins, plus the field's budget relatively;
    - the energy and peak-power statistics, the field's budget twice (a squared
      modulus) times the square root of the samples (a peak read from a normwise
      bound), plus the energy sum's own ``gamma(n)``.
    """
    original, grid = phase_a_case
    spec = StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2)
    controller = original.stepper.controller
    if weighted:
        controller = dataclasses.replace(
            controller, weights=np.linspace(0.5, 1.5, grid.n_omega)
        )
    problem = dataclasses.replace(
        original,
        stats_spec=spec,
        stepper=dataclasses.replace(
            original.stepper,
            controller=controller,
            schedule=FixedSchedule.uniform(0.0, 1e-3, 2.5e-4),
        ),
    )
    with ExitStack() as cleanup:
        members: list[NativeBackend] = []
        singles: list[NativeBackend] = []
        for index in range(count):
            one = dataclasses.replace(
                problem,
                initial=dataclasses.replace(
                    problem.initial, field0=problem.initial.field0 * (index + 1)
                ),
            )
            for destination in (members, singles):
                owner = NativeBackend(one, grid=grid)
                cleanup.callback(owner.close)
                owner.set_initial(one.initial.field0, None)
                destination.append(owner)
        batch = NativeBatchBackend(members)
        cleanup.callback(batch.close)
        assert isinstance(batch, BatchBackend)
        assert batch.max_batch == count
        batch.propagate_to(1e-3)
        statistics = batch.stats(spec)
        field_bound = step_bound(grid.n_to, 4)
        statistic_bound = 2 * math.sqrt(grid.n_to) * field_bound + gamma(grid.n_to)
        for index, (actual, single) in enumerate(
            zip(batch.observe_members(), singles, strict=True)
        ):
            single.propagate_to(1e-3)
            expected = single.observe()
            assert_within(
                actual.require_field(),
                expected.require_field(),
                field_bound,
                what="field",
            )
            for name in ("z", "dz", "accepted"):
                np.testing.assert_array_equal(
                    getattr(actual.step_history, name),
                    getattr(expected.step_history, name),
                    err_msg=name,
                )
            np.testing.assert_allclose(
                actual.step_history.error_norm,
                expected.step_history.error_norm,
                rtol=field_bound,
                # Twice: the weighted controller's bin weights reach 1.5.
                atol=2 * step_bound(grid.n_to, 1) / controller.rtol,
            )
            np.testing.assert_array_equal(actual.step_stats.z, expected.step_stats.z)
            np.testing.assert_array_equal(
                actual.step_stats.step, expected.step_stats.step
            )
            for name in expected.step_stats.values:
                assert_within(
                    actual.step_stats.values[name],
                    expected.step_stats.values[name],
                    statistic_bound,
                    what=name,
                )
            for name, value in single.stats(spec).items():
                assert_within(statistics[name][index], value[0], statistic_bound)
        assert batch.metadata()["batch_fused_operations"] == ("accept_error_partials",)
        sizing = cast(
            dict[str, dict[str, int]], batch.metadata()["batch_execution_sizing"]
        )
        elements = count * math.prod(problem.shape[1:])
        assert sizing[f"kernels.stage.{elements}"]["elements"] == elements
        assert sizing[f"kernels.stage.{elements}"]["threads"] >= 1
        assert batch.memory_ledger()["batch/update_descriptors"] > 0
        batch.close()
        batch.close()
        assert batch.memory_ledger() == {}


def test_native_batch_rejects_unprepared_and_closed_members(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    with ExitStack() as cleanup:
        owner = NativeBackend(problem, grid=grid)
        cleanup.callback(owner.close)
        owner.set_initial(problem.initial.field0, None)
        state = owner._native_state  # pyright: ignore[reportPrivateUsage]
        batch = _core.BatchState([state])
        cleanup.callback(batch.close)
        with pytest.raises(RuntimeError, match="prepare this trial"):
            batch.accept(1e-4)
        owner.close()
        with pytest.raises(RuntimeError, match="closed"):
            batch.accept(1e-4)
        batch.close()
        with pytest.raises(RuntimeError, match="closed"):
            batch.accept(1e-4)


def test_a_batch_of_adams_members_replays_on_solprop_s_march(
    phase_a_case: PhaseACase,
) -> None:
    """A lockstep batch replays its Adams members through solprop's march (SEP 39).

    The batch advances each member's `replay_step`, the Python march, so it says so
    once and its members' marches record it.
    """
    original, grid = phase_a_case
    problem = dataclasses.replace(
        original,
        stepper=dataclasses.replace(
            original.stepper,
            integrator=Multistep(),
            schedule=FixedSchedule.uniform(0.0, 1e-3, 2.5e-4),
        ),
    )
    with ExitStack() as cleanup:
        members: list[NativeBackend] = []
        for _ in range(2):
            owner = NativeBackend(problem, grid=grid)
            cleanup.callback(owner.close)
            owner.set_initial(problem.initial.field0, None)
            members.append(owner)
        with pytest.warns(HostMarchWarning, match="lockstep"):
            batch = NativeBatchBackend(members)
        cleanup.callback(batch.close)
        for member in members:
            assert member.metadata()["adams_march"] == "host"
