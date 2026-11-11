# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native groups against their members alone.

A native group (solprop SEP 47 section 6.7) runs each member in its own native session;
the transitional lockstep batch it replaced was deleted in SEP 47's phase 5, with the
fused batch update state it stepped.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from bounds import assert_within, gamma, step_bound
from conftest import PhaseACase
from session_runs import alone, assert_same, grouped, one_thread
from solprop.backends.session import Start
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

from solprop_native import NativeBackend


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_group_members_match_their_lone_sessions(
    phase_a_case: PhaseACase, count: int, weighted: bool
) -> None:
    """Each member of a native group agrees with the same member run alone.

    Both replay one prescribed schedule with the explicit pair, each member in its own
    native session (solprop SEP 47 section 6.7): bit for bit at one thread, where the
    reductions' order is fixed, and otherwise to rounding (solprop SEP 25). Bounds,
    fixed beforehand:

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
    original, run_request, grid = phase_a_case
    spec = StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2)
    controller = run_request.stepper.controller
    if weighted:
        controller = dataclasses.replace(
            controller, weights=np.linspace(0.5, 1.5, grid.n_omega)
        )
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            controller=controller,
            schedule=FixedSchedule.uniform(0.0, 1e-3, 2.5e-4),
        )
    ).for_run(stats=spec)
    problems = [
        dataclasses.replace(
            original,
            initial=dataclasses.replace(
                original.initial, field0=original.initial.field0 * (index + 1)
            ),
        )
        for index in range(count)
    ]
    requests = [run_request] * count
    group = grouped(
        [NativeBackend(one, run_request, grid=grid) for one in problems],
        problems,
        requests,
        [Start.entrance(one) for one in problems],
    )
    singles = [
        alone(NativeBackend(one, run_request, grid=grid), one, run_request, start)
        for one, start in ((one, Start.entrance(one)) for one in problems)
    ]
    field_bound = step_bound(grid.n_to, 4)
    statistic_bound = 2 * math.sqrt(grid.n_to) * field_bound + gamma(grid.n_to)
    for member, single in zip(group, singles, strict=True):
        assert [plane["z"] for plane in member] == [plane["z"] for plane in single]
        if one_thread():
            assert_same(member, single)
        for actual, expected in zip(member, single, strict=True):
            assert_within(actual["field"], expected["field"], field_bound, what="field")
            for name in ("z", "dz", "accepted"):
                np.testing.assert_array_equal(
                    actual["attempts"][name], expected["attempts"][name], err_msg=name
                )
            np.testing.assert_allclose(
                actual["attempts"]["error_norm"],
                expected["attempts"]["error_norm"],
                rtol=field_bound,
                # Twice: the weighted controller's bin weights reach 1.5.
                atol=2 * step_bound(grid.n_to, 1) / controller.rtol,
            )
            np.testing.assert_array_equal(actual["row_z"], expected["row_z"])
            for name, values in expected["rows"].items():
                assert_within(actual["rows"][name], values, statistic_bound, what=name)
            for name, value in expected["stats"].items():
                assert_within(actual["stats"][name], value, statistic_bound, what=name)
