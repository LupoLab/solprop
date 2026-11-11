# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify accepted-step split damping through complete managed Julia sessions."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.split_cases import linear_split_rounding_bound, split_case
from backends.julia.test_cpu_observations import _owner
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.session import Start

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _run(case, grid, mode, *, julia, start=None):
    with (
        _owner(case, grid, mode, julia=julia)
        .prepare(case.problem, case.request)
        .start(Start.entrance(case.problem) if start is None else start) as session
    ):
        result = drive(Started(session), export=IDENTITY, deadline=60.0)
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        return result, session.execution_record(), session.monitor_record()


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
@pytest.mark.parametrize("steps", [1, 8])
def test_split_without_source_or_observers_matches_each_accepted_step(
    kind, method, steps
):
    case, grid, mode = split_case(kind, method, steps=steps)
    bound = linear_split_rounding_bound(case)
    actual, record, _ = _run(case, grid, mode, julia=True)
    expected, _, _ = _run(case, grid, mode, julia=False)
    assert actual.status.attempts == expected.status.attempts == steps
    assert actual.status.rejected == expected.status.rejected == 0
    # Adams skips proven absent sources. RK methods still invoke their zero
    # source at every stage; the ledger records that actual work faithfully.
    if method == "adams":
        assert record["source_evaluations"] == 0
    elif method == "low_storage":
        assert record["source_evaluations"] == 4 * steps
    else:
        assert (
            record["source_evaluations"]
            == len(case.request.integration.tableau.nodes) * steps
        )
    for got, want in zip(actual.fields, expected.fields, strict=True):
        assert np.linalg.norm((got - want).ravel()) <= bound
        assert np.all(np.isfinite(got))
        assert np.count_nonzero(got[..., ~case.problem.grids.active_mask]) == 0
    initial_norm = np.linalg.norm(actual.fields[0].ravel())
    assert np.linalg.norm(actual.fields[-1].ravel()) < 0.95 * initial_norm


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_uniform_split_has_the_independent_exact_amplitude_decay(kind):
    case, grid, mode = split_case(kind, rate="uniform")
    bound = linear_split_rounding_bound(case)
    actual, _, _ = _run(case, grid, mode, julia=True)
    rate = case.problem.linear.phi_constant
    assert rate is not None
    initial = np.where(case.problem.grids.active_mask, case.problem.initial.field0, 0.0)
    # This scalar temporal multiplier commutes with every diagonal spectral flow.
    # The analytic solution is the sole final linear flow times exp(-alpha*z/2).
    expected = initial * np.exp(rate * 8e-6 - 0.1)
    assert np.linalg.norm((actual.fields[-1] - expected).ravel()) <= bound


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_split_monitor_records_exact_removal_and_never_omits_a_quiet_collar(kind):
    case, grid, mode = split_case(kind, rate="uniform", observers=("monitor",))
    monitor = case.request.boundary.monitor
    assert monitor is not None
    monitor.check = replace(monitor.check, quiet_collar=True)
    bound = linear_split_rounding_bound(case)
    actual, _, record = _run(case, grid, mode, julia=True)
    assert record is not None and record["collar_quiet"] is True
    checks = record["quiet_checks"]
    assert isinstance(checks, int) and checks >= 2
    # Uniform power loss removes exactly 1-exp(-alpha*L) of launch energy.
    # The deliberately loose full-run field budget also bounds each inverse and
    # energy reduction in this short case. It gives q=2e+e² in normalized energy.
    # Each of eight removed-energy terms contributes at most q, the launch
    # denominator at most q; the ratio error is at most (8+1)q/(1-q).
    relative = bound / np.linalg.norm(actual.fields[0].ravel())
    energy_error = 2 * relative + relative**2
    assert 0 < energy_error < 0.5
    allowance = 9 * energy_error / (1 - energy_error)
    removed = record["removed_fraction"]
    assert isinstance(removed, float)
    assert abs(removed + np.expm1(-0.2)) <= allowance
    assert np.linalg.norm(actual.fields[-1].ravel()) < 0.95 * np.linalg.norm(
        actual.fields[0].ravel()
    )


@pytest.mark.parametrize("statistics", [False, True])
@pytest.mark.parametrize("monitor", [False, True])
def test_statistics_and_monitor_are_independent_of_required_split_finishing(
    statistics, monitor
):
    observers = tuple(
        name
        for name, enabled in (("statistics", statistics), ("monitor", monitor))
        if enabled
    )
    case, grid, mode = split_case(observers=observers)
    baseline, _, _ = split_case()
    bound = linear_split_rounding_bound(case)
    actual, _, record = _run(case, grid, mode, julia=True)
    expected, _, _ = _run(baseline, grid, mode, julia=False)
    assert np.linalg.norm((actual.fields[-1] - expected.fields[-1]).ravel()) <= bound
    observations = [
        event
        for batch in actual.batches
        for event in batch.events
        if isinstance(event, ObservationReady)
    ]
    assert all(bool(event.stats) is statistics for event in observations)
    assert (record is not None) is monitor
    if record is not None:
        watched, skipped = record["steps"], record["skipped"]
        assert isinstance(watched, int) and isinstance(skipped, int)
        assert watched + skipped == 8


@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
def test_field_restart_does_not_repeat_the_already_applied_split(method):
    case, grid, mode = split_case(method=method)
    bound = linear_split_rounding_bound(case)
    expected, _, _ = _run(case, grid, mode, julia=False)
    # Restart at the half-length plane with its already damped accepted field.
    start = Start(
        field=expected.fields[1].copy(),
        z=4e-6,
        plane_index=1,
        plane_pending=False,
        resume=True,
    )
    actual, _, _ = _run(case, grid, mode, julia=True, start=start)
    assert actual.status.attempts == 4 and actual.status.rejected == 0
    assert np.linalg.norm((actual.fields[-1] - expected.fields[-1]).ravel()) <= bound
