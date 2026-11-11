# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify narrow-state group delivery and DF32 certificate failures."""

import os
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.test_cpu_df32 import _case, _large_phase_case, _run
from sessions.executors import IDENTITY
from solprop.backends.events import ObservationReady, Outcome, Terminal, join_attempts
from solprop.backends.export import ExportRecipe
from solprop.backends.julia._group import JuliaGroupSession
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.backends.sessions import start_group
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _scaled(case, factor):
    field = factor * case.problem.initial.field0
    field.setflags(write=False)
    return Translation(
        replace(case.problem, initial=replace(case.problem.initial, field0=field)),
        case.request,
    )


def _group(cases, wide, cleanup):
    owners = []
    for case in cases:
        owner = JuliaBackend(
            case.problem,
            case.request,
            options=ExecutionOptions(
                backend_options={
                    "setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"],
                    "wide": wide,
                }
            ),
        )
        cleanup.callback(owner.close)
        owners.append(owner)
    starts = [Start.entrance(case.problem) for case in cases]
    group = start_group(
        owners,
        [case.problem for case in cases],
        [case.request for case in cases],
        starts,
    )
    cleanup.callback(group.close)
    assert isinstance(group, JuliaGroupSession)
    assert all(start.field is None for start in starts)
    return group, owners


@pytest.mark.parametrize("wide", ["binary64", "df32"])
@pytest.mark.parametrize("low", [False, True])
def test_binary32_groups_meet_refined_reference_and_exact_delivery_contract(wide, low):
    factors = (0.75, 1.0)
    cases = [_scaled(_case(low, 8, single=True), factor) for factor in factors]
    fields = [[], []]
    batches = []
    recipe = ExportRecipe(power=True, optics=False)
    with ExitStack() as cleanup:
        group, owners = _group(cases, wide, cleanup)
        while not group.finished:
            batch = group.advance()
            batches.append(batch)
            for index, events in enumerate(batch.members):
                for event in events:
                    if isinstance(event, ObservationReady):
                        fields[index].append(group.export(index, event.ticket, recipe))
            group.acknowledge(batch.receipt)
        assert batches[-1].outcome is Outcome.COMPLETED
        for index, factor in enumerate(factors):
            status = group.member_status(index)
            assert status.attempts == 8 and status.rejected == 0
            record = group.execution_record(index)
            assert record["precision"] == "single" and record["wide"] == wide
            assert record["batch_strategy"] == "julia_group_session"
            history, _ = join_attempts(
                (event.history, event.step_stats)
                for batch in batches
                for event in batch.members[index]
            )
            assert history.n_steps == 8 and np.all(history.accepted)
            coarse = _run(_scaled(_case(low, 16), factor))
            reference = _run(_scaled(_case(low, 32), factor))
            for actual, previous, expected in zip(
                fields[index], coarse.fields, reference.fields, strict=True
            ):
                # Ordinary replay uses SEP33's fixed field/energy envelopes.
                # Independent binary64 refinement must settle 100 times tighter.
                scale = np.linalg.norm(expected)
                assert scale > 0.0 and np.all(np.isfinite(actual))
                assert np.linalg.norm(previous - expected) / scale < 1e-5
                assert np.linalg.norm(actual - expected) / scale < 1e-3
                energy = float(np.vdot(expected, expected).real)
                assert abs(float(np.vdot(previous, previous).real) / energy - 1) < 1e-6
                assert abs(float(np.vdot(actual, actual).real) / energy - 1) < 1e-4
                assert actual.dtype == np.complex128 and not actual.flags.writeable
        assert any(key.startswith("group.") for key in owners[0].memory_ledger())
        assert not any(key.startswith("group.") for key in owners[1].memory_ledger())
        group.close()
        assert all(owner.closed for owner in owners)
        assert all(np.all(np.isfinite(field)) for member in fields for field in member)


@pytest.mark.parametrize("low", [False, True])
def test_df32_group_failure_preserves_the_refusing_member_before_source_work(low):
    cases = [_case(low, 8, single=True), _large_phase_case(low)]
    with ExitStack() as cleanup:
        group, _ = _group(cases, "df32", cleanup)
        launch = group.advance()
        event = launch.members[1][0]
        assert isinstance(event, ObservationReady)
        retained = group.export(1, event.ticket, IDENTITY).copy()
        group.acknowledge(launch.receipt)
        failed = group.advance()
        assert failed.outcome is Outcome.FAILED
        assert "wide certificate refused" in str(failed.error)
        assert group.member_status(0).z > group.member_status(1).z
        for index, events in enumerate(failed.members):
            (terminal,) = events
            assert isinstance(terminal, Terminal) and terminal.error is failed.error
            assert terminal.ticket is not None
            assert terminal.history is not None
            assert terminal.history.n_steps == group.member_status(index).attempts
        refusing = failed.members[1][0]
        assert isinstance(refusing, Terminal)
        ticket = refusing.ticket
        assert ticket is not None
        # The rejected certificate permits no numerical write to this member.
        np.testing.assert_array_equal(group.export(1, ticket, IDENTITY), retained)
        assert group.execution_record(1)["source_evaluations"] == 0
        group.acknowledge(failed.receipt)
