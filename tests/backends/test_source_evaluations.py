# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Count actual RK source dispatch, including rejection and excluding stage reuse."""

from contextlib import closing
from typing import Any

import numpy as np
import pytest

from sessions.runs import Records
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper import FixedSchedule


@pytest.mark.medium
def test_rk_counts_reuse_and_reset(backend_class, linear_problem, narrow_grid):
    problem, run_request = linear_problem(
        narrow_grid,
        save_planes=[0.001, 0.002],
        schedule=FixedSchedule.uniform(0.0, 0.002, 0.001),
        auto_init=False,
    )
    # The record at each plane of the backend's own session, then of a second session
    # on the same backend: the count restarts with the run.
    with closing(backend_class(problem, run_request, grid=narrow_grid)) as engine:
        first = Records()
        session_run(engine, problem, run_request, owns=False, between=first)
        counts = [record["source_evaluations"] for _, record in first.taken]
        # NumPy intentionally evaluates every stage. Compiled DOPRI5 reuses
        # its FSAL stage; integer work metadata is an exact structural contract.
        expected = 14 if engine.name == "numpy_ref" else 13
        assert counts == [0, 7, expected]
        second = Records()
        session_run(
            engine, problem, run_request, owns=False, between=second, until=0.001
        )
        counts = [record["source_evaluations"] for _, record in second.taken]
        assert counts == [0, 7]


@pytest.mark.medium
def test_python_counter_tracks_dispatch_through_rejection(
    linear_problem,
    narrow_grid,
    monkeypatch,
):
    # The reference's loop, whose trial the scripted norms below replace. The native
    # march (SEP 45) takes its norms itself, and no switch selects solprop's loop on
    # native any more (SEP 47 section 6.3); test_rk_counts_reuse_and_reset counts it.
    problem, run_request = linear_problem(
        narrow_grid,
        save_planes=[0.002],
        dz_init=0.001,
        auto_init=False,
    )
    reference: Any = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    with closing(reference) as engine:
        calls = []
        build = engine._runge_kutta

        def scripted(workspace):
            # The session's own Runge-Kutta engine, its trial's norms scripted.
            stepper = build(workspace)
            original_trial = stepper.trial

            def trial(z, dz):
                before = engine.metadata()["source_evaluations"]
                original_trial(z, dz)
                calls.append(engine.metadata()["source_evaluations"] - before)
                return 4.0 if len(calls) == 1 else 0.01

            monkeypatch.setattr(stepper, "trial", trial)
            return stepper

        monkeypatch.setattr(engine, "_runge_kutta", scripted)
        dispatched = []
        original_source = engine.workspace.nonlinear

        def source(*args, **kwargs):
            dispatched.append(None)
            return original_source(*args, **kwargs)

        monkeypatch.setattr(engine.workspace, "nonlinear", source)
        record: dict[str, Any] = {}
        result = session_run(engine, problem, run_request, record=record)
        history, _ = result.joined
        assert int(np.sum(~np.asarray(history.accepted))) == 1
        assert calls[0] == 7
        count = record["source_evaluations"]
        assert count == sum(calls)
        assert count > sum(calls[1:])
        assert count == len(dispatched)
        assert np.isfinite(result.fields[-1]).all()
