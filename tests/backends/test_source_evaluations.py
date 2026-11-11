# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Count actual RK source dispatch, including rejection and excluding stage reuse."""

from contextlib import closing
from typing import Any

import numpy as np
import pytest

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
    with closing(backend_class(problem, run_request, grid=narrow_grid)) as engine:
        engine.set_initial(problem.initial.field0, None)
        assert engine.metadata()["source_evaluations"] == 0
        engine.propagate_to(0.001)
        assert engine.metadata()["source_evaluations"] == 7
        engine.propagate_to(0.002)
        # NumPy intentionally evaluates every stage. Compiled DOPRI5 reuses
        # its FSAL stage; integer work metadata is an exact structural contract.
        expected = 14 if engine.name == "numpy_ref" else 13
        assert engine.metadata()["source_evaluations"] == expected
        engine.set_initial(problem.initial.field0, None)
        assert engine.metadata()["source_evaluations"] == 0
        engine.propagate_to(0.001)
        assert engine.metadata()["source_evaluations"] == 7


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
        engine.set_initial(problem.initial.field0, None)
        calls = []
        original_trial = engine._trial_step

        def trial(z, dz):
            before = engine.metadata()["source_evaluations"]
            _, advanced = original_trial(z, dz)
            calls.append(engine.metadata()["source_evaluations"] - before)
            return (4.0 if len(calls) == 1 else 0.01), advanced

        monkeypatch.setattr(engine, "_trial_step", trial)
        dispatched = []
        original_source = engine._nonlinear

        def source(*args, **kwargs):
            dispatched.append(None)
            return original_source(*args, **kwargs)

        monkeypatch.setattr(engine, "_nonlinear", source)
        summary = engine.propagate_to(0.002)
        assert summary.n_rejected == 1
        assert calls[0] == 7
        count = engine.metadata()["source_evaluations"]
        assert count == sum(calls)
        assert count > sum(calls[1:])
        assert count == len(dispatched)
        assert np.isfinite(engine.observe().field).all()
