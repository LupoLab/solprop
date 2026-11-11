# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate DF32 PythonCall binding ownership and refusal before mutation."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.test_cpu_df32 import _case, _prepared
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import Outcome
from solprop.backends.julia._stop import StopLease
from solprop.backends.julia._wide import bind_df32_start
from solprop.backends.session import Start, StopToken
from solprop.exceptions import BackendError

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


@pytest.mark.parametrize("low", [False, True])
@pytest.mark.parametrize("mismatch", ["identity", "coordinate"])
def test_factory_rejects_mismatched_binding_without_losing_preparation(low, mismatch):
    case = _case(low, 8, single=True)
    prepared = _prepared(case)
    owner = prepared._owner
    runtime = prepared._runtime
    assert owner._df32_operator is not None
    binding = bind_df32_start(owner._df32_operator, prepared._transfer, 0.0)
    transfer = replace(binding.run) if mismatch == "identity" else binding.run
    z = 0.01 if mismatch == "coordinate" else 0.0
    field = case.problem.initial.field0.copy()
    before = field.tobytes()
    lease = StopLease(runtime, StopToken())
    try:
        with pytest.raises(BackendError, match=r"exact owned run|actual bound start"):
            runtime.own_call(
                "start_low_storage" if low else "start_runge_kutta",
                prepared._preparation,
                transfer,
                field,
                z,
                0,
                True,
                lease.flag,
                None,
                False,
                None,
                None,
                binding,
                owners=(lease,),
            )
        assert field.tobytes() == before
        assert not owner.closed and len(owner._sessions) == 0
        finish = float(case.request.observation.save_planes[-1])
        start = Start(
            field=field,
            z=finish,
            plane_index=len(case.request.observation.save_planes),
            plane_pending=False,
            resume=True,
        )
        with prepared.start(start, owns=False) as session:
            result = drive(Started(session), export=IDENTITY)
            assert result.status.outcome is Outcome.COMPLETED
            assert result.status.attempts == 0
            assert session.execution_record()["source_evaluations"] == 0
            assert start.field is None
        assert not owner.closed
    finally:
        lease.close()
        owner.close()


@pytest.mark.parametrize("low", [False, True])
def test_off_edge_start_refuses_before_any_numerical_factory(low, monkeypatch):
    case = _case(low, 8, single=True)
    prepared = _prepared(case)
    owner = prepared._owner
    start = Start(field=case.problem.initial.field0.copy(), z=0.006)
    original = start.field
    assert original is not None
    before = original.tobytes()

    def no_julia(*_, **__):
        pytest.fail("the invalid actual start reached Julia construction")

    try:
        with monkeypatch.context() as scope:
            scope.setattr(type(prepared._runtime), "own_call", no_julia)
            with pytest.raises(BackendError, match="replay"):
                prepared.start(start)
        assert start.field is original and original.tobytes() == before
        assert not owner.closed and len(owner._sessions) == 0
        assert np.all(np.isfinite(original))
    finally:
        owner.close()
