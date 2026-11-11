# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify isolated Julia timing probes against matched reference trials."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.bounds import rk_rounding_budget
from backends.julia.cases import adams_case, cpu_case, low_storage_case
from backends.julia.split_cases import linear_split_rounding_bound, split_case
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions, SessionError, Start
from solprop.exceptions import BackendError
from solprop.translate.problem import Initial, Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _owner(case, **context):
    return JuliaBackend(
        case.problem,
        case.request,
        options=ExecutionOptions(
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
        ),
        **context,
    )


@pytest.mark.parametrize("factory", [cpu_case, adams_case])
def test_embedded_probe_owns_its_start_and_agrees_with_reference(factory):
    case = factory()
    # Both probes use the request's embedded tableau from the unchanged load. The
    # established eight-step cubic bound conservatively encloses this first trial.
    bound = rk_rounding_budget(cpu_case())
    reference = NumpyRefBackend(case.problem, case.request)
    owner = _owner(case)
    field = np.array(case.problem.initial.field0)
    before = field.copy()
    start = Start(field, 0.0, monitor={"unused": object()})
    try:
        expected = reference.timing_probe(Start.entrance(case.problem))
        actual = owner.timing_probe(start)
        assert start.field is None and not owner._sessions
        np.testing.assert_array_equal(field, before)
        result = None
        for _ in range(2):
            expected.trial(dz=0.0025)
            actual.trial(dz=0.0025)
            result = actual.field()
            assert np.linalg.norm(result - expected.field()) <= bound
            assert result.flags.owndata and result.dtype == np.complex128
            with pytest.raises(BackendError, match="take a trial"):
                actual.field()
        assert result is not None
        retained = result.copy()
        actual.trial(dz=0.001)
        np.testing.assert_array_equal(result, retained)
        assert any(name.startswith("timing.") for name in owner.memory_ledger())
        assert not owner._sessions
    finally:
        reference.close()
        owner.close()
    with pytest.raises(SessionError, match="closed"):
        actual.trial()


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("method", ["rk", "adams"])
@pytest.mark.parametrize("resume", [False, True])
def test_probe_applies_input_noise_and_split_once_without_monitoring(
    kind, method, resume
):
    base, grid, mode = split_case(kind, method, steps=1, rate="varying")
    draw = base.problem.initial.field0 * complex(1 / 64, -1 / 128)
    problem = replace(
        base.problem,
        initial=Initial(base.problem.initial.field0, draw, "input", 0.0),
        flags=replace(base.problem.flags, has_noise=True),
    )
    case = Translation(problem, base.request)
    loaded = problem.initial.field0 if resume else problem.initial.field0 + draw
    budget_case = Translation(
        replace(problem, initial=replace(problem.initial, field0=loaded)), case.request
    )
    # Contractive split propagation: the pre-derived transform/phase bound includes
    # independent transforms and the actual loaded input magnitude.
    bound = linear_split_rounding_bound(budget_case)
    owner = _owner(case, grid=grid, modes=[mode])
    reference = NumpyRefBackend(problem, case.request, grid=grid, modes=[mode])
    try:
        expected = reference.timing_probe(
            Start(problem.initial.field0, 0.0, resume=resume)
        )
        actual = owner.timing_probe(Start(problem.initial.field0, 0.0, resume=resume))
        expected.trial()
        actual.trial()
        assert np.linalg.norm(actual.field() - expected.field()) <= bound
        assert not owner._sessions
    finally:
        reference.close()
        owner.close()


def test_low_storage_probe_has_no_readable_field_and_owner_closes_it():
    case = low_storage_case()
    owner = _owner(case)
    try:
        probe = owner.timing_probe(Start.entrance(case.problem))
        probe.trial()
        probe.trial()
        with pytest.raises(BackendError, match=r"low-storage.*no trial field"):
            probe.field()
        assert any(name.startswith("timing.") for name in owner.memory_ledger())
    finally:
        owner.close()
    with pytest.raises(SessionError, match="closed"):
        probe.trial()
