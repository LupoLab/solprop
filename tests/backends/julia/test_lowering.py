# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep descriptor lowering owned, lazy, cached and explicit about admission."""

import gc
import weakref
from dataclasses import fields, replace

import numpy as np
import pytest
from benchmarks.records import SuiteError
from benchmarks.worker import _Loaded

from backends.julia.cases import cpu_case
from solprop.backends.julia import backend
from solprop.backends.julia._lower import lower_envelope, lower_runge_kutta
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError
from solprop.translate.precision import SINGLE


def test_lowering_caches_owned_read_only_arrays_and_releases_its_problem():
    case = cpu_case()
    problem = case.problem
    transfer = lower_envelope(problem)
    assert lower_envelope(problem) is transfer
    assert transfer.operator.shape == (problem.shape[-1], problem.shape[0])
    assert not np.shares_memory(transfer.synthesis, problem.transform.synthesis)
    assert not np.shares_memory(transfer.projection, problem.transform.projection)
    assert not np.shares_memory(transfer.crop_index, problem.grids.crop_index)
    for descriptor in fields(transfer):
        value = getattr(transfer, descriptor.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            assert value.flags.owndata
    assert transfer.kerr is not None
    assert problem.responses.kerr_coefficients is not None
    assert transfer.kerr[0] == problem.responses.kerr_coefficients.coefficient
    original = weakref.ref(problem)
    del case, problem
    gc.collect()
    assert original() is None


def test_run_lowering_preserves_actual_replay_lengths_beyond_the_adaptive_ceiling():
    case = cpu_case(dz_max=1e-5)
    transfer = lower_runge_kutta(case.problem, case.request)
    assert case.request.replay is not None
    assert np.array_equal(transfer.steps, case.request.replay.schedule.steps)
    assert np.array_equal(transfer.edges, case.request.replay.schedule.edges)
    assert not np.shares_memory(transfer.steps, case.request.replay.schedule.steps)
    assert np.all(transfer.steps > transfer.control.dz_max)
    assert np.array_equal(transfer.active, case.problem.grids.active_mask)
    assert transfer.active.dtype == np.bool_
    assert np.all(transfer.weights == 1.0)
    assert not transfer.matrix.flags.writeable


def test_binary32_refuses_before_a_runtime_is_selected(monkeypatch):
    case = cpu_case()

    def no_runtime(**_):
        pytest.fail("admission must precede Julia initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    with pytest.raises(BackendError, match="binary32"):
        JuliaBackend(replace(case.problem, precision=SINGLE), case.request)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": True}, "unsupported|does not support"),
        ({"wide": 1}, "wide"),
        ({"wide": "double"}, "wide"),
        ({"setup_id": 4}, "setup_id"),
        ({"setup_id": ""}, "setup_id"),
        ({"wide": "df32"}, "binary64"),
    ],
)
def test_execution_options_refuse_before_preparation(options, message):
    case = cpu_case()
    with pytest.raises(BackendError, match=message):
        JuliaBackend(
            case.problem,
            case.request,
            options=ExecutionOptions(backend_options=options),
        )


def test_execution_owner_copies_options_without_loading_julia():
    case = cpu_case()
    options = ExecutionOptions(backend_options={"wide": "binary64", "setup_id": "one"})
    owner = JuliaBackend(case.problem, case.request, options=options)
    owned_options = options.backend_options
    assert isinstance(owned_options, dict)
    owned_options["setup_id"] = "two"
    assert owner.metadata()["backend_options"] == {
        "wide": "binary64",
        "setup_id": "one",
    }
    assert "julia_threads" not in owner.metadata()
    assert (
        owner.memory_ledger()["transfer.operator"] == case.problem.initial.field0.nbytes
    )
    owner.close()
    assert owner.closed


def test_foreign_numerical_request_refuses_before_runtime_selection(monkeypatch):
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)

    def no_runtime(**_):
        pytest.fail("a foreign request must refuse before Julia initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    foreign = replace(
        case.request, error_control=replace(case.request.error_control, dz_max=1e-4)
    )
    with pytest.raises(BackendError, match="error control"):
        owner.prepare(case.problem, foreign)
    owner.close()


def test_legacy_benchmark_window_refuses_an_execution_only_owner():
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    try:
        with pytest.raises(SuiteError, match="driver load boundary"):
            _Loaded(owner)
    finally:
        owner.close()
