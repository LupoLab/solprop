# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify actual CPU forwarding and library limits over prepared session lifetimes."""

import os
from pathlib import Path

import pytest

from backends.julia.cases import cpu_case
from solprop.backends.julia import _accelerate
from solprop.backends.julia._runtime import get_runtime
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


@pytest.mark.parametrize("first_closed", [0, 1])
def test_verified_cpu_provider_and_overlapping_preparations(first_closed):
    setup_id = os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]
    runtime = get_runtime(setup_id=setup_id)
    provider = runtime.cpu_blas
    assert provider is not None
    assert provider.provider in ("accelerate", "openblas")
    assert provider.interface in ("ilp64", "lp64")
    apple = _accelerate._library() if provider.provider == "accelerate" else None
    original = None if apple is None else apple.num_threads
    case = cpu_case()
    owners = [
        JuliaBackend(
            case.problem,
            case.request,
            options=ExecutionOptions(backend_options={"setup_id": setup_id}),
        )
        for _ in range(2)
    ]
    try:
        prepared = [owner.prepare(case.problem, case.request) for owner in owners]
        for factory in prepared:
            with factory.start(Start.entrance(case.problem), owns=False) as session:
                record = session.execution_record()
                assert record["blas_provider"] == provider.provider
                assert record["blas_primary_library"] == Path(provider.library).name
                assert record["blas_integer_interface"] == provider.interface
                assert record["blas_thread_scope"] == provider.thread_scope
                if provider.provider == "accelerate":
                    assert apple is not None and apple.num_threads == 1
                    assert record["accelerate_threading_mode"] == "single"
                    assert record["accelerate_threads"] == 1
                    assert record["blas_fallback_library"]
                else:
                    assert record["blas_provider_reason"]
                    assert "accelerate_threads" not in record
        owners[first_closed].close()
        if apple is not None:
            assert apple.num_threads == 1
        owners[1 - first_closed].close()
        if apple is not None:
            assert apple.num_threads == original
    finally:
        for owner in owners:
            owner.close()
