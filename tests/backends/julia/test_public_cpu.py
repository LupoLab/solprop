# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise lazy public selection and request-specific Julia CPU preparation."""

import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from benchmarks.builders import build
from benchmarks.registry import CASES
from benchmarks.worker import TIMING_BOUNDARY, timed_run

from backends.julia.cases import adams_case, cpu_case, low_storage_case
from backends.julia.test_node_plan import (
    _memory_case,
    _workspace_payload,
)
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends import registry
from solprop.backends.julia import backend
from solprop.backends.julia._node_plan import plan_julia_nodes
from solprop.backends.julia._session import JuliaSession
from solprop.backends.session import ExecutionOptions, Start
from solprop.driver import SimulationSettings
from solprop.exceptions import BackendError
from solprop.output.schema import FieldRetention
from solprop.stats import StatsSpec
from solprop.stepper import Multistep
from solprop.translate.problem import Translation


def _integer_record(record: Mapping[str, object], name: str) -> int:
    value = record[name]
    assert isinstance(value, int)
    return value


def test_public_selection_is_lazy_and_reports_the_missing_extra(monkeypatch):
    case = cpu_case()
    present = False
    find_spec = importlib.util.find_spec
    imported = {name: sys.modules.get(name) for name in ("juliacall", "solprop_native")}

    def probe(name, *args):
        if name == "juliacall":
            return importlib.machinery.ModuleSpec(name, None) if present else None
        return find_spec(name, *args)

    def refuse_runtime(**_):
        pytest.fail("selection must not initialize Julia")

    monkeypatch.setattr(importlib.util, "find_spec", probe)
    monkeypatch.setattr(backend, "get_runtime", refuse_runtime)
    assert "julia" not in registry.available()
    with pytest.raises(ImportError, match=r"install solprop\[julia\]"):
        registry.select("julia", case.problem, case.request)
    present = True
    assert "julia" in registry.available()
    owner = registry.select(
        "julia",
        case.problem,
        case.request,
        options=ExecutionOptions(backend_options={"setup_id": "selected"}),
    )
    assert owner.name == "julia"
    assert owner.metadata()["backend_options"] == {"setup_id": "selected"}
    assert owner.metadata()["node_chunk_source"] == "default"
    owner.close()
    assert {name: sys.modules.get(name) for name in imported} == imported
    with pytest.raises(BackendError, match="does not support backend_options"):
        registry.select(
            "julia",
            case.problem,
            case.request,
            options=ExecutionOptions(backend_options={"unknown": True}),
        )


def test_public_adams_settings_admit_the_optional_owner():
    settings = SimulationSettings(
        save_planes=[0.02],
        dz_max=0.02,
        dz_init=0.001,
        backend="julia",
        integrator=Multistep(),
    )
    assert settings.backend == "julia"


@pytest.mark.parametrize("device", ["cuda", "rocm", "metal"])
def test_direct_cpu_owner_refuses_other_devices_before_lowering(device, monkeypatch):
    case = cpu_case()

    def forbidden(*_args, **_kwargs):
        pytest.fail("an unsupported device reached descriptor lowering")

    monkeypatch.setattr(backend, "lower_modal", forbidden)
    with pytest.raises(BackendError, match="julia executes on device 'cpu'"):
        backend.JuliaBackend(
            case.problem, case.request, options=ExecutionOptions(device=device)
        )


def test_adapted_request_budget_refuses_before_initializing_julia(monkeypatch):
    case, grid, mode = _memory_case("envelope", "modal", "mixed")
    plain = case.request.for_run(stats=StatsSpec(names=(), periodic=()))
    source_plan = plan_julia_nodes(case.problem, plain)
    observed_plan = plan_julia_nodes(case.problem, case.request)
    assert observed_plan.per_node_bytes > source_plan.per_node_bytes
    owner = backend.JuliaBackend(
        case.problem,
        plain,
        grid=grid,
        modes=(mode,),
        options=ExecutionOptions(node_budget=source_plan.per_node_bytes),
    )

    def refuse_runtime(**_):
        pytest.fail("the complete run budget must refuse before Julia allocation")

    monkeypatch.setattr(backend, "get_runtime", refuse_runtime)
    with pytest.raises(BackendError, match="node"):
        owner.prepare(case.problem, case.request)
    owner.close()


_MANAGED = pytest.mark.skipif(
    os.environ.get("SOLPROP_TEST_JULIA") != "1",
    reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
)


@_MANAGED
@pytest.mark.medium
@pytest.mark.parametrize("factory", [cpu_case, adams_case, low_storage_case])
def test_public_cpu_selection_runs_every_integrator(factory):
    case = factory()
    owner = registry.select(
        "julia",
        case.problem,
        case.request,
        grid=window(),
        modes=(fibre().mode,),
        options=ExecutionOptions(
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
        ),
    )
    with owner.prepare(case.problem, case.request).start(
        Start.entrance(case.problem)
    ) as session:
        result = drive(Started(session), export=IDENTITY)
        assert result.status.attempts == 8
        assert np.isfinite(result.fields[-1]).all()
        record = session.execution_record()
        assert record["backend"] == "julia" and record["device"] == "cpu"
        payload, budget = record["node_chunk_array_bytes"], record["node_budget"]
        assert isinstance(payload, int) and isinstance(budget, int)
        assert payload <= budget


@_MANAGED
@pytest.mark.medium
def test_prepared_requests_keep_independent_budgeted_chunks():
    case, grid, mode = _memory_case("envelope", "modal", "mixed")
    plain = case.request.for_run(stats=StatsSpec(names=(), periodic=()))
    plain_case = Translation(case.problem, plain)
    source_plan = plan_julia_nodes(case.problem, plain)
    budget = 3 * source_plan.per_node_bytes
    options = ExecutionOptions(
        node_budget=budget,
        backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
    )
    observed_plan = plan_julia_nodes(case.problem, case.request, options)
    assert observed_plan.plan.chunk < 3
    owner = registry.select(
        "julia",
        case.problem,
        plain,
        grid=grid,
        modes=(mode,),
        options=options,
    )
    assert isinstance(owner, backend.JuliaBackend)
    try:
        first = owner.prepare(plain_case.problem, plain_case.request)
        retained = owner._preparation
        second = owner.prepare(case.problem, case.request)
        assert owner._preparation is retained
        assert first.metadata()["node_chunk"] == 3
        assert second.metadata()["node_chunk"] == observed_plan.plan.chunk
        for prepared in (second, first, second):
            with prepared.start(Start.entrance(case.problem), owns=False) as session:
                assert isinstance(session, JuliaSession)
                record = session.execution_record()
                assert record["node_chunk"] == prepared.metadata()["node_chunk"]
                bound = sum(
                    _integer_record(record, key)
                    for key in (
                        "node_chunk_array_bytes",
                        "node_fixed_workspace_bytes",
                        "node_full_volume_workspace_bytes",
                    )
                )
                assert _workspace_payload(session.memory_ledger()) <= bound
        assert first.metadata()["node_chunk"] == 3
        assert second.metadata()["node_chunk"] == observed_plan.plan.chunk
    finally:
        owner.close()


@_MANAGED
@pytest.mark.medium
def test_public_simulation_and_common_benchmark_window_complete():
    simulation = build(CASES["smoke"].configuration("default", 1, "julia"))
    simulation.settings = replace(
        simulation.settings,
        backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
    )
    output = simulation.run(progress="none")
    assert output.n_saved > 1 and np.isfinite(output.fields()[-1]).all()
    problem, request = simulation.translate(), simulation._run_request()
    owner = registry.select(
        "julia",
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options(),
    )
    try:
        record, field = timed_run(simulation, problem, owner, FieldRetention("all"))
        assert record["timing_boundary"] == TIMING_BOUNDARY
        assert record["wall_seconds"] >= record["seconds"] > 0
        assert record["source_evaluations"] > 0
        assert record["source_evaluations_after"] > record["source_evaluations_before"]
        assert (
            record["execution"]["setup_id"]
            == simulation.settings.backend_options["setup_id"]
        )
        assert field is not None and np.isfinite(field).all()
    finally:
        owner.close()
