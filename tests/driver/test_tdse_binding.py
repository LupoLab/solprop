# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise provider contracts without importing or running the atomic engine."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from solprop.driver import tdse
from solprop.exceptions import (
    BackendError,
    MaterialError,
    MemoryBudgetError,
    ResponseBudgetError,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSEKerrModel,
    TDSESampling,
)
from solprop.translate.tdse import AtomicCallLedger, TDSEDescriptor

_GRID = FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 1600e-9), 10e-15)
_SAMPLES = _GRID.n_to


@dataclass(frozen=True)
class _Build:
    execution_space: str = "OpenMP"
    source_revision: str = "876b72d"


@dataclass(frozen=True)
class _Model:
    name: str = "helium_tong05"
    version: int = 1


@dataclass(frozen=True)
class _Info:
    rows: int
    samples: int
    spacing: float
    shadow_order: int
    outputs: str
    step: float


@dataclass(frozen=True)
class _Outcome:
    failure: str = "none"
    failure_step: int | None = None

    @property
    def ok(self):
        return self.failure == "none"


@dataclass(frozen=True)
class _Monitor:
    initial_norm: np.ndarray
    norm: np.ndarray
    removed_component: np.ndarray | None = None


class _Full(SimpleNamespace):
    pass


class _Compact(SimpleNamespace):
    pass


class _Handle:
    def __init__(self, provider, name, entries=()):
        self.provider = provider
        self.name = name
        self.entries = entries

    def memory_ledger(self):
        return SimpleNamespace(entries=self.entries)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.provider.closed.append(self.name)


class _Plan(_Handle):
    def __init__(self, provider, **values):
        rows = values["rows"]
        super().__init__(
            provider,
            f"plan-{rows}",
            (
                SimpleNamespace(name="operators.shared", bytes=100, shared=True),
                SimpleNamespace(name="state", bytes=rows * 10, shared=False),
            ),
        )
        self.info = _Info(
            rows,
            values["samples"],
            values["spacing"],
            values["shadow_order"],
            values["outputs"],
            values["max_step_au"],
        )
        self.provider.created.append(self)

    def evaluate(self, fields):
        self.provider.calls.append(np.array(fields, copy=True))
        if self.provider.failure == "raise":
            raise RuntimeError("synthetic native failure")
        first = 2 * fields
        third = 3 * fields**3
        # Exact integers make this an algebra/contract oracle, not a numerical
        # approximation to a helium atom. Every call starts from zero history.
        total = first + third + fields**5
        orders = (1,) if self.info.shadow_order == 1 else (1, 3)
        common = {
            "info": self.info,
            "valid": np.ones(fields.shape, dtype=bool),
            "outcomes": tuple(_Outcome() for _ in fields),
            "monitors": _Monitor(np.ones(len(fields)), np.ones(len(fields))),
        }
        if self.info.outputs == "full":
            components = {
                ("total_current", 1): first,
                ("total_current", 3): third,
                ("charge_dipole", 3): 5 * fields**3,
            }
            raw = _Full(
                **common,
                plasma=SimpleNamespace(
                    total_current=total, neglected_current_bound=np.zeros_like(fields)
                ),
                shadow=SimpleNamespace(
                    orders=orders, component=lambda name, order: components[name, order]
                ),
            )
        else:
            raw = _Compact(
                **common,
                total_current=total,
                shadow_orders=orders,
                shadow_total_current=np.stack(
                    [first] if orders == (1,) else [first, third], axis=1
                ),
                neglected_current_bound=np.zeros_like(fields),
            )
        if self.provider.failure == "row":
            raw.valid[0, -1] = False
            raw.outcomes = (_Outcome("norm_growth", 7), *raw.outcomes[1:])
        if self.provider.failure == "labels":
            raw.shadow_orders = (3, 1)
        if self.provider.failure == "nonfinite":
            raw.total_current[0, -1] = np.nan
        if self.provider.failure == "bounds":
            raw.neglected_current_bound = np.zeros((1, fields.shape[1]))
        if self.provider.failure == "monitors":
            raw.monitors = _Monitor(np.ones(1), np.ones(1))
        raw = _malformed_result(raw, self.provider.failure)
        self.provider.responses.append(raw)
        return raw


def _malformed_result(raw, failure):
    if failure == "monitor_nonfinite":
        raw.monitors.norm[0] = np.inf
    elif failure == "info":
        raw.info = replace(raw.info, spacing=raw.info.spacing * 2)
    elif failure == "full_shadow_missing":
        raw.shadow = None
    elif failure == "full_labels":
        raw.shadow.orders = (3, 1)
    elif failure == "compact_shadow_missing":
        raw.shadow_total_current = None
    elif failure == "compact_extents":
        raw.shadow_total_current = raw.shadow_total_current[:, :, :-1]
    elif failure == "current_extents":
        raw.total_current = raw.total_current[:, :-1]
    elif failure == "outcome_count":
        raw.outcomes = raw.outcomes[:-1]
    elif failure == "wrong_type":
        return SimpleNamespace(**vars(raw))
    elif failure == "missing_diagnostics":
        del raw.monitors
    return raw


def _saved_metadata(binding):
    return json.loads(json.dumps(binding.metadata(), allow_nan=False))


class _DrivenError(Exception):
    """Stand-in for the provider's native-failure exception."""


class _Provider:
    DrivenError = _DrivenError
    GridResponse = _Full
    TotalCurrentResponse = _Compact

    def __init__(self):
        self.created = []
        self.calls = []
        self.responses = []
        self.closed = []
        self.build = _Build()
        self.failure = None
        self.assembled = 0
        self.model_version = 1
        self.preparation_interrupt = False

    def build_info(self):
        return self.build

    def atomic_model(self, name):
        return _Model(name, self.model_version)

    def unit_scale(self, quantity):
        return SimpleNamespace(si_per_atomic_unit=1.0)

    def radial_basis(self, edges, orders):
        return SimpleNamespace(edges=edges, orders=orders)

    def angular_basis(self, lmax, m):
        return SimpleNamespace(lmax=lmax, m=m)

    def complex_scaling(self, radius, angle, infinite_order, decay):
        return SimpleNamespace(
            radius=radius, angle=angle, infinite_order=infinite_order, decay=decay
        )

    def discretization(self, radial, angular, model, scaling, cuts):
        def assemble():
            self.assembled += 1
            return _Handle(
                self, "operators", (SimpleNamespace(name="shared", bytes=100),)
            )

        return SimpleNamespace(
            radial=radial,
            angular=angular,
            model=model,
            scaling=scaling,
            cuts=cuts,
            assemble=assemble,
        )

    def state_selection(self, angular, index):
        return angular, index

    def spectral_options(self, **values):
        return SimpleNamespace(**values)

    def spectral_setup(self, operators, options):
        if self.preparation_interrupt:
            raise KeyboardInterrupt
        return _Handle(self, "setup", (SimpleNamespace(name="spectral", bytes=200),))

    def response_plan(self, setup, **values):
        return _Plan(self, **values)

    def solve_preset(self, target, preset):
        basis = _basis()
        return SimpleNamespace(
            name=preset,
            cut_energy=basis.cut_energy,
            source="synthetic preset source",
            discretization=self.Discretization(
                self.RadialBasis(basis.radial_edges, basis.radial_orders),
                self.AngularBasis(basis.lmax, basis.m),
                self.atomic_model(basis.potential),
                None,
                (),
            ),
        )

    RadialBasis = radial_basis
    AngularBasis = angular_basis
    ComplexScaling = complex_scaling
    Discretization = discretization
    StateSelection = state_selection
    SpectralOptions = spectral_options
    SpectralSetup = spectral_setup
    ResponsePlan = response_plan


def _basis():
    return TDSEBasis("contract-test", (0.0, 1e-9), (4,), 3, 1e-17, source="synthetic")


def _descriptor(model: TDSEKerrModel = "optmat_cubic", basis: TDSEBasis | None = None):
    basis = _basis() if basis is None else basis
    response = TDSEGasResponse(
        "helium",
        basis,
        _GRID.dt_fine / 2,
        model,
        1.0 if model == "tdse_native" else 2.0,
        TDSECalibration(
            "test",
            "research",
            basis.identity,
            (120e-9, 1600e-9),
            source="synthetic",
            limitations=("test only",),
        ),
        TDSESampling(),
        cubic_scale=1.5 if model == "tdse_shadow" else None,
    )
    return TDSEDescriptor.from_response(response, _GRID)


@pytest.fixture
def provider(monkeypatch):
    fake = _Provider()
    monkeypatch.setattr(tdse, "_load_provider", lambda: cast("tdse._Provider", fake))
    return fake


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow", "tdse_native"])
def test_binding_maps_required_results_without_scaling(provider, model):
    binding = tdse.prepare_binding(
        _descriptor(model), TDSEExecution(max_batch_rows=2), 5
    )
    fields = np.arange(5 * _SAMPLES, dtype=np.float64).reshape(5, _SAMPLES) / 16
    binding.reserve(1)
    result = binding.evaluate(fields, z=0.2, purpose="predictor")
    assert_array_equal(result.total_current, 2 * fields + 3 * fields**3 + fields**5)
    assert_array_equal(result.first_order_current, 2 * fields)
    if model == "tdse_native":
        assert result.third_order_current is None
    else:
        assert_array_equal(result.third_order_current, 3 * fields**3)
    if model == "tdse_shadow":
        assert_array_equal(result.third_order_dipole, 5 * fields**3)
    else:
        assert result.third_order_dipole is None
    assert result.total_current.flags.owndata
    assert not result.total_current.flags.writeable
    assert [plan.info.rows for plan in provider.created] == [1, 2]
    assert [call.shape[0] for call in provider.calls] == [2, 2, 1]
    assert binding.ledger.attempted == 3
    assert [item["row"] for item in json.loads(result.outcomes_json)] == list(range(5))
    binding.close()
    assert provider.closed == ["plan-2", "plan-1", "setup", "operators"]
    binding.close()
    with pytest.raises(BackendError, match="closed"):
        binding.reserve(1)


def test_binding_diagnostic_reuse_requires_exact_fields(provider):
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(), 1)
    fields = np.ones((1, _SAMPLES))
    binding.reserve(2)
    first = binding.evaluate(fields, z=0.0, purpose="predictor")
    # Diagnostic reuse confirms equality with the accepted field supplied by the
    # caller. Repeated predictor calls still execute and reset every atomic row.
    cached = binding.evaluate(fields.copy(), z=0.0, purpose="diagnostic")
    assert cached is first
    repeated = binding.evaluate(fields, z=0.0, purpose="predictor")
    assert_array_equal(first.total_current, repeated.total_current)
    assert len(provider.calls) == 2
    binding.reserve(1)
    changed = fields.copy()
    changed[0, 1] = np.nextafter(1.0, 2.0)
    binding.evaluate(changed, z=0.0, purpose="diagnostic")
    assert binding.ledger.attempted == 3
    assert binding.ledger.diagnostic_calls == 1
    assert binding.metadata()["diagnostic_cache_hits"] == 1
    binding.close()


def test_binding_budget_is_reserved_before_any_chunk(provider):
    binding = tdse.prepare_binding(
        _descriptor(), TDSEExecution(max_batch_rows=2, max_atomic_calls=5), 5
    )
    with pytest.raises(ResponseBudgetError):
        binding.reserve(2)
    assert provider.calls == []
    binding.reserve(1)
    binding.evaluate(np.ones((5, _SAMPLES)), z=0.2, purpose="predictor")
    assert binding.ledger.attempted == 3
    with pytest.raises(ResponseBudgetError):
        binding.reserve(1)
    assert len(provider.calls) == 3
    binding.close()


@pytest.mark.parametrize(
    "failure", ["raise", "row", "labels", "nonfinite", "bounds", "monitors"]
)
def test_binding_failed_calls_stay_charged_and_preserve_provenance(provider, failure):
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(max_batch_rows=2), 3)
    provider.failure = failure
    binding.reserve(2)
    with pytest.raises(BackendError, match=r"z=0.25 m, rows \[0, 2\), input="):
        binding.evaluate(np.ones((3, _SAMPLES)), z=0.25, purpose="corrected")
    assert binding.ledger.attempted == 1
    assert binding.ledger.failed_calls == 1
    assert binding.ledger.reserved == 0
    metadata = binding.metadata()
    assert len(cast("list[object]", metadata["failures"])) == 1
    assert json.loads(json.dumps(metadata, allow_nan=False))["failures"][0]["error"]
    binding.close()


def test_binding_admits_memory_before_native_setup(provider):
    with pytest.raises(MemoryBudgetError):
        tdse.prepare_binding(_descriptor(), TDSEExecution(memory_limit=1), 2)
    assert provider.assembled == 0
    assert provider.calls == []


def test_binding_memory_budget_includes_optical_host_without_duplicate_ledger(provider):
    descriptor = _descriptor()
    estimate = sum(
        tdse.estimate_binding_memory(descriptor, TDSEExecution(), 2).values()
    )
    with pytest.raises(MemoryBudgetError, match="combined"):
        tdse.prepare_binding(
            descriptor,
            TDSEExecution(memory_limit=estimate + 99),
            2,
            host_memory_bytes=100,
        )
    assert provider.assembled == 0
    binding = tdse.prepare_binding(
        descriptor, TDSEExecution(), 2, host_memory_bytes=100
    )
    assert "optical.host" not in binding.memory_ledger()
    assert binding.metadata()["optical_host_memory_bytes"] == 100
    binding.close()


def test_binding_memory_estimate_does_not_import_native_engine(monkeypatch):
    def unexpected_import():
        pytest.fail("memory estimate imported the optional provider")

    monkeypatch.setattr(tdse, "_load_provider", unexpected_import)
    assert (
        sum(tdse.estimate_binding_memory(_descriptor(), TDSEExecution(), 2).values())
        > 0
    )


@pytest.mark.parametrize(
    "device,space", [("cpu", "OpenMP"), ("cuda", "Cuda"), ("hip", "HIP")]
)
def test_binding_requires_matching_device_and_revision(provider, device, space):
    provider.build = _Build(space)
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(device=device), 1)
    assert "plan.1.state" in binding.memory_ledger()
    assert sum(binding.memory_ledger().values()) >= 310
    binding.close()
    provider.build = _Build("other")
    with pytest.raises(BackendError, match="native build"):
        tdse.prepare_binding(_descriptor(), TDSEExecution(device=device), 1)
    provider.build = _Build(space, "old0000")
    with pytest.raises(BackendError, match="pinned"):
        tdse.prepare_binding(_descriptor(), TDSEExecution(device=device), 1)


def test_binding_input_and_resume_contract(provider):
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(max_atomic_calls=10), 1)
    binding.restore_ledger(AtomicCallLedger(5, 3, 0, 2, 1, 1))
    assert binding.ledger.attempted == 3
    with pytest.raises(BackendError, match="sample count"):
        binding.evaluate(np.ones((1, _SAMPLES - 1)), z=0.0, purpose="startup")
    assert provider.calls == []
    binding.reserve(1)
    binding.evaluate(np.ones((1, _SAMPLES)), z=0.0, purpose="startup")
    assert binding.ledger.attempted == 4
    assert binding.ledger.failed_calls == 1
    binding.close()


def test_basis_preset_is_resolved_without_native_assembly(provider):
    basis = tdse.basis_from_preset("helium", "production")
    assert basis.radial_edges == _basis().radial_edges
    assert basis.potential == "helium_tong05"
    assert "synthetic preset source" in basis.source
    assert provider.assembled == 0
    with pytest.raises(MaterialError, match="helium"):
        tdse.basis_from_preset("argon", "production")


def test_binding_missing_optional_provider_is_actionable(monkeypatch):
    def missing(name):
        raise ImportError("synthetic missing dependency")

    monkeypatch.setattr(tdse.importlib, "import_module", missing)
    with pytest.raises(BackendError, match="optional pyroraptor"):
        tdse.basis_from_preset("helium", "production")


def test_binding_interrupted_preparation_closes_created_resources(provider):
    provider.preparation_interrupt = True
    with pytest.raises(KeyboardInterrupt):
        tdse.prepare_binding(_descriptor(), TDSEExecution(), 1)
    assert provider.closed == ["operators"]


def test_binding_module_import_does_not_load_optional_engine():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import solprop.driver.tdse; "
            "assert 'pyroraptor' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_binding_rejects_unsupported_interpreter_before_native_import(monkeypatch):
    monkeypatch.setattr(tdse.sys, "version_info", (3, 15, 0))
    with pytest.raises(BackendError, match=r"Python 3\.14"):
        tdse.basis_from_preset("helium", "production")


def test_binding_without_distribution_metadata_still_records_native_identity(
    monkeypatch,
):
    provider = _Provider()
    monkeypatch.setattr(tdse.importlib, "import_module", lambda name: provider)

    def missing_distribution(name):
        raise tdse.PackageNotFoundError(name)

    monkeypatch.setattr(tdse, "distribution", missing_distribution)
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(), 1)
    metadata = _saved_metadata(binding)
    assert metadata["build"]["distribution"] is None
    assert metadata["build"]["source_pin"].startswith(provider.build.source_revision)
    assert provider.assembled == 1
    assert provider.calls == []
    binding.close()


@pytest.mark.parametrize("direct_url", [None, '{"url": "file:///provider"}'])
def test_binding_records_an_installed_distribution(monkeypatch, direct_url):
    """The optional extra, when installed, adds its version and source to the record."""
    provider = _Provider()
    monkeypatch.setattr(tdse.importlib, "import_module", lambda name: provider)

    class _Distribution:
        version = "1.2.3"

        def read_text(self, name):
            assert name == "direct_url.json"
            return direct_url

    monkeypatch.setattr(tdse, "distribution", lambda name: _Distribution())
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(), 1)
    build = _saved_metadata(binding)["build"]
    assert build["distribution"] == "1.2.3"
    assert build["direct_url"] == (
        None if direct_url is None else {"url": "file:///provider"}
    )
    binding.close()


def test_native_metadata_encoding_preserves_scalar_types_and_refuses_opaque_values():
    values = {
        "step": np.int64(7),
        "bound": np.float64(0.25),
        "samples": np.array([1, 2]),
    }
    assert json.loads(tdse._encoded(values)) == {
        "step": 7,
        "bound": 0.25,
        "samples": [1, 2],
    }
    with pytest.raises(TypeError, match="Unsupported TDSE metadata value"):
        tdse._encoded({"opaque_handle": object()})


@pytest.mark.parametrize("n_nodes", [0, -1, True])
def test_binding_and_estimate_reject_invalid_row_counts_before_provider_work(
    provider, n_nodes
):
    with pytest.raises(BackendError, match="positive integer n_nodes"):
        tdse.prepare_binding(_descriptor(), TDSEExecution(), n_nodes)
    with pytest.raises(BackendError, match="positive integer n_nodes"):
        tdse.estimate_binding_memory(_descriptor(), TDSEExecution(), n_nodes)
    assert provider.assembled == 0
    assert provider.created == []


@pytest.mark.parametrize("host_bytes", [-1, True])
def test_binding_refuses_invalid_optical_memory_before_provider_work(
    provider, host_bytes
):
    with pytest.raises(BackendError, match="host_memory_bytes"):
        tdse.prepare_binding(
            _descriptor(), TDSEExecution(), 1, host_memory_bytes=host_bytes
        )
    assert provider.assembled == 0


def test_binding_rejects_changed_atomic_model_before_allocation(provider):
    provider.model_version = 2
    with pytest.raises(BackendError, match="potential revision"):
        tdse.prepare_binding(_descriptor(), TDSEExecution(), 1)
    assert provider.assembled == 0
    assert provider.created == []


def test_binding_forwards_explicit_exterior_scaling_without_changing_units(
    provider, monkeypatch
):
    basis = replace(
        _basis(), scaling_radius=1e-9, scaling_angle=0.15, infinite_order=4, decay=2e9
    )
    captured = []
    factory = provider.ComplexScaling

    def record_scaling(*args):
        captured.append(args)
        return factory(*args)

    monkeypatch.setattr(provider, "ComplexScaling", record_scaling)
    binding = tdse.prepare_binding(_descriptor(basis=basis), TDSEExecution(), 1)
    assert captured == [(1e-9, 0.15, 4, 2e9)]
    assert (
        _saved_metadata(binding)["configuration"]["response"]["basis"]["scaling_radius"]
        == 1e-9
    )
    binding.close()


def test_later_plan_preparation_failure_closes_previously_created_handles(
    provider, monkeypatch
):
    factory = provider.ResponsePlan

    def create_plan(setup, **values):
        if values["rows"] == 2:
            raise RuntimeError("second plan allocation failed")
        return factory(setup, **values)

    monkeypatch.setattr(provider, "ResponsePlan", create_plan)
    with pytest.raises(
        BackendError, match="preparation failed: second plan allocation"
    ):
        tdse.prepare_binding(_descriptor(), TDSEExecution(max_batch_rows=2), 3)
    assert provider.closed == ["plan-1", "setup", "operators"]
    assert provider.calls == []


def test_measured_memory_refusal_closes_all_resources_after_estimated_admission(
    provider, monkeypatch
):
    descriptor = _descriptor()
    execution = TDSEExecution(max_batch_rows=2)
    estimate = sum(tdse.estimate_binding_memory(descriptor, execution, 3).values())
    factory = provider.SpectralSetup

    def oversized_setup(*args):
        setup = factory(*args)
        setup.entries = (
            SimpleNamespace(name="unexpected_workspace", bytes=estimate + 1),
        )
        return setup

    monkeypatch.setattr(provider, "SpectralSetup", oversized_setup)
    with pytest.raises(MemoryBudgetError, match="combined persistent-data"):
        tdse.prepare_binding(descriptor, replace(execution, memory_limit=estimate), 3)
    assert provider.assembled == 1
    assert len(provider.created) == 2
    assert provider.closed == ["plan-2", "plan-1", "setup", "operators"]
    assert provider.calls == []


@pytest.mark.parametrize("z", [np.nan, np.inf])
def test_binding_invalid_position_does_not_spend_reserved_calls(provider, z):
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(), 1)
    binding.reserve(1)
    with pytest.raises(BackendError, match="finite z"):
        binding.evaluate(np.ones((1, _SAMPLES)), z=z, purpose="predictor")
    assert binding.ledger.attempted == 0
    assert binding.ledger.reserved == 1
    binding.release_reservation()
    assert binding.ledger.reserved == 0
    assert provider.calls == []
    binding.close()


@pytest.mark.parametrize(
    "model,failure,message",
    [
        ("optmat_cubic", "info", "metadata disagree"),
        ("tdse_shadow", "full_shadow_missing", "full response with shadow"),
        ("tdse_shadow", "full_labels", "incorrect shadow-order labels"),
        ("optmat_cubic", "compact_shadow_missing", "incorrect shadow-order labels"),
        ("optmat_cubic", "compact_extents", "incorrect shadow array extents"),
        ("optmat_cubic", "current_extents", "incorrect current extents"),
        ("optmat_cubic", "wrong_type", "different result type"),
        ("optmat_cubic", "outcome_count", "failed atomic row"),
        ("optmat_cubic", "monitor_nonfinite", "row monitor"),
        ("optmat_cubic", "missing_diagnostics", "monitors"),
    ],
)
def test_malformed_native_results_preserve_diagnostics_and_charge_attempt(
    provider, model, failure, message
):
    binding = tdse.prepare_binding(_descriptor(model), TDSEExecution(), 2)
    provider.failure = failure
    binding.reserve(1)
    with pytest.raises(BackendError, match=message):
        binding.evaluate(np.ones((2, _SAMPLES)), z=0.5, purpose="corrected")
    assert binding.ledger.attempted == binding.ledger.failed_calls == 1
    assert binding.ledger.reserved == 0
    failed = _saved_metadata(binding)["failures"][0]
    assert failed["row_start"] == 0 and failed["row_stop"] == 2
    assert failed["input_identity"]
    if failure == "monitor_nonfinite":
        assert failed["native"]["monitors"]["norm"][0] == "nonfinite:Infinity"
    elif failure == "missing_diagnostics":
        assert failed["native"]["unreadable_native_record"] == "_Compact"
    else:
        assert failed["native"]["valid"] == [[True] * _SAMPLES] * 2
    binding.close()


def test_second_chunk_failure_preserves_first_chunk_and_allows_fresh_retry(
    provider, monkeypatch
):
    binding = tdse.prepare_binding(_descriptor(), TDSEExecution(max_batch_rows=2), 3)
    original = _Plan.evaluate

    def second_chunk_fails(plan, fields):
        provider.failure = "row" if len(provider.calls) == 1 else None
        return original(plan, fields)

    monkeypatch.setattr(_Plan, "evaluate", second_chunk_fails)
    binding.reserve(2)
    with pytest.raises(BackendError, match=r"rows \[2, 3\)"):
        binding.evaluate(np.ones((3, _SAMPLES)), z=0.5, purpose="corrected")
    assert binding.ledger.attempted == 2
    assert binding.ledger.failed_calls == 1
    assert binding.ledger.reserved == 0
    metadata = _saved_metadata(binding)
    assert metadata["last_call"]["chunks"][0]["row_stop"] == 2
    assert metadata["failures"][0]["row_start"] == 2
    binding.reserve(1)
    result = binding.evaluate(np.ones((3, _SAMPLES)), z=0.5, purpose="corrected")
    assert_array_equal(result.total_current, np.full((3, _SAMPLES), 6.0))
    assert binding.ledger.attempted == 4
    assert len(provider.created) == 2
    binding.close()
