# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bind haCC helium through a fake provider (SEP 20 section 7.1 items 6 to 10, 19).

The fake (``tests/hacc_cases.py``) implements the provider contract of SEP 20
section 5.7 with a manufactured current whose order-one part is exact, so every
array comparison here is a copy or a bitwise-defined formula.
"""

import dataclasses
import inspect
import json
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests import hacc_cases
from tests.hacc_cases import (
    FakeArgon,
    FakeBuild,
    FakeHelium,
    FakeVelocityGauge,
    fake_objects,
    hacc_response,
    install,
    manufactured_currents,
    preset_basis,
    tiny_basis,
)

from solprop.constants import EPS0, C
from solprop.driver import tdse
from solprop.exceptions import (
    BackendError,
    MaterialError,
    MemoryBudgetError,
    TDSEDomainWarning,
    TDSEUnqualifiedModelWarning,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import TDSEAtomicWindow, TDSEExecution
from solprop.translate.tdse import TDSEDescriptor

_GRID = FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 1600e-9), 10e-15)
_N = _GRID.n_to


def _descriptor(basis=None, **changes):
    return TDSEDescriptor.from_response(hacc_response(basis, **changes), _GRID)


def _fields(rows, scale=1.0):
    row = np.arange(rows, dtype=np.float64)[:, None]
    sample = np.arange(_N, dtype=np.float64)[None, :]
    return scale * (((row + sample) % 7) - 3) / 4


def _meta(binding):
    """Return the binding's metadata as plain JSON containers."""
    return json.loads(json.dumps(binding.metadata(), allow_nan=False))


def _bind(descriptor, execution=None, n_nodes=3, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        return tdse.prepare_binding(
            descriptor,
            TDSEExecution(max_batch_rows=2) if execution is None else execution,
            n_nodes,
            **kwargs,
        )


def _expected(descriptor, fields, law="pointwise"):
    """Return the padded (total, first) the binding must return for these fields."""
    window = descriptor.atomic_window
    gate = descriptor.source_gate()
    if window is None:
        first, total = manufactured_currents(fields, descriptor.spacing, law)
        return total, first
    span = slice(window.first, window.last + 1)
    first, total = manufactured_currents(
        fields[:, span] * gate[span], descriptor.spacing, law
    )
    padded = [np.zeros_like(fields) for _ in range(2)]
    padded[0][:, span], padded[1][:, span] = total, first
    return padded[0], padded[1]


@pytest.fixture
def fake(monkeypatch):
    return install(monkeypatch)


# --- item 7: preparation ---------------------------------------------------------


def test_preset_preparation_passes_the_exact_provider_arguments(fake):
    descriptor = _descriptor(preset_basis(2))
    binding = _bind(descriptor, n_nodes=5)
    window = descriptor.atomic_window
    assert window is not None
    assert fake.preset_requests == [2]
    assert [arguments["rows"] for arguments in fake.plan_arguments] == [1, 2]
    for arguments in fake.plan_arguments:
        assert arguments == {
            "rows": arguments["rows"],
            "spacing": descriptor.spacing,
            "samples": window.samples,
            "max_step_au": 0.0088,
            "field": "electric_field",
            "shadow_order": 1,
            "outputs": "total_current",
            "norm_growth_limit": 2.0,
        }
        assert type(arguments["spacing"]) is float
        assert type(arguments["max_step_au"]) is float
        assert type(arguments["samples"]) is int
    # The model is entered first and closed after every plan.
    assert fake.events[0] == "enter-model"
    binding.close()
    assert fake.events[-1] == "close-model"
    assert all(event.startswith("close-plan") for event in fake.events[-3:-1])
    metadata = _meta(binding)
    assert metadata["model"] == "hacc"
    assert metadata["hacc"]["preset"] == preset_basis(2).preset
    assert metadata["hacc"]["preset_record"]["variant"] == "standard"
    assert metadata["hacc"]["preset_record"]["step_margin"] == 0.62
    assert metadata["hacc"]["qualified"] is True
    assert metadata["hacc"]["closure"] is True
    assert metadata["hacc"]["complement"] == "not_applicable"
    assert metadata["atomic_window"] == window.to_json()
    assert metadata["plans"]["2"]["total_current_qualified"] is True


@pytest.mark.parametrize("ions", [1, 2])
def test_fast_preset_is_verified_against_the_fast_variant(fake, ions):
    basis = preset_basis(ions, "fast")
    binding = _bind(_descriptor(basis))
    # The binding asks the provider for the record's own variant, and the plans
    # take the fast step ceiling.
    assert fake.preset_requests == [ions] and fake.variant_requests == ["fast"]
    assert {arguments["max_step_au"] for arguments in fake.plan_arguments} == {0.02045}
    metadata = _meta(binding)
    assert (
        metadata["hacc"]["label"]
        == basis.label
        == f"haCC({1 if ions == 1 else 5}) fast"
    )
    assert metadata["hacc"]["preset"] == basis.preset
    record = metadata["hacc"]["preset_record"]
    assert (record["variant"], record["step_margin"]) == ("fast", 0.9)
    binding.close()


def test_fast_record_is_refused_by_a_provider_serving_the_standard_identity(fake):
    # A provider whose fast request returns another identity (an older build, or a
    # changed definition) is refused before any plan, as for a standard record.
    fake.preset_changes = {"identity": preset_basis(1).preset}
    with pytest.raises(BackendError, match="expected"):
        _bind(_descriptor(preset_basis(1, "fast")))
    assert fake.plan_arguments == []


def test_custom_preparation_calls_prepare_and_warns_once(fake):
    basis = tiny_basis()
    with pytest.warns(TDSEUnqualifiedModelWarning, match="custom haCC") as record:
        binding = tdse.prepare_binding(
            _descriptor(basis), TDSEExecution(max_batch_rows=2), 3
        )
    assert len(record) == 1
    assert "closure" not in str(record[0].message)
    atom, native, gauge, cut = fake.prepared[0]
    assert (atom, native, gauge) == fake_objects(basis)
    assert cut == basis.cut_energy
    assert fake.preset_requests == []
    metadata = _meta(binding)
    hacc = metadata["hacc"]
    assert isinstance(hacc, dict)
    assert (hacc["custom"], hacc["qualified"], hacc["closure"]) == (True, False, True)
    assert metadata["domain"] == {"reasons": {}, "flagged_calls": []}
    binding.close()


def test_shared_model_memory_is_counted_once_across_plan_sizes(fake):
    binding = _bind(_descriptor(), n_nodes=3)
    ledger = binding.memory_ledger()
    assert ledger["model.record"] == fake.model_bytes
    assert ledger["plan.1.march.state"] == fake.plan_bytes
    assert ledger["plan.2.march.state"] == 2 * fake.plan_bytes
    assert not any(
        name.endswith("model.record") and name != "model.record" for name in ledger
    )
    binding.close()


@pytest.mark.parametrize("which", ["model", "plan"])
def test_measured_memory_above_the_limit_is_refused_before_any_call(fake, which):
    descriptor = _descriptor()
    estimate = sum(
        tdse.estimate_binding_memory(descriptor, TDSEExecution(), 3).values()
    )
    if which == "model":
        fake.model_bytes = 10**9
    else:
        fake.plan_bytes = 10**9
    with pytest.raises(MemoryBudgetError, match="combined persistent-data"):
        _bind(descriptor, TDSEExecution(memory_limit=estimate + 10**6), 3)
    assert fake.calls == []
    assert fake.events[-1] == "close-model"
    assert (fake.plan_arguments == []) is (which == "model")


@pytest.mark.parametrize("control", ["mixed", "velocity"])
def test_custom_basis_without_the_closure_is_a_recorded_research_control(fake, control):
    if control == "mixed":
        fake.closure_admitted = False
        basis = tiny_basis()
    else:
        basis = tiny_basis(gauge_radius=None)
    descriptor = _descriptor(basis)
    with pytest.warns(TDSEUnqualifiedModelWarning, match="escaped-electron") as record:
        binding = tdse.prepare_binding(descriptor, TDSEExecution(max_batch_rows=2), 3)
    assert len(record) == 1
    outputs = [arguments["outputs"] for arguments in fake.plan_arguments]
    # A mixed gauge is first asked for the compact closed current, which the
    # provider refuses without the closure; the velocity gauge goes straight to full.
    assert outputs == (
        ["total_current", "full", "full"] if control == "mixed" else ["full", "full"]
    )
    fields = _fields(3)
    binding.reserve(1)
    result = binding.evaluate(fields, z=0.0, purpose="predictor")
    total, first = _expected(descriptor, fields)
    assert_array_equal(result.total_current, total)
    # j1 is the full record's in-box order one; the fake has no total orders.
    assert_array_equal(result.first_order_current, first)
    metadata = _meta(binding)
    hacc = metadata["hacc"]
    assert isinstance(hacc, dict)
    assert hacc["closure"] is False and hacc["native_outputs"] == "full"
    chunk = json.loads(result.monitors_json)["chunks"][0]
    assert chunk["neglected_current_bound"] is None
    assert chunk["neglected_current_bound_max"] is None
    binding.close()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"identity": "hacc-helium-1-v2-0123456789abcdef"}, "expected"),
        ({"cut_energy": 1.0e-16}, "reproduce"),
        ({"atom": FakeHelium(1, 5)}, "reproduce"),
    ],
)
def test_preset_the_provider_does_not_reproduce_is_refused(fake, change, message):
    fake.preset_changes = change
    with pytest.raises(BackendError, match=message):
        _bind(_descriptor(preset_basis(1)))
    assert fake.plan_arguments == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [("plan_info", "metadata disagree"), ("plan_unqualified", "contract violation")],
)
def test_preset_plan_info_must_describe_the_model_and_its_closure(
    fake, failure, message
):
    fake.failure = failure
    with pytest.raises(BackendError, match=message):
        _bind(_descriptor(preset_basis(1)))
    assert fake.events[-1] == "close-model"


def test_custom_plan_whose_closure_flag_contradicts_its_mode_is_refused(fake):
    fake.failure = "plan_unqualified"
    with pytest.raises(BackendError, match="closure flag"):
        _bind(_descriptor())


def test_hacc_refuses_vendor_blas_device_and_missing_surface(fake):
    with pytest.raises(BackendError, match="no complement option"):
        _bind(_descriptor(), TDSEExecution(complement="vendor_blas"))
    assert fake.prepared == []
    fake.build = FakeBuild("Cuda")
    fake.device_admissible = False
    with pytest.raises(BackendError, match="device route"):
        _bind(_descriptor(preset_basis(1)), TDSEExecution(device="cuda"))
    fake.device_admissible = True
    binding = _bind(_descriptor(preset_basis(1)), TDSEExecution(device="cuda"))
    binding.close()
    fake.hacc = object()
    with pytest.raises(BackendError, match=r"no haCC consumer surface.*876b72d1"):
        _bind(_descriptor(), TDSEExecution(device="cuda"))


@pytest.mark.parametrize(
    ("build", "device", "message"),
    [
        (FakeBuild("OpenMP", "c2833fa"), "cpu", "pinned"),
        (FakeBuild("OpenMP"), "cuda", "native build"),
    ],
)
def test_stale_revision_and_device_mismatch_are_refused(fake, build, device, message):
    fake.build = build
    with pytest.raises(BackendError, match=message):
        _bind(_descriptor(), TDSEExecution(device=device))
    assert fake.prepared == []


def test_plan_construction_driven_error_is_a_preparation_failure(fake):
    fake.failure, fake.fail_rows = "plan_driven", 2
    with pytest.raises(BackendError, match=r"preparation failed.*\[native\]"):
        _bind(_descriptor(), n_nodes=3)
    # The first plan and then the model are closed.
    assert fake.events[-2:] == [
        f"close-plan-1-{fake.plan_arguments[0]['samples']}",
        "close-model",
    ]
    assert fake.calls == []


# --- item 6: the binding side of the atomic window ------------------------------------


def test_binding_drives_the_window_samples_times_the_gate_and_pads_zeros(fake):
    descriptor = _descriptor(preset_basis(1))
    window = descriptor.atomic_window
    gate = descriptor.source_gate()
    assert window is not None and gate is not None
    binding = _bind(descriptor, n_nodes=3)
    fields = _fields(3)
    binding.reserve(1)
    result = binding.evaluate(fields, z=0.1, purpose="startup")
    span = slice(window.first, window.last + 1)
    passed = np.concatenate(fake.calls, axis=0)
    assert_array_equal(passed, fields[:, span] * gate[span])
    total, first = _expected(descriptor, fields)
    assert_array_equal(result.total_current, total)
    assert_array_equal(result.first_order_current, first)
    assert_array_equal(result.total_current[:, : window.first], 0.0)
    assert_array_equal(result.total_current[:, window.last + 1 :], 0.0)
    assert result.valid.all() and result.total_current.shape == fields.shape
    # Provider time zero is t_s, recorded with the window.
    metadata = _meta(binding)
    recorded = metadata["atomic_window"]
    assert isinstance(recorded, dict)
    assert recorded["t_s"] == window.t_s == float(_GRID.t_fine[window.first])
    drive = metadata["last_call"]["drive_diagnostics"]
    assert drive["drive_final_vector_potential_v_s_per_m"] == [
        -value for value in drive["drive_area_v_s_per_m"]
    ]
    binding.close()


def test_explicit_full_lattice_hacc_drives_every_sample_without_a_warning(fake):
    descriptor = _descriptor(preset_basis(1), atomic_window=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binding = tdse.prepare_binding(descriptor, TDSEExecution(), 1)
        binding.reserve(1)
        binding.evaluate(_fields(1), z=0.0, purpose="startup")
    assert fake.calls[0].shape == (1, _N)
    assert_array_equal(fake.calls[0], _fields(1))
    binding.close()


def test_causal_fake_windowed_source_equals_full_lattice_run_on_tapered_field(fake):
    fake.law = "causal"
    windowed = _descriptor(preset_basis(1))
    gate = windowed.source_gate()
    assert gate is not None
    full = _descriptor(preset_basis(1), atomic_window=None)
    fields = _fields(2)
    sources = []
    for descriptor, drive in ((windowed, fields), (full, fields * gate)):
        binding = _bind(descriptor, n_nodes=2)
        binding.reserve(1)
        result = binding.evaluate(drive, z=0.0, purpose="startup")
        sources.append((result.total_current - result.first_order_current) * gate)
        binding.close()
    # Stopping at t_e adds nothing after the gate closes, and the earlier samples
    # of the full run sum only zeros: the two gated sources agree bitwise.
    assert_array_equal(sources[0], sources[1])
    assert np.any(sources[0])


# --- item 8: evaluation ----------------------------------------------------------


def test_valid_call_returns_owning_read_only_padded_arrays(fake):
    fake.bound_scale = 10.0
    descriptor = _descriptor(preset_basis(1))
    binding = _bind(descriptor, n_nodes=3)
    fields = _fields(3)
    fields[2] = 0.0
    binding.reserve(1)
    result = binding.evaluate(fields, z=0.0, purpose="predictor")
    total, first = _expected(descriptor, fields)
    assert_array_equal(result.total_current, total)
    assert_array_equal(result.first_order_current, first)
    assert result.total_current.flags.owndata
    assert not result.total_current.flags.writeable
    assert [item["row"] for item in json.loads(result.outcomes_json)] == [0, 1, 2]
    chunks = json.loads(result.monitors_json)["chunks"]
    assert chunks[0]["monitors"]["non_finite"] == [False, False]
    # A bound ten times the peak current is admitted and recorded, never refused.
    rows = [
        *chunks[0]["neglected_current_bound"],
        *chunks[1]["neglected_current_bound"],
    ]
    assert rows[0] == {"relative_to_peak": 10.0, "zero_reference": False}
    assert rows[2] == {"relative_to_peak": None, "zero_reference": True}
    assert binding.ledger.attempted == 2 and binding.ledger.failed_calls == 0
    # The exact-input diagnostic cache works as for SAE.
    assert binding.evaluate(fields.copy(), z=0.0, purpose="diagnostic") is result
    assert _meta(binding)["diagnostic_cache_hits"] == 1
    binding.close()


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_type",
        "info",
        "info_atom",
        "labels",
        "bound_none",
        "bound_negative",
        "bound_nonfinite",
        "bound_shape",
        "extents",
        "current_extents",
        "row",
        "valid",
        "nonfinite",
        "monitor_dtype_initial_norm",
        "monitor_dtype_norm",
        "monitor_dtype_peak_norm",
        "monitor_dtype_monitored_step",
        "monitor_dtype_non_finite",
        "monitor_shape",
        "monitor_nonfinite",
    ],
)
def test_malformed_hacc_results_are_charged_contract_failures(fake, failure):
    binding = _bind(_descriptor(preset_basis(1)), n_nodes=2)
    fake.failure = failure
    binding.reserve(1)
    with pytest.raises(BackendError, match=r"rows \[0, 2\)"):
        binding.evaluate(_fields(2), z=0.5, purpose="corrected")
    assert binding.ledger.attempted == binding.ledger.failed_calls == 1
    assert binding.ledger.reserved == 0
    failed = _meta(binding)["failures"]
    assert failed[0]["classification"] == "contract"
    binding.close()


@pytest.mark.parametrize("failure", ["full_shadow_missing", "full_labels"])
def test_malformed_full_records_of_a_control_are_refused(fake, failure):
    binding = _bind(_descriptor(tiny_basis(gauge_radius=None)), n_nodes=1)
    fake.failure = failure
    binding.reserve(1)
    with pytest.raises(BackendError, match="shadow"):
        binding.evaluate(_fields(1), z=0.0, purpose="corrected")
    assert binding.ledger.failed_calls == 1
    binding.close()


def test_chunks_recombine_in_row_order(fake):
    descriptor = _descriptor(preset_basis(1))
    binding = _bind(descriptor, TDSEExecution(max_batch_rows=2), n_nodes=5)
    fields = _fields(5)
    binding.reserve(1)
    result = binding.evaluate(fields, z=0.0, purpose="predictor")
    assert [call.shape[0] for call in fake.calls] == [2, 2, 1]
    total, _ = _expected(descriptor, fields)
    assert_array_equal(result.total_current, total)
    binding.close()


# --- item 9: error mapping -------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        ("raise_driven", "native"),
        ("raise_value", "interface"),
        ("raise_type", "interface"),
        ("work_then_raise", "native"),
    ],
)
def test_provider_exceptions_are_charged_and_classified(fake, failure, kind):
    binding = _bind(_descriptor(preset_basis(1)), n_nodes=3)
    fake.failure = failure
    binding.reserve(2)
    with pytest.raises(BackendError, match=rf"\[{kind}\]"):
        binding.evaluate(_fields(3), z=0.25, purpose="corrected")
    # Charged before invocation, never refunded; only the unused chunk is released.
    assert binding.ledger.attempted == 1
    assert binding.ledger.failed_calls == 1
    assert binding.ledger.reserved == 0
    failures = _meta(binding)["failures"]
    assert isinstance(failures, list)
    assert failures[0]["classification"] == kind
    assert fake.work_done == (failure == "work_then_raise")
    binding.close()


@pytest.mark.parametrize("change", ["nonfinite", "float32"])
def test_binding_validation_refuses_bad_fields_before_any_charge(fake, change):
    binding = _bind(_descriptor(), n_nodes=1)
    fields = _fields(1)
    if change == "nonfinite":
        fields[0, 3] = np.nan
    else:
        fields = fields.astype(np.float32)
    binding.reserve(1)
    with pytest.raises(BackendError, match=r"float64|finite"):
        binding.evaluate(fields, z=0.0, purpose="startup")
    assert binding.ledger.attempted == 0 and fake.calls == []
    binding.close()


def test_preparation_interface_errors_are_classified(fake, monkeypatch):
    def refuse(model, **values):
        raise TypeError("rows must be a plain int")

    monkeypatch.setattr(fake.hacc, "ResponsePlan", refuse)
    with pytest.raises(BackendError, match=r"preparation failed.*\[interface\]"):
        _bind(_descriptor(preset_basis(1)))


# --- item 10: domain flags -------------------------------------------------------


def _amplitude(intensity):
    return np.sqrt(2 * intensity / (EPS0 * C))


def test_call_above_the_preset_intensity_is_flagged_and_warned_once(fake):
    binding = _bind(_descriptor(preset_basis(1)), n_nodes=1)
    above = np.zeros((1, _N))
    above[0, _N // 2] = 1.01 * _amplitude(8e18)
    with pytest.warns(TDSEDomainWarning, match="peak input intensity") as record:
        binding.reserve(1)
        binding.evaluate(above, z=0.1, purpose="predictor")
        binding.reserve(1)
        binding.evaluate(above * 1.01, z=0.2, purpose="predictor")
    assert len(record) == 1
    metadata = _meta(binding)
    domain = metadata["domain"]
    assert isinstance(domain, dict)
    assert len(domain["flagged_calls"]) == 2
    assert metadata["last_call"]["domain"]["outside_preset_domain"] is True
    assert "intensity" in domain["reasons"]
    binding.close()


def test_call_within_the_preset_domain_carries_no_flag(fake):
    binding = _bind(_descriptor(preset_basis(1)), n_nodes=1)
    within = np.zeros((1, _N))
    within[0, _N // 2] = 0.99 * _amplitude(8e18)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binding.reserve(1)
        binding.evaluate(within, z=0.1, purpose="predictor")
    metadata = _meta(binding)
    assert metadata["last_call"]["domain"]["outside_preset_domain"] is False
    assert metadata["domain"] == {"reasons": {}, "flagged_calls": []}
    binding.close()


def test_peak_before_the_drive_taper_sets_the_flag(fake):
    descriptor = _descriptor(preset_basis(1))
    window = descriptor.atomic_window
    assert window is not None
    binding = _bind(descriptor, n_nodes=1)
    hidden = np.zeros((1, _N))
    hidden[0, window.first - 1] = 2 * _amplitude(8e18)  # outside the window
    with pytest.warns(TDSEDomainWarning):
        binding.reserve(1)
        binding.evaluate(hidden, z=0.0, purpose="startup")
    record = _meta(binding)["last_call"]["domain"]
    assert record["driven_peak_intensity_w_per_m2"] == [0.0]
    assert record["peak_intensity_w_per_m2"][0] > 8e18
    binding.close()


@pytest.mark.parametrize("relation", ["above", "equal", "below"])
def test_window_duration_is_flagged_only_above_the_preset_window(fake, relation):
    descriptor = _descriptor(preset_basis(1))
    window = descriptor.atomic_window
    assert window is not None
    duration = window.atomic_duration
    fake.preset_window = {
        "above": np.nextafter(duration, 0.0),
        "equal": duration,
        "below": 2 * duration,
    }[relation]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binding = tdse.prepare_binding(descriptor, TDSEExecution(), 1)
    flagged = [one for one in caught if issubclass(one.category, TDSEDomainWarning)]
    reasons = _meta(binding)["domain"]["reasons"]
    assert (len(flagged), "duration" in reasons) == (
        (1, True) if relation == "above" else (0, False)
    )
    binding.close()


def test_carrier_mismatch_is_its_own_recorded_reason(fake):
    fake.preset_changes = {"wavelength": 1030e-9}
    with pytest.warns(TDSEDomainWarning, match="carrier"):
        binding = tdse.prepare_binding(_descriptor(preset_basis(1)), TDSEExecution(), 1)
    reasons = _meta(binding)["domain"]["reasons"]
    assert set(reasons) == {"carrier"}
    assert reasons["carrier"]["preset_wavelength"] == 1030e-9
    binding.close()


def test_custom_basis_has_no_domain_flags(fake):
    fake.preset_window = 1e-18
    binding = _bind(_descriptor(), n_nodes=1)
    strong = np.zeros((1, _N))
    strong[0, _N // 2] = 10 * _amplitude(8e18)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binding.reserve(1)
        binding.evaluate(strong, z=0.0, purpose="startup")
    metadata = _meta(binding)
    assert "domain" not in metadata["last_call"]
    assert metadata["domain"] == {"reasons": {}, "flagged_calls": []}
    binding.close()


# --- item 19: resolvers ------------------------------------------------------------


@pytest.mark.parametrize("variant", ["standard", "fast"])
@pytest.mark.parametrize("ions", [1, 2])
def test_hacc_preset_returns_si_records_and_a_research_calibration(fake, ions, variant):
    preset = (
        tdse.hacc_preset(ions)
        if variant == "standard"
        else tdse.hacc_preset(ions, variant)
    )
    expected = preset_basis(ions, variant)
    assert fake.variant_requests == [variant]
    assert preset.basis.preset == preset.identity == expected.preset
    assert dataclasses.replace(preset.basis, source=expected.source) == expected
    assert preset.basis.source.startswith("pyroraptor@876b72d1")
    assert preset.variant == preset.basis.variant == variant
    assert preset.step_margin == (0.62 if variant == "standard" else 0.9)
    assert preset.max_atomic_step == expected.max_step_au * hacc_cases.AU_TIME
    # The fast records are the standard ones plus the measured fast-to-standard
    # change, as the provider serves them (exact: the fake's own sums).
    delta = 0.0 if variant == "standard" else hacc_cases.FAST_DELTA
    assert (preset.accuracy, preset.gauge_sensitivity) == (
        0.0122 + delta,
        0.0011 + delta,
    )
    assert preset.box_sensitivity == (0.0095 if ions == 1 else 0.0093) + delta
    assert (preset.max_peak_intensity, preset.wavelength) == (8.0e18, 800e-9)
    assert preset.window == 60e-15 and preset.device_admissible is True
    calibration = preset.calibration()
    assert calibration.status == "research" and calibration.uncertainty is None
    assert calibration.basis_identity == preset.basis.identity
    assert calibration.identity == preset.identity
    assert calibration.wavelength_range == (62e-9, 1.6e-6)
    text = " ".join(calibration.limitations)
    for word in ("window", "intensity", "closure surface", "diagnostic", "accuracy"):
        assert word in text
    assert f"numerics variant {variant!r}" in text
    assert fake.prepared == []
    # The record admits the preset's own response, at scale one.
    response = hacc_response(
        preset.basis, max_atomic_step=preset.max_atomic_step, calibration=calibration
    )
    assert isinstance(response.basis, hacc_cases.HaccBasis)
    assert response.basis.qualified


@pytest.mark.parametrize("ions", [0, 3, True, 1.0])
def test_hacc_preset_refuses_other_ion_counts_before_the_provider(fake, ions):
    with pytest.raises(MaterialError, match="ions"):
        inspect.unwrap(tdse.hacc_preset)(ions)
    assert fake.preset_requests == []


@pytest.mark.parametrize("variant", ["slow", "Fast", "", 1, None])
def test_hacc_preset_refuses_other_variants_before_the_provider(fake, variant):
    with pytest.raises(MaterialError, match="variant"):
        inspect.unwrap(tdse.hacc_preset)(1, variant)
    assert fake.preset_requests == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"variant": "standard"}, "returned the 'standard' haCC preset"),
        ({"identity": "hacc-helium-1-v1-28f0eae146b1d665"}, "does not name"),
    ],
)
def test_hacc_preset_refuses_a_provider_serving_another_variant(fake, change, message):
    fake.preset_changes = change
    with pytest.raises(BackendError, match=message):
        tdse.hacc_preset(1, "fast")


def test_hacc_basis_converts_provider_objects_into_a_custom_record(fake):
    expected = tiny_basis()
    atom, native, gauge = fake_objects(expected)
    converted = tdse.hacc_basis(
        atom,
        native,
        gauge,
        expected.cut_energy,
        expected.max_step_au,
        source=expected.source,
    )
    assert converted == expected and converted.preset is None
    velocity = tdse.hacc_basis(
        atom,
        native,
        FakeVelocityGauge(),
        expected.cut_energy,
        expected.max_step_au,
        source=expected.source,
    )
    assert velocity.gauge_radius is None
    total = tdse.hacc_basis(
        atom,
        dataclasses.replace(
            native, angular=hacc_cases.FakeAngularLimit(total=expected.angular_order)
        ),
        gauge,
        expected.cut_energy,
        expected.max_step_au,
        source=expected.source,
    )
    assert total.angular_limit == "total"


@pytest.mark.parametrize(
    ("atom", "gauge", "message"),
    [
        (FakeArgon(), None, "argon"),
        (FakeHelium(3, 6), None, "ions=3"),
        ("helium", None, "Helium atom"),
        (FakeHelium(1, 1), "length", "gauge"),
    ],
)
def test_hacc_basis_refuses_what_the_record_cannot_represent(
    fake, atom, gauge, message
):
    basis = tiny_basis()
    _, native, mixed = fake_objects(basis)
    with pytest.raises(MaterialError, match=message):
        tdse.hacc_basis(
            atom,
            native,
            mixed if gauge is None else gauge,
            basis.cut_energy,
            basis.max_step_au,
            source="refusal test",
        )


def test_resolvers_refuse_a_provider_without_the_hacc_surface(fake):
    fake.hacc = object()
    with pytest.raises(BackendError, match="haCC consumer surface"):
        tdse.hacc_preset(1)
    with pytest.raises(BackendError, match="haCC consumer surface"):
        tdse.hacc_basis(None, None, None, 1.0, 1.0, source="x")


def test_pinned_revision_is_the_documented_pin():
    assert tdse.pinned_revision() == "876b72d10b5286478125b842ea45548a5051b033"


def test_window_record_is_reconstructed_unchanged(fake):
    descriptor = _descriptor(
        preset_basis(1), atomic_window=TDSEAtomicWindow(6e-15, 1e-15)
    )
    binding = _bind(descriptor, n_nodes=1)
    assert _meta(binding)["atomic_window"]["duration"] == 6e-15
    binding.close()


# --- remaining diagnostic and preparation branches --------------------------------


def test_custom_plan_value_errors_other_than_the_closure_are_interface_failures(
    fake, monkeypatch
):
    def refuse(model, **values):
        raise ValueError("samples out of range")

    monkeypatch.setattr(fake.hacc, "ResponsePlan", refuse)
    with pytest.raises(BackendError, match=r"samples out of range \[interface\]"):
        _bind(_descriptor())


def _protocol():
    from solprop.driver.tdse_consistency import protocol_field  # noqa: PLC0415

    spacing, field = protocol_field(800e-9)
    return field[None, :], spacing


@pytest.mark.parametrize(
    ("basis", "failure", "message"),
    [
        (preset_basis(1), "plan_info", "metadata disagree"),
        (preset_basis(1), "bound_none", "bounds are malformed"),
        (preset_basis(1), "bound_negative", "bounds are malformed"),
        (preset_basis(1), "row", "failed diagnostic row"),
        (preset_basis(1), "labels", "compact diagnostic"),
        (tiny_basis(gauge_radius=None), "full_shadow_missing", "full diagnostic"),
    ],
)
def test_malformed_diagnostic_rows_are_refused(fake, basis, failure, message):
    binding: Any = _bind(_descriptor(basis), n_nodes=1)
    field, spacing = _protocol()
    fake.failure = failure
    with pytest.raises(BackendError, match=message):
        binding.diagnostic_currents(field, spacing)
    binding.close()


def test_full_record_diagnostic_of_a_control_takes_the_in_box_order_one(fake):
    binding: Any = _bind(_descriptor(tiny_basis(gauge_radius=None)), n_nodes=1)
    field, spacing = _protocol()
    total, first = binding.diagnostic_currents(field, spacing)
    expected_first, expected_total = manufactured_currents(field, spacing)
    assert_array_equal(first, expected_first[0])
    assert_array_equal(total, expected_total[0])
    binding.close()


def test_sae_binding_has_no_consistency_diagnostic(monkeypatch):
    from tests.driver.test_tdse_binding import _descriptor as sae  # noqa: PLC0415
    from tests.driver.test_tdse_binding import _Provider  # noqa: PLC0415

    provider = _Provider()
    monkeypatch.setattr(tdse, "_load_provider", lambda: provider)
    binding: Any = tdse.prepare_binding(sae("tdse_native"), TDSEExecution(), 1)
    field, spacing = _protocol()
    with pytest.raises(BackendError, match="needs a haCC binding"):
        binding.diagnostic_currents(field, spacing)
    binding.close()


def test_diagnostic_session_classifies_preparation_failures_and_reports(
    fake, monkeypatch
):
    session = tdse.DiagnosticSession(preset_basis(1), TDSEExecution())
    metadata = _meta(session)
    assert metadata["hacc"]["preset"] == preset_basis(1).preset
    assert metadata["ledger"]["attempted"] == 0
    session.close()
    session.close()

    def refuse(*args, **kwargs):
        raise TypeError("gauge must be MixedGauge or VelocityGauge")

    monkeypatch.setattr(fake.hacc, "prepare", refuse)
    with pytest.raises(BackendError, match=r"preparation failed.*\[interface\]"):
        tdse.DiagnosticSession(tiny_basis(), TDSEExecution())


def test_budget_refuses_an_invalid_explicit_call_count():
    from solprop.translate.tdse import AtomicCallBudget  # noqa: PLC0415

    budget = AtomicCallBudget(None, 2)
    with pytest.raises(BackendError, match="calls per evaluation"):
        budget.reserve(1, calls=0)
    budget.reserve(1, calls=1)
    with pytest.raises(BackendError, match="calls per evaluation"):
        budget.begin_evaluation(purpose="diagnostic", calls=True)


# --- deferred propagation plans: the diagnostic plan never adds to their peak -----


def _deferred(descriptor, execution=None, n_nodes=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        return tdse.prepare_binding(
            descriptor,
            TDSEExecution(max_batch_rows=2) if execution is None else execution,
            n_nodes,
            defer_plans=True,
        )


def test_deferred_plans_close_the_diagnostic_plan_before_building(fake):
    descriptor = _descriptor(preset_basis(1))
    window = descriptor.atomic_window
    assert window is not None
    estimate = sum(
        tdse.estimate_binding_memory(descriptor, TDSEExecution(), 3).values()
    )
    # The propagation plans (rows 1 and 2) and the one-row diagnostic each fit beside
    # the model; together they would not.
    fake.plan_bytes = 10**6
    limit = estimate + fake.model_bytes + 3 * fake.plan_bytes + 10**4
    binding: Any = _deferred(
        descriptor, TDSEExecution(max_batch_rows=2, memory_limit=limit)
    )
    assert fake.plan_arguments == [] and not binding.plans_built
    with pytest.raises(BackendError, match="call build_plans"):
        binding.reserve(1)
    with pytest.raises(BackendError, match="call build_plans"):
        binding.evaluate(_fields(3), z=0.0, purpose="startup")
    field, spacing = _protocol()
    binding.diagnostic_currents(field, spacing)
    binding.build_plans()
    assert binding.plans_built
    diagnostic = f"plan-1-{field.shape[1]}"
    assert fake.events.index(f"close-{diagnostic}") < fake.events.index(
        f"enter-plan-1-{window.samples}"
    )
    assert "plan.2.march.state" in binding.memory_ledger()
    binding.reserve(1)
    binding.evaluate(_fields(3), z=0.0, purpose="startup")
    with pytest.raises(BackendError, match="already built"):
        binding.build_plans()
    binding.close()


def test_overlapping_plans_would_have_exceeded_that_limit(fake):
    descriptor = _descriptor(preset_basis(1))
    estimate = sum(
        tdse.estimate_binding_memory(descriptor, TDSEExecution(), 3).values()
    )
    fake.plan_bytes = 10**6
    limit = estimate + fake.model_bytes + 3 * fake.plan_bytes + 10**4
    binding: Any = _bind(
        descriptor, TDSEExecution(max_batch_rows=2, memory_limit=limit)
    )
    field, spacing = _protocol()
    with pytest.raises(MemoryBudgetError):
        binding.diagnostic_currents(field, spacing)
    binding.close()


@pytest.mark.parametrize("failure", ["memory", "plan_driven", "plan_info"])
def test_a_failed_deferred_build_closes_every_resource(fake, failure):
    descriptor = _descriptor(preset_basis(1))
    if failure == "memory":
        estimate = sum(
            tdse.estimate_binding_memory(descriptor, TDSEExecution(), 3).values()
        )
        execution = TDSEExecution(memory_limit=estimate + fake.model_bytes + 10)
        fake.plan_bytes = 10**6
    else:
        execution = TDSEExecution(max_batch_rows=2)
    binding: Any = _deferred(descriptor, execution)
    fake.failure, fake.fail_rows = failure, 1
    expected = MemoryBudgetError if failure == "memory" else BackendError
    with pytest.raises(expected):
        binding.build_plans()
    assert fake.events[-1] == "close-model"
    with pytest.raises(BackendError, match="closed"):
        binding.build_plans()


def test_a_custom_deferred_build_warns_when_its_closure_is_known(fake):
    fake.closure_admitted = False
    binding: Any = tdse.prepare_binding(
        _descriptor(tiny_basis()), TDSEExecution(), 1, defer_plans=True
    )
    with pytest.warns(TDSEUnqualifiedModelWarning, match="escaped-electron"):
        binding.build_plans()
    binding.close()


def test_defer_plans_must_be_a_bool_and_is_ignored_for_sae(monkeypatch):
    from tests.driver.test_tdse_binding import _descriptor as sae  # noqa: PLC0415
    from tests.driver.test_tdse_binding import _Provider  # noqa: PLC0415

    provider = _Provider()
    monkeypatch.setattr(tdse, "_load_provider", lambda: provider)
    binding: Any = tdse.prepare_binding(
        sae("tdse_native"), TDSEExecution(), 1, defer_plans=True
    )
    assert binding.plans_built and provider.created
    binding.close()
    with pytest.raises(BackendError, match="defer_plans"):
        inspect.unwrap(tdse.prepare_binding)(
            sae("tdse_native"), TDSEExecution(), 1, defer_plans=1
        )
