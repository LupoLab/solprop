# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check TDSE lattice preservation, owning results and transactional budgets."""

import dataclasses
import hashlib
import inspect
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tests.hacc_cases import hacc_response, preset_basis

from legacy_geometry import LEGACY_GRID
from solprop import fourier
from solprop.exceptions import BackendError, ResponseBudgetError, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEConsistency,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.output.schema import serialise
from solprop.translate.tdse import (
    AtomicCallBudget,
    AtomicCallLedger,
    TDSEDescriptor,
    TDSEResult,
    _response_from_json,
    refine_response_grid,
    waveform_identity,
)


@pytest.fixture
def grid():
    # The grid the pinned identities below were taken on.
    return FieldGrid.from_limits(0.01, 800e-9, (120e-9, 1.6e-6), 60e-15, **LEGACY_GRID)


@pytest.fixture
def response():
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="contract fixture"
    )
    calibration = TDSECalibration(
        "research-fixture",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="contract fixture",
        limitations=("unqualified",),
    )
    return TDSEGasResponse(
        "helium", basis, 1e-19, "optmat_cubic", 2.0, calibration, TDSESampling(4)
    )


def test_response_refinement_preserves_parent_coordinates_and_transforms(grid):
    refined = refine_response_grid(grid, TDSESampling(4))
    assert refined.n_to == 4 * grid.n_to
    assert refined.n_omega == grid.n_omega
    assert refined.dt == grid.dt
    assert refined.d_omega == grid.d_omega
    assert refined.n_to * refined.dt_fine == grid.n_to * grid.dt_fine
    assert refined.omega_collar == grid.omega_collar
    assert refined.t_collar == grid.t_collar
    assert_array_equal(refined.t, grid.t)
    assert_array_equal(refined.omega, grid.omega)
    assert_array_equal(refined.t_fine[::4], grid.t_fine)
    assert_array_equal(refined.omega_fine[: grid.n_omega_fine], grid.omega_fine)
    samples = np.exp(-((grid.t / 4e-15) ** 2)) * np.cos(grid.omega0 * grid.t)
    spectrum = fourier.forward(samples, kind="field") * grid.omega_window
    coarse = fourier.inverse(spectrum, kind="field")
    fine = fourier.inverse(
        fourier.oversample(spectrum, refined.n_to, kind="field"), kind="field"
    )
    # Two FFTs differ only by transform rounding; amplitudes include the pad scale.
    tolerance = 16 * np.finfo(float).eps * np.log2(refined.n_to)
    assert_allclose(
        fine[:: refined.n_to // grid.n_t], coarse, rtol=tolerance, atol=tolerance
    )
    assert_allclose(
        fourier.crop(
            fourier.oversample(spectrum, refined.n_to, kind="field"),
            grid.n_t,
            kind="field",
        ),
        spectrum,
        rtol=2 * np.finfo(float).eps,
        atol=0,
    )


def test_sampling_one_reuses_grid_and_envelope_is_refused(grid):
    assert refine_response_grid(grid, TDSESampling()) is grid
    envelope = EnvelopeGrid.from_limits(0.01, 800e-9, (400e-9, 1.6e-6), 60e-15)
    with pytest.raises(TranslationError, match="FieldGrid"):
        refine_response_grid(envelope, TDSESampling(4))


def test_sampling_refuses_spacing_underflow_before_allocating_refined_grid(grid):
    with pytest.raises(TranslationError, match="underflows"):
        refine_response_grid(grid, TDSESampling(1 << 1023))


@pytest.mark.parametrize(
    "policy",
    [
        ("optmat_cubic", 2.0, None, 3, "total_current"),
        ("tdse_shadow", 2.0, 1.537, 3, "full"),
        ("tdse_native", 1.0, None, 1, "total_current"),
        ("tdse_native", 1.47, None, 1, "total_current"),
    ],
)
def test_descriptor_resolves_exact_source_and_setup_policy(grid, response, policy):
    model, scale, cubic, order, outputs = policy
    chosen = dataclasses.replace(
        response, kerr_model=model, remainder_scale=scale, cubic_scale=cubic
    )
    refined = refine_response_grid(grid, chosen.sampling)
    descriptor = TDSEDescriptor.from_response(chosen, refined)
    assert descriptor.shadow_order == order
    assert descriptor.outputs == outputs
    assert descriptor.samples == refined.n_to
    assert descriptor.spacing == refined.dt_fine
    assert descriptor.time_origin == grid.t_fine[0]
    specification = json.loads(descriptor.configuration_json)
    assert specification["response"]["basis"]["radial_edges"] == list(
        response.basis.radial_edges
    )
    assert specification["lattice"]["period"] == grid.n_to * grid.dt_fine
    assert (
        specification["lattice"]["atomic_duration"]
        == (refined.n_to - 1) * refined.dt_fine
    )
    assert descriptor.initial_l == descriptor.initial_index == 0
    assert descriptor.growing == "remove"
    assert descriptor.window_duration == (refined.n_to - 1) * refined.dt_fine
    assert specification["setup"]["window_duration"] == descriptor.window_duration
    assert not descriptor.accept_removed_component
    assert TDSEDescriptor.from_response(chosen, refined).identity == descriptor.identity
    with pytest.raises(TranslationError, match="identity"):
        dataclasses.replace(descriptor, identity="not the digest")
    with pytest.raises(TranslationError, match="disagree"):
        dataclasses.replace(descriptor, remainder_scale=7.0)
    with pytest.raises(TranslationError, match="disagree"):
        dataclasses.replace(descriptor, window_duration=None)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"samples": 3}, "at least four samples"),
        ({"spacing": 0.0}, "invalid time lattice"),
        ({"spacing": float("inf")}, "invalid time lattice"),
        ({"max_atomic_step": -1e-19}, "invalid time lattice"),
        ({"time_origin": float("nan")}, "invalid time lattice"),
    ],
)
def test_descriptor_refuses_invalid_scalar_lattice_before_identity_comparison(
    grid, response, changes, message
):
    descriptor = TDSEDescriptor.from_response(
        response, refine_response_grid(grid, response.sampling)
    )
    with pytest.raises(TranslationError, match=message):
        dataclasses.replace(descriptor, **changes)


def _result(**changes):
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    kwargs = {
        "total_current": values,
        "first_order_current": values / 2,
        "third_order_current": values / 4,
        "third_order_dipole": values / 8,
        "input_identity": "field-hash",
        "configuration_identity": "descriptor-hash",
        "valid": np.ones((3, 4), dtype=bool),
        "shadow_orders": (1, 3),
        "monitors_json": '{"norm": [1.0, 1.0, 1.0]}',
    }
    kwargs.update(changes)
    return TDSEResult(**kwargs)


def test_results_own_every_array_and_keep_unscaled_sources():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = _result(total_current=values)
    values[:] = -99
    assert_array_equal(result.total_current, np.arange(12).reshape(3, 4))
    for name in (
        "total_current",
        "first_order_current",
        "third_order_current",
        "third_order_dipole",
        "valid",
    ):
        array = getattr(result, name)
        assert array.flags.owndata and not array.flags.writeable
    native = _result(
        shadow_orders=(1,), third_order_current=None, third_order_dipole=None
    )
    assert native.third_order_current is None
    compact = _result(third_order_dipole=None)
    assert compact.third_order_dipole is None


def test_result_refuses_invalid_rows_shapes_missing_orders_and_nonfinite_values():
    valid = np.ones((3, 4), dtype=bool)
    valid[1, 2] = False
    with pytest.raises(BackendError, match="row 1, sample 2"):
        _result(valid=valid)
    with pytest.raises(BackendError, match="validity"):
        _result(valid=np.ones((2, 4), dtype=bool))
    with pytest.raises(BackendError, match="shape"):
        _result(first_order_current=np.zeros((3, 3)))
    with pytest.raises(BackendError, match="order label"):
        _result(third_order_current=None)
    with pytest.raises(BackendError, match="requires order 3"):
        _result(shadow_orders=(1,), third_order_current=None)
    with pytest.raises(BackendError, match="nonfinite"):
        _result(total_current=np.full((3, 4), np.nan))
    with pytest.raises(BackendError, match="identities"):
        _result(input_identity="")


@pytest.mark.parametrize("shape", [(0, 4), (3, 0)])
def test_result_refuses_empty_node_or_sample_axis(shape):
    with pytest.raises(BackendError, match="nonempty"):
        _result(total_current=np.empty(shape))


@pytest.mark.parametrize("orders", [(), (3,), (1, 2), (1, 3, 7)])
def test_result_refuses_unknown_or_incomplete_shadow_order_labels(orders):
    with pytest.raises(BackendError, match="available odd orders"):
        _result(shadow_orders=orders)


@pytest.mark.parametrize("value", ['{"norm": NaN}', '{"norm": 1e999}', "[]", "invalid"])
def test_result_diagnostics_require_finite_json_object(value):
    with pytest.raises(BackendError, match="monitors_json"):
        _result(monitors_json=value)


def test_waveform_identity_uses_values_shape_and_every_row():
    field = np.arange(24, dtype=np.float64).reshape(3, 8)
    assert waveform_identity(field) == waveform_identity(np.asfortranarray(field))
    assert waveform_identity(field) != waveform_identity(field.reshape(6, 4))
    changed = field.copy()
    changed[2, -1] += 1
    assert waveform_identity(field) != waveform_identity(changed)
    with pytest.raises(BackendError, match="finite"):
        waveform_identity(np.full((1, 4), np.nan))
    with pytest.raises(BackendError, match="nonempty"):
        waveform_identity(np.empty((0, 4)))


def test_budget_reserves_whole_trial_and_counts_chunks_without_rollback():
    budget = AtomicCallBudget(5, 2)
    budget.reserve(2, z=0.4)
    budget.begin_evaluation(purpose="startup")
    budget.charge(purpose="startup")
    budget.charge(purpose="startup")
    budget.begin_evaluation(purpose="predictor")
    budget.charge(purpose="predictor")
    budget.record_failure()
    budget.release_reservation()
    assert budget.snapshot() == AtomicCallLedger(5, 3, 0, 2, 0, 1)
    with pytest.raises(ResponseBudgetError) as caught:
        budget.reserve(2, z=0.5)
    assert caught.value.z == 0.5
    assert caught.value.field is None
    assert caught.value.calls == 3
    assert budget.snapshot().attempted == 3
    budget.reserve(1)
    budget.begin_evaluation(purpose="diagnostic")
    budget.charge(purpose="diagnostic")
    budget.charge(purpose="diagnostic")
    budget.release_reservation()
    assert budget.snapshot() == AtomicCallLedger(5, 5, 0, 3, 2, 1)
    with pytest.raises(ResponseBudgetError):
        budget.reserve(1)


def test_budget_detects_unreserved_calls_and_inconsistent_purposes():
    budget = AtomicCallBudget(None, 1)
    with pytest.raises(BackendError, match="reservation"):
        budget.begin_evaluation(purpose="predictor")
    with pytest.raises(BackendError, match="unreserved"):
        budget.charge(purpose="predictor")
    with pytest.raises(BackendError, match="failure"):
        budget.record_failure()
    budget.reserve(1)
    with pytest.raises(BackendError, match="prior"):
        budget.reserve(1)
    budget.begin_evaluation(purpose="corrected")
    with pytest.raises(BackendError, match="mislabelled"):
        budget.charge(purpose="predictor")
    budget.charge(purpose="corrected")
    budget.record_failure()
    with pytest.raises(BackendError, match="failure"):
        budget.record_failure()
    budget.release_reservation()
    assert budget.snapshot().attempted == budget.snapshot().failed_calls == 1


@pytest.mark.parametrize("calls", [0, -1, True])
def test_budget_refuses_invalid_reservation_without_changing_accounting(calls):
    budget = AtomicCallBudget(4, 2)
    before = budget.snapshot()
    with pytest.raises(BackendError, match="positive integer"):
        budget.reserve(calls)
    assert budget.snapshot() == before


def test_budget_refuses_unknown_purpose_without_spending_reservation():
    budget = AtomicCallBudget(4, 2)
    budget.reserve(1)
    before = budget.snapshot()
    # The production guard remains necessary without test-only Literal checking.
    with pytest.raises(BackendError, match="unknown TDSE evaluation purpose"):
        inspect.unwrap(AtomicCallBudget.begin_evaluation)(budget, purpose="unknown")
    assert budget.snapshot() == before


def test_resume_retains_cumulative_charges_and_uses_the_new_explicit_cap():
    prior = AtomicCallLedger(4, 3, 1, 2, 1, 1)
    resumed = AtomicCallBudget(8, 2)
    resumed.restore(prior)
    assert resumed.snapshot() == AtomicCallLedger(8, 3, 0, 2, 1, 1)
    resumed.reserve(1)
    resumed.begin_evaluation(purpose="startup")
    resumed.charge(purpose="startup")
    resumed.charge(purpose="startup")
    resumed.release_reservation()
    assert resumed.snapshot().attempted == 5
    with pytest.raises(BackendError, match="before beginning"):
        resumed.restore(prior)
    with pytest.raises(BackendError, match="below"):
        AtomicCallBudget(2, 2).restore(prior)
    with pytest.raises(BackendError, match="subcounts"):
        AtomicCallBudget(8, 2).restore(dataclasses.replace(prior, failed_calls=4))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"attempted": -1}, "nonnegative integers"),
        ({"reserved": -1}, "nonnegative integers"),
        ({"logical_evaluations": -1}, "nonnegative integers"),
        ({"diagnostic_calls": -1}, "nonnegative integers"),
        ({"failed_calls": -1}, "nonnegative integers"),
        ({"limit": 0}, "original call limit"),
        ({"limit": 3}, "original call limit"),
    ],
)
def test_restore_revalidates_ledger_before_modifying_live_budget(changes, message):
    prior = dataclasses.replace(AtomicCallLedger(4, 3, 1, 2, 1, 1), **changes)
    budget = AtomicCallBudget(8, 2)
    before = budget.snapshot()
    with pytest.raises(BackendError, match=message):
        budget.restore(prior)
    assert budget.snapshot() == before


@pytest.mark.parametrize(("limit", "chunks"), [(0, 1), (3, 0), (-1, 2)])
def test_budget_refuses_invalid_configuration(limit, chunks):
    with pytest.raises(TranslationError, match="positive"):
        AtomicCallBudget(limit, chunks)


def test_atomic_ledger_record_roundtrip_and_restore_releases_reserved_calls():
    snapshot = AtomicCallLedger(7, 4, 2, 3, 1, 1)
    restored = AtomicCallLedger.from_record(dataclasses.asdict(snapshot))
    assert restored == snapshot
    budget = AtomicCallBudget(9, 2)
    budget.restore(restored)
    assert budget.snapshot() == AtomicCallLedger(9, 4, 0, 3, 1, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 0},
        {"attempted": -1},
        {"reserved": 1.0},
        {"logical_evaluations": True},
        {"limit": 0},
        {"limit": True},
        {"limit": 3},
        {"diagnostic_calls": 5},
        {"failed_calls": 5},
    ],
)
def test_atomic_ledger_refuses_invalid_persisted_counts(changes):
    record = dataclasses.asdict(AtomicCallLedger(7, 4, 2, 3, 1, 1))
    with pytest.raises(BackendError):
        AtomicCallLedger.from_record({**record, **changes})


def test_atomic_ledger_requires_all_fields_and_accepts_unlimited_budget():
    record = dataclasses.asdict(AtomicCallLedger(None, 4, 2, 3, 1, 1))
    assert AtomicCallLedger.from_record(record).limit is None
    del record["reserved"]
    with pytest.raises(BackendError, match="missing or unknown"):
        AtomicCallLedger.from_record(record)


@pytest.mark.parametrize(
    "mutation",
    ["target", "basis", "calibration", "sampling", "orders", "setup", "schema"],
)
def test_descriptor_refuses_self_consistent_but_unsupported_json(
    grid, response, mutation
):
    descriptor = TDSEDescriptor.from_response(
        response, refine_response_grid(grid, response.sampling)
    )
    record = json.loads(descriptor.configuration_json)
    fields = {}
    if mutation == "target":
        record["response"]["target"] = "argon"
    elif mutation == "basis":
        record["response"]["basis"]["potential"] = "argon_tong05"
    elif mutation == "calibration":
        record["response"]["calibration"]["basis_identity"] = "wrong basis"
    elif mutation == "sampling":
        record["response"]["sampling"]["refinement"] = 3
    elif mutation == "orders":
        record["shadow_order"] = fields["shadow_order"] = 1
    elif mutation == "setup":
        record["setup"]["initial_l"] = fields["initial_l"] = 1
    else:
        record["schema"] = "future-unsupported-schema"
    encoded = json.dumps(record, sort_keys=True)
    with pytest.raises(TranslationError, match="invalid"):
        dataclasses.replace(
            descriptor,
            configuration_json=encoded,
            identity=hashlib.sha256(encoded.encode()).hexdigest(),
            **fields,
        )


def _expected_descriptor_json(response, refined):
    """Encode the one descriptor layout independently of the production encoder."""
    record = dataclasses.asdict(response)
    del record["consistency"]
    record = {
        "schema": "solprop.tdse-descriptor.v5",
        "response": record,
        "setup": {
            "initial_l": 0,
            "initial_index": 0,
            "growing": "remove",
            "growth_tolerance": 1e-3,
            "window_duration": (refined.n_to - 1) * refined.dt_fine,
            "accept_removed_component": False,
        },
        "lattice": {
            "samples": refined.n_to,
            "spacing": refined.dt_fine,
            "time_origin": float(refined.t_fine[0]),
            "period": refined.n_to * refined.dt_fine,
            "atomic_duration": (refined.n_to - 1) * refined.dt_fine,
            "coarse_samples": refined.n_t,
            "coarse_spacing": refined.dt,
            "reference_wavelength": refined.reference_wavelength,
            "omega_collar": dataclasses.asdict(refined.omega_collar),
            "time_collar": dataclasses.asdict(refined.t_collar),
            "interpolation": "provider_cubic",
            "tail_policy": "unchanged",
        },
        "shadow_order": 1 if response.kerr_model == "tdse_native" else 3,
        "outputs": "full" if response.kerr_model == "tdse_shadow" else "total_current",
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow", "tdse_native"])
def test_every_descriptor_has_the_one_layout(grid, response, model):
    chosen = dataclasses.replace(
        response,
        kerr_model=model,
        cubic_scale=1.537 if model == "tdse_shadow" else None,
    )
    refined = refine_response_grid(grid, chosen.sampling)
    expected = _expected_descriptor_json(chosen, refined)
    descriptor = TDSEDescriptor.from_response(chosen, refined)
    assert descriptor.configuration_json == expected
    assert descriptor.identity == hashlib.sha256(expected.encode()).hexdigest()
    record = json.loads(expected)
    # No source filter is recorded: the native source is gated, the cubic ones not.
    assert "source_filter" not in record["response"]
    assert descriptor.gated is (model == "tdse_native")
    assert record["response"]["model"] == "sae"
    assert record["response"]["atomic_window"] is None
    # The provider checks growing modes over exactly the propagated atomic run.
    assert descriptor.window_duration == record["lattice"]["atomic_duration"]
    assert TDSEDescriptor(**dataclasses.asdict(descriptor)) == descriptor


def test_native_identity_tracks_temporal_collar_geometry(grid, response):
    native = dataclasses.replace(response, kerr_model="tdse_native")
    refined = refine_response_grid(grid, native.sampling)
    changed = dataclasses.replace(
        refined,
        t_collar=dataclasses.replace(
            refined.t_collar,
            one_low=0.95 * refined.t_collar.one_low,
            one_high=0.95 * refined.t_collar.one_high,
        ),
    )
    assert TDSEDescriptor.from_response(native, refined).identity != (
        TDSEDescriptor.from_response(native, changed).identity
    )
    assert_array_equal(changed.t_fine, refined.t_fine)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_v1",
        "schema_v2",
        "schema_v3",
        "schema_v4",
        "a_source_filter",
        "missing_model",
        "missing_window",
        "wrong_closure",
        "raw_tail_policy",
        "bad_collar",
        "missing_collar",
    ],
)
def test_descriptor_refuses_inconsistent_records(grid, response, mutation):
    # Only the one layout is read: the earlier descriptor versions are refused by
    # their schema tag alone, whatever they hold.
    native = dataclasses.replace(response, kerr_model="tdse_native")
    descriptor = TDSEDescriptor.from_response(
        native, refine_response_grid(grid, native.sampling)
    )
    record = json.loads(descriptor.configuration_json)
    collar = record["lattice"]["time_collar"]
    mutations = {
        "schema_v1": lambda: record.update(schema="solprop.tdse-descriptor.v1"),
        "schema_v2": lambda: record.update(schema="solprop.tdse-descriptor.v2"),
        "schema_v3": lambda: record.update(schema="solprop.tdse-descriptor.v3"),
        "schema_v4": lambda: record.update(schema="solprop.tdse-descriptor.v4"),
        "a_source_filter": lambda: record["response"].update(
            source_filter="gated_derivative"
        ),
        "missing_model": lambda: record["response"].pop("model"),
        "missing_window": lambda: record["response"].pop("atomic_window"),
        "wrong_closure": lambda: record["response"].update(kerr_model="optmat_cubic"),
        "raw_tail_policy": lambda: record["lattice"].update(tail_policy="gated"),
        "bad_collar": lambda: collar.update(one_high=collar["zero_high"]),
        "missing_collar": lambda: record["lattice"].pop("time_collar"),
    }
    mutations[mutation]()
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with pytest.raises(TranslationError, match="invalid"):
        dataclasses.replace(
            descriptor,
            configuration_json=encoded,
            identity=hashlib.sha256(encoded.encode()).hexdigest(),
        )


# The identities and response-record digests of three native descriptors in the
# version-5 layout, pinned when the source filter was removed (the record digests once
# every field was recorded at its default), so that any later change to either layout
# is a deliberate one.
_PINNED = {
    "sae_collar": (
        "3a269c90ad9e947093772b9a27d862edc2b5fed157e8d3597221bf6bafab3d39",
        "7b988d72152102d15b4020a211b6422aa6148f8f8821ac0e7f97cf57483dd5a5",
    ),
    "sae_window": (
        "d07327940d0377edaec75ed802f464a6556902d8be454624303807ec302db00c",
        "d33f1252dd1af6759b69efdeac8221ecbc717c87659e15d9769be889a91f56c7",
    ),
    "hacc_window": (
        "67bbfa21c5a26c5897afd84a763500b3fa23d2dc12a9cceee6efa88302b1d66f",
        "f711baa697c224a6bf08856abeb1e8bb14d8c18d0d5b95c0982f0fbaca9bdc07",
    ),
}


@pytest.mark.parametrize("kind", sorted(_PINNED))
def test_native_descriptors_keep_their_pinned_identities(grid, response, kind):
    native = dataclasses.replace(
        response,
        kerr_model="tdse_native",
        remainder_scale=1.0,
    )
    declared = {
        "sae_collar": native,
        "sae_window": dataclasses.replace(
            native, atomic_window=TDSEAtomicWindow(40e-15, 5e-15)
        ),
        "hacc_window": hacc_response(
            preset_basis(1),
            sampling=TDSESampling(4),
            atomic_window=TDSEAtomicWindow(40e-15, 5e-15),
        ),
    }[kind]
    descriptor = TDSEDescriptor.from_response(
        declared, refine_response_grid(grid, declared.sampling)
    )
    identity, response_record = _PINNED[kind]
    assert descriptor.identity == identity
    assert (
        hashlib.sha256(
            json.dumps(serialise(declared), sort_keys=True).encode()
        ).hexdigest()
        == response_record
    )


def _hacc_descriptor(grid, **changes):
    return TDSEDescriptor.from_response(hacc_response(**changes), grid)


@pytest.mark.parametrize("kind", ["preset", "custom", "sae_window"])
def test_a_windowed_descriptor_round_trips_each_response_kind(grid, response, kind):
    if kind == "preset":
        declared = hacc_response(preset_basis(2))
    elif kind == "custom":
        declared = hacc_response()
    else:
        declared = dataclasses.replace(
            response,
            kerr_model="tdse_native",
            sampling=TDSESampling(),
            atomic_window=TDSEAtomicWindow(40e-15, 4e-15),
        )
    descriptor = TDSEDescriptor.from_response(declared, grid)
    record = json.loads(descriptor.configuration_json)
    assert record["schema"] == "solprop.tdse-descriptor.v5"
    assert descriptor.gated
    assert _response_from_json(descriptor.configuration_json) == declared
    assert descriptor.response == declared
    assert descriptor.model == declared.model
    assert record["response"]["model"] == declared.model
    assert "consistency" not in record["response"]
    window = descriptor.atomic_window
    assert window is not None
    assert record["lattice"]["atomic_window"] == window.to_json()
    assert record["lattice"]["atomic_duration"] == window.atomic_duration
    if kind == "sae_window":
        assert "kind" not in record["response"]["basis"]
        assert record["setup"]["window_duration"] == (
            (window.samples - 1) * descriptor.spacing
        )
        assert descriptor.window_duration == record["setup"]["window_duration"]
    else:
        assert isinstance(declared.basis, HaccBasis)
        assert record["response"]["basis"]["kind"] == "hacc_helium"
        assert record["response"]["basis"]["preset"] == declared.basis.preset
        assert record["setup"] == {
            "initial": "ground_root",
            "restart": "every_evaluation",
            "growing": "refuse_below_cut",
            "norm_growth_limit": 2.0,
        }
        assert descriptor.window_duration is None
        assert descriptor.growing == "refuse_below_cut"
    assert (descriptor.shadow_order, descriptor.outputs) == (1, "total_current")
    assert TDSEDescriptor(**dataclasses.asdict(descriptor)) == descriptor


def test_unwindowed_hacc_descriptor_keeps_the_grid_collar_as_its_gate(grid):
    descriptor = _hacc_descriptor(grid, atomic_window=None)
    record = json.loads(descriptor.configuration_json)
    assert record["schema"] == "solprop.tdse-descriptor.v5"
    assert descriptor.gated
    assert descriptor.atomic_window is None and descriptor.source_gate() is None
    assert "atomic_window" not in record["lattice"]
    assert record["response"]["atomic_window"] is None


def test_consistency_policy_never_changes_descriptor_identity(grid):
    plain = _hacc_descriptor(grid)
    policies = (
        TDSEConsistency(warn=0.05, refuse=0.6),
        TDSEConsistency(enforce=False),
    )
    for policy in policies:
        changed = _hacc_descriptor(grid, consistency=policy)
        assert changed.identity == plain.identity
        assert changed.configuration_json == plain.configuration_json
    assert serialise(hacc_response()) == serialise(
        hacc_response(consistency=policies[0])
    )


def _tamper(record, mutation):
    """Apply one named change to a windowed haCC record; return changed scalars."""
    response, lattice, setup = record["response"], record["lattice"], record["setup"]
    window = lattice.get("atomic_window")
    changes = {
        "preset": lambda: response["basis"].update(
            preset="hacc-helium-1-v1-28f0eae146b1d665"
        ),
        "ions": lambda: response["basis"].update(ions=2),
        "kind": lambda: response["basis"].update(kind="argon_tdcis"),
        "model": lambda: response.update(model="sae"),
        "window_first": lambda: window.update(first=window["first"] + 1),
        "window_gate": lambda: window["gate"].__setitem__(1, window["gate"][1] * 1.01),
        "window_missing": lambda: lattice.pop("atomic_window"),
        "declared_window": lambda: response["atomic_window"].update(taper=2e-15),
        "setup_growth": lambda: setup.update(norm_growth_limit=3.0),
        "setup_restart": lambda: setup.update(restart="continue"),
        "setup_extra": lambda: setup.update(window_duration=1e-15),
        "v3_schema": lambda: record.update(schema="solprop.tdse-descriptor.v3"),
    }
    scalars = {
        "order": ("shadow_order", 3, record),
        "outputs": ("outputs", "full", record),
        "remainder_scale": ("remainder_scale", 1.537, response),
    }
    if mutation in scalars:
        name, value, target = scalars[mutation]
        target[name] = value
        return {name: value}
    changes[mutation]()
    return {}


@pytest.mark.parametrize(
    "mutation",
    [
        "preset",
        "ions",
        "kind",
        "model",
        "window_first",
        "window_gate",
        "window_missing",
        "declared_window",
        "setup_growth",
        "setup_restart",
        "setup_extra",
        "order",
        "outputs",
        "v3_schema",
        "remainder_scale",
    ],
)
def test_a_hacc_descriptor_refuses_tampering(grid, mutation):
    descriptor = _hacc_descriptor(grid)
    record = json.loads(descriptor.configuration_json)
    fields = _tamper(record, mutation)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with pytest.raises(TranslationError, match="invalid"):
        dataclasses.replace(
            descriptor,
            configuration_json=encoded,
            identity=hashlib.sha256(encoded.encode()).hexdigest(),
            **fields,
        )


def test_a_hacc_descriptor_refuses_changed_scalars(grid):
    descriptor = _hacc_descriptor(grid)
    for name, value in (
        ("growing", "remove"),
        ("window_duration", 1e-15),
        ("max_atomic_step", 2 * descriptor.max_atomic_step),
    ):
        with pytest.raises(TranslationError, match="disagree"):
            dataclasses.replace(descriptor, **{name: value})
