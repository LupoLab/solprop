# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Archive what the binding passes to and receives from a fake provider (SEP 20).

Section 7.1 items 21 and 22 at the binding: SAE and haCC records in the required
and full selections, over several chunks, read back exactly; the source arrays of
both selections agree bitwise with an unarchived binding's; failures keep their
input and exception; and a failed write stops the binding before its next provider
call, with a provider failure kept primary. The fakes' currents are exact
polynomials of the field, so every comparison is a copy.
"""

import inspect
import json
import warnings
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests.driver.test_tdse_binding import _Compact, _Full, _Handle, _Outcome
from tests.driver.test_tdse_binding import _descriptor as sae_descriptor
from tests.driver.test_tdse_binding import _Provider as SaeProvider
from tests.hacc_cases import (
    hacc_response,
    install,
    preset_basis,
    tiny_basis,
)
from tests.tdse_archive_cases import Info, SaeMonitors, Shadow, writer

from solprop.driver import tdse
from solprop.driver.tdse_consistency import check_consistency, protocol_field
from solprop.exceptions import (
    BackendError,
    TDSEArchiveError,
    TDSEUnqualifiedModelWarning,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output import TDSEArchive, open_tdse_archive
from solprop.output import tdse_archive as archive_module
from solprop.translate.tdse import TDSEDescriptor

_SAE_SAMPLES = 64
_GRID = FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 1600e-9), 10e-15)


# --- a provider whose SAE records carry every field ----------------------------------


class _RichPlan(_Handle):
    """An SAE plan returning complete records; full and compact agree exactly."""

    def __init__(self, provider, **values):
        rows = values["rows"]
        super().__init__(
            provider,
            f"plan-{rows}",
            (SimpleNamespace(name="state", bytes=rows * 10, shared=False),),
        )
        self.info = Info(
            rows,
            values["samples"],
            values["spacing"],
            values["outputs"],
            values["shadow_order"],
        )
        provider.created.append(self)

    def evaluate(self, fields):
        provider = self.provider
        provider.calls.append(np.array(fields, copy=True))
        if provider.failure == "raise":
            raise provider.DrivenError("synthetic native failure")
        rows, samples = fields.shape
        first, third = 2 * fields, 3 * fields**3
        total = first + third + fields**5
        orders = (1,) if self.info.shadow_order == 1 else (1, 3)
        zeros = np.zeros_like(fields)

        def ordered(*parts):
            return np.stack(parts[: len(orders)], axis=1)

        common = {
            "time": np.arange(samples) * self.info.spacing,
            "outcomes": tuple(_Outcome() for _ in range(rows)),
            "valid": np.ones((rows, samples), dtype=bool),
            "info": self.info,
            "monitors": SaeMonitors(
                np.ones(rows),
                np.ones(rows),
                np.ones(rows),
                np.zeros(rows),
                np.full(rows, 40, dtype=np.int64),
                None,
                None,
                None,
            ),
        }
        if self.info.outputs == "full":
            raw = _Full(
                **common,
                vector_potential=-np.cumsum(fields, axis=1),
                electric_field=fields.copy(),
                charge_dipole=fields / 2,
                current=first.copy(),
                dipole_acceleration=-fields,
                outside=zeros.copy(),
                plasma=SimpleNamespace(
                    radius=1e-9,
                    potential_slope_bound=1.0,
                    inbox_acceleration=-fields,
                    free_acceleration=zeros.copy(),
                    total_acceleration=-fields,
                    inbox_current=total.copy(),
                    free_current=zeros.copy(),
                    total_current=total,
                    outside_integral=zeros.copy(),
                    neglected_acceleration_bound=zeros.copy(),
                    neglected_current_bound=zeros.copy(),
                ),
                cut_term=0.0,
                shadow=Shadow(
                    orders=orders,
                    radius=1e-9,
                    charge_dipole=ordered(fields, 5 * fields**3),
                    current=ordered(first, third),
                    total_acceleration=ordered(-fields, -fields),
                    total_current=ordered(first, third),
                ),
            )
            if provider.failure == "nonfinite":
                raw.plasma.total_current[0, -1] = np.nan
        else:
            raw = _Compact(
                **common,
                total_current=total,
                shadow_total_current=ordered(first, third),
                shadow_orders=orders,
                radius=1e-9,
                potential_slope_bound=1.0,
                neglected_acceleration_bound=zeros.copy(),
                neglected_current_bound=zeros.copy(),
            )
            if provider.failure == "nonfinite":
                raw.total_current[0, -1] = np.nan
        provider.responses.append(raw)
        return raw


class _RichProvider(SaeProvider):
    def response_plan(  # pyright: ignore[reportIncompatibleMethodOverride] - a richer plan
        self, setup, **values
    ):
        return _RichPlan(self, **values)

    ResponsePlan = response_plan


@pytest.fixture
def sae(monkeypatch):
    fake = _RichProvider()
    monkeypatch.setattr(tdse, "_load_provider", lambda: fake)
    return fake


@pytest.fixture
def fake(monkeypatch):
    return install(monkeypatch)


def _sae_fields(rows):
    return np.arange(rows * _SAE_SAMPLES, dtype=np.float64).reshape(rows, -1) / 64


def _hacc_descriptor(basis=None, **changes):
    return TDSEDescriptor.from_response(hacc_response(basis, **changes), _GRID)


def _hacc_fields(rows):
    row = np.arange(rows, dtype=np.float64)[:, None]
    sample = np.arange(_GRID.n_to, dtype=np.float64)[None, :]
    return (((row + sample) % 7) - 3) / 4


def _bind(descriptor, n_nodes, archive=None, rows=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        return tdse.prepare_binding(
            descriptor,
            TDSEExecution(max_batch_rows=rows),
            n_nodes,
            archive=archive,
        )


def _evaluate(binding, fields, **kwargs):
    binding.reserve(1)
    try:
        return binding.evaluate(
            fields,
            z=kwargs.pop("z", 0.2),
            purpose=kwargs.pop("purpose", "predictor"),
            density=kwargs.pop("density", 2.5e24),
        )
    finally:
        binding.release_reservation()


def _same_result(first, second):
    for name in (
        "total_current",
        "first_order_current",
        "third_order_current",
        "third_order_dipole",
        "valid",
    ):
        one, other = getattr(first, name), getattr(second, name)
        if one is None:
            assert other is None, name
        else:
            assert_array_equal(one, other, strict=True)
    # The nonlinear remainder the source consumes, bitwise.
    assert_array_equal(
        first.total_current - first.first_order_current,
        second.total_current - second.first_order_current,
        strict=True,
    )


# --- item 21: round trips and full/required parity ------------------------------------


@pytest.mark.parametrize("outputs", ["required", "full"])
@pytest.mark.parametrize("model", ["tdse_native", "optmat_cubic", "tdse_shadow"])
def test_sae_records_round_trip_and_the_source_is_unchanged(
    sae, tmp_path, model, outputs
):
    descriptor = sae_descriptor(model)
    fields = _sae_fields(5)
    plain = _bind(descriptor, 5)
    expected = _evaluate(plain, fields)
    plain.close()
    calls_before = len(sae.calls)
    archive = writer(tmp_path / "a.h5", outputs=outputs)
    binding: Any = _bind(descriptor, 5, archive)
    result = _evaluate(binding, fields)
    binding.record_trial(accepted=False)
    _same_result(expected, result)
    native = "full" if outputs == "full" else descriptor.outputs
    assert {plan.info.outputs for plan in sae.created[-2:]} == {native}
    responses = sae.responses[calls_before:]
    assert [call.shape[0] for call in sae.calls[calls_before:]] == [2, 2, 1]
    record: Any = binding.metadata()["archive"]
    assert record["native_outputs"] == native and record["outputs"] == outputs
    assert record["evaluations"] == 1 and record["trials"] == 1
    assert "archive.metadata_cache" in binding.memory_ledger()
    binding.close()
    archive.close()
    schema = archive_module.record_serializer("sae", native)
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert evaluation.complete and evaluation.outcome == "returned"
        assert (evaluation.trial, evaluation.role) == (0, "rejected")
        assert evaluation.evaluated_role == "predictor"
        assert (evaluation.z, evaluation.density) == (0.2, 2.5e24)
        labels = evaluation.labels
        assert labels["input_identity"] == result.input_identity
        assert labels["configuration_identity"] == descriptor.identity
        assert labels["native_outputs"] == native
        assert labels["descriptor_outputs"] == descriptor.outputs
        assert labels["provider_time_zero"] == labels["optical_time_origin"]
        assert (labels["window_first"], labels["window_last"]) == (0, 63)
        assert [(chunk.row_start, chunk.row_stop) for chunk in evaluation.chunks] == [
            (0, 2),
            (2, 4),
            (4, 5),
        ]
        assert_array_equal(reader.load(evaluation, "prewindow_field"), fields)
        for index, raw in enumerate(responses):
            chunk = evaluation.chunks[index]
            assert chunk.serializer == schema.serializer
            assert_array_equal(
                reader.load(evaluation, "drive", chunk=index),
                fields[chunk.row_start : chunk.row_stop],
                strict=True,
            )
            stored: Any = reader.raw(evaluation, index)
            total = (
                stored["plasma"]["total_current"]
                if native == "full"
                else stored["total_current"]
            )
            expected_total = (
                raw.plasma.total_current if native == "full" else raw.total_current
            )
            assert_array_equal(total, expected_total, strict=True)
            assert stored["monitors"]["monitored_step"].dtype == np.int64
            orders = (
                stored["shadow"]["orders"]
                if native == "full"
                else stored["shadow_orders"]
            )
            assert orders == ((1,) if descriptor.shadow_order == 1 else (1, 3))
        with pytest.raises(TDSEArchiveError, match="stores no array"):
            reader.load(evaluation, "gate")


@pytest.mark.parametrize("outputs", ["required", "full"])
@pytest.mark.parametrize("basis", ["preset", "custom", "velocity"])
def test_hacc_records_round_trip_with_window_gate_and_time_map(
    fake, tmp_path, basis, outputs
):
    record = {
        "preset": preset_basis(1),
        "custom": tiny_basis(),
        "velocity": tiny_basis(gauge_radius=None),
    }[basis]
    descriptor = _hacc_descriptor(record)
    window = descriptor.atomic_window
    gate = descriptor.source_gate()
    assert window is not None and gate is not None
    fields = _hacc_fields(3)
    plain = _bind(descriptor, 3)
    expected = _evaluate(plain, fields, purpose="startup")
    plain.close()
    start = len(fake.responses)
    archive = writer(tmp_path / "a.h5", outputs=outputs)
    binding: Any = _bind(descriptor, 3, archive)
    result = _evaluate(binding, fields, purpose="startup")
    _same_result(expected, result)
    meta: Any = binding.metadata()
    closure = basis != "velocity"
    native = "full" if outputs == "full" or not closure else "total_current"
    assert meta["hacc"]["closure"] is closure
    assert meta["hacc"]["native_outputs"] == native
    binding.close()
    archive.close()
    span = slice(window.first, window.last + 1)
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert evaluation.role == "startup" and evaluation.trial == 0
        labels = evaluation.labels
        assert labels["model"] == "hacc" and labels["native_outputs"] == native
        assert (labels["window_first"], labels["window_last"]) == (
            window.first,
            window.last,
        )
        assert labels["provider_time_zero"] == window.t_s
        assert labels["provider_samples"] == window.samples
        assert labels["lattice_samples"] == _GRID.n_to
        assert_array_equal(reader.load(evaluation, "gate"), gate, strict=True)
        assert_array_equal(reader.load(evaluation, "prewindow_field"), fields)
        for index, raw in enumerate(fake.responses[start:]):
            chunk = evaluation.chunks[index]
            drive = reader.load(evaluation, "drive", chunk=index)
            # The exact gated input the provider was passed.
            assert_array_equal(
                drive,
                fields[chunk.row_start : chunk.row_stop, span] * gate[span],
                strict=True,
            )
            assert_array_equal(drive, fake.calls[start + index], strict=True)
            stored: Any = reader.raw(evaluation, index)
            assert (
                stored["serializer"]
                == archive_module.record_serializer("hacc", native).serializer
            )
            assert_array_equal(stored["total_current"], raw.total_current, strict=True)
            if native == "full":
                assert_array_equal(stored["outside"], raw.outside, strict=True)
                assert stored["shadow"]["orders"] == (1,)
                assert (stored["shadow"]["total_current"] is None) is not closure
            assert (stored["neglected_current_bound"] is None) is not closure


def test_a_custom_basis_learns_its_closure_from_its_first_full_plan(fake, tmp_path):
    archive = writer(tmp_path / "a.h5")
    descriptor = _hacc_descriptor(tiny_basis())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        binding: Any = tdse.prepare_binding(
            descriptor, TDSEExecution(), 1, archive=archive, defer_plans=True
        )
    assert binding.native_outputs == "full"
    assert binding.metadata()["hacc"]["closure"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        binding.build_plans()
    assert binding.metadata()["hacc"]["closure"] is True
    assert [arguments["outputs"] for arguments in fake.plan_arguments] == ["full"]
    binding.close()
    archive.close()


# --- item 22: failures ------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "classification"),
    [
        ("raise_driven", "native"),
        ("nonfinite", "contract"),
        ("raise_value", "interface"),
    ],
)
def test_a_raising_evaluation_records_its_input_and_exception(
    fake, tmp_path, failure, classification
):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 3, archive)
    fake.failure = failure
    fields = _hacc_fields(3)
    with pytest.raises(BackendError, match=classification):
        _evaluate(binding, fields, purpose="corrected")
    assert binding.ledger.failed_calls == 1 and len(fake.calls) == 1
    binding.close()
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert evaluation.complete and evaluation.outcome == "raised"
        assert evaluation.evaluated_role == "corrector"
        assert evaluation.error is not None
        assert evaluation.error["classification"] == classification
        assert evaluation.error["chunk"] == 0
        assert_array_equal(reader.load(evaluation, "prewindow_field"), fields)
        stored_drive = reader.load(evaluation, "drive", chunk=1)
        assert stored_drive.shape[0] == 1
        if failure == "nonfinite":
            # The raw non-finite sample the binding refused is kept as returned.
            stored = reader.load(evaluation, "total_current", chunk=0)
            assert np.isnan(stored[0, -1])
            assert_array_equal(stored, fake.responses[0].total_current, strict=True)
        else:
            assert evaluation.chunks[0].serializer is None


def _break_record(monkeypatch, *, after=0):
    """Make the archive's record writer fail on the given call."""
    original = archive_module.TDSEArchiveWriter._write_record
    seen = []

    def broken(self, group, raw, schema, where):
        seen.append(where)
        if len(seen) > after and where == "raw":
            raise OSError("synthetic write failure")
        original(self, group, raw, schema, where)

    monkeypatch.setattr(archive_module.TDSEArchiveWriter, "_write_record", broken)


def test_a_failed_write_stops_before_the_next_provider_call(
    fake, tmp_path, monkeypatch
):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 3, archive)
    _break_record(monkeypatch)
    with pytest.raises(TDSEArchiveError, match="synthetic write failure"):
        _evaluate(binding, _hacc_fields(3))
    # The first chunk was paid for and validated; the second was never called.
    assert len(fake.calls) == 1
    assert binding.ledger.attempted == 1 and binding.ledger.failed_calls == 0
    assert binding.ledger.reserved == 0
    binding.reserve(1)
    with pytest.raises(TDSEArchiveError, match="failed earlier"):
        binding.evaluate(_hacc_fields(3), z=0.3, purpose="predictor")
    assert len(fake.calls) == 1
    binding.release_reservation()
    binding.close()
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert not evaluation.complete and reader.truncated


def test_a_provider_failure_stays_primary_when_its_record_fails(
    fake, tmp_path, monkeypatch
):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 3, archive)
    fake.failure = "nonfinite"
    _break_record(monkeypatch)
    with pytest.raises(BackendError, match="contract") as error:
        _evaluate(binding, _hacc_fields(3))
    notes = "\n".join(error.value.__notes__)
    assert "could not store this result" in notes
    assert "could not record this failure" in notes
    binding.close()
    archive.close()


def test_a_native_failure_stays_primary_when_its_record_fails(
    fake, tmp_path, monkeypatch
):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 3, archive)
    fake.failure = "raise_driven"

    def broken(*args, **kwargs):
        raise TDSEArchiveError("disk gone")

    monkeypatch.setattr(archive_module.TDSEArchiveWriter, "write_failure", broken)
    with pytest.raises(BackendError, match="native") as error:
        _evaluate(binding, _hacc_fields(3))
    assert "could not record this failure: disk gone" in "\n".join(
        error.value.__notes__
    )
    monkeypatch.undo()
    binding.close()
    archive.close()


def test_an_input_that_cannot_be_written_makes_no_provider_call(
    fake, tmp_path, monkeypatch
):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 3, archive)

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(h5py.Group, "create_dataset", full)
    with pytest.raises(TDSEArchiveError, match="disk is full"):
        _evaluate(binding, _hacc_fields(3))
    monkeypatch.undo()
    assert fake.calls == []
    assert binding.ledger.attempted == 0
    binding.close()
    archive.close()


def test_an_outcome_that_cannot_be_written_stops_the_run(fake, tmp_path, monkeypatch):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 1, archive)
    original = archive_module.TDSEArchiveWriter.finish_evaluation

    def broken(self, evaluation, outcome):
        raise archive_module.TDSEArchiveError("synthetic outcome failure")

    monkeypatch.setattr(archive_module.TDSEArchiveWriter, "finish_evaluation", broken)
    with pytest.raises(TDSEArchiveError, match="synthetic outcome failure"):
        _evaluate(binding, _hacc_fields(1))
    monkeypatch.setattr(archive_module.TDSEArchiveWriter, "finish_evaluation", original)
    assert binding.ledger.reserved == 0
    binding.close()
    archive.close()


def test_a_misused_reservation_is_recorded_against_its_evaluation(fake, tmp_path):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 1, archive)
    binding.reserve(1)
    binding.release_reservation()
    with pytest.raises(BackendError, match="complete unused reservation"):
        binding.evaluate(_hacc_fields(1), z=0.0, purpose="predictor")
    assert fake.calls == []
    binding.close()
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert evaluation.outcome == "raised"
        assert evaluation.error is not None
        assert evaluation.error["classification"] == "contract"
        assert evaluation.error["chunk"] == -1


# --- trial labels, density and the diagnostic -----------------------------------------


def test_trials_are_numbered_by_reservation_and_need_one(fake, tmp_path):
    archive = writer(tmp_path / "a.h5")
    binding: Any = _bind(_hacc_descriptor(preset_basis(1)), 1, archive)
    with pytest.raises(BackendError, match="reserved trial"):
        binding.record_trial(accepted=True)
    fields = _hacc_fields(1)
    _evaluate(binding, fields, purpose="startup")
    binding.record_trial(accepted=True)
    _evaluate(binding, fields)
    binding.record_trial(accepted=False)
    _evaluate(binding, fields * 2)
    # A diagnostic of exactly the last field is served from the cache: no call, no
    # archived evaluation.
    binding.reserve(1)
    binding.evaluate(fields * 2, z=0.2, purpose="diagnostic")
    binding.release_reservation()
    binding.record_trial(accepted=True)
    with pytest.raises(BackendError, match="density"):
        _evaluate(binding, fields, density=-1.0)
    binding.close()
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        roles = [(item.trial, item.role) for item in reader.evaluations()]
        assert roles == [(0, "startup"), (1, "rejected"), (2, "predictor")]
        assert reader.sessions[0].trials == {
            0: "accepted",
            1: "rejected",
            3: "accepted",
        }


def test_the_consistency_diagnostic_is_archived_with_its_own_lattice(fake, tmp_path):
    archive = writer(tmp_path / "a.h5")
    descriptor = _hacc_descriptor(preset_basis(1))
    binding: Any = tdse.prepare_binding(
        descriptor, TDSEExecution(), 1, archive=archive, defer_plans=True
    )
    spacing, field = protocol_field(800e-9)
    total, _ = binding.diagnostic_currents(field[None, :], spacing)
    fake.failure = "raise_driven"
    with pytest.raises(BackendError, match="consistency diagnostic failed"):
        binding.diagnostic_currents(field[None, :], spacing)
    binding.close()
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        good, failed = reader.evaluations()
        assert good.role == "diagnostic" and good.trial is None and good.z is None
        assert good.labels["lattice"] == "consistency_protocol"
        assert good.labels["provider_time_zero"] == 0.0
        assert good.labels["configuration_identity"] == descriptor.identity
        assert "optical_time_origin" not in good.labels
        assert_array_equal(
            reader.load(good, "drive", chunk=0), field[None, :], strict=True
        )
        assert_array_equal(
            reader.load(good, "total_current", chunk=0)[0], total, strict=True
        )
        assert failed.outcome == "raised" and failed.error is not None
        assert failed.error["classification"] == "native"


def test_a_diagnostic_write_failure_is_raised_or_kept_secondary(
    fake, tmp_path, monkeypatch
):
    archive = writer(tmp_path / "a.h5")
    binding: Any = tdse.prepare_binding(
        _hacc_descriptor(preset_basis(1)),
        TDSEExecution(),
        1,
        archive=archive,
        defer_plans=True,
    )
    spacing, field = protocol_field(800e-9)
    _break_record(monkeypatch)
    with pytest.raises(TDSEArchiveError, match="synthetic write failure"):
        binding.diagnostic_currents(field[None, :], spacing)
    binding.close()
    archive.close()
    # The record writer stays broken for a second binding whose provider also
    # returns a malformed result.
    other = writer(tmp_path / "b.h5")
    binding = tdse.prepare_binding(
        _hacc_descriptor(preset_basis(1)),
        TDSEExecution(),
        1,
        archive=other,
        defer_plans=True,
    )
    fake.failure = "row"
    with pytest.raises(BackendError, match="consistency diagnostic failed") as error:
        binding.diagnostic_currents(field[None, :], spacing)
    notes = "\n".join(error.value.__notes__)
    assert "could not store" in notes and "could not record" in notes
    binding.close()
    other.close()


def test_the_standalone_check_archives_its_one_call_and_a_hit_none(
    fake, tmp_path, monkeypatch
):
    from tests.hacc_cases import alpha_law  # noqa: PLC0415

    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))
    fake.response = alpha_law(1.0)
    response = hacc_response(preset_basis(1))
    result = check_consistency(
        response,
        TDSEExecution(max_atomic_calls=4),
        reference_wavelength=800e-9,
        tdse_archive=TDSEArchive(tmp_path / "check.h5", outputs="required"),
    )
    assert result.outcome == "passed"
    hit = check_consistency(
        response,
        TDSEExecution(max_atomic_calls=4),
        reference_wavelength=800e-9,
        tdse_archive=TDSEArchive(tmp_path / "hit.h5"),
    )
    assert hit.from_cache
    with open_tdse_archive(tmp_path / "check.h5") as reader:
        (session,) = reader.sessions
        assert session.kind == "consistency_check" and session.optical_reference is None
        (evaluation,) = reader.evaluations()
        assert evaluation.role == "diagnostic" and evaluation.complete
        assert evaluation.labels["native_outputs"] == "total_current"
        assert "configuration_identity" not in evaluation.labels
    with open_tdse_archive(tmp_path / "hit.h5") as reader:
        assert list(reader.evaluations()) == []
    with pytest.raises(TDSEArchiveError, match="never overwritten"):
        check_consistency(
            response,
            TDSEExecution(max_atomic_calls=4),
            reference_wavelength=800e-9,
            refresh=True,
            tdse_archive=TDSEArchive(tmp_path / "check.h5"),
        )
    with pytest.raises(BackendError, match="TDSEArchive"):
        inspect.unwrap(check_consistency)(
            response,
            TDSEExecution(),
            reference_wavelength=800e-9,
            tdse_archive="check.h5",  # pyright: ignore[reportArgumentType]
        )


# --- memory and arguments ------------------------------------------------------------


def test_the_archive_is_admitted_in_the_memory_estimate(sae):
    descriptor = sae_descriptor("tdse_native")
    execution = TDSEExecution(max_batch_rows=2)
    plain = tdse.estimate_binding_memory(descriptor, execution, 5)
    required = tdse.estimate_binding_memory(
        descriptor, execution, 5, archive="required"
    )
    full = tdse.estimate_binding_memory(descriptor, execution, 5, archive="full")
    assert "archive.metadata_cache" not in plain
    assert required["archive.conversion_buffer"] == 8 * 3 * 2 * descriptor.samples
    assert required["estimate.host_histories"] == plain["estimate.host_histories"]
    assert full["estimate.host_histories"] > required["estimate.host_histories"]


def test_prepare_binding_refuses_an_archive_that_is_not_a_writer(sae):
    from inspect import unwrap  # noqa: PLC0415

    with pytest.raises(BackendError, match="TDSEArchiveWriter"):
        unwrap(tdse.prepare_binding)(
            sae_descriptor("tdse_native"), TDSEExecution(), 1, archive="a.h5"
        )


def test_an_archive_without_writes_records_its_reference(sae, tmp_path):
    archive = writer(tmp_path / "a.h5", outputs="required")
    binding: Any = _bind(sae_descriptor("tdse_native"), 1, archive)
    record = json.loads(json.dumps(binding.metadata(), allow_nan=False))["archive"]
    assert record["run_uuid"] == archive.run_uuid == "run-1"
    assert record["evaluations"] == 0 and record["native_outputs"] == "total_current"
    binding.close()
    archive.close()


def test_a_session_preparation_records_its_archive(fake, tmp_path):
    archive = writer(tmp_path / "a.h5")
    session = tdse.DiagnosticSession(preset_basis(1), TDSEExecution(), archive=archive)
    meta: Any = session.metadata()
    assert meta["archive"]["outputs"] == "full"
    assert "archive.metadata_cache" in meta["memory_bytes"]
    session.close()
    archive.close()
