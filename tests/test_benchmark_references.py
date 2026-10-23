# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify reference invalidation and retained diagnostics on synthetic fields."""

import dataclasses
import os
import sys

import numpy as np
import pytest
from benchmarks import collection, system, worker
from benchmarks.builders import fixed_schedule
from benchmarks.qualification import (
    Qualification,
    envelope_failures,
    recheck_row,
    reference_lineage,
    reviewed_reference,
)
from benchmarks.records import SuiteError, atomic_record, canonical, file_hash
from benchmarks.references import (
    ADVANCED_ENVELOPES,
    EXERCISE_ENVELOPES,
    MODAL_ENVELOPES,
    ReferenceReceipt,
    approved_envelope,
    check,
    field_errors,
    measurement_identity,
    reference_path,
    spectral_errors,
)
from benchmarks.registry import ADAPTIVE, CASES, FIXED, KERR

from solprop.constants import C
from solprop.output import MemoryOutput


def test_reset_evidence_retains_undefined_statistics_without_nonfinite_json(
    monkeypatch,
):
    expected = {
        "stats/omega0": np.array([1.0, np.nan]),
        "stats/fwhm_t": np.array([np.nan]),
    }
    monkeypatch.setattr(worker, "reset_columns", lambda _: expected)
    witness = worker.ResetWitness(np.ones((1, 1), dtype=complex), expected)
    output = MemoryOutput()
    result = worker.reset_differences(witness, output)
    assert result["stats/omega0"]["same_undefined"]
    assert result["stats/omega0"]["compared_finite_count"] == 1
    assert result["stats/fwhm_t"]["absolute_l2"] is None
    canonical(result)
    monkeypatch.setattr(
        worker,
        "reset_columns",
        lambda _: {
            "stats/omega0": np.array([1.0, 2.0]),
            "stats/fwhm_t": np.array([np.nan]),
        },
    )
    difference = worker.reset_differences(witness, output)["stats/omega0"]
    assert not difference["same_undefined"]
    for name, value in (("stats/energy", np.inf), ("step_history/norm", np.nan)):
        bad = {name: np.array([value])}
        monkeypatch.setattr(worker, "reset_columns", lambda _, values=bad: values)
        with pytest.raises(SuiteError, match="non-finite reset"):
            worker.reset_differences(worker.ResetWitness(witness.fields, bad), output)


@pytest.fixture
def receipt(tmp_path):
    configuration = {"case": "smoke", "case_hash": "test-case", "track": "adaptive-rk"}
    fields = np.ones((2, 1, 1, 3), dtype=complex)
    np.save(tmp_path / "fields.npy", fields)
    np.save(tmp_path / "band.npy", np.array([True, True, False]))
    atomic_record(tmp_path / "evidence.json", {"status": "synthetic-test"})
    record = ReferenceReceipt(
        revision=1,
        case_hash=configuration["case_hash"],
        measurement_hash=measurement_identity(configuration),
        fields_file="fields.npy",
        fields_hash=file_hash(tmp_path / "fields.npy"),
        band_file="band.npy",
        band_hash=file_hash(tmp_path / "band.npy"),
        accuracy_bound=1e-5,
        refinement_floor=1e-9,
        evidence_file="evidence.json",
        evidence_hash=file_hash(tmp_path / "evidence.json"),
        protocol="synthetic-test",
        qualified_for_adoption=True,
    )
    path = tmp_path / "reference.json"
    atomic_record(path, dataclasses.asdict(record))
    return configuration, fields, record, path


def test_reference_envelope_and_identity_invalidation(receipt):
    configuration, fields, record, path = receipt
    # Multiplicative perturbations have exactly their magnitude as relative L2
    # error, up to the rounding in the three-bin norm.
    assert check(fields * (1 + 1e-6), configuration, path)["status"] == "passed"
    failed = check(fields * (1 + 1e-4), configuration, path)
    assert failed["status"] == "invalid"
    assert not failed["qualified_for_adoption"]
    with pytest.raises(SuiteError, match="identity"):
        check(fields, {**configuration, "case_hash": "revised"}, path)
    atomic_record(path, dataclasses.asdict(dataclasses.replace(record, revision=2)))
    with pytest.raises(SuiteError, match="revision"):
        check(fields, configuration, path)


def test_later_samples_reuse_only_validated_arm_and_reference(receipt):
    configuration, fields, record, path = receipt
    validated = worker.accuracy(fields, configuration, {"reference": str(path)})
    sample = {
        "configuration_id": "configuration",
        "arm": "candidate",
        "sample_id": "first",
        "status": "ok",
        "accuracy": validated,
    }
    previous = collection.prior_validation(
        [sample], "configuration", "candidate", str(path)
    )
    assert previous is not None
    assert previous["validation_sample_id"] == "first"
    assert "plane_errors" not in previous  # These were not measured again.
    request = {"prior_validation": previous}
    assert worker.accuracy(fields, configuration, request)["status"] == "passed"
    assert worker.accuracy(fields[:, :0], configuration, request)["status"] == "invalid"
    assert (
        worker.accuracy(fields * np.nan, configuration, request)["status"] == "invalid"
    )
    for changed in (
        {"arm": "baseline"},
        {"configuration_id": "different"},
        {"status": "invalid"},
        {"accuracy_overridden": True},
        {"accuracy": {"status": "not-qualified"}},
    ):
        assert (
            collection.prior_validation(
                [sample | changed], "configuration", "candidate", str(path)
            )
            is None
        )
    repeated = sample | {"sample_id": "second", "accuracy": previous}
    cached = collection.prior_validation(
        [repeated], "configuration", "candidate", str(path)
    )
    assert cached is not None
    assert cached["validation_sample_id"] == "first"
    atomic_record(
        path, dataclasses.asdict(dataclasses.replace(record, policy_revision=2))
    )
    assert (
        collection.prior_validation([sample], "configuration", "candidate", str(path))
        is None
    )
    path.write_text("broken", encoding="utf-8")
    assert (
        collection.prior_validation([sample], "configuration", "candidate", str(path))
        is None
    )
    assert (
        collection.prior_validation([sample], "configuration", "candidate", None)
        is None
    )


@pytest.mark.parametrize("artifact", ["fields", "band", "evidence"])
def test_reference_artifact_hashes_are_checked(receipt, artifact):
    configuration, fields, record, path = receipt
    (path.parent / getattr(record, f"{artifact}_file")).write_bytes(b"changed")
    with pytest.raises(SuiteError, match="hash mismatch"):
        check(fields, configuration, path)


def test_reference_path_cannot_escape_bundle(tmp_path):
    with pytest.raises(SuiteError, match="escapes"):
        reference_path(
            {"fields_file": "../outside.npy", "fields_hash": "unused"},
            "fields",
            tmp_path,
        )


def test_field_metric_refuses_invalid_or_zero_reference():
    truth = np.ones((1, 2), complex)
    for fields, reference in (
        (truth, truth[:, :1]),
        (truth, truth * 0),
        (truth * np.nan, truth),
    ):
        with pytest.raises(SuiteError):
            field_errors(fields, reference)


def test_spectral_diagnostics_retain_weak_bins_uv_and_parseval_energy():
    truth = np.array([[[[1.0, 1e-2, 1e-3]]]], dtype=complex)
    omega = 2 * np.pi * C / np.array([100e-9, 125e-9, 200e-9])
    epsilon = 1e-6
    measured = spectral_errors(truth * (1 + epsilon), truth, omega, "rdw")
    assert measured["weak"]["selected_bins"] == 2
    assert measured["uv"]["selected_bins"] == 2
    # These synthetic scalings give analytic norm and energy differences;
    # 8 epsilon_machine bounds the small sums and subtraction of unit energies.
    rounding = 8 * np.finfo(float).eps
    assert measured["weak"]["value"] == pytest.approx(epsilon, abs=rounding)
    assert measured["uv"]["value"] == pytest.approx(epsilon, abs=rounding)
    assert measured["energy"]["value"] == pytest.approx(
        2 * epsilon + epsilon**2, abs=rounding
    )
    missing = spectral_errors(truth, truth, omega / 10, "rdw")
    assert missing["uv"]["value"] is None
    assert missing["uv"]["reason"]
    with pytest.raises(SuiteError, match="contract"):
        spectral_errors(truth, truth, omega[:1], "ctc")
    with pytest.raises(SuiteError, match="zero-norm"):
        spectral_errors(truth * 0, truth * 0, omega, "ctc")


def test_refinement_floor_includes_retained_bands():
    errors = {"coarse": 1e-10, "matched": 1e-7, "candidate": 1e-7}
    metrics = {"coarse": {"uv": {"value": 2e-5}}, "candidate": {"uv": {"value": 3e-5}}}
    failures = envelope_failures(errors, metrics, 1e-4, adaptive=True)
    assert len(failures) == 1
    assert "coarse uv" in failures[0]
    assert not envelope_failures(errors, metrics, 1e-4, adaptive=False)


def test_approved_limits_apply_only_to_reviewed_measurements():
    limits = (("rdw", 1e-4), ("ctc", 1e-5), ("radial_filament_256", 1e-4))
    for case, bound in (
        *limits,
        *(
            (item.case, 1e-4)
            for item in (*MODAL_ENVELOPES, *ADVANCED_ENVELOPES, *EXERCISE_ENVELOPES)
        ),
    ):
        for track in ADAPTIVE:
            configuration = CASES[case].measurement(track)
            policy = approved_envelope(configuration)
            assert policy is not None
            assert policy.bound == bound
            assert approved_envelope({**configuration, "stop": 0.5}) is None
            assert approved_envelope({**configuration, "revision": 2}) is None
        if FIXED in CASES[case].tracks:
            assert approved_envelope(CASES[case].measurement(FIXED)) is None


def test_energy_remains_diagnostic_during_qualification():
    errors = {"coarse": 1e-10, "matched": 1e-7, "candidate": 1e-7}
    metrics = {"candidate": {"energy": {"value": 0.1, "role": "diagnostic"}}}
    assert not envelope_failures(errors, metrics, 1e-4, adaptive=True)


@pytest.mark.parametrize(
    "name,track,count",
    [
        ("rdw", FIXED, 10),
        ("ctc", FIXED, 10),
        ("radial_filament_256", FIXED, 10),
        ("kerr_beam_384", KERR, 11),
    ],
)
def test_fixed_schedule_has_declared_entrance_count_without_a_zero_tail(
    name, track, count
):
    case = CASES[name]
    configuration = case.measurement(track)
    schedule = fixed_schedule(case, configuration)
    # The saved endpoint is a sum of the prescribed steps: a handful of additions
    # permits 8 eps of relative rounding, never a tolerance-sized extra step.
    rounding = 8 * np.finfo(float).eps
    assert schedule.edges[count] == pytest.approx(configuration["stop"], rel=rounding)
    assert schedule.z_end == pytest.approx(case.length, rel=rounding)
    if track == KERR:
        assert schedule.n_steps == 11
        assert schedule.edges[1] == pytest.approx(configuration["start"], rel=rounding)


def test_reference_revisions_preserve_prior_evidence_and_advance_policy(
    receipt, tmp_path
):
    configuration, _, record, path = receipt
    options = Qualification(
        tmp_path / "replacement",
        60,
        None,
        2e-5,
        supersedes=path.parent,
        revision_note="Reviewed wider accuracy margin",
    )
    # The fixture alias must match the named bundle receipt used by the CLI.
    named = path.with_name("smoke-adaptive-rk.json")
    named.write_bytes(path.read_bytes())
    original = named.read_bytes()
    lineage = reference_lineage(configuration, options)
    assert lineage["reference_revision"] == 2
    assert lineage["policy_revision"] == 2
    assert lineage["supersedes"]
    assert named.read_bytes() == original
    with pytest.raises(SuiteError, match="new policy revision"):
        reference_lineage(
            configuration, dataclasses.replace(options, policy_revision=1)
        )
    with pytest.raises(SuiteError, match="revision-note"):
        reference_lineage(
            configuration, dataclasses.replace(options, revision_note=None)
        )
    with pytest.raises(SuiteError, match="positive"):
        reference_lineage(
            configuration, dataclasses.replace(options, reference_revision=0)
        )
    with pytest.raises(SuiteError, match="different measurement"):
        reference_lineage({**configuration, "stop": 1.0}, options)
    assert record.accuracy_bound == 1e-5


@pytest.mark.parametrize("reviewed", [False, True])
def test_recheck_changes_policy_without_propagation(receipt, tmp_path, reviewed):
    configuration, fields, record, path = receipt
    evidence_path = path.parent / record.evidence_file
    atomic_record(
        evidence_path,
        {
            "errors_against_tight": {
                "coarse": 1e-10,
                "matched": 1e-7,
                "candidate": 1e-7,
            },
            "spectral_metrics_against_tight": {},
            "records": {
                name: {
                    "status": "ok",
                    "reset_evidence": {
                        "plane_errors": [1e-12],
                        "history_statistics_noise": {"same_columns": True},
                        "fresh_counts": {"accepted": 10},
                        "reset_counts": [{"accepted": 10}],
                    },
                }
                for name in ("tight", "coarse", "matched", "candidate")
            },
        },
    )
    updated = dataclasses.replace(record, evidence_hash=file_hash(evidence_path))
    named = path.with_name("smoke-adaptive-rk.json")
    atomic_record(named, dataclasses.asdict(updated))
    before = named.read_bytes()
    options = Qualification(
        tmp_path / "rechecked",
        60,
        None,
        2e-5,
        revision_note="Margin review",
        reviewed=reviewed,
    )
    report = recheck_row({"configuration": configuration}, path.parent, options)
    assert report["status"] == "envelope-passed"
    result = check(fields, configuration, options.output / named.name)
    assert result["status"] == "passed"
    assert result["reference_revision"] == 2
    assert result["policy_revision"] == 2
    assert result["qualified_for_adoption"] is reviewed
    assert named.read_bytes() == before


def test_reviewed_reference_requires_retained_reset_evidence():
    reset = {
        "plane_errors": [1e-12],
        "history_statistics_noise": {"same_columns": True},
        "fresh_counts": {"accepted": 10},
        "reset_counts": [{"accepted": 10}],
    }
    records = {
        name: {"status": "ok", "reset_evidence": reset}
        for name in ("tight", "coarse", "matched", "candidate")
    }
    assert reviewed_reference({"records": records}, 1e-4)
    with pytest.raises(SuiteError, match="positive accuracy"):
        reviewed_reference({"records": records}, None)
    for bad in (
        {"status": "failed"},
        {"status": "ok", "reset_evidence": {**reset, "plane_errors": [1e-3]}},
        {"status": "ok", "reset_evidence": {**reset, "reset_counts": []}},
    ):
        with pytest.raises(SuiteError, match="candidate lacks"):
            reviewed_reference({"records": {**records, "candidate": bad}}, 1e-4)


@pytest.mark.medium
def test_real_worker_accuracy_override_retains_failed_check_and_timing(tmp_path):
    configuration = {
        **CASES["smoke"].measurement(ADAPTIVE[1]),
        "backend": "numpy_ref",
        "threads": 1,
    }
    environment = system.thread_environment(1, "budget", dict(os.environ))
    original = tmp_path / "original.npy"
    initial = collection.launch(
        sys.executable,
        {"configuration": configuration, "save_fields": str(original)},
        environment,
        tmp_path / "initial",
        (60, 2 * 2**30),
        system.eligible_cpus(),
    )
    assert initial["status"] == "ok", initial
    # A deliberately incorrect reference creates a large finite mismatch, without
    # modifying the simulation or relying on round-off at the acceptance boundary.
    truth_path = tmp_path / "incorrect.npy"
    np.save(truth_path, 2 * np.load(original, allow_pickle=False))
    evidence_path = tmp_path / "evidence.json"
    atomic_record(evidence_path, {"test": "intentional finite reference mismatch"})
    receipt = ReferenceReceipt(
        1,
        configuration["case_hash"],
        measurement_identity(configuration),
        truth_path.name,
        file_hash(truth_path),
        original.name + ".band.npy",
        file_hash(tmp_path / (original.name + ".band.npy")),
        1e-5,
        None,
        evidence_path.name,
        file_hash(evidence_path),
        "synthetic-test",
    )
    path = tmp_path / "reference.json"
    atomic_record(path, dataclasses.asdict(receipt))
    for override in (False, True):
        result = collection.launch(
            sys.executable,
            {
                "configuration": configuration,
                "reference": str(path),
                "allow_accuracy_failure": override,
            },
            environment,
            tmp_path / str(override),
            (60, 2 * 2**30),
            system.eligible_cpus(),
        )
        assert result["status"] == ("ok" if override else "invalid")
        assert result["returncode"] == (0 if override else 1)
        assert result["accuracy"]["status"] == "invalid"
        assert result["accuracy_overridden"] is override
        assert result["seconds"] > 0
        assert result["measurement_peak_bytes"] > 0
