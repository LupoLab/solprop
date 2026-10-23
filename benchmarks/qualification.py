# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare independently retained reference, refinement and reset evidence."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shutil
import sys
import uuid
from pathlib import Path

from benchmarks.collection import launch
from benchmarks.records import (
    Record,
    SuiteError,
    atomic_record,
    file_hash,
    identity,
    now,
    read_record,
    run_lock,
)
from benchmarks.references import (
    ReferenceReceipt,
    approved_envelope,
    field_errors,
    measurement_identity,
    reference_path,
    spectral_errors,
)
from benchmarks.registry import ADAPTIVE
from benchmarks.system import admission, thread_environment

__all__ = [
    "Qualification",
    "envelope_failures",
    "qualify",
    "qualify_row",
    "recheck_row",
    "reference_lineage",
    "reviewed_reference",
    "run_reference",
]


@dataclasses.dataclass(frozen=True)
class Qualification:
    """Declare the bounded qualification run and optional reviewed envelope."""

    output: Path
    timeout: float
    memory_limit: int | None
    accuracy_bound: float | None
    reference_revision: int | None = None
    policy_revision: int | None = None
    supersedes: Path | None = None
    revision_note: str | None = None
    reviewed: bool = False


def reviewed_reference(evidence: Record, bound: float | None) -> bool:
    """Verify the evidence required by an explicit retained-reference review.

    The caller confirms review of the history, statistics and work-count
    differences. These diagnostics have case-dependent conditioning; this helper
    does not invent a universal rounding tolerance or demand identical histories.
    The existing field ceiling also applies to the reset field witness.
    """
    if bound is None or not math.isfinite(bound) or bound <= 0:
        raise SuiteError("reviewed comparison requires a positive accuracy bound")
    records = evidence.get("records", {})
    for name in ("tight", "coarse", "matched", "candidate"):
        record = records.get(name, {})
        reset = record.get("reset_evidence") or {}
        errors = reset.get("plane_errors", [])
        if (
            record.get("status") != "ok"
            or not errors
            or not all(math.isfinite(value) and 0 <= value <= bound for value in errors)
            or not reset.get("history_statistics_noise")
            or not reset.get("fresh_counts")
            or not reset.get("reset_counts")
        ):
            raise SuiteError(
                f"{name} lacks successful reset/reference evidence "
                "within the field ceiling"
            )
    return True


def reference_lineage(configuration: Record, options: Qualification) -> Record:
    """Version a replacement without changing the workload or old receipts."""
    if any(
        value is not None and value < 1
        for value in (options.reference_revision, options.policy_revision)
    ):
        raise SuiteError("reference and policy revisions must be positive")
    if options.accuracy_bound is not None and (
        not math.isfinite(options.accuracy_bound) or options.accuracy_bound <= 0
    ):
        raise SuiteError("accuracy bound must be positive and finite")
    previous = None
    if options.supersedes is not None:
        previous = read_record(
            options.supersedes
            / f"{configuration['case']}-{configuration['track']}.json"
        )
        if previous["measurement_hash"] != measurement_identity(configuration):
            raise SuiteError("superseded reference belongs to a different measurement")
        if not options.revision_note or not options.revision_note.strip():
            raise SuiteError("superseding a reference requires --revision-note")
    old_reference = previous.get("reference_revision", 1) if previous else 0
    old_policy = previous.get("policy_revision", 1) if previous else 0
    changed_bound = (
        previous is not None
        and previous.get("accuracy_bound") != options.accuracy_bound
    )
    reference_revision = options.reference_revision or old_reference + 1
    approved = approved_envelope(configuration)
    approved_revision = (
        approved.revision
        if approved and approved.bound == options.accuracy_bound
        else 1
    )
    policy_revision = options.policy_revision or max(
        approved_revision, old_policy + int(changed_bound)
    )
    if reference_revision <= old_reference or policy_revision < max(1, old_policy):
        raise SuiteError(
            "reference revisions must advance and policy revisions cannot decrease"
        )
    if changed_bound and policy_revision <= old_policy:
        raise SuiteError("a changed accuracy bound requires a new policy revision")
    return {
        "reference_revision": reference_revision,
        "policy_revision": policy_revision,
        "supersedes": identity(previous) if previous else None,
        "revision_note": options.revision_note,
    }


def recheck_row(row: Record, source: Path, options: Qualification) -> Record:
    """Apply a revised bound to retained evidence without running a simulation."""
    configuration = row["configuration"]
    prefix = f"{configuration['case']}-{configuration['track']}"
    previous = read_record(source / f"{prefix}.json")
    envelope = approved_envelope(configuration)
    bound = options.accuracy_bound
    if bound is None:
        bound = envelope.bound if envelope else previous.get("accuracy_bound")
    options = dataclasses.replace(options, accuracy_bound=bound, supersedes=source)
    lineage = reference_lineage(configuration, options)
    evidence = read_record(reference_path(previous, "evidence", source))
    metrics = evidence.get("spectral_metrics_against_tight", {})
    if configuration["case"] in ("rdw", "ctc") and not metrics:
        raise SuiteError(
            "retained evidence lacks required band metrics; prepare a new reference"
        )
    failures = (
        []
        if bound is None
        else envelope_failures(
            evidence["errors_against_tight"],
            metrics,
            bound,
            adaptive=configuration["track"] in ADAPTIVE,
        )
    )
    reviewed = {
        **evidence,
        "created": now(),
        "rechecked_from": identity(previous),
        "lineage": lineage,
        "accuracy_bound": bound,
        "bound_source": "explicit reviewed recheck" if options.accuracy_bound else None,
        "status": "envelope-failed"
        if failures
        else "envelope-passed"
        if bound
        else "measured-unreviewed",
        "failures": failures,
    }
    evidence_path = options.output / f"{prefix}-evidence.json"
    atomic_record(evidence_path, reviewed, immutable=True)
    if failures:
        raise SuiteError("; ".join(failures) + "; recheck evidence retained")
    replacement = {
        **previous,
        **lineage,
        "accuracy_bound": bound,
        "qualified_for_adoption": reviewed_reference(evidence, bound)
        if options.reviewed
        else False,
    }
    for artifact in ("fields", "band", "omega"):
        if previous.get(f"{artifact}_file") is None:
            continue
        original = reference_path(previous, artifact, source)
        target = options.output / "arrays" / f"{prefix}-{artifact}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        replacement[f"{artifact}_file"] = str(target.relative_to(options.output))
        replacement[f"{artifact}_hash"] = file_hash(target)
    replacement.update(
        evidence_file=evidence_path.name, evidence_hash=file_hash(evidence_path)
    )
    atomic_record(options.output / f"{prefix}.json", replacement, immutable=True)
    return reviewed


def envelope_failures(
    errors: Record, metrics: Record, bound: float, *, adaptive: bool
) -> list[str]:
    """Check a reviewed envelope and all retained refinement diagnostics."""
    failures = []
    if adaptive and bound < 10 * errors["coarse"]:
        failures.append("field bound is below ten times the reference refinement floor")
    if any(errors[name] > bound for name in ("matched", "candidate")):
        failures.append("reviewed field envelope is not met")
    for suffix, measured in metrics.items():
        for name, item in measured.items():
            if name == "energy":
                continue
            limit = bound
            if suffix == "coarse":
                if not adaptive:
                    continue
                limit /= 10
            if item["value"] is None or item["value"] > limit:
                failures.append(
                    f"{suffix} {name} exceeds its reviewed envelope/refinement limit"
                )
    return failures


def run_reference(
    configuration: Record,
    row: Record,
    options: Qualification,
    profile: Record,
    name: str,
) -> Record:
    """Run one separately admitted reference or matched-method check."""
    directory = options.output / name
    directory.mkdir(parents=True, exist_ok=True)
    admit = admission(row["estimate_bytes"], options.memory_limit)
    if not admit["admitted"]:
        record = {"status": "refused", "admission": admit}
    else:
        fields_path = directory / "fields.npy"
        record = launch(
            sys.executable,
            {
                "configuration": configuration,
                "repetitions": 1,
                "verify_reset": True,
                "save_fields": str(fields_path.resolve()),
            },
            thread_environment(
                configuration["threads"], profile["policy"], dict(os.environ)
            ),
            directory / "worker",
            (options.timeout, admit["limit_bytes"]),
            tuple(profile["allocation"]),
        )
    atomic_record(directory / "result.json", record, immutable=True)
    if record["status"] != "ok":
        raise SuiteError(
            f"qualification {name}: {record['status']}: {record.get('error', admit)}"
        )
    return record


def qualify_row(row: Record, plan: Record, options: Qualification) -> Record:
    """Preserve measured evidence; never manufacture an accuracy envelope."""
    import numpy as np  # noqa: PLC0415 - explicit qualification only

    configuration = row["configuration"]
    envelope = approved_envelope(configuration)
    if options.accuracy_bound is None and envelope is not None:
        options = dataclasses.replace(options, accuracy_bound=envelope.bound)
    lineage = reference_lineage(configuration, options)
    prefix = f"{configuration['case']}-{configuration['track']}"
    oracle = {**configuration, "backend": "numpy_ref", "threads": 1}
    adaptive = configuration["track"] in ADAPTIVE
    tight = (
        {**oracle, "track": ADAPTIVE[1], "reference_rtol": 1e-10}
        if adaptive
        else oracle
    )
    coarse = {**tight, "reference_rtol": 1e-9} if adaptive else oracle
    records = {}
    for suffix, settings in (
        ("tight", tight),
        ("coarse", coarse),
        ("matched", oracle),
        ("candidate", configuration),
    ):
        print(f"qualify {prefix}: {suffix}", flush=True)
        records[suffix] = run_reference(
            settings, row, options, plan["profile"], f"{prefix}-{suffix}"
        )
    paths = {
        suffix: options.output / f"{prefix}-{suffix}" / "fields.npy"
        for suffix in records
    }
    fields = {
        suffix: np.load(path, allow_pickle=False, mmap_mode="r")
        for suffix, path in paths.items()
    }
    band = np.load(f"{paths['tight']}.band.npy", allow_pickle=False)
    errors = {
        suffix: max(field_errors(value[..., band], fields["tight"][..., band]))
        for suffix, value in fields.items()
        if suffix != "tight"
    }
    omega_path = Path(f"{paths['tight']}.omega.npy")
    omega = np.load(omega_path, allow_pickle=False)
    metrics = {
        suffix: spectral_errors(value, fields["tight"], omega, configuration["case"])
        for suffix, value in fields.items()
        if suffix != "tight"
    }
    parity = max(
        field_errors(fields["candidate"][..., band], fields["matched"][..., band])
    )
    evidence = {
        "created": now(),
        "configuration": configuration,
        "records": records,
        "errors_against_tight": errors,
        "spectral_metrics_against_tight": metrics,
        "matched_method_difference": parity,
        "bound_source": dataclasses.asdict(envelope)
        if envelope is not None and envelope.bound == options.accuracy_bound
        else "explicit reviewed CLI envelope"
        if options.accuracy_bound
        else None,
        "status": "measured-unreviewed",
        "lineage": lineage,
    }
    bound = options.accuracy_bound
    failures = []
    if bound is not None:
        if not np.isfinite(bound) or bound <= 0:
            raise SuiteError("accuracy bound must be positive and finite")
        failures = envelope_failures(errors, metrics, bound, adaptive=adaptive)
        evidence["status"] = "envelope-failed" if failures else "envelope-passed"
    evidence["failures"] = failures
    evidence_path = options.output / f"{prefix}-evidence.json"
    atomic_record(evidence_path, evidence, immutable=True)
    if failures:
        raise SuiteError("; ".join(failures) + "; evidence retained")
    receipt = ReferenceReceipt(
        1,
        configuration["case_hash"],
        measurement_identity(configuration),
        str(paths["tight"].relative_to(options.output)),
        file_hash(paths["tight"]),
        str(paths["tight"].relative_to(options.output)) + ".band.npy",
        file_hash(Path(f"{paths['tight']}.band.npy")),
        bound,
        errors["coarse"] if adaptive else None,
        evidence_path.name,
        file_hash(evidence_path),
        "tight-rk-1e-10-refined-1e-9" if adaptive else "matched-fixed-numpy",
        omega_file=str(omega_path.relative_to(options.output)),
        omega_hash=file_hash(omega_path),
        **lineage,
    )
    atomic_record(
        options.output / f"{prefix}.json", dataclasses.asdict(receipt), immutable=True
    )
    return evidence


def qualify(plan: Record, args: argparse.Namespace) -> int:
    """Run explicit qualification separately from ordinary measurement bundles."""
    output = args.out or args.references / uuid.uuid4().hex
    options = Qualification(
        output,
        args.timeout,
        args.memory_limit,
        args.accuracy_bound,
        args.reference_revision,
        args.policy_revision,
        args.supersedes,
        args.revision_note,
        args.reviewed,
    )
    if args.reviewed and args.recheck is None:
        raise SuiteError("--reviewed applies to retained evidence with --recheck")
    if args.recheck is not None and args.supersedes not in (None, args.recheck):
        raise SuiteError("--recheck and --supersedes must name the same source bundle")
    with run_lock(output):
        atomic_record(output / "plan.json", plan, immutable=True)
        seen = set()
        for row in plan["rows"]:
            key = measurement_identity(row["configuration"])
            if key in seen:
                continue
            seen.add(key)
            if row["status"] != "ready":
                atomic_record(
                    output / f"excluded-{identity(row)}.json", row, immutable=True
                )
                continue
            if args.recheck is None:
                qualify_row(row, plan, options)
            else:
                recheck_row(row, args.recheck, options)
    print(f"Qualification evidence: {output.resolve()}")
    return 0
