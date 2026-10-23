# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Export an explicit public view without copying private bundle contents."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from benchmarks.comparison import completed
from benchmarks.records import (
    Record,
    SuiteError,
    atomic_record,
    attempts,
    distribution,
    identity,
    read_record,
    run_lock,
    write_csv,
)

__all__ = ["export_public", "public_record", "scalar_columns", "summaries"]

CONFIGURATION_COLUMNS = (
    "case",
    "revision",
    "case_hash",
    "track",
    "backend",
    "threads",
    "policy",
    "start",
    "stop",
    "step",
    "rtol",
    "atol",
    "observations",
    "fields",
    "statistics",
    "noise_seed",
    "reset",
    "measurement_version",
)
SAMPLE_COLUMNS = (
    "attempt_id",
    "sample_id",
    "arm",
    "round",
    "configuration_id",
    "created",
    "phase",
    "status",
    "clean",
    "seconds",
    "startup_seconds",
    "peak_rss_bytes",
    "sampled_peak_bytes",
    "measurement_peak_bytes",
    "estimator_defect",
    "accuracy_overridden",
    "warning_unreviewed",
    "returncode",
)
OMISSIONS = (
    "hostnames and original machine aliases",
    "local paths, executables, source locations and raw logs",
    "process identities, command arguments and environment values",
    "free-form diagnostics, errors and compiler/link flags",
)


def scalar_columns(record: Record, names: tuple[str, ...]) -> Record:
    """Retain named scalar fields; nested details stay in their JSON records."""
    return {
        name: record[name]
        for name in names
        if name in record
        and isinstance(record[name], (str, int, float, bool, type(None)))
    }


def _label(value: object) -> str | None:
    # Machine/build labels cannot contain paths, shell fragments or CSV formulae.
    return (
        value
        if isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .+_-]*", value)
        else None
    )


def public_record(manifest: Record, rows: list[Record], alias: str) -> Record:
    """Create a whitelisted publication snapshot, retaining failures and unknowns."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", alias):
        raise SuiteError(
            "public alias must use letters, digits, dots, dashes or underscores"
        )
    plan = manifest["plan"]
    configurations = []
    for row in plan["rows"]:
        values = scalar_columns(row["configuration"], CONFIGURATION_COLUMNS)
        if any(
            _label(values.get(name)) is None for name in ("case", "track", "backend")
        ):
            raise SuiteError("configuration has an invalid public label")
        configurations.append(
            {
                "id": row["id"],
                "configuration": values,
                "schedule": row["configuration"].get("schedule", []),
                "status": row["status"],
                "estimate_bytes": row.get("estimate_bytes"),
                "reference": row.get("reference"),
            }
        )
    environments = {}
    for arm, environment in plan.get("environments", {}).items():
        build = environment.get("build_info", {})
        environments[arm] = {
            "packages": {
                name: _label(version)
                for name, version in environment.get("packages", {}).items()
            },
            "source": scalar_columns(
                environment.get("source") or {}, ("revision", "dirty", "dirty_hash")
            ),
            "extension_hash": environment.get("extension_hash"),
            "build": {
                key: value
                for key in (
                    "compiler_id",
                    "compiler_version",
                    "execution_space",
                    "kokkos_version",
                    "build_type",
                    "fftw_provider",
                    "blas_provider",
                    "coverage_instrumented",
                )
                if (value := build.get(key)) is not None
                and (isinstance(value, (bool, int, float)) or _label(value) is not None)
            },
        }
    samples = []
    for row in rows:
        sample = scalar_columns(row, SAMPLE_COLUMNS)
        accuracy = row.get("accuracy", {})
        sample.update(
            accuracy_status=accuracy.get("status", "unavailable"),
            reference_id=accuracy.get("reference_id"),
            qualified_for_adoption=accuracy.get("qualified_for_adoption", False),
            peak_rss_unavailable_reason=None
            if row.get("peak_rss_bytes") is not None
            else "not recorded by worker",
            repetitions=[
                scalar_columns(
                    repetition,
                    (
                        "seconds",
                        "accepted",
                        "rejected",
                        "source_evaluations",
                        "endpoint",
                        "observation_seconds",
                    ),
                )
                for repetition in row.get("repetitions", [])
            ],
        )
        samples.append(sample)
    hardware = scalar_columns(
        plan.get("system", {}),
        (
            "system",
            "machine",
            "logical_cpus",
            "physical_cores",
            "memory_bytes",
            "processor",
            "cpu_model",
        ),
    )
    return {
        "publication_schema_version": 1,
        "suite_version": plan["suite_version"],
        "result_schema_version": plan["schema_version"],
        "created": manifest["created"],
        "bundle_id": identity(manifest),
        "alias": alias,
        "mode": plan["mode"],
        "exploratory": plan.get("exploratory", False),
        "profile": scalar_columns(
            plan["profile"],
            ("revision", "production", "policy", "explicit", "topology_hash"),
        ),
        "hardware": hardware,
        "hardware_unavailable_reason": None
        if hardware
        else "not retained by this bundle version",
        "environments": environments,
        "configurations": configurations,
        "samples": samples,
        "omissions": list(OMISSIONS),
    }


def summaries(rows: list[Record]) -> list[Record]:
    """Summarize final attempts with separate all-sample and clean populations."""
    groups: dict[tuple[str, str], list[Record]] = {}
    for row in completed(rows):
        groups.setdefault((row["configuration_id"], row["arm"]), []).append(row)
    result = []
    for (configuration, arm), group in groups.items():
        for population in ("all", "clean"):
            selected = [
                row
                for row in group
                if row["status"] == "ok"
                and (population == "all" or row.get("clean") is True)
            ]
            result.append(
                {
                    "configuration_id": configuration,
                    "arm": arm,
                    "population": population,
                    **asdict(distribution([row["seconds"] for row in selected])),
                    "failed_samples": sum(row["status"] != "ok" for row in group),
                }
            )
    return result


def export_public(source: Path, destination: Path, alias: str) -> Record:
    """Write a separate immutable public JSON snapshot and flat CSV views.

    Publication does not copy raw files, alter the local bundle, or commit files.
    The caller chooses a public machine alias independently of its local profile.
    """
    if source.resolve() == destination.resolve():
        raise SuiteError("publication destination must differ from the local bundle")
    if destination.exists() and any(destination.iterdir()):
        raise SuiteError("publication destination must be empty or new")
    with run_lock(source):
        result = public_record(
            read_record(source / "manifest.json"), attempts(source), alias
        )
    with run_lock(destination):
        atomic_record(destination / "manifest.json", result, immutable=True)
        write_csv(
            destination / "samples.csv",
            [
                {key: value for key, value in row.items() if key != "repetitions"}
                for row in result["samples"]
            ],
        )
        write_csv(
            destination / "repetitions.csv",
            [
                {"attempt_id": row["attempt_id"], "repetition": index, **repetition}
                for row in result["samples"]
                for index, repetition in enumerate(row["repetitions"])
            ],
        )
        write_csv(
            destination / "configurations.csv",
            [
                {
                    "configuration_id": row["id"],
                    **row["configuration"],
                    "status": row["status"],
                    "estimate_bytes": row["estimate_bytes"],
                }
                for row in result["configurations"]
            ],
        )
        write_csv(destination / "summary.csv", summaries(result["samples"]))
        write_csv(
            destination / "cases.csv",
            list(
                {
                    row["configuration"]["case_hash"]: scalar_columns(
                        row["configuration"], ("case", "revision", "case_hash")
                    )
                    for row in result["configurations"]
                }.values()
            ),
        )
        write_csv(
            destination / "runs.csv",
            [
                scalar_columns(
                    result,
                    (
                        "bundle_id",
                        "created",
                        "mode",
                        "alias",
                        "suite_version",
                        "result_schema_version",
                        "exploratory",
                    ),
                )
            ],
        )
        write_csv(
            destination / "system.csv",
            [
                {
                    "alias": alias,
                    **result["hardware"],
                    "unavailable_reason": result["hardware_unavailable_reason"],
                }
            ],
        )
    return result
