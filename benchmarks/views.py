# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Derive normalized local CSV views from the manifest and immutable attempts."""

from __future__ import annotations

from pathlib import Path

from benchmarks.publication import CONFIGURATION_COLUMNS, scalar_columns
from benchmarks.records import Record, identity, read_record, write_csv

__all__ = ["write_views"]


def write_views(directory: Path, rows: list[Record]) -> None:
    """Keep scalar tables linked by bundle, configuration and attempt identities."""
    path = directory / "manifest.json"
    if not path.exists():
        return
    manifest = read_record(path)
    plan = manifest["plan"]
    run_id = identity(manifest)
    comparison_path = directory / "comparison.json"
    comparison = read_record(comparison_path) if comparison_path.exists() else {}
    write_csv(
        directory / "runs.csv",
        [
            {
                "run_id": run_id,
                "created": manifest["created"],
                **scalar_columns(
                    plan, ("schema_version", "suite_version", "mode", "exploratory")
                ),
                "collection_seconds": comparison.get("collection_seconds"),
                "collection_seconds_reason": None
                if "collection_seconds" in comparison
                else "collection incomplete",
            }
        ],
    )
    configurations = [
        {
            "configuration_id": row["id"],
            **scalar_columns(row["configuration"], CONFIGURATION_COLUMNS),
            "status": row["status"],
            "reason": row.get("reason"),
            "estimate_bytes": row.get("estimate_bytes"),
            "estimate_status": row.get("estimate_status"),
        }
        for row in plan["rows"]
    ]
    write_csv(directory / "configurations.csv", configurations)
    write_csv(
        directory / "cases.csv",
        list(
            {
                row.get("case_hash", row["case"]): scalar_columns(
                    row, ("case", "revision", "case_hash")
                )
                for row in configurations
            }.values()
        ),
    )
    system = plan.get("system", {})
    write_csv(
        directory / "system.csv",
        [
            {
                "run_id": run_id,
                **scalar_columns(
                    system,
                    (
                        "system",
                        "machine",
                        "logical_cpus",
                        "physical_cores",
                        "memory_bytes",
                        "processor",
                        "cpu_model",
                        "cpu_model_reason",
                        "cpu_topology_reason",
                        "core_residency",
                        "core_residency_reason",
                    ),
                ),
                "unavailable_reason": None
                if system
                else "not retained by this bundle version",
            }
        ],
    )
    write_csv(
        directory / "allocations.csv",
        [
            {
                "run_id": run_id,
                "cpu": cpu,
                "production": plan["profile"]["production"],
                "policy": plan["profile"]["policy"],
                "profile_explicit": plan["profile"]["explicit"],
                **next(
                    (
                        scalar_columns(
                            item, ("core", "socket", "core_reason", "socket_reason")
                        )
                        for item in system.get("cpu_topology") or []
                        if item["cpu"] == cpu
                    ),
                    {},
                ),
            }
            for cpu in plan["profile"]["allocation"]
        ],
    )
    write_csv(
        directory / "processes.csv",
        [
            {
                "attempt_id": row["attempt_id"],
                **scalar_columns(
                    row.get("process", {}),
                    (
                        "pid",
                        "created",
                        "cpu_seconds",
                        "cpu_reason",
                        "rss_samples",
                        "missed_samples",
                        "poll_interval_seconds",
                    ),
                ),
                "unavailable_reason": None
                if row.get("process")
                else "no retained process receipt",
            }
            for row in rows
        ],
    )
    libraries = []
    for arm, environment in plan.get("environments", {}).items():
        found = environment.get("libraries", [])
        for library in found or [{}]:
            libraries.append(
                {
                    "arm": arm,
                    "scope": "inventory",
                    **scalar_columns(
                        library,
                        (
                            "user_api",
                            "internal_api",
                            "prefix",
                            "filepath",
                            "version",
                            "num_threads",
                            "threading_layer",
                            "architecture",
                        ),
                    ),
                    "unavailable_reason": None
                    if library
                    else "no supported pool observed",
                }
            )
    for row in rows:
        for library in row.get("provenance", {}).get("libraries", []):
            libraries.append(
                {
                    "arm": row["arm"],
                    "attempt_id": row["attempt_id"],
                    "scope": "after measurement",
                    **scalar_columns(
                        library,
                        (
                            "user_api",
                            "internal_api",
                            "prefix",
                            "filepath",
                            "version",
                            "num_threads",
                            "threading_layer",
                            "architecture",
                        ),
                    ),
                }
            )
    write_csv(directory / "libraries.csv", libraries)
    write_csv(
        directory / "coverage.csv",
        [
            {
                "run_id": run_id,
                "status": "not-collected",
                "reason": "native coverage requires its own collection",
            }
        ],
    )
