# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Version, identify and persist benchmark records without scientific imports."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "LOCAL_ROOT",
    "SCHEMA_VERSION",
    "SUITE_VERSION",
    "Distribution",
    "Record",
    "SuiteError",
    "atomic_record",
    "attempts",
    "canonical",
    "distribution",
    "export_tables",
    "file_hash",
    "identity",
    "metadata_value",
    "now",
    "read_record",
    "run_lock",
    "write_csv",
]

# JSON is the boundary to independently installed worker environments and native
# build records. Dynamic fields stay in these records, never in the case builders.
type Record = dict[str, Any]

SUITE_VERSION = 1
SCHEMA_VERSION = 1
LOCAL_ROOT = Path(".benchmarks") / "suite"


class SuiteError(ValueError):
    """Reject an invalid benchmark request or persisted record."""


def canonical(value: object) -> str:
    """Encode identity data deterministically, refusing non-finite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value: object) -> str:
    """Identify resolved values independently of their source spelling."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def metadata_value(value: Any) -> Any:
    """Encode unbounded metadata intervals without permitting non-finite samples.

    Material validity domains can legitimately contain infinity. Tag those
    values explicitly in metadata; timing and numerical arrays still refuse them.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_metadata": str(value)}
    if isinstance(value, dict):
        return {key: metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_value(item) for item in value]
    return value


def file_hash(path: Path) -> str:
    """Hash an actual file, including an installed native extension."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now() -> str:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def read_record(path: Path) -> Record:
    """Read an object record, rejecting malformed and non-finite JSON."""

    def invalid(value: str) -> None:
        raise SuiteError(f"{path}: non-finite JSON value {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(value, dict):
        raise SuiteError(f"{path}: expected a JSON object")
    return value


def atomic_record(path: Path, value: Record, *, immutable: bool = False) -> None:
    """Publish a flushed record atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise SuiteError(f"refusing to replace completed record {path}")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # The controller holds the run lock, so no other owner can create path
        # between the existence check and this same-filesystem rename.
        Path(temporary).replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary).unlink()


@contextlib.contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    """Hold a kernel-released lock, including across a controller crash."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415 - platform-specific

            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise SuiteError(f"run already locked: {directory}") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415 - POSIX lock

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SuiteError(f"run already locked: {directory}") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def write_csv(path: Path, rows: Sequence[Record]) -> None:
    """Write a derived UTF-8 table, with explicit JSON for nested values."""
    keys = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".csv-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: canonical(value)
                        if isinstance(value, (dict, list, tuple))
                        else ""
                        if value is None
                        else value
                        for key, value in row.items()
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary).unlink()


@dataclasses.dataclass(frozen=True)
class Distribution:
    """Keep descriptive statistics; one observation has no dispersion estimate."""

    count: int
    median: float | None
    mad: float | None


def distribution(values: Sequence[float]) -> Distribution:
    """Summarize positive finite times without selecting best observations."""
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise SuiteError("sample times must be finite and positive")
    if not values:
        return Distribution(0, None, None)
    median = statistics.median(values)
    mad = (
        None
        if len(values) < 2
        else statistics.median(abs(value - median) for value in values)
    )
    return Distribution(len(values), median, mad)


def attempts(directory: Path) -> list[Record]:
    """Read only completed immutable shards; temporary writes are ignored."""
    return [
        read_record(path) for path in sorted((directory / "attempts").glob("*.json"))
    ]


def export_tables(directory: Path) -> None:
    """Rebuild derived tables from durable attempts after collection or resume."""
    rows = attempts(directory)
    write_csv(directory / "samples.csv", rows)
    write_csv(
        directory / "accuracy.csv",
        [
            {
                "attempt_id": row["attempt_id"],
                "sample_id": row["sample_id"],
                "arm": row["arm"],
                "configuration_id": row["configuration_id"],
                **row.get("accuracy", {"status": "unavailable"}),
            }
            for row in rows
        ],
    )
    from benchmarks.comparison import completed  # noqa: PLC0415

    groups: dict[tuple[str, str], list[Record]] = {}
    for row in completed(rows):
        groups.setdefault((row["configuration_id"], row["arm"]), []).append(row)
    summaries: list[Record] = []
    for (configuration, arm), group in groups.items():
        valid = [row for row in group if row["status"] == "ok"]
        for population in ("all", "clean"):
            selected = (
                valid
                if population == "all"
                else [row for row in valid if row.get("clean") is True]
            )
            times = distribution([row["seconds"] for row in selected])
            peaks = [
                row["peak_rss_bytes"]
                for row in selected
                if row.get("peak_rss_bytes") is not None
            ]
            starts = [
                row["startup_seconds"]
                for row in selected
                if row.get("startup_seconds") is not None
            ]
            summaries.append(
                {
                    "configuration_id": configuration,
                    "arm": arm,
                    "population": population,
                    **dataclasses.asdict(times),
                    "peak_rss_bytes": max(peaks) if peaks else None,
                    "startup_seconds": statistics.median(starts) if starts else None,
                    "failed_attempts": len(group) - len(valid),
                }
            )
    write_csv(directory / "summary.csv", summaries)
    from benchmarks.views import write_views  # noqa: PLC0415

    write_views(directory, rows)
