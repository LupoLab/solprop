# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Version, identify and persist benchmark records without scientific imports."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "LOCAL_ROOT",
    "Record",
    "SuiteError",
    "atomic_record",
    "canonical",
    "file_hash",
    "identity",
    "metadata_value",
    "now",
    "read_record",
]

# JSON is the boundary to independently installed worker environments and native
# build records. Dynamic fields stay in these records, never in the case builders.
type Record = dict[str, Any]

LOCAL_ROOT = Path(".benchmarks")


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
        Path(temporary).replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary).unlink()
