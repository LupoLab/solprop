# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Store durable queue results without loading Python objects from files."""

import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from solprop.exceptions import OutputError
from solprop.output.hdf5 import _dataset
from solprop.output.scan import (
    PointResult,
    _array_paths,
    _json,
    _write_array,
    validate_result,
)

__all__ = ["artifact_digest", "read_artifact", "write_artifact"]


def artifact_digest(path: Path) -> str:
    """Hash a complete artifact with bounded memory.

    Parameters
    ----------
    path : pathlib.Path
        Closed artifact file.

    Returns
    -------
    str
        SHA-256 of the exact file bytes.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_artifact(
    path: Path, result: PointResult, identity: Mapping[str, str]
) -> None:
    """Write and fsync one private artifact before publishing its filename.

    Parameters
    ----------
    path : pathlib.Path
        Unique, unpublished worker-local filename on the shared filesystem.
    result : PointResult
        Complete successful or failed point result.
    identity : mapping
        Scan digest, point digest and ownership token.

    Raises
    ------
    OutputError
        If the status or successful payload violates its contract.
    """
    if result.status not in ("done", "failed"):
        raise OutputError("only completed or failed results are durable artifacts")
    if result.status == "done":
        validate_result(result)
    elif result.arrays:
        raise OutputError("failed artifacts must not retain numerical arrays")
    with h5py.File(path, "x") as file:
        file.attrs.update(artifact_schema_version=1, **identity)
        record = {
            field.name: getattr(result, field.name)
            for field in dataclasses.fields(result)
            if field.name != "arrays"
        }
        file.create_dataset("record", data=_json(record), dtype=h5py.string_dtype())
        arrays = file.create_group("arrays")
        for name, array in result.arrays.items():
            parent, _, leaf = name.rpartition("/")
            _write_array(
                arrays.require_group(parent) if parent else arrays, leaf, array
            )
        file.flush()
    # Windows commits a file only through a handle open for writing.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def read_artifact(
    path: Path, identity: Mapping[str, str], *, digest: str | None = None
) -> PointResult:
    """Validate and read a durable point without unpickling executable content.

    Parameters
    ----------
    path : pathlib.Path
        Published artifact file.
    identity : mapping
        Required scan, point and optional owner identifiers.
    digest : str or None, optional
        Expected digest from the durable queue state.

    Returns
    -------
    PointResult
        Validated result with host arrays.

    Raises
    ------
    OutputError
        If the file is from another scan, point or owner, or its digest changed.
    """
    if digest is not None and artifact_digest(path) != digest:
        raise OutputError(f"artifact digest changed: {path}")
    with h5py.File(path, "r") as file:
        if file.attrs.get("artifact_schema_version") != 1 or any(
            file.attrs.get(key) != value for key, value in identity.items()
        ):
            raise OutputError("artifact scan, point, owner or schema does not match")
        record = json.loads(str(_dataset(file, "record").asstr()[()]))
        arrays = {
            name: np.asarray(_dataset(file, f"arrays/{name}")[...])
            for name in _array_paths(cast(h5py.Group, file["arrays"]))
        }
    result = PointResult(**record, arrays=arrays)
    if result.status == "done":
        validate_result(result)
    elif result.status != "failed":
        raise OutputError("artifact status must be done or failed")
    elif result.arrays:
        raise OutputError("failed artifacts must not retain numerical arrays")
    return result
