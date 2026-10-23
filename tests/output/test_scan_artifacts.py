# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reject malformed durable results before a collector can publish them."""

import dataclasses
import json

import h5py
import numpy as np
import pytest

from output.test_scan import result
from solprop.exceptions import OutputError
from solprop.output.hdf5 import _dataset, _group
from solprop.output.scan import PointResult
from solprop.output.scan_artifacts import read_artifact, write_artifact


@pytest.mark.parametrize(
    "payload",
    [PointResult(0, "running"), PointResult(0, "failed", arrays={"z": np.zeros(1)})],
)
def test_artifact_write_refuses_incomplete_or_failed_numerical_payload(
    tmp_path, payload
):
    path = tmp_path / "point.h5"
    with pytest.raises(OutputError):
        write_artifact(path, payload, {"scan": "identity"})
    assert not path.exists()


@pytest.mark.parametrize("change", [{"scan": "other"}, {"artifact_schema_version": 2}])
def test_artifact_read_refuses_wrong_identity_or_schema(tmp_path, change):
    path = tmp_path / "point.h5"
    write_artifact(path, result(0), {"scan": "identity"})
    with h5py.File(path, "a") as file:
        file.attrs.update(change)
    with pytest.raises(OutputError, match="does not match"):
        read_artifact(path, {"scan": "identity"})


def test_artifact_read_refuses_unfinished_status(tmp_path):
    path = tmp_path / "point.h5"
    write_artifact(path, result(0), {})
    with h5py.File(path, "a") as file:
        record = json.loads(str(_dataset(file, "record").asstr()[()]))
        record["status"] = "running"
        _dataset(file, "record")[()] = json.dumps(record)
    with pytest.raises(OutputError, match="status must be"):
        read_artifact(path, {})


def test_artifact_failed_read_refuses_injected_arrays(tmp_path):
    path = tmp_path / "point.h5"
    failed = dataclasses.replace(result(0), status="failed", arrays={}, error="failed")
    write_artifact(path, failed, {})
    restored = read_artifact(path, {})
    assert restored.status == "failed"
    assert restored.error == "failed"
    assert not restored.arrays
    with h5py.File(path, "a") as file:
        _group(file, "arrays").create_dataset("z", data=np.zeros(1))
    with pytest.raises(OutputError, match="failed artifacts"):
        read_artifact(path, {})
