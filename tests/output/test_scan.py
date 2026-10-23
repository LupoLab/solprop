# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise commit ordering and recovery independently of propagation cost."""

import dataclasses
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Mapping
from typing import cast

import h5py
import numpy as np
import pytest

from solprop.driver.scan import Scan
from solprop.exceptions import OutputError, ProcessingError
from solprop.output._file_lock import lock, unlock
from solprop.output._scan_lock import lock_path, read_scan
from solprop.output.hdf5 import _dataset, _group
from solprop.output.scan import PointResult, ScanWriter
from solprop.output.schema import _example_metadata
from solprop.processing.scans import ScanOutput
from solprop.processing.scans import open as open_scan


def test_failed_scan_point_keeps_provenance_and_masks_numerical_arrays(tmp_path):
    path = tmp_path / "failure.h5"
    writer = ScanWriter(path, Scan("failure", x=[1]), {})
    complete = result(0)
    failed = dataclasses.replace(complete, status="failed", error="reducer failed")
    writer.commit(failed)
    writer.close()
    point = ScanOutput(path).point(0)
    assert point["status"] == "failed"
    assert point["configuration_hash"] == complete.configuration_hash
    assert "metadata" in point
    assert "reduced/test" not in point
    assert not any(isinstance(value, np.ndarray) for value in point.values())
    writer = ScanWriter(path, Scan("failure", x=[1]), {})
    writer.commit(PointResult(0, "failed", error="builder now fails"))
    writer.close()
    assert "metadata" not in ScanOutput(path).point(0)


def test_failed_scan_point_validates_available_metadata_before_publication(tmp_path):
    writer = ScanWriter(tmp_path / "failure.h5", Scan("failure", x=[1]), {})
    try:
        with pytest.raises(OutputError, match="does not match"):
            writer.commit(
                dataclasses.replace(
                    result(0), status="failed", configuration_hash="0" * 64
                )
            )
        assert writer.state(0)[0] == "todo"
    finally:
        writer.close()


def result(index, *, rows=2, planes=3):
    metadata = _example_metadata()
    metadata = dataclasses.replace(
        metadata, configuration={"power": index, "shared": np.eye(3)}
    )
    return PointResult(
        index,
        "done",
        metadata.configuration_hash(),
        metadata.to_json(),
        {
            "z": np.arange(planes, dtype=float),
            "stats/energy": np.full((planes, 2), index, dtype=float),
            "step_stats/z": np.arange(rows, dtype=float),
            "step_stats/step": np.arange(rows),
            "step_stats/electrons": np.ones((rows, 2)),
        },
    )


def test_scan_csr_global_order_mask_and_common_metadata(tmp_path):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", power=[1, 2, 3]), {})
    writer.commit(result(2, rows=3))
    writer.commit(result(0, rows=1))
    writer.close()
    output = ScanOutput(path)
    grid = output.as_grid("stats/energy")
    assert np.all(np.asarray(grid.mask)[1])
    assert not np.any(np.asarray(grid.mask)[[0, 2]])
    assert list(output.layouts.values()) == [(0, 2)]
    with h5py.File(path) as file:
        np.testing.assert_array_equal(file["step_stats/z/offsets"], [0, 1, 1, 4])
        np.testing.assert_array_equal(file["step_stats/z/values"], [0.0, 0.0, 1.0, 2.0])
        assert _dataset(file, "stats/energy").shape == (3, 3, 2)
        assert "configuration.shared" in json.loads(
            str(file["common"].attrs["constant_fields"])
        )
        assert "configuration.power" in json.loads(
            str(file["common"].attrs["varying_fields"])
        )
    assert (
        cast(Mapping, cast(Mapping, output.point(2)["metadata"])["configuration"])[
            "power"
        ]
        == 2
    )


def test_open_returns_the_reader_of_a_collected_scan(tmp_path):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", power=[1, 2]), {})
    writer.commit(result(1))
    writer.close()
    output = open_scan(path)
    assert isinstance(output, ScanOutput)
    assert output.status.tolist() == ["todo", "done"]


def test_scan_recovery_reads_committed_staging_before_layouts(tmp_path):
    path = tmp_path / "scan.h5"
    scan = Scan("scan", power=[1, 2])
    writer = ScanWriter(path, scan, {})
    writer.mark_running(1, 0.0)
    writer.commit(result(0))
    writer._stage(result(1))
    # Simulate termination between payload flush and done. The session is closed
    # rather than only the file so the writer's advisory lock is released here,
    # as it would be by the death of the controller process.
    writer.close(consolidate=False)
    output = ScanOutput(path)
    assert output.status.tolist() == ["done", "running"]
    assert "stats/energy" not in output.point(1)
    assert np.all(np.asarray(output.as_grid("stats/energy").mask)[1])
    resumed = ScanWriter(path, scan, {})
    resumed.commit(result(1, rows=4))
    resumed.close()
    assert output.status.tolist() == ["done", "done"]
    with h5py.File(path) as file:
        np.testing.assert_array_equal(file["step_stats/z/offsets"], [0, 2, 6])


def test_scan_different_layouts_refuse_grid(tmp_path):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", power=[1, 2]), {})
    writer.commit(result(0))
    writer.commit(result(1, planes=4))
    writer.close()
    output = ScanOutput(path)
    assert len(output.layouts) == 2
    with pytest.raises(ProcessingError, match="incompatible"):
        output.as_grid("stats/energy")
    with h5py.File(path) as file:
        assert "stats" not in file


@pytest.mark.parametrize(
    "changes",
    [
        {"configuration_hash": "invalid"},
        {"configuration_hash": "0" * 64},
        {"metadata_json": "[]"},
        {"arrays": {}},
        {"arrays": {"z": np.zeros((1, 1))}},
        {"arrays": {"z": np.zeros(1), "../bad": np.zeros(1)}},
        {"arrays": {"z": np.array([np.nan])}},
        {"arrays": {"z": np.arange(3), "stats/energy": np.ones((2,))}},
        {
            "arrays": {
                "z": np.arange(3),
                "step_stats/z": np.ones(1),
                "step_stats/energy": np.ones(2),
            }
        },
        {"arrays": {"z": np.arange(3), "fields/values": np.ones((2, 1, 1, 2))}},
    ],
)
def test_scan_bad_payload_never_publishes_done(tmp_path, changes):
    writer = ScanWriter(tmp_path / "scan.h5", Scan("scan", x=[1]), {})
    with pytest.raises(OutputError):
        writer.commit(dataclasses.replace(result(0), **changes))
    assert writer.state(0)[0] == "todo"
    assert len(_group(writer.file, "staged")) == 0
    writer.close()


@pytest.mark.parametrize("status,index", [("unknown", 0), ("done", -1), ("done", 1)])
def test_scan_writer_refuses_invalid_publication(tmp_path, status, index):
    writer = ScanWriter(tmp_path / "scan.h5", Scan("scan", x=[1]), {})
    try:
        with pytest.raises(OutputError):
            writer.commit(dataclasses.replace(result(index), status=status))
        assert writer.state(0)[0] == "todo"
    finally:
        writer.close()


def test_scan_writer_skips_without_mutating_and_rejects_conflict(tmp_path):
    writer = ScanWriter(tmp_path / "scan.h5", Scan("scan", x=[1]), {})
    try:
        writer.commit(PointResult(0, "skipped"))
        assert writer.state(0)[0] == "todo"
        with pytest.raises(OutputError, match="different physics"):
            writer.commit(PointResult(0, "conflict", error="different physics"))
        assert writer.state(0)[0] == "todo"
    finally:
        writer.close()


def test_scan_writer_refuses_existing_foreign_file(tmp_path):
    path = tmp_path / "foreign.h5"
    with h5py.File(path, "w") as file:
        file.attrs["foreign"] = True
    with pytest.raises(OutputError, match="recognised scan"):
        ScanWriter(path, Scan("scan", x=[1]), {})
    with h5py.File(path, "r") as file:
        assert file.attrs["foreign"]


def test_scan_reader_refuses_invalid_global_point(tmp_path):
    path = tmp_path / "scan.h5"
    ScanWriter(path, Scan("scan", x=[1]), {}).close()
    with pytest.raises(ProcessingError, match="out of range"):
        ScanOutput(path).point(1)


@pytest.mark.parametrize(
    "values",
    [np.asarray(["argon", "helium"]), np.asarray([{"x": 1}, {"x": 2}], dtype=object)],
)
def test_scan_writer_preserves_text_and_json_axis_values(tmp_path, values):
    path = tmp_path / "scan.h5"
    ScanWriter(path, Scan("axes", species=values), {}).close()
    with h5py.File(path, "r") as file:
        axis = _dataset(file, "points/values/species").asstr()[...]
        actual = (
            [json.loads(value) for value in axis]
            if values.dtype.kind == "O"
            else np.asarray(axis).tolist()
        )
    assert actual == values.tolist()


def test_scan_recovers_after_layout_removal_and_preserves_embedded_metadata(tmp_path):
    path = tmp_path / "scan.h5"
    scan = Scan("scan", x=[1])
    writer = ScanWriter(path, scan, {})
    writer.commit(result(0))
    writer.close()
    with h5py.File(path, "a") as file:
        del file["layouts"]
        group = _group(file, "staged/0")
        references = json.loads(str(group.attrs["metadata_refs"]))
        references["extension"] = [1, 2]
        group.attrs["metadata_refs"] = json.dumps(references)
    assert cast(Mapping, ScanOutput(path).point(0)["metadata"])["extension"] == [1, 2]
    ScanWriter(path, scan, {}).close()
    assert ScanOutput(path).status.tolist() == ["done"]


def test_scan_writer_refuses_to_demote_a_completed_point(tmp_path):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", x=[1]), {})
    try:
        writer.commit(result(0))
        with pytest.raises(OutputError, match="point 0 is already complete"):
            writer.commit(PointResult(0, "failed", error="builder now raises"))
        assert writer.state(0)[0] == "done"
        assert len(_group(writer.file, "staged/0")) > 0
    finally:
        writer.close()
    point = ScanOutput(path).point(0)
    assert point["status"] == "done"
    np.testing.assert_array_equal(point["stats/energy"], np.zeros((3, 2)))


def test_scan_writer_holds_an_exclusive_lock_for_its_session(tmp_path):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", x=[1]), {})
    sidecar = lock_path(path)
    assert sidecar.exists()
    descriptor = os.open(sidecar, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            lock(descriptor, shared=True, blocking=False)
        writer.close()
        lock(descriptor, shared=True, blocking=False)
        unlock(descriptor)
    finally:
        os.close(descriptor)


def test_scan_reader_refuses_a_missing_file_before_creating_the_sidecar(tmp_path):
    path = tmp_path / "missing.h5"
    with pytest.raises(FileNotFoundError, match="no scan file"), read_scan(path):
        pass  # pragma: no cover - never entered
    assert not lock_path(path).exists()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="a read-only directory mode is not enforced on Windows",
)
def test_scan_reader_degrades_to_an_unlocked_read_where_it_cannot_write(
    tmp_path, caplog
):
    scan = Scan("locked", power=[1.0, 2.0])
    path = tmp_path / "scan.h5"
    ScanWriter(path, scan, {}).close(consolidate=False)
    lock_path(path).unlink()
    tmp_path.chmod(0o555)
    try:
        with (
            caplog.at_level(logging.DEBUG, logger="solprop.output._scan_lock"),
            read_scan(path) as file,
        ):
            assert "scan_schema_version" in file.attrs
    finally:
        tmp_path.chmod(0o755)
    assert not lock_path(path).exists()
    assert "without a lock" in caplog.text


def test_scan_reader_waits_for_an_open_writer_and_says_so(tmp_path, caplog):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", x=[1]), {})
    writer.commit(result(0))
    seen: list[str] = []

    def read():
        with read_scan(path) as file:
            seen.append(str(_dataset(file, "points/status").asstr()[0]))

    with caplog.at_level(logging.INFO, logger="solprop.output._scan_lock"):
        reader = threading.Thread(target=read)
        reader.start()
        try:
            deadline = time.monotonic() + 10.0
            # The writer holds the lock until the reader has certainly blocked on
            # it, so the contended branch is reached without racing the release.
            while "waiting for the writer" not in caplog.text:
                assert time.monotonic() < deadline, "the reader never blocked"
                time.sleep(0.01)
        finally:
            writer.close()
            reader.join(10.0)
    assert not reader.is_alive()
    assert seen == ["done"]


def test_partial_staging_never_exposes_unvalidated_metadata(tmp_path):
    path = tmp_path / "scan.h5"
    writer = ScanWriter(path, Scan("scan", x=[1]), {})
    pending = result(0)
    metadata = json.loads(pending.metadata_json)
    metadata["versions"] = None
    writer._stage(dataclasses.replace(pending, metadata_json=json.dumps(metadata)))
    writer.close()
    assert ScanOutput(path).point(0)["status"] == "todo"
    assert "metadata" not in ScanOutput(path).point(0)
