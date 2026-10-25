# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify snapshot publication, bounded storage, ownership and corrupt-input refusal."""

import dataclasses
import inspect
import json
import shutil
from typing import cast

import h5py
import numpy as np
import pytest

from solprop.exceptions import OutputError
from solprop.output import (
    HDF5Output,
    RunOutcome,
    SavedPlane,
    SnapshotController,
    read_snapshot,
    restore_snapshot,
)
from solprop.output import snapshots as module
from solprop.output._writer_lock import WriterLock
from solprop.output.schema import GitProvenance, _example_metadata


@pytest.fixture
def working(tmp_path):
    output = HDF5Output(tmp_path / "working.h5")
    output.open(_example_metadata())
    assert output.metadata is not None
    output.save(
        SavedPlane(
            z=0.125, field=np.ones((1, 1, output.metadata.grid.n_omega), complex)
        )
    )
    output.record_outcome(RunOutcome("running", 0.125))
    controller = SnapshotController(tmp_path / "snapshots")
    controller._start()
    try:
        yield output, controller
    finally:
        output.close()
        controller._finish()


def take(working):
    output, controller = working
    controller.request()
    controller._service(
        output, None, next_save_index=0, require_history=False, persist=lambda: None
    )
    return controller.last_result


@pytest.fixture
def generation(working):
    result = take(working)
    assert result.disposition == "published", result.failure
    return result.generation


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan"), True])
def test_invalid_interval_is_refused(tmp_path, interval):
    with pytest.raises(OutputError, match="interval"):
        SnapshotController(tmp_path, interval_seconds=interval)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_is_refused(tmp_path, capacity):
    with pytest.raises(OutputError, match="max_snapshots"):
        inspect.unwrap(SnapshotController.__init__)(
            SnapshotController.__new__(SnapshotController),
            tmp_path,
            max_snapshots=capacity,
        )


@pytest.mark.parametrize("git", [None, GitProvenance("1" * 40, "HEAD", False)])
def test_snapshot_records_current_execution_provenance_without_changing_identity(
    working, git
):
    output, controller = working
    original = output.metadata
    current = dataclasses.replace(original, versions={"solprop": "current"}, git=git)
    controller.request()
    controller._service(
        output,
        None,
        next_save_index=0,
        require_history=False,
        persist=lambda: None,
        provenance=current,
    )
    generation = controller.last_result.generation
    info = read_snapshot(generation)
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["versions"] == {"solprop": "current"}
    assert manifest["checkpoint"]["git"] == (
        None if git is None else dataclasses.asdict(git)
    )
    assert info.configuration_hash == original.configuration_hash()
    with HDF5Output(info.files.output, readonly=True) as stored:
        retained = stored.metadata
        assert retained is not None
        assert retained.versions == original.versions
        assert retained.git == original.git


def test_publication_is_exact_read_only_and_restore_is_writable(
    working, generation, tmp_path
):
    output, _ = working
    info = read_snapshot(generation)
    # Opening HDF5 changes its superblock flags; compare both closed files.
    output.suspend()
    try:
        assert info.files.output.read_bytes() == output.path.read_bytes()
    finally:
        output.reopen()
    assert info.n_saved == 1 and info.next_save_index == 0 and info.z == 0.125
    assert info.files.tdse_archive is None
    assert info.files.output.stat().st_mode & 0o222 == 0
    restored = restore_snapshot(generation, tmp_path / "restored")
    assert restored.output.read_bytes() == info.files.output.read_bytes()
    assert restored.output.stat().st_mode & 0o200
    assert restored.tdse_archive is None
    with pytest.raises(OutputError, match="immutable"):
        HDF5Output(info.files.output, append=True).open(output.metadata)
    with pytest.raises(OutputError, match="new directory"):
        restore_snapshot(generation, tmp_path / "restored")
    with pytest.raises(OutputError, match="new directory"):
        restore_snapshot(generation, generation / "child")


def test_capacity_retains_pending_request_and_then_recovers(working, monkeypatch):
    output, controller = working
    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    paths = [take(working).generation for _ in range(3)]
    assert take(working).disposition == "deferred"
    assert len(list(controller._directory.glob("snapshot-*"))) == 3
    module._discard_staging(
        paths[0]
    )  # The external controller retires an uploaded copy.
    controller._service(
        output, None, next_save_index=0, require_history=False, persist=lambda: None
    )
    assert controller.last_result.disposition == "deferred"
    clock[0] += 60
    controller._service(
        output, None, next_save_index=0, require_history=False, persist=lambda: None
    )
    assert controller.last_result.disposition == "published"


def test_elapsed_requests_and_explicit_retry_are_coalesced(working, monkeypatch):
    output, controller = working
    now = controller._last_success
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    controller._interval = 10
    controller._service(
        output, None, next_save_index=0, require_history=False, persist=lambda: None
    )
    assert controller.last_result is None
    # Past the interval, not on it: from a real monotonic origin, (now + 10) - now can
    # round to just below 10 and leave the request not yet due.
    now += 11
    controller._service(
        output, None, next_save_index=0, require_history=False, persist=lambda: None
    )
    latest = controller.last_result
    assert latest is not None and latest.disposition == "published"
    generation = latest.generation
    controller._service(
        output, None, next_save_index=0, require_history=False, persist=lambda: None
    )
    latest = controller.last_result
    assert latest is not None and latest.generation == generation


def test_request_during_copy_remains_pending(working, monkeypatch):
    _, controller = working
    original = module._copy_generation

    def request_again(*args):
        controller.request()
        return original(*args)

    monkeypatch.setattr(module, "_copy_generation", request_again)
    take(working)
    assert controller._due()


@pytest.mark.parametrize("operation", ["copy", "checksum", "manifest", "mkdir"])
def test_failed_publication_preserves_previous_generations(
    working, generation, monkeypatch, operation
):
    expected = read_snapshot(generation)
    original = expected.files.output.read_bytes()
    if operation == "copy":
        monkeypatch.setattr(
            module,
            "_copy_file",
            lambda *args: (_ for _ in ()).throw(OSError("copy failed")),
        )
    elif operation == "checksum":
        monkeypatch.setattr(module, "_digest", lambda path: "f" * 64)
    elif operation == "manifest":
        monkeypatch.setattr(
            module,
            "_publish_manifest",
            lambda *args: (_ for _ in ()).throw(OSError("manifest failed")),
        )
    else:
        monkeypatch.setattr(
            module,
            "_checkpoint",
            lambda *args: (_ for _ in ()).throw(OutputError("bad checkpoint")),
        )
    result = take(working)
    assert result.disposition == "failed" and result.failure
    assert expected.files.output.read_bytes() == original
    assert len(list(working[1]._directory.glob("snapshot-*"))) == 1
    assert not list(working[1]._directory.glob(".snapshot-*"))
    assert working[0]._file is not None


@pytest.mark.parametrize("phase", ["persist", "suspend", "reopen"])
def test_writer_failures_stop_and_release_resources(
    working, monkeypatch, caplog, phase
):
    output, controller = working

    def fail(*args, **kwargs):
        raise OutputError(f"{phase} failed")

    persist = fail if phase == "persist" else lambda: None
    if phase != "persist":
        monkeypatch.setattr(output, phase, fail)
    controller.request()
    with pytest.raises(OutputError, match=f"{phase} failed"):
        controller._service(
            output, None, next_save_index=0, require_history=False, persist=persist
        )
    assert controller.last_result.disposition == "failed"
    assert "snapshot transaction failed" in caplog.text
    assert f"{phase} failed" in caplog.text
    if phase == "reopen":
        read_snapshot(controller.last_result.generation)


def test_writer_ownership_survives_suspension(working):
    output, controller = working
    output.suspend()
    try:
        with pytest.raises(OutputError, match="another writer"):
            HDF5Output(output.path, append=True).open(output.metadata)
        with pytest.raises(OutputError, match="another writer"):
            SnapshotController(controller._directory)._start()
        with pytest.raises(OutputError, match="active run"):
            controller._start()
    finally:
        output.reopen()
    output._writer_lock.acquire()  # Re-entrant ownership does not open a second lease.


@pytest.mark.parametrize("changed", ["inode", "counts", "outcome", "configuration"])
def test_reopen_refuses_a_changed_optical_file(working, changed):
    output, _ = working
    output.suspend()
    if changed == "inode":
        other = output.path.with_suffix(".other")
        shutil.copyfile(output.path, other)
        other.replace(output.path)
    else:
        with h5py.File(output.path, "r+") as file:
            if changed == "counts":
                file.attrs["n_saved"] = 0
            elif changed == "outcome":
                del file["meta/outcome"]
            else:
                value = json.loads(cast("h5py.Dataset", file["meta/json"])[()].decode())
                value["initial_z"] = 0.01
                del file["meta/json"]
                file.create_dataset("meta/json", data=json.dumps(value))
    with pytest.raises(OutputError, match=r"changed|replaced"):
        output.reopen()


def test_reopen_requires_a_suspended_writer(working):
    with pytest.raises(OutputError, match="not suspended"):
        working[0].reopen()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "schema",
        "checksum",
        "identity",
        "escape",
        "roles",
        "duplicate",
        "object",
        "files",
    ],
)
def test_read_refuses_incomplete_corrupt_and_mismatched_generations(generation, change):
    path = generation / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.chmod(0o600)
    if change == "missing":
        path.unlink()
    else:
        if change == "schema":
            manifest["schema"] = "future"
        elif change == "checksum":
            manifest["files"][0]["sha256"] = "0" * 64
        elif change == "identity":
            manifest["checkpoint"]["next_save_index"] += 1
        elif change == "escape":
            manifest["files"][0]["path"] = "../working.h5"
        elif change == "roles":
            manifest["files"][0]["role"] = "archive"
        elif change == "duplicate":
            manifest["files"].append({**manifest["files"][0], "role": "archive"})
        elif change == "object":
            manifest = []
        else:
            manifest["files"] = []
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OutputError):
        read_snapshot(generation)


def test_incomplete_copy_is_not_a_published_snapshot(generation):
    staged = generation.with_name("." + generation.name)
    generation.rename(staged)
    with pytest.raises(OutputError, match="incomplete"):
        read_snapshot(staged)


def test_manifest_with_a_symlink_is_refused(generation, tmp_path):
    info = read_snapshot(generation)
    outside = tmp_path / "outside.h5"
    shutil.copyfile(info.files.output, outside)
    info.files.output.chmod(0o600)  # published read-only; Windows cannot unlink that
    info.files.output.unlink()
    try:
        info.files.output.symlink_to(outside)
    except OSError as error:
        # Windows creates symlinks only with Developer Mode or elevation.
        pytest.skip(f"this account cannot create symlinks: {error}")
    with pytest.raises(OutputError, match="escapes"):
        read_snapshot(generation)


def test_restore_failure_removes_only_its_staging(generation, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("restore copy failed")

    monkeypatch.setattr(module, "_copy_file", fail)
    with pytest.raises(OutputError, match="restore copy failed"):
        restore_snapshot(generation, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".restored-*"))


def test_unremovable_staging_counts_against_capacity(working, monkeypatch, caplog):
    output, controller = working
    (controller._directory / ".snapshot-incomplete").mkdir()
    (controller._directory / ".snapshot-other").mkdir()
    take(working)
    assert take(working).disposition == "deferred"
    monkeypatch.setattr(
        module.shutil, "rmtree", lambda *args: (_ for _ in ()).throw(OSError("busy"))
    )
    module._discard_staging(controller._directory / ".snapshot-incomplete")
    assert "incomplete snapshot remains" in caplog.text
    assert output._file is not None


def test_unrelated_manifest_does_not_make_a_directory_immutable(tmp_path):
    (tmp_path / "manifest.json").write_text("invalid", encoding="utf-8")
    lease = WriterLock(tmp_path / "file")
    lease.acquire()
    lease.release()
    lease.release()
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    lease.acquire()
    lease.release()
