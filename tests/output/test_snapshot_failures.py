# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise snapshot transaction failures without replacing the primary exception."""

import dataclasses
import json
import signal
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from tests.output import test_snapshots as cases
from tests.tdse_archive_cases import begin, writer

from solprop._interrupts import deferred_handlers
from solprop.exceptions import OutputError, TDSEArchiveError
from solprop.output import (
    HDF5Output,
    _writer_lock,
    read_snapshot,
    restore_snapshot,
    resume,
)
from solprop.output import snapshots as module

working = cases.working
generation = cases.generation
take = cases.take


def fail(message):
    def operation(*args, **kwargs):
        raise OutputError(message)

    return operation


def test_lock_failure_closes_the_unused_descriptor(tmp_path, monkeypatch):
    monkeypatch.setattr(_writer_lock, "lock", fail("lock failed"))
    lease = _writer_lock.WriterLock(tmp_path / "file")
    with pytest.raises(OutputError, match="lock failed"):
        lease.acquire()
    assert lease._descriptor is None


@pytest.mark.parametrize("operation", ["open", "lock"])
def test_writer_lock_os_failure_uses_the_output_error_hierarchy(
    tmp_path, monkeypatch, operation
):
    def failed(*args, **kwargs):
        raise OSError("ownership unavailable")

    target = _writer_lock.os if operation == "open" else _writer_lock
    monkeypatch.setattr(target, operation, failed)
    lease = _writer_lock.WriterLock(tmp_path / "file")
    with pytest.raises(OutputError, match="ownership unavailable"):
        lease.acquire()
    assert lease._descriptor is None


def test_copy_and_reopening_failures_preserve_both_reasons(working, monkeypatch):
    output, _ = working
    monkeypatch.setattr(module, "_copy_file", fail("copy failed"))
    monkeypatch.setattr(output, "reopen", fail("reopen failed"))
    with pytest.raises(OutputError, match="reopen failed") as error:
        take(working)
    assert "copy failed" in str(error.value.__notes__)


@pytest.mark.parametrize("suspend", [True, False])
def test_both_writers_are_cleaned_after_one_fails(
    working, tmp_path, monkeypatch, suspend
):
    output, _ = working
    archive = writer(tmp_path / "archive.h5")
    monkeypatch.setattr(output, "reopen", fail("optical reopen"))
    monkeypatch.setattr(
        archive, "suspend" if suspend else "reopen", fail("archive failure")
    )
    try:
        with (
            pytest.raises(OutputError, match="archive failure") as error,
            module._suspended_writers(output, archive),
        ):
            pass
        assert "optical reopen" in str(error.value.__notes__)
    finally:
        archive.close()


def test_failed_stage_allocation_never_removes_an_existing_generation(working):
    output, controller = working
    existing = controller._directory / ".snapshot-occupied"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("owned elsewhere", encoding="utf-8")
    output.suspend()
    try:
        result = module._copy_generation(
            controller._directory, "snapshot-occupied", output.path, None
        )
    finally:
        output.reopen()
    assert result.failure is not None and sentinel.exists()


def test_manifest_failure_after_rename_leaves_a_verifiable_generation(
    working, monkeypatch
):
    _, controller = working
    original = module._sync_directory

    def sync(path):
        if path == controller._directory:
            raise OSError("directory sync failed")
        original(path)

    monkeypatch.setattr(module, "_sync_directory", sync)
    result = take(working)
    assert result.disposition == "failed" and result.generation is not None
    read_snapshot(result.generation)


def test_directory_fsync_is_skipped_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    module._sync_directory(tmp_path)


@pytest.mark.parametrize("change", ["copy", "checkpoint", "destination"])
def test_restore_revalidates_copied_files_and_does_not_overwrite(
    generation, tmp_path, monkeypatch, change
):
    destination = tmp_path / "restored"
    original = module._copy_file

    def copy(*args):
        result = original(*args)
        if change == "copy":
            return dataclasses.replace(result, sha256="0" * 64)
        if change == "checkpoint":
            monkeypatch.setattr(module, "_checkpoint", lambda *args: {})
        else:
            destination.mkdir(exist_ok=True)
        return result

    monkeypatch.setattr(module, "_copy_file", copy)
    with pytest.raises(OutputError, match=r"changed|differs|appeared"):
        restore_snapshot(generation, destination)
    assert not list(tmp_path.glob(".restored-*"))
    if destination.exists():
        assert not list(destination.iterdir())


def test_unexpected_manifest_shape_is_refused(generation):
    manifest = generation / "manifest.json"
    manifest.chmod(0o600)
    manifest.write_text("null", encoding="utf-8")
    with pytest.raises(OutputError, match="object"):
        read_snapshot(generation)


@pytest.mark.parametrize(
    "change", ["outcome", "status", "position", "history", "marker"]
)
def test_checkpoint_validation_is_independent_of_checksums(
    generation, monkeypatch, change
):
    info = read_snapshot(generation)
    original = resume(info.files.output)
    assert original.outcome is not None
    if change == "outcome":
        state = dataclasses.replace(original, outcome=None)
    elif change == "status":
        state = dataclasses.replace(
            original, outcome=dataclasses.replace(original.outcome, status="complete")
        )
    elif change == "position":
        state = dataclasses.replace(
            original, outcome=dataclasses.replace(original.outcome, z=0.2)
        )
    else:
        state = original
        info.files.output.chmod(0o600)
        with h5py.File(info.files.output, "r+") as file:
            key = "meta/annotations/snapshot_checkpoint"
            marker = {
                "generation": info.generation,
                "z": info.z,
                "next_save_index": 0,
                "requires_march": change == "history",
            }
            if change == "marker":
                marker["next_save_index"] = -1
            del file[key]
            file.create_dataset(key, data=json.dumps(marker))
    monkeypatch.setattr(module, "resume", lambda path: state)
    records = module._file_records(
        json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    )
    with pytest.raises(OutputError, match=r"endpoint|history"):
        module._checkpoint(generation, records)


def test_snapshot_rejects_an_unreferenced_archive(generation):
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    records = module._file_records(manifest)
    records.append(dataclasses.replace(records[0], role="archive", path="extra.h5"))
    with pytest.raises(OutputError, match="unreferenced"):
        module._checkpoint(generation, records)


def test_custom_signal_handler_waits_until_both_files_are_open(working, monkeypatch):
    output, _ = working
    called = []

    def own(number, frame):
        called.append(output._file is not None)

    previous = signal.signal(signal.SIGTERM, own)
    copy = module._copy_file

    def during(*args):
        signal.raise_signal(signal.SIGTERM)
        assert not called
        return copy(*args)

    monkeypatch.setattr(module, "_copy_file", during)
    try:
        assert take(working).disposition == "published"
        assert called == [True]
        assert signal.getsignal(signal.SIGTERM) is own
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_deferred_handlers_leave_a_worker_threads_handlers_alone():
    called = []

    def work():
        with deferred_handlers():
            called.append(True)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert called == [True]


def test_archive_suspension_keeps_ownership_and_session(tmp_path):
    archive = writer(tmp_path / "archive.h5")
    try:
        with pytest.raises(TDSEArchiveError, match="not suspended"):
            archive.reopen()
        archive.suspend()
        with pytest.raises(TDSEArchiveError, match="suspended"):
            archive.ensure_writable()
        with pytest.raises(TDSEArchiveError, match="another writer"):
            writer(archive.path, kind="resume")
        archive.reopen()
        assert archive.reference()["session"] == 0
        begin(archive, np.random.default_rng(0))
        with pytest.raises(TDSEArchiveError, match="open evaluation"):
            archive.suspend()
    finally:
        archive.close()


@pytest.mark.parametrize("change", ["inode", "session", "open"])
def test_archive_refuses_unsafe_reopening(tmp_path, monkeypatch, change):
    archive = writer(tmp_path / "archive.h5")
    archive.suspend()
    if change == "inode":
        contents = archive.path.read_bytes()
        # Renamed aside rather than deleted: a freed inode number can be reused at
        # once (ext4), which would make the replacement the same file to stat.
        archive.path.rename(tmp_path / "replaced.h5")
        archive.path.write_bytes(contents)
    elif change == "session":
        with h5py.File(archive.path, "r+") as file:
            file[archive._session_name].attrs["closed"] = True
    else:
        monkeypatch.setattr(h5py, "File", fail("cannot open"))
    try:
        with pytest.raises(
            (TDSEArchiveError, OutputError), match=r"replaced|changed|cannot open"
        ):
            archive.reopen()
    finally:
        archive.close()


def test_read_only_optical_writer_cannot_be_suspended(generation):
    info = read_snapshot(generation)
    with (
        HDF5Output(info.files.output, readonly=True) as output,
        pytest.raises(OutputError, match="reading only"),
    ):
        assert isinstance(output, HDF5Output)
        output.suspend()


@pytest.mark.parametrize("writer_kind", ["optical", "archive"])
def test_flush_failure_is_a_fatal_suspend_error(
    working, tmp_path, monkeypatch, writer_kind
):
    optical, _ = working
    archive = writer(tmp_path / "archive.h5")
    target = optical if writer_kind == "optical" else archive

    def flush_failure():
        raise OSError("flush failed")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(target._file, "flush", flush_failure)
            with pytest.raises(OutputError, match="flush failed"):
                target.suspend()
    finally:
        archive.close()


def test_missing_backend_history_stops_clearly(working):
    from solprop.driver.run import _MarchKeeper  # noqa: PLC0415 - failure seam only

    keeper = object.__new__(_MarchKeeper)
    keeper._kept_z = None
    keeper._policy = None
    keeper._save = None
    keeper._snapshot = None
    with pytest.raises(OutputError, match="backend and output"):
        keeper.keep(0.125, required=True)
    keeper._save = lambda state: None
    keeper._snapshot = lambda: None
    with pytest.raises(OutputError, match="accepted endpoint"):
        keeper.keep(0.125, required=True)


def test_snapshot_admission_rejects_invalid_controller_and_batch(working):
    import inspect  # noqa: PLC0415 - bypass annotations to check production refusal

    from solprop.driver.run import check_snapshot_request  # noqa: PLC0415
    from solprop.exceptions import DriverError  # noqa: PLC0415

    output, controller = working
    check = inspect.unwrap(check_snapshot_request)
    with pytest.raises(DriverError, match="SnapshotController"):
        check(None, output, object(), None, "all")
    with pytest.raises(DriverError, match="unbatched"):
        check(SimpleNamespace(shape=(2, 1, 1, 4)), output, controller, None, "all")


@pytest.mark.parametrize("primary", [True, False])
def test_run_cleanup_retains_primary_failure_and_attempts_every_resource(primary):
    import inspect  # noqa: PLC0415 - fault injection bypasses resource protocols

    from solprop.driver.simulation import _close_run  # noqa: PLC0415

    events = []

    def resource(name):
        def close():
            events.append(name)
            raise OutputError(name)

        return SimpleNamespace(close=close)

    original = OutputError("primary") if primary else None
    close = inspect.unwrap(_close_run)
    resources = [
        resource(name) for name in ("backend", "binding", "archive", "optical")
    ]
    if original is None:
        with pytest.raises(OutputError, match="backend") as caught:
            close(*resources, None)
        error = caught.value
    else:
        close(*resources, original)
        error = original
    assert events == ["backend", "binding", "archive", "optical"]
    assert "optical" in str(error.__notes__)


def test_long_failed_copy_backs_off_from_the_end_of_the_attempt(working, monkeypatch):
    _, controller = working
    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def copy(*args):
        clock[0] += 120
        raise OSError("slow copy failed")

    monkeypatch.setattr(module, "_copy_file", copy)
    assert take(working).disposition == "failed"
    assert not controller._due()
    clock[0] += 60
    assert controller._due()


def test_interrupted_archive_open_releases_its_writer_ownership(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(h5py, "File", interrupted)
        with pytest.raises(KeyboardInterrupt):
            writer(tmp_path / "archive.h5")
    writer(tmp_path / "archive.h5").close()


@pytest.mark.parametrize("kind", ["optical", "archive"])
@pytest.mark.parametrize("write_failure", [True, False])
def test_failed_close_retains_ownership_until_a_successful_retry(
    working, tmp_path, monkeypatch, kind, write_failure
):
    optical, _ = working
    target = optical if kind == "optical" else writer(tmp_path / "archive.h5")
    file = target._file

    def cannot_close():
        raise OSError("handle still open")

    with monkeypatch.context() as patch:
        patch.setattr(file, "close", cannot_close)
        if write_failure:
            if kind == "optical":
                patch.setattr(target, "_trim", fail("checkpoint write failed"))
            else:
                patch.setattr(file, "flush", fail("checkpoint write failed"))
        with pytest.raises(
            OutputError,
            match="checkpoint write failed" if write_failure else "handle still open",
        ) as caught:
            target.close()
        if write_failure:
            assert "handle still open" in str(caught.value.__notes__)
        assert target._writer_lock._descriptor is not None
        with pytest.raises(OutputError, match="another writer"):
            _writer_lock.WriterLock(target.path).acquire()
    target.close()
    assert target._writer_lock._descriptor is None


def test_annotation_cannot_release_a_suspended_writers_lock(working):
    output, _ = working
    output.suspend()
    try:
        with pytest.raises(OutputError, match="suspended"):
            output.annotate("external", {"status": "still computing"})
        assert output._writer_lock._descriptor is not None
    finally:
        output.reopen()


@pytest.mark.parametrize("kind", ["optical", "archive"])
def test_reopen_validation_retains_its_error_and_failed_handle_ownership(
    working, tmp_path, monkeypatch, kind
):
    output, _ = working
    target = output if kind == "optical" else writer(tmp_path / "archive.h5")
    target.suspend()
    with h5py.File(target.path, "r+") as file:
        if kind == "optical":
            file.attrs["n_saved"] = 100
        else:
            file[target._session_name].attrs["closed"] = True
    original = h5py.File

    def cannot_close():
        raise OSError("close rejected handle failed")

    with monkeypatch.context() as patch:

        def opened(*args, **kwargs):
            file = original(*args, **kwargs)
            patch.setattr(file, "close", cannot_close)
            return file

        patch.setattr(h5py, "File", opened)
        with pytest.raises((OutputError, TDSEArchiveError), match="changed") as caught:
            target.reopen()
        assert "close rejected handle failed" in str(caught.value.__notes__)
        assert target._writer_lock._descriptor is not None
        assert target._file is not None
    target.close()
    assert target._writer_lock._descriptor is None


@pytest.mark.parametrize("kind", ["optical", "archive"])
def test_closed_suspended_writer_cannot_reopen_without_ownership(
    working, tmp_path, kind
):
    output, _ = working
    target = output if kind == "optical" else writer(tmp_path / "archive.h5")
    target.suspend()
    target.close()
    try:
        with pytest.raises((OutputError, TDSEArchiveError), match="not suspended"):
            target.reopen()
    finally:
        target.close()


def test_append_adoption_failure_closes_the_opened_file(working, monkeypatch):
    output, _ = working
    output.close()
    appended = HDF5Output(output.path, append=True)
    opened = []
    original = h5py.File

    def track(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(h5py, "File", track)
    monkeypatch.setattr(appended, "_adopt_state", fail("invalid append state"))
    with pytest.raises(OutputError, match="invalid append state"):
        appended.open(output.metadata)
    assert len(opened) == 1 and not opened[0].id.valid
    assert appended._writer_lock._descriptor is None


def test_optical_reopen_io_failure_keeps_ownership_for_cleanup(working, monkeypatch):
    output, _ = working
    output.suspend()

    def failed(*args, **kwargs):
        raise OSError("reopen I/O failure")

    with monkeypatch.context() as patch:
        patch.setattr(h5py, "File", failed)
        with pytest.raises(OutputError, match=r"cannot reopen.*reopen I/O failure"):
            output.reopen()
        assert output._writer_lock._descriptor is not None
    output.reopen()


def test_archive_file_creation_error_releases_ownership(tmp_path, monkeypatch):
    path = tmp_path / "archive.h5"

    def failed(*args, **kwargs):
        raise OSError("archive creation failed")

    with monkeypatch.context() as patch:
        patch.setattr(h5py, "File", failed)
        with pytest.raises(TDSEArchiveError, match="archive creation failed"):
            writer(path)
    writer(path).close()


@pytest.mark.parametrize("primary", [True, False])
def test_archive_cleanup_preserves_a_preparation_error(primary):
    import inspect  # noqa: PLC0415 - inject a failed resource into private cleanup

    from solprop.driver.simulation import _close_tdse_archive  # noqa: PLC0415

    def failed():
        raise TDSEArchiveError("archive close failed")

    close = inspect.unwrap(_close_tdse_archive)
    close(None, None)
    resource = SimpleNamespace(close=failed)
    if primary:
        error = OutputError("provider preparation failed")
        close(resource, error)
        assert "archive close failed" in str(error.__notes__)
    else:
        with pytest.raises(TDSEArchiveError, match="archive close failed"):
            close(resource, None)


@pytest.mark.parametrize("close_failure", [False, True])
@pytest.mark.parametrize(
    "phase", ["cache", "flush", "session", "existing", "interrupt"]
)
def test_archive_initial_setup_closes_before_releasing_ownership(
    tmp_path, monkeypatch, phase, close_failure
):
    from solprop.output import tdse_archive as archive_module  # noqa: PLC0415

    path = tmp_path / "archive.h5"
    if phase == "existing":
        writer(path, run_uuid="other").close()
    opened = []
    leases = []
    file_type = h5py.File
    lock_type = _writer_lock.WriterLock

    def track_lock(path):
        lease = lock_type(path)
        leases.append(lease)
        return lease

    def broken(*args, **kwargs):
        if phase == "interrupt":
            raise KeyboardInterrupt("setup interrupted")
        raise OSError("setup failed")

    with monkeypatch.context() as patch:

        def track_file(*args, **kwargs):
            file = file_type(*args, **kwargs)
            opened.append(file)
            if close_failure:
                patch.setattr(file, "close", fail("close failed"))
            if phase == "flush":
                patch.setattr(file, "flush", broken)
            return file

        patch.setattr(archive_module, "WriterLock", track_lock)
        patch.setattr(h5py, "File", track_file)
        if phase in ("cache", "interrupt"):
            patch.setattr(archive_module, "_cap_metadata_cache", broken)
        elif phase == "session":
            patch.setattr(h5py.Group, "create_dataset", broken)
        expected = KeyboardInterrupt if phase == "interrupt" else TDSEArchiveError
        with pytest.raises(expected, match=r"setup|another run") as caught:
            writer(path, resume=phase == "existing")
        assert len(opened) == 1
        assert bool(opened[0].id.valid) == close_failure
        if close_failure:
            assert "close failed" in str(caught.value.__notes__)
            with pytest.raises(OutputError, match="another writer"):
                lock_type(path).acquire()
        else:
            assert leases[0]._descriptor is None
    # No writer is returned on setup failure; production retains an uncertain
    # lease until process exit. The fault-injection test owns and cleans both.
    opened[0].close()
    leases[0].release()


def test_append_adoption_and_cleanup_failure_keeps_primary_and_ownership(
    working, monkeypatch
):
    output, _ = working
    output.close()
    appended = HDF5Output(output.path, append=True)
    file_type = h5py.File
    with monkeypatch.context() as patch:

        def opened(*args, **kwargs):
            file = file_type(*args, **kwargs)
            patch.setattr(file, "close", fail("append close failed"))
            return file

        patch.setattr(h5py, "File", opened)
        patch.setattr(appended, "_adopt_state", fail("invalid append state"))
        with pytest.raises(OutputError, match="invalid append state") as caught:
            appended.open(output.metadata)
        assert "append close failed" in str(caught.value.__notes__)
        assert appended._file is not None and appended._file.id.valid
        with pytest.raises(OutputError, match="another writer"):
            _writer_lock.WriterLock(output.path).acquire()
    appended.close()
    assert appended._file is None
    assert appended._writer_lock._descriptor is None


@pytest.mark.parametrize("write_failure", [False, True])
def test_closed_annotation_close_failure_retains_ownership_and_is_retryable(
    working, monkeypatch, write_failure
):
    from solprop.output import hdf5 as optical_module  # noqa: PLC0415

    output, _ = working
    output.close()
    file_type = h5py.File
    with monkeypatch.context() as patch:

        def opened(*args, **kwargs):
            file = file_type(*args, **kwargs)
            patch.setattr(file, "close", fail("annotation close failed"))
            return file

        patch.setattr(h5py, "File", opened)
        if write_failure:
            patch.setattr(
                optical_module, "_write_annotation", fail("annotation write failed")
            )
        with pytest.raises(
            OutputError,
            match="annotation write failed"
            if write_failure
            else "annotation close failed",
        ) as caught:
            output.annotate("external", {"status": "stored"})
        if write_failure:
            assert "annotation close failed" in str(caught.value.__notes__)
        assert output.closed and output._file is not None
        assert output._file.id.valid
        with pytest.raises(OutputError, match="another writer"):
            _writer_lock.WriterLock(output.path).acquire()
    output.close()
    assert output._file is None and output._writer_lock._descriptor is None
