# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Publish immutable, verified checkpoints while a propagation stays alive.

Only closed files are copied. The driver pauses at a committed accepted endpoint;
the two writers retain their advisory locks and reopen in the same session. Cloud
transport and retirement belong to the caller. See ``docs/guide/results/output.md``.
"""

import hashlib
import json
import logging
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np

from solprop._interrupts import deferred_handlers
from solprop.exceptions import OutputError, TDSEArchiveError
from solprop.output._writer_lock import WriterLock
from solprop.output.hdf5 import HDF5Output, ResumeState, resume
from solprop.output.schema import RunMetadata
from solprop.output.tdse_archive import (
    ArchiveOutputs,
    TDSEArchive,
    TDSEArchiveWriter,
    open_tdse_archive,
    structured_json,
)

__all__ = [
    "SnapshotController",
    "SnapshotFiles",
    "SnapshotInfo",
    "SnapshotResult",
    "read_snapshot",
    "restore_snapshot",
]

logger = logging.getLogger(__name__)
_SCHEMA = "solprop.snapshot.v1"
_ANNOTATION = "snapshot_checkpoint"
_BUFFER_BYTES = 1024 * 1024
_RETRY_SECONDS = 60.0


@dataclass(frozen=True)
class SnapshotResult:
    """Describe the most recent snapshot attempt.

    Attributes
    ----------
    disposition : {'published', 'deferred', 'failed'}
        Whether a generation was published, capacity was full, or copying failed.
    generation : pathlib.Path or None
        Published directory, if one exists, including after a reopening failure.
    z : float
        Accepted checkpoint position, m.
    bytes_copied : int
        Successfully copied file bytes, excluding the manifest.
    copy_seconds : float
        Time spent copying and verifying files, s.
    pause_seconds : float
        Total time spent servicing this request, including reopening, s.
    failure : str or None
        Failure description, absent on success or capacity deferral.
    """

    disposition: Literal["published", "deferred", "failed"]
    generation: Path | None
    z: float
    bytes_copied: int
    copy_seconds: float
    pause_seconds: float
    failure: str | None = None


@dataclass(frozen=True)
class SnapshotFiles:
    """Locate a verified checkpoint's optical file and optional archive.

    Attributes
    ----------
    output : pathlib.Path
        Optical HDF5 file; restored files may be passed to ``Simulation.run``.
    tdse_archive : TDSEArchive or None
        Matching archive request, preserving its output selection.
    """

    output: Path
    tdse_archive: TDSEArchive | None


@dataclass(frozen=True)
class SnapshotInfo:
    """Describe a verified immutable generation.

    Attributes
    ----------
    directory : pathlib.Path
        Published generation directory.
    files : SnapshotFiles
        Read-only files; use ``restore_snapshot`` before continuing them.
    generation : str
        Unique generation identifier.
    z : float
        Accepted position, m.
    n_saved : int
        Number of committed optical observations.
    next_save_index : int
        Next scheduled save beyond this endpoint.
    bytes_copied : int
        Sum of the optical and archive file sizes, bytes.
    configuration_hash : str
        Scientific configuration identity.
    """

    directory: Path
    files: SnapshotFiles
    generation: str
    z: float
    n_saved: int
    next_save_index: int
    bytes_copied: int
    configuration_hash: str


@dataclass(frozen=True)
class _FileRecord:
    """Identify one copied file.

    Attributes
    ----------
    role : str
        Optical or archive role.
    path : str
        Relative filename within the generation.
    size : int
        File bytes.
    sha256 : str
        Hexadecimal checksum.
    """

    role: str
    path: str
    size: int
    sha256: str


class SnapshotController:
    """Request bounded local snapshots at committed accepted steps.

    Parameters
    ----------
    directory : pathlib.Path or str
        Store containing immutable generations. One active run owns a store.
    interval_seconds : float or None, optional
        Positive finite monotonic-clock interval, s; None enables explicit requests
        only. The clock begins after preparation and restarts after publication.
    max_snapshots : int, optional
        Maximum generation count, including incomplete copies; a positive integer,
        three by default. A full store defers requests without stopping propagation.

    Raises
    ------
    OutputError
        If the interval or capacity is invalid.

    Examples
    --------
    >>> from solprop.output import SnapshotController
    >>> snapshots = SnapshotController("snapshots", interval_seconds=1800)
    >>> snapshots.request()  # serviced after the next committed accepted step
    >>> snapshots.last_result is None
    True
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        interval_seconds: float | None = None,
        max_snapshots: int = 3,
    ) -> None:
        if interval_seconds is not None and (
            isinstance(interval_seconds, bool)
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise OutputError("snapshot interval_seconds must be positive and finite")
        if (
            isinstance(max_snapshots, bool)
            or not isinstance(max_snapshots, int)
            or max_snapshots < 1
        ):
            raise OutputError("snapshot max_snapshots must be a positive integer")
        self._directory = Path(directory).resolve()
        self._interval = interval_seconds
        self._capacity = max_snapshots
        self._lease = WriterLock(self._directory / "store")
        self._active = False
        self._requested = 0
        self._attempted = 0
        self._pending = False
        self._last_success = 0.0
        self._retry_at = 0.0
        self._last_result: SnapshotResult | None = None

    @property
    def last_result(self) -> SnapshotResult | None:
        """Return the latest attempt, or None before any request is serviced.

        Returns
        -------
        SnapshotResult or None
            The result, including total pause time and bytes copied.
        """
        return self._last_result

    def request(self) -> None:
        """Request a snapshot without I/O, waiting or acquiring a lock.

        Suitable for a Python signal handler in the propagation thread. Multiple
        requests coalesce; one arriving during a copy remains pending afterwards.
        """
        self._requested += 1

    def _start(self) -> None:
        """Own the store for this run, starting the interval after preparation."""
        if self._active:
            raise OutputError("snapshot controller already belongs to an active run")
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lease.acquire()
        self._active = True
        self._last_success = time.monotonic()

    def _finish(self) -> None:
        """Release only this controller's store ownership."""
        self._lease.release()
        self._active = False

    def _due(self) -> bool:
        """Coalesce interval and explicit requests, honouring failure backoff.

        Returns
        -------
        bool
            Whether an attempt is due at this boundary.
        """
        now = time.monotonic()
        explicit = self._requested != self._attempted
        if explicit or (
            self._interval is not None and now - self._last_success >= self._interval
        ):
            self._pending = True
        return self._pending and (explicit or now >= self._retry_at)

    def _service(
        self,
        output: HDF5Output,
        archive: TDSEArchiveWriter | None,
        *,
        next_save_index: int,
        require_history: bool,
        persist: Callable[[], None],
        provenance: RunMetadata | None = None,
    ) -> None:
        """Publish one requested endpoint, reopening both writers before returning.

        Parameters
        ----------
        output : HDF5Output
            Live optical writer at a committed accepted endpoint.
        archive : TDSEArchiveWriter or None
            Optional matching live archive.
        next_save_index : int
            Index of the next scheduled save beyond this endpoint.
        require_history : bool
            Whether Adams history must be present.
        persist : callable
            Persist the retained march and running execution record without observing.
        provenance : RunMetadata or None, optional
            Current execution's version and Git record. Defaults to the writer's
            metadata; the driver supplies its current record when resuming.
        """
        if not self._due():
            return
        assert output.outcome is not None
        provenance = output.metadata if provenance is None else provenance
        assert provenance is not None
        z = output.outcome.z
        started = time.monotonic()
        requested = self._requested
        self._attempted = requested
        self._retry_at = started + _RETRY_SECONDS
        count = sum(
            path.name.startswith(("snapshot-", ".snapshot-"))
            for path in self._directory.iterdir()
        )
        if count >= self._capacity:
            self._last_result = SnapshotResult(
                "deferred", None, z, 0, 0.0, time.monotonic() - started
            )
            logger.warning(
                "snapshot deferred at z=%g m: store capacity %d reached",
                z,
                self._capacity,
            )
            return
        generation = "snapshot-" + uuid.uuid4().hex
        publication = _Publication(None, 0, 0.0, None)
        failure: BaseException | None = None
        # TERM must not escape while either file is closed, including cleanup.
        with deferred_handlers():
            try:
                persist()
                output.annotate(
                    _ANNOTATION,
                    {
                        "generation": generation,
                        "z": z,
                        "next_save_index": next_save_index,
                        "requires_march": require_history,
                        "versions": dict(provenance.versions),
                        "git": None
                        if provenance.git is None
                        else asdict(provenance.git),
                    },
                )
                with _suspended_writers(output, archive):
                    publication = _copy_generation(
                        self._directory,
                        generation,
                        output.path,
                        None if archive is None else archive.path,
                    )
                failure = publication.failure
            except BaseException as error:
                failure = error
                if publication.failure is not None:
                    error.add_note(
                        f"snapshot publication also failed: {publication.failure}"
                    )
                logger.error("snapshot transaction failed at z=%g m: %s", z, error)
                raise
            finally:
                self._last_result = SnapshotResult(
                    "published"
                    if publication.path is not None and failure is None
                    else "failed",
                    publication.path,
                    z,
                    publication.size,
                    publication.seconds,
                    time.monotonic() - started,
                    None if failure is None else f"{type(failure).__name__}: {failure}",
                )
            if failure is not None:
                self._retry_at = time.monotonic() + _RETRY_SECONDS
                logger.warning("snapshot publication failed at z=%g m: %s", z, failure)
            else:
                assert publication.path is not None
                self._last_success = time.monotonic()
                self._pending = self._requested != requested
                logger.info(
                    "snapshot %s: %d bytes, %.6f s copying, %.6f s paused",
                    publication.path,
                    publication.size,
                    publication.seconds,
                    self._last_result.pause_seconds,
                )


@dataclass(frozen=True)
class _Publication:
    """Retain publication details even if working writers subsequently fail.

    Attributes
    ----------
    path : pathlib.Path or None
        Published generation, if the rename succeeded.
    size : int
        Verified copied bytes.
    seconds : float
        Copy and verification duration, s.
    failure : BaseException or None
        Copy or publication failure, if any.
    """

    path: Path | None
    size: int
    seconds: float
    failure: BaseException | None


@contextmanager
def _suspended_writers(
    optical: HDF5Output,
    archive: TDSEArchiveWriter | None,
) -> Iterator[None]:
    """Reopen every suspended writer, preserving the primary failure on cleanup.

    Parameters
    ----------
    optical : HDF5Output
        Optical writer to suspend.
    archive : TDSEArchiveWriter or None
        Optional archive writer to suspend after the optical writer.

    Yields
    ------
    None
        With both HDF5 handles closed and writer ownership retained.
    """
    suspended: list[HDF5Output | TDSEArchiveWriter] = []
    primary: BaseException | None = None
    try:
        for writer in (optical, archive):
            if writer is not None:
                writer.suspend()
                suspended.append(writer)
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        reopening: BaseException | None = None
        for writer in reversed(suspended):
            try:
                writer.reopen()
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"writer reopening also failed: {error}")
                elif reopening is None:
                    reopening = error
                else:
                    reopening.add_note(f"another writer could not reopen: {error}")
        if reopening is not None:
            raise reopening


def _copy_generation(
    store: Path,
    generation: str,
    optical: Path,
    archive: Path | None,
) -> _Publication:
    """Copy closed files and publish once; recover from copy/publication failures.

    Parameters
    ----------
    store : pathlib.Path
        Owned snapshot store.
    generation : str
        Unique generation name.
    optical, archive : pathlib.Path or None
        Closed working files; only the archive may be absent.

    Returns
    -------
    _Publication
        Publication path, copied bytes, elapsed seconds and any failure.
    """
    staging = store / ("." + generation)
    started = time.monotonic()
    copied: list[_FileRecord] = []
    published = None
    failure = None
    created = False
    try:
        staging.mkdir()
        created = True
        paths = [optical.resolve()]
        if archive is not None:
            paths.append(archive.resolve())
        common = Path(os.path.commonpath([path.parent for path in paths]))
        for role, source in zip(("optical", "archive"), paths, strict=False):
            relative = Path("files") / source.relative_to(common)
            copied.append(_copy_file(source, staging / relative, role, relative))
        checkpoint = _checkpoint(staging, copied)
        manifest = {
            "schema": _SCHEMA,
            "generation": generation,
            "created": datetime.now(UTC).isoformat(),
            "checkpoint": checkpoint,
            "files": [asdict(record) for record in copied],
        }
        _publish_manifest(staging, manifest)
        destination = store / generation
        staging.rename(destination)
        published = destination
        _sync_directory(store)
    except (OSError, OutputError, TDSEArchiveError, ValueError) as error:
        failure = error
    finally:
        if created and staging.exists():
            _discard_staging(staging)
    return _Publication(
        published, sum(row.size for row in copied), time.monotonic() - started, failure
    )


def _digest(path: Path) -> str:
    """Hash a closed file without loading it into memory.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _copy_file(
    source: Path, destination: Path, role: str, relative: Path
) -> _FileRecord:
    """Copy and independently verify a closed source with bounded host buffers.

    Parameters
    ----------
    source, destination : pathlib.Path
        Closed source and new destination file.
    role : str
        Optical or archive role in the pair.
    relative : pathlib.Path
        Destination relative to its generation directory.

    Returns
    -------
    _FileRecord
        Verified filename, size in bytes and SHA-256.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while block := incoming.read(_BUFFER_BYTES):
            outgoing.write(block)
            digest.update(block)
            size += len(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    checksum = digest.hexdigest()
    if destination.stat().st_size != size or _digest(destination) != checksum:
        raise OutputError(f"snapshot copy checksum mismatch: {relative}")
    return _FileRecord(role, relative.as_posix(), size, checksum)


def _sync_directory(path: Path) -> None:
    """Persist directory entries where the platform provides directory fsync.

    Parameters
    ----------
    path : pathlib.Path
        Directory whose entries must be durable.
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_manifest(directory: Path, manifest: Mapping[str, object]) -> None:
    """Publish the manifest only after every copied file is durable and read-only.

    Parameters
    ----------
    directory : pathlib.Path
        Private staging generation.
    manifest : mapping
        Verified identities, counters and checksums to publish last.
    """
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        else:
            _sync_directory(path)
    temporary = directory / ".manifest-writing"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(structured_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o444)
    temporary.rename(directory / "manifest.json")
    _sync_directory(directory)


def _discard_staging(path: Path) -> None:
    """Remove only this attempt's staging data, retaining failures against capacity.

    Parameters
    ----------
    path : pathlib.Path
        Staging directory owned by the failed attempt.
    """
    try:
        # Windows cannot unlink read-only files; publication may have reached chmod.
        for item in path.rglob("*"):
            if item.is_file():
                item.chmod(0o600)
        shutil.rmtree(path)
    except OSError as error:
        logger.warning("incomplete snapshot remains at %s: %s", path, error)


def _object(value: object) -> dict[str, object]:
    """Require a JSON object, keeping untrusted manifest values explicit.

    Parameters
    ----------
    value : object
        Decoded JSON or structured provenance value.

    Returns
    -------
    dict
        Mapping with string keys.
    """
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OutputError("snapshot manifest requires an object")
    return dict(cast("Mapping[str, object]", value))


def _file_records(manifest: Mapping[str, object]) -> list[_FileRecord]:
    """Validate the small manifest file table before touching any named path.

    Parameters
    ----------
    manifest : mapping
        Decoded manifest containing a file table.

    Returns
    -------
    list of _FileRecord
        Optical file followed by the optional archive.
    """
    values = manifest.get("files")
    if not isinstance(values, list) or len(values) not in (1, 2):
        raise OutputError("snapshot must contain an optical file and optional archive")
    records: list[_FileRecord] = []
    for value in values:
        row = _object(value)
        role, path, size, checksum = (
            row.get(key) for key in ("role", "path", "size", "sha256")
        )
        if (
            not isinstance(role, str)
            or role not in ("optical", "archive")
            or not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum)
        ):
            raise OutputError("invalid snapshot file record")
        records.append(_FileRecord(role, path, size, checksum))
    if [record.role for record in records] not in (["optical"], ["optical", "archive"]):
        raise OutputError("snapshot file roles must be optical, then optional archive")
    if len({record.path for record in records}) != len(records):
        raise OutputError("snapshot files must have distinct paths")
    return records


def _checkpoint(directory: Path, records: list[_FileRecord]) -> dict[str, object]:
    """Cross-check the copied optical endpoint and all referenced archive identities.

    Parameters
    ----------
    directory : pathlib.Path
        Generation containing closed files.
    records : list of _FileRecord
        Validated file table.

    Returns
    -------
    dict
        Checkpoint identity and provenance checked against the actual files.
    """
    optical = directory / records[0].path
    state = resume(optical)
    outcome = state.outcome
    if (
        outcome is None
        or outcome.status != "running"
        or outcome.z != state.plane.z
        or state.checkpoint is not None
        or state.plane.field is None
        or not np.all(np.isfinite(state.plane.field))
    ):
        raise OutputError(
            "snapshot requires a running, finite accepted optical endpoint"
        )
    with HDF5Output(optical, readonly=True) as stored:
        marker = _object(stored.annotations.get(_ANNOTATION))
    next_index = marker.get("next_save_index")
    if (
        marker.get("z") != state.plane.z
        or not isinstance(marker.get("generation"), str)
        or type(marker.get("requires_march")) is not bool
        or type(next_index) is not int
        or next_index < 0
    ):
        raise OutputError(
            "snapshot checkpoint marker disagrees with the accepted endpoint"
        )
    if marker["requires_march"] and (
        state.march is None or state.march.z != state.plane.z
    ):
        raise OutputError("snapshot is missing the accepted Adams history")
    execution = _object(json.loads(structured_json(outcome.execution)))
    tdse = _object(execution.get("tdse", {}))
    reference = tdse.get("archive")
    archive_record: dict[str, object] | None = None
    if reference is not None:
        archive_record = _archive_checkpoint(directory, records, state, reference)
    elif len(records) != 1:
        raise OutputError("snapshot has an unreferenced archive")
    return _object(
        json.loads(
            structured_json(
                {
                    **marker,
                    "n_saved": state.n_saved,
                    "configuration_hash": state.metadata.configuration_hash(),
                    "provider": state.metadata.tdse,
                    "outcome": asdict(outcome),
                    "archive": archive_record,
                    "versions": marker.get("versions", dict(state.metadata.versions)),
                    "git": marker.get("git", state.metadata.git),
                }
            )
        )
    )


def _archive_checkpoint(
    directory: Path, records: list[_FileRecord], state: ResumeState, reference: object
) -> dict[str, object]:
    """Verify that the archive belongs to this accepted optical checkpoint.

    Parameters
    ----------
    directory : pathlib.Path
        Generation containing the pair.
    records : list of _FileRecord
        Optical and archive file table.
    state : ResumeState
        Accepted optical endpoint.
    reference : object
        Archive reference from the optical execution record.

    Returns
    -------
    dict
        Matching archive identity, session, counters and context.
    """
    optical = directory / records[0].path
    if len(records) != 2:
        raise OutputError("snapshot omits the referenced TDSE archive")
    ref = _object(reference)
    archive_path = directory / records[1].path
    if ref.get("relative_to") != "optical_file" or not isinstance(ref.get("path"), str):
        raise OutputError("snapshot requires a relative TDSE archive reference")
    if (optical.parent / cast("str", ref["path"])).resolve() != archive_path.resolve():
        raise OutputError(
            "snapshot archive filename does not match the optical reference"
        )
    with open_tdse_archive(archive_path) as archive:
        sessions = archive.sessions
        if not sessions or archive.run_uuid != ref.get("run_uuid"):
            raise OutputError("snapshot archive UUID mismatch")
        session = sessions[-1]
        if (
            session.index != ref.get("session")
            or session.run_uuid != archive.run_uuid
            or session.closed
        ):
            raise OutputError("snapshot archive session mismatch")
        if any(
            session.totals.get(name) != ref.get(name)
            for name in (
                "evaluations",
                "array_bytes",
                "bytes_written",
                "write_seconds",
            )
        ):
            raise OutputError("snapshot archive counters mismatch")
        context = session.context
        if context.get(
            "optical_configuration_hash"
        ) != state.metadata.configuration_hash() or context.get(
            "descriptor_identity"
        ) != _object(state.metadata.tdse).get("identity"):
            raise OutputError(
                "snapshot archive configuration/provider identity mismatch"
            )
        evaluations = [
            item for item in archive.evaluations() if item.session == session.index
        ]
        if len(evaluations) != ref.get("evaluations") or any(
            not item.complete or (item.trial is not None and item.trial_outcome is None)
            for item in evaluations
        ):
            raise OutputError(
                "snapshot archive has incomplete or mismatched evaluations/trials"
            )
        return {
            "run_uuid": archive.run_uuid,
            "session": session.index,
            "outputs": session.outputs,
            "evaluations": len(evaluations),
            "trials": {str(key): value for key, value in session.trials.items()},
            "context": context,
        }


def read_snapshot(directory: Path | str) -> SnapshotInfo:
    """Verify a published generation's checksums, identities and accepted state.

    Parameters
    ----------
    directory : pathlib.Path or str
        Completed generation containing ``manifest.json``.

    Returns
    -------
    SnapshotInfo
        Verified read-only file locations and checkpoint identity.

    Raises
    ------
    OutputError
        If publication is incomplete, a checksum differs, or files disagree.
    """
    directory = Path(directory).resolve()
    try:
        if directory.name.startswith("."):
            raise OutputError("incomplete snapshot staging directory")
        manifest = _object(
            json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        )
        if manifest.get("schema") != _SCHEMA:
            raise OutputError("unsupported snapshot manifest schema")
        records = _file_records(manifest)
        for record in records:
            path = directory / record.path
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise OutputError("snapshot file escapes its generation")
            if path.stat().st_size != record.size or _digest(path) != record.sha256:
                raise OutputError(f"snapshot checksum mismatch: {record.path}")
        checkpoint = _checkpoint(directory, records)
        if checkpoint != manifest.get("checkpoint") or checkpoint[
            "generation"
        ] != manifest.get("generation"):
            raise OutputError("snapshot manifest and checkpoint identities disagree")
        archive = checkpoint["archive"]
        request = (
            None
            if archive is None
            else TDSEArchive(
                directory / records[1].path,
                outputs=cast("ArchiveOutputs", _object(archive)["outputs"]),
            )
        )
        return SnapshotInfo(
            directory,
            SnapshotFiles(directory / records[0].path, request),
            cast("str", manifest["generation"]),
            cast("float", checkpoint["z"]),
            cast("int", checkpoint["n_saved"]),
            cast("int", checkpoint["next_save_index"]),
            sum(record.size for record in records),
            cast("str", checkpoint["configuration_hash"]),
        )
    except (OSError, ValueError, KeyError, TypeError, TDSEArchiveError) as error:
        raise OutputError(f"cannot verify snapshot {directory}: {error}") from error


def restore_snapshot(directory: Path | str, destination: Path | str) -> SnapshotFiles:
    """Restore a verified snapshot into a new writable working directory.

    Parameters
    ----------
    directory : pathlib.Path or str
        Immutable published generation; never modified.
    destination : pathlib.Path or str
        New directory, which must not exist. Files keep their relative layout.

    Returns
    -------
    SnapshotFiles
        Paths to pass to ``Simulation.run(..., resume=True)`` in a fresh process.

    Raises
    ------
    OutputError
        If verification fails, the destination exists, or restoration fails.
    """
    info = read_snapshot(directory)
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(info.directory):
        raise OutputError(
            "snapshot restoration needs a new directory outside the generation"
        )
    staging = destination.with_name("." + destination.name + "-" + uuid.uuid4().hex)
    created = False
    try:
        staging.mkdir(parents=True)
        created = True
        manifest = _object(
            json.loads((info.directory / "manifest.json").read_text(encoding="utf-8"))
        )
        records = _file_records(manifest)
        for record in records:
            relative = Path(record.path)
            copied = _copy_file(
                info.directory / relative, staging / relative, record.role, relative
            )
            if copied != record:
                raise OutputError("snapshot changed during restoration")
        if _checkpoint(staging, records) != manifest["checkpoint"]:
            raise OutputError("restored checkpoint differs from its manifest")
        if destination.exists():
            raise OutputError("snapshot restore destination appeared during copying")
        staging.rename(destination)
        _sync_directory(destination.parent)
    except (OSError, ValueError) as error:
        raise OutputError(f"cannot restore snapshot: {error}") from error
    finally:
        if created and staging.exists():
            _discard_staging(staging)
    output = destination / info.files.output.relative_to(info.directory)
    request = info.files.tdse_archive
    archive = (
        None
        if request is None
        else TDSEArchive(
            destination / request.path.relative_to(info.directory),
            outputs=request.outputs,
        )
    )
    return SnapshotFiles(output, archive)
