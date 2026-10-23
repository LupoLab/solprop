# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Drain durable point work across controllers sharing a filesystem."""

import dataclasses
import functools
import hashlib
import inspect
import json
import math
import pickle
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from solprop.driver._scan_model import Scan, ScanPoint
from solprop.driver.executors._queue_state import (
    atomic_json,
    finish,
    live_owner,
    locked,
    owner,
    read_json,
    state,
    sync_directory,
)
from solprop.driver.executors._request import ScanRequest
from solprop.driver.executors.pool import Pool
from solprop.exceptions import DriverError, OutputError
from solprop.output.hdf5 import _dataset, _group
from solprop.output.scan import PointResult, ScanWriter, _hash, _json
from solprop.output.scan_artifacts import artifact_digest, read_artifact, write_artifact
from solprop.output.schema import _decode
from solprop.processing.scans import ScanOutput

__all__ = ["Queue", "collect"]


def _point(job: object) -> ScanPoint:
    """Require a work description carrying a globally indexed ScanPoint.

    Parameters
    ----------
    job : object
        Point or work description carrying a globally indexed ScanPoint.

    Returns
    -------
    ScanPoint
        Validated global point description.
    """
    point = job if isinstance(job, ScanPoint) else getattr(job, "point", None)
    if not isinstance(point, ScanPoint):
        raise DriverError("Queue work must carry a ScanPoint")
    return point


def _point_id(point: ScanPoint) -> str:
    """Bind a claim to its global index, variable values and derived seed.

    Parameters
    ----------
    point : ScanPoint
        Globally indexed point with variable values and resolved seed.

    Returns
    -------
    str
        SHA-256 binding the point values, index and seed.
    """
    return _hash(_json(point))


def _filename(manifest: Mapping, index: int) -> str:
    """Keep artifact names independent of task and chunk ordering.

    Parameters
    ----------
    manifest : Mapping
        Immutable scan description and builder identity.
    index : int
        Zero-based global point index.

    Returns
    -------
    str
        Stable global point artifact filename.
    """
    return f"{manifest['name']}_{index:05d}.h5"


def _builder_key(build: object) -> str:
    """Identify serialised builder parameters and available script source.

    Parameters
    ----------
    build : object
        Pickleable builder, including captured construction parameters.

    Returns
    -------
    str
        Digest of serialised construction parameters and source bytes.
    """
    original = build
    while isinstance(original, functools.partial):
        original = original.func
    target = original if inspect.isfunction(original) else type(original)
    try:
        source = inspect.getsourcefile(target)
    except TypeError:
        source = None
    code = (
        Path(source).read_bytes()
        if source is not None and Path(source).is_file()
        else b""
    )
    return hashlib.sha256(pickle.dumps(build) + code).hexdigest()


def _restore_scan(manifest: Mapping) -> Scan:
    """Reconstruct portable axes without importing recorded classes.

    Parameters
    ----------
    manifest : Mapping
        Immutable scan description and builder identity.

    Returns
    -------
    Scan
        Full Cartesian scan reconstructed from portable axis metadata.
    """
    description = json.loads(manifest["description"])
    variables = {
        name: cast(np.ndarray, _decode(description["variables"][name]))
        for name in description["order"]
    }
    return Scan(description["name"], base_seed=description["base_seed"], **variables)


@dataclass(frozen=True)
class _Claim:
    """Bind work to one replaceable queue ownership token."""

    job: object
    token: str
    point_id: str


@dataclass(frozen=True)
class _QueueWork:
    """Publish worker results only while their ownership remains current."""

    path: Path
    work: Callable

    def probe(self, claim: _Claim) -> Mapping[str, int]:
        """Measure the wrapped worker before admitting local processes.

        Parameters
        ----------
        claim : _Claim
            Point description and unique ownership token.

        Returns
        -------
        mapping of str to int
            Baseline RSS, peak RSS and backend allocation bytes.
        """
        probe = getattr(self.work, "probe", None)
        if not callable(probe):
            raise DriverError("Queue requires the scan worker's memory probe")
        return cast(Mapping[str, int], probe(claim.job))

    def __call__(self, claim: _Claim) -> PointResult:
        """Compute and publish one claimed point.

        Parameters
        ----------
        claim : _Claim
            Point description and unique ownership token.

        Returns
        -------
        PointResult
            Published result, or skipped if ownership was superseded.
        """
        result = self.work(claim.job)
        if not isinstance(result, PointResult):
            raise DriverError("Queue work must return a PointResult")
        return self._publish(claim, result)

    def failure(self, claim: _Claim, error: str) -> PointResult:
        """Publish an abnormal worker exit as a retryable failed point.

        Parameters
        ----------
        claim : _Claim
            Point description and unique ownership token.
        error : str
            Captured worker failure or abnormal process exit.

        Returns
        -------
        PointResult
            Durably recorded worker failure.
        """
        return self._publish(
            claim, PointResult(_point(claim.job).index, "failed", error=error)
        )

    def _publish(self, claim: _Claim, result: PointResult) -> PointResult:
        """Commit an artifact only if its ownership token is still current.

        Parameters
        ----------
        claim : _Claim
            Point description and unique ownership token.
        result : PointResult
            Validated successful or failed worker result.

        Returns
        -------
        PointResult
            Published result, or skipped after ownership changed.
        """
        manifest = read_json(self.path / "manifest.json")
        index = _point(claim.job).index
        if result.index != index:
            raise OutputError("worker result index differs from its claim")
        temporary = (
            self.path / "points" / f".{index}.{claim.token}.{uuid.uuid4().hex}.h5"
        )
        identity = {
            "scan_id": manifest["scan_id"],
            "point_id": claim.point_id,
            "owner": claim.token,
        }
        try:
            write_artifact(temporary, result, identity)
            digest = artifact_digest(temporary)
            with locked(self.path / "state.lock"):
                record = state(self.path)
                row = record["rows"][index]
                if row.get("token") != claim.token or row["status"] != "running":
                    return PointResult(
                        index, "skipped", error="superseded queue ownership"
                    )
                destination = self.path / "points" / _filename(manifest, index)
                temporary.replace(destination)
                sync_directory(destination.parent)
                row.update(
                    status=result.status,
                    artifact_digest=digest,
                    configuration_hash=result.configuration_hash,
                    updated=time.time(),
                )
                finish(self.path, record)
            return result
        finally:
            temporary.unlink(missing_ok=True)


class _Heartbeat:
    """Refresh this controller's leases independently of point completion.

    Parameters
    ----------
    path : Path
        Queue directory or record filename used by this operation.
    controller : str
        Unique controller identifier whose leases must be refreshed.
    age : float
        Positive lease duration in seconds.
    """

    def __init__(self, path: Path, controller: str, age: float) -> None:
        self.path = path
        self.controller = controller
        self.interval = min(30.0, age / 3)
        self.stop = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._refresh, daemon=True)

    def _refresh(self) -> None:
        """Refresh owned leases and capture filesystem failures for the controller."""
        try:
            while not self.stop.wait(self.interval):
                with locked(self.path / "state.lock"):
                    record = state(self.path)
                    changed = False
                    for row in record["rows"]:
                        if (
                            row["status"] == "running"
                            and row.get("owner", {}).get("controller")
                            == self.controller
                        ):
                            row["updated"] = time.time()
                            changed = True
                    if changed:
                        atomic_json(self.path / "queue.json", record)
        except Exception:
            self.error = traceback.format_exc()

    def check(self) -> None:
        """Raise a captured heartbeat failure before dispatch continues."""
        if self.error is not None:
            raise DriverError(f"queue lease refresh failed:\n{self.error}")

    def close(self) -> None:
        """Stop the refresher and propagate any failure."""
        self.stop.set()
        self.thread.join()
        self.check()


@dataclass
class Queue:
    """Drain a shared durable queue using a local persistent process pool.

    Attributes
    ----------
    path : pathlib.Path, str or None
        Queue directory. None derives it from the collected output path for the CLI.
    stale_age : float
        Positive remote lease duration, s. Live local owners remain protected.
    pool : Pool
        Resource policy for workers within this controller.
    chunk : tuple of int or None
        Optional zero-based round-robin chunk number and count.
    """

    path: Path | str | None
    stale_age: float = 600.0
    pool: Pool = field(default_factory=Pool)
    chunk: tuple[int, int] | None = None
    _scan_id: str = field(init=False, default="", repr=False)
    _retry: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate lease duration and round-robin selection."""
        if sys.platform == "win32":
            # The lease protocol probes owners with os.kill(pid, 0), which on Windows
            # terminates the process instead, and fsyncs directories, which Windows
            # refuses: the shared queue is POSIX only.
            raise DriverError(
                "the Queue executor is not supported on Windows; use Local or Pool"
            )
        if not math.isfinite(self.stale_age) or self.stale_age <= 0:
            raise DriverError("queue stale_age must be positive and finite")
        if self.chunk is not None and (
            self.chunk[1] < 1 or not 0 <= self.chunk[0] < self.chunk[1]
        ):
            raise DriverError("chunk must satisfy 0 <= k < n")

    def validate(self, work: Callable) -> None:
        """Validate serialisation before preparing the durable queue.

        Parameters
        ----------
        work : Callable
            Pickleable point worker with its memory probe.
        """
        self.pool.validate(work)

    def prepare(self, request: ScanRequest) -> None:
        """Create or validate the manifest under an automatically released lock.

        Parameters
        ----------
        request : ScanRequest
            Scan identity, retention settings and worker to bind before dispatch.
        """
        scan, settings, output, work = (
            request.scan,
            request.settings,
            request.output,
            request.work,
        )
        retry_failed = request.retry_failed
        self.path = (
            Path(self.path)
            if self.path is not None
            else Path(output).with_suffix(".queue")
        )
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "points").mkdir(exist_ok=True)
        description = _json(
            {
                "name": scan.name,
                "variables": scan.variables,
                "order": list(scan.variables),
                "base_seed": scan.base_seed,
                "settings": settings,
            }
        )
        self._scan_id = _hash(description)
        self._retry = retry_failed
        manifest = {
            "queue_schema_version": 1,
            "scan_id": self._scan_id,
            "name": scan.name,
            "description": description,
            "builder": _builder_key(getattr(work, "build", work)),
        }
        with locked(self.path / "state.lock"):
            saved = self.path / "manifest.json"
            if saved.exists() and read_json(saved) != manifest:
                raise OutputError(
                    "queue scan, settings or builder differs from its manifest"
                )
            if not saved.exists():
                atomic_json(saved, manifest)
            if (
                not (self.path / "queue.json").exists()
                and not (self.path / ".done").exists()
            ):
                finish(
                    self.path,
                    {
                        "scan_id": self._scan_id,
                        "rows": [
                            {
                                "status": "todo",
                                "point_id": _point_id(scan.point(index)),
                                "updated": 0.0,
                            }
                            for index in range(scan.size)
                        ],
                    },
                )

    def _claims(
        self, points: Iterable, holder: Mapping, heartbeat: _Heartbeat
    ) -> Iterator[_Claim]:
        """Claim eligible points atomically while preserving live owners.

        Parameters
        ----------
        points : Iterable
            Lazily supplied point descriptions retaining global indices.
        holder : Mapping
            Controller host, process creation identity and unique identifier.
        heartbeat : _Heartbeat
            Running lease refresher whose failure stops further dispatch.

        Yields
        ------
        _Claim
            Eligible point carrying a newly published ownership token.
        """
        path = Path(cast(Path, self.path))
        for job in points:
            heartbeat.check()
            point = _point(job)
            if self.chunk is not None and point.index % self.chunk[1] != self.chunk[0]:
                continue
            with locked(path / "state.lock"):
                record = state(path)
                row = record["rows"][point.index]
                if row["point_id"] != _point_id(point):
                    raise OutputError("queue point values or seed changed")
                if (
                    row["status"] == "done"
                    or (row["status"] == "failed" and not self._retry)
                    or (row["status"] == "running" and live_owner(row, self.stale_age))
                ):
                    continue
                token = uuid.uuid4().hex
                row.update(
                    status="running",
                    token=token,
                    owner=dict(holder),
                    updated=time.time(),
                )
                atomic_json(path / "queue.json", record)
            yield _Claim(job, token, row["point_id"])

    def run[T, R](self, points: Iterable[T], work: Callable[[T], R]) -> Iterator[R]:
        """Run claimed points while a separate thread refreshes their leases.

        Parameters
        ----------
        points : Iterable[T]
            Lazily supplied point descriptions retaining global indices.
        work : Callable[[T], R]
            Pickleable point worker with its memory probe.

        Yields
        ------
        object
            Completed result from the prepared execution policy.
        """
        if not self._scan_id or self.path is None:
            raise DriverError("Queue must be prepared by run_scan before dispatch")
        path = Path(self.path)
        holder = owner()
        heartbeat = _Heartbeat(path, holder["controller"], self.stale_age)
        heartbeat.thread.start()
        count = 0
        try:
            for result in self.pool.run(
                self._claims(points, holder, heartbeat), _QueueWork(path, work)
            ):
                heartbeat.check()
                count += 1
                yield cast(R, result)
            if count == 0:
                with locked(path / "state.lock"):
                    if not (path / ".done").exists():
                        atomic_json(
                            path / ".nowork",
                            {
                                "controller": holder,
                                "recorded": time.time(),
                                "reason": (
                                    "no eligible points; inspect live owners "
                                    "and failed rows"
                                ),
                            },
                        )
        finally:
            heartbeat.close()
            directory = path / "controllers"
            directory.mkdir(exist_ok=True)
            atomic_json(
                directory / f"{holder['controller']}.json",
                {
                    "owner": holder,
                    "pool": dataclasses.asdict(self.pool),
                    "returned_points": count,
                    "recorded": time.time(),
                },
            )


def collect(path: Path | str, output: Path | str) -> ScanOutput:
    """Idempotently collect validated artifacts through one locked HDF5 writer.

    Parameters
    ----------
    path : pathlib.Path or str
        Queue directory containing its manifest and published point artifacts.
    output : pathlib.Path or str
        Collected HDF5 destination.

    Returns
    -------
    ScanOutput
        Reader masking unfinished points.

    Raises
    ------
    OutputError
        If a scan, point, configuration or artifact digest conflicts.
    """
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (
        manifest.get("queue_schema_version") != 1
        or _hash(manifest["description"]) != manifest["scan_id"]
    ):
        raise OutputError("invalid queue manifest version or description digest")
    scan = _restore_scan(manifest)
    settings = json.loads(manifest["description"])["settings"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with locked(path / "state.lock"):
        record = state(path)
    if record["scan_id"] != manifest["scan_id"] or len(record["rows"]) != scan.size:
        raise OutputError("queue state differs from its manifest")
    # `ScanWriter` itself holds the collected file's sidecar lock for its whole
    # session, which is what serialises two collectors and any reader. An outer
    # lock on the same sidecar would contend with it: flock is owned by the open
    # file description, so two descriptors deadlock even inside one process.
    writer = ScanWriter(output, scan, settings)
    changed = False
    try:
        if "artifact_digest" not in _group(writer.file, "points"):
            _group(writer.file, "points").create_dataset(
                "artifact_digest", shape=(scan.size,), dtype=h5py.string_dtype()
            )
            changed = True
        changed |= _collect_points(path, manifest, record, writer)
        seen = set(
            json.loads(str(writer.file.attrs.get("collected_controllers", "[]")))
        )
        reports = sorted((path / "controllers").glob("*.json")) + sorted(
            (path / "submissions").glob("*.json")
        )
        for report in reports:
            key = f"{report.parent.name}/{report.stem}"
            if key not in seen:
                kind = (
                    "Queue controller"
                    if report.parent.name == "controllers"
                    else "Slurm submission"
                )
                writer.record_execution(kind, read_json(report))
                seen.add(key)
                changed = True
        if changed:
            writer.file.attrs["collected_controllers"] = _json(sorted(seen))
            writer.record_execution(
                "Queue collection",
                {
                    "queue": str(path),
                    "scan_id": manifest["scan_id"],
                    "builder": manifest["builder"],
                },
            )
    finally:
        writer.close(consolidate=changed)
    return ScanOutput(output)


def _collect_points(
    path: Path, manifest: Mapping, record: Mapping, writer: ScanWriter
) -> bool:
    """Validate and collect the unchanged point artifacts in a queue snapshot.

    Parameters
    ----------
    path : pathlib.Path
        Shared queue directory.
    manifest : mapping
        Validated immutable scan and builder identity.
    record : mapping
        Queue snapshot under the state lock.
    writer : ScanWriter
        Exclusively held controller writer.

    Returns
    -------
    bool
        Whether any point changed the collected file.
    """
    changed = False
    scan = _restore_scan(manifest)
    for index, row in enumerate(record["rows"]):
        if row["status"] not in ("done", "failed"):
            previous, _, updated = writer.state(index)
            if (
                row["status"] == "running"
                and previous != "done"
                and (previous != "running" or updated != row["updated"])
            ):
                writer.mark_running(index, row["updated"])
                changed = True
            continue
        identity = {
            "scan_id": manifest["scan_id"],
            "point_id": _point_id(scan.point(index)),
            "owner": row["token"],
        }
        # A failed artifact can be replaced by a retry while collection
        # runs. Read under the publication lock, skipping newer claims.
        with locked(path / "state.lock"):
            if state(path)["rows"][index] != row:
                continue
            result = read_artifact(
                path / "points" / _filename(manifest, index),
                identity,
                digest=row["artifact_digest"],
            )
        if (
            result.index != index
            or result.configuration_hash != row["configuration_hash"]
        ):
            raise OutputError(
                "artifact index or configuration differs from queue state"
            )
        previous, digest, _ = writer.state(index)
        if (
            previous == "failed"
            and result.status == "failed"
            and (
                _dataset(writer.file, "points/artifact_digest").asstr()[index]
                == row["artifact_digest"]
            )
        ):
            continue
        if previous == "done":
            if (
                digest != result.configuration_hash
                or _dataset(writer.file, "points/artifact_digest").asstr()[index]
                != row["artifact_digest"]
            ):
                raise OutputError("collected point conflicts with its durable artifact")
            continue
        writer.commit(result)
        changed = True
        _dataset(writer.file, "points/artifact_digest")[index] = row["artifact_digest"]
        writer.file.flush()
    return changed
