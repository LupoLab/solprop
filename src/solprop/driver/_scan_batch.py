# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Coordinate backend-preserving batches and ordinary per-point scan captures.

A batch of compatible points runs as one group (SEP 47 section 6.7): this module is the
group's service loop. It advances the group one common window at a time and saves each
member's planes to that member's own capture, under the ordinary single-point schema.
"""

import dataclasses
import logging
import math
import time
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory

from solprop.backends import select
from solprop.backends.events import GroupBatch, ObservationReady, Terminal
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import Backend
from solprop.backends.session import GroupSession, Start
from solprop.backends.sessions import start_group
from solprop.driver._scan_run import (
    _Capture,
    _field_retention,
    _Job,
    _Work,
    launch_statistics,
)
from solprop.driver.batch import BatchMember, compatible_groups, shared_table_bytes
from solprop.driver.executors.batch import Batch
from solprop.driver.executors.pool import _memory_limit, peak_rss
from solprop.driver.run import _history_of
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError
from solprop.output.scan import PointResult
from solprop.output.schema import BackendRecord, RunMetadata, SavedPlane, StepStats
from solprop.processing.scans import _provenance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Prepared:
    """Keep a job's one constructed simulation and its singleton hash record."""

    job: _Job
    simulation: Simulation
    metadata: RunMetadata


_CONSUMER = ExportRecipe(power=True, optics=False)
"""The field a capture stores: output amplitudes in the internal FFT convention."""


def _construct(simulation: Simulation, cleanup: ExitStack) -> Backend:
    """Construct a member's selected backend with its cleanup registered.

    Parameters
    ----------
    simulation : Simulation
        Independently translated member.
    cleanup : contextlib.ExitStack
        Owner of the backend's lifetime until a group takes it over.

    Returns
    -------
    solprop.backends.protocol.Backend
        The backend, holding the member's parameter tables and no state yet.
    """
    owner = select(
        simulation.settings.backend,
        simulation.translate(),
        simulation._run_request(),
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options(),
    )
    cleanup.callback(owner.close)
    return owner


def _prepare(job: _Job, work: _Work) -> _Prepared | PointResult:
    """Validate a point's seed and hash once, before batch admission.

    Parameters
    ----------
    job : _Job
        Global point and existing completion digest.
    work : _Work
        Builder and retention policy.

    Returns
    -------
    _Prepared or PointResult
        Pending simulation, skipped point, hash conflict or builder failure.
    """
    try:
        simulation = work.simulation(job)
        metadata = simulation.metadata()
        digest = metadata.configuration_hash()
        if job.cached_hash:
            if job.cached_hash != digest:
                return PointResult(
                    job.point.index,
                    "conflict",
                    digest,
                    metadata.to_json(),
                    error=f"point {job.point.index} has a different configuration",
                    metadata_arrays=metadata.record_arrays(),
                )
            return PointResult(job.point.index, "skipped", configuration_hash=digest)
        return _Prepared(job, simulation, metadata)
    except Exception:
        return work.failure(job, traceback.format_exc())


def _admit(group: Sequence[_Prepared], work: _Work, executor: Batch) -> int:
    """Probe member workspaces and admit a bounded group with twenty percent headroom.

    All translated tables and the writer already reside in the baseline RSS. Their
    unique byte count is recorded, not added again. The first workspace uses the
    larger of measured RSS and baseline plus its ledger; only subsequent workspaces
    are additive. Two output copies cover capture consolidation and writer transfer.

    Parameters
    ----------
    group : sequence of _Prepared
        Bounded compatible simulations, already translated.
    work : _Work
        Retention policy used to estimate point outputs.
    executor : Batch
        Resource bounds and destination for the admission calculation.

    Returns
    -------
    int
        Positive number of members admitted simultaneously.

    Raises
    ------
    DriverError
        If the allocation cannot accommodate even one member.
    """
    limit = _memory_limit()
    if executor.memory_limit is not None:
        limit = min(limit, executor.memory_limit)
    baseline = peak_rss()
    if baseline >= limit * 4 // 5:
        raise DriverError("Batch memory budget cannot fit one member with headroom")
    ledger = output = packing = 0
    device_free: int | None = None
    for member in group:
        simulation = member.simulation
        problem = simulation.translate()
        with ExitStack() as cleanup:
            owner = _construct(simulation, cleanup)
            statistics = launch_statistics(owner, problem)
            ledger = max(ledger, sum(owner.memory_ledger().values()))
            output = max(output, work._output_bytes(simulation, statistics))
            free = owner.available_device_memory()
            if free is not None:
                device_free = free if device_free is None else min(device_free, free)
        field_bytes = math.prod(problem.shape) * 16
        # Construction briefly holds old member updates and their packed replacements.
        # The reference also holds concatenated stage inputs and response outputs.
        packing = max(
            packing,
            field_bytes * (20 if simulation.settings.backend == "numpy_ref" else 5)
            + 3
            * math.prod(problem.shape[:-1])
            * math.ceil(problem.shape[-1] / 256)
            * 8,
        )
    measured = peak_rss()
    first = max(measured, baseline + ledger)
    available = limit * 4 // 5 - first + ledger
    per_member = ledger + packing + 2 * output
    count = min(len(group), max(0, available // max(1, per_member)))
    host_count = count
    if device_free is not None:
        # The live probe is excluded from free bytes. Its complete ledger is a
        # conservative bound on device ownership (it may include host buffers).
        # Keep 20% of that allocation plus current free space unused. Counting
        # all ledger bytes as device storage errs towards a smaller batch.
        device_count = (4 * (device_free + ledger)) // (5 * max(1, ledger + packing))
        count = min(count, max(0, device_count))
    executor.budgets.append(
        {
            "limit": limit,
            "baseline_rss": baseline,
            "probe_rss": measured,
            "member_ledger": ledger,
            "member_output": output,
            "member_packing_peak": packing,
            "shared_table_bytes_in_baseline": shared_table_bytes(
                member.simulation.translate() for member in group
            ),
            "requested": len(group),
            "admitted": count,
            "host_admitted": host_count,
            "device_free_during_probe": device_free,
            "device_member_upper_bound": ledger + packing
            if device_free is not None
            else None,
            "estimated_peak": first
            + (count - 1) * ledger
            + count * (packing + 2 * output)
            if count
            else first,
            "headroom_percent": 20,
        }
    )
    if count < 1:
        raise DriverError("Batch memory budget cannot fit one member with headroom")
    return count


def _plane(
    session: GroupSession,
    member: int,
    event: ObservationReady,
    *,
    keep: bool,
) -> SavedPlane:
    """Build the plane a member's capture stores from its observation.

    Parameters
    ----------
    session : solprop.backends.session.GroupSession
        The group, its batch pending.
    member : int
        The member's position.
    event : solprop.backends.events.ObservationReady
        The member's observation.
    keep : bool
        Whether the scan reads this plane's field. Otherwise no field crosses.

    Returns
    -------
    solprop.output.schema.SavedPlane
        The plane, its field in true-power amplitudes and the internal convention.
    """
    rows = event.step_stats
    return SavedPlane(
        z=event.z,
        field=session.export(member, event.ticket, _CONSUMER) if keep else None,
        stats=dict(event.stats),
        step_history=_history_of(event.history),
        step_stats=StepStats(z=rows.z, step=rows.step, values=dict(rows.values)),
        noise_phase=session.noise_phase(member, event.ticket) if keep else None,
    )


def _service(
    session: GroupSession,
    batch: GroupBatch,
    captures: Sequence[_Capture],
    errors: list[str],
    *,
    keep: bool,
) -> None:
    """Save each running member's plane, and fail each member whose watch stopped it.

    A boundary stop ends its member alone, as a lone run of it would fail (SEP 21);
    a group-wide failure is raised, and fails every member.

    Parameters
    ----------
    session : solprop.backends.session.GroupSession
        The group, its batch pending.
    batch : solprop.backends.events.GroupBatch
        One common window's events.
    captures : sequence of _Capture
        Independent member outputs in dispatch order.
    errors : list of str
        Per-member failures; a failed member receives no further planes.
    keep : bool
        Whether the scan reads this plane's field.

    Raises
    ------
    BaseException
        The group-wide failure, when the window ended the group.
    """
    if batch.error is not None:
        raise batch.error
    summary = next(
        (
            event.summary
            for event in _observations(batch)
            if not event.summary.met_tolerance
        ),
        None,
    )
    if summary is not None:
        logger.warning(
            "a batch step was too coarse for its scaled tolerance at "
            "z = %.6e m: maximum member error norm %.3e against 1.0",
            summary.z,
            summary.max_norm,
        )
    for index, events in enumerate(batch.members):
        for event in events:
            if isinstance(event, Terminal) and event.error is not None:
                if not errors[index]:
                    errors[index] = "".join(traceback.format_exception(event.error))
            elif isinstance(event, ObservationReady) and not errors[index]:
                try:
                    captures[index].save(_plane(session, index, event, keep=keep))
                except Exception:
                    errors[index] = traceback.format_exc()


def _observations(batch: GroupBatch) -> Iterator[ObservationReady]:
    """Yield the save-plane observations of a window, in member order.

    Parameters
    ----------
    batch : solprop.backends.events.GroupBatch
        The window's events.

    Yields
    ------
    solprop.backends.events.ObservationReady
        Each member's plane, the launch excluded.
    """
    for events in batch.members:
        for event in events:
            if isinstance(event, ObservationReady) and event.plane_index >= 0:
                yield event


def _run(group: Sequence[_Prepared], work: _Work) -> Iterator[PointResult]:
    """Run one admitted group and preserve independently reduced point results.

    Parameters
    ----------
    group : sequence of _Prepared
        Compatible, admitted simulations with global indices.
    work : _Work
        Capture and reduction policy.

    Yields
    ------
    PointResult
        Complete per-member output or an explicit runtime failure.
    """
    started = time.perf_counter()
    captures: list[_Capture] = []
    results: list[PointResult] = []
    errors = [""] * len(group)
    records: Sequence[Mapping[str, object] | None] = [None] * len(group)

    def failed(index: int, error: str) -> PointResult:
        """Retain available provenance when one member fails.

        Parameters
        ----------
        index : int
            Member position within this admitted group.
        error : str
            Traceback from propagation, capture or reduction.

        Returns
        -------
        PointResult
            Failure with the available backend record and no numerical arrays.
        """
        member = group[index]
        capture = captures[index] if index < len(captures) else None
        metadata = (
            capture.metadata
            if capture is not None and capture.metadata is not None
            else member.metadata
        )
        return PointResult(
            member.job.point.index,
            "failed",
            member.metadata.configuration_hash(),
            metadata.to_json(),
            reductions={
                r.name: _provenance(r, metadata=member.metadata)
                for r in work.options.reductions
            },
            error=error,
            wall_time=time.perf_counter() - started,
            n_steps=0 if capture is None else capture.steps,
            n_rejected=0 if capture is None else capture.rejected,
            boundary_monitor=records[index],
            metadata_arrays=metadata.record_arrays(),
        )

    try:
        with ExitStack() as cleanup:
            problems = [member.simulation.translate() for member in group]
            owners = [_construct(member.simulation, cleanup) for member in group]
            # Each point watches its own window, as it would run alone (SEP 21).
            session = start_group(
                owners,
                problems,
                [
                    member.simulation._run_request().for_run(
                        monitor=member.simulation._boundary_monitor()
                    )
                    for member in group
                ],
                [Start.entrance(problem) for problem in problems],
            )
            cleanup.callback(session.close)
            launch = session.advance()
            for index, member in enumerate(group):
                # The monitors were attached first, so the record holds the launch's
                # zeros; the point keeps the closing record instead.
                backend = {
                    name: value
                    for name, value in session.execution_record(index).items()
                    if name != "boundary_monitor"
                }
                scratch = cleanup.enter_context(
                    TemporaryDirectory(prefix="solprop-batch-")
                )
                capture = _Capture(work.options, member.simulation, Path(scratch))
                cleanup.callback(capture.close)
                capture.open(
                    dataclasses.replace(
                        member.metadata, backend=BackendRecord.from_metadata(backend)
                    )
                )
                capture.retain(
                    _field_retention(capture).resolved(capture.count),
                    tuple(problems[index].shape),
                )
                captures.append(capture)
            # One scan's options and one group's planes: one rule for all its members.
            batch = launch
            plane = 0
            while True:
                _service(
                    session,
                    batch,
                    captures,
                    errors,
                    keep=captures[0].retention.keeps(plane),
                )
                session.acknowledge(batch.receipt)
                if session.finished:
                    break
                batch = session.advance()
                plane += 1
            records = [session.monitor_record(index) for index in range(len(group))]
            for index, (member, capture, error) in enumerate(
                zip(group, captures, errors, strict=True)
            ):
                capture.close()
                if error:
                    results.append(failed(index, error))
                    continue
                try:
                    arrays = capture.results()
                    if capture.metadata is None:
                        raise DriverError(
                            f"point {member.job.point.index} completed without "
                            "the run metadata its capture opened with"
                        )
                    results.append(
                        PointResult(
                            member.job.point.index,
                            "done",
                            member.metadata.configuration_hash(),
                            capture.metadata.to_json(),
                            arrays,
                            {
                                r.name: _provenance(r, metadata=member.metadata)
                                for r in work.options.reductions
                            },
                            wall_time=time.perf_counter() - started,
                            n_steps=capture.steps,
                            n_rejected=capture.rejected,
                            boundary_monitor=records[index],
                            metadata_arrays=capture.metadata.record_arrays(),
                        )
                    )
                except Exception:
                    results.append(failed(index, traceback.format_exc()))
    except Exception:
        error = traceback.format_exc()
        results = [failed(index, error) for index in range(len(group))]
    yield from results


def _stopping(work: _Work) -> bool:
    """Return whether the scan's stop flag is set.

    Parameters
    ----------
    work : _Work
        The scan's work, which holds the caller's flag.

    Returns
    -------
    bool
        True once the flag is set.
    """
    return work.stop is not None and work.stop.is_set()


def execute(
    jobs: Iterable[_Job], work: _Work, executor: Batch
) -> Iterator[PointResult]:
    """Build bounded groups and execute their independently selected backend.

    Parameters
    ----------
    jobs : iterable of _Job
        Pending and previously completed points in dispatch order.
    work : _Work
        Simulation builder and requested output contract.
    executor : Batch
        Size and memory limits, plus recorded admission calculations.

    Yields
    ------
    PointResult
        Globally indexed result suitable for the controller's ordinary writer.
    """
    backend: str | None = None
    for chunk in batched(jobs, executor.size, strict=False):
        pending: dict[int, _Prepared] = {}
        for job in chunk:
            prepared = _prepare(job, work)
            if isinstance(prepared, PointResult):
                yield prepared
                continue
            selected = prepared.simulation.settings.backend
            if backend is not None and selected != backend:
                raise DriverError(
                    f"Batch refuses mixed backends {backend!r} and {selected!r}"
                )
            backend = selected
            pending[job.point.index] = prepared
        members = (BatchMember(index, one.simulation) for index, one in pending.items())
        for compatible in compatible_groups(members, executor.size):
            group = tuple(pending[member.index] for member in compatible)
            if _stopping(work):
                # Between groups (SEP 41, S17): a claimed group not yet run is given
                # back, so a rerun computes it.
                yield from (work.stopped(member.job) for member in group)
                continue
            count = _admit(group, work, executor)
            for admitted in batched(group, count, strict=False):
                if _stopping(work):
                    yield from (work.stopped(member.job) for member in admitted)
                    continue
                yield from _run(admitted, work)
