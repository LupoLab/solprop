# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build scan points, collect bounded output, and coordinate durable results."""

import copy
import dataclasses
import hashlib
import inspect
import logging
import math
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

from solprop._interrupts import StopFlag, StopRequested
from solprop.backends import select
from solprop.driver._scan_context import current_executor
from solprop.driver._scan_model import Scan, ScanPoint, name_seed
from solprop.driver.executors import Batch, Executor, Local
from solprop.driver.executors._request import ScanRequest
from solprop.driver.executors.pool import PlaneScalars, peak_rss
from solprop.driver.executors.queue import Queue, collect
from solprop.driver.executors.slurm import Slurm
from solprop.driver.memory import admit_memory
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError, OutputError
from solprop.fourier import spectrum_to_optics
from solprop.log import (
    BarReporter,
    InterruptibleReporter,
    NullScanReporter,
    ProgressChoice,
    ProgressReporter,
    ScanReporter,
    SummarisingReporter,
    WatchingScanReporter,
    is_interactive,
)
from solprop.output.hdf5 import HDF5Output, _dataset, open_file
from solprop.output.memory import MemoryOutput
from solprop.output.scan import PointResult, ScanWriter
from solprop.output.schema import (
    FieldRetention,
    RunMetadata,
    SavedPlane,
    field_retention_refusal,
)
from solprop.processing.scans import Reduction, ScanOutput, _Builtin, _provenance

logger = logging.getLogger(__name__)

STOPPED = "stopped by the caller's stop flag; not committed, so a rerun computes it"
"""The error text of a point the scan's stop flag stopped (SEP 41, S17)."""

_PROGRESS_CHOICES = ("auto", "bar", "log", "none")

_DURABLE_STOP = (
    "{name} refuses stop=: its points run in other processes or on other machines, "
    "which a flag in this process cannot reach. Stop them with the queue's own tools: "
    "scancel for a Slurm job, or SIGTERM to a queue's controller processes; a rerun "
    "then claims what they left"
)


@dataclass(frozen=True)
class _Options:
    """Declare which worker results to retain."""

    reductions: tuple[Reduction, ...]
    fields: str
    field_planes: tuple[int, ...]
    stats: str | tuple[str, ...]


@dataclass(frozen=True)
class _Job:
    """Carry a global point and its existing configuration digest."""

    point: ScanPoint
    cached_hash: str = ""


def _plane_index(index: int, count: int) -> int:
    """Resolve a saved-plane index and reject out-of-range selections.

    Parameters
    ----------
    index : int
        Global point index, or saved-plane index where specified.
    count : int
        Number of saved planes, including the input.

    Returns
    -------
    int
        Zero-based resolved saved-plane index.
    """
    resolved = index + count if index < 0 else index
    if not 0 <= resolved < count:
        raise DriverError(
            f"field/reduction plane {index} is outside {count} saved planes"
        )
    return resolved


def _checked(
    reduction: Reduction, value: np.ndarray, metadata: RunMetadata
) -> np.ndarray:
    """Validate a reduction against its declared array contract.

    Parameters
    ----------
    reduction : Reduction
        Reduction declaration including shape, dtype and units.
    value : np.ndarray
        Value to validate or serialise.
    metadata : RunMetadata
        Complete single-run metadata.

    Returns
    -------
    numpy.ndarray
        Validated independent copy of the reduction result.
    """
    array = np.asarray(value)
    expected = reduction.shape(metadata)
    if array.shape != expected or array.dtype != reduction.dtype:
        raise OutputError(
            f"reduction {reduction.name!r} returned {array.shape}/{array.dtype}, "
            f"expected {expected}/{reduction.dtype}"
        )
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise OutputError(f"reduction {reduction.name!r} returned non-finite values")
    return array.copy()


class _Capture(MemoryOutput):
    """Keep statistics, selected fields and plane-local reductions only.

    Parameters
    ----------
    options : _Options
        Validated field, statistic and reduction retention policy.
    simulation : Simulation
        Simulation whose translated shape determines storage.
    scratch : Path
        Worker-local temporary directory.
    """

    def __init__(
        self, options: _Options, simulation: Simulation, scratch: Path
    ) -> None:
        """Initialize the declared scan component."""
        super().__init__(block=1)
        self.options = options
        self.count = len(simulation.settings.save_planes) + 1
        problem = simulation.translate()
        self._field_shape = tuple(problem.shape[1:])
        self.selected = (
            set(range(self.count))
            if options.fields == "all"
            else {_plane_index(i, self.count) for i in options.field_planes}
        )
        self.targets = {
            r.name: _plane_index(r.plane, self.count)
            for r in options.reductions
            if isinstance(r, _Builtin)
        }
        self.arrays: dict[str, list[np.ndarray]] = {}
        self.reduced: dict[str, np.ndarray] = {}
        self.steps = 0
        self.rejected = 0
        self.custom_path = scratch / "output.h5"
        self.spool = (
            HDF5Output(self.custom_path)
            if any(not isinstance(r, _Builtin) for r in options.reductions)
            else None
        )

    def held_bytes(
        self,
        n_planes: int,
        plane_bytes: int,
        *,
        kept: int | None = None,
        field_bytes: int = 0,
    ) -> int:
        """Return nothing for the run's admission: the scan admits what it retains.

        The capture never fills the store's own blocks; the arrays it keeps are
        counted by the scan's own admission of its members.

        Parameters
        ----------
        n_planes : int
            The planes the run will save.
        plane_bytes : int
            The bytes of one plane.
        kept : int or None, optional
            The planes that keep their field.
        field_bytes : int, optional
            The bytes of one plane's field.

        Returns
        -------
        int
            Zero.
        """
        del n_planes, plane_bytes, kept, field_bytes
        return 0

    def retain(self, retention: FieldRetention, field_shape: tuple[int, ...]) -> None:
        """Adopt the run's field retention, and hand it to the spool of a reduction.

        Parameters
        ----------
        retention : solprop.output.schema.FieldRetention
            Which planes keep their field, resolved.
        field_shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the run's state.
        """
        super().retain(retention, field_shape)
        if self.spool is not None:
            self.spool.retain(retention, field_shape)

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Initialize retained statistics and any temporary field spool.

        Parameters
        ----------
        metadata : RunMetadata
            Complete single-run metadata.
        force : bool
            Whether to replace existing metadata.
        """
        super().open(metadata, force=force)
        if self.spool is not None:
            self.spool.open(metadata)

    def _keep(self, name: str, value: np.ndarray | float) -> None:
        """Copy retained data before the backend reuses its buffers.

        Parameters
        ----------
        name : str
            Dataset path or column name.
        value : np.ndarray | float
            Value to validate or serialise.
        """
        self.arrays.setdefault(name, []).append(np.asarray(value).copy())

    def save(self, plane: SavedPlane) -> None:
        """Capture requested fields and statistics before the next step.

        Parameters
        ----------
        plane : SavedPlane
            Saved plane in the internal Fourier convention, or its index.
        """
        metadata = self.metadata
        if metadata is None:
            raise OutputError("scan capture must be opened before saving")
        index = self._n_saved
        self._keep("z", plane.z)
        if index in self.selected:
            # The run keeps the field of every selected plane (`_retained`).
            assert plane.field is not None
            self._keep(
                "fields/values",
                spectrum_to_optics(plane.field, kind=metadata.grid.kind),
            )
            self._keep("fields/planes", index)
            if plane.noise_phase is not None:
                self._keep("fields/noise_phase", plane.noise_phase)
        self.steps += plane.step_history.n_steps
        self.rejected += plane.step_history.n_rejected
        keep = self.options.stats
        for name, value in plane.stats.items():
            if keep == "all" or (not isinstance(keep, str) and name in keep):
                self._keep(f"stats/{name}", value)
        if keep != "none":
            self._keep("step_stats/z", plane.step_stats.z)
            self._keep("step_stats/step", plane.step_stats.step)
            self._keep("step_stats/counts", plane.step_stats.n_rows)
            for name, value in plane.step_stats.values.items():
                if keep == "all" or (not isinstance(keep, str) and name in keep):
                    self._keep(f"step_stats/{name}", value)
        chosen = [
            r
            for r in self.options.reductions
            if isinstance(r, _Builtin) and self.targets[r.name] == index
        ]
        if chosen:
            view = MemoryOutput(block=1)
            view.open(metadata)
            if plane.field is None:
                # A plane kept without its field serves the reductions of its
                # statistics; the run keeps the field of any other's target plane.
                view.retain(FieldRetention("none"), self.field_shape)
            view.save(plane)
            shape_metadata = dataclasses.replace(
                metadata,
                configuration=dict(metadata.configuration)
                | {"scan_field_shape": self.field_shape},
            )
            for reduction in chosen:
                self.reduced[reduction.name] = _checked(
                    reduction, reduction.evaluate(view, 0), shape_metadata
                )
            view.close()
        if self.spool is not None:
            self.spool.save(plane)
        self._n_saved += 1

    def _last_saved_z(self) -> float:
        """Return the position of the last plane the capture kept.

        The capture keeps its columns itself rather than in the store; a stopped run
        reads this back before it records its terminal checkpoint.

        Returns
        -------
        float
            The position, m.
        """
        return float(self.arrays["z"][-1])

    def close(self) -> None:
        """Close the temporary spool and the capture lifecycle."""
        if self.spool is not None:
            self.spool.close()
        super().close()

    def results(self) -> dict[str, np.ndarray]:
        """Combine retained columns and evaluate complete-output reducers.

        Returns
        -------
        dict
            Retained arrays keyed by their storage paths.
        """
        if self.spool is not None:
            with open_file(self.custom_path) as output:
                if output.metadata is None:
                    raise OutputError("custom reduction output has no metadata")
                for reduction in self.options.reductions:
                    if not isinstance(reduction, _Builtin):
                        self.reduced[reduction.name] = _checked(
                            reduction, reduction(output), output.metadata
                        )
        result = {
            name: (
                np.concatenate(parts, axis=0)
                if name.startswith("step_stats/") and name != "step_stats/counts"
                else np.stack(parts)
            )
            for name, parts in self.arrays.items()
        }
        result.update(
            {f"reduced/{name}": value for name, value in self.reduced.items()}
        )
        return result


class _Display(NullScanReporter):
    """The outer display a named progress choice means: one bar, log lines, or nothing.

    Parameters
    ----------
    choice : {'auto', 'bar', 'log', 'none'}
        The named choice; ``"auto"`` draws a bar when the process is interactive.
    """

    def __init__(self, choice: ProgressChoice) -> None:
        """Hold the choice, with no bar drawn."""
        self.choice = choice
        self.bar: BarReporter | None = None
        self.started = time.perf_counter()

    def scan_started(self, n_points: int) -> None:
        """Draw the bar, when one is chosen.

        Parameters
        ----------
        n_points : int
            The selected points, the bar's extent.
        """
        self.started = time.perf_counter()
        if self.choice == "bar" or (self.choice == "auto" and is_interactive()):
            self.bar = BarReporter(position=0, leave=True, echo=False)
            self.bar.started(0.0, float(max(1, n_points)), n_points)

    def point_finished(self, index: int, status: str, wall: float) -> None:
        """Log a point that did not fail, and advance the bar.

        Parameters
        ----------
        index : int
            The point's global index.
        status : str
            Its status; a failure is logged with its traceback when it is collected.
        wall : float
            Its build and run time, s.
        """
        if status != "failed" and self.choice != "none":
            logger.info("scan point %d: %s (%.3f s)", index, status, wall)
        if self.bar is not None and self.bar.bar is not None:
            self.bar.bar.update(1)

    def close(self) -> None:
        """Close the bar, if one was drawn."""
        if self.bar is not None:
            self.bar.finished(time.perf_counter() - self.started)
            self.bar = None


class _ScanEvents:
    """Deliver a scan's point boundaries, and its points' runs, to a scan reporter.

    Lives in the controlling process only. A point's start is reported when it is
    dispatched (``Local``, ``Pool``) or, for an executor that reports none, just before
    its result is collected.

    Parameters
    ----------
    reporter : solprop.log.ScanReporter
        Where the events go.
    scan : Scan
        The scan, which looks a collected point up by its global index.
    """

    def __init__(self, reporter: ScanReporter, scan: Scan) -> None:
        """Hold the reporter, with no point started."""
        self.reporter = reporter
        self.scan = scan
        self.started: set[int] = set()
        self.watchers: dict[int, ProgressReporter] = {}

    def begin(self, point: ScanPoint) -> ProgressReporter | None:
        """Report a point's start and return the reporter for its run, if any.

        Parameters
        ----------
        point : ScanPoint
            The point dispatched.

        Returns
        -------
        solprop.log.ProgressReporter or None
            The reporter :meth:`solprop.log.WatchingScanReporter.point_reporter`
            returned, or ``None``.
        """
        self.started.add(point.index)
        self.reporter.point_started(point.index, point)
        if not isinstance(self.reporter, WatchingScanReporter):
            return None
        watcher = self.reporter.point_reporter(point.index)
        if watcher is not None:
            self.watchers[point.index] = watcher
        return watcher

    def relay(
        self,
        index: int,
        call: str,
        args: tuple[object, ...],
        stop: StopFlag | None,
    ) -> None:
        """Make a reporter call a pool worker sent for its point's run.

        Parameters
        ----------
        index : int
            The point's global index.
        call : str
            ``"summary"``, ``"started"``, ``"plane"``, ``"finished"`` or
            ``"interrupted"``.
        args : tuple
            The call's arguments as :class:`~solprop.driver.executors.pool.Pool`'s
            worker sends them.
        stop : StopFlag or None
            The caller's flag, which a worker's stop stands for.
        """
        watcher = self.watchers.get(index)
        if watcher is None:
            return
        if call == "summary":
            if isinstance(watcher, SummarisingReporter):
                watcher.summary(cast(str, args[0]))
        elif call == "started":
            z_start, z_end, n_planes = cast(tuple[float, float, int], args)
            watcher.started(z_start, z_end, n_planes)
        elif call == "plane":
            z, summary = cast(tuple[float, PlaneScalars], args)
            watcher.plane(z, summary)
        elif call == "finished":
            watcher.finished(cast(float, args[0]))
        elif call == "interrupted" and isinstance(watcher, InterruptibleReporter):
            z, stopped, text = cast(tuple[float, bool, str], args)
            error: BaseException = (
                StopRequested(stop)
                if stopped and stop is not None
                else DriverError(f"scan point {index} was interrupted: {text}")
            )
            watcher.interrupted(z, error)

    def finish(self, result: PointResult) -> None:
        """Report a collected point, starting it first if nothing has.

        Parameters
        ----------
        result : PointResult
            The point's result.
        """
        if result.index not in self.started:
            self.started.add(result.index)
            self.reporter.point_started(result.index, self.scan.point(result.index))
        self.started.discard(result.index)
        self.watchers.pop(result.index, None)
        self.reporter.point_finished(result.index, result.status, result.wall_time)


@dataclass
class _Work:
    """Build and run independent simulations inside the chosen executor.

    ``stop`` and ``events`` belong to the controlling process and are not pickled: a
    pool worker receives the work without them, and runs each point under a stop flag
    and a reporter of its pipe's (``run_point``).
    """

    build: Callable[[ScanPoint], Simulation]
    options: _Options
    scan_name: str
    watched: bool = False
    stop: StopFlag | None = None
    events: _ScanEvents | None = None

    def __getstate__(self) -> dict[str, object]:
        """Drop the controller's flag and reporter, which stay in its process.

        Returns
        -------
        dict
            The fields a worker needs.
        """
        state: dict[str, object] = dict(self.__dict__)
        return state | {"stop": None, "events": None}

    def dispatched(self, job: _Job) -> None:
        """Report that a pool sent a point to a worker.

        Parameters
        ----------
        job : _Job
            The point.
        """
        if self.events is not None:
            self.events.begin(job.point)

    def relay(self, job: _Job, call: str, args: tuple[object, ...]) -> None:
        """Make a reporter call a pool worker sent for its point's run.

        Parameters
        ----------
        job : _Job
            The point the worker runs.
        call : str
            The reporter method.
        args : tuple
            Its arguments.
        """
        if self.events is not None:
            self.events.relay(job.point.index, call, args, self.stop)

    def stopped(self, job: _Job, wall_time: float = 0.0) -> PointResult:
        """Return the result of a point the stop flag stopped, which is not committed.

        Parameters
        ----------
        job : _Job
            The point.
        wall_time : float, optional
            What it ran for before it stopped, s.

        Returns
        -------
        PointResult
            Its ``"stopped"`` result.
        """
        return PointResult(
            job.point.index, "stopped", error=STOPPED, wall_time=wall_time
        )

    def run_batch(self, jobs: Iterable[_Job], executor: Batch) -> Iterator[PointResult]:
        """Delegate compatible jobs to the common lockstep scan driver.

        Parameters
        ----------
        jobs : iterable of _Job
            Global jobs, including previously completed configuration hashes.
        executor : Batch
            Batch-size and memory policy, with recorded admission calculations.

        Yields
        ------
        PointResult
            One durable result per dispatched point.
        """
        from solprop.driver._scan_batch import (  # noqa: PLC0415 - shared worker types
            execute,
        )

        yield from execute(jobs, self, executor)

    def simulation(self, job: _Job) -> Simulation:
        """Construct a simulation and give its noise the point's seed.

        Noise the builder left unseeded is drawn again from ``point.seed``, or from
        :func:`~solprop.driver._scan_model.name_seed` on a scan without seeds, so
        that the point is reproducible and a rerun matches the record of its first
        run; noise the builder seeded must have used ``point.seed``.

        Parameters
        ----------
        job : _Job
            Global point and previously committed configuration digest.

        Returns
        -------
        Simulation
            Constructed simulation with reproducible noise.
        """
        simulation = self.build(job.point)
        if not isinstance(simulation, Simulation):
            raise DriverError("scan builder must return a Simulation")
        point = job.point
        simulation._reseed(
            name_seed(self.scan_name, point.index) if point.seed is None else point.seed
        )
        if simulation.settings.fields != "all" or simulation.settings.field_planes:
            raise DriverError(
                "a scan keeps the fields run_scan's fields= and field_planes= ask for, "
                "and its point builders keep their settings' default fields='all': "
                "drop fields and field_planes from the builder's SimulationSettings"
            )
        if simulation.settings.step_planes:
            raise DriverError(
                "a scan saves the same planes at every point; step_planes saves each "
                "point's own steps, so it is not scanned: pass step_planes=0"
            )
        requested = self.options.stats
        available = simulation.translate().stats_spec.names
        if not isinstance(requested, str) and set(requested) - set(available):
            raise DriverError(
                "requested scan statistics are not collected by the simulation"
            )
        for reduction in self.options.reductions:
            if (
                isinstance(reduction, _Builtin)
                and reduction.operation in ("beam_radius", "on_axis_intensity")
                and reduction.operation not in available
            ):
                raise DriverError(
                    f"reduction {reduction.name} requires its simulation statistic"
                )
        noise = simulation.noise
        if noise is not None and (
            noise.seed is None or (point.seed is not None and noise.seed != point.seed)
        ):
            msg = (
                f"scan noise must have a recorded seed matching point.seed: point "
                f"{point.index} drew its noise from seed {noise.seed} and the point's "
                f"is {point.seed}; pass seed=point.seed to the constructor, or leave "
                "the seed unset so the scan supplies it"
            )
            raise DriverError(msg)
        return simulation

    def probe(self, job: _Job) -> Mapping[str, int]:
        """Measure backend allocations without propagating a point.

        Parameters
        ----------
        job : _Job
            Global point and previously committed configuration digest.

        Returns
        -------
        collections.abc.Mapping
            Peak RSS, baseline RSS and named backend allocation bytes.
        """
        baseline = peak_rss()
        simulation = self.simulation(job)
        admit_memory(simulation.translate(), simulation.settings.backend)
        backend = select(
            simulation.settings.backend,
            simulation.translate(),
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            # None, not `simulation.noise`: the realisation the builders return is in
            # true sqrt(W), while `initial.noise0` has already been divided by the
            # entrance weight, and StepDriver prefers a passed realisation over it.
            backend.set_initial(simulation.translate().initial.field0, None)
            return {
                "baseline_rss": baseline,
                "rss": peak_rss(),
                "ledger": sum(backend.memory_ledger().values()),
                "output_bytes": self._output_bytes(
                    simulation, backend.stats(simulation.translate().stats_spec)
                ),
            }
        finally:
            backend.close()

    def failure(self, job: _Job, error: str) -> PointResult:
        """Record an abnormal worker exit without losing other completed points.

        Parameters
        ----------
        job : _Job
            Point assigned to the terminated worker.
        error : str
            Child traceback or abnormal-exit explanation.

        Returns
        -------
        PointResult
            Failure of this attempt, which never demotes a committed point.
        """
        # A crash is a crash whatever the point's stored state: only `__call__`
        # can see a genuine configuration disagreement, and only it reports one.
        # Calling a dead worker a conflict aborted the whole scan through
        # `ScanWriter.commit` and presented the traceback as different physics.
        return PointResult(job.point.index, "failed", error=error)

    def _output_bytes(
        self, simulation: Simulation, statistics: Mapping[str, np.ndarray]
    ) -> int:
        """Estimate retained plane data and reductions for resource admission.

        Parameters
        ----------
        simulation : Simulation
            Probed point with its final translation.
        statistics : mapping
            One plane of statistics with resolved species and mode axes.

        Returns
        -------
        int
            Retained array bytes; adaptive periodic rows use the memory headroom.
        """
        count = len(simulation.settings.save_planes) + 1
        selected = (
            count
            if self.options.fields == "all"
            else len(
                {_plane_index(index, count) for index in self.options.field_planes}
            )
        )
        field_bytes = math.prod(simulation.translate().shape[1:]) * 16 * selected
        if simulation.noise is not None:
            field_bytes *= 2
        keep = self.options.stats
        stat_bytes = count * sum(
            value.nbytes
            for name, value in statistics.items()
            if keep == "all" or (not isinstance(keep, str) and name in keep)
        )
        spec = simulation.translate().stats_spec
        schedule = simulation.translate().stepper.schedule
        periodic_bytes = 0
        if keep != "none" and schedule is not None and spec.periodic:
            rows = schedule.n_steps // spec.period
            periodic_bytes = (
                rows
                * (
                    16
                    + sum(
                        value.nbytes
                        for name, value in statistics.items()
                        if name in spec.periodic
                        and (
                            keep == "all"
                            or (not isinstance(keep, str) and name in keep)
                        )
                    )
                )
                + count * 8
            )
        metadata = dataclasses.replace(
            simulation.metadata(),
            configuration=dict(simulation.metadata().configuration)
            | {"scan_field_shape": simulation.translate().shape[1:]},
        )
        reduced_bytes = sum(
            math.prod(reduction.shape(metadata)) * reduction.dtype.itemsize
            for reduction in self.options.reductions
        )
        return field_bytes + stat_bytes + periodic_bytes + reduced_bytes + count * 8

    def __call__(self, job: _Job) -> PointResult:
        """Run a point in this process, under the caller's flag and reporter.

        Parameters
        ----------
        job : _Job
            Global point and previously committed configuration digest.

        Returns
        -------
        object
            Completed calculation, captured point failure, or a stopped point.
        """
        reporter = None if self.events is None else self.events.begin(job.point)
        return self.run_point(job, self.stop, reporter)

    def run_point(
        self, job: _Job, stop: StopFlag | None, reporter: ProgressReporter | None
    ) -> PointResult:
        """Execute the declared worker calculation.

        Parameters
        ----------
        job : _Job
            Global point and previously committed configuration digest.
        stop : StopFlag or None
            The flag the point's run polls; when it stops the run, the point is
            ``"stopped"``, not done.
        reporter : solprop.log.ProgressReporter or None
            Where the point's run reports; ``None`` runs it silently.

        Returns
        -------
        object
            Completed calculation, captured point failure, or a stopped point.
        """
        started = time.perf_counter()
        metadata_json = digest = ""
        capture: _Capture | None = None
        if stop is not None and stop.is_set():
            return self.stopped(job)
        try:
            simulation = self.simulation(job)
            metadata = simulation.metadata()
            digest = metadata.configuration_hash()
            metadata_json = metadata.to_json()
            if job.cached_hash:
                if job.cached_hash != digest:
                    return PointResult(
                        job.point.index,
                        "conflict",
                        digest,
                        metadata.to_json(),
                        error=f"point {job.point.index} has a different configuration",
                    )
                return PointResult(
                    job.point.index, "skipped", configuration_hash=digest
                )
            # The run closes the spool and results() closes its reader before this
            # exits; ignoring cleanup errors keeps a file Windows still holds open
            # from turning a finished point into a failed one.
            with TemporaryDirectory(
                prefix="solprop-scan-", ignore_cleanup_errors=True
            ) as scratch:
                capture = _Capture(self.options, simulation, Path(scratch))
                _retained(simulation, capture).run(
                    capture,
                    progress="none" if reporter is None else reporter,
                    stop=stop,
                )
                outcome = capture.outcome
                # The run returns normally only when complete or stopped by `stop`.
                if outcome is not None and outcome.status != "complete":
                    return self.stopped(job, time.perf_counter() - started)
                arrays = capture.results()
                if capture.metadata is not None:
                    metadata_json = capture.metadata.to_json()
                return PointResult(
                    job.point.index,
                    "done",
                    digest,
                    metadata_json,
                    arrays,
                    {
                        r.name: _provenance(r, metadata=metadata)
                        for r in self.options.reductions
                    },
                    wall_time=time.perf_counter() - started,
                    n_steps=capture.steps,
                    n_rejected=capture.rejected,
                )
        except Exception:
            if capture is not None and capture.metadata is not None:
                metadata_json = capture.metadata.to_json()
            return PointResult(
                job.point.index,
                "failed",
                digest,
                metadata_json,
                error=traceback.format_exc(),
                wall_time=time.perf_counter() - started,
            )


def _retained(simulation: Simulation, capture: _Capture) -> Simulation:
    """Return the point's simulation keeping only the fields the scan reads.

    The fields the scan retains, those its built-in reductions read at their target
    planes, and, while a custom reduction that reads fields spools the run, every
    field (SEP 22, Q3). The builder's own simulation is left as it is: a builder may
    return one object for every point.

    Parameters
    ----------
    simulation : Simulation
        The point's simulation, translated.
    capture : _Capture
        The point's capture, which knows the selected and target planes.

    Returns
    -------
    Simulation
        A shallow copy sharing the translation, with the scan's field retention.
    """
    options = capture.options
    spooled = any(
        not isinstance(reduction, _Builtin) and getattr(reduction, "reads_fields", True)
        for reduction in options.reductions
    )
    if options.fields == "all" or spooled:
        return simulation
    needed = set(capture.selected) | {
        capture.targets[reduction.name]
        for reduction in options.reductions
        if isinstance(reduction, _Builtin) and reduction.reads_fields
    }
    retained = copy.copy(simulation)
    retained.settings = dataclasses.replace(
        simulation.settings,
        fields="planes" if needed else "none",
        field_planes=tuple(sorted(needed)),
    )
    return retained


def _jobs(
    scan: Scan,
    writer: ScanWriter,
    retry: bool,
    age: float,
    stop: StopFlag | None,
    claims: dict[int, tuple[str, float]],
) -> Iterator[_Job]:
    """Claim eligible points lazily as dispatch capacity becomes available.

    Parameters
    ----------
    scan : Scan
        Cartesian scan or selection retaining global indices.
    writer : ScanWriter
        Controller-owned scan file.
    retry : bool
        Whether failed points are eligible.
    age : float
        Positive abandonment interval, s.
    stop : StopFlag or None
        Once set, no further point is claimed.
    claims : dict
        Filled with each claimed point's state and update time before the claim, so
        that a stopped point is given back as it was (`_release`).

    Yields
    ------
    object
        Result of the declared operation.
    """
    for point in scan.points():
        if stop is not None and stop.is_set():
            return
        state, digest, updated = writer.state(point.index)
        if state == "failed" and not retry:
            continue
        if state == "running" and time.time() - updated < age:
            continue
        if state != "done":
            claims[point.index] = state, updated
            writer.mark_running(point.index, time.time())
        yield _Job(point, digest if state == "done" else "")


def _release(writer: ScanWriter, index: int, claim: tuple[str, float]) -> None:
    """Give a stopped point back as it was before this scan claimed it.

    A stopped point is not committed, so a rerun computes it again: left ``running``,
    a rerun would skip it for ``stale_age``.

    Parameters
    ----------
    writer : ScanWriter
        Controller-owned scan file.
    index : int
        The point's global index.
    claim : tuple
        Its state and update time before the claim.
    """
    state, updated = claim
    _dataset(writer.file, "points/status")[index] = state
    _dataset(writer.file, "points/updated")[index] = updated
    writer.file.flush()


def execute_scan(  # noqa: PLR0913 - delegated public controls
    scan: Scan,
    build: Callable[[ScanPoint], Simulation],
    *,
    output: Path | str,
    executor: Executor | None,
    reductions: Sequence[Reduction],
    fields: str,
    field_planes: Sequence[int],
    stats: str | Sequence[str],
    retry_failed: bool,
    stale_age: float,
    progress: ProgressChoice | ScanReporter,
    stop: StopFlag | None = None,
) -> ScanOutput:
    """Validate, dispatch and commit a scan through one writer.

    Parameters
    ----------
    scan : Scan
        Cartesian scan or selection retaining global indices.
    build : Callable[[ScanPoint], Simulation]
        Callable constructing a Simulation from a ScanPoint.
    output : Path | str
        Output handler for reductions, or collected destination path for execution.
    executor : Executor | None
        Execution policy; None selects the CLI policy or Local.
    reductions : Sequence[Reduction]
        Worker-side reduction declarations.
    fields : str
        Field retention mode: none, planes or all.
    field_planes : Sequence[int]
        Explicit saved-plane indices; negative indices count from the end.
    stats : str | Sequence[str]
        All, none, or explicit statistic names.
    retry_failed : bool
        Whether to dispatch previously failed points.
    stale_age : float
        Positive abandonment interval, s.
    progress : {'auto', 'bar', 'log', 'none'} or solprop.log.ScanReporter
        Outer progress display, or a reporter that replaces it.
    stop : StopFlag or None, optional
        The scan's stop flag; refused by Queue and Slurm.

    Returns
    -------
    ScanOutput
        Reader for the collected scan.
    """
    _validate_options(fields, field_planes, stats, stale_age, reductions)
    if progress not in _PROGRESS_CHOICES and not isinstance(
        progress, ScanReporter
    ):  # pragma: no cover - refused by the annotation where it is checked
        raise DriverError(
            "progress must be 'auto', 'bar', 'log', 'none' or a ScanReporter, got "
            f"{progress!r}"
        )
    if executor is None:
        executor = current_executor() or Local()
    if stop is not None and isinstance(executor, (Queue, Slurm)):
        raise DriverError(_DURABLE_STOP.format(name=type(executor).__name__))
    options = _Options(
        tuple(reductions),
        fields,
        tuple(field_planes),
        stats if isinstance(stats, str) else tuple(stats),
    )
    display: _Display | None = None
    if isinstance(progress, str):
        display = _Display(progress)
        reporter: ScanReporter = display
    else:
        reporter = progress
    events = _ScanEvents(reporter, scan)
    work = _Work(
        build,
        options,
        scan.name,
        watched=display is None and isinstance(progress, WatchingScanReporter),
        stop=stop,
        events=events,
    )
    validate = getattr(executor, "validate", None)
    if callable(validate):
        validate(work)
    settings = {
        "fields": fields,
        "field_planes": list(field_planes),
        "stats": stats,
        "reductions": {r.name: _provenance(r) for r in reductions},
    }
    writer = None
    claims: dict[int, tuple[str, float]] = {}
    if isinstance(executor, (Queue, Slurm)):
        executor.prepare(ScanRequest(scan, settings, output, work, retry_failed))
        jobs = (_Job(point) for point in scan.points())
    else:
        writer = ScanWriter(output, scan, settings)
        jobs = _jobs(scan, writer, retry_failed, stale_age, stop, claims)
    counts = dict.fromkeys(("finished", "stopped"), 0)
    try:
        events.reporter.scan_started(len(scan))
        for result in executor.run(jobs, work):
            _collect(result, writer, claims.pop(result.index, None))
            counts["stopped" if result.status == "stopped" else "finished"] += 1
            events.finish(result)
        if stop is not None and stop.is_set():
            logger.info(
                "scan %r stopped by the caller's stop flag: %d points collected, %d "
                "stopped and not committed; a rerun computes the stopped and the "
                "undispatched points",
                scan.name,
                counts["finished"],
                counts["stopped"],
            )
        events.reporter.scan_finished()
    finally:
        if display is not None:
            display.close()
        _finish_execution(executor, build, writer, output)
    return ScanOutput(output)


def _collect(
    result: PointResult, writer: ScanWriter | None, claim: tuple[str, float] | None
) -> None:
    """Commit a collected point, or give a stopped one back, and log a failure.

    Parameters
    ----------
    result : PointResult
        The point's result.
    writer : ScanWriter or None
        Local controller writer, or None for a durable queue.
    claim : tuple or None
        The point's state and update time before this scan claimed it.
    """
    if result.status == "stopped":
        if writer is not None and claim is not None:
            _release(writer, result.index, claim)
        return
    # Re-deriving a committed point can fail without that point being any
    # less complete: an edited builder, a moved data file, a worker killed
    # while it only rehashed. `ScanWriter.commit` refuses such a demotion
    # outright, so the failure is reported and the remaining points run.
    stale = (
        result.status == "failed"
        and writer is not None
        and writer.state(result.index)[0] == "done"
    )
    if writer is not None and not stale:
        writer.commit(result)
    if result.status == "failed":
        logger.error(
            "scan point %d failed%s:\n%s",
            result.index,
            " while revisiting a completed point, which is kept as it is"
            if stale
            else "",
            result.error,
        )


def _finish_execution(
    executor: Executor,
    build: Callable[[ScanPoint], Simulation],
    writer: ScanWriter | None,
    output: Path | str,
) -> None:
    """Record controller provenance and consolidate the durable output.

    Parameters
    ----------
    executor : Executor
        Selected execution policy and measured resource settings.
    build : Callable[[ScanPoint], Simulation]
        Builder whose source identifies this invocation.
    writer : ScanWriter or None
        Local controller writer, or None for a durable queue.
    output : pathlib.Path or str
        Collected scan destination.
    """
    policy = (
        dataclasses.asdict(executor)
        if dataclasses.is_dataclass(executor) and not isinstance(executor, type)
        else {}
    )
    try:
        source = inspect.getsourcefile(
            build if inspect.isfunction(build) else type(build)
        )
    except TypeError:
        source = None
    if source is not None and Path(source).is_file():
        policy["script_hash"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        policy["script"] = source
    if writer is not None:
        writer.record_execution(type(executor).__qualname__, policy)
        writer.close()
    elif isinstance(executor, Queue) and executor.path is not None:
        collect(executor.path, output)
    elif isinstance(executor, Slurm) and executor.queue_path is not None:
        collect(executor.queue_path, output)


def _validate_options(
    fields: str,
    field_planes: Sequence[int],
    stats: str | Sequence[str],
    stale_age: float,
    reductions: Sequence[Reduction],
) -> None:
    """Reject ambiguous retention controls before opening the output.

    Parameters
    ----------
    fields : str
        Field retention mode: none, planes or all.
    field_planes : Sequence[int]
        Explicit saved-plane indices; negative indices count from the end.
    stats : str | Sequence[str]
        All, none, or explicit statistic names.
    stale_age : float
        Positive abandonment interval, s.
    reductions : Sequence[Reduction]
        Worker-side reduction declarations.
    """
    refusal = field_retention_refusal(fields, field_planes)
    if refusal is not None:
        raise DriverError(refusal)
    if not math.isfinite(stale_age) or stale_age <= 0:
        raise DriverError("stale_age must be positive and finite")
    if isinstance(stats, str) and stats not in ("all", "none"):
        raise DriverError("stats must be all, none or a sequence of names")
    names = [r.name for r in reductions]
    if len(set(names)) != len(names) or any(not n or "/" in n for n in names):
        raise DriverError("reduction names must be unique nonempty dataset names")
