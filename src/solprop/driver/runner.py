# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run one translated simulation many times on one backend, input by input.

:class:`SimulationRunner` keeps a backend between runs that differ only in their input,
and :func:`check_derived_problem` refuses a derived problem that differs in anything
else. The largest field each optmat rate saw is reported here after a run, a runner's
or a simulation's own. The manual page is ``docs/guide/setup/running.md``.
"""

import dataclasses
import functools
import logging
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from solprop._display import Displayed, line
from solprop.backends.protocol import Backend
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.driver.inputs import (
    _borrowed_input,
    _realisation,
    _without_input,
)
from solprop.driver.resume import _open_output
from solprop.driver.routes.hollow import _warn_sampled_peak
from solprop.driver.run import field_retention, run_problem
from solprop.driver.settings import _same_setting
from solprop.exceptions import DriverError, MaterialRangeWarning
from solprop.log import (
    DEFAULT_PROGRESS_PERIOD,
    ProgressChoice,
    ProgressReporter,
    StopFlag,
)
from solprop.modes.material import JsonValue
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import (
    OptmatRate,
    PeakFieldReport,
    peak_field_report,
)
from solprop.nonlinear.responses import PlasmaResponse
from solprop.output.hdf5 import HDF5Output
from solprop.output.metadata import PEAK_FIELDS_ANNOTATION, RunMetadata
from solprop.output.store import FieldRetention, Output, PlaneStore
from solprop.physdata import _optmat
from solprop.physdata.selection import MaterialLog
from solprop.pulses import Pulse, SpatialPulse, SpatialPulseSum
from solprop.translate.envelope_plasma import CycleIntegralData, cycle_moments
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import flux_of, tdse_descriptor, tdse_of
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = ["SimulationRunner", "check_derived_problem"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _LoadedRun:
    """What a runner's march needs, its input held by its start alone.

    Attributes
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The derived problem, without its input (:func:`_without_input`).
    start : solprop.backends.session.Start
        The input's start. The run's session takes its field, so that nothing holds
        the input through the march (issue #447).
    handler : solprop.output.store.Output
        Where the planes go.
    metadata : solprop.output.metadata.RunMetadata
        The run's record.
    retention : solprop.output.store.FieldRetention
        Which planes keep their field.
    boundary_monitor : solprop.monitor.BoundaryMonitor or None
        The run's monitor of the window's edges.
    material_log : solprop.physdata.selection.MaterialLog
        The derived run's material log, which the march goes on with.
    """

    problem: PropagationProblem
    start: Start
    handler: Output
    metadata: RunMetadata
    retention: FieldRetention
    boundary_monitor: BoundaryMonitor | None
    material_log: MaterialLog


class SimulationRunner(Displayed):
    """One translated problem and one backend, run for many inputs (SEP 22 section 5.2).

    Built by :meth:`~solprop.driver.simulation.Simulation.runner`: the problem is
    translated (or the cached translation reused) and one backend is constructed, once.
    Each :meth:`run` then derives the input with
    :meth:`~solprop.driver.simulation.Simulation.with_input`, loads it into the same
    backend and runs the save-plane loop, returning that input's own handler. A large
    problem so pays its backend's allocation and transform planning once rather than for
    every input.

    A run on the reused backend repeats a fresh backend's run, to the rounding that
    separates any two runs: loading an input restores everything the march reads
    (the reference's :meth:`solprop.backends.host.driver.HostDriver.set_initial`, the
    native backend's own load). Each run's record is
    its own input's, so each has its own configuration hash.

    Use it as a context manager, or call :meth:`close`: the backend holds about
    fifteen copies of the state until then. A closed runner refuses :meth:`run`. A
    run that raises leaves the runner open, since the next input resets everything;
    a backend that closed itself closes the runner.

    Parameters
    ----------
    simulation : Simulation
        The run to repeat. Its settings, and the translation they fix, are used as
        they are; only the input changes from run to run.

    Raises
    ------
    DriverError
        If the simulation has a TDSE response, whose atomic binding has its own
        lifecycle and is never reused.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.pulses import SechPulse
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1.2e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3, rtol=1e-6,
    ... )
    >>> def pulse(power):
    ...     return SechPulse(wavelength=835e-9, width=100e-15, peak_power=power)
    >>> with simulation.runner() as runner:
    ...     outputs = [
    ...         runner.run(pulse(power), configuration={"peak_power": power},
    ...                    progress="none")
    ...         for power in (5.0, 9.8)
    ...     ]
    >>> [output.n_saved for output in outputs], runner.closed
    ([3, 3], True)
    """

    def __init__(self, simulation: "Simulation") -> None:
        """Translate, admit and construct the backend, once."""
        problem = simulation.translate()
        if tdse_descriptor(problem.route) is not None:
            msg = (
                "a TDSE response binds an atomic runtime with its own lifecycle, which "
                "a runner cannot reuse across inputs; run each input with "
                "simulation.with_input(...).run()"
            )
            raise DriverError(msg)
        settings = simulation.settings
        request = simulation._run_request()
        self._simulation = simulation
        self._problem = problem
        self._request = request
        self._lock = threading.Lock()
        self._closed = False
        self.runs = 0
        with _optmat.material_context(simulation._material_log):
            _warn_sampled_peak(settings.quadrature, request.observation.stats)
            self._backend = select(
                settings.backend,
                problem,
                request,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
                options=settings.execution_options(),
            )

    def __repr__(self) -> str:
        """Return a one-line description: the simulation, the runs and the state.

        Returns
        -------
        str
            For example ``SimulationRunner(Simulation(...), 3 runs, open)``.
        """
        return line(
            "SimulationRunner",
            [
                ("", repr(self._simulation)),
                ("", f"{self.runs} runs"),
                ("", "closed" if self._closed else "open"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "SimulationRunner", [
            ("simulation", repr(self._simulation)),
            ("backend", self._simulation.settings.backend),
            ("runs", self.runs),
            ("state", "closed" if self._closed else "open"),
        ]

    def __enter__(self) -> "SimulationRunner":
        """Return the runner, so that a ``with`` block closes it on the way out.

        Returns
        -------
        SimulationRunner
            This runner.
        """
        return self

    def __exit__(self, *details: object) -> None:
        """Close the runner when the block ends, whatever happened inside it.

        Parameters
        ----------
        *details : object
            The exception being raised, if any; it is not suppressed.
        """
        self.close()

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the runner, and its backend, are closed."""
        return self._closed

    @property
    def backend(self) -> Backend:  # numpydoc ignore=RT01
        """Backend every run is stepped on."""
        return self._backend

    def close(self) -> None:
        """Close the backend, releasing its buffers; calling it twice does nothing.

        A run in progress on another thread finishes first: the close waits for it.
        """
        with self._lock:
            self._close_held()

    def _close_held(self) -> None:
        """Close the backend, the runner's lock being held by the caller."""
        if self._closed:
            return
        self._closed = True
        self._backend.close()

    def run(
        self,
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        *,
        output: Output | Path | str | None = None,
        configuration: Mapping[str, object] | None = None,
        noise: NoiseRealisation | Literal["keep"] | None = "keep",
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
        input_identity: str | None = None,
    ) -> Output:
        """Run one input on the runner's backend and return its handler.

        Parameters
        ----------
        initial : Pulse, SpatialPulse or numpy.ndarray
            The input, as :meth:`~solprop.driver.simulation.Simulation.with_input` takes
            it. An array the record keeps by digest
            (:func:`~solprop.output.serialisation.recorded_input`) is not copied: it is
            read while this call loads it, and the runner holds no reference to it, nor
            to any copy, once it is loaded, so a caller that keeps none itself has it
            freed before the first step.
        output : solprop.output.store.Output or pathlib.Path or str or None, optional
            Where this input's planes go: a handler, a path for an HDF5 file, or
            ``None`` for a new :class:`~solprop.output.memory.MemoryOutput`.
        configuration : collections.abc.Mapping or None, optional
            Caller keys recorded in this input's run record and hash, such as the
            delay of a scan (:meth:`~solprop.driver.simulation.Simulation.with_input`).
        noise : NoiseRealisation, {'keep'} or None, optional
            The noise, as for :meth:`~solprop.driver.simulation.Simulation.with_input`:
            kept by default, or replaced by another realisation of the same model.
        progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
            Where progress goes (:func:`solprop.driver.run.run_problem`).
        progress_period : float, optional
            Seconds between the line reporter's periodic lines, s.
        input_identity : str or None, optional
            What an array input is, recorded in its place and not checked, so that its
            bytes are not digested
            (:meth:`~solprop.driver.simulation.Simulation.with_input`, whose warning
            applies: a wrong identity passes a resume and a scan's cache).

        Returns
        -------
        solprop.output.store.Output
            This input's handler, closed; a memory handler stays readable.

        Raises
        ------
        DriverError
            If the runner is closed or already running an input; if the derived
            problem is not the runner's but for its input (a different launch
            position, or a shared leaf replaced); if the noise model differs from
            the runner's, which the backend could not carry; or if `input_identity`
            is empty or names anything but an array input.
        """
        if not self._lock.acquire(blocking=False):
            msg = (
                "this runner is already running an input: one backend steps one input "
                "at a time, so build one runner per thread or process"
            )
            raise DriverError(msg)
        try:
            if self._closed:
                msg = "this runner is closed; build another with simulation.runner()"
                raise DriverError(msg)
            loaded = self._load(
                initial,
                output=output,
                configuration=configuration,
                noise=noise,
                input_identity=input_identity,
            )
            # The start holds the input, and every frame that saw it has returned but
            # this one; the session takes it from the start, so the march holds no
            # volume for it (issue #447).
            del initial
            return self._march(
                loaded, progress=progress, progress_period=progress_period
            )
        except BaseException:
            # A backend that closed itself cannot run another input.
            if self._backend.closed:
                self._close_held()
            raise
        finally:
            self._lock.release()

    def _takes(self, simulation: "Simulation") -> bool:
        """Return whether a simulation is this runner's but for its input."""
        try:
            check_derived_problem(
                self._problem,
                simulation.translate(),
                self._request,
                simulation._run_request(),
            )
        except DriverError as error:
            logger.debug("a new backend for this run: %s", error)
            return False
        return True

    def _run_point(
        self,
        simulation: "Simulation",
        output: Output,
        *,
        progress: ProgressReporter | ProgressChoice | None,
        stop: StopFlag | None,
    ) -> None:
        """Run a simulation this runner :meth:`_takes`, as its own ``run`` would.

        For a scan's worker, which has built the point's simulation and holds its
        record already. Unlike :meth:`run` it honours `stop`, and the handler's record
        is the simulation's own.
        """
        with self._lock:
            simulation._run(output, progress=progress, stop=stop, kept=self._backend)
            self.runs += 1

    def _load(
        self,
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        *,
        output: Output | Path | str | None,
        configuration: Mapping[str, object] | None,
        noise: NoiseRealisation | Literal["keep"] | None,
        input_identity: str | None,
    ) -> _LoadedRun:
        """Derive one input's run, open its handler and build the input's start.

        The derived simulation and its problem are this frame's alone: they go when it
        returns, and the start alone holds the converted launch of a modal route or the
        view of the input, until the run's session takes it.
        """
        derived = self._simulation.with_input(
            _borrowed_input(initial, stated=input_identity is not None),
            noise=noise,
            configuration=configuration,
            input_identity=input_identity,
        )
        problem = derived.translate()
        request = derived._run_request()
        check_derived_problem(self._problem, problem, self._request, request)
        settings = self._simulation.settings
        with _optmat.material_context(derived._material_log):
            metadata = derived.metadata()
            handler, _ = _open_output(output, metadata, resume=False)
            retention = field_retention(request, settings.fields, settings.field_planes)
            monitor = derived._boundary_monitor()
            start = Start.entrance(
                problem, noise=None if isinstance(noise, str) else _realisation(problem)
            )
        return _LoadedRun(
            _without_input(problem),
            start,
            handler,
            metadata,
            retention,
            monitor,
            derived._material_log,
        )

    def _march(
        self,
        loaded: _LoadedRun,
        *,
        progress: ProgressReporter | ProgressChoice | None,
        progress_period: float,
    ) -> Output:
        """Run the loaded input's save-plane loop and finish its handler.

        - ``progress_period``: seconds between periodic lines, s.
        """
        settings = self._simulation.settings
        problem, handler = loaded.problem, loaded.handler
        with _optmat.material_context(loaded.material_log):
            run_problem(
                problem,
                self._backend,
                handler,
                start=loaded.start,
                metadata=loaded.metadata,
                stats_spec=self._request.observation.stats,
                progress=progress,
                progress_period=progress_period,
                resume_history=settings.resume_history,
                fields=loaded.retention.fields,
                field_planes=loaded.retention.field_planes,
                boundary_monitor=loaded.boundary_monitor,
            )
            reports = _peak_field_reports(problem, handler, self._simulation.responses)
        if reports and isinstance(handler, PlaneStore):
            handler.annotate(
                PEAK_FIELDS_ANNOTATION,
                [cast("JsonValue", report.to_dict()) for report in reports],
            )
        self.runs += 1
        return handler


_SHARED_LEAVES = (
    "grids",
    "linear",
    "transform",
    "responses",
)
"""The leaves :meth:`Simulation.with_input` shares with the translated problem.

The route is shared but for a spatial route's transverse design, whose check
:meth:`Simulation.with_input` reruns on the new launch: its variant and its other
leaves are compared beside these (:func:`check_derived_problem`). The statistics are
the request's, shared too."""


def check_derived_problem(
    constructed: PropagationProblem,
    derived: PropagationProblem,
    constructed_request: RunRequest,
    derived_request: RunRequest,
) -> None:
    """Refuse a derived problem that is not the constructed one but for its input.

    What a :class:`SimulationRunner` checks before loading an input into the backend it
    built for `constructed` (SEP 22 section 5.2): the leaves
    :meth:`~solprop.driver.simulation.Simulation.with_input` shares are the same
    objects, the statistics too, the stepper and the flags, which it rebuilds, agree
    field by field (the save planes elementwise), and the launch position and the noise
    model are the backend's.

    Parameters
    ----------
    constructed : solprop.translate.problem.PropagationProblem
        The problem the backend was built for.
    derived : solprop.translate.problem.PropagationProblem
        The problem of one input.
    constructed_request : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    derived_request : solprop.translate.run_request.RunRequest
        The request of the input's run.

    Raises
    ------
    DriverError
        Naming the first leaf or setting that differs.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.runner import check_derived_problem
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> problem = simulation.translate()
    >>> derived = simulation.with_input(problem.initial.field0)
    >>> check_derived_problem(
    ...     problem, derived.translate(), simulation._run_request(),
    ...     derived._run_request(),
    ... ) is None
    True
    """
    for name in _SHARED_LEAVES:
        if getattr(derived, name) is not getattr(constructed, name):
            msg = (
                f"the input's problem does not share the runner's {name}: a runner "
                "steps one translated problem and changes only its input"
            )
            raise DriverError(msg)
    if derived_request.observation.stats is not constructed_request.observation.stats:
        msg = (
            "the input's problem does not share the runner's stats_spec: a runner "
            "steps one translated problem and changes only its input"
        )
        raise DriverError(msg)
    if (
        type(derived.route) is not type(constructed.route)
        or tdse_of(derived.route) is not tdse_of(constructed.route)
        or flux_of(derived.route) is not flux_of(constructed.route)
    ):
        msg = (
            "the input's problem does not share the runner's route: a runner steps "
            "one translated problem and changes only its input"
        )
        raise DriverError(msg)
    if derived.initial.z != constructed.initial.z:
        msg = (
            f"the input is launched at z = {derived.initial.z!r} m and the runner's "
            f"problem at {constructed.initial.z!r} m: the save planes and the "
            "schedule's cursor belong to the problem the backend was built for, so a "
            "different initial_z needs its own simulation"
        )
        raise DriverError(msg)
    if derived.initial.noise_model != constructed.initial.noise_model:
        msg = (
            f"the input carries the noise model {derived.initial.noise_model!r} and "
            f"the runner's problem {constructed.initial.noise_model!r}: the backend "
            "carries the model it was built with, and would drop or misapply another"
        )
        raise DriverError(msg)
    for record, built, name in (
        (derived_request.stepper, constructed_request.stepper, "stepper"),
        (derived.flags, constructed.flags, "flags"),
    ):
        for entry in dataclasses.fields(record):
            if not _same_setting(
                getattr(built, entry.name), getattr(record, entry.name)
            ):
                msg = (
                    f"the input's {name}.{entry.name} differs from the runner's: a "
                    "runner steps one translated problem and changes only its input"
                )
                raise DriverError(msg)


def _peak_field_reports(
    problem: PropagationProblem,
    output: Output,
    responses: Sequence[object],
) -> tuple[PeakFieldReport, ...]:
    """Report the largest field each optmat-rate population saw, after a run.

    The populations' peak rates at the save planes are read back from the run's
    output -- ``species_peak_ionisation_rate``, or ``peak_ionisation_rate`` for one
    population -- and inverted through each rate
    (:func:`solprop.nonlinear.optmat_rates.peak_field_report`).

    - Returns: one per population with an optmat rate; none when the run carries no such
      rate or recorded no peak rate, which is logged.
    """
    labels = problem.responses.labels
    rates = [
        (
            one.species
            or (labels[0] if len(labels) == 1 else str(getattr(one.gas, "name", ""))),
            one.rate,
        )
        for one in responses
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(one.rate, OptmatRate)
    ]
    if not rates:
        return ()
    if isinstance(output, HDF5Output) and output.closed:
        with HDF5Output(output.path, readonly=True) as reader:
            return _peak_field_reports(problem, reader, responses)
    if "species_peak_ionisation_rate" in output.stat_names:
        rows = np.asarray(output["stats/species_peak_ionisation_rate"])
        peaks = np.max(rows, axis=0)
    elif "peak_ionisation_rate" in output.stat_names and len(labels) == 1:
        rows = np.asarray(output["stats/peak_ionisation_rate"])
        peaks = np.array([np.max(rows)])
    else:
        logger.info(
            "the run recorded no peak ionisation rate, so the largest field its "
            "optmat rates saw is not reported"
        )
        return ()
    envelope = problem.responses.envelope_plasma
    cycles = (
        {}
        if envelope is None
        else {channel.label: channel.cycle for channel in envelope.channels}
    )
    reports = []
    for label, rate in rates:
        report = peak_field_report(
            rate,
            float(peaks[labels.index(label)]),
            label=label,
            # An envelope's statistic is its own cycle average of its table, at
            # linear polarisation the cosine moment alone.
            average=None
            if label not in cycles
            else functools.partial(_linear_moment, cycles[label]),
        )
        if report.reached_maximum:
            _optmat.warn_once(
                ("rate-reached-maximum", rate.material_records[0].digest),
                f"{label}: the run reached the maximum field "
                f"{report.maximum_field:.4g} V/m of optmat's {report.model}, above "
                "which its rate is held at its value there; the largest field is at "
                "least that",
                MaterialRangeWarning,
            )
        logger.info(
            "%s: largest field %s V/m (maximum %.4g V/m); rate-weighted energy there "
            "%s against the constant %.6g J",
            label,
            "none" if report.largest_field is None else f"{report.largest_field:.4g}",
            report.maximum_field,
            "none"
            if report.departure is None
            else f"departs by {report.departure:+.3e}",
            report.constant_energy,
        )
        reports.append(report)
    return tuple(reports)


def _linear_moment(cycle: CycleIntegralData, amplitude: float) -> float:
    """Return an envelope run's cycle-averaged rate at a linear amplitude.

    - ``amplitude``: envelope amplitude in V/m.
    - Returns: the rate in s^-1 its statistic records there.
    """
    return float(cycle_moments(amplitude, 0.0, cycle)[0])
