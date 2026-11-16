# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's scheduler: planes, halts, records and the seam (SEP 47, SEP 52).

:class:`HostDriver` schedules the reference's run: the save planes and step planes,
the halts and yields of its session, the step history, the periodic statistics rows,
the boundary monitor's and the transverse edge's checks, the noise load and the seam's
``set_initial``/``propagate_to``/``observe``. It computes nothing state-sized: the
state and its operations are a workspace's
(:class:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace`), and each method's
march (:class:`~solprop.backends.host.runge_kutta.RungeKuttaMarch`,
:class:`~solprop.backends.host.adams.HostAdamsMarch`,
:class:`~solprop.backends.host.low_storage.HostLowStorageMarch`) holds its own
position, proposal and history, steps through its engine and asks the driver only
through the narrow :class:`~solprop.backends.host.marches.MarchHost`. A subclass
supplies the workspace, the engines and the marches. SEP 52 moves this scheduling into
the reference's session and deletes the driver.
"""

import dataclasses
import logging
import math
import time
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._helpers import PLAIN, Helpers
from solprop._interrupts import poll
from solprop._ownership import hold
from solprop.backends.engines import AcceptRecord, RungeKuttaEngine
from solprop.backends.host.marches import MarchState, in_flight_refusal
from solprop.backends.node_chunk import NodeChunkPlan, plan_node_chunk
from solprop.backends.protocol import (
    Backend,
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
    TimingProbe,
)
from solprop.backends.records import collar_rests
from solprop.backends.session import (
    ExecutionOptions,
    PreparedExecution,
    Start,
)
from solprop.exceptions import BackendError, BoundaryError
from solprop.grid import TimeGrid
from solprop.log import NOT_ESTIMATED, NullReporter, ProgressReporter, StopFlag
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.records import MarchSnapshot
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.stepper.controller import ControlRecord
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import (
    reached,
    replay_tolerance,
)
from solprop.translate.modal import QUADRATURE_ALARM, warn_quadrature
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.problem import PropagationProblem, PropagatorMode
from solprop.translate.routes import transverse_of
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    from solprop.backends.host.adams import HostAdamsMarch
    from solprop.backends.host.low_storage import HostLowStorageMarch
    from solprop.backends.numpy_ref.engines import NumpyRungeKutta
    from solprop.backends.numpy_ref.workspace import NumpyWorkspace

__all__ = [
    "INITIAL_STEP_STATS_CAPACITY",
    "HostDriver",
    "reached",
    "replay_tolerance",
]

_StateT = TypeVar("_StateT")


INITIAL_STEP_STATS_CAPACITY = 64
"""Rows a plane's statistics buffer holds before it first has to grow."""


class _StepStatsBuffer:
    """The rows a periodic statistics subset produces between two save planes.

    Grown by doubling from a preallocated block and **reset without releasing it**, so
    that after the first plane a run allocates nothing per step: the capacity a plane
    needed is the capacity the next plane starts with, which is a better estimator of a
    plane's step count than anything the driver could predict. Growing the datasets by
    concatenation on every step would be quadratic.

    The shape of each statistic is learnt from the first row rather than read from
    :class:`solprop.stats.StatDefinition`, whose ``shape`` is a documentation string and
    whose windowed entries depend on the spec's window count.
    """

    def __init__(self) -> None:
        self._capacity = 0
        self._n_rows = 0
        self._z = np.zeros(0)
        self._step = np.zeros(0, dtype=np.int64)
        self._values: dict[str, np.ndarray] = {}

    def clear(self) -> None:
        """Drop the rows and the allocation, as a fresh load does."""
        self.__init__()

    def reset(self) -> None:
        """Forget the rows of the plane just finished, keeping the allocation."""
        self._n_rows = 0

    def append(self, z: float, step: int, values: Mapping[str, np.ndarray]) -> None:
        """Add one row.

        - ``z``: position the row was taken at, m.
        - ``values``: what the backend's ``stats`` returned, keyed by name.
        """
        if self._capacity == 0:
            self._allocate(values)
        elif self._n_rows == self._capacity:
            self._grow()
        index = self._n_rows
        self._z[index] = z
        self._step[index] = step
        for name, value in values.items():
            row = self._values.get(name)
            if row is None or np.shape(value) != row.shape[1:]:
                expected = None if row is None else row.shape[1:]
                msg = (
                    f"the {name!r} statistic came back with shape "
                    f"{np.shape(value)} and this plane's rows are {expected}"
                )
                raise BackendError(msg)
            row[index] = value
        self._n_rows = index + 1

    def drain(self) -> StepStats:
        """Return the rows as a frozen record and forget them.

        - Returns: copies, because the buffer is about to be written over, held so that
          the record takes them without another.
        """
        rows = self._n_rows
        drained = StepStats(
            z=hold(self._z[:rows].copy()),
            step=hold(self._step[:rows].copy()),
            values={
                name: hold(value[:rows].copy()) for name, value in self._values.items()
            },
        )
        self.reset()
        return drained

    def _allocate(self, values: Mapping[str, np.ndarray]) -> None:
        """Lay the block out from the first row's shapes.

        - ``values``: the first row, whose shapes every later row of this plane must
          match.
        """
        self._capacity = INITIAL_STEP_STATS_CAPACITY
        self._z = np.zeros(self._capacity)
        self._step = np.zeros(self._capacity, dtype=np.int64)
        self._values = {
            name: np.zeros((self._capacity, *np.shape(value)), dtype=np.float64)
            for name, value in values.items()
        }

    def _grow(self) -> None:
        """Double the block, copying what is in it."""
        self._capacity *= 2
        self._z = np.resize(self._z, self._capacity)
        self._step = np.resize(self._step, self._capacity)
        self._values = {
            name: np.concatenate([value, np.zeros_like(value)])
            for name, value in self._values.items()
        }


class _TrialProbe:
    """Single trials of a loaded :class:`HostDriver`, committing nothing (SEP 47 6.6).

    The :class:`solprop.backends.protocol.TimingProbe` this driver returns. A
    Runge-Kutta trial (the driver's Runge-Kutta engine) leaves the loaded state, the
    position and the history alone. The low-storage march has no separate trial: its
    step advances the discarded registers in place and leaves no trial field.

    - ``start``: the loaded position, m, and the replayed itinerary's index there.
    """

    def __init__(
        self,
        driver: "HostDriver[object]",
        start: tuple[float, int],
    ) -> None:
        """Hold the loaded driver; no trial is taken yet."""
        self._driver = driver
        self._z, self._index = start
        self._length: float | None = None

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one trial step and keep its end state for :meth:`field`.

        - ``z``: where the trial starts, m; the loaded position by default.
        - ``dz``: its length, m, positive; the problem's ``dz_init`` by default, or the
          scheduled step at the loaded position for the low-storage march.
        """
        driver = self._driver
        start = self._z if z is None else float(z)
        march = driver._low_storage
        if march is not None:
            schedule = driver._stepper.schedule
            assert schedule is not None  # the translation refuses the march without one
            index = min(self._index, schedule.n_steps - 1)
            length = float(schedule.steps[index]) if dz is None else float(dz)
            cast("RungeKuttaEngine", march.engine).trial(start, length)
            return
        length = float(driver._stepper.dz_init) if dz is None else float(dz)
        driver._runge_kutta_engine.trial(start, length)
        self._length = length

    def field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the last trial's end state on the host, as a kept step would be.

        The trial's state after the accepted-step boundary treatments (the band mask,
        and the split-step absorber where it runs). A trial is held beside the loaded
        state, so finishing it leaves the loaded state alone, and no monitor is bound,
        so the finish takes no split sums.

        - Returns: an owning copy, internal units and the FFT convention, shape
          ``(n_modes, n_pol, n_omega)``.
        """
        driver = self._driver
        if driver._low_storage is not None:
            msg = (
                "the low-storage march accepts its step in place and has no trial "
                "field to read back"
            )
            raise BackendError(msg)
        if self._length is None:
            raise BackendError("take a trial before reading its field")
        engine = driver._runge_kutta_engine
        finished, _ = driver._workspace.finish(engine.trial_state, self._length)
        self._length = None
        return np.array(finished)


class _SessionHooks:
    """The driver as its marches' :class:`~solprop.backends.host.marches.MarchHost`.

    The narrow calls a march makes of the session that schedules it, forwarded to the
    driver's own bookkeeping, so that no march reads or writes the driver's fields.
    """

    def __init__(self, driver: "HostDriver[object]") -> None:
        self._driver = driver

    @property
    def collar_quiet(self) -> bool:
        """Whether the stages leave a quiet in-stage temporal collar out."""
        return self._driver._collar_quiet

    def poll_stop(self) -> None:
        """Raise when the run's stop flag is set."""
        self._driver._poll_stop()

    def cap(self, proposal: float, z: float) -> float:
        """Hold a proposal to the quiet collar's, the edge's and the split's limits."""
        driver = self._driver
        return driver._split_capped(
            driver._transverse_capped(driver._quiet_capped(proposal), z)
        )

    def record(self, end: float, dz: float, norm: float, accepted: bool) -> None:
        """Record one attempt."""
        self._driver._record(end, dz, norm, accepted)

    def absorb(self, record: AcceptRecord) -> None:
        """Accumulate an accepted step's split collar's sums."""
        self._driver._absorb(record)

    def stepped(self) -> None:
        """Account the step just committed."""
        self._driver._take_step_stats()

    def natural_stop(self, target: float) -> bool:
        """Count an accepted step; return whether to stop at its end."""
        return self._driver._natural_stop(target)

    def natural_stop_due(self, target: float) -> bool:
        """Return whether the march returns where it stands."""
        return self._driver._natural_stop_due(target)


class _March(Protocol):
    """What the driver asks of whichever march its run steps with."""

    state: MarchState

    @property
    def holding(self) -> bool: ...

    def load(self, z: float, proposal: float, cursor: int) -> None: ...

    def advance(self, target: float) -> None: ...


class HostDriver[_StateT](ABC):  # noqa: UP049 - the protected state type is private
    """The reference's scheduler, with the workspace and the marches left to a subclass.

    A subclass supplies the workspace (``_workspace``), the Runge-Kutta engine the
    timing probe uses (``_runge_kutta_engine``) and the march its run steps with
    (``_march``, with ``_adams`` or ``_low_storage`` set when it is that march), each
    march reporting to the driver's hooks (``_hooks``). The driver gives them the whole
    of :class:`solprop.backends.protocol.Backend`'s marching behaviour: the save planes
    and step planes, the halts and yields, the step history, the statistics rows, the
    boundary monitor and the transverse edge, and both noise models' loads.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem to step. The subclass refuses what it cannot execute
        before calling this constructor.
    request : solprop.translate.run_request.RunRequest
        The run's policies: the method, the error control, the itinerary, the planes
        and the statistics (SEP 47 section 5.2). The subclass checks it with the
        problem (:func:`solprop.translate.problem.validate_run`) before calling this
        constructor; a session started on this backend serves a request that differs
        from it only in one run's start, retention, monitor and history
        (:meth:`~solprop.translate.run_request.RunRequest.for_run`).
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, which the statistics need and the contract does not
        carry.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, for the statistics that need them.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.
    options : solprop.backends.session.ExecutionOptions, optional
        The run's execution options: an explicit node budget or an exact node chunk,
        from which the node chunk is planned here, before anything is allocated
        (:func:`solprop.backends.node_chunk.plan_node_chunk`). ``None`` for the
        automatic choice.

    Attributes
    ----------
    name : str
        The registry name, set by the subclass and used in every log line.
    propagator_default : {'tabulate', 'recompute'}
        What this backend resolves the contract's ``"auto"`` propagator mode to. The
        default suits a backend that evaluates two complex exponentials per stage; one
        that takes both signs from a single ``exp`` and ``sincos`` overrides it.
    helpers : solprop._helpers.Helpers
        The implementation of the per-stage and per-step scalar evaluations -- the
        stage tables' Clenshaw rows and the Adams basis -- which the driver, march and
        engines pass down rather than choosing one (SEP 29 section 6.6): ``PLAIN``, the
        uncompiled forms, for the NumPy reference.
    """

    name: str = "driver"
    propagator_default: PropagatorMode = "tabulate"
    helpers: Helpers = PLAIN

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        options: ExecutionOptions | None = None,
    ) -> None:
        """Adopt a validated problem and request and start the run at the entrance."""
        self._problem = problem
        self._request = request
        # The stepping policy as one record, read by the march at every step.
        self._stepper = request.stepper
        # SEP 47 section 6.5: the node chunk is an execution choice, planned from the
        # problem and the options.
        self._options = ExecutionOptions() if options is None else options
        self._chunk_plan: NodeChunkPlan = plan_node_chunk(problem, self._options)
        self._adams: HostAdamsMarch | None = None
        # The low-storage march (SEP 24), set at construction by a backend that
        # executes it; its engine owns the flag of a step left in flight.
        self._low_storage: HostLowStorageMarch | None = None
        self._grid = grid
        self._modes = modes
        self._profiles = profiles
        self._logger = logging.getLogger(type(self).__module__)
        self._closed = False
        # The marches' narrow view of the driver (SEP 52 section 5.2).
        self._hooks = _SessionHooks(cast("HostDriver[object]", self))
        # Whether the plane's quadrature alarm was reported (warned once per plane).
        self._alarm_reported = False
        # Natural-step saving (SEP 14 phase 3): accepted steps since the load, and
        # whether a step that crossed a densely observed plane ends on a step to save.
        self._natural_count = 0
        self._natural_due = False
        self._halt_ahead = False
        self._halt_requested = False
        self._halt_check: Callable[[], bool] | None = None
        self._halted = False
        self._history: list[tuple[float, float, float, bool, float]] = []
        self._clock = time.perf_counter()
        self._progress: ProgressReporter = NullReporter()
        self._progress_started = False
        # The statistics channel between planes. The spec a row is evaluated with is
        # the periodic subset promoted to `names`, so the row goes through the very
        # same `stats` the save planes use and is bitwise what a plane would record.
        spec = request.observation.stats
        self._row_spec: StatsSpec | None = (
            None
            if not spec.periodic
            else dataclasses.replace(spec, names=spec.periodic, periodic=(), period=1)
        )
        self._row_period = spec.period
        self._accepted_steps = 0
        self._rows = _StepStatsBuffer()

    # --- what a subclass supplies -----------------------------------------------

    _workspace: "NumpyWorkspace"
    """The state and every state-sized operation on it."""

    _runge_kutta_engine: "NumpyRungeKutta"
    """The Runge-Kutta engine: the Runge-Kutta march's and the timing probe's."""

    _march: _March
    """The march the run steps with."""

    @property
    def _z(self) -> float:
        """Return the march's position, m."""
        return self._march.state.z

    @property
    def _proposal(self) -> float:
        """Return the march's proposal, m."""
        return self._march.state.proposal

    @property
    def _schedule_index(self) -> int:
        """Return the march's replay cursor."""
        return self._march.state.cursor

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Compute the requested statistics of the state in one fused pass.

        Called by the session at a plane and by the step loop's row storage, at the
        march's position.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name.
        """
        self._require_state()
        return self._workspace.stats(spec, self._z)

    def _on_accepted_step(self) -> None:
        """Score whatever an accepted step leaves behind, before the next is proposed.

        The quadrature alarm is scored here, from buffers the last stage left behind,
        and a value above :data:`~solprop.translate.modal.QUADRATURE_ALARM` is
        reported once per plane. It is a statistic and a warning, never a controller
        input.
        """
        value, z = self._workspace.score_quadrature()
        if value > QUADRATURE_ALARM and not self._alarm_reported:
            self._alarm_reported = True
            warn_quadrature(value, z)

    def _reset_step_stats(self, *, release: bool) -> None:
        """Discard periodic rows at a load or at the start of a plane.

        - ``release``: true for a new load, which releases the old storage; false for
          a new plane, which retains capacity. The accepted-step period is the
          march's and is unaffected.
        """
        if release:
            self._rows.clear()
        else:
            self._rows.reset()

    def _record_step_stats(self, spec: StatsSpec, z: float, step: int) -> None:
        """Evaluate and retain one selected accepted-step statistics row.

        Called after boundary treatments and after the state and position have been
        committed. A device override must retain the values without copying them to
        the host, using the same evaluator as :meth:`stats`.

        - ``z``: position of the committed state, m.
        - ``step``: the attempt's index in the history since the previous observation,
          rejected attempts included; supplied by the march, not inferred from rows.
        """
        self._rows.append(z, step, self.stats(spec))

    def _drain_step_stats(self) -> StepStats:
        """Return owning host rows at observation and retain reusable capacity.

        A device override copies the packed values once here. Repeated drains return
        no rows until more are recorded.
        """
        return self._rows.drain()

    @property
    def _propagator_mode(self) -> PropagatorMode:
        """Return the propagator mode this run actually uses.

        - Returns: ``"tabulate"`` or ``"recompute"``, with the contract's ``"auto"``
          resolved to :attr:`propagator_default`.
        """
        asked = self._stepper.propagator_mode
        if asked == "auto" and isinstance(
            self._stepper.integrator, LowStorageRungeKutta
        ):
            # One volume of exp(h D / 2) reused while the step holds is the cheaper
            # flow on a CPU; a varying operator changes it every step (SEP 24
            # section 4.3).
            return "tabulate" if self._problem.flags.constant_linop else "recompute"
        if asked == "auto" and self._problem.linear.separable is not None:
            return "recompute"
        return self.propagator_default if asked == "auto" else asked

    def prepare(
        self, problem: PropagationProblem, request: RunRequest
    ) -> PreparedExecution:
        """Prepare a run of this backend for one problem and request (SEP 47 5.2).

        The backend lowered the method, the error control, the itinerary, the planes
        and the statistics where it was built; preparing binds the problem and the
        request a run serves -- this backend's own, or one run's adaptation of it
        (:meth:`~solprop.translate.run_request.RunRequest.for_run`) -- and each start
        of the result opens the reference's own session
        (:class:`~solprop.backends.host.session.HostSession`).

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The translated problem: this backend's, or one derived from it with
            another input (:func:`solprop.driver.runner.check_derived_problem`).
        request : solprop.translate.run_request.RunRequest
            The run's request.

        Returns
        -------
        solprop.backends.session.PreparedExecution
            The prepared execution.
        """
        from solprop.backends.host.session import (  # noqa: PLC0415 - it builds on this module
            HostPrepared,
        )

        # A concrete driver is a backend: this base is the march, and the subclass
        # supplies the rest of the protocol (`memory_ledger`, `close`). They are not
        # abstract here, so that a protocol check can refuse a driver without.
        return HostPrepared(cast("Backend", self), problem, request)

    @property
    def request(self) -> RunRequest:  # numpydoc ignore=RT01
        """Request this backend was built for.

        Its method, error control, itinerary, planes and statistics; a run adapts
        it with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        return self._request

    @property
    def node_chunk(self) -> int:  # numpydoc ignore=RT01
        """The nodes a response pass of this backend holds at once.

        The planned chunk (:attr:`chunk_plan`).
        """
        return self._chunk_plan.chunk

    @property
    def chunk_plan(self) -> NodeChunkPlan:  # numpydoc ignore=RT01
        """Node-chunk plan and what chose it."""
        return self._chunk_plan

    def available_device_memory(self) -> int | None:
        """Return None for host execution; device backends report free bytes.

        Returns
        -------
        int or None
            None for the shared host implementation.
        """
        return None

    def metadata(self) -> dict[str, object]:
        """Return what this execution was, for the run's record.

        A saved run has to say what produced it: the thread configuration alone was
        measured to move one step by a factor of nine, so a result whose execution is
        not recorded is not reproducible. The base record is the name and the resolved
        propagator mode and, for Adams runs, cumulative logical source calls; the
        low-storage march's evaluations are physical, ``source_evaluations``, never
        ``response_calls`` (SEP 24 section 5); a backend with pools, plans or wisdom
        adds them.

        Returns
        -------
        dict
            JSON-serialisable scalars, keyed by name.
        """
        record: dict[str, object] = {
            "backend": self.name,
            "propagator_mode": self._propagator_mode,
            # SEP 47 section 4.3: execution metadata, outside the resume key.
            **self._chunk_plan.record(),
            "node_chunk": self.node_chunk,
        }
        if self._problem.responses.plasma_channels:
            record["plasma_source"] = plasma_source_record(self._problem.grids)
        if self._adams is not None:
            record["response_calls"] = self._adams.ledger.to_record()
        if self._low_storage is not None:
            record["source_evaluations"] = self._low_storage.source_evaluations
        elif self._adams is None:
            record["source_evaluations"] = self._runge_kutta_engine.evaluations
        if self._monitor is not None:
            record["boundary_monitor"] = self._monitor.summary()
        return record

    def march_snapshot(self) -> MarchSnapshot | None:
        """Return the Adams march's state at the accepted position, for a resume.

        Returns
        -------
        solprop.output.records.MarchSnapshot or None
            The snapshot, or ``None`` for the Runge-Kutta and low-storage marches and
            wherever the Adams march has nothing a resume could continue
            (:meth:`solprop.backends.host.adams.HostAdamsMarch.snapshot`).

        Raises
        ------
        BackendError
            If a low-storage step was abandoned in flight.
        """
        self._refuse_in_flight()
        if self._low_storage is not None:
            return self._low_storage.snapshot()
        return None if self._adams is None else self._adams.snapshot()

    def march_snapshot_nbytes(self, z: float) -> int | None:
        """Return the bytes :meth:`march_snapshot` would copy at `z`, copying nothing.

        Parameters
        ----------
        z : float
            The position the snapshot is wanted at, m.

        Returns
        -------
        int or None
            The state and every engine row, or ``None`` where a snapshot would be
            ``None`` or stand somewhere other than `z`.

        Raises
        ------
        BackendError
            If a low-storage step was abandoned in flight.
        """
        self._refuse_in_flight()
        if self._low_storage is not None or self._adams is None or self._z != z:
            return None
        return self._adams.snapshot_nbytes()

    def march_export_refusal(self) -> str | None:
        """Return ``None``: every march the reference admits carries its history.

        Returns
        -------
        None
            The Adams engines export and install their rows; the Runge-Kutta and
            low-storage marches keep none.
        """
        return None

    def restore_march(self, snapshot: MarchSnapshot) -> str | None:
        """Continue the Adams march from a snapshot at the loaded position.

        Called after :meth:`set_initial` at the snapshot's position. The state is
        replaced by the snapshot's own, exactly as the interrupted run held it.

        Parameters
        ----------
        snapshot : solprop.output.records.MarchSnapshot
            The snapshot a file holds.

        Returns
        -------
        str or None
            ``None`` when the history was continued, otherwise why the march starts
            afresh, as a resume did before snapshots were kept.
        """
        if self._adams is None:
            march = "low-storage" if self._low_storage is not None else "Runge-Kutta"
            return f"the run steps with the {march} march, which keeps no history"
        return self._adams.load_snapshot(snapshot)

    def restore_call_ledgers(
        self, logical: Mapping[str, object], atomic: Mapping[str, object] | None = None
    ) -> None:
        """Restore shared Adams call counts without restoring numerical history.

        Non-TDSE backends inherit logical budget persistence while refusing atomic
        counts. The caller first loads the saved field, which resets Adams history;
        the next source evaluation charges startup against these cumulative counts. A
        Runge-Kutta or low-storage march keeps no logical ledger, so nothing is
        installed for it.

        Parameters
        ----------
        logical : mapping
            Persisted logical response counts; empty on ordinary RK runs.
        atomic : mapping or None, optional
            Persisted atomic counts; absent on non-TDSE runs.

        Raises
        ------
        BackendError
            If this backend is handed counters it does not implement.
        """
        if atomic:
            raise BackendError(
                f"{self.name} does not restore atomic response call ledgers"
            )
        if logical and self._adams is not None:
            self._adams.restore_ledger(logical)

    # --- the per-stage bundle ------------------------------------------------------

    # --- the seam ----------------------------------------------------------------

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
        resume: bool = False,
    ) -> None:
        """Load a state and the noise realisation at a position.

        The state is masked to the active band once here, as the scaffold does: the
        operator and the prefactor are exactly zero outside it, so out-of-band content
        would neither propagate nor decay and would sit in the error norm for the whole
        run. The traditional ``input`` noise model adds its realisation to the
        state at the run's launch position and never again; the ``modified`` model keeps
        it apart and adds it at every stage, so the propagating field never carries it.

        A **resume** loads a state from a plane the run already saved, so it already
        carries whatever the ``input`` model added at the launch and must not be given a
        second realisation, and a replayed schedule continues from the edge that plane
        sits on. A `z` past the launch is always one. Position cannot tell for a file
        holding only its launch plane, which comes back at the launch itself: its caller
        says so with `resume`. The launch is the entrance unless the run starts inside
        the guide (SEP 37: an interior launch with ``input`` noise takes its draw
        there). The controller starts afresh from ``dz_init`` either way
        (``docs/dev/stepper.md`` section 3).

        Everything else a march reads is restored too, so that a backend loaded again
        after a run -- a :class:`~solprop.driver.runner.SimulationRunner`'s, or
        after a failed run -- repeats a fresh backend's run (SEP 22 section 4.7): a
        dense observation's look-ahead, an Adams march's call ledger and last-rejection
        flag, a low-storage march's source-evaluation count and the boundary monitor,
        with the collar it may have rested, here, and the quadrature alarm, whose value
        the entrance plane's ``quadrature_error`` reads before any march, in the
        overrides of the backends that keep one. A resume restores its
        ledger afterwards (:meth:`restore_call_ledgers`).

        Loading is the one entry that clears a low-storage step abandoned in flight
        (SEP 24 section 6.2): the state it holds is replaced, and the march's flow
        table is rebuilt from the next step.

        Parameters
        ----------
        field : numpy.ndarray
            The state in the canonical layout, complex, shape
            ``(n_modes, n_pol, n_omega)``, in the square root of watts and the FFT
            convention.
        noise : solprop.noise.NoiseRealisation or None
            The realisation to carry along the run, or ``None`` to take the one the
            problem was assembled with.
        z : float, optional
            The position the state belongs to, m. Zero, the entrance, by default.
        resume : bool, optional
            Whether `field` came back from a saved plane of this run, and so never takes
            the ``input`` draw, whatever `z`. ``False`` by default.

        Raises
        ------
        BackendError
            If the backend is closed, the field does not match the contract's shape,
            `z` lies outside the operator table, or a schedule is being replayed and `z`
            is not one of its edges.
        """
        self._check_open()
        values = np.ascontiguousarray(field, dtype=np.complex128)
        if values.shape != self._problem.shape:
            msg = (
                f"the initial state has shape {values.shape}, expected "
                f"{self._problem.shape} from the contract"
            )
            raise BackendError(msg)
        position = self._checked_position(float(z))
        cursor = self._schedule_index_at(position)
        self._workspace.load(values, noise, z=position, resume=resume)
        # The march starts afresh: its position, proposal, cursor and norms, and the
        # rest of what it reads, as a fresh backend holds it (SEP 22 4.7) -- a dense
        # observation's look-ahead, an Adams march's ledger and last-rejection flag,
        # a low-storage march's count and a step it left in flight.
        self._march.load(position, self._stepper.dz_init, cursor)
        self._alarm_reported = False
        self._history = []
        self._natural_count = 0
        self._natural_due = False
        self._halt_ahead = False
        self._halt_requested = False
        self._halt_check = None
        self._clock = time.perf_counter()
        # A resume is a second run for a reporter's purposes: it starts from a different
        # z and its own clock, so the next plane announces itself afresh.
        self._progress_started = False
        # The period counts from the load, so a resumed run's rows are not the
        # uninterrupted run's rows unless each plane's accepted-step count is a
        # multiple of the period. Said in `docs/guide/results/statistics.md`.
        self._accepted_steps = 0
        self._reset_step_stats(release=True)
        # A boundary monitor watches one run: the next attaches its own or none, and
        # a collar the last one rested goes back into the stages (SEP 21).
        self._monitor = None
        self._bind_monitor(None)
        self._set_collar_quiet(False)

    def set_progress(self, reporter: ProgressReporter) -> None:
        """Attach the reporter every attempted step is reported to.

        The next call to :meth:`propagate_to` announces the run to it. Attaching a
        reporter changes nothing the propagation computes -- the calls are observers,
        made after the step is decided -- so a run with one is bitwise the run without.

        Parameters
        ----------
        reporter : solprop.log.ProgressReporter
            Where the step events go; :class:`solprop.log.NullReporter` turns them off.
        """
        self._progress = reporter
        self._progress_started = False

    _stop: StopFlag | None = None
    """The flag the march polls (:meth:`set_stop`); ``None`` by default."""

    def set_stop(self, flag: StopFlag | None) -> None:
        """Attach the flag the march polls, or detach it with ``None``.

        A run polls its own flag before each trial (SEP 41, S17), where the state is
        the last accepted one, so a stop abandons nothing but the trial not yet taken.
        Polling changes nothing the propagation computes.

        Parameters
        ----------
        flag : solprop.log.StopFlag or None
            Any object with an ``is_set()`` method, such as a
            :class:`threading.Event`.
        """
        self._stop = flag

    def _poll_stop(self) -> None:
        """Raise when the run's stop flag is set; called before every trial or call."""
        poll(self._stop)

    def propagate_to(self, z: float) -> StepSummary:
        """Step the state to a position, landing on it exactly.

        Adaptive by default and a replay when the contract carries a schedule. Either
        way the embedded estimate is computed and recorded every step, so a schedule too
        coarse for the tolerance it is scaled by is reported rather than silent
        (``docs/dev/stepper.md`` section 8). The low-storage march has no estimate: it
        replays its schedule and records 0.0, "not estimated" (SEP 24 section 6.3).

        Parameters
        ----------
        z : float
            The save plane to reach, m, at or ahead of the current position and inside
            the operator table.

        Returns
        -------
        StepSummary
            What the call cost and achieved.

        Raises
        ------
        BackendError
            If the backend is closed, no initial state was loaded, a low-storage step
            was abandoned in flight, or ``z`` is behind the current position or outside
            the table.
        ConvergenceError
            If the controller asks for a step below the contract's ``dz_min``, or a
            replayed step's error norm, or the low-storage march's state, is not
            finite.
        """
        self._require_state()
        target = float(z)
        self._check_target(target)
        # `started` is announced here rather than from `set_initial`, because the driver
        # attaches the reporter after the state is loaded and because this is the first
        # moment at which the run is actually about to cost anything.
        if not self._progress_started:
            self._progress_started = True
            settings = self._stepper
            self._progress.started(self._z, settings.z_end, len(settings.save_planes))
        started = time.perf_counter()
        first = len(self._history)
        # Beside `_begin_plane` and not inside it: neither backend's override calls
        # `super()`, so a base-class body there would silently do nothing.
        if not self._halted:
            # A call resuming after a halt short of its target continues the same
            # plane: its rows and per-plane latches carry on (SEP 47 section 5.1).
            self._reset_step_stats(release=False)
            self._begin_plane()
        self._halted = False
        # Each march replays a schedule or steps adaptively itself; the low-storage
        # march replays only (SEP 24).
        self._march.advance(target)
        if self._natural_due and reached(self._z, target) and not self._march.holding:
            # The step owed as a step plane ends on the target itself, whose plane
            # observes it: it is not observed a second time there (`_natural_stop`).
            self._natural_due = False
        taken = self._history[first:]
        record = ControlRecord.from_norms(
            self._stepper.controller,
            [entry[2] for entry in taken if entry[3]],
            n_rejected=sum(1 for entry in taken if not entry[3]),
        )
        if not record.met_tolerance:
            self._logger.warning(
                "%s: a step was too coarse for the tolerance it is scaled by on the "
                "way to %.6e m; the largest scaled error norm was %.3e against the "
                "requested 1.0 (rtol %.1e)",
                self.name,
                target,
                record.max_norm,
                record.requested_rtol,
            )
        self._logger.debug(
            "%s: reached %.6e m in %d steps (%d rejected), achieved/requested "
            "tolerance ratio %s",
            self.name,
            self._z,
            record.n_accepted + record.n_rejected,
            record.n_rejected,
            # The run's integrator decides: the low-storage march estimates no error.
            NOT_ESTIMATED
            if isinstance(self._stepper.integrator, LowStorageRungeKutta)
            else f"{record.tolerance_ratio:.3e}",
        )
        if not self._halted:
            # A halt short of the target is no plane (SEP 47): no plane check there.
            self._monitor_plane()
        return StepSummary(
            z=self._z,
            n_steps=len(taken),
            n_rejected=record.n_rejected,
            max_norm=record.max_norm,
            mean_norm=record.mean_norm,
            wall_time=time.perf_counter() - started,
            met_tolerance=record.met_tolerance,
        )

    def observe(self, *, field: bool = True) -> Observation:
        """Return the state at the current position, as host arrays.

        The field is in the lab frame and the **FFT convention**
        (``spectral_convention = "fft"``): the output layer conjugates it on the way to
        disk and the processing accessors on the way to a user. The step
        history covers everything since the previous call.

        With ``field=False`` (SEP 22 section 5.1) neither the field nor the noise phase
        crosses to the host -- a device backend copies nothing -- and the observation
        carries ``None`` for both. Nothing scans the field for finiteness then: the
        step loop accepts only finite error norms and a replay raises on a non-finite
        one. The spatial collar diagnostic, which reads the field, is not evaluated;
        the backend counts such planes in :meth:`metadata` and logs the first.

        Parameters
        ----------
        field : bool, optional
            Whether the field and the noise phase cross to the host. Default ``True``.

        Returns
        -------
        Observation
            The field, the noise phase and the step history.

        Raises
        ------
        BackendError
            If the backend is closed or no initial state was loaded.
        """
        state = self._require_state()
        history = _history_record(self._history)
        whole = cast("np.ndarray", state) if field else None
        if whole is not None:
            # The state itself, frozen as an observed array always was so that the
            # next step replaces it rather than accepting in place, and held, so the
            # observation takes a view of it without a copy. A state that only views
            # memory the march owns is frozen but not held, and the observation copies
            # it.
            hold(whole)
        plane = Observation(
            z=self._z,
            field=whole,
            noise_phase=self._workspace.noise_phase(self._z) if field else None,
            step_history=history,
            n_steps=history.n_steps,
            n_rejected=history.n_rejected,
            step_stats=self._drain_step_stats(),
        )
        self._history = []
        self._clock = time.perf_counter()
        return plane

    @property
    def in_flight(self) -> tuple[float, float] | None:  # numpydoc ignore=RT01
        """Low-storage step a failure abandoned in flight, or ``None``.

        The step's start and length, m, while the state is neither of its ends
        (``solprop.backends.low_storage``).
        """
        return self._abandoned()

    def _abandoned(self) -> tuple[float, float] | None:
        """Return the step a failure left in flight, or ``None``: the march's."""
        return None if self._low_storage is None else self._low_storage.in_flight

    @property
    def position(self) -> float:  # numpydoc ignore=RT01
        """Where the march stands, m.

        The committed state's position, or the end of a low-storage step a failure
        abandoned, which leaves the state in flight (:attr:`in_flight`).
        """
        return float(self._z)

    @property
    def halted(self) -> bool:  # numpydoc ignore=RT01
        """Whether the last march stopped at a requested halt short of its end.

        True while the halt latch of :meth:`request_halt` or the yield check of
        :meth:`set_halt_check` stopped it.
        """
        return self._halted

    @property
    def holding_plane(self) -> bool:  # numpydoc ignore=RT01
        """Whether a dense observation holds the state between endpoints.

        True while the march's dense observation holds an interpolated plane, the
        march ahead of it.
        """
        return self._march.holding

    def drain_attempts(self) -> Observation:
        """Drain the attempts recorded since the last observation, with no field.

        For a state left in flight, which :meth:`observe` refuses: the attempts and
        periodic rows recorded before the abandoned step, drained once, as an
        observation drains them.

        Returns
        -------
        solprop.backends.protocol.Observation
            The attempts and rows at :attr:`position`, with no field.
        """
        entries, self._history = self._history, []
        history = _history_record(entries)
        return Observation(
            z=self.position,
            field=None,
            noise_phase=None,
            step_history=history,
            n_steps=history.n_steps,
            n_rejected=history.n_rejected,
            step_stats=self._drain_step_stats(),
        )

    def timing_probe(self, start: Start) -> TimingProbe:
        """Load a start and return single trials of this run's march from it.

        The start's field, noise, position and resume flag are loaded as by
        :meth:`set_initial`; its method history, ledgers and monitor continuation are
        not read. A Runge-Kutta trial leaves the loaded state alone; a low-storage
        march's step advances the discarded registers in place, so a backend probed so
        is closed afterwards (SEP 47 section 6.6).

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the trials start; its field is taken.

        Returns
        -------
        solprop.backends.protocol.TimingProbe
            The probe.
        """
        self.set_initial(
            np.asarray(start.take_field()),
            start.noise,
            z=float(start.z),
            resume=start.resume,
        )
        return _TrialProbe(
            cast("HostDriver[object]", self),
            (float(start.z), self._schedule_index),
        )

    # --- the step loop -------------------------------------------------------------

    def _begin_plane(self) -> None:
        """Reset whatever describes only the steps of the plane about to be taken.

        The quadrature alarm and its latch, so that each plane reports its own.
        """
        self._workspace.reset_quadrature()
        self._alarm_reported = False

    def _natural_stop(self, target: float) -> bool:
        """Count an accepted step; return whether to stop at its end to save it.

        With ``step_planes = k`` the end of every ``k``-th accepted step since the load
        is saved as a plane (SEP 14 phase 3): ``propagate_to`` returns there, short of
        its target, and the driver calls it again. A step that lands on the target is
        saved as the target; one that crosses a densely observed target is saved after
        it, when the next call finds the stop due (:meth:`_natural_stop_due`).

        - ``target``: the position the call is stepping to, m.
        """
        every = self._stepper.step_planes
        self._natural_count += 1
        if every == 0 or self._natural_count % every:
            return self._halted_short(target)
        if reached(self._z, target):
            self._natural_due = not reached(target, self._z)
            return False
        return True

    def request_halt(self) -> None:
        """Stop the march at the next accepted endpoint short of its target.

        The execution session's snapshot latch (SEP 47 section 5.5): ``propagate_to``
        returns at the end of the next accepted step, as at a natural stop, without
        counting it as one, so the session can service a snapshot there. An endpoint on
        the target itself is observed as the plane, and the latch stays set for the
        session to find.
        """
        self._halt_requested = True

    def clear_halt(self) -> None:
        """Withdraw a halt requested with :meth:`request_halt`, once it is serviced."""
        self._halt_requested = False

    def halt_ahead(self) -> None:
        """Halt the next march at the endpoint a dense observation held, if short.

        The execution session's snapshot latch beside interpolated planes (SEP 47
        section 5.5): the march stands ahead of the plane it observed, at the end of
        the step that crossed it. When the next march restores that endpoint and the
        latch is set, it returns there at once, unless a step plane is owed there
        first or the target is another plane the step crossed, so that the snapshot is
        taken at that endpoint rather than wait for a step that crosses no plane.
        """
        self._halt_ahead = True

    def set_halt_check(self, check: Callable[[], bool] | None) -> None:
        """Ask `check` at every accepted endpoint whether to halt there.

        The execution session's bounded yield (SEP 47 section 5.1): when `check`
        returns True the halt latch of :meth:`request_halt` is set, so the march
        returns at that endpoint. It is asked only at safe boundaries and changes
        nothing the march computes.

        Parameters
        ----------
        check : callable or None
            Returns whether a yield is due; ``None`` asks nothing.
        """
        self._halt_check = check

    def poll_halt(self) -> None:
        """Latch a halt when the yield check says one is due."""
        check = self._halt_check
        if check is not None and not self._halt_requested and check():
            self._halt_requested = True

    def _halted_short(self, target: float) -> bool:
        """Return whether a halt stops the march here, short of `target`.

        - ``target``: the position the call is stepping to, m.
        - Returns: True when a halt is requested or the yield check asks for one, and
          the endpoint is short of `target`.
        """
        if reached(self._z, target):
            return False
        self.poll_halt()
        if self._halt_requested:
            self._halted = True
        return self._halt_requested

    def _natural_stop_due(self, target: float) -> bool:
        """Return whether a step to save ends before `target`, with the march there.

        Called at the start of a march, once a dense observation's march state is
        restored: a step that crossed the last observed plane and was the ``k``-th is
        saved now, unless `target` is another plane it also crossed; otherwise a halt
        asked for there (:meth:`halt_ahead`) stops the march there.

        - ``target``: the position the call is stepping to, m.
        """
        ahead, self._halt_ahead = self._halt_ahead, False
        if self._natural_due and not reached(self._z, target):
            self._natural_due = False
            return True
        if ahead and self._halt_requested and not reached(self._z, target):
            # The restored endpoint of :meth:`halt_ahead`, every plane its step
            # crossed observed.
            self._halted = True
            return True
        return False

    _monitor: BoundaryMonitor | None = None
    """The run's boundary monitor (SEP 21), fed every committed step; ``None``."""

    _split_removed: float = 0.0
    """What the ``"split_step"`` collar took at the accepted steps since the monitor's
    last reading, in its units (:attr:`solprop.monitor.BoundaryEnergies.removed`):
    each accepted step's :class:`~solprop.backends.engines.AcceptRecord` adds to it
    (:meth:`_absorb`) and :meth:`_boundary_energies` drains it."""

    _split_found: Float[np.ndarray, " 4"] | None = None
    """The energy and the guards' and collar's energies of the power the last accepted
    step's split found (:attr:`solprop.monitor.BoundaryEnergies.found`), which the
    monitor judges the occupancies on: the latest accepted step's, and ``None`` before
    the first such step."""

    reads_transverse_edge: bool = False
    """Whether this backend reads a free-space run's transverse edge on the host
    (SEP 40 section 5.3). A backend that does not leaves the run ``unwatched``, and
    says so before the first step."""

    _collar_quiet: bool = False
    """Whether the modal stage leaves the in-stage temporal collar term out: the run
    opted in, the monitor last found the collar quiet (:data:`solprop.monitor.QUIET`)
    and the route can rest it (:meth:`_collar_can_rest`; SEP 21, ADR-057)."""

    def attach_monitor(self, monitor: BoundaryMonitor) -> None:
        """Watch the time window's edges from here on, starting with the current state.

        The launch is recorded now; every committed step is then accounted at the
        same point as the periodic statistics rows (``_take_step_stats``), on
        every march, after the state is finished and the position advanced.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            A monitor with nothing recorded.

        Raises
        ------
        BackendError
            If the monitor's weights do not match the grid, or its columns the state.
        """
        n_t = self._problem.grids.n_t
        n_modes, _, _ = self._problem.shape
        columns = monitor.columns
        given = None if columns is None else columns.shape
        if monitor.weights.shape != (4, n_t) or given not in {None, (n_modes,)}:
            msg = (
                f"{self.name}: the boundary monitor's weights {monitor.weights.shape} "
                f"or columns {given} do not match the grid's {n_t} samples and the "
                f"state's {n_modes} modes"
            )
            raise BackendError(msg)
        if monitor.split:
            # The split collar's crossing limit is read again at every check.
            monitor.crossing = self._problem.split_crossing
        self._monitor = monitor
        self._bind_monitor(monitor)
        workspace = self._workspace
        # The noise draw is a host array already, read once.
        monitor.launch(
            self._boundary_energies(),
            self._z,
            floor=workspace.input_floor,
        )
        self._set_collar_quiet(monitor.collar_quiet)
        # A resumed fixed schedule is held to the state it resumes from at once.
        self._refuse_scheduled_crossing(monitor)
        edge = monitor.transverse
        if edge is not None:
            floor = workspace.input_floor
            schedule = self._stepper.schedule
            self._require_state()
            edge.launch(
                workspace.edge_reading(
                    edge,
                    # A resumed launch reads no share, but continues the current.
                    full=not edge.resumed,
                    current=edge.every_step and edge.resumed,
                ),
                self._z,
                noise=None if floor is None else np.asarray(floor),
                clean_columns=workspace.clean_columns(edge),
                steps=None if schedule is None else np.asarray(schedule.steps),
                reads=self.reads_transverse_edge,
                declared=_declared_discard(self._problem),
            )

    def _read_transverse(self, monitor: BoundaryMonitor, *, plane: bool) -> None:
        """Read the transverse edge where its spacing or a saved plane asks (SEP 40).

        - ``plane``: whether a saved plane has just been reached, which is always read.
        """
        edge = monitor.transverse
        if edge is None:
            return
        # Read before the next step would carry past the read (`TransverseEdge.due`).
        due = edge.due(self._z, self._next_step())
        # A plane the step that reached it already read has nothing to add.
        read = (plane or due) and edge.wants_read(self._z)
        # A collar's inward current accumulates at every committed step.
        accumulate = edge.every_step and not plane
        if not (read or accumulate):
            return
        self._require_state()
        reading = self._workspace.edge_reading(edge, faces=read, current=accumulate)
        if accumulate:
            assert reading.current is not None
            edge.step(reading.current, self._z)
        if read:
            edge.read(reading, self._z)

    def edge_memory(self, space: object) -> int:
        """Return what reading the edge holds in the state's memory space.

        What :func:`solprop.driver.estimate.estimate` adds: it attaches no monitor.

        Parameters
        ----------
        space : object
            The run's transverse grid.

        Returns
        -------
        int
            Zero by default: a read on the host holds nothing beside the state there.
        """
        del space
        return 0

    def _next_step(self) -> float:
        """Return the step the march takes next, m.

        A replay takes its schedule's and never updates the proposal, which stays
        ``dz_init``: read from that, an edge read comes early or late by the difference.

        - Returns: the schedule's next step on a replay, zero past its end, and
          otherwise the controller's proposal.
        """
        schedule = self._stepper.schedule
        if schedule is None:
            return self._proposal
        if self._schedule_index >= schedule.n_steps:
            return 0.0
        return float(schedule.steps[self._schedule_index])

    def _transverse_capped(self, proposal: float, z: float) -> float:
        """Return the proposal, held to the transverse edge's read spacing (SEP 40 5.3).

        A low-source run commits long steps, and light could cross the edge unseen
        within one; the read spacing is a ceiling on the step instead. The step is not
        made to land on a read: the edge is read before a step would pass one.

        - ``proposal``: the controller's proposal, m.
        - ``z``: where the step starts, m.
        - Returns: the proposal, or the read spacing, m.
        """
        monitor = self._monitor
        edge = None if monitor is None else monitor.transverse
        if edge is None:
            return proposal
        return min(proposal, edge.ceiling(z))

    def _split_capped(self, proposal: float) -> float:
        """Return the proposal, held to the split collar's crossing limit.

        The boundary monitor reads the limit from the state's occupancy at every
        check (:attr:`solprop.monitor.BoundaryMonitor.step_cap`): light that spreads
        into faster-walking bins shortens the steps until the next check, which reads
        it again.

        - ``proposal``: the controller's proposal, m.
        - Returns: the proposal, or the cap, m.
        """
        monitor = self._monitor
        return proposal if monitor is None else min(proposal, monitor.step_cap)

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Read a boundary monitor's energies from now on, or stop reading them.

        Called when a monitor is attached, with nothing yet taken by a split, and
        with ``None`` when a state is loaded: either empties the split collar's
        accumulator (SEP 52 section 5.2), and the workspace takes the split's sums at
        each accepted step's finish while a monitor of that scheme is bound.

        - ``monitor``: the run's monitor, or ``None`` to stop.
        """
        self._split_removed = 0.0
        self._split_found = None
        self._workspace.bind_monitor(monitor)

    def _absorb(self, record: AcceptRecord) -> None:
        """Accumulate an accepted step's split collar's sums (SEP 52 section 5.2).

        The removal adds over every accepted step since the previous check, which the
        next reading drains; the found occupancies are the latest accepted step's. A
        step that took no split sums -- no collar of that scheme, or no monitor --
        leaves both as they were.
        """
        if record.found is not None:
            self._split_removed += record.removed
            self._split_found = record.found

    def _boundary_energies(self) -> BoundaryEnergies:
        """Return the boundary monitor's reading of the state (SEP 21, ADR-057).

        The workspace's reading of the committed state (its energy, its four weighted
        energies and its occupancy), with what the split collar took at the accepted
        steps since the previous reading, which this drains, and the energies of the
        power the last step's split found.
        """
        monitor = self._monitor
        if monitor is None:
            raise BackendError(f"{self.name}: no boundary monitor is attached")
        removed, self._split_removed = self._split_removed, 0.0
        reading = self._workspace.reading()
        return BoundaryEnergies(
            total=reading.total,
            regions=reading.regions,
            removed=removed,
            found=self._split_found,
            occupancy=reading.occupancy,
        )

    def _check_boundaries(
        self, monitor: BoundaryMonitor, *, plane: bool = False
    ) -> None:
        """Account the committed step at the current position with the monitor.

        - ``plane``: true for a saved plane's check of a step the monitor already
          skipped.
        """
        self._require_state()
        self._check_at(
            monitor,
            self._boundary_energies(),
            self._z,
            self._schedule_index,
            plane=plane,
        )

    def _check_at(
        self,
        monitor: BoundaryMonitor,
        energies: BoundaryEnergies,
        z: float,
        cursor: int,
        *,
        plane: bool = False,
    ) -> None:
        """Book a reading of the step committed at `z` with the monitor (SEP 45 4.10).

        - ``z``: the step's end, m.
        """
        monitor.step(energies, z, plane=plane)
        self._set_collar_quiet(monitor.collar_quiet)
        self._refuse_scheduled_crossing(monitor, cursor=cursor, z=z)

    def _refuse_scheduled_crossing(
        self,
        monitor: BoundaryMonitor,
        *,
        cursor: int | None = None,
        z: float | None = None,
    ) -> None:
        """Stop a fixed schedule whose steps before the next check pass the cap.

        Translation admitted the schedule on the launch's occupancy; light that has
        since spread into faster-walking bins lowers the cap, and a schedule cannot
        shorten its steps (:class:`solprop.boundaries.SplitCrossing`). The cap holds
        until the next check, which the monitor's cadence makes due within
        :meth:`~solprop.monitor.BoundaryMonitor.steps_to_check` steps at most, so the
        longest of those is held to it. Read at the launch too, so that a resumed
        schedule is held to the state it resumes from.

        - ``cursor``: the replay cursor after the step read; ``None`` for the driver's
          own.
        - ``z``: the step's end, m; ``None`` for the driver's position.
        """
        schedule = self._stepper.schedule
        first = self._schedule_index if cursor is None else cursor
        position = self._z if z is None else z
        if schedule is None or first >= schedule.n_steps:
            return
        ahead = schedule.steps[first : first + monitor.steps_to_check()]
        step = float(np.max(ahead))
        cap = monitor.step_cap
        if step > cap:
            msg = (
                f"at z = {position:.6e} m the split temporal collar's crossing limit "
                f"fell to {cap:.3e} m, below the fixed schedule's step {step:.3e} m "
                "before the next check: light has spread into bins that walk faster "
                "than the launch's, and a step that long could carry it through the "
                "collar and round the window undamped. Take shorter steps, step "
                'adaptively, or use absorber_scheme="nonlinear"'
            )
            raise BoundaryError(msg)

    def _set_collar_quiet(self, quiet: bool) -> None:
        """Say whether the collar is quiet.

        The term is left out only where the route can rest it
        (:meth:`_collar_can_rest`), and the monitor is told what the stages do, so
        that it adds no removal while the term is out. A change drops a reused
        first-same-as-last stage, which was evaluated under the other setting
        (:meth:`_collar_changed`); a backend whose stages run elsewhere passes the
        flag on there.
        """
        quiet = quiet and self._collar_can_rest()
        if self._monitor is not None:
            self._monitor.note_collar(quiet)
        if quiet != self._collar_quiet:
            self._collar_quiet = quiet
            self._workspace.collar_quiet = quiet
            self._collar_changed()

    def _quiet_capped(self, proposal: float) -> float:
        """Return the proposal, held to the monitor's interval while the collar rests.

        A left-out collar absorbs nothing, and the monitor can only see the committed
        states: one long step could carry light from a quiet collar deep into it,
        unabsorbed. While the term is out no step is longer than the interval the
        monitor checks a quiet window at (:attr:`BoundaryMonitor.interval`, the
        guide's length over :data:`solprop.monitor.CHECKS`), so light has to walk
        about twenty windows over the run to cross the collar between two checks.

        - ``proposal``: the controller's proposal, m.
        - Returns: the proposal, or the monitor's interval while the collar is left out,
          m.
        """
        monitor = self._monitor
        if not self._collar_quiet or monitor is None:
            return proposal
        return min(proposal, monitor.interval)

    def _collar_can_rest(self) -> bool:
        """Return whether this run's stages may leave a quiet temporal collar out.

        The one rule every backend reads (:func:`solprop.backends.records.collar_rests`,
        SEP 21 D15).
        """
        return collar_rests(self._problem, self._stepper)

    def _collar_changed(self) -> None:  # noqa: B027 - optional by design
        """Drop whatever was evaluated under the other collar setting.

        The default holds nothing; a backend that reuses the last stage of a step as
        the first of the next forgets it here.
        """

    def _monitor_plane(self) -> None:
        """Close the boundary monitor's row for the plane just reached (SEP 21, D5).

        A plane's state is checked even when its last step was not.
        """
        monitor = self._monitor
        if monitor is None:
            return
        if monitor.stale:
            self._check_boundaries(monitor, plane=True)
        self._read_transverse(monitor, plane=True)
        monitor.plane(self._z)

    def _take_step_stats(self) -> None:
        """Record one statistics row when this accepted step is the period-th.

        Called by every march (:meth:`_SessionHooks.stepped`) **after** the engine's
        accept and after the march's position is advanced, which is the only point at
        which both describe the step just committed: an adaptive march records the
        attempt before it accepts and a replay after, so anything reading the state or
        the position from inside ``_on_accepted_step`` would be one accepted step stale
        on an adaptive run and correct on a replayed one.

        The row is :meth:`stats` of the periodic subset, so there is exactly one
        implementation of every statistic and a row taken at a save plane is bitwise
        what that plane records; a second implementation of a statistic could drift
        from the first.
        """
        self._monitor_step()
        self._statistics_step()

    def _monitor_step(self) -> None:
        """Check or skip the step just committed, as the monitor's cadence asks.

        The boundary monitor's part of :meth:`_take_step_stats`: a quiet window is
        checked every few steps, without reading the state in between
        (:data:`solprop.monitor.CADENCE`), and the transverse edge read where due.
        """
        monitor = self._monitor
        if monitor is not None:
            if monitor.due(self._z):
                self._check_boundaries(monitor)
            else:
                monitor.skip(self._z)
            self._read_transverse(monitor, plane=False)

    def _statistics_step(self) -> None:
        """Take the statistics row of the step just committed when it is due.

        The statistics part of :meth:`_take_step_stats`, at the driver's position.
        """
        if self._row_spec is None:
            return
        self._accepted_steps += 1
        if self._accepted_steps % self._row_period:
            return
        # The attempt this row belongs to is the one `_record` has just appended, and
        # the history counts rejections, so the index is not `row * period`.
        self._record_step_stats(self._row_spec, self._z, len(self._history) - 1)

    def _record(self, z: float, dz: float, norm: float, accepted: bool) -> None:
        """Append one step to the history since the last observation.

        - ``z``: position at the end of the step, m.
        - ``dz``: length of the step, m.
        - ``norm``: the scaled error norm as executed: a landing step clipped short
          records its own, not the proposal's equivalent. A diverged attempt's
          (overflowed or NaN) is recorded as infinity, which the controller never
          accepts.
        """
        if not math.isfinite(norm):
            norm = math.inf
        self._history.append((z, dz, norm, accepted, time.perf_counter() - self._clock))
        self._progress.step(z, dz, accepted)
        if accepted:
            self._on_accepted_step()

    # --- noise ---------------------------------------------------------------------

    # --- guards --------------------------------------------------------------------

    def _require_state(self) -> _StateT:
        """Return the current state, refusing a closed or unloaded backend."""
        self._check_open()
        state = self._workspace.state
        if state is None:
            msg = "no initial state: call set_initial before propagating or observing"
            raise BackendError(msg)
        self._refuse_in_flight()
        return cast("_StateT", state)

    def _refuse_in_flight(self) -> None:
        """Refuse every entry that reads or advances a state left between two steps.

        The low-storage march accepts each step in place (SEP 24 section 6.2): from
        the step's start to its commit the state is neither the step's start nor its
        end, so an exception or a signal between them leaves nothing a reader may use
        and nothing a march may continue from. Only :meth:`set_initial` clears it.
        """
        pending = self._abandoned()
        if pending is not None:
            raise BackendError(in_flight_refusal(self.name, *pending))

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the backend has released its buffers, as a failure may."""
        return self._closed

    def _check_open(self) -> None:
        """Refuse a backend that has been closed."""
        if self._closed:
            msg = "this backend is closed; its buffers were released"
            raise BackendError(msg)

    def _checked_position(self, z: float) -> float:
        """Return a position the state may be loaded at, refusing one off the table.

        - ``z``: the requested position, m.
        """
        low, high = self._problem.linear.z_span
        if not (math.isfinite(z) and low <= z <= high):
            msg = (
                f"a state can only be loaded inside the operator table, which covers "
                f"[{low:.6g}, {high:.6g}] m, and z = {z!r} does not"
            )
            raise BackendError(msg)
        return z

    def _schedule_index_at(self, z: float) -> int:
        """Return how many steps of a replayed schedule a position has consumed.

        - ``z``: the position the state is loaded at, m. On a replay it must be an edge;
          every save plane is one (``validate`` checks it), so the refusal fires only
          for a position the run never stopped at, from which a replay could not
          continue in step.
        """
        schedule = self._stepper.schedule
        if schedule is None:
            return 0
        edges = schedule.edges
        index = int(np.argmin(np.abs(edges - z)))
        if abs(float(edges[index]) - z) > replay_tolerance(schedule, z):
            msg = (
                f"a replayed schedule can only be resumed at one of its edges, and "
                f"z = {z:.6g} m is not one; the nearest is {float(edges[index]):.6g} m"
            )
            raise BackendError(msg)
        return index

    def _check_target(self, target: float) -> None:
        """Refuse a save plane behind the current position or outside the table.

        - ``target``: the requested position, m.
        """
        if not math.isfinite(target):
            msg = f"the save plane must be finite, got {target!r}"
            raise BackendError(msg)
        if target < self._z:
            msg = (
                f"cannot propagate backwards: asked for z = {target:.6g} m from "
                f"{self._z:.6g} m"
            )
            raise BackendError(msg)
        span = self._problem.linear.z_span
        if target > span[1]:
            msg = (
                f"z = {target:.6g} m lies beyond the operator table, which covers "
                f"[{span[0]:.6g}, {span[1]:.6g}] m"
            )
            raise BackendError(msg)


def _declared_discard(problem: PropagationProblem) -> bool:
    """Return whether a run declared content above ``discard_above`` (SEP 40 5.3).

    - Returns: True when its transverse specification sets ``discard_above``.
    """
    design = transverse_of(problem.route)
    return design is not None and design.specification.discard_above is not None


def _history_record(
    entries: Sequence[tuple[float, float, float, bool, float]],
) -> BackendHistory:
    """Return the attempts since the last observation as a history record.

    - `entries`: one ``(z, dz, error_norm, accepted, wall_time)`` per attempt.
    - Returns: the attempts, in arrays made here and held, so the record takes them
      without a copy.
    """
    columns = list(zip(*entries, strict=True)) if entries else [()] * 5
    return BackendHistory(
        z=hold(np.array(columns[0], dtype=np.float64)),
        dz=hold(np.array(columns[1], dtype=np.float64)),
        error_norm=hold(np.array(columns[2], dtype=np.float64)),
        accepted=hold(np.array(columns[3], dtype=np.bool_)),
        wall_time=hold(np.array(columns[4], dtype=np.float64)),
    )
