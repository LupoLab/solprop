# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference backend: its lowering, its marches and the refusals (SEP 52).

One :class:`NumpyRefBackend` steps one problem in plain NumPy. It builds the
workspace that holds the state and does every state-sized operation
(:class:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace`) and, for each run, the
engines a march asks of it
(:class:`~solprop.backends.numpy_ref.engines.NumpyRungeKutta`,
:class:`~solprop.backends.numpy_ref.multistep.HostHistory`,
:class:`~solprop.backends.low_storage.HostLowStorage`), the run's march and its dense
output (:meth:`NumpyRefBackend.method`); it keeps ``prepare``, the metadata, the
memory ledger, the timing probe and ``close``. Every run is the reference's own
session's (:class:`~solprop.backends.host.session.HostSession`), which schedules it.
The step control follows ``docs/dev/stepper.md`` section 3 exactly: the controller's
*proposal* is its state, the executed step's own norm decides acceptance and is what
the history records, and the equivalent norm of the proposal is what the controller
and a rejection see.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from jaxtyping import Complex

from solprop._helpers import Helpers
from solprop.backends.exponential import (
    CarrierReference,
    refuse_carrier_frame,
    refuse_formulation,
)
from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.runge_kutta import RungeKuttaMarch
from solprop.backends.host.session import HostMethod, HostPrepared, start_position
from solprop.backends.ledger import CallLedger
from solprop.backends.low_storage import HostLowStorage
from solprop.backends.node_chunk import NodeChunkPlan, plan_node_chunk
from solprop.backends.numpy_ref.engines import NumpyRungeKutta
from solprop.backends.numpy_ref.workspace import NumpyWorkspace
from solprop.backends.protocol import TimingProbe
from solprop.backends.records import adams_policy, runge_kutta_policy
from solprop.backends.session import ExecutionOptions, PreparedExecution, Start
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.problem import (
    MAX_N_POL,
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
    validate_run,
)
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    SpatialRoute,
    route_name,
    tdse_descriptor,
)
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

if TYPE_CHECKING:
    from solprop.backends.protocol import Backend

__all__ = ["NumpyRefBackend"]

logger = logging.getLogger(__name__)


class NumpyRefBackend:
    """Propagate one problem in plain NumPy: the oracle.

    The reference of ``docs/dev/backends.md`` section 2. It implements
    :class:`solprop.backends.protocol.Backend` and steps a single problem on either
    grid, on every route -- mode-averaged or multimode on the fixed transverse
    quadrature, scalar or vector-isotropic, the general modal guide, the spatial
    geometries and the TDSE response -- with every march: interaction-picture
    Runge-Kutta, the low-storage step and both Adams formulations. Mode-averaged
    propagation is the one-node case of the modal pipeline and not a second code path.

    **Refusals.** A problem is refused by name, with
    :class:`solprop.exceptions.BackendError`, when its route is not one of the union's,
    when an Adams method meets what it does not step, when a declared Raman term meets
    two polarisation components without a tensor kernel, or when the transform and the
    state would both carry the transverse components. A term silently dropped would
    run, converge and be wrong.

    **Convention.** Every array held or returned is in the FFT convention of
    ``docs/dev/conventions.md`` section 2; the backend never conjugates. The
    output layer converts on the way to disk and the processing accessors on the way to
    a user.

    **The grid.** The statistics delegate to :func:`solprop.stats.collect`, which needs
    a :class:`solprop.grid.TimeGrid`; the kernel contract carries the array group
    :class:`solprop.translate.masks.Grids` and no grid object, so the grid is supplied
    here instead. This is a known gap in the contract.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated kernel problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies: method, error control, itinerary, planes and statistics.
        :func:`~solprop.translate.problem.validate_run` checks the pair again here,
        before anything is allocated: validation is the backend's defence, not the
        caller's courtesy.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, needed only by the statistics.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, needed only by the ``zdw`` statistic.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name, needed only by the profile statistics.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Driver-owned persistent atomic plans, required exactly when the problem
        declares a TDSE response. The driver closes this binding after the backend.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options, from which the node chunk is planned; ``None``
        for the automatic choice.

    Attributes
    ----------
    name : str
        The registry name, ``"numpy_ref"``.
    reads_transverse_edge : bool
        True: the reference reads a free-space run's transverse edge on the host.
    helpers : solprop._helpers.Helpers
        The uncompiled stage-table and Adams-basis evaluations, ``PLAIN``, which the
        engines and marches are passed rather than choosing one (SEP 29 section 6.6).

    Raises
    ------
    BackendError
        Naming what the backend refuses.

    Examples
    --------
    >>> from solprop.backends.numpy_ref import NumpyRefBackend
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.assembly import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> from solprop.backends.events import Progress
    >>> from solprop.backends.session import Start
    >>> backend = NumpyRefBackend(problem, request, grid=grid)
    >>> session = backend.prepare(problem, request).start(Start.entrance(problem))
    >>> kinds = []  # the barriers; a bounded yield on a slow host adds a Progress
    >>> while not session.finished:
    ...     batch = session.advance()
    ...     kinds += [
    ...         type(event).__name__
    ...         for event in batch.events
    ...         if not isinstance(event, Progress)
    ...     ]
    ...     session.acknowledge(batch.receipt)
    >>> kinds
    ['ObservationReady', 'ObservationReady', 'Terminal']
    >>> session.close()
    """

    reads_transverse_edge: bool = True

    name: str = "numpy_ref"
    # The oracle evaluates its stage tables and Adams bases with the uncompiled forms,
    # never with the compiled code it exists to check (SEP 29 section 6.6).
    helpers: Helpers = NumpyWorkspace.helpers
    # The reference forms `exp(+dPhi)` and `exp(-dPhi)` afresh at every stage, so
    # keeping the six distinct exponents of a step is worth its memory here where it is
    # not on a backend that takes both signs from one transcendental.
    propagator_default: PropagatorMode = "tabulate"

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
        options: ExecutionOptions | None = None,
    ) -> None:
        """Adopt a validated problem and request; refuse what is not executed here."""
        _refuse_unsupported(problem, request.stepper)
        validate_run(problem, request)
        if (tdse_descriptor(problem.route) is None) != (tdse_binding is None):
            raise BackendError(
                "TDSE problems require an explicit host response binding"
            )
        self._problem = problem
        self._request = request
        # The stepping policy as one record, read by the marches.
        self._stepper = request.stepper
        # SEP 47 section 6.5: the node chunk is an execution choice, planned from the
        # problem and the options before anything is allocated.
        self._options = ExecutionOptions() if options is None else options
        self._chunk_plan: NodeChunkPlan = plan_node_chunk(problem, self._options)
        self._closed = False
        self._workspace = NumpyWorkspace(
            problem,
            request,
            chunk_plan=self._chunk_plan,
            propagator_mode=self._propagator_mode,
            grid=grid,
            modes=modes,
            profiles=profiles,
            tdse_binding=tdse_binding,
            name=self.name,
        )
        stepper = self._stepper
        integrator = stepper.integrator
        # The Adams method and its compensation's treatment (SEP 16 Q8), which a test
        # witness may switch (`_use_adams`).
        self._adams_method: Multistep | None = (
            integrator if isinstance(integrator, Multistep) else None
        )
        self._compensation: Literal["transported", "untransported"] = "transported"
        # The latest run's method, whose counts the metadata reports: a session's or the
        # timing probe's, each built afresh (SEP 22 section 4.7).
        self._run = self.method(None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "numpy_ref: %s grid, shape %s, %d Phi intervals, %s propagators, "
                "%s schedule",
                problem.grid_kind,
                problem.shape,
                problem.linear.n_intervals,
                self._propagator_mode,
                "replayed" if stepper.schedule is not None else "adaptive",
            )

    # --- the lowering of one run ---------------------------------------------------

    def method(self, host: object) -> HostMethod:
        """Return a fresh march for one run, its dense output and its history's rows.

        Each run steps with engines and a march of its own, over the backend's
        workspace, so that a backend loaded again repeats a fresh backend's run
        (SEP 22 section 4.7): its call ledger, its counts, its held trial and its
        dense output start afresh. The run's session is the march's host.

        Parameters
        ----------
        host : object
            The run's session, the march's
            :class:`~solprop.backends.host.marches.MarchHost`; ``None`` for a march
            that never steps.

        Returns
        -------
        solprop.backends.host.session.HostMethod
            The march, with its dense output when the run interpolates its planes and
            the Adams history's rows.
        """
        stepper = self._stepper
        integrator = stepper.integrator
        if self._adams_method is not None:
            method = self._adams(self._adams_method, host)
        elif isinstance(integrator, LowStorageRungeKutta):
            # The host engine is the oracle (SEP 24 section 6.5).
            schedule = stepper.schedule
            assert schedule is not None  # the translation refused it without one
            engine = HostLowStorage(
                self._workspace, integrator, schedule, self._propagator_mode
            )
            method = HostMethod(
                HostLowStorageMarch(
                    engine,
                    integrator,
                    host,
                    schedule=schedule,
                    proposal=stepper.dz_init,
                    lag=stepper.controller.lag,
                    name=self.name,
                )
            )
        else:
            method = self._runge_kutta_method(host)
        self._run = method
        return method

    def _runge_kutta_method(self, host: object) -> HostMethod:
        """Return the Runge-Kutta march over a fresh engine, and its dense output."""
        from solprop.backends.numpy_ref.dense import (  # noqa: PLC0415 - it builds on the engines
            RungeKuttaDense,
        )

        stepper = self._stepper
        engine = self._runge_kutta(self._workspace)
        policy = runge_kutta_policy(self._problem, stepper)
        march = RungeKuttaMarch(
            engine,
            policy,
            host,
            schedule=stepper.schedule,
            proposal=stepper.dz_init,
            name=self.name,
        )
        dense = RungeKuttaDense(self._workspace, engine) if policy.interpolate else None
        return HostMethod(march, dense)

    def _runge_kutta(self, workspace: NumpyWorkspace) -> NumpyRungeKutta:
        """Return the Runge-Kutta engine over the workspace; a test may script one."""
        return NumpyRungeKutta(workspace, self._stepper.tableau)

    def _use_adams(
        self,
        method: Multistep,
        compensation: Literal["transported", "untransported"] = "transported",
    ) -> None:
        """Step the next runs with this Adams method and compensation treatment.

        - ``compensation``: how a varying operator's linear compensation is integrated
          (SEP 16 Q8): ``"transported"``, the delivered treatment, or
          ``"untransported"``, kept as a test witness only.
        """
        self._adams_method = method
        self._compensation = compensation
        self._run = self.method(None)

    def _adams(self, method: Multistep, host: object) -> HostMethod:
        """Return the Adams march over a fresh engine, its dense output and rows."""
        # The engine reads this package's passes, so it is imported when a run asks
        # for it rather than with the package.
        from solprop.backends.numpy_ref.dense import AdamsDense  # noqa: PLC0415
        from solprop.backends.numpy_ref.multistep import (  # noqa: PLC0415
            AdamsKernels,
            ExponentialKernels,
            HostHistory,
        )

        compensation = self._compensation
        stepper = self._stepper
        interpolate = bool(stepper.interpolate_planes)
        engine = HostHistory(
            self._workspace,
            method,
            AdamsKernels(helpers=self.helpers),
            # SEP 16's exponential formulation keeps untransported samples in the same
            # store and shares everything else: the lifecycle, the controller, the
            # ledger.
            ExponentialKernels(helpers=self.helpers)
            if method.formulation == "exponential"
            else None,
            compensation,
            interpolate=interpolate,
        )
        carrier = engine.carrier
        # Every number the controller and the landings read, from one record (SEP 39
        # section 4.2). An in-stage temporal collar's largest rate, the one stiff
        # term the exponential history integrates, bounds each order's stable step.
        policy = adams_policy(
            self._problem,
            stepper,
            method,
            carrier=isinstance(carrier, CarrierReference) and carrier.active,
        )
        march = HostAdamsMarch(
            engine,
            method,
            host,
            policy=policy,
            field=self._workspace.committed,
            schedule=stepper.schedule,
            interpolate=interpolate,
            shape=self._problem.shape,
            proposal=stepper.dz_init,
            lag=stepper.controller.lag,
            name=self.name,
        )
        dense = AdamsDense(self._workspace, engine, march) if interpolate else None
        return HostMethod(march, dense, engine.rows)

    # --- the backend protocol --------------------------------------------------------

    @property
    def request(self) -> RunRequest:  # numpydoc ignore=RT01
        """Request this backend was built for.

        Its method, error control, itinerary, planes and statistics; a run adapts
        it with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        return self._request

    @property
    def workspace(self) -> NumpyWorkspace:  # numpydoc ignore=RT01
        """The workspace: the state and every state-sized operation on it."""
        return self._workspace

    @property
    def march(
        self,
    ) -> RungeKuttaMarch | HostAdamsMarch | HostLowStorageMarch:  # numpydoc ignore=RT01
        """The latest run's march: a session's, the timing probe's, or a fresh one."""
        return self._run.march

    @property
    def node_chunk(self) -> int:  # numpydoc ignore=RT01
        """The nodes a response pass of this backend holds at once."""
        return self._chunk_plan.chunk

    @property
    def chunk_plan(self) -> NodeChunkPlan:  # numpydoc ignore=RT01
        """Node-chunk plan and what chose it."""
        return self._chunk_plan

    @property
    def _propagator_mode(self) -> PropagatorMode:
        """Return the propagator mode this run uses, the contract's ``"auto"`` resolved.

        One volume of ``exp(h D / 2)`` reused while the step holds is the cheaper
        low-storage flow on a CPU, and a varying operator changes it every step (SEP 24
        section 4.3); a separable operator recomputes.
        """
        asked = self._stepper.propagator_mode
        if asked == "auto" and isinstance(
            self._stepper.integrator, LowStorageRungeKutta
        ):
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
        return HostPrepared(cast("Backend", self), problem, request)

    def timing_probe(self, start: Start) -> TimingProbe:
        """Load a start and return single trials of this run's march from it.

        The start's field, noise, position and resume flag are loaded as a session
        loads them; its method history, ledgers and monitor continuation are not read.
        A Runge-Kutta trial leaves the loaded state alone; a low-storage march's step
        advances the discarded registers in place, so a backend probed so is closed
        afterwards (SEP 47 section 6.6).

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the trials start; its field is taken.

        Returns
        -------
        solprop.backends.protocol.TimingProbe
            The probe.

        Raises
        ------
        BackendError
            If the backend is closed or the start does not fit the problem.
        """
        self._check_open()
        values = np.ascontiguousarray(start.take_field(), dtype=np.complex128)
        if values.shape != self._problem.shape:
            msg = (
                f"the initial state has shape {values.shape}, expected "
                f"{self._problem.shape} from the contract"
            )
            raise BackendError(msg)
        position, cursor = start_position(self._problem, self._stepper, float(start.z))
        self._workspace.load(values, start.noise, z=position, resume=start.resume)
        self._workspace.bind_monitor(None)
        self._workspace.collar_quiet = False
        method = self.method(None)
        march = method.march
        if isinstance(march, HostLowStorageMarch):
            return _TrialProbe(self, march.engine, position, cursor, low_storage=True)
        engine = (
            march.engine
            if isinstance(march, RungeKuttaMarch)
            else self._runge_kutta(self._workspace)
        )
        return _TrialProbe(self, engine, position, cursor, low_storage=False)

    def available_device_memory(self) -> int | None:
        """Return None for host execution.

        Returns
        -------
        int or None
            None: the reference runs on the host.
        """
        return None

    def metadata(self) -> dict[str, object]:
        """Return what this execution was, for the run's record.

        A saved run has to say what produced it: the thread configuration alone was
        measured to move one step by a factor of nine, so a result whose execution is
        not recorded is not reproducible. The record is the name, the resolved
        propagator mode and the node chunk; for an Adams run the latest run's
        cumulative logical source calls, and otherwise its physical
        ``source_evaluations`` (SEP 24 section 5); and the workspace's facts: the
        benchmark-gated plasma source and a host source's ledger, execution and last
        evaluation. A PEC predictor is explicitly labelled and is never reported as an
        accepted-field diagnostic.

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
        march = self._run.march
        if isinstance(march, RungeKuttaMarch):
            engine = march.engine
            record["source_evaluations"] = (
                engine.evaluations if isinstance(engine, NumpyRungeKutta) else 0
            )
        elif isinstance(march, HostLowStorageMarch):
            record["source_evaluations"] = march.source_evaluations
        else:
            ledger: CallLedger = march.ledger
            record["response_calls"] = ledger.to_record()
        record.update(self._workspace.metadata())
        return record

    def march_export_refusal(self) -> str | None:
        """Return ``None``: every march the reference admits carries its history.

        Returns
        -------
        None
            The Adams engines export and install their rows; the Runge-Kutta and
            low-storage marches keep none.
        """
        return None

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
            Zero: a read on the host holds nothing beside the state there.
        """
        del space
        return 0

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the backend works in, in bytes.

        The workspace's
        (:meth:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace.memory_ledger`):
        what one step has resident at its peak, every buffer named even when it is
        empty.

        Returns
        -------
        dict
            Buffer name to bytes.
        """
        return self._workspace.memory_ledger()

    def close(self) -> None:
        """Release the buffers. Idempotent."""
        self._workspace.close()
        march = self._run.march
        if isinstance(march, HostLowStorageMarch):
            engine = march.engine
            assert isinstance(engine, HostLowStorage)
            engine.close()
        self._closed = True

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the backend has released its buffers, as a failure may."""
        return self._closed

    def _check_open(self) -> None:
        """Refuse a backend that has been closed."""
        if self._closed:
            msg = "this backend is closed; its buffers were released"
            raise BackendError(msg)


class _TrialProbe:
    """Single trials of a loaded backend, committing nothing (SEP 47 6.6).

    The :class:`solprop.backends.protocol.TimingProbe` the backend returns. A
    Runge-Kutta trial leaves the loaded state, the position and any history alone. The
    low-storage march has no separate trial: its step advances the discarded registers
    in place and leaves no trial field.
    """

    def __init__(
        self,
        backend: NumpyRefBackend,
        engine: object,
        z: float,
        index: int,
        *,
        low_storage: bool,
    ) -> None:
        """Hold the loaded backend and its engine; no trial is taken yet."""
        self._backend = backend
        self._engine = engine
        self._z = z
        self._index = index
        self._low_storage = low_storage
        self._length: float | None = None

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one trial step and keep its end state for :meth:`field`.

        - ``z``: where the trial starts, m; the loaded position by default.
        - ``dz``: its length, m, positive; the problem's ``dz_init`` by default, or the
          scheduled step at the loaded position for the low-storage march.
        """
        stepper = self._backend.request.stepper
        start = self._z if z is None else float(z)
        if self._low_storage:
            schedule = stepper.schedule
            assert schedule is not None  # the translation refuses the march without one
            index = min(self._index, schedule.n_steps - 1)
            length = float(schedule.steps[index]) if dz is None else float(dz)
            cast("HostLowStorage", self._engine).trial(start, length)
            return
        length = float(stepper.dz_init) if dz is None else float(dz)
        cast("NumpyRungeKutta", self._engine).trial(start, length)
        self._length = length

    def field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the last trial's end state on the host, as a kept step would be.

        The trial's state after the accepted-step boundary treatments (the band mask,
        and the split-step absorber where it runs). A trial is held beside the loaded
        state, so finishing it leaves the loaded state alone, and no monitor is bound,
        so the finish takes no split sums. Refused for the low-storage march, and with
        no trial since the last read-back.

        - Returns: an owning copy, internal units and the FFT convention, shape
          ``(n_modes, n_pol, n_omega)``.
        """
        if self._low_storage:
            msg = (
                "the low-storage march accepts its step in place and has no trial "
                "field to read back"
            )
            raise BackendError(msg)
        if self._length is None:
            raise BackendError("take a trial before reading its field")
        engine = cast("NumpyRungeKutta", self._engine)
        finished, _ = self._backend.workspace.finish(engine.trial_state, self._length)
        self._length = None
        return np.array(finished)


def _refuse_adams(problem: PropagationProblem, stepper: StepperSettings) -> None:
    """Refuse what the Adams march does not execute (SEP 13 sections 2 and 6.2).

    The interaction formulation's history is of source samples transported along a
    diagonal flow, so a coupled flow is refused by name rather than transported through
    inverse actions; the exponential formulation refuses the coupled-flow sections it
    does not step (:func:`solprop.backends.exponential.refuse_formulation`) and the
    carrier frames it does not deliver. The ``modified`` noise is
    admitted for both formulations (SEP 16 section 4.4): its draw is fixed for the run
    and added to each sample's field at that sample's own position, never to the
    propagating state, so conditional on the draw the source is a deterministic,
    nonautonomous function of position that the history interpolates. The split-step
    temporal collar is admitted: it changes
    the state only inside the collar, where its first-order splitting error already
    lies (``docs/dev/stepper.md`` section 6), and it removes the collar's stability
    limit, which binds the Adams methods long before Runge-Kutta's (SEP 13 phase 2).
    """
    integrator = stepper.integrator
    if not isinstance(integrator, Multistep):
        return
    if (
        problem.linear.coupled_flow is not None
        and integrator.formulation != "exponential"
    ):
        # SEP 16 Q3: only the exponential formulation steps a coupled flow; the
        # interaction one would transport its history through inverse actions.
        msg = (
            f"the {integrator.name} integrator does not step a problem with a "
            "coupled linear flow; use Multistep(formulation='exponential') or the "
            "Runge-Kutta march"
        )
        raise BackendError(msg)
    refuse_formulation(problem, stepper, "numpy_ref")
    # SEP 35: the scalar shift, on the guided routes.
    refuse_carrier_frame(problem, stepper, "numpy_ref", delivered=frozenset({"scalar"}))


def _refuse_unsupported(problem: PropagationProblem, stepper: StepperSettings) -> None:
    """Refuse, by name, every problem this backend does not execute."""
    # The route first, by its variant (SEP 47 section 6.2): the reference executes
    # every route of the union and the modal route's TDSE extension.
    match problem.route:
        case (
            ModalRoute()
            | GeneralRoute()
            | SpatialRoute(kind="radial" | "cartesian" | "planar" | "line")
        ):
            pass
        case route:
            msg = f"numpy_ref does not execute the {route_name(route)} route"
            raise BackendError(msg)
    _refuse_adams(problem, stepper)
    if (
        problem.flags.has_raman
        and problem.transform.n_pol * problem.shape[1] > 1
        and (
            not problem.responses.raman_terms
            or not all(term.tensor_capable for term in problem.responses.raman_terms)
        )
    ):
        raise BackendError(
            "vector Raman requires tensor-capable terms; found scalar_only"
        )
    _, n_pol, _ = problem.shape
    if problem.transform.n_pol * n_pol > MAX_N_POL:
        msg = (
            f"the transform resolves {problem.transform.n_pol} transverse components "
            f"and the state carries {n_pol}; the components belong to one of the two "
            "and not to both, so this problem would count them twice"
        )
        raise BackendError(msg)
