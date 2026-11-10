# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's step loop, with the kernels left abstract (SEP 47).

March, replay, landing, controller bookkeeping, history, the noise model and the seam's
``set_initial``/``propagate_to``/``observe``/``close`` do not depend on how a stage is
evaluated: they touch the state and ``HostDriver._trial_step`` and nothing
else. Written once here, they are the step control of every NumPy reference backend.

**The reference owns this loop** since SEP 47 phase 2: :class:`HostDriver` is a copy of
the inherited driver the native bridge still runs on
(:class:`solprop.backends.driver.StepDriver`), so that the reference shares stateless
numerical modules -- the controller cores, tableaux, coefficient evaluation and stage
tables -- with that bridge, and transitionally the session lifecycle and the host
engines (the exceptions :mod:`solprop.backends.host` records). The copies started
identical; their decisions on identical inputs stay bit-identical (the decision traces
of ``tests/backends``).
What a backend supplies is ``_trial_step``, ``_apply_mask`` and ``stats``, plus
``_finish`` when it has boundary treatments of its own; two optional hooks,
``_begin_plane`` and ``_on_accepted_step``, are where a backend scores whatever a plane
or an accepted step leaves behind. The march owns the periodic statistics schedule.
Its default storage hooks call ``stats`` and retain host rows.
"""

import dataclasses
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._helpers import PLAIN, Helpers
from solprop._interrupts import committing, poll
from solprop.backends import rk_core
from solprop.backends.node_chunk import NodeChunkPlan, plan_node_chunk
from solprop.backends.protocol import (
    Backend,
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
    TimingProbe,
)
from solprop.backends.rk_core import rk_policy
from solprop.backends.session import (
    ExecutionOptions,
    PreparedExecution,
    PreparedGroup,
    Start,
)
from solprop.exceptions import BackendError, BoundaryError, ConvergenceError
from solprop.grid import TimeGrid
from solprop.log import NOT_ESTIMATED, NullReporter, ProgressReporter, StopFlag
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.schema import MarchSnapshot
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.stepper.controller import ControlRecord, PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import LANDING_ULPS, FixedSchedule, StepTargets, landing
from solprop.stepper.tableau import extension_coefficients, extension_value
from solprop.translate.linop import delta_phi, unitary_phase
from solprop.translate.masks import noise_spectrum
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.problem import PropagationProblem, PropagatorMode
from solprop.translate.responses import prefactor_at
from solprop.translate.routes import transverse_of
from solprop.translate.run_request import RunRequest
from solprop.translate.stage import (
    StageScalars,
    anchor,
    combine,
    scalars,
    scalars_at,
    terms,
)
from solprop.translate.tdse import TDSEPurpose
from solprop.transverse_monitor import TransverseEdge, _EdgeReading

if TYPE_CHECKING:
    from solprop.backends.host.adams import HostAdamsMarch
    from solprop.backends.host.low_storage import HostLowStorageMarch

__all__ = [
    "INITIAL_STEP_STATS_CAPACITY",
    "HostDriver",
    "reached",
    "replay_tolerance",
]

_StateT = TypeVar("_StateT")


def reached(position: float, target: float) -> bool:
    """Return whether a march at ``position`` has arrived at ``target``.

    Parameters
    ----------
    position : float
        Current position, m.
    target : float
        Position being marched to, m.

    Returns
    -------
    bool
        True when no further step is due.

    Examples
    --------
    >>> from solprop.backends.host.driver import reached
    >>> reached(0.5, 1.0), reached(1.0, 1.0)
    (False, True)
    """
    tolerance = LANDING_ULPS * float(np.spacing(max(abs(position), abs(target))))
    return bool(position >= target - tolerance)


def replay_tolerance(schedule: FixedSchedule, position: float) -> float:
    """Return how far a replayed schedule's edge may sit from a target it lands on, m.

    A schedule's edges are the running sum of its steps, and a sum of ``n`` rounded
    steps can drift from the position they were chosen to reach by about ``n`` ulps:
    ``FixedSchedule(np.full(160, L / 160))`` ends 12 ulps short of ``L``. The slack
    is therefore :data:`~solprop.stepper.schedule.LANDING_ULPS` or the step count,
    whichever is larger, in ulps of the larger of the position and the schedule's
    length -- still far inside the ``1e-9`` of the length that
    :meth:`~solprop.translate.problem.PropagationProblem.validate` allows an edge to
    miss a save plane by, so no step is ever skipped.

    Parameters
    ----------
    schedule : solprop.stepper.schedule.FixedSchedule
        The schedule being replayed.
    position : float
        The position the edge is compared with, m.

    Returns
    -------
    float
        The slack, m.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.host.driver import replay_tolerance
    >>> from solprop.stepper.schedule import FixedSchedule
    >>> schedule = FixedSchedule(np.full(160, 1e-3 / 160))
    >>> bool(1e-3 - schedule.z_end <= replay_tolerance(schedule, 1e-3))
    True
    """
    ulps = max(LANDING_ULPS, schedule.n_steps)
    return ulps * float(np.spacing(max(schedule.total_length, abs(position))))


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

        Parameters
        ----------
        z : float
            Position the row was taken at, m.
        step : int
            Index of the accepted step it was taken after, into the plane's history.
        values : collections.abc.Mapping
            What the backend's ``stats`` returned, keyed by name, without a batch
            axis.

        Raises
        ------
        BackendError
            If a statistic's shape is not the one the first row of this plane set.
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

        Returns
        -------
        solprop.backends.protocol.StepStats
            Copies, because the buffer is about to be written over.
        """
        rows = self._n_rows
        drained = StepStats(
            z=self._z[:rows].copy(),
            step=self._step[:rows].copy(),
            values={name: value[:rows].copy() for name, value in self._values.items()},
        )
        self.reset()
        return drained

    def _allocate(self, values: Mapping[str, np.ndarray]) -> None:
        """Lay the block out from the first row's shapes.

        Parameters
        ----------
        values : collections.abc.Mapping
            The first row, whose shapes every later row of this plane must match.
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
    Runge-Kutta trial (:meth:`HostDriver._trial_step`) leaves the loaded state, the
    position and the history alone. The low-storage march has no separate trial: its
    step advances the discarded registers in place and leaves no trial field.

    Parameters
    ----------
    driver : HostDriver
        The driver, loaded from the probe's start.
    start : tuple
        The loaded position, m, and the replayed itinerary's index there.
    """

    def __init__(
        self,
        driver: "HostDriver[object]",  # noqa: UP037 - the driver class follows
        start: tuple[float, int],
    ) -> None:
        """Hold the loaded driver; no trial is taken yet."""
        self._driver = driver
        self._z, self._index = start
        self._trial: tuple[object, float] | None = None

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one trial step and keep its end state for :meth:`field`.

        Parameters
        ----------
        z : float or None, optional
            Where the trial starts, m; the loaded position by default.
        dz : float or None, optional
            Its length, m, positive; the problem's ``dz_init`` by default, or the
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
            march.engine.step(start, length)
            return
        length = float(driver._stepper.dz_init) if dz is None else float(dz)
        _, state = driver._trial_step(start, length)
        self._trial = (state, length)

    def field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the last trial's end state on the host, as a kept step would be.

        The trial's state after the accepted-step boundary treatments (the band mask,
        and the split-step absorber where it runs). A host trial is a new array beside
        the loaded state, so finishing it leaves the loaded state alone.

        Returns
        -------
        numpy.ndarray
            An owning copy, internal units and the FFT convention, shape
            ``(n_modes, n_pol, n_omega)``.

        Raises
        ------
        solprop.exceptions.BackendError
            With no trial since the last readback, or for the low-storage march, which
            has no trial state.
        """
        driver = self._driver
        if driver._low_storage is not None:
            msg = (
                "the low-storage march accepts its step in place and has no trial "
                "field to read back"
            )
            raise BackendError(msg)
        if self._trial is None:
            raise BackendError("take a trial before reading its field")
        state, length = self._trial
        field = np.array(driver._observed_field(driver._finish(state, length))[0])
        self._trial = None
        return field


class HostDriver[_StateT](ABC):  # noqa: UP049 - the protected state type is private
    """The interaction-picture step loop, with the stage evaluation left to a subclass.

    A subclass supplies ``_trial_step`` and ``_apply_mask`` and gets the whole of
    :class:`solprop.backends.protocol.Backend`'s marching behaviour: adaptive stepping
    with the controller of ``docs/dev/stepper.md`` section 3, replay of a contract's
    fixed schedule, landing exactly on save planes and hard kinks, the step history, and
    both noise models, ``input`` and ``modified``.

    **The controller's state is its own proposal**, never the clipped or stretched step
    that was executed, and what the history and a rejection see is the norm the proposal
    would have scored while what decides acceptance is the error the executed step
    committed. Both are section 3's, and both matter: the first costs a run hundreds of
    steps if it is got wrong, the second throws away short landing steps that were
    accurate.

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
    max_batch : int
        How many problems the backend steps at once.
    propagator_default : {'tabulate', 'recompute'}
        What this backend resolves the contract's ``"auto"`` propagator mode to. The
        default suits a backend that evaluates two complex exponentials per stage; one
        that takes both signs from a single ``exp`` and ``sincos`` overrides it.
    helpers : solprop._helpers.Helpers
        The implementation of the per-stage and per-step scalar evaluations -- the
        stage tables' Clenshaw rows and the Adams basis -- which the driver, march and
        engines pass down rather than choosing one (SEP 29 section 6.6): ``PLAIN``, the
        uncompiled forms, for the NumPy reference.
    observes_fresh_field : bool
        Whether ``_observed_field`` makes a new array at every call and keeps no
        reference to it, so that :meth:`observe` can say its field is owned
        (:attr:`solprop.backends.protocol.Observation.field_owned`). ``False`` by
        default: the array-backed state is observed as it is, frozen, and left
        behind by the next step.
    """

    name: str = "driver"
    max_batch: int = 1
    propagator_default: PropagatorMode = "tabulate"
    helpers: Helpers = PLAIN
    observes_fresh_field: ClassVar[bool] = False

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
        # executes it, and the step it has left in flight: from the step's start to
        # its commit the state, accepted in place, is neither end of the step.
        self._low_storage: HostLowStorageMarch | None = None
        self._in_flight: tuple[float, float] | None = None
        # The fixed arrays of the seam's device form, held for the whole run; the
        # weights that multiply them are a bundle of host floats per stage.
        self._terms = terms(problem)
        # Hoisted per step by `_hoist`; the entrance's until the first step asks.
        self._anchor = anchor(problem, 0.0, helpers=self.helpers)
        # A step's bundles taken together by `_stage_bundles`, by anchor and
        # position; `_hoist` drops them.
        self._prepared: dict[tuple[float, float], StageScalars] = {}
        self._grid = grid
        self._modes = modes
        self._on_axis: Float[np.ndarray, "n_rows n_modes"] | None = None
        # The on-axis plasma scans actually run (SEP 19), so that a test can see that
        # a rule with a node on the axis supplies that row instead of a fresh scan.
        self.axis_scans = 0
        self._axis_node: int | None = None
        self._axis_node_known = False
        self._profiles = profiles
        self._logger = logging.getLogger(type(self).__module__)
        self._state: _StateT | None = None
        self._noise: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None = None
        self._z = 0.0
        self._proposal = self._stepper.dz_init
        self._clean_columns: Float[np.ndarray, " n_columns"] | None = None
        self._schedule_index = 0
        self._closed = False
        # The controller reads one norm a fixed lag back, and the one before it
        # (`PIController.propose`): a run keeps only those, not one per step.
        self._accepted_norms: deque[float] = deque(
            maxlen=self._stepper.controller.lag + 2
        )
        # Dense output (SEP 14 phase 3): the last accepted step that crossed a plane,
        # as its anchor, length and extension coefficients, and the march's own
        # position and host state while an interpolated plane is observed.
        self._dense_step: tuple[float, float, list[np.ndarray]] | None = None
        self._dense_ahead: tuple[float, np.ndarray] | None = None
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
        # Field-free planes at which the spatial collar diagnostic was not evaluated
        # since the load (SEP 22, Q2), recorded in `metadata` and logged once.
        # Physical RK source dispatches, including rejected trials and excluding
        # recycled stages. SEP 30 records work rather than estimating it from steps.
        self._rk_source_evaluations = 0

    # --- what a subclass supplies -----------------------------------------------

    @abstractmethod
    def _trial_step(self, z: float, dz: float) -> tuple[float, _StateT]:
        """Take one interaction-picture Runge-Kutta step and score it.

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the trial step, m, positive.

        Returns
        -------
        tuple
            The scaled error norm and the state at the end of the step. The state is
            *raw*: the boundary treatments of :meth:`_finish` are applied by the march,
            and only to a step it keeps.
        """

    @abstractmethod
    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> _StateT:
        """Apply the hard band mask to a state that has not been through a step.

        :meth:`HostDriver.set_initial` masks the state it is given, as the scaffold
        does, and that is the only place the driver needs the mask on its own.

        Parameters
        ----------
        state : numpy.ndarray
            The state in the canonical layout.

        Returns
        -------
        numpy.ndarray
            The state, with everything outside the active band exactly zero.
        """

    def _finish(
        self,
        state: _StateT,
        dz: float,
    ) -> _StateT:
        """Apply the once-per-accepted-step boundary treatments to a kept step.

        The hard band mask, and the temporal absorber when the ``"split_step"`` absorber
        scheme is active, followed by the mask again: its product in time spreads
        content into every bin, so the band is imposed last. Called by the march and
        by the replay on the state they are about to adopt, never on one they are
        about to throw away: a rejected step would otherwise pay the absorber's
        transform pair for a result nobody reads. While a monitor of that scheme is
        attached, a backend's split also adds what it takes to the monitor's next
        reading (:attr:`_split_removed`, :meth:`_boundary_energies`), at every
        accepted step, and keeps the energies of the power it found
        (:attr:`_split_found`), which the occupancies are judged on.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the end of an accepted step, in the canonical layout.
        dz : float
            The length of that step, m, which the absorber's rate needs.

        Returns
        -------
        numpy.ndarray
            The state as the run carries it on.
        """
        del dz
        return self._apply_mask(
            cast(Complex[np.ndarray, "n_batch n_modes n_pol n_omega"], state)
        )

    def _reserve_source_calls(self, count: int) -> None:
        """Reserve a complete trial's external source calls when a service is bound.

        Parameters
        ----------
        count : int
            Number of logical source evaluations required by the trial.

        Raises
        ------
        NotImplementedError
            Always: the driver that runs an Adams march (the reference backend) keeps
            its own reservations; a group's members reserve theirs.
        """
        raise NotImplementedError

    def _release_source_calls(self) -> None:
        """Release the unused part of an external source-call reservation.

        Raises
        ------
        NotImplementedError
            Always, as :meth:`_reserve_source_calls`.
        """
        raise NotImplementedError

    def _adams_source(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        anchor: float,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> tuple[
        Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ]:
        """Evaluate the source once for the low-call Adams march (SEP 13, SEP 14).

        The one expensive evaluation of an Adams step: the nonlinear source at a
        lab-frame field, with the run's noise and in-stage absorber, returned in the
        interaction frame of the step's anchor. A backend that executes
        :class:`~solprop.stepper.multistep.Multistep` overrides this and
        :meth:`_adams_propagator`; the default refuses, naming the backend.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`, in the canonical layout.
        anchor, position : float
            The step's anchor and the evaluation position, m.
        purpose : str, optional
            Atomic-call accounting label; ignored when no atomic service is bound.

        Returns
        -------
        tuple
            ``U(anchor, position) N(lab, position)`` and ``U(position, anchor)``, with
            ``U(z_2, z_1) = exp(Phi(z_2) - Phi(z_1))``.

        Raises
        ------
        BackendError
            Always, on a backend that does not execute the Adams march.
        """
        del lab, anchor, position, purpose
        msg = f"{self.name} does not execute the low-call Adams integrator"
        raise BackendError(msg)

    def _adams_lab_source(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate the source once, untransported, for exponential Adams (SEP 16).

        The complete source assembly ``F`` -- the nonlinear response, the run's noise
        draw at `position`'s absolute phase and the in-stage absorbers -- at a
        lab-frame field, returned in the lab frame with no propagator applied: the
        sample the exponential formulation integrates against the linear flow. A
        backend that executes that formulation overrides this; the default refuses,
        naming the backend.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`, in the canonical layout.
        position : float
            The evaluation position, m.
        purpose : str, optional
            Atomic-call accounting label; ignored when no atomic service is bound.

        Returns
        -------
        numpy.ndarray
            ``F(lab, position)`` in the lab frame.

        Raises
        ------
        BackendError
            Always, on a backend that does not execute the exponential formulation.
        """
        del lab, position, purpose
        msg = (
            f"{self.name} does not execute the exponential Adams formulation; use "
            "Multistep(formulation='interaction') or numpy_ref"
        )
        raise BackendError(msg)

    def _adams_block(self, z: float, dz: float) -> object:
        """Return the frozen coupled block of a step from `z`, or ``None`` (SEP 16).

        The exponential formulation steps a coupled flow's constant-Schur and
        frozen-residual sections against the reference ``(Delta Phi / h) I + G_s``
        (SEP 16 section 4.5), with ``G_s`` the frozen block of the section the step is
        anchored in. ``None``, the default, is a problem with no coupled flow; a
        backend that holds one returns its
        :class:`~solprop.backends.multistep.CoupledBlock`.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m.

        Returns
        -------
        object
            The step's ``CoupledBlock``, or ``None``.
        """
        del z, dz
        return None

    def _adams_propagator(
        self, anchor: float, position: float
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the linear propagator of an Adams step, ``U(position, anchor)``.

        Parameters
        ----------
        anchor, position : float
            The step's anchor and its end, m.

        Returns
        -------
        numpy.ndarray
            ``exp(Phi(position) - Phi(anchor))`` in the canonical layout, an array the
            caller owns.
        """
        bundle = self._stage_scalars(anchor, position)
        return np.exp(self._delta_phi(bundle))

    def _adams_state(self) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the accepted state as the Adams march reads it, a host array.

        The default is the protected state itself; a backend whose state lives
        elsewhere returns a host copy it keeps current through :meth:`_adams_adopt`.

        Returns
        -------
        numpy.ndarray
            The accepted state in the canonical layout.
        """
        return cast(
            Complex[np.ndarray, "n_batch n_modes n_pol n_omega"], self._require_state()
        )

    def _adams_adopt(
        self, advanced: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"], dz: float
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Adopt an accepted Adams step: its boundary treatments, then the state.

        Parameters
        ----------
        advanced : numpy.ndarray
            The step's end state before the boundary treatments, a host array.
        dz : float
            The step, m, which the split-step absorber needs.

        Returns
        -------
        numpy.ndarray
            The state the run carries on, as :meth:`_adams_state` will return it.
        """
        finished = self._finish(cast(_StateT, advanced), dz)
        self._state = finished
        return cast(Complex[np.ndarray, "n_batch n_modes n_pol n_omega"], finished)

    def _adams_restore(
        self, z: float, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> None:
        """Restore an accepted Adams state after interpolation or a failed correction.

        Parameters
        ----------
        z : float
            The position of the accepted state, m.
        state : numpy.ndarray
            Its accepted state there, as :meth:`_adams_state` returned it.
        """
        self._z = z
        self._state = cast(_StateT, state)

    @property
    def _norm_controller(self) -> PIController:
        """Return the controller whose tolerances scale this backend's error norms.

        The problem's own; a backend whose state is held at a scale ``s`` divides its
        absolute tolerance by ``s`` (SEP 33 section 4.3).

        Returns
        -------
        solprop.stepper.controller.PIController
            The controller.
        """
        return self._stepper.controller

    def _observed_field(
        self, state: _StateT
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return an owning host view of a backend's protected state.

        Array-backed CPU implementations use the identity. A device backend overrides
        this hook to synchronize and copy without teaching the shared march its state
        representation, and sets :attr:`observes_fresh_field` when every call makes a
        new array.

        Parameters
        ----------
        state : object
            The backend's protected state representation.

        Returns
        -------
        numpy.ndarray
            An owning host ``complex128`` field in canonical layout.
        """
        return cast(Complex[np.ndarray, "n_batch n_modes n_pol n_omega"], state)

    @abstractmethod
    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Compute the requested statistics in one fused pass.

        Declared here, not only on
        :class:`solprop.backends.protocol.Backend`, because the shared step loop calls
        it through the default row-storage hook. A device backend shares its statistic
        evaluator between this method and its device row-storage hook.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name, without a batch axis.
        """

    def _on_accepted_step(self) -> None:  # noqa: B027 - optional by design
        """Score whatever an accepted step leaves behind, before the next is proposed.

        The quadrature alarm is scored here, from buffers the last stage left behind.
        The default does nothing.
        """

    def _reset_step_stats(self, *, release: bool) -> None:
        """Discard periodic rows at a load or at the start of a plane.

        Parameters
        ----------
        release : bool
            True for a new load, which releases the old storage; false for a new
            plane, which retains capacity. The accepted-step period is owned by the
            march and is unaffected by this hook.
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

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Periodic subset promoted to ``names``, with no periodic subset itself.
        z : float
            Position of the committed state, m.
        step : int
            Attempt index in the history since the previous observation, including
            rejected attempts. It is supplied by the march, not inferred from rows.
        """
        self._rows.append(z, step, self.stats(spec))

    def _drain_step_stats(self) -> StepStats:
        """Return owning host rows at observation and retain reusable capacity.

        Returns
        -------
        solprop.backends.protocol.StepStats
            Rows since the start of this plane. A device override copies the packed
            values once here. Repeated drains return no rows until more are recorded.
        """
        return self._rows.drain()

    @property
    def _propagator_mode(self) -> PropagatorMode:
        """Return the propagator mode this run actually uses.

        Returns
        -------
        str
            ``"tabulate"`` or ``"recompute"``, with the contract's ``"auto"`` resolved
            to :attr:`propagator_default`.
        """
        asked = self._stepper.propagator_mode
        if asked == "auto" and self._low_storage is not None:
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
            another input (:func:`solprop.driver.simulation.check_derived_problem`).
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
        # abstract here, so that a batch's protocol check can refuse a driver without.
        return HostPrepared(cast("Backend", self), problem, request)

    def prepare_group(
        self,
        members: Sequence[Backend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
    ) -> PreparedGroup:
        """Prepare a group of compatible members of this backend (SEP 47 section 6.7).

        The reference's own group
        (:class:`~solprop.backends.host.group_session.HostGroupSession`): its members
        advance in lockstep through one fixed schedule on the reference's group runner
        (:class:`~solprop.backends.host.group.HostGroup` or
        :class:`~solprop.backends.numpy_ref.batch.NumpyBatchBackend`, chosen by
        :func:`solprop.backends.registry.select_batch`).

        Parameters
        ----------
        members : sequence of solprop.backends.protocol.Backend
            One constructed backend per member, this one first.
        problems : sequence of solprop.translate.problem.PropagationProblem
            Each member's independently translated problem.
        requests : sequence of solprop.translate.run_request.RunRequest
            Each member's run request.

        Returns
        -------
        solprop.backends.session.PreparedGroup
            The prepared group.
        """
        from solprop.backends.host.group_session import (  # noqa: PLC0415 - it builds on this module
            HostPreparedGroup,
        )

        return HostPreparedGroup(members, problems, requests)

    @property
    def request(self) -> RunRequest:
        """Return the request this backend was built for.

        Returns
        -------
        solprop.translate.run_request.RunRequest
            Its method, error control, itinerary, planes and statistics; a run adapts
            it with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        return self._request

    @property
    def node_chunk(self) -> int:
        """Return the nodes a response pass of this backend holds at once.

        Returns
        -------
        int
            The planned chunk (:attr:`chunk_plan`).
        """
        return self._chunk_plan.chunk

    @property
    def chunk_plan(self) -> NodeChunkPlan:
        """Return the node-chunk plan and what chose it.

        Returns
        -------
        solprop.backends.node_chunk.NodeChunkPlan
            The plan made at construction.
        """
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
            record["source_evaluations"] = self._rk_source_evaluations
        if self._monitor is not None:
            record["boundary_monitor"] = self._monitor.summary()
        return record

    def march_snapshot(self) -> MarchSnapshot | None:
        """Return the Adams march's state at the accepted position, for a resume.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
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
        """Return why this backend's Adams march cannot carry its history, or ``None``.

        Returns
        -------
        str or None
            The reason
            (:meth:`solprop.backends.host.adams.HostAdamsMarch.export_refusal`), or
            ``None`` for a march that exports and installs its history and for the
            Runge-Kutta and low-storage marches, which keep none.
        """
        return None if self._adams is None else self._adams.export_refusal()

    def restore_march(self, snapshot: MarchSnapshot) -> str | None:
        """Continue the Adams march from a snapshot at the loaded position.

        Called after :meth:`set_initial` at the snapshot's position. The state is
        replaced by the snapshot's own, exactly as the interrupted run held it.

        Parameters
        ----------
        snapshot : solprop.output.schema.MarchSnapshot
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
        return self._adams.load(snapshot)

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
            # The march imports its driver, so the ledger type is loaded only when
            # a constructed backend restores a saved run.
            from solprop.backends.multistep import CallLedger  # noqa: PLC0415

            self._adams.ledger = CallLedger.from_record(logical)

    # --- the per-stage bundle ------------------------------------------------------

    def _hoist(self, z0: float) -> None:
        """Evaluate once per step what every stage of it subtracts.

        ``Phi`` is cumulative, so each stage's increment is against the same anchor.
        On the RDW gradient the Clenshaw pass over the scalar tables is the largest
        part of what the host does per step, and this is a sixth of it.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        """
        self._anchor = anchor(self._problem, z0, helpers=self.helpers)
        self._prepared.clear()

    def _stage_bundles(
        self, z0: float, positions: Sequence[float]
    ) -> tuple[StageScalars, ...]:
        """Return the host scalars of several stages of one step, taken together.

        A trial knows all its stage positions before its first stage, and reading each
        scalar table once over all of them (:func:`solprop.translate.stage.scalars_at`)
        costs a Python call and a Clenshaw dispatch per table rather than per table and
        position. The bundles equal :meth:`_stage_scalars`'s bit for bit, and are kept
        until the next :meth:`_hoist`: a later :meth:`_stage_scalars` at the same anchor
        and position returns the kept bundle, so a step loop written one stage at a
        time takes them without change. Bundles of several anchors are kept side by
        side (kernel schema 29): the low-storage march's hooks read each of its three
        positions as its own anchor. A caller passes only positions the trial will
        evaluate.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        positions : sequence of float
            The stages' positions, m.

        Returns
        -------
        tuple of solprop.translate.stage.StageScalars
            One bundle per position, in order.
        """
        prepared = self._prepared
        missing = [z1 for z1 in dict.fromkeys(positions) if (z0, z1) not in prepared]
        if missing:
            held = self._anchor if self._anchor.z == z0 else None
            for bundle in scalars_at(
                self._problem, z0, missing, at=held, helpers=self.helpers
            ):
                prepared[z0, bundle.z1] = bundle
        return tuple(prepared[z0, z1] for z1 in positions)

    def _stage_scalars(self, z0: float, z1: float) -> StageScalars:
        """Return the host scalars of one stage, from the anchor to the node.

        The anchor's own weights come from :meth:`_hoist` when the caller has already
        taken them for this step, and are evaluated here otherwise. A bundle
        :meth:`_stage_bundles` took for this anchor and position is returned as kept.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        z1 : float
            The stage's position, m.

        Returns
        -------
        solprop.translate.stage.StageScalars
            The weights and the scalar path values.
        """
        kept = self._prepared.get((z0, z1))
        if kept is not None:
            return kept
        held = self._anchor if self._anchor.z == z0 else None
        return scalars(self._problem, z0, z1, at=held, helpers=self.helpers)

    def _delta_phi(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the stage's propagator exponent from the bundle.

        One linear combination of the terms this backend holds over the bundle's
        floats. A problem that took the z table has no such form and falls back to
        ``delta_phi``; a compiled backend that holds the table itself overrides this
        with its own evaluation.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray
            ``Phi(z1) - Phi(z0)``, dimensionless.
        """
        if bundle.phi_delta is None or self._terms.phi is None:
            return delta_phi(
                self._problem.linear, bundle.z0, bundle.z1, helpers=self.helpers
            )
        return combine(bundle.phi_delta, self._terms.phi)

    def _prefactor(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the response prefactor at the stage's position, from the bundle.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray
            The prefactor times the crop scale, in m/(V s) per unit polarisation.
        """
        if bundle.prefactor is None or self._terms.prefactor is None:
            return prefactor_at(
                self._problem.responses, bundle.z1, helpers=self.helpers
            )
        return combine(bundle.prefactor, self._terms.prefactor)

    def _unitary_phase(
        self, bundle: StageScalars
    ) -> Float[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return ``Im Phi`` at the stage's position, from the same terms.

        An expansion of ``Phi`` is cumulative and vanishes at the start of the range,
        so its value at a position is the bundle's ``phi_at`` against the same arrays
        and the noise costs no second representation.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray
            The phase in radians.
        """
        if bundle.phi_at is None or self._terms.phi is None:
            return unitary_phase(self._problem.linear, bundle.z1)
        return np.imag(combine(bundle.phi_at, self._terms.phi))

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
        after a run -- a :class:`~solprop.driver.simulation.SimulationRunner`'s, or
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
        # The march holds its state with a leading axis of one, privately: the layout
        # its kernels and the native build's read (SEP 47 section 6.2).
        values = values[np.newaxis]
        position = self._checked_position(float(z))
        drawn = self._noise_field(noise)
        model = self._problem.initial.noise_model
        self._clean_columns = None
        if (
            model == "input"
            and drawn is not None
            and not resume
            and position == self._problem.initial.z
        ):
            if transverse_of(self._problem.route) is not None:
                self._clean_columns = _clean_launch_columns(
                    values[0], drawn[0], self._problem.grids.active_mask
                )
            values = values + drawn
        self._state = self._apply_mask(values)
        self._noise = drawn if model == "modified" else None
        # The input draw rides in the state, a resumed one's too; the boundary monitor
        # reads it as the vacuum floor beneath the light it judges (SEP 21, Q24).
        self._input_floor = drawn if model == "input" else None
        self._z = position
        self._proposal = self._stepper.dz_init
        self._schedule_index = self._schedule_index_at(position)
        self._accepted_norms.clear()
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
        # The rest of what a march reads, as a fresh backend holds it (SEP 22 4.7).
        self._dense_step = None
        self._dense_ahead = None
        if self._adams is not None:
            self._adams.begin_run()
        self._in_flight = None
        self._rk_source_evaluations = 0
        if self._low_storage is not None:
            self._low_storage.reset()
            # Its count is the loaded run's, as the Adams ledger is (SEP 22 4.7).
            self._low_storage.source_evaluations = 0
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
        """Raise when the run's stop flag is set; called before every trial or call.

        Raises
        ------
        solprop.exceptions.StopRequested
            If the flag is set.
        """
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
        schedule = self._stepper.schedule
        if self._low_storage is not None:
            # The low-storage march replays only (SEP 24); the translation refused
            # it without a schedule.
            assert schedule is not None
            self._low_storage.replay_to(schedule, target)
        # An Adams march replays a schedule itself, with its own history and ledger.
        elif schedule is not None and self._adams is None:
            self._replay_to(schedule, target)
        else:
            self._march_to(target)
        if (
            self._natural_due
            and reached(self._z, target)
            and self._dense_ahead is None
            and (self._adams is None or self._adams._ahead is None)
        ):
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
            # The run's integrator, not this driver's march: a lockstep runner holds
            # none of its own while its members each hold one.
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
        history = BackendHistory(
            z=np.array([entry[0] for entry in self._history], dtype=np.float64),
            dz=np.array([entry[1] for entry in self._history], dtype=np.float64),
            error_norm=np.array(
                [entry[2] for entry in self._history], dtype=np.float64
            ),
            accepted=np.array([entry[3] for entry in self._history], dtype=np.bool_),
            wall_time=np.array([entry[4] for entry in self._history], dtype=np.float64),
        )
        whole = self._observed_field(state) if field else None
        if whole is not None and not self.observes_fresh_field:
            # The observation is a view of the held state without its leading axis:
            # the state itself is frozen, as an observed array always was, so that
            # the next step replaces it rather than accepting in place.
            whole.setflags(write=False)
        plane = Observation(
            z=self._z,
            field=None if whole is None else whole[0],
            noise_phase=self._noise_phase() if field else None,
            step_history=history,
            n_steps=history.n_steps,
            n_rejected=history.n_rejected,
            step_stats=self._drain_step_stats(),
            field_owned=field and self.observes_fresh_field,
        )
        self._history = []
        self._clock = time.perf_counter()
        return plane

    @property
    def in_flight(self) -> tuple[float, float] | None:
        """Return the low-storage step a failure abandoned in flight, or ``None``.

        Returns
        -------
        tuple of float or None
            The step's start and length, m, while the state is neither of its ends
            (``solprop.backends.low_storage``).
        """
        return self._in_flight

    @property
    def position(self) -> float:
        """Return where the march stands, m.

        Returns
        -------
        float
            The committed state's position, or the end of a step a group took but
            neither validated nor recorded, which leaves the state in flight
            (:attr:`in_flight`).
        """
        return float(self._z)

    @property
    def halted(self) -> bool:
        """Return whether the last march stopped at a requested halt short of its end.

        Returns
        -------
        bool
            True while the halt latch of :meth:`request_halt` or the yield check of
            :meth:`set_halt_check` stopped it.
        """
        return self._halted

    @property
    def holding_plane(self) -> bool:
        """Return whether a dense observation holds the state between endpoints.

        Returns
        -------
        bool
            True while this driver's own (a Runge-Kutta march's) or its Adams march's
            observation holds an interpolated plane, the march ahead of it.
        """
        return self._dense_ahead is not None or (
            self._adams is not None and self._adams._ahead is not None
        )

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
        history = BackendHistory(
            z=np.array([entry[0] for entry in entries], dtype=np.float64),
            dz=np.array([entry[1] for entry in entries], dtype=np.float64),
            error_norm=np.array([entry[2] for entry in entries], dtype=np.float64),
            accepted=np.array([entry[3] for entry in entries], dtype=np.bool_),
            wall_time=np.array([entry[4] for entry in entries], dtype=np.float64),
        )
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

    def _begin_plane(self) -> None:  # noqa: B027 - optional by design
        """Reset whatever describes only the steps of the plane about to be taken.

        The default does nothing; a backend that warns once per plane clears its latch
        here.
        """

    def _march_to(self, target: float) -> None:
        """Step adaptively to a position, landing on it exactly.

        Parameters
        ----------
        target : float
            Position to reach, m.
        """
        settings = self._stepper
        policy = self._controller_policy()
        dense = settings.interpolate_planes
        if dense and self._dense_ahead is not None:
            # The march is ahead of the last interpolated plane; resume it there.
            self._adams_restore(*self._dense_ahead)
            self._dense_ahead = None
        if self._natural_stop_due(target):
            return
        breakpoints = self._problem.linear.phi_breakpoints
        if not settings.interval_cap:
            # The cumulative table makes the flow exact across its intervals, so only
            # the table's two ends bound a step (SEP 14 section 4.1).
            breakpoints = breakpoints[[0, -1]]
        targets = StepTargets(
            # Dense output steps across the save planes; the end of the run and the
            # hard kinks are still landed on (SEP 14 section 4.3).
            save_planes=np.empty(0)
            if dense
            else np.asarray(settings.save_planes, dtype=np.float64),
            hard_kinks=self._problem.linear.hard_kinks,
            breakpoints=breakpoints,
            dz_max=settings.dz_max,
            z_end=settings.z_end if dense else target,
            stretch=settings.stretch,
        )
        while not reached(self._z, target):
            self._poll_stop()
            # The controller's state is its own proposal, never the clipped or
            # stretched step that was executed: otherwise every save plane restarts it
            # from a sliver and it regrows at the growth cap, which over two hundred
            # planes costs more steps than the run needs between them (stepper.md 3).
            stop = landing(
                self._z,
                self._split_capped(
                    self._transverse_capped(self._quiet_capped(self._proposal))
                ),
                targets,
            )
            norm, advanced = self._trial_step(self._z, stop.dz)
            # What the controller learns from, and what a rejection is scaled by, is the
            # norm the *proposal* would have scored; what decides acceptance, and what
            # the history records, is the error the executed step actually committed,
            # so a short landing step that was accurate is neither thrown away nor
            # written up as a failure.
            equivalent, accepted = rk_core.decide(policy, norm, stop.dz, self._proposal)
            # A landing snaps to its target rather than accumulating z + dz, so that a
            # save plane is hit exactly however many steps have gone before.
            end = stop.z_target if stop.z_target is not None else self._z + stop.dz
            self._record(end, stop.dz, norm, accepted)
            if accepted:
                if dense and not reached(target, end):
                    # The step crosses the plane: hold its extension for the
                    # observation, and for any later plane it also crosses.
                    self._dense_step = self._dense_record(self._z, stop.dz)
                finished = self._finish(advanced, stop.dz)
                with committing():
                    self._state = finished
                    self._z = end
                self._accepted_norms.append(equivalent)
                if policy.auto_init and len(self._accepted_norms) == 1:
                    self._start_up(equivalent)
                # The next proposal grows or shrinks once per accepted step, from the
                # history the step completed. A rejected step is retried at the step
                # rejection_factor left: applying this factor again there undid the
                # shrink, and with a growth cap above about 1.27 a step whose norm sat
                # just above one was rejected unchanged forever.
                self._proposal = rk_core.propose(
                    policy, self._proposal, self._accepted_norms
                )
                self._take_step_stats()
                if self._natural_stop(target):
                    return
            else:
                self._proposal = rk_core.reject(policy, self._proposal, equivalent)

        if dense and not reached(target, self._z):
            self._observe_inside(target)

    def _dense_record(
        self, z: float, dz: float
    ) -> tuple[float, float, list[np.ndarray]]:
        """Contract an accepted step into its continuous extension's coefficients.

        Called before the step's state is adopted. With ``v0`` the state at the step's
        anchor, ``k_i`` its stage derivatives in the anchor's frame and ``v1 = v0 + h
        sum_i b_i k_i``, the coefficients are DOPRI5's (Hairer, Noersett and Wanner,
        II.6): ``v0``, ``v1 - v0``, ``h k1 - (v1 - v0)``, ``(v1 - v0) - h k7 -
        (h k1 - (v1 - v0))`` and ``h sum_i d_i k_i``.

        Parameters
        ----------
        z : float
            The step's anchor, m.
        dz : float
            The step, m.

        Returns
        -------
        tuple
            The anchor, the step and the five coefficients, host arrays.
        """
        start = self._observed_field(cast(_StateT, self._state)).reshape(
            (1, *self._problem.shape)
        )
        coefficients = extension_coefficients(
            self._stepper.tableau, start, self._dense_stages(), dz
        )
        return z, dz, coefficients

    def _observe_inside(self, position: float) -> None:
        """Make the state the interpolated field at `position` and hold the march's.

        Parameters
        ----------
        position : float
            The plane, m, inside the last accepted step.
        """
        assert self._dense_step is not None
        z, dz, coefficients = self._dense_step
        start = coefficients[0]
        value = extension_value(coefficients, (position - z) / dz)
        # A bin the linear flow attenuates or amplifies by more than an e-fold over the
        # step -- the spectral collar's, outside the controlled band -- has an
        # interaction-picture derivative growing exponentially across it, which no
        # polynomial in theta follows: interpolated there it returns orders of
        # magnitude more than the field. Its source is negligible beside its own loss,
        # so the anchor's linear flow is its honest value.
        stiff = np.abs(np.real(self._delta_phi(self._stage_scalars(z, z + dz)))) > 1.0
        value = np.where(stiff, start, value)
        self._dense_ahead = (
            self._z,
            self._observed_field(cast(_StateT, self._state)).reshape(
                (1, *self._problem.shape)
            ),
        )
        self._state = self._apply_mask(self._adams_propagator(z, position) * value)
        self._z = position

    def _dense_stages(self) -> list[np.ndarray]:
        """Return the last accepted step's stage derivatives, host arrays.

        In the step's anchor frame and the canonical layout, one per tableau stage. A
        backend that observes densely overrides this; the default refuses.

        Returns
        -------
        list of numpy.ndarray
            The stage derivatives.

        Raises
        ------
        BackendError
            Always, on a backend without dense output.
        """
        msg = f"{self.name} does not observe densely"
        raise BackendError(msg)

    def _controller_policy(self) -> rk_core.RungeKuttaPolicy:
        """Return the Runge-Kutta march's controller policy (SEP 45).

        Returns
        -------
        solprop.backends.rk_core.RungeKuttaPolicy
            The run's controller, floor, cap and start-up flag.
        """
        return rk_policy(self._stepper)

    def _start_up(self, norm: float) -> None:
        """Take the step after the first from the first step's own estimate.

        The first proposal is a guess; its scaled norm says how far it is from the
        tolerance. The next proposal is the deadbeat step that norm implies, at most
        :data:`~solprop.stepper.controller.START_GROWTH` times the first, and the
        history is told the norm the first step would have scored at that length, so
        the lagged controller does not apply the guess's margin a second time (SEP 14
        section 4.2). A resumed run starts afresh and takes the rule again.

        A norm small enough to meet the cap says only that the guess was far inside
        the tolerance; it is often round-off, which differs between backends. Rescaled
        into the history it would steer every later step by that round-off, so the
        history records zero instead -- no estimate -- and the backends take the same
        steps.

        Parameters
        ----------
        norm : float
            The first accepted step's norm at its proposed length.
        """
        self._proposal, self._accepted_norms[-1] = rk_core.start_up(
            self._controller_policy(), self._proposal, norm
        )

    def _replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay the contract's fixed schedule to a position.

        A replay takes its steps as given and never rejects, but the embedded estimate
        costs nothing extra and is computed and recorded every step, so that a schedule
        too coarse for its tolerance is reported rather than silent
        (``docs/dev/stepper.md`` section 8). The position is read off the schedule's
        edges rather than accumulated, so every save plane is hit exactly -- which
        :meth:`~solprop.translate.problem.PropagationProblem.validate` has already
        checked the schedule can do.

        Parameters
        ----------
        schedule : solprop.stepper.schedule.FixedSchedule
            The schedule to replay.
        target : float
            Position to reach, m.

        Raises
        ------
        BackendError
            If the schedule runs out before the target is reached.
        ConvergenceError
            If a step's error norm is not finite: the field has diverged, and a replay
            cannot reject the step, so it stops there rather than carry NaN on.
        """
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        while self._schedule_index < schedule.n_steps and self._z < target - slack:
            self._poll_stop()
            dz = float(schedule.steps[self._schedule_index])
            norm, advanced = self._trial_step(self._z, dz)
            if not math.isfinite(norm):
                msg = (
                    f"{self.name}: replayed step {self._schedule_index + 1} of "
                    f"{schedule.n_steps}, {dz:.4g} m from z = {self._z:.6g} m, has a "
                    f"non-finite error norm ({norm!r}): the field has diverged. A "
                    "replayed schedule takes its steps as given and cannot reject "
                    "one, so a step beyond an explicit stability limit -- the "
                    "in-stage temporal collar's or a coupled flow's advisory step -- "
                    "grows until it overflows; the translation refuses a declared step "
                    "past the collar's limit and warns on one past the flow's. Take "
                    "shorter steps, or let the controller choose them"
                )
                raise ConvergenceError(msg)
            end = float(edges[self._schedule_index + 1])
            finished = self._finish(advanced, dz)
            with committing():
                self._state = finished
                self._schedule_index += 1
                self._z = end
            self._record(self._z, dz, norm, accepted=True)
            self._accepted_norms.append(norm)
            self._take_step_stats()
            if self._natural_stop(target):
                return
        if self._z < target - slack:
            msg = (
                f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                f"cannot reach z = {target:.6g} m"
            )
            raise BackendError(msg)

    def _natural_stop(self, target: float) -> bool:
        """Count an accepted step; return whether to stop at its end to save it.

        With ``step_planes = k`` the end of every ``k``-th accepted step since the load
        is saved as a plane (SEP 14 phase 3): ``propagate_to`` returns there, short of
        its target, and the driver calls it again. A step that lands on the target is
        saved as the target; one that crosses a densely observed target is saved after
        it, when the next call finds the stop due (:meth:`_natural_stop_due`).

        Parameters
        ----------
        target : float
            The position the call is stepping to, m.

        Returns
        -------
        bool
            Whether the march stops here, short of `target`.
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

        Parameters
        ----------
        target : float
            The position the call is stepping to, m.

        Returns
        -------
        bool
            True when a halt is requested or the yield check asks for one, and the
            endpoint is short of `target`.
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

        Parameters
        ----------
        target : float
            The position the call is stepping to, m.

        Returns
        -------
        bool
            Whether to return at once, at the step's end.
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

    _input_floor: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None = None
    """The input-noise draw the launch carries, the monitor's floor; ``None``."""

    _split_removed: float = 0.0
    """What the ``"split_step"`` collar took at the accepted steps since the monitor's
    last reading, in its units (:attr:`solprop.monitor.BoundaryEnergies.removed`):
    a host backend's :meth:`_finish` adds to it while a monitor of that scheme is
    attached (:attr:`_split_monitor`), and :meth:`_boundary_energies` drains it."""

    _split_found: Float[np.ndarray, " 4"] | None = None
    """The energy and the guards' and collar's energies of the power the last accepted
    step's split found (:attr:`solprop.monitor.BoundaryEnergies.found`), which the
    monitor judges the occupancies on: set by a host backend's :meth:`_finish` with
    :attr:`_split_removed`, and ``None`` before the first such step."""

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
        # The noise draw is a host array already, read once.
        monitor.launch(
            self._boundary_energies(self._require_state()),
            self._z,
            floor=self._input_floor,
        )
        self._set_collar_quiet(monitor.collar_quiet)
        # A resumed fixed schedule is held to the state it resumes from at once.
        self._refuse_scheduled_crossing(monitor)
        edge = monitor.transverse
        if edge is not None:
            floor = self._input_floor
            schedule = self._stepper.schedule
            state = self._require_state()
            edge.launch(
                self._edge_reading(
                    state,
                    edge,
                    # A resumed launch reads no share, but continues the current.
                    full=not edge.resumed,
                    current=edge.every_step and edge.resumed,
                ),
                self._z,
                noise=None if floor is None else np.asarray(floor[0]),
                clean_columns=self._edge_clean_columns(edge),
                steps=None if schedule is None else np.asarray(schedule.steps),
                reads=self.reads_transverse_edge,
                declared=_declared_discard(self._problem),
            )

    def _read_transverse(self, monitor: BoundaryMonitor, *, plane: bool) -> None:
        """Read the transverse edge where its spacing or a saved plane asks (SEP 40).

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            The attached monitor.
        plane : bool
            Whether a saved plane has just been reached, which is always read.
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
        reading = self._edge_reading(
            self._require_state(), edge, faces=read, current=accumulate
        )
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

    def _edge_reading(
        self,
        state: _StateT,
        edge: TransverseEdge,
        *,
        faces: bool = True,
        current: bool = False,
        full: bool = False,
    ) -> _EdgeReading:
        """Return what the transverse edge asks of a state (SEP 44).

        The default reduces a host copy of the field, the reference
        (``_HostEdgeReduction``); a backend whose state
        lives elsewhere reduces it there and returns the numbers, without a copy of it.

        Parameters
        ----------
        state : object
            The backend's protected state representation.
        edge : solprop.transverse_monitor.TransverseEdge
            The run's edge.
        faces : bool, optional
            Whether the nodes, the columns and any collar's strip are wanted.
        current : bool, optional
            Whether a collar's inward current is wanted.
        full : bool, optional
            Whether every node is wanted, not only the rows a read uses.

        Returns
        -------
        solprop.transverse_monitor._EdgeReading
            The reading.
        """
        return edge.reduction.reading(
            np.asarray(self._observed_field(state)[0]),
            faces=faces,
            current=current,
            full=full,
        )

    def _edge_clean_columns(
        self, edge: TransverseEdge
    ) -> Float[np.ndarray, " n_columns"] | None:
        """Return each column's launched energy without the input noise, or ``None``.

        What the safety ramp is judged on, so a run with no ramp needs none. It is
        formed where the load has both parts, which needs no state on the host. Only a
        resume that found no record of the monitor has none, and then the state has to
        cross once.

        Parameters
        ----------
        edge : solprop.transverse_monitor.TransverseEdge
            The run's edge.

        Returns
        -------
        numpy.ndarray or None
            The unweighted energies, or ``None`` with no noise, no ramp or on a resumed
            launch.
        """
        floor = self._input_floor
        if floor is None or edge.resumed or edge.design is None:
            return None
        if self._clean_columns is not None:
            return self._clean_columns
        state = np.asarray(self._observed_field(self._require_state())[0])
        return edge.reduction.column_power(state - np.asarray(floor[0]))

    def _next_step(self) -> float:
        """Return the step the march takes next, m.

        A replay takes its schedule's and never updates the proposal, which stays
        ``dz_init``: read from that, an edge read comes early or late by the difference.

        Returns
        -------
        float
            The schedule's next step on a replay, zero past its end, and otherwise the
            controller's proposal.
        """
        schedule = self._stepper.schedule
        if schedule is None:
            return self._proposal
        if self._schedule_index >= schedule.n_steps:
            return 0.0
        return float(schedule.steps[self._schedule_index])

    def _transverse_capped(self, proposal: float) -> float:
        """Return the proposal, held to the transverse edge's read spacing (SEP 40 5.3).

        A low-source run commits long steps, and light could cross the edge unseen
        within one; the read spacing is a ceiling on the step instead. The step is not
        made to land on a read: the edge is read before a step would pass one.

        Parameters
        ----------
        proposal : float
            The controller's proposal, m.

        Returns
        -------
        float
            The proposal, or the read spacing, m.
        """
        monitor = self._monitor
        edge = None if monitor is None else monitor.transverse
        if edge is None:
            return proposal
        return min(proposal, edge.ceiling(self._z))

    def _split_capped(self, proposal: float) -> float:
        """Return the proposal, held to the split collar's crossing limit.

        The boundary monitor reads the limit from the state's occupancy at every
        check (:attr:`solprop.monitor.BoundaryMonitor.step_cap`): light that spreads
        into faster-walking bins shortens the steps until the next check, which reads
        it again.

        Parameters
        ----------
        proposal : float
            The controller's proposal, m.

        Returns
        -------
        float
            The proposal, or the cap, m.
        """
        monitor = self._monitor
        return proposal if monitor is None else min(proposal, monitor.step_cap)

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Prepare to read a boundary monitor's energies, or stop reading them.

        Called when a monitor is attached, with nothing yet taken by a split, and
        with ``None`` when a state is loaded. The default keeps its reading on the
        host; a backend whose state lives elsewhere overrides this to hold the
        monitor's weights (:attr:`solprop.monitor.BoundaryMonitor.weights`, and the
        columns' quadrature) where it reduces, and to add each accepted step's split
        removal there from now on (:meth:`_boundary_energies`).

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor or None
            The run's monitor, or ``None`` to stop.
        """
        del monitor
        self._split_removed = 0.0
        self._split_found = None

    @property
    def _split_monitor(self) -> BoundaryMonitor | None:
        """Return the monitor a host backend's finish adds the split's removal for.

        Returns
        -------
        solprop.monitor.BoundaryMonitor or None
            The attached monitor under the ``"split_step"`` scheme, else ``None``.
        """
        monitor = self._monitor
        return monitor if monitor is not None and monitor.split else None

    def _boundary_energies(self, state: _StateT) -> BoundaryEnergies:
        """Return the boundary monitor's reading of a state (SEP 21, ADR-057).

        The state's energy and its four weighted energies
        (:class:`solprop.monitor.BoundaryEnergies`), what the split collar took at
        the accepted steps since the previous reading, which this drains, and the
        energies of the power the last step's split found. The default
        reduces on the host (:meth:`solprop.monitor.BoundaryMonitor.energies`), the
        reference; a backend whose state lives elsewhere reduces it there and returns
        the five numbers, without a copy of the field.

        Parameters
        ----------
        state : object
            The backend's protected state representation.

        Returns
        -------
        solprop.monitor.BoundaryEnergies
            The reading.

        Raises
        ------
        BackendError
            If no monitor is attached.
        """
        monitor = self._monitor
        if monitor is None:
            raise BackendError(f"{self.name}: no boundary monitor is attached")
        removed, self._split_removed = self._split_removed, 0.0
        field = self._observed_field(state)
        crossing = monitor.crossing
        return dataclasses.replace(
            monitor.energies(field),
            removed=removed,
            found=self._split_found,
            occupancy=None
            if crossing is None
            else crossing.occupancy(field, monitor.columns),
        )

    def _check_boundaries(
        self, monitor: BoundaryMonitor, *, plane: bool = False
    ) -> None:
        """Account the committed step at the current position with the monitor.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            The attached monitor.
        plane : bool, optional
            True for a saved plane's check of a step the monitor already skipped.
        """
        self._check_at(
            monitor,
            self._boundary_energies(self._require_state()),
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

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            The attached monitor.
        energies : solprop.monitor.BoundaryEnergies
            The reading of the committed state.
        z : float
            The step's end, m.
        cursor : int
            The replay cursor after the step.
        plane : bool, optional
            True for a saved plane's check of a step the monitor already skipped.
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

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            The attached monitor, just read.
        cursor : int or None, optional
            The replay cursor after the step read; ``None`` for the driver's own.
        z : float or None, optional
            The step's end, m; ``None`` for the driver's position.

        Raises
        ------
        BoundaryError
            Naming the cap and the step.
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

        Parameters
        ----------
        quiet : bool
            The monitor's verdict.
        """
        quiet = quiet and self._collar_can_rest()
        if self._monitor is not None:
            self._monitor.note_collar(quiet)
        if quiet != self._collar_quiet:
            self._collar_quiet = quiet
            self._collar_changed()

    def _quiet_capped(self, proposal: float) -> float:
        """Return the proposal, held to the monitor's interval while the collar rests.

        A left-out collar absorbs nothing, and the monitor can only see the committed
        states: one long step could carry light from a quiet collar deep into it,
        unabsorbed. While the term is out no step is longer than the interval the
        monitor checks a quiet window at (:attr:`BoundaryMonitor.interval`, the
        guide's length over :data:`solprop.monitor.CHECKS`), so light has to walk
        about twenty windows over the run to cross the collar between two checks.

        Parameters
        ----------
        proposal : float
            The controller's proposal, m.

        Returns
        -------
        float
            The proposal, or the monitor's interval while the collar is left out, m.
        """
        monitor = self._monitor
        if not self._collar_quiet or monitor is None:
            return proposal
        return min(proposal, monitor.interval)

    def _collar_can_rest(self) -> bool:
        """Return whether this run's stages may leave a quiet temporal collar out.

        One rule for every backend: a binary64 modal or radial stage with the
        in-stage (``"nonlinear"``) scheme and a temporal collar alone. The
        low-storage march, coupled flows, the general modal route, a real-space
        collar sharing the term, single precision, and the Cartesian, line and
        planar routes, whose stages keep the term whatever the monitor says, keep
        the collar in every stage (SEP 21 D15).

        Returns
        -------
        bool
            True when the term may be left out while the collar is quiet.
        """
        problem = self._problem
        responses = problem.responses
        geometry = problem.transform.geometry
        resting_geometry = geometry is None or (
            geometry.kind == "radial" and not np.any(geometry.absorber_rate)
        )
        return (
            self._low_storage is None
            and responses.temporal_absorber is not None
            and responses.absorber_scheme == "nonlinear"
            and problem.linear.coupled_flow is None
            and problem.transform.modal_tables is None
            and not problem.precision.single
            and resting_geometry
        )

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

        Called from the two step loops **after** ``_finish`` and after ``self._z`` is
        advanced, which is the only point at which both describe the step just
        committed: ``_march_to`` records the attempt before it finishes the state and
        ``_replay_to`` after, so anything reading ``self._state`` or ``self._z`` from
        inside ``_on_accepted_step`` would be one accepted step stale on an adaptive run
        and correct on a replayed one.

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

        Parameters
        ----------
        z : float
            Position at the end of the step, m.
        dz : float
            Length of the step, m.
        norm : float
            The scaled error norm the attempt scored, as executed: a landing step
            clipped short records its own norm, not the proposal's equivalent. A
            diverged attempt, whose norm overflowed or was NaN, is recorded as
            infinity; the controller never accepts one.
        accepted : bool
            Whether the step was kept.
        """
        if not math.isfinite(norm):
            norm = math.inf
        self._history.append((z, dz, norm, accepted, time.perf_counter() - self._clock))
        self._progress.step(z, dz, accepted)
        if accepted:
            self._on_accepted_step()

    # --- noise ---------------------------------------------------------------------

    def _noise_field(
        self, noise: NoiseRealisation | None
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None:
        """Return the noise realisation to carry, in the canonical layout.

        Parameters
        ----------
        noise : solprop.noise.NoiseRealisation or None
            The realisation passed to :meth:`set_initial`.

        Returns
        -------
        numpy.ndarray or None
            The noise spectrum, or ``None`` when no noise model is active.
        """
        if self._problem.initial.noise_model == "none":
            return None
        if noise is not None:
            return noise_spectrum(noise, self._problem.shape)[np.newaxis]
        noise0 = self._problem.initial.noise0
        return None if noise0 is None else noise0[np.newaxis]

    def _noise_at(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None:
        """Return the shot-noise field advanced to a stage's position.

        ``N(z) = N0 exp(i Im Phi(omega, z))``: dispersion yes, loss and gain no, from
        the same terms as the linear flow, so the noise costs one multiply-add per
        element per stage and no transform.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray or None
            The advanced noise, or ``None`` when the modified model is not in use.
        """
        if self._noise is None:
            return None
        return self._noise * np.exp(1j * self._unitary_phase(bundle))

    def _noise_phase(
        self,
    ) -> Float[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return ``Im Phi`` at the current position, or ``None`` without a noise model.

        Returns
        -------
        numpy.ndarray or None
            The phase in radians, in the canonical layout.
        """
        if self._problem.initial.noise_model == "none":
            return None
        phase = self._unitary_phase(self._stage_scalars(self._z, self._z))
        return np.ascontiguousarray(
            np.broadcast_to(phase, (1, *self._problem.shape))[0]
        )

    # --- guards --------------------------------------------------------------------

    def _axis_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
        scan: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        """Return the four on-axis plasma statistics of one batch member (SEP 19).

        A rule with a node on the axis hands over that node's row from this
        observation's node scan; otherwise the axis row is scanned once
        (:func:`solprop.backends.populations.axis_populations`).

        Parameters
        ----------
        field : numpy.ndarray or None
            One batch member at this plane, shape (n_modes, n_pol, n_omega), or ``None``
            from a backend whose :meth:`_axis_row_populations` reads its own state.
        scan : tuple or None
            This observation's channel fractions and peak rates at every node, when
            the node statistics already scanned them.

        Returns
        -------
        dict
            The four ``on_axis_*`` statistics.

        Raises
        ------
        BackendError
            If the backend was built without its mode set.
        """
        if self._modes is None:
            msg = (
                "the on-axis plasma statistics need the mode set, which the kernel "
                "contract does not carry; construct the backend with "
                f"select({self.name!r}, problem, request, modes=modes)"
            )
            raise BackendError(msg)
        from solprop.backends.populations import (  # noqa: PLC0415 - a cycle
            axis_node,
            axis_statistics,
        )

        if not self._axis_node_known:
            self._axis_node_known = True
            self._axis_node = axis_node(
                self._problem,
                mode_matrix(
                    self._modes,
                    on_axis_rule(self._modes[0].dimlimits(0.0)[0]),
                    0.0,
                    components=("x", "y"),
                ).synthesis,
            )
        if scan is not None and self._axis_node is not None:
            fractions, peaks = scan
            node = self._axis_node
            return axis_statistics(
                self._problem, self._z, fractions[:, node], peaks[:, node]
            )
        self.axis_scans += 1
        fractions, peaks = self._axis_row_populations(
            field, self._on_axis_synthesis(self._modes)
        )
        return axis_statistics(self._problem, self._z, fractions, peaks)

    def _axis_row_populations(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
        axis: Float[np.ndarray, "n_rows n_modes"],
    ) -> tuple[Float[np.ndarray, " n_channels"], Float[np.ndarray, " n_channels"]]:
        """Scan this plane's axis row: one row, every channel.

        A device backend overrides this hook to read only the row from its state.

        Parameters
        ----------
        field : numpy.ndarray or None
            One batch member at this plane, shape (n_modes, n_pol, n_omega).
        axis : numpy.ndarray
            The mode set's synthesis on the axis at this position
            (:meth:`_on_axis_synthesis`).

        Returns
        -------
        tuple of numpy.ndarray
            The final ionised fraction and the largest rate on the axis, per channel.

        Raises
        ------
        BackendError
            If this backend needs the field and was not given it.
        """
        if field is None:
            raise BackendError("the on-axis statistics scan needs the field")
        from solprop.backends.populations import (  # noqa: PLC0415 - a cycle
            axis_populations,
        )

        return axis_populations(self._problem, field, self._z, axis)

    def _on_axis_synthesis(
        self, modes: Sequence[Mode]
    ) -> Float[np.ndarray, "n_rows n_modes"]:
        """Return the mode set's synthesis matrix on the axis, at the current position.

        Built once at ``z = 0`` and rescaled by the transform's own taper factor: the
        axis is the single node ``r = 0`` of
        :func:`solprop.modes.quadrature.on_axis_rule`, so the law that rescales the
        transform's matrices -- a scale-invariant profile goes as ``1/a`` and its
        normalisation as ``a^2`` -- applies to it exactly, and a set that is neither
        constant nor scale invariant never reaches a backend (``translate.modal``).
        Rebuilding the matrix at every call was what kept ``peak_intensity`` out of the
        between-planes channel.

        Parameters
        ----------
        modes : collections.abc.Sequence
            The mode set the backend was built with.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_modes)``, one row per Cartesian component on the axis.
        """
        if self._on_axis is None:
            rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
            matrices = mode_matrix(modes, rule, 0.0, components=("x", "y"))
            self._on_axis = np.ascontiguousarray(matrices.synthesis)
        taper = self._problem.transform.taper_scale
        if taper is None:
            return self._on_axis
        return self._on_axis * float(taper(self._z))

    def _require_grid(self) -> TimeGrid:
        """Return the grid the state lives on, or say why statistics cannot be taken.

        Returns
        -------
        solprop.grid.TimeGrid
            The grid given at construction.

        Raises
        ------
        BackendError
            If none was given.
        """
        if self._grid is None:
            msg = (
                "statistics need the grid the state lives on, which the kernel "
                f"contract does not carry; construct the backend with select("
                f"{self.name!r}, problem, request, grid=grid)"
            )
            raise BackendError(msg)
        return self._grid

    def _require_state(self) -> _StateT:
        """Return the current state, refusing a closed or unloaded backend.

        Returns
        -------
        numpy.ndarray
            The state.

        Raises
        ------
        BackendError
            If the backend is closed, no initial state was loaded, or a low-storage
            step was abandoned in flight.
        """
        self._check_open()
        if self._state is None:
            msg = "no initial state: call set_initial before propagating or observing"
            raise BackendError(msg)
        self._refuse_in_flight()
        return self._state

    def _refuse_in_flight(self) -> None:
        """Refuse every entry that reads or advances a state left between two steps.

        The low-storage march accepts each step in place (SEP 24 section 6.2): from
        the step's start to its commit the state is neither the step's start nor its
        end, so an exception or a signal between them leaves nothing a reader may use
        and nothing a march may continue from. Only :meth:`set_initial` clears it.

        Raises
        ------
        BackendError
            Naming the step that was abandoned.
        """
        if self._in_flight is None:
            return
        z, dz = self._in_flight
        msg = (
            f"{self.name}: the low-storage step of {dz:.6g} m from z = {z:.6g} m was "
            "abandoned in flight, and the state it was accepting in place holds "
            "neither end of the step; load a state with set_initial (resume from the "
            "last saved plane) before observing or propagating again"
        )
        raise BackendError(msg)

    @property
    def closed(self) -> bool:
        """Return whether the backend has released its buffers, as a failure may.

        Returns
        -------
        bool
            ``True`` when it can run no other input.
        """
        return self._closed

    def _check_open(self) -> None:
        """Refuse a backend that has been closed.

        Raises
        ------
        BackendError
            If ``close`` has been called.
        """
        if self._closed:
            msg = "this backend is closed; its buffers were released"
            raise BackendError(msg)

    def _checked_position(self, z: float) -> float:
        """Return a position the state may be loaded at, refusing one off the table.

        Parameters
        ----------
        z : float
            The requested position, m.

        Returns
        -------
        float
            The same position.

        Raises
        ------
        BackendError
            If it is not finite or lies outside the operator table.
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

        Parameters
        ----------
        z : float
            The position the state is loaded at, m.

        Returns
        -------
        int
            The index of the schedule's edge at `z`, and 0 without a schedule.

        Raises
        ------
        BackendError
            If a schedule is being replayed and `z` is not one of its edges. Every save
            plane is an edge, which
            :meth:`~solprop.translate.problem.PropagationProblem.validate` has checked,
            so this fires only for a position the run never stopped at -- from which a
            replay could not continue in step.
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

        Parameters
        ----------
        target : float
            The requested position, m.

        Raises
        ------
        BackendError
            Saying why the position cannot be reached.
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


def _clean_launch_columns(
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    noise: Complex[np.ndarray, "n_modes n_pol n_omega"],
    active: np.ndarray,
) -> Float[np.ndarray, " n_modes"]:
    """Return each column's energy in a masked launch, less the input noise it carries.

    That is ``mask(field + noise) - noise``: the launch on the active band, and the
    noise alone off it, as a supplied realisation needn't be masked. Formed from the
    parts, so that no state has to come back from where it lives to subtract the noise.

    Parameters
    ----------
    field : numpy.ndarray
        The launch before the noise is added.
    noise : numpy.ndarray
        The realisation added to it.
    active : numpy.ndarray
        The active band's mask, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The unweighted energies, shape ``(n_modes,)``.
    """
    inside = np.abs(field[..., active]) ** 2
    outside = np.abs(noise[..., ~active]) ** 2
    return np.sum(inside, axis=(1, 2)) + np.sum(outside, axis=(1, 2))


def _declared_discard(problem: PropagationProblem) -> bool:
    """Return whether a run declared content above ``discard_above`` (SEP 40 5.3).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.

    Returns
    -------
    bool
        True when its transverse specification sets ``discard_above``.
    """
    design = transverse_of(problem.route)
    return design is not None and design.specification.discard_above is not None
