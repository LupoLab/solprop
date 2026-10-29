# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The step loop every backend runs, with the kernels left abstract.

March, replay, landing, controller bookkeeping, history, the noise model and the seam's
``set_initial``/``propagate_to``/``observe``/``close`` do not depend on how a stage is
evaluated: they touch the state and ``StepDriver._trial_step`` and nothing
else. Written once here, they are the same
step control on every backend, so a change to ``docs/dev/stepper.md`` section 3 is made
in one place and a disagreement between two backends can only be the kernels.

**Every backend runs this loop**, the reference included: the oracle is the passes and
the per-element arithmetic, not the march, and a march written twice has to be kept
identical by hand.
What a backend supplies is ``_trial_step``, ``_apply_mask`` and ``stats``, plus
``_finish`` when it has boundary treatments of its own; two optional hooks,
``_begin_plane`` and ``_on_accepted_step``, are where a backend scores whatever a plane
or an accepted step leaves behind. The march owns the periodic statistics schedule.
Its default storage hooks call ``stats`` and retain host rows; a device backend can
override the three storage hooks to retain device rows until observation.
"""

import dataclasses
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._helpers import NUMBA, Helpers
from solprop._interrupts import committing
from solprop.backends.protocol import (
    Observation,
    StepHistory,
    StepStats,
    StepSummary,
)
from solprop.backends.spatial import collar_field_fraction
from solprop.exceptions import BackendError, ConvergenceError
from solprop.grid import TimeGrid
from solprop.log import NOT_ESTIMATED, NullReporter, ProgressReporter
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.schema import MarchSnapshot
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.stepper.controller import (
    START_GROWTH,
    ControlRecord,
    PIController,
    limit_step,
)
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import LANDING_ULPS, FixedSchedule, StepTargets, landing
from solprop.stepper.tableau import extension_coefficients, extension_value
from solprop.translate.linop import delta_phi, unitary_phase
from solprop.translate.masks import noise_spectrum
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.problem import KernelProblem, PropagatorMode
from solprop.translate.responses import prefactor_at
from solprop.translate.stage import (
    StageScalars,
    anchor,
    combine,
    scalars,
    scalars_at,
    terms,
)
from solprop.translate.tdse import TDSEPurpose

if TYPE_CHECKING:
    from solprop.backends.low_storage import LowStorageMarch
    from solprop.backends.multistep import AdamsMarch

__all__ = [
    "INITIAL_STEP_STATS_CAPACITY",
    "StepDriver",
    "reached",
    "refuse_general_leaves",
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
    >>> from solprop.backends.driver import reached
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
    :meth:`~solprop.translate.problem.KernelProblem.validate` allows an edge to miss a
    save plane by, so no step is ever skipped.

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
    >>> from solprop.backends.driver import replay_tolerance
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
            What the backend's ``stats`` returned, keyed by name, each with a leading
            batch axis.

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


class StepDriver[_StateT](ABC):  # noqa: UP049 - the protected state type is private
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
    problem : solprop.translate.problem.KernelProblem
        The validated problem to step. The subclass refuses what it cannot execute
        before calling this constructor.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, which the statistics need and the contract does not
        carry.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, for the statistics that need them.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.

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
        stage tables' Clenshaw rows and the Adams basis -- which the shared driver,
        march and engines pass down rather than choosing one (SEP 29 section 6.6).
        Each backend sets its own: ``PLAIN`` for the NumPy reference, ``NUMBA`` (the
        default) for ``numba_cpu``, the C++ bindings for the native backend.
    """

    name: str = "driver"
    max_batch: int = 1
    propagator_default: PropagatorMode = "tabulate"
    helpers: Helpers = NUMBA

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
    ) -> None:
        """Adopt a validated problem and start the run at the entrance."""
        self._problem = problem
        self._adams: AdamsMarch | None = None
        # The low-storage march (SEP 24), set at construction by a backend that
        # executes it, and the step it has left in flight: from the step's start to
        # its commit the state, accepted in place, is neither end of the step.
        self._low_storage: LowStorageMarch | None = None
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
        self._proposal = problem.stepper.dz_init
        self._schedule_index = 0
        self._closed = False
        self._accepted_norms: list[float] = []
        # Dense output (SEP 14 phase 3): the last accepted step that crossed a plane,
        # as its anchor, length and extension coefficients, and the march's own
        # position and host state while an interpolated plane is observed.
        self._dense_step: tuple[float, float, list[np.ndarray]] | None = None
        self._dense_ahead: tuple[float, np.ndarray] | None = None
        # Natural-step saving (SEP 14 phase 3): accepted steps since the load, and
        # whether a step that crossed a densely observed plane ends on a step to save.
        self._natural_count = 0
        self._natural_due = False
        self._history: list[tuple[float, float, float, bool, float]] = []
        self._clock = time.perf_counter()
        self._progress: ProgressReporter = NullReporter()
        self._progress_started = False
        # The statistics channel between planes. The spec a row is evaluated with is
        # the periodic subset promoted to `names`, so the row goes through the very
        # same `stats` the save planes use and is bitwise what a plane would record.
        spec = problem.stats_spec
        self._row_spec: StatsSpec | None = (
            None
            if not spec.periodic
            else dataclasses.replace(spec, names=spec.periodic, periodic=(), period=1)
        )
        self._row_period = spec.period
        self._accepted_steps = 0
        self._rows = _StepStatsBuffer()
        # A low-storage march's state at its last field-free plane (SEP 22, Q14): one
        # volume, allocated at the first such plane and listed by the in-tree
        # ledgers, so a step abandoned in flight still leaves a resumable state.
        self._held: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None = None
        self._held_z: float | None = None
        # Field-free planes at which the spatial collar diagnostic was not evaluated
        # since the load (SEP 22, Q2), recorded in `metadata` and logged once.
        self._collar_skipped = 0
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

        :meth:`StepDriver.set_initial` masks the state it is given, as the scaffold
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
        accepted step.

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
        """
        del count

    def _release_source_calls(self) -> None:
        """Release the unused part of an external source-call reservation."""
        return

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

    def _adams_engine(
        self,
        method: Multistep,
        compensation: Literal["transported", "untransported"],
    ) -> object:
        """Return an engine that keeps the Adams history where this backend's state is.

        SEP 16 phase 3b: :class:`solprop.backends.multistep.AdamsMarch` keeps the
        controller and hands every state-sized operation of its fused and exponential
        paths to an engine. ``None``, the default, takes the host engine
        (:class:`solprop.backends.multistep.HostHistory`) over this backend's other
        Adams hooks; Kokkos returns one resident in its native state.

        Parameters
        ----------
        method : solprop.stepper.multistep.Multistep
            The method.
        compensation : {'transported', 'untransported'}
            A varying operator's compensation treatment.

        Returns
        -------
        object
            The backend's own :class:`~solprop.backends.multistep.AdamsEngine`, or
            ``None`` for the host one.
        """
        del method, compensation
        return None

    def _low_storage_engine(self, method: LowStorageRungeKutta) -> object:
        """Return an engine that keeps the low-storage registers where the state is.

        SEP 24 section 6.4: :class:`solprop.backends.low_storage.LowStorageMarch`
        replays the schedule and commits each step; an engine
        (:class:`~solprop.backends.low_storage.LowStorageEngine`) holds the registers
        and makes the step's four passes. ``None``, the default, takes the host engine
        (:class:`~solprop.backends.low_storage.HostLowStorage`) over this backend's
        :meth:`_adams_lab_source` and :meth:`_adams_propagator`.

        Parameters
        ----------
        method : solprop.stepper.low_storage.LowStorageRungeKutta
            The march.

        Returns
        -------
        object
            The backend's own engine, or ``None`` for the host one.
        """
        del method
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
        return self._problem.stepper.controller

    def _observed_field(
        self, state: _StateT
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return an owning host view of a backend's protected state.

        Array-backed CPU implementations use the identity. A device backend overrides
        this hook to synchronize and copy without teaching the shared march its state
        representation.

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
            Small host arrays keyed by statistic name, each with a leading batch axis.
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
        asked = self._problem.stepper.propagator_mode
        if asked == "auto" and self._low_storage is not None:
            # One volume of exp(h D / 2) reused while the step holds is the cheaper
            # flow on a CPU; a varying operator changes it every step (SEP 24
            # section 4.3).
            return "tabulate" if self._problem.flags.constant_linop else "recompute"
        if asked == "auto" and self._problem.linear.separable is not None:
            return "recompute"
        return self.propagator_default if asked == "auto" else asked

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
        }
        if self._problem.responses.plasma_channels:
            record["plasma_source"] = plasma_source_record(self._problem.grids)
        if self._adams is not None:
            record["response_calls"] = dataclasses.asdict(self._adams.ledger)
        if self._low_storage is not None:
            record["source_evaluations"] = self._low_storage.source_evaluations
        elif self._adams is None:
            record["source_evaluations"] = self._rk_source_evaluations
        if self._collar_skipped:
            record["collar_diagnostic_skipped"] = self._collar_skipped
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
            (:meth:`solprop.backends.multistep.AdamsMarch.snapshot`).

        Raises
        ------
        BackendError
            If a low-storage step was abandoned in flight.
        """
        self._refuse_in_flight()
        if self._low_storage is not None:
            return self._low_storage.snapshot()
        return None if self._adams is None else self._adams.snapshot()

    def march_export_refusal(self) -> str | None:
        """Return why this backend's Adams march cannot carry its history, or ``None``.

        Returns
        -------
        str or None
            The reason (:meth:`solprop.backends.multistep.AdamsMarch.export_refusal`),
            or ``None`` for a march that exports and installs its history and for the
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
        the next source evaluation charges startup against these cumulative counts.

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
        if logical:
            if self._adams is None:
                raise BackendError("logical response calls require an Adams run")
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
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
    ) -> None:
        """Load a state and the noise realisation at a position.

        The state is masked to the active band once here, as the scaffold does: the
        operator and the prefactor are exactly zero outside it, so out-of-band content
        would neither propagate nor decay and would sit in the error norm for the whole
        run. The traditional ``input`` noise model adds its realisation to the
        state at the run's launch position and never again; the ``modified`` model keeps
        it apart and adds it at every stage, so the propagating field never carries it.

        A `z` past the launch is a **resume**: the state came back from a save plane the
        run already reached, so it already carries whatever the ``input`` model added at
        the launch and must not be given a second realisation, and a replayed schedule
        continues from the edge that plane sits on. The launch is the entrance unless
        the run starts inside the guide (SEP 37: an interior launch with ``input`` noise
        takes its draw there). The controller starts afresh from
        ``dz_init`` either way (``docs/dev/stepper.md`` section 3).

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
            ``(n_batch, n_modes, n_pol, n_omega)``, in the square root of watts and the
            FFT convention.
        noise : solprop.noise.NoiseRealisation or None
            The realisation to carry along the run, or ``None`` to take the one the
            problem was assembled with.
        z : float, optional
            The position the state belongs to, m. Zero, the entrance, by default.

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
        drawn = self._noise_field(noise)
        model = self._problem.initial.noise_model
        if (
            model == "input"
            and drawn is not None
            and position == self._problem.initial.z
        ):
            values = values + drawn
        self._state = self._apply_mask(values)
        self._noise = drawn if model == "modified" else None
        # The input draw rides in the state; the boundary monitor reads it as the
        # vacuum floor beneath the light it judges (SEP 21, Q24).
        self._input_floor = drawn if model == "input" else None
        self._z = position
        self._proposal = self._problem.stepper.dz_init
        self._schedule_index = self._schedule_index_at(position)
        self._accepted_norms = []
        self._history = []
        self._natural_count = 0
        self._natural_due = False
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
        self._held_z = None
        self._collar_skipped = 0
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
            settings = self._problem.stepper
            self._progress.started(self._z, settings.z_end, len(settings.save_planes))
        started = time.perf_counter()
        first = len(self._history)
        # Beside `_begin_plane` and not inside it: neither backend's override calls
        # `super()`, so a base-class body there would silently do nothing.
        self._reset_step_stats(release=False)
        self._begin_plane()
        schedule = self._problem.stepper.schedule
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
        taken = self._history[first:]
        record = ControlRecord.from_norms(
            self._problem.stepper.controller,
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
            if isinstance(self._problem.stepper.integrator, LowStorageRungeKutta)
            else f"{record.tolerance_ratio:.3e}",
        )
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
        the backend counts such planes in :meth:`metadata` and logs the first. Under
        the low-storage march the state is held, one volume, so that a step abandoned
        in flight can still be checkpointed at this plane
        (:meth:`held_observation`).

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
        if self._low_storage is not None:
            self._hold(None if field else state)
        history = StepHistory(
            z=np.array([entry[0] for entry in self._history], dtype=np.float64),
            dz=np.array([entry[1] for entry in self._history], dtype=np.float64),
            error_norm=np.array(
                [entry[2] for entry in self._history], dtype=np.float64
            ),
            accepted=np.array([entry[3] for entry in self._history], dtype=np.bool_),
            wall_time=np.array([entry[4] for entry in self._history], dtype=np.float64),
        )
        plane = Observation(
            z=self._z,
            field=self._observed_field(state) if field else None,
            noise_phase=self._noise_phase() if field else None,
            step_history=history,
            n_steps=history.n_steps,
            n_rejected=history.n_rejected,
            step_stats=self._drain_step_stats(),
        )
        geometry = self._problem.transform.geometry
        if geometry is not None and plane.field is None:
            self._skip_collar_diagnostic()
        elif geometry is not None and plane.field is not None:
            # This is a saved-plane diagnostic, paid only after observation has
            # already requested host fields. It never causes a device transfer
            # in the periodic statistics lane or changes the accepted state.
            operations = 64 * self._problem.shape[1] + 16 * math.ceil(
                math.log2(self._problem.grids.n_t)
            )
            rounding = operations * np.finfo(float).eps
            threshold = max(
                self._problem.stepper.controller.rtol, rounding / (1 - rounding)
            )
            for member in plane.field:
                fraction = collar_field_fraction(self._problem, member)
                if fraction > threshold:
                    self._logger.warning(
                        "spatial collar contains %.6e of the beam's field norm at "
                        "z = %.6e m, above diagnostic threshold %.6e; increase the "
                        "transverse "
                        "window to check the influence of its boundary",
                        fraction,
                        self._z,
                        threshold,
                    )
        self._history = []
        self._clock = time.perf_counter()
        return plane

    def held_observation(self) -> Observation | None:
        """Return the state the low-storage march held at its last field-free plane.

        SEP 22, Q14: a low-storage step accepts in place, so a step abandoned in
        flight leaves a state that is neither of its ends. When the last plane kept
        no field, this is what the driver checkpoints instead: the state as it was
        observed there, with an empty step history (the plane's own was saved with
        it) and its noise phase.

        Returns
        -------
        Observation or None
            The held plane, or ``None`` when nothing is held: another march, a plane
            that kept its field since, or a fresh load.

        Raises
        ------
        BackendError
            If the backend is closed.
        """
        self._check_open()
        if self._held_z is None:
            return None
        z = self._held_z
        phase = None
        if self._problem.initial.noise_model != "none":
            phase = np.ascontiguousarray(
                np.broadcast_to(
                    self._unitary_phase(self._stage_scalars(z, z)), self._problem.shape
                )
            )
        return Observation(
            z=z,
            field=self._held_field(),
            noise_phase=phase,
            step_history=StepHistory(
                np.empty(0), np.empty(0), np.empty(0), np.empty(0, bool), np.empty(0)
            ),
            n_steps=0,
            n_rejected=0,
        )

    def _hold(self, state: _StateT | None) -> None:
        """Hold a low-storage march's state at a field-free plane, or release the hold.

        The buffer is allocated at the first field-free plane and kept for reuse: a
        plane that keeps its field only marks the hold released, and the ledger and the
        admission go on charging the one volume.

        Parameters
        ----------
        state : object or None
            The state at a field-free plane, or ``None`` at a plane that keeps its
            field, which resumes by itself.
        """
        if state is None:
            self._held_z = None
            return
        values = self._observed_field(state)
        if self._held is None:
            self._held = np.empty_like(values)
        np.copyto(self._held, values)
        self._held_z = self._z

    def _held_field(self) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return a host copy of the state :meth:`_hold` holds.

        Called by :meth:`held_observation` only while a plane is held. A backend that
        holds the plane where its own state lives (solprop-native, in its memory
        space) overrides this with its own one copy to the host, beside :meth:`_hold`.

        Returns
        -------
        numpy.ndarray
            The held state, an owned host array of the problem's shape.
        """
        assert self._held is not None  # the plane is held, so its buffer exists
        return self._held.copy()

    def _skip_collar_diagnostic(self) -> None:
        """Count a field-free plane whose collar diagnostic was not evaluated.

        Logged at the first such plane after a load, and recorded by
        :meth:`metadata` (SEP 22, Q2): the diagnostic transforms the field, which a
        field-free plane does not observe.
        """
        if self._collar_skipped == 0:
            self._logger.info(
                "the spatial collar diagnostic is not evaluated on planes that keep no "
                "field (first at z = %.6e m); keep the field on a plane to check the "
                "collar's share of the fluence there",
                self._z,
            )
        self._collar_skipped += 1

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
        settings = self._problem.stepper
        controller = settings.controller
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
            # The controller's state is its own proposal, never the clipped or
            # stretched step that was executed: otherwise every save plane restarts it
            # from a sliver and it regrows at the growth cap, which over two hundred
            # planes costs more steps than the run needs between them (stepper.md 3).
            stop = landing(self._z, self._quiet_capped(self._proposal), targets)
            norm, advanced = self._trial_step(self._z, stop.dz)
            # What the controller learns from, and what a rejection is scaled by, is the
            # norm the *proposal* would have scored; what decides acceptance, and what
            # the history records, is the error the executed step actually committed,
            # so a short landing step that was accurate is neither thrown away nor
            # written up as a failure.
            equivalent = controller.equivalent_norm(
                norm, dz_taken=stop.dz, dz_proposed=self._proposal
            )
            accepted = controller.accepts(norm)
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
                if settings.auto_init and len(self._accepted_norms) == 1:
                    self._start_up(equivalent)
                # The next proposal grows or shrinks once per accepted step, from the
                # history the step completed. A rejected step is retried at the step
                # rejection_factor left: applying this factor again there undid the
                # shrink, and with a growth cap above about 1.27 a step whose norm sat
                # just above one was rejected unchanged forever.
                self._proposal = limit_step(
                    self._proposal,
                    controller.propose(self._accepted_norms),
                    settings.dz_min,
                    settings.dz_max,
                )
                self._take_step_stats()
                if self._natural_stop(target):
                    return
            else:
                self._proposal = limit_step(
                    self._proposal,
                    controller.rejection_factor(equivalent),
                    settings.dz_min,
                    settings.dz_max,
                )

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
            self._problem.shape
        )
        coefficients = extension_coefficients(
            self._problem.stepper.tableau, start, self._dense_stages(), dz
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
                self._problem.shape
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
        settings = self._problem.stepper
        controller = settings.controller
        first = self._proposal
        factor = controller.start_factor(norm)
        self._proposal = limit_step(first, factor, settings.dz_min, settings.dz_max)
        self._accepted_norms[-1] = (
            0.0
            if factor >= START_GROWTH
            else controller.equivalent_norm(
                norm, dz_taken=first, dz_proposed=self._proposal
            )
        )

    def _replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay the contract's fixed schedule to a position.

        A replay takes its steps as given and never rejects, but the embedded estimate
        costs nothing extra and is computed and recorded every step, so that a schedule
        too coarse for its tolerance is reported rather than silent
        (``docs/dev/stepper.md`` section 8). The position is read off the schedule's
        edges rather than accumulated, so every save plane is hit exactly -- which
        :meth:`~solprop.translate.problem.KernelProblem.validate` has already checked
        the schedule can do.

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
        every = self._problem.stepper.step_planes
        self._natural_count += 1
        if every == 0 or self._natural_count % every:
            return False
        if reached(self._z, target):
            self._natural_due = not reached(target, self._z)
            return False
        return True

    def _natural_stop_due(self, target: float) -> bool:
        """Return whether a step to save ends before `target`, with the march there.

        Called at the start of a march, once a dense observation's march state is
        restored: a step that crossed the last observed plane and was the ``k``-th is
        saved now, unless `target` is another plane it also crossed.

        Parameters
        ----------
        target : float
            The position the call is stepping to, m.

        Returns
        -------
        bool
            Whether to return at once, at the step's end.
        """
        if self._natural_due and not reached(self._z, target):
            self._natural_due = False
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
        """
        self._monitor = monitor
        self._bind_monitor(monitor)
        # The noise draw is a host array already, read once.
        monitor.launch(
            self._boundary_energies(self._require_state()),
            self._z,
            floor=self._input_floor,
        )
        self._set_collar_quiet(monitor.collar_quiet)

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
        (:class:`solprop.monitor.BoundaryEnergies`), and what the split collar took at
        the accepted steps since the previous reading, which this drains. The default
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
        return dataclasses.replace(
            monitor.energies(self._observed_field(state)), removed=removed
        )

    def _check_boundaries(self, monitor: BoundaryMonitor) -> None:
        """Account the committed step at the current position with the monitor.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            The attached monitor.
        """
        monitor.step(self._boundary_energies(self._require_state()), self._z)
        self._set_collar_quiet(monitor.collar_quiet)

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

        One rule for every backend: a binary64 modal stage with the in-stage
        (``"nonlinear"``) scheme and a temporal collar alone. The low-storage march,
        coupled flows, the general modal route, a radial real-space collar sharing
        the term and single precision keep the collar in every stage (SEP 21 D15).

        Returns
        -------
        bool
            True when the term may be left out while the collar is quiet.
        """
        problem = self._problem
        responses = problem.responses
        geometry = problem.transform.geometry
        real_space = (
            geometry is not None
            and geometry.kind == "radial"
            and bool(np.any(geometry.absorber_rate))
        )
        return (
            self._low_storage is None
            and responses.temporal_absorber is not None
            and responses.absorber_scheme == "nonlinear"
            and problem.linear.coupled_flow is None
            and problem.transform.modal_tables is None
            and not problem.precision.single
            and not real_space
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
            self._check_boundaries(monitor)
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
        monitor = self._monitor
        if monitor is not None:
            # A quiet window is checked every few steps, without reading the state
            # in between (`solprop.monitor.CADENCE`).
            if monitor.due(self._z):
                self._check_boundaries(monitor)
            else:
                monitor.skip(self._z)
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
            return noise_spectrum(noise, self._problem.shape)
        return self._problem.initial.noise0

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
    ) -> Float[np.ndarray, "n_batch n_modes n_pol n_omega"] | None:
        """Return ``Im Phi`` at the current position, or ``None`` without a noise model.

        Returns
        -------
        numpy.ndarray or None
            The phase in radians, in the canonical layout.
        """
        if self._problem.initial.noise_model == "none":
            return None
        phase = self._unitary_phase(self._stage_scalars(self._z, self._z))
        return np.ascontiguousarray(np.broadcast_to(phase, self._problem.shape))

    # --- guards --------------------------------------------------------------------

    def _axis_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scan: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        """Return the four on-axis plasma statistics of one batch member (SEP 19).

        A rule with a node on the axis hands over that node's row from this
        observation's node scan; otherwise the axis row is scanned once
        (:func:`solprop.backends.populations.axis_populations`).

        Parameters
        ----------
        field : numpy.ndarray
            One batch member at this plane, shape (n_modes, n_pol, n_omega).
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
                f"select({self.name!r}, problem, modes=modes)"
            )
            raise BackendError(msg)
        from solprop.backends.populations import (  # noqa: PLC0415 - a cycle
            axis_node,
            axis_populations,
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
        fractions, peaks = axis_populations(
            self._problem, field, self._z, self._on_axis_synthesis(self._modes)
        )
        return axis_statistics(self._problem, self._z, fractions, peaks)

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
                f"{self.name!r}, problem, grid=grid)"
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
            :meth:`~solprop.translate.problem.KernelProblem.validate` has checked, so
            this fires only for a position the run never stopped at -- from which a
            replay could not continue in step.
        """
        schedule = self._problem.stepper.schedule
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


def refuse_general_leaves(
    problem: KernelProblem, backend: str, *, admits: Sequence[str] = ()
) -> None:
    """Refuse, by name, each schema-12 general-modal leaf a backend cannot execute.

    The four optional leaves of the general modal route arrived at kernel schema 12
    with no backend behind them. A
    schema check alone cannot stand in for this: a separately built backend compares
    against solprop's imported constant, so a freshly built wheel would accept a
    problem carrying a representation it does not implement and silently run the old
    physical model. Each leaf is therefore named here, one refusal each, and each
    check is deleted only when that backend executes that exact representation.

    Every backend in this repository names all four leaves and admits through the
    shared list in :mod:`solprop.backends.numpy_ref.admission`, so this refusal guards
    a separately built backend. A backend that executes a leaf names it in `admits` and
    checks the exact representation it supports itself: admitting a leaf is a statement
    about one combination of contents, not about the field being present.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.
    backend : str
        The backend's registry name, used in the refusal.
    admits : sequence of str, optional
        Dotted leaf names this backend executes, for example
        ``("transform.modal_tables",)``. Empty by default: a backend refuses every
        one until it has a kernel for it.

    Raises
    ------
    BackendError
        Naming the backend and the leaf it cannot execute.
    """
    for name, leaf in (
        ("linear.coupled_flow", problem.linear.coupled_flow),
        ("linear.modal_metric", problem.linear.modal_metric),
        ("transform.modal_tables", problem.transform.modal_tables),
        ("responses.regional_responses", problem.responses.regional_responses),
    ):
        if leaf is not None and name not in admits:
            msg = (
                f"{backend} does not execute {name}: the general modal route is not "
                "implemented on this backend, and a leaf silently ignored would run, "
                "converge and be the wrong physics"
            )
            raise BackendError(msg)
