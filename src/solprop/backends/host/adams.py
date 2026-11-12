# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's interaction-picture Adams march (SEP 13, SEP 14, SEP 47).

The controller, history, ledger and lifecycle of :class:`HostAdamsMarch`, the NumPy
reference's own since SEP 47 phase 2 (the shared march it was copied from is deleted
with the native backend's step driver, phase 5). The state-sized arithmetic is the
shared stateless code (:mod:`solprop.backends.host`): the march hands every operation
of its fused and exponential paths to the host engine
(:class:`solprop.backends.numpy_ref.multistep.HostHistory`). The decisions are
:mod:`solprop.backends.adams_core`'s functions of one policy record.

The method, its lifecycle and its numbers are those of
:mod:`solprop.backends.numpy_ref.multistep`'s module documentation.
"""

import math
from typing import Any, Literal, cast

import numpy as np

from solprop._interrupts import committing
from solprop.backends import adams_core
from solprop.backends.exponential import CarrierReference
from solprop.backends.host.driver import HostDriver, reached, replay_tolerance
from solprop.backends.ledger import CallLedger, Derivative
from solprop.backends.numpy_ref.multistep import (
    AdamsEngine,
    AdamsKernels,
    ExponentialKernels,
    HostHistory,
    _State,
)
from solprop.exceptions import BackendError, ConvergenceError, ResponseBudgetError
from solprop.output.records import MarchSnapshot
from solprop.stepper.controller import limit_step
from solprop.stepper.multistep import MarchPolicy, Multistep, march_policy
from solprop.stepper.schedule import FixedSchedule, landing
from solprop.translate.tdse import TDSEPurpose

__all__ = ["HostAdamsMarch"]


class HostAdamsMarch:
    """March a reference backend's state with an interaction-picture Adams method.

    Parameters
    ----------
    backend : solprop.backends.host.driver.HostDriver
        The backend whose state, source evaluation and propagators are used. It owns
        the step history and the statistics rows; this object owns the derivative
        history and the call ledger.
    method : solprop.stepper.multistep.Multistep
        The method.
    compensation : {'transported', 'untransported'}, optional
        How a varying operator's linear compensation is integrated (SEP 16 Q8):
        ``"transported"``, the delivered treatment, or ``"untransported"``, kept as a
        test witness only.
    engine : object, optional
        An :class:`~solprop.backends.numpy_ref.multistep.AdamsEngine`; ``None``, the
        default, takes the host engine
        (:class:`~solprop.backends.numpy_ref.multistep.HostHistory`) over the backend's
        hooks. A scripted engine drives the controller alone, for the decision traces
        of SEP 39 section 7.1. Typed ``object`` because the engine protocol is
        structural and not checkable at run time; the members the path calls are
        checked here instead.

    Raises
    ------
    solprop.exceptions.BackendError
        If `engine` lacks a member its path calls.
    """

    def __init__(
        self,
        backend: HostDriver[Any],
        method: Multistep,
        *,
        compensation: Literal["transported", "untransported"] = "transported",
        engine: object = None,
    ) -> None:
        """Hold the backend and start with no history."""
        self._backend = backend
        self._method = method
        # Whether the run interpolates its planes, and with what (ADR-036, ADR-039).
        self._interpolate = backend._stepper.interpolate_planes
        self._kernels = AdamsKernels(helpers=backend.helpers)
        # SEP 16's exponential formulation keeps untransported samples in the same
        # store and shares everything else: the lifecycle, the controller, the ledger.
        self._exponential: ExponentialKernels | None = None
        if method.formulation == "exponential":
            self._exponential = ExponentialKernels(helpers=backend.helpers)
        # Every state-sized operation goes to an engine (SEP 16 phase 3b), the host
        # one unless a scripted engine is given.
        if engine is not None:
            _check_engine(engine, exponential=self._exponential is not None)
            self._engine: AdamsEngine = cast("AdamsEngine", engine)
        else:
            self._engine = HostHistory(
                backend, method, self._kernels, self._exponential, compensation
            )
        self._history: list[Derivative] = []
        self._rejections = 0
        self._last_rejected = False
        self._order = 1 if method.variable_order else method.order
        self._ahead: tuple[float, object] | None = None
        self.ledger = CallLedger()
        # An in-stage temporal collar adds -alpha(t) / 2 to the derivative: in the
        # exponential formulation, whose linear flow is exact, it is the one stiff term
        # the explicit history integrates, and its largest rate bounds each order's
        # stable step, which the order choice weighs (`_choose_order`).
        responses = backend._problem.responses
        collar = responses.temporal_absorber
        collar_rate = (
            0.5 * float(np.max(collar))
            if collar is not None
            and responses.absorber_scheme == "nonlinear"
            and method.formulation == "exponential"
            else 0.0
        )
        # Every number the controller and the landings read, from one record (SEP 39
        # section 4.2); the decisions are `adams_core`'s functions of it.
        settings = backend._stepper
        linear = backend._problem.linear
        carrier = self._engine.carrier
        self._policy: MarchPolicy = march_policy(
            method,
            dz_min=settings.dz_min,
            dz_max=settings.dz_max,
            stretch=settings.stretch,
            z_end=settings.z_end,
            save_planes=np.asarray(settings.save_planes, dtype=np.float64).tolist(),
            hard_kinks=np.asarray(linear.hard_kinks, dtype=np.float64).tolist(),
            breakpoints=np.asarray(linear.phi_breakpoints, dtype=np.float64).tolist(),
            interpolate=self._interpolate,
            collar_rate=collar_rate,
            carrier=isinstance(carrier, CarrierReference) and carrier.active,
        )
        # Whether the stages left a quiet collar out at the last attempt: the step
        # after it returns is held to the collar's stable step (`_collar_reentry`).
        self._collar_was_quiet = False
        # A floor an accepted step's next proposal fell below, raised once the step is
        # recorded (`_decide`, `_raise_floor`).
        self._floor: ConvergenceError | None = None

    def reset(self) -> None:
        """Forget the history: the state was loaded or changed outside the march."""
        self._history = []
        self._rejections = 0
        self._engine.reset()
        self._ahead = None

    def begin_run(self) -> None:
        """Start a run's accounting afresh: the ledger and the last attempt's flags.

        Called by :meth:`~solprop.backends.host.driver.HostDriver.set_initial`, so that
        a backend loaded for another run counts that run's calls alone and chooses its
        first order and step as a fresh backend would (SEP 22 section 4.7, SEP 39
        R9): the last rejection and the collar's flag are cleared. A resume restores
        its persisted ledger and flags afterwards.
        """
        self.ledger = CallLedger()
        self._last_rejected = False
        self._collar_was_quiet = False
        self._floor = None

    def snapshot(self) -> MarchSnapshot | None:
        """Return the march's state at its accepted position, for a resume.

        Returns
        -------
        solprop.output.records.MarchSnapshot or None
            The snapshot, or ``None`` where there is nothing a resume could continue:
            no history yet, or the march ahead of an interpolated observation.
        """
        if not self._history or self._ahead is not None:
            return None
        export = self._engine.snapshot_rows
        backend = self._backend
        return MarchSnapshot(
            z=float(backend._z),
            # The snapshot's state is the trajectory's own; the march holds it with a
            # leading axis of one.
            state=np.array(backend._adams_state()[0], dtype=np.complex128, copy=True),
            proposal=float(backend._proposal),
            method=repr(self._method),
            order=int(self._order),
            rejections=int(self._rejections),
            last_rejected=bool(self._last_rejected),
            positions=np.array([entry.z for entry in self._history], dtype=np.float64),
            identities=tuple(entry.identity for entry in self._history),
            slots=np.array([entry.slot for entry in self._history], dtype=np.int64),
            rows=export(),
            collar_was_quiet=self._collar_was_quiet,
            history_key=self._method.history_key,
        )

    def snapshot_nbytes(self) -> int | None:
        """Return the bytes :meth:`snapshot` would copy, copying nothing.

        Returns
        -------
        int or None
            The state and every engine row, or ``None`` where :meth:`snapshot` returns
            ``None``.
        """
        if not self._history or self._ahead is not None:
            return None
        state = (
            math.prod(self._backend._problem.shape) * np.dtype(np.complex128).itemsize
        )
        return state + self._engine.snapshot_nbytes()

    def load(self, snapshot: MarchSnapshot) -> str | None:
        """Continue from a snapshot taken where the backend's state now stands.

        The backend has been loaded at the snapshot's position; this installs the
        accepted state exactly as it was held, the history, the order, the rejection
        count and the proposed step, so that the march goes on as the uninterrupted
        one would have. Nothing is changed when the snapshot cannot be used.

        Parameters
        ----------
        snapshot : solprop.output.records.MarchSnapshot
            What :meth:`snapshot` returned, possibly read back from a file.

        Returns
        -------
        str or None
            ``None`` when the history was continued, otherwise why it starts afresh.
        """
        backend = self._backend
        reason = self._refusal(snapshot)
        if reason is not None:
            return reason
        load_rows = self._engine.load_rows
        # The engine's rows keep the march's layout, a leading axis of one beside
        # their own (the snapshot's disk format, SEP 47 section 6.7).
        shape = (1, *backend._problem.shape)
        try:
            load_rows(snapshot.rows, shape)
        except BackendError as error:
            return str(error)
        backend._adams_restore(snapshot.z, np.array(snapshot.state, copy=True)[None])
        self._history = [
            Derivative(float(z), self._engine.value(int(slot)), identity, int(slot))
            for z, identity, slot in zip(
                snapshot.positions, snapshot.identities, snapshot.slots, strict=True
            )
        ]
        self._order = snapshot.order
        self._rejections = snapshot.rejections
        self._last_rejected = snapshot.last_rejected
        self._collar_was_quiet = snapshot.collar_was_quiet
        self._ahead = None
        backend._proposal = snapshot.proposal
        return None

    def _refusal(self, snapshot: MarchSnapshot) -> str | None:
        """Return why a snapshot cannot continue this march, or ``None``.

        - Returns: the reason, or ``None`` when the snapshot fits.
        """
        backend = self._backend
        checks = (
            (
                # Starting afresh is always safe.
                snapshot.history_key is None,
                "the snapshot names no history key",
            ),
            (
                snapshot.history_key != self._method.history_key,
                f"the snapshot's history was kept by {snapshot.history_key}, and "
                f"this run steps with {self._method.history_key}",
            ),
            (
                snapshot.z != backend._z,
                f"the snapshot is at z = {snapshot.z!r} m and the run resumes at "
                f"z = {backend._z!r} m",
            ),
            (
                snapshot.state.shape != tuple(backend._problem.shape),
                f"the snapshot's state has shape {snapshot.state.shape}",
            ),
            (
                # A landing snaps to its plane, a rounding from the step's z + dz.
                not math.isclose(
                    float(snapshot.positions[-1]), snapshot.z, rel_tol=1e-12
                ),
                "the snapshot's newest sample is not at its position",
            ),
            (
                bool(
                    np.any(
                        (snapshot.slots < 0) | (snapshot.slots >= self._engine.n_slots)
                    )
                ),
                "the snapshot's history names no row of this engine",
            ),
        )
        return next((reason for failed, reason in checks if failed), None)

    # --- evaluation -------------------------------------------------------------------

    def _source(
        self,
        lab: _State,
        anchor: float,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> tuple[_State, _State]:
        """Return the source at a lab-frame field, in the anchor's frame, and U.

        - ``anchor``, ``position``: the frame's anchor and the evaluation position, m.
        - Returns: ``U(anchor, position) N(lab, position)`` and ``U(position, anchor)``.
        """
        return self._backend._adams_source(lab, anchor, position, purpose=purpose)

    # --- the march --------------------------------------------------------------------

    def march_to(self, target: float) -> None:
        """Step the backend's state adaptively to `target`, landing on it exactly.

        Parameters
        ----------
        target : float
            Position to reach, m.

        Raises
        ------
        ConvergenceError
            If the proposal falls below the contract's ``dz_min``.
        """
        backend = self._backend
        settings = backend._stepper
        dense = self._interpolate
        if self._ahead is not None:
            # The march is ahead of the last observation; resume it there.
            self._engine.restore(*self._ahead)
            self._ahead = None
        if backend._natural_stop_due(target):
            return
        if settings.schedule is not None:
            self._replay_to(settings.schedule, target)
            return
        targets = adams_core.targets(self._policy, target)
        while not reached(backend._z, target):
            backend._poll_stop()
            self._afford()
            try:
                if not self._history:
                    self._start()
                proposal = self._collar_reentry(
                    backend._split_capped(
                        backend._transverse_capped(
                            self._quiet_capped(self._capped(backend._proposal))
                        )
                    )
                )
                stop = landing(backend._z, proposal, targets)
                dz, end, kink = adams_core.approach(
                    self._policy, backend._z, stop, targets, target
                )
                # The accepted step moves the position with the state, and restarts
                # the history at a kink, in one commit (`_land`).
                accepted, norm = self._attempt(dz, landing=(end, kink))
                backend._record(end, dz, norm, accepted)
                self._raise_floor()
                if accepted:
                    backend._take_step_stats()
                    if backend._natural_stop(target):
                        return
            finally:
                backend._release_source_calls()
        if dense and backend._z > target:
            self._ahead = (backend._z, self._engine.hold())
            backend._state = cast("Any", self._observe(target))
            backend._z = target
            self.ledger.dense_observations += 1

    def _replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay a prescribed schedule's steps to `target` (SEP 16 section 5).

        The schedule fixes the accepted positions: every step is taken as given and
        accepted, as the Runge-Kutta replay does (``docs/dev/stepper.md`` section 8),
        while the predictor-corrector estimate is still computed and recorded, so a
        schedule too coarse for its tolerance is reported rather than silent. The
        order is chosen by the same rule as an adaptive run's, from the same
        estimates; with no rejection it never has to wait a step to rise, so a replay
        of an adaptive run's steps can choose differently where that run rejected,
        while a replay of a replay is identical. A hard kink, an edge of the schedule,
        restarts the history as it does in an adaptive run.

        - ``target``: position to reach, m.
        """
        backend = self._backend
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        while (
            backend._schedule_index < schedule.n_steps and backend._z < target - slack
        ):
            backend._poll_stop()
            dz = float(schedule.steps[backend._schedule_index])
            end = float(edges[backend._schedule_index + 1])
            norm = self.replay_step(dz, end, slack)
            backend._schedule_index += 1
            backend._record(end, dz, norm, accepted=True)
            backend._accepted_norms.append(norm)
            self._raise_floor()
            backend._take_step_stats()
            if backend._natural_stop(target):
                return
        if backend._z < target - slack:
            msg = (
                f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                f"cannot reach z = {target:.6g} m"
            )
            raise BackendError(msg)

    def replay_step(self, dz: float, end: float, slack: float) -> float:
        """Take one prescribed step from the backend's position and accept it.

        The unit a replay and a lockstep group advance by
        (:class:`solprop.backends.host.group.HostGroup`): the budget, the startup of
        an empty history, the step and, when `end` is a hard kink, the restart.

        Parameters
        ----------
        dz : float
            The step, m.
        end : float
            The schedule's edge it ends on, m.
        slack : float
            How far an edge may sit from a kink it lands on, m.

        Returns
        -------
        float
            The step's scaled error norm, recorded and not acted on.

        Raises
        ------
        ConvergenceError
            If the norm is not finite: the field has diverged, and a replay cannot
            reject the step.
        """
        self._afford()
        try:
            if not self._history:
                self._start()
            kink = adams_core.replay_kink(self._policy, end, slack)
            _, norm = self._attempt(dz, fixed=True, landing=(end, kink))
            if not math.isfinite(norm):
                msg = (
                    f"{self._backend.name}: a replayed {self._method.name} step of "
                    f"{dz:.4g} m ending at z = {end:.6g} m has a non-finite error norm "
                    f"({norm!r}): the field has diverged. A replayed schedule takes "
                    "its steps as given and cannot reject one, so a step beyond an "
                    "explicit stability limit -- the in-stage temporal collar's or a "
                    "coupled flow's advisory step -- grows until it overflows; the "
                    "translation refuses a declared step past the collar's limit and "
                    "warns on one past the flow's. Take shorter steps, or let the "
                    "controller choose them"
                )
                raise ConvergenceError(msg)
        finally:
            self._backend._release_source_calls()
        return norm

    def _observe(self, position: float) -> object:
        """Return the field at `position` inside the last accepted step.

        Interpolated and carried to `position` by the exact linear flow (SEP 13
        section 6.4, SEP 16 section 4.3); never a history point, and no source call.

        - ``position``: the observation, m, inside the last accepted step.
        """
        return self._engine.observe(position)

    def _afford(self) -> None:
        """Refuse a trial the call budget cannot pay for, before any call is made."""
        self._raise_floor()
        budget = self._policy.max_calls
        cost = (0 if self._history else 1) + 1
        spent = self.ledger.total
        backend = self._backend
        if budget is not None and spent + cost > budget:
            raise self._budget_error(cost)
        backend._reserve_source_calls(cost)

    def _budget_error(self, cost: int) -> ResponseBudgetError:
        """Return the error of a trial the call budget cannot pay for."""
        backend = self._backend
        spent = self.ledger.total
        msg = (
            f"the source-call budget of {self._policy.max_calls} is spent ({spent} "
            f"charged) at z = {backend._z:.6g} m: the next trial needs {cost}; "
            "the run stops at its last accepted step, unfinished"
        )
        return ResponseBudgetError(
            msg, z=backend._z, field=backend._adams_state()[0].copy(), calls=spent
        )

    def _start(self) -> None:
        """Evaluate the source at the accepted field to (re)start the history."""
        backend = self._backend
        z = backend._z
        self.ledger.startup += 1
        value = self._engine.start(z)
        self._history = [Derivative(z, value, "accepted", 0)]
        self._order = 1 if self._policy.variable_order else self._policy.order

    def _capped(self, proposal: float) -> float:
        """Return the proposal shortened to the carrier frame's phase cap.

        Under a carrier frame no step covers more than
        :data:`~solprop.stepper.multistep.CARRIER_PHASE_CAP` of carrier phase (SEP 35
        section 6.4), measured by one chord of the reference element's operator over
        the proposal (:class:`CarrierChord`), the largest over the batch
        (:func:`~solprop.backends.adams_core.carrier_capped`). Applied before the
        landing rule, so a shortened step is not a landing; a fixed schedule's replay
        is not capped.

        - ``proposal``: the controller's proposal, m.
        - Returns: the proposal, or the shorter step the cap allows, m.
        """
        if not self._policy.carrier:
            return proposal
        carrier = cast("CarrierReference", self._engine.carrier)
        return adams_core.carrier_capped(
            self._policy, self._backend._z, proposal, carrier.chord_phase
        )

    def _quiet_capped(self, proposal: float) -> float:
        """Return the proposal held to the monitor's interval while the collar rests.

        :func:`~solprop.backends.adams_core.quiet_capped` of the backend's collar and
        monitor, the rule of the driver's own ``_quiet_capped``.

        - ``proposal``: the controller's proposal, m.
        - Returns: the proposal, or the monitor's interval while the collar is left out,
          m.
        """
        backend = self._backend
        monitor = backend._monitor
        return adams_core.quiet_capped(
            proposal,
            quiet=backend._collar_quiet,
            interval=None if monitor is None else monitor.interval,
        )

    def _attempt(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one step of length `dz` and commit it if accepted.

        - ``dz``: the step, m.
        - ``fixed``: whether the step is a prescribed schedule's, accepted whatever its
          norm.
        - ``landing``: where an accepted step ends, m, and whether that is a hard kink,
          which restarts the history; ``None`` ends it at ``z + dz`` and restarts
          nothing.
        - Returns: whether the step was accepted, and its scaled error norm.
        """
        if self._exponential is not None:
            return self._attempt_exponential(dz, fixed=fixed, landing=landing)
        return self._attempt_fused(dz, fixed=fixed, landing=landing)

    def _attempt_fused(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one step of the one-order-higher PEC family on the fused path.

        The arithmetic of :meth:`_attempt`, :meth:`_correct`, :meth:`_commit` and
        :meth:`_next_order` for this family, in the engine's passes
        (:meth:`HostHistory.fused_trial`): the defects of the neighbouring orders are
        scored in the same pass as the step's own, before the decision.

        - ``dz``: the step, m.
        - ``landing``: where an accepted step ends and whether it is a hard kink
          (:meth:`_attempt`).
        - Returns: whether the step was accepted, and its scaled error norm.
        """
        z, nodes, slots, order, orders = self._stencil(dz)
        self.ledger.predictor += 1
        norms = self._engine.fused_trial(z, dz, nodes, slots, order, orders)
        return self._decide(dz, order, orders, norms, fixed=fixed, landing=landing)

    def _stencil(
        self, dz: float
    ) -> tuple[float, list[float], np.ndarray, int, list[int]]:
        """Return a trial's anchor, nodes, rows, order and scored orders.

        :func:`~solprop.backends.adams_core.stencil` of the history.

        - ``dz``: the step, m.
        - Returns: the anchor, m; the history's normalised positions and store rows,
          newest first; the step's order; and the orders scored, its own first.
        """
        z = self._backend._z
        history = self._history
        stencil = adams_core.stencil(
            self._policy,
            z,
            dz,
            [entry.z for entry in history],
            [entry.slot for entry in history],
            self._order,
            last_rejected=self._last_rejected,
        )
        return z, stencil.nodes, stencil.rows, stencil.order, stencil.orders

    def _decide(
        self,
        dz: float,
        order: int,
        orders: list[int],
        norms: np.ndarray,
        *,
        fixed: bool,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Reject a held trial, or accept it and choose the next order and step.

        The new sample takes the oldest row past the method's need.

        - ``dz``: the step, m.
        - ``landing``: where an accepted step ends and whether it is a hard kink
          (:meth:`_attempt`).
        - Returns: whether the step was accepted, and its scaled error norm.
        """
        backend, policy, engine = self._backend, self._policy, self._engine
        norm = float(norms[0])
        if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
            self.ledger.rejected += 1
            self._reject(dz, norm, order)
            return False, norm
        z = backend._z
        keep = self._history[-policy.order :] if policy.order else []
        used = {entry.slot for entry in keep}
        free = next(slot for slot in range(engine.n_slots) if slot not in used)
        rows = np.array([entry.slot for entry in keep], dtype=np.int64)
        with committing():
            if self._exponential is None:
                engine.fused_accept(rows, free, interpolate=self._interpolate)
            else:
                engine.exponential_accept(rows, free, interpolate=self._interpolate)
            self._history = [
                *keep,
                Derivative(z + dz, engine.value(free), "predicted", free),
            ]
            self.ledger.accepted_steps += 1
            self._rejections = 0
            self.ledger.orders.append(order)
            next_order, control = self._choose_order(
                order, dict(zip(orders, norms.tolist(), strict=True)), dz
            )
            self._last_rejected = False
            self._land(z, dz, landing)
            self._propose(
                dz,
                adams_core.factor(
                    policy, control, next_order, growth=policy.max_growth
                ),
            )
        return True, norm

    def _propose(self, dz: float, factor: float) -> None:
        """Set an accepted step's next proposal, deferring a floor it falls below.

        Called inside an accepted step's commit, after the position, the history and
        the order have moved, so that a run stopped by the floor leaves the state of the
        step it accepted at that step's end (SEP 39 section 4.4). The floor is raised
        once the march has recorded the step (:meth:`_raise_floor`).

        - ``dz``: the accepted step, m.
        """
        policy = self._policy
        try:
            self._backend._proposal = limit_step(
                dz, factor, policy.dz_min, policy.dz_max
            )
        except ConvergenceError as error:
            self._floor = error

    def _raise_floor(self) -> None:
        """Raise the floor an accepted step's proposal fell below, once it is recorded.

        Every caller that records an accepted step calls this straight after, and every
        attempt calls it before it starts (:meth:`_afford`), so a floor is never
        passed over.
        """
        error, self._floor = self._floor, None
        if error is not None:
            raise error

    def _land(self, z: float, dz: float, landing: tuple[float, bool] | None) -> None:
        """Move the backend to an accepted step's end, restarting at a kink.

        Called inside the commit that installed the step's state, so that the state
        and its position change together whatever interrupts the march afterwards.

        - ``z``: the step's start, m.
        - ``dz``: the step, m.
        - ``landing``: where it ends and whether that is a hard kink (:meth:`_attempt`).
        """
        end, kink = (z + dz, False) if landing is None else landing
        self._backend._z = end
        if kink:
            self._history = []
            self.ledger.restarts += 1

    def _attempt_exponential(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one exponential Adams step and commit it if accepted (SEP 16).

        The controller of :meth:`_attempt_fused` with the engine's exponential trial
        (:meth:`HostHistory.exponential_trial`): untransported samples, integrated
        exactly against the step's flow; an accepted step re-anchors nothing.

        - ``dz``: the step, m.
        - ``landing``: where an accepted step ends and whether it is a hard kink
          (:meth:`_attempt`).
        - Returns: whether the step was accepted, and its scaled error norm.
        """
        z, nodes, slots, order, orders = self._stencil(dz)
        self.ledger.predictor += 1
        norms = self._engine.exponential_trial(z, dz, nodes, slots, order, orders)
        return self._decide(dz, order, orders, norms, fixed=fixed, landing=landing)

    def _collar_reentry(self, proposal: float) -> float:
        """Return the proposal held to the collar's stable step as its term returns.

        :func:`~solprop.backends.adams_core.collar_reentry`, recording the collar's
        flag at this attempt.

        - ``proposal``: the controller's proposal, m.
        - Returns: the proposal, or the collar's stable step when the term has just
          returned, m.
        """
        quiet = self._backend._collar_quiet
        capped = adams_core.collar_reentry(
            self._policy,
            proposal,
            self._order,
            was_quiet=self._collar_was_quiet,
            quiet=quiet,
        )
        self._collar_was_quiet = quiet
        return capped

    def _choose_order(
        self, current: int, scores: dict[int, float], dz: float
    ) -> tuple[int, float]:
        """Store the next order; return the order and norm that set the next step.

        :func:`~solprop.backends.adams_core.choose_order`. In the exponential
        formulation each order is weighed by the step its defect allows capped at the
        step an in-stage collar leaves it stable at: near that limit the higher orders'
        defects carry the collar's growing mode, and a choice on them alone was decided
        by rounding -- runs a rounding apart settled in different orders, with calls
        40 % apart. The cap weighs the choice only; the step is the chosen order's
        defect's, as before.

        - ``dz``: the accepted step, m.
        - Returns: the order and the norm the next step is controlled with.
        """
        stored, control, norm = adams_core.choose_order(
            self._policy, current, scores, dz, quiet=self._backend._collar_quiet
        )
        self._order = stored
        return control, norm

    def _reject(self, dz: float, norm: float, order: int) -> None:
        """Shrink the proposal after a rejected step; restart a failing history.

        :func:`~solprop.backends.adams_core.rejection`: a rejection leaves the accepted
        history untouched, and after
        :data:`~solprop.stepper.multistep.RESTART_AFTER` in a row the method restarts
        at order one from the accepted state, paying one startup call.

        - ``dz``: the rejected step, m.
        - ``norm``: its scaled defect, possibly not finite.
        """
        policy = self._policy
        self.ledger.rejected_steps += 1
        self._last_rejected = True
        outcome = adams_core.rejection(
            policy, norm, order, rejections=self._rejections, held=len(self._history)
        )
        self._rejections = outcome.rejections
        if outcome.restart:
            self._history = []
            self.ledger.failure_restarts += 1
        self._backend._proposal = limit_step(
            dz, outcome.factor, policy.dz_min, policy.dz_max
        )


_ENGINE_MEMBERS = (
    "n_slots",
    "carrier",
    "dense",
    "reset",
    "hold",
    "restore",
    "start",
    "value",
    "observe",
)
"""What every march path asks of an engine, beside its own trial and acceptance."""


def _check_engine(engine: object, *, exponential: bool) -> None:
    """Refuse an injected engine that lacks a member its path calls."""
    family = "exponential" if exponential else "fused"
    wanted = (*_ENGINE_MEMBERS, f"{family}_trial", f"{family}_accept")
    missing = [name for name in wanted if not hasattr(engine, name)]
    if missing:
        msg = f"an Adams engine for the {family} path lacks {', '.join(missing)}"
        raise BackendError(msg)
