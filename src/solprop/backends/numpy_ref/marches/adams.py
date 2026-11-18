# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's Adams march (SEP 13, SEP 14, SEP 39, SEP 52).

The controller, history, ledger and lifecycle of :class:`HostAdamsMarch`, over the
engine protocol (:class:`solprop.backends.engines.AdamsEngine`) that native's
``march::Engine`` mirrors: the reference's engine is
:class:`solprop.backends.numpy_ref.multistep.HostHistory`. The march holds its own state
-- position, proposal and replay cursor
(:class:`~solprop.backends.numpy_ref.marches.host.MarchState`), the history's entries,
the order, the rejections and the call ledger -- and asks its session only through
:class:`~solprop.backends.numpy_ref.marches.host.MarchHost`. The decisions are
:mod:`solprop.backends.adams_core`'s functions of one policy record.

The method, its lifecycle and its numbers are those of
:mod:`solprop.backends.numpy_ref.multistep`'s module documentation.
"""

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, get_protocol_members

import numpy as np

from solprop._interrupts import committing
from solprop.backends import adams_core
from solprop.backends.engines import AdamsEngine, AttemptOutcome, HistoryEntry
from solprop.backends.ledger import CallLedger
from solprop.backends.numpy_ref.marches.host import (
    MarchHost,
    MarchState,
    replay_divergence,
    replay_shortfall,
)
from solprop.exceptions import BackendError, ConvergenceError, ResponseBudgetError
from solprop.output.records import MarchSnapshot
from solprop.stepper.controller import limit_step
from solprop.stepper.multistep import MarchPolicy, Multistep
from solprop.stepper.schedule import FixedSchedule, landing, reached, replay_tolerance

__all__ = ["AcceptedStep", "HostAdamsMarch"]


@dataclass(frozen=True, eq=False)
class AcceptedStep:
    """The last accepted step, as the session's dense observation reads it.

    The counterpart of a native march call's ``step_z``, ``step_dz``, ``step_order``,
    ``step_nodes`` and ``step_carrier``. Equality is by identity (``eq=False``)
    because it holds an array.

    Attributes
    ----------
    z : float
        Where the step started, m.
    dz : float
        Its length, m.
    order : int
        Its order.
    nodes : tuple of float
        The history's normalised positions it read, newest first, ``order`` of them.
    carrier : numpy.ndarray
        Its trial's carrier rotations
        (:meth:`~solprop.backends.engines.AdamsEngine.step_carrier`); empty without a
        carrier frame.
    """

    z: float
    dz: float
    order: int
    nodes: tuple[float, ...]
    carrier: np.ndarray


class HostAdamsMarch:
    """March a state with an Adams method over an engine.

    Parameters
    ----------
    engine : solprop.backends.engines.AdamsEngine
        The engine that holds the history's rows.
    method : solprop.stepper.multistep.Multistep
        The method.
    host : solprop.backends.numpy_ref.marches.host.MarchHost
        The session the march reports to.
    policy : solprop.stepper.multistep.MarchPolicy
        Every number the controller and the landings read (SEP 39 section 4.2).
    field : callable
        Returns the committed state, for a budget's error.
    schedule : solprop.stepper.schedule.FixedSchedule or None, optional
        The schedule to replay, or ``None`` to step adaptively.
    interpolate : bool, optional
        Whether the run interpolates its planes (ADR-036, ADR-039): each accepted
        step's record is then kept for the session's dense observation
        (:attr:`last_step`).
    shape : tuple of int, optional
        The state's shape, which a snapshot's is checked against.
    proposal : float, optional
        The first proposal, m.
    name : str, optional
        The backend's name, for messages.

    Attributes
    ----------
    state : solprop.backends.numpy_ref.marches.host.MarchState
        The march's position, proposal and cursor.
    ledger : solprop.backends.ledger.CallLedger
        The run's call counts.
    last_step : AcceptedStep or None
        The last accepted step of an interpolated run; ``None`` before one, and
        always without interpolation.

    Raises
    ------
    solprop.exceptions.BackendError
        If `engine` lacks a member the march calls.
    """

    def __init__(
        self,
        engine: object,
        method: Multistep,
        host: object,
        *,
        policy: MarchPolicy,
        field: Callable[[], np.ndarray],
        schedule: FixedSchedule | None = None,
        interpolate: bool = False,
        shape: tuple[int, ...] = (),
        proposal: float = 0.0,
        name: str = "numpy_ref",
    ) -> None:
        """Hold the engine and start with no history."""
        _check_engine(engine)
        self._engine: AdamsEngine = cast("AdamsEngine", engine)
        self._method = method
        self._host = cast("MarchHost", host)
        self._policy = policy
        self._field = field
        self._schedule = schedule
        self._interpolate = interpolate
        self._shape = shape
        self._name = name
        self.state = MarchState(z=0.0, proposal=proposal)
        self._history: list[HistoryEntry] = []
        self._rejections = 0
        self._last_rejected = False
        self._order = 1 if method.variable_order else method.order
        # The engine's history rows: the method's order and two more.
        self._n_slots = method.order + 2
        self.last_step: AcceptedStep | None = None
        self.ledger = CallLedger()
        # Whether the stages left a quiet collar out at the last attempt: the step
        # after it returns is held to the collar's stable step (`_collar_reentry`).
        self._collar_was_quiet = False
        # A floor an attempt's next proposal fell below, raised once the attempt is
        # recorded (`_propose`, `_reject`, `_raise_floor`).
        self._floor: ConvergenceError | None = None

    def load(self, z: float, proposal: float, cursor: int) -> None:
        """Start a run afresh at a loaded position.

        The history and the held trial are forgotten, and the run's accounting starts
        afresh -- the ledger, the last rejection and the collar's flag -- so that a
        backend loaded for another run counts that run's calls alone and chooses its
        first order and step as a fresh backend would (SEP 22 section 4.7, SEP 39
        R9). A resume restores its persisted ledger and flags afterwards.

        Parameters
        ----------
        z : float
            The position, m.
        proposal : float
            The first proposal, m.
        cursor : int
            The replay cursor at `z`.
        """
        self.state.load(z, proposal, cursor)
        self.ledger = CallLedger()
        self._last_rejected = False
        self._collar_was_quiet = False
        self._floor = None
        self._history = []
        self._rejections = 0
        self.last_step = None

    def restore_ledger(self, logical: Mapping[str, object]) -> None:
        """Continue a resumed run's cumulative call counts.

        Parameters
        ----------
        logical : collections.abc.Mapping
            The persisted logical counts
            (:meth:`solprop.backends.ledger.CallLedger.to_record`).
        """
        self.ledger = CallLedger.from_record(logical)

    def snapshot_fields(self) -> dict[str, Any] | None:
        """Return the march's part of a snapshot at its accepted position.

        The session adds the state and the engine's rows
        (:class:`~solprop.output.records.MarchSnapshot`'s ``state`` and ``rows``).

        Returns
        -------
        dict or None
            The snapshot's other fields, or ``None`` with no history to continue.
        """
        if not self._history:
            return None
        state = self.state
        return {
            "z": float(state.z),
            "proposal": float(state.proposal),
            "method": repr(self._method),
            "order": int(self._order),
            "rejections": int(self._rejections),
            "last_rejected": bool(self._last_rejected),
            "positions": np.array(
                [entry.z for entry in self._history], dtype=np.float64
            ),
            "identities": tuple(entry.identity for entry in self._history),
            "slots": np.array([entry.row for entry in self._history], dtype=np.int64),
            "collar_was_quiet": self._collar_was_quiet,
            "history_key": self._method.history_key,
        }

    @property
    def in_flight(self) -> None:  # numpydoc ignore=RT01
        """No step in flight: an Adams trial never overwrites the committed state."""
        return

    @property
    def has_history(self) -> bool:  # numpydoc ignore=RT01
        """Whether the march holds a history a snapshot could continue."""
        return bool(self._history)

    def refusal(self, snapshot: MarchSnapshot) -> str | None:
        """Return why a snapshot cannot continue this march, or ``None``.

        Parameters
        ----------
        snapshot : solprop.output.records.MarchSnapshot
            The snapshot, possibly read back from a file.

        Returns
        -------
        str or None
            The reason, or ``None`` when it can.
        """
        return self._refusal(snapshot)

    def continue_from(self, snapshot: MarchSnapshot) -> None:
        """Continue from a snapshot taken where the march now stands.

        Installs the history, the order, the rejection count, the flags and the
        proposed step, so that the march goes on as the uninterrupted one would have;
        the session installs the state and the engine's rows, once :meth:`refusal`
        found nothing.

        Parameters
        ----------
        snapshot : solprop.output.records.MarchSnapshot
            The snapshot.
        """
        self.state.z = snapshot.z
        self._history = [
            HistoryEntry(float(z), cast("Any", identity), int(slot))
            for z, identity, slot in zip(
                snapshot.positions, snapshot.identities, snapshot.slots, strict=True
            )
        ]
        self._order = snapshot.order
        self._rejections = snapshot.rejections
        self._last_rejected = snapshot.last_rejected
        self._collar_was_quiet = snapshot.collar_was_quiet
        self.state.proposal = snapshot.proposal

    def _refusal(self, snapshot: MarchSnapshot) -> str | None:
        """Return why a snapshot cannot continue this march, or ``None``."""
        z = self.state.z
        n_slots = self._n_slots
        checks = (
            (
                snapshot.history_key != self._method.history_key,
                f"the snapshot's history was kept by {snapshot.history_key}, and "
                f"this run steps with {self._method.history_key}",
            ),
            (
                snapshot.z != z,
                f"the snapshot is at z = {snapshot.z!r} m and the run resumes at "
                f"z = {z!r} m",
            ),
            (
                snapshot.state.shape != tuple(self._shape),
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
                bool(np.any((snapshot.slots < 0) | (snapshot.slots >= n_slots))),
                "the snapshot's history names no row of this engine",
            ),
        )
        return next((reason for failed, reason in checks if failed), None)

    # --- the march --------------------------------------------------------------------

    def advance(self, target: float) -> None:
        """Step towards `target`: the schedule's replay, or adaptively.

        Parameters
        ----------
        target : float
            The position to reach, m.
        """
        self.march_to(target)

    def march_to(self, target: float) -> None:
        """Step adaptively to `target`, landing on it exactly, or replay towards it.

        Parameters
        ----------
        target : float
            Position to reach, m.

        Raises
        ------
        ConvergenceError
            If the proposal falls below the contract's ``dz_min``.
        """
        host, state = self._host, self.state
        if self._schedule is not None:
            self._replay_to(self._schedule, target)
            return
        targets = adams_core.targets(self._policy, target)
        while not reached(state.z, target):
            host.poll_stop()
            with self._attempt_scope() as attempt:
                if not self._history:
                    self._start()
                proposal = self._collar_reentry(
                    host.cap(self._capped(state.proposal), state.z)
                )
                stop = landing(state.z, proposal, targets)
                dz, end, kink = adams_core.approach(
                    self._policy, state.z, stop, targets, target
                )
                # The accepted step moves the position with the state, and restarts
                # the history at a kink, in one commit (`_land`).
                accepted, norm = self._attempt(dz, landing=(end, kink))
                host.record(end, dz, norm, accepted)
                self._end(attempt, accepted=accepted)
            if accepted:
                host.stepped()
                if host.natural_stop(target):
                    return

    def _replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay a prescribed schedule's steps to `target` (SEP 16 section 5).

        The schedule fixes the accepted positions: every step is taken as given and
        accepted, as the Runge-Kutta replay does (``docs/dev/stepper.md`` section 8),
        while the predictor-corrector estimate is still computed and recorded, so a
        schedule too coarse for its tolerance is reported rather than silent. The
        order is chosen by the same rule as an adaptive run's, from the same
        estimates. A hard kink, an edge of the schedule, restarts the history as it
        does in an adaptive run.

        - ``target``: position to reach, m.
        """
        host, state = self._host, self.state
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        while state.cursor < schedule.n_steps and state.z < target - slack:
            host.poll_stop()
            dz = float(schedule.steps[state.cursor])
            end = float(edges[state.cursor + 1])
            with self._attempt_scope() as attempt:
                norm = self.replay_step(dz, end, slack)
                state.cursor += 1
                host.record(end, dz, norm, accepted=True)
                attempt.end(AttemptOutcome.ACCEPTED)
            self._raise_floor()
            host.stepped()
            if host.natural_stop(target):
                return
        if state.z < target - slack:
            raise BackendError(replay_shortfall(float(schedule.z_end), target))

    def replay_step(self, dz: float, end: float, slack: float) -> float:
        """Take one prescribed step from the march's position and accept it.

        The unit a replay advances by: the startup of an empty history, the step and,
        when `end` is a hard kink, the restart. Called inside an attempt, whose
        reservation the caller opened.

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
        if not self._history:
            self._start()
        kink = adams_core.replay_kink(self._policy, end, slack)
        _, norm = self._attempt(dz, fixed=True, landing=(end, kink))
        if not math.isfinite(norm):
            step = (
                f"{self._name}: a replayed {self._method.name} step of "
                f"{dz:.4g} m ending at z = {end:.6g} m"
            )
            raise ConvergenceError(replay_divergence(step, norm))
        return norm

    def _end(self, attempt: "_Attempt", *, accepted: bool) -> None:
        """End a recorded attempt with its decision, then raise a floor it fell below.

        As native's march does (``adams_march.cpp``): an accepted attempt ends first,
        and a failure there is the run's error, its step committed; a rejected
        attempt's floor is the run's error, and a failure of its end rides on it as a
        note (SEP 49 section 6.3, issue #651).
        """
        if accepted:
            attempt.end(AttemptOutcome.ACCEPTED)
            self._raise_floor()
            return
        error, self._floor = self._floor, None
        if error is None:
            attempt.end(AttemptOutcome.REJECTED)
            return
        try:
            attempt.end(AttemptOutcome.REJECTED)
        except Exception as cleanup:
            error.add_note(f"releasing the source reservation failed: {cleanup}")
        raise error

    @contextmanager
    def _attempt_scope(self) -> Iterator["_Attempt"]:
        """Open an attempt with its host-source reservation; end it on every exit.

        The budget is checked and the reservation made before the attempt's first
        evaluation (SEP 49 section 6.2). An attempt the body does not end -- a
        failure, a floor, a diverged replay -- ends abandoned, which only releases
        what it reserved; that release's own failure is attached to the error as a
        note, never raised in its place (SEP 49 section 6.3).
        """
        attempt = _Attempt(self._engine)
        self._afford(attempt)
        try:
            yield attempt
        except BaseException as error:
            if attempt.open:
                try:
                    attempt.end(AttemptOutcome.ABANDONED)
                except Exception as cleanup:
                    error.add_note(
                        f"releasing the source reservation failed: {cleanup}"
                    )
            raise

    def _afford(self, attempt: "_Attempt") -> None:
        """Refuse a trial the call budget cannot pay for, before any call is made."""
        self._raise_floor()
        budget = self._policy.max_calls
        cost = (0 if self._history else 1) + 1
        spent = self.ledger.total
        z = self.state.z
        if budget is not None and spent + cost > budget:
            msg = (
                f"the source-call budget of {budget} is spent ({spent} charged) at "
                f"z = {z:.6g} m: the next trial needs {cost}; the run stops at its "
                "last accepted step, unfinished"
            )
            raise ResponseBudgetError(msg, z=z, field=self._field().copy(), calls=spent)
        try:
            attempt.begin(cost)
        except ResponseBudgetError as error:
            raise ResponseBudgetError(
                "the TDSE atomic-call budget cannot reserve the next trial "
                f"at z = {z:.6g} m ({error.calls} charged); "
                "the run stops at its last accepted step, unfinished",
                z=z,
                field=self._field().copy(),
                calls=error.calls,
            ) from None

    def _start(self) -> None:
        """Evaluate the source at the accepted field to (re)start the history."""
        z = self.state.z
        self.ledger.startup += 1
        self._engine.start(z)
        self._history = [HistoryEntry(z, "accepted", 0)]
        self._order = 1 if self._policy.variable_order else self._policy.order

    def _capped(self, proposal: float) -> float:
        """Return the proposal shortened to the carrier frame's phase cap.

        Under a carrier frame no step covers more than
        :data:`~solprop.stepper.multistep.CARRIER_PHASE_CAP` of carrier phase (SEP 35
        section 6.4), measured by one chord of the reference element's operator over
        the proposal (:func:`~solprop.backends.adams_core.carrier_capped`). Applied
        before the landing rule, so a shortened step is not a landing; a fixed
        schedule's replay is not capped.

        - ``proposal``: the controller's proposal, m.
        """
        if not self._policy.carrier:
            return proposal
        return adams_core.carrier_capped(
            self._policy, self.state.z, proposal, self._engine.chord_phase
        )

    def _attempt(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one step of length `dz` and commit it if accepted.

        - ``fixed``: whether the step is a prescribed schedule's, accepted whatever its
          norm.
        - ``landing``: where an accepted step ends, m, and whether that is a hard kink,
          which restarts the history; ``None`` ends it at ``z + dz`` and restarts
          nothing.
        - Returns: whether the step was accepted, and its scaled error norm.
        """
        stencil = adams_core.stencil(
            self._policy,
            self.state.z,
            dz,
            [entry.z for entry in self._history],
            [entry.row for entry in self._history],
            self._order,
            last_rejected=self._last_rejected,
        )
        self.ledger.predictor += 1
        norms = self._engine.trial(self.state.z, dz, stencil)
        return self._decide(dz, stencil, norms, fixed=fixed, landing=landing)

    def _decide(
        self,
        dz: float,
        stencil: adams_core.Stencil,
        norms: np.ndarray,
        *,
        fixed: bool,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Reject a held trial, or accept it and choose the next order and step.

        The new sample takes the oldest row past the method's need.

        - ``landing``: where an accepted step ends and whether it is a hard kink
          (:meth:`_attempt`).
        - Returns: whether the step was accepted, and its scaled error norm.
        """
        policy, engine, order = self._policy, self._engine, stencil.order
        norm = float(norms[0])
        if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
            self.ledger.rejected += 1
            self._reject(dz, norm, order)
            if fixed:
                # A replay records no rejected attempt (`replay_step`): its floor is
                # raised at once.
                self._raise_floor()
            return False, norm
        z = self.state.z
        keep = self._history[-policy.order :] if policy.order else []
        used = {entry.row for entry in keep}
        free = next(slot for slot in range(self._n_slots) if slot not in used)
        with committing():
            record = engine.accept(dz, free, stencil, keep)
            self._host.absorb(record)
            self._history = [*keep, HistoryEntry(z + dz, "predicted", free)]
            self.ledger.accepted_steps += 1
            self._rejections = 0
            self.ledger.orders.append(order)
            next_order, control = self._choose_order(
                order, dict(zip(stencil.orders, norms.tolist(), strict=True)), dz
            )
            self._last_rejected = False
            if self._interpolate:
                # The step's record for the session's dense observation, its trial's
                # carrier rotations among it (`adams_march.cpp`'s `step_carrier`).
                self.last_step = AcceptedStep(
                    z=z,
                    dz=dz,
                    order=order,
                    nodes=tuple(stencil.nodes[:order]),
                    carrier=engine.step_carrier(order),
                )
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
        """
        policy = self._policy
        try:
            self.state.proposal = limit_step(dz, factor, policy.dz_min, policy.dz_max)
        except ConvergenceError as error:
            self._floor = error

    def _raise_floor(self) -> None:
        """Raise the floor an attempt's proposal fell below, once it is recorded.

        Every caller that records an attempt calls this straight after, and every
        attempt calls it before it starts (:meth:`_afford`), so a floor is never
        passed over.
        """
        error, self._floor = self._floor, None
        if error is not None:
            raise error

    def _land(self, z: float, dz: float, landing: tuple[float, bool] | None) -> None:
        """Move to an accepted step's end, restarting at a kink.

        Called inside the commit that installed the step's state, so that the state
        and its position change together whatever interrupts the march afterwards.
        """
        end, kink = (z + dz, False) if landing is None else landing
        self.state.z = end
        if kink:
            self._history = []
            self.ledger.restarts += 1

    def _collar_reentry(self, proposal: float) -> float:
        """Return the proposal held to the collar's stable step as its term returns.

        :func:`~solprop.backends.adams_core.collar_reentry`, recording the collar's
        flag at this attempt.
        """
        quiet = self._host.collar_quiet
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
        """
        stored, control, norm = adams_core.choose_order(
            self._policy, current, scores, dz, quiet=self._host.collar_quiet
        )
        self._order = stored
        return control, norm

    def _reject(self, dz: float, norm: float, order: int) -> None:
        """Shrink the proposal after a rejected step; restart a failing history.

        :func:`~solprop.backends.adams_core.rejection`: a rejection leaves the accepted
        history untouched, and after
        :data:`~solprop.stepper.multistep.RESTART_AFTER` in a row the method restarts
        at order one from the accepted state, paying one startup call. A proposal
        below the floor is left as it was, and the floor is raised once the march has
        recorded the rejected attempt (:meth:`_raise_floor`; issue #651).
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
        try:
            self.state.proposal = limit_step(
                dz, outcome.factor, policy.dz_min, policy.dz_max
            )
        except ConvergenceError as error:
            self._floor = error


class _Attempt:
    """An attempt's source reservation, ended exactly once (native ``Attempt``)."""

    def __init__(self, engine: object) -> None:
        self._engine = cast("AdamsEngine", engine)
        self.open = False

    def begin(self, cost: int) -> None:
        """Reserve the attempt's calls; the attempt is open from here."""
        self._engine.begin_attempt(cost)
        self.open = True

    def end(self, outcome: AttemptOutcome) -> None:
        """End with the decision; closed first, so a failure here is not abandoned."""
        self.open = False
        self._engine.end_attempt(outcome)


_ENGINE_MEMBERS = tuple(sorted(get_protocol_members(AdamsEngine)))
"""What the march asks of an engine: the protocol's members, sorted."""


def _check_engine(engine: object) -> None:
    """Refuse an engine that lacks a member the march calls."""
    missing = [name for name in _ENGINE_MEMBERS if not hasattr(engine, name)]
    if missing:
        msg = f"an Adams engine lacks {', '.join(missing)}"
        raise BackendError(msg)
