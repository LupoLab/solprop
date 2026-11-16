# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's Runge-Kutta march, adaptive and replayed (SEP 45, SEP 52).

:class:`RungeKuttaMarch` holds the march's state (:class:`~solprop.backends.host.
marches.MarchState`) and makes its decisions with :mod:`solprop.backends.rk_core`'s
functions of one policy record; every state-sized operation is its engine's
(:class:`~solprop.backends.engines.RungeKuttaEngine`). It is the counterpart of
native's ``rk_march.cpp``, whose decisions on identical inputs are its own (the
decision traces of ``tests/backends``).

**The controller's state is its own proposal**, never the clipped or stretched step
that was executed, and what the history and a rejection see is the norm the proposal
would have scored while what decides acceptance is the error the executed step
committed (``docs/dev/stepper.md`` section 3).
"""

import math
from collections import deque
from typing import cast

import numpy as np

from solprop._interrupts import committing
from solprop.backends import rk_core
from solprop.backends.engines import RungeKuttaEngine
from solprop.backends.host.marches import MarchHost, MarchState
from solprop.backends.records import RungeKuttaPolicy
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper.schedule import (
    FixedSchedule,
    StepTargets,
    landing,
    reached,
    replay_tolerance,
)

__all__ = ["RungeKuttaMarch"]


class RungeKuttaMarch:
    """Step a state adaptively, or replay a schedule, with an embedded pair.

    Parameters
    ----------
    engine : solprop.backends.engines.RungeKuttaEngine
        The engine; beside the protocol, the march's dense observation calls its
        ``hold``, ``restore`` and ``observe``, and a load its ``reset``.
    policy : solprop.backends.records.RungeKuttaPolicy
        The controller and the landings.
    host : solprop.backends.host.marches.MarchHost
        The session the march reports to.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        The schedule to replay, or ``None`` to step adaptively.
    proposal : float
        The first proposal, m.
    name : str, optional
        The backend's name, for messages.

    Attributes
    ----------
    state : solprop.backends.host.marches.MarchState
        The march's position, proposal, cursor and accepted norms.
    """

    def __init__(
        self,
        engine: object,
        policy: RungeKuttaPolicy,
        host: object,
        *,
        schedule: FixedSchedule | None,
        proposal: float,
        name: str = "numpy_ref",
    ) -> None:
        """Hold the engine and the policy; the march stands at the entrance."""
        self._engine = cast("RungeKuttaEngine", engine)
        self._policy = policy
        self._host = cast("MarchHost", host)
        self._schedule = schedule
        self._name = name
        self.state = MarchState(
            z=0.0, proposal=proposal, norms=deque(maxlen=policy.controller.lag + 2)
        )
        # The march's own position and state while an interpolated plane is observed.
        self._ahead: tuple[float, object] | None = None

    @property
    def holding(self) -> bool:  # numpydoc ignore=RT01
        """Whether a dense observation holds the state at a plane, the march ahead."""
        return self._ahead is not None

    def load(self, z: float, proposal: float, cursor: int) -> None:
        """Start afresh at a loaded position.

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
        self._ahead = None
        self._engine.reset()  # pyright: ignore[reportAttributeAccessIssue]

    def advance(self, target: float) -> None:
        """Replay the schedule, or step adaptively, towards `target`.

        Parameters
        ----------
        target : float
            The position to reach, m.
        """
        if self._schedule is not None:
            self.replay_to(self._schedule, target)
        else:
            self.march_to(target)

    def march_to(self, target: float) -> None:
        """Step adaptively to `target`, landing on it exactly.

        Parameters
        ----------
        target : float
            The position to reach, m.

        Raises
        ------
        ConvergenceError
            If the controller asks for a step below the policy's floor.
        """
        policy, host, state, engine = self._policy, self._host, self.state, self._engine
        dense = policy.interpolate
        if dense and self._ahead is not None:
            # The march is ahead of the last interpolated plane; resume it there.
            z, held = self._ahead
            engine.restore(z, held)  # pyright: ignore[reportAttributeAccessIssue]
            state.z = z
            self._ahead = None
        if host.natural_stop_due(target):
            return
        # The policy's landings (SEP 45 section 4.2): only the operator table's two
        # ends bound a step without the interval cap (SEP 14 section 4.1), and dense
        # output steps across the save planes, still landing on the end of the run and
        # the hard kinks (SEP 14 section 4.3).
        targets = StepTargets(
            save_planes=np.asarray(policy.save_planes, dtype=np.float64),
            hard_kinks=np.asarray(policy.hard_kinks, dtype=np.float64),
            breakpoints=np.asarray(policy.breakpoints, dtype=np.float64),
            dz_max=policy.dz_max,
            z_end=policy.z_end if dense else target,
            stretch=policy.stretch,
        )
        while not reached(state.z, target):
            host.poll_stop()
            # The controller's state is its own proposal, never the clipped or
            # stretched step that was executed: otherwise every save plane restarts it
            # from a sliver and it regrows at the growth cap, which over two hundred
            # planes costs more steps than the run needs between them (stepper.md 3).
            stop = landing(state.z, host.cap(state.proposal, state.z), targets)
            norm = engine.trial(state.z, stop.dz)
            # What the controller learns from, and what a rejection is scaled by, is the
            # norm the *proposal* would have scored; what decides acceptance, and what
            # the history records, is the error the executed step actually committed,
            # so a short landing step that was accurate is neither thrown away nor
            # written up as a failure.
            equivalent, accepted = rk_core.decide(policy, norm, stop.dz, state.proposal)
            # A landing snaps to its target rather than accumulating z + dz, so that a
            # save plane is hit exactly however many steps have gone before.
            end = stop.z_target if stop.z_target is not None else state.z + stop.dz
            host.record(end, stop.dz, norm, accepted)
            if accepted:
                if dense and not reached(target, end):
                    # The step crosses the plane: hold its extension for the
                    # observation, and for any later plane it also crosses.
                    engine.extend(state.z, stop.dz)
                with committing():
                    record = engine.accept(stop.dz)
                    state.z = end
                host.absorb(record)
                state.norms.append(equivalent)
                if policy.auto_init and len(state.norms) == 1:
                    self._start_up(equivalent)
                # The next proposal grows or shrinks once per accepted step, from the
                # history the step completed. A rejected step is retried at the step
                # rejection_factor left: applying this factor again there undid the
                # shrink, and with a growth cap above about 1.27 a step whose norm sat
                # just above one was rejected unchanged forever.
                state.proposal = rk_core.propose(policy, state.proposal, state.norms)
                host.stepped()
                if host.natural_stop(target):
                    return
            else:
                state.proposal = rk_core.reject(policy, state.proposal, equivalent)
        if dense and not reached(target, state.z):
            self._observe_inside(target)

    def _observe_inside(self, position: float) -> None:
        """Make the state the interpolated field at `position` and hold the march's.

        - ``position``: the plane, m, inside the last accepted step.
        """
        engine, state = self._engine, self.state
        observed = engine.observe(position)  # pyright: ignore[reportAttributeAccessIssue]
        self._ahead = (state.z, engine.hold())  # pyright: ignore[reportAttributeAccessIssue]
        engine.restore(position, observed)  # pyright: ignore[reportAttributeAccessIssue]
        state.z = position

    def _start_up(self, norm: float) -> None:
        """Take the step after the first from the first step's own estimate.

        The first proposal is a guess; its scaled norm says how far it is from the
        tolerance. The next proposal is the deadbeat step that norm implies, at most
        :data:`~solprop.stepper.controller.START_GROWTH` times the first, and the
        history is told the norm the first step would have scored at that length, so
        the lagged controller does not apply the guess's margin a second time (SEP 14
        section 4.2). A norm small enough to meet the cap is recorded as zero, no
        estimate, so that round-off does not steer the later steps.
        """
        state = self.state
        state.proposal, state.norms[-1] = rk_core.start_up(
            self._policy, state.proposal, norm
        )

    def replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay a fixed schedule to `target`.

        A replay takes its steps as given and never rejects, but the embedded estimate
        costs nothing extra and is computed and recorded every step, so that a schedule
        too coarse for its tolerance is reported rather than silent
        (``docs/dev/stepper.md`` section 8). The position is read off the schedule's
        edges rather than accumulated, so every save plane is hit exactly.

        Parameters
        ----------
        schedule : solprop.stepper.schedule.FixedSchedule
            The schedule.
        target : float
            The position to reach, m.

        Raises
        ------
        ConvergenceError
            If a replayed step's error norm is not finite.
        BackendError
            If the schedule ends before `target`.
        """
        host, state, engine = self._host, self.state, self._engine
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        while state.cursor < schedule.n_steps and state.z < target - slack:
            host.poll_stop()
            dz = float(schedule.steps[state.cursor])
            norm = engine.trial(state.z, dz)
            if not math.isfinite(norm):
                msg = (
                    f"{self._name}: replayed step {state.cursor + 1} of "
                    f"{schedule.n_steps}, {dz:.4g} m from z = {state.z:.6g} m, has a "
                    f"non-finite error norm ({norm!r}): the field has diverged. A "
                    "replayed schedule takes its steps as given and cannot reject "
                    "one, so a step beyond an explicit stability limit -- the "
                    "in-stage temporal collar's or a coupled flow's advisory step -- "
                    "grows until it overflows; the translation refuses a declared step "
                    "past the collar's limit and warns on one past the flow's. Take "
                    "shorter steps, or let the controller choose them"
                )
                raise ConvergenceError(msg)
            end = float(edges[state.cursor + 1])
            with committing():
                record = engine.accept(dz)
                state.cursor += 1
                state.z = end
            host.absorb(record)
            host.record(state.z, dz, norm, accepted=True)
            state.norms.append(norm)
            host.stepped()
            if host.natural_stop(target):
                return
        if state.z < target - slack:
            msg = (
                f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                f"cannot reach z = {target:.6g} m"
            )
            raise BackendError(msg)
