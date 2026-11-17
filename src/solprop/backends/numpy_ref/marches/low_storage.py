# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's low-storage interaction-picture RK4 march (SEP 24, SEP 52).

:class:`HostLowStorageMarch` replays a prescribed schedule through a
:class:`~solprop.backends.engines.RungeKuttaEngine` with the documented destructive
trial: the reference's is :class:`solprop.backends.numpy_ref.engines.NumpyLowStorage`,
which holds the registers and makes the step's four passes. The engine owns the
in-flight flag and the finish; the march owns the committed position, in its own
:class:`~solprop.backends.numpy_ref.marches.host.MarchState`, and the count of committed
source evaluations. The method, the register form and the in-flight contract are those
of :mod:`solprop.backends.low_storage`'s module documentation.
"""

import logging
import math
from collections import deque
from typing import cast

from solprop._interrupts import committing
from solprop.backends.engines import RungeKuttaEngine
from solprop.backends.numpy_ref.marches.host import (
    MarchHost,
    MarchState,
    in_flight_refusal,
)
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import FixedSchedule, replay_tolerance

__all__ = ["HostLowStorageMarch"]

logger = logging.getLogger(__name__)


class HostLowStorageMarch:
    """Replay a schedule with the low-storage RK4IP march, accepting in place.

    It replays a prescribed schedule, which is the only thing it does (SEP 24 section
    6.2). Each step's trial overwrites the state; the accept finishes it (the hard band
    mask, and the split-step collar where chosen) and the march moves its position in
    the same commit, then records the step through its session.

    Parameters
    ----------
    engine : solprop.backends.engines.RungeKuttaEngine
        The engine.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.
    host : solprop.backends.numpy_ref.marches.host.MarchHost
        The session the march reports to.
    schedule : solprop.stepper.schedule.FixedSchedule
        The schedule it replays.
    proposal : float
        The proposal it keeps, m, which a replay never updates.
    lag : int, optional
        The lag of the controller whose norms the replay keeps.
    name : str, optional
        The backend's name, for messages.

    Attributes
    ----------
    state : solprop.backends.numpy_ref.marches.host.MarchState
        The march's position, proposal, cursor and accepted norms.
    source_evaluations : int
        Source evaluations of the committed steps since the state was last loaded,
        four a step: a physical count, never a logical response call (SEP 24 section
        5). A step that did not commit -- abandoned in flight, or diverged -- adds none.

    Notes
    -----
    The march has no error estimate to warn on. It logs a warning, once per call of
    :meth:`replay_to`, when an accepted step's stiffness indicator
    (:attr:`~solprop.backends.engines.AcceptRecord.stiffness`) passes RK4's
    negative-real stability interval
    (:data:`~solprop.stepper.low_storage.LOW_STORAGE_REAL_INTERVAL`): an explicitly
    integrated damping term the translation cannot bound in advance -- plasma loss --
    is then near the step's stability limit.
    """

    def __init__(
        self,
        engine: object,
        method: LowStorageRungeKutta,
        host: object,
        *,
        schedule: FixedSchedule,
        proposal: float,
        lag: int = 1,
        name: str = "numpy_ref",
    ) -> None:
        """Hold the engine; the march stands at the entrance."""
        self._engine = cast("RungeKuttaEngine", engine)
        self._method = method
        self._host = cast("MarchHost", host)
        self._schedule = schedule
        self._name = name
        self._warned = False
        # The step whose trial the engine left in flight, while it is.
        self._pending: tuple[float, float] | None = None
        self.state = MarchState(z=0.0, proposal=proposal, norms=deque(maxlen=lag + 2))
        self.source_evaluations = 0

    @property
    def engine(self) -> object:  # numpydoc ignore=RT01
        """Engine that holds the registers."""
        return self._engine

    @property
    def in_flight(self) -> tuple[float, float] | None:  # numpydoc ignore=RT01
        """The step a failure left in flight, its start and length, m; or ``None``."""
        return self._pending if self._engine.in_flight else None

    def load(self, z: float, proposal: float, cursor: int) -> None:
        """Start afresh at a loaded position; a step left in flight is cleared.

        Parameters
        ----------
        z : float
            The position, m.
        proposal : float
            The proposal, m.
        cursor : int
            The replay cursor at `z`.
        """
        self.state.load(z, proposal, cursor)
        self._pending = None
        # Its count is the loaded run's, as the Adams ledger is (SEP 22 4.7).
        self.source_evaluations = 0

    def refusal(self) -> str | None:
        """Return why the state cannot be read or advanced, or ``None``.

        Returns
        -------
        str or None
            The message of a step left in flight, or ``None``.
        """
        pending = self.in_flight
        return None if pending is None else in_flight_refusal(self._name, *pending)

    def advance(self, target: float) -> None:
        """Replay the schedule towards `target`.

        Parameters
        ----------
        target : float
            The position to reach, m.
        """
        self.replay_to(self._schedule, target)

    def replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay the schedule's steps to `target`, mirroring the Adams replay.

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
        """
        host, state = self._host, self.state
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        self._warned = False
        while state.cursor < schedule.n_steps and state.z < target - slack:
            host.poll_stop()
            dz = float(schedule.steps[state.cursor])
            end = float(edges[state.cursor + 1])
            norm = self.replay_step(dz, end, slack)
            state.cursor += 1
            host.record(end, dz, norm, accepted=True)
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

    def replay_step(self, dz: float, end: float, slack: float) -> float:
        """Take one prescribed step from the march's position and accept it.

        The unit a replay advances by, with
        :meth:`solprop.backends.numpy_ref.marches.adams.HostAdamsMarch.replay_step`'s
        signature.

        Parameters
        ----------
        dz : float
            The step, m.
        end : float
            The schedule's edge it ends on, m.
        slack : float
            How far an edge may sit from a target it lands on, m; unused, since the
            march restarts nothing at a kink.

        Returns
        -------
        float
            The norm the history records: 0.0, "not estimated".

        Raises
        ------
        BackendError
            If an earlier step was abandoned in flight.
        ConvergenceError
            If the accepted state is not finite: the field has diverged, and a replay
            cannot reject the step. Raised in flight, so the state is lost with it.
        """
        del slack
        refusal = self.refusal()
        if refusal is not None:
            raise BackendError(refusal)
        state = self.state
        z = state.z
        self._pending = (z, dz)
        # The host passes are Python calls a signal may interrupt (the engine is not
        # atomic): an interrupted step stays in flight, and only the move commits.
        squared = self._engine.trial(z, dz)
        if not math.isfinite(squared):
            msg = (
                f"{self._name}: a replayed {self._method.name} step of {dz:.4g} m "
                f"from z = {z:.6g} m ended in a non-finite state: the field has "
                "diverged. A replayed schedule takes its steps as given and cannot "
                "reject one, so a step beyond an explicit stability limit -- the "
                "in-stage collars' -- grows until it overflows; the translation "
                "refuses a declared step past the collars' limit. The state was "
                "accepted in place and is lost; take shorter steps"
            )
            raise ConvergenceError(msg)
        with committing():
            record = self._engine.accept(dz)
            state.z = end
            self._pending = None
            self.source_evaluations += self._method.evaluations
        self._host.absorb(record)
        self._check_stiffness(z, dz, record.stiffness)
        return 0.0

    def _check_stiffness(self, z: float, dz: float, stiffness: float) -> None:
        """Warn, once per replay, when a step's stiffness passes RK4's interval.

        - ``z``: where the step started, m.
        - ``dz``: the step, m.
        """
        interval = self._method.real_stability_interval
        if self._warned or not stiffness > interval:
            return
        self._warned = True
        logger.warning(
            "%s: the %s march's stiffness indicator h rho = %.3g passed RK4's "
            "stability interval %.4g on the step of %.4g m from z = %.6g m; an "
            "explicitly integrated damping term (plasma loss, an in-stage collar) is "
            "near the step's stability limit and may make the replay diverge. Take "
            "shorter steps",
            self._name,
            self._method.name,
            stiffness,
            interval,
            dz,
            z,
        )
