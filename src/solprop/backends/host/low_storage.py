# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's low-storage interaction-picture RK4 march (SEP 24, SEP 47).

:class:`HostLowStorageMarch` replays a prescribed schedule and accepts each step in
place, through the host engine (:class:`solprop.backends.low_storage.HostLowStorage`),
which holds the registers and makes the step's four passes. It is the NumPy
reference's own since SEP 47 phase 2, sharing only the stateless engine; the native
session replays the march natively. The method, the register form and the in-flight
contract are those of :mod:`solprop.backends.low_storage`'s module documentation.
"""

import logging
import math
from typing import Any

from solprop._interrupts import committing
from solprop.backends.host.driver import HostDriver, replay_tolerance
from solprop.backends.low_storage import HostLowStorage
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import FixedSchedule

__all__ = ["HostLowStorageMarch"]

logger = logging.getLogger(__name__)


class HostLowStorageMarch:
    """Replay a schedule with the low-storage RK4IP march, accepting in place.

    Follows :class:`solprop.backends.host.adams.HostAdamsMarch`: it owns its registers
    through an engine and replays a prescribed schedule, which is the only thing it
    does (SEP 24 section 6.2). Each step calls the engine, applies the backend's
    ``_finish`` (the hard band mask, and the split-step collar where chosen) and moves
    the position in one commit, then records the step through the backend.

    Parameters
    ----------
    backend : solprop.backends.host.driver.HostDriver
        The backend whose state is marched; it owns the step history and the
        statistics rows.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.

    Attributes
    ----------
    source_evaluations : int
        Source evaluations of the committed steps since the state was last loaded
        (``HostDriver.set_initial``, which sets it to zero), four a step: a physical
        count, never a logical response call (SEP 24 section 5). A step that did not
        commit -- abandoned in flight, or diverged -- adds none.

    Notes
    -----
    The march has no error estimate to warn on. It logs a warning, once per call of
    :meth:`replay_to`, when the engine's stiffness indicator passes RK4's
    negative-real stability interval
    (:data:`~solprop.stepper.low_storage.LOW_STORAGE_REAL_INTERVAL`): an explicitly
    integrated damping term the translation cannot bound in advance -- plasma loss --
    is then near the step's stability limit.
    """

    def __init__(self, backend: HostDriver[Any], method: LowStorageRungeKutta) -> None:
        """Hold the backend and its engine."""
        self._backend = backend
        self._method = method
        self._warned = False
        # The host engine is the oracle (SEP 24 section 6.5).
        self._engine = HostLowStorage(backend, method)
        self.source_evaluations = 0

    @property
    def engine(self) -> HostLowStorage:  # numpydoc ignore=RT01
        """Engine that holds the registers."""
        return self._engine

    def reset(self) -> None:
        """Forget what the engine holds: the state was loaded outside the march."""
        self._engine.reset()

    def close(self) -> None:
        """Release the engine's registers and table, with the backend's buffers."""
        self._engine.close()

    def snapshot(self) -> None:
        """Return nothing for a resume to continue: the march keeps no history.

        A resume at a schedule edge needs no history and no recycled stage
        (SEP 24 section 5), as under ``RungeKutta()``.
        """
        return

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
        backend = self._backend
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        self._warned = False
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

        The unit a lockstep batch advances each member by
        (:class:`solprop.backends.host.group.HostGroup`), with
        :meth:`solprop.backends.host.adams.HostAdamsMarch.replay_step`'s signature.

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
        backend = self._backend
        backend._refuse_in_flight()
        z = backend._z
        backend._in_flight = (z, dz)
        # The host passes are Python calls a signal may interrupt (the engine is not
        # atomic): an interrupted step stays in flight, and only the move commits.
        squared = self._engine.step(z, dz)
        finished = self._accept(z, dz, squared)
        with committing():
            self._move(finished, end)
        self._check_stiffness(z, dz)
        return 0.0

    def _move(self, finished: object, end: float) -> None:
        """Make the finished state the backend's at the step's end; clear the flag.

        - ``end``: the schedule's edge the step ends on, m.
        """
        backend = self._backend
        backend._state = finished
        backend._z = end
        backend._in_flight = None
        self.source_evaluations += self._method.evaluations

    def _accept(self, z: float, dz: float, squared: float) -> object:
        """Refuse a diverged step, or apply the backend's finish to the state.

        - ``z``: where the step started, m.
        - ``dz``: the step, m.
        """
        backend = self._backend
        if not math.isfinite(squared):
            msg = (
                f"{backend.name}: a replayed {self._method.name} step of {dz:.4g} m "
                f"from z = {z:.6g} m ended in a non-finite state: the field has "
                "diverged. A replayed schedule takes its steps as given and cannot "
                "reject one, so a step beyond an explicit stability limit -- the "
                "in-stage collars' -- grows until it overflows; the translation "
                "refuses a declared step past the collars' limit. The state was "
                "accepted in place and is lost; take shorter steps"
            )
            raise ConvergenceError(msg)
        return backend._finish(backend._state, dz)

    def _check_stiffness(self, z: float, dz: float) -> None:
        """Warn, once per replay, when a step's stiffness passes RK4's interval.

        - ``z``: where the step started, m.
        - ``dz``: the step, m.
        """
        stiffness = self._engine.stiffness
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
            self._backend.name,
            self._method.name,
            stiffness,
            interval,
            dz,
            z,
        )
