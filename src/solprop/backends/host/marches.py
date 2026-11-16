# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What every reference march holds, and what it asks of its session (SEP 52).

A march -- :class:`~solprop.backends.host.runge_kutta.RungeKuttaMarch`,
:class:`~solprop.backends.host.adams.HostAdamsMarch` or
:class:`~solprop.backends.host.low_storage.HostLowStorageMarch` -- keeps its
position, proposal, replay cursor and accepted norms in a :class:`MarchState` of its
own, hands every state-sized operation to its engine
(:mod:`solprop.backends.engines`) and asks the session that schedules it
(:class:`~solprop.backends.host.session.HostSession`) only through the narrow
:class:`MarchHost`: whether to stop, how far a step may go, and what to record when an
attempt ends. It touches no other object's fields; the session reads its state, and
holds the dense observation of a plane a step crossed (SEP 52 section 5.2).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from solprop.backends.engines import AcceptRecord

__all__ = ["MarchHost", "MarchState", "in_flight_refusal"]


@dataclass(eq=False)
class MarchState:
    """A march's position, proposal, replay cursor and accepted norms (SEP 45 4.2).

    Attributes
    ----------
    z : float
        The committed state's position, m.
    proposal : float
        The controller's proposal, m: its own state, never the executed step.
    cursor : int
        A replay's next step: how many steps of the schedule are consumed.
    norms : collections.deque
        The accepted steps' norms the controller reads, oldest first, at most its
        lag and two more.
    """

    z: float
    proposal: float
    cursor: int = 0
    norms: deque[float] = field(default_factory=deque)

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
        self.z = z
        self.proposal = proposal
        self.cursor = cursor
        self.norms.clear()


class MarchHost(Protocol):
    """What a march asks of the session that schedules it.

    The reference's session implements it
    (:class:`~solprop.backends.host.session.HostSession`); a decision-trace harness
    scripts it.
    """

    @property
    def collar_quiet(self) -> bool:  # numpydoc ignore=RT01
        """Whether the stages leave a quiet in-stage temporal collar out (SEP 21)."""
        ...

    def poll_stop(self) -> None:
        """Raise when the run's stop flag is set; called before every attempt."""
        ...

    def cap(self, proposal: float, z: float) -> float:
        """Return a proposal held to the boundary monitor's and the edge's limits.

        Parameters
        ----------
        proposal : float
            The controller's proposal, m.
        z : float
            Where the step starts, m.

        Returns
        -------
        float
            The proposal, or the quiet collar's interval, the transverse edge's read
            spacing or the split collar's crossing limit when shorter, m.
        """
        ...

    def record(self, end: float, dz: float, norm: float, accepted: bool) -> None:
        """Record one attempt in the history since the last observation.

        Parameters
        ----------
        end : float
            The attempt's end, m.
        dz : float
            Its length, m.
        norm : float
            The scaled error norm as executed.
        accepted : bool
            Whether it was accepted.
        """
        ...

    def absorb(self, record: AcceptRecord) -> None:
        """Accumulate an accepted step's split collar's sums (SEP 52 section 5.2).

        Parameters
        ----------
        record : solprop.backends.engines.AcceptRecord
            What the step's finish reported.
        """
        ...

    def stepped(self) -> None:
        """Account the step just committed: the monitor, the edge and the row."""
        ...

    def natural_stop(self, target: float) -> bool:
        """Count an accepted step; return whether to stop at its end.

        Parameters
        ----------
        target : float
            The position the call is stepping to, m.

        Returns
        -------
        bool
            Whether a step plane, a halt or a warning barrier stops the march here.
        """
        ...


def in_flight_refusal(name: str, z: float, dz: float) -> str:
    """Return why a state a low-storage step left in flight cannot be used.

    The step accepts in place (SEP 24 section 6.2): from its start to its commit the
    state is neither the step's start nor its end, so an exception or a signal between
    them leaves nothing a reader may use and nothing a march may continue from.

    Parameters
    ----------
    name : str
        The backend's name.
    z : float
        Where the step started, m.
    dz : float
        The step, m.

    Returns
    -------
    str
        The message.
    """
    return (
        f"{name}: the low-storage step of {dz:.6g} m from z = {z:.6g} m was "
        "abandoned in flight, and the state it was accepting in place holds "
        "neither end of the step; start a new session from the last saved plane "
        "before observing or propagating again"
    )
