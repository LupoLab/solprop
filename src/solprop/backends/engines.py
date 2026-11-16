# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The engine protocols: what a march asks of the state it advances (SEP 52 §5.2).

A march owns its controller, landing, history and call ledger; an engine does every
state-sized operation a march asks of the workspace that holds the state. The two
protocols here mirror the native marches' C++ interfaces one-to-one, every argument
included: :class:`RungeKuttaEngine` is ``rk::Engine``
(``native/include/solprop_native/rk_march.hpp``), which the low-storage step implements
too, and :class:`AdamsEngine` is ``march::Engine``
(``native/include/solprop_native/adams_march.hpp``). The reference's engines are
:class:`~solprop.backends.numpy_ref.engines.NumpyRungeKutta`, the low-storage step
:class:`~solprop.backends.numpy_ref.engines.NumpyLowStorage` and the Adams history
:class:`~solprop.backends.numpy_ref.multistep.HostHistory`.

**The commit contract.**

- **The Runge-Kutta trial** keeps its stages and end state in the workspace. It commits
  nothing until ``accept``, and a rejected trial leaves the committed state as it was.
- **The low-storage step is the same engine, with one documented difference.** Its
  trial overwrites the state, the register ``A``. From the start of ``trial`` until
  ``accept`` returns, ``in_flight`` is true. An exception or a non-finite result in that
  interval leaves no committed state at that position, so the session ends the run
  failed with no checkpoint there. The engine owns the in-flight flag and the finish;
  the march owns the committed position. The stiffness diagnostic is the engine's, a
  field of its :class:`AcceptRecord` (NaN where the engine does not form it). The march
  reads it from there, and the session reads the march's state, never a private field.
- **The Adams accept** receives the retained history entries, oldest first, which the
  interaction formulation's transport uses.
- **step_carrier** returns the last trial's carrier rotations. The march stores them in
  the accepted step's record, and the session's dense observation applies them to the
  interpolation weights.
- **The dense hold, the dense observation and the snapshot rows** are the session's; no
  engine holds, restores or observes.
- **The TDSE reservation and the trial's outcome** are the engine's, as
  ``begin_attempt`` and ``end_attempt``.
- **The split collar's sums.** An accepted step's finish computes them in the
  workspace, from the masked field before the split, and returns them in its
  :class:`AcceptRecord`; the session accumulates and drains them. The removal adds over
  every accepted step since the previous check, and the next boundary reading drains
  it; the found occupancies are those of the latest accepted step; a rejected trial, a
  dense observation and a probe trial contribute nothing, because none of them finishes
  a step; loading a state empties the accumulator, and attaching a monitor, at a launch
  or a resume, starts it from zero.

A march holds its own state and touches no other object's fields.
"""

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from jaxtyping import Float

from solprop.backends.adams_core import Stencil

__all__ = [
    "AcceptRecord",
    "AdamsEngine",
    "AttemptOutcome",
    "BoundaryReading",
    "HistoryEntry",
    "RungeKuttaEngine",
]


class AttemptOutcome(enum.IntEnum):
    """How an attempt ended, for a host source's reservation (``HostOutcome``, SEP 49).

    Attributes
    ----------
    REJECTED : int
        The trial was rejected.
    ACCEPTED : int
        The trial was accepted and committed.
    ABANDONED : int
        The attempt ended without a decision: a failure or a stop.
    """

    REJECTED = 0
    ACCEPTED = 1
    ABANDONED = 2


@dataclass(frozen=True)
class HistoryEntry:
    """One entry of an Adams march's history (``march::Entry``).

    Attributes
    ----------
    z : float
        Where its sample was evaluated, m.
    identity : {"accepted", "predicted"}
        Whether the sample is the accepted state's, or the predicted state's that PEC
        stores in its place.
    row : int
        The engine's history row that holds it.
    """

    z: float
    identity: Literal["accepted", "predicted"]
    row: int


@dataclass(frozen=True, eq=False)
class BoundaryReading:
    """The boundary monitor's reading of the committed state (``monitor_reading``).

    The state's own sums (:class:`solprop.monitor.BoundaryEnergies`, less the split
    collar's sums, which travel in each step's :class:`AcceptRecord`). Equality is by
    identity (``eq=False``) because it holds arrays.

    Attributes
    ----------
    total : float
        The state's energy, ``sum_t P(t)``, in the proxy's units of energy.
    regions : numpy.ndarray
        Shape ``(4,)``: the low guard's, the high guard's and the collar's weighted
        energies, and the collar's removal rate, energy per metre.
    occupancy : numpy.ndarray or None
        The state's energy per walk-off bucket of the split collar's crossing limit;
        ``None`` without one.
    """

    total: float
    regions: Float[np.ndarray, " 4"]
    occupancy: Float[np.ndarray, " n_buckets"] | None = None


@dataclass(frozen=True, eq=False)
class AcceptRecord:
    """What an accepted step's finish reports (SEP 52 §5.2).

    Equality is by identity (``eq=False``) because it holds an array.

    Attributes
    ----------
    stiffness : float
        The low-storage step's stiffness indicator ``h rho``, dimensionless (SEP 24
        section 6.3); NaN where the engine does not form it, on every other engine and
        on the native low-storage engine.
    removed : float
        What the split temporal collar took at this step, from the masked field before
        the split, in the proxy's units of energy; 0.0 under the in-stage scheme or
        without a collar.
    found : numpy.ndarray or None
        Shape ``(4,)``: the energy and the first three weighted energies of the power
        the step's split found, taken halfway through its factor; ``None`` under the
        in-stage scheme or without a collar.
    """

    stiffness: float = math.nan
    removed: float = 0.0
    found: Float[np.ndarray, " 4"] | None = None


class RungeKuttaEngine(Protocol):
    """What a Runge-Kutta march asks of its engine (``rk::Engine``).

    The embedded Runge-Kutta trial and, with the documented difference of the module's
    commit contract, the low-storage step.
    """

    def trial(self, z: float, dz: float) -> float:
        """Form a trial from `z` over `dz`; return its scaled error norm.

        The low-storage step overwrites the state, and returns 0.0 ("not estimated"),
        or a non-finite value when the field diverged.

        Parameters
        ----------
        z : float
            Where the trial starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The scaled error norm.
        """
        ...

    def accept(self, dz: float) -> AcceptRecord:
        """Adopt the trial: promote its recycled stage, then finish and commit it.

        Parameters
        ----------
        dz : float
            The accepted step, m.

        Returns
        -------
        AcceptRecord
            The finish's stiffness diagnostic and split collar's sums.
        """
        ...

    def extend(self, z: float, dz: float) -> None:
        """Hold the accepted trial from `z` over `dz` as a dense extension.

        Called before the trial is adopted, for the session's dense observation.

        Parameters
        ----------
        z : float
            Where the trial started, m.
        dz : float
            The step, m.
        """
        ...

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        BoundaryReading
            The reading.
        """
        ...

    @property
    def in_flight(self) -> bool:  # numpydoc ignore=RT01
        """Whether a step that overwrites the state in place was left partway."""
        ...


class AdamsEngine(Protocol):
    """What an Adams march asks of its engine (``march::Engine``).

    The engine keeps the history's rows where the state lives; the march keeps the
    entries (:class:`HistoryEntry`) and passes them in.
    """

    def start(self, z: float) -> None:
        """Evaluate the history's first sample at the accepted state at `z` into row 0.

        Parameters
        ----------
        z : float
            The accepted position, m.
        """
        ...

    def trial(
        self, z: float, dz: float, stencil: Stencil
    ) -> Float[np.ndarray, " n_orders"]:
        """Form a trial from `z` over `dz`; return its scored orders' norms.

        Parameters
        ----------
        z : float
            Where the trial starts, m.
        dz : float
            The step, m.
        stencil : solprop.backends.adams_core.Stencil
            The history's nodes and rows, newest first, and the orders scored.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order, the step's own first.
        """
        ...

    def accept(
        self, dz: float, free: int, stencil: Stencil, kept: Sequence[HistoryEntry]
    ) -> AcceptRecord:
        """Adopt the held trial; the new sample takes row `free`.

        Parameters
        ----------
        dz : float
            The accepted step, m.
        free : int
            The history row the new sample takes.
        stencil : solprop.backends.adams_core.Stencil
            The trial's stencil.
        kept : collections.abc.Sequence of HistoryEntry
            The history the step keeps, oldest first, which the interaction
            formulation transports.

        Returns
        -------
        AcceptRecord
            The finish's split collar's sums.
        """
        ...

    def chord_phase(self, z1: float, z2: float) -> float:
        """Return the carrier chord's largest phase over ``[z1, z2]``.

        Parameters
        ----------
        z1, z2 : float
            The interval, m.

        Returns
        -------
        float
            The phase, rad.
        """
        ...

    def step_carrier(self, order: int) -> Float[np.ndarray, " n_rotations"]:
        """Return the last trial's carrier rotations over its `order` newest rows.

        Parameters
        ----------
        order : int
            The rows the dense observation's formula reads.

        Returns
        -------
        numpy.ndarray
            The shift's imaginary part, then each row's rotation as its real and
            imaginary parts; empty without a carrier frame.
        """
        ...

    def begin_attempt(self, cost: int) -> None:
        """Start an attempt: a host source's reservation (SEP 49 section 6.2).

        Called after the budget check and before the attempt's first evaluation.

        Parameters
        ----------
        cost : int
            The attempt's logical response calls.
        """
        ...

    def end_attempt(self, outcome: AttemptOutcome) -> None:
        """End an attempt: a host source's outcome.

        Parameters
        ----------
        outcome : AttemptOutcome
            How it ended.
        """
        ...

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        BoundaryReading
            The reading.
        """
        ...
