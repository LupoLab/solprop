# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a backend through its own session and keep what it observed.

SEP 47 phase 5 takes the native backend off the legacy direct API (``set_initial``,
``propagate_to``, ``observe``, ``stats``): a run starts through
:func:`solprop.backends.sessions.start_session`, which opens the native session for a
native backend and the reference's own session for ``numpy_ref``. :func:`run` drives
one to its end, or to a named plane, and returns each observation's field (exported as
the session holds it, the internal FFT convention ``observe`` returned), statistics,
summary and attempts, with the execution record, so that a test which compared two
backends' planes compares them the same way through either session.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from solprop.backends.events import (
    ObservationKind,
    ObservationReady,
    Outcome,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.session import Start
from solprop.backends.sessions import start_session

IDENTITY = ExportRecipe(power=False, optics=False)
"""The state as the session holds it, in the internal FFT convention."""


@dataclasses.dataclass(frozen=True)
class Plane:
    """One observation of a run.

    Attributes
    ----------
    z : float
        Its position, m.
    kind : solprop.backends.events.ObservationKind
        Why it was observed.
    plane_index : int
        The save plane it answers; ``-1`` for the launch.
    field : numpy.ndarray
        Its field, exported with :data:`IDENTITY`.
    stats : dict
        Its statistics by catalogue name.
    summary : solprop.backends.protocol.StepSummary
        The cost and tolerance of the propagation that reached it.
    history : solprop.backends.protocol.BackendHistory
        Every attempt since the previous observation.
    rows : solprop.backends.protocol.StepStats
        The periodic rows since the previous observation.
    record : dict or None
        The session's execution record here, when asked for.
    """

    z: float
    kind: ObservationKind
    plane_index: int
    field: np.ndarray
    stats: dict[str, np.ndarray]
    summary: StepSummary
    history: BackendHistory
    rows: StepStats
    record: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class Ran:
    """What one session observed.

    Attributes
    ----------
    launch : Plane or None
        The launch plane, when the start owed it.
    planes : tuple of Plane
        Every later observation, in order.
    terminal : solprop.backends.events.Terminal or None
        The terminal event; ``None`` for a run stopped at a named plane.
    record : dict
        The execution record when the run stopped.
    """

    launch: Plane | None
    planes: tuple[Plane, ...]
    terminal: Terminal | None
    record: dict[str, Any]

    @property
    def final(self) -> Plane:
        """Return the last observation after the launch."""
        return self.planes[-1]

    @property
    def history(self) -> BackendHistory:
        """Return every attempt up to the last observation."""
        observed = [self.launch] if self.launch is not None else []
        return join_attempts(
            (one.history, one.rows) for one in (*observed, *self.planes)
        )[0]

    @property
    def n_steps(self) -> int:
        """Return the attempts up to the last observation."""
        return self.history.n_steps

    @property
    def met_tolerance(self) -> bool:
        """Return whether every propagation between observations met its tolerance."""
        return all(one.summary.met_tolerance for one in self.planes)


def run(  # noqa: PLR0913 - the session context and what to keep
    owner: Any,
    problem: Any,
    request: Any,
    start: Start | None = None,
    *,
    owns: bool = True,
    until: float | None = None,
    records: bool = False,
    raises: bool = True,
) -> Ran:
    """Drive `owner` through its own session and keep every observation.

    Parameters
    ----------
    owner : solprop.backends.protocol.Backend
        The backend, constructed for `problem`.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    owns : bool, optional
        Whether closing the session closes `owner`.
    until : float or None, optional
        Stop after the first observation at this position, m (the launch included).
    records : bool, optional
        Keep the execution record at every observation.
    raises : bool, optional
        Raise a failed run's error instead of returning it in the terminal.

    Returns
    -------
    Ran
        What the session observed.
    """
    session = start_session(
        owner,
        problem,
        request,
        Start.entrance(problem) if start is None else start,
        owns=owns,
    )
    launch: Plane | None = None
    planes: list[Plane] = []
    chunks: list[tuple[BackendHistory | None, StepStats | None]] = []
    terminal = None
    record: dict[str, Any] = {}
    try:
        while not session.finished:
            batch = session.advance()
            stop = False
            for event in batch.events:
                if (
                    isinstance(event, ServiceRequired)
                    and event.kind is ServiceKind.WARNING
                ):
                    emit_warning(event)
                if isinstance(event, Terminal):
                    terminal = event
                    continue
                history = getattr(event, "history", None)
                if history is not None:
                    chunks.append((history, getattr(event, "step_stats", None)))
                if not isinstance(event, ObservationReady):
                    continue
                attempts, rows = join_attempts(chunks)
                chunks = []
                plane = Plane(
                    z=float(event.z),
                    kind=event.kind,
                    plane_index=event.plane_index,
                    field=np.array(session.export(event.ticket, IDENTITY)),
                    stats={
                        name: np.array(value) for name, value in event.stats.items()
                    },
                    summary=event.summary,
                    history=attempts,
                    rows=rows,
                    record=dict(session.execution_record()) if records else None,
                )
                if event.kind is ObservationKind.LAUNCH:
                    launch = plane
                else:
                    planes.append(plane)
                stop = stop or (until is not None and plane.z == until)
            record = dict(session.execution_record())
            session.acknowledge(batch.receipt)
            if stop:
                break
    finally:
        session.close()
    if (
        raises
        and terminal is not None
        and terminal.outcome is Outcome.FAILED
        and terminal.error is not None
    ):
        raise terminal.error
    return Ran(launch, tuple(planes), terminal, record)


def final_field(
    owner: Any, problem: Any, request: Any, z: float | None = None, **options: Any
) -> np.ndarray:
    """Return the last observed field of a session run, checking where it landed.

    Parameters
    ----------
    owner, problem, request
        As :func:`run` takes them.
    z : float or None, optional
        Where the last observation must be, m; unchecked by default.
    **options
        Passed to :func:`run`.

    Returns
    -------
    numpy.ndarray
        The field, ``(n_modes, n_pol, n_omega)``.
    """
    ran = run(owner, problem, request, **options)
    if z is not None:
        assert ran.final.z == z, (ran.final.z, z)
    return ran.final.field
