# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a constructed backend through its own session and keep what it observed.

SEP 47 phase 5 retired the native backend's direct driver calls (``set_initial``,
``propagate_to``, ``observe``): every native run is the native session's
(:func:`~solprop.backends.sessions.start_session`). These tests hand a backend they
built to its session, drive it to the end with the conformance suite's
:func:`sessions.executors.drive`, and read the planes it observed, each exported in the
state's own convention (:data:`sessions.executors.IDENTITY`), which is what the
retired ``observe().require_field()`` returned.

The statistics of a loaded state, which the retired direct ``stats`` call read, are
the launch plane's: a session started with its launch plane owed observes them first
(:func:`launched`, :func:`launch_statistics`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sessions.executors import IDENTITY, Run, Started, drive, entrance
from solprop.backends.events import ObservationReady, Outcome, Terminal
from solprop.backends.session import Start
from solprop.backends.sessions import start_session

if TYPE_CHECKING:
    from solprop.backends.session import RunSession
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest


@dataclass(frozen=True)
class Landed:
    """What a session observed at its last plane, and how it got there.

    Attributes
    ----------
    run : sessions.executors.Run
        Everything the session reported.
    field : numpy.ndarray
        The last plane's state, in the internal convention, shape
        ``(n_modes, n_pol, n_omega)``.
    stats : dict
        The last plane's statistics.
    n_accepted, n_rejected : int
        The accepted and rejected attempts of the whole run.
    record : dict
        The session's execution record, read before it closed.
    ledger : dict or None
        The backend's memory ledger while the session held it, or ``None`` for a
        backend that keeps none.
    """

    run: Run
    field: np.ndarray
    stats: dict[str, np.ndarray]
    n_accepted: int
    n_rejected: int
    record: dict[str, object]
    ledger: dict[str, int] | None


def run_session(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    *,
    start: Start | None = None,
    noise: Any = None,
) -> Landed:
    """Drive `backend`'s session for one run to its end.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`; the caller closes it.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run's request.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    noise : solprop.noise.NoiseRealisation or None, optional
        The noise realisation of the default entrance.

    Returns
    -------
    Landed
        The run and its last plane.

    Raises
    ------
    BaseException
        The run's own failure, when it ended :attr:`~solprop.backends.events.Outcome.
        FAILED`, so that a failed run is never read as a result.
    """
    if start is None:
        start = entrance(problem)
        if noise is not None:
            start = Start(field=start.field, z=start.z, noise=noise)
    session = start_session(backend, problem, request, start, owns=False)
    try:
        run = drive(Started(session), export=IDENTITY)
        record = dict(session.execution_record())
        ledger = backend.memory_ledger() if hasattr(backend, "memory_ledger") else None
    finally:
        session.close()
    if run.terminal.outcome is Outcome.FAILED:
        assert run.terminal.error is not None
        raise run.terminal.error
    accepted = sum(int(np.count_nonzero(h.accepted)) for h in run.histories)
    return Landed(
        run=run,
        field=run.fields[-1],
        stats={
            name: np.asarray(value)
            for name, value in run.observations[-1].stats.items()
        },
        n_accepted=accepted,
        n_rejected=run.attempts - accepted,
        record=record,
        ledger=ledger,
    )


def _launch_observation(session: RunSession) -> ObservationReady:
    """Advance `session` to its first observation, the launch plane, unacknowledged.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        A session started with its launch plane owed.

    Returns
    -------
    solprop.backends.events.ObservationReady
        The launch plane's observation, its batch still pending.

    Raises
    ------
    BaseException
        The session's own failure, when it fails before observing.
    """
    while not session.finished:
        batch = session.advance(work_limit=1)
        for event in batch.events:
            if isinstance(event, Terminal) and event.error is not None:
                raise event.error
            if isinstance(event, ObservationReady):
                return event
        session.acknowledge(batch.receipt)
    msg = "the session observed no launch plane"
    raise AssertionError(msg)


@contextmanager
def launched(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    field: np.ndarray | None = None,
    *,
    start: Start | None = None,
) -> Iterator[tuple[RunSession, ObservationReady]]:
    """Yield `backend`'s own session and the observation of its launch plane.

    The session starts from `field` at the problem's entrance with the launch plane
    owed, and is closed on exit; its observation is pending, so its ticket exports the
    loaded state as the session holds it.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`; the caller closes it.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request, whose statistics the launch plane carries.
    field : numpy.ndarray or None, optional
        The loaded state; the problem's launch by default.
    start : solprop.backends.session.Start or None, optional
        The start itself, in place of `field`: a resumed plane's, or one with noise.

    Yields
    ------
    tuple
        The session and its launch observation.
    """
    if start is None:
        launch = problem.initial.field0 if field is None else field
        start = Start(field=np.array(launch), z=float(problem.initial.z))
    session = start_session(backend, problem, request, start, owns=False)
    try:
        yield session, _launch_observation(session)
    finally:
        session.close()


def launch_statistics(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    field: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return the statistics `backend`'s own session observes at its launch plane.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`; the caller closes it.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request, whose statistics are read.
    field : numpy.ndarray or None, optional
        The loaded state; the problem's launch by default.

    Returns
    -------
    dict
        The statistics of the loaded state, copied.
    """
    with launched(backend, problem, request, field) as (_, observation):
        return {name: np.array(value) for name, value in observation.stats.items()}


def launch_field(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    field: np.ndarray | None = None,
    *,
    start: Start | None = None,
) -> np.ndarray:
    """Return the state `backend`'s own session holds at its launch plane.

    The load crosses the state into the session, and the export of the launch plane
    back out, in the state's own convention: what the retired load and
    ``observe().require_field()`` did.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`; the caller closes it.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    field : numpy.ndarray or None, optional
        The loaded state; the problem's launch by default.
    start : solprop.backends.session.Start or None, optional
        The start itself, in place of `field`.

    Returns
    -------
    numpy.ndarray
        The held state, copied.
    """
    with launched(backend, problem, request, field, start=start) as (
        session,
        observation,
    ):
        return np.array(session.export(observation.ticket, IDENTITY))
