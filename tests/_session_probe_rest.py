# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a constructed backend through its execution session, for tests outside it.

SEP 47 phase 5: a native backend no longer offers the step driver's direct calls
(``set_initial``, ``propagate_to``, ``observe``, ``stats``); every run goes through
the session its prepared execution starts (:func:`solprop.backends.sessions.
start_session`). These are the few lines the driver, flow and data tests share to run a
backend they built to the end of a request and read what it observed: the session is
the reference's :class:`~solprop.backends.host.session.HostSession` on ``numpy_ref``
and the native session on ``native``, so one call serves both sides of a parity check.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sessions.executors import IDENTITY, Run, Started, drive
from solprop.backends.events import EventBatch, ObservationReady
from solprop.backends.export import ExportRecipe
from solprop.backends.session import RunSession, Start
from solprop.backends.sessions import start_session
from solprop.stats import StatsSpec
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest


def bare(request: RunRequest) -> RunRequest:
    """Return a request that observes no statistics and watches no boundary.

    What the step driver's direct calls ran: a march between planes, with no
    statistics evaluated and no monitor attached.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Returns
    -------
    solprop.translate.run_request.RunRequest
        The same method, control, itinerary and planes, without statistics or monitor.
    """
    return request.for_run(stats=StatsSpec())


def to_planes(request: RunRequest, planes: Sequence[float]) -> RunRequest:
    """Return a request whose run observes `planes` and ends at the last of them.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request.
    planes : sequence of float
        The save planes, m, increasing.

    Returns
    -------
    solprop.translate.run_request.RunRequest
        The request with these planes, its other policies kept.
    """
    stepper = dataclasses.replace(
        request.stepper, save_planes=np.asarray(planes, dtype=np.float64)
    )
    return request.with_stepper(stepper)


def run_session(  # noqa: PLR0913 - the session's start and its probes
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    start: Start | None = None,
    *,
    recipe: ExportRecipe | None = IDENTITY,
    between: Callable[[RunSession, EventBatch], None] | None = None,
    owns: bool = True,
) -> Run:
    """Drive `backend`'s session from `start` to its end; return what it observed.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    recipe : solprop.backends.export.ExportRecipe or None, optional
        The export of every observed field; the state as the session holds it (the
        lab frame, the FFT convention) by default, as the step driver observed it.
    between : callable or None, optional
        Called with the session and each batch before it is acknowledged.
    owns : bool, optional
        Whether the session closes the backend when it closes.

    Returns
    -------
    sessions.executors.Run
        The batches, the exported fields and the status; the session is closed.
    """
    session = start_session(
        backend,
        problem,
        request,
        Start.entrance(problem) if start is None else start,
        owns=owns,
    )
    try:
        return drive(Started(session), export=recipe, between=between)
    finally:
        session.close()


def final_field(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    start: Start | None = None,
) -> np.ndarray:
    """Return the state at the end of `request`'s run of `backend`, as a host copy.

    The run observes no statistics and watches no boundary (:func:`bare`).

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`; the session closes it.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.

    Returns
    -------
    numpy.ndarray
        The last plane's state, ``(n_modes, n_pol, n_omega)``, in the FFT convention.
    """
    run = run_session(backend, problem, bare(request), start)
    last = run.observations[-1]
    assert isinstance(last, ObservationReady)
    # The run's end; a replay lands on its schedule's last edge, the steps' sum, which
    # may differ from the end by the sum's rounding.
    end = float(request.stepper.z_end)
    assert math.isclose(last.z, end, rel_tol=1e-12), (last.z, end)
    return run.fields[-1]


def saved(run: Run) -> list[tuple[ObservationReady, np.ndarray]]:
    """Return a run's save-plane observations with their fields, the launch left out.

    Parameters
    ----------
    run : sessions.executors.Run
        The run, driven with an export recipe.

    Returns
    -------
    list
        ``(observation, field)`` for every observation of a save plane, in order.
    """
    return [
        (event, field)
        for event, field in zip(run.observations, run.fields, strict=True)
        if event.plane_index >= 0
    ]


class Snapshots:
    """Take the method history at every observation a session makes.

    Passed as :func:`run_session`'s ``between``; :attr:`taken` holds
    ``(observation, snapshot)`` in order, the snapshot ``None`` where the march holds no
    history there (a Runge-Kutta march, the launch, a dense plane).
    """

    def __init__(self) -> None:
        self.taken: list[tuple[ObservationReady, Any]] = []

    def __call__(self, session: RunSession, batch: EventBatch) -> None:
        """Take each observation's snapshot while its batch is pending.

        Parameters
        ----------
        session : solprop.backends.session.RunSession
            The session.
        batch : solprop.backends.events.EventBatch
            Its pending batch.
        """
        for event in batch.events:
            if isinstance(event, ObservationReady):
                self.taken.append((event, session.march_snapshot(event.ticket)))

    def at_plane(self, index: int) -> Any:
        """Return the snapshot at save plane `index`.

        Parameters
        ----------
        index : int
            The save plane's index into the request's planes.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            Its snapshot.
        """
        [found] = [
            snapshot for event, snapshot in self.taken if event.plane_index == index
        ]
        return found


def run_error(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    start: Start | None = None,
    *,
    owns: bool = True,
) -> BaseException | None:
    """Return why a run of `backend` failed: refused at its preparation or start, or
    its session's terminal error; ``None`` for a run that completed.

    The native backend refuses some runs by name before any session starts, where the
    reference's session starts and fails at its first evaluation; either is the run's
    failure.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    owns : bool, optional
        Whether the session closes the backend when it closes.

    Returns
    -------
    BaseException or None
        The failure.
    """
    try:
        run = run_session(backend, problem, request, start, recipe=None, owns=owns)
    except Exception as error:
        return error
    return run.terminal.error
