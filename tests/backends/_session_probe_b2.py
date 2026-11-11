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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sessions.executors import IDENTITY, Run, Started, drive, entrance
from solprop.backends.events import Outcome
from solprop.backends.session import Start
from solprop.backends.sessions import start_session

if TYPE_CHECKING:
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
    """

    run: Run
    field: np.ndarray
    stats: dict[str, np.ndarray]
    n_accepted: int
    n_rejected: int


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
    )
