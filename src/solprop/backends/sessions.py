# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run layer's one door to a backend's sessions (SEP 47 phases 2 to 5).

:func:`start_session` starts a run through the backend's own preparation shape (SEP
47 section 5.2), ``backend.prepare(problem, request).start(start, ...)``, so a backend
chooses its session and nothing here names one. The NumPy reference's prepared
execution opens its own session
(:class:`solprop.backends.host.session.HostSession`, over
:class:`solprop.backends.host.driver.HostDriver`); a native backend's
(``solprop_native.session.NativePrepared``) opens the native execution session
(``solprop_native.session.NativeSession``), on a CPU or a device build alike, which
refuses by name, before the start is loaded, what it does not run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from solprop.backends.session import RunSession, Start
from solprop.backends.session_core import AttemptListener
from solprop.log import StopFlag
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # The scripted drivers of the test suite stand in for a backend.
    from solprop.backends.protocol import Backend

__all__ = ["start_session"]


def start_session(
    backend: Backend,
    problem: PropagationProblem,
    request: RunRequest,
    start: Start,
    *,
    listener: AttemptListener | None = None,
    started: Callable[[float, float, int], None] | None = None,
    stop: StopFlag | None = None,
    owns: bool = True,
) -> RunSession:
    """Start the session that serves `backend` for one run.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`. A backend reused from an earlier
        run is loaded again from `start`, which restores everything a march reads.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start
        Where the run starts; the session takes its field.
    listener : callable or None, optional
        Called with each attempt as it is made, for a progress reporter.
    started : callable or None, optional
        Called with the start, the end and the plane count when propagation begins.
    stop : solprop.log.StopFlag or None, optional
        The run's stop token.
    owns : bool, optional
        Whether closing the session closes the backend.

    Returns
    -------
    solprop.backends.session.RunSession
        The session the backend's prepared execution opens: the native session
        (``solprop_native.session.NativeSession``) for a native backend, and a
        :class:`~solprop.backends.host.session.HostSession` for a reference
        backend.
    """
    prepared = backend.prepare(problem, request)
    return prepared.start(
        start, listener=listener, started=started, stop=stop, owns=owns
    )
