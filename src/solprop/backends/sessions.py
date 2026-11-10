# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run layer's one door to a backend's sessions (SEP 47 phases 2 and 3).

:func:`start_session` starts a run through the backend's own preparation shape (SEP
47 section 5.2), ``backend.prepare(problem, request).start(start, ...)``, so a backend
chooses its session and nothing here names one. The NumPy reference's prepared
execution opens its own session
(:class:`solprop.backends.host.session.HostSession`, over
:class:`solprop.backends.host.driver.HostDriver`); a native backend's
(``solprop_native.session.NativePrepared``) opens the native execution session
(``solprop_native.session.NativeSession``) whenever it admits the run, before the
start is loaded -- on a device build, SEP 47 phase 3's representative device path
(``solprop_native.session.device_refusal``) -- and the transitional session over the
inherited driver (:class:`solprop.backends.driver_session.DriverSession`) for what it
refuses by name, the reason logged at debug level; any other driver's opens the
transitional one.

Groups use the same ownership boundary: :func:`start_group` validates the common
inputs and delegates to the first member's group factory. Each backend selects its
own implementation and refuses unsupported groups before consuming their starts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from solprop.backends.driver_session import AttemptListener
from solprop.backends.session import (
    GroupSession,
    RunSession,
    SessionError,
    Start,
    check_group,
)
from solprop.log import StopFlag
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # The scripted drivers of the test suite stand in for a backend.
    from solprop.backends.protocol import ExecutionBackend

__all__ = ["start_group", "start_session"]


def start_session(  # noqa: PLR0913 - the session context
    backend: ExecutionBackend,
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
    backend : solprop.backends.protocol.ExecutionBackend
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
        (``solprop_native.session.NativeSession``) for a native backend whenever
        it admits the run; a :class:`~solprop.backends.host.session.HostSession` for
        a reference backend; otherwise the transitional session over the legacy
        driver, which serves what the native session does not run yet (SEP 47 Part
        II).
    """
    prepared = backend.prepare(problem, request)
    return prepared.start(
        start, listener=listener, started=started, stop=stop, owns=owns
    )


def start_group(
    members: Sequence[ExecutionBackend],
    problems: Sequence[PropagationProblem],
    requests: Sequence[RunRequest],
    starts: Sequence[Start],
    *,
    stop: StopFlag | None = None,
) -> GroupSession:
    """Start the group that advances compatible members through one fixed schedule.

    Parameters
    ----------
    members : collections.abc.Sequence of solprop.backends.protocol.ExecutionBackend
        One constructed backend per member, all of one backend; the group owns them
        once constructed.
    problems : collections.abc.Sequence of solprop.translate.problem.PropagationProblem
        Each member's independently translated problem.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request, with its own fresh boundary monitor.
    starts : collections.abc.Sequence of solprop.backends.session.Start
        Each member's entrance.
    stop : solprop.log.StopFlag or None, optional
        A group-wide stop token.

    Returns
    -------
    solprop.backends.session.GroupSession
        The started group: the native group session for native members it admits,
        the transitional group otherwise.

    Raises
    ------
    solprop.backends.session.SessionError
        If the sequences differ in length or are empty, or if a start is not a fresh
        entrance.
    """
    check_group(len(members), problems, requests, starts)
    if len({id(member) for member in members}) != len(members):
        raise SessionError("a session group requires distinct member owners")
    if any(member.name != members[0].name for member in members):
        raise SessionError("a session group refuses mixed backends")
    return members[0].start_group(members, problems, requests, starts, stop=stop)
