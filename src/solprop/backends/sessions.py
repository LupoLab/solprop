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

:func:`start_group` is the door for a group of compatible members (SEP 47 section
6.7), through the same shape: the group's first member prepares the group,
``members[0].prepare_group(members, problems, requests).start(starts, ...)``, so a
backend chooses its group and nothing here names one either. The native backend's
(``solprop_native.group.NativePreparedGroup``) opens the native group session
(``solprop_native.group.NativeGroupSession``, one native session per member), on a CPU
or a device build, which refuses by name what it does not run
(``solprop_native.group.refusal``); the reference's opens its own group
(:class:`solprop.backends.host.group_session.HostGroupSession`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from solprop.backends.session import GroupSession, RunSession, Start, check_group
from solprop.backends.session_core import AttemptListener
from solprop.log import StopFlag
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # The scripted drivers of the test suite stand in for a backend.
    from solprop.backends.protocol import Backend

__all__ = ["start_group", "start_session"]


def start_session(  # noqa: PLR0913 - the session context
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


def start_group(
    members: Sequence[Backend],
    problems: Sequence[PropagationProblem],
    requests: Sequence[RunRequest],
    starts: Sequence[Start],
    *,
    stop: StopFlag | None = None,
) -> GroupSession:
    """Start the group that advances compatible members through one fixed schedule.

    Parameters
    ----------
    members : collections.abc.Sequence of solprop.backends.protocol.Backend
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
        The group the backend's prepared group opens
        (:meth:`~solprop.backends.protocol.Backend.prepare_group`): the reference's
        own group for reference members, and the native group session for native
        members.

    Raises
    ------
    solprop.backends.session.SessionError
        If the sequences differ in length or are empty, or if a start is not a fresh
        entrance.
    solprop.exceptions.BackendError
        If the backend runs no group, or refuses this one.
    """
    check_group(len(members), problems, requests, starts)
    prepared = members[0].prepare_group(members, problems, requests)
    return prepared.start(starts, stop=stop)
