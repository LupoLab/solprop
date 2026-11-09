# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which run session serves a loaded backend (SEP 47 phase 2).

The NumPy reference runs in its own session
(:class:`solprop.backends.host.session.HostSession`, over
:class:`solprop.backends.host.driver.HostDriver`); every other backend -- the native
bridge, until its own session replaces it in phase 3 -- runs in the transitional
session over the inherited driver
(:class:`solprop.backends.driver_session.DriverSession`). :func:`start_session` chooses
by the driver's type, so the run layer has one door.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from solprop.backends.driver_session import AttemptListener, DriverSession
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.session import HostSession
from solprop.backends.session import Start
from solprop.log import StopFlag
from solprop.translate.problem import KernelProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # The scripted drivers of the test suite stand in for a backend.
    from solprop.backends.protocol import Backend

__all__ = ["start_session"]


def start_session(  # noqa: PLR0913 - the session context
    backend: Backend,
    problem: KernelProblem,
    request: RunRequest,
    start: Start | None,
    *,
    listener: AttemptListener | None = None,
    started: Callable[[float, float, int], None] | None = None,
    stop: StopFlag | None = None,
    owns: bool = True,
) -> DriverSession:
    """Start the session that serves `backend` for one run.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The loaded backend, freshly constructed for `problem`, holding no state yet
        unless `start` is ``None``.
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start or None
        Where the run starts; ``None`` when the caller already loaded the backend and
        the run starts at its first plane with the launch plane owed (transitional).
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
    solprop.backends.driver_session.DriverSession
        A :class:`~solprop.backends.host.session.HostSession` for a reference
        backend, or the transitional session for any other.
    """
    session = HostSession if isinstance(backend, HostDriver) else DriverSession
    return session(
        backend,
        problem,
        request,
        start,
        listener=listener,
        started=started,
        stop=stop,
        owns=owns,
    )
