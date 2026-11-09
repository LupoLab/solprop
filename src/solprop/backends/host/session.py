# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's run session (SEP 47 sections 5 and 6.6).

:class:`HostSession` is the :class:`~solprop.backends.session.RunSession` that serves
every ``numpy_ref`` run, in both precisions: the launch-before-monitor order, the plane
schedule, the snapshot latch, bounded yields, chunked attempts, exports and the terminal
outcome with its checkpoint eligibility, over the reference's own driver
(:class:`solprop.backends.host.driver.HostDriver`). It imports nothing of the inherited
driver the native bridge still runs on.

**A recorded exception.** The reference otherwise shares only stateless numerical
modules with the native bridge, but for phase 2 its lifecycle is the transitional
session's (:class:`solprop.backends.driver_session.DriverSession`), reused by
subclassing: it drives any driver through the backend protocol and the halt and yield
hooks a host driver keeps (``request_halt``, ``clear_halt``, ``set_halt_check``,
``poll_halt`` and the latch ``_halted``). The exception is removed when a later step of
SEP 47 restructures this session into the reference's own.
"""

from __future__ import annotations

from collections.abc import Callable

from solprop.backends.driver_session import AttemptListener, DriverSession
from solprop.backends.host.driver import HostDriver
from solprop.backends.protocol import Backend
from solprop.backends.session import Start
from solprop.exceptions import BackendError
from solprop.log import StopFlag
from solprop.translate.problem import KernelProblem
from solprop.translate.run_request import RunRequest

__all__ = ["HostSession"]


class HostSession(DriverSession):
    """The reference's :class:`~solprop.backends.session.RunSession`.

    Parameters
    ----------
    driver : solprop.backends.protocol.Backend
        The reference backend (a :class:`~solprop.backends.host.driver.HostDriver`),
        freshly constructed for `problem`, holding no state yet unless `start` is
        ``None``. The session owns it from here and closes it with itself when `owns`
        is set.
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start or None
        Where the run starts; ``None`` when the caller already loaded the driver and
        the run starts at its first plane with the launch plane owed (transitional).
    listener : callable or None, optional
        Called with each attempt as it is made, for a progress reporter. It observes
        only; it cannot change the run.
    started : callable or None, optional
        Called with the start, the end and the plane count when propagation begins.
    stop : solprop.log.StopFlag or None, optional
        The run's stop token, polled before every trial.
    owns : bool, optional
        Whether closing the session closes the driver. A caller that reuses its driver
        for later runs (a runner, a scan) keeps it open.

    Raises
    ------
    solprop.exceptions.BackendError
        If `driver` is not a reference driver.
    """

    def __init__(  # noqa: PLR0913 - the session context
        self,
        driver: Backend,
        problem: KernelProblem,
        request: RunRequest,
        start: Start | None,
        *,
        listener: AttemptListener | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> None:
        """Refuse a driver that is not the reference's, and load the starting state."""
        if not isinstance(driver, HostDriver):
            msg = (
                f"the reference session serves a host driver, not "
                f"{type(driver).__name__}"
            )
            raise BackendError(msg)
        super().__init__(
            driver,
            problem,
            request,
            start,
            listener=listener,
            started=started,
            stop=stop,
            owns=owns,
        )
