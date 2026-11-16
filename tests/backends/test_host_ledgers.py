# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A reference backend with no atomic service refuses atomic call ledgers.

A resumed start's ledgers are restored by the reference's session
(:class:`solprop.backends.host.session.HostSession`): the logical Adams counts every
Adams march keeps, and atomic counts only into a TDSE binding. A run without one is
refused them by name rather than silently dropping a budget a resume must honour.
"""

import pytest

from scripted_driver import HostScriptedDriver
from sessions.executors import soliton, window
from solprop.backends.session import Start
from solprop.exceptions import BackendError


def _start(problem, ledgers):
    return Start(
        field=problem.initial.field0, z=float(problem.initial.z), ledgers=ledgers
    )


def test_a_host_driver_without_an_atomic_service_refuses_atomic_ledgers():
    problem, request = soliton()
    backend = HostScriptedDriver(problem, request, grid=window())
    prepared = backend.prepare(problem, request)
    try:
        with pytest.raises(
            BackendError, match="atomic call ledger requires a TDSE binding"
        ):
            prepared.start(
                _start(problem, {"logical": {}, "atomic": {"attempted": 1}}),
                owns=False,
            )
        # Nothing atomic, and a march with no logical ledger: nothing to install.
        for ledgers in ({"logical": {}, "atomic": {}}, {"logical": {}}):
            prepared.start(_start(problem, ledgers), owns=False).close()
    finally:
        backend.close()
