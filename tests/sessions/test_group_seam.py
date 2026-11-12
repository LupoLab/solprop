# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A group starts through its backend's own ``prepare_group`` (SEP 47 section 6.7).

:func:`solprop.backends.sessions.start_group` asks the members' backend to prepare the
group, so a backend that runs no group refuses it there, before any member is loaded.
"""

from contextlib import ExitStack

import pytest

from sessions.test_group import member
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.backends.sessions import start_group
from solprop.exceptions import BackendError


def test_a_backend_that_runs_no_group_refuses_it_at_the_door(monkeypatch):
    def refused(self, members, problems, requests):
        raise BackendError("the reference runs no group")

    monkeypatch.setattr(NumpyRefBackend, "prepare_group", refused)
    simulations = [member(index, "numpy_ref") for index in range(2)]
    problems = [one.translate() for one in simulations]
    with ExitStack() as cleanup:
        owners = []
        for simulation in simulations:
            owner = select(
                "numpy_ref",
                simulation.translate(),
                simulation._run_request(),
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            )
            cleanup.callback(owner.close)
            owners.append(owner)
        with pytest.raises(BackendError, match="the reference runs no group"):
            start_group(
                owners,
                problems,
                [one._run_request() for one in simulations],
                [Start.entrance(problem) for problem in problems],
            )
