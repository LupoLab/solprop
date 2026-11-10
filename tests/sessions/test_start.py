# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A session's fresh start, :meth:`~solprop.backends.session.Start.entrance`.

A session starts one trajectory (SEP 47 section 5.1); the entrance of a problem whose
initial state carries several members is refused rather than silently truncated.
"""

import dataclasses

import numpy as np
import pytest

from sessions.executors import soliton
from solprop.backends.session import Start
from solprop.exceptions import BackendError


def test_the_entrance_is_the_launch_state_with_its_plane_owed():
    problem = soliton()
    start = Start.entrance(problem)
    np.testing.assert_array_equal(start.field, problem.initial.field0[0])
    assert start.z == problem.initial.z
    assert start.plane_index == 0
    assert start.plane_pending


def test_the_entrance_of_a_problem_with_two_members_is_refused():
    problem = soliton()
    field0 = problem.initial.field0
    batched = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial, field0=np.concatenate([field0, field0])
        ),
    )
    with pytest.raises(BackendError, match="has 2 members"):
        Start.entrance(batched)
