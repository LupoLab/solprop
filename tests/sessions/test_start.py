# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A session's fresh start, :meth:`~solprop.backends.session.Start.entrance`.

A session starts one trajectory (SEP 47 section 5.1); the entrance of a problem whose
initial state carries several members is refused rather than silently truncated.
"""

import numpy as np

from sessions.executors import soliton
from solprop.backends.session import Start


def test_the_entrance_is_the_launch_state_with_its_plane_owed():
    problem = soliton().problem
    start = Start.entrance(problem)
    np.testing.assert_array_equal(start.field, problem.initial.field0)
    assert start.z == problem.initial.z
    assert start.plane_index == 0
    assert start.plane_pending
