# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s own constructor refuses the TDSE route."""

import pytest

from backends import test_tdse_march
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.exceptions import BackendError

march_problem = test_tdse_march.march_problem


def test_direct_numba_construction_refuses_tdse_before_allocating(march_problem):
    """The direct constructor enforces the same named refusal as the registry."""
    problem, grid = march_problem
    with pytest.raises(BackendError, match="numba_cpu does not execute TDSE"):
        NumbaCpuBackend(problem, grid=grid)
