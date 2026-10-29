# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march's statistics rows are the Python march's (SEP 39 4.8).

A native call returns to the host at every accepted step on which a periodic row falls
due, and books the steps between as skipped (``StepDriver._skip_step``): the rows are
taken at the same steps, from the same states, as the Python march takes them.
"""

import dataclasses

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

NativeBackend = pytest.importorskip("solprop_native").NativeBackend


def rows(march: str):
    """Return each plane's statistics rows on `march`, rows every third step."""
    simulation = ROUTES["gnlse"](None)
    simulation.settings = dataclasses.replace(
        simulation.settings, integrator=Multistep()
    )
    problem = dataclasses.replace(
        simulation.translate(),
        stats_spec=StatsSpec(names=("energy",), periodic=("energy",), period=3),
    )
    backend = NativeBackend(problem, grid=simulation.grid, adams_march=march)
    try:
        assert backend._adams.native == (march == "native")
        backend.set_initial(problem.initial.field0, None)
        found = []
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
            found.append(backend.observe().step_stats)
        return found
    finally:
        backend.close()


def test_rows_every_third_step_are_the_python_march_s():
    native, host = rows("native"), rows("host")
    assert sum(one.n_rows for one in native) > 0
    for found, expected in zip(native, host, strict=True):
        np.testing.assert_array_equal(found.step, expected.step)
        np.testing.assert_allclose(found.z, expected.z, rtol=1e-12, atol=0.0)
        # The same steps on one build, so the states agree to rounding (SEP 25).
        np.testing.assert_allclose(
            found.values["energy"], expected.values["energy"], rtol=1e-12
        )
