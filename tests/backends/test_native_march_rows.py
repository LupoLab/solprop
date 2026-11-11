# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march's statistics rows are the reference's (SEP 39 4.8).

A native call returns to the host at every accepted step on which a periodic row falls
due, and books the steps between as skipped (``StepDriver._account_step``): the rows are
taken at every third accepted step, from the states there, as the NumPy reference's own
Adams march takes them.
"""

import dataclasses

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

NativeBackend = pytest.importorskip("solprop_native").NativeBackend


def rows(implementation: type):
    """Return each plane's rows and accepted ends, rows every third step."""
    simulation = ROUTES["gnlse"](None)
    simulation.settings = dataclasses.replace(
        simulation.settings, integrator=Multistep()
    )
    problem = simulation.translate()
    run_request = simulation._run_request().for_run(
        stats=StatsSpec(names=("energy",), periodic=("energy",), period=3)
    )
    backend = implementation(problem, run_request, grid=simulation.grid)
    try:
        if implementation is NativeBackend:
            assert backend._adams.native
        backend.set_initial(problem.initial.field0, None)
        found = []
        for plane in run_request.stepper.save_planes:
            backend.propagate_to(float(plane))
            observed = backend.observe()
            history = observed.step_history
            found.append((observed.step_stats, history.z[history.accepted]))
        return found
    finally:
        backend.close()


def test_rows_every_third_step_are_the_reference_s():
    native, reference = rows(NativeBackend), rows(NumpyRefBackend)
    assert sum(one.n_rows for one, _ in native) > 0
    for (found, ends), (expected, reference_ends) in zip(
        native, reference, strict=True
    ):
        # The two marches take the same steps (tests/backends/adams_traces.py holds
        # their decisions to one another's), each row at an accepted endpoint; the
        # proposals, formed from norms that agree to rounding, place the endpoints
        # to 1e-10 of the run's length.
        span = 1e-10 * float(ends[-1])
        np.testing.assert_allclose(ends, reference_ends, rtol=0.0, atol=span)
        assert np.isin(found.z, ends).all()
        np.testing.assert_array_equal(found.step, expected.step)
        np.testing.assert_allclose(found.z, expected.z, rtol=0.0, atol=span)
        # The reference takes the same steps, so the states agree to rounding (SEP 25).
        np.testing.assert_allclose(
            found.values["energy"], expected.values["energy"], rtol=1e-12
        )
