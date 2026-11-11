# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march's statistics rows are the reference's (SEP 39 4.8).

The native session's Adams march takes a periodic row at every accepted step on which
one falls due: the rows are taken at every third accepted step, from the states there,
as the NumPy reference's own Adams march, in its own session, takes them.
"""

import dataclasses

import numpy as np
import pytest

from capability_routes import ROUTES
from sessions.executors import Started, drive, entrance
from solprop.backends import available
from solprop.backends.events import ObservationReady
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.sessions import start_session
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
        session = start_session(
            backend, problem, run_request, entrance(problem), owns=False
        )
        try:
            if implementation is NativeBackend:
                assert type(backend._resident).__name__ == "_ResidentAdams"
            run = drive(Started(session))
        finally:
            session.close()
    finally:
        backend.close()
    # Each save plane's rows and attempts since the previous one; the launch plane
    # has none.
    planes = [
        (rows, history.z[history.accepted])
        for event, history, rows in run.assembled()
        if isinstance(event, ObservationReady)
    ]
    assert len(planes) == len(run_request.stepper.save_planes) + 1
    return planes[1:]


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
