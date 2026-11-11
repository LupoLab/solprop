# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 3b: the Adams history resident in Kokkos's engine (ADR-043).

Kokkos builds its own engine (``NativeBackend._resident``) for either formulation on
the modal routes when a native session starts its Adams march: the history's rows, the
trial arithmetic, the norms, adoption, transport and dense observation stay in the
native state. These tests pin
that every case takes the resident engine, since SEP 47's phase 5 the only one native
has (the host-engine comparisons went with the inherited driver's fallbacks), and hold
its device-formed divisor to the reference's. The generic Hermite interpolant is a
comparison method's (``ComparisonMultistep``), which native refuses (solprop SEP 47
section 6.3); the reference steps it (``test_multistep_march.py``). The parity of
either against the reference backend is the shared suites'
(``test_exponential_routes.py``,
``test_compiled_multistep.py``).
"""

import math
from typing import cast

import numpy as np
import pytest

from backends._session_probe_b2 import run_session
from capability_routes import ROUTES
from sessions.executors import entrance
from solprop.backends import available
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.sessions import start_session
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    # The host engine these tests compare with runs solprop's march, with the warning
    # tests/backends/test_native_march_admission.py asserts (SEP 39).
    pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning"),
]

NativeBackend = pytest.importorskip("solprop_native").NativeBackend
backend_module = pytest.importorskip("solprop_native.backend")

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27
PLANES = list(np.linspace(SOLITON_PERIOD / 16, SOLITON_PERIOD / 2, 8))


def engine_of(route: str, integrator: Multistep) -> str:
    """Return the name of the engine Kokkos's march takes on a route."""
    simulation = ROUTES[route](None)
    simulation.settings = type(simulation.settings)(
        **{
            **{
                name: getattr(simulation.settings, name)
                for name in simulation.settings.__dataclass_fields__
            },
            "integrator": integrator,
        }
    )
    problem = simulation.translate()
    run_request = simulation._run_request()
    backend = NativeBackend(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        # The session builds its march over the engine when it starts.
        session = start_session(
            backend, problem, run_request, entrance(problem), owns=False
        )
        try:
            return type(backend._resident).__name__
        finally:
            session.close()
    finally:
        backend.close()


@pytest.mark.parametrize(
    "route",
    [
        "gnlse",
        "capillary (mode-averaged)",
        "capillary (multimode)",
        "general guide (constant)",
        "radial",
        "cartesian",
        "line",
        "planar",
    ],
)
def test_both_formulations_are_resident_on_every_constant_route(route):
    # Every route here has a constant or separable operator.
    assert engine_of(route, Multistep(formulation="interaction")) == "_ResidentAdams"
    assert engine_of(route, Multistep(formulation="exponential")) == "_ResidentAdams"


def test_a_varying_operator_is_resident_in_both_formulations():
    route = "capillary (pressure gradient)"
    assert engine_of(route, Multistep(formulation="interaction")) == "_ResidentAdams"
    assert engine_of(route, Multistep(formulation="exponential")) == "_ResidentAdams"


def test_a_divisor_formed_on_the_host_takes_the_same_steps():
    # solprop's controller has the norm's divisor formed in the exponential trial's
    # last pass, from the accepted state's magnitude its first pass reduces. The
    # reference forms the same formula's divisor on the host from its own magnitude:
    # on a pressure gradient, whose compensation joins every defect, both take the
    # same steps, with the same calls, to fields that agree to rounding.
    from backends.test_exponential_varying import LENGTH, gradient  # noqa: PLC0415

    simulation = gradient(
        save_planes=list(np.linspace(LENGTH / 4, LENGTH, 4)), rtol=1e-7
    )
    problem = simulation.translate()
    run_request = simulation._run_request()
    assert type(run_request.stepper.controller) is PIController

    def march(implementation):
        backend = implementation(
            problem,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            landed = run_session(backend, problem, run_request)
            run, record = landed.run, landed.record
            if implementation is NativeBackend:
                assert type(backend._resident).__name__ == "_ResidentAdams"
            calls = cast("dict[str, int]", record["response_calls"])
            total = calls["startup"] + calls["predictor"] + calls["corrected"]
            return np.array(run.fields), total, run.attempts
        finally:
            backend.close()

    fields, calls, steps = march(NativeBackend)
    host, host_calls, host_steps = march(NumpyRefBackend)
    assert (calls, steps) == (host_calls, host_steps)
    np.testing.assert_allclose(fields, host, rtol=0, atol=1e-12 * np.max(np.abs(host)))
