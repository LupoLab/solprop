# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which march steps a native Adams run (solprop SEP 39 section 6.6).

The native march steps ``Multistep()`` in either formulation, and the one-order-higher
PEC comparisons, adaptive or replayed, on the routes whose trial is one bundle of stage
rows; any other configuration runs solprop's Python march on the same
resident engine, with a ``HostMarchWarning`` that names the reason, and the run's
metadata records which march stepped it. The private switch the benchmark harness uses
selects the Python march without a warning.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.driver import Simulation
from solprop.exceptions import HostMarchWarning
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.stepper.schedule import FixedSchedule

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

NativeBackend = pytest.importorskip("solprop_native").NativeBackend


def built(simulation, **keywords):
    """Return the native backend for `simulation`, translated."""
    problem = simulation.translate()
    return NativeBackend(
        problem, grid=simulation.grid, modes=simulation.modes, **keywords
    )


def on_route(route: str, integrator) -> Simulation:
    """Return `route`'s simulation stepped by `integrator`."""
    simulation = ROUTES[route](None)
    settings = dataclasses.replace(simulation.settings, integrator=integrator)
    simulation.settings = settings
    return simulation


@pytest.mark.parametrize(
    "route",
    [
        "gnlse",
        "capillary (mode-averaged)",
        "capillary (multimode)",
        "capillary (pressure gradient)",
        "radial",
    ],
)
def test_the_default_march_is_native_without_a_warning(route):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(on_route(route, Multistep()))
    try:
        assert backend._adams.native
        assert backend.metadata()["adams_march"] == "native"
    finally:
        backend.close()


@pytest.mark.parametrize(
    "integrator",
    [
        Multistep(formulation="interaction"),
        ComparisonMultistep(corrector="pec", order=4, corrector_order=5),
        ComparisonMultistep(
            corrector="pec", order=4, corrector_order=5, interpolant="hermite"
        ),
    ],
    ids=["interaction", "pec4/5", "pec4/5 hermite"],
)
def test_the_fused_methods_are_native(integrator):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(on_route("gnlse", integrator))
    try:
        assert backend._adams.native
    finally:
        backend.close()


def test_a_replayed_schedule_is_native():
    stepper = on_route("capillary (mode-averaged)", Multistep()).translate().stepper
    simulation = on_route("capillary (mode-averaged)", Multistep())
    schedule = FixedSchedule.uniform(
        0.0,
        float(stepper.z_end),
        # Below the in-stage collar's stable step, which a replay must respect.
        1e-7,
        save_planes=np.asarray(stepper.save_planes).tolist(),
    )
    simulation.settings = dataclasses.replace(simulation.settings, schedule=schedule)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(simulation)
    try:
        assert backend._adams.native
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("route", "integrator", "reason"),
    [
        (
            "gnlse",
            ComparisonMultistep(corrector="pece", order=4),
            "a comparison method's general path",
        ),
        ("general guide (moving)", Multistep(), "stage rows are not one bundle"),
    ],
    ids=["pece", "moving maps"],
)
def test_an_uncovered_run_takes_the_host_march_and_says_why(route, integrator, reason):
    with pytest.warns(HostMarchWarning, match=reason):
        backend = built(on_route(route, integrator))
    try:
        assert not backend._adams.native
        assert reason in backend._adams.refusal
        assert backend.metadata()["adams_march"] == "host"
    finally:
        backend.close()


def test_the_carrier_frame_is_native():
    # The chord of a constant or an expanded operator is formed natively; a z table's
    # stays solprop's.
    from backends.test_carrier_frame import (  # noqa: PLC0415 - a heavy module
        SCALAR,
        capillary,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(capillary(SCALAR))
    try:
        assert backend._adams.native
        assert backend._adams._policy.carrier
    finally:
        backend.close()


def test_the_private_switch_takes_the_host_march_quietly():
    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(on_route("gnlse", Multistep()), adams_march="host")
    try:
        assert not backend._adams.native
        assert backend._adams.refusal == "the private switch asked for it"
        assert backend.metadata()["adams_march"] == "host"
    finally:
        backend.close()


def test_a_runge_kutta_run_records_no_march():
    backend = built(ROUTES["gnlse"](None))
    try:
        if backend._adams is None:
            assert "adams_march" not in backend.metadata()
    finally:
        backend.close()
