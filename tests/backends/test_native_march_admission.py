# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which march steps a native Adams run (solprop SEP 39 section 6.6).

The native march steps ``Multistep()`` in either formulation, and the one-order-higher
PEC comparisons, adaptive or replayed, on the routes whose trial is one bundle of stage
rows; any other configuration runs solprop's Python march on the same
resident engine, with a ``HostMarchWarning`` that names the reason, and the run's
metadata records which march stepped it. The backend's test interface
(``solprop_native.backend._ADAMS_MARCH``, no longer an environment variable or a
constructor keyword: solprop SEP 47 section 6.3) selects the Python march without a
warning. The other comparison variants are the
reference's alone (solprop SEP 47 section 6.3): the backend refuses them by name before
it allocates anything.
"""

import copy
import dataclasses
import warnings

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.driver import Simulation
from solprop.exceptions import BackendError, HostMarchWarning
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.stepper.schedule import FixedSchedule

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

NativeBackend = pytest.importorskip("solprop_native").NativeBackend
backend_module = pytest.importorskip("solprop_native.backend")


def built(simulation, **keywords):
    """Return the native backend for `simulation`, translated."""
    problem = simulation.translate()
    run_request = simulation._run_request()
    return NativeBackend(
        problem, run_request, grid=simulation.grid, modes=simulation.modes, **keywords
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
    stepper = on_route("capillary (mode-averaged)", Multistep())._run_request().stepper
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
    "integrator",
    [
        ComparisonMultistep(corrector="pece", order=4),
        ComparisonMultistep(corrector="pece", order=4, corrector_order=5),
        ComparisonMultistep(corrector="none", order=3),
        ComparisonMultistep(corrector="pec", order=4),
        ComparisonMultistep(corrector="pec", order=4, corrector_order=4),
    ],
    ids=["pece4", "pece4/5", "ab3", "pec4 matched", "pec4/4"],
)
def test_a_comparison_variant_s_general_path_is_refused_before_allocation(
    integrator, monkeypatch
):
    # solprop SEP 47 section 6.3: reference-only, refused by name at admission. The
    # native state, the first allocation, is never built.
    built_states = []
    monkeypatch.setattr(
        backend_module._core, "State", lambda *args: built_states.append(args)
    )
    with pytest.raises(BackendError, match="reference-only") as refused:
        built(on_route("gnlse", integrator))
    assert f"comparison variant {integrator.name} " in str(refused.value)
    assert built_states == []


@pytest.mark.parametrize("leaf", ["operator", "prefactor"])
def test_a_representation_the_stage_tables_do_not_read_is_refused_at_admission(
    leaf, monkeypatch
):
    # solprop SEP 47 section 6.3: every representation the translation forms is read
    # by the native stage tables, and the problem's own invariants keep each
    # representation's leaves present, so no built-in configuration reaches a row
    # the host forms. A problem stripped of its leaves anyway is refused by name at
    # admission, before the native state is built, not stepped on host rows.
    simulation = on_route("gnlse", Multistep())
    problem = simulation.translate()
    request = simulation._run_request()
    owner = copy.copy(problem.linear if leaf == "operator" else problem.responses)
    object.__setattr__(
        owner, "phi_constant" if leaf == "operator" else "prefactor_constant", None
    )
    stripped = copy.copy(problem)
    object.__setattr__(stripped, "linear" if leaf == "operator" else "responses", owner)
    assert not backend_module.stage_tables_available(stripped)
    built_states = []
    monkeypatch.setattr(
        backend_module._core, "State", lambda *args: built_states.append(args)
    )
    with pytest.raises(BackendError, match="does not read") as refused:
        NativeBackend(stripped, request, grid=simulation.grid, modes=simulation.modes)
    assert f"the {leaf}'s 'constant' representation" in str(refused.value)
    assert built_states == []


@pytest.mark.parametrize(
    "route", ["general guide (moving)", "capillary (coherent wall)", "radial"]
)
def test_moving_maps_and_coupled_flows_are_native(route):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(on_route(route, Multistep()))
    try:
        assert backend._adams.native
    finally:
        backend.close()


def test_the_host_engine_takes_the_host_march(monkeypatch):
    # A coupled flow off the eigenbasis route or wider than sixteen channels keeps the
    # host engine (ADR-044), whose history the native march cannot reach.
    monkeypatch.setattr(NativeBackend, "_adams_engine", lambda *_: None)
    with pytest.warns(HostMarchWarning, match="the host engine"):
        backend = built(on_route("capillary (coherent wall)", Multistep()))
    try:
        assert not backend._adams.native
    finally:
        backend.close()


@pytest.mark.skipif(
    pytest.importorskip("solprop_native").build_info().device_backend != "none",
    reason="device builds refuse the carrier frame by name",
)
def test_the_carrier_frame_is_native():
    # The chord of a constant, an expanded or a tabulated operator is formed natively
    # (tests/sessions/test_native_session_carrier_table.py for the z table's).
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


def test_the_test_interface_takes_the_host_march_quietly(monkeypatch):
    monkeypatch.setattr("solprop_native.backend._ADAMS_MARCH", "host")
    with warnings.catch_warnings():
        warnings.simplefilter("error", HostMarchWarning)
        backend = built(on_route("gnlse", Multistep()))
    try:
        assert not backend._adams.native
        assert backend._adams.refusal == "the test interface asked for it"
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
