# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 39 phases 1 and 3: an Adams trial's stage scalars and tables formed natively.

Where a resident trial's stage rows are one bundle repeated, the native state reads the
scalar tables itself, uploaded once per run, and forms the trial's coefficient tables
and rate from its nodes (``_ResidentAdams.staged``), in either formulation. It performs
the reads of ``translate.stage.scalars`` by the same Clenshaw helper, in the same order,
so its scalars are solprop's exactly; with deterministic transform plans a staged march
then takes the host-packed march's steps, and its fields agree to rounding -- measured
bit for bit (SEP 39 section 4.9), held here to a rounding bound (SEP 25).
"""

import dataclasses
import itertools
import warnings

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import rate_weights

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
]

native = pytest.importorskip("solprop_native")
NativeBackend = native.NativeBackend
NATIVE = pytest.importorskip("solprop_native._helpers").NATIVE

STAGED = (
    "gnlse",
    "capillary (mode-averaged)",
    "capillary (envelope)",
    "capillary (multimode)",
    "capillary (pressure gradient)",
    "general guide (constant)",
    "radial",
    "cartesian",
    "line",
    "planar",
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""Every route: a constant, expanded or separable operator and prefactor, beside a
coupled flow or a moving guide's maps too (SEP 39 phase 4)."""

UNSTAGED: tuple[str, ...] = ()

COUPLED = (
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""The routes beside a coupled flow, which the exponential formulation alone steps."""


def translated(route: str, integrator: Multistep | None = None):
    """Return a route's simulation and problem under `integrator`, the exponential
    Multistep by default."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=integrator or Multistep()
        )
        return simulation, simulation.translate()


def backend_of(simulation, problem):
    return NativeBackend(
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )


def positions(problem) -> list[tuple[float, float]]:
    """Return anchor and stage pairs across the run, edges and breakpoints included."""
    start, end = float(problem.initial.z), float(problem.stepper.z_end)
    edges = [float(edge) for edge in problem.linear.phi_breakpoints]
    inner = sorted({*np.linspace(start, end, 7).tolist(), *edges})
    return [(start, end), (end, end), *itertools.pairwise(inner)]


@pytest.mark.parametrize("route", STAGED + UNSTAGED)
def test_only_repeated_bundles_are_staged(route):
    simulation, problem = translated(route)
    backend = backend_of(simulation, problem)
    try:
        assert backend._adams is not None
        assert backend._adams._engine.staged is (route in STAGED)
    finally:
        backend.close()


@pytest.mark.parametrize("route", STAGED)
def test_the_native_stage_scalars_are_solprop_s_bundle(route):
    simulation, problem = translated(route)
    backend = backend_of(simulation, problem)
    try:
        state = backend._native_state
        stages = problem.stepper.tableau.n_stages
        for z0, z1 in positions(problem):
            backend._hoist(z0)
            arrays, material = backend._resident_arrays(z0, z1)
            expected = dict(
                zip(
                    (
                        "phi_delta",
                        "phi_at",
                        "prefactor",
                        "density",
                        "taper",
                        "preionisation",
                        "raman_rates",
                    ),
                    arrays,
                    strict=True,
                )
            )
            expected |= {
                "kerr_density": material["kerr_density"],
                "plasma_densities": material["plasma_densities"],
                "plasma_preionisation": material["plasma_preionisation"],
                "raman_densities": material["raman_densities"],
                "dispersive_densities": material.get(
                    "dispersive_densities", np.empty(0)
                ),
                "flow_weights": material.get("flow_weights", np.empty(0)),
                "transform_weights": material.get("transform_weights", np.empty(0)),
            }
            found = state.adams_stage_scalars(z0, z1)
            for name, values in expected.items():
                flat = np.asarray(values, dtype=np.float64).reshape(-1)
                assert np.array_equal(np.asarray(found[name]), flat), (name, z0, z1)
            assert len(found["density"]) == stages
            if "flow_section" in material:
                assert found["flow_section"] == material["flow_section"]
    finally:
        backend.close()


def test_the_native_rate_is_read_on_the_closing_side():
    simulation, problem = translated("capillary (pressure gradient)")
    backend = backend_of(simulation, problem)
    try:
        linear = problem.linear
        assert linear.phi_expansion is not None
        edges = [float(edge) for edge in linear.rate.breakpoints]
        probes = [*edges, *np.linspace(edges[0], edges[-1], 11)]
        for z in probes:
            expected = rate_weights(linear, z, before=True, helpers=NATIVE)
            found = backend._native_state.adams_stage_rate(float(z))
            assert np.array_equal(np.asarray(found), expected), z
    finally:
        backend.close()


def march(route, monkeypatch, *, staged, integrator=None):
    """Run a route's planes on native, staged or with the host forming the stages."""
    simulation, problem = translated(route, integrator)
    with monkeypatch.context() as patch:
        if not staged:
            patch.setattr(NativeBackend, "_stage_tables", lambda self: None)
        backend = backend_of(simulation, problem)
    try:
        assert backend._adams is not None
        assert backend._adams._engine.staged is staged
        backend.set_initial(problem.initial.field0, None)
        fields, steps = [], []
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
            observed = backend.observe()
            fields.append(observed.field.copy())
            steps += list(observed.step_history.dz)
        return np.array(fields), np.array(steps), backend._adams.ledger.total
    finally:
        backend.close()


@pytest.mark.parametrize(
    "integrator",
    [Multistep(), Multistep(formulation="interaction")],
    ids=["exponential", "interaction"],
)
@pytest.mark.parametrize("route", STAGED)
def test_a_staged_march_takes_the_host_packed_march_s_steps(
    route, integrator, monkeypatch
):
    if integrator.formulation == "interaction" and route in COUPLED:
        pytest.skip("the interaction formulation is refused beside a coupled flow")
    # Deterministic transform plans, so that the two paths read the same transforms;
    # they then differ in no operation, and were measured equal bit for bit.
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    fields, steps, calls = march(route, monkeypatch, staged=True, integrator=integrator)
    other, other_steps, other_calls = march(
        route, monkeypatch, staged=False, integrator=integrator
    )
    assert calls == other_calls
    np.testing.assert_allclose(steps, other_steps, rtol=1e-12, atol=0)
    scale = np.max(np.abs(other))
    np.testing.assert_allclose(fields, other, rtol=0, atol=1e-13 * scale)
