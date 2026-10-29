# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 39 phase 1: an exponential trial's stage scalars and tables formed natively.

Where a resident exponential trial's stage rows are one bundle repeated, the native
state reads the scalar tables itself, uploaded once per run, and forms the trial's
coefficient tables and rate from its nodes (``_ResidentAdams.staged``). It performs the
reads of ``translate.stage.scalars`` by the same Clenshaw helper, in the same order, so
its scalars are solprop's to the bit; with deterministic transform plans a staged march
therefore takes the host-staged march's steps and gives its fields bit for bit
(SEP 39 sections 4.8 and 4.9).
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
from solprop.translate.stage import scalars

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
)
"""The routes whose trial rows are one bundle repeated: a constant or expanded
operator and prefactor, with no z table, separable operator, dispersive channel,
moving map or coupled flow."""

UNSTAGED = (
    "general guide (moving)",
    "cartesian",
    "line",
    "planar",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)


def translated(route: str):
    """Return a route's simulation and problem under the exponential Multistep."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=Multistep()
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
            bundle = scalars(problem, z0, z1, helpers=NATIVE)
            arrays, material = backend._repeated_arrays(bundle)
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
            }
            found = state.adams_stage_scalars(z0, z1)
            for name, values in expected.items():
                flat = np.asarray(values, dtype=np.float64).reshape(-1)
                assert np.array_equal(np.asarray(found[name]), flat), (name, z0, z1)
            assert len(found["density"]) == stages
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


def march(route, monkeypatch, *, staged):
    """Run a route's planes on native, staged or with the host forming the stages."""
    simulation, problem = translated(route)
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


@pytest.mark.parametrize("route", STAGED)
def test_a_staged_march_is_the_host_staged_march_to_the_bit(route, monkeypatch):
    # Deterministic transform plans, so that two runs of either path agree to the bit.
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    fields, steps, calls = march(route, monkeypatch, staged=True)
    other, other_steps, other_calls = march(route, monkeypatch, staged=False)
    assert calls == other_calls
    assert np.array_equal(steps, other_steps)
    assert np.array_equal(fields, other)
