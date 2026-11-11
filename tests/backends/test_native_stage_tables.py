# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 39 phases 1 and 3: an Adams trial's stage scalars and tables formed natively.

Where a resident trial's stage rows are one bundle repeated, the native state reads the
scalar tables itself, uploaded once per run, and forms the trial's coefficient tables
and rate from its nodes (``_ResidentAdams.staged``), in either formulation. It performs
the reads of ``translate.stage.scalars`` by the same Clenshaw helper, in the same order,
so its scalars are solprop's exactly. The host-packed march these tables were measured
against bit for bit (SEP 39 section 4.9) was deleted with the inherited driver's
fallbacks in SEP 47's phase 5: every route the backend admits is staged.
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


def translated(route: str, integrator: Multistep | None = None):
    """Return a route's simulation, problem and request under `integrator`, the
    exponential Multistep by default."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=integrator or Multistep()
        )
        return simulation, simulation.translate(), simulation._run_request()


def backend_of(simulation, problem, run_request):
    return NativeBackend(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )


def positions(problem, run_request) -> list[tuple[float, float]]:
    """Return anchor and stage pairs across the run, edges and breakpoints included."""
    start, end = float(problem.initial.z), float(run_request.stepper.z_end)
    edges = [float(edge) for edge in problem.linear.phi_breakpoints]
    inner = sorted({*np.linspace(start, end, 7).tolist(), *edges})
    return [(start, end), (end, end), *itertools.pairwise(inner)]


@pytest.mark.parametrize("route", STAGED)
def test_every_route_is_staged(route):
    simulation, problem, run_request = translated(route)
    backend = backend_of(simulation, problem, run_request)
    try:
        assert backend._adams is not None
        assert backend._native_state.has_stage_tables()
    finally:
        backend.close()


@pytest.mark.parametrize("route", STAGED)
def test_the_native_stage_scalars_are_solprop_s_bundle(route):
    simulation, problem, run_request = translated(route)
    backend = backend_of(simulation, problem, run_request)
    try:
        state = backend._native_state
        stages = run_request.stepper.tableau.n_stages
        for z0, z1 in positions(problem, run_request):
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
    simulation, problem, run_request = translated("capillary (pressure gradient)")
    backend = backend_of(simulation, problem, run_request)
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
