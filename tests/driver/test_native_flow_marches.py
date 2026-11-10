# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow on native: every case, both marches, both formation routes.

``test_native_coupled_flow.py`` runs its own launches on the default march at its own
bound, and forms the actions on the taper alone. Here is what that leaves: the eight
cases of ``tests/flow_parity.py`` under ``B_10`` on both marches, the Runge-Kutta march
beside a capillary wall carrying a real Kerr source through forward and inverse actions,
with the check that the flow matters; every stage's action on all eight cases and both
routes at ``eps_form``; and random nonnormal blocks through up to four squarings against
the per-bin bound, which the C++ test holds only to ``norm(lT) <= 2`` at a fixed 4096
eps.
"""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from flow_parity import (
    CASES,
    QUIET,
    UNIT,
    build,
    departure,
    flow_scales,
    formation_bound,
    run,
    section_formation_bounds,
    ten_step_bound,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import (
    _coupled_flow_maps,
    _flow_action,
    _triangular_exponential,
)
from solprop.translate.coupled import schur_blocks, schur_eigenbasis

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
    # Noise-free launches into time windows kept tight for speed: the boundary monitor
    # warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning"),
]


@pytest.fixture(scope="module")
def cases():
    """Build each case's simulation and problem once for the module, on demand."""
    built = {}

    def get(name):
        if name not in built:
            simulation = build(name)
            built[name] = (simulation, simulation.translate())
        return built[name]

    return get


def on_march(problem, march):
    """Return the problem on the default integrator or on the Runge-Kutta march."""
    if march == "default":
        return problem
    return replace(problem, stepper=replace(problem.stepper, integrator=None))


@pytest.mark.medium
@pytest.mark.parametrize("march", ["default", "runge_kutta"])
@pytest.mark.parametrize("name", CASES)
def test_ten_steps_are_the_reference_within_the_bound(
    cases, name, march, record_property
):
    """Ten fixed steps on native against ``numpy_ref``, within ``B_10``.

    The bound (``tests/flow_parity.py``), stated before the measurement:
    ``C^S exp(Phi) (||a_0||/||a_L||) S eps_step`` with ``C`` the declared
    conditioning, ``Phi`` the launch's nonlinear scale and ``eps_step`` the diagonal
    half's 32 u, the source's share and the flow's own products. The run is also
    checked to exercise the flow at all: without the leaf it departs by more than a
    thousand bounds.
    """
    simulation, problem = cases(name)
    problem = on_march(problem, march)
    grid = simulation.grid
    reference = run(NumpyRefBackend, problem, grid)
    actual = run(native.NativeBackend, problem, grid)
    bound = ten_step_bound(flow_scales(problem, grid, reference))
    measured = departure(actual, reference)
    record_property("bound", bound)
    record_property("measured", measured)
    assert bound > 0.0
    assert measured <= bound
    bare = replace(problem, linear=replace(problem.linear, coupled_flow=None))
    assert departure(run(native.NativeBackend, bare, grid), reference) > 1e3 * bound


def _spectral(stack):
    return np.linalg.norm(stack, ord=2, axis=(-2, -1))


@pytest.mark.medium
@pytest.mark.parametrize("route", ["eigenbasis", "scaling_and_squaring"])
@pytest.mark.parametrize("name", CASES)
def test_the_flow_s_actions_are_the_reference_s_within_the_formation_bound(
    cases, name, route, monkeypatch, record_property
):
    """Every stage's native pair is the reference's within ``eps_form``.

    Bound, stated first (``tests/flow_parity.py``): the native action and the
    reference's are each within their route's formation bound of the exact
    ``Z exp(l T) Z^H``, so within the sum of the two, relative to the action; the whole
    step's measured conditioning is the reference's singular-value product within
    ``32 n u``, half the certificate's own allowance. The second route sets the
    eigenbasis limit to zero, so every bin is formed by native's scaling and squaring.
    """
    simulation, problem = cases(name)
    if route == "scaling_and_squaring":
        monkeypatch.setattr(
            "solprop.translate.coupled.eigenbasis_limit", lambda n_modes: 0.0
        )
    held = backend_module._host_flow(problem)
    assert held is not None
    assert np.all(held.route) if route == "eigenbasis" else not np.any(held.route)
    reference = _coupled_flow_maps(problem)
    assert reference is not None
    nodes = problem.stepper.tableau.nodes
    fractions = np.array([*(float(node) for node in nodes), 1.0])
    dz = float(np.max(problem.stepper.schedule.steps))
    bound = formation_bound(problem, dz)
    assert 0.0 < bound < 1e-12
    backend = native.NativeBackend(problem, grid=simulation.grid)
    worst = 0.0
    try:
        state = backend._native_state
        shape = (fractions.size, -1, reference.n_modes, reference.n_modes)
        for section in range(reference.frozen.shape[0]):
            measured = state.form_flow_actions(section, dz, fractions)[0]
            forward = state.flow_actions_to_host(inverse=False).reshape(shape)
            inverse = state.flow_actions_to_host(inverse=True).reshape(shape)
            for slot, fraction in enumerate(fractions):
                if fraction == 0.0:
                    continue
                pair = _flow_action(reference, section, float(fraction) * dz)
                for got, want in zip((forward[slot], inverse[slot]), pair, strict=True):
                    error = _spectral(got - want) / _spectral(want)
                    worst = max(worst, float(np.max(error)))
            pair = _flow_action(reference, section, dz)
            whole = _spectral(pair[0]) * _spectral(pair[1])
            assert np.max(abs(measured / whole - 1.0)) <= 32 * reference.n_modes * UNIT
    finally:
        backend.close()
    record_property("bound", bound)
    record_property("measured", worst)
    assert worst <= bound


def _held(flow, _problem):
    """Stand in for ``_host_flow``: the random blocks whatever the problem."""
    return flow


def _random_flow(active, vectors, triangular, *, route):
    """Return a host flow of one constant section over random blocks."""
    left, right, eigenvalues, own = schur_eigenbasis(vectors, triangular)
    route = own if route is None else route
    n_modes = triangular.shape[-1]
    return backend_module._HostFlow(
        active=active,
        scored=np.ones(int(np.count_nonzero(active)), dtype=bool),
        vectors=vectors[None],
        triangular=triangular[None],
        frozen=np.zeros((1, *triangular.shape), dtype=complex),
        condition_bound=np.full((1, triangular.shape[0]), 1e6),
        residual_terms=None,
        left=np.ascontiguousarray(left[None]),
        right=np.ascontiguousarray(right[None]),
        eigenvalues=np.ascontiguousarray(eigenvalues[None]),
        route=route[None],
        row_sum=np.array(
            [float(np.max(np.sum(abs(triangular), axis=-1), initial=0.0))]
        ),
    ), n_modes


@pytest.mark.medium
@pytest.mark.parametrize("length", [2e-4, 3e-3])
def test_the_formed_actions_are_the_reference_s_on_either_route(
    cases, length, monkeypatch
):
    """Both routes of native's formation against ``Z exp(+-l T) Z^H``.

    Bound, stated first: the formation bound of ``tests/flow_parity.py``
    (``section_formation_bounds``) for each bin at each length, with the bin's own
    certificate ``exp(l (mu_max - mu_min))``, for the route the bin takes; forcing
    every bin to scaling and squaring bounds it by twice the reference's. Random
    blocks of this scale have ``||l T||`` between 0.3 and 5, so the reference squares
    zero to four times; the longer length exercises the squarings. The blocks replace
    a real case's in the payload that crosses to the engine, so the formation reached
    is the production one with the original's operands.
    """
    simulation, problem = cases("scalar")
    rng = np.random.default_rng(6)
    n_active = 5
    n_modes = problem.shape[0]
    assert n_modes == 3
    shape = (n_modes, n_modes, n_active)
    generator = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 300.0
    vectors, triangular = schur_blocks(generator)
    blocks = np.moveaxis(generator, -1, 0)
    hermitian = np.linalg.eigvalsh(
        0.5 * (blocks + np.conj(np.swapaxes(blocks, -1, -2)))
    )
    certificate = np.exp(length * (hermitian[:, -1] - hermitian[:, 0]))
    active = np.zeros(problem.grids.n_omega, dtype=bool)
    active[: 2 * n_active : 2] = True
    # The engine takes the tableau's stage count and the whole step; the first three
    # are the original's fractions, and the rest repeat them.
    count = problem.stepper.tableau.n_stages + 1
    fractions = np.resize([0.0, 0.5, 1.0], count)
    adjoint = np.conj(np.swapaxes(vectors, -1, -2))
    own = schur_eigenbasis(vectors, triangular)[3]
    assert np.any(own), "the eigenbasis route must be taken somewhere"
    for taken in (None, np.zeros_like(own)):
        flow, _ = _random_flow(active, vectors, triangular, route=taken)
        monkeypatch.setattr(backend_module, "_host_flow", partial(_held, flow))
        backend = native.NativeBackend(problem, grid=simulation.grid)
        try:
            state = backend._native_state
            state.form_flow_actions(0, length, fractions)
            forward = state.flow_actions_to_host(inverse=False).reshape(
                count, n_active, n_modes, n_modes
            )
            inverse = state.flow_actions_to_host(inverse=True).reshape(
                count, n_active, n_modes, n_modes
            )
        finally:
            backend.close()
        assert not np.any(forward[0]) and not np.any(inverse[0])
        lengths = fractions[1:3] * length
        bounds = section_formation_bounds(
            vectors, triangular, lengths, certificate, route=taken
        )
        for slot in (1, 2):
            exponent = fractions[slot] * length * triangular
            for got, sign in ((forward[slot], 1.0), (inverse[slot], -1.0)):
                expected = vectors @ _triangular_exponential(sign * exponent) @ adjoint
                error = _spectral(got - expected) / _spectral(expected)
                assert np.all(error <= bounds[slot - 1])
