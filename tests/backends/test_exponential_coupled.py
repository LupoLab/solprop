# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 5: the exponential formulation beside a coupled flow (Q3).

On a coupled section the step's reference is ``(Delta Phi / h) I + G_s``, the frozen
block of the section beside the cumulative diagonal, which is a multiple of the identity
in the channel axis; the trial's flow and moments are matrix functions of it on the bins
that carry a block (:func:`solprop.backends.multistep.coupled_moments`). These tests
check the matrix moments against independent evaluations, the march against the
Runge-Kutta march on the constant-Schur cases of ``tests/flow_parity.py``, exactness
without a source, and what stays refused: the interaction formulation and the CF4 and
(until it is delivered) frozen-residual sections.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

import flow_parity
from solprop.backends import available
from solprop.backends.multistep import (
    AdamsMarch,
    CoupledBlock,
    CoupledExponentialKernels,
    coupled_moments,
)
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.multistep import NumbaExponentialKernels
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError, SolpropWarning
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.translate.coupled import schur_blocks, schur_eigenbasis

pytestmark = pytest.mark.medium

EXPONENTIAL = Multistep(formulation="exponential")


def test_the_matrix_moments_are_the_integrals_they_stand_for():
    # k! phi_{k+1}(X) = int_0^1 exp((1 - s) X) s^k ds, with X = x I + h G and G a
    # nonnormal block: checked by quadrature, independently of the augmented matrix.
    rng = np.random.default_rng(16)
    generator = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    generator[np.tril_indices(3, -1)] *= 0.05
    vectors, triangular = schur_blocks(generator[:, :, np.newaxis])
    shift = -0.4 + 2.5j
    exponent = np.full((1, 3, 1, 1), shift)
    block = CoupledBlock(
        active=np.array([True]), vectors=vectors, triangular=triangular, section=0
    )
    dz = 0.3
    moments = coupled_moments(exponent, 4, block, dz)
    total = shift * np.eye(3) + dz * generator
    np.testing.assert_allclose(moments.flow_matrix[0], expm(total), atol=1e-13)
    for k in range(4):
        exact, _ = quad_vec(
            lambda s, k=k: expm((1 - s) * total) * s**k, 0.0, 1.0, epsrel=1e-12
        )
        np.testing.assert_allclose(moments.matrices[k, 0], exact, atol=1e-11)
    assert math.isclose(moments.matrices[0].shape[-1], 3)


def test_the_eigenbasis_route_matches_the_augmented_exponential():
    # The compiled backends' evaluation: on the bins of the flow actions' route each
    # function is W f(x + h lambda) Y. On a nonnormal block whose eigenvalues are well
    # separated it equals the reference's augmented exponential; a bin with a
    # coincident pair is off the route and keeps the augmented exponential.
    rng = np.random.default_rng(17)
    generator = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    generator[:, 2, 1] = generator[:, 1, 2] = 0.0
    generator[1] = np.triu(generator[1])
    generator[1, 1, 1] = generator[1, 0, 0]
    vectors, triangular = schur_blocks(np.moveaxis(generator, 0, -1))
    basis = schur_eigenbasis(vectors, triangular)
    assert basis[3].tolist() == [True, False]
    exponent = np.full((1, 3, 1, 2), -0.3 + 4.0j)
    reference = CoupledBlock(
        active=np.array([True, True]),
        vectors=vectors,
        triangular=triangular,
        section=0,
    )
    routed = dataclasses.replace(reference, basis=basis)
    # No bin on the route: every function is the augmented exponential's.
    left, right, eigenvalues, _ = basis
    unrouted = dataclasses.replace(
        reference, basis=(left, right, eigenvalues, np.zeros(2, dtype=bool))
    )
    for dz in (1e-2, 0.4):
        expected = coupled_moments(exponent, 7, reference, dz)
        for block in (routed, unrouted):
            found = coupled_moments(exponent, 7, block, dz)
            np.testing.assert_allclose(
                found.flow_matrix, expected.flow_matrix, atol=1e-13
            )
            np.testing.assert_allclose(found.matrices, expected.matrices, atol=1e-13)
        # The flow alone, no moment: what a startup or an order-one step asks for.
        flow = coupled_moments(exponent, 0, routed, dz)
        np.testing.assert_allclose(flow.flow_matrix, expected.flow_matrix, atol=1e-13)


def test_the_compiled_carry_is_the_reference_s_on_and_off_the_route():
    # Each sample's compensation carried to the step's end: the compiled pass on a
    # block wholly on the eigenbasis route, the reference's own beside a bin off it or
    # with no route, and in every case a sample with no flow weights (no residual
    # term) and one of span zero (the untransported witness's, carried by nothing).
    rng = np.random.default_rng(18)
    generator = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    vectors, triangular = schur_blocks(np.moveaxis(generator, 0, -1))
    basis = schur_eigenbasis(vectors, triangular)
    assert basis[3].tolist() == [True, True]
    plain = CoupledBlock(
        active=np.array([True, True]), vectors=vectors, triangular=triangular, section=0
    )
    left, right, eigenvalues, _ = basis
    blocks = {
        "routed": dataclasses.replace(plain, basis=basis),
        "partly routed": dataclasses.replace(
            plain, basis=(left, right, eigenvalues, np.array([True, False]))
        ),
        "no route": plain,
    }
    exponent = np.full((1, 3, 1, 2), -0.3 + 4.0j)
    shape = (3, *exponent.shape)
    rates = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    states = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    no_weights: list[np.ndarray | None] = [None, None, None]
    rows = (rates, states, no_weights)
    slots, spans = np.array([0, 1, 2]), np.array([0.5, 0.25, 0.0])
    dz = 0.3
    reference = CoupledExponentialKernels()
    compiled = NumbaExponentialKernels().coupled()
    # Each object's coupled passes are itself.
    assert reference.coupled() is reference
    assert compiled.coupled() is compiled
    for name, block in blocks.items():
        expected = reference.carry_coupled(exponent, dz, rows, slots, spans, block)
        found = compiled.carry_coupled(exponent, dz, rows, slots, spans, block)
        # The routed pass forms W f(x + h lambda) Y where the reference forms the
        # augmented exponential: within a few hundred ulp of these O(10) rows.
        scale = float(np.max(np.abs(expected)))
        assert float(np.max(np.abs(found - expected))) <= 1e-13 * scale, name
        # The span-zero sample is the compensation itself, carried by nothing.
        np.testing.assert_allclose(
            expected[2], (rates[2] - exponent / dz) * states[2], rtol=1e-15
        )


def run(name, integrator, rtol, backend_class=NumpyRefBackend, **overrides):
    """Run a flow-parity case adaptively to its end; return the field and the march."""
    with warnings.catch_warnings():
        for category in flow_parity.QUIET:
            warnings.simplefilter("ignore", category)
        # The whole window as one step allowed: the weak collar's advisory warning.
        warnings.simplefilter("ignore", SolpropWarning)
        simulation = flow_parity.build(
            name,
            schedule=None,
            dz_init=1e-6,
            dz_max=flow_parity.LENGTH,
            rtol=rtol,
            **overrides,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=integrator
        )
        problem = simulation.translate()
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        return backend.observe().require_field(), backend._adams
    finally:
        backend.close()


@pytest.mark.parametrize("name", ["scalar", "te_tm"])
def test_a_constant_schur_section_agrees_with_runge_kutta(name):
    # A strong pulse: the source changes the field by 8e-3 over the window.
    reference, _ = run(name, RungeKutta(), 1e-11, energy=2e-3)
    field, march = run(name, EXPONENTIAL, 1e-9, energy=2e-3)
    assert march is not None and march.ledger.total < 20
    # Measured 2.6e-12 and 5.7e-13 at rtol 1e-9.
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert error < 1e-10


def test_without_a_source_the_coupled_flow_is_exact():
    # A negligible source (1e-20 J): the chord's matrix flow carries the state exactly.
    exact, _ = run("scalar", RungeKutta(), 1e-11, energy=1e-20)
    field, march = run("scalar", EXPONENTIAL, 1e-6, energy=1e-20)
    assert march is not None
    np.testing.assert_allclose(field, exact, rtol=0, atol=1e-12 * np.abs(exact).max())


def test_the_interaction_formulation_still_refuses_a_coupled_flow():
    with pytest.raises(BackendError, match="does not step a problem with a coupled"):
        run("scalar", Multistep(formulation="interaction"), 1e-6)


@pytest.mark.parametrize(
    ("name", "bound"), [("gradient", 1e-10), ("taper", 1e-9)], ids=["gradient", "taper"]
)
def test_a_frozen_residual_flow_agrees_with_runge_kutta(name, bound):
    # The residual, linear in the state, joins the transported compensation; the taper
    # crosses section edges, where the history's residual is rebuilt against the new
    # section's block. Measured 1.2e-11 (gradient) and 1.4e-10 (taper) at rtol 1e-9,
    # 1.4e-8 on the taper at 1e-7: the error follows the tolerance.
    reference, _ = run(name, RungeKutta(), 1e-11, energy=2e-3)
    field, march = run(name, EXPONENTIAL, 1e-9, energy=2e-3)
    assert march is not None and march._engine.compensated
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert error < bound


def test_the_untransported_witness_is_for_diagonal_operators_only():
    with warnings.catch_warnings():
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build("gradient")
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=EXPONENTIAL
        )
        problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        with pytest.raises(BackendError, match="diagonal operators"):
            AdamsMarch(backend, EXPONENTIAL, compensation="untransported")
    finally:
        backend.close()


COMPILED: dict[str, type] = {"numba_cpu": NumbaCpuBackend}
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED["native"] = NativeBackend


@pytest.mark.parametrize("name", ["scalar", "te_tm", "gradient", "taper"])
@pytest.mark.parametrize("backend_name", sorted(COMPILED))
def test_a_compiled_backend_follows_the_reference_beside_a_coupled_flow(
    backend_name, name
):
    # Each backend exposes the reference's frozen block and residual, and its lab-frame
    # source runs without the flow's pipeline: numba_cpu's compiled passes and
    # the native backend's resident ones take the same calls as the reference.
    reference, march = run(name, EXPONENTIAL, 1e-8, energy=2e-3)
    field, compiled = run(name, EXPONENTIAL, 1e-8, COMPILED[backend_name], energy=2e-3)
    assert march is not None and compiled is not None
    # Kokkos steps a coupled block in its resident engine (SEP 16 phase 6).
    engine = type(compiled._engine).__name__
    assert engine == ("_ResidentAdams" if backend_name == "native" else "HostHistory")
    assert compiled.ledger.total == march.ledger.total
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-11 * scale)


@pytest.mark.skipif("native" not in available(), reason="native is not installed")
def test_the_resident_channel_limit_is_the_estimates():
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        ADAMS_MAX_CHANNELS,
    )

    from solprop.backends.memory import RESIDENT_CHANNELS  # noqa: PLC0415

    assert ADAMS_MAX_CHANNELS == RESIDENT_CHANNELS
