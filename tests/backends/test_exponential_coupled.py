# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 5: the exponential formulation beside a coupled flow (Q3).

On a coupled section the step's reference is ``(Delta Phi / h) I + G_s``, the frozen
block of the section beside the cumulative diagonal, which is a multiple of the identity
in the channel axis; the trial's flow and moments are matrix functions of it on the bins
that carry a block
(:func:`solprop.backends.numpy_ref.adams_kernels.coupled_moments`). These
tests check the matrix moments against independent evaluations, the march against the
Runge-Kutta march on the constant-Schur cases of ``tests/flow_parity.py``, exactness
without a source, and what stays refused: the interaction formulation.
"""

import dataclasses
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

import flow_parity
from sessions.runs import adams_ledger
from sessions.runs import run as session_run
from solprop.backends import available
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.adams_kernels import (
    CoupledBlock,
    ExponentialKernels,
    coupled_moments,
)
from solprop.exceptions import BackendError, SolpropWarning
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.translate.coupled import schur_blocks

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
    exponent = np.full((3, 1, 1), shift)
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


def test_the_reference_carry_keeps_a_span_zero_sample():
    # Each sample's compensation carried to the step's end, with a sample with no flow
    # weights (no residual term) and one of span zero (the untransported witness's,
    # carried by nothing).
    rng = np.random.default_rng(18)
    generator = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    vectors, triangular = schur_blocks(np.moveaxis(generator, 0, -1))
    block = CoupledBlock(
        active=np.array([True, True]), vectors=vectors, triangular=triangular, section=0
    )
    exponent = np.full((3, 1, 2), -0.3 + 4.0j)
    shape = (3, *exponent.shape)
    rates = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    states = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    no_weights: list[np.ndarray | None] = [None, None, None]
    rows = (rates, states, no_weights)
    slots, spans = np.array([0, 1, 2]), np.array([0.5, 0.25, 0.0])
    dz = 0.3
    reference = ExponentialKernels()
    # The carry written out with SciPy's matrix exponential, by no route of the
    # kernels', so that the augmented exponential is compared with something else.
    unrouted = np.empty_like(rates)
    for row, span in enumerate(spans):
        for index in range(generator.shape[0]):
            action = expm(
                span * (exponent[0, 0, 0] * np.eye(3) + dz * generator[index])
            )
            term = (rates[row] - exponent / dz)[:, 0, index] * states[row][:, 0, index]
            unrouted[row, :, 0, index] = action @ term
    expected = reference.carry_coupled(exponent, dz, rows, slots, spans, block)
    # Within a few hundred ulp of these O(10) rows.
    scale = float(np.max(np.abs(unrouted)))
    assert float(np.max(np.abs(expected - unrouted))) <= 1e-13 * scale
    # The span-zero sample is the compensation itself, carried by nothing.
    np.testing.assert_allclose(
        expected[2], (rates[2] - exponent / dz) * states[2], rtol=1e-15
    )


def run(name, integrator, rtol, backend_class=NumpyRefBackend, **overrides):
    """Run a flow-parity case adaptively to its end; return the field and the run.

    The run's ``ledger`` is its Adams call ledger (``None`` for Runge-Kutta), and its
    ``march`` the reference's own march (``None`` on native, whose march is the
    session's).
    """
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
        run_request = simulation.run_request()
    backend = backend_class(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        record: dict[str, object] = {}
        field = session_run(backend, problem, record=record).fields[-1]
        adams = isinstance(integrator, Multistep)
        return field, SimpleNamespace(
            march=backend.march if backend.name == "numpy_ref" else None,
            ledger=adams_ledger(backend, record) if adams else None,
        )
    finally:
        backend.close()


@pytest.mark.parametrize("name", ["scalar", "te_tm"])
def test_a_constant_schur_section_agrees_with_runge_kutta(name):
    # A strong pulse: the source changes the field by 8e-3 over the window.
    reference, _ = run(name, RungeKutta(), 1e-11, energy=2e-3)
    field, march = run(name, EXPONENTIAL, 1e-9, energy=2e-3)
    assert march.ledger is not None and march.ledger.total < 20
    # Measured 2.6e-12 and 5.7e-13 at rtol 1e-9.
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert error < 1e-10


def test_without_a_source_the_coupled_flow_is_exact():
    # A negligible source (1e-20 J): the chord's matrix flow carries the state exactly.
    exact, _ = run("scalar", RungeKutta(), 1e-11, energy=1e-20)
    field, march = run("scalar", EXPONENTIAL, 1e-6, energy=1e-20)
    assert march.march is not None
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
    assert march.march is not None and march.march._engine.compensated
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
        run_request = simulation.run_request()
    backend = NumpyRefBackend(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    try:
        with pytest.raises(BackendError, match="diagonal operators"):
            backend._use_adams(EXPONENTIAL, compensation="untransported")
    finally:
        backend.close()


COMPILED: dict[str, type] = {}
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED["native"] = NativeBackend


@pytest.mark.parametrize("name", ["scalar", "te_tm", "gradient", "taper"])
@pytest.mark.parametrize("backend_name", sorted(COMPILED))
def test_a_compiled_backend_follows_the_reference_beside_a_coupled_flow(
    backend_name, name
):
    # Each backend exposes the reference's frozen block and residual, and its lab-frame
    # source runs without the flow's pipeline: the native session's resident passes
    # (SEP 16 phase 6) take the same calls as the reference.
    reference, march = run(name, EXPONENTIAL, 1e-8, energy=2e-3)
    field, compiled = run(name, EXPONENTIAL, 1e-8, COMPILED[backend_name], energy=2e-3)
    assert march.ledger is not None and compiled.ledger is not None
    assert compiled.ledger.total == march.ledger.total
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-11 * scale)
