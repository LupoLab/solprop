# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage march's register kernels against SEP 24 section 4.3's flow table.

Each kernel of :mod:`solprop.backends.numba_cpu.low_storage` is compared on small
synthetic arrays with the flow table written out in NumPy, the separable operator and
prefactor from :func:`solprop.translate.spatial.separable_values`. This module runs in
the coverage lane's interpreted pass (``poe cov-kernels``), so every kernel and body
line is counted, and compiled in the default lane.

**The bound, stated first**: a fold is at most four products and three adds of O(1)
operands after the element's principal root (fewer than 128 operations, as the
Cartesian kernels' test counts them), so every comparison is within ``128 eps``.
"""

import numpy as np
import pytest

from backends.test_numba_planar_kernels import data_of as slab_data_of
from backends.test_numba_planar_kernels import leaf as slab_leaf
from solprop.backends.bodies import FOLD_COMPLETE, FOLD_EXCHANGE, FOLD_SECOND, PLAIN
from solprop.backends.numba_cpu import low_storage as compiled
from solprop.translate.spatial import SeparableOperator, separable_values

BOUND = 128 * np.finfo(float).eps
"""The pointwise chain's bound, relative to O(1) operands."""

STEP = 0.3
"""An O(1) step, so that every term of the table is of the same size."""


def table_row(stage, step, source, a, b, c, p, q):  # noqa: PLR0913, PLR0917 - a table cell
    """Return ``A``, ``B`` and ``C`` after one complete pass of section 4.3's table."""
    if stage == 0:
        moved = p * a
        return moved, moved + step / 6 * p * source, moved + step / 2 * p * source
    if stage == 1:
        return a, b + step / 3 * source, a + step / 2 * source
    if stage == 2:
        return a, b + step / 3 * source, q * (a + step * source)
    return q * b + step / 6 * source, b, c


def random(rng, shape):
    """Return O(1) complex values."""
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("stage", range(4))
def test_the_fold_body_is_the_table_and_its_two_halves_sum_to_it(stage):
    """A complete fold is the table's row; the exchange then the second fold too."""
    rng = np.random.default_rng(stage)
    t, s, a, b, c, p, q = random(rng, 7)
    whole = PLAIN.low_storage_fold(stage, FOLD_COMPLETE, STEP, t + s, a, b, c, p, q)
    np.testing.assert_allclose(
        whole, table_row(stage, STEP, t + s, a, b, c, p, q), rtol=BOUND
    )
    half = PLAIN.low_storage_fold(stage, FOLD_EXCHANGE, STEP, t, a, b, c, p, q)
    split = PLAIN.low_storage_fold(stage, FOLD_SECOND, STEP, s, *half, p, q)
    # The registers a pass leaves unread after it may differ; those it defines agree.
    defined = {0: (0, 1, 2), 1: (1, 2), 2: (1, 2), 3: (0,)}[stage]
    for index in defined:
        assert split[index] == pytest.approx(whole[index], rel=BOUND)


def free_leaf(n_pol):
    """Return a three-column Cartesian leaf with loss, gain and an inactive bin."""
    return SeparableOperator(
        k_squared=np.array([1, 4 + 1j, 4 - 1j, 0, 9], complex),
        kperp_sq=np.array([0.0, 2.0, 4.0]),
        k_squared_low=np.zeros(5, complex),
        kperp_sq_low=np.zeros(3),
        frame=np.arange(5, dtype=float) / 8,
        spectral_rate=np.arange(5, dtype=float) / 16,
        prefactor_numerator=np.full(5, -2j),
        prefactor_clamped=np.full(5, -0.5j),
        active=np.array([True, True, True, True, False]),
        evanescent_rate=0.5,
        n_pol=n_pol,
        z_span=(0.0, 1.0),
        kperp_rate=np.array([0.0, 0.25, 1.5]),
    )


def free_data(coefficients):
    """Pack a leaf as ``NumbaCpuBackend._separable`` does."""
    return (
        coefficients.k_squared,
        coefficients.kperp_sq,
        coefficients.k_squared_low,
        coefficients.kperp_sq_low,
        coefficients.frame,
        coefficients.spectral_rate,
        coefficients.prefactor_numerator,
        coefficients.prefactor_clamped,
        coefficients.active,
        coefficients.evanescent_rate,
        coefficients.n_pol,
        coefficients.kperp_rate,
    )


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_cartesian_kernels_are_the_table_with_the_separable_operator(
    n_pol, parallel, noise
):
    """Input, table and every fold mode, tabulated and recomputed, with the noise."""
    coefficients = free_leaf(n_pol)
    data = free_data(coefficients)
    operator, prefactor = separable_values(coefficients)
    shape = (3 * n_pol, 5)
    rate, factor = operator.reshape(shape), prefactor.reshape(shape)
    flow = np.exp(0.5 * STEP * rate)
    window = np.linspace(0.5, 1, 5)
    active = coefficients.active
    mask = np.array([True, False, True, True, True])
    kernels = compiled.make_free_kernels(noise=noise, parallel=parallel)
    table = np.empty(shape, complex)
    kernels.table(data, 0.5 * STEP, table)
    np.testing.assert_allclose(table, flow, rtol=BOUND)
    rng = np.random.default_rng(n_pol)
    drawn = random(rng, shape)
    position, scale = 0.7, 0.25

    def advanced(z):
        return drawn * np.exp(1j * z * np.imag(rate)) if noise else 0

    register = random(rng, shape)
    scratch = np.empty(shape, complex)
    kernels.input(data, register, drawn, position, scale, scratch)
    np.testing.assert_allclose(
        scratch, scale * (register + advanced(position)), rtol=BOUND
    )
    for stage in range(4):
        for tabulated in (True, False):
            for mode, term, forward in (
                (FOLD_COMPLETE, 1, False),
                (FOLD_COMPLETE, 1, True),
                (FOLD_COMPLETE, 2, False),
                (FOLD_COMPLETE, 0, False),
                (FOLD_EXCHANGE, 1, False),
                (FOLD_SECOND, 2, False),
            ):
                a, b, c, spectrum = (random(rng, shape) for _ in range(4))
                registers = (a.copy(), b.copy(), c.copy())
                work = spectrum.copy()
                partials = np.full(shape[0], np.nan)
                kernels.fold(
                    data,
                    (stage, mode, term),
                    STEP,
                    work,
                    window,
                    active,
                    registers,
                    table,
                    tabulated,
                    (drawn, position, scale, forward),
                    mask,
                    partials,
                )
                source = {
                    0: np.zeros(shape, complex),
                    1: factor * window * spectrum,
                    2: active * spectrum,
                }[term]
                expected = [
                    np.vectorize(PLAIN.low_storage_fold)(
                        stage, mode, STEP, source, a, b, c, flow, flow
                    )[index]
                    for index in range(3)
                ]
                name = f"pass {stage} mode {mode} term {term} {tabulated} {forward}"
                writes = (
                    (0,)
                    if stage == 3
                    else (1,)
                    if forward
                    else (0, 1, 2)
                    if stage == 0
                    else (1, 2)
                )
                for index in writes:
                    np.testing.assert_allclose(
                        registers[index],
                        expected[index],
                        rtol=BOUND,
                        atol=BOUND,
                        err_msg=name,
                    )
                if mode == FOLD_EXCHANGE:
                    again = a if stage == 0 else c
                    np.testing.assert_allclose(
                        work, scale * (again + advanced(position)), rtol=BOUND
                    )
                elif forward and stage < 3:
                    np.testing.assert_allclose(
                        work, scale * (expected[2] + advanced(position)), rtol=BOUND
                    )
                final = stage == 3 and mode != FOLD_EXCHANGE
                if final:
                    norm = np.sum(np.abs(expected[0][:, mask]) ** 2, axis=1)
                    np.testing.assert_allclose(partials, norm, rtol=BOUND)
                else:
                    assert np.all(np.isnan(partials)), name


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_planar_kernels_are_the_table_with_the_wall_operator(n_pol, parallel):
    """The planar input with the noise, the table and the fold of both halves."""
    coefficients = slab_leaf(n_pol)
    data = slab_data_of(coefficients)
    operator, prefactor = separable_values(coefficients)
    shape = (4 * n_pol, 5)
    rate, factor = operator.reshape(shape), prefactor.reshape(shape)
    flow = np.exp(0.5 * STEP * rate)
    kernels = compiled.make_slab_kernels(noise=True, parallel=parallel)
    plain = compiled.make_slab_kernels(noise=False, parallel=parallel)
    table = np.empty(shape, complex)
    kernels.table(data, 0.5 * STEP, table)
    np.testing.assert_allclose(table, flow, rtol=BOUND)
    rng = np.random.default_rng(7 + n_pol)
    drawn, register = random(rng, shape), random(rng, shape)
    rows = np.empty(shape, complex)
    kernels.input(data, register, drawn, 0.4, 0.5, rows)
    np.testing.assert_allclose(
        rows, 0.5 * (register + drawn * np.exp(0.4j * np.imag(rate))), rtol=BOUND
    )
    plain.input(data, register, drawn, 0.4, 0.5, rows)
    np.testing.assert_allclose(rows, 0.5 * register, rtol=BOUND)
    window = np.linspace(0.5, 1, 5)
    active = coefficients.active
    mask = np.ones(5, dtype=bool)
    for stage in range(4):
        for tabulated in (True, False):
            for terms in ((True, True), (True, False), (False, True), (False, False)):
                a, b, c, material, absorbed = (random(rng, shape) for _ in range(5))
                registers = (a.copy(), b.copy(), c.copy())
                partials = np.zeros(shape[0])
                kernels.fold(
                    data,
                    stage,
                    STEP,
                    terms,
                    (material, absorbed),
                    window,
                    active,
                    registers,
                    table,
                    tabulated,
                    mask,
                    partials,
                )
                source = np.zeros(shape, complex)
                if terms[0]:
                    source += factor * window * material
                if terms[1]:
                    source += active * absorbed
                expected = table_row(stage, STEP, source, a, b, c, flow, flow)
                for got, want in zip(registers, expected, strict=True):
                    np.testing.assert_allclose(got, want, rtol=BOUND, atol=BOUND)
                if stage == 3:
                    np.testing.assert_allclose(
                        partials, np.sum(np.abs(expected[0]) ** 2, axis=1), rtol=BOUND
                    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_the_modal_kernels_pad_the_input_and_fold_the_source(contiguous, parallel):
    """The padded input with its noise and the fold with the absorber's term."""
    rng = np.random.default_rng(3)
    shape, fine = (3, 4), 8
    for absorber in (False, True):
        check_modal_fold(rng, shape, fine, contiguous, parallel, absorber=absorber)
    crop = np.array([0, 1, 2, 3]) if contiguous else np.array([0, 1, 6, 7])
    register, drawn = random(rng, shape), random(rng, shape)
    for noise in (False, True):
        kernels = compiled.make_modal_kernels(
            contiguous=contiguous, noise=noise, absorber=True, parallel=parallel
        )
        padded = np.full((3, fine), np.nan, complex)
        kernels.input(register, drawn, crop, padded, 0.5)
        expected = np.zeros((3, fine), complex)
        expected[:, crop] = 0.5 * (register + (drawn if noise else 0))
        np.testing.assert_allclose(padded, expected, rtol=BOUND, atol=0)


def check_modal_fold(rng, shape, fine, contiguous, parallel, *, absorber):
    """Fold every pass on the modal kernels, with or without the absorber's term."""
    crop = np.array([0, 1, 2, 3]) if contiguous else np.array([0, 1, 6, 7])
    kernels = compiled.make_modal_kernels(
        contiguous=contiguous, noise=False, absorber=absorber, parallel=parallel
    )
    window = np.linspace(0.5, 1, 4)
    mask = np.array([True, True, False, True])
    p, q = random(rng, shape), random(rng, shape)
    for stage in range(4):
        for active in (True, False):
            a, b, c, prefactor, damping = (random(rng, shape) for _ in range(5))
            spectrum = random(rng, (3, fine))
            registers = (a.copy(), b.copy(), c.copy())
            partials = np.zeros(3)
            kernels.fold(
                stage,
                STEP,
                active,
                spectrum,
                crop,
                prefactor,
                window,
                damping,
                registers,
                (p, q),
                mask,
                partials,
            )
            index = np.arange(4) if contiguous else crop
            source = (
                prefactor * window * spectrum[:, index] + (damping if absorber else 0)
                if active
                else np.zeros(shape, complex)
            )
            expected = table_row(stage, STEP, source, a, b, c, p, q)
            for got, want in zip(registers, expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=BOUND, atol=BOUND)
            if stage == 3:
                np.testing.assert_allclose(
                    partials,
                    np.sum(np.abs(expected[0][:, mask]) ** 2, axis=1),
                    rtol=BOUND,
                )


def test_the_forward_exponential_is_numpy_s():
    exponent = np.array([[0.1 + 2j, -0.5 - 1j], [0j, 1.5 + 0.25j]])
    out = np.empty_like(exponent)
    compiled.exponentiate_forward(exponent, out)
    np.testing.assert_allclose(out, np.exp(exponent), rtol=BOUND)
    compiled.exponentiate_forward(exponent, exponent)
    np.testing.assert_allclose(exponent, out, rtol=0, atol=0)


def test_the_terms_exponential_is_the_combination_s():
    """The flow from the terms of ``Phi`` with no exponent array: ``combine``'s sum.

    Three O(1) terms: their weighted sum is two products and two adds an element, well
    inside ``BOUND`` before the exponential. A single term weighted by one is the
    term itself, so its flow is the forward exponential's exactly.
    """
    rng = np.random.default_rng(7)
    terms = random(rng, (3, 2, 5))
    weights = np.array([0.25, -0.5, 1.0])
    out = np.empty((2, 5), dtype=np.complex128)
    compiled.exponentiate_terms(weights, terms, out)
    np.testing.assert_allclose(
        out, np.exp(np.tensordot(weights, terms, axes=1)), rtol=BOUND
    )
    alone = np.empty_like(out)
    compiled.exponentiate_terms(np.array([1.0]), terms[:1], out)
    compiled.exponentiate_forward(np.ascontiguousarray(terms[0]), alone)
    np.testing.assert_array_equal(out, alone)
