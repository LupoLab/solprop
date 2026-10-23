# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route's compiled kernels against the reference arithmetic.

Each kernel of :mod:`solprop.backends.numba_cpu.general_modal` replaces one ``einsum``
or one loop of :func:`solprop.backends.numpy_ref.general_modal.general_source`, and each
is compared here with that expression on a small random chunk -- two nodes of three, so
that the chunk's offset and the ownership test are exercised -- with the transforms left
out, because they are FFTW's and are compared end to end in
``tests/backends/test_numba_general_modal.py``.

The bound is **1e-14 relative** to the largest magnitude of each output, written
before the run: the longest reduction here is the contraction's 27 terms, whose
rounding is below ``30 u = 3.3e-15`` of the sum of the magnitudes, and random
operands of one sign-balanced scale keep that sum within a factor of three of the
largest output. Bitwise agreement is not expected anywhere (SEP 25): numpy's complex
product is fused into multiply-adds on this host, and a compiled kernel may contract
and order its sums as it likes; the real cubic's symmetric contraction, the
parallel combination and the leading-bin (``contiguous``) synthesis and projection are
held to the same bound.

This module is small enough to run interpreted, which is how the coverage lane
counts the kernel bodies (``poe cov-kernels``).
"""

import numpy as np
import pytest

from solprop.backends.bodies import GENERAL_PAIRS, PLAIN
from solprop.backends.numba_cpu.general_modal import MAP_BLOCK, make_general_kernels
from solprop.backends.numpy_ref.general_modal import (
    expand_pair_history,
    symmetric_cubic,
    symmetric_kerr_coefficients,
    symmetric_monomials,
)
from solprop.constants import EPS0

NODES, MODES, N_OMEGA, N_TO = 3, 2, 5, 8
FIRST, COUNT = 1, 2
TOLERANCE = 1e-14


def random_complex(generator, *shape):
    """Return a complex array of unit scale."""
    return generator.standard_normal(shape) + 1j * generator.standard_normal(shape)


def assert_close(got, expected):
    """Assert agreement to :data:`TOLERANCE` of the expected output's scale."""
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=TOLERANCE * scale)


@pytest.fixture(scope="module")
def generator():
    return np.random.default_rng(20260923)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_the_synthesis_writes_every_bin_of_the_chunk(generator, parallel, contiguous):
    """``E = sum_m S a`` at the chunk's nodes, zero outside the crop, stale bins too.

    Through ``crop`` and, on a field grid's leading-bin crop, directly.
    """
    kernels = make_general_kernels(
        analytic=True, delayed=False, parallel=parallel, contiguous=contiguous
    )
    synthesis = random_complex(generator, NODES, 3, MODES, N_OMEGA)
    scaled = random_complex(generator, MODES, N_OMEGA)
    crop = np.arange(N_OMEGA, dtype=np.intp)
    fine = np.full((3 * COUNT, N_TO), np.nan + 0j)
    kernels.synthesise(synthesis, scaled, crop, FIRST, COUNT, fine)
    expected = np.zeros_like(fine)
    expected[:, crop] = np.einsum(
        "qcmw,mw->qcw", synthesis[FIRST : FIRST + COUNT], scaled
    ).reshape(3 * COUNT, N_OMEGA)
    assert_close(fine, expected)


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("delayed", [True, False])
def test_the_response_is_the_reference_contraction(generator, analytic, delayed):
    """The instantaneous cubic, and the six-entry drive when a delay is declared."""
    kernels = make_general_kernels(analytic=analytic, delayed=delayed, parallel=False)
    signal = random_complex(generator, 3 * COUNT, N_TO)
    samples = signal if analytic else signal.real
    tensor = generator.standard_normal((NODES, 3, 3, 3, 3))
    polarisation = np.full((3 * COUNT, N_TO), np.nan)
    drive = np.zeros((6 * COUNT, 2 * N_TO))
    kernels.respond(samples, tensor, FIRST, COUNT, polarisation, drive)
    field = np.asarray(samples).reshape(COUNT, 3, N_TO)
    pair = np.einsum("qkt,qlt->qklt", field, field.conj())
    weighted = np.einsum("qijkl,qklt->qijt", tensor[FIRST : FIRST + COUNT], pair)
    expected = np.real(EPS0 * np.einsum("qijt,qjt->qit", weighted, field))
    assert_close(polarisation, expected.reshape(3 * COUNT, N_TO))
    if not delayed:
        assert not np.any(drive)
        return
    real = np.real(pair) if not analytic else 0.5 * np.real(pair)
    stored = np.stack([real[:, row, column] for row, column in GENERAL_PAIRS], axis=1)
    assert_close(drive[:, :N_TO], stored.reshape(6 * COUNT, N_TO))
    assert not np.any(drive[:, N_TO:])


@pytest.mark.parametrize("parallel", [False, True])
def test_the_real_response_is_the_symmetric_contraction(generator, parallel):
    """The real form's kernel is the reference's `symmetric_cubic`, to its rounding.

    Real products and sums in the reference's order, within the module's bound;
    against the dense order it is within the contraction's rounding, which the
    response test above bounds.
    """
    kernels = make_general_kernels(analytic=False, delayed=False, parallel=parallel)
    samples = generator.standard_normal((3 * COUNT, N_TO))
    tensor = generator.standard_normal((NODES, 3, 3, 3, 3))
    polarisation = np.full((3 * COUNT, N_TO), np.nan)
    kernels.respond(samples, tensor, FIRST, COUNT, polarisation, polarisation)
    expected = EPS0 * symmetric_cubic(
        symmetric_kerr_coefficients(tensor[FIRST : FIRST + COUNT]),
        samples.reshape(COUNT, 3, N_TO),
    )
    assert_close(polarisation, expected.reshape(3 * COUNT, N_TO))


def test_the_symmetric_bodies_are_the_references_order():
    """The rows are the reference's coefficients and the monomials its order."""
    generator = np.random.default_rng(29)
    tensor = generator.standard_normal((3, 3, 3, 3))
    coefficients = symmetric_kerr_coefficients(tensor[None])[0]
    rows = tuple(PLAIN.general_symmetric_row(tensor, i) for i in range(3))
    assert_close(np.array(rows), coefficients)
    samples = generator.standard_normal((3, 1))
    field = samples[:, 0]
    monomials = tuple(
        (field[j] * field[k]) * field[last] for j, k, last in symmetric_monomials(3)
    )
    expected = tuple(PLAIN.general_symmetric_sum(row, monomials) for row in rows)
    assert PLAIN.general_real_kerr(rows, samples, 0, 0) == expected
    assert_close(
        np.array(expected), symmetric_cubic(coefficients[None], samples[None])[0, :, 0]
    )


def test_the_delayed_kernels_touch_the_owned_nodes_alone(generator):
    """The product, the contraction and the addition, over one response's nodes.

    Node ``FIRST`` belongs to response 1 and node ``FIRST + 1`` to response 0, so
    the pass of response 0 must leave the first node's rows as they were: a
    convolution is not linear in its kernel's owner.
    """
    kernels = make_general_kernels(analytic=False, delayed=True, parallel=False)
    owner = np.array([0, 1, 0], dtype=np.int32)
    kernel = random_complex(generator, N_TO + 1)
    drive_spectrum = random_complex(generator, 6 * COUNT, N_TO + 1)
    product = np.zeros_like(drive_spectrum)
    kernels.product(kernel, drive_spectrum, owner, 0, FIRST, COUNT, product)
    assert not np.any(product[:6])
    assert_close(product[6:], kernel * drive_spectrum[6:])

    history = generator.standard_normal((6 * COUNT, 2 * N_TO))
    tensor = generator.standard_normal((3, 3, 3, 3))
    samples = random_complex(generator, 3 * COUNT, N_TO)
    scale = 1.0 / (2 * N_TO)
    total = np.zeros((3 * COUNT, N_TO))
    kernels.contract(history, tensor, owner, 0, FIRST, COUNT, samples, scale, total)
    full = expand_pair_history(history[:, :N_TO].reshape(COUNT, 6, N_TO) * scale)
    expected = np.einsum(
        "ijkl,qklt,qjt->qit", tensor, full, np.real(samples).reshape(COUNT, 3, N_TO)
    )
    assert not np.any(total[:3])
    assert_close(total[3:], expected[1])

    polarisation = generator.standard_normal((3 * COUNT, N_TO))
    before = polarisation.copy()
    kernels.add_delayed(polarisation, total, 3 * COUNT)
    assert_close(polarisation, before + EPS0 * total)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_the_projection_adds_the_chunk_to_the_source(generator, parallel, contiguous):
    """``source += sum_qc Q P`` over the chunk, in one partial per channel."""
    kernels = make_general_kernels(
        analytic=True, delayed=False, parallel=parallel, contiguous=contiguous
    )
    projection = random_complex(generator, MODES, NODES, 3, N_OMEGA)
    spectrum = random_complex(generator, 3 * COUNT, N_TO // 2 + 1)
    crop = np.arange(N_OMEGA, dtype=np.intp)
    source = random_complex(generator, MODES, N_OMEGA)
    before = source.copy()
    partial = np.full_like(source, np.nan)
    kernels.project(projection, spectrum, crop, FIRST, COUNT, partial, source)
    added = np.einsum(
        "mqcw,qcw->mw",
        projection[:, FIRST : FIRST + COUNT],
        spectrum.reshape(COUNT, 3, -1)[..., crop],
    )
    assert_close(partial, added)
    assert_close(source, before + added)


def test_the_plain_bodies_are_the_reference_per_sample():
    """The interpreted family on one sample, which is what the kernels inline."""
    generator = np.random.default_rng(3)
    tensor = generator.standard_normal((3, 3, 3, 3))
    samples = random_complex(generator, 3, 1)
    values = PLAIN.general_kerr(tensor, samples, 0, 0)
    field = samples[:, 0]
    expected = np.einsum(
        "ijkl,j,k,l->i", tensor, field, field, field.conj(), optimize=False
    )
    assert_close(np.array(values), expected)
    first, second = samples[0, 0], samples[1, 0]
    assert PLAIN.general_pair_drive(first, second, True) == pytest.approx(
        0.5 * (first * np.conj(second)).real, rel=1e-15
    )
    assert PLAIN.general_pair_drive(first.real, second.real, False) == (
        first.real * second.real
    )


def test_the_parallel_combination_is_the_serial_one():
    """Blocks shared by threads, each element a sum of five terms: to its rounding.

    Random coefficients over three blocks and a partial fourth, with the zero
    weights of another section, on both compilations and at the default block.
    """
    rng = np.random.default_rng(41)
    size = 3 * 64 + 17
    terms = rng.standard_normal((10, size)) + 1j * rng.standard_normal((10, size))
    weights = np.zeros(10)
    weights[5:] = rng.uniform(-1.0, 1.0, 5)
    outputs = []
    for parallel in (False, True):
        kernels = make_general_kernels(analytic=False, delayed=False, parallel=parallel)
        for block in (64, MAP_BLOCK):
            out = np.full(size, np.nan + 0j)
            kernels.combine(weights, terms, out, block)
            outputs.append(out)
    for out in outputs[1:]:
        assert_close(out, outputs[0])
    # Sequential from the first nonzero term, which is what the bound assumes.
    expected = np.zeros(size, dtype=complex)
    for term in range(5, 10):
        expected = expected + weights[term] * terms[term]
    assert_close(outputs[0], expected)
