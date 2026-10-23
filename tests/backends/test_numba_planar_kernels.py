# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar kind's compiled kernels against the frozen host operator.

Each kernel of :mod:`solprop.backends.numba_cpu.planar` is compared on a small synthetic
separable leaf carrying ``kappa_sq`` -- two channels, loss, gain, an evanescent column
and an inactive bin -- with :func:`solprop.translate.spatial.separable_values`, the
reference's operator and prefactor, and with the reference's stage expression. This
module runs in the coverage lane's interpreted pass (``poe cov-kernels``), so every
kernel and body line is counted, and compiled in the default lane.

**The bound, stated first**: as the Cartesian kernels' test
(``tests/backends/test_numba_cpu_kernels.py``), the principal root and the complete
pointwise chain are fewer than 128 scalar operations on O(1) operands, so every
comparison is within ``128 eps``, the parallel kernels against the serial ones too;
what is exact is asserted exactly -- the interpreted element with a zero wall term is
the free-space element (subtracting a zero rounds nothing), a y transform of one
buffer leaves the other half's, and a hard mask stores zeros.
"""

import numpy as np
import pytest

from solprop.backends.bodies import PLAIN
from solprop.backends.numba_cpu import kernels, planar
from solprop.exceptions import BackendError
from solprop.translate.spatial import SeparableOperator, separable_values

BOUND = 128 * np.finfo(float).eps
"""The pointwise chain's bound, relative to O(1) operands."""


def leaf(n_pol):
    """Return a two-channel planar leaf: four columns, five frequencies."""
    active = np.array([True, True, True, True, False])
    wall = np.array(
        [[0.25 + 0.01j, 0.5 - 0.02j, 1.0 + 0.1j, 5.0, 0], [1.5, 0.5j, -0.5, 0.1j, 0]]
    )
    return SeparableOperator(
        k_squared=np.array([1, 4 + 1j, 4 - 1j, 2, 9], complex),
        kperp_sq=np.array([0.0, 0.0, 2.0, 2.0]),
        k_squared_low=np.array([1e-17, 0, -1e-17, 0, 0], complex),
        kperp_sq_low=np.array([0.0, 1e-17, 0.0, 0.0]),
        frame=np.arange(5, dtype=float) / 8,
        spectral_rate=np.arange(5, dtype=float) / 16,
        prefactor_numerator=np.full(5, -2j),
        prefactor_clamped=np.full(5, -0.5j),
        active=active,
        evanescent_rate=0.5,
        n_pol=n_pol,
        z_span=(0.0, 1.0),
        kappa_sq=wall,
    )


def data_of(coefficients):
    """Pack a leaf as :func:`solprop.backends.numba_cpu.planar.slab_data` does."""
    assert coefficients.kappa_sq is not None
    return (
        coefficients.k_squared,
        coefficients.kperp_sq,
        coefficients.k_squared_low,
        coefficients.kperp_sq_low,
        coefficients.kappa_sq,
        coefficients.frame,
        coefficients.spectral_rate,
        coefficients.prefactor_numerator,
        coefficients.prefactor_clamped,
        coefficients.active,
        coefficients.evanescent_rate,
        coefficients.n_pol,
        coefficients.n_channels,
    )


def test_the_slab_element_is_the_free_element_with_the_wall_subtracted_last():
    """A zero wall term is the free-space element to the last bit; a wall is ``q``."""
    operands = (4 + 1j, 2.0, 1e-17 + 0j, 1e-17)
    tail = (0.125, 0.0625, -2j, -0.5j, True, 0.5)
    # The slab element carries no transverse collar: the free element's rate zero.
    free = PLAIN.spatial_element(*operands, *tail, 0.0)
    assert PLAIN.slab_element(*operands, 0j, *tail) == free
    wall = 0.5 - 0.25j
    square = ((operands[0] - operands[1]) + (operands[2] - operands[3])) - wall
    assert PLAIN.slab_element(*operands, wall, *tail) == PLAIN.longitudinal_rate(
        square, *tail, 0.0
    )
    # An evanescent column is clamped as in free space, and an inactive bin is zero.
    rate, prefactor = PLAIN.slab_element(1 + 0j, 4.0, 0j, 0.0, 0j, *tail)
    assert rate.real == pytest.approx(-0.5 - 0.5 * 0.0625)
    assert prefactor == -0.5j
    assert PLAIN.slab_element(*operands, wall, 0.0, 0.0, -2j, -0.5j, False, 0.5) == (
        0j,
        0j,
    )


@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_planar_kernels_match_the_frozen_host_operator(n_pol):
    """Propagator, phase and stage finish against ``separable_values``.

    The stage is the reference's ``(prefactor * window * material + mask *
    absorbed) * exp(-dPhi)`` with each term present or absent.
    """
    coefficients = leaf(n_pol)
    data = data_of(coefficients)
    operator, prefactor = separable_values(coefficients)
    shape = (4 * n_pol, 5)
    rate, factor = operator.reshape(shape), prefactor.reshape(shape)
    kernels_ = planar.make_kernels(parallel=False)
    forward, inverse = np.empty(shape, complex), np.empty(shape, complex)
    kernels_.write_propagator(data, 0.25, forward, inverse)
    np.testing.assert_allclose(forward, np.exp(0.25 * rate), rtol=BOUND)
    np.testing.assert_allclose(inverse * forward, 1, rtol=BOUND)
    phase = np.empty(shape)
    planar.write_phase(data, 0.25, phase)
    np.testing.assert_allclose(phase, 0.25 * np.imag(rate), rtol=0, atol=BOUND)
    generator = np.random.default_rng(49)
    material = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    absorbed = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    window = np.linspace(0, 1, 5)
    mask = coefficients.active
    stage = np.empty(shape, complex)
    for flags in ((True, True), (True, False), (False, True), (False, False)):
        kernels_.finish_stage(
            data, material, absorbed, window, mask, inverse, stage, flags
        )
        expected = np.zeros(shape, complex)
        if flags[0]:
            expected += factor * window * material
        if flags[1]:
            expected += mask * absorbed
        np.testing.assert_allclose(
            stage, expected * inverse, rtol=BOUND, atol=BOUND, err_msg=str(flags)
        )
    # The channels are distinct: the same k_y on the other channel is another rate.
    assert not np.allclose(rate[0], rate[n_pol])


def test_the_parallel_planar_kernels_are_the_serial_ones():
    """Each column's arithmetic is the same in either loop, to the module's bound."""
    data = data_of(leaf(1))
    shape = (4, 5)
    generator = np.random.default_rng(3)
    material = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    absorbed = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    outputs = []
    for parallel in (False, True):
        compiled = planar.make_kernels(parallel=parallel)
        forward, inverse = np.empty(shape, complex), np.empty(shape, complex)
        compiled.write_propagator(data, 0.5, forward, inverse)
        stage = np.empty(shape, complex)
        compiled.finish_stage(
            data,
            material,
            absorbed,
            np.ones(5),
            np.ones(5, dtype=bool),
            inverse,
            stage,
            (True, True),
        )
        outputs.append((forward, inverse, stage))
    for serial, parallel in zip(*outputs, strict=True):
        np.testing.assert_allclose(parallel, serial, rtol=BOUND, atol=0)


@pytest.mark.parametrize("absorber", [False, True])
def test_the_y_plans_are_the_unnormalised_y_transform(absorber):
    """The pair acts on y alone, in place, and the absorber's half rides the forward."""
    plans = planar.build_slab_plans(
        (4, 3, 5), absorber=absorber, flag="estimate", threads=1
    )
    assert plans.spectrum.shape == (2 if absorber else 1, 4, 3, 5)
    generator = np.random.default_rng(11)
    values = generator.normal(size=(4, 3, 5)) + 1j * generator.normal(size=(4, 3, 5))
    plans.rows[:] = values.reshape(12, 5)
    plans.inverse.execute()
    np.testing.assert_allclose(
        plans.rows.reshape(4, 3, 5), 4 * np.fft.ifft(values, axis=0), rtol=0, atol=BOUND
    )
    plans.rows[:] = values.reshape(12, 5)
    if absorber:
        plans.absorbed[:] = 2 * values.reshape(12, 5)
    plans.forward_all.execute()
    transformed = np.fft.fft(values, axis=0).reshape(12, 5)
    np.testing.assert_allclose(plans.rows, transformed, rtol=0, atol=16 * BOUND)
    if absorber:
        np.testing.assert_allclose(
            plans.absorbed, 2 * transformed, rtol=0, atol=32 * BOUND
        )
        # The material half's own forward leaves the absorber's half alone.
        before = plans.absorbed.copy()
        plans.forward.execute()
        np.testing.assert_array_equal(plans.absorbed, before)
    else:
        assert plans.forward_all is plans.forward
    assert plans.plan_heap_growth >= 0
    assert plans.planning_seconds >= 0


def test_the_y_plans_refuse_an_unknown_flag():
    with pytest.raises(BackendError, match="not a planning flag"):
        planar.build_slab_plans(
            (4, 3, 5), absorber=False, flag="exhaustive!", threads=1
        )


def test_the_hard_mask_takes_the_rows_to_the_state():
    """The split-step finish's last pass is the Cartesian mask kernel, unchanged."""
    rows = np.arange(10, dtype=complex).reshape(2, 5) + 1
    mask = np.array([True, False, True, True, False])
    state = np.empty_like(rows)
    kernels.apply_hard_mask(rows, mask, state)
    np.testing.assert_array_equal(state, rows * mask)
