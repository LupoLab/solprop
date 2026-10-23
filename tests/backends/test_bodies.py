# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The per-element bodies: the families agree, and agree with the reference.

Two claims, and they are different claims.

The first is that the families agree: whatever ``define_bodies`` is handed as a
compiler, the arithmetic it produces is the same to its rounding -- a few roundings
for everything algebraic, which a compiler may contract (SEP 25), and one unit in the
last place where a libm transcendental is involved, because numba's ``sin``, ``cos``,
``exp`` and ``log`` are LLVM's and not CPython's. That is why the coverage lane's
numbers and the default lane's are not identical, and it is measured here rather than
assumed.

The second is that the bodies reproduce the reference passes they were read off,
sample by sample -- because a family that agrees with itself and with nothing else
would pass the first claim while being wrong. Each reproduction is held to the
rounding of the operations both sides perform, derived and stated on the test.
"""

import math

import numpy as np
import pytest

from bounds import higham_gamma
from rate_oracles import adk_oracle
from solprop._numba_cache import njit
from solprop.backends.bodies import PLAIN, Bodies, RateArrays, define_bodies
from solprop.backends.numpy_ref import passes
from solprop.nonlinear.responses import PlasmaResponse
from solprop.translate.ionisation import build_rate_table
from solprop.translate.responses import (
    KerrCoefficients,
    plasma_coefficients,
)

# Built once for the module: an njit dispatcher compiles on its first call, and the
# point of the comparison is the arithmetic rather than the compiler.
HOST = define_bodies(njit(inline="always", cache=True))

FIELD_RANGE = (1e9, 2e11)
EPS = np.finfo(float).eps


@pytest.mark.parametrize("field", [0.0, -2.7, 1.2 - 3.4j])
def test_raman_body_extraction_preserves_host_operation_order(field):
    real_field = not isinstance(field, complex)
    expected_drive = (
        field * field
        if real_field
        else 0.5 * (field.real * field.real + field.imag * field.imag)
    )
    total, factor, convolution, window = field * 0.3, 2.7, -0.24, 0.67
    expected_finish = (total + factor * field * convolution) * window
    # A drive is at most three roundings and a finish four, on either side, of terms
    # that do not cancel here: eight eps relative (exact at a zero field).
    for family in (PLAIN, HOST):
        drive = family.raman_drive(field.real, field.imag, real_field)
        assert drive == pytest.approx(expected_drive, rel=8 * EPS, abs=0.0)
        finish = family.raman_finish(total, field, convolution, factor, window)
        assert finish == pytest.approx(expected_finish, rel=8 * EPS, abs=0.0)


@pytest.mark.parametrize("rate", [0.0, 1e12, 1e16])
def test_raman_damped_body_retains_normalization_and_decay(rate):
    value, delay = -2.3, 3e-15
    expected = value * math.exp(-rate * delay)
    # The predeclared64epsilon envelope includes both exponential evaluations
    # and the product; the body retains the undamped row's sign and units.
    bound = 64 * np.finfo(float).eps * abs(expected)
    assert bound > 0
    for family in (PLAIN, HOST):
        assert abs(family.raman_damped(value, rate, delay) - expected) <= bound


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (0.0, 0.0),
        (3.0, -4.0),
        (1e-200, 2.5e-201),
        (1e200, 2.5e199),
        (-1e-200, 1e200),
        (1e-320, 1e-320),
    ],
)
def test_scaled_magnitude_preserves_tiny_and_huge_finite_components(first, second):
    expected = math.hypot(first, second)
    # Five scaled arithmetic steps and two sqrt/hypot paths fit64epsilon; two
    # smallest-subnormal quanta cover absolute rounding for a subnormal result.
    bound = 64 * np.finfo(float).eps * expected + 2 * np.nextafter(0.0, 1.0)
    assert bound > 0
    for family in (PLAIN, HOST):
        actual = family.magnitude_pair(first, second)
        assert math.isfinite(actual)
        assert abs(actual - expected) <= bound


@pytest.fixture(scope="module")
def table():
    """An ADK table for argon over the range the plasma benchmarks use."""
    return build_rate_table(adk_oracle("Ar"), FIELD_RANGE)


@pytest.fixture(scope="module")
def arrays(table):
    """The same table as the flat arrays a kernel takes."""
    return RateArrays.of(table)


def bind(bodies: Bodies, decorate):
    """Return one function calling every body, compiled by ``decorate``.

    A kernel binds each body to a closure local rather than reaching through the
    :class:`Bodies` record: numba cannot type the dataclass as a global, which is the
    whole reason the factory exists.
    """
    rate_at, loss_scale, propagator_pair = (
        bodies.rate_at,
        bodies.loss_scale,
        bodies.propagator_pair,
    )
    kerr_real, kerr_analytic = bodies.kerr_real, bodies.kerr_analytic
    trapezoid, fraction_at, accumulate = (
        bodies.trapezoid,
        bodies.fraction_at,
        bodies.accumulate,
    )

    @decorate
    def every(magnitude, rate_arrays, field, factor, exponent):
        rate = rate_at(magnitude, rate_arrays)
        scale = loss_scale(rate, 0.25, field * field, 2.5e-18)
        forward_re, forward_im, inverse_re, inverse_im = propagator_pair(
            -0.5 * exponent, exponent
        )
        modulus = field * field + 0.25
        real, imag = kerr_analytic(
            field, 0.5, modulus, field * field - 0.25, field, factor, 0.5, 0.25
        )
        return (
            rate,
            scale,
            forward_re,
            forward_im,
            inverse_re,
            inverse_im,
            kerr_real(field, field * field, factor),
            real,
            imag,
            trapezoid(field, 0.5 * field, 1e-16),
            fraction_at(magnitude * 1e-11, 0.125),
            accumulate(1.0, field, 0.25),
        )

    return every


#: The tuple positions of ``bind``'s results that pass through a libm transcendental.
TRANSCENDENTAL = frozenset({0, 1, 2, 3, 4, 5, 10})

# The state a real column sees: below the table, inside it at both ends, above it, and
# the underflowing sample the loss guard exists for.
MAGNITUDES = (0.0, 1e-200, 5e8, 1e9, 3.7e9, 1e11, 2e11, 5e11)


def horner_terms(arrays, magnitude):
    """Return the rate cubic's rounding scale at `magnitude`, zero outside.

    ``sum_i |c_i l^i|``, the Horner sum's terms, plus the offset's rounding carried
    through the cubic's slope, expressed in units of ``gamma_6`` so the caller's
    ``gamma_6`` factor applies to both.
    """
    if not arrays.field_min <= magnitude <= arrays.field_max:
        return 0.0
    log_field = math.log(magnitude)
    last = arrays.coefficients.shape[1] - 1
    index = min(
        max(int((log_field - arrays.log_field_min) / arrays.d_log_field), 0), last
    )
    knot = arrays.log_field_min + index * arrays.d_log_field
    local = abs(log_field - knot)
    c = np.abs(arrays.coefficients[:, index])
    terms = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
    # The offset `local` is itself rounded, and contraction may round it once rather
    # than twice: a few units in the last place of the larger of log E and the knot,
    # moving the cubic by its slope times that.
    slope = (3.0 * c[0] * local + 2.0 * c[1]) * local + c[2]
    offset = 2.0 * EPS * max(abs(log_field), abs(knot))
    return float(terms + slope * offset / higham_gamma(6))


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_families_agree(arrays, magnitude):
    """The compiled family and the interpreted one are the same arithmetic, rounded.

    Each algebraic result is at most four roundings on either side, which a compiler
    may contract differently, of terms whose cancellation costs at most a factor of
    two on these samples (the analytic cubic's real part at the smallest field): 16
    eps relative.
    """
    plain = bind(PLAIN, lambda function: function)
    host = bind(HOST, njit(cache=True))
    arguments = (magnitude, arrays, magnitude, 1e-30, 0.75)
    wanted, got = plain(*arguments), host(*arguments)
    assert len(got) == 12
    for index, (one, other) in enumerate(zip(wanted, got, strict=True)):
        if index in {0, 1}:
            # The rate is exp of a cubic in log E: the compiled Horner sum contracts
            # (SEP 25), so each side's cubic is within gamma_6 of its terms'
            # magnitudes, plus the rounded offset into the interval through the
            # cubic's slope; the exponential turns that into a relative error, beside
            # the two libms' last place; four times that for the stated margin.
            relative = 2.3e-16 + 2 * higham_gamma(6) * horner_terms(arrays, magnitude)
            assert other == pytest.approx(one, rel=4 * relative, abs=0.0)
        elif index == 10:
            # 1 - (1 - p) exp(-X) cancels: the two exponentials' last place and the
            # subtraction's rounding are absolute, and relative to the small fraction
            # they grow by 1/f.
            absolute = 0.875 * 2.3e-16 + 2 * EPS
            assert other == pytest.approx(one, rel=0.0, abs=absolute)
        elif index in TRANSCENDENTAL:
            # numba's exp/log/sin/cos are LLVM's, CPython's are the platform libm's;
            # the two differ in the last place and no source-level choice closes it.
            assert other == pytest.approx(one, rel=2.3e-16, abs=0.0)
        else:
            assert other == pytest.approx(one, rel=16 * EPS, abs=0.0)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_rate_reproduces_the_reference_table(table, arrays, magnitude):
    """``rate_at`` is ``RateTable.__call__`` on one sample."""
    wanted = float(table(np.array([magnitude]))[0])
    assert PLAIN.rate_at(magnitude, arrays) == wanted


def test_the_rate_reproduces_the_reference_pass_over_a_column(table, arrays):
    """``rate_at`` over a column is :func:`passes.ionisation_rate` on that column."""
    field = np.geomspace(1e8, 5e11, 512).reshape(1, 1, -1)
    wanted = passes.ionisation_rate(field, table)
    got = np.array([PLAIN.rate_at(abs(value), arrays) for value in field[0, 0]])
    # The scalar and vector paths use the platform libm separately: one ulp.
    np.testing.assert_array_max_ulp(got, wanted[0], maxulp=1)
    assert np.count_nonzero(got) > 0


def test_the_knot_index_is_clamped_at_both_ends_of_the_table():
    """The two clamps exist for a log that rounds outside the span it came from.

    ``field_min`` and ``field_max`` are the first and last knots, so an index outside
    ``[0, n_intervals - 1]`` can only come from ``log`` disagreeing in the last place
    with the ``log_field_min`` the table stored. That is a real possibility and not a
    dead branch, and it is provoked here with a table whose stored logarithm is
    deliberately off rather than by hoping for a round-off that lands the right way.
    """
    coefficients = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    span = RateArrays(
        coefficients=coefficients,
        log_field_min=math.log(1.0),
        d_log_field=1.0,
        field_min=1.0,
        field_max=math.e**2,
        rate_max=1e15,
    )
    # The last knot itself: the index is one past the last interval and is pulled back
    # onto it, so the rate is the interval's own cubic at its right-hand end.
    assert PLAIN.rate_at(span.field_max, span) == pytest.approx(math.e**2)
    # A stored logarithm above the field it belongs to drives the index negative.
    shifted = span._replace(log_field_min=math.log(1.0) + 1.5)
    assert PLAIN.rate_at(1.0, shifted) == pytest.approx(math.exp(-1.5))


def test_the_propagator_pair_is_both_signs_of_the_exponential():
    """One real ``exp`` and one ``sincos`` give what two complex exponentials give."""
    for real, imag in ((0.0, 0.0), (-0.5, 1.25), (1e-3, -7.0), (-30.0, 1e4)):
        exponent = complex(real, imag)
        forward_re, forward_im, inverse_re, inverse_im = PLAIN.propagator_pair(
            real, imag
        )
        forward, inverse = np.exp(exponent), np.exp(-exponent)
        # Two ulp: the pair reaches the same numbers by a different route, so the
        # agreement is round-off in the last place and not identity.
        assert complex(forward_re, forward_im) == pytest.approx(forward, rel=2e-16)
        assert complex(inverse_re, inverse_im) == pytest.approx(inverse, rel=2e-16)


def test_the_real_cubic_reproduces_the_reference_kerr_pass():
    """``kerr_real`` is the reference Kerr pass's real form, sample by sample."""
    kerr = KerrCoefficients(3.5e-30, "real", 0.0, 0.0, 0.0, False, True)
    field = np.linspace(-2e10, 2e10, 64).reshape(1, 1, -1)
    wanted = passes.kerr_polarisation(field, kerr, 2.5e25)
    factor = kerr.coefficient * 2.5e25
    got = np.array(
        [PLAIN.kerr_real(value, value * value, factor) for value in field[0, 0]]
    )
    # A cube and a scaling, three roundings on either side, without cancellation.
    np.testing.assert_allclose(got, wanted[0, 0], rtol=6 * EPS, atol=0)
    assert np.max(np.abs(got)) > 0.0


def analytic(values, kerr, density):
    """Apply ``kerr_analytic`` over a chunk, forming the sums as a kernel would."""
    factor = kerr.coefficient * density
    modulus = np.sum(values.real**2 + values.imag**2, axis=-2)
    square = np.sum(values**2, axis=-2)
    return np.array(
        [
            [
                complex(
                    *PLAIN.kerr_analytic(
                        values[0, component, index].real,
                        values[0, component, index].imag,
                        modulus[0, index],
                        square[0, index].real,
                        square[0, index].imag,
                        factor,
                        kerr.self_factor,
                        kerr.conjugate_factor,
                    )
                )
                for index in range(values.shape[-1])
            ]
            for component in range(values.shape[-2])
        ]
    )


@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_analytic_cubic_agrees_with_the_reference_kerr_pass(n_pol):
    """``kerr_analytic`` is the analytic form of the same pass, to a bounded margin.

    Not bitwise, and it cannot be: the reference forms ``sum |a_q|^2`` as
    ``np.abs(values) ** 2``, a square root squared, and multiplies the scaled square by
    the conjugate with numpy's own complex product, which is not ``(ac - bd, ad + bc)``
    in plain arithmetic. A kernel that reproduced either would pay a square root per
    sample to be *less* accurate. Both are reported, not patched.

    Bound: two elementwise products and one sum of two, so 4 eps by Higham; measured
    1.18 eps at one component and 0.50 at two.
    """
    kerr = KerrCoefficients(3.5e-30, "analytic", 0.5, 0.25, 0.0, False, False)
    column = np.linspace(-2e10, 2e10, 64) + 1j * np.linspace(1e9, -1e9, 64)
    values = np.stack([column, 0.4 * column + 3e9j][:n_pol])[np.newaxis]
    wanted = passes.kerr_polarisation(values, kerr, 2.5e25)
    got = analytic(values, kerr, 2.5e25)
    scale = np.max(np.abs(wanted))
    assert scale > 0.0
    assert np.max(np.abs(got - wanted[0])) / scale < 4 * np.finfo(float).eps


def test_the_scan_bodies_reproduce_the_reference_plasma_scan(table, arrays):
    """The three running sums, one sample at a time, are the reference's scan."""
    compiled = plasma_coefficients(
        PlasmaResponse("Ar", table, preionisation=0.125), None, z_max=1.0
    )
    dt, preionisation = 1e-16, 0.125
    field = 1e10 * np.sin(np.linspace(0.0, 12.0, 256))
    column = field.reshape(1, 1, -1)
    rate = passes.ionisation_rate(column, table)
    fraction, free_current = passes.plasma_scan(
        column, rate, compiled, preionisation=preionisation, dt=dt
    )
    wanted = passes.plasma_current(
        column, rate, fraction, free_current, compiled, density=2.5e25, dt=dt
    )

    exposure = current = polarisation = 0.0
    fractions = np.empty_like(field)
    got = np.empty_like(field)
    integrand = previous_rate = previous_current = 0.0
    for index, value in enumerate(field):
        this_rate = PLAIN.rate_at(abs(value), arrays)
        if index:
            exposure += PLAIN.trapezoid(this_rate, previous_rate, dt)
        fractions[index] = this_fraction = PLAIN.fraction_at(exposure, preionisation)
        this_integrand = this_fraction * compiled.charge_mass_ratio * value
        if index:
            current += PLAIN.trapezoid(this_integrand, integrand, dt)
        scale = PLAIN.loss_scale(
            this_rate, this_fraction, value * value, compiled.ionisation_potential
        )
        this_total = current + scale * value
        if index:
            polarisation += PLAIN.trapezoid(this_total, previous_current, dt)
        got[index] = 2.5e25 * polarisation
        previous_rate, integrand, previous_current = (
            this_rate,
            this_integrand,
            this_total,
        )

    # The rate's one-ulp scalar/vector libm difference reaches the accumulated exposure
    # and its final exponential. Bound both that path and the three fixed-order current
    # scans by 4 n eps, following the operation-count rule.
    bound = 4 * field.size * np.finfo(float).eps
    np.testing.assert_allclose(fractions, fraction[0], rtol=bound, atol=0.0)
    scale = np.max(np.abs(wanted[0, 0]))
    assert scale > 0.0
    relative = np.max(np.abs(got - wanted[0, 0])) / scale
    assert relative < bound
    assert np.max(np.abs(got)) > 0.0


def test_the_loss_guard_returns_zero_where_the_divisor_vanishes():
    """The guard is the reference's ``where=``, not a division that makes a NaN."""
    assert PLAIN.loss_scale(1e15, 0.5, 0.0, 2.5e-18) == 0.0
    assert PLAIN.loss_scale(0.0, 0.5, 1e20, 2.5e-18) == 0.0
    assert PLAIN.loss_scale(1e15, 0.5, 1e20, 2.5e-18) == pytest.approx(
        2.5e-18 * 1e15 * 0.5 / 1e20
    )


def test_the_fraction_is_the_reference_s_and_saturates():
    """``fraction_at`` is ``1 - (1 - p) exp(-W)``, bounded by its own preionisation."""
    assert PLAIN.fraction_at(0.0, 0.25) == 0.25
    assert PLAIN.fraction_at(1e3, 0.25) == 1.0
    assert PLAIN.fraction_at(0.5, 0.25) == 1.0 - 0.75 * math.exp(-0.5)


def test_the_accumulation_is_the_reference_pass_on_one_element():
    """``accumulate`` is :func:`passes.accumulate` without the array."""
    total = np.array([[[[1.0 + 2.0j]]]])
    value = np.array([[[[0.5 - 0.25j]]]])
    wanted = passes.accumulate(total, value, 0.375)
    got = complex(
        PLAIN.accumulate(total[0, 0, 0, 0].real, value[0, 0, 0, 0].real, 0.375),
        PLAIN.accumulate(total[0, 0, 0, 0].imag, value[0, 0, 0, 0].imag, 0.375),
    )
    assert got == wanted[0, 0, 0, 0]


@pytest.mark.parametrize("field", [0.0, -2.7, 1.2 - 3.4j])
@pytest.mark.parametrize("split", [False, True])
def test_shared_absorber_preserves_existing_formula(field, split):
    rate, distance = 2.3, 0.7
    factor = math.exp((-0.5 * rate) * distance) if split else -0.5 * rate
    expected = factor * field
    for family in (PLAIN, HOST):
        actual = family.absorber_sample(field, rate, distance, split)
        # The same scalar order is exact on the host paths.
        assert actual == expected


@pytest.mark.parametrize("omega", [1.0, 1e12, 2.35e15])
@pytest.mark.parametrize("current,edge", [(0j, 0j), (2 - 3j, -5 + 7j)])
def test_held_current_body_is_the_active_frequency_identity(omega, current, edge):
    expected = current / (1j * omega) - edge / (1j * omega) ** 2
    # Two equivalent complex products, one addition and reciprocal rounding.
    allowance = 16 * EPS * (abs(current) / omega + abs(edge) / omega**2)
    for family in (PLAIN, HOST):
        actual = family.held_plasma_source(current, edge, 1.0 / omega)
        if allowance == 0:
            assert actual == 0j
        else:
            assert allowance > 0
            assert abs(actual - expected) <= allowance
