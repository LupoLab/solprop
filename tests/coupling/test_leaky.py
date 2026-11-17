# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The closed forms behind the capillary envelope's per-field evaluation.

Each is checked against an independent quadrature or a direct evaluation;
``tests/coupling/test_envelopes.py`` checks the whole evaluation through ``couple``.
"""

import math

import numpy as np
import pytest
from scipy.special import jn_zeros, jv

from solprop.coupling import _leaky as leaky
from solprop.coupling.envelopes import FieldEnvelope
from solprop.coupling.models import _lp_parameter
from solprop.exceptions import CouplingError

RADIUS = 75e-6
ZEROS = jn_zeros(0, 24)


def integral(p, q, radius):
    """Return int_0^R J0(p r) J0(q r) r dr on a 1000-node Gauss--Legendre rule.

    The integrand is smooth and oscillates fewer than 30 times; the rule integrates
    it to rounding, independently of Lommel's closed form.
    """
    x, w = np.polynomial.legendre.leggauss(1000)
    r = 0.5 * radius * (x + 1.0)
    return complex(np.sum(0.5 * radius * w * jv(0, p * r) * jv(0, q * r) * r))


@pytest.mark.parametrize(
    ("p", "q", "rel"),
    [
        # Lommel's quotient, real and complex, low and high order: to rounding.
        (ZEROS[0] / RADIUS, ZEROS[3] / RADIUS, 1e-12),
        (ZEROS[0] * (1 - 1e-2j) / RADIUS, ZEROS[0] * (1 + 1e-2j) / RADIUS, 1e-12),
        (ZEROS[23] * (1 - 1e-2j) / RADIUS, ZEROS[22] / RADIUS, 1e-12),
        (
            ZEROS[4] * (1 - 3e-3j) / (1.3 * RADIUS),
            ZEROS[2] * (1 - 1e-2j) / RADIUS,
            1e-12,
        ),
        # The midpoint branch, below |p - q| R = 1e-4: second order, about 1e-9.
        (ZEROS[5] * (1 + 2e-5) / RADIUS, ZEROS[5] / RADIUS, 1e-8),
        (ZEROS[0] * (1 - 3e-5j) / RADIUS, ZEROS[0] / RADIUS, 1e-8),
        (ZEROS[0] / RADIUS, ZEROS[0] / RADIUS, 1e-12),
    ],
)
def test_the_disc_overlap_is_lommels_integral(p, q, rel):
    got = complex(leaky.disc_overlap(np.array(p), np.array(q), RADIUS))
    assert got == pytest.approx(integral(p, q, RADIUS), rel=rel)


def test_inner_products_are_normalised_and_hermitian():
    """Unit diagonal, conjugate symmetric; across radii only the smaller disc counts."""
    delta = np.array([1e-2, 2e-3])
    p = leaky.leaky_wavenumbers(ZEROS[:6], RADIUS, delta)
    gram = leaky.inner_products(p, RADIUS, p, RADIUS)
    np.testing.assert_allclose(np.diagonal(gram, axis1=1, axis2=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(gram, np.conj(np.swapaxes(gram, 1, 2)), atol=1e-15)
    q = leaky.leaky_wavenumbers(ZEROS[:2], 0.6 * RADIUS, delta)
    cross = leaky.inner_products(p, RADIUS, q, 0.6 * RADIUS)
    norm_p = integral(np.conj(p[0, 1]), p[0, 1], RADIUS).real
    norm_q = integral(np.conj(q[0, 0]), q[0, 0], 0.6 * RADIUS).real
    expected = integral(np.conj(p[0, 1]), q[0, 0], 0.6 * RADIUS) / math.sqrt(
        norm_p * norm_q
    )
    assert complex(cross[0, 1, 0]) == pytest.approx(expected, rel=1e-11)
    norms = leaky.leaky_norms(ZEROS[:6], RADIUS, delta)
    assert norms[0, 1] == pytest.approx(2 * math.pi * norm_p, rel=1e-12)


@pytest.mark.parametrize("product", [1e-3, 0.1, 0.75, 2.0])
def test_the_taylor_series_meets_its_remainder_bound(product):
    """The truncated series of conj J0(u (1 - i delta) x) is within its bound
    everywhere, and the counted series within the tolerance."""
    u = ZEROS[[0, 9, 23]]
    delta = product / u[-1]
    x = np.linspace(0.0, 1.0, 401)
    exact = np.conj(jv(0, np.outer(u * (1 - 1j * delta), x)))
    for count in (2, 5, leaky.taylor_count(product)):
        terms = leaky.taylor_profiles(u, x, count)
        series = np.einsum(
            "n,nmx->mx",
            (1j * delta) ** np.arange(count)
            / np.array([math.factorial(n) for n in range(count)]),
            terms,
        )
        bound = leaky.taylor_tail(u * delta, count)
        assert np.all(np.abs(series - exact) <= bound[:, None] * (1 + 1e-9) + 1e-15)
    assert leaky.taylor_tail(np.array(product), leaky.taylor_count(product)) <= (
        leaky.TAYLOR_TOLERANCE
    )


def test_the_radial_order_of_capillary_and_lp_profiles():
    """HE1m's u is the m-th J0 zero; LP0m's U lies just below it."""
    np.testing.assert_array_equal(leaky.radial_order(ZEROS), np.arange(1, 25))
    for m in (1, 2, 3):
        u, _ = _lp_parameter(10.0, m)
        assert leaky.radial_order(np.array([u]))[0] == m


def test_the_launch_discrepancy_is_exact_without_error_and_bounds_with_it():
    """With no error it is the change of launched power; with an error it bounds every
    overlap the error allows."""
    rng = np.random.default_rng(3)
    n = 5
    o_s = rng.normal(size=(1, n)) + 1j * rng.normal(size=(1, n))
    o_l = o_s + 0.05 * (rng.normal(size=(1, n)) + 1j * rng.normal(size=(1, n)))
    base = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    g_l = (np.eye(n) + 0.02 * (base + base.conj().T))[None]
    g_s = np.eye(n)[None]
    power = np.array([40.0])

    def direct(overlap):
        leaky = np.abs(np.linalg.solve(g_l[0], overlap)) ** 2
        return np.sum(np.abs(leaky - np.abs(o_s[0]) ** 2)) / power[0]

    exact = leaky.launch_discrepancy((o_s, g_s), (o_l, g_l), power, np.zeros((1, n)))
    assert exact[0] == pytest.approx(direct(o_l[0]), rel=1e-12)
    # The same overlaps and Gram under both models launch the same powers.
    same = leaky.launch_discrepancy((o_l, g_l), (o_l, g_l), power, np.zeros((1, n)))
    assert same[0] == 0.0
    error = np.full((1, n), 0.02)
    bounded = leaky.launch_discrepancy((o_s, g_s), (o_l, g_l), power, error)
    for _ in range(200):
        shift = rng.normal(size=n) + 1j * rng.normal(size=n)
        shift *= 0.02 * rng.uniform() / np.abs(shift)
        assert direct(o_l[0] + shift) <= bounded[0]
    # A bin with no incident power has no discrepancy.
    empty = leaky.launch_discrepancy(
        (o_s, g_s), (o_l, g_l), np.zeros(1), np.zeros((1, n))
    )
    assert empty[0] == 0.0


def test_the_taylor_tail_neither_overflows_nor_runs_on():
    """Formed in logarithms: finite where the factorial would overflow; a product
    needing more than the limit of terms is refused, never accepted as NaN."""
    tail = leaky.taylor_tail(np.array([0.0, 0.5, 60.0]), 171)
    assert np.all(np.isfinite(tail))
    assert tail[0] == 0.0
    assert leaky.taylor_count(0.75) == 14
    with pytest.raises(CouplingError, match="more than 200 terms"):
        leaky.taylor_count(60.0)


def test_a_field_envelope_allows_the_evaluated_discrepancy():
    envelope = FieldEnvelope("x", "capillary", {"leaky": (0.0, 1e-2)}, 1e-8, "r")
    assert envelope.allowance({"field": 2e-3}) == pytest.approx(2e-3 + 1e-8)
    assert envelope.covers({"leaky": 5e-3}) is None
    assert "outside" in str(envelope.covers({"leaky": 2e-2}))
