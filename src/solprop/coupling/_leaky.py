# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The scalar capillary model's omitted leaky field, evaluated for the field coupled.

The scalar model projects onto :math:`J_0(u_mr/a)`; Marcatili's leaky HE1m core field
is :math:`J_0(u_m(1 - i\delta)r/a)`, :math:`\delta = |\nu_n|c/(\omega a)` (Marcatili
and Schmeltzer 1964, doi:10.1002/j.1538-7305.1964.tb04108.x). How far the two models'
launches differ depends on the field: the departure of mode m grows with
:math:`u_m\delta`, so a field with energy in high orders sees far more of it than a
Gaussian beam. No fixed bound covers every field usefully, so the discrepancy is
evaluated for the field being coupled, bin by bin (:func:`launch_discrepancy`): the
summed change of the power launched into each channel,
:math:`\sum_c\bigl||y_{L,c}|^2 - |y_{S,c}|^2\bigr|/P`, with :math:`y = G^{-1}o` the
least-squares amplitudes on each model's unit profiles and P the incident power, the
same for both: a beam's energy on its plane, a capillary exit's modal power
:math:`\sum|c|^2`. The launched amplitudes are what the run receives, and its modes
carry power :math:`|y|^2`; raw overlaps :math:`|\langle\hat v, f\rangle|^2` are not
modal powers where the leaky profiles are not orthogonal. A capillary exit into an
identical capillary launches the same amplitudes under both models, and this
discrepancy is zero there. It bounds the change of the total launched power too.

Every profile is a J0 on a disc, so every overlap has the closed form of Lommel's
integral, for complex wavenumbers and different radii alike (:func:`disc_overlap`).
A capillary exit (a modal source) is a finite sum of such profiles, and its
discrepancy is exact. An incident beam is sampled on a plane, and its leaky overlaps
come from the Taylor series of the conjugate leaky profile in :math:`\delta`,

.. math:: \overline{J_0(u(1 - i\delta)x)} = \sum_n \frac{(i\delta)^n}{n!}\,
   (ux)^nJ_0^{(n)}(ux),

whose terms are real, frequency-independent profiles (:func:`taylor_profiles`): the
beam's overlap with each is formed once on the plane, and every bin's leaky overlap
is a polynomial in its :math:`\delta`. Since :math:`|J_0^{(n)}| \le 1` the series'
remainder after N terms (n = 0 to N - 1) is at most :math:`(u\delta)^Ne^{u\delta}/N!`
on the core (:func:`taylor_tail`), so its overlap with the field is bounded by
Cauchy--Schwarz, and the launch discrepancy is bounded over every overlap it allows.

The closed forms are exact to rounding but for Lommel's midpoint branch, whose
relative error is below 2e-9 (``tests/coupling/test_leaky.py``); an overlap error
:math:`\eta` moves a launched power by at most :math:`2\eta + \eta^2` of it, so the
evaluation's numerical error is below 1e-8 of the incident power.
"""

import numpy as np
from scipy.special import comb, gammaln, jn_zeros, jv

from solprop.exceptions import CouplingError

_NEAR = 1e-4
"""Below this :math:`|(p - q)R|` Lommel's quotient cancels; there the integral is
taken at the midpoint wavenumber, whose error is second order, below 2e-9 relative."""

TAYLOR_TOLERANCE = 1e-12
"""Pointwise bound on the Taylor series' remainder relative to the profile's peak:
:func:`taylor_count` takes terms until it is met."""

TAYLOR_LIMIT = 200
r"""Most Taylor terms: within the envelope's range (orders to 24, delta to 1e-2)
:math:`u\delta \le 0.75` needs 14; a product that needs more is refused."""


def disc_overlap(p: np.ndarray, q: np.ndarray, radius: float) -> np.ndarray:
    r"""Return :math:`\int_0^R J_0(pr)J_0(qr)\,r\,dr` for broadcast wavenumbers.

    Lommel: :math:`R[qJ_0(pR)J_1(qR) - pJ_1(pR)J_0(qR)]/(q^2 - p^2)`, analytic in
    both, so complex wavenumbers are taken as they are. Where :math:`|p - q|R` is
    below 1e-4 the quotient cancels; there the integral is
    :math:`R^2[J_0^2 + J_1^2](mR)/2` at the midpoint m, exact to second order since
    the integral is symmetric in p and q.

    - ``p``, ``q``: wavenumbers, 1/m, broadcast together; real or complex.
    - ``radius``: the upper limit R, m.
    - Returns: m^2, complex, the broadcast shape.
    """
    p, q = np.broadcast_arrays(
        np.asarray(p, dtype=np.complex128), np.asarray(q, dtype=np.complex128)
    )
    pr, qr = p * radius, q * radius
    near = np.abs(pr - qr) < _NEAR
    safe = np.where(near, qr + 1.0, qr)
    lommel = (
        radius
        * radius
        * (safe * jv(0, pr) * jv(1, safe) - pr * jv(1, pr) * jv(0, safe))
        / (safe * safe - pr * pr)
    )
    middle = 0.5 * (pr + qr)
    equal = 0.5 * radius * radius * (jv(0, middle) ** 2 + jv(1, middle) ** 2)
    return np.where(near, equal, lommel)


def inner_products(
    p: np.ndarray, radius_p: float, q: np.ndarray, radius_q: float
) -> np.ndarray:
    r"""Return the normalised inner products of two sets of J0 disc profiles.

    :math:`\langle\hat v_a, \hat v_b\rangle = \int\overline{\hat v_a}\hat v_b\,dA`
    with :math:`v = J_0(pr)` inside its radius and zero outside, normalised to unit
    norm; :math:`\overline{J_0(pr)} = J_0(\bar pr)`.

    - ``p``: wavenumbers of the first set, 1/m, shape ``(n_omega, n_a)``.
    - ``q``: wavenumbers of the second set, 1/m, shape ``(n_omega, n_b)``.
    - ``radius_p``, ``radius_q``: each set's disc radius, m.
    - Returns: dimensionless, complex, shape ``(n_omega, n_a, n_b)``.
    """
    shared = min(radius_p, radius_q)
    raw = disc_overlap(np.conj(p)[:, :, None], q[:, None, :], shared)
    norm_p = np.real(disc_overlap(np.conj(p), p, radius_p))
    norm_q = np.real(disc_overlap(np.conj(q), q, radius_q))
    return raw / np.sqrt(norm_p[:, :, None] * norm_q[:, None, :])


def leaky_wavenumbers(u: np.ndarray, radius: float, delta: np.ndarray) -> np.ndarray:
    r"""Return :math:`u_m(1 - i\delta)/a` per bin and profile, 1/m.

    - ``u``: each profile's J0 zero, shape ``(n_modes,)``.
    - ``delta``: the leaky parameter per bin, shape ``(n_omega,)``; zero gives the
      scalar profile.
    - Returns: complex, shape ``(n_omega, n_modes)``.
    """
    return u[None, :] * (1.0 - 1j * delta[:, None]) / radius


def radial_order(u: np.ndarray) -> np.ndarray:
    """Return each profile's radial order m from its core wavenumber.

    A capillary HE1m profile has u the m-th J0 zero; an LP0m one has U between the
    (m-1)-th J1 zero and the m-th J0 zero, so above the (m-1)-th J0 zero. Either way m
    is one more than the number of J0 zeros below u.
    """
    zeros = jn_zeros(0, int(np.max(u) / np.pi) + 2)
    return 1 + np.searchsorted(zeros, np.asarray(u) * (1.0 - 1e-12), side="left")


def taylor_count(product: float) -> int:
    r"""Return how many Taylor terms bring the remainder below the tolerance.

    - ``product``: the largest :math:`u\delta` of the profiles and bins.
    - Returns: N such that :func:`taylor_tail` of N terms is at most
      :data:`TAYLOR_TOLERANCE`.
    - Refuses, with a CouplingError, a product that needs more than
      :data:`TAYLOR_LIMIT` terms.
    """
    count = 1
    while float(taylor_tail(np.array(product), count)) > TAYLOR_TOLERANCE:
        count += 1
        if count > TAYLOR_LIMIT:
            msg = (
                f"the leaky profile's Taylor series needs more than {TAYLOR_LIMIT} "
                f"terms at u delta = {product:.3g}"
            )
            raise CouplingError(msg)
    return count


def taylor_tail(product: np.ndarray, count: int) -> np.ndarray:
    r"""Return the pointwise bound on the leaky profile's Taylor remainder.

    After `count` terms, :math:`\sum_{n\ge N}(u\delta)^n/n! \le
    (u\delta)^Ne^{u\delta}/N!` with :math:`|x| \le 1` and :math:`|J_0^{(n)}| \le 1`.

    - ``product``: :math:`u\delta`, non-negative, any shape.
    - Returns: the bound, the same shape; formed in logarithms, so it neither
      overflows nor turns NaN.
    """
    product = np.asarray(product, dtype=np.float64)
    with np.errstate(divide="ignore"):
        logarithm = count * np.log(product) + product - gammaln(count + 1)
    return np.asarray(np.exp(logarithm))


def taylor_profiles(u: np.ndarray, x: np.ndarray, count: int) -> np.ndarray:
    r"""Return the Taylor series' profiles :math:`(ux)^nJ_0^{(n)}(ux)`.

    :math:`J_0^{(n)}(z) = 2^{-n}\sum_k(-1)^k\binom nkJ_{2k-n}(z)`.

    - ``u``: J0 zeros, shape ``(n_modes,)``.
    - ``x``: radius over the core radius, shape ``(n_x,)``.
    - Returns: shape ``(count, n_modes, n_x)``, real.
    """
    z = u[:, None] * x[None, :]
    # Each order once: J_-m = (-1)^m J_m, so orders 0 to count - 1 serve every term.
    bessel = [jv(order, z) for order in range(count)]

    def order(m: int) -> np.ndarray:
        """Return J_m(z) for any integer m from the cached non-negative orders."""
        return bessel[m] if m >= 0 else (-1) ** m * bessel[-m]

    terms = []
    for n in range(count):
        derivative = (
            sum((-1) ** k * comb(n, k) * order(2 * k - n) for k in range(n + 1))
            / 2.0**n
        )
        terms.append(z**n * derivative)
    return np.array(terms)


def leaky_norms(u: np.ndarray, radius: float, delta: np.ndarray) -> np.ndarray:
    r"""Return :math:`\int|J_0(u(1 - i\delta)r/a)|^2dA` per bin and profile, m^2.

    - ``delta``: shape ``(n_omega,)``.
    - Returns: shape ``(n_omega, n_modes)``, real.
    """
    p = leaky_wavenumbers(u, radius, delta)
    return 2.0 * np.pi * np.real(disc_overlap(np.conj(p), p, radius))


def launch_discrepancy(
    scalar: tuple[np.ndarray, np.ndarray],
    leaky: tuple[np.ndarray, np.ndarray],
    power: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    r"""Return the summed change of launched channel power per bin, bounded.

    :math:`\sum_c\bigl||y_{L,c}|^2 - |y_{S,c}|^2\bigr|/P` with :math:`y = G^{-1}o`.
    Each leaky overlap is known to within `error`; the leaky amplitudes then move by
    at most :math:`\epsilon = \|e\|/\lambda_{\min}(G_L)`, and the sum by at most
    :math:`2\|y_L\|\epsilon + \epsilon^2` (Cauchy--Schwarz), which is added.

    - ``scalar``, ``leaky``: each model's channel overlaps with the field, shape
      ``(n_omega, n_channels)``, and channel Gram, shape
      ``(n_omega, n_channels, n_channels)``.
    - ``power``: the incident power P per bin, in the overlaps' squared units, shape
      ``(n_omega,)``; a bin without it has no discrepancy.
    - ``error``: bound on each leaky overlap's error, shape ``(n_omega, n_channels)``.
    - Returns: dimensionless, shape ``(n_omega,)``.
    """
    live = power > 0.0
    result = np.zeros(power.shape)
    if not np.any(live):
        return result
    (o_s, g_s), (o_l, g_l) = scalar, leaky
    y_s = np.linalg.solve(g_s[live], o_s[live][:, :, None])[:, :, 0]
    y_l = np.linalg.solve(g_l[live], o_l[live][:, :, None])[:, :, 0]
    change = np.sum(np.abs(np.abs(y_l) ** 2 - np.abs(y_s) ** 2), axis=1)
    smallest = np.linalg.eigvalsh(g_l[live])[:, 0]
    shift = np.sqrt(np.sum(error[live] ** 2, axis=1)) / smallest
    slack = 2.0 * np.linalg.norm(y_l, axis=1) * shift + shift**2
    result[live] = (change + slack) / power[live]
    return result
