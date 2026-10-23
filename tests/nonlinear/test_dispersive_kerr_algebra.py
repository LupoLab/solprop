# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the algebra of SEP 6's dispersive electronic cubic before it is implemented.

Phase 0 of SEP 6 derives FFT forms of two frequency-dependent cubics, the grouped gas
polynomial and the symmetric factorisation, with the per-leg bound of section 4.10, on
field and envelope frames, for the full cubic and for the number-conserving sector.
These tests hold the forms in ``tests/dispersive_kerr_oracle.py`` to an explicit triplet
sum and to the invariants the proposal relies on. The coefficients are synthetic
(``A = 0.05``, ``B = 0.02``, ``B' = -0.003`` per ``(k / 10)**2``, SEP 6 section 4.9):
they check algebra, not material values.

Tolerances. Every comparison is between two evaluations of the same finite sum in
double precision: at most ``24**3`` triplet products, or a few FFTs of 96 to 128
points. The worst relative error measured is 1.3e-14, against the triplet sum; the
work, photon and envelope comparisons are at or below 2e-15. 1e-12 leaves a factor of
about 80 for the summation order of another platform.
"""

import numpy as np
import pytest

from dispersive_kerr_oracle import (
    Polynomial,
    Symmetric,
    analytic,
    direct_sum,
    isotropic,
    polynomial_cubic,
    polynomial_envelope,
    symmetric_cubic,
    symmetric_envelope,
)
from solprop.constants import EPS0
from solprop.nonlinear.responses import KerrResponse

RTOL = 1e-12
N = 128
BAND = 12  # occupied bins 1..BAND: the cubic reaches 3 * BAND < N / 2, no aliasing
POLYNOMIAL = Polynomial(a=0.05, b=0.02, b_prime=-0.003)
BOUNDED = Polynomial(a=0.05, b=0.02, b_prime=-0.003, d_cap=(7 / 10) ** 2)
MODELS = {
    "polynomial": (polynomial_cubic, POLYNOMIAL),
    "polynomial bounded": (polynomial_cubic, BOUNDED),
    "symmetric": (symmetric_cubic, Symmetric(POLYNOMIAL)),
    "symmetric bounded": (symmetric_cubic, Symmetric(BOUNDED)),
}
ENVELOPES = {
    "polynomial": (polynomial_envelope, BOUNDED),
    "symmetric": (symmetric_envelope, Symmetric(BOUNDED)),
}
T = 2 * np.pi * np.arange(N) / N


def multitone(seed: int, n_pol: int = 1, band: int = BAND) -> np.ndarray:
    """Return a real field with random amplitudes on bins 1..band, no DC."""
    rng = np.random.default_rng(seed)
    spectrum = np.zeros((n_pol, N), complex)
    amplitude = rng.normal(size=(n_pol, band)) + 1j * rng.normal(size=(n_pol, band))
    spectrum[:, 1 : band + 1] = amplitude
    spectrum[:, -band:] = amplitude[:, ::-1].conj()
    field = np.real(np.fft.ifft(spectrum, axis=-1)) * N / band
    return field if n_pol > 1 else field[0]


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def derivative(field: np.ndarray) -> np.ndarray:
    k = np.fft.fftfreq(N, 1.0 / N)
    return np.real(np.fft.ifft(1j * k * np.fft.fft(field, axis=-1), axis=-1))


def work(field: np.ndarray, polarisation: np.ndarray) -> float:
    """Return the normalised net work ``sum E' P``, zero for a lossless response."""
    rate = derivative(field)
    return float(
        abs(np.sum(rate * polarisation))
        / (np.linalg.norm(rate) * np.linalg.norm(polarisation))
    )


def photon_rate(field: np.ndarray, polarisation: np.ndarray) -> float:
    """Return the normalised photon-number rate ``sum_k>0 Im(U* P_k)``."""
    u = np.fft.fft(analytic(field), axis=-1)
    p = np.fft.fft(analytic(polarisation), axis=-1)
    return float(abs(np.sum(np.imag(u.conj() * p))) / np.sum(np.abs(u) * np.abs(p)))


@pytest.mark.parametrize("sector", ["full", "conserving"])
@pytest.mark.parametrize("name", list(MODELS))
def test_fft_form_matches_the_triplet_sum(name, sector):
    """The grouped FFT forms equal the explicit sum, bound and sector included."""
    cubic, model = MODELS[name]
    field = multitone(1)
    expected = direct_sum(field, model.kernel, sector=sector)
    assert relative(cubic(field, model, sector=sector), expected) < RTOL


@pytest.mark.parametrize("sector", ["full", "conserving"])
@pytest.mark.parametrize("name", list(MODELS))
def test_isotropic_vector_form_matches_the_triplet_sum(name, sector):
    """The scalar kernel times the isotropic tensor, for a two-component field."""
    cubic, model = MODELS[name]
    field = multitone(2, n_pol=2)
    expected = direct_sum(field, model.kernel, sector=sector)
    assert relative(cubic(field, model, form=isotropic, sector=sector), expected) < RTOL


@pytest.mark.parametrize("sector", ["full", "conserving"])
@pytest.mark.parametrize("name", list(MODELS))
def test_both_sectors_do_no_net_work(name, sector):
    """Section 4.9's lossless identity, with the bound and in each sector."""
    cubic, model = MODELS[name]
    field = multitone(3)
    assert work(field, cubic(field, model, sector=sector)) < RTOL
    vector = multitone(4, n_pol=2)
    assert work(vector, cubic(vector, model, form=isotropic, sector=sector)) < RTOL


@pytest.mark.parametrize("name", list(MODELS))
def test_only_the_conserving_sector_conserves_photon_number(name):
    """Two-in, two-out quartets conserve photons; the harmonic quartets do not."""
    cubic, model = MODELS[name]
    field = multitone(5)
    assert photon_rate(field, cubic(field, model, sector="conserving")) < RTOL
    assert photon_rate(field, cubic(field, model, sector="full")) > 1e-3


def test_output_only_weighting_and_the_legacy_partial_cubic_do_work():
    """Negative checks: section 4.3's output multiplier, the field-grid thg=False rule.

    The output-only form multiplies the constant cubic by the Kerr diagonal at the
    output. The package's field-grid ``thg=False`` cubic is ``Re[(3/4)|u|^2 u]``
    without the projection onto positive frequencies; on a band this broad it keeps
    part of the one-into-three exchange without its partner.
    """
    field = multitone(6)
    cube = field**3
    k = np.fft.fftfreq(N, 1.0 / N)
    output_only = np.real(
        np.fft.ifft(np.fft.fft(cube) * POLYNOMIAL.kerr(POLYNOMIAL.weight(k)))
    )
    assert work(field, output_only) > 1e-3
    legacy = KerrResponse(chi3=1.0, thg=False).evaluate(
        field[None, :], 1.0, kind="field"
    )
    assert work(field, legacy[0] / EPS0) > 1e-3


def test_constant_limit_is_the_package_cubic():
    """With no dispersion both forms are the existing cubics.

    The full cubic is ``KerrResponse(thg=True)`` on a field grid. The conserving sector
    is the package's ``thg=False`` rule wherever ``|u|^2 u`` has no negative-frequency
    content (a band narrower than an octave), and differs from it beyond.
    """
    flat = Polynomial(a=0.0, b=0.0, b_prime=0.0)
    field = multitone(7)
    full = KerrResponse(chi3=1.0, thg=True).evaluate(field[None, :], 1.0, kind="field")
    assert relative(polynomial_cubic(field, flat), full[0] / EPS0) < RTOL
    assert relative(symmetric_cubic(field, Symmetric(flat)), full[0] / EPS0) < RTOL
    narrow = np.cos(9 * T) + 0.7 * np.sin(11 * T + 0.3) + 0.4 * np.cos(12 * T)
    legacy = KerrResponse(chi3=1.0, thg=False).evaluate(
        narrow[None, :], 1.0, kind="field"
    )
    conserving = polynomial_cubic(narrow, flat, sector="conserving")
    assert relative(conserving, legacy[0] / EPS0) < RTOL
    broad = KerrResponse(chi3=1.0, thg=False).evaluate(
        field[None, :], 1.0, kind="field"
    )
    assert (
        relative(polynomial_cubic(field, flat, sector="conserving"), broad[0] / EPS0)
        > 1e-3
    )


def test_monochromatic_field_takes_the_kerr_and_harmonic_quartets():
    """``cos(k t)`` gives ``(3/4) gamma_Kerr`` at ``k``, ``(1/4) gamma_THG`` at ``3k``.

    The polynomial evaluates the two processes at their own sums: Kerr ``S2 = 4d``,
    ``S4 = 4d^2``; THG ``S2 = 12d``, ``S4 = 84d^2``. The factorisation takes the Kerr
    diagonal exactly and the harmonic as ``g(3k) g(k)^3``.
    """
    k0 = 9
    d = POLYNOMIAL.weight(np.array(k0))
    field = np.cos(k0 * T)
    spectrum = np.fft.fft(polynomial_cubic(field, POLYNOMIAL)) * 2 / N
    kerr = 1 + 4 * 0.05 * d + (16 * 0.02 - 4 * 0.003) * d**2
    harmonic = 1 + 12 * 0.05 * d + 144 * 0.02 * d**2 - 84 * 0.003 * d**2
    assert spectrum[k0].real == pytest.approx(0.75 * kerr, rel=RTOL)
    assert spectrum[3 * k0].real == pytest.approx(0.25 * harmonic, rel=RTOL)
    model = Symmetric(POLYNOMIAL)
    spectrum = np.fft.fft(symmetric_cubic(field, model)) * 2 / N
    g = model.g(np.array([k0, 3 * k0]))
    assert spectrum[k0].real == pytest.approx(0.75 * kerr, rel=RTOL)
    assert spectrum[3 * k0].real == pytest.approx(0.25 * g[1] * g[0] ** 3, rel=RTOL)


@pytest.mark.parametrize("sector", ["full", "conserving"])
@pytest.mark.parametrize("name", list(ENVELOPES))
def test_envelope_frame_is_the_field_frame_whatever_the_carrier(name, sector):
    """Multipliers at absolute frequency make the result independent of the carrier.

    The envelope ``A = u exp(-i m0 t)`` is evaluated with every multiplier at the
    absolute bin and the harmonic terms carried by ``exp(+-2 i m0 t)``; it must equal
    the field frame's analytic polarisation for any ``m0``, scalar and vector.
    """
    envelope_form, model = ENVELOPES[name]
    field_form = polynomial_cubic if name == "polynomial" else symmetric_cubic
    for field, form in ((multitone(8), None), (multitone(9, n_pol=2), isotropic)):
        kwargs = {"form": form} if form is not None else {}
        reference = analytic(field_form(field, model, sector=sector, **kwargs))
        for carrier in (5, 8):
            envelope = analytic(field) * np.exp(-1j * carrier * T)
            value = envelope_form(envelope, carrier, model, sector=sector, **kwargs)
            assert relative(value * np.exp(1j * carrier * T), reference) < RTOL


def test_band_projection_on_every_leg_keeps_the_identity():
    """Section 4.5's band rule: project inputs and output alike, or lose the identity.

    The projection to ``|k| <= 8``, inside the field's band of 12, applied to the field
    entering every product and to the polarisation leaving it restricts a symmetric
    kernel and does no net work. Applied to the output alone, the inputs above the band
    still feed outputs inside it, the kernel is no longer symmetric, and it does.
    """
    k = np.fft.fftfreq(N, 1.0 / N)
    band = np.abs(k) <= 8

    def project(series):
        return np.real(np.fft.ifft(np.fft.fft(series) * band))

    field = multitone(10)
    inside = project(field)
    both = project(polynomial_cubic(inside, BOUNDED))
    assert work(inside, both) < RTOL
    output_only = project(polynomial_cubic(field, BOUNDED))
    assert work(field, output_only) > 1e-3


def test_leg_bound_leaves_every_quartet_within_the_fitted_sums_exact():
    """No quartet with ``S2 <= S2_max`` has a leg beyond ``sqrt(3 S2_max / 4)``.

    The four signed legs sum to zero, so each is minus the sum of the other three and
    Cauchy-Schwarz gives ``k_j**2 <= (3/4) S2``. A bound at ``d_cap = (3/4) S2_max``
    therefore changes no quartet within the fitted sums.
    """
    s2_max = 4 * (12 / 10) ** 2
    bounded = Polynomial(a=0.05, b=0.02, b_prime=-0.003, d_cap=0.75 * s2_max)
    rng = np.random.default_rng(11)
    inputs = rng.integers(-40, 41, size=(200_000, 3))
    legs = np.column_stack([inputs.sum(1), inputs])
    inside = (POLYNOMIAL.weight(legs).sum(1) <= s2_max) & np.all(legs != 0, axis=1)
    assert inside.sum() > 1000
    # Structural: a cap that does not bind is never applied, so the kernels match.
    np.testing.assert_array_equal(
        bounded.kernel(legs[inside]), POLYNOMIAL.kernel(legs[inside])
    )
    assert np.any(bounded.kernel(legs[~inside]) != POLYNOMIAL.kernel(legs[~inside]))
