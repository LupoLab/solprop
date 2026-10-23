# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop import fourier
from solprop.constants import EPS0
from solprop.exceptions import GridError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.responses import (
    KerrResponse,
    ResponseTraits,
    _check_evaluate,
    _fundamental_cubic,
    thg_carrier,
)

# A narrowband pulse, deliberately long compared with the optical cycle: the field and
# envelope forms agree exactly only when the analytic signal recovers the envelope, and
# that fails at the level of the spectral overlap between the positive and negative
# frequency lobes.
CENTRE = 1030e-9
DURATION = 120e-15  # intensity FWHM
GRID = {
    "length": 0.1,
    "reference_wavelength": CENTRE,
    "wavelength_limits": (700e-9, 2.0e-6),
    "trange": 2e-12,
}

# The parity between the two grids is an algebraic identity spoiled only by the tail of
# the pulse spectrum that reaches below zero frequency. For a 120 fs pulse at 1030 nm
# that tail is far below double precision, so what is left is the round-off of the
# analytic signal's transform pair over n_to samples: a few times n*eps. Measured worst
# relative deviation for this case is 4.1e-15; the bound is 1e-10.
PARITY_RTOL = 1e-10


def gaussian_envelope(t, amplitude=1.0, chirp=0.0):
    """A chirped Gaussian envelope in the same units the responses take."""
    sigma = DURATION / (2.0 * np.sqrt(np.log(2.0)))
    return amplitude * np.exp(-((t / sigma) ** 2) * (1.0 - 1j * chirp))


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(**GRID)


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(**GRID, thg=True)


# ------------------------------------------------------------------------- traits


def test_traits_are_declared_as_the_design_requires():
    traits = KerrResponse(chi3=1e-24).traits
    assert traits == ResponseTraits(
        domain="pointwise",
        grids=("field", "envelope"),
        polarisation="vector_isotropic",
        z_profiles=("density",),
    )


def test_traits_are_the_same_whatever_the_third_harmonic_flag():
    assert KerrResponse(1e-24, thg=True).traits == KerrResponse(1e-24, False).traits


# ------------------------------------------------------------------- scalar forms


def test_field_response_with_third_harmonic_is_the_plain_cubic():
    field = np.array([[0.0, 1.0, -2.0, 3.0]])
    out = KerrResponse(chi3=2.0).evaluate(field, 3.0, kind="field")
    # rho eps0 chi3 E**3. The reference cubes in one step where the response squares
    # and then multiplies, so the two differ by at most one rounding of the product.
    assert np.allclose(out, (3.0 * EPS0 * 2.0) * field**3, rtol=2e-16, atol=0.0)


def test_envelope_response_without_third_harmonic_is_three_quarters_of_the_cubic():
    envelope = np.array([[1.0 + 2.0j, -3.0j]])
    out = KerrResponse(chi3=2.0, thg=False).evaluate(envelope, 3.0, kind="envelope")
    expected = (3.0 * EPS0 * 2.0) * 0.75 * np.abs(envelope) ** 2 * envelope
    # Round-off only: the reference groups the same three factors differently.
    assert np.allclose(out, expected, rtol=1e-15, atol=0.0)


def test_polarisation_scales_linearly_with_susceptibility_and_density():
    envelope = np.array([[1.0 + 0.5j, 2.0 - 1.0j]])
    base = KerrResponse(chi3=1.0, thg=False).evaluate(envelope, 1.0, kind="envelope")
    scaled = KerrResponse(chi3=3.0, thg=False).evaluate(envelope, 5.0, kind="envelope")
    assert np.allclose(scaled, 15.0 * base, rtol=1e-15, atol=0.0)


def test_polarisation_is_cubic_in_the_field_amplitude():
    field = np.array([[0.3, -1.2, 0.7]])
    kerr = KerrResponse(chi3=1.0)
    one = kerr.evaluate(field, 1.0, kind="field")
    two = kerr.evaluate(2.0 * field, 1.0, kind="field")
    assert np.allclose(two, 8.0 * one, rtol=1e-15, atol=0.0)


# ------------------------------------------------------------------- vector forms


def test_vector_form_reproduces_the_isotropic_expansion():
    # The compact (sum |a|^2, sum a^2) form must equal the expanded cross-term form
    # that Luna writes out (Nonlinear.jl:160-169). Both are exact algebra, so the only
    # difference is the order of the floating-point operations.
    rng = np.random.default_rng(20260906)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    y = rng.normal(size=64) + 1j * rng.normal(size=64)
    out = _fundamental_cubic(np.stack([x, y]))
    expected_x = 0.75 * (
        (np.abs(x) ** 2 + 2 / 3 * np.abs(y) ** 2) * x + 1 / 3 * np.conj(x) * y**2
    )
    expected_y = 0.75 * (
        (np.abs(y) ** 2 + 2 / 3 * np.abs(x) ** 2) * y + 1 / 3 * np.conj(y) * x**2
    )
    assert np.allclose(out[0], expected_x, rtol=1e-14, atol=0.0)
    assert np.allclose(out[1], expected_y, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("thg", [True, False])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_vector_reduces_to_scalar_for_linear_polarisation(kind, thg, envelope_grid):
    rng = np.random.default_rng(11)
    n_to = 32
    if kind == "field":
        column = rng.normal(size=n_to)
        carrier = None
    else:
        column = rng.normal(size=n_to) + 1j * rng.normal(size=n_to)
        carrier = thg_carrier(1.0e15, np.linspace(-1e-13, 1e-13, n_to)) if thg else None
    kerr = KerrResponse(chi3=1.5, thg=thg)
    scalar = kerr.evaluate(column[None, :], 2.0, kind=kind, carrier=carrier)
    vector = kerr.evaluate(
        np.stack([column, np.zeros_like(column)]), 2.0, kind=kind, carrier=carrier
    )
    # Every cross term is multiplied by the zero component, so the surviving
    # component is the scalar answer. It is not bitwise identical on a field grid
    # because the analytic signal is transformed in a batch of two rather than one.
    assert np.allclose(vector[0], scalar[0], rtol=1e-14, atol=0.0)
    # The empty component is exactly zero: every term in it carries a factor of zero.
    assert np.array_equal(vector[1], np.zeros_like(vector[1]))


# ------------------------------------------------------------- field/envelope parity


def test_field_and_envelope_responses_agree_for_a_narrowband_pulse(
    field_grid, envelope_grid
):
    t = field_grid.t_fine
    omega0 = field_grid.omega0
    envelope = gaussian_envelope(t, amplitude=3.0e8, chirp=0.4)
    field = np.real(envelope * np.exp(1j * omega0 * t))

    kerr = KerrResponse(chi3=1e-22, thg=False)
    on_field = kerr.evaluate(field[None, :], 1.0, kind="field")
    on_envelope = kerr.evaluate(envelope[None, :], 1.0, kind="envelope")
    demodulated = np.real(on_envelope * np.exp(1j * omega0 * t))

    worst = np.max(np.abs(on_field - demodulated)) / np.max(np.abs(on_field))
    assert 0.0 < worst < PARITY_RTOL


def test_the_field_response_recovers_the_envelope_response_exactly_in_algebra():
    # Independently of any pulse, the two cubics differ by exactly the carrier factor,
    # because the fundamental cubic is homogeneous of degree (2, 1) in (a, conj a).
    rng = np.random.default_rng(5)
    envelope = rng.normal(size=(1, 16)) + 1j * rng.normal(size=(1, 16))
    phase = np.exp(1j * rng.normal(size=16))
    assert np.allclose(
        _fundamental_cubic(envelope * phase),
        _fundamental_cubic(envelope) * phase,
        rtol=1e-14,
        atol=0.0,
    )


# ------------------------------------------------------------------ third harmonic


def band_fraction(spectrum, omega, low, high):
    """Fraction of the spectral power of `spectrum` inside [low, high]."""
    inside = (omega >= low) & (omega <= high)
    total = np.sum(np.abs(spectrum) ** 2)
    return float(np.sum(np.abs(spectrum[inside]) ** 2) / total)


def test_field_response_radiates_the_third_harmonic_only_when_declared(field_grid):
    t = field_grid.t_fine
    omega0 = field_grid.omega0
    envelope = gaussian_envelope(t, amplitude=3.0e8)
    field = np.real(envelope * np.exp(1j * omega0 * t))[None, :]

    omega = field_grid.omega_fine
    band = (2.5 * omega0, 3.5 * omega0)
    with_thg = fourier.forward(
        KerrResponse(1e-22, thg=True).evaluate(field, 1.0, kind="field"), kind="field"
    )[0]
    without = fourier.forward(
        KerrResponse(1e-22, thg=False).evaluate(field, 1.0, kind="field"), kind="field"
    )[0]
    # For a monochromatic field the cubic splits as (3/4) at the fundamental and
    # (1/4) at the third harmonic, so the harmonic band holds
    # (1/4)^2 / ((3/4)^2 + (1/4)^2) = 1/10 of the power exactly. A 120 fs pulse is
    # monochromatic to well inside 2 %.
    assert band_fraction(with_thg, omega, *band) == pytest.approx(0.1, rel=2e-2)
    # Without it the band holds only the numerical floor of the analytic signal.
    assert band_fraction(without, omega, *band) < 1e-20


def test_envelope_response_radiates_the_third_harmonic_only_when_declared(
    envelope_grid,
):
    t = envelope_grid.t_fine
    omega0 = envelope_grid.omega0
    envelope = gaussian_envelope(t, amplitude=3.0e8)[None, :]
    carrier = thg_carrier(omega0, t)

    omega = envelope_grid.omega_fine
    band = (2.5 * omega0, 3.5 * omega0)
    with_thg = fourier.forward(
        KerrResponse(1e-22, thg=True).evaluate(
            envelope, 1.0, kind="envelope", carrier=carrier
        ),
        kind="envelope",
    )[0]
    without = fourier.forward(
        KerrResponse(1e-22, thg=False).evaluate(envelope, 1.0, kind="envelope"),
        kind="envelope",
    )[0]
    # The same 1/10 split as on the field grid, reached by a different route: here
    # the harmonic is an explicit term rather than part of a real cubic.
    assert band_fraction(with_thg, omega, *band) == pytest.approx(0.1, rel=2e-2)
    assert band_fraction(without, omega, *band) < 1e-20


def test_thg_carrier_is_the_squared_carrier():
    t = np.linspace(-1e-14, 1e-14, 33)
    omega0 = 1.8e15
    assert np.allclose(thg_carrier(omega0, t), np.exp(1j * omega0 * t) ** 2)


def test_thg_carrier_rejects_a_non_positive_frequency():
    with pytest.raises(GridError, match="omega0 must be positive"):
        thg_carrier(0.0, np.zeros(4))


# ------------------------------------------------------------------- error paths


def test_evaluate_rejects_an_unknown_layout():
    with pytest.raises(GridError, match="kind must be"):
        _check_evaluate(np.zeros((1, 4)), "spectral", None, needs_carrier=False)


def test_evaluate_rejects_more_than_two_polarisation_components():
    with pytest.raises(GridError, match="n_pol must be 1 or 2"):
        KerrResponse(1.0).evaluate(np.zeros((3, 4)), 1.0, kind="field")


@pytest.mark.parametrize("shape", [(4,), (1, 2, 4)])
def test_evaluate_rejects_an_array_that_is_not_two_dimensional(shape):
    # Through the guard directly: the jaxtyping hook rejects the rank before the
    # library's own check is reached, exactly as for `fourier._check_kind`.
    with pytest.raises(GridError, match="field_t must have shape"):
        _check_evaluate(np.zeros(shape), "field", None, needs_carrier=False)


def test_field_layout_rejects_a_complex_array():
    with pytest.raises(GridError, match="real field"):
        KerrResponse(1.0).evaluate(np.zeros((1, 4), dtype=complex), 1.0, kind="field")


def test_envelope_layout_rejects_a_real_array():
    with pytest.raises(GridError, match="complex envelope"):
        KerrResponse(1.0).evaluate(np.zeros((1, 4)), 1.0, kind="envelope")


def test_envelope_third_harmonic_needs_a_carrier():
    with pytest.raises(GridError, match="needs the carrier"):
        KerrResponse(1.0, thg=True).evaluate(
            np.zeros((1, 4), dtype=complex), 1.0, kind="envelope"
        )


def test_a_carrier_is_rejected_where_it_would_be_ignored():
    with pytest.raises(GridError, match="only used on an envelope grid"):
        KerrResponse(1.0).evaluate(
            np.zeros((1, 4)), 1.0, kind="field", carrier=np.ones(4, dtype=complex)
        )


def test_a_carrier_of_the_wrong_length_is_rejected():
    # Through the guard directly, for the same reason as the rank check above.
    with pytest.raises(GridError, match="carrier has shape"):
        _check_evaluate(
            np.zeros((1, 4), dtype=complex),
            "envelope",
            np.ones(5, dtype=complex),
            needs_carrier=True,
        )


def test_a_negative_density_is_rejected():
    with pytest.raises(GridError, match="density must be non-negative"):
        KerrResponse(1.0).evaluate(np.zeros((1, 4)), -1.0, kind="field")
