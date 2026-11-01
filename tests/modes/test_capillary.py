# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import logging
import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import cast

import mpmath as mp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from witness_materials import ONE_BAR_0C, borzsonyi, ermolov_helium

from bounds import EPS, assert_within
from solprop.constants import EPS0, C
from solprop.exceptions import MaterialRangeWarning, ModeError
from solprop.modes import (
    EVANESCENT_INDEX_FLOOR,
    MarcatiliMode,
    MarcatiliTerms,
    Mode,
    ModeKind,
    bessel_zero,
)
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.base import alpha, beta, beta1, beta2, zdw
from solprop.physdata import Gas
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.mixtures import Mixture
from solprop.profiles import (
    Constant,
    DensityPath,
    FromCallable,
    Linear,
    PressureGradient,
)

MODELS = ["full", "reduced"]
KINDS = ["HE", "TE", "TM"]

# 800 nm, the wavelength every Luna capillary test uses.
OMEGA_800 = 2.0 * math.pi * C / 800e-9

# Gate for the contour derivative below. Measured worst case over every family, both
# models, a fixed and a tapered radius, and 300 nm to 2 um: 2.5e-12 relative. The
# floor is rounding in the sampled values amplified by the 1 / radius of Cauchy's
# formula, not anything about the model, which is exact.
CONTOUR_TOLERANCE = 1e-11


def omega_of(wavelength):
    return 2.0 * math.pi * C / np.asarray(wavelength)


def reference_mode(model="full", **kwargs):
    """A mode with a constant cladding and Börzsönyi's argon, both fixed by the test.

    The gas index is supplied (``witness_materials``), so the measured gates below stay
    those of the fit they were measured on, whatever optmat's argon becomes.
    """
    defaults = {"cladding": 1.45, "model": model, "susceptibility": borzsonyi("Ar")}
    return MarcatiliMode(50e-6, "Ar", 2.0, **(defaults | kwargs))


def contour_derivative(mode, omega, core_radius, density, radius=0.1):
    """Cauchy's formula for dn/d(omega) on a small circle, to machine precision.

    The effective index is holomorphic in omega -- powers, a square root and a
    rational cladding term -- so the trapezoid rule on a circle around the point
    converges spectrally and, unlike a finite difference, involves no subtractive
    cancellation. It pins the derivative to about 1e-13 relative, three orders better
    than any difference of the same function, and it needs no second implementation of
    the algebra: it only asks whether the returned derivative really is the derivative
    of the returned value.

    The sampled terms are held fixed and only the frequency in them is moved into the
    complex plane, so this checks the calculus of the model itself. The chain rule
    through the material models is covered by the finite-difference ladder and by the
    symbolic witness.

    The circle is *large* -- a tenth of the frequency -- which is the opposite of what
    a finite difference wants and is the whole point. The trapezoid rule's truncation
    error falls like the radius to the power of the node count, so at 64 nodes it is
    gone by any radius; what is left is rounding in the sampled values amplified by
    the 1 / radius in the formula, which a *small* circle makes worse. Measured over
    every family and both models: 1e-12 relative at a hundredth of the frequency,
    1e-13 at a tenth. See CONTOUR_TOLERANCE.
    """
    n_points = 64
    terms = mode.sampled_terms(omega)
    # Only the frequency inside the terms is moved, so the material models must be
    # frequency independent for the continuation to be the same function.
    assert np.all(terms.dcore_susceptibility_domega == 0.0)
    assert np.all(terms.dcladding_term_domega == 0.0)
    step = radius * omega
    total = np.zeros_like(np.asarray(omega), dtype=complex)
    for index in range(n_points):
        offset = step * np.exp(2j * math.pi * index / n_points)
        shifted = replace(mode.sampled_terms(omega), omega=omega + offset)
        value = mode._model_values(shifted, core_radius, density)[0]
        total = total + value / offset
    return total / n_points


# ----------------------------------------------------------------- Bessel zeros


def test_bessel_zero_matches_known_values():
    # j_{0,1} and j_{1,1}, the two zeros every capillary paper quotes.
    assert bessel_zero(0, 1) == pytest.approx(2.4048255576957728, rel=1e-14)
    assert bessel_zero(1, 1) == pytest.approx(3.8317059702075125, rel=1e-14)


def test_bessel_zero_folds_negative_orders():
    """J_-p = (-1)^p J_p for integer p, so the zeros coincide; Luna relies on it."""
    assert bessel_zero(-2, 1) == bessel_zero(2, 1)
    assert MarcatiliMode(125e-6, n=-1, m=1).u_nm == pytest.approx(5.1356, rel=1e-4)


def test_bessel_zero_rejects_a_non_positive_index():
    with pytest.raises(ModeError, match="positive integer"):
        bessel_zero(0, 0)


# ------------------------------------------------------------- the construction


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind", KINDS)
def test_a_mode_of_every_family_satisfies_the_protocol(kind, model):
    n = 0 if kind in ("TE", "TM") else 1
    mode = MarcatiliMode(125e-6, "He", 1.0, kind=kind, n=n, model=model)
    assert isinstance(mode, Mode)
    assert mode.dimlimits() == ("polar", (0.0, 0.0), (125e-6, 2.0 * math.pi))
    assert mode.scale_invariant
    assert mode.z_constant
    assert mode.azimuthal_order >= 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "HE", "n": 0}, "non-zero azimuthal index"),
        ({"kind": "TE", "n": 1}, "needs n = 0"),
        ({"kind": "TM", "n": 2}, "needs n = 0"),
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"loss": -1.0}, "must not be negative"),
        ({"m": 0}, "positive integer"),
    ],
)
def test_a_bad_mode_is_refused(kwargs, message):
    with pytest.raises(ModeError, match=message):
        MarcatiliMode(125e-6, "He", 1.0, **kwargs)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("kind", "EH", "kind must be"),
        ("model", "exact", "model must be"),
        ("loss", "high", "loss must be"),
    ],
)
def test_a_value_outside_a_declared_literal_is_refused(attribute, value, message):
    """The production path, where no runtime type checker stands in front of it.

    Under the test session's beartype hook these arguments never reach __post_init__:
    the Literal annotations reject them first, which is the point of having them. In
    production nothing checks, and without this guard 'EH' would fall through to the
    TM branch of the cladding term and give a wrong answer silently. So the guard is
    exercised the only way it can be, by driving __post_init__ directly.
    """
    mode = MarcatiliMode(125e-6, "He", 1.0)
    object.__setattr__(mode, attribute, value)
    with pytest.raises(ModeError, match=message):
        mode.__post_init__()


def test_a_non_positive_radius_is_refused():
    with pytest.raises(ModeError, match="core radius must be positive"):
        MarcatiliMode(0.0, "He", 1.0)
    tapered = MarcatiliMode(Linear(125e-6, -1e-6, 1.0), "He", 1.0)
    with pytest.raises(ModeError, match="core radius must be positive"):
        tapered.neff(np.array([OMEGA_800]), z=1.0)


def test_repr_names_the_mode_its_fill_and_its_model():
    text = repr(MarcatiliMode(125e-6, "He", 1.0, n=2, m=3, model="reduced"))
    assert "HE23" in text
    assert "125 um" in text
    assert "He" in text
    assert "reduced" in text
    assert "evacuated" in repr(MarcatiliMode(125e-6))


def test_a_frequency_at_or_below_zero_is_refused():
    """A hollow capillary has no mode at DC, and a NaN there would poison Phi."""
    mode = MarcatiliMode(125e-6, "He", 1.0)
    with pytest.raises(ModeError, match="no mode at zero frequency"):
        mode.neff(np.array([0.0]))
    with pytest.raises(ModeError, match="no mode at zero frequency"):
        mode.dneff_domega(np.array([-1.0]))
    with pytest.raises(ModeError, match="must be finite"):
        mode.neff(np.array([np.nan]))


# ------------------------------------------------------------------- the models


@pytest.mark.parametrize("model", MODELS)
def test_loss_is_loss(model):
    """The trap: a sign slip here turns absorption into gain, silently."""
    mode = MarcatiliMode(125e-6, "He", 1.0, model=model)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 32))
    index = mode.neff(omega)
    assert np.all(np.imag(index) > 0.0)
    assert np.all(alpha(mode, omega) > 0.0)


@pytest.mark.parametrize("model", MODELS)
def test_loss_scales_as_the_inverse_cube_of_the_radius(model):
    """The reduced model's loss term is exactly a^-3, and the full model follows it."""
    omega = np.array([OMEGA_800])
    small = MarcatiliMode(125e-6, "He", 1.0, model=model)
    large = MarcatiliMode(250e-6, "He", 1.0, model=model)
    ratio = float(alpha(small, omega)[0] / alpha(large, omega)[0])
    # Exact for the reduced model; the full model differs by the next order in 1/(ka),
    # which is 3e-6 of the value at these radii.
    assert ratio == pytest.approx(8.0, rel=1e-5)


def test_the_two_models_agree_where_the_expansion_is_good():
    """The reduced model is the leading term of the full one in 1/(ka).

    At 125 um and 800 nm, (u c / (omega a))^2 is 3e-6, so the neglected term is of
    that order squared relative to the refractivity -- the two must agree to about
    1e-5 relative in n - 1, and they do to 1.4e-5.
    """
    omega = omega_of(np.linspace(400e-9, 1000e-9, 32))
    full = MarcatiliMode(125e-6, "He", 1.0, model="full").neff_minus_one(omega)
    reduced = MarcatiliMode(125e-6, "He", 1.0, model="reduced").neff_minus_one(omega)
    assert np.max(np.abs(reduced / full - 1.0)) < 2e-5


def test_neff_minus_one_beats_the_cancelling_form():
    """Forming n - 1 directly, not as (1 + x) - 1: about eight digits at a gas fill.

    The comparison is against the same quantity built through unity in float64. The
    two agree only to the cancellation floor of that route -- roughly eps / (n - 1),
    about 1e-11 relative at a susceptibility of 1e-4 -- and the direct form is the one
    that is right, which is why the assertion is that they differ by *more* than
    rounding as well as by less than the cancellation bound.
    """
    mode = MarcatiliMode(125e-6, "Ar", 1.0, model="reduced", cladding=1.45)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 64))
    direct = mode.neff_minus_one(omega)
    through_unity = mode.neff(omega) - 1.0
    relative = np.abs(through_unity / direct - 1.0)
    assert np.max(relative) < 1e-10
    assert np.max(relative) > 1e-16


def test_single_gas_refractivity_is_rounding_exact_at_a_complex_argument():
    """The single-gas full model rationalises its root, as a mixture does.

    NumPy 2.5's complex ``log1p`` forms ``1 + z`` and rounds there, so the former
    single-gas spelling ``expm1(log1p(z)/2)`` was 1.26e-12 relative off at
    ``z = 1e-4 + 1e-7j``. The terms are set so that the argument is exactly that
    number (no waveguide term, unit density); mpmath at 50 digits is the reference.
    A realistic lossy-cladding fill is checked the same way, its argument mirrored
    in float64 from the production terms so only the root is under test.
    """
    mode = reference_mode(cladding=1.45 + 1e-3j)
    terms = mode.sampled_terms(omega_of(np.array([400e-9, 600e-9, 1000e-9])))
    exact = replace(
        terms,
        omega=terms.omega[:1],
        u_nm=0.0,
        core_susceptibility=np.array([1e-4 + 1e-7j]),
        dcore_susceptibility_domega=np.array([0.0j]),
        cladding_term=terms.cladding_term[:1],
        dcladding_term_domega=terms.dcladding_term_domega[:1],
    )
    density = float(mode.density(np.array([0.0]))[0])
    scale = C / (terms.omega * 50e-6)
    waveguide = (terms.u_nm * scale) ** 2 * (
        1.0 - 1j * terms.cladding_term * scale
    ) ** 2
    cases = [
        (mode.refractivity_from_terms(exact, 50e-6, 1.0), np.array([1e-4 + 1e-7j])),
        (
            mode.refractivity_from_terms(terms, 50e-6, density),
            terms.core_susceptibility * density - waveguide,
        ),
    ]
    for value, argument in cases:
        with mp.workdps(50):
            reference = np.array(
                [complex(mp.sqrt(1 + mp.mpc(z.real, z.imag)) - 1) for z in argument]
            )
        error = np.abs(value - reference) / np.abs(reference)
        # Measured at most 1.3e-19 and 3.4e-16 relative; the former spelling was
        # 1.26e-12 and 1.6e-13 off.
        assert np.all(error < 4 * np.finfo(float).eps)
    naive = np.expm1(0.5 * np.log1p(np.array([1e-4 + 1e-7j])))
    assert abs(naive[0] - cases[0][0][0]) > 1e-13 * abs(cases[0][0][0])


@pytest.mark.parametrize("model", MODELS)
def test_neff_is_one_plus_the_refractivity(model):
    mode = MarcatiliMode(125e-6, "He", 1.0, model=model)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    # One addition of unity on a separately evaluated refractivity: a few roundings.
    np.testing.assert_allclose(
        mode.neff(omega), 1.0 + mode.neff_minus_one(omega), rtol=4 * EPS, atol=0
    )


def test_an_evacuated_capillary_guides_below_the_vacuum_index():
    """With no gas the waveguide term is all there is, so Re n_eff < 1."""
    mode = MarcatiliMode(125e-6)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    assert np.all(np.real(mode.neff(omega)) < 1.0)
    assert np.all(mode.density(np.array([0.0, 1.0])) == 0.0)


# ------------------------------------------------------------ the loss options


@pytest.mark.parametrize("model", MODELS)
def test_loss_false_removes_the_imaginary_part(model):
    mode = MarcatiliMode(125e-6, "He", 1.0, model=model, loss=False)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    assert np.all(np.imag(mode.neff(omega)) == 0.0)
    assert np.all(np.imag(mode.dneff_domega(omega)) == 0.0)


@pytest.mark.parametrize("model", MODELS)
def test_a_loss_scale_multiplies_the_model_loss(model):
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    full = MarcatiliMode(125e-6, "He", 1.0, model=model)
    scaled = MarcatiliMode(125e-6, "He", 1.0, model=model, loss=0.25)
    np.testing.assert_allclose(
        np.imag(scaled.neff(omega)), 0.25 * np.imag(full.neff(omega)), rtol=1e-15
    )
    np.testing.assert_allclose(
        np.real(scaled.neff(omega)), np.real(full.neff(omega)), rtol=0.0, atol=0.0
    )


def test_a_loss_callable_replaces_the_model_loss_with_an_absolute_alpha():
    """A float scales, a callable replaces: see the guide's capillary model."""
    curve = np.vectorize(lambda w: 1e-15 * w)
    mode = MarcatiliMode(125e-6, "He", 1.0, loss=curve)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    np.testing.assert_allclose(alpha(mode, omega), curve(omega), rtol=1e-14)
    plain = MarcatiliMode(125e-6, "He", 1.0)
    np.testing.assert_allclose(
        np.real(mode.neff(omega)), np.real(plain.neff(omega)), rtol=0.0, atol=0.0
    )


def test_a_loss_callable_with_a_derivative_is_analytic():
    class Curve:
        def __call__(self, omega):
            return 1e-30 * omega**2

        def derivative(self, omega):
            return 2e-30 * omega

    mode = MarcatiliMode(125e-6, "He", 1.0, loss=Curve())
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    # Im n = c alpha / (2 omega) = c * 1e-30 * omega / 2, so d(Im n)/d(omega) is a
    # constant that the quotient rule must reproduce exactly.
    np.testing.assert_allclose(
        np.imag(mode.dneff_domega(omega)), 0.5 * C * 1e-30, rtol=1e-12
    )


def test_a_loss_callable_without_a_derivative_is_differenced_and_says_so(caplog):
    curve = np.vectorize(lambda w: 1e-30 * w**2)
    with caplog.at_level("WARNING", logger="solprop.modes.capillary"):
        mode = MarcatiliMode(125e-6, "He", 1.0, loss=curve)
    assert "Richardson" in caplog.text
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    # Same curve as the analytic case above, so the difference must land on the same
    # constant, to the 1e-10 relative floor the constructor advertises.
    np.testing.assert_allclose(
        np.imag(mode.dneff_domega(omega)), 0.5 * C * 1e-30, rtol=1e-9
    )


# ----------------------------------------------------- the analytic derivative


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind", KINDS)
def test_dneff_domega_matches_the_contour_derivative(kind, model):
    """The sharpest check available, and it costs no second implementation.

    Cauchy's integral formula on a circle of relative radius 1e-4, sampled by a
    64-point trapezoid rule, is spectrally accurate and free of subtractive
    cancellation, so it pins the derivative to about 1e-13 relative -- three orders
    better than any finite difference of the same function. An evacuated core with a
    constant complex cladding exercises the whole Marcatili algebra (the Bessel zero,
    v_n of each family, the square root, every power of a and omega) without asking a
    material model for a complex frequency.
    """
    n = 0 if kind in ("TE", "TM") else 2
    mode = MarcatiliMode(50e-6, kind=kind, n=n, model=model, cladding=1.45 + 0.01j)
    omega = omega_of(np.linspace(300e-9, 2000e-9, 24))
    analytic = mode.dneff_domega(omega)
    reference = contour_derivative(mode, omega, 50e-6, 0.0)
    assert (
        np.max(np.abs(analytic - reference)) / np.max(np.abs(analytic))
        < CONTOUR_TOLERANCE
    )


@pytest.mark.parametrize("model", MODELS)
def test_dneff_domega_matches_finite_differences_at_three_steps(model):
    """A three-step ladder, with the floor of the comparison stated.

    A central difference of n_eff has truncation error (h^2/6) n''' and rounding error
    eps |n| / h. Because n_eff is 1 + something of order 1e-5, the rounding term is
    eps/h relative to *unity*, so the best a plain central difference can do here is
    about 1e-6 relative to the derivative -- it is the test that is limited, not the
    model, which the contour check above shows to 1e-13. The three steps bracket that
    optimum: 1e-4 is truncation limited, 1e-6 is rounding limited, 1e-5 is near the
    best. The worst of the three is 5.4e-6 relative, so 5e-5 keeps an order in hand
    while still catching any real algebraic slip, which would be of order one.
    """
    mode = MarcatiliMode(
        125e-6, "Ar", 2.0, susceptibility=borzsonyi("Ar"), model=model, cladding=1.45
    )
    omega = omega_of(np.linspace(450e-9, 950e-9, 16))
    analytic = mode.dneff_domega(omega)
    scale = np.max(np.abs(analytic))
    for relative_step in (1e-4, 1e-5, 1e-6):
        step = relative_step * omega
        difference = (
            mode.neff_minus_one(omega + step) - mode.neff_minus_one(omega - step)
        ) / (2.0 * step)
        assert np.max(np.abs(difference - analytic)) / scale < 5e-5


def test_the_derivative_of_a_taper_and_a_gradient_follows_z():
    """z enters only through two scalars, so the derivative must move with them."""
    fill = PressureGradient.two_point(1.0, 2.0, 0.5)
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "Ar", fill, cladding=1.45)
    omega = np.array([OMEGA_800])
    assert not mode.z_constant
    assert not mode.radius_is_constant
    start, end = mode.dneff_domega(omega, 0.0), mode.dneff_domega(omega, 1.0)
    # atol must be zero: these values are of order 1e-21, so the default absolute
    # tolerance would call any two of them equal.
    assert not np.allclose(start, end, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("model", MODELS)
def test_the_derivative_of_a_taper_is_analytic_at_both_ends(model):
    """The contour check again, now with the radius moving under it."""
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), model=model, cladding=1.45)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 8))
    for z in (0.0, 0.5, 1.0):
        core_radius = float(mode.core_radius(np.array([z]))[0])
        analytic = mode.dneff_domega(omega, z)
        reference = contour_derivative(mode, omega, core_radius, 0.0)
        assert (
            np.max(np.abs(analytic - reference)) / np.max(np.abs(analytic))
            < CONTOUR_TOLERANCE
        )


# ---------------------------------------------------------- the derivative along z


Z_TAPER = Linear(50e-6, 40e-6, 2e-4)
"""The qualification taper's slope, 50 to 40 micrometres over 0.2 mm."""

Z_GRADIENT = PressureGradient.two_point(1.0, 0.1, 2e-4)
"""A falling pressure over the same length, 1 to 0.1 bar."""


def parameter_contour(mode, omega, radius, density, name):
    """Cauchy's formula for dn/d(radius) or dn/d(density) on a small circle.

    The model's index is holomorphic in both scalars -- a square root of a polynomial
    in `1/a`, linear in `rho` -- so the trapezoid rule on a circle of relative radius
    `step` pins the partial derivative with no subtractive cancellation, exactly as
    `contour_derivative` does for the frequency. The scalars themselves are typed
    real, so the relative perturbation `1 + delta` is carried where they enter: the
    radius only through the product `omega a` of the waveguide term, the density only
    through `gamma rho`, which the sampled terms hold.
    """
    step, n_points = 1e-4, 64
    terms = mode.sampled_terms(omega)
    total = np.zeros(omega.shape, dtype=complex)
    for index in range(n_points):
        delta = step * np.exp(2j * math.pi * index / n_points)
        shifted = (
            replace(terms, omega=omega * (1.0 + delta))
            if name == "radius"
            else replace(
                terms, core_susceptibility=terms.core_susceptibility * (1.0 + delta)
            )
        )
        total = total + mode._model_values(shifted, radius, density)[0] / delta
    return total / n_points / (radius if name == "radius" else density)


def fourth_order_slope(index, z, step):
    """Return `[8(f(z+h/2) - f(z-h/2)) - (f(z+h) - f(z-h))]/(6h)`, fourth order."""
    return (
        8.0 * (index(z + step / 2) - index(z - step / 2))
        - (index(z + step) - index(z - step))
    ) / (6.0 * step)


@pytest.mark.parametrize("model", MODELS)
def test_dneff_dz_is_the_chain_rule_through_both_scalars(model):
    """`dn/dz = (dn/da) a' + (dn/drho) rho'`, each partial against Cauchy's formula.

    Bound, stated first: the contour pins each partial to its rounding over the
    circle's radius, about 1e-16/1e-4 relative, so 1e-9 keeps an order in hand
    (measured 1.7e-10 full, 3.2e-13 reduced);
    the profiles' slopes are their own analytic ones. A tapered radius **and** a
    falling pressure are declared together, with a constant cladding so that every
    term of the model -- the Bessel zero, the square root, every power of `a`, the
    gas -- carries a slope.
    """
    mode = MarcatiliMode(
        Z_TAPER,
        "Ar",
        Z_GRADIENT,
        m=2,
        model=model,
        susceptibility=borzsonyi("Ar"),
        cladding=1.45,
    )
    omega = omega_of(np.linspace(450e-9, 1000e-9, 12))
    z = 7.3e-5
    position = np.array([z])
    radius = float(mode.core_radius(position)[0])
    density = float(mode.density(position)[0])
    expected = parameter_contour(mode, omega, radius, density, "radius") * float(
        mode.core_radius.derivative(position)[0]
    ) + parameter_contour(mode, omega, radius, density, "density") * float(
        mode.density.derivative(position)[0]
    )
    analytic = mode.dneff_dz(omega, z)
    assert np.max(abs(analytic - expected)) / np.max(abs(analytic)) < 1e-9


@pytest.mark.parametrize(
    ("radius", "pressure", "step", "gate"),
    [(Z_TAPER, 1.0, 1e-6, 1e-8), (50e-6, Z_GRADIENT, 2e-5, 5e-5)],
    ids=["taper", "gradient"],
)
def test_dneff_dz_is_the_finite_difference_of_the_implemented_index(
    radius, pressure, step, gate
):
    """A fourth-order central difference along z of what `neff_minus_one` returns.

    Bounds, stated first. The implemented index carries about 3e-16 of absolute
    rounding (its fourth differences at a micrometre spacing), so the difference has a
    rounding error of about `3 * 3e-16/h` and a truncation of `(h/2)^4 |f^(5)|/30`.
    **Taper**, `h` = 1 micrometre: the radius term's slope reaches 0.39 1/m, so the
    rounding is 2e-9 of it and the truncation, `(h a'/a)^4` ~ 1e-12, nothing: gate
    1e-8. **Gradient**, `h` = 20 micrometres: the gas term's slope is only 1.6e-6 1/m,
    so the rounding is 3e-5 of it, and `p = sqrt(p0^2 + (pL^2 - p0^2) z/L)` puts the
    truncation near 8e-6: gate 5e-5. Measured 2.9e-10 and 3.6e-6. The contour check
    above is the sharp one; this says the derivative is the slope of the index the run
    actually uses.
    """
    mode = MarcatiliMode(radius, "He", pressure, m=2)
    omega = omega_of(np.linspace(450e-9, 1000e-9, 7))
    z = 7.3e-5
    analytic = np.real(mode.dneff_dz(omega, z))
    difference = np.real(
        fourth_order_slope(lambda at: mode.neff_minus_one(omega, at), z, step)
    )
    assert np.max(abs(analytic - difference)) / np.max(abs(analytic)) < gate


def test_dneff_dz_follows_the_loss_specification_and_a_mixture():
    """The slope of what `neff` returns, whatever the loss setting does to it.

    `loss=False` drops the imaginary part, a float scales it, a measured curve has no
    z in it, and the real part moves with none of them. A fill whose populations are
    declared separately sums their slopes, checked against the fourth-order difference
    at the taper's gate.
    """
    omega = omega_of(np.linspace(450e-9, 1000e-9, 7))
    z = 7.3e-5
    reference = MarcatiliMode(Z_TAPER, "He", 1.0, m=2).dneff_dz(omega, z)
    lossless = MarcatiliMode(Z_TAPER, "He", 1.0, m=2, loss=False).dneff_dz(omega, z)
    scaled = MarcatiliMode(Z_TAPER, "He", 1.0, m=2, loss=0.5).dneff_dz(omega, z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        measured = MarcatiliMode(
            Z_TAPER, "He", 1.0, m=2, loss=lambda values: np.full(values.shape, 1e-3)
        ).dneff_dz(omega, z)
    # The loss setting touches only the imaginary part, structurally: the real part
    # is the same array, and a lossless or z-free loss is an explicit zero slope.
    np.testing.assert_array_equal(np.real(lossless), np.real(reference))
    np.testing.assert_array_equal(np.imag(lossless), 0.0)
    np.testing.assert_allclose(np.imag(scaled), 0.5 * np.imag(reference), rtol=1e-15)
    np.testing.assert_array_equal(np.real(measured), np.real(reference))
    np.testing.assert_array_equal(np.imag(measured), 0.0)
    assert np.all(np.imag(reference) != 0.0)
    # The reduced model drops its loss term, slope included, when declared lossless.
    polynomial = MarcatiliMode(Z_TAPER, "He", 1.0, m=2, model="reduced", loss=False)
    slope = polynomial.dneff_dz(omega, z)
    np.testing.assert_array_equal(np.imag(slope), 0.0)
    difference = np.real(
        fourth_order_slope(lambda at: polynomial.neff_minus_one(omega, at), z, 1e-6)
    )
    assert np.max(abs(np.real(slope) - difference)) / np.max(abs(slope)) < 1e-8
    mixed = MarcatiliMode(
        Z_TAPER, mixture=Mixture.partial_pressures({"He": Z_GRADIENT, "Ar": 0.5}), m=2
    )
    slope = np.real(mixed.dneff_dz(omega, z))
    difference = np.real(
        fourth_order_slope(lambda at: mixed.neff_minus_one(omega, at), z, 1e-6)
    )
    assert np.max(abs(slope - difference)) / np.max(abs(slope)) < 1e-8


def test_dneff_dz_is_zero_where_a_clamp_holds_the_index():
    """Where the evanescent floor or the no-gain clamp binds, so does the slope.

    The two cases of the floor tests below, on a tapered core: the floored index is a
    constant, and the clamped loss is zero, so each slope is zero there, exactly as
    `dneff_domega`'s is.
    """
    omega = omega_of(np.array([50e-6]))
    floored = MarcatiliMode(Linear(1e-6, 0.9e-6, 1e-3), model="full", cladding=0.5)
    assert np.real(floored.neff(omega, 5e-4))[0] == pytest.approx(
        EVANESCENT_INDEX_FLOOR, rel=1e-12
    )
    assert floored.dneff_dz(omega, 5e-4)[0] == 0.0
    clamped = MarcatiliMode(
        Linear(1e-6, 0.9e-6, 1e-3), model="full", cladding=0.05 + 0.01j
    )
    assert np.imag(clamped.neff(omega, 5e-4))[0] == 0.0
    slope = clamped.dneff_dz(omega, 5e-4)
    assert np.imag(slope)[0] == 0.0
    assert np.real(slope)[0] != 0.0


def test_dneff_dz_is_refused_where_it_would_differentiate_another_index():
    """A structured index is not Marcatili's, and a sampled path has no slope."""
    omega = np.array([OMEGA_800])
    lattice = TubeLattice(180e-9, 10.38e-6, 6)
    with pytest.raises(ModeError, match="VincettiMode computes its index another way"):
        VincettiMode(15e-6, "Ar", 1.0, lattice=lattice).dneff_dz(omega)
    sampled = FromCallable(lambda z: 50e-6 - 10.0 * np.asarray(z), (0.0, 1e-3))
    with pytest.raises(ModeError, match="core radius profile's analytic derivative"):
        MarcatiliMode(sampled, "He", 1.0).dneff_dz(omega)


# ------------------------------------------------------- the evanescent floor


def test_the_full_model_floors_an_evanescent_index():
    """Documented, not hidden: Luna applies the same floor inside its kernel."""
    # A 1 um core with a cladding index below unity -- a metal, or any solid in the
    # extreme ultraviolet -- puts the square root's argument on the negative real axis,
    # which is the case the floor exists for.
    mode = MarcatiliMode(1e-6, model="full", cladding=0.5)
    omega = omega_of(np.array([50e-6]))
    index = mode.neff(omega)
    assert np.real(index)[0] == pytest.approx(EVANESCENT_INDEX_FLOOR, rel=1e-12)
    assert np.imag(index)[0] >= 0.0
    # The floored value is constant, so its derivative is reported as zero.
    assert mode.dneff_domega(omega)[0] == 0.0


def test_the_reduced_model_has_no_floor():
    """As in Luna: the expansion is a polynomial, and nothing clamps it.

    Lossless, so that the arithmetic is transparent: n_eff = 1 - (u c / (omega a))^2 / 2
    is far below zero for a core this small, and stays there rather than being pulled
    up to the floor. A complex cladding term would add a *real* contribution through
    i v, which is a separate quirk of the reduced model and would muddy the point.
    """
    mode = MarcatiliMode(1e-6, model="reduced", cladding=0.5, loss=False)
    omega = omega_of(np.array([50e-6]))
    assert np.real(mode.neff(omega))[0] < -100.0


def test_a_cladding_index_far_below_one_is_clamped_to_zero_loss_and_logged(caplog):
    """The branch of the square root that reads as gain is refused, and said so.

    With a cladding index of 0.05 + 0.01i -- a metal in the extreme ultraviolet, far
    outside anything Marcatili's expansion describes -- the complex square root lands
    where Im n_eff < 0. Luna clamps only inside its real-part floor, so it returns gain
    there; a passive capillary cannot amplify, so the loss is clamped at zero,
    its derivative with it, and the mode logs the fact once.
    """
    mode = MarcatiliMode(1e-6, model="full", cladding=0.05 + 0.01j)
    omega = omega_of(np.array([50e-6]))
    with caplog.at_level(logging.WARNING, logger="solprop.modes.capillary"):
        index = mode.neff(omega)
        derivative = mode.dneff_domega(omega)
    assert np.real(index)[0] > EVANESCENT_INDEX_FLOOR
    assert np.imag(index)[0] == 0.0
    assert np.imag(derivative)[0] == 0.0
    assert sum("gain" in record.message for record in caplog.records) == 1


def test_the_floor_leaves_a_guided_index_untouched():
    mode = MarcatiliMode(125e-6, "He", 1.0)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 16))
    assert np.all(np.real(mode.neff(omega)) > 0.5)


# ----------------------------------------------- normalisation and effective area


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", [1, 2, 3])
def test_normalisation_matches_the_quadrature_it_stands_for(kind, m):
    """The closed forms are exact; check them against the integrals that define them.

    N = (1/2) sqrt(eps0/mu0) int |e|^2 dA over the disc, by a dense product rule. The
    tolerance is the rule's own convergence, not the model's: 200 radial x 64
    azimuthal Gauss-Legendre points reproduce a Bessel product to 1e-12.
    """
    n = 0 if kind in ("TE", "TM") else 2
    mode = MarcatiliMode(125e-6, "He", 1.0, kind=kind, n=n, m=m)
    radius = 125e-6
    nodes_r, weights_r = np.polynomial.legendre.leggauss(200)
    nodes_t, weights_t = np.polynomial.legendre.leggauss(64)
    r = radius * 0.5 * (nodes_r + 1.0)
    theta = math.pi * (nodes_t + 1.0)
    grid_r, grid_t = np.meshgrid(r, theta, indexing="ij")
    points = np.stack([grid_r.ravel(), grid_t.ravel()], axis=-1)
    field = mode.field(points).reshape(*grid_r.shape, 2)
    jacobian = (
        grid_r * (radius * 0.5 * weights_r)[:, None] * (math.pi * weights_t)[None, :]
    )
    integral = float(np.sum(jacobian * np.sum(field**2, axis=-1)))
    assert mode.normalisation() == pytest.approx(0.5 * EPS0 * C * integral, rel=1e-12)

    second = float(np.sum(jacobian * np.sum(field**2, axis=-1)))
    fourth = float(np.sum(jacobian * np.sum(field**2, axis=-1) ** 2))
    assert mode.aeff() == pytest.approx(second**2 / fourth, rel=1e-9)


def test_aeff_reproduces_lunas_stored_value():
    # Luna test_capillary.jl asserts Aeff(MarcatiliMode(75e-6, :HeB, 1.0)) =
    # 8.42157534886545e-9.
    # The effective area is purely geometric, so it does not depend on the gas or the
    # cladding data and the agreement is exact to the quadrature of the J^4 integral.
    assert MarcatiliMode(75e-6, "He", 1.0).aeff() == pytest.approx(
        8.42157534886545e-09, rel=1e-12
    )


def test_normalisation_and_aeff_scale_as_the_square_of_the_radius():
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", 1.0)
    ratio = (75e-6 / 125e-6) ** 2
    assert mode.normalisation(1.0) / mode.normalisation(0.0) == pytest.approx(
        ratio, rel=1e-14
    )
    assert mode.aeff(1.0) / mode.aeff(0.0) == pytest.approx(ratio, rel=1e-14)


# ------------------------------------------------------------------- the field


def test_he11_is_uniformly_polarised_along_x():
    """The unrotated HE11 points along x (ADR-031); Luna's phi = 0 points along y."""
    mode = MarcatiliMode(125e-6, "He", 1.0)
    theta = np.linspace(0.0, 2.0 * math.pi, 9)
    points = np.stack([np.full_like(theta, 30e-6), theta], axis=-1)
    field = mode.field(points)
    np.testing.assert_allclose(field[:, 1], 0.0, atol=1e-15)
    assert np.all(field[:, 0] > 0.0)
    np.testing.assert_allclose(field[:, 0], field[0, 0], rtol=1e-14)


def test_the_rotation_angle_turns_the_polarisation():
    """phi turns the mode from x towards y, so the quarter turn is y-polarised."""
    rotated = MarcatiliMode(125e-6, "He", 1.0, phi=math.pi / 2)
    points = np.array([[30e-6, 0.0]])
    field = rotated.field(points)[0]
    assert abs(field[0]) < 1e-15
    assert abs(field[1]) > 0.0
    # For HE_1m the angle is the polarisation's own, measured from x towards y.
    oblique = MarcatiliMode(125e-6, "He", 1.0, phi=0.3).field(points)[0]
    assert math.atan2(oblique[1], oblique[0]) == pytest.approx(0.3, rel=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_default_hybrid_mode_is_the_x_polarised_even_member(n):
    """ADR-031: phi = 0 is `even` and leads with x for every n, the quarter turn odd.

    Asserted on the field rather than on the name, along the theta = 0 ray where the
    even pattern (cos((n-1)theta), -sin((n-1)theta)) is purely x and its quarter-turn
    partner purely y. A mode with n > 1 vanishes on the axis itself, so the ray is
    sampled a nanometre out, as the cross-family parity test does. This
    is a deliberate quarter-turn divergence from Luna, whose phi = 0 is y-polarised.
    """
    default = MarcatiliMode(125e-6, "He", 1.0, n=n)
    partner = default.orthogonal_polarisation
    assert partner is not None
    ray = np.array([[1e-9, 0.0], [10e-6, 0.0], [60e-6, 0.0]])
    principal, orthogonal = default.field(ray).T
    assert np.all(abs(principal) > 0.0)
    # Measured, not guessed: on this ray the orthogonal column is *exactly* zero at
    # phi = 0 for n = 1, 2, 3 (the general path evaluates -sin(0), the HE_1m fast
    # path writes a literal zero), and 6.123e-17 of the principal one at the quarter
    # turn for n = 2, 3, which is cos(-pi/2) failing to round to zero. The ratio is
    # independent of r, so 1e-15 is a sixteenfold margin on a rounding-level number
    # rather than a physical smallness bound.
    assert np.all(abs(orthogonal) < 1e-15 * abs(principal))
    turned_x, turned_y = partner.field(ray).T
    assert np.all(abs(turned_y) > 0.0)
    assert np.all(abs(turned_x) < 1e-15 * abs(turned_y))
    # The same mode seen twice: a quarter turn moves no power off the ray.
    np.testing.assert_allclose(abs(turned_y), abs(principal), rtol=1e-14)
    assert default.channel_label.parity == "even"
    assert partner.channel_label.parity == "odd"
    assert default.channel_label.display_label == f"HE{n},1:even"
    assert partner.channel_label.display_label == f"HE{n},1:odd"


def test_the_field_vanishes_on_the_wall():
    """u_nm is a zero of the radial function, which is what makes the mode guided."""
    for kind, n in ((("HE"), 1), (("HE"), 3), (("TE"), 0), (("TM"), 0)):
        mode = MarcatiliMode(125e-6, "He", 1.0, kind=cast("ModeKind", kind), n=n)
        wall = np.array([[125e-6, 0.7]])
        np.testing.assert_allclose(mode.field(wall), 0.0, atol=1e-14)


def test_the_field_magnitude_is_azimuthally_uniform_for_a_hybrid_mode():
    """|e| = |J_{n-1}(u r/a)| whatever theta, which is why HE_1m needs no theta rule."""
    mode = MarcatiliMode(125e-6, "He", 1.0, n=3, m=2)
    theta = np.linspace(0.0, 2.0 * math.pi, 17)
    points = np.stack([np.full_like(theta, 40e-6), theta], axis=-1)
    magnitude = np.linalg.norm(mode.field(points), axis=-1)
    np.testing.assert_allclose(magnitude, magnitude[0], rtol=1e-13)


def test_the_field_scales_with_the_radius_for_a_taper():
    """Scale invariance, which the transform's taper rescaling rests on."""
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", 1.0)
    xi, theta = 0.37, 1.1
    at_start = mode.field(np.array([[xi * 125e-6, theta]]), 0.0)
    at_end = mode.field(np.array([[xi * 75e-6, theta]]), 1.0)
    np.testing.assert_allclose(at_start, at_end, rtol=1e-14)


# ------------------------------------------------------------------- the traits


@pytest.mark.parametrize(
    ("kind", "n", "expected"),
    [
        ("HE", 1, 0),
        ("HE", 2, 1),
        ("HE", 4, 3),
        ("HE", -1, 2),
        ("TE", 0, 1),
        ("TM", 0, 1),
    ],
)
def test_azimuthal_order_is_never_negative(kind, n, expected):
    """Luna returns n - 1 unsigned, which is negative for HE_-1m and understates the
    theta rule a cubic product needs."""
    assert MarcatiliMode(125e-6, kind=kind, n=n).azimuthal_order == expected


def test_z_constant_and_radius_is_constant_are_different_questions():
    """A gradient changes the index but not the field profile; the transform cares."""
    gradient = MarcatiliMode(125e-6, "He", PressureGradient.two_point(1.0, 1.0, 0.1))
    assert not gradient.z_constant
    assert gradient.radius_is_constant
    taper = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", 1.0)
    assert not taper.z_constant
    assert not taper.radius_is_constant
    plain = MarcatiliMode(125e-6, "He", 1.0)
    assert plain.z_constant
    assert plain.radius_is_constant


# --------------------------------------------------------- the z factorisation


@pytest.mark.parametrize("model", MODELS)
def test_the_sampled_terms_reproduce_the_direct_evaluation(model):
    """The translation layer will use this path, so it must not be a second answer.

    The factorised evaluation regroups the direct one's roundings; it is held to the
    ladder's one-evaluation budget of 16 eps (``accuracy.md`` section 7), normwise.
    """
    fill = PressureGradient.two_point(1.0, 2.0, 0.5)
    mode = MarcatiliMode(
        Linear(125e-6, 75e-6, 1.0), "Ar", fill, model=model, cladding=1.45
    )
    omega = omega_of(np.linspace(400e-9, 1000e-9, 32))
    terms = mode.sampled_terms(omega)
    for z in (0.0, 0.3, 1.0):
        radius = float(mode.core_radius(np.array([z]))[0])
        density = float(mode.density(np.array([z]))[0])
        assert_within(
            mode.neff_from_terms(terms, radius, density),
            mode.neff(omega, z),
            16 * EPS,
            what="neff",
        )
        assert_within(
            mode.dneff_domega_from_terms(terms, radius, density),
            mode.dneff_domega(omega, z),
            16 * EPS,
            what="dneff/domega",
        )


def test_the_sampled_terms_carry_the_three_scalars_the_operator_needs():
    mode = MarcatiliMode(125e-6, "He", 1.0)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 8))
    terms = mode.sampled_terms(omega)
    assert terms.u_nm == mode.u_nm
    assert terms.core_susceptibility.shape == omega.shape
    assert terms.cladding_term.shape == omega.shape
    assert np.iscomplexobj(terms.cladding_term)
    np.testing.assert_array_equal(terms.omega, omega)


@pytest.mark.parametrize("model", MODELS)
def test_a_mode_frozen_at_its_own_parameters_reproduces_it(model):
    """The parameter expansion of the kernel contract rests on this being the same mode.

    The frozen mode evaluates the same closed form at the sampled parameters, so the
    two agree to the ladder's one-evaluation budget of 16 eps, normwise.
    """
    fill = PressureGradient.two_point(1.0, 2.0, 0.5)
    mode = MarcatiliMode(
        Linear(125e-6, 75e-6, 1.0), "Ar", fill, model=model, cladding=1.45
    )
    omega = omega_of(np.linspace(400e-9, 1000e-9, 32))
    for z in (0.0, 0.3, 1.0):
        frozen = mode.at_parameters(
            {
                name: float(path(np.array([z]))[0])
                for name, path in mode.parameter_paths.items()
            }
        )
        assert frozen.z_constant
        assert_within(frozen.neff(omega), mode.neff(omega, z), 16 * EPS, what="neff")
        assert_within(
            frozen.dneff_domega(omega),
            mode.dneff_domega(omega, z),
            16 * EPS,
            what="dneff/domega",
        )
        assert frozen.aeff() == pytest.approx(mode.aeff(z), rel=16 * EPS, abs=0)


def test_freezing_one_path_leaves_the_other_varying():
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "Ar", 2.0)
    frozen = mode.at_parameters({"core_radius": 100e-6})
    assert frozen.radius == 100e-6
    assert frozen.core_radius.is_constant
    assert frozen.density is mode.density


def test_fill_rows_are_the_same_only_for_the_same_models_and_absences():
    """A frozen copy shares its origin's terms only when every row is the same.

    The models must be the same objects in the same order, a mixture's absences
    equal, and a single gas's model is never a mixture's row; anything else samples
    afresh.
    """
    from solprop.modes.capillary import _same_rows  # noqa: PLC0415

    argon, neon = object(), object()
    assert _same_rows((argon,), (argon,))
    assert not _same_rows((argon,), (neon,))
    assert _same_rows(((argon, False), (neon, True)), ((argon, False), (neon, True)))
    assert not _same_rows(
        ((argon, False), (neon, True)), ((argon, False), (neon, False))
    )
    assert not _same_rows(((argon, False),), ((argon, False), (neon, False)))
    assert not _same_rows((argon,), ((argon, False),))


def test_the_declared_paths_are_both_present_even_when_constant():
    mode = MarcatiliMode(125e-6, "He", 1.0)
    paths = mode.parameter_paths
    assert sorted(paths) == ["core_radius", "density"]
    assert all(path.is_constant for path in paths.values())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"pressure": 1.0}, "not a declared path"),
        ({"core_radius": 0.0}, "core radius must be positive"),
        ({"density": -1.0}, "must not be negative"),
    ],
)
def test_freezing_at_an_unusable_parameter_is_refused(values, message):
    mode = MarcatiliMode(125e-6, "He", 1.0)
    with pytest.raises(ModeError, match=message):
        mode.at_parameters(values)


def test_mismatched_sampled_terms_are_refused():
    with pytest.raises(ModeError, match="disagree in shape"):
        MarcatiliTerms(
            omega=np.array([1.0, 2.0]),
            u_nm=2.4,
            core_susceptibility=np.array([1.0]),
            dcore_susceptibility_domega=np.array([1.0]),
            cladding_term=np.array([1.0 + 0j]),
            dcladding_term_domega=np.array([1.0 + 0j]),
        )


# --------------------------------------------------------------- the cladding


def test_a_constant_cladding_is_accepted_and_kills_its_dispersion():
    mode = MarcatiliMode(50e-6, "Ar", 2.0, cladding=1.45)
    omega = omega_of(np.linspace(400e-9, 1000e-9, 8))
    terms = mode.sampled_terms(omega)
    # A constant cladding has an explicit zero derivative, not a computed one.
    np.testing.assert_array_equal(terms.dcladding_term_domega, 0.0)
    np.testing.assert_allclose(terms.cladding_term, terms.cladding_term[0], rtol=0.0)


def test_the_default_cladding_is_fused_silica_and_disperses():
    mode = MarcatiliMode(125e-6, "He", 1.0)
    assert isinstance(mode.cladding_index, GlassIndex)
    assert mode.cladding_index.name == "SiO2"
    omega = omega_of(np.linspace(400e-9, 1000e-9, 8))
    terms = mode.sampled_terms(omega)
    assert np.any(terms.dcladding_term_domega != 0.0)


@pytest.mark.parametrize(("kind", "n"), [("HE", 1), ("TE", 0), ("TM", 0)])
def test_the_cladding_term_matches_its_closed_form(kind, n):
    permittivity = (1.45 + 0.01j) ** 2
    expected = {
        "HE": (permittivity + 1.0) / (2.0 * np.sqrt(permittivity - 1.0)),
        "TE": 1.0 / np.sqrt(permittivity - 1.0),
        "TM": permittivity / np.sqrt(permittivity - 1.0),
    }[kind]
    mode = MarcatiliMode(50e-6, kind=kind, n=n, cladding=1.45 + 0.01j)
    terms = mode.sampled_terms(np.array([OMEGA_800]))
    assert complex(terms.cladding_term[0]) == pytest.approx(
        complex(expected), rel=1e-14
    )


# ------------------------------------------------------------------- property


@pytest.mark.medium
@settings(max_examples=150, deadline=None)
@given(
    radius=st.floats(min_value=20e-6, max_value=500e-6),
    pressure=st.floats(min_value=1e-3, max_value=50.0),
    wavelength=st.floats(min_value=400e-9, max_value=1000e-9),
    model=st.sampled_from(MODELS),
)
def test_a_capillary_mode_is_physical_over_its_parameter_space(
    radius, pressure, wavelength, model
):
    """Guided, lossy and finite everywhere a real capillary experiment lives."""
    mode = MarcatiliMode(
        radius, "Ar", pressure, susceptibility=borzsonyi("Ar"), model=model
    )
    omega = np.array([2.0 * math.pi * C / wavelength])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        index = complex(mode.neff(omega)[0])
        derivative = complex(mode.dneff_domega(omega)[0])
        group_delay = float(beta1(mode, omega)[0])
    assert math.isfinite(np.real(index))
    assert math.isfinite(np.imag(index))
    assert np.imag(index) > 0.0
    assert 0.9 < np.real(index) < 1.1
    assert math.isfinite(np.real(derivative))
    # The group index stays within a few per cent of unity for a hollow capillary.
    assert 0.9 < group_delay * C < 1.1


def test_a_taper_and_a_gradient_can_be_combined():
    fill = PressureGradient.from_stages([0.0, 0.6, 1.5], [0.0, 5.0, 0.0])
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 1.5), Gas.Ar, fill)
    assert isinstance(mode.density, DensityPath)
    np.testing.assert_array_equal(mode.density.breakpoints, [0.0, 0.6, 1.5])
    omega = np.array([OMEGA_800])
    values = [complex(mode.neff(omega, z)[0]) for z in (0.0, 0.6, 1.5)]
    assert values[1].real > values[0].real
    assert values[1].real > values[2].real


def test_a_constant_radius_and_pressure_give_constant_profiles():
    mode = MarcatiliMode(125e-6, "He", 1.0)
    assert isinstance(mode.core_radius, Constant)
    assert mode.core_radius.is_constant
    assert mode.density.is_constant


# ------------------------------------------------- the symbolic witness (sympy)

SYMBOLIC = json.loads(
    (Path(__file__).parents[1] / "data/symbolic_marcatili.json").read_text()
)
SOLPROP_CASES = sorted(
    key for key in SYMBOLIC["cases"] if SYMBOLIC["cases"][key]["source"] == "solprop"
)

# The reference numbers Luna's own test_capillary.jl carries, which its
# test/symbolic_marcatilli.py produced. Transcribed here so that the regenerated
# witness is checked against the published values, not only against solprop.
LUNA_PUBLISHED = {
    "luna-reduced-silica-like": {
        "beta": [7857866.63899973, 3.33744405855838e-9, -1.68187063100098e-29],
        "zdw": 7.289431065526978e-07,
    },
    "luna-full-silica-like": {
        "beta": [7857865.68069111, 3.33744357233436e-9, -1.68088637707608e-29],
        "zdw": 7.289459086128214e-07,
        "alpha": 2.21505826015050,
    },
    "luna-full-metal-like": {
        "beta": [7857863.68326066, 3.33744527275634e-9, -1.89802237417366e-29],
        "zdw": 7.225347947615157e-07,
        "alpha": 0.0290115706883820,
    },
}


def symbolic_mode(case):
    """Build the mode a stored symbolic case describes."""
    return MarcatiliMode(
        SYMBOLIC["provenance"]["radius_m"],
        SYMBOLIC["provenance"]["gas"],
        SYMBOLIC["provenance"]["pressure_bar"],
        # The witness's index, Börzsönyi's argon fit, supplied explicitly.
        susceptibility=borzsonyi(SYMBOLIC["provenance"]["gas"]),
        temperature=SYMBOLIC["provenance"]["temperature_k"],
        model=case["model"],
        cladding=complex(*case["cladding"]),
    )


@pytest.mark.parametrize("key", LUNA_PUBLISHED)
def test_the_symbolic_witness_reproduces_lunas_published_numbers(key):
    """Validates the witness itself before anything is compared against it.

    tests/data/generate_symbolic_marcatili.py is a port of Luna's own sympy script,
    and its 'luna' cases use the constants that script hard-codes. If the port is
    faithful they must reproduce the numbers Luna's test_capillary.jl asserts, and
    they do to 1e-13 -- Luna's own tolerances on the same numbers run from 1e-15 for
    beta to 3e-7 for beta_2, because it reaches them by finite differences.
    """
    stored = SYMBOLIC["cases"][key]
    published = LUNA_PUBLISHED[key]
    for order, expected in enumerate(published["beta"]):
        assert stored["beta"][order] == pytest.approx(expected, rel=1e-12)
    assert stored["zdw"] == pytest.approx(published["zdw"], rel=1e-12)
    if "alpha" in published:
        assert stored["alpha"] == pytest.approx(published["alpha"], rel=1e-12)


@pytest.mark.parametrize("key", SOLPROP_CASES)
def test_beta_and_beta1_match_the_symbolic_derivation_to_rounding(key):
    """The sharpest statement that can be made about the analytic derivative.

    sympy differentiates the closed-form effective index symbolically; solprop
    differentiates it by hand and chains through the material models. The inputs are
    shared, so what is compared is purely the calculus -- and beta_1, which is
    (n + omega dn/domega)/c and therefore the whole content of dneff_domega, agrees to
    2e-16. Luna's own gate on the same quantity is 1e-13, because it differences.
    """
    case = SYMBOLIC["cases"][key]
    mode = symbolic_mode(case)
    omega = np.array([2.0 * math.pi * C / SYMBOLIC["provenance"]["wavelength_m"]])
    assert float(beta(mode, omega)[0]) == pytest.approx(case["beta"][0], rel=1e-14)
    assert float(beta1(mode, omega)[0]) == pytest.approx(case["beta"][1], rel=1e-14)


@pytest.mark.parametrize("key", SOLPROP_CASES)
def test_alpha_matches_the_symbolic_derivation_to_rounding(key):
    """Which is also the sign: a symbolic Im n_eff > 0 and a positive alpha."""
    case = SYMBOLIC["cases"][key]
    omega = np.array([2.0 * math.pi * C / SYMBOLIC["provenance"]["wavelength_m"]])
    assert case["alpha"] > 0.0
    assert float(alpha(symbolic_mode(case), omega)[0]) == pytest.approx(
        case["alpha"], rel=1e-14
    )


@pytest.mark.parametrize("key", SOLPROP_CASES)
def test_beta2_and_zdw_match_the_symbolic_derivation_to_the_differencing_floor(key):
    """beta_2 is the one quantity solprop differences, so this is where it shows.

    Against an exact symbolic second derivative it lands at 1.3e-8 relative at worst,
    which is the Richardson floor quoted on BETA2_RELATIVE_STEP; the gate is 1e-7. The
    zero-dispersion wavelength inherits it and comes out at 7.5e-10, because the root
    is where beta_2 crosses zero steeply.
    """
    case = SYMBOLIC["cases"][key]
    mode = symbolic_mode(case)
    omega = np.array([2.0 * math.pi * C / SYMBOLIC["provenance"]["wavelength_m"]])
    assert float(beta2(mode, omega)[0]) == pytest.approx(case["beta"][2], rel=1e-7)
    with warnings.catch_warnings():
        # The witness brackets the root over 188 nm to 18.8 um, so the sweep steps
        # outside the argon fit's 400 nm to 1 um range on the way. The symbolic
        # derivation extrapolates the same expression over the same span, which is
        # what makes the comparison like for like.
        warnings.simplefilter("ignore", MaterialRangeWarning)
        root = zdw(mode, wavelength_limits=(200e-9, 3e-6))
    assert root == pytest.approx(case["zdw"], rel=1e-8)


# --------------------------------------------------------------- injected gas model


def luna_helium():
    """Return Luna's helium susceptibility: the same fit, its own reference state.

    Luna divides the Ermolov three-pole expansion by the 1 bar density although the fit
    is referenced to 1 atm, so its per-particle susceptibility is the 1 atm / 1 bar
    density ratio too large. The fit shape is identical, so the variant is solprop's
    former helium default (supplied in ``witness_materials``) with the reference state
    Luna assumed -- no separate table.
    """
    return replace(ermolov_helium(), reference_state=ONE_BAR_0C)


def test_an_injected_susceptibility_replaces_the_one_the_gas_selects():
    mode = MarcatiliMode(125e-6, "He", 0.8, susceptibility=luna_helium())
    injected = mode.susceptibility
    assert injected is not None
    assert injected.reference_state.pressure_bar == 1.0


def test_an_injected_susceptibility_leaves_the_density_path_alone():
    # The gas still sets the equation of state: only the refractive data are replaced.
    default = MarcatiliMode(125e-6, "He", 0.8)
    injected = MarcatiliMode(125e-6, "He", 0.8, susceptibility=luna_helium())
    position = np.array([0.0])
    assert float(injected.density(position)[0]) == float(default.density(position)[0])


def test_the_luna_variant_moves_the_refractivity_by_the_density_ratio():
    """The whole difference is one normalisation, so the shift is flat in frequency."""
    fit = ermolov_helium()
    # Inside the fit's 90-1000 nm range, so the comparison is of the model rather
    # than of its extrapolation.
    omega = np.linspace(0.8, 8.0, 32) * OMEGA_800
    ratio = luna_helium().susceptibility(omega) / fit.susceptibility(omega)
    expected = fit.reference_density / luna_helium().reference_density
    # A pure rescaling: exact to rounding, and the same at every frequency, which is
    # what distinguishes a normalisation error from a different fit.
    np.testing.assert_allclose(ratio, expected, rtol=1e-14)
    assert abs(expected - 1.01324) < 1e-5


def test_the_mode_level_source_is_gone():
    # SEP 3, Phase 4b: a named gas is optmat's; another index is supplied explicitly.
    with pytest.raises(TypeError, match="source"):
        MarcatiliMode(125e-6, "He", 0.8, source="borzsonyi")  # pyright: ignore[reportCallIssue]


def test_rotating_a_mode_keeps_its_susceptibility():
    # `replace` hands back the object the mode already holds, which is not a clash.
    for mode in (
        MarcatiliMode(125e-6, "Kr", 1.0),
        MarcatiliMode(125e-6, "Kr", 1.0, susceptibility=borzsonyi("Kr")),
    ):
        assert mode.rotated(0.1).susceptibility is mode.susceptibility


def test_an_evacuated_capillary_still_has_no_susceptibility():
    assert MarcatiliMode(125e-6).susceptibility is None
