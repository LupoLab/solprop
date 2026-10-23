# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the hollow planar (slab) mode against closed forms, papers and RectMode."""

import inspect
import itertools
import json
import math
import warnings
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import erf, jn_zeros
from witness_materials import ermolov_helium, kingston_argon, malitson_silica

from bounds import assert_within
from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, ModeError
from solprop.modes import (
    MarcatiliMode,
    ModeCollection,
    RectMode,
    SlabMode,
    SlabPolarisation,
    gaussian_coupling,
    gaussian_to_per_unit_width,
    on_axis_rule,
    per_unit_width_to_gaussian,
    taper_scale,
    transverse_measure,
)
from solprop.modes.protocol import PolarisationComponent
from solprop.modes.quadrature import (
    cartesian_nodes,
    mode_matrix,
    quadrature_rule,
    transverse_quadrature,
)
from solprop.physdata import Mixture, aluminium, silver
from solprop.profiles import Linear

EPS = np.finfo(float).eps
OMEGA_800 = np.asarray([2 * np.pi * C / 800e-9])


def field_attenuation(mode, omega=OMEGA_800, z=0.0):
    """Return Im(beta), the field attenuation in 1/m."""
    return np.imag(mode.neff(omega, z)) * omega / C


def profile(order, xi):
    """Return the slab profile of one order on the unit half-gap."""
    phase = order * math.pi * xi / 2
    return math.cos(phase) if order % 2 else math.sin(phase)


# --------------------------------------------------------------- the closed forms


@pytest.mark.parametrize("wall", ["glass", "silver", "aluminium"])
def test_slab_attenuation_matches_the_first_order_closed_form(wall):
    """alpha_p = (1/a)(p lambda / 4a)^2 Re q (Laakmann eq. 5, b -> infinity).

    Evacuated, so eps_co = 1 and q_TE = 1/sqrt(eps_cl - 1), q_TM = eps_cl q_TE are
    computed here from the wall permittivity alone. The model keeps the squared
    complex factor (1 - i lambda q / 2 pi a)^2, which the closed form truncates at
    first order: their relative difference is O(lambda |q| / 2 pi a) plus the
    O((p lambda / 4a)^2) of the square root, and the bound is three times their sum.
    """
    cladding = {"glass": 1.45, "silver": silver(), "aluminium": aluminium()}[wall]
    a = 500e-6
    wavelength = 800e-9
    if isinstance(cladding, float):
        permittivity = complex(cladding**2)
    else:
        permittivity = complex(cladding.permittivity(OMEGA_800)[0])
    te = 1 / np.sqrt(permittivity - 1)
    coefficients: dict[SlabPolarisation, complex] = {
        "te": te,
        "tm": permittivity * te,
    }
    for polarisation, q in coefficients.items():
        for order in (1, 2, 3):
            mode = SlabMode(
                a, cladding=cladding, order=order, polarisation=polarisation
            )
            closed = (order * wavelength / (4 * a)) ** 2 * np.real(q) / a
            bound = 3 * (
                wavelength * abs(q) / (2 * math.pi * a)
                + (order * wavelength / (4 * a)) ** 2
            )
            assert_allclose(field_attenuation(mode), closed, rtol=bound, atol=0)


def test_slab_loss_scales_as_order_squared_and_inverse_gap_cubed():
    """p^2 and a^-3, each to the first-order truncation bound of the test above."""
    a = 400e-6
    base = field_attenuation(SlabMode(a, cladding=1.45))[0]
    for order in (2, 3, 5):
        ratio = field_attenuation(SlabMode(a, cladding=1.45, order=order))[0] / base
        # Only the O((p lambda/4a)^2) square-root term depends on the order.
        assert_allclose(ratio, order**2, rtol=3 * (order * 800e-9 / (4 * a)) ** 2)
    ratio = base / field_attenuation(SlabMode(2 * a, cladding=1.45))[0]
    # The O(lambda |q| / 2 pi a) term is what depends on the gap.
    assert_allclose(ratio, 8.0, rtol=3 * 800e-9 / (2 * math.pi * a) / 1.05)


def test_slab_tm_over_te_is_the_wall_permittivity_ratio():
    """TM/TE is eps_cl/eps_co for glass and about n^2 + kappa^2 for a metal (Miyagi 36).

    For glass the coefficient ratio is exact; the model adds only the first-order
    truncation of the squared factor. Miyagi's metal form drops the -1 beside
    eps_cl and assumes kappa >> n, so it holds to O(1/|eps_cl|), the bound used.
    """
    a = 500e-6
    te = field_attenuation(SlabMode(a, cladding=1.45))[0]
    tm = field_attenuation(SlabMode(a, cladding=1.45, polarisation="tm"))[0]
    assert_allclose(
        tm / te, 1.45**2, rtol=3 * 800e-9 * 1.45**2 / (2 * math.pi * a * 1.05)
    )
    index = complex(np.sqrt(silver().permittivity(OMEGA_800)[0]))
    te = field_attenuation(SlabMode(a, cladding=silver()))[0]
    tm = field_attenuation(SlabMode(a, cladding=silver(), polarisation="tm"))[0]
    assert_allclose(tm / te, abs(index) ** 2, rtol=3 / abs(index) ** 2)
    # A gas fill moves the TM coefficient by eps_co.
    filled = SlabMode(a, "Ar", 10.0, cladding=1.45, polarisation="tm")
    assert field_attenuation(filled)[0] != tm


def test_akturk_and_arnold_numbers_are_reproduced():
    """Akturk et al. OL 34, 1462 (2009) and Arnold et al. NJP 12, 073015 (2010).

    Akturk: 2a = 127 um, 800 nm, silica n = 1.45, TE, 20 cm: 5.8 % power loss for
    p = 1 and 41 % for p = 3, printed to the rounding of their last digit. Arnold:
    p = 3 is damped nine times and p = 2 four times more than p = 1, to the
    square-root truncation (p lambda / 4a)^2.
    """
    first = SlabMode(63.5e-6, cladding=1.45)
    third = replace(first, order=3)
    lost = [
        1 - math.exp(-2 * field_attenuation(mode)[0] * 0.2) for mode in (first, third)
    ]
    assert 0.0575 <= lost[0] < 0.0585
    assert 0.405 <= lost[1] < 0.415
    base = field_attenuation(first)[0]
    for order, factor in ((2, 4.0), (3, 9.0)):
        ratio = field_attenuation(replace(first, order=order))[0] / base
        assert_allclose(ratio, factor, rtol=3 * (order * 800e-9 / (4 * 63.5e-6)) ** 2)


def test_the_te_fundamental_loses_about_a_quarter_of_the_capillary():
    """At half-gap = radius, TE1/HE11 = 1/(8 (u11/2 pi)^2 (n^2 + 1)) and TM1 n^2 that.

    Marcatili-Schmeltzer against the slab form, both first order in lambda/a, so the
    ratio holds to their truncations, bounded by lambda n^2 / (2 pi a) with a factor
    of three.
    """
    a = 250e-6
    capillary = field_attenuation(MarcatiliMode(a, cladding=1.45))[0]
    u11 = float(jn_zeros(0, 1)[0])
    expected = 1 / (8 * (u11 / (2 * math.pi)) ** 2 * (1.45**2 + 1))
    bound = 3 * 800e-9 * 1.45**2 / (2 * math.pi * a)
    te = field_attenuation(SlabMode(a, cladding=1.45))[0] / capillary
    tm = field_attenuation(SlabMode(a, cladding=1.45, polarisation="tm"))[0] / capillary
    assert_allclose(te, expected, rtol=bound)
    assert_allclose(tm, 1.45**2 * expected, rtol=bound)
    assert 0.27 < te < 0.28 and 0.57 < tm < 0.58


@pytest.mark.parametrize(("polarisation", "partner"), [("te", "y"), ("tm", "x")])
def test_the_rectangle_approaches_the_slab_as_its_height_grows(polarisation, partner):
    """RectMode(a, b) -> SlabMode(a) as b grows: phase as b^-2 and loss as b^-3.

    The rectangle's y wall pair adds ``(n lambda / 4b)^2 (1 - i lambda q / 2 pi b)^2``
    to the decrement, whose real part is O(b^-2) and imaginary part O(b^-3). The
    square root also crosses it with the x pair's own loss, an O(b^-2) imaginary
    term of relative size ``b (p lambda / 4a)^2 Re q_x / (a Re q_y)`` -- at most
    1.1e-3 here (TM, b = 1.6 mm) -- which is what bounds the observed order: three
    doublings of b, 1e-3 on the order (measured 3.9e-4 at worst).
    """
    a = 50e-6
    slab = SlabMode(a, cladding=1.45, polarisation=polarisation)
    reference = slab.neff_minus_one(OMEGA_800)[0]
    differences = [
        RectMode(a, b, cladding=1.45, polarisation=partner).neff_minus_one(OMEGA_800)[0]
        - reference
        for b in (0.2e-3, 0.4e-3, 0.8e-3, 1.6e-3)
    ]
    for coarse, fine in itertools.pairwise(differences):
        assert_allclose(math.log2(coarse.real / fine.real), 2.0, atol=1e-3)
        assert_allclose(math.log2(coarse.imag / fine.imag), 3.0, atol=1e-3)


def baseline_modes():
    # The table held the former defaults' numbers (the registry's gases, silica and
    # stitched metal tables), which SEP 3's Phase 4b deleted; it is now recorded on
    # the gases and silica supplied explicitly (Kingston argon, Ermolov helium,
    # Malitson's silica: ``witness_materials``), optmat's metals and optmat's mixture.
    taper = Linear(50e-6, 40e-6, 0.1)
    silica = malitson_silica()
    return (
        RectMode(
            50e-6, 12.5e-6, "Ar", 5.0, susceptibility=kingston_argon(), cladding=1.45
        ),
        RectMode(
            50e-6,
            12.5e-6,
            "Ar",
            5.0,
            susceptibility=kingston_argon(),
            cladding=silica,
            polarisation="y",
            n=2,
            m=3,
        ),
        RectMode(
            taper,
            30e-6,
            "He",
            1.0,
            susceptibility=ermolov_helium(),
            cladding=silver(),
            polarisation="y",
        ),
        RectMode(taper, taper, cladding=aluminium(), n=3, m=2),
        RectMode(
            50e-6,
            20e-6,
            mixture=Mixture.fill({"He": 0.25, "Ar": 0.75}, 2.0),
            cladding=0.05 + 3.5j,
            polarisation="y",
        ),
    )


def test_rectmode_is_unchanged_by_the_shared_wall_pair():
    """The wall-pair extraction left every RectMode number where it was.

    Each mode's ``neff``, ``neff - 1`` and its derivative at each position is compared
    normwise with the record at 64 eps: the ladder's one-evaluation budget of 16 eps
    (``accuracy.md`` section 7) for a closed form of a few tens of roundings, times
    four for another platform's libm in the wall's complex square roots and the
    transcendental terms. The signature and the dataclass fields stay exact.

    The table was written on main 5fcfa7cf, before the extraction: five guides --
    glass, silver, aluminium and a constant complex wall; argon, helium and a
    mixture; both polarisations; (1,1), (2,3), (3,2); constant, one-axis and
    two-axis tapers -- at 31 wavelengths and three positions.
    """
    reference = json.loads(
        (Path(__file__).parents[1] / "data/rectangular_before_slab.json").read_text()
    )
    assert str(inspect.signature(RectMode)) == reference["signature"]
    assert [field.name for field in fields(RectMode)] == reference["fields"]
    omega = 2 * np.pi * C / np.linspace(600e-9, 900e-9, 31)
    with warnings.catch_warnings():
        # Helium and argon fits are outside their published range at 600 nm; the
        # table was captured with the same notices silenced.
        warnings.simplefilter("ignore")
        values = np.asarray(
            [
                [getattr(mode, name)(omega, z) for z in (0.0, 0.05, 0.1)]
                for mode in baseline_modes()
                for name in ("neff", "neff_minus_one", "dneff_domega")
            ]
        )
    expected = np.asarray(reference["real"]) + 1j * np.asarray(reference["imag"])
    assert values.shape == expected.shape
    for index in np.ndindex(values.shape[:2]):
        assert_within(values[index], expected[index], 64 * EPS, what=str(index))


# --------------------------------------------------------------- the derivative


@pytest.mark.parametrize("polarisation", ["te", "tm"])
def test_slab_derivative_by_contour_and_finite_difference(polarisation):
    mode = SlabMode(50e-6, cladding=0.05 + 3.5j, order=2, polarisation=polarisation)
    omega = 2 * np.pi * C / np.linspace(300e-9, 2e-6, 7)
    terms = mode._terms(omega, 0.0)
    total = np.zeros_like(omega, dtype=complex)
    for phase in np.arange(64) * 2 * np.pi / 64:
        offset = 0.1 * omega * np.exp(1j * phase)
        value = mode._model_values(replace(terms, omega=omega + offset), 50e-6)[0]
        total = total + value / offset
    # RectMode's contour tolerance: rounding amplified by the Cauchy radius, with
    # exponentially small 64-node trapezoid truncation.
    assert_allclose(mode.dneff_domega(omega), total / 64, rtol=1e-11, atol=0)
    omega = 2 * np.pi * C / np.linspace(450e-9, 900e-9, 7)
    h = omega * 1e-3
    for material in (
        SlabMode(50e-6, "Ar", 10.0, polarisation=polarisation),
        SlabMode(50e-6, "Ar", 10.0, cladding=silver(), polarisation=polarisation),
    ):
        wide = (
            material.neff_minus_one(omega + h) - material.neff_minus_one(omega - h)
        ) / (2 * h)
        narrow = (
            material.neff_minus_one(omega + h / 2)
            - material.neff_minus_one(omega - h / 2)
        ) / h
        # Richardson h^4 truncation and cancellation at gas refractivities ~1e-3,
        # RectMode's bound for the same material chain.
        assert_allclose(
            material.dneff_domega(omega), (4 * narrow - wide) / 3, rtol=1e-8, atol=0
        )


# --------------------------------------------------------------- the rule


def test_slab_closed_forms_and_orthonormality_on_the_rule():
    """N = eps0 c a / 2 and A_eff = 4a/3 per unit width; the Gram matrix is one.

    Six orders on cartesian_nodes(6) = 42 Gauss nodes: the quadratic products of
    the Gram matrix are resolved far below the quartic bound, so 1e-12 is the
    reduction's rounding with room to spare.
    """
    a = 63.5e-6
    modes = [SlabMode(a, order=order) for order in range(1, 7)]
    rule = transverse_quadrature(modes, n_r=cartesian_nodes(6), radial_rule="gauss")
    assert rule.kind == "slab" and not rule.full and rule.n_nodes == 42
    limits = modes[0].dimlimits()
    assert limits == ("slab", (-a,), (a,))
    for mode in modes:
        intensity = np.sum(mode.field(rule.nodes(limits)) ** 2, axis=1)
        weights = rule.weights(limits)
        square, quartic = weights @ intensity, weights @ intensity**2
        assert_allclose(EPS0 * C / 2 * square, mode.normalisation(), rtol=1e-12)
        assert_allclose(square**2 / quartic, mode.aeff(), rtol=1e-12)
    matrices = mode_matrix(modes, rule, components=("y",))
    assert_allclose(matrices.projection @ matrices.synthesis, np.eye(6), atol=1e-12)
    # The adaptive overlap, independent of the rule.
    assert_allclose(ModeCollection(modes).orthonormality(), np.eye(6), atol=1e-12)


def test_slab_quartic_bound_is_the_cartesian_one():
    """Every quartic product of orders <= h reaches 1e-12 at cartesian_nodes(h).

    The reference is an adaptive integral on the unit half-gap at 1e-14; the
    bound is relative to int V^4 = 3/4, the scale of the Kerr projection. One
    node fewer than the measured count is not enough for h = 3 (22 measured),
    which is what makes it a bound rather than a guess.
    """
    for highest in (1, 2, 3, 4):
        count = cartesian_nodes(highest)
        rule = quadrature_rule("slab", full=False, n_r=count)
        x, w = rule.xi_r, rule.w_r
        assert w is not None
        for orders in (
            (i, j, k, m)
            for i in range(1, highest + 1)
            for j in range(i, highest + 1)
            for k in range(j, highest + 1)
            for m in range(k, highest + 1)
        ):
            values = np.prod([[profile(p, one) for one in x] for p in orders], axis=0)
            reference = quad(
                lambda xi, orders=orders: math.prod(profile(p, xi) for p in orders),
                -1,
                1,
                epsabs=1e-14,
                epsrel=1e-14,
                limit=200,
            )[0]
            assert abs(w @ values - reference) < 0.75e-12
    rule = quadrature_rule("slab", full=False, n_r=21)
    x, w = rule.xi_r, rule.w_r
    values = np.array([profile(3, one) ** 4 for one in x])
    assert abs(w @ values - 0.75) > 0.75e-12


def test_slab_rule_kronrod_alarm_on_axis_and_warnings():
    modes = ModeCollection.slab(63.5e-6, (1, 3, 5), "Ar", 1.0)
    rule = transverse_quadrature(modes.modes, n_r=36, radial_rule="gauss_kronrod")
    assert rule.kronrod and rule.has_embedded_rule and rule.n_r == 37
    limits = modes.dimlimits()
    assert_allclose(rule.weights(limits).sum(), 2 * 63.5e-6, rtol=1e-14)
    assert_allclose(rule.weights(limits, coarse=True).sum(), 2 * 63.5e-6, rtol=1e-14)
    with pytest.warns(UserWarning, match="x node count of a slab rule"):
        transverse_quadrature(modes.modes, n_r=12)
    with pytest.raises(ModeError, match="uniform in y"):
        transverse_quadrature(modes.modes, n_r=36, full=True)
    axis = on_axis_rule("slab")
    assert axis.kind == "slab" and not axis.full
    on_axis = mode_matrix(modes.modes, axis, components=("x", "y")).synthesis
    # cos(0) = 1 on the y component of each odd TE order, over sqrt(N).
    expected = 1 / math.sqrt(SlabMode(63.5e-6, "Ar", 1.0).normalisation())
    assert_allclose(on_axis[1], expected, rtol=1e-15)
    # A TE slab mode has no x component: an explicit zero row.
    assert_array_equal(on_axis[0], 0.0)


# --------------------------------------------------------------- tapers


def test_slab_taper_is_scale_invariant_and_rescales_by_the_root_of_the_gap():
    """The normalised field goes as a^-1/2 and the weights as a (N = eps0 c a / 2)."""
    path = Linear(64e-6, 16e-6, 0.1)
    modes = [SlabMode(path, cladding=1.45, order=order) for order in (1, 3)]
    assert all(mode.scale_invariant for mode in modes)
    assert not modes[0].z_constant and not modes[0].radius_is_constant
    # 16e-6 is a quarter of 64e-6 exactly, so the ratio carries no rounding.
    assert taper_scale(modes[0], 0.1) == 2.0
    rule = transverse_quadrature(modes, n_r=24, radial_rule="gauss")
    rescaled = mode_matrix(modes, rule, 0.0).rescaled(taper_scale(modes[0], 0.05))
    direct = mode_matrix(modes, rule, 0.05)
    # The profiles at x = a xi agree to the rounding of the phase p pi x / 2a,
    # which moves a value near a node of the profile by a few eps of the peak;
    # the scalar normalisation adds one rounding. Four eps of the largest entry.
    for moved, built in (
        (rescaled.synthesis, direct.synthesis),
        (rescaled.projection, direct.projection),
    ):
        scale = float(np.max(np.abs(built)))
        assert_allclose(moved, built, rtol=0, atol=4 * EPS * scale)
    assert modes[0].aeff(0.1) == 0.25 * modes[0].aeff()
    assert modes[0].normalisation(0.1) == 0.25 * modes[0].normalisation()
    values = {
        name: float(one(np.asarray([0.05]))[0])
        for name, one in modes[0].parameter_paths.items()
    }
    frozen = modes[0].at_parameters(values)
    assert frozen.z_constant
    assert_allclose(float(frozen.width_x(np.asarray([0.0]))[0]), 40e-6, rtol=2 * EPS)
    # The same closed form at the sampled parameters: the one-evaluation 16 eps.
    assert_within(frozen.neff(OMEGA_800), modes[0].neff(OMEGA_800, 0.05), 16 * EPS)
    assert modes[0].at_parameters({}).half_gap is path
    with pytest.raises(ModeError, match="unknown slab parameters"):
        modes[0].at_parameters({"radius": 1.0})
    with pytest.raises(ModeError, match="half-gap must be positive"):
        modes[0].at_parameters({"half_gap": -1.0})


# --------------------------------------------------------------- coupling and y


def test_the_arnold_coupling_and_its_optima():
    """Arnold 2010 section 3.1: at w = 0.735a, 98.8 % of the beam enters p = 1.

    99.4 % of the power falling inside the gap does. Two optima exist and the
    tests pin both: the share of the power **inside the gap**
    that enters p = 1 peaks at 0.7353a -- Arnold's 0.735a, and Laakmann's 0.74a
    for the square guide -- while the share of the **whole** beam peaks at
    0.7032a (98.93 %), Laakmann's 0.69a analogue. The optimiser's 1e-6 tolerance
    on w/a is far below the three digits compared.
    """
    modes = [SlabMode(1.0, order=order) for order in (1, 2, 3, 5)]
    shares = gaussian_coupling(modes, 0.735)
    assert round(shares[0], 3) == 0.988
    # 0.99464: Arnold prints "about 99.4 %", truncating.
    assert 0.994 <= shares[0] / erf(math.sqrt(2) / 0.735) < 0.995
    assert shares[1] < 1e-30  # a centred beam does not reach an even order
    assert shares.sum() < erf(math.sqrt(2) / 0.735)

    def whole(waist):
        return -gaussian_coupling(modes[:1], waist)[0]

    def inside(waist):
        return whole(waist) / erf(math.sqrt(2) / waist)

    in_gap = minimize_scalar(inside, bounds=(0.6, 0.9), method="bounded").x
    total = minimize_scalar(whole, bounds=(0.6, 0.9), method="bounded").x
    assert 0.72 <= in_gap <= 0.75 and round(in_gap, 3) == 0.735
    assert round(total, 3) == 0.703
    # Akturk 2009: a 50 um radius into 2a = 127 um puts 98 % in p = 1, < 1 % in p = 3.
    akturk = gaussian_coupling(
        [SlabMode(63.5e-6, order=order) for order in (1, 3)], 50e-6
    )
    assert round(akturk[0], 2) == 0.98 and akturk[1] < 0.01


def test_gaussian_coupling_matches_an_adaptive_overlap():
    """The rule's projection against an adaptive integral, offset beam, even orders.

    256 Gauss nodes against quad at 1e-12, split at the beam centre: the bound is
    ten times quad's own tolerance, for the share's quadratic dependence on the
    overlap.
    """
    a = 50e-6
    modes = [SlabMode(a, order=order) for order in range(1, 6)]
    for waist, offset in ((0.7 * a, 0.0), (0.4 * a, 0.2 * a), (0.05 * a, -0.5 * a)):
        shares = gaussian_coupling(modes, waist, offset=offset)
        for order, share in enumerate(shares, start=1):
            overlap = quad(
                lambda x, order=order, waist=waist, offset=offset: (
                    math.exp(-(((x - offset) / waist) ** 2)) * profile(order, x / a)
                ),
                -a,
                a,
                points=[min(max(offset, -a), a)],
                epsabs=1e-13 * waist,
                epsrel=1e-12,
                limit=400,
            )[0]
            expected = overlap**2 / (a * waist * math.sqrt(math.pi / 2))
            assert_allclose(share, expected, rtol=1e-11, atol=1e-14)
    tm = gaussian_coupling([SlabMode(a, polarisation="tm")], 0.7 * a)
    assert_allclose(tm, gaussian_coupling([SlabMode(a)], 0.7 * a), rtol=0)
    # One linearly polarised beam launches one family: a TE/TM set is refused
    # without a component, and with one the other family takes exactly nothing.
    pair = list(ModeCollection.with_polarisations([SlabMode(a)]))
    with pytest.raises(ModeError, match="TE and TM modes and no component="):
        gaussian_coupling(pair, 0.7 * a)
    launches: tuple[tuple[PolarisationComponent, int], ...] = (("y", 0), ("x", 1))
    for component, taken in launches:
        shares = gaussian_coupling(pair, 0.7 * a, component=component)
        assert_allclose(shares[taken], tm[0], rtol=0)
        assert shares[1 - taken] == 0.0
    assert gaussian_coupling(pair, 0.7 * a, component="y").sum() < 1
    with pytest.raises(ModeError, match="at least one"):
        gaussian_coupling([], 1.0)
    with pytest.raises(ModeError, match="positive"):
        gaussian_coupling(modes, 0.0)
    with pytest.raises(ModeError, match="polar cross-section"):
        gaussian_coupling([MarcatiliMode(50e-6)], 1e-5)


def test_the_gaussian_in_y_helpers_round_trip():
    """P = P' w_y sqrt(pi/2) for a 1/e field radius w_y (Luna's planar_params)."""
    w_y = 2.5e-3
    total = per_unit_width_to_gaussian(3.0, w_y)
    assert_allclose(total, 3.0 * w_y * math.sqrt(math.pi / 2), rtol=EPS)
    assert_allclose(gaussian_to_per_unit_width(total, w_y), 3.0, rtol=2 * EPS)
    # The integral that defines the width, done adaptively.
    width = quad(lambda y: math.exp(-2 * (y / w_y) ** 2), -np.inf, np.inf)[0]
    assert_allclose(per_unit_width_to_gaussian(1.0, w_y), width, rtol=1e-12)
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ModeError, match="w_y"):
            gaussian_to_per_unit_width(1.0, bad)


# --------------------------------------------------------------- the mode's surface


def test_slab_labels_components_partners_and_measure():
    te = SlabMode(50e-6, "Ar", 1.0, order=3)
    tm = te.orthogonal_polarisation
    assert te.channel_label.display_label == "TE3"
    assert tm.channel_label.display_label == "TM3" and tm.channel_label.family == "TM"
    assert te.principal_component() == "y" and tm.principal_component() == "x"
    assert tm.orthogonal_polarisation.polarisation == "te"
    assert not te.polarisation_degenerate and te.azimuthal_order == 3
    assert te.transverse_measure == "per_unit_width"
    assert abs(te.neff(OMEGA_800)[0] - tm.neff(OMEGA_800)[0]) > 0
    pairs = ModeCollection.with_polarisations([te])
    assert [label.display_label for label in pairs.channel_labels] == ["TE3", "TM3"]
    assert pairs.transverse_measure == "per_unit_width"
    points = np.asarray([[0.0, 7.0], [50e-6, 0.0], [51e-6, 0.0], [-60e-6, 1.0]])
    values = te.field(points)
    assert_allclose(values[:, 0], 0.0, atol=0)
    # cos(3 pi x / 2a): one at the centre, rounding-level zero at the plate,
    # exactly zero outside, and independent of y.
    assert values[0, 1] == 1.0 and abs(values[1, 1]) < 1e-15
    assert values[2, 1] == 0.0 and values[3, 1] == 0.0
    even = SlabMode(50e-6, order=2)
    assert_allclose(even.field(np.asarray([[25e-6, 0.0]]))[0, 1], 1.0, rtol=EPS)
    assert "TE3" in repr(te) and "gap 100 um" in repr(te)
    assert "per unit width" in str(te)
    assert "SlabMode TM3" in str(tm)
    assert "evacuated" in repr(SlabMode(50e-6))


def test_slab_collection_constructor_and_measure_refusals():
    modes = ModeCollection.slab(
        Linear(60e-6, 50e-6, 0.1), (1, 3, 5), mixture=Mixture.fill({"Ar": 1.0}, 1.0)
    )
    assert [label.display_label for label in modes.channel_labels] == [
        "TE1",
        "TE3",
        "TE5",
    ]
    assert not modes.needs_full_rule and modes.scale_invariant
    assert transverse_measure([MarcatiliMode(125e-6)]) == "area"
    with pytest.raises(ModeError, match="at least one order"):
        ModeCollection.slab(50e-6, ())
    with pytest.raises(ModeError, match="duplicate physical channel"):
        ModeCollection.slab(50e-6, (1, 1))
    with pytest.raises(ModeError, match=r"MarcatiliMode\) measures 'area'"):
        ModeCollection([SlabMode(50e-6), MarcatiliMode(50e-6)])
    with pytest.raises(ModeError, match="per_unit_width"):
        transverse_measure([RectMode(50e-6, 20e-6), SlabMode(50e-6)])
    with pytest.raises(ModeError, match="at least one mode"):
        transverse_measure([])
    with pytest.raises(ModeError, match="share a cross-section"):
        ModeCollection([SlabMode(50e-6), SlabMode(60e-6, order=3)])


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"order": 0}, "order must be an integer"),
        ({"order": True}, "order must be an integer"),
        ({"half_gap": 0.0}, "half-gap must be positive"),
        ({"half_gap": Linear(50e-6, -1e-6, 0.1)}, "half-gap must be positive"),
        ({"gas": "Ar", "mixture": Mixture.fill({"He": 1.0}, 1.0)}, "mixture carries"),
        ({"loss": -1.0}, "loss scale"),
    ],
)
def test_slab_constructor_refusals(keywords, message):
    arguments = {"half_gap": 50e-6, **keywords}
    with pytest.raises(ModeError, match=message):
        SlabMode(**arguments)


def test_slab_quadrature_and_model_refusals():
    # The runtime type hook refuses an unknown polarisation first in the test
    # session; production has only the constructor's own check, reached here.
    mode = SlabMode(50e-6)
    object.__setattr__(mode, "polarisation", "y")
    with pytest.raises(ModeError, match="slab polarisation"):
        mode.__post_init__()
    with pytest.raises(ModeError, match="uniform in y"):
        quadrature_rule("slab", full=True, n_r=8)
    mode = SlabMode(50e-6)
    with pytest.raises(ModeError, match="finite with shape"):
        mode.field(np.full((1, 2), np.nan))
    with pytest.raises(ModeError, match="omega must be positive"):
        mode.neff(np.asarray([-1.0]))
    with pytest.raises(ModeError, match="wall contrast"):
        SlabMode(50e-6, cladding=1.0).neff(OMEGA_800)
    terms = mode._terms(OMEGA_800, 0.0)
    with np.errstate(invalid="ignore"), pytest.raises(ModeError, match="non-finite"):
        mode._model_values(replace(terms, chi=np.asarray([np.nan + 0j])), 50e-6)
    loss = replace(mode, loss=False)
    assert np.imag(loss.neff(OMEGA_800))[0] == 0.0


# ------------------------------------------------------------- first-order validity


def _validity_warnings(mode, omega):
    """Evaluate the mode twice and return the validity warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mode.neff(omega)
        mode.neff(omega)
        mode.at_parameters({"half_gap": 0.9e-6}).neff(omega)
    return [one for one in caught if issubclass(one.category, GrazingIncidenceWarning)]


def test_a_slab_past_its_first_order_ratios_warns_once():
    """p lambda / 4a = 0.2 at 800 nm on a 1 um half-gap, twice the 0.1.

    Once per mode: a second evaluation and a frozen copy of the same mode stay silent.
    """
    caught = _validity_warnings(SlabMode(1e-6, cladding=1.45), OMEGA_800)
    assert len(caught) == 1
    assert "p lambda / 4a = 0.2 at 800 nm" in str(caught[0].message)


def test_a_slab_within_its_first_order_ratios_is_silent():
    """On a 50 um half-gap both ratios are below 0.01 over 400 nm to 2 um."""
    omega = 2 * math.pi * C / np.linspace(400e-9, 2e-6, 64)
    for polarisation in ("te", "tm"):
        mode = SlabMode(50e-6, cladding=1.45, order=3, polarisation=polarisation)
        with warnings.catch_warnings():
            warnings.simplefilter("error", GrazingIncidenceWarning)
            mode.neff(omega)


def test_a_metal_slab_warns_on_its_wall_correction():
    """Silver's TM-like |q| puts the wall ratio past 0.1 before the grazing angle.

    On a 10 um half-gap at 2 um, p lambda / 4a is 0.05 while lambda |q| / (2 pi a) is
    0.47 for the TM-like (E normal to the plates) reflection, |q| = 14.6.
    """
    mode = SlabMode(10e-6, cladding=silver(), polarisation="tm")
    with pytest.warns(GrazingIncidenceWarning, match=r"lambda \|q\| / \(2 pi a\)"):
        mode.neff(np.array([2 * math.pi * C / 2e-6]))
