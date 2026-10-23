# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import itertools
import json
import math
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.special import roots_legendre

from bounds import EPS, assert_within
from solprop.exceptions import ModeError
from solprop.modes import ChannelLabel, CrossSection, MarcatiliMode, RectMode
from solprop.modes.quadrature import (
    ModeMatrices,
    RadialRule,
    cartesian_nodes,
    gauss_kronrod,
    mode_matrix,
    on_axis_rule,
    quadrature_rule,
    taper_scale,
    transverse_quadrature,
)
from solprop.profiles import Linear

REFERENCE = json.loads((Path(__file__).parents[1] / "data/luna_modes.json").read_text())

RADIUS = 125e-6
DISC: tuple[CrossSection, tuple[float, float], tuple[float, float]] = (
    "polar",
    (0.0, 0.0),
    (RADIUS, 2.0 * math.pi),
)

# The normalised transverse fields satisfy (1/2) sqrt(eps0/mu0) int |Exy|^2 dA = 1 and
# the projection carries that (1/2) sqrt(eps0/mu0) beside the weights, so the discrete
# inner product of the mode matrix over the rule must be the identity: one on the
# diagonal and zero off it (docs/dev/conventions.md section 4).
GRAM_UNIT = 1.0


class DummyMode:
    """The smallest thing that satisfies the Mode protocol, for the error paths."""

    z_constant = True
    scale_invariant = True
    azimuthal_order = 0
    channel_label = ChannelLabel("dummy", None, None, None, None, "dummy")

    def principal_component(self, z=0.0):
        return None

    def neff(self, omega, z=0.0):
        return np.ones_like(omega, dtype=complex)

    def neff_minus_one(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def dneff_domega(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def aeff(self, z=0.0):
        return 1.0

    def dimlimits(
        self, z=0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        return DISC

    def field(self, points, z=0.0):
        return np.zeros((points.shape[0], 2))


def gram(matrices):
    """The discrete inner product of a mode set, on the rule it was built on."""
    return matrices.projection @ matrices.synthesis


# ------------------------------------------------------------- Gauss and Kronrod


@pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 16, 24])
def test_the_kronrod_extension_is_a_proper_embedded_pair(n):
    """Interlacing, positivity, and a coarse rule that is a strict subset."""
    nodes, weights, coarse = gauss_kronrod(n)
    assert nodes.size == 2 * n + 1
    assert np.all(np.diff(nodes) > 0.0)
    assert np.all(np.abs(nodes) < 1.0)
    assert np.all(weights > 0.0)
    assert np.all(coarse >= 0.0)
    # The Gauss nodes are every other Kronrod node, which is what makes the two rules
    # share nodes and the estimate free.
    gauss_nodes, gauss_weights = roots_legendre(n)
    np.testing.assert_allclose(nodes[1::2], gauss_nodes, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(coarse[1::2], gauss_weights, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(coarse[0::2], 0.0)
    assert float(weights.sum()) == pytest.approx(2.0, rel=1e-15)
    assert float(coarse.sum()) == pytest.approx(2.0, rel=1e-14)


@pytest.mark.parametrize("n", [2, 3, 7, 10, 16])
def test_the_kronrod_extension_is_exact_to_its_stated_degree(n):
    """Exact to degree 3n + 1, and not beyond: the defining property of the rule.

    A symmetric rule integrates every odd power exactly for free, so the teeth of this
    test are in the even degrees. The gate is 1e-13 absolute against integrals of
    order one, which is the conditioning of summing 2n + 1 terms of alternating sign
    at degree 50.
    """
    nodes, weights, _ = gauss_kronrod(n)
    for degree in range(3 * n + 2):
        exact = 2.0 / (degree + 1) if degree % 2 == 0 else 0.0
        assert float(np.sum(weights * nodes**degree)) == pytest.approx(exact, abs=1e-13)


@pytest.mark.parametrize("n", [2, 3, 7])
def test_the_kronrod_extension_is_not_exact_beyond_its_degree(n):
    """Which is what makes the exactness above a statement rather than a coincidence.

    Demonstrated at low order on purpose. The excess above the bound falls off fast
    with n, so by n = 16 the first even degree that visibly fails is 70 and by n = 24
    none does within double precision -- the rule is then so accurate that "not exact"
    stops being measurable, which says nothing about its stated degree.
    """
    nodes, weights, _ = gauss_kronrod(n)
    degree = 3 * n + 2 if n % 2 == 0 else 3 * n + 3
    exact = 2.0 / (degree + 1)
    assert abs(float(np.sum(weights * nodes**degree)) - exact) / exact > 1e-9


@pytest.mark.parametrize("n", [2, 3, 7, 10, 16])
def test_the_embedded_gauss_rule_is_exact_to_its_own_degree(n):
    nodes, _, coarse = gauss_kronrod(n)
    for degree in range(2 * n):
        exact = 2.0 / (degree + 1) if degree % 2 == 0 else 0.0
        assert float(np.sum(coarse * nodes**degree)) == pytest.approx(exact, abs=1e-13)


def test_the_kronrod_order_must_be_positive():
    with pytest.raises(ModeError, match="needs n >= 1"):
        gauss_kronrod(0)


@pytest.mark.parametrize("key", sorted(REFERENCE["kronrod"]))
def test_the_kronrod_rule_matches_quadgks(key):
    """Laurie's algorithm computed here against QuadGK's implementation of it.

    solprop computes the extension rather than shipping a table, so that any order is
    available; this is the check that the recursion was transcribed correctly. QuadGK
    stores the n + 1 non-positive nodes and mirrors them, so the stored payload is
    unpacked accordingly. Agreement is 5e-16 on the nodes and 1e-15 on the weights,
    which is rounding in two independent eigen-decompositions.
    """
    n = REFERENCE_KRONROD_ORDERS[key]
    stored = np.asarray(REFERENCE["kronrod"][key])
    half_nodes = stored[: n + 1]
    half_weights = stored[n + 1 : 2 * n + 2]
    gauss_weights = stored[2 * n + 2 :]
    expected_nodes = np.concatenate([half_nodes, -half_nodes[:-1][::-1]])
    expected_weights = np.concatenate([half_weights, half_weights[:-1][::-1]])
    order = np.argsort(expected_nodes)
    nodes, weights, coarse = gauss_kronrod(n)
    np.testing.assert_allclose(nodes, expected_nodes[order], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(weights, expected_weights[order], rtol=0.0, atol=1e-14)
    # QuadGK returns the Gauss weights for the positive half only, mirrored.
    stored_gauss = np.concatenate([gauss_weights, gauss_weights[::-1][n % 2 :]])
    np.testing.assert_allclose(
        np.sort(coarse[coarse > 0.0]), np.sort(stored_gauss), rtol=0.0, atol=1e-14
    )


REFERENCE_KRONROD_ORDERS = {"kronrod-7": 7, "kronrod-10": 10, "kronrod-16": 16}


@pytest.mark.parametrize("key", sorted(REFERENCE["gauss"]))
def test_the_gauss_rule_matches_quadgks(key):
    n = int(key.split("-")[1])
    stored = np.asarray(REFERENCE["gauss"][key])
    nodes, weights = roots_legendre(n)
    # scipy and QuadGK reach the same rule by different eigen-solvers, so the
    # agreement is a couple of ulps of a number of order one.
    np.testing.assert_allclose(nodes, stored[:n], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(weights, stored[n:], rtol=0.0, atol=1e-14)


# ----------------------------------------------------------------- the polar rule


@pytest.mark.parametrize("kronrod", [False, True])
def test_the_radial_rule_integrates_the_disc(kronrod):
    rule = quadrature_rule(
        "polar", full=False, n_r=24, radial_rule="gauss_kronrod" if kronrod else "gauss"
    )
    assert rule.n_theta == 1
    for coarse in (False, True):
        area = float(rule.weights(DISC, coarse=coarse).sum())
        assert area == pytest.approx(math.pi * RADIUS**2, rel=1e-14)


@pytest.mark.parametrize("kronrod", [False, True])
def test_the_radial_rule_integrates_a_power_of_the_radius(kronrod):
    """int r^4 dA over the disc is 2 pi a^6 / 6, and the rule is exact for it."""
    rule = quadrature_rule(
        "polar", full=False, n_r=24, radial_rule="gauss_kronrod" if kronrod else "gauss"
    )
    nodes = rule.nodes(DISC)
    weights = rule.weights(DISC)
    value = float(np.sum(weights * nodes[:, 0] ** 4))
    assert value == pytest.approx(2.0 * math.pi * RADIUS**6 / 6.0, rel=1e-13)


def test_the_kronrod_radial_rule_rounds_the_node_count_up():
    """2n + 1 >= n_r, so 24 becomes 25 and the embedded rule has 12 Gauss nodes."""
    rule = quadrature_rule("polar", full=False, n_r=24, radial_rule="gauss_kronrod")
    assert rule.n_r == 25
    assert rule.kronrod
    assert rule.has_embedded_rule
    assert rule.w_r_coarse is not None
    assert int(np.count_nonzero(rule.w_r_coarse)) == 12


def test_a_plain_gauss_rule_has_no_embedded_estimate():
    """And says so, because a zero error estimate is not a converged answer."""
    rule = quadrature_rule("polar", full=False, n_r=24)
    assert not rule.has_embedded_rule
    np.testing.assert_array_equal(rule.weights(DISC), rule.weights(DISC, coarse=True))


def test_the_full_polar_rule_integrates_an_azimuthal_product():
    """int r^2 cos^2(theta) dA = pi a^4 / 4."""
    rule = quadrature_rule("polar", full=True, n_r=16, n_theta=8)
    nodes = rule.nodes(DISC)
    for coarse in (False, True):
        weights = rule.weights(DISC, coarse=coarse)
        value = float(np.sum(weights * nodes[:, 0] ** 2 * np.cos(nodes[:, 1]) ** 2))
        assert value == pytest.approx(math.pi * RADIUS**4 / 4.0, rel=1e-13)


@pytest.mark.parametrize("n_theta", [4, 8, 16])
def test_the_theta_rule_is_exact_to_degree_n_theta_minus_one(n_theta):
    """The periodic trapezoid is a Gauss rule on the circle, and fails just above.

    Its exactness bound is what fixes the n_theta a quartic mode product needs, so it
    is asserted from both sides: every harmonic below the bound integrates to zero,
    and the one at the bound aliases onto the constant instead.
    """
    rule = quadrature_rule("polar", full=True, n_r=2, n_theta=n_theta)
    theta = rule.xi_theta
    weights = rule.w_theta
    for degree in range(1, n_theta):
        assert float(np.sum(weights * np.cos(degree * theta))) == pytest.approx(
            0.0, abs=1e-13
        )
        assert float(np.sum(weights * np.sin(degree * theta))) == pytest.approx(
            0.0, abs=1e-13
        )
    aliased = float(np.sum(weights * np.cos(n_theta * theta)))
    assert abs(aliased) == pytest.approx(2.0 * math.pi, rel=1e-13)


def test_the_theta_nodes_are_offset_from_the_axes():
    """The offset is the whole reason the fixed rule cannot be fooled as pcubature was.

    On-grid nodes at 0, pi/2, pi and 3 pi/2 annihilate E_x E_y for a superposition of
    hybrid modes, so two successive levels of an adaptive rule both return zero and it
    declares convergence on a wrong value.
    """
    rule = quadrature_rule("polar", full=True, n_r=2, n_theta=8)
    for node in rule.xi_theta:
        assert min(abs(node - k * math.pi / 2) for k in range(5)) > 0.1


def test_the_coarse_theta_rule_takes_every_other_node():
    rule = quadrature_rule("polar", full=True, n_r=2, n_theta=8)
    assert int(np.count_nonzero(rule.w_theta_coarse)) == 4
    assert float(rule.w_theta_coarse.sum()) == pytest.approx(2.0 * math.pi, rel=1e-15)


def test_an_odd_theta_count_has_no_embedded_rule():
    rule = quadrature_rule("polar", full=True, n_r=2, n_theta=5)
    np.testing.assert_array_equal(rule.w_theta, rule.w_theta_coarse)


# ------------------------------------------------------------ the cartesian rule


def test_the_cartesian_rule_integrates_a_rectangle():
    rule = quadrature_rule("cartesian", full=True, n_r=8, n_theta=6)
    limits = ("cartesian", (-2.0, -1.0), (2.0, 1.0))
    nodes = rule.nodes(limits)
    weights = rule.weights(limits)
    assert float(weights.sum()) == pytest.approx(8.0, rel=1e-14)
    value = float(np.sum(weights * nodes[:, 0] ** 2 * nodes[:, 1] ** 2))
    assert value == pytest.approx((2.0 * 8.0 / 3.0) * (2.0 / 3.0), rel=1e-14)


def test_a_cartesian_rule_cannot_be_azimuthally_symmetric():
    with pytest.raises(ModeError, match="no azimuthally symmetric form"):
        quadrature_rule("cartesian", full=False, n_r=8)


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "polar", "full": False, "n_r": 1}, "n_r >= 2"),
        ({"kind": "polar", "full": True, "n_r": 8, "n_theta": 1}, "n_theta >= 2"),
    ],
)
def test_a_bad_rule_is_refused(kwargs, message):
    with pytest.raises(ModeError, match=message):
        quadrature_rule(**kwargs)


def test_an_unknown_cross_section_is_refused():
    """The production path: under the test hook the Literal rejects it first.

    Without the guard an unknown kind would fall through to the Cartesian branch and
    build a rule that is silently wrong, so it is exercised through the function
    beartype wrapped rather than the wrapper.
    """
    unwrapped = getattr(quadrature_rule, "__wrapped__", quadrature_rule)
    bad = cast("CrossSection", "spherical")
    with pytest.raises(ModeError, match="kind must be"):
        unwrapped(bad, full=True, n_r=8)


def test_a_rule_refuses_a_cross_section_it_does_not_match():
    rule = quadrature_rule("polar", full=False, n_r=4)
    limits = ("cartesian", (-1.0, -1.0), (1.0, 1.0))
    with pytest.raises(ModeError, match="cross-section is cartesian"):
        rule.nodes(limits)
    with pytest.raises(ModeError, match="cross-section is cartesian"):
        rule.weights(limits)


# ---------------------------------------------------- choosing the rule for modes


def test_a_set_of_he1m_modes_needs_only_the_radial_rule():
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in (1, 2, 3, 4)]
    rule = transverse_quadrature(modes, n_r=24)
    assert not rule.full
    assert rule.n_nodes == 24


def test_a_set_with_azimuthal_structure_takes_the_full_rule():
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0),
        MarcatiliMode(RADIUS, "He", 1.0, n=2),
    ]
    rule = transverse_quadrature(modes, n_r=16, n_theta=16)
    assert rule.full
    assert rule.n_nodes == 16 * 16


def test_a_coarse_theta_rule_warns_with_the_bound_it_needs():
    """A warning, not an error: the caller may know what it will integrate. But it
    is the only thing between an aliased Kerr term and a plausible answer."""
    modes = [MarcatiliMode(RADIUS, "He", 1.0, n=4)]
    with pytest.warns(UserWarning, match=r"n_theta >= 13"):
        transverse_quadrature(modes, n_r=16, n_theta=8)


def test_a_coarse_cartesian_rule_warns_on_either_axis():
    """The same alarm for a rectangular set, whose two counts are x and y.

    Its bound is measured, not derived: 22 nodes per axis reach 1e-12 on the
    (3, 2) quartic product and `cartesian_nodes` rounds that to 24.
    """
    modes = [RectMode(50e-6, 12.5e-6, n=3, m=2)]
    assert cartesian_nodes(3) == 24
    for n_r, n_theta in ((12, 24), (24, 12)):
        with pytest.warns(UserWarning, match=r"24 nodes per axis"):
            transverse_quadrature(modes, n_r=n_r, n_theta=n_theta)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rule = transverse_quadrature(modes, n_r=24, n_theta=24)
    assert rule.n_nodes == 24 * 24


def test_the_full_rule_can_be_forced_either_way():
    modes = [MarcatiliMode(RADIUS, "He", 1.0)]
    assert transverse_quadrature(modes, n_r=8, full=True).full
    assert not transverse_quadrature(modes, n_r=8, n_theta=8, full=False).full


def test_a_rule_needs_a_mode():
    with pytest.raises(ModeError, match="at least one mode"):
        transverse_quadrature([], n_r=8)


# ------------------------------------------------------------ the mode matrices


def test_the_mode_matrices_have_the_kernel_contract_layout():
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in (1, 2)]
    rule = transverse_quadrature(modes, n_r=16, full=True, n_theta=8)
    matrices = mode_matrix(modes, rule)
    assert matrices.n_nodes == 128
    assert matrices.n_pol == 2
    assert matrices.synthesis.shape == (128 * 2, 2)
    assert matrices.projection.shape == (2, 128 * 2)
    assert matrices.projection_coarse.shape == (2, 128 * 2)
    # Row i * n_pol + p of the synthesis is component p at node i.
    points = rule.nodes(rule_limits(modes))
    field = modes[0].field(points) / math.sqrt(modes[0].normalisation())
    np.testing.assert_allclose(
        matrices.synthesis[:, 0].reshape(128, 2), field, rtol=0.0, atol=0.0
    )


def rule_limits(modes):
    return modes[0].dimlimits(0.0)


def test_a_scalar_run_carries_one_component():
    modes = [MarcatiliMode(RADIUS, "He", 1.0)]
    rule = transverse_quadrature(modes, n_r=8)
    matrices = mode_matrix(modes, rule, components=("x",))
    assert matrices.n_pol == 1
    assert matrices.synthesis.shape == (8, 1)


@pytest.mark.parametrize("components", [(), ("z",), ("x", "z")])
def test_bad_components_are_refused(components):
    modes = [MarcatiliMode(RADIUS, "He", 1.0)]
    rule = transverse_quadrature(modes, n_r=8)
    with pytest.raises(ModeError, match="non-empty subset"):
        mode_matrix(modes, rule, components=components)


def test_mode_matrices_need_a_mode():
    rule = quadrature_rule("polar", full=False, n_r=8)
    with pytest.raises(ModeError, match="at least one mode"):
        mode_matrix([], rule)


def test_modes_with_different_cross_sections_are_refused():
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0),
        MarcatiliMode(2 * RADIUS, "He", 1.0),
    ]
    rule = transverse_quadrature(modes, n_r=8)
    with pytest.raises(ModeError, match="share a cross-section"):
        mode_matrix(modes, rule)


def test_a_mode_without_a_closed_form_normalisation_is_refused():
    class Bare(DummyMode):
        def field(self, points, z=0.0):
            return np.ones((points.shape[0], 2))

    rule = quadrature_rule("polar", full=False, n_r=4)
    with pytest.raises(ModeError, match="provides no normalisation"):
        mode_matrix([Bare()], rule)


# ------------------------------------------------------------- orthonormality


@pytest.mark.parametrize("n_r", [24, 32])
def test_he1m_modes_are_orthonormal_on_the_radial_rule(n_r):
    """The rule integrates the mode overlaps exactly, which is what makes it usable.

    The normalisation comes from the closed form, not from the rule, so this is a
    real statement about the quadrature rather than a tautology. Measured deviation
    from the identity: 3e-16 at n_r = 24 -- rounding in a 24-term sum -- so the gate
    is 1e-14.
    """
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in range(1, 7)]
    rule = transverse_quadrature(modes, n_r=n_r)
    matrices = mode_matrix(modes, rule, components=("x",))
    deviation = np.max(np.abs(gram(matrices) / GRAM_UNIT - np.eye(len(modes))))
    assert deviation < 1e-14


def test_a_mixed_two_dimensional_set_is_orthonormal_on_the_full_rule():
    """HE, TE and TM together, each rotated, which is what needs the theta rule."""
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0, n=1, m=1),
        MarcatiliMode(RADIUS, "He", 1.0, n=1, m=2, phi=math.pi / 2),
        MarcatiliMode(RADIUS, "He", 1.0, n=2, m=1, phi=-math.pi / 4),
        MarcatiliMode(RADIUS, "He", 1.0, n=0, m=1, kind="TE"),
        MarcatiliMode(RADIUS, "He", 1.0, n=0, m=2, kind="TM"),
        MarcatiliMode(RADIUS, "He", 1.0, n=3, m=1),
    ]
    assert [mode.azimuthal_order for mode in modes] == [0, 0, 1, 1, 1, 2]
    rule = transverse_quadrature(modes, n_r=32, n_theta=16)
    matrices = mode_matrix(modes, rule)
    deviation = np.max(np.abs(gram(matrices) / GRAM_UNIT - np.eye(len(modes))))
    assert deviation < 1e-13


def test_the_coarse_rule_is_also_orthonormal():
    """It has to be, or the embedded estimate would measure the rule, not the field."""
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in (1, 2, 3)]
    rule = transverse_quadrature(modes, n_r=32, radial_rule="gauss_kronrod")
    matrices = mode_matrix(modes, rule, components=("x",))
    coarse = matrices.projection_coarse @ matrices.synthesis
    assert np.max(np.abs(coarse / GRAM_UNIT - np.eye(len(modes)))) < 1e-12


# ------------------------------------------------------- the Kerr overlap tensor


def kerr_tensor(modes, n_r, *, radial_rule: RadialRule = "gauss"):
    """The quartic overlap tensor the Kerr term projects through."""
    rule = transverse_quadrature(modes, n_r=n_r, radial_rule=radial_rule)
    matrices = mode_matrix(modes, rule, components=("x",))
    fields = matrices.synthesis.T
    weights = rule.weights(modes[0].dimlimits(0.0))
    return np.einsum("ap,bp,cp,dp,p->abcd", fields, fields, fields, fields, weights)


@pytest.mark.parametrize(("n_r", "kronrod"), [(32, False), (41, True)])
def test_the_kerr_overlap_tensor_is_converged(n_r, kronrod):
    """Exact by construction, which is the argument for a fixed rule (ADR-010).

    The Kerr polarisation is cubic in the field, so its projection is a quartic in the
    transverse profiles; for Bessel functions of r/a a Gauss rule of modest order
    integrates that to rounding. Against a 2048-node reference the deviation is 2e-14
    of the tensor's scale at n_r = 32 -- and that is the *reference* rounding, not the
    rule: summing 2048 terms of magnitude 1e13 carries a bound of 2048 eps, or 5e-13
    relative. Luna's design note reports the same convergence differently normalised
    (5e-16 at n_r = 24 for four HE_1m modes). The gate is that derived 2048-eps
    reference bound.
    """
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in range(1, 5)]
    reference = kerr_tensor(modes, 2048)
    tensor = kerr_tensor(
        modes, n_r, radial_rule="gauss_kronrod" if kronrod else "gauss"
    )
    relative = np.max(np.abs(tensor - reference)) / np.max(np.abs(reference))
    assert relative < 2048 * np.finfo(float).eps


def vector_quartics(modes, n_theta, *, contracted):
    """The quartic products of a mixed-family set's Cartesian fields on one rule.

    ``contracted`` returns ``int (psi_a . psi_b)(psi_c . psi_d) dA``: every isotropic
    Kerr contraction of a real basis is a permutation of it, so it is what the Kerr
    and Raman terms project through. Otherwise the uncontracted component products
    ``int psi_a^p psi_b^q psi_c^p psi_d^q dA``, the general quartic the rule claims.
    """
    rule = transverse_quadrature(modes, n_r=32, n_theta=n_theta, full=True)
    matrices = mode_matrix(modes, rule)
    fields = matrices.synthesis.T.reshape(len(modes), matrices.n_nodes, 2)
    weights = rule.weights(modes[0].dimlimits(0.0))
    if contracted:
        dot = np.einsum("anp,bnp->abn", fields, fields)
        return np.einsum("abn,cdn,n->abcd", dot, dot, weights)
    return np.einsum(
        "anp,bnq,cnp,dnq,n->abcdpq", fields, fields, fields, fields, weights
    )


MIXED_FAMILY_SETS = {
    # The vector wall's order nu = 1: the J_0 and J_2 families.
    "nu=1": [
        MarcatiliMode(50e-6, "He", 1.0, n=order, m=index)
        for order in (1, -1)
        for index in (1, 2)
    ],
    # Four families at once, and a set whose polarisation angles differ.
    "mixed": [
        MarcatiliMode(50e-6, "He", 1.0, n=1, m=1),
        MarcatiliMode(50e-6, "He", 1.0, n=1, m=2),
        MarcatiliMode(50e-6, "He", 1.0, kind="TE", n=0, m=1),
        MarcatiliMode(50e-6, "He", 1.0, kind="TM", n=0, m=1),
        MarcatiliMode(50e-6, "He", 1.0, n=-1, m=1),
    ],
    "rotated": [
        MarcatiliMode(50e-6, "He", 1.0, n=1, m=1),
        MarcatiliMode(50e-6, "He", 1.0, n=-1, m=1, phi=math.pi / 5),
        MarcatiliMode(50e-6, "He", 1.0, n=2, m=1, phi=math.pi / 7),
    ],
}


@pytest.mark.parametrize("name", sorted(MIXED_FAMILY_SETS))
def test_the_theta_rule_is_exact_for_a_mixed_family_kerr_contraction(name):
    """``4 h + 1`` angular nodes are exact for a mixed-family set.

    Bound, stated first: the rule is exact in exact arithmetic -- each Cartesian
    component of a Marcatili mode is a trigonometric polynomial of degree ``h`` in
    the angle, a quartic product has degree ``4 h``, and ``N`` equally spaced nodes
    integrate every harmonic below ``N`` -- so the only departure from a 257-node
    reference on the same radial rule is rounding: the reference sums 32 x 257 terms,
    whose bound is ``8224 eps`` = 1.8e-12 of the tensor's scale. Measured: 1.4e-14 to
    2.4e-14 for the contraction and the uncontracted products alike at ``4 h + 1``
    and at ``8 h + 1``. The contraction is exact on fewer nodes (three for "nu=1",
    five for "mixed", seven for "rotated") because the dot products cancel harmonics,
    but the uncontracted products are not: one node short, on ``4 h``, they depart by
    7e-2 to 8e-2, which is what makes ``4 h + 1`` the rule and the exactness above a
    measurement rather than a tautology.
    """
    modes = MIXED_FAMILY_SETS[name]
    highest = max(mode.azimuthal_order for mode in modes)
    assert highest == 2
    bound = 32 * 257 * np.finfo(float).eps
    for contracted in (True, False):
        reference = vector_quartics(modes, 257, contracted=contracted)
        scale = np.max(np.abs(reference))
        for n_theta in (4 * highest + 1, 8 * highest + 1):
            tensor = vector_quartics(modes, n_theta, contracted=contracted)
            assert np.max(np.abs(tensor - reference)) / scale < bound
    reference = vector_quartics(modes, 257, contracted=False)
    with pytest.warns(UserWarning, match="below the exactness bound"):
        short = vector_quartics(modes, 4 * highest, contracted=False)
    assert np.max(np.abs(short - reference)) / np.max(np.abs(reference)) > 1e-2


def test_a_rule_too_coarse_for_the_kerr_tensor_is_visibly_wrong():
    """Which is what makes the convergence above a statement rather than a hope."""
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in range(1, 5)]
    reference = kerr_tensor(modes, 512)
    tensor = kerr_tensor(modes, 12)
    assert np.max(np.abs(tensor - reference)) / np.max(np.abs(reference)) > 1e-6


def test_the_effective_overlap_tensor_is_symmetric_under_permutation():
    """A symmetric rule makes the spatial part of Kerr energy conservation exact at
    any order, independently of whether the quadrature has converged."""
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in (1, 2, 3)]
    tensor = kerr_tensor(modes, 9)
    for axes in itertools.permutations(range(4)):
        # Exact in exact arithmetic; einsum sums the same products in a different
        # order for each permutation, which costs a couple of ulps.
        np.testing.assert_allclose(
            np.transpose(tensor, axes), tensor, rtol=1e-14, atol=0.0
        )


# ------------------------------------------------------------- taper rescaling


def test_the_taper_rescaling_matches_re_evaluation():
    """Two scalars per position instead of thousands of Bessel evaluations.

    Not bitwise: rescaling multiplies by a ratio where re-evaluation divides by a
    different square root, so the two differ in the last bits. The agreement is 1.7e-14
    relative at worst over the three positions, which is a handful of ulps, and 1e-13
    is the gate.
    """
    mode = MarcatiliMode(Linear(100e-6, 50e-6, 1.0), "He", 1.0, m=2)
    modes = [mode]
    rule = transverse_quadrature(modes, n_r=12, n_theta=8, full=True)
    at_start = mode_matrix(modes, rule, 0.0)
    for z in (0.25, 0.5, 1.0):
        rescaled = at_start.rescaled(taper_scale(mode, z))
        rebuilt = mode_matrix(modes, rule, z)
        np.testing.assert_allclose(
            rescaled.synthesis, rebuilt.synthesis, rtol=1e-13, atol=0.0
        )
        np.testing.assert_allclose(
            rescaled.projection, rebuilt.projection, rtol=1e-13, atol=0.0
        )
        np.testing.assert_allclose(
            rescaled.projection_coarse, rebuilt.projection_coarse, rtol=1e-13, atol=0.0
        )


def test_the_taper_rescaling_leaves_the_round_trip_invariant():
    """S W is what the transform actually applies, and it must not move at all."""
    mode = MarcatiliMode(Linear(100e-6, 50e-6, 1.0), "He", 1.0)
    matrices = mode_matrix([mode], transverse_quadrature([mode], n_r=16))
    before = matrices.projection @ matrices.synthesis
    after_matrices = matrices.rescaled(2.0)
    after = after_matrices.projection @ after_matrices.synthesis
    np.testing.assert_allclose(after, before, rtol=1e-15, atol=0.0)


def test_taper_scale_is_the_radius_ratio():
    mode = MarcatiliMode(Linear(100e-6, 50e-6, 1.0), "He", 1.0)
    assert taper_scale(mode, 0.0) == pytest.approx(1.0, rel=1e-15)
    assert taper_scale(mode, 1.0) == pytest.approx(2.0, rel=1e-15)
    assert taper_scale(mode, 1.0, reference=1.0) == pytest.approx(1.0, rel=1e-15)


def test_a_rescaling_must_be_positive():
    modes = [MarcatiliMode(RADIUS, "He", 1.0)]
    matrices = mode_matrix(modes, transverse_quadrature(modes, n_r=4))
    with pytest.raises(ModeError, match="must be positive"):
        matrices.rescaled(0.0)


def test_taper_scale_refuses_a_mode_it_cannot_rescale():
    class NotInvariant(DummyMode):
        scale_invariant = False

    with pytest.raises(ModeError, match="not scale invariant"):
        taper_scale(NotInvariant(), 1.0)

    class Rectangular(DummyMode):
        def dimlimits(
            self, z=0.0
        ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
            return ("cartesian", (-1.0, -1.0), (1.0, 1.0))

    # A fixed Cartesian cross-section now supports the same scale contract.
    assert taper_scale(Rectangular(), 1.0) == 1.0


# ------------------------------------------------------------ against Luna's rule


@pytest.mark.parametrize("key", sorted(REFERENCE["rule_cases"]))
def test_the_nodes_and_weights_match_lunas(key):
    """Node for node and weight for weight, including the Jacobian and the ordering.

    This is where the halving of the radial weights under the map from [-1, 1] to
    (0, 1) is pinned: Luna's design note omits that factor in its formula while its
    code applies it, so the code is what solprop matches.
    """
    spec = REFERENCE["rule_cases"][key]
    stored = np.asarray(REFERENCE["rules"][key])
    rule = quadrature_rule(
        "polar",
        full=spec["full"],
        n_r=spec["n_r"],
        n_theta=spec["n_theta"],
        radial_rule="gauss_kronrod" if spec["kronrod"] else "gauss",
    )
    count = rule.n_nodes
    assert stored.size == 4 * count
    limits = ("polar", (0.0, 0.0), (spec["radius_m"], 2.0 * math.pi))
    expected_nodes = stored[: 2 * count].reshape(count, 2)
    # 1e-12 relative: scipy's roots_legendre and QuadGK's Golub-Welsch agree on the
    # nodes to a few ulps, and the weights inherit that through the r Jacobian.
    np.testing.assert_allclose(rule.nodes(limits), expected_nodes, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        rule.weights(limits), stored[2 * count : 3 * count], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        rule.weights(limits, coarse=True), stored[3 * count :], rtol=1e-12, atol=1e-30
    )


def test_the_mode_matrices_are_a_frozen_dataclass():
    modes = [MarcatiliMode(RADIUS, "He", 1.0)]
    matrices = mode_matrix(modes, transverse_quadrature(modes, n_r=4))
    assert isinstance(matrices, ModeMatrices)
    with pytest.raises(AttributeError):
        # setattr rather than an assignment, so the static checker does not reject
        # the very thing the test asserts is rejected at run time.
        setattr(matrices, "n_pol", 2)  # noqa: B010 - see above


def test_the_on_axis_rule_sits_at_the_origin_with_one_node():
    rule = on_axis_rule("polar")
    assert rule.n_nodes == 1
    assert float(rule.xi_r[0]) == 0.0
    assert float(rule.xi_theta[0]) == 0.0


def test_the_on_axis_rule_evaluates_a_mode_at_its_own_peak():
    """HE_11 peaks on the axis, so its normalised profile there is a known number."""
    mode = MarcatiliMode(125e-6, "He", 1.0)
    got = mode_matrix([mode], on_axis_rule("polar"), 0.0, components=("x", "y"))
    # e_hat(0) = J_0(0) / sqrt(N) with N = (pi/2) a^2 J_1(u_11)^2 sqrt(eps0/mu0), so
    # the synthesised on-axis field per sqrt(W) is 1/sqrt(N) exactly.
    expected = 1.0 / math.sqrt(mode.normalisation(0.0))
    np.testing.assert_allclose(
        float(np.max(np.abs(got.synthesis))), expected, rtol=1e-14
    )


def test_the_on_axis_rule_supports_rectangular_guides_and_refuses_unknown_kinds():
    rule = on_axis_rule("cartesian")
    np.testing.assert_array_equal(
        rule.nodes(("cartesian", (-2.0, -1.0), (2.0, 1.0))), [[0.0, 0.0]]
    )
    with pytest.raises(ModeError, match="unknown cross-section"):
        inspect.unwrap(on_axis_rule)("unknown")


def test_the_on_axis_synthesis_rescales_like_the_transform_matrices():
    """The law the backends' on-axis cache rests on.

    Built at ``z = 0`` and multiplied by ``taper_scale``, the axis row is the one built
    at ``z``: the axis is the node ``r = 0`` of a one-node rule, and a scale-invariant
    profile at ``r = a * xi`` goes as ``1/a`` with its normalisation as ``a^2``.
    """
    from solprop.modes.quadrature import (  # noqa: PLC0415 - the three this test is about
        mode_matrix,
        on_axis_rule,
        taper_scale,
    )

    modes = [MarcatiliMode(Linear(100e-6, 50e-6, 1.0), "He", 1.0, m=m) for m in (1, 2)]
    rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
    at_entrance = mode_matrix(modes, rule, 0.0, components=("x", "y")).synthesis
    for z in (0.25, 0.7, 1.0):
        built = mode_matrix(modes, rule, z, components=("x", "y")).synthesis
        np.testing.assert_allclose(
            at_entrance * taper_scale(modes[0], z), built, rtol=1e-12, atol=0.0
        )


@pytest.mark.parametrize("kronrod", [False, True])
def test_composite_radial_panels_are_exact_with_embedded_coarse_weights(kronrod):
    breaks = (0.21, 0.63)
    base = quadrature_rule(
        "polar", full=False, n_r=9, radial_rule="gauss_kronrod" if kronrod else "gauss"
    )
    rule = quadrature_rule(
        "polar",
        full=False,
        n_r=9,
        radial_rule="gauss_kronrod" if kronrod else "gauss",
        breaks=breaks,
    )
    assert rule.breaks == breaks
    assert rule.n_r == 3 * base.n_r
    for panel, (left, right) in enumerate(
        zip((0.0, *breaks), (*breaks, 1.0), strict=True)
    ):
        selection = slice(panel * base.n_r, (panel + 1) * base.n_r)
        nodes = rule.xi_r[selection]
        assert rule.w_r is not None and rule.w_r_coarse is not None
        assert base.w_r_coarse is not None
        # Affine mapping must retain exactly the embedded coarse-node selection
        # (its zeros are structural), and scale each weight by the panel's width to
        # the rounding of that product.
        coarse = rule.w_r_coarse[selection]
        np.testing.assert_array_equal(coarse == 0.0, base.w_r_coarse == 0.0)
        np.testing.assert_allclose(
            coarse, (right - left) * base.w_r_coarse, rtol=2 * EPS, atol=0
        )
        for degree in range(8):
            exact = (right ** (degree + 1) - left ** (degree + 1)) / (degree + 1)
            for weights in (rule.w_r, rule.w_r_coarse):
                assert weights is not None
                # Degree seven lies inside both four-node Gauss and fine exactness.
                np.testing.assert_allclose(
                    weights[selection] @ nodes**degree, exact, rtol=2e-14, atol=0
                )


# The twenty-four rules the composite-break work had to leave alone, recorded from
# the tree before that change. Comparing the branch with itself cannot fail; this
# compares it with the record.
FIXED_RULES = Path(__file__).parents[1] / "data/quadrature_rules_before_breaks.json"


def test_the_fixed_rules_are_what_they_were_before_radial_breaks():
    """The rules match their record to the rounding of their eigensolve.

    Gauss and Gauss-Kronrod nodes and weights come from a symmetric tridiagonal
    eigenproblem of order ``n_r``, backward stable to ``O(n_r u)`` of its norm, so each
    array is held normwise to ``4 n_r eps`` (a factor-of-four margin on ``n_r`` eps)
    of the record; another platform's LAPACK moves the last bits within that. The
    metadata and the coarse rules' zero weights are structural and exact.
    """
    record = json.loads(FIXED_RULES.read_text())["rules"]
    assert len(record) == 24
    for entry in record:
        rule = quadrature_rule(
            entry["kind"],
            full=True,
            n_r=entry["n_r"],
            n_theta=16,
            radial_rule="gauss_kronrod" if entry["kronrod"] else "gauss",
        )
        assert rule.breaks == ()
        assert [rule.n_r, rule.n_theta, rule.kronrod, rule.full] == entry["meta"]
        for name in (
            "xi_r",
            "w_r",
            "w_r_coarse",
            "xi_theta",
            "w_theta",
            "w_theta_coarse",
        ):
            actual = np.asarray(getattr(rule, name), dtype=float)
            expected = np.asarray(entry[name], dtype=float)
            label = (
                f"{entry['kind']} kronrod={entry['kronrod']} n_r={entry['n_r']} {name}"
            )
            if name.endswith("_coarse"):
                np.testing.assert_array_equal(
                    actual == 0.0, expected == 0.0, err_msg=label
                )
            # The eigensolve's O(n u) in each array's own length: the radial arrays
            # have n_r points, the angular ones n_theta (Linux LAPACK and Accelerate
            # differ by 7e-15 on the 16 angular weights of an 8-point radial rule).
            points = rule.n_theta if "theta" in name else rule.n_r
            assert_within(actual, expected, 4 * points * EPS, what=label)
    # An explicit empty break list is the same rule, which is what the driver
    # passes for every Cartesian set.
    explicit = quadrature_rule("polar", full=True, n_r=12, n_theta=8, breaks=())
    default = quadrature_rule("polar", full=True, n_r=12, n_theta=8)
    for name in ("xi_r", "w_r", "w_r_coarse", "xi_theta", "w_theta", "w_theta_coarse"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))


@pytest.mark.parametrize("breaks", [(0.0,), (1.0,), (np.nan,), (0.5, 0.4), (0.4, 0.4)])
def test_invalid_radial_breaks_are_refused(breaks):
    with pytest.raises(ModeError, match="radial breaks"):
        quadrature_rule("polar", full=False, n_r=8, breaks=breaks)


def test_cartesian_radial_breaks_are_refused():
    with pytest.raises(ModeError, match="radial breaks"):
        quadrature_rule("cartesian", full=True, n_r=8, breaks=(0.5,))
