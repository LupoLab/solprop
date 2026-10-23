# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare variational radial modes with independently layered and analytic spectra."""

import importlib
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from witness_materials import absorbing_silica

from modal_oracles import register_modal_oracles
from modal_provider_cases import OMEGA as STEP_OMEGA
from modal_provider_cases import make_provider
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes.graded_index import (
    MAX_EXTERIOR_TAIL_FRACTION,
    GradedIndexProvider,
    PowerLawPermittivity,
)
from solprop.modes.guide import radial_node_rule
from solprop.modes.material import IndexMaterial
from solprop.modes.vector import ModalNodeRule, ModalProvider

OMEGA = 2 * np.pi * C / 1.03e-6


def provider(family: Literal["TE", "TM", "LP"] = "TE", elements=32, exponent=2.0):
    return GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005, exponent),
        exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
        radius=25e-6,
        family=family,
        count=2,
        elements=elements,
        extent=2.0,
    )


@pytest.mark.parametrize("family", ["TE", "TM", "LP"])
def test_radial_modes_have_unit_power_and_exact_exterior(family):
    p = provider(family)
    nodes = p.node_rule(OMEGA)
    sample = p.solve(OMEGA, 0.0, nodes=nodes)
    # FE eigenvectors are normalised separately from the supplied quadrature, so
    # the sampled Gram is an independent measurement of that quadrature; 1e-9 is
    # the declared physical Gram engineering target, fixed before this was
    # measured. Measured largest component defect at 32 elements: 4.2e-14 (TE),
    # 1.5e-14 (TM), 3.6e-14 (LP) -- four orders of margin. The tail bound is
    # 4.1e-22 against the declared 1e-12, and the P1 interface residual is 0.019
    # (TE), 0.009 (TM), 0.004 (LP) against the first-order 4/elements = 0.125.
    assert_allclose(sample.flux_metric, np.eye(2), rtol=0, atol=1e-9)
    assert np.all(p.tail_bound(OMEGA, outer_radius=50e-6) < 1e-12)
    assert np.all(sample.boundary_residual < 4 / p.elements)
    assert len(p.regions) == 2
    assert p.identity["approximation"] == (
        "weak-guidance-LP" if family == "LP" else "vector-sector"
    )


@pytest.mark.slow
@pytest.mark.parametrize("family", ["TE", "TM"])
@pytest.mark.parametrize("exponent", [2.0, 4.0])
def test_radial_layer_and_domain_refinement(family, exponent):
    # Fixed before observing: second-order P1 beta convergence, engineering
    # agreement of 1e-6 at 96 core elements, and 1e-9 for independent domain growth.
    relative_target = 1e-6
    register_modal_oracles()
    reference = importlib.import_module("modal_design.graded_index")
    law = reference.Grin(quartic=0.0 if exponent == 2 else 1.0)
    values = []
    for elements in (24, 48, 96):
        p = provider(family, elements, exponent)
        values.append(p._solution(OMEGA, 0.0).beta[0] * 25e-6)
    order = np.log2(abs((values[0] - values[1]) / (values[1] - values[2])))
    assert 1.5 <= order <= 5
    # Layer counts change while the variational reference remains fixed.
    layers = [reference.layer_root(law, values[-1], n, family) for n in (48, 96, 192)]
    layer_order = np.log2(abs((layers[0] - layers[1]) / (layers[1] - layers[2])))
    assert 1.5 <= layer_order <= 5
    assert abs(values[-1] / layers[-1] - 1) < relative_target
    domains = [
        replace(provider(family, 48, exponent), extent=x)._solution(OMEGA, 0.0).beta[0]
        for x in (1.5, 2.0, 2.5)
    ]
    assert_allclose(domains, domains[-1], rtol=1e-9, atol=0)
    # Profile phase and shape are independent of a comparison of eigenvalues.
    p = provider(family, 96, exponent)
    solution = p._solution(OMEGA, 0.0)
    x = np.linspace(0.01, 1.5, 200)
    exact = reference.layer_profile(law, layers[-1], 192, family, x).real
    actual = np.interp(x, solution.nodes, solution.profiles[:, 0])
    actual /= np.linalg.norm(actual)
    exact /= np.linalg.norm(exact)
    if np.vdot(actual, exact) < 0:
        exact = -exact
    # P1 L2 interpolation O(h²); 10*h² allows radial curvature and normalisation.
    assert np.linalg.norm(actual - exact) < 10 / 96**2


def _parabolic_groups(groups):
    """Return the ideal parabolic group constants for group numbers `groups`, 1/m."""
    q = OMEGA / C * 25e-6 * 1.45
    return np.sqrt(q * q - 2 * q * np.sqrt(0.01) * np.asarray(groups)) / 25e-6


@pytest.mark.slow
def test_parabolic_mode_groups_and_lp_vector_limit():
    """Each sector against the closed form of **its own** group number.

    ``beta_g = sqrt(q^2 - 2 q sqrt(2 Delta) g)/a`` with ``g = 2m + l + 1``, so the
    two ``l = 0`` LP channels are groups 1 and 3 and the two ``l = 1`` channels --
    which is what TE and TM are, since a TE_0m field satisfies the scalar radial
    equation at azimuthal harmonic **one** -- are groups 2 and 4. Comparing a
    vector sector against ``g = (1, 3)`` measures the l = 0 to l = 1 group
    separation, 4.5e-4 here, and not any vector correction; against its own groups
    every sector agrees to the same discretisation order as the LP one.
    """
    p = provider("LP", 96)
    actual = p._solution(OMEGA, 0.0).beta
    # Finite-core analytic ground states decay well before rho=1; P1 beta
    # discretisation at h=1/96 dominates this 1e-6 relative engineering budget.
    assert_allclose(actual, _parabolic_groups([1, 3]), rtol=1e-6, atol=0)
    lp = replace(p, azimuthal_order=1)._solution(OMEGA, 0.0).beta
    te = provider("TE", 96)._solution(OMEGA, 0.0).beta
    tm = provider("TM", 96)._solution(OMEGA, 0.0).beta
    # The same P1 budget the l = 0 sector is held to, against the l = 1 groups:
    # measured 1.36e-7 and 6.80e-7 for TE and the same for TM at 96 elements.
    assert_allclose(te, _parabolic_groups([2, 4]), rtol=1e-6, atol=0)
    assert_allclose(tm, _parabolic_groups([2, 4]), rtol=1e-6, atol=0)
    # The TE sector **is** the scalar l = 1 sector: same operator, same assembly,
    # so the two eigenvalues differ by at most a reordering of the solve's
    # roundings; 1e-15 (4.5 eps) holds that to the eigenvalues' own rounding.
    assert_allclose(te, lp, rtol=1e-15, atol=0)
    # What the genuine vector correction is worth here: TM against TE, 4.0e-10 and
    # 9.3e-10 at 96 elements -- six orders below the 4.5e-4 that comparing against
    # the l = 0 groups produces, which is the group separation and not physics.
    assert_allclose(tm, te, rtol=1e-8, atol=0)
    assert np.max(np.abs(te / _parabolic_groups([1, 3]) - 1.0)) > 4e-4


def test_radial_refusals():
    p = provider()
    with pytest.raises(ModeError, match="hybrid GRIN"):
        replace(p, family="HE")
    with pytest.raises(ModeError, match="order zero"):
        replace(p, azimuthal_order=1)
    with pytest.raises(ModeError, match="elements"):
        replace(p, elements=2)
    with pytest.raises(ModeError, match="weak-guidance"):
        replace(
            provider("LP"), permittivity=PowerLawPermittivity(IndexMaterial(1.6), 0.1)
        ).solve(OMEGA, 0.0, nodes=p.node_rule(OMEGA))
    with pytest.raises(ModeError, match="matching radius"):
        p.tail_bound(OMEGA, outer_radius=25e-6)


def test_a_channel_too_near_cutoff_for_the_domain_is_refused_by_name():
    # Five TE channels at 48 elements. At a 25.00 um core the fifth keeps
    # 5.97e-8 of its power beyond the matching radius and is bound on this
    # domain; at 22.98 um that has risen to 3.24e-4, above the declared
    # MAX_EXTERIOR_TAIL_FRACTION of 1e-4, and the exact Bessel continuation --
    # not the discretised core -- would be carrying the mode.
    bound = replace(provider("TE", 48), count=5)
    outside = bound.tail_bound(OMEGA, outer_radius=2 * 25e-6)
    assert np.max(outside) < MAX_EXTERIOR_TAIL_FRACTION
    assert_allclose(np.max(outside), 5.967e-8, rtol=1e-3, atol=0)
    with pytest.raises(ModeError, match="TE05 carries"):
        replace(bound, radius=22.98e-6)._solution(OMEGA, 0.0)
    # The weakest reference case of the graded-index study keeps a factor 15 of
    # margin: the second LP channel at contrast 0.001, at 6.57e-6.
    weak = replace(
        provider("LP", 96),
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.001),
        exterior=IndexMaterial(1.45 * np.sqrt(1 - 0.002)),
        azimuthal_order=1,
    )
    assert_allclose(
        np.max(weak.tail_bound(OMEGA, outer_radius=2 * 25e-6)),
        6.567e-6,
        rtol=1e-3,
        atol=0,
    )


def test_the_weak_guidance_guard_reads_the_axis_not_a_panel():
    # Hoisted out of _eps, which the eigenvalue iteration calls per element about
    # 1e4 times per solve, and which took the maximum over whichever panel it was
    # handed. The declared quantity is the axis contrast against the exterior.
    law = PowerLawPermittivity(IndexMaterial(1.45), 0.005)
    p = replace(provider("LP"), permittivity=law)
    assert p._weak_guidance_contrast(OMEGA) == pytest.approx(
        1.45**2 / (1.45**2 * 0.99) - 1, rel=64 * np.finfo(float).eps
    )
    assert p._weak_guidance_contrast(OMEGA) <= 0.02
    # The guard fires before any matrix is assembled, so _solution refuses too.
    with pytest.raises(ModeError, match="weak-guidance"):
        replace(
            p, permittivity=PowerLawPermittivity(IndexMaterial(1.6), 0.1)
        )._solution(OMEGA, 0.0)
    # A TE sector is a vector sector and is not held to the LP contrast.
    assert replace(p, family="TE")._weak_guidance_contrast(OMEGA) > 0


def test_radial_parameter_and_material_refusals():
    p = provider()
    # The path is named as every other ParameterisedMode names it, and a name
    # left out keeps its path.
    assert p.parameter_paths["core_radius"](np.array([0.0]))[0] == 25e-6
    assert p.at_parameters({"core_radius": 30e-6}).radius == 30e-6
    assert p.at_parameters({}).radius == p.radius
    assert p.z_domain == (0.0, float("inf"))
    assert isinstance(p.z_breakpoints, tuple)
    with pytest.raises(ModeError, match="unknown radial provider parameters"):
        p.at_parameters({"diameter": 25e-6})
    with pytest.raises(ModeError, match="radius path"):
        p.at_parameters({"core_radius": -1.0})
    with pytest.raises(ModeError, match="contrast"):
        PowerLawPermittivity(IndexMaterial(1.45), 0.0)
    with pytest.raises(ModeError, match="radial_fraction"):
        p.permittivity.permittivity(OMEGA, np.array([1.1]))
    with pytest.raises(ModeError, match="frequency"):
        p._solution(0.0, 0.0)
    with pytest.raises(ModeError, match="guided interval"):
        replace(p, exterior=IndexMaterial(1.5))._solution(OMEGA, 0.0)
    with pytest.raises(ModeError, match="cutoff"):
        replace(p, radius=1e-9)._solution(OMEGA, 0.0)
    with pytest.raises(ModeError, match="lossless"):
        # A complex silica of the caller's: optmat's is held transparent (SEP 3, Q3).
        replace(p, exterior=IndexMaterial(absorbing_silica()))._solution(OMEGA, 0.0)
    nodes = p.node_rule(OMEGA)
    with pytest.raises(ModeError, match="ownership"):
        p.solve(OMEGA, 0.0, nodes=replace(nodes, region_index=1 - nodes.region_index))


def test_lp_nonzero_angular_order_and_exterior_continuation():
    p = replace(provider("LP"), azimuthal_order=1, elements=16)
    nodes = radial_node_rule((25e-6, 50e-6), 60e-6, n_r=32, n_theta=17)
    regions = (np.hypot(nodes.xy[:, 0], nodes.xy[:, 1]) > 25e-6).astype(np.int32)
    sample = p.solve(OMEGA, 0.0, nodes=replace(nodes, region_index=regions))
    assert np.all(np.isfinite(sample.electric))
    assert np.all(p.tail_bound(OMEGA, outer_radius=60e-6) > 0)


@pytest.mark.slow
@pytest.mark.parametrize("contrast", [0.001, 0.005, 0.0098])
def test_lp_validity_at_declared_contrast_samples(contrast):
    law = PowerLawPermittivity(IndexMaterial(1.45), contrast)
    exterior = IndexMaterial(1.45 * np.sqrt(1 - 2 * contrast))
    p = replace(
        provider("LP", 96), permittivity=law, exterior=exterior, azimuthal_order=1
    )
    lp = p._solution(OMEGA, 0.0).beta
    for family in ("TE", "TM"):
        vector = replace(p, family=family, azimuthal_order=0)._solution(OMEGA, 0.0).beta
        # Same predeclared 0.5% beta budget at three contrasts up to the guard.
        assert_allclose(lp, vector, rtol=0.005, atol=0)


@pytest.mark.parametrize("family", ["TE", "TM"])
def test_longitudinal_axis_limit_matches_near_axis(family):
    p = replace(provider(family), count=1)
    xy = np.array([[0.0, 0.0], [25e-6 * 1e-10, 0.0], [25e-6 * 0.1, 0.0]])
    nodes = ModalNodeRule(xy, np.ones(3) * 1e-12, np.zeros(3, np.int32), {})
    sample = p.solve(OMEGA, 0.0, nodes=nodes)
    component = sample.magnetic if family == "TE" else sample.electric
    # The first P1 element is linear; f/r tends to its slope. At rho=1e-10
    # the TM permittivity differs only quadratically; 64 epsilon covers arithmetic.
    assert_allclose(
        component[0, 2], component[1, 2], rtol=64 * np.finfo(float).eps, atol=0
    )
    assert abs(component[0, 2, 0]) > 0


def test_every_provider_builds_its_rule_through_one_signature():
    radial = provider()
    step = make_provider()
    for one in (radial, step):
        assert isinstance(one, ModalProvider)
    # The same call reaches both: frequency, then position, then the two counts.
    requested = ((radial, OMEGA), (step, STEP_OMEGA))
    rules = [one.node_rule(omega, 0.0, n_r=4, n_theta=9) for one, omega in requested]
    for rule, (_, omega) in zip(rules, requested, strict=True):
        # Both certify the power their outer radius discards, at the frequency and
        # the position they were asked for, and both say which those were.
        bound = np.asarray(rule.identity["tail_power_bound"], dtype=float)
        assert rule.identity["omega"] == pytest.approx(omega)
        # The certificate names its own frequency: a tail bound holds where it was
        # taken and nowhere else, so from_provider takes it at the lowest knot.
        assert rule.identity["tail_power_bound_omega"] == pytest.approx(omega)
        assert rule.identity["z"] == 0.0
        assert bound.size >= 1
        assert np.all((bound >= 0) & (bound < MAX_EXTERIOR_TAIL_FRACTION))
    # The radial rule's certificate is the provider's own exact tail integral.
    assert_allclose(
        np.asarray(rules[0].identity["tail_power_bound"], dtype=float),
        radial.tail_bound(OMEGA, outer_radius=2.0 * 25e-6),
        rtol=0,
        atol=0,
    )


def test_the_radial_rule_chooses_the_angular_count_its_order_needs():
    # The default stays the nine the TE/TM sectors have always used -- their one
    # angular harmonic needs five -- and rises where a higher LP order needs more:
    # 4(h + 1) + 1 is exact for the quartic products of an order-h channel.
    scalar = provider().node_rule(OMEGA)
    dipole = replace(provider("LP"), azimuthal_order=1).node_rule(OMEGA)
    quadrupole = replace(provider("LP"), azimuthal_order=3).node_rule(OMEGA)
    assert (scalar.identity["n_theta"], dipole.identity["n_theta"]) == (9, 9)
    assert quadrupole.identity["n_theta"] == 17


def _by_radius(rule, sample):
    """Return the rule's distinct radii and the x field on each, per channel."""
    radii = np.round(np.hypot(rule.xy[:, 0], rule.xy[:, 1]), 15)
    distinct, first = np.unique(radii, return_index=True)
    field = sample.electric[:, 0, :]
    # Every node at one radius carries the same value, the profile having no angle;
    # the node's radius is rebuilt from its (x, y), which costs a few ulp of the
    # profile's scale (more, relatively, near a radial node of the profile).
    scale = 16 * np.finfo(float).eps * float(np.max(np.abs(field)))
    for radius, index in zip(distinct, first, strict=True):
        group = field[radii == radius]
        assert_allclose(
            group, np.broadcast_to(field[index], group.shape), rtol=0, atol=scale
        )
    return distinct, field[first]


def test_an_lp_sector_of_order_zero_is_exact_on_two_angular_nodes():
    """Two angular nodes give the nine-node sector to rounding.

    An ``LP_0m`` field is ``psi(r)`` in one fixed polarisation, so every product
    the rule integrates -- the Poynting normalisation, the flux metric, the quartic
    Kerr overlap -- is independent of the angle and an equally spaced angular rule
    of any size integrates it exactly. The default is therefore two nodes.
    """
    sector = provider("LP")
    default = sector.node_rule(OMEGA)
    nine = sector.node_rule(OMEGA, n_theta=9)
    assert (default.identity["n_theta"], nine.identity["n_theta"]) == (2, 9)
    assert default.xy.shape[0] * 9 == nine.xy.shape[0] * 2
    few = sector.solve(OMEGA, 0.0, nodes=default)
    many = sector.solve(OMEGA, 0.0, nodes=nine)
    # The two rules sum the same radial integrand over 2 and 9 equal angular
    # weights, so they agree to a few ulp of the largest term, not bit for bit.
    rounding = 64 * np.finfo(float).eps
    assert_allclose(few.generator, many.generator, rtol=rounding, atol=0)
    scale = float(np.max(np.abs(many.flux_metric)))
    assert_allclose(few.flux_metric, many.flux_metric, rtol=0, atol=rounding * scale)
    radii_few, profile_few = _by_radius(default, few)
    radii_many, profile_many = _by_radius(nine, many)
    # The radii are rebuilt from each rule's (x, y) and rounded to 1e-15 m, so a
    # rounding difference can move one by that quantum at most.
    assert_allclose(radii_few, radii_many, rtol=0, atol=2e-15)
    peak = float(np.max(np.abs(profile_many)))
    assert_allclose(profile_few, profile_many, rtol=0, atol=rounding * peak)

    def kerr_overlap(rule, sample):
        e = sample.electric[:, 0, :]
        return np.einsum("q,qi,qj,qk,ql->ijkl", rule.weights, e, e.conj(), e, e.conj())

    overlap = kerr_overlap(nine, many)
    assert_allclose(
        kerr_overlap(default, few),
        overlap,
        rtol=0,
        atol=rounding * float(np.max(np.abs(overlap))),
    )
