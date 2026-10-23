# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CF4 varying flow's guards on synthetic generators.

Every generator here has an exact solution the tests compare against, written out
independently of :mod:`solprop.translate.coupled`: a rotating frame, whose ordered
exponential is ``R(w z) expm((D - w J) z)`` for ``A(z) = R(w z) D R(w z)^T``. Every
bound is stated in the docstring of the test that takes it, before the measurement.
"""

import inspect
import math

import numpy as np
import pytest
from scipy.linalg import expm

from solprop.exceptions import SolpropWarning, TranslationError
from solprop.translate.coupled import (
    CF4_GROWTH_LIMIT,
    CF4_NODES,
    CF4_WEIGHTS,
    cf4_defect,
    cf4_flow,
    cf4_growth_bound,
    constant_schur_flow,
    frozen_residual_flow,
    section_expansion,
)

ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])
"""The generator of a plane rotation, ``R(t) = expm(t J)``."""

RATE = 3.0
"""Rotation rate of the rotating frame, rad/m."""


def rotation(angle):
    """Return the plane rotation by `angle`."""
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )


def rotating(diagonal):
    """Return ``z -> R(3 z) diag R(3 z)^T``, the rotating generator."""

    def operator(z):
        basis = rotation(RATE * z)
        return basis @ np.diag(diagonal) @ basis.T

    return operator


def exact(diagonal, z0, z1):
    """Return the exact ordered exponential of `rotating(diagonal)` over ``[z0, z1]``.

    In the co-rotating frame ``b = R(-3z) a`` the generator is the constant
    ``diag - 3 J``, so ``a(z1) = R(3 z1) expm((diag - 3J)(z1 - z0)) R(-3 z0) a(z0)``.
    """
    return (
        rotation(RATE * z1)
        @ expm((np.diag(diagonal) - RATE * ROTATION) * (z1 - z0))
        @ rotation(-RATE * z0)
    )


def leaf_coefficients(operator, edges):
    """Return the Chebyshev coefficients of `operator` on `edges`, one bin wide."""
    coefficients, tail = section_expansion(
        lambda positions: np.stack([operator(z)[:, :, None] for z in positions]),
        edges,
    )
    return coefficients.astype(complex), tail


def cf4_step(operator, z, length):
    """Return the two CF4 factors of one step, independently of the package."""
    early = operator(z + CF4_NODES[0] * length)
    late = operator(z + CF4_NODES[1] * length)
    small, large = CF4_WEIGHTS
    return (
        expm(length * (large * early + small * late)),
        expm(length * (small * early + large * late)),
    )


# -------------------------------------------------------------- the scheme itself


def test_the_cf4_constants_are_the_gauss_nodes_and_the_matched_weights():
    """The nodes are Gauss-Legendre's; the weights solve the two matching conditions.

    ``a + b = 1/2`` matches the integral of the generator, ``b - a = 1/sqrt(3)`` the
    cubic Magnus commutator; ``a`` is negative, which is what lets an internal factor
    grow even when every generator is dissipative.
    """
    nodes, weights = np.polynomial.legendre.leggauss(2)
    assert pytest.approx(tuple(0.5 * (nodes + 1.0)), abs=1e-15) == CF4_NODES
    assert weights == pytest.approx([1.0, 1.0], abs=1e-15)
    small, large = CF4_WEIGHTS
    assert small + large == pytest.approx(0.5, abs=1e-16)
    assert large - small == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-15)
    assert small < 0.0 < large


def test_the_ordering_is_fourth_order_and_the_reversed_one_is_second():
    """The early-weighted factor acts first; the other order loses two orders.

    Against the exact solution of the lossless rotating frame -- a nonnormal ordered
    flow whose generator does not commute with itself along z -- over 16, 32 and 64
    steps of a metre. Predeclared gate: the implemented ordering's observed orders
    within 4 +- 0.3, the reversed one's within 2 +- 0.3. Measured 3.998 and 4.000,
    and 1.995 and 1.999.
    """
    operator = rotating([1j, -2j])
    reference = exact([1j, -2j], 0.0, 1.0)
    errors: dict[str, list[float]] = {"implemented": [], "reversed": []}
    for count in (16, 32, 64):
        for name, recorded in errors.items():
            flow = np.eye(2, dtype=complex)
            for index in range(count):
                first, second = cf4_step(operator, index / count, 1.0 / count)
                pair = second @ first if name == "implemented" else first @ second
                flow = pair @ flow
            recorded.append(float(np.linalg.norm(flow - reference, 2)))
    for name, expected in (("implemented", 4.0), ("reversed", 2.0)):
        orders = np.log2(np.array(errors[name][:-1]) / np.array(errors[name][1:]))
        assert np.all(abs(orders - expected) < 0.3), (name, orders)


# ------------------------------------------------------------------ internal growth

DISSIPATIVE = [0.0, -200.0]
"""The dissipative case: eigenvalues 0 and -200/m, rotating at 3 rad/m."""


def test_the_dissipative_case_reproduces_its_internal_amplification():
    """One CF4 action over a metre: factors of 1898, an action of 5e5, an exact 0.96.

    The recorded numbers, reproduced from the scheme alone: each internal factor has
    spectral norm about 1898, the CF4 action 502653, the exact action 0.9564, while
    every generator on the way is negative semidefinite. The growth bound of
    :func:`solprop.translate.coupled.cf4_flow`, evaluated on the exact Hermitian spread
    of 200/m, is ``exp((2 sqrt(3) - 3)/6 * 200) = 5.23e6`` over the largest the exact
    action can be (``exp(h m_hi) = 1``); the measured product of the two factors,
    3.60e6, lies under it -- the bound is tight to a factor 1.45 here -- and so does the
    action.
    """
    operator = rotating(DISSIPATIVE)
    first, second = cf4_step(operator, 0.0, 1.0)
    norms = [float(np.linalg.norm(factor, 2)) for factor in (first, second)]
    action = float(np.linalg.norm(second @ first, 2))
    truth = float(np.linalg.norm(exact(DISSIPATIVE, 0.0, 1.0), 2))
    assert norms == pytest.approx([1898.0, 1898.0], rel=1e-3)
    assert action == pytest.approx(502653.0, rel=1e-5)
    assert truth == pytest.approx(0.9564, rel=1e-4)
    growth = math.exp((2.0 * math.sqrt(3.0) - 3.0) / 6.0 * 200.0)
    assert norms[0] * norms[1] <= growth
    assert action <= growth
    assert growth / (norms[0] * norms[1]) < 1.5


@pytest.fixture(scope="module")
def dissipative_leaf():
    """The dissipative generator on 64 sections of a metre."""
    edges = np.linspace(0.0, 1.0, 65)
    coefficients, tail = leaf_coefficients(rotating(DISSIPATIVE), edges)
    return edges, coefficients, tail


def test_the_growth_guard_refuses_the_dissipative_case_by_name(dissipative_leaf):
    """The declared default limit refuses the dissipative case, section by section.

    The case is recorded with a metre-long step, so ``dz_max = 1`` m. The conditioning
    limit is lifted so that it is the **growth** guard that refuses; the refusal names
    the bound, the limit and the fallback that could be declared.
    """
    edges, coefficients, tail = dissipative_leaf
    with pytest.raises(TranslationError, match=r"reject 64 of its 64 section") as info:
        cf4_flow(
            np.array([1e15]),
            edges,
            coefficients,
            None,
            dz_max=1.0,
            condition_limit=math.inf,
            representation_error=tail,
            gauge_id="gauge-a",
            certificate_id="cf4",
        )
    assert "internal growth bound" in str(info.value)
    assert f"limit {CF4_GROWTH_LIMIT:.4g}" in str(info.value)
    assert "fallback='frozen_residual'" in str(info.value)


def test_the_enclosed_growth_bound_holds_and_says_how_loose_it_is(dissipative_leaf):
    """The bound the leaf declares is above every internal product it can meet.

    Admitted with the limits lifted, the leaf's growth bound is built from the
    **enclosed** Hermitian spread -- Weyl's inequality on the stored expansion, over
    the whole reach of a metre-long step -- rather than from the exact 200/m, so it is
    looser than 5.23e6 by the sections' own variation. Bound, stated first: it must
    exceed the measured product of the two factors of every one-metre step and every
    half-metre step the reach allows, and it must fall towards the exact spread's bound
    as the sections narrow. Measured 1.08e7 at 64 sections -- a factor 2.06 above the
    exact-spread bound and 3.0 above the worst product measured -- against 9.43e7 at
    16 sections and 6.27e6 at 256.
    """
    edges, coefficients, tail = dissipative_leaf
    flow = cf4_flow(
        np.array([1e15]),
        edges,
        coefficients,
        None,
        dz_max=1.0,
        condition_limit=math.inf,
        growth_limit=math.exp(60.0),
        representation_error=tail,
        gauge_id="gauge-a",
        certificate_id="cf4",
    )
    declared = float(np.max(cf4_growth_bound(flow)))
    operator = rotating(DISSIPATIVE)
    measured = 0.0
    for length, starts in ((1.0, [0.0]), (0.5, [0.0, 0.25, 0.5])):
        for start in starts:
            first, second = cf4_step(operator, start, length)
            measured = max(
                measured,
                float(np.linalg.norm(first, 2) * np.linalg.norm(second, 2)),
            )
    exact_spread = math.exp((2.0 * math.sqrt(3.0) - 3.0) / 6.0 * 200.0)
    assert measured <= exact_spread <= declared
    assert declared / exact_spread < 3.0
    coarse_edges = np.linspace(0.0, 1.0, 17)
    coarse, coarse_tail = leaf_coefficients(operator, coarse_edges)
    looser = cf4_flow(
        np.array([1e15]),
        coarse_edges,
        coarse,
        None,
        dz_max=1.0,
        condition_limit=math.inf,
        growth_limit=math.exp(60.0),
        representation_error=coarse_tail,
        gauge_id="gauge-a",
        certificate_id="cf4",
    )
    assert float(np.max(cf4_growth_bound(looser))) > declared


def test_a_declared_fallback_is_taken_only_when_the_frozen_block_admits_it(
    dissipative_leaf,
):
    """The fallback is recoverable only where the frozen split's own bound holds.

    The frozen block of this case has conditioning ``exp(200)`` over a metre, so
    falling back under the default conditioning limit is refused by name too; with
    that limit lifted, every section falls back, with a warning naming them.
    """
    edges, coefficients, tail = dissipative_leaf
    arguments = {
        "dz_max": 1.0,
        "representation_error": tail,
        "gauge_id": "gauge-a",
        "certificate_id": "cf4",
        "fallback": "frozen_residual",
    }
    with pytest.raises(TranslationError, match="cannot fall back either"):
        cf4_flow(
            np.array([1e15]),
            edges,
            coefficients,
            None,
            condition_limit=1e4,
            **arguments,
        )
    with pytest.warns(SolpropWarning, match="fall back to 'frozen_residual'"):
        flow = cf4_flow(
            np.array([1e15]),
            edges,
            coefficients,
            None,
            condition_limit=math.inf,
            **arguments,
        )
    assert flow.section_methods == ("frozen_residual",) * 64
    # A fallen-back section has no internal factors, so no growth to bound.
    assert np.all(cf4_growth_bound(flow) == 1.0)
    frozen = frozen_residual_flow(
        np.array([1e15]),
        edges,
        coefficients,
        None,
        dz_max=1.0,
        condition_limit=math.inf,
        representation_error=tail,
        gauge_id="gauge-a",
        certificate_id="cf4",
    )
    for name in ("generator_error", "residual_norm_bound", "inverse_condition_bound"):
        assert np.array_equal(getattr(flow, name), getattr(frozen, name))


# ------------------------------------------------------ the rows of a CF4 section


def mild_leaf():
    """A mildly dissipative rotating generator on two sections of a centimetre.

    Loss 20/m rotating at 3 rad/m over 1 cm: well inside the asymptotic regime, so
    both guards admit it at ``dz_max = 5e-3`` m.
    """
    edges = np.array([0.0, 5e-3, 1e-2])
    coefficients, tail = leaf_coefficients(rotating([0.0, -20.0]), edges)
    return edges, coefficients, tail


def test_a_cf4_section_declares_no_residual_and_its_pair_bound():
    """The three rows of an admitted CF4 section, each against what it claims.

    ``residual_norm_bound`` is zero, the whole generator being in the exponential;
    ``generator_error`` is the expansion's tail alone; ``inverse_condition_bound``
    bounds the product of the four factor norms of every forward and inverse pair a
    step anchored in the section can form. That last is checked against the factors
    themselves -- formed here independently, by SciPy, at 41 starts and three lengths
    up to ``dz_max`` -- and against the exact Hermitian spread, 20/m, which the
    enclosure must contain.
    """
    edges, coefficients, tail = mild_leaf()
    flow = cf4_flow(
        np.array([1e15]),
        edges,
        coefficients,
        None,
        dz_max=5e-3,
        condition_limit=10.0,
        representation_error=tail,
        gauge_id="mild",
        certificate_id="cf4",
    )
    assert flow.section_methods == ("cf4", "cf4")
    assert np.all(flow.residual_norm_bound == 0.0)
    assert np.all(flow.generator_error == tail)
    declared = float(np.max(flow.inverse_condition_bound))
    assert declared >= math.exp(2.0 / math.sqrt(3.0) * 5e-3 * 20.0)
    operator = rotating([0.0, -20.0])
    for length in (5e-3, 2.5e-3, 1e-3):
        for start in np.linspace(0.0, 1e-2 - length, 41):
            first, second = cf4_step(operator, float(start), length)
            product = np.prod(
                [
                    np.linalg.norm(factor, 2)
                    for factor in (first, second, np.linalg.inv(first))
                ]
            ) * np.linalg.norm(np.linalg.inv(second), 2)
            assert product <= declared
    power = (2.0 - math.sqrt(3.0)) / 4.0
    assert np.allclose(
        cf4_growth_bound(flow), flow.inverse_condition_bound**power, rtol=1e-15
    )


def test_a_constant_section_is_exact_under_cf4_and_its_bounds_are_the_frozen_ones():
    """A generator that does not vary is integrated exactly, and spread is spread.

    With ``A_1 = A_2 = G`` the two factors are ``exp(h G/2)`` twice, so a CF4 step is
    the exact action to rounding, and :func:`cf4_defect` estimates a defect at the
    rounding floor. The enclosure of a constant section is its own spectrum, so the
    growth bound is ``exp(2|a| dz (m_hi - m_lo))`` exactly.
    """
    block = np.zeros((1, 1, 2, 2, 1), dtype=complex)
    block[0, 0, 0, 0] = -1.0
    block[0, 0, 1, 1] = -101.0
    flow = cf4_flow(
        np.array([1e15]),
        np.array([0.0, 1e-3]),
        block,
        None,
        dz_max=1e-3,
        condition_limit=10.0,
        representation_error=0.0,
        gauge_id="constant",
        certificate_id="cf4",
    )
    assert float(cf4_growth_bound(flow)[0, 0]) == pytest.approx(
        math.exp((2.0 * math.sqrt(3.0) - 3.0) / 6.0 * 1e-3 * 100.0), rel=1e-14
    )
    assert float(np.max(cf4_defect(flow, 1e-3))) < 1e-15


def test_a_step_reaching_the_next_section_is_bounded_by_its_enclosure_too():
    """A step anchored near a section's end reaches into the next one; not backwards.

    Two constant sections, the first the lossier. A step anchored in the first can
    end in the second, so the first's bound is the union's -- here its own spread,
    100/m, the larger -- while a step anchored in the second never reaches back, so
    the second's is its own 10/m however long ``dz_max`` is.
    """
    block = np.zeros((2, 1, 2, 2, 1), dtype=complex)
    block[0, 0, 1, 1] = -100.0
    block[1, 0, 1, 1] = -10.0
    flow = cf4_flow(
        np.array([1e15]),
        np.array([0.0, 1e-3, 2e-3]),
        block,
        None,
        dz_max=5e-4,
        condition_limit=10.0,
        representation_error=0.0,
        gauge_id="two",
        certificate_id="cf4",
    )
    exponent = 2.0 / math.sqrt(3.0) * 5e-4
    assert flow.inverse_condition_bound[:, 0] == pytest.approx(
        [math.exp(exponent * 100.0), math.exp(exponent * 10.0)], rel=1e-14
    )
    block[0, 0, 1, 1], block[1, 0, 1, 1] = -10.0, -100.0
    reversed_flow = cf4_flow(
        np.array([1e15]),
        np.array([0.0, 1e-3, 2e-3]),
        block,
        None,
        dz_max=5e-4,
        condition_limit=10.0,
        representation_error=0.0,
        gauge_id="two",
        certificate_id="cf4",
    )
    assert reversed_flow.inverse_condition_bound[:, 0] == pytest.approx(
        [math.exp(exponent * 100.0), math.exp(exponent * 100.0)], rel=1e-14
    )


def test_the_builder_checks_its_own_arguments():
    """A growth limit below one, a non-finite one and an unknown fallback are named."""
    edges, coefficients, tail = mild_leaf()
    base = {
        "dz_max": 5e-3,
        "condition_limit": 10.0,
        "representation_error": tail,
        "gauge_id": "mild",
        "certificate_id": "cf4",
    }
    for bad in (0.5, math.nan, math.inf):
        with pytest.raises(TranslationError, match="growth_limit"):
            cf4_flow(
                np.array([1e15]), edges, coefficients, None, growth_limit=bad, **base
            )
    # A string the annotation would stop in the test session's runtime check, so the
    # builder's own door is reached around it, as a production caller reaches it.
    unchecked = inspect.unwrap(cf4_flow)
    with pytest.raises(TranslationError, match="fallback"):
        unchecked(
            np.array([1e15]), edges, coefficients, None, fallback="midpoint", **base
        )


def test_the_defect_estimator_refuses_what_it_cannot_estimate():
    """A factored flow has no dense generator to step, and a step must be positive."""
    edges, coefficients, tail = mild_leaf()
    flow = cf4_flow(
        np.array([1e15]),
        edges,
        coefficients,
        None,
        dz_max=5e-3,
        condition_limit=10.0,
        representation_error=tail,
        gauge_id="mild",
        certificate_id="cf4",
    )
    with pytest.raises(TranslationError, match="positive step"):
        cf4_defect(flow, 0.0)
    block = np.zeros((2, 2, 1), dtype=complex)
    constant = constant_schur_flow(
        np.array([1e15]),
        np.array([0.0, 1e-3]),
        block,
        dz_max=1e-4,
        condition_limit=10.0,
        gauge_id="lab",
        certificate_id="schur",
    )
    factored = constant.__class__(
        constant.omega,
        constant.z_edges,
        None,
        np.zeros((1, 2, 2, 1), dtype=complex),
        np.zeros((1, 1, 1), dtype=complex),
        None,
        None,
        None,
        constant.generator_error,
        constant.residual_norm_bound,
        constant.inverse_condition_bound,
        ("cf4",),
        "lab",
        "cf4",
    )
    with pytest.raises(TranslationError, match="factored"):
        cf4_defect(factored, 1e-4)
    assert np.all(cf4_defect(constant, 1e-4) == 0.0)


def test_the_step_doubling_estimate_tracks_the_true_local_error():
    """The step-doubling estimator against the exact rotating-frame solution.

    ``(16/15) ||V(h) - V(h/2)^2||`` estimates the full step's local defect
    asymptotically. Predeclared: within 5 % of the true error of one CF4 step, from
    the start of the section, at 2.5, 1.25 and 0.625 mm on the mildly dissipative
    case, and falling at the step's fifth power (observed order within 5 +- 0.3).
    Measured ratios 1.000002, 1.0001 and 1.002, and orders 4.98 and 4.99.
    """
    edges, coefficients, tail = mild_leaf()
    flow = cf4_flow(
        np.array([1e15]),
        edges,
        coefficients,
        None,
        dz_max=5e-3,
        condition_limit=10.0,
        representation_error=tail,
        gauge_id="mild",
        certificate_id="cf4",
    )
    operator = rotating([0.0, -20.0])
    truths, estimates = [], []
    for length in (2.5e-3, 1.25e-3, 6.25e-4):
        first, second = cf4_step(operator, 0.0, length)
        reference = exact([0.0, -20.0], 0.0, length)
        truths.append(
            float(np.linalg.norm(second @ first - reference, 2))
            / float(np.linalg.norm(reference, 2))
        )
        # The estimator samples the section's start, middle and end and keeps the
        # worst. A rotating generator is the same one up to a rotation at every
        # start, and a rotation leaves both norms alone, so every anchor's truth is
        # the start's.
        estimates.append(float(cf4_defect(flow, length)[0, 0]))
    ratios = np.array(estimates) / np.array(truths)
    assert np.all(abs(ratios - 1.0) < 0.05), ratios
    orders = np.log2(np.array(truths[:-1]) / np.array(truths[1:]))
    assert np.all(abs(orders - 5.0) < 0.3), orders
