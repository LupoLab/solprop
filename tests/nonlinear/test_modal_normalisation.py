# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent vector Kerr coefficients, with physical coefficients inside the core."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modal_oracles import register_modal_oracles
from modal_provider_cases import OMEGA, make_provider
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes import StepIndexMode
from solprop.modes.guide import ModalGuide, StepIndexProvider
from solprop.modes.material import IndexMaterial
from solprop.nonlinear.normalisation import modal_kerr_coefficient

CORE_INDEX = 1.45
"""The synthetic strand's core index; not a measured silica claim."""


def core_chi3(nodes, n2=2.6e-20):
    """Return the strand's third-order susceptibility, zero outside the core."""
    return 4 * CORE_INDEX**2 * EPS0 * C * n2 / 3 * (nodes.region_index == 0)


def gamma(sample, nodes, n2=2.6e-20, channel=0):
    """Return the production coefficient of one channel on this rule."""
    return modal_kerr_coefficient(
        sample.electric[:, :, channel], nodes.weights, core_chi3(nodes, n2), OMEGA
    )


def test_the_coefficient_is_the_projected_polarisation():
    # An independent path to the same number: build the physical Kerr polarisation
    # of the analytic signal at unit amplitude, then project it with the bound
    # reciprocal functional -i omega W^-1 int(E* . P) dA / 4. The production
    # expression contracts the two into one scalar invariant instead, so the two
    # agree only if that contraction is right. Predeclared bound: 1e-12 relative,
    # the rounding of two differently grouped sums over the same ~3e3-node
    # quadrature (each term positive, so no cancellation beyond the grouping).
    bound = 1e-12
    provider = make_provider(radius=250e-9)
    nodes = provider.node_rule(OMEGA, n_r=48)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    electric = sample.electric[:, :, 0]
    chi3 = core_chi3(nodes)
    polarisation = (
        EPS0
        * chi3[:, None]
        / 4
        * (
            2 * np.sum(abs(electric) ** 2, axis=1)[:, None] * electric
            + np.sum(electric**2, axis=1)[:, None] * electric.conj()
        )
    )
    flux = float(np.real(sample.flux_metric[0, 0]))
    projected = (
        -1j
        * OMEGA
        / 4
        * np.sum(nodes.weights[:, None] * electric.conj() * polarisation)
        / flux
    )
    # The functional is purely imaginary for a lossless instantaneous response:
    # a real gamma, no gain.
    assert abs(projected.real) < bound * abs(projected)
    assert_allclose(-projected.imag, gamma(sample, nodes), rtol=bound, atol=0)


@pytest.mark.slow
def test_longitudinal_kerr():
    # The rounded stored witness has nine significant digits; 2e-9 relative allows
    # its rounding. Independent radial convergence is required separately below.
    witness_bound = 2e-9
    quadrature_bound = 1e-9
    provider = make_provider(radius=250e-9)
    values = []
    for order in (48, 96, 192):
        rule = provider.node_rule(OMEGA, n_r=order)
        sample = provider.solve(OMEGA, 0.0, nodes=rule)
        values.append(gamma(sample, rule))
    assert_allclose(values[-1], 0.541771892, rtol=witness_bound, atol=0)
    assert_allclose(values, values[-1], rtol=quadrature_bound, atol=0)
    register_modal_oracles()
    reference = importlib.import_module("modal_design.vector_modes")
    result = reference.vector_case(
        1.45, 1.0, 250e-9, 835e-9, radial_order=192, descriptors=[("HE", 1, 1)]
    )
    assert_allclose(
        values[-1], result["rows"][0]["gamma"], rtol=quadrature_bound, atol=0
    )
    transverse = sample.electric.copy()
    transverse[:, 2] = 0

    assert gamma(replace(sample, electric=transverse), rule) < values[-1]


def test_weak_guidance_limit():
    # The 0.5% engineering weak-guidance coefficient target, declared in advance.
    bound = 0.005
    mode = StepIndexMode(25e-6, core=1.45, cladding=1.449, profile_wavelength=835e-9)
    provider = ModalGuide.from_step_index(
        [mode], core=IndexMaterial(1.45), cladding=IndexMaterial(1.449)
    ).field_provider
    assert isinstance(provider, StepIndexProvider)
    rule = provider.node_rule(OMEGA)
    sample = provider.solve(OMEGA, 0.0, nodes=rule)
    scalar = OMEGA / C * 2.6e-20 / mode.aeff()
    assert abs(gamma(sample, rule) / scalar - 1) < bound


def test_material_coefficient_is_applied_once_inside_core():
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=24)
    sample = p.solve(OMEGA, 0.0, nodes=nodes)
    first = gamma(sample, nodes, n2=2.6e-20)
    # The reciprocal functional is linear in the material coefficient. The
    # 32-epsilon bound permits scalar multiplication around the quadrature sum.
    assert_allclose(
        gamma(sample, nodes, n2=5.2e-20),
        2 * first,
        rtol=32 * np.finfo(float).eps,
        atol=0,
    )
    assert gamma(sample, nodes, n2=0.0) == 0.0
    assert (
        gamma(sample, replace(nodes, region_index=np.ones(len(nodes.xy), np.int32)))
        == 0.0
    )
    # One susceptibility for every node is the same as a uniform array.
    uniform = float(np.max(core_chi3(nodes)))
    assert modal_kerr_coefficient(
        sample.electric[:, :, 0], nodes.weights, uniform, OMEGA
    ) == modal_kerr_coefficient(
        sample.electric[:, :, 0],
        nodes.weights,
        np.full(len(nodes.weights), uniform),
        OMEGA,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"electric": np.zeros((5, 2), complex)}, "three-component field"),
        ({"weights": np.ones(4)}, "three-component field"),
        ({"omega": 0.0}, "must be positive"),
        ({"omega": float("nan")}, "must be positive"),
        ({"chi3": float("inf")}, "finite"),
        ({"electric": np.full((5, 3), np.nan, complex)}, "finite"),
    ],
)
def test_the_coefficient_refuses_what_it_cannot_integrate(change, message):
    arguments = {
        "electric": np.ones((5, 3), complex),
        "weights": np.ones(5),
        "chi3": 1e-22,
        "omega": OMEGA,
    }
    with pytest.raises(ModeError, match=message):
        modal_kerr_coefficient(**{**arguments, **change})


def test_the_sampled_flux_divides_the_coefficient_exactly():
    """`metric=W` is one division and nothing else.

    The route divides by the sampled `W_00` through the `W^-1` inside its
    projection map, so the coefficient has to divide by the same number in the
    same place, to one rounding of a single divide.
    """
    arguments = _METRIC_CASE
    unit = modal_kerr_coefficient(**arguments)
    assert unit > 0.0
    assert modal_kerr_coefficient(**arguments, metric=None) == unit
    assert modal_kerr_coefficient(**arguments, metric=1.0) == unit
    assert modal_kerr_coefficient(**arguments, metric=0.25) == unit / 0.25


_METRIC_CASE = {
    "electric": np.ones((5, 3), complex),
    "weights": np.ones(5),
    "chi3": 1e-22,
    "omega": OMEGA,
}
"""One channel on five unit weights: the arithmetic case for the metric argument."""


def test_a_dense_metric_is_the_inverse_entry_and_not_the_reciprocal_diagonal():
    """`[W^-1]_mm` is `1/W_mm` only when W is diagonal, and the route uses the first.

    The metric argument is the one the tensor contraction shares with the route, which
    projects with `W^-1`. A diagonal metric reproduces the scalar statement exactly; an
    off-diagonal one does not, and the difference is what the coefficient would have
    been wrong by.
    """
    diagonal = np.diag([0.25, 4.0])
    unit = modal_kerr_coefficient(**_METRIC_CASE)
    for row in (0, 1):
        assert modal_kerr_coefficient(
            **_METRIC_CASE, metric=diagonal, channel=row
        ) == pytest.approx(unit / diagonal[row, row], rel=1e-15)
    # Off the diagonal the two part company: [W^-1]_00 of [[1, c], [c, 1]] is
    # 1/(1 - c^2), not 1.
    coupling = 0.3
    dense = np.array([[1.0, coupling], [coupling, 1.0]])
    expected = unit / (1.0 - coupling**2)
    assert modal_kerr_coefficient(
        **_METRIC_CASE, metric=dense, channel=0
    ) == pytest.approx(expected, rel=1e-14)
    assert abs(expected / unit - 1.0) == pytest.approx(0.098901, rel=1e-4)
    # A complex Hermitian metric -- a circular partner pair -- is admitted too.
    hermitian = np.array([[1.0, 0.3j], [-0.3j, 1.0]])
    assert modal_kerr_coefficient(
        **_METRIC_CASE, metric=hermitian, channel=1
    ) == pytest.approx(expected, rel=1e-14)


def test_a_one_channel_metric_needs_no_channel_index():
    """A square metric of one row is that channel's own, so `channel=` is optional."""
    unit = modal_kerr_coefficient(**_METRIC_CASE)
    assert modal_kerr_coefficient(
        **_METRIC_CASE, metric=np.array([[0.5]])
    ) == pytest.approx(unit / 0.5, rel=1e-15)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"metric": 0.0}, "positive W_mm"),
        ({"metric": -1.0}, "positive W_mm"),
        ({"metric": float("nan")}, "positive W_mm"),
        ({"metric": 1.0, "channel": 1}, "a scalar was given"),
        ({"channel": 2}, "the identity metric has one channel"),
        ({"metric": np.ones((2, 3))}, "square"),
        ({"metric": np.zeros((0, 0))}, "square"),
        ({"metric": np.eye(2)}, "which of the metric's 2 rows"),
        ({"metric": np.eye(2), "channel": 5}, "which of the metric's 2 rows"),
        ({"metric": np.array([[1.0, 2.0], [0.0, 1.0]]), "channel": 0}, "not Hermitian"),
        ({"metric": np.diag([1.0, 0.0]), "channel": 0}, "positive definite"),
        ({"metric": np.diag([1.0, -1.0]), "channel": 0}, "positive definite"),
    ],
)
def test_an_unusable_metric_is_refused_by_name(change, message):
    """Every way of declaring a metric that is not one is named, not approximated."""
    with pytest.raises(ModeError, match=message):
        modal_kerr_coefficient(**_METRIC_CASE, **change)
