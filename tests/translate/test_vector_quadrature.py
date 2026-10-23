# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check physical panels, angular products and independently bounded exterior flux."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modal_provider_cases import OMEGA, make_provider
from solprop.exceptions import ModeError
from solprop.modes._stepindex_vector import _power
from solprop.modes.guide import radial_node_rule


def test_material_panels_integrate_each_annulus():
    nodes = radial_node_rule((1e-6, 2e-6), 4e-6, n_r=4, n_theta=17)
    areas = np.bincount(nodes.region_index, weights=nodes.weights)
    # Gauss integrates r exactly; 64 epsilon permits weight and area summation.
    assert_allclose(
        areas,
        np.pi * np.array([1, 3, 12]) * 1e-12,
        rtol=64 * np.finfo(float).eps,
        atol=0,
    )
    with pytest.raises(ModeError, match="outer_radius"):
        radial_node_rule((), float("nan"), n_r=4, n_theta=9)


def test_mixed_family_radial_angular_and_tail_refinement():
    p = make_provider(
        descriptors=(
            ("HE", 1, 1, "even"),
            ("HE", 1, 1, "odd"),
            ("TE", 0, 1, "even"),
            ("TM", 0, 1, "even"),
            ("EH", 1, 1, "even"),
            ("HE", 2, 1, "odd"),
        )
    )
    defects = []
    for order in (24, 48, 96):
        nodes = p.node_rule(OMEGA, n_r=order, n_theta=25)
        sample = p.solve(OMEGA, 0.0, nodes=nodes)
        defects.append(np.linalg.norm(sample.flux_metric - np.eye(6), ord=2))
        assert np.max(np.asarray(nodes.identity["tail_power_bound"])) < 1e-12
    # The fixed Gram target; radial refinement independently of tail/angles.
    assert defects[-1] < 1e-9
    assert defects[1] < 1e-9
    angular = p.solve(OMEGA, 0.0, nodes=p.node_rule(OMEGA, n_r=96, n_theta=13))
    assert_allclose(angular.flux_metric, sample.flux_metric, rtol=0, atol=1e-9)
    assert np.linalg.cond(sample.flux_metric) < 1 + 1e-8


@pytest.mark.parametrize("radius", [250e-9, 1.5e-6, 25e-6])
def test_the_power_error_estimate_is_relative_at_every_scale(radius):
    # The quadrature's own error estimate has to mean the same thing for a
    # 250 nm strand and a 25 um fibre, because the node rule's discarded-power
    # certificate divides one by the other. A fixed epsabs of 1e-12 asked for
    # 4.8e-11 of the strand's power but 1.3e-14 of the fibre's, and it dominated
    # the exterior piece so completely that the tail's own estimate was 1.6e-2
    # relative -- useless as a bound. With a relative request the tails resolve
    # to about 1e-12 relative at every scale.
    p = make_provider(radius=radius)
    mode = p.modes[0]
    total, error = _power(mode, OMEGA, 0.0)
    assert 0 < error < 1e-11 * total
    tail, tail_error = _power(mode, OMEGA, 0.0, 2.0)
    assert tail > 0
    assert 0 < tail_error < 1e-10 * tail
