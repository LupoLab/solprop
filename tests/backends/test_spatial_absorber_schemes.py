# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the spatial in-stage rate independent of the temporal absorber policy.

Each case requests its module's real-space collar (SEP 40) through its own
``problem_for``.
"""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends import test_free_absorber, test_radial_absorber
from backends.test_radial_nonlinear import derivative
from bounds import gamma

absorber_case = test_radial_absorber.absorber_case
free_absorber = test_free_absorber.free_absorber


def assert_spatial_rate(case, spatial_backend_class, temporal_policy, problem_for):
    grid, _, _, start, index, absorption, _, _ = case
    # The same independent Bessel/DFT operator applies with an absent temporal
    # rate or one applied after a step. Budget dense reductions and time FFTs
    # against absolute summands, before evaluating any backend derivative.
    bound = gamma(128 * (start.shape[0] + grid.n_to)) * np.max(
        np.abs(absorption) @ np.abs(start[:, :, index])
    )
    expected = absorption @ start[:, :, index]
    problem = problem_for(case, 8)
    rate = None
    # The case's collar has zero temporal rates, which translate to no temporal
    # absorber; the split policy is given a fine rate beside zeros on the coarse axis.
    coarse = np.zeros(grid.n_t)
    coarse.setflags(write=False)
    if temporal_policy == "split_step":
        rate = np.full(grid.n_to, 1000.0)
        rate.setflags(write=False)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            temporal_absorber=None if rate is None else coarse,
            temporal_absorber_fine=rate,
            absorber_scheme="split_step" if rate is not None else "nonlinear",
        ),
    )
    measured = derivative(spatial_backend_class, problem, grid)
    assert_allclose(measured[:, :, index], expected, rtol=0, atol=bound)


@pytest.mark.parametrize("temporal_policy", ["absent", "split_step"])
def test_radial_rate_survives_independent_temporal_policy(
    absorber_case, spatial_backend_class, temporal_policy
):
    assert_spatial_rate(
        absorber_case,
        spatial_backend_class,
        temporal_policy,
        test_radial_absorber.problem_for,
    )


@pytest.mark.parametrize("temporal_policy", ["absent", "split_step"])
def test_cartesian_rate_survives_independent_temporal_policy(
    free_absorber, spatial_backend_class, temporal_policy
):
    assert_spatial_rate(
        free_absorber,
        spatial_backend_class,
        temporal_policy,
        test_free_absorber.problem_for,
    )
