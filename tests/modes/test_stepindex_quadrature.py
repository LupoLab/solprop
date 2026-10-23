# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify composite quadrature and the exact modes' electric-field overlaps."""

import math
from dataclasses import dataclass, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import dblquad

from solprop.constants import EPS0, C
from solprop.modes import StepIndexMode
from solprop.modes.collection import overlap
from solprop.modes.quadrature import mode_matrix, quadrature_rule, transverse_quadrature
from solprop.output.schema import serialise


@dataclass(frozen=True, eq=False)
class IntegrationMode(StepIndexMode):
    """Use dimensionless radii so adaptive absolute tolerances resolve the overlap.

    This test-only view puts both modes on a common disc and splits it at the
    physical interface. The Jacobian cancels against the scaled normalisations.
    """

    lower: float = 0.0
    outer: float = 1.0

    def dimlimits(self, z=0.0):
        return "polar", (self.lower, 0.0), (self.outer, 2 * math.pi)

    def field(self, points, z=0.0):
        return super().field(
            points * np.asarray([self.core_radius(np.asarray([z]))[0], 1.0]), z
        )

    def normalisation(self, z=0.0):
        return super().normalisation(z) / self.core_radius(np.asarray([z]))[0] ** 2


def overlap_values(delta):
    core = 1.45
    cladding = core * math.sqrt(1 - 2 * delta)
    radius = 6 * 1030e-9 / (2 * math.pi * math.sqrt(core**2 - cladding**2))
    first = IntegrationMode(
        radius, core=core, cladding=cladding, profile_wavelength=1030e-9
    )
    others = (
        replace(first, m=2),
        replace(first, n=2),
        replace(first, n=2, parity="odd"),
    )
    result = []
    for second in others:
        outer = (
            max(
                StepIndexMode.dimlimits(first)[2][0],
                StepIndexMode.dimlimits(second)[2][0],
            )
            / radius
        )
        result.append(
            overlap(first, second)
            + overlap(
                replace(first, lower=1.0, outer=outer),
                replace(second, lower=1.0, outer=outer),
            )
        )
    return np.asarray(result)


@pytest.mark.parametrize("kronrod", [False, True])
def test_stepindex_composite_pair_gram_and_effective_area(kronrod):
    mode = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
    partner = mode.orthogonal_polarisation
    assert partner is not None
    modes = [mode, partner]
    rule = transverse_quadrature(
        modes, n_r=24, n_theta=16, radial_rule="gauss_kronrod" if kronrod else "gauss"
    )
    assert rule.breaks == (1.25e-6 / mode.dimlimits()[2][0],)
    record = serialise(rule)
    assert isinstance(record, dict)
    assert record["breaks"] == list(rule.breaks)
    matrices = mode_matrix(modes, rule)
    gram = matrices.projection @ matrices.synthesis
    # The even/odd pair's off-diagonals vanish by angular symmetry in any rule
    # (measured 1.7e-18), so the radial split is what the DIAGONAL shows: 24
    # nodes per analytic panel reach 7.8e-16 against 1.4e-12 at 16 nodes.
    diagonal = float(np.max(np.abs(np.diag(gram) - 1.0)))
    assert diagonal < 1e-12
    assert np.max(np.abs(gram - np.diag(np.diag(gram)))) < 1e-15
    weights = rule.weights(mode.dimlimits())
    intensity = np.sum(mode.field(rule.nodes(mode.dimlimits())) ** 2, axis=1)
    square, quartic = weights @ intensity, weights @ intensity**2
    assert_allclose(EPS0 * C / 2 * square, mode.normalisation(), rtol=2e-12, atol=0)
    assert_allclose(square**2 / quartic, mode.aeff(), rtol=2e-12, atol=0)
    plain = quadrature_rule("polar", full=True, n_r=24, n_theta=16)
    unsplit = mode_matrix(modes, plain)
    # The unsplit rule straddles the radial-field discontinuity, and its miss is
    # on the diagonal too: measured 0.00329 here and 0.0033 at 48 nodes, so the
    # bound does not depend on the node count.
    unsplit_gram = unsplit.projection @ unsplit.synthesis
    assert float(np.max(np.abs(np.diag(unsplit_gram) - 1.0))) > 1e-3


def test_stepindex_integrals_against_adaptive_cartesian_components():
    mode = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
    radius = float(mode.core_radius(np.asarray([0.0]))[0])
    outer = mode.dimlimits()[2][0] / radius
    values = []
    for power in (1, 2):

        def integrand(r, theta, exponent=power):
            value = mode.field(np.asarray([[r * radius, theta]]))[0]
            return float(np.sum(value**2) ** exponent * r)

        value = sum(
            dblquad(integrand, 0.0, 2 * math.pi, left, right, epsabs=0.0, epsrel=2e-11)[
                0
            ]
            for left, right in ((0.0, 1.0), (1.0, outer))
        )
        values.append(radius**2 * value)
    # Independent adaptive angular integration, with a 2e-11 requested tolerance.
    assert_allclose(EPS0 * C / 2 * values[0], mode.normalisation(), rtol=4e-11, atol=0)
    assert_allclose(values[0] ** 2 / values[1], mode.aeff(), rtol=8e-11, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("delta", [0.005, 0.05])
def test_stepindex_exact_electric_overlaps_are_not_power_orthogonality(delta):
    values = overlap_values(delta)
    # Symmetry makes the different azimuthal orders orthogonal, while HE11/HE12
    # is O(Delta): measured 2.536e-4 at Delta = 0.005 and 3.026e-3 at 0.05, a
    # coefficient of 0.051 and 0.061. A slow-lane measurement.
    assert np.all(np.isfinite(values))
    assert 0.04 * delta < abs(values[0]) < 0.07 * delta
    assert_allclose(values[1:], 0, rtol=0, atol=1e-10)
