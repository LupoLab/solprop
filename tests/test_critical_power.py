# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Derive the Townes normalization independently of the propagated diagnostic."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_bvp

from hankel_oracles import gamma
from solprop.constants import EPS0, C
from solprop.exceptions import MaterialError
from solprop.spatial import critical_power
from spatial_records import record_metric


def test_critical_power_matches_the_independent_townes_boundary_value_problem(request):
    # Published alpha=1.86225 has five decimal places; half a final digit is the
    # predeclared rounding allowance. The oracle's residual and quadrature budgets
    # are separate and do not come from the propagated beam or its collapse point.
    mass_bound = 5e-6
    residual_budget = 1e-9
    quadrature_budget = 1e-10
    radius = np.linspace(0, 30, 600)
    initial = np.vstack((2 * np.exp(-radius), -2 * np.exp(-radius)))
    initial[1, 0] = 0
    result = solve_bvp(
        lambda r, q: np.vstack((q[1], q[0] - q[0] ** 3)),
        lambda left, right: np.array([left[1], right[0]]),
        radius,
        initial,
        S=np.array([[0.0, 0.0], [0.0, -1.0]]),
        tol=residual_budget,
        max_nodes=10000,
    )
    assert result.success, result.message
    assert np.max(result.rms_residuals) <= residual_budget
    assert result.y[0, 0] > 2
    assert np.min(result.y[0]) >= -residual_budget
    mass, quadrature_error = quad(
        lambda r: float(result.sol(r)[0] ** 2 * r),
        0,
        30,
        epsabs=quadrature_budget / 8,
        epsrel=0,
        limit=500,
    )
    assert quadrature_error < quadrature_budget
    # Preserve the source-precision miss as a negative control. The corrected
    # six-significant-figure coefficient is rounded from the BVP mass, not fitted
    # to the propagated beam, and must satisfy the same half-last-digit bound.
    assert abs(mass - 1.86225) > mass_bound
    assert abs(mass - 1.86226) < mass_bound
    record_metric(
        request,
        "Townes radial mass (1)",
        "rounded BVP",
        abs(mass - 1.86226),
        mass_bound,
    )
    record_metric(
        request,
        "Townes BVP residual (1)",
        "r=30",
        np.max(result.rms_residuals),
        residual_budget,
    )
    record_metric(
        request,
        "Townes mass quadrature (1)",
        "r=30",
        quadrature_error,
        quadrature_budget,
    )
    # The BVP mass bounds integral(|E_a|^2 dS), in which the index cancels; the
    # physical paraxial power is n0 eps0 c / 2 times it (axial Poynting flux).
    wavelength, chi3, index = 800e-9, 3e-26, 1.45
    expected = index * mass * EPS0 * C * wavelength**2 / (3 * np.pi * chi3)
    bound = mass_bound / mass + gamma(128)
    assert_allclose(
        critical_power(wavelength, chi3, index=index), expected, rtol=bound, atol=0
    )
    # 2*Townes mass > Gaussian's exact negative-Hamiltonian sufficient mass.
    assert 2 * (mass - mass_bound) > 2


def test_critical_power_has_wavelength_squared_and_inverse_susceptibility_scaling():
    base = critical_power(800e-9, 3e-26, index=1.0)
    # Log/exp evaluation and cancellation across decades, fewer than 128 eps.
    assert_allclose(
        critical_power(1600e-9, 3e-26, index=1.0), 4 * base, rtol=gamma(128), atol=0
    )
    assert_allclose(
        critical_power(800e-9, 6e-26, index=1.0), base / 2, rtol=gamma(128), atol=0
    )
    # At fixed chi3 the physical power is linear in n0: the field threshold is
    # index-free and the paraxial flux is n0 eps0 c |E|^2 / 2.
    assert_allclose(
        critical_power(800e-9, 3e-26, index=2.5), 2.5 * base, rtol=gamma(128), atol=0
    )


@pytest.mark.parametrize(
    "wavelength,chi3",
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (np.inf, 1.0),
        (1.0, np.nan),
        (1e300, 1e-300),
        (1e-300, 1e300),
    ],
)
def test_critical_power_refuses_invalid_or_unrepresentable_inputs(wavelength, chi3):
    with pytest.raises(MaterialError, match="critical power"):
        critical_power(wavelength, chi3, index=1.0)


@pytest.mark.parametrize("index", [0.0, -1.0, np.inf, np.nan])
def test_critical_power_refuses_an_invalid_index(index):
    with pytest.raises(MaterialError, match="critical power"):
        critical_power(800e-9, 3e-26, index=index)
