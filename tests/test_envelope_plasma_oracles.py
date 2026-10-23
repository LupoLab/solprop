# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the independent oracle calculations, without installing a new response."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special

from envelope_oracle_angular import (
    BUDGETS,
    axis_moments,
    cartesian_oracle,
    order_record,
    polynomial_experiments,
    population,
    population_experiments,
    reconstructed_current,
    refined_moments,
)
from envelope_oracle_ppt import (
    circular_coulomb_audit,
    circular_short_range,
    elliptic_action,
    elliptic_short_range,
)
from envelope_oracle_spectral import (
    carrier_envelope_experiments,
    causal_case,
    causal_experiments,
    spectral_experiments,
)
from envelope_oracle_yudin import yudin_action_integral, yudin_exponent
from solprop.exceptions import ConvergenceError


def test_polynomial_both_components_work_and_covariance():
    for row in polynomial_experiments():
        # 64 operations per angular node, absolute integrands bounded by 32.
        bound = 64 * 128 * np.finfo(float).eps * 32
        assert row["rate_error"] < bound
        assert row["current_error"] < bound
        assert row["work_error"] < bound
        assert row["covariance_error"] < bound
    # The counterexample must reject collinear current despite equal work.
    assert polynomial_experiments()[2]["collinear_relative_error"] > 0.18


@pytest.mark.parametrize("power", [2, 4, 6, 8])
def test_scalar_polynomial_analytic_cycle_average(power):
    average, current, _ = cartesian_oracle(np.array([1.7 + 0j]), lambda e: e**power)
    coefficient = math.comb(power, power // 2) / 2**power
    expected = 1.7**power * coefficient
    # Relative rounding allowance covers quadrature and power evaluation.
    assert_allclose(average, expected, rtol=128 * np.finfo(float).eps, atol=0)
    assert_allclose(
        current, [2 * expected / 1.7], rtol=256 * np.finfo(float).eps, atol=0
    )


def test_zero_field_and_degenerate_axes():
    rate, current, _ = cartesian_oracle(np.zeros(2, dtype=complex), lambda e: e**4)
    assert rate == 0
    assert np.array_equal(current, np.zeros(2))
    _, transformed = reconstructed_current(np.zeros(2, dtype=complex), lambda e: e**4)
    assert np.array_equal(transformed, np.zeros(2))
    for ratio in (0, 1e-14, 1 - 1e-14, 1):
        moments = axis_moments(1.0, ratio, lambda e: e**4, 64)
        # Integrands are degree-four trigonometric polynomials; roundoff dominates.
        assert_allclose(
            moments,
            [(3 + ratio**2) / 8, (1 + 3 * ratio**2) / 8],
            rtol=128 * np.finfo(float).eps,
            atol=0,
        )


def test_unresolved_angular_discontinuity_requires_an_explicit_split():
    def step_rate(field):
        return np.where(field > 0.873, 1.0, 0.0)

    with pytest.raises(ConvergenceError, match="512"):
        refined_moments(1.0, 0.2, step_rate, ())
    actual, _ = refined_moments(1.0, 0.2, step_rate, (0.873,))
    reference = axis_moments(1.0, 0.2, step_rate, 1024, edges=(0.873,))
    assert_allclose(actual, reference, rtol=BUDGETS["angular_rtol"], atol=0)


@pytest.mark.parametrize("preionisation", [0, 0.2, 1])
def test_population_first_sample_and_stable_limits(preionisation):
    fraction, survival = population(np.full(3, 1e-18), 0.5, preionisation)
    assert fraction[0] == preionisation
    assert np.all(np.isfinite(fraction))
    assert np.all(fraction >= preionisation)
    assert np.all(survival >= 0)
    if preionisation == 0:
        assert fraction[-1] == 1e-18
    fraction, survival = population(np.full(3, 1000.0), 0.5, preionisation)
    assert fraction[-1] == 1
    assert survival[-1] == 0
    # Vacuum multiplies each physical source by zero even with an occupied fraction.
    assert np.array_equal(0 * fraction, np.zeros(3))


@pytest.mark.slow
def test_trapezoid_population_has_three_level_order():
    assert population_experiments()["trapezoid"]["status"] == "pass"


def test_floor_and_superconvergence_do_not_claim_an_order_pass():
    assert order_record([1e-16, 1e-17, 1e-18], 2, 1e-16)["status"].startswith(
        "inconclusive"
    )
    assert order_record([1e-1, 1e-4, 1e-7], 2, 1e-14)["status"] == "fail"


def test_band_factors_have_physical_signs_and_carrier_invariance():
    result = spectral_experiments()
    assert result["inactive_exact_zero"]
    assert result["relabel_error"] < BUDGETS["spectral_rounding_relative"]
    assert all(value <= 0 for value in result["uppe_loss_real"])
    assert all(value >= 0 for value in result["uppe_drude_imaginary"])
    # The off-carrier values distinguish 1/omega from the erroneous omega/omega0².
    assert_allclose(
        result["uppe_drude_imaginary"][3:7],
        [4 / 3, 1, 0.8, 2 / 3],
        rtol=BUDGETS["spectral_rounding_relative"],
        atol=0,
    )


@pytest.mark.slow
def test_causal_integration_three_level_order_and_residual_diagnostic():
    result = causal_experiments()
    assert result["temporal"]["status"] == "pass"
    controlled = causal_case(16, 2, 4096, 12)
    assert controlled["collared_relative_error"] < BUDGETS["window_relative_target"]
    assert (
        controlled["loss_integral_relative_error"] < BUDGETS["window_relative_target"]
    )
    outside = causal_case(16, 4, 4096, 2)
    assert outside["collared_relative_error"] > 0.1


@pytest.mark.slow
def test_carrier_envelope_work_and_controlled_regime():
    result = carrier_envelope_experiments()
    assert result["work_temporal"]["status"] == "pass"
    for row in result["cases"]:
        if (
            row["cycles_sigma"] == 8
            and row["peak_instantaneous_rate_over_carrier"] == 0.001
        ):
            assert (
                max(row["loss_component_relative_errors"])
                < BUDGETS["controlled_envelope_relative_target"]
            )
            assert (
                row["drude_relative_error"]
                < BUDGETS["controlled_envelope_relative_target"]
            )


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1, 3])
def test_elliptic_action_recovers_linear_keldysh(gamma):
    tau, action = elliptic_action(gamma, 0)
    exact = (1 + 1 / (2 * gamma**2)) * np.arcsinh(gamma) - math.sqrt(1 + gamma**2) / (
        2 * gamma
    )
    # The action subtracts O(1/gamma) terms; this bounds cancellation at gamma=0.1.
    assert_allclose(tau, np.arcsinh(gamma), rtol=1e-12, atol=0)
    assert_allclose(action, exact, rtol=1e-12, atol=0)
    assert elliptic_action(gamma, 1)[1] > action


def test_circular_orbital_rates_converge_and_resolve_rotation():
    low, _ = circular_short_range(0.5792, 0.06, 0.057, 32, 256)
    high, tail = circular_short_range(0.5792, 0.06, 0.057, 64, 512)
    assert_allclose(low, high, rtol=BUDGETS["short_range_ppt_quadrature_rtol"], atol=0)
    assert tail < BUDGETS["short_range_ppt_quadrature_rtol"]
    assert high[3] > high[2] > high[1] > 0


def test_circular_coulomb_audit_static_limit_and_hydrogenic_reduction():
    field, omega = 0.06, 0.057
    correction, scale = circular_coulomb_audit(0.5, field, omega)
    # For Ip=1/2 and Z=1, kappa=n*=1; the expressions reduce algebraically.
    rounding = 16 * np.finfo(float).eps
    assert_allclose(correction, 1 / (field**2 + omega**2), rtol=rounding, atol=0)
    assert_allclose(scale, omega / math.sqrt(field), rtol=rounding, atol=0)
    static, scale = circular_coulomb_audit(0.5, field, 0)
    assert_allclose(static, field**-2, rtol=rounding, atol=0)
    assert scale == 0


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1, 2, 5])
def test_yudin_phase_formula_matches_complex_action_and_peak_keldysh(gamma):
    phase = np.array([0.0, 0.4, 1.2, np.pi / 2])
    # Both analytic evaluations cancel O(gamma) terms to recover O(gamma**3).
    rounding = 2048 * np.finfo(float).eps / min(gamma**2, 1)
    actual = yudin_exponent(gamma, phase)
    assert_allclose(actual, yudin_action_integral(gamma, phase), rtol=rounding, atol=0)
    assert_allclose(
        actual[0], gamma**2 * elliptic_action(gamma, 0)[1], rtol=rounding, atol=0
    )
    assert np.all(actual > 0)
    assert np.isfinite(actual[-1])


def test_elliptic_ppt_photon_sum_linear_limit_independent_dawson_integral():
    potential, field, omega = 0.5792, 0.06, 0.057
    gamma = omega * math.sqrt(2 * potential) / field
    tau, action = elliptic_action(gamma, 0)
    tangent = np.tanh(tau)
    threshold = potential / omega * (1 + 1 / (2 * gamma**2))
    excess = (
        np.arange(math.floor(threshold) + 1, math.floor(threshold) + 513) - threshold
    )
    # Popov (1968), Eq. (51), independent analytic reduction of the Bessel integral.
    expected = (
        omega
        / np.pi
        * math.sqrt(omega * tangent / (2 * potential))
        * math.exp(-2 * potential / omega * action)
        * np.sum(
            np.exp(-2 * (tau - tangent) * excess)
            * special.dawsn(np.sqrt(2 * tangent * excess))
        )
    )
    actual, tail = elliptic_short_range(potential, field, omega, 0, 0)
    assert_allclose(
        actual, expected, rtol=BUDGETS["short_range_ppt_quadrature_rtol"], atol=0
    )
    assert tail < BUDGETS["short_range_ppt_quadrature_rtol"]
