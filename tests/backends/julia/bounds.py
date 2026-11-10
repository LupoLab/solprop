# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bound roundoff for short, matched-history scalar RK and Adams witnesses."""

import math
from fractions import Fraction

import numpy as np

from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.responses import ResponseStage
from solprop.translate.run_request import MethodKind


def _envelopes(nodes):
    # Coefficientwise absolute product before cancellation, divided by exact
    # node differences. Integrate the positive polynomial on [0,1]. This bounds
    # every exponentially weighted basis for Re(hD)=0, including its formation.
    nodes = [Fraction(float(node)) for node in nodes]
    result = []
    for j, node in enumerate(nodes):
        polynomial = [Fraction(1)]
        denominator = Fraction(1)
        for k, other in enumerate(nodes):
            if j == k:
                continue
            product = [Fraction(0)] * (len(polynomial) + 1)
            for degree, coefficient in enumerate(polynomial):
                product[degree] += abs(other) * coefficient
                product[degree + 1] += coefficient
            polynomial = product
            denominator *= abs(node - other)
        integral = (
            sum(
                coefficient / (degree + 1)
                for degree, coefficient in enumerate(polynomial)
            )
            / denominator
        )
        result.append(math.nextafter(float(integral), math.inf))
    return result


def _kerr_bound(case):
    problem, request = case.problem, case.request
    assert request.replay is not None and not request.observation.dense
    assert problem.grid_kind == "envelope" and not problem.flags.has_noise
    response = problem.responses
    assert response.response_order == (ResponseStage.KERR, ResponseStage.ACCUMULATE)
    assert not response.raman_terms and not response.plasma_channels
    assert not response.spectral_responses
    assert response.temporal_absorber is None
    assert response.temporal_absorber_fine is None
    assert problem.transform.taper_scale is None
    assert not problem.linear.hard_kinks.size
    kerr = response.kerr_coefficients
    assert kerr is not None and kerr.cubic_form == "analytic"
    assert not kerr.take_real and not kerr.analytic_signal and kerr.harmonic_factor == 0
    density = kerr.density if kerr.density is not None else response.density
    assert density.coefficients.shape == (1, 1)
    assert (
        problem.transform.synthesis.shape
        == problem.transform.projection.shape
        == (1, 1)
    )
    prefactor = response.prefactor_constant
    assert prefactor is not None and np.all(np.isfinite(prefactor))
    # The one-node material map gives the cubic bound directly in internal state
    # units. This includes its actual normalization rather than assuming gamma is
    # unchanged by a future alteration to the test's mode or prefactor.
    return (
        abs(float(kerr.coefficient))
        * (abs(kerr.self_factor) + abs(kerr.conjugate_factor))
        * abs(float(density.coefficients[0, 0]))
        * abs(complex(problem.transform.synthesis[0, 0])) ** 3
        * abs(complex(problem.transform.projection[0, 0]))
        * float(np.max(np.abs(prefactor)))
    )


def _exact_stages(case, maximum_order, gamma_coefficient, initial):
    replay = case.request.replay.schedule
    steps, edges = replay.steps, replay.edges
    rate = case.problem.linear.phi_constant
    accepted_bound = initial
    source_bounds = [gamma_coefficient * initial**3]
    positions = [float(edges[0])]
    stages = []
    for index, step in enumerate(steps):
        z = float(edges[index])
        order = min(maximum_order, len(positions))
        old_positions = positions[-order:][::-1]
        nodes = [(position - z) / step for position in old_positions]
        assert all(abs(node) <= 6 for node in nodes)
        assert all(
            abs(a - b) >= 0.99 for j, a in enumerate(nodes) for b in nodes[j + 1 :]
        )
        predictor = sum(_envelopes(nodes))
        corrector = _envelopes([1.0, *nodes])
        endpoint, older = corrector[0], sum(corrector[1:])
        theta = float(np.max(np.abs(rate))) * float(step)
        assert theta <= 64
        p_bound = accepted_bound + step * predictor * max(source_bounds[-order:])
        new_source_bound = gamma_coefficient * p_bound**3
        accepted_bound += step * (
            older * max(source_bounds[-order:]) + endpoint * new_source_bound
        )
        stages.append(
            (step, order, predictor, endpoint, older, theta, p_bound, accepted_bound)
        )
        source_bounds.append(new_source_bound)
        positions.append(float(z + step))
    return stages


def _gamma(count):
    unit = np.finfo(float).eps / 2
    assert count * unit < 1
    return (count * unit) / (1 - count * unit)


def _wiener_data(case):
    gamma_coefficient = _kerr_bound(case)
    field = case.problem.initial.field0
    assert field.shape[:2] == (1, 1)
    n = field.shape[-1]
    assert case.problem.grids.n_to == n
    rate = case.problem.linear.phi_constant
    assert rate is not None and np.all(np.real(rate) == 0)
    replay = case.request.replay.schedule
    assert replay.n_steps == 8
    steps, edges = replay.steps, replay.edges
    assert np.all(steps == steps[0]) and edges[0] == 0

    gamma_coefficient = math.nextafter(gamma_coefficient * (1 + _gamma(32)), math.inf)
    # The time window is a convolution in spectrum, not a Wiener contraction.
    # Add its FFT rounding envelope before using its discrete Wiener norm.
    time_window = case.problem.grids.t_window_fine
    window_norm = math.fsum(map(float, abs(np.fft.fft(time_window)))) / n
    window_norm += _gamma(8 * math.ceil(math.log2(n))) * np.linalg.norm(time_window)
    gamma_coefficient *= math.nextafter(window_norm, math.inf)
    assert np.all(abs(case.problem.grids.omega_window) <= 1)
    # W = ||raw spectrum||_1/n is the discrete Wiener algebra norm: unitary
    # diagonal flow is an isometry, spectral masks are contractions, and
    # ||F(a)||_W <= gamma*||a||_W**3. A bootstrap ball is verified below.
    initial = math.nextafter(math.fsum(map(float, np.abs(field).ravel())) / n, math.inf)
    fft_levels = math.ceil(math.log2(n))
    source_gamma = _gamma(16 * fft_levels + 128)
    # FFT backward error in l2, then ||spectrum||_1 <= sqrt(n)||spectrum||_2.
    return n, gamma_coefficient, initial, source_gamma


def adams_rounding_budget(case, maximum_order):
    """Bound two fixed PEC histories using Wiener-norm polynomial majorants.

    This applies only to the explicitly checked scalar, constant-coefficient Kerr
    envelope with eight uniform steps, pure imaginary diagonal flow and no additional
    source. It bounds independent rounding in the same discrete method, not truncation
    error or adaptive accuracy. The recurrence follows the predictor, sampled source
    and corrector separately; stored predictor errors never become accepted-state
    source errors implicitly. The inputs alone set the bound before either run.
    """
    method = case.request.integration.method
    assert type(method) is Multistep
    assert not method.variable_order and method.order == maximum_order
    assert method.formulation == "exponential" and method.carrier_frame == "off"
    n, gamma_coefficient, initial, source_gamma = _wiener_data(case)
    unit = np.finfo(float).eps / 2
    stages = _exact_stages(case, maximum_order, gamma_coefficient, initial)
    largest_exact = max(initial, *(max(stage[-2:]) for stage in stages))
    # Select a bootstrap ball from the exact algebraic majorant, before observing
    # either backend result. The explicit error recurrence must fit the 1% margin.
    radius = math.nextafter(1.01 * largest_exact, math.inf)
    lipschitz = 3 * gamma_coefficient * radius**2
    source_roundoff = math.sqrt(n) * source_gamma * gamma_coefficient * radius**3
    state_error = 0.0
    source_errors = [source_roundoff]
    largest_radius = initial
    for step, order, predictor, endpoint, older, theta, p_bound, a_bound in stages:
        # 64(q+1)^3 dominates basis products/divisors, both dense contractions,
        # moment scaling and combination. The protected phi helper contributes
        # its independently qualified 512eps envelope; 16u*theta covers the
        # eight-step endpoint/argument-formation conditioning.
        rho = _gamma(64 * (order + 1) ** 3) + 1024 * unit + 16 * unit * theta
        p_round = rho * radius * (1 + step * gamma_coefficient * radius**2 * predictor)
        c_round = (
            rho
            * radius
            * (1 + step * gamma_coefficient * radius**2 * (endpoint + older))
        )
        p_error = state_error + step * predictor * max(source_errors[-order:]) + p_round
        new_source_error = lipschitz * p_error + source_roundoff
        state_error += (
            step * (older * max(source_errors[-order:]) + endpoint * new_source_error)
            + c_round
        )
        largest_radius = max(largest_radius, p_bound + p_error, a_bound + state_error)
        assert largest_radius < radius
        source_errors.append(new_source_error)
    # Both independent engines are compared with the same discrete PEC recurrence.
    # ||delta spectrum||_2 <= ||delta spectrum||_1 = n*||delta||_W.
    absolute_bound = 2 * n * state_error
    assert absolute_bound > 0
    return absolute_bound


def _rk_majorants(case, coefficient, initial, matrix, weights):
    accepted = initial
    stages = []
    for step in case.request.replay.schedule.steps:
        source_bounds, stage_bounds = [], []
        for row in range(len(weights)):
            stage = accepted + step * sum(
                abs(matrix[row, column]) * source_bounds[column]
                for column in range(row)
            )
            stage_bounds.append(stage)
            source_bounds.append(coefficient * stage**3)
        accepted += step * sum(
            abs(weight) * bound
            for weight, bound in zip(weights, source_bounds, strict=True)
        )
        stages.append((step, stage_bounds, accepted))
    return stages


def rk_rounding_budget(case):
    """Bound matched explicit RK histories with a checked discrete-stage radius.

    Midpoint RK4IP's forward and inverse picture transforms are Wiener isometries
    on this lossless diagonal problem, so its classical four-stage coefficients
    bound the same stage algebra. The local roundoff charge also includes those
    extra flows and its accumulator updates. Negative embedded-tableau coefficients
    enter through absolute values; no analytic solution's peak is assumed to bound
    the discrete stage fields (#656).
    """
    integration = case.request.integration
    if integration.kind is MethodKind.LOW_STORAGE:
        assert type(integration.method) is LowStorageRungeKutta
        # The fixed classical RK4 weights of SEP24, in midpoint picture coordinates.
        matrix = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        weights = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    else:
        assert integration.kind is MethodKind.RUNGE_KUTTA
        assert np.all(
            (integration.tableau.nodes >= 0) & (integration.tableau.nodes <= 1)
        )
        matrix = integration.tableau.matrix
        weights = integration.tableau.solution_weights
    assert np.all(np.triu(matrix) == 0)
    n, coefficient, initial, source_gamma = _wiener_data(case)
    stages = _rk_majorants(case, coefficient, initial, matrix, weights)
    largest = max(initial, *(max(*inputs, accepted) for _, inputs, accepted in stages))
    radius = math.nextafter(1.01 * largest, math.inf)
    lipschitz = 3 * coefficient * radius**2
    source_roundoff = math.sqrt(n) * source_gamma * coefficient * radius**3
    state_error = 0.0
    rate = case.problem.linear.phi_constant
    unit = np.finfo(float).eps / 2
    for step, inputs, accepted in stages:
        theta = float(np.max(np.abs(rate))) * step
        assert theta <= 64
        # Each source combination has at most s complex multiplies/adds plus two
        # picture flows;32(s+1)^2 exceeds eight real operations per complex term,
        # row-weight formation, and RK4IP's extra accumulator passes.16u*theta
        # covers endpoint subtraction/phase formation over these eight fixed steps.
        rho = _gamma(32 * (len(weights) + 1) ** 2) + 16 * unit * theta
        source_errors = []
        for row in range(len(weights)):
            coefficients = abs(matrix[row, :row])
            local = (
                rho * radius * (1 + step * coefficient * radius**2 * sum(coefficients))
            )
            stage_error = (
                state_error
                + step
                * sum(
                    weight * error
                    for weight, error in zip(coefficients, source_errors, strict=True)
                )
                + local
            )
            assert inputs[row] + stage_error < radius
            source_errors.append(lipschitz * stage_error + source_roundoff)
        local = rho * radius * (1 + step * coefficient * radius**2 * sum(abs(weights)))
        state_error += (
            step
            * sum(
                abs(weight) * error
                for weight, error in zip(weights, source_errors, strict=True)
            )
            + local
        )
        assert accepted + state_error < radius
    absolute_bound = 2 * n * state_error
    assert np.isfinite(absolute_bound) and absolute_bound > 0
    return absolute_bound
