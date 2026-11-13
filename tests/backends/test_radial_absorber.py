# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spatial collar is an in-stage rate with a derived RK remainder budget.

The collar is SEP 40's designed real-space collar, requested with a
:class:`~solprop.boundaries.SpatialAbsorber` on a grid with room outside its aperture;
its rate is the Planck profile ``alpha0 (1 - W)`` on the nodes.
"""

import dataclasses
import math
from itertools import pairwise

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.special import gammainc, j0

from backends.radial_absorber_cases import (
    COLLAR,
    LENGTH,
    problem_for,
    radial_absorber_case,
)
from sessions.runs import native_trial, run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.spatial import collar_field_fraction
from solprop.boundaries import SpatialAbsorber
from solprop.constants import C
from solprop.exceptions import TranslationError
from solprop.stepper import DOPRI5
from spatial_records import record_metric

EPS = np.finfo(float).eps


def exponential_tail(x, order):
    """Bound a positive exponential series from the requested order onwards."""
    if order <= 0:
        return math.exp(x)
    # gammainc(n,x)=exp(-x) sum_{j=n}^infinity x^j/j!, without cancellation.
    return math.exp(x) * gammainc(order, x)


def rk_bound(operator_norm, absorber_norm, steps):
    """Majorize the sixth-and-higher Taylor terms of exact and IP-RK flows."""
    h = LENGTH / steps
    tableau = DOPRI5
    bound = exponential_tail(h * (operator_norm + absorber_norm), 6)

    def path(stage, weight, length, sum_nodes):
        # Each path contributes h^l A(t_i)...A(t_j); A(t)=exp(-Dt) A exp(Dt).
        # The outside exp(Dh) and each conjugation give this exponential majorant.
        exponential = h * operator_norm * (1 + 2 * sum_nodes)
        term = (
            weight
            * (h * absorber_norm) ** length
            * exponential_tail(exponential, 6 - length)
        )
        for previous in range(stage):
            coefficient = abs(float(tableau.matrix[stage, previous]))
            if coefficient:
                term += path(
                    previous,
                    weight * coefficient,
                    length + 1,
                    sum_nodes + float(tableau.nodes[previous]),
                )
        return term

    for stage, weight in enumerate(tableau.solution_weights):
        bound += path(stage, abs(float(weight)), 1, float(tableau.nodes[stage]))
    # Order five cancels lower terms. Telescoping n steps adds this stability
    # envelope; it is set before either an expm reference or a backend is run.
    return steps * bound * math.exp(LENGTH * (operator_norm + absorber_norm))


@pytest.fixture(params=["field_real", "field_analytic", "envelope"])
def absorber_case(request):
    return radial_absorber_case(request.param)


@pytest.mark.parametrize("temporal_rate", [0.0, 1000.0])
def test_spatial_absorber_derivative_matches_independent_bessel_operator(
    absorber_case, spatial_backend_class, temporal_rate, request
):
    grid, _, _, start, index, absorption, _, _ = absorber_case
    # A uniform temporal power rate commutes with the Bessel operator and adds
    # -rate/2 times the identity, including the analytic-field representation.
    temporal = 0.5 * temporal_rate * np.eye(start.shape[0])
    # Two dense reductions and temporal transforms, plus special-function/scaling
    # arithmetic. Use the absolute summand scale, not a cancellation-prone result.
    operations = 128 * start.shape[0] + 32 * int(np.log2(grid.n_to))
    gamma = operations * EPS / (1 - operations * EPS)
    bound = gamma * float(
        np.max((np.abs(absorption) + temporal) @ np.abs(start[:, :, index]))
    )
    expected = (absorption - temporal) @ start[:, :, index]
    problem, run_request = problem_for(absorber_case, 8)
    rate = np.full(grid.n_to, temporal_rate)
    rate.setflags(write=False)
    # The case's collar has zero temporal rates, which translate to no temporal
    # absorber at all; the coarse axis is given back as zeros beside the fine rate.
    coarse = np.zeros(grid.n_t)
    coarse.setflags(write=False)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, temporal_absorber=coarse, temporal_absorber_fine=rate
        ),
    )
    backend = spatial_backend_class(problem, run_request, grid=grid)
    try:
        if isinstance(backend, NumpyRefBackend):
            backend.set_initial(problem.initial.field0, None)
            backend._hoist(0.0)
            measured = backend._nonlinear(
                problem.initial.field0[None],
                backend._stage_scalars(0.0, 0.0),
                np.ones((1, *problem.shape), complex),
            )[0]
        else:
            # The native state's own trial (``rk_trial`` on the uploaded stage
            # tables), loaded with the launch: its first stage is N(E, z=0).
            native_trial(backend, 0.0, LENGTH / 8, field=problem.initial.field0)
            measured = np.asarray(backend._native_state.stage_to_host(0)).reshape(
                start.shape
            )
        assert_allclose(measured[:, :, index], expected, rtol=0, atol=bound)
        record_metric(
            request,
            "absorber derivative (V)",
            f"{grid.kind}, analytic={bool(absorber_case[-1])}, rate={temporal_rate}",
            np.max(np.abs(measured[:, :, index] - expected)),
            bound,
            spatial_backend_class.name,
        )
        assert backend.memory_ledger()["spatial_absorber_matrix"] > 0
    finally:
        backend.close()


def test_spatial_absorber_step_refinement_matches_matrix_exponential(
    absorber_case, spatial_backend_class, request
):
    grid, _, _, start, index, absorption, phase, _ = absorber_case
    norm_d = float(np.max(np.abs(phase)))
    norm_a = float(np.linalg.norm(absorption, ord=np.inf))
    steps = (8, 16, 32)
    # Remainder, expm and dot-product arithmetic, then optical-frame cancellation.
    rounding = 256 * EPS * len(phase) * math.exp(LENGTH * (norm_d + norm_a))
    rounding += 64 * EPS * LENGTH * grid.omega[index] / C
    bounds = [rk_bound(norm_d, norm_a, count) + rounding for count in steps]
    reference = expm(LENGTH * (np.diag(phase) + absorption)) @ start[:, :, index]
    scale = float(np.max(np.abs(start)))
    measured_errors = []
    for count, bound in zip(steps, bounds, strict=True):
        problem, run_request = problem_for(absorber_case, count)
        backend = spatial_backend_class(problem, run_request, grid=grid)
        try:
            landed = run(backend, problem, run_request)
            assert landed.observations[-1].z == LENGTH
            measured = landed.fields[-1][:, :, index]
            error = float(np.max(np.abs(measured - reference))) / scale
            measured_errors.append(error)
            assert error <= bound, (count, error, bound)
            record_metric(
                request,
                f"absorber refinement {grid.kind}, "
                f"analytic={bool(absorber_case[-1])} (relative)",
                count,
                error,
                bound,
                spatial_backend_class.name,
            )
        finally:
            backend.close()
    # Require resolved fifth-order improvement until arithmetic dominates. The
    # window admits the next term of the RK expansion, independently of backend.
    for coarse, fine in pairwise(measured_errors):
        if fine > 16 * rounding:
            assert 20 <= coarse / fine <= 45, (measured_errors, rounding)


@pytest.mark.parametrize("illumination", ["confined", "extended"])
def test_spatial_collar_diagnostic_reports_the_measured_fraction(
    absorber_case, spatial_backend_class, illumination
):
    grid, space, _, _, _, _, _, _ = absorber_case
    problem, run_request = problem_for(absorber_case, 8)
    field = problem.initial.field0.copy()
    if illumination == "extended":
        # A highest-order Bessel column fills the disk and deliberately lights
        # the collar; the original confined spectrum remains a negative control.
        field[:] = 0
        field[-1, :, absorber_case[4]] = [1, 0.7j]
    kernel = j0(space.r[:, None] * space.k_r[None, :])
    # Monochromatic field: temporal and polarization factors cancel in the ratio.
    physical = (kernel * space.spectral_weights[None, :] / (2 * np.pi)) @ field[:, 0, :]
    area = space.radial_weights * np.sum(np.abs(physical) ** 2, axis=1)
    expected = float(np.sum(area[space.window < 1]) / np.sum(area))
    operations = 128 * space.n_r + 16 * grid.n_omega
    bound = operations * EPS / (1 - operations * EPS)
    measured = collar_field_fraction(problem, field)
    assert_allclose(measured, expected, rtol=bound, atol=0)
    if illumination == "extended":
        assert expected > run_request.stepper.controller.rtol
    else:
        assert expected < run_request.stepper.controller.rtol
    # The end-of-plane log of this share is retired (SEP 40 section 5.3): the
    # monitor's transverse edge judges the boundary instead.
    del spatial_backend_class


def test_spatial_rate_stability_refusal_is_not_lifted_by_temporal_split(
    absorber_case,
):
    # A thousand nepers at the band's edge: a collar of about 3e6 1/m, whose
    # stability limit is far below the case's eighth of a 0.1 mm run.
    strong = SpatialAbsorber(kappa=COLLAR.kappa, attenuation=1000.0)
    with pytest.raises(TranslationError, match="spatial collar stability limit"):
        problem_for(absorber_case, 8, spatial_absorber=strong)
    with pytest.raises(TranslationError, match="spatial absorber is always in-stage"):
        problem_for(
            absorber_case, 8, spatial_absorber=strong, absorber_scheme="split_step"
        )


def test_the_ledger_counts_the_composed_absorber_matrix(absorber_case):
    problem, run_request = problem_for(absorber_case, 8)
    backend = NumpyRefBackend(problem, run_request, grid=absorber_case[0])
    try:
        matrix = backend.memory_ledger()["spatial_absorber_matrix"]
    finally:
        backend.close()
    assert matrix > 0
