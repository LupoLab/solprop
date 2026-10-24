# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spatial collar is an in-stage rate with a derived RK remainder budget."""

import dataclasses
import math
import tracemalloc
from functools import partial
from itertools import pairwise

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.special import gammainc, j0

from solprop.backends.memory import estimate_memory, ledger_groups
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.spatial import collar_field_fraction
from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, Absorber
from solprop.constants import C
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.spatial import SpatialMedium
from solprop.stepper import DOPRI5, FixedSchedule, PIController
from solprop.translate.problem import assemble
from spatial_records import record_metric

EPS = np.finfo(float).eps
LENGTH = 1e-4


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
    family = request.param
    constructor = (
        partial(EnvelopeGrid.from_limits, thg=False)
        if family == "envelope"
        else FieldGrid.from_limits
    )
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(300e-6, 16, window_factor=0.5)
    reference_length = 5 * LENGTH
    boundary = Absorber(
        np.zeros(grid.n_omega),
        np.zeros(grid.n_t),
        np.zeros(grid.n_to),
        grid.active_mask,
        reference_length,
    )
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    start = np.zeros((space.n_r, 2, grid.n_omega), dtype=np.complex128)
    start[:, 0, index] = np.exp(-(((np.arange(space.n_r) - 4) / 3) ** 2))
    start[:, 1, index] = 0.7j * start[:, 0, index]
    # Direct Fourier-Bessel forward and inverse formulas, independent of the
    # backend's composed matrix and with no temporal pad/crop factors.
    kernel = j0(space.r[:, None] * space.k_r[None, :])
    forward = kernel.T * space.radial_weights[None, :] / (2 * np.pi)
    inverse = kernel * space.spectral_weights[None, :] / (2 * np.pi)
    rate = TEMPORAL_ALPHA_LENGTH * (1 - space.window) / reference_length
    absorption = forward @ (-0.5 * rate[:, None] * inverse)
    wave = grid.omega[index] / C
    phase = 1j * space.k_r**2 / (np.sqrt(wave**2 - space.k_r**2) + wave)
    responses = (
        [KerrResponse(chi3=0.0, thg=False)] if family == "field_analytic" else []
    )
    return grid, space, boundary, start, index, absorption, phase, responses


def problem_for(case, steps, **options):
    grid, space, boundary, start, _, _, _, responses = case
    return assemble(
        grid,
        None,
        responses,
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        absorber=boundary,
        controller=PIController(rtol=1e-5, atol=1e-12),
        save_planes=[LENGTH],
        dz_init=LENGTH / steps,
        dz_max=LENGTH,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        **options,
    )


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
    problem = problem_for(absorber_case, 8)
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
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if isinstance(backend, NumpyRefBackend):
            backend._hoist(0.0)
            measured = backend._nonlinear(
                problem.initial.field0,
                backend._stage_scalars(0.0, 0.0),
                np.ones(problem.shape, complex),
            )[0]
        else:
            backend._trial_step(0.0, LENGTH / 8)
            measured = (
                backend._native_state.stage_to_host(0)
                if backend.name == "native"
                else backend._stages[0].copy()
            ).reshape(start.shape)
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
        problem = problem_for(absorber_case, count)
        backend = spatial_backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            measured = backend.observe().require_field()[0, :, :, index]
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


@pytest.mark.parametrize("temporal", [False, True])
def test_the_numba_term_is_the_references_on_its_own_samples_and_allocates_nothing(
    absorber_case, temporal
):
    """The planned transform and real GEMM against the reference's formula.

    The reference forms ``-rate/2 x + A x`` and transforms it through ``scipy.fft``;
    ``numba_cpu`` takes the composed matrix through a real GEMM (an analytic or
    envelope row by its real view) and transforms through its own FFTW plan. Both are
    read off the same fine samples the last stage left in the plan's buffer, so what
    differs is only the two summation orders.
    """
    grid = absorber_case[0]
    problem = problem_for(absorber_case, 8)
    if temporal:
        # A ramp rather than a constant, so that the rate is read sample by sample.
        rate = np.linspace(0.0, 2000.0, grid.n_to)
        rate.setflags(write=False)
        # Beside zeros on the coarse axis: the case's zero-rate collar translates to
        # no temporal absorber at all.
        coarse = np.zeros(grid.n_t)
        coarse.setflags(write=False)
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, temporal_absorber=coarse, temporal_absorber_fine=rate
            ),
        )
    else:
        # No temporal collar at all, so the radial term stands alone.
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, temporal_absorber=None, temporal_absorber_fine=None
            ),
        )
        rate = np.zeros(grid.n_to)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, LENGTH / 8)
        measured = backend._absorber_term().copy()
        samples = backend._plans.modal_time.copy()
        matrix = backend._spatial_absorber.copy()
        tracemalloc.start()
        backend._absorber_term()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    if problem.grid_kind == "field":
        samples = samples.real
    n_r, n_pol = problem.shape[1:3]
    rows = samples.reshape(n_r, -1)
    term = (-0.5 * rate * samples + (matrix @ rows).reshape(samples.shape)).reshape(
        n_r, n_pol, -1
    )
    expected = (
        passes.to_frequency(term, grid.crop_index, kind=problem.grid_kind)
        * grid.active_mask
    ).reshape(measured.shape)
    # Each time sample is an n_r-term dot product plus the rate's term, so differs by
    # gamma_(n_r+1) of its absolute summands; a bin sums n_to of those, and each FFT
    # adds about log2(n_to) roundings of the same absolute scale. Twice that, for the
    # two transforms.
    n_to = grid.n_to
    absolute = np.abs(matrix) @ np.abs(rows) + (0.5 * rate * np.abs(samples)).reshape(
        n_r, -1
    )
    operations = n_r + 1 + 4 * int(np.log2(n_to))
    bound = 2 * operations * EPS * n_to * float(np.max(absolute))
    assert bound > 0
    assert_allclose(measured, expected, rtol=0, atol=bound)
    # Nothing field-sized: one column of the fine spectrum is the smallest buffer
    # the term would need if it allocated one; what is left is Python's own views.
    assert peak < n_to * 16
    assert ledger["in_stage_absorber_term"] == measured.nbytes
    assert ledger["in_stage_absorber_spectrum"] > 0


@pytest.mark.parametrize("illumination", ["confined", "extended"])
def test_spatial_collar_diagnostic_reports_the_measured_fraction(
    absorber_case, spatial_backend_class, caplog, illumination
):
    grid, space, _, _, _, _, _, _ = absorber_case
    problem = problem_for(absorber_case, 8)
    field = problem.initial.field0[0].copy()
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
        assert expected > problem.stepper.controller.rtol
    else:
        assert expected < problem.stepper.controller.rtol
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(field[np.newaxis], None)
        backend.observe()
        if illumination == "extended":
            assert "spatial collar contains" in caplog.text
            assert f"{measured:.6e}" in caplog.text
        else:
            assert "spatial collar contains" not in caplog.text
    finally:
        backend.close()


def test_spatial_rate_stability_refusal_is_not_lifted_by_temporal_split(
    absorber_case,
):
    case = list(absorber_case)
    case[2] = dataclasses.replace(
        case[2], reference_length=case[2].reference_length / 1000
    )
    with pytest.raises(TranslationError, match="spatial collar stability limit"):
        problem_for(case, 8)
    with pytest.raises(TranslationError, match="spatial absorber is always in-stage"):
        problem_for(case, 8, absorber_scheme="split_step")


def test_memory_estimate_counts_the_composed_absorber_matrix(absorber_case):
    """The estimate sizes the radial absorber matrix without forming the product."""
    problem = problem_for(absorber_case, 8)
    backend = NumpyRefBackend(problem, grid=absorber_case[0])
    try:
        held = ledger_groups(backend.memory_ledger())
        matrix = backend.memory_ledger()["spatial_absorber_matrix"]
    finally:
        backend.close()
    assert matrix > 0
    assert estimate_memory(problem, "numpy_ref").groups["base"] == held["base"]
