# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check TDSE P/J coupling and embedded diagnostics without an atomic provider."""

import dataclasses
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.tdse import (
    atomic_edge_source,
    atomic_sources,
    project_atomic_sources,
)
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.stepper import PIController
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import assemble
from solprop.translate.responses import prefactor_at
from solprop.translate.stage import scalars
from solprop.translate.tdse import (
    TDSEDescriptor,
    TDSEResult,
    _response_from_json,
    waveform_identity,
)
from solprop.translate.tdse_source import current_numerator


@pytest.fixture
def projection_case(request):
    grid = FieldGrid.from_limits(0.001, 800e-9, (500e-9, 1e-6), 8e-15)
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    rule = transverse_quadrature([mode], n_r=16, radial_rule="gauss_kronrod")
    problem = assemble(
        grid,
        mode,
        (),
        np.zeros((1, 1, grid.n_omega), dtype=np.complex128),
        controller=PIController(rtol=1e-2, atol=1e-14),
        save_planes=[0.001],
        dz_init=1e-5,
        dz_max=1e-4,
        density=1.0,
        quadrature=rule,
        components=("x",),
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 3, 1e-16, source="projection fixture"
    )
    calibration = TDSECalibration(
        "manufactured",
        "research",
        basis.identity,
        (500e-9, 1e-6),
        source="projection fixture",
        limitations=("no atomic qualification",),
    )
    model = (
        "optmat_cubic"
        if getattr(request, "param", None) == "optmat_cubic"
        else "tdse_native"
    )
    response = TDSEGasResponse(
        "helium", basis, 1e-19, model, 1.0, calibration, TDSESampling()
    )
    problem = dataclasses.replace(
        problem,
        tdse=TDSEDescriptor.from_response(response, grid),
        current_numerator=current_numerator(grid),
    )
    bundle = scalars(problem, 0.0, 0.0)
    return problem, grid, bundle, prefactor_at(problem.responses, 0.0)


def _project(polarisation, current, result, problem, bundle, prefactor, **options):
    """Project as the backend does, with a zero edge array for a gated source.

    These tests exercise the projection and its factors, not the gate: a native
    source is gated (SEP 26) and its projection requires an edge array, and a zero
    one leaves the spectrum the plain transform of the current handed over.
    """
    if problem.tdse is not None and problem.tdse.gated:
        options.setdefault("edge", np.zeros_like(current))
    return project_atomic_sources(
        polarisation, current, result, problem, bundle, prefactor, **options
    )


def response_record(problem, current, *, cancelled=None):
    first = np.zeros_like(current[:, 0, :]) if cancelled is None else cancelled
    return TDSEResult(
        current[:, 0, :] + first,
        first,
        input_identity="manufactured field",
        configuration_identity=problem.tdse.identity,
        valid=np.ones(first.shape, dtype=bool),
        shadow_orders=(1,),
    )


def carrier_case(problem, grid):
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    omega = grid.omega[index]
    time = np.arange(grid.n_to) * grid.dt_fine
    profile = problem.transform.synthesis[:, 0] / grid.pad_scale
    return index, omega, time, profile


def test_current_conductivity_has_analytic_attenuation_and_modal_normalization(
    projection_case,
):
    problem, grid, bundle, prefactor = projection_case
    index, _, _, _ = carrier_case(problem, grid)
    field = np.zeros((1, 1, grid.n_omega), dtype=np.complex128)
    field[0, 0, index] = 1.3 + 0.7j
    modal = passes.to_time(field, grid.crop_index, grid.n_to, kind="field")
    physical = passes.synthesise(modal, problem.transform.synthesis, 1.0, 1)
    conductivity = 1e-5
    current = conductivity * physical
    source, diagnostics = _project(
        np.zeros_like(current),
        current,
        response_record(problem, current),
        problem,
        bundle,
        prefactor,
    )
    expected = -conductivity * field / (2 * EPS0 * C * power_weight_at(problem, 0.0))
    expected[..., ~grid.active_mask] = 0.0
    # The 17-point radial rule integrates this smooth HE11 squared profile to
    # rounding; the bound includes modal quadrature and two FFT reduction paths.
    bound = 512 * np.finfo(float).eps * grid.n_to
    assert_allclose(source, expected, rtol=bound, atol=bound * np.max(np.abs(expected)))
    assert diagnostics["available"] is True
    assert np.real(source[0, 0, index] / field[0, 0, index]) < 0


def test_periodic_dipole_and_its_current_give_the_same_source(projection_case):
    problem, grid, bundle, prefactor = projection_case
    _, omega, time, profile = carrier_case(problem, grid)
    dipole = 1e-30 * profile[:, None, None] * np.cos(omega * time)[None, None, :]
    current = (
        -1e-30 * omega * profile[:, None, None] * np.sin(omega * time)[None, None, :]
    )
    zeros = np.zeros_like(current)
    by_dipole, _ = _project(
        dipole,
        zeros,
        response_record(problem, zeros),
        problem,
        bundle,
        prefactor,
    )
    by_current, _ = _project(
        zeros,
        current,
        response_record(problem, current),
        problem,
        bundle,
        prefactor,
    )
    # The carrier is an integer FFT mode, hence the periodic boundary term is
    # zero. The tolerance bounds sin/cos evaluation and FFT accumulation.
    bound = 512 * np.finfo(float).eps * grid.n_to
    assert_allclose(
        by_current, by_dipole, rtol=bound, atol=bound * np.max(np.abs(by_dipole))
    )


def test_embedded_indicator_includes_current_missed_by_the_coarse_nodes(
    projection_case,
):
    problem, grid, bundle, prefactor = projection_case
    _, omega, time, _ = carrier_case(problem, grid)
    omitted = int(np.flatnonzero(problem.transform.projection_coarse[0] == 0)[0])
    current = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    current[omitted, 0] = np.cos(omega * time)
    source, diagnostics = _project(
        np.zeros_like(current),
        current,
        response_record(problem, current),
        problem,
        bundle,
        prefactor,
    )
    assert np.any(source != 0)
    for name in ("complete_source", "remainder_source"):
        metric = diagnostics[name]
        assert isinstance(metric, dict)
        assert metric["floor_dominated"] is False
        assert metric["absolute_difference"] > metric["roundoff_floor"]
        # Every nonzero source row has zero embedded weight; the discrepancy
        # is the whole source, independently of the transform's normalization.
        assert metric["relative_difference"] == pytest.approx(
            1.0, rel=8 * np.finfo(float).eps
        )
    json.dumps(diagnostics, allow_nan=False)


def test_exact_shadow_cancellation_does_not_claim_relative_convergence(projection_case):
    problem, grid, bundle, prefactor = projection_case
    _, omega, time, profile = carrier_case(problem, grid)
    zero = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    cancelled = profile[:, None] * np.cos(omega * time)[None, :]
    source, diagnostics = _project(
        zero,
        zero,
        response_record(problem, zero, cancelled=cancelled),
        problem,
        bundle,
        prefactor,
    )
    assert_array_equal(source, np.zeros_like(source))
    for name in ("complete_source", "remainder_source"):
        metric = diagnostics[name]
        assert isinstance(metric, dict)
        assert metric["roundoff_floor"] > 0
        assert metric["floor_dominated"] is True
        assert metric["relative_difference"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_unpaired_rule_reports_unavailable_instead_of_a_zero_error(projection_case):
    problem, grid, bundle, prefactor = projection_case
    problem = dataclasses.replace(
        problem,
        transform=dataclasses.replace(
            problem.transform,
            projection_coarse=problem.transform.projection,
        ),
    )
    current = np.ones((problem.transform.n_nodes, 1, grid.n_to))
    _, diagnostics = _project(
        np.zeros_like(current),
        current,
        response_record(problem, current),
        problem,
        bundle,
        prefactor,
    )
    assert diagnostics["available"] is False
    complete, remainder = (
        diagnostics["complete_source"],
        diagnostics["remainder_source"],
    )
    assert isinstance(complete, dict) and isinstance(remainder, dict)
    assert complete["relative_difference"] is None
    assert remainder["absolute_difference"] is None


def test_projection_refuses_inconsistent_or_nonfinite_nodal_inputs(projection_case):
    problem, grid, bundle, prefactor = projection_case
    current = np.ones((problem.transform.n_nodes, 1, grid.n_to))
    result = response_record(problem, current)
    with pytest.raises(BackendError, match="shapes"):
        _project(current[:-1], current, result, problem, bundle, prefactor)
    current[0, 0, 0] = np.nan
    with pytest.raises(BackendError, match="finite"):
        _project(current, current, result, problem, bundle, prefactor)


@pytest.mark.parametrize("missing", ["tdse", "current_numerator"])
def test_projection_requires_both_separate_source_descriptors(projection_case, missing):
    problem, grid, bundle, prefactor = projection_case
    current = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    result = response_record(problem, current)
    problem = dataclasses.replace(problem, **{missing: None})
    with pytest.raises(BackendError, match="independent current and polarisation"):
        _project(current, current, result, problem, bundle, prefactor)


def test_projection_refuses_mismatched_cancellation_rows(projection_case):
    problem, grid, bundle, prefactor = projection_case
    current = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    result = response_record(problem, current[:-1])
    with pytest.raises(BackendError, match="cancellation operands disagree"):
        _project(current, current, result, problem, bundle, prefactor)


def test_projection_refuses_nonfinite_spectral_factors(projection_case):
    problem, grid, bundle, prefactor = projection_case
    current = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    result = response_record(problem, current)
    prefactor = np.full_like(prefactor, np.nan)
    with pytest.raises(BackendError, match="projected source is nonfinite"):
        _project(current, current, result, problem, bundle, prefactor)


@pytest.mark.parametrize("projection_case", ["optmat_cubic"], indirect=True)
def test_cubic_projection_requires_third_order_cancellation_operands(projection_case):
    problem, grid, bundle, prefactor = projection_case
    current = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    result = response_record(problem, current)
    with pytest.raises(BackendError, match="third-order cancellation operands"):
        _project(current, current, result, problem, bundle, prefactor)


@pytest.mark.parametrize("projection_case", ["optmat_cubic"], indirect=True)
def test_cubic_roundoff_floor_counts_the_unscaled_third_order_operand(projection_case):
    problem, grid, bundle, prefactor = projection_case
    zero = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    first = np.ones((problem.transform.n_nodes, grid.n_to))
    linear = response_record(problem, zero, cancelled=first)
    base = dataclasses.replace(
        linear, shadow_orders=(1, 3), third_order_current=np.zeros_like(first)
    )
    cubic = dataclasses.replace(
        linear, total_current=2 * first, shadow_orders=(1, 3), third_order_current=first
    )
    source, before = _project(zero, zero, base, problem, bundle, prefactor)
    changed, after = _project(zero, zero, cubic, problem, bundle, prefactor)
    assert_array_equal(source, changed)
    for name in ("complete_source", "remainder_source"):
        base_metric, cubic_metric = before[name], after[name]
        assert isinstance(base_metric, dict) and isinstance(cubic_metric, dict)
        assert base_metric["roundoff_floor"] > 0
        # Both exact remainders vanish. Absolute raw operands double from
        # |J|+|J1|=2 to |J|+|J1|+|J3|=4, so their floor must double as well.
        assert cubic_metric["roundoff_floor"] == pytest.approx(
            2 * base_metric["roundoff_floor"], rel=8 * np.finfo(float).eps, abs=0
        )
        assert cubic_metric["relative_difference"] is None


def test_projection_refuses_overflow_in_a_cancelled_source_diagnostic(projection_case):
    problem, grid, bundle, prefactor = projection_case
    zero = np.zeros((problem.transform.n_nodes, 1, grid.n_to))
    extreme = np.full((problem.transform.n_nodes, grid.n_to), np.finfo(float).max)
    result = response_record(problem, zero, cancelled=extreme)
    # The physical difference is exactly zero but |J|+|J1| overflows: emitting a
    # finite-looking zero-error diagnostic would conceal the unusable bound.
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(BackendError, match="projected diagnostic is nonfinite"),
    ):
        _project(zero, zero, result, problem, bundle, prefactor)


def test_filtered_projection_keeps_core_impulse_and_counts_cancelled_operands(
    projection_case,
):
    problem, grid, bundle, prefactor = projection_case
    response = dataclasses.replace(
        _response_from_json(problem.tdse.configuration_json),
    )
    problem = dataclasses.replace(
        problem, tdse=TDSEDescriptor.from_response(response, grid)
    )
    fields = np.zeros((problem.transform.n_nodes, grid.n_to))
    nonlinear = np.zeros_like(fields)
    nonlinear[:, 0] = 4.0
    nonlinear[:, grid.n_to // 2] = 1.0
    first = np.full_like(fields, 2.0**30)
    record = TDSEResult(
        first + nonlinear,
        first,
        input_identity=waveform_identity(fields),
        configuration_identity=problem.tdse.identity,
        valid=np.ones(fields.shape, dtype=bool),
        shadow_orders=(1,),
    )
    polarization, current = atomic_sources(fields, record, problem, bundle)
    # Both impulses sit where g' = 0 (outside the collar and on its plateau), so the
    # edge term is exactly zero and the gated source is the truncating one here.
    edge = atomic_edge_source(record, problem, bundle)
    assert edge is not None
    assert_array_equal(edge, 0)
    source, diagnostics = _project(
        polarization, current, record, problem, bundle, prefactor, edge=edge
    )
    # The edge impulse is exactly removed, the central impulse exactly retained.
    # Its DFT is (-1)^k. The remaining scale is the independent modal sum and
    # current coefficient, with no finite-window or integration approximation.
    expected = (
        problem.current_numerator
        / power_weight_at(problem, bundle.z1)
        * np.sum(problem.transform.projection, axis=1)[:, None, None]
        * bundle.density
        / bundle.taper_scale
        * (-1.0) ** np.arange(grid.n_omega)
    )
    bound = 16 * np.finfo(float).eps * problem.transform.n_nodes
    assert_allclose(source, expected, rtol=bound, atol=bound * np.max(abs(expected)))
    without_linear = dataclasses.replace(
        record, total_current=nonlinear, first_order_current=np.zeros_like(first)
    )
    _, small = _project(
        polarization, current, without_linear, problem, bundle, prefactor, edge=edge
    )
    for name in ("complete_source", "remainder_source"):
        large_metric, small_metric = diagnostics[name], small[name]
        assert isinstance(large_metric, dict) and isinstance(small_metric, dict)
        assert large_metric["roundoff_floor"] > 1e6 * small_metric["roundoff_floor"]
    assert_array_equal(record.total_current, first + nonlinear)
    assert_array_equal(record.first_order_current, first)
