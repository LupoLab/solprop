# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check microscopic shadow accounting independently of an atomic provider.

The manufactured arrays are exact dyadic numbers. They identify accidental
shadow scaling and duplicate density factors without integration error or a
physical helium calibration entering the comparison.
"""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tests.hacc_cases import hacc_response

from solprop.backends.numpy_ref.tdse import atomic_sources
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse
from solprop.nonlinear.tdse import (
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import assemble
from solprop.translate.responses import ResponseStage
from solprop.translate.stage import scalars
from solprop.translate.tdse import TDSEDescriptor, TDSEResult, waveform_identity
from solprop.translate.tdse_source import current_numerator

ROUNDING = 8 * np.finfo(float).eps
"""A few rounded products of the nondyadic density and scales, on either side."""

CHI3 = 2.0**-100


@pytest.fixture
def source_case():
    grid = FieldGrid.from_limits(0.001, 800e-9, (500e-9, 1e-6), 8e-15)
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    problem = assemble(
        grid,
        mode,
        [KerrResponse(chi3=CHI3, thg=True)],
        np.zeros((1, 1, grid.n_omega), dtype=np.complex128),
        controller=PIController(rtol=1e-2, atol=1e-12),
        save_planes=[0.001],
        dz_init=1e-5,
        dz_max=1e-4,
        density=8.0,
        quadrature=transverse_quadrature([mode], n_r=4),
        components=("x",),
        integrator=Multistep(formulation="interaction"),
        stats_spec=StatsSpec(names=("energy",)),
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 3, 1e-16, source="source algebra fixture"
    )
    calibration = TDSECalibration(
        "source-algebra",
        "research",
        basis.identity,
        (500e-9, 1e-6),
        source="source algebra fixture",
        limitations=("no atomic qualification",),
    )
    response = TDSEGasResponse(
        "helium",
        basis,
        1e-19,
        "tdse_shadow",
        2.0,
        calibration,
        TDSESampling(),
        cubic_scale=3.0,
    )
    # The ordinary compiler supplies an independently specified analytic Kerr
    # coefficient. Attaching a manufactured descriptor isolates source algebra
    # from material admission and avoids installing the optional atomic engine.
    problem = dataclasses.replace(
        problem,
        tdse=TDSEDescriptor.from_response(response, grid),
        current_numerator=current_numerator(grid),
    )
    bundle = scalars(problem, 0.0, 0.0)
    row = np.arange(problem.transform.n_nodes, dtype=np.int64)[:, None]
    sample = np.arange(grid.n_to, dtype=np.int64)[None, :]
    fields = np.asarray((row + sample) % 4 - 2, dtype=np.float64)
    return problem, grid, bundle, response, fields


def model_case(case, model, *, remainder_scale=2.0, cubic_scale=3.0):
    problem, grid, bundle, response, fields = case
    response = dataclasses.replace(
        response,
        kerr_model=model,
        remainder_scale=1.0 if model == "tdse_native" else remainder_scale,
        cubic_scale=cubic_scale if model == "tdse_shadow" else None,
    )
    compiled = problem.responses
    if model != "optmat_cubic":
        # Only the optmat closure owns an ordinary compiled cubic response.
        compiled = dataclasses.replace(
            compiled, kerr_coefficients=None, response_order=(ResponseStage.ACCUMULATE,)
        )
    problem = dataclasses.replace(
        problem,
        tdse=TDSEDescriptor.from_response(response, grid),
        responses=compiled,
    )
    return problem, bundle, fields


def microscopic_record(problem, fields, *, sign=1.0, cancel=False):
    first = sign * 2.0**20 * fields
    third = sign * 4.0 * fields**3
    remainder = np.zeros_like(fields) if cancel else sign * 2.0**-3 * fields**5
    dipole = sign * 2.0**-20 * fields**3
    assert problem.tdse is not None
    native = problem.tdse.kerr_model == "tdse_native"
    result = TDSEResult(
        first + third + remainder,
        first,
        input_identity=waveform_identity(sign * fields),
        configuration_identity=problem.tdse.identity,
        valid=np.ones(fields.shape, dtype=bool),
        shadow_orders=(1,) if native else (1, 3),
        third_order_current=None if native else third,
        third_order_dipole=None if native else dipole,
    )
    return result, third, remainder, dipole


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow", "tdse_native"])
def test_each_closure_has_its_own_cubic_and_remainder_definition(source_case, model):
    problem, bundle, fields = model_case(source_case, model)
    result, third, remainder, dipole = microscopic_record(problem, fields)
    polarisation, current = atomic_sources(fields, result, problem, bundle)
    assert problem.tdse is not None
    # The native source is gated by the grid collar (SEP 26); the cubic ones are not.
    gate = source_case[1].t_window_fine
    expected_current = bundle.density * (
        (third + remainder) * gate if model == "tdse_native" else 2.0 * remainder
    )
    # Every current operand and scale is dyadic and exactly representable; only the
    # nondyadic density and the gate round, a few products on either side: ROUNDING.
    assert_allclose(current[:, 0, :], expected_current, rtol=ROUNDING, atol=0)
    if model == "tdse_shadow":
        assert_allclose(
            polarisation[:, 0, :], 3.0 * bundle.density * dipole, rtol=ROUNDING, atol=0
        )
    elif model == "tdse_native":
        assert_array_equal(polarisation, np.zeros_like(polarisation))
    else:
        expected = (
            EPS0 * CHI3 * bundle.density * fields**3 * problem.grids.t_window_fine
        )
        # eps0 and the source collar are nondyadic; allow only a handful of
        # rounding operations, with zero absolute tolerance at exact zero nodes.
        assert_allclose(
            polarisation[:, 0, :], expected, rtol=8 * np.finfo(float).eps, atol=0
        )
    assert not np.shares_memory(current, result.total_current)
    if result.third_order_dipole is not None:
        assert not np.shares_memory(polarisation, result.third_order_dipole)


@pytest.mark.parametrize("scale", [1.0, 1.47, 2.0])
def test_native_scale_preserves_complete_nonlinearity_after_linear_subtraction(
    source_case, scale
):
    problem, grid, bundle, response, fields = source_case
    response = dataclasses.replace(
        response, kerr_model="tdse_native", remainder_scale=scale, cubic_scale=None
    )
    problem = dataclasses.replace(
        problem,
        tdse=TDSEDescriptor.from_response(response, grid),
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=None,
            response_order=(ResponseStage.ACCUMULATE,),
        ),
    )
    result, third, remainder, _ = microscopic_record(problem, fields)
    original = result.total_current.copy()
    polarisation, current = atomic_sources(fields, result, problem, bundle)
    # The large linear shadow cancels exactly in the dyadic manufactured data.
    # Only multiplication by the gate and the nondyadic scale/density can round.
    assert_allclose(
        current[:, 0, :],
        bundle.density * scale * ((third + remainder) * grid.t_window_fine),
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    assert_array_equal(polarisation, np.zeros_like(polarisation))
    assert_array_equal(result.total_current, original)
    assert not result.total_current.flags.writeable
    assert not np.shares_memory(current, result.total_current)


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow"])
def test_unscaled_linear_and_cubic_shadows_cancel_exactly(source_case, model):
    problem, bundle, fields = model_case(
        source_case, model, remainder_scale=7.0, cubic_scale=5.0
    )
    result, _, _, _ = microscopic_record(problem, fields, cancel=True)
    original = result.total_current.copy()
    _, current = atomic_sources(fields, result, problem, bundle)
    assert_array_equal(current, np.zeros_like(current))
    assert_array_equal(result.total_current, original)
    assert not result.total_current.flags.writeable


def test_cubic_and_remainder_calibrations_are_independent(source_case):
    problem, bundle, fields = model_case(source_case, "tdse_shadow")
    result, _, remainder, dipole = microscopic_record(problem, fields)
    base_p, base_j = atomic_sources(fields, result, problem, bundle)
    changed_p_problem, _, _ = model_case(source_case, "tdse_shadow", cubic_scale=5.0)
    changed_p_result, _, _, _ = microscopic_record(changed_p_problem, fields)
    changed_p, unchanged_j = atomic_sources(
        fields, changed_p_result, changed_p_problem, bundle
    )
    # The calibrated half moves by its factor, to the density's few roundings; the
    # other half is the same evaluation again.
    assert_allclose(
        changed_p[:, 0, :], 5.0 * bundle.density * dipole, rtol=ROUNDING, atol=0
    )
    assert_allclose(unchanged_j, base_j, rtol=ROUNDING, atol=0)
    changed_j_problem, _, _ = model_case(
        source_case, "tdse_shadow", remainder_scale=7.0
    )
    changed_j_result, _, _, _ = microscopic_record(changed_j_problem, fields)
    unchanged_p, changed_j = atomic_sources(
        fields, changed_j_result, changed_j_problem, bundle
    )
    assert_allclose(unchanged_p, base_p, rtol=ROUNDING, atol=0)
    assert_allclose(
        changed_j[:, 0, :], 7.0 * bundle.density * remainder, rtol=ROUNDING, atol=0
    )


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow", "tdse_native"])
def test_density_is_applied_once_and_zero_density_has_zero_source(source_case, model):
    problem, bundle, fields = model_case(source_case, model)
    result, _, _, _ = microscopic_record(problem, fields)
    base_p, base_j = atomic_sources(fields, result, problem, bundle)
    doubled = dataclasses.replace(
        bundle, density=2.0 * bundle.density, kerr_density=2.0 * bundle.kerr_density
    )
    doubled_p, doubled_j = atomic_sources(fields, result, problem, doubled)
    # Structural: doubling a factor is a binary scaling, exact in any order.
    assert_array_equal(doubled_p, 2.0 * base_p)
    assert_array_equal(doubled_j, 2.0 * base_j)
    zero = dataclasses.replace(bundle, density=0.0, kerr_density=0.0)
    zero_p, zero_j = atomic_sources(fields, result, problem, zero)
    assert_array_equal(zero_p, np.zeros_like(zero_p))
    assert_array_equal(zero_j, np.zeros_like(zero_j))


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow", "tdse_native"])
def test_reversing_the_field_and_odd_atomic_response_reverses_both_sources(
    source_case, model
):
    problem, bundle, fields = model_case(source_case, model)
    result, _, _, _ = microscopic_record(problem, fields)
    reversed_result, _, _, _ = microscopic_record(problem, fields, sign=-1.0)
    direct_p, direct_j = atomic_sources(fields, result, problem, bundle)
    reversed_p, reversed_j = atomic_sources(-fields, reversed_result, problem, bundle)
    # Structural: a sign is exact through every product.
    assert_array_equal(reversed_p, -direct_p)
    assert_array_equal(reversed_j, -direct_j)


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow"])
def test_cubic_replacement_refuses_a_missing_third_order_label(source_case, model):
    problem, bundle, fields = model_case(source_case, model)
    result, _, _, _ = microscopic_record(problem, fields)
    missing = dataclasses.replace(
        result, shadow_orders=(1,), third_order_current=None, third_order_dipole=None
    )
    with pytest.raises(BackendError, match="third-order current"):
        atomic_sources(fields, missing, problem, bundle)


def test_shadow_closure_refuses_an_absent_charge_dipole(source_case):
    problem, bundle, fields = model_case(source_case, "tdse_shadow")
    result, _, _, _ = microscopic_record(problem, fields)
    missing = dataclasses.replace(result, third_order_dipole=None)
    with pytest.raises(BackendError, match="third-order charge dipole"):
        atomic_sources(fields, missing, problem, bundle)


@pytest.mark.parametrize("identity", ["input_identity", "configuration_identity"])
def test_response_identity_must_match_the_exact_field_and_configuration(
    source_case, identity
):
    problem, bundle, fields = model_case(source_case, "tdse_shadow")
    result, _, _, _ = microscopic_record(problem, fields)
    wrong = dataclasses.replace(result, **{identity: "a different request"})
    with pytest.raises(BackendError, match="identify this configuration and field"):
        atomic_sources(fields, wrong, problem, bundle)
    changed = fields.copy()
    changed[0, 0] += 1.0
    with pytest.raises(BackendError, match="identify this configuration and field"):
        atomic_sources(changed, result, problem, bundle)


def test_atomic_sources_preserve_nonzero_endpoints_area_and_residual_current(
    source_case,
):
    problem, bundle, fields = model_case(source_case, "tdse_shadow")
    fields = np.ones_like(fields)
    result, _, remainder, dipole = microscopic_record(problem, fields)
    polarisation, current = atomic_sources(fields, result, problem, bundle)
    # Constant histories have nonzero pulse area and a physical residual at both
    # ends. A source taper or mean subtraction would destroy these invariants.
    assert np.any(problem.grids.t_window_fine < 1.0)
    assert_allclose(
        polarisation[:, 0, :], 3.0 * bundle.density * dipole, rtol=ROUNDING, atol=0
    )
    assert_allclose(
        current[:, 0, :], 2.0 * bundle.density * remainder, rtol=ROUNDING, atol=0
    )
    assert np.all(polarisation[..., [0, -1]] != 0.0)
    assert np.all(current[..., [0, -1]] != 0.0)
    assert np.all(np.sum(current, axis=-1) != 0.0)


def test_atomic_source_requires_an_atomic_descriptor(source_case):
    problem, bundle, fields = model_case(source_case, "tdse_shadow")
    result, _, _, _ = microscopic_record(problem, fields)
    problem = dataclasses.replace(problem, tdse=None)
    with pytest.raises(BackendError, match="requires a TDSE descriptor"):
        atomic_sources(fields, result, problem, bundle)


@pytest.mark.parametrize("channel", ["current", "polarisation"])
def test_atomic_source_refuses_overflow_from_finite_microscopic_values(
    source_case, channel
):
    problem, bundle, fields = model_case(source_case, "tdse_shadow")
    result, _, _, _ = microscopic_record(problem, fields)
    extreme = np.full(fields.shape, np.finfo(float).max)
    if channel == "current":
        result = dataclasses.replace(
            result,
            total_current=extreme,
            first_order_current=np.zeros_like(fields),
            third_order_current=np.zeros_like(fields),
        )
    else:
        result = dataclasses.replace(result, third_order_dipole=extreme)
    # Finiteness of each provider operand alone cannot admit its density/scaled
    # assembly. The manufactured overflow must become an explicit backend error.
    with (
        pytest.warns(RuntimeWarning, match="overflow"),
        pytest.raises(BackendError, match="produced a non-finite value"),
    ):
        atomic_sources(fields, result, problem, bundle)


@pytest.mark.parametrize("scale", [1.0, 1.537])
def test_native_source_collar_preserves_raw_currents_and_subtracts_before_gating(
    source_case, scale
):
    problem, grid, bundle, response, fields = source_case
    response = dataclasses.replace(
        response,
        kerr_model="tdse_native",
        remainder_scale=scale,
        cubic_scale=None,
    )
    problem, bundle, fields = model_case(source_case, "tdse_native")
    problem = dataclasses.replace(
        problem, tdse=TDSEDescriptor.from_response(response, grid)
    )
    result, third, remainder, _ = microscopic_record(problem, fields)
    raw = result.total_current.copy()
    first = result.first_order_current.copy()
    polarisation, current = atomic_sources(fields, result, problem, bundle)
    expected = bundle.density * scale * (third + remainder) * grid.t_window_fine
    # The manufactured subtraction is exact; gating and nondyadic scaling add
    # at most three rounded products, with unchanged exact zeros at the edges.
    assert_allclose(current[:, 0], expected, rtol=8 * np.finfo(float).eps, atol=0)
    assert_array_equal(current[:, :, 0], 0)
    assert_array_equal(polarisation, 0)
    assert_array_equal(result.total_current, raw)
    assert_array_equal(result.first_order_current, first)
    assert not result.total_current.flags.writeable
    assert not np.shares_memory(current, result.total_current)
    reverse, _, _, _ = microscopic_record(problem, fields, sign=-1.0)
    _, reversed_current = atomic_sources(-fields, reverse, problem, bundle)
    assert_array_equal(reversed_current, -current)


def test_native_source_collar_preserves_linear_cancellation(source_case):
    problem, grid, bundle, response, fields = source_case
    response = dataclasses.replace(
        response,
        kerr_model="tdse_native",
        cubic_scale=None,
    )
    problem, bundle, fields = model_case(source_case, "tdse_native")
    problem = dataclasses.replace(
        problem, tdse=TDSEDescriptor.from_response(response, grid)
    )
    first = 2.0**40 * fields
    result = TDSEResult(
        first,
        first,
        input_identity=waveform_identity(fields),
        configuration_identity=problem.tdse.identity,
        valid=np.ones(fields.shape, dtype=bool),
        shadow_orders=(1,),
    )
    polarisation, current = atomic_sources(fields, result, problem, bundle)
    assert_array_equal(polarisation, 0)
    assert_array_equal(current, 0)


# --- SEP 20 section 7.1 item 14 -------------------------------------------------------


@pytest.mark.parametrize("window", [None, "explicit"])
def test_hacc_source_is_the_sae_native_formula_with_its_gate(source_case, window):

    problem, grid, bundle, response, fields = source_case
    declared = TDSEAtomicWindow(6e-15, 1e-15) if window else None
    native = dataclasses.replace(
        response,
        kerr_model="tdse_native",
        remainder_scale=1.0,
        cubic_scale=None,
        atomic_window=declared,
    )
    hacc = hacc_response(atomic_window=declared)
    compiled = dataclasses.replace(
        problem.responses,
        kerr_coefficients=None,
        response_order=(ResponseStage.ACCUMULATE,),
    )
    results = []
    for declared_response in (native, hacc):
        case = dataclasses.replace(
            problem,
            tdse=TDSEDescriptor.from_response(declared_response, grid),
            responses=compiled,
        )
        result, _, _, _ = microscopic_record(case, fields)
        polarisation, current = atomic_sources(fields, result, case, bundle)
        # The native remainder keeps every order above the first, the cubic too.
        remainder = result.total_current - result.first_order_current
        results.append((polarisation, current, remainder, case))
    # Structural: the native polarisation is exactly zero on both; the currents are
    # one formula evaluated twice, to the density's few roundings.
    assert_array_equal(results[0][0], results[1][0])
    assert_allclose(results[0][1], results[1][1], rtol=ROUNDING, atol=0)
    polarisation, current, remainder, case = results[1]
    assert case.tdse is not None
    gate = case.tdse.source_gate()
    expected_gate = grid.t_window_fine if gate is None else gate
    # Same formula: density times the gated remainder at scale one, P = 0.
    assert_allclose(
        current[:, 0, :],
        bundle.density * (remainder * expected_gate),
        rtol=ROUNDING,
        atol=0,
    )
    assert_array_equal(polarisation, 0.0)
    if window:
        assert gate is not None and not np.array_equal(gate, grid.t_window_fine)
