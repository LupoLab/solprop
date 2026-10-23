# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify Cartesian Raman convolution, complete runs and modal representations."""

import math
from dataclasses import fields, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import hilbert

from backends.test_numba_cpu_kernels import run_column
from bounds import gamma, ladder_bound
from former_raman import FORMER_H2, former_response
from solprop.backends.numba_cpu import fft, kernels
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.constants import EPS0
from solprop.driver.simulation import _undamped_model_key
from solprop.exceptions import BackendError, TranslationError
from solprop.profiles import Constant
from solprop.translate.problem import _raman_chunk
from solprop.translate.responses import RamanCoefficients, raman_coefficients
from solprop.translate.stage import scalars
from test_vector_raman_qualification import gain_experiment
from vector_raman_cases import raman_response, result, vector_problem
from vector_raman_tensor_oracle import BUDGETS, cartesian_direct, convolution_bound


def test_reference_pass_rejects_three_physical_components():
    with pytest.raises(BackendError, match="one or two physical components"):
        passes.raman_polarisation(
            np.zeros((1, 3, 16)),
            np.zeros(17, complex),
            kind="field",
            thg=True,
            density=0.0,
        )


@pytest.mark.parametrize("count", [18, 30])
@pytest.mark.parametrize("material", ["H2", "SF6"])
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_fftw_tensor_chain_matches_causal_sums_at_non_power_of_two_lengths(
    count, material, family
):
    problem, _, _ = vector_problem(family, material)
    spec = kernels.column_spec(problem.responses, problem.grid_kind, 2, parallel=False)
    rng = np.random.default_rng(4048)
    values = rng.standard_normal((2, 2, count))
    if family == "envelope":
        values = values + 1j * rng.standard_normal(values.shape)
    analytic = (
        hilbert(np.real(values), axis=-1) if family == "field_analytic" else values
    )
    nodes = np.ascontiguousarray(analytic.reshape(4, count))
    plans = fft.build_plans(
        family, 4, count, count, flag="estimate", chunk_columns=2 * spec.raman_drives
    )
    window = np.linspace(0.7, 1.0, count)
    run_column(spec, problem.responses, nodes, window, drive=plans.drive)
    dt = 0.07
    delay = np.arange(count) * dt
    hv = 0.7 * np.sin(2.3 * delay) * np.exp(-0.17 * delay)
    hr = 1.1 * np.sin(0.9 * delay) if material == "H2" else np.zeros(count)
    effective = [hv + hr / 4]
    if spec.raman_drives == 3:
        effective.append(3 * hr / 4)
    spectrum = np.fft.rfft(np.array(effective), n=2 * count, axis=-1) * dt
    plans.raman_to_frequency.execute()
    kernels.scale_tensor_spectrum(plans.drive_spectrum, spectrum, 1.0 / (2 * count))
    plans.raman_to_time.execute()
    result = np.zeros((4, count), complex if family == "envelope" else float)
    kernels.make_raman_finish(spec)(
        result, nodes, plans.convolved[:, :count], window, EPS0
    )
    for node in range(2):
        expected = cartesian_direct(
            analytic[node], hv, hr, dt, envelope=family != "field_real"
        )
        if family == "field_analytic":
            # The analytic drive produces a real response matrix. Its real final
            # contraction is the required multiplication by the real carrier.
            expected = expected.real
        bound = EPS0 * convolution_bound(
            analytic[node], hv, hr, dt, envelope=family != "field_real", fft=True
        )
        assert bound > 0
        assert_allclose(
            result.reshape(2, 2, count)[node],
            EPS0 * expected * window,
            rtol=0,
            atol=bound,
        )


@pytest.mark.parametrize("steps", [1, 8])
@pytest.mark.parametrize("material", ["H2", "SF6"])
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
@pytest.mark.medium
def test_vector_complete_run_matches_reference_and_modal_routes(
    material, family, steps, backend_class
):
    problem, grid, modes = vector_problem(family, material, steps=steps)
    expected = result(NumpyRefBackend, problem, grid, modes)
    actual = result(backend_class, problem, grid, modes)
    # Per RHS: three convolution FFT pairs, synthesis/projection dot products and
    # tensor assembly. Eight steps use seven stages; two runs contribute errors.
    per_rhs = 3 * 512 * np.finfo(float).eps * math.ceil(
        math.log2(2 * grid.n_to)
    ) + 8 * gamma(problem.transform.n_nodes)
    for got, reference, evaluations in (
        (actual[0], expected[0], 1),
        (actual[1], expected[1], 7 * steps),
    ):
        scale = float(np.max(abs(reference)))
        bound = ladder_bound(
            backend_class.name, 2 * evaluations * per_rhs, grid.n_to, 3 * evaluations
        )
        assert scale > 0 and bound > 0
        assert_allclose(got, reference, rtol=0, atol=bound * scale)
    for name, value in expected[2].items():
        # Statistics add a temporal reduction and a squared-modulus operation.
        bound = 112 * per_rhs + 8 * gamma(grid.n_to * problem.transform.n_nodes)
        assert_allclose(actual[2][name], value, rtol=bound, atol=0)
    row_problem, row_grid, row_modes = vector_problem(
        family, material, route="rows", steps=steps
    )
    rows = result(NumpyRefBackend, row_problem, row_grid, row_modes)
    assert_allclose(
        rows[1], expected[1], rtol=0, atol=112 * per_rhs * np.max(abs(expected[1]))
    )
    drives = 1 if material == "SF6" else 3
    ledger = actual[3]
    assert ledger["node_raman_drive"] == 5 * drives * 2 * grid.n_to * 8
    assert ledger["node_raman_convolved"] == ledger["node_raman_drive"]
    assert ledger["node_raman_spectrum"] == 5 * drives * (grid.n_to + 1) * 16
    assert (
        ledger["raman_kernel_spectrum"]
        == (1 if drives == 1 else 2) * (grid.n_to + 1) * 16
    )


def test_schema_ten_freezes_tensor_leaf_fields():
    assert tuple(field.name for field in fields(RamanCoefficients)) == (
        "undamped",
        "decay_rates",
        "fraction",
        "thg",
        "tensor_weights",
        "tensor_capable",
        "label",
        "density",
    )


@pytest.mark.parametrize("material", ["H2", "SF6"])
def test_compiled_tensor_rows_keep_physical_labels_and_damping(material):
    problem, grid, _ = vector_problem(material=material)
    response = raman_response(material, thg=False)
    leaf = problem.responses.raman_terms[0]
    expected = [
        (0.25, 0.75) if term.contraction == "rank2" else (1.0, 0.0)
        for term in response.tensor_terms
    ]
    np.testing.assert_array_equal(leaf.tensor_weights, expected)
    assert not leaf.tensor_weights.flags.writeable
    assert leaf.tensor_capable
    scalar = raman_coefficients(
        response, None, grid, Constant(1e25), n_pol=1, span=1.0, tol=1e-12
    )
    assert scalar is not None
    np.testing.assert_array_equal(scalar.tensor_weights, leaf.tensor_weights)


@pytest.mark.parametrize("count", [17, 64, 97])
def test_effective_kernel_convolution_matches_physical_term_sums(count):
    # Distinct signed kernels, dephasings, species densities and fractions ensure
    # combining before damping or applying total density to amplitude fails.
    dt = 0.07
    delays = np.arange(count) * dt
    response = former_response(FORMER_H2)
    problem, _, _ = vector_problem()
    template = problem.responses.raman_terms[0]
    rates, densities = (0.1, 0.3, 0.0, 0.2), (0.7, 1.9)
    components = []
    hv, hr = np.zeros(count), np.zeros(count)
    for i, fraction in enumerate((0.6, 0.9)):
        rows = np.zeros((2, 2 * count))
        rows[0, :count] = np.sin((1.3 + i) * delays)
        rows[1, :count] = -0.4 * np.sin((2.1 + i) * delays)
        components.append(replace(template, undamped=rows, fraction=fraction))
        hr += densities[i] * fraction * rows[0, :count] * np.exp(-rates[2 * i] * delays)
        hv += (
            densities[i]
            * fraction
            * rows[1, :count]
            * np.exp(-rates[2 * i + 1] * delays)
        )
    rng = np.random.default_rng(4003)
    values = rng.standard_normal((3, 2, count)) + 1j * rng.standard_normal(
        (3, 2, count)
    )
    spectrum = passes.tensor_raman_spectrum(tuple(components), rates, densities, dt)
    actual = passes.raman_polarisation(
        values, spectrum, kind="envelope", thg=response.thg, density=1.0
    )
    for node in range(3):
        expected = EPS0 * cartesian_direct(values[node], hv, hr, dt, envelope=True)
        bound = EPS0 * convolution_bound(
            values[node], hv, hr, dt, envelope=True, fft=True
        )
        assert_allclose(actual[node], expected, rtol=0, atol=bound)


@pytest.mark.parametrize("spoiling", ["rows", "weights", "nan"])
def test_tensor_leaf_rejects_inconsistent_or_nonfinite_numeric_contract(spoiling):
    problem, _, _ = vector_problem()
    leaf = problem.responses.raman_terms[0]
    if spoiling == "rows":
        changes = {"tensor_weights": np.zeros((1, 2))}
    elif spoiling == "weights":
        changes = {"tensor_weights": np.full((leaf.n_terms, 2), np.nan)}
    else:
        changes = {"undamped": np.full(leaf.undamped.shape, np.nan)}
    with pytest.raises(TranslationError, match=r"tensor_weights|finite"):
        replace(leaf, **changes)


@pytest.mark.parametrize("spoiling", ["empty", "density", "rates", "scalar"])
def test_tensor_spectrum_refuses_missing_or_incapable_channels(spoiling):
    problem, grid, _ = vector_problem()
    components = problem.responses.raman_terms
    rates, densities = (0.0, 0.0), (1.0,)
    if spoiling == "empty":
        components = ()
    elif spoiling == "density":
        densities = ()
    elif spoiling == "rates":
        rates = ()
    else:
        components = (replace(components[0], tensor_capable=False),)
    with pytest.raises(BackendError, match="tensor-capable"):
        passes.tensor_raman_spectrum(components, rates, densities, grid.dt_fine)


def test_undamped_reuse_distinguishes_tensor_roles():
    response = former_response(FORMER_H2)
    changed = replace(
        response,
        tensor_terms=tuple(
            replace(term, contraction="isotropic") for term in response.tensor_terms
        ),
    )
    assert _undamped_model_key(response) != _undamped_model_key(changed)


@pytest.mark.parametrize("material, drives", [("H2", 3), ("SF6", 1)])
def test_node_admission_includes_complete_tensor_histories(material, drives):
    problem, grid, _ = vector_problem(material=material, chunk=None)
    per_node = 3 * 16 * 2 * grid.n_to + drives * (48 * grid.n_to + 16)
    transformed = _raman_chunk(
        problem.transform, problem.responses, 2, grid.n_to, 2 * per_node
    )
    assert transformed.node_chunk == 2
    with pytest.raises(TranslationError, match="complete Raman node history"):
        _raman_chunk(problem.transform, problem.responses, 2, grid.n_to, per_node - 1)


def test_vector_kernel_cache_refreshes_all_dynamic_inputs(backend_class):
    problem, grid, modes = vector_problem()
    backend = backend_class(problem, grid=grid, modes=modes)
    bundle = scalars(problem, 0.0, 0.0)
    try:
        backend._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        backend._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        assert backend._raman_transforms == 1
        rates = list(bundle.raman_rates)
        for index in range(len(rates)):
            rates[index] += 1e12
            backend._raman_spectrum(tuple(rates), bundle.raman_densities)
            assert backend._raman_transforms == 2 + index
        # Single-component amplitude is applied after convolution, so density
        # changes alone need no kernel rebuild (unlike mixtures).
        backend._raman_spectrum(tuple(rates), (2 * bundle.raman_densities[0],))
        assert backend._raman_transforms == 1 + len(rates)
    finally:
        backend.close()


@pytest.mark.slow
@pytest.mark.parametrize(
    "state", ["co-linear", "cross-linear", "co-circular", "cross-circular"]
)
def test_vector_gain_propagation_on_each_backend(state, backend_class):
    record = gain_experiment(state, backend_class.name)
    assert (
        record["sideband_relative_error"]
        < BUDGETS["propagation_numerical_relative_target"]
    )
    assert (
        record["pump_fractional_change"]
        < BUDGETS["undepleted_pump_fractional_change_max"]
    )
