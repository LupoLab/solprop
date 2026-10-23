# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check complete held-source rows, chunk boundaries and non-plasma separation."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_band_plasma_parity import CASES, case
from tests.backends.test_radial_nonlinear import derivative
from tests.bounds import gamma
from tests.driver.test_prop_capillary import build

from gated_source import _gated_source
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.stage import scalars

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_source_chunking_and_zero_after_a_nonzero_evaluation(name, rule):
    grid, problem = case(name)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, plasma_integral=design_running_integral(grid, rule)
        ),
    )
    # Each transform has at most n_nodes terms; the two current scans and
    # transforms add O(N log N) rounding. The same sampled history is being
    # projected in different chunk orders, so no time-discretisation term enters.
    relative = gamma(64 * grid.n_to + 32 * problem.transform.n_nodes)
    assert relative > 0
    reference = derivative(NumpyRefBackend, problem, grid)
    whole = dataclasses.replace(
        problem,
        transform=dataclasses.replace(
            problem.transform, node_chunk=problem.transform.n_nodes
        ),
    )
    backend = NumpyRefBackend(whole, grid=grid)
    try:
        bundle = backend._stage_scalars(0.0, 0.0)
        state = problem.initial.field0
        ones = np.ones_like(state)
        backend._hoist(0.0)
        result = backend._nonlinear(state, bundle, ones)[0]
        allowance = relative * np.max(abs(reference))
        assert allowance > 0
        assert np.max(abs(result - reference)) <= allowance
        zero = backend._nonlinear(np.zeros_like(state), bundle, ones)
        assert np.array_equal(zero, np.zeros_like(zero))
        assert np.array_equal(
            result[..., ~grid.active_mask],
            np.zeros_like(result[..., ~grid.active_mask]),
        )
    finally:
        backend.close()


@pytest.mark.parametrize("name", ["mode_averaged", "analytic", "multimode", "mixture"])
def test_removing_kerr_does_not_change_the_held_minus_gated_source(name):
    grid, problem = case(name)
    kerr = problem.responses.kerr_coefficients
    assert kerr is not None
    absent = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=dataclasses.replace(kerr, coefficient=0.0),
        ),
    )

    # Linear source separation: cancellation can hide the plasma difference, so
    # scale by the four original operands, never by the difference under test.
    relative = gamma(64 * grid.n_to + 32 * problem.transform.n_nodes)
    results = []
    for value in (problem, absent):
        held = derivative(NumpyRefBackend, value, grid)
        with _gated_source():
            gated = derivative(NumpyRefBackend, value, grid)
        results.append((held, gated))
    allowance = relative * sum(np.max(abs(item)) for pair in results for item in pair)
    assert allowance > 0
    assert (
        np.max(abs((results[0][0] - results[0][1]) - (results[1][0] - results[1][1])))
        <= allowance
    )


def test_mixture_production_assembles_both_current_channels():
    grid, problem = case("mixture")
    bundle = scalars(problem, 0.0, 0.0)
    transform, grids = problem.transform, problem.grids
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend._hoist(0.0)
        nodes = passes.synthesise(
            passes.to_time(
                problem.initial.field0[0], grids.crop_index, grids.n_to, kind="field"
            ),
            transform.synthesis,
            bundle.taper_scale,
            transform.n_pol,
        )
        sources = backend._column_sources(nodes, bundle, None)
        assert sources.current is not None
        # The compatibility polarisation entry point still integrates the physical
        # current; production consumes the separate streams instead.
        physical = passes.nonlinear_polarisation(
            nodes,
            problem.responses,
            grids.t_window_fine,
            kind="field",
            density=bundle.density,
            preionisation=bundle.preionisation,
            dt=grids.dt_fine,
            scalars=bundle,
        )
        expected_physical = (
            sources.polarisation
            + grids.t_window_fine
            * problem.responses.plasma_integral.integrate(
                sources.current, grids.dt_fine
            )
        )
        relative = gamma(64 * grids.n_to)
        assert relative > 0
        np.testing.assert_allclose(
            physical,
            expected_physical,
            rtol=0,
            atol=float(relative * np.max(abs(expected_physical))),
        )
        projected = [
            passes.project(
                values, transform.projection, bundle.taper_scale, transform.n_pol
            )
            for values in (
                sources.polarisation,
                grids.t_window_fine * sources.current,
                grids.t_window_fine_derivative * sources.current,
            )
        ]
        spectra = [
            passes.to_frequency(value, grids.crop_index, kind="field")
            for value in projected
        ]
        expected = spectra[0].copy()
        active = grids.active_mask
        q = 1j * grids.omega[active]
        expected[..., active] += (
            spectra[1][..., active] / q - spectra[2][..., active] / q**2
        )
        weight = backend._prefactor(bundle) * grids.omega_window
        expected *= weight
        actual = backend._nonlinear(
            problem.initial.field0, bundle, np.ones(problem.shape, complex)
        )[0]
        # Projection chunk order, three FFTs and assembly of identical currents.
        allowance = gamma(64 * grids.n_to + 32 * transform.n_nodes) * np.linalg.norm(
            expected
        )
        assert allowance > 0
        assert np.linalg.norm(actual - expected) <= allowance
        source_metadata = backend.metadata()["plasma_source"]
        assert isinstance(source_metadata, dict)
        assert source_metadata["form"] == "held_current"
        assert backend.memory_ledger()["held_current_streams"] > 0
    finally:
        backend.close()


@pytest.mark.parametrize("rule", ["band", "trapezoid"])
@pytest.mark.parametrize("name", ["mode_averaged", "mixture"])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("preionisation", [0.0, 0.1])
def test_real_column_decomposition_in_assembled_units(
    rule, name, components, preionisation, record_property
):
    grid, problem = case(name)
    grids, transform = problem.grids, problem.transform
    integral = design_running_integral(grid, rule)
    bundle = scalars(problem, 0.0, 0.0)
    # Optmat's real He rate (and Ar for the mixture), a few-cycle field and
    # elliptical two-component version. General modal plasma is still refused,
    # so the admitted plasma routes have one or two components, never three.
    field = 5e10 * np.exp(-((grid.t_fine / 4e-15) ** 2))
    values = [field * np.cos(grid.omega0 * grid.t_fine)]
    if components == 2:
        values.append(0.5 * field * np.sin(grid.omega0 * grid.t_fine))
    nodes = np.broadcast_to(
        np.stack(values), (transform.n_nodes, components, grid.n_to)
    )
    current = passes._plasma_channels_current(
        nodes,
        problem.responses.plasma_channels,
        bundle.plasma_densities,
        (preionisation,) * len(problem.responses.plasma_channels),
        grids.dt_fine,
        integral,
    )
    polarisation = integral.integrate(current, grids.dt_fine)
    gate, slope = grids.t_window_fine, grids.t_window_fine_derivative
    active, crop = grids.active_mask, grids.crop_index[grids.active_mask]
    q = 1j * grids.omega[active]
    # Project each component with the actual scalar modal rule; the units and
    # prefactor are those of the stage source, not an unweighted nodal norm.
    spectra = []
    for product in (
        gate * polarisation,
        gate * current,
        slope * polarisation,
        slope * current,
    ):
        spectra.append(
            np.fft.rfft(
                passes.project(product, transform.projection, bundle.taper_scale, 1)
            )[..., crop]
        )
    v, x, y, z = spectra
    a_j = x / q - z / q**2
    a_p = v - y / q - z / q**2
    defect = q * v - x - y
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend._hoist(0.0)
        weight = backend._prefactor(bundle)[..., active] * grids.omega_window[active]
    finally:
        backend.close()
    allowance = (
        32
        * np.finfo(float).eps
        * abs(weight)
        * (abs(v) + abs(x / q) + abs(y / q) + abs(z / q**2))
    )
    assert np.all(abs(weight * (a_p - a_j - defect / q)) <= allowance)
    assert np.linalg.norm(current) > 0
    record_property(
        "closure_over_held",
        float(np.linalg.norm(weight * defect / q) / np.linalg.norm(weight * a_j)),
    )


@pytest.mark.parametrize("kerr_present", [False, True])
def test_raman_amplitude_does_not_change_the_held_minus_gated_source(kerr_present):
    simulation = build(gas="H2", raman=True)
    problem, grid = simulation.translate(), simulation.grid
    coefficients = problem.responses.kerr_coefficients
    assert coefficients is not None and problem.responses.raman_terms
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=dataclasses.replace(
                coefficients,
                coefficient=coefficients.coefficient if kerr_present else 0.0,
            ),
        ),
    )
    without = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            raman_terms=tuple(
                dataclasses.replace(term, undamped=np.zeros_like(term.undamped))
                for term in problem.responses.raman_terms
            ),
        ),
    )
    results = []
    for value in (problem, without):
        held = derivative(NumpyRefBackend, value, grid)
        with _gated_source():
            gated = derivative(NumpyRefBackend, value, grid)
        results.append((held, gated))
    # Four assembled operands: cancellation in the plasma difference must not
    # reduce the rounding budget for the unchanged Raman convolution and Kerr.
    relative = gamma(64 * grid.n_to + 32 * problem.transform.n_nodes)
    allowance = relative * sum(np.max(abs(item)) for pair in results for item in pair)
    assert allowance > 0
    assert (
        np.max(abs((results[0][0] - results[0][1]) - (results[1][0] - results[1][1])))
        <= allowance
    )
