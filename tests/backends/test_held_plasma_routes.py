# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check complete held-source rows, chunk boundaries and non-plasma separation."""

import dataclasses

import numpy as np
import pytest

from backends.plasma_column_cases import CASES, case, options_for
from backends.test_radial_nonlinear import derivative
from bounds import gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes.plasma import plasma_channels_current
from solprop.backends.numpy_ref.passes.responses import nonlinear_polarisation
from solprop.backends.numpy_ref.passes.transforms import (
    project,
    synthesise,
    to_frequency,
    to_time,
)
from solprop.backends.session import ExecutionOptions
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
    grid, problem, run_request = case(name)
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
    reference = derivative(
        NumpyRefBackend, problem, run_request, grid, options_for(name)
    )
    backend = NumpyRefBackend(
        problem,
        run_request,
        grid=grid,
        options=ExecutionOptions(node_chunk=problem.transform.n_nodes),
    )
    try:
        bundle = backend.workspace.stage_scalars(0.0, 0.0)
        state = problem.initial.field0
        ones = np.ones_like(state)
        backend.workspace.hoist(0.0)
        result = backend.workspace.nonlinear(state, bundle, ones)
        allowance = relative * np.max(abs(reference))
        assert allowance > 0
        assert np.max(abs(result - reference)) <= allowance
        zero = backend.workspace.nonlinear(np.zeros_like(state), bundle, ones)
        assert np.array_equal(zero, np.zeros_like(zero))
        assert np.array_equal(
            result[..., ~grid.active_mask],
            np.zeros_like(result[..., ~grid.active_mask]),
        )
    finally:
        backend.close()


def test_mixture_production_assembles_both_current_channels():
    grid, problem, run_request = case("mixture")
    bundle = scalars(problem, 0.0, 0.0)
    transform, grids = problem.transform, problem.grids
    backend = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        backend.workspace.hoist(0.0)
        nodes = synthesise(
            to_time(problem.initial.field0, grids.crop_index, grids.n_to, kind="field"),
            transform.synthesis,
            bundle.taper_scale,
            transform.n_pol,
        )
        sources = backend.workspace.stage_source._column_sources(nodes, bundle, None)
        assert sources.current is not None
        # The compatibility polarisation entry point still integrates the physical
        # current; production consumes the separate streams instead.
        physical = nonlinear_polarisation(
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
            project(values, transform.projection, bundle.taper_scale, transform.n_pol)
            for values in (
                sources.polarisation,
                grids.t_window_fine * sources.current,
                grids.t_window_fine_derivative * sources.current,
            )
        ]
        spectra = [
            to_frequency(value, grids.crop_index, kind="field") for value in projected
        ]
        expected = spectra[0].copy()
        active = grids.active_mask
        q = 1j * grids.omega[active]
        expected[..., active] += (
            spectra[1][..., active] / q - spectra[2][..., active] / q**2
        )
        weight = backend.workspace.prefactor(bundle) * grids.omega_window
        expected *= weight
        actual = backend.workspace.nonlinear(
            problem.initial.field0, bundle, np.ones(problem.shape, complex)
        )
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
    grid, problem, run_request = case(name)
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
    current = plasma_channels_current(
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
            np.fft.rfft(project(product, transform.projection, bundle.taper_scale, 1))[
                ..., crop
            ]
        )
    v, x, y, z = spectra
    a_j = x / q - z / q**2
    a_p = v - y / q - z / q**2
    defect = q * v - x - y
    backend = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        backend.workspace.hoist(0.0)
        weight = (
            backend.workspace.prefactor(bundle)[..., active]
            * grids.omega_window[active]
        )
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
