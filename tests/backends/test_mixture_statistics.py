# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Species axes, transverse populations and final inventory accounting."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mixture_cases import mixture_case
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.populations import population_statistics
from solprop.exceptions import BackendError, OutputError
from solprop.grid import FreeGrid, RadialGrid
from solprop.nonlinear.responses import PlasmaResponse
from solprop.stats import StatsSpec, species_statistics
from solprop.translate.chebyshev import build
from solprop.translate.free import free_transform
from solprop.translate.problem import Flags
from solprop.translate.radial import radial_transform
from solprop.translate.responses import compile_responses


def test_densest_total_node_differs_from_each_individual_species_peak():
    density = np.array([2.0, 3.0, 4.0])
    fraction = np.array([[0.8, 0.1, 0.4], [0.1, 0.7, 0.3], [0.0, 0.0, 0.0]])
    peaks = np.array([5.0, 6.0, 0.0])
    potentials = np.array([10.0, 20.0, 0.0])
    weights = np.array([0.25, 0.5, 0.25])
    result = species_statistics(density, fraction, peaks, potentials, weights)
    # Total electron density is [1.9, 2.3, 1.7]; species are sampled together at
    # node 1, despite species 0 having its own maximum at node 0. Products and sums
    # of a few positive terms: a few roundings, whatever the order. The sampled
    # inputs themselves are copied, exactly.
    rounding = 8 * np.finfo(float).eps
    assert_allclose(
        result["species_electron_density"], density * fraction[:, 1], rtol=rounding
    )
    assert result["electron_density"] == pytest.approx(
        sum(density * fraction[:, 1]), rel=rounding
    )
    assert_array_equal(result["species_ionisation"], fraction)
    assert_array_equal(result["species_density"], density)
    assert_array_equal(result["species_peak_ionisation_rate"], peaks)
    assert result["peak_ionisation_rate"] == 6.0
    expected = np.array(
        [
            sum(w * rho * f * ip for w, f in zip(weights, row, strict=True))
            for rho, row, ip in zip(density, fraction, potentials, strict=True)
        ]
    )
    assert_allclose(result["species_ionisation_energy"], expected, rtol=rounding)


@pytest.mark.parametrize(
    "quantity", ["density", "fraction", "peak", "potential", "weight"]
)
def test_population_statistics_refuse_invalid_domains(quantity):
    arrays = [np.ones(2), np.zeros((2, 3)), np.zeros(2), np.ones(2), np.ones(3)]
    index = ["density", "fraction", "peak", "potential", "weight"].index(quantity)
    arrays[index].flat[0] = -1.0
    with pytest.raises(OutputError, match="non-negative"):
        species_statistics(*arrays)


def test_numpy_species_statistics_match_the_independent_node_scans():
    mixture, grid, _, _, problem = mixture_case()
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    names = (
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
        "electron_density",
        "peak_ionisation_rate",
    )
    values = backend.stats(StatsSpec(names=names, periodic=names[:2]))
    nodes = backend._nodes_at(backend._require_state()[0])
    rates, fractions = passes.plasma_channel_fractions(
        nodes,
        problem.responses.plasma_channels,
        tuple(float(c.preionisation(0.0)) for c in problem.responses.plasma_channels),
        grid.dt_fine,
    )
    expected = species_statistics(
        mixture.density(np.array([0.0]))[:, 0],
        fractions[:, :, -1],
        rates.max(axis=(1, 2)),
        np.array([c.ionisation_potential for c in problem.responses.plasma_channels]),
        np.array([float(problem.transform.aeff(0.0))]),
    )
    # The same scans by two routes: the plasma ladder's running-sum budget, 4 n eps.
    bound = 4 * grid.n_to * np.finfo(float).eps
    for name in names:
        assert_allclose(values[name][0], expected[name], rtol=bound, err_msg=name)
    backend.close()


@pytest.mark.parametrize("channels", [0, 1])
def test_inactive_components_keep_the_species_axis(backend_class, channels):
    mixture, grid, mode, responses, original = mixture_case()
    selected = tuple(
        response
        for response in responses
        if not isinstance(response, PlasmaResponse)
        or (channels == 1 and response.species == mixture.labels[0])
    )
    compiled = compile_responses(
        selected, grid, mode, density=mode.density, z_max=grid.length
    )
    problem = dataclasses.replace(
        original,
        responses=compiled,
        flags=Flags.from_content(
            shape=original.shape,
            linear=original.linear,
            transform=original.transform,
            responses=compiled,
            initial=original.initial,
        ),
    )
    names = (
        "species_ionisation",
        "species_electron_density",
        "species_density",
        "species_ionisation_energy",
        "species_peak_ionisation_rate",
    )
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(StatsSpec(names=names))
        assert values["species_ionisation"].shape == (1, 2, 1)
        # Structural: an inactive component's statistics are exactly zero.
        assert_array_equal(values["species_ionisation"][:, channels:], 0.0)
        assert_array_equal(values["species_ionisation_energy"][:, channels:], 0.0)
        if channels:
            totals = backend.stats(
                StatsSpec(names=("electron_density", "peak_ionisation_rate"))
            )
            # A second statistics evaluation of the same state: the plasma ladder's
            # running-sum budget, 4 n eps.
            bound = 4 * grid.n_to * np.finfo(float).eps
            assert_allclose(
                totals["electron_density"],
                values["species_electron_density"].sum(axis=1),
                rtol=bound,
            )
            assert_allclose(
                totals["peak_ionisation_rate"],
                values["species_peak_ionisation_rate"].max(axis=1),
                rtol=bound,
            )
    finally:
        backend.close()


@pytest.mark.parametrize("taper", [None, 2.0])
def test_resolved_species_energy_uses_physical_weights_and_taper(taper):

    _, grid, _, _, original = mixture_case()
    weights = np.array([1e-12, 2e-12, 4e-12])
    transform = dataclasses.replace(
        original.transform,
        synthesis=np.ones((3, 1)),
        projection=np.ones((1, 3)),
        projection_coarse=np.ones((1, 3)),
        node_weights=weights,
        kind="modal",
        taper_scale=None
        if taper is None
        else build(lambda z: np.full(z.shape, taper), 0.0, grid.length),
    )
    problem = dataclasses.replace(original, transform=transform)
    fractions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    actual = population_statistics(problem, 0.0, fractions, np.zeros_like(fractions))
    density = np.array([1e24, 2e24])
    potential = np.array([1e-18, 2e-18])
    expected = (
        np.sum(fractions * weights, axis=1)
        * density
        * potential
        / (1.0 if taper is None else taper**2)
    )
    # Three weighted additions and three products per species.
    assert_allclose(
        actual["species_ionisation_energy"],
        expected,
        rtol=12 * np.finfo(float).eps,
        atol=0.0,
    )


@pytest.mark.parametrize("geometry", ["radial", "free"])
def test_spatial_species_energy_uses_the_geometry_measure(geometry):

    _, grid, _, _, original = mixture_case()
    transform = (
        radial_transform(RadialGrid(1e-3, 8), grid, n_pol=1, z_max=grid.length)
        if geometry == "radial"
        else free_transform(
            FreeGrid(1e-3, 4, 1e-3, 4), grid, n_pol=1, z_max=grid.length
        )
    )
    problem = dataclasses.replace(original, transform=transform)
    fractions = np.full((2, transform.n_nodes), 0.25)
    actual = population_statistics(problem, 0.0, fractions, np.zeros_like(fractions))
    assert transform.geometry is not None
    measure = np.sum(transform.geometry.spatial_weights)
    expected = measure * np.array([1e24 * 1e-18, 2e24 * 2e-18]) * 0.25
    # Sum of n_nodes non-negative quadrature weights and four products.
    assert_allclose(
        actual["species_ionisation_energy"],
        expected,
        rtol=(transform.n_nodes + 8) * np.finfo(float).eps,
        atol=0.0,
    )


def test_population_mapping_refuses_wrong_channel_extents_and_missing_measure():

    _, _, _, _, original = mixture_case()
    with pytest.raises(BackendError, match="channel arrays of shape"):
        population_statistics(original, 0.0, np.zeros((1, 1)), np.zeros((1, 1)))
    transform = dataclasses.replace(
        original.transform,
        synthesis=np.ones((3, 1)),
        projection=np.ones((1, 3)),
        projection_coarse=np.ones((1, 3)),
        kind="modal",
    )
    with pytest.raises(BackendError, match="physical transverse node weights"):
        population_statistics(
            dataclasses.replace(original, transform=transform),
            0.0,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
        )


@pytest.mark.parametrize("quantity", [0, 2, 3, 4])
def test_species_statistics_refuse_mismatched_extents(quantity):
    arrays = [np.ones(2), np.full((2, 3), 0.1), np.ones(2), np.ones(2), np.ones(3)]
    arrays[quantity] = arrays[quantity][:-1]
    with pytest.raises(OutputError, match="matching species and node extents"):
        species_statistics(*arrays)
