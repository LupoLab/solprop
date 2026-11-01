# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The schema-15 ``"slab"`` geometry and the separable leaf's ``kappa_sq``."""

import dataclasses
import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from solprop.boundaries import absorber
from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.linear import Frame
from solprop.modes import PlanarGrid, SlabMode
from solprop.profiles import Linear
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.stepper import PIController
from solprop.translate.planar import (
    _warn_grazing,  # pyright: ignore[reportPrivateUsage]
    planar_operators,
    planar_transform,
)
from solprop.translate.problem import assemble
from solprop.translate.spatial import (
    SeparableOperator,
    SpatialGeometry,
    separable_values,
)

EPS = np.finfo(float).eps
HALF_GAP = 63.5e-6
GRID = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 400e-15)

pytestmark = pytest.mark.filterwarnings(
    # The FFT padding reaches beyond the argon fit; the notice is not the subject.
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


@pytest.fixture(scope="module")
def space():
    modes = [SlabMode(HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3)]
    return PlanarGrid(modes, 1e-3, 16, collar_fraction=0.25)


@pytest.fixture(scope="module")
def problem(space):
    return assemble(
        GRID,
        None,
        [],
        np.zeros((space.n_columns, 1, GRID.n_omega), complex),
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=0.01,
        spatial_grid=space,
        medium=SpatialMedium("Ar", 1.0),
        absorber=absorber(GRID, boundary_length=1.0),
    )


def test_the_geometry_is_the_slab_kind_with_columns_and_nodes_apart(space, problem):
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "slab"
    assert geometry.shape == (space.x.size, 16)
    assert geometry.n_columns == space.n_columns == problem.transform.n_modes
    assert problem.transform.n_nodes == space.n_nodes
    assert problem.transform.synthesis.shape == (space.x.size, 2)
    assert_array_equal(problem.transform.synthesis, space.synthesis * GRID.pad_scale)
    (k_y,) = geometry.spectral_axes
    assert_array_equal(k_y, space.k_y)
    # The column weight is the raw DFT's dy/n_y over eps0 c/2: the one energy formula.
    assert_allclose(
        geometry.spectral_weights, 2 * space.dy / (EPS0 * C * 16), rtol=1e-15
    )
    assert_allclose(
        geometry.spatial_weights, np.tile(space.x_weights, 16) * space.dy, rtol=1e-15
    )
    # The y collar alone, on every x node of a row, and the rate it sets.
    rows = geometry.window.reshape(16, -1)
    assert_array_equal(rows, np.repeat(space.window_y[:, None], space.x.size, axis=1))
    assert np.any(rows < 1)
    assert np.all(geometry.absorber_rate[geometry.window == 1] == 0)
    assert np.all(geometry.absorber_rate[geometry.window < 1] > 0)
    assert geometry.inverse_scale == 1 / 16
    assert problem.stepper.propagator_mode == "auto"


def test_the_operator_at_ky_zero_is_each_slab_mode_to_its_rounding(space, problem):
    """Re sqrt(k^2 n_co^2 - kappa^2) against omega Re n_eff / c: a few ulps of beta."""
    separable = problem.linear.separable
    assert separable is not None
    assert separable.kappa_sq is not None
    assert separable.n_channels == 2
    operator, prefactor = separable_values(separable)
    active = GRID.active_mask
    omega = GRID.omega[active]
    for channel, mode in enumerate(space.modes):
        beta = omega * mode.neff(omega) / C
        expected = (
            -beta.imag
            - 1j * (beta.real - separable.frame[active])
            - 0.5 * separable.spectral_rate[active]
        )
        assert_allclose(
            operator[0, channel, 0, active],
            expected,
            rtol=0,
            atol=8 * EPS * float(np.max(np.abs(beta))),
        )
        assert_allclose(
            prefactor[channel, 0, active],
            -0.5j * omega / (EPS0 * C * np.real(mode.neff(omega))) * GRID.crop_scale,
            rtol=16 * EPS,
        )
    # A column selection evaluates what the whole evaluation does, elementwise on
    # fewer columns; a vector loop may round its tail differently, so it is held to
    # a few roundings (4 eps) of the largest value.
    rows = slice(5, 19)
    part_operator, part_prefactor = separable_values(separable, rows)
    assert_allclose(
        part_operator[0],
        operator[0, rows],
        rtol=0,
        atol=4 * EPS * float(np.max(np.abs(operator))),
    )
    assert_allclose(
        part_prefactor,
        prefactor[rows],
        rtol=0,
        atol=4 * EPS * float(np.max(np.abs(prefactor))),
    )
    # The frame is the first channel's at k_y = 0, as the modal route's.
    assert separable.frame[0] == pytest.approx(
        float(GRID.omega[0] * space.modes[0].neff(GRID.omega[:1])[0].real / C),
        rel=1e-15,
    )


def test_a_high_ky_column_is_evanescent_and_clamped(space, problem):
    separable = problem.linear.separable
    assert separable is not None
    wide = PlanarGrid(space.modes, 1e-7, 4, collar_fraction=0.0)
    medium = SpatialMedium("Ar", 1.0)
    dispersion = SpatialDispersion(medium, GRID, Frame.comoving(GRID))
    _, data = planar_operators(wide, dispersion, n_pol=1, z_max=0.1)
    operator, prefactor = separable_values(data)
    # k_y = pi/dy far beyond every beta: the evanescent cap and the literal clamp.
    column = 2 * 2
    active = GRID.active_mask
    assert_allclose(np.real(operator[0, column, 0, active]), -data.evanescent_rate)
    assert_array_equal(prefactor[column, 0, active], data.prefactor_clamped[active])


def test_the_geometry_refuses_slab_arrays_of_the_wrong_meaning(problem):
    geometry = problem.transform.geometry
    assert geometry is not None
    window = geometry.window.copy()
    centre = geometry.shape[0] * (geometry.shape[1] // 2)
    assert window[centre] == 1
    window[centre + 1] = 0.5
    with pytest.raises(TranslationError, match="slab window must depend on y alone"):
        dataclasses.replace(geometry, window=window)
    with pytest.raises(TranslationError, match="a multiple of n_y"):
        dataclasses.replace(geometry, spectral_weights=geometry.spectral_weights[:-1])
    with pytest.raises(TranslationError, match="on_axis and spectral_weights"):
        dataclasses.replace(geometry, on_axis=geometry.on_axis[:-16])
    with pytest.raises(TranslationError, match=r"radial .*free .*slab"):
        dataclasses.replace(geometry, shape=(geometry.shape[0],))


def test_the_leaf_refuses_a_malformed_wall_term(problem):
    separable = problem.linear.separable
    assert separable is not None
    assert separable.kappa_sq is not None
    with pytest.raises(TranslationError, match="kappa_sq must be"):
        dataclasses.replace(separable, kappa_sq=separable.kappa_sq[:, :-1])
    with pytest.raises(TranslationError, match="n_channels dividing"):
        dataclasses.replace(
            separable, kappa_sq=np.zeros((5, separable.k_squared.size), complex)
        )
    inactive = int(np.flatnonzero(~separable.active)[0])
    spoiled = separable.kappa_sq.copy()
    spoiled[0, inactive] = 1.0
    with pytest.raises(TranslationError, match="vanish at inactive"):
        dataclasses.replace(separable, kappa_sq=spoiled)


def test_the_problem_refuses_a_slab_geometry_without_its_own_operator(problem):
    separable = problem.linear.separable
    assert separable is not None
    spoiled = dataclasses.replace(separable, kappa_sq=None)
    with pytest.raises(TranslationError, match="requires separable kappa_sq"):
        dataclasses.replace(
            problem,
            linear=dataclasses.replace(problem.linear, separable=spoiled),
            responses=dataclasses.replace(problem.responses, separable=spoiled),
        ).validate()
    spoiled = dataclasses.replace(separable, kperp_sq=separable.kperp_sq[::-1].copy())
    with pytest.raises(TranslationError, match="k_y\\*\\*2 per column"):
        dataclasses.replace(
            problem,
            linear=dataclasses.replace(problem.linear, separable=spoiled),
            responses=dataclasses.replace(problem.responses, separable=spoiled),
        ).validate()
    with pytest.raises(TranslationError, match="slab propagation refuses"):
        dataclasses.replace(
            problem,
            stepper=dataclasses.replace(problem.stepper, propagator_mode="tabulate"),
        ).validate()


def test_a_free_geometry_refuses_a_wall_term():
    space = FreeGrid(1e-3, 4, 1e-3, 4, collar_fraction=0.0)
    free = assemble(
        GRID,
        None,
        [],
        np.zeros((16, 1, GRID.n_omega), complex),
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=0.01,
        spatial_grid=space,
        medium=SpatialMedium(),
    )
    separable = free.linear.separable
    assert separable is not None
    assert separable.kappa_sq is None
    spoiled = dataclasses.replace(
        separable, kappa_sq=np.zeros((1, separable.k_squared.size), complex)
    )
    with pytest.raises(TranslationError, match="belongs to the slab geometry alone"):
        dataclasses.replace(
            free,
            linear=dataclasses.replace(free.linear, separable=spoiled),
            responses=dataclasses.replace(free.responses, separable=spoiled),
        ).validate()


def test_the_translation_refuses_two_components_and_a_gas_path(space):
    with pytest.raises(TranslationError, match="scalar state, n_pol = 1"):
        planar_transform(space, GRID, n_pol=2, z_max=0.1)
    with pytest.raises(TranslationError, match="positive finite span"):
        planar_transform(space, GRID, n_pol=1, z_max=-1.0)
    with pytest.raises(TranslationError, match="cannot hold one complete"):
        planar_transform(space, GRID, n_pol=1, z_max=0.1, node_budget=16)
    medium = SpatialMedium("Ar", Linear(1.0, 0.5, 0.1))
    dispersion = SpatialDispersion(medium, GRID, Frame.comoving(GRID))
    with pytest.raises(TranslationError, match="requires constant media"):
        planar_operators(space, dispersion, n_pol=1, z_max=0.1)


def test_a_column_at_its_cutoff_is_reported(space, problem):
    """A wall term placing the cutoff on a k_y sample raises the grazing warning."""
    separable = problem.linear.separable
    assert separable is not None
    assert separable.kappa_sq is not None
    kappa = np.zeros_like(separable.kappa_sq)
    bin_index = int(np.flatnonzero(separable.active)[0])
    # Re(k^2 eps - kappa^2) = k_y[1]^2 exactly, the residuals aside.
    kappa[0, bin_index] = separable.k_squared[bin_index] - space.k_y[1] ** 2
    grazing = dataclasses.replace(separable, kappa_sq=kappa)
    with pytest.warns(GrazingIncidenceWarning, match="planar grazing-incidence"):
        _warn_grazing(space, GRID, grazing)
    # The fixture's own coefficients are nowhere near a cutoff.
    _warn_grazing(space, GRID, separable)
    assert math.isfinite(float(np.max(np.abs(separable.kappa_sq))))


def test_a_column_near_its_cutoff_but_resolved_is_not_reported(space, problem):
    """A k_y sample inside the search radius whose distance the arithmetic resolves."""
    separable = problem.linear.separable
    assert separable is not None
    assert separable.kappa_sq is not None
    bin_index = int(np.flatnonzero(separable.active)[0])
    material = separable.k_squared[bin_index]
    wall = material - space.k_y[1] ** 2
    # Half the search radius, 64 eps (|k^2| + |kappa^2|), away from the sample: inside
    # the candidate window, but far outside the few-rounding-unit uncertainty.
    offset = 32 * np.finfo(float).eps * (abs(material) + abs(wall))
    kappa = np.zeros_like(separable.kappa_sq)
    kappa[0, bin_index] = wall - offset
    near = dataclasses.replace(separable, kappa_sq=kappa)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GrazingIncidenceWarning)
        _warn_grazing(space, GRID, near)


def test_a_malformed_slab_transform_is_refused(problem):
    transform = problem.transform
    with pytest.raises(TranslationError, match="slab synthesis must have shape"):
        dataclasses.replace(
            transform,
            synthesis=transform.synthesis[:, :1],
            projection=transform.projection[:1],
            projection_coarse=transform.projection_coarse[:1],
        )
    assert isinstance(transform.geometry, SpatialGeometry)
    assert isinstance(problem.linear.separable, SeparableOperator)
