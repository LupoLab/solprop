# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Freeze and validate schema-5 geometry and separable material records."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hankel_oracles import gamma
from solprop.boundaries import STABILITY_RADIUS
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, LineGrid, RadialGrid
from solprop.linear import Frame
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.stepper import PIController
from solprop.translate.free import free_transform
from solprop.translate.linop import constant_phi, delta_phi, tabulate, unitary_phase
from solprop.translate.problem import assemble
from solprop.translate.radial import radial_transform
from solprop.translate.responses import compile_responses, prefactor_at
from solprop.translate.spatial import (
    SeparableOperator,
    SpatialGeometry,
    separable_values,
)
from solprop.translate.stage import terms


def test_spatial_schema_fifteen_freezes_complete_field_names():
    # Schema 5 froze the geometry and separable fields; schema 15 appended the slab
    # channels' wall term, kappa_sq, and schema 31 the transverse collar's rate.
    assert tuple(one.name for one in dataclasses.fields(SpatialGeometry)) == (
        "kind",
        "shape",
        "axes",
        "spectral_axes",
        "spatial_weights",
        "spectral_weights",
        "coordinates",
        "window",
        "absorber_rate",
        "on_axis",
        "forward_scale",
        "inverse_scale",
    )
    assert tuple(one.name for one in dataclasses.fields(SeparableOperator)) == (
        "k_squared",
        "kperp_sq",
        "k_squared_low",
        "kperp_sq_low",
        "frame",
        "spectral_rate",
        "prefactor_numerator",
        "prefactor_clamped",
        "active",
        "evanescent_rate",
        "n_pol",
        "z_span",
        "kappa_sq",
        "kperp_rate",
    )


@pytest.fixture(params=["radial", "free"])
def geometry(request):
    if request.param == "radial":
        space = RadialGrid(1e-3, 8)
        grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
        return radial_transform(space, grid, n_pol=1, z_max=grid.length).geometry
    return free_geometry()


def free_geometry():
    """Build unequal axes with the explicit schema-4 FFT convention."""
    x, y = np.arange(-2, 2, dtype=float) * 1e-5, np.arange(-3, 3, dtype=float) * 2e-5
    size = x.size * y.size
    area = 2e-10
    return SpatialGeometry(
        kind="free",
        shape=(x.size, y.size),
        axes=(x, y),
        spectral_axes=(
            2 * np.pi * np.fft.fftfreq(x.size, 1e-5),
            2 * np.pi * np.fft.fftfreq(y.size, 2e-5),
        ),
        spatial_weights=np.full(size, area),
        spectral_weights=np.full(size, area / size),
        coordinates=np.column_stack((np.tile(x, y.size), np.repeat(y, x.size))),
        window=np.ones(size),
        absorber_rate=np.zeros(size),
        on_axis=np.full(size, 1 / size, dtype=np.complex128),
        forward_scale=1.0,
        inverse_scale=1 / size,
    )


def test_geometry_owns_immutable_arrays(geometry):
    values = geometry.window.copy()
    clone = dataclasses.replace(geometry, window=values)
    values[:] = 0
    assert (
        np.all(clone.window == 1)
        if geometry.kind == "free"
        else np.array_equal(clone.window, geometry.window)
    )
    arrays = (
        *clone.axes,
        *clone.spectral_axes,
        clone.spatial_weights,
        clone.spectral_weights,
        clone.coordinates,
        clone.window,
        clone.absorber_rate,
        clone.on_axis,
    )
    assert all(not one.flags.writeable and one.flags.c_contiguous for one in arrays)


@pytest.mark.parametrize(
    "name",
    ["spatial_weights", "spectral_weights", "window", "absorber_rate", "on_axis"],
)
def test_geometry_rejects_array_shape_disagreement(geometry, name):
    values = getattr(geometry, name)[:-1]
    with pytest.raises(TranslationError, match=name):
        dataclasses.replace(geometry, **{name: values})


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("spatial_weights", 0.0, "positive"),
        ("spectral_weights", -1.0, "positive"),
        ("window", 1.1, "window"),
        ("absorber_rate", -1.0, "absorber_rate"),
        ("window", np.nan, "finite"),
    ],
)
def test_geometry_rejects_invalid_measures_or_collar(geometry, name, value, match):
    values = getattr(geometry, name).copy()
    values[0] = value
    with pytest.raises(TranslationError, match=match):
        dataclasses.replace(geometry, **{name: values})


def test_geometry_rejects_axis_and_coordinate_disagreement(geometry):
    with pytest.raises(TranslationError, match="kind and shape"):
        dataclasses.replace(geometry, shape=())
    with pytest.raises(TranslationError, match="integers of at least 2"):
        dataclasses.replace(geometry, shape=(1, *geometry.shape[1:]))
    with pytest.raises(TranslationError, match="coordinates"):
        dataclasses.replace(geometry, coordinates=geometry.coordinates[::-1])
    with pytest.raises(TranslationError, match="increasing"):
        dataclasses.replace(geometry, axes=(geometry.axes[0][::-1], *geometry.axes[1:]))
    with pytest.raises(TranslationError, match="spectral_axes"):
        dataclasses.replace(geometry, spectral_axes=())
    with pytest.raises(TranslationError, match="scalings"):
        dataclasses.replace(geometry, inverse_scale=2 * geometry.inverse_scale)
    with pytest.raises(TranslationError, match="scalings"):
        dataclasses.replace(geometry, forward_scale=np.inf)
    with pytest.raises(TranslationError, match="finite and positive"):
        dataclasses.replace(geometry, forward_scale=np.nan)
    with pytest.raises(TranslationError, match="finite and positive"):
        dataclasses.replace(
            geometry,
            forward_scale=-geometry.forward_scale,
            inverse_scale=-geometry.inverse_scale,
        )
    if geometry.kind == "free":
        with pytest.raises(TranslationError, match="FFT order"):
            dataclasses.replace(
                geometry,
                spectral_axes=(
                    np.fft.fftshift(geometry.spectral_axes[0]),
                    geometry.spectral_axes[1],
                ),
            )
        x = geometry.axes[0].copy()
        x[1] += 1e-6
        with pytest.raises(TranslationError, match="uniformly"):
            dataclasses.replace(geometry, axes=(x, geometry.axes[1]))
    else:
        with pytest.raises(TranslationError, match="positive and increasing"):
            dataclasses.replace(geometry, spectral_axes=(-geometry.spectral_axes[0],))


@pytest.fixture
def coefficients():
    return SeparableOperator(
        k_squared=np.array([25 + 0j, 24 + 10j, 24 - 10j, 0j]),
        kperp_sq=np.array([0.0, 25.0, 1e6]),
        k_squared_low=np.zeros(4, complex),
        kperp_sq_low=np.zeros(3),
        frame=np.ones(4),
        spectral_rate=np.arange(4, dtype=float),
        prefactor_numerator=np.full(4, -2j),
        prefactor_clamped=np.full(4, -3j),
        active=np.array([True, True, True, False]),
        evanescent_rate=200.0,
        n_pol=2,
        z_span=(0.0, 1.0),
    )


def test_separable_values_cover_propagating_cutoff_loss_and_gain(coefficients):
    # k=(5,+/-i) squares, plus a vacuum branch. The row at q^2=25 is
    # evanescent by the documented real-square classification, including equality.
    q = coefficients.k_squared[None] - coefficients.kperp_sq[:, None]
    root = np.sqrt(q)
    propagating = q.real > 0
    loss = np.where(propagating, root.imag, np.minimum(np.abs(root.imag), 200))
    expected = (
        -loss - 1j * (root.real - 1)
    ) * coefficients.active - coefficients.spectral_rate / 2
    prefactor = np.full(q.shape, -3j)
    for i, j in zip(*np.nonzero(propagating), strict=True):
        prefactor[i, j] = -2j / root[i, j].real
    prefactor *= coefficients.active
    actual, measured_prefactor = separable_values(coefficients)
    assert actual.shape == coefficients.value_shape == (1, 3, 2, 4)
    assert_allclose(actual[0, :, 0], expected, rtol=gamma(16), atol=0)
    assert_allclose(measured_prefactor[:, 0], prefactor, rtol=gamma(16), atol=0)
    np.testing.assert_array_equal(actual[0, :, 0], actual[0, :, 1])


@pytest.mark.parametrize(
    "name",
    [
        "k_squared",
        "kperp_sq",
        "k_squared_low",
        "kperp_sq_low",
        "frame",
        "spectral_rate",
        "prefactor_numerator",
        "prefactor_clamped",
        "active",
    ],
)
def test_separable_coefficients_own_their_arrays(coefficients, name):
    original = getattr(coefficients, name)
    values = original.copy()
    clone = dataclasses.replace(coefficients, **{name: values})
    assert not np.shares_memory(getattr(clone, name), values)
    assert not getattr(clone, name).flags.writeable
    with pytest.raises(TranslationError, match="nonempty"):
        dataclasses.replace(coefficients, **{name: values[:0]})


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("kperp_sq", np.array([-1.0, 25.0, 1e6]), "nonnegative"),
        ("frame", np.ones(2), "frequency count"),
        ("kperp_sq_low", np.zeros(2), "transverse count"),
        ("k_squared_low", np.zeros(2, complex), "frequency count"),
        ("kperp_sq_low", np.full(3, np.inf), "finite"),
        ("k_squared_low", np.full(4, complex(np.nan)), "finite"),
        ("spectral_rate", -np.ones(4), "nonnegative"),
        ("prefactor_numerator", np.ones(4, dtype=complex), "imaginary"),
        ("evanescent_rate", 0.0, "positive"),
        ("n_pol", 3, "n_pol"),
        ("z_span", (1.0, 0.0), "increasing"),
        ("k_squared", np.full(4, complex(np.inf)), "finite"),
    ],
)
def test_separable_coefficients_refuse_invalid_values(coefficients, name, value, match):
    with pytest.raises(TranslationError, match=match):
        dataclasses.replace(coefficients, **{name: value})


def test_a_gas_medium_refuses_an_explicit_response_density():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = RadialGrid(1e-3, 8)
    gas = SpatialMedium("Ar", 2.0)

    def build(medium, density):
        return assemble(
            grid,
            None,
            [],
            np.zeros((space.n_r, 1, grid.n_omega), complex),
            spatial_grid=space,
            medium=medium,
            density=density,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[grid.length],
            dz_max=grid.length,
            dz_init=grid.length,
        )

    # The linear operator takes the medium's density; a second one would scale only
    # the responses.
    with pytest.raises(TranslationError, match=r"Ar medium carries its density"):
        build(gas, 1e25)
    # The density meant is the medium's own, for both.
    assert build(gas.at_density(1e25), None).device_ready


@pytest.fixture
def free_contract():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    geometry = free_geometry()
    space = RadialGrid(1e-3, geometry.spatial_weights.size)
    model = SpatialDispersion(SpatialMedium(), grid, Frame.comoving(grid))
    original = assemble(
        grid,
        None,
        [],
        np.zeros((space.n_r, 1, grid.n_omega), complex),
        spatial_grid=space,
        medium=model.medium,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    square, frame, numerator, clamped = model.shared_values()
    x, y = geometry.spectral_axes
    transverse = (x[None, :] ** 2 + y[:, None] ** 2).ravel()
    coefficients = SeparableOperator(
        k_squared=square,
        kperp_sq=transverse,
        k_squared_low=np.zeros_like(square),
        kperp_sq_low=np.zeros_like(transverse),
        frame=frame,
        spectral_rate=np.zeros(grid.n_omega),
        prefactor_numerator=numerator * grid.crop_scale,
        prefactor_clamped=clamped * grid.crop_scale,
        active=grid.active_mask,
        evanescent_rate=200.0,
        n_pol=1,
        z_span=(
            float(original.linear.phi_breakpoints[0]),
            float(original.linear.phi_breakpoints[-1]),
        ),
    )
    return dataclasses.replace(
        original,
        transform=dataclasses.replace(
            original.transform,
            geometry=geometry,
            kind="free",
            synthesis=np.empty((0, 0)),
            projection=np.empty((0, 0)),
            projection_coarse=np.empty((0, 0)),
        ),
        linear=dataclasses.replace(
            original.linear,
            phi_representation="separable",
            separable=coefficients,
            phi_constant=None,
            phi_coefficients=None,
        ),
        responses=dataclasses.replace(
            original.responses,
            prefactor_representation="separable",
            separable=coefficients,
            prefactor=None,
            prefactor_constant=None,
        ),
    )


def test_separable_contract_carries_no_full_coefficient_volume(free_contract):
    problem = free_contract
    problem.validate()
    bundle = terms(problem)
    assert bundle.phi is None and bundle.prefactor is None
    assert bundle.separable is problem.linear.separable
    assert bundle.separable is not None
    assert problem.device_ready
    rate, prefactor = separable_values(bundle.separable)
    # Constant-media differences use D*(z2-z1), with no subtraction of large
    # cumulative phases. One rounded multiplication suffices.
    actual = delta_phi(problem.linear, 0.2e-3, 0.7e-3)
    assert_allclose(actual, rate * (0.7e-3 - 0.2e-3), rtol=gamma(2), atol=0)
    assert_allclose(
        unitary_phase(problem.linear, 0.7e-3),
        np.imag(rate) * 0.7e-3,
        rtol=gamma(2),
        atol=0,
    )
    np.testing.assert_array_equal(prefactor_at(problem.responses, 0.5e-3), prefactor)
    with pytest.raises(TranslationError, match="outside"):
        prefactor_at(problem.responses, 1.0)


def test_separable_prefactor_compiles_and_reports_shared_data(free_contract, caplog):
    problem = free_contract
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    with caplog.at_level("DEBUG", logger="solprop.translate.responses"):
        compiled = compile_responses(
            [],
            grid,
            None,
            density=1.0,
            z_max=grid.length,
            mode_averaged=False,
            translated_prefactor=problem.linear.separable,
        )
    assert compiled.separable is problem.linear.separable
    assert "no coefficient volume" in caplog.text


def test_spatial_contract_checks_geometry_before_spatial_execution(free_contract):
    problem = free_contract
    without_geometry = dataclasses.replace(
        problem,
        transform=dataclasses.replace(
            problem.transform,
            geometry=None,
            kind="modal",
            node_chunk=1,
            synthesis=np.ones((1, 1)),
            projection=np.ones((1, 1)),
            projection_coarse=np.ones((1, 1)),
        ),
        # A problem without geometry carries no plane measure; dropping it too
        # leaves the separable operator as the one inconsistency.
        flux=None,
    )
    with pytest.raises(TranslationError, match="requires free or slab geometry"):
        without_geometry._check_spatial()
    without_geometry._check_spatial_collar_stability()
    without_separable = dataclasses.replace(
        problem,
        linear=constant_phi(np.zeros(problem.shape, complex), 1e-3),
        responses=dataclasses.replace(
            problem.responses,
            prefactor_representation="constant",
            separable=None,
            prefactor=tabulate(
                lambda z: np.zeros((len(z), *problem.shape[1:]), complex), 0.0, 1e-3
            ),
            prefactor_constant=np.zeros(problem.shape[1:], complex),
        ),
    )
    with pytest.raises(TranslationError, match="constant separable"):
        without_separable._check_spatial()


def test_free_absorber_stability_refuses_only_above_the_unitary_transverse_limit(
    free_contract,
):
    problem = free_contract
    geometry = problem.transform.geometry
    assert geometry is not None
    # The transverse FFT pair is unitary in the energy measure, so free geometry
    # scales the largest rate by its rounding model alone, gamma(128 n_columns), with
    # no matrix norm; the limit is 2 STABILITY_RADIUS over that rate. A rate one part
    # in a million above the limit's inverse is refused, one below is not.
    rounding = 128 * geometry.spatial_weights.size * np.finfo(float).eps
    rounding /= 1 - rounding
    assert problem.stepper.schedule is None
    threshold = 2 * STABILITY_RADIUS / (problem.stepper.dz_min * (1 + rounding))
    for factor, refused in ((1 - 1e-6, False), (1 + 1e-6, True)):
        rate = np.full_like(geometry.absorber_rate, threshold * factor)
        candidate = dataclasses.replace(
            problem,
            transform=dataclasses.replace(
                problem.transform,
                geometry=dataclasses.replace(geometry, absorber_rate=rate),
            ),
        )
        if refused:
            with pytest.raises(TranslationError, match="spatial collar stability"):
                candidate.validate()
        else:
            candidate.validate()


def test_separable_contract_rejects_unshared_or_inconsistent_data(free_contract):
    problem = free_contract
    copied = dataclasses.replace(problem.responses.separable)
    with pytest.raises(TranslationError, match="share the same"):
        dataclasses.replace(
            problem, responses=dataclasses.replace(problem.responses, separable=copied)
        ).validate()
    with pytest.raises(TranslationError, match="tabulate"):
        dataclasses.replace(
            problem,
            stepper=dataclasses.replace(problem.stepper, propagator_mode="tabulate"),
        ).validate()
    with pytest.raises(TranslationError, match="one linear interval"):
        dataclasses.replace(problem.linear, phi_degrees=np.zeros(1, dtype=np.intp))
    with pytest.raises(TranslationError, match="exactly"):
        dataclasses.replace(problem.linear, separable=None)
    with pytest.raises(TranslationError, match="exactly"):
        dataclasses.replace(problem.responses, separable=None)


@pytest.mark.parametrize("field", ["active", "kperp_sq"])
def test_separable_contract_rejects_mismatched_axes_or_mask(free_contract, field):
    problem = free_contract
    coefficients = problem.linear.separable
    assert coefficients is not None
    values = getattr(coefficients, field).copy()
    values[0] = not values[0] if field == "active" else 1.0
    changed = dataclasses.replace(coefficients, **{field: values})
    with pytest.raises(TranslationError, match=field):
        dataclasses.replace(
            problem,
            linear=dataclasses.replace(problem.linear, separable=changed),
            responses=dataclasses.replace(problem.responses, separable=changed),
        ).validate()


def test_a_cartesian_line_is_the_only_single_sample_axis():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    line = free_transform(LineGrid(1e-3, 8), grid, n_pol=1, z_max=grid.length).geometry
    assert line is not None
    assert line.kind == "free" and line.shape == (8, 1)
    with pytest.raises(TranslationError, match="integers of at least 2"):
        dataclasses.replace(line, shape=(1, 8))
    with pytest.raises(TranslationError, match="single y sample"):
        dataclasses.replace(line, axes=(line.axes[0], np.array([1e-3])))
    with pytest.raises(TranslationError, match="single y sample"):
        dataclasses.replace(
            line, spectral_axes=(line.spectral_axes[0], np.array([1.0]))
        )
    radial = radial_transform(
        RadialGrid(1e-3, 8), grid, n_pol=1, z_max=grid.length
    ).geometry
    assert radial is not None
    with pytest.raises(TranslationError, match="integers of at least 2"):
        dataclasses.replace(radial, shape=(1,))
