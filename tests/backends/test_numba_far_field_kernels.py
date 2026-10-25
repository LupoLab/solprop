# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s banded far-field pass (SEP 23 section 6; section 7 items 3, 10, 11).

The three bodies against the reference's definitions, the pass against
``numpy_ref``'s :func:`solprop.spatial.aperture_statistics` on field and envelope
grids, one and two polarisations, a plane and a line; the ``W = 1`` identities; thread
counts 1, 2 and 4; the shared-mask path; hard-edge membership at an edge; the band
edge cases; and the refusals. Every comparison of two separate computations is held to
a rounding bound fixed before measuring (SEP 25): the pass and the reference add the
same terms in different orders, so their difference is bounded by ``gamma_m`` times
the sum of the terms' magnitudes, which the reference forms from ``|E|`` and
``|alpha|`` itself.

A kernel test module: ``poe cov-kernels`` runs it with the compiler disabled.
"""

import dataclasses
import functools
import math

import numpy as np
import pytest
from numba import config, get_num_threads, set_num_threads

from bounds import assert_within, gamma
from solprop import fourier
from solprop.backends import select
from solprop.backends import spatial as backend_spatial
from solprop.backends.bodies import PLAIN
from solprop.backends.memory import estimate_memory, far_field_memory
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numba_cpu.spatial import FarFieldPass
from solprop.backends.spatial import (
    FarFieldLayout,
    far_field_bands,
    far_field_layout,
    far_field_partials,
)
from solprop.boundaries import TransverseCollar, absorber
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    APERTURE_PROFILES,
    FAR_FIELD_STATISTICS,
    ApertureProfile,
    SpatialMedium,
    aperture_profile,
    aperture_statistics,
    mask_plane_radius,
)
from solprop.stats import ApertureSet, ApertureWeighting, StatsSpec
from solprop.stepper import PIController, RungeKutta
from solprop.translate.spatial import (
    aperture_tables,
    axial_admittance,
    axial_factors,
    longitudinal_root,
)

DISTANCE = 0.25
"""Exit face to mask plane, m."""

QUADRANTS = np.array(
    [
        [-np.inf, 0.0, -np.inf, 0.0],
        [0.0, np.inf, -np.inf, 0.0],
        [-np.inf, 0.0, 0.0, np.inf],
        [0.0, np.inf, 0.0, np.inf],
    ]
)

WITH_AXIS = (*FAR_FIELD_STATISTICS, "on_axis_spectrum")


@functools.cache
def time_grid(kind):
    if kind == "envelope":
        return EnvelopeGrid.from_limits(
            1e-3, 800e-9, (700e-9, 900e-9), 100e-15, thg=False
        )
    return FieldGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 60e-15)


def holes(
    profile: ApertureProfile = "tanh",
    weighting: ApertureWeighting = "amplitude",
    *,
    shared=False,
    size=1.0,
):
    """Three apertures at the mask plane; `shared` gives them one centre.

    `size` scales every length, so that the holes span several column images on a
    finely sampled grid, whose images are far apart.
    """
    centres = (
        [[0.5e-3, -0.3e-3]] * 3
        if shared
        else [[0.0, 0.0], [1.0e-3, 0.0], [-0.6e-3, 0.8e-3]]
    )
    edges = {
        "hard": None,
        "tanh": [1e-4, 1e-4, 1e-4],
        "rcos": [1e-4, 1e-4, 1e-4],
        "supergauss": [4.0, 6.0, 2.0],
    }[profile]
    lengths = size if profile in ("tanh", "rcos") else 1.0
    return ApertureSet(
        centres=size * np.array(centres),
        diameters=size * np.array([2.0e-3, 1.0e-3, 1.6e-3]),
        distances=[DISTANCE] * 3,
        edges=None if edges is None else lengths * np.array(edges),
        profile=profile,
        weighting=weighting,
    )


FINE = 50.0
"""The aperture scale on the finely sampled grid of :func:`fine_space`."""


def fine_space():
    """A quarter-micron grid: its outer columns are evanescent at every frequency.

    Its column images lie some 6 cm apart in the mask plane at 800 nm, so the holes
    are scaled by :data:`FINE` to span several of them.
    """
    return FreeGrid(1.5e-6, 12, 1.25e-6, 10, window_factor=0.1)


def spec_of(apertures, names=FAR_FIELD_STATISTICS, regions=QUADRANTS):
    return StatsSpec(names=tuple(names), apertures=apertures, spectral_regions=regions)


def random_field(space, n_pol, grid, *, seed=7):
    """A spectrum with content in every column, the evanescent ones among them."""
    rng = np.random.default_rng(seed)
    shape = (space.n_columns, n_pol, grid.n_omega)
    field = 1e3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    if grid.kind == "field":
        field[..., 0] = field[..., -1] = 0.0  # a real signal's end bins
    return field


def problem_of(grid, space, field, spec, *, medium=None, collar=None):
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(1 + 1e-6j) if medium is None else medium,
        [],
        field,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[1e-4],
            dz_max=1e-4,
            dz_init=1e-4,
            absorber=absorber(grid, temporal=False),
            backend="numba_cpu",
            integrator=RungeKutta(),
            stats=spec,
            transverse_absorber=collar,
        ),
    )
    return simulation.translate()


def reference(names, grid, problem, field, tables):
    """``numpy_ref``'s pass, with the on-axis spectrum of the contraction.

    The on-axis spectrum pairs the coherent axis amplitude with its magnetic
    closure, the same contraction of ``conj(eta) E`` (SEP 34).
    """
    names = tuple({name for name in names if name in FAR_FIELD_STATISTICS})
    operator = problem.linear.separable
    values = aperture_statistics(
        names,
        grid,
        field,
        tables,
        axial=functools.partial(axial_factors, operator, grid.omega),
    )
    closing = np.conj(axial_admittance(operator, grid.omega, slice(None)))
    axis = np.einsum("r,rpw->pw", tables.on_axis, field)
    closure = np.einsum("r,rpw->pw", tables.on_axis, closing[:, None] * field)
    factor = fourier.spectral_density(
        np.ones(grid.n_omega, complex), grid.dt, kind=grid.kind
    )
    values["on_axis_spectrum"] = (
        0.5 * EPS0 * C * factor * np.sum(np.real(axis * np.conj(closure)), axis=0)
    )
    return values


def magnitudes(names, grid, problem, field, tables):
    """The reference formed from ``|E|`` and ``|alpha|``: the sums of the magnitudes."""
    absolute = dataclasses.replace(tables, on_axis=np.abs(tables.on_axis) + 0j)
    names = (*names, "beam_spectrum")
    return reference(names, grid, problem, np.abs(field) + 0j, absolute)


def bound_for(magnitude, problem, name):
    """The rounding bound of one statistic, elementwise, fixed before measuring.

    Each value is a sum over the columns of up to 8 rounded operations per term, and
    a squared modulus of a complex sum at most triples the relative bound: 3
    gamma(8 n + 64) of the sum of the magnitudes. An aperture's weight adds the
    transcendental routines' error (tanh, sin, exp: two ulps of one, an absolute
    error, which a tail's cancellation in ``1 - tanh`` does not shrink): 8 eps of
    the beam's magnitudes for ``W^p`` and 12 eps of the axis's for the squared
    amplitude.
    """
    bound = 3 * gamma(8 * problem.shape[1] + 64) * np.abs(magnitude[name])
    if name == "aperture_spectrum":
        bound = bound + 8 * np.finfo(float).eps * magnitude["beam_spectrum"]
    if name == "aperture_on_axis_spectrum":
        bound = bound + 12 * np.finfo(float).eps * magnitude["on_axis_spectrum"]
    return bound


def assert_matches(actual, expected, magnitude, problem, names=WITH_AXIS):
    for name in names:
        assert_within(
            actual[name],
            expected[name],
            bound_for(magnitude, problem, name),
            elementwise=True,
            what=name,
        )


# ----------------------------------------------------------------------- the bodies


def test_the_root_body_is_the_reference_root_and_the_stage_rate_is_unchanged():
    square = np.array([4e12 + 1e3j, -2e10 + 5e2j, -1e12 + 0j, 1e2 - 3e1j])
    for family in (PLAIN, HOST):
        for value in square:
            root, propagating, attenuation = family.longitudinal_root(value, 200.0)
            assert root == np.sqrt(value)
            assert propagating == (value.real > 0)
            expected = (
                np.sqrt(value).imag
                if value.real > 0
                else min(abs(np.sqrt(value).imag), 200.0)
            )
            assert attenuation == expected
            rate, _ = family.longitudinal_rate(
                value, 1.0, 0.25, 1j, 2j, True, 200.0, 0.5
            )
            assert rate == complex(-attenuation - 0.25, -(root.real - 1.0)) - 0.125


def test_the_radius_body_is_the_reference_radius_to_the_bit():
    # The hard-mask contract: the same correctly rounded operations on the same
    # inputs, so every decision at an edge is the reference's.
    rng = np.random.default_rng(3)
    scale = rng.uniform(1e-9, 1e-7, 40)
    k_x, k_y = rng.uniform(-1e5, 1e5, 7), rng.uniform(-1e5, 1e5, 7)
    centre = (3.1e-4, -7.3e-4)
    expected = mask_plane_radius(scale, k_x, k_y, centre)
    for family in (PLAIN, HOST):
        actual = np.array(
            [
                [family.mask_plane_radius(s, x, y, *centre) for s in scale]
                for x, y in zip(k_x, k_y, strict=True)
            ]
        )
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("profile", APERTURE_PROFILES)
def test_the_weight_body_is_the_profile_with_its_zeros(profile):
    half_width, edge = 1.0, {"hard": 0.0, "supergauss": 4.0}.get(profile, 0.125)
    inverse = 1.0 / edge if profile in ("tanh", "rcos") else 0.0
    code = APERTURE_PROFILES.index(profile)
    radii = np.array([0.0, 0.5, 0.875, 1.0, 1.0625, 1.125, 1.5, 3.0])
    expected = aperture_profile(radii, half_width, edge, profile)
    for family in (PLAIN, HOST):
        weights = np.array(
            [
                family.aperture_weight(
                    rho, 0.0, 1.0, half_width, inverse, edge, 10.0, code
                )
                for rho in radii
            ]
        )
        # One transcendental per weight, against NumPy's: a few ulps.
        np.testing.assert_allclose(weights, expected, rtol=4e-16, atol=4e-16)
        # Exactly zero at zero scale and where the image reaches the distance.
        assert (
            family.aperture_weight(0.5, 0.0, 0.0, 1.0, inverse, edge, 10.0, code) == 0
        )
        assert (
            family.aperture_weight(0.5, 10.0, 1.0, 1.0, inverse, edge, 10.0, code) == 0
        )
    if profile == "rcos":
        # Compact support: exactly one and zero beyond the edge.
        assert HOST.aperture_weight(0.875, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, code) == 1.0
        assert HOST.aperture_weight(1.125, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, code) == 0.0


def test_the_region_body_is_half_open_on_the_physical_wavevector():
    for family in (PLAIN, HOST):
        # The stored axis is the negated physical one.
        assert family.region_member(-0.0, -1.0, 0.0, math.inf, 0.0, 2.0)
        assert family.region_member(0.0, 0.0, 0.0, math.inf, 0.0, 2.0)
        assert not family.region_member(1.0, -1.0, 0.0, math.inf, 0.0, 2.0)
        assert not family.region_member(0.0, -2.0, 0.0, math.inf, 0.0, 2.0)
        assert not family.region_member(-5.0, -5.0, -math.inf, 0.0, -math.inf, 0.0)
        assert family.region_member(5.0, 5.0, -math.inf, 0.0, -math.inf, 0.0)


# ------------------------------------------------------------- the layout and bands


def test_the_layout_takes_only_the_present_slots():
    layout = far_field_layout(WITH_AXIS, n_apertures=3, n_regions=4, n_pol=2)
    # Each amplitude holds its electric sum and its magnetic closure, 4 n_pol slots.
    assert layout == FarFieldLayout(
        beam=0,
        absorbed=1,
        regions=5,
        apertures=9,
        stride=9,
        collected=0,
        amplitude=1,
        axis=36,
        n_slots=44,
    )
    assert layout is not None
    assert layout.array().tolist() == [0, 1, 5, 9, 9, 0, 1, 36, 44]
    alone = far_field_layout(
        ("aperture_on_axis_spectrum",), n_apertures=2, n_regions=0, n_pol=1
    )
    assert alone is not None
    assert (alone.apertures, alone.stride, alone.collected, alone.amplitude) == (
        0,
        4,
        -1,
        0,
    )
    assert far_field_layout(("energy",), n_apertures=0, n_regions=0, n_pol=1) is None
    # A region request without regions takes no slot.
    assert (
        far_field_layout(("region_spectrum",), n_apertures=0, n_regions=0, n_pol=1)
        is None
    )


def test_the_bands_follow_the_shapes_and_the_budget(monkeypatch):
    assert far_field_bands(5, 7, 3, 10).tolist() == [0, 5, 10, 15, 20, 25, 30, 35]
    assert far_field_bands(10, 1, 3, 10).tolist() == [0, 10]  # one block of a line
    # A budget of two bands over seven rows: three and four rows.
    monkeypatch.setattr(backend_spatial, "FAR_FIELD_BAND_BUDGET", 2 * 8 * 3 * 10)
    assert far_field_bands(5, 7, 3, 10).tolist() == [0, 15, 35]
    # A budget below one band's partials still takes one band.
    monkeypatch.setattr(backend_spatial, "FAR_FIELD_BAND_BUDGET", 1)
    assert far_field_bands(5, 7, 3, 10).tolist() == [0, 35]


# ------------------------------------------------------- item 10: against numpy_ref


@pytest.mark.parametrize(
    ("kind", "n_pol", "profile", "weighting"),
    [
        ("envelope", 1, "tanh", "amplitude"),
        ("envelope", 2, "rcos", "quadrature"),
        ("field", 1, "supergauss", "amplitude"),
        ("field", 2, "hard", "amplitude"),
    ],
)
def test_the_pass_matches_the_reference(kind, n_pol, profile, weighting):
    grid = time_grid(kind)
    space = fine_space()
    spec = spec_of(holes(profile, weighting, size=FINE))
    field = random_field(space, n_pol, grid)
    problem = problem_of(
        grid, space, field, spec, collar=TransverseCollar(4e3, fraction=0.5)
    )
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    values, axis = FarFieldPass(problem, parallel=False)(WITH_AXIS, grid, field, tables)
    assert set(values) == set(WITH_AXIS)
    assert axis is not None
    assert [part.shape for part in axis] == [(n_pol, grid.n_omega)] * 2
    expected = reference(WITH_AXIS, grid, problem, field, tables)
    assert_matches(
        values, expected, magnitudes(WITH_AXIS, grid, problem, field, tables), problem
    )
    # Every channel but the evanescent one is live on this fixture (an evanescent
    # column carries no axial flux, SEP 34), and every aperture collects.
    assert np.all(np.max(values["aperture_spectrum"], axis=1) > 0)
    assert np.all(values["absorbed_power"][[0, 1, 3]] > 0)
    assert values["absorbed_power"][2] == 0.0
    assert problem.linear.separable is not None
    _, propagating, _ = longitudinal_root(problem.linear.separable)
    assert not np.all(propagating)


def test_the_pass_matches_the_reference_on_a_line():
    grid = time_grid("envelope")
    space = LineGrid(4e-4, 130, window_factor=0)  # three blocks of columns
    apertures = ApertureSet(
        centres=[[0.0, 0.0], [1.2e-3, 0.0]],
        diameters=[1e-3, 6e-4],
        distances=[DISTANCE, 0.5],
        edges=[5e-5, 5e-5],
        profile="rcos",
        weighting="quadrature",
    )
    regions = np.array([[-np.inf, 0.0, -1.0, 1.0], [0.0, 2e4, -np.inf, np.inf]])
    spec = spec_of(apertures, regions=regions)
    field = random_field(space, 1, grid, seed=11)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    assert far_field_partials(problem)[1] == 3
    values, _ = FarFieldPass(problem, parallel=False)(WITH_AXIS, grid, field, tables)
    expected = reference(WITH_AXIS, grid, problem, field, tables)
    assert_matches(
        values, expected, magnitudes(WITH_AXIS, grid, problem, field, tables), problem
    )


def test_the_operator_slots_follow_gain_and_zero_frequency():
    # A field grid's zero-frequency bin and a gain medium: the material channel is
    # negative, the admittance and every weight zero at zero frequency.
    grid = time_grid("field")
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0)
    spec = spec_of(holes("hard"))
    field = random_field(space, 1, grid)
    field[..., 0] = 1e3  # content at zero frequency
    problem = problem_of(grid, space, field, spec, medium=SpatialMedium(1 - 1e-5j))
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    values, _ = FarFieldPass(problem, parallel=False)(WITH_AXIS, grid, field, tables)
    expected = reference(WITH_AXIS, grid, problem, field, tables)
    assert values["absorbed_power"][3] < 0
    assert values["beam_spectrum"][0] == 0.0
    assert np.all(values["aperture_spectrum"][:, 0] == 0.0)
    assert np.all(values["aperture_on_axis_spectrum"][:, 0] == 0.0)
    # The material channel's terms are signed: its bound is that of the magnitudes.
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    names = tuple(name for name in WITH_AXIS if name != "absorbed_power")
    assert_matches(values, expected, absolute, problem, names)
    np.testing.assert_allclose(
        values["absorbed_power"],
        expected["absorbed_power"],
        rtol=0,
        atol=3
        * gamma(8 * problem.shape[1] + 64)
        * np.max(np.abs(absolute["absorbed_power"])),
    )


def test_a_partial_request_takes_only_its_slots():
    # The beam spectrum without the absorbed power, the on-axis spectra through the
    # holes without their collected spectra.
    grid = time_grid("envelope")
    space = fine_space()
    names = ("beam_spectrum", "aperture_on_axis_spectrum")
    spec = spec_of(holes("supergauss", size=FINE), names, regions=np.empty((0, 4)))
    field = random_field(space, 1, grid)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    values, axis = FarFieldPass(problem, parallel=False)(names, grid, field, tables)
    assert set(values) == set(names)
    assert axis is None
    assert_matches(
        values,
        reference(names, grid, problem, field, tables),
        magnitudes(names, grid, problem, field, tables),
        problem,
        names,
    )


# -------------------------------------------------------------- item 3: W = 1


def test_a_hole_enclosing_every_image_is_the_beam_and_the_axis():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 12, window_factor=0)
    k_max = np.hypot(np.max(np.abs(space.k_x)), np.max(np.abs(space.k_y)))
    omega_min = float(np.min(grid.omega))
    # Every column propagates at every frequency, and its image lies inside.
    assert k_max < omega_min / C
    reach = DISTANCE * C * k_max / omega_min
    apertures = ApertureSet(
        centres=[[0.0, 0.0]], diameters=[4 * reach], distances=[DISTANCE]
    )
    assert 2 * reach < DISTANCE
    spec = StatsSpec(
        names=("aperture_spectrum", "aperture_on_axis_spectrum", "beam_spectrum"),
        apertures=apertures,
    )
    field = random_field(space, 2, grid)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    assert tables is not None
    # Every weight is exactly one.
    for column in range(space.n_columns):
        kx = tables.k_x[column % space.n_x]
        ky = tables.k_y[column // space.n_x]
        for scale in tables.scale[0]:
            radius = HOST.mask_plane_radius(scale, kx, ky, 0.0, 0.0)
            assert (
                HOST.aperture_weight(
                    radius,
                    radius,
                    scale,
                    tables.half_widths[0],
                    0.0,
                    0.0,
                    DISTANCE,
                    0,
                )
                == 1.0
            )
    names = (*spec.names, "on_axis_spectrum")
    values, _ = FarFieldPass(problem, parallel=False)(names, grid, field, tables)
    absolute = magnitudes(names, grid, problem, field, tables)
    assert_within(
        values["aperture_spectrum"][0],
        values["beam_spectrum"],
        bound_for(absolute, problem, "beam_spectrum"),
        elementwise=True,
    )
    assert_within(
        values["aperture_on_axis_spectrum"][0],
        values["on_axis_spectrum"],
        bound_for(absolute, problem, "on_axis_spectrum"),
        elementwise=True,
    )


# ------------------------------------------------- item 11: threads and the paths


def shared_case():
    grid = time_grid("envelope")
    space = fine_space()
    spec = spec_of(holes("tanh", shared=True, size=FINE))
    field = random_field(space, 2, grid, seed=5)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    return grid, field, problem, tables


@pytest.mark.skipif(
    int(vars(config)["NUMBA_NUM_THREADS"]) < 4,
    reason="needs four numba threads",
)
def test_the_pass_does_not_depend_on_the_thread_count(thread_budget_lifted):
    grid, field, problem, tables = shared_case()
    pass_ = FarFieldPass(problem, parallel=True)
    assert far_field_partials(problem)[1] == 10  # one band per row
    before = get_num_threads()
    results = {}
    try:
        for threads in (1, 2, 4):
            set_num_threads(threads)
            results[threads], _ = pass_(WITH_AXIS, grid, field, tables)
    finally:
        set_num_threads(before)
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    for threads in (2, 4):
        # The bands and every sum's order are fixed by the shapes: within the
        # rounding bound (they agree to the bit, which is not asserted).
        assert_matches(results[threads], results[1], absolute, problem)


def test_the_shared_mask_path_is_the_general_one():
    grid, field, problem, tables = shared_case()
    pass_ = FarFieldPass(problem, parallel=False)
    general, _ = pass_(WITH_AXIS, grid, field, tables, shared=False)
    shared, _ = pass_(WITH_AXIS, grid, field, tables)  # taken: one centre, one z
    forced, _ = pass_(WITH_AXIS, grid, field, tables, shared=True)
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    assert_matches(shared, general, absolute, problem)
    assert_matches(forced, general, absolute, problem)
    assert_matches(
        general, reference(WITH_AXIS, grid, problem, field, tables), absolute, problem
    )


def test_a_column_at_a_hard_edge_is_decided_as_the_reference_decides():
    # One column carries the light; the hole's edge passes through its image at one
    # bin, to the bit, and a hair inside and outside.
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 10, window_factor=0)
    column, bin_ = 29, grid.n_omega // 2
    tables = aperture_tables(
        spec_of(holes("hard"), ("aperture_spectrum",)),
        grid,
        space,
    )
    assert tables is not None
    centre = (2e-4, -1e-4)
    radius = float(
        mask_plane_radius(
            tables.scale[0, bin_ : bin_ + 1],
            tables.k_x[column % space.n_x : column % space.n_x + 1],
            tables.k_y[column // space.n_x : column // space.n_x + 1],
            centre,
        )[0, 0]
    )
    field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    field[column, 0, bin_] = 1e3
    for diameter in (
        2 * radius,
        np.nextafter(2 * radius, 0),
        np.nextafter(2 * radius, 1),
    ):
        apertures = ApertureSet(
            centres=[list(centre)], diameters=[diameter], distances=[DISTANCE]
        )
        spec = StatsSpec(names=("aperture_spectrum",), apertures=apertures)
        problem = problem_of(grid, space, field, spec)
        tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
        values, _ = FarFieldPass(problem, parallel=False)(
            spec.names, grid, field, tables
        )
        expected = reference(spec.names, grid, problem, field, tables)
        inside = diameter / 2 >= radius
        # Structural: the membership is exact, so the one term is in both or neither.
        assert (values["aperture_spectrum"][0, bin_] > 0) == inside
        assert (expected["aperture_spectrum"][0, bin_] > 0) == inside


# ------------------------------------------------------ buffers, bands, refusals


def test_the_partials_are_held_once_and_estimated_to_the_byte(monkeypatch):
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 2.5e-4, 10, window_factor=0.1)
    spec = spec_of(holes("tanh"), WITH_AXIS)
    field = random_field(space, 2, grid)
    problem = problem_of(grid, space, field, spec)
    n_slots, n_bands = far_field_partials(problem)
    assert (n_slots, n_bands) == (5 + 4 + 3 * 9 + 8, 10)
    rows = far_field_memory(problem, "numba_cpu")
    assert rows == {"statistics_partials": 8 * (n_bands + 1) * n_slots * grid.n_omega}
    assert far_field_memory(problem, "numpy_ref") == {}
    # Kokkos holds the partials alone: its native pass adds the bands as it finishes.
    assert far_field_memory(problem, "native") == {
        "far_field_partials": 8 * n_bands * n_slots * grid.n_omega
    }
    constructed = select("numba_cpu", problem, grid=grid)
    try:
        assert constructed.memory_ledger()["statistics_partials"] == sum(rows.values())
    finally:
        constructed.close()
    assert (
        estimate_memory(problem, "numba_cpu").groups["base"]
        == estimate_memory(problem, "numpy_ref").groups["base"]
        + rows["statistics_partials"]
    )
    # A band budget of two bands: the same values, within the rounding bound.
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    pass_ = FarFieldPass(problem, parallel=False)
    many, _ = pass_(WITH_AXIS, grid, field, tables)
    monkeypatch.setattr(
        backend_spatial, "FAR_FIELD_BAND_BUDGET", 2 * 8 * n_slots * grid.n_omega
    )
    few, _ = pass_(WITH_AXIS, grid, field, tables)
    assert_matches(
        few, many, magnitudes(WITH_AXIS, grid, problem, field, tables), problem
    )


def test_a_request_beyond_the_held_partials_takes_its_own():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0)
    field = random_field(space, 1, grid)
    bare = problem_of(grid, space, field, StatsSpec(names=("energy",)))
    assert far_field_partials(bare) == (0, 0)
    assert far_field_memory(bare, "numba_cpu") == {}
    pass_ = FarFieldPass(bare, parallel=False)
    assert pass_.nbytes == 0
    assert pass_(("energy",), grid, field, None) == ({}, None)
    spec = spec_of(holes("rcos"))
    tables = aperture_tables(spec, grid, space)
    values, axis = pass_(spec.names, grid, field, tables)
    assert axis is None
    assert_matches(
        values,
        reference(spec.names, grid, bare, field, tables),
        magnitudes(spec.names, grid, bare, field, tables),
        bare,
        spec.names,
    )


def test_the_pass_refuses_what_it_cannot_form():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0)
    field = random_field(space, 1, grid)
    problem = problem_of(grid, space, field, spec_of(holes("tanh")))
    pass_ = FarFieldPass(problem, parallel=False)
    with pytest.raises(BackendError, match="need the tables"):
        pass_(("beam_spectrum",), grid, field, None)
    tables = aperture_tables(
        problem.stats_spec, problem.grids, problem.transform.geometry
    )
    with pytest.raises(BackendError, match="one centre and one distance"):
        pass_(("aperture_spectrum",), grid, field, tables, shared=True)
    radial = Simulation.from_radial(
        grid,
        RadialGrid(3e-4, 8),
        SpatialMedium(),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 1e-4),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[1e-4],
            dz_max=1e-4,
            dz_init=1e-4,
            absorber=None,
            backend="numba_cpu",
        ),
    ).translate()
    with pytest.raises(BackendError, match="Cartesian or line"):
        FarFieldPass(radial, parallel=False)
    assert far_field_partials(radial) == (0, 0)
    # The pass reads the operator's root as the plane's; any other measure refuses.
    foreign = dataclasses.replace(problem, flux=None)
    with pytest.raises(BackendError, match="own separable plane measure"):
        FarFieldPass(foreign, parallel=False)


def test_numba_cpu_folds_the_axis_statistics_into_the_pass():
    # The default free-space set: on_axis_spectrum is the pass's W = 1 slot and
    # on_axis_intensity reads its amplitude; both agree with numpy_ref.
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 16, 3e-4, 16, window_factor=0.1)
    launch = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 6e-5
    ).build(grid, space, medium=SpatialMedium(1 + 1e-6j))
    names = ("energy", "on_axis_intensity", "on_axis_spectrum", "beam_spectrum")
    problem = problem_of(grid, space, launch, StatsSpec(names=names))
    values = {}
    for backend in ("numpy_ref", "numba_cpu"):
        owner = select(backend, problem, grid=grid)
        try:
            owner.set_initial(problem.initial.field0, None)
            values[backend] = owner.stats(problem.stats_spec)
        finally:
            owner.close()
    for name in names:
        # A Gaussian's terms are positive at the axis: the summation bound of the
        # value itself, with the contraction's and the transform's rounding.
        assert_within(
            values["numba_cpu"][name],
            values["numpy_ref"][name],
            3 * gamma(8 * space.n_columns + 64) + 64 * np.finfo(float).eps,
            what=name,
        )
