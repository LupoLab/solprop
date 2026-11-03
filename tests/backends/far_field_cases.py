# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The far-field pass's fixtures, its reference and their rounding bound (SEP 23).

The grids, apertures and fields a compiled pass is measured on
(``test_native_far_field.py``), and the reference it is measured against:
``numpy_ref``'s :func:`solprop.spatial.aperture_statistics`. Every comparison of two
separate computations is held to a rounding bound fixed before measuring (SEP 25): the
pass and the reference add the same terms in different orders, so their difference is
bounded by ``gamma_m`` times the sum of the terms' magnitudes, which the reference forms
from ``|E|`` and ``|alpha|`` itself.
"""

import dataclasses
import functools

import numpy as np

from bounds import assert_within, gamma
from solprop import fourier
from solprop.constants import EPS0, C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.spatial import (
    FAR_FIELD_STATISTICS,
    ApertureProfile,
    aperture_statistics,
)
from solprop.stats import ApertureSet, ApertureWeighting, StatsSpec
from solprop.translate.spatial import axial_admittance, axial_factors

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


def masks(profile: ApertureProfile = "hard", weighting="amplitude"):
    """Every shape at once, sized for :func:`fine_space`'s column images.

    A half-plane and its complement on each axis, a rectangle, an ellipse and a circle.
    ``supergauss`` keeps its rows to the shapes it has a meaning on, and leaves the
    half-planes hard beside them, one set of mixed profiles.
    """
    shapes = (
        "half_plane_x_ge",
        "half_plane_x_lt",
        "half_plane_y_ge",
        "half_plane_y_lt",
        "rectangle",
        "ellipse",
        "circle",
    )
    smooth = profile in ("tanh", "rcos")
    edge = 4.0 if profile == "supergauss" else 5e-3 if smooth else 0.0
    profiles = tuple(
        "hard"
        if profile == "supergauss" and shape.startswith("half_plane")
        else profile
        for shape in shapes
    )
    return ApertureSet(
        centres=[
            [2e-2, 0.0],
            [2e-2, 0.0],
            [0.0, -1e-2],
            [0.0, -1e-2],
            [1e-2, -5e-3],
            [-1e-2, 1e-2],
            [2e-2, 2e-2],
        ],
        diameters=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6e-2],
        distances=[DISTANCE] * len(shapes),
        edges=None
        if profile == "hard"
        else [0.0 if one == "hard" else edge for one in profiles],
        shapes=shapes,
        profiles=profiles,
        extents=[[0.0, 0.0]] * 4 + [[0.12, 0.08], [0.1, 0.06], [0.0, 0.0]],
        weighting=weighting,
    )


def fine_space():
    """A quarter-micron grid: its outer columns are evanescent at every frequency.

    Its column images lie some 6 cm apart in the mask plane at 800 nm, so the holes
    are scaled by :data:`FINE` to span several of them.
    """
    return FreeGrid(1.5e-6, 12, 1.25e-6, 10, collar_fraction=0.1)


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


def bound_for(magnitude, problem, name, factor=1):
    """The rounding bound of one statistic, elementwise, fixed before measuring.

    Each value is a sum over the columns of up to 8 rounded operations per term, and
    a squared modulus of a complex sum at most triples the relative bound: 3
    gamma(8 n + 64) of the sum of the magnitudes. An aperture's weight adds the
    transcendental routines' error (tanh, sin, exp: two ulps of one, an absolute
    error, which a tail's cancellation in ``1 - tanh`` does not shrink): 8 eps of
    the beam's magnitudes for ``W^p`` and 12 eps of the axis's for the squared
    amplitude. `factor` is how many edges a row's weight multiplies: a rectangle's two
    double the weight's error, to first order.
    """
    bound = 3 * gamma(8 * problem.shape[1] + 64) * np.abs(magnitude[name])
    if name == "aperture_spectrum":
        bound = bound + factor * 8 * np.finfo(float).eps * magnitude["beam_spectrum"]
    if name == "aperture_on_axis_spectrum":
        bound = (
            bound + factor * 12 * np.finfo(float).eps * magnitude["on_axis_spectrum"]
        )
    return bound


def assert_matches(actual, expected, magnitude, problem, names=WITH_AXIS, *, factor=1):
    for name in names:
        assert_within(
            actual[name],
            expected[name],
            bound_for(magnitude, problem, name, factor),
            elementwise=True,
            what=name,
        )
