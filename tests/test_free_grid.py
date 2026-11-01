# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cartesian discrete-transform oracles, independent of propagation backends."""

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hankel_oracles import gamma
from solprop.exceptions import GridError
from solprop.grid import FreeGrid, LineGrid
from spatial_records import record_metric


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (8, 5)])
def test_free_grid_flattening_transform_parseval_and_axis(shape, request):
    space = FreeGrid(2e-3, shape[0], 3e-3, shape[1], collar_fraction=0)
    size = space.n_columns
    # Geometric DFT sums and twiddle evaluation: manual Cartesian budgets,
    # derived before executing either FFT or the direct reconstruction.
    roundoff = gamma(64 * size)
    energy_bound = gamma(128 * size)
    rng = np.random.default_rng(773)
    field = rng.normal(size=(space.n_y, space.n_x, 2)) + 1j * rng.normal(
        size=(space.n_y, space.n_x, 2)
    )
    spectrum = np.fft.fft2(field, axes=(0, 1))
    norm = np.sum(np.abs(field))
    roundtrip = np.max(np.abs(np.fft.ifft2(spectrum, axes=(0, 1)) - field)) / norm
    assert roundtrip < roundoff
    direct = np.einsum(
        "ia,jb,ijq->abq",
        np.exp(
            -2j
            * np.pi
            * np.outer(np.arange(space.n_y), np.arange(space.n_y))
            / space.n_y
        ),
        np.exp(
            -2j
            * np.pi
            * np.outer(np.arange(space.n_x), np.arange(space.n_x))
            / space.n_x
        ),
        field,
    )
    measured = np.max(np.abs(spectrum - direct)) / norm
    assert measured < roundoff
    energy = np.sum(
        space.spatial_weights[:, None] * np.abs(field.reshape(size, 2)) ** 2
    )
    spectral = np.sum(
        space.spectral_weights[:, None] * np.abs(spectrum.reshape(size, 2)) ** 2
    )
    defect = abs(spectral / energy - 1)
    assert defect < energy_bound
    # A single non-Nyquist Fourier harmonic reconstructs to exactly one at zero,
    # including the half-pixel origin of an odd sample count.
    harmonic = np.exp(
        1j * (space.k_x[1] * space.x[None, :] + space.k_y[1] * space.y[:, None])
    )
    axial = np.sum(space.on_axis * np.fft.fft2(harmonic).ravel())
    axis_error = abs(axial - 1)
    assert axis_error < roundoff
    assert_allclose(
        space.r.reshape(space.n_y, space.n_x),
        np.hypot(space.x[None, :], space.y[:, None]),
        rtol=0,
        atol=0,
    )
    for name, value, bound in (
        ("round trip", roundtrip, roundoff),
        ("direct DFT", measured, roundoff),
        ("Parseval", defect, energy_bound),
        ("on axis", axis_error, roundoff),
    ):
        record_metric(request, f"Cartesian {name} (relative)", str(shape), value, bound)
    assert all(
        not getattr(space, name).flags.writeable
        for name in (
            "x",
            "y",
            "k_x",
            "k_y",
            "window",
            "r",
            "on_axis",
            "kperp_sq",
            "spatial_weights",
            "spectral_weights",
        )
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("radius_x", 0),
        ("radius_y", np.inf),
        ("n_x", 1),
        ("n_y", True),
        ("collar_fraction", -0.1),
        ("collar_fraction", np.inf),
        ("collar_fraction", np.nan),
    ],
)
def test_free_grid_rejects_invalid_dimensions(name, value):
    args = {"radius_x": 1e-3, "n_x": 8, "radius_y": 2e-3, "n_y": 6}
    args[name] = value
    with pytest.raises(GridError, match=name):
        FreeGrid(**args)


def test_free_grid_collar_extends_requested_domain():
    space = FreeGrid(2e-3, 40, 3e-3, 32, collar_fraction=0.25)
    assert space.x[0] == -2.5e-3
    assert space.y[0] == -3.75e-3
    inside = (np.abs(space.x)[None, :] <= space.radius_x) & (
        np.abs(space.y)[:, None] <= space.radius_y
    )
    assert np.all(space.window.reshape(space.n_y, space.n_x)[inside] == 1)
    assert np.all(space.window.reshape(space.n_y, space.n_x)[0] == 0)
    clone = dataclasses.replace(space)
    object.__setattr__(clone, "x", clone.x.copy())
    with pytest.raises(GridError, match="read-only"):
        clone.validate()


@pytest.mark.parametrize("radii", [(1e-300, 1e-3), (1e308, 1e308), (1e-150, 1e-170)])
def test_free_grid_refuses_unrepresentable_derived_measures(radii):
    with pytest.raises(GridError, match="representable"):
        FreeGrid(radii[0], 4, radii[1], 6, collar_fraction=0)


@pytest.mark.parametrize(
    "name,value", [("window", 2.0), ("spectral_weights", 0.0), ("kperp_sq", -1.0)]
)
def test_free_grid_validation_refuses_corrupt_derived_values(name, value):
    space = FreeGrid(1e-3, 8, 1e-3, 8)
    values = np.full(space.n_columns, value)
    values.setflags(write=False)
    object.__setattr__(space, name, values)
    with pytest.raises(GridError, match="invalid"):
        space.validate()


def test_line_grid_is_the_x_axis_of_a_free_grid_with_a_unit_y_measure():
    line = LineGrid(2e-3, 12, collar_fraction=0.2)
    plane = FreeGrid(2e-3, 12, 3e-3, 8, collar_fraction=0.2)
    assert isinstance(line, FreeGrid)
    # The same axis construction on the same arguments: products of integers and one
    # spacing, correctly rounded, so the axes coincide exactly (structure).
    np.testing.assert_array_equal(line.x, plane.x)
    np.testing.assert_array_equal(line.k_x, plane.k_x)
    assert line.y.tolist() == [0.0] and line.k_y.tolist() == [0.0]
    assert (line.n_y, line.radius_y, line.n_columns) == (1, math.inf, 12)
    assert line.transverse_measure == "per_unit_width"
    spacing = 2 * (2e-3 * 1.2 / 12)
    # The measure is the x spacing times one metre of y. The spacing here is formed in
    # this test's own order, so each weight agrees to the rounding of its few
    # operations, 4 eps; r = |x| and k_x^2 + 0 are exact in IEEE arithmetic.
    rounding = 4 * np.finfo(float).eps
    np.testing.assert_allclose(
        line.spatial_weights, np.full(12, spacing), rtol=rounding
    )
    np.testing.assert_allclose(
        line.spectral_weights, np.full(12, spacing / 12), rtol=rounding
    )
    np.testing.assert_array_equal(line.kperp_sq, plane.k_x**2)
    np.testing.assert_array_equal(line.r, np.abs(plane.x))
    # The plane's row through y = 0 has a unit y collar: the line's collar is x's.
    np.testing.assert_array_equal(line.window, plane.window.reshape(8, 12)[4])
    # A single non-Nyquist harmonic reconstructs to exactly one at x = 0.
    harmonic = np.exp(1j * line.k_x[1] * line.x)
    assert abs(np.sum(line.on_axis * np.fft.fft(harmonic)) - 1) < gamma(64 * 12)
    assert dataclasses.replace(line, radius_x=1e-3).x[0] == -1.2e-3


def test_line_grid_parseval_is_per_metre_of_y():
    line = LineGrid(1e-3, 16, collar_fraction=0)
    field = np.random.default_rng(5).normal(size=16) + 0j
    # One FFT and two sums of 16 terms: fewer than 64 rounded operations per term.
    direct = float(np.sum(line.spatial_weights * np.abs(field) ** 2))
    spectral = float(np.sum(line.spectral_weights * np.abs(np.fft.fft(field)) ** 2))
    assert abs(spectral / direct - 1) < gamma(64 * 16)


@pytest.mark.parametrize(
    "arguments,match",
    [
        ((0.0, 8), "radius_x"),
        ((math.inf, 8), "radius_x"),
        ((1e-3, 1), "n_x"),
        ((1e-3, True), "n_x"),
        ((1e-3, 8, -1.0), "collar_fraction"),
        ((1e-300, 8), "representable"),
    ],
)
def test_line_grid_refuses_invalid_dimensions(arguments, match):
    with pytest.raises(GridError, match=match):
        LineGrid(*arguments)


def test_free_grid_still_needs_two_samples_in_y():
    with pytest.raises(GridError, match="n_y must be an integer of at least 2"):
        FreeGrid(1e-3, 8, 1e-3, 1)


def test_line_grid_prints_its_measure():
    line = LineGrid(2e-3, 128)
    assert repr(line) == "LineGrid(x=2 mm x 128)"
    assert "per metre of y" in str(line)
