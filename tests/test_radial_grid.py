# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.exceptions import GridError
from solprop.grid import RadialGrid


def test_radial_grid_carries_the_qdht_measure_and_declared_collar():
    grid = RadialGrid(1e-3, 64)
    assert grid.validate() is None
    assert grid.r is grid.hankel.r
    assert grid.k_r is grid.hankel.k_r
    assert grid.radial_weights is grid.hankel.radial_weights
    assert grid.spectral_weights is grid.hankel.spectral_weights
    assert np.all(grid.window[grid.r <= 0.9 * grid.radius] == 1)
    assert np.all(grid.window[grid.r > 0.9 * grid.radius] < 1)
    assert not grid.window.flags.writeable
    assert np.all(np.diff(grid.window) <= 0)
    # The weights are positive integration measures, not annular point areas
    # derived from neighbouring r differences.
    assert np.all(grid.radial_weights > 0)
    assert np.all(grid.spectral_weights > 0)


def test_radial_grid_can_disable_the_collar_explicitly():
    grid = RadialGrid(1e-3, 8, window_factor=0)
    assert_allclose(grid.window, np.ones(8), rtol=0, atol=0)


@pytest.mark.parametrize("fraction", [-0.1, 1, 2, np.inf, np.nan])
def test_radial_grid_rejects_invalid_window_fraction(fraction):
    with pytest.raises(GridError, match="window_factor"):
        RadialGrid(1e-3, 8, fraction)


@pytest.mark.parametrize(
    "change",
    ["shape", "dtype", "mutable", "nan", "negative", "unordered", "edge", "window"],
)
def test_radial_grid_validation_rejects_corrupted_data(change):
    grid = RadialGrid(1e-3, 8)
    values = np.array(grid.r)
    if change == "shape":
        values = values[:-1]
    elif change == "dtype":
        values = values.astype(np.float32)
    elif change == "nan":
        values[0] = np.nan
    elif change == "negative":
        values[0] = -1
    elif change == "unordered":
        values = values[::-1].copy()
    elif change == "edge":
        values[-1] = grid.radius
    elif change == "window":
        values = np.full(8, 2.0)
    if change != "mutable":
        values.setflags(write=False)
    if change == "window":
        object.__setattr__(grid, "window", values)
    else:
        object.__setattr__(grid.hankel, "r", values)
    with pytest.raises(GridError, match="RadialGrid"):
        grid.validate()
