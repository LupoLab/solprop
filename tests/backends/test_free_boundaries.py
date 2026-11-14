# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise Cartesian backend diagnostics."""

import copy

import numpy as np
import pytest

from backends.spatial_cases import make_problem
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError, GridError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.spatial import cartesian_fluence_widths


@pytest.fixture
def free_backend_problem():
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 4, 4e-4, 6, collar_fraction=0)
    problem, run_request = make_problem(
        grid,
        space,
        np.zeros((space.n_columns, 1, grid.n_omega), complex),
        [KerrResponse(1e-22, thg=False)],
    )
    return grid, problem, run_request


def test_cartesian_reference_stage_names_missing_geometry(free_backend_problem):
    grid, problem, run_request = free_backend_problem
    backend = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        bundle = backend._stage_scalars(0.0, 0.0)
        transform = copy.copy(problem.transform)
        object.__setattr__(transform, "geometry", None)
        route = copy.copy(problem.route)
        object.__setattr__(route, "transform", transform)
        object.__setattr__(problem, "route", route)
        with pytest.raises(BackendError, match=r"Cartesian nonlinear.*geometry"):
            backend._nonlinear(
                problem.initial.field0[None],
                bundle,
                np.ones((1, *problem.shape), complex),
            )
    finally:
        backend.close()


@pytest.mark.parametrize("radius", [1.4e308, float(np.nextafter(0.0, 1.0))])
def test_free_grid_names_unrepresentable_extents(radius):
    with pytest.raises(GridError, match="extent or spacing"):
        FreeGrid(radius, 4, 1.0, 6, collar_fraction=0.5)


def test_cartesian_widths_reject_nonnumeric_samples():
    with pytest.raises(GridError, match="must be real"):
        cartesian_fluence_widths([0.0, 1.0], [0.0, 1.0], ["bad"] * 4)


def test_cartesian_edge_peak_has_no_inscribed_threshold_width():
    widths = cartesian_fluence_widths([0.0, 1.0], [0.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    assert widths[0] == 0.0
    assert np.all(np.isnan(widths[1:]))


def test_cartesian_widths_report_nonrepresentable_moment_arithmetic():
    axis = np.array([-8e307, 0.0, 8e307])
    fluence = np.zeros(9)
    fluence[0] = 1.0
    # Every input is finite. The far, zero-fluence corner's radius overflows;
    # both NumPy diagnostics remain visible and the public function raises.
    with (
        pytest.warns(RuntimeWarning, match="overflow|invalid"),
        pytest.raises(GridError, match="moments are not representable"),
    ):
        cartesian_fluence_widths(axis, axis, fluence)
