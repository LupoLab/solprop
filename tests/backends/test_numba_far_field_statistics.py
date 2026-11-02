# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s far-field statistics in a run, beside the reference's (SEP 23).

The launch's statistics against ``numpy_ref``, the planar refusal and a whole run;
``test_numba_far_field_kernels.py`` holds its pass, and ``test_far_field_statistics.py``
what every backend shares.
"""

import dataclasses

import numpy as np
import pytest

from backends.planar_cases import planar_case
from backends.test_far_field_statistics import (
    ALL_SIX,
    entrance,
    far_spec,
    free_simulation,
    gaussian,
)
from bounds import assert_within, gamma
from solprop.backends import select
from solprop.boundaries import absorber
from solprop.exceptions import OutputError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec

# As in ``test_far_field_statistics.py``: the launches here are noise-free in tight
# time windows, and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")


@pytest.fixture(scope="module")
def grid():
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (700e-9, 900e-9), 100e-15, thg=False)


def test_numba_cpu_forms_the_far_field_statistics_through_the_reference_pass(grid):
    space = FreeGrid(3e-4, 24, 3e-4, 16, window_factor=0.1)
    simulation = free_simulation(
        grid,
        space,
        gaussian(60e-6),
        medium=SpatialMedium(1 + 1e-6j),
        absorber=absorber(grid),
    )
    values = {
        name: entrance(simulation, name)[0] for name in ("numpy_ref", "numba_cpu")
    }
    assert set(values["numba_cpu"]) == set(ALL_SIX)
    for name in ALL_SIX:
        # One host pass over the same state on both: the summation bound.
        assert_within(
            values["numba_cpu"][name],
            values["numpy_ref"][name],
            gamma(4 * (space.n_columns + 16)),
            what=name,
        )
        assert values["numpy_ref"][name].shape[0] == 1  # the batch axis


# The planar fixture's family warns about its own collar and fill, as in the planar
# backend tests; neither is what this refusal is about.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
def test_planar_geometry_is_refused_before_any_step():
    spec = StatsSpec(names=("energy", "beam_spectrum"))
    planar_grid, _, problem = planar_case("kerr", steps=0)
    problem = dataclasses.replace(problem, stats_spec=spec)
    with pytest.raises(OutputError, match="hybrid planar"):
        select("numba_cpu", problem, grid=planar_grid)


def test_a_run_records_every_far_field_statistic_at_every_plane(grid):
    space = FreeGrid(3e-4, 16, 3e-4, 16, window_factor=0.1)
    simulation = free_simulation(
        grid,
        space,
        gaussian(60e-6),
        save_planes=[5e-5, 1e-4],
        dz_max=5e-5,
        dz_init=5e-5,
        absorber=absorber(grid),
        spec=far_spec((*ALL_SIX, "energy"), periodic=("absorbed_power",), period=1),
        backend="numba_cpu",
    )
    output = simulation.run(progress="none")
    shapes = {
        "aperture_spectrum": (3, 2, grid.n_omega),
        "aperture_on_axis_spectrum": (3, 2, grid.n_omega),
        "region_spectrum": (3, 4, grid.n_omega),
        "beam_spectrum": (3, grid.n_omega),
        "absorbed_power": (3, 4),
    }
    for name, shape in shapes.items():
        assert np.asarray(output[f"stats/{name}"]).shape == shape, name
    assert np.asarray(output["step_stats/absorbed_power"]).shape[1:] == (4,)
    beam = np.asarray(output["stats/beam_spectrum"])
    # Parseval at every plane: 2 gamma_m of the total.
    assert_within(
        np.sum(beam, axis=-1) * grid.d_omega,
        np.asarray(output["stats/energy"]),
        2 * gamma(space.n_columns * grid.n_omega + 16),
    )
