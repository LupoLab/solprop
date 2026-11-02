# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s kernel regime on a radial run."""

from backends import test_radial_runs
from solprop.backends.numba_cpu import fft
from solprop.backends.numba_cpu.backend import NumbaCpuBackend
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.problem import assemble

time_grid = test_radial_runs.time_grid
radial_case = test_radial_runs.radial_case


def test_a_radial_run_keeps_the_general_kernel_regime(radial_case):
    """The lower modal kernel cutoff was measured on fibre transforms only.

    Taking "not Cartesian" for "fibre" would give a radial run with more than one node
    the 262 144-element modal cutoff that no radial measurement backs; the fibre
    policies are keyed on the absence of a spatial geometry instead.
    """
    grid, space, pulse = radial_case
    problem = assemble(
        grid,
        None,
        [],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    assert problem.transform.geometry is not None
    assert problem.transform.n_nodes > 1
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        metadata = backend.metadata()
    finally:
        backend.close()
    assert metadata["kernel_regime_elements"] == fft.REGIME_ELEMENTS
    assert metadata["kernel_regime_elements"] != fft.MODAL_REGIME_ELEMENTS
    assert metadata["plasma_fraction_reuse"] is False
