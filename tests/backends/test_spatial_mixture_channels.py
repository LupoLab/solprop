# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-channel propagation and population statistics in both spatial measures."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.test_mixture_ladder import NAMES, run
from mixture_cases import mixture_case
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule
from solprop.translate.problem import assemble


@pytest.mark.parametrize("geometry", ["radial", "free"])
@pytest.mark.parametrize("thg", [False, True])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_spatial_channels_and_all_population_statistics(
    geometry, thg, n_pol, spatial_equivalence_backend_class
):
    mixture, grid, _, responses, original = mixture_case()
    space = (
        RadialGrid(6e-4, 8, collar_fraction=0.0)
        if geometry == "radial"
        else FreeGrid(3e-4, 4, 4e-4, 4, collar_fraction=0.0)
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9),
        80e-6,
        jones=(1,) if n_pol == 1 else (1, 0.5j),
    )
    responses = tuple(
        dataclasses.replace(response, thg=thg)
        if isinstance(response, KerrResponse)
        else response
        for response in responses
    )
    problem = assemble(
        grid,
        None,
        responses,
        pulse.build(grid, space, medium=SpatialMedium(mixture)),
        spatial_grid=space,
        medium=SpatialMedium(mixture),
        controller=original.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / 8,
        schedule=FixedSchedule.uniform(0, grid.length, grid.length / 8),
    )
    # Two exposure scans and two accumulated current integrals cost at most
    # 64 eps per temporal sample; spatial synthesis/projection adds 128 eps
    # per node. Sum over the eight seven-stage steps, before either run.
    bound = (
        (64 * grid.n_to * 2 + 128 * problem.transform.n_nodes)
        * 7
        * 8
        * np.finfo(float).eps
    )
    assert bound > 0.0
    oracle = run(NumpyRefBackend, problem, grid)
    actual = run(spatial_equivalence_backend_class, problem, grid)
    assert actual[1].n_steps == oracle[1].n_steps == 8
    assert np.linalg.norm(actual[0] - oracle[0]) / np.linalg.norm(oracle[0]) < bound
    for values, reference in zip(actual[2:], oracle[2:], strict=True):
        for name in NAMES:
            assert_allclose(
                values[name],
                reference[name],
                rtol=bound,
                atol=bound if name == "species_ionisation" else 0.0,
            )
