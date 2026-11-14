# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify spatial execution choices without changing numerical contracts."""

import numpy as np
import pytest

from backends.spatial_cases import make_problem
from backends.test_radial_nonlinear import derivative
from bounds import gamma
from legacy_geometry import LEGACY_GRID
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from spatial_records import record_metric

RADIAL_BLAS_CASES = [
    (256, 1, 256, 300e-15),
    (256, 2, 256, 300e-15),
    (256, 1, 73, 300e-15),
    (128, 1, 128, 300e-15),
    (256, 1, 256, 150e-15),
    (384, 1, 384, 600e-15),
]
"""Shapes on both sides of the packed real BLAS route's eligibility: ``(n_r, n_pol,
chunk, duration)``."""


def radial_blas_case(n_r, n_pol, chunk, duration):
    """Return the grid, the Kerr problem and the node chunk of one radial BLAS shape."""
    grid = EnvelopeGrid.from_limits(
        1e-4, 800e-9, (600e-9, 1.2e-6), duration, thg=False, **LEGACY_GRID
    )
    space = RadialGrid(6e-4, n_r, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        jones=(1 + 0j,) if n_pol == 1 else (1 + 0j, 0.5j),
    )
    problem, run_request = make_problem(
        grid,
        space,
        # make_problem's medium is the vacuum.
        pulse.build(grid, space, medium=SpatialMedium()),
        [KerrResponse(1e-22, thg=False)],
    )
    return grid, problem, run_request, ExecutionOptions(node_chunk=chunk)


@pytest.mark.parametrize(("n_r", "n_pol", "chunk", "duration"), RADIAL_BLAS_CASES)
def test_radial_real_blas_preserves_transform_and_memory_contract(
    spatial_backend_class, n_r, n_pol, chunk, duration, request
):
    grid, problem, run_request, options = radial_blas_case(n_r, n_pol, chunk, duration)
    # Two radial reductions and temporal transforms, with the existing chunk
    # ladder's association allowance, fixed before either backend runs.
    bound = gamma(64 * n_r + 32 * grid.n_to)
    reference = derivative(NumpyRefBackend, problem, run_request, grid, options)
    actual = derivative(spatial_backend_class, problem, run_request, grid, options)
    np.testing.assert_allclose(
        actual, reference, rtol=0, atol=bound * np.max(np.abs(reference))
    )
    record_metric(
        request,
        "radial real BLAS derivative (relative)",
        f"r={n_r}, p={n_pol}, chunk={chunk}, time={grid.n_to}",
        np.max(np.abs(actual - reference)) / np.max(np.abs(reference)),
        bound,
        spatial_backend_class.name,
    )
    backend = spatial_backend_class(problem, run_request, grid=grid, options=options)
    try:
        metadata, ledger = backend.metadata(), backend.memory_ledger()
        if backend.name == "native":
            # Device builds keep their existing provider.
            packed = str(metadata["synthesis_provider"]).endswith("-packed-real")
            eligible = (
                metadata["device_backend"] == "none"
                and n_pol == 1
                and chunk == n_r
                and n_r >= 256
                and grid.n_to >= 256
                and 22 * 16 * n_r * grid.n_to <= 64 * 1024 * 1024
            )
            assert packed == eligible
            if packed:
                assert n_pol == 1 and chunk == n_r
                assert ledger["radial_blas_buffers"] == 32 * n_r * grid.n_to
                assert 22 * 16 * n_r * grid.n_to <= 64 * 1024 * 1024
            else:
                assert ledger["radial_blas_buffers"] == 0
    finally:
        backend.close()
