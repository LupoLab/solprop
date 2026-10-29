# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep large radial envelope transforms eligible for device BLAS."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.problem import assemble

from solprop_native import NativeBackend, build_info


@pytest.mark.parametrize(
    ("n_r", "n_pol", "chunk", "eligible"),
    [
        (256, 1, 256, True),
        (256, 1, 73, True),
        (256, 2, 256, False),
        (128, 1, 128, False),
    ],
)
def test_radial_short_envelope_uses_device_blas(
    n_r: int, n_pol: int, chunk: int, eligible: bool
) -> None:
    info = build_info()
    if info.device_backend == "none" or info.gemm_provider != "HYBRID":
        pytest.skip("default device HYBRID dispatch qualification")
    if not info.tpl_blas_enabled:
        pytest.skip("device vendor BLAS is disabled")
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 600e-15, thg=False)
    assert 256 <= grid.n_to < 1024
    space = RadialGrid(6e-4, n_r, window_factor=0)
    problem = assemble(
        grid,
        None,
        [],
        np.zeros((n_r, n_pol, grid.n_omega), dtype=complex),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-16),
        save_planes=[1e-4],
        dz_init=1e-5,
        dz_max=1e-5,
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )
    backend = NativeBackend(problem, grid=grid)
    try:
        metadata = backend.metadata()
        expected = "kokkosblas-tpl" if eligible else "custom"
        assert metadata["synthesis_provider"] == expected
        assert metadata["projection_provider"] == expected
        assert backend.memory_ledger()["radial_blas_buffers"] == 0
    finally:
        backend.close()


@pytest.mark.parametrize("n_r", [256, 512])
@pytest.mark.parametrize("chunk", [73, 156])
@pytest.mark.parametrize("family", ["analytic", "real", "envelope"])
def test_radial_partial_chunks_preserve_the_reference_source(
    n_r: int, chunk: int, family: str
) -> None:
    """Check both filament sizes, short tails and unchanged transform families."""
    grid = (
        EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
        if family == "envelope"
        else FieldGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    )
    space = RadialGrid(1e-3, n_r)
    start = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4
    ).build(grid, space, medium=SpatialMedium())
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=family == "real")],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        density=1.0,
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )
    # The established radial source ladder: FFT depth and two length-n_r dot
    # products, with an absolute majorant for cancellation at dark bins.
    eps = np.finfo(float).eps
    bound = 16 * eps * math.ceil(math.log2(grid.n_to)) + 2 * (
        n_r * eps / (1 - n_r * eps)
    )
    assert bound > 0
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage]
        expected = np.asarray(
            reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
                reference._require_state(),  # pyright: ignore[reportPrivateUsage]
                reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
                np.ones(problem.shape, complex),
            )[0]
        )
    finally:
        reference.close()
    backend = NativeBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        # A host analytic field synthesises the retained spectrum; the others, and a
        # device, keep their time-domain products.
        info = build_info()
        retained = (
            family == "analytic"
            and info.device_backend == "none"
            and info.gemm_provider != "CUSTOM"
        )
        assert (
            str(backend.metadata()["synthesis_provider"]).endswith(
                "-retained-interleaved-real"
            )
            == retained
        )
        assert (ledger["retained_synthesis"] > 0) == retained
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        state = backend._native_state  # pyright: ignore[reportPrivateUsage]
        actual = np.asarray(state.stage_to_host(0)).reshape(problem.shape[1:])
        assert np.isfinite(actual).all()
        scale = float(np.max(np.abs(expected)))
        assert scale > 0
        np.testing.assert_allclose(actual, expected, rtol=0, atol=bound * scale)
        assert np.count_nonzero(actual[..., ~grid.active_mask]) == 0
        backend.set_initial(np.zeros_like(problem.initial.field0), None)
        backend._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        assert np.count_nonzero(state.stage_to_host(0)) == 0
        assert backend.memory_ledger() == ledger
    finally:
        backend.close()
