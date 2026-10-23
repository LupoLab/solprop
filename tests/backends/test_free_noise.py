# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Advance modified spatial noise into the nonlinear stage on each backend."""

from functools import partial

import numpy as np
import pytest

from bounds import gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.noise import spatial_noise
from solprop.nonlinear.responses import KerrResponse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.problem import assemble
from spatial_records import record_metric


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_cartesian_modified_noise_enters_the_rotated_nonlinear_field(
    spatial_backend_class, kind, n_pol, request, monkeypatch
):
    length = 1e-4
    constructor = (
        FieldGrid.from_limits
        if kind == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(length, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    noise = spatial_noise(
        grid, space, np.random.default_rng(94), n_pol=n_pol, model="modified"
    )
    initial = np.zeros_like(noise.spectrum)
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        initial,
        spatial_grid=space,
        medium=SpatialMedium(),
        noise=noise,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_init=length / 8,
        dz_max=length,
    )
    position = length / 2
    # Conjugate-rationalise the vacuum difference independently of the stored
    # coefficients. Bound both temporal/transverse transforms and cubic algebra,
    # plus subtraction of the optical carrier in the production phase.
    bound = gamma(128 * (space.n_columns + grid.n_to))
    bound += gamma(64) * position * float(grid.omega.max()) / C
    wave = grid.omega[grid.active_mask] / C
    q = wave[None, :] ** 2 - space.kperp_sq[:, None]
    phase = position * space.kperp_sq[:, None] / (np.sqrt(q) + wave[None, :])
    rotated = np.zeros_like(initial)
    rotated[..., grid.active_mask] = noise.spectrum[..., grid.active_mask] * np.exp(
        1j * phase[:, None, :]
    )
    reference = NumpyRefBackend(problem, grid=grid)
    backend = spatial_backend_class(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(position)
        expected = reference._nonlinear(
            rotated[None],
            reference._stage_scalars(position, position),
            np.ones(problem.shape, complex),
        )[0]
        backend.set_initial(problem.initial.field0, noise)
        # Zero distance isolates one nonlinear evaluation while still exercising
        # each backend's real stage-input/noise path at nonzero absolute z.
        captured = []
        if isinstance(backend, NumpyRefBackend):
            nonlinear = backend._nonlinear

            def capture(lab, bundle, inverse):
                stage = nonlinear(lab, bundle, inverse)
                if not captured:
                    captured.append(stage.copy())
                return stage

            monkeypatch.setattr(backend, "_nonlinear", capture)
        backend._trial_step(position, 0.0)
        if isinstance(backend, NumpyRefBackend):
            actual = captured[0].reshape(expected.shape)
        elif backend.name == "native":
            actual = backend._native_state.stage_to_host(0).reshape(expected.shape)
        else:
            actual = backend._stages[0].reshape(expected.shape)
        scale = np.max(np.abs(expected))
        assert scale > 0
        defect = np.max(np.abs(actual - expected)) / scale
        assert defect < bound, (defect, bound)
        record_metric(
            request,
            "Cartesian modified noise nonlinear stage (relative)",
            f"{kind}, p={n_pol}",
            defect,
            bound,
            backend.name,
        )
    finally:
        backend.close()
        reference.close()
