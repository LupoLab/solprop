# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise radial loss, gain, evanescence and cumulative density on each backend."""

import dataclasses
from contextlib import nullcontext
from functools import partial

import numpy as np
import pytest
from scipy.integrate import quad_vec

from hankel_oracles import gamma
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.linear import Frame
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.chebyshev import DEFAULT_TOL
from solprop.translate.problem import assemble
from spatial_records import record_metric


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("material", ["loss", "gain", "evanescent", "varying"])
def test_radial_medium_flows_match_an_independent_cumulative_oracle(
    spatial_backend_class, kind, material, request
):
    length = 1e-4
    constructor = (
        FieldGrid.from_limits
        if kind == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(length, 800e-9, (600e-9, 1.2e-6), 150e-15)
    radius = 1e-6 if material == "evanescent" else 6e-4
    space = RadialGrid(radius, 16, window_factor=0.0)
    medium = (
        SpatialMedium("Ar", pressure=Linear(0.5, 1.5, length))
        if material == "varying"
        else SpatialMedium(
            1.5 + 1e-6j
            if material == "loss"
            else 1.5 - 1e-6j
            if material == "gain"
            else 1.0
        )
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), radius / 6
    )
    initial = pulse.build(grid, space)
    frame = dataclasses.replace(Frame.comoving(grid), beta0=0.0, beta1=0.0)
    # The varying argon medium extrapolated the registry's fit; optmat's argon is
    # qualified over the band (SEP 3, Phase 4a), and no medium warns.
    diagnostic = nullcontext()
    with diagnostic:
        omega = grid.omega[grid.active_mask]

        def direct_rate(z):
            wave = omega * medium.index(omega, z) / C
            square = wave[None] ** 2 - space.k_r[:, None] ** 2
            root = np.sqrt(square)
            loss = np.where(
                np.real(square) > 0,
                np.imag(root),
                np.minimum(np.abs(np.imag(root)), 200.0),
            )
            return -loss - 1j * np.real(root)

        centre = direct_rate(length / 2)
        wave_max = float(np.max(omega * np.abs(medium.index(omega, length)) / C))
        # Fixed zero frame has no cancellation-conditioned small denominator.
        # The table's independent validation target, root arithmetic, eight exact
        # diagonal steps and the explicitly bounded quadrature determine this gate.
        phase_bound = (8 * DEFAULT_TOL + gamma(128)) * wave_max * length
        quadrature_budget = 1e-10
        if material == "varying":
            integral, quadrature_error = quad_vec(
                lambda z: direct_rate(float(z)) - centre,
                0.0,
                length,
                epsabs=quadrature_budget / 8,
                epsrel=0.0,
            )
            assert quadrature_error <= quadrature_budget
            phase = centre * length + integral
            phase_bound += quadrature_budget
        else:
            phase = centre * length
        relative_bound = np.expm1(phase_bound) + gamma(64)
        expected = np.zeros_like(initial)
        expected[..., grid.active_mask] = (
            initial[..., grid.active_mask] * np.exp(phase)[:, None]
        )
        absolute_bound = relative_bound * float(np.max(np.abs(expected)))
        problem = assemble(
            grid,
            None,
            [],
            initial,
            spatial_grid=space,
            medium=medium,
            frame=frame,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[length],
            dz_max=length,
            dz_init=length / 8,
            schedule=FixedSchedule.uniform(0.0, length, length / 8),
        )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length)
        actual = backend.observe().require_field()[0]
        assert np.max(np.abs(actual - expected)) <= absolute_bound
        record_metric(
            request,
            "material flow (V m)",
            f"{kind}, {material}",
            np.max(np.abs(actual - expected)),
            absolute_bound,
            spatial_backend_class.name,
        )
    finally:
        backend.close()
