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


def axial_root(medium, omega, space, z):
    """Return ``Re k_z`` of every column on the active band, all propagating."""
    square = (omega * medium.index(omega, z) / C) ** 2
    root = np.real(np.sqrt(square[None] - space.k_r[:, None] ** 2))
    assert np.all(root > 0)
    return root


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
            # SEP 34 section 4.4: the forward amplitude carries the endpoint factor
            # sqrt(Re k_z(0) / Re k_z(L)) on propagating bins (all of them here). Its
            # log-ratio is resolved to a relative 1e-9 of itself (the translation
            # test's derivation), added to the phase budget absolutely.
            amplitude = -0.5 * np.log(
                axial_root(medium, omega, space, length)
                / axial_root(medium, omega, space, 0.0)
            )
            phase = phase + amplitude
            phase_bound += 1e-9 * float(np.max(np.abs(amplitude)))
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


def test_radial_gradient_flow_conserves_axial_flux_and_resumes(spatial_backend_class):
    length = 1e-4
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, length))
    initial = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 1e-4
    ).build(grid, space)
    omega = grid.omega[grid.active_mask]
    assert np.all(np.imag(medium.index(omega, 0.0)) == 0)
    assert np.all(np.imag(medium.index(omega, length)) == 0)

    def admittance(z):
        # Y_z = eps0 c^2 Re k_z / omega; the common eps0 c^2 / omega cancels per bin.
        return axial_root(medium, omega, space, z)

    problem = assemble(
        grid,
        None,
        [],
        initial,
        spatial_grid=space,
        medium=medium,
        frame=dataclasses.replace(Frame.comoving(grid), beta0=0.0, beta1=0.0),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length / 2, length],
        dz_max=length,
        dz_init=length / 8,
        schedule=FixedSchedule.uniform(0.0, length, length / 8),
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length / 2)
        middle = backend.observe().require_field().copy()
        backend.propagate_to(length)
        straight = backend.observe().require_field().copy()
        backend.set_initial(middle, None, z=length / 2)
        backend.propagate_to(length)
        resumed = backend.observe().require_field()
    finally:
        backend.close()
    entrance = admittance(0.0) * np.abs(initial[:, 0][:, grid.active_mask]) ** 2
    exit_flux = admittance(length) * np.abs(straight[0, :, 0][:, grid.active_mask]) ** 2
    # Lossless argon: Y_z |E|^2 is invariant under the represented flow up to twice
    # the endpoint term's representation error (relative 1e-9 of a log-ratio below
    # 2e-5 here) and the eight diagonal steps' rounding, 256 units on |E|^2.
    exponent = float(np.max(np.abs(np.log(admittance(length) / admittance(0.0)))))
    budget = (1e-9 * exponent + gamma(256)) * float(np.max(entrance))
    assert np.max(np.abs(exit_flux - entrance)) <= budget
    assert np.max(np.abs(exit_flux - entrance)) > 0 or exponent == 0
    # Steps and a resume compose the cumulative endpoint term: the resumed march
    # ends where the straight one does, to the rounding of four diagonal steps.
    assert np.max(np.abs(resumed - straight)) <= gamma(64) * np.max(np.abs(straight))
    # Without the endpoint factor, |E|^2 itself would be conserved; it is not.
    plain = np.abs(straight[0, :, 0][:, grid.active_mask]) ** 2
    entrance_plain = np.abs(initial[:, 0][:, grid.active_mask]) ** 2
    assert np.max(np.abs(plain - entrance_plain)) > 100 * budget
