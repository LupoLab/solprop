# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cartesian response order, physical reality and separable prefactor oracles."""

import dataclasses
from functools import partial

import numpy as np
import pytest

from backends.test_radial_nonlinear import derivative, make_problem
from bounds import gamma
from cartesian_oracles import longitudinal_square
from solprop import fourier
from solprop.constants import EPS0, C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from spatial_records import record_metric


def gaussian_input(grid, space, *, waist, energy, jones):
    """Return a Gaussian beam's internal spectrum, scaled by the paraxial measure.

    A 30 fs Gaussian temporal pulse times a Gaussian profile of 1/e field radius
    ``waist``, Jones-normalised and confined to the active band, with
    ``eps0 c / 2 sum_j w_j |E_j|^2`` equal to ``energy``: an explicit amplitude, with
    no claim about the axial flux of its evanescent columns.
    """
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy).build(grid)
    x, y = np.meshgrid(space.x, space.y)
    profile = np.fft.fft2(np.exp(-(x**2 + y**2) / waist**2)).ravel()
    profile /= np.sqrt(
        0.5 * EPS0 * C * np.sum(space.spectral_weights * abs(profile) ** 2)
    )
    vector = np.asarray(jones, dtype=complex)
    vector = fourier.to_internal(vector / np.linalg.norm(vector))
    return profile[:, None, None] * vector[None, :, None] * temporal * grid.active_mask


@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("spatial_scale", [1.0, 0.001])
def test_cartesian_kerr_matches_physical_column_oracle(
    spatial_backend_class, family, n_pol, spatial_scale, request
):
    constructor = (
        partial(EnvelopeGrid.from_limits, thg=False)
        if family == "envelope"
        else FieldGrid.from_limits
    )
    grid = constructor(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-4 * spatial_scale, 9, 4e-4 * spatial_scale, 12, window_factor=0)
    # The oracle's subject is the response, not the launch: at the 0.001 scale the
    # beam is sub-wavelength and mostly evanescent, which a launch normalised in
    # axial flux refuses, so the input is an explicit array of known amplitude.
    initial = gaussian_input(
        grid,
        space,
        waist=80e-6 * spatial_scale,
        energy=3e-9 * spatial_scale**2,
        jones=(1, 0.4j) if n_pol == 2 else (1,),
    )
    response = KerrResponse(1e-22, thg=family == "field_real")
    # Bound each transform by its 1-norm, cubic products by 6*max|E|^3,
    # and the final prefactor by its active-band maximum. There are fewer
    # than 8 transverse/temporal transforms and 128 scalar operations in
    # this reference-to-compiled comparison. This is declared before either
    # backend derivative is evaluated.
    physical_spectrum = np.fft.ifft2(
        initial.reshape(space.n_y, space.n_x, n_pol, grid.n_omega), axes=(0, 1)
    ).reshape(initial.shape)
    fine = np.zeros(
        (
            space.n_columns,
            n_pol,
            grid.n_to if grid.kind == "envelope" else grid.n_to // 2 + 1,
        ),
        complex,
    )
    fine[..., grid.crop_index] = grid.pad_scale * physical_spectrum
    physical = (
        np.fft.ifft(fine, axis=-1)
        if grid.kind == "envelope"
        else np.fft.irfft(fine, n=grid.n_to, axis=-1)
    )
    transverse = space.kperp_sq[:, None]
    square = (grid.omega[None, :] / C) ** 2 - transverse
    prefactor = np.zeros((space.n_columns, grid.n_omega), complex)
    propagating = (square > 0) & grid.active_mask
    np.divide(
        -0.5j * grid.omega[None, :] ** 2 / (EPS0 * C**2),
        np.sqrt(np.maximum(square, 0)),
        out=prefactor,
        where=propagating,
    )
    clamped = (square <= 0) & grid.active_mask
    prefactor[clamped] = np.broadcast_to(-0.5j * grid.omega, square.shape)[clamped]
    bound = (
        gamma(128 * (space.n_columns + grid.n_to))
        * 6
        * EPS0
        * response.chi3
        * np.max(np.abs(physical)) ** 3
        * grid.n_to
        * space.n_columns
        * np.max(np.abs(prefactor))
        * grid.crop_scale
    )
    assert bound > 0
    # Preserve the legacy absolute budget above, including its old square formula.
    # Only the expected physical coefficient changes to the independently evaluated
    # unsquared-input target; a larger resolved prefactor cannot widen this gate.
    square = np.real(longitudinal_square(space, grid))
    propagating = (square > 0) & grid.active_mask
    prefactor[:] = 0
    np.divide(
        -0.5j * grid.omega[None, :] ** 2 / (EPS0 * C**2),
        np.sqrt(np.maximum(square, 0)),
        out=prefactor,
        where=propagating,
    )
    clamped = (square <= 0) & grid.active_mask
    prefactor[clamped] = np.broadcast_to(-0.5j * grid.omega, square.shape)[clamped]
    polarisation = (
        np.stack([response.evaluate(one, 1.0, kind=grid.kind) for one in physical])
        * grid.t_window_fine
    )
    temporal = (
        np.fft.fft(polarisation, axis=-1)
        if grid.kind == "envelope"
        else np.fft.rfft(polarisation, axis=-1)
    )
    projected = np.fft.fft2(
        temporal[..., grid.crop_index].reshape(
            space.n_y, space.n_x, n_pol, grid.n_omega
        ),
        axes=(0, 1),
    ).reshape(initial.shape)
    expected = projected * prefactor[:, None, :] * grid.crop_scale * grid.omega_window
    problem = make_problem(grid, space, initial, [response])
    # Force a remainder chunk, independently of the memory-derived default.
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    actual = derivative(spatial_backend_class, problem, grid)
    defect = np.max(np.abs(actual - expected))
    assert defect < bound, (defect, bound)
    record_metric(
        request,
        "Cartesian Kerr corrected cutoff, legacy bound (V/m²)",
        f"{family}, p={n_pol}, transverse scale={spatial_scale}",
        defect,
        bound,
        spatial_backend_class.name,
    )


def test_resolved_grazing_allows_backend_construction_and_a_zero_field_trial(
    spatial_backend_class,
):
    grid = FieldGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-7, 9, 4e-7, 12, window_factor=0)
    problem = make_problem(
        grid,
        space,
        np.zeros((space.n_columns, 1, grid.n_omega), complex),
        [KerrResponse(1e-22)],
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        error, advanced = backend._trial_step(0.0, 1e-7)
        # All terms are homogeneous in E. Cutoff correction cannot create a field.
        assert error == 0.0
        np.testing.assert_array_equal(
            backend._observed_field(advanced), problem.initial.field0
        )
    finally:
        backend.close()


@pytest.mark.parametrize("residual", [-(2.0**-30), 0.0, 2.0**-30])
def test_cartesian_linear_flow_resolves_both_sides_and_exact_cutoff(
    spatial_backend_class, residual
):
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-7, 9, 4e-7, 12, window_factor=0)
    initial = np.zeros((space.n_columns, 2, grid.n_omega), complex)
    initial[1, 0, grid.active_mask] = 1
    initial[1, 1, grid.active_mask] = 0.4j
    problem = make_problem(grid, space, initial, [], steps=1)
    original = problem.linear.separable
    assert original is not None
    # Equal high parts make q exactly the supplied representable residual. Its
    # three values test the inherited clamp at equality and either side. The
    # nonpositive attenuation here is sqrt(-q), well below the existing cap.
    data = dataclasses.replace(
        original,
        k_squared=np.full(grid.n_omega, space.kperp_sq[1], complex),
        kperp_sq_low=np.zeros(space.n_columns),
        k_squared_low=np.full(grid.n_omega, residual, complex),
        frame=np.zeros(grid.n_omega),
        spectral_rate=np.zeros(grid.n_omega),
    )
    problem = dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=data),
        responses=dataclasses.replace(problem.responses, separable=data),
    )
    root = np.sqrt(complex(residual))
    expected = initial.copy()
    expected[1] *= np.exp((-root.imag - 1j * root.real) * grid.length)
    # Two exponential paths and the final product cost fewer than 64 rounded
    # operations on O(1) amplitudes; the represented square root is exact here.
    bound = 64 * np.finfo(float).eps
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(grid.length)
        actual = backend.observe().require_field()[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)
    finally:
        backend.close()
