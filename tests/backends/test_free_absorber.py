# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check Cartesian collar derivatives and step-count independence."""

import dataclasses
from functools import partial

import numpy as np
import pytest
from scipy.linalg import expm

from backends.test_radial_absorber import LENGTH, problem_for, rk_bound
from backends.test_radial_nonlinear import derivative
from bounds import gamma
from solprop.backends.spatial import collar_field_fraction
from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, Absorber
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from spatial_records import record_metric


@pytest.fixture(params=["field_real", "field_analytic", "envelope"])
def free_absorber(request):
    constructor = (
        partial(EnvelopeGrid.from_limits, thg=False)
        if request.param == "envelope"
        else FieldGrid.from_limits
    )
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(1e-3, 8, 1.2e-3, 6, window_factor=0.5)
    boundary = Absorber(
        np.zeros(grid.n_omega),
        np.zeros(grid.n_t),
        np.zeros(grid.n_to),
        grid.active_mask,
        5 * LENGTH,
    )
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    field = np.exp(-((space.r / 0.8e-3) ** 2))
    start = np.zeros((space.n_columns, 2, grid.n_omega), complex)
    start[:, 0, index] = np.fft.fft2(field.reshape(space.n_y, space.n_x)).ravel()
    start[:, 1, index] = 0.7j * start[:, 0, index]
    # Build the DFT from its geometric sums, independently of FFT calls in
    # either backend, so the matrix exponential is a distinct absorber oracle.
    ix, iy = (
        np.tile(np.arange(space.n_x), space.n_y),
        np.repeat(np.arange(space.n_y), space.n_x),
    )
    forward = np.exp(
        -2j
        * np.pi
        * (
            ix[:, None] * ix[None, :] / space.n_x
            + iy[:, None] * iy[None, :] / space.n_y
        )
    )
    inverse = np.conj(forward.T) / space.n_columns
    rate = TEMPORAL_ALPHA_LENGTH * (1 - space.window) / boundary.reference_length
    absorption = forward @ (-0.5 * rate[:, None] * inverse)
    wave = grid.omega[index] / C
    phase = 1j * space.kperp_sq / (np.sqrt(wave**2 - space.kperp_sq) + wave)
    responses = (
        [KerrResponse(0.0, thg=False)] if request.param == "field_analytic" else []
    )
    return grid, space, boundary, start, index, absorption, phase, responses


def test_cartesian_absorber_is_the_physical_multiplication_rate(
    free_absorber, spatial_backend_class, request
):
    grid, space, _, start, index, absorption, _, _ = free_absorber
    bound = gamma(128 * (space.n_columns + grid.n_to)) * np.max(
        np.abs(absorption) @ np.abs(start[:, :, index])
    )
    expected = absorption @ start[:, :, index]
    problem = problem_for(free_absorber, 8)
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    actual = derivative(spatial_backend_class, problem, grid)
    assert np.max(np.abs(actual[:, :, index] - expected)) < bound
    record_metric(
        request,
        "Cartesian spatial absorber derivative",
        request.node.callspec.params["free_absorber"],
        np.max(np.abs(actual[:, :, index] - expected)),
        bound,
        spatial_backend_class.name,
    )


@pytest.mark.parametrize("steps", [8, 16, 32])
def test_cartesian_absorber_has_no_per_step_window_loss(
    free_absorber, spatial_backend_class, steps, request
):
    grid, space, _, start, index, absorption, phase, _ = free_absorber
    linear_norm = np.max(np.abs(phase))
    # The physical absorber is unitarily similar to diag(-alpha/2), so its
    # spectral norm is known without measuring a matrix norm.
    absorber_norm = TEMPORAL_ALPHA_LENGTH / (10 * LENGTH)
    bound = rk_bound(linear_norm, absorber_norm, steps)
    bound += gamma(256 * (space.n_columns + grid.n_to)) * steps
    bound += gamma(64) * LENGTH * grid.omega[index] / C
    assert 0 < bound < 0.01
    expected = expm(LENGTH * (np.diag(phase) + absorption)) @ start[:, :, index]
    problem = problem_for(free_absorber, steps)
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        result = backend.propagate_to(LENGTH)
        actual = backend.observe().require_field()[0]
        defect = np.linalg.norm(actual[:, :, index] - expected) / np.linalg.norm(
            start[:, :, index]
        )
        assert defect < bound, (defect, bound)
        assert result.n_steps == steps
        record_metric(
            request,
            "Cartesian spatial absorber matrix exponential (relative)",
            f"{request.node.callspec.params['free_absorber']}, steps={steps}",
            defect,
            bound,
            spatial_backend_class.name,
        )

    finally:
        backend.close()


def test_cartesian_collar_diagnostic_reports_illuminated_window(
    free_absorber, spatial_backend_class, caplog, request
):
    grid, space, _, start, _, _, _, _ = free_absorber
    problem = problem_for(free_absorber, 8)
    field = np.exp(-((space.r / 0.8e-3) ** 2))
    expected = np.sum(field[space.window < 1] ** 2) / np.sum(field**2)
    bound = gamma(128 * (space.n_columns + grid.n_to))
    assert expected > problem.stepper.controller.rtol
    measured = collar_field_fraction(problem, start)
    assert abs(measured - expected) < bound
    record_metric(
        request,
        "Cartesian collar energy fraction",
        request.node.callspec.params["free_absorber"],
        abs(measured - expected),
        bound,
        spatial_backend_class.name,
    )

    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.observe()
        assert "spatial collar contains" in caplog.text
        assert f"{measured:.6e}" in caplog.text
    finally:
        backend.close()


def test_cartesian_split_step_temporal_absorber_reconstructs_physical_field(
    free_absorber, spatial_backend_class, request
):
    grid, space, boundary, start, _, _, _, _ = free_absorber
    # A complex transverse spectrum is essential: a displaced asymmetric
    # profile distinguishes transverse inverse -> temporal irfft from its
    # invalid reverse order. The physical field remains real.
    shifted = np.roll(
        np.exp(-((space.r / 0.7e-3) ** 2)).reshape(space.n_y, space.n_x), 1, axis=1
    )
    spectral = np.fft.fft2(shifted).ravel()
    start = start * spectral[:, None, None]
    rate = (1 + np.cos(2 * np.pi * np.arange(grid.n_t) / grid.n_t)) / LENGTH
    boundary = dataclasses.replace(boundary, temporal_rate=rate)
    case = (grid, space, boundary, start, *free_absorber[4:])
    problem = problem_for(case, 8)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, absorber_scheme="split_step"),
        transform=dataclasses.replace(problem.transform, node_chunk=7),
    )
    dz = LENGTH / 8
    bound = gamma(128 * (space.n_columns + grid.n_t)) * np.max(np.abs(start))
    physical = np.fft.ifft2(
        start.reshape(space.n_y, space.n_x, 2, grid.n_omega), axes=(0, 1)
    )
    time = (
        np.fft.ifft(physical, axis=-1)
        if grid.kind == "envelope"
        else np.fft.irfft(physical, n=grid.n_t, axis=-1)
    )
    time *= np.exp(-0.5 * dz * rate)
    temporal = (
        np.fft.fft(time, axis=-1)
        if grid.kind == "envelope"
        else np.fft.rfft(time, axis=-1)
    )
    expected = (
        np.fft.fft2(temporal, axes=(0, 1)).reshape(start.shape) * grid.active_mask
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if backend.name == "native":
            # A zero-length trial loads the unmodified field into advanced;
            # finish then exercises precisely the once-per-step absorber.
            backend._trial_step(0.0, 0.0)
            backend._native_state.finish(dz)
            actual = backend._native_state.field_to_host().reshape(start.shape)
        else:
            actual = backend._finish(problem.initial.field0.copy(), dz)[0]
        assert np.max(np.abs(actual - expected)) < bound
        record_metric(
            request,
            "Cartesian split-step temporal absorber",
            request.node.callspec.params["free_absorber"],
            np.max(np.abs(actual - expected)),
            bound,
            spatial_backend_class.name,
        )

    finally:
        backend.close()
