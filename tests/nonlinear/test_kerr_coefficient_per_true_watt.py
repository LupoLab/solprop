# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the full source-chain Kerr slope per true watt."""

from dataclasses import replace

import numpy as np
import pytest

from solprop import fourier
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import EPS0, C
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, StepIndexMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse
from solprop.stepper import PIController
from solprop.translate import stage
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import assemble


def true_watt_slope(problem, grid):
    # A periodic CW has no pulse edges. Remove only the temporal diagnostic
    # collar, whose spectral leakage otherwise contaminates a round-off test.
    problem = replace(
        problem, grids=replace(problem.grids, t_window_fine=np.ones(grid.n_to))
    )
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        state = np.zeros(problem.shape, complex)
        carrier = int(np.argmin(abs(grid.omega - grid.omega0)))
        state[..., carrier] = grid.n_t / (2 if grid.kind == "field" else 1)
        state /= np.sqrt(power_weight_at(problem, 0.0))
        derivative = backend._nonlinear(
            state, stage.scalars(problem, 0.0, 0.0), np.ones(problem.shape, complex)
        )
        amplitude = fourier.inverse(state[0, 0, 0], kind=grid.kind)
        response = fourier.inverse(derivative[0, 0, 0], kind=grid.kind)
        if grid.kind == "field":
            amplitude = fourier.analytic_signal(amplitude)
            response = fourier.analytic_signal(response)
        # The plateau at the centre excludes the existing temporal collar.
        centre = int(np.argmin(abs(grid.t)))
        return -float(np.imag(response[centre] / amplitude[centre])), carrier
    finally:
        backend.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("grid_class", [EnvelopeGrid, FieldGrid])
@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("pressure", [1.0, 20.0])
def test_capillary_kerr_coefficient_per_true_watt(grid_class, resolved, pressure):
    grid = grid_class.from_limits(1e-3, 835e-9, (700e-9, 1e-6), 400e-15)
    mode = MarcatiliMode(125e-6, "Ar", pressure, cladding=1.45, loss=False)
    chi3 = 2.2e-26  # Synthetic susceptibility; density=1 isolates the normalisation.
    problem = assemble(
        grid,
        [mode] if resolved else mode,
        [KerrResponse(chi3=chi3, thg=False)],
        np.zeros((1, 1, grid.n_omega), complex),
        controller=PIController(rtol=1e-10, atol=1e-18),
        save_planes=[grid.length],
        dz_init=1e-5,
        dz_max=grid.length,
        density=1.0,
        shock=True,
        quadrature=transverse_quadrature([mode], n_r=32, radial_rule="gauss_kronrod")
        if resolved
        else None,
    )
    measured, carrier = true_watt_slope(problem, grid)
    omega = grid.omega[carrier]
    index = float(np.real(mode.neff(np.array([omega]))[0]))
    expected = 3 * omega * chi3 / (4 * EPS0 * C**2 * index**2 * mode.aeff())
    # The full-chain FFT/quadrature reduction allowance, fixed before measurement.
    bound = 256 * np.finfo(float).eps * (grid.n_omega + problem.transform.n_nodes)
    np.testing.assert_allclose(measured, expected, rtol=bound, atol=0)


@pytest.mark.parametrize("fibre", [False, True])
def test_gnlse_kerr_coefficient_per_true_watt(fibre):
    wavelength, n2 = 835e-9, 2.6e-20
    grid = EnvelopeGrid.from_limits(1e-3, wavelength, (700e-9, 1e-6), 400e-15)
    mode = StepIndexMode(
        1.25e-6, core=1.45, cladding=1.0, profile_wavelength=wavelength
    )
    gamma = n2 * grid.omega0 / (C * mode.aeff())
    simulation = Simulation.from_gnlse(
        None if fibre else gamma,
        grid.length,
        None if fibre else [0.0, 0.0],
        lambda0=wavelength,
        grid=grid,
        fwhm=30e-15,
        peak_power=1.0,
        raman=False,
        shock=False,
        backend="numpy_ref",
        **({"fibre": mode, "n2": n2} if fibre else {}),
    )
    problem = simulation.translate()
    measured, _ = true_watt_slope(problem, grid)
    bound = 256 * np.finfo(float).eps * (grid.n_omega + problem.transform.n_nodes)
    np.testing.assert_allclose(measured, gamma, rtol=bound, atol=0)
