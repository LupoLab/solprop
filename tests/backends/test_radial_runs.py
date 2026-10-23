# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.backends.numba_cpu import fft
from solprop.backends.numba_cpu.backend import NumbaCpuBackend
from solprop.constants import EPS0, C
from solprop.fourier import energy_omega
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.linear import Frame
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.linop import delta_phi
from solprop.translate.problem import assemble
from solprop.translate.responses import prefactor_at

EPS = np.finfo(float).eps


@pytest.fixture(params=["field", "envelope"])
def time_grid(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    return constructor(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)


@pytest.fixture
def radial_case(time_grid):
    space = RadialGrid(0.6e-3, 32)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )
    return time_grid, space, pulse


@pytest.mark.parametrize("jones", [(1 + 0j,), (1 + 0j, 0j), (1 + 0j, 1j)])
def test_spatial_pulse_preserves_whole_beam_energy(radial_case, jones):
    grid, space, pulse = radial_case
    pulse = SpatialPulse(pulse.temporal, pulse.waist, jones=jones)
    # Positive sums: temporal spectral energy, transverse integration and one
    # amplitude rescaling. Their accumulated first-order errors are gamma below.
    operations = 8 * space.n_r + 4 * grid.n_omega
    bound = operations * EPS / (1 - operations * EPS)
    field = pulse.build(grid, space)
    measured = (
        0.5
        * EPS0
        * C
        * np.sum(
            space.spectral_weights[:, None]
            * energy_omega(field, grid.dt, kind=grid.kind)
        )
    )
    assert_allclose(measured, 3e-9, rtol=bound, atol=0)


def test_radial_operator_and_prefactor_match_vacuum_dispersion(radial_case):
    grid, space, pulse = radial_case
    wave = grid.omega[grid.active_mask] / C
    transverse = space.k_r[:, None] ** 2
    root = np.sqrt(wave**2 - transverse)
    expected = np.zeros((space.n_r, grid.n_omega), dtype=np.complex128)
    expected[:, grid.active_mask] = 1j * transverse / (root + wave)
    # sqrt and frame subtraction amplify roundoff by the unreduced optical wave
    # number; use that scale before evaluating the operator being qualified.
    bound = 64 * EPS * float(wave.max()) * grid.length
    problem = assemble(
        grid,
        None,
        [],
        pulse.build(grid, space),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    assert problem.transform.geometry is not None
    assert problem.transform.geometry.kind == "radial"
    assert not problem.flags.mode_averaged
    assert problem.flags.constant_linop
    assert_allclose(
        delta_phi(problem.linear, 0.0, grid.length)[0, :, 0],
        expected * grid.length,
        rtol=0,
        atol=bound,
    )
    wanted = np.zeros_like(expected)
    wanted[:, grid.active_mask] = (
        -0.5j
        * grid.omega[grid.active_mask] ** 2
        / (EPS0 * C**2 * root)
        * grid.crop_scale
    )
    # mu0=1/(eps0*c^2) and the direct/root forms contain fewer than 32 operations.
    assert_allclose(
        prefactor_at(problem.responses, 0.0)[:, 0], wanted, rtol=32 * EPS, atol=0
    )


@pytest.mark.parametrize("steps", [1, 8])
def test_radial_linear_propagation_on_each_backend(
    radial_case, spatial_backend_class, steps
):
    grid, space, pulse = radial_case
    start = pulse.build(grid, space)
    wave = grid.omega[grid.active_mask] / C
    transverse = space.k_r[:, None] ** 2
    phase = np.zeros((space.n_r, grid.n_omega), dtype=np.complex128)
    phase[:, grid.active_mask] = (
        1j * transverse / (np.sqrt(wave**2 - transverse) + wave) * grid.length
    )
    bound = 64 * EPS * (float(wave.max()) * grid.length + steps)
    expected = start * np.exp(phase)[:, None, :] * grid.active_mask
    problem = assemble(
        grid,
        None,
        [],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / steps,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(grid.length)
        actual = backend.observe().require_field()[0]
        assert summary.n_steps == steps
        assert_allclose(actual, expected, rtol=0, atol=bound * np.max(np.abs(expected)))
    finally:
        backend.close()


def test_spatial_launch_prepropagation_uses_the_exact_operator(radial_case):
    grid, space, pulse = radial_case
    frame = Frame.comoving(grid)
    distance = -0.4e-3
    launched = SpatialPulse(pulse.temporal, pulse.waist, propz=distance)
    operator = SpatialDispersion(SpatialMedium(), grid, frame).operator(space.k_r**2)
    # This verifies launch plumbing independently of the radial assembly; the
    # operator itself is qualified against the rationalised vacuum expression above.
    expected = pulse.build(grid, space) * np.exp(operator * distance)[:, None, :]
    assert_allclose(launched.build(grid, space), expected, rtol=8 * EPS, atol=0)


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
        pulse.build(grid, space),
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
