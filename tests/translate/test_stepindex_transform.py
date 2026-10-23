# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the frozen-profile transform and continuous step-index taper operator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec

from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid
from solprop.linear import DEFAULT_LOSS_POLICY, Frame, log_power_weight
from solprop.modes import StepIndexMode, transverse_quadrature
from solprop.profiles import Linear
from solprop.translate.linop import _operator_sampler, build_phi, delta_phi
from solprop.translate.modal import multimode_transform


def test_stepindex_single_mode_transform_and_taper_refusal():
    mode = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
    grid = EnvelopeGrid.from_limits(0.15, 835e-9, (400e-9, 1400e-9), 1e-12)
    rule = transverse_quadrature([mode], n_r=24)
    transform = multimode_transform([mode], rule, grid, z_max=0.15)
    assert transform.n_modes == 1
    assert transform.n_pol == 2
    # A constant area table has only arithmetic rounding from interpolation.
    assert_allclose(
        [float(transform.aeff(z)) for z in (0.0, 0.075, 0.15)],
        mode.aeff(),
        rtol=2e-15,
        atol=0,
    )
    taper = StepIndexMode(Linear(1.25e-6, 1.4e-6, 0.15), profile_wavelength=835e-9)
    with pytest.raises(TranslationError, match="not scale invariant"):
        multimode_transform(
            [taper], transverse_quadrature([taper], n_r=24), grid, z_max=0.15
        )


@pytest.mark.slow
def test_stepindex_taper_has_a_continuous_tabulated_operator():
    # Two centimetres, not fifteen: the panel structure of the Phi table and the
    # root finding at its Chebyshev nodes are what this test exercises, and
    # neither depends on the length.
    z_max = 0.02
    mode = StepIndexMode(Linear(1.25e-6, 1.4e-6, z_max), profile_wavelength=835e-9)
    grid = EnvelopeGrid.from_limits(z_max, 835e-9, (500e-9, 1100e-9), 50e-15)
    frame = Frame.comoving(grid)
    table = build_phi(mode, grid, frame, z_max=z_max, representation="table")
    assert table.phi_representation == "table"
    sampler = _operator_sampler((mode,), grid, frame, 1, DEFAULT_LOSS_POLICY, None)
    # Restated for the shortened taper on the same fraction of the length as
    # before: measured 4.073e-14 against the table's own 1e-10 target.
    z1, z2 = (0.1 * z_max, 0.85 * z_max)
    reference, _ = quad_vec(sampler, z1, z2, epsabs=1e-13)
    active = grid.omega[grid.active_mask]
    reference[..., grid.active_mask] -= 0.5 * (
        log_power_weight(mode, active, z2) - log_power_weight(mode, active, z1)
    )
    # The oracle includes the closed-form log term; the bound is unchanged by it.
    assert (
        np.max(np.abs(delta_phi(table, z1, z2) - reference)) / np.max(np.abs(reference))
        < 2e-13
    )
    omega = np.asarray([grid.omega0])
    positions = np.linspace(0.0, z_max, 61)
    neff = np.asarray([complex(mode.neff(omega, float(one))[0]) for one in positions])
    # The interval itself, not two adjacent doubles: a switch to another radial
    # root would move n_eff by about 1e-3, where the taper's own variation over
    # sixty-one samples is 7.377e-5 per step and smooth (second differences
    # measured 4.146e-7).
    assert np.all(np.diff(np.real(neff)) > 0)
    assert float(np.max(np.abs(np.diff(neff, 2)))) < 1e-6
