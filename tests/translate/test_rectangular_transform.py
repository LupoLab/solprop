# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify rectangular geometry scaling and the three-parameter z table."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec

from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid
from solprop.linear import DEFAULT_LOSS_POLICY, Frame, log_power_weight
from solprop.modes import MarcatiliMode, RectMode
from solprop.modes.quadrature import mode_matrix, taper_scale, transverse_quadrature
from solprop.profiles import Linear, PressureGradient
from solprop.translate.linop import _operator_sampler, build_phi, delta_phi
from solprop.translate.modal import _radius_kinks, multimode_transform
from solprop.translate.problem import DEFAULT_NODE_BUDGET, _transform


def test_rectangular_shared_taper_rescales_matrices_and_tracks_kinks():
    path = Linear(50e-6, 25e-6, 0.1)
    mode = RectMode(path, path)
    rule = transverse_quadrature([mode], n_r=24, n_theta=24)
    entrance = mode_matrix([mode], rule)
    for z in (0.025, 0.1):
        direct = mode_matrix([mode], rule, z)
        scaled = entrance.rescaled(taper_scale(mode, z))
        # Identical dimensionless nodes leave only width and normalisation rounding.
        assert_allclose(scaled.synthesis, direct.synthesis, rtol=2e-13, atol=0)
        assert_allclose(scaled.projection, direct.projection, rtol=2e-13, atol=0)
    assert _radius_kinks(mode) == tuple(path.breakpoints)
    grid = EnvelopeGrid.from_limits(0.1, 400e-9, (350e-9, 450e-9), 50e-15)
    transform = multimode_transform([mode], rule, grid, z_max=0.1)
    assert transform.n_modes == 1 and transform.n_pol == 2


def test_rectangular_three_varying_parameters_use_z_table():
    mode = RectMode(
        Linear(50e-6, 45e-6, 0.1),
        Linear(12.5e-6, 11e-6, 0.1),
        "Ar",
        PressureGradient.two_point(0.1, 5.0, 2.0),
    )
    assert sum(not path.is_constant for path in mode.parameter_paths.values()) == 3
    grid = EnvelopeGrid.from_limits(0.1, 400e-9, (350e-9, 450e-9), 50e-15)
    frame = Frame.comoving(grid)
    table = build_phi(mode, grid, frame, z_max=0.1)
    assert table.phi_representation == "table"
    sampler = _operator_sampler((mode,), grid, frame, 1, DEFAULT_LOSS_POLICY, None)
    # Builder target 1e-10 plus independent integral rounding, as in the taper test.
    z1, z2 = (0.017, 0.083)
    reference, _ = quad_vec(sampler, z1, z2, epsabs=1e-13)
    active = grid.omega[grid.active_mask]
    reference[..., grid.active_mask] -= 0.5 * (
        log_power_weight(mode, active, z2) - log_power_weight(mode, active, z1)
    )
    # The oracle includes the closed-form log term; the bound is unchanged by it.
    assert (
        np.max(np.abs(delta_phi(table, z1, z2) - reference)) / np.max(np.abs(reference))
        < 2e-10
    )


def test_a_non_degenerate_partner_is_refused_by_its_trait_not_its_class():
    grid = EnvelopeGrid.from_limits(0.1, 400e-9, (350e-9, 450e-9), 50e-15)
    mode = MarcatiliMode(125e-6)
    # Everything keeps the degenerate default unless it declares otherwise.
    assert getattr(mode, "polarisation_degenerate", True) is True
    assert RectMode(50e-6, 12.5e-6).polarisation_degenerate is False
    # Declare it on a capillary mode instead: the refusal reads the trait, so the
    # translation layer needs no concrete mode class to make it.
    object.__setattr__(mode, "polarisation_degenerate", False)
    with pytest.raises(TranslationError, match="distinct effective indices"):
        _transform(
            [mode],
            grid,
            None,
            z_max=0.1,
            n_pol=2,
            components=None,
            node_budget=DEFAULT_NODE_BUDGET,
            margin=0.05,
            tol=1e-10,
        )
