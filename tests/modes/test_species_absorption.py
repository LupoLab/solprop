# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify first-order species absorption separately from total modal loss."""

from dataclasses import replace

import numpy as np
import pytest

from mixture_cases import mixture_case
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes import MarcatiliMode
from solprop.physdata import Mixture
from solprop.profiles import Constant


@pytest.mark.parametrize("loss", [False, True, 0.0, 0.5, 2.0])
def test_absorption_by_species_is_density_times_cross_section(loss):
    mixture, grid, mode, _, _ = mixture_case()
    first, second = mixture.components
    optical = replace(
        first.species.susceptibility, absorption_cross_section=np.full(2, 2e-25)
    )
    first = replace(first, species=replace(first.species, susceptibility=optical))
    mixture = Mixture.of(first, second)
    mode = replace(mode, mixture=mixture, loss=loss)
    omega = grid.omega[grid.active_mask]
    # Spline evaluation and omega/c conversion each require fewer than 16 rounded
    # operations on this constant cross section; allocate 64 eps in total.
    bound = 64 * np.finfo(float).eps
    result = mode.absorption_by_species(omega)
    np.testing.assert_allclose(result[0], float(loss) * 0.2, rtol=bound, atol=0)
    np.testing.assert_array_equal(result[1], np.zeros(omega.size))


def test_species_loss_refuses_callable_and_invalid_frequency():
    _, _, mode, _, _ = mixture_case()
    for omega in (np.array([0.0]), np.array([-1.0]), np.array([np.nan])):
        with pytest.raises(ModeError, match="positive finite omega"):
            mode.absorption_by_species(omega)
    with pytest.raises(ModeError, match="callable loss"):
        replace(mode, loss=np.ones_like).absorption_by_species(
            np.array([2 * np.pi * C / 800e-9])
        )


def test_legacy_gas_and_vacuum_have_one_zero_absorption_row():
    omega = np.array([2 * np.pi * C / 800e-9])
    for mode in (MarcatiliMode(30e-6), MarcatiliMode(30e-6, "He", 1.0)):
        np.testing.assert_array_equal(
            mode.absorption_by_species(omega), np.zeros((1, 1))
        )


def test_zero_population_contributes_no_attributed_loss():
    mixture, grid, mode, _, _ = mixture_case()
    mixture = mixture.with_densities({label: Constant(0) for label in mixture.labels})
    mode = replace(mode, mixture=mixture, loss=True)
    np.testing.assert_array_equal(
        mode.absorption_by_species(grid.omega[grid.active_mask]),
        np.zeros((2, np.count_nonzero(grid.active_mask))),
    )
