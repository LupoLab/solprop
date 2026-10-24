# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupling models' declarations and scalar profile sets, and their refusals.

SEP 31 section 4.3: a declared basis change at every bin, the budget's cone, the
weak-guidance LP0m eigenvalue below its cutoff, and the scalar profile sets a
capillary's or a step-index fibre's HE1m modes give, with every inconsistent set
refused by name.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.special import jn_zeros

from solprop.coupling import BasisChange, CouplingModel, ScalarModeBasis
from solprop.coupling.models import _lp_parameter, cone_angle
from solprop.exceptions import CouplingError
from solprop.modes import MarcatiliMode, StepIndexMode

CORE, CLADDING = 1.45, 1.44
"""A weakly guiding step: Delta = 6.9e-3."""


def step_index_mode(radius=4e-6, **keywords):
    """Return a step-index mode of the weak step, profiled at 800 nm."""
    return StepIndexMode(
        radius, core=CORE, cladding=CLADDING, profile_wavelength=800e-9, **keywords
    )


def test_a_frequency_dependent_basis_change_is_taken_bin_by_bin(grid):
    """A trailing frequency axis is moved first; another bin count is refused."""
    phases = np.exp(1j * np.linspace(0.0, 1.0, grid.n_omega))
    gauge = np.zeros((2, 2, grid.n_omega), dtype=complex)
    gauge[0, 0] = phases
    gauge[1, 1] = phases.conj()
    change = BasisChange(gauge, "0" * 64, "1" * 64)
    matrices = change.at(grid.n_omega)
    assert matrices.shape == (grid.n_omega, 2, 2)
    np.testing.assert_array_equal(matrices[:, 0, 0], phases)
    np.testing.assert_array_equal(matrices[:, 0, 1], 0.0)
    constant = BasisChange(np.eye(2, dtype=complex), "0" * 64, "1" * 64).at(
        grid.n_omega
    )
    assert constant.shape == (grid.n_omega, 2, 2)
    with pytest.raises(CouplingError, match=rf"has {grid.n_omega} bins, not 5"):
        change.at(5)


def test_the_cone_is_the_budget_s_and_same_basis_has_none():
    """A model with a budget admits its cone; exact recovery has no angular budget."""
    model = CouplingModel("scalar_paraxial", model_error=0.05)
    assert model.cone == cone_angle(0.05)
    assert 1 - math.cos(model.cone) == pytest.approx(0.05 / 8, rel=1e-14)
    with pytest.raises(CouplingError, match="no angular budget"):
        _ = CouplingModel("same_basis").cone


def test_an_lp_mode_below_its_cutoff_is_refused():
    """LP02 exists only above the first zero of J1; at or below it there is no root."""
    cutoff = float(jn_zeros(1, 1)[0])
    u, w = _lp_parameter(1.2 * cutoff, 2)
    assert cutoff < u < float(jn_zeros(0, 2)[-1])
    assert u**2 + w**2 == pytest.approx((1.2 * cutoff) ** 2, rel=1e-14)
    for v in (0.9 * cutoff, cutoff):
        with pytest.raises(CouplingError, match="LP02 is cut off"):
            _lp_parameter(v, 2)


def test_a_step_index_basis_prints_and_has_no_leaky_field(grid):
    """A dielectric fibre has no wall term, so no leaky parameter; a zero cone has
    no angular content."""
    profiles = ScalarModeBasis.from_modes([step_index_mode()], grid)
    assert repr(profiles) == "ScalarModeBasis(step_index, HE1,1:even, a=4e-06 m)"
    assert profiles.leaky(grid.omega) is None
    np.testing.assert_array_equal(profiles.cone_gram(0.0), np.zeros((1, 1)))
    assert np.all(profiles.cone_gram(1e6) > 0.0)


def test_a_scalar_basis_refuses_nonfinite_parameters(grid):
    """The profile parameters and Jones vectors are finite.

    A wrong shape is refused by the test session's shape checker before the
    constructor's own guard, which is the only one outside it.
    """
    profiles = ScalarModeBasis.from_modes([step_index_mode()], grid)
    with pytest.raises(CouplingError, match=r"u has shape \(1,\)"):
        dataclasses.replace(profiles, u=np.array([np.nan]))
    with pytest.raises(CouplingError, match=r"jones has shape \(1, 2\)"):
        dataclasses.replace(profiles, jones=np.array([[np.inf, 0.0]]))


def test_a_scalar_basis_refuses_sets_it_has_no_profiles_for(grid):
    """An empty set, mixed families, capillaries of two radii, a step-index mode with
    angular structure and step-index modes of two fibres are refused by name."""
    with pytest.raises(CouplingError, match="at least one mode"):
        ScalarModeBasis.from_modes([], grid)
    with pytest.raises(CouplingError, match="'MarcatiliMode', 'StepIndexMode'"):
        ScalarModeBasis.from_modes(
            [MarcatiliMode(75e-6, "Ar", 3.0), step_index_mode()], grid
        )
    with pytest.raises(CouplingError, match="different radii"):
        ScalarModeBasis.from_modes(
            [MarcatiliMode(75e-6, "Ar", 3.0), MarcatiliMode(50e-6, "Ar", 3.0)], grid
        )
    with pytest.raises(CouplingError, match="HE2,1:even has angular structure"):
        ScalarModeBasis.from_modes([step_index_mode(6e-6, n=2)], grid)
    with pytest.raises(CouplingError, match="share one fibre and profile"):
        ScalarModeBasis.from_modes([step_index_mode(), step_index_mode(4.4e-6)], grid)
