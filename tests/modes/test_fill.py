# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The shared gas fill of the hollow and antiresonant guides."""

import numpy as np
import pytest
from witness_materials import ermolov_helium, kingston_argon

from solprop.exceptions import ModeError
from solprop.modes import GasFillMixin, MarcatiliMode, TubeLattice, VincettiMode
from solprop.physdata import Mixture
from solprop.profiles import Linear, PressureGradient


def baseline_modes():
    # A constant fill, a tapered TM guide in a gradient and an antiresonant guide.
    return (
        MarcatiliMode(
            125e-6, "He", 1.0, susceptibility=ermolov_helium(), cladding=1.45
        ),
        MarcatiliMode(
            Linear(125e-6, 100e-6, 0.1),
            "Ar",
            PressureGradient.two_point(0.1, 5.0, 1.0),
            susceptibility=kingston_argon(),
            kind="TM",
            n=0,
            cladding=1.45,
        ),
        VincettiMode(
            15e-6,
            "Ar",
            10.0,
            susceptibility=kingston_argon(),
            lattice=TubeLattice(180e-9, 10.38e-6, 6),
            cladding=1.45,
        ),
    )


def test_fill_rebinding_retains_concrete_geometry_and_clears_legacy_material():
    mixture = Mixture.fill({"He": 0.25, "Ar": 0.75}, 2.0)
    for mode in baseline_modes():
        rebound = mode.with_mixture(mixture)
        assert type(rebound) is type(mode)
        assert isinstance(rebound, MarcatiliMode)
        assert rebound.mixture is mixture
        assert rebound.gas is None and rebound.pressure == 0.0
        assert rebound.susceptibility is None
        assert rebound.radius == mode.radius
        assert rebound.cladding == mode.cladding
    with pytest.raises(ModeError, match="concrete mode dataclass"):
        GasFillMixin().with_mixture(mixture)


def test_fill_rejects_invalid_loss_and_freezes_single_species_density():
    mode = MarcatiliMode(125e-6)
    object.__setattr__(mode, "loss", object())
    with pytest.raises(ModeError, match="loss must be"):
        mode._check_fill_arguments()
    filled = MarcatiliMode(125e-6, mixture=Mixture.fill({"He": 1.0}, 1.0))
    frozen = filled.at_parameters({"density": 1e25})
    assert frozen.mixture is not None
    assert float(frozen.mixture.components[0].density(np.asarray([0.0]))[0]) == 1e25
