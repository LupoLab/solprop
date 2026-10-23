# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Guard pre-extraction numerical values and shared fill behaviour."""

import inspect
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from witness_materials import ermolov_helium, kingston_argon

from bounds import EPS, assert_within
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes import GasFillMixin, MarcatiliMode, TubeLattice, VincettiMode
from solprop.physdata import Mixture
from solprop.profiles import Linear, PressureGradient


def baseline_modes():
    # The table holds solprop's former registry gases (Ermolov helium, Kingston argon),
    # supplied explicitly since SEP 3's Phase 4b deleted the registry and the
    # mode-level source=; the equation of state is optmat's.
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


def baseline_values():
    omega = 2 * np.pi * C / np.linspace(600e-9, 900e-9, 31)
    return np.asarray(
        [
            [getattr(mode, name)(omega, z) for z in (0.0, 0.05, 0.1)]
            for mode in baseline_modes()
            for name in ("neff", "neff_minus_one", "dneff_domega")
        ]
    )


def test_fill_extraction_preserves_fixed_grid_and_dataclass_surface():
    """The modes evaluate as they did before the fill was extracted.

    Each mode's ``neff``, ``neff - 1`` and its derivative at each position is compared
    normwise with the recorded values at 64 eps: the ladder's one-evaluation budget of
    16 eps (``accuracy.md`` section 7) for a closed form of a few tens of roundings,
    times four for another platform's libm in the transcendental terms. The signature
    and the dataclass fields are structural and stay exact.
    """
    reference = json.loads(
        (Path(__file__).parents[1] / "data/fill_before_extraction.json").read_text()
    )
    expected = np.asarray(reference["real"]) + 1j * np.asarray(reference["imag"])
    assert str(inspect.signature(MarcatiliMode)) == reference["signature"]
    assert [field.name for field in fields(MarcatiliMode)] == reference["fields"]
    actual = baseline_values()
    assert actual.shape == expected.shape
    for index in np.ndindex(actual.shape[:2]):
        assert_within(actual[index], expected[index], 64 * EPS, what=str(index))


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
