# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Continuous-profile component traits and mixed-family collection construction."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.exceptions import ModeError
from solprop.modes import MarcatiliMode, ModeCollection
from solprop.nonlinear.gnlse import SpecifiedMode


@pytest.mark.parametrize(
    "phi, expected",
    [
        (0.0, "x"),
        (math.pi, "x"),
        (math.pi / 2, "y"),
        (-math.pi / 2, "y"),
        (3 * math.pi / 2, "y"),
        (math.pi / 4, None),
        (np.nextafter(math.pi / 2, 0.0), None),
    ],
)
def test_principal_component_is_a_continuous_profile_trait(phi, expected):
    """phi = 0 points along x and the quarter turn along y (ADR-031)."""
    mode = MarcatiliMode(100e-6, phi=phi)
    assert mode.principal_component() == expected
    points = np.column_stack([np.full(11, 30e-6), np.linspace(0, 2 * math.pi, 11)])
    field = mode.field(points)
    if expected is not None:
        zero_axis = 1 if expected == "x" else 0
        assert np.count_nonzero(field[:, zero_axis]) == 0
    # HE_1m has no azimuthal dependence, including at arbitrary rotation angles.
    assert_allclose(field, np.broadcast_to(field[0], field.shape), rtol=0, atol=0)


@pytest.mark.parametrize("kind, n", [("HE", 2), ("HE", -1), ("TE", 0), ("TM", 0)])
def test_azimuthal_modes_require_both_cartesian_components(kind, n):
    mode = MarcatiliMode(100e-6, kind=kind, n=n)
    assert mode.principal_component() is None


def test_dispersion_surrogate_declares_its_actual_x_profile():
    mode = SpecifiedMode.from_taylor([1.0], 1.0)
    assert mode.principal_component() == "x"
    assert mode.field(np.array([[0.0, 0.0]]))[0, 1] == 0


@pytest.mark.parametrize("phi", [np.nan, np.inf, -np.inf])
def test_nonfinite_rotation_is_rejected(phi):
    with pytest.raises(ModeError, match="phi must be finite"):
        MarcatiliMode(100e-6, phi=phi)


def test_marcatili_collection_preserves_indices_and_material_settings():
    collection = ModeCollection.marcatili(
        100e-6,
        "Ar",
        1.0,
        ["HE", "HE", "TE", "TM"],
        [1, 2, 0, 0],
        [2, 1, 1, 2],
        model="reduced",
        loss=False,
        cladding=1.45,
    )
    indices = []
    for mode in collection:
        assert isinstance(mode, MarcatiliMode)
        indices.append((mode.kind, mode.n, mode.m))
        assert mode.model == "reduced" and mode.loss is False
    assert indices == [
        ("HE", 1, 2),
        ("HE", 2, 1),
        ("TE", 0, 1),
        ("TM", 0, 2),
    ]
    assert collection.needs_full_rule


@pytest.mark.parametrize(
    "kinds, n, m", [([], [], []), (["HE"], [1, 2], [1]), (["HE"], [1], [])]
)
def test_marcatili_collection_rejects_mismatched_index_sequences(kinds, n, m):
    with pytest.raises(ModeError, match="equal length"):
        ModeCollection.marcatili(100e-6, "Ar", 1.0, kinds, n, m)


def test_marcatili_collection_rejects_invalid_family_indices():
    with pytest.raises(ModeError, match="needs n = 0"):
        ModeCollection.marcatili(100e-6, "Ar", 1.0, ["TM"], [1], [1])
