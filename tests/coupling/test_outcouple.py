# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The matched outcoupling of a modal plane with both polarisations (SEP 31)."""

import math

import numpy as np
import pytest

from solprop.coupling import (
    FieldPlane,
    Outcouple,
    PlaneBasis,
    PlaneGeometry,
    ScalarModeBasis,
)
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.modes import MarcatiliMode


def test_both_polarisation_columns_of_a_mode_are_synthesised_as_x_and_y():
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (700e-9, 920e-9), 200e-15)
    modes = [MarcatiliMode(75e-6, "Ar", 3.0, m=m) for m in (1, 2)]
    profiles = ScalarModeBasis.from_modes(modes, grid)
    width = 30e-15 / (2 * math.sqrt(math.log(2)))
    envelope = np.exp(-(((grid.omega - grid.omega0) * width) ** 2) / 4)
    envelope = envelope * grid.active_mask
    samples = np.zeros((2, 2, grid.n_omega), dtype=complex)
    # HE11 carries x and, a quarter period later, half its amplitude in y; HE12
    # carries a little x only.
    samples[0, 0] = envelope
    samples[0, 1] = 0.5j * envelope
    samples[1, 0] = 0.2 * envelope
    plane = FieldPlane.from_samples(
        samples,
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=PlaneBasis("modal", "0" * 64, profiles.labels),
        profiles=profiles,
    )
    space = RadialGrid(500e-6, 512)
    result = Outcouple(space).apply(plane)
    synthesised = result.plane.samples
    assert synthesised.shape == (512, 2, grid.n_omega)
    # Orthonormal profiles: the y share is HE11's |0.5|^2 against the x columns'
    # 1 + 0.2^2, to the modes' effective-index factors (1e-4) and the synthesis
    # window.
    weights = np.asarray(space.spectral_weights)[:, None]
    x_energy = np.sum(weights * np.abs(synthesised[:, 0]) ** 2)
    y_energy = np.sum(weights * np.abs(synthesised[:, 1]) ** 2)
    assert y_energy / x_energy == pytest.approx(0.25 / 1.04, rel=1e-3)
    # Matched exit: the field's flux and the modal power agree to the profiles'
    # obliquity, as for a single polarisation.
    np.testing.assert_allclose(
        np.sum(result.synthesised), np.sum(result.modal), rtol=1e-6
    )
    assert np.sum(result.modal) == pytest.approx(plane.energy(), rel=1e-15)
