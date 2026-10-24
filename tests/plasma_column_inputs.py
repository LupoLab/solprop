# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic inputs and baseline arguments for the compiled plasma column tests."""

import math
from typing import NamedTuple

import numpy as np

from solprop.backends.bodies import RateArrays

__all__ = ["PlasmaScalars", "inputs", "reference_arguments"]


class PlasmaScalars(NamedTuple):
    """Group the column's host scalar parameters without changing units.

    Attributes
    ----------
    kerr_factor : float
        Kerr coefficient times density, in the column's units.
    charge_mass_ratio : float
        Electron charge-to-mass factor from the compiled plasma response.
    ionisation_potential : float
        Ionisation energy in J.
    density : float
        Number density in m^-3.
    preionisation : float
        Initial ionised fraction in [0, 1].
    dt : float
        Positive fine-grid sample interval in s.
    """

    kerr_factor: float
    charge_mass_ratio: float
    ionisation_potential: float
    density: float
    preionisation: float
    dt: float


def inputs(samples, nodes, preionisation):
    axis = np.arange(samples, dtype=float)
    field = np.tile(3 * np.cos(axis / 3.1), (nodes, 1))
    field *= np.linspace(0.5, 1.5, nodes)[:, None]
    field[:, ::7] = 0.0
    if samples > 1:
        field[:, 1] = 1e-170
    coefficients = np.zeros((4, 2))
    coefficients[2] = 2.0
    coefficients[3] = np.log(4.0) + 2 * np.arange(2) * math.log(2.0)
    table = RateArrays(coefficients, 0.0, math.log(2.0), 1.0, 4.0, 64.0)
    scalars = PlasmaScalars(1e-4, 3.0, 0.2, 2.0, preionisation, 1e-3)
    coefficients.flags.writeable = False
    return field, table, scalars, np.linspace(0.1, 1.0, samples)


def reference_arguments(field, table, scalars, window):
    nodes, samples = field.shape
    return (
        field,
        np.full_like(field, np.nan),
        np.zeros((nodes, 2 * samples)),
        window,
        table,
        np.ones(samples, complex),
        scalars.kerr_factor,
        0.0,
        0.0,
        0.0,
        scalars.charge_mass_ratio,
        scalars.ionisation_potential,
        scalars.density,
        scalars.preionisation,
        scalars.dt,
        np.full((nodes, 2), -1.0),
    )
