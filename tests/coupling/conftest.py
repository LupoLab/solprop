# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared beams and grids for the transport tests.

A 30 fs pulse at 800 nm on a narrow envelope grid keeps every transport cheap while
leaving a band of several dozen frequencies for the per-bin ledger.
"""

from collections.abc import Callable

import numpy as np
import pytest

from solprop.coupling.elements import SpatialState
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium

WAVELENGTH = 800e-9
ENERGY = 1e-6

BeamFactory = Callable[..., SpatialState]


@pytest.fixture(scope="module")
def grid() -> EnvelopeGrid:
    """Return a narrow envelope grid around 800 nm."""
    return EnvelopeGrid.from_limits(0.1, WAVELENGTH, (750e-9, 850e-9), 200e-15)


@pytest.fixture(scope="module")
def air() -> SpatialMedium:
    """Return vacuum, whose index is exactly one."""
    return SpatialMedium()


@pytest.fixture(scope="module")
def centre(grid: EnvelopeGrid) -> int:
    """Return the storage index of the bin nearest the carrier."""
    return int(np.argmin(np.abs(grid.omega - grid.omega0)))


@pytest.fixture(scope="module")
def beam(grid: EnvelopeGrid, air: SpatialMedium) -> BeamFactory:
    """Return a factory of Gaussian or super-Gaussian beam planes."""

    def make(  # noqa: PLR0913 - the beam's independent parameters
        space: RadialGrid | FreeGrid,
        waist: float,
        *,
        medium: SpatialMedium = air,
        order: int = 1,
        jones: tuple[complex, ...] = (1.0 + 0.0j,),
        fwhm: float = 30e-15,
        wavelength: float = WAVELENGTH,
    ) -> SpatialState:
        pulse = GaussianPulse(wavelength=wavelength, fwhm=fwhm, energy=ENERGY)
        spatial = SpatialPulse(pulse, waist, order=order, jones=jones)
        spectrum = spatial.build(grid, space, medium=medium)
        return SpatialState(grid, space, spectrum, medium)

    return make
