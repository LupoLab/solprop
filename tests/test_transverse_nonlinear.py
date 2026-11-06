# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A nonlinear field reaching the transverse boundary (SEP 40 section 7.5, Part II).

The suite's ``spatial_absorber`` case scaled paraxially by 1/8, now with its Kerr
response at 0.9 of the critical power and noise off, against a bare wall far enough
out that the field never reaches it, at the same radial spacing. The aperture's
field error at 1 cm reproduces the full 8 cm case's: about 1e-1 as configured, 4e-5
on a bare wall at twice the radius and 1.7e-6 on a requested collar at f = 2.6.
"""

import dataclasses
import math
import warnings
from typing import Any, cast

import numpy as np
import pytest
from scipy.special import j0

from solprop.boundaries import SpatialAbsorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.output.schema import PlaneStore
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium, critical_power
from solprop.stepper import PIController
from test_transverse_resolution import LENGTH, RADIUS, W0

RADII = np.linspace(0.0, RADIUS, 481)
"""Where the fields are compared: the requested aperture."""


def kerr_suite(
    fraction: float, specification: SpatialAbsorber | bool | None
) -> Simulation:
    """Return the scaled suite case with Kerr at 0.9 P_cr, on `fraction` of room."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (500e-9, 1.6e-6), 250e-15)
    space = RadialGrid(RADIUS, round(64 * (1 + fraction)), collar_fraction=fraction)
    medium = SpatialMedium("Ar", 1.0)
    chi3 = gamma3("Ar", wavelength=800e-9)
    density = float(medium.density(np.array([0.0]))[0])
    index = float(np.real(medium.index(np.array([2 * math.pi * C / 800e-9]))[0]))
    power = 0.9 * critical_power(800e-9, chi3 * density, index=index)
    energy = power * 30e-15 * math.sqrt(math.pi / (4 * math.log(2)))
    return Simulation.from_radial(
        grid,
        space,
        medium,
        [KerrResponse(chi3, thg=False)],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), W0),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-7, atol=0.0),
            save_planes=[LENGTH],
            dz_max=LENGTH,
            dz_init=LENGTH * 1e-4,
            backend="numpy_ref",
        ),
        noise=None,
        spatial_absorber=specification,  # pyright: ignore[reportArgumentType]
    )


def aperture_field(simulation: Simulation) -> tuple[np.ndarray, dict[str, Any]]:
    """Run; return the field at L on the aperture's radii and the edge's record."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    assert output.outcome is not None
    monitor = output.outcome.execution.get("boundary_monitor") or {}
    assert isinstance(monitor, dict)
    space = simulation.spatial_grid
    assert isinstance(space, RadialGrid)
    hankel = space.hankel
    # The Fourier-Bessel series of the columns, exact on any radii.
    coefficient = 2.0 / (hankel.radius**2 * hankel.j1**2)
    spectrum = np.asarray(output["field"][-1])[:, 0, :]
    field = j0(np.outer(RADII, hankel.k_r)) @ (coefficient[:, None] * spectrum)
    return field, cast("dict[str, Any]", monitor.get("transverse", {}))


def aperture_error(field: np.ndarray, reference: np.ndarray) -> float:
    """Return the area-weighted L2 error on the aperture, over the reference."""
    weight = 2 * np.pi * RADII
    weight[-1] /= 2
    difference = np.sum(weight[:, None] * np.abs(field - reference) ** 2)
    return float(np.sqrt(difference / np.sum(weight[:, None] * np.abs(reference) ** 2)))


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    """A bare wall at three times the radius: within 2e-7 of one at eight times."""
    return aperture_field(kerr_suite(2.0, False))[0]


@pytest.mark.medium
def test_the_nonlinear_case_as_configured_is_stopped_and_is_wrong(
    reference: np.ndarray,
) -> None:
    with (
        warnings.catch_warnings(),
        pytest.raises(BoundaryError, match="transverse edge") as caught,
    ):
        warnings.simplefilter("ignore", BoundaryWarning)
        kerr_suite(0.0, None).run(progress="none")
    assert "a radius of at least" in str(caught.value)
    # Unwatched, it completes about 1e-1 wrong (Part II: above 1e-2).
    simulation = kerr_suite(0.0, None)
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    assert aperture_error(aperture_field(simulation)[0], reference) > 1e-2


@pytest.mark.medium
def test_a_bare_wall_at_twice_the_radius_meets_the_witness(
    reference: np.ndarray,
) -> None:
    field, record = aperture_field(kerr_suite(1.0, None))
    assert 4e-5 / 3 <= aperture_error(field, reference) <= 3 * 4e-5
    # The wall's edge share reads conservatively: light there has not yet come back.
    assert record["transverse_status"] == "warned"


@pytest.mark.medium
def test_a_requested_collar_meets_the_witness_and_reads_clear(
    reference: np.ndarray,
) -> None:
    field, record = aperture_field(kerr_suite(2.6, SpatialAbsorber()))
    assert 1.7e-6 / 3 <= aperture_error(field, reference) <= 3 * 1.7e-6
    # The strip and the current, on a nonlinear run, read below the warning.
    assert record["transverse_status"] == "clear"
    assert record["returned_max"] < 1e-6
    assert record["current"] < 1e-6
