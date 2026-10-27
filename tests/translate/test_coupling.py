# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The frame record and the port metric of a translated problem (SEP 31)."""

import dataclasses

import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import TranslationError
from solprop.grid import FieldGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.coupling import frame_record, port_metric


def radial():
    grid = FieldGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 400e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-3],
        dz_max=2e-4,
        dz_init=2e-4,
        backend="numpy_ref",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    return Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 16),
        SpatialMedium("Ar", 1.0),
        [],
        pulse,
        settings=settings,
        noise=None,
    )


def test_an_operator_without_its_frame_tables_has_no_frame_record():
    simulation = radial()
    linear = simulation.translate().linear
    assert frame_record(linear, 0.0) is not None
    # An operator supplied without its frame tables leaves the entrance frame
    # uncalibrated: there is nothing to integrate.
    assert frame_record(dataclasses.replace(linear, beta0=None), 0.0) is None


def test_a_spatial_problem_without_its_plane_measure_has_no_port_metric():
    simulation = radial()
    problem = simulation.translate()
    assert port_metric(problem, simulation.grid, 0.0).kind == "admittance"
    # A hand-assembled, unvalidated problem that dropped its plane measure.
    with pytest.raises(TranslationError, match="plane measure"):
        port_metric(dataclasses.replace(problem, flux=None), simulation.grid, 0.0)
