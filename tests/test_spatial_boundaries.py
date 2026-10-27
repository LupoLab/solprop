# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise spatial input, assembly and saved-statistic boundary failures."""

import dataclasses
import inspect
from contextlib import nullcontext

import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import DriverError, OutputError, PulseError, TranslationError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.output.schema import SavedPlane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium
from solprop.stats import StatsSpec, collect, default_spec
from solprop.stepper import PIController
from solprop.translate.problem import assemble
from solprop.translate.responses import compile_responses


@pytest.fixture
def case():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(6e-4, 16)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9)
    pulse = SpatialPulse(temporal, 60e-6)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    return grid, space, pulse, settings


@pytest.mark.parametrize(
    "changes",
    [
        {"waist": 0.0},
        {"order": 0},
        {"order": True},
        {"propz": np.inf},
        {"jones": ()},
        {"jones": (0j,)},
        {"jones": (complex(np.nan),)},
        {"jones": (1j, 1j, 1j)},
    ],
)
def test_spatial_pulse_rejects_invalid_specification(case, changes):
    with pytest.raises(PulseError):
        dataclasses.replace(case[2], **changes)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"waist": 1e-100}, "unresolved"),
        ({"propz": -1e8}, "overflow"),
    ],
)
def test_spatial_pulse_rejects_unresolved_or_overflowing_launch(case, changes, match):
    grid, space, pulse, _ = case
    with pytest.raises(PulseError, match=match):
        dataclasses.replace(pulse, **changes).build(
            grid, space, medium=SpatialMedium(1 + 1j)
        )


@pytest.mark.parametrize(
    "shape,match", [("two_columns", "scalar single-column"), ("nonfinite", "nonfinite")]
)
def test_spatial_pulse_checks_the_temporal_pulse_result(case, shape, match):
    grid, space, pulse, _ = case

    class Temporal:
        def build(self, grid):
            if shape == "two_columns":
                return np.zeros((2, 1, grid.n_omega), complex)
            return np.full((1, 1, grid.n_omega), complex(np.inf))

    diagnostic = (
        pytest.warns(RuntimeWarning, match="invalid value")
        if shape == "nonfinite"
        else nullcontext()
    )
    with diagnostic, pytest.raises(PulseError, match=match):
        dataclasses.replace(pulse, temporal=Temporal()).build(
            grid, space, medium=SpatialMedium()
        )


@pytest.mark.parametrize("which", ["space_only", "medium_only", "modes", "pulse_only"])
def test_simulation_constructor_rejects_incomplete_spatial_context(case, which):
    grid, space, pulse, settings = case
    options = {"spatial_grid": space, "medium": SpatialMedium()}
    modes = ()
    if which == "space_only":
        options.pop("medium")
    elif which == "medium_only":
        options.pop("spatial_grid")
    elif which == "pulse_only":
        options = {}
    else:
        modes = (object(),)
    with pytest.raises(DriverError):
        inspect.unwrap(Simulation.__init__)(
            object.__new__(Simulation),
            grid,
            modes,
            [],
            pulse,
            settings=settings,
            **options,
        )


def test_direct_spatial_constructor_derives_a_default_frame(case):
    grid, space, pulse, settings = case
    simulation = Simulation(
        grid,
        (),
        [],
        pulse,
        settings=settings,
        spatial_grid=space,
        medium=SpatialMedium(),
    )
    assert simulation.frame is not None
    assert simulation.translate().shape[1] == space.n_r


@pytest.mark.parametrize("changes", [{"schema_version": 3}, {"spatial": None}])
def test_spatial_metadata_requires_geometry_and_new_output_schema(case, changes):
    grid, space, pulse, settings = case
    metadata = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    ).metadata()
    with pytest.raises(OutputError, match="schema 4"):
        dataclasses.replace(metadata, **changes)


def test_spatial_statistics_refuse_modal_quantities_and_keep_density(case):
    grid, space, pulse, _ = case
    assert default_spec("radial", with_profiles=True).names[-1] == "density"
    with pytest.raises(OutputError, match="modal zdw"):
        default_spec("radial", with_zdw=True)
    field = pulse.build(grid, space, medium=SpatialMedium())
    plane = SavedPlane(z=0.0, field=field)
    with pytest.raises(OutputError, match="not a spatial"):
        collect(
            StatsSpec(names=("peak_power",)),
            grid,
            plane,
            flux=SpatialFlux.resolve(grid, space, SpatialMedium()),
        )


def test_spatial_response_compilation_requires_a_translated_prefactor(case):
    grid = case[0]
    with pytest.raises(TranslationError, match="translated_prefactor"):
        compile_responses([], grid, None, density=1.0, z_max=grid.length)


@pytest.mark.parametrize("change", ["polarisation", "matrix"])
def test_radial_transform_rejects_incompatible_scalar_map(case, change):
    grid, space, pulse, settings = case
    problem = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    ).translate()
    transform = problem.transform
    options = (
        {"_n_pol": 2}
        if change == "polarisation"
        else {
            "synthesis": transform.synthesis[:-1, :-1],
            "projection": transform.projection[:-1, :-1],
            "projection_coarse": transform.projection_coarse[:-1, :-1],
        }
    )
    with pytest.raises(TranslationError, match=r"scalar map|synthesis"):
        dataclasses.replace(transform, **options)


def test_spatial_response_compilation_rejects_modal_averaging(case):
    grid, space, pulse, settings = case
    problem = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    ).translate()
    with pytest.raises(TranslationError, match="mode_averaged=False"):
        compile_responses(
            [],
            grid,
            None,
            density=1.0,
            z_max=grid.length,
            translated_prefactor=problem.responses.prefactor,
        )


def test_radial_assembly_rejects_wrong_column_count(case):
    grid, space, pulse, settings = case
    initial = pulse.build(grid, space, medium=SpatialMedium())[:-1]
    with pytest.raises(DriverError, match="spatial input"):
        Simulation.from_radial(
            grid, space, SpatialMedium(), [], initial, settings=settings, noise=None
        )
    with pytest.raises(TranslationError, match="spatial columns"):
        assemble(
            grid,
            None,
            [],
            initial,
            spatial_grid=space,
            medium=SpatialMedium(),
            controller=settings.controller,
            save_planes=[grid.length],
            dz_init=grid.length,
            dz_max=grid.length,
        )
