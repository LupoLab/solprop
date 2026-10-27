# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The input-noise default and its recorded resolution (SEP 37 phase 2)."""

import json

import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings, prop_gnlse
from solprop.driver.simulation import (
    INTERIOR_LAUNCH_NOISE,
    MIXED_PRECISION_NOISE,
    SINGLE_PRECISION_NOISE,
)
from solprop.driver.summary import _noise
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.noise import photons_per_bin
from solprop.output.schema import NoiseResolution, RunMetadata
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

GNLSE = {
    "lambda0": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "pulseshape": "sech",
    "n_save": 3,
    "backend": "numpy_ref",
}
"""A short ideal soliton: cheap, and enough to resolve the noise keyword."""


def _gnlse(**overrides):
    return Simulation.from_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], **GNLSE, **overrides
    )


def test_the_default_resolves_to_input_noise_and_records_it():
    simulation = _gnlse()
    assert simulation.noise is not None
    assert simulation.noise.model == "input"
    assert simulation.noise.seed is not None
    assert simulation.noise_resolution == NoiseResolution("default", "input")
    assert simulation.metadata().noise_resolution == NoiseResolution("default", "input")


@pytest.mark.parametrize("model", ["input", "modified"])
def test_an_explicit_model_is_recorded_as_requested(model):
    simulation = _gnlse(noise=model, seed=3)
    assert simulation.noise is not None
    assert simulation.noise.model == model
    assert simulation.noise.seed == 3
    assert simulation.noise_resolution == NoiseResolution(model, model)


def test_none_switches_noise_off_explicitly():
    simulation = _gnlse(noise=None)
    assert simulation.noise is None
    assert simulation.noise_resolution == NoiseResolution(None, None)


def test_a_seed_beside_no_noise_is_refused():
    with pytest.raises(DriverError, match="seed requires a noise model"):
        _gnlse(noise=None, seed=3)


def test_single_precision_falls_back_and_says_why():
    simulation = _gnlse(precision="single")
    assert simulation.noise is None
    assert simulation.noise_resolution == NoiseResolution(
        "default", None, SINGLE_PRECISION_NOISE
    )
    assert _noise(simulation.metadata()) == f"off (default: {SINGLE_PRECISION_NOISE})"


def test_mixed_precision_falls_back_until_a_model_is_qualified():
    simulation = _gnlse(precision="mixed")
    assert simulation.noise is None
    assert simulation.noise_resolution == NoiseResolution(
        "default", None, MIXED_PRECISION_NOISE
    )
    assert _noise(simulation.metadata()) == f"off (default: {MIXED_PRECISION_NOISE})"


def test_a_seed_beside_a_fallback_is_refused():
    with pytest.raises(DriverError, match="single precision carries no noise"):
        _gnlse(precision="single", seed=3)


def test_the_capillary_and_radial_routes_default_to_input_noise():
    capillary = Simulation.from_capillary(
        100e-6,
        0.01,
        "He",
        1.0,
        lambda0=800e-9,
        fwhm=30e-15,
        energy=1e-6,
        wavelength_limits=(400e-9, 2e-6),
        trange=200e-15,
        mode_average=True,
        n_save=2,
        backend="numpy_ref",
    )
    assert capillary.noise_resolution == NoiseResolution("default", "input")
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    radial = Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 16),
        SpatialMedium(),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6),
        settings=settings,
    )
    assert radial.noise is not None
    assert radial.noise_resolution == NoiseResolution("default", "input")


def test_the_summary_marks_a_default_model():
    metadata = _gnlse(seed=7).metadata()
    assert _noise(metadata) == "input model (default), seed 7"
    assert _noise(_gnlse(noise="input", seed=7).metadata()) == "input model, seed 7"


def test_the_resolution_survives_the_stored_record():
    metadata = _gnlse(seed=7).metadata()
    back = RunMetadata.from_json(metadata.to_json())
    assert back.noise_resolution == metadata.noise_resolution
    assert back.configuration_hash() == metadata.configuration_hash()


def test_a_record_without_a_resolution_reads_as_not_recorded():
    document = json.loads(_gnlse(seed=7).metadata().to_json())
    del document["noise_resolution"]
    assert RunMetadata.from_json(json.dumps(document)).noise_resolution is None


def test_the_resolution_stays_outside_the_resume_key():
    default = _gnlse(seed=7).metadata()
    explicit = _gnlse(noise="input", seed=7).metadata()
    assert default.noise_resolution != explicit.noise_resolution
    assert default.configuration_hash() == explicit.configuration_hash()


def _interior(simulation):
    """Return a launch half way along the run, of the run's own entrance spectrum."""
    return simulation.with_input(
        np.asarray(simulation.translate().initial.field0[0]),
        initial_z=simulation.grid.length / 2,
    )


def test_an_interior_launch_drops_a_default_draw():
    launched = _interior(_gnlse(seed=7))
    assert launched.noise is None
    assert launched.noise_resolution == NoiseResolution(
        "default", None, INTERIOR_LAUNCH_NOISE
    )


def test_an_explicit_input_draw_enters_an_interior_launch():
    base = _gnlse(noise=None)
    clean = _interior(base).run(progress="none")
    seeded = base.with_input(
        np.asarray(base.translate().initial.field0[0]),
        initial_z=base.grid.length / 2,
        noise=_gnlse(noise="input", seed=7).noise,
    )
    noisy = seeded.run(progress="none")
    assert noisy.metadata is not None
    assert noisy.metadata.noise is not None
    first = np.asarray(noisy["field"][0]) - np.asarray(clean["field"][0])
    active = base.translate().grids.active_mask
    photons = photons_per_bin(base.grid, first[0, 0])
    np.testing.assert_allclose(photons[active], 1.0, rtol=1e-9)


def test_prop_gnlse_records_the_default():
    run = prop_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], progress="none", seed=1, **GNLSE
    )
    metadata = run.metadata
    assert metadata is not None
    assert metadata.noise is not None
    assert metadata.noise_resolution == NoiseResolution("default", "input")
