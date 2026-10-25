# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Radial input, noise, output geometry and resume compatibility."""

import dataclasses
import hashlib
import json
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from flux_oracles import admittance
from solprop.backends.registry import available
from solprop.constants import EPS0, HBAR, C
from solprop.driver import Simulation, SimulationSettings, prop_radial
from solprop.exceptions import DriverError, NoiseError, OutputError
from solprop.fourier import spectral_density
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.noise import spatial_noise
from solprop.output.hdf5 import HDF5Output, check_cache, open_file
from solprop.output.schema import RunMetadata, SavedPlane
from solprop.physdata.selection import MaterialManifest
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy
from solprop.stepper import PIController
from solprop.translate.linop import delta_phi
from spatial_records import record_metric


def _vacuum(grid, space):
    """Return the vacuum measure of the plane at z = 0 (SEP 34)."""
    return SpatialFlux.resolve(grid, space, SpatialMedium())


EPS = np.finfo(float).eps


@pytest.fixture(params=["field", "envelope"])
def case(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(6e-4, 32)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    return grid, space, pulse, settings


@pytest.mark.parametrize("backend", available())
def test_radial_interface_caches_and_saves_geometry(case, backend, request):
    grid, space, pulse, settings = case
    settings = dataclasses.replace(settings, backend=backend)
    simulation = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings
    )
    assert simulation.translate() is simulation.translate()
    metadata = simulation.metadata()
    # Preserve the compact pulse specification; serializing the built transverse
    # volume here would duplicate the input and make Cartesian metadata enormous.
    assert metadata.configuration["initial"] is pulse
    assert metadata.schema_version == 26
    assert metadata.simulation_kind == "radial"
    assert metadata.spatial is not None
    assert metadata.spatial["stored_field_units"] == "V m"
    assert metadata.spatial["spectral_axis_names"] == ("k_r",)
    assert (
        RunMetadata.from_json(metadata.to_json()).configuration_hash()
        == metadata.configuration_hash()
    )
    handler = prop_radial(grid, space, SpatialMedium(), [], pulse, settings=settings)
    fields = handler.fields("fft")
    assert fields.shape == (2, space.n_r, 1, grid.n_omega)
    # Two energy reductions and a diagonal unitary propagation. The input is
    # masked by the driver; compare with its saved entrance, not an unmasked pulse.
    bound = (16 * space.n_r + 8 * grid.n_omega) * EPS
    assert_allclose(
        spatial_energy(fields[-1], _vacuum(grid, space)),
        spatial_energy(fields[0], _vacuum(grid, space)),
        rtol=bound,
        atol=0,
    )
    record_metric(
        request,
        "radial interface energy (relative)",
        grid.kind,
        abs(
            spatial_energy(fields[-1], _vacuum(grid, space))
            / spatial_energy(fields[0], _vacuum(grid, space))
            - 1
        ),
        bound,
        backend=backend,
    )


@pytest.mark.parametrize("model", ["input", "modified"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_spatial_noise_has_one_photon_in_every_column_bin(case, model, n_pol, request):
    grid, space, _, _ = case
    # sqrt normalization, complex phase, squared modulus and density multiplication
    # involve fewer than 32 rounded operations per bin; there is no long reduction.
    bound = 32 * EPS / (1 - 32 * EPS)
    draw = spatial_noise(
        _vacuum(grid, space), np.random.default_rng(313), model=model, n_pol=n_pol
    )
    # A bin's photon is measured in the entrance metric, its real admittance (the
    # vacuum obliquity) times the column's area weight; a bin that is no forward
    # power port carries none (SEP 34 section 4.3).
    eta = admittance(grid, space.k_r**2)
    measured = (
        0.5
        * EPS0
        * C
        * space.spectral_weights[:, None, None]
        * eta[:, None, :]
        * spectral_density(draw.spectrum, grid.dt, kind=grid.kind)
        * grid.d_omega
    )
    wanted = np.broadcast_to(
        (HBAR * grid.omega * (eta > 0))[:, None, :], measured.shape
    )
    assert_allclose(measured, wanted, rtol=bound, atol=0)
    errors = np.divide(
        np.abs(measured - wanted), wanted, out=np.zeros_like(wanted), where=wanted > 0
    )
    record_metric(
        request,
        "spatial noise photons per bin (relative)",
        f"{grid.kind}, {model}, p={n_pol}",
        float(np.max(errors)),
        bound,
    )
    other = spatial_noise(
        _vacuum(grid, space),
        np.random.default_rng(313),
        model="modified" if model == "input" else "input",
        n_pol=n_pol,
    )
    np.testing.assert_array_equal(draw.spectrum, other.spectrum)
    assert draw.seed == 313 and not draw.spectrum.flags.writeable


@pytest.mark.parametrize("model", ["input", "modified"])
@pytest.mark.parametrize("backend", available())
def test_radial_simulation_records_noise_model_and_seed(case, model, backend, request):
    grid, space, pulse, settings = case
    settings = dataclasses.replace(settings, backend=backend)
    simulation = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=model, seed=33
    )
    assert simulation.noise is not None
    assert simulation.noise.seed == 33
    assert simulation.translate().initial.noise_model == model
    noise = simulation.metadata().noise
    assert noise is not None and noise.seed == 33
    # With no nonlinear source, both noise models obey the exact linear flow
    # of their saved entrance. The phase subtraction and complex exponential
    # cost fewer than 64 rounded operations per optical phase magnitude.
    phase = delta_phi(simulation.translate().linear, 0.0, grid.length)[0]
    bound = 64 * EPS * (1 + float(np.max(np.abs(phase))))
    fields = simulation.run().fields("fft")
    expected = np.exp(phase) * fields[0]
    assert_allclose(fields[-1], expected, rtol=bound, atol=0)
    errors = np.divide(
        np.abs(fields[-1] - expected),
        np.abs(expected),
        out=np.zeros(expected.shape),
        where=np.abs(expected) > 0,
    )
    record_metric(
        request,
        "spatial noise linear flow (relative)",
        f"{grid.kind}, {model}",
        float(np.max(errors)),
        bound,
        backend=backend,
    )


def test_spatial_resume_refuses_geometry_changes(case, tmp_path):
    grid, space, pulse, settings = case
    settings = dataclasses.replace(settings, backend="numpy_ref")
    simulation = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings
    )
    path = tmp_path / "radial.h5"
    simulation.run(path)
    assert check_cache(path, simulation.metadata())
    simulation.run(path, resume=True)
    changed = Simulation.from_radial(
        grid,
        dataclasses.replace(space, radius=space.radius * 2),
        SpatialMedium(),
        [],
        pulse,
        settings=settings,
    )
    with pytest.raises(OutputError, match="spatial"):
        changed.run(path, resume=True)


def test_version_three_file_retains_its_original_hash_and_reads(case, tmp_path):
    grid, space, pulse, settings = case
    # A version-three file keys on no manifest: every run records its selection now
    # (the registry selection that once made such a run is deleted, SEP 3 Phase 4b),
    # so the legacy record is given the empty manifest directly.
    simulation = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings
    )
    current = simulation.metadata()
    legacy = dataclasses.replace(
        current,
        schema_version=3,
        simulation_kind="multimode",
        spatial=None,
        materials=MaterialManifest(),
    )
    # Recreate the literal v3 hash independently, without a spatial key. This is
    # exactly the legacy format; pretending a v4 hash were v3 would not test reading.
    document = json.loads(legacy.to_json())
    assert "spatial" not in document
    keys = (
        "schema_version",
        "simulation_kind",
        "grid",
        "modes",
        "configuration",
        "noise",
        "stepper",
    )
    payload = json.dumps(
        {key: document[key] for key in keys},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    )
    assert legacy.configuration_hash() == hashlib.sha256(payload.encode()).hexdigest()
    path = tmp_path / "legacy.h5"
    with HDF5Output(path) as handler:
        handler.open(legacy)
        handler.save(SavedPlane(z=0.0, field=simulation.translate().initial.field0[0]))
    with open_file(path) as stored:
        assert stored.n_saved == 1
        assert stored.fields("fft").shape[0] == 1
    assert check_cache(path, legacy)
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(path, current)


def test_radial_input_rejects_ambiguous_noise_and_shape(case):
    grid, space, pulse, settings = case
    with pytest.raises(DriverError, match="seed"):
        Simulation.from_radial(
            grid, space, SpatialMedium(), [], pulse, settings=settings, seed=1
        )
    with pytest.raises(DriverError, match="shape"):
        Simulation.from_radial(
            grid,
            space,
            SpatialMedium(),
            [],
            np.zeros((1, 1, grid.n_omega)),
            settings=settings,
        )
    with pytest.raises(NoiseError, match="n_pol"):
        spatial_noise(_vacuum(grid, space), np.random.default_rng(33), n_pol=3)
