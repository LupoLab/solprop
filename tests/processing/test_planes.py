# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Saved planes read back as coherent field planes (SEP 31)."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.coupling import reference_at
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import ProcessingError
from solprop.fourier import spectrum_to_internal, spectrum_to_optics
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.modes import (
    GradedIndexProvider,
    IndexMaterial,
    MaterialKerr,
    ModalGuide,
    PowerLawPermittivity,
)
from solprop.output.hdf5 import open_file
from solprop.output.memory import MemoryOutput
from solprop.output.schema import SavedPlane, StepHistory
from solprop.processing import electric_field, field_plane, magnetic_field
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import analytic_field
from solprop.stepper import PIController

EPS = np.finfo(float).eps


def capillary(length=2e-3, n_save=3, **changes):
    arguments = {
        "lambda0": 800e-9,
        "wavelength_limits": (400e-9, 1.6e-6),
        "trange": 200e-15,
        "fwhm": 20e-15,
        "energy": 20e-6,
        "modes": 2,
        "n_save": n_save,
        "plasma": False,
        "noise": None,
        "backend": "numpy_ref",
        **changes,
    }
    return Simulation.from_capillary(100e-6, length, "Ar", 1.0, **arguments)


def spatial(space, pressure: float | PressureGradient = 1.0):
    grid = FieldGrid.from_limits(2e-3, 800e-9, (600e-9, 1.2e-6), 400e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-3, 2e-3],
        dz_max=2e-4,
        dz_init=2e-4,
        backend="numpy_ref",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    medium = SpatialMedium("Ar", pressure)
    if isinstance(space, RadialGrid):
        return Simulation.from_radial(grid, space, medium, [], pulse, settings=settings)
    return Simulation.from_free(grid, space, medium, [], pulse, settings=settings)


@pytest.fixture(scope="module")
def capillary_run():
    simulation = capillary()
    return simulation, simulation.run(progress="none")


def test_a_modal_plane_is_the_stored_spectrum_in_its_rebuilt_reference(capillary_run):
    simulation, output = capillary_run
    plane = field_plane(output)
    assert plane.kind == "modal" and plane.units == "sqrt(W)"
    # The plane holds the stored array, a copy: an exact structural contract.
    assert np.array_equal(plane.samples, output.fields()[-1])
    assert plane.geometry.z == float(output["z"][-1])
    assert plane.reference == reference_at(output.metadata.reference, plane.geometry.z)
    # The group delay of 2 mm of argon-filled capillary, about 2 mm / c.
    assert plane.reference.delay == pytest.approx(2e-3 / 2.998e8, rel=1e-3)
    assert plane.metric.kind == "identity"
    assert plane.basis.labels == tuple(
        mode.channel_label.display_label for mode in simulation.modes
    )
    assert plane.basis.identity == simulation.input_port().basis.identity
    provenance = plane.provenance
    assert provenance.source == output.metadata.configuration_hash()
    assert (provenance.plane, provenance.z, provenance.lineage) == (2, 2e-3, ())
    middle = field_plane(output, z=1e-3)
    assert middle.provenance.plane == 1
    assert middle.reference.delay == pytest.approx(
        0.5 * plane.reference.delay, rel=1e-9
    )


def test_a_selection_names_one_stored_plane_or_is_refused(capillary_run):
    _, output = capillary_run
    with pytest.raises(ProcessingError, match="no plane is stored"):
        field_plane(output, z=1.5e-3)
    with pytest.raises(ProcessingError, match="not both"):
        field_plane(output, 0, z=1e-3)
    with pytest.raises(ProcessingError, match="out of range"):
        field_plane(output, 5)


def test_a_file_without_the_schema_24_record_is_refused(capillary_run):
    _, output = capillary_run
    older = MemoryOutput()
    older.open(dataclasses.replace(output.metadata, schema_version=23, reference=None))
    with pytest.raises(ProcessingError, match="schema 23"):
        field_plane(older)
    empty = MemoryOutput()
    with pytest.raises(ProcessingError, match="run record"):
        field_plane(empty)


def test_the_record_survives_a_file(capillary_run, tmp_path):
    simulation, memory = capillary_run
    path = tmp_path / "capillary.h5"
    simulation.run(path, progress="none")
    with open_file(path) as stored:
        assert stored.metadata is not None and memory.metadata is not None
        assert stored.metadata.schema_version == 24
        assert stored.metadata.reference == memory.metadata.reference
        assert (
            stored.metadata.configuration_hash() == memory.metadata.configuration_hash()
        )
        plane = field_plane(stored)
    expected = field_plane(memory)
    # A run repeated on the same backend; its stored planes agree to its rounding.
    assert_allclose(
        plane.samples,
        expected.samples,
        rtol=0,
        atol=1e-12 * np.max(np.abs(expected.samples)),
    )
    assert plane.reference == expected.reference
    assert plane.basis.identity == expected.basis.identity


@pytest.mark.parametrize(
    "space", [RadialGrid(6e-4, 32), FreeGrid(4e-4, 16, 4e-4, 16)], ids=["radial", "xy"]
)
def test_a_spatial_plane_carries_its_medium_s_measure(space, tmp_path):
    simulation = spatial(space)
    path = tmp_path / "spatial.h5"
    memory = simulation.run(progress="none")
    simulation.run(path, progress="none")
    for output in (memory, open_file(path)):
        plane = field_plane(output, 1)
        assert plane.kind == (
            "radial" if isinstance(space, RadialGrid) else "cartesian"
        )
        assert plane.metric.kind == "admittance" and plane.metric.flux is None
        # The run's own energy statistic reads the same plane measure (SEP 34);
        # two orders of the same weighted sum.
        energy = float(output["stats/energy"][1])
        assert plane.energy() == pytest.approx(energy, rel=1e-12)
        assert plane.certificate is None
        assert plane.basis.identity == simulation.input_port().basis.identity
    with_source = field_plane(memory, 1, source=simulation)
    assert with_source.medium is simulation.medium
    assert with_source.metric.flux is not None
    other = spatial(space, pressure=2.0)
    with pytest.raises(ProcessingError, match="not the simulation"):
        field_plane(memory, 1, source=other)


def test_a_gradient_plane_carries_its_scoped_path_certificate():
    simulation = spatial(
        RadialGrid(6e-4, 32), PressureGradient.two_point(2e-3, 1.0, 2.0)
    )
    plane = field_plane(simulation.run(progress="none"))
    assert plane.certificate is not None
    assert plane.certificate["scope"] == "linear-background forward field"
    with pytest.raises(TypeError):
        plane.certificate["scope"] = "anything"  # pyright: ignore[reportIndexIssue]


@pytest.fixture(scope="module")
def general_run():
    wavelength, length, contrast = 1.03e-6, 1e-5, 0.0095
    grid = FieldGrid.from_limits(length, wavelength, (1.0e-6, 1.06e-6), 100e-15)
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), contrast),
        exterior=IndexMaterial(1.45 * np.sqrt(1.0 - 2.0 * contrast)),
        radius=25e-6,
        family="LP",
        count=1,
        elements=16,
        extent=2.0,
        core_responses=(MaterialKerr(n2=2.7e-20, at=wavelength),),
    )
    active = grid.omega[grid.active_mask]
    knots = np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 9)
    guide = ModalGuide.from_provider(
        provider,
        knots,
        nodes=provider.node_rule(float(knots[0]), 0.0, n_r=2, n_theta=2),
    )
    simulation = Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        lambda0=wavelength,
        fwhm=30e-15,
        energy=1e-12,
        dz_max=length / 2,
        dz_init=length / 2,
        n_save=2,
        backend="numpy_ref",
    )
    return simulation, simulation.run(progress="none")


def test_a_general_plane_reads_its_fields_with_its_source(general_run):
    simulation, output = general_run
    bare = field_plane(output)
    assert bare.kind == "general_modal" and bare.nodal is None
    assert bare.metric.kind == "identity"
    plane = field_plane(output, source=simulation)
    assert plane.nodal is not None
    tables, guide = simulation.general.tables, simulation.guide
    # The nodal spectra are processing.modal's, in the optics convention: through
    # the one boundary and back they give its analytic signals exactly.
    grid = simulation.grid
    for accessor, stored in (
        (electric_field, plane.nodal.electric),
        (magnetic_field, plane.nodal.magnetic),
    ):
        assert stored is not None
        signal = accessor(
            output, tables, plane=-1, analytic=True, convention="fft", guide=guide
        )
        rebuilt = analytic_field(grid, spectrum_to_internal(stored, kind=grid.kind))
        assert np.array_equal(rebuilt, signal)
    assert plane.basis.identity == simulation.input_port().basis.identity
    port = simulation.input_port()
    assert port.electric is not None and port.nodes is not None
    with pytest.raises(ProcessingError, match="general modal guide"):
        field_plane(output, source=capillary())


def test_planar_runs_are_refused_by_name(capillary_run):
    _, output = capillary_run
    reference = dict(output.metadata.reference)
    reference["basis"] = {**reference["basis"], "route": "planar"}
    planar = MemoryOutput()
    planar.open(dataclasses.replace(output.metadata, reference=reference))
    for plane in output.fields():
        planar.save(dataclasses.replace(_saved(output), field=_internal(plane)))
    with pytest.raises(ProcessingError, match="planar"):
        field_plane(planar)


def _saved(output):
    return SavedPlane(z=0.0, field=None, step_history=StepHistory.empty(), stats={})


def _internal(plane):
    return spectrum_to_internal(plane, kind="field")


def test_the_launch_plane_is_the_launched_spectrum_in_the_optics_convention(
    capillary_run,
):
    simulation, output = capillary_run
    plane = field_plane(output, 0)
    expected = spectrum_to_optics(simulation._field0, kind="field")
    # The true-power boundary there and back: a square root and two products.
    assert_allclose(
        plane.samples, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected))
    )
