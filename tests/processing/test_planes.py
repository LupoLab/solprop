# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Saved planes read back as coherent field planes (SEP 31)."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.adiabatic import AdiabaticAdmission
from solprop.coupling import basis_identity, reference_at
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import CouplingError, ProcessingError
from solprop.fourier import spectrum_to_internal, spectrum_to_optics
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.modes import (
    GradedIndexProvider,
    IndexMaterial,
    MarcatiliMode,
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
from solprop.translate.coupling import input_port

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

EPS = np.finfo(float).eps


def capillary(length=2e-3, n_save=3, **changes):
    arguments = {
        "wavelength": 800e-9,
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
    return Simulation.from_capillary(
        100e-6, length, "Ar", 1.0, **{"noise": None, **arguments}
    )


def spatial(space, pressure: float | PressureGradient = 1.0):
    grid = FieldGrid.from_limits(2e-3, 800e-9, (600e-9, 1.2e-6), 400e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-3, 2e-3],
        dz_max=2e-4,
        dz_init=2e-4,
        backend="numpy_ref",
        # The coarse box launches into the safety ramp's band (SEP 40 section 5.3).
        spatial_absorber=False,
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    medium = SpatialMedium("Ar", pressure)
    if isinstance(space, RadialGrid):
        return Simulation.from_radial(
            grid, space, medium, [], pulse, settings=settings, noise=None
        )
    return Simulation.from_free(
        grid, space, medium, [], pulse, settings=settings, noise=None
    )


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


def test_a_record_without_its_plane_reference_is_refused(capillary_run):
    _, output = capillary_run
    bare = MemoryOutput()
    bare.open(dataclasses.replace(output.metadata, reference=None))
    with pytest.raises(ProcessingError, match="no plane-reference record"):
        field_plane(bare)
    empty = MemoryOutput()
    with pytest.raises(ProcessingError, match="run record"):
        field_plane(empty)


def test_the_record_survives_a_file(capillary_run, tmp_path):
    simulation, memory = capillary_run
    path = tmp_path / "capillary.h5"
    simulation.run(path, progress="none")
    with open_file(path) as stored:
        assert stored.metadata is not None and memory.metadata is not None
        assert stored.metadata.schema_version == 2
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


def test_a_gradient_plane_carries_its_rebuilt_path_certificate(tmp_path):
    simulation = spatial(
        RadialGrid(6e-4, 32), PressureGradient.two_point(2e-3, 1.0, 2.0)
    )
    admitted = simulation.translate().flux
    assert admitted is not None and admitted.admission is not None
    expected = admitted.admission
    path = tmp_path / "gradient.h5"
    memory = simulation.run(progress="none")
    simulation.run(path, progress="none")
    for output in (memory, open_file(path)):
        plane = field_plane(output)
        certificate = plane.certificate
        assert isinstance(certificate, AdiabaticAdmission)
        # The record rebuilds the run's own certificate: its numbers are copied
        # through the file, so they are exact; the path's identity comes back as
        # the text it was recorded as.
        assert certificate.z_span == expected.z_span
        assert certificate.variation == expected.variation
        assert certificate.field_error == expected.field_error
        assert certificate.reflection == expected.reflection
        assert certificate.accuracy == expected.accuracy
        assert np.array_equal(certificate.support, expected.support)
        assert certificate.identity == {"recorded": repr(dict(expected.identity))}
        with pytest.raises(dataclasses.FrozenInstanceError):
            certificate.variation = 0.0  # pyright: ignore[reportAttributeAccessIssue]


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
        wavelength=wavelength,
        fwhm=30e-15,
        energy=1e-12,
        dz_max=length / 2,
        dz_init=length / 2,
        n_save=2,
        backend="numpy_ref",
        noise=None,
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


def rewritten(output, **changes):
    """Return a memory copy of an output whose run record has the given changes."""
    copy = MemoryOutput()
    copy.open(dataclasses.replace(output.metadata, **changes))
    for saved in output.planes():
        copy.save(saved)
    return copy


def test_a_modified_noise_run_is_refused_by_name_and_an_input_noise_run_read():
    output = capillary(length=5e-4, n_save=2, noise="modified").run(progress="none")
    with pytest.raises(ProcessingError, match=r"modified noise model.*noise_phase"):
        field_plane(output)
    # The same planes under the input model carry their background in the saved
    # field itself, so they are exchanged as stored.
    assert output.metadata is not None
    assert output.metadata.noise is not None
    noise = dataclasses.replace(output.metadata.noise, model="input")
    plane = field_plane(rewritten(output, noise=noise))
    assert plane.geometry.z == pytest.approx(5e-4, rel=0, abs=1e-15)


def test_an_output_without_a_saved_plane_is_refused(capillary_run):
    _, output = capillary_run
    empty = MemoryOutput()
    empty.open(output.metadata)
    with pytest.raises(ProcessingError, match="no saved plane"):
        field_plane(empty)


def test_a_source_that_is_not_the_writing_simulation_is_refused(capillary_run):
    _, output = capillary_run
    with pytest.raises(ProcessingError, match="object has no metadata"):
        field_plane(output, source=object())
    with pytest.raises(ProcessingError, match="not the simulation"):
        field_plane(output, source=capillary(length=3e-3))


def test_a_mode_family_with_angular_structure_has_no_scalar_profiles():
    modes = [
        MarcatiliMode(100e-6, "Ar", 1.0),
        MarcatiliMode(100e-6, "Ar", 1.0, kind="TE", n=0),
    ]
    simulation = capillary(length=1e-3, n_save=2, modes=modes)
    plane = field_plane(simulation.run(progress="none"), source=simulation)
    # The TE01 mode is not a scalar HE1m profile: the plane carries none, and the
    # scalar model is not offered for it.
    assert plane.kind == "modal" and plane.profiles is None


def test_a_parent_record_without_its_plane_gives_no_lineage(capillary_run):
    _, output = capillary_run
    reference = {**output.metadata.reference, "parent": {"model": "same_basis"}}
    plane = field_plane(rewritten(output, reference=reference))
    assert plane.provenance.lineage == ()


def test_a_modal_plane_saved_off_the_planned_positions_has_an_unrecorded_basis(
    capillary_run,
):
    simulation, output = capillary_run
    reference = dict(output.metadata.reference)
    bases = reference["basis"]
    # Every planned position moved a tenth of a millimetre: no saved plane is one.
    reference["basis"] = {**bases, "z": [one + 1e-4 for one in bases["z"]]}
    plane = field_plane(rewritten(output, reference=reference))
    assert plane.basis.identity == basis_identity("modal", {"unrecorded at": 2e-3})
    assert plane.basis.identity != simulation.input_port().basis.identity


@pytest.fixture(scope="module")
def radial_run():
    return spatial(RadialGrid(6e-4, 32)).run(progress="none")


def test_a_spatial_record_without_a_known_grid_or_measure_is_refused(radial_run):
    metadata = radial_run.metadata
    configuration = dict(metadata.configuration)
    del configuration["spatial_grid"]
    with pytest.raises(ProcessingError, match="no transverse grid"):
        field_plane(rewritten(radial_run, configuration=configuration))
    configuration["spatial_grid"] = {"__class__": "solprop.modes.PlanarGrid"}
    with pytest.raises(ProcessingError, match="not exchanged"):
        field_plane(rewritten(radial_run, configuration=configuration))
    spatial_record = dict(metadata.spatial)
    del spatial_record["flux_plan"]
    with pytest.raises(ProcessingError, match="no plane measure"):
        field_plane(rewritten(radial_run, spatial=spatial_record))


def test_a_general_plane_without_its_source_has_an_unknown_dense_metric(general_run):
    _, output = general_run
    modal = dict(output.metadata.modal)
    modal["metric"] = {**modal["metric"], "kind": "dense"}
    plane = field_plane(rewritten(output, modal=modal))
    assert plane.metric.kind == "unknown" and "source=" in plane.metric.reason
    with pytest.raises(CouplingError, match="unknown"):
        plane.energy()


def test_a_general_source_with_other_tables_is_refused(general_run):
    simulation, output = general_run
    modal = dict(output.metadata.modal)
    modal["tables"] = {**modal["tables"], "gauge_id": "another gauge"}
    with pytest.raises(ProcessingError, match="not the reference"):
        field_plane(rewritten(output, modal=modal), source=simulation)


def test_a_general_port_admits_the_vector_model_only_with_a_fixed_reference_and_guide(
    general_run,
):
    simulation, _ = general_run
    problem, grid, tables = (
        simulation.translate(),
        simulation.grid,
        simulation.general.tables,
    )
    port = simulation.input_port()
    assert port.models == ("same_basis", "vector_paraxial")
    common = {"route": "general_modal", "basis": port.basis, "frame": None}
    bare = input_port(problem, grid, tables=tables, **common)
    # Without its guide the port carries the E/H reference but no contrast, so it
    # admits only the identity model.
    assert bare.models == ("same_basis",)
    assert bare.electric is not None and bare.contrast is None
    # Tables whose synthesis varies along z (two degrees, as a tapered guide's):
    # there is no one reference to sample, and the identity model alone is offered.
    assert tables.synthesis is not None and tables.projection is not None
    moving = dataclasses.replace(
        tables,
        synthesis=np.concatenate([tables.synthesis] * 2, axis=1),
        projection=np.concatenate([tables.projection] * 2, axis=1),
    )
    tapered = input_port(problem, grid, tables=moving, guide=simulation.guide, **common)
    assert tapered.models == ("same_basis",)
    assert tapered.electric is None and tapered.nodes is None
