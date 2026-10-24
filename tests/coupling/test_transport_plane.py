# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transport of a field plane: its record follows the field (SEP 31 §4.5-4.7)."""

import numpy as np
import pytest

from solprop.coupling import (
    Aperture,
    FieldPlane,
    FreeSpace,
    Mirror,
    PlanarInterface,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    SpectralTransfer,
    ThinLens,
    basis_identity,
    transport,
)
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import CouplingError
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.linear import Frame
from solprop.optics import GratingPair, TaylorPhase, littrow_angle
from solprop.processing import field_plane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium, aperture_profile
from solprop.stepper import PIController

ENERGY = 1e-6
CONDUCTOR = {"reflectivity_p": 1.0, "reflectivity_s": -1.0}


def _plane(grid, space, medium, waist=0.3e-3, jones=(1.0 + 0.0j,), geometry=None):
    """Return a Gaussian beam plane built by the package's beam builder."""
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY)
    return FieldPlane.from_pulse(
        SpatialPulse(pulse, waist, jones=jones),
        grid,
        PlaneGeometry(0.0) if geometry is None else geometry,
        space=space,
        medium=medium,
    )


@pytest.fixture(scope="module")
def radial_run():
    """A short linear radial run in argon, saved at 1 mm and 2 mm."""
    grid = EnvelopeGrid.from_limits(2e-3, 800e-9, (600e-9, 1.2e-6), 400e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-3, 2e-3],
        dz_max=2e-4,
        dz_init=2e-4,
        backend="numpy_ref",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 2.5e-5
    )
    medium = SpatialMedium("Ar", 1.0)
    simulation = Simulation.from_radial(
        grid, RadialGrid(3e-4, 64), medium, [], pulse, settings=settings
    )
    return simulation, simulation.run(progress="none")


def test_free_space_continues_a_radial_run(radial_run):
    simulation, output = radial_run
    start = field_plane(output, 1, source=simulation)
    end = field_plane(output, 2, source=simulation)
    # A 25 um waist diffracts measurably over the millimetre (z_R ~ 2.5 mm).
    on_axis = np.sum(end.node_spectra()[0]) / np.sum(start.node_spectra()[0])
    assert abs(on_axis) < 0.95
    gap = FreeSpace(1e-3, simulation.medium)
    # Both apply the exact k_z on the same QDHT columns and lattice, the run in five
    # exponential steps: they agree to the rounding of those products, ~1e-13 of
    # the field (measured 6e-13); 1e-11 leaves margin across platforms.
    framed, report = transport(start, [gap], frame=simulation.frame)
    assert framed.reference.calibrated == ("phase", "delay")
    assert framed.reference.phase == pytest.approx(end.reference.phase, rel=1e-12)
    assert framed.reference.delay == pytest.approx(end.reference.delay, rel=1e-12)
    scale = np.linalg.norm(end.samples)
    assert np.linalg.norm(framed.samples - end.samples) <= 1e-11 * scale
    assert framed.geometry.z == pytest.approx(end.geometry.z, abs=1e-15)
    assert framed.basis.identity == end.basis.identity
    assert framed.energy() == pytest.approx(end.energy(), rel=1e-12)
    assert report.frame == "item 0: the plane's frame"
    # Without the frame only the group delay is removed; the stored spectrum
    # differs by the carrier phase, which the reference keeps, so the absolute
    # spectra agree.
    plain, report = transport(start, [gap])
    assert "no frame given" in report.frame
    assert plain.reference.phase == start.reference.phase
    assert plain.reference.delay == pytest.approx(end.reference.delay, rel=1e-12)
    difference = np.linalg.norm(plain.absolute() - end.absolute())
    assert difference <= 1e-11 * np.linalg.norm(end.absolute())
    assert plain.provenance.history[-2] == "transport[0]: FreeSpace(L=0.001 m)"
    assert plain.provenance.source == start.provenance.source


def test_the_plane_record_follows_the_elements():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air = SpatialMedium()
    plane = _plane(grid, RadialGrid(4e-3, 128), air)
    pair = GratingPair(600e3, littrow_angle(600e3, 800e-9), 0.01, 800e-9)
    gdd = TaylorPhase((0.0, 0.0, 100e-30), 800e-9)
    focus, report = transport(
        plane,
        [
            ThinLens(1.0, model_error=1e-3),
            FreeSpace(0.25, air, frame_delay=1e-12, frame_phase=0.5),
            gdd,
            FreeSpace(0.25, air),
        ],
    )
    # Each gap adds what it removed: its declared term, or beta1 L of vacuum.
    assert focus.reference.delay == pytest.approx(1e-12 + 0.25 / 299792458.0)
    assert focus.reference.phase == 0.5
    # A Taylor phase keeps its affine terms, so the reference stays calibrated.
    assert focus.reference.calibrated == ("phase", "delay")
    assert report.reference_in == plane.reference
    assert report.reference_out == focus.reference
    assert focus.geometry.z == 0.5
    assert focus.geometry.origin == (0.0, 0.0, 0.5)
    assert focus.metric.flux is not None
    assert focus.energy() == pytest.approx(report.energy_out, rel=1e-12)
    assert abs(report.ledger_residual) <= 1e-14 * report.energy_in
    assert [step.split(":")[0] for step in focus.provenance.history[-5:]] == [
        "transport[0]",
        "transport[1]",
        "transport[2]",
        "transport[3]",
        "transport frame",
    ]
    # A grating pair removes its phase and group delay: both become unknown.
    compressed, _ = transport(focus, [SpectralTransfer(pair)])
    assert compressed.reference.calibrated == ()
    with pytest.raises(CouplingError, match="not known"):
        compressed.absolute()


def test_a_mirror_turns_the_plane_axes_with_the_polarisation():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air = SpatialMedium()
    space = FreeGrid(1.5e-3, 32, 1.5e-3, 32)
    plane = _plane(grid, space, air, jones=(1.0, 0.5j))
    fold = Mirror(**CONDUCTOR, incident=(0, 0, 1), outgoing=(1, 0, 0), s_axis=(0, 1, 0))
    folded, report = transport(plane, [fold, FreeSpace(0.1, air)])
    assert folded.geometry.normal == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)
    # The outgoing frame keeps the p/s components of the incoming axes: x (along
    # p_in = s x k_in) maps to p_out = s x k_out = -z.
    assert folded.geometry.x_axis == pytest.approx((0.0, 0.0, -1.0), abs=1e-15)
    assert np.cross(folded.geometry.x_axis, folded.geometry.y_axis) == pytest.approx(
        folded.geometry.normal
    )
    # The gap after the fold moves the plane along the new normal.
    assert folded.geometry.origin == pytest.approx((0.1, 0.0, 0.0), abs=1e-15)
    assert folded.samples.shape[1] == 2
    assert folded.energy() == pytest.approx(report.energy_out, rel=1e-12)
    declared = PlaneGeometry(
        0.0, normal=folded.geometry.normal, x_axis=folded.geometry.x_axis
    )
    received, _ = transport(plane, [fold, FreeSpace(0.1, air)], geometry=declared)
    assert received.geometry is declared
    with pytest.raises(CouplingError, match="arrives with normal"):
        transport(plane, [fold], geometry=PlaneGeometry(0.0))


def test_an_interface_changes_the_plane_medium_and_metric():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air, glass = SpatialMedium(), SpatialMedium(1.5)
    plane = _plane(grid, RadialGrid(4e-3, 64), air)
    inside, report = transport(plane, [PlanarInterface(air, glass)])
    assert inside.medium is glass
    assert inside.energy() == pytest.approx(0.96 * ENERGY, rel=1e-7)
    assert report.specified_loss == pytest.approx(0.04 * ENERGY, rel=1e-7)


@pytest.mark.parametrize("weighting", ["amplitude", "quadrature"])
def test_aperture_weighting_matches_the_in_run_statistics(weighting):
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air = SpatialMedium()
    space = RadialGrid(2e-3, 128)
    plane = _plane(grid, space, air, waist=0.5e-3)
    radius, edge = 0.4e-3, 0.1e-3
    aperture = Aperture(
        radius, model_error=0.05, edge=edge, kind="tanh", weighting=weighting
    )
    clipped, report = transport(plane, [aperture])
    profile = aperture_profile(np.asarray(space.r), radius, edge, "tanh")
    factor = profile if weighting == "amplitude" else np.sqrt(profile)
    before, after = plane.node_spectra(), clipped.node_spectra()
    inside = space.r < radius
    # The screen multiplies the node samples; the QDHT round trip that carries it
    # back to columns is accurate to ~1e-9 of the field on this grid.
    np.testing.assert_allclose(
        after[inside],
        factor[inside, None, None] * before[inside],
        rtol=0,
        atol=1e-8 * np.max(np.abs(before)),
    )
    assert weighting in report.elements[0].element


def test_the_two_weightings_differ_on_a_soft_edge():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air = SpatialMedium()
    plane = _plane(grid, RadialGrid(2e-3, 128), air, waist=0.5e-3)
    energies = {
        weighting: transport(
            plane,
            [
                Aperture(
                    0.4e-3,
                    model_error=0.05,
                    edge=0.1e-3,
                    kind="tanh",
                    weighting=weighting,
                )
            ],
        )[0].energy()
        for weighting in ("amplitude", "quadrature")
    }
    # W <= 1, so W^2 <= W: the amplitude reading transmits less.
    assert energies["amplitude"] < 0.99 * energies["quadrature"]
    # The literal is checked at the call in the test session; a value set past it
    # is still refused by the element.
    unknown = Aperture(0.4e-3, model_error=0.05)
    object.__setattr__(unknown, "weighting", "field")
    with pytest.raises(CouplingError, match="weighting"):
        transport(plane, [unknown])


def test_transport_refuses_what_it_cannot_carry():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air = SpatialMedium()
    basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["test"]}))
    samples = np.zeros((1, 1, grid.n_omega), complex)
    modal = FieldPlane.from_samples(
        samples, grid, PlaneGeometry(0.0), exterior="zero", basis=basis
    )
    with pytest.raises(CouplingError, match="Outcouple"):
        transport(modal, [FreeSpace(0.1, air)])
    space = RadialGrid(4e-3, 64)
    plane = _plane(grid, space, air)
    bare = FieldPlane(
        plane.kind,
        plane.samples,
        plane.grid,
        plane.units,
        plane.basis,
        plane.reference,
        plane.geometry,
        plane.metric,
    )
    with pytest.raises(CouplingError, match="source=simulation"):
        transport(bare, [])
    nodes = plane.node_spectra()
    cropped = FieldPlane.from_samples(
        nodes, grid, PlaneGeometry(0.0), exterior="unknown", space=space, medium=air
    )
    with pytest.raises(CouplingError, match="unknown"):
        transport(cropped, [FreeSpace(0.1, air)])
    # A local screen needs no exterior; the record is carried as it is.
    lensed, _ = transport(cropped, [ThinLens(1.0, model_error=1e-3)])
    assert lensed.exterior == "unknown"
    with pytest.raises(CouplingError, match="neither an element"):
        transport(plane, [object()])


def test_a_fixed_frame_sets_the_removed_affine_term():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    air = SpatialMedium()
    plane = _plane(grid, RadialGrid(4e-3, 64), air)
    gap = FreeSpace(0.2, air)
    fixed = Frame(grid.omega0, subtract_beta0=True, beta0=3.0, beta1=4e-9)
    moved, _ = transport(plane, [gap], frame=fixed)
    # The run's reference law: phase (beta0 - omega_o beta1) L, delay beta1 L.
    assert moved.reference.delay == pytest.approx(4e-9 * 0.2, rel=1e-15)
    assert moved.reference.phase == pytest.approx(
        (3.0 - grid.omega0 * 4e-9) * 0.2, rel=1e-15
    )
    kept = Frame(grid.omega0, subtract_beta0=False)
    moved, _ = transport(plane, [gap], frame=kept)
    assert moved.reference.phase == pytest.approx(
        -grid.omega0 * 0.2 / 299792458.0, rel=1e-12
    )
    back, _ = transport(moved, [FreeSpace(-0.2, air)], frame=kept)
    assert back.provenance.history[-2].endswith("(reconstruction)")
    assert back.reference.phase == pytest.approx(0.0, abs=1e-9)
    assert back.reference.delay == pytest.approx(0.0, abs=1e-24)


def test_a_planar_plane_is_refused():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    basis = PlaneBasis("planar", basis_identity("planar", {}))
    reference = _plane(grid, RadialGrid(4e-3, 16), SpatialMedium())
    planar = FieldPlane(
        "planar",
        np.zeros((4, 1, grid.n_omega), complex),
        grid,
        "sqrt(W/m)",
        basis,
        reference.reference,
        reference.geometry,
        PlaneMetric("identity"),
    )
    with pytest.raises(CouplingError, match="planar plane"):
        transport(planar, [])
