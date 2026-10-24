# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The field plane, its references, geometry, metric and identities (SEP 31)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.coupling import (
    FieldPlane,
    NodalField,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    PlaneProvenance,
    PlaneReference,
    ScalarModeBasis,
    basis_identity,
    lattice_identity,
    modal_basis_identity,
    reference_at,
    space_record,
)
from solprop.coupling.plane import _frozen
from solprop.exceptions import CouplingError
from solprop.fourier import spectrum_to_optics
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.output.schema import serialise
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy

EPS = np.finfo(float).eps


def envelope_grid(trange=1e-12):
    return EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1.2e-6), trange)


def modal_plane(grid=None, **changes):
    grid = envelope_grid() if grid is None else grid
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    samples = spectrum_to_optics(pulse.build(grid), kind=grid.kind)
    basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["test"]}))
    return FieldPlane.from_samples(
        samples, grid, PlaneGeometry(0.0), exterior="zero", basis=basis, **changes
    )


def test_a_plane_s_geometry_is_right_handed():
    geometry = PlaneGeometry(0.1, normal=(1.0, 0.0, 0.0), x_axis=(0.0, 1.0, 0.0))
    assert geometry.y_axis == (0.0, 0.0, 1.0)
    assert geometry.oriented_as(geometry)
    assert not geometry.oriented_as(PlaneGeometry(0.1))


@pytest.mark.parametrize(
    "changes",
    [
        {"normal": (0.0, 0.0, 2.0)},
        {"x_axis": (0.0, 0.0, 1.0)},
        {"z": float("nan")},
    ],
)
def test_a_plane_s_geometry_refuses_what_is_not_a_frame(changes):
    arguments = {"z": 0.0, **changes}
    with pytest.raises(CouplingError):
        PlaneGeometry(**arguments)


def test_a_reference_composes_and_keeps_only_shared_calibrations():
    one = PlaneReference(0.5, 2e-12)
    two = PlaneReference(0.25, 1e-12, ("delay",))
    difference = one.minus(two)
    assert (difference.phase, difference.delay) == (0.25, 1e-12)
    assert difference.calibrated == ("delay",)
    assert one.plus(two).calibrated == ("delay",)
    assert PlaneReference.from_record(one.record()) == one
    omega = np.array([0.0, 2e15])
    # One exponential of a product of two doubles: a few rounding errors.
    assert_allclose(
        one.factor(omega), np.exp(1j * (0.5 + omega * 2e-12)), rtol=8 * EPS, atol=0
    )


@pytest.mark.parametrize("arguments", [(float("inf"), 0.0), (0.0, float("nan"))])
def test_a_reference_refuses_nonfinite_values(arguments):
    with pytest.raises(CouplingError):
        PlaneReference(*arguments)


def test_the_persisted_record_rebuilds_the_reference():
    frame = {
        "frequency_origin": 2e15,
        "phase": {"breakpoints": [0.0, 1.0], "coefficients": [[0.5, 0.5]]},
        "delay": {"breakpoints": [0.0, 1.0], "coefficients": [[1e-9, 1e-9]]},
    }
    record = {"origin": PlaneReference(0.1, 1e-12).record(), "frame": frame}
    # On [0, 1] the series 0.5 (T0 + T1) is z, so Phi0(z) = z and T(z) = 2e-9 z.
    reference = reference_at(record, 0.5)
    assert reference.delay == pytest.approx(1e-12 + 1e-9, rel=1e-15)
    assert reference.phase == pytest.approx(0.1 + 0.5 - 2e15 * 1e-9, rel=1e-14)
    assert reference.calibrated == ("phase", "delay")
    unknown = reference_at({"origin": record["origin"], "frame": None}, 0.5)
    assert unknown.calibrated == ()
    with pytest.raises(CouplingError, match="outside the frame record"):
        reference_at(record, 1.5)


def test_a_plane_owns_read_only_samples():
    grid = envelope_grid()
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    samples = spectrum_to_optics(pulse.build(grid), kind=grid.kind)
    basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["test"]}))
    plane = FieldPlane.from_samples(
        samples, grid, PlaneGeometry(0.0), exterior="zero", basis=basis
    )
    assert not plane.samples.flags.writeable
    before = plane.samples.copy()
    samples[...] = 0.0
    # An exact structural contract: the plane is a copy, untouched by its source.
    assert np.array_equal(plane.samples, before)
    with pytest.raises(ValueError, match="read-only"):
        plane.samples[0, 0, 0] = 1.0
    assert "array(" not in repr(plane)


def test_a_modal_plane_measures_its_energy_with_the_identity_metric():
    plane = modal_plane()
    # Parseval on 512 bins: a sum of squares, rounding of order n EPS.
    assert plane.energy() == pytest.approx(1e-6, rel=1e-12)
    dense = modal_plane(
        metric=PlaneMetric(
            "dense", matrix=np.ones((1, 1, plane.grid.n_omega), dtype=complex)
        )
    )
    assert dense.energy() == pytest.approx(plane.energy(), rel=1e-14)


def test_a_plane_refuses_inconsistent_parts():
    plane = modal_plane()
    with pytest.raises(CouplingError, match="samples have shape"):
        FieldPlane(
            "modal",
            plane.samples[..., :-1],
            plane.grid,
            plane.units,
            plane.basis,
            plane.reference,
            plane.geometry,
            plane.metric,
        )
    with pytest.raises(CouplingError, match="needs a radial basis"):
        FieldPlane(
            "radial",
            plane.samples,
            plane.grid,
            plane.units,
            plane.basis,
            plane.reference,
            plane.geometry,
            plane.metric,
        )
    with pytest.raises(CouplingError, match="not known"):
        modal_plane(reference=PlaneReference(calibrated=("delay",))).absolute()


@pytest.mark.parametrize(
    ("kind", "arguments", "match"),
    [
        ("dense", {}, "carries"),
        ("identity", {"weights": np.ones(2)}, "carries"),
        ("unknown", {}, "says why"),
    ],
)
def test_a_metric_carries_exactly_what_its_kind_names(kind, arguments, match):
    with pytest.raises(CouplingError, match=match):
        PlaneMetric(kind, **arguments)  # pyright: ignore[reportArgumentType] - refused


def test_an_unknown_metric_measures_nothing():
    plane = modal_plane(metric=PlaneMetric("unknown", reason="not at hand"))
    with pytest.raises(CouplingError, match="not at hand"):
        plane.energy()


@pytest.mark.parametrize(
    "space", [RadialGrid(4e-4, 48), FreeGrid(3e-4, 32, 3e-4, 32)], ids=["radial", "xy"]
)
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_a_beam_plane_carries_its_physical_energy_and_node_round_trip(space, kind):
    constructor = FieldGrid if kind == "field" else EnvelopeGrid
    grid = constructor.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    medium = SpatialMedium("Ar", 1.0)
    beam = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    plane = FieldPlane.from_pulse(
        beam, grid, PlaneGeometry(0.0), space=space, medium=medium
    )
    flux = SpatialFlux.resolve(grid, space, medium)
    built = beam.build(grid, space, medium=medium)
    # The same weighted sum of squares as spatial_energy, in another order.
    assert plane.energy() == pytest.approx(spatial_energy(built, flux), rel=1e-12)
    # The requested energy is the physical flux through the plane (SEP 34).
    assert plane.energy() == pytest.approx(3e-9, rel=1e-9)
    assert plane.units == ("V m" if isinstance(space, RadialGrid) else "V/m")
    nodes = plane.node_spectra()
    again = FieldPlane.from_samples(
        nodes, grid, PlaneGeometry(0.0), exterior="zero", space=space, medium=medium
    )
    # A transform and its inverse: the QDHT and FFT are well conditioned, so the round
    # trip is a few hundred roundings of the largest sample.
    scale = float(np.max(np.abs(plane.samples)))
    assert_allclose(again.samples, plane.samples, rtol=0, atol=1e3 * EPS * scale)
    assert again.basis.identity == plane.basis.identity


def test_from_samples_refuses_ambiguous_or_incomplete_declarations():
    grid = envelope_grid()
    space = RadialGrid(4e-4, 16)
    values = np.zeros((16, 1, grid.n_omega), complex)
    basis = PlaneBasis("modal", basis_identity("modal", {"modes": []}))
    geometry = PlaneGeometry(0.0)
    with pytest.raises(CouplingError, match="space or coefficients"):
        FieldPlane.from_samples(values, grid, geometry, exterior="zero")
    with pytest.raises(CouplingError, match="medium"):
        FieldPlane.from_samples(values, grid, geometry, exterior="zero", space=space)
    with pytest.raises(CouplingError, match="16 nodes"):
        FieldPlane.from_samples(
            values[:4],
            grid,
            geometry,
            exterior="zero",
            space=space,
            medium=SpatialMedium(),
        )
    radial = PlaneBasis.of_space(space)
    with pytest.raises(CouplingError, match="modal basis"):
        FieldPlane.from_samples(values, grid, geometry, exterior="zero", basis=radial)
    with pytest.raises(CouplingError, match="from_pulse takes"):
        FieldPlane.from_pulse(
            object(), grid, geometry, space=space, medium=SpatialMedium()
        )
    assert basis.labels == ()


def test_the_record_of_a_grid_is_the_record_a_file_keeps():
    for space in (
        RadialGrid(1e-3, 64),
        FreeGrid(1e-3, 32, 2e-3, 16),
        LineGrid(1e-3, 64),
    ):
        assert space_record(space) == serialise(space)
    with pytest.raises(CouplingError, match="planar"):
        from solprop.modes.slab import PlanarGrid  # noqa: PLC0415

        space_record(object.__new__(PlanarGrid))


def test_a_lattice_is_its_samples_not_its_band_or_length():
    one = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
    longer = EnvelopeGrid.from_limits(5.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
    wider = EnvelopeGrid.from_limits(1.0, 800e-9, (500e-9, 1.2e-6), 1e-12)
    assert lattice_identity(one) == lattice_identity(longer)
    assert lattice_identity(one) != lattice_identity(wider)
    assert lattice_identity(one) != lattice_identity(envelope_grid(2e-12))


def test_a_mode_set_s_basis_is_its_value_at_the_plane():
    grid = envelope_grid()
    short = [MarcatiliMode(75e-6, "He", 1.0)]
    same = [MarcatiliMode(75e-6, "He", 1.0)]
    wider = [MarcatiliMode(80e-6, "He", 1.0)]
    assert modal_basis_identity(short, grid, 0.0) == modal_basis_identity(
        same, grid, 0.3
    )
    assert modal_basis_identity(short, grid, 0.0) != modal_basis_identity(
        wider, grid, 0.0
    )


def test_provenance_appends_its_history():
    provenance = PlaneProvenance("abc").then("one").then("two")
    assert provenance.history == ("one", "two")
    assert provenance.record()["source"] == "abc"


def test_a_basis_identity_is_a_digest():
    with pytest.raises(CouplingError, match="SHA-256"):
        PlaneBasis("modal", "short")
    with pytest.raises(CouplingError, match="transverse grid"):
        PlaneBasis("radial", basis_identity("radial", {}))


def test_a_nodal_field_owns_finite_copies_of_matching_shapes():
    n_omega = 4
    nodes, weights = np.zeros((2, 2)), np.ones(2)
    electric = np.ones((2, 3, n_omega), complex)
    nodal = NodalField(nodes, weights, electric)
    assert nodal.magnetic is None and not nodal.electric.flags.writeable
    with pytest.raises(CouplingError, match="nodes must be finite"):
        NodalField(np.full((2, 2), np.nan), weights, electric)
    # A magnetic field of another shape is refused at the boundary; under the test
    # session's runtime annotation checks it never reaches it, so the helper is
    # asked directly.
    with pytest.raises(CouplingError, match="magnetic has shape"):
        _frozen(electric[..., :-1], np.complex128, "magnetic", electric.shape)


def test_a_metric_of_another_plane_s_shape_measures_nothing():
    plane = modal_plane()
    n_omega = plane.grid.n_omega
    dense = PlaneMetric("dense", matrix=np.ones((2, 2, n_omega), complex))
    with pytest.raises(CouplingError, match="metric shape"):
        dense.spectral_energy(plane.samples, plane.grid)
    admittance = PlaneMetric(
        "admittance",
        admittance=np.ones((2, n_omega), complex),
        weights=np.ones(2),
        scale=1.0,
    )
    with pytest.raises(CouplingError, match="admittance shape"):
        admittance.spectral_energy(plane.samples, plane.grid)


def test_scalar_profiles_describe_the_columns_of_a_modal_plane():
    plane = modal_plane()
    profiles = ScalarModeBasis(
        "capillary",
        ("HE11", "HE12"),
        75e-6,
        np.array([2.405, 5.520]),
        np.zeros(2),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.ones((2, plane.grid.n_omega)),
    )
    # Two profiles for a plane of one column.
    with pytest.raises(CouplingError, match="scalar profiles"):
        modal_plane(profiles=profiles)


def test_a_modal_plane_has_no_node_samples():
    with pytest.raises(CouplingError, match="no node samples"):
        modal_plane().node_spectra()
