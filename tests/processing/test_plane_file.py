# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A field plane's file form: every part comes back, and a loaded plane couples."""

import dataclasses

import h5py
import numpy as np
import pytest

from processing.test_planes import capillary, general_simulation_and_output, spatial
from solprop.coupling import (
    CouplingModel,
    FieldPlane,
    PlaneBasis,
    PlaneGeometry,
    PlaneKind,
    PlaneMetric,
    PlaneReference,
    basis_identity,
    couple,
)
from solprop.exceptions import OutputError, ProcessingError
from solprop.fourier import spectrum_to_optics
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.output.hdf5 import open_file
from solprop.processing import (
    PLANE_FILE_SCHEMA,
    field_plane,
    load_field_plane,
    save_field_plane,
)
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]


def round_trip(plane: FieldPlane, tmp_path) -> FieldPlane:
    path = tmp_path / "plane.h5"
    save_field_plane(plane, path)
    return load_field_plane(path)


def equal(first: object, second: object) -> bool:
    """Compare two parts: arrays exactly, dataclasses field by field.

    The file's contract is lossless, so the comparison is a structural one.
    """
    if first is None or second is None:
        return first is second
    if isinstance(first, np.ndarray):
        return np.array_equal(first, np.asarray(second))
    if dataclasses.is_dataclass(first):
        return all(
            equal(getattr(first, field.name), getattr(second, field.name))
            for field in dataclasses.fields(first)
            if field.name not in ("identity", "_norms")
            or isinstance(getattr(first, field.name), str)
        )
    if isinstance(first, tuple | list):
        return len(first) == len(second) and all(  # type: ignore[arg-type]
            equal(a, b)
            for a, b in zip(first, second, strict=True)  # type: ignore[arg-type]
        )
    return first == second


def assert_same_plane(loaded: FieldPlane, plane: FieldPlane) -> None:
    """Assert every part but the grid object, which is rebuilt, came back."""
    assert np.array_equal(loaded.samples, plane.samples)
    assert np.array_equal(loaded.grid.omega, plane.grid.omega)
    assert np.array_equal(loaded.grid.t, plane.grid.t)
    assert loaded.grid.kind == plane.grid.kind
    assert loaded.grid.omega0 == plane.grid.omega0
    for name in (
        "kind",
        "units",
        "exterior",
        "reference",
        "geometry",
        "provenance",
        "analytic",
    ):
        assert getattr(loaded, name) == getattr(plane, name), name
    assert loaded.basis.identity == plane.basis.identity
    assert loaded.basis.labels == plane.basis.labels
    assert type(loaded.basis.space) is type(plane.basis.space)
    assert equal(loaded.metric, plane.metric)
    assert equal(loaded.nodal, plane.nodal)
    assert equal(loaded.profiles, plane.profiles)
    assert loaded.medium is None


def test_a_modal_plane_comes_back_whole(tmp_path):
    output = capillary().run(progress="none")
    plane = field_plane(output)
    assert_same_plane(round_trip(plane, tmp_path), plane)


def test_a_modal_plane_with_its_profiles_comes_back_whole(tmp_path):
    simulation = capillary()
    plane = field_plane(simulation.run(progress="none"), source=simulation)
    assert plane.profiles is not None
    loaded = round_trip(plane, tmp_path)
    assert_same_plane(loaded, plane)
    assert loaded.profiles is not None
    assert loaded.profiles.identity == plane.profiles.identity


@pytest.mark.parametrize(
    "space",
    [RadialGrid(6e-4, 32), FreeGrid(6e-4, 24, 5e-4, 16), LineGrid(4e-4, 32)],
    ids=["radial", "cartesian", "line"],
)
def test_a_spatial_plane_comes_back_whole(space, tmp_path):
    plane = field_plane(spatial(space).run(progress="none"))
    loaded = round_trip(plane, tmp_path)
    assert_same_plane(loaded, plane)
    # The energy is a function of the stored parts: a structural check, to the bit.
    assert loaded.energy() == plane.energy()


def test_a_plane_read_from_a_file_saves_as_it_read(tmp_path):
    source = tmp_path / "run.h5"
    spatial(RadialGrid(6e-4, 32)).run(source, progress="none")
    plane = field_plane(open_file(source))
    assert_same_plane(round_trip(plane, tmp_path), plane)


def test_a_gradient_plane_keeps_its_certificate_through_two_saves(tmp_path):
    gradient = PressureGradient.two_point(1.0, 1.2, 2e-3)
    plane = field_plane(spatial(RadialGrid(6e-4, 32), gradient).run(progress="none"))
    assert plane.certificate is not None
    first = tmp_path / "first"
    first.mkdir()
    loaded = round_trip(plane, first)
    assert loaded.certificate is not None
    again = round_trip(loaded, tmp_path)
    assert again.certificate is not None
    assert again.certificate.identity == loaded.certificate.identity
    assert again.certificate.field_error == plane.certificate.field_error
    assert again.certificate.z_span == plane.certificate.z_span
    assert np.array_equal(again.certificate.support, plane.certificate.support)
    assert again.certificate.accuracy == plane.certificate.accuracy


def test_a_general_plane_keeps_its_nodal_fields(tmp_path):
    simulation, output = general_simulation_and_output()
    plane = field_plane(output, source=simulation)
    assert plane.nodal is not None
    assert_same_plane(round_trip(plane, tmp_path), plane)


def test_a_dense_metric_comes_back(tmp_path):
    plane = _bare_plane()
    n_omega = plane.grid.n_omega
    dense = PlaneMetric("dense", matrix=np.ones((1, 1, n_omega), dtype=complex))
    with_dense = dataclasses.replace(plane, metric=dense)
    assert_same_plane(round_trip(with_dense, tmp_path), with_dense)


def test_a_loaded_plane_launches_as_the_original_does(tmp_path):
    simulation = capillary()
    plane = field_plane(simulation.run(progress="none"))
    port = simulation.input_port()
    model = CouplingModel("same_basis")
    direct = couple(plane, port, model=model)
    loaded = couple(round_trip(plane, tmp_path), port, model=model)
    # Same bits in, same arithmetic: the identity model copies the coefficients.
    assert np.array_equal(direct.launch.samples, loaded.launch.samples)
    assert loaded.launch.origin == direct.launch.origin
    assert loaded.launch.provenance == direct.launch.provenance


def test_an_analytic_record_comes_back(tmp_path):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (700e-9, 900e-9), 300e-15)

    def beam(r, omega):
        spectral = np.exp(-(((omega - grid.omega0) * 20e-15) ** 2))
        return np.exp(-((r / 50e-6) ** 2))[:, None] * spectral[None, :]

    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        beam,
        space=RadialGrid(300e-6, 128),
        index=np.where(grid.omega > 0, 1.0, 0.0),
    )
    assert plane.analytic is not None
    assert_same_plane(round_trip(plane, tmp_path), plane)


def test_a_live_medium_is_refused_by_name(tmp_path):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 200e-15)
    plane = FieldPlane.from_pulse(
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 1e-4),
        grid,
        PlaneGeometry(0.0),
        space=RadialGrid(6e-4, 32),
        medium=SpatialMedium(),
    )
    path = tmp_path / "plane.h5"
    with pytest.raises(ProcessingError, match="live medium"):
        save_field_plane(plane, path)
    assert not path.exists()
    with_flux_only = dataclasses.replace(plane, medium=None)
    with pytest.raises(ProcessingError, match="live medium"):
        save_field_plane(with_flux_only, path)


def _bare_plane(kind: PlaneKind = "modal") -> FieldPlane:
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 200e-15)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9)
    return FieldPlane(
        kind,
        spectrum_to_optics(pulse.build(grid), kind=grid.kind),
        grid,
        "sqrt(W)",
        PlaneBasis(kind, basis_identity(kind, {"modes": ["one"]})),
        PlaneReference(0.0, 0.0),
        PlaneGeometry(0.0),
        PlaneMetric("identity"),
    )


def test_a_planar_plane_is_refused_by_name(tmp_path):
    with pytest.raises(ProcessingError, match="not exchanged"):
        save_field_plane(_bare_plane("planar"), tmp_path / "plane.h5")


def test_an_existing_file_is_not_replaced(tmp_path):
    path = tmp_path / "plane.h5"
    save_field_plane(_bare_plane(), path)
    before = path.read_bytes()
    with pytest.raises(OutputError, match="cannot create"):
        save_field_plane(_bare_plane(), path)
    assert path.read_bytes() == before


def test_a_file_that_is_not_a_plane_is_refused(tmp_path):
    missing = tmp_path / "missing.h5"
    with pytest.raises(OutputError, match="cannot read"):
        load_field_plane(missing)
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as file:
        file.create_dataset("samples", data=np.zeros(3))
    with pytest.raises(OutputError, match="not a field-plane file"):
        load_field_plane(other)
    text = tmp_path / "text.h5"
    text.write_text("not hdf5", encoding="utf-8")
    with pytest.raises(OutputError, match="cannot read"):
        load_field_plane(text)


def test_another_schema_is_refused_by_name(tmp_path):
    path = tmp_path / "plane.h5"
    save_field_plane(_bare_plane(), path)
    with h5py.File(path, "a") as file:
        file.attrs["field_plane_schema"] = PLANE_FILE_SCHEMA + 1
    with pytest.raises(OutputError, match="reads schema 1 only"):
        load_field_plane(path)


@pytest.mark.parametrize("part", ["samples", "grid", "metric"])
def test_a_damaged_file_is_refused(tmp_path, part):
    path = tmp_path / "plane.h5"
    save_field_plane(_bare_plane(), path)
    with h5py.File(path, "a") as file:
        del file[part]
    with pytest.raises(OutputError, match="cannot read"):
        load_field_plane(path)


def test_a_path_may_be_text(tmp_path):
    path = str(tmp_path / "plane.h5")
    save_field_plane(_bare_plane(), path)
    assert load_field_plane(path).kind == "modal"
