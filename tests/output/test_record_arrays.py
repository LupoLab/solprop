# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A record's large arrays are hashed by their bytes and stored beside its text (#309).

Digests and stored bytes are identity and lossless storage, so every comparison here is
exact (class P). A changed array must still change the hash, and refuse a resume by
name.
"""

import dataclasses
import json
from collections.abc import Mapping

import h5py
import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import OutputError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.output import HDF5Output, check_cache, open_file
from solprop.output.hdf5 import _read_record, _write_record
from solprop.output.scan import PointResult, ScanWriter, read_point
from solprop.output.schema import RECORD_ARRAY_LIMIT, RunMetadata, _example_metadata
from solprop.processing.planes import field_plane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.translate.flux import flux_from_record

# A small window: its edges are not what is tested here.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LARGE = RECORD_ARRAY_LIMIT + 1


def holding(values: object) -> RunMetadata:
    """Return a record whose configuration holds `values`."""
    return dataclasses.replace(_example_metadata(), configuration={"rate": values})


def large(dtype: type = float) -> np.ndarray:
    """Return an array one element over the limit, of distinct values."""
    return np.arange(LARGE).astype(dtype)


def test_an_array_goes_beside_the_text_only_above_the_limit():
    small = holding(np.ones(RECORD_ARRAY_LIMIT))
    assert small.record_arrays() == {}
    assert "__ndarray__" in json.loads(small.to_json())["configuration"]["rate"]
    values = large()
    record = holding(values)
    held = record.record_arrays()
    stored = json.loads(record.to_json())["configuration"]["rate"]
    assert stored["shape"] == [LARGE] and stored["dtype"] == "float64"
    assert held[stored["__record_array__"]] is values


def test_an_array_of_objects_stays_in_the_text():
    record = holding(np.array(["a"] * LARGE))
    assert record.record_arrays() == {}


@pytest.mark.parametrize(
    "changed",
    [
        lambda values: np.where(np.arange(LARGE) == 7, np.nextafter(7.0, 8.0), values),
        lambda values: values.astype(np.float32),
        lambda values: values.reshape(1, LARGE),
    ],
    ids=["one-element", "dtype", "shape"],
)
def test_a_changed_array_changes_the_hash_and_is_named_once(changed):
    values = large()
    one, two = holding(values), holding(changed(values))
    assert one.configuration_hash() != two.configuration_hash()
    assert one.differences(two) == ("configuration.rate",)
    assert one.summarise_differences(two) == "these fields changed: configuration.rate"
    assert holding(values.copy()).configuration_hash() == one.configuration_hash()


def test_a_signed_zero_changes_the_hash():
    values = np.zeros(LARGE)
    negative = values.copy()
    negative[3] = -0.0
    first, second = holding(values), holding(negative)
    assert first.configuration_hash() != second.configuration_hash()


def test_the_record_digests_the_same_in_either_byte_order():
    values = large()
    swapped = values.astype(values.dtype.newbyteorder(">"))
    assert holding(swapped).configuration_hash() == holding(values).configuration_hash()


@pytest.mark.parametrize(
    "values",
    [large(), large(complex) * (1 + 2j), large() % 2 == 0, large(int)],
    ids=["real", "complex", "boolean", "integer"],
)
def test_a_record_round_trips_through_its_text_and_arrays(values):
    record = holding(values)
    back = RunMetadata.from_json(record.to_json(), record.record_arrays())
    assert back.configuration_hash() == record.configuration_hash()
    read = np.asarray(back.configuration["rate"])
    np.testing.assert_array_equal(read, values)
    assert read.dtype == values.dtype
    # And on again, from what was read.
    again = RunMetadata.from_json(back.to_json(), back.record_arrays())
    assert again.configuration_hash() == record.configuration_hash()


def test_an_array_named_twice_is_read_and_stored_once():
    values = large()
    record = dataclasses.replace(
        _example_metadata(), configuration={"one": values, "two": values.copy()}
    )
    assert len(record.record_arrays()) == 1
    back = RunMetadata.from_json(record.to_json(), record.record_arrays())
    assert back.configuration["one"] is back.configuration["two"]
    assert back.configuration_hash() == record.configuration_hash()


SAME_BYTES = {
    "one": np.zeros(2 * LARGE),
    "two": np.zeros((2, LARGE)),
    "three": np.zeros(2 * LARGE, dtype=np.int64),
}
"""Arrays of equal bytes and different layouts, which a digest cannot tell apart."""


def check_same_bytes(configuration: Mapping[str, object]) -> None:
    """Check that each array came back with its own dtype and shape."""
    for name, values in SAME_BYTES.items():
        back = configuration[name]
        assert isinstance(back, np.ndarray)
        assert back.dtype == values.dtype and back.shape == values.shape
        assert np.array_equal(back, values)


def test_arrays_of_equal_bytes_and_different_layouts_stay_apart():
    record = dataclasses.replace(_example_metadata(), configuration=SAME_BYTES)
    assert len(record.record_arrays()) == len(SAME_BYTES)
    back = RunMetadata.from_json(record.to_json(), record.record_arrays())
    check_same_bytes(back.configuration)
    assert back.configuration_hash() == record.configuration_hash()


def test_a_file_keeps_arrays_of_equal_bytes_apart(tmp_path):
    record = dataclasses.replace(_example_metadata(), configuration=SAME_BYTES)
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as file:
        _write_record(file, record)
    with h5py.File(path) as file:
        back = _read_record(file)
    check_same_bytes(back.configuration)
    assert back.configuration_hash() == record.configuration_hash()


def test_a_scan_keeps_arrays_of_equal_bytes_apart(tmp_path):
    from solprop.driver.scan import Scan  # noqa: PLC0415 - the layer above

    path = tmp_path / "scan.h5"
    record = dataclasses.replace(_example_metadata(), configuration=SAME_BYTES)
    writer = ScanWriter(path, Scan("scan", x=[1]), {})
    writer.commit(_point(0, record))
    writer.close()
    with h5py.File(path) as file:
        arrays = file["record_arrays"]
        assert isinstance(arrays, h5py.Group)
        held = {str(name): np.asarray(arrays[name]) for name in arrays}
    back = RunMetadata.from_json(record.to_json(), held)
    check_same_bytes(back.configuration)


def test_an_array_that_did_not_change_is_not_named():
    values = large()
    one = dataclasses.replace(
        _example_metadata(), configuration={"rate": values, "gamma": 0.1}
    )
    two = dataclasses.replace(
        _example_metadata(), configuration={"rate": values.copy(), "gamma": 0.2}
    )
    assert one.differences(two) == ("configuration.gamma",)


def test_a_record_read_without_its_arrays_is_refused_by_name():
    record = holding(large())
    with pytest.raises(OutputError, match="not beside it"):
        RunMetadata.from_json(record.to_json())


def _written(path, record: RunMetadata):
    """Write `record` into a new file at `path`."""
    with HDF5Output(path) as handler:
        handler.open(record)


def test_a_file_holds_the_arrays_beside_the_record(tmp_path):
    path = tmp_path / "run.h5"
    values = large()
    record = holding(values)
    _written(path, record)
    digest = next(iter(record.record_arrays()))
    with h5py.File(path) as file:
        document = json.loads(file["meta/json"][()])  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        assert document["configuration"]["rate"]["__record_array__"] == digest
        held = file[f"meta/arrays/{digest}"]
        assert isinstance(held, h5py.Dataset)
        np.testing.assert_array_equal(held[()], values)
    with open_file(path) as stored:
        assert stored.metadata is not None
        np.testing.assert_array_equal(stored.metadata.configuration["rate"], values)
        assert stored.metadata.configuration_hash() == record.configuration_hash()
    assert check_cache(path, record)


def test_a_changed_array_in_a_file_refuses_a_resume_by_name(tmp_path):
    path = tmp_path / "run.h5"
    _written(path, holding(large()))
    other = large()
    other[9] += 1.0
    with pytest.raises(OutputError, match=r"these fields changed: configuration\.rate"):
        check_cache(path, holding(other))


def test_a_file_whose_array_was_altered_is_refused_on_reading(tmp_path):
    path = tmp_path / "run.h5"
    record = holding(large())
    _written(path, record)
    digest = next(iter(record.record_arrays()))
    with h5py.File(path, "r+") as file:
        held = file[f"meta/arrays/{digest}"]
        assert isinstance(held, h5py.Dataset)
        held[0] = 99.0
    with pytest.raises(OutputError, match="does not match that name"):
        open_file(path)


def test_writing_a_record_again_leaves_only_its_arrays(tmp_path):
    path = tmp_path / "run.h5"
    second = large()
    second[0] = 5.0
    with HDF5Output(path) as handler:
        handler.open(holding(large()))
        handler.open(holding(second), force=True)
    with h5py.File(path) as file:
        arrays = file["meta/arrays"]
        assert isinstance(arrays, h5py.Group)
        assert len(arrays) == 1


def _free(n: int = 72) -> Simulation:
    """Return a small Cartesian run whose per-node arrays are over the limit."""
    assert n * n > RECORD_ARRAY_LIMIT
    grid = EnvelopeGrid.from_limits(1e-5, 800e-9, (700e-9, 900e-9), 100e-15, thg=False)
    return Simulation.from_free(
        grid,
        FreeGrid(1e-3, n, 1e-3, n),
        SpatialMedium(),
        [KerrResponse(1e-22, thg=False)],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 1e-4),
        noise=None,
        settings=SimulationSettings(
            save_planes=[1e-5], dz_max=1e-5, dz_init=1e-5, backend="numpy_ref"
        ),
    )


def test_a_spatial_record_is_small_and_its_flux_plan_reads_back(tmp_path):
    simulation = _free()
    record = simulation.metadata()
    assert len(record.to_json()) < 100_000
    path = tmp_path / "run.h5"
    _written(path, record)
    with open_file(path) as stored:
        assert stored.metadata is not None
        spatial = stored.metadata.spatial
        assert spatial is not None
        read = flux_from_record(spatial["flux_plan"])  # pyright: ignore[reportArgumentType]
    assert record.spatial is not None
    live = flux_from_record(record.spatial["flux_plan"])  # pyright: ignore[reportArgumentType]
    np.testing.assert_array_equal(read.base.k_squared, live.base.k_squared)
    np.testing.assert_array_equal(read.base.active, live.base.active)
    for name in ("spatial_weights", "spectral_weights", "window", "absorber_rate"):
        np.testing.assert_array_equal(spatial[name], record.spatial[name])


@pytest.mark.medium
def test_a_plane_exports_from_a_file_through_the_stored_flux_plan(tmp_path):
    path = tmp_path / "run.h5"
    _free().run(path, progress="none")
    with open_file(path) as stored:
        plane = field_plane(stored)
    assert plane.samples.size > 0
    assert plane.metric.admittance is not None


def test_a_noise_realisation_travels_beside_the_record(tmp_path):
    grid = EnvelopeGrid.from_limits(1e-5, 800e-9, (700e-9, 900e-9), 100e-15, thg=False)
    simulation = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1e-26],
        wavelength=800e-9,
        grid=grid,
        width=30e-15,
        peak_power=1.0,
        shape="sech",
        n_save=2,
        backend="numpy_ref",
        noise="input",
        seed=3,
    )
    record = simulation.metadata()
    assert record.noise is not None
    spectrum = np.ones((1, 1, LARGE), complex) * 0.5j
    big = dataclasses.replace(
        record, noise=dataclasses.replace(record.noise, spectrum=spectrum)
    )
    path = tmp_path / "run.h5"
    _written(path, big)
    with open_file(path) as stored:
        assert stored.metadata is not None and stored.metadata.noise is not None
        np.testing.assert_array_equal(stored.metadata.noise.spectrum, spectrum)
        assert stored.metadata.noise.seed == 3
        assert stored.metadata.configuration_hash() == big.configuration_hash()


def _point(index: int, record: RunMetadata) -> PointResult:
    """Return a done scan point holding `record`."""
    return PointResult(
        index,
        "done",
        record.configuration_hash(),
        record.to_json(),
        {
            "z": np.arange(3.0),
            "stats/energy": np.ones((3, 2)),
            "step_stats/z": np.arange(2.0),
            "step_stats/step": np.arange(2),
        },
        metadata_arrays=record.record_arrays(),
    )


def test_a_scan_stores_a_shared_array_once(tmp_path):
    from solprop.driver.scan import Scan  # noqa: PLC0415 - the layer above

    path = tmp_path / "scan.h5"
    scan = Scan("scan", x=[1, 2])
    values = large()
    record = holding(values)
    writer = ScanWriter(path, scan, {})
    writer.commit(_point(0, record))
    writer.commit(_point(1, record))
    writer.close()
    digest = next(iter(record.record_arrays()))
    with h5py.File(path) as file:
        arrays = file["record_arrays"]
        assert isinstance(arrays, h5py.Group)
        assert list(arrays) == [digest]
    point = read_point(path, 1)
    assert isinstance(point["metadata"], dict)
    # The point's document stays JSON, and names the array the file holds.
    assert point["metadata"]["configuration"]["rate"]["__record_array__"] == digest  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]


def test_a_point_whose_record_does_not_match_its_arrays_is_refused(tmp_path):
    from solprop.driver.scan import Scan  # noqa: PLC0415 - the layer above

    scan = Scan("scan", x=[1])
    record = holding(large())
    writer = ScanWriter(tmp_path / "scan.h5", scan, {})
    missing = dataclasses.replace(_point(0, record), metadata_arrays={})
    with pytest.raises(OutputError, match="not beside it"):
        writer.commit(missing)
    writer.close()
