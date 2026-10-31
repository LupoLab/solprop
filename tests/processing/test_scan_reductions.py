# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check reduction contracts and the spatial statistics adapters."""

import dataclasses
from typing import cast

import h5py
import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import ProcessingError
from solprop.grid import FieldGrid
from solprop.output.memory import MemoryOutput
from solprop.output.scan import PointResult, ScanWriter
from solprop.output.schema import (
    GridRecord,
    SavedPlane,
    _example_metadata,
)
from solprop.processing import scans
from solprop.processing.axes import grid_from_record
from solprop.processing.fields import get_spectral_density
from solprop.processing.scans import (
    beam_radius,
    energy_in_band,
    energy_in_window,
    on_axis_intensity,
    spectrum_in_band,
)
from solprop.pulses import SechPulse


def test_spatial_reductions_retain_existing_statistics():
    metadata = _example_metadata()
    with MemoryOutput() as output:
        output.open(metadata)
        # The reductions consume recorded statistics; their spatial definitions
        # have independent analytic tests under test_stats and the spatial suites.
        output.save(
            SavedPlane(
                z=0.0,
                field=np.zeros(
                    (1, 1, grid_from_record(metadata.grid).n_omega), complex
                ),
                stats={
                    "on_axis_intensity": np.asarray(2e14),
                    "beam_radius": np.asarray([2e-5, 2.1e-5]),
                },
            )
        )
        for reducer in (on_axis_intensity(), beam_radius()):
            value = reducer(output)
            assert value.shape == reducer.shape(metadata)
            np.testing.assert_array_equal(value, output[f"stats/{reducer.name}"][-1])


@pytest.mark.parametrize(
    "factory, bounds",
    [
        (energy_in_band, (0.0, 1e-6)),
        (energy_in_band, (1e-6, 1e-6)),
        (energy_in_window, (1.0, -1.0)),
    ],
)
def test_invalid_reduction_bounds(factory, bounds):
    with pytest.raises(ProcessingError):
        factory(*bounds)


@pytest.mark.parametrize(
    "initial",
    [
        None,
        np.ones((1, 1, 8), complex),
        SechPulse(wavelength=800e-9, width=30e-15, peak_power=1.0),
    ],
)
def test_field_reduction_shape_uses_declared_initial_or_requires_context(initial):
    metadata = dataclasses.replace(
        _example_metadata(), configuration={"initial": initial}
    )
    reduction = scans.field_at_plane()
    if initial is None:
        with pytest.raises(ProcessingError, match="input field shape"):
            reduction.shape(metadata)
    else:
        expected = (
            initial
            if isinstance(initial, np.ndarray)
            else initial.build(grid_from_record(metadata.grid))
        )
        assert reduction.shape(metadata) == expected.shape


@pytest.mark.parametrize("name,shape", [("", ()), ("bad/name", ()), ("valid", (-1,))])
def test_custom_reduction_refuses_invalid_contract(name, shape):
    with pytest.raises(ProcessingError, match="valid name and shape"):
        scans.custom(name, lambda output: np.asarray(0.0), shape, "J")


def test_builtin_reducer_requires_metadata_and_known_operation():
    with pytest.raises(ProcessingError, match="output metadata"):
        energy_in_band(200e-9, 400e-9)(MemoryOutput())
    invalid = dataclasses.replace(
        cast(scans._Builtin, scans.peak_power()), operation="unknown"
    )
    with pytest.raises(ProcessingError, match="unknown reduction"):
        invalid(MemoryOutput())


def test_custom_reducer_without_inspectable_source_records_absence(monkeypatch):
    def unavailable(function):
        raise OSError("source file no longer available")

    monkeypatch.setattr(scans.inspect, "getsource", unavailable)
    provenance = scans._provenance(
        scans.custom("external", lambda output: np.asarray(0.0), (), "J")
    )
    assert provenance["source_hash"] is None
    assert not provenance["source_available"]


def test_scan_reader_requires_supported_schema_and_known_axes(tmp_path):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as file:
        file.attrs["scan_schema_version"] = 99
    with pytest.raises(ProcessingError, match="unsupported scan"):
        scans.ScanOutput(path)
    other = tmp_path / "valid.h5"
    ScanWriter(other, Scan("valid", x=[1]), {}).close()
    with pytest.raises(ProcessingError, match="unknown scan variable"):
        scans.ScanOutput(other).where(missing=1)


def build_field_point(point):
    # A field grid reaching down to 110 nm, so its stored axis holds the zero and
    # Nyquist bins where a wavelength is undefined; a 0.1 mm capillary keeps the run
    # to a few steps.
    return Simulation.from_capillary(
        125e-6,
        1e-4,
        "He",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(110e-9, 4e-6),
        trange=200e-15,
        fwhm=10e-15,
        energy=float(point.values["energy"]),
        modes=1,
        raman=False,
        plasma=False,
        n_save=2,
        backend="numpy_ref",
        noise=None,
    )


@pytest.mark.medium
def test_spectrum_in_band_on_a_field_run_matches_its_shape_and_abscissa(tmp_path):
    scan = Scan("field", energy=[1e-6, 2e-6])
    reduction = spectrum_in_band(200e-9, 1.5e-6)
    result = run_scan(
        scan,
        build_field_point,
        output=tmp_path / "scan.h5",
        reductions=[reduction],
        progress="none",
    )
    assert result.status.tolist() == ["done", "done"]
    singleton = build_field_point(scan.point(0)).run(progress="none")
    assert singleton.metadata is not None
    assert singleton.metadata.grid.kind == "field"
    (n_x,) = reduction.shape(singleton.metadata)
    assert n_x > 0
    assert result.as_grid("reduced/spectrum_in_band").shape == (2, n_x)
    wavelength = result.spectrum_wavelength()
    np.testing.assert_array_equal(wavelength, result.spectrum_wavelength(index=1))
    assert wavelength.shape == (n_x,)
    assert np.all(np.diff(wavelength) > 0)
    assert wavelength[0] >= 200e-9 and wavelength[-1] <= 1.5e-6
    x, density = get_spectral_density(
        singleton, plane=-1, modes="sum", spec_range=(200e-9, 1.5e-6)
    )
    np.testing.assert_array_equal(wavelength, x)
    np.testing.assert_array_equal(result.point(0)["reduced/spectrum_in_band"], density)


def _spectrum_point(index, grid, reduction):
    metadata = dataclasses.replace(_example_metadata(), grid=GridRecord.from_grid(grid))
    (n_x,) = reduction.shape(metadata)
    return PointResult(
        index,
        "done",
        configuration_hash=metadata.configuration_hash(),
        metadata_json=metadata.to_json(),
        arrays={"z": np.zeros(1), f"reduced/{reduction.name}": np.zeros(n_x)},
        reductions={reduction.name: scans._provenance(reduction, metadata=metadata)},
    )


def test_spectrum_wavelength_refuses_ambiguous_or_missing_axes(tmp_path):
    path = tmp_path / "scan.h5"
    reduction = spectrum_in_band(400e-9, 1.2e-6)
    impostor = scans.custom("spectrum_in_band", lambda output: np.zeros(3), (3,), "J")
    grids = [
        FieldGrid.from_limits(1.0, 800e-9, (300e-9, 2e-6), 200e-15),
        FieldGrid.from_limits(1.0, 800e-9, (300e-9, 2e-6), 400e-15),
    ]
    writer = ScanWriter(path, Scan("spectra", x=[0, 1, 2, 3]), {})
    for index, grid in enumerate(grids):
        writer.commit(_spectrum_point(index, grid, reduction))
    writer.commit(_spectrum_point(2, grids[0], impostor))
    writer.close()
    result = scans.open(path)
    for index, grid in enumerate(grids):
        expected = scans._band_wavelength(GridRecord.from_grid(grid), (400e-9, 1.2e-6))
        np.testing.assert_array_equal(result.spectrum_wavelength(index), expected)
    with pytest.raises(ProcessingError, match="different wavelength axes"):
        result.spectrum_wavelength()
    with pytest.raises(ProcessingError, match="out of range"):
        result.spectrum_wavelength(4)
    for index in (2, 3):
        with pytest.raises(ProcessingError, match=f"at point {index}"):
            result.spectrum_wavelength(index)
