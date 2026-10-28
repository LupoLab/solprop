# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate worker inputs, retention contracts and reducer results before commit."""

import dataclasses
import inspect
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from driver.test_scan_batch import build
from solprop.driver import Simulation, _scan_run
from solprop.driver._scan_run import _Capture, _checked, _Job, _Options, _Work
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import DriverError, OutputError
from solprop.output.schema import SavedPlane
from solprop.processing import scans


@pytest.fixture
def point():
    return Scan("worker", base_seed=42, power=[1.0]).point(0)


def test_worker_probe_records_backend_and_retention_memory(point):
    work = _Work(build, _Options((), "all", (), "all"), "worker")
    report = work.probe(_Job(point))
    assert report["baseline_rss"] > 0
    assert report["rss"] >= report["baseline_rss"]
    assert report["ledger"] > 0
    assert report["output_bytes"] > 0


@pytest.mark.parametrize("selection", [(99,), (-99,)])
def test_scan_refuses_field_plane_outside_saved_range(tmp_path, point, selection):
    with pytest.raises(DriverError, match="outside"):
        _Capture(_Options((), "planes", selection, "none"), build(point), tmp_path)


def test_capture_requires_metadata_before_saving(tmp_path, point):
    simulation = build(point)
    capture = _Capture(_Options((), "none", (), "none"), simulation, tmp_path)
    with pytest.raises(OutputError, match="opened before saving"):
        capture.save(SavedPlane(z=0.0, field=simulation.translate().initial.field0[0]))


@pytest.mark.parametrize(
    "value", [np.asarray(1, dtype=np.int64), np.ones(2), np.asarray(np.nan)]
)
def test_custom_reduction_checks_dtype_shape_and_finiteness(point, value):
    reduction = scans.custom("invalid", lambda output: np.asarray(0.0), (), "J")
    with pytest.raises(OutputError, match="returned"):
        _checked(reduction, value, build(point).metadata())


def test_worker_rejects_missing_requested_statistics(point):
    work = _Work(build, _Options((), "none", (), ("beam_radius",)), "worker")
    with pytest.raises(DriverError, match="not collected"):
        work.simulation(_Job(point))


def test_worker_rejects_spatial_reducer_without_collected_statistic(point):
    work = _Work(build, _Options((scans.beam_radius(),), "none", (), "none"), "worker")
    with pytest.raises(DriverError, match="requires its simulation statistic"):
        work.simulation(_Job(point))


def test_worker_rejects_noise_seed_mismatch(point):
    original = build(point)
    assert original.noise is not None
    wrong = original.with_input(
        original.translate().initial.field0[0],
        noise=dataclasses.replace(original.noise, seed=43),
    )
    work = _Work(lambda point: wrong, _Options((), "none", (), "none"), "worker")
    with pytest.raises(DriverError, match=r"matching point\.seed"):
        work.simulation(_Job(point))


@pytest.mark.parametrize("names", [("duplicate", "duplicate"), ("bad/name",)])
def test_scan_rejects_duplicate_or_invalid_reduction_names(tmp_path, names):
    reductions = [
        dataclasses.replace(cast(scans._Builtin, scans.peak_power()), name=name)
        for name in names
    ]
    path = tmp_path / "scan.h5"
    with pytest.raises(DriverError, match="reduction names"):
        run_scan(
            Scan("validation", power=[1]), build, output=path, reductions=reductions
        )
    assert not path.exists()


@pytest.mark.parametrize(
    "invalid", ["shape", "nonfinite", "noise_shape", "noise_policy"]
)
def test_input_update_refuses_incompatible_replacement(point, invalid):
    simulation = build(point)
    field = simulation.translate().initial.field0[0].copy()
    noise = simulation.noise
    if invalid == "shape":
        field = np.repeat(field, 2, axis=0)
    elif invalid == "nonfinite":
        field[0, 0, 0] = np.nan
    elif invalid == "noise_shape":
        assert noise is not None
        noise = dataclasses.replace(
            noise, spectrum=np.repeat(noise.spectrum, 2, axis=0)
        )
    else:
        # The session hook checks Literal values first; this also exercises the
        # production boundary when the optional runtime hook is absent.
        with pytest.raises(DriverError, match="noise must be"):
            inspect.unwrap(Simulation.with_input)(simulation, field, noise="invalid")
        return
    with pytest.raises(DriverError, match=r"shape|finite"):
        simulation.with_input(field, noise=noise)


def test_worker_refuses_builder_returning_wrong_type(point):
    work = _Work(Mock(return_value=None), _Options((), "none", (), "none"), "worker")
    with pytest.raises(DriverError, match="return a Simulation"):
        work.simulation(_Job(point))


def test_capture_refuses_custom_spool_without_metadata(point, tmp_path, monkeypatch):
    custom = scans.custom("constant", lambda output: np.asarray(0.0), (), "J")
    capture = _Capture(_Options((custom,), "none", (), "none"), build(point), tmp_path)
    monkeypatch.setattr(_scan_run, "open_file", lambda path: _scan_run.MemoryOutput())
    with pytest.raises(OutputError, match="has no metadata"):
        capture.results()


@pytest.mark.parametrize("progress", ["bar", "auto"])
def test_scan_outer_progress_handles_global_subsets(tmp_path, monkeypatch, progress):
    monkeypatch.setattr(_scan_run, "is_interactive", lambda: True)
    scan = Scan("progress", base_seed=42, power=[1.0, 2.0, 3.0])
    assert len(scan) == 3
    selected = scan.subset([2])
    assert len(selected) == 1
    result = run_scan(selected, build, output=tmp_path / "scan.h5", progress=progress)
    assert result.status.tolist() == ["todo", "todo", "done"]


def test_scan_mixed_reducers_and_filtered_statistics(point, tmp_path):
    options = _Options(
        (
            scans.peak_power(),
            scans.custom("constant", lambda output: np.asarray(1.0), (), "1"),
        ),
        "none",
        (),
        ("peak_power",),
    )
    work = _Work(build, options, "worker")
    result = work(_Job(point))
    assert result.status == "done"
    assert "stats/peak_power" in result.arrays
    assert "step_stats/energy" not in result.arrays
    assert "reduced/constant" in result.arrays
    assert work.probe(_Job(point))["output_bytes"] > 0


def test_worker_refuses_cached_configuration_mismatch(point):
    result = _Work(build, _Options((), "none", (), "none"), "worker")(
        _Job(point, "0" * 64)
    )
    assert result.status == "conflict"


def test_uninstrumented_retention_validation_rejects_unknown_fields():
    with pytest.raises(DriverError, match="fields must be"):
        _scan_run._validate_options("unknown", (), "none", 10.0, ())


def test_probe_does_not_charge_noise_or_periodic_retention_when_disabled(point):
    base = build(point)
    simulation = base.with_input(base.translate().initial.field0[0], noise=None)
    work = _Work(lambda point: simulation, _Options((), "all", (), "none"), "worker")
    assert work.probe(_Job(point))["output_bytes"] > 0


@pytest.mark.parametrize("fails", [False, True])
def test_worker_incomplete_output_never_publishes_done(
    point, tmp_path, monkeypatch, fails
):
    simulation = build(point)

    def incomplete(*args, **kwargs):
        if fails:
            raise RuntimeError("output failed before open")

    monkeypatch.setattr(simulation, "run", incomplete)
    destination = tmp_path / "scan.h5"
    scan = Scan("incomplete", base_seed=42, power=[1.0])
    if fails:
        output = run_scan(
            scan, lambda point: simulation, output=destination, progress="none"
        )
        assert output.status.tolist() == ["failed"]
    else:
        with pytest.raises(OutputError, match="saved z axis"):
            run_scan(
                scan, lambda point: simulation, output=destination, progress="none"
            )
        assert scans.ScanOutput(destination).status.tolist() == ["running"]


def test_execution_provenance_handles_uninspectable_builder(tmp_path, monkeypatch):
    def missing(*args):
        raise TypeError("extension callable has no source file")

    monkeypatch.setattr(_scan_run.inspect, "getsourcefile", missing)
    destination = tmp_path / "scan.h5"
    writer = _scan_run.ScanWriter(destination, Scan("source", power=[1.0]), {})
    _scan_run._finish_execution(_scan_run.Local(), build, writer, destination)
    assert not writer.file.id.valid
    # A preparation failure can leave no writer to finalize.
    absent = tmp_path / "absent.h5"
    _scan_run._finish_execution(_scan_run.Local(), build, None, absent)
    assert not absent.exists()
