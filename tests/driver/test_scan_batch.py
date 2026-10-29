# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise public batch dispatch with ordinary scan output and member provenance."""

import json
from collections.abc import Mapping
from functools import partial
from typing import cast

import h5py
import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER, assert_points_matched
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.backends.registry import resolve_name
from solprop.driver import Simulation, _scan_batch
from solprop.driver.executors import Batch, Local
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import DriverError, OutputError
from solprop.grid import EnvelopeGrid
from solprop.output.hdf5 import _dataset
from solprop.processing import scans
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


def build(point, backend="numpy_ref"):
    # The former grid without a collar: a short fixed schedule is refused beside the
    # default collar, and these tests are about batched scans, not boundaries.
    grid = EnvelopeGrid.from_limits(0.001, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID)
    return Simulation.from_gnlse(
        0.11 * (point.index + 1),
        0.001,
        [0.0, 0.0, -1.183e-26 * (point.index + 1)],
        lambda0=835e-9,
        grid=grid,
        absorber=bare_absorber(grid),
        width=100e-15,
        peak_power=float(point.index + 1),
        pulseshape="sech",
        save_planes=[0.0005, 0.001],
        schedule=FixedSchedule.uniform(
            0.0, 0.001, 0.00025, save_planes=[0.0005, 0.001]
        ),
        stats=StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2),
        noise="modified",
        seed=point.seed,
        # The fixed order leaves no controller decision on the prescribed schedule, so
        # a batch member and its solo run are one discrete march (SEP 25 4.4).
        integrator=FIXED_ORDER,
        backend=backend,
    )


# A native lockstep batch replays its Adams members on solprop's march, with the
# warning native/tests/test_batch.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native", "auto"])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_batch_scan_matches_local_with_partial_groups(tmp_path, backend, size):
    if backend == "native":
        pytest.importorskip("solprop_native")
    scan = Scan("batch", base_seed=772, power=[1.0, 2.0, 3.0])
    builder = partial(build, backend=backend)
    executor = Batch(size)
    options = {
        "fields": "planes",
        "field_planes": [-1],
        "progress": "none",
        "reductions": [scans.peak_power()],
    }
    reference = run_scan(
        scan, builder, output=tmp_path / "local.h5", executor=Local(), **options
    )
    result = run_scan(
        scan, builder, output=tmp_path / "batch.h5", executor=executor, **options
    )
    assert result.status.tolist() == ["done"] * 3
    # Not bitwise: a batch's lockstep reductions and thread layout may order sums
    # differently from the member run alone.
    assert_points_matched(result, reference, scan, builder, what="batch member")
    assert executor.budgets
    for budget in executor.budgets:
        admitted, peak, limit = (
            budget["admitted"],
            budget["estimated_peak"],
            budget["limit"],
        )
        assert admitted is not None and admitted >= 1
        assert peak is not None and limit is not None
        assert peak <= limit * 4 // 5
    with h5py.File(tmp_path / "batch.h5", "r") as file:
        for index in range(3):
            recorded = json.loads(_dataset(file, "points/backend")[index])
            assert recorded["name"] == resolve_name(backend)
            assert recorded["settings"]["batch_size"] <= size
    run_scan(scan, builder, output=tmp_path / "batch.h5", executor=executor, **options)
    assert executor.budgets == []


@pytest.mark.parametrize("size", [0, -1, True])
def test_batch_rejects_invalid_size(size):
    with pytest.raises(DriverError, match="positive integer"):
        Batch(size)


def test_batch_rejects_non_scan_work():
    with pytest.raises(DriverError, match="run_scan"):
        list(Batch(2).run([1], lambda value: value))


def test_batch_refuses_memory_before_backend_construction(tmp_path, monkeypatch):
    constructed = []
    monkeypatch.setattr(_scan_batch, "_load", lambda *args: constructed.append(args))
    scan = Scan("batch", base_seed=772, power=[1.0])
    with pytest.raises(DriverError, match="cannot fit one member"):
        run_scan(
            scan,
            build,
            output=tmp_path / "batch.h5",
            executor=Batch(2, 1),
            progress="none",
        )
    assert constructed == []


def test_batch_refuses_mixed_backend_scan(tmp_path):
    def mixed(point):
        return build(point, "numpy_ref" if point.index == 0 else "numba_cpu")

    scan = Scan("batch", base_seed=772, power=[1.0, 2.0])
    with pytest.raises(DriverError, match="mixed backends"):
        run_scan(
            scan,
            mixed,
            output=tmp_path / "batch.h5",
            executor=Batch(2),
            progress="none",
        )


@pytest.mark.parametrize("free_fraction", [0, 1])
def test_batch_device_budget_refuses_or_splits_before_group_allocation(
    tmp_path, monkeypatch, free_fraction
):
    monkeypatch.setattr(
        NumbaCpuBackend,
        "available_device_memory",
        lambda self: free_fraction * sum(self.memory_ledger().values()),
    )
    scan = Scan("device_budget", base_seed=313, power=[1.0, 2.0, 3.0])
    executor = Batch(3)
    options = {
        "output": tmp_path / "budget.h5",
        "executor": executor,
        "progress": "none",
    }
    builder = partial(build, backend="numba_cpu")
    if free_fraction == 0:
        with pytest.raises(DriverError, match="cannot fit one member"):
            run_scan(scan, builder, **options)
        assert executor.budgets[0]["admitted"] == 0
    else:
        result = run_scan(scan, builder, **options)
        assert result.status.tolist() == ["done"] * 3
        assert executor.budgets[0]["admitted"] == 1
        assert executor.budgets[0]["host_admitted"] == 3


def test_batch_custom_reducer_failure_isolated_and_retryable(tmp_path):
    scan = Scan("batch", base_seed=772, power=[1.0, 2.0, 3.0])
    fail = [True]

    def reduce(output):
        assert output.n_saved == 3
        assert output.fields().shape[0] == 3
        assert output.metadata is not None
        assert output.metadata.noise is not None
        if fail[0] and output.metadata.noise.seed == scan.point(1).seed:
            raise RuntimeError("deliberate member reduction failure")
        return np.asarray(output["stats/energy"][-1].sum())

    options = {
        "output": tmp_path / "batch.h5",
        "executor": Batch(3),
        "reductions": [scans.custom("energy", reduce, (), "J")],
        "progress": "none",
    }
    result = run_scan(scan, build, **options)
    assert result.status.tolist() == ["done", "failed", "done"]
    assert "deliberate member reduction failure" in str(result.point(1)["error"])
    failed = result.point(1)
    assert len(str(failed["configuration_hash"])) == 64
    metadata = cast(Mapping, failed["metadata"])
    assert metadata["noise"]["seed"] == scan.point(1).seed
    assert metadata["backend"]["settings"]["batch_size"] == 3
    assert not any(name.startswith(("fields/", "stats/")) for name in failed)
    assert "fields/values" not in result.point(0)
    fail[0] = False
    result = run_scan(scan, build, retry_failed=True, **options)
    assert result.status.tolist() == ["done"] * 3


def test_batch_plane_capture_failure_isolated_before_later_planes(
    tmp_path, monkeypatch
):
    scan = Scan("capture_failure", base_seed=772, power=[1.0, 2.0, 3.0])
    original = _scan_batch._Capture.save
    calls = []

    def capture(self, plane):
        assert self.metadata is not None and self.metadata.noise is not None
        seed = self.metadata.noise.seed
        calls.append(seed)
        if seed == scan.point(1).seed:
            raise OutputError("member capture unavailable")
        return original(self, plane)

    monkeypatch.setattr(_scan_batch._Capture, "save", capture)
    result = run_scan(
        scan, build, output=tmp_path / "scan.h5", executor=Batch(3), progress="none"
    )
    assert result.status.tolist() == ["done", "failed", "done"]
    assert calls.count(scan.point(1).seed) == 1
    assert calls.count(scan.point(0).seed) == 3
    assert "member capture unavailable" in str(result.point(1)["error"])


def test_batch_march_failure_preserves_each_members_provenance(tmp_path, monkeypatch):
    def failed(*args):
        raise RuntimeError("batch execution unavailable")

    monkeypatch.setattr(NumpyBatchBackend, "propagate_to", failed)
    scan = Scan("march_failure", base_seed=772, power=[1.0, 2.0])
    result = run_scan(
        scan, build, output=tmp_path / "scan.h5", executor=Batch(2), progress="none"
    )
    assert result.status.tolist() == ["failed", "failed"]
    for point in scan.points():
        stored = result.point(point.index)
        assert "batch execution unavailable" in str(stored["error"])
        assert cast(Mapping, stored["metadata"])["noise"]["seed"] == point.seed
        assert not any(isinstance(value, np.ndarray) for value in stored.values())


def test_batch_builder_failure_does_not_discard_successful_members(tmp_path):
    def builder(point):
        if point.index == 1:
            raise RuntimeError("point construction unavailable")
        return build(point)

    scan = Scan("builder_failure", base_seed=772, power=[1.0, 2.0, 3.0])
    result = run_scan(
        scan, builder, output=tmp_path / "scan.h5", executor=Batch(3), progress="none"
    )
    assert result.status.tolist() == ["done", "failed", "done"]


def test_batch_requires_the_metadata_its_capture_opened_with(tmp_path, monkeypatch):
    original = _scan_batch._Capture.close

    def closed(self):
        original(self)
        self._metadata = None

    monkeypatch.setattr(_scan_batch._Capture, "close", closed)
    scan = Scan("no_metadata", base_seed=772, power=[1.0, 2.0])
    result = run_scan(
        scan, build, output=tmp_path / "scan.h5", executor=Batch(2), progress="none"
    )
    assert result.status.tolist() == ["failed", "failed"]
    assert "without the run metadata" in str(result.point(0)["error"])


def test_batch_refuses_changed_configuration_of_completed_point(tmp_path):
    scan = Scan("changed", base_seed=772, power=[1.0])
    options = {"output": tmp_path / "scan.h5", "executor": Batch(2), "progress": "none"}
    run_scan(scan, build, **options)

    def changed(point):
        simulation = build(point)
        return simulation.with_input(simulation.translate().initial.field0[0] * 2)

    with pytest.raises(OutputError, match="different configuration"):
        run_scan(scan, changed, **options)
