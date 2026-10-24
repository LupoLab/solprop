# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import json
import logging
import os
import time
from collections.abc import Mapping
from typing import cast

import h5py
import numpy as np
import pytest

from march_bounds import (
    FIXED_ORDER,
    STATISTIC_FACTOR,
    assert_matched,
    simulation_bound,
)
from solprop import _process
from solprop.driver._scan_context import _EXECUTOR
from solprop.driver.executors import Local, pool
from solprop.driver.executors.pool import MemoryBudget, Pool
from solprop.driver.scan import Scan, run_scan
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError, OutputError, ProcessingError
from solprop.noise import input_noise
from solprop.output._scan_lock import lock_path
from solprop.output.hdf5 import _dataset, _group
from solprop.processing import scans
from solprop.pulses import SechPulse
from solprop.scan import main
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


@pytest.fixture(autouse=True)
def _controller_without_history(monkeypatch):
    """Size the pool from the scan alone, not from this test process's past.

    The pool reserves the controller's peak resident memory. Here the controller is a
    long-lived test worker whose peak comes from whatever tests it ran before, which on
    a small runner (16 GB, several workers) leaves no room for one scan worker. These
    tests are about the scan, not the budget, which has its own test below.
    """
    monkeypatch.setattr(pool, "peak_rss", lambda: 0)


def build_point(point):
    """Return a short soliton fibre on a prescribed schedule at a fixed order.

    So that a scan point and the same simulation run alone are one discrete march,
    compared within its rounding budget (SEP 25 4.4): no controller decision is
    left for rounding to fork.
    """
    return Simulation.from_gnlse(
        0.11,
        0.001,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=float(point.values["power"]),
        pulseshape="sech",
        n_save=3,
        schedule=FixedSchedule.uniform(
            0.0, 0.001, 0.00025, save_planes=[0.0005, 0.001]
        ),
        integrator=FIXED_ORDER,
        backend="numpy_ref",
    )


def assert_same_march(stored, direct, point, *, statistic=False):
    """Assert a scan's stored array is the point's own run, to the march's budget."""
    bound = simulation_bound(build_point(point))
    assert_matched(stored, direct, STATISTIC_FACTOR * bound if statistic else bound)


def broken_reducer(output):
    raise RuntimeError("deliberate reducer failure")


def custom_energy(output):
    return np.asarray(output["stats/energy"][-1].sum())


def test_scan_order_global_subsets_and_seed_independence():
    scan = Scan("scan", base_seed=91, power=[1.0, 2.0], gas=["Ar", "He", "Ne"])
    assert scan.shape == (2, 3)
    assert [p.multi_index for p in scan.points()] == list(np.ndindex(2, 3))
    assert [p.index for p in scan.subset(range(2, 5)).points()] == [2, 3, 4]
    assert [p.index for p in scan.chunk(1, 2).points()] == [1, 3, 5]
    for point in scan.chunk(1, 2).points():
        assert point.seed == scan.point(point.index).seed
    axes = scan.variables
    axes["power"][0] = 9
    assert scan.point(0).values["power"] == 1
    assert len({p.seed for p in scan.points()}) == 6
    explicit = Scan("seeds", base_seed=91, seed=[7, 8])
    assert [p.seed for p in explicit.points()] == [7, 8]


@pytest.mark.parametrize(
    "call",
    [
        lambda: Scan("bad/name"),
        lambda: Scan("scan", power=[[1, 2]]),
        lambda: Scan("scan", seed=[-1]),
        lambda: Scan("scan", base_seed=-1),
        lambda: Scan("scan", power=[1]).point(1),
        lambda: Scan("scan", power=[1]).chunk(1, 1),
        lambda: Scan("scan", power=[1]).subset([0, 0]),
        lambda: Scan("scan", power=[1]).subset([0]).add("extra", [1]),
        lambda: Scan("scan", power=[1]).add("power", [2]),
        lambda: Scan("scan").add("", [1]),
    ],
)
def test_scan_invalid_declarations(call):
    with pytest.raises(DriverError):
        call()


def test_input_update_shares_tables_and_changes_metadata():
    base = build_point(Scan("scan", power=[1.0]).point(0))
    original = base.translate()
    pulse = SechPulse(wavelength=835e-9, width=100e-15, peak_power=2.0)
    updated = base.with_input(pulse)
    new = updated.translate()
    assert new.linear is original.linear
    assert new.transform is original.transform
    assert new.responses is original.responses
    assert new.initial is not original.initial
    assert (
        updated.metadata().configuration_hash() != base.metadata().configuration_hash()
    )
    fresh = build_point(Scan("scan", power=[2.0]).point(0))
    # The launch is formed host-side by the same pulse arithmetic either way: exact.
    np.testing.assert_array_equal(new.initial.field0, fresh.translate().initial.field0)
    # Two runs of one problem: the same march, within its rounding budget.
    assert_same_march(
        updated.run(progress="none").fields(),
        fresh.run(progress="none").fields(),
        Scan("scan", power=[2.0]).point(0),
    )


@pytest.mark.medium
@pytest.mark.parametrize(
    "executor", [Local(), Pool(workers=2), Pool(workers=2, fresh=True)]
)
def test_scan_equals_single_points_and_reruns_skip(tmp_path, executor):
    scan = Scan("scan", power=[1.0, 2.0])
    path = tmp_path / "scan.h5"
    output = run_scan(
        scan,
        build_point,
        executor=executor,
        output=path,
        reductions=[scans.peak_power()],
        fields="all",
        progress="none",
    )
    assert output.status.tolist() == ["done", "done"]
    for point in scan.points():
        direct = build_point(point).run(progress="none")
        stored = output.point(point.index)
        # Not bitwise: a worker process's library state may order sums differently.
        assert_same_march(stored["fields/values"], direct.fields(), point)
        assert_same_march(
            stored["stats/energy"], direct["stats/energy"], point, statistic=True
        )
    before = output.point(0)["fields/values"]
    run_scan(
        scan,
        build_point,
        executor=executor,
        output=path,
        reductions=[scans.peak_power()],
        fields="all",
        progress="none",
    )
    # A rerun skips the completed point: what is stored is untouched, exactly.
    np.testing.assert_array_equal(output.point(0)["fields/values"], before)
    assert output.as_grid("reduced/peak_power").shape == (2,)
    np.testing.assert_array_equal(output.where(power=2.0), [1])


def test_reductions_and_selected_fields(tmp_path):
    scan = Scan("scan", power=[1.0])
    reductions = [
        scans.peak_power(),
        scans.mode_energies(),
        scans.energy_in_band(600e-9, 1e-6),
        scans.spectrum_in_band(600e-9, 1e-6),
        scans.field_at_plane(),
        scans.energy_in_window(-100e-15, 100e-15),
        scans.custom("custom_energy", custom_energy, (), "J"),
    ]
    result = run_scan(
        scan,
        build_point,
        output=tmp_path / "scan.h5",
        reductions=reductions,
        fields="planes",
        field_planes=[0, -1],
        progress="none",
    )
    assert result.status.tolist() == ["done"]
    direct = build_point(scan.point(0)).run(progress="none")
    point = result.point(0)
    for reduction in reductions:
        assert_same_march(
            point[f"reduced/{reduction.name}"],
            reduction(direct),
            scan.point(0),
            statistic=True,
        )
    assert_same_march(point["fields/values"], direct.fields()[[0, -1]], scan.point(0))


def test_failure_does_not_publish_results_and_requires_retry(tmp_path):
    scan = Scan("scan", power=[1.0, 2.0])
    reducer = scans.custom("fail", broken_reducer, (), "1")
    path = tmp_path / "scan.h5"
    result = run_scan(
        scan, build_point, output=path, reductions=[reducer], progress="none"
    )
    assert result.status.tolist() == ["failed", "failed"]
    assert "deliberate reducer failure" in str(result.point(0)["error"])
    with h5py.File(path) as file:
        for index in range(2):
            staged = _group(file, f"staged/{index}")
            assert "metadata_refs" in staged.attrs
            assert len(staged) == 0
            assert not any(
                isinstance(value, np.ndarray) for value in result.point(index).values()
            )
    with pytest.raises(ProcessingError, match="no completed"):
        result.as_grid("reduced/fail")
    result = run_scan(
        scan, build_point, output=path, reductions=[reducer], progress="none"
    )
    assert result.status.tolist() == ["failed", "failed"]


def test_memory_budget_accounts_for_headroom_and_refuses_no_capacity():
    budget = MemoryBudget(
        limit=1000, controller=100, per_worker=200, cores=8, threads=1
    )
    assert budget.workers() == 3
    assert budget.workers(2) == 2
    with pytest.raises(DriverError, match="cannot fit"):
        dataclasses.replace(budget, limit=200).workers()


def test_scan_contract_change_refuses_append(tmp_path):
    path = tmp_path / "scan.h5"
    run_scan(Scan("scan", power=[1.0]), build_point, output=path, progress="none")
    with pytest.raises(OutputError, match="axes"):
        run_scan(Scan("scan", power=[2.0]), build_point, output=path, progress="none")


def periodic_point(point):

    simulation = build_point(point)
    return Simulation(
        simulation.grid,
        simulation.modes,
        simulation.responses,
        simulation._field0,
        settings=dataclasses.replace(
            simulation.settings,
            stats=StatsSpec(
                names=("energy", "peak_power"), periodic=("energy",), period=1
            ),
        ),
    )


def test_periodic_statistics_equal_single_run(tmp_path):
    scan = Scan("periodic", power=[1.0, 2.0])
    result = run_scan(
        scan, periodic_point, output=tmp_path / "scan.h5", progress="none"
    )
    assert result.status.tolist() == ["done", "done"]
    for point in scan.points():
        direct = periodic_point(point).run(progress="none")
        # The rows' positions and counts follow the prescribed schedule: exact.
        for name in ("z", "step", "counts"):
            np.testing.assert_array_equal(
                result.point(point.index)[f"step_stats/{name}"],
                direct[f"step_stats/{name}"],
            )
        assert_same_march(
            result.point(point.index)["step_stats/energy"],
            direct["step_stats/energy"],
            point,
            statistic=True,
        )


def test_scan_reclaims_stale_running_and_leaves_recent_claim(tmp_path):

    scan = Scan("scan", power=[1.0, 2.0])
    path = tmp_path / "scan.h5"
    output = run_scan(scan.subset([0]), build_point, output=path, progress="none")
    with h5py.File(path, "a") as file:
        _dataset(file, "points/status")[1] = "running"
        _dataset(file, "points/updated")[1] = time.time()
    run_scan(scan, build_point, output=path, progress="none")
    assert output.status.tolist() == ["done", "running"]
    with h5py.File(path, "a") as file:
        _dataset(file, "points/updated")[1] = 0.0
    run_scan(scan, build_point, output=path, progress="none")
    assert output.status.tolist() == ["done", "done"]


def test_resume_builder_failure_keeps_the_completed_point_and_its_arrays(
    tmp_path, caplog
):
    scan = Scan("resume", power=[1.0, 2.0])
    path = tmp_path / "scan.h5"
    first = run_scan(scan, build_point, output=path, fields="all", progress="none")
    assert first.status.tolist() == ["done", "done"]
    before = np.asarray(first.point(0)["fields/values"])

    def broken(point):
        if point.index == 0:
            raise RuntimeError("transient builder failure")
        return build_point(point)

    with caplog.at_level(logging.ERROR, logger="solprop.driver._scan_run"):
        resumed = run_scan(scan, broken, output=path, fields="all", progress="none")
    assert resumed.status.tolist() == ["done", "done"]
    # The completed point is kept, not recomputed: its stored arrays are untouched.
    np.testing.assert_array_equal(resumed.point(0)["fields/values"], before)
    assert "transient builder failure" in caplog.text
    with h5py.File(path) as file:
        assert len(_group(file, "staged/0")) > 0
    # The controller's advisory lock is taken on the direct route too, not only
    # by a queue collection, so a reader in another process can honour it.
    assert lock_path(path).exists()


def test_pool_rejects_unpickleable_builder_before_output(tmp_path):
    path = tmp_path / "scan.h5"
    with pytest.raises(DriverError, match="pickleable"):
        run_scan(
            Scan("scan", power=[1.0]),
            lambda point: build_point(point),  # noqa: PLW0108 - intentionally unpickleable
            executor=Pool(),
            output=path,
            progress="none",
        )
    assert not path.exists()


@pytest.mark.parametrize(
    "controls",
    [
        {"fields": "planes"},
        {"fields": "none", "field_planes": [0]},
        {"stale_age": 0.0},
        {"stats": "energy"},
        {"reductions": [scans.peak_power(), scans.peak_power()]},
    ],
)
def test_scan_rejects_invalid_retention(tmp_path, controls):
    with pytest.raises(DriverError):
        run_scan(
            Scan("scan", power=[1.0]),
            build_point,
            output=tmp_path / "scan.h5",
            **controls,
        )


def test_cli_local_and_explicit_executor_precedence(tmp_path):

    script = tmp_path / "scan_script.py"
    marker = tmp_path / "marker"
    script.write_text(
        "from pathlib import Path\n"
        "from solprop.scan import current_executor\n"
        "from solprop.driver.executors import Local\n"
        "assert isinstance(current_executor(), Local)\nPath("
        + repr(str(marker))
        + ').write_text("ran")\n'
    )
    assert main([str(script), "--local"]) == 0
    assert marker.read_text() == "ran"


def noisy_point(point):

    base = build_point(point)
    return base.with_input(
        base.initial, noise=input_noise(base.grid, np.random.default_rng(point.seed))
    )


def test_scan_noise_seeds_and_replacement_flags(tmp_path):
    scan = Scan("noise", base_seed=23, power=[1.0, 2.0])
    result = run_scan(
        scan, noisy_point, output=tmp_path / "noise.h5", fields="all", progress="none"
    )
    assert result.status.tolist() == ["done", "done"]
    for point in scan.points():
        direct = noisy_point(point)
        assert direct.translate().flags.has_noise
        disabled = direct.with_input(direct.initial, noise=None)
        assert not disabled.translate().flags.has_noise
        assert disabled.translate().initial.noise0 is None
        # The same stored realisation on both sides: a matched march (SEP 25 4.5).
        assert_same_march(
            result.point(point.index)["fields/values"],
            direct.run(progress="none").fields(),
            point,
        )
    with h5py.File(result.path) as file:
        assert list(_dataset(file, "points/seed").asstr()[...]) == list(
            _dataset(file, "points/noise_seed").asstr()[...]
        )


@pytest.mark.medium
def test_scan_backend_selection_is_preserved(tmp_path):

    def numba(point):
        base = build_point(point)
        return Simulation(
            base.grid,
            base.modes,
            base.responses,
            base.initial,
            settings=dataclasses.replace(base.settings, backend="numba_cpu"),
        )

    scan = Scan("numba", power=[1.0])
    result = run_scan(
        scan, numba, output=tmp_path / "numba.h5", fields="all", progress="none"
    )
    assert result.status.tolist() == ["done"]
    assert_same_march(
        result.point(0)["fields/values"],
        numba(scan.point(0)).run(progress="none").fields(),
        scan.point(0),
    )
    assert (
        cast(Mapping, cast(Mapping, result.point(0)["metadata"])["backend"])["name"]
        == "numba_cpu"
    )


def crashing_builder(point):
    if point.index == 0:
        os._exit(23)
    return build_point(point)


@pytest.mark.medium
def test_pool_records_abnormal_worker_exit_and_continues(tmp_path):
    scan = Scan("crash", power=[1.0, 2.0])
    result = run_scan(
        scan,
        crashing_builder,
        executor=Pool(workers=1),
        output=tmp_path / "crash.h5",
        progress="none",
    )
    assert result.status.tolist() == ["failed", "done"]
    assert "exited without" in str(result.point(0)["error"])


def crashes_when_marked(point):
    if point.index == 0 and os.environ.get("SOLPROP_TEST_CRASH") == "1":
        os._exit(23)
    return build_point(point)


@pytest.mark.medium
def test_worker_crash_on_a_completed_point_keeps_it_and_runs_the_rest(
    tmp_path, monkeypatch
):
    scan = Scan("crash", power=[1.0, 2.0])
    path = tmp_path / "crash.h5"
    run_scan(scan.subset([0]), crashes_when_marked, output=path, progress="none")
    monkeypatch.setenv("SOLPROP_TEST_CRASH", "1")
    result = run_scan(
        scan,
        crashes_when_marked,
        executor=Pool(workers=1),
        output=path,
        progress="none",
    )
    # The crash is the worker's, not the committed point's: it stays done with
    # its own error column untouched, and the pending point still runs.
    assert result.status.tolist() == ["done", "done"]
    assert result.point(0)["error"] == ""


def test_explicit_executor_overrides_cli_default(tmp_path):

    class RefuseExecutor:
        def run(self, points, work):
            raise AssertionError("CLI default must not override an explicit executor")

    token = _EXECUTOR.set(RefuseExecutor())
    try:
        result = run_scan(
            Scan("override", power=[1.0]),
            build_point,
            executor=Local(),
            output=tmp_path / "scan.h5",
            progress="none",
        )
        assert result.status.tolist() == ["done"]
    finally:
        _EXECUTOR.reset(token)


def test_pool_rejects_unpickleable_reducer_before_output(tmp_path):
    reducer = scans.custom("value", lambda output: np.asarray(output["z"][-1]), (), "m")
    path = tmp_path / "scan.h5"
    with pytest.raises(DriverError, match="pickleable"):
        run_scan(
            Scan("scan", power=[1.0]),
            build_point,
            reductions=[reducer],
            executor=Pool(),
            output=path,
            progress="none",
        )
    assert not path.exists()


def test_linux_allocation_uses_affinity_slurm_and_cgroup(tmp_path, monkeypatch):
    # The CPU quota is the thread budget's (ADR-052), read from this process's own
    # cgroup: a two-CPU quota on a fake root, through the real reader.
    root = tmp_path / "root"
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/self/cgroup").write_text("0::/\n")
    (root / "sys/fs/cgroup").mkdir(parents=True)
    (root / "sys/fs/cgroup/cpu.max").write_text("200000 100000")
    monkeypatch.setattr(
        _process, "_own_cgroup_cpus", lambda: _process.cgroup_cpus(root)
    )
    monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    memory = tmp_path / "memory.max"
    memory.write_text(str(1024**3))
    missing = tmp_path / "missing"
    monkeypatch.setattr(
        pool,
        "Path",
        lambda name: {"/sys/fs/cgroup/memory.max": memory}.get(name, missing),
    )
    monkeypatch.setattr(pool.sys, "platform", "linux")
    monkeypatch.setattr(
        pool.os, "sched_getaffinity", lambda pid: {2, 4, 6, 8}, raising=False
    )
    monkeypatch.setattr(
        pool.os,
        "sysconf",
        lambda key: {"SC_PHYS_PAGES": 2**21, "SC_PAGE_SIZE": 4096}[key],
        raising=False,  # Windows has no sysconf; this test simulates Linux
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    assert pool._cpus() == (2, 4)
    assert pool._memory_limit() == 1024**3


def build_numba_point(point):
    simulation = build_point(point)
    simulation.settings = dataclasses.replace(simulation.settings, backend="numba_cpu")
    return simulation


def test_scan_records_numba_execution_policies_per_point(tmp_path):
    scan = Scan("numba_policies", power=[1.0, 2.0])
    destination = tmp_path / "policies.h5"
    result = run_scan(
        scan,
        build_numba_point,
        output=destination,
        fields="none",
        stats="none",
        progress="none",
    )
    assert result.status.tolist() == ["done", "done"]
    keys = {
        "parallel_nodes",
        "kernel_regime_elements",
        "modal_projection",
        "raman_fftw_threads",
        "raman_tail_nodes",
        "raman_tail_fftw_threads",
        "plasma_fraction_reuse",
        "column_plasma_magnitude",
    }
    with h5py.File(destination, "r") as file:
        for index in range(2):
            backend = json.loads(str(_dataset(file, "points/backend").asstr()[index]))
            assert backend["name"] == "numba_cpu"
            settings = backend["settings"]
            assert keys <= settings.keys()
            metadata = cast(Mapping, result.point(index)["metadata"])
            assert metadata["backend"]["settings"] == settings
