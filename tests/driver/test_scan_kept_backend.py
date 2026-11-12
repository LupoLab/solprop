# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A local scan on a device keeps one backend while its points differ only in input.

solprop #298. The device is a patched predicate, so that the worker's choices run on any
build, and a real device runs this module too: what a kept backend owes a point is the
same on each, which is the point it would have computed on a backend of its own.
"""

import threading
from collections.abc import Mapping
from functools import partial
from typing import cast

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER, assert_points_matched
from solprop.driver import Simulation, _scan_batch, _scan_run, runner, simulation
from solprop.driver.executors import Local, Pool
from solprop.driver.scan import Scan, run_scan
from solprop.grid import EnvelopeGrid
from solprop.noise import input_noise
from solprop.processing import scans
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


def base(backend="numpy_ref"):
    """Return a short soliton fibre on a prescribed schedule, one discrete march."""
    grid = EnvelopeGrid.from_limits(0.001, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID)
    return Simulation.from_gnlse(
        0.11,
        0.001,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        grid=grid,
        absorber=bare_absorber(grid),
        width=100e-15,
        peak_power=1.0,
        shape="sech",
        save_planes=[0.0005, 0.001],
        schedule=FixedSchedule.uniform(
            0.0, 0.001, 0.00025, save_planes=[0.0005, 0.001]
        ),
        stats=StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2),
        integrator=FIXED_ORDER,
        backend=backend,
    )


def with_pulse(shared, point):
    """Return the shared simulation with a point's own pulse and noise."""
    pulse = SechPulse(
        wavelength=835e-9, width=100e-15, peak_power=float(point.values["power"])
    )
    return shared.with_input(
        pulse, noise=input_noise(shared.grid, np.random.default_rng(point.seed))
    )


def _record(metadata):
    """Return a point's record without when it was made and how long planning took."""
    backend = metadata["backend"]
    settings = {
        name: value
        for name, value in backend["settings"].items()
        if name != "planning_seconds"
    }
    return {
        name: {**backend, "settings": settings} if name == "backend" else value
        for name, value in metadata.items()
        if name != "created"
    }


SCAN = Scan("kept", base_seed=772, power=[1.0, 2.0, 3.0])
OPTIONS = {"fields": "all", "progress": "none", "reductions": [scans.peak_power()]}


@pytest.fixture
def made(monkeypatch):
    """Watch the backends a run constructs, and that two are never held at once."""
    backends = []
    original = simulation.select

    def watched(*arguments, **keywords):
        assert all(backend.closed for backend in backends)
        backends.append(original(*arguments, **keywords))
        return backends[-1]

    for module in (simulation, runner):
        monkeypatch.setattr(module, "select", watched)
    return backends


@pytest.fixture
def on_a_device(monkeypatch):
    monkeypatch.setattr(_scan_run, "_device_build", lambda name: True)


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_a_device_scan_of_inputs_keeps_one_backend_and_the_same_points(
    tmp_path, monkeypatch, made, backend
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    builder = partial(with_pulse, base(backend))
    monkeypatch.setattr(_scan_run, "_device_build", lambda name: False)
    reference = run_scan(SCAN, builder, output=tmp_path / "alone.h5", **OPTIONS)
    assert len(made) == 3
    made.clear()
    monkeypatch.setattr(_scan_run, "_device_build", lambda name: True)
    result = run_scan(SCAN, builder, output=tmp_path / "kept.h5", **OPTIONS)
    assert result.status.tolist() == ["done"] * 3
    assert len(made) == 1
    assert made[0].closed
    # Each point's own noise is on the backend, or these would not match.
    assert_points_matched(result, reference, SCAN, builder, what="kept point")
    for point in SCAN.points():
        ours, theirs = result.point(point.index), reference.point(point.index)
        assert ours["configuration_hash"] == theirs["configuration_hash"]
        assert _record(ours["metadata"]) == _record(theirs["metadata"])
        assert (
            cast(Mapping, ours["boundary_monitor"]).keys()
            == cast(Mapping, theirs["boundary_monitor"]).keys()
        )
    # A rerun finds them all.
    assert (
        run_scan(SCAN, builder, output=tmp_path / "kept.h5", **OPTIONS).status.tolist()
        == ["done"] * 3
    )


def test_a_point_that_shares_no_translation_takes_its_own_backend(
    tmp_path, made, on_a_device
):
    result = run_scan(
        SCAN,
        lambda point: with_pulse(base(), point),
        output=tmp_path / "scan.h5",
        **OPTIONS,
    )
    assert result.status.tolist() == ["done"] * 3
    assert len(made) == 3
    assert all(backend.closed for backend in made)


def test_a_change_of_problem_closes_the_kept_backend_first(tmp_path, made, on_a_device):
    # The fixture refuses a second while the first is open, so this also holds that.
    one, other = base(), base()

    def builder(point):
        return with_pulse(one if point.index < 2 else other, point)

    result = run_scan(
        Scan("kept", base_seed=772, power=[1.0, 2.0, 3.0]),
        builder,
        output=tmp_path / "scan.h5",
        **OPTIONS,
    )
    assert result.status.tolist() == ["done"] * 3
    assert len(made) == 2
    assert all(backend.closed for backend in made)


def test_a_point_off_the_device_closes_the_kept_backend_first(
    tmp_path, monkeypatch, made
):
    # The fixture refuses a second while the first is open, so this also holds that.
    shared = base()
    off_the_device = [False]
    monkeypatch.setattr(_scan_run, "_device_build", lambda name: not off_the_device[0])

    def builder(point):
        off_the_device[0] = point.index == 1
        return with_pulse(shared, point)

    result = run_scan(SCAN, builder, output=tmp_path / "scan.h5", **OPTIONS)
    assert result.status.tolist() == ["done"] * 3
    assert len(made) == 3
    assert all(backend.closed for backend in made)


def test_a_failed_point_takes_the_kept_backend_with_it(
    tmp_path, monkeypatch, made, on_a_device
):
    builder = partial(with_pulse, base())
    reference = run_scan(SCAN, builder, output=tmp_path / "alone.h5", **OPTIONS)
    made.clear()
    original = _scan_batch._Capture.save

    def capture(self, plane):
        if self.metadata.noise.seed == SCAN.point(1).seed:
            raise RuntimeError("deliberate capture failure")
        original(self, plane)

    with monkeypatch.context() as patched:
        patched.setattr(_scan_batch._Capture, "save", capture)
        result = run_scan(SCAN, builder, output=tmp_path / "scan.h5", **OPTIONS)
    assert result.status.tolist() == ["done", "failed", "done"]
    assert "deliberate capture failure" in str(result.point(1)["error"])
    assert len(made) == 2
    assert all(backend.closed for backend in made)
    # The next point is what it would have been.
    ours = np.asarray(result.point(2)["fields/values"])
    theirs = np.asarray(reference.point(2)["fields/values"])
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9 * np.max(np.abs(theirs)))


def test_a_stop_closes_the_kept_backend_and_a_rerun_finishes(
    tmp_path, made, on_a_device
):
    flag = threading.Event()
    shared = base()

    def builder(point):
        # Set after the check before the point, so the point itself sees the stop.
        if point.index == 1:
            flag.set()
        return with_pulse(shared, point)

    output = run_scan(SCAN, builder, output=tmp_path / "scan.h5", stop=flag, **OPTIONS)
    assert output.status.tolist() == ["done", "todo", "todo"]
    assert all(backend.closed for backend in made)
    rerun = run_scan(
        SCAN, partial(with_pulse, shared), output=tmp_path / "scan.h5", **OPTIONS
    )
    assert rerun.status.tolist() == ["done"] * 3


def test_a_scan_off_a_device_keeps_no_backend(tmp_path, made):
    # A CPU's backend costs little beside its march, and its memory goes back to the
    # next point's build (solprop #298), so a numpy_ref scan builds one a point.
    assert not _scan_run._device_build("numpy_ref")
    run_scan(
        SCAN,
        partial(with_pulse, base()),
        output=tmp_path / "scan.h5",
        executor=Local(),
        **OPTIONS,
    )
    assert len(made) == 3


def test_a_pool_keeps_no_backend(tmp_path, monkeypatch):
    # Each point's process holds its own, and what a pool admits assumes it's freed.
    monkeypatch.setattr(_scan_run, "_device_build", lambda name: True)
    kept = []
    original = _scan_run._Work.__init__

    def recording(self, *arguments, **keywords):
        original(self, *arguments, **keywords)
        kept.append(self.keeps_backend)

    monkeypatch.setattr(_scan_run._Work, "__init__", recording)
    run_scan(
        SCAN, _module_point, output=tmp_path / "scan.h5", executor=Pool(1), **OPTIONS
    )
    assert kept == [False]


def _module_point(point):
    return with_pulse(base(), point)
