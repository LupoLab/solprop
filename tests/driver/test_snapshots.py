# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Snapshot committed endpoints without disturbing the live optical/atomic march."""

import dataclasses
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests.driver.test_tdse_natural_output import _simulation, assert_no_extra_call
from tests.hacc_cases import alpha_law, install, plain

import solprop.driver.run as run_module
import solprop.driver.simulation as simulation_module
from march_bounds import TIGHTER, assert_in_envelope
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.exceptions import DriverError, ResponseBudgetError, TDSEDomainWarning
from solprop.log import NullReporter
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output import (
    HDF5Output,
    SnapshotController,
    TDSEArchive,
    open_file,
    open_tdse_archive,
    read_snapshot,
    restore_snapshot,
    resume,
)
from solprop.output import snapshots as module
from solprop.output.schema import PlaneStore
from solprop.stepper import Multistep, PIController, RungeKutta

pytestmark = [
    pytest.mark.medium,
    # The fixtures here launch noise-free into time windows kept tight for speed, which
    # hold more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]


def _run(simulation, output=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEDomainWarning)
        result = simulation.run(output, **{"progress": "none", **kwargs})
    assert isinstance(result, PlaneStore) and result.outcome is not None
    return result


class Requests(NullReporter):
    def __init__(self, controller, count=3):
        self.controller = controller
        self.remaining = count

    def step(self, z, dz, accepted):
        if accepted and self.remaining:
            self.controller.request()
            self.remaining -= 1


def ordinary(backend="numpy_ref", integrator=None, step_planes=1, **keywords):
    if integrator is None:
        integrator = RungeKutta() if backend == "native" else Multistep()
    return Simulation.from_gnlse(
        0.11,
        0.001,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        backend=backend,
        integrator=integrator,
        n_save=2,
        step_planes=step_planes,
        dz_init=1e-4,
        dz_max=1e-4,
        **{"noise": None, **keywords},
    )


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))
    provider = install(monkeypatch)
    provider.response = alpha_law(1.0, lambda5=2.0**-200)
    return provider


def test_several_tdse_snapshots_keep_the_same_plans_and_all_calls(
    fake, tmp_path, monkeypatch, caplog
):
    controller = SnapshotController(tmp_path / "snapshots")
    reporter = Requests(controller)
    built = []
    live = []
    identities = []
    copy = module._copy_generation
    run = simulation_module.run_problem

    def run_with_live_resources(problem, backend, output, **kwargs):
        live.append((backend, kwargs["tdse_binding"]))
        return run(problem, backend, output, **kwargs)

    def watched(*args):
        backend, binding = live[-1]
        identities.append(
            (
                id(backend),
                id(backend._adams),
                id(backend._tdse_binding),
                id(binding._hacc.model),
                tuple(id(plan) for plan in binding._plans.values()),
            )
        )
        before = (len(fake.calls), len(fake.prepared), len(fake.plan_arguments))
        publication = copy(*args)
        assert before == (len(fake.calls), len(fake.prepared), len(fake.plan_arguments))
        built.append(before[1:])
        return publication

    monkeypatch.setattr(module, "_copy_generation", watched)
    monkeypatch.setattr(simulation_module, "run_problem", run_with_live_resources)
    simulation = _simulation()
    output = _run(
        simulation,
        tmp_path / "run.h5",
        progress=reporter,
        snapshots=controller,
        tdse_archive=TDSEArchive(tmp_path / "run.tdse.h5"),
    )
    assert controller.last_result is not None
    assert controller.last_result.disposition == "published"
    assert (
        controller.last_result.pause_seconds >= controller.last_result.copy_seconds > 0
    )
    assert controller.last_result.bytes_copied > 0
    assert len(built) == 3 and all(item == built[0] for item in built)
    assert len(live) == 1 and all(item == identities[0] for item in identities)
    assert len(fake.prepared) == 1
    assert output.outcome is not None
    assert_no_extra_call(output.outcome)
    assert len(fake.calls) == plain(output.outcome.atomic_calls)["attempted"]
    assert plain(output.outcome.logical_calls)["rejected_steps"] > 0
    with open_tdse_archive(tmp_path / "run.tdse.h5") as archive:
        assert len(archive.sessions) == 1
        assert archive.sessions[0].closed
        run_uuid = archive.run_uuid
    generations = sorted(controller._directory.glob("snapshot-*"))
    assert len(generations) == 3
    for index, generation in enumerate(generations):
        _verify_tdse_generation(
            generation,
            tmp_path / f"restore-{index}",
            run_uuid,
            output.outcome.logical_calls["startup"],
            caplog,
        )


def _verify_tdse_generation(generation, destination, run_uuid, startup, caplog):
    info = read_snapshot(generation)
    state = resume(info.files.output)
    assert state.outcome is not None
    assert state.outcome.status == "running" and state.march is not None
    assert state.plane.z == state.march.z == info.z
    assert (
        state.metadata.configuration_hash()
        == _simulation().metadata().configuration_hash()
    )
    assert info.files.tdse_archive is not None
    with open_tdse_archive(info.files.tdse_archive.path) as archive:
        assert archive.run_uuid == run_uuid
        assert len(archive.sessions) == 1 and not archive.sessions[0].closed
        assert all(e.complete for e in archive.evaluations())
    restored = restore_snapshot(generation, destination)
    assert restored.tdse_archive is not None
    assert restored.output.read_bytes() == info.files.output.read_bytes()
    assert (
        restored.tdse_archive.path.read_bytes()
        == info.files.tdse_archive.path.read_bytes()
    )
    caplog.clear()
    with caplog.at_level("INFO", logger="solprop.driver.simulation"):
        _run(
            _simulation(),
            restored.output,
            resume=True,
            tdse_archive=restored.tdse_archive,
        )
    assert "continuing the Adams history" in caplog.text
    after = resume(restored.output)
    assert after.outcome is not None
    assert after.outcome.status == "complete"
    # The pressure kink restarts both histories once; restoration adds no startup.
    assert plain(after.outcome.logical_calls)["startup"] == startup
    with (
        open_file(info.files.output) as original,
        open_file(restored.output) as continued,
    ):
        assert_array_equal(original["field"], continued["field"][: info.n_saved])
    with open_tdse_archive(restored.tdse_archive.path) as archive:
        assert archive.run_uuid == run_uuid
        assert [session.kind for session in archive.sessions] == ["run", "resume"]


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_ordinary_snapshot_resumes_in_a_fresh_process(backend, tmp_path):
    if backend == "native":
        pytest.importorskip("solprop_native")
    controller = SnapshotController(tmp_path / "snapshots")
    ordinary(backend).run(
        tmp_path / "working.h5", snapshots=controller, progress=Requests(controller)
    )
    root = Path(__file__).resolve().parents[2]
    for index, generation in enumerate(controller._directory.glob("snapshot-*")):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from tests.driver.test_snapshots import ordinary
from solprop.output import restore_snapshot, resume
files = restore_snapshot(sys.argv[1], sys.argv[2])
before = resume(files.output)
ordinary(sys.argv[3]).run(files.output, resume=True, progress="none")
after = resume(files.output)
assert after.outcome.status == "complete"
assert (after.outcome.logical_calls.get("startup", 0)
        == before.outcome.logical_calls.get("startup", 0))
""",
                str(generation),
                str(tmp_path / f"restored-{index}"),
                backend,
            ],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join([str(root), str(root / "tests")]),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_snapshot_requires_history_even_above_the_automatic_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run_module, "MARCH_SNAPSHOT_AUTO_BYTES", 0)
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    ordinary().run(tmp_path / "working.h5", snapshots=controller, progress="none")
    assert controller.last_result is not None
    assert controller.last_result.disposition == "published"
    assert controller.last_result.generation is not None
    assert (
        resume(read_snapshot(controller.last_result.generation).files.output).march
        is not None
    )


# Step planes are no longer a requirement: a snapshot is serviced at the next accepted
# endpoint whatever the saved-step sampling (SEP 47 section 5.5).
@pytest.mark.parametrize("policy", ["memory", "dense", "fields", "history"])
def test_snapshot_admission_precedes_provider_preparation(fake, tmp_path, policy):
    simulation = _simulation()
    changes = {
        "fields": {"fields": "none"},
        "history": {"resume_history": False},
    }
    if policy == "dense":
        simulation = ordinary()
        changes["dense"] = {"interpolate_planes": True}
    if policy in changes:
        simulation.settings = dataclasses.replace(
            simulation.settings, **changes[policy]
        )
    controller = SnapshotController(tmp_path / "snapshots")
    path = None if policy == "memory" else tmp_path / "run.h5"
    with pytest.raises(DriverError, match="snapshot"):
        _run(simulation, path, snapshots=controller)
    assert not fake.prepared


def test_snapshots_do_not_reset_cumulative_atomic_budget(fake, tmp_path):
    simulation = _simulation()
    simulation.settings = dataclasses.replace(
        simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=30)
    )
    controller = SnapshotController(tmp_path / "snapshots")
    with pytest.raises(ResponseBudgetError):
        _run(
            simulation,
            tmp_path / "run.h5",
            snapshots=controller,
            tdse_archive=TDSEArchive(tmp_path / "run.tdse.h5"),
            progress=Requests(controller),
        )
    state = resume(tmp_path / "run.h5")
    assert state.outcome is not None
    assert state.outcome.status == "budget_exhausted"
    assert plain(state.outcome.atomic_calls)["attempted"] == 30
    with pytest.raises(ResponseBudgetError):
        _run(
            simulation,
            tmp_path / "run.h5",
            resume=True,
            tdse_archive=TDSEArchive(tmp_path / "run.tdse.h5"),
        )
    state = resume(tmp_path / "run.h5")
    assert state.outcome is not None
    assert plain(state.outcome.atomic_calls)["attempted"] == 30


def test_copy_failure_keeps_live_run_and_previous_generation(
    fake, tmp_path, monkeypatch
):
    controller = SnapshotController(tmp_path / "snapshots")
    original = module._copy_file
    copies = 0

    def failing(*args):
        nonlocal copies
        copies += 1
        if copies == 3:
            raise OSError("synthetic full disk")
        return original(*args)

    monkeypatch.setattr(module, "_copy_file", failing)
    output = _run(
        _simulation(),
        tmp_path / "run.h5",
        snapshots=controller,
        tdse_archive=TDSEArchive(tmp_path / "run.tdse.h5"),
        progress=Requests(controller, 2),
    )
    assert output.outcome is not None
    assert output.outcome.status == "complete"
    assert controller.last_result is not None
    assert controller.last_result.disposition == "failed"
    assert controller.last_result.failure is not None
    assert "synthetic full disk" in controller.last_result.failure
    (generation,) = controller._directory.glob("snapshot-*")
    read_snapshot(generation)
    assert not list(controller._directory.glob(".snapshot-*"))


def subprocess_probe(directory, phase, *arguments):
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.driver.snapshot_worker",
            str(directory),
            phase,
            *map(str, arguments),
        ],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(root), str(root / "tests")]),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signal delivery")
@pytest.mark.parametrize("phase", ["checkpoint", "copy", "reopen", "interrupt"])
def test_signal_is_delivered_only_after_safe_reopening(tmp_path, phase):
    result = subprocess_probe(tmp_path, phase)
    assert result.returncode == 0, result.stdout + result.stderr
    generation = next((tmp_path / "snapshots").glob("snapshot-*"))
    resumed = subprocess_probe(tmp_path, "restore", generation)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGKILL")
def test_kill_during_staging_preserves_previous_generation(tmp_path):
    result = subprocess_probe(tmp_path, "kill")
    assert result.returncode == -9, result.stdout + result.stderr
    assert len(list((tmp_path / "snapshots").glob(".snapshot-*"))) == 1
    generations = list((tmp_path / "snapshots").glob("snapshot-*"))
    assert len(generations) == 1
    read_snapshot(generations[0])
    resumed = subprocess_probe(tmp_path, "restore", generations[0])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr


def test_snapshot_and_resumed_fields_meet_the_tighter_reference_envelope(
    fake, tmp_path
):
    baseline = _run(_simulation())
    tighter = _run(
        _simulation(
            rtol=cast("PIController", _simulation().settings.controller).rtol / TIGHTER
        )
    )
    controller = SnapshotController(tmp_path / "snapshots")
    output = _run(
        _simulation(),
        tmp_path / "working.h5",
        snapshots=controller,
        progress=Requests(controller),
    )
    assert isinstance(output, HDF5Output)
    with open_file(output.path) as stored:
        fields = [stored.fields(convention="fft")[-1]]
    for index, generation in enumerate(controller._directory.glob("snapshot-*")):
        restored = restore_snapshot(generation, tmp_path / f"restore-{index}")
        final = _run(_simulation(), restored.output, resume=True)
        assert isinstance(final, HDF5Output)
        with open_file(final.path) as stored:
            fields.append(stored.fields(convention="fft")[-1])
    # Adaptive runs: SEP 25's envelope against the same case at rtol / 100.
    assert_in_envelope(
        fields,
        tighter.fields(convention="fft")[-1],
        baseline.fields(convention="fft")[-1],
        what="snapshot/resumed endpoint",
    )


def test_referenced_archive_cannot_be_omitted_on_snapshot_resume(fake, tmp_path):
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    _run(
        _simulation(),
        tmp_path / "working.h5",
        snapshots=controller,
        tdse_archive=TDSEArchive(tmp_path / "archive.h5"),
    )
    generation = next(controller._directory.glob("snapshot-*"))
    resumed = subprocess_probe(tmp_path, "restore", generation)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    files = restore_snapshot(generation, tmp_path / "restored-second")
    count = len(fake.prepared)
    with pytest.raises(DriverError, match="referenced"):
        _run(
            _simulation(),
            files.output,
            resume=True,
            snapshots=SnapshotController(tmp_path / "new-snapshots"),
        )
    assert len(fake.prepared) == count


@pytest.mark.skipif("native" not in available(), reason="needs solprop-native")
def test_native_adams_snapshots_carry_the_resident_history(tmp_path, caplog):
    # The native backend's resident Adams history is exported into each periodic
    # snapshot and installed by its resume (SEP 29 section 6.6), which continues it.
    def adams(**keywords):
        return ordinary("native", Multistep(), **keywords)

    baseline = _run(adams())
    rtol = cast("PIController", adams().settings.controller).rtol
    tighter = _run(adams(rtol=rtol / TIGHTER))
    controller = SnapshotController(tmp_path / "snapshots")
    _run(
        adams(),
        tmp_path / "working.h5",
        snapshots=controller,
        progress=Requests(controller),
    )
    generations = sorted(controller._directory.glob("snapshot-*"))
    assert generations
    fields = []
    for index, generation in enumerate(generations):
        files = restore_snapshot(generation, tmp_path / f"restore-{index}")
        march = resume(files.output).march
        assert march is not None and march.rows["store"].any()
        with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
            final = _run(adams(), files.output, resume=True)
        assert "the Adams history starts afresh" not in caplog.text
        assert isinstance(final, HDF5Output)
        with open_file(final.path) as stored:
            fields.append(stored.fields(convention="fft")[-1])
    # Adaptive runs: SEP 25's envelope against the same case at rtol / 100.
    assert_in_envelope(
        fields,
        tighter.fields(convention="fft")[-1],
        baseline.fields(convention="fft")[-1],
        what="native snapshot/resumed endpoint",
    )


@pytest.mark.parametrize("checkpoint_failure", [True, False])
def test_output_close_failure_does_not_replace_checkpoint_failure(
    tmp_path, monkeypatch, checkpoint_failure
):
    from solprop.exceptions import OutputError  # noqa: PLC0415 - fault injection

    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()

    def trim(self, file):
        raise OutputError("trim failure")

    monkeypatch.setattr(HDF5Output, "_trim", trim)
    if checkpoint_failure:
        annotate = HDF5Output.annotate

        def checkpoint(self, name, value):
            if name == module._ANNOTATION:
                raise OutputError("checkpoint failure")
            return annotate(self, name, value)

        monkeypatch.setattr(HDF5Output, "annotate", checkpoint)
    with pytest.raises(
        OutputError,
        match="checkpoint failure" if checkpoint_failure else "trim failure",
    ) as error:
        ordinary().run(tmp_path / "working.h5", snapshots=controller, progress="none")
    if checkpoint_failure:
        assert "trim failure" in str(error.value.__notes__)
    assert not controller._active


def test_ordinary_legacy_resume_can_start_taking_snapshots(tmp_path):
    class Stop(NullReporter):
        def step(self, z, dz, accepted):
            if accepted:
                raise KeyboardInterrupt

    path = tmp_path / "legacy.h5"
    with pytest.raises(KeyboardInterrupt):
        ordinary().run(path, progress=Stop())
    with h5py.File(path, "r+") as file:
        del file["meta/outcome"]
        del file["terminal_checkpoint"]
    assert resume(path).outcome is None
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    _run(ordinary(), path, resume=True, snapshots=controller)
    generation = next(controller._directory.glob("snapshot-*"))
    info = read_snapshot(generation)
    files = restore_snapshot(generation, tmp_path / "restored")
    assert info.z > 0
    assert files.tdse_archive is None
    _run(ordinary(), files.output, resume=True)


def _history(path):
    """Return a closed file's whole step history: starts, lengths and acceptance."""
    history = resume(path).history
    return history.z, history.h, history.accepted


def assert_contiguous(path, length):
    """Assert every accepted step starts where the one before it ended, once each."""
    starts, lengths, accepted = _history(path)
    starts, ends = starts[accepted], (starts + lengths)[accepted]
    assert starts[0] == 0.0
    assert np.all(np.diff(starts) > 0), "an accepted step is recorded twice"
    np.testing.assert_allclose(starts[1:], ends[:-1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(ends[-1], length, rtol=1e-12, atol=0.0)


def test_snapshots_without_step_planes_resume_from_a_running_checkpoint(tmp_path):
    # SEP 47 section 5.5: no step planes; a request due before the first step is
    # serviced at the first accepted endpoint, between saved planes, as a running
    # checkpoint, and later ones at the endpoint after the plane that noticed them.
    length = 0.001
    baseline = _run(ordinary(step_planes=0))
    rtol = cast("PIController", ordinary().settings.controller).rtol
    tighter = _run(ordinary(step_planes=0, rtol=rtol / TIGHTER))
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    working = tmp_path / "working.h5"
    _run(
        ordinary(step_planes=0),
        working,
        snapshots=controller,
        progress=Requests(controller),
    )
    # A snapshot neither lands nor observes: the march is the one without snapshots.
    for got, expected in zip(
        _history(working),
        (
            baseline["step_history/z"],
            baseline["step_history/h"],
            baseline["step_history/accepted"],
        ),
        strict=True,
    ):
        assert_array_equal(got, expected)
    with open_file(working) as stored:
        assert stored.terminal_checkpoint is None  # superseded by the next plane
        planes = np.asarray(stored["z"]).tolist()
    generations = sorted(controller._directory.glob("snapshot-*"))
    assert len(generations) >= 2
    fields = []
    running = 0
    for index, generation in enumerate(generations):
        files = restore_snapshot(generation, tmp_path / f"restore-{index}")
        state = resume(files.output)
        if state.checkpoint is not None:
            running += 1
            assert state.checkpoint.landed
            assert state.plane.z not in planes
            assert state.outcome is not None and state.outcome.status == "running"
            assert state.march is not None and state.march.z == state.plane.z
        final = _run(ordinary(step_planes=0), files.output, resume=True)
        assert isinstance(final, HDF5Output)
        assert final.terminal_checkpoint is None
        assert_contiguous(files.output, length)
        with open_file(final.path) as stored:
            assert np.asarray(stored["z"]).tolist() == planes
            fields.append(stored.fields(convention="fft")[-1])
    assert running >= 1
    # Adaptive runs: SEP 25's envelope against the same case at rtol / 100.
    assert_in_envelope(
        fields,
        tighter.fields(convention="fft")[-1],
        baseline.fields(convention="fft")[-1],
        what="running-checkpoint resumed endpoint",
    )


def test_a_resume_from_a_running_checkpoint_keeps_the_history_once(tmp_path):
    # The attempts before the running checkpoint reach the next plane once, in the
    # live run and across a resume from the checkpoint alike.
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    working = tmp_path / "working.h5"
    _run(ordinary(step_planes=0), working, snapshots=controller)
    (generation,) = controller._directory.glob("snapshot-*")
    files = restore_snapshot(generation, tmp_path / "restored")
    state = resume(files.output)
    assert state.checkpoint is not None
    before = state.history.n_steps
    assert before >= 1
    _run(ordinary(step_planes=0), files.output, resume=True)
    starts, _, _ = _history(files.output)
    # The checkpoint's attempts are the uninterrupted run's first ones, exactly.
    assert_array_equal(starts[:before], _history(working)[0][:before])
    assert_contiguous(working, 0.001)
    assert_contiguous(files.output, 0.001)


def _assert_checkpoint_holds_every_attempt(path):
    """Assert a file's terminal checkpoint carries every attempt since the launch."""
    state = resume(path)
    assert state.checkpoint is not None
    assert state.outcome is not None and state.outcome.status == "failed"
    history = state.history
    starts, ends = (
        history.z[history.accepted],
        (history.z + history.h)[history.accepted],
    )
    assert starts.size >= 1 and starts[0] == 0.0
    np.testing.assert_allclose(starts[1:], ends[:-1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(ends[-1], state.plane.z, rtol=1e-12, atol=0.0)


def test_a_failure_after_a_running_checkpoint_keeps_its_attempts(tmp_path, monkeypatch):
    # The running checkpoint is written, then the snapshot transaction fails outside
    # the session: the terminal checkpoint replacing it still holds the attempts
    # since the last plane, which a resume continues once each.
    from solprop.exceptions import OutputError  # noqa: PLC0415 - fault injection

    annotate = HDF5Output.annotate

    def failing(self, name, value):
        if name == module._ANNOTATION:
            raise OutputError("annotation failure")
        return annotate(self, name, value)

    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    path = tmp_path / "working.h5"
    with monkeypatch.context() as patch:
        patch.setattr(HDF5Output, "annotate", failing)
        with pytest.raises(OutputError, match="annotation failure"):
            ordinary(step_planes=0).run(path, snapshots=controller, progress="none")
    _assert_checkpoint_holds_every_attempt(path)
    _run(ordinary(step_planes=0), path, resume=True)
    assert_contiguous(path, 0.001)


def test_an_interrupt_out_of_a_propagation_keeps_the_running_attempts(tmp_path):
    # A KeyboardInterrupt escapes the propagation (the session catches exceptions
    # only) three accepted steps in, after a running checkpoint at the first.
    class Interrupt(NullReporter):
        accepted = 0

        def step(self, z, dz, accepted):
            self.accepted += accepted
            if self.accepted == 3:
                raise KeyboardInterrupt

    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    path = tmp_path / "working.h5"
    with pytest.raises(KeyboardInterrupt):
        ordinary(step_planes=0).run(path, snapshots=controller, progress=Interrupt())
    assert len(list(controller._directory.glob("snapshot-*"))) == 1
    _assert_checkpoint_holds_every_attempt(path)
    assert resume(path).history.accepted.sum() >= 2
    _run(ordinary(step_planes=0), path, resume=True)
    assert_contiguous(path, 0.001)
