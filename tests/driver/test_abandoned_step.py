# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a low-storage step abandoned in flight leaves behind (ADR-067).

The march updates its state in place, so a step that dies partway leaves neither of its
ends. It writes no terminal checkpoint and says why, the file resumes from a last plane
that kept its field and from no other, and a stop between steps checkpoints as ever.
"""

import dataclasses
import logging
import threading
from typing import cast

import numpy as np
import pytest

from march_bounds import assert_matched, march_bound
from solprop.backends.driver import StepDriver
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.driver.run import run_problem
from solprop.exceptions import DriverError, StopRequested
from solprop.log import NullReporter
from solprop.output import MemoryOutput, SnapshotController, open_file, resume
from solprop.stepper import FixedSchedule, LowStorageRungeKutta

# A soliton in a tight window, without a collar: its edges are not what is tested here.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.02
STEPS = 8


class AbandonedError(RuntimeError):
    """What stops a step in flight."""


class _NoCheckpoints(MemoryOutput):
    """A store that declares no checkpoint: `isinstance` looks at the class."""

    record_outcome = None  # pyright: ignore[reportAssignmentType] - a store without it


def built(backend: str = "numpy_ref", *, fields: str = "none") -> Simulation:
    """Return a short replayed soliton, keeping no field by default."""
    schedule = FixedSchedule.uniform(
        0.0, LENGTH, LENGTH / STEPS, save_planes=[LENGTH / 2]
    )
    simulation = Simulation.from_gnlse(
        0.11,
        LENGTH,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        save_planes=[LENGTH / 2, LENGTH],
        schedule=schedule,
        integrator=LowStorageRungeKutta(),
        backend=backend,
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        absorber=False,
        noise=None,
    )
    simulation.settings = dataclasses.replace(simulation.settings, fields=fields)
    return simulation


def _abandon_a_step(monkeypatch, *, after: int) -> None:
    """Raise out of the reference's step `after` source evaluations from the start."""
    original = NumpyRefBackend._adams_lab_source
    count = [0]

    def evaluate(self, lab, position, **keywords):
        count[0] += 1
        if count[0] == after:
            raise AbandonedError
        return original(self, lab, position, **keywords)

    monkeypatch.setattr(NumpyRefBackend, "_adams_lab_source", evaluate)


def test_a_step_abandoned_in_flight_leaves_no_checkpoint_and_says_why(
    monkeypatch, tmp_path
):
    """The plane before it kept no field, so nothing can resume this file, by name."""
    path = tmp_path / "run.h5"
    _abandon_a_step(monkeypatch, after=4 * 5 + 2)
    with pytest.raises(AbandonedError) as caught:
        built().run(path, progress="none")
    notes = " ".join(caught.value.__notes__)
    assert "no terminal checkpoint is written" in notes
    assert "abandoned in flight" in notes
    assert "cannot be resumed" in notes
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None
        assert stored["z"].tolist() == [0.0, LENGTH / 2]
        assert stored.outcome is not None and stored.outcome.status == "failed"
        assert "no terminal checkpoint" in stored.outcome.message
    monkeypatch.undo()
    with pytest.raises(DriverError, match="no field"):
        built().run(path, progress="none", resume=True)


def test_a_step_abandoned_after_a_kept_plane_still_resumes_from_it(
    monkeypatch, tmp_path
):
    path = tmp_path / "run.h5"
    _abandon_a_step(monkeypatch, after=4 * 5 + 2)
    with pytest.raises(AbandonedError) as caught:
        built(fields="all").run(path, progress="none")
    assert "from which resume=True continues" in " ".join(caught.value.__notes__)
    monkeypatch.undo()
    built(fields="all").run(path, progress="none", resume=True)
    with open_file(path) as stored:
        assert stored["z"].tolist() == [0.0, LENGTH / 2, LENGTH]


@pytest.mark.parametrize("backend", available())
def test_a_stop_between_steps_checkpoints_as_ever(backend, tmp_path):
    """The flag is polled between steps, where the state is a step's end."""
    path = tmp_path / "run.h5"
    flag = threading.Event()

    class StopAtTheFirstPlane(NullReporter):
        def plane(self, z, summary) -> None:
            flag.set()

    simulation = built(backend)
    simulation.run(path, progress=StopAtTheFirstPlane(), stop=flag)
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is not None
        assert stored.terminal_checkpoint.plane.field is not None
    built(backend).run(path, progress="none", resume=True)
    with open_file(path) as stored:
        assert stored["z"].tolist()[-1] == LENGTH


def test_a_step_abandoned_natively_leaves_no_checkpoint(monkeypatch, tmp_path):
    """The native engine's own exception, the one way it abandons a step in flight."""
    pytest.importorskip("solprop_native")
    path = tmp_path / "run.h5"
    simulation = built("native")
    with simulation.runner() as runner:
        march = cast("StepDriver", runner.backend)._low_storage
        assert march is not None
        engine = march.engine
        original = engine.step
        calls = []

        def failing(z, dz):
            calls.append(z)
            squared = original(z, dz)
            if len(calls) == STEPS // 2 + 1:
                raise AbandonedError
            return squared

        monkeypatch.setattr(engine, "step", failing)
        with pytest.raises(AbandonedError) as caught:
            runner.run(simulation._field0.copy(), output=path, progress="none")
    assert "cannot be resumed" in " ".join(caught.value.__notes__)
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None


def test_a_stop_without_a_checkpoint_does_not_say_one_was_written(caplog):
    flag = threading.Event()

    class StopAtTheFirstPlane(NullReporter):
        def plane(self, z, summary) -> None:
            flag.set()

    simulation = built()
    with (
        caplog.at_level(logging.INFO, logger="solprop.driver.run"),
        pytest.raises(StopRequested),
        simulation.runner() as runner,
    ):
        backend = runner.backend
        run_problem(
            simulation.translate(),
            backend,
            _NoCheckpoints(),
            start=Start.entrance(simulation.translate()),
            metadata=simulation.metadata(),
            progress=StopAtTheFirstPlane(),
            fields="none",
            stop=flag,
        )
    assert "no state was written there" in caplog.text
    assert "resume=True continues" not in caplog.text


def test_a_step_abandoned_after_a_running_checkpoint_falls_back_to_it(
    monkeypatch, tmp_path
):
    """SEP 47 section 5.5: the latest running checkpoint, not the last saved plane.

    A snapshot due before the first step is serviced at its end, between the launch
    and the first plane, as a running checkpoint; the third step is abandoned.
    """
    whole, path = tmp_path / "whole.h5", tmp_path / "run.h5"
    built(fields="all").run(whole, progress="none")
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    _abandon_a_step(monkeypatch, after=4 * 2 + 2)
    with pytest.raises(AbandonedError) as caught:
        built(fields="all").run(path, progress="none", snapshots=controller)
    notes = " ".join(caught.value.__notes__)
    assert "falls back to its latest checkpoint" in notes
    with open_file(path) as stored:
        checkpoint = stored.terminal_checkpoint
        assert checkpoint is not None and checkpoint.plane.z == LENGTH / STEPS
        assert stored.outcome is not None and stored.outcome.status == "failed"
        assert stored.outcome.z == checkpoint.plane.z
        assert stored["z"].tolist() == [0.0]
    monkeypatch.undo()
    built(fields="all").run(path, progress="none", resume=True)
    # The schedule is prescribed: its steps agree exactly, once each, and the field to
    # the matched march's rounding budget (SEP 25).
    for name in ("z", "h", "accepted"):
        assert np.array_equal(
            getattr(resume(path).history, name), getattr(resume(whole).history, name)
        ), name
    with open_file(path) as stored, open_file(whole) as reference:
        assert stored.terminal_checkpoint is None
        assert stored["z"].tolist() == reference["z"].tolist()
        assert_matched(
            np.asarray(stored["field"]),
            np.asarray(reference["field"]),
            march_bound(*_translated(built())),
            what="field",
        )


def _translated(simulation):
    """Return a simulation's problem and its run's request."""
    return simulation.translate(), simulation._run_request()
