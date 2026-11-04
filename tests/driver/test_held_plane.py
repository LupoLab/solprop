# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A low-storage run that never resumes can drop its held plane (SEP 22, Q14).

With ``held_plane=False``, a statistics-only run holds one state volume less by the
backend's ledger and by the admission, on both backends; a step abandoned in flight
leaves no terminal checkpoint and says why, a stop between steps checkpoints as ever,
and an output with nowhere to put a checkpoint is the same run whatever the setting
says. The ledgers are exact accounting. The setting moves no number the march computes,
compared exactly on the reference, whose repeated run does the same arithmetic.
"""

import dataclasses
import logging
import math
import threading
from typing import cast

import numpy as np
import pytest

from solprop.backends.driver import StepDriver
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.driver.memory import admit_memory
from solprop.driver.run import run_problem
from solprop.exceptions import DriverError, StopRequested
from solprop.log import NullReporter
from solprop.output import MemoryOutput, open_file
from solprop.output.schema import FieldRetention
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


def built(
    backend: str = "numpy_ref", *, held: bool = True, fields: str = "none"
) -> Simulation:
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
    simulation.settings = dataclasses.replace(
        simulation.settings, held_plane=held, fields=fields
    )
    return simulation


def ledger(simulation: Simulation, output) -> tuple[int, int]:
    """Run on a runner, returning the held row and the volume, in bytes."""
    with simulation.runner() as runner:
        runner.run(simulation._field0.copy(), output=output, progress="none")
        return (
            runner.backend.memory_ledger()["held_plane"],
            16 * math.prod(simulation.translate().shape),
        )


@pytest.mark.parametrize("backend", available())
def test_the_ledger_and_the_admission_drop_one_volume(backend, tmp_path):
    on = built(backend)
    off = built(backend, held=False)
    held, volume = ledger(on, tmp_path / "on.h5")
    assert held == volume
    assert ledger(off, tmp_path / "off.h5") == (0, volume)
    problem = off.translate()
    rule = FieldRetention("none", ()).resolved(3)
    charged = admit_memory(problem, backend, "none", retention=rule).estimate
    spared = admit_memory(
        problem, backend, "none", retention=rule, held_plane=False
    ).estimate
    assert charged.groups["held_plane"] == volume
    assert "held_plane" not in spared.groups
    assert charged.total - spared.total == volume


def test_the_estimate_a_caller_reads_says_the_same():
    """On the reference: it times a trial step, which a native march does not take."""
    on, off = built(), built(held=False)
    volume = 16 * math.prod(off.translate().shape)
    assert on.estimate().memory - off.estimate().memory == volume


@pytest.mark.parametrize("backend", available())
def test_a_file_says_it_has_no_held_state(backend, tmp_path):
    built(backend, held=False).run(tmp_path / "off.h5", progress="none")
    built(backend).run(tmp_path / "on.h5", progress="none")
    with open_file(tmp_path / "off.h5") as stored:
        assert stored.metadata is not None and stored.metadata.backend is not None
        assert stored.metadata.backend.settings["held_plane"] is False
    with open_file(tmp_path / "on.h5") as stored:
        assert stored.metadata is not None and stored.metadata.backend is not None
        assert "held_plane" not in stored.metadata.backend.settings


def test_the_setting_is_outside_the_resume_key_and_moves_no_number(tmp_path):
    on, off = built(fields="all"), built(fields="all", held=False)
    assert on.metadata().configuration_hash() == off.metadata().configuration_hash()
    # Execution settings may change after the translation, as `fields` may.
    on.translate()
    on.settings = dataclasses.replace(on.settings, held_plane=False)
    built().run(tmp_path / "a.h5", progress="none")
    built(held=False).run(tmp_path / "b.h5", progress="none")
    with open_file(tmp_path / "a.h5") as one, open_file(tmp_path / "b.h5") as two:
        assert np.array_equal(one["stats/energy"], two["stats/energy"])


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
        built(held=False).run(path, progress="none")
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
        built(held=False).run(path, progress="none", resume=True)


def test_a_step_abandoned_after_a_kept_plane_still_resumes_from_it(
    monkeypatch, tmp_path
):
    path = tmp_path / "run.h5"
    _abandon_a_step(monkeypatch, after=4 * 5 + 2)
    with pytest.raises(AbandonedError) as caught:
        built(held=False, fields="all").run(path, progress="none")
    assert "from which resume=True continues" in " ".join(caught.value.__notes__)
    monkeypatch.undo()
    built(held=False, fields="all").run(path, progress="none", resume=True)
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

    simulation = built(backend, held=False)
    simulation.run(path, progress=StopAtTheFirstPlane(), stop=flag)
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is not None
        assert stored.terminal_checkpoint.plane.field is not None
    built(backend, held=False).run(path, progress="none", resume=True)
    with open_file(path) as stored:
        assert stored["z"].tolist()[-1] == LENGTH


def test_a_step_abandoned_natively_leaves_no_checkpoint(monkeypatch, tmp_path):
    """The native engine's own exception, the one way it abandons a step in flight."""
    pytest.importorskip("solprop_native")
    path = tmp_path / "run.h5"
    simulation = built("native", held=False)
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
        assert runner.backend.memory_ledger()["held_plane"] == 0
    assert "cannot be resumed" in " ".join(caught.value.__notes__)
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None


@pytest.mark.parametrize("backend", available())
def test_an_output_without_checkpoints_holds_nothing_whatever_the_setting(backend):
    """It would never read the state, so it is neither charged nor held."""
    simulation = built(backend)
    output = _NoCheckpoints()
    held, _ = ledger(simulation, output)
    assert held == 0
    assert output.metadata is not None and output.metadata.backend is not None
    assert output.metadata.backend.settings["held_plane"] is False
    problem = simulation.translate()
    rule = FieldRetention("none", ()).resolved(3)
    groups = admit_memory(
        problem, backend, "none", output=_NoCheckpoints(), retention=rule
    ).estimate.groups
    assert "held_plane" not in groups


@pytest.mark.parametrize("backend", available())
def test_a_later_run_that_holds_nothing_releases_an_earlier_run_s_volume(
    backend, tmp_path
):
    simulation = built(backend)
    volume = 16 * math.prod(simulation.translate().shape)
    with simulation.runner() as runner:
        field = simulation._field0.copy()
        runner.run(field, output=tmp_path / "held.h5", progress="none")
        assert runner.backend.memory_ledger()["held_plane"] == volume
        runner.run(field, output=_NoCheckpoints(), progress="none")
        assert runner.backend.memory_ledger()["held_plane"] == 0
        # And the next run holds again, from a new buffer.
        runner.run(field, output=tmp_path / "again.h5", progress="none")
        assert runner.backend.memory_ledger()["held_plane"] == volume


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
        backend.set_initial(simulation.translate().initial.field0, None)
        run_problem(
            simulation.translate(),
            backend,
            _NoCheckpoints(),
            metadata=simulation.metadata(),
            progress=StopAtTheFirstPlane(),
            fields="none",
            stop=flag,
        )
    assert "no state was written there" in caplog.text
    assert "resume=True continues" not in caplog.text


def test_a_backend_that_has_no_such_setting_runs_as_ever():
    """The driver asks only a backend that has it; one of its own need not."""

    class Plain(NumpyRefBackend):
        set_held_plane = None  # pyright: ignore[reportAssignmentType] - without it

    simulation = built()
    problem = simulation.translate()
    backend = Plain(problem, grid=simulation.grid, modes=simulation.modes)
    backend.set_initial(problem.initial.field0, None)
    output = MemoryOutput()
    run_problem(
        problem,
        backend,
        output,
        metadata=simulation.metadata(),
        progress="none",
        fields="none",
    )
    backend.close()
    assert output.n_saved == 3
