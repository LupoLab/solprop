# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A portable stop: a run's own flag stops it, and the file resumes (SEP 41, S17)."""

import threading

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop._interrupts import StopRequested
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.output.hdf5 import open_file
from solprop.stepper import FixedSchedule, RungeKutta

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.01  # m, a sixteenth of the soliton period
N_SAVE = 5  # the entrance and four planes
STEPS = 40


class StopAfter(NullReporter):
    """A reporter that sets a flag after some accepted steps, from inside the run."""

    def __init__(self, flag, steps):
        self.flag = flag
        self.steps = steps
        self.accepted = 0
        self.interrupted_at = None

    def step(self, z, dz, accepted):
        self.accepted += accepted
        if self.accepted >= self.steps:
            self.flag.set()

    def interrupted(self, z, error):
        self.interrupted_at = (z, error)


def _read(path):
    """Return a closed file's outcome, saved-plane count and last field."""
    with open_file(path) as handler:
        return handler.outcome, handler.n_saved, np.asarray(handler["field"][-1])


def _scheduled(build_simulation, backend="numpy_ref"):
    """The soliton fibre on a fixed schedule: a resume takes the same steps."""
    planes = np.linspace(0.0, LENGTH, N_SAVE)[1:]
    grid = EnvelopeGrid.from_limits(
        LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return build_simulation(
        length=LENGTH,
        grid=grid,
        absorber=bare_absorber(grid),
        n_save=N_SAVE,
        schedule=FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / STEPS, save_planes=list(planes)
        ),
        integrator=RungeKutta(),
        backend=backend,
    )


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_set_flag_stops_the_run_which_returns_and_resumes(
    build_simulation, tmp_path, backend
):
    path = tmp_path / "stopped.h5"
    flag = threading.Event()
    reporter = StopAfter(flag, 15)
    simulation = _scheduled(build_simulation, backend)
    handler = simulation.run(path, progress=reporter, stop=flag)
    assert handler is not None
    outcome, _, _ = _read(path)
    assert outcome is not None
    assert outcome.status == "failed"
    assert "stopped by the caller's stop flag" in (outcome.message or "")
    # The flag is polled before each trial: the run stops within one step of it.
    assert 15 * LENGTH / STEPS <= outcome.z <= 16 * LENGTH / STEPS + 1e-15
    assert reporter.interrupted_at is not None
    assert isinstance(reporter.interrupted_at[1], StopRequested)
    _scheduled(build_simulation, backend).run(path, resume=True)
    _scheduled(build_simulation, backend).run(tmp_path / "whole.h5")
    finished, _, field = _read(path)
    _, _, reference = _read(tmp_path / "whole.h5")
    assert finished is not None
    assert finished.status == "complete"
    # A replayed schedule resumed is the uninterrupted computation, to rounding
    # (class R): the same steps from the same stored state.
    np.testing.assert_allclose(
        field, reference, rtol=0, atol=1e-12 * np.abs(reference).max()
    )


def test_a_flag_set_before_the_run_stops_it_before_its_first_step(
    build_simulation, tmp_path
):
    flag = threading.Event()
    flag.set()
    path = tmp_path / "entrance.h5"
    _scheduled(build_simulation).run(path, stop=flag)
    outcome, saved, _ = _read(path)
    assert saved == 1
    assert outcome is not None
    assert outcome.z == 0.0
    _scheduled(build_simulation).run(path, resume=True)
    assert _read(path)[1] == N_SAVE


def test_a_run_on_a_worker_thread_stops_and_another_run_does_not(
    build_simulation, tmp_path
):
    flags = {name: threading.Event() for name in ("stopped", "free")}
    reporters = {
        "stopped": StopAfter(flags["stopped"], 5),
        "free": NullReporter(),
    }
    errors = []

    def work(name):
        try:
            _scheduled(build_simulation).run(
                tmp_path / f"{name}.h5", progress=reporters[name], stop=flags[name]
            )
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=work, args=(name,)) for name in flags]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    stopped = _read(tmp_path / "stopped.h5")[0]
    free = _read(tmp_path / "free.h5")[0]
    assert stopped is not None and stopped.status == "failed"
    assert free is not None and free.status == "complete"


def test_another_run_s_flag_is_not_caught_here(build_simulation, tmp_path):
    other = threading.Event()

    class Foreign(NullReporter):
        def step(self, z, dz, accepted):
            raise StopRequested(other)

    with pytest.raises(StopRequested):
        _scheduled(build_simulation).run(
            tmp_path / "foreign.h5", progress=Foreign(), stop=threading.Event()
        )
