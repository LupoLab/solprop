# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A portable stop: a run's own flag stops it, and the file resumes (SEP 41, S17)."""

import dataclasses
import threading

import numpy as np
import pytest

from driver.test_resume_history import assert_accurate_run, tight_run
from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import TIGHTER
from solprop.backends.registry import available
from solprop.backends.session import DEFAULT_WORK_LIMIT
from solprop.exceptions import StopRequested
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.output.hdf5 import open_file
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, RungeKutta

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


@pytest.mark.parametrize(
    "backend",
    [
        "numpy_ref",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                "native" not in available(), reason="native is not installed"
            ),
        ),
    ],
)
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
    # The reporter hears each advance's attempts as its chunk arrives (SEP 47 5.3),
    # and the flag is polled before the next advance, which runs at most to the next
    # event, here the next saved plane: the run stops between the flag's step and it.
    planes = np.linspace(0.0, LENGTH, N_SAVE)[1:]
    after = 16 * LENGTH / STEPS - 1e-15
    upper = float(min(plane for plane in planes if plane >= after))
    assert 15 * LENGTH / STEPS <= outcome.z <= upper + 1e-15
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


NATIVE = pytest.param(
    "native",
    marks=pytest.mark.skipif(
        "native" not in available(), reason="native is not installed"
    ),
)

MARCHES = {
    "adams_exponential": Multistep(),
    "adams_interaction": Multistep(formulation="interaction"),
    "adams_replay": Multistep(),
    "runge_kutta": RungeKutta(),
    "low_storage": LowStorageRungeKutta(),
}
"""Every march a run can take, keyed by name."""

REPLAY_STEPS = {"adams_replay": 4000, "low_storage": 400}
"""The steps of the marches that take a schedule's: the uniform steps an Adams replay of
this soliton stays stable on, and few enough that the host's per-step work stays small
for a march it steps from Python."""

LENGTHS = {"runge_kutta": 0.02, "low_storage": 0.02}
"""The marches stepped from Python run a tenth of :data:`ONE_PLANE_LENGTH`, m: they stop
at once, and only their resume runs the whole length. On native the Runge-Kutta march
is compiled (SEP 45) and runs the whole length, as the Adams marches do."""

FLAGGED_AT = 20
"""The accepted step whose report sets the flag."""

ONE_PLANE_LENGTH = 0.2
"""The one-plane run's length, m, with an N = 10 soliton: the adaptive Adams marches
take about a thousand steps, far more than a native call's bound past
:data:`FLAGGED_AT`."""

ONE_PLANE_RTOL = 1e-6
"""An adaptive one-plane run's relative tolerance."""


def _length(march, backend):
    """Return `march`'s one-plane run's length on `backend`, m."""
    if backend == "native" and march == "runge_kutta":
        return ONE_PLANE_LENGTH
    return LENGTHS.get(march, ONE_PLANE_LENGTH)


def _one_plane(march, backend, *, rtol=ONE_PLANE_RTOL):
    """Return an N = 10 soliton saved only at its end, with no boundary monitor.

    Nothing but a native call's own bound returns a native Adams march to Python here:
    no plane, statistics row or monitor check falls due before the end of the run.
    """
    length = _length(march, backend)
    grid = EnvelopeGrid.from_limits(
        length, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    schedule = None
    if march in REPLAY_STEPS:
        schedule = FixedSchedule.uniform(
            0.0, length, length / REPLAY_STEPS[march], save_planes=[length]
        )

    def build(build_simulation):
        simulation = build_simulation(
            order=10,
            length=length,
            grid=grid,
            absorber=bare_absorber(grid),
            n_save=2,
            schedule=schedule,
            integrator=MARCHES[march],
            backend=backend,
            rtol=rtol,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, boundary_check=None
        )
        return simulation

    return build


def _steps_after_the_flag(march, backend):
    """Return the most accepted steps a march can report after the flag's report.

    The reporter hears each advance's attempts as the advance's chunk arrives (SEP 47
    section 5.3), and the flag is polled before the next advance, on every backend and
    march: the rest of the chunk whose attempt set the flag reports after it, at most
    one advance's bound of attempts.
    """
    del march, backend
    return DEFAULT_WORK_LIMIT - 1


@pytest.mark.parametrize("backend", ["numpy_ref", NATIVE])
@pytest.mark.parametrize("march", MARCHES)
def test_a_flag_stops_a_one_plane_run_within_its_march_s_bound(
    build_simulation, tmp_path, march, backend
):
    path = tmp_path / "stopped.h5"
    flag = threading.Event()
    reporter = StopAfter(flag, FLAGGED_AT)
    simulation = _one_plane(march, backend)(build_simulation)
    assert simulation.run(path, progress=reporter, stop=flag) is not None
    outcome, saved, _ = _read(path)
    assert outcome is not None
    assert outcome.status == "failed"
    assert "stopped by the caller's stop flag" in (outcome.message or "")
    assert reporter.interrupted_at is not None
    assert isinstance(reporter.interrupted_at[1], StopRequested)
    assert saved == 1  # the entrance alone: the run never reached its plane
    assert 0.0 < outcome.z < _length(march, backend)
    assert reporter.accepted >= FLAGGED_AT
    assert reporter.accepted - FLAGGED_AT <= _steps_after_the_flag(march, backend)


@pytest.mark.medium
@pytest.mark.parametrize("backend", [NATIVE])
@pytest.mark.parametrize("march", MARCHES)
def test_a_stopped_one_plane_run_resumes_to_its_end(
    build_simulation, tmp_path, march, backend
):
    build = _one_plane(march, backend)
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build(build_simulation).run(whole, progress="none")
    flag = threading.Event()
    build(build_simulation).run(part, progress=StopAfter(flag, FLAGGED_AT), stop=flag)
    stopped = _read(part)[0]
    assert stopped is not None
    assert stopped.status == "failed"
    build(build_simulation).run(part, resume=True, progress="none")
    finished, saved, field = _read(part)
    assert finished is not None
    assert finished.status == "complete"
    assert finished.z == _length(march, backend)
    assert saved == 2
    if march in REPLAY_STEPS:
        # A replayed schedule resumed is the uninterrupted computation, to rounding
        # (class R): the same steps from the same stored state.
        reference = _read(whole)[2]
        np.testing.assert_allclose(
            field, reference, rtol=0, atol=1e-12 * np.abs(reference).max()
        )
        return

    def tight():
        return _one_plane(march, backend, rtol=ONE_PLANE_RTOL / TIGHTER)(
            build_simulation
        )

    assert_accurate_run(whole, part, tight_run(tight, tmp_path / "tight.h5"))


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
