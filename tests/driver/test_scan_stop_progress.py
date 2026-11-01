# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A scan's portable stop and its point-by-point progress (SEP 41, S4 and S17)."""

import contextlib
import logging
import multiprocessing as mp
import sys
import threading
from multiprocessing.connection import Connection
from typing import cast

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER
from solprop.driver import _scan_batch
from solprop.driver._scan_run import _collect, _Job, _Options, _ScanEvents, _Work
from solprop.driver.executors import Batch, Local, Queue, Slurm, pool
from solprop.driver.executors.pool import PlaneScalars, Pool, _Channel, _WorkerReporter
from solprop.driver.scan import Scan, run_scan
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError, StopRequested
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter, NullScanReporter, PlaneSummary
from solprop.stepper import FixedSchedule

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.001  # m
STEPS = 40
PLANES = [0.0005, 0.001]  # m, after the entrance
SLOW_STEPS = 5000  # about two seconds on numpy_ref, against a stop within one step


@pytest.fixture(autouse=True)
def _controller_without_history(monkeypatch):
    """Size the pool from the scan alone, not from this test process's past."""
    monkeypatch.setattr(pool, "peak_rss", lambda: 0)


def _fibre(power, steps, n_planes):
    """Return a short soliton fibre on a prescribed schedule, as the scan tests do."""
    planes = list(np.linspace(0.0, LENGTH, n_planes + 1)[1:])
    grid = EnvelopeGrid.from_limits(
        LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return Simulation.from_gnlse(
        0.11,
        LENGTH,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        grid=grid,
        absorber=bare_absorber(grid),
        width=100e-15,
        peak_power=float(power),
        shape="sech",
        n_save=n_planes + 1,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps, save_planes=planes),
        integrator=FIXED_ORDER,
        backend="numpy_ref",
    )


def build_point(point):
    """A forty-step point with two planes after the entrance."""
    return _fibre(point.values["power"], STEPS, len(PLANES))


def slow_point(point):
    """A point long enough that a stop sent after its first plane reaches it running.

    Its first plane comes after a two-hundredth of the run, so the controller hears of
    it, sets the flag and sends the stop while most of the run is still ahead.
    """
    return _fibre(point.values["power"], SLOW_STEPS, 200)


class Recorder(NullScanReporter):
    """Record every scan call, and watch each point's run when asked to."""

    def __init__(self, watch=True):
        self.calls = []
        self.watch = watch
        self.points = {}

    def scan_started(self, n_points):
        self.calls.append(("scan_started", n_points))

    def point_started(self, index, point):
        assert getattr(point, "index", None) == index
        self.calls.append(("point_started", index))

    def point_finished(self, index, status, wall):
        self.calls.append(("point_finished", index, status, wall))

    def scan_finished(self):
        self.calls.append(("scan_finished",))

    def point_reporter(self, index):
        if not self.watch:
            return None
        self.points[index] = PointRecorder()
        return self.points[index]

    def finished(self):
        """Return the (index, status) of each finished point, in order."""
        return [(c[1], c[2]) for c in self.calls if c[0] == "point_finished"]


class PointRecorder(NullReporter):
    """Record one point's run."""

    def __init__(self):
        self.calls = []

    def summary(self, text):
        self.calls.append(("summary", text))

    def started(self, z_start, z_end, n_planes):
        self.calls.append(("started", z_start, z_end, n_planes))

    def step(self, z, dz, accepted):
        self.calls.append(("step", z))

    def plane(self, z, summary):
        self.calls.append(("plane", z, summary))

    def finished(self, total):
        self.calls.append(("finished", total))

    def interrupted(self, z, error):
        self.calls.append(("interrupted", z, error))

    def names(self):
        return [call[0] for call in self.calls if call[0] != "step"]


class StopAtStep(Recorder):
    """Set the flag from inside one point's run, after some of its steps."""

    def __init__(self, flag, index, steps):
        super().__init__()
        self.flag, self.index, self.steps = flag, index, steps

    def point_reporter(self, index):
        watcher = super().point_reporter(index)
        if index == self.index:
            flag, steps = self.flag, self.steps

            class Stopping(PointRecorder):
                def step(self, z, dz, accepted):
                    super().step(z, dz, accepted)
                    if len(self.calls) - 2 >= steps:  # summary and started first
                        flag.set()

            watcher = self.points[index] = Stopping()
        return watcher


class StopAfterPoint(Recorder):
    """Set the flag once a given point has finished."""

    def __init__(self, flag, index):
        super().__init__(watch=False)
        self.flag, self.index = flag, index

    def point_finished(self, index, status, wall):
        super().point_finished(index, status, wall)
        if index == self.index:
            self.flag.set()


class StopAtPlane(Recorder):
    """Set the flag when a point's first plane arrives, relayed from a pool worker."""

    def __init__(self, flag):
        super().__init__()
        self.flag = flag

    def point_reporter(self, index):
        flag = self.flag

        class Stopping(PointRecorder):
            def plane(self, z, summary):
                super().plane(z, summary)
                flag.set()

        self.points[index] = Stopping()
        return self.points[index]


def _assert_scan_order(reporter, n_points):
    """Check the scan's calls: one start, each point started before it finished."""
    calls = reporter.calls
    assert calls[0] == ("scan_started", n_points)
    assert calls[-1] == ("scan_finished",)
    open_points = set()
    for call in calls[1:-1]:
        if call[0] == "point_started":
            assert call[1] not in open_points
            open_points.add(call[1])
        else:
            assert call[1] in open_points
            open_points.remove(call[1])
    assert not open_points


def _assert_watched(watcher, planes, *, finished=True):
    """Check a point's run was reported: started, its planes, then finished."""
    names = watcher.names()
    # The summary block comes before the start, and the resources line after the end.
    assert names[:2] == ["summary", "started"]
    names = [name for name in names if name != "summary"]
    seen = [call for call in watcher.calls if call[0] == "plane"]
    if planes is None:
        assert seen
    else:
        # The plane positions are the schedule's own numbers, passed unchanged.
        assert [call[1] for call in seen] == planes
    for call in seen:
        assert isinstance(call[2], PlaneSummary)
        assert call[2].n_steps > 0
    assert names[-1] == ("finished" if finished else "interrupted")


def test_local_stop_mid_point_is_not_committed_and_a_rerun_computes_it(tmp_path):
    scan = Scan("stop", power=[1.0, 2.0, 3.0])
    path = tmp_path / "scan.h5"
    flag = threading.Event()
    reporter = StopAtStep(flag, index=1, steps=10)
    output = run_scan(
        scan, build_point, output=path, executor=Local(), progress=reporter, stop=flag
    )
    # The finished point is kept, the stopped one is given back, the third was never
    # claimed: a rerun computes both.
    assert output.status.tolist() == ["done", "todo", "todo"]
    assert reporter.finished() == [(0, "done"), (1, "stopped")]
    _assert_scan_order(reporter, 3)
    stopped = reporter.points[1]
    assert stopped.names()[-1] == "interrupted"
    error = next(c for c in stopped.calls if c[0] == "interrupted")[2]
    assert isinstance(error, StopRequested)
    assert error.flag is flag
    # Polled before every trial: the run stops within one step of the flag.
    steps = [call for call in stopped.calls if call[0] == "step"]
    assert len(steps) == 10
    rerun = Recorder()
    again = run_scan(
        scan, build_point, output=path, executor=Local(), progress=rerun, stop=None
    )
    assert again.status.tolist() == ["done", "done", "done"]
    assert rerun.finished() == [(0, "skipped"), (1, "done"), (2, "done")]


def test_a_flag_set_between_local_points_dispatches_no_more(tmp_path):
    scan = Scan("between", power=[1.0, 2.0, 3.0])
    flag = threading.Event()
    reporter = StopAfterPoint(flag, index=0)
    output = run_scan(
        scan,
        build_point,
        output=tmp_path / "scan.h5",
        executor=Local(),
        progress=reporter,
        stop=flag,
    )
    assert output.status.tolist() == ["done", "todo", "todo"]
    assert reporter.finished() == [(0, "done")]
    assert reporter.calls[-1] == ("scan_finished",)


def test_a_flag_set_before_the_scan_dispatches_nothing(tmp_path, caplog):
    flag = threading.Event()
    flag.set()
    reporter = Recorder()
    with caplog.at_level(logging.INFO, logger="solprop.driver._scan_run"):
        output = run_scan(
            Scan("before", power=[1.0, 2.0]),
            build_point,
            output=tmp_path / "scan.h5",
            # No point is claimed, so no worker is spawned.
            executor=Pool(workers=1, pin=False),
            progress=reporter,
            stop=flag,
        )
    assert output.status.tolist() == ["todo", "todo"]
    assert reporter.calls == [("scan_started", 2), ("scan_finished",)]
    assert "stopped by the caller's stop flag" in caplog.text


def test_local_progress_reaches_the_scan_reporter_and_each_point(tmp_path):
    scan = Scan("watch", power=[1.0, 2.0])
    reporter = Recorder()
    run_scan(
        scan,
        build_point,
        output=tmp_path / "scan.h5",
        executor=Local(),
        progress=reporter,
    )
    _assert_scan_order(reporter, 2)
    assert [c[:3] for c in reporter.calls if c[0] == "point_finished"] == [
        ("point_finished", 0, "done"),
        ("point_finished", 1, "done"),
    ]
    assert all(c[3] > 0 for c in reporter.calls if c[0] == "point_finished")
    for index in (0, 1):
        _assert_watched(reporter.points[index], PLANES)
        # In this process the point's reporter also sees every step.
        assert sum(c[0] == "step" for c in reporter.points[index].calls) == STEPS


@pytest.mark.medium
def test_pool_progress_is_relayed_from_its_workers(tmp_path):
    scan = Scan("relay", power=[1.0, 2.0, 3.0])
    reporter = Recorder()
    output = run_scan(
        scan,
        build_point,
        output=tmp_path / "scan.h5",
        executor=Pool(workers=2, pin=False),
        progress=reporter,
    )
    assert output.status.tolist() == ["done", "done", "done"]
    _assert_scan_order(reporter, 3)
    finished = [c for c in reporter.calls if c[0] == "point_finished"]
    assert sorted(c[1] for c in finished) == [0, 1, 2]
    assert all(c[2] == "done" and c[3] > 0 for c in finished)
    for index in range(3):
        watcher = reporter.points[index]
        _assert_watched(watcher, PLANES)
        # A worker sends no step: it is on the hot path.
        assert not any(call[0] == "step" for call in watcher.calls)
        assert all(
            isinstance(call[2], PlaneScalars)
            for call in watcher.calls
            if call[0] == "plane"
        )
        # Every relayed call of a point arrives before its result.
        position = reporter.calls.index(next(c for c in finished if c[1] == index))
        assert reporter.calls[position][0] == "point_finished"


@pytest.mark.medium
def test_pool_stop_message_stops_the_point_in_flight(tmp_path):
    scan = Scan("pool_stop", power=[1.0, 2.0])
    flag = threading.Event()
    reporter = StopAtPlane(flag)
    output = run_scan(
        scan,
        slow_point,
        output=tmp_path / "scan.h5",
        executor=Pool(workers=1, pin=False),
        progress=reporter,
        stop=flag,
    )
    # The worker's run polled the pipe and stopped; nothing more was dispatched.
    assert output.status.tolist() == ["todo", "todo"]
    assert reporter.finished() == [(0, "stopped")]
    _assert_scan_order(reporter, 2)
    watcher = reporter.points[0]
    # The stop reaches the worker a few planes after the first, at most.
    _assert_watched(watcher, None, finished=False)
    _, z, error = next(c for c in watcher.calls if c[0] == "interrupted")
    assert 0 < z < LENGTH / 2
    # The worker's stop stands for the caller's flag.
    assert isinstance(error, StopRequested)
    assert error.flag is flag


def test_batch_stops_between_groups(tmp_path):
    scan = Scan("batch_stop", power=[1.0, 2.0, 3.0])
    flag = threading.Event()
    reporter = StopAfterPoint(flag, index=0)
    output = run_scan(
        scan,
        build_point,
        output=tmp_path / "scan.h5",
        executor=Batch(1),
        progress=reporter,
        stop=flag,
    )
    assert output.status.tolist() == ["done", "todo", "todo"]
    assert reporter.finished() == [(0, "done")]
    # A batch reports a point's start only as its result is collected.
    assert reporter.calls[1:3] == [
        ("point_started", 0),
        ("point_finished", 0, "done", reporter.calls[2][3]),
    ]
    rerun = run_scan(scan, build_point, output=tmp_path / "scan.h5", executor=Batch(1))
    assert rerun.status.tolist() == ["done", "done", "done"]


def test_batch_gives_back_a_claimed_group_it_did_not_run(tmp_path):
    flag = threading.Event()

    def build(point):
        # The second member is built after the first: the flag is set before the
        # group they form would run.
        if point.index == 1:
            flag.set()
        return build_point(point)

    reporter = Recorder(watch=False)
    output = run_scan(
        Scan("batch_group", power=[1.0, 2.0]),
        build,
        output=tmp_path / "scan.h5",
        executor=Batch(2),
        progress=reporter,
        stop=flag,
    )
    assert output.status.tolist() == ["todo", "todo"]
    assert reporter.finished() == [(0, "stopped"), (1, "stopped")]


def test_batch_stops_between_the_admitted_parts_of_a_group(tmp_path, monkeypatch):
    # Admission splits a compatible group when memory holds fewer members at once.
    monkeypatch.setattr(_scan_batch, "_admit", lambda group, work, executor: 1)
    flag = threading.Event()
    reporter = StopAfterPoint(flag, index=0)
    output = run_scan(
        Scan("batch_parts", power=[1.0, 2.0]),
        build_point,
        output=tmp_path / "scan.h5",
        executor=Batch(2),
        progress=reporter,
        stop=flag,
    )
    assert output.status.tolist() == ["done", "todo"]
    assert reporter.finished() == [(0, "done"), (1, "stopped")]


@pytest.mark.skipif(
    sys.platform == "win32", reason="the Queue and Slurm executors are POSIX only"
)
@pytest.mark.parametrize("name", ["queue", "slurm"])
def test_durable_executors_refuse_a_stop_flag(tmp_path, name):
    executor = Queue(tmp_path / "scan.queue") if name == "queue" else Slurm("scan.py")
    with pytest.raises(DriverError, match="scancel"):
        run_scan(
            Scan("durable", power=[1.0]),
            build_point,
            output=tmp_path / "scan.h5",
            executor=executor,
            stop=threading.Event(),
        )
    # Refused before the queue or the output exists.
    assert not (tmp_path / "scan.queue").exists()
    assert not (tmp_path / "scan.h5").exists()


@pytest.mark.parametrize("choice", ["log", "none"])
def test_named_progress_choices_still_work(tmp_path, caplog, choice):
    with caplog.at_level(logging.INFO, logger="solprop.driver._scan_run"):
        output = run_scan(
            Scan("named", power=[1.0]),
            build_point,
            output=tmp_path / "scan.h5",
            progress=choice,
        )
    assert output.status.tolist() == ["done"]
    assert ("scan point 0: done" in caplog.text) is (choice == "log")


def test_a_scan_reporter_replaces_the_display_lines(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="solprop.driver._scan_run"):
        run_scan(
            Scan("quiet", power=[1.0]),
            build_point,
            output=tmp_path / "scan.h5",
            progress=Recorder(watch=False),
        )
    assert "scan point 0" not in caplog.text


def test_progress_must_be_a_choice_or_a_scan_reporter(tmp_path):
    # The test session's runtime type check refuses it first; without it, run_scan.
    with pytest.raises((DriverError, TypeCheckError), match="ScanReporter"):
        run_scan(
            Scan("bad", power=[1.0]),
            build_point,
            output=tmp_path / "scan.h5",
            progress=NullReporter(),  # pyright: ignore[reportArgumentType]
        )
    assert not (tmp_path / "scan.h5").exists()


def test_the_worker_channel_keeps_commands_while_it_polls_for_a_stop():
    controller, worker = mp.Pipe()
    channel = _Channel(cast(Connection, worker))
    assert not channel.is_set()
    controller.send(("run", 1))
    controller.send("stop")
    controller.send(None)
    # Polling reads ahead without consuming the commands, and a stop stays set.
    assert channel.is_set()
    assert channel.is_set()
    assert channel.command() == ("run", 1)
    assert channel.command() is None


def test_the_worker_reporter_sends_scalars_never_a_field():
    controller, worker = mp.Pipe()
    reporter = _WorkerReporter(_Channel(cast(Connection, worker)))
    summary = PlaneScalars(8, 1, 0.5, 0.25)
    reporter.summary("block")
    reporter.started(0.0, 1.0, 3)
    reporter.step(0.1, 0.1, True)
    reporter.plane(np.float64(0.5), summary)
    reporter.interrupted(0.5, StopRequested(threading.Event()))
    reporter.finished(2.0)
    received = []
    while controller.poll():
        received.append(controller.recv())
    assert received == [
        ("progress", "summary", ("block",)),
        ("progress", "started", (0.0, 1.0, 3)),
        ("progress", "plane", (0.5, summary)),
        (
            "progress",
            "interrupted",
            (0.5, True, "StopRequested: stopped by the caller's stop flag"),
        ),
        ("progress", "finished", (2.0,)),
    ]
    assert type(received[1][2][0]) is float


class PlainScanReporter:
    """A scan reporter written to the four calls alone, which watches no point."""

    def __init__(self):
        self.calls = []

    def scan_started(self, n_points):
        self.calls.append("scan_started")

    def point_started(self, index, point):
        self.calls.append("point_started")

    def point_finished(self, index, status, wall):
        self.calls.append(status)

    def scan_finished(self):
        self.calls.append("scan_finished")


def test_a_reporter_without_point_reporter_watches_no_point(tmp_path):
    reporter = PlainScanReporter()
    run_scan(
        Scan("plain", power=[1.0]),
        build_point,
        output=tmp_path / "scan.h5",
        progress=reporter,
    )
    assert reporter.calls == ["scan_started", "point_started", "done", "scan_finished"]


def test_the_bar_choice_draws_one_outer_bar(tmp_path, capsys):
    output = run_scan(
        Scan("bar", power=[1.0]),
        build_point,
        output=tmp_path / "scan.h5",
        progress="bar",
    )
    assert output.status.tolist() == ["done"]
    assert "1/1" in capsys.readouterr().err


class Plain:
    """A reporter with the four basic calls only: no summary and no interruption."""

    def __init__(self):
        self.calls = []

    def started(self, z_start, z_end, n_planes):
        self.calls.append("started")

    def step(self, z, dz, accepted):
        self.calls.append("step")

    def plane(self, z, summary):
        self.calls.append("plane")

    def finished(self, total):
        self.calls.append("finished")


def test_relayed_calls_reach_only_what_a_point_reporter_implements():
    flag = threading.Event()
    full, plain = PointRecorder(), Plain()
    reporters = {0: full, 1: plain, 2: None}

    class Points(NullScanReporter):
        def point_reporter(self, index):
            return reporters[index]

    scan = Scan("relay", power=[1.0, 2.0, 3.0])
    events = _ScanEvents(Points(), scan)
    for index in range(3):
        events.begin(scan.point(index))
    summary = PlaneScalars(4, 0, 0.5, 0.1)
    for index in range(3):
        for call, args in [
            ("summary", ("block",)),
            ("started", (0.0, 1.0, 2)),
            ("plane", (0.5, summary)),
            ("interrupted", (0.5, True, "StopRequested: stopped")),
            ("interrupted", (0.5, False, "KeyboardInterrupt: ")),
            ("finished", (1.0,)),
        ]:
            events.relay(index, call, args, flag)
    assert plain.calls == ["started", "plane", "finished"]
    assert full.names() == [
        "summary",
        "started",
        "plane",
        "interrupted",
        "interrupted",
        "finished",
    ]
    stopped, other = (c[2] for c in full.calls if c[0] == "interrupted")
    assert isinstance(stopped, StopRequested)
    assert stopped.flag is flag
    # Anything else that ended a worker's run is named, not raised again.
    assert isinstance(other, DriverError)
    assert "KeyboardInterrupt" in str(other)


def test_work_outside_the_controller_reports_nothing_and_a_set_flag_stops_it():
    work = _Work(build_point, _Options((), "none", (), "all"), "outside")
    job = _Job(Scan("outside", power=[1.0]).point(0))
    # A worker's copy has no events: the controller's hooks do nothing there.
    work.dispatched(job)
    work.relay(job, "plane", (0.5, PlaneScalars(1, 0, 0.1, 0.1)))
    flag = threading.Event()
    flag.set()
    result = work.run_point(job, flag, None)
    assert result.status == "stopped"
    assert "rerun computes it" in result.error
    # A stopped point never claimed is simply not committed.
    _collect(result, None, None)


@pytest.mark.parametrize("arrives", ["before", "during"])
@pytest.mark.parametrize("watched", [True, False])
def test_a_worker_runs_controlled_work_under_its_pipe(arrives, watched):
    class Controlled:
        stop = None

        def __init__(self):
            self.watched = watched

        def run_point(self, point, stop, reporter):
            if reporter is not None:
                reporter.summary("block")
                reporter.plane(0.5, PlaneScalars(2, 0, 0.3, 0.1))
            return point, stop.is_set()

        def dispatched(self, point):
            raise AssertionError("the controller's hook, not the worker's")

        def relay(self, point, call, args):
            raise AssertionError("the controller's hook, not the worker's")

        def __call__(self, point):
            raise AssertionError("controlled work runs through run_point")

    parent, child = mp.Pipe()
    messages = (
        [("run", 7), "stop", None]
        if arrives == "during"
        else ["stop", ("run", 7), None]
    )
    for message in messages:
        parent.send(message)
    pool._serve(cast(Connection, child), Controlled(), ())
    received = []
    # The worker closed its end: read until the pipe says so.
    with contextlib.suppress(EOFError):
        while parent.poll():
            received.append(parent.recv())
    progress = (
        [
            ("progress", "summary", ("block",)),
            ("progress", "plane", (0.5, PlaneScalars(2, 0, 0.3, 0.1))),
        ]
        if watched
        else []
    )
    # A stop sent while the point runs is read by its poll, one sent before by the
    # wait for the command: either way the run sees it, and the commands survive.
    assert received == [*progress, (True, (7, True))]
    parent.close()
