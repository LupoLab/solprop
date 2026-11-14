# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's record: its events, summary, job and stop file, read back by anyone."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from solprop import _durable
from solprop.backends.session import StopToken
from solprop.service import record
from solprop.service.messages import Event, ServiceError
from solprop.service.record import (
    EVENTS,
    JOB,
    EventLog,
    Job,
    alive,
    clear_stop,
    process_stamp,
    read_events,
    read_job,
    read_status,
    request_stop,
    stop_requested,
    watch_stop,
    write_job,
)


def test_rows_and_the_ending_are_kept_and_read_from_where_a_reader_was(tmp_path):
    log = EventLog(tmp_path, "r1")
    log.put(Event("r1", "axes", {"x": np.arange(3.0)}))
    log.put(Event("r1", "row", {"index": 0, "density": np.ones(3)}))
    log.close()
    first, offset = read_events(tmp_path)
    assert [event.kind for event in first] == ["axes", "row"]
    assert np.array_equal(first[1].data["density"], np.ones(3))

    more = EventLog(tmp_path, "r1")  # a continued run appends
    more.put(Event("r1", "stopped", {"path": "run.h5", "detail": "a traceback"}))
    more.close()
    later, end = read_events(tmp_path, offset)
    assert [event.kind for event in later] == ["stopped"] and end > offset
    status = read_status(tmp_path)
    assert status is not None
    assert status["ending"] == {"kind": "stopped", "data": {"path": "run.h5"}}


def test_a_frame_still_being_written_waits_for_the_next_read(tmp_path):
    log = EventLog(tmp_path, "r1")
    log.put(Event("r1", "row", {"index": 0}))
    log.close()
    whole = (tmp_path / EVENTS).read_bytes()
    with (tmp_path / EVENTS).open("ab") as stream:
        stream.write(whole[:9])
    events, offset = read_events(tmp_path)
    assert len(events) == 1 and offset == len(whole)
    assert read_events(tmp_path / "missing", 5) == ([], 5)


def test_a_record_that_holds_something_else_is_refused(tmp_path):
    from solprop.service.frames import encode  # noqa: PLC0415 - only here

    (tmp_path / EVENTS).write_bytes(encode([1, 2]))
    with pytest.raises(ServiceError, match="other than a run's events"):
        read_events(tmp_path)


def test_progress_is_kept_on_a_period_and_the_summary_has_the_latest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(record, "RECORD_PROGRESS_PERIOD", 3600.0)
    log = EventLog(tmp_path, "r1")
    for z in (0.1, 0.2, 0.3):
        log.put(Event("r1", "progress", {"z": z}))
    log.put(Event("r1", "started", {"z_end": 1.0}))
    log.put(Event("r1", "scan_started", {"points": 4}))
    log.put(Event("r1", "point", {"index": 0, "status": None}))
    log.put(Event("r1", "point", {"index": 0, "status": "done", "done": 1}))
    log.put(Event("r1", "row", {"index": 0}))
    log.close()
    kinds = [event.kind for event in read_events(tmp_path)[0]]
    assert kinds.count("progress") == 1
    status = read_status(tmp_path)
    assert status is not None
    assert status["progress"] == {"z": 0.3} and status["started"] == {"z_end": 1.0}
    assert (status["points"], status["done"], status["rows"]) == (4, 1, 1)


def test_a_quiet_run_still_beats(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "HEARTBEAT", 0.05)
    log = EventLog(tmp_path, "r1")
    time.sleep(0.3)
    status = read_status(tmp_path)
    log.close()
    assert status is not None and status["run_id"] == "r1"


def test_an_event_that_cannot_be_written_costs_the_record_not_the_run(tmp_path):
    log = EventLog(tmp_path, "r1")
    log.put(Event("r1", "row", {"bad": {1, 2}}))  # not plain data
    log.put(Event("r1", "finished", {}))
    log.close()
    assert [event.kind for event in read_events(tmp_path)[0]] == ["finished"]


def test_the_job_record_round_trips_and_a_torn_one_reads_as_none(tmp_path):
    assert read_job(tmp_path) is None
    job = Job.here("r1", "direct", threads=4)
    write_job(tmp_path, job)
    assert read_job(tmp_path) == job
    (tmp_path / JOB).write_text('{"run_id": "r1"', encoding="utf-8")
    assert read_job(tmp_path) is None
    (tmp_path / JOB).write_text('{"run_id": "r1", "other": 1}', encoding="utf-8")
    assert read_job(tmp_path) is None
    (tmp_path / JOB).write_text("[1]", encoding="utf-8")
    assert read_job(tmp_path) is None


def _heard(token: StopToken) -> tuple[threading.Event, list[str]]:
    """Attach a listener to `token`, as a session does; say when and where it fired."""
    heard, threads = threading.Event(), []

    def listener() -> None:
        threads.append(threading.current_thread().name)
        heard.set()

    token.attach(listener)
    return heard, threads


def test_the_stop_file_asks_and_is_watched(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "STOP_PERIOD", 0.01)
    stop, done = StopToken(), threading.Event()
    heard, threads = _heard(stop)
    thread = watch_stop(tmp_path, stop, done)
    assert not stop_requested(tmp_path)
    request_stop(tmp_path)
    # The watching thread sets the token, and a session's listener hears it there.
    assert heard.wait(5.0) and stop.is_set() and stop_requested(tmp_path)
    assert threads == [thread.name]
    thread.join(5.0)
    clear_stop(tmp_path)
    assert not stop_requested(tmp_path)
    ended = StopToken()
    quiet = watch_stop(tmp_path, ended, done)
    done.set()
    quiet.join(5.0)
    assert not ended.is_set()


def test_a_process_is_alive_only_while_it_is_the_one_recorded():
    assert alive(os.getpid(), process_stamp(os.getpid()))
    assert alive(os.getpid(), None)
    assert not alive(0, None)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    stamp = process_stamp(child.pid)
    child.wait()
    assert not alive(child.pid, stamp or "a start long ago")
    if stamp is not None:
        assert not alive(os.getpid(), stamp + "0")


def test_the_record_s_files_live_in_the_run_s_folder(tmp_path):
    log = EventLog(tmp_path, "r1")
    log.close()
    assert sorted(path.name for path in Path(tmp_path).iterdir()) == [
        "events.bin",
        "status.json",
    ]


def test_a_continued_run_sets_its_earlier_records_aside(tmp_path):
    from solprop.service.record import STATUS, set_aside  # noqa: PLC0415

    for _ in range(2):
        (tmp_path / EVENTS).write_bytes(b"x")
        (tmp_path / STATUS).write_text("{}", encoding="utf-8")
        request_stop(tmp_path)
        set_aside(tmp_path)
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["events-1.bin", "events-2.bin"]


def test_a_run_tied_to_a_window_stops_when_the_window_goes(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "STOP_PERIOD", 0.01)
    monkeypatch.setattr(record, "OWNER_PERIOD", 0.02)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    stamp = process_stamp(child.pid)
    child.wait()
    stop, done = StopToken(), threading.Event()
    heard, _ = _heard(stop)
    watch_stop(tmp_path, stop, done, owner=(child.pid, stamp or "gone"))
    assert heard.wait(5.0) and stop.is_set() and stop_requested(tmp_path)
    clear_stop(tmp_path)
    alive_stop, alive_done = StopToken(), threading.Event()
    me = (os.getpid(), process_stamp(os.getpid()))
    thread = watch_stop(tmp_path, alive_stop, alive_done, owner=me)
    time.sleep(0.2)
    assert not alive_stop.is_set()  # its window lives: the run goes on
    alive_done.set()
    thread.join(5.0)


def test_a_job_record_that_cannot_be_made_costs_nothing_else(tmp_path):
    def broken() -> Job:
        raise OSError("no ps")

    log = EventLog(tmp_path, "r1", job=broken)
    log.put(Event("r1", "finished", {}))
    log.close()
    assert log.job is None and read_job(tmp_path) is None
    assert [event.kind for event in read_events(tmp_path)[0]] == ["finished"]


def test_a_summary_held_open_is_replaced_once_let_go(tmp_path, monkeypatch):
    from solprop.service.record import write_json  # noqa: PLC0415 - only here

    real = Path.replace
    refusals = iter([True, True, False])

    def held(self, target):
        if next(refusals, False):
            raise PermissionError("in use")  # as Windows says of an open file
        return real(self, target)

    monkeypatch.setattr(Path, "replace", held)
    write_json(tmp_path / "s.json", {"a": 1})
    assert read_status(tmp_path) is None and (tmp_path / "s.json").exists()
    monkeypatch.setattr(_durable, "REPLACE_TRIES", 2)
    refusals = iter([True, True])
    with pytest.raises(PermissionError):
        write_json(tmp_path / "t.json", {"a": 1})
    assert not list(tmp_path.glob(".t.json.*"))  # its temporary file is gone


def test_a_run_continued_the_moment_its_ending_is_recorded_finds_its_record_let_go(
    tmp_path, monkeypatch
):
    from solprop.service.record import STATUS, set_aside  # noqa: PLC0415

    real_replace = Path.replace
    real_write_json = record.write_json
    log: EventLog | None = None
    failures: list[BaseException] = []

    def windows_replace(self, target):
        # Windows refuses to rename a file another handle holds open.
        if self.name == EVENTS and log is not None and not log._stream.closed:
            raise PermissionError("in use")
        return real_replace(self, target)

    def write_json(path, value):
        real_write_json(path, value)
        if path.name == STATUS and "ending" in value:
            # The harshest schedule: a window continues the run as soon as the ending
            # is recorded, before the old writer does anything more.
            try:
                job = read_job(tmp_path)
                assert job is not None and job.ended is not None
                set_aside(tmp_path)
            except BaseException as error:
                failures.append(error)

    monkeypatch.setattr(Path, "replace", windows_replace)
    monkeypatch.setattr(record, "write_json", write_json)
    log = EventLog(tmp_path, "r1", job=lambda: Job.here("r1", "direct"))
    log.put(Event("r1", "row", {"index": 0}))
    log.put(Event("r1", "stopped", {"path": "run.h5"}))
    log.put(Event("r1", "log", {"message": "after the ending"}))
    log.close()
    assert failures == []
    # Every event once, in the record set aside; nothing rewritten after it.
    assert not (tmp_path / EVENTS).exists() and not (tmp_path / STATUS).exists()
    (tmp_path / "events-1.bin").replace(tmp_path / EVENTS)
    assert [e.kind for e in read_events(tmp_path)[0]] == ["row", "stopped"]
