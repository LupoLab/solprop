# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify durable queue ownership and collection on a local filesystem."""

import dataclasses
import functools
import json
import multiprocessing
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from driver.test_scans import assert_same_march, build_point
from solprop.driver.executors import queue as module
from solprop.driver.executors._queue_state import (
    atomic_json,
    owner,
    state,
)
from solprop.driver.executors._request import ScanRequest
from solprop.driver.executors.pool import Pool, peak_rss
from solprop.driver.executors.queue import Queue, _Heartbeat, _QueueWork, collect
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import DriverError, OutputError
from solprop.output.hdf5 import _dataset
from solprop.output.scan import PointResult
from solprop.output.schema import _example_metadata

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the Queue executor is POSIX only and refuses Windows",
)


@dataclasses.dataclass(frozen=True)
class Work:
    def probe(self, point):
        return {"baseline_rss": peak_rss(), "rss": peak_rss(), "ledger": 0}

    def __call__(self, point):
        metadata = dataclasses.replace(
            _example_metadata(), configuration={"index": point.index}
        )
        return PointResult(
            point.index,
            "done",
            metadata.configuration_hash(),
            metadata.to_json(),
            {"z": np.array([0.0, 1.0]), "stats/energy": np.array([1.0, 1.0])},
        )


def prepare(tmp_path, *, size=2):
    scan = Scan("queue", x=np.arange(size))
    queue = Queue(tmp_path / "queue", pool=Pool(workers=1))
    queue.prepare(ScanRequest(scan, {}, tmp_path / "scan.h5", Work()))
    return scan, queue


def claim(queue, point, holder=None):
    holder = owner() if holder is None else holder
    heartbeat = _Heartbeat(Path(queue.path), holder["controller"], queue.stale_age)
    return next(queue._claims([point], holder, heartbeat))


def publish(queue, claimed):
    return _QueueWork(Path(queue.path), Work())(claimed)


@pytest.mark.parametrize(
    "options",
    [
        {"stale_age": 0.0},
        {"stale_age": float("nan")},
        {"chunk": (1, 1)},
        {"chunk": (0, 0)},
    ],
)
def test_queue_refuses_invalid_lease_and_chunk(options):
    with pytest.raises(DriverError):
        Queue(None, **options)


def test_queue_requires_preparation_and_indexed_work():
    with pytest.raises(DriverError, match="prepared"):
        list(Queue(None).run([], abs))
    with pytest.raises(DriverError, match="ScanPoint"):
        module._point(object())


def test_queue_builder_key_includes_partial_arguments():
    assert module._builder_key(functools.partial(pow, 2)) != module._builder_key(
        functools.partial(pow, 3)
    )
    assert module._builder_key(abs) == module._builder_key(abs)


def test_queue_work_requires_probe_and_result_contract(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    claimed = claim(queue, scan.point(0))
    assert queue.path is not None
    assert _QueueWork(Path(queue.path), Work()).probe(claimed)["rss"] > 0
    with pytest.raises(DriverError, match="memory probe"):
        _QueueWork(Path(queue.path), abs).probe(claimed)
    with pytest.raises(DriverError, match="PointResult"):
        _QueueWork(Path(queue.path), lambda point: 1)(claimed)
    with pytest.raises(OutputError, match="differs from its claim"):
        _QueueWork(Path(queue.path), Work())._publish(
            claimed, Work()(scan.point(0)).__replace__(index=1)
        )


def test_queue_refuses_changed_claim_coordinates(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    with pytest.raises(OutputError, match="values or seed changed"):
        claim(queue, dataclasses.replace(scan.point(0), seed=123))


@pytest.mark.parametrize("tamper", ["schema", "description", "state"])
def test_collection_refuses_malformed_manifest_or_state(tmp_path, tamper):
    _, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    filename = path / ("queue.json" if tamper == "state" else "manifest.json")
    document = module.read_json(filename)
    if tamper == "schema":
        document["queue_schema_version"] = 99
    elif tamper == "description":
        document["description"] = "{}"
    else:
        document["scan_id"] = "another-scan"
    atomic_json(filename, document)
    with pytest.raises(OutputError, match="manifest"):
        collect(path, tmp_path / "result.h5")


def test_collection_refuses_configuration_disagreement(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    publish(queue, claim(queue, scan.point(0)))
    assert queue.path is not None
    path = Path(queue.path)
    record = state(path)
    record["rows"][0]["configuration_hash"] = "0" * 64
    atomic_json(path / ".done", record)
    with pytest.raises(OutputError, match="configuration differs"):
        collect(path, tmp_path / "result.h5")


def test_collection_refuses_conflicting_already_collected_point(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    publish(queue, claim(queue, scan.point(0)))
    assert queue.path is not None
    path = Path(queue.path)
    output = tmp_path / "result.h5"
    collect(path, output)
    with h5py.File(output, "a") as file:
        _dataset(file, "points/configuration_hash")[0] = "0" * 64
    with pytest.raises(OutputError, match="conflicts with its durable artifact"):
        collect(path, output)


def test_queue_heartbeat_propagates_filesystem_failure(tmp_path, monkeypatch):
    heartbeat = _Heartbeat(tmp_path, "controller", 1.0)
    monkeypatch.setattr(heartbeat.stop, "wait", lambda interval: False)
    heartbeat._refresh()
    with pytest.raises(DriverError, match="lease refresh failed"):
        heartbeat.check()


def test_queue_heartbeat_ignores_rows_owned_by_other_controllers(tmp_path, monkeypatch):
    _, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    heartbeat = _Heartbeat(path, "another-controller", 1.0)
    waits = iter([False, True])
    monkeypatch.setattr(heartbeat.stop, "wait", lambda interval: next(waits))
    before = (path / "queue.json").read_bytes()
    heartbeat._refresh()
    heartbeat.check()
    assert (path / "queue.json").read_bytes() == before


def test_queue_claims_only_its_global_round_robin_chunk(tmp_path):
    scan, queue = prepare(tmp_path, size=4)
    queue.chunk = (1, 2)
    assert queue.path is not None
    holder = owner()
    heartbeat = _Heartbeat(Path(queue.path), holder["controller"], 60.0)
    claimed = list(queue._claims(scan.points(), holder, heartbeat))
    assert [module._point(item.job).index for item in claimed] == [1, 3]


def test_completed_queue_returns_no_work_without_idle_marker(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    publish(queue, claim(queue, scan.point(0)))
    assert list(queue.run(scan.points(), Work())) == []
    assert queue.path is not None
    assert not (Path(queue.path) / ".nowork").exists()


def test_collection_does_not_publish_superseded_snapshot(tmp_path, monkeypatch):
    scan, queue = prepare(tmp_path, size=1)
    publish(queue, claim(queue, scan.point(0)))
    assert queue.path is not None
    path = Path(queue.path)
    original = module.state
    reads = []

    def state_then_takeover(directory):
        snapshot = original(directory)
        if not reads:
            current = json.loads(json.dumps(snapshot))
            current["rows"][0].update(status="running", token="new-owner")
            atomic_json(path / "queue.json", current)
        reads.append(True)
        return snapshot

    monkeypatch.setattr(module, "state", state_then_takeover)
    output = collect(path, tmp_path / "result.h5")
    assert output.status.tolist() == ["todo"]
    assert "stats/energy" not in output.point(0)


def test_queue_artifacts_collect_idempotently(tmp_path):
    scan, queue = prepare(tmp_path)
    for point in reversed(list(scan.points())):
        assert publish(queue, claim(queue, point)).status == "done"
    assert queue.path is not None
    path = Path(queue.path)
    assert (path / ".done").exists()
    assert not (path / "queue.json").exists()
    result = collect(path, tmp_path / "scan.h5")
    assert result.status.tolist() == ["done", "done"]
    np.testing.assert_array_equal(result.as_grid("stats/energy"), np.ones((2, 2)))
    collect(path, tmp_path / "scan.h5")
    assert result.status.tolist() == ["done", "done"]


def test_live_local_owner_is_not_stolen_by_elapsed_time(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    first = claim(queue, scan.point(0))
    assert queue.path is not None
    path = Path(queue.path)
    record = state(path)
    record["rows"][0]["updated"] = 0.0
    atomic_json(path / "queue.json", record)
    with pytest.raises(StopIteration):
        claim(queue, scan.point(0))
    assert state(path)["rows"][0]["token"] == first.token


def test_dead_local_owner_recovered_immediately(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    first = claim(queue, scan.point(0))
    assert queue.path is not None
    path = Path(queue.path)
    record = state(path)
    record["rows"][0]["owner"]["pid"] = 2147483647
    atomic_json(path / "queue.json", record)
    second = claim(queue, scan.point(0))
    assert second.token != first.token
    assert publish(queue, second).status == "done"
    before = (path / "points/queue_00000.h5").read_bytes()
    assert publish(queue, first).status == "skipped"
    assert (path / "points/queue_00000.h5").read_bytes() == before


def test_remote_owner_requires_an_expired_lease(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    holder = owner() | {"host": "another-node.invalid"}
    first = claim(queue, scan.point(0), holder)
    with pytest.raises(StopIteration):
        claim(queue, scan.point(0))
    record = state(path)
    record["rows"][0]["updated"] = time.time() - queue.stale_age - 1.0
    atomic_json(path / "queue.json", record)
    assert claim(queue, scan.point(0)).token != first.token


def test_completion_marker_precedes_active_queue_removal(tmp_path, monkeypatch):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    original = Path.unlink
    observed = []

    def unlink(filename, *args, **kwargs):
        if filename == path / "queue.json":
            assert (path / ".done").exists()
            observed.append(True)
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    publish(queue, claim(queue, scan.point(0)))
    assert observed == [True]


def test_collection_rejects_modified_artifact(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    publish(queue, claim(queue, scan.point(0)))
    with h5py.File(path / "points/queue_00000.h5", "a") as file:
        file.attrs["tampered"] = True
    with pytest.raises(OutputError, match="digest changed"):
        collect(path, tmp_path / "scan.h5")


def test_queue_refuses_different_scan_manifest(tmp_path):
    _, queue = prepare(tmp_path, size=1)
    with pytest.raises(OutputError, match="manifest"):
        queue.prepare(
            ScanRequest(Scan("queue", x=[9]), {}, tmp_path / "scan.h5", Work())
        )


@pytest.mark.medium
def test_queue_runs_real_simulations_and_matches_local(tmp_path):
    scan = Scan("real", power=[1.0, 2.0])
    queue = Queue(tmp_path / "queue", pool=Pool(workers=2))
    result = run_scan(
        scan,
        build_point,
        executor=queue,
        output=tmp_path / "scan.h5",
        fields="all",
        progress="none",
    )
    assert result.status.tolist() == ["done", "done"]
    for point in scan.points():
        # A queue worker's run and the local one: the same march, to its budget.
        assert_same_march(
            result.point(point.index)["fields/values"],
            build_point(point).run(progress="none").fields(),
            point,
        )


# Wall-clock bounds for the spawned processes. Each child imports solprop and
# spawns its own pool workers, which on a loaded machine takes tens of seconds; these
# bounds only stop a hung test, and a pass returns as soon as the children finish.
SPAWN_TIMEOUT = 300.0


def controller(queue_path, output_path, start):
    scan = Scan("controllers", x=np.arange(6))
    queue = Queue(queue_path, pool=Pool(workers=1))
    queue.prepare(ScanRequest(scan, {}, output_path, Work()))
    # Claim together: otherwise a controller still importing on a loaded machine can
    # find every point already done, and the queue is not shared.
    start.wait(timeout=SPAWN_TIMEOUT)
    list(queue.run(scan.points(), Work()))
    collect(queue_path, output_path)


def join_all(processes, timeout):
    """Join processes against one shared deadline, s."""
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))


@pytest.mark.medium
def test_two_controllers_share_claims_and_one_collector(tmp_path):
    queue_path = tmp_path / "queue"
    output_path = tmp_path / "output.h5"
    context = multiprocessing.get_context("spawn")
    start = context.Barrier(2)
    processes = [
        context.Process(target=controller, args=(queue_path, output_path, start))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        join_all(processes, 2 * SPAWN_TIMEOUT)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        join_all(processes, 60.0)
    result = collect(queue_path, output_path)
    assert result.status.tolist() == ["done"] * 6
    assert len(list((queue_path / "points").glob("*.h5"))) == 6
    holders = {row["owner"]["controller"] for row in state(queue_path)["rows"]}
    assert len(holders) == 2


def test_heartbeat_refreshes_only_its_controller(tmp_path, monkeypatch):
    scan, queue = prepare(tmp_path)
    holder = owner()
    claim(queue, scan.point(0), holder)
    claim(queue, scan.point(1))
    assert queue.path is not None
    path = Path(queue.path)
    before = state(path)
    refreshed = threading.Event()
    write = module.atomic_json

    def record(filename, value):
        write(filename, value)
        refreshed.set()

    monkeypatch.setattr(module, "atomic_json", record)
    heartbeat = _Heartbeat(path, holder["controller"], 0.03)
    heartbeat.thread.start()
    try:
        assert refreshed.wait(timeout=2.0)
    finally:
        heartbeat.close()
    after = state(path)
    assert after["rows"][0]["updated"] > before["rows"][0]["updated"]
    assert after["rows"][1]["updated"] == before["rows"][1]["updated"]


def test_collection_repetition_does_not_grow_audit_or_change_file(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    publish(queue, claim(queue, scan.point(0)))
    output = tmp_path / "scan.h5"
    collect(path, output)
    before = output.read_bytes()
    collect(path, output)
    assert output.read_bytes() == before


def test_failed_queue_rows_require_retry_and_emit_nowork(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    claimed = claim(queue, scan.point(0))
    work = _QueueWork(path, Work())
    work.failure(claimed, "deliberate worker failure")
    assert list(queue.run(scan.points(), Work())) == []
    assert (path / ".nowork").exists()
    queue.prepare(
        ScanRequest(scan, {}, tmp_path / "scan.h5", Work(), retry_failed=True)
    )
    publish(queue, claim(queue, scan.point(0)))
    assert (path / ".done").exists()
    assert not (path / ".nowork").exists()


def _hold_claim(path, ready):
    scan, queue = prepare(path, size=1)
    claim(queue, scan.point(0))
    ready.set()
    # Held until the test kills it: a holder that outlived a slow test by exiting
    # on its own would leave a dead owner before the test meant one.
    time.sleep(10 * SPAWN_TIMEOUT)


@pytest.mark.medium
def test_killed_controller_claim_is_recovered_without_waiting_for_age(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_claim, args=(tmp_path, ready))
    process.start()
    try:
        assert ready.wait(timeout=SPAWN_TIMEOUT)
        assert process.is_alive()
        scan, queue = prepare(tmp_path, size=1)
        assert list(queue.run(scan.points(), Work())) == []
        process.terminate()
        process.join(timeout=60.0)
        assert process.exitcode is not None
        publish(queue, claim(queue, scan.point(0)))
        assert state(tmp_path / "queue")["rows"][0]["status"] == "done"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=60.0)


def test_failed_collection_is_idempotent(tmp_path):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    _QueueWork(path, Work()).failure(claim(queue, scan.point(0)), "expected failure")
    output = tmp_path / "scan.h5"
    collect(path, output)
    before = output.read_bytes()
    collect(path, output)
    assert output.read_bytes() == before


def test_crash_between_artifact_rename_and_state_is_recomputed(tmp_path, monkeypatch):
    scan, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    first = claim(queue, scan.point(0))
    original_finish = module.finish

    def crash(*args):
        raise OSError("interrupted state publication")

    monkeypatch.setattr(module, "finish", crash)
    with pytest.raises(OSError, match="interrupted state"):
        publish(queue, first)
    assert state(path)["rows"][0]["status"] == "running"
    assert len(list((path / "points").glob("*.h5"))) == 1
    assert collect(path, tmp_path / "scan.h5").status.tolist() == ["running"]
    record = state(path)
    record["rows"][0]["owner"]["host"] = "expired-remote-controller"
    record["rows"][0]["updated"] = 0.0
    atomic_json(path / "queue.json", record)
    monkeypatch.setattr(module, "finish", original_finish)
    replacement = claim(queue, scan.point(0))
    assert replacement.token != first.token
    publish(queue, replacement)
    assert collect(path, tmp_path / "scan.h5").status.tolist() == ["done"]


def test_collection_refuses_changed_manifest_description(tmp_path):
    _, queue = prepare(tmp_path, size=1)
    assert queue.path is not None
    path = Path(queue.path)
    manifest = module.read_json(path / "manifest.json")
    manifest["description"] += " "
    atomic_json(path / "manifest.json", manifest)
    with pytest.raises(OutputError, match="description digest"):
        collect(path, tmp_path / "scan.h5")
