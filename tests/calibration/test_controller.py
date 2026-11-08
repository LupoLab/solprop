# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The calibration controller, on scripted children (SEP 46 §7 items 4, 5, 11)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, cast

import pytest

from solprop.calibration import _native, controller
from solprop.calibration.controller import (
    LEVELS,
    CalibrationReport,
    Candidate,
    builtin_wait,
    calibrate,
    candidates,
    nested,
)
from solprop.exceptions import CalibrationError

type Factor = Callable[[str, int], float]


class _Scripted(controller._Child):  # pyright: ignore[reportPrivateUsage]
    """A child whose window times follow a script: seconds = 1 ms x factor."""

    factor: ClassVar[Factor] = staticmethod(lambda name, call: 1.0)
    # A record's factor: the grain candidates' windows, by the record they run.
    by_record: ClassVar[Callable[[Mapping[str, object] | None, int], float]] = (
        staticmethod(lambda record, call: 1.0)
    )
    shape: ClassVar[list[int]] = [1, 1, 1, 4096]
    sizing: ClassVar[dict[str, object]] = {
        "kernels.stage.4096": {"threads": 8},
        "kernels.response.98304.samples.4096": {"threads": 4},
    }
    fail: ClassVar[frozenset[str]] = frozenset()
    fail_prepare: ClassVar[frozenset[str]] = frozenset()
    started: ClassVar[list[str]] = []

    def __init__(
        self, candidate: Candidate, cpus: Sequence[int], environ: Mapping[str, str]
    ) -> None:
        self.candidate = candidate
        self.cpus = tuple(cpus)
        self.environ = dict(environ)
        self.report: dict[str, object] = {}
        self.calls = 0
        self.pending: Mapping[str, object] = {}
        self.prepared: list[str] = []
        self.stopped = False
        self.readers = ()
        _Scripted.started.append(candidate.name)

    def send(self, request: Mapping[str, object]) -> None:
        self.pending = request

    def receive(self, timeout: float) -> object:
        request = self.pending
        failing = self.fail_prepare if request["op"] == "prepare" else self.fail
        if self.candidate.name in failing:
            raise controller._ChildError(f"{self.candidate.name} exited")  # pyright: ignore[reportPrivateUsage]
        if request["op"] == "prepare":
            problems = request["problems"]
            assert isinstance(problems, dict)
            self.prepared = list(cast(dict[str, float], problems))
            return {
                "problems": {
                    name: {"step": 1e-3, "shape": list(type(self).shape)}
                    for name in problems
                },
                "process": {
                    "threads": self.candidate.threads or 8,
                    "wait": self.candidate.wait or "spin",
                },
                "execution": _native.execution.builtin("Linux").native(),
            }
        if request["op"] == "sizing":
            return {name: dict(type(self).sizing) for name in self.prepared}
        if request["op"] == "regions":
            return {"copy": 1e-6, "empty": 1e-6}
        assert request["op"] == "time"
        self.calls += 1
        record = cast(Mapping[str, object] | None, request.get("record"))
        return (
            1e-3
            * type(self).factor(self.candidate.name, self.calls)
            * type(self).by_record(record, self.calls)
        )

    def request(self, request: Mapping[str, object], timeout: float) -> object:
        self.send(request)
        return self.receive(timeout)

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def scripted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[type[_Scripted], dict[str, str]]:
    """Scripted children on an eight-core allocation with an empty store."""

    class Script(_Scripted):
        started: ClassVar[list[str]] = []

    monkeypatch.setattr(controller, "_Child", Script)
    monkeypatch.setattr(controller, "_telemetry", lambda path=None: None)
    monkeypatch.setattr(controller, "_native_budget", lambda cpus: 8)
    monkeypatch.setattr(controller._process, "top_tier_cores", lambda: None)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(controller, "nested", lambda topology, cpus: [])
    environ = {"SOLPROP_CACHE_DIR": str(tmp_path / "store"), "PATH": ""}
    return Script, environ


def _rows(environ: Mapping[str, str]) -> list[dict[str, object]]:
    """Return the current generation's rows, as a reader finds them."""
    store = Path(environ["SOLPROP_CACHE_DIR"]) / "machines"
    for current in store.glob("*/current"):
        name = current.read_text(encoding="utf-8").strip()
        generation = current.parent / f"{name}.json"
        return json.loads(generation.read_text(encoding="utf-8"))["rows"]
    return []


def test_the_built_in_wait_spins_unless_the_team_is_wider_than_the_top_tier() -> None:
    assert builtin_wait(6, 6) == "spin"
    assert builtin_wait(18, 6) == "passive"
    assert builtin_wait(36, None) == "spin"


def _allocation(
    cores: int, *, kinds: tuple[int, ...] | None = None, sockets: int = 1, l3: int = 1
) -> object:
    return _native.allocation.AllocationClass(
        cores=cores,
        kinds=kinds or (cores,),
        l2_groups=0,
        l3_groups=l3,
        numa_nodes=sockets,
        sockets=sockets,
        smt=False,
    )


def test_candidates_try_the_other_wait_and_narrower_teams() -> None:
    standard, quick = LEVELS["standard"], LEVELS["quick"]
    # A two-socket server: the other wait, one socket, half of it, passive on a socket.
    server = candidates(standard, _allocation(36, sockets=2), 36)  # pyright: ignore[reportArgumentType]
    assert [c.name for c in server] == ["passive/36", "spin/18", "passive/18"]
    # Its L3 groups: an EPYC with two eight-core groups per socket.
    epyc = candidates(standard, _allocation(32, sockets=2, l3=4), 32)  # pyright: ignore[reportArgumentType]
    assert [c.name for c in epyc] == ["passive/32", "spin/16", "spin/8", "passive/16"]
    # Two kinds of core: the fastest kind's, at the rule's wait for each width.
    mac = candidates(standard, _allocation(18, kinds=(6, 12)), 18, top_tier=6)  # pyright: ignore[reportArgumentType]
    # 9 is wider than the top tier, so the rule's wait there is passive.
    assert [c.name for c in mac] == ["spin/18", "passive/9", "spin/6", "spin/9"]
    assert len(candidates(quick, _allocation(36, sockets=2), 36)) == 3  # pyright: ignore[reportArgumentType]
    # One core: only the other wait.
    assert [c.name for c in candidates(standard, _allocation(1), 1)] == ["passive/1"]  # pyright: ignore[reportArgumentType]


def test_a_childs_environment_drops_the_users_settings() -> None:
    base = {
        "SOLPROP_THREADS": "4",
        "OMP_WAIT_POLICY": "active",
        "SOLPROP_FFTW_WISDOM": "/w",
        "HOME": "/h",
    }
    built = Candidate().environment(base)
    assert built == {
        "HOME": "/h",
        "SOLPROP_CALIBRATION": "off",
        "SOLPROP_FFTW_PLAN": "estimate",
        "PYTHONUNBUFFERED": "1",
    }
    spin = Candidate(threads=6, wait="spin")
    assert spin.name == "spin/6"
    assert spin.environment(base)["SOLPROP_NATIVE_THREADS"] == "6"
    assert spin.environment(base)["SOLPROP_NATIVE_WAIT_REQUEST"] == "spin"
    assert Candidate(threads=6).name == "default/6"


def test_nested_allocations_are_one_socket_node_and_cache_group() -> None:
    topology = _native.topology.Topology(
        cpus=tuple(range(8)),
        core={cpu: cpu for cpu in range(8)},
        socket={cpu: cpu // 4 for cpu in range(8)},
        numa={cpu: cpu // 4 for cpu in range(8)},
        l2={cpu: cpu for cpu in range(8)},
        l3={cpu: cpu // 2 for cpu in range(8)},
    )
    assert nested(topology, range(8)) == [(0, 1, 2, 3), (0, 1)]
    assert nested(topology, [4, 5, 6, 7]) == [(4, 5)]
    assert nested(topology, [0, 1]) == []


def test_telemetry_reads_running_tasks_and_steal(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text(
        "cpu  100 0 100 700 0 0 0 100 0 0\ncpu0 1 2 3 4 5 6 7 8\nprocs_running 3\n",
        encoding="utf-8",
    )
    values = controller._telemetry(stat)  # pyright: ignore[reportPrivateUsage]
    assert values == {"total": 1000.0, "steal": 100.0, "running": 3.0}
    assert controller._telemetry(tmp_path / "missing") is None  # pyright: ignore[reportPrivateUsage]
    busy = controller._busy  # pyright: ignore[reportPrivateUsage]
    assert busy(None, values, 4) == ""
    quiet = {"total": 0.0, "steal": 0.0, "running": 2.0}
    assert busy(quiet, {"total": 1000.0, "steal": 50.0, "running": 2.0}, 0) == ""
    assert "stolen" in busy(quiet, {"total": 1000.0, "steal": 200.0, "running": 2.0}, 0)
    loaded = {"total": 0.0, "steal": 0.0, "running": 9.0}
    assert busy(loaded, loaded, 4) == "7 other tasks running for 4 idle CPUs"


def test_a_candidate_that_wins_every_round_and_its_validation_is_published(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.factor = staticmethod(lambda name, call: 0.8 if name == "spin/4" else 1.0)
    lines: list[str] = []
    report = calibrate(
        "standard", environ=environ, cpus=range(8), progress=lines.append
    )
    assert report.status == "published"
    assert report.chosen == {"threads": 4, "wait": "spin"}
    assert report.builtin == {"threads": 8, "wait": "spin"}
    assert report.reason.startswith("spin/4 won every round and its validation")
    assert report.path is not None
    assert any("validating spin/4" in line for line in lines)
    evidence = report.evidence
    assert evidence["noise"] == 0.02
    validation = evidence["validation"]
    assert isinstance(validation, dict)
    assert validation["passed"] is True
    rows = _rows(environ)
    assert len(rows) == 1
    assert rows[0]["process"] == {"threads": 4, "wait": "spin"}
    assert rows[0]["fitted_under"] == {"threads": 4, "wait": "spin"}
    text = report.describe()
    assert "threads: 4 (built-in 8)" in text
    assert "wait: spin\n" in text


def test_no_winner_publishes_the_built_in_rules(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    # spin/4 wins on average but loses one round.
    script.factor = staticmethod(
        lambda name, call: (
            (1.2 if (call - 1) // 3 == 1 else 0.7) if name == "spin/4" else 1.0
        )
    )
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "published"
    assert report.chosen == report.builtin
    assert report.reason.startswith("the built-in rules kept")
    assert "validation" not in report.evidence


def test_a_winner_that_fails_validation_is_dropped(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    # Faster in the rounds' mean of windows, but its last window is slower.
    calls: dict[str, int] = {}

    def factor(name: str, call: int) -> float:
        if name != "spin/4":
            return 1.0
        calls[name] = call
        # Each round times three windows; validation follows three rounds of them.
        in_validation = call > 9
        window = (call - 1) % 3
        if in_validation and window == 2:
            return 1.5
        return 0.6

    script.factor = staticmethod(factor)
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "published"
    assert report.chosen == report.builtin
    validation = report.evidence["validation"]
    assert isinstance(validation, dict)
    assert validation["passed"] is False


def test_noise_above_the_limit_publishes_nothing(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    # The built-in child varies by 30 % between its two slots of a round.
    script.factor = staticmethod(
        lambda name, call: (
            (1.3 if (call - 1) // 3 % 2 else 1.0) if name == "built-in" else 1.0
        )
    )
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "inconclusive"
    assert report.reason.startswith("A/A noise")
    assert _rows(environ) == []
    found = _native.store.Store(Path(environ["SOLPROP_CACHE_DIR"]))
    assert found.inconclusive_since(report.fingerprint, _key(report)) is not None


def _key(report: CalibrationReport) -> str:
    return _native.allocation.AllocationClass.read(report.allocation).key()


def test_a_busy_host_or_a_short_budget_is_inconclusive(
    scripted: tuple[type[_Scripted], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, environ = scripted
    loaded = {"total": 0.0, "steal": 0.0, "running": 500.0}
    monkeypatch.setattr(controller, "_telemetry", lambda path=None: loaded)
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "inconclusive"
    assert "other tasks running" in report.reason
    monkeypatch.setattr(controller, "_telemetry", lambda path=None: None)
    # The whole budget is the reserve: no round fits.
    short = calibrate("quick", 1e-3, environ=environ, cpus=range(8))
    assert short.status == "inconclusive"
    assert short.reason == "too few rounds within the budget"


def test_a_failed_child_publishes_nothing_and_every_child_is_reaped(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.fail_prepare = frozenset({"built-in"})
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "inconclusive"
    assert "built-in exited" in report.reason
    assert _rows(environ) == []
    # A built-in child that fails every window leaves no noise to judge by.
    script.fail_prepare = frozenset()
    script.fail = frozenset({"built-in"})
    failed = calibrate("quick", environ=environ, cpus=range(8))
    assert failed.status == "inconclusive"
    assert failed.reason == "too few rounds within the budget"


def test_a_failed_round_of_one_candidate_is_dropped(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.fail = frozenset({"passive/8"})
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "published"
    verdicts = report.evidence["verdicts"]
    assert isinstance(verdicts, dict)
    assert verdicts["passive/8"]["reason"] == "0 of 3 rounds"  # pyright: ignore[reportUnknownMemberType]


def test_a_held_claim_skips_the_calibration(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    machine = _native.topology.read_topology()
    allocation = _native.allocation.classify(
        machine,
        tuple(range(8)),
        controller._process.cgroup_cpus(),  # pyright: ignore[reportPrivateUsage]
    )
    from solprop import calibration  # noqa: PLC0415

    found = calibration.choice(environ | {"SOLPROP_CALIBRATION": "use"})
    assert found.fingerprint is not None
    store = _native.store.Store(Path(environ["SOLPROP_CACHE_DIR"]))
    claim = store.claim(found.fingerprint.digest, allocation.key(), 1e12)
    assert claim is not None
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert report.status == "held"
    assert script.started == []
    claim.release()


def test_nested_allocations_get_rows_while_the_budget_allows(
    scripted: tuple[type[_Scripted], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, environ = scripted
    monkeypatch.setattr(controller, "nested", lambda topology, cpus: [(0, 1, 2, 3)])
    report = calibrate("standard", environ=environ, cpus=range(8))
    inner = report.evidence["nested"]
    assert isinstance(inner, list)
    assert [item["status"] for item in inner] == ["published"]  # pyright: ignore[reportUnknownVariableType]
    assert len(_rows(environ)) == 2
    assert "nested" in report.describe()
    # A budget spent on the allocation itself skips the nested one.
    ticks = iter(range(10_000))
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(ticks) * 5.0)
    skipped = calibrate("standard", 300.0, environ=environ, cpus=range(8))
    inner = skipped.evidence["nested"]
    assert isinstance(inner, list)
    assert [item["status"] for item in inner] == ["skipped"]  # pyright: ignore[reportUnknownVariableType]


def test_a_higher_level_row_survives_a_lower_level_calibration(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.factor = staticmethod(lambda name, call: 0.8 if name == "spin/4" else 1.0)
    calibrate("standard", environ=environ, cpus=range(8))
    script.factor = staticmethod(lambda name, call: 1.0)
    calibrate("quick", environ=environ, cpus=range(8))
    rows = _rows(environ)
    assert [row["level"] for row in rows] == ["standard"]
    assert rows[0]["process"] == {"threads": 4, "wait": "spin"}
    # Another allocation's row is kept beside it.
    calibrate("quick", environ=environ, cpus=range(4))
    assert sorted(str(row["level"]) for row in _rows(environ)) == ["quick", "standard"]


def test_levels_and_budgets_are_checked() -> None:
    with pytest.raises(CalibrationError, match="unknown calibration level"):
        calibrate("forever")
    with pytest.raises(CalibrationError, match="must be positive"):
        calibrate("quick", 0.0)


def test_the_built_in_width_is_the_smallest_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = controller._native_budget  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        controller._process, "physical_cores", lambda cpus: len(cpus) // 2
    )  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(controller._process, "top_tier_cores", lambda: None)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(controller._process, "cgroup_cpus", lambda: None)  # pyright: ignore[reportPrivateUsage]
    assert budget(range(8)) == 4
    monkeypatch.setattr(controller._process, "cgroup_cpus", lambda: 3)  # pyright: ignore[reportPrivateUsage]
    assert budget(range(8)) == 3
    assert budget(()) >= 1


def test_a_real_childs_errors_are_reported_and_it_is_reaped() -> None:
    real = controller._Child  # pyright: ignore[reportPrivateUsage]
    error = controller._ChildError  # pyright: ignore[reportPrivateUsage]
    environ = dict(controller.os.environ)
    child = real(Candidate(), (), environ)
    try:
        with pytest.raises(error, match="unknown request"):
            child.request({"op": "colour"}, 60.0)
        # Nothing asked, nothing answered within the time allowed.
        with pytest.raises(error, match="overran the deadline"):
            child.receive(0.0)
    finally:
        child.stop()
    assert child.process.returncode == 0
    # The exit request's own reply, then the end of the child's output.
    assert child.receive(1.0) is True
    with pytest.raises(error, match="exited"):
        child.receive(1.0)
    with pytest.raises(error):
        child.send({"op": "exit"})


def test_a_child_that_ignores_its_exit_request_is_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = controller._Child  # pyright: ignore[reportPrivateUsage]
    child = real(Candidate(), (), dict(controller.os.environ))
    monkeypatch.setattr(child, "send", lambda request: None)
    child.stop()
    assert child.process.returncode is not None
    assert child.process.returncode != 0


def test_a_child_failing_during_validation_drops_the_winner(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.factor = staticmethod(lambda name, call: 0.8 if name == "spin/4" else 1.0)

    original = script.receive

    def receive(self: _Scripted, timeout: float) -> object:
        # The winner's windows fail once the rounds are over (five rounds of five).
        if (
            self.candidate.name == "spin/4"
            and self.calls >= 25
            and (self.pending.get("op") == "time")
        ):
            raise controller._ChildError("spin/4 exited")  # pyright: ignore[reportPrivateUsage]
        return original(self, timeout)

    script.receive = receive  # type: ignore[method-assign]
    report = calibrate("standard", environ=environ, cpus=range(8))
    assert report.status == "published"
    assert report.chosen == report.builtin
    validation = report.evidence["validation"]
    assert isinstance(validation, dict)
    assert validation["passed"] is False


def test_an_unreadable_or_equal_row_is_replaced_on_publication(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    _, environ = scripted
    first = calibrate("quick", environ=environ, cpus=range(8))
    assert first.path is not None
    record = json.loads(first.path.read_text(encoding="utf-8"))
    record["rows"].append(dict(record["rows"][0], allocation={"cores": "many"}))
    first.path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.warns(_native.store.StoreWarning, match="not an allocation class"):
        calibrate("quick", environ=environ, cpus=range(8))
    rows = _rows(environ)
    assert len(rows) == 1


def test_a_childs_standard_error_keeps_its_last_lines() -> None:
    import io  # noqa: PLC0415
    import types  # noqa: PLC0415

    real = controller._Child  # pyright: ignore[reportPrivateUsage]
    child = real.__new__(real)
    child.errors = []
    lines = "".join(f"line {index}\n" for index in range(30))
    child.process = types.SimpleNamespace(stderr=io.StringIO(lines))  # type: ignore[assignment]
    child._drain()  # pyright: ignore[reportPrivateUsage]
    assert child.errors == [f"line {index}" for index in range(10, 30)]


def test_admitted_grain_settings_are_validated_and_published(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.by_record = staticmethod(
        lambda record, call: 0.8 if record and record["medium_threads"] == 2 else 1.0
    )
    report = calibrate("standard", environ=environ, cpus=range(8))
    assert report.status == "published"
    assert report.chosen["medium_threads"] == 2
    assert report.builtin["medium_threads"] == 4
    assert "grain: admitted: medium 2" in report.reason
    (row,) = _rows(environ)
    assert row["workspace"] == {"medium_threads": 2}
    evidence = row["evidence"]
    assert isinstance(evidence, dict)
    grain_evidence = evidence["grain"]
    assert isinstance(grain_evidence, dict)
    assert grain_evidence["validation"]["passed"] is True  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]


def test_a_grain_that_fails_its_validation_is_dropped(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    timed: list[int] = []

    def by_record(record: Mapping[str, object] | None, call: int) -> float:
        if record and record["medium_threads"] == 2:
            timed.append(call)
            # Faster in its rounds, slower once every window is timed together.
            return 0.8 if len(timed) <= 25 else 1.3
        return 1.0

    script.by_record = staticmethod(by_record)
    report = calibrate("standard", environ=environ, cpus=range(8))
    assert "medium_threads" not in report.chosen
    assert "the combined grain failed its validation" in report.reason


def test_the_grain_reports_why_it_kept_the_table(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    # No candidate wins.
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert "grain: no grain candidate won every round" in report.reason
    # No operation a candidate could move, and no compact problem.
    script.sizing = {}
    script.shape = [1, 4, 1, 4096]
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert "grain: no candidate changes an operation's width" in report.reason


def test_a_noisy_or_failed_grain_keeps_the_table(
    scripted: tuple[type[_Scripted], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    script, environ = scripted
    measured = controller.selection.noise
    calls: list[int] = []

    def noise(builtin: object, again: object, minimum: int) -> float | None:
        # The process session's noise, then the grain session's: far above the limit.
        calls.append(minimum)
        return measured(builtin, again, minimum) if len(calls) == 1 else 0.5  # type: ignore[arg-type]

    monkeypatch.setattr(controller.selection, "noise", noise)
    noisy = calibrate("standard", environ=environ, cpus=range(8))
    assert "grain: inconclusive grain noise" in noisy.reason
    monkeypatch.setattr(controller.selection, "noise", measured)

    original = script.receive

    def receive(self: _Scripted, timeout: float) -> object:
        if self.pending.get("op") == "sizing":
            raise controller._ChildError("built-in exited")  # pyright: ignore[reportPrivateUsage]
        if self.pending.get("op") == "time" and self.pending.get("record"):
            raise controller._ChildError("built-in exited")  # pyright: ignore[reportPrivateUsage]
        return original(self, timeout)

    script.receive = receive  # type: ignore[method-assign]
    failed = calibrate("quick", environ=environ, cpus=range(8))
    assert "grain: built-in exited" in failed.reason


def test_a_failed_grain_window_is_dropped(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    original = script.receive

    def receive(self: _Scripted, timeout: float) -> object:
        record = self.pending.get("record")
        if isinstance(record, dict) and record.get("compact_threads") == 4:
            raise controller._ChildError("compact window failed")  # pyright: ignore[reportPrivateUsage]
        return original(self, timeout)

    script.receive = receive  # type: ignore[method-assign]
    report = calibrate("quick", environ=environ, cpus=range(8))
    evidence = report.evidence["grain"]
    assert isinstance(evidence, dict)
    verdicts = evidence["verdicts"]
    assert isinstance(verdicts, dict)
    assert verdicts["compact 4"]["reason"] == "0 of 3 rounds"  # pyright: ignore[reportUnknownMemberType]


def test_a_candidate_that_changes_no_problem_is_left_out(
    scripted: tuple[type[_Scripted], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    script, environ = scripted
    script.shape = [1, 4, 1, 4096]
    monkeypatch.setattr(controller.grain, "affected", lambda *args: False)
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert "grain: no candidate changes an operation's width" in report.reason


def test_a_grain_out_of_time_keeps_the_table(
    scripted: tuple[type[_Scripted], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    script, environ = scripted
    sized: list[bool] = []
    original = script.receive

    def receive(self: _Scripted, timeout: float) -> object:
        if self.pending.get("op") == "sizing":
            sized.append(True)
        return original(self, timeout)

    script.receive = receive  # type: ignore[method-assign]
    deadline = controller._Deadline  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(deadline, "remaining", lambda self: 0.0 if sized else 1e9)
    report = calibrate("quick", environ=environ, cpus=range(8))
    assert "grain: inconclusive grain noise" in report.reason


def test_a_failed_validation_window_fails_the_grain(
    scripted: tuple[type[_Scripted], dict[str, str]],
) -> None:
    script, environ = scripted
    script.by_record = staticmethod(
        lambda record, call: 0.8 if record and record["medium_threads"] == 2 else 1.0
    )
    windows: list[int] = []
    original = script.receive

    def receive(self: _Scripted, timeout: float) -> object:
        record = self.pending.get("record")
        if isinstance(record, dict) and record.get("medium_threads") == 2:
            windows.append(1)
            if len(windows) > 25:
                raise controller._ChildError("validation window failed")  # pyright: ignore[reportPrivateUsage]
        return original(self, timeout)

    script.receive = receive  # type: ignore[method-assign]
    report = calibrate("standard", environ=environ, cpus=range(8))
    assert "the combined grain failed its validation" in report.reason
