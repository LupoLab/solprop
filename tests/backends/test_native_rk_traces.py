# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Runge-Kutta march against the decision corpus (SEP 45 section 7.2).

``tests/data/rk_traces.json.gz`` holds the decisions solprop's march made on scripted
norms before SEP 45 refactored it. The native march (``native/src/rk_march.cpp``),
driven through its adapter on a scripted engine, must make every one of them: each
attempt's step and norm, acceptance, landing, the start-up rule, the caps, natural
stops, and how it stopped -- at a call's bound of one attempt, seven, and the default,
so that where a call returns moves no decision. A one-attempt call books each attempt's
state, compared whole; longer calls are compared on the attempts and the end.

A replayed schedule runs through the march's replay (phase 4).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tests.backends.rk_traces import Scenario, _Harness, same_trace

from solprop.backends import available
from solprop.backends.driver import reached
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper.schedule import FixedSchedule

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
backend_module = pytest.importorskip("solprop_native.backend")
core = pytest.importorskip("solprop_native._core")

CORPUS = Path(__file__).parents[1] / "data" / "rk_traces.json.gz"


def _entries() -> list[dict[str, Any]]:
    with gzip.open(CORPUS, "rt") as handle:
        return json.load(handle)["scenarios"]


ENTRIES = _entries()


class _Progress:
    def step(self, z: float, dz: float, accepted: bool) -> None:
        pass


class _NativeHarness(_Harness):
    """The corpus's stub backend with the driver hooks the native adapter calls."""

    def __init__(self, scenario: Scenario, runner: Any) -> None:
        super().__init__(scenario)
        self.runner = runner
        self._history: list[Any] = []
        self._progress = cast("Any", _Progress())
        self._clock = 0.0
        self._row_spec = None
        self._row_period = 1
        self._accepted_steps = 0
        monitor = self.stub_monitor
        if monitor is not None:
            # A real monitor returns the march at its checks, where it reads a new cap
            # or quiet verdict; the script's changes stand in for them.
            changes = sorted({*scenario.quiet, *scenario.split} - {0})

            def steps_to_check() -> int:
                ahead = [key - self.accepted for key in changes if key > self.accepted]
                return min(ahead) if ahead else 1 << 30

            monitor.steps_to_check = steps_to_check  # pyright: ignore[reportAttributeAccessIssue]
            # Never checked, and an edge never read: only their caps reach the march.
            monitor.checked_at = None  # pyright: ignore[reportAttributeAccessIssue]
            edge: Any = monitor.transverse
            if edge is not None:
                edge.every_step = False
                edge.next_read = math.inf

    def _record_attempt(
        self, z: float, dz: float, norm: float, accepted: bool, wall: float
    ) -> None:
        self.close()
        event = self.runner.log()[len(self.events_of_attempts())]
        self.pending = {"z": event["z"], "dz": event["dz"]}
        self._record(z, dz, norm, accepted)

    def _poll_stop(self) -> None:
        # Before a call adopts its state: the last attempt's effect is complete.
        self.close()

    def events_of_attempts(self) -> list[dict[str, Any]]:
        return [one for one in self.events if "end" in one]

    def _account_step(self, z: float, cursor: int, energies: Any = None) -> None:
        self._take_step_stats()
        self._natural_count += 1


def _policy_free(harness: _NativeHarness) -> dict[str, object]:
    march = backend_module._NativeRungeKuttaMarch(harness)
    return march._policy()


def run_native(scenario: Scenario, max_attempts: int) -> dict[str, Any]:
    """Run a scenario through the native adapter on the scripted native march."""
    probe = _NativeHarness(scenario, None)
    policy = _policy_free(probe)
    norms = np.ascontiguousarray(scenario.norms, dtype=np.float64)
    schedule = (
        None
        if scenario.schedule is None
        else FixedSchedule(np.array(scenario.schedule, dtype=np.float64))
    )
    steps = edges = np.empty(0, dtype=np.float64)
    if schedule is not None:
        steps = np.ascontiguousarray(schedule.steps, dtype=np.float64)
        edges = np.ascontiguousarray(schedule.edges, dtype=np.float64)
    runner = core.ScriptedRkMarch(policy, norms, scenario.h0 or 0.0, steps, edges)
    harness = _NativeHarness(scenario, runner)
    march = backend_module._NativeRungeKuttaMarch(harness, runner=runner)
    march.MAX_ATTEMPTS = max_attempts
    targets = (
        list(scenario.targets)
        if scenario.targets is not None
        else sorted({*scenario.save_planes, scenario.z_end})
        if scenario.save_planes
        else [scenario.z_end]
    )
    end: dict[str, Any] = {"stop": "done"}
    try:
        for target in targets:
            while True:
                if schedule is None:
                    march.march_to(target)
                else:
                    march.replay_to(schedule, target)
                harness.close()
                harness.events.append({"target": target, **harness.snapshot()})
                if reached(harness._z, target):
                    break
    except (BackendError, ConvergenceError) as error:
        end = {"stop": type(error).__name__, "message": str(error)}
    except RuntimeError as error:
        if "ran out of norms" not in str(error):
            raise
        end = {"stop": "exhausted"}
    harness.close()
    # A trial the march made and never recorded: a replay's diverged step.
    logged = runner.log()
    made = len(harness.events_of_attempts())
    end["in_flight"] = (
        {"z": logged[made]["z"], "dz": logged[made]["dz"]} if len(logged) > made else {}
    )
    end.update(harness.snapshot())
    return {"events": harness.events, "end": end}


def _attempts(trace: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("z", "dz", "end", "step", "norm", "accepted")
    return [{key: one[key] for key in keys} for one in trace["events"] if "end" in one]


@pytest.mark.parametrize("max_attempts", [1, 7, 256])
@pytest.mark.parametrize("entry", ENTRIES, ids=[e["scenario"]["name"] for e in ENTRIES])
def test_the_native_march_makes_the_corpus_decisions(entry, max_attempts):
    scenario = Scenario.from_json(entry["scenario"])
    oracle = entry["trace"]
    trace = run_native(scenario, max_attempts)
    assert same_trace(_attempts(trace), _attempts(oracle))
    for name in ("stop", "message", "position", "proposal", "norms", "accepted"):
        assert same_trace(trace["end"].get(name), oracle["end"].get(name)), name
    if max_attempts == 1:
        assert same_trace(trace, oracle)
