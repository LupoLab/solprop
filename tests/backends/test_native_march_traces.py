# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march against the Python march's decision traces (SEP 39 7.1).

The corpus of ``tests/data/adams_traces.json.gz`` was taken from the Python march before
SEP 39 refactored it. Here the same scenarios run through the native Adams record
driven a call at a time (``native_trace_marches.TraceAdamsMarch``) with the native
march's loop and decisions, its field arithmetic replaced by the script of norms
(``_core.ScriptedMarch``). Each call is bound to one attempt, so the state the adapter
adopts after each call is the state after that attempt, and every stencil, decision,
proposal, order, landing, restart and ledger counter is compared exactly with the
reference's. A second pass lets the calls run to their events with the production bound
and compares the attempts and the end state.
"""

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.backends.adams_traces import (
    Scenario,
    _HostHarness,
    _march_state,
    run_scenario,
)

from solprop.backends import available
from solprop.exceptions import BackendError, ConvergenceError, ResponseBudgetError
from solprop.log import NullReporter
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

native = pytest.importorskip("solprop_native")
_core = pytest.importorskip("solprop_native._core")

from tests.backends.native_trace_marches import TraceAdamsMarch  # noqa: E402

CORPUS = Path(__file__).parents[1] / "data" / "adams_traces.json.gz"

CHANGED = {"floor_short_landing", "floor_after_rejections"}
"""Scenarios a later fix changed -- phase 0's commit order, and the recorded rejection
before a floor (issue #651); the fixed march is the reference."""


def _corpus() -> list[dict[str, Any]]:
    with gzip.open(CORPUS, "rt") as handle:
        return json.load(handle)["scenarios"]


ENTRIES = _corpus()


class _Progress(NullReporter):
    """Progress the harness does not report."""


class _Monitor:
    """The quiet collar's step cap, its checks due every step; no split collar's
    crossing cap, and no transverse edge."""

    step_cap = math.inf
    transverse = None

    def __init__(self, interval: float) -> None:
        self.interval = interval

    def steps_to_check(self) -> int:
        return 1

    @property
    def checked_at(self) -> None:
        return None


class _Runner:
    """The scripted native march, sorting its log into the harness's events."""

    def __init__(self, harness: _NativeHarness, norms: list[list[float]]) -> None:
        self.harness = harness
        self.norms = np.array(norms, dtype=np.float64).reshape(-1, 3)
        self.scripted = None
        self.seen = 0

    def run(self, state: dict[str, Any], bounds: dict[str, Any]) -> dict[str, Any]:
        harness = self.harness
        if self.scripted is None:
            march = harness.march
            assert march is not None
            policy = {
                name: getattr(march._policy, name)
                for name in march._policy.__dataclass_fields__
            }
            policy["quiet_interval"] = harness.scenario.quiet_interval
            schedule = harness.scenario.schedule
            steps = edges = np.empty(0, dtype=np.float64)
            if schedule is not None:
                replayed = FixedSchedule(np.array(schedule, dtype=np.float64))
                steps = np.ascontiguousarray(replayed.steps, dtype=np.float64)
                edges = np.ascontiguousarray(replayed.edges, dtype=np.float64)
            self.scripted = _core.ScriptedMarch(
                policy, self.norms, harness.scenario.carrier_rate or 0.0, steps, edges
            )
        bounds = {**bounds, "max_attempts": harness.max_attempts}
        result = self.scripted.run(state, bounds)
        log = self.scripted.log()
        held = [slot for _, _, slot in state["history"]] or [0]
        for event in log[self.seen :]:
            if "start" in event:
                harness.events.append(dict(event))
            else:
                harness.queue.append((dict(event), held))
        self.seen = len(log)
        return result


class _NativeHarness(_HostHarness):
    """The stub backend with the driver hooks the native march's adapter calls."""

    def __init__(self, scenario: Scenario, max_attempts: int) -> None:
        super().__init__(scenario)
        self.max_attempts = max_attempts
        self.queue: list[tuple[dict[str, Any], list[int]]] = []
        self._history: list[tuple[float, float, float, bool, float]] = []
        self._progress = _Progress()
        self._clock = 0.0
        self._row_spec = None
        self._accepted_steps = 0
        self._natural_count = 0
        self._stepper.step_planes = 0  # pyright: ignore[reportAttributeAccessIssue]
        if scenario.quiet_interval is not None:
            self._monitor = _Monitor(scenario.quiet_interval)  # pyright: ignore[reportAttributeAccessIssue]

    def _record_attempt(
        self, z: float, dz: float, norm: float, accepted: bool, wall: float
    ) -> None:
        event, held = self.queue.pop(0)
        if accepted:
            march = self.march
            assert march is not None
            event["kept"] = held[-march._policy.order :]
        self.pending = event
        self._record(z, dz, norm, accepted)

    def _account_step(self, z: float, cursor: int, energies: Any = None) -> None:
        self._take_step_stats()


def run_native(scenario: Scenario, max_attempts: int) -> dict[str, Any]:
    """Run a scenario through the native adapter on the scripted native march."""
    harness = _NativeHarness(scenario, max_attempts)
    method = Multistep(formulation="exponential", **scenario.method)
    runner = _Runner(harness, scenario.norms)
    march = TraceAdamsMarch(harness, method, runner=runner)
    harness.march = march
    if scenario.targets is not None:
        targets = list(scenario.targets)
    elif scenario.interpolate or not scenario.save_planes:
        targets = [scenario.z_end]
    else:
        targets = sorted({*scenario.save_planes, scenario.z_end})
    end: dict[str, Any] = {"stop": "done"}
    try:
        for target in targets:
            march.march_to(target)
            harness.events.append({"target": target, "position": harness._z})
    except (BackendError, ConvergenceError, ResponseBudgetError) as error:
        end = {"stop": type(error).__name__, "message": str(error)}
    if harness.queue:
        harness.pending = harness.queue[0][0]
    end["in_flight"] = dict(harness.pending)
    end.update(_march_state(harness, march))
    end["orders"] = list(march.ledger.orders)
    return {"events": harness.events, "end": end}


def expected(entry: dict[str, Any]) -> dict[str, Any]:
    """The reference's trace: the corpus's, or the fixed Python march's where phase 0
    changed it."""
    if entry["scenario"]["name"] in CHANGED:
        return run_scenario(Scenario.from_json(entry["scenario"]))
    return entry["trace"]


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=[entry["scenario"]["name"] for entry in ENTRIES]
)
def test_the_native_march_takes_the_reference_s_decisions(entry):
    scenario = Scenario.from_json(entry["scenario"])
    assert run_native(scenario, max_attempts=1) == expected(entry)


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=[entry["scenario"]["name"] for entry in ENTRIES]
)
def test_calls_to_their_events_take_the_same_attempts(entry):
    # With the production bound, the state after each attempt within a call is not
    # handed back, so the attempts and the end are compared.
    scenario = Scenario.from_json(entry["scenario"])
    found = run_native(scenario, max_attempts=256)
    reference = expected(entry)

    def attempts(trace):
        return [
            {key: event[key] for key in ("end", "step", "norm", "accepted", "order")}
            for event in trace["events"]
            if "accepted" in event
        ]

    assert attempts(found) == attempts(reference)
    assert found["end"] == reference["end"]


def test_a_rejection_below_the_floor_records_its_attempt_in_both_marches():
    # Issue #651: one order-one trial at z = 0 of 0.25 m, rejected with norm four, and
    # its shrunk proposal (0.1 m) below the 0.2 m floor. Each march records the
    # completed attempt once, then raises the floor with the accepted state, the
    # proposal and the ledger as they were.
    case = Scenario(
        name="rejected_floor_accounting",
        method={"order": 1, "variable_order": False},
        z0=0.0,
        proposal=0.25,
        dz_min=0.2,
        dz_max=1.0,
        z_end=1.0,
        norms=[[4.0, 4.0, 4.0]],
    )
    reference = run_scenario(case)
    for trace in (reference, run_native(case, max_attempts=1)):
        rows = [row for row in trace["events"] if "accepted" in row]
        assert [
            (row["end"], row["step"], row["norm"], row["accepted"], row["order"])
            for row in rows
        ] == [(0.25, 0.25, 4.0, False, 1)]
        end = trace["end"]
        assert end["stop"] == "ConvergenceError"
        assert "below the floor dz_min = 2.000e-01 m" in end["message"]
        assert (end["position"], end["proposal"]) == (0.0, 0.25)
        assert end["history"] == [[0.0, "accepted", 0]]
        # startup, predictor, rejected, accepted and rejected steps (`LEDGER`).
        assert end["ledger"][:5] == [1, 1, 1, 0, 1]
        assert end["in_flight"] == {}
    assert run_native(case, max_attempts=1) == reference
