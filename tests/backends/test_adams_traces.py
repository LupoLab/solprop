# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march against its decision traces on scripted norms (SEP 39 section 7.1).

``tests/data/adams_traces.json.gz`` holds the decisions the march made, on each
scenario's scripted norms, before SEP 39 refactored it into a pure controller core
(``tests/backends/adams_traces.py``). The march must reproduce every one exactly:
the stencils it asked the engine for, every acceptance, order, proposal, landing,
restart and ledger counter, and how it stopped.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from tests.backends.adams_traces import Scenario, run_scenario

STATE = (
    "position",
    "proposal",
    "stored_order",
    "rejections",
    "last_rejected",
    "collar_was_quiet",
    "quiet",
    "history",
    "ledger",
)
"""The march's state a trace records after each attempt."""

CORPUS = Path(__file__).parents[1] / "data" / "adams_traces.json.gz"


def _corpus() -> list[dict[str, Any]]:
    with gzip.open(CORPUS, "rt") as handle:
        return json.load(handle)["scenarios"]


ENTRIES = _corpus()

CHANGED = {
    # SEP 39 phase 0 fixes the commit order (section 4.4): an accepted landing whose
    # next proposal falls below the floor is recorded, and leaves the position at its
    # end, before the floor is raised. The oracle keeps the march as it was.
    "floor_short_landing",
}


def test_the_corpus_was_generated_before_the_refactor():
    with gzip.open(CORPUS, "rt") as handle:
        assert json.load(handle)["generated_from"] == "a500d2d"


UNCHANGED = [entry for entry in ENTRIES if entry["scenario"]["name"] not in CHANGED]


@pytest.mark.parametrize(
    "entry", UNCHANGED, ids=[entry["scenario"]["name"] for entry in UNCHANGED]
)
def test_the_march_reproduces_its_trace(entry):
    scenario = Scenario.from_json(entry["scenario"])
    assert run_scenario(scenario) == entry["trace"]


def test_a_floor_after_an_accepted_landing_is_raised_once_the_step_is_committed():
    # Two planes 4e-7 m apart with a floor of 1e-6 m: the landing between them is
    # accepted, and its next proposal (at most twice the step) is below the floor.
    # Before the fix the march raised with the state adopted at the plane and the
    # position still at the step's start, and recorded nothing; now the step is
    # recorded, the position is its end, and then the floor is raised.
    (entry,) = [
        one for one in ENTRIES if one["scenario"]["name"] == "floor_short_landing"
    ]
    scenario = Scenario.from_json(entry["scenario"])
    oracle = entry["trace"]
    assert oracle["end"]["stop"] == "ConvergenceError"
    trace = run_scenario(scenario)
    plane = scenario.save_planes[1]
    flight = oracle["end"]["in_flight"]
    state = {key: value for key, value in trace["end"].items() if key in STATE}
    row = {
        **flight,
        "end": plane,
        "step": flight["dz"],
        "norm": scenario.norms[-1][0],
        "accepted": True,
        **state,
    }
    assert trace["events"] == [*oracle["events"], row]
    expected = {**oracle["end"], "in_flight": {}, "position": plane}
    assert trace["end"] == expected
    assert oracle["end"]["position"] == flight["z"] != plane
