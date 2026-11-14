# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march against its decision traces on scripted norms (SEP 39 section 7.1).

``tests/data/adams_traces.json.gz`` holds the decisions the march made, on each
scenario's scripted norms, before SEP 39 refactored it into a pure controller core
(``tests/backends/adams_traces.py``). The march must reproduce every one exactly:
the stencils it asked the engine for, every acceptance, order, proposal, landing,
restart and ledger counter, and how it stopped. The fused trial runs the same
scenarios, held to the exponential march's decisions with the collar off.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from tests.backends.adams_traces import Scenario, run_scenario

from solprop.stepper.multistep import Multistep

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
    # Issue #651: a rejection whose shrunk proposal falls below the floor is recorded
    # before the floor is raised, as an accepted step's is. The corpus keeps the
    # historical omission; it is not regenerated.
    "floor_after_rejections",
}


UNCHANGED = [entry for entry in ENTRIES if entry["scenario"]["name"] not in CHANGED]


@pytest.mark.parametrize(
    "entry", UNCHANGED, ids=[entry["scenario"]["name"] for entry in UNCHANGED]
)
def test_the_march_reproduces_its_trace(entry):
    scenario = Scenario.from_json(entry["scenario"])
    assert run_scenario(scenario) == entry["trace"]


FUSED = [entry for entry in ENTRIES if entry["scenario"]["carrier_rate"] is None]
"""The scenarios the fused path runs: its formulation refuses a carrier frame."""


@pytest.mark.parametrize(
    "entry", FUSED, ids=[entry["scenario"]["name"] for entry in FUSED]
)
def test_the_fused_march_takes_the_exponential_decisions_without_the_collar(entry):
    # Issue #270: the fused trial (the interaction formulation, every order of the
    # one-order-higher PEC family) runs the exponential path's controller, except
    # that an in-stage collar's stability does not weigh its order choice. Its trace
    # on every scenario is therefore the exponential march's with the collar off: the
    # corpus's own where the scenario has no collar and no later fix changed it. The
    # scripted engine refuses the other family's trial, so the fused path is the one
    # taken.
    scenario = Scenario.from_json(entry["scenario"])
    fused = run_scenario(scenario, formulation="interaction")
    if scenario.collar_rate == 0.0 and scenario.name not in CHANGED:
        exponential = entry["trace"]
    else:
        exponential = run_scenario(dataclasses.replace(scenario, collar_rate=0.0))
    assert fused == _named_fused(exponential, scenario)


def _named_fused(trace: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Return an exponential trace with its stopping message naming the fused method."""
    end = trace["end"]
    if "message" not in end:
        return trace
    names = [
        Multistep(formulation=formulation, **scenario.method).name
        for formulation in ("exponential", "interaction")
    ]
    return {**trace, "end": {**end, "message": end["message"].replace(*names)}}


@pytest.mark.parametrize("name", ["collar_cap_binding", "collar_quiet_toggle"])
def test_the_fused_march_ignores_a_collar_the_exponential_march_steps_around(name):
    # The scenarios where the collar's stable step changes the exponential march's
    # decisions (the collar weighs the order choice, so only these variable-order
    # ones): the comparison above with the collar off is not vacuous on them.
    (entry,) = [one for one in ENTRIES if one["scenario"]["name"] == name]
    scenario = Scenario.from_json(entry["scenario"])
    assert scenario.collar_rate > 0.0
    assert run_scenario(scenario, formulation="interaction") != entry["trace"]


@pytest.mark.parametrize("formulation", ["exponential", "interaction"])
def test_a_floor_after_an_accepted_landing_is_raised_once_the_step_is_committed(
    formulation,
):
    # Two planes 4e-7 m apart with a floor of 1e-6 m: the landing between them is
    # accepted, and its next proposal (at most twice the step) is below the floor.
    # Before the fix the march raised with the state adopted at the plane and the
    # position still at the step's start, and recorded nothing; now the step is
    # recorded, the position is its end, and then the floor is raised. The fused
    # path commits through the same `_decide` (issue #270).
    (entry,) = [
        one for one in ENTRIES if one["scenario"]["name"] == "floor_short_landing"
    ]
    scenario = Scenario.from_json(entry["scenario"])
    oracle = entry["trace"]
    assert oracle["end"]["stop"] == "ConvergenceError"
    trace = run_scenario(scenario, formulation=formulation)
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


@pytest.mark.parametrize("formulation", ["exponential", "interaction"])
def test_a_floor_after_a_rejection_is_raised_once_the_attempt_is_recorded(formulation):
    # Issue #651: the march's last trial is rejected and its shrunk proposal is below
    # the floor. Before the fix the floor was raised inside the rejection and the
    # completed trial left no row; now it has one, and everything else -- the
    # proposal, the history, the ledger, the message -- is the corpus's.
    (entry,) = [
        one for one in ENTRIES if one["scenario"]["name"] == "floor_after_rejections"
    ]
    scenario = Scenario.from_json(entry["scenario"])
    oracle = entry["trace"]
    assert oracle["end"]["stop"] == "ConvergenceError"
    trace = run_scenario(scenario, formulation=formulation)
    flight = oracle["end"]["in_flight"]
    state = {key: value for key, value in trace["end"].items() if key in STATE}
    row = {
        **flight,
        "end": flight["z"] + flight["dz"],
        "step": flight["dz"],
        "norm": scenario.norms[-1][0],
        "accepted": False,
        **state,
    }
    assert trace["events"] == [*oracle["events"], row]
    assert trace["end"] == {**oracle["end"], "in_flight": {}}
