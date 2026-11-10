# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Runge-Kutta march against its decision traces (SEP 45 section 7.2).

``tests/data/rk_traces.json.gz`` holds the decisions the march made, on each
scenario's scripted norms, before SEP 45 refactored it into a pure controller core
(``tests/backends/rk_traces.py``). The march must reproduce every one exactly: every
step it asked for, every acceptance, proposal, landing and norm history, the caps and
the start-up rule, natural stops, and how it stopped. Exactly on glibc, which recorded
it; elsewhere the numbers to the C library's rounding of ``pow`` (``same_trace``).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from tests.backends.rk_traces import Scenario, run_scenario, same_trace

CORPUS = Path(__file__).parents[1] / "data" / "rk_traces.json.gz"


def _corpus() -> dict[str, Any]:
    with gzip.open(CORPUS, "rt") as handle:
        return json.load(handle)


ENTRIES = _corpus()["scenarios"]


def test_the_corpus_was_generated_before_the_refactor():
    # The feature branch's head with #566 merged, before rk_core existed.
    assert _corpus()["generated_from"] == "a0d9691231bf"


def test_the_corpus_reaches_every_decision_it_was_written_for():
    stops = {
        entry["scenario"]["name"]: entry["trace"]["end"]["stop"] for entry in ENTRIES
    }
    for name in ("floor_on_rejection", "floor_at_start_up", "floor_on_proposal"):
        assert stops[name] == "ConvergenceError", name
    for name in ("default", "caps_at_landings", "natural_stops", "replay"):
        assert stops[name] == "done", name


@pytest.mark.parametrize("host", [True, False], ids=["host", "bridge"])
@pytest.mark.parametrize(
    "entry", ENTRIES, ids=[entry["scenario"]["name"] for entry in ENTRIES]
)
def test_the_march_reproduces_its_trace(entry, host):
    # The reference's own driver and the native bridge's copy (SEP 47 phase 2) both
    # reproduce the corpus, so the reference's decisions stay the native march's.
    scenario = Scenario.from_json(entry["scenario"])
    assert same_trace(run_scenario(scenario, host=host), entry["trace"])
