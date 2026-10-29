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

CORPUS = Path(__file__).parents[1] / "data" / "adams_traces.json.gz"


def _corpus() -> list[dict[str, Any]]:
    with gzip.open(CORPUS, "rt") as handle:
        return json.load(handle)["scenarios"]


ENTRIES = _corpus()


def test_the_corpus_was_generated_before_the_refactor():
    with gzip.open(CORPUS, "rt") as handle:
        assert json.load(handle)["generated_from"] == "a500d2d"


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=[entry["scenario"]["name"] for entry in ENTRIES]
)
def test_the_march_reproduces_its_trace(entry):
    scenario = Scenario.from_json(entry["scenario"])
    assert run_scenario(scenario) == entry["trace"]
