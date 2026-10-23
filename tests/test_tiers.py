# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every test is in exactly one tier, whichever lane is running.

``docs/dev/testing.md`` section 2 fixes three tiers -- fast (no marker), ``medium`` and
``slow`` -- beside the two selections that are lanes rather than tiers, ``gpu`` and
``benchmark``. A test carrying two of them would run in two lanes and be budgeted in
neither, and the budgets the tiers exist to hold would stop meaning anything.

The recording hook is in ``tests/conftest.py`` and runs before the ``-m``
deselection, so it sees the whole collected suite: this file checks the slow lane from
inside the default one.
"""

import pytest

TIERS = ("medium", "slow", "gpu", "benchmark")

# Below this many files the run is a single module or a hand-picked selection, not the
# suite, and the hook has nothing to be complete about.
SUITE_FILES = 5


def files_of(collected):
    return {node.split("::")[0] for node in collected}


def test_every_collected_test_is_in_exactly_one_tier(collected_tiers):
    doubled = {node: tiers for node, tiers in collected_tiers.items() if len(tiers) > 1}
    assert doubled == {}


def test_the_recording_hook_sees_the_whole_suite_and_not_only_this_lane(
    collected_tiers,
):
    """Otherwise the check above would pass vacuously on an empty or partial map."""
    assert any(
        node.endswith("::test_every_collected_test_is_in_exactly_one_tier")
        for node in collected_tiers
    )
    seen = {tier for tiers in collected_tiers.values() for tier in tiers}
    assert seen <= set(TIERS)
    if len(files_of(collected_tiers)) < SUITE_FILES:
        pytest.skip("a hand-picked selection collects too little to be complete about")
    # `slow` is deselected in the lane this test normally runs in, so seeing it here is
    # what says the hook ran before the deselection rather than after it.
    assert {"medium", "slow"} <= seen
