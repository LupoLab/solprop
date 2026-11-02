# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which CI artifact ``poe fetch-wheel`` installs for a checkout."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "native" / "scripts" / "fetch_wheel.py"
spec = importlib.util.spec_from_file_location("fetch_wheel", SCRIPT)
assert spec is not None and spec.loader is not None
fetch_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_wheel)


def artifact(sha, created, *, expired=False):
    return {
        "expired": expired,
        "created_at": created,
        "workflow_run": {"head_sha": sha, "id": 1},
    }


def test_the_newest_artifact_in_the_history_wins():
    artifacts = [artifact("old", "2026-11-01"), artifact("new", "2026-11-02")]
    assert fetch_wheel.newest_in_history(artifacts, lambda _: True) is artifacts[1]


def test_a_commit_outside_the_history_is_passed_over():
    artifacts = [artifact("mine", "2026-11-01"), artifact("other_pr", "2026-11-02")]
    assert (
        fetch_wheel.newest_in_history(artifacts, lambda sha: sha == "mine")
        is artifacts[0]
    )


def test_an_expired_artifact_is_passed_over():
    artifacts = [
        artifact("old", "2026-11-01"),
        artifact("new", "2026-11-02", expired=True),
    ]
    assert fetch_wheel.newest_in_history(artifacts, lambda _: True) is artifacts[0]


def test_no_match_gives_none():
    assert (
        fetch_wheel.newest_in_history([artifact("x", "2026-11-01")], lambda _: False)
        is None
    )
