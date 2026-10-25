# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Split a test lane into shards of whole modules, balanced by their measured seconds.

CI runs a lane as several jobs, each with ``SOLPROP_TEST_SHARD=i/n``, and
``tests/conftest.py`` deselects the other shards' modules. The modules are dealt longest
first to the shard with the least work so far, by the seconds in
``tests/data/shard_weights.json``. Run as a script, this module rewrites that file from
JUnit reports of the lanes::

    uv run poe test --junitxml=report.xml
    uv run python tests/shards.py report.xml

Several reports (one per shard, or one per platform) are summed per module and the
largest total kept. The weights only balance the shards; a stale file costs time, never
coverage, since every module is in exactly one shard.
"""

import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from xml.etree import ElementTree

__all__ = ["WEIGHTS", "deal", "module_of", "parse", "weigh"]

WEIGHTS = Path(__file__).parent / "data" / "shard_weights.json"


def parse(value: str) -> tuple[int, int]:
    """Return the zero-based index and the count of shard ``"i/n"``, 1 <= i <= n."""
    try:
        index, count = (int(part) for part in value.split("/"))
    except ValueError as error:
        msg = f"SOLPROP_TEST_SHARD must be i/n, and is {value!r}"
        raise ValueError(msg) from error
    if not 1 <= index <= count:
        msg = f"SOLPROP_TEST_SHARD must have 1 <= i <= n, and is {value!r}"
        raise ValueError(msg)
    return index - 1, count


def module_of(nodeid: str) -> str:
    """Return the file part of a node ID: the unit a shard takes whole."""
    return nodeid.split("::", 1)[0]


def deal(items: Iterable[object], count: int) -> list[set[str]]:
    """Return the modules of each of `count` shards for the collected `items`."""
    modules = {module_of(item.nodeid) for item in items}  # pyright: ignore[reportAttributeAccessIssue]
    weights = _weights()
    typical = statistics.median(weights.values()) if weights else 1.0
    load = {module: weights.get(module, typical) for module in modules}
    shards: list[set[str]] = [set() for _ in range(count)]
    totals = [0.0] * count
    for module in sorted(load, key=lambda name: (-load[name], name)):
        lightest = min(range(count), key=lambda index: (totals[index], index))
        shards[lightest].add(module)
        totals[lightest] += load[module]
    return shards


def _weights() -> dict[str, float]:
    """Return the committed seconds per module (empty when the file is absent)."""
    if not WEIGHTS.is_file():
        return {}
    return json.loads(WEIGHTS.read_text(encoding="utf-8"))["seconds"]


def weigh(reports: Sequence[Path]) -> dict[str, float]:
    """Return the seconds per module of the JUnit `reports`, the largest per module."""
    totals: dict[str, float] = {}
    for report in reports:
        seconds: defaultdict[str, float] = defaultdict(float)
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            path = case.get("file")
            if path:
                # Windows reports name files with its own separator; node IDs use "/".
                seconds[path.replace("\\", "/")] += float(case.get("time") or 0.0)
        for module, value in seconds.items():
            totals[module] = max(totals.get(module, 0.0), value)
    return {module: round(totals[module], 1) for module in sorted(totals)}


if __name__ == "__main__":
    seconds = weigh([Path(argument) for argument in sys.argv[1:]])
    WEIGHTS.write_text(
        json.dumps({"seconds": seconds}, indent=1) + "\n", encoding="utf-8"
    )
