# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CI shards: every module in exactly one, dealt by weight, the same every time."""

import json
from types import SimpleNamespace

import pytest

import shards


def items(*nodeids):
    """Return stand-ins for collected items, which the deal reads the node IDs of."""
    return [SimpleNamespace(nodeid=nodeid) for nodeid in nodeids]


@pytest.fixture
def weights(monkeypatch, tmp_path):
    """Point the deal at a weights file of the test's own."""
    path = tmp_path / "shard_weights.json"
    monkeypatch.setattr(shards, "WEIGHTS", path)

    def write(seconds):
        path.write_text(json.dumps({"seconds": seconds}), encoding="utf-8")

    return write


def test_parse_reads_a_one_based_shard():
    assert shards.parse("1/4") == (0, 4)
    assert shards.parse("4/4") == (3, 4)


@pytest.mark.parametrize("value", ["", "2", "a/b", "0/4", "5/4", "1/2/3"])
def test_parse_refuses_anything_but_i_of_n(value):
    with pytest.raises(ValueError, match="SOLPROP_TEST_SHARD"):
        shards.parse(value)


def test_module_of_is_the_file_part():
    assert shards.module_of("tests/a/test_b.py::test_c[1]") == "tests/a/test_b.py"
    assert shards.module_of("src/solprop/grid.py") == "src/solprop/grid.py"


def test_every_module_lands_in_exactly_one_shard(weights):
    weights({})
    collected = items(*(f"tests/test_{index}.py::test" for index in range(11)))
    dealt = shards.deal(collected, 3)
    assert sorted(module for shard in dealt for module in shard) == sorted(
        {shards.module_of(item.nodeid) for item in collected}
    )
    assert all(dealt)


def test_the_longest_modules_are_dealt_first_to_the_lightest_shard(weights):
    weights({"a.py": 10.0, "b.py": 6.0, "c.py": 5.0, "d.py": 4.0})
    dealt = shards.deal(items("a.py::t", "b.py::t", "c.py::t", "d.py::t"), 2)
    assert dealt == [{"a.py", "d.py"}, {"b.py", "c.py"}]


def test_an_unweighed_module_counts_as_the_median(weights):
    weights({"a.py": 1.0, "b.py": 2.0, "c.py": 30.0})
    dealt = shards.deal(items("a.py::t", "b.py::t", "c.py::t", "new.py::t"), 2)
    assert dealt == [{"c.py"}, {"a.py", "b.py", "new.py"}]


def test_without_weights_the_deal_is_still_deterministic(weights):
    collected = items("b.py::t", "a.py::t", "c.py::t")
    assert shards.deal(collected, 2) == shards.deal(list(reversed(collected)), 2)


def test_weigh_sums_each_module_and_keeps_the_largest_report(tmp_path):
    def report(name, cases):
        rows = "".join(
            f'<testcase file="{path}" name="t" time="{time}"/>' for path, time in cases
        )
        path = tmp_path / name
        path.write_text(f"<testsuites><testsuite>{rows}</testsuite></testsuites>")
        return path

    first = report("one.xml", [("a.py", 1.0), ("a.py", 2.5), ("b.py", 1.0), ("", 9)])
    second = report("two.xml", [("a.py", 1.0), ("b.py", 4.0)])
    assert shards.weigh([first, second]) == {"a.py": 3.5, "b.py": 4.0}


def test_the_committed_weights_name_test_files():
    seconds = json.loads(shards.WEIGHTS.read_text(encoding="utf-8"))["seconds"]
    assert seconds
    assert all(path.endswith(".py") and value >= 0 for path, value in seconds.items())
