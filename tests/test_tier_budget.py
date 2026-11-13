# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The tier budgets are read from JUnit reports that carry each test's tier."""

import pytest

import tier_budget
from tier_budget import FAST_EACH, MEDIUM_EACH, PROPERTY, Case


def report(tmp_path, cases, name="junit.xml"):
    """Write a JUnit file in the layout pytest's legacy family writes."""
    rows = []
    for node, tier, seconds, *skipped in cases:
        path, _, test = node.partition("::")
        classname = path.removesuffix(".py").replace("/", ".")
        properties = (
            f'<properties><property name="{PROPERTY}" value="{tier}"/></properties>'
            if tier is not None
            else ""
        )
        skip = '<skipped message="no device"/>' if skipped else ""
        rows.append(
            f'<testcase classname="{classname}" name="{test}" file="{path}" '
            f'time="{seconds}">{properties}{skip}</testcase>'
        )
    path = tmp_path / name
    path.write_text(f"<testsuites><testsuite>{''.join(rows)}</testsuite></testsuites>")
    return path


def test_every_report_carries_the_tests_tier(request):
    """The conftest hook records the tier this test is in: none, so fast."""
    assert (PROPERTY, "fast") in request.node.user_properties
    assert tier_budget.tier_of(()) == "fast"
    assert tier_budget.tier_of(("medium",)) == "medium"


def test_read_takes_the_node_the_tier_and_the_seconds(tmp_path):
    path = report(
        tmp_path,
        [
            ("tests/a/test_b.py::test_c[1]", "fast", 0.5),
            ("tests/test_d.py::test_e", "medium", 7.25),
            ("tests/test_d.py::test_f", "gpu", 0.0, "skipped"),
            ("tests/test_d.py::test_g", None, 1.0),
        ],
    )
    assert tier_budget.read([path]) == [
        Case("tests/a/test_b.py::test_c[1]", "fast", 0.5),
        Case("tests/test_d.py::test_e", "medium", 7.25),
        Case("tests/test_d.py::test_g", None, 1.0),
    ]


def test_read_names_a_class_and_a_windows_path_as_node_ids_do(tmp_path):
    path = tmp_path / "windows.xml"
    path.write_text(
        '<testsuites><testsuite><testcase classname="tests.a.test_b.TestC" '
        'name="test_d" file="tests\\a\\test_b.py" time="2.0"/></testsuite></testsuites>'
    )
    (case,) = tier_budget.read([path])
    assert case.node == "tests/a/test_b.py::TestC::test_d"


RATCHET = 100.0
"""A medium tier's ratchet for these tests, in seconds."""


def test_the_committed_ratchet_is_a_positive_total():
    assert tier_budget.load_ratchet() > 0


def test_a_fast_test_of_a_second_or_more_fails_and_is_named():
    cases = [
        Case("a.py::slow_fast", "fast", FAST_EACH),
        Case("a.py::quick", "fast", FAST_EACH / 2),
        Case("c.py::untiered", None, 9.0),
    ]
    verdict = tier_budget.judge(cases, RATCHET)
    assert [case.node for case in verdict.slow_fast] == ["a.py::slow_fast"]
    assert verdict.totals == {"fast": (2, 1.5 * FAST_EACH)}
    assert verdict.untiered == 1
    assert not verdict.within
    text = tier_budget.render(verdict, "macos-arm64")
    assert f"1 tests take {FAST_EACH:g} s or more" in text
    assert "`a.py::slow_fast`" in text
    assert "1 tests carry no tier property" in text


def test_a_medium_test_of_30_s_or_more_fails_within_the_ratchet():
    cases = [
        Case("b.py::long", "medium", MEDIUM_EACH),
        Case("b.py::short", "medium", MEDIUM_EACH / 2),
    ]
    verdict = tier_budget.judge(cases, 2 * MEDIUM_EACH)
    assert [case.node for case in verdict.slow_medium] == ["b.py::long"]
    assert verdict.medium_seconds <= verdict.medium_ratchet
    assert not verdict.within
    text = tier_budget.render(verdict, "macos-arm64")
    assert f"1 tests take {MEDIUM_EACH:g} s or more" in text
    assert "`b.py::long`" in text
    assert "which is within it." in text


def test_a_medium_total_over_the_ratchet_fails_with_every_test_short():
    each = MEDIUM_EACH / 2
    cases = [Case(f"b.py::t{n}", "medium", each) for n in range(3)]
    verdict = tier_budget.judge(cases, 3 * each - 1)
    assert not verdict.slow_medium
    assert verdict.medium_seconds == 3 * each
    assert not verdict.within
    assert f"{3 * each:.1f} s, which exceeds it." in tier_budget.render(verdict, "a")


def test_a_lane_within_its_budgets_says_so(tmp_path):
    path = report(
        tmp_path,
        [
            ("a.py::t", "fast", FAST_EACH / 2),
            ("b.py::t", "medium", MEDIUM_EACH / 2),
            ("b.py::u", "medium", MEDIUM_EACH / 2),
        ],
    )
    verdict = tier_budget.judge(tier_budget.read([path]), MEDIUM_EACH)
    assert verdict.medium_seconds == verdict.medium_ratchet
    assert verdict.within
    text = tier_budget.render(verdict, "a lane")
    assert f"0 tests take {FAST_EACH:g} s or more." in text
    assert f"0 tests take {MEDIUM_EACH:g} s or more." in text
    assert "which is within it." in text


@pytest.mark.parametrize(("enforce", "status"), [(False, 0), (True, 1)])
def test_main_reports_and_fails_only_when_asked(tmp_path, capsys, enforce, status):
    ratchet = tmp_path / "ratchet.toml"
    ratchet.write_text(f"[medium]\nmax_total = {RATCHET:g}\n")
    one = report(tmp_path, [("b.py::t", "medium", RATCHET / 4)], "one.xml")
    two = report(tmp_path, [("c.py::t", "medium", RATCHET)], "two.xml")
    arguments = [str(one), str(two), "--title", "shards", "--ratchet", str(ratchet)]
    assert tier_budget.main([*arguments, "--enforce"] if enforce else arguments) == (
        status
    )
    out = capsys.readouterr().out
    assert out.startswith("### Tier budgets: shards")
    assert f"ratchet's {RATCHET:g} s: {1.25 * RATCHET:.1f} s, which exceeds it." in out
