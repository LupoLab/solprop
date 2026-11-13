# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold the test tiers to the budgets of ``docs/dev/testing.md`` section 2, from JUnit.

``tests/conftest.py`` writes each test's tier into its report as the JUnit property
``tier`` (``fast`` for a test without a tier marker). Run as a script, this module reads
the JUnit reports of one lane -- the shards of one platform, which together hold each
test once -- and prints, as Markdown, what the budgets say of them::

    uv run poe test --junitxml=report.xml
    uv run python tests/tier_budget.py report.xml

The budgets are the page's, as written: a fast test takes **under a second each**, and
the medium tier takes **60 s for the tier**, the sum of its tests' times, which is what
``poe test-medium`` takes when it runs them one after another. A test's time is
JUnit's: its setup, call and teardown, so a module fixture counts against the first test
that builds it. The slow lane's budget is a lane's wall time, which no report of the
default lane holds. Skipped tests cost nothing and are left out.

The report is advisory: it exits 0 whatever it finds, unless ``--enforce`` asks for a
failing status when a budget is exceeded. Standard library only, so that it runs in any
environment that holds the files.
"""

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

__all__ = [
    "FAST_EACH",
    "MEDIUM_TOTAL",
    "PROPERTY",
    "Case",
    "Verdict",
    "judge",
    "main",
    "read",
    "render",
    "tier_of",
]

PROPERTY = "tier"
"""The name of the JUnit property that carries a test's tier."""

FAST_EACH = 1.0
"""The fast tier's budget: under a second each, in seconds."""

MEDIUM_TOTAL = 60.0
"""The medium tier's budget: 60 s for the tier, in seconds."""

SHOWN = 15
"""How many of the slowest tests a report names per tier."""


def tier_of(markers: Sequence[str]) -> str:
    """Return the tier a test's tier markers put it in: its one marker, or ``fast``."""
    return markers[0] if markers else "fast"


@dataclass(frozen=True)
class Case:
    """One test as a report recorded it: its node, its tier and its seconds."""

    node: str
    tier: str | None
    seconds: float


@dataclass(frozen=True)
class Verdict:
    """What the budgets say of one lane's cases."""

    totals: dict[str, tuple[int, float]]
    slow_fast: tuple[Case, ...]
    medium: tuple[Case, ...]
    untiered: int

    @property
    def medium_seconds(self) -> float:
        """The medium tier's summed seconds."""
        return self.totals.get("medium", (0, 0.0))[1]

    @property
    def within(self) -> bool:
        """Whether every budget holds."""
        return not self.slow_fast and self.medium_seconds <= MEDIUM_TOTAL


def _node(case: ElementTree.Element) -> str:
    """Return a test case's node ID as pytest prints it, from its JUnit attributes."""
    path = (case.get("file") or "").replace("\\", "/")
    module = path.removesuffix(".py").replace("/", ".")
    classname = case.get("classname") or ""
    inner = classname.removeprefix(module).lstrip(".") if path else ""
    return "::".join([path, *filter(None, inner.split(".")), case.get("name") or ""])


def read(reports: Iterable[Path]) -> list[Case]:
    """Return the cases that ran in `reports`, skipped ones left out."""
    cases = []
    for report in reports:
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            if case.find("skipped") is not None:
                continue
            tier = next(
                (
                    item.get("value")
                    for item in case.iter("property")
                    if item.get("name") == PROPERTY
                ),
                None,
            )
            cases.append(Case(_node(case), tier, float(case.get("time") or 0.0)))
    return cases


def judge(cases: Sequence[Case]) -> Verdict:
    """Return the per-tier counts and seconds, and the cases each budget names."""
    totals: dict[str, tuple[int, float]] = {}
    for case in cases:
        if case.tier is not None:
            count, seconds = totals.get(case.tier, (0, 0.0))
            totals[case.tier] = (count + 1, seconds + case.seconds)
    slowest = sorted(cases, key=lambda case: (-case.seconds, case.node))
    return Verdict(
        totals=dict(sorted(totals.items())),
        slow_fast=tuple(
            case
            for case in slowest
            if case.tier == "fast" and case.seconds >= FAST_EACH
        ),
        medium=tuple(case for case in slowest if case.tier == "medium"),
        untiered=sum(case.tier is None for case in cases),
    )


def _table(cases: Sequence[Case]) -> list[str]:
    """Return a Markdown table of the first :data:`SHOWN` cases, after a blank line."""
    rows = [f"| {case.seconds:.1f} | `{case.node}` |" for case in cases[:SHOWN]]
    return ["", "| s | test |", "|---:|---|", *rows] if rows else []


def render(verdict: Verdict, title: str) -> str:
    """Return the verdict as Markdown, headed by `title`."""
    lines = [f"### Tier budgets: {title}", ""]
    lines += ["| tier | tests | seconds |", "|---|---:|---:|"]
    lines += [
        f"| {tier} | {count} | {seconds:.1f} |"
        for tier, (count, seconds) in verdict.totals.items()
    ]
    if verdict.untiered:
        lines += ["", f"{verdict.untiered} tests carry no tier property."]
    fast = verdict.slow_fast
    lines += [
        "",
        f"**fast**, under {FAST_EACH:g} s each: {len(fast)} tests take longer"
        + (", the slowest:" if fast else "."),
        *_table(fast),
    ]
    medium = verdict.medium_seconds
    over = "exceeds" if medium > MEDIUM_TOTAL else "is within"
    lines += [
        "",
        f"**medium**, {MEDIUM_TOTAL:g} s for the tier: {medium:.1f} s, which {over} "
        f"it" + (", the slowest:" if verdict.medium else "."),
        *_table(verdict.medium),
    ]
    return "\n".join(lines) + "\n"


def main(arguments: Sequence[str] | None = None) -> int:
    """Print the report of the given JUnit files; return the exit status."""
    parser = argparse.ArgumentParser(
        description="Report the test tiers against their budgets."
    )
    parser.add_argument("reports", nargs="+", type=Path, help="one lane's JUnit files")
    parser.add_argument("--title", default="the lane", help="what the reports are of")
    parser.add_argument(
        "--enforce", action="store_true", help="exit 1 when a budget is exceeded"
    )
    options = parser.parse_args(arguments)
    verdict = judge(read(options.reports))
    sys.stdout.write(render(verdict, options.title))
    return 1 if options.enforce and not verdict.within else 0


if __name__ == "__main__":
    raise SystemExit(main())
