# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Apply SEP 30's descriptive comparison rules to complete persisted selections."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from benchmarks.records import Record, distribution

__all__ = [
    "aa_control",
    "clean_pairs",
    "compare",
    "compare_configuration",
    "completed",
    "geometric",
    "needs_more",
    "noisy",
    "repeatability",
    "resource_flags",
]


def completed(rows: Sequence[Record]) -> list[Record]:
    """Select the latest successful attempt per sample, retaining failures on disk."""
    selected: dict[str, Record] = {}
    for row in rows:
        if row.get("phase", "measurement") != "measurement":
            continue
        previous = selected.get(row["sample_id"])
        if previous is None or row["status"] == "ok" or previous["status"] != "ok":
            selected[row["sample_id"]] = row
    return list(selected.values())


def clean_pairs(
    baseline: Sequence[Record], candidate: Sequence[Record]
) -> tuple[list[Record], list[Record]]:
    """Drop both members of a contaminated pair without joining round indices."""
    left = {
        row["round"]: row
        for row in baseline
        if row.get("clean") and row["status"] == "ok"
    }
    right = {
        row["round"]: row
        for row in candidate
        if row.get("clean") and row["status"] == "ok"
    }
    rounds = sorted(left.keys() & right.keys())
    return [left[index] for index in rounds], [right[index] for index in rounds]


def repeatability(rows: Sequence[Record]) -> float | None:
    """Use only adjacent original clean rounds, with at least four differences."""
    times = {row["round"]: row["seconds"] for row in rows}
    changes = [
        abs(math.log(times[index + 1] / value))
        for index, value in times.items()
        if index + 1 in times
    ]
    return 2 * statistics.median(changes) if len(changes) >= 4 else None


def noisy(rows: Sequence[Record]) -> bool:
    """Decide whether more complete standard rounds are required."""
    values = distribution([row["seconds"] for row in rows])
    return values.count < 5 or (
        values.mad is not None
        and values.median is not None
        and 2 * values.mad / values.median > 0.05
    )


def needs_more(
    rows: Sequence[Record], configuration_ids: Sequence[str], arms: tuple[str, ...]
) -> bool:
    """Extend for dispersion, eligibility or unavailable adjacent-round evidence."""
    active = completed(rows)
    for configuration in configuration_ids:
        groups = [
            [
                row
                for row in active
                if row["configuration_id"] == configuration and row["arm"] == arm
            ]
            for arm in arms
        ]
        if any(
            group and sum(not row.get("clean") for row in group) / len(group) > 0.2
            for group in groups
        ):
            return True
        if len(groups) == 2:
            groups[0], groups[1] = clean_pairs(*groups)
            if repeatability(groups[0]) is None:
                return True
        else:
            groups = [
                [row for row in group if row.get("clean") and row["status"] == "ok"]
                for group in groups
            ]
        if any(noisy(group) for group in groups):
            return True
    return False


def resource_flags(baseline: Sequence[Record], candidate: Sequence[Record]) -> Record:
    """Compare matched sample-count memory maxima and startup medians separately."""
    result: Record = {}
    for name, reduction, threshold in (
        ("peak_rss_bytes", max, 0.05),
        ("startup_seconds", statistics.median, 0.1),
    ):
        left = [row[name] for row in baseline if row.get(name) is not None]
        right = [row[name] for row in candidate if row.get(name) is not None]
        if not left or len(left) != len(right):
            result[name] = {
                "status": "unavailable",
                "reason": "missing telemetry or unmatched counts",
            }
            continue
        before, after = reduction(left), reduction(right)
        change = after - before
        ratio = after / before if before else None
        result[name] = {
            "baseline": before,
            "candidate": after,
            "difference": change,
            "ratio": ratio,
            "flag": change > 0
            and (
                (ratio is not None and ratio > 1 + threshold)
                or (name == "peak_rss_bytes" and change > 67108864)
            ),
        }
    return result


def compare_configuration(
    baseline: Sequence[Record],
    candidate: Sequence[Record],
    *,
    paired: bool = True,
    aa_band: float | None = None,
) -> Record:
    """Evaluate one configuration without hiding failures or contamination."""
    if any(
        row["status"] == "invalid" or row.get("accuracy", {}).get("status") == "invalid"
        for row in (*baseline, *candidate)
    ):
        return {"status": "numerically-invalid"}
    left, right = completed(baseline), completed(candidate)
    if (
        any(row["status"] == "refused" for row in right)
        and any(row["status"] == "ok" for row in left)
        and not any(row["status"] == "refused" for row in left)
    ):
        return {
            "status": "regression",
            "resource_regression": (
                "candidate memory admission refused; baseline succeeded"
            ),
        }
    if not left or not right or any(row["status"] != "ok" for row in (*left, *right)):
        return {"status": "incomplete"}
    if any(
        row.get("resolution_hash") != left[0].get("resolution_hash")
        for row in (*left, *right)
    ):
        return {
            "status": "incompatible",
            "reason": "resolved method, inputs or output differ",
        }
    references = {
        row.get("accuracy", {}).get("reference_id")
        for row in (*left, *right)
        if row.get("accuracy", {}).get("reference_id") is not None
    }
    if len(references) > 1:
        return {
            "status": "incompatible",
            "reason": (
                "reference or accuracy-policy revision differs; revalidate both arms"
            ),
        }
    if paired:
        clean_left, clean_right = clean_pairs(left, right)
    else:
        clean_left, clean_right = (
            [row for row in group if row.get("clean")] for group in (left, right)
        )
    band = repeatability(clean_left) if paired else aa_band
    before = distribution([row["seconds"] for row in clean_left])
    after = distribution([row["seconds"] for row in clean_right])
    ratio = after.median / before.median if before.median and after.median else None
    conservative = None
    if (
        before.median is not None
        and before.mad is not None
        and after.median is not None
        and after.mad is not None
    ):
        denominator = before.median - 2 * before.mad
        if denominator > 0:
            conservative = (after.median + 2 * after.mad) / denominator
    eligible = (
        min(before.count, after.count) >= 5
        and not noisy(clean_left)
        and not noisy(clean_right)
        and all(
            sum(not row.get("clean", False) for row in group) / len(group) <= 0.2
            for group in (left, right)
        )
    )
    qualified = all(
        row.get("accuracy", {}).get("status") == "passed"
        and row.get("accuracy", {}).get("reference_id") is not None
        and row.get("accuracy", {}).get("qualified_for_adoption", False)
        and not row.get("warning_unreviewed")
        and not row.get("estimator_defect")
        for row in (*left, *right)
    )
    status = (
        "eligible"
        if eligible and qualified and band is not None and conservative is not None
        else "unresolved"
    )
    return {
        "status": status,
        "ratio": ratio,
        "conservative_ratio": conservative,
        "repeatability_band": band,
        "clean_counts": [before.count, after.count],
        "all_counts": [len(left), len(right)],
        "qualified": qualified,
        "timing_regression": ratio is not None and ratio > 1.1,
        "all_resources": resource_flags(left, right),
        "clean_resources": resource_flags(clean_left, clean_right),
        "all_ratio": statistics.median(row["seconds"] for row in right)
        / statistics.median(row["seconds"] for row in left),
    }


def geometric(values: Sequence[float]) -> float:
    """Give every declared member equal weight."""
    return math.exp(statistics.fmean(math.log(value) for value in values))


def aa_control(plan: Record, rows: Sequence[Record]) -> Record:
    """Retain matched clean A/A log-ratio bands for compatible later comparisons."""
    bands = {}
    for configuration in plan["rows"]:
        members = [
            row
            for row in completed(rows)
            if row["configuration_id"] == configuration["id"]
        ]
        left, right = clean_pairs(
            [row for row in members if row["arm"] == "baseline"],
            [row for row in members if row["arm"] == "candidate"],
        )
        eligible = (
            plan["mode"] == "standard"
            and min(len(left), len(right)) >= 5
            and all(
                sum(not row.get("clean") for row in members if row["arm"] == arm)
                <= 0.2 * sum(row["arm"] == arm for row in members)
                for arm in ("baseline", "candidate")
            )
        )
        ratios = [b["seconds"] / a["seconds"] for a, b in zip(left, right, strict=True)]
        bands[configuration["id"]] = {
            "status": "measured" if eligible else "unresolved",
            "clean_pairs": len(left),
            "median_ratio": statistics.median(ratios) if ratios else None,
            "band": 2 * statistics.median(abs(math.log(value)) for value in ratios)
            if eligible
            else None,
        }
    return {
        "profile": plan["profile"],
        "baseline": plan["environments"]["baseline"],
        "configurations": bands,
    }


def compare(
    plan: Record,
    rows: Sequence[Record],
    *,
    paired: bool = True,
    aa_bands: Record | None = None,
) -> Record:
    """Compare the complete selected matrix, with separate production aggregates."""
    details = []
    for row in plan["rows"]:
        configuration = row["configuration"]
        members = [item for item in rows if item["configuration_id"] == row["id"]]
        verdict = compare_configuration(
            [item for item in members if item["arm"] == "baseline"],
            [item for item in members if item["arm"] == "candidate"],
            paired=paired,
            aa_band=None
            if aa_bands is None
            else aa_bands.get(row["id"], {}).get("band"),
        )
        details.append(
            {
                "configuration_id": row["id"],
                "case": configuration["case"],
                "threads": configuration["threads"],
                "track": configuration["track"],
                **verdict,
            }
        )
    if plan["mode"] != "standard" or plan.get("exploratory"):
        return {
            "status": "exploratory",
            "reason": "descriptive comparison only; quick or overridden collection",
            "configurations": details,
        }
    for failure in ("numerically-invalid", "incompatible", "regression", "incomplete"):
        if any(row["status"] == failure for row in details):
            return {"status": failure, "configurations": details}
    aggregates = []
    for width in sorted({row["threads"] for row in details}):
        groups = {
            track: [
                row
                for row in details
                if row["threads"] == width and row["track"] == track
            ]
            for track in {row["track"] for row in details}
        }
        groups["adaptive-combined"] = [
            row
            for row in details
            if row["threads"] == width and row["track"].startswith("adaptive-")
        ]
        for track, group in groups.items():
            if not group:
                continue
            if any(row["status"] != "eligible" for row in group):
                aggregates.append(
                    {"threads": width, "track": track, "status": "unresolved"}
                )
                continue
            tracks = {item["track"] for item in group}
            weights = [
                1
                / len(tracks)
                / sum(other["track"] == item["track"] for other in group)
                for item in group
            ]
            ratio = math.exp(
                sum(
                    weight * math.log(item["ratio"])
                    for weight, item in zip(weights, group, strict=True)
                )
            )
            conservative = math.exp(
                sum(
                    weight * math.log(item["conservative_ratio"])
                    for weight, item in zip(weights, group, strict=True)
                )
            )
            band = sum(
                weight * item["repeatability_band"]
                for weight, item in zip(weights, group, strict=True)
            )
            passed = ratio < 0.95 and conservative < 1 and -math.log(ratio) > band
            aggregates.append(
                {
                    "threads": width,
                    "track": track,
                    "ratio": ratio,
                    "conservative_ratio": conservative,
                    "repeatability_band": band,
                    "status": "passed"
                    if passed
                    else "regression"
                    if ratio > 1.05
                    else "unresolved",
                }
            )
    adoption = [
        row
        for row in aggregates
        if row["threads"] == plan["profile"]["production"]
        and row["track"] == "adaptive-combined"
    ]
    if not adoption and plan.get("fixed_assessment"):
        adoption = [
            row for row in aggregates if row["threads"] == plan["profile"]["production"]
        ]
    status = (
        "unresolved"
        if not adoption
        else "passed"
        if all(row["status"] == "passed" for row in adoption)
        else (
            "regression"
            if any(row["status"] == "regression" for row in adoption)
            else "unresolved"
        )
    )
    return {"status": status, "configurations": details, "aggregates": aggregates}
