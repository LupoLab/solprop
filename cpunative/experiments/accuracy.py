# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze convergence envelopes before inspecting a candidate trajectory.

Use ``freeze`` on three independent reference trajectories, then ``check`` on
candidate/native runs. The policy is fixed here, not fitted to candidate errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def freeze(paths: list[Path], destination: Path) -> None:
    refs = [np.load(p, allow_pickle=False) for p in paths]
    for ref in refs[1:]:
        np.testing.assert_array_equal(ref["z"], refs[0]["z"])
    fields = [ref["field"] for ref in refs]
    scale = max(float(np.linalg.norm(field)) for field in fields)
    coarse = float(np.linalg.norm(fields[0] - fields[1])) / scale
    fine = float(np.linalg.norm(fields[1] - fields[2])) / scale
    # Three successive decade reductions of local tolerance must reduce the
    # observed global field error by at least two. Otherwise refine again.
    if not (fine < 0.5 * coarse):
        raise ValueError(f"reference has not converged: {coarse=}, {fine=}")
    # Transform/scan rounding floor, separate from the adaptive accuracy envelope.
    metadata = json.loads(paths[-1].with_suffix(".json").read_text(encoding="utf-8"))
    floor = 64 * np.finfo(float).eps * metadata["n_to"]
    budgets = {}
    for name in ["field", *sorted(k for k in refs[0].files if k.startswith("stat_"))]:
        values = [ref[name] for ref in refs]
        magnitude = max(float(np.nanmax(np.abs(value))) for value in values)
        if name == "field":
            magnitude = max(
                float(np.linalg.norm(v.reshape(len(v), -1), axis=1).max())
                for v in values
            )
            distance = max(
                float(
                    np.linalg.norm((v - values[-1]).reshape(len(v), -1), axis=1).max()
                )
                for v in values[:-1]
            )
        else:
            distance = max(
                float(np.nanmax(np.abs(v - values[-1]))) for v in values[:-1]
            )
        # Four times the observed standard-to-converged discrepancy, with a
        # positive rounding floor even for exactly unchanged reference outputs.
        budgets[name] = max(4 * distance, floor * max(magnitude, np.finfo(float).tiny))
    record = dict(
        reference=str(paths[-1].resolve()),
        policy="four-times-reference-discrepancy-v1",
        coarse_difference=coarse,
        fine_difference=fine,
        budgets=budgets,
    )
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def check(envelope: Path, candidate: Path) -> None:
    policy = json.loads(envelope.read_text(encoding="utf-8"))
    reference = np.load(policy["reference"], allow_pickle=False)
    actual = np.load(candidate, allow_pickle=False)
    np.testing.assert_array_equal(actual["z"], reference["z"])
    ratios = {}
    for name, budget in policy["budgets"].items():
        difference = actual[name] - reference[name]
        if name == "field":
            error = float(
                np.linalg.norm(difference.reshape(len(difference), -1), axis=1).max()
            )
        else:
            error = float(np.nanmax(np.abs(difference)))
        ratios[name] = error / budget
        if not np.array_equal(np.isnan(actual[name]), np.isnan(reference[name])):
            raise ValueError(f"{name}: NaN locations differ")
        if not np.isfinite(error) or error > budget:
            raise ValueError(f"{name}: {error=} exceeds {budget=}")
    print(json.dumps(dict(candidate=str(candidate), ratios=ratios)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("freeze")
    make.add_argument("references", nargs=3, type=Path)
    make.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser("check")
    compare.add_argument("envelope", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.references, args.out)
    else:
        check(args.envelope, args.candidate)


if __name__ == "__main__":
    main()
