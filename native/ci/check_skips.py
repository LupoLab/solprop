#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fail on a skipped test whose reason isn't an expected one (SEP 43, section 7).

A test that skips because native is missing, or short of something, otherwise passes
silently. ``python native/ci/check_skips.py junit.xml [...]`` reads pytest's JUnit files
and exits 1, naming every test skipped for a reason ``EXPECTED`` doesn't list. To accept
a new reason, add it here with why it can't run.

Standard library only, so that it runs in any environment that holds the files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree

__all__ = ["EXPECTED", "main", "unexpected_skips"]

# Each reason in full, except for the commit hash, which varies.
EXPECTED: tuple[str, ...] = (
    # Windows has no POSIX processes, signals, locks or directory modes.
    r"the Queue executor is POSIX only and refuses Windows",
    r"the Slurm \(and its Queue\) executor is POSIX only and refuses Windows",
    r"the Queue and Slurm executors are POSIX only",
    r"POSIX process signal delivery",
    r"POSIX SIGKILL",
    r"POSIX shared locks",
    r"Windows has no SIGKILL",
    r"a read-only directory mode is not enforced on Windows",
    # A platform's own branch, on the other platforms.
    r"the real Windows query",
    # Other backends and packages a lane doesn't install.
    r"mlx is not installed",
    r"could not import 'mlx\.core': .*",
    r"this GPU computes float32 products even with TF32 enabled",
    r"Pyroraptor at the pinned revision is not installed",
    # A lane on native has its wheel; these are the no-native lane's.
    r"solprop-native is installed",
    # Parametrized cases that apply to one backend's route only.
    r"numpy_ref executes every schema-12 leaf in one combination",
    r"the Kokkos route's parity; numpy_ref is its oracle",
    r"Kokkos's propagator scratch has one fixed residency policy",
    r"native forms a staged trial's stage scalars in its own state and scores the "
    r"alarm there \(SEP 45 phase 2\): no host bundle carries the stage",
    # The parametrization reaches a case its subject excludes.
    r"the interaction formulation is refused beside a coupled flow",
    r"this case took the expansion",
    r"this case took the constant fast path",
    r"a process SEP is structured by its subject",
    # A build whose target or compiler does not fuse (a device build, or a baseline
    # without fused multiply-add) cannot show a contraction; a distribution wheel for a
    # fusing baseline fails instead (tests/backends/test_native_strict_witness.py).
    r"this build does not fuse a product and a sum, so the witness proves nothing",
    # Settings of the lane: ci.yml's coverage lane runs the type checks.
    r"the runtime type checks are off \(SOLPROP_TEST_TYPECHECK=0\)",
    # A record cites a commit that a squash removed from the history.
    r"[0-9a-f]{12} is not in this checkout \(squashed or rebased\)",
    # pytest-xdist's report for a collected module with nothing to run.
    r"collection skipped",
)


def _reason(element: ElementTree.Element) -> str:
    """Return a skip's reason, without pytest's ``Skipped:`` prefix."""
    message = (element.get("message") or "").strip()
    return message.removeprefix("Skipped:").strip()


def unexpected_skips(path: Path) -> list[tuple[str, str]]:
    """Return ``(test, reason)`` for each skip in a JUnit file that isn't expected."""
    found = []
    for case in ElementTree.parse(path).getroot().iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None or skipped.get("type") == "pytest.xfail":
            continue
        reason = _reason(skipped)
        if not any(re.fullmatch(pattern, reason) for pattern in EXPECTED):
            name = f"{case.get('classname')}::{case.get('name')}"
            found.append((name, reason))
    return found


def main(argv: list[str] | None = None) -> int:
    """Check every file named; print each unexpected skip; return 1 if there are any."""
    paths = [Path(item) for item in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        print("check_skips: name at least one JUnit file", file=sys.stderr)
        return 2
    found = [item for path in paths for item in unexpected_skips(path)]
    for name, reason in found:
        print(f"unexpected skip: {name}: {reason}")
    print(f"check_skips: {len(found)} unexpected skip(s) in {len(paths)} file(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
