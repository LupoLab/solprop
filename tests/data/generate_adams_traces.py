# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate ``tests/data/adams_traces.json.gz``: the Adams march's decisions on scripts.

Run once, from the march as it stood before SEP 39's refactor (main a500d2d, the head
of LupoLab/solprop#56), and committed. It is the oracle the refactored Python core and
the native march must both reproduce (SEP 39 section 7.1, R1). Regenerating it from a
later march would bless whatever that march does, so the script refuses to overwrite
the file unless given ``--force``, which only a deliberate change to the controller's
algorithm, recorded in SEP 39's Part II, may use.

Usage::

    uv run python tests/data/generate_adams_traces.py [--force]
"""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.backends.adams_traces import Scenario, run_scenario

OUT = Path(__file__).with_name("adams_traces.json.gz")


def _random_norms(
    seed: int,
    count: int,
    *,
    low: float = 3e-3,
    high: float = 2.0,
    specials: bool = True,
) -> list:
    """Return `count` triples of log-uniform norms, with the specials sprinkled in."""
    rng = np.random.default_rng(seed)
    norms = np.exp(rng.uniform(math.log(low), math.log(high), size=(count, 3)))
    rows = norms.tolist()
    if specials:
        values = [
            0.0,
            1.0,
            math.nextafter(1.0, 2.0),
            math.nextafter(1.0, 0.0),
            math.inf,
        ]
        for index in rng.choice(count, size=min(count // 10, 30), replace=False):
            rows[int(index)][int(rng.integers(0, 3))] = float(rng.choice(values))
    return rows


def _scenarios() -> list[Scenario]:
    """Return the corpus's scripted marches (SEP 39 section 7.1's list)."""
    planes = [round(0.1 * k, 10) for k in range(1, 10)]
    common: dict[str, Any] = {
        "z0": 0.0,
        "proposal": 1e-4,
        "dz_min": 1e-12,
        "dz_max": 0.02,
        "z_end": 1.0,
    }
    scenarios = [
        Scenario(
            "default", {}, norms=_random_norms(1, 400), save_planes=planes, **common
        ),
        Scenario("default_one_target", {}, norms=_random_norms(2, 300), **common),
    ]
    # Fixed-order startup and post-restart growth (R1): one sample held and a norm of
    # 0.25 first, then a kink restart part-way.
    for order in range(2, 7):
        norms = [[0.25, 0.25, 0.25]] * 12 + _random_norms(10 + order, 250)
        scenarios.append(
            Scenario(
                f"fixed_order_{order}",
                {"order": order, "variable_order": False},
                norms=norms,
                save_planes=planes,
                kinks=[0.5],
                **common,
            )
        )
    scenarios += [
        Scenario(
            "variable_order_low_max",
            {"order": 2},
            norms=_random_norms(20, 300),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "collar_cap_binding",
            {},
            norms=_random_norms(21, 400, low=1e-6, high=1.5),
            save_planes=planes,
            collar_rate=50.0,
            **common,
        ),
        Scenario(
            "collar_quiet_toggle",
            {},
            norms=_random_norms(22, 400, low=1e-6, high=1.5),
            save_planes=planes,
            collar_rate=50.0,
            quiet={0: False, 5: True, 20: False, 40: True, 60: False, 90: True},
            quiet_interval=4e-3,
            **common,
        ),
        Scenario(
            "rejection_cascades",
            {},
            norms=(
                [[0.01, 0.01, 0.01]] * 15
                + [[2.0, 1.5, 3.0]] * 3
                + [[0.1, 0.1, 0.1]] * 10
                + [
                    [5.0, 5.0, 5.0],
                    [math.inf, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [9.0, 9.0, 9.0],
                ]
                + [[math.nan, 0.5, 0.5]]
                + _random_norms(23, 250)
            ),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "budget_edge",
            {"max_calls": 23},
            norms=_random_norms(24, 100),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "floor_after_rejections",
            {},
            norms=[[0.01, 0.01, 0.01]] * 3 + [[100.0, 100.0, 100.0]] * 40,
            z0=0.0,
            proposal=1e-3,
            dz_min=1e-5,
            dz_max=0.05,
            z_end=0.1,
        ),
        Scenario(
            "floor_short_landing",
            {},
            norms=[[1e-3, 1e-3, 1e-3]] * 200,
            save_planes=[0.01, 0.01 + 4e-7, 0.02],
            z0=0.0,
            proposal=1e-3,
            dz_min=1e-6,
            dz_max=0.05,
            z_end=0.1,
        ),
        Scenario(
            "equal_approach",
            {},
            norms=_random_norms(25, 300, low=1e-3, high=0.9),
            save_planes=[0.137, 0.291, 0.3, 0.613, 0.911],
            **common,
        ),
        Scenario(
            "equal_approach_off_short_landings",
            {"equal_approach": False},
            norms=_random_norms(26, 300, low=1e-3, high=0.9),
            save_planes=[0.137, 0.137 + 1e-9, 0.291, 0.291 + 1e-7, 0.613, 0.911],
            **common,
        ),
        Scenario(
            "carrier_cap",
            {},
            norms=_random_norms(27, 300, low=1e-5, high=0.9),
            save_planes=planes,
            carrier_rate=200.0,
            **common,
        ),
        Scenario(
            "nondefault_factors",
            {
                "max_growth": 1.5,
                "min_factor": 0.5,
                "safety": 0.9,
                "equal_approach": False,
            },
            norms=_random_norms(28, 300),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "thresholds",
            {},
            norms=[[0.01, 0.01, 0.01]] * 10
            + [
                [1.0, 0.5, 0.5],
                [math.nextafter(1.0, 2.0), 0.5, 0.5],
                [math.nextafter(1.0, 0.0), 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [math.nan, math.nan, math.nan],
                [0.5, math.nan, 0.5],
                [0.5, 0.5, math.inf],
            ]
            + _random_norms(29, 200),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "interval_cap_and_kinks",
            {"interval_cap": True},
            norms=_random_norms(30, 400),
            save_planes=planes,
            breakpoints=[0.0, 0.03, 0.031, 0.2, 0.5, 1.0],
            kinks=[0.03, 0.2, 0.7],
            **common,
        ),
        Scenario(
            "kinks_on_planes",
            {},
            norms=_random_norms(31, 300),
            save_planes=planes,
            kinks=[0.3, 0.45, 0.6],
            **common,
        ),
        Scenario(
            "dense_planes",
            {},
            norms=_random_norms(32, 300),
            save_planes=planes,
            interpolate=True,
            targets=[*planes, 1.0],
            **common,
        ),
        Scenario(
            "dense_planes_with_kinks",
            {"order": 4},
            norms=_random_norms(33, 300),
            save_planes=planes,
            interpolate=True,
            targets=[*planes, 1.0],
            kinks=[0.35, 0.7],
            **common,
        ),
        Scenario(
            "replay_160",
            {},
            norms=_random_norms(34, 170, high=0.9, specials=False),
            schedule=[1e-3 / 160] * 160,
            targets=[1e-3],
            z0=0.0,
            proposal=1e-5,
            dz_min=1e-12,
            dz_max=1e-3,
            z_end=1e-3,
        ),
        Scenario(
            "replay_kinks_on_edges",
            {"order": 3},
            norms=_random_norms(35, 120, high=0.9, specials=False),
            schedule=[1e-2] * 100,
            kinks=[0.2, 0.5],
            targets=[0.5, 1.0],
            **{**common, "proposal": 1e-2},
        ),
        Scenario(
            "replay_exhausted",
            {},
            norms=_random_norms(36, 60, high=0.9, specials=False),
            schedule=[1e-2] * 50,
            targets=[1.0],
            **{**common, "proposal": 1e-2},
        ),
        Scenario(
            "replay_diverged",
            {},
            norms=[[0.1, 0.1, 0.1]] * 20
            + [[math.inf, 0.1, 0.1]]
            + [[0.1, 0.1, 0.1]] * 10,
            schedule=[1e-2] * 100,
            targets=[1.0],
            **{**common, "proposal": 1e-2},
        ),
    ]
    # Random settings: order, variable order, growth and factors drawn per seed.
    rng = np.random.default_rng(40)
    for seed in range(8):
        method = {
            "order": int(rng.integers(1, 7)),
            "variable_order": bool(rng.integers(0, 2)),
            "max_growth": float(rng.choice([1.2, 2.0, 4.0])),
            "min_factor": float(rng.choice([0.1, 0.2, 0.5])),
            "safety": float(rng.choice([0.7, 0.8, 1.0])),
            "equal_approach": bool(rng.integers(0, 2)),
        }
        scenarios.append(
            Scenario(
                f"random_{seed}",
                method,
                norms=_random_norms(100 + seed, 300, low=1e-6),
                save_planes=planes,
                kinks=[0.45] if seed % 2 else [],
                collar_rate=float(rng.choice([0.0, 50.0])),
                quiet={0: False, 10: True, 30: False} if seed % 3 == 0 else {},
                quiet_interval=4e-3,
                **common,
            )
        )
    return scenarios


def main() -> int:
    """Write the corpus, refusing to overwrite it without ``--force``."""
    if OUT.exists() and "--force" not in sys.argv:
        sys.stdout.write(f"{OUT} is the pre-refactor oracle; --force replaces it.\n")
        return 1
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    entries = []
    for scenario in _scenarios():
        trace = run_scenario(scenario)
        # Keep only the norms the march used: a march that asks for more has diverged
        # from the oracle, and the script running out says so.
        used = sum("accepted" in event for event in trace["events"])
        used += bool(trace["end"]["in_flight"])
        scenario.norms = scenario.norms[:used]
        if run_scenario(scenario) != trace:
            msg = f"scenario {scenario.name!r} is not reproducible from its used norms"
            raise RuntimeError(msg)
        entries.append({"scenario": scenario.to_json(), "trace": trace})
    corpus = {"generated_from": revision, "scenarios": entries}
    text = json.dumps(corpus, separators=(",", ":")) + "\n"
    # A fixed modification time keeps the compressed file identical across runs.
    with gzip.GzipFile(OUT, "wb", compresslevel=9, mtime=0) as handle:
        handle.write(text.encode())
    size, count = OUT.stat().st_size, len(corpus["scenarios"])
    sys.stdout.write(f"wrote {OUT} ({size} bytes, {count} scenarios)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
