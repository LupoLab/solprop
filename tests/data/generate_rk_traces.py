# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate ``tests/data/rk_traces.json.gz``: the Runge-Kutta march's decisions.

Run once, from the march as it stood before SEP 45's refactor (the head of
LupoLab/solprop#563 with #566 merged), and committed. It is the oracle the refactored
Python core and the native march must both reproduce (SEP 45 section 7.2).
Regenerating it from a later march would bless whatever that march does, so the script
refuses to overwrite the file unless given ``--force``, which only a deliberate change
to the controller's algorithm, recorded in SEP 45's Part II, may use.

Usage::

    uv run python tests/data/generate_rk_traces.py [--force]
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

from tests.backends.rk_traces import Scenario, run_scenario

OUT = Path(__file__).with_name("rk_traces.json.gz")

CONTROLLER = {"rtol": 1e-6, "atol": 0.0}
"""The controller's tolerances; the norms are scripted, so only its gains matter."""


def _random_norms(
    seed: int,
    count: int,
    *,
    low: float = 0.02,
    high: float = 4.0,
    specials: bool = True,
) -> list[float]:
    """Return `count` log-uniform noise factors, with the special values sprinkled in.

    With a scenario's ``h0`` each multiplies the model norm ``(dz / h0) ** (q + 1)``,
    so the march finds an equilibrium step near ``h0`` and these decide the rest.
    """
    rng = np.random.default_rng(seed)
    norms = np.exp(rng.uniform(math.log(low), math.log(high), size=count)).tolist()
    if specials:
        values = [
            0.0,
            1.0,
            math.nextafter(1.0, 2.0),
            math.nextafter(1.0, 0.0),
            math.inf,
        ]
        for index in rng.choice(count, size=min(count // 10, 30), replace=False):
            norms[int(index)] = float(rng.choice(values))
    return norms


def _scenarios() -> list[Scenario]:
    """Return the corpus's scripted marches (SEP 45 section 7.2's list)."""
    planes = [round(0.1 * k, 10) for k in range(1, 10)]
    common: dict[str, Any] = {
        "z0": 0.0,
        "proposal": 1e-4,
        "dz_min": 1e-12,
        "dz_max": 0.02,
        "z_end": 1.0,
        "h0": 0.005,
    }
    scenarios = [
        Scenario(
            "default",
            dict(CONTROLLER),
            norms=_random_norms(1, 600),
            save_planes=planes,
            **common,
        ),
        Scenario("one_target", dict(CONTROLLER), norms=_random_norms(2, 400), **common),
        Scenario(
            "no_start_up",
            dict(CONTROLLER),
            norms=_random_norms(3, 400),
            save_planes=planes,
            auto_init=False,
            **common,
        ),
        # The start-up rule's cap: a first norm far inside the tolerance records zero.
        Scenario(
            "start_up_capped",
            dict(CONTROLLER),
            norms=[1e-30, *_random_norms(4, 300)],
            save_planes=planes,
            **common,
        ),
        Scenario(
            "start_up_zero",
            dict(CONTROLLER),
            norms=[0.0, *_random_norms(5, 300)],
            save_planes=planes,
            **common,
        ),
    ]
    for lag in (0, 1, 2):
        for free in (False, True):
            scenarios.append(
                Scenario(
                    f"lag_{lag}{'_reject_free' if free else ''}",
                    {**CONTROLLER, "lag": lag, "reject_free": free},
                    norms=_random_norms(10 + 2 * lag + int(free), 400, high=12.0),
                    save_planes=planes,
                    **common,
                )
            )
    scenarios += [
        Scenario(
            "gains_and_factors",
            {
                **CONTROLLER,
                "safety": 0.9,
                "max_factor": 2.0,
                "min_factor": 0.1,
                "beta1": 0.7 / 5,
                "beta2": -0.4 / 5,
            },
            norms=_random_norms(20, 400, high=8.0),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "estimator_order_2",
            {**CONTROLLER, "estimator_order": 2, "beta1": 0.2, "beta2": -1 / 15},
            norms=_random_norms(21, 400, high=8.0),
            save_planes=planes,
            **common,
        ),
        Scenario(
            "rejection_cascades",
            dict(CONTROLLER),
            norms=[0.5] * 5 + [50.0, 20.0, 5.0, 1.5] * 3 + _random_norms(22, 300),
            save_planes=planes,
            **common,
        ),
        # Landings: planes, kinks, interval caps, a stretch that reaches a landing.
        Scenario(
            "kinks_and_intervals",
            dict(CONTROLLER),
            norms=_random_norms(23, 600),
            save_planes=planes,
            breakpoints=[0.0, 0.13, 0.31, 0.47, 0.5, 0.77, 1.0],
            kinks=[0.31, 0.77],
            interval_cap=True,
            **common,
        ),
        Scenario(
            "stretch_wide",
            dict(CONTROLLER),
            norms=_random_norms(24, 600),
            save_planes=[round(0.0125 * k, 10) for k in range(1, 80)],
            stretch=1.5,
            **common,
        ),
        Scenario(
            "targets_beyond_planes",
            dict(CONTROLLER),
            norms=_random_norms(25, 400),
            targets=[0.05, 0.3, 0.3000000001, 0.7, 1.0],
            **common,
        ),
        Scenario(
            "natural_stops",
            dict(CONTROLLER),
            norms=_random_norms(26, 400),
            save_planes=planes,
            step_planes=7,
            **common,
        ),
        # Caps, each on its own, then together with landings and rejections.
        Scenario(
            "quiet_cap",
            dict(CONTROLLER),
            norms=_random_norms(30, 500),
            save_planes=planes,
            quiet={0: False, 5: True, 20: False, 40: True, 60: False, 90: True},
            quiet_interval=4e-3,
            **common,
        ),
        Scenario(
            "transverse_cap",
            dict(CONTROLLER),
            norms=_random_norms(31, 500),
            save_planes=planes,
            ceiling=3e-3,
            **common,
        ),
        Scenario(
            "split_cap",
            dict(CONTROLLER),
            norms=_random_norms(32, 500),
            save_planes=planes,
            split={0: 5e-3, 10: 2e-3, 30: math.inf, 50: 1e-3, 80: 7e-3},
            **common,
        ),
        # A cap meeting a clipped and a stretched landing: caps that just reach, just
        # miss and lie on the planes.
        Scenario(
            "caps_at_landings",
            dict(CONTROLLER),
            norms=_random_norms(33, 600),
            save_planes=[round(0.004 * k, 10) for k in range(1, 240)],
            split={0: 0.004, 25: 0.0041, 50: 0.00399, 75: 0.008},
            ceiling=0.012,
            **common,
        ),
        # The first, capped step under the start-up rule: the controller learns from
        # its own proposal, not the cap.
        Scenario(
            "capped_first_step",
            dict(CONTROLLER),
            norms=_random_norms(34, 300),
            save_planes=planes,
            split={0: 2e-5, 3: math.inf},
            **common,
        ),
        # Scripted norms, not the model's, so that a capped step is rejected at will.
        Scenario(
            "rejection_under_cap",
            dict(CONTROLLER),
            norms=[0.3] * 4
            + [8.0, 3.0, 1.2, 0.9] * 4
            + _random_norms(35, 300, low=1e-3, high=0.3),
            save_planes=planes,
            split={0: 3e-5, 30: math.inf},
            quiet={0: True, 12: False},
            quiet_interval=5e-5,
            **{**common, "h0": None},
        ),
        # The floor at each of its three sites.
        Scenario(
            "floor_on_rejection",
            dict(CONTROLLER),
            norms=[0.5] * 3 + [1e6] * 40,
            z0=0.0,
            proposal=1e-4,
            dz_min=1e-6,
            dz_max=0.02,
            z_end=1.0,
        ),
        Scenario(
            "floor_at_start_up",
            dict(CONTROLLER),
            norms=[0.9, 0.5, 0.5],
            z0=0.0,
            proposal=1e-4,
            dz_min=9e-5,
            dz_max=0.02,
            z_end=1.0,
        ),
        Scenario(
            "floor_on_proposal",
            dict(CONTROLLER),
            norms=[1.0] * 60,
            z0=0.0,
            proposal=1e-4,
            dz_min=2e-5,
            dz_max=0.02,
            z_end=1.0,
            auto_init=False,
        ),
        # Replay of a fixed schedule, with a natural stop and a non-finite norm.
        Scenario(
            "replay",
            dict(CONTROLLER),
            norms=_random_norms(40, 200, specials=False),
            save_planes=[0.25, 0.5, 0.75],
            schedule=[0.01] * 100,
            step_planes=9,
            **common,
        ),
        Scenario(
            "replay_diverged",
            dict(CONTROLLER),
            norms=[0.1] * 10 + [math.inf],
            save_planes=[0.25, 0.5, 0.75],
            schedule=[0.01] * 100,
            **common,
        ),
    ]
    return scenarios


def _commit() -> str:
    """Return the checkout's head commit and whether it is dirty."""
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def main() -> int:
    """Write the corpus, refusing to overwrite it without ``--force``."""
    if OUT.exists() and "--force" not in sys.argv[1:]:
        sys.stderr.write(f"{OUT} exists; pass --force only for a recorded change\n")
        return 1
    corpus = {
        "generated_from": _commit(),
        "scenarios": [
            {"scenario": one.to_json(), "trace": run_scenario(one)}
            for one in _scenarios()
        ],
    }
    with gzip.open(OUT, "wt", encoding="utf-8") as handle:
        json.dump(corpus, handle, sort_keys=True)
    sys.stdout.write(f"wrote {OUT}: {len(corpus['scenarios'])} scenarios\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
