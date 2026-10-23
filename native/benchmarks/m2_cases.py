# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The M2 benchmark cases: vacuum-ultraviolet dispersive-wave emission in helium.

The full case is the published RDW example (a 7.5 fs, 275 uJ pulse at 800 nm in a
1.5 m, 125 um-radius capillary with a pressure gradient from 0.8 bar to vacuum, four
modes); the reduced case is its short, two-mode companion. ``expected_steps`` records
the step count of each case's independent reference run.
"""

from __future__ import annotations

type Value = float | int | str | bool | tuple[float, float]

RDW: dict[str, Value] = {
    "gas": "He",
    "lambda0": 800e-9,
    "tau_fwhm": 7.5e-15,
    "energy": 275e-6,
    "loss": True,
}

CASES: dict[str, dict[str, Value]] = {
    "m2_full": {
        "radius": 125e-6,
        "length": 1.5,
        "pressure": (0.8, 0.0),
        "modes": 4,
        "lambda_min": 90e-9,
        "lambda_max": 4e-6,
        "trange": 400e-15,
        "n_save": 201,
        "expected_steps": 4365,
    },
    "m2_reduced": {
        "radius": 125e-6,
        "length": 1.0,
        "pressure": (1.2, 0.0),
        "modes": 2,
        "lambda_min": 100e-9,
        "lambda_max": 1.5e-6,
        "trange": 200e-15,
        "n_save": 51,
        "expected_steps": 348,
    },
}

FAMILIES: dict[str, dict[str, Value]] = dict.fromkeys(CASES, RDW)
