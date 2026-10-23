# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the fixed-transverse-wavenumber rule on the committed FEM witness.

The nine smooth argon rows and descriptive 3% near-resonance split are those of the
Vincetti FEM comparison. These are one-fibre regression witnesses, never universal
accuracy.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from witness_materials import kingston_argon

from solprop.constants import C
from solprop.modes import IndexTable, TabulatedHollowMode, TabulatedMode
from solprop.modes.base import beta2, db_per_m

WITNESS = json.loads(
    (Path(__file__).parents[1] / "data/jcmwave_vincetti.json").read_text()
)
VACUUM = np.asarray(WITNESS["fills"]["vacuum"]["rows"])
ARGON = np.asarray(WITNESS["fills"]["argon"]["rows"])
RESONANCES = np.asarray(WITNESS["resonance_wavelengths_m"])


def table_from_rows(rows, *, policy: Literal["log", "linear"] = "log"):
    return IndexTable.from_wavelength(
        rows[:, 0],
        rows[:, 3] + 1j * rows[:, 4],
        area=rows[:, 6],
        loss_interpolation=policy,
    )


def away(wavelength):
    # The same descriptive split as the Vincetti FEM witness.
    return np.min(abs(wavelength[:, None] / RESONANCES[None, :] - 1), axis=1) >= 0.03


def fem_measurements():
    table = table_from_rows(VACUUM)
    mode = TabulatedHollowMode(
        15e-6,
        "Ar",
        10.0,
        temperature=293.0,
        susceptibility=kingston_argon(),
        table=table,
        reference_wavelength=800e-9,
    )
    omega = 2 * np.pi * C / ARGON[:, 0]
    residual = np.real(mode.neff(omega)) - ARGON[:, 3]
    smooth = ARGON[:, 0] > 650e-9
    ratio = ARGON[:, 5] / db_per_m(mode, omega)
    area_error = ARGON[:, 6] / table.effective_area(omega) - 1
    filled = TabulatedMode(
        15e-6,
        "Ar",
        10.0,
        temperature=293.0,
        susceptibility=kingston_argon(),
        table=table_from_rows(ARGON),
        reference_wavelength=800e-9,
    )
    first = np.flatnonzero(ARGON[:, 0] > 580e-9)[1:-1]
    beta = omega * ARGON[:, 3] / C
    slopes = []
    # Unequal-spacing second difference, formed directly from raw rows.
    for row in first:
        below, above = omega[row] - omega[row + 1], omega[row - 1] - omega[row]
        slopes.append(
            2
            * (
                beta[row - 1] * below
                - beta[row] * (below + above)
                + beta[row + 1] * above
            )
            / (below * above * (below + above))
        )
    spline_beta2 = beta2(filled, omega[first])
    cv = []
    for row in np.flatnonzero(away(VACUUM[:, 0])):
        if row in (0, len(VACUUM) - 1):
            continue  # Leave-one-out endpoints would require forbidden extrapolation.
        left_out = table_from_rows(np.delete(VACUUM, row, axis=0))
        prediction = left_out.neff([2 * np.pi * C / VACUUM[row, 0]])[0].real
        cv.append((float(VACUUM[row, 0]), float(abs(prediction - VACUUM[row, 3]))))
    # Probe each interval locally: FEM evaluation remains at the rows, while
    # positivity is a property of the interpolant between them.
    linear = table_from_rows(VACUUM, policy="linear")
    nodes = np.asarray(linear.omega)
    queries = nodes[:-1, None] + np.diff(nodes)[:, None] * np.linspace(0.05, 0.95, 19)
    raw_loss = linear._imaginary(queries)
    crossed = np.flatnonzero(np.any(raw_loss < 0, axis=1))
    return {
        "smooth_count": int(smooth.sum()),
        "smooth_index_max": float(np.max(abs(residual[smooth]))),
        "near_index_max": float(np.max(abs(residual[~away(ARGON[:, 0])]))),
        "away_loss_ratio": [
            float(np.min(ratio[away(ARGON[:, 0])])),
            float(np.max(ratio[away(ARGON[:, 0])])),
        ],
        "smooth_loss_ratio": [
            float(np.min(ratio[smooth])),
            float(np.max(ratio[smooth])),
        ],
        "smooth_area_relative_max": float(np.max(abs(area_error[smooth]))),
        "beta2_relative_max": float(np.max(abs(np.asarray(slopes) / spline_beta2 - 1))),
        "beta2_raw_fs2_per_m": (np.asarray(slopes) * 1e30).tolist(),
        "beta2_table_fs2_per_m": (spline_beta2 * 1e30).tolist(),
        "cross_validation_max": max(error for _, error in cv),
        "cross_validation_rows": cv,
        "negative_loss_intervals_nm": [
            [
                float(2 * np.pi * C / nodes[i + 1] * 1e9),
                float(2 * np.pi * C / nodes[i] * 1e9),
            ]
            for i in crossed
        ],
    }


@pytest.fixture(scope="module")
def measurements():
    return fem_measurements()


def test_fem_rule_smooth_residual(measurements):
    assert measurements["smooth_count"] == 9
    # The historical ~2.2e-8 residual plus 15%, with a nonzero lower bound to reject
    # comparison against the same array.
    assert 1.1e-8 < measurements["smooth_index_max"] < 2.245444e-8 * 1.15


def test_fem_near_residual_loss_area_and_cross_validation(measurements):
    # Measured 2.828706e-5 near resonance; half the measurement
    # rejects a degenerate comparison, 15% above it detects a regression.
    assert 1.4e-5 < measurements["near_index_max"] < 2.828707e-5 * 1.15
    # Measured FEM/table ratios: all 14 away rows include a 14% discrepancy
    # at the short-wave edge; the nine smooth rows differ by under 0.74%.
    np.testing.assert_allclose(
        measurements["away_loss_ratio"], [0.858670702, 0.998425846], rtol=0.15, atol=0
    )
    np.testing.assert_allclose(
        measurements["smooth_loss_ratio"], [0.992609394, 0.998425846], rtol=0.15, atol=0
    )
    # The Vincetti FEM comparison's independent fill-insensitivity bound, unchanged.
    assert 0 < measurements["smooth_area_relative_max"] < 1.2e-4
    # Maximum leave-one-out index error 9.151125e-6. This measures
    # sparse-row interpolation (endpoints excluded), not the fill rule.
    assert 4.5e-6 < measurements["cross_validation_max"] < 9.151125e-6 * 1.15


def test_filled_table_beta2_against_raw_differences(measurements):
    # Raw unequal-step differences average curvature across a finite interval;
    # a cubic's derivative is local. Measured worst difference 7.26993%, with
    # the witness's nonuniform 29.5-33.4 nm spacings; retain a 15% margin
    # and a lower guard; this is a discretization limit of the witness rows.
    assert 0.035 < measurements["beta2_relative_max"] < 0.0726993 * 1.15


def test_filled_table_returns_fem_knots():
    table = table_from_rows(ARGON)
    mode = TabulatedMode(15e-6, table=table, reference_wavelength=800e-9)
    omega = 2 * np.pi * C / ARGON[:, 0]
    # Cubic evaluation and exp(log(extinction)): eight rounding operations.
    np.testing.assert_allclose(
        mode.neff(omega),
        ARGON[:, 3] + 1j * ARGON[:, 4],
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )


def test_log_interpolation_keeps_loss_positive(measurements):
    assert measurements["negative_loss_intervals_nm"]
    table = table_from_rows(VACUUM)
    for low, high in measurements["negative_loss_intervals_nm"]:
        omega = 2 * np.pi * C / (np.linspace(low, high, 21)[1:-1] * 1e-9)
        assert np.all(table.alpha(omega) > 0)
