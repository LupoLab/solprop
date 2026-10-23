# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record the wall-contrast differential, not an accuracy claim."""

from contextlib import nullcontext

import numpy as np
import pytest

from solprop.constants import C
from solprop.modes import IndexTable, TabulatedHollowMode, TubeLattice, VincettiMode
from solprop.modes.base import beta2, db_per_m


def vincetti_measurements():
    # Diagnostic geometry agreed in the plan: q=1 at 630 nm lies inside
    # 400-1200 nm; t/r=0.05 is inside the published geometry range.
    lattice = TubeLattice(300e-9, 6e-6, 8)
    wavelength = np.linspace(400e-9, 1200e-9, 400)
    omega = 2 * np.pi * C / wavelength
    vacuum = VincettiMode(15e-6, lattice=lattice, cladding=1.45)
    table = IndexTable(omega, vacuum.neff(omega))
    query = omega[3:-3]
    near = abs(wavelength[3:-3] / (2 * 300e-9 * np.sqrt(1.45**2 - 1)) - 1) < 0.03
    result = {}
    # The mandatory upper wavelength exceeded the registry's gas fit; optmat's argon
    # is qualified over it (SEP 3, Phase 4a), so no range warning is raised.
    with nullcontext():
        for pressure in (0.0, 10.0, 30.0):
            model = TabulatedHollowMode(15e-6, "Ar", pressure, table=table)
            direct = VincettiMode(15e-6, "Ar", pressure, lattice=lattice, cladding=1.45)
            index_error = np.real(model.neff(query)) - np.real(direct.neff(query))
            beta2_error = beta2(model, query) - beta2(direct, query)
            loss_ratio = db_per_m(direct, query) / db_per_m(model, query)
            for name, mask in (("near", near), ("away", ~near)):
                result[f"{pressure:g}bar_{name}"] = {
                    "index_max": float(np.max(abs(index_error[mask]))),
                    "beta2_max_fs2_per_m": float(np.max(abs(beta2_error[mask])) * 1e30),
                    "loss_ratio_min": float(np.min(loss_ratio[mask])),
                    "loss_ratio_max": float(np.max(loss_ratio[mask])),
                }
    return result


def test_vincetti_differential_is_finite_and_nonzero():
    result = vincetti_measurements()
    assert set(result) == {
        f"{pressure}bar_{region}"
        for pressure in (0, 10, 30)
        for region in ("near", "away")
    }
    for row in result.values():
        assert np.all(np.isfinite(list(row.values())))
    assert result["30bar_near"]["index_max"] > 0
    # A fixed 400-knot diagnostic, maxima at those samples. These
    # measured differences include interpolation in beta2, whose evacuated
    # baseline is recorded too. A 15% regression margin preserves the finding.
    expected = {
        "10bar_near": [9.186367e-6, 268703.158, 0.527408490, 1.673320019],
        "10bar_away": [8.341320e-6, 275901.013, 0.548304924, 2.027198056],
        "30bar_near": [2.694483e-5, 746847.161, 0.161488284, 2.558383658],
        "30bar_away": [2.398201e-5, 710911.390, 0.267633359, 4.737426441],
    }
    for name, values in expected.items():
        np.testing.assert_allclose(
            list(result[name].values()), values, rtol=0.15, atol=0
        )
    for region, curvature in (("near", 2887.864425), ("away", 4564.024204)):
        # Vacuum knots reproduce the input to rounding; the derivative of a
        # cubic through 400 points cannot resolve every narrow resonance.
        row = result[f"0bar_{region}"]
        assert row["index_max"] <= np.finfo(float).eps
        assert row["beta2_max_fs2_per_m"] == pytest.approx(curvature, rel=0.15, abs=0)
