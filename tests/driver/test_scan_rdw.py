# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare a reduced 4 by 2 RDW scan with independent singleton runs.

The points are adaptive, so a scan point is not promised its singleton's steps or bits
(SEP 25 section 7.2): each is held, as its singleton is, to the accuracy envelope of the
same point at a hundredth of the tolerance (``tests/march_bounds.py``).
"""

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from march_bounds import TIGHTER, assert_in_envelope
from solprop.driver import Simulation
from solprop.driver.scan import Scan, run_scan
from solprop.driver.simulation import WAVEGUIDE_RTOL
from solprop.processing.scans import energy_in_band
from solprop.stats import StatsSpec

WITNESS = json.loads((Path(__file__).parents[1] / "data/luna_rdw.json").read_text())


def build_rdw(point, rtol=WAVEGUIDE_RTOL):
    # Only the case's geometry is read from the Luna witness: the scan is compared with
    # its own singletons, not with Luna, so nothing here needs freezing.
    case = WITNESS["reduced"]
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        simulation = Simulation.from_capillary(
            case["radius"],
            case["length"],
            "He",
            (float(point.values["pressure"]), 0.05),
            lambda0=WITNESS["lambda0"],
            wavelength_limits=(case["lambda_min"], case["lambda_max"]),
            trange=case["trange"],
            fwhm=WITNESS["tau_fwhm"],
            energy=float(point.values["energy"]),
            modes=case["modes"],
            n_r=16,
            n_save=3,
            raman=False,
            plasma=False,
            thg=False,
            backend="numba_cpu",
            rtol=rtol,
            stats=StatsSpec(names=("energy", "peak_power")),
            noise=None,
        )
        simulation.translate()
    return simulation


@pytest.mark.medium
def test_reduced_rdw_scan_equals_its_singletons(tmp_path):
    scan = Scan("reduced_rdw", pressure=[0.8, 0.9, 1.0, 1.2], energy=[200e-6, 275e-6])
    reduction = energy_in_band(200e-9, 400e-9)
    result = run_scan(
        scan,
        build_rdw,
        output=tmp_path / "rdw.h5",
        reductions=[reduction],
        fields="planes",
        field_planes=[-1],
        progress="none",
    )
    assert result.status.tolist() == ["done"] * 8
    for point in scan.points():
        singleton = build_rdw(point).run(progress="none")
        tight = build_rdw(point, rtol=WAVEGUIDE_RTOL / TIGHTER).run(progress="none")
        stored = result.point(point.index)
        assert_in_envelope(
            [stored["fields/values"]],
            tight.fields()[-1:],
            singleton.fields()[-1:],
            what=f"point {point.index}'s field",
        )
        assert_in_envelope(
            [stored["reduced/energy_in_band"]],
            reduction(tight),
            reduction(singleton),
            what=f"point {point.index}'s band energy",
        )
