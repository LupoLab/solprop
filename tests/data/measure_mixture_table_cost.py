# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the supported independent-profile z table at the full CtC grid size.

This is a representation cost, not a propagation or model-accuracy claim.
The geometry and spectral resolution are the stored full CtC case; its hydrogen
fill is replaced by three independently varying air populations at six bar.
"""

import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from solprop.grid import FieldGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.physdata import Mixture
from solprop.profiles import Linear
from solprop.translate.linop import build_phi

OUTPUT = Path("build/mixture-evidence/mixture-ctc-table-cost.json")


def main():
    logging.basicConfig(level=logging.INFO)
    threshold = max(1.0, 0.25 * (os.cpu_count() or 1))
    load = os.getloadavg()[0]
    if load > threshold:
        raise RuntimeError(f"load {load} exceeds quiet threshold {threshold}")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    # The full CtC shape as it was when this cost was measured (its stored Luna
    # comparison has since been retired).
    case = {
        "radius": 16e-6,
        "length": 0.2,
        "pressure": 40.0,
        "modes": 6,
        "lambda_min": 170e-9,
        "lambda_max": 4e-6,
        "trange": 4e-12,
        "n_save": 201,
    }
    grid = FieldGrid.from_limits(
        case["length"], 515e-9, (case["lambda_min"], case["lambda_max"]), case["trange"]
    )
    fill = Mixture.air(6.0, temperature=300.0)
    components = tuple(
        dataclasses.replace(
            component,
            density=Linear(
                float(component.density(np.array([0.0]))[0]),
                (index + 1) / 10 * float(component.density(np.array([0.0]))[0]),
                grid.length,
            ),
        )
        for index, component in enumerate(fill.components)
    )
    mixture = Mixture.of(*components, temperature=300.0)
    modes = tuple(
        MarcatiliMode(
            case["radius"], mixture=mixture, m=index + 1, loss=False, cladding=1.45
        )
        for index in range(case["modes"])
    )
    record = {
        "purpose": "Phi z-table cost at full CtC geometry and grid size; no ledger row",
        "load_before": load,
        "quiet_threshold": threshold,
        "n_omega": grid.n_omega,
        "n_to": grid.n_to,
        "n_modes": len(modes),
        "geometry": case,
        "fill": mixture.serialise(),
        "accuracy": (
            "Production construction tolerances unchanged; "
            "accuracy qualified separately"
        ),
    }
    OUTPUT.with_name("mixture-ctc-table-declaration.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    start = time.perf_counter()
    table = build_phi(modes, grid, Frame.comoving(grid), z_max=grid.length)
    record["build_seconds"] = time.perf_counter() - start
    assert table.phi_representation == "table"
    assert table.phi_coefficients is not None
    record.update(
        phi_coefficient_bytes=table.phi_coefficients.nbytes,
        phi_coefficient_shape=list(table.phi_coefficients.shape),
        intervals=len(table.phi_degrees),
        degrees=table.phi_degrees.tolist(),
        load_after=os.getloadavg()[0],
    )
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
