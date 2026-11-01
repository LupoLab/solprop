# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure a dense transform's cost per source call by shape, for one build.

The `HYBRID` GEMM policy chooses between the vendor GEMM and the custom kernels by
shape. This times ten fixed IP-DOPRI5 steps (sixty-one source calls) of short solprop
problems whose transforms span the shapes the policy decides on -- a radial field
route from 2 to 256 nodes, and one- to six-mode capillaries as real fields and as
envelopes -- and records each problem's chosen providers beside its median cost per
call over five warmed repetitions. Run it once per build to compare policies (for
example the installed build against one with ``SOLPROP_NATIVE_GEMM_PROVIDER`` forced),
and on a device build to find that device's crossover::

    OMP_NUM_THREADS=1 SOLPROP_NATIVE_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        python benchmarks/gemm_shapes.py --output gemm-shapes.json
"""

# ruff: noqa: I001 -- Kokkos must load before solprop can import pyFFTW on macOS.

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Any, TypedDict

import solprop_native
import numpy as np
from solprop.backends.registry import select
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController

STEPS = 10
"""Fixed steps a radial problem takes: six source calls each and one to start."""


class Row(TypedDict):
    """One problem's shape, its providers and its cost per source call."""

    kind: str
    size: int
    envelope: bool | None
    shape: list[int]
    n_t: int
    synthesis: str
    projection: str
    calls: int
    per_call_ms: float


def radial(nodes: int, trange: float) -> Simulation:
    """Return a radial field route of `nodes` Hankel nodes on a fixed schedule."""
    length = 1e-3
    grid = FieldGrid.from_limits(length, 800e-9, (500e-9, 1.6e-6), trange)
    settings = SimulationSettings(
        PIController(rtol=1e-6, atol=0.0),
        save_planes=[length],
        dz_max=length / STEPS,
        dz_init=length / STEPS,
        schedule=FixedSchedule(np.full(STEPS, length / STEPS)),
        backend="numpy_ref",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 100e-6
    )
    return Simulation.from_radial(
        grid,
        RadialGrid(1e-3, nodes),
        SpatialMedium("Ar", 1.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        pulse,
        settings=settings,
    )


def capillary(modes: int, trange: float, *, envelope: bool) -> Simulation:
    """Return a short argon capillary of `modes` modes, stepped adaptively."""
    length = 1e-3
    return Simulation.from_capillary(
        75e-6,
        length,
        "Ar",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(300e-9, 2e-6),
        trange=trange,
        envelope=envelope,
        fwhm=30e-15,
        energy=10e-6,
        modes=modes,
        plasma=False,
        raman=False,
        rtol=1e-6,
        backend="numpy_ref",
        save_planes=[length],
    )


def measure(simulation: Simulation, kind: str, size: int, envelope: bool | None) -> Row:
    """Time one problem's march five times after a warm one; return its row."""
    problem: Any = simulation.translate()
    seconds: list[float] = []
    calls = 0
    providers = ("", "")
    for repetition in range(6):
        backend: Any = select(
            "native",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        if repetition == 0:
            native = backend._native_state
            providers = (
                str(native.synthesis_provider),
                str(native.projection_provider),
            )
        backend.set_initial(problem.initial.field0.copy(), None, z=0.0)
        clock = time.perf_counter()
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        seconds.append(time.perf_counter() - clock)
        calls = 6 * int(backend.observe().n_steps) + 1
        backend.close()
    return {
        "kind": kind,
        "size": size,
        "envelope": envelope,
        "shape": [int(value) for value in problem.shape],
        "n_t": int(problem.grids.n_t),
        "synthesis": providers[0],
        "projection": providers[1],
        "calls": calls,
        "per_call_ms": 1e3 * statistics.median(seconds[1:]) / calls,
    }


def main() -> None:
    """Measure every shape for the importable build and write the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    rows: list[Row] = []
    for nodes in (2, 4, 8, 16, 32, 64, 128, 256):
        for trange in (100e-15, 400e-15):
            rows.append(measure(radial(nodes, trange), "radial", nodes, None))
            print(rows[-1], flush=True)
    for modes in (1, 2, 3, 6):
        for trange in (200e-15, 800e-15):
            for envelope in (False, True):
                simulation = capillary(modes, trange, envelope=envelope)
                rows.append(measure(simulation, "capillary", modes, envelope))
                print(rows[-1], flush=True)
    record = {
        "build": dataclasses.asdict(solprop_native.build_info()),
        "load": os.getloadavg(),
        "rows": rows,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
