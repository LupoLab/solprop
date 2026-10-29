# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure Raman strategy changes across one to eight resolved H2 modes.

The optional developed input embeds or truncates an existing three-mode spectrum.
It is a matched source-kernel input, not a propagated trajectory for each mode count.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from solprop.driver import Simulation

from solprop_cpunative.backend import CPUNativeBackend, _stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--expected-threads", type=int, required=True)
    args = parser.parse_args()
    stored = None if args.trajectory is None else np.load(args.trajectory)
    for modes in range(1, 9):
        simulation = Simulation.from_capillary(
            16e-6,
            0.13,
            "H2",
            40.0,
            lambda0=515e-9,
            wavelength_limits=(220e-9, 4e-6),
            trange=2e-12,
            fwhm=400e-15,
            energy=10e-6,
            modes=modes,
            loss=0.0,
            seed=2016,
        )
        problem = simulation.translate()
        inputs = [("launch", 0.0, problem.initial.field0)]
        if stored is not None:
            peak = np.asarray(stored["stat_peak_intensity"])
            plane = int(np.argmax(peak.reshape(len(stored["z"]), -1).max(axis=1)))
            source = stored["field"][plane].reshape(1, -1, 1, problem.grids.n_omega)
            field = np.zeros_like(problem.initial.field0)
            retained = min(modes, source.shape[1])
            field[:, :retained] = source[:, :retained]
            inputs.append(("developed-embedding", float(stored["z"][plane]), field))
        backend = CPUNativeBackend(problem)
        try:
            assert backend._cpu.threads == args.expected_threads
            for name, z, field in inputs:
                backend.set_initial(field, None, z=z)
                stage = _stage(backend._stage_scalars(z, z))
                for _ in range(5):
                    backend._cpu.evaluate(stage)
                start = time.perf_counter()
                for _ in range(args.calls):
                    result = backend._cpu.evaluate(stage)
                elapsed = (time.perf_counter() - start) / args.calls
                assert np.isfinite(result).all()
                print(
                    json.dumps(
                        dict(
                            modes=modes,
                            nodes=problem.transform.n_nodes,
                            chunk=problem.transform.node_chunk,
                            input=name,
                            threads=backend._cpu.threads,
                            seconds=elapsed,
                            calls=args.calls,
                            build=backend.metadata()["cpunative"],
                        )
                    ),
                    flush=True,
                )
        finally:
            backend.close()


if __name__ == "__main__":
    main()
