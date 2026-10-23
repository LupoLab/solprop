# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record a ten-step M2 identity from either the schema-10 or schema-11 source.

Run in separate processes with PYTHONPATH selecting each solprop source and the
matching Kokkos wheel, plus this checkout's tests directory for power_identity.
The baseline receives the independently pre-divided launch in its assembled
contract, retaining the same response tables. Compare the resulting NPZ arrays
with ``tests/bounds.py``'s ``assert_within`` at a rounding bound fixed from the ten
fixed steps (``step_budget``); they are evidence, not a portable golden result.

``--backend`` takes a registry name, so this covers every backend the tree offers.
Nothing here imports CUDA, which matters because the default lane imports this module
for its doctests.
"""

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path

# Initialise Kokkos before importing the wheel-private pyFFTW OpenMP runtime.
# The base-pair doctest collector also imports this measurement script.
try:
    # The optional package is out of tree and deliberately absent from the
    # static-analysis source, so the checker cannot resolve it on the base pair.
    import solprop_native as native  # pyright: ignore[reportMissingImports]
except ImportError:
    native = None

import numpy as np

import solprop
from power_identity import constant_fill_m2
from solprop.backends.registry import select
from solprop.exceptions import MaterialRangeWarning
from solprop.translate.problem import SCHEMA_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = constant_fill_m2()
        problem = simulation.translate()
        root_weight = np.ones(problem.shape)
        for mode_index, mode in enumerate(simulation.modes):
            active = simulation.grid.active_mask
            root_weight[0, mode_index, 0, active] = np.sqrt(
                np.real(mode.neff(simulation.grid.omega[active], 0.0))
            )
    assert SCHEMA_VERSION in (10, 11, 12, 13)
    if SCHEMA_VERSION == 10:
        assert problem.initial.noise0 is not None
        problem = replace(
            problem,
            initial=replace(
                problem.initial,
                field0=problem.initial.field0 / root_weight,
                noise0=problem.initial.noise0 / root_weight,
            ),
        )
    owner = select(
        arguments.backend, problem, grid=simulation.grid, modes=simulation.modes
    )
    assert problem.stepper.schedule is not None
    assert problem.initial.noise0 is not None
    assert problem.linear.phi_coefficients is not None
    try:
        owner.set_initial(problem.initial.field0, None)
        history = [owner.observe().require_field().copy()]
        for position in problem.stepper.schedule.edges[1:11]:
            summary = owner.propagate_to(float(position))
            assert summary.n_steps == 1 and summary.n_rejected == 0
            history.append(owner.observe().require_field().copy())
        np.savez(
            arguments.output,
            field=np.stack(history),
            initial=problem.initial.field0,
            noise=problem.initial.noise0,
            phi=problem.linear.phi_coefficients,
        )
    finally:
        owner.close()
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "backend": arguments.backend,
                "solprop_source": solprop.__file__,
                "native_source": None if native is None else native.__file__,
                "steps": 10,
                "window_m": 0.001,
                "pressure_bar": 1.2,
                "seed": 31,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
