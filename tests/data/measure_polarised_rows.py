# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record a ten-step two-component capillary run, launched by Cartesian component.

The companion of :mod:`measure_power_identity` for the change of convention that turned
the capillary ``phi = 0`` from the y-polarised member of its degenerate pair into the
x-polarised one. A single-component run is unchanged by that (a circular
capillary is rotationally symmetric); a **two-component** run keeps every number and
exchanges the two rows of its mode axis, because ``ModeCollection.with_polarisations``
orders the pair as (mode, its quarter turn) and that pair reads (y, x) before the change
and (x, y) after it.

The launch here is placed by each mode's own ``principal_component()``, so the script
describes the same physical pulse under either convention and runs unchanged on both
source trees. Run it in separate processes with ``PYTHONPATH`` selecting each
solprop source and the matching Kokkos wheel, then exchange the mode rows of one stored
``field`` and compare the two with ``tests/bounds.py``'s ``assert_within``: the same
method on the same ten-step schedule, so a rounding bound fixed from the step count
(``step_budget``) applies, not exact equality.

``--backend`` takes a registry name, so this covers every backend the tree offers.
"""

import argparse
import json
import math
from pathlib import Path

# Initialise Kokkos before importing the wheel-private pyFFTW OpenMP runtime, as
# `measure_power_identity` does; the default lane imports this module for its doctests.
try:
    # The optional package is out of tree and deliberately absent from the
    # static-analysis source, so the checker cannot resolve it on the base pair.
    import solprop_native as native  # pyright: ignore[reportMissingImports]
except ImportError:
    native = None

import numpy as np

import solprop
from solprop.backends.registry import select
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.modes.collection import ModeCollection
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import KERNEL_SCHEMA_VERSION, assemble

LENGTH = 0.001
STEPS = 10
ANGLE = math.pi / 5
PHASE = math.pi / 3


def build():
    """Return the ten-step two-component problem, its grid and its mode set."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    modes = ModeCollection.with_polarisations([mode])
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-6).build(grid)
    # One Jones vector, addressed by direction rather than by row, so that the two
    # trees launch the same physical pulse into rows that carry opposite directions.
    jones = {
        "x": math.cos(ANGLE),
        "y": math.sin(ANGLE) * complex(math.cos(PHASE), math.sin(PHASE)),
    }
    initial = np.concatenate(
        [jones[str(one.principal_component(0.0))] * pulse for one in modes], axis=0
    )
    problem = assemble(
        grid,
        modes,
        [KerrResponse(chi3=1e-26, thg=True)],
        initial,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / STEPS,
        quadrature=transverse_quadrature(modes, n_r=24),
        components=("x", "y"),
        density=1.0,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
    )
    return problem, grid, modes


def main():
    """Propagate ten steps and store the mode-axis history and the launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    problem, grid, modes = build()
    owner = select(arguments.backend, problem, grid=grid, modes=modes)
    assert problem.stepper.schedule is not None
    try:
        owner.set_initial(problem.initial.field0, None)
        history = [owner.observe().require_field().copy()]
        for position in problem.stepper.schedule.edges[1 : STEPS + 1]:
            summary = owner.propagate_to(float(position))
            assert summary.n_steps == 1 and summary.n_rejected == 0
            history.append(owner.observe().require_field().copy())
        np.savez(
            arguments.output,
            field=np.stack(history),
            initial=problem.initial.field0,
            synthesis=problem.transform.synthesis,
        )
    finally:
        owner.close()
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": KERNEL_SCHEMA_VERSION,
                "backend": arguments.backend,
                "solprop_source": solprop.__file__,
                "native_source": None if native is None else native.__file__,
                "steps": STEPS,
                "window_m": LENGTH,
                "rows": [one.channel_label.display_label for one in modes],
                "components": [str(one.principal_component(0.0)) for one in modes],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
