# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record a ten-step M2 identity.

Run in separate processes with PYTHONPATH selecting each solprop source and the
matching wheel, plus this checkout's tests directory for power_identity. Compare the
resulting NPZ arrays
with ``tests/bounds.py``'s ``assert_within`` at a rounding bound fixed from the ten
fixed steps (``step_budget``); they are evidence, not a portable golden result.

``--backend`` takes a registry name, so this covers every backend the tree offers.
Nothing here imports CUDA, which matters because the default lane imports this module
for its doctests.
"""

import argparse
import dataclasses
import json
import warnings
from pathlib import Path

# Import the optional native package first, as a run would.
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
from solprop.backends.events import ObservationReady
from solprop.backends.export import ExportRecipe
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.exceptions import MaterialRangeWarning
from solprop.stats import StatsSpec


def record_steps(name, problem, run_request, steps, **context):
    """Return the launch and the state after each of a fixed schedule's first steps.

    The run goes through the backend's session (SEP 47 phase 5), with a save plane at
    each of the schedule's first `steps` edges and no statistics: each plane is one
    accepted step and no rejection after the previous one. The states are as the
    session holds them, in the FFT convention, ``(steps + 1, n_modes, n_pol,
    n_omega)``.
    """
    schedule = run_request.stepper.schedule
    assert schedule is not None
    edges = np.asarray(schedule.edges[1 : steps + 1], dtype=np.float64)
    request = run_request.with_stepper(
        dataclasses.replace(run_request.stepper, save_planes=edges)
    ).for_run(stats=StatsSpec())
    owner = select(name, problem, request, **context)
    session = start_session(owner, problem, request, Start.entrance(problem))
    state = ExportRecipe(power=False, optics=False)
    history, accepted, rejected = [], 0, 0
    try:
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                attempts = getattr(event, "history", None)
                if attempts is not None:
                    accepted += int(np.count_nonzero(attempts.accepted))
                    rejected += int(np.count_nonzero(~attempts.accepted))
                if isinstance(event, ObservationReady):
                    assert (accepted, rejected) == (int(event.plane_index >= 0), 0)
                    accepted = 0
                    history.append(np.array(session.export(event.ticket, state)))
            if batch.terminal is not None and batch.terminal.error is not None:
                raise batch.terminal.error
            session.acknowledge(batch.receipt)
    finally:
        session.close()
    assert len(history) == steps + 1
    return np.stack(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = constant_fill_m2()
        problem = simulation.translate()
        run_request = simulation._run_request()
    assert problem.initial.noise0 is not None
    assert problem.linear.phi_coefficients is not None
    field = record_steps(
        arguments.backend,
        problem,
        run_request,
        10,
        grid=simulation.grid,
        modes=simulation.modes,
    )
    np.savez(
        arguments.output,
        field=field,
        initial=problem.initial.field0,
        noise=problem.initial.noise0,
        phi=problem.linear.phi_coefficients,
    )
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
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
