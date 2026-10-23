# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Final states compared with their stored records, within a matched march's budget.

A route that should not have moved (an opt-in feature left off, a section with nothing
to score) is checked against the final state it gave when it was recorded, stored by
``ndarrays_regression`` beside the test. Every recorded case is ten steps of a
prescribed schedule at the Adams method's fixed order, so a run today is the recorded
run's discrete march and differs from it by rounding alone: another build of a
dependency, a contracted or reordered kernel, another thread layout (SEP 25, class R).
The comparison is to that march's budget (:func:`record_bound`), fixed from the
problem's operation counts before the run, never to its bits.

A phase that changes the arithmetic of a recorded route recaptures the record after
checking that the old and the new state agree within this bound, and says so in its
commit; a deliberate change to the route recaptures it with ``pytest --force-regen``
and names the change.
"""

import numpy as np

from bounds import step_evaluations
from march_bounds import simulation_bound


def record_bound(simulation, state) -> float:
    """Return the relative normwise budget of a recorded run of `simulation`.

    The matched march's budget at the launch's measured nonlinear phase
    (:func:`march_bounds.simulation_bound`). A coupled flow adds its actions'
    formation and conditioning: ``tests/flow_parity.py``'s ten-step bound, derived for
    the Runge-Kutta march's solution weights and scaled here to the Adams weights'
    absolute sum where that is larger.
    """
    bound = simulation_bound(simulation)
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    if flow is not None and flow.generator is not None:
        from flow_parity import RK_WEIGHTS, flow_scales, ten_step_bound  # noqa: PLC0415

        scale = max(1.0, step_evaluations(problem) / RK_WEIGHTS)
        flow_bound = scale * ten_step_bound(
            flow_scales(problem, simulation.grid, state)
        )
        bound = max(bound, flow_bound)
    return bound


def check_state(ndarrays_regression, state, bound: float) -> None:
    """Compare a complex final state with its record, within a relative `bound`.

    `bound` is normwise, relative to the state's norm; every element is then within
    ``bound ||state||`` of its record, which is the elementwise tolerance asked of
    ``ndarrays_regression``.
    """
    state = np.asarray(state, dtype=np.complex128)
    tolerance = {"rtol": 0.0, "atol": bound * float(np.linalg.norm(state))}
    ndarrays_regression.check(
        {
            "real": np.ascontiguousarray(state.real),
            "imag": np.ascontiguousarray(state.imag),
        },
        default_tolerance=tolerance,
    )
