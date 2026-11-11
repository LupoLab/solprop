# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep a natural save at a hard kink from retaining pre-kink Adams history."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sessions.runs import run
from solprop.backends import available
from solprop.backends.events import ObservationReady
from solprop.backends.host.session import HostSession
from solprop.backends.registry import select
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import constant_phi, delta_phi


@pytest.mark.parametrize("backend_name", ["numpy_ref", "native"])
@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_natural_save_at_hard_kink_restarts_before_the_next_trial(
    narrow_grid, linear_problem, backend_name, formulation
):
    if backend_name not in available():
        pytest.skip(f"{backend_name} is not installed")
    # Binary-exact positions isolate the kink/save coincidence from landing roundoff.
    step = 2.0**-20
    kink, end = 2 * step, 4 * step
    problem, run_request = linear_problem(
        narrow_grid,
        save_planes=[end],
        integrator=Multistep(formulation=formulation),
        dz_init=step,
        dz_max=step,
        step_planes=2,
    )
    # A marker on an analytic linear problem tests history ownership independently
    # of a material jump; the exact propagator remains the field witness.
    assert problem.linear.phi_constant is not None
    split = constant_phi(problem.linear.phi_constant, end, kinks=[kink])
    problem = replace(
        problem,
        linear=replace(
            problem.linear,
            phi_breakpoints=split.phi_breakpoints,
            phi_coefficients=split.phi_coefficients,
            phi_degrees=split.phi_degrees,
            hard_kinks=split.hard_kinks,
        ),
    )
    backend = select(backend_name, problem, run_request, grid=narrow_grid)
    marches: list[tuple[float, dict[str, int], int, list[int]]] = []

    def observed(session, batch):
        marches.extend(
            (event.z, *_march(session))
            for event in batch.events
            if isinstance(event, ObservationReady)
        )

    try:
        result = run(backend, problem, between=observed)
    finally:
        backend.close()
    # The launch, the natural save at the kink after two steps, and the end.
    assert [z for z, *_ in marches] == [0.0, kink, end]
    _, at_kink, history, orders = marches[1]
    assert at_kink == {
        "accepted_steps": 2,
        "rejected_steps": 0,
        "restarts": 1,
        "startup": 1,
    }
    assert history == 0
    steps_before_restart = len(orders)
    _, final, _, orders = marches[2]
    assert final == {
        "accepted_steps": 4,
        "rejected_steps": 0,
        "restarts": 1,
        "startup": 2,
    }
    assert orders[steps_before_restart] == 1
    start = problem.initial.field0[0, 0] * problem.grids.active_mask
    exact = np.exp(delta_phi(problem.linear, 0.0, end))[0, 0, 0] * start
    # Four exact linear steps differ only by propagator multiplication roundoff.
    assert_allclose(
        result.fields[-1][0, 0],
        exact,
        rtol=0.0,
        atol=64 * np.finfo(float).eps * np.max(np.abs(start)),
    )


def _march(session: Any) -> tuple[dict[str, int], int, list[int]]:
    """Return a session's Adams march: four of its counters, its history's length and
    the order of every accepted step.

    The reference's march is its driver's (:class:`~solprop.backends.host.session.
    HostSession`); the native session's is the native march's state.
    """
    names = ("accepted_steps", "rejected_steps", "restarts", "startup")
    if isinstance(session, HostSession):
        march = cast("Any", session.driver)._adams
        counters = {name: int(getattr(march.ledger, name)) for name in names}
        return counters, len(march._history), list(march.ledger.orders)
    state = session.native.march_state(0)
    counters = {name: int(state["ledger"][name]) for name in names}
    return counters, len(state["history"]), [int(order) for order in state["orders"]]
