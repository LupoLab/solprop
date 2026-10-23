# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep a natural save at a hard kink from retaining pre-kink Adams history."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.backends import available
from solprop.backends.driver import StepDriver
from solprop.backends.registry import select
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import constant_phi, delta_phi


@pytest.mark.parametrize("backend_name", ["numpy_ref", "numba_cpu", "native"])
@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_natural_save_at_hard_kink_restarts_before_the_next_trial(
    narrow_grid, linear_problem, backend_name, formulation
):
    if backend_name not in available():
        pytest.skip(f"{backend_name} is not installed")
    # Binary-exact positions isolate the kink/save coincidence from landing roundoff.
    step = 2.0**-20
    kink, end = 2 * step, 4 * step
    problem = linear_problem(
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
    backend = select(backend_name, problem, grid=narrow_grid)
    try:
        assert isinstance(backend, StepDriver)
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(end)
        at_kink = backend.observe()
        assert at_kink.z == kink
        march = backend._adams
        assert march is not None
        assert march.ledger.accepted_steps == 2
        assert march.ledger.restarts == 1
        assert march.ledger.startup == 1
        assert march._history == []
        steps_before_restart = len(march.ledger.orders)

        backend.propagate_to(end)
        final = backend.observe()
        assert final.z == end
        assert march.ledger.restarts == 1
        assert march.ledger.startup == 2
        assert march.ledger.orders[steps_before_restart] == 1
        assert march.ledger.accepted_steps == 4
        assert march.ledger.rejected_steps == 0
        start = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
        exact = np.exp(delta_phi(problem.linear, 0.0, end))[0, 0, 0] * start
        # Four exact linear steps differ only by propagator multiplication roundoff.
        assert_allclose(
            final.require_field()[0, 0, 0],
            exact,
            rtol=0.0,
            atol=64 * np.finfo(float).eps * np.max(np.abs(start)),
        )
    finally:
        backend.close()
