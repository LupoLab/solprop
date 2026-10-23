# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""P0 of the verification ladder: linear only, exactly.

``docs/dev/stepper.md`` section 11, rung P0. With the nonlinear term switched off the
interaction picture is the whole scheme, so the propagated spectrum must equal
``exp(Phi(L)) E0`` whatever the step size and however many steps were taken. That is the
one property the cumulative operator exists to provide (see docs/dev/decisions.md), and
it is the first thing a backend can get wrong: a right-endpoint rule for the operator is
first order in ``h`` here and its error is invisible to the error estimate.
"""

import numpy as np
import pytest

from solprop.stepper import FixedSchedule
from solprop.translate.linop import delta_phi

LENGTH = 0.05  # m, a fiftieth of the soliton period: enough steps to accumulate error


def propagate(backend_class, problem, grid, z):
    """Run a problem to a plane and return the masked initial and final spectra."""
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(z)
    plane = backend.observe()
    backend.close()
    return summary, plane.field


def exact(problem, z):
    """Return ``exp(Phi(z)) E0``, with the initial state masked as the backend is."""
    start = problem.initial.field0 * problem.grids.active_mask
    return np.exp(delta_phi(problem.linear, 0.0, z)) * start


def relative(got, want):
    """Return the largest absolute deviation as a fraction of the largest value."""
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.mark.parametrize("with_kerr", [True, False])
def test_a_constant_operator_is_the_exact_linear_flow(
    backend_class, linear_problem, narrow_grid, zero_kerr, with_kerr
):
    """Every pass may run and contribute nothing, or be skipped altogether."""
    responses = [zero_kerr] if with_kerr else []
    problem = linear_problem(narrow_grid, save_planes=[LENGTH], responses=responses)
    assert problem.flags.constant_linop
    summary, final = propagate(backend_class, problem, narrow_grid, LENGTH)
    # Every step multiplies by one propagator factor, so the round-off is bounded by
    # n_steps * eps = 43 * 2.2e-16 = 9.5e-15 on the largest bin (Higham); the gate's
    # 1e-14 is that bound rounded up. Measured 4.8e-16 over 43 steps, with and without
    # the passes running.
    assert summary.n_steps > 8
    assert relative(final, exact(problem, LENGTH)) < 1e-14


def test_a_z_varying_operator_is_the_exact_linear_flow(
    backend_class,
    linear_problem,
    narrow_grid,
    substitute,
    synthetic_expansion,
    expanded_prefactor,
):
    """A nasty synthetic ``D(omega, z)``: sinusoidal dispersion and a growing loss.

    ``Phi`` is separable -- one array times the dispersion's integral plus another
    times the loss's -- which is exactly the two-term expansion the seam's device form
    is, so the case runs on a compiled backend as well as on the oracle.
    """
    base = linear_problem(narrow_grid, save_planes=[LENGTH])
    detuning = base.grids.omega - base.grids.omega0
    active = base.grids.active_mask
    terms = [0.5j * detuning**2 * active, -0.5 * active]

    def weights_of(z):
        """Return the two closed-form integrals of ``D`` along z."""
        z = np.asarray(z, dtype=float)
        dispersion = -1.2e-26 * (z + 0.4 * np.sin(60.0 * z) / 60.0)
        loss = 4.0 * z + 0.5 * z**2
        return np.stack([dispersion, loss], axis=-1)

    problem = substitute(
        base,
        linear=synthetic_expansion(base, terms, weights_of),
        responses=expanded_prefactor(base.responses),
    )
    assert not problem.flags.constant_linop
    summary, final = propagate(backend_class, problem, narrow_grid, LENGTH)
    # The table's own approximation error cancels exactly: the sum of the steps'
    # differences of one polynomial telescopes to Phi(L) - Phi(0) whatever the
    # interpolant is. What is left is the round-off of two Clenshaw passes per step,
    # relative to an exponent that reaches 1085 rad at the top of the band, so
    # 2 * 43 * 1085 * eps = 2.1e-11 there; at the peak of the pulse, where this measure
    # normalises, the exponent is small and the measured deviation is 6.4e-16.
    assert summary.n_steps > 8
    assert relative(final, exact(problem, LENGTH)) < 1e-13


@pytest.mark.parametrize("n_steps", [2, 7, 31])
def test_the_linear_flow_is_independent_of_the_step_count(
    backend_class, linear_problem, narrow_grid, n_steps
):
    """A replayed schedule of any length reaches the same place, to round-off."""
    problem = linear_problem(
        narrow_grid,
        save_planes=[LENGTH],
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / n_steps),
    )
    summary, final = propagate(backend_class, problem, narrow_grid, LENGTH)
    assert summary.n_steps == n_steps
    # As above, with the step count as the only variable: 31 * 2.2e-16 = 6.8e-15.
    # Measured 9.6e-17, 1.9e-16 and 9.6e-16 at 2, 7 and 31 steps -- linear in the count,
    # which is what a correlated rotation applied every step predicts.
    assert relative(final, exact(problem, LENGTH)) < 1e-14


def test_a_linear_run_meets_its_tolerance_and_records_every_step(
    backend_class, linear_problem, narrow_grid
):
    """A zero embedded estimate is a met tolerance, and every step is still recorded."""
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(LENGTH)
    plane = backend.observe()
    backend.close()
    assert summary.met_tolerance
    assert summary.max_norm == 0.0
    assert plane.step_history.n_steps == summary.n_steps
    assert np.all(plane.step_history.accepted)
    assert plane.z == LENGTH
    # The controller grows by its cap every step while the estimate stays at zero, so
    # the steps are strictly increasing until dz_max binds.
    steps = plane.step_history.dz
    assert steps[1] >= steps[0]
