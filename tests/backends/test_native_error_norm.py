# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's embedded error norm and recycled stage, against the reference's.

The norm decides every step, so a wrong one moves every adaptive run. Two levels:

* **The reduction.** The reference's norm applied to native's own operands: the seven
  stages native read back through ``stage_to_host``, combined here with the tableau's
  error weights, and native's state before and after the trial. Native's scalar is
  that norm within the reduction's ``3 eps`` plus the rounding of forming the
  embedded difference on each side, carried from its operands: the difference of seven
  stages cancels, so that rounding is not relative to the difference itself. The
  original compared two kernels on random unit operands at ``3 eps`` and ``4 eps``
  flat; native's kernels can only be fed by a trial, so the operand term is unavoidable
  and this is weaker by exactly it. The weights, the mask, the order of division and
  norm, and the separation from the scalar form are as sharp as the original's.
* **The decision.** Native's whole trial against the reference's whole trial, each with
  its own stages: the state's one-step bound over the controller's scale.
"""

import dataclasses
import math

import numpy as np
import pytest

from bounds import EPS, gamma, higham_gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper import PIController

native = pytest.importorskip("solprop_native")

STEP_RELATIVE = 16 * EPS
"""The one-step rung of ``accuracy.md`` section 7, relative to the state."""

FORMATION = 12
"""Roundings in one embedded-difference bin on one side: at most seven weighted stages
(each a product and a sum, counted once by ``gamma``), the step length, the complex
propagator's product and its own ``exp``."""


def _problem(capillary_problem, controller=None):
    problem, run_request = capillary_problem()
    if controller is None:
        return problem, run_request
    return problem, run_request.with_stepper(
        dataclasses.replace(run_request.stepper, controller=controller)
    )


def _native_trial(problem, run_request, grid, dz):
    """Open native on a problem, take one trial, and return what it scored."""
    backend = native.NativeBackend(problem, run_request, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    score, _ = backend._trial_step(0.0, dz)
    return backend, score


def _operands(backend, reference, problem, run_request, dz):
    """Return native's stages and state, and the reference's propagator, 4-D."""
    tableau = run_request.stepper.tableau
    state = backend._native_state
    shape = (1, *problem.shape)
    stages = [
        np.asarray(state.stage_to_host(index)).reshape(shape)
        for index in range(len(tableau.nodes))
    ]
    reference.set_initial(problem.initial.field0, None)
    reference._trial_step(0.0, dz)
    step = reference._stage_scalars(0.0, dz)
    propagator = reference._propagator(reference._exponents(0.0, dz)(1.0, step))
    return stages, propagator


def _difference(stages, weights, propagator, dz):
    """Return the embedded difference at the step's end, and its rounding per bin."""
    error = np.zeros_like(stages[0])
    majorant = np.zeros(stages[0].shape)
    for stage, weight in zip(stages, weights, strict=True):
        error = error + float(weight) * stage
        majorant = majorant + abs(float(weight)) * abs(stage)
    carried = propagator * (dz * error)
    return carried, 2 * gamma(FORMATION) * abs(propagator) * dz * majorant


@pytest.mark.parametrize("dz", [1e-3, 1e-2])
def test_the_rk_norm_is_the_reference_norm_of_native_s_operands(
    capillary_problem, field_grid, dz, record_property
):
    """The masked norm against ``passes.error_norm``, the bound fixed from operands."""
    problem, run_request = _problem(capillary_problem)
    assert not np.all(problem.grids.norm_mask), "the mask has to act"
    reference = NumpyRefBackend(problem, run_request, grid=field_grid)
    backend, score = _native_trial(problem, run_request, field_grid, dz)
    try:
        stages, propagator = _operands(backend, reference, problem, run_request, dz)
        carried, rounding = _difference(
            stages, run_request.stepper.tableau.error_weights, propagator, dz
        )
        before = problem.initial.field0
        backend._finish(backend._native_state, dz)
        after = backend.observe().require_field()
        wanted = reference._error_norm(carried, before[None], after[None])
        controller = run_request.stepper.controller
        mask = problem.grids.norm_mask
        scale = float(
            controller.scale(
                float(np.linalg.norm(before[..., mask])),
                float(np.linalg.norm(after[..., mask])),
            )
        )
        bound = 3 * EPS * wanted + float(np.linalg.norm(rounding[..., mask])) / scale
        assert wanted > 0.0
        # The term the original did not have must stay small enough to mean something.
        assert bound < 1e-6 * wanted
        record_property("measured", abs(score - wanted) / wanted)
        record_property("bound", bound / wanted)
        assert abs(score - wanted) <= bound
    finally:
        backend.close()
        reference.close()


@pytest.mark.parametrize("dz", [1e-3, 1e-2])
def test_a_whole_trial_scores_as_the_reference_s_trial_does(
    capillary_problem, field_grid, dz, record_property
):
    """Each backend's own stages: the state's one-step rung over the scale."""
    problem, run_request = _problem(capillary_problem)
    reference = NumpyRefBackend(problem, run_request, grid=field_grid)
    backend, score = _native_trial(problem, run_request, field_grid, dz)
    try:
        reference.set_initial(problem.initial.field0, None)
        wanted, _ = reference._trial_step(0.0, dz)
    finally:
        backend.close()
        reference.close()
    rtol = run_request.stepper.controller.rtol
    assert wanted > 0.0
    record_property("measured", abs(score - wanted))
    record_property("bound", STEP_RELATIVE / rtol)
    assert abs(score - wanted) <= STEP_RELATIVE / rtol


def test_each_norm_is_read_from_its_own_state(capillary_problem, field_grid):
    """The accept's sums: the state before the step, and the state after it."""
    problem, run_request = _problem(capillary_problem)
    dz = 1e-2
    backend = native.NativeBackend(problem, run_request, grid=field_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        partials = np.asarray(backend._prepare_trial(0.0, dz))
        old, new = float(partials[0].sum()), float(partials[1].sum())
        backend._finish(backend._native_state, dz)
        after = backend.observe().require_field()
    finally:
        backend.close()
    mask = problem.grids.norm_mask
    before = problem.initial.field0
    wanted_old = float(np.sum(np.abs(before[..., mask]) ** 2))
    wanted_new = float(np.sum(np.abs(after[..., mask]) ** 2))
    # Two sums of the same samples in other orders, and the band mask between the
    # accept and `after`, which only the norm's bins pass.
    bound = 4 * higham_gamma(before[..., mask].size)
    assert abs(wanted_new - wanted_old) > 8 * bound * wanted_old, "the step must show"
    assert abs(old - wanted_old) <= bound * wanted_old
    assert abs(new - wanted_new) <= bound * wanted_new


def _weights(n_omega):
    return np.linspace(0.5, 2.0, n_omega)


def test_the_weighted_residual_divides_each_bin_before_the_norm(
    capillary_problem, field_grid, record_property
):
    """A per-bin scale inside the norm, on native's own residual.

    ``weighted_error`` divides native's stored residual by any scale: the original's
    ``linspace(0.5, 2)`` goes straight in. A single summed residual divided by a scalar
    is a different number, more than ``1e-3`` from it, and is the defect this exists
    to catch: the C++ witness uses a uniform weight, which cannot tell the two apart.
    """
    n_omega = field_grid.n_omega
    weights = _weights(n_omega)
    problem, run_request = _problem(
        capillary_problem, PIController(rtol=1e-8, atol=1e-14, weights=weights)
    )
    reference = NumpyRefBackend(problem, run_request, grid=field_grid)
    dz = 1e-2
    backend, _ = _native_trial(problem, run_request, field_grid, dz)
    try:
        stages, propagator = _operands(backend, reference, problem, run_request, dz)
        carried, rounding = _difference(
            stages, run_request.stepper.tableau.error_weights, propagator, dz
        )
        partials = backend._native_state.weighted_error(weights)
        got = math.sqrt(float(np.sum(partials)))
    finally:
        backend.close()
        reference.close()
    mask = problem.grids.norm_mask
    wanted = float(np.linalg.norm(carried[..., mask] / weights[mask]))
    bound = 4 * EPS * wanted + float(
        np.linalg.norm(rounding[..., mask] / weights[mask])
    )
    assert wanted > 0.0
    assert bound < 1e-6 * wanted
    record_property("measured", abs(got - wanted) / wanted)
    record_property("bound", bound / wanted)
    assert abs(got - wanted) <= bound
    flat = float(np.linalg.norm(carried[..., mask])) / float(np.mean(weights))
    assert abs(flat - wanted) / wanted > 1e-3
    assert bound < 1e-3 * abs(flat - wanted), "the bound can see the scalar form"


def test_a_weighted_controller_scores_as_the_reference_s_does(
    capillary_problem, field_grid, record_property
):
    """Through the controller's own per-bin scale, at the whole trial's level."""
    weights = _weights(field_grid.n_omega)
    problem, run_request = _problem(
        capillary_problem, PIController(rtol=1e-8, atol=1e-14, weights=weights)
    )
    dz = 1e-2
    reference = NumpyRefBackend(problem, run_request, grid=field_grid)
    backend, score = _native_trial(problem, run_request, field_grid, dz)
    try:
        reference.set_initial(problem.initial.field0, None)
        wanted, _ = reference._trial_step(0.0, dz)
    finally:
        backend.close()
        reference.close()
    assert wanted > 0.0
    bound = STEP_RELATIVE / (1e-8 * float(np.min(weights)))
    record_property("measured", abs(score - wanted))
    record_property("bound", bound)
    assert abs(score - wanted) <= bound


def test_the_recycled_stage_is_the_one_it_replaces(
    capillary_problem, field_grid, record_property
):
    """``k1^{n+1} = exp(dPhi^n) k7^n``, with both collars and a varying operator.

    The tableau's last row is its solution weights, so stage seven's input is the
    accept pass's own result and its derivative is the next step's stage one, rebased
    for the re-anchored interaction picture. Here against the stage native would
    otherwise have evaluated, read with ``stage_to_host``.

    The identity agreed at this order on Numba only because the in-stage absorber term
    is masked to the active band: unmasked, the stage writes outside the band the
    equation is solved on, the accept pass's hard mask removes it, and the identity
    failed at 7.4e-7 there.
    """
    problem, run_request = capillary_problem()
    backend = native.NativeBackend(problem, run_request, grid=field_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.02)
        state = backend._native_state
        assert state.fsal_ready
        z = float(backend._z)
        dz = float(backend._history[-1][1])
        before = state.rk_source_evaluations
        backend._trial_step(z, dz)
        # Six evaluations: the first stage was not evaluated, it was recycled.
        assert state.rk_source_evaluations == before + 6
        recycled = np.asarray(state.stage_to_host(0)).copy()
        # The same stage from scratch, at the same state and step.
        state.clear_fsal()
        before = state.rk_source_evaluations
        backend._trial_step(z, dz)
        assert state.rk_source_evaluations == before + 7
        fresh = np.asarray(state.stage_to_host(0)).copy()
        gap = float(np.linalg.norm(recycled - fresh) / np.linalg.norm(fresh))
        # The derivative's sensitivity to an elementwise relative perturbation of the
        # state, of the size and kind rounding makes: a measured conditioning estimate,
        # not an operation count, taken before the bound is set and independent of the
        # gap it bounds, so the bound can still fail.
        rng = np.random.default_rng(7)
        delta = 1e-11
        field = backend.observe().require_field().copy()
        field *= 1.0 + delta * np.exp(2j * np.pi * rng.random(field.shape))
        backend._native_state.set_initial(
            np.ascontiguousarray(field.reshape(-1)), np.empty(0, dtype=np.complex128)
        )
        backend._trial_step(z, dz)
        moved = np.asarray(state.stage_to_host(0))
        sensitivity = float(np.linalg.norm(moved - fresh) / np.linalg.norm(fresh))
        sensitivity /= delta
    finally:
        backend.close()
    # Rounding, not the identity, is all that separates the two (SEP 25 section 4.3):
    # the stage-seven input and the accepted state are the same sum, formed by two
    # kernels whose contracted roundings differ, each within gamma_20 of the state
    # (seven weighted stages, the propagator's complex product); that departure is
    # carried by the derivative's sensitivity, plus 32 eps for the exp(+dPhi) that
    # undoes the stage's exp(-dPhi).
    bound = 2 * higham_gamma(20) * sensitivity + 32 * EPS
    assert sensitivity > 0.0
    record_property("measured", gap)
    record_property("bound", bound)
    assert gap < bound, (gap, sensitivity)
