# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the error-norm scaling and the PI step controller."""

import dataclasses
import math

import numpy as np
import pytest

from solprop.exceptions import ConvergenceError
from solprop.stepper.controller import (
    DZ_MIN_RELATIVE,
    ControlRecord,
    PIController,
    default_dz_min,
    limit_step,
    weak_norm_scale,
    weighted_norm_scale,
)

BASE_CONTROLLER = PIController(rtol=1e-8, atol=0.0)


def make_controller(**overrides):
    """Build a controller with the shipped defaults and the given fields replaced."""
    return dataclasses.replace(BASE_CONTROLLER, **overrides)


# --- the norm scales -------------------------------------------------------------


def test_the_weak_scale_makes_a_norm_of_one_mean_exactly_at_tolerance():
    scale = weak_norm_scale(1.0, 1.0, rtol=1e-8, atol=0.0)
    # An error of rtol * ‖y‖ is by definition the requested tolerance.
    assert 1e-8 / scale == 1.0


def test_the_weak_scale_takes_the_larger_of_the_two_state_norms():
    assert weak_norm_scale(2.0, 3.0, 1.0, 0.0) == 3.0
    assert weak_norm_scale(3.0, 2.0, 1.0, 0.0) == 3.0


def test_atol_is_additive_and_not_a_floor_on_the_state_norm():
    # Luna's weaknorm uses atol as max(‖y‖, atol), so it never engages at realistic
    # field magnitudes. Additive means it still contributes at ‖y‖ = 1.
    assert weak_norm_scale(1.0, 1.0, rtol=1e-8, atol=1e-9) == pytest.approx(1.1e-8)
    assert weak_norm_scale(0.0, 0.0, rtol=1e-8, atol=1e-30) == 1e-30


def test_the_weighted_scale_reduces_to_the_weak_scale_for_unit_weights():
    weighted = weighted_norm_scale(2.0, 3.0, 1e-8, 1e-30, np.ones(4))
    weak = weak_norm_scale(2.0, 3.0, 1e-8, 1e-30)
    # Bitwise: with w = 1 the two expressions are the same multiplication.
    assert np.array_equal(weighted, np.full(4, weak))


def test_the_weighted_scale_tightens_only_the_bins_it_is_asked_to():
    weights = np.array([1.0, 1e-3])
    scale = weighted_norm_scale(1.0, 1.0, 1e-8, 0.0, weights)
    assert scale[0] == pytest.approx(1e-8)
    assert scale[1] == pytest.approx(1e-11)


def test_an_empty_bin_cannot_collapse_the_weighted_scale():
    # Trap 1: the failure mode of per-element scaling is a bin whose own magnitude is
    # ~0, giving a scale of ~0 and an unbounded norm. The magnitude here is global, so
    # the scale is bounded below by rtol * ‖y‖ * min(w) whatever any single bin holds.
    weights = np.array([1.0, 1.0, 1.0])
    scale = weighted_norm_scale(1.0, 1.0, 1e-8, 0.0, weights)
    assert np.all(scale == 1e-8)
    assert float(5e-15 / scale[2]) < 1e-6


# --- the controller --------------------------------------------------------------


@pytest.mark.parametrize(
    ("norm", "expected"),
    [(0.1, "grow"), (1.0, "safety"), (10.0, "shrink")],
)
def test_the_factor_responds_to_the_norm_in_the_right_direction(norm, expected):
    control = make_controller(lag=0)
    factor = control.propose([norm])
    if expected == "grow":
        assert factor > 1.0
    elif expected == "shrink":
        assert factor < 1.0
    else:
        # At the set point the PI term is unity and only the safety factor is left.
        assert factor == pytest.approx(control.safety)


def test_the_factor_is_clamped_at_both_ends():
    control = make_controller(lag=0, min_factor=0.25, max_factor=2.0)
    assert control.propose([1e-30]) == 2.0
    assert control.propose([1e30]) == 0.25


def test_a_zero_norm_grows_by_the_cap_rather_than_dividing_by_zero():
    control = make_controller(lag=0)
    assert control.propose([0.0]) == control.max_factor


def test_a_non_finite_norm_shrinks_by_the_cap():
    control = make_controller(lag=0)
    assert control.propose([math.nan]) == control.min_factor
    assert control.propose([math.inf]) == control.min_factor


def test_the_integral_term_actually_changes_the_factor():
    control = make_controller(lag=0)
    # The second entry is the *previous* norm; a different history must move the
    # factor, or beta2 is doing nothing.
    assert control.propose([0.5, 0.5]) != pytest.approx(control.propose([0.1, 0.5]))


def test_a_non_usable_previous_norm_falls_back_to_the_set_point():
    control = make_controller(lag=0)
    at_set_point = control.propose([1.0, 0.5])
    assert control.propose([math.nan, 0.5]) == at_set_point
    assert control.propose([0.0, 0.5]) == at_set_point


# --- lag -------------------------------------------------------------------------


def test_lag_zero_uses_the_norm_of_the_step_just_taken():
    control = make_controller(lag=0)
    assert control.propose([0.9, 0.4]) == control.propose([0.9, 0.4, 0.4][:2])
    # The last entry is the current norm and the one before it the previous norm.
    assert control.propose([0.7, 0.3]) != control.propose([0.3, 0.7])


def test_lag_defers_the_update_by_exactly_one_step():
    lagged = make_controller(lag=1)
    prompt = make_controller(lag=0)
    history = [0.9, 0.4, 0.2, 0.7]
    # The factor a lag-1 controller applies after n steps is the one a lag-0
    # controller would have applied after n-1: the update arrives one step late.
    for length in range(2, len(history) + 1):
        assert lagged.propose(history[:length]) == prompt.propose(history[: length - 1])


@pytest.mark.parametrize("lag", [0, 1, 2, 4])
def test_the_lag_queue_is_primed_with_unit_factors(lag):
    control = make_controller(lag=lag)
    for length in range(lag + 1):
        assert control.propose([0.5] * length) == 1.0
    assert control.propose([0.5] * (lag + 1)) != 1.0


# --- acceptance and rejection ----------------------------------------------------


@pytest.mark.parametrize("norm", [0.0, 1e-12, 0.5, 1.0, 1.0000001, 1e12])
def test_reject_free_never_rejects_a_finite_norm(norm):
    assert make_controller(reject_free=True).accepts(norm)


@pytest.mark.parametrize("norm", [math.nan, math.inf, -math.inf])
def test_a_non_finite_norm_is_rejected_even_when_reject_free(norm):
    # The carve-out from "never rejects": no lagged update can absorb a poisoned
    # state, so the step has to be retried smaller.
    assert not make_controller(reject_free=True).accepts(norm)
    assert not make_controller().accepts(norm)


@pytest.mark.parametrize(
    ("norm", "accepted"), [(0.5, True), (1.0, True), (1.0000001, False)]
)
def test_the_rejecting_mode_accepts_exactly_up_to_one(norm, accepted):
    assert make_controller().accepts(norm) is accepted


def test_a_rejection_can_never_grow_the_step():
    control = make_controller()
    assert control.rejection_factor(1.0000001) <= 1.0
    assert control.rejection_factor(1e6) == control.min_factor
    assert control.rejection_factor(math.nan) == control.min_factor
    assert control.rejection_factor(0.0) == 1.0


def test_the_rejection_exponent_follows_the_estimator_order():
    fourth = make_controller(estimator_order=4)
    second = make_controller(estimator_order=2)
    assert fourth.rejection_factor(4.0) == pytest.approx(0.8 * 4.0 ** (-1 / 5))
    assert second.rejection_factor(4.0) == pytest.approx(0.8 * 4.0 ** (-1 / 3))


def test_the_gains_for_an_order_four_estimator_are_the_defaults():
    built = PIController.for_estimator_order(4, rtol=1e-8, atol=0.0)
    default = make_controller()
    # Bitwise, so the classmethod really is the rule the defaults were taken from.
    assert (built.beta1, built.beta2) == (default.beta1, default.beta2)
    assert built.estimator_order == 4


def test_the_gains_scale_with_the_estimator_order():
    built = PIController.for_estimator_order(2, rtol=1e-8, atol=0.0)
    assert built.beta1 == pytest.approx(0.6 / 3)
    assert built.beta2 == pytest.approx(-0.2 / 3)


# --- replay determinism ----------------------------------------------------------


def test_the_same_norms_give_the_same_factors_bitwise():
    control = make_controller()
    history = [0.9, 0.31, 0.44, 1.0, 0.02, 0.77]
    first = [control.propose(history[:n]) for n in range(len(history) + 1)]
    second = [control.propose(list(history[:n])) for n in range(len(history) + 1)]
    twin = PIController(rtol=1e-8, atol=0.0)
    third = [twin.propose(history[:n]) for n in range(len(history) + 1)]
    assert np.array_equal(first, second)
    assert np.array_equal(first, third)


def test_the_scale_dispatches_on_whether_weights_were_given():
    weak = make_controller()
    weighted = make_controller(weights=np.array([1.0, 0.5]))
    assert weak.scale(2.0, 3.0) == weak_norm_scale(2.0, 3.0, 1e-8, 0.0)
    assert np.array_equal(
        weighted.scale(2.0, 3.0),
        weighted_norm_scale(2.0, 3.0, 1e-8, 0.0, np.array([1.0, 0.5])),
    )


def test_the_controller_does_not_adopt_the_caller_s_weight_array():
    weights = np.array([1.0, 0.5])
    control = make_controller(weights=weights)
    weights[1] = 99.0
    assert control.weights is not None
    assert control.weights[1] == 0.5
    with pytest.raises(ValueError, match="read-only"):
        control.weights[0] = 2.0


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rtol": 0.0}, "rtol must be positive"),
        ({"atol": -1.0}, "atol must be non-negative"),
        ({"safety": 1.5}, "safety must lie in"),
        ({"min_factor": 2.0}, "min_factor < 1 < max_factor"),
        ({"max_factor": 0.9}, "min_factor < 1 < max_factor"),
        ({"beta1": 0.0}, "beta1 must be positive"),
        ({"lag": -1}, "lag must be non-negative"),
        ({"estimator_order": 0}, "estimator_order must be at least 1"),
        ({"weights": np.array([1.0, 0.0])}, "finite and positive"),
        ({"weights": np.array([1.0, np.nan])}, "finite and positive"),
    ],
)
def test_controller_validation_rejects_bad_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_controller(**overrides)


def test_an_unknown_norm_region_is_refused_by_name():
    # The suite's runtime type hook refuses a region outside the Literal before the
    # controller's own check runs; without it (as installed) the check is what
    # refuses, so it is exercised on the constructed object directly.
    controller = make_controller()
    object.__setattr__(controller, "region", "everywhere")
    with pytest.raises(ValueError, match="region must be 'active' or 'band'"):
        controller.__post_init__()


# --- the achieved/requested record -----------------------------------------------


def test_the_record_carries_the_achieved_over_requested_ratio():
    control = make_controller()
    record = ControlRecord.from_norms(control, [0.2, 0.8, 0.5], n_rejected=1)
    assert record.requested_rtol == 1e-8
    assert record.requested_atol == 0.0
    assert (record.n_accepted, record.n_rejected) == (3, 1)
    assert record.tolerance_ratio == record.max_norm == 0.8
    assert record.mean_norm == pytest.approx(0.5)
    assert record.rejection_rate == 0.25


def test_an_empty_record_is_well_defined():
    record = ControlRecord.from_norms(make_controller(), [])
    assert (record.max_norm, record.mean_norm, record.rejection_rate) == (0.0, 0.0, 0.0)


# --- the step-size floor ---------------------------------------------------------


def test_limit_step_clamps_to_the_ceiling():
    assert limit_step(1e-3, 1000.0, dz_min=0.0, dz_max=1e-2) == 1e-2


def test_limit_step_raises_rather_than_silently_accepting_below_the_floor():
    # Luna's steplims! clamps to min_dt and marks the step accepted, abandoning the
    # tolerance with no record anywhere in the output. This is the non-port of that.
    with pytest.raises(ConvergenceError, match="below the floor"):
        limit_step(1e-3, 1e-9, dz_min=1e-6, dz_max=1e-2)


def test_limit_step_rejects_an_inverted_range():
    # Luna resolves min_dt > max_dt to max_dt through an if/elseif and says nothing.
    with pytest.raises(ValueError, match="below dz_min"):
        limit_step(1e-3, 1.0, dz_min=1e-2, dz_max=1e-6)


# --- the floor -------------------------------------------------------------------


def test_the_default_floor_is_relative_to_the_run_length():
    assert default_dz_min(2.0) == pytest.approx(2.0 * DZ_MIN_RELATIVE)


@pytest.mark.parametrize("length", [0.0, -1.0, math.inf, math.nan])
def test_the_default_floor_rejects_a_bad_length(length):
    with pytest.raises(ValueError, match="length must be finite and positive"):
        default_dz_min(length)


def test_the_default_floor_never_binds_a_legitimate_fine_scale():
    # Trap 2: the finest local scale on a real fibre is four orders below the median
    # step, so a floor of 1e-9 of the length sits far below any step a healthy run
    # takes. A metre stepping at 1e-4 of its length, shrunk by the cap, is still clear.
    step = limit_step(1e-4, 0.2, dz_min=default_dz_min(1.0), dz_max=1.0)
    assert step == pytest.approx(2e-5)


# --- the equivalent norm ---------------------------------------------------------


def test_the_equivalent_norm_is_the_norm_itself_for_an_unclipped_step():
    assert BASE_CONTROLLER.equivalent_norm(0.3, dz_taken=1e-3, dz_proposed=1e-3) == 0.3


def test_a_sliver_s_norm_is_scaled_back_up_to_the_proposal():
    # Order-4 estimator: the local error goes as h**5, so a step a tenth as long
    # scores 1e-5 of what the proposal would have.
    norm = BASE_CONTROLLER.equivalent_norm(1e-6, dz_taken=1e-4, dz_proposed=1e-3)
    assert norm == pytest.approx(0.1)


def test_a_stretched_step_s_norm_is_scaled_down():
    norm = BASE_CONTROLLER.equivalent_norm(1.0, dz_taken=1.1e-3, dz_proposed=1e-3)
    assert norm == pytest.approx(1.1**-5)


def test_the_equivalent_norm_follows_the_estimator_order():
    control = make_controller(estimator_order=6)
    norm = control.equivalent_norm(1.0, dz_taken=0.5, dz_proposed=1.0)
    assert norm == pytest.approx(2.0**7)


@pytest.mark.parametrize("norm", [math.nan, math.inf])
def test_a_non_finite_norm_passes_through_the_rescaling(norm):
    result = BASE_CONTROLLER.equivalent_norm(norm, dz_taken=1e-4, dz_proposed=1e-3)
    assert not math.isfinite(result)


@pytest.mark.parametrize("field", ["dz_taken", "dz_proposed"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
def test_the_rescaling_rejects_a_bad_length(field, value):
    lengths = {"dz_taken": 1e-3, "dz_proposed": 1e-3, field: value}
    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        BASE_CONTROLLER.equivalent_norm(0.5, **lengths)


def test_a_landing_does_not_stall_the_controller():
    # The regrowth problem: a controller that learns from a sliver restarts from it
    # and regrows at max_factor per step. With the proposal kept as the state and the
    # sliver's norm rescaled, the next proposal is what it would have been anyway.
    control = make_controller(lag=0)
    proposal, sliver = 1e-3, 1e-5
    # Under the h**5 model the sliver scores 1e-10 of the proposal's norm of 0.3.
    sliver_norm = 0.3 * (sliver / proposal) ** 5
    history = [0.3, 0.3, control.equivalent_norm(sliver_norm, sliver, proposal)]
    next_step = limit_step(proposal, control.propose(history), dz_min=0.0, dz_max=1.0)
    assert next_step == pytest.approx(proposal * control.propose([0.3, 0.3, 0.3]))
    # ... and nowhere near the sliver times the growth cap.
    assert next_step > 0.5 * proposal


def test_the_record_names_the_worst_step_and_whether_tolerance_was_met():
    record = ControlRecord.from_norms(BASE_CONTROLLER, [0.2, 1.4, 0.5])
    assert record.worst_step == 1
    assert not record.met_tolerance
    assert ControlRecord.from_norms(BASE_CONTROLLER, [0.2, 0.9]).met_tolerance


def test_an_empty_record_has_no_worst_step_and_meets_tolerance():
    record = ControlRecord.from_norms(BASE_CONTROLLER, [])
    assert record.worst_step == -1
    assert record.met_tolerance
