# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for fixed schedules, the three-level order gate and step landing."""

import itertools
import math

import numpy as np
import pytest

from solprop.exceptions import ConvergenceError
from solprop.stepper.schedule import (
    ORDER_MARGIN,
    FixedSchedule,
    LandingTarget,
    StepTargets,
    calibrate,
    discover,
    landing,
    observed_order,
    replay,
)


def make_targets(**overrides):
    """Build a plausible target set: four save planes, one kink, three intervals."""
    fields = {
        "save_planes": np.array([0.25, 0.5, 0.75, 1.0]),
        "hard_kinks": np.array([0.4]),
        # A deliberately uneven table: the narrow middle interval must not cap steps
        # taken in the wide ones (trap 2).
        "breakpoints": np.array([0.0, 0.3, 0.32, 1.0]),
        "dz_max": 0.5,
        "z_end": 1.0,
    }
    return StepTargets(**(fields | overrides))


def march(targets, proposed_dz):
    """March from z_start to z_end with a constant proposal, snapping on landings."""
    z = 0.0
    visited: list[tuple[float, LandingTarget]] = []
    for _ in range(1000):
        result = landing(z, proposed_dz, targets)
        z = result.z_target if result.z_target is not None else z + result.dz
        visited.append((z, result.target))
        if z >= targets.z_end:
            return visited
    raise AssertionError("the march did not reach the end")


# --- FixedSchedule ---------------------------------------------------------------


def test_a_schedule_reports_its_span_and_extremes():
    schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]), z_start=0.1)
    assert schedule.n_steps == 3
    assert schedule.z_start == 0.1
    assert schedule.z_end == pytest.approx(1.1)
    assert schedule.total_length == pytest.approx(1.0)
    assert (schedule.max_step, schedule.min_step) == (0.4, 0.2)
    np.testing.assert_allclose(schedule.edges, [0.1, 0.5, 0.9, 1.1])


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_refining_by_a_power_of_two_divides_each_step_exactly(factor):
    # This is what makes the ladder nested: every level lands where the level above
    # it did. Dividing a double by a power of two is exact, so each piece is exactly
    # h/factor; only the longer cumulative sum can move the endpoint, and then by an
    # ulp or two, which is the tolerance below.
    schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]))
    refined = schedule.refined(factor)
    assert refined.n_steps == 3 * factor
    assert np.array_equal(refined.steps, np.repeat(schedule.steps / factor, factor))
    assert refined.z_end == pytest.approx(
        schedule.z_end, rel=0.0, abs=4 * np.finfo(np.float64).eps
    )


def test_refining_by_one_is_the_identity():
    schedule = FixedSchedule(np.array([0.4, 0.6]))
    assert schedule.refined(1) is schedule


def test_rescaling_by_one_is_the_identity():
    schedule = FixedSchedule(np.array([0.4, 0.6]))
    assert schedule.rescaled(1.0) is schedule


def test_rescaling_keeps_the_span_and_follows_the_discovered_profile():
    # Steps 0.4, 0.4, 0.2: the run went twice as slowly over the last fifth. Halving
    # the scale must keep that ratio, not flatten it.
    schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]))
    finer = schedule.rescaled(0.5)
    assert finer.z_end == pytest.approx(schedule.z_end, rel=0.0, abs=1e-15)
    assert finer.n_steps > schedule.n_steps
    assert finer.max_step == pytest.approx(0.2)
    assert finer.min_step == pytest.approx(0.1)


def test_rescaling_evens_up_a_final_sliver():
    schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]))
    coarse = schedule.rescaled(2.0)
    # A plain walk would end 0.8 then a 0.2 sliver; evening them up keeps every step
    # at or below the scale that was asked for.
    assert coarse.steps.tolist() == [0.5, 0.5]
    assert coarse.z_end == pytest.approx(1.0)


def test_discover_records_an_adaptive_run():
    schedule = discover([0.1, 0.2, 0.15], z_start=0.05)
    assert schedule.n_steps == 3
    assert schedule.z_end == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (np.array([]), "at least one step"),
        (np.array([0.1, 0.0]), "finite and positive"),
        (np.array([0.1, -0.2]), "finite and positive"),
        (np.array([0.1, np.nan]), "finite and positive"),
    ],
)
def test_schedule_validation_rejects_a_bad_record(steps, message):
    with pytest.raises(ValueError, match=message):
        discover(steps)


def test_schedule_validation_rejects_a_non_finite_start():
    with pytest.raises(ValueError, match="z_start must be finite"):
        FixedSchedule(np.array([0.1]), z_start=math.inf)


def test_refining_and_rescaling_reject_useless_factors():
    schedule = FixedSchedule(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="at least 1"):
        schedule.refined(0)
    with pytest.raises(ValueError, match="finite and positive"):
        schedule.rescaled(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        schedule.rescaled(math.inf)


def test_a_schedule_does_not_adopt_the_caller_s_array():
    steps = np.array([0.5, 0.5])
    schedule = FixedSchedule(steps)
    steps[0] = 9.0
    assert schedule.steps[0] == 0.5
    with pytest.raises(ValueError, match="read-only"):
        schedule.steps[0] = 9.0


# --- the order gate --------------------------------------------------------------


def test_observed_order_reads_a_clean_ratio():
    assert observed_order(1.6e-3, 1e-4) == pytest.approx(4.0)
    assert observed_order(2**8, 1.0, refinement=2) == pytest.approx(8.0)


@pytest.mark.parametrize(("coarse", "fine"), [(0.0, 1e-4), (1e-3, 0.0), (-1.0, 1.0)])
def test_observed_order_declines_to_read_a_vanished_difference(coarse, fine):
    assert math.isnan(observed_order(coarse, fine))


def synthetic(order, constant=1.0):
    """Return an observable whose error falls exactly as ``h**order``."""
    return lambda schedule: constant * schedule.max_step**order


def test_calibrate_measures_the_order_of_a_clean_ladder():
    base = FixedSchedule(np.full(8, 0.125))
    calibration = calibrate(base, synthetic(5), order=5, requested=1e-8)
    assert calibration.observed_order == pytest.approx(5.0)
    assert calibration.conclusive
    assert "observed order 5.00" in calibration.message
    # Richardson at order 5 with r = 2: the base error is d0 * 32/31.
    assert calibration.achieved_error == pytest.approx(
        calibration.differences[0] * 32 / 31
    )
    assert calibration.ratio == pytest.approx(calibration.achieved_error / 1e-8)


def test_the_order_gate_fires_on_a_synthetic_order_eight_sequence():
    base = FixedSchedule(np.full(8, 0.125))
    with pytest.raises(ConvergenceError, match=r"observed order 8\.00 far exceeds"):
        calibrate(base, synthetic(8), order=4, requested=1e-8)


def test_the_order_gate_is_a_strict_inequality_at_the_margin():
    base = FixedSchedule(np.full(8, 0.125))
    # Exactly at p + 3 the ladder is allowed through; a hair above it is not. The
    # margin has to tolerate the +0.6 a strong nonlinearity produces while catching
    # the +9 of a schedule that is off its convergence curve.
    on_the_line = calibrate(base, synthetic(4 + ORDER_MARGIN), order=4, requested=1e-8)
    assert on_the_line.observed_order == pytest.approx(float(4 + ORDER_MARGIN))
    with pytest.raises(ConvergenceError, match="far exceeds"):
        calibrate(base, synthetic(4 + ORDER_MARGIN + 0.01), order=4, requested=1e-8)


def test_a_ladder_below_the_round_off_floor_is_refused_a_verdict():
    base = FixedSchedule(np.full(8, 0.125))
    # The differences are ~1e-15 here, so a floor of 1e-15 puts them inside the
    # 1000x band and no order may be pronounced.
    calibration = calibrate(
        base, synthetic(5, constant=1e-11), order=5, requested=1e-8, floor=1e-15
    )
    assert not calibration.conclusive
    assert "does NOT mean the step size is adequate" in calibration.message


def test_an_order_gate_cannot_fire_below_the_floor():
    base = FixedSchedule(np.full(8, 0.125))
    # An order-8 ladder that sits under the floor is inconclusive, not rejected:
    # refusing to pronounce is not the same as pronouncing a failure.
    calibration = calibrate(
        base, synthetic(8, constant=1e-11), order=4, requested=1e-8, floor=1.0
    )
    assert not calibration.conclusive


def test_a_ladder_whose_finer_difference_vanishes_reads_no_order():
    base = FixedSchedule(np.full(8, 0.125))
    # The observable moves between the first two levels and then stops moving, so
    # the coarse difference is well above any floor but the ratio is undefined.
    stalls = lambda schedule: 1.0 if schedule.n_steps == 8 else 0.0  # noqa: E731
    calibration = calibrate(base, stalls, order=5, requested=1e-8)
    assert math.isnan(calibration.observed_order)
    assert not calibration.conclusive
    assert "no order can be read" in calibration.message


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"order": 0}, "order must be at least 1"),
        ({"requested": 0.0}, "requested must be positive"),
        ({"refinement": 1}, "refinement must be at least 2"),
        ({"floor": -1.0}, "floor must be non-negative"),
    ],
)
def test_calibrate_validation_rejects_bad_settings(overrides, message):
    base = FixedSchedule(np.full(4, 0.25))
    arguments = {"order": 5, "requested": 1e-8} | overrides
    with pytest.raises(ValueError, match=message):
        calibrate(base, synthetic(5), **arguments)


def test_calibrate_accepts_a_refinement_other_than_two():
    base = FixedSchedule(np.full(9, 1 / 9))
    calibration = calibrate(base, synthetic(5), order=5, requested=1e-8, refinement=3)
    assert calibration.refinement == 3
    assert calibration.observed_order == pytest.approx(5.0)


# --- replay ----------------------------------------------------------------------


def test_replay_scales_the_schedule_by_the_measured_order():
    base = FixedSchedule(np.full(8, 0.125))
    calibration = calibrate(base, synthetic(5), order=5, requested=1e-8)
    # A 32x smaller error at order 5 needs half the step, so twice the steps.
    scaled = replay(base, calibration, calibration.achieved_error / 32)
    assert calibration.scale_for(calibration.achieved_error / 32) == pytest.approx(0.5)
    assert scaled.n_steps == 16
    assert scaled.z_end == pytest.approx(base.z_end)


def test_replay_refuses_an_inconclusive_calibration():
    base = FixedSchedule(np.full(8, 0.125))
    calibration = calibrate(
        base, synthetic(5, constant=1e-11), order=5, requested=1e-8, floor=1e-15
    )
    with pytest.raises(ConvergenceError, match="inconclusive calibration"):
        replay(base, calibration, 1e-12)


def test_scale_for_rejects_a_non_positive_target():
    base = FixedSchedule(np.full(8, 0.125))
    calibration = calibrate(base, synthetic(5), order=5, requested=1e-8)
    with pytest.raises(ValueError, match="target_error must be positive"):
        calibration.scale_for(0.0)


# --- landing ---------------------------------------------------------------------


def test_a_march_hits_every_save_plane_exactly():
    targets = make_targets()
    visited = march(targets, proposed_dz=0.5)
    reached = [z for z, _ in visited]
    for plane in targets.save_planes:
        assert float(plane) in reached


def test_a_march_never_overshoots_a_kink_or_the_end():
    targets = make_targets()
    for proposed in (0.05, 0.13, 0.5, 10.0):
        positions = [0.0, *[z for z, _ in march(targets, proposed)]]
        assert max(positions) == pytest.approx(targets.z_end)
        # No step may straddle the kink: it must appear as a visited position.
        straddles = [
            (before, after)
            for before, after in itertools.pairwise(positions)
            if before < 0.4 < after
        ]
        assert not straddles


def test_landing_on_a_plane_does_not_make_the_next_step_zero():
    targets = make_targets()
    result = landing(0.25, 0.5, targets)
    assert result.dz > 0.0
    assert result.z_target == 0.4


def test_the_containing_interval_caps_the_step():
    targets = make_targets()
    # Inside the narrow interval [0.30, 0.32] the width binds ...
    inside = landing(0.305, 0.5, targets)
    assert inside.target is LandingTarget.INTERVAL_WIDTH
    assert inside.dz == pytest.approx(0.02)
    assert inside.z_target is None


def test_a_narrow_interval_elsewhere_does_not_cap_the_step():
    # ... but the same narrow interval must not bind a step taken in a wide one.
    # Capping globally by the narrowest interval is trap 2: on a real fibre that is
    # four orders below the median width and turns a smooth metre into 1e6 steps.
    targets = make_targets(save_planes=np.array([1.0]), hard_kinks=np.array([]))
    result = landing(0.0, 0.25, targets)
    assert result.target is LandingTarget.NONE
    assert result.dz == 0.25


def test_the_maximum_step_caps_when_nothing_else_does():
    targets = make_targets(
        save_planes=np.array([1.0]),
        hard_kinks=np.array([]),
        breakpoints=np.array([0.0, 1.0]),
        dz_max=0.2,
    )
    result = landing(0.0, 10.0, targets)
    assert result.target is LandingTarget.MAX_STEP
    assert result.dz == 0.2
    assert result.z_target is None


def test_the_end_of_the_run_is_landed_on_and_never_passed():
    # Luna's loop always takes one step past zmax. This one stops on it.
    targets = make_targets(save_planes=np.array([0.25]), hard_kinks=np.array([]))
    result = landing(0.8, 0.5, targets)
    assert result.target is LandingTarget.END
    assert result.z_target == 1.0
    assert result.dz == pytest.approx(0.2)


def test_a_save_plane_sitting_on_a_kink_is_reported_as_the_save_plane():
    targets = make_targets(
        save_planes=np.array([0.4, 1.0]),
        hard_kinks=np.array([0.4]),
        breakpoints=np.array([0.0, 1.0]),
    )
    result = landing(0.0, 0.5, targets)
    assert result.target is LandingTarget.SAVE_PLANE
    assert result.z_target == 0.4


def test_a_kink_before_the_next_plane_wins():
    # z = 0.35 sits in the wide interval [0.32, 1.0], so nothing caps the step and
    # the kink at 0.4 is nearer than the plane at 0.5.
    result = landing(0.35, 0.5, make_targets())
    assert result.z_target == 0.4
    assert result.target is LandingTarget.HARD_KINK


@pytest.mark.parametrize(
    ("z", "dz", "message"),
    [
        (0.5, 0.0, "dz must be finite and positive"),
        (0.5, math.nan, "dz must be finite and positive"),
        (-0.1, 0.1, "outside the operator table"),
        (1.0, 0.1, "already at or past z_end"),
    ],
)
def test_landing_validation_rejects_impossible_requests(z, dz, message):
    with pytest.raises(ValueError, match=message):
        landing(z, dz, make_targets())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"save_planes": np.array([0.5, 0.5])}, "save_planes must be strictly"),
        ({"hard_kinks": np.array([0.5, 0.2])}, "hard_kinks must be strictly"),
        ({"breakpoints": np.array([0.0])}, "at least one interval"),
        ({"breakpoints": np.array([0.0, np.nan])}, "breakpoints has a non-finite"),
        ({"dz_max": 0.0}, "dz_max must be finite and positive"),
        ({"z_end": math.inf}, "z_end must be finite"),
    ],
)
def test_step_target_validation_rejects_a_broken_set(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_targets(**overrides)


# --- replay determinism ----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.37, 0.5, 1.7, 3.0])
def test_rescaling_the_same_schedule_twice_is_bitwise_identical(factor):
    # Replaying a scaled schedule has to be a replay, not an approximation: every
    # ensemble member and every batch lane in lockstep must take exactly the same
    # steps, or the numerical error stops being a common systematic.
    schedule = FixedSchedule(np.array([0.4, 0.4, 0.2, 0.13, 0.07]))
    first = schedule.rescaled(factor)
    second = FixedSchedule(np.array([0.4, 0.4, 0.2, 0.13, 0.07])).rescaled(factor)
    assert np.array_equal(first.steps, second.steps)
    assert np.array_equal(first.edges, second.edges)


def test_refining_the_same_schedule_twice_is_bitwise_identical():
    schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]))
    assert np.array_equal(schedule.refined(4).steps, schedule.refined(4).steps)


# --- stretch ---------------------------------------------------------------------


def test_a_target_just_beyond_the_proposal_is_landed_on_when_stretch_allows():
    # The plane at 0.25 lies 4 % beyond the proposal, within the default 10 % stretch.
    result = landing(0.0, 0.24, make_targets())
    assert result.target is LandingTarget.SAVE_PLANE
    assert result.z_target == 0.25
    assert result.stretched
    assert result.dz == pytest.approx(0.25)


def test_a_target_beyond_the_stretch_is_not_reached():
    result = landing(0.0, 0.20, make_targets())  # 0.25 > 1.1 * 0.20
    assert result.target is LandingTarget.NONE
    assert not result.stretched
    assert result.dz == 0.20


def test_a_stretch_of_one_switches_stretching_off():
    result = landing(0.0, 0.24, make_targets(stretch=1.0))
    assert result.target is LandingTarget.NONE
    assert result.dz == 0.24


def test_a_stretch_never_exceeds_the_maximum_step():
    # The plane at 0.25 lies within 1.1 x the capped step but beyond dz_max.
    result = landing(0.0, 0.5, make_targets(dz_max=0.24))
    assert result.target is LandingTarget.MAX_STEP
    assert result.dz == 0.24
    assert not result.stretched


def test_a_landing_within_the_cap_is_not_reported_as_stretched():
    result = landing(0.0, 0.5, make_targets())
    assert result.z_target == 0.25
    assert not result.stretched


def test_the_stretch_is_relative_to_the_interval_cap():
    # Inside the narrow interval [0.30, 0.32] the width (0.02) caps the step; a plane
    # at 0.326 lies 0.021 ahead, within 1.1 x 0.02, so the step stretches to it.
    targets = make_targets(save_planes=np.array([0.326, 0.5, 0.75, 1.0]))
    result = landing(0.305, 0.5, targets)
    assert result.target is LandingTarget.SAVE_PLANE
    assert result.stretched
    assert result.dz == pytest.approx(0.021)


def test_stretching_saves_the_sliver_on_a_march():
    # Planes every 0.25 with a proposal of 0.24: without stretch each plane costs a
    # 0.24 step and a 0.01 sliver; with it the plane is reached in one.
    plain = march(make_targets(hard_kinks=np.array([]), stretch=1.0), 0.24)
    stretched = march(make_targets(hard_kinks=np.array([])), 0.24)
    assert len(plain) == 8
    assert len(stretched) == 4
    assert all(target is LandingTarget.SAVE_PLANE for _, target in stretched[:-1])


@pytest.mark.parametrize("stretch", [0.9, 0.0, math.nan, math.inf])
def test_step_targets_reject_a_bad_stretch(stretch):
    with pytest.raises(ValueError, match="stretch must be finite and at least 1"):
        make_targets(stretch=stretch)


# --- uniform schedules -----------------------------------------------------------


def test_a_uniform_schedule_never_exceeds_its_step_and_lands_on_every_target():
    schedule = FixedSchedule.uniform(
        0.0, 1.0, 0.07, save_planes=[0.25, 0.5, 1.0], hard_kinks=[0.4]
    )
    assert schedule.max_step <= 0.07 * (1 + 1e-12)
    for position in (0.25, 0.4, 0.5, 1.0):
        # Edges accumulate over at most fifteen steps here, so a few ulp suffices.
        assert np.min(np.abs(schedule.edges - position)) <= 8 * np.spacing(1.0)
    assert schedule.z_end == pytest.approx(1.0)


def test_uniform_steps_are_equal_within_each_segment():
    schedule = FixedSchedule.uniform(0.0, 1.0, 0.3, save_planes=[0.5])
    np.testing.assert_allclose(schedule.steps, 0.25)
    assert schedule.n_steps == 4


def test_a_segment_that_is_a_whole_number_of_steps_gains_no_sliver():
    # 0.9 / 0.3 is 3.0000000000000004 in floating point; a bare ceil would give 4.
    assert FixedSchedule.uniform(0.0, 0.9, 0.3).n_steps == 3


def test_a_step_longer_than_a_segment_gives_one_step_per_segment():
    schedule = FixedSchedule.uniform(0.0, 1.0, 5.0, save_planes=[0.3, 0.6])
    np.testing.assert_allclose(schedule.steps, [0.3, 0.3, 0.4])


def test_targets_at_the_ends_and_duplicates_are_harmless():
    schedule = FixedSchedule.uniform(
        0.0, 1.0, 0.5, save_planes=[0.0, 0.5, 1.0], hard_kinks=[0.5]
    )
    np.testing.assert_allclose(schedule.steps, [0.5, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"z_start": 0.0, "z_end": 0.0, "dz": 0.1}, "need z_end > z_start"),
        ({"z_start": 0.0, "z_end": math.inf, "dz": 0.1}, "need z_end > z_start"),
        ({"z_start": 0.0, "z_end": 1.0, "dz": 0.0}, "dz must be finite and positive"),
        ({"z_start": 0.0, "z_end": 1.0, "dz": 0.1, "save_planes": [1.5]}, "within"),
        ({"z_start": 0.0, "z_end": 1.0, "dz": 0.1, "hard_kinks": [-0.1]}, "within"),
        (
            {"z_start": 0.0, "z_end": 1.0, "dz": 0.1, "hard_kinks": [math.nan]},
            "must be finite",
        ),
    ],
)
def test_uniform_validation_rejects_a_bad_request(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FixedSchedule.uniform(**kwargs)


def test_a_uniform_schedule_refines_like_any_other():
    assert FixedSchedule.uniform(0.0, 1.0, 0.25).refined(2).n_steps == 8
