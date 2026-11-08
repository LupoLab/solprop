# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The calibration's selection rule, on deterministic fake timings (SEP 46 §7.5)."""

from __future__ import annotations

import pytest

from solprop.calibration.selection import (
    DROP_ORDER,
    MAX_CANDIDATES,
    NOISE_FLOOR,
    admit,
    choose,
    fallbacks,
    noise,
    ratios,
    validate,
)

BUILTIN = [1.0, 1.0, 1.0, 1.0, 1.0]


def test_noise_is_the_largest_aa_deviation_and_at_least_the_floor() -> None:
    assert noise(BUILTIN, [1.0] * 5, minimum=5) == NOISE_FLOOR
    assert noise(BUILTIN, [1.0, 1.06, 0.95, 1.0, 1.0], minimum=5) == 0.06
    # A failed round is dropped; too few complete rounds give no noise at all.
    assert noise(BUILTIN, [1.0, None, 1.0, 1.0, 1.0], minimum=4) == NOISE_FLOOR
    assert noise(BUILTIN, [1.0, None, 1.0, 1.0, 1.0], minimum=5) is None


def test_ratios_drop_failed_and_non_finite_rounds() -> None:
    assert ratios([0.5, float("nan"), None, 2.0], [1.0, 1.0, 1.0, 0.0]) == [0.5]


def test_a_candidate_must_win_every_round_by_more_than_the_noise() -> None:
    clear = admit([0.9, 0.91, 0.89, 0.9, 0.92], BUILTIN, noise=0.02, minimum=5)
    assert clear.admitted
    assert clear.score == 0.9
    assert clear.worst == 0.92
    # A tie keeps the built-in value.
    tie = admit([1.0] * 5, BUILTIN, noise=0.02, minimum=5)
    assert not tie.admitted
    # Winning on average but not in every round is not enough: one outlier round.
    outlier = admit([0.7, 0.7, 0.7, 0.7, 1.01], BUILTIN, noise=0.02, minimum=5)
    assert not outlier.admitted
    assert outlier.score == 0.7
    # Within the noise is not a win.
    marginal = admit([0.97] * 5, BUILTIN, noise=0.04, minimum=5)
    assert not marginal.admitted
    assert "worst ratio 0.970 against 0.960" in marginal.reason


def test_drift_cancels_in_interleaved_pairs() -> None:
    # The machine slows by 10 % a round; the candidate stays 8 % faster in each pair.
    drifting = [1.0 * 1.1**index for index in range(5)]
    candidate = [0.92 * value for value in drifting]
    assert admit(candidate, drifting, noise=0.02, minimum=5).admitted


def test_missing_rounds_keep_the_built_in() -> None:
    verdict = admit([0.5, None, None, 0.5, 0.5], BUILTIN, noise=0.02, minimum=5)
    assert not verdict.admitted
    assert verdict.score is None
    assert verdict.reason == "3 of 5 rounds"
    assert verdict.record()["rounds"] == 3


def test_the_admitted_candidate_with_the_lowest_score_is_chosen() -> None:
    chosen, verdicts = choose(
        {
            "spin": [0.8] * 5,
            "passive": [0.9] * 5,
            "noisy": [0.5, 0.5, 0.5, 0.5, 1.2],
        },
        BUILTIN,
        noise=0.02,
        minimum=5,
    )
    assert chosen == "spin"
    assert not verdicts["noisy"].admitted
    nothing, _ = choose({"slow": [1.1] * 5}, BUILTIN, noise=0.02, minimum=5)
    assert nothing is None
    with pytest.raises(ValueError, match="at most 4"):
        choose(
            {str(index): [0.5] * 5 for index in range(MAX_CANDIDATES + 1)},
            BUILTIN,
            noise=0.02,
            minimum=5,
        )


def test_a_combination_passes_only_with_no_slower_window_and_a_gain() -> None:
    passed = validate(
        {"gnlse": ([0.8] * 3, [1.0] * 3), "radial": ([1.01] * 3, [1.0] * 3)},
        noise=0.02,
        minimum=3,
    )
    assert passed.passed
    assert passed.geomean == pytest.approx((0.8 * 1.01) ** 0.5)
    assert passed.record()["windows"] == {"gnlse": 0.8, "radial": 1.01}
    level = validate({"a": ([1.0] * 3, [1.0] * 3)}, noise=0.02, minimum=3)
    assert not level.passed
    assert level.reason == "geometric mean 1.000"
    short = validate({"a": ([0.8, None, None], [1.0] * 3)}, noise=0.02, minimum=3)
    assert not short.passed
    assert short.reason == "a: 1 of 3 rounds"
    assert not validate({}, noise=0.02, minimum=3).passed


def test_groups_are_dropped_in_a_fixed_order() -> None:
    assert DROP_ORDER == ("grain", "fft", "process")
    assert fallbacks(["fft", "process"]) == [("fft", "process"), ("process",)]
    assert fallbacks([]) == []
    with pytest.raises(ValueError, match="unknown groups"):
        fallbacks(["colour"])


def test_a_candidate_slower_on_one_problem_is_not_admitted() -> None:
    # A large win on one problem carries the geometric mean of every round, but the
    # candidate may not be slower than one plus the noise on any problem.
    verdict = admit(
        [0.8, 0.82, 0.81],
        [1.0, 1.0, 1.0],
        0.02,
        3,
        {"small": [0.5, 0.52, 0.51], "large": [1.04, 1.01, 1.05]},
    )
    assert not verdict.admitted
    assert verdict.reason == "slower on large: median ratio 1.040 against 1.020"
    assert verdict.windows == {"small": 0.51, "large": 1.04}
    # Within the noise on every problem: admitted, and a failed round is dropped.
    fine = admit(
        [0.8] * 3, [1.0] * 3, 0.02, 3, {"small": [0.6] * 3, "large": [1.01, None, 1.0]}
    )
    assert fine.admitted
    assert fine.record()["windows"] == {"small": 0.6, "large": 1.005}
    chosen, verdicts = choose(
        {"a": [0.8] * 3, "b": [0.9] * 3},
        [1.0] * 3,
        0.02,
        3,
        {"a": {"large": [1.3] * 3}},
    )
    assert chosen == "b"
    assert verdicts["a"].reason.startswith("slower on large")
