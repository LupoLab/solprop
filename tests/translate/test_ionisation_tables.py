# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ionisation rate table: accuracy, clipping, the cache key and the error paths.

The rates tabulated are optmat's PPT form (``OptmatRate(gas, 800e-9,
model="ppt_ilkov")``), the physical rate a run tabulates, and the test-only ADK oracle
(``tests/rate_oracles.py``), a closed form that reaches the rate ceiling and does not
move with optmat. Both replaced solprop's own ``PPT`` and ``ADK``, deleted in SEP 3,
Phase 4b; the table is the subject, not the rate.
"""

import math
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from convergence_records import Column, Study, record
from rate_oracles import adk_oracle
from solprop.exceptions import TranslationError
from solprop.nonlinear.ionisation import MAX_RATE, barrier_suppression_field
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.physdata import Gas
from solprop.translate.ionisation import (
    DEFAULT_RTOL,
    SCHEMA_VERSION,
    RateTable,
    build_rate_table,
    clear_rate_table_cache,
    rate_table_key,
)

NOBLE = [Gas.He, Gas.Ne, Gas.Ar, Gas.Kr, Gas.Xe]
NOBLE_IDS = [gas.value for gas in NOBLE]
WAVELENGTH = 800e-9


def barrier(gas):
    """The barrier-suppression field of a noble gas, V/m.

    On the oracle's potential and unit charge, the former registry's inputs, so the
    ranges below are the ones the tests were written for.
    """
    return barrier_suppression_field(adk_oracle(gas).ionisation_potential, 1.0)


def ppt(gas, wavelength=WAVELENGTH):
    """optmat's PPT form of a noble gas, tabulated up to its maximum field."""
    return OptmatRate(gas, wavelength, model="ppt_ilkov")


def default_range(gas):
    """The field range a propagation would ask for: up to twice the barrier field.

    The lower end is Luna's, ``E_b / 2500`` (PR #426); the table raises it to the rate's
    own threshold, below which there is nothing to interpolate.
    """
    return (barrier(gas) / 2500.0, 2.0 * barrier(gas))


@pytest.fixture(autouse=True)
def _empty_cache():
    """Every test starts with no memoised tables, so build counts mean what they say."""
    clear_rate_table_cache()
    yield
    clear_rate_table_cache()


# ------------------------------------------------------------- accuracy


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_the_table_reproduces_the_rate_to_the_requested_tolerance(gas):
    rate = ppt(gas)
    table = build_rate_table(rate, default_range(gas))
    assert table.error <= DEFAULT_RTOL
    # Probed on a grid unrelated to the one the build used, so this is not the build's
    # own convergence test repeated.
    probe = np.geomspace(table.field_min, table.field_max, 5000)
    assert np.max(np.abs(table(probe) / rate(probe) - 1.0)) <= DEFAULT_RTOL


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_the_adk_table_reproduces_the_rate(gas):
    rate = adk_oracle(gas)
    table = build_rate_table(rate, default_range(gas))
    probe = np.geomspace(table.field_min, table.field_max, 5000)
    assert np.max(np.abs(table(probe) / rate(probe) - 1.0)) <= DEFAULT_RTOL


def test_a_tighter_tolerance_gives_a_bigger_table():
    gas = Gas.Ar
    loose = build_rate_table(ppt(gas), default_range(gas), rtol=1e-4)
    tight = build_rate_table(ppt(gas), default_range(gas), rtol=1e-8)
    assert tight.n_knots > loose.n_knots
    assert tight.error < loose.error


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_the_tabulated_rate_is_smooth(gas):
    # The property a fixed transverse quadrature depends on: log W has bounded curvature
    # in log E and no chatter, so a Gauss rule over a mode profile converges spectrally
    # rather than algebraically.
    table = build_rate_table(ppt(gas), default_range(gas))
    log_field = np.linspace(math.log(table.field_min), math.log(table.field_max), 4096)
    curvature = np.diff(table.log_evaluate(log_field), n=2)
    assert np.all(np.isfinite(curvature))
    # The curvature must itself be a smooth function of the field: a corner in log W
    # makes it jump by its own size from one sample to the next (0.87 to 1.01 for a
    # clamped barrier-suppression correction). Measured 6e-4 to 1.1e-3 for optmat's PPT
    # of the noble gases.
    roughness = np.max(np.abs(np.diff(curvature))) / np.max(np.abs(curvature))
    assert roughness < 0.01


# ---------------------------------------------------------------- clipping


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_the_table_is_exactly_zero_below_its_lowest_field(gas):
    table = build_rate_table(ppt(gas), default_range(gas))
    below = np.geomspace(1e3, table.field_min * (1.0 - 1e-9), 64)
    # Exactly zero, like the rate itself: the plasma loss term divides by the field.
    assert np.array_equal(table(below), np.zeros_like(below))
    assert float(table(0.0)) == 0.0


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_the_table_holds_its_last_value_above_its_highest_field(gas):
    table = build_rate_table(ppt(gas), default_range(gas))
    above = np.geomspace(table.field_max * 1.001, 1e30, 64)
    # Held flat rather than extrapolated: a cubic in log W continued past the last knot
    # is what turns a diverged step into an overflow and then a NaN.
    assert np.all(table(above) == table.rate_max)
    assert table.rate_max == pytest.approx(float(table(table.field_max)), rel=1e-12)
    assert np.all(np.isfinite(table(above)))


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_the_lowest_field_is_the_rates_threshold(gas):
    rate = ppt(gas)
    table = build_rate_table(rate, default_range(gas))
    # The requested lower end is far below the threshold for every noble gas, so the
    # table starts at the threshold instead, with the margin that survives log and exp.
    assert default_range(gas)[0] < rate.threshold
    assert rate.threshold <= table.field_min < rate.threshold * 1.001
    assert float(rate(table.field_min)) > 0.0


def test_the_highest_field_stops_below_a_ceiling_that_bites():
    # Argon's uncorrected ADK reaches one ionisation per atomic unit of time inside the
    # default range. The clip puts a corner in log W, so the table stops just below it
    # and returns the ceiling above.
    gas = Gas.Ar
    rate = adk_oracle(gas)
    span = (barrier(gas) / 2500.0, 20.0 * barrier(gas))
    assert float(rate(span[1])) == MAX_RATE
    table = build_rate_table(rate, span)
    assert table.field_max < span[1]
    assert table.rate_max < MAX_RATE
    assert float(table(span[1])) == table.rate_max


# ------------------------------------------------------------ the cache key


def test_the_key_does_not_depend_on_keyword_order():
    first = adk_oracle(Gas.Ar, occupancy=1, cycle_averaged=True)
    second = adk_oracle(Gas.Ar, cycle_averaged=True, occupancy=1)
    assert rate_table_key(first, (1e9, 1e11), 1e-6) == rate_table_key(
        second, (1e9, 1e11), 1e-6
    )


def test_the_key_starts_with_the_schema_version():
    key = rate_table_key(adk_oracle(Gas.He), (1e9, 1e11), 1e-6)
    assert key[0] == SCHEMA_VERSION
    assert hash(key) == hash(rate_table_key(adk_oracle(Gas.He), (1e9, 1e11), 1e-6))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            (adk_oracle(Gas.He), (1e9, 1e11), 1e-6),
            (adk_oracle(Gas.Ne), (1e9, 1e11), 1e-6),
        ),
        (
            (adk_oracle(Gas.He), (1e9, 1e11), 1e-6),
            (adk_oracle(Gas.He), (2e9, 1e11), 1e-6),
        ),
        (
            (adk_oracle(Gas.He), (1e9, 1e11), 1e-6),
            (adk_oracle(Gas.He), (1e9, 2e11), 1e-6),
        ),
        (
            (adk_oracle(Gas.He), (1e9, 1e11), 1e-6),
            (adk_oracle(Gas.He), (1e9, 1e11), 1e-7),
        ),
        (
            (adk_oracle(Gas.He), (1e9, 1e11), 1e-6),
            (adk_oracle(Gas.He, cycle_averaged=True), (1e9, 1e11), 1e-6),
        ),
    ],
)
def test_anything_that_changes_the_table_changes_the_key(first, second):
    assert rate_table_key(*first) != rate_table_key(*second)


def test_the_digest_is_stable_across_processes():
    # Python salts the hash of a string per process, so a digest for a cache filename
    # cannot come from hash(); this is the guard on that.
    tests = str(Path(__file__).resolve().parents[1])
    program = (
        f"import sys; sys.path.insert(0, {tests!r});"
        "from rate_oracles import adk_oracle;"
        "from solprop.translate.ionisation import build_rate_table;"
        "print(build_rate_table(adk_oracle('He'), (2e9, 1e11), rtol=1e-3).key)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    here = build_rate_table(adk_oracle("He"), (2e9, 1e11), rtol=1e-3)
    assert result.stdout.strip() == here.key
    assert len(here.key) == 16


def test_the_cache_returns_the_same_table():
    rate = ppt(Gas.Ne)
    span = default_range(Gas.Ne)
    first = build_rate_table(rate, span)
    assert build_rate_table(rate, span) is first
    # An equal rate built separately is the same key, so it hits the same entry.
    assert build_rate_table(ppt(Gas.Ne), span) is first
    clear_rate_table_cache()
    assert build_rate_table(rate, span) is not first


def test_the_cache_can_be_bypassed():
    rate = adk_oracle(Gas.He)
    span = default_range(Gas.He)
    first = build_rate_table(rate, span, use_cache=False)
    second = build_rate_table(rate, span, use_cache=False)
    assert first is not second
    assert np.array_equal(first.log_rate, second.log_rate)
    # Nothing was recorded, so the next cached build is a fresh object.
    assert build_rate_table(rate, span) is not first


# ------------------------------------------------------------ the table object


def test_the_knots_are_uniform_in_the_logarithm_of_the_field():
    table = build_rate_table(adk_oracle(Gas.Ar), default_range(Gas.Ar))
    spacing = np.diff(table.log_field)
    assert np.allclose(spacing, table.d_log_field, rtol=1e-12)
    assert table.field[0] == pytest.approx(table.field_min, rel=1e-12)
    assert table.field[-1] == pytest.approx(table.field_max, rel=1e-12)
    assert table.n_knots == table.coefficients.shape[1] + 1


def test_the_table_is_even_in_the_field_and_keeps_its_shape():
    table = build_rate_table(adk_oracle(Gas.Ar), default_range(Gas.Ar))
    fields = np.geomspace(table.field_min, table.field_max, 32)
    assert np.array_equal(table(fields), table(-fields))
    assert table(fields[0]).shape == ()
    assert table(fields.reshape(4, 8)).shape == (4, 8)


def test_the_stored_knot_values_agree_with_the_interpolant():
    table = build_rate_table(adk_oracle(Gas.Kr), default_range(Gas.Kr))
    # log_rate is diagnostic; evaluation goes through the coefficients, and the two must
    # agree to the rounding of the spline solve.
    assert np.allclose(table.log_evaluate(table.log_field), table.log_rate, atol=1e-12)


def test_the_build_is_summarised_at_debug_level(caplog):
    # Translation summaries are logged at debug so a run can say what it built and how
    # big it is, guarded so that nothing is formatted when nobody is listening.
    with caplog.at_level("DEBUG", logger="solprop.translate.ionisation"):
        table = build_rate_table(adk_oracle(Gas.He), (2e9, 1e11), rtol=1e-3)
    assert table.key in caplog.text
    assert "knots" in caplog.text
    assert "Ammosov" in caplog.text


# --------------------------------------------------------------- error paths


@pytest.mark.parametrize(
    "span",
    [
        (0.0, 1e11),
        (-1e9, 1e11),
        (1e11, 1e9),
        (1e9, 1e9),
        (math.inf, 1e11),
        (1e9, math.nan),
    ],
)
def test_a_bad_field_range_is_rejected(span):
    with pytest.raises(TranslationError, match="field_range must be"):
        build_rate_table(adk_oracle(Gas.He), span)


@pytest.mark.parametrize("bad", [0.0, -1e-6, math.inf, math.nan])
def test_a_bad_tolerance_is_rejected(bad):
    with pytest.raises(TranslationError, match="rtol must be positive"):
        build_rate_table(adk_oracle(Gas.He), (1e9, 1e11), rtol=bad)


def test_a_range_entirely_below_the_threshold_is_rejected():
    with pytest.raises(TranslationError, match="nothing to tabulate"):
        build_rate_table(adk_oracle(Gas.He), (1e6, 1e8))


def test_a_range_entirely_above_the_ceiling_is_rejected():
    gas = Gas.Ar
    span = (0.5 * barrier(gas), 2.0 * barrier(gas))
    # A ceiling below the rate at the bottom of the range leaves nothing to tabulate.
    ceiling = float(adk_oracle(gas)(span[0])) / 10.0
    with pytest.raises(TranslationError, match="at its ceiling"):
        build_rate_table(adk_oracle(gas, maximum=ceiling), span)


def test_a_rate_that_will_not_resolve_is_rejected():
    # 65 knots resolve optmat's argon PPT to about 1e-5 in log W, far from 1e-12.
    gas = Gas.Ar
    with pytest.raises(TranslationError, match="did not reach"):
        build_rate_table(ppt(gas), default_range(gas), rtol=1e-12, max_knots=65)


def working_table():
    """A valid table to mutate in the validation tests."""
    return build_rate_table(adk_oracle(Gas.He), (2e9, 1e11), rtol=1e-3, use_cache=False)


def test_a_malformed_table_is_rejected():
    good = working_table()
    fields = {
        "log_rate": good.log_rate,
        "coefficients": good.coefficients,
        "log_field_min": good.log_field_min,
        "d_log_field": good.d_log_field,
        "field_min": good.field_min,
        "field_max": good.field_max,
        "rate_max": good.rate_max,
        "error": good.error,
        "key": good.key,
    }
    with pytest.raises(TranslationError, match="d_log_field must be positive"):
        RateTable(**{**fields, "d_log_field": 0.0})
    with pytest.raises(TranslationError, match="must have 4 rows"):
        RateTable(**{**fields, "coefficients": good.coefficients[:3]})
    with pytest.raises(TranslationError, match="one column per interval"):
        RateTable(**{**fields, "coefficients": good.coefficients[:, :-1]})
    with pytest.raises(TranslationError, match="0 < field_min < field_max"):
        RateTable(**{**fields, "field_min": good.field_max * 2.0})
    with pytest.raises(TranslationError, match="must be finite"):
        RateTable(**{**fields, "rate_max": math.inf})
    broken = good.log_rate.copy()
    broken[0] = math.nan
    with pytest.raises(TranslationError, match="must be finite"):
        RateTable(**{**fields, "log_rate": broken})


# ------------------------------------------------------------------- slow lane


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_a_smooth_rate_keeps_the_table_small(gas):
    # A converged table of a rate with no corner in log W costs a few hundred knots;
    # measured 257 for optmat's PPT of every noble gas at the default tolerance.
    assert build_rate_table(ppt(gas), default_range(gas)).n_knots <= 1024


# This witness subtracts log rates of order 10--70 to measure errors down to
# 3e-9. Its floating-point resolution belongs to those operands, not the residual.
# The power-of-two operand bound is verified on every probe below. Budget sixteen
# rounded stages per evaluated difference (six Horner operations, three local-
# coordinate operations, exp/log and subtraction, plus four coefficient values),
# and compare two independently rounded machine evaluations. This is a numerical
# resolution allowance, not uncertainty in the physical ionisation model.
IONISATION_LOG_SCALE = 128.0
IONISATION_LOG_OPERATIONS = 32
IONISATION_LOG_ROUNDOFF = (
    IONISATION_LOG_OPERATIONS
    * np.finfo(float).eps
    / (1.0 - IONISATION_LOG_OPERATIONS * np.finfo(float).eps)
    * IONISATION_LOG_SCALE
)
IONISATION_RESOLVED_FACTOR = 1024

# The record of the ladder below, the "ionisation rate tables" row of
# `docs/dev/accuracy.md` section 1. Every rung must retain at least ten binary digits
# above the absolute resolution floor, independently of the fourth-order gate.
IONISATION_KNOTS = Study(
    name="ionisation_knots",
    title="Interpolation error of an ionisation rate table against its knot count",
    case={
        "rate": "optmat's PPT (ppt_ilkov), neon, 800 nm",
        "why_neon": "its rate carries no clamped barrier-suppression correction, so "
        "nothing but the interpolation error is measured",
        "field_range": "the default, Eb/2500 to 2 Eb, raised to the rate's threshold "
        "and lowered to its maximum field",
        "probe": "9999 points, uniform in log field",
    },
    axis={"name": "n_intervals", "unit": "intervals", "label": "cubic intervals"},
    columns=(
        Column(
            name="knots",
            unit="knots",
            why="the table's own size, an input rather than a measurement, so it is "
            "compared exactly",
            rtol=0.0,
        ),
        Column(
            name="error",
            unit="1 (absolute, in the natural logarithm of the rate)",
            why="absolute evaluation-scale roundoff: 32 rounded operations at "
            "a verified log-rate magnitude bound of 128, including both machines",
            rtol=0.0,
            atol=IONISATION_LOG_ROUNDOFF,
        ),
        Column(
            name="ratio",
            unit="1",
            why="each error exceeds 1024 absolute roundoff floors; propagating "
            "both operand bounds gives relative ratio uncertainty 2/(1024-1); "
            "null on the coarsest rung, which has no pair",
            rtol=2.0 / (IONISATION_RESOLVED_FACTOR - 1),
        ),
    ),
)


def test_the_error_falls_at_fourth_order_as_the_knots_double(request):
    # A cubic interpolant of a smooth function has error h^4 |f''''| / 384, so doubling
    # the knot count divides the error by sixteen. Neon is used because its rate carries
    # no clamped correction at all, so nothing but the interpolation error is measured.
    rate = ppt(Gas.Ne)
    span = default_range(Gas.Ne)
    errors = []
    for intervals in (64, 128, 256):
        # rtol below anything achievable forces the build to stop at max_knots, and the
        # error it reports is then the one at that knot count.
        with pytest.raises(TranslationError, match="did not reach"):
            build_rate_table(
                rate,
                span,
                rtol=1e-30,
                n_intervals=intervals,
                max_knots=intervals + 1,
                use_cache=False,
            )
        table = build_rate_table(
            rate,
            span,
            rtol=1e30,
            n_intervals=intervals,
            max_knots=intervals + 1,
            use_cache=False,
        )
        log_field = np.linspace(
            math.log(table.field_min), math.log(table.field_max), 9999
        )
        interpolated = table.log_evaluate(log_field)
        reference = np.log(rate(np.exp(log_field)))
        assert np.max(np.abs(interpolated)) <= IONISATION_LOG_SCALE
        assert np.max(np.abs(reference)) <= IONISATION_LOG_SCALE
        errors.append(float(np.max(np.abs(interpolated - reference))))
    # Measured ratios are close to sixteen; 10 to 24 allows for the fifth-derivative
    # term that is not yet negligible at 64 intervals. The errors themselves are
    # 1.123e-6, 5.127e-8 and 3.203e-9, so 21.90 and 16.01, and they are the record the
    # first figure of `docs/dev/accuracy.md` section 1 is drawn from. (On solprop's
    # former PPT, deleted in SEP 3, Phase 4b: 7.439e-6, 4.815e-7 and 3.062e-8.)
    _assert_resolved_fourth_order(errors)
    counts = (64, 128, 256)
    record(
        IONISATION_KNOTS,
        request=request,
        rows=[
            {
                "series": "PPT neon",
                "axis": intervals,
                "knots": float(intervals + 1),
                "error": error,
                "ratio": None if index == 0 else errors[index - 1] / error,
            }
            for index, (intervals, error) in enumerate(zip(counts, errors, strict=True))
        ],
    )


def _assert_resolved_fourth_order(errors):
    assert all(math.isfinite(error) for error in errors)
    assert min(errors) > IONISATION_RESOLVED_FACTOR * IONISATION_LOG_ROUNDOFF
    for coarse, fine in pairwise(errors):
        assert 10.0 < coarse / fine < 24.0


@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0, 32.0])
def test_knot_ladder_witness_rejects_incorrect_convergence_order(ratio):
    # Far above the cancellation floor: failure must be the order gate itself.
    with pytest.raises(AssertionError):
        _assert_resolved_fourth_order([1e-3, 1e-3 / ratio, 1e-3 / ratio**2])


def test_knot_ladder_witness_refuses_unresolved_error_ratios():
    # An exact ratio of sixteen is uninformative when its denominator is noise.
    with pytest.raises(AssertionError):
        _assert_resolved_fourth_order(
            [
                256 * IONISATION_LOG_ROUNDOFF,
                16 * IONISATION_LOG_ROUNDOFF,
                IONISATION_LOG_ROUNDOFF,
            ]
        )
