# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The band rule's design and certification (SEP 18, section 7 item 1)."""

import math

import numpy as np
import pytest

from bounds import assert_within, stencil_bound
from plasma_cases import zero_rate
from rate_oracles import adk_oracle
from solprop.exceptions import GridError, TranslationError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Gas
from solprop.profiles import Constant
from solprop.translate import plasma_integration
from solprop.translate.plasma_integration import (
    ABOVE_BAND_BOUND,
    DESIGN_TOLERANCE,
    MAX_HALF_WIDTH,
    design_running_integral,
    integration_record,
    trapezoid_multiplier,
)
from solprop.translate.responses import _plasma_integral, compile_responses


def ladder_grid(ffac):
    """Section 7 item 4's helium grid at oversampling factor `ffac`."""
    return FieldGrid.from_limits(1.0, 800e-9, (90e-9, 4e-6), 400e-15, ffac=ffac)


def composite(taps, theta):
    """H = C rho, evaluated afresh here rather than through the design module."""
    response = taps[0] + sum(
        2.0 * taps[k] * np.cos(k * theta) for k in range(1, taps.size)
    )
    return response * trapezoid_multiplier(theta)


# SEP 18's certified ladder (section 4.3's table): K and the certified in-band error.
LADDER = [
    (3.6, 8, 6.7e-6),
    (4.0, 7, 3.6e-6),
    (5.0, 5, 4.4e-6),
    (6.0, 4, 4.92e-6),
    (8.0, 3, 5.3e-6),
    (12.0, 2, 9.6e-6),
    (24.0, 2, 1.5e-7),
    (48.0, 1, 1.9e-6),
    (96.0, 1, 1.2e-7),
]


@pytest.mark.parametrize(("ffac", "half_width", "error"), LADDER)
def test_the_design_ladder_is_the_sep_s(ffac, half_width, error):
    grid = ladder_grid(ffac)
    rule = design_running_integral(grid, "band")
    assert rule.policy == "band"
    assert rule.half_width == half_width
    assert rule.certified_error == pytest.approx(error, rel=0.05)
    assert rule.dt == grid.dt_fine
    assert rule.band_edge == grid.omega_max_window
    # C(0) = 1 to round-off.
    assert rule.taps[0] + 2.0 * rule.taps[1:].sum() == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize(("ffac", "half_width", "error"), LADDER)
def test_the_certificate_holds_on_points_the_design_never_saw(ffac, half_width, error):
    """An independent check: 100003 points, none of them a fit node or a sample."""
    grid = ladder_grid(ffac)
    rule = design_running_integral(grid, "band")
    theta_b = grid.omega_max_window * grid.dt_fine
    band = np.linspace(0.0, theta_b, 100003)
    in_band = float(np.max(np.abs(composite(rule.taps, band) - 1.0)))
    assert in_band <= rule.certified_error * (1.0 + 1e-9)
    assert in_band <= DESIGN_TOLERANCE
    above = np.linspace(theta_b, math.pi, 100003)
    assert float(np.max(np.abs(composite(rule.taps, above)))) <= ABOVE_BAND_BOUND


def test_a_grid_below_ffac_3_55_is_refused_by_name():
    with pytest.raises(TranslationError, match=r"K <= 8.*best 1\.2.e-05.*finer dt"):
        design_running_integral(ladder_grid(3.5), "band")


def test_a_fine_enough_grid_needs_no_filter():
    """K = 0 once 1 - rho(theta_b) is inside the tolerance, still recorded as band."""
    rule = design_running_integral(ladder_grid(700.0), "band")
    assert rule.policy == "band"
    assert rule.half_width == 0
    assert rule.taps.tolist() == [1.0]
    assert rule.dt is not None and rule.band_edge is not None
    theta_b = rule.dt * rule.band_edge
    assert rule.certified_error == pytest.approx(
        1.0 - float(trapezoid_multiplier(np.array(theta_b))), rel=1e-9
    )


def test_a_user_dt_gets_its_own_design():
    """The taps follow dt_fine and the band edge, not the oversampling factor alone."""
    grid = FieldGrid.from_limits(
        1.0, 800e-9, (90e-9, 4e-6), 400e-15, dt=0.8 * ladder_grid(6.0).dt_fine * 6 / 2
    )
    rule = design_running_integral(grid, "band")
    assert rule.dt == grid.dt_fine
    assert rule.certified_error <= DESIGN_TOLERANCE
    assert 0 < rule.half_width <= MAX_HALF_WIDTH


def test_the_trapezoid_is_the_legacy_constant():
    rule = design_running_integral(ladder_grid(6.0), "trapezoid")
    assert rule.policy == "trapezoid"
    assert rule.taps.tolist() == [1.0]
    assert (rule.dt, rule.band_edge, rule.boundary, rule.exposure) == (
        None,
        None,
        "none",
        "none",
    )


def test_the_design_is_shared_by_every_problem_on_one_grid():
    assert design_running_integral(ladder_grid(6.0), "band") is (
        design_running_integral(ladder_grid(6.0), "band")
    )


def test_the_record_carries_the_taps_exactly():
    # Lossless storage of the designed taps: a structural contract.
    rule = design_running_integral(ladder_grid(6.0), "band")
    record = integration_record(rule)
    assert record["policy"] == "band"
    assert record["half_width"] == 4
    assert np.array_equal(np.asarray(record["taps"]), rule.taps)
    assert all(type(tap) is float for tap in record["taps"])  # type: ignore[union-attr]
    assert record["theta_b"] == rule.dt * rule.band_edge  # type: ignore[operator]
    assert record["exposure"] == "floor-at-zero"


# ---------------------------------------------------------- the value type


def test_a_band_rule_needs_its_spacing_and_band_edge():
    with pytest.raises(GridError, match="positive finite dt"):
        RunningIntegral("band", np.ones(1), band_edge=1.0)
    with pytest.raises(GridError, match="positive finite band_edge"):
        RunningIntegral("band", np.ones(1), dt=1.0)


def test_the_trapezoid_is_one_identity_tap():
    with pytest.raises(GridError, match="one identity tap"):
        RunningIntegral(
            "trapezoid", np.array([1.1, -0.05]), boundary="none", exposure="none"
        )


def test_the_taps_are_frozen():
    rule = design_running_integral(ladder_grid(6.0), "band")
    with pytest.raises(ValueError, match="read-only"):
        rule.taps[0] = 2.0


def test_a_window_shorter_than_the_reach_is_refused():
    rule = RunningIntegral("band", np.array([1.1, -0.05]), dt=1.0, band_edge=1.0)
    with pytest.raises(GridError, match="at least 2 samples"):
        rule.integrate(np.ones(1), 1.0)


def test_point_reflection_is_exact_for_affine_integrands():
    """Constants and ramps pass the filter unchanged, ends included."""
    rule = design_running_integral(ladder_grid(6.0), "band")
    n = np.arange(32.0)
    for values in (np.full(32, 3.0), 2.0 - 0.5 * n):
        assert np.allclose(rule.filter(values), values, rtol=0.0, atol=1e-13)


def test_the_filter_sums_the_declared_stencil():
    """Centre first, then each pair added before it is weighted, to rounding.

    The reference below sums in the declared order; a vectorised or reassociated
    filter is admitted within the paired stencil's rounding bound
    (:func:`bounds.stencil_bound`, SEP 25 section 4.6), doubled for the two sums.
    """
    rule = design_running_integral(ladder_grid(6.0), "band")
    values = np.random.default_rng(18).normal(size=64)
    half, taps = rule.half_width, rule.taps
    reach = np.arange(half, 0, -1)
    extended = np.concatenate(
        [
            2.0 * values[:1] - values[reach],
            values,
            2.0 * values[-1:] - values[values.size - 1 - reach[::-1]],
        ]
    )
    expected = np.empty_like(values)
    for n in range(values.size):
        total = taps[0] * values[n]
        for k in range(1, half + 1):
            total = total + taps[k] * (extended[half + n - k] + extended[half + n + k])
        expected[n] = total
    bound = 2 * stencil_bound(taps, extended)[half : half + values.size]
    assert_within(rule.filter(values), expected, bound, elementwise=True)


# ----------------------------------------------------------- translation


def test_a_problem_compiles_one_rule_for_its_grid():
    grid = ladder_grid(6.0)
    compiled = compile_responses(
        [PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar), integration="band")],
        grid,
        MarcatiliMode(125e-6, "He", 1.0, cladding=1.45),
        density=Constant(1.0),
        z_max=1.0,
        ionisation_field_range=(1e9, 1e11),
    )
    assert compiled.plasma_integral is design_running_integral(grid, "band")


def test_a_problem_without_plasma_carries_the_trapezoid():
    assert _plasma_integral((), ladder_grid(6.0)).policy == "trapezoid"


def test_mixed_rules_in_one_problem_are_refused_by_name():
    """The channels share their current integrals, so they share one rule."""
    responses = (
        PlasmaResponse(Gas.Ar, zero_rate, species="argon", integration="band"),
        PlasmaResponse(Gas.He, zero_rate, species="helium", integration="trapezoid"),
    )
    with pytest.raises(TranslationError, match="argon: 'band', helium: 'trapezoid'"):
        _plasma_integral(responses, ladder_grid(6.0))


# ------------------------------------------------------------- the refusals


def test_a_band_edge_at_nyquist_is_refused():
    """``ffac = 2`` puts 1.1 omega_max above the fine grid's Nyquist frequency."""
    with pytest.raises(TranslationError, match="Nyquist"):
        design_running_integral(ladder_grid(2.0), "band")


def test_an_ill_conditioned_fit_is_refused(monkeypatch):
    """No admitted grid nears the limit (4.3e4 at most), so it is lowered here."""
    monkeypatch.setattr(plasma_integration, "CONDITION_LIMIT", 0.5)
    plasma_integration._design.cache_clear()
    try:
        with pytest.raises(TranslationError, match="ill-conditioned"):
            design_running_integral(ladder_grid(6.0), "band")
    finally:
        plasma_integration._design.cache_clear()


def test_the_certificate_refines_an_interior_maximum():
    """A peak between samples is found by the bounded search, not just sampled."""
    found = plasma_integration._certified_max(
        lambda theta: np.sin(theta * 1.234567), 0.0, 3.0
    )
    assert found == pytest.approx(1.0, abs=1e-14)


def test_malformed_taps_are_refused():
    with pytest.raises(GridError, match="non-empty finite"):
        RunningIntegral("band", np.array([np.nan]), dt=1.0, band_edge=1.0)
    with pytest.raises(GridError, match="non-empty finite"):
        RunningIntegral("band", np.zeros(0), dt=1.0, band_edge=1.0)


def test_a_band_rule_reflects_and_floors():
    with pytest.raises(GridError, match="point reflection"):
        RunningIntegral("band", np.ones(1), dt=1.0, band_edge=1.0, boundary="none")
