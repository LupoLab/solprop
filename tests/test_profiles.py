# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import itertools
import json
import logging
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad

from solprop.constants import K_B
from solprop.exceptions import TranslationError
from solprop.physdata import Gas, IdealGas, density, species
from solprop.physdata.species import OptmatGas, Species
from solprop.profiles import (
    DEFAULT_PRESSURE_FLOOR_FRACTION,
    MIN_STAGE_PRESSURE,
    Constant,
    DensityPath,
    FromCallable,
    Linear,
    PiecewiseLinear,
    PressureGradient,
    Scaled,
    Sum,
    ZProfile,
    as_profile,
    hydraulic_integral,
    profile_identity,
)
from solprop.translate.chebyshev import build_from_paths


def profiles_under_test():
    """One instance of every concrete profile, for the shared protocol tests."""
    fill = PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.2])
    bore = PiecewiseLinear.from_stages(
        [0.0, 0.3, 1.0, 1.5], [125e-6, 100e-6, 60e-6, 90e-6]
    )
    return [
        Constant(125e-6, (0.0, 1.5)),
        Linear(125e-6, 75e-6, 1.5),
        PiecewiseLinear.from_stages([0.0, 0.5, 1.5], [1.0, 3.0, 2.0]),
        fill,
        PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.2], radius=bore),
        DensityPath(Gas.He, fill),
        FromCallable(np.cos, (0.0, 1.5), slope=lambda z: -np.sin(z)),
    ]


ALL_PROFILES = profiles_under_test()
PROFILE_IDS = [type(profile).__name__ for profile in ALL_PROFILES]


# ------------------------------------------------------------- the protocol


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=PROFILE_IDS)
def test_every_profile_satisfies_the_protocol(profile):
    assert isinstance(profile, ZProfile)
    start, end = profile.z_span
    assert end > start
    edges = profile.breakpoints
    assert edges.ndim == 1
    assert np.all(np.diff(edges) > 0.0)
    assert isinstance(profile.derivative_is_analytic, bool)
    assert isinstance(profile.is_constant, bool)


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=PROFILE_IDS)
def test_every_profile_is_vectorised_and_shape_preserving(profile):
    start, end = profile.z_span
    z = np.linspace(start, min(end, start + 1.5), 17)
    values = profile(z)
    assert values.shape == z.shape
    assert profile.derivative(z).shape == z.shape
    assert np.all(np.isfinite(values))
    # Element by element, so a vectorised path cannot disagree with itself.
    for position, expected in zip(z, values, strict=True):
        assert profile(np.array([position]))[0] == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=PROFILE_IDS)
def test_every_profile_rejects_a_non_finite_position(profile):
    with pytest.raises(TranslationError, match="must be finite"):
        profile(np.array([np.nan]))
    with pytest.raises(TranslationError, match="must be finite"):
        profile.derivative(np.array([np.inf]))


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=PROFILE_IDS)
def test_analytic_derivatives_match_finite_differences(profile):
    """Richardson-extrapolated central differences, away from the kinks.

    A central difference across a kink compares two different stages and is
    meaningless, so the sample points are placed at stage midpoints. What is left is
    the h^2 truncation error, cancelled by Richardson, and rounding; the residual is
    below 1e-9 relative for every path here, and 1e-7 keeps two orders in hand.
    """
    if not profile.derivative_is_analytic:
        pytest.skip("declares a differenced derivative")
    start, end = profile.z_span
    edges = np.unique(np.concatenate([[start], profile.breakpoints, [end]]))
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    z = np.concatenate(
        [np.linspace(a, b, 7)[1:-1] for a, b in itertools.pairwise(edges)] + [midpoints]
    )
    analytic = profile.derivative(z)
    if np.max(np.abs(analytic)) == 0.0:
        assert np.all(analytic == 0.0)
        return
    step = 1e-4 * np.min(np.diff(edges))
    wide = (profile(z + step) - profile(z - step)) / (2.0 * step)
    narrow = (profile(z + step / 2) - profile(z - step / 2)) / step
    richardson = (4.0 * narrow - wide) / 3.0
    assert np.max(np.abs(richardson - analytic)) / np.max(np.abs(analytic)) < 1e-7


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=PROFILE_IDS)
def test_every_profile_can_be_handed_to_the_chebyshev_builder(profile):
    """A profile is usable as a path with no adaptor: that is the point of the shape."""
    start, end = profile.z_span
    end = min(end, start + 1.5)
    interior = profile.breakpoints[
        (profile.breakpoints > start) & (profile.breakpoints < end)
    ]
    table = build_from_paths(
        [profile],
        lambda _z, value: value[:, None] * np.array([1.0, 2.0]),
        start,
        end,
        mandatory_breakpoints=interior,
    )
    assert not table.budget_exceeded
    np.testing.assert_allclose(table.hard_kinks, interior, rtol=0.0, atol=0.0)
    probe = 0.5 * (start + end)
    np.testing.assert_allclose(
        table(probe), profile(np.array([probe]))[0] * np.array([1.0, 2.0]), rtol=1e-9
    )


# --------------------------------------------------------------- Constant


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (Constant(1.0), True),
        (Linear(1.0, 1.0, 1.0), True),
        (Linear(1.0, 2.0, 1.0), False),
        (PiecewiseLinear.from_stages([0.0, 1.0, 2.0], [3.0, 3.0, 3.0]), True),
        (PiecewiseLinear.from_stages([0.0, 1.0, 2.0], [3.0, 3.0, 4.0]), False),
        (PressureGradient.two_point(1.0, 2.0, 2.0), True),
        (PressureGradient.two_point(1.0, 2.0, 1.0), False),
        (DensityPath(Gas.He, Constant(2.0, (0.0, 1.0))), True),
        (DensityPath(Gas.He, PressureGradient.two_point(1.0, 2.0, 1.0)), False),
        (FromCallable(lambda z: np.full_like(z, 2.0), (0.0, 1.0)), False),
    ],
)
def test_is_constant_reports_whether_the_path_varies(profile, expected):
    """The transform reuses its matrices on a constant path, so this is load bearing."""
    assert profile.is_constant is expected


def test_constant_is_flat_and_kinkless():
    profile = Constant(2.5)
    z = np.linspace(0.0, 1000.0, 8)
    np.testing.assert_array_equal(profile(z), np.full(8, 2.5))
    np.testing.assert_array_equal(profile.derivative(z), np.zeros(8))
    assert profile.breakpoints.size == 0
    assert profile.z_span == (0.0, math.inf)


# ------------------------------------------------------- PiecewiseLinear


def test_piecewise_linear_breaks_at_every_stage_boundary():
    profile = PiecewiseLinear.from_stages([0.0, 0.4, 1.0], [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(profile.breakpoints, [0.0, 0.4, 1.0])
    np.testing.assert_allclose(
        profile(np.array([0.0, 0.2, 0.4, 0.7, 1.0])), [1.0, 2.0, 3.0, 2.5, 2.0]
    )


def test_piecewise_linear_clamps_outside_its_range():
    profile = PiecewiseLinear.from_stages([0.0, 1.0], [1.0, 3.0])
    np.testing.assert_allclose(profile(np.array([-1.0, 2.0])), [1.0, 3.0])
    np.testing.assert_array_equal(profile.derivative(np.array([-1.0, 2.0])), [0.0, 0.0])


def test_piecewise_linear_slope_is_the_stage_slope():
    profile = PiecewiseLinear.from_stages([0.0, 0.4, 1.0], [1.0, 3.0, 2.0])
    np.testing.assert_allclose(
        profile.derivative(np.array([0.2, 0.7])), [5.0, -5.0 / 3.0]
    )


@pytest.mark.parametrize(
    ("positions", "values", "message"),
    [
        ([0.0], [1.0], "at least two positions"),
        ([1.0, 0.0], [1.0, 2.0], "strictly increasing"),
        ([0.0, 0.0], [1.0, 2.0], "strictly increasing"),
        ([0.0, np.inf], [1.0, 2.0], "must be finite"),
        ([0.0, 1.0], [1.0, 2.0, 3.0], "one value per position"),
        ([0.0, 1.0], [1.0, np.nan], "values must be finite"),
    ],
)
def test_piecewise_linear_rejects_a_bad_table(positions, values, message):
    with pytest.raises(TranslationError, match=message):
        PiecewiseLinear.from_stages(positions, values)


# ------------------------------------------------------------------ Linear


def test_linear_is_the_two_stage_case():
    taper = Linear(125e-6, 75e-6, 1.5)
    z = np.linspace(-0.5, 2.0, 32)
    equivalent = PiecewiseLinear.from_stages([0.0, 1.5], [125e-6, 75e-6])
    # The same interpolation of the same two stages: a few roundings of each value
    # and slope, 4 eps. The declared breakpoints are the stored positions, exactly.
    rounding = 4 * np.finfo(float).eps
    np.testing.assert_allclose(taper(z), equivalent(z), rtol=rounding, atol=0)
    np.testing.assert_allclose(
        taper.derivative(z), equivalent.derivative(z), rtol=rounding, atol=0
    )
    np.testing.assert_array_equal(taper.breakpoints, equivalent.breakpoints)
    assert taper.z_span == (0.0, 1.5)


def test_linear_needs_a_positive_length():
    with pytest.raises(TranslationError, match="positive length"):
        Linear(1.0, 2.0, 0.0)


# --------------------------------------------------------- PressureGradient


def test_pressure_gradient_follows_the_steady_flow_profile():
    fill = PressureGradient.two_point(1.5, 0.8, 0.2)
    z = np.linspace(0.0, 1.5, 33)
    expected = np.sqrt(0.8**2 + (z / 1.5) * (0.2**2 - 0.8**2))
    np.testing.assert_allclose(fill(z), expected, rtol=1e-15)


def test_pressure_gradient_breaks_at_every_stage_boundary():
    positions = [0.0, 0.4, 0.9, 1.5]
    fill = PressureGradient.from_stages(positions, [0.1, 5.0, 3.0, 0.5])
    np.testing.assert_array_equal(fill.breakpoints, positions)
    # Continuous at each boundary, and equal to the declared stage pressure there.
    np.testing.assert_allclose(fill(np.asarray(positions)), fill.pressures, rtol=1e-15)


def test_pressure_gradient_clamps_outside_the_fill():
    fill = PressureGradient.two_point(1.0, 2.0, 1.0)
    np.testing.assert_allclose(fill(np.array([-1.0, 5.0])), [2.0, 1.0], rtol=1e-15)
    np.testing.assert_array_equal(fill.derivative(np.array([-1.0, 5.0])), [0.0, 0.0])


def test_pressure_gradient_derivative_is_the_analytic_one():
    fill = PressureGradient.two_point(1.5, 0.8, 0.2)
    z = np.linspace(0.05, 1.45, 21)
    expected = (0.2**2 - 0.8**2) / (2.0 * 1.5 * fill(z))
    np.testing.assert_allclose(fill.derivative(z), expected, rtol=1e-15)


def test_a_vacuum_end_is_floored_and_logged(caplog):
    """The substitution is loud: a silent one is exactly the trap being avoided."""
    with caplog.at_level(logging.WARNING, logger="solprop.profiles"):
        fill = PressureGradient.two_point(1.5, 0.8, 0.0)
    assert fill.floor == pytest.approx(DEFAULT_PRESSURE_FLOOR_FRACTION * 0.8)
    assert fill.pressures[-1] == fill.floor
    assert "trap 5" in caplog.text
    assert "1.5" in caplog.text
    # The point of the floor: a finite slope everywhere on the fill.
    assert np.all(np.isfinite(fill.derivative(np.linspace(0.0, 1.5, 101))))


def test_the_floor_scales_with_the_fill_pressure():
    low = PressureGradient.two_point(1.0, 0.8, 0.0)
    high = PressureGradient.two_point(1.0, 40.0, 0.0)
    assert high.floor / low.floor == pytest.approx(40.0 / 0.8)


def test_an_explicit_floor_is_used_as_given(caplog):
    with caplog.at_level(logging.WARNING, logger="solprop.profiles"):
        fill = PressureGradient.two_point(1.0, 1.0, 0.0, floor=1e-3)
    assert fill.floor == 1e-3
    assert fill.pressures[-1] == 1e-3
    assert "trap 5" in caplog.text


def test_a_zero_floor_keeps_the_vacuum_end_and_warns_harder(caplog):
    """Reproducing a Luna result must stay possible, but not by accident."""
    with caplog.at_level(logging.WARNING, logger="solprop.profiles"):
        fill = PressureGradient.two_point(1.0, 1.0, 0.0, floor=0.0)
    assert fill.pressures[-1] == 0.0
    assert "floor=0" in caplog.text
    # And this is what the floor exists to prevent.
    assert not np.isfinite(fill.derivative(np.array([1.0]))[0])


def test_a_fill_with_no_vacuum_end_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="solprop.profiles"):
        PressureGradient.two_point(1.0, 1.0, 0.5)
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("positions", "pressures", "message"),
    [
        ([0.0], [1.0], "at least two positions"),
        ([1.0, 0.0], [1.0, 2.0], "strictly increasing"),
        ([0.0, 1.0], [1.0, 2.0, 3.0], "one pressure per position"),
        ([0.0, 1.0], [1.0, -1.0], "not negative"),
        ([0.0, 1.0], [0.0, 0.0], "at least one non-zero"),
        ([0.0, 1.0], [1e-200, 0.0], "must stay above"),
    ],
)
def test_pressure_gradient_rejects_a_bad_table(positions, pressures, message):
    with pytest.raises(TranslationError, match=message):
        PressureGradient.from_stages(positions, pressures)


@pytest.mark.parametrize(
    "material",
    [Gas.Ar, species("Ar"), Species("Ar", species("Ar").susceptibility, IdealGas())],
    ids=["name", "optmat species", "ideal-gas species"],
)
def test_a_pressure_extremum_between_probes_evaluates_the_equation_of_state(
    caplog, material
):
    """A peak narrower than the construction probes.

    The spline is sampled between the extremes the breakpoints and 65 interior probes
    see; a 0.2 mm Gaussian peak between two of them was never sampled, and extrapolating
    the two nearly coincident knots of a one-bar fill gave arbitrary numbers, which the
    floor then turned into vacuum. The equation of state is the answer there.
    """
    peak = FromCallable(
        lambda z: 1.0 + 9.0 * np.exp(-(((z - 0.05078125) / 2e-4) ** 2)), (0.0, 0.1)
    )
    path = DensityPath(material, peak, 300.0)
    assert path._spline is not None
    assert path._spline.x[-1] < 1.001
    with caplog.at_level(logging.WARNING, logger="solprop"):
        at_peak, at_edge = path(np.array([0.05078125, 0.0]))
    # The path's own equation of state: optmat's for a name, the species' otherwise.
    state = (
        material.equation_of_state
        if isinstance(material, Species)
        else OptmatGas(Gas.Ar)
    )
    assert at_peak == state.density(10.0, 300.0)
    assert at_edge == state.density(1.0, 300.0)
    assert "outside the range sampled at construction" in caplog.text
    # The slope of the path is that of the pressure through the spline's end slope.
    assert np.isfinite(path.derivative(np.array([0.05078125, 0.0505]))).all()


def test_pressure_gradient_rejects_a_negative_floor():
    with pytest.raises(TranslationError, match="floor must not be negative"):
        PressureGradient.two_point(1.0, 1.0, 0.5, floor=-1e-6)


def test_pressure_gradient_needs_a_positive_length():
    with pytest.raises(TranslationError, match="positive length"):
        PressureGradient.two_point(0.0, 1.0, 0.5)


def test_the_direct_constructor_does_not_floor():
    """`from_stages` owns the floor; the raw constructor is for a validated table."""
    fill = PressureGradient(
        positions=np.array([0.0, 1.0]), pressures=np.array([1.0, 0.0]), floor=0.0
    )
    assert fill.pressures[-1] == 0.0
    with pytest.raises(TranslationError, match="strictly increasing"):
        PressureGradient(
            positions=np.array([1.0, 0.0]), pressures=np.array([1.0, 2.0]), floor=0.0
        )
    with pytest.raises(TranslationError, match="one pressure per position"):
        PressureGradient(
            positions=np.array([0.0, 1.0]), pressures=np.array([1.0]), floor=0.0
        )
    with pytest.raises(TranslationError, match="not negative"):
        PressureGradient(
            positions=np.array([0.0, 1.0]), pressures=np.array([1.0, -1.0]), floor=0.0
        )


# --------------------------------------------------------------- DensityPath


def test_density_path_matches_direct_equation_of_state_calls():
    """The spline is a performance device, not an approximation the physics feels.

    1e-9 relative is far tighter than the equation of state's own accuracy and than
    any use of the density, and it is what 1024 samples of a nearly linear function
    actually deliver.
    """
    fill = PressureGradient.two_point(1.5, 5.0, 0.5)
    path = DensityPath(Gas.Ar, fill)
    z = np.linspace(0.0, 1.5, 41)
    direct = np.array([density(Gas.Ar, float(value)) for value in fill(z)])
    np.testing.assert_allclose(path(z), direct, rtol=1e-9)


def test_density_path_is_analytic_when_its_pressure_path_is():
    fill = PressureGradient.two_point(1.0, 2.0, 0.5)
    assert DensityPath(Gas.He, fill).derivative_is_analytic
    rough = FromCallable(lambda z: 1.0 + z, (0.0, 1.0))
    assert not DensityPath(Gas.He, rough).derivative_is_analytic


def test_density_path_inherits_the_pressure_breakpoints_and_span():
    fill = PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.2])
    path = DensityPath(Gas.Ne, fill)
    np.testing.assert_array_equal(path.breakpoints, fill.breakpoints)
    assert path.z_span == fill.z_span


def test_density_path_handles_a_constant_fill():
    """A degenerate pressure range must still produce a usable spline."""
    path = DensityPath(Gas.He, Constant(2.0, (0.0, 1.0)))
    z = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(path(z), density(Gas.He, 2.0), rtol=1e-9)
    np.testing.assert_allclose(path.derivative(z), 0.0, atol=1e-6)


def test_density_path_falls_with_a_falling_pressure():
    fill = PressureGradient.two_point(1.0, 3.0, 0.1)
    path = DensityPath(Gas.He, fill)
    z = np.linspace(0.05, 0.95, 16)
    assert np.all(np.diff(path(z)) < 0.0)
    assert np.all(path.derivative(z) < 0.0)


def test_density_path_accepts_a_gas_name():
    fill = PressureGradient.two_point(1.0, 1.0, 0.5)
    assert DensityPath("He", fill).gas is Gas.He


# --------------------------------------------------------------- FromCallable


def test_from_callable_uses_the_supplied_slope():
    path = FromCallable(np.sin, (0.0, 1.0), slope=np.cos)
    z = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(path.derivative(z), np.cos(z), rtol=0.0, atol=0.0)
    assert path.derivative_is_analytic


def test_from_callable_differences_when_it_must():
    """The declared accuracy is what a differenced path is allowed to claim.

    Richardson-extrapolated central differences on a smooth path reach about 1e-10
    relative; the flag says the caller must not expect the 1e-16 that ADR-006 needs
    for beta_1.
    """
    path = FromCallable(np.sin, (0.0, 1.0))
    assert not path.derivative_is_analytic
    z = np.linspace(0.1, 0.9, 9)
    np.testing.assert_allclose(path.derivative(z), np.cos(z), rtol=1e-9)


def test_from_callable_declares_its_breakpoints():
    path = FromCallable(np.sin, (0.0, 1.0), kinks=np.array([0.3, 0.7]))
    np.testing.assert_array_equal(path.breakpoints, [0.3, 0.7])
    assert path.z_span == (0.0, 1.0)


@pytest.mark.parametrize(
    ("span", "kinks", "message"),
    [
        ((1.0, 1.0), (), "end > start"),
        ((0.0, 1.0), [0.7, 0.3], "strictly increasing"),
        ((0.0, 1.0), [-0.1, 0.5], "inside the span"),
        ((0.0, 1.0), [0.5, 1.5], "inside the span"),
    ],
)
def test_from_callable_rejects_a_bad_declaration(span, kinks, message):
    with pytest.raises(TranslationError, match=message):
        FromCallable(np.sin, span, kinks=np.asarray(kinks, dtype=float))


# ------------------------------------------------------------------ as_profile


def test_as_profile_wraps_a_number_and_passes_a_profile_through():
    wrapped = as_profile(125e-6)
    assert isinstance(wrapped, Constant)
    assert wrapped.value == 125e-6
    assert as_profile(3)(np.array([0.0]))[0] == 3.0
    existing = Linear(1.0, 2.0, 1.0)
    assert as_profile(existing) is existing


def test_as_profile_rejects_a_bool():
    """A bool passes the numeric tower but is never a radius or a pressure."""
    with pytest.raises(TranslationError, match="expected a number or a ZProfile"):
        as_profile(True)


# ------------------------------------------------------------------- property


def test_a_fill_into_the_vacuum_floor_never_drops_below_it():
    # 4.875 bar into the default floor, 4.875e-4 bar. The form
    # sqrt(p0**2 + f (p1**2 - p0**2)) cancels catastrophically as f -> 1 and returned a
    # pressure 1.58e-12 below the floor; the convex combination cannot.
    fill = PressureGradient.two_point(1.0, 4.875, 0.0)
    lowest = float(np.min(fill.pressures))
    z = np.linspace(0.0, 1.0, 100001)
    assert np.all(fill(z) >= lowest)
    assert float(fill(np.array([1.0]))[0]) == lowest


@settings(max_examples=200, deadline=None)
@given(
    length=st.floats(min_value=1e-3, max_value=10.0),
    start_bar=st.floats(min_value=0.0, max_value=50.0),
    end_bar=st.floats(min_value=0.0, max_value=50.0),
    fraction=st.floats(min_value=0.0, max_value=1.0),
)
def test_a_fill_is_monotone_bounded_and_finite(length, start_bar, end_bar, fraction):
    """No fill in the supported range produces a non-physical pressure or slope.

    A pressure below about 1e-154 bar squares to zero, which would collapse the
    steady-flow profile to 0/0; hypothesis found that, and the constructor now refuses
    it rather than returning NaN. It then found a fill from 0 to 6.5e-159 bar, whose
    highest pressure squares to a subnormal but whose floored end squares to zero, so
    the slope at the entrance was infinite; the guard is now ``MIN_STAGE_PRESSURE``.
    """
    highest = max(start_bar, end_bar)
    if highest < MIN_STAGE_PRESSURE:
        with pytest.raises(TranslationError):
            PressureGradient.two_point(length, start_bar, end_bar)
        return
    fill = PressureGradient.two_point(length, start_bar, end_bar)
    z = np.array([fraction * length])
    value = float(fill(z)[0])
    lower, upper = float(np.min(fill.pressures)), float(np.max(fill.pressures))
    assert math.isfinite(value)
    assert lower - 1e-12 <= value <= upper + 1e-12
    assert math.isfinite(float(fill.derivative(z)[0]))


def test_a_fill_whose_floored_end_would_underflow_is_refused():
    """The example hypothesis found: a subnormal square passes an exact-zero guard."""
    with pytest.raises(TranslationError, match="must stay above"):
        PressureGradient.two_point(1.0, 0.0, 6.45379260485051e-159)
    fill = PressureGradient.two_point(1.0, 0.0, 2.0 * MIN_STAGE_PRESSURE)
    assert math.isfinite(float(fill.derivative(np.array([0.0]))[0]))


def test_custom_species_scalar_density_and_spline_share_the_ideal_gas_law():
    material = dataclasses.replace(species("He"), equation_of_state=IdealGas())
    path = DensityPath(material, PressureGradient.two_point(1.0, 1.0, 2.0), 300.0)
    pressure = 1.5
    expected = pressure * 1e5 / (K_B * 300.0)
    # The scalar EOS is algebraic; the cubic spline reproduces this linear law
    # within a few operations of roundoff.
    assert path._density(pressure) == pytest.approx(
        expected, rel=8 * np.finfo(float).eps, abs=0
    )
    assert path(np.array([0.0]))[0] == pytest.approx(
        1e5 / (K_B * 300), rel=8 * np.finfo(float).eps, abs=0
    )


def test_profile_identity_separates_paths_a_repr_cannot():
    # numpy's repr summarises any array over 1000 elements, so two physically
    # different 1200-knot tapers print identically; the structural identity has
    # the SHA-256 of their float64 bytes and separates them.
    positions = np.linspace(0.0, 1.0, 1200)
    first = PiecewiseLinear(positions, np.linspace(1.0, 0.0, 1200))
    values = np.linspace(1.0, 0.0, 1200)
    values[600] += 1e-9
    second = PiecewiseLinear(positions, values)
    assert repr(np.asarray(first.values)) == repr(np.asarray(second.values))
    assert profile_identity(first) != profile_identity(second)
    assert profile_identity(first) == profile_identity(
        PiecewiseLinear(positions, np.linspace(1.0, 0.0, 1200))
    )
    values_identity = profile_identity(first)["values"]
    assert isinstance(values_identity, dict)
    assert values_identity["n"] == 1200
    assert "sha256" in values_identity


def test_profile_identity_does_not_depend_on_print_options():
    path = PiecewiseLinear(np.linspace(0.0, 1.0, 1200), np.linspace(1.0, 0.0, 1200))
    short = PiecewiseLinear.from_stages([0.0, 1.5], [125e-6, 75e-6])
    before = (profile_identity(path), profile_identity(short))
    original = np.get_printoptions()
    try:
        np.set_printoptions(precision=1, threshold=2, suppress=True)
        assert (profile_identity(path), profile_identity(short)) == before
    finally:
        np.set_printoptions(**original)


def test_profile_identity_is_exact_and_covers_every_declared_path():
    # Exact float.hex, not a rounded decimal: two radii a single ulp apart are
    # different paths and the identity says so.
    one = profile_identity(2.5e-7)
    other = profile_identity(float(np.nextafter(2.5e-7, 1.0)))
    assert one != other
    assert one == {
        "kind": "Constant",
        "value": (2.5e-7).hex(),
        "span": ["0x0.0p+0", "inf"],
    }
    identities = [
        profile_identity(path)
        for path in profiles_under_test()
        if not isinstance(path, FromCallable)
    ]
    kinds = {identity["kind"] for identity in identities}
    assert kinds == {
        "Constant",
        "PiecewiseLinear",
        "Linear",
        "PressureGradient",
        "DensityPath",
    }
    assert all(json.dumps(identity, sort_keys=True) for identity in identities)
    nested = profile_identity(Sum((Scaled(Linear(1.0, 0.0, 1.5), 2.0), Constant(1.0))))
    parts = nested["paths"]
    assert isinstance(parts, list)
    scaled, constant = parts
    assert isinstance(scaled, dict) and isinstance(constant, dict)
    assert (scaled["kind"], constant["kind"]) == ("Scaled", "Constant")
    inner = scaled["path"]
    assert isinstance(inner, dict)
    assert inner["kind"] == "Linear"


def test_profile_identity_refuses_an_arbitrary_callable():
    # A repr is not an identity: two different lambdas share one, and a measured
    # path has no defining numbers to digest.
    with pytest.raises(TranslationError, match="no structural identity"):
        profile_identity(FromCallable(np.sin, (0.0, 1.0), slope=np.cos))

    class Elsewhere:
        breakpoints = np.empty(0)
        jumps = np.empty(0)
        z_span = (0.0, 1.0)
        derivative_is_analytic = True
        is_constant = False

        def __call__(self, z):
            return np.zeros_like(z)

        def derivative(self, z):
            return np.zeros_like(z)

        def variation(self, start, end):
            return Constant(0.0).variation(start, end)

    with pytest.raises(TranslationError, match="no structural identity"):
        profile_identity(Elsewhere())


# ------------------------------------------------------- a fill that follows the bore

# Pennetta et al., APL Photonics 4, 056105 (2019), Fig. 5, with p0 = 10 bar and
# pL = 1 bar. The paper draws the geometries without stating them; these follow its
# Fig. 4 taper (32 um to 21.5 um diameter) and are illustrative: (a) a uniform section,
# a linear down-taper and a uniform waist, (b) linear down- and up-tapers around a
# central waist.
PENNETTA_A = PiecewiseLinear.from_stages(
    [0.0, 0.4, 0.6, 1.0], [16e-6, 16e-6, 10.75e-6, 10.75e-6]
)
PENNETTA_B = PiecewiseLinear.from_stages(
    [0.0, 0.2, 0.35, 0.65, 0.8, 1.0], [16e-6, 16e-6, 8e-6, 8e-6, 16e-6, 16e-6]
)


def _smooth(path):
    """The same radius as an opaque callable, so the fill integrates it by Chebyshev."""
    return FromCallable(path, path.z_span, path.breakpoints, slope=path.derivative)


def _eq6(radius, z, p0=10.0, pl=1.0, length=1.0):
    """Pennetta eq. (6) with the integral of a^-4 by adaptive quadrature."""

    def integral(upper):
        points = [b for b in radius.breakpoints if 0.0 < b < upper]
        value, error = quad(
            lambda x: float(radius(np.array([x]))[0]) ** -4,
            0.0,
            upper,
            points=points or None,
            epsabs=0.0,
            epsrel=1e-13,
            limit=200,
        )
        return value, error

    total, total_error = integral(length)
    values, errors = [], []
    for position in z:
        part, part_error = integral(float(position)) if position > 0 else (0.0, 0.0)
        values.append(math.sqrt(p0**2 - (p0**2 - pl**2) * part / total))
        errors.append(part_error / total + total_error * part / total**2)
    return np.array(values), np.array(errors)


@pytest.mark.parametrize("radius", [PENNETTA_A, PENNETTA_B], ids=["5a", "5b"])
@pytest.mark.parametrize("route", ["closed form", "chebyshev"])
def test_a_tapered_fill_is_pennettas_eq6(radius, route):
    bore = radius if route == "closed form" else _smooth(radius)
    fill = PressureGradient.two_point(1.0, 10.0, 1.0, radius=bore)
    z = np.linspace(0.0, 1.0, 41)
    expected, error = _eq6(radius, z)
    # dp = d(p^2) / (2p): the quadrature's own error estimate on I(z)/I(L), carried
    # into the pressure, plus a few ulps of the evaluation.
    bound = (100.0 - 1.0) * error / (2.0 * expected) + 8 * np.finfo(float).eps
    assert np.all(np.abs(fill(z) - expected) <= bound * expected)
    assert np.max(bound) < 1e-11


def test_a_waist_carries_most_of_the_pressure_drop():
    # Pennetta Fig. 5(b): the narrow waist decouples the two ends.
    tapered = PressureGradient.two_point(1.0, 10.0, 1.0, radius=PENNETTA_B)
    uniform = PressureGradient.two_point(1.0, 10.0, 1.0)
    before, after = tapered(np.array([0.35, 0.65]))
    assert before > 9.0 > uniform(np.array([0.35]))[0]
    assert after < 5.0


@pytest.mark.parametrize("route", ["closed form", "chebyshev"])
def test_a_tapered_fill_derivative_matches_a_finite_difference(route):
    bore = PENNETTA_B if route == "closed form" else _smooth(PENNETTA_B)
    fill = PressureGradient.from_stages([0.0, 0.5, 1.0], [2.0, 6.0, 0.3], radius=bore)
    assert fill.derivative_is_analytic
    edges = fill.breakpoints
    z = np.concatenate(
        [np.linspace(a, b, 5)[1:-1] for a, b in itertools.pairwise(edges)]
    )
    step = 1e-4 * np.min(np.diff(edges))
    wide = (fill(z + step) - fill(z - step)) / (2 * step)
    narrow = (fill(z + step / 2) - fill(z - step / 2)) / step
    richardson = (4 * narrow - wide) / 3
    analytic = fill.derivative(z)
    # Richardson leaves an h^4 truncation and rounding; 1e-7 as for every profile.
    assert np.max(np.abs(richardson - analytic)) / np.max(np.abs(analytic)) < 1e-7


def test_a_tapered_fill_breaks_at_the_union_of_stages_and_radius_kinks():
    fill = PressureGradient.from_stages(
        [0.0, 0.5, 1.0], [2.0, 6.0, 0.3], radius=PENNETTA_B
    )
    np.testing.assert_array_equal(
        fill.breakpoints, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    )
    # Kinks outside the fill are not the fill's.
    short = PressureGradient.from_stages([0.1, 0.3], [2.0, 1.0], radius=PENNETTA_B)
    np.testing.assert_array_equal(short.breakpoints, [0.1, 0.2, 0.3])


@pytest.mark.parametrize("radius", [125e-6, Constant(125e-6), Linear(1e-4, 1e-4, 2.0)])
def test_a_constant_radius_is_todays_fill(radius):
    # A constant radius is dropped (no radius), so the fill runs today's computation
    # on the same stages: an explicitly exact no-op, compared exactly.
    today = PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.0])
    bored = PressureGradient.from_stages(
        [0.0, 0.6, 1.5], [1.0, 5.0, 0.0], radius=radius
    )
    z = np.linspace(-0.2, 1.7, 211)
    assert bored.radius is None
    assert np.array_equal(bored(z), today(z))
    assert np.array_equal(bored.derivative(z), today.derivative(z))
    assert bored.breakpoints is bored.positions
    assert profile_identity(bored) == profile_identity(today)


def test_the_closed_form_and_chebyshev_routes_agree():
    closed = PressureGradient.from_stages(
        [0.0, 0.7, 1.0], [8.0, 1.0, 0.0], radius=PENNETTA_A
    )
    smooth = PressureGradient.from_stages(
        [0.0, 0.7, 1.0], [8.0, 1.0, 0.0], radius=_smooth(PENNETTA_A)
    )
    z = np.linspace(0.0, 1.0, 301)
    # Both integrate a^-4 to about 1e-13 relative (the closed form exactly, the
    # interpolant to its convergence tolerance), so the pressures agree to 1e-12.
    np.testing.assert_allclose(smooth(z), closed(z), rtol=1e-12)
    np.testing.assert_allclose(smooth.derivative(z), closed.derivative(z), rtol=1e-12)
    np.testing.assert_allclose(
        hydraulic_integral(_smooth(PENNETTA_A), [0.0, 0.7, 1.0]),
        hydraulic_integral(PENNETTA_A, [0.0, 0.7, 1.0]),
        rtol=1e-12,
    )


@settings(max_examples=100, deadline=None)
@given(
    radii=st.lists(st.floats(5e-6, 500e-6), min_size=2, max_size=6),
    pressures=st.lists(st.floats(0.0, 40.0), min_size=2, max_size=4),
    increasing=st.booleans(),
)
def test_a_tapered_fill_is_monotone_and_bounded_on_each_stage(
    radii, pressures, increasing
):
    radii = sorted(radii, reverse=not increasing)
    bore = PiecewiseLinear.from_stages(np.linspace(0.0, 1.0, len(radii)), radii)
    if max(pressures) < 1e-3:
        pressures = [*pressures[:-1], 1.0]
    fill = PressureGradient.from_stages(
        np.linspace(0.0, 1.0, len(pressures)), pressures, radius=bore
    )
    for (a, b), (pa, pb) in zip(
        itertools.pairwise(fill.positions),
        itertools.pairwise(fill.pressures),
        strict=True,
    ):
        values = fill(np.linspace(a, b, 33))
        assert np.all(np.isfinite(values))
        low, high = min(pa, pb), max(pa, pb)
        # The convex combination keeps p^2 between its ends to a few ulps.
        assert np.all(values >= low * (1 - 1e-14)) and np.all(
            values <= high * (1 + 1e-14)
        )
        steps = np.diff(values) * np.sign(pb - pa)
        assert np.all(steps >= -1e-14 * high)


def test_a_tapered_fill_reports_its_bore():
    fill = PressureGradient.two_point(1.0, 2.0, 0.5, radius=PENNETTA_A)
    assert "following the bore" in repr(fill)
    text = str(fill)
    assert "following the bore" in text and "bore" in text
    identity = profile_identity(fill)
    assert identity["radius"] == profile_identity(PENNETTA_A)
    with pytest.raises(TranslationError, match="no structural identity"):
        profile_identity(
            PressureGradient.two_point(1.0, 2.0, 0.5, radius=_smooth(PENNETTA_A))
        )


def test_a_radius_the_fill_cannot_follow_is_refused():
    with pytest.raises(TranslationError, match="finite and positive"):
        PressureGradient.two_point(1.0, 2.0, 0.5, radius=Linear(1e-4, -1e-4, 1.0))
    # An undeclared kink: |z - 0.5| is not analytic, so the interpolant cannot converge.
    kinked = FromCallable(lambda z: 1e-4 * (1 + np.abs(z - 0.4)), (0.0, 1.0))
    with pytest.raises(TranslationError, match="did not converge"):
        PressureGradient.two_point(1.0, 2.0, 0.5, radius=kinked)
    with pytest.raises(TranslationError, match="finite and positive"):
        hydraulic_integral(0.0, [0.0, 1.0])


def test_the_hydraulic_integral_of_a_uniform_bore_is_length_over_a4():
    # a^4 and the division round once each.
    np.testing.assert_allclose(
        hydraulic_integral(1e-4, [0.0, 0.25, 1.0]),
        np.array([0.25, 0.75]) / 1e-16,
        rtol=4 * np.finfo(float).eps,
    )
