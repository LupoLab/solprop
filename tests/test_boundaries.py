# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math
import warnings

import numpy as np
import pytest

from solprop.boundaries import (
    DEFAULT_BOUNDARY_N,
    MAX_ALPHA_LENGTH,
    MIN_COLLAR_ATTENUATION,
    MIN_COLLAR_SAMPLES,
    TEMPORAL_ALPHA_LENGTH,
    Absorber,
    SpatialAbsorber,
    absorber,
    apply_band_mask,
    apply_split_step,
    collar_rate,
    custom_absorber,
    nonlinear_absorber_term,
    split_step_factor,
    temporal_collar_rate,
    warn_walkoff,
)
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import GridError, SolpropWarning
from solprop.grid import Collar, EnvelopeGrid, FieldGrid, FreeGrid
from solprop.linear import Frame
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.output.schema import serialise
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController, RungeKutta
from solprop.translate.routes import transverse_of

EPS = np.finfo(np.float64).eps

# The RDW VUV target simulation M2 (docs/dev/accuracy.md section 5), and a
# supercontinuum-style envelope case; the same two the grid tests use.
RDW = {
    "length": 1.5,
    "reference_wavelength": 800e-9,
    "wavelength_limits": (90e-9, 4e-6),
    "trange": 400e-15,
}
SCG = {
    "length": 1.0,
    "reference_wavelength": 835e-9,
    "wavelength_limits": (400e-9, 2e-6),
    "trange": 20e-12,
}


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(**RDW)


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(**SCG, thg=True)


GRIDS = ["field_grid", "envelope_grid"]


# ------------------------------------------------------- the rate conversion


def test_the_rate_reproduces_the_collar_over_one_reference_length():
    # The definition: alpha = -2 log W / ell, so exp(-alpha ell / 2) == W. The factor
    # of two makes one reference length apply the profile to the field exactly once.
    window = np.linspace(1e-6, 1.0, 512)
    length = 0.075
    alpha = collar_rate(window, length)
    # Nothing here is near the cap, so the round trip is exact to the rounding of a
    # logarithm and an exponential: a few units in the last place.
    assert np.allclose(np.exp(-alpha * length / 2.0), window, rtol=8 * EPS)


def test_the_rate_is_exactly_zero_in_the_flat_region():
    alpha = collar_rate(np.array([1.0, 1.0, 0.5]), 0.1)
    # Exactly zero, and positive zero: a linear operator carrying -alpha/2 must be
    # untouched in band, and a negative zero would print and compare as one.
    assert alpha[0] == 0.0
    assert math.copysign(1.0, float(alpha[0])) == 1.0
    assert np.array_equal(alpha[:2], np.zeros(2))


def test_the_rate_is_capped_where_the_collar_reaches_zero():
    length = 0.075
    alpha = collar_rate(np.array([0.0, 1e-300, 1e-6]), length)
    ceiling = MAX_ALPHA_LENGTH / length
    assert alpha[0] == ceiling
    # 1e-300 would give alpha ell = 1382, which is finite but pointless; the cap is what
    # keeps exp(+alpha dz / 2) finite in the interaction picture as well.
    assert alpha[1] == ceiling
    assert alpha[2] < ceiling
    assert np.all(np.isfinite(alpha))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_a_bad_reference_length_is_rejected(bad):
    # Not merely useless but destructive: zero makes the rate 0/0 in the interior and
    # NaNs the grid, and a negative length inverts the clamp into a gain everywhere.
    with pytest.raises(GridError, match="reference_length must be positive"):
        collar_rate(np.array([1.0, 0.5]), bad)


@pytest.mark.parametrize("bad", [np.array([1.5, 0.5]), np.array([-0.1, 0.5])])
def test_a_profile_outside_the_unit_interval_is_rejected(bad):
    with pytest.raises(GridError, match=r"must lie in \[0, 1\]"):
        collar_rate(bad, 0.1)


# ------------------------------------------------------------- rate semantics


@pytest.mark.parametrize("n_steps", [1, 10, 100, 1000])
def test_the_absorption_depends_on_the_distance_and_not_on_the_subdivision(n_steps):
    # The property the whole scheme exists for. A fixed profile multiplied in once per
    # accepted step would give a total of W^n with n set by rtol.
    alpha = collar_rate(np.array([1.0, 0.9, 0.5, 1e-3]), 0.075)
    total = 0.3
    rng = np.random.default_rng(20260905)
    steps = rng.random(n_steps)
    steps *= total / steps.sum()
    product = np.ones_like(alpha)
    for step in steps:
        product = product * split_step_factor(alpha, float(step))
    once = split_step_factor(alpha, total)
    # n multiplications of doubles, each with half an ulp of rounding, give at worst
    # n * eps; at n = 1000 that is 2e-13.
    assert np.allclose(product, once, rtol=n_steps * EPS)


def test_the_split_factor_is_the_field_factor_not_the_power_factor():
    # alpha is a power coefficient, so the field carries alpha/2. Getting this wrong
    # would make the temporal absorber twice as strong as the spectral one built from
    # the same profile.
    alpha = collar_rate(np.array([0.5]), 0.075)
    factor = float(split_step_factor(alpha, 0.075)[0])
    assert factor == pytest.approx(0.5, rel=8 * EPS)
    assert factor**2 == pytest.approx(math.exp(-float(alpha[0]) * 0.075), rel=8 * EPS)


def test_a_zero_step_leaves_the_field_alone():
    alpha = collar_rate(np.array([1.0, 0.5]), 0.075)
    field = np.array([[1.0 + 2.0j, 3.0 - 1.0j]])
    # An explicitly exact no-op: a zero step multiplies by exp(0) = 1.
    assert np.array_equal(apply_split_step(field, alpha, 0.0), field)


@pytest.mark.parametrize("bad", [-1e-3, math.inf, math.nan])
def test_a_bad_step_is_rejected(bad):
    with pytest.raises(GridError, match="step must be non-negative"):
        split_step_factor(collar_rate(np.array([0.5]), 0.075), bad)


# ------------------------------------------- the two ADR-025 applications


def test_the_two_temporal_applications_agree_to_first_order_in_the_step():
    # (a) the split factor exp(-alpha h / 2) and (b) the term -alpha E / 2 inside the
    # nonlinear evaluation. One Euler step of (b) is the first two terms of (a), so the
    # difference is (alpha h / 2)^2 / 2 and falls by four when the step halves. If the
    # split factor were exp(-alpha h) instead, the two would differ at first order and
    # this would be flat.
    alpha = collar_rate(np.array([0.9, 0.5, 0.1]), 0.075)
    field = np.array([1.0, 1.0, 1.0])

    def deviation(step):
        exact = apply_split_step(field, alpha, step)
        euler = field + step * nonlinear_absorber_term(field, alpha)
        return float(np.max(np.abs(exact - euler)))

    coarse, fine = 1e-3, 5e-4
    assert deviation(coarse) / deviation(fine) == pytest.approx(4.0, rel=0.02)
    # And the coefficient is the one the expansion predicts: with x = alpha h / 2,
    # exp(-x) - (1 - x) = x^2/2 - x^3/6 + O(x^4), and x is 0.015 here, so the cubic term
    # is half a per cent and the quartic is 6e-5.
    x = float(alpha.max()) * fine / 2.0
    assert deviation(fine) == pytest.approx(0.5 * x**2 * (1.0 - x / 3.0), rel=1e-4)


def test_the_in_term_is_the_derivative_of_the_split_factor():
    alpha = collar_rate(np.array([0.9, 0.5]), 0.075)
    field = np.array([2.0 + 1.0j, -1.0 + 0.5j])
    step = 1e-9
    numerical = (apply_split_step(field, alpha, step) - field) / step
    assert np.allclose(numerical, nonlinear_absorber_term(field, alpha), rtol=1e-6)


def test_both_applications_broadcast_over_the_leading_axes():
    alpha = collar_rate(np.linspace(0.2, 1.0, 8), 0.075)
    field = np.ones((2, 3, 8), dtype=complex)
    assert apply_split_step(field, alpha, 0.01).shape == (2, 3, 8)
    assert nonlinear_absorber_term(field, alpha).shape == (2, 3, 8)
    assert np.allclose(
        apply_split_step(field, alpha, 0.01)[0, 0], split_step_factor(alpha, 0.01)
    )


# ---------------------------------------------------------------- the hard mask


def test_the_hard_band_mask_is_idempotent():
    # Why it stays a per-step 0/1 multiply rather than becoming a rate: applying it any
    # number of times gives the same answer, exactly (a hard mask, a structural zero).
    rng = np.random.default_rng(11)
    spectrum = rng.normal(size=(2, 16)) + 1j * rng.normal(size=(2, 16))
    mask = rng.random(16) > 0.4
    once = apply_band_mask(spectrum, mask)
    assert np.array_equal(apply_band_mask(once, mask), once)
    assert np.array_equal(once[:, ~mask], np.zeros((2, int(np.sum(~mask)))))
    assert np.array_equal(once[:, mask], spectrum[:, mask])


# ------------------------------------------------------------- from a grid


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_absorber_reproduces_the_collars_over_one_reference_length(
    grid_name, request
):
    grid = request.getfixturevalue(grid_name)
    boundary = absorber(grid)
    length = boundary.reference_length
    rate, window = boundary.spectral_rate, grid.omega_window
    # Only where the cap does not bite, and only inside the active band, where the rate
    # is deliberately zeroed rather than capped. The temporal rates are not the
    # logarithm; the tests below say what they are.
    unclamped = (window > math.exp(-MAX_ALPHA_LENGTH / 2.0)) & (rate > 0.0)
    assert np.allclose(
        np.exp(-rate[unclamped] * length / 2.0), window[unclamped], rtol=8 * EPS
    )


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_temporal_rate_is_the_bounded_complement_of_the_collar(grid_name, request):
    """The temporal collar is ``A (1 - W) / ell``, not the capped logarithm."""
    grid = request.getfixturevalue(grid_name)
    boundary = absorber(grid)
    length = boundary.reference_length
    # Each side forms A (1 - W) / ell from a window W in [0, 1], evaluated from the
    # same collar: W carries a few ulps of one (a library exponential), and the
    # subtraction, product and quotient three roundings more, so the two agree to
    # 16 eps of the peak rate A / ell in every sample (a rounding bound, not a fit).
    tolerance = 16 * EPS * TEMPORAL_ALPHA_LENGTH / length
    for rate, window, axis in (
        (boundary.temporal_rate, grid.t_window, grid.t),
        (boundary.temporal_rate_fine, grid.t_window_fine, grid.t_fine),
    ):
        np.testing.assert_allclose(
            rate,
            TEMPORAL_ALPHA_LENGTH * (1.0 - window) / length,
            rtol=0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            rate,
            temporal_collar_rate(axis, grid.t_collar, length),
            rtol=0,
            atol=tolerance,
        )
        # Exactly zero over the flat region, and positive zero.
        flat = window == 1.0
        assert np.array_equal(rate[flat], np.zeros(int(flat.sum())))
        assert math.copysign(1.0, float(rate[flat][0])) == 1.0
        # Bounded at the edge of the window, well inside the validation ceiling.
        assert float(np.max(rate)) == pytest.approx(TEMPORAL_ALPHA_LENGTH / length)
        assert float(np.max(rate)) < MAX_ALPHA_LENGTH / length / 4
        # Monotone on each edge, rising towards the ends of the window.
        assert np.all(np.diff(rate[axis < grid.t_collar.one_low]) <= 0.0)
        assert np.all(np.diff(rate[axis > grid.t_collar.one_high]) >= 0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_temporal_rate_keeps_the_integrated_attenuation_of_the_logarithm(
    grid_name, request
):
    """The calibration of ``TEMPORAL_ALPHA_LENGTH``: the walk-off protection stands.

    The capped logarithm's integral over the collar was 6.3 per collar width per
    reference length; the profile matches it to a few per cent, so every walk-off
    figure and the meaning of ``boundary_N`` are unchanged.
    """
    grid = request.getfixturevalue(grid_name)
    boundary = absorber(grid)
    logarithm = collar_rate(grid.t_window, boundary.reference_length)
    old = float(np.trapezoid(logarithm, dx=grid.dt))
    new = float(np.trapezoid(boundary.temporal_rate, dx=grid.dt))
    assert new == pytest.approx(old, rel=0.05)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_absorber_is_zero_in_band_and_outside_the_active_mask(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    boundary = absorber(grid)
    # In band there is nothing to absorb ...
    assert np.array_equal(
        boundary.spectral_rate[grid.inband_mask],
        np.zeros(int(np.sum(grid.inband_mask))),
    )
    # ... and outside the active mask the linear operator is zero and the nonlinear
    # polarisation is masked, so a rate there would only amplify the band edge in the
    # interaction picture. conventions.md section 3 puts the hard limit here.
    assert np.array_equal(
        boundary.spectral_rate[~grid.active_mask],
        np.zeros(int(np.sum(~grid.active_mask))),
    )
    assert np.array_equal(boundary.band_mask, grid.active_mask)
    assert np.any(boundary.spectral_rate > 0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_reference_length_follows_the_strength_or_the_explicit_length(
    grid_name, request
):
    grid = request.getfixturevalue(grid_name)
    assert absorber(grid).reference_length == grid.length / DEFAULT_BOUNDARY_N
    assert absorber(grid, boundary_N=5).reference_length == grid.length / 5
    assert absorber(grid, boundary_length=0.02).reference_length == 0.02
    # An explicit length wins over the strength, as in Luna PR #438.
    assert absorber(grid, boundary_N=5, boundary_length=0.02).reference_length == 0.02
    # A stronger absorber is a shorter reference length and hence a larger rate.
    strong = absorber(grid, boundary_N=40).temporal_rate
    weak = absorber(grid, boundary_N=10).temporal_rate
    assert np.all(strong[weak > 0] > weak[weak > 0])


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_temporal_rates_agree_where_the_two_time_axes_meet(grid_name, request):
    # Both windows come from the one collar geometry, so the coarse and fine rates are
    # the same number wherever their samples coincide. Luna builds each from its own
    # axis's extremes, which differ at the upper edge.
    grid = request.getfixturevalue(grid_name)
    boundary = absorber(grid)
    stride = grid.n_to // grid.n_t
    # The same expression at the same times, each with a few ulps of the peak rate
    # (the bound of the test above).
    np.testing.assert_allclose(
        boundary.temporal_rate_fine[::stride],
        boundary.temporal_rate,
        rtol=0,
        atol=16 * EPS * TEMPORAL_ALPHA_LENGTH / boundary.reference_length,
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_a_bad_strength_or_length_is_rejected(bad, field_grid):
    with pytest.raises(GridError, match="boundary_N must be positive"):
        absorber(field_grid, boundary_N=bad)
    with pytest.raises(GridError, match="boundary_length must be positive"):
        absorber(field_grid, boundary_length=bad)


def test_the_absorber_is_summarised_at_debug_level(caplog, field_grid):
    with caplog.at_level("DEBUG", logger="solprop.boundaries"):
        absorber(field_grid)
    assert "reference length" in caplog.text
    assert "temporal collar" in caplog.text


# ------------------------------------------------------- degenerate collars


def test_a_collar_with_no_width_is_rejected(field_grid):
    # A temporal window built naively is exactly this for a trange that lands on a
    # power of two: identically one except for a single sample, so such a grid has no
    # temporal absorber at all.
    half = 0.5 * float(np.max(field_grid.t) - np.min(field_grid.t))
    flat = half * (1.0 - 1e-15)
    degenerate = dataclasses.replace(
        field_grid, t_collar=Collar(-half, -flat, flat, half)
    )
    with pytest.raises(GridError, match="zero width"):
        absorber(degenerate)


def test_a_collar_of_a_few_samples_is_rejected(field_grid):
    # Wide enough not to be a step in exact arithmetic, but resolved by two samples, so
    # the discrete rate jumps to its cap almost at once.
    half = 0.5 * float(np.max(field_grid.t) - np.min(field_grid.t))
    flat = half - 2.5 * field_grid.dt
    narrow = dataclasses.replace(field_grid, t_collar=Collar(-half, -flat, flat, half))
    with pytest.raises(GridError, match=f"fewer than the {MIN_COLLAR_SAMPLES}"):
        absorber(narrow)


def test_a_degenerate_spectral_collar_is_rejected(field_grid):
    edge = field_grid.omega_collar
    degenerate = dataclasses.replace(
        field_grid,
        omega_collar=Collar(
            edge.one_low - 1e-30, edge.one_low, edge.one_high, edge.zero_high
        ),
    )
    with pytest.raises(GridError, match="spectral collar"):
        absorber(degenerate)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_grids_solprop_builds_are_never_degenerate(grid_name, request):
    # The guard above never fires on a grid from from_limits, because it
    # guarantees a minimum collar width; this is the statement that it does.
    grid = request.getfixturevalue(grid_name)
    assert grid.t_collar.width_low > MIN_COLLAR_SAMPLES * grid.dt
    assert grid.omega_collar.width_low > MIN_COLLAR_SAMPLES * grid.d_omega
    assert absorber(grid).reference_length > 0.0


# ------------------------------------------------------------ the Absorber object


def build_fields(grid):
    """The keyword arguments of a valid absorber for a grid."""
    boundary = absorber(grid)
    return {
        "spectral_rate": boundary.spectral_rate,
        "temporal_rate": boundary.temporal_rate,
        "temporal_rate_fine": boundary.temporal_rate_fine,
        "band_mask": boundary.band_mask,
        "reference_length": boundary.reference_length,
    }


def test_a_malformed_absorber_is_rejected(field_grid):
    fields = build_fields(field_grid)
    with pytest.raises(GridError, match="reference_length must be positive"):
        Absorber(**{**fields, "reference_length": 0.0})
    with pytest.raises(GridError, match="band_mask has shape"):
        Absorber(**{**fields, "band_mask": fields["band_mask"][:-1]})
    negative = fields["temporal_rate"].copy()
    negative[0] = -1.0
    with pytest.raises(GridError, match="temporal_rate must lie"):
        Absorber(**{**fields, "temporal_rate": negative})
    huge = fields["spectral_rate"].copy()
    huge[0] = 1e30
    with pytest.raises(GridError, match="spectral_rate must lie"):
        Absorber(**{**fields, "spectral_rate": huge})
    infinite = fields["temporal_rate_fine"].copy()
    infinite[0] = math.inf
    with pytest.raises(GridError, match="temporal_rate_fine must be finite"):
        Absorber(**{**fields, "temporal_rate_fine": infinite})


# --------------------------------------------------------------- walk-off


def crossing_nepers(grid, group_delay_span, boundary_n=DEFAULT_BOUNDARY_N):
    """The field attenuation a walking packet accumulates crossing the collar."""
    rate = absorber(grid, boundary_N=boundary_n).temporal_rate
    return 0.5 * float(np.trapezoid(rate, dx=grid.dt)) * grid.length / group_delay_span


def test_the_walk_off_warning_fires_when_the_absorber_is_too_weak(field_grid):
    # A packet whose group delay drifts by more than the time window itself crosses the
    # collar too fast to be absorbed, reaches the edge alive, and wraps around.
    span = 2e-12
    assert crossing_nepers(field_grid, span) < MIN_COLLAR_ATTENUATION
    with pytest.warns(SolpropWarning, match="wrapping around"):
        warn_walkoff(field_grid, span)


def test_the_walk_off_warning_is_silent_when_the_absorber_is_strong_enough(field_grid):
    span = 100e-15
    assert crossing_nepers(field_grid, span) > MIN_COLLAR_ATTENUATION
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(field_grid, span)
        # Nothing walks anywhere, so there is nothing to check.
        warn_walkoff(field_grid, 0.0)
        warn_walkoff(field_grid, -1e-12)


def test_a_stronger_absorber_silences_the_warning(field_grid):
    span = 2e-12
    with pytest.warns(SolpropWarning):
        warn_walkoff(field_grid, span, boundary_N=DEFAULT_BOUNDARY_N)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(field_grid, span, boundary_N=200)
        warn_walkoff(field_grid, span, boundary_length=field_grid.length / 200)


def test_the_warning_says_how_much_stronger_the_absorber_must_be(field_grid):
    span = 2e-12
    with pytest.warns(SolpropWarning) as record:
        warn_walkoff(field_grid, span)
    message = str(record[0].message)
    assert "widen trange" in message
    assert "boundary_N above" in message
    suggested = float(message.rsplit("above", 1)[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(field_grid, span, boundary_N=suggested * 1.01)


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_a_non_finite_walk_off_span_is_rejected(bad, field_grid):
    with pytest.raises(GridError, match="group_delay_span must be finite"):
        warn_walkoff(field_grid, bad)


def test_the_walk_off_check_validates_its_reference_length(field_grid):
    with pytest.raises(GridError, match="boundary_N must be positive"):
        warn_walkoff(field_grid, 1e-13, boundary_N=0.0)


def test_the_suggested_boundary_n_holds_for_a_collar_of_its_own_length(field_grid):
    """The suggestion counts from the length in force, not from the default strength."""
    span = 2e-12
    length = field_grid.length / 3.0
    with pytest.warns(SolpropWarning) as record:
        warn_walkoff(field_grid, span, boundary_length=length)
    message = str(record[0].message)
    suggested = float(message.rsplit("above", 1)[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(field_grid, span, boundary_N=suggested * 1.01)
    with pytest.warns(SolpropWarning):
        warn_walkoff(field_grid, span, boundary_N=suggested * 0.99)


@pytest.mark.parametrize("span", [1.001e-12, 1.2345e-12, 2e-12, 9.994e-12])
def test_the_suggested_walkoff_holds_as_printed(field_grid, span):
    """The three digits shown never fall below the span, so the text can be taken up."""
    with pytest.warns(SolpropWarning) as record:
        warn_walkoff(field_grid, span, boundary_length=field_grid.length / 3.0)
    printed = float(str(record[0].message).split("walkoff=")[1].split(")")[0])
    assert span <= printed <= 1.01 * span
    length = absorber(field_grid, walkoff=printed).in_stage_length
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(field_grid, span, boundary_length=length)


# --------------------------------------------------- a collar sized from a walk-off


def edge_width(grid):
    """The width of one edge of a grid's symmetric temporal collar, s."""
    collar = grid.t_collar
    assert collar.width_low == pytest.approx(collar.width_high)
    return collar.width_low


def test_a_walkoff_sets_the_length_the_derivation_gives(field_grid):
    """``ell_c = 6.3 w L / (Gamma dtau)``, on the integral ``warn_walkoff`` reads.

    The integral of ``1 - W`` over both edges is ``w`` by the Planck taper's symmetry;
    the trapezoid on the axis, which stops one sample short of closing the period,
    reads ``w - dt``. So the bound is that one sample, and the sampled form is exact.
    """
    drift = 3e-12
    width = edge_width(field_grid)
    collar = absorber(field_grid, walkoff=drift)
    closed_form = 6.3 * width * field_grid.length / (MIN_COLLAR_ATTENUATION * drift)
    assert collar.in_stage_length == pytest.approx(
        closed_form, rel=2.0 * field_grid.dt / width
    )
    assert collar.in_stage_length == pytest.approx(
        closed_form * (width - field_grid.dt) / width, rel=1e-6
    )
    assert float(collar.temporal_rate.max()) == pytest.approx(
        TEMPORAL_ALPHA_LENGTH / collar.in_stage_length, rel=1e-3
    )


def test_a_walkoff_of_eighteen_collar_widths_is_the_default_strength(field_grid):
    """SEP 21's figure: ``AN = 252`` protects ``252 w / (2 Gamma) = 18.2 w``."""
    width = edge_width(field_grid)
    drift = 252.0 * width / (2.0 * MIN_COLLAR_ATTENUATION)
    assert drift / width == pytest.approx(18.2, abs=0.05)
    collar = absorber(field_grid, walkoff=drift)
    assert collar.in_stage_length == pytest.approx(
        field_grid.length / DEFAULT_BOUNDARY_N, rel=2.0 * field_grid.dt / width
    )


@pytest.mark.parametrize(
    "strength",
    [{}, {"boundary_N": 60.0}, {"boundary_length": 0.01}],
    ids=["default", "boundary_N", "boundary_length"],
)
def test_a_walkoff_moves_only_the_temporal_rates(field_grid, strength):
    plain = absorber(field_grid, **strength)
    collar = absorber(field_grid, walkoff=1e-12, **strength)
    assert collar.reference_length == plain.reference_length
    assert collar.collar_length is not None
    assert np.array_equal(collar.spectral_rate, plain.spectral_rate)
    assert np.array_equal(collar.band_mask, plain.band_mask)
    assert not np.array_equal(collar.temporal_rate, plain.temporal_rate)
    # The temporal collar is the one the walk-off asks for, whatever set the spectral.
    assert np.array_equal(
        collar.temporal_rate,
        temporal_collar_rate(field_grid.t, field_grid.t_collar, collar.in_stage_length),
    )
    assert np.array_equal(
        collar.temporal_rate_fine,
        temporal_collar_rate(
            field_grid.t_fine, field_grid.t_collar, collar.in_stage_length
        ),
    )
    assert collar.in_stage_length == absorber(field_grid, walkoff=1e-12).in_stage_length


def test_no_walkoff_leaves_the_absorber_as_it_was(field_grid):
    """The default run's rates, record and so its configuration hash do not move."""
    collar = absorber(field_grid)
    assert collar.collar_length is None
    length = collar.reference_length
    assert np.array_equal(
        collar.temporal_rate,
        temporal_collar_rate(field_grid.t, field_grid.t_collar, length),
    )
    assert serialise(collar) == serialise(absorber(field_grid, walkoff=None))


def test_the_configuration_hash_follows_the_walkoff_and_nothing_else():
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1200e-9), 2e-12, thg=False)

    def configuration_hash(collar):
        built = walking_settings(1.0, absorber=collar, prefactor_index=1.0)
        simulation = Simulation(
            grid,
            SpecifiedMode.from_taylor([7e6, 4.8e-9], omega_ref=grid.omega0),
            [],
            GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-9),
            settings=built,
        )
        return simulation.metadata().configuration_hash()

    plain = absorber(grid)
    # Built field by field as an absorber was before the keyword existed.
    before = Absorber(
        spectral_rate=plain.spectral_rate,
        temporal_rate=temporal_collar_rate(
            grid.t, grid.t_collar, plain.reference_length
        ),
        temporal_rate_fine=temporal_collar_rate(
            grid.t_fine, grid.t_collar, plain.reference_length
        ),
        band_mask=plain.band_mask,
        reference_length=plain.reference_length,
    )
    assert configuration_hash(plain) == configuration_hash(before)
    assert configuration_hash(plain) != configuration_hash(
        absorber(grid, walkoff=1e-12)
    )


@pytest.mark.parametrize("fields", [RDW, SCG], ids=["rdw", "scg"])
@pytest.mark.parametrize("drift", [1e-15, 1e-13, 3e-13, 1e-12, 7e-12, 1e-10])
def test_a_collar_sized_for_a_drift_passes_the_walk_off_check_at_that_drift(
    fields, drift
):
    """The keyword and the warning are one condition; a little more drift fails it."""
    grid = FieldGrid.from_limits(**fields)
    length = absorber(grid, walkoff=drift).in_stage_length
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(grid, drift, boundary_length=length)
    with pytest.raises(SolpropWarning, match="walkoff="), warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_walkoff(grid, 1.01 * drift, boundary_length=length)


def test_a_slice_given_its_own_drift_keeps_its_parent_s_temporal_collar():
    """The rate depends on the speed, ``walkoff / length``, and not on the length."""

    def built(length):
        return FieldGrid.from_limits(length, 800e-9, (400e-9, 2e-6), 400e-15)

    drift = 1e-12
    parent = absorber(built(1.0), walkoff=drift)
    piece = absorber(
        built(0.25),
        walkoff=0.25 * drift,
        boundary_length=parent.reference_length,
    )
    # A few operations on the same numbers apart, so a few ulps of the rate.
    np.testing.assert_allclose(
        piece.temporal_rate, parent.temporal_rate, rtol=8 * EPS, atol=0.0
    )
    np.testing.assert_allclose(
        piece.spectral_rate, parent.spectral_rate, rtol=8 * EPS, atol=0.0
    )
    assert piece.reference_length == parent.reference_length
    # Its parent's whole drift over a quarter of the length is four times the speed.
    fast = absorber(built(0.25), walkoff=drift)
    assert fast.in_stage_length == pytest.approx(parent.in_stage_length / 4.0)


def test_a_custom_rate_keeps_the_walkoff_s_strength(field_grid):
    collar = absorber(field_grid, walkoff=1e-12)
    summed = custom_absorber(field_grid, 1e5, collar=collar)
    assert summed.in_stage_length == collar.in_stage_length
    assert np.array_equal(summed.temporal_rate, collar.temporal_rate)


@pytest.mark.parametrize("bad", [0.0, -1e-12, math.inf, -math.inf, math.nan])
def test_a_walkoff_must_be_positive_and_finite(field_grid, bad):
    with pytest.raises(GridError, match="walkoff must be positive and finite"):
        absorber(field_grid, walkoff=bad)


@pytest.mark.parametrize("extreme", [5e-324, 1e308])
def test_a_walkoff_whose_length_is_not_representable_is_refused(field_grid, extreme):
    with pytest.raises(GridError, match="asks for a collar length"):
        absorber(field_grid, walkoff=extreme)


def test_a_walkoff_is_refused_beside_a_spectral_only_collar(field_grid):
    with pytest.raises(GridError, match="temporal=False has none"):
        absorber(field_grid, walkoff=1e-12, temporal=False)


def test_a_walkoff_is_not_bounded_by_the_window(field_grid):
    """A drift of ten windows: the packet is absorbed on its first crossing."""
    window = field_grid.n_t * field_grid.dt
    collar = absorber(field_grid, walkoff=10.0 * window)
    assert collar.in_stage_length < absorber(field_grid).in_stage_length
    assert np.all(np.isfinite(collar.temporal_rate))


def test_a_walkoff_needs_a_collar_to_size(field_grid):
    half = field_grid.t_collar.zero_high
    bare = dataclasses.replace(field_grid, t_collar=Collar(-half, -half, half, half))
    with pytest.raises(GridError, match="zero width"):
        absorber(bare, walkoff=1e-12)


def test_a_walkoff_leaves_a_free_space_transverse_collar_alone():
    """SEP 21 feared this field also set the spatial strength; SEP 40 separated them."""
    grid = EnvelopeGrid.from_limits(2e-4, 800e-9, (650e-9, 1000e-9), 600e-15)
    space = FreeGrid(1e-4, 40, 1e-4, 40, collar_fraction=1.0)

    def design(collar):
        simulation = Simulation.from_free(
            grid,
            space,
            SpatialMedium(),
            [],
            SpatialPulse(
                GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-9), 20e-6
            ),
            settings=walking_settings(2e-4, absorber=collar),
            noise=None,
            spatial_absorber=SpatialAbsorber(kappa=2.5e5),
        )
        return transverse_of(simulation.translate().route)

    plain, walking = design(absorber(grid)), design(absorber(grid, walkoff=1e-14))
    assert plain is not None
    assert walking is not None
    assert plain.collar_rate is not None
    assert walking.collar_rate is not None
    assert plain.has_collar
    assert plain.alpha0 == walking.alpha0
    assert np.array_equal(plain.collar_rate, walking.collar_rate)
    assert np.array_equal(plain.ramp_rate, walking.ramp_rate)


def walking_settings(length, *, backend="numpy_ref", **extra):
    """Settings for a linear run of ``length`` with a tight controller."""
    return SimulationSettings(
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=[length],
        dz_max=length / 20.0,
        dz_init=length / 100.0,
        integrator=RungeKutta(),
        backend=backend,
        boundary_check=None,
        **extra,
    )


def transmitted_power(grid, collar, drift, *, backend="numpy_ref", scheme=None):
    """The share of a packet's energy left after it drifts by ``drift`` over the grid.

    A mode whose only dispersion is ``beta_1``, in a frame that does not follow it, is a
    rigid translation at ``drift / length``: nothing else in the run moves the energy.
    """
    beta0, beta1 = 7e6, 4.8e-9
    mode = SpecifiedMode.from_taylor([beta0, beta1], omega_ref=grid.omega0)
    frame = Frame(
        grid.omega0,
        subtract_beta0=True,
        beta0=beta0,
        beta1=beta1 - drift / grid.length,
    )
    settings = walking_settings(
        grid.length,
        backend=backend,
        absorber=collar,
        absorber_scheme=scheme,
        prefactor_index=1.0,
    )
    pulse = GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-9)
    output = Simulation(grid, mode, [], pulse, settings=settings, frame=frame).run()
    field = np.asarray(output["field"])
    return float(np.sum(np.abs(field[-1]) ** 2) / np.sum(np.abs(field[0]) ** 2))


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
@pytest.mark.parametrize("stretch", [1.0, 0.5, 2.0])
def test_a_walkoff_collar_removes_light_of_the_stated_walk_off(backend, stretch):
    """A packet of the stated drift leaves a millionth of its power, and no more.

    The packet crosses both edges rigidly, so its power falls by exactly the integral
    of the rate over the ring divided by the speed: ``exp(-2 Gamma)`` of the sizing,
    corrected by the one sample the sizing's trapezoid leaves out of the periodic sum.
    A drift of half the collar's or twice its stated value shows the exponent is
    linear in the speed, and that the light has no other way out. The bound on the
    exponent is the controller's ``rtol``; measured 6e-9 on both backends.
    """
    if backend == "native":
        pytest.importorskip("solprop_native")
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1200e-9), 2e-12, thg=False)
    window = grid.n_t * grid.dt
    # Long enough to clear the far edge entirely, and short of the launch's own image.
    drift = 0.8 * window
    collar = absorber(grid, walkoff=drift / stretch)
    closed_period = np.sum(temporal_collar_rate(grid.t, grid.t_collar, 1.0)) * grid.dt
    expected = closed_period * grid.length / (collar.in_stage_length * drift)
    assert expected == pytest.approx(2.0 * MIN_COLLAR_ATTENUATION / stretch, rel=0.02)
    power = transmitted_power(grid, collar, drift, backend=backend)
    assert -math.log(power) == pytest.approx(expected, rel=1e-7)


def test_without_a_temporal_collar_the_same_packet_loses_nothing():
    """So what the walk-off collar removes above is the collar's."""
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1200e-9), 2e-12, thg=False)
    collar = absorber(grid, temporal=False)
    power = transmitted_power(grid, collar, 0.8 * grid.n_t * grid.dt)
    assert power == pytest.approx(1.0, abs=1e-6)


def test_the_split_scheme_removes_the_same_light():
    """The split's factors telescope, within its crossing limit, to the same exponent.

    The split's first-order error in the step and the sizing's left-out sample, 0.7 %
    here, set the bound; measured 0.5 %.
    """
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1200e-9), 2e-12, thg=False)
    drift = 0.8 * grid.n_t * grid.dt
    power = transmitted_power(
        grid, absorber(grid, walkoff=drift), drift, scheme="split_step"
    )
    assert -math.log(power) == pytest.approx(2.0 * MIN_COLLAR_ATTENUATION, rel=0.02)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_the_temporal_profile_rejects_a_bad_reference_length(bad):
    with pytest.raises(GridError, match="reference_length must be positive"):
        temporal_collar_rate(np.array([0.0]), Collar(-2.0, -1.0, 1.0, 2.0), bad)
