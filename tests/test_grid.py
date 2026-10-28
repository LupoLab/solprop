# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math
import re

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from solprop import fourier
from solprop.constants import C
from solprop.exceptions import GridError
from solprop.grid import (
    DEFAULT_COLLAR_FRACTION,
    Collar,
    EnvelopeGrid,
    FieldGrid,
    TimeGrid,
    _fft_size,
    _frequency_limits,
    _needed_window,
    fft_cost,
    planck_taper,
)

EPS = np.finfo(np.float64).eps

# The RDW VUV target simulation M2 (docs/dev/accuracy.md section 5): He gradient
# capillary, 7.5 fs at 800 nm, 90 nm to 4 um, 400 fs window. Luna builds n_t = 4096,
# n_to = 8192 for it.
RDW = {
    "length": 1.5,
    "reference_wavelength": 800e-9,
    "wavelength_limits": (90e-9, 4e-6),
    "trange": 400e-15,
}
# A supercontinuum-style envelope case: 835 nm carrier, 400 nm to 2 um, 20 ps window.
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


def is_smooth(n):
    """Whether n factorises over 2, 3 and 5 alone."""
    for prime in (2, 3, 5):
        while n % prime == 0:
            n //= prime
    return n == 1


# --- the Planck taper -----------------------------------------------------------------


def test_planck_taper_reaches_its_limits():
    x = np.linspace(-2.0, 2.0, 401)
    window = planck_taper(x, -1.0, -0.5, 0.5, 1.0)
    assert np.all(window[x <= -1.0] == 0.0)
    assert np.all(window[x >= 1.0] == 0.0)
    assert np.all(window[(x >= -0.5) & (x <= 0.5)] == 1.0)
    assert np.all((window >= 0.0) & (window <= 1.0))


def test_planck_taper_is_monotonic_on_each_edge():
    rising = planck_taper(np.linspace(-1.0, -0.5, 200), -1.0, -0.5, 0.5, 1.0)
    assert np.all(np.diff(rising) >= 0.0)
    falling = planck_taper(np.linspace(0.5, 1.0, 200), -1.0, -0.5, 0.5, 1.0)
    assert np.all(np.diff(falling) <= 0.0)


def test_planck_taper_is_symmetric_for_symmetric_breakpoints():
    x = np.linspace(-1.0, 1.0, 257)
    window = planck_taper(x, -1.0, -0.4, 0.4, 1.0)
    assert_allclose(window, window[::-1], rtol=0.0, atol=8 * EPS)


def test_planck_taper_is_half_at_the_midpoint_of_an_edge():
    # The Planck taper is 1/2 where its exponent vanishes, at the centre of the edge.
    assert_allclose(planck_taper(np.array([-0.75]), -1.0, -0.5, 0.5, 1.0), [0.5])


@pytest.mark.parametrize(
    "points",
    [
        (1.0, 1.0, 2.0, 3.0),  # zero-width rising edge
        (0.0, 1.0, 2.0, 2.0),  # zero-width falling edge
        (0.0, 2.0, 1.0, 3.0),  # flat region inverted
    ],
)
def test_planck_taper_rejects_degenerate_breakpoints(points):
    with pytest.raises(GridError, match="breakpoints must satisfy"):
        planck_taper(np.array([0.5]), *points)


def test_collar_reports_its_edge_widths():
    collar = Collar(0.0, 1.0, 4.0, 6.0)
    assert collar.width_low == 1.0
    assert collar.width_high == 2.0
    assert_allclose(collar.window(np.array([0.5, 2.0, 7.0])), [0.5, 1.0, 0.0])


# --- construction: the sizes the grids pin --------------------------------------------


def test_field_grid_reproduces_the_rdw_vuv_sizes():
    # Luna's geometry, as the frozen witnesses state it: powers of two, collar inside.
    grid = FieldGrid.from_limits(
        **RDW, fft_size="pow2", collar_fraction=0.05, collar_placement="inside"
    )
    assert (grid.n_t, grid.n_to) == (4096, 8192)
    assert (grid.n_omega, grid.n_omega_fine) == (2049, 4097)


def test_the_default_collar_lies_outside_the_request(field_grid):
    # SEP 21: the whole requested window is flat, and the collar takes a tenth of the
    # realised window beyond it at each end.
    assert_allclose(field_grid.t_collar.one_high, RDW["trange"] / 2, rtol=8 * EPS)
    assert_allclose(field_grid.t_collar.one_low, -RDW["trange"] / 2, rtol=8 * EPS)
    window = field_grid.n_to * field_grid.dt_fine
    assert window >= RDW["trange"] / (1 - 2 * DEFAULT_COLLAR_FRACTION)
    assert (field_grid.n_t, field_grid.n_to) == (5120, 10240)


def test_a_collar_placement_must_be_outside_or_inside():
    # The runtime type check refuses it at from_limits; the sizing helper refuses it
    # for any caller without one.
    assert _needed_window(1.0, 0.1, "inside") == 1.0
    with pytest.raises(GridError, match="collar_placement must be"):
        _needed_window(1.0, 0.1, "middle")


def test_a_grid_without_room_for_a_taper_takes_the_spectral_collar_alone(field_grid):
    # A collar of one sample each side cannot taper: the default absorber falls back
    # to its spectral collar rather than refuse the run.
    from solprop.driver.simulation import default_collar  # noqa: PLC0415

    edges = field_grid.t_collar
    narrow = dataclasses.replace(
        field_grid,
        t_collar=dataclasses.replace(
            edges,
            zero_low=edges.one_low - field_grid.dt,
            zero_high=edges.one_high + field_grid.dt,
        ),
    )
    assert default_collar(field_grid).temporal
    assert not default_collar(narrow).temporal


def test_auto_sizes_cost_no_more_than_the_power_of_two():
    for trange in (137e-15, 400e-15, 550e-15, 923e-15):
        auto = FieldGrid.from_limits(**{**RDW, "trange": trange})
        pow2 = FieldGrid.from_limits(**{**RDW, "trange": trange}, fft_size="pow2")
        assert fft_cost(auto.n_to) + fft_cost(auto.n_t) <= fft_cost(
            pow2.n_to
        ) + fft_cost(pow2.n_t)
        assert is_smooth(auto.n_to)
        assert auto.n_to % auto.n_t == 0


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_coarse_and_fine_grids_satisfy_the_invariants(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    ratio = grid.n_to // grid.n_t
    assert grid.n_to % grid.n_t == 0
    # A structural contract of the grid: dt is built from the integer ratio, so this
    # holds exactly rather than to round-off.
    assert grid.dt / grid.dt_fine == ratio
    # The coarse time samples are the fine ones taken every ratio-th, exactly: both
    # axes are correctly rounded products of the same exactly represented integers.
    assert np.array_equal(grid.t, grid.t_fine[::ratio])
    assert grid.t[grid.n_t // 2] == 0.0
    assert grid.t_fine[grid.n_to // 2] == 0.0
    grid.validate()


def test_field_grid_frequency_axis_is_the_rfft_axis(field_grid):
    assert field_grid.n_t == 2 * (field_grid.n_omega - 1)
    assert field_grid.n_to == 2 * (field_grid.n_omega_fine - 1)
    assert_allclose(
        field_grid.omega,
        np.arange(field_grid.n_omega) * field_grid.d_omega,
        rtol=8 * EPS,
    )
    assert_allclose(field_grid.dt, np.pi / field_grid.omega[-1], rtol=8 * EPS)
    # The ratio of the two Nyquist frequencies is the oversampling factor.
    assert_allclose(
        field_grid.omega_fine[-1] / field_grid.omega[-1],
        field_grid.n_to / field_grid.n_t,
        rtol=8 * EPS,
    )


def test_envelope_grid_frequency_axis_is_in_fft_order(envelope_grid):
    grid = envelope_grid
    assert grid.n_omega == grid.n_t
    assert grid.n_omega_fine == grid.n_to
    relative = grid.omega - grid.omega0
    # FFT order: the leading half rises from zero, the trailing half is negative.
    assert relative[0] == 0.0
    assert np.all(np.diff(relative[: grid.n_t // 2]) > 0.0)
    assert np.all(relative[grid.n_t // 2 :] < 0.0)
    assert_allclose(grid.dt, 2 * np.pi / (grid.n_omega * grid.d_omega), rtol=8 * EPS)
    # In FFT order the informative ratio is that of the minima, not of the last bins:
    # the last bin is one step below the carrier on both grids (Luna Grid.jl:245).
    assert_allclose(
        (grid.omega_fine - grid.omega0).min() / relative.min(),
        grid.n_to / grid.n_t,
        rtol=8 * EPS,
    )


def test_envelope_grid_does_not_oversample_without_third_harmonic():
    grid = EnvelopeGrid.from_limits(**SCG)
    assert grid.n_to == grid.n_t
    assert grid.pad_scale == grid.crop_scale == 1.0
    assert grid.ffac == 2.0


def test_envelope_grid_oversamples_for_third_harmonic(envelope_grid):
    assert envelope_grid.n_to > envelope_grid.n_t
    assert envelope_grid.ffac == 6.0


def test_a_finer_dt_turns_on_envelope_oversampling():
    coarse = EnvelopeGrid.from_limits(**SCG)
    fine = EnvelopeGrid.from_limits(**SCG, dt=coarse.dt_fine / 4)
    assert fine.dt_fine < coarse.dt_fine
    assert fine.n_to > fine.n_t


def test_a_finer_dt_is_honoured_on_the_field_grid():
    default = FieldGrid.from_limits(**RDW)
    finer = FieldGrid.from_limits(**RDW, dt=default.dt_fine / 2)
    assert_allclose(finer.dt_fine, default.dt_fine / 2, rtol=8 * EPS)


def test_a_coarser_dt_is_ignored_when_the_window_demands_more():
    default = FieldGrid.from_limits(**RDW)
    asked = FieldGrid.from_limits(**RDW, dt=default.dt_fine * 10)
    assert asked.dt_fine == default.dt_fine


def test_ffac_changes_the_whole_grid():
    six = FieldGrid.from_limits(**RDW)
    four = FieldGrid.from_limits(**RDW, ffac=4.0)
    assert four.dt_fine > six.dt_fine
    assert four.d_omega != six.d_omega
    # For a response with no third harmonic, ffac = 4 makes the crop land on the fine
    # grid's last bin, so the oversampling disappears and the coarse grid is the fine
    # one
    # (Luna's RealGrid docstring, Grid.jl:43-48): half the memory and half the
    # transforms.
    assert (six.n_to // six.n_t, four.n_to // four.n_t) == (2, 1)
    assert four.n_t == four.n_to


def test_envelope_grid_covers_the_low_frequency_edge_for_a_blue_reference():
    # Luna sizes the envelope grid from the upper edge alone and then crops
    # symmetrically,
    # which drops the whole low-frequency half when the carrier sits near the blue edge.
    grid = EnvelopeGrid.from_limits(1.0, 100e-9, (90e-9, 4e-6), 400e-15, thg=True)
    assert grid.omega.min() <= 0.5 * grid.omega_min
    assert grid.omega.max() >= grid.omega_max_window


# --- masks, windows and collars -------------------------------------------------------


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_masks_select_the_documented_bands(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    assert np.array_equal(
        grid.active_mask,
        (grid.omega > 0.5 * grid.omega_min) & (grid.omega < grid.omega_max_window),
    )
    assert np.array_equal(
        grid.inband_mask,
        (grid.omega >= grid.omega_min) & (grid.omega <= grid.omega_max),
    )
    # The in-band region is strictly inside the active band, which is Luna's sidx.
    assert np.all(grid.active_mask[grid.inband_mask])
    assert grid.inband_mask.sum() < grid.active_mask.sum()


def test_frequency_limits_come_from_the_wavelength_limits(field_grid):
    low, high = RDW["wavelength_limits"]
    assert_allclose(field_grid.omega_min, 2 * np.pi * C / high, rtol=8 * EPS)
    assert_allclose(field_grid.omega_max, 2 * np.pi * C / low, rtol=8 * EPS)
    assert_allclose(
        field_grid.omega_max_window, 1.1 * field_grid.omega_max, rtol=8 * EPS
    )
    assert_allclose(field_grid.omega0, 2 * np.pi * C / 800e-9, rtol=8 * EPS)


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_spectral_window_is_one_in_band_and_zero_outside_the_collar(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    window = grid.omega_window
    assert np.all(window[grid.inband_mask] == 1.0)
    assert np.all(window[grid.omega >= grid.omega_max_window] == 0.0)
    assert np.all(window[grid.omega <= 0.5 * grid.omega_min] == 0.0)
    assert window.shape == (grid.n_omega,)


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_temporal_collar_is_never_degenerate(grid_name, request):
    # Luna puts the flat region at +/- trange/2 and the zeros at the ends of the axis,
    # so
    # on a power-of-two grid where the realised window equals the requested one the
    # collar
    # has zero width and apodises nothing. Both fixtures are exactly that case.
    grid = request.getfixturevalue(grid_name)
    full = grid.n_to * grid.dt_fine
    assert grid.t_collar.width_low >= 0.05 * full * (1 - 8 * EPS)
    assert grid.t_collar.width_high >= 0.05 * full * (1 - 8 * EPS)
    assert grid.t_window.shape == (grid.n_t,)
    assert grid.t_window_fine.shape == (grid.n_to,)
    assert grid.t_window[0] == 0.0
    assert grid.t_window_fine[0] == 0.0
    assert grid.t_window[grid.n_t // 2] == 1.0


def test_the_two_temporal_windows_agree_at_coincident_samples(envelope_grid):
    # Both come from the one collar geometry, unlike Luna's, which are built from each
    # axis's own extremes and so differ by dt - dt_fine at the upper edge.
    ratio = envelope_grid.n_to // envelope_grid.n_t
    # The same collar at the same (exactly coincident) times: a window in [0, 1] from a
    # library exponential, a few ulps of one in each sample.
    assert_allclose(
        envelope_grid.t_window,
        envelope_grid.t_window_fine[::ratio],
        rtol=0,
        atol=8 * EPS,
    )


def test_a_wider_collar_fraction_widens_the_collar_and_the_window():
    narrow = FieldGrid.from_limits(**RDW, collar_fraction=0.05)
    wide = FieldGrid.from_limits(**RDW, collar_fraction=0.25)
    assert wide.t_collar.width_low > narrow.t_collar.width_low
    assert wide.n_t * wide.dt > narrow.n_t * narrow.dt
    # The request stays flat either way.
    assert wide.t_collar.one_high == narrow.t_collar.one_high


def test_a_zero_collar_fraction_realises_the_request():
    # No temporal collar, so the window need hold only the request (SEP 21, Q16).
    grid = FieldGrid.from_limits(**RDW, collar_fraction=0.0, fft_size="smooth")
    assert grid.n_to * grid.dt_fine < 1.05 * RDW["trange"]


def test_a_window_longer_than_requested_keeps_lunas_collar_edges():
    # Just past a transform-size boundary the realised window is nearly twice the
    # request,
    # so the requested +/- trange/2 already leaves more than collar_fraction of it and
    # the
    # geometry is Luna's, unwidened.
    trange = 210e-15
    grid = FieldGrid.from_limits(
        **{**RDW, "trange": trange}, fft_size="pow2", collar_placement="inside"
    )
    assert grid.n_to * grid.dt_fine > 1.9 * trange
    assert_allclose(grid.t_collar.one_high, trange / 2, rtol=8 * EPS)


# --- oversampling fields --------------------------------------------------------------


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_crop_index_and_scales_match_the_fourier_helpers(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    assert np.array_equal(
        grid.crop_index, fourier.crop_index(grid.n_t, grid.n_to, kind=grid.kind)
    )
    assert grid.crop_index.shape == (grid.n_omega,)
    assert grid.pad_scale * grid.crop_scale == 1.0
    assert grid.pad_scale == grid.n_to / grid.n_t
    # The coarse axis really is the fine axis at those indices.
    assert_allclose(grid.omega, grid.omega_fine[grid.crop_index], rtol=8 * EPS)


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_a_grid_round_trips_a_pulse_through_its_fine_grid(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    rng = np.random.default_rng(19)
    spectrum = rng.standard_normal(grid.n_omega) + 1j * rng.standard_normal(
        grid.n_omega
    )
    spectrum *= grid.omega_window
    fine = fourier.oversample(spectrum, grid.n_to, kind=grid.kind)
    assert fine.shape == (grid.n_omega_fine,)
    recovered = fourier.crop(fine, grid.n_t, kind=grid.kind)
    assert_allclose(recovered, spectrum, rtol=8 * EPS, atol=0.0)
    # Energy is preserved because the window has already zeroed the end bins.
    assert_allclose(
        fourier.energy_omega(fine, grid.dt_fine, kind=grid.kind),
        fourier.energy_omega(spectrum, grid.dt, kind=grid.kind),
        rtol=grid.n_to * EPS,
        atol=0.0,
    )


# --- transform sizes ------------------------------------------------------------------


@pytest.mark.parametrize("factory", [FieldGrid, EnvelopeGrid])
def test_pow2_sizes_are_powers_of_two(factory):
    grid = factory.from_limits(**RDW, fft_size="pow2")
    for size in (grid.n_t, grid.n_to):
        assert size == 1 << (size.bit_length() - 1)


@pytest.mark.parametrize("factory", [FieldGrid, EnvelopeGrid])
@pytest.mark.parametrize("trange", [50e-15, 137e-15, 400e-15, 923e-15])
def test_smooth_sizes_are_even_and_two_three_five_smooth(factory, trange):
    grid = factory.from_limits(**{**RDW, "trange": trange}, fft_size="smooth")
    for size in (grid.n_t, grid.n_to):
        assert size % 2 == 0
        assert is_smooth(size)
    # The coarse size divides the fine one, which is what keeps the time samples
    # coincident; Luna's power-of-two rounding of a smooth size can duplicate bins.
    assert grid.n_to % grid.n_t == 0


def test_smooth_sizes_track_the_requirement_more_closely_than_powers_of_two():
    smooth = FieldGrid.from_limits(**RDW, fft_size="smooth")
    pow2 = FieldGrid.from_limits(**RDW, fft_size="pow2")
    assert smooth.n_to <= pow2.n_to
    assert smooth.n_t < pow2.n_t


# --- immutability and equality --------------------------------------------------------


@pytest.mark.parametrize("axis", ["t", "omega", "t_fine", "omega_fine"])
def test_stored_axes_are_read_only(field_grid, axis):
    with pytest.raises(ValueError, match="read-only"):
        getattr(field_grid, axis)[0] = 1.0


def test_a_grid_does_not_alias_the_arrays_it_was_built_from():
    axis = np.zeros(8)
    grid = TimeGrid(
        length=1.0,
        reference_wavelength=800e-9,
        t=axis,
        omega=np.arange(5, dtype=float),
        t_fine=axis,
        omega_fine=np.arange(5, dtype=float),
        dt=1.0,
        dt_fine=1.0,
        ffac=6.0,
        omega_collar=Collar(1.0, 2.0, 3.0, 4.0),
        t_collar=Collar(-2.0, -1.0, 1.0, 2.0),
    )
    assert axis.flags.writeable
    assert not grid.t.flags.writeable
    assert grid.t is not axis


def test_grids_compare_by_identity(field_grid):
    # Two grids built the same way hold equal arrays, but elementwise comparison is
    # never
    # what a caller means and would raise on a whole axis, so equality is identity.
    twin = FieldGrid.from_limits(**RDW)
    same = field_grid
    assert field_grid == same
    assert field_grid != twin
    assert isinstance(hash(field_grid), int)


# --- error paths ----------------------------------------------------------------------


@pytest.mark.parametrize("factory", [FieldGrid, EnvelopeGrid])
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"length": -1.0}, "non-negative distance"),
        ({"reference_wavelength": 0.0}, "reference_wavelength must be positive"),
        ({"reference_wavelength": -800e-9}, "reference_wavelength must be positive"),
        ({"trange": 0.0}, "trange must be a positive"),
        ({"trange": -1e-15}, "trange must be a positive"),
    ],
)
def test_from_limits_rejects_bad_scalars(factory, override, message):
    with pytest.raises(GridError, match=message):
        factory.from_limits(**{**RDW, **override})


@pytest.mark.parametrize("factory", [FieldGrid, EnvelopeGrid])
@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"dt": 0.0}, "dt must be a positive"),
        ({"dt": -1e-18}, "dt must be a positive"),
        ({"ffac": 1.5}, "ffac must be at least 2"),
        ({"ffac": 0.0}, "ffac must be at least 2"),
        ({"collar_fraction": 0.5}, "collar_fraction must lie in"),
        ({"collar_fraction": -0.1}, "collar_fraction must lie in"),
        ({"fft_size": "fast"}, "fft_size must be"),
    ],
)
def test_from_limits_rejects_bad_keywords(factory, keywords, message):
    with pytest.raises(GridError, match=message):
        factory.from_limits(**RDW, **keywords)


@pytest.mark.parametrize("factory", [FieldGrid, EnvelopeGrid])
@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ((4e-6, 90e-9), "strictly increasing"),
        ((0.0, 4e-6), "strictly increasing"),
        ((-90e-9, 4e-6), "strictly increasing"),
        ((1e-6, 1e-6), "strictly increasing"),
    ],
)
def test_from_limits_rejects_bad_wavelength_limits(factory, limits, message):
    with pytest.raises(GridError, match=message):
        factory.from_limits(**{**RDW, "wavelength_limits": limits})


@pytest.mark.parametrize("limits", [(90e-9,), (90e-9, 1e-6, 4e-6)])
def test_wavelength_limits_must_be_a_pair(limits):
    # Checked on the private helper: the public annotation is tuple[float, float], so in
    # the test session beartype rejects a wrong-length tuple before this check runs.
    with pytest.raises(GridError, match="exactly two values"):
        _frequency_limits(limits)


def corrupt(grid, **changes):
    """Return a copy of a grid with fields replaced, bypassing constructor checks."""
    clone = object.__new__(type(grid))
    for name, value in vars(grid).items():
        object.__setattr__(clone, name, value)
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"omega": np.arange(7, dtype=float)}, "n_omega is 7"),
        ({"omega_fine": np.arange(9, dtype=float)}, "n_omega_fine is 9"),
        ({"dt_fine": 1.0}, "dt/dt_fine"),
        ({"ffac": 1.0}, "ffac must be at least 2"),
    ],
)
def test_validate_catches_a_corrupted_grid(changes, message):
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 40e-15)
    with pytest.raises(GridError, match=message):
        corrupt(grid, **changes).validate()


def test_validate_catches_a_fine_grid_that_is_not_a_multiple_of_the_coarse_one():
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 40e-15)
    shrunk = grid.n_to - 2  # still even and consistent, but no longer a multiple of n_t
    with pytest.raises(GridError, match="integer multiple"):
        corrupt(
            grid,
            t_fine=np.arange(shrunk, dtype=float) * grid.dt_fine,
            omega_fine=np.arange(shrunk // 2 + 1, dtype=float) * grid.d_omega,
        ).validate()


def test_validate_catches_time_axes_that_do_not_coincide():
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 40e-15)
    shifted = grid.t_fine + grid.dt_fine
    with pytest.raises(GridError, match="does not coincide"):
        corrupt(grid, t_fine=shifted).validate()


def test_validate_catches_a_field_frequency_axis_that_is_not_the_rfft_axis():
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 40e-15)
    # dt and dt_fine are scaled together so that their ratio invariant still holds and
    # the dt == pi/omega[-1] check is the one that fires.
    with pytest.raises(GridError, match="expected pi/omega"):
        corrupt(grid, dt=grid.dt * 1.5, dt_fine=grid.dt_fine * 1.5).validate()
    # Perturb an interior bin only, so that omega[-1] and hence the dt check survive and
    # the rfft-spacing check is the one that fires.
    bent = grid.omega.copy()
    bent[1] += grid.d_omega
    with pytest.raises(GridError, match="rfft axis"):
        corrupt(grid, omega=bent).validate()


def test_validate_catches_an_envelope_frequency_axis_that_is_not_in_fft_order():
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15, thg=True)
    with pytest.raises(GridError, match="expected -pi/min"):
        corrupt(grid, dt=grid.dt * 1.5, dt_fine=grid.dt_fine * 1.5).validate()
    with pytest.raises(GridError, match=re.escape("min(omega_fine")):
        corrupt(grid, omega_fine=grid.omega_fine * 2.0).validate()
    with pytest.raises(GridError, match="FFT-order axis"):
        corrupt(grid, omega=np.sort(grid.omega)).validate()


def test_the_base_grid_has_no_layout_of_its_own():
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 40e-15)
    with pytest.raises(NotImplementedError):
        TimeGrid._validate_spectrum(grid)


# --- properties -----------------------------------------------------------------------


@given(
    factory=st.sampled_from([FieldGrid, EnvelopeGrid]),
    wavelength_low=st.floats(min_value=90e-9, max_value=700e-9),
    span=st.floats(min_value=2.0, max_value=40.0),
    trange=st.floats(min_value=20e-15, max_value=800e-15),
    ffac=st.sampled_from([2.0, 4.0, 6.0]),
    fft_size=st.sampled_from(["pow2", "smooth"]),
    collar_fraction=st.floats(min_value=0.01, max_value=0.45),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_every_grid_satisfies_its_invariants(  # noqa: PLR0913, PLR0917 - one parameter per hypothesis strategy
    factory, wavelength_low, span, trange, ffac, fft_size, collar_fraction
):
    grid = factory.from_limits(
        1.0,
        800e-9,
        (wavelength_low, wavelength_low * span),
        trange,
        ffac=ffac,
        fft_size=fft_size,
        collar_fraction=collar_fraction,
    )
    grid.validate()
    ratio = grid.n_to // grid.n_t
    assert grid.n_to % grid.n_t == 0
    assert grid.dt / grid.dt_fine == ratio
    assert np.array_equal(grid.t, grid.t_fine[::ratio])
    assert grid.n_t % 2 == 0 and grid.n_to % 2 == 0
    if fft_size == "smooth":
        assert is_smooth(grid.n_t) and is_smooth(grid.n_to)
    # The realised window always covers the request, and the collar always has room.
    full = grid.n_to * grid.dt_fine
    assert full >= trange * (1 - 8 * EPS)
    assert grid.t_collar.width_low > 0.0
    assert grid.t_collar.width_high > 0.0
    # The coarse grid reaches the apodised window edge, or as near it as the fine grid
    # allows: at ffac = 2 the fine Nyquist is omega_max itself, below the collar's foot.
    reach = min(grid.omega_max_window, grid.omega_fine.max())
    assert grid.omega.max() >= reach - grid.d_omega
    assert grid.active_mask.any()


@given(
    n_half=st.integers(min_value=4, max_value=200),
    fft_size=st.sampled_from(["pow2", "smooth"]),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_transform_sizes_are_at_least_the_request(n_half, fft_size):
    size = _fft_size(float(n_half), fft_size)
    assert size >= n_half
    assert size % 2 == 0
    if fft_size == "smooth":
        assert is_smooth(size)
        # Smoothness is the point: it never overshoots by as much as a power of two can.
        assert size <= 1 << (max(n_half - 1, 1)).bit_length()
    else:
        assert size == 1 << (size.bit_length() - 1)


@pytest.mark.medium
@given(
    factory=st.sampled_from([FieldGrid, EnvelopeGrid]),
    wavelength_low=st.floats(min_value=50e-9, max_value=1.4e-6),
    span=st.floats(min_value=1.2, max_value=60.0),
    trange=st.floats(min_value=10e-15, max_value=50e-12),
    reference=st.floats(min_value=100e-9, max_value=3e-6),
)
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grids_are_valid_over_a_wide_parameter_range(
    factory, wavelength_low, span, trange, reference
):
    grid = factory.from_limits(
        1.0, reference, (wavelength_low, wavelength_low * span), trange
    )
    grid.validate()
    assert math.isfinite(grid.dt) and grid.dt > 0.0
    assert np.all(np.isfinite(grid.omega))
    assert np.all(np.isfinite(grid.t))
    assert np.all((grid.omega_window >= 0.0) & (grid.omega_window <= 1.0))
    assert np.all((grid.t_window >= 0.0) & (grid.t_window <= 1.0))


@pytest.mark.parametrize("factory", [FieldGrid, EnvelopeGrid])
def test_construction_logs_the_realised_grid(factory, caplog):
    with caplog.at_level("DEBUG", logger="solprop.grid"):
        grid = factory.from_limits(**RDW)
    assert f"n_t={grid.n_t}" in caplog.text
    assert f"n_to={grid.n_to}" in caplog.text


def test_the_widening_fallback_keeps_the_oversampling():
    # A 2-3-5-smooth fine size can have too few divisors to leave any oversampling, and
    # the fine grid is then widened to a multiple of the coarse size the request
    # implies.
    # These parameters take that path; without it n_t would collapse to n_to.
    grid = FieldGrid.from_limits(
        1.0, 800e-9, (90e-9, 4e-6), 9.78e-12, fft_size="smooth"
    )
    assert grid.n_to % grid.n_t == 0
    assert grid.n_to > grid.n_t
    assert is_smooth(grid.n_t) and is_smooth(grid.n_to)
    grid.validate()
