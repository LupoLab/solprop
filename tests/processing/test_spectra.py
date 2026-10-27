# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.grid import EnvelopeGrid
from solprop.processing import spectra
from solprop.processing._examples import example_output
from solprop.processing.time import energy


def gaussian(x, centre, width):
    return np.exp(-0.5 * ((x - centre) / width) ** 2)


def test_bandpass_is_one_inside_and_zero_outside(grid):
    window = spectra.bandpass(grid, (700e-9, 900e-9))
    two_pi_c = 2 * np.pi * 299792458.0
    inside = (grid.omega > two_pi_c / 890e-9) & (grid.omega < two_pi_c / 710e-9)
    assert np.all(window[inside] == 1.0)
    far = (grid.omega > two_pi_c / 400e-9) & (grid.omega < two_pi_c / 300e-9)
    assert np.all(window[far] == 0.0)


def test_bandpass_takes_its_limits_in_either_order(grid):
    assert np.array_equal(
        spectra.bandpass(grid, (900e-9, 700e-9)),
        spectra.bandpass(grid, (700e-9, 900e-9)),
    )


def test_a_narrower_taper_is_sharper(grid):
    wide = spectra.bandpass(grid, (700e-9, 900e-9))
    narrow = spectra.bandpass(grid, (700e-9, 900e-9), width=4 * grid.d_omega)
    # A narrower edge admits fewer partially transmitted bins.
    assert _edge_bins(narrow) < _edge_bins(wide)


def _edge_bins(window):
    """Return the number of partially transmitted bins: the two edges together."""
    return np.count_nonzero((window > 0.0) & (window < 1.0))


def _band_width(limits):
    """Return a pass band's angular width, rad/s, formed as `bandpass` forms it."""
    low, high = sorted(2.0 * np.pi * C / limit for limit in limits)
    return high - low


def test_a_wide_band_keeps_the_64_bin_edge():
    """Past 256 bins the default edge is Luna's 64 bins (DEFAULT_BANDPASS_BINS)."""
    # A 4 ps window: bins of 1.6e12 rad/s, so 500-1500 nm spans about 1600 of them.
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (400e-9, 2e-6), 4e-12)
    limits = (500e-9, 1500e-9)
    assert _band_width(limits) > 4 * spectra.DEFAULT_BANDPASS_BINS * grid.d_omega
    assert np.array_equal(
        spectra.bandpass(grid, limits),
        spectra.bandpass(grid, limits, width=64 * grid.d_omega),
    )


def test_a_narrow_band_gets_edges_a_quarter_of_its_width(grid):
    """A band narrower than 256 bins is not swamped by its own edges."""
    limits = (790e-9, 810e-9)
    band = _band_width(limits)
    assert band < 4 * spectra.DEFAULT_BANDPASS_BINS * grid.d_omega
    window = spectra.bandpass(grid, limits)
    assert np.array_equal(window, spectra.bandpass(grid, limits, width=band / 4))
    # Each Planck edge is antisymmetric about its midpoint and transmits half its width,
    # so the window's integrated width is the band's times 1 + 1/4, up to the bin
    # quantisation of the two edges and the band (one bin each, three in all).
    integrated = float(np.sum(window)) * grid.d_omega
    assert integrated == pytest.approx(1.25 * band, abs=3 * grid.d_omega)
    assert _edge_bins(window) < _edge_bins(
        spectra.bandpass(grid, limits, width=64 * grid.d_omega)
    )


def test_an_empty_band_needs_an_explicit_width(grid):
    with pytest.raises(ProcessingError, match="width must be positive"):
        spectra.bandpass(grid, (800e-9, 800e-9))
    assert spectra.bandpass(grid, (800e-9, 800e-9), width=4 * grid.d_omega).max() > 0.0


@pytest.mark.parametrize("limits", [(0.0, 900e-9), (-1e-6, 900e-9)])
def test_bandpass_refuses_a_non_positive_wavelength(grid, limits):
    with pytest.raises(ProcessingError, match="must be positive"):
        spectra.bandpass(grid, limits)


def test_bandpass_refuses_a_non_positive_width(grid):
    with pytest.raises(ProcessingError, match="width must be positive"):
        spectra.bandpass(grid, (700e-9, 900e-9), width=0.0)


@pytest.mark.parametrize(("axis", "turn"), [("f", C), ("omega", 2 * np.pi * C)])
def test_a_frequency_band_is_the_same_window_as_its_wavelengths(grid, axis, turn):
    in_wavelength = spectra.bandpass(grid, (700e-9, 900e-9))
    on_axis = spectra.bandpass(grid, (turn / 900e-9, turn / 700e-9), axis=axis)
    np.testing.assert_allclose(on_axis, in_wavelength, atol=1e-12)


def test_a_frequency_band_may_start_at_zero(grid):
    window = spectra.bandpass(grid, (0.0, C / 800e-9), axis="f")
    red = (grid.omega > 0.0) & (grid.omega < 2 * np.pi * C / 900e-9)
    blue = grid.omega > 2 * np.pi * C / 600e-9
    assert np.any(red)
    assert np.all(window[red] == 1.0)
    assert np.all(window[blue] == 0.0)


def test_bandpass_refuses_a_negative_frequency(grid):
    with pytest.raises(ProcessingError, match="must not be negative"):
        spectra.bandpass(grid, (-1e14, 1e15), axis="f")


def test_a_passband_builds_the_window_of_its_axis_and_width(grid):
    band = spectra.Passband((1e14, 4e14), "f", width=8 * grid.d_omega)
    np.testing.assert_array_equal(
        spectra._as_window(grid, band),
        spectra.bandpass(grid, (1e14, 4e14), axis="f", width=8 * grid.d_omega),
    )


def test_a_passband_defaults_to_wavelength(grid):
    np.testing.assert_array_equal(
        spectra._as_window(grid, spectra.Passband((700e-9, 900e-9))),
        spectra._as_window(grid, (700e-9, 900e-9)),
    )


@pytest.mark.parametrize(
    ("limits", "axis", "match"),
    [
        ((0.0, 900e-9), "wavelength", "must be positive"),
        ((-1.0, 1e15), "omega", "must not be negative"),
    ],
)
def test_a_passband_refuses_what_names_no_light(limits, axis, match):
    with pytest.raises(ProcessingError, match=match):
        spectra.Passband(limits, axis)


def test_an_accessor_takes_a_passband():
    output = example_output()
    band = spectra.Passband((0.0, C / 800e-9), "f")
    red = float(energy(output, plane=0, bandpass=band))
    total = float(energy(output, plane=0))
    assert 0.0 < red < total


def test_specres_preserves_the_integral():
    x = np.linspace(400e-9, 1200e-9, 4001)
    density = gaussian(x, 800e-9, 20e-9)
    before = float(np.trapezoid(density, x))
    smooth_x, smooth = spectra.specres(x, density, 60e-9)
    after = float(np.trapezoid(smooth, smooth_x))
    # The kernel has unit area and is truncated at three resolution elements, where a
    # Gaussian is below 1e-11 of its peak; the signal is far from the ends of the range,
    # so nothing else leaves. The trapezoid rule on either axis is the larger error.
    assert after == pytest.approx(before, rel=1e-6)


def test_specres_broadens_a_narrow_line_to_the_resolution():
    x = np.linspace(700e-9, 900e-9, 4001)
    density = gaussian(x, 800e-9, 1e-9)
    smooth_x, smooth = spectra.specres(x, density, 20e-9)
    half = smooth > 0.5 * smooth.max()
    width = smooth_x[half][-1] - smooth_x[half][0]
    # Convolving a 1 nm line with a 20 nm kernel gives sqrt(20^2 + 2.355^2) = 20.1 nm,
    # measured here on a grid of 2 nm steps.
    assert width == pytest.approx(20.1e-9, abs=2e-9)


def test_specres_accepts_an_explicit_measure():
    x = np.linspace(400e-9, 1200e-9, 2001)
    density = gaussian(x, 800e-9, 20e-9)
    weights = np.full_like(x, x[1] - x[0])
    default_x, default_smooth = spectra.specres(x, density, 60e-9)
    given_x, given_smooth = spectra.specres(x, density, 60e-9, weights=weights)
    # On a uniform axis the central differences are the spacing, so the two agree.
    assert np.array_equal(default_x, given_x)
    assert np.allclose(default_smooth, given_smooth, rtol=1e-14)


def test_specres_smooths_every_trace_of_a_stack():
    x = np.linspace(400e-9, 1200e-9, 2001)
    stack = np.stack([gaussian(x, 700e-9, 20e-9), gaussian(x, 900e-9, 20e-9)])
    _, smooth = spectra.specres(x, stack, 40e-9)
    assert smooth.shape[0] == 2
    assert smooth[0].argmax() < smooth[1].argmax()


def test_specres_crops_to_a_requested_range():
    x = np.linspace(400e-9, 1200e-9, 2001)
    smooth_x, smooth = spectra.specres(
        x, gaussian(x, 800e-9, 20e-9), 20e-9, spec_range=(700e-9, 900e-9)
    )
    assert smooth_x[0] == pytest.approx(700e-9)
    assert smooth_x[-1] == pytest.approx(900e-9)
    assert smooth.shape == smooth_x.shape


def test_specres_clips_a_range_to_the_data():
    x = np.linspace(400e-9, 1200e-9, 2001)
    smooth_x, _ = spectra.specres(
        x, gaussian(x, 800e-9, 20e-9), 20e-9, spec_range=(100e-9, 5e-6)
    )
    assert smooth_x[0] == pytest.approx(400e-9)
    assert smooth_x[-1] == pytest.approx(1200e-9)


def test_specres_samples_the_output_axis_as_asked():
    x = np.linspace(400e-9, 1200e-9, 2001)
    density = gaussian(x, 800e-9, 20e-9)
    ten, _ = spectra.specres(x, density, 40e-9)
    two, _ = spectra.specres(x, density, 40e-9, samples=2)
    assert ten.size == pytest.approx(5 * two.size, rel=0.02)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resolution": 0.0}, "resolution must be positive"),
        ({"resolution": 1e-9, "samples": 0}, "samples must be at least"),
    ],
)
def test_specres_refuses_a_degenerate_request(kwargs, message):
    x = np.linspace(400e-9, 1200e-9, 101)
    density = gaussian(x, 800e-9, 20e-9)
    resolution = kwargs.pop("resolution")
    with pytest.raises(ProcessingError, match=message):
        spectra.specres(x, density, resolution, **kwargs)


def test_specres_refuses_an_axis_that_does_not_increase():
    x = np.linspace(1200e-9, 400e-9, 101)
    with pytest.raises(ProcessingError, match="strictly increasing"):
        spectra.specres(x, np.ones_like(x), 20e-9)


def test_specres_refuses_a_range_that_misses_the_data():
    x = np.linspace(400e-9, 1200e-9, 101)
    with pytest.raises(ProcessingError, match="does not overlap"):
        spectra.specres(x, np.ones_like(x), 20e-9, spec_range=(2e-6, 3e-6))
