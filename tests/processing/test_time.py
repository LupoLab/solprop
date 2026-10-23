# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop import stats
from solprop.exceptions import ModeError, ProcessingError
from solprop.processing import axes, fields, spectra, time


def test_moment_is_the_weighted_mean():
    x = np.linspace(-1.0, 1.0, 101)
    assert time.moment(x, np.ones_like(x)) == pytest.approx(0.0, abs=1e-15)
    # The mean of the 25 samples above 0.5 on a 101-point axis from -1 to 1.
    assert time.moment(x, (x > 0.5).astype(float)) == pytest.approx(0.76, abs=1e-12)


def test_half_max_range_of_a_gaussian_is_analytic():
    x = np.linspace(-4.0, 4.0, 8001)
    low, high = time.half_max_range(x, np.exp(-(x**2)))
    # A Gaussian exp(-x^2) is half its peak at x = +/- sqrt(ln 2); the axis has 1e-3
    # steps and the crossing is interpolated, so the error is second order in that.
    assert high - low == pytest.approx(2 * np.sqrt(np.log(2)), abs=1e-6)


def test_the_outer_width_spans_a_satellite():
    x = np.linspace(-10.0, 10.0, 4001)
    trace = np.exp(-((x - 1.0) ** 2)) + 0.9 * np.exp(-((x + 5.0) ** 2))
    assert time.fwhm(x, trace) < 2.0
    assert time.fwhm(x, trace, outer=True) > 6.0


@pytest.mark.parametrize(
    "trace", [np.zeros(64), -np.ones(64), np.ones(64), np.linspace(0.0, 1.0, 64)]
)
def test_a_width_that_is_undefined_is_nan(trace):
    x = np.linspace(-1.0, 1.0, 64)
    assert np.isnan(time.fwhm(x, trace))


def test_the_outer_width_of_a_trace_reaching_the_edge_is_nan():
    x = np.linspace(-1.0, 1.0, 64)
    trace = np.ones(64)
    trace[0] = 0.0
    assert np.isnan(time.fwhm(x, trace, outer=True))


def test_energy_calls_the_statistic_bit_for_bit(output, grid):
    stored = output.fields("fft")
    for index in range(stored.shape[0]):
        expected = float(np.sum(stats.energy(grid, stored[index])))
        assert float(time.energy(output, plane=index)) == expected


def test_peak_power_calls_the_statistic_bit_for_bit(output, grid):
    stored = output.fields("fft")[0]
    assert float(time.peak_power(output, plane=0)) == float(
        stats.peak_power(grid, stored)[0]
    )


def test_fwhm_t_calls_the_statistic_bit_for_bit(output, grid):
    stored = output.fields("fft")[0]
    assert float(time.fwhm_t(output, plane=0)) == float(stats.fwhm_t(grid, stored))
    assert float(time.fwhm_t(output, plane=0, outer=True)) == float(
        stats.fwhm_t_outer(grid, stored)
    )


def test_the_general_width_agrees_with_the_statistic(output, grid):
    t, power = fields.get_power(output, plane=-1)
    # One crossing implementation serves both, so the only difference left is how the
    # trace is formed (the accessor's power against the statistic's own): round-off.
    assert time.fwhm(t, power) == pytest.approx(
        float(stats.fwhm_t(grid, output.fields("fft")[-1])), rel=1e-12
    )


@pytest.mark.parametrize("outer", [False, True])
def test_the_statistic_measures_its_trace_with_the_general_crossing(grid, outer):
    """The temporal width and half_max_range are one implementation."""
    rng = np.random.default_rng(7)
    # A pulse with a satellite above half its height, both well inside the window.
    power = np.exp(-((grid.t / 30e-15) ** 2)) + 0.6 * np.exp(
        -(((grid.t - 120e-15) / 20e-15) ** 2)
    )
    power = power * (1.0 + 1e-3 * rng.standard_normal(power.size))
    low, high = time.half_max_range(grid.t, power, outer=outer)
    assert np.isfinite(high - low)
    assert float(stats._power_half_width(grid, power, outermost=outer)) == high - low


def test_the_duration_follows_the_dispersion_law(output, synthetic):
    widths = time.fwhm_t(output)
    positions = output["z"]
    expected = [synthetic.broadened_fwhm(float(z)) for z in positions]
    # The closed form of a Gaussian under pure group-velocity dispersion; the sampled
    # trace resolves the width with dt/tau of a few per cent, so the crossing is found
    # to second order in that.
    assert np.allclose(widths, expected, rtol=2e-3)


def test_energy_is_conserved_by_pure_dispersion(output):
    values = time.energy(output)
    assert np.allclose(values, values[0], rtol=1e-12)


def test_the_peak_power_falls_as_the_pulse_broadens(output):
    values = time.peak_power(output)
    assert values[-1] < 0.55 * values[0]


def test_oversampling_raises_the_measured_peak(output):
    coarse = float(time.peak_power(output, plane=0))
    fine = float(time.peak_power(output, plane=0, oversampling=8))
    # The launch peak sits on a sample, where the interpolated peak is the sampled
    # one up to the FFT's rounding, which differs between FFTW builds.
    assert fine >= coarse * (1 - 16 * np.finfo(float).eps)


def test_a_bandpass_reduces_the_energy(output, grid):
    total = float(time.energy(output, plane=0))
    # An explicit 4-bin edge: sharper than even the default quarter-band edge.
    window = spectra.bandpass(grid, (790e-9, 810e-9), width=4 * grid.d_omega)
    inside = float(time.energy(output, plane=0, bandpass=window))
    assert 0.1 * total < inside < 0.9 * total


def test_a_bandpass_window_must_fit_the_grid(output, grid):
    with pytest.raises(ProcessingError, match="one value per stored bin"):
        time.energy(output, plane=0, bandpass=np.ones(3))


def test_the_time_bandwidth_of_a_transform_limited_gaussian(output):
    product = float(time.time_bandwidth(output, plane=0))
    # 2 ln 2 / pi = 0.4413 for a Gaussian. The spectral density carries only about
    # six samples across its own width on this 400 fs window, so the interpolated
    # half-maximum crossing is second order in that ratio: below one per cent.
    assert product == pytest.approx(2 * np.log(2) / np.pi, rel=1e-2)


def test_the_time_bandwidth_grows_with_chirp(output):
    assert time.time_bandwidth(output, plane=-1) > 1.9 * time.time_bandwidth(
        output, plane=0
    )


@pytest.mark.parametrize("method", ["moment", "peak"])
def test_the_arrival_time_of_a_delayed_pulse(synthetic, kind, method):
    delay = 40e-15
    output = synthetic.build_run(kind, t0=delay, beta2=0.0)
    measured = float(time.arrival_time(output, plane=0, method=method))
    # The pulse builder places the peak at t0 exactly; the moment of a symmetric pulse
    # is the same point, and the peak is found on a grid of dt ~ 1 fs.
    assert measured == pytest.approx(
        delay, abs=1e-15 if method == "moment" else 1e-15 * 2
    )


def test_arrival_time_refuses_an_unknown_method():
    with pytest.raises(ProcessingError, match="method must be one of"):
        axes._check_choice("centroid", time.ARRIVAL_METHODS, "method")


def test_the_field_autocorrelation_is_the_energy_at_zero_delay(output):
    delay, correlation = time.field_autocorrelation(output, plane=0)
    zero = complex(correlation[delay.size // 2])
    # The Wiener-Khinchin sum times the sample spacing is the energy integral, which is
    # the same functional solprop.fourier evaluates in the frequency domain.
    assert zero.real == pytest.approx(float(time.energy(output, plane=0)), rel=1e-12)
    assert zero.imag == pytest.approx(0.0, abs=1e-16 * abs(zero.real))


def test_the_intensity_autocorrelation_peaks_at_zero_delay(output):
    delay, correlation = time.intensity_autocorrelation(output, plane=0)
    assert int(np.argmax(correlation)) == delay.size // 2
    # The correlation of a non-negative trace is non-negative; cancellation in the
    # transform pair puts the empty delays a few ulp of the peak below zero.
    assert np.all(correlation >= -1e-12 * correlation.max())


def test_the_intensity_autocorrelation_of_a_gaussian_is_wider_by_root_two(output):
    delay, correlation = time.intensity_autocorrelation(output, plane=0)
    width = time.fwhm(delay, correlation)
    # The autocorrelation of a Gaussian of width tau is a Gaussian of width sqrt(2) tau.
    assert width == pytest.approx(
        np.sqrt(2) * float(time.fwhm_t(output, plane=0)), rel=5e-3
    )


def test_the_coherence_time_of_a_gaussian_spectrum(output, synthetic):
    measured = float(time.coherence_time(output, plane=0))
    # For a transform-limited Gaussian the field autocorrelation has the width of the
    # pulse, so its squared modulus is narrower by sqrt(2).
    assert measured == pytest.approx(synthetic.fwhm / np.sqrt(2), rel=5e-3)


@pytest.mark.parametrize(
    ("modes", "expected"), [(None, (2,)), (0, ()), ([0, 1], (2,)), ("sum", ())]
)
def test_mode_selection_shapes_a_metric(synthetic, modes, expected):
    output = synthetic.build_run("envelope", n_modes=2, n_pol=2)
    assert np.shape(time.energy(output, plane=0, modes=modes)) == expected
    assert np.shape(time.peak_power(output, plane=0, modes=modes)) == expected
    assert np.shape(time.fwhm_t(output, plane=0, modes=modes)) == expected


def test_a_summed_peak_power_is_the_peak_of_the_sum(synthetic, grid):
    output = synthetic.build_run("envelope", n_modes=2)
    total = float(time.peak_power(output, plane=0, modes="sum"))
    each = time.peak_power(output, plane=0)
    # The modes here peak at the same instant, so the sum of the peaks is the peak of
    # the sum; what matters is that it is neither the largest peak nor a vector.
    assert total == pytest.approx(float(np.sum(each)), rel=1e-12)
    assert total > float(np.max(each))


def test_a_summed_energy_adds_the_modes(synthetic):
    output = synthetic.build_run("envelope", n_modes=2)
    assert float(time.energy(output, plane=0, modes="sum")) == pytest.approx(
        float(np.sum(time.energy(output, plane=0))), rel=1e-15
    )


def test_metrics_refuse_a_mode_that_does_not_exist(synthetic):
    output = synthetic.build_run("envelope", n_modes=2)
    with pytest.raises(ModeError, match="out of range"):
        time.fwhm_t(output, plane=0, modes=5)
    with pytest.raises(ModeError, match="out of range"):
        time.fwhm_t(output, plane=0, modes=[0, 5])


@pytest.mark.parametrize(
    ("modes", "expected"),
    [(None, [[0], [1]]), ("sum", [[0, 1]]), (1, [[1]]), ([0, 1], [[0], [1]])],
)
def test_mode_groups_lists_the_groups_a_metric_is_evaluated_over(modes, expected):
    assert time._mode_groups(2, modes) == expected
