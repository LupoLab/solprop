# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the pulse builders: exact scaling, parity and the sign of the phase."""

import logging
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from solprop import fourier
from solprop.constants import C
from solprop.exceptions import PulseError
from solprop.grid import EnvelopeGrid, FieldGrid, TimeGrid
from solprop.pulses import (
    CustomPulse,
    CWPulse,
    DataPulse,
    GaussianPulse,
    SechPulse,
    ShapedPulse,
)

EPS = np.finfo(np.float64).eps

# A 30 fs pulse at 800 nm in a 2 ps window. Narrow enough in time that the window edges
# hold nothing (exp(-4 ln2 (1000/30)^2) is below 1e-800), narrow enough in frequency
# that the field grid's fold at zero frequency is unmeasurable, and small enough that
# the whole file runs in a couple of seconds.
WAVELENGTH = 800e-9
FWHM = 30e-15
LIMITS = (400e-9, 2e-6)
TRANGE = 2e-12
ENERGY = 1e-6

GRIDS = ("field_grid", "envelope_grid")


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)


def analytic(spectrum, grid):
    """Return the analytic signal of a built spectrum; its modulus squared is power."""
    samples = fourier.inverse(spectrum, kind=grid.kind)
    if grid.kind == "envelope":
        return samples
    return fourier.analytic_signal(np.real(samples))


def centroid(spectrum, grid):
    """Return the intensity-weighted mean arrival time of a built spectrum, s."""
    power = np.abs(analytic(spectrum, grid)) ** 2
    return float(np.sum(grid.t * power) / np.sum(power))


def gaussian_data(n_data=2048, phase=None):
    """A tabulated 30 fs Gaussian spectrum at 800 nm, with an optional phase."""
    omega = np.linspace(1.4e15, 3.4e15, n_data)
    omega_p = 2.0 * np.pi * C / WAVELENGTH
    intensity = np.exp(-((omega - omega_p) ** 2) * FWHM**2 / (4.0 * math.log(2.0)))
    angle = np.zeros_like(omega) if phase is None else phase(omega)
    return omega, intensity, angle


# --- energy and peak power ------------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
@pytest.mark.parametrize(
    "pulse",
    [
        GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY),
        GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, m=3),
        SechPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY),
        SechPulse(wavelength=WAVELENGTH, width=FWHM, energy=ENERGY),
        CustomPulse(
            wavelength=WAVELENGTH,
            shape_fn=lambda t: np.clip(1.0 - np.abs(t) / FWHM, 0.0, None),
            energy=ENERGY,
        ),
    ],
    ids=["gauss", "supergauss", "sech-fwhm", "sech-width", "custom"],
)
def test_the_requested_energy_is_realised_exactly(pulse, grid_name, request):
    grid = request.getfixturevalue(grid_name)
    spectrum = pulse.build(grid)[0, 0]
    energy = fourier.energy_omega(spectrum, grid.dt, kind=grid.kind)
    # One rescaling by a square root, then one re-evaluation of the same sum: the target
    # is 1e-12 relative and a handful of epsilon is what the arithmetic gives.
    assert_allclose(energy, ENERGY, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_requested_peak_power_is_realised_exactly(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, peak_power=3e9)
    power = np.abs(analytic(pulse.build(grid)[0, 0], grid)) ** 2
    # The peak is a maximum over the same samples the scaling was derived from, so only
    # the transform pair's round-off separates the two.
    assert_allclose(np.max(power), 3e9, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_peak_power_is_taken_after_the_spectral_phase(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    chirped = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, peak_power=3e9, phase=(0.0, 0.0, 2e-27)
    )
    flat = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, peak_power=3e9)
    chirped_energy = fourier.energy_omega(
        chirped.build(grid)[0, 0], grid.dt, kind=grid.kind
    )
    flat_energy = fourier.energy_omega(flat.build(grid)[0, 0], grid.dt, kind=grid.kind)
    # Luna's "peak power after applying any spectral phases": stretching the pulse at a
    # fixed peak must raise the energy, and here by a large factor.
    assert chirped_energy > 3.0 * flat_energy


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_spectral_phase_does_not_change_the_energy(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    plain = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY)
    phased = GaussianPulse(
        wavelength=WAVELENGTH,
        fwhm=FWHM,
        energy=ENERGY,
        phase=(1.1, 5e-15, 3e-27, 1e-41),
        t0=100e-15,
    )
    for pulse in (plain, phased):
        energy = fourier.energy_omega(pulse.build(grid)[0, 0], grid.dt, kind=grid.kind)
        assert_allclose(energy, ENERGY, rtol=1e-12, atol=0.0)


def test_the_built_spectrum_has_the_canonical_layout(field_grid):
    spectrum = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY).build(
        field_grid
    )
    assert spectrum.shape == (1, 1, field_grid.n_omega)
    assert spectrum.dtype == np.complex128
    assert spectrum.flags.c_contiguous


# --- shapes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pulse", "half_time"),
    [
        (GaussianPulse(wavelength=WAVELENGTH, fwhm=2.0, energy=1.0), 1.0),
        (GaussianPulse(wavelength=WAVELENGTH, fwhm=2.0, energy=1.0, m=4), 1.0),
        (SechPulse(wavelength=WAVELENGTH, fwhm=2.0, energy=1.0), 1.0),
    ],
    ids=["gauss", "supergauss", "sech"],
)
def test_every_shape_realises_its_half_maximum_at_half_the_fwhm(pulse, half_time):
    values = pulse.intensity(np.array([0.0, half_time]))
    # The definition of the FWHM, so only the round-off of two transcendentals.
    assert_allclose(values, [1.0, 0.5], rtol=8 * EPS, atol=0.0)


def test_a_super_gaussian_falls_off_faster_than_a_gaussian():
    t = np.array([1.5])
    gauss = GaussianPulse(wavelength=WAVELENGTH, fwhm=2.0, energy=1.0)
    flat_top = GaussianPulse(wavelength=WAVELENGTH, fwhm=2.0, energy=1.0, m=5)
    assert float(flat_top.intensity(t)[0]) < float(gauss.intensity(t)[0])


@pytest.mark.parametrize("m", [1, 2, 8])
def test_a_super_gaussian_stays_finite_far_outside_the_pulse(m):
    # The exponent is clipped before it is raised to the power 2m, so an order that
    # would otherwise overflow the intermediate underflows the exponential instead.
    values = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=1e-15, energy=1.0, m=m
    ).intensity(np.array([0.0, 1.0, 1e10]))
    assert np.all(np.isfinite(values))
    # The clip caps the exponent at 745, whose exponential is the smallest subnormal.
    assert values[-1] < 1e-300


def test_a_sech_stays_finite_far_outside_the_pulse():
    values = SechPulse(wavelength=WAVELENGTH, width=1e-15, energy=1.0).intensity(
        np.array([0.0, 1e10])
    )
    assert np.all(np.isfinite(values))
    # sech^2 of the clip is 8e-295, zero for every purpose and finite in double.
    assert values[-1] < 1e-290


def test_the_sech_natural_width_follows_from_the_fwhm():
    by_fwhm = SechPulse(wavelength=WAVELENGTH, fwhm=2.0, energy=1.0)
    by_width = SechPulse(
        wavelength=WAVELENGTH, width=1.0 / math.log(1.0 + math.sqrt(2.0)), energy=1.0
    )
    t = np.linspace(-3.0, 3.0, 51)
    # The same closed form reached two ways; the conversion is one division.
    assert_allclose(by_fwhm.intensity(t), by_width.intensity(t), rtol=8 * EPS, atol=0.0)


# --- field and envelope parity --------------------------------------------------------


def matched_grids():
    """A field and an envelope grid whose stored spectra sample the same frequencies."""
    # A dt finer than either grid requires forces both to the same fine size and hence
    # to the same bin spacing; putting the envelope carrier on an exact multiple of that
    # spacing then makes the envelope's bins a subset of the field grid's.
    dt = 1e-16
    field = FieldGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE, dt=dt)
    index = round(2.0 * np.pi * C / WAVELENGTH / field.d_omega)
    reference = 2.0 * np.pi * C / (index * field.d_omega)
    envelope = EnvelopeGrid.from_limits(1.0, reference, LIMITS, TRANGE, dt=dt)
    return field, envelope


def test_the_matched_grids_really_share_a_bin_spacing():
    field, envelope = matched_grids()
    # Both spacings are 2 pi / (n_t dt) with the same product formed from different
    # integer factors, so they agree to a couple of units in the last place.
    assert_allclose(envelope.d_omega, field.d_omega, rtol=4 * EPS, atol=0.0)


def test_a_pulse_has_the_same_spectral_density_on_both_grids():
    field, envelope = matched_grids()
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY)
    on_field = fourier.spectral_density(
        pulse.build(field)[0, 0], field.dt, kind="field"
    )
    on_envelope = fourier.spectral_density(
        pulse.build(envelope)[0, 0], envelope.dt, kind="envelope"
    )
    index = np.rint(envelope.omega / field.d_omega).astype(int)
    common = (index >= 0) & (index < field.n_omega)
    # The target is 1e-10 on the common band. Both are spectrally accurate
    # approximations of the same continuous transform, so the difference is the
    # aliasing of a Gaussian sampled far above its Nyquist frequency; measure it
    # against the peak, because the far tails are 1e-300 and carry no information.
    peak = float(np.max(on_field))
    assert_allclose(
        on_envelope[common], on_field[index[common]], rtol=0.0, atol=1e-10 * peak
    )


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_gaussian_pulse_has_the_analytic_gaussian_spectrum(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY)
    density = fourier.spectral_density(pulse.build(grid)[0, 0], grid.dt, kind=grid.kind)
    omega_p = 2.0 * np.pi * C / WAVELENGTH
    # A Gaussian of intensity FWHM tau has S(omega) proportional to
    # exp(-(omega - omega_p)^2 tau^2 / (4 ln 2)).
    expected = np.exp(-((grid.omega - omega_p) ** 2) * FWHM**2 / (4.0 * math.log(2.0)))
    keep = density > 1e-8 * density.max()
    scale = float(np.max(density) / np.max(expected[keep]))
    # Only aliasing and truncation separate the sampled transform from the closed form,
    # and both are exponentially small here; 1e-9 leaves three orders in hand.
    assert_allclose(density[keep] / scale, expected[keep], rtol=1e-9, atol=0.0)


# --- signs and the phase origin -------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_transform_limited_pulse_has_a_flat_optics_phase(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    spectrum = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY).build(
        grid
    )[0, 0]
    band = np.abs(spectrum) > 0.05 * np.abs(spectrum).max()
    optics = fourier.spectrum_to_optics(spectrum, kind=grid.kind)
    spread = np.ptp(np.unwrap(np.angle(optics[band])))
    # The pulse sits at t = 0, so its optics phase is constant; what is left is the
    # phase of a number whose imaginary part is a few epsilon of its real part.
    assert spread < 1e-12
    # The stored spectrum, by contrast, carries the (-1)^k ramp of the half-window
    # origin: neighbouring storage bins are pi apart. On the envelope grid the band is
    # two runs at the two ends of the FFT-order axis, so only adjacent bins count.
    index = np.flatnonzero(band)
    adjacent = index[:-1][np.diff(index) == 1]
    step = np.abs(np.angle(spectrum[adjacent + 1]) - np.angle(spectrum[adjacent]))
    assert_allclose(step, np.pi, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
@pytest.mark.parametrize("t0", [-150e-15, 0.0, 250e-15])
def test_a_positive_delay_moves_the_pulse_later(t0, grid_name, request):
    grid = request.getfixturevalue(grid_name)
    spectrum = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, t0=t0
    ).build(grid)[0, 0]
    # The centroid is a ratio of two sums over n_t samples of a pulse that is
    # numerically zero at the window edges, so the only error is the transform's.
    assert_allclose(centroid(spectrum, grid), t0, rtol=0.0, atol=1e-12 * TRANGE)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_group_delay_coefficient_moves_the_pulse_later(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    spectrum = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, phase=(0.0, 120e-15)
    ).build(grid)[0, 0]
    # The first-order Taylor coefficient is the group delay, so it must act exactly as
    # t0 does; that is what makes the two ways of writing a delay interchangeable.
    assert_allclose(centroid(spectrum, grid), 120e-15, rtol=0.0, atol=1e-12 * TRANGE)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_positive_gdd_chirps_the_pulse_positively(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    gdd = 2e-27
    spectrum = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, phase=(0.0, 0.0, gdd)
    ).build(grid)[0, 0]
    signal = analytic(spectrum, grid)
    power = np.abs(signal) ** 2
    instantaneous = np.gradient(np.unwrap(np.angle(signal)), grid.dt)
    keep = power > 0.01 * power.max()
    slope = np.polyfit(grid.t[keep], instantaneous[keep], 1, w=np.sqrt(power[keep]))[0]
    assert slope > 0.0
    with_negative = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, phase=(0.0, 0.0, -gdd)
    ).build(grid)[0, 0]
    mirror = analytic(with_negative, grid)
    mirror_power = np.abs(mirror) ** 2
    mirror_keep = mirror_power > 0.01 * mirror_power.max()
    mirror_slope = np.polyfit(
        grid.t[mirror_keep],
        np.gradient(np.unwrap(np.angle(mirror)), grid.dt)[mirror_keep],
        1,
        w=np.sqrt(mirror_power[mirror_keep]),
    )[0]
    # The two chirps are exact mirror images of one another, up to the weighted fit's
    # own conditioning; 1e-6 relative is far tighter than the sign question needs.
    assert_allclose(mirror_slope, -slope, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_taylor_phase_gives_the_group_delay_it_asks_for(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    gdd = 2e-27
    spectrum = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, phase=(0.0, 0.0, gdd)
    ).build(grid)[0, 0]
    optics = fourier.spectrum_to_optics(spectrum, kind=grid.kind)
    band = np.abs(spectrum) > 0.05 * np.abs(spectrum).max()
    omega = grid.omega[band]
    order = np.argsort(omega)
    delay = np.gradient(np.unwrap(np.angle(optics[band]))[order], omega[order])
    detuning = omega[order] - 2.0 * np.pi * C / WAVELENGTH
    measured = float(np.polyfit(detuning, delay, 1)[0])
    # The group delay is read back with a second-order finite difference on the bin
    # spacing, whose leading error on a quadratic phase is a few parts in a thousand.
    assert_allclose(measured, gdd, rtol=5e-3, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_spectrum_helpers_round_trip_a_built_pulse(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    spectrum = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, phase=(0.3, 0.0, 1e-27)
    ).build(grid)[0, 0]
    optics = fourier.spectrum_to_optics(spectrum, kind=grid.kind)
    # Conjugation and a multiplication by +-1 are exact: a structural contract.
    assert np.array_equal(
        fourier.spectrum_to_internal(optics, kind=grid.kind), spectrum
    )


def test_the_carrier_envelope_phase_is_the_zeroth_taylor_coefficient(field_grid):
    plain = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY).build(
        field_grid
    )[0, 0]
    shifted = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, phase=(0.5 * np.pi,)
    ).build(field_grid)[0, 0]
    band = np.abs(plain) > 0.05 * np.abs(plain).max()
    difference = np.angle(
        fourier.spectrum_to_optics(shifted, kind="field")[band]
        / fourier.spectrum_to_optics(plain, kind="field")[band]
    )
    # A constant phase offset, recovered from a ratio of two complex numbers.
    assert_allclose(difference, 0.5 * np.pi, rtol=1e-9, atol=0.0)


# --- tabulated data -------------------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_data_pulse_lands_where_its_measured_phase_says(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    arrival = 180e-15
    omega, intensity, _ = gaussian_data()
    pulse = DataPulse.from_spectral_density(
        omega, intensity, omega * arrival, energy=ENERGY
    )
    spectrum = pulse.build(grid)[0, 0]
    # A linear optics-convention spectral phase is an arrival time; that it comes out
    # as one is the whole point of routing absolute phases through
    # spectrum_to_internal. The tolerance is the spline's, not the transform's.
    assert_allclose(centroid(spectrum, grid), arrival, rtol=0.0, atol=1e-4 * arrival)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_data_pulse_reproduces_the_shape_it_was_given(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    omega, intensity, angle = gaussian_data()
    pulse = DataPulse.from_spectral_density(omega, intensity, angle, energy=ENERGY)
    spectrum = pulse.build(grid)[0, 0]
    density = fourier.spectral_density(spectrum, grid.dt, kind=grid.kind)
    reference = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY)
    expected = fourier.spectral_density(
        reference.build(grid)[0, 0], grid.dt, kind=grid.kind
    )
    keep = expected > 1e-6 * expected.max()
    # A cubic spline of a Gaussian sampled at 2048 points over 2e15 rad/s, so about
    # forty points per spectral sigma. Its h**4 error is smallest at the peak and
    # largest six decades down the tails, where it reaches 4e-8 relative; 1e-7 keeps a
    # factor of two in hand.
    assert_allclose(density[keep], expected[keep], rtol=1e-7, atol=0.0)


def test_a_data_pulse_carries_the_requested_energy(field_grid):
    omega, intensity, angle = gaussian_data()
    pulse = DataPulse.from_spectral_density(omega, intensity, angle, energy=ENERGY)
    energy = fourier.energy_omega(
        pulse.build(field_grid)[0, 0], field_grid.dt, kind="field"
    )
    assert_allclose(energy, ENERGY, rtol=1e-12, atol=0.0)


def test_a_data_pulse_defaults_its_reference_to_the_spectral_centre_of_mass():
    omega, intensity, angle = gaussian_data()
    pulse = DataPulse.from_spectral_density(omega, intensity, angle, energy=ENERGY)
    # The data is symmetric about the carrier, so the centre of mass is the carrier;
    # the residual is the asymmetry of the sampled window, a part in 1e6.
    assert_allclose(pulse.reference_wavelength, WAVELENGTH, rtol=1e-5, atol=0.0)
    given = DataPulse.from_spectral_density(
        omega, intensity, angle, energy=ENERGY, wavelength=1030e-9
    )
    assert given.reference_wavelength == 1030e-9


def test_a_data_pulse_from_a_time_record_matches_the_pulse_it_recorded(field_grid):
    t = np.linspace(-1600e-15, 1600e-15, 16384)
    omega_p = 2.0 * np.pi * C / WAVELENGTH
    record = np.exp(-2.0 * math.log(2.0) * (t / FWHM) ** 2) * np.cos(omega_p * t)
    pulse = DataPulse.from_time(t, record, energy=ENERGY)
    density = fourier.spectral_density(
        pulse.build(field_grid)[0, 0], field_grid.dt, kind="field"
    )
    reference = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY)
    expected = fourier.spectral_density(
        reference.build(field_grid)[0, 0], field_grid.dt, kind="field"
    )
    keep = expected > 1e-6 * expected.max()
    # The record's own bin spacing is 2 pi / 3.2 ps, about twenty points per spectral
    # sigma, and the spline then costs another factor; 1e-6 leaves an order in hand.
    assert_allclose(density[keep], expected[keep], rtol=1e-6, atol=0.0)


def test_a_time_record_that_is_not_centred_still_lands_where_it_sat(field_grid):
    # The record's own axis offset is carried by an explicit linear phase, so a pulse
    # sitting at 100 fs in a record that starts at -50 fs still arrives at 100 fs.
    t = np.linspace(-50e-15, 750e-15, 4096)
    omega_p = 2.0 * np.pi * C / WAVELENGTH
    record = np.exp(-2.0 * math.log(2.0) * ((t - 100e-15) / FWHM) ** 2) * np.cos(
        omega_p * t
    )
    pulse = DataPulse.from_time(t, record, energy=ENERGY)
    arrival = centroid(pulse.build(field_grid)[0, 0], field_grid)
    assert_allclose(arrival, 100e-15, rtol=0.0, atol=1e-3 * 100e-15)


def test_a_complex_time_record_is_read_as_the_analytic_signal(field_grid):
    t = np.linspace(-400e-15, 400e-15, 4096)
    omega_p = 2.0 * np.pi * C / WAVELENGTH
    shape = np.exp(-2.0 * math.log(2.0) * (t / FWHM) ** 2)
    real = DataPulse.from_time(t, shape * np.cos(omega_p * t), energy=ENERGY)
    complex_record = DataPulse.from_time(
        t, shape * np.exp(-1j * omega_p * t), energy=ENERGY
    )
    keep = np.abs(real.spectrum) > 1e-6 * np.abs(real.spectrum).max()
    # The two records describe the same field, so their spectra agree in shape once
    # each is normalised by its own peak; the analytic signal is twice the real
    # field's one-sided spectrum, which the normalisation removes.
    assert_allclose(
        np.abs(complex_record.spectrum[keep]) / np.abs(complex_record.spectrum).max(),
        np.abs(real.spectrum[keep]) / np.abs(real.spectrum).max(),
        rtol=1e-9,
        atol=0.0,
    )


def test_a_data_pulse_warns_when_the_grid_reaches_past_the_data(field_grid, caplog):
    omega, intensity, angle = gaussian_data()
    pulse = DataPulse.from_spectral_density(omega, intensity, angle, energy=ENERGY)
    with caplog.at_level(logging.WARNING, logger="solprop.pulses"):
        pulse.build(field_grid)
    assert "the input is zero outside the data" in caplog.text


def test_a_data_pulse_is_zero_outside_the_range_of_its_data(field_grid):
    omega, intensity, angle = gaussian_data()
    pulse = DataPulse.from_spectral_density(omega, intensity, angle, energy=ENERGY)
    spectrum = pulse.build(field_grid)[0, 0]
    outside = (field_grid.omega < omega[0]) | (field_grid.omega > omega[-1])
    assert np.all(spectrum[outside] == 0.0)


# --- continuous wave ------------------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
@pytest.mark.parametrize("bandwidth", [0.0, 2e-9])
def test_a_cw_field_realises_its_average_power(bandwidth, grid_name, request):
    grid = request.getfixturevalue(grid_name)
    pulse = CWPulse(
        wavelength=WAVELENGTH, average_power=5.0, bandwidth=bandwidth, seed=4
    )
    signal = analytic(pulse.build(grid)[0, 0], grid)
    flat = grid.t_window == 1.0
    mean = float(np.mean(np.abs(signal[flat]) ** 2))
    # A mean over the same samples the scaling was derived from.
    assert_allclose(mean, 5.0, rtol=1e-12, atol=0.0)


def test_a_monochromatic_cw_field_occupies_one_bin(envelope_grid):
    spectrum = CWPulse(wavelength=WAVELENGTH, average_power=5.0).build(envelope_grid)
    occupied = np.flatnonzero(np.abs(spectrum[0, 0]) > 0.0)
    assert occupied.size == 1
    nearest = envelope_grid.omega[occupied[0]]
    assert abs(nearest - 2.0 * np.pi * C / WAVELENGTH) <= 0.5 * envelope_grid.d_omega


def test_a_cw_field_is_reproducible_from_its_seed(envelope_grid):
    first = CWPulse(
        wavelength=WAVELENGTH, average_power=5.0, bandwidth=2e-9, seed=4
    ).build(envelope_grid)
    again = CWPulse(
        wavelength=WAVELENGTH, average_power=5.0, bandwidth=2e-9, seed=4
    ).build(envelope_grid)
    other = CWPulse(
        wavelength=WAVELENGTH, average_power=5.0, bandwidth=2e-9, seed=5
    ).build(envelope_grid)
    # The same draw (the generator's stream is exact) through an exponential, a
    # hyperbolic cosine, a quotient and the average-power rescaling (a sum over the
    # band): a few ulps per bin and gamma(n) of the sum, so 16 eps plus that of the
    # peak. Another seed is another realisation, far outside it.
    bound = (16 + envelope_grid.n_omega) * EPS * float(np.max(np.abs(first)))
    assert_allclose(again, first, rtol=0, atol=bound)
    assert np.max(np.abs(other - first)) > 1e3 * bound


def test_a_cw_field_is_masked_to_the_active_band(envelope_grid):
    spectrum = CWPulse(
        wavelength=WAVELENGTH, average_power=5.0, bandwidth=2e-9, seed=4
    ).build(envelope_grid)[0, 0]
    assert np.all(spectrum[~envelope_grid.active_mask] == 0.0)


def test_a_broadband_cw_field_is_wider_than_a_monochromatic_one(envelope_grid):
    narrow = CWPulse(wavelength=WAVELENGTH, average_power=5.0).build(envelope_grid)
    broad = CWPulse(
        wavelength=WAVELENGTH, average_power=5.0, bandwidth=5e-9, seed=4
    ).build(envelope_grid)
    assert np.count_nonzero(np.abs(broad[0, 0]) > 0.0) > np.count_nonzero(
        np.abs(narrow[0, 0]) > 0.0
    )


# --- error paths ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"energy": None}, "one of energy or peak_power must be given"),
        ({"peak_power": 1e9}, "only one of energy or peak_power"),
        ({"energy": -1e-6}, "energy must be a finite positive number"),
        ({"energy": 0.0}, "energy must be a finite positive number"),
        ({"energy": math.nan}, "energy must be a finite positive number"),
        ({"wavelength": 0.0}, "wavelength must be a finite positive number"),
        ({"wavelength": -800e-9}, "wavelength must be a finite positive number"),
        ({"fwhm": -30e-15}, "fwhm must be a finite positive number"),
        ({"m": 0}, "m must be a super-Gaussian order"),
        ({"phase": (1.0, math.inf)}, "phase coefficient must be finite"),
    ],
)
def test_gaussian_pulse_validation_rejects_bad_settings(overrides, message):
    settings_ = {"wavelength": WAVELENGTH, "fwhm": FWHM, "energy": ENERGY, **overrides}
    with pytest.raises(PulseError, match=message):
        GaussianPulse(**settings_)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "one of fwhm or width must be given"),
        ({"fwhm": FWHM, "width": FWHM}, "only one of fwhm or width"),
        ({"width": -1e-15}, "width must be a finite positive number"),
        ({"fwhm": math.inf}, "fwhm must be a finite positive number"),
    ],
)
def test_sech_pulse_validation_rejects_bad_durations(overrides, message):
    with pytest.raises(PulseError, match=message):
        SechPulse(wavelength=WAVELENGTH, energy=ENERGY, **overrides)


@pytest.mark.parametrize("t0", [1e-11, -1e-11, math.nan])
def test_a_delay_outside_the_window_is_rejected(t0, field_grid):
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, t0=t0)
    with pytest.raises(PulseError, match="t0 must lie strictly inside"):
        pulse.build(field_grid)


@pytest.mark.parametrize(
    ("shape_fn", "message"),
    [
        (lambda t: -np.ones_like(t), "finite and non-negative"),
        (lambda t: np.full_like(t, np.nan), "finite and non-negative"),
        (np.zeros_like, "zero at every sample"),
    ],
    ids=["negative", "not-finite", "all-zero"],
)
def test_a_bad_custom_envelope_is_rejected(shape_fn, message, field_grid):
    pulse = CustomPulse(wavelength=WAVELENGTH, shape_fn=shape_fn, energy=ENERGY)
    with pytest.raises(PulseError, match=message):
        pulse.build(field_grid)


def test_an_envelope_too_small_to_scale_is_rejected(field_grid):
    # 1e-320 is subnormal, so the energy of the built spectrum underflows to exactly
    # zero and no finite factor reaches the requested energy. Raising here is what
    # keeps an inf out of the initial state.
    pulse = CustomPulse(
        wavelength=WAVELENGTH,
        shape_fn=lambda t: np.full_like(t, 1e-320),
        energy=ENERGY,
    )
    with pytest.raises(PulseError, match="has no energy on this grid"):
        pulse.build(field_grid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"omega": np.linspace(1e15, 3e15, 8), "spectrum": np.ones(7, complex)},
            "omega and spectrum must have the same length",
        ),
        (
            {"omega": np.linspace(1e15, 3e15, 3), "spectrum": np.ones(3, complex)},
            "at least 4 points",
        ),
        (
            {
                "omega": np.linspace(1e15, 3e15, 8),
                "spectrum": np.full(8, np.nan, complex),
            },
            "must be finite everywhere",
        ),
        (
            {"omega": np.linspace(3e15, 1e15, 8), "spectrum": np.ones(8, complex)},
            "strictly increasing and positive",
        ),
        (
            {"omega": np.linspace(0.0, 3e15, 8), "spectrum": np.ones(8, complex)},
            "strictly increasing and positive",
        ),
        (
            {
                "omega": np.linspace(1e15, 3e15, 8),
                "spectrum": np.ones(8, complex),
                "energy": -1.0,
            },
            "energy must be a finite positive number",
        ),
        (
            {
                "omega": np.linspace(1e15, 3e15, 8),
                "spectrum": np.ones(8, complex),
                "wavelength": -1.0,
            },
            "wavelength must be a finite positive number",
        ),
    ],
)
def test_data_pulse_validation_rejects_bad_data(kwargs, message):
    with pytest.raises(PulseError, match=message):
        DataPulse(**{"energy": ENERGY, **kwargs})


def test_data_outside_the_grid_band_is_rejected(field_grid):
    omega = np.linspace(1e13, 5e13, 16)
    pulse = DataPulse.from_spectral_density(
        omega, np.ones_like(omega), np.zeros_like(omega), energy=ENERGY
    )
    with pytest.raises(PulseError, match="does not reach the grid's active band"):
        pulse.build(field_grid)


def test_a_data_pulse_rejects_a_delay_outside_the_window(field_grid):
    omega, intensity, angle = gaussian_data()
    pulse = DataPulse.from_spectral_density(
        omega, intensity, angle, energy=ENERGY, t0=1e-11
    )
    with pytest.raises(PulseError, match="t0 must lie strictly inside"):
        pulse.build(field_grid)


def test_from_spectral_density_rejects_mismatched_arrays():
    omega = np.linspace(1e15, 3e15, 8)
    with pytest.raises(PulseError, match="intensity and spectral_phase"):
        DataPulse.from_spectral_density(omega, np.ones(8), np.zeros(7), energy=ENERGY)


@pytest.mark.parametrize(
    ("t", "field", "message"),
    [
        (np.linspace(0.0, 1e-12, 8), np.ones(7), "t and field must have the same"),
        (np.linspace(0.0, 1e-12, 3), np.ones(3), "at least 4 samples"),
        (np.array([0.0, 1e-15, 3e-15, 9e-15]), np.ones(4), "uniformly spaced"),
        (np.linspace(1e-12, 0.0, 8), np.ones(8), "uniformly spaced"),
    ],
    ids=["mismatch", "too-short", "non-uniform", "decreasing"],
)
def test_from_time_rejects_a_bad_record(t, field, message):
    with pytest.raises(PulseError, match=message):
        DataPulse.from_time(t, field, energy=ENERGY)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wavelength": 0.0}, "wavelength must be a finite positive number"),
        ({"average_power": -1.0}, "average_power must be a finite positive number"),
        ({"bandwidth": -1e-9}, "bandwidth must be a finite non-negative"),
        ({"bandwidth": math.inf}, "bandwidth must be a finite non-negative"),
    ],
)
def test_cw_pulse_validation_rejects_bad_settings(overrides, message):
    settings_ = {"wavelength": WAVELENGTH, "average_power": 5.0, **overrides}
    with pytest.raises(PulseError, match=message):
        CWPulse(**settings_)


def test_a_cw_wavelength_outside_the_band_is_rejected(envelope_grid):
    with pytest.raises(PulseError, match="lies outside the"):
        CWPulse(wavelength=10e-6, average_power=5.0).build(envelope_grid)


def test_an_envelope_of_the_wrong_length_is_rejected(field_grid):
    # Written as a subclass rather than a CustomPulse because the test session's
    # jaxtyping hook rejects the mismatched shape at the annotated boundary before the
    # check in _power_envelope runs. Outside the test session it is the only guard.
    class WrongLength(ShapedPulse):
        def intensity(self, t):
            return np.zeros(3)

    pulse = WrongLength(wavelength=WAVELENGTH, energy=ENERGY)
    with pytest.raises(PulseError, match="one value per time sample"):
        pulse.build(field_grid)


def test_the_base_class_has_no_envelope_of_its_own():
    with pytest.raises(NotImplementedError):
        ShapedPulse(wavelength=WAVELENGTH, energy=ENERGY).intensity(np.zeros(4))


def test_every_builder_satisfies_the_pulse_protocol(field_grid):
    omega, intensity, angle = gaussian_data(n_data=64)
    builders = [
        GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY),
        SechPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY),
        CustomPulse(wavelength=WAVELENGTH, shape_fn=np.ones_like, energy=ENERGY),
        DataPulse.from_spectral_density(omega, intensity, angle, energy=ENERGY),
        CWPulse(wavelength=WAVELENGTH, average_power=5.0),
    ]
    for builder in builders:
        assert builder.build(field_grid).shape == (1, 1, field_grid.n_omega)


# --- properties -----------------------------------------------------------------------


@given(
    fwhm=st.floats(min_value=20e-15, max_value=200e-15),
    energy=st.floats(min_value=1e-9, max_value=1e-2),
    gdd=st.floats(min_value=-1e-27, max_value=1e-27),
    t0=st.floats(min_value=-300e-15, max_value=300e-15),
    kind=st.sampled_from(GRIDS),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_energy_and_arrival_hold_for_any_pulse(fwhm, energy, gdd, t0, kind):
    grid: TimeGrid = (
        FieldGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)
        if kind == "field_grid"
        else EnvelopeGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)
    )
    spectrum = GaussianPulse(
        wavelength=WAVELENGTH, fwhm=fwhm, energy=energy, phase=(0.0, 0.0, gdd), t0=t0
    ).build(grid)[0, 0]
    realised = fourier.energy_omega(spectrum, grid.dt, kind=grid.kind)
    assert_allclose(realised, energy, rtol=1e-12, atol=0.0)
    # A symmetric spectral phase leaves the centroid where the delay put it, whatever
    # the chirp does to the duration. The bound is a tenth of a sample: the centroid of
    # a chirped pulse is a ratio of two sums with cancellation, and the strategies allow
    # a stretch factor of about seven, which brings the tails towards the window edge.
    assert_allclose(centroid(spectrum, grid), t0, rtol=0.0, atol=0.1 * grid.dt)
