# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grid suggestions: a starting point from the input pulse (SEP 41, S10)."""

import numpy as np
import pytest

from solprop.constants import C
from solprop.exceptions import GridError
from solprop.grid import suggest
from solprop.pulses import CWPulse, DataPulse, GaussianPulse, PulseSum, SechPulse


def test_a_short_capillary_pulse_gets_a_band_ten_bandwidths_wide():
    pulse = GaussianPulse(wavelength=800e-9, fwhm=7.5e-15, energy=275e-6)
    suggestion = suggest(pulse)
    shortest, longest = suggestion.wavelength_limits
    bandwidth = 2 * np.log(2) / np.pi / 7.5e-15
    # Exact up to the rounding outwards to two figures (class P on the rule).
    assert shortest <= C / (C / 800e-9 + 10 * bandwidth) < shortest * 1.05
    assert longest == 4e-6  # the floor at a fifth of the carrier
    assert suggestion.trange >= 30 * 7.5e-15
    assert "starting point" in suggestion.note


def test_a_long_fibre_pulse_spans_an_octave_about_its_carrier():
    pulse = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8)
    shortest, longest = suggest(pulse, route="gnlse").wavelength_limits
    centre = C / 835e-9
    assert shortest <= C / (1.5 * centre) < shortest * 1.05
    assert longest >= C / (0.5 * centre) > longest / 1.05


def test_delays_and_several_pulses_widen_the_window_and_the_band():
    pump = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    seed = GaussianPulse(wavelength=400e-9, fwhm=30e-15, energy=1e-9, t0=200e-15)
    alone = suggest(pump)
    both = suggest([pump, PulseSum((pump, seed))])
    # Each window is rounded to two figures, so compare with a rounding allowance.
    assert both.trange >= (alone.trange + 400e-15) * (1 - 1e-12)
    assert both.wavelength_limits[0] < alone.wavelength_limits[0]


def test_a_measured_spectrum_reads_its_band_from_the_data():
    frequency = np.linspace(300e12, 450e12, 400)
    power = np.exp(-(((frequency - 375e12) / 10e12) ** 2))
    pulse = DataPulse.from_spectral_density(
        2 * np.pi * frequency, power, np.zeros_like(power), energy=1e-6
    )
    shortest, longest = suggest(pulse).wavelength_limits
    assert shortest < 800e-9 < longest


def test_what_cannot_be_suggested_is_refused_at_its_parameter():
    with pytest.raises(GridError, match="CWPulse") as caught:
        suggest(CWPulse(wavelength=800e-9, average_power=1.0))
    assert caught.value.parameter == "pulses"
    with pytest.raises(GridError, match="no pulses"):
        suggest([])
    with pytest.raises(GridError, match="radial") as route:
        suggest(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), route="radial"
        )
    assert route.value.parameter == "route"
