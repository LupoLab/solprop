# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh

from solprop.exceptions import ModeError
from solprop.plotting.spectrograms import spectrogram


def test_a_spectrogram_has_a_time_axis_and_a_spectral_axis(single):
    figure = spectrogram(single, 0.0)
    ax = figure.axes[0]
    assert ax.get_xlabel() == "Time (fs)"
    assert ax.get_ylabel() == "Wavelength (nm)"
    # The map and its colour bar.
    assert len(figure.axes) == 2


@pytest.mark.parametrize(
    ("axis", "label"),
    [
        ("wavelength", "Wavelength (nm)"),
        ("f", "Frequency (PHz)"),
        ("omega", "Angular frequency (rad/fs)"),
    ],
)
def test_a_spectrogram_draws_each_spectral_axis(single, axis, label):
    assert spectrogram(single, 0.0, axis).axes[0].get_ylabel() == label


def test_a_spectrogram_takes_a_frequency_range_in_hertz(single):
    figure = spectrogram(single, 0.0, "f", spec_range=(0.0, 1e15))
    assert figure.axes[0].get_ylim() == pytest.approx((0.0, 1.0))


def test_a_spectrogram_puts_the_pulse_at_the_carrier(single):
    figure = spectrogram(single, 0.0, trange=(-100e-15, 100e-15))
    mesh = figure.axes[0].collections[0]
    assert isinstance(mesh, QuadMesh)
    values = np.asarray(mesh.get_array())
    ordinate = np.asarray(mesh.get_coordinates())[:, 0, 1]
    brightest = float(ordinate[int(np.argmax(np.max(values, axis=1)))])
    # A transform-limited pulse at 800 nm, gated over 10 fs: the gate is broader in
    # frequency than the pulse, so the peak sits within a gate bandwidth of 800 nm.
    assert 600.0 < brightest < 1100.0


def test_the_default_gate_follows_the_time_range(single):
    narrow = spectrogram(single, 0.0, trange=(-20e-15, 20e-15))
    wide = spectrogram(single, 0.0, trange=(-200e-15, 200e-15))
    # A wider range means a wider gate, so fewer slices at the same number of positions
    # but a finer spectral axis is not what changes; the shapes differ, which is enough
    # to show the default is not a constant.
    assert (
        np.asarray(narrow.axes[0].collections[0].get_array()).shape
        != np.asarray(wide.axes[0].collections[0].get_array()).shape
    )


def test_a_spectrogram_accepts_an_explicit_gate_and_position_count(single):
    figure = spectrogram(single, 0.0, gate_fwhm=8e-15, positions=64)
    assert np.asarray(figure.axes[0].collections[0].get_array()).shape[1] > 10


def test_a_logarithmic_spectrogram_is_capped_at_zero_decibels(single):
    figure = spectrogram(single, 0.0, log=True, db_min=-30.0)
    mesh = figure.axes[0].collections[0]
    assert mesh.get_clim() == (-30.0, 0.0)
    assert float(np.max(np.asarray(mesh.get_array()))) == pytest.approx(0.0, abs=1e-12)


def test_a_spectrogram_defaults_to_the_end_of_the_run(single):
    end = spectrogram(single).axes[0].collections[0].get_array()
    last = spectrogram(single, 0.1).axes[0].collections[0].get_array()
    assert np.array_equal(np.asarray(end), np.asarray(last))


def test_a_spectrogram_draws_into_a_given_axis(single):
    figure, ax = plt.subplots()
    assert spectrogram(single, 0.0, ax=ax) is figure


def test_a_multimode_run_must_name_one_mode(multimode):
    with pytest.raises(ModeError, match="name a single mode"):
        spectrogram(multimode, 0.0)
    assert spectrogram(multimode, 0.0, modes=1) is not None


def test_a_spectrogram_reads_a_file_handler(single, on_file):
    memory = spectrogram(single, 0.0).axes[0].collections[0].get_array()
    from_file = spectrogram(on_file, 0.0).axes[0].collections[0].get_array()
    assert np.array_equal(np.asarray(memory), np.asarray(from_file))
