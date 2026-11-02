# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measured spectra read from text files by `solprop.pulses.load_spectrum`.

Every file is written into ``tmp_path`` from an analytic Gaussian, at ``%.17g``, which
round-trips a double exactly, so the only errors are the reader's and the spline's.
"""

import io
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from runtime_types import unchecked
from solprop import fourier
from solprop.constants import C
from solprop.exceptions import PulseError, SolpropWarning
from solprop.grid import EnvelopeGrid
from solprop.pulses import DataPulse, GaussianPulse, load_spectrum

# A 30 fs pulse at 800 nm, tabulated over 554 nm to 1.35 um so that the data covers the
# grid's whole active band, in a 4 ps window wide enough for the chirped pulses below.
WAVELENGTH = 800e-9
OMEGA_P = 2.0 * np.pi * C / WAVELENGTH
FWHM = 30e-15
ENERGY = 1e-6
N_DATA = 2048


@pytest.fixture(scope="module")
def grid():
    return EnvelopeGrid.from_limits(0.01, WAVELENGTH, (600e-9, 1.1e-6), 4e-12)


def gaussian_density(omega):
    """The power spectral density per unit angular frequency of the 30 fs pulse."""
    return np.exp(-((omega - OMEGA_P) ** 2) * FWHM**2 / (4.0 * math.log(2.0)))


def frequency_table(n_data=N_DATA):
    """The angular frequency axis and density of the Gaussian, uniform in omega."""
    omega = np.linspace(1.4e15, 3.4e15, n_data)
    return omega, gaussian_density(omega)


def write(path, *columns, delimiter=" ", header=""):
    """Write the columns as a text table at full precision and return the path."""
    np.savetxt(
        path,
        np.column_stack(columns),
        fmt="%.17g",
        delimiter=delimiter,
        header=header,
        encoding="utf-8",
    )
    return path


def built(pulse, grid):
    return pulse.build(grid)[0, 0]


def analytic(spectrum, grid):
    """The analytic signal of a spectrum built on the envelope grid."""
    return fourier.inverse(spectrum, kind=grid.kind)


def centroid(spectrum, grid):
    """The intensity-weighted mean arrival time of a built spectrum, s."""
    power = np.abs(analytic(spectrum, grid)) ** 2
    return float(np.sum(grid.t * power) / np.sum(power))


def density(spectrum, grid):
    return fourier.spectral_density(spectrum, grid.dt, kind=grid.kind)


def phase_of(pulse):
    """The phase a pulse holds as given, which load_spectrum always sets."""
    assert pulse.spectral_phase is not None
    return pulse.spectral_phase


def loaded_phase(*arguments, **keywords):
    return phase_of(load_spectrum(*arguments, **keywords))


def relative_l2(values, reference):
    return np.linalg.norm(values - reference) / np.linalg.norm(reference)


def chirped(n_data, group_delay_dispersion=20000e-30):
    """A Gaussian spectrum at 800 nm with the given dispersion, over +-20 % in omega."""
    omega = np.linspace(0.8 * OMEGA_P, 1.2 * OMEGA_P, n_data)
    power = np.exp(-(((omega - OMEGA_P) / (0.03 * OMEGA_P)) ** 2))
    angle = 0.5 * group_delay_dispersion * (omega - OMEGA_P) ** 2
    return omega, power, angle


def wrap(angle):
    return np.angle(np.exp(1j * angle))


@pytest.fixture
def frequency_file(tmp_path):
    omega, power = frequency_table()
    return write(tmp_path / "frequency.txt", omega / (2.0 * np.pi), power)


@pytest.fixture
def wavelength_file(tmp_path):
    # Uniform in wavelength over the same span, as a spectrometer samples, holding the
    # density per unit wavelength: P_lambda = P_omega |d omega / d lambda|, with
    # |d omega / d lambda| = 2 pi c / lambda**2.
    wavelength = np.linspace(2.0 * np.pi * C / 3.4e15, 2.0 * np.pi * C / 1.4e15, N_DATA)
    omega = 2.0 * np.pi * C / wavelength
    power = gaussian_density(omega) * 2.0 * np.pi * C / wavelength**2
    return write(tmp_path / "wavelength.txt", wavelength * 1e9, power)


def read(file, axis):
    if axis == "frequency":
        return load_spectrum(file, energy=ENERGY)
    return load_spectrum(file, energy=ENERGY, axis="wavelength", axis_unit="nm")


# --- energy, centre and the wavelength axis -------------------------------------------


# Class T against the analytic spectrum: the cubic spline's h**4 error through a
# Gaussian of about forty samples per spectral sigma, measured as 4e-8 six decades down
# its tails on the frequency axis (as for DataPulse itself). The wavelength axis is
# 1.16 times coarser in omega at the carrier, and 1.16**4 is 1.8: measured 1.8e-7
# there. Each keeps a factor of two.
SHAPE_RTOL = {"frequency": 1e-7, "wavelength": 4e-7}


@pytest.mark.parametrize("axis", ["frequency", "wavelength"])
def test_a_file_reads_to_the_stated_energy_and_shape(
    axis, frequency_file, wavelength_file, grid
):
    file = frequency_file if axis == "frequency" else wavelength_file
    spectrum = built(read(file, axis), grid)
    # Class R: the build measures the energy with the grid's own functional and
    # rescales once, so this is a few roundings of a sum over n_omega bins.
    assert_allclose(
        fourier.energy_omega(spectrum, grid.dt, kind=grid.kind),
        ENERGY,
        rtol=1e-12,
        atol=0.0,
    )
    reference = density(
        built(GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY), grid),
        grid,
    )
    keep = reference > 1e-6 * reference.max()
    assert_allclose(
        density(spectrum, grid)[keep], reference[keep], rtol=SHAPE_RTOL[axis], atol=0
    )


@pytest.mark.parametrize("axis", ["frequency", "wavelength"])
def test_a_file_reads_to_its_centre_wavelength_within_one_sample(
    axis, frequency_file, wavelength_file, grid
):
    file = frequency_file if axis == "frequency" else wavelength_file
    spectrum_density = density(built(read(file, axis), grid), grid)
    centre = float(np.sum(grid.omega * spectrum_density) / np.sum(spectrum_density))
    table = np.loadtxt(file)
    spacing = (
        np.abs(np.diff(table[:, 0] * 2.0 * np.pi)).max()
        if axis == "frequency"
        # One wavelength sample, as an angular frequency at the centre.
        else 2.0 * np.pi * C * np.abs(np.diff(table[:, 0] * 1e-9)).max() / WAVELENGTH**2
    )
    # Class T: the SEP's bound, one sample of the file. The Gaussian is symmetric in
    # omega, so its centroid is the carrier to far better than that; a missing or
    # inverted wavelength Jacobian moves it by 2 sigma_lambda**2 / lambda, about one
    # sample, and the shape test above catches that by orders of magnitude.
    assert abs(centre - OMEGA_P) < spacing


def band_share(spectrum, grid):
    """The share of a built pulse's energy below the carrier frequency."""
    values = density(spectrum, grid)
    return float(np.sum(values[grid.omega < OMEGA_P]) / np.sum(values))


def test_a_wavelength_file_keeps_the_energy_of_each_band_where_it_was(
    frequency_file, wavelength_file, grid
):
    reference = band_share(
        built(GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY), grid),
        grid,
    )
    from_frequency = band_share(built(read(frequency_file, "frequency"), grid), grid)
    from_wavelength = band_share(built(read(wavelength_file, "wavelength"), grid), grid)
    # Class T, the shape test's bound: each density is the analytic one to 4e-7
    # relative, so each share is too, and the two agree with each other to 8e-7.
    assert_allclose(from_frequency, reference, rtol=4e-7, atol=0.0)
    assert_allclose(from_wavelength, reference, rtol=4e-7, atol=0.0)
    assert_allclose(from_wavelength, from_frequency, rtol=8e-7, atol=0.0)
    # The premise: the file holds P_omega / lambda**2. Read without the Jacobian it
    # would be weighted by omega**2, and with the Jacobian inverted by omega**4; each
    # moves the share by more than a percent.
    omega = grid.omega
    values = density(
        built(GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY), grid),
        grid,
    )
    for weight in (omega**2, omega**4):
        weighted = values * weight
        share = np.sum(weighted[omega < OMEGA_P]) / np.sum(weighted)
        assert abs(share - reference) > 1e-3


# --- phase ----------------------------------------------------------------------------


@pytest.mark.parametrize(("convention", "sign"), [("optics", 1.0), ("fft", -1.0)])
def test_a_delay_lands_on_the_side_its_convention_says(
    convention, sign, tmp_path, grid
):
    delay = 200e-15
    omega, power = frequency_table()
    # The same file, a pure delay of +200 fs written as +omega tau: optics reads it as
    # a pulse 200 fs late, the FFT convention (a file written for Luna.jl) as 200 fs
    # early, because the FFT convention's spectrum is the optics one's conjugate.
    file = write(tmp_path / "delay.txt", omega / (2.0 * np.pi), power, omega * delay)
    pulse = load_spectrum(file, energy=ENERGY, phase_convention=convention)
    # Class R: a cubic spline reproduces a linear phase exactly, and a symmetric error
    # in the magnitude's spline moves no centroid, so what is left is the rounding of
    # the transform and of the two sums (measured 3e-16 of the delay); 1e-9 is far
    # below any sign error, which moves it by twice the delay.
    assert_allclose(
        centroid(built(pulse, grid), grid), sign * delay, rtol=0.0, atol=1e-9 * delay
    )


def test_a_large_chirp_in_an_unwrapped_column_survives(tmp_path, grid):
    omega, power, angle = chirped(201)
    # The premise: the phase steps by more than 3 pi between samples, so no unwrapping
    # of the wrapped column could recover it.
    assert np.abs(np.diff(angle)).max() > 3.0 * np.pi
    file = write(tmp_path / "chirp.txt", omega / (2.0 * np.pi), power, angle)
    pulse = load_spectrum(file, energy=ENERGY)
    # Class P: an unwrapped column spans far more than 2 pi, so it is taken as it
    # stands, and %.17g round-trips each double.
    assert np.array_equal(phase_of(pulse), angle)
    reference = built(
        DataPulse.from_spectral_density(omega, power, angle, energy=ENERGY), grid
    )
    # Class R: the axis went through hertz and back, a rounding or two of omega, which
    # moves the phase by d(phi)/d(omega) times that, below 1e-11 rad here.
    assert relative_l2(built(pulse, grid), reference) < 1e-10
    wrapped_file = write(
        tmp_path / "wrapped.txt", omega / (2.0 * np.pi), power, wrap(angle)
    )
    with pytest.warns(SolpropWarning, match="undersampled"):
        wrapped = load_spectrum(wrapped_file, energy=ENERGY)
    # The same data wrapped is built quite wrongly, which is why the column matters.
    assert relative_l2(built(wrapped, grid), reference) > 0.1


def floored_chirp(rng, n_data=2001, group_delay_dispersion=2000e-30):
    """A chirped Gaussian whose wings below the floor hold noise in power and phase."""
    omega, power, angle = chirped(n_data, group_delay_dispersion)
    noise = power < 1e-4
    noisy_power = np.where(noise, 3e-5 * rng.random(n_data), power)
    noisy_phase = np.where(noise, rng.uniform(-np.pi, np.pi, n_data), wrap(angle))
    return omega, noisy_power, noisy_phase, angle, ~noise


def test_a_wrapped_column_unwraps_beside_a_noise_floor(tmp_path):
    rng = np.random.default_rng(20261101)
    omega, power, phase, angle, band = floored_chirp(rng)
    # The premise: within the band the true phase steps by well under pi, so its
    # wrapped form can be unwrapped there.
    assert np.abs(np.diff(angle[band])).max() < 0.2 * np.pi
    file = write(tmp_path / "floor.txt", omega / (2.0 * np.pi), power, phase)
    read_phase = loaded_phase(file, energy=ENERGY)
    turns = (read_phase[band] - angle[band]) / (2.0 * np.pi)
    # Class R: the band's phase is the true one plus a whole number of turns, to the
    # rounding of np.angle and np.unwrap on phases of up to 30 rad.
    assert_allclose(turns, np.round(turns[0]), rtol=0.0, atol=1e-12)
    # Class P: beyond the band the phase is held at the band's edges, never the noise.
    first, last = np.flatnonzero(band)[[0, -1]]
    assert np.all(read_phase[:first] == read_phase[first])
    assert np.all(read_phase[last + 1 :] == read_phase[last])


def test_noise_inside_the_band_does_not_steer_the_unwrapping(tmp_path):
    rng = np.random.default_rng(7)
    omega, power, angle = chirped(2001, 2000e-30)
    # Dropouts in the band: forty samples whose power falls into the noise and whose
    # measured phase is random. Unwrapped through them, the phase picks up a wrong turn
    # with a probability of about step / 2 pi at each.
    band = np.flatnonzero(power > 1e-2)
    dropouts = rng.choice(band[1:-1], size=40, replace=False)
    noisy_power = power.copy()
    noisy_power[dropouts] = 1e-6
    phase = wrap(angle)
    phase[dropouts] = rng.uniform(-np.pi, np.pi, dropouts.size)
    file = write(tmp_path / "dropouts.txt", omega / (2.0 * np.pi), noisy_power, phase)
    kept = np.setdiff1d(np.flatnonzero(power > 1e-4), dropouts)

    def turns(read_phase):
        return (read_phase[kept] - angle[kept]) / (2.0 * np.pi)

    floored = turns(loaded_phase(file, energy=ENERGY))
    # Class R, as beside a noise floor: one whole number of turns over the whole band.
    assert_allclose(floored, np.round(floored[0]), rtol=0.0, atol=1e-12)
    # The premise: with no floor the noise does steer it, and the jumps say so.
    with pytest.warns(SolpropWarning, match="undersampled"):
        unfloored = turns(loaded_phase(file, energy=ENERGY, floor=0.0))
    assert np.ptp(np.round(unfloored)) >= 1.0


def test_a_phase_in_degrees_reads_as_the_same_phase_in_radians(tmp_path):
    omega, power, angle = chirped(2001, 2000e-30)
    frequency = omega / (2.0 * np.pi)
    radians = write(tmp_path / "rad.txt", frequency, power, wrap(angle))
    degrees = write(tmp_path / "deg.txt", frequency, power, np.rad2deg(wrap(angle)))
    expected = loaded_phase(radians, energy=ENERGY)
    # Class R: two roundings of the degree conversion on wrapped values below pi,
    # carried through the unwrapping sum of up to 30 rad.
    assert_allclose(
        loaded_phase(degrees, energy=ENERGY, phase_unit="deg"),
        expected,
        rtol=0.0,
        atol=1e-12,
    )
    # The premise: read as radians, a wrapped column in degrees spans 360 and is taken
    # as it stands, which is a different phase.
    as_radians = loaded_phase(degrees, energy=ENERGY)
    assert np.abs(as_radians - expected).max() > 1.0


def test_an_undersampled_wrapped_column_warns(tmp_path):
    # At 801 samples this chirp steps through pi at 1.9 spectral sigma, where 1.6 % of
    # the power lies beyond; the unwrapped column builds 13.5 % wrong from there on.
    omega, power, angle = chirped(801)
    file = write(tmp_path / "under.txt", omega / (2.0 * np.pi), power, wrap(angle))
    with pytest.warns(SolpropWarning, match="undersampled"):
        load_spectrum(file, energy=ENERGY)


def test_a_finely_sampled_wrapped_column_does_not_warn(tmp_path):
    # At 4001 samples the same chirp reaches pi per sample only far below the floor;
    # the test session turns any warning into an error.
    omega, power, angle = chirped(4001)
    file = write(tmp_path / "fine.txt", omega / (2.0 * np.pi), power, wrap(angle))
    pulse = load_spectrum(file, energy=ENERGY)
    band = power > 1e-4
    turns = (phase_of(pulse)[band] - angle[band]) / (2.0 * np.pi)
    # Class R, as beside a noise floor.
    assert_allclose(turns, np.round(turns[0]), rtol=0.0, atol=1e-11)


def test_wrapped_overrides_the_range_test(tmp_path):
    # A flat power, so that every sample is above the floor and none is held.
    omega = np.linspace(1.4e15, 3.4e15, 256)
    power = np.ones_like(omega)
    small = np.linspace(-3.0, 3.0, omega.size)
    file = write(tmp_path / "small.txt", omega / (2.0 * np.pi), power, small)
    # Class P: a column taken as it stands is the column itself.
    assert np.array_equal(loaded_phase(file, energy=ENERGY, wrapped=False), small)
    # Within 2 pi, so taken as wrapped by default; its steps are small, so unwrapping
    # it changes nothing either.
    assert_allclose(loaded_phase(file, energy=ENERGY), small, rtol=0, atol=1e-15)
    # A column spanning more than 2 pi is unwrapped when told to be: a sawtooth of
    # wrapped turns, offset by one turn halfway, reads as the line it wraps.
    line = np.linspace(0.0, 40.0, omega.size)
    offset = wrap(line) + np.where(np.arange(omega.size) > 128, 2.0 * np.pi, 0.0)
    told = write(tmp_path / "told.txt", omega / (2.0 * np.pi), power, offset)
    # Class R: the rounding of np.angle and np.unwrap on phases of up to 40 rad.
    assert_allclose(
        loaded_phase(told, energy=ENERGY, wrapped=True),
        line,
        rtol=0.0,
        atol=1e-12,
    )


# --- layout ---------------------------------------------------------------------------


def test_a_two_column_file_has_a_flat_phase(frequency_file):
    pulse = load_spectrum(frequency_file, energy=ENERGY)
    # Class P: no phase column is a zero phase, exactly.
    assert np.array_equal(phase_of(pulse), np.zeros(N_DATA))


@pytest.mark.parametrize(
    ("delimiter", "header"),
    [
        (" ", ""),
        ("\t", ""),
        (",", ""),
        (", ", ""),
        (",", "frequency (Hz), power, phase"),
        (" ", "frequency power phase"),
    ],
    ids=["space", "tab", "comma", "comma-space", "csv-header", "header"],
)
def test_comma_and_whitespace_delimiters_read_alike(delimiter, header, tmp_path):
    omega, power, angle = chirped(64, 200e-30)
    columns = (omega / (2.0 * np.pi), power, angle)
    reference = load_spectrum(write(tmp_path / "plain.txt", *columns), energy=ENERGY)
    path = tmp_path / "other.txt"
    write(path, *columns, delimiter=delimiter)
    # A header row of names, comments and blank lines, as a hand-edited file has them.
    body = path.read_text(encoding="utf-8")
    extra = f"{header}\n" if header else ""
    path.write_text(f"# measured\n\n{extra}{body}\n# end\n", encoding="utf-8")
    pulse = load_spectrum(path, energy=ENERGY)
    # Class P: the same numbers read from the same text, whatever separates them.
    assert np.array_equal(pulse.omega, reference.omega)
    assert np.array_equal(pulse.spectrum, reference.spectrum)


def test_a_byte_order_mark_and_trailing_comments_are_read_past(tmp_path):
    omega, power = frequency_table(64)
    plain = write(tmp_path / "plain.csv", omega / (2.0 * np.pi), power, delimiter=",")
    lines = plain.read_text(encoding="utf-8").splitlines()
    marked = tmp_path / "marked.csv"
    text = "﻿frequency,power\n" + "\n".join(f"{row}  # sample" for row in lines)
    marked.write_text(text, encoding="utf-8")
    # Class P: the same numbers.
    assert np.array_equal(
        load_spectrum(marked, energy=ENERGY).spectrum,
        load_spectrum(plain, energy=ENERGY).spectrum,
    )


def test_a_descending_axis_is_sorted_with_its_columns(tmp_path):
    omega, power, angle = chirped(201)
    frequency = omega / (2.0 * np.pi)
    ascending = load_spectrum(
        write(tmp_path / "up.txt", frequency, power, angle), energy=ENERGY
    )
    descending = load_spectrum(
        write(tmp_path / "down.txt", frequency[::-1], power[::-1], angle[::-1]),
        energy=ENERGY,
    )
    # Class P: sorting moves the rows and nothing else.
    assert np.array_equal(descending.omega, ascending.omega)
    assert np.array_equal(descending.spectrum, ascending.spectrum)
    assert np.array_equal(phase_of(descending), phase_of(ascending))


def test_columns_picks_other_columns(tmp_path):
    omega, power, angle = chirped(201)
    frequency = omega / (2.0 * np.pi)
    ordered = load_spectrum(
        write(tmp_path / "ordered.txt", frequency, power, angle), energy=ENERGY
    )
    shuffled = write(
        tmp_path / "shuffled.txt",
        np.zeros_like(frequency),
        power,
        np.ones_like(frequency),
        frequency,
        angle,
    )
    picked = load_spectrum(shuffled, energy=ENERGY, columns=(3, 1, 4))
    # Class P: the same columns, read from other places.
    assert np.array_equal(picked.spectrum, ordered.spectrum)
    flat = load_spectrum(shuffled, energy=ENERGY, columns=(3, 1))
    assert np.array_equal(phase_of(flat), np.zeros(omega.size))


@pytest.mark.parametrize(
    ("axis", "unit", "scale"),
    [
        ("frequency", "Hz", 1.0),
        ("frequency", "GHz", 1e-9),
        ("frequency", "THz", 1e-12),
        ("frequency", "PHz", 1e-15),
        ("frequency", "rad/s", 2.0 * np.pi),
        ("frequency", "rad/fs", 2.0 * np.pi * 1e-15),
        ("wavelength", "m", 1.0),
        ("wavelength", "um", 1e6),
        ("wavelength", "\N{MICRO SIGN}m", 1e6),
        ("wavelength", "\N{GREEK SMALL LETTER MU}m", 1e6),
        ("wavelength", "nm", 1e9),
    ],
)
def test_an_axis_reads_in_any_of_its_units(axis, unit, scale, tmp_path):
    omega, power = frequency_table(256)
    si = omega / (2.0 * np.pi) if axis == "frequency" else 2.0 * np.pi * C / omega
    file = write(tmp_path / "unit.txt", si * scale, power)
    pulse = load_spectrum(file, energy=ENERGY, axis=axis, axis_unit=unit)
    default = load_spectrum(
        write(tmp_path / "si.txt", si, power), energy=ENERGY, axis=axis
    )
    # Class R: a scaling to the unit and back, a few roundings of omega.
    assert_allclose(pulse.omega, default.omega, rtol=8 * np.finfo(float).eps, atol=0)


def test_a_stream_reads_as_the_file_does(frequency_file):
    stream = io.StringIO(frequency_file.read_text(encoding="utf-8"))
    # Class P: the same text, read from memory.
    assert np.array_equal(
        load_spectrum(stream, energy=ENERGY).spectrum,
        load_spectrum(frequency_file, energy=ENERGY).spectrum,
    )


def test_a_slightly_negative_power_is_read_as_zero(tmp_path):
    omega, power = frequency_table(256)
    clipped = load_spectrum(
        write(tmp_path / "zero.txt", omega / (2.0 * np.pi), power), energy=ENERGY
    )
    # Background subtraction leaves a little below zero where there is no signal.
    dipped = np.where(power < 1e-6, -0.01, power)
    zeroed = np.where(power < 1e-6, 0.0, power)
    read = load_spectrum(
        write(tmp_path / "dip.txt", omega / (2.0 * np.pi), dipped), energy=ENERGY
    )
    expected = load_spectrum(
        write(tmp_path / "zeroed.txt", omega / (2.0 * np.pi), zeroed), energy=ENERGY
    )
    # Class P: a clipped value is an exact zero.
    assert np.array_equal(read.spectrum, expected.spectrum)
    assert not np.array_equal(read.spectrum, clipped.spectrum)


# --- refusals -------------------------------------------------------------------------


GOOD_ROWS = "1e14 0.1\n2e14 1\n3e14 1\n4e14 0.1\n"


@pytest.mark.parametrize(
    ("text", "keywords", "message"),
    [
        ("1e14 0.1\n2e14 1\nx\n", {}, "line 3 of the stream is not a row of numbers"),
        ("1e14 0.1 0\n2e14 1\n", {}, "line 2 of the stream has 2 columns"),
        ("1e14 0.1, 0\n", {}, "line 1 of the stream is not a row of numbers"),
        ("# nothing\n\n", {}, "no rows of numbers"),
        ("freq power\n", {}, "no rows of numbers"),
        ("1e14\n2e14\n3e14\n4e14\n", {}, "at least two columns"),
        (GOOD_ROWS, {"columns": (0, 2)}, "asks for column 2"),
        (GOOD_ROWS, {"columns": (0,)}, "two or three distinct"),
        (GOOD_ROWS, {"columns": (1, 1)}, "two or three distinct"),
        (GOOD_ROWS, {"columns": (0, -1)}, "two or three distinct"),
        ("1e14 0.1\n2e14 nan\n3e14 1\n4e14 0.1\n", {}, "not finite"),
        ("1e14 0.1\n2e14 inf\n3e14 1\n4e14 0.1\n", {}, "not finite"),
        ("1e14 0.1\n2e14 1\n3e14 0.1\n", {}, "at least 4 rows"),
        ("0 0.1\n2e14 1\n3e14 1\n4e14 0.1\n", {}, "frequency column must be positive"),
        ("1e14 0.1\n2e14 1\n2e14 1\n4e14 0.1\n", {}, "repeats a value"),
        ("1e14 0\n2e14 0\n3e14 0\n4e14 0\n", {}, "no positive value"),
        ("1e14 -0.2\n2e14 1\n3e14 1\n4e14 0.1\n", {}, "more negative than a tenth"),
        (GOOD_ROWS, {"axis_unit": "nm"}, "not a unit of frequency"),
        (GOOD_ROWS, {"axis": "wavelength", "axis_unit": "THz"}, "not a unit of wave"),
        (GOOD_ROWS, {"floor": 1.0}, r"floor must be a share .* \[0, 1\)"),
        (GOOD_ROWS, {"floor": -1e-3}, "floor must be a share"),
        (GOOD_ROWS, {"floor": math.nan}, "floor must be a share"),
        (GOOD_ROWS, {"energy": 0.0}, "energy must be a finite positive number"),
    ],
)
def test_a_bad_file_or_keyword_is_refused_by_name(text, keywords, message):
    arguments = {"energy": ENERGY} | keywords
    with pytest.raises(PulseError, match=message):
        load_spectrum(io.StringIO(text), **arguments)


def test_a_missing_file_is_refused_by_name(tmp_path):
    with pytest.raises(PulseError, match=r"no spectrum file at .*absent\.txt"):
        load_spectrum(tmp_path / "absent.txt", energy=ENERGY)


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"axis": "time"}, "axis must be 'frequency' or 'wavelength'"),
        ({"phase_unit": "grad"}, "phase_unit must be 'rad' or 'deg'"),
        ({"phase_convention": "physics"}, "phase_convention must be 'optics' or 'fft'"),
        ({"columns": (0, 1.0)}, "two or three distinct"),
    ],
)
def test_a_keyword_outside_its_values_is_refused_by_name(keywords, message):
    # Production code is not type-checked at runtime, so the reader's own refusal is
    # what a caller meets; the test session's type hook is stepped past to reach it.
    with pytest.raises(PulseError, match=message):
        unchecked(load_spectrum)(io.StringIO(GOOD_ROWS), energy=ENERGY, **keywords)
