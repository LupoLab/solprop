# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop import fourier, stats
from solprop.exceptions import ModeError, OutputError, ProcessingError
from solprop.output import MemoryOutput
from solprop.processing import axes, fields

ACCESSORS = {
    "spectrum": lambda out: fields.get_spectrum(out, plane=-1)[1],
    "spectrum_fft": lambda out: fields.get_spectrum(out, convention="fft")[1],
    "density": lambda out: fields.get_spectral_density(out, "wavelength")[1],
    "field": lambda out: fields.get_field(out, plane=-1, oversampling=2)[1],
    "power": lambda out: fields.get_power(out, z=0.05)[1],
    "phase": lambda out: fields.get_phase(out, plane=0)[1],
}


@pytest.mark.parametrize("name", sorted(ACCESSORS))
def test_memory_and_file_give_identical_arrays(handlers, name):
    memory, file_handler = handlers
    # The two handlers store a run bitwise identically, so an accessor reading either
    # must return the same bits, not merely the same numbers.
    assert np.array_equal(ACCESSORS[name](memory), ACCESSORS[name](file_handler))


def test_spectrum_modulus_squared_is_the_spectral_density(output, grid):
    _, spectrum = fields.get_spectrum(output, plane=-1)
    stored = output.fields("fft")[-1]
    expected = fourier.spectral_density(stored, grid.dt, kind=grid.kind)
    keep = axes._keep_index(grid, "active")
    # The accessor scales by the square root of the density prefactor, so the two agree
    # to the rounding of that square root: a few ulp of the largest value.
    assert np.allclose(
        np.abs(spectrum) ** 2, expected[0, 0, keep], rtol=1e-14, atol=0.0
    )


def test_the_convention_switch_is_the_exact_conjugation(output, grid):
    _, optics = fields.get_spectrum(output, plane=-1, convention="optics")
    _, internal = fields.get_spectrum(output, plane=-1, convention="fft")
    keep = axes._keep_index(grid, "active")
    stored = output["field"][-1, 0, 0]
    norm = fields._fft_norm(grid)
    # The file holds the optics convention, so one accessor is the stored array scaled
    # and the other is fourier's boundary helper -- the conjugation and the move of the
    # phase origin -- applied to it: both bit for bit.
    assert np.array_equal(optics, stored[keep] * norm)
    expected = fourier.spectrum_to_internal(stored, kind=grid.kind)[keep]
    assert np.array_equal(internal, expected * norm)


@pytest.mark.parametrize("axis", ["omega", "f", "wavelength"])
def test_the_spectral_density_integrates_to_the_energy(output, grid, axis):
    _, density = fields.get_spectral_density(output, axis, plane=0, modes="sum")
    omega = grid.omega[axes._keep_index(grid, "active")]
    _, jacobian = axes.spectral_axis(omega, axis)
    weights = grid.d_omega / jacobian
    if axis == "wavelength":
        weights = weights[::-1]
    total = float(np.sum(density * weights))
    expected = float(np.sum(stats.energy(grid, output.fields("fft")[0])))
    # The change of variable is exact bin by bin, so the only differences are the energy
    # outside the active band, which the collar has driven to zero, and the order of the
    # sum: both at the level of round-off on a reduction of this length.
    assert total == pytest.approx(expected, rel=1e-12)


def test_the_wavelength_axis_is_ascending(output):
    wavelength, _ = fields.get_spectral_density(output, "wavelength", plane=0)
    assert np.all(np.diff(wavelength) > 0.0)


def test_a_wavelength_axis_needs_a_band_without_zero_frequency(synthetic):
    output = synthetic.build_run("field")
    with pytest.raises(ProcessingError, match="not positive"):
        fields.get_spectral_density(output, "wavelength", plane=0, band="all")


def test_the_band_keyword_chooses_how_many_bins_come_back(output, grid):
    active, _ = fields.get_spectrum(output, plane=0)
    every, _ = fields.get_spectrum(output, plane=0, band="all")
    assert active.size == int(grid.active_mask.sum())
    assert every.size == grid.n_omega


def test_a_spectral_range_crops_the_density(output):
    x, density = fields.get_spectral_density(
        output, "wavelength", plane=0, spec_range=(700e-9, 900e-9)
    )
    assert x.min() >= 700e-9
    assert x.max() <= 900e-9
    assert density.shape == x.shape


@pytest.mark.parametrize("resolution", [None, 5e12])
def test_a_spectral_range_that_selects_nothing_is_refused_naming_its_axis(
    output, resolution
):
    """Metres on the frequency axis, the obvious slip, name the axis and its units."""
    with pytest.raises(ProcessingError, match=r"selects nothing on the 'f' axis.* Hz"):
        fields.get_spectral_density(
            output, "f", plane=0, spec_range=(600e-9, 1.1e-6), resolution=resolution
        )


def test_the_time_domain_convention_switch_is_a_conjugation(output):
    _, optics = fields.get_field(output, plane=0, convention="optics")
    _, internal = fields.get_field(output, plane=0, convention="fft")
    # For a time-domain array the boundary is the conjugation alone: no origin shift.
    assert np.array_equal(optics, np.conj(internal))


def test_a_resolution_smooths_a_frequency_axis_too(output):
    sharp_x, sharp = fields.get_spectral_density(output, "f", plane=0)
    smooth_x, smooth = fields.get_spectral_density(
        output, "f", plane=0, resolution=5e12
    )
    assert smooth.max() < sharp.max()
    assert np.all(np.diff(smooth_x) > 0.0)
    assert sharp_x.size != smooth_x.size


def test_a_resolution_smooths_the_density(output):
    _, sharp = fields.get_spectral_density(output, "wavelength", plane=0)
    smooth_x, smooth = fields.get_spectral_density(
        output, "wavelength", plane=0, resolution=30e-9
    )
    assert smooth.max() < sharp.max()
    assert smooth_x.size == smooth.size


def test_every_plane_comes_back_when_none_is_selected(output, synthetic):
    _, spectra = fields.get_spectrum(output)
    assert spectra.shape[0] == synthetic.n_z


def test_a_position_selects_the_nearest_plane(output):
    assert fields.nearest_plane(output, 0.049) == 2
    by_z = fields.get_spectrum(output, z=0.049)[1]
    assert np.array_equal(by_z, fields.get_spectrum(output, plane=2)[1])


@pytest.mark.parametrize(
    ("modes", "expected"), [(None, (2, 2)), (0, (2,)), ([0, 1], (2, 2))]
)
def test_mode_selection_shapes_the_spectrum(synthetic, modes, expected):
    output = synthetic.build_run("envelope", n_modes=2, n_pol=2)
    _, spectrum = fields.get_spectrum(output, plane=0, modes=modes)
    assert spectrum.shape[:-1] == expected


def test_a_modal_sum_of_complex_amplitudes_is_refused(synthetic):
    output = synthetic.build_run("envelope", n_modes=2)
    with pytest.raises(ModeError, match="meaningless"):
        fields.get_spectrum(output, plane=0, modes="sum")
    with pytest.raises(ModeError, match="meaningless"):
        fields.get_phase(output, plane=0, modes="sum")


def test_a_modal_sum_adds_the_densities(synthetic):
    output = synthetic.build_run("envelope", n_modes=2)
    _, each = fields.get_spectral_density(output, "omega", plane=0)
    _, total = fields.get_spectral_density(output, "omega", plane=0, modes="sum")
    assert np.allclose(total, each.sum(axis=0), rtol=1e-15)


def test_the_real_field_is_the_real_part_of_the_analytic_signal(output):
    t, analytic = fields.get_field(output, plane=0, kind="analytic")
    t_real, real = fields.get_field(output, plane=0, kind="real")
    assert np.array_equal(t, t_real)
    # The optics convention conjugates, which leaves the real part alone.
    assert np.allclose(real, np.real(analytic), rtol=0.0, atol=1e-30)


def test_the_envelope_is_the_analytic_signal_without_its_carrier(output, grid):
    t, analytic = fields.get_field(output, plane=0, kind="analytic")
    _, envelope = fields.get_field(output, plane=0, kind="envelope")
    # In the optics convention the carrier is exp(+i omega0 t) (conventions.md 2).
    assert np.allclose(envelope, analytic * np.exp(1j * grid.omega0 * t), rtol=1e-12)


def test_the_power_is_the_modulus_squared_of_the_analytic_signal(output):
    _, analytic = fields.get_field(output, plane=0, kind="analytic")
    _, power = fields.get_power(output, plane=0)
    assert np.allclose(power, np.abs(analytic) ** 2, rtol=1e-14)


def test_oversampling_refines_the_axis_and_keeps_the_energy(output, grid):
    t_one, power_one = fields.get_power(output, plane=0)
    t_four, power_four = fields.get_power(output, plane=0, oversampling=4)
    assert t_four.size == 4 * t_one.size
    assert np.isclose(t_four[1] - t_four[0], (t_one[1] - t_one[0]) / 4, rtol=1e-14)
    energy_one = float(np.sum(power_one)) * float(t_one[1] - t_one[0])
    energy_four = float(np.sum(power_four)) * float(t_four[1] - t_four[0])
    # Zero-padding in frequency is exact interpolation of a band-limited trace, so the
    # two Riemann sums differ only by the quadrature error of the coarser one.
    assert energy_four == pytest.approx(energy_one, rel=1e-12)


def test_the_oversampled_axis_stays_centred(output):
    t, _ = fields.get_power(output, plane=0, oversampling=2)
    assert np.isclose(t[t.size // 2], 0.0, atol=1e-30)


def test_a_bandpass_removes_what_is_outside_it(output):
    _, full = fields.get_spectral_density(output, "wavelength", plane=0)
    _, power = fields.get_power(output, plane=0, bandpass=(1.4e-6, 1.9e-6))
    _, unfiltered = fields.get_power(output, plane=0)
    # The pulse is centred at 800 nm, so a window in the infrared keeps almost nothing.
    assert power.max() < 1e-6 * unfiltered.max()
    assert full.max() > 0.0


def test_a_bandpass_keeps_what_is_inside_it(output):
    _, inside = fields.get_power(output, plane=0, bandpass=(600e-9, 1100e-9))
    _, unfiltered = fields.get_power(output, plane=0)
    # The 30 fs pulse is far narrower than a 500 nm window, so the window is a no-op to
    # the accuracy of the Planck taper at the pulse's own spectral edges.
    assert np.allclose(inside, unfiltered, rtol=1e-6, atol=1e-6 * unfiltered.max())


def test_the_spectral_phase_of_a_dispersed_pulse_is_the_expected_parabola(
    output, grid, synthetic
):
    omega, first = fields.get_phase(output, plane=0)
    _, last = fields.get_phase(output, plane=-1)
    inband = np.abs(omega - grid.omega0) < 0.15 * grid.omega0
    added = (last - first)[inband]
    expected = (
        0.5 * synthetic.beta2 * (omega[inband] - grid.omega0) ** 2 * synthetic.length
    )
    # The optics convention makes positive group-velocity dispersion a positive
    # parabola; unwrapping fixes the branch only up to a constant, which is removed.
    added = added - added[added.size // 2]
    expected = expected - expected[expected.size // 2]
    assert np.allclose(added, expected, rtol=1e-6, atol=1e-6)


def test_get_field_refuses_an_unknown_representation():
    with pytest.raises(ProcessingError, match="kind must be one of"):
        axes._check_choice("power", fields.FIELD_KINDS, "kind")


def test_get_field_refuses_an_oversampling_below_one(output):
    with pytest.raises(ProcessingError, match="at least 1"):
        fields.get_field(output, plane=0, oversampling=0)


def test_an_unopened_handler_has_no_grid():
    with pytest.raises(OutputError, match="no metadata"):
        fields.get_spectrum(MemoryOutput())
