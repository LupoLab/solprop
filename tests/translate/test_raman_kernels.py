# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sampling the Raman kernel onto the doubled grid, and caching its spectrum.

Named for the kernels rather than ``test_raman`` because ``tests/`` has no packages:
module basenames must be unique across the whole tree, and
``tests/nonlinear/test_raman.py`` has the name (``docs/dev/testing.md`` section 6).
"""

import logging
import math

import numpy as np
import pytest

from bounds import EPS, assert_within, fft_bound
from former_raman import FORMER_H2, FORMER_N2, former_response
from solprop.constants import AMAGAT, EPS0
from solprop.exceptions import TranslationError
from solprop.grid import Collar, EnvelopeGrid, FieldGrid
from solprop.nonlinear.raman import (
    DampedOscillator,
    MolecularOptions,
    RamanResponse,
    normalised_damped_oscillator,
    raman_drive,
    response_for,
    undamped_dephasing,
)
from solprop.profiles import Constant, FromCallable
from solprop.translate.raman import (
    MAX_DECAY_EXPONENT,
    WINDOW_RESIDUAL_TOLERANCE,
    RamanKernels,
    _flat_samples,
    build_raman_kernels,
    decay_rate_profiles,
    kernel_spectrum,
    kernel_taper,
    sample_kernel,
    undamped_kernels,
)

GRID = {
    "length": 0.1,
    "reference_wavelength": 800e-9,
    "wavelength_limits": (400e-9, 2e-6),
    "trange": 2e-12,
}
DENSITY = 0.917 * AMAGAT

# A coherence far shorter than any window here, for the cases that must not warn about
# truncation: 50 fs against an untapered span of picoseconds.
SHORT = RamanResponse(
    models=(normalised_damped_oscillator(1e-30, 2.0 * math.pi * 40e12, 5e-14),)
)


@pytest.fixture
def grid():
    return EnvelopeGrid.from_limits(**GRID)


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(**GRID)


def delays_of(grid):
    """The delay axis the kernel is sampled on."""
    return np.arange(grid.n_to) * grid.dt_fine


# ------------------------------------------------------------------------- the taper


def test_the_taper_is_one_at_the_start_and_zero_at_the_end(grid):
    taper = kernel_taper(grid)
    assert taper.shape == (grid.n_to,)
    assert float(taper[0]) == 1.0
    assert float(taper[-1]) == 0.0


def test_the_taper_is_exactly_one_over_the_flat_region(grid):
    # Exactly, not nearly: an explicit one, so the kernel there is the model's own.
    taper = kernel_taper(grid)
    assert np.all(taper[: _flat_samples(grid)] == 1.0)


def test_the_taper_falls_monotonically_over_the_collar(grid):
    taper = kernel_taper(grid)
    falling = taper[_flat_samples(grid) :]
    assert np.all(np.diff(falling) <= 0.0)
    assert 0.0 < float(falling[len(falling) // 2]) < 1.0


def test_the_taper_edge_has_the_collars_own_width(grid):
    # The taper follows the grid's own boundary rather than a fixed fraction of the
    # grid, so there is one statement of how much of the window is collar.
    taper = kernel_taper(grid)
    delays = delays_of(grid)
    span = float(delays[-1])
    started = delays[taper < 1.0][0]
    assert span - started == pytest.approx(grid.t_collar.width_high, rel=1e-3)


def test_a_collar_wider_than_the_response_span_is_refused(grid):
    span = (grid.n_to - 1) * grid.dt_fine
    wide = Collar(-2.0 * span, -1.5 * span, 1.5 * span, 3.0 * span)
    broken = FieldGrid(
        length=grid.length,
        reference_wavelength=grid.reference_wavelength,
        t=grid.t,
        omega=grid.omega,
        t_fine=grid.t_fine,
        omega_fine=grid.omega_fine,
        dt=grid.dt,
        dt_fine=grid.dt_fine,
        ffac=grid.ffac,
        omega_collar=grid.omega_collar,
        t_collar=wide,
    )
    with pytest.raises(TranslationError, match="not a taper inside a response span"):
        kernel_taper(broken)


# ------------------------------------------------------------------ the sampled kernel


def test_the_kernel_is_zero_over_the_padding(grid):
    kernel = sample_kernel(SHORT, grid, DENSITY)
    assert kernel.shape == (2 * grid.n_to,)
    # Exactly zero: the padding is what makes the transform product a *linear*
    # convolution, and any leakage into it would wrap the response onto the drive.
    assert np.all(kernel[grid.n_to :] == 0.0)


def test_the_kernel_is_causal(grid):
    # h(0) = 0 for every model here: a medium cannot respond in the instant it is
    # driven, and the first sample of the kernel is that instant.
    assert float(sample_kernel(SHORT, grid, DENSITY)[0]) == 0.0


def test_the_kernel_equals_the_model_over_the_flat_region(grid):
    kernel = sample_kernel(former_response(FORMER_H2), grid, DENSITY)
    flat = _flat_samples(grid)
    expected = former_response(FORMER_H2).impulse(delays_of(grid)[:flat], DENSITY)
    # The taper is exactly one there, the fraction multiplies the sum of the same
    # terms; a reordered sum of a few damped oscillations moves each sample by a few
    # roundings of the largest term, held to 16 eps of the kernel's peak.
    np.testing.assert_allclose(
        kernel[:flat], expected, rtol=0, atol=16 * EPS * float(np.max(np.abs(expected)))
    )


def test_the_kernel_is_attenuated_over_the_tapered_region(grid):
    kernel = sample_kernel(former_response(FORMER_H2), grid, DENSITY)
    flat = _flat_samples(grid)
    untapered = former_response(FORMER_H2).impulse(delays_of(grid), DENSITY)
    assert np.max(np.abs(kernel[flat:])) < np.max(np.abs(untapered[flat:]))
    assert float(kernel[grid.n_to - 1]) == 0.0


def test_each_term_is_damped_by_its_own_dephasing(grid):
    # The whole point of the per-term kernel: hydrogen's rotational coherence lives
    # 2.9 ns and its vibrational one 0.83 ns at this density, so damping the sum by
    # either one alone would be wrong for the other.
    response = former_response(FORMER_H2)
    rotation, vibration = response.models
    rotational_time = rotation.dephasing.dephasing_time(DENSITY)
    vibrational_time = vibration.dephasing.dephasing_time(DENSITY)
    assert rotational_time > 2.0 * vibrational_time
    delays = delays_of(grid)
    rows = undamped_kernels(response, grid)
    expected = rows[0, : grid.n_to] * rotation.dephasing.damping(delays, DENSITY)
    expected = expected + rows[1, : grid.n_to] * vibration.dephasing.damping(
        delays, DENSITY
    )
    kernel = sample_kernel(response, grid, DENSITY)
    # Two damped rows summed in another arrangement: 16 eps of the kernel's peak.
    np.testing.assert_allclose(
        kernel[: grid.n_to],
        expected,
        rtol=0,
        atol=16 * EPS * float(np.max(np.abs(expected))),
    )


def test_the_undamped_rows_are_one_per_model(grid):
    rows = undamped_kernels(former_response(FORMER_H2), grid)
    assert rows.shape == (2, 2 * grid.n_to)
    assert np.all(rows[:, grid.n_to :] == 0.0)


def test_the_undamped_rows_carry_no_dephasing(grid):
    # The rows must be the density-independent half, or caching them would be wrong.
    first = undamped_kernels(former_response(FORMER_H2), grid)
    second = undamped_kernels(former_response(FORMER_H2), grid)
    assert np.array_equal(first, second)


def test_the_fraction_scales_the_sampled_kernel(grid):
    whole = sample_kernel(response_for("SiO2"), grid, DENSITY)
    part = sample_kernel(response_for("SiO2", fraction=0.18), grid, DENSITY)
    assert np.allclose(part, 0.18 * whole, rtol=2e-16, atol=0.0)


# --------------------------------------------------------------- the kernel spectrum


def test_the_spectrum_has_the_one_sided_length_of_the_doubled_grid(grid):
    spectrum = kernel_spectrum(sample_kernel(SHORT, grid, DENSITY), grid.dt_fine)
    assert spectrum.shape == (grid.n_to + 1,)
    assert np.iscomplexobj(spectrum)


def backend_polarisation(response, field, density, dt, kind):
    """The polarisation a backend would form from the kernel spectrum."""
    n_to = field.shape[1]
    kernel = np.zeros(2 * n_to)
    kernel[:n_to] = response.impulse(np.arange(n_to) * dt, density)
    drive = np.zeros(2 * n_to)
    drive[:n_to] = raman_drive(field, kind=kind, thg=response.thg)
    spectrum = kernel_spectrum(kernel, dt)
    convolved = np.fft.irfft(spectrum * np.fft.rfft(drive), n=2 * n_to)[:n_to]
    return (EPS0 * density) * field * convolved[None, :]


@pytest.mark.parametrize("material", ["SiO2", "N2"])
def test_the_spectrum_normalisation_reproduces_the_reference(material):
    # The dt that turns the discrete sum into an integral is folded into the kernel
    # spectrum here and applied on the other side of the product in the reference; the
    # two are the same convolution, each a round trip of forward and inverse transforms
    # of length 2 n_to, so they agree normwise within twice the round-trip bound
    # B_fft(2 n_to) of accuracy.md section 7.
    response = (
        response_for("SiO2") if material == "SiO2" else former_response(FORMER_N2)
    )
    n_to, dt = 512, 4e-15
    t = (np.arange(n_to) - n_to // 2) * dt
    field = (1e9 * np.exp(-((t / 4e-14) ** 2)))[None, :]
    expected = response.evaluate(field, DENSITY, kind="field", dt=dt)
    assert_within(
        backend_polarisation(response, field, DENSITY, dt, "field"),
        expected,
        2 * fft_bound(2 * n_to),
    )


def test_the_spectrum_is_the_transform_of_the_kernel_times_dt(grid):
    kernel = sample_kernel(SHORT, grid, DENSITY)
    # One forward transform and a scale, possibly by another FFT library: within the
    # round-trip bound B_fft (accuracy.md section 7), normwise.
    assert_within(
        kernel_spectrum(kernel, grid.dt_fine),
        np.fft.rfft(kernel) * grid.dt_fine,
        fft_bound(kernel.size),
    )


def test_the_convolution_scaling_is_the_direct_sum():
    # The whole normalisation, against an explicit O(n^2) sum: the transform pair's
    # 1/N and the folded dt together have to reproduce sum_j h[i-j] d[j] dt.
    rng = np.random.default_rng(20260905)
    n_to, dt = 48, 3e-15
    kernel = np.zeros(2 * n_to)
    kernel[1:n_to] = rng.normal(size=n_to - 1)
    drive = np.zeros(2 * n_to)
    drive[:n_to] = rng.normal(size=n_to)
    spectrum = kernel_spectrum(kernel, dt)
    convolved = np.fft.irfft(spectrum * np.fft.rfft(drive), n=2 * n_to)[:n_to]
    direct = np.array(
        [float(np.dot(kernel[: i + 1], drive[i::-1])) * dt for i in range(n_to)]
    )
    # A 96-point transform pair against a 48-term dot product: a few times
    # log2(96) * eps of the largest partial sum, which is the peak of the result.
    assert np.max(np.abs(convolved - direct)) < 1e-14 * np.max(np.abs(direct))


@pytest.mark.parametrize("bad", [0.0, -1e-15, math.inf, math.nan])
def test_a_bad_sample_spacing_is_refused_by_the_spectrum(bad):
    with pytest.raises(TranslationError, match="positive, finite sample spacing"):
        kernel_spectrum(np.zeros(8), bad)


# A two-dimensional kernel is refused by the jaxtyping annotation before the guard
# below is reached, so only the shapes a real one-dimensional array can take are here.
@pytest.mark.parametrize("bad", [np.zeros(7), np.zeros(8) + 0j])
def test_a_kernel_that_is_not_real_and_even_is_refused(bad):
    with pytest.raises(TranslationError, match="real array of even length"):
        kernel_spectrum(bad, 1e-15)


# ------------------------------------------------------------------------ the cache


def test_the_spectrum_is_transformed_once_per_density(grid):
    kernels = build_raman_kernels(SHORT, grid)
    assert kernels.recomputations == 0
    first = kernels.spectrum(DENSITY)
    assert kernels.recomputations == 1
    again = kernels.spectrum(DENSITY)
    # The same object, not merely an equal one: the cache must not be re-transforming.
    assert again is first
    assert kernels.recomputations == 1


def test_a_new_density_recomputes_the_spectrum(grid):
    kernels = build_raman_kernels(SHORT, grid)
    first = kernels.spectrum(DENSITY)
    second = kernels.spectrum(2.0 * DENSITY)
    assert kernels.recomputations == 2
    assert second is not first
    # Only the memo's single entry is kept, so going back re-transforms.
    third = kernels.spectrum(DENSITY)
    assert kernels.recomputations == 3
    assert np.array_equal(third, first)


def test_the_cached_spectrum_is_read_only(grid):
    spectrum = build_raman_kernels(SHORT, grid).spectrum(DENSITY)
    with pytest.raises(ValueError, match="read-only"):
        spectrum[0] = 0.0


def test_the_cached_spectrum_is_the_sampled_one(grid):
    kernels = build_raman_kernels(SHORT, grid)
    direct = kernel_spectrum(sample_kernel(SHORT, grid, DENSITY), grid.dt_fine)
    assert np.array_equal(kernels.spectrum(DENSITY), direct)


def test_the_kernel_object_reports_its_terms_and_dephasings(grid):
    response = former_response(FORMER_H2)
    kernels = build_raman_kernels(response, grid)
    assert kernels.n_terms == 2
    assert kernels.dephasings == tuple(m.dephasing for m in response.models)


def test_a_kernel_that_does_not_match_its_response_is_refused(grid):
    with pytest.raises(TranslationError, match="rows but the response has"):
        RamanKernels(
            undamped=np.zeros((3, 2 * grid.n_to)),
            response=response_for("SiO2"),
            grid=grid,
        )


def test_a_kernel_that_does_not_match_its_grid_is_refused(grid):
    with pytest.raises(TranslationError, match="samples long but the doubled"):
        RamanKernels(
            undamped=np.zeros((1, 16)), response=response_for("SiO2"), grid=grid
        )


# ------------------------------------------------------------ the truncation note

LOGGER = "solprop.translate.raman"
NOTE = "longer than the untapered span"


def truncation_notes(caplog):
    """The debug records saying a coherence outlives the window."""
    return [record for record in caplog.records if NOTE in record.getMessage()]


def test_a_coherence_the_window_cannot_hold_is_noted(caplog, grid):
    # Hydrogen's rotational coherence lives nanoseconds and the untapered span here is
    # picoseconds. That is not an error -- the drive is zero outside the window, so the
    # polarisation inside it is exact -- but the kernel spectrum resolves the line only
    # to the window's width, and the debug log says so.
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        sample_kernel(former_response(FORMER_H2), grid, DENSITY)
    assert truncation_notes(caplog)


def test_the_note_names_the_dephasing_time_and_the_span_it_needs(caplog, grid):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        sample_kernel(former_response(FORMER_N2), grid, DENSITY)
    message = truncation_notes(caplog)[0].getMessage()
    assert "dephasing time" in message
    assert "would hold it" in message


@pytest.mark.parametrize("case", ["short", "glass", "vacuum"])
def test_a_coherence_the_window_holds_is_not_noted(case, caplog, grid):
    # 50 fs against 2.8 ps has decayed to exp(-56) before the taper; silica carries its
    # decay inside the model and has decayed to about 1e-6, so judging by the dephasing
    # time alone would note every glass; at zero density the polarisation is zero.
    response, density = {
        "short": (SHORT, DENSITY),
        "glass": (response_for("SiO2"), DENSITY),
        "vacuum": (former_response(FORMER_N2), 0.0),
    }[case]
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        sample_kernel(response, grid, density)
    assert not truncation_notes(caplog)


def test_the_note_is_logged_once_per_density_not_once_per_call(caplog, grid):
    kernels = build_raman_kernels(former_response(FORMER_N2), grid)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        kernels.spectrum(DENSITY)
        first = len(truncation_notes(caplog))
        kernels.spectrum(DENSITY)
        second = len(truncation_notes(caplog))
        kernels.spectrum(2.0 * DENSITY)
        third = len(truncation_notes(caplog))
    # The second call hits the memo and adds nothing; the third is a new density.
    assert first > 0
    assert second == first
    assert third > second


def test_the_residual_tolerance_is_a_small_fraction():
    assert 0.0 < WINDOW_RESIDUAL_TOLERANCE < 1e-2


# ------------------------------------------------------------------ both grid kinds


@pytest.mark.parametrize("builder", [FieldGrid, EnvelopeGrid])
def test_the_kernel_does_not_depend_on_the_grid_kind(builder):
    # The kernel is a property of the medium and the time axis, not of the spectral
    # layout: only the drive differs between the two grids.
    built = builder.from_limits(**GRID)
    kernel = sample_kernel(SHORT, built, DENSITY)
    expected = SHORT.impulse(delays_of(built), DENSITY) * kernel_taper(built)
    # The same three factors in a different order: the sampler applies the taper to
    # each term and the fraction to the sum, so the products round differently by one
    # ulp of the peak.
    peak = float(np.max(np.abs(expected)))
    assert peak > 0.0
    assert np.max(np.abs(kernel[: built.n_to] - expected)) < 4e-16 * peak


def test_a_term_with_no_amplitude_is_not_noted(caplog, grid):
    # A zero kernel has no residual to measure, and dividing by its peak would be a
    # division by zero rather than a truncation.
    silent = RamanResponse(
        models=(DampedOscillator(0.0, 2.0 * math.pi * 40e12, undamped_dephasing()),)
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        kernel = sample_kernel(silent, grid, DENSITY)
    assert np.all(kernel == 0.0)
    assert not truncation_notes(caplog)


def test_building_the_kernel_logs_its_shape(caplog, grid):
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.raman"):
        build_raman_kernels(former_response(FORMER_H2), grid)
    assert "Raman kernel: 2 term(s)" in caplog.text
    assert "taper over the last" in caplog.text


# --------------------------------------------------------------- decay-rate paths


def test_the_decay_rate_paths_follow_the_density(grid):
    response = former_response(FORMER_H2)
    density = FromCallable(lambda z: DENSITY * (1.0 + z), (0.0, 1.0), np.array([0.5]))
    paths = decay_rate_profiles(response, density, grid)
    assert len(paths) == 2
    z = np.array([0.0, 0.25, 0.5, 1.0])
    for path, model in zip(paths, response.models, strict=True):
        expected = [model.dephasing.decay_rate(float(rho)) for rho in density(z)]
        # The path evaluates the same dephasing at the same densities, vectorised
        # where the reference is scalar: a few roundings, 4 eps.
        np.testing.assert_allclose(path(z), np.asarray(expected), rtol=4 * EPS, atol=0)
        # The rate can kink only where the density does.
        assert np.array_equal(path.breakpoints, density.breakpoints)
        assert path.z_span == density.z_span


def test_a_collapsed_coherence_gets_a_finite_capped_rate(grid):
    # A Dicke-narrowed line has an infinite decay rate at zero density; the path
    # carries the finite cap instead, which damps the term to nothing within one
    # sample -- and the density multiplies the polarisation by zero there anyway.
    response = former_response(FORMER_H2, MolecularOptions(vibration=False))
    density = FromCallable(lambda z: DENSITY * z, (0.0, 1.0))
    (path,) = decay_rate_profiles(response, density, grid)
    at_vacuum = float(path(np.array([0.0]))[0])
    assert math.isfinite(at_vacuum)
    assert at_vacuum == MAX_DECAY_EXPONENT / grid.dt_fine
    assert math.exp(-at_vacuum * grid.dt_fine) < 1e-300
    assert float(path(np.array([1.0]))[0]) < at_vacuum


def test_a_glass_has_a_zero_decay_rate_everywhere(grid):
    (path,) = decay_rate_profiles(response_for("SiO2"), Constant(1.0), grid)
    assert np.all(path(np.linspace(0.0, 1.0, 5)) == 0.0)


def test_the_kernel_object_hands_out_the_same_paths(grid):
    density = Constant(DENSITY)
    kernels = build_raman_kernels(former_response(FORMER_N2), grid)
    z = np.array([0.0, 0.3])
    direct = decay_rate_profiles(kernels.response, density, grid)
    for one, other in zip(kernels.decay_rates(density), direct, strict=True):
        assert np.array_equal(one(z), other(z))
