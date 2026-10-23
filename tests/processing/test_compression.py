# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compressors: recovery of known chirps, global optima, honest failure, plots."""

import logging
import math
from contextlib import nullcontext
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.optics import ChirpedMirror, Material, TaylorPhase, stored_factor
from solprop.output import MemoryOutput
from solprop.physdata import calcium_fluoride
from solprop.plotting import time_1d
from solprop.processing import compression, get_field, get_power, time
from solprop.processing._examples import (
    example_general_output,
    example_grid,
    example_output,
)
from solprop.processing.axes import grid_of
from solprop.pulses import GaussianPulse

WAVELENGTH = 800e-9
OMEGA0 = 2.0 * math.pi * C / WAVELENGTH
FWHM = 30e-15
# 230 nm to 9.7 um: wider than the example grid's band, so no range warning.
WINDOW = calcium_fluoride(model="malitson_1963/sellmeier")


def gaussian(kind="envelope", phase=(), wavelength=WAVELENGTH, energy=1e-6, fwhm=FWHM):
    grid = example_grid(kind)
    pulse = GaussianPulse(wavelength=wavelength, fwhm=fwhm, energy=energy, phase=phase)
    return grid, pulse.build(grid)


def phase_only_mirror(gdd=-60e-30):
    wavelength = np.linspace(600e-9, 1000e-9, 801)
    return ChirpedMirror.from_data(
        wavelength,
        None,
        gdd=np.full(wavelength.size, gdd),
        reference_wavelength=WAVELENGTH,
        band=(650e-9, 950e-9),
    )


# --------------------------------------------------------------------- Taylor phase


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_a_known_gdd_is_recovered_with_the_opposite_sign(kind):
    gdd = 900e-30
    grid, stored = gaussian(kind, phase=(0.0, 0.0, gdd))
    found = compression.compress_taylor(grid=grid, spectrum=stored)
    assert found.success, found.message
    assert found.kind == "taylor"
    assert found.coefficients[:2] == (0.0, 0.0)
    # The refinement stops at 1e-9 of its bracket, but the peak of a pulse is quadratic
    # in the residual chirp, so the objective stops resolving the coefficient at the
    # square root of double precision: 1e-7 or so of its value.
    assert found.coefficients[2] == pytest.approx(-gdd, rel=1e-6)
    assert found.quality == pytest.approx(1.0, abs=1e-8)
    assert found.fwhm_after == pytest.approx(found.fwhm_limit, rel=1e-6)
    assert found.fwhm_limit == pytest.approx(FWHM, rel=2e-4)
    assert found.fwhm_before > 2.0 * found.fwhm_after
    assert found.peak_power_before < found.peak_power_after <= found.peak_power_limit
    assert found.evaluations > 0


@pytest.mark.parametrize("bound_factor", [5.0, 7.0, 11.0])
def test_the_refinement_refines_whatever_the_scan_grid_happens_to_be(bound_factor):
    # With the default factor of 8 the 65-point scan has a sample exactly on the answer
    # for a pure-GDD Gaussian, which once hid a refinement that never ran: SciPy's
    # `xatol` is absolute, 1e-5, and a GDD bracket is 1e-27 s^2 wide. At these factors
    # the nearest sample is 3 to 9 per cent away, so only a working refinement passes.
    gdd = 900e-30
    grid, stored = gaussian(phase=(0.0, 0.0, gdd))
    found = compression.compress_taylor(
        grid=grid, spectrum=stored, bound_factor=bound_factor
    )
    assert found.success, found.message
    assert found.coefficients[2] == pytest.approx(-gdd, rel=1e-6)
    assert found.quality == pytest.approx(1.0, abs=1e-10)


def test_a_thin_insertion_is_refined_below_the_scan_spacing():
    # A wedge scanned over +-100 um: the scan's spacing is 3 um and SciPy's default
    # absolute tolerance, 10 um, would stop the refinement before it began.
    grid, clean = gaussian(fwhm=10e-15)
    stored = clean * stored_factor(grid, Material(WINDOW, 37e-6, WAVELENGTH))
    found = compression.compress_material(
        grid=grid, spectrum=stored, medium=WINDOW, thickness_range=(-100e-6, 100e-6)
    )
    assert found.success, found.message
    # The objective is flat to 1e-16 within a few nanometres of the optimum.
    assert found.thickness == pytest.approx(-37e-6, abs=5e-9)


def test_a_refinement_that_does_not_improve_on_the_scan_says_so():
    # A kink exactly on a scan sample: the bounded search, which never evaluates its
    # bracket's centre, cannot beat it.
    best, success, message = compression._scan_then_refine(
        lambda x: abs(x) ** 0.25, -1.0, 1.0
    )
    assert best == 0.0
    assert success
    assert "did not improve on the scan" in message


def test_third_order_is_recovered_by_the_seeded_global_search():
    gdd, tod = 700e-30, 25000e-45
    grid, stored = gaussian(phase=(0.0, 0.0, gdd, tod))
    first = compression.compress_taylor(grid=grid, spectrum=stored, order=3, seed=3)
    again = compression.compress_taylor(grid=grid, spectrum=stored, order=3, seed=3)
    other = compression.compress_taylor(grid=grid, spectrum=stored, order=3, seed=11)
    assert first.success, first.message
    assert first.coefficients == again.coefficients
    # Differential evolution is polished by L-BFGS-B, whose default gradient tolerance
    # leaves the coefficients good to about 1e-3 of their values.
    assert first.coefficients[2] == pytest.approx(-gdd, rel=2e-3)
    assert first.coefficients[3] == pytest.approx(-tod, rel=2e-2)
    assert other.coefficients[2] == pytest.approx(first.coefficients[2], rel=2e-3)
    assert first.quality > 1.0 - 1e-5


def test_a_global_search_that_ends_on_its_box_says_which_order(caplog):
    grid, stored = gaussian(phase=(0.0, 0.0, 700e-30, 25000e-45))
    with caplog.at_level(logging.WARNING, logger="solprop.processing.compression"):
        found = compression.compress_taylor(
            grid=grid, spectrum=stored, order=3, bound_factor=0.2
        )
    assert not found.success
    assert "lies on its bound" in found.message
    assert caplog.records


def test_second_order_alone_improves_a_pulse_carrying_third_order_but_says_so():
    grid, stored = gaussian(phase=(0.0, 0.0, 700e-30, 60000e-45))
    found = compression.compress_taylor(grid=grid, spectrum=stored)
    assert found.peak_power_after > found.peak_power_before
    assert found.quality < 0.95


def test_a_transform_limited_pulse_is_left_alone():
    grid, stored = gaussian()
    found = compression.compress_taylor(grid=grid, spectrum=stored)
    assert found.success
    # No chirp to scale the box with: the box falls back to the limit's own width.
    assert abs(found.coefficients[2]) < 1e-30
    assert found.quality == pytest.approx(1.0, abs=1e-10)


def test_an_output_is_compressed_at_its_last_plane_by_default():
    run = example_output()
    found = compression.compress_taylor(run)
    assert found.success, found.message
    assert found.quality > 1.0 - 1e-6
    # The example doubles its duration over its length by pure dispersion.
    assert found.fwhm_before == pytest.approx(2.0 * found.fwhm_limit, rel=2e-2)
    first = compression.compress_taylor(run, plane=0)
    assert abs(first.coefficients[2]) < 1e-3 * abs(found.coefficients[2])


def test_a_multimode_output_is_compressed_on_its_summed_power():
    run = example_output(n_modes=3, n_pol=2)
    together = compression.compress_taylor(run)
    alone = compression.compress_taylor(run, modes=0)
    pair = compression.compress_taylor(run, modes=[0, 1])
    # Every mode of the example carries the same pulse at a different amplitude, so
    # the chirp found is the same and the summed power grows with every mode added.
    assert together.coefficients[2] == pytest.approx(alone.coefficients[2], rel=1e-6)
    assert alone.peak_power_limit < pair.peak_power_limit < together.peak_power_limit
    with pytest.raises(ProcessingError, match="does not select"):
        compression.compress_taylor(run, modes=5)
    with pytest.raises(ProcessingError, match="does not select"):
        compression.compress_taylor(run, modes=[])


def test_a_bandpass_restricts_what_is_compressed():
    grid, stored = gaussian(phase=(0.0, 0.0, 900e-30))
    narrow = compression.compress_taylor(
        grid=grid, spectrum=stored, bandpass=(780e-9, 820e-9)
    )
    whole = compression.compress_taylor(grid=grid, spectrum=stored)
    assert narrow.fwhm_limit > whole.fwhm_limit
    assert narrow.peak_power_limit < whole.peak_power_limit


# ------------------------------------------------------------------------ material


def test_the_thickness_that_was_inserted_is_removed():
    thickness = 12e-3
    grid, clean = gaussian()
    stored = clean * stored_factor(grid, Material(WINDOW, thickness, WAVELENGTH))
    found = compression.compress_material(
        grid=grid, spectrum=stored, medium=WINDOW, thickness_range=(-30e-3, 5e-3)
    )
    assert found.success, found.message
    assert found.kind == "material"
    # As for the Taylor search: a quadratic optimum resolved to 1e-5 of the bracket.
    assert found.thickness == pytest.approx(-thickness, rel=1e-4)
    assert found.quality == pytest.approx(1.0, abs=1e-8)
    assert isinstance(found.transfer, Material)


def test_the_global_optimum_is_found_among_several():
    grid, red = gaussian(wavelength=900e-9, energy=1e-6)
    _, blue = gaussian(wavelength=700e-9, energy=0.5e-6)
    stored = red * stored_factor(
        grid, Material(WINDOW, 40e-3, WAVELENGTH)
    ) + blue * stored_factor(grid, Material(WINDOW, 10e-3, WAVELENGTH))
    found = compression.compress_material(
        grid=grid, spectrum=stored, medium=WINDOW, thickness_range=(-60e-3, 0.0)
    )
    # An independent oracle: a dense scan of the same objective.
    scan = np.linspace(-60e-3, 0.0, 1201)
    peaks = [
        np.max(
            compression._pulse_of(
                None, grid, stored, plane=-1, modes=None, bandpass=None, oversampling=2
            ).power(Material(WINDOW, float(value), WAVELENGTH))
        )
        for value in scan
    ]
    best = float(scan[int(np.argmax(peaks))])
    # The scan's spacing is 0.05 mm.
    assert found.thickness == pytest.approx(best, abs=6e-5)
    # The objective really has more than one optimum: the two colours compress at
    # different insertions, and walk apart in time between them.
    interior = np.array(peaks[1:-1])
    maxima = (interior > np.array(peaks[:-2])) & (interior > np.array(peaks[2:]))
    assert int(np.sum(maxima & (interior > 0.5 * max(peaks)))) > 1


def test_an_optimum_on_a_bound_is_reported_and_logged(caplog):
    grid, clean = gaussian()
    stored = clean * stored_factor(grid, Material(WINDOW, 12e-3, WAVELENGTH))
    with caplog.at_level(logging.WARNING, logger="solprop.processing.compression"):
        found = compression.compress_material(
            grid=grid, spectrum=stored, medium=WINDOW, thickness_range=(-5e-3, 5e-3)
        )
    assert not found.success
    assert "lower bound" in found.message
    assert found.thickness == pytest.approx(-5e-3, rel=1e-3)
    assert any(
        "did not converge cleanly" in record.message for record in caplog.records
    )
    with caplog.at_level(logging.WARNING, logger="solprop.processing.compression"):
        upper = compression.compress_material(
            grid=grid, spectrum=stored, medium=WINDOW, thickness_range=(-40e-3, -30e-3)
        )
    assert "upper bound" in upper.message


def test_a_gas_path_needs_its_pressure():
    grid, stored = gaussian(phase=(0.0, 0.0, -300e-30))
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        found = compression.compress_material(
            grid=grid,
            spectrum=stored,
            medium="Ar",
            pressure=5.0,
            thickness_range=(0.0, 10.0),
            wavelength=WAVELENGTH,
        )
    assert 0.0 < found.thickness < 10.0
    assert found.peak_power_after > found.peak_power_before


# ------------------------------------------------------------------------- mirrors


def test_the_number_of_bounces_that_was_undone_is_found():
    mirror = phase_only_mirror()
    grid, clean = gaussian()
    stored = clean * stored_factor(grid, mirror.bounces(-3))
    found = compression.compress_mirrors(
        grid=grid, spectrum=stored, mirror=mirror, max_reflections=6
    )
    assert found.success, found.message
    assert found.kind == "mirrors"
    assert found.reflections == 3
    assert math.isnan(found.thickness)
    assert [row[0] for row in found.table] == list(range(7))
    qualities = [row[2] for row in found.table]
    assert qualities[3] == max(qualities)
    # The pulse's wings beyond the mirror's band are lost, a few parts in 1e6.
    assert found.quality == pytest.approx(1.0, abs=1e-4)


def test_a_wedge_trims_what_whole_bounces_cannot():
    mirror = phase_only_mirror()
    grid, clean = gaussian()
    chirp = Material(WINDOW, -2.5e-3, WAVELENGTH)
    stored = (
        clean * stored_factor(grid, mirror.bounces(-3)) * stored_factor(grid, chirp)
    )
    found = compression.compress_mirrors(
        grid=grid,
        spectrum=stored,
        mirror=mirror,
        max_reflections=5,
        medium=WINDOW,
        thickness_range=(2e-3, 3e-3),
    )
    assert found.success, found.message
    assert found.reflections == 3
    assert found.thickness == pytest.approx(2.5e-3, rel=1e-3)
    assert found.quality == pytest.approx(1.0, abs=1e-4)


def test_using_every_reflection_allowed_is_not_a_clean_answer(caplog):
    mirror = phase_only_mirror()
    grid, clean = gaussian()
    stored = clean * stored_factor(grid, mirror.bounces(-8))
    with caplog.at_level(logging.WARNING, logger="solprop.processing.compression"):
        found = compression.compress_mirrors(
            grid=grid, spectrum=stored, mirror=mirror, max_reflections=4
        )
    assert found.reflections == 4
    assert not found.success
    assert "all 4 reflections" in found.message


def test_a_shipped_mirror_is_named_and_its_loss_lowers_the_quality():
    # Ten femtoseconds, so that a bounce's 40 to 90 fs^2 outweighs its 0.6 % loss; at
    # thirty the two tie and one bounce fewer is, correctly, as good.
    grid, clean = gaussian(fwhm=10e-15)
    single = ChirpedMirror.lookup("PC70")
    stored = clean * stored_factor(grid, single.bounces(-6))
    found = compression.compress_mirrors(
        grid=grid, spectrum=stored, mirror="PC70", max_reflections=9
    )
    assert found.reflections == 6
    # Six reflections of a real coating: a few per cent are not returned.
    assert 0.9 < found.quality < 0.99


# ---------------------------------------------------------------- limit, apply, CEP


def test_the_transform_limit_of_a_chirped_gaussian_is_the_unchirped_one():
    grid, clean = gaussian()
    _, chirped = gaussian(phase=(0.0, 0.0, 900e-30, 30000e-45))
    limit = compression.transform_limit(grid=grid, spectrum=chirped)
    plain = compression.transform_limit(grid=grid, spectrum=clean)
    # The same modulus by construction; the chirp's unit phasor costs an ulp per bin.
    assert limit[0] == pytest.approx(plain[0], rel=1e-12)
    assert limit[1] == pytest.approx(plain[1], rel=1e-12)
    assert limit[0] == pytest.approx(FWHM, rel=2e-4)


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_the_accessors_take_the_same_element(kind):
    run = example_output(kind)
    found = compression.compress_taylor(run)
    t, power = get_power(run, plane=-1, transfer=found.transfer, oversampling=2)
    assert float(np.max(power)) == pytest.approx(found.peak_power_after, rel=1e-12)
    assert time.fwhm(t, power) == pytest.approx(found.fwhm_after, rel=1e-12)
    _, limit = get_power(run, plane=-1, transform_limit=True, oversampling=2)
    assert float(np.max(limit)) == pytest.approx(found.peak_power_limit, rel=1e-12)
    _, field = get_field(run, plane=-1, transfer=found.transfer, oversampling=2)
    np.testing.assert_allclose(np.abs(field) ** 2, power, rtol=1e-12)
    # That the default path is unchanged is not something a call can assert against
    # itself; tests/processing/test_fields.py holds `get_power` and `get_field`.


def test_apply_transfer_is_the_spectrum_times_the_element():
    run = example_output(n_modes=2)
    element = TaylorPhase((0.1, 0.0, 300e-30), WAVELENGTH)
    omega, after = compression.apply_transfer(element, run)
    _, before = compression.apply_transfer(TaylorPhase((), WAVELENGTH), run)
    assert after.shape == (2, 1, omega.size)
    assert np.all(np.diff(omega) > 0.0)
    np.testing.assert_array_equal(after, before * element(omega))
    assert compression.apply_transfer(element, run, modes=1)[1].shape == (
        1,
        1,
        omega.size,
    )


def test_the_plot_draws_the_compressed_pulse_and_its_default_is_unchanged():
    run = example_output()
    found = compression.compress_taylor(run)
    plain = time_1d(run, widths=False)
    default = np.array(plain.axes[0].lines[0].get_ydata())
    plt.close(plain)
    figure = time_1d(run, transfer=found.transfer, widths=False)
    compressed = np.asarray(figure.axes[0].lines[0].get_ydata())
    plt.close(figure)
    limited = time_1d(run, transform_limit=True, widths=False)
    limit = np.asarray(limited.axes[0].lines[0].get_ydata())
    plt.close(limited)
    again = time_1d(run, widths=False)
    assert np.array_equal(again.axes[0].lines[0].get_ydata(), default)
    plt.close(again)
    assert compressed.max() > 1.9 * default.max()
    assert compressed.max() == pytest.approx(limit.max(), rel=1e-6)


def test_the_carrier_envelope_phase_puts_the_field_maximum_on_the_envelope_peak():
    # Four femtoseconds: a many-cycle pulse's peak field does not depend on the phase
    # enough to be found (at 30 fs the dependence is below the sampling's own ripple).
    for kind in ("field", "envelope"):
        grid, stored = gaussian(kind, phase=(1.3,), fwhm=4e-15)
        found = compression.optimise_cep(grid=grid, spectrum=stored, oversampling=8)
        # The pulse was built with a carrier-envelope phase of 1.3 rad; the field's
        # maximum reaches the envelope's peak when that phase is taken out. The peak is
        # quadratic in the phase, so a bounded search good to 1e-5 of 0.1 rad gives it
        # to 1e-3 rad, and the peak itself to the oversampled grid's 1e-5.
        assert found.phase == pytest.approx(-1.3, abs=1e-3)
        envelope = math.sqrt(
            compression.transform_limit(grid=grid, spectrum=stored, oversampling=8)[1]
        )
        assert found.peak_field == pytest.approx(envelope, rel=1e-5)
        assert isinstance(found.transfer, TaylorPhase)
    _, near_pi = gaussian("envelope", phase=(math.pi - 0.05,), fwhm=4e-15)
    wrapped = compression.optimise_cep(grid=grid, spectrum=near_pi, oversampling=8)
    assert -math.pi < wrapped.phase <= math.pi
    # The optimum sits across the branch cut, where a search over [-pi, pi] is blind.
    assert abs(math.remainder(wrapped.phase + math.pi - 0.05, 2.0 * math.pi)) < 1e-3


# -------------------------------------------------------------------------- refusals


def test_the_two_input_forms_are_not_mixed():
    grid, stored = gaussian()
    run = example_output()
    for call in (
        compression.compress_taylor,
        lambda: compression.compress_taylor(run, grid=grid, spectrum=stored),
        lambda: compression.compress_taylor(grid=grid),
        lambda: compression.transform_limit(spectrum=stored),
    ):
        with pytest.raises(ProcessingError, match="not both"):
            call()


def test_unusable_requests_are_refused_by_name():
    grid, stored = gaussian()
    with pytest.raises(ProcessingError, match="order must be at least 2"):
        compression.compress_taylor(grid=grid, spectrum=stored, order=1)
    with pytest.raises(ProcessingError, match="bound_factor must be positive"):
        compression.compress_taylor(grid=grid, spectrum=stored, bound_factor=0.0)
    with pytest.raises(ProcessingError, match="must be ordered"):
        compression.compress_material(
            grid=grid, spectrum=stored, medium=WINDOW, thickness_range=(1e-3, -1e-3)
        )
    with pytest.raises(ProcessingError, match="at least 1"):
        compression.compress_mirrors(
            grid=grid, spectrum=stored, mirror="PC70", max_reflections=0
        )
    with pytest.raises(ProcessingError, match="together, or neither"):
        compression.compress_mirrors(
            grid=grid, spectrum=stored, mirror="PC70", max_reflections=2, medium=WINDOW
        )
    with pytest.raises(ProcessingError, match="no power to compress"):
        compression.compress_taylor(grid=grid, spectrum=np.zeros_like(stored))


def _spatial_output():
    """Return the example run relabelled as a radial (spatial) one."""
    metadata = example_output().metadata
    assert metadata is not None
    spatial = MemoryOutput()
    spatial.open(
        replace(metadata, simulation_kind="radial", spatial={"geometry": "radial"})
    )
    return spatial


@pytest.mark.parametrize(
    "call",
    [
        compression.transform_limit,
        compression.compress_taylor,
        lambda output: compression.compress_material(
            output, medium=WINDOW, thickness_range=(0.0, 1e-3)
        ),
        lambda output: compression.compress_mirrors(
            output, mirror="PC70", max_reflections=2
        ),
        compression.optimise_cep,
    ],
)
@pytest.mark.parametrize(
    ("build", "capability"),
    [(example_general_output, "general_modal"), (_spatial_output, "spatial")],
)
def test_a_basis_whose_summed_power_is_not_physical_is_refused_by_name(
    call, build, capability
):
    """Only an orthogonal modal output's summed |a|^2 is a power."""
    with pytest.raises(ProcessingError, match=f"this is a {capability} output"):
        call(build())


def test_an_output_without_metadata_is_refused():
    with pytest.raises(ProcessingError, match="no metadata"):
        compression.transform_limit(MemoryOutput())


def test_a_one_dimensional_spectrum_is_a_single_mode():
    grid, stored = gaussian(phase=(0.0, 0.0, 500e-30))
    flat = compression.compress_taylor(grid=grid, spectrum=stored[0, 0])
    full = compression.compress_taylor(grid=grid, spectrum=stored)
    assert flat.coefficients == full.coefficients
    assert grid_of(example_output()).kind == "envelope"
