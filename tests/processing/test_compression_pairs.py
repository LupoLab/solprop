# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical stage recovery, fixed selection and honest search failures."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from compressor_reference import MALITSON_SILICA, reference_phase
from solprop.constants import C
from solprop.exceptions import MaterialError, OpticsGeometryError, ProcessingError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.optics import (
    GratingPair,
    PrismGeometry,
    PrismPair,
    brewster_angle,
    brewster_apex,
    littrow_angle,
    stored_factor,
)
from solprop.output import MemoryOutput
from solprop.processing import compression
from solprop.processing._examples import (
    example_general_output,
    example_grid,
    example_output,
)
from solprop.processing.axes import _plane_field, grid_of
from solprop.pulses import GaussianPulse

WAVELENGTH = 800e-9
BAND = (600e-9, 1100e-9)
SILICA = MALITSON_SILICA


def grating(distance=0.0004, band=BAND):
    return GratingPair(
        600e3, distance, littrow_angle(600e3, WAVELENGTH), WAVELENGTH, band=band
    )


def prism(insertion=0.015, separation=0.6, band=BAND):
    return PrismPair(
        SILICA,
        PrismGeometry(
            separation,
            insertion,
            0.003,
            brewster_apex(SILICA, WAVELENGTH),
            brewster_angle(SILICA, WAVELENGTH),
        ),
        WAVELENGTH,
        band=band,
    )


def inverse_pulse(pair, *, kind="envelope", fwhm=15e-15, sampling=1):
    cls = EnvelopeGrid if kind == "envelope" else FieldGrid
    # A 2 ps window holds the deliberately stretched pulses. This builds spectra
    # only: no propagation is run. Doubling sampling lowers the Nyquist wavelength.
    grid = cls.from_limits(0.001, WAVELENGTH, (400e-9 / sampling, 2e-6), 2e-12)
    clean = GaussianPulse(wavelength=WAVELENGTH, fwhm=fwhm, energy=1e-6).build(grid)
    selected = (
        grid.active_mask
        & (grid.omega >= 2 * math.pi * C / pair.band[1])
        & (grid.omega <= 2 * math.pi * C / pair.band[0])
    )
    phase = np.zeros(grid.n_omega)
    phase[selected] = reference_phase(pair, grid.omega[selected])
    # Inverse of optics +phi is internal +i*phi. Fixed selection is made by the
    # compressor, not by throwing away invalid trial frequencies in this fixture.
    return grid, clean * np.exp(1j * phase)


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_grating_search_recovers_known_separation(kind):
    target = grating()
    grid, stored = inverse_pulse(target, kind=kind)
    found = compression.compress_gratings(
        pair=replace(target, separation=0.0007),
        separation_range=(0.0, 0.001),
        grid=grid,
        spectrum=stored,
    )
    assert found.success, found.message
    assert isinstance(found.transfer, GratingPair)
    # The peak loses ~1e-6 at nanometre distance errors for this broad pulse;
    # double precision resolves its quadratic maximum to much less than that.
    assert found.transfer.separation == pytest.approx(target.separation, abs=2e-9)
    assert found.quality >= 1 - 1e-6
    assert found.peak_power_after > found.peak_power_before
    assert found.evaluations >= 65
    assert "normal separation" in str(found)
    assert "selected pulse" in str(found)
    assert "fixed working band" in str(found)
    _energy_conserved(grid, stored, found.transfer)


def _energy_conserved(grid, stored, pair):
    selected = np.abs(stored_factor(grid, pair))
    before = stored * selected
    after = stored * stored_factor(grid, pair)
    # Per-bin unit modulus plus one reduction: budget 8 eps per retained bin.
    tolerance = 8 * np.finfo(float).eps * np.count_nonzero(selected)
    assert np.sum(abs(after) ** 2) == pytest.approx(
        np.sum(abs(before) ** 2), rel=tolerance
    )


def test_prism_insertion_search_at_fixed_separation():
    target = prism()
    grid, stored = inverse_pulse(target)
    start = replace(target, geometry=replace(target.geometry, insertion=0.012))
    found = compression.compress_prisms(
        pair=start, insertion_range=(0.008, 0.021), grid=grid, spectrum=stored
    )
    assert found.success, found.message
    assert isinstance(found.transfer, PrismPair)
    geometry = found.transfer.geometry
    assert replace(geometry, insertion=start.geometry.insertion) == start.geometry
    # ~1.4e5 fs^2/m insertion sensitivity, with a 15 fs pulse; 2 nm is well below
    # the measured 1e-6 peak resolution, unlike a narrow-band two-knob assertion.
    assert geometry.insertion == pytest.approx(target.geometry.insertion, abs=2e-9)
    assert found.quality >= 1 - 1e-6
    assert found.peak_power_after > found.peak_power_before
    assert "H 15 mm" in str(found)
    _energy_conserved(grid, stored, found.transfer)


@pytest.mark.medium
def test_prism_joint_search_and_feasible_boundaries():
    target = prism(insertion=0.0123, separation=0.53)
    grid, stored = inverse_pulse(target, fwhm=7e-15)
    options = {
        "pair": prism(),
        "insertion_range": (0.006, 0.02),
        "separation_range": (0.3, 0.8),
        "grid": grid,
        "spectrum": stored,
    }
    first = compression.compress_prisms(**options)
    again = compression.compress_prisms(**options)
    assert first.success, first.message
    assert isinstance(first.transfer, PrismPair)
    assert isinstance(again.transfer, PrismPair)
    assert first.transfer.geometry == again.transfer.geometry
    assert first.evaluations == again.evaluations
    # The short pulse distinguishes the correlated R,H directions by higher orders.
    # These tolerances bound their phase error below ~0.003 rad on significant bins.
    assert first.transfer.geometry.separation == pytest.approx(0.53, abs=2e-4)
    assert first.transfer.geometry.insertion == pytest.approx(0.0123, abs=2e-6)
    assert first.quality >= 1 - 1e-6
    selected = (grid.omega >= 2 * math.pi * C / BAND[1]) & (
        grid.omega <= 2 * math.pi * C / BAND[0]
    )
    difference = first.transfer.phase(grid.omega[selected]) - target.phase(
        grid.omega[selected]
    )
    weights = abs(stored[0, 0, selected]) ** 2
    assert np.sqrt(np.sum(weights * difference**2) / np.sum(weights)) < 0.003


@pytest.mark.parametrize(
    "coordinate,side",
    [
        ("insertion", "lower"),
        ("insertion", "upper"),
        ("separation", "lower"),
        ("separation", "upper"),
    ],
)
@pytest.mark.medium
def test_prism_joint_search_reports_each_motor_bound(coordinate, side):
    bounds = {"insertion": (0.011, 0.015), "separation": (0.4, 0.6)}
    settings = {"insertion": 0.013, "separation": 0.5}
    settings[coordinate] = bounds[coordinate][side == "upper"]
    target = prism(**settings)
    grid, stored = inverse_pulse(target, fwhm=7e-15)
    found = compression.compress_prisms(
        pair=prism(),
        grid=grid,
        spectrum=stored,
        insertion_range=bounds["insertion"],
        separation_range=bounds["separation"],
    )
    assert not found.success
    assert f"{coordinate} lies on its {side} bound" in found.message
    assert found.quality >= 1 - 1e-6


def test_separation_only_and_partly_infeasible_prism_search():
    target = prism(insertion=0.007, separation=0.03, band=(700e-9, 900e-9))
    grid, stored = inverse_pulse(target)
    found = compression.compress_prisms(
        pair=target, separation_range=(0.0, 0.06), grid=grid, spectrum=stored
    )
    assert found.success
    assert isinstance(found.transfer, PrismPair)
    assert found.transfer.geometry.insertion == target.geometry.insertion
    assert found.transfer.geometry.separation == pytest.approx(0.03, abs=2e-6)
    assert found.quality >= 1 - 1e-6
    for separation_range in [None, (0.01, 0.05)]:
        with pytest.raises(ProcessingError, match="no feasible"):
            compression.compress_prisms(
                pair=target,
                grid=grid,
                spectrum=stored,
                insertion_range=(0.0, 1e-4),
                separation_range=separation_range,
            )


def test_grating_unachievable_sign_keeps_boundary_answer():
    pair = grating()
    grid = example_grid()
    clean = GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=1e-6).build(grid)
    wrong_sign = clean * stored_factor(grid, pair)
    found = compression.compress_gratings(
        pair=pair, separation_range=(0.0, 0.0006), grid=grid, spectrum=wrong_sign
    )
    assert not found.success
    assert "lower bound" in found.message
    assert isinstance(found.transfer, GratingPair)
    assert found.transfer.separation == 0.0


def test_fixed_band_cannot_improve_trial_by_clipping():
    grid = example_grid()
    target = prism(band=(700e-9, 900e-9))
    stored = GaussianPulse(wavelength=WAVELENGTH, fwhm=10e-15, energy=1e-6).build(grid)
    pulse = compression._pair_pulse(
        None,
        grid,
        stored,
        pair=target,
        plane=-1,
        modes=None,
        bandpass=None,
        oversampling=2,
    )
    mask = (np.abs(stored_factor(grid, target)) > 0).astype(float)
    np.testing.assert_array_equal(pulse.stored, stored * mask)
    baseline = pulse.stored.copy()
    for insertion in [0.012, 0.015, 0.018]:
        trial = replace(target, geometry=replace(target.geometry, insertion=insertion))
        pulse.objective(trial)
        np.testing.assert_array_equal(pulse.stored, baseline)
        _energy_conserved(grid, stored, trial)
    # A pair can pass at the reference yet miss the red/blue part of a wider grid.
    narrow = prism(insertion=0.003, band=(790e-9, 810e-9))
    with pytest.raises(OpticsGeometryError):
        replace(narrow, band=(600e-9, 1100e-9))
    for spectrum in [np.zeros_like(stored), stored]:
        bandpass = None if not np.any(spectrum) else np.zeros(grid.n_omega)
        with pytest.raises(ProcessingError, match="no power"):
            compression.compress_prisms(
                pair=target,
                insertion_range=(0.01, 0.02),
                grid=grid,
                spectrum=spectrum,
                bandpass=bandpass,
            )


@pytest.mark.parametrize("kind", ["envelope", "field"])
@pytest.mark.parametrize("family", ["gratings", "prisms"])
def test_compressors_input_and_grid_parity(kind, family):
    run = example_output(kind, n_modes=3, n_pol=2)
    grid = grid_of(run)
    stored = _plane_field(run, -1, None, convention="fft")
    function = getattr(compression, f"compress_{family}")
    optics = (
        {"pair": grating(), "separation_range": (0.0, 0.001)}
        if family == "gratings"
        else {"pair": prism(), "insertion_range": (0.008, 0.02)}
    )
    selection = {"modes": [0, 2], "bandpass": (700e-9, 950e-9), "oversampling": 3}
    saved = function(run, **optics, **selection)
    raw = function(grid=grid, spectrum=stored, **optics, **selection)
    # Exactly the same selected arrays and deterministic search path.
    assert saved.quality == raw.quality
    assert saved.peak_power_after == raw.peak_power_after
    assert saved.evaluations == raw.evaluations
    if family == "gratings":
        assert saved.transfer == raw.transfer
    else:
        assert saved.transfer.geometry == raw.transfer.geometry
    single = function(grid=grid, spectrum=stored[0, 0], **optics)
    shaped = function(grid=grid, spectrum=stored[0:1, 0:1], **optics)
    assert single.quality == shaped.quality


@pytest.mark.parametrize("family", ["gratings", "prisms"])
def test_pairs_reject_unsupported_power_objectives_before_reading_fields(
    family, monkeypatch
):
    general = example_general_output()
    spatial = MemoryOutput()
    metadata = example_output().metadata
    assert metadata is not None
    spatial.open(
        replace(
            metadata,
            simulation_kind="radial",
            spatial={"geometry": "radial"},
        )
    )

    def no_pulse(*args, **kwargs):
        pytest.fail("unsupported capability reached the pulse objective")

    monkeypatch.setattr(compression, "_pulse_of", no_pulse)
    function = getattr(compression, f"compress_{family}")
    options = (
        {"pair": grating(), "separation_range": (0.0, 0.001)}
        if family == "gratings"
        else {"pair": prism(), "insertion_range": (0.008, 0.02)}
    )
    for output in [general, spatial]:
        with pytest.raises(ProcessingError, match="physical-flux"):
            function(output, **options)


@pytest.mark.parametrize(
    "bounds",
    [(0.0, 0.0), (0.01, 0.0), (-0.01, 0.01), (0.0, math.inf), (math.nan, 0.01)],
)
def test_pair_search_rejects_bad_motor_ranges(bounds):
    with pytest.raises(ProcessingError, match="strictly ordered"):
        compression.compress_gratings(pair=grating(), separation_range=bounds)
    with pytest.raises(ProcessingError, match="strictly ordered"):
        compression.compress_prisms(pair=prism(), insertion_range=bounds)


def test_pair_search_input_errors():
    options = {"pair": grating(), "separation_range": (0.0, 0.001)}
    grid = example_grid()
    for bad in [
        np.zeros((1, grid.n_omega), complex),
        np.zeros(5, complex),
        np.full(grid.n_omega, complex(math.nan)),
        np.zeros((0, 1, grid.n_omega), complex),
    ]:
        with pytest.raises(ProcessingError, match="spectrum must"):
            compression.compress_gratings(grid=grid, spectrum=bad, **options)
    with pytest.raises(ProcessingError, match="at least one"):
        compression.compress_prisms(pair=prism())
    with pytest.raises(ProcessingError, match="not both"):
        compression.compress_gratings(**options)
    with pytest.raises(ProcessingError, match="not both"):
        compression.compress_gratings(example_output(), grid=grid, **options)
    with pytest.raises(ProcessingError, match="metadata"):
        compression.compress_gratings(MemoryOutput(), **options)
    with pytest.raises(ProcessingError, match="oversampling"):
        compression.compress_gratings(
            grid=grid,
            spectrum=np.ones(grid.n_omega, complex),
            oversampling=0,
            **options,
        )


@pytest.mark.medium
def test_pair_recovery_converges_with_temporal_sampling():
    target = prism()
    results = []
    for sampling in [1, 2]:
        grid, stored = inverse_pulse(target, sampling=sampling)
        results.append(
            compression.compress_prisms(
                pair=target,
                insertion_range=(0.008, 0.021),
                grid=grid,
                spectrum=stored,
                oversampling=2 * sampling,
            )
        )
    # Both discretisations place the unchirped maximum exactly at the time origin.
    # The independently imposed phase is accurate to 5e-8 rad, far below 1e-6 power.
    assert min(result.quality for result in results) >= 1 - 1e-6
    assert results[0].peak_power_limit == pytest.approx(
        results[1].peak_power_limit, rel=1e-6
    )
    assert results[0].fwhm_after == pytest.approx(results[1].fwhm_after, rel=2e-3)


def test_pair_without_band_keeps_the_full_active_selection():
    grid = example_grid()
    stored = GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=1e-6).build(grid)
    pair = GratingPair(100.0, 0.0, littrow_angle(100.0, WAVELENGTH), WAVELENGTH)
    found = compression.compress_gratings(
        grid=grid, spectrum=stored, pair=pair, separation_range=(0.0, 0.001)
    )
    assert not found.success
    assert "all active frequencies" in str(found)
    # FFT butterflies and power reductions accumulate O(eps * log2(n_t)) error.
    assert found.quality == pytest.approx(
        1.0, rel=0.0, abs=16 * np.finfo(float).eps * math.log2(grid.n_t)
    )


def test_fixed_band_between_grid_bins_is_empty():
    grid = example_grid()
    pair = replace(
        grating(), wavelength=800.123456e-9, band=(800.123455e-9, 800.123457e-9)
    )
    with pytest.raises(ProcessingError, match="no active frequencies"):
        compression.compress_gratings(
            grid=grid,
            spectrum=np.ones(grid.n_omega, complex),
            pair=pair,
            separation_range=(0.0, 0.001),
        )


def test_no_feasible_grating_order_is_reported_clearly():
    grid = example_grid()
    pair = GratingPair(1.2e6, 0.001, littrow_angle(1.2e6, WAVELENGTH), WAVELENGTH)
    # This grid extends to 2 um, beyond the first-order branch of this grating.
    with pytest.raises(ProcessingError, match="no feasible"):
        compression.compress_gratings(
            grid=grid,
            spectrum=np.ones(grid.n_omega, complex),
            pair=pair,
            separation_range=(0.0, 0.001),
        )


def test_material_failure_during_search_is_not_swallowed(monkeypatch):
    def bad_phase(self, omega):
        raise MaterialError("test material model failed")

    target = prism()
    grid = example_grid()
    monkeypatch.setattr(PrismPair, "phase", bad_phase)
    with pytest.raises(MaterialError, match="material model failed"):
        compression.compress_prisms(
            grid=grid,
            spectrum=np.ones(grid.n_omega, complex),
            pair=target,
            insertion_range=(0.01, 0.02),
        )


def test_feasibility_scan_keeps_isolated_samples_and_handles_no_improvement():
    # A degenerate sampled feasible set must not cause a refinement over invalid
    # neighbours; a constant score must leave its best sampled point available.
    def isolated(coordinate):
        return 1.0 if coordinate == 0.5 else math.inf

    assert compression._feasible_scan(isolated)[:2] == (0.5, 1.0)
    coordinate, value, _ = compression._feasible_scan(lambda x: 1.0)
    assert coordinate == 0.0 and value == 1.0


@pytest.mark.parametrize("axis", [0, 1])
def test_boundary_refinement_improves_a_global_search_approaching_an_edge(axis):
    def objective(coordinates):
        return 1 + coordinates[axis] + (coordinates[1 - axis] - 0.314159) ** 2

    coordinates, success, message = compression._pair_search(objective, ["H", "R"])
    assert not success
    assert "lower bound" in message
    assert coordinates[axis] == 0.0
    # The minimiser is quadratic in the free coordinate and flat at sqrt(eps).
    assert coordinates[1 - axis] == pytest.approx(0.314159, abs=3e-8)


def test_search_iteration_limit_is_not_reported_as_convergence(monkeypatch):

    def exhausted(*args, **kwargs):
        return SimpleNamespace(x=np.array([0.5, 0.5]), fun=2.0, success=False)

    monkeypatch.setattr(compression, "differential_evolution", exhausted)
    _, success, message = compression._pair_search(
        lambda coordinates: 1 + sum((value - 0.5) ** 2 for value in coordinates),
        ["H", "R"],
    )
    assert not success
    assert "iteration limit" in message


@pytest.mark.medium
def test_joint_prism_search_accepts_a_partly_infeasible_motor_box():
    target = prism(insertion=0.007, separation=0.03, band=(650e-9, 1050e-9))
    grid, stored = inverse_pulse(target, fwhm=10e-15)
    found = compression.compress_prisms(
        pair=target,
        grid=grid,
        spectrum=stored,
        insertion_range=(0.0, 0.012),
        separation_range=(0.0, 0.06),
    )
    assert found.success, found.message
    assert isinstance(found.transfer, PrismPair)
    assert found.quality >= 1 - 1e-6
    assert found.transfer.band == target.band
    assert (
        found.transfer.geometry.first_face_distance
        == target.geometry.first_face_distance
    )
    _energy_conserved(grid, stored, found.transfer)
