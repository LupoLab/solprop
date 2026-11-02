# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the plane-statistics catalogue, its definitions and the collector."""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from legacy_geometry import LEGACY_GRID
from solprop import fourier, stats
from solprop.constants import C
from solprop.exceptions import ModeError, OutputError, SolpropWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode
from solprop.output.schema import SavedPlane
from solprop.profiles import Constant, PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stats import (
    CATALOGUE,
    METRIC_WEIGHTED_STATISTICS,
    PROFILE_STATISTICS,
    StatsSpec,
    collect,
    default_spec,
    energy,
    energy_window,
    fwhm_t,
    fwhm_t_outer,
    inband_wavelengths,
    omega0,
    peak_power,
    peak_power_total,
    peak_power_window,
    profile_value,
    zdw,
)

WAVELENGTH = 800e-9
FWHM = 30e-15
LIMITS = (400e-9, 2e-6)
TRANGE = 1e-12
ENERGY = 1e-6
PEAK = 1e9
SIGMA = 30e-15
# Power ~ exp(-t**2 / sigma**2) for an amplitude ~ exp(-t**2 / 2 sigma**2).
SIGMA_FWHM = 2.0 * SIGMA * np.sqrt(np.log(2.0))
GRIDS = ("field_grid", "envelope_grid")

# The pulse builders realise the requested energy and peak power exactly, so the only
# error in reading them back is the round trip through the transform pair: a handful of
# epsilon times the reduction length of a few thousand bins.
EXACT = 1e-12


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE, **LEGACY_GRID)


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE, **LEGACY_GRID)


@pytest.fixture
def narrow_grid():
    # Helium's Ermolov expansion is fitted over 90-1000 nm, so a grid that stays inside
    # it lets the zero-dispersion search run without a MaterialRangeWarning.
    return EnvelopeGrid.from_limits(1.0, 400e-9, (150e-9, 900e-9), 500e-15)


@pytest.fixture
def capillary():
    return MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)


def gaussian(grid, *, energy_j=None, peak_w=None):
    """Build one Gaussian pulse on a grid."""
    return GaussianPulse(
        wavelength=WAVELENGTH, fwhm=FWHM, energy=energy_j, peak_power=peak_w
    ).build(grid)


def envelope_of(grid, peaks, *, sigma=SIGMA, delay=0.0):
    """Build a multimode envelope whose modes are Gaussians of the given peak powers.

    The power of each is `peak * exp(-(t - delay)**2 / sigma**2)`, so its full width at
    half maximum is `2 sigma sqrt(ln 2)`.
    """
    shape = np.exp(-((grid.t - delay) ** 2) / (2.0 * sigma**2))
    columns = np.array([np.sqrt(peak) * shape for peak in peaks], dtype=complex)
    return fourier.forward(columns[:, np.newaxis, :], kind="envelope")


def plane_of(field, z=0.0):
    """Wrap a field in a saved plane."""
    return SavedPlane(z=z, field=field)


# ------------------------------------------------------------------------- energy


@pytest.mark.parametrize("grid_name", GRIDS)
def test_energy_reproduces_the_requested_pulse_energy(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    values = energy(grid, gaussian(grid, energy_j=ENERGY))
    assert values.shape == (1, 1)
    assert_allclose(values[0, 0], ENERGY, rtol=EXACT, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_energy_is_additive_over_modes(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    one = gaussian(grid, energy_j=ENERGY)
    stacked = np.concatenate([one, 2.0 * one], axis=0)
    values = energy(grid, stacked)
    # Doubling the amplitude quadruples the energy; the functional is a plain sum of
    # squared moduli, so the ratio is exact to round-off.
    assert_allclose(values[1, 0] / values[0, 0], 4.0, rtol=EXACT, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_tiling_windows_partition_the_energy(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    built = gaussian(grid, energy_j=ENERGY)
    windows = np.array(
        [
            [-np.inf, 2.0e15],
            [2.0e15, 2.4e15],
            [2.4e15, 2.8e15],
            [2.8e15, np.inf],
        ]
    )
    per_window = energy_window(grid, built, windows)
    assert per_window.shape == (4, 1, 1)
    # The gate is half-open, so tiling windows assign every bin exactly once and the
    # sum is the same set of terms as the total, up to the summation order.
    assert_allclose(per_window.sum(), energy(grid, built).sum(), rtol=EXACT, atol=0.0)


def test_a_window_outside_the_band_holds_no_energy(envelope_grid):
    built = gaussian(envelope_grid, energy_j=ENERGY)
    empty = np.array([[1.0e17, 2.0e17]])
    assert float(energy_window(envelope_grid, built, empty)[0, 0, 0]) == 0.0


# --------------------------------------------------------------------- peak power


@pytest.mark.parametrize("grid_name", GRIDS)
def test_peak_power_reproduces_the_requested_peak(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    values = peak_power(grid, gaussian(grid, peak_w=PEAK))
    assert values.shape == (1,)
    assert_allclose(values[0], PEAK, rtol=EXACT, atol=0.0)


def test_peak_power_sums_the_polarisation_components(envelope_grid):
    one = gaussian(envelope_grid, peak_w=PEAK)
    both = np.concatenate([one, one], axis=1)
    # Two identical components of one mode carry twice the power at every instant.
    assert_allclose(peak_power(envelope_grid, both)[0], 2.0 * PEAK, rtol=EXACT)


def test_the_windowed_peak_power_is_one_scalar_per_mode(envelope_grid):
    # A windowed all-modes reduction that is a no-op on a vector would store a row of
    # per-mode powers instead of the peak of the mode sum. Two modes peaking at the same
    # instant are the case that separates the two answers.
    field = envelope_of(envelope_grid, [5.0, 0.5])
    whole = np.array([[-np.inf, np.inf]])
    windowed = peak_power_window(envelope_grid, field, whole)
    assert windowed.shape == (1, 2)
    assert_allclose(windowed[0], [5.0, 0.5], rtol=EXACT)
    total = peak_power_total(envelope_grid, field)
    assert total.shape == ()
    assert_allclose(float(total), 5.5, rtol=EXACT)


def test_peak_power_total_is_not_the_largest_per_mode_peak(envelope_grid):
    field = envelope_of(envelope_grid, [5.0, 0.5])
    assert (
        float(peak_power_total(envelope_grid, field))
        > peak_power(envelope_grid, field).max()
    )


def test_peak_power_total_takes_the_peak_of_the_sum_not_the_sum_of_the_peaks(
    envelope_grid,
):
    # Modes that peak far apart: the sum of the per-mode peaks is 5.5 W, but no instant
    # ever carries that much.
    early = envelope_of(envelope_grid, [5.0], delay=-300e-15)
    late = envelope_of(envelope_grid, [0.5], delay=300e-15)
    field = np.concatenate([early, late], axis=0)
    # The peak falls between samples, so the largest sample sits below it by at most
    # 1 - exp(-(dt/2 sigma)**2), about 3e-4 here; the other mode's tail is negligible.
    assert_allclose(float(peak_power_total(envelope_grid, field)), 5.0, rtol=1e-3)


def test_windowed_peak_power_rejects_light_outside_the_window(envelope_grid):
    built = gaussian(envelope_grid, peak_w=PEAK)
    empty = np.array([[1.0e17, 2.0e17]])
    assert float(peak_power_window(envelope_grid, built, empty)[0, 0]) == 0.0


# ------------------------------------------------------------------------ omega0


def test_the_centroid_of_a_symmetric_envelope_spectrum_is_its_centre(envelope_grid):
    # An envelope spectrum with equal weight in the bins at +/- k about the reference
    # has its centroid at the reference exactly, because the pairs cancel term by term.
    field = np.zeros((1, 1, envelope_grid.n_omega), dtype=complex)
    for offset in (1, 5, 17):
        field[0, 0, offset] = 1.0
        field[0, 0, -offset] = 1.0
    field[0, 0, 0] = 2.0
    assert_allclose(
        omega0(envelope_grid, field)[0, 0], envelope_grid.omega0, rtol=EXACT
    )


def test_the_centroid_of_a_symmetric_field_spectrum_is_its_centre(field_grid):
    centre = field_grid.n_omega // 2
    field = np.zeros((1, 1, field_grid.n_omega), dtype=complex)
    for offset in (0, 3, 11):
        field[0, 0, centre - offset] = 1.0
        field[0, 0, centre + offset] = 1.0
    assert_allclose(
        omega0(field_grid, field)[0, 0], field_grid.omega[centre], rtol=EXACT
    )


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_centroid_of_a_pulse_sits_near_its_carrier(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    carrier = 2.0 * np.pi * C / WAVELENGTH
    centroid = omega0(grid, gaussian(grid, energy_j=ENERGY))[0, 0]
    # A 30 fs Gaussian is 15 THz wide against a 375 THz carrier, so the centroid sits
    # within a few per cent of it; the offset is the spectrum's own asymmetry in omega.
    assert_allclose(centroid, carrier, rtol=0.05)


def test_the_centroid_of_an_empty_column_is_not_a_number(envelope_grid):
    field = np.zeros((2, 1, envelope_grid.n_omega), dtype=complex)
    field[0, 0, 3] = 1.0
    values = omega0(envelope_grid, field)
    assert np.isfinite(values[0, 0])
    assert np.isnan(values[1, 0])


# --------------------------------------------------------------------------- fwhm


def test_the_temporal_width_of_a_triangle_is_exact(envelope_grid):
    # A power profile that is piecewise linear makes the crossing interpolation exact,
    # so the only error left is the transform round trip: 1e-12.
    base = 41.0 * envelope_grid.dt
    power = np.clip(1.0 - np.abs(envelope_grid.t) / base, 0.0, None)
    samples = np.sqrt(power).astype(complex)[np.newaxis, np.newaxis, :]
    field = fourier.forward(samples, kind="envelope")
    assert_allclose(float(fwhm_t(envelope_grid, field)), base, rtol=1e-12, atol=0.0)
    assert_allclose(
        float(fwhm_t_outer(envelope_grid, field)), base, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_temporal_width_of_a_gaussian_matches_the_analytic_value(
    grid_name, request
):
    grid = request.getfixturevalue(grid_name)
    width = float(fwhm_t(grid, gaussian(grid, energy_j=ENERGY)))
    # Linear interpolation of a smooth curve between two samples is second order in the
    # sample spacing relative to the feature it resolves, so (dt/fwhm)**2 bounds it.
    assert_allclose(width, FWHM, rtol=(grid.dt / FWHM) ** 2, atol=0.0)


def test_the_two_widths_differ_when_a_satellite_sits_above_half_maximum(envelope_grid):
    main = envelope_of(envelope_grid, [1.0])
    satellite = envelope_of(envelope_grid, [0.8], delay=400e-15)
    field = main + satellite
    inner = float(fwhm_t(envelope_grid, field))
    outer = float(fwhm_t_outer(envelope_grid, field))
    # The inner pair stops at the first sample below half maximum beside the main peak,
    # so it reports the main Gaussian alone; the sampling error is second order in
    # dt / fwhm as elsewhere.
    assert_allclose(inner, SIGMA_FWHM, rtol=(envelope_grid.dt / SIGMA_FWHM) ** 2)
    # The outer pair spans the satellite as well, so it is the separation plus a width.
    assert outer > 400e-15


def test_the_temporal_width_of_an_empty_plane_is_not_a_number(envelope_grid):
    field = np.zeros((1, 1, envelope_grid.n_omega), dtype=complex)
    assert np.isnan(fwhm_t(envelope_grid, field))
    assert np.isnan(fwhm_t_outer(envelope_grid, field))


def test_the_temporal_width_of_a_flat_trace_is_not_a_number(envelope_grid):
    # A continuous wave never falls to half its maximum, so it has no width at all.
    samples = np.ones((1, 1, envelope_grid.n_t), dtype=complex)
    field = fourier.forward(samples, kind="envelope")
    assert np.isnan(fwhm_t(envelope_grid, field))


def test_the_outer_width_is_not_a_number_when_the_pulse_fills_the_window(
    envelope_grid,
):
    # Power above half maximum at the very first sample: there is no sample outside it
    # to interpolate against.
    power = np.linspace(1.0, 0.0, envelope_grid.n_t)
    samples = np.sqrt(power).astype(complex)[np.newaxis, np.newaxis, :]
    field = fourier.forward(samples, kind="envelope")
    assert np.isnan(fwhm_t_outer(envelope_grid, field))


# --------------------------------------------------------- profiles and dispersion


def test_profile_value_reads_the_profile_at_the_plane():
    gradient = PressureGradient.two_point(1.0, 2.0, 0.5)
    # The steady-flow fill interpolates p**2 linearly, so at three quarters of the way
    # p = sqrt(4 + 0.75 (0.25 - 4)) bar, exactly and in bar (conventions.md section 1).
    expected = np.sqrt(4.0 + 0.75 * (0.25 - 4.0))
    assert_allclose(float(profile_value(gradient, 0.75)), expected, rtol=1e-12)


def test_inband_wavelengths_are_the_grid_limits(envelope_grid):
    shortest, longest = inband_wavelengths(envelope_grid)
    assert_allclose([shortest, longest], LIMITS, rtol=1e-12)


def test_zdw_finds_the_zero_of_the_group_velocity_dispersion(capillary):
    value = zdw([capillary], 0.0, wavelength_limits=(150e-9, 900e-9))
    assert value.shape == (1,)
    # The same root the mode's own accessor reports; see tests/modes/test_base.py.
    # 376.4 nm on optmat's helium since SEP 3's Phase 4a (364.8 nm on the registry's),
    # at 294 K; 376.7 nm since the default temperature became 293.15 K.
    assert_allclose(value[0], 376.7e-9, rtol=1e-4)


def test_zdw_refuses_a_mode_with_no_zero_in_the_window(capillary):
    with pytest.raises(ModeError, match="does not change sign"):
        zdw([capillary], 0.0, wavelength_limits=(500e-9, 900e-9))


def test_collect_records_nan_when_the_band_holds_no_zero(capillary):
    """A run asking for ``zdw`` is not stopped because its band has no zero to find."""
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
    field = gaussian(grid, energy_j=ENERGY)
    with pytest.warns(SolpropWarning, match="records nan for modes"):
        values = collect(
            StatsSpec(names=("zdw",)), grid, plane_of(field), modes=[capillary]
        )
    assert values["zdw"].shape == (1,)
    assert np.isnan(values["zdw"]).all()


def test_collect_records_each_modes_zero_on_its_own(capillary):
    """One mode with no zero in the band leaves the other's value in place."""
    grid = EnvelopeGrid.from_limits(1.0, 400e-9, (250e-9, 1.2e-6), 1e-12)
    field = gaussian(grid, energy_j=ENERGY)
    narrow = MarcatiliMode(2e-6, "He", 0.4, model="reduced", cladding=1.45)
    reference = zdw([capillary], 0.0, wavelength_limits=inband_wavelengths(grid))[0]
    with pytest.warns(SolpropWarning, match=r"records nan for modes \[1\]"):
        values = collect(
            StatsSpec(names=("zdw",)),
            grid,
            plane_of(field),
            modes=[capillary, narrow],
        )
    assert values["zdw"][0] == reference
    assert np.isnan(values["zdw"][1])


# ------------------------------------------------------------------------- the spec


def test_a_spec_defaults_to_no_statistics_and_no_windows():
    spec = StatsSpec()
    assert spec.names == ()
    assert spec.n_windows == 0
    assert spec.energy_windows.shape == (0, 2)


def test_a_spec_freezes_its_windows():
    spec = StatsSpec(names=("energy",), energy_windows=np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="read-only"):
        spec.energy_windows[0, 0] = 3.0


def test_a_spec_refuses_an_unknown_statistic():
    with pytest.raises(OutputError, match="unknown statistics"):
        StatsSpec(names=("energy", "banana"))


def test_a_spec_accepts_the_implemented_radial_width():
    assert StatsSpec(names=("fwhm_r",)).names == ("fwhm_r",)
    assert CATALOGUE["fwhm_r"].requires == ("spatial_geometry",)


def test_a_spec_refuses_a_reserved_catalogue_entry(monkeypatch):
    monkeypatch.setitem(
        CATALOGUE,
        "future_width",
        dataclasses.replace(CATALOGUE["fwhm_r"], implemented=False),
    )
    with pytest.raises(OutputError, match="reserved but not implemented"):
        StatsSpec(names=("future_width",))


@pytest.mark.parametrize(
    "name",
    [
        "electron_density",
        "peak_ionisation_rate",
        "peak_intensity",
        "quadrature_error",
    ],
)
def test_collect_refuses_a_backend_statistic_and_says_where_it_lives(
    envelope_grid, name
):
    """A backend forms these from its own buffers; a saved plane does not carry them."""
    assert CATALOGUE[name].backend
    spec = StatsSpec(names=(name,))
    with pytest.raises(OutputError, match=r"Backend\.stats"):
        collect(spec, envelope_grid, plane_of(gaussian(envelope_grid, energy_j=ENERGY)))


def test_the_field_statistics_are_not_backend_owned():
    assert not any(
        CATALOGUE[name].backend
        for name in ("energy", "peak_power", "fwhm_t", "zdw", "density")
    )


def test_a_spec_refuses_a_repeated_statistic():
    with pytest.raises(OutputError, match="these repeat"):
        StatsSpec(names=("energy", "energy"))


def test_a_spec_refuses_a_periodic_statistic_it_does_not_record():
    with pytest.raises(OutputError, match="periodic subset"):
        StatsSpec(names=("energy",), periodic=("peak_power",))


def test_a_spec_refuses_a_period_below_one():
    with pytest.raises(OutputError, match="at least 1"):
        StatsSpec(names=("energy",), period=0)


def test_a_spec_refuses_a_window_with_a_nan_edge():
    with pytest.raises(OutputError, match="NaN edge"):
        StatsSpec(energy_windows=np.array([[np.nan, 1.0]]))


def test_a_spec_refuses_a_window_that_is_not_ordered():
    with pytest.raises(OutputError, match=r"lo < hi"):
        StatsSpec(energy_windows=np.array([[2.0, 1.0]]))


def test_a_spec_accepts_a_periodic_subset():
    spec = StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=10)
    assert spec.periodic == ("energy",)
    assert spec.period == 10


@pytest.mark.parametrize("name", ["zdw"])
def test_a_spec_refuses_a_periodic_statistic_no_backend_evaluates_between_planes(name):
    """The trait is the gate, and the refusal names both the statistic and the set."""
    assert not CATALOGUE[name].between_planes
    with pytest.raises(OutputError, match="cannot be evaluated between save planes"):
        StatsSpec(names=("energy", name), periodic=(name,))


def test_a_spec_refuses_spatial_widths_between_planes():
    """Fluence widths are implemented at saved planes, with an explicit trait."""
    with pytest.raises(OutputError, match="cannot be evaluated between save planes"):
        StatsSpec(names=("fwhm_r",), periodic=("fwhm_r",))


def test_every_periodic_default_is_one_the_catalogue_allows():
    """Every name a run may put in `periodic` is one `_check_between_planes` admits."""
    allowed = {name for name, entry in CATALOGUE.items() if entry.between_planes}
    spec = StatsSpec(names=tuple(sorted(allowed)), periodic=tuple(sorted(allowed)))
    assert set(spec.periodic) == allowed


# ------------------------------------------------------------------- the catalogue


@pytest.mark.parametrize(
    "kind_and_flags",
    [
        ("mode_averaged", False, False),
        ("mode_averaged", True, True),
        ("multimode", False, False),
        ("multimode", True, True),
    ],
    ids=["averaged", "averaged-full", "multimode", "multimode-full"],
)
def test_every_default_name_is_in_the_catalogue_and_implemented(kind_and_flags):
    kind, with_profiles, with_zdw = kind_and_flags
    spec = default_spec(kind, with_profiles=with_profiles, with_zdw=with_zdw)
    assert spec.names
    for name in spec.names:
        assert CATALOGUE[name].implemented


def test_the_multimode_default_adds_the_total_peak_power():
    averaged = set(default_spec("mode_averaged").names)
    multimode = set(default_spec("multimode").names)
    assert multimode - averaged == {"peak_power_total"}


def test_every_catalogue_entry_describes_itself():
    for name, definition in CATALOGUE.items():
        assert definition.name == name
        assert definition.summary.endswith(".")
        assert definition.shape and definition.units


# -------------------------------------------------------------------- the collector


def test_collect_returns_the_documented_shapes(envelope_grid, capillary, narrow_grid):
    field = envelope_of(narrow_grid, [1.0, 0.25])
    spec = StatsSpec(
        names=(
            "omega0",
            "energy",
            "energy_window",
            "peak_power",
            "peak_power_window",
            "peak_power_total",
            "fwhm_t",
            "fwhm_t_outer",
            "density",
            "pressure",
            "core_radius",
            "zdw",
        ),
        energy_windows=np.array([[1.0e15, 3.0e15], [3.0e15, 6.0e15]]),
    )
    profiles = {
        "density": Constant(2.5e25),
        "pressure": Constant(1.0e5),
        "core_radius": Constant(125e-6),
    }
    values = collect(
        spec,
        narrow_grid,
        plane_of(field, z=0.25),
        modes=[capillary, capillary],
        profiles=profiles,
    )
    assert set(values) == set(spec.names)
    expected = {
        "omega0": (2, 1),
        "energy": (2, 1),
        "energy_window": (2, 2, 1),
        "peak_power": (2,),
        "peak_power_window": (2, 2),
        "peak_power_total": (),
        "fwhm_t": (),
        "fwhm_t_outer": (),
        "density": (),
        "pressure": (),
        "core_radius": (),
        "zdw": (2,),
    }
    assert {name: value.shape for name, value in values.items()} == expected


def test_collect_evaluates_nothing_for_an_empty_spec(envelope_grid):
    field = gaussian(envelope_grid, energy_j=ENERGY)
    assert collect(StatsSpec(), envelope_grid, plane_of(field)) == {}


def test_collect_agrees_with_the_statistics_it_calls(envelope_grid):
    field = gaussian(envelope_grid, energy_j=ENERGY)
    spec = StatsSpec(names=("energy", "peak_power"))
    values = collect(spec, envelope_grid, plane_of(field))
    # collect calls these very functions on the same plane: the same computation.
    assert np.array_equal(values["energy"], energy(envelope_grid, field))
    assert np.array_equal(values["peak_power"], peak_power(envelope_grid, field))


def test_collect_refuses_a_windowed_statistic_without_windows(envelope_grid):
    field = gaussian(envelope_grid, energy_j=ENERGY)
    spec = StatsSpec(names=("energy_window",))
    with pytest.raises(OutputError, match="at least one energy window"):
        collect(spec, envelope_grid, plane_of(field))


def test_collect_refuses_zdw_without_modes(envelope_grid):
    field = gaussian(envelope_grid, energy_j=ENERGY)
    with pytest.raises(OutputError, match="needs the propagated modes"):
        collect(StatsSpec(names=("zdw",)), envelope_grid, plane_of(field))


@pytest.mark.parametrize("name", PROFILE_STATISTICS)
def test_collect_refuses_a_profile_statistic_without_its_profile(envelope_grid, name):
    field = gaussian(envelope_grid, energy_j=ENERGY)
    with pytest.raises(OutputError, match=f"the {name!r} statistic needs"):
        collect(StatsSpec(names=(name,)), envelope_grid, plane_of(field))


def test_collect_names_the_profiles_it_was_given(envelope_grid):
    field = gaussian(envelope_grid, energy_j=ENERGY)
    with pytest.raises(OutputError, match=r"collect was given \['density'\]"):
        collect(
            StatsSpec(names=("pressure",)),
            envelope_grid,
            plane_of(field),
            profiles={"density": Constant(1.0)},
        )


def test_the_manual_page_states_the_trait_the_catalogue_holds():
    """Keep the manual's between-plane traits consistent with the catalogue.

    The page's catalogue tables tell a user what may go in a periodic subset, so a
    trait changed in the code and not on the page would be a promise the
    package does not keep. Parsed rather than eyeballed, as the kernel contract's
    tables are (``tests/translate/test_problem.py``).
    """
    page = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "guide"
        / "results"
        / "statistics-catalogue.md"
    )
    text = page.read_text(encoding="utf-8")
    rows = {}
    for block in re.finditer(
        r"   \* - ``([a-z0-9_]+)``\n((?:     - .*\n(?:       .*\n)*)*)", text
    ):
        name, body = block.group(1), block.group(2)
        cells = [line[7:] for line in body.splitlines() if line.startswith("     - ")]
        if name in CATALOGUE and len(cells) >= 4:
            rows[name] = cells[2]
    # The catalogue tables include every implemented entry, including spatial widths.
    documented = {name for name, entry in CATALOGUE.items() if entry.implemented}
    assert set(rows) == documented
    for name, cell in rows.items():
        assert cell == ("yes" if CATALOGUE[name].between_planes else "no"), name


# ---------------------------------------- the metric-weighted statistics


def metric_of(n_modes, n_omega, *, defect=1e-3, off=2e-3):
    """A Hermitian positive-definite metric, the identity plus a declared defect."""
    block = np.broadcast_to(
        np.eye(n_modes, dtype=complex)[:, :, None], (n_modes, n_modes, n_omega)
    ).copy()
    block *= 1.0 - defect
    if n_modes > 1:
        block[0, 1] = off + 1j * off
        block[1, 0] = off - 1j * off
    return block


def two_channel(grid):
    """Two Gaussian channels of unequal energy on one grid."""
    one = gaussian(grid, energy_j=1e-6)
    other = np.roll(gaussian(grid, energy_j=4e-7), 3, axis=-1)
    return np.concatenate([one, other], axis=0)


@pytest.mark.parametrize("name", sorted(METRIC_WEIGHTED_STATISTICS))
def test_no_metric_is_the_identity_and_is_what_it_was(field_grid, name):
    """`metric=None` is the fast path and changes no existing number.

    The default is ``metric=None``, so both calls run one computation: an explicitly
    exact no-op, compared exactly.
    """
    field = two_channel(field_grid)
    function = getattr(stats, name)
    assert np.array_equal(
        np.asarray(function(field_grid, field)),
        np.asarray(function(field_grid, field, metric=None)),
        equal_nan=True,
    )
    identity = np.broadcast_to(
        np.eye(2, dtype=complex)[:, :, None], (2, 2, field_grid.n_omega)
    ).copy()
    weighted = np.asarray(function(field_grid, field, metric=identity))
    assert_allclose(
        weighted,
        np.asarray(function(field_grid, field)),
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
    )


def test_the_metric_weighted_energy_is_the_quadratic_form(field_grid):
    """`physical_energy` with W is `sum_k w_k a.H W a`, by an independent sum."""
    field = two_channel(field_grid)
    metric = metric_of(2, field_grid.n_omega)
    weighted = float(stats.physical_energy(field_grid, field, metric=metric))
    # The same quantity, formed without the whitening: the metric-weighted
    # spectral density integrated with the grid's own Parseval weight.
    quadratic = np.einsum(
        "mpk,mnk,npk->k", field.conj(), metric, field, optimize=True
    ).real
    direct = float(
        np.sum(
            fourier.energy_omega(
                np.sqrt(quadratic)[None, None, :].astype(complex),
                field_grid.dt,
                kind=field_grid.kind,
            )
        )
    )
    assert_allclose(weighted, direct, rtol=1e-12, atol=0.0)
    # And it differs from the unweighted number by the metric's own defect.
    plain = float(stats.physical_energy(field_grid, field))
    assert 1e-4 < abs(weighted / plain - 1.0) < 1e-2


def test_the_whitening_is_the_positive_definite_square_root(field_grid):
    field = two_channel(field_grid)
    metric = metric_of(2, field_grid.n_omega)
    whitened = stats._whitened(field, metric)
    assert_allclose(
        np.einsum("mpk,mpk->k", whitened.conj(), whitened).real,
        np.einsum("mpk,mnk,npk->k", field.conj(), metric, field).real,
        rtol=1e-12,
        atol=0.0,
    )


def test_the_metric_weighted_flux_and_widths_move_together(field_grid):
    """One quantity, four functionals: the peak and the widths share the trace."""
    field = two_channel(field_grid)
    metric = metric_of(2, field_grid.n_omega, defect=0.2, off=0.0)
    peak = float(stats.analytic_flux(field_grid, field, metric=metric))
    plain = float(stats.analytic_flux(field_grid, field))
    # A metric of (1 - 0.2) I scales the physical flux by exactly that.
    assert_allclose(peak, 0.8 * plain, rtol=1e-12, atol=0.0)
    # A uniform scale cannot move a half-maximum width.
    for name in ("fwhm_t", "fwhm_t_outer"):
        function = getattr(stats, name)
        assert_allclose(
            float(function(field_grid, field, metric=metric)),
            float(function(field_grid, field)),
            rtol=1e-12,
            atol=0.0,
        )


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_the_spectral_flux_integrates_to_the_physical_energy(
    field_grid, envelope_grid, kind
):
    """``d_omega`` times the density is the energy, on both grid layouts.

    The closure that makes ``spectral_flux`` a density of ``physical_energy`` and
    not a second quantity. It is
    exact on an envelope grid and on every interior bin of a field grid; the zero
    and Nyquist bins are the analytic signal's documented exception
    (:func:`solprop.fourier.spectral_density`) and any grid this package builds
    apodises them to zero, which is asserted here rather than assumed.
    """
    grid = field_grid if kind == "field" else envelope_grid
    field = two_channel(grid)
    density = stats.spectral_flux(grid, field)
    assert density.shape == (grid.n_omega,)
    total = float(np.sum(density) * grid.d_omega)
    exact = float(stats.physical_energy(grid, field))
    # The documented difference and no other: the density weights the two end bins
    # 4 where `energy_omega` weights them 1, because the analytic signal does not
    # double them. On an envelope grid there is no such exception and the closure
    # is exact to rounding.
    overcount = (
        0.0
        if kind == "envelope"
        else 3.0 * grid.dt / grid.n_t * float(np.sum(np.abs(field[..., [0, -1]]) ** 2))
    )
    assert_allclose(total - exact, overcount, rtol=1e-9, atol=1e-30)
    # And on this pulse that exception is nothing: the two end bins hold 5.2e-34 of
    # the energy, so the closure holds to rounding anyway (measured 0.0 here).
    assert abs(total / exact - 1.0) < 1e-13


def test_the_spectral_flux_is_the_metric_weighted_density(field_grid):
    """With W it is the quadratic form per bin, by an independently written sum."""
    field = two_channel(field_grid)
    metric = metric_of(2, field_grid.n_omega)
    weighted = stats.spectral_flux(field_grid, field, metric=metric)
    # The same density formed without the whitening: the metric's quadratic form
    # per bin, scaled by the field grid's own `2 dt^2 / pi` Jacobian
    # (`docs/dev/conventions.md` section 4).
    quadratic = np.einsum(
        "mpk,mnk,npk->k", field.conj(), metric, field, optimize=True
    ).real
    direct = 2.0 * field_grid.dt**2 / np.pi * quadratic
    assert_allclose(weighted, direct, rtol=1e-12, atol=0.0)
    # And the weighted density integrates to the weighted energy, as the
    # unweighted one does.
    assert_allclose(
        float(np.sum(weighted) * field_grid.d_omega),
        float(stats.physical_energy(field_grid, field, metric=metric)),
        rtol=1e-13,
        atol=0.0,
    )


def test_the_spectral_flux_is_not_available_between_planes():
    """An ``n_omega``-wide row per period is a cost, so it is a plane quantity.

    The catalogue's ``between_planes`` rule is about cost per step, not
    difficulty, and this is the only general modal statistic that is not a scalar.
    """
    assert not CATALOGUE["spectral_flux"].between_planes
    assert CATALOGUE["physical_energy"].between_planes
    with pytest.raises(OutputError, match="spectral_flux"):
        StatsSpec(names=("spectral_flux",), periodic=("spectral_flux",))


@pytest.mark.parametrize(("off", "defined"), [(1e-16, False), (2e-3, True)])
def test_omega0_leaves_a_whitened_column_of_rounding_noise_undefined(
    field_grid, off, defined
):
    """An empty column is empty relative to the plane, not to zero.

    After a whitening an unlaunched channel is not identically zero: it carries
    ``W^(1/2)_{01}`` times its launched partner. Where that share is real -- an
    off-diagonal of 2e-3 -- the centroid is a real number and is reported. Where it
    is the whitening's own rounding -- the 1e-16 off-diagonal a unitary transform
    of a certified reference actually produces, which lands 1e-32 of the launched
    density in the column -- there is no centroid to report and it is ``nan``.
    """
    launched = gaussian(field_grid, energy_j=1e-6)
    field = np.concatenate([launched, np.zeros_like(launched)], axis=0)
    metric = metric_of(2, field_grid.n_omega, off=off)
    centroid = stats.omega0(field_grid, field, metric=metric)
    assert np.isfinite(centroid[0, 0])
    if defined:
        assert np.isfinite(centroid[1, 0])
    else:
        assert np.isnan(centroid[1, 0])


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        (np.zeros((3, 3, 4), complex), "needs W of shape"),
        (np.zeros((2, 2, 8), complex), "positive-definite"),
    ],
)
def test_a_metric_weighted_statistic_refuses_a_metric_it_cannot_use(metric, message):
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 20e-15)
    field = np.zeros((2, 1, grid.n_omega), dtype=complex)
    shaped = (
        metric
        if metric.shape[2] == grid.n_omega
        else np.broadcast_to(metric[:, :, :1], (*metric.shape[:2], grid.n_omega))
    )
    with pytest.raises(OutputError, match=message):
        stats.physical_energy(grid, field, metric=shaped)


def test_collect_hands_the_metric_only_to_the_statistics_that_take_one(field_grid):
    field = two_channel(field_grid)
    metric = metric_of(2, field_grid.n_omega, defect=0.2, off=0.0)
    spec = StatsSpec(names=("energy", "physical_energy", "analytic_flux", "fwhm_t"))
    plain = collect(spec, field_grid, plane_of(field))
    weighted = collect(spec, field_grid, plane_of(field), metric=metric)
    # `energy` is a coordinate quantity and takes no metric, so it is unchanged.
    assert_allclose(weighted["energy"], plain["energy"], rtol=0.0, atol=0.0)
    for name in ("physical_energy", "analytic_flux"):
        assert_allclose(weighted[name], 0.8 * plain[name], rtol=1e-12, atol=0.0)
    assert "zdw" not in METRIC_WEIGHTED_STATISTICS
