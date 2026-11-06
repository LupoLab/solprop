# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading a radial beam from its file, held to the run's own measures."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.output import open_file
from solprop.physdata import chi3, refractive_index
from solprop.processing import (
    far_field,
    field_spectrogram,
    fluence_along,
    on_axis_power,
    on_axis_spectra,
    on_axis_spectrogram,
    radial_intensity,
    radial_spectrum,
)
from solprop.processing.beams import _per_solid_angle
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.recipe import Recipe
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    column_spectrum,
    critical_power,
    node_spectrum,
    on_axis_field,
    on_axis_intensity,
    on_axis_trace,
    spatial_energy,
    spatial_fluence,
)
from solprop.tools import beam_params, beam_through_lens

WAIST = 1e-3
LENGTH = 0.1
ENERGY = 2e-4
WAVELENGTH = 800e-9


@pytest.fixture(scope="module")
def plane() -> tuple[np.ndarray, SpatialFlux]:
    grid = EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, (600e-9, 1100e-9), 200e-15)
    space = RadialGrid(4 * WAIST, 64)
    medium = SpatialMedium("Ar", pressure=1.0)
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=ENERGY)
    spectrum = SpatialPulse(pulse, WAIST).build(grid, space, medium=medium)
    return spectrum, SpatialFlux.resolve(grid, space, medium)


@pytest.fixture(scope="module")
def radial_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    recipe = Recipe.example("radial_kerr")
    path = tmp_path_factory.mktemp("beam") / "run.h5"
    recipe.run(path)
    return path


def test_the_node_spectrum_integrates_to_the_fluence(plane) -> None:
    spectrum, flux = plane
    grid = flux.grid
    total = node_spectrum(spectrum, flux).sum(axis=1) * grid.d_omega
    assert total == pytest.approx(spatial_fluence(spectrum, flux), rel=1e-6, abs=1e-9)


def test_the_column_spectrum_sums_to_the_energy(plane) -> None:
    spectrum, flux = plane
    assert column_spectrum(spectrum, flux).sum() == pytest.approx(
        spatial_energy(spectrum, flux), rel=1e-12
    )


def test_the_on_axis_trace_peaks_at_the_on_axis_intensity(plane) -> None:
    spectrum, flux = plane
    trace = on_axis_trace(spectrum, flux)
    assert trace.size == flux.grid.t.size
    assert trace.max() == pytest.approx(on_axis_intensity(spectrum, flux))
    assert on_axis_field(spectrum, flux).shape == (1, flux.grid.n_omega)


def test_the_readers_agree_with_the_file_s_own_measure(radial_file: Path) -> None:
    with open_file(str(radial_file)) as output:
        x, on_axis = on_axis_spectra(output)
        kept = np.asarray(output["field_planes"])
        assert on_axis.shape == (kept.size, x.size)
        assert np.all(np.diff(x) > 0)
        r, x_r, across = radial_spectrum(output)
        assert across.shape == (r.size, x_r.size)
        # The innermost node is nearly on the axis of this smooth beam.
        assert across[0] == pytest.approx(
            on_axis[-1], rel=0.01, abs=1e-3 * on_axis.max()
        )
        low, high = x_r[0], x_r[-1]
        assert low < WAVELENGTH < high
        # The nearest plane to z = 0 is the first, and it holds the launched beam.
        assert radial_spectrum(output, z=0.0)[2][0] == pytest.approx(
            radial_spectrum(output, plane=0)[2][0]
        )
        t, power = on_axis_power(output)
        assert t.size == power.size
        assert power.max() > 0
        fine_t, fine = on_axis_power(output, oversampling=4)
        assert fine_t.size == fine.size == 4 * t.size
        assert fine[::4] == pytest.approx(power, abs=1e-9 * power.max())


def test_the_fluence_and_intensity_are_the_measures_of_the_plane(
    radial_file: Path,
) -> None:
    with open_file(str(radial_file)) as output:
        z, r, fluence = fluence_along(output)
        assert fluence.shape == (z.size, r.size)
        r_i, t, flux = radial_intensity(output, plane=0)
    assert flux.shape == (r_i.size, t.size)
    assert fluence[0].max() > 0
    # The fluence is the flux integrated in time.
    dt = float(t[1] - t[0])
    assert (flux.sum(axis=1) * dt)[0] == pytest.approx(fluence[0, 0], rel=1e-3)


def test_a_gaussian_s_far_field_has_its_diffraction_angle(tmp_path: Path) -> None:
    # A wide aperture: the transverse wavevectors are spaced pi / R, so the angle needs
    # several of them inside the beam's divergence.
    text = (
        Recipe.example("radial_kerr")
        .to_text()
        .replace('radius = "4 mm", n_r = 64', 'radius = "16 mm", n_r = 128')
    )
    path = tmp_path / "run.h5"
    Recipe.loads(text).run(path)
    with open_file(str(path)) as output:
        theta, x, values = far_field(output, plane=0)
    assert theta[0] == 0.0
    # The carrier's row: energy per solid angle falls to 1/e^2 at lambda / (pi w0).
    row = values[:, int(np.argmin(np.abs(x - WAVELENGTH)))]
    expected = WAVELENGTH / (math.pi * WAIST)
    falls = np.interp(np.exp(-2.0), row[::-1] / row.max(), theta[::-1])
    assert falls == pytest.approx(expected, rel=0.03)
    # And the map integrates to the plane's energy.
    per_x = 2 * np.pi * np.trapezoid(values * np.sin(theta)[:, None], theta, axis=0)
    assert np.trapezoid(per_x, x) == pytest.approx(ENERGY, rel=0.03)


def test_the_on_axis_spectrogram_sits_at_the_carrier_and_the_pulse(
    radial_file: Path,
) -> None:
    with open_file(str(radial_file)) as output:
        t, x, values = on_axis_spectrogram(output, plane=0, positions=64)
    assert values.shape == (x.size, t.size)
    row, column = np.unravel_index(np.argmax(values), values.shape)
    assert x[row] == pytest.approx(WAVELENGTH, rel=0.05)
    assert abs(t[column]) < 30e-15


def test_a_field_spectrogram_refuses_what_it_cannot_gate() -> None:
    t = np.linspace(-1e-13, 1e-13, 256)
    field = np.cos(2e15 * t)
    assert field_spectrogram(t, field, positions=8)[2].shape[1] > 0
    with pytest.raises(ProcessingError, match="gate position"):
        field_spectrogram(t, field, positions=0)
    with pytest.raises(ProcessingError, match="no gate position"):
        field_spectrogram(t, field, time_range=(1.0, 2.0))
    with pytest.raises(ProcessingError, match="within the time window"):
        field_spectrogram(t, field, gate_fwhm=1.0)


def test_the_beam_readers_refuse_another_grid(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    Recipe.example("cartesian_beam").run(path)
    with (
        open_file(str(path)) as output,
        pytest.raises(ProcessingError, match="radial grid"),
    ):
        radial_spectrum(output)


def test_the_beam_numbers_are_the_lesson_s() -> None:
    omega = 2 * math.pi * C / WAVELENGTH
    index = float(refractive_index("Ar", omega, pressure=1.0))
    power = critical_power(
        WAVELENGTH, chi3("Ar", pressure=1.0, wavelength=WAVELENGTH), index=index
    )
    ratio = 3.0
    beam = beam_params(300e-6, WAVELENGTH, "Ar", 1.0, ratio * power)
    rayleigh = math.pi * index * (300e-6) ** 2 / WAVELENGTH
    assert beam.rayleigh_range == pytest.approx(rayleigh)
    assert beam.critical_power == pytest.approx(power)
    assert beam.power_ratio == pytest.approx(ratio)
    # As plot_03_filament.py writes Marburger's formula out.
    marburger = 0.367 * rayleigh / math.sqrt((math.sqrt(ratio) - 0.852) ** 2 - 0.0219)
    assert beam.collapse_distance == pytest.approx(marburger)


def test_below_the_critical_power_nothing_collapses() -> None:
    beam = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 1e9)
    assert beam.power_ratio < 1.0
    assert beam.collapse_distance is None
    # Just above it, the collapse lies far ahead.
    power = beam.critical_power
    near = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 1.05 * power)
    assert near.collapse_distance is not None
    assert near.collapse_distance > 3 * near.rayleigh_range


def test_a_focus_draws_the_collapse_nearer_by_the_lens_law() -> None:
    power = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 1e9).critical_power
    free = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 3 * power)
    focused = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 3 * power, focus=1.0)
    assert free.collapse_distance is not None
    assert focused.collapse_distance is not None
    # At launch the beam is wider by 1 + (d/z_R)^2 in area, and converges with radius
    # d + z_R^2/d.
    z_r = free.rayleigh_range
    wider = free.collapse_distance * (1.0 + (1.0 / z_r) ** 2)
    curvature = 1.0 + z_r**2
    assert 1 / focused.collapse_distance == pytest.approx(1 / wider + 1 / curvature)
    assert focused.collapse_distance < free.collapse_distance
    # Just above the critical power, a beam a Rayleigh range past its waist diverges
    # faster than it self-focuses, and has no collapse ahead.
    gone = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 1.05 * power, focus=-z_r)
    assert gone.collapse_distance is None


def test_a_waist_just_ahead_is_nearly_collimated() -> None:
    power = beam_params(3e-4, WAVELENGTH, "Ar", 1.0, 1e9).critical_power
    free = beam_params(3e-4, WAVELENGTH, "Ar", 1.0, 3 * power)
    for shift in (1e-6, -1e-6):
        near = beam_params(3e-4, WAVELENGTH, "Ar", 1.0, 3 * power, focus=shift)
        assert near.collapse_distance == pytest.approx(free.collapse_distance, rel=1e-4)


def test_the_linear_radius_is_a_gaussian_beam_s() -> None:
    beam = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 1e9, focus=0.5)
    z = np.array([0.5, 0.5 + beam.rayleigh_range])
    assert beam.radius(z) == pytest.approx(1e-3 * np.array([1.0, math.sqrt(2.0)]))


def test_the_linear_radius_is_what_a_run_measures(tmp_path: Path) -> None:
    text = Recipe.example("radial_kerr").to_text()
    # Vacuum-like and low energy: linear, with a focus 2 m ahead of the launch.
    text = text.replace('"0.2 mJ"', '"1 nJ"').replace(
        'waist = "1 mm"', 'waist = "1 mm"\npropz = "-2 m"'
    )
    path = tmp_path / "run.h5"
    Recipe.loads(text).run(path)
    with open_file(str(path)) as output:
        radius = np.asarray(output["stats/beam_radius"])[:, 1]
        z = np.asarray(output["z"])
    beam = beam_params(1e-3, WAVELENGTH, "Ar", 1.0, 1e3, focus=2.0)
    assert beam.radius(z) == pytest.approx(radius, rel=2e-3)


def test_a_lens_makes_the_textbook_waist() -> None:
    waist, distance = beam_through_lens(1.0, 5e-3, WAVELENGTH)
    assert waist == pytest.approx(WAVELENGTH / (math.pi * 5e-3), rel=1e-3)
    assert distance == pytest.approx(1.0, rel=1e-3)
    # A weak focus falls short of the focal length.
    waist, distance = beam_through_lens(10.0, 3e-4, WAVELENGTH)
    assert distance < 10.0
    assert waist < 3e-4 * 2


def test_the_filament_example_runs_above_the_critical_power() -> None:
    recipe = Recipe.example("filament")
    assert recipe.route == "radial"
    preview = recipe.preview_input()
    keywords: Any = recipe.to_dict()["build"]
    assert keywords["initial"]["waist"] == "300 um"
    beam = beam_params(300e-6, WAVELENGTH, "Ar", 1.0, float(preview.peak_power))
    assert 2.5 < beam.power_ratio < 3.5


def test_a_wide_angle_s_solid_angle_takes_the_obliquity() -> None:
    # Columns evenly filled out to 80 degrees: the energy per steradian integrates over
    # the sphere back to the columns' total, which a paraxial measure would overstate.
    omega = 2 * math.pi * C / WAVELENGTH
    k = omega / C
    k_r = np.linspace(0.0005, 0.98, 4000) * k
    widths = np.gradient(k_r)
    energy = np.ones_like(k_r)
    theta, density = _per_solid_angle(energy, k_r, widths, omega)
    total = 2 * np.pi * np.trapezoid(density * np.sin(theta), theta)
    assert total == pytest.approx(energy.sum(), rel=1e-2)


def test_a_run_off_a_radial_grid_is_refused(tmp_path: Path) -> None:
    text = Recipe.example("soliton").to_text().replace("n_save = 201", "n_save = 2")
    path = tmp_path / "fibre.h5"
    Recipe.loads(text).run(path)
    with (
        open_file(str(path)) as output,
        pytest.raises(ProcessingError, match="not on a radial grid"),
    ):
        on_axis_spectra(output)


def test_a_file_with_no_plane_measure_is_refused(
    radial_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with open_file(str(radial_file)) as output:
        metadata = output.metadata
        assert metadata is not None
        monkeypatch.setattr(type(metadata), "spatial", property(lambda _self: {}))
        with pytest.raises(ProcessingError, match="no plane measure"):
            far_field(output)


def test_a_beam_of_two_polarisations_has_no_on_axis_spectrogram(
    radial_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solprop.processing import beams  # noqa: PLC0415 - only here

    real = beams.on_axis_field
    monkeypatch.setattr(
        beams, "on_axis_field", lambda *args: np.repeat(real(*args), 2, axis=0)
    )
    with (
        open_file(str(radial_file)) as output,
        pytest.raises(ProcessingError, match="two polarisations"),
    ):
        on_axis_spectrogram(output)


def test_a_real_field_s_on_axis_spectrogram_keeps_its_carrier(tmp_path: Path) -> None:
    text = (
        Recipe.example("radial_kerr")
        .to_text()
        .replace("EnvelopeGrid.from_limits", "FieldGrid.from_limits")
        .replace(", thg = false", "")
    )
    path = tmp_path / "field.h5"
    Recipe.loads(text).run(path)
    with open_file(str(path)) as output:
        _t, x, values = on_axis_spectrogram(output, plane=0, positions=64)
    row = np.unravel_index(np.argmax(values), values.shape)[0]
    assert x[row] == pytest.approx(WAVELENGTH, rel=0.05)
