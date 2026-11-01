# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent layout and observable checks for the offline Luna comparison."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from data import compare_spatial_witnesses as witness
from data.compare_spatial_witnesses import (
    checked_gap,
    cladding_from,
    compare_observables,
    gamma_n,
    observables,
    stored_plane,
)
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.modes import MarcatiliMode
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController


def test_julia_cartesian_axes_use_x_fastest_with_unequal_sizes(tmp_path):
    values = np.arange(3 * 4 * 5).reshape(1, 3, 4, 5).astype(complex)
    with h5py.File(tmp_path / "layout.h5", "w") as file:
        file["Eω"] = values
        actual = stored_plane(file, 0, "free")
    for ix in range(3):
        for iy in range(4):
            np.testing.assert_array_equal(actual[ix + 3 * iy, 0], values[0, ix, iy])


def test_diagnostic_metrics_have_known_scaling_and_do_not_certify_a_bound():
    reference = {
        "spectrum": np.ones((3, 4)),
        "mode_energy": np.ones((3, 2)),
        "axis_peak": np.ones(3),
    }
    actual = {key: 4 * value for key, value in reference.items()}
    result = compare_observables(actual, reference)
    # Doubling the field quadruples energy and intensity: every relative gap is 3.
    assert result["exit_content_relative"] == 3
    assert result["exit_spectral_shape_l2"] == 3
    assert result["axis_peak_along_z_relative_linf"] == 3
    reference["axis_peak"][:] = 0
    assert (
        compare_observables(actual, reference)["axis_peak_along_z_relative_linf"]
        is None
    )


def test_input_bound_failure_is_recorded_and_raised():
    checks = []
    bound = gamma_n(64)
    with pytest.raises(ValueError, match="exceeds"):
        checked_gap(checks, "deliberate miss", np.ones(2) * 2, np.ones(2), bound)
    assert checks == [
        {"name": "deliberate miss", "value": 1.0, "bound": bound, "pass": False}
    ]
    with pytest.raises(ValueError, match="shapes"):
        checked_gap([], "shape", np.ones(2), np.ones(3), bound)


def test_witness_cladding_keeps_signed_extinction_samples(tmp_path):
    omega = np.linspace(2e15, 4e15, 16)
    samples = np.linspace(1.4, 1.5, 16) - 1e-8j
    with h5py.File(tmp_path / "media.h5", "w") as file:
        file["omega"] = omega
        file["active"] = np.ones(16, bool)
        file["cladding"] = samples
        model = cladding_from(file)
    # Interior interpolation nodes avoid a range-endpoint warning from rounding.
    np.testing.assert_allclose(
        model.index(omega[1:-1]), samples[1:-1], rtol=gamma_n(128), atol=0
    )


def test_cartesian_observables_recover_uniform_plane_wave_energy_and_intensity():
    grid = EnvelopeGrid.from_limits(0.001, 800e-9, (600e-9, 1.2e-6), 100e-15)
    space = FreeGrid(1e-3, 4, 2e-3, 6, collar_fraction=0)
    field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    amplitude = 3.0
    field[0, 0, 0] = amplitude * space.n_columns * grid.n_t
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(),
        [],
        field,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-10),
            save_planes=[grid.length],
            dz_max=grid.length,
            dz_init=grid.length,
        ),
        noise=None,
    )
    _, energy, peak = observables(simulation, field)
    intensity = 0.5 * EPS0 * C * amplitude**2
    expected = (
        intensity * (2 * space.radius_x) * (2 * space.radius_y) * grid.n_t * grid.dt
    )
    # Positive reductions and two FFT scalings are inside gamma(128).
    np.testing.assert_allclose(energy.sum(), expected, rtol=gamma_n(128), atol=0)
    np.testing.assert_allclose(peak, intensity, rtol=gamma_n(128), atol=0)
    field[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        observables(simulation, field)
    field[:] = 0
    with pytest.raises(ValueError, match="positive"):
        observables(simulation, field)


def test_aligned_product_route_preserves_entrance_and_observables(tmp_path):
    grid = EnvelopeGrid.from_limits(0.001, 800e-9, (600e-9, 1.2e-6), 100e-15)
    mode = MarcatiliMode(100e-6, loss=False)
    values = np.zeros((1, 1, grid.n_omega), complex)
    values[0, 0, 0] = grid.n_t
    with h5py.File(tmp_path / "aligned.h5", "w") as file:
        file["Eω"] = values
        scalar = stored_plane(file, 0, "modal")
        product = stored_plane(file, 0, "modal_product")
    np.testing.assert_array_equal(product[:, 0], 0)
    np.testing.assert_array_equal(product[:, 1], scalar[:, 0])
    simulation = Simulation(
        grid,
        [mode],
        [],
        scalar,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-10),
            save_planes=[grid.length],
            dz_max=grid.length,
            dz_init=grid.length,
        ),
    )
    # A zero component contributes exactly zero to each quadratic observable.
    for actual, expected in zip(
        observables(simulation, product), observables(simulation, scalar), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=gamma_n(64), atol=0)


def test_comparison_records_failed_input_and_refuses_overwrite(tmp_path):
    source = tmp_path / "absent.h5"
    assert not witness.run_one(
        source, tmp_path / "media.h5", "numpy_ref", 1e-6, tmp_path
    )
    record = json.loads((tmp_path / "absent-numpy_ref-1e-06.json").read_text())
    assert record["status"] == "failed"
    assert "FileNotFoundError" in record["traceback"]
    with pytest.raises(FileExistsError):
        witness.run_one(source, tmp_path / "media.h5", "numpy_ref", 1e-6, tmp_path)


def test_nonfinite_axis_intensity_is_a_failure(monkeypatch):
    grid = EnvelopeGrid.from_limits(0.001, 800e-9, (600e-9, 1.2e-6), 100e-15)
    space = FreeGrid(1e-3, 4, 2e-3, 6, collar_fraction=0)
    monkeypatch.setattr(witness, "on_axis_intensity", lambda *args: np.nan)
    simulation = SimpleNamespace(grid=grid, spatial_grid=space)
    field = np.ones((space.n_columns, 1, grid.n_omega), complex)
    with pytest.raises(ValueError, match="on-axis intensity"):
        observables(simulation, field)
