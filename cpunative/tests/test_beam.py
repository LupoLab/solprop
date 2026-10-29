# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small source and fixed-step checks for the Cartesian performance experiment."""

import numpy as np
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import select
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta

from solprop_cpunative.beam import CartesianBackend


@pytest.mark.parametrize(
    ("threads", "nx", "ny"), [(1, 12, 10), (4, 14, 10), (4, 96, 80)]
)
@pytest.mark.parametrize("fft_schedule", ["auto", "direct", "split"])
def test_cartesian_source_and_four_steps(threads, nx, ny, fft_schedule, monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", str(threads))
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    monkeypatch.setenv("SOLPROP_CPUNATIVE_CARTESIAN_FFT", fft_schedule)
    length = 2e-6
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (500e-9, 1.2e-6), 100e-15, thg=False, fft_size="smooth"
    )
    simulation = Simulation.from_free(
        grid,
        FreeGrid(120e-6, nx, 90e-6, ny, window_factor=0),
        SpatialMedium(),
        [KerrResponse(chi3=2.2e-22, thg=False)],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 40e-6),
        settings=SimulationSettings(
            save_planes=[length],
            dz_max=length,
            dz_init=length / 4,
            integrator=LowStorageRungeKutta(scheme="rk4ip"),
            schedule=FixedSchedule.uniform(0, length, length / 4),
            backend="cpunative",
            stats=StatsSpec(
                names=(
                    "energy",
                    "peak_intensity",
                    "on_axis_intensity",
                    "on_axis_spectrum",
                    "beam_radius",
                    "fwhm_r",
                )
            ),
        ),
    )
    problem = simulation.translate()
    candidate = select("cpunative", problem, grid=grid)
    assert isinstance(candidate, CartesianBackend)
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        for backend in (candidate, reference):
            backend.set_initial(problem.initial.field0, None)
        retained = candidate.observe().require_field()
        # Observations must remain independent of resident state and later copies.
        retained_copy = retained.copy()
        another = candidate._cpu.field()
        another.fill(0)
        np.testing.assert_array_equal(retained, retained_copy)
        got = candidate._cpu.evaluate(
            candidate._stage_scalars(0, 0).kerr_density
        ).reshape(problem.shape)
        expected = reference._adams_lab_source(reference._state, 0)
        # Transform and four-step rounding allowance, fixed independently of results.
        bound = 128 * np.finfo(float).eps * (grid.n_t + nx + ny)
        assert np.linalg.norm(got - expected) < bound * np.linalg.norm(expected)
        for backend in (candidate, reference):
            backend.propagate_to(length)
        got = candidate.observe().require_field()
        np.testing.assert_array_equal(retained, retained_copy)
        expected = reference.observe().require_field()
        assert np.linalg.norm(got - expected) < bound * np.linalg.norm(expected)
        assert np.isfinite(got).all()
        got_stats, expected_stats = (
            candidate.stats(problem.stats_spec),
            reference.stats(problem.stats_spec),
        )
        assert set(got_stats) == set(expected_stats) == set(problem.stats_spec.names)
        for name, expected in expected_stats.items():
            scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
            np.testing.assert_allclose(
                got_stats[name], expected, rtol=bound, atol=bound * scale, err_msg=name
            )
        assert candidate._cpu.memory_bytes == sum(candidate.memory_ledger().values())
    finally:
        candidate.close()
        reference.close()
