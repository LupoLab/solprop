# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check scalar Cartesian plasma on bounded physical-space chunks."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_julia import READY, simulation

from bounds import higham_gamma
from solprop.backends import select
from solprop.backends.julia import JuliaBackend
from solprop.backends.julia._runtime import runtime
from solprop.backends.memory import estimate_memory
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.grid import FieldGrid, FreeGrid, LineGrid
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, RungeKutta


@pytest.mark.parametrize("line", [False, True])
@pytest.mark.parametrize("rk4", [False, True])
@pytest.mark.parametrize("scale", [0.0, 36.0])
def test_cartesian_plasma_source_and_ten_steps(
    monkeypatch: pytest.MonkeyPatch, line: bool, rk4: bool, scale: float
) -> None:
    """Check current assembly, observations, chunk tails and accepted state."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    monkeypatch.setenv("SOLPROP_JULIA_SPATIAL_LAYOUT", "strided")
    monkeypatch.setenv("SOLPROP_JULIA_PARALLEL_COPY", "1")
    monkeypatch.setenv("SOLPROP_JULIA_REAL_ROOT", "1")
    grid = FieldGrid.from_limits(0.01, 800e-9, (300e-9, 2e-6), 250e-15)
    settings = dataclasses.replace(
        simulation().settings,
        quadrature=None,
        components=None,
        integrator=LowStorageRungeKutta() if rk4 else RungeKutta(),
        absorber=absorber(grid),
        absorber_scheme="split_step" if rk4 else "nonlinear",
        ionisation_field_range=(1e7, 1e11),
        schedule=FixedSchedule(np.full(10, 1e-6)),
        stats=default_spec("free"),
        save_planes=[1e-5],
    )
    sim = Simulation.from_free(
        grid,
        LineGrid(600e-6, 12) if line else FreeGrid(600e-6, 8, 700e-6, 10),
        SpatialMedium("Ar", 1.0),
        [
            KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False),
            PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)),
        ],
        SpatialPulse(
            GaussianPulse(
                wavelength=800e-9, fwhm=15e-15, energy=2e-3 if line else 1e-6
            ),
            200e-6,
        ),
        settings=settings,
    )
    problem = sim.translate()
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    candidate = select("julia", problem, grid=sim.grid, modes=sim.modes)
    reference = select("numpy_ref", problem, grid=sim.grid, modes=sim.modes)
    assert isinstance(candidate, JuliaBackend)
    assert isinstance(reference, NumpyRefBackend)
    try:
        # Random phases exercise the transverse axis and final partial block.
        rng = np.random.default_rng(42)
        field = (
            scale
            * problem.initial.field0
            * np.exp(1j * rng.uniform(-0.1, 0.1, problem.shape))
        )
        for engine in (candidate, reference):
            engine.set_initial(field, None)
        got = np.asarray(
            runtime().evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        want = reference._nonlinear(
            field, reference._stage_scalars(0.0, 0.0), np.ones_like(field)
        )
        # Temporal/transverse transforms, rate spline and strict FIR scans;
        # ten DOPRI steps contain at most 70 source evaluations.
        bound = higham_gamma(
            128
            * problem.grids.n_to
            * (problem.shape[1] + problem.responses.plasma_integral.taps.size)
        )
        assert bound > 0
        if scale:
            assert np.linalg.norm(want) > 0
        assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
        spec = StatsSpec(
            names=(
                "species_ionisation",
                "species_electron_density",
                "electron_density",
                "peak_ionisation_rate",
                "plasma_undershoot",
                "plasma_work_defect",
                "plasma_boundary_level",
            )
        )
        calls = candidate.metadata()["source_evaluations"]
        before = candidate.observe().field
        got_stats, want_stats = candidate.stats(spec), reference.stats(spec)
        for name in spec.names:
            np.testing.assert_allclose(
                got_stats[name], want_stats[name], rtol=bound, atol=bound
            )
        np.testing.assert_array_equal(candidate.observe().field, before)
        assert candidate.metadata()["source_evaluations"] == calls
        if scale:
            assert np.max(got_stats["species_ionisation"]) > 1e-8
        for engine in (candidate, reference):
            engine.propagate_to(1e-5)
        got_field, want_field = candidate.observe().field, reference.observe().field
        assert got_field is not None and want_field is not None
        assert np.linalg.norm(got_field - want_field) <= 70 * bound * np.linalg.norm(
            want_field
        )
        assert estimate_memory(problem, "julia").total == sum(
            candidate.memory_ledger().values()
        )
        assert np.all(got_field[..., ~problem.grids.active_mask] == 0)
    finally:
        candidate.close()
        reference.close()
