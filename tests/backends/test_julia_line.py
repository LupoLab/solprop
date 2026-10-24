# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare scalar line-geometry sources and short propagation with NumPy."""

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
from solprop.grid import EnvelopeGrid, FieldGrid, LineGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import FixedSchedule, RungeKutta


@pytest.mark.parametrize("envelope", [False, True])
def test_line_source_and_ten_steps(envelope: bool) -> None:
    """Keep field and envelope line sources within the reference rounding budget."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    grid = (
        EnvelopeGrid.from_limits(0.01, 800e-9, (400e-9, 1.2e-6), 150e-15, thg=False)
        if envelope
        else FieldGrid.from_limits(0.01, 800e-9, (400e-9, 1.2e-6), 150e-15)
    )
    settings = dataclasses.replace(
        simulation().settings,
        quadrature=None,
        components=None,
        integrator=RungeKutta(),
        absorber=absorber(grid),
        schedule=FixedSchedule(np.full(10, 1e-6)),
        stats=default_spec("free"),
        save_planes=[1e-5],
    )
    sim = Simulation.from_free(
        grid,
        LineGrid(600e-6, 12),
        SpatialMedium("Ar", 1.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 200e-6
        ),
        settings=settings,
    )
    problem = sim.translate()
    candidate = select("julia", problem, grid=sim.grid, modes=sim.modes)
    reference = select("numpy_ref", problem, grid=sim.grid, modes=sim.modes)
    assert isinstance(candidate, JuliaBackend)
    assert isinstance(reference, NumpyRefBackend)
    try:
        # Random phases exercise the transverse axis and final partial block.
        rng = np.random.default_rng(42)
        field = problem.initial.field0 * np.exp(1j * rng.uniform(-1, 1, problem.shape))
        for engine in (candidate, reference):
            engine.set_initial(field, None)
        got = np.asarray(
            runtime().evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        want = reference._nonlinear(
            field, reference._stage_scalars(0.0, 0.0), np.ones_like(field)
        )
        # Conservative source bound for temporal/transverse transforms and cubic
        # products. Ten DOPRI steps contain at most 70 source evaluations.
        bound = higham_gamma(128 * problem.grids.n_to * problem.shape[1])
        assert np.linalg.norm(want) > 0
        assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
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
    finally:
        candidate.close()
        reference.close()
