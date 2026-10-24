# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check omission of structurally zero Cartesian absorber transforms."""

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
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import FixedSchedule, RungeKutta


@pytest.mark.parametrize("absorption", ["none", "temporal", "spatial"])
@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_zero_absorber_source_and_ten_steps(
    monkeypatch: pytest.MonkeyPatch, absorption: str, envelope: bool, split: bool
) -> None:
    """Preserve active losses and omit only an exactly absent additive source."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    monkeypatch.setenv("SOLPROP_JULIA_SPATIAL_LAYOUT", "strided")
    monkeypatch.setenv("SOLPROP_JULIA_PARALLEL_COPY", "1")
    monkeypatch.setenv("SOLPROP_JULIA_SKIP_ZERO_ABSORBER", "1")
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
        absorber=absorber(grid) if absorption != "none" else None,
        absorber_scheme="split_step" if split else "nonlinear",
        schedule=FixedSchedule(np.full(10, 1e-6)),
        stats=default_spec("free"),
        save_planes=[1e-5],
    )
    sim = Simulation.from_free(
        grid,
        FreeGrid(
            600e-6,
            8,
            700e-6,
            10,
            window_factor=0.1 if absorption == "spatial" else 0.0,
        ),
        SpatialMedium("Ar", 1.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 200e-6
        ),
        settings=settings,
    )
    problem = sim.translate()
    # The Absorber supplies the physical collar's reference length. Isolate its
    # spatial contribution after translation; TransverseCollar instead belongs
    # to the diagonal wavevector operator and needs no additive source FFT.
    if absorption == "spatial":
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, temporal_absorber=None, temporal_absorber_fine=None
            ),
            flags=dataclasses.replace(problem.flags, has_temporal_absorber=False),
        )
    assert problem.transform.geometry is not None
    assert bool(np.any(problem.transform.geometry.absorber_rate)) == (
        absorption == "spatial"
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    candidate = select("julia", problem, grid=sim.grid, modes=sim.modes)
    reference = select("numpy_ref", problem, grid=sim.grid, modes=sim.modes)
    assert isinstance(candidate, JuliaBackend)
    assert isinstance(reference, NumpyRefBackend)
    try:
        # Random phases exercise both transverse axes and the final partial block.
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
        assert bool(candidate._engine.config.in_stage_absorber) == (
            absorption == "spatial" or (absorption == "temporal" and not split)
        )
        assert estimate_memory(problem, "julia").total == sum(
            candidate.memory_ledger().values()
        )
    finally:
        candidate.close()
        reference.close()
