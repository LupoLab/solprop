# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Focused experimental checks for the additional SEP 30 execution paths."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_julia import READY, simulation

from bounds import EPS, higham_gamma
from solprop.backends import select
from solprop.backends.exponential import phi_functions
from solprop.backends.julia._runtime import runtime
from solprop.backends.memory import estimate_memory
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, RungeKutta


@pytest.mark.parametrize("kind", ["resolved", "radial"])
def test_field_plasma_source_and_short_window(kind):
    """Exercise ionisation, the band scan and held-current assembly together."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    sim = simulation(trange=250e-15)
    responses = (*sim.responses, PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)))
    if kind == "radial":
        grid = FieldGrid.from_limits(0.01, 800e-9, (300e-9, 2e-6), 250e-15)
        sim = Simulation.from_radial(
            grid,
            RadialGrid(600e-6, 24),
            SpatialMedium("Ar", 1.0),
            responses,
            SpatialPulse(
                GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=1e-6), 420e-6
            ),
            settings=dataclasses.replace(
                sim.settings,
                quadrature=None,
                components=None,
                stats=default_spec("radial"),
            ),
        )
    else:
        sim.responses = responses
    sim.settings = dataclasses.replace(
        sim.settings,
        ionisation_field_range=(1e7, 1e11),
        schedule=FixedSchedule(np.full(8, 1e-6)),
        save_planes=[8e-6],
    )
    problem = sim.translate()
    # Raise the field into the ionising regime without changing the rate table.
    field = problem.initial.field0 * 12
    engines = [
        select(name, problem, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    candidate, reference = engines
    try:
        for engine in engines:
            engine.set_initial(field, None)
        jl = runtime()
        got = np.asarray(
            jl.evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        bundle = reference._stage_scalars(0.0, 0.0)
        want = reference._nonlinear(field, bundle, np.ones_like(field))
        # Rate splines, filtered time scans and transforms accumulate rounding;
        # allow a conservative linear bound in samples and filter width.
        operations = (
            128 * problem.grids.n_to * (1 + problem.responses.plasma_integral.taps.size)
        )
        budget = higham_gamma(operations)
        assert np.linalg.norm(got - want) <= budget * np.linalg.norm(want)
        for engine in engines:
            engine.propagate_to(8e-6)
        assert np.linalg.norm(
            candidate.observe().field - reference.observe().field
        ) <= 64 * budget * np.linalg.norm(reference.observe().field)
        assert candidate.metadata()["source_evaluations"] == 49
        assert reference.metadata()["source_evaluations"] == 56
        assert budget > EPS
    finally:
        for engine in engines:
            engine.close()


def test_cartesian_envelope_rk4ip():
    """Compare a rectangular envelope grid on the exact four-pass schedule."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (500e-9, 1.2e-6), 150e-15, thg=False)
    settings = dataclasses.replace(
        simulation().settings,
        quadrature=None,
        components=None,
        integrator=LowStorageRungeKutta(),
        absorber=absorber(grid),
        schedule=FixedSchedule(np.full(8, 1e-6)),
        stats=default_spec("free"),
        save_planes=[8e-6],
    )
    sim = Simulation.from_free(
        grid,
        FreeGrid(600e-6, 8, 700e-6, 10),
        SpatialMedium("Ar", 1.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 200e-6
        ),
        settings=settings,
    )
    p = sim.translate()
    engines = [
        select(name, p, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    try:
        for engine in engines:
            engine.set_initial(p.initial.field0, None)
            engine.propagate_to(8e-6)
        got, want = [e.observe().field for e in engines]
        # Eight RK4 steps, each with four Fourier/nonlinear evaluations.
        bound = higham_gamma(32 * 128 * p.grids.n_to)
        assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
        assert all(e.metadata()["source_evaluations"] == 32 for e in engines)
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("pressure", [1.0, (1.0, 0.99)])
def test_adams_short_replay_reset_and_rejected_candidate(pressure):
    """Check resident history, compensation and an uncommitted trial."""
    if not READY:
        pytest.skip("private Julia runtime is required")

    sim = Simulation.from_capillary(
        125e-6,
        0.01,
        "Ar",
        pressure,
        lambda0=800e-9,
        wavelength_limits=(300e-9, 2e-6),
        trange=100e-15,
        fwhm=15e-15,
        energy=1e-6,
        modes=2,
        plasma=False,
        raman=False,
        integrator=Multistep(),
        backend="julia",
        save_planes=[8e-6, 0.01],
        schedule=FixedSchedule(np.full(10000, 1e-6)),
        rtol=1e-8,
        atol=1e-14,
    )
    p = sim.translate()
    engines = [
        select(name, p, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    try:
        for engine in engines:
            engine.set_initial(p.initial.field0, None)
            engine.propagate_to(8e-6)
        got, want = [e.observe().field for e in engines]
        bound = higham_gamma(128 * p.grids.n_to * 64)
        assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
        candidate = engines[0]
        assert estimate_memory(p, "julia").total == sum(
            candidate.memory_ledger().values()
        )
        ledger = candidate.metadata()["response_calls"]
        assert ledger["startup"] == 1 and ledger["predictor"] == 8
        assert ledger["accepted_steps"] == 8
        march = candidate._adams
        z, nodes, slots, order, orders = march._stencil(1e-6)
        norms = march._engine.exponential_trial(z, 1e-6, nodes, slots, order, orders)
        assert np.all(np.isfinite(norms))
        np.testing.assert_array_equal(candidate.observe().field, got)
        saved = candidate.march_snapshot()
        assert saved is not None
        candidate.propagate_to(9e-6)
        continued = candidate.observe().field
        candidate.set_initial(saved.state, None, z=saved.z)
        assert candidate.restore_march(saved) is None
        candidate.propagate_to(9e-6)
        assert np.linalg.norm(
            candidate.observe().field - continued
        ) <= bound * np.linalg.norm(continued)
        snapshot = march._engine.snapshot_rows()
        march._engine.load_rows(snapshot, p.shape)
        for name, row in snapshot.items():
            np.testing.assert_array_equal(march._engine.snapshot_rows()[name], row)
        candidate.set_initial(p.initial.field0, None)
        candidate.propagate_to(8e-6)
        assert np.linalg.norm(
            candidate.observe().field - got
        ) <= bound * np.linalg.norm(got)
    finally:
        for engine in engines:
            engine.close()


def test_phi_functions_at_zero_join_and_far():
    """Check the protected scalar recurrence against the independent oracle."""
    if not READY:
        pytest.skip("private Julia runtime is required")

    jl = runtime()
    for x in (0j, 1e-12j, -0.1 + 0.4j, 2.999j, 3.001j, -5 + 12j):
        for order in (1, 4, 7):
            got = np.asarray(jl.phi_snapshot(x, order))
            want = phi_functions(x, order)
            # Scaled series and <=3 doubling levels; include upward-recurrence
            # conditioning at the join, as in the existing phi witnesses.
            np.testing.assert_allclose(got, want, rtol=256 * EPS, atol=256 * EPS)


@pytest.mark.parametrize("thg", [False, True])
def test_scalar_raman_source(thg):
    """Compare the scalar Kerr/Raman source for both harmonic conventions."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    sim = Simulation.from_capillary(
        125e-6,
        0.01,
        "H2",
        40.0,
        temperature=294.0,
        lambda0=800e-9,
        wavelength_limits=(300e-9, 2e-6),
        trange=500e-15,
        fwhm=30e-15,
        energy=1e-6,
        modes=2,
        plasma=False,
        raman=True,
        thg=thg,
        integrator=RungeKutta(),
        backend="julia",
        save_planes=[1e-4],
    )
    p = sim.translate()
    engines = [
        select(name, p, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    try:
        candidate, reference = engines
        field = p.initial.field0
        got = np.asarray(
            runtime().evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        want = reference._nonlinear(
            field, reference._stage_scalars(0.0, 0.0), np.ones_like(field)
        )
        # Two doubled-grid Fourier transforms and two modal contractions, with
        # independent summation orders and optional analytic-signal transforms.
        bound = higham_gamma(128 * p.grids.n_to * p.transform.n_nodes)
        assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
        assert estimate_memory(p, "julia").total == sum(
            candidate.memory_ledger().values()
        )
    finally:
        for engine in engines:
            engine.close()


def test_cartesian_exact_real_root_and_complex_fallback():
    """Keep the principal root branch, signed zero and grazing arithmetic."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    jl = runtime()
    values = [
        complex(real, imag)
        for real in (-1e300, -1e-300, -1.0, -0.0, 0.0, 1e-300, 1.0, 1e300)
        for imag in (-0.0, 0.0)
    ] + [complex(3, 4), complex(-3, 4), complex(-3, -4), complex(1, 1e-30)]
    for value in values:
        got = complex(jl.free_root(value, True))
        want = complex(np.sqrt(np.complex128(value)))
        # One real square root, or the same complex-library fallback; allow a
        # few rounding units for the independent library implementations.
        assert abs(got - want) <= 8 * EPS * abs(want)
        if want == 0:
            assert np.signbit(got.real) == np.signbit(want.real)
            assert np.signbit(got.imag) == np.signbit(want.imag)
