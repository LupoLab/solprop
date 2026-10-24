# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the scalar modal envelope extension with short GNLSE witnesses."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_julia import READY

from bounds import higham_gamma
from solprop.backends import select
from solprop.backends.julia import JuliaBackend
from solprop.backends.julia._runtime import runtime
from solprop.backends.memory import estimate_memory
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.stepper import FixedSchedule, Multistep, RungeKutta


@pytest.mark.parametrize("raman", [False, True])
@pytest.mark.parametrize("method", ["rk", "adams"])
@pytest.mark.parametrize("split", [False, True])
def test_gnlse_source_short_window_and_reset(raman, method, split):
    """Check Kerr, Raman/shock, complex transforms, masking and both absorbers."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    sim = Simulation.from_gnlse(
        0.11,
        1e-3,
        [0.0, 0.0, -11.83e-27],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=100.0,
        pulseshape="sech",
        raman=raman,
        shock=raman,
        integrator=RungeKutta() if method == "rk" else Multistep(),
        schedule=FixedSchedule(np.full(8, 1e-5)) if method == "rk" else None,
        dz_init=1e-5,
        dz_max=1e-5,
        save_planes=[8e-5],
        rtol=1e-9,
        atol=1e-14,
        backend="julia",
    )
    sim.settings = dataclasses.replace(
        sim.settings, absorber_scheme="split_step" if split else "nonlinear"
    )
    problem = sim.translate()
    engines = [
        select(name, problem, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    candidate, reference = engines
    assert isinstance(candidate, JuliaBackend)
    assert isinstance(reference, NumpyRefBackend)
    try:
        field = problem.initial.field0
        for engine in engines:
            engine.set_initial(field, None)
        got_source = np.asarray(
            runtime().evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        want_source = reference._nonlinear(
            field, reference._stage_scalars(0.0, 0.0), np.ones_like(field)
        )
        # Two temporal FFTs, a doubled Raman convolution, and scalar contractions.
        # Bound each transform conservatively by a linear sample-count budget.
        budget = higham_gamma(128 * problem.grids.n_to)
        assert np.linalg.norm(got_source - want_source) <= budget * np.linalg.norm(
            want_source
        )
        for engine in engines:
            engine.propagate_to(8e-5)
        got, want = (engine.observe().field for engine in engines)
        assert got is not None and want is not None
        # At most 64 source/flow evaluations in this weak eight-step witness.
        assert np.linalg.norm(got - want) <= 64 * budget * np.linalg.norm(want)
        assert np.all(got[..., ~problem.grids.active_mask] == 0)
        assert estimate_memory(problem, "julia").total == sum(
            candidate.memory_ledger().values()
        )
        candidate.set_initial(field, None)
        candidate.propagate_to(8e-5)
        reset = candidate.observe().field
        assert reset is not None
        assert np.linalg.norm(reset - got) <= (64 * budget * np.linalg.norm(got))
        for value in candidate.stats(problem.stats_spec).values():
            assert np.all(np.isfinite(value))
    finally:
        for engine in engines:
            engine.close()
