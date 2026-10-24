# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare accepted-field population observations with the scalar reference."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_julia import READY, simulation

from bounds import higham_gamma
from solprop.backends import select
from solprop.backends.julia import JuliaBackend
from solprop.backends.julia._runtime import runtime
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


@pytest.mark.parametrize("averaged", [False, True])
@pytest.mark.parametrize("scale", [0.0, 12.0])
def test_population_statistics_and_continuation(averaged: bool, scale: float) -> None:
    """Observe the accepted field without disturbing source counts or FSAL."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    sim = simulation(
        kind="averaged" if averaged else "resolved", trange=250e-15, modes=2
    )
    sim.responses = (*sim.responses, PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)))
    sim.settings = dataclasses.replace(
        sim.settings,
        ionisation_field_range=(1e7, 1e11),
        schedule=FixedSchedule(np.full(10, 1e-6)),
        save_planes=[1e-5],
    )
    problem = sim.translate()
    candidate = select("julia", problem, grid=sim.grid, modes=sim.modes)
    reference = select("numpy_ref", problem, grid=sim.grid, modes=sim.modes)
    assert isinstance(candidate, JuliaBackend)
    assert isinstance(reference, NumpyRefBackend)
    field = scale * problem.initial.field0
    try:
        for engine in (candidate, reference):
            engine.set_initial(field, None)
        spec = StatsSpec(
            names=(
                "species_density",
                "species_electron_density",
                "species_ionisation",
                "species_ionisation_energy",
                "species_peak_ionisation_rate",
                "electron_density",
                "peak_ionisation_rate",
                "plasma_undershoot",
                "plasma_work_defect",
                "plasma_boundary_level",
            )
        )
        before = candidate.observe().field
        calls = candidate.metadata()["source_evaluations"]
        got, want = candidate.stats(spec), reference.stats(spec)
        # Transform, rate-spline and filtered-prefix rounding scale linearly in
        # samples. Ratio diagnostics also need an absolute allowance near zero.
        bound = higham_gamma(128 * problem.grids.n_to)
        assert bound > 0
        for name in spec.names:
            np.testing.assert_allclose(got[name], want[name], rtol=bound, atol=bound)
        np.testing.assert_array_equal(candidate.observe().field, before)
        assert candidate.metadata()["source_evaluations"] == calls
        raw = runtime().population_snapshot(
            candidate._engine, field[0, :, 0, :].T.copy()
        )
        parts = np.asarray(raw[2])
        reference_parts = np.zeros(6)
        for nodes in reference._plasma_nodes(field[0]):
            reference_parts = np.maximum(
                reference_parts,
                passes.plasma_integration_diagnostics(
                    nodes,
                    problem.responses.plasma_channels,
                    (0.0,),
                    problem.grids.dt_fine,
                    problem.responses.plasma_integral,
                ),
            )
        # Compare numerator/denominator parts before their potentially ill-conditioned
        # ratios; use the pair's natural scale, not one mixed-unit vector norm.
        for first in (0, 2, 4):
            magnitude = max(reference_parts[first : first + 2])
            np.testing.assert_allclose(
                parts[first : first + 2],
                reference_parts[first : first + 2],
                rtol=bound,
                atol=bound * magnitude,
            )
        candidate.propagate_to(5e-6)
        candidate.stats(spec)
        candidate.propagate_to(1e-5)
        observed = candidate.observe().field
        candidate.set_initial(field, None)
        candidate.propagate_to(1e-5)
        untouched = candidate.observe().field
        assert observed is not None and untouched is not None
        assert np.linalg.norm(observed - untouched) <= 70 * bound * np.linalg.norm(
            untouched
        )
    finally:
        candidate.close()
        reference.close()
