# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check exact-zero plasma columns and reuse after an ionising source."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_julia import READY, simulation

from bounds import higham_gamma
from solprop.backends import select
from solprop.backends.julia._runtime import runtime
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse


@pytest.mark.parametrize("preionisation", [0.0, 0.03])
def test_zero_rate_after_ionising_source(
    monkeypatch: pytest.MonkeyPatch, preionisation: float
) -> None:
    """Retain pre-existing electrons and clear reused zero-current outputs."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    monkeypatch.setenv("SOLPROP_JULIA_SKIP_ZERO_PLASMA", "1")
    sim = simulation(trange=250e-15)
    sim.responses = (
        *sim.responses,
        PlasmaResponse("Ar", OptmatRate("Ar", 800e-9), preionisation=preionisation),
    )
    # Isolate current reuse from the temporal absorber. At the weak input the
    # pulse is almost entirely outside its loss support, so an error relative
    # only to the tiny resulting loss does not bound inverse-FFT cancellation.
    # Active absorbers have separate source and propagation checks.
    sim.settings = dataclasses.replace(
        sim.settings, ionisation_field_range=(1e7, 1e11), absorber=None
    )
    problem = sim.translate()
    candidate, reference = [
        select(name, problem, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    try:
        # The strong source populates scratch before two zero-rate evaluations;
        # the final strong source checks that skipping has not damaged reuse.
        bound = higham_gamma(
            128 * problem.grids.n_to * (1 + problem.responses.plasma_integral.taps.size)
        )
        assert bound > 0
        for scale in (36.0, 0.0, 1e-4, 36.0):
            field = scale * problem.initial.field0
            got = np.asarray(
                runtime().evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
            ).T[None, :, None, :]
            want = reference._nonlinear(
                field, reference._stage_scalars(0.0, 0.0), np.ones_like(field)
            )
            assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
            if scale == 0:
                assert np.all(got == 0)
            if scale == 36:
                assert np.max(np.asarray(candidate._engine.work.plasma.peaks)) > 0
            else:
                assert np.all(np.asarray(candidate._engine.work.plasma.peaks) == 0)
                np.testing.assert_allclose(
                    np.asarray(candidate._engine.work.plasma.fractions),
                    preionisation,
                    rtol=bound,
                    atol=bound,
                )
    finally:
        candidate.close()
        reference.close()
