# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check omission of exactly absent modal and radial additive losses."""

import dataclasses

import numpy as np
import pytest
from tests.backends.test_julia import READY, simulation

from bounds import higham_gamma
from solprop.backends import select
from solprop.backends.julia._runtime import runtime
from solprop.stepper import FixedSchedule


@pytest.mark.parametrize("kind", ["resolved", "averaged", "radial"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_modal_absorber_source_and_steps(
    monkeypatch: pytest.MonkeyPatch, kind: str, active: bool, split: bool
) -> None:
    """Preserve active losses and skip absent ones under both temporal schemes."""
    if not READY:
        pytest.skip("private Julia runtime is required")
    monkeypatch.setenv("SOLPROP_JULIA_SKIP_ZERO_MODAL_ABSORBER", "1")
    sim = simulation(
        kind=kind,
        absorber_scheme="split_step" if split else "nonlinear",
        schedule=FixedSchedule(np.full(10, 1e-5)),
    )
    original = sim.translate()
    if active:
        problem = original
    else:
        transform = original.transform
        if kind == "radial":
            assert transform.geometry is not None
            transform = dataclasses.replace(
                transform,
                geometry=dataclasses.replace(
                    transform.geometry,
                    absorber_rate=np.zeros_like(transform.geometry.absorber_rate),
                ),
            )
        problem = dataclasses.replace(
            original,
            transform=transform,
            responses=dataclasses.replace(
                original.responses, temporal_absorber=None, temporal_absorber_fine=None
            ),
            flags=dataclasses.replace(original.flags, has_temporal_absorber=False),
        )
    problem = dataclasses.replace(
        problem,
        transform=dataclasses.replace(
            problem.transform, node_chunk=min(7, problem.transform.n_nodes)
        ),
    )
    candidate, reference = [
        select(name, problem, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    try:
        field = problem.initial.field0
        for engine in (candidate, reference):
            engine.set_initial(field, None)
        got = np.asarray(
            runtime().evaluate_b(candidate._engine, field[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        want = reference._nonlinear(
            field, reference._stage_scalars(0.0, 0.0), np.ones_like(field)
        )
        # The source combines FFTs, cubic products and modal/radial sums; ten
        # DOPRI steps use no more than seventy source evaluations.
        bound = higham_gamma(128 * problem.grids.n_to * problem.transform.n_nodes)
        assert bound > 0 and np.linalg.norm(want) > 0
        assert np.linalg.norm(got - want) <= bound * np.linalg.norm(want)
        for engine in (candidate, reference):
            engine.propagate_to(1e-4)
        got, want = candidate.observe().field, reference.observe().field
        assert got is not None and want is not None
        assert np.linalg.norm(got - want) <= 70 * bound * np.linalg.norm(want)
        assert bool(candidate._engine.config.modal_in_stage_absorber) == (
            active and (kind == "radial" or not split)
        )
    finally:
        candidate.close()
        reference.close()
