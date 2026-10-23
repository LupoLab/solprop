# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A collar of zero temporal rates compiles no temporal pass (SEP 22 section 3, 5.4).

The spectral-only collar is compared with today's translation of the same collar,
which handed the kernels arrays of zeros. Under the in-stage scheme those added exact
zeros, so the reference backend's run is unchanged exactly; under the split scheme
they cost a transverse transform pair and the recycled stage, so the runs agree to a
rounding bound fixed before they are made (``docs/dev/testing.md`` section 3).
"""

import dataclasses

import numpy as np
import pytest

from bounds import assert_within
from driver.test_field_free_planes import cartesian, replay_bound
from solprop.backends.registry import available, select
from solprop.boundaries import absorber
from solprop.driver.run import run_problem
from solprop.exceptions import TranslationError
from solprop.output import MemoryOutput
from solprop.output.schema import serialise


def spectral_only(backend: str, scheme: str, **changes):
    simulation = cartesian(backend)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        # A reference length long enough that the spatial collar, which follows it,
        # does not bound the replayed step.
        absorber=absorber(simulation.grid, boundary_length=0.02, temporal=False),
        absorber_scheme=scheme,
        **changes,
    )
    return simulation


def today(problem):
    """Return the problem as translated before SEP 22: zero arrays at the seam."""
    coarse = np.zeros(problem.grids.n_t)
    fine = np.zeros(problem.grids.n_to)
    coarse.setflags(write=False)
    fine.setflags(write=False)
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, temporal_absorber=coarse, temporal_absorber_fine=fine
        ),
    )


def run(simulation, problem):
    backend = select(simulation.settings.backend, problem, grid=simulation.grid)
    output = MemoryOutput()
    try:
        backend.set_initial(problem.initial.field0, None)
        fsal = getattr(backend, "_fsal", None)
        run_problem(
            problem,
            backend,
            output,
            metadata=simulation.metadata(),
            progress="none",
        )
    finally:
        backend.close()
    return output, fsal


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_spectral_only_collar_runs_as_today_without_the_pass(backend, scheme):
    simulation = spectral_only(backend, scheme)
    problem = simulation.translate()
    assert problem.responses.temporal_absorber is None
    assert problem.responses.temporal_absorber_fine is None
    bound = replay_bound(simulation)
    elided, recycled = run(simulation, problem)
    before, held = run(simulation, today(problem))
    if backend == "numpy_ref" and scheme == "nonlinear":
        # Exact zeros were added in every stage: the reference's run is unchanged
        # bit for bit, the structural contract SEP 22 section 7 asks for.
        assert np.array_equal(elided["field"], before["field"])
    assert_within(elided["field"], before["field"], bound, what="field")
    assert np.array_equal(elided["step_history/h"], before["step_history/h"])
    if backend == "numba_cpu":
        # The split pass forbade recycling the last stage; without it the stage is
        # recycled under either scheme.
        assert recycled is True
        assert held is (scheme == "nonlinear")


def test_the_record_names_the_spectral_only_collar():
    simulation = spectral_only("numpy_ref", "split_step")
    record = serialise(simulation.metadata().configuration["absorber"])
    assert isinstance(record, dict)
    assert record["temporal"] is False
    assert simulation.metadata().configuration["absorber_scheme"] == "split_step"


def test_the_split_scheme_s_refusal_of_interpolated_planes_needs_a_pass():
    # A collar that absorbs nothing in time has no per-step factor to miss.
    spectral_only(
        "numpy_ref", "split_step", schedule=None, interpolate_planes=True
    ).translate()
    simulation = cartesian("numpy_ref", replay=False, interpolate=True)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        absorber=absorber(simulation.grid, boundary_length=0.02),
        absorber_scheme="split_step",
    )
    with pytest.raises(TranslationError, match="per-step temporal collar"):
        simulation.translate()
