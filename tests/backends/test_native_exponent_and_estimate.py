# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A separable route's compiled exponent, and a varying operator's estimate (SEP 43).

From ``test_numba_adams_passes.py`` and ``test_numba_low_storage.py``.
"""

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("solprop_native")


from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

from backends._session_probe_b2 import run_session
from backends.forced_thread_cases import route_adams_problem
from backends.low_storage_cases import varying_capillary, with_mode
from bounds import EPS, assert_within
from solprop.backends.memory import (
    reference_base_memory,
)
from solprop.stepper import FixedSchedule
from solprop.translate.linop import delta_phi

ELEMENT_BOUND = 64 * EPS
"""One element's arithmetic, a few dozen operations, contracted differently at most."""


def test_a_separable_exponent_is_the_host_volume():
    """The compiled exponent's element formula against the host's ``delta_phi``.

    ``numba_cpu``'s test read the compiled exponent. Native's ``_delta_phi`` is the
    host's own on this route, so comparing that would be a self-comparison, and its
    exponent stays in the engine; it is read here through what it does. With no
    response, one exponential Adams step of length ``h`` is ``exp(x)`` of the launch,
    element by element, so ``x`` moves the state by itself: the retired ``64 eps``
    of the largest exponent, on the launch's own scale, plus the exponential's rounding.
    Weaker in that a phase error on a bin the launch leaves empty is not seen.
    """
    _, problem, run_request = route_adams_problem("cartesian", "exponential")
    end = 3e-6
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            response_order=problem.responses.response_order[-1:],
            kerr_coefficients=None,
            # No collar at all: a split one would act after the exponent's own step.
            temporal_absorber=None,
            temporal_absorber_fine=None,
        ),
    )
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule(np.array([end])),
            save_planes=np.array([end]),
        )
    )
    grid_simulation, _, _ = route_adams_problem("cartesian", "exponential")
    backend = NativeBackend(problem, run_request, grid=grid_simulation.grid)
    try:
        # The backend's session: its launch plane is the loaded state, masked to the
        # band, and its last the state after the one step.
        landed = run_session(backend, problem, run_request)
    finally:
        backend.close()
    assert [event.z for event in landed.run.observations] == [0.0, end]
    loaded, got = landed.run.fields[0], landed.field
    # The exponent keeps the operator tables' leading axis of one.
    exponent = delta_phi(problem.linear, 0.0, end)[0]
    assert_within(
        got,
        np.exp(exponent) * loaded,
        ELEMENT_BOUND * float(np.max(np.abs(exponent))) + 16 * EPS,
        what="exponent",
    )
    assert float(np.max(np.abs(exponent))) > 1.0


@pytest.mark.medium
def test_a_varying_operator_holds_its_pair_or_none():
    """A varying modal operator: ``P`` and ``Q`` held, or formed per element.

    The ledger's march rows are the reference's where the table is held, and native
    holds no scratch recomputing.
    """
    simulation = varying_capillary()
    for mode, held in (("tabulate", 2), ("recompute", 0)):
        problem, run_request = with_mode(
            simulation.translate(), simulation._run_request(), mode
        )
        field = 16 * math.prod(problem.shape)
        backend = NativeBackend(
            problem, run_request, grid=simulation.grid, modes=simulation.modes
        )
        try:
            ledger = backend.memory_ledger()
            chunk = backend.node_chunk
        finally:
            backend.close()
        assert ledger["propagators"] == held * field
        reference = reference_base_memory(
            problem,
            run_request.stepper,
            tabulating=mode == "tabulate",
            embedded=not np.array_equal(
                problem.transform.projection, problem.transform.projection_coarse
            ),
            noise=False,
            statistics_weight=0,
            spatial_absorber=0,
            node_chunk=chunk,
        )
        assert ledger["state"] == reference["state"]
        if mode == "tabulate":
            assert ledger["propagators"] == reference["propagators"]
