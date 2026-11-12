# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""An unresolved angular source refuses in the statistics path too (SEP 43).

``numba_cpu``'s ``test_host_refuses_unresolved_angular_source_and_recovers`` held both
passes: the source and the populations refused, wrote nothing, and a zero field then
recovered. ``test_envelope_plasma.py`` holds the source; this is the other.
"""

from __future__ import annotations

import numpy as np
import pytest
from session_runs import drain, run
from solprop.backends.events import Outcome
from solprop.backends.session import Start
from solprop.exceptions import ConvergenceError
from solprop.stats import StatsSpec
from test_envelope_plasma import envelope_case, envelope_options, terms

import solprop_native.backend as adapter
from solprop_native import NativeBackend, _core
from solprop_native.backend import (
    _lower,  # pyright: ignore[reportPrivateUsage]
    _LowerPlan,  # pyright: ignore[reportPrivateUsage]
)

SPEC = StatsSpec(names=("electron_density", "peak_ionisation_rate"))


@pytest.mark.parametrize(
    "geometry",
    ["mode_averaged", "free"],
)
def test_the_statistics_refuse_and_a_quiet_field_recovers(
    monkeypatch: pytest.MonkeyPatch, geometry: str
) -> None:
    problem, run_request, grid = envelope_case(geometry, 1, mixture=False)
    payload = _lower(
        problem,
        run_request,
        _LowerPlan(terms=terms(problem), node_chunk=envelope_options(geometry).chunk),
    )
    nodes = payload.plasma.envelope_rule_nodes.reshape(-1, 4, 512)
    # As the source's test: alternating incompatible rules must fail refinement.
    for level in (1, 3):
        order = 64 << level
        nodes[:, level, :order] = 0.5 * (nodes[:, level, :order] + 1)
    payload.plasma.envelope_rule_nodes = nodes

    def corrupt_payload(*args: object, **keywords: object) -> _core.Problem:
        return payload

    monkeypatch.setattr(adapter, "_lower", corrupt_payload)
    # The launch plane's statistics meet the bad rule before any step.
    run_request = run_request.for_run(stats=SPEC)
    owner = NativeBackend(
        problem, run_request, grid=grid, options=envelope_options(geometry)
    )
    try:
        loud = np.asarray(10.0 * problem.initial.field0, dtype=np.complex128)
        session = owner.prepare(problem, run_request).start(
            Start(field=loud, z=0.0), owns=False
        )
        try:
            terminal = drain(session)
        finally:
            session.close()
        assert terminal.outcome is Outcome.FAILED
        assert isinstance(terminal.error, ConvergenceError), terminal.error
        assert "angular refinement" in str(terminal.error)
        assert terminal.z == 0.0
        # A quiet field below threshold: no refinement, and no failure left behind.
        quiet = Start(field=np.zeros_like(problem.initial.field0), z=0.0)
        launched = run(owner, problem, run_request, quiet).all_planes()[0]
        assert np.all(np.asarray(launched.stats["peak_ionisation_rate"]) == 0.0)
    finally:
        owner.close()
