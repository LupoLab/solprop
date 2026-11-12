# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A failure outside a march call keeps its solprop error class (SEP 47 section 5.5).

The native session catches an exception raised outside the march's own error returns
-- a plane's statistics evaluation, say -- and ends the run with it. Its class is the
one the marches give the same native exception (``rk_march.cpp``, ``adams_march.cpp``):
an unresolved angular source is a ``ConvergenceError``, as the reference raises, not a
``RuntimeError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from solprop.backends.events import Terminal
from solprop.backends.session import Start
from solprop.exceptions import ConvergenceError
from test_envelope_plasma import envelope_case, envelope_options, terms

import solprop_native.backend as adapter
from solprop_native import NativeBackend, _core
from solprop_native.backend import (
    _lower,  # pyright: ignore[reportPrivateUsage]
    _LowerPlan,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("geometry", ["mode_averaged", "free"])
def test_an_unresolved_launch_statistic_ends_the_run_as_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch, geometry: str
) -> None:
    problem, run_request, grid = envelope_case(geometry, 1, mixture=False)
    payload = _lower(
        problem,
        run_request,
        _LowerPlan(terms=terms(problem), node_chunk=envelope_options(geometry).chunk),
    )
    nodes = payload.plasma.envelope_rule_nodes.reshape(-1, 4, 512)
    # Alternating incompatible rules fail refinement (`test_envelope_plasma.py`).
    for level in (1, 3):
        order = 64 << level
        nodes[:, level, :order] = 0.5 * (nodes[:, level, :order] + 1)
    payload.plasma.envelope_rule_nodes = nodes

    def corrupt_payload(*args: object, **keywords: object) -> _core.Problem:
        return payload

    monkeypatch.setattr(adapter, "_lower", corrupt_payload)
    owner = NativeBackend(
        problem, run_request, grid=grid, options=envelope_options(geometry)
    )
    start = Start(
        field=np.asarray(10.0 * problem.initial.field0, dtype=np.complex128),
        z=float(problem.initial.z),
    )
    terminal = None
    with owner.prepare(problem, run_request).start(start) as session:
        while terminal is None and not session.finished:
            batch = session.advance()
            terminal = next(
                (event for event in batch.events if isinstance(event, Terminal)), None
            )
            session.acknowledge(batch.receipt)
    assert terminal is not None
    # The launch plane's populations refuse before any step is taken.
    assert isinstance(terminal.error, ConvergenceError)
    assert "angular refinement" in str(terminal.error)
