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

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from solprop.backends.events import Terminal
from solprop.backends.session import Start
from solprop.exceptions import ConvergenceError
from test_envelope_plasma import envelope_case, envelope_options, terms

import solprop_native.backend as adapter
from solprop_native import NativeBackend
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("geometry", ["mode_averaged", "free"])
def test_an_unresolved_launch_statistic_ends_the_run_as_a_convergence_error(
    monkeypatch: pytest.MonkeyPatch, geometry: str
) -> None:
    problem, run_request, grid = envelope_case(geometry, 1, mixture=False)
    payload = _native_problem(
        problem,
        run_request,
        terms(problem),
        node_chunk=envelope_options(geometry).chunk,
    )
    nodes = cast(npt.NDArray[np.float64], payload["envelope_rule_nodes"]).copy()
    # Alternating incompatible rules fail refinement (`test_envelope_plasma.py`).
    for level in (1, 3):
        order = 64 << level
        nodes[:, level, :order] = 0.5 * (nodes[:, level, :order] + 1)
    payload["envelope_rule_nodes"] = nodes

    def corrupt_payload(*args: object, **keywords: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(adapter, "_native_problem", corrupt_payload)
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
