# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A failure outside a march call keeps its solprop error class (SEP 47 section 5.5).

The native session catches an exception raised outside the march's own error returns
-- a plane's statistics evaluation, say -- and ends the run with it. Its class is the
one the marches give the same native exception (``rk_march.cpp``, ``adams_march.cpp``):
an unresolved angular source is a ``ConvergenceError``, as the reference raises, not a
``RuntimeError``. A native refusal the adapter recognises by its text keeps that text
in the C++ sources, so that a reworded message cannot lose its solprop class unseen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from solprop.backends.events import Terminal
from solprop.backends.session import Start
from solprop.exceptions import ConvergenceError
from test_envelope_plasma import envelope_case, envelope_options, terms

import solprop_native.backend as adapter
import solprop_native.session as native_session
from solprop_native import NativeBackend, _core, _statistics
from solprop_native._lowering import (
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
        # The backend's own lowering, its set-once data included, with the bad rule.
        built = _lower(*args, **keywords)  # pyright: ignore[reportArgumentType]
        built.plasma.envelope_rule_nodes = nodes
        return built

    # Patched where the backend looks it up: backend.py imports it from _lowering.
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


def test_every_refusal_the_adapter_recognises_by_text_is_raised_by_the_cpp() -> None:
    root = Path(__file__).parents[1]
    sources = {
        path: path.read_text(encoding="utf-8")
        for folder in ("src", "include")
        for path in sorted((root / folder).rglob("*"))
        if path.suffix in {".cpp", ".hpp", ".inc"}
    }
    # The C++ tree was found: over a hundred files, the session's among them.
    assert len(sources) > 50
    assert root / "src" / "session" / "session.cpp" in sources
    markers = [
        *(marker for marker, _ in _statistics._TRANSLATED),  # pyright: ignore[reportPrivateUsage]
        native_session._POWER_WEIGHTS_REFUSAL,  # pyright: ignore[reportPrivateUsage]
    ]
    assert len(markers) == 8
    missing = [
        marker
        for marker in markers
        if not any(marker in text for text in sources.values())
    ]
    assert missing == []
