# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's split-collar crossing cap is the reference's (solprop #440).

The native session keeps the boundary monitor (SEP 47 section 6.4), whose split collar
reads a crossing cap at each check (``BoundaryMonitor.step_cap``).
``NativeSession.monitor_cap`` reads it after every march, outside the record whose
layout is the reference's; here it is held to the reference monitor's after the same
fixed march.
"""

import math
from typing import Any, cast

import pytest

from capability_routes import LENGTH
from monitor_cases import MARCHES, probe, record_after_march, with_copy
from solprop.backends.events import Terminal
from solprop.backends.registry import select
from solprop.backends.session import Start

pytest.importorskip("solprop_native.session")


def _native_cap(simulation: Any) -> float:
    """Run the edge-reaching launch through a native session; return its cap."""
    problem = simulation.translate()
    request = simulation._run_request().for_run(monitor=simulation._boundary_monitor())
    backend = select(
        "native",
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options(),
    )
    start = Start(
        field=with_copy(problem.initial.field0, simulation.grid.n_t),
        z=float(problem.initial.z),
    )
    with backend.prepare(problem, request).start(start) as session:
        terminal = None
        while terminal is None:
            batch = session.advance()
            terminal = next(
                (event for event in batch.events if isinstance(event, Terminal)), None
            )
            session.acknowledge(batch.receipt)
        assert terminal.error is None
        assert float(terminal.z) == pytest.approx(LENGTH)
        # The native session's own reading (`NativeSession.monitor_cap`).
        return cast("Any", session).monitor_cap()


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_crossing_cap_after_a_march_is_the_reference_s(march: str) -> None:
    simulation = probe("gnlse", "split_step", MARCHES[march])
    theirs = record_after_march(simulation, "numpy_ref")["step_cap"]
    ours = _native_cap(simulation)
    assert math.isfinite(theirs) and math.isfinite(ours)
    # The cap is a bucket's: states agreeing to rounding fill the same buckets, but a
    # share on the 1e-6 threshold may tip into the next, a factor of 10**(9/128)
    # apart (`test_monitor_reduction.py`).
    assert ours == pytest.approx(theirs, rel=0.18)
