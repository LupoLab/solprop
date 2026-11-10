# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a session reports beside its events: records, method history and tickets.

The service loop reads a session's execution and monitor records, its method-history
snapshot at a pending observation, and a ticket for the committed state after a failure
outside the session (SEP 47 sections 5.4 and 5.5). Each executor answers the same
tests, and an export converted in place for one recipe stays the consumer's when the
next consumer asks for another.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from sessions.executors import (
    CONSUMER,
    STORED,
    ScriptedError,
    soliton,
    spy_monitor,
)
from solprop.backends.events import ObservationKind, ObservationReady, Outcome
from solprop.backends.session import RunSession
from solprop.stepper import Multistep

# The soliton's 1 ps window is tight for a monitor's guards; the window is not tested.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")


def _first_plane(session: RunSession):
    """Acknowledge every batch before the first save plane; return its pending batch.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        A session that has not advanced.

    Returns
    -------
    tuple
        The pending batch and its plane observation.
    """
    while True:
        batch = session.advance()
        for event in batch.events:
            if (
                isinstance(event, ObservationReady)
                and event.kind is ObservationKind.PLANE
            ):
                return batch, event
        assert batch.terminal is None, "the run ended before its first plane"
        session.acknowledge(batch.receipt)


def test_a_second_recipe_after_an_in_place_conversion_leaves_the_first_alone(start):
    """SEP 47 section 5.4: an export handed out is never written again.

    The plane store's recipe comes first, which a session whose driver owns the
    observed field converts where it is; a repeat of it is the same bytes, and the
    consumer recipe that follows observes the raw field again rather than undo the
    conversion under the store's feet.
    """
    session = start(soliton()).session
    _, plane = _first_plane(session)
    stored = session.export(plane.ticket, STORED)
    kept = stored.copy()
    assert session.export(plane.ticket, STORED).tobytes() == kept.tobytes()
    consumer = np.array(session.export(plane.ticket, CONSUMER))
    out = np.empty_like(kept)
    assert session.export(plane.ticket, STORED, out=out) is out
    assert stored.tobytes() == kept.tobytes()
    assert out.tobytes() == kept.tobytes()
    shift = np.where(np.arange(kept.shape[-1]) % 2, -1.0, 1.0)
    np.testing.assert_array_equal(kept, shift * np.conj(consumer))


def test_a_runge_kutta_session_reports_no_method_history(start, executor):
    """A Runge-Kutta march keeps no history: no snapshot, no size, nothing refused."""
    session = start(soliton()).session
    _, plane = _first_plane(session)
    assert session.march_snapshot(plane.ticket) is None
    assert session.march_snapshot_nbytes() is None
    # The driver-facing answers the run layer reads, which every executor gives.
    assert session.march_export_refusal() is None  # pyright: ignore[reportAttributeAccessIssue]
    assert session.march_refusal is None  # pyright: ignore[reportAttributeAccessIssue]
    record = session.execution_record()
    assert isinstance(record, Mapping)
    assert record
    assert session.monitor_record() is None


def test_an_adams_session_reports_its_history_at_the_pending_plane(start, executor):
    """SEP 29 and 47 section 5.5: an Adams march's history is snapshotted at a plane."""
    if executor.scripted:
        pytest.skip("a scripted driver runs no Adams march")
    if executor.name == "native-session-low-storage":
        pytest.skip("this executor replays every problem by the low-storage march")
    session = start(soliton(integrator=Multistep())).session
    _, plane = _first_plane(session)
    assert session.march_export_refusal() is None  # pyright: ignore[reportAttributeAccessIssue]
    snapshot = session.march_snapshot(plane.ticket)
    assert snapshot is not None
    assert snapshot.z == plane.z
    nbytes = session.march_snapshot_nbytes()
    assert isinstance(nbytes, int)
    assert nbytes > 0


def test_a_monitored_session_reports_its_monitor_record(start, executor):
    """SEP 21: a session that feeds a monitor reports its record from the launch.

    A session that owns its monitor natively leaves the request's monitor untouched,
    so only its record shows what it did (SEP 47 section 6.4).
    """
    problem = soliton()
    log: list[tuple[str, float]] = []
    session = start(problem, problem.request.for_run(monitor=spy_monitor(log))).session
    _first_plane(session)
    record = session.monitor_record()
    assert isinstance(record, Mapping)
    assert record
    assert bool(log) is not executor.native_monitor


def test_a_closed_session_offers_no_pending_ticket(start):
    """A failure outside a closed session leaves nothing to checkpoint."""
    session = start(soliton()).session
    _first_plane(session)
    session.close()
    assert session.pending_ticket() is None  # pyright: ignore[reportAttributeAccessIssue]


def test_a_poisoned_session_offers_no_pending_ticket(start_scripted):
    """SEP 47 section 6.7: a state left in flight cannot be checkpointed."""
    session = start_scripted(soliton(), fail_at=3, in_flight=True).session
    while True:
        batch = session.advance()
        if batch.terminal is not None:
            break
        session.acknowledge(batch.receipt)
    assert batch.terminal.outcome is Outcome.FAILED
    assert isinstance(batch.terminal.error, ScriptedError)
    assert batch.terminal.ticket is None
    assert session.pending_ticket() is None  # pyright: ignore[reportAttributeAccessIssue]
