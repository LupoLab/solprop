# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep copied Julia exports bound to their ticket across Python callbacks."""

from dataclasses import replace

import numpy as np
import pytest

from backends.julia.test_session_boundary import IDENTITY, _frame, _start, boundary
from solprop.backends.events import ExportTicket
from solprop.backends.julia import _calls
from solprop.backends.julia._session import JuliaSession
from solprop.backends.session import SessionError

__all__ = ["boundary"]


def _read(session: JuliaSession, ticket: ExportTicket, phase: bool) -> np.ndarray:
    """Read one detached field or phase from the actual Python session."""
    result = session.noise_phase(ticket) if phase else session.export(ticket, IDENTITY)
    assert result is not None
    return result


def _prepare(boundary):
    """Supply phase copying to the existing behavioral engine fixture."""

    def phase_copy(output, owner, generation):
        assert owner.pending and generation == owner.generation
        output.fill(owner.frame.report.z)

    setattr(boundary.engine.module, "copy_session_noise_phase!", phase_copy)
    session = _start(boundary, _frame(1), _frame(2, z=0.25))
    # Numerical draw lowering has separate tests. These tests exercise receipt,
    # copy and cache ownership using the same scripted boundary as field exports.
    session._has_noise = True
    session._group_chunk = 9
    first = session.advance()
    ticket = session.pending_ticket()
    assert ticket is not None
    return session, first, ticket


def _replace_as_group_terminal(session: JuliaSession) -> ExportTicket:
    """Replace a selected coordinate without requiring a new numerical generation."""
    previous = session._report
    assert previous is not None
    frame = _frame(7, z=0.5)
    frame.report = replace(frame.report, generation=previous.generation)
    raw = session._engine.borrow()
    raw.frame = frame
    raw.finished = True
    session._group_event(
        frame.report, 9, error=RuntimeError("external consumer failure")
    )
    ticket = session.pending_ticket()
    assert ticket is not None and ticket.generation == previous.generation
    return ticket


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("callback", ["advance", "terminal", "close"])
def test_callback_after_copy_cannot_publish_into_a_replacement_cache(
    boundary, monkeypatch, phase, callback
):
    session, first, ticket = _prepare(boundary)
    original = _calls.call_void
    name_to_interrupt = "copy_session_noise_phase!" if phase else "export_session!"
    newer: list[tuple[ExportTicket, np.ndarray]] = []
    called = False

    def after_copy(runtime, name, *args):
        nonlocal called
        original(runtime, name, *args)
        if name != name_to_interrupt or called:
            return
        called = True
        # Julia's guarded entry has returned. A Python signal handler can run at
        # this same boundary, before the outer export publishes its copied array.
        if callback == "close":
            session.close()
            return
        if callback == "advance":
            session.acknowledge(first.receipt)
            session.advance()
            replacement = session.pending_ticket()
            assert replacement is not None
        else:
            replacement = _replace_as_group_terminal(session)
        newer.append((replacement, _read(session, replacement, phase)))

    monkeypatch.setattr(_calls, "call_void", after_copy)
    old = _read(session, ticket, phase)
    assert called and not old.flags.writeable
    np.testing.assert_array_equal(old, 0.0 if phase else 1.0 + 0.5j)
    if callback == "close":
        assert session.closed and not session._exports and not session._phases
    else:
        replacement, expected_cache = newer[0]
        got = _read(session, replacement, phase)
        assert got is expected_cache
        expected_z = 0.25 if callback == "advance" else 0.5
        np.testing.assert_array_equal(
            got, expected_z if phase else 1 + expected_z + 0.5j
        )
        monkeypatch.setattr(_calls, "memory", lambda *_args, **_keywords: {})
        ledger = session.memory_ledger()
        key = "input_noise.phase_export" if phase else "export_power_False_optics_False"
        assert ledger[key] == got.nbytes
        assert len(session._phases if phase else session._exports) == 1
        session.close()
    assert not session._exports and not session._phases
    # Closing destroys retained caches, not the detached result already delivered.
    np.testing.assert_array_equal(old, 0.0 if phase else 1.0 + 0.5j)


@pytest.mark.parametrize("phase", [False, True])
def test_failed_copy_retries_without_publishing_or_reentering_for_stale_tickets(
    boundary, monkeypatch, phase
):
    session, first, ticket = _prepare(boundary)
    original = _calls.call_void
    name_to_fail = "copy_session_noise_phase!" if phase else "export_session!"
    copies = 0

    def fail_once(runtime, name, *args):
        nonlocal copies
        original(runtime, name, *args)
        if name == name_to_fail:
            copies += 1
            if copies == 1:
                raise RuntimeError("failure after foreign copy")

    monkeypatch.setattr(_calls, "call_void", fail_once)
    with pytest.raises(RuntimeError, match="after foreign copy"):
        _read(session, ticket, phase)
    assert not session._exports and not session._phases
    got = _read(session, ticket, phase)
    assert _read(session, ticket, phase) is got and copies == 2
    session.acknowledge(first.receipt)
    with pytest.raises(SessionError):
        _read(session, ticket, phase)
    assert copies == 2
    session.close()
    np.testing.assert_array_equal(got, 0.0 if phase else 1.0 + 0.5j)
