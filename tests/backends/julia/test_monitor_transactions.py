# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep monitor continuation and its numerical owner in one commit across signals."""

import math
import signal
from dataclasses import replace

import pytest

from backends.julia import test_session_boundary as boundary_tests
from backends.julia.test_monitor_records import _old_record, _payload
from solprop import _interrupts
from solprop.backends.julia import _session
from solprop.exceptions import BackendError
from solprop.monitor import BoundaryCheck

boundary = boundary_tests.boundary


def _install_monitor(boundary, monkeypatch, *, interrupt=False):
    boundary.prepared._monitor = BoundaryCheck()
    fired = False

    def restore(session, numerical, plane):
        nonlocal fired
        if not hasattr(session, "monitor_resume"):
            session.monitor_resume = numerical
        if interrupt and not fired:
            fired = True
            _interrupts._handle(signal.SIGINT, None)

    def read(runtime, handle, generation):
        session = handle.borrow()
        valid = (
            session.generation == 0 and not session.pending
            if generation == 0
            else session.pending and generation == session.generation
        )
        if not valid:
            raise BackendError("monitor read is stale or not pending")
        resuming = hasattr(session, "monitor_resume")
        record = getattr(session, "monitor_resume", None)
        payload = _payload(
            generation, attached=False, resuming=resuming, committed=math.nan
        )
        return replace(
            payload,
            record={
                **payload.record,
                "launch_energy": math.nan if record is None else record.launch,
            },
        )

    setattr(boundary.engine.module, "restore_session_monitor!", restore)
    monkeypatch.setattr(_session, "read_monitor", read)


@pytest.mark.parametrize("when", ["initial", "pending", "acknowledged"])
def test_continuation_is_valid_at_each_unattached_launch_boundary(
    boundary, monkeypatch, when
):
    _install_monitor(boundary, monkeypatch)
    session = boundary_tests._start(boundary, boundary_tests._frame(0))
    if when != "initial":
        launch = session.advance()
        if when == "acknowledged":
            session.acknowledge(launch.receipt)
    previous = session.monitor_record()
    session.continue_monitor({"record": _old_record()})
    monitor = session._monitor
    assert monitor is not None and monitor.resume is not None
    assert monitor.resume.numerical is not None
    assert monitor.resume.numerical.launch == 10.0
    record = session.monitor_record()
    assert record is not None
    if when == "acknowledged":
        assert previous is not None
        assert record["launch_energy"] == previous["launch_energy"]
    else:
        assert record["launch_energy"] == 10.0


def test_signal_after_engine_restore_keeps_first_record_on_retry(boundary, monkeypatch):
    _install_monitor(boundary, monkeypatch, interrupt=True)
    session = boundary_tests._start(boundary, boundary_tests._frame(0))
    with pytest.raises(KeyboardInterrupt):
        session.continue_monitor({"record": _old_record()})
    session.continue_monitor({"record": {**_old_record(), "launch_energy": 20.0}})
    engine = boundary.engine.sessions[0]
    assert engine.monitor_resume.launch == 10.0
    monitor = session._monitor
    assert monitor is not None and monitor.resume is not None
    assert monitor.resume.numerical is not None
    assert monitor.resume.numerical.launch == engine.monitor_resume.launch
