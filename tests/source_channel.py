# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Serve the TDSE binding's provider in this process, through the worker's codec.

The binding talks to its provider through a worker process (SEP 49 section 6.5). Tests
whose fake providers are built and altered in the test itself serve them here instead:
the same adapter, every request and reply encoded to the protocol's bytes and decoded
again, but no process. The worker itself is tested with an importable provider in
``tests/driver/test_source_worker.py``.
"""

import io

from solprop.driver import _pyroraptor, _source_worker, tdse
from solprop.driver._source_worker import RemoteError, WorkerError


def _round_trip(value, computed=None):
    """Return `value` as the other end of a pipe would read it."""
    buffer = io.BytesIO()
    _source_worker.write_message(buffer.write, "reply", value, computed)
    buffer.seek(0)
    message = _source_worker.read_message(buffer.read, 1 << 34)
    assert message is not None
    return message[1]


def _remote(error, adapter):
    report = _source_worker._error_report(error, adapter)
    return RemoteError(
        report["type"], report["message"], report["classification"], report["traceback"]
    )


class InProcessWorker:
    """A worker's request protocol, its adapter served in this process."""

    def __init__(self, make=_pyroraptor.make_adapter):
        try:
            self._adapter = make()
            self.identity = _round_trip(self._adapter.identity())
        except Exception as error:
            raise _remote(error, None) from error
        self.requests = []
        self.closed = False

    def request(self, kind, value=None):
        if self.closed:
            raise WorkerError("the source worker is not running")
        self.requests.append(kind)
        sent = _round_trip(value)
        try:
            reply = self._adapter.handle(kind, sent)
        except Exception as error:
            raise _remote(error, self._adapter) from error
        return _round_trip(reply, self._adapter.computed)

    def close(self):
        if not self.closed:
            self.closed = True
            self._adapter.close()

    kill = close


def in_process(monkeypatch):
    """Make the binding serve its provider in this process; return the workers made."""
    workers = []

    def open_provider(**options):
        del options
        worker = InProcessWorker()
        workers.append(worker)
        return worker

    monkeypatch.setattr(tdse, "_open_provider", open_provider)
    return workers


def install(monkeypatch, provider):
    """Serve `provider` as the binding's provider, in this process; return it."""
    monkeypatch.setattr(_pyroraptor, "_load_provider", lambda: provider)
    in_process(monkeypatch)
    return provider
