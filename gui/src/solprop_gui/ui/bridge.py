# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service as Qt signals: the window's one way to reach solprop.

The clients deliver replies and events from their reader threads; the bridge
re-emits them as signals, which Qt queues to the window's thread. The window never
waits on the service.

Two transports give the same calls (design 13.3): by default a kernel child and a
worker child per run, which end with the window (option B); with
``SOLPROP_GUI_SERVICE=1``, this user's solprop service (option C), a process of its own
that keeps runs going when the window closes, and lists them for a window to attach
to (:mod:`solprop_gui.service.remote`).
"""

import itertools
import os
from collections.abc import Mapping, Sequence

from PySide6.QtCore import QObject, Signal

from solprop_gui.service.client import KernelClient, RunClient
from solprop_gui.service.messages import Event, Request
from solprop_gui.service.remote import RemoteRun, ServiceConnection
from solprop_gui.service.worker import RunSpec

__all__ = ["ServiceBridge"]


class ServiceBridge(QObject):
    """Start the kernel, send it requests, and start runs.

    Parameters
    ----------
    parent : PySide6.QtCore.QObject or None, optional
        The owner.
    service : bool or None, optional
        Whether to speak to the solprop service (option C); by default, whether
        ``SOLPROP_GUI_SERVICE`` is ``1``.

    Attributes
    ----------
    replied : Signal
        Emitted with each wanted :class:`~solprop_gui.service.messages.Reply`.
    run_event : Signal
        Emitted with each :class:`~solprop_gui.service.messages.Event` of any run.
    kernel_lost : Signal
        Emitted when the kernel process (or the service) ends unasked; the next
        request starts it again.
    runs_listed : Signal
        Emitted with the service's runs, after :meth:`list_runs`.
    service_failed : Signal
        Emitted with the reason when the service cannot be reached; the bridge then
        goes on with the kernel and workers of its own (option B).
    service : bool
        Whether the bridge speaks to the service (option C).
    """

    replied = Signal(object)
    run_event = Signal(object)
    kernel_lost = Signal()
    runs_listed = Signal(object)
    service_failed = Signal(str)
    _unreachable = Signal(str, object, object)

    def __init__(
        self, parent: QObject | None = None, *, service: bool | None = None
    ) -> None:
        """Make the bridge; the kernel or the service starts with the first request."""
        super().__init__(parent)
        self.service = (
            os.environ.get("SOLPROP_GUI_SERVICE") == "1" if service is None else service
        )
        self._kernel: KernelClient | ServiceConnection | None = None
        self._revision = 0
        self._closed = False
        # One source of request ids for both transports: a request held while the
        # service was sought keeps its id when the window's own kernel takes it.
        self._ids = itertools.count(1)
        self.runs: dict[str, RunClient | RemoteRun] = {}
        self._unreachable.connect(self._fall_back)

    def _client(self) -> KernelClient | ServiceConnection:
        """Return the kernel's client (or the service's), starting it if need be.

        Returns
        -------
        KernelClient or ServiceConnection
            The client.
        """
        if isinstance(self._kernel, ServiceConnection) and self._kernel.failed:
            # No service could be reached, and the news is queued to this thread: go on
            # with the window's own kernel now, rather than seeking the service again,
            # whose connection the fallback would close with what it held.
            self.service = False
            self._kernel = None
        if self._kernel is None or not self._kernel.alive:
            if self._kernel is not None:
                self._kernel.close(timeout=0.5)
            if self.service:
                # Connects on its own thread: the window never waits for it.
                self._kernel = ServiceConnection(
                    self.replied.emit,
                    self.run_event.emit,
                    on_runs=self.runs_listed.emit,
                    on_lost=self.kernel_lost.emit,
                    on_failed=self._unreachable.emit,
                    ids=self._ids,
                )
            else:
                self._kernel = KernelClient(
                    self.replied.emit, on_lost=self.kernel_lost.emit, ids=self._ids
                )
            self._kernel.revision = self._revision
        return self._kernel

    def _fall_back(
        self, reason: str, requests: Sequence[Request], runs: Sequence[RunSpec]
    ) -> None:
        """Go on without the service: the window's own kernel takes what it held.

        Queued to the window's thread from the connection's.

        Parameters
        ----------
        reason : str
            Why the service could not be reached.
        requests : Sequence of Request
            The requests held while it was sought, sent again with their ids.
        runs : Sequence of RunSpec
            The runs started meanwhile, which never reached a worker: said lost.
        """
        if self._closed:
            return
        self.service = False
        if isinstance(self._kernel, ServiceConnection):
            self._kernel.close(timeout=0.5)
            self._kernel = None
        self.service_failed.emit(reason)
        client = self._client()
        for request in requests:
            client.submit(request)  # pyright: ignore[reportAttributeAccessIssue]
        for spec in runs:
            self.run_event.emit(Event(spec.run_id, "lost", {"exitcode": None}))

    def set_revision(self, revision: int) -> None:
        """Set the recipe revision whose replies are wanted.

        Parameters
        ----------
        revision : int
            The revision.
        """
        self._revision = revision
        if self._kernel is not None:
            self._kernel.revision = revision

    def request(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        revision: int = 0,
        key: str = "",
    ) -> int:
        """Send one request to the kernel.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping or None, optional
            Its arguments.
        revision : int, optional
            The recipe revision it is for; 0 for one about no recipe.
        key : str, optional
            Its latest-wins key.

        Returns
        -------
        int
            The request's id; -1 once the bridge is closed, when nothing is sent.
        """
        if self._closed:
            return -1
        return self._client().request(kind, payload, revision=revision, key=key)

    def start_run(self, spec: RunSpec) -> RunClient | RemoteRun:
        """Start a run's worker (in the service, with option C).

        Parameters
        ----------
        spec : solprop_gui.service.worker.RunSpec
            What to run.

        Returns
        -------
        RunClient or RemoteRun
            Its handle, whose events arrive through :attr:`run_event`.
        """
        client = self._client()
        run = (
            client.start_run(spec)
            if isinstance(client, ServiceConnection)
            else RunClient(spec, self.run_event.emit)
        )
        self.runs[spec.run_id] = run
        return run

    def list_runs(self) -> bool:
        """Ask the service for its runs; they arrive through :attr:`runs_listed`.

        Returns
        -------
        bool
            Whether it was asked: only the service holds runs a window did not start.
        """
        if not self.service or self._closed:
            return False
        client = self._client()
        return isinstance(client, ServiceConnection) and client.list_runs()

    def attach(self, spec: RunSpec) -> RemoteRun | None:
        """Take the events of a run the service holds, replayed from its start.

        Parameters
        ----------
        spec : solprop_gui.service.worker.RunSpec
            The run's identity and folder.

        Returns
        -------
        RemoteRun or None
            Its handle; ``None`` without the service.
        """
        client = self._client() if self.service and not self._closed else None
        if not isinstance(client, ServiceConnection):
            return None
        run = client.attach(spec)
        self.runs[spec.run_id] = run
        return run

    def close(self) -> None:
        """End the kernel (or leave the service), for good; runs go on on their own.

        A request after this is not sent: it would start a kernel that nothing ends.
        """
        self._closed = True
        if self._kernel is not None:
            self._kernel.close(timeout=2.0)
            self._kernel = None
