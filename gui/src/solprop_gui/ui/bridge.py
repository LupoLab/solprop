# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service as Qt signals: the window's one way to reach solprop.

The clients deliver replies and events from their reader threads; the bridge
re-emits them as signals, which Qt queues to the window's thread. The window never
waits on the service.
"""

from collections.abc import Mapping

from PySide6.QtCore import QObject, Signal

from solprop_gui.service.client import KernelClient, RunClient
from solprop_gui.service.worker import RunSpec

__all__ = ["ServiceBridge"]


class ServiceBridge(QObject):
    """Start the kernel, send it requests, and start runs.

    Parameters
    ----------
    parent : PySide6.QtCore.QObject or None, optional
        The owner.

    Attributes
    ----------
    replied : Signal
        Emitted with each wanted :class:`~solprop_gui.service.messages.Reply`.
    run_event : Signal
        Emitted with each :class:`~solprop_gui.service.messages.Event` of any run.
    kernel_lost : Signal
        Emitted when the kernel process ends unasked; the next request restarts it.
    """

    replied = Signal(object)
    run_event = Signal(object)
    kernel_lost = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Make the bridge; the kernel starts with the first request."""
        super().__init__(parent)
        self._kernel: KernelClient | None = None
        self._revision = 0
        self._closed = False
        self.runs: dict[str, RunClient] = {}

    def _client(self) -> KernelClient:
        """Return the kernel's client, starting the kernel if it is not running.

        Returns
        -------
        solprop_gui.service.client.KernelClient
            The client.
        """
        if self._kernel is None or not self._kernel.alive:
            if self._kernel is not None:
                self._kernel.close(timeout=0.5)
            self._kernel = KernelClient(
                self.replied.emit, on_lost=self.kernel_lost.emit
            )
            self._kernel.revision = self._revision
        return self._kernel

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

    def start_run(self, spec: RunSpec) -> RunClient:
        """Start a run's worker.

        Parameters
        ----------
        spec : solprop_gui.service.worker.RunSpec
            What to run.

        Returns
        -------
        solprop_gui.service.client.RunClient
            Its client, whose events arrive through :attr:`run_event`.
        """
        client = RunClient(spec, self.run_event.emit)
        self.runs[spec.run_id] = client
        return client

    def close(self) -> None:
        """End the kernel, for good; runs still going are left to finish on their own.

        A request after this is not sent: it would start a kernel that nothing ends.
        """
        self._closed = True
        if self._kernel is not None:
            self._kernel.close(timeout=2.0)
            self._kernel = None
