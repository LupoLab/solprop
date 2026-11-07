# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service as Qt signals: the window's one way to reach solprop.

The window's kernel (checks, previews, the sheet's numbers, readings of this
computer's results files) is a child of the window's
(:class:`~solprop.service.client.KernelClient`).
Every run, on this computer or another, is a job of its own that keeps its record in
its folder (design 13.4): the bridge starts it through a machine's helper, this
computer's included, and follows its record, through the hub
(:class:`~solprop_gui.ui.machines_hub.MachineHub`). A run outlives the window that
started it, and any window can follow it again. A kernel request about a file of a run
on another machine (its ``path`` in one of the folders the bridge knows there) is
answered by that machine's kernel, so the window reads a remote run as it reads its
own.

The clients deliver replies and events from their reader threads; the bridge re-emits
them as signals, which Qt queues to the window's thread. The window never waits.
"""

import itertools
from collections.abc import Mapping
from typing import Final

from PySide6.QtCore import QObject, Signal
from solprop.service.client import KernelClient, ReplyFilter
from solprop.service.machines import Machine
from solprop.service.messages import Event, Reply, Request
from solprop.service.worker import RunSpec, spec_value

from solprop_gui.ui.machines_hub import MachineHub, MachineRun

__all__ = ["HERE_PERIOD", "THERE_PERIOD", "ServiceBridge", "follow_period"]

HERE_PERIOD: Final = 0.1
"""How often a run on this computer is read for new events, s: about one read in ten
frames of the live view, at a few hundredths of a percent of a core."""

THERE_PERIOD: Final = 0.5
"""How often a run on another machine is read for new events, s."""


def follow_period(machine: Machine) -> float:
    """Return how often a machine's runs are read for new events.

    Parameters
    ----------
    machine : solprop.service.machines.Machine
        The machine.

    Returns
    -------
    float
        :data:`HERE_PERIOD` for this computer, :data:`THERE_PERIOD` for another, s.

    Examples
    --------
    >>> follow_period(Machine("local")), follow_period(Machine("lab", "me@lab"))
    (0.1, 0.5)
    """
    return HERE_PERIOD if machine.local else THERE_PERIOD


class ServiceBridge(QObject):
    """Start the kernel, send it requests, and start and follow runs.

    Parameters
    ----------
    parent : PySide6.QtCore.QObject or None, optional
        The owner.

    Attributes
    ----------
    replied : Signal
        Emitted with each wanted :class:`~solprop.service.messages.Reply`.
    run_event : Signal
        Emitted with each :class:`~solprop.service.messages.Event` of any run.
    kernel_lost : Signal
        Emitted when the kernel process ends unasked; the next request starts it
        again.
    hub : solprop_gui.ui.machines_hub.MachineHub
        The connections to the machines' helpers, this computer's included.
    runs : dict
        The handles of the runs followed, by run.
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
        self._ids = itertools.count(1)
        self.runs: dict[str, MachineRun] = {}
        from solprop_gui.askpass import askpass_program  # noqa: PLC0415 - light

        self.hub = MachineHub(self, askpass=askpass_program())
        self.hub.run_event.connect(self.run_event.emit)
        self.hub.replied.connect(self._machine_replied)
        # The folders of runs on other machines, by folder: whose kernel reads them.
        self.remote_folders: dict[str, str] = {}
        self._remote_filter = ReplyFilter()

    def _client(self) -> KernelClient:
        """Return the kernel's client, starting it if need be.

        Returns
        -------
        solprop.service.client.KernelClient
            The client.
        """
        if self._kernel is None or not self._kernel.alive:
            if self._kernel is not None:
                self._kernel.close(timeout=0.5)
            self._kernel = KernelClient(
                self.replied.emit, on_lost=self.kernel_lost.emit, ids=self._ids
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
        self._remote_filter.revision = revision
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
        machine = self.machine_of(payload)
        if machine is not None:
            request = Request(
                kind, dict(payload or {}), next(self._ids), revision=revision, key=key
            )
            self._remote_filter.sent(request)
            self.hub.kernel(machine, request)
            return request.request_id
        return self._client().request(kind, payload, revision=revision, key=key)

    def machine_of(self, payload: Mapping[str, object] | None) -> str | None:
        """Return the machine whose kernel answers a request, or ``None`` for here.

        Parameters
        ----------
        payload : Mapping or None
            The request's arguments: a ``path`` (or ``base``) in the folder of a run
            on another machine sends it there, as does a ``machine`` named in it.

        Returns
        -------
        str or None
            The machine's name.
        """
        if not payload:
            return None
        named = payload.get("machine")
        if isinstance(named, str) and named in self.hub.connections:
            return named
        for key in ("path", "base"):
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            for folder, machine in self.remote_folders.items():
                if value == folder or value.startswith(folder.rstrip("/") + "/"):
                    return machine
        return None

    def _machine_replied(self, reply: Reply) -> None:
        """Pass on a machine kernel's reply if it is still wanted.

        Parameters
        ----------
        reply : solprop.service.messages.Reply
            The reply.
        """
        if self._remote_filter.accepts(reply):
            self.replied.emit(reply)

    def start_on(
        self,
        machine: Machine,
        spec: RunSpec,
        resources: Mapping[str, object] | None = None,
    ) -> MachineRun:
        """Start a run as a job on a machine, this computer included, and follow it.

        Parameters
        ----------
        machine : solprop.service.machines.Machine
            The machine.
        spec : solprop.service.worker.RunSpec
            What to run; its folder is on the machine.
        resources : Mapping or None, optional
            Its Slurm resources, for a machine whose runs start through Slurm; the
            machine's own by default.

        Returns
        -------
        MachineRun
            Its handle, whose events arrive through :attr:`run_event`.
        """
        self.hub.ensure(machine)
        if not machine.local:
            # A run of this computer's is read by the window's own kernel.
            self.remote_folders[spec.folder] = machine.name
        body = {
            "spec": spec_value(spec),
            "launcher": machine.launcher,
            "resources": dict(resources or machine.resources),
        }

        def started(value: object, error: object) -> None:
            """Say a run that could not be started on its machine failed.

            Parameters
            ----------
            value : object
                The start's summary, unused.
            error : object
                Why it failed, or ``None``.
            """
            del value
            if error is not None:
                data = {"path": f"{spec.folder}/run.h5", "message": str(error)}
                self.run_event.emit(Event(spec.run_id, "failed", data))
                return
            # Followed once started: a continued run's record is new only then.
            self.hub.begin(machine.name, spec.folder)

        run = self.hub.follow(
            machine.name, spec.folder, period=follow_period(machine), send=False
        )
        self.hub.call(machine.name, "start", body, started)
        self.runs[spec.run_id] = run
        return run

    def attach_on(self, machine: Machine, run_id: str, folder: str) -> MachineRun:
        """Follow a run on a machine, this computer included, from its record's start.

        Parameters
        ----------
        machine : solprop.service.machines.Machine
            The machine.
        run_id : str
            The run.
        folder : str
            Its folder there.

        Returns
        -------
        MachineRun
            Its handle.
        """
        self.hub.ensure(machine)
        if not machine.local:
            # A run of this computer's is read by the window's own kernel.
            self.remote_folders[folder] = machine.name
        run = self.hub.follow(machine.name, folder, period=follow_period(machine))
        self.runs[run_id] = run
        return run

    def know_folder(self, machine: str, folder: str) -> None:
        """Send requests about a folder's files to a machine's kernel.

        Parameters
        ----------
        machine : str
            The machine.
        folder : str
            The folder there.
        """
        self.remote_folders[folder] = machine

    def close(self) -> None:
        """End the kernel and the connections, for good; runs go on on their own.

        A request after this is not sent: it would start a kernel that nothing ends.
        """
        self._closed = True
        self.hub.close()
        if self._kernel is not None:
            self._kernel.close(timeout=2.0)
            self._kernel = None
