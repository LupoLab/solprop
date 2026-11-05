# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The results file open in Analyse: what every page shares, apart from any page.

The file, what it can give, the plane the scrubber is on and the options the pages have
in common. A page asks the kernel through :meth:`AnalysisSession.ask`, whose replies
about a file no longer open are dropped.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QObject, Signal

from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.service.messages import Reply
from solprop_gui.ui.bridge import ServiceBridge

__all__ = ["AnalysisSession"]

Done = Callable[[Any], None]


class _Modes:
    """What the compressor reads of the mode selected.

    Parameters
    ----------
    session : AnalysisSession
        The session.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Hold the session."""
        self.session = session

    def currentText(self) -> str:  # noqa: N802 - the combo box's name
        """Return what the selection is called.

        Returns
        -------
        str
            The mode's label, or ``"mode sum"``.
        """
        session = self.session
        if isinstance(session.mode, int) and session.mode < len(session.labels):
            return session.labels[session.mode]
        return "mode sum"


class _Plane:
    """The plane selected, in the words the compressor reads it in.

    Parameters
    ----------
    session : AnalysisSession
        The session.
    """

    def __init__(self, session: AnalysisSession) -> None:
        """Hold the session."""
        self.session = session
        self.modes = _Modes(session)

    @property
    def planes(self) -> list[int]:
        """Return the planes that kept a field.

        Returns
        -------
        list of int
            Their indices.
        """
        return self.session.planes

    @property
    def current_z(self) -> float | None:
        """Return the position selected.

        Returns
        -------
        float or None
            The position, m.
        """
        return self.session.z

    def index(self) -> int:
        """Return the plane selected.

        Returns
        -------
        int
            Its index in the file.
        """
        return self.session.index

    def selection(self) -> object:
        """Return the modes selected.

        Returns
        -------
        object
            ``"sum"`` or a mode's index.
        """
        return self.session.mode


class AnalysisSession(QObject):
    """One open results file.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The service.
    parent : PySide6.QtCore.QObject or None, optional
        The owner.

    Attributes
    ----------
    opened : Signal
        Emitted when a file opens, before it says what it can give.
    described : Signal
        Emitted with what the file can give.
    plane_moved : Signal
        Emitted when the plane selected changes.
    options_changed : Signal
        Emitted when the options the pages share change.
    mode_changed : Signal
        Emitted when the modes selected change.
    setup_found : Signal
        Emitted with the recipe the file records.
    path : str or None
        The file.
    capabilities : dict
        What it can give, the kernel's ``describe_output``.
    planes : list of int
        The planes that kept a field.
    positions : list of float
        Where each is, m.
    labels : list of str
        The modes' names.
    place : int
        Which of :attr:`planes` is selected.
    mode : object
        The modes selected: ``"sum"`` or an index.
    options : solprop_gui.model.plot_options.PlotOptions
        The options the pages share.
    plane : object
        The plane selected, in the words the compressor reads.
    """

    opened = Signal()
    described = Signal(object)
    plane_moved = Signal()
    options_changed = Signal()
    mode_changed = Signal()
    setup_found = Signal(str, str)

    def __init__(self, bridge: ServiceBridge, parent: QObject | None = None) -> None:
        """Hold no file."""
        super().__init__(parent)
        self.bridge = bridge
        self.path: str | None = None
        self.capabilities: dict[str, object] = {}
        self.planes: list[int] = []
        self.positions: list[float] = []
        self.labels: list[str] = []
        self.place = 0
        self.mode: object = "sum"
        self.options = PlotOptions(floor=-40.0)
        self.plane = _Plane(self)
        self._pending: dict[int, tuple[str | None, Done, Done | None]] = {}
        bridge.replied.connect(self._on_reply)

    @property
    def index(self) -> int:
        """Return the plane selected, as the file numbers it.

        Returns
        -------
        int
            Its index; 0 with no planes.
        """
        return self.planes[self.place] if self.place < len(self.planes) else 0

    @property
    def z(self) -> float | None:
        """Return the position selected.

        Returns
        -------
        float or None
            The position, m.
        """
        return self.positions[self.place] if self.place < len(self.positions) else None

    def open(self, path: str) -> None:
        """Open a results file, at its last plane.

        Parameters
        ----------
        path : str
            The file.
        """
        self.path = path
        self.capabilities = {}
        self.planes = []
        self.positions = []
        self.labels = []
        self.place = 0
        self.mode = "sum"
        self.opened.emit()
        self.ask("describe_output", {"path": path}, self._describe, key="file-describe")

    def _describe(self, value: Mapping[str, object]) -> None:
        """Take what the file can give.

        Parameters
        ----------
        value : Mapping
            The kernel's ``describe_output`` reply.
        """
        self.capabilities = dict(value)
        z: list[float] = [float(v) for v in value.get("z") or []]  # pyright: ignore[reportGeneralTypeIssues]
        kept = [int(p) for p in value.get("field_planes") or []]  # pyright: ignore[reportGeneralTypeIssues]
        self.planes = [p for p in kept if p < len(z)]
        self.positions = [z[p] for p in self.planes]
        self.labels = [str(label) for label in value.get("labels") or []]  # pyright: ignore[reportGeneralTypeIssues]
        self.place = max(len(self.planes) - 1, 0)
        self.described.emit(self.capabilities)

    def find_setup(self) -> None:
        """Ask for the recipe the file records."""
        if self.path is not None:
            path = self.path
            self.ask(
                "recipe_of",
                {"path": path, "resolved": True},
                lambda text: self.setup_found.emit(str(text), path),
                key="file-setup",
            )

    def ask(
        self,
        kind: str,
        payload: Mapping[str, object],
        done: Done,
        *,
        key: str,
        failed: Done | None = None,
    ) -> None:
        """Ask the kernel something about the open file.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping
            Its arguments.
        done : callable
            Takes the reply's value.
        key : str
            Its latest-wins key: a later request of the key replaces an earlier one.
        failed : callable or None, optional
            Takes solprop's reason when it refuses.
        """
        request = self.bridge.request(kind, payload, key=key)
        self._pending[request] = (self.path, done, failed)

    def _on_reply(self, reply: Reply) -> None:
        """Hand a reply to who asked, if it is about the file still open.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        entry = self._pending.pop(reply.request_id, None)
        if entry is None or entry[0] != self.path:
            return
        _path, done, failed = entry
        if reply.ok:
            done(reply.value)
        elif failed is not None:
            failed(str(reply.error))

    def move_to(self, place: int) -> None:
        """Select a plane.

        Parameters
        ----------
        place : int
            Which of :attr:`planes`.
        """
        place = max(min(place, len(self.planes) - 1), 0)
        if place != self.place:
            self.place = place
            self.plane_moved.emit()

    def move_near(self, z: float) -> None:
        """Select the plane nearest a position.

        Parameters
        ----------
        z : float
            The position, m.
        """
        if self.positions:
            distances = [abs(p - z) for p in self.positions]
            self.move_to(distances.index(min(distances)))

    def set_mode(self, mode: object) -> None:
        """Select modes.

        Parameters
        ----------
        mode : object
            ``"sum"`` or a mode's index.
        """
        if mode != self.mode:
            self.mode = mode
            self.mode_changed.emit()

    def set_options(self, **changes: object) -> None:
        """Change the options the pages share.

        Parameters
        ----------
        **changes : object
            The fields of :class:`~solprop_gui.model.plot_options.PlotOptions`.
        """
        options = replace(self.options, **changes)  # pyright: ignore[reportArgumentType]
        if options != self.options:
            self.options = options
            self.options_changed.emit()
