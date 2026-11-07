# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a run runs on: its machine, its backend (a recipe value), and its threads.

The machine is this computer or one the window remembers (design 13.4); the backends
and thread counts offered are that machine's own.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from PySide6.QtCore import Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget
from solprop.service.machines import LOCAL, Machine

from solprop_gui.ui.theme import named_label

__all__ = ["MACHINES_ITEM", "RunOptions", "thread_choices"]

MACHINES_ITEM: Final = "\u0000machines"
"""The machine list's last item, which opens the machines dialog."""

_STATES: Final = {
    "connecting": "connecting…",
    "up": "",
    "down": "not reachable; retrying",
    "refused": "refused",
    "closed": "",
}
"""What the line says of a machine's connection."""

_NAMES: Final = {
    "auto": "Automatic",
    "native": "Native (CPU)",
    "numpy_ref": "NumPy reference (slow)",
    "mlx": "MLX GPU, single precision",
}
"""What a backend is called here."""


def thread_choices(budget: int) -> list[int]:
    """Return the thread counts offered: powers of two below the budget, and it.

    Parameters
    ----------
    budget : int
        The threads a run gets by default.

    Returns
    -------
    list of int
        The counts, largest first.

    Examples
    --------
    >>> thread_choices(18)
    [18, 16, 8, 4, 2, 1]
    >>> thread_choices(1)
    [1]
    """
    counts = {budget}
    count = 1
    while count < budget:
        counts.add(count)
        count *= 2
    return sorted(counts, reverse=True)


class RunOptions(QWidget):
    """The backend and thread count a run uses, on one line.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    machine_chosen : Signal
        Emitted with a machine's name as one is picked (:data:`LOCAL` for this
        computer).
    machines_wanted : Signal
        Emitted when *Machines…* is picked.
    resources_wanted : Signal
        Emitted when a Slurm machine's resources are to be changed.
    backend_chosen : Signal
        Emitted with the backend's name as one is picked.
    threads_chosen : Signal
        Emitted with a thread count as one is picked, or ``None`` for the default.
    backend : PySide6.QtWidgets.QComboBox
        The backends.
    threads : PySide6.QtWidgets.QComboBox
        The thread counts.
    """

    backend_chosen = Signal(str)
    threads_chosen = Signal(object)
    machine_chosen = Signal(str)
    machines_wanted = Signal()
    resources_wanted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the two choices, with nothing known yet."""
        super().__init__(parent)
        self.setObjectName("page")
        self.machine = QComboBox()
        self.machine.setObjectName("run-machine")
        self.machine.activated.connect(self._machine_picked)
        self.machine.addItem("This computer", LOCAL)
        self.machine.addItem("Machines…", MACHINES_ITEM)
        self.connection = QLabel()
        self.connection.setObjectName("note")
        self.resources = QToolButton()
        self.resources.setObjectName("run-resources")
        self.resources.setAutoRaise(True)
        self.resources.setToolTip("What the Slurm job asks for: change it")
        self.resources.clicked.connect(self.resources_wanted.emit)
        self.resources.hide()
        self.backend = QComboBox()
        self.backend.setObjectName("run-backend")
        self.backend.activated.connect(self._backend_picked)
        self.threads = QComboBox()
        self.threads.setObjectName("run-threads")
        self.threads.activated.connect(self._threads_picked)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(named_label("Runs on", "note"))
        row.addWidget(self.machine)
        row.addWidget(self.connection)
        row.addWidget(self.resources)
        row.addWidget(self.backend)
        row.addWidget(self.threads)
        row.addStretch(1)
        self._budget = 0

    def set_options(self, options: Mapping[str, object], threads: int | None) -> None:
        """Offer what this machine has.

        Parameters
        ----------
        options : Mapping
            The kernel's ``run_options`` reply.
        threads : int or None
            The count chosen before, or ``None`` for the default.
        """
        installed: Sequence[str] = options["double"]  # pyright: ignore[reportAssignmentType]
        single: Sequence[str] = options["single"]  # pyright: ignore[reportAssignmentType]
        self.backend.clear()
        for name in ("auto", *installed):
            self.backend.addItem(_NAMES.get(name, name), name)
        self.backend.addItem(_NAMES["mlx"], "mlx")
        if "mlx" not in single or not options.get("metal"):
            # Shown, and why it cannot be picked: it needs MLX and a Metal GPU.
            model = self.backend.model()
            assert isinstance(model, QStandardItemModel)  # a combo box's own
            item = model.item(self.backend.count() - 1)
            item.setEnabled(False)
            item.setToolTip(
                "Needs an Apple-silicon Mac and solprop's mlx extra "
                "(pip install solprop[mlx])."
            )
        self._budget = int(options["threads"])  # pyright: ignore[reportArgumentType]
        self.threads.clear()
        for count in thread_choices(self._budget):
            text = (
                f"All {count} threads" if count == self._budget else f"{count} threads"
            )
            self.threads.addItem(text, None if count == self._budget else count)
        self.threads.setToolTip(f"The default is set by {options['limit']}.")
        self.show_threads(threads)

    def set_machines(self, machines: Sequence[Machine], current: str) -> None:
        """Offer the remembered machines, and show the one runs go to.

        Parameters
        ----------
        machines : Sequence of solprop.service.machines.Machine
            The machines, this computer first.
        current : str
            The chosen one's name.
        """
        self.machine.clear()
        for machine in machines:
            text = "This computer" if machine.local else machine.name
            self.machine.addItem(text, machine.name)
        self.machine.addItem("Machines…", MACHINES_ITEM)
        index = self.machine.findData(current)
        self.machine.setCurrentIndex(max(index, 0))

    def show_machine(
        self, machine: Machine, state: str = "", detail: object = None
    ) -> None:
        """Show the chosen machine's connection, and its Slurm resources if any.

        Parameters
        ----------
        machine : solprop.service.machines.Machine
            The machine.
        state : str, optional
            Its connection's state.
        detail : object, optional
            What goes with it: why it is down or refused.
        """
        index = self.machine.findData(machine.name)
        if index >= 0:
            self.machine.setCurrentIndex(index)
        said = "" if machine.local else _STATES.get(state, state)
        self.connection.setText(said)
        self.connection.setToolTip("" if detail is None else str(detail))
        slurm = not machine.local and machine.launcher == "slurm"
        self.resources.setVisible(slurm)
        if slurm:
            from solprop.service.launch import SlurmResources  # noqa: PLC0415 - light

            brief = SlurmResources.from_value(machine.resources).brief
            self.resources.setText(f"Slurm: {brief} · Change…")

    def show_threads(self, threads: int | None) -> None:
        """Show a thread count.

        Parameters
        ----------
        threads : int or None
            The count, or ``None`` for the default.
        """
        index = self.threads.findData(threads)
        self.threads.setCurrentIndex(max(index, 0))

    def show_backend(self, name: object, *, offered: bool) -> None:
        """Show the recipe's backend.

        Parameters
        ----------
        name : object
            Its ``backend``, or anything else for the default.
        offered : bool
            Whether the route takes a backend at all.
        """
        self.backend.setEnabled(offered)
        self.backend.setToolTip(
            "" if offered else "This route runs on the native backend."
        )
        index = self.backend.findData(name if isinstance(name, str) else "auto")
        self.backend.setCurrentIndex(max(index, 0))

    def _machine_picked(self, index: int) -> None:
        """Announce a machine picked, or ask for the machines dialog.

        Parameters
        ----------
        index : int
            Its item.
        """
        name = str(self.machine.itemData(index))
        if name == MACHINES_ITEM:
            self.machines_wanted.emit()
            return
        self.machine_chosen.emit(name)

    def _backend_picked(self, index: int) -> None:
        """Announce a backend picked.

        Parameters
        ----------
        index : int
            Its item.
        """
        self.backend_chosen.emit(str(self.backend.itemData(index)))

    def _threads_picked(self, index: int) -> None:
        """Announce a thread count picked.

        Parameters
        ----------
        index : int
            Its item.
        """
        self.threads_chosen.emit(self.threads.itemData(index))
