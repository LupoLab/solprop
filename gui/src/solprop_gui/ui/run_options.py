# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a run runs on: its backend, a recipe value, and its threads, the window's."""

from collections.abc import Mapping, Sequence
from typing import Final

from PySide6.QtCore import Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QWidget

from solprop_gui.ui.theme import named_label

__all__ = ["RunOptions", "thread_choices"]

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

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the two choices, with nothing known yet."""
        super().__init__(parent)
        self.setObjectName("page")
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
