# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set a machine up: the script shown first, run only when asked (design 13.4).

The dialog shows the exact script that will run on the machine, with the folder the
environment goes in and any wheels to copy over first, and runs it over SSH only on
*Set up*. Its output is shown as it comes; at the end the environment's Python becomes
the machine's.
"""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from solprop.service.install import (
    DEFAULT_ROOT,
    copy_wheels,
    run_setup,
    setup_script,
)
from solprop.service.machines import Machine
from solprop.service.messages import ServiceError

__all__ = ["SetupDialog"]


class SetupDialog(QDialog):
    """Show and run the script that puts solprop on a machine.

    Parameters
    ----------
    machine : solprop.service.machines.Machine
        The machine.
    version : str
        The solprop version to install: the window's own.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    askpass : str or None, optional
        A program ``ssh`` asks a password or code of.

    Attributes
    ----------
    python : str or None
        The environment's Python, once set up.
    """

    _line = Signal(str)
    _done = Signal(object, object)

    def __init__(
        self,
        machine: Machine,
        version: str,
        parent: QWidget | None = None,
        *,
        askpass: str | None = None,
    ) -> None:
        """Lay out the folder, the wheels, the script and the log."""
        super().__init__(parent)
        self.setWindowTitle(f"Set up {machine.name}")
        self.setMinimumWidth(720)
        self.machine = machine
        self.version = version
        self.askpass = askpass
        self.python: str | None = None
        self.wheels: list[Path] = []

        self.root = QLineEdit(DEFAULT_ROOT)
        self.root.textChanged.connect(self._show_script)
        self.wheel_label = QLabel("from PyPI")
        pick = QPushButton("Add wheels…")
        pick.clicked.connect(self._pick_wheels)
        wheels = QHBoxLayout()
        wheels.addWidget(self.wheel_label, 1)
        wheels.addWidget(pick)
        form = QFormLayout()
        form.addRow("Environments go in", self.root)
        form.addRow("solprop " + version, wheels)
        self.note = QLabel()
        self.note.setObjectName("note")
        self.note.setWordWrap(True)
        if ".dev" in version or "+" in version:
            self.note.setText(
                f"This window runs a development build of solprop ({version}), "
                "which PyPI does not have: add its wheels (solprop's and "
                "solprop-native's for the machine), or close this and name a Python "
                "on the machine that has it."
            )
        self.script = QPlainTextEdit()
        self.script.setReadOnly(True)
        self.script.setObjectName("setup-script")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("setup-log")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.run_button = self.buttons.addButton(
            "Set up", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.run_button.clicked.connect(self.start)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.note)
        layout.addWidget(QLabel("What will run on the machine:"))
        layout.addWidget(self.script, 2)
        layout.addWidget(QLabel("What it says:"))
        layout.addWidget(self.log, 1)
        layout.addWidget(self.buttons)
        self._line.connect(self.log.appendPlainText)
        self._done.connect(self._finished)
        self._show_script()

    def _wheel_folder(self) -> str:
        """Return where wheels go on the machine: beside the environments.

        Returns
        -------
        str
            The folder.
        """
        return f"{self.root.text().strip().rstrip('/') or DEFAULT_ROOT}/wheels"

    def _show_script(self) -> None:
        """Show the script as it now stands."""
        folder = self._wheel_folder() if self.wheels else None
        root = self.root.text().strip() or DEFAULT_ROOT
        self.script.setPlainText(setup_script(self.version, root, wheels=folder))
        if self.wheels:
            names = ", ".join(wheel.name for wheel in self.wheels)
            self.wheel_label.setText(f"from {names}, copied first, then PyPI")
        else:
            self.wheel_label.setText("from PyPI")

    def _pick_wheels(self) -> None:
        """Choose wheels to copy to the machine and install from."""
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "Wheels to install", "", "Wheels (*.whl)"
        )
        if chosen:
            self.add_wheels([Path(name) for name in chosen])

    def add_wheels(self, wheels: list[Path]) -> None:
        """Install from these wheels too, copied to the machine first.

        Parameters
        ----------
        wheels : list of pathlib.Path
            The wheels.
        """
        self.wheels = [*self.wheels, *wheels]
        self._show_script()

    def start(self) -> None:
        """Copy any wheels, then run the script, on a thread of its own."""
        import threading  # noqa: PLC0415 - only here

        self.run_button.setEnabled(False)
        self.log.clear()
        script = self.script.toPlainText()
        wheels = list(self.wheels)
        folder = self._wheel_folder()

        def work() -> None:
            """Set the machine up, saying what happens."""
            try:
                if wheels:
                    self._line.emit(f"Copying {len(wheels)} wheels to {folder}")
                    copy_wheels(self.machine, wheels, folder, askpass=self.askpass)
                python = run_setup(
                    self.machine, script, self._line.emit, askpass=self.askpass
                )
            except (ServiceError, OSError) as error:
                self._done.emit(None, str(error))
                return
            self._done.emit(python, None)

        threading.Thread(target=work, daemon=True).start()

    def _finished(self, python: object, error: object) -> None:
        """Say how the setup ended, and keep the environment's Python.

        Parameters
        ----------
        python : object
            The environment's Python, or ``None``.
        error : object
            Why it failed, or ``None``.
        """
        self.run_button.setEnabled(True)
        if error is not None:
            self.log.appendPlainText(f"\nNot set up: {error}")
            return
        self.python = str(python)
        self.log.appendPlainText(f"\nSet up: runs on {self.machine.name} use {python}")
