# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The machines dialog: add, test and describe the machines runs can go to (13.4).

A machine is an SSH host, picked from the user's SSH configuration or typed, with
where solprop is on it, where its runs go, and how they start there: directly, or as
Slurm jobs from a template of what they ask for and the lines the site needs. Nothing
about any one site is built in; the partitions offered are the ones the machine's own
``sinfo`` lists.
"""

from collections.abc import Sequence

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from solprop.service.launch import SlurmResources
from solprop.service.machines import LOCAL, Machine, ssh_hosts

from solprop_gui.ui.machines_hub import MachineHub

__all__ = ["MachinesDialog"]


def _lines(text: str) -> tuple[str, ...]:
    """Return a text box's non-empty lines.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    tuple of str
        Its lines, stripped, without the empty ones.
    """
    return tuple(line.strip() for line in text.splitlines() if line.strip())


class MachinesDialog(QDialog):
    """Edit the machines the window remembers.

    Parameters
    ----------
    machines : Sequence of solprop.service.machines.Machine
        The machines, this computer first (it is not shown).
    hub : solprop_gui.ui.machines_hub.MachineHub
        The window's connections, which *Test* uses.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    current : str or None, optional
        The machine to show first.

    Attributes
    ----------
    machines : list of solprop.service.machines.Machine
        The machines as edited, this computer first; read them after ``exec``.
    """

    def __init__(  # noqa: PLR0915 - the dialog's layout
        self,
        machines: Sequence[Machine],
        hub: MachineHub,
        parent: QWidget | None = None,
        *,
        current: str | None = None,
    ) -> None:
        """Lay out the list and the form."""
        super().__init__(parent)
        self.setWindowTitle("Machines")
        self.setMinimumWidth(720)
        self.hub = hub
        self.machines = list(machines)
        self._shown: str | None = None

        self.list = QListWidget()
        self.list.setObjectName("machine-list")
        self.list.currentItemChanged.connect(self._picked)
        add = QPushButton("Add")
        add.setMenu(self._add_menu())
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove)
        side = QVBoxLayout()
        side.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        side.addLayout(buttons)

        self.form = QWidget()
        form = QFormLayout(self.form)
        self.name = QLineEdit()
        self.destination = QLineEdit()
        self.destination.setPlaceholderText(
            "a host of your SSH configuration, or user@host"
        )
        self.port = QSpinBox()
        self.port.setRange(0, 65535)
        self.port.setSpecialValueText("as SSH is configured")
        self.python = QLineEdit()
        self.python.setPlaceholderText("python3, or ~/env/bin/python")
        self.workspace = QLineEdit()
        self.launcher = QComboBox()
        self.launcher.addItem("Directly, as a process of its own", "direct")
        self.launcher.addItem("As Slurm jobs", "slurm")
        self.launcher.currentIndexChanged.connect(self._launcher_changed)
        form.addRow("Name", self.name)
        form.addRow("SSH host", self.destination)
        form.addRow("Port", self.port)
        form.addRow("solprop's Python there", self.python)
        form.addRow("Runs go in", self.workspace)
        form.addRow("Runs start", self.launcher)

        self.slurm = QGroupBox("What each Slurm job asks for")
        slurm = QFormLayout(self.slurm)
        self.partition = QComboBox()
        self.partition.setEditable(True)
        self.partition.lineEdit().setPlaceholderText("the site's default")  # pyright: ignore[reportOptionalMemberAccess]
        ask = QPushButton("Ask the machine")
        ask.setToolTip("List the partitions the machine's sinfo offers")
        ask.clicked.connect(self._ask_partitions)
        partition_row = QHBoxLayout()
        partition_row.addWidget(self.partition, 1)
        partition_row.addWidget(ask)
        self.time = QLineEdit()
        self.time.setPlaceholderText("12:00:00, or 2-00:00:00")
        self.cpus = QSpinBox()
        self.cpus.setRange(1, 4096)
        self.memory = QLineEdit()
        self.memory.setPlaceholderText("16G, or empty for the site's default")
        self.gres = QLineEdit()
        self.gres.setPlaceholderText("gpu:1, or empty")
        self.extra = QPlainTextEdit()
        self.extra.setPlaceholderText("--account=myproject\n--constraint=epyc")
        self.extra.setMaximumHeight(70)
        self.setup = QPlainTextEdit()
        self.setup.setPlaceholderText("module load gcc")
        self.setup.setMaximumHeight(70)
        slurm.addRow("Partition", partition_row)
        slurm.addRow("Time limit", self.time)
        slurm.addRow("CPUs", self.cpus)
        slurm.addRow("Memory", self.memory)
        slurm.addRow("Generic resources", self.gres)
        slurm.addRow("More #SBATCH options", self.extra)
        slurm.addRow("Lines before the run", self.setup)
        form.addRow(self.slurm)

        test = QPushButton("Test the connection")
        test.clicked.connect(self._test)
        self.status = QLabel()
        self.status.setObjectName("note")
        self.status.setWordWrap(True)
        form.addRow(test, self.status)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        main = QHBoxLayout()
        main.addLayout(side, 1)
        main.addWidget(self.form, 3)
        outer = QVBoxLayout(self)
        outer.addLayout(main)
        outer.addWidget(box)
        hub.state_changed.connect(self._state)
        self._fill(current)

    # -------------------------------------------------------------------- list

    def _add_menu(self) -> QMenu:
        """Return the Add button's menu: the SSH configuration's hosts, or a typed one.

        Returns
        -------
        PySide6.QtWidgets.QMenu
            The menu.
        """
        menu = QMenu(self)
        hosts = menu.addMenu("From the SSH configuration")
        found = ssh_hosts()
        for host in found:
            hosts.addAction(host, lambda host=host: self._add(host, host))
        hosts.setEnabled(bool(found))
        menu.addAction("Type a host…", lambda: self._add("new machine", ""))
        return menu

    def _fill(self, current: str | None) -> None:
        """Show the machines, and pick one.

        Parameters
        ----------
        current : str or None
            The one to pick; the first by default.
        """
        self.list.clear()
        for machine in self.machines:
            if not machine.local:
                self.list.addItem(QListWidgetItem(machine.name))
        rows = [self.list.item(i).text() for i in range(self.list.count())]
        if current in rows:
            self.list.setCurrentRow(rows.index(current))
        elif rows:
            self.list.setCurrentRow(0)
        self.form.setEnabled(bool(rows))

    def _add(self, name: str, destination: str) -> None:
        """Add a machine and show it.

        Parameters
        ----------
        name : str
            Its name, made unique.
        destination : str
            Its SSH destination.
        """
        self._keep()
        taken = {m.name for m in self.machines}
        unique, count = name, 2
        while unique in taken or unique == LOCAL:
            unique, count = f"{name} {count}", count + 1
        self.machines.append(Machine(unique, destination or None))
        self._shown = None
        self._fill(unique)

    def _remove(self) -> None:
        """Forget the machine shown; its runs stay on it."""
        if self._shown is None:
            return
        self.machines = [m for m in self.machines if m.name != self._shown]
        self.hub.forget(self._shown)
        self._shown = None
        self._fill(None)

    def _picked(self, item: QListWidgetItem | None) -> None:
        """Keep the form's edits, and show another machine.

        Parameters
        ----------
        item : PySide6.QtWidgets.QListWidgetItem or None
            The machine's row.
        """
        self._keep()
        if item is None:
            return
        machine = next(m for m in self.machines if m.name == item.text())
        self._show(machine)

    # -------------------------------------------------------------------- form

    def _show(self, machine: Machine) -> None:
        """Put a machine in the form.

        Parameters
        ----------
        machine : solprop.service.machines.Machine
            The machine.
        """
        self._shown = machine.name
        self.name.setText(machine.name)
        self.destination.setText(machine.destination or "")
        self.port.setValue(machine.port or 0)
        self.python.setText(machine.python)
        self.workspace.setText(machine.workspace)
        self.launcher.setCurrentIndex(max(self.launcher.findData(machine.launcher), 0))
        resources = SlurmResources.from_value(machine.resources)
        self.partition.setCurrentText(resources.partition or "")
        self.time.setText(resources.time or "")
        self.cpus.setValue(resources.cpus)
        self.memory.setText(resources.memory or "")
        self.gres.setText(resources.gres or "")
        self.extra.setPlainText("\n".join(resources.extra))
        self.setup.setPlainText("\n".join(resources.setup))
        self._launcher_changed()
        state = self.hub.states.get(machine.name, "")
        self.status.setText(state)

    def _edited(self) -> Machine:
        """Return the machine as the form describes it.

        Returns
        -------
        solprop.service.machines.Machine
            The machine.
        """
        resources = SlurmResources(
            partition=self.partition.currentText().strip() or None,
            time=self.time.text().strip() or None,
            cpus=self.cpus.value(),
            memory=self.memory.text().strip() or None,
            gres=self.gres.text().strip() or None,
            extra=_lines(self.extra.toPlainText()),
            setup=_lines(self.setup.toPlainText()),
        )
        from dataclasses import asdict  # noqa: PLC0415 - only here

        return Machine(
            self.name.text().strip() or str(self._shown),
            self.destination.text().strip() or None,
            self.port.value() or None,
            self.python.text().strip() or "python3",
            self.workspace.text().strip() or "~/solprop-runs",
            str(self.launcher.currentData()),
            asdict(resources),
        )

    def _keep(self) -> None:
        """Keep the form's edits to the machine shown."""
        if self._shown is None:
            return
        edited = self._edited()
        self.machines = [edited if m.name == self._shown else m for m in self.machines]
        item = self.list.currentItem()
        if item is not None and item.text() == self._shown:
            item.setText(edited.name)
        self._shown = edited.name

    def _launcher_changed(self) -> None:
        """Show the Slurm template only for a machine whose runs are Slurm jobs."""
        self.slurm.setVisible(self.launcher.currentData() == "slurm")

    # ------------------------------------------------------------- connection

    def _test(self) -> None:
        """Connect to the machine shown, and say what it is."""
        self._keep()
        machine = next((m for m in self.machines if m.name == self._shown), None)
        if machine is None or machine.destination is None:
            self.status.setText("Give the machine's SSH host first.")
            return
        self.hub.forget(machine.name)
        self.status.setText("connecting…")
        self.hub.ensure(machine)

    def _state(self, name: str, state: str, detail: object) -> None:
        """Say how the shown machine's connection went.

        Parameters
        ----------
        name : str
            The machine.
        state : str
            Its connection's state.
        detail : object
            The helper's hello, or why it is down or refused.
        """
        if name != self._shown:
            return
        if state == "up" and isinstance(detail, dict):
            slurm = "with Slurm" if detail.get("slurm") else "without Slurm"
            self.status.setText(
                f"Connected: {detail.get('host')}, solprop {detail.get('solprop')}, "
                f"{slurm}."
            )
        elif state in {"down", "refused"}:
            self.status.setText(f"{state}: {detail}")
        else:
            self.status.setText(state)

    def _ask_partitions(self) -> None:
        """Fill the partitions from the machine's ``sinfo``."""
        if self._shown is None:
            return
        name = self._shown

        def listed(value: object, error: object) -> None:
            if error is not None:
                self.status.setText(f"No partitions: {error}")
                return
            current = self.partition.currentText()
            self.partition.clear()
            for entry in value if isinstance(value, list) else []:
                label = str(entry["name"])  # pyright: ignore[reportIndexIssue]
                self.partition.addItem(label)
            self.partition.setCurrentText(current)

        if name not in self.hub.connections:
            self._test()
        self.hub.call(name, "slurm_info", {}, listed)

    def accept(self) -> None:
        """Keep the last edits, and close."""
        self._keep()
        super().accept()
