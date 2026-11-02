# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The design tool: two values set at once from design targets (design 7.5).

A capillary is often designed backwards: the pressure (or the radius) that puts the
zero-dispersion point at a wavelength, then the energy that gives the pulse a soliton
order there (``tools.capillary_design``). Two pins would keep that intent in the
recipe; the tool instead solves the same pins once, in solprop's two stages, and
writes the values it finds as plain literals, in one undo step.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from solprop.recipe.errors import RecipeError

from solprop_gui.model.quantities import rounded
from solprop_gui.registry.explore import design_lines
from solprop_gui.registry.pages import label
from solprop_gui.registry.units import pretty
from solprop_gui.ui.pin import PinField

__all__ = ["DesignDialog"]


def _unit(entry: Mapping[str, object]) -> str | None:
    """Return a keyword's unit, as the kernel describes it.

    Parameters
    ----------
    entry : Mapping
        The keyword's description.

    Returns
    -------
    str or None
        The unit of its first form that has one.
    """
    forms: Sequence[Mapping[str, object]] = entry.get("forms") or []  # pyright: ignore[reportAssignmentType]
    return next((str(f["unit"]) for f in forms if f.get("unit") is not None), None)


def _display(entry: Mapping[str, object]) -> str | None:
    """Return the unit a keyword is shown in.

    Parameters
    ----------
    entry : Mapping
        The keyword's description.

    Returns
    -------
    str or None
        Its display unit, else its own.
    """
    display = entry.get("display")
    return str(display) if display is not None else _unit(entry)


def _pretty(literal: str) -> str:
    """Return a literal as the window shows it: its unit in print (µJ, not uJ).

    Parameters
    ----------
    literal : str
        ``"<number> <unit>"``, or a bare number.

    Returns
    -------
    str
        The literal for reading.
    """
    number, _, unit = literal.partition(" ")
    return f"{number} {pretty(unit)}" if unit else number


class DesignDialog(QDialog):
    """Solve a guide keyword and the pulse size together, and write both.

    Parameters
    ----------
    guide : Sequence of Mapping
        The guide keywords it can solve for (``describe`` entries, with ``pins``).
    pulse : Mapping
        The pulse size it solves for.
    pins : Mapping
        The setup's current pins, by keyword, to start from.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    solve_wanted : Signal
        Emitted with the pins to solve, by keyword.
    apply_wanted : Signal
        Emitted with the literals to write, a list of ``(keyword, value)``.
    """

    solve_wanted = Signal(object)
    apply_wanted = Signal(object)

    def __init__(
        self,
        guide: Sequence[Mapping[str, object]],
        pulse: Mapping[str, object],
        pins: Mapping[str, object],
        parent: QWidget | None = None,
    ) -> None:
        """Make one pin editor per guide keyword, the pulse's, and the buttons."""
        super().__init__(parent)
        self.setWindowTitle("Design the capillary")
        self.setObjectName("design-dialog")
        self.entries = {str(entry["name"]): entry for entry in (*guide, pulse)}
        self.pulse = str(pulse["name"])
        self.fields: dict[str, PinField] = {}
        self.guide = QComboBox()
        self.guide.setObjectName("design-guide")
        self.guide_fields = QStackedWidget()
        for entry in guide:
            name = str(entry["name"])
            self.guide.addItem(label(name), name)
            self.guide_fields.addWidget(self._field(entry, pins))
        self.guide.currentIndexChanged.connect(self.guide_fields.setCurrentIndex)
        if any(str(entry["name"]) == "pressure" for entry in guide):
            self.guide.setCurrentIndex(self.guide.findData("pressure"))
        pulse_field = self._field(pulse, pins)
        self.outcome = QLabel(
            "Give the targets, then Solve: solprop solves the guide first, then the "
            "pulse at it."
        )
        self.outcome.setObjectName("design-outcome")
        self.outcome.setWordWrap(True)
        self.literals: list[tuple[str, str]] = []

        form = QFormLayout()
        form.addRow("Solve for", self.guide)
        form.addRow(self.guide_fields)
        form.addRow(QLabel(f"and {label(self.pulse).lower()}"))
        form.addRow(pulse_field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.solve_button = QPushButton("Solve")
        self.solve_button.setDefault(True)
        self.solve_button.clicked.connect(self.solve)
        self.apply_button = QPushButton("Write values")
        self.apply_button.setToolTip(
            "Write the solved values into the setup as plain values (undoable)."
        )
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply)
        buttons.addButton(self.solve_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.apply_button, QDialogButtonBox.ButtonRole.ApplyRole)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.outcome)
        layout.addWidget(buttons)

    def _field(
        self, entry: Mapping[str, object], pins: Mapping[str, object]
    ) -> PinField:
        """Make a keyword's pin editor, showing its pin if the setup has one.

        Parameters
        ----------
        entry : Mapping
            The keyword's description.
        pins : Mapping
            The setup's pins.

        Returns
        -------
        PinField
            The editor.
        """
        name = str(entry["name"])
        field = PinField(name, entry["pins"], _unit(entry), _display(entry))  # pyright: ignore[reportArgumentType]
        current = pins.get(name)
        if isinstance(current, Mapping):
            field.set_value(current)
        field.changed.connect(self._stale)
        field.refused.connect(self.show_error)
        self.fields[name] = field
        return field

    def chosen(self) -> tuple[str, str]:
        """Return the two keywords solved for.

        Returns
        -------
        tuple of str
            The guide keyword picked and the pulse size.
        """
        return str(self.guide.currentData()), self.pulse

    def _stale(self, *_args: object) -> None:
        """Forget a solution once a target changes.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        self.literals = []
        self.apply_button.setEnabled(False)

    def solve(self) -> None:
        """Ask for the design, if both targets are complete."""
        pins: dict[str, object] = {}
        try:
            for name in self.chosen():
                entry = self.fields[name].entry()
                if entry is None:
                    self.show_error(f"Give every target of the {label(name).lower()}.")
                    return
                pins[name] = entry
        except (RecipeError, ValueError) as error:
            self.show_error(str(error))
            return
        self._stale()
        self.outcome.setText("Solving…")
        self.solve_wanted.emit(pins)

    def show_solution(self, solution: Mapping[str, object]) -> None:
        """Show the kernel's solution, ready to write.

        Parameters
        ----------
        solution : Mapping
            The ``design`` reply: ``values``, ``design`` and ``design_error``.
        """
        values: Mapping[str, float] = solution.get("values") or {}  # pyright: ignore[reportAssignmentType]
        self.literals = []
        for name in self.chosen():
            if name not in values:
                continue
            entry = self.entries[name]
            unit, display = _unit(entry), _display(entry)
            self.literals.append(
                (name, rounded(float(values[name]), unit or "1", display or "1"))
            )
        lines = [f"{label(name)}: {_pretty(value)}" for name, value in self.literals]
        numbers: Mapping[str, float] = solution.get("design") or {}  # pyright: ignore[reportAssignmentType]
        lines += design_lines(numbers)
        if solution.get("design_error"):
            lines.append(f"(No design numbers: {solution['design_error']})")
        self.outcome.setText("\n".join(lines))
        self.apply_button.setEnabled(bool(self.literals))

    def show_error(self, text: str) -> None:
        """Say why the design cannot be solved.

        Parameters
        ----------
        text : str
            The reason.
        """
        self._stale()
        self.outcome.setText(text)

    def apply(self) -> None:
        """Hand the solved values on to be written."""
        if self.literals:
            self.apply_wanted.emit(list(self.literals))
