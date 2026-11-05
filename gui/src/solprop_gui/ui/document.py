# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The open setup: its recipe, undo, checks and files, apart from any window.

Both windows hold one of these. It owns the recipe model, the undo stack, the kernel's
check and summary of each revision, the route's description, saving and the recovery
copy, and says what happened through signals, so a window only shows the result.
"""

import zlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack
from PySide6.QtWidgets import QFileDialog, QWidget

from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.service.messages import Reply
from solprop_gui.ui.bridge import ServiceBridge

__all__ = ["CHECK_DELAY_MS", "Change", "Confirm", "SetupDocument"]

CHECK_DELAY_MS = 400
"""The pause in typing after which the recipe is checked, ms."""

Confirm = Callable[[str, str], str]
"""Asks the user a question; returns ``"save"``, ``"discard"``, ``"stop"`` or
``"cancel"``."""

Change = tuple[str, str, object, object]
"""One change to the recipe: the keyword, its table, the new value and the old."""


class _SetValues(QUndoCommand):
    """Set, or unset, values of the recipe together; undone by setting them back.

    A pin and a value of the same keyword replace each other, so pinning a keyword
    unsets its value in one step, and undoing it sets the value back.

    Parameters
    ----------
    document : SetupDocument
        The setup it changes.
    changes : list of Change
        The changes.
    merging : bool, optional
        Whether it joins the edit before it of the same keyword, as the digits of one
        number typed do, so that one undo takes back the number.
    """

    def __init__(
        self, document: SetupDocument, changes: list[Change], *, merging: bool = False
    ) -> None:
        """Hold the changes."""
        super().__init__(f"Set {changes[0][0]}")
        self.document = document
        self.changes = changes
        self.merging = merging

    def id(self) -> int:
        """Return what it merges with: typing into the same keywords, or nothing.

        Returns
        -------
        int
            A key of the keywords and their tables; ``-1`` for no merging.
        """
        if not self.merging:
            return -1
        names = ",".join(f"{table}/{name}" for name, table, _new, _old in self.changes)
        return zlib.crc32(names.encode()) & 0x7FFFFFFF

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802 - Qt's name
        """Take a following edit of the same keywords into this one.

        Parameters
        ----------
        other : PySide6.QtGui.QUndoCommand
            The next command.

        Returns
        -------
        bool
            Whether it was taken.
        """
        if not isinstance(other, _SetValues) or not other.merging:
            return False
        self.changes = [
            (name, table, later[2], old)
            for (name, table, _new, old), later in zip(
                self.changes, other.changes, strict=True
            )
        ]
        return True

    def redo(self) -> None:
        """Set the new values."""
        self.document.apply_values([(n, t, new) for n, t, new, _ in self.changes])

    def undo(self) -> None:
        """Set the old values back."""
        self.document.apply_values(
            [(n, t, old) for n, t, _, old in reversed(self.changes)]
        )


class _SetTable(QUndoCommand):
    """Replace a whole table of the recipe (its ``[scan]``), as one undo step.

    Parameters
    ----------
    document : SetupDocument
        The setup it changes.
    table : str
        The table.
    new, old : Mapping or None
        Its contents after and before; ``None`` for no table.
    """

    def __init__(
        self,
        document: SetupDocument,
        table: str,
        new: Mapping[str, object] | None,
        old: object,
    ) -> None:
        """Hold the change."""
        super().__init__(f"Set the {table}")
        self.document = document
        self.table = table
        self.new = new
        self.old = old if isinstance(old, Mapping) else None

    def redo(self) -> None:
        """Set the new contents."""
        self.document.apply_table(self.table, self.new)

    def undo(self) -> None:
        """Set the old contents back."""
        self.document.apply_table(self.table, self.old)


class SetupDocument(QObject):
    """The setup a window has open.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The service.
    confirm : Confirm
        How to ask before losing work.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The window, which owns the save dialog.

    Attributes
    ----------
    opened : Signal
        Emitted when a recipe becomes the open one, before it is described.
    loaded : Signal
        Emitted at the end of opening a recipe, for a window to bring it forward.
    described : Signal
        Emitted with the route when its description is at hand.
    values_changed : Signal
        Emitted when values change.
    state_changed : Signal
        Emitted when the title or the saved state may have changed.
    checked : Signal
        Emitted with the issues solprop found in the latest revision.
    summarised : Signal
        Emitted with the summary, or ``None`` and why there is none.
    message : Signal
        Emitted with something to tell the user.
    model : solprop_gui.model.recipe.RecipeModel or None
        The open recipe.
    base : str or None
        The directory its data files resolve against.
    descriptions : dict
        The kernel's ``describe`` replies, by route.
    issues : list of Mapping
        The issues of the latest check.
    undo : PySide6.QtGui.QUndoStack
        The edits made since the recipe opened.
    """

    opened = Signal()
    loaded = Signal()
    described = Signal(str)
    values_changed = Signal()
    state_changed = Signal()
    checked = Signal(list)
    summarised = Signal(object, object)
    message = Signal(str)

    def __init__(
        self,
        bridge: ServiceBridge,
        confirm: Confirm,
        parent: QWidget | None = None,
    ) -> None:
        """Hold no setup yet."""
        super().__init__(parent)
        self.bridge = bridge
        self.confirm = confirm
        self.window = parent
        self.model: RecipeModel | None = None
        self.base: str | None = None
        self.descriptions: dict[str, Mapping[str, object]] = {}
        self.issues: list[Mapping[str, object]] = []
        self._answered = False
        self.undo = QUndoStack(self)
        self._pending: dict[int, str] = {}
        self._adapt: dict[int, Callable[[RecipeModel], None]] = {}
        self.check_timer = QTimer(self)
        self.check_timer.setSingleShot(True)
        self.check_timer.timeout.connect(self.check)
        self.autosave = QTimer(self)
        self.autosave.timeout.connect(self.write_recovery)
        self.autosave.start(30_000)
        bridge.replied.connect(self.on_reply)

    def _ask(
        self,
        kind: str,
        payload: Mapping[str, object],
        purpose: str,
        *,
        revision: int = 0,
    ) -> int:
        """Send a request, remembering what its reply is for.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping
            Its arguments.
        purpose : str
            What the reply is for; also its latest-wins key.
        revision : int, optional
            The recipe revision it is for.

        Returns
        -------
        int
            The request.
        """
        request = self.bridge.request(kind, payload, revision=revision, key=purpose)
        self._pending[request] = purpose
        return request

    def on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the setup's requests.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        purpose = self._pending.pop(reply.request_id, None)
        if purpose is None:
            return
        value = reply.value
        adapt = self._adapt.pop(reply.request_id, None)
        if purpose.startswith("describe-") and reply.ok and isinstance(value, dict):
            route = str(value["route"])
            self.descriptions[route] = value
            self.described.emit(route)
        elif purpose == "example" and reply.ok and isinstance(value, str):
            model = RecipeModel(value)
            if adapt is not None:
                adapt(model)
            self.set_model(model, base=None)
        elif purpose == "check":
            self.issues = list(value) if reply.ok else [_failure(reply)]  # pyright: ignore[reportArgumentType]
            self._answered = True
            self.checked.emit(self.issues)
        elif purpose == "summarise":
            self.summarised.emit(value if reply.ok else None, reply.error)
        # A failed check or summary has said so above, in its own place.
        if not reply.ok and reply.detail and purpose not in ("check", "summarise"):
            self.message.emit(f"{purpose}: {reply.error}")

    # ------------------------------------------------------------------ opening

    def open_example(
        self, name: str, adapt: Callable[[RecipeModel], None] | None = None
    ) -> None:
        """Start a setup from one of solprop's examples.

        Parameters
        ----------
        name : str
            The example's name.
        adapt : callable or None, optional
            What to change in its recipe before it opens.
        """
        if self.may_discard():
            request = self._ask("example", {"name": name}, "example")
            if adapt is not None:
                self._adapt[request] = adapt

    def open_recipe(self, path: str | Path) -> None:
        """Open a recipe file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        """
        if not self.may_discard():
            return
        try:
            model = RecipeModel.load(path)
        except (OSError, ValueError) as error:
            self.message.emit(f"Cannot open {path}: {error}")
            return
        self.set_model(model, base=str(Path(path).parent))

    def open_text(self, text: str, base: str | None) -> None:
        """Open a recipe's text as a new, unsaved setup.

        Parameters
        ----------
        text : str
            The recipe.
        base : str or None
            The directory its data files resolve against.
        """
        if self.may_discard():
            self.set_model(RecipeModel(text), base=base)

    def set_model(self, model: RecipeModel, *, base: str | None) -> None:
        """Make a recipe the open one.

        Parameters
        ----------
        model : solprop_gui.model.recipe.RecipeModel
            The recipe.
        base : str or None
            The directory its data files resolve against.
        """
        self.model = model
        self.base = base
        self.undo.clear()
        self.issues = []
        self._answered = False
        self.opened.emit()
        route = model.route
        if route in self.descriptions:
            self.described.emit(route)
        else:
            self._ask("describe", {"route": route}, f"describe-{route}")
        self.state_changed.emit()
        self.schedule_check(now=True)
        self.loaded.emit()

    def snapshot(self) -> tuple[str, str | None, str | None]:
        """Return the recipe to run: its text, data directory and title.

        Returns
        -------
        tuple
            The text, the directory its data files resolve against, and its title.

        Raises
        ------
        RuntimeError
            If no recipe is open.
        """
        if self.model is None:
            msg = "no setup is open"
            raise RuntimeError(msg)
        return self.model.to_text(), self.base, self.model.title

    # ------------------------------------------------------------------ editing

    def edit(
        self, name: str, table: str, value: object, *, merging: bool = False
    ) -> None:
        """Change one value, or one pin, as an undoable command.

        A value set on a pinned keyword removes its pin, and a pin removes the
        keyword's value, in the same command.

        Parameters
        ----------
        name : str
            The keyword.
        table : str
            Its table: ``"build"``, ``"execution"`` or ``"pins"``.
        value : object
            The new value or pin, or :data:`~solprop_gui.model.recipe.MISSING`.
        merging : bool, optional
            Whether it joins the edit before it of the same keyword, as typing does.
        """
        changes = self._changes(name, table, value)
        if changes:
            self.undo.push(_SetValues(self, changes, merging=merging))

    def replace_pins(
        self,
        released: Iterable[tuple[str, object]],
        pinned: tuple[str, Mapping[str, object]] | None = None,
    ) -> None:
        """Change a setup's targets in one undoable command.

        Parameters
        ----------
        released : Iterable of tuple
            Each pinned keyword to free, and the plain value it keeps.
        pinned : tuple or None, optional
            A keyword and the ``[pins]`` entry that now sets it, replacing its value.
        """
        changes: list[Change] = []
        for name, value in released:
            changes += self._changes(name, "build", value)
        if pinned is not None:
            changes += self._changes(pinned[0], "pins", dict(pinned[1]))
        if changes:
            self.undo.push(_SetValues(self, changes))

    def edit_values(
        self, values: Iterable[tuple[str, object]], *, merging: bool = False
    ) -> bool:
        """Change several ``[build]`` values in one undoable command.

        Parameters
        ----------
        values : Iterable of tuple
            Each keyword and its value.
        merging : bool, optional
            Whether it joins the edit before it of the same keywords, as typing does.

        Returns
        -------
        bool
            Whether anything changed.
        """
        changes = [
            change
            for name, value in values
            for change in self._changes(name, "build", value)
        ]
        if changes:
            self.undo.push(_SetValues(self, changes, merging=merging))
        return bool(changes)

    def _changes(self, name: str, table: str, value: object) -> list[Change]:
        """Return the changes that set one value or pin.

        Parameters
        ----------
        name : str
            The keyword.
        table : str
            Its table: ``"build"``, ``"execution"`` or ``"pins"``.
        value : object
            The new value or pin, or :data:`~solprop_gui.model.recipe.MISSING`.

        Returns
        -------
        list of Change
            The value's change, and the removal of the pin or value it replaces.
        """
        if self.model is None:
            return []
        old = self.model.get(name, table)
        changes: list[Change] = []
        if old != value:
            changes.append((name, table, value, old))
        other = {"pins": "build", "build": "pins"}.get(table)
        if other is not None and (table == "build" or value is not MISSING):
            replaced = self.model.get(name, other)
            if replaced is not MISSING:
                changes.append((name, other, MISSING, replaced))
        return changes

    def apply_values(self, values: list[tuple[str, str, object]]) -> None:
        """Set values in the recipe and announce the change (the commands' action).

        Parameters
        ----------
        values : list of tuple
            Each keyword, its table and its value, or
            :data:`~solprop_gui.model.recipe.MISSING`.
        """
        if self.model is None:
            return
        for name, table, value in values:
            try:
                if value is MISSING:
                    self.model.unset(name, table)
                else:
                    self.model.set(name, value, table)
            except ValueError as error:
                self.message.emit(f"{name}: {error}")
        self.values_changed.emit()
        self.state_changed.emit()
        self.schedule_check()

    def edit_table(self, table: str, value: Mapping[str, object] | None) -> None:
        """Set a whole table as an undoable command.

        Parameters
        ----------
        table : str
            The table.
        value : Mapping or None
            Its contents; ``None`` removes it.
        """
        if self.model is None:
            return
        old = self.model.document.get(table)
        new = None if value is None else dict(value)
        if old != new:
            self.undo.push(_SetTable(self, table, new, old))

    def apply_table(self, table: str, value: Mapping[str, object] | None) -> None:
        """Set a whole table and announce the change (the command's action).

        Parameters
        ----------
        table : str
            The table.
        value : Mapping or None
            Its contents; ``None`` removes it.
        """
        if self.model is None:
            return
        self.model.set_table(table, value)
        self.values_changed.emit()
        self.state_changed.emit()
        self.schedule_check()

    # ------------------------------------------------------------------ checking

    def schedule_check(self, *, now: bool = False) -> None:
        """Check the recipe after a pause in typing.

        Parameters
        ----------
        now : bool, optional
            Check at once.
        """
        self.check_timer.start(0 if now else CHECK_DELAY_MS)

    def check(self) -> None:
        """Ask the kernel to check and summarise the current revision."""
        if self.model is None:
            return
        revision = self.model.revision
        self.bridge.set_revision(revision)
        payload = {"text": self.model.to_text(), "base": self.base}
        self._ask("check", payload, "check", revision=revision)
        self._ask("summarise", payload, "summarise", revision=revision)

    @property
    def can_run(self) -> bool:
        """Return whether solprop's latest check leaves the setup runnable.

        Returns
        -------
        bool
            Whether a setup is open, solprop has answered on it, and has no error
            against it.
        """
        return (
            self.model is not None
            and self._answered
            and not any(issue.get("severity") == "error" for issue in self.issues)
        )

    # ------------------------------------------------------------------ saving

    def save(self) -> bool:
        """Save the setup to its file, asking for one if it has none.

        Returns
        -------
        bool
            Whether it was saved.
        """
        if self.model is None:
            return False
        if self.model.path is None:
            return self.save_as()
        self.model.save()
        self.state_changed.emit()
        return True

    def save_as(self, path: str | Path | None = None) -> bool:
        """Save the setup to a file the user picks.

        Parameters
        ----------
        path : str or pathlib.Path or None, optional
            The file; ``None`` asks for it.

        Returns
        -------
        bool
            Whether it was saved.
        """
        if self.model is None:
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self.window, "Save setup", "recipe.toml", "solprop recipes (*.toml)"
            )
        if not path:
            return False
        self.model.save(path)
        self.base = str(Path(path).parent)
        self.state_changed.emit()
        return True

    def recovery_path(self) -> Path:
        """Return where the recovery copy of an unsaved setup is written.

        Returns
        -------
        pathlib.Path
            The file.
        """
        folder = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        return Path(folder or Path.home()) / "recovery.toml"

    def write_recovery(self) -> None:
        """Write a recovery copy of the setup if it has unsaved changes."""
        if self.model is not None and self.model.dirty:
            self.model.write_recovery(self.recovery_path())

    def may_discard(self) -> bool:
        """Ask before replacing a setup with unsaved changes.

        Returns
        -------
        bool
            Whether to go on.
        """
        if self.model is None or not self.model.dirty:
            return True
        answer = self.confirm(
            "Unsaved setup", "The setup has changes that are not saved. Save them?"
        )
        if answer == "save":
            return self.save()
        return answer == "discard"

    def close(self) -> None:
        """Stop the timers: the window is closing."""
        self.check_timer.stop()
        self.autosave.stop()


def _failure(reply: Reply) -> dict[str, object]:
    """Return a failed check as one issue.

    Parameters
    ----------
    reply : solprop_gui.service.messages.Reply
        The failed reply.

    Returns
    -------
    dict
        An error at no parameter.
    """
    return {"message": reply.error, "parameter": None, "severity": "error"}
