# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe being edited: its authored document, revision and saved state.

The window's one model of a setup (design 4, 12.5 and 12.6). It holds what the user
set and nothing else, as solprop's recipe document, edited value by value at parameter
paths; every edit raises the revision, which the kernel's replies are matched against.
A recipe opened and left unedited saves verbatim, comments included; an edited one is
written canonically by solprop's writer.

It uses only the light parts of ``solprop.recipe``: the TOML document and parameter
paths. Checking, building and running are the kernel's.
"""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from solprop.recipe.document import Document, dump_text, parse_text
from solprop.recipe.errors import RecipeError
from solprop.recipe.paths import (
    Attribute,
    Index,
    Key,
    format_path,
    get_path,
    parse_path,
    set_path,
)

__all__ = ["MISSING", "RecipeModel", "path_text"]


class _Missing:
    """The value of a keyword the recipe does not set."""

    def __repr__(self) -> str:
        """Return its name.

        Returns
        -------
        str
            ``MISSING``.
        """
        return "MISSING"


MISSING: Final = _Missing()
"""What :meth:`RecipeModel.get` returns for a value the recipe does not set."""


class RecipeModel:
    """One recipe, edited by parameter path.

    Parameters
    ----------
    text : str
        The recipe's TOML text.
    path : pathlib.Path or None, optional
        The file it was read from, or will be saved to.

    Attributes
    ----------
    path : pathlib.Path or None
        Its file.
    revision : int
        Raised by every edit, starting at 1.

    Raises
    ------
    RecipeError
        If the text is not a recipe solprop reads.

    Examples
    --------
    >>> model = RecipeModel('''
    ... [recipe]
    ... format = 1
    ... route = "capillary"
    ... [build]
    ... radius = "125 um"
    ... ''')
    >>> model.set("radius", "100 um")
    '125 um'
    >>> model.get("radius"), model.revision, model.dirty
    ('100 um', 2, True)
    """

    def __init__(self, text: str, *, path: Path | None = None) -> None:
        """Read the text; the model starts unedited and saved."""
        self._document: Document = parse_text(text)
        self._text: str | None = text
        self.path = path
        self.revision = 1
        self._saved = 1

    @classmethod
    def load(cls, path: str | Path) -> RecipeModel:
        """Read a recipe file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.

        Returns
        -------
        RecipeModel
            The model, unedited, saved to that file.
        """
        path = Path(path)
        return cls(path.read_text(encoding="utf-8"), path=path)

    # ----------------------------------------------------------------- reading

    @property
    def route(self) -> str:
        """Return the route the recipe builds.

        Returns
        -------
        str
            ``[recipe] route``.
        """
        return str(self._document["recipe"]["route"])

    @property
    def title(self) -> str | None:
        """Return the recipe's title.

        Returns
        -------
        str or None
            ``[recipe] title``, if it has one.
        """
        title = self._document["recipe"].get("title")
        return None if title is None else str(title)

    @property
    def document(self) -> Document:
        """Return a copy of the recipe's document.

        Returns
        -------
        Document
            Its tables.
        """
        return copy.deepcopy(self._document)

    @property
    def edited(self) -> bool:
        """Return whether any value has been changed since the text was read.

        Returns
        -------
        bool
            False while the recipe would save verbatim.
        """
        return self._text is None

    @property
    def dirty(self) -> bool:
        """Return whether there are changes not yet saved.

        Returns
        -------
        bool
            True after an edit, until the next save.
        """
        return self.revision != self._saved

    def get(self, path: str, table: str = "build") -> object:
        """Return the value at a parameter path.

        Parameters
        ----------
        path : str
            The path, in the table.
        table : str, optional
            The table: ``"build"`` (the default), ``"execution"`` or ``"pins"``.

        Returns
        -------
        object
            The value as the recipe holds it, or :data:`MISSING` where it sets none.
        """
        try:
            return get_path(self._document.get(table, {}), path)
        except RecipeError:
            return MISSING

    def to_text(self) -> str:
        """Return the recipe's TOML text.

        Returns
        -------
        str
            The text it was read from while unedited; canonical TOML otherwise.
        """
        if self._text is not None:
            return self._text
        return dump_text(self._document)

    # ----------------------------------------------------------------- editing

    def _edited(self) -> None:
        """Mark an edit: a new revision, and no verbatim text."""
        self._text = None
        self.revision += 1

    def set(self, path: str, value: object, table: str = "build") -> object:
        """Set the value at a parameter path.

        Parameters
        ----------
        path : str
            The path, in the table.
        value : object
            The value, as a recipe holds it (a quantity string, a number, a list, a
            table); :data:`MISSING` removes it.
        table : str, optional
            The table: ``"build"`` (the default), ``"execution"`` or ``"pins"``.

        Returns
        -------
        object
            The value it replaced, or :data:`MISSING`; an undo sets it back.

        Raises
        ------
        RecipeError
            If the path cannot be reached, or the value is not one a recipe holds.
        """
        if value is MISSING:
            return self.unset(path, table)
        previous = self.get(path, table)
        tables = dict(self._document)
        tables[table] = set_path(self._document.get(table, {}), path, value)
        self._document = parse_text(dump_text(tables))
        self._edited()
        return previous

    def unset(self, path: str, table: str = "build") -> object:
        """Remove the value at a parameter path, so solprop's default applies.

        Parameters
        ----------
        path : str
            The path, in the table.
        table : str, optional
            The table: ``"build"`` (the default), ``"execution"`` or ``"pins"``.

        Returns
        -------
        object
            The value it removed, or :data:`MISSING` when there was none.

        Raises
        ------
        RecipeError
            If the path ends on an array element, which cannot be removed alone.
        """
        previous = self.get(path, table)
        if previous is MISSING:
            return MISSING
        steps = parse_path(path)
        last = steps[-1]
        if isinstance(last, Index):
            msg = f"{path}: an array element cannot be removed alone"
            raise RecipeError(msg)
        name = last.name if isinstance(last, Attribute) else last.key
        current = self._document.get(table, {})
        if len(steps) == 1:
            parent = {k: v for k, v in current.items() if k != name}
            tables = {**self._document, table: parent}
        else:
            owner = get_path(current, steps[:-1])
            assert isinstance(owner, dict)
            trimmed = {k: v for k, v in owner.items() if k != name}
            tables = {**self._document, table: set_path(current, steps[:-1], trimmed)}
        if not tables[table] and table != "build":
            del tables[table]
        self._document = parse_text(dump_text(tables))
        self._edited()
        return previous

    def set_table(self, table: str, value: Mapping[str, object] | None) -> object:
        """Replace a whole table, whose keys are not paths (the ``[scan]`` table's).

        Parameters
        ----------
        table : str
            The table: not ``"build"``, whose keys are paths.
        value : Mapping or None
            Its keys and values; ``None`` removes the table.

        Returns
        -------
        object
            The table it replaced, or :data:`MISSING`.

        Raises
        ------
        ValueError
            If the table is ``"build"``.

        Examples
        --------
        >>> text = "[recipe]", "format = 1", 'route = "gnlse"', "[build]"
        >>> model = RecipeModel(chr(10).join(text))
        >>> model.set_table("scan", {"name": "s", "pulses[0].energy": [1, 2]})
        MISSING
        >>> model.document["scan"]["pulses[0].energy"]
        [1, 2]
        >>> model.set_table("scan", None)["name"]
        's'
        >>> "scan" in model.document
        False
        >>> model.set_table("build", {})
        Traceback (most recent call last):
        ...
        ValueError: the build table is set by path
        """
        if table == "build":
            msg = "the build table is set by path"
            raise ValueError(msg)
        previous = copy.deepcopy(self._document.get(table, MISSING))
        tables = dict(self._document)
        if value is None:
            tables.pop(table, None)
        else:
            tables[table] = dict(value)
        self._document = parse_text(dump_text(tables))
        self._edited()
        return previous

    def set_title(self, title: str | None) -> str | None:
        """Set the recipe's title.

        Parameters
        ----------
        title : str or None
            The title; ``None`` removes it.

        Returns
        -------
        str or None
            The title it replaced.
        """
        previous = self.title
        recipe = dict(self._document["recipe"])
        if title is None:
            recipe.pop("title", None)
        else:
            recipe["title"] = title
        self._document = parse_text(dump_text({**self._document, "recipe": recipe}))
        self._edited()
        return previous

    # ----------------------------------------------------------------- saving

    def save(self, path: str | Path | None = None) -> Path:
        """Write the recipe to its file, or to a new one.

        Parameters
        ----------
        path : str or pathlib.Path or None, optional
            The file; ``None`` writes to :attr:`path`.

        Returns
        -------
        pathlib.Path
            The file written, now the model's.

        Raises
        ------
        ValueError
            If no file is given and the model has none.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            msg = "the recipe has no file yet: give one"
            raise ValueError(msg)
        target.write_text(self.to_text(), encoding="utf-8")
        self.path = target
        self._saved = self.revision
        return target

    def write_recovery(self, path: str | Path) -> Path:
        """Write a recovery copy, leaving the saved state as it is.

        Parameters
        ----------
        path : str or pathlib.Path
            The recovery file.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_text(), encoding="utf-8")
        return target


def path_text(*steps: str | int) -> str:
    """Write a parameter path from its steps.

    Parameters
    ----------
    *steps : str or int
        Names (written ``.name`` where they are identifiers, ``["key"]`` otherwise)
        and indices.

    Returns
    -------
    str
        The path.

    Examples
    --------
    >>> path_text("launch", "HE1,1:even", "share")
    'launch["HE1,1:even"].share'
    """
    parts = []
    for step in steps:
        if isinstance(step, int):
            parts.append(Index(step))
        elif step.isidentifier():
            parts.append(Attribute(step))
        else:
            parts.append(Key(step))
    return format_path(parts)
