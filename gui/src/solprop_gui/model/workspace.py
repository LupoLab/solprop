# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workspace: a folder of run folders (design 4).

Each run has its own folder, ``<date>-<title>``, holding ``run.h5``, ``recipe.toml``
and ``run.log``. The folder is the truth: the list of runs is read from the disk each
time, and copying a run's folder elsewhere loses nothing.
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

__all__ = ["DEFAULT_WORKSPACE", "RunEntry", "list_runs", "new_run_folder", "slug"]

DEFAULT_WORKSPACE = Path.home() / "solprop-runs"
"""The workspace when the user has chosen none."""

_OUTPUT = "run.h5"
_RECIPE = "recipe.toml"


def slug(title: str | None) -> str:
    """Return a title as a folder name's part.

    Parameters
    ----------
    title : str or None
        The title.

    Returns
    -------
    str
        Lower case letters, digits and hyphens, at most 40 characters; ``"run"`` for
        a title with none of them.

    Examples
    --------
    >>> slug("RDW in a helium capillary!")
    'rdw-in-a-helium-capillary'
    """
    text = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return text[:40].rstrip("-") or "run"


def new_run_folder(root: Path, title: str | None, today: date | None = None) -> Path:
    """Make a new, empty run folder in the workspace.

    Parameters
    ----------
    root : pathlib.Path
        The workspace.
    title : str or None
        The run's title.
    today : datetime.date or None, optional
        The date it is named by; ``None`` is today.

    Returns
    -------
    pathlib.Path
        The folder, made: ``<date>-<slug>``, with ``-2``, ``-3``, ... appended where
        that name is taken.
    """
    stem = f"{(today or date.today()).isoformat()}-{slug(title)}"
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / stem
    number = 1
    while True:
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
            candidate = root / f"{stem}-{number}"
            continue
        return candidate


@dataclass(frozen=True)
class RunEntry:
    """One run folder of the workspace.

    Attributes
    ----------
    folder : pathlib.Path
        The folder.
    output : pathlib.Path or None
        Its results file, if it has one.
    recipe : pathlib.Path or None
        Its recipe, if it has one.
    modified : float
        When its results file, or else the folder, last changed (seconds since the
        epoch).
    """

    folder: Path
    output: Path | None
    recipe: Path | None
    modified: float

    @property
    def name(self) -> str:
        """Return the run's name, its folder's.

        Returns
        -------
        str
            The folder's name.
        """
        return self.folder.name


def list_runs(root: Path) -> list[RunEntry]:
    """List the run folders of a workspace, newest first.

    Parameters
    ----------
    root : pathlib.Path
        The workspace.

    Returns
    -------
    list of RunEntry
        Every folder holding a results file or a recipe; empty when the workspace
        does not exist.
    """
    if not root.is_dir():
        return []
    entries: list[RunEntry] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        output = folder / _OUTPUT
        recipe = folder / _RECIPE
        has_output = output.is_file()
        has_recipe = recipe.is_file()
        if not (has_output or has_recipe):
            continue
        modified = (output if has_output else folder).stat().st_mtime
        entries.append(
            RunEntry(
                folder,
                output if has_output else None,
                recipe if has_recipe else None,
                modified,
            )
        )
    entries.sort(key=lambda entry: (entry.modified, entry.name), reverse=True)
    return entries
