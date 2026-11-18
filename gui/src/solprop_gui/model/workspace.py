# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workspace: a folder of run folders (design 4).

Each run has its own folder, ``<date>-<title>``, holding ``run.h5``, ``recipe.toml``
and ``run.log``. The folder is the truth: the list of runs is read from the disk each
time (:func:`solprop.service.runs.list_runs`), and copying a run's folder elsewhere
loses nothing. This module names new run folders.
"""

import re
from datetime import date, datetime
from pathlib import Path, PurePosixPath

__all__ = ["DEFAULT_WORKSPACE", "new_run_folder", "remote_run_folder", "slug"]

DEFAULT_WORKSPACE = Path.home() / "solprop-runs"
"""The workspace when the user has chosen none."""


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


def remote_run_folder(
    workspace: str, home: str, title: str | None, now: datetime | None = None
) -> str:
    """Return a new run's folder on another machine, which cannot be listed first.

    The name is the date, the title's slug and the time of day, so two runs of one
    title on one day do not meet.

    Parameters
    ----------
    workspace : str
        The machine's workspace, as its settings give it (``~`` for its home).
    home : str
        The machine's home folder, as its helper says.
    title : str or None
        The run's title.
    now : datetime.datetime or None, optional
        The time to name it by; now by default.

    Returns
    -------
    str
        The folder, a POSIX path on that machine, with ``~`` written out.

    Examples
    --------
    >>> when = datetime(2026, 11, 7, 14, 5, 9)
    >>> remote_run_folder("~/runs/", "/home/me", "RDW in helium", when)
    '/home/me/runs/2026-11-07-rdw-in-helium-140509'
    """
    when = datetime.now() if now is None else now
    base = workspace.rstrip("/") or "~"
    if base == "~" or base.startswith("~/"):
        base = home.rstrip("/") + base[1:]
    name = f"{when:%Y-%m-%d}-{slug(title)}-{when:%H%M%S}"
    return str(PurePosixPath(base) / name)
