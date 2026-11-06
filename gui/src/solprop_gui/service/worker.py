# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The worker process: one per run, from a recipe revision to a results folder.

The window starts a worker with a :class:`RunSpec`, an immutable copy of the recipe
revision it runs, so editing the setup afterwards never changes a running run. The
worker writes ``recipe.toml``, ``run.h5`` and ``run.log`` into the run's folder and
streams events back (:mod:`solprop_gui.service.run`).

This module is imported in the window's process (it is the child's entry point), so it
imports nothing heavy; :func:`main` imports the run code in the child.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Connection

from solprop_gui.service.reduce import Display

__all__ = ["LOG", "OUTPUT", "RECIPE", "SCAN_OUTPUT", "RunSpec", "main"]

OUTPUT = "run.h5"
"""The results file in a run's folder."""

RECIPE = "recipe.toml"
"""The authored recipe in a run's folder, as it was when the run started."""

LOG = "run.log"
"""solprop's log of the run, in its folder."""

SCAN_OUTPUT = "scan.h5"
"""A scan's results file, in its folder: its points, their status and reductions."""


@dataclass(frozen=True)
class RunSpec:
    """What a worker runs, fixed when the run starts.

    Attributes
    ----------
    run_id : str
        The run's identity, its folder's name.
    folder : str
        The run's folder.
    recipe : str or None
        The recipe's TOML text for a new run; ``None`` to continue the folder's
        results file from the recipe it records.
    base : str or None
        The directory the recipe's data files resolve against.
    resume : bool or str or None
        For a continued run, the ``resume=`` mode solprop's ``resumability`` gave
        (``True``, ``"interpolated"`` or ``"force"``); ``None`` for a new run.
    display : Display
        The bins the live views are reduced onto.
    threads : int or None
        The run's thread budget (``SOLPROP_THREADS``); ``None`` for solprop's
        default.
    reductions : tuple of Mapping
        What a scan records at each point beyond its statistics
        (:func:`solprop_gui.service.scan_reductions.reductions`).

    Examples
    --------
    >>> RunSpec("r1", "/tmp/r1", recipe="...").resume is None
    True
    """

    run_id: str
    folder: str
    recipe: str | None = None
    base: str | None = None
    resume: bool | str | None = None
    display: Display = field(default_factory=Display)
    threads: int | None = None
    reductions: tuple[Mapping[str, object], ...] = ()


def main(spec: RunSpec, events: Connection, control: Connection) -> None:
    """Run one recipe revision in this process, streaming its events.

    Parameters
    ----------
    spec : RunSpec
        What to run.
    events : multiprocessing.connection.Connection
        Where the run's events go.
    control : multiprocessing.connection.Connection
        Where stop and subscribe requests come from.
    """
    if spec.threads is not None:
        os.environ["SOLPROP_THREADS"] = str(spec.threads)
    from solprop_gui.service.run import execute  # noqa: PLC0415 - child only

    execute(spec, events, control)
