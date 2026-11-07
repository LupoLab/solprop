# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The worker process: one per run, from a recipe revision to a results folder.

The window starts a worker with a :class:`RunSpec`, an immutable copy of the recipe
revision it runs, so editing the setup afterwards never changes a running run. The
worker writes ``recipe.toml``, ``run.h5`` and ``run.log`` into the run's folder and
streams events back (:mod:`solprop.service.run`).

This module is imported in the window's process (it is the child's entry point), so it
imports nothing heavy; :func:`main` imports the run code in the child.
"""

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection

from solprop.service.reduce import Display

__all__ = [
    "LOG",
    "OUTPUT",
    "RECIPE",
    "SCAN_OUTPUT",
    "SPEC",
    "RunSpec",
    "main",
    "spec_from_value",
    "spec_value",
]

OUTPUT = "run.h5"
"""The results file in a run's folder."""

RECIPE = "recipe.toml"
"""The authored recipe in a run's folder, as it was when the run started."""

LOG = "run.log"
"""solprop's log of the run, in its folder."""

SCAN_OUTPUT = "scan.h5"
"""A scan's results file, in its folder: its points, their status and reductions."""

SPEC = "spec.json"
"""What a run started without a window runs (:func:`spec_value`), in its folder."""


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
        (:func:`solprop.service.scan_reductions.reductions`).
    record : bool
        Keep the run's record in its folder (:mod:`solprop.service.record`): its
        events, its summary and ``job.json``, and a stop file it obeys.
    launcher : str
        How it was started, as its ``job.json`` says: ``"window"``, ``"direct"`` or
        ``"slurm"``.

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
    record: bool = False
    launcher: str = "window"


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
    from solprop.service.run import execute  # noqa: PLC0415 - child only

    execute(spec, events, control)


def spec_value(spec: RunSpec) -> dict[str, object]:
    """Return a run's spec as plain data, as ``spec.json`` and the helper carry it.

    Parameters
    ----------
    spec : RunSpec
        The spec.

    Returns
    -------
    dict
        Its fields, the display's as a dict of its own.

    Examples
    --------
    >>> spec = RunSpec("r1", "/tmp/r1", recipe="...", threads=4)
    >>> spec_from_value(spec_value(spec)) == spec
    True
    """
    value = asdict(spec)
    value["reductions"] = [dict(entry) for entry in spec.reductions]
    return value


def spec_from_value(value: Mapping[str, object]) -> RunSpec:
    """Return a run's spec from its plain data (:func:`spec_value`).

    Parameters
    ----------
    value : Mapping
        The fields.

    Returns
    -------
    RunSpec
        The spec; fields the data leaves out take their defaults.

    Raises
    ------
    solprop.exceptions.ServiceError
        If the data names a field a spec does not have.
    """
    from solprop.exceptions import ServiceError  # noqa: PLC0415 - light, rare

    fields = dict(value)
    display = fields.pop("display", None)
    if isinstance(display, Mapping):
        shown = {
            key: tuple(item) if isinstance(item, list) else item  # pyright: ignore[reportUnknownArgumentType]
            for key, item in display.items()  # pyright: ignore[reportUnknownVariableType]
        }
        fields["display"] = Display(**shown)  # pyright: ignore[reportArgumentType]
    reductions = fields.pop("reductions", ())
    fields["reductions"] = tuple(dict(entry) for entry in reductions)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownArgumentType, reportUnknownVariableType]
    try:
        return RunSpec(**fields)  # pyright: ignore[reportArgumentType]
    except TypeError as error:
        msg = f"not a run's spec: {error}"
        raise ServiceError(msg) from error
