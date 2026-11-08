# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a run runs: its spec, fixed when it starts, and the files of its folder.

A run is started with a :class:`RunSpec`, an immutable copy of the recipe revision it
runs, so editing the setup afterwards never changes a running run. The spec is written
to the run's folder (``spec.json``) and run there as a job of its own
(:mod:`solprop.service.job`), which writes ``recipe.toml``, ``run.h5``, ``run.log`` and
its record.

This module is imported in the window's process, so it imports nothing heavy.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from solprop.service.reduce import Display

__all__ = [
    "LOG",
    "OUTPUT",
    "RECIPE",
    "SCAN_OUTPUT",
    "SPEC",
    "RunSpec",
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
    owner : tuple or None
        The process of the window the run is tied to, as its number and start
        (:func:`~solprop.service.record.process_stamp`): the run stops itself, as Stop
        does, when that process has gone. ``None`` for a run that outlives windows.
    launcher : str
        How it was started, as its ``job.json`` says: ``"direct"`` or ``"slurm"``.

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
    owner: tuple[int, str | None] | None = None
    launcher: str = "direct"


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
    solprop.service.messages.ServiceError
        If the data names a field a spec does not have.
    """
    from solprop.service.messages import ServiceError  # noqa: PLC0415 - light, rare

    fields = dict(value)
    display = fields.pop("display", None)
    if isinstance(display, Mapping):
        shown = {
            key: tuple(item) if isinstance(item, list) else item  # pyright: ignore[reportUnknownArgumentType]
            for key, item in display.items()  # pyright: ignore[reportUnknownVariableType]
        }
        fields["display"] = Display(**shown)  # pyright: ignore[reportArgumentType]
    owner = fields.pop("owner", None)
    if isinstance(owner, list | tuple):
        fields["owner"] = (int(owner[0]), None if owner[1] is None else str(owner[1]))  # pyright: ignore[reportUnknownArgumentType]
    reductions = fields.pop("reductions", ())
    fields["reductions"] = tuple(dict(entry) for entry in reductions)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownArgumentType, reportUnknownVariableType]
    try:
        return RunSpec(**fields)  # pyright: ignore[reportArgumentType]
    except TypeError as error:
        msg = f"not a run's spec: {error}"
        raise ServiceError(msg) from error
