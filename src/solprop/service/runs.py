# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The runs in a workspace and the state of each, read from their folders (13.4).

Nothing here keeps state: a run's state is what its folder says (its record,
:mod:`solprop.service.record`, and how it was started, ``submitted.json``), with the
scheduler's word for a Slurm job and the operating system's for a process of this
machine. So any helper on the machine, started at any time, lists the same runs in the
same states.

The states, beside those of a run's ending (``finished``, ``stopped``, ``failed``):

``queued``
    Submitted to Slurm and waiting; the summary's ``detail`` says why.
``starting``
    Started, and nothing recorded yet.
``running``
    Its process lives, or its job runs and it has written its summary lately.
``lost``
    Ended without saying so: its process has gone, or its job ended or was cancelled
    before the run could record an ending. Its file may still be resumable.
"""

import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from solprop.service.launch import SUBMITTED, SlurmState
from solprop.service.record import (
    HEARTBEAT,
    STATUS,
    alive,
    read_job,
    read_status,
    stop_requested,
)
from solprop.service.worker import OUTPUT, RECIPE, SCAN_OUTPUT, SPEC

__all__ = ["STALE", "list_runs", "slurm_jobs", "summarise"]

STALE: Final = 4 * HEARTBEAT
"""How long a run on another machine may go without rewriting its summary before it
is taken for lost, s: a few heartbeats, so a slow shared filesystem is not a death."""

_ENDED_SLURM: Final = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)


def _read(path: Path) -> dict[str, Any] | None:
    """Return a small JSON file, or ``None`` when missing or torn."""
    import json  # noqa: PLC0415 - light, only here

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def _title(folder: Path) -> str:
    """Return a run's title, from its recipe, or its folder's name."""
    try:
        text = (folder / RECIPE).read_text(encoding="utf-8")
    except OSError:
        return folder.name
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "title" and value.strip().startswith('"'):
            return value.strip().strip('"') or folder.name
    return folder.name


def slurm_jobs(folders: Sequence[Path]) -> list[str]:
    """Return the Slurm jobs of the runs that have not recorded an ending.

    Parameters
    ----------
    folders : Sequence of pathlib.Path
        The runs' folders.

    Returns
    -------
    list of str
        Their jobs, to ask the scheduler about in one call.
    """
    jobs: list[str] = []
    for folder in folders:
        status = read_status(folder)
        if status is not None and "ending" in status:
            continue
        submitted = _read(folder / SUBMITTED)
        if submitted is not None and submitted.get("slurm_job"):
            jobs.append(str(submitted["slurm_job"]))
    return jobs


def summarise(
    folder: Path,
    slurm: Mapping[str, SlurmState] | None = None,
    *,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Return a run's summary, as a list of runs shows it.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    slurm : Mapping or None, optional
        The states of the workspace's Slurm jobs
        (:func:`~solprop.service.launch.slurm_states`).
    now : Callable, optional
        The clock, s since the epoch.

    Returns
    -------
    dict
        ``run_id``, ``folder``, ``title``, ``state`` (as the module lists them),
        ``detail`` (a reason or a message), ``launcher``, ``slurm_job``,
        ``submitted``, ``started`` and ``ended`` (s since the epoch, or ``None``),
        ``progress`` (the last, or ``None``), ``points`` and ``done`` for a scan,
        ``scan``, ``output`` (its results file, or ``None`` before one exists),
        ``record`` (whether it keeps one) and ``stopping`` (asked to stop and not yet
        ended).
    """
    status = read_status(folder) or {}
    job = read_job(folder)
    submitted = _read(folder / SUBMITTED) or {}
    scan = (folder / SCAN_OUTPUT).exists() or "points" in status
    output = folder / (SCAN_OUTPUT if scan else OUTPUT)
    ending = status.get("ending")
    slurm_job = submitted.get("slurm_job") or (job.slurm_job if job else None)
    summary: dict[str, Any] = {
        "run_id": folder.name,
        "folder": str(folder),
        "title": _title(folder),
        "launcher": submitted.get("launcher") or (job.launcher if job else None),
        "slurm_job": slurm_job,
        "submitted": submitted.get("submitted"),
        "started": job.started if job else None,
        "ended": job.ended if job else None,
        "progress": status.get("progress"),
        "points": status.get("points"),
        "done": status.get("done"),
        "scan": scan,
        "output": str(output) if output.exists() else None,
        "record": job is not None or bool(submitted) or (folder / STATUS).exists(),
        "stopping": False,
        "detail": "",
    }
    if isinstance(ending, Mapping):
        summary["state"] = str(ending.get("kind"))
        summary["detail"] = str(ending.get("data", {}).get("message", ""))  # pyright: ignore[reportUnknownMemberType]
        return summary
    summary["stopping"] = stop_requested(folder)
    if not summary["record"]:
        # A run with no record (one a window ran without one, or a copied folder):
        # only its file can say anything.
        summary["state"] = "unknown" if output.exists() else "empty"
        return summary
    state, detail = _living_state(folder, job, submitted, status, slurm or {}, now())
    summary["state"], summary["detail"] = state, detail
    return summary


def _living_state(  # noqa: PLR0917 - called positionally
    folder: Path,
    job: Any,
    submitted: Mapping[str, Any],
    status: Mapping[str, Any],
    slurm: Mapping[str, SlurmState],
    now: float,
) -> tuple[str, str]:
    """Return the state of a run that has recorded no ending.

    - ``submitted``: its ``submitted.json``.
    - ``status``: its ``status.json``.
    - ``now``: the time, s since the epoch.
    - Returns: the state and its detail.
    """
    del folder
    slurm_job = submitted.get("slurm_job") or (job.slurm_job if job else None)
    if slurm_job is not None:
        known = slurm.get(str(slurm_job))
        if known is None and job is None:
            # Not listed, and not started: just submitted (the scheduler lists a job
            # a moment later), or asked about without the scheduler.
            submitted_at = submitted.get("submitted")
            recent = (
                isinstance(submitted_at, int | float) and now - submitted_at < STALE
            )
            return ("queued", "") if recent else ("lost", "")
        return _slurm_state(known, job, status, now)
    here = socket.gethostname()
    if job is not None and job.host == here:
        return ("running", "") if alive(job.pid, job.stamp) else ("lost", "")
    if job is None and submitted.get("host") == here:
        pid = int(submitted.get("pid", -1))
        if alive(pid, submitted.get("stamp")):
            return "starting", ""
        return "lost", "it ended before it started"
    return ("running", "") if _fresh(status, now) else ("lost", "")


def _slurm_state(
    known: SlurmState | None, job: Any, status: Mapping[str, Any], now: float
) -> tuple[str, str]:
    """Return the state of a Slurm run that has recorded no ending.

    - ``known``: the scheduler's state of its job; ``None`` where neither the queue nor
      the accounts know it (yet, or any longer).
    - ``status``: its ``status.json``.
    - ``now``: the time, s since the epoch.
    - Returns: the state and its detail.
    """
    if known is None:
        return ("running", "") if _fresh(status, now) else ("lost", "")
    if known.state in {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING"}:
        return "queued", known.reason
    if known.state in _ENDED_SLURM:
        return "lost", known.state.lower().replace("_", " ")
    return ("running", "") if job is not None else ("starting", "")


def _fresh(status: Mapping[str, Any], now: float) -> bool:
    """Return whether a run has rewritten its summary lately.

    - ``status``: its ``status.json``.
    - ``now``: the time, s since the epoch.
    """
    updated = status.get("updated")
    return isinstance(updated, int | float) and now - float(updated) < STALE


def list_runs(
    workspace: Path, slurm: Callable[[Sequence[str]], Mapping[str, SlurmState]] | None
) -> list[dict[str, Any]]:
    """Return the summary of every run in a workspace, newest first.

    Parameters
    ----------
    workspace : pathlib.Path
        The folder that holds one folder per run.
    slurm : Callable or None
        What asks the scheduler for its jobs' states
        (:func:`~solprop.service.launch.slurm_states`); ``None`` where there is no
        Slurm.

    Returns
    -------
    list of dict
        Each run's summary (:func:`summarise`), for each folder holding a recipe, a
        spec or a results file.
    """
    if not workspace.is_dir():
        return []
    marks = (RECIPE, SPEC, OUTPUT, SCAN_OUTPUT, SUBMITTED)
    folders = [
        folder
        for folder in workspace.iterdir()
        if folder.is_dir() and any((folder / mark).exists() for mark in marks)
    ]
    folders.sort(key=lambda folder: folder.stat().st_mtime, reverse=True)
    jobs = slurm_jobs(folders)
    states = slurm(jobs) if slurm is not None and jobs else {}
    return [summarise(folder, states) for folder in folders]
