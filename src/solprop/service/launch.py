# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Start a run without a window: directly, or as a Slurm job (design 13.4).

Either way the run is ``python -m solprop.service.job FOLDER``
(:mod:`solprop.service.job`), with its spec in the folder, and either way it outlives
whoever started it. What started it is written to ``submitted.json`` at once, so a run
still waiting in Slurm's queue, which has written nothing of its own yet, can be found
and listed.

Directly, the run is a process in a session of its own, so closing the SSH session or
the helper that started it does not end it. Through Slurm it is a batch job from a
template the user edits (:class:`SlurmResources`): its resources and any lines a site
needs (module loads, an environment), and nothing else. The job asks for ``SIGTERM``
two minutes before its time limit, so solprop writes a checkpoint the run can be
resumed from.
"""

import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from solprop._process import BLAS_IDLE_DEFAULTS, child_environment
from solprop.service.messages import ServiceError
from solprop.service.record import process_stamp, set_aside, write_json
from solprop.service.worker import SPEC, RunSpec, spec_value

__all__ = [
    "JOB_OUTPUT",
    "SLURM_SCRIPT",
    "SUBMITTED",
    "SlurmResources",
    "SlurmState",
    "cancel_slurm",
    "prepare",
    "slurm_script",
    "slurm_states",
    "start_direct",
    "submit_slurm",
]

SUBMITTED: Final = "submitted.json"
"""How a run was started, written by whoever started it, in its folder."""

JOB_OUTPUT: Final = "job.out"
"""The run's process's own output (what solprop does not log), in its folder."""

STARTED: Final = "started.txt"
"""What a direct run's starter writes: the run's process, or why it did not start."""

SLURM_SCRIPT: Final = "job.slurm"
"""The batch script a Slurm run was submitted with, in its folder."""

WARNING_SECONDS: Final = 120
"""How long before its time limit a Slurm job is sent ``SIGTERM``, s."""

_JOB_ID: Final = re.compile(r"^(\d+)")


def prepare(spec: RunSpec) -> Path:
    """Make a run's folder and write the spec its job will run.

    Parameters
    ----------
    spec : solprop.service.worker.RunSpec
        The run; its folder is made.

    Returns
    -------
    pathlib.Path
        The folder.

    Raises
    ------
    solprop.service.messages.ServiceError
        If the folder already holds a run that has not been continued.

    Notes
    -----
    A continued run's earlier record is set aside
    (:func:`~solprop.service.record.set_aside`).
    """
    folder = Path(spec.folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    if spec.resume is None and (folder / "run.h5").exists():
        msg = f"{folder} already holds a run; a new run needs a folder of its own"
        raise ServiceError(msg)
    if spec.resume is not None:
        set_aside(folder)
    write_json(folder / SPEC, spec_value(spec))
    return folder


def _job_command(python: str, folder: Path) -> list[str]:
    """Return the command that runs a folder's job."""
    return [python, "-m", "solprop.service.job", str(folder)]


def start_direct(
    folder: Path,
    python: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Start a prepared run as a process that outlives this one.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder (:func:`prepare`).
    python : str or None, optional
        The Python to run it with; ``None`` for this process's own.
    environment : Mapping or None, optional
        Variables to set for it, beside this process's own.

    Returns
    -------
    int
        Its process number.

    Raises
    ------
    solprop.service.messages.ServiceError
        If it could not be started, with why.
    """
    command = _job_command(python or sys.executable, folder)
    env = child_environment({**os.environ, **(environment or {})})
    # Started through a short-lived Python that starts the job and leaves at once:
    # the job is then no child of this process, so it leaves no zombie behind a
    # long-lived helper, and outlives it (a session of its own; detached on Windows).
    # The starter writes the job's process number to a file, not to a pipe: a pipe
    # the job could inherit would keep this call waiting.
    started = folder / STARTED
    started.unlink(missing_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _DETACH,
            str(folder / JOB_OUTPUT),
            str(started),
            *command,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        check=False,
        timeout=60,
    )
    said = started.read_text(encoding="utf-8").strip() if started.exists() else ""
    if result.returncode != 0 or not said.isdigit():
        msg = f"the run could not be started (status {result.returncode}): {said}"
        raise ServiceError(msg)
    pid = int(said)
    import socket  # noqa: PLC0415 - light, only here

    write_json(
        folder / SUBMITTED,
        {
            "launcher": "direct",
            "host": socket.gethostname(),
            "pid": pid,
            "stamp": process_stamp(pid),
            "submitted": time.time(),
        },
    )
    return pid


_DETACH: Final = """\
import os, subprocess, sys
output, started, command = sys.argv[1], sys.argv[2], sys.argv[3:]
options = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
if sys.platform == "win32":
    options["creationflags"] = (
        subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    )
else:
    options["start_new_session"] = True
try:
    with open(output, "ab") as stream:
        process = subprocess.Popen(command, stdout=stream, **options)
except OSError as error:
    with open(started, "w") as said:
        said.write(str(error))
    os._exit(1)
with open(started, "w") as said:
    said.write(str(process.pid))
os._exit(0)
"""
"""The starter's code: it starts the job, says its process, and leaves."""


@dataclass(frozen=True)
class SlurmResources:
    """What a Slurm run asks for, and the lines a site needs: a machine's template.

    Attributes
    ----------
    partition : str or None
        The partition (``--partition``); ``None`` for the site's default.
    time : str or None
        The time limit, as Slurm writes it (``"12:00:00"``, ``"2-00:00:00"``).
    cpus : int
        The CPUs of its one task (``--cpus-per-task``); the run's thread budget
        unless the spec gives one.
    memory : str or None
        Its memory (``--mem``, as ``"16G"``); ``None`` for the site's default.
    gres : str or None
        Generic resources, as ``"gpu:1"`` (``--gres``).
    extra : tuple of str
        Further ``#SBATCH`` options, one per entry, as ``"--account=abc"``.
    setup : tuple of str
        Shell lines run before the job, as ``"module load gcc"``.

    Examples
    --------
    >>> SlurmResources(partition="gpu", time="12:00:00", cpus=8, gres="gpu:1").brief
    'gpu · 8 CPUs · gpu:1 · 12:00:00'
    """

    partition: str | None = None
    time: str | None = None
    cpus: int = 1
    memory: str | None = None
    gres: str | None = None
    extra: tuple[str, ...] = ()
    setup: tuple[str, ...] = field(default=())

    @property
    def brief(self) -> str:  # numpydoc ignore=RT01
        """The resources in one line, as the window shows them.

        The partition, CPUs, memory, generic resources and time that are set.
        """
        parts = [
            self.partition,
            f"{self.cpus} CPU" + ("s" if self.cpus != 1 else ""),
            self.memory,
            self.gres,
            self.time,
        ]
        return " · ".join(part for part in parts if part)

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> "SlurmResources":
        """Return resources from their plain form (:func:`dataclasses.asdict`).

        Parameters
        ----------
        value : Mapping
            The fields; lists stand for tuples.

        Returns
        -------
        SlurmResources
            The resources.

        Examples
        --------
        >>> SlurmResources.from_value({"cpus": 4, "setup": ["module load x"]}).setup
        ('module load x',)
        """
        fields = {
            key: tuple(item) if isinstance(item, list) else item  # pyright: ignore[reportUnknownArgumentType]
            for key, item in value.items()
        }
        return cls(**fields)  # pyright: ignore[reportArgumentType]


def slurm_script(
    folder: Path, python: str, resources: SlurmResources, *, threads: int | None = None
) -> str:
    """Return the batch script of a prepared run.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder, on a filesystem the compute nodes share.
    python : str
        The machine's Python, as the compute nodes reach it.
    resources : SlurmResources
        What the job asks for, and the site's lines.
    threads : int or None, optional
        The run's thread budget; ``None`` takes the job's CPUs.

    Returns
    -------
    str
        The script.

    Examples
    --------
    >>> resources = SlurmResources(cpus=4)
    >>> script = slurm_script(Path("/runs/r1"), "/env/bin/python", resources)
    >>> "#SBATCH --cpus-per-task=4" in script, script.splitlines()[-1]
    (True, 'exec /env/bin/python -m solprop.service.job /runs/r1')
    """
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", folder.name)[:40]
    posix = folder.as_posix()  # a cluster's path, whatever this computer writes
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=solprop-{name}",
        f"#SBATCH --output={shlex.quote(posix + '/' + JOB_OUTPUT)}",
        f"#SBATCH --chdir={shlex.quote(posix)}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(resources.cpus)}",
        f"#SBATCH --signal=B:TERM@{WARNING_SECONDS}",
    ]
    if resources.partition:
        lines.append(f"#SBATCH --partition={resources.partition}")
    if resources.time:
        lines.append(f"#SBATCH --time={resources.time}")
    if resources.memory:
        lines.append(f"#SBATCH --mem={resources.memory}")
    if resources.gres:
        lines.append(f"#SBATCH --gres={resources.gres}")
    lines.extend(
        f"#SBATCH {option.strip()}" for option in resources.extra if option.strip()
    )
    lines.extend(line for line in resources.setup if line.strip())
    budget = str(threads) if threads is not None else "${SLURM_CPUS_PER_TASK:-1}"
    lines.append(f"export SOLPROP_THREADS={budget}")
    # A job is a fresh Python: its BLAS pools spin unless told (solprop issue #383).
    lines.extend(
        f"export {name}=${{{name}:-{value}}}"
        for name, value in BLAS_IDLE_DEFAULTS.items()
    )
    command = " ".join(
        shlex.quote(part) for part in [python, "-m", "solprop.service.job", posix]
    )
    lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"


def _run(
    command: Sequence[str], timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run a scheduler command and return what it printed.

    - ``timeout``: the most seconds to wait.
    """
    try:
        return subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=timeout
        )
    except FileNotFoundError as error:
        msg = f"{command[0]} is not on this machine: is Slurm installed here?"
        raise ServiceError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"{command[0]} did not answer within {timeout:g} s"
        raise ServiceError(msg) from error


def submit_slurm(
    folder: Path, python: str, resources: SlurmResources, *, threads: int | None = None
) -> str:
    """Submit a prepared run as a Slurm batch job.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder (:func:`prepare`).
    python : str
        The machine's Python, as the compute nodes reach it.
    resources : SlurmResources
        What the job asks for.
    threads : int or None, optional
        The run's thread budget; ``None`` takes the job's CPUs.

    Returns
    -------
    str
        The job's identity.

    Raises
    ------
    solprop.service.messages.ServiceError
        If ``sbatch`` refuses the job, with what it said.
    """
    script = folder / SLURM_SCRIPT
    script.write_text(
        slurm_script(folder, python, resources, threads=threads), encoding="utf-8"
    )
    result = _run(["sbatch", "--parsable", str(script)])
    found = _JOB_ID.match(result.stdout.strip())
    if result.returncode != 0 or found is None:
        said = (result.stderr or result.stdout).strip() or f"status {result.returncode}"
        msg = f"Slurm refused the job: {said}"
        raise ServiceError(msg)
    job = found.group(1)
    write_json(
        folder / SUBMITTED,
        {
            "launcher": "slurm",
            "slurm_job": job,
            "resources": asdict(resources),
            "submitted": time.time(),
        },
    )
    return job


@dataclass(frozen=True)
class SlurmState:
    """A Slurm job's state, as the scheduler gives it.

    Attributes
    ----------
    state : str
        Slurm's state: ``"PENDING"``, ``"RUNNING"``, ``"COMPLETED"``,
        ``"CANCELLED"``, ``"TIMEOUT"``, ``"FAILED"``, ``"OUT_OF_MEMORY"``, ...
    reason : str
        Why a pending job waits (``"Priority"``, ``"Resources"``); empty otherwise.
    """

    state: str
    reason: str = ""


def slurm_states(jobs: Sequence[str]) -> dict[str, SlurmState]:
    """Return the states of Slurm jobs: from the queue, else from the accounts.

    Parameters
    ----------
    jobs : Sequence of str
        The jobs.

    Returns
    -------
    dict
        Each job's state, by job; a job neither command knows is left out.
    """
    if not jobs:
        return {}
    listed = ",".join(jobs)
    found: dict[str, SlurmState] = {}
    queue = _run(["squeue", "--noheader", "--format=%i|%T|%r", f"--jobs={listed}"])
    if queue.returncode == 0:
        for line in queue.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3:
                reason = "" if parts[2] in {"None", "(null)"} else parts[2]
                found[parts[0]] = SlurmState(parts[1], reason)
    missing = [job for job in jobs if job not in found]
    if missing:
        accounts = _run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--format=JobID,State",
                f"--jobs={','.join(missing)}",
            ]
        )
        if accounts.returncode == 0:
            for line in accounts.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) == 2 and parts[0] in missing:
                    # "CANCELLED by 1234" names who cancelled it.
                    found[parts[0]] = SlurmState(parts[1].split()[0])
    return found


def cancel_slurm(job: str) -> None:
    """Cancel a Slurm job: ``SIGTERM`` to the run, which leaves it resumable.

    Parameters
    ----------
    job : str
        The job.

    Raises
    ------
    solprop.service.messages.ServiceError
        If ``scancel`` refuses, with what it said.
    """
    result = _run(["scancel", job])
    if result.returncode != 0:
        said = (result.stderr or result.stdout).strip() or f"status {result.returncode}"
        msg = f"Slurm did not cancel job {job}: {said}"
        raise ServiceError(msg)
