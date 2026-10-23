# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Emit and submit quoted Slurm array jobs with explicit controller resources."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from solprop.driver.executors._queue_state import atomic_json
from solprop.driver.executors._request import ScanRequest
from solprop.driver.executors.pool import _THREAD_VARIABLES, Pool
from solprop.driver.executors.queue import Queue
from solprop.exceptions import DriverError

__all__ = ["Slurm"]


def _memory_bytes(value: str) -> int:
    """Interpret positive Slurm memory quantities, whose suffix-free unit is MiB.

    Parameters
    ----------
    value : str
        JSON-serialisable record to publish atomically.

    Returns
    -------
    int
        Memory allocation in bytes.
    """
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT]?)", value.upper())
    if match is None:
        raise DriverError(
            "Slurm mem must be a positive integer with optional K/M/G/T suffix"
        )
    exponent = {"K": 1, "M": 2, "G": 3, "T": 4, "": 2}[match[2]]
    return int(match[1]) * 1024**exponent


@dataclass
class Slurm:
    """Submit an array of queue or chunk controllers without changing their backend.

    Attributes
    ----------
    script : pathlib.Path or str
        Python scan script with its ordinary main guard.
    ntasks : int
        Positive array task count; Slurm task IDs run from one through ntasks.
    cpus_per_task : int
        Positive allocated CPUs per controller task.
    mem : str
        Positive Slurm memory allocation per task; no suffix means MiB.
    time : str
        Slurm wall-clock limit, for example 01:00:00.
    partition : str or None
        Optional scheduler partition.
    gres : str or None
        Optional generic resources, for example gpu:h200:2.
    mode : {'queue', 'chunk'}
        Shared claims or disjoint global round-robin subsets.
    instances : int
        Zero means one controller; positive values split each task into instances.
    queue_path : pathlib.Path, str or None
        Shared queue directory, derived from the output when unspecified.
    workers : int or None
        Maximum local worker processes per instance.
    threads : int
        Threads per local worker, positive.
    fresh : bool
        Whether every point uses a fresh process.
    workdir : pathlib.Path, str or None
        Shared working directory; defaults to the submitting process's directory.
    job_id : str or None
        Scheduler identifier after a successful submission.
    """

    script: Path | str
    ntasks: int = 1
    cpus_per_task: int = 1
    mem: str = "4G"
    time: str = "01:00:00"
    partition: str | None = None
    gres: str | None = None
    mode: Literal["queue", "chunk"] = "queue"
    instances: int = 0
    queue_path: Path | str | None = None
    workers: int | None = None
    threads: int = 1
    fresh: bool = False
    workdir: Path | str | None = None
    job_id: str | None = field(init=False, default=None)
    _queue: Queue | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate resources and reject directive injection."""
        if sys.platform == "win32":
            raise DriverError("the Slurm executor is not supported on Windows")
        if min(self.ntasks, self.cpus_per_task, self.threads) < 1 or self.instances < 0:
            raise DriverError("Slurm task, CPU and thread counts must be positive")
        if self.mode not in ("queue", "chunk"):
            raise DriverError("Slurm mode must be queue or chunk")
        if self.cpus_per_task < max(1, self.instances) * self.threads:
            raise DriverError(
                "Slurm allocation cannot fit the requested instances and threads"
            )
        if self.workers is not None and self.workers < 1:
            raise DriverError("Slurm workers must be positive")
        _memory_bytes(self.mem)
        for value in (str(self.script), self.time, self.partition, self.gres):
            if value is not None and any(char in value for char in "\r\n\x00"):
                raise DriverError("Slurm arguments must not contain line breaks or NUL")

    @property
    def _has_gpu(self) -> bool:
        """Report whether the allocation carries a GPU, independently of instances.

        Returns
        -------
        bool
            Whether the generic resources name a GPU.
        """
        return self.gres is not None and "gpu" in self.gres

    def validate(self, work: Callable) -> None:
        """Refuse non-serialisable work before submitting a job.

        Parameters
        ----------
        work : Callable
            Pickleable point worker with its memory probe.
        """
        Pool().validate(work)

    def prepare(self, request: ScanRequest) -> None:
        """Bind the submission or task to the same durable manifest.

        Parameters
        ----------
        request : ScanRequest
            Scan identity, retention settings and worker to bind before dispatch.
        """
        output = request.output
        self.queue_path = (
            Path(self.queue_path)
            if self.queue_path is not None
            else Path(output).with_suffix(".queue").resolve()
        )
        self.workdir = (
            Path.cwd() if self.workdir is None else Path(self.workdir).resolve()
        )
        count = max(1, self.instances)
        chunk = None
        if os.environ.get("SOLPROP_SLURM_WORKER") == "1" and self.mode == "chunk":
            task = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
            instance = int(os.environ.get("SOLPROP_SCAN_INSTANCE", "0"))
            chunk = (task * count + instance, self.ntasks * count)
        pool = Pool(
            workers=self.workers or (1 if self._has_gpu else None),
            memory_limit=_memory_bytes(self.mem) // count,
            threads=self.threads,
            fresh=self.fresh,
        )
        self._queue = Queue(self.queue_path, pool=pool, chunk=chunk)
        self._queue.prepare(request)

    def render(self) -> str:
        """Return a Bash array script with quoted paths and explicit resource limits.

        Returns
        -------
        str
            Complete script suitable for sbatch or a local task-body test.
        """
        count = max(1, self.instances)
        script = Path(self.script).resolve()
        directory = (
            Path(self.workdir).resolve() if self.workdir is not None else Path.cwd()
        )
        queue = (
            Path(self.queue_path).resolve()
            if self.queue_path is not None
            else script.with_suffix(".queue")
        )
        directives = {
            "array": f"1-{self.ntasks}",
            "cpus-per-task": str(self.cpus_per_task),
            "mem": self.mem,
            "time": self.time,
        }
        if self.partition is not None:
            directives["partition"] = self.partition
        if self.gres is not None:
            directives["gres"] = self.gres
        lines = [
            "#!/usr/bin/env bash",
            *[
                f"#SBATCH --{key}={shlex.quote(value)}"
                for key, value in directives.items()
            ],
            "set -euo pipefail",
            f"cd -- {shlex.quote(str(directory))}",
            ': "${SLURM_ARRAY_TASK_ID:?set the Slurm array task ID}"',
            (
                "if (( SLURM_ARRAY_TASK_ID < 1 || "
                f"SLURM_ARRAY_TASK_ID > {self.ntasks} )); then exit 2; fi"
            ),
        ]
        # One worker per device is a property of the allocation, not of the
        # instance split: a single-instance GPU task must pin it too, or the
        # array task sizes its pool by cores and contends for one device.
        # Splitting CUDA_VISIBLE_DEVICES between instances is the separate
        # question, and only a positive instance count asks it.
        assigned = self.instances > 0 and self._has_gpu
        if assigned:
            lines += [
                (
                    "IFS=, read -r -a devices <<< "
                    '"${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"'
                ),
                (
                    f"if (( ${{#devices[@]}} < {count} )) || "
                    '[[ -z "${devices[0]}" ]]; then '
                    'echo "insufficient assigned GPUs" >&2; exit 2; fi'
                ),
            ]
        lines += [
            "pids=()",
            f"for (( instance=0; instance<{count}; instance++ )); do",
            "  (",
            "    export SOLPROP_SLURM_WORKER=1",
            '    export SOLPROP_SCAN_INSTANCE="$instance"',
            (
                "    export SOLPROP_SCAN_CPU_OFFSET=$(( instance * "
                f"{self.cpus_per_task // count} ))"
            ),
            f"    export SOLPROP_SCAN_CPU_COUNT={self.cpus_per_task // count}",
        ]
        if assigned:
            lines.append('    export CUDA_VISIBLE_DEVICES="${devices[$instance]}"')
        for variable in _THREAD_VARIABLES:
            lines.append(f"    export {variable}={self.threads}")
        lines.append("    export OMP_WAIT_POLICY=passive")
        command = [
            sys.executable,
            "-m",
            "solprop.scan",
            str(script),
            "--queue",
            str(queue),
            "--threads",
            str(self.threads),
            "--memory-limit",
            str(_memory_bytes(self.mem) // count),
        ]
        workers = self.workers or (1 if self._has_gpu else None)
        if workers is not None:
            command += ["--workers", str(workers)]
        if self.fresh:
            command.append("--fresh")
        suffix = ""
        if self.mode == "chunk":
            suffix = (
                f' --chunk "$(( (SLURM_ARRAY_TASK_ID - 1) * {count} + instance ))'
                f'/{self.ntasks * count}"'
            )
        lines += [
            f"    exec {shlex.join(command)}{suffix}",
            "  ) &",
            '  pids+=("$!")',
            "done",
            "status=0",
            'for pid in "${pids[@]}"; do',
            '  if ! wait "$pid"; then status=1; fi',
            "done",
            'exit "$status"',
        ]
        return "\n".join(lines) + "\n"

    def submit(self, destination: Path | str | None = None) -> str:
        """Write the script and submit it with sbatch.

        Parameters
        ----------
        destination : pathlib.Path, str or None, optional
            Generated script path; defaults to submission.slurm in the queue.

        Returns
        -------
        str
            Scheduler job identifier returned by sbatch --parsable.

        Raises
        ------
        DriverError
            If sbatch is unavailable or refuses the submission.
        """
        queue = (
            Path(self.queue_path)
            if self.queue_path is not None
            else Path(self.script).resolve().with_suffix(".queue")
        )
        target = (
            Path(destination) if destination is not None else queue / "submission.slurm"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(target)],
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise DriverError(f"Slurm submission failed: {error}") from error
        self.job_id = result.stdout.strip()
        reports = queue / "submissions"
        reports.mkdir(parents=True, exist_ok=True)
        atomic_json(
            reports / f"{uuid.uuid4().hex}.json",
            {
                "job_id": self.job_id,
                "script": str(target.resolve()),
                "script_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "submitted": time.time(),
            },
        )
        return self.job_id

    def run[T, R](self, points: Iterable[T], work: Callable[[T], R]) -> Iterator[R]:
        """Submit once, or drain this array task's prepared inner queue.

        Parameters
        ----------
        points : Iterable[T]
            Lazily supplied point descriptions retaining global indices.
        work : Callable[[T], R]
            Pickleable point worker with its memory probe.

        Yields
        ------
        object
            Completed result from the prepared execution policy.
        """
        if self._queue is None:
            raise DriverError("Slurm must be prepared by run_scan")
        if os.environ.get("SOLPROP_SLURM_WORKER") == "1":
            yield from self._queue.run(points, work)
        else:
            self.submit()
