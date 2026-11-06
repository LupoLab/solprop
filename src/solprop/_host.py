# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The machine a run executes on, and what the run has cost it so far.

A saved run says what code produced it (``RunMetadata.versions`` and ``.git``) and how
the backend sized its pools (``BackendRecord``). This module adds the rest of what is
needed to inspect a case afterwards: the architecture, the CPU model, the CPU counts,
the memory and the operating system (:func:`machine`), and the wall time, CPU time and
peak resident memory since a :class:`RunClock` started (:func:`usage`). The driver
writes both into each execution's record (:class:`solprop.output.schema.RunOutcome`),
so a resume on another machine keeps its own.

Neither costs a run anything measurable: :func:`machine` is asked once per process (at
most one bounded subprocess, for the CPU model) and :func:`usage` is one ``getrusage``
call per save plane. No host name is recorded: a file is shared, and the CPU model and
counts are what identify the hardware.
"""

import dataclasses
import functools
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from solprop._process import allowed_cpus

__all__ = ["ObserverSpend", "RunClock", "cpu_model", "lscpu_model", "machine", "usage"]

# A probe that hangs must not stall a run that is otherwise ready to start.
_TIMEOUT = 5.0


def _command(arguments: list[str]) -> str | None:
    """Return a short query's standard output, or ``None`` if it cannot run.

    Parameters
    ----------
    arguments : list of str
        The command and its arguments, never passed through a shell.

    Returns
    -------
    str or None
        The output, or ``None`` when the command is absent, fails or times out.
    """
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_TIMEOUT,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout if result.returncode == 0 else None


def lscpu_model(output: str) -> str | None:
    """Name an Arm or other CPU from ``lscpu``, e.g. ``ARM Neoverse-V2``.

    Arm Linux kernels give ``/proc/cpuinfo`` no model name; ``lscpu`` decodes the
    implementer and part numbers into a vendor and a core name.

    Parameters
    ----------
    output : str
        What ``lscpu`` printed.

    Returns
    -------
    str or None
        The vendor and model, or ``None`` when ``lscpu`` names none.

    Examples
    --------
    >>> from solprop._host import lscpu_model
    >>> lscpu_model(chr(10).join(["Vendor ID: ARM", "Model name: Neoverse-V2"]))
    'ARM Neoverse-V2'
    """
    fields = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in output.splitlines())
        if value.strip()
    }
    model = fields.get("Model name")
    if not model or model == "-":
        return None
    vendor = fields.get("Vendor ID", "")
    # x86 vendor IDs (GenuineIntel, AuthenticAMD) are not names to prefix.
    if not vendor or vendor.startswith(("Genuine", "Authentic")) or vendor in model:
        return model
    return f"{vendor} {model}"


def _windows_model() -> str | None:
    """Read the processor name Windows keeps in its registry.

    Returns
    -------
    str or None
        The name, or ``None`` off Windows or when the key cannot be read.
    """
    try:
        import winreg  # noqa: PLC0415  # pyright: ignore[reportMissingModuleSource]

        with winreg.OpenKey(  # pyright: ignore[reportAttributeAccessIssue]
            winreg.HKEY_LOCAL_MACHINE,  # pyright: ignore[reportAttributeAccessIssue]
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError, OSError:
        return None
    return str(value).strip() or None


def cpu_model() -> str | None:
    """Return the CPU's marketing name, as the operating system reports it.

    Returns
    -------
    str or None
        For example ``"AMD EPYC 9R45"`` or ``"Apple M4 Max"``; ``None`` when no query
        answers.
    """
    system = platform.system()
    if system == "Linux":
        try:
            contents = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except OSError:
            contents = ""
        match = re.search(r"^model name\s*:\s*(.+)$", contents, re.MULTILINE)
        return match[1].strip() if match else lscpu_model(_command(["lscpu"]) or "")
    if system == "Darwin":
        output = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        return (output or "").strip() or None
    if system == "Windows":
        return _windows_model()
    return None


def _memory_bytes() -> int | None:
    """Return the machine's physical memory, or ``None`` where it cannot be read.

    Returns
    -------
    int or None
        Bytes.
    """
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except AttributeError, OSError, ValueError:
        return None
    return pages * size if pages > 0 and size > 0 else None


@functools.cache
def machine() -> dict[str, object]:
    """Describe the machine this process runs on, asked once per process.

    Returns
    -------
    dict
        ``system``, ``release`` and ``machine`` (the architecture) from
        :mod:`platform`; ``cpu_model``; ``logical_cpus`` (the machine's) and
        ``usable_cpus`` (the ones the scheduler granted this process); ``memory_bytes``
        (physical); ``slurm_job_id`` under Slurm. A value that cannot be read is
        ``None``. Plain JSON values; never a host name.

    Examples
    --------
    >>> from solprop._host import machine
    >>> record = machine()
    >>> record["usable_cpus"] >= 1 and isinstance(record["machine"], str)
    True
    """
    record: dict[str, object] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model() or platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "usable_cpus": len(allowed_cpus()),
        "memory_bytes": _memory_bytes(),
    }
    if "SLURM_JOB_ID" in os.environ:
        record["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    return record


@dataclass
class ObserverSpend:
    """What a run's plane observer cost it (SEP 41, S3).

    Attributes
    ----------
    calls : int
        Planes the observer was shown.
    seconds : float
        Seconds spent inside the observer, in the march's thread.
    transfers : int
        Fields fetched from a device for planes whose field the output keeps not.
    """

    calls: int = 0
    seconds: float = 0.0
    transfers: int = 0


@dataclass(frozen=True)
class RunClock:
    """When a run started, on the wall clock and on the process's CPU clock.

    Attributes
    ----------
    wall : float
        :func:`time.perf_counter` at the start, s.
    cpu : float
        :func:`time.process_time` at the start: the process's user and system time,
        every thread's, s.
    observer : ObserverSpend
        What a plane observer has cost the run so far.
    """

    wall: float
    cpu: float
    observer: ObserverSpend = dataclasses.field(default_factory=ObserverSpend)

    @classmethod
    def start(cls) -> "RunClock":  # noqa: UP037 - bound after the class body
        """Return the clock of a run starting now.

        Returns
        -------
        RunClock
            Both clocks, read now.
        """
        return cls(time.perf_counter(), time.process_time())


def usage(clock: RunClock) -> dict[str, float | int | None]:
    """Return what a run has spent since `clock` started: wall, CPU and peak memory.

    Parameters
    ----------
    clock : RunClock
        The run's start.

    Returns
    -------
    dict
        ``wall_time_s`` and ``cpu_time_s`` (every thread's user and system time) since
        the start, and ``peak_rss_bytes``, the **process's** largest resident memory so
        far: in a session running several runs it includes the earlier ones, and on a
        device it does not count device memory; ``None`` where :mod:`resource` is
        absent (Windows). A run with a plane observer adds ``observer_time_s`` and
        ``observer_transfers``, the device fetches made for it.

    Examples
    --------
    >>> from solprop._host import RunClock, usage
    >>> record = usage(RunClock.start())
    >>> record["wall_time_s"] >= 0.0 and record["cpu_time_s"] >= 0.0
    True
    """
    spent: dict[str, float | int | None] = {
        "wall_time_s": time.perf_counter() - clock.wall,
        "cpu_time_s": time.process_time() - clock.cpu,
        "peak_rss_bytes": None,
    }
    if clock.observer.calls:
        spent["observer_time_s"] = clock.observer.seconds
        spent["observer_transfers"] = clock.observer.transfers
    try:
        import resource  # noqa: PLC0415 - POSIX only
    except ImportError:  # pragma: no cover - Windows
        return spent
    # macOS reports the peak in bytes, Linux and the BSDs in KiB.
    scale = 1 if sys.platform == "darwin" else 1024
    spent["peak_rss_bytes"] = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
    )
    return spent
