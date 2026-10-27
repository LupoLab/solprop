# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect optional topology and binding evidence outside timed propagation."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from benchmarks.records import Record, SuiteError

__all__ = [
    "allocation_details",
    "cpu_list",
    "gpu_model",
    "hardware_details",
    "linux_topology",
    "lscpu_model",
]


def cpu_list(value: str) -> list[int]:
    """Expand a Linux CPU-list range without treating an absent list as CPU zero."""
    result = set()
    for item in value.strip().split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise SuiteError("invalid Linux CPU list")
        ends = [int(part) for part in item.split("-")]
        if ends[-1] < ends[0]:
            raise SuiteError("descending Linux CPU range")
        result.update(range(ends[0], ends[-1] + 1))
    return sorted(result)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def linux_topology(root: Path, allocation: tuple[int, ...]) -> list[Record]:
    """Read physical core, socket, sibling and NUMA identities for allocated CPUs."""
    result = []
    for cpu in allocation:
        directory = root / f"cpu{cpu}"
        record: Record = {"cpu": cpu}
        for key, filename in (
            ("core", "core_id"),
            ("socket", "physical_package_id"),
        ):
            value = _read(directory / "topology" / filename)
            record[key] = int(value) if value is not None and value.isdigit() else None
            record[f"{key}_reason"] = (
                None if record[key] is not None else "sysfs value unavailable"
            )
        siblings = _read(directory / "topology" / "thread_siblings_list")
        record["siblings"] = cpu_list(siblings) if siblings else None
        record["siblings_reason"] = (
            None if siblings else "sysfs sibling list unavailable"
        )
        nodes = sorted(
            int(path.name[4:])
            for path in directory.glob("node[0-9]*")
            if path.name[4:].isdigit()
        )
        record["numa_nodes"] = nodes or None
        record["numa_reason"] = None if nodes else "sysfs NUMA mapping unavailable"
        result.append(record)
    return result


def _command(arguments: list[str]) -> str | None:
    """Return a short query's standard output, or ``None`` if it fails."""
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    return result.stdout if result.returncode == 0 else None


def lscpu_model(output: str) -> str | None:
    """Name an Arm or other CPU from ``lscpu``, e.g. ``ARM Neoverse-V2``.

    Arm Linux kernels give ``/proc/cpuinfo`` no model name; ``lscpu`` decodes the
    implementer and part numbers into a vendor and a core name.
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
    """Read the processor name Windows keeps in its registry."""
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


def hardware_details(allocation: tuple[int, ...]) -> Record:
    """Record the CPU model and, on Linux, the per-CPU topology."""
    system = platform.system()
    model = None
    cores: list[Record] = []
    if system == "Linux":
        contents = _read(Path("/proc/cpuinfo")) or ""
        match = re.search(r"^model name\s*:\s*(.+)$", contents, re.MULTILINE)
        model = match[1] if match else lscpu_model(_command(["lscpu"]) or "")
        cores = linux_topology(Path("/sys/devices/system/cpu"), allocation)
    elif system == "Darwin":
        output = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        model = output.strip() if output else None
    elif system == "Windows":
        model = _windows_model()
    return {
        "cpu_model": model,
        "cpu_model_reason": None if model else "model query unavailable",
        "cpu_topology": cores or None,
        "cpu_topology_reason": None
        if cores
        else "per-CPU topology not exposed by this collector on this OS",
    }


def allocation_details() -> Record:
    """Record scheduler and memory-node constraints with unverified exclusivity."""
    status = _read(Path("/proc/self/status")) or ""
    result: Record = {}
    for key, field in (
        ("allowed_cpus", "Cpus_allowed_list"),
        ("allowed_memory_nodes", "Mems_allowed_list"),
    ):
        match = re.search(rf"^{field}:\s*(.+)$", status, re.MULTILINE)
        result[key] = cpu_list(match[1]) if match else None
        result[f"{key}_reason"] = None if match else "Linux process status unavailable"
    result["scheduler"] = {
        key: os.environ[key]
        for key in (
            "SLURM_JOB_ID",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_CPUS_PER_NODE",
            "SLURM_CPU_BIND_TYPE",
            "SLURM_CPU_BIND_LIST",
        )
        if key in os.environ
    }
    result["exclusivity"] = "unverified; requires the scheduler allocation record"
    return result


def gpu_model(nvidia: str | None, amd: str | None) -> str | None:
    """Name the first visible GPU from ``nvidia-smi`` or ``rocm-smi`` output.

    `nvidia` is ``nvidia-smi --query-gpu=name --format=csv,noheader``; `amd` is
    ``rocm-smi --showproductname --csv``, whose ``Card Series`` column names the part.
    """
    for line in (nvidia or "").splitlines():
        if line.strip():
            return line.strip()
    lines = [line for line in (amd or "").splitlines() if line.strip()]
    if len(lines) > 1:
        header = [cell.strip() for cell in lines[0].split(",")]
        if "Card Series" in header:
            value = lines[1].split(",")[header.index("Card Series")].strip()
            if value:
                return value if value.startswith("AMD") else f"AMD {value}"
    return None


def gpu_details() -> Record:
    """Record the first visible GPU's model, or why it is unknown."""
    model = gpu_model(
        _command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        _command(["rocm-smi", "--showproductname", "--csv"]),
    )
    return {
        "gpu_model": model,
        "gpu_model_reason": None if model else "no nvidia-smi or rocm-smi answer",
    }
