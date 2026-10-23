# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small machine record shared by isolated accelerator benchmarks."""

from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Mapping
from typing import TypedDict


class DeviceRecord(TypedDict):
    """Execution-space and live accelerator telemetry for one sample."""

    device_backend: str
    device_arch: str
    device_name: str
    device_uuid: str
    device_memory_total_mib: float
    device_memory_used_mib: float
    device_utilization_percent: float
    device_temperature_c: float
    device_pstate: str
    device_clock_mhz: float
    device_power_w: float
    device_team_size: int
    device_reduction_size: int


def _number(value: str) -> float:
    """Parse one no-units NVIDIA CSV value, retaining unavailable values as NaN."""
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _integer_metadata(metadata: Mapping[str, object], name: str) -> int:
    """Read one integer-valued native metadata field without silent coercion."""
    value = metadata.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"metadata field {name!r} is not an integer")
    return int(value)


def device_record(metadata: Mapping[str, object]) -> DeviceRecord:
    """Return build identity and one post-sample NVIDIA telemetry snapshot."""
    backend = str(metadata.get("device_backend", "none"))
    record: DeviceRecord = {
        "device_backend": backend,
        "device_arch": str(metadata.get("device_arch", "host")),
        "device_name": "not applicable",
        "device_uuid": "not applicable",
        "device_memory_total_mib": 0.0,
        "device_memory_used_mib": 0.0,
        "device_utilization_percent": 0.0,
        "device_temperature_c": 0.0,
        "device_pstate": "not applicable",
        "device_clock_mhz": 0.0,
        "device_power_w": 0.0,
        "device_team_size": _integer_metadata(metadata, "device_team_size"),
        "device_reduction_size": _integer_metadata(metadata, "device_reduction_size"),
    }
    if backend != "cuda":
        return record
    query = (
        "name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,"
        "pstate,clocks.sm,power.draw"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(io.StringIO(completed.stdout), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 9:
        raise RuntimeError("CUDA benchmarks require exactly one visible NVIDIA GPU")
    values = rows[0]
    record.update(
        {
            "device_name": values[0],
            "device_uuid": values[1],
            "device_memory_total_mib": _number(values[2]),
            "device_memory_used_mib": _number(values[3]),
            "device_utilization_percent": _number(values[4]),
            "device_temperature_c": _number(values[5]),
            "device_pstate": values[6],
            "device_clock_mhz": _number(values[7]),
            "device_power_w": _number(values[8]),
        }
    )
    return record
