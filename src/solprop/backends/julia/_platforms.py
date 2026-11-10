# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Identify supported Julia hosts and the checksummed runtime archives."""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from solprop.exceptions import BackendError

JULIA_VERSION = "1.13.3"
BRIDGE_VERSION = "0.9.37"
DEVICES = ("cpu", "cuda", "rocm", "metal")


@dataclass(frozen=True)
class Release:
    """An official Julia archive and its published SHA-256.

    Attributes
    ----------
    url : str
        Official release archive URL.
    sha256 : str
        Expected hexadecimal SHA-256 digest.
    """

    url: str
    sha256: str


# Verified against both official files on 2026-11-10:
# https://julialang-s3.julialang.org/bin/checksums/julia-1.13.3.sha256
# https://julialang-s3.julialang.org/bin/versions.json
RELEASES: dict[str, Release] = {
    "linux-x86_64": Release(
        "https://julialang-s3.julialang.org/bin/linux/x64/1.13/"
        "julia-1.13.3-linux-x86_64.tar.gz",
        "5c61cb4b40332cb5a3504e7779e2f5dac093caa4a1116ded19f01413f85c63e4",
    ),
    "linux-aarch64": Release(
        "https://julialang-s3.julialang.org/bin/linux/aarch64/1.13/"
        "julia-1.13.3-linux-aarch64.tar.gz",
        "9ceb053bee6dfb8189bd40afd6925de3a9b7ff8fac8d01a5fcdd84af171ddf03",
    ),
    "darwin-aarch64": Release(
        "https://julialang-s3.julialang.org/bin/mac/aarch64/1.13/"
        "julia-1.13.3-macaarch64.tar.gz",
        "b9f0e8450fbcb392aa3e8b43fa1fc09993665a31bf705d7ef8084f0ac4ad3765",
    ),
    "win32-x86_64": Release(
        "https://julialang-s3.julialang.org/bin/winnt/x64/1.13/julia-1.13.3-win64.zip",
        "1fae8fc95e6199b3d8f6d7a3791810e45e1b3c480eb4b9f5bb211789badfc729",
    ),
}


def host_platform() -> str:
    """Return a supported host key, refusing unsupported architectures by name.

    Returns
    -------
    str
        Normalized operating system and architecture key.

    Raises
    ------
    BackendError
        If no pinned archive supports this operating system and architecture.
    """
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    key = f"{sys.platform}-{machine}"
    if key not in RELEASES:
        raise BackendError(
            f"the julia backend does not support {key}; supported setup platforms "
            f"are {', '.join(RELEASES)}"
        )
    return key


def same_minor(version: str) -> bool:
    """Return whether an installed release belongs to the selected Julia minor.

    Parameters
    ----------
    version : str
        Julia version string to check.

    Returns
    -------
    bool
        Whether the version shares the pinned major and minor release.
    """
    match = re.fullmatch(r"(\d+)\.(\d+)\.\d+(?:[-+][\w.]+)?", version)
    return match is not None and match.groups() == tuple(JULIA_VERSION.split(".")[:2])


def device_refusal(device: str, host: str) -> str | None:
    """Return the named platform or system-library prerequisite that is missing.

    Parameters
    ----------
    device : str
        Requested device environment.
    host : str
        Normalized operating system and architecture key.

    Returns
    -------
    str or None
        An actionable refusal, or ``None`` when static prerequisites are present.
        Device hardware is discovered later by the requested Julia extension.
    """
    if device not in DEVICES:
        return f"the julia backend has no device {device!r}; choose {DEVICES}"
    if device == "metal" and host != "darwin-aarch64":
        return "the julia metal device requires macOS on Apple silicon"
    if device in {"cuda", "rocm"} and not host.startswith("linux-"):
        return f"the julia {device} device is supported on Linux only"
    if device == "rocm":
        roots = [
            Path(one)
            for one in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if one
        ]
        roots.extend((Path("/usr/lib"), Path("/usr/lib/x86_64-linux-gnu")))
        installations = [Path(os.environ.get("ROCM_PATH", "/opt/rocm"))]
        installations.extend(Path("/opt").glob("rocm-*"))
        roots.extend(root / lib for root in installations for lib in ("lib", "lib64"))
        missing = [
            name
            for name in ("amdhip64", "hsa-runtime64", "rocblas", "rocfft")
            if not any(any(root.glob(f"lib{name}.so*")) for root in roots)
        ]
        if missing:
            return (
                "the julia rocm device needs discoverable system ROCm libraries; "
                f"missing {', '.join(missing)} (set ROCM_PATH or LD_LIBRARY_PATH)"
            )
    return None
