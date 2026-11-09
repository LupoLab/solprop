# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What makes a calibration's measurements portable (solprop SEP 46 section 5).

A fingerprint digests the machine (its processor model, its cores of each kind, its
cache, NUMA and socket topology and SMT), the operating system, the native build and
the calibration's own version. The build is read from ``_build.json``, which the
build writes beside the extension, and from a digest of the chosen extension file, so
that nothing loads the extension (and its OpenMP runtime) to compute it. A new
release, a rebuilt extension, a changed calibration or a different machine finds no
matching profile and runs on the built-in table: nothing stale is ever applied.

This module is standard library only and loads nothing of the package.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ._topology import Topology

__all__ = ["CALIBRATION_VERSION", "Fingerprint", "build_identity", "fingerprint"]

CALIBRATION_VERSION = 2
"""The calibration's own version, raised when its probes, problems or rules change.

2: the transforms' grain moves by shape and no plan is captured at patient (solprop
ADR-075)."""


@dataclass(frozen=True)
class Fingerprint:
    """A machine and build's identity, and its store key.

    Attributes
    ----------
    digest : str
        The store's directory name for this machine and build: 24 hexadecimal digits.
    machine : dict
        The topology's record, the operating system and the architecture.
    build : dict
        The build record, the extension's variant, digest and version.
    calibration : int
        :data:`CALIBRATION_VERSION`.
    """

    digest: str
    machine: dict[str, object]
    build: dict[str, object]
    calibration: int

    def record(self) -> dict[str, object]:
        """Return the fingerprint as JSON data."""
        return {
            "digest": self.digest,
            "machine": self.machine,
            "build": self.build,
            "calibration": self.calibration,
        }


def build_identity(packages: Sequence[Path], stem: str = "_core") -> dict[str, object]:
    """Return the native build's identity without loading it.

    `packages` are the ``solprop_native`` package's directories (an editable install
    keeps its sources and its built files apart) and `stem` the chosen extension's
    (``_cpu.select_variant``). The extension file is digested whole: two installs of
    one wheel agree, any rebuild differs.
    """
    record: object = None
    for directory in packages:
        try:
            record = json.loads((directory / "_build.json").read_text(encoding="utf-8"))
            break
        except OSError, ValueError:
            continue
    files = sorted(
        path
        for directory in packages
        for path in directory.glob(f"{stem}.*")
        if path.suffix in {".so", ".pyd", ".dylib"}
    )
    digest = None
    if files:
        hasher = hashlib.blake2b(digest_size=16)
        with files[0].open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    try:
        release = version("solprop-native")
    except PackageNotFoundError:
        release = None
    return {
        "record": record,
        "variant": stem,
        "extension": digest,
        "version": release,
    }


def fingerprint(
    topology: Topology,
    build: dict[str, object],
    system: str | None = None,
    machine: str | None = None,
) -> Fingerprint:
    """Return the fingerprint of `topology` running `build` on this system."""
    identity = topology.record() | {
        "system": system or platform.system(),
        "architecture": machine or platform.machine(),
    }
    text = json.dumps(
        {"machine": identity, "build": build, "calibration": CALIBRATION_VERSION},
        sort_keys=True,
    )
    return Fingerprint(
        digest=hashlib.blake2b(text.encode(), digest_size=12).hexdigest(),
        machine=identity,
        build=build,
        calibration=CALIBRATION_VERSION,
    )
