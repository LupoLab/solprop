# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Immutable identities and records for published Julia installations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from solprop.backends.julia._artifacts import child, mapping, string_list, text_value
from solprop.exceptions import BackendError


@dataclass(frozen=True)
class SetupRecord:
    """A complete Julia installation selected by its immutable identity.

    Attributes
    ----------
    setup_id : str
        SHA-256 identity of the installation's inputs.
    engine_digest : str
        Identity of the Julia sources and bridge contract.
    bundle_digest : str
        Identity of the installer and selected environment templates.
    julia_executable : pathlib.Path
        The actual Julia executable, not a mutable launcher alias.
    julia_version : str
        The exact Julia version used for installation.
    julia_binary_digest : str
        SHA-256 of the executable checked at setup.
    bridge_version : str
        The exactly paired JuliaCall and PythonCall version.
    depot : pathlib.Path
        The private package and artifact depot.
    engine : pathlib.Path
        The installation's retained copy of the engine.
    projects : tuple
        Ordered ``(device, project directory)`` pairs, CPU first.
    manifests : tuple
        Ordered ``(device, source manifest SHA-256)`` pairs.
    inventories : tuple
        Ordered ``(device, inventory path, inventory SHA-256)`` triples.
    platform : str
        The platform for which setup was prepared.
    cpu_target : str
        Julia's requested compilation target.
    cpu_name : str
        The CPU name Julia reported while preparing the environment.
    cuda_runtime : str or None
        Explicit CUDA runtime preference, or ``None`` without CUDA.
    cuda_local : bool
        Whether CUDA uses system toolkit libraries.
    elapsed_seconds : float
        The time spent creating this installation, seconds.
    """

    setup_id: str
    engine_digest: str
    bundle_digest: str
    julia_executable: Path
    julia_version: str
    julia_binary_digest: str
    bridge_version: str
    depot: Path
    engine: Path
    projects: tuple[tuple[str, Path], ...]
    manifests: tuple[tuple[str, str], ...]
    inventories: tuple[tuple[str, Path, str], ...]
    platform: str
    cpu_target: str
    cpu_name: str
    cuda_runtime: str | None
    cuda_local: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class SetupStatus:
    """A published setup's compatibility, without starting Julia.

    Attributes
    ----------
    setup_id : str
        The publication directory's identity.
    record : SetupRecord or None
        The parsed installation, or ``None`` for a malformed publication.
    compatible : bool
        Whether its entire stack matches this installation and is complete.
    reason : str or None
        The refusal explaining incompatibility.
    tested_runtime : bool
        Whether the Julia release is the pinned, qualified release.
    depot_bytes : int
        The total file size of the private depot, bytes.
    prerequisites : tuple
        ``(device, refusal or None)`` pairs for system prerequisites.
    package_versions : tuple
        ``(package name, version)`` pairs for the engine and installed device packages.
    """

    setup_id: str
    record: SetupRecord | None
    compatible: bool
    reason: str | None
    tested_runtime: bool
    depot_bytes: int
    prerequisites: tuple[tuple[str, str | None], ...]
    package_versions: tuple[tuple[str, str], ...]


def identity(inputs: dict[str, object]) -> str:
    """Hash a canonical, JSON-compatible installation-input record.

    Parameters
    ----------
    inputs : dict
        String-keyed installation inputs containing finite JSON values.

    Returns
    -------
    str
        The input record's SHA-256 identity, independent of dictionary order.
    """
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_inputs(record: SetupRecord) -> dict[str, object]:
    """Return the installation inputs that determine the published identity.

    Parameters
    ----------
    record : SetupRecord
        Planned or published installation.

    Returns
    -------
    dict
        Canonical inputs, excluding incidental timing and publication paths.
    """
    return {
        "engine_digest": record.engine_digest,
        "bundle_digest": record.bundle_digest,
        "julia_executable": str(record.julia_executable),
        "julia_version": record.julia_version,
        "julia_binary_digest": record.julia_binary_digest,
        "bridge_version": record.bridge_version,
        "platform": record.platform,
        "cpu_target": record.cpu_target,
        "cpu_name": record.cpu_name,
        "cuda_runtime": record.cuda_runtime,
        "cuda_local": record.cuda_local,
        "devices": [device for device, _ in record.projects],
        "manifests": dict(record.manifests),
    }


def write_record(record: SetupRecord, directory: Path) -> None:
    """Write the last file of a staged installation, before its atomic rename.

    Parameters
    ----------
    record : SetupRecord
        Complete installation with its final publication paths.
    directory : pathlib.Path
        Unpublished staging directory receiving the record.
    """
    published = record.engine.parents[1]
    content = record_inputs(record) | {
        "schema_version": 1,
        "complete": True,
        "setup_id": record.setup_id,
        "depot": str(record.depot),
        "engine": record.engine.relative_to(published).as_posix(),
        "projects": {
            name: path.relative_to(published).as_posix()
            for name, path in record.projects
        },
        "inventories": {
            name: {"path": path.relative_to(published).as_posix(), "sha256": digest}
            for name, path, digest in record.inventories
        },
        "elapsed_seconds": record.elapsed_seconds,
    }
    path = directory / "setup.json"
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_record(directory: Path) -> SetupRecord:
    """Read a publication and verify that its fields still encode its identity.

    Parameters
    ----------
    directory : pathlib.Path
        Published setup directory named by its identity.

    Returns
    -------
    SetupRecord
        The structurally valid installation record.

    Raises
    ------
    BackendError
        If the record is incomplete, inconsistent or does not match its identity.
    """
    with (directory / "setup.json").open(encoding="utf-8") as stream:
        content = mapping(json.load(stream), "publication")
    if (
        type(content.get("schema_version")) is not int
        or content["schema_version"] != 1
        or content.get("complete") is not True
    ):
        raise BackendError(
            "the julia setup is incomplete or has an unknown record format"
        )
    devices = string_list(content, "devices")
    projects = mapping(content.get("projects"), "projects")
    manifests = mapping(content.get("manifests"), "manifests")
    inventories = mapping(content.get("inventories"), "inventories")
    if any(set(table) != set(devices) for table in (projects, manifests, inventories)):
        raise BackendError("the julia setup has inconsistent environment records")
    runtime = content.get("cuda_runtime")
    local = content.get("cuda_local")
    seconds = content.get("elapsed_seconds")
    if (runtime is not None and not isinstance(runtime, str)) or type(
        local
    ) is not bool:
        raise BackendError("the julia setup has invalid CUDA preferences")
    if (
        not isinstance(seconds, (float, int))
        or isinstance(seconds, bool)
        or not 0 <= seconds < float("inf")
    ):
        raise BackendError("the julia setup has an invalid elapsed time")
    record = SetupRecord(
        setup_id=text_value(content, "setup_id"),
        engine_digest=text_value(content, "engine_digest"),
        bundle_digest=text_value(content, "bundle_digest"),
        julia_executable=Path(text_value(content, "julia_executable")),
        julia_version=text_value(content, "julia_version"),
        julia_binary_digest=text_value(content, "julia_binary_digest"),
        bridge_version=text_value(content, "bridge_version"),
        depot=Path(text_value(content, "depot")),
        engine=child(directory, text_value(content, "engine")),
        projects=tuple(
            (name, child(directory, text_value(projects, name))) for name in devices
        ),
        manifests=tuple((name, text_value(manifests, name)) for name in devices),
        inventories=tuple(
            (
                name,
                child(directory, text_value(mapping(inventories[name], name), "path")),
                text_value(mapping(inventories[name], name), "sha256"),
            )
            for name in devices
        ),
        platform=text_value(content, "platform"),
        cpu_target=text_value(content, "cpu_target"),
        cpu_name=text_value(content, "cpu_name"),
        cuda_runtime=runtime,
        cuda_local=local,
        elapsed_seconds=float(seconds),
    )
    if (
        record.setup_id != directory.name
        or identity(record_inputs(record)) != record.setup_id
    ):
        raise BackendError("the julia setup identity does not match its publication")
    return record
