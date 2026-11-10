# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate the installer's complete artifact inventory before Julia can load."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import cast

from solprop.backends.julia._source import file_digest
from solprop.exceptions import BackendError


class MissingDepotContentError(BackendError):
    """A wholly absent shared tree that explicit setup may safely reinstall."""


def mapping(value: object, label: str) -> dict[str, object]:
    """Read a string-keyed metadata table, refusing malformed setup metadata.

    Parameters
    ----------
    value : object
        Decoded metadata value.
    label : str
        Table description for a refusal.

    Returns
    -------
    dict
        The validated string-keyed table.
    """
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BackendError(f"the julia setup has an invalid {label} table")
    return cast(dict[str, object], value)


def text_value(table: dict[str, object], key: str) -> str:
    """Read a required nonempty string from setup metadata.

    Parameters
    ----------
    table : dict
        Decoded metadata table.
    key : str
        Required field name.

    Returns
    -------
    str
        The field's validated nonempty text.
    """
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise BackendError(f"the julia setup has an invalid {key}")
    return value


def string_list(table: dict[str, object], key: str) -> tuple[str, ...]:
    """Read a list of strings without coercing invalid metadata values.

    Parameters
    ----------
    table : dict
        Decoded metadata table.
    key : str
        Required array field name.

    Returns
    -------
    tuple of str
        The field's validated strings, preserving their order.
    """
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(one, str) for one in value):
        raise BackendError(f"the julia setup has an invalid {key} list")
    return tuple(cast(list[str], value))


def table_list(table: dict[str, object], key: str) -> tuple[dict[str, object], ...]:
    """Read a metadata array of tables.

    Parameters
    ----------
    table : dict
        Decoded metadata table.
    key : str
        Required table-array field name.

    Returns
    -------
    tuple of dict
        The validated string-keyed tables, preserving their order.
    """
    values = table.get(key)
    if not isinstance(values, list):
        raise BackendError(f"the julia setup has an invalid {key} list")
    return tuple(mapping(value, key) for value in values)


def child(root: Path, name: str) -> Path:
    """Return an inventory path contained in its declared root.

    Parameters
    ----------
    root : pathlib.Path
        Parent directory containing the inventoried resource.
    name : str
        Resource path in portable relative POSIX form.

    Returns
    -------
    pathlib.Path
        The contained path, whose symlink target must remain in `root`.
    """
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or ":" in name
    ):
        raise BackendError(
            f"the julia setup contains an unsafe inventory path {name!r}"
        )
    result = root.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise BackendError(f"the julia setup inventory escapes its root: {name!r}")
    return result


def refuse_overrides(depot: Path) -> None:
    """Refuse external artifact substitutions in the private managed depot.

    Parameters
    ----------
    depot : pathlib.Path
        Managed package and artifact depot.
    """
    overrides = depot / "artifacts" / "Overrides.toml"
    if overrides.exists() or overrides.is_symlink():
        raise BackendError(
            "the julia managed depot contains artifacts/Overrides.toml; artifact "
            "overrides are incompatible with its verified inventory"
        )


def check_inventory(path: Path, project: Path, depot: Path) -> None:
    """Check the complete selected package/artifact inventory using files alone.

    This never invokes Pkg, a subprocess, or a downloader. Verification during setup
    checks tree hashes; this preflight checks every inventoried file before a loader
    could attempt a missing lazy artifact.

    Parameters
    ----------
    path : pathlib.Path
        Completed TOML inventory emitted by the installer.
    project : pathlib.Path
        Installed application project directory.
    depot : pathlib.Path
        Shared private package and artifact depot.

    Raises
    ------
    BackendError
        If the inventory is inconsistent, a required resource is absent, or an
        override or an unsafe resource path invalidates the recorded installation.
    """
    refuse_overrides(depot)
    with path.open("rb") as stream:
        inventory = mapping(tomllib.load(stream), "inventory")
    if (
        type(inventory.get("schema_version")) is not int
        or inventory["schema_version"] != 1
        or inventory.get("complete") is not True
    ):
        raise BackendError("the julia artifact inventory was not completed")
    if text_value(inventory, "project") != project.name:
        raise BackendError(
            "the julia artifact inventory belongs to another environment"
        )
    text_value(inventory, "platform")
    for filename, key in (
        ("Project.toml", "project_sha256"),
        ("Manifest.toml", "manifest_sha256"),
    ):
        if text_value(inventory, key) != file_digest(project / filename):
            raise BackendError(
                f"the julia {project.name} {filename} changed after setup"
            )
    packages = table_list(inventory, "packages")
    artifacts = table_list(inventory, "artifacts")
    with (project / "Manifest.toml").open("rb") as stream:
        manifest = mapping(tomllib.load(stream), "manifest")
    dependencies = mapping(manifest.get("deps"), "manifest dependencies")
    expected = {
        (name, text_value(entry, "uuid"))
        for name in dependencies
        for entry in table_list(dependencies, name)
        if "git-tree-sha1" in entry
    }
    actual = {
        (text_value(entry, "name"), text_value(entry, "uuid")) for entry in packages
    }
    if actual != expected:
        raise BackendError("the julia artifact inventory omits manifest packages")
    selected = _package_selections(packages, depot)
    verified: set[tuple[str, str]] = set()
    for artifact in artifacts:
        key = _check_artifact(artifact, depot)
        if key in verified:
            raise BackendError(
                "the julia artifact inventory repeats a selected artifact"
            )
        verified.add(key)
    counts = (
        ("package_count", len(packages)),
        ("artifact_selection_count", len(selected)),
        ("artifact_verified_count", len(verified)),
    )
    if any(
        type(inventory.get(key)) is not int or inventory[key] != count
        for key, count in counts
    ):
        raise BackendError(
            "the julia artifact inventory has inconsistent completion counts"
        )
    if selected != verified:
        raise BackendError("the julia artifact inventory omits a selected artifact")


def _package_selections(
    packages: tuple[dict[str, object], ...], depot: Path
) -> set[tuple[str, str]]:
    """Validate installed packages and collect their exact artifact selections.

    Parameters
    ----------
    packages : tuple of dict
        Installer package records.
    depot : pathlib.Path
        Private depot containing the recorded package sources.

    Returns
    -------
    set of tuple
        Unique package UUID and artifact name pairs.
    """
    selections: set[tuple[str, str]] = set()
    identities: set[str] = set()
    for package in packages:
        name = text_value(package, "name")
        uuid = text_value(package, "uuid")
        if uuid in identities:
            raise BackendError("the julia artifact inventory repeats a package")
        identities.add(uuid)
        location = child(depot, text_value(package, "path"))
        if not location.exists() and not location.is_symlink():
            raise MissingDepotContentError(
                f"the julia setup is missing package {name}; run solprop julia setup"
            )
        if not (location / "Project.toml").is_file():
            raise BackendError(
                f"the julia package tree {location} is incomplete; stop its users, "
                "remove that tree, then run solprop julia setup"
            )
        selector = text_value(package, "selector")
        names = string_list(package, "selected_artifacts")
        if selector not in {"none", "default", "hook"} or (
            selector == "none" and names
        ):
            raise BackendError(
                f"the julia setup has an invalid artifact selector for {name}"
            )
        if len(names) != len(set(names)):
            raise BackendError(
                f"the julia setup repeats an artifact selection for {name}"
            )
        selections.update((uuid, artifact) for artifact in names)
    return selections


def _check_artifact(artifact: dict[str, object], depot: Path) -> tuple[str, str]:
    """Check every inventoried file of one selected and verified artifact.

    Parameters
    ----------
    artifact : dict
        Installer artifact record.
    depot : pathlib.Path
        Private depot containing the artifact tree.

    Returns
    -------
    tuple of str
        Owning package UUID and artifact name.
    """
    name = text_value(artifact, "name")
    uuid = text_value(artifact, "package_uuid")
    tree_hash = text_value(artifact, "tree_hash")
    if re.fullmatch(r"[0-9a-f]{40}", tree_hash) is None:
        raise BackendError(f"the julia artifact {name} has an invalid tree hash")
    if text_value(artifact, "path") != f"artifacts/{tree_hash}":
        raise BackendError(f"the julia artifact {name} has an inconsistent location")
    location = child(depot, f"artifacts/{tree_hash}")
    if not location.exists() and not location.is_symlink():
        raise MissingDepotContentError(
            f"the julia setup is missing artifact {name}; run solprop julia setup"
        )
    if not location.is_dir():
        raise BackendError(f"the julia artifact {name} path is not a directory")
    files = string_list(artifact, "files")
    if len(set(files)) != len(files):
        raise BackendError(f"the julia artifact {name} repeats inventoried files")
    for file in files:
        if not child(location, file).is_file():
            raise BackendError(
                f"the julia setup is missing artifact {name} file {file}; "
                f"stop its users, remove the incomplete tree {location}, "
                "then run solprop julia setup"
            )
    return uuid, name
