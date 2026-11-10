# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hash the bundled Julia engine without importing solprop or a Julia runtime.

The wheel build hook loads this file by path. Keep it standard-library only so that
the build and an editable installation use exactly the same source identity.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes.

    Parameters
    ----------
    path : pathlib.Path
        File whose raw bytes determine the digest.

    Returns
    -------
    str
        The hexadecimal SHA-256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_files(bundle: Path, paths: Iterable[Path]) -> str:
    """Hash sorted relative paths and their contents, without ambiguous separators.

    Parameters
    ----------
    bundle : pathlib.Path
        Root containing every input file.
    paths : collections.abc.Iterable of pathlib.Path
        Files to hash. Symlinks must resolve inside `bundle`.

    Returns
    -------
    str
        The SHA-256 digest, independent of the root's absolute location.
    """
    digest = hashlib.sha256()
    root = bundle.resolve()
    for path in sorted(paths):
        path.resolve().relative_to(root)
        name = path.relative_to(bundle).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def engine_digest(bundle: Path) -> str:
    """Return the identity of the engine sources and Python bridge contract.

    Parameters
    ----------
    bundle : pathlib.Path
        The Julia package root, from an editable checkout or wheel package data.

    Returns
    -------
    str
        A SHA-256 hexadecimal digest of ``Project.toml``, ``bridge.toml`` and every
        file under ``src`` and ``ext``.

    Raises
    ------
    OSError
        If a required source is absent or cannot be read.
    ValueError
        If a source symlink escapes the bundle.
    """
    paths = engine_files(bundle)
    if not any(path.suffix == ".jl" for path in paths):
        raise FileNotFoundError(f"the Julia engine has no source files in {bundle}")
    return digest_files(bundle, paths)


def engine_files(bundle: Path) -> tuple[Path, ...]:
    """Return the engine files covered by its source digest.

    Parameters
    ----------
    bundle : pathlib.Path
        Julia package root.

    Returns
    -------
    tuple of pathlib.Path
        Project, bridge contract and every source or extension file.
    """
    paths = [bundle / "Project.toml", bundle / "bridge.toml"]
    paths.extend(
        path
        for directory in ("src", "ext")
        for path in (bundle / directory).rglob("*")
        if path.is_file()
    )
    return tuple(paths)


def source_bundle() -> Path:
    """Locate wheel package data, or the engine in this editable checkout.

    Returns
    -------
    pathlib.Path
        The source root. Packaged sources must match the wheel's generated digest.

    Raises
    ------
    OSError
        If the packaged digest or required source file is absent.
    ValueError
        If packaged source bytes do not match their generated identity.
    """
    package = Path(__file__).resolve().parent
    checkout = package.parents[3]
    editable = checkout / "julia"
    if (checkout / "pyproject.toml").is_file() and editable.is_dir():
        return editable
    bundle = package / "_bundle"
    expected = (package / "_engine_digest.txt").read_text(encoding="utf-8").strip()
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or engine_digest(bundle) != expected
    ):
        raise ValueError(
            "the packaged Julia engine differs from its wheel digest; reinstall solprop"
        )
    return bundle


def setup_files(bundle: Path, devices: tuple[str, ...]) -> tuple[Path, ...]:
    """Return the installation inputs beside the separately hashed engine sources.

    Parameters
    ----------
    bundle : pathlib.Path
        Julia package root.
    devices : tuple of str
        Selected device environments in canonical order.

    Returns
    -------
    tuple of pathlib.Path
        Installer scripts, projects, manifests and recorded toolkit defaults.
    """
    files = [bundle / "environments" / "versions.toml"]
    files.extend(path for path in (bundle / "scripts").rglob("*") if path.is_file())
    if bundle / "scripts" / "instantiate.jl" not in files:
        files.append(bundle / "scripts" / "instantiate.jl")
    for device in devices:
        project = bundle / "environments" / device
        files.extend((project / "Project.toml", project / "Manifest.toml"))
        runtime = project / "runtime.toml"
        if runtime.is_file():
            files.append(runtime)
    return tuple(files)
