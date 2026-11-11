# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Match loaded FFTW libraries to setup evidence before offering persistent plans."""

import hashlib
import json
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from solprop.backends.julia._artifacts import (
    child,
    mapping,
    string_list,
    table_list,
    text_value,
)
from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._fft_cache import (
    FFTCache,
    FFTEngineIdentity,
    load_fft_cache,
)
from solprop.backends.julia._identity import SetupRecord
from solprop.backends.julia._ownership import JuliaHandle
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._source import file_digest
from solprop.calibration import _native
from solprop.exceptions import BackendError

__all__ = ["FFTPlanRecord", "load_planning_cache", "read_planning", "verified_identity"]

logger = logging.getLogger(__name__)
_FACTS = (
    "julia_version",
    "julia_machine",
    "julia_cpu_name",
    "provider",
    "fftw_package_version",
    "fftw_library_version",
    "library_f64",
    "library_f32",
)


def _runtime_facts(runtime: JuliaRuntime) -> dict[str, str]:
    """Copy actual FFT provider identities without retaining Julia wrappers.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized runtime, inside its guarded entry boundary.

    Returns
    -------
    dict
        Detached strings identifying the loaded provider and libraries.
    """
    record = runtime.engine.fft_runtime_identity()
    return {name: str(getattr(record, name)) for name in _FACTS}


def _package(dependencies: dict[str, object], name: str) -> dict[str, object]:
    """Require one resolved dependency record for an FFT identity component.

    Parameters
    ----------
    dependencies : dict
        Parsed manifest dependency table.
    name : str
        Required package name.

    Returns
    -------
    dict
        The unique resolved package record.

    Raises
    ------
    BackendError
        If the package is absent or ambiguous.
    """
    packages = table_list(dependencies, name)
    if len(packages) != 1:
        raise BackendError(f"Julia FFT identity needs one resolved {name} package")
    return packages[0]


def _preferences(setup: SetupRecord, provider: str) -> str:
    """Hash selected project preferences together with the actual FFT provider.

    Parameters
    ----------
    setup : SetupRecord
        Verified installed project stack.
    provider : str
        Actual loaded FFT provider.

    Returns
    -------
    str
        SHA-256 of provider and each project's optional preference file digest.
    """
    record = {
        "provider": provider,
        "projects": [
            (device, file_digest(path) if path.is_file() else None)
            for device, project in setup.projects
            for path in (project / "LocalPreferences.toml",)
        ],
    }
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def verified_identity(
    setup: SetupRecord, loaded: Mapping[str, str]
) -> FFTEngineIdentity:
    """Fingerprint actual FFT libraries only after matching their selected inventory.

    Parameters
    ----------
    setup : SetupRecord
        Complete installation already verified before runtime import.
    loaded : mapping of str to str
        Detached facts returned by the running engine's ``fft_runtime_identity``.

    Returns
    -------
    FFTEngineIdentity
        Actual runtime and file-byte identities, without installation-specific paths.

    Raises
    ------
    BackendError
        For a runtime/package mismatch or a loaded library outside selected files.
    OSError
        If verified evidence or a loaded library is no longer readable.
    """
    if set(loaded) != set(_FACTS) or loaded["julia_version"] != setup.julia_version:
        raise BackendError("Julia FFT runtime identity disagrees with its setup")
    if loaded["provider"] != "fftw":
        raise BackendError("Julia FFT planning requires the inventoried FFTW provider")
    project = dict(setup.projects)["cpu"]
    with (project / "Manifest.toml").open("rb") as stream:
        manifest = mapping(tomllib.load(stream), "manifest")
    dependencies = mapping(manifest.get("deps"), "manifest dependencies")
    fftw, jll = _package(dependencies, "FFTW"), _package(dependencies, "FFTW_jll")
    if text_value(fftw, "version") != loaded["fftw_package_version"]:
        raise BackendError("the loaded Julia FFTW package differs from its manifest")
    inventory = next(path for device, path, _ in setup.inventories if device == "cpu")
    with inventory.open("rb") as stream:
        recorded = mapping(tomllib.load(stream), "inventory")
    selected = tuple(
        artifact
        for artifact in table_list(recorded, "artifacts")
        if text_value(artifact, "package_uuid") == text_value(jll, "uuid")
    )
    if not selected:
        raise BackendError("Julia FFTW libraries have no selected artifact inventory")
    permitted = {
        child(child(setup.depot, text_value(artifact, "path")), name).resolve()
        for artifact in selected
        for name in string_list(artifact, "files")
    }
    libraries: list[tuple[str, str]] = []
    for precision, key in (("float64", "library_f64"), ("float32", "library_f32")):
        path = Path(loaded[key])
        if not path.is_absolute() or path.resolve() not in permitted:
            raise BackendError(
                f"Julia FFTW {precision} library is outside its selected artifact files"
            )
        libraries.append((precision, file_digest(path)))
    return FFTEngineIdentity(
        setup.engine_digest,
        setup.julia_version,
        setup.julia_binary_digest,
        setup.cpu_target,
        loaded["julia_cpu_name"],
        loaded["julia_machine"],
        loaded["provider"],
        text_value(fftw, "version"),
        text_value(fftw, "git-tree-sha1"),
        text_value(jll, "version"),
        text_value(jll, "git-tree-sha1"),
        loaded["fftw_library_version"],
        tuple(
            (text_value(one, "name"), text_value(one, "tree_hash")) for one in selected
        ),
        tuple(libraries),
        _preferences(setup, loaded["provider"]),
    )


def load_planning_cache(runtime: JuliaRuntime) -> FFTCache:
    """Offer plans identified by verified runtime files and the shared CPU topology.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime whose complete setup has been checked.

    Returns
    -------
    FFTCache
        Detached cache offers. A lightweight installation without native metadata
        keeps plans in this process only; it never imports a native runtime.
    """
    runtime.check()
    loaded = invoke(lambda: _runtime_facts(runtime), "Julia FFT identity read failed")
    identity = verified_identity(runtime.setup, loaded)
    environment = dict(os.environ)
    try:
        topology = _native.topology.read_topology().record()
    except BackendError:
        # The metadata-only loader is absent in a lightweight installation. Do not
        # invent a second canonical topology or persist an incomplete machine key.
        topology = {"unavailable": "native distribution metadata"}
        environment["SOLPROP_CALIBRATION"] = "off"
        environment.pop("SOLPROP_FFTW_WISDOM", None)
        logger.warning(
            "julia: CPU topology metadata is unavailable; "
            "FFTW plans stay in this process"
        )
    return load_fft_cache(identity, topology, environ=environment)


@dataclass(frozen=True)
class FFTPlanRecord:
    """Own facts about plans actually created by one successful start.

    Attributes
    ----------
    measured, applied, rejected : tuple
        Detached key/string pairs for new wisdom, applied wisdom and cache refusals.
    reused : tuple of str
        Sorted transform keys actually recovered from offered wisdom.
    """

    measured: tuple[tuple[str, str], ...]
    applied: tuple[tuple[str, str], ...]
    rejected: tuple[tuple[str, str], ...]
    reused: tuple[str, ...]


def _planning_record(runtime: JuliaRuntime, handle: JuliaHandle) -> FFTPlanRecord:
    """Detach successful planning facts before releasing their owned wisdom maps.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized runtime, inside its guarded entry boundary.
    handle : JuliaHandle
        Owning planning context for this successful start.

    Returns
    -------
    FFTPlanRecord
        Owning strings for measured, applied, rejected and reused plans.
    """
    raw = runtime.engine.fft_planning_transfer(handle.borrow())
    result = FFTPlanRecord(
        tuple((str(pair[0]), str(pair[1])) for pair in raw.measured),
        tuple((str(pair[0]), str(pair[1])) for pair in raw.applied),
        tuple((str(pair[0]), str(pair[1])) for pair in raw.rejected),
        tuple(str(key) for key in raw.reused),
    )
    getattr(runtime.engine, "release_fft_planning!")(handle.borrow())
    return result


def read_planning(runtime: JuliaRuntime, handle: JuliaHandle) -> FFTPlanRecord:
    """Detach planning facts on main and release the numerical context's wisdom copies.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized process runtime.
    handle : JuliaHandle
        Context owned for this start only; failed starts never publish its facts.

    Returns
    -------
    FFTPlanRecord
        Independent immutable strings and distinct transform keys, with no foreign
        wrappers. The execution context no longer retains any wisdom strings.
    """
    runtime.check()
    return invoke(
        lambda: _planning_record(runtime, handle),
        "Julia FFT planning record read failed",
    )
