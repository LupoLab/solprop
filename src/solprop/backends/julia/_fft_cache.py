# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Persist the Julia preparation's measured FFTW plans by transform (ADR-076).

The numerical engine plans against its own arrays. This module only identifies,
loads and publishes detached wisdom records after successful preparation. No
native runtime, Julia runtime or calibration winner is imported here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import tempfile
import threading
from collections.abc import Generator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from solprop.calibration import store_directory
from solprop.exceptions import BackendError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses atomic replacement without flock
    fcntl = None

__all__ = [
    "FFTCache",
    "FFTEngineIdentity",
    "FFTFingerprint",
    "clear_fft_cache",
    "fingerprint",
    "forget_fft_cache",
    "load_fft_cache",
    "publish_fft_cache",
    "refresh_fft_cache",
]

type _Effort = Literal["estimate", "measure", "patient"]
type _Version = tuple[int, int, int, int, int]

_FORMAT = 1
_LOCK = threading.Lock()
_read: dict[tuple[str, Path], tuple[_Version, dict[str, str]]] = {}
_measured: dict[str, dict[str, str]] = {}
logger = logging.getLogger(__name__)


def _digest(value: str, length: int, name: str) -> None:
    """Validate a canonical lowercase content identity.

    Parameters
    ----------
    value : str
        Supplied hexadecimal digest.
    length : int
        Required digit count.
    name : str
        Identity field used in an admission error.
    """
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise BackendError(
            f"julia FFT {name} needs a {length}-digit hexadecimal digest"
        )


def _pairs(
    pairs: tuple[tuple[str, str], ...], length: int, name: str
) -> tuple[tuple[str, str], ...]:
    """Validate and sort immutable named identities.

    Parameters
    ----------
    pairs : tuple
        Named content identities.
    length : int
        Required hexadecimal digit count.
    name : str
        Identity family used in an admission error.

    Returns
    -------
    tuple
        Distinct nonempty names and validated digests, sorted by name.
    """
    if not pairs or len({key for key, _ in pairs}) != len(pairs):
        raise BackendError(f"julia FFT {name} must have distinct named entries")
    for key, digest in pairs:
        if not key:
            raise BackendError(f"julia FFT {name} has an empty entry name")
        _digest(digest, length, name)
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class FFTEngineIdentity:
    """Describe the verified engine and loaded FFT libraries without runtime handles.

    Attributes
    ----------
    engine_digest, julia_binary_digest : str
        SHA-256 identities of the retained engine and actual Julia executable.
    julia_version, cpu_target, cpu_name, machine : str
        Exact Julia version, selected compilation target, actual runtime CPU name
        and Julia machine triplet.
    provider : str
        The provider loaded by FFTW.jl, normally ``"fftw"``.
    fftw_package_version, fftw_jll_version : str
        Exact versions from the verified environment.
    fftw_package_tree, fftw_jll_tree : str
        Git tree SHA-1 identities of those two packages.
    fftw_library_version : str
        The actual FFTW C library version reported by the loaded provider.
    artifacts : tuple
        Named, selected artifact Git tree SHA-1 identities.
    libraries : tuple
        ``("float32", SHA256)`` and ``("float64", SHA256)`` identities of the
        actual selected library files, matched to the verified inventory.
    preferences_digest : str
        SHA-256 of the effective owned preferences, including the provider.
    """

    engine_digest: str
    julia_version: str
    julia_binary_digest: str
    cpu_target: str
    cpu_name: str
    machine: str
    provider: str
    fftw_package_version: str
    fftw_package_tree: str
    fftw_jll_version: str
    fftw_jll_tree: str
    fftw_library_version: str
    artifacts: tuple[tuple[str, str], ...]
    libraries: tuple[tuple[str, str], ...]
    preferences_digest: str

    def __post_init__(self) -> None:
        """Validate complete supplied identities before any cache lookup."""
        for name in ("engine_digest", "julia_binary_digest", "preferences_digest"):
            _digest(getattr(self, name), 64, name)
        for name in ("fftw_package_tree", "fftw_jll_tree"):
            _digest(getattr(self, name), 40, name)
        for name in (
            "julia_version",
            "cpu_target",
            "cpu_name",
            "machine",
            "provider",
            "fftw_package_version",
            "fftw_jll_version",
            "fftw_library_version",
        ):
            if not getattr(self, name):
                raise BackendError(f"julia FFT identity needs {name}")
        object.__setattr__(self, "artifacts", _pairs(self.artifacts, 40, "artifacts"))
        libraries = _pairs(self.libraries, 64, "libraries")
        if tuple(key for key, _ in libraries) != ("float32", "float64"):
            raise BackendError("julia FFT identity needs both precision libraries")
        object.__setattr__(self, "libraries", libraries)


@dataclass(frozen=True)
class FFTFingerprint:
    """Own an immutable machine, runtime and library identity.

    Attributes
    ----------
    digest : str
        SHA-256 of the canonical identity JSON.
    canonical : str
        Canonical JSON, including the copied pure topology record and OS/architecture.
    """

    digest: str
    canonical: str


@dataclass(frozen=True)
class FFTCache:
    """Own one preparation's detached offered wisdom and publication destination.

    Attributes
    ----------
    fingerprint : FFTFingerprint
        Complete machine and engine identity.
    path : pathlib.Path or None
        Bound absolute publication path, or ``None`` for process-only caching.
    offered : tuple
        Owned ``(transform key, wisdom string)`` pairs, sorted by key.
    file_supplied : bool
        Whether a compatible file supplied any plan.
    effort : str
        Explicit ``estimate``, ``measure`` or ``patient`` preparation effort.
    """

    fingerprint: FFTFingerprint
    path: Path | None
    offered: tuple[tuple[str, str], ...]
    file_supplied: bool
    effort: _Effort


def fingerprint(
    identity: FFTEngineIdentity, topology: Mapping[str, object]
) -> FFTFingerprint:
    """Copy and fingerprint a verified engine and the pure topology record.

    Parameters
    ----------
    identity : FFTEngineIdentity
        Actual loaded-library and immutable setup identity.
    topology : Mapping[str, object]
        Finite JSON values from the shared topology record; no runtime is imported.

    Returns
    -------
    FFTFingerprint
        Detached canonical JSON and its digest, independent of mapping order.

    Raises
    ------
    BackendError
        If the supplied topology cannot be represented as finite JSON.
    """
    record = {
        "format": _FORMAT,
        "engine": asdict(identity),
        "topology": dict(topology),
        "system": platform.system(),
        "architecture": platform.machine(),
    }
    try:
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise BackendError(
            "julia FFT topology must contain finite JSON values"
        ) from error
    return FFTFingerprint(
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical
    )


def _effort(environ: Mapping[str, str]) -> _Effort:
    """Resolve the shared explicit FFTW effort policy.

    Parameters
    ----------
    environ : Mapping[str, str]
        Environment policy.

    Returns
    -------
    str
        The requested effort, or measure when unset.
    """
    value = environ.get("SOLPROP_FFTW_PLAN", "") or "measure"
    if value not in ("estimate", "measure", "patient"):
        raise BackendError(
            "julia SOLPROP_FFTW_PLAN must be estimate, measure or patient"
        )
    return cast(_Effort, value)


def _path(found: FFTFingerprint, environ: Mapping[str, str]) -> Path | None:
    """Bind the fingerprint's default or overridden publication path.

    Parameters
    ----------
    found : FFTFingerprint
        Complete prepared engine identity.
    environ : Mapping[str, str]
        Environment policy.

    Returns
    -------
    pathlib.Path or None
        Absolute file path, or no file when disabled/unusable.
    """
    named = environ.get("SOLPROP_FFTW_WISDOM", "").strip()
    try:
        if named:
            base = Path(named).expanduser()
            if not base.name or base.name in {".", ".."}:
                raise ValueError("the wisdom override names no file")
            return base.with_name(f"{base.name}.julia-{found.digest}").absolute()
        mode = environ.get("SOLPROP_CALIBRATION", "").strip().lower() or "auto"
        if mode not in ("auto", "use", "off"):
            raise ValueError("SOLPROP_CALIBRATION must be auto, use or off")
        if mode == "off":
            return None
        return (
            store_directory(environ)
            / "machines"
            / f"julia-{found.digest}"
            / "wisdom.json"
        ).absolute()
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(
            "julia: cannot keep FFTW plans (%s); planning without a file", error
        )
        return None


def _version(path: Path) -> _Version | None:
    """Identify a complete cache file without opening its contents.

    Parameters
    ----------
    path : pathlib.Path
        Bound cache location.

    Returns
    -------
    tuple or None
        Device, inode, modification/change times and size, or absent file.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    # Atomic replacements can deliberately retain mtime and size; their inode or
    # ctime still identifies the new complete file on ordinary local filesystems.
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _plans(value: object) -> dict[str, str]:
    """Copy a validated map of transform keys to wisdom strings.

    Parameters
    ----------
    value : object
        Parsed JSON or copied candidate mapping.

    Returns
    -------
    dict
        Independent string pairs.
    """
    if not isinstance(value, dict):
        raise ValueError("plans must be a mapping of text keys to wisdom strings")
    found = cast(dict[object, object], value)
    if not all(
        isinstance(key, str) and isinstance(plan, str) for key, plan in found.items()
    ):
        raise ValueError("plans must be a mapping of text keys to wisdom strings")
    return dict(cast(dict[str, str], found))


def _parse(path: Path, text: str, found: FFTFingerprint) -> dict[str, str]:
    """Ignore incompatible or malformed records with an actionable warning.

    Parameters
    ----------
    path : pathlib.Path
        Source named in a diagnostic.
    text : str
        Complete JSON contents.
    found : FFTFingerprint
        Required machine and engine identity.

    Returns
    -------
    dict
        Compatible string pairs, or an empty mapping after a warning.
    """
    try:
        raw: object = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("the record is not a mapping")
        data = cast(dict[str, object], raw)
        if type(data.get("format")) is not int or data["format"] != _FORMAT:
            raise ValueError("the cache format is incompatible")
        if (
            data.get("fingerprint") != found.digest
            or data.get("identity") != found.canonical
        ):
            raise ValueError("the machine, runtime or FFT library identity differs")
        return _plans(data.get("plans"))
    except ValueError as error:
        logger.warning("julia: ignoring FFTW plans in %s: %s", path, error)
        return {}


def _load(path: Path, found: FFTFingerprint) -> dict[str, str]:
    """Read a changed complete file while the process cache lock is held.

    Parameters
    ----------
    path : pathlib.Path
        Bound cache location.
    found : FFTFingerprint
        Required machine and engine identity.

    Returns
    -------
    dict
        Private cached string pairs, copied by the outer caller before release.
    """
    try:
        version = _version(path)
        if version is None:
            return {}
        cached = _read.get((found.digest, path))
        if cached is not None and cached[0] == version:
            return cached[1]
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("julia: cannot read FFTW plans from %s: %s", path, error)
        return {}
    plans = _parse(path, text, found)
    _read[found.digest, path] = version, plans
    return plans


def load_fft_cache(
    identity: FFTEngineIdentity,
    topology: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> FFTCache:
    """Offer this fingerprint's file plans beneath its own process measurements.

    Parameters
    ----------
    identity : FFTEngineIdentity
        Actual loaded engine identity.
    topology : Mapping[str, object]
        The shared pure topology record.
    environ : Mapping[str, str] or None, optional
        Environment policy, using the process environment when omitted.

    Returns
    -------
    FFTCache
        Immutable offered plans and publication path. Unusable files warn and
        planning continues; ``off`` without an override uses process plans alone.
    """
    environment = os.environ if environ is None else environ
    effort = _effort(environment)
    found = fingerprint(identity, topology)
    path = _path(found, environment)
    with _LOCK:
        stored = _load(path, found) if path is not None else {}
        offered = stored | _measured.get(found.digest, {})
        return FFTCache(
            found, path, tuple(sorted(offered.items())), bool(stored), effort
        )


def refresh_fft_cache(cache: FFTCache) -> FFTCache:
    """Refresh offers for a later start using the preparation's bound identity.

    Parameters
    ----------
    cache : FFTCache
        Previously verified identity, destination and planning effort.

    Returns
    -------
    FFTCache
        New immutable offers, including successful measurements made after the
        original preparation and any replaced complete cache file. Clearing the
        cache removes old offers even from a still-live prepared owner.
    """
    found, path = cache.fingerprint, cache.path
    with _LOCK:
        stored = _load(path, found) if path is not None else {}
        offered = stored | _measured.get(found.digest, {})
        return FFTCache(
            found, path, tuple(sorted(offered.items())), bool(stored), cache.effort
        )


@contextlib.contextmanager
def _exclusive(path: Path) -> Generator[None]:
    """Serialize cooperating writers through read, merge and atomic replacement.

    Parameters
    ----------
    path : pathlib.Path
        Publication path; the writer lock sits beside it.

    Yields
    ------
    None
        While this process owns the cooperative file lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f"{path.name}.lock").open("a", encoding="ascii") as lock:
        if fcntl is not None:  # pragma: no branch - Windows has no flock
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _replace(path: Path, text: str) -> None:
    """Publish one complete UTF-8 record and clean every unpublished temporary.

    Parameters
    ----------
    path : pathlib.Path
        Complete-file destination.
    text : str
        Serialized UTF-8 JSON record.
    """
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            partial = Path(stream.name)
            stream.write(text)
            stream.flush()
        partial.replace(path)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def publish_fft_cache(cache: FFTCache, measured: Mapping[str, str]) -> bool:
    """Remember successful measurements and merge them into the optional file.

    Parameters
    ----------
    cache : FFTCache
        Detached identity and destination from preparation.
    measured : Mapping[str, str]
        Only this successful preparation's measured wisdom, by transform key.

    Returns
    -------
    bool
        Whether the file changed. A failed publication warns, leaves no partial
        file, and retains the measurements for later preparations in this process.

    Raises
    ------
    BackendError
        If measured records are not text pairs. Call only after preparation succeeds.
    """
    try:
        new = _plans(dict(measured))
    except ValueError as error:
        raise BackendError("julia measured FFT plans must be text pairs") from error
    if not new:
        return False
    found, path = cache.fingerprint, cache.path
    with _LOCK:
        _measured.setdefault(found.digest, {}).update(new)
        if path is None:
            return False
        try:
            with _exclusive(path):
                current = (
                    _parse(path, path.read_text(encoding="utf-8"), found)
                    if _version(path) is not None
                    else {}
                )
                merged = current | new
                changed = merged != current
                if changed:
                    record = {
                        "format": _FORMAT,
                        "fingerprint": found.digest,
                        "identity": found.canonical,
                        "plans": merged,
                    }
                    _replace(path, json.dumps(record, sort_keys=True, indent=0))
                version = _version(path)
        except (OSError, UnicodeDecodeError, MemoryError) as error:
            logger.warning("julia: cannot store FFTW plans in %s: %s", path, error)
            return False
        if version is not None:
            _read[found.digest, path] = version, merged
        return changed


def forget_fft_cache(fingerprint: str | None = None) -> None:
    """Clear process measurements and file-read state for a calibration-clear hook.

    Parameters
    ----------
    fingerprint : str or None, optional
        One fingerprint to forget, or every Julia FFT fingerprint when omitted.
        Files and already-detached preparation records are unchanged.
    """
    with _LOCK:
        if fingerprint is None:
            _measured.clear()
            _read.clear()
        else:
            _measured.pop(fingerprint, None)
            for key in tuple(_read):
                if key[0] == fingerprint:
                    del _read[key]


def clear_fft_cache(environ: Mapping[str, str] | None = None) -> int:
    """Clear this machine's Julia plan files and process offers without loading Julia.

    Parameters
    ----------
    environ : mapping of str to str or None, optional
        Cache destination policy; the process environment by default.

    Returns
    -------
    int
        Removed compatible Julia cache files. Unrelated, malformed and inaccessible
        files remain untouched. Existing numerical plans continue to be usable.
    """
    # These are shared pure machine records, not the native execution extension.
    from solprop.calibration import _native  # noqa: PLC0415 - lazy clear operation

    environment = os.environ if environ is None else environ
    forget_fft_cache()
    try:
        topology = _native.topology.read_topology().record()
    except BackendError:
        return 0
    candidates: set[Path] = set()
    try:
        candidates.update(
            store_directory(environment).glob("machines/julia-*/wisdom.json")
        )
        named = environment.get("SOLPROP_FFTW_WISDOM", "").strip()
        if named:
            base = Path(named).expanduser().absolute()
            prefix = f"{base.name}.julia-"
            candidates.update(
                path
                for path in base.parent.iterdir()
                if path.name.startswith(prefix)
                and len(path.name) == len(prefix) + 64
                and all(char in "0123456789abcdef" for char in path.name[len(prefix) :])
            )
    except OSError, RuntimeError, ValueError:
        # The other destination can still contain readable, removable plans.
        pass
    removed = 0
    for path in candidates:
        try:
            if not _clearable(path, topology):
                continue
            with _exclusive(path):
                if _clearable(path, topology):
                    path.unlink()
                    removed += 1
        except OSError, UnicodeDecodeError, ValueError:
            continue
    return removed


def _clearable(path: Path, topology: Mapping[str, object]) -> bool:
    """Match complete owned metadata before removing a stored plan file.

    Parameters
    ----------
    path : pathlib.Path
        Candidate default or explicitly named Julia cache file.
    topology : mapping
        Current machine record from the backend-neutral topology reader.

    Returns
    -------
    bool
        Whether a valid Julia fingerprint identifies this machine and platform.
    """
    record: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        return False
    data = cast(dict[str, object], record)
    canonical, digest = data.get("identity"), data.get("fingerprint")
    if (
        type(data.get("format")) is not int
        or data["format"] != _FORMAT
        or not isinstance(canonical, str)
        or not isinstance(digest, str)
        or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest
        or not (
            path.parent.name == f"julia-{digest}"
            or path.name.endswith(f".julia-{digest}")
        )
    ):
        return False
    identity: object = json.loads(canonical)
    if not isinstance(identity, dict):
        return False
    found = cast(dict[str, object], identity)
    return (
        found.get("format") == _FORMAT
        and found.get("system") == platform.system()
        and found.get("architecture") == platform.machine()
        and found.get("topology") == json.loads(json.dumps(dict(topology)))
    )
