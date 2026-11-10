# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare and select immutable Julia environments without implicit runtime loading."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from solprop.backends.julia import _platforms, _source
from solprop.backends.julia._artifacts import (
    MissingDepotContentError,
    check_inventory,
    child,
    mapping,
    refuse_overrides,
    text_value,
)
from solprop.backends.julia._identity import (
    SetupRecord,
    SetupStatus,
    identity,
    read_record,
    record_inputs,
    write_record,
)
from solprop.calibration import store_directory
from solprop.exceptions import BackendError
from solprop.output import _file_lock


@dataclass(frozen=True)
class _Runtime:
    """An installed runtime resolved through its actual executable."""

    executable: Path
    version: str
    binary_digest: str
    cpu_name: str


@dataclass(frozen=True)
class _Request:
    """Canonical installation choices before any depot mutation."""

    devices: tuple[str, ...]
    cuda_runtime: str | None
    cuda_local: bool
    cpu_target: str


def depot_directory() -> Path:
    """Return the managed depot without creating or opening anything in it.

    Returns
    -------
    pathlib.Path
        The absolute private depot selected by the environment or calibration store.
    """
    named = os.environ.get("SOLPROP_JULIA_DEPOT")
    return (
        Path(named).expanduser() if named else store_directory() / "julia"
    ).resolve()


def setup(
    *,
    devices: Sequence[str] = ("cpu",),
    julia: Path | None = None,
    cuda_runtime: str | None = None,
    cuda_local: bool = False,
    cpu_target: str = "native",
) -> SetupRecord:
    """Install and verify a complete Julia stack, then publish it atomically.

    This explicit operation may download Julia, packages and artifacts. Calling
    :func:`status` or :func:`select_setup` never downloads anything or starts Julia.
    Existing published identities remain immutable; changed choices create another.

    Parameters
    ----------
    devices : collections.abc.Sequence of str, optional
        Requested devices: ``cpu``, ``cuda``, ``rocm`` and ``metal``. CPU is always
        included, and duplicates are removed in that canonical order.
    julia : pathlib.Path or None, optional
        An installed Julia executable of the pinned minor release. When omitted,
        use a compatible executable on PATH or download the pinned release.
    cuda_runtime : str or None, optional
        CUDA runtime release, for example ``"12.8"``. The recorded newest supported
        release is used when CUDA is requested without a local toolkit.
    cuda_local : bool, optional
        Use the system CUDA toolkit. Requires the CUDA device.
    cpu_target : str, optional
        Julia's compilation target; name a portable target for a shared cluster.

    Returns
    -------
    SetupRecord
        The complete installation and its immutable selection identity.

    Raises
    ------
    BackendError
        If the extra, runtime, platform or device prerequisites are missing, inputs
        are invalid, or installation, verification or publication fails.
    """
    started = time.monotonic()
    try:
        bridge = _bridge_version()
        host = _platforms.host_platform()
        bundle = _source.source_bundle()
        request = _request(devices, cuda_runtime, cuda_local, cpu_target, bundle)
        for device in request.devices:
            if reason := _platforms.device_refusal(device, host):
                raise BackendError(reason)
        _check_manifest_bridge(bundle, request.devices, bridge)
        depot = depot_directory()
        depot.mkdir(parents=True, exist_ok=True)
        with _locked(depot):
            refuse_overrides(depot)
            runtime = _runtime(julia, depot, host)
            record = _planned_record(bundle, depot, runtime, request, host)
            published = depot / "setups" / record.setup_id
            if published.exists():
                existing = read_record(published)
                try:
                    _check_record(existing, bundle, depot, bridge, host)
                except MissingDepotContentError:
                    return _install(record, bundle, started, repair=existing)
                return existing
            return _install(record, bundle, started)
    except (
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        raise BackendError(f"the julia setup failed: {error}") from error


def status() -> tuple[SetupStatus, ...]:
    """List published Julia setups and their compatibility using files alone.

    Returns
    -------
    tuple of SetupStatus
        Identities in lexical order, including malformed or incomplete publications
        with the reason they cannot be selected. Unpublished staging directories are
        excluded. No Julia runtime or network operation is invoked.
    """
    depot = depot_directory()
    publications = depot / "setups"
    if not publications.is_dir():
        return ()
    size = _directory_bytes(depot)
    return tuple(
        _status(directory, depot, size)
        for directory in sorted(publications.iterdir())
        if directory.is_dir() and (directory / "setup.json").is_file()
    )


def select_setup(*, setup_id: str | None = None, device: str = "cpu") -> SetupRecord:
    """Select one complete installation before starting Julia or a device extension.

    Parameters
    ----------
    setup_id : str or None, optional
        An exact identity from :func:`setup` or :func:`status`. Without one, exactly
        one compatible installation containing `device` must exist.
    device : str, optional
        The requested device, ``cpu`` by default.

    Returns
    -------
    SetupRecord
        The selected installation, after checking its entire installed stack.

    Raises
    ------
    BackendError
        If no matching setup exists, several match, a selected setup is incomplete,
        or a device prerequisite is missing.
    """
    if device not in _platforms.DEVICES:
        raise BackendError(f"the julia backend has no device {device!r}")
    found = status()
    if setup_id is not None:
        chosen = next((one for one in found if one.setup_id == setup_id), None)
        if chosen is None:
            raise BackendError(
                f"the julia setup {setup_id!r} does not exist; run solprop julia setup"
            )
        if not chosen.compatible or chosen.record is None:
            raise BackendError(
                f"the julia setup {setup_id} cannot run: {chosen.reason}"
            )
        candidates = [chosen.record]
    else:
        candidates = [
            one.record
            for one in found
            if one.compatible
            and one.record is not None
            and device in dict(one.record.projects)
        ]
    if not candidates:
        detail = next((one.reason for one in found if one.reason), "no matching setup")
        raise BackendError(
            f"the julia backend has no usable {device} setup ({detail}); "
            f"run solprop julia setup{' --device ' + device if device != 'cpu' else ''}"
        )
    if len(candidates) > 1:
        raise BackendError(
            "the julia backend has several compatible setups: "
            + ", ".join(one.setup_id for one in candidates)
            + "; select one with backend_options={'setup_id': '<identity>'}"
        )
    record = candidates[0]
    if device not in dict(record.projects):
        raise BackendError(
            f"the julia setup has no {device} environment; "
            f"run solprop julia setup --device {device}"
        )
    if reason := _platforms.device_refusal(device, record.platform):
        raise BackendError(reason)
    return record


def _bridge_version() -> str:
    """Check the Python bridge's package metadata without importing the bridge.

    Returns
    -------
    str
        The exactly pinned installed JuliaCall version.
    """
    if importlib.util.find_spec("juliacall") is None:
        raise BackendError(
            'the julia backend needs the extra: install "solprop[julia]"'
        )
    try:
        version = importlib.metadata.version("juliacall")
    except importlib.metadata.PackageNotFoundError as error:
        raise BackendError(
            'the julia bridge has no package metadata; reinstall "solprop[julia]"'
        ) from error
    if version != _platforms.BRIDGE_VERSION:
        raise BackendError(
            f"the julia bridge needs juliacall=={_platforms.BRIDGE_VERSION}, "
            f"found {version}; "
            'reinstall "solprop[julia]"'
        )
    return version


def _request(
    devices: Sequence[str],
    cuda_runtime: str | None,
    cuda_local: bool,
    cpu_target: str,
    bundle: Path,
) -> _Request:
    """Validate and canonically order setup choices before resolving their identity.

    Parameters
    ----------
    devices : collections.abc.Sequence of str
        Requested device environments.
    cuda_runtime : str or None
        Explicit toolkit release, or the bundled default when required.
    cuda_local : bool
        Whether to discover a local toolkit.
    cpu_target : str
        Julia compilation target.
    bundle : pathlib.Path
        Bundled environment defaults.

    Returns
    -------
    _Request
        Validated canonical installation choices.
    """
    if isinstance(devices, str) or any(
        device not in _platforms.DEVICES for device in devices
    ):
        raise BackendError(
            f"the julia devices must be a sequence drawn from {_platforms.DEVICES}"
        )
    ordered = tuple(
        device for device in _platforms.DEVICES if device == "cpu" or device in devices
    )
    if type(cuda_local) is not bool:
        raise BackendError("the julia cuda_local setting must be boolean")
    if (
        not isinstance(cpu_target, str)
        or not cpu_target.strip()
        or any(character in cpu_target for character in "\x00\r\n")
    ):
        raise BackendError("the julia CPU target must be a nonempty single-line target")
    if "cuda" not in ordered and (cuda_runtime is not None or cuda_local):
        raise BackendError("the julia CUDA preferences require --device cuda")
    if cuda_runtime is not None and cuda_local:
        raise BackendError("the julia CUDA runtime and local toolkit are alternatives")
    if "cuda" in ordered and cuda_runtime is None and not cuda_local:
        path = bundle / "environments" / "cuda" / "runtime.toml"
        if not path.is_file():
            raise BackendError(
                "the julia CUDA environment has no recorded default runtime; "
                "name --cuda-runtime"
            )
        with path.open("rb") as stream:
            cuda_runtime = text_value(
                mapping(tomllib.load(stream), "CUDA runtime"), "cuda_runtime"
            )
    if cuda_runtime is not None and (
        not isinstance(cuda_runtime, str)
        or re.fullmatch(r"\d+\.\d+(?:\.\d+)?", cuda_runtime) is None
    ):
        raise BackendError("the julia CUDA runtime must be a release such as '12.8'")
    return _Request(ordered, cuda_runtime, cuda_local, cpu_target)


def _check_manifest_bridge(bundle: Path, devices: tuple[str, ...], bridge: str) -> None:
    """Require every selected application manifest to pin the paired bridge.

    Parameters
    ----------
    bundle : pathlib.Path
        Bundled environment templates.
    devices : tuple of str
        Selected application environments.
    bridge : str
        Exactly pinned installed JuliaCall version.
    """
    for device in devices:
        path = bundle / "environments" / device / "Manifest.toml"
        with path.open("rb") as stream:
            manifest = mapping(tomllib.load(stream), "manifest")
        deps = mapping(manifest.get("deps"), "manifest dependencies")
        entries = deps.get("PythonCall")
        if not isinstance(entries, list) or len(entries) != 1:
            raise BackendError(
                f"the julia {device} manifest must pin exactly one PythonCall"
            )
        version = text_value(mapping(entries[0], "PythonCall"), "version")
        if version != bridge:
            raise BackendError(
                f"the julia {device} manifest pins PythonCall {version}, "
                f"but JuliaCall is {bridge}"
            )


@contextmanager
def _locked(depot: Path) -> Iterator[None]:
    """Hold a process-safe depot lock through verification and atomic publication.

    Parameters
    ----------
    depot : pathlib.Path
        Existing private depot directory.

    Yields
    ------
    None
        Control while this process owns the depot lock.
    """
    # The kernel owns this lock, so an interrupted or killed installer releases it.
    with (depot / "setup.lock").open("a+b") as stream:
        _file_lock.lock(stream.fileno())
        try:
            yield
        finally:
            _file_lock.unlock(stream.fileno())


def _probe(executable: Path) -> _Runtime:
    """Resolve a launcher to the real executable, version and native CPU target.

    Parameters
    ----------
    executable : pathlib.Path
        Installed runtime or launcher to execute during explicit setup.

    Returns
    -------
    _Runtime
        Actual executable, compatible version, binary digest and native CPU name.
    """
    result = subprocess.run(
        [
            str(executable),
            "--startup-file=no",
            "--history-file=no",
            "-e",
            'print(VERSION, "\\n", joinpath(Sys.BINDIR, Base.julia_exename()), '
            '"\\n", Sys.CPU_NAME)',
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    fields = result.stdout.strip().splitlines()
    if result.returncode or len(fields) != 3 or not _platforms.same_minor(fields[0]):
        raise BackendError(
            "the julia backend needs Julia "
            f"{_platforms.JULIA_VERSION.rsplit('.', 1)[0]}.x; "
            f"{executable} reported {result.stdout.strip() or result.stderr.strip()!r}"
        )
    actual = Path(fields[1]).resolve()
    return _Runtime(actual, fields[0], _source.file_digest(actual), fields[2])


def _runtime(executable: Path | None, depot: Path, host: str) -> _Runtime:
    """Choose an installed matching runtime or obtain the checked official archive.

    Parameters
    ----------
    executable : pathlib.Path or None
        Explicit executable, or ``None`` to try PATH and the pinned archive.
    depot : pathlib.Path
        Private depot owning any downloaded runtime.
    host : str
        Supported release-table platform key.

    Returns
    -------
    _Runtime
        Verified matching installed runtime.
    """
    if executable is not None:
        return _probe(executable.expanduser().resolve())
    if located := shutil.which("julia"):
        try:
            return _probe(Path(located))
        except BackendError, OSError, subprocess.TimeoutExpired:
            # An unrelated installed Julia is left alone; setup obtains its own pin.
            pass
    directory = depot / "runtimes" / f"{_platforms.JULIA_VERSION}-{host}"
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".runtime-", dir=directory.parent
        ) as staging:
            temporary = Path(staging)
            release = _platforms.RELEASES[host]
            archive = temporary / Path(release.url).name
            _download(release, archive)
            extracted = temporary / "extracted"
            extracted.mkdir()
            _extract(archive, extracted)
            _runtime_executable(extracted)
            extracted.rename(directory)
    return _probe(_runtime_executable(directory))


def _download(release: _platforms.Release, archive: Path) -> None:
    """Download an official archive and verify its pinned checksum before use.

    Parameters
    ----------
    release : _platforms.Release
        Official archive URL and expected SHA-256.
    archive : pathlib.Path
        Temporary file receiving the archive.
    """
    with (
        urllib.request.urlopen(release.url, timeout=60) as response,
        archive.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream)
    if _source.file_digest(archive) != release.sha256:
        raise BackendError(
            "the julia runtime archive SHA-256 does not match the official release"
        )


def _extract(archive: Path, destination: Path) -> None:
    """Extract the verified archive while refusing members that escape its root.

    Parameters
    ----------
    archive : pathlib.Path
        SHA-256-verified tar or ZIP archive.
    destination : pathlib.Path
        Empty staging directory receiving the runtime.
    """
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                child(destination, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise BackendError(
                        "the julia runtime ZIP contains an unexpected symbolic link"
                    )
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")


def _runtime_executable(directory: Path) -> Path:
    """Find the single platform executable in an unpacked official runtime.

    Parameters
    ----------
    directory : pathlib.Path
        Extracted runtime root.

    Returns
    -------
    pathlib.Path
        The sole platform executable under a ``bin`` directory.
    """
    names = ("julia.exe",) if sys.platform == "win32" else ("julia",)
    found = [
        path
        for name in names
        for path in directory.rglob(name)
        if path.is_file() and path.parent.name == "bin"
    ]
    if len(found) != 1:
        raise BackendError(
            "the julia runtime archive does not contain one unambiguous executable"
        )
    return found[0]


def _planned_record(
    bundle: Path, depot: Path, runtime: _Runtime, request: _Request, host: str
) -> SetupRecord:
    """Derive a deterministic publication identity from all installation inputs.

    Parameters
    ----------
    bundle : pathlib.Path
        Current source and application templates.
    depot : pathlib.Path
        Private depot receiving the installation.
    runtime : _Runtime
        Verified executable and version.
    request : _Request
        Canonical environment and preference choices.
    host : str
        Supported host key.

    Returns
    -------
    SetupRecord
        Planned immutable identity and final paths, without completed inventories.
    """
    digest = _source.engine_digest(bundle)
    record = SetupRecord(
        setup_id="pending",
        engine_digest=digest,
        bundle_digest=_source.digest_files(
            bundle, _source.setup_files(bundle, request.devices)
        ),
        julia_executable=runtime.executable,
        julia_version=runtime.version,
        julia_binary_digest=runtime.binary_digest,
        bridge_version=_platforms.BRIDGE_VERSION,
        depot=depot,
        engine=Path("engine") / digest,
        projects=tuple(
            (name, Path("engine") / digest / "environments" / name)
            for name in request.devices
        ),
        manifests=tuple(
            (
                name,
                _source.file_digest(bundle / "environments" / name / "Manifest.toml"),
            )
            for name in request.devices
        ),
        inventories=(),
        platform=host,
        cpu_target=request.cpu_target,
        cpu_name=runtime.cpu_name,
        cuda_runtime=request.cuda_runtime,
        cuda_local=request.cuda_local,
        elapsed_seconds=0.0,
    )
    setup_id = identity(record_inputs(record))
    root = depot / "setups" / setup_id
    return replace(
        record,
        setup_id=setup_id,
        engine=root / record.engine,
        projects=tuple((name, root / path) for name, path in record.projects),
    )


def _install(
    record: SetupRecord,
    bundle: Path,
    started: float,
    *,
    repair: SetupRecord | None = None,
) -> SetupRecord:
    """Build and verify a retained engine and environment stack before publication.

    Parameters
    ----------
    record : SetupRecord
        Planned installation identity and final paths.
    bundle : pathlib.Path
        Current source and application templates.
    started : float
        Setup start on the monotonic clock, seconds.
    repair : SetupRecord or None, optional
        Existing immutable publication whose wholly missing shared trees may be
        restored. Existing published files are never rewritten.

    Returns
    -------
    SetupRecord
        Atomically published installation with complete verified inventories.
    """
    published = record.depot / "setups" / record.setup_id
    published.parent.mkdir(exist_ok=True)
    staging_root = record.depot / "staging"
    staging_root.mkdir(exist_ok=True)
    devices = tuple(name for name, _ in record.projects)
    with tempfile.TemporaryDirectory(prefix="setup-", dir=staging_root) as staging:
        temporary = Path(staging)
        engine = temporary / "engine" / record.engine_digest
        for source in (
            *_source.engine_files(bundle),
            *_source.setup_files(bundle, devices),
        ):
            destination = engine / source.relative_to(bundle)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if (
            _source.engine_digest(engine) != record.engine_digest
            or _source.digest_files(engine, _source.setup_files(engine, devices))
            != record.bundle_digest
        ):
            raise BackendError(
                "the julia sources changed while setup was copying them; retry setup"
            )
        projects = tuple((name, engine / "environments" / name) for name in devices)
        _write_preferences(record, dict(projects))
        _check_loader_configuration(engine, projects)
        inventories: list[tuple[str, Path, str]] = []
        for name, project in projects:
            inventory = temporary / f"inventory-{name}.toml"
            _instantiate(record, project, projects, inventory)
            check_inventory(inventory, project, record.depot)
            inventories.append(
                (name, published / inventory.name, _source.file_digest(inventory))
            )
        complete = replace(
            record,
            inventories=tuple(inventories),
            elapsed_seconds=time.monotonic() - started,
        )
        if repair is not None:
            if complete.inventories != repair.inventories:
                raise BackendError(
                    "the julia repaired inventory differs from its immutable record"
                )
            _check_record(
                repair, bundle, record.depot, record.bridge_version, record.platform
            )
            return repair
        write_record(complete, temporary)
        # Only this rename makes the completed stack visible to readers.
        temporary.rename(published)
    return complete


def _preferences(record: SetupRecord) -> str:
    """Render the exact preferences shared by CUDA's runtime and compiler packages.

    Parameters
    ----------
    record : SetupRecord
        Canonical toolkit choices in the installation identity.

    Returns
    -------
    str
        Deterministic LocalPreferences TOML text.
    """
    settings = []
    if record.cuda_runtime is not None:
        settings.append(f'version = "{record.cuda_runtime}"')
    if record.cuda_local:
        settings.append("local = true")
    return "\n".join(
        f"[{name}]\n" + "\n".join(settings) + "\n"
        for name in ("CUDA_Runtime_jll", "CUDA_Compiler_jll")
    )


def _write_preferences(record: SetupRecord, projects: dict[str, Path]) -> None:
    """Write preferences only inside this unpublished installation.

    Parameters
    ----------
    record : SetupRecord
        Canonical toolkit choices.
    projects : dict
        Device names mapped to unpublished project directories.
    """
    if "cuda" in projects:
        (projects["cuda"] / "LocalPreferences.toml").write_text(
            _preferences(record), encoding="utf-8"
        )


def _instantiate(
    record: SetupRecord,
    project: Path,
    projects: tuple[tuple[str, Path], ...],
    inventory: Path,
) -> None:
    """Run the explicit installer with one depot and the complete selected stack.

    Parameters
    ----------
    record : SetupRecord
        Runtime, depot and compilation target for this installation.
    project : pathlib.Path
        Active environment to instantiate.
    projects : tuple of tuple
        Complete installed environment stack in canonical order.
    inventory : pathlib.Path
        TOML destination written only after successful verification.
    """
    environment = os.environ | {
        "JULIA_DEPOT_PATH": str(record.depot),
        "JULIA_LOAD_PATH": os.pathsep.join(
            [str(path) for _, path in projects] + ["@stdlib"]
        ),
        "JULIA_CPU_TARGET": record.cpu_target,
        "JULIA_CONDAPKG_BACKEND": "Null",
        "JULIA_PYTHONCALL_EXE": sys.executable,
        "JULIA_PKG_PRECOMPILE_AUTO": "0",
        "JULIA_PKG_OFFLINE": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        str(record.julia_executable),
        "--startup-file=no",
        "--history-file=no",
        f"--project={project}",
        str(project.parents[1] / "scripts" / "instantiate.jl"),
        str(inventory),
    ]
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode:
        raise BackendError(
            f"the julia {project.name} environment installer failed "
            f"with exit status {result.returncode}"
        )


def _status(directory: Path, depot: Path, size: int) -> SetupStatus:
    """Translate filesystem validation into an inspectable status or refusal.

    Parameters
    ----------
    directory : pathlib.Path
        Published installation directory.
    depot : pathlib.Path
        Current private depot.
    size : int
        Total depot file bytes.

    Returns
    -------
    SetupStatus
        Parsed record and compatibility, including the reason for a refusal.
    """
    record = None
    reason = None
    prerequisites: tuple[tuple[str, str | None], ...] = ()
    package_versions: tuple[tuple[str, str], ...] = ()
    try:
        record = read_record(directory)
        host = _platforms.host_platform()
        _check_record(record, _source.source_bundle(), depot, _bridge_version(), host)
        prerequisites = tuple(
            (name, _platforms.device_refusal(name, host)) for name, _ in record.projects
        )
        package_versions = _package_versions(record)
    except (BackendError, OSError, ValueError) as error:
        reason = str(error)
    return SetupStatus(
        directory.name,
        record,
        reason is None,
        reason,
        record is not None and record.julia_version == _platforms.JULIA_VERSION,
        size,
        prerequisites,
        package_versions,
    )


def _package_versions(record: SetupRecord) -> tuple[tuple[str, str], ...]:
    """Read the engine and vendor versions from the retained source manifests.

    Parameters
    ----------
    record : SetupRecord
        Published and validated installation.

    Returns
    -------
    tuple of tuple
        Engine and selected vendor package names paired with their recorded versions.
    """
    with (record.engine / "Project.toml").open("rb") as stream:
        engine = mapping(tomllib.load(stream), "engine project")
    versions = [
        (text_value(engine, "name"), str(engine.get("version", "unversioned source")))
    ]
    vendors = {"cuda": "CUDA", "rocm": "AMDGPU", "metal": "Metal"}
    for device, project in record.projects:
        if device in vendors:
            name = vendors[device]
            with (project / "Manifest.toml").open("rb") as stream:
                manifest = mapping(tomllib.load(stream), "manifest")
            entries = mapping(manifest.get("deps"), "manifest dependencies").get(name)
            if not isinstance(entries, list) or len(entries) != 1:
                raise BackendError(f"the julia {device} manifest must pin one {name}")
            versions.append((name, text_value(mapping(entries[0], name), "version")))
    return tuple(versions)


def _check_record(
    record: SetupRecord, bundle: Path, depot: Path, bridge: str, host: str
) -> None:
    """Verify all immutable installed inputs without any runtime or network call.

    Parameters
    ----------
    record : SetupRecord
        Parsed installation record.
    bundle : pathlib.Path
        Current engine sources and environment templates.
    depot : pathlib.Path
        Current private depot.
    bridge : str
        Exactly pinned installed JuliaCall version.
    host : str
        Current supported host key.
    """
    devices = tuple(name for name, _ in record.projects)
    if (
        not devices
        or devices != tuple(name for name in _platforms.DEVICES if name in devices)
        or devices[0] != "cpu"
    ):
        raise BackendError("the julia setup has an invalid installed-device stack")
    if record.platform != host or record.depot != depot:
        raise BackendError("the julia setup belongs to another platform or depot")
    if record.bridge_version != bridge:
        raise BackendError(
            f"the julia setup needs JuliaCall {record.bridge_version}, found {bridge}"
        )
    if not _platforms.same_minor(record.julia_version):
        raise BackendError(
            "the julia setup uses another Julia minor; run solprop julia setup"
        )
    if _source.file_digest(record.julia_executable) != record.julia_binary_digest:
        raise BackendError(
            "the julia setup executable changed; run solprop julia setup"
        )
    if record.engine_digest != _source.engine_digest(
        bundle
    ) or record.engine_digest != _source.engine_digest(record.engine):
        raise BackendError("the julia engine sources changed; run solprop julia setup")
    if record.bundle_digest != _source.digest_files(
        bundle, _source.setup_files(bundle, devices)
    ) or record.bundle_digest != _source.digest_files(
        record.engine, _source.setup_files(record.engine, devices)
    ):
        raise BackendError(
            "the julia environment templates changed; run solprop julia setup"
        )
    if tuple(name for name, _, _ in record.inventories) != devices:
        raise BackendError(
            "the julia setup lacks an inventory for an installed environment"
        )
    refuse_overrides(depot)
    _check_loader_configuration(record.engine, record.projects)
    for name, project in record.projects:
        _check_preferences(record, name, project)
    for name, path, digest in record.inventories:
        if _source.file_digest(path) != digest:
            raise BackendError(
                f"the julia {name} artifact inventory changed after setup"
            )
    for name, path, _ in record.inventories:
        check_inventory(path, dict(record.projects)[name], depot)


def _check_preferences(record: SetupRecord, name: str, project: Path) -> None:
    """Refuse changed or inherited preferences across the complete installed stack.

    Parameters
    ----------
    record : SetupRecord
        Recorded toolkit choices and source root.
    name : str
        Selected device name.
    project : pathlib.Path
        Retained application environment.
    """
    for parent in (record.engine, record.engine / "environments"):
        if _present(parent / "LocalPreferences.toml"):
            raise BackendError("the julia setup has unrecorded inherited preferences")
    path = project / "LocalPreferences.toml"
    if name == "cuda":
        if path.read_text(encoding="utf-8") != _preferences(record):
            raise BackendError("the julia CUDA preferences changed after setup")
    elif _present(path):
        raise BackendError(f"the julia {name} environment has unrecorded preferences")


def _present(path: Path) -> bool:
    """Check for a loader input, including a dangling symbolic link.

    Parameters
    ----------
    path : pathlib.Path
        Potential unrecorded loader input.

    Returns
    -------
    bool
        Whether a file-system entry exists at this path.
    """
    return path.exists() or path.is_symlink()


def _check_loader_configuration(
    engine: Path, projects: tuple[tuple[str, Path], ...]
) -> None:
    """Refuse Julia loader aliases and ancestor workspace redirections.

    Parameters
    ----------
    engine : pathlib.Path
        Retained engine source root.
    projects : tuple of tuple
        Complete selected application environment stack.
    """
    # Base.loading gives these aliases precedence over the explicitly owned files.
    roots = (engine, engine / "environments", *(path for _, path in projects))
    for directory in roots:
        aliases = [
            directory / name
            for name in (
                "JuliaProject.toml",
                "JuliaManifest.toml",
                "JuliaLocalPreferences.toml",
            )
        ]
        aliases.extend(directory.glob("*Manifest-v*.toml"))
        if any(_present(alias) for alias in aliases):
            raise BackendError(
                f"the julia setup has an unrecorded loader alias in {directory}"
            )
    for _, project in projects:
        _check_workspace_parents(project)


def _check_workspace_parents(project: Path) -> None:
    """Refuse an ancestor workspace that replaces an owned project's manifest.

    Parameters
    ----------
    project : pathlib.Path
        Owned application environment whose effective base must remain itself.
    """
    # Match Base.base_project's home boundary and explicit workspace project list.
    home = Path.home().resolve()
    started_in_home = str(project).startswith(str(home))
    for parent in project.parents:
        if started_in_home and not str(parent).startswith(str(home)):
            break
        source = next(
            (
                parent / name
                for name in ("JuliaProject.toml", "Project.toml")
                if (parent / name).is_file()
            ),
            None,
        )
        if source is None:
            continue
        with source.open("rb") as stream:
            content = mapping(tomllib.load(stream), "ancestor project")
        workspace = content.get("workspace")
        if workspace is None:
            continue
        members = mapping(workspace, "ancestor workspace").get("projects")
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, str):
                    raise BackendError(
                        "the julia ancestor workspace has invalid projects"
                    )
                if (parent / member).resolve() == project.resolve():
                    raise BackendError(
                        f"the julia setup is redirected by ancestor workspace {source}"
                    )


def _directory_bytes(directory: Path) -> int:
    """Count allocated file payloads without following directory symlinks.

    Parameters
    ----------
    directory : pathlib.Path
        Depot root whose files are counted.

    Returns
    -------
    int
        Sum of file payload sizes, bytes.
    """
    total = 0
    for path in directory.rglob("*"):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            # Installers may rename or remove their private staging files mid-scan.
            continue
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total
