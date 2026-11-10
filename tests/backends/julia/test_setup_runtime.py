# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from solprop.backends.julia import _platforms, _setup, _source
from solprop.exceptions import BackendError


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("linux", "x86_64", "linux-x86_64"),
        ("linux", "aarch64", "linux-aarch64"),
        ("darwin", "arm64", "darwin-aarch64"),
        ("win32", "AMD64", "win32-x86_64"),
    ],
)
def test_supported_platforms_have_checksums(monkeypatch, system, machine, expected):
    monkeypatch.setattr(_platforms.sys, "platform", system)
    monkeypatch.setattr(_platforms.platform, "machine", lambda: machine)
    assert _platforms.host_platform() == expected
    release = _platforms.RELEASES[expected]
    assert release.url.startswith("https://julialang-s3.julialang.org/")
    assert len(release.sha256) == 64


def test_windows_arm_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(_platforms.sys, "platform", "win32")
    monkeypatch.setattr(_platforms.platform, "machine", lambda: "ARM64")
    with pytest.raises(BackendError, match="does not support win32-aarch64"):
        _platforms.host_platform()


@pytest.mark.parametrize(
    ("version", "expected"),
    [("1.13.3", True), ("1.13.4", True), ("1.12.6", False), ("unknown", False)],
)
def test_runtime_requires_selected_minor(version, expected):
    assert _platforms.same_minor(version) is expected


@pytest.mark.parametrize(
    ("device", "host", "reason"),
    [
        ("vulkan", "linux-x86_64", "no device"),
        ("metal", "linux-x86_64", "Apple silicon"),
        ("cuda", "win32-x86_64", "Linux only"),
        ("rocm", "darwin-aarch64", "Linux only"),
        ("metal", "darwin-aarch64", None),
        ("cpu", "linux-aarch64", None),
        ("cuda", "linux-aarch64", "Jetson and SBSA artifact selection"),
        ("cuda", "linux-x86_64", None),
    ],
)
def test_device_platform_prerequisites(device, host, reason):
    actual = _platforms.device_refusal(device, host)
    if reason is None:
        assert actual is None
    else:
        assert reason in actual


def test_rocm_requires_all_system_libraries_without_loading_them(tmp_path, monkeypatch):
    installation = tmp_path / "rocm"
    library = installation / "lib"
    library.mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(installation))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "other"))
    original = Path.glob

    def isolated(path, pattern):
        return original(path, pattern) if path.is_relative_to(tmp_path) else iter(())

    monkeypatch.setattr(Path, "glob", isolated)
    refusal = _platforms.device_refusal("rocm", "linux-x86_64")
    assert refusal is not None
    assert "rocblas" in refusal
    assert "rocfft" in refusal
    for name in ("amdhip64", "hsa-runtime64", "rocblas", "rocfft"):
        (library / f"lib{name}.so.1").write_bytes(b"fixture library")
    assert _platforms.device_refusal("rocm", "linux-x86_64") is None


def _tar(path):
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("julia/bin/julia")
        content = b"fixture runtime"
        entry.mode = 0o755
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))


@pytest.mark.parametrize("located", [None, "unrelated-julia"])
def test_runtime_download_is_verified_and_published_once(
    tmp_path, monkeypatch, located
):
    monkeypatch.setattr(_setup.shutil, "which", lambda name: located)
    monkeypatch.setattr(_setup.sys, "platform", "linux")
    downloads = []

    def download(release, archive):
        downloads.append(archive)
        _tar(archive)

    def probe(path):
        if str(path) == "unrelated-julia":
            raise BackendError("incompatible Julia")
        return _setup._Runtime(
            path, _platforms.JULIA_VERSION, _source.file_digest(path), "fixture"
        )

    monkeypatch.setattr(_setup, "_download", download)
    monkeypatch.setattr(_setup, "_probe", probe)
    first = _setup._runtime(None, tmp_path, "linux-x86_64")
    assert first.executable.read_bytes() == b"fixture runtime"
    assert "/runtimes/" in first.executable.as_posix()
    assert _setup._runtime(None, tmp_path, "linux-x86_64") == first
    assert len(downloads) == 1
    assert not list((tmp_path / "runtimes").glob(".runtime-*"))


def test_explicit_or_matching_path_runtime_is_never_downloaded(tmp_path, monkeypatch):
    runtime = _setup._Runtime(
        tmp_path / "actual", _platforms.JULIA_VERSION, "0" * 64, "fixture"
    )
    paths = []

    def probe(path):
        paths.append(path)
        return runtime

    monkeypatch.setattr(_setup, "_probe", probe)
    monkeypatch.setattr(_setup.shutil, "which", lambda name: str(tmp_path / "launcher"))
    assert _setup._runtime(tmp_path / "explicit", tmp_path, "linux-x86_64") is runtime
    assert _setup._runtime(None, tmp_path, "linux-x86_64") is runtime
    assert paths == [tmp_path / "explicit", tmp_path / "launcher"]


def test_failed_runtime_download_leaves_no_published_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(_setup.shutil, "which", lambda name: None)

    def failure(*args):
        raise BackendError("archive SHA-256 mismatch")

    monkeypatch.setattr(_setup, "_download", failure)
    with pytest.raises(BackendError, match="SHA-256"):
        _setup._runtime(None, tmp_path, "linux-x86_64")
    assert not list((tmp_path / "runtimes").iterdir())


def test_runtime_archive_requires_one_unambiguous_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(_setup.sys, "platform", "linux")
    with pytest.raises(BackendError, match="one unambiguous"):
        _setup._runtime_executable(tmp_path)
    for version in ("one", "two"):
        root = tmp_path / version / "bin"
        root.mkdir(parents=True)
        (root / "julia").write_bytes(b"fixture")
    with pytest.raises(BackendError, match="one unambiguous"):
        _setup._runtime_executable(tmp_path)


def test_windows_archive_and_executable_are_supported(tmp_path, monkeypatch):
    archive = tmp_path / "julia.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("julia/bin/julia.exe", "fixture")
    destination = tmp_path / "extract"
    _setup._extract(archive, destination)
    monkeypatch.setattr(_setup.sys, "platform", "win32")
    assert (
        _setup._runtime_executable(destination)
        == destination / "julia" / "bin" / "julia.exe"
    )


def test_zip_symbolic_links_are_rejected(tmp_path):
    archive = tmp_path / "julia.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        entry = zipfile.ZipInfo("link")
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(entry, "target")
    with pytest.raises(BackendError, match="symbolic link"):
        _setup._extract(archive, tmp_path / "extract")


def test_depot_default_uses_calibration_store(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLPROP_JULIA_DEPOT", raising=False)
    monkeypatch.setattr(_setup, "store_directory", lambda: tmp_path)
    assert _setup.depot_directory() == tmp_path / "julia"
