# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend keeps FFTW wisdom in a shared file only when one is named."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest
from conftest import PhaseACase

from solprop_native import NativeBackend, _core, _wisdom

_PREAMBLE = "(fftw-test fftw_wisdom #x1 #x2 #x3 #x4"


class _FakeFftw:
    """FFTW's wisdom as a set of entry lines under one preamble."""

    def __init__(self, *entries: str) -> None:
        self.entries: set[str] = set(entries)
        self.exports = 0

    def export_fftw_wisdom(self) -> str:
        self.exports += 1
        body = "".join(f"  ({entry})\n" for entry in sorted(self.entries))
        return f"{_PREAMBLE}\n{body})\n"

    def import_fftw_wisdom(self, text: str) -> bool:
        lines = text.splitlines()
        if not lines or lines[0] != _PREAMBLE or lines[-1] != ")":
            return False
        self.entries |= {line.strip()[1:-1] for line in lines[1:-1]}
        return True


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeFftw:
    """Give the wisdom module a fake FFTW and forget what it knew of every file."""
    fftw = _FakeFftw("plan a")
    monkeypatch.setattr(_wisdom, "_core", fftw)
    monkeypatch.setattr(_wisdom, "_imported", {})
    monkeypatch.setattr(_wisdom, "_known", {})
    return fftw


def test_nothing_is_kept_unless_a_file_is_named(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeFftw
) -> None:
    monkeypatch.delenv(_wisdom.ENVIRONMENT, raising=False)
    assert _wisdom.wisdom_file() is None
    assert _wisdom.wisdom_file("") is None
    assert not _wisdom.load(None)
    assert not _wisdom.store(None)
    assert fake.exports == 0


def test_the_native_file_is_keyed_beside_the_named_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_wisdom.ENVIRONMENT, str(tmp_path / "wisdom"))
    path = _wisdom.wisdom_file()
    if _core.export_fftw_wisdom() is None:
        assert path is None
        return
    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith("wisdom.native-")
    assert path == _wisdom.wisdom_file(str(tmp_path / "wisdom"))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _wisdom.wisdom_file("~/other") == tmp_path / f"other.{path.suffix[1:]}"


def test_a_build_without_fftw_wisdom_keeps_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Device:
        @staticmethod
        def export_fftw_wisdom() -> None:
            return None

    monkeypatch.setattr(_wisdom, "_core", _Device())
    _wisdom._key.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        assert _wisdom.wisdom_file(str(tmp_path / "wisdom")) is None
        assert _wisdom.digest() is None
    finally:
        _wisdom._key.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_the_processor_is_named_from_its_model_lines() -> None:
    processor = _wisdom._processor  # pyright: ignore[reportPrivateUsage]
    x86 = "processor\t: 0\nmodel name\t: Example CPU @ 2.60GHz\nflags\t: fpu\n"
    assert processor(x86) == "Example CPU @ 2.60GHz"
    arm = "CPU implementer\t: 0x41\nCPU part\t: 0xd0c\nCPU part\t: 0xd0c\n"
    assert processor(arm) == "0x41 0xd0c"
    assert isinstance(processor("nothing to see\n"), str)
    assert isinstance(processor(), str)


def test_wisdom_round_trips_and_an_unchanged_run_writes_nothing(
    fake: _FakeFftw, tmp_path: Path
) -> None:
    path = tmp_path / "nested" / "wisdom.native-key"
    assert not _wisdom.load(path)
    assert _wisdom.store(path)
    assert path.read_text(encoding="ascii") == fake.export_fftw_wisdom()
    version = path.stat().st_mtime_ns
    # Nothing planned since: no lock, no read, no write.
    assert not _wisdom.store(path)
    assert path.stat().st_mtime_ns == version
    # A process that knows the file already does not read it again.
    assert _wisdom.load(path)
    # A fresh process imports the file.
    _wisdom._imported.clear()  # pyright: ignore[reportPrivateUsage]
    _wisdom._known.clear()  # pyright: ignore[reportPrivateUsage]
    fake.entries = set()
    assert _wisdom.load(path)
    assert fake.entries == {"plan a"}
    assert not _wisdom.store(path)


def test_a_writer_merges_what_others_stored(fake: _FakeFftw, tmp_path: Path) -> None:
    path = tmp_path / "wisdom.native-key"
    other = _FakeFftw("plan b")
    path.write_text(other.export_fftw_wisdom(), encoding="ascii")
    assert _wisdom.store(path)
    assert fake.entries == {"plan a", "plan b"}
    merged = _FakeFftw()
    assert merged.import_fftw_wisdom(path.read_text(encoding="ascii"))
    assert merged.entries == {"plan a", "plan b"}
    # Our plans are in the file now: a merge adds nothing, and nothing is written.
    _wisdom._known.clear()  # pyright: ignore[reportPrivateUsage]
    version = path.stat().st_mtime_ns
    assert not _wisdom.store(path)
    assert path.stat().st_mtime_ns == version
    assert not list(tmp_path.glob(".*"))


def test_a_writer_waits_for_the_lock(fake: _FakeFftw, tmp_path: Path) -> None:
    path = tmp_path / "wisdom.native-key"
    if _wisdom.fcntl is None:  # pragma: no cover - Windows: there is no writer lock
        pytest.skip("no writer lock on this platform")
    writer = threading.Thread(target=_wisdom.store, args=(path,))
    with _wisdom._exclusive(path):  # pyright: ignore[reportPrivateUsage]
        writer.start()
        writer.join(0.2)
        assert writer.is_alive()
        assert not path.exists()
    writer.join()
    assert path.read_text(encoding="ascii") == fake.export_fftw_wisdom()


def test_foreign_or_unreadable_wisdom_is_reported_and_ignored(
    fake: _FakeFftw, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    foreign = tmp_path / "foreign"
    foreign.write_text("not wisdom at all", encoding="ascii")
    with caplog.at_level(logging.WARNING, logger=_wisdom.__name__):
        assert not _wisdom.load(foreign)
        # A writer replaces it with this process's wisdom.
        assert _wisdom.store(foreign)
    assert "is not this FFTW's wisdom" in caplog.text
    assert foreign.read_text(encoding="ascii") == fake.export_fftw_wisdom()
    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff\xfe")
    unwritable = tmp_path / "a file" / "wisdom"
    unwritable.parent.write_text("", encoding="ascii")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_wisdom.__name__):
        assert not _wisdom.load(binary)
        assert not _wisdom.store(unwritable)
    assert "cannot read FFTW wisdom" in caplog.text
    assert "cannot store FFTW wisdom" in caplog.text


def test_a_failed_replace_leaves_no_partial_file(
    fake: _FakeFftw, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wisdom.native-key"

    def refuse(self: Path, target: Path) -> Path:
        raise PermissionError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", refuse)
    assert not _wisdom.store(path)
    assert not path.exists()
    assert sorted(one.name for one in tmp_path.iterdir()) == [f"{path.name}.lock"]


def test_the_digest_follows_the_wisdom(fake: _FakeFftw) -> None:
    before = _wisdom.digest()
    assert before is not None
    assert len(before) == 16
    fake.entries.add("plan c")
    assert _wisdom.digest() != before


def test_a_backend_keeps_and_reuses_its_plans(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    problem, grid = phase_a_case
    monkeypatch.setenv(_wisdom.ENVIRONMENT, str(tmp_path / "wisdom"))
    path = _wisdom.wisdom_file()
    first = NativeBackend(problem, grid=grid)
    try:
        metadata = first.metadata()
        assert metadata["wisdom_loaded"] is False
        if path is None:
            assert metadata["wisdom_digest"] is None
            return
        assert isinstance(metadata["wisdom_digest"], str)
        assert path.is_file()
        assert _core.import_fftw_wisdom(path.read_text(encoding="ascii"))
    finally:
        first.close()
    second = NativeBackend(problem, grid=grid)
    try:
        assert second.metadata()["wisdom_loaded"] is True
    finally:
        second.close()
