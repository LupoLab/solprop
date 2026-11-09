# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend keeps the FFTW plans its runs measure, by default (ADR-076)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest
from conftest import PhaseACase

import solprop_native
from solprop_native import NativeBackend, _core, _wisdom

_KEY = "machine-and-build"


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget what the wisdom module read and measured; give it a fixed machine key."""
    monkeypatch.setattr(_wisdom, "_read", {})
    monkeypatch.setattr(_wisdom, "_measured", {})
    monkeypatch.setattr(_wisdom, "machine_key", lambda: _KEY)
    monkeypatch.setattr(_wisdom, "_keeps_wisdom", lambda: True)


def _plans(path: Path) -> dict[str, str]:
    """Return a plan file's plans."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == _wisdom.FORMAT
    return data["plans"]


@pytest.mark.usefixtures("fresh")
def test_the_default_file_is_the_machines_in_the_store(tmp_path: Path) -> None:
    environ = {"SOLPROP_CACHE_DIR": str(tmp_path), "SOLPROP_CALIBRATION": "auto"}
    path = _wisdom.wisdom_file(environ)
    assert path == tmp_path / "machines" / _KEY / _wisdom.FILE
    assert _wisdom.wisdom_file(environ | {"SOLPROP_CALIBRATION": "use"}) == path


@pytest.mark.usefixtures("fresh")
def test_off_keeps_no_default_file_but_a_named_one(tmp_path: Path) -> None:
    environ = {"SOLPROP_CACHE_DIR": str(tmp_path), "SOLPROP_CALIBRATION": "off"}
    assert _wisdom.wisdom_file(environ) is None
    named = environ | {_wisdom.ENVIRONMENT: str(tmp_path / "wisdom")}
    assert _wisdom.wisdom_file(named) == tmp_path / f"wisdom.native-{_KEY}"


@pytest.mark.usefixtures("fresh")
def test_a_named_file_is_keyed_beside_it(tmp_path: Path) -> None:
    # pathlib expands ``~`` from the process's HOME.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(tmp_path))
        found = _wisdom.wisdom_file({_wisdom.ENVIRONMENT: "~/other"})
    assert found == tmp_path / f"other.native-{_KEY}"


@pytest.mark.usefixtures("fresh")
@pytest.mark.parametrize("named", [".", "..", "~nouser-solprop-test/w"])
def test_an_unusable_name_warns_and_keeps_nothing(
    named: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_wisdom.__name__):
        assert _wisdom.wisdom_file({_wisdom.ENVIRONMENT: named}) is None
    assert "cannot keep FFTW plans" in caplog.text


@pytest.mark.usefixtures("fresh")
def test_a_store_without_a_home_warns_and_keeps_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def homeless() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(homeless))
    with caplog.at_level(logging.WARNING, logger=_wisdom.__name__):
        assert _wisdom.wisdom_file({"SOLPROP_CALIBRATION": "auto"}) is None
    assert "Could not determine home directory" in caplog.text


def test_a_build_without_fftw_wisdom_keeps_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Device:
        @staticmethod
        def export_fftw_wisdom() -> None:
            return None

    monkeypatch.setattr(_wisdom, "_core", _Device())
    _wisdom._keeps_wisdom.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        assert _wisdom.wisdom_file({"SOLPROP_CACHE_DIR": str(tmp_path)}) is None
        assert _wisdom.digest({}) is None
    finally:
        _wisdom._keeps_wisdom.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_the_machine_key_is_the_profiles_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wisdom.machine_key.cache_clear()
    try:
        key = _wisdom.machine_key()
        assert len(key) == 24
        found = solprop_native.PROFILE.fingerprint
        if found is not None:
            assert key == found.digest
        # Under SOLPROP_CALIBRATION=off the package took none, and it is computed.
        monkeypatch.setattr(
            solprop_native, "PROFILE", solprop_native.PROFILE.__class__(mode="off")
        )
        _wisdom.machine_key.cache_clear()
        assert _wisdom.machine_key() == key
    finally:
        _wisdom.machine_key.cache_clear()


@pytest.mark.usefixtures("fresh")
def test_plans_round_trip_and_an_unchanged_run_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "wisdom.json"
    assert _wisdom.plans(path) == ({}, False)
    assert _wisdom.store(path, {"r2c:a": "(a)"})
    assert _plans(path) == {"r2c:a": "(a)"}
    version = path.stat().st_mtime_ns
    # Nothing new: no write.
    assert not _wisdom.store(path, {"r2c:a": "(a)"})
    assert not _wisdom.store(path, {})
    assert path.stat().st_mtime_ns == version
    assert _wisdom.plans(path) == ({"r2c:a": "(a)"}, True)
    # This process's own plans are offered too, over the file's, and alone without one.
    _wisdom.remember({"c2r:b": "(b)"})
    assert _wisdom.plans(path) == ({"r2c:a": "(a)", "c2r:b": "(b)"}, True)
    assert _wisdom.plans(None) == ({"c2r:b": "(b)"}, False)
    assert not _wisdom.store(None, {"c2r:b": "(b)"})


@pytest.mark.usefixtures("fresh")
def test_a_writer_merges_what_others_stored(tmp_path: Path) -> None:
    path = tmp_path / "wisdom.json"
    path.write_text(
        json.dumps({"format": 1, "plans": {"r2c:a": "(a)"}}), encoding="utf-8"
    )
    assert _wisdom.store(path, {"c2r:b": "(b)"})
    assert _plans(path) == {"r2c:a": "(a)", "c2r:b": "(b)"}
    # A plan measured again, since its wisdom no longer sufficed, replaces it.
    assert _wisdom.store(path, {"r2c:a": "(a2)"})
    assert _plans(path)["r2c:a"] == "(a2)"
    assert not list(tmp_path.glob(".*"))


@pytest.mark.usefixtures("fresh")
def test_a_profiles_keys_are_never_stored(tmp_path: Path) -> None:
    path = tmp_path / "wisdom.json"
    assert not _wisdom.store(path, {"r2c:a": "(a)"}, profiled={"r2c:a"})
    assert not path.exists()
    assert _wisdom.store(path, {"r2c:a": "(a)", "c2r:b": "(b)"}, profiled={"r2c:a"})
    assert _plans(path) == {"c2r:b": "(b)"}


@pytest.mark.usefixtures("fresh")
def test_a_writer_waits_for_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "wisdom.json"
    if _wisdom.fcntl is None:  # pragma: no cover - Windows: there is no writer lock
        pytest.skip("no writer lock on this platform")
    writer = threading.Thread(target=_wisdom.store, args=(path, {"r2c:a": "(a)"}))
    with _wisdom._exclusive(path):  # pyright: ignore[reportPrivateUsage]
        writer.start()
        writer.join(0.2)
        assert writer.is_alive()
        assert not path.exists()
    writer.join()
    assert _plans(path) == {"r2c:a": "(a)"}


@pytest.mark.usefixtures("fresh")
@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        "[1, 2]",
        json.dumps({"format": 0, "plans": {}}),
        json.dumps({"format": 1, "plans": {"r2c:a": 3}}),
    ],
)
def test_a_malformed_file_is_reported_ignored_and_replaced(
    text: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "wisdom.json"
    path.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=_wisdom.__name__):
        assert _wisdom.plans(path) == ({}, False)
        assert _wisdom.store(path, {"r2c:a": "(a)"})
    assert "ignoring the FFTW plans" in caplog.text
    assert _plans(path) == {"r2c:a": "(a)"}


@pytest.mark.usefixtures("fresh")
def test_an_unreadable_or_unwritable_file_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff\xfe")
    unwritable = tmp_path / "a file" / "wisdom.json"
    unwritable.parent.write_text("", encoding="ascii")
    with caplog.at_level(logging.WARNING, logger=_wisdom.__name__):
        assert _wisdom.plans(binary) == ({}, False)
        assert not _wisdom.store(unwritable, {"r2c:a": "(a)"})
    assert "cannot read FFTW plans" in caplog.text
    assert "cannot store FFTW plans" in caplog.text


@pytest.mark.usefixtures("fresh")
def test_a_failed_replace_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wisdom.json"

    def refuse(self: Path, target: Path) -> Path:
        raise PermissionError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", refuse)
    assert not _wisdom.store(path, {"r2c:a": "(a)"})
    assert not path.exists()
    assert sorted(one.name for one in tmp_path.iterdir()) == [f"{path.name}.lock"]


@pytest.mark.usefixtures("fresh")
def test_the_digest_follows_the_plans() -> None:
    before = _wisdom.digest({"r2c:a": "(a)"})
    assert before is not None
    assert len(before) == 16
    assert _wisdom.digest({"r2c:a": "(a)", "c2r:b": "(b)"}) != before


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Measure every transform, keep plans under `tmp_path`; return the file."""
    if _core.export_fftw_wisdom() is None:
        pytest.skip("this build keeps no FFTW wisdom")
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "measure")
    monkeypatch.setenv(_wisdom.ENVIRONMENT, str(tmp_path / "wisdom"))
    monkeypatch.setattr(_wisdom, "_read", {})
    monkeypatch.setattr(_wisdom, "_measured", {})
    _core.forget_fftw_wisdom()
    path = _wisdom.wisdom_file()
    assert path is not None
    return path


def _planning(backend: NativeBackend) -> dict[str, object]:
    return backend._native_state.fft_planning  # pyright: ignore[reportPrivateUsage]


def test_a_second_run_of_a_shape_plans_without_measuring(
    phase_a_case: PhaseACase, measured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, grid = phase_a_case
    first = NativeBackend(problem, grid=grid)
    try:
        metadata = first.metadata()
        found = _planning(first)
        assert metadata["wisdom_loaded"] is False
        assert isinstance(metadata["wisdom_digest"], str)
        assert found["measured"]
        assert found["reused"] == []
        kept = _plans(measured)
        assert kept == found["measured"]
    finally:
        first.close()
    # A new process: nothing of this one's plans in memory or in FFTW.
    monkeypatch.setattr(_wisdom, "_read", {})
    monkeypatch.setattr(_wisdom, "_measured", {})
    _core.forget_fftw_wisdom()
    second = NativeBackend(problem, grid=grid)
    try:
        metadata = second.metadata()
        found = _planning(second)
        assert metadata["wisdom_loaded"] is True
        assert metadata["wisdom_digest"] is not None
        assert found["measured"] == {}
        assert set(found["reused"]) == set(kept)
    finally:
        second.close()
    assert _plans(measured) == kept


def test_off_still_reuses_a_processs_own_plans(
    phase_a_case: PhaseACase, measured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, grid = phase_a_case
    monkeypatch.delenv(_wisdom.ENVIRONMENT)
    assert _wisdom.wisdom_file() is None  # the tests run under off
    first = NativeBackend(problem, grid=grid)
    try:
        assert _planning(first)["measured"]
    finally:
        first.close()
    second = NativeBackend(problem, grid=grid)
    try:
        assert _planning(second)["measured"] == {}
        assert _planning(second)["reused"]
        assert second.metadata()["wisdom_loaded"] is False
    finally:
        second.close()
    assert not measured.exists()


def test_a_profiles_winner_takes_precedence_over_stored_plans(
    phase_a_case: PhaseACase, measured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, grid = phase_a_case
    first = NativeBackend(problem, grid=grid)
    try:
        kept = dict(_planning(first)["measured"])  # pyright: ignore[reportArgumentType]
    finally:
        first.close()
    winner, *others = sorted(kept)
    # A profile holding one of the transforms, at measure for the rest: its own wisdom
    # plans it, the stored plans the rest, and its key is never stored.
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "")
    monkeypatch.setattr(
        solprop_native,
        "PLANNING",
        {
            "mode": "profile",
            "efforts": {"rows": "measure", "threaded_rows": "measure"},
            "wisdom": {winner: kept[winner]},
        },
    )
    measured.write_text(
        json.dumps({"format": 1, "plans": {key: kept[key] for key in others}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(_wisdom, "_read", {})
    monkeypatch.setattr(_wisdom, "_measured", {})
    second = NativeBackend(problem, grid=grid)
    try:
        found = _planning(second)
        assert found["missing"] == []
        assert winner in found["groups"]  # pyright: ignore[reportOperatorIssue]
        assert winner not in found["reused"]  # pyright: ignore[reportOperatorIssue]
        assert found["measured"] == {}
        assert set(found["reused"]) == set(others)  # pyright: ignore[reportArgumentType]
    finally:
        second.close()
    assert sorted(_plans(measured)) == others


def test_an_estimate_run_takes_the_stored_plans_without_seeding(
    phase_a_case: PhaseACase, measured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transform left at estimate imports its stored plan for its estimate plan."""
    problem, grid = phase_a_case
    first = NativeBackend(problem, grid=grid)
    try:
        kept = sorted(_planning(first)["measured"])  # pyright: ignore[reportArgumentType]
    finally:
        first.close()
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    monkeypatch.setattr(_wisdom, "_read", {})
    monkeypatch.setattr(_wisdom, "_measured", {})
    _core.forget_fftw_wisdom()
    second = NativeBackend(problem, grid=grid)
    try:
        found = _planning(second)
        assert found["measured"] == {}
        assert set(found["reused"]) == set(kept)  # pyright: ignore[reportArgumentType]
    finally:
        second.close()
