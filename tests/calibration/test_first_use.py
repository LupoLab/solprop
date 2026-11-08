# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calibrating on first use, before a scan's workers, and never twice (SEP 46 §7.14)."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest

from solprop import calibration
from solprop.calibration import _native, controller, first_use, problems
from solprop.exceptions import CalibrationNotice


@pytest.fixture
def auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[tuple[str, tuple[int, ...]]]:
    """An empty store under SOLPROP_CALIBRATION=auto; calibrations are recorded."""
    monkeypatch.setenv("SOLPROP_CALIBRATION", "auto")
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("SOLPROP_CALIBRATION_ROLE", raising=False)
    monkeypatch.setattr(first_use, "_DONE", set())
    monkeypatch.setattr(first_use, "_device_build", lambda: False)
    asked: list[tuple[str, tuple[int, ...]]] = []

    def fake(
        level: str, budget: object = None, *, environ: object, cpus: object
    ) -> object:
        asked.append((level, tuple(cpus)))  # type: ignore[arg-type]
        return controller.CalibrationReport(
            level=level, status="published", fingerprint="f", allocation={}
        )

    monkeypatch.setattr(controller, "calibrate", fake)
    return asked


def _publish(level: str, cpus: tuple[int, ...]) -> None:
    """Publish a row of `level` for the class of `cpus` on this machine and build."""
    found = calibration.choice()
    assert found.fingerprint is not None
    machine = _native.topology.read_topology()
    allocation = _native.allocation.classify(
        machine,
        cpus,
        first_use._process.cgroup_cpus(),  # pyright: ignore[reportPrivateUsage]
    )
    store = _native.store
    store.Store(calibration.store_directory()).publish(
        store.Generation(
            fingerprint=found.fingerprint.digest,
            name=store.generation_name(),
            created="2026-11-08",
            rows=(store.Row(allocation=allocation.record(), level=level),),
        )
    )


def test_a_first_native_run_calibrates_quickly_once_with_a_notice(
    auto: list[tuple[str, tuple[int, ...]]],
) -> None:
    with pytest.warns(CalibrationNotice, match="calibrating the native backend"):
        report = first_use.ensure()
    assert report is not None
    assert [level for level, _ in auto] == ["quick"]
    # Not twice in one process.
    assert first_use.ensure() is None
    assert len(auto) == 1


def test_a_process_that_loaded_native_refreshes_its_profile(
    auto: list[tuple[str, tuple[int, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    refreshed: list[bool] = []
    monkeypatch.setattr(
        solprop_native, "refresh_profile", lambda: refreshed.append(True)
    )
    with pytest.warns(CalibrationNotice):
        first_use.ensure()
    assert refreshed == [True]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("SOLPROP_CALIBRATION", "use"),
        ("SOLPROP_CALIBRATION", "off"),
        ("SOLPROP_CALIBRATION_ROLE", "worker"),
    ],
)
def test_use_off_and_workers_never_measure(
    auto: list[tuple[str, tuple[int, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)
    assert first_use.ensure() is None
    assert first_use.for_workers([0]) is None
    assert auto == []


def test_a_device_build_is_left_to_its_own_calibration(
    auto: list[tuple[str, tuple[int, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(first_use, "_device_build", lambda: True)
    assert first_use.ensure() is None
    assert auto == []


def test_an_existing_row_is_used_and_a_scan_wants_a_standard_one(
    auto: list[tuple[str, tuple[int, ...]]],
) -> None:
    cpus = tuple(first_use._process.allowed_cpus())  # pyright: ignore[reportPrivateUsage]
    _publish("quick", cpus)
    assert first_use.ensure() is None
    # A quick row is not enough for a scan's workers: they calibrate at standard.
    with pytest.warns(CalibrationNotice, match="standard"):
        first_use.for_workers(cpus)
    assert auto == [("standard", cpus)]
    _publish("standard", cpus)
    first_use._DONE.clear()  # pyright: ignore[reportPrivateUsage]
    assert first_use.for_workers(cpus) is None


def test_a_recent_inconclusive_attempt_waits_a_day(
    auto: list[tuple[str, tuple[int, ...]]],
) -> None:
    found = calibration.choice()
    assert found.fingerprint is not None and found.allocation is not None
    store = _native.store.Store(calibration.store_directory())
    store.record_inconclusive(found.fingerprint.digest, found.allocation.key())
    assert first_use.ensure() is None
    store.record_inconclusive(
        found.fingerprint.digest,
        found.allocation.key(),
        when=time.time() - first_use.RETRY_SECONDS - 1,
    )
    with pytest.warns(CalibrationNotice):
        first_use.ensure()
    assert len(auto) == 1


def test_concurrent_first_runs_wait_for_the_calibrating_one(
    auto: list[tuple[str, tuple[int, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    found = calibration.choice()
    assert found.fingerprint is not None and found.allocation is not None
    store = _native.store.Store(calibration.store_directory())
    claim = store.claim(found.fingerprint.digest, found.allocation.key(), 1e12)
    assert claim is not None
    waited: list[float] = []

    def sleep(seconds: float) -> None:
        waited.append(seconds)
        claim.release()

    monkeypatch.setattr(first_use.time, "sleep", sleep)
    assert first_use.ensure() is None
    assert waited
    assert auto == []
    # Another host's claim: proceed on the table without calibrating or waiting.
    path = store.directory(found.fingerprint.digest) / f"claim-{found.allocation.key()}"
    path.write_text(
        json.dumps(
            {"host": socket.gethostname() + "-other", "pid": 1, "deadline": 1e12}
        ),
        encoding="utf-8",
    )
    first_use._DONE.clear()  # pyright: ignore[reportPrivateUsage]
    assert first_use.ensure() is None
    assert auto == []


def test_the_build_record_says_whether_the_build_runs_on_a_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detect = first_use.__dict__["_device_build"]
    cpu, gpu, empty = tmp_path / "cpu", tmp_path / "gpu", tmp_path / "empty"
    for directory, backend in ((cpu, "none"), (gpu, "cuda")):
        directory.mkdir()
        (directory / "_build.json").write_text(
            json.dumps({"device_backend": backend}), encoding="utf-8"
        )
    empty.mkdir()
    monkeypatch.setattr(_native, "packages", lambda: (empty, cpu))
    assert detect() is False
    monkeypatch.setattr(_native, "packages", lambda: (gpu,))
    assert detect() is True
    monkeypatch.setattr(_native, "packages", lambda: (empty,))
    assert detect() is False


def test_selecting_native_and_admitting_a_scan_ask_for_a_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solprop.backends import registry  # noqa: PLC0415
    from solprop.driver.executors import pool  # noqa: PLC0415

    asked: list[str] = []
    monkeypatch.setattr(first_use, "ensure", lambda: asked.append("ensure"))

    class StopError(Exception):
        pass

    def stop(*args: object) -> None:
        raise StopError

    monkeypatch.setattr(registry, "_check_required_device", stop)
    with pytest.raises(StopError):
        registry.select("native", problems.build("gnlse").translate())
    assert asked == ["ensure"]
    monkeypatch.setattr(first_use, "for_workers", lambda cpus: asked.append("workers"))
    monkeypatch.setattr(pool, "_spawn", stop)
    with pytest.raises(StopError):
        pool.Pool(threads=1)._admit(lambda: None, None)  # pyright: ignore[reportPrivateUsage]
    assert asked == ["ensure", "workers"]


def test_without_the_native_backend_first_use_does_nothing(
    auto: list[tuple[str, tuple[int, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from solprop.exceptions import BackendError  # noqa: PLC0415

    def missing() -> tuple[Path, ...]:
        raise BackendError("install solprop-native")

    monkeypatch.setattr(_native, "packages", missing)
    assert first_use.ensure() is None
    assert first_use.for_workers([0]) is None
    assert auto == []
