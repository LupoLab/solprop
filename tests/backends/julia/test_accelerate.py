# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep Apple calling-thread enums separate from ordinary BLAS widths."""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from solprop import _process
from solprop.backends.julia import _accelerate
from solprop.backends.julia._accelerate import acquire_accelerate
from solprop.exceptions import BackendError


class Mode:
    def __init__(self, initial=0):
        self.current = initial
        self.calls = []
        self.refuse = None
        self.ignore = False

    def get(self):
        return self.current

    def set(self, mode):
        self.calls.append(mode)
        if mode == self.refuse:
            return -1
        if not self.ignore:
            self.current = mode
        return 0


@pytest.fixture
def apple(monkeypatch, tmp_path):
    mode = Mode()
    library = _accelerate._AccelerateLibrary(
        str(tmp_path / "Accelerate"), mode, mode.get, mode.set, threading.get_ident()
    )
    monkeypatch.setattr(_process, "_BLAS_POOLS", {})
    monkeypatch.setattr(_accelerate, "_library", lambda: library)
    return mode, library


@pytest.mark.parametrize("initial", [0, 1])
@pytest.mark.parametrize("release_first", [0, 1])
def test_overlapping_owners_use_single_mode_and_restore_exact_enum(
    apple, initial, release_first
):
    mode, _ = apple
    mode.current = initial
    first, second = acquire_accelerate(7), acquire_accelerate(2)
    assert first is not None and second is not None
    holds = (first, second)
    assert mode.current == 1
    for hold, requested in zip(holds, (7, 2), strict=True):
        assert hold.requested == requested
        record = hold.metadata()
        assert record["accelerate_threads_requested"] == requested
        assert record["accelerate_threading_mode"] == "single"
        assert record["accelerate_threads"] == 1
        assert record["accelerate_thread_scope"] == "calling-thread"
    holds[release_first].close()
    holds[release_first].close()
    assert mode.current == 1
    holds[1 - release_first].close()
    assert mode.current == initial and not _process._BLAS_POOLS
    record = holds[0].metadata()
    assert record["accelerate_threads"] == (1 if initial else None)
    assert record["accelerate_threading_mode"] == ("single" if initial else "automatic")
    assert all(value in (0, 1) for value in mode.calls)


def test_other_blas_widths_keep_their_independent_lease(apple, tmp_path):
    mode, _ = apple

    class Other:
        filepath = str(tmp_path / "OpenBLAS")
        num_threads = 8

        def set_num_threads(self, count):
            self.num_threads = count

    other = Other()
    lease = _process.acquire_host_blas((other,), 3)
    hold = acquire_accelerate(5)
    assert hold is not None
    assert mode.current == 1 and other.num_threads == 3
    hold.close()
    assert mode.current == 0 and other.num_threads == 3
    lease.restore_original_limits()
    assert other.num_threads == 8


@pytest.mark.parametrize("requested", [0, -1, True, False, 1.5])
def test_invalid_requests_do_not_touch_threading_state(apple, requested):
    mode, _ = apple
    with pytest.raises((ValueError, TypeError), match=r"positive integer|int"):
        acquire_accelerate(requested)
    assert mode.current == 0 and mode.calls == [] and not _process._BLAS_POOLS


def test_missing_control_does_not_acquire_a_pool(monkeypatch):
    monkeypatch.setattr(_accelerate, "_library", lambda: None)
    before = dict(_process._BLAS_POOLS)
    assert acquire_accelerate(3) is None
    assert before == _process._BLAS_POOLS


def test_unknown_initial_enum_refuses_without_mutation(apple):
    mode, _ = apple
    mode.current = 9
    with pytest.raises(BackendError, match="unknown threading mode"):
        acquire_accelerate(1)
    assert mode.calls == [] and not _process._BLAS_POOLS


@pytest.mark.parametrize("ignore", [False, True])
def test_failed_single_mode_acquisition_restores_and_releases(apple, ignore):
    mode, _ = apple
    mode.refuse = None if ignore else 1
    mode.ignore = ignore
    with pytest.raises(BackendError, match=r"refused|did not retain"):
        acquire_accelerate(2)
    assert mode.current == 0 and mode.calls == [1, 0]
    assert not _process._BLAS_POOLS


def test_failed_final_restore_keeps_the_owner_for_retry(apple):
    mode, _ = apple
    hold = acquire_accelerate(4)
    assert hold is not None
    mode.refuse = 0
    with pytest.raises(BackendError, match="refused"):
        hold.close()
    assert mode.current == 1 and _process._BLAS_POOLS
    mode.refuse = None
    hold.close()
    assert mode.current == 0 and not _process._BLAS_POOLS


def test_failed_hold_construction_precedes_all_native_mutation(apple, monkeypatch):
    mode, _ = apple
    failure = MemoryError("hold construction")

    def fail(*_):
        raise failure

    monkeypatch.setattr(_accelerate, "AccelerateHold", fail)
    with pytest.raises(MemoryError, match="hold construction") as caught:
        acquire_accelerate(4)
    assert caught.value is failure
    assert mode.current == 0 and mode.calls == [] and not _process._BLAS_POOLS


def test_unacquired_owner_close_has_no_native_effect(apple):
    mode, library = apple
    hold = _accelerate.AccelerateHold(1, library)
    hold.close()
    assert mode.current == 0 and mode.calls == []


def test_private_controller_refuses_ordinary_count_tokens(apple):
    mode, library = apple
    with pytest.raises(BackendError, match="single mode"):
        library.set_num_threads(7)
    assert mode.calls == []


def test_controller_checks_its_exact_acquiring_thread(apple):
    _, library = apple
    moved = replace(library, thread=-1)
    with pytest.raises(BackendError, match="acquiring thread"):
        _ = moved.num_threads


def test_workers_cannot_acquire_read_or_restore_the_main_thread_hold(apple):
    mode, _ = apple
    hold = acquire_accelerate(3)
    assert hold is not None
    with ThreadPoolExecutor(max_workers=1) as pool:
        for operation in (lambda: acquire_accelerate(2), hold.metadata, hold.close):
            with pytest.raises(BackendError, match="main thread"):
                pool.submit(operation).result(timeout=5)
    assert mode.current == 1
    hold.close()
    assert mode.current == 0


def test_nonapple_discovery_never_loads_a_library(monkeypatch):
    monkeypatch.setattr(_accelerate.sys, "platform", "linux")

    def forbidden(*_):
        raise AssertionError("unneeded library load")

    monkeypatch.setattr(_accelerate.ctypes, "CDLL", forbidden)
    assert _accelerate._library() is None


@pytest.mark.parametrize("missing", ["library", "symbol"])
def test_unavailable_apple_controls_are_reported_as_absent(monkeypatch, missing):
    monkeypatch.setattr(_accelerate.sys, "platform", "darwin")

    def absent(*_):
        raise OSError("missing library")

    if missing == "library":
        monkeypatch.setattr(_accelerate.ctypes, "CDLL", absent)
    else:
        monkeypatch.setattr(_accelerate.ctypes, "CDLL", lambda *_: object())

        def missing_symbol(*_):
            raise AttributeError("missing symbol")

        monkeypatch.setattr(_accelerate.ctypes, "CFUNCTYPE", lambda *_: missing_symbol)
    assert _accelerate._library() is None


def test_import_does_not_load_julia_or_native_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import solprop.backends.julia._accelerate; "
            "assert 'juliacall' not in sys.modules; "
            "assert 'solprop_native' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_actual_apple_mode_is_thread_local_and_restored(monkeypatch):
    if sys.platform != "darwin":
        assert _accelerate._library() is None
        return
    monkeypatch.setattr(_process, "_BLAS_POOLS", {})
    library = _accelerate._library()
    if library is None:
        # The documented compatibility path uses OpenBLAS before macOS 15.
        assert acquire_accelerate(3) is None
        return
    original = library.num_threads
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_before = pool.submit(library.getter).result(timeout=5)
        first = acquire_accelerate(4)
        second = acquire_accelerate(2)
        assert first is not None and second is not None
        try:
            assert library.num_threads == 1
            assert first.metadata()["accelerate_threads"] == 1
            assert second.metadata()["accelerate_threads_requested"] == 2
            assert pool.submit(library.getter).result(timeout=5) == worker_before
            first.close()
            assert library.num_threads == 1
        finally:
            first.close()
            second.close()
        assert library.num_threads == original
        assert pool.submit(library.getter).result(timeout=5) == worker_before
