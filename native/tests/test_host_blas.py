# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""NumPy's and SciPy's BLAS pools hold one thread while a native backend lives."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pytest

from solprop_native import _host_blas


@dataclass
class _Library:
    filepath: str
    internal_api: str
    num_threads: int = 16


@dataclass
class _Limiter:
    controller: _Controller
    limits: int
    restored: bool = False

    def restore_original_limits(self) -> None:
        self.restored = True
        for library in self.controller.lib_controllers:
            library.num_threads = 16


@dataclass
class _Controller:
    lib_controllers: list[_Library]
    limiters: list[_Limiter] = field(default_factory=list)

    def select(self, **criteria: object) -> _Controller:
        def wanted(library: _Library) -> bool:
            for name, value in criteria.items():
                values = value if isinstance(value, list) else [value]
                if name == "user_api":
                    if "blas" not in values:
                        return False
                elif getattr(library, name) not in values:
                    return False
            return True

        chosen = _Controller([one for one in self.lib_controllers if wanted(one)])
        chosen.limiters = self.limiters
        return chosen

    def limit(self, limits: int) -> _Limiter:
        for library in self.lib_controllers:
            library.num_threads = limits
        limiter = _Limiter(self, limits)
        self.limiters.append(limiter)
        return limiter


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no backend holding the pools."""
    monkeypatch.setattr(_host_blas, "_holders", 0)
    monkeypatch.setattr(_host_blas, "_limiter", None)
    monkeypatch.setattr(_host_blas, "_reached", "unset")
    monkeypatch.delenv("SOLPROP_BLAS_THREADS", raising=False)


def _machine() -> tuple[_Controller, _Library, _Library]:
    numpy = _Library("/site/numpy.libs/libscipy_openblas64_.so", "openblas")
    native = _Library("/site/native/libopenblas.so.0", "openblas")
    return _Controller([numpy, native]), numpy, native


def test_other_blas_pools_are_held_and_the_last_backend_restores_them() -> None:
    controller, numpy, native = _machine()
    first = _host_blas.hold(native.filepath, 16, controller)
    assert first == "openblas: 1 thread"
    assert (numpy.num_threads, native.num_threads) == (1, 16)
    assert _host_blas.hold(native.filepath, 16, controller) == first
    assert len(controller.limiters) == 1
    _host_blas.release()
    assert numpy.num_threads == 1
    _host_blas.release()
    assert controller.limiters[0].restored
    assert numpy.num_threads == 16
    _host_blas.release()  # nothing held: no effect
    assert numpy.num_threads == 16


def test_an_explicit_count_is_kept_within_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, numpy, native = _machine()
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", "4")
    assert _host_blas.hold(native.filepath, 2, controller) == "openblas: 2 threads"
    assert numpy.num_threads == 2
    _host_blas.release()


def test_without_another_blas_nothing_is_held() -> None:
    native = _Library("/site/native/libopenblas.so.0", "openblas")
    controller = _Controller([native])
    assert _host_blas.hold(native.filepath, 16, controller) == "none"
    assert native.num_threads == 16
    _host_blas.release()


def test_without_threadpoolctl_the_pools_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    assert _host_blas.hold("", 16) == "unavailable"
    _host_blas.release()


def test_the_real_walk_finds_numpys_blas() -> None:
    pytest.importorskip("threadpoolctl")
    reached = _host_blas.hold("", 16)
    try:
        assert reached in {"none"} or reached.endswith("1 thread")
    finally:
        _host_blas.release()
