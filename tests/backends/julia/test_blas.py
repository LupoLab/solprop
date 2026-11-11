# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep Julia, construction and native BLAS lifetimes independently owned."""

import gc
import threading
from types import SimpleNamespace
from weakref import ref

import pytest
import threadpoolctl

from backends.julia.cases import cpu_case
from solprop import _interrupts, _process
from solprop.backends.julia import _accelerate, _blas, _runtime, backend
from solprop.backends.julia._blas import CPUBlasInfo, acquire_blas
from solprop.backends.julia._ownership import OwnerRegistry
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia.backend import JuliaBackend
from solprop.exceptions import BackendError
from test_construction_blas import Controller, Library


@pytest.fixture
def apple_mode(monkeypatch, tmp_path):
    mode = SimpleNamespace(value=0, refuse=None)

    def setter(value):
        if value == mode.refuse:
            return -1
        mode.value = value
        return 0

    library = _accelerate._AccelerateLibrary(
        str(tmp_path / "Accelerate"),
        mode,
        lambda: mode.value,
        setter,
        threading.get_ident(),
    )
    provider = CPUBlasInfo(
        "accelerate", library.filepath, "ilp64", "/julia/OpenBLAS", "calling_thread", ""
    )
    monkeypatch.setattr(_accelerate, "_library", lambda: library)
    return mode, library, provider


@pytest.mark.parametrize("release_first", [0, 1])
def test_verified_apple_provider_joins_lifetime_limits_and_keeps_enum_out_of_counts(
    libraries, apple_mode, release_first, monkeypatch
):
    mode, library, provider = apple_mode
    libraries.append(Library(library.filepath))
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", "3")
    first, second = acquire_blas(provider=provider), acquire_blas(provider=provider)
    assert mode.value == 1
    record = first.metadata()
    assert record["blas_provider"] == "accelerate"
    assert record["blas_integer_interface"] == "ilp64"
    assert record["blas_fallback_library"] == "OpenBLAS"
    assert record["blas_libraries"] == [{"library": "libblas", "threads": 3}]
    assert record["accelerate_threads_requested"] == 3
    assert record["accelerate_threads"] == 1
    holds = (first, second)
    holds[release_first].close()
    assert mode.value == 1 and libraries[0].num_threads == 3
    holds[1 - release_first].close()
    assert mode.value == 0 and libraries[0].num_threads == 12
    assert not _process._BLAS_POOLS


def test_apple_framework_used_by_native_shares_the_calling_thread_mode(
    libraries, apple_mode, monkeypatch
):
    mode, library, provider = apple_mode
    libraries.append(Library(library.filepath))
    monkeypatch.setitem(
        _process.sys.modules,
        "solprop_native",
        SimpleNamespace(
            build_info=lambda: SimpleNamespace(blas_library=library.filepath)
        ),
    )
    hold = acquire_blas(provider=provider)
    assert mode.value == 1
    assert hold.metadata()["blas_libraries"] == [{"library": "libblas", "threads": 1}]
    hold.close()
    assert mode.value == 0


def test_verified_apple_provider_requires_its_documented_control(
    libraries, apple_mode, monkeypatch
):
    _, _, provider = apple_mode
    monkeypatch.setattr(_accelerate, "_library", lambda: None)
    with pytest.raises(BackendError, match="no threading control"):
        acquire_blas(provider=provider)
    assert libraries[0].num_threads == 12 and not _process._BLAS_POOLS


def test_process_pool_acquisition_failure_releases_the_apple_mode(
    libraries, apple_mode, monkeypatch
):
    mode, library, provider = apple_mode
    acquire = _process.acquire_host_blas
    failure = MemoryError("process pool construction")

    def failing(controllers, threads):
        if tuple(controllers) == (library,):
            return acquire(controllers, threads)
        raise failure

    monkeypatch.setattr(_process, "acquire_host_blas", failing)
    with pytest.raises(MemoryError, match="process pool construction") as caught:
        acquire_blas(provider=provider)
    assert caught.value is failure
    assert mode.value == 0 and libraries[0].num_threads == 12
    assert not _process._BLAS_POOLS


def test_apple_release_failure_still_restores_other_pools_and_retains_a_retry(
    libraries, apple_mode
):
    mode, _, provider = apple_mode
    hold = acquire_blas(provider=provider)
    mode.refuse = 0
    with pytest.raises(BackendError, match="refused"):
        hold.close()
    assert libraries[0].num_threads == 12 and mode.value == 1
    assert tuple(_blas._FAILED_RELEASES.values()) == (hold,)
    mode.refuse = None
    retry = acquire_blas(provider=provider)
    assert not _blas._FAILED_RELEASES
    retry.close()
    assert mode.value == 0 and not _process._BLAS_POOLS


def test_host_release_failure_still_restores_the_apple_mode(
    libraries, apple_mode, monkeypatch
):
    mode, _, provider = apple_mode
    hold = acquire_blas(provider=provider)
    setter = libraries[0].set_num_threads

    def fail_once(threads):
        monkeypatch.setattr(libraries[0], "set_num_threads", setter)
        raise RuntimeError("foreign host restore")

    monkeypatch.setattr(libraries[0], "set_num_threads", fail_once)
    with pytest.raises(RuntimeError, match="foreign host restore"):
        hold.close()
    assert mode.value == 0 and tuple(_blas._FAILED_RELEASES.values()) == (hold,)
    hold.close()
    assert libraries[0].num_threads == 12 and not _process._BLAS_POOLS


def test_hold_allocation_failure_precedes_either_native_mutation(
    libraries, apple_mode, monkeypatch
):
    mode, _, provider = apple_mode

    def refuse(*_):
        raise MemoryError("BLAS hold")

    monkeypatch.setattr(_blas, "BlasHold", refuse)
    with pytest.raises(MemoryError, match="BLAS hold"):
        acquire_blas(provider=provider)
    assert mode.value == 0 and libraries[0].num_threads == 12
    assert not _process._BLAS_POOLS


@pytest.mark.parametrize("retry_fails", [False, True])
def test_orderly_cleanup_retries_unpublished_acquisitions_and_other_holds(
    libraries, apple_mode, monkeypatch, retry_fails
):
    mode, _, provider = apple_mode
    original = _accelerate._library
    library = original()
    assert library is not None
    setter = library.setter
    failed = RuntimeError("Apple acquisition")
    mode.broken = True

    def break_both(value):
        result = setter(value)
        if mode.broken:
            raise failed
        return result

    monkeypatch.setattr(
        _accelerate,
        "_library",
        lambda: _accelerate._AccelerateLibrary(
            library.filepath, mode, library.getter, break_both, library.thread
        ),
    )
    with pytest.raises(RuntimeError, match="Apple acquisition"):
        acquire_blas(provider=provider)
    assert len(_process._BLAS_PENDING) == 1
    released = []
    other = _blas.BlasHold(1, (), lambda: released.append(True))
    _blas._FAILED_RELEASES[id(other)] = other
    if retry_fails:
        with pytest.raises(RuntimeError, match="Apple acquisition"):
            _blas._release_pending()
        assert released == [True] and len(_process._BLAS_PENDING) == 1
    mode.broken = False
    _blas._release_pending()
    assert released == [True]
    assert mode.value == 0 and not _process._BLAS_PENDING
    assert not _process._BLAS_POOLS and not _blas._FAILED_RELEASES


@pytest.fixture(autouse=True)
def _isolated_fft_preparation(monkeypatch):
    """Keep foreign BLAS lifetime witnesses independent of FFT artifact fixtures."""
    monkeypatch.setattr(backend, "load_planning_cache", lambda _: None)


@pytest.fixture
def libraries(monkeypatch):
    loaded = [Library("/numpy/libblas", 12)]
    monkeypatch.setattr(_process, "_BLAS_POOLS", {})
    monkeypatch.setattr(_process, "_BLAS_PENDING", {})
    monkeypatch.setattr(_blas, "_FAILED_RELEASES", {})
    monkeypatch.setattr(
        _process, "thread_budget", lambda: _process.ThreadBudget(4, "affinity")
    )
    monkeypatch.setattr(
        threadpoolctl, "ThreadpoolController", lambda: Controller(loaded)
    )
    monkeypatch.delenv("SOLPROP_BLAS_THREADS", raising=False)
    return loaded


@pytest.mark.parametrize("construction_first", [False, True])
@pytest.mark.parametrize("release_first", ["construction", "julia"])
def test_blas_owners_restore_only_the_remaining_shared_limit(
    libraries, monkeypatch, construction_first, release_first
):
    construction = _process.host_blas()
    if construction_first:
        construction.__enter__()
    # Julia's newly loaded library was absent from the outer construction walk.
    libraries.append(Library("/julia/libblas", 8))
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", "3")
    hold = acquire_blas()
    if not construction_first:
        monkeypatch.setenv("SOLPROP_BLAS_THREADS", "1")
        construction.__enter__()
    assert libraries[0].num_threads == 1
    assert libraries[1].num_threads == (3 if construction_first else 1)
    if release_first == "construction":
        construction.__exit__(None, None, None)
        assert [library.num_threads for library in libraries] == [3, 3]
        assert hold.metadata() == {
            "blas_threads_requested": 3,
            "blas_libraries": [
                {"library": "libblas", "threads": 3},
                {"library": "libblas", "threads": 3},
            ],
        }
        hold.close()
        hold.close()
    else:
        hold.close()
        hold.close()
        assert libraries[0].num_threads == 1
        assert libraries[1].num_threads == (8 if construction_first else 1)
        construction.__exit__(None, None, None)
    assert [library.num_threads for library in libraries] == [12, 8]


def test_blas_discovery_excludes_the_loaded_native_private_library(
    libraries, monkeypatch
):
    libraries.append(Library("/native/libblas", 6))
    monkeypatch.setitem(
        _process.sys.modules,
        "solprop_native",
        SimpleNamespace(
            build_info=lambda: SimpleNamespace(blas_library="/native/libblas")
        ),
    )
    hold = acquire_blas()
    assert [library.num_threads for library in libraries] == [1, 6]
    hold.close()
    assert [library.num_threads for library in libraries] == [12, 6]


def test_blas_invalid_request_leaves_loaded_pools_unchanged(libraries, monkeypatch):
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", "invalid")
    with pytest.raises(ValueError, match="SOLPROP_BLAS_THREADS"):
        acquire_blas()
    assert libraries[0].num_threads == 12


def test_blas_request_is_bounded_by_the_process_grant(libraries, monkeypatch):
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", "8")
    hold = acquire_blas()
    assert libraries[0].num_threads == hold.requested == 4
    hold.close()


def test_blas_empty_discovery_does_not_touch_a_pool(libraries):
    libraries.clear()
    hold = acquire_blas()
    assert hold.metadata() == {"blas_threads_requested": 1, "blas_libraries": []}
    hold.close()


def test_failed_registry_release_retains_only_the_python_lease_for_retry(
    libraries, monkeypatch
):
    hold = acquire_blas()
    registry = OwnerRegistry()
    handle = registry.retain(object(), on_release=hold.close)
    setter = libraries[0].set_num_threads

    def fail_once(threads):
        monkeypatch.setattr(libraries[0], "set_num_threads", setter)
        raise RuntimeError("test foreign setter failure")

    monkeypatch.setattr(libraries[0], "set_num_threads", fail_once)
    with pytest.raises(BackendError, match="owner cleanup failed"):
        handle.close()
    assert not registry._entries
    assert tuple(_blas._FAILED_RELEASES.values()) == (hold,)
    next_owner = acquire_blas()
    assert not _blas._FAILED_RELEASES
    assert libraries[0].num_threads == 1
    next_owner.close()
    assert libraries[0].num_threads == 12
    registry.close()


@pytest.mark.parametrize("abandon", [False, True])
def test_preparation_owns_the_blas_hold_until_main_thread_release(
    libraries, monkeypatch, setup_record, abandon
):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())
    engine = SimpleNamespace(prepare_modal=lambda *_: object())
    runtime = JuliaRuntime(setup_record, 2, None, engine, 1)
    monkeypatch.setattr(backend, "get_runtime", lambda **_: runtime)
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    prepared = owner.prepare(case.problem, case.request)
    held = ref(owner)
    assert libraries[0].num_threads == 1
    assert owner.metadata()["blas_threads_requested"] == 1
    if abandon:
        del owner, prepared
        gc.collect()
        assert held() is None
        runtime.owners.drain()
    else:
        owner.close()
    assert libraries[0].num_threads == 12
    assert not runtime.owners._entries
    runtime.owners.close()


@pytest.mark.parametrize("fail", [False, True])
def test_device_preparation_forwards_target_without_cpu_blas_or_fft_resources(
    monkeypatch, setup_record, fail
):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())
    selected = []
    arguments = []

    def construct(*args):
        arguments.append(args)
        if fail:
            raise RuntimeError("device preparation failure")
        return object()

    def cpu_only(*args, **kwargs):
        pytest.fail("device preparation must not request CPU BLAS or FFT resources")

    runtime = JuliaRuntime(
        setup_record, 2, None, SimpleNamespace(prepare_modal=construct), 1
    )

    def get_runtime(*, setup_id, device):
        selected.append((setup_id, device))
        return runtime

    monkeypatch.setattr(backend, "get_runtime", get_runtime)
    monkeypatch.setattr(backend, "acquire_blas", cpu_only)
    monkeypatch.setattr(backend, "load_planning_cache", cpu_only)
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    # Public admission remains closed until hardware qualification. Exercise the
    # internal transaction after the separate, pure capability admission boundary.
    owner._device = "cuda"
    if fail:
        with pytest.raises(BackendError, match="device preparation failure"):
            owner.prepare(case.problem, case.request)
        assert owner._preparation is None and owner._runtime is None
        assert not runtime.owners._entries
    else:
        prepared = owner.prepare(case.problem, case.request)
        assert prepared._owner is owner
        assert owner._preparation is not None
    assert selected == [(None, "cuda")]
    assert len(arguments) == 1
    assert arguments[0][1:] == (2, owner._node_chunk, "cuda")
    assert owner._blas is None and owner._fft_cache is None
    assert owner.metadata()["device"] == "cuda"
    assert "blas_threads_requested" not in owner.metadata()
    owner.close()
    assert not runtime.owners._entries
    runtime.owners.close()


def test_failed_numerical_preparation_releases_its_unpublished_blas_hold(
    libraries, monkeypatch, setup_record
):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())

    def fail(*_):
        assert libraries[0].num_threads == 1
        raise RuntimeError("test constructor failure")

    engine = SimpleNamespace(prepare_modal=fail)
    runtime = JuliaRuntime(setup_record, 2, None, engine, 1)
    monkeypatch.setattr(backend, "get_runtime", lambda **_: runtime)
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    with pytest.raises(BackendError, match="test constructor failure"):
        owner.prepare(case.problem, case.request)
    assert libraries[0].num_threads == 12
    assert not runtime.owners._entries
    assert owner._blas is None and owner._preparation is None
    owner.close()
    runtime.owners.close()


def test_deferred_interrupt_observes_a_fully_owned_preparation(
    libraries, monkeypatch, setup_record
):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())
    monkeypatch.setattr(_interrupts, "_GUARD", _interrupts._Guard())

    def construct(*_):
        _interrupts._GUARD.pending = KeyboardInterrupt()
        return object()

    engine = SimpleNamespace(prepare_modal=construct)
    runtime = JuliaRuntime(setup_record, 2, None, engine, 1)
    monkeypatch.setattr(backend, "get_runtime", lambda **_: runtime)
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    with pytest.raises(KeyboardInterrupt):
        owner.prepare(case.problem, case.request)
    assert owner._runtime is runtime
    assert owner._preparation is not None and owner._blas is not None
    assert libraries[0].num_threads == 1
    owner.close()
    assert libraries[0].num_threads == 12
    assert not runtime.owners._entries
    runtime.owners.close()


def test_constructor_failure_remains_primary_when_blas_restoration_needs_retry(
    libraries, monkeypatch, setup_record
):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())
    setter = libraries[0].set_num_threads

    def fail_restore(threads):
        monkeypatch.setattr(libraries[0], "set_num_threads", setter)
        raise RuntimeError("test restoration failure")

    def construct(*_):
        monkeypatch.setattr(libraries[0], "set_num_threads", fail_restore)
        raise RuntimeError("test construction failure")

    runtime = JuliaRuntime(
        setup_record, 2, None, SimpleNamespace(prepare_modal=construct), 1
    )
    monkeypatch.setattr(backend, "get_runtime", lambda **_: runtime)
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    with pytest.raises(BackendError, match="test construction failure") as caught:
        owner.prepare(case.problem, case.request)
    assert any("test restoration failure" in note for note in caught.value.__notes__)
    assert len(_blas._FAILED_RELEASES) == 1
    retry = acquire_blas()
    retry.close()
    assert libraries[0].num_threads == 12 and not _blas._FAILED_RELEASES
    owner.close()
    runtime.owners.close()
