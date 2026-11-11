# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate process identity and initialization before exercising the real bridge."""

import os
import signal
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from solprop.backends.julia import _blas, _runtime
from solprop.exceptions import BackendError


@pytest.fixture
def isolated_runtime(monkeypatch, setup_record):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())
    monkeypatch.setattr(_runtime.os, "environ", dict(os.environ))
    monkeypatch.setattr(_runtime.sys, "_xoptions", {})
    monkeypatch.delitem(_runtime.sys.modules, "juliacall", raising=False)
    monkeypatch.setattr(_runtime, "dllist", lambda: [])
    monkeypatch.setattr(
        _runtime._process, "thread_budget", lambda: SimpleNamespace(threads=2)
    )
    selected = []

    def select(**kwargs):
        selected.append(kwargs)
        return setup_record

    monkeypatch.setattr(_runtime, "select_setup", select)
    evaluations = []
    values = {
        "collect(LOAD_PATH)": [str(path) for _, path in setup_record.projects]
        + ["@stdlib"],
        "collect(DEPOT_PATH)": [str(setup_record.depot)],
        "Base.active_project()": str(setup_record.projects[0][1] / "Project.toml"),
        "Base.Threads.nthreads(:default)": 2,
        "Base.Threads.nthreads(:interactive)": 0,
        "string(VERSION)": "1.13.3",
        "string(Base.pkgversion(PythonCall))": "0.9.37",
        (
            "Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing"
        ): True,
    }

    def evaluate(code):
        evaluations.append(code)
        return values.get(code)

    compiled = []
    engine = SimpleNamespace(stop_setter_address=lambda: compiled.append(True) or 1)
    setattr(
        engine,
        "configure_cpu_blas!",
        lambda: SimpleNamespace(
            provider="openblas",
            library="libopenblas.so",
            interface="ilp64",
            fallback_library="",
            thread_scope="process",
            reason="not-macos",
        ),
    )
    namespace = SimpleNamespace(seval=evaluate, SolpropJulia=engine)
    imports = []
    bridge = SimpleNamespace(__version__="0.9.37", newmodule=lambda _: namespace)

    def load(name):
        imports.append(name)
        return bridge

    importer = SimpleNamespace(import_module=load)
    monkeypatch.setattr(_runtime, "importlib", importer)
    actual_handler = object()
    restored = []
    monkeypatch.setattr(
        _runtime,
        "signal",
        SimpleNamespace(
            SIGINT=signal.SIGINT,
            getsignal=lambda _: actual_handler,
            signal=lambda number, handler: restored.append((number, handler)),
        ),
    )
    return SimpleNamespace(
        selected=selected,
        evaluations=evaluations,
        values=values,
        compiled=compiled,
        imports=imports,
        importer=importer,
        bridge=bridge,
        namespace=namespace,
        actual_handler=actual_handler,
        restored=restored,
    )


def test_import_and_inspection_never_start_the_runtime(isolated_runtime):
    assert not _runtime._STATE.attempted
    assert isolated_runtime.imports == []


def test_first_preparation_pins_stack_and_restores_actual_sigint(
    isolated_runtime, setup_record
):
    runtime = _runtime.get_runtime()
    assert runtime.setup == setup_record
    assert runtime.threads == 2
    assert runtime.cpu_blas == _blas.CPUBlasInfo(
        "openblas", "libopenblas.so", "ilp64", "", "process", "not-macos"
    )
    assert isolated_runtime.imports == ["juliacall"]
    assert isolated_runtime.compiled == [True]
    assert isolated_runtime.restored == [
        (signal.SIGINT, isolated_runtime.actual_handler)
    ]
    assert os.environ["JULIA_LOAD_PATH"] == os.pathsep.join(
        [str(path) for _, path in setup_record.projects] + ["@stdlib"]
    )
    assert os.environ["JULIA_DEPOT_PATH"] == str(setup_record.depot)
    assert os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "yes"
    assert os.environ["PYTHON_JULIACALL_THREADS"] == "2,0"
    runtime.check()


def test_runtime_installs_blas_cleanup_before_bridge_and_owner_shutdown(
    isolated_runtime, monkeypatch
):
    order = []
    monkeypatch.setattr(_blas._CLEANUP, "registered", False)
    monkeypatch.setattr(_blas.atexit, "register", order.append)
    importer = isolated_runtime.importer.import_module

    def load(name):
        assert order == [_blas._release_pending]
        order.append(name)
        return importer(name)

    monkeypatch.setattr(isolated_runtime.importer, "import_module", load)
    runtime = _runtime.get_runtime()
    assert order == [_blas._release_pending, "juliacall", runtime.owners.close]
    _blas.prepare_blas_cleanup()
    assert len(order) == 3
    runtime.owners.close()


def test_repeated_preparation_rechecks_inventory_before_loading_device(
    isolated_runtime,
):
    first = _runtime.get_runtime()
    assert _runtime.get_runtime(device="cuda") is first
    assert _runtime.get_runtime(device="cuda") is first
    assert len(isolated_runtime.selected) == 3
    assert isolated_runtime.evaluations.count("import CUDA") == 1
    assert isolated_runtime.imports == ["juliacall"]


def test_missing_artifact_prevents_import_or_extension(monkeypatch, isolated_runtime):
    def missing(**_):
        raise BackendError("missing artifact: run solprop julia setup")

    select = _runtime.select_setup
    monkeypatch.setattr(_runtime, "select_setup", missing)
    with pytest.raises(BackendError, match="missing artifact"):
        _runtime.get_runtime()
    assert isolated_runtime.imports == []
    monkeypatch.setattr(_runtime, "select_setup", select)
    _runtime.get_runtime()
    monkeypatch.setattr(_runtime, "select_setup", missing)
    with pytest.raises(BackendError, match="missing artifact"):
        _runtime.get_runtime(device="cuda")
    assert "import CUDA" not in isolated_runtime.evaluations
    assert _runtime._STATE.failure is None


def test_new_setup_requires_fresh_process(monkeypatch, isolated_runtime, setup_record):
    _runtime.get_runtime()
    different = replace(setup_record, setup_id="e" * 64)
    monkeypatch.setattr(_runtime, "select_setup", lambda **_: different)
    with pytest.raises(BackendError, match="requires a fresh process"):
        _runtime.get_runtime(setup_id=different.setup_id)
    assert isolated_runtime.imports == ["juliacall"]


def test_same_identity_in_another_depot_cannot_validate_the_running_stack(
    monkeypatch, isolated_runtime, setup_record, tmp_path
):
    _runtime.get_runtime()
    relocated = replace(setup_record, depot=tmp_path / "different-depot")
    monkeypatch.setattr(_runtime, "select_setup", lambda **_: relocated)
    with pytest.raises(BackendError, match="environment location"):
        _runtime.get_runtime(device="cuda")
    assert "import CUDA" not in isolated_runtime.evaluations


def test_first_worker_refuses_then_main_thread_can_initialize(isolated_runtime):
    failures = []

    def first():
        try:
            _runtime.get_runtime()
        except BackendError as error:
            failures.append(str(error))

    worker = threading.Thread(target=first)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(failures) == 1 and "main thread" in failures[0]
    assert isolated_runtime.imports == []
    assert not _runtime._STATE.attempted
    _runtime.get_runtime()
    assert isolated_runtime.imports == ["juliacall"]


def test_worker_refuses_during_main_initialization_without_waiting(isolated_runtime):
    entered = threading.Event()
    results = []
    failures = []

    def concurrent():
        entered.set()
        try:
            results.append(_runtime.get_runtime())
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=concurrent)
    original = isolated_runtime.importer.import_module

    def load(name):
        worker.start()
        assert entered.wait(timeout=5)
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert not results
        return original(name)

    isolated_runtime.importer.import_module = load
    _runtime.get_runtime()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(failures) == 1 and "main thread" in str(failures[0])
    assert results == []
    assert isolated_runtime.imports == ["juliacall"]


def test_recursive_main_initialization_refuses_without_waiting(isolated_runtime):
    original = isolated_runtime.importer.import_module

    def load(name):
        with pytest.raises(BackendError, match="recursive julia runtime"):
            _runtime.get_runtime()
        return original(name)

    isolated_runtime.importer.import_module = load
    runtime = _runtime.get_runtime()
    assert _runtime.get_runtime() is runtime
    assert isolated_runtime.imports == ["juliacall"]


def test_runtime_wrappers_and_calls_refuse_workers_after_start(isolated_runtime):
    runtime = _runtime.get_runtime()
    errors = []

    def worker():
        for operation in (
            _runtime.get_runtime,
            runtime.check,
            lambda: runtime.namespace,
            lambda: runtime.engine,
            lambda: runtime.own_call("StopFlag", 0),
        ):
            try:
                operation()
            except BackendError as error:
                errors.append(str(error))
        repr(runtime)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 5
    assert all("main thread" in error for error in errors)
    assert isolated_runtime.imports == ["juliacall"]


def test_interrupted_shutdown_hook_registration_roots_and_poisons_startup(
    isolated_runtime, monkeypatch
):
    def interrupted(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(_runtime.atexit, "register", interrupted)
    with pytest.raises(KeyboardInterrupt) as captured:
        _runtime.get_runtime()
    assert captured.value.__context__ is None
    assert _runtime._STATE.runtime is not None
    assert isinstance(_runtime._STATE.failure, KeyboardInterrupt)
    with pytest.raises(BackendError, match="previously failed"):
        _runtime.get_runtime()


@pytest.mark.parametrize(
    "failure", [RuntimeError("bridge failed"), KeyboardInterrupt()]
)
def test_partial_initialization_poisoning_and_handler_restoration(
    isolated_runtime, failure
):
    def broken(_):
        raise failure

    isolated_runtime.importer.import_module = broken
    expected = (
        KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else BackendError
    )
    with pytest.raises(expected):
        _runtime.get_runtime()
    assert isolated_runtime.restored == [
        (signal.SIGINT, isolated_runtime.actual_handler)
    ]
    with pytest.raises(BackendError, match="previously failed"):
        _runtime.get_runtime()
    assert _runtime._STATE.failure is not failure
    assert isinstance(_runtime._STATE.failure, expected)
    assert _runtime._STATE.failure.__context__ is None


@pytest.mark.parametrize(
    ("expression", "wrong"),
    [
        ("string(VERSION)", "1.13.2"),
        ("string(Base.pkgversion(PythonCall))", "0.9.36"),
        (
            "Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing",
            False,
        ),
    ],
)
def test_loaded_versions_and_bridge_extension_are_verified(
    isolated_runtime, expression, wrong
):
    isolated_runtime.values[expression] = wrong
    with pytest.raises(BackendError, match="failed to start"):
        _runtime.get_runtime()
    assert _runtime._STATE.failure is not None


def test_loaded_juliacall_version_is_verified(isolated_runtime):
    isolated_runtime.bridge.__version__ = "0.9.36"
    with pytest.raises(BackendError, match="failed to start"):
        _runtime.get_runtime()


def test_partial_blas_provider_configuration_poisons_runtime(
    isolated_runtime, monkeypatch
):
    calls = []

    def broken():
        calls.append(True)
        raise RuntimeError("provider verification failed after forwarding")

    monkeypatch.setattr(
        isolated_runtime.namespace.SolpropJulia, "configure_cpu_blas!", broken
    )
    with pytest.raises(BackendError, match="provider verification failed"):
        _runtime.get_runtime()
    with pytest.raises(BackendError, match="previously failed"):
        _runtime.get_runtime()
    assert calls == [True]
    assert _runtime._STATE.runtime is None
    assert isolated_runtime.compiled == []


def test_partial_device_loading_poisoning(isolated_runtime):
    runtime = _runtime.get_runtime()

    def broken(_):
        raise RuntimeError("device initialization failed")

    isolated_runtime.namespace.seval = broken
    with pytest.raises(BackendError, match="cuda extension failed"):
        _runtime.get_runtime(device="cuda")
    with pytest.raises(BackendError, match="fresh process"):
        runtime.check()
    with pytest.raises(BackendError, match="previously failed"):
        _runtime.get_runtime()
    assert runtime.devices == {"cpu"}


def test_foreign_bridge_import_refuses_before_mutation(monkeypatch, isolated_runtime):
    monkeypatch.setitem(_runtime.sys.modules, "juliacall", SimpleNamespace())
    with pytest.raises(BackendError, match="imported outside"):
        _runtime.get_runtime()
    assert not _runtime._STATE.attempted
    assert isolated_runtime.restored == []


def test_command_line_bridge_overrides_refuse(monkeypatch, isolated_runtime):
    monkeypatch.setitem(_runtime.sys._xoptions, "juliacall-sysimage", "foreign.so")
    with pytest.raises(BackendError, match="-X juliacall"):
        _runtime.get_runtime()
    assert isolated_runtime.imports == []


@pytest.mark.parametrize(
    "library", ["libjulia.dylib", "libjulia.so.1.13", "libjulia.dll"]
)
def test_foreign_libjulia_refuses_before_bridge_import(
    monkeypatch, isolated_runtime, library
):
    monkeypatch.setattr(_runtime, "dllist", lambda: ["", f"/foreign/{library}"])
    with pytest.raises(BackendError, match="libjulia was loaded outside"):
        _runtime.get_runtime()
    assert isolated_runtime.imports == []
    assert not _runtime._STATE.attempted


@pytest.mark.parametrize(
    ("expression", "wrong"),
    [
        ("collect(LOAD_PATH)", ["@v1.13", "@stdlib"]),
        ("collect(DEPOT_PATH)", ["/ambient/depot"]),
        ("Base.active_project()", "/ambient/Project.toml"),
        ("Base.Threads.nthreads(:default)", 4),
        ("Base.Threads.nthreads(:interactive)", 1),
    ],
)
def test_actual_loaded_paths_and_pools_must_match_setup(
    isolated_runtime, expression, wrong
):
    isolated_runtime.values[expression] = wrong
    with pytest.raises(BackendError, match="failed to start"):
        _runtime.get_runtime()
    assert not any(
        "import SolpropJulia" in code for code in isolated_runtime.evaluations
    )


def test_ambient_bridge_options_cannot_bypass_verified_setup(isolated_runtime):
    os.environ["PYTHON_JULIACALL_LIB"] = "foreign.so"
    os.environ["PYTHON_JULIACALL_INIT"] = "no"
    os.environ["JULIA_LOAD_PATH"] = "@v1.13"
    _runtime.get_runtime()
    assert "PYTHON_JULIACALL_LIB" not in os.environ
    assert os.environ["PYTHON_JULIACALL_INIT"] == "yes"
    assert "@v1.13" not in os.environ["JULIA_LOAD_PATH"]


def test_fork_before_start_is_safe_but_fork_after_start_refuses(isolated_runtime):
    _runtime._STATE.pid = -1
    runtime = _runtime.get_runtime()
    assert _runtime._STATE.pid == os.getpid()
    _runtime._STATE.pid = -1
    with pytest.raises(BackendError, match="after fork"):
        _runtime.get_runtime()
    with pytest.raises(BackendError, match="after fork"):
        runtime.check()


def test_smaller_thread_budget_requires_new_runtime(monkeypatch, isolated_runtime):
    _runtime.get_runtime()
    monkeypatch.setattr(
        _runtime._process, "thread_budget", lambda: SimpleNamespace(threads=1)
    )
    with pytest.raises(BackendError, match="current process budget"):
        _runtime.get_runtime()
