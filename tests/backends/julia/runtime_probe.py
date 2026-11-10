# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise one real packaged Julia lifecycle in an isolated Python process.

The pytest caller owns the copied depot and bounds the entire process lifetime.
Importing this helper for doctest collection never initializes either runtime.
"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import gc
import importlib
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
import weakref
from pathlib import Path

_LIVE_EXIT_HANDLES = []

_WORKLOAD = r"""
const probe_thread_mask = Ref(UInt(0))
const probe_retained = Any[]

function probe_batch()
    identifiers = zeros(Int, Threads.nthreads(:default) * 4)
    Threads.@threads :static for item in eachindex(identifiers)
        values = [sin(Float64(index)) for index in 1:512]
        @assert isfinite(sum(values))
        identifiers[item] = Threads.threadid()
    end
    for identifier in identifiers
        probe_thread_mask[] |= UInt(1) << (identifier - 1)
    end
    GC.gc(false)
    return nothing
end

function probe_work(flag, maximum_batches, ready)
    probe_thread_mask[] = 0
    isempty(ready) || write(ready, "entered")
    for batch in 1:maximum_batches
        probe_batch()
        SolpropJulia.stop_requested(flag) && return batch
    end
    return maximum_batches
end

function probe_window(flag, seconds, ready)
    probe_thread_mask[] = 0
    deadline = time_ns() + UInt(round(seconds * 1e9))
    isempty(ready) || write(ready, "entered")
    batches = 0
    while time_ns() < deadline
        probe_batch()
        batches += 1
        SolpropJulia.stop_requested(flag) && break
    end
    return batches
end

function probe_polyester()
    buffer = zeros(Float64, 1024)
    identifiers = zeros(Int, length(buffer))
    # A local named `values` shadows Base.values and defeats @batch capture analysis.
    SolpropJulia.Polyester.@batch for item in eachindex(buffer)
        buffer[item] = Float64(item)
        identifiers[item] = Threads.threadid()
    end
    # Every integer and partial sum is exactly representable in binary64.
    @assert sum(buffer) == 1024 * 1025 / 2
    mask = UInt(0)
    for identifier in identifiers
        mask |= UInt(1) << (identifier - 1)
    end
    GC.gc(false)
    return mask
end

@eval SolpropJulia function probe_owner(path)
    object = Ref(Int(1))
    finalizer(object) do _
        write(path, "finalized")
    end
    return object
end

function probe_fft()
    values = ComplexF64.(1:64)
    plan = SolpropJulia.FFTW.plan_fft(values)
    transformed = plan * values
    recovered = SolpropJulia.FFTW.ifft(transformed)
    # Two length-64 transforms: this is a rounding check, not propagation parity.
    @assert maximum(abs.(recovered .- values)) <= 128eps(Float64) * maximum(abs, values)
    push!(probe_retained, (plan, values, transformed))
    return length(values)
end

nothing
"""


def _runtime_module():
    return importlib.import_module("solprop.backends.julia._runtime")


def _evaluate(runtime, expression):
    def evaluate():
        result = runtime.namespace.seval(expression)
        assert type(result) in (type(None), bool, int, float, str)
        return result

    return importlib.import_module("solprop.backends.julia._bridge").invoke(
        evaluate, "runtime probe evaluation failed"
    )


def _call(runtime, name, *arguments, engine=False):
    def call():
        handle_type = importlib.import_module(
            "solprop.backends.julia._ownership"
        ).JuliaHandle
        module = runtime.engine if engine else runtime.namespace
        function = getattr(module, name)
        borrowed = tuple(
            argument.borrow() if isinstance(argument, handle_type) else argument
            for argument in arguments
        )
        result = function._jl_call_nogil(*borrowed)
        assert type(result) in (type(None), bool, int, float, str)
        return result

    return importlib.import_module("solprop.backends.julia._bridge").invoke(
        call, f"runtime probe {name} failed"
    )


def _runtime():
    runtime = _runtime_module().get_runtime(
        setup_id=os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]
    )
    requested = int(os.environ["SOLPROP_THREADS"])
    assert runtime.threads == requested, (runtime.threads, requested)
    assert int(_evaluate(runtime, "Threads.nthreads(:default)")) == requested
    assert int(_evaluate(runtime, "Threads.nthreads(:interactive)")) == 0
    return runtime


def _stop(runtime):
    token_type = importlib.import_module("solprop.backends.session").StopToken
    lease_type = importlib.import_module("solprop.backends.julia._stop").StopLease
    token = token_type()
    return token, lease_type(runtime, token)


def _workload(runtime):
    _evaluate(runtime, _WORKLOAD)
    _, lease = _stop(runtime)
    _call(runtime, "probe_work", lease.flag, 1, "")
    lease.close()


def _wait(predicate, seconds=15.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "probe synchronization timed out"
        time.sleep(0.002)


def _join(worker, failures):
    worker.join(timeout=60)
    assert not worker.is_alive(), "foreign Python thread did not return"
    assert not failures, failures


def _worker(action):
    failures = []

    def run():
        try:
            action()
        except BaseException as error:
            failures.append(repr(error))

    return threading.Thread(target=run), failures


def _record(output, case, **values):
    content = {
        "case": case,
        "pid": os.getpid(),
        "platform": sys.platform,
        "requested_threads": int(os.environ["SOLPROP_THREADS"]),
        **values,
    }
    output.write_text(json.dumps(content, sort_keys=True) + "\n", encoding="utf-8")


def _worker_first(output):
    module = _runtime_module()
    refused = []

    def first():
        try:
            module.get_runtime(setup_id=os.environ["SOLPROP_TEST_JULIA_SETUP_ID"])
        except importlib.import_module("solprop.exceptions").BackendError as error:
            refused.append(str(error))

    worker, failures = _worker(first)
    worker.start()
    _join(worker, failures)
    assert len(refused) == 1 and "main thread" in refused[0], refused
    assert "juliacall" not in sys.modules
    assert not module._STATE.attempted
    runtime = _runtime()
    _record(output, "worker_first", actual_threads=runtime.threads, refused=refused)


def _startup_race(output):
    module = _runtime_module()
    arrived = [threading.Event(), threading.Event()]
    refused = []
    workers = []
    failures = []
    armed = [True]

    def caller(index):
        arrived[index].set()
        try:
            module.get_runtime(setup_id=os.environ["SOLPROP_TEST_JULIA_SETUP_ID"])
        except importlib.import_module("solprop.exceptions").BackendError as error:
            assert "main thread" in str(error)
            refused.append(index)

    def audit(event, _arguments):
        # The pinned bridge locates libjulia in a real subprocess. Start competing
        # callers at that boundary while main-thread initialization owns its lock.
        if event == "subprocess.Popen" and armed[0]:
            armed[0] = False
            for index in range(2):
                worker, errors = _worker(lambda index=index: caller(index))
                workers.append(worker)
                failures.append(errors)
                worker.start()
            assert all(ready.wait(timeout=5) for ready in arrived)
            # Main still owns startup's lock. A worker must refuse before waiting
            # on it, so none of these joins depends on initialization completing.
            for worker, errors in zip(workers, failures, strict=True):
                worker.join(timeout=5)
                assert not worker.is_alive(), "worker waited for Julia startup"
                assert not errors, errors

    sys.addaudithook(audit)
    runtime = _runtime()
    assert not armed[0], "the real JuliaCall startup boundary was not observed"
    for worker, errors in zip(workers, failures, strict=True):
        _join(worker, errors)
    assert sorted(refused) == [0, 1]
    _record(
        output,
        "startup_race",
        actual_threads=runtime.threads,
        refused_before_initialization_completed=len(refused),
    )


def _cancellation(output):
    runtime = _runtime()
    _workload(runtime)
    before = int(_evaluate(runtime, "Threads.maxthreadid()"))
    token, lease = _stop(runtime)
    ready = output.with_suffix(".entered")
    ticks = [0]
    python_gc_rounds = [0]

    def request():
        _wait(ready.exists)
        deadline = time.monotonic() + 0.03
        while time.monotonic() < deadline:
            for _ in range(128):
                ticks[0] += 1
            gc.collect()
            python_gc_rounds[0] += 1
        token.set()

    worker, failures = _worker(request)
    worker.start()
    maximum_batches = 100_000
    completed = int(
        _call(runtime, "probe_work", lease.flag, maximum_batches, str(ready))
    )
    _join(worker, failures)
    assert 0 < completed < maximum_batches, completed
    assert ticks[0] > 0, "Python did not progress during the GIL-free Julia call"
    assert python_gc_rounds[0] > 0
    assert _call(runtime, "stop_requested", lease.flag, engine=True)
    mask = int(_evaluate(runtime, "probe_thread_mask[]"))
    assert mask.bit_count() == runtime.threads, (mask, runtime.threads)
    assert int(_evaluate(runtime, "Threads.maxthreadid()")) == before
    lease.close()
    already_set = importlib.import_module("solprop.backends.julia._stop").StopLease(
        runtime, token
    )
    assert _call(runtime, "stop_requested", already_set.flag, engine=True)
    already_set.close()
    _record(
        output,
        "cancellation",
        actual_threads=runtime.threads,
        batches=completed,
        gil_ticks=ticks[0],
        python_gc_rounds=python_gc_rounds[0],
        participating_threads=mask.bit_count(),
        foreign_thread_adopted=False,
    )


def _last_owner(output):
    runtime = _runtime()
    _workload(runtime)
    before = int(_evaluate(runtime, "Threads.maxthreadid()"))
    token, lease = _stop(runtime)
    lease.close()
    entered = threading.Event()
    gc_ready = output.with_suffix(".gc-entered")

    def gate():
        entered.set()
        _wait(gc_ready.exists)

    detach_gate = token.attach(gate)
    lease = importlib.import_module("solprop.backends.julia._stop").StopLease(
        runtime, token
    )
    owner = weakref.ref(lease._listener.storage)
    reclaimed = []
    finalizer = weakref.finalize(
        lease._listener.storage, lambda: reclaimed.append(threading.get_ident())
    )
    worker, failures = _worker(token.set)
    worker.start()
    assert entered.wait(timeout=10)
    lease.close()
    del lease
    gc.collect()
    assert owner() is not None and finalizer.alive
    _, live = _stop(runtime)
    _call(runtime, "probe_window", live.flag, 0.15, str(gc_ready))
    _join(worker, failures)
    live.close()
    detach_gate()
    gc.collect()
    assert owner() is None and not finalizer.alive
    assert reclaimed == [worker.ident], reclaimed
    assert int(_evaluate(runtime, "Threads.maxthreadid()")) == before
    _record(
        output,
        "last_owner",
        actual_threads=runtime.threads,
        reclaimed_on_stopping_thread=True,
        foreign_thread_adopted=False,
    )


def _refused_operations(runtime, lease, handle):
    token_type = importlib.import_module("solprop.backends.session").StopToken
    lease_type = importlib.import_module("solprop.backends.julia._stop").StopLease
    return {
        "runtime selection": lambda: _runtime_module().get_runtime(
            setup_id=runtime.setup.setup_id
        ),
        "runtime check": runtime.check,
        "namespace access": lambda: runtime.namespace,
        "engine access": lambda: runtime.engine,
        "owned construction": lambda: runtime.own_call("StopFlag", 0),
        "owner retention": lambda: runtime.owners.retain(None),
        "owner drain": runtime.owners.drain,
        "owner shutdown": runtime.owners.close,
        "handle borrow": handle.borrow,
        "handle close": handle.close,
        "stop construction": lambda: lease_type(runtime, token_type()),
        "stop flag": lambda: lease.flag,
        "stop close": lease.close,
    }


def _require_refusals(operations):
    error_type = importlib.import_module("solprop.exceptions").BackendError
    refused = []
    for name, operation in operations.items():
        try:
            operation()
        except error_type as error:
            assert "main thread" in str(error), (name, str(error))
            refused.append(name)
        else:
            raise AssertionError(f"foreign thread unexpectedly entered {name}")
    return refused


def _worker_refusals(output, *, during_gc=False):
    runtime = _runtime()
    _workload(runtime)
    token, lease = _stop(runtime)
    handle = lease.flag
    operations = _refused_operations(runtime, lease, handle)
    before = int(_evaluate(runtime, "Threads.maxthreadid()"))
    ready = output.with_suffix(".refusal-gc")
    refused = []

    def attempt():
        if during_gc:
            _wait(ready.exists)
        try:
            for _ in range(3):
                refused.extend(_require_refusals(operations))
        finally:
            token.set()

    worker, failures = _worker(attempt)
    worker.start()
    if during_gc:
        _call(runtime, "probe_work", handle, 100_000, str(ready))
    _join(worker, failures)
    assert refused == list(operations) * 3
    assert _call(runtime, "stop_requested", handle, engine=True)
    assert int(_evaluate(runtime, "Threads.maxthreadid()")) == before
    lease.close()
    _record(
        output,
        "worker_refusals_gc" if during_gc else "worker_refusals",
        actual_threads=runtime.threads,
        refused_operations=list(operations),
        refused_calls=len(refused),
        foreign_thread_adopted=False,
    )


def _worker_refusals_gc(output):
    _worker_refusals(output, during_gc=True)


def _collect_owner(runtime, marker):
    # PythonCall can retain a Python wrapper through a Julia Py object. Reclaiming
    # that wrapper and then its Julia payload needs successive bridge GC rounds.
    for _ in range(4):
        gc.collect()
        _evaluate(runtime, "GC.gc(true)")
        if marker.exists():
            return
    raise AssertionError("a released Julia owner survived four full GC rounds")


def _owner_drop(output):
    runtime = _runtime()
    _workload(runtime)
    marker = output.with_suffix(".finalized")
    handle = runtime.own_call("probe_owner", str(marker))
    holder = [handle]
    dropped = []
    observed = weakref.ref(handle)
    finalizer = weakref.finalize(handle, lambda: dropped.append(threading.get_ident()))
    del handle
    before = int(_evaluate(runtime, "Threads.maxthreadid()"))

    def discard():
        held = holder.pop()
        assert isinstance(repr(held), str)
        del held
        gc.collect()

    worker, failures = _worker(discard)
    worker.start()
    _join(worker, failures)
    assert observed() is None and not finalizer.alive
    assert dropped == [worker.ident]
    assert not marker.exists(), "foreign destruction released a retained Julia owner"
    runtime.owners.drain()
    _collect_owner(runtime, marker)
    assert marker.read_text(encoding="utf-8") == "finalized"
    assert int(_evaluate(runtime, "Threads.maxthreadid()")) == before
    _record(
        output,
        "owner_drop",
        actual_threads=runtime.threads,
        handle_reclaimed_on_worker=True,
        julia_owner_released_on_main=True,
        foreign_thread_adopted=False,
    )


def _sanitized_error(output):
    runtime = _runtime()
    _workload(runtime)
    bridge = importlib.import_module("solprop.backends.julia._bridge")
    error_type = importlib.import_module("solprop.exceptions").BackendError
    marker = output.with_suffix(".failed-owner-finalized")

    def failing_operation():
        # Both this wrapper and JuliaError's wrappers must be destroyed on main.
        temporary = runtime.engine.probe_owner(str(marker))
        assert temporary is not None
        runtime.namespace.seval('error("owned bridge probe failure")')
        raise AssertionError("the Julia error probe unexpectedly returned")

    caught = []
    try:
        bridge.invoke(failing_operation, "sanitized runtime probe")
    except error_type as failure:
        caught.append(failure)
    assert len(caught) == 1
    failure = caught[0]
    assert failure.__cause__ is None and failure.__context__ is None
    assert all(isinstance(argument, str) for argument in failure.args)
    del failure
    _collect_owner(runtime, marker)
    before = int(_evaluate(runtime, "Threads.maxthreadid()"))
    formatted = []

    def format_and_drop():
        failure = caught.pop()
        formatted.extend(
            [str(failure), repr(failure), "".join(traceback.format_exception(failure))]
        )
        del failure
        gc.collect()

    worker, failures = _worker(format_and_drop)
    worker.start()
    _join(worker, failures)
    assert len(formatted) == 3
    assert all("owned bridge probe failure" in text for text in formatted)
    assert int(_evaluate(runtime, "Threads.maxthreadid()")) == before
    _record(
        output,
        "sanitized_error",
        actual_threads=runtime.threads,
        formatted_and_released_on_worker=True,
        failed_helper_owner_released_on_main=True,
        foreign_thread_adopted=False,
    )


def _polyester(output):
    runtime = _runtime()
    _workload(runtime)
    mask = int(_call(runtime, "probe_polyester"))
    assert mask.bit_count() == runtime.threads
    _record(
        output,
        "polyester",
        actual_threads=runtime.threads,
        participating_threads=mask.bit_count(),
    )


def _signal_handler(mode, received):
    if mode == "ignored":
        return signal.SIG_IGN
    if mode == "default":
        return signal.default_int_handler

    def handler(number, _frame):
        received.append(number)

    return handler


def _sigint_delivery(runtime, output, mode, received, index, during):
    previous = len(received)
    interrupted = False
    lease = None
    worker = None
    failures = []
    try:
        if during:
            _, lease = _stop(runtime)
            ready = output.with_suffix(f".signal-{index}")

            def send():
                _wait(ready.exists)
                os.kill(os.getpid(), signal.SIGINT)

            worker, failures = _worker(send)
            worker.start()
            _call(runtime, "probe_window", lease.flag, 0.2, str(ready))
        else:
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if worker is not None:
            _join(worker, failures)
        if lease is not None:
            lease.close()
    if mode == "custom":
        _wait(lambda: len(received) > previous)
    assert interrupted == (mode == "default"), (mode, interrupted)
    assert len(received) - previous == int(mode == "custom"), received


def _sigint(output, mode):
    received = []
    handler = _signal_handler(mode, received)
    signal.signal(signal.SIGINT, handler)
    runtime = _runtime()
    assert signal.getsignal(signal.SIGINT) is handler
    _workload(runtime)
    _sigint_delivery(runtime, output, mode, received, 0, False)
    _sigint_delivery(runtime, output, mode, received, 1, True)
    # Real handler changes after Julia initialization must also take effect.
    for index, changed in enumerate(("ignored", "custom"), start=2):
        signal.signal(signal.SIGINT, _signal_handler(changed, received))
        _sigint_delivery(runtime, output, changed, received, index, True)
    assert _evaluate(runtime, "1 + 1") == 2
    time.sleep(0.05)
    assert len(received) == (3 if mode == "custom" else 1), received
    _record(
        output,
        "sigint",
        actual_threads=runtime.threads,
        initial_handler=mode,
        deliveries=4,
        custom_callbacks=len(received),
    )


def _sigterm(output, mode):
    mode, phase = mode.split(":")
    received = []
    handler = signal.SIG_DFL if mode == "default" else _signal_handler(mode, received)
    signal.signal(signal.SIGTERM, handler)
    runtime = _runtime()
    _record(
        output,
        "sigterm",
        actual_threads=runtime.threads,
        initial_handler=mode,
        phase=phase,
        stage="ready-for-accepted-termination",
    )
    if phase == "idle":
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)
    else:
        assert phase == "work"
        _workload(runtime)
        _, lease = _stop(runtime)
        ready = output.with_suffix(".term-entered")

        def terminate():
            _wait(ready.exists)
            os.kill(os.getpid(), signal.SIGTERM)

        worker, _ = _worker(terminate)
        worker.start()
        _call(runtime, "probe_window", lease.flag, 2.0, str(ready))
    raise AssertionError("the pinned Julia SIGTERM behavior unexpectedly changed")


def _preflight(output, mode):
    setup_module = importlib.import_module("solprop.backends.julia._setup")
    error_type = importlib.import_module("solprop.exceptions").BackendError
    attempted = []

    def audit(event, _arguments):
        if event.startswith(("socket.", "urllib.")) or event == "subprocess.Popen":
            attempted.append(event)
            raise AssertionError(f"preflight attempted non-filesystem work: {event}")

    sys.addaudithook(audit)
    refusal = None
    try:
        setup_module.select_setup(setup_id=os.environ["SOLPROP_TEST_JULIA_SETUP_ID"])
    except error_type as error:
        refusal = str(error)
    assert not attempted
    assert "juliacall" not in sys.modules
    assert (refusal is not None) == (mode == "incomplete"), refusal
    if refusal is not None:
        assert "missing" in refusal or "does not exist" in refusal, refusal
    _record(output, "preflight", mode=mode, network_calls=0, subprocess_calls=0)


def _devices(output):
    runtime = _runtime()
    installed = set(dict(runtime.setup.projects)) - {"cpu"}
    modules = {"cuda": "CUDA", "rocm": "AMDGPU", "metal": "Metal"}
    error_type = importlib.import_module("solprop.exceptions").BackendError
    loaded = []
    refused = []
    worker_refused = []
    for name, module in modules.items():
        assert not _evaluate(
            runtime, f'any(key.name == "{module}" for key in keys(Base.loaded_modules))'
        )

        def worker_load(name=name):
            try:
                _runtime_module().get_runtime(
                    setup_id=runtime.setup.setup_id, device=name
                )
            except error_type as error:
                assert "main thread" in str(error), str(error)
                worker_refused.append(name)
            else:
                raise AssertionError("a Python worker loaded a Julia device")

        worker, failures = _worker(worker_load)
        worker.start()
        _join(worker, failures)
        assert not _evaluate(
            runtime, f'any(key.name == "{module}" for key in keys(Base.loaded_modules))'
        )
        try:
            selected = _runtime_module().get_runtime(
                setup_id=runtime.setup.setup_id, device=name
            )
        except error_type as error:
            assert name not in installed
            assert f"no {name} environment" in str(error), str(error)
            refused.append(name)
        else:
            assert name in installed and selected is runtime
            loaded.append(name)
        present = bool(
            _evaluate(
                runtime,
                f'any(key.name == "{module}" for key in keys(Base.loaded_modules))',
            )
        )
        assert present == (name in installed)
        if present:
            extension = f"SolpropJulia{module}Ext"
            assert _evaluate(
                runtime, f"Base.get_extension(SolpropJulia, :{extension}) !== nothing"
            )
    assert set(loaded) == installed and set(refused) == set(modules) - installed
    _record(
        output,
        "devices",
        actual_threads=runtime.threads,
        loaded_devices=loaded,
        refused_devices=refused,
        worker_refused_devices=worker_refused,
    )


def _spawned_worker(output):
    runtime = _runtime()
    _workload(runtime)
    assert _call(runtime, "probe_fft") == 64
    _record(Path(output), "spawn_child", actual_threads=runtime.threads)


def _spawn(output):
    runtime = _runtime()
    child_output = output.with_suffix(".child.json")
    process = multiprocessing.get_context("spawn").Process(
        target=_spawned_worker, args=(str(child_output),)
    )
    process.start()
    process.join(timeout=90)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)
        raise AssertionError("spawned Julia worker timed out")
    assert process.exitcode == 0, process.exitcode
    child = json.loads(child_output.read_text(encoding="utf-8"))
    assert child["pid"] != os.getpid()
    assert child["actual_threads"] == runtime.threads
    assert _runtime() is runtime
    _record(output, "spawn", actual_threads=runtime.threads, child=child)


def _shutdown(output):
    runtime = _runtime()
    _workload(runtime)
    assert _call(runtime, "probe_fft") == 64
    token, lease = _stop(runtime)
    token.set()
    assert _call(runtime, "stop_requested", lease.flag, engine=True)
    released = output.with_suffix(".owner-released")
    finalized = output.with_suffix(".julia-finalized")

    def release_marker():
        released.write_text("released", encoding="utf-8")

    handle = runtime.own_call(
        "probe_owner",
        str(finalized),
        on_release=release_marker,
    )
    # This handle remains live in a module global. Registry shutdown must release
    # it even though its finalizer never queued an abandoned identity.
    _LIVE_EXIT_HANDLES.append(handle)

    def closing():
        assert _call(runtime, "stop_requested", lease.flag, engine=True)
        assert not released.exists() and handle._finalizer.alive
        lease.close()
        _record(
            output,
            "shutdown",
            actual_threads=runtime.threads,
            atexit_ran=True,
            unclosed_live_handle=True,
        )

    # Keep the FFT plan and stop owner alive into Python finalization. The caller
    # checks the actual process exit, after Julia's own shutdown has also run.
    atexit.register(closing)


def _native(output, mode):
    assert "solprop_native" not in sys.modules
    assert "juliacall" not in sys.modules
    native = importlib.import_module("solprop_native") if mode == "before" else None
    assert "juliacall" not in sys.modules
    runtime = _runtime()
    assert "juliacall" in sys.modules
    assert ("solprop_native" in sys.modules) == (mode == "before")
    if native is None:
        native = importlib.import_module("solprop_native")
    native.check_build()
    numpy = importlib.import_module("numpy")
    core = importlib.import_module("solprop_native._core")
    volume = numpy.ones((4, 4, 4), dtype=numpy.complex128)
    _workload(runtime)
    for _ in range(3):
        timings = core.reference_fft3_pairs(volume, 1, 2)
        assert len(timings) == 2 and all(value > 0.0 for value in timings)
        assert _call(runtime, "probe_fft") == 64
    assert native.build_info().runtime_threads == runtime.threads
    _record(
        output,
        "native",
        actual_threads=runtime.threads,
        import_order=mode,
        native_threads=native.build_info().runtime_threads,
        fft_rounds=3,
    )


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(120)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", default="")
    arguments = parser.parse_args()
    cases = {
        "worker_first": _worker_first,
        "startup_race": _startup_race,
        "cancellation": _cancellation,
        "last_owner": _last_owner,
        "worker_refusals": _worker_refusals,
        "worker_refusals_gc": _worker_refusals_gc,
        "owner_drop": _owner_drop,
        "sanitized_error": _sanitized_error,
        "polyester": _polyester,
        "devices": _devices,
        "spawn": _spawn,
        "shutdown": _shutdown,
    }
    with_modes = {
        "sigint": _sigint,
        "sigterm": _sigterm,
        "preflight": _preflight,
        "native": _native,
    }
    try:
        if arguments.case in with_modes:
            with_modes[arguments.case](arguments.output, arguments.mode)
        else:
            cases[arguments.case](arguments.output)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
