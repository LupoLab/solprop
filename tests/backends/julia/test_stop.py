# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check ownership through detach races independently of Julia runtime startup."""

import ctypes
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest

from solprop.backends.julia import _runtime
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._stop import StopLease
from solprop.backends.session import SessionError, StopToken
from solprop.exceptions import BackendError


@pytest.fixture
def stop_runtime(setup_record):
    destroyed_on = []
    writes = []

    class Flag:
        def __hash__(self):
            raise AssertionError("detachment must not hash a foreign flag")

        def __init__(self, address):
            self.address = address

        @property
        def value(self):
            return ctypes.c_uint8.from_address(self.address).value

        def __eq__(self, other):
            pytest.fail("detachment must not inspect a foreign flag")

        def __del__(self):
            destroyed_on.append(threading.get_ident())

    def store(address):
        ctypes.cast(address, ctypes.POINTER(ctypes.c_uint8))[0] = 1
        writes.append(address)

    # This callback tests Python ownership only. Packaged-bridge subprocess probes
    # separately test the real raw LLVM leaf, threaded GC and GIL-free progress.
    setter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(store)
    engine = SimpleNamespace(
        StopFlag=Flag,
        stop_setter_address=lambda: ctypes.cast(setter, ctypes.c_void_p).value,
    )
    runtime = JuliaRuntime(setup_record, 2, None, engine)
    return runtime, destroyed_on, writes


def test_already_set_token_reaches_new_flag(stop_runtime):
    runtime, destroyed_on, writes = stop_runtime
    token = StopToken()
    token.set()
    lease = StopLease(runtime, token)
    assert lease.flag.value == 1
    assert len(writes) == 1
    lease.close()
    lease.close()
    gc.collect()
    assert destroyed_on == [threading.get_ident()]
    with pytest.raises(SessionError, match="closed"):
        _ = lease.flag


def test_close_retains_owner_until_captured_listener_finishes(stop_runtime):
    runtime, destroyed_on, writes = stop_runtime
    token = StopToken()
    entered = threading.Event()
    proceed = threading.Event()

    def wait():
        entered.set()
        assert proceed.wait(timeout=5)

    detach_gate = token.attach(wait)
    lease = StopLease(runtime, token)
    assert lease._listener is not None
    owner = weakref.ref(lease._listener.storage)
    worker = threading.Thread(target=token.set)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        lease.close()
        gc.collect()
        assert owner() is not None
        assert destroyed_on == [threading.get_ident()]
        assert writes == []
    finally:
        proceed.set()
        worker.join(timeout=5)
        detach_gate()
    assert not worker.is_alive()
    assert len(writes) == 1
    gc.collect()
    assert owner() is None
    assert destroyed_on == [threading.get_ident()]


def test_detached_listener_receives_no_new_requests(stop_runtime):
    runtime, _, writes = stop_runtime
    token = StopToken()
    lease = StopLease(runtime, token)
    lease.close()
    token.set()
    assert writes == []


def test_token_clear_does_not_reset_an_existing_session_flag(stop_runtime):
    runtime, _, _ = stop_runtime
    token = StopToken()
    first = StopLease(runtime, token)
    token.set()
    token.clear()
    second = StopLease(runtime, token)
    assert first.flag.value == 1
    assert second.flag.value == 0
    first.close()
    second.close()


def test_detaching_one_of_two_listeners_never_compares_foreign_flags(stop_runtime):
    runtime, _, writes = stop_runtime
    token = StopToken()
    first = StopLease(runtime, token)
    second = StopLease(runtime, token)
    second.close()
    token.set()
    assert first.flag.value == 1
    assert len(writes) == 1
    first.close()


@pytest.mark.parametrize("forked", [False, True])
def test_unusable_runtime_refuses_before_flag_allocation(
    monkeypatch, stop_runtime, forked
):
    runtime, destroyed_on, _ = stop_runtime
    state = _runtime._State()
    if forked:
        state.pid = -1
    else:
        state.failure = RuntimeError("a device failed")
    monkeypatch.setattr(_runtime, "_STATE", state)
    with pytest.raises(BackendError, match="fresh process"):
        StopLease(runtime, StopToken())
    assert destroyed_on == []
