# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep wrapper destruction and failure formatting on the Julia owner thread."""

import gc
import threading
import traceback
import weakref

import pytest

from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._ownership import OwnerRegistry
from solprop.backends.session import SessionError
from solprop.exceptions import BackendError


def _worker(operation):
    failures = []

    def run():
        try:
            operation()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    return failures


def _tracked(events):
    class Wrapper:
        def __repr__(self):
            pytest.fail("representing a handle must not represent its Julia value")

        def __del__(self):
            events.append(("destroy", threading.get_ident()))

    return Wrapper()


def test_foreign_last_reference_only_queues_owner_release():
    registry = OwnerRegistry()
    events = []
    handles = [
        registry.retain(
            _tracked(events),
            on_release=lambda: events.append(("detach", threading.get_ident())),
        )
    ]
    wrapper = weakref.ref(handles[0].borrow())
    assert not handles[0]._finalizer.atexit
    assert _worker(lambda: (handles.clear(), gc.collect())) == []
    assert wrapper() is not None and events == []
    registry.drain()
    assert wrapper() is None
    assert events == [
        ("detach", threading.get_ident()),
        ("destroy", threading.get_ident()),
    ]
    registry.close()


def test_runtime_close_releases_live_and_abandoned_handles():
    registry = OwnerRegistry()
    events = []
    live = registry.retain(_tracked(events))
    abandoned = registry.retain(_tracked(events))
    del abandoned
    registry.close()
    registry.close()
    live.close()
    assert events == [("destroy", threading.get_ident())] * 2
    with pytest.raises(SessionError, match="owners are closed"):
        live.borrow()
    with pytest.raises(SessionError, match="owners are closed"):
        registry.retain(1)


def test_every_wrapper_operation_refuses_foreign_threads():
    registry = OwnerRegistry()
    events = []
    handle = registry.retain(_tracked(events))
    for operation in (
        handle.borrow,
        handle.close,
        registry.drain,
        registry.close,
        lambda: registry.retain(1),
    ):
        errors = _worker(operation)
        assert len(errors) == 1
        assert isinstance(errors[0], BackendError)
        assert "main thread" in str(errors[0])
    assert events == []
    alias = handle
    assert _worker(lambda: (repr(handle), repr(registry), handle == alias)) == []
    handle.close()
    handle.close()
    assert events == [("destroy", threading.get_ident())]
    registry.close()


def test_borrowed_buffer_survives_abandonment_until_owner_drain():
    registry = OwnerRegistry()
    events = []
    buffer = _tracked(events)
    reference = weakref.ref(buffer)
    handles = [registry.retain(1, owners=(buffer,))]
    del buffer
    assert _worker(handles.clear) == []
    assert reference() is not None
    registry.drain()
    assert reference() is None
    assert events == [("destroy", threading.get_ident())]
    registry.close()


@pytest.mark.parametrize("release", ["close", "drain"])
def test_owner_finalizers_cannot_reenter_bridge_lifecycle(release):
    registry = OwnerRegistry()
    refused = []

    class Reentering:
        def __del__(self):
            for operation in (registry.drain, registry.close):
                try:
                    operation()
                except SessionError as error:
                    refused.append(str(error))

    handle = registry.retain(Reentering())
    if release == "close":
        handle.close()
    else:
        del handle
        registry.drain()
    assert len(refused) == 2
    assert all("during owner release" in reason for reason in refused)
    registry.close()


def test_owner_identity_cannot_be_reused_after_close():
    registry = OwnerRegistry()
    first = registry.retain(1)
    first.close()
    second = registry.retain(2)
    assert first._key != second._key
    with pytest.raises(SessionError, match="handle is closed"):
        first.borrow()
    assert second.borrow() == 2
    registry.close()


def test_forked_owner_refuses_access_and_never_releases_parent(monkeypatch):
    registry = OwnerRegistry()
    events = []
    handle = registry.retain(_tracked(events))
    with monkeypatch.context() as child:
        child.setattr(registry, "_pid", -1)
        with pytest.raises(BackendError, match="after fork"):
            handle.borrow()
        registry.close()
        assert events == []
    registry.close()
    assert events == [("destroy", threading.get_ident())]


@pytest.mark.parametrize("broken_format", [False, True])
def test_failure_has_no_foreign_wrappers_in_its_context_or_traceback(broken_format):
    events = []

    class ForeignError(Exception):
        def __str__(self):
            assert threading.current_thread() is threading.main_thread()
            events.append(("format", threading.get_ident()))
            if broken_format:
                raise ValueError("formatter failed")
            return "engine failure"

    def call():
        wrapper = _tracked(events)
        raise ForeignError(wrapper)

    try:
        invoke(call, "advance failed")
    except BackendError as error:
        caught = error
    else:
        pytest.fail("the bridge failure was swallowed")
    assert caught.__cause__ is None and caught.__context__ is None
    assert events == [
        ("format", threading.get_ident()),
        ("destroy", threading.get_ident()),
    ]
    assert _worker(lambda: traceback.format_exception(caught)) == []
    expected = "message could not be formatted" if broken_format else "engine failure"
    assert expected in str(caught)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_control_exceptions_discard_foreign_tracebacks(exception):
    events = []

    def call():
        wrapper = _tracked(events)
        assert wrapper is not None
        raise exception()

    with pytest.raises(exception) as captured:
        invoke(call, "control failure")
    assert captured.value.__context__ is None
    assert events == [("destroy", threading.get_ident())]


def test_bridge_refuses_before_calling_a_foreign_operation():
    entered = []
    errors = _worker(lambda: invoke(lambda: entered.append(True), "unused"))
    assert len(errors) == 1 and isinstance(errors[0], BackendError)
    assert entered == []
    assert invoke(lambda: 42, "unused") == 42


def test_recursive_bridge_call_refuses_without_poisoning_later_calls():
    with pytest.raises(BackendError, match="recursive julia bridge entry"):
        invoke(lambda: invoke(lambda: 1, "nested"), "outer")
    assert invoke(lambda: 42, "unused") == 42


@pytest.mark.parametrize("exception", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("shutdown", [False, True])
def test_failed_cleanup_releases_wrappers_before_failure_can_leave_main(
    exception, shutdown
):
    registry = OwnerRegistry()
    events = []

    def fail():
        raise exception("cleanup interrupted")

    first = registry.retain(_tracked(events), on_release=fail)
    second = registry.retain(_tracked(events))
    expected = KeyboardInterrupt if exception is KeyboardInterrupt else BackendError
    with pytest.raises(expected) as captured:
        if shutdown:
            registry.close()
        else:
            first.close()
    assert events == [("destroy", threading.get_ident())] * (2 if shutdown else 1)
    assert captured.value.__context__ is None
    assert _worker(lambda: traceback.format_exception(captured.value)) == []
    registry.close()
    second.close()
    assert events == [("destroy", threading.get_ident())] * 2


def test_system_exit_with_failing_object_formatter_is_detached_on_main():
    events = []

    class Code:
        def __str__(self):
            raise ValueError("format failed")

        def __del__(self):
            events.append(threading.get_ident())

    def exit_with_owner():
        raise SystemExit(Code())

    with pytest.raises(SystemExit) as captured:
        invoke(exit_with_owner, "exit failed")
    assert captured.value.code == "exception message could not be formatted"
    assert captured.value.__context__ is None
    assert events == [threading.get_ident()]


@pytest.mark.parametrize("grouped", [False, True])
def test_nested_exception_frames_are_cleared_before_foreign_python_gc(grouped):
    events = []

    def inner():
        wrapper = _tracked(events)
        error = ValueError(wrapper)
        # A context cycle is valid Python exception state; its identity guard must
        # terminate and all wrappers must still be reclaimed on main.
        error.__context__ = error
        raise error

    def outer():
        try:
            inner()
        except ValueError as error:
            if grouped:
                raise ExceptionGroup("group", [error]) from None
            raise RuntimeError("outer") from error

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(BackendError):
            invoke(outer, "nested failure")
        assert events == [("destroy", threading.get_ident())]
        assert _worker(gc.collect) == []
        assert events == [("destroy", threading.get_ident())]
    finally:
        if was_enabled:
            gc.enable()
