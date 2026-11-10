# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A backend chooses its own session through its preparation (SEP 47 section 5.2).

``start_session`` names no session: it starts ``backend.prepare(problem,
request).start(start, ...)``. The reference prepares its own session, a native backend
the native session where it admits the run and the transitional one, with the reason,
where it does not, and any other driver the transitional one.
"""

import pytest

from scripted_driver import ScriptedDriver
from sessions.executors import fibre, soliton, window
from solprop.backends.driver_session import DriverPrepared, DriverSession
from solprop.backends.host.session import HostSession
from solprop.backends.registry import available, select
from solprop.backends.session import PreparedExecution, SessionError, Start
from solprop.backends.sessions import start_session

NATIVE = pytest.mark.skipif(
    "native" not in available(), reason="solprop-native is not installed"
)


def _select(name, problem, request):
    return select(name, problem, request, grid=window(), modes=[fibre().mode])


def test_the_reference_prepares_its_own_session():
    problem, request = soliton()
    backend = _select("numpy_ref", problem, request)
    try:
        prepared = backend.prepare(problem, request)
        assert isinstance(prepared, PreparedExecution)
        assert prepared.backend == "numpy_ref"
        with start_session(
            backend, problem, request, Start.entrance(problem), owns=False
        ) as session:
            assert isinstance(session, HostSession)
    finally:
        backend.close()


def test_any_other_driver_prepares_the_transitional_session():
    problem, request = soliton()
    driver = ScriptedDriver(problem, request, grid=window())
    prepared = driver.prepare(problem, request)
    assert isinstance(prepared, DriverPrepared)
    with prepared.start(Start.entrance(problem), owns=False) as session:
        assert type(session) is DriverSession
    prepared.close()
    prepared.close()  # idempotent
    assert driver.closed
    with pytest.raises(SessionError, match="closed"):
        prepared.start(Start.entrance(problem))


@NATIVE
def test_a_native_backend_prepares_the_native_session_where_it_admits_the_run():
    from solprop_native.session import NativeSession  # noqa: PLC0415 - optional

    problem, request = soliton()
    backend = _select("native", problem, request)
    try:
        prepared = backend.prepare(problem, request)
        with prepared.start(Start.entrance(problem), owns=False) as session:
            assert isinstance(session, NativeSession)
        assert prepared.refused is None
    finally:
        backend.close()


@NATIVE
def test_a_native_refusal_prepares_the_transitional_session_and_says_why(
    monkeypatch,
):
    problem, request = soliton()
    backend = _select("native", problem, request)
    monkeypatch.setattr(
        "solprop_native.session.session_refusal", lambda *_: "the test's reason"
    )
    try:
        prepared = backend.prepare(problem, request)
        with start_session(
            backend, problem, request, Start.entrance(problem), owns=False
        ) as session:
            assert type(session) is DriverSession
        with prepared.start(Start.entrance(problem), owns=False):
            pass
        assert prepared.refused == "the test's reason"
    finally:
        backend.close()


def test_a_preparation_refuses_a_request_its_backend_was_not_built_for():
    """A backend lowered its method, error control and planes where it was built, so
    a run may change only what ``RunRequest.for_run`` changes."""
    problem, request = soliton()
    driver = ScriptedDriver(problem, request, grid=window())
    assert isinstance(
        driver.prepare(problem, request.for_run(start_plane=1)), DriverPrepared
    )
    other = request.with_stepper(request.stepper)
    with pytest.raises(SessionError, match="integration is not the backend's own"):
        driver.prepare(problem, other)
    with pytest.raises(SessionError, match="integration is not the backend's own"):
        start_session(driver, problem, other, Start.entrance(problem), owns=False)


@NATIVE
def test_a_closed_native_preparation_refuses_to_start():
    problem, request = soliton()
    backend = _select("native", problem, request)
    prepared = backend.prepare(problem, request)
    prepared.close()
    with pytest.raises(SessionError, match="closed"):
        prepared.start(Start.entrance(problem))
