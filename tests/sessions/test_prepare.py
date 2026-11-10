# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A backend chooses its own session through its preparation (SEP 47 section 5.2).

``start_session`` names no session: it starts ``backend.prepare(problem,
request).start(start, ...)``. The reference prepares its own session, a native backend
the native session where it admits the run and the transitional one, with the reason,
where it does not, and any other driver the transitional one.
"""

from typing import Any, cast

import pytest

from scripted_driver import ScriptedDriver
from sessions.executors import fibre, soliton, window
from solprop.backends.driver_session import DriverPrepared, DriverSession
from solprop.backends.host.session import HostSession
from solprop.backends.protocol import Backend
from solprop.backends.registry import available, select
from solprop.backends.session import (
    PreparedExecution,
    SessionError,
    Start,
    check_group,
)
from solprop.backends.sessions import start_session
from solprop.exceptions import BackendError

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
    from solprop_native.session import (  # noqa: PLC0415 - optional  # pyright: ignore[reportMissingImports]
        NativePrepared,
        NativeSession,
    )

    problem, request = soliton()
    backend = _select("native", problem, request)
    try:
        prepared = backend.prepare(problem, request)
        with prepared.start(Start.entrance(problem), owns=False) as session:
            assert isinstance(session, NativeSession)
        assert isinstance(prepared, NativePrepared)
        assert cast("Any", prepared).refused is None
    finally:
        backend.close()


@NATIVE
def test_a_native_refusal_prepares_the_transitional_session_and_says_why(
    monkeypatch,
):
    from solprop_native.session import (  # noqa: PLC0415 - optional  # pyright: ignore[reportMissingImports]
        NativePrepared,
    )

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
        assert isinstance(prepared, NativePrepared)
        assert cast("Any", prepared).refused == "the test's reason"
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
    # The script is the march alone (no memory ledger); the refusal comes first.
    backend = cast("Backend", driver)
    with pytest.raises(SessionError, match="integration is not the backend's own"):
        start_session(backend, problem, other, Start.entrance(problem), owns=False)


@NATIVE
def test_a_closed_native_preparation_refuses_to_start():
    problem, request = soliton()
    backend = _select("native", problem, request)
    prepared = backend.prepare(problem, request)
    prepared.close()
    with pytest.raises(SessionError, match="closed"):
        prepared.start(Start.entrance(problem))


# --- what the problem does not carry, refused by name before any session -----------

UNAVAILABLE = (
    (("energy", "plasma_undershoot"), "read the field-grid plasma column"),
    (("electron_density",), "need a compiled plasma response"),
    (("energy", "beam_radius"), "'beam_radius' statistic requires spatial geometry"),
)
"""Statistics a Kerr-only fibre does not carry, and the reference's own refusal of
each at its first plane (SEP 47 section 6.3)."""


def _executors():
    """Each executor's preparation, by name: the reference, the native session, the
    transitional session over the native backend, and a scripted driver."""
    executors = ["numpy_ref", "scripted"]
    if "native" in available():
        executors += ["native", "native-bridge"]
    return executors


@pytest.mark.parametrize("executor", _executors())
@pytest.mark.parametrize(("names", "match"), UNAVAILABLE)
def test_what_a_problem_does_not_carry_is_refused_on_every_executor(
    monkeypatch, executor, names, match
):
    """Refused where the run is prepared, before any session or workspace of the run
    exists, and never left to the transitional session to refuse at the first plane."""
    from solprop.stats import StatsSpec  # noqa: PLC0415

    problem, request = soliton()
    if executor == "native-bridge":
        monkeypatch.setattr(
            "solprop_native.session.session_refusal", lambda *_: "the test's bridge"
        )
    backend = (
        ScriptedDriver(problem, request, grid=window())
        if executor == "scripted"
        else _select(executor.removesuffix("-bridge"), problem, request)
    )
    run = request.for_run(stats=StatsSpec(names=names))
    try:
        with pytest.raises(BackendError, match=match) as refused:
            backend.prepare(problem, run)
        assert not isinstance(refused.value, SessionError)
        with pytest.raises(BackendError, match=match):
            start_session(
                backend,  # pyright: ignore[reportArgumentType] - a scripted driver
                problem,
                run,
                Start.entrance(problem),
                owns=False,
            )
    finally:
        backend.close()


@pytest.mark.parametrize(("names", "match"), UNAVAILABLE)
def test_the_run_layer_refuses_it_before_a_session_starts(names, match):
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.stats import StatsSpec  # noqa: PLC0415

    for backend in ("numpy_ref", *(("native",) if "native" in available() else ())):
        simulation = Simulation.from_gnlse(
            0.11,
            0.01,
            [0.0, 0.0, -1.183e-26],
            wavelength=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            width=100e-15,
            peak_power=9.8,
            shape="sech",
            stats=StatsSpec(names=names),
            backend=backend,
        )
        with pytest.raises(BackendError, match=match):
            simulation.run()


def test_a_group_refuses_it_for_any_member():
    from solprop.stats import StatsSpec  # noqa: PLC0415

    problem, request = soliton()
    run = request.for_run(stats=StatsSpec(names=("plasma_work_defect",)))
    entrance = Start.entrance(problem)
    with pytest.raises(BackendError, match="field-grid plasma column"):
        check_group(2, [problem, problem], [request, run], [entrance, entrance])


def test_a_spatial_route_refuses_a_modal_statistic_on_every_executor():
    from backends.test_native_widths import _simulation  # noqa: PLC0415
    from solprop.grid import FreeGrid  # noqa: PLC0415
    from solprop.stats import StatsSpec  # noqa: PLC0415

    grid, problem, request = _simulation("envelope", FreeGrid(3e-4, 16, 4e-4, 12), 1)
    run = request.for_run(stats=StatsSpec(names=("energy", "fwhm_t", "omega0")))
    for name in ("numpy_ref", *(("native",) if "native" in available() else ())):
        backend = select(name, problem, request, grid=grid)
        try:
            with pytest.raises(
                BackendError,
                match=r"statistics \['fwhm_t', 'omega0'\] are not spatial saved-field",
            ):
                backend.prepare(problem, run)
        finally:
            backend.close()
