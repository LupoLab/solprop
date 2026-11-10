# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's device matrix against the bridge, on a device.

SEP 47 phase 4 (section 6.3): on a CUDA or HIP build the native session serves what a
device build runs today -- Adams and the embedded Runge-Kutta tableaux, adaptive and
replayed, the low-storage replay, dense output, the radial, Cartesian, line and planar
spatial routes with their fluence widths, a free-space run's transverse edge and a
collar's current, the input and modified noise models, a coupled flow wider than 16
channels, and fixed-schedule groups (the native group session, one native session per
member) -- instead of leaving them to the transitional bridge
(:class:`~solprop.backends.driver_session.DriverSession` over the backend's own
march), which phase 5 deletes. A coupled flow with bins off its eigenbasis route,
which the device steps on solprop's host engine, is still the bridge's.

Each feature runs on the same device in the native session and on the bridge, most
through the production door (:func:`solprop.backends.sessions.start_session`) with
the bridge forced for the baseline. Both run the same strict marches over the same
device kernels and make their decisions on the host from the same device reductions,
so the control trace, the rows, the plane statistics and the exported fields agree
bitwise, as on a CPU build's one thread. The exceptions are the CPU suite's own, held
to its bounds: a Cartesian plane's widths (``std::hypot`` against Python's), and the
transverse record's ramp sums and a split floor's share (summed in order natively,
pairwise in solprop). The session crosses the field to the host only for an export.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from backends.test_transverse_backends import radial_collar
from sessions.executors import (
    CONSUMER,
    IDENTITY,
    STORED,
    Run,
    Started,
    assert_same_run,
    coupled_flow,
    drive,
    entrance,
    fibre,
    trace,
    window,
)
from sessions.test_group import PLANES as GROUP_PLANES
from sessions.test_group import member
from sessions.test_native_group import INTEGRATORS, _record, _same, _started
from sessions.test_native_session import (
    CASES,
    _case,
    _driver_session,
    _make_native,
)
from sessions.test_native_session_history import _noisy
from sessions.test_native_session_statistics import WIDTHS, _beam, _periodic
from sessions.test_native_session_transverse import (
    MARCHES,
    _both,
    _columns,
    _compare,
    _samples,
)
from solprop.backends.driver_session import DriverGroupSession, DriverSession
from solprop.backends.events import ObservationReady
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.exceptions import BoundaryWarning
from solprop.monitor import BoundaryCheck
from solprop.stats import StatsSpec
from test_transverse_reads import REREVIEW, packet_run

if TYPE_CHECKING:
    from solprop.backends.events import EventBatch
    from solprop.backends.export import ExportRecipe
    from solprop.backends.session import RunSession
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest

native_session = pytest.importorskip("solprop_native.session")
native_group = pytest.importorskip("solprop_native.group")

pytestmark = pytest.mark.gpu

DEVICE_CASES = tuple(sorted(CASES))
"""Every case of the CPU session's bitwise suite: adaptive and replayed, with
rejections, periodic rows on two periods and step planes, every temporal statistic,
the low-storage replay and dense output in both Adams formulations."""

ROUTES = ("radial", "plane", "line", "planar")
"""The spatial routes, each with its own fluence-width rule."""

NOISE_MARCHES = ("adams", "dense", "low-storage")
"""The marches the noise models run on: adaptive Adams, densely observed, replayed."""


@pytest.fixture(scope="module", autouse=True)
def _device() -> None:
    """Skip the module without a device build of the native backend."""
    native = pytest.importorskip("solprop_native")
    if native.build_info().device_backend == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")


def _forced(*_: object) -> str:
    """Refuse every run, so that the production door serves the bridge."""
    return "the bridge, forced by the test"


class _PhaseTaker:
    """Keep the noise phase of every observation of a run, as a service loop takes it.

    A small class rather than a closure on the per-batch path.
    """

    def __init__(self, found: list[np.ndarray]) -> None:
        self.found = found

    def __call__(self, session: RunSession, batch: EventBatch) -> None:
        for event in batch.events:
            if isinstance(event, ObservationReady):
                phase = session.noise_phase(event.ticket)  # pyright: ignore[reportAttributeAccessIssue]
                assert phase is not None
                self.found.append(np.array(phase))


def _door(  # noqa: PLR0913 - the run, its context and how it is driven
    problem: PropagationProblem,
    request: RunRequest,
    context: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    bridge: bool,
    recipe: ExportRecipe = IDENTITY,
    between: _PhaseTaker | None = None,
) -> Run:
    """Run a problem through the production door, as served or on the forced bridge.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run's request.
    context : dict
        The backend's grid, modes and profiles.
    monkeypatch : pytest.MonkeyPatch
        Forces the bridge (``session_refusal``) for this run alone.
    bridge : bool
        Whether to force the bridge.
    recipe : solprop.backends.export.ExportRecipe, optional
        The export at each observation.
    between : _PhaseTaker or None, optional
        Called with the session and each batch before it is acknowledged.

    Returns
    -------
    sessions.executors.Run
        The run.
    """
    with monkeypatch.context() as patch:
        if bridge:
            patch.setattr("solprop_native.session.session_refusal", _forced)
        backend = select("native", problem, request, **context)
        start = Start(field=problem.initial.field0, z=float(problem.initial.z))
        session = start_session(backend, problem, request, start)
        try:
            served = DriverSession if bridge else native_session.NativeSession
            assert isinstance(session, served), type(session).__name__
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BoundaryWarning)
                return drive(Started(session), export=recipe, between=between)
        finally:
            session.close()


@pytest.mark.parametrize("recipe", [CONSUMER, STORED], ids=["consumer", "stored"])
@pytest.mark.parametrize("case", DEVICE_CASES)
def test_the_device_session_gives_the_drivers_trace_and_fields(case, recipe):
    """SEP 47 sections 7.1 and 11: the device session's run is the driver's, bitwise.

    Every attempt's end, step, norm and acceptance, every observation, row and plane
    statistic, and every exported field, against the driver session on the same
    device; the session's field reaches the host only as its exports, an interpolated
    plane's too.
    """
    problem, request = _case(**CASES[case])
    baseline = drive(_driver_session(problem, request), export=recipe)
    session = _make_native(problem, request)
    run = drive(Started(session), export=recipe)
    status = session.native.status()
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    to_host, _ = backend._native_state.crossed_bytes()  # pyright: ignore[reportPrivateUsage]
    session.close()
    assert_same_run(run, baseline)
    assert run.status.attempts == baseline.status.attempts
    assert run.status.rejected == baseline.status.rejected
    if case == "rk-rejections":
        assert run.status.rejected > 0
    # One crossing per export, of the field alone: no row, statistic or decision read
    # the field on the host.
    nbytes = np.prod(problem.shape) * 16
    assert status.exports == len(run.fields)
    assert to_host == status.export_bytes == status.exports * nbytes


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
@pytest.mark.parametrize("route", ROUTES)
def test_a_spatial_route_is_the_bridges_on_the_device(route, monkeypatch):
    """A beam on each spatial route, its fluence widths at the planes and its periodic
    rows at every step, in the native session and on the bridge: the same trace and
    fields bitwise, and the same widths but for a Cartesian plane's extent, which the
    session takes with ``std::hypot`` and the bridge's host finishing with Python's,
    held to the CPU suite's 16 units of rounding."""
    grid, problem, run_request = _beam(route)
    spec = StatsSpec(names=WIDTHS, periodic=_periodic(WIDTHS), period=1)
    request = run_request.for_run(stats=spec)
    assert native_session.refusal(problem, request, entrance(problem)) is None
    context = {"grid": grid}
    run = _door(problem, request, context, monkeypatch, bridge=False)
    baseline = _door(problem, request, context, monkeypatch, bridge=True)
    if route != "plane":
        assert_same_run(run, baseline)
        return
    got, expected = trace(run), trace(baseline)
    assert [one[:5] for one in got] == [one[:5] for one in expected]
    for field, other in zip(run.fields, baseline.fields, strict=True):
        assert field.tobytes() == other.tobytes()
    for one, other in zip(run.observations, baseline.observations, strict=True):
        assert set(one.stats) == set(other.stats)
        for name in one.stats:
            np.testing.assert_allclose(
                one.stats[name],
                other.stats[name],
                rtol=16 * np.finfo(float).eps,
                atol=0.0,
                err_msg=f"{name} at {one.z}",
            )


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_collar_s_current_and_edge_are_the_bridges_on_the_device(march):
    """A radial run with a requested collar, read and judged by the session's own
    transverse edge on the device, its collar's current taken at every step: the
    trace, warnings and records the bridge's, but for the ramp's sums."""
    driver, native = _both(lambda: radial_collar("native", MARCHES[march]()))
    _compare(driver, native, 48)
    assert native.record is not None
    assert native.record["transverse"]["spatial_removed"] > 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_noisy_packet_s_edge_is_the_bridges_on_the_device(march):
    """A free-space packet on a line with the input noise, its transverse edge
    relaxed and read to the end: the bridge's run and records, but for the ramp's
    sums and the split floor's share."""

    def simulation() -> Any:
        return packet_run(
            REREVIEW,
            integrator=MARCHES[march](),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
            noise="input",
        )

    driver, native = _both(simulation)
    _compare(driver, native, _columns(simulation()), samples=_samples(simulation()))
    assert native.run.terminal.outcome is driver.run.terminal.outcome
    assert native.record is not None
    assert native.record["transverse"]["reads"] > 0


@pytest.mark.parametrize("march", NOISE_MARCHES)
@pytest.mark.parametrize("noise", ["input", "modified"])
def test_a_noisy_run_is_the_bridges_on_the_device(noise, march, monkeypatch):
    """The input and modified noise models, monitored, on adaptive Adams, densely
    observed and on the low-storage replay: the same noise phase at every plane, and
    the same trace and fields, bitwise."""
    if march == "dense" and noise == "modified":
        pytest.skip("the translation refuses dense output beside the modified noise")
    simulation = _noisy(noise, check=True, march=march)
    problem = simulation.translate()
    request = simulation._run_request().for_run(  # pyright: ignore[reportPrivateUsage]
        monitor=simulation._boundary_monitor()  # pyright: ignore[reportPrivateUsage]
    )
    assert problem.initial.noise_model == noise
    context = {"grid": simulation.grid, "modes": simulation.modes}
    phases: dict[bool, list[np.ndarray]] = {False: [], True: []}
    runs = {
        bridge: _door(
            problem,
            request,
            context,
            monkeypatch,
            bridge=bridge,
            between=_PhaseTaker(phases[bridge]),
        )
        for bridge in (False, True)
    }
    assert len(phases[False]) == len(phases[True]) > 2
    for got, expected in zip(phases[False], phases[True], strict=True):
        assert got.tobytes() == expected.tobytes()
    assert_same_run(runs[False], runs[True])


@pytest.mark.filterwarnings("ignore:the temporal absorber attenuates")
def test_a_wide_coupled_flow_is_the_bridges_on_the_device(monkeypatch):
    """A 17-channel coupled block, ten fixed exponential Adams steps on the resident
    engine of the device, whose passes take a thread per (bin, channel) at any width:
    the session's run is the bridge's over the same engine, bitwise. Its agreement
    with ``numpy_ref`` and solprop's host engine within the ten-step bound is
    ``tests/driver/test_native_flow_session.py``'s."""
    problem, request, context = coupled_flow("wide")
    assert problem.shape[0] == 17
    assert native_session.refusal(problem, request) is None
    run = _door(problem, request, context, monkeypatch, bridge=False)
    baseline = _door(problem, request, context, monkeypatch, bridge=True)
    assert run.status.attempts == baseline.status.attempts == 10
    assert_same_run(run, baseline)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("name", sorted(INTEGRATORS))
def test_a_native_group_is_the_lockstep_groups_on_the_device(name):
    """A group of three fixed-schedule members on each march, in the native group
    session -- one native session per member on the device -- and in the transitional
    lockstep group: each member's fields, statistics, attempts and rows the same,
    bitwise."""
    recorded = []
    for transitional in (False, True):
        with ExitStack() as cleanup:
            simulations = [
                member(index, "native", INTEGRATORS[name]) for index in range(3)
            ]
            group = _started(simulations, cleanup, transitional=transitional)
            served = (
                DriverGroupSession if transitional else native_group.NativeGroupSession
            )
            assert isinstance(group, served)
            recorded.append(_record(group))
    native, lockstep = recorded
    for index, (ours, theirs) in enumerate(zip(native, lockstep, strict=True)):
        assert len(ours) == len(theirs) == len(GROUP_PLANES) + 1
        for plane, (one, other) in enumerate(zip(ours, theirs, strict=True)):
            for key in one:
                assert _same(one[key], other[key]), (index, plane, key)


def test_the_production_door_serves_the_device_matrix_natively(monkeypatch):
    """SEP 47 phase 4: on a device build ``start_session`` serves an admitted run in
    the native session -- dense output and the low-storage replay among them -- and
    the bridge only where the session refuses."""
    for options in ({}, CASES["adams-dense"], CASES["low-storage"]):
        problem, request = _case(**options)
        assert native_session.refusal(problem, request) is None
        backend = select(
            "native", problem, request, grid=window(), modes=[fibre().mode]
        )
        session = start_session(backend, problem, request, entrance(problem))
        try:
            assert isinstance(session, native_session.NativeSession)
        finally:
            session.close()
    problem, request = _case()
    monkeypatch.setattr("solprop_native.session.session_refusal", _forced)
    backend = select("native", problem, request, grid=window(), modes=[fibre().mode])
    session = start_session(backend, problem, request, entrance(problem))
    try:
        assert isinstance(session, DriverSession)
    finally:
        session.close()
