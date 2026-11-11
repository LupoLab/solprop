# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's device matrix on a device, against the reference.

SEP 47 phase 4 (section 6.3): on a CUDA or HIP build the native session serves what a
device build runs -- Adams and the embedded Runge-Kutta tableaux, adaptive and
replayed, the low-storage replay, dense output, the radial, Cartesian, line and planar
spatial routes with their fluence widths, a free-space run's transverse edge and a
collar's current, the input and modified noise models, a coupled flow wider than 16
channels or off its eigenbasis route, and fixed-schedule groups (the native group
session, one native session per member). Since phase 5 deleted the transitional bridge
it was first held to bit for bit, each feature runs on the device in the native session
and is held to the NumPy reference session on the same problem within the CPU suite's
budgets (``test_native_session.py``'s :func:`assert_meets_reference`, the transverse
suite's decisions), and to itself: a repeated run on the device is the same bits, as
on a CPU build's one thread, its decisions made on the host from the same device
reductions. A group's member is its lone session's. The session crosses the field to
the host only for an export.
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
    COUPLED_KINDS,
    IDENTITY,
    STORED,
    Run,
    Started,
    assert_same_run,
    coupled_flow,
    drive,
    entrance,
    fibre,
    window,
)
from sessions.test_group import PLANES as GROUP_PLANES
from sessions.test_group import member
from sessions.test_native_group import INTEGRATORS, _lone, _record, _same, _started
from sessions.test_native_session import (
    CASES,
    DENSE,
    DEVICE_PLANE_BOUND,
    _case,
    _control_trace,
    _make_native,
    _reference_session,
    assert_meets_reference,
)
from sessions.test_native_session_history import _assert_phases_meet, _noisy
from sessions.test_native_session_statistics import WIDTHS, _beam, _periodic
from sessions.test_native_session_transverse import (
    MARCHES,
    _both,
    _compare,
)
from solprop.backends.events import ObservationReady
from solprop.backends.host.session import HostSession
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
the low-storage replay and dense output in both Adams formulations. The Runge-Kutta
tableau's dense output is :data:`RK_DENSE`'s, held to a bound."""

RK_DENSE = {
    "runge-kutta": {"step_planes": 3, "rtol": 1e-8},
    "runge-kutta-rejections": DENSE["runge-kutta-rejections"],
}
"""DOPRI5's dense output, adaptive and with rejections (the CPU suite's
``test_runge_kutta_dense_planes_repeat_and_follow_the_reference``)."""

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


def _door(
    problem: PropagationProblem,
    request: RunRequest,
    context: dict[str, Any],
    *,
    name: str = "native",
    recipe: ExportRecipe = IDENTITY,
    between: _PhaseTaker | None = None,
) -> Run:
    """Run a problem through the production door on a backend.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run's request.
    context : dict
        The backend's grid, modes and profiles.
    name : str, optional
        The backend: ``"native"``, served by the native session, or ``"numpy_ref"``,
        by the reference's.
    recipe : solprop.backends.export.ExportRecipe, optional
        The export at each observation.
    between : _PhaseTaker or None, optional
        Called with the session and each batch before it is acknowledged.

    Returns
    -------
    sessions.executors.Run
        The run.
    """
    backend = select(name, problem, request, **context)
    start = Start(field=problem.initial.field0, z=float(problem.initial.z))
    session = start_session(backend, problem, request, start)
    try:
        served = native_session.NativeSession if name == "native" else HostSession
        assert isinstance(session, served), type(session).__name__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            return drive(Started(session), export=recipe, between=between)
    finally:
        session.close()


@pytest.mark.parametrize("recipe", [CONSUMER, STORED], ids=["consumer", "stored"])
@pytest.mark.parametrize("case", DEVICE_CASES)
def test_the_device_session_repeats_itself_and_meets_the_reference(case, recipe):
    """SEP 47 sections 7.1 and 11: the device session's run, twice and against the
    reference.

    Every attempt's end, step, norm and acceptance, every observation, row and plane
    statistic, and every exported field repeat bitwise on the device, and the run
    meets the reference session's (:func:`assert_meets_reference`); the session's
    field reaches the host only as its exports, an interpolated plane's too.
    """
    problem, request = _case(**CASES[case])
    reference = drive(_reference_session(problem, request), export=recipe)
    again = _make_native(problem, request)
    baseline = drive(Started(again), export=recipe)
    again.close()
    session = _make_native(problem, request)
    run = drive(Started(session), export=recipe)
    status = session.native.status()
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    to_host, _ = backend._native_state.crossed_bytes()  # pyright: ignore[reportPrivateUsage]
    session.close()
    assert_same_run(run, baseline)
    assert_meets_reference(run, reference, plane_bound=DEVICE_PLANE_BOUND)
    if case == "rk-rejections":
        assert run.status.rejected > 0
    # One crossing per export, of the field alone: no row, statistic or decision read
    # the field on the host.
    nbytes = np.prod(problem.shape) * 16
    assert status.exports == len(run.fields)
    assert to_host == status.export_bytes == status.exports * nbytes


@pytest.mark.parametrize("case", sorted(RK_DENSE))
def test_runge_kutta_dense_planes_repeat_and_follow_the_reference_on_the_device(case):
    """DOPRI5's dense output in the device session: every attempt, norm, acceptance,
    row and observation position, and every plane, the same bits when repeated (the
    session reloads the committed state after an observation), and the run within the
    reference's budgets (:func:`assert_meets_reference`)."""
    options = {"step_planes": 0, **RK_DENSE[case]}
    problem, request = _case(interpolate=True, **options)
    reference = drive(_reference_session(problem, request), export=CONSUMER)
    runs = []
    for _ in range(2):
        session = _make_native(problem, request)
        runs.append(drive(Started(session), export=CONSUMER))
        session.close()
    run, again = runs
    assert len(run.fields) > 2
    assert _control_trace(again) == _control_trace(run)
    for field, other in zip(again.fields, run.fields, strict=True):
        assert field.tobytes() == other.tobytes()
    assert_meets_reference(run, reference, plane_bound=DEVICE_PLANE_BOUND)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
@pytest.mark.parametrize("route", ROUTES)
def test_a_spatial_route_repeats_and_meets_the_reference_on_the_device(route):
    """A beam on each spatial route, its fluence widths at the planes and its periodic
    rows at every step, in the native session: a repeated run's trace and fields the
    same bits, and the run within the reference's budgets
    (:func:`assert_meets_reference`)."""
    grid, problem, run_request = _beam(route)
    spec = StatsSpec(names=WIDTHS, periodic=_periodic(WIDTHS), period=1)
    request = run_request.for_run(stats=spec)
    assert native_session.refusal(problem, request, entrance(problem)) is None
    context = {"grid": grid}
    run = _door(problem, request, context)
    assert_same_run(_door(problem, request, context), run)
    assert_meets_reference(
        run,
        _door(problem, request, context, name="numpy_ref"),
        plane_bound=DEVICE_PLANE_BOUND,
    )


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_collar_s_current_and_edge_meet_the_reference_on_the_device(march):
    """A radial run with a requested collar, read and judged by the session's own
    transverse edge on the device, its collar's current taken at every step: the
    reference's decisions (the transverse suite's ``_compare``), and a repeated run
    the same bits."""
    reference, native = _both(lambda: radial_collar("native", MARCHES[march]()))
    _compare(reference, native)
    assert native.record is not None
    assert native.record["transverse"]["spatial_removed"] > 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_noisy_packet_s_edge_meets_the_reference_on_the_device(march):
    """A free-space packet on a line with the input noise, its transverse edge
    relaxed and read to the end: the reference's decisions, and a repeated run the
    same bits."""

    def simulation() -> Any:
        return packet_run(
            REREVIEW,
            integrator=MARCHES[march](),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
            noise="input",
        )

    reference, native = _both(simulation)
    _compare(reference, native)
    assert native.run.terminal.outcome is reference.run.terminal.outcome
    assert native.record is not None
    assert native.record["transverse"]["reads"] > 0


@pytest.mark.parametrize("march", NOISE_MARCHES)
@pytest.mark.parametrize("noise", ["input", "modified"])
def test_a_noisy_run_repeats_and_meets_the_reference_on_the_device(noise, march):
    """The input and modified noise models, monitored, on adaptive Adams, densely
    observed and on the low-storage replay: a repeated run's noise phase at every
    plane, trace and fields the same bits, and the reference's phases and run within
    their budgets."""
    if march == "dense" and noise == "modified":
        pytest.skip("the translation refuses dense output beside the modified noise")
    simulation = _noisy(noise, check=True, march=march)
    problem = simulation.translate()
    request = simulation._run_request().for_run(  # pyright: ignore[reportPrivateUsage]
        monitor=simulation._boundary_monitor()  # pyright: ignore[reportPrivateUsage]
    )
    assert problem.initial.noise_model == noise
    context = {"grid": simulation.grid, "modes": simulation.modes}
    kinds = (("native", "native"), ("again", "native"), ("reference", "numpy_ref"))
    phases: dict[str, list[np.ndarray]] = {kind: [] for kind, _ in kinds}
    runs = {
        kind: _door(
            problem, request, context, name=name, between=_PhaseTaker(phases[kind])
        )
        for kind, name in kinds
    }
    assert len(phases["native"]) == len(phases["again"]) > 2
    for got, expected in zip(phases["again"], phases["native"], strict=True):
        assert got.tobytes() == expected.tobytes()
    assert_same_run(runs["again"], runs["native"])
    _assert_phases_meet(phases["native"], phases["reference"], problem.shape)
    assert_meets_reference(runs["native"], runs["reference"])


@pytest.mark.filterwarnings("ignore:the temporal absorber attenuates")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning")
@pytest.mark.parametrize("kind", COUPLED_KINDS)
def test_a_wide_or_off_route_coupled_flow_repeats_itself_on_the_device(kind):
    """A 17-channel coupled block, and a block with every bin off its eigenbasis route,
    ten fixed exponential Adams steps on the resident engine of the device: its passes
    take a thread per (bin, channel) at any width, and a thread per bin off the route
    forms that bin's flow, moments and carries by the augmented exponential. A
    repeated run on the device is the same bits. Its agreement with ``numpy_ref``
    within the ten-step bounds is ``tests/driver/test_native_flow_session.py``'s."""
    problem, request, context = coupled_flow(kind)
    if kind == "wide":
        assert problem.shape[0] == 17
    assert native_session.refusal(problem, request) is None
    run = _door(problem, request, context)
    baseline = _door(problem, request, context)
    assert run.status.attempts == baseline.status.attempts == 10
    assert_same_run(run, baseline)


@pytest.mark.parametrize("name", sorted(INTEGRATORS))
def test_a_native_group_member_is_its_lone_session_on_the_device(name):
    """A group of three fixed-schedule members on each march, in the native group
    session -- one native session per member on the device -- and each member alone in
    its native session: each member's fields, statistics, attempts and rows the same,
    bitwise."""
    with ExitStack() as cleanup:
        simulations = [member(index, "native", INTEGRATORS[name]) for index in range(3)]
        group = _started(simulations, cleanup)
        assert isinstance(group, native_group.NativeGroupSession)
        native = _record(group)
    alone = [_lone(member(index, "native", INTEGRATORS[name])) for index in range(3)]
    for index, (ours, theirs) in enumerate(zip(native, alone, strict=True)):
        assert len(ours) == len(theirs) == len(GROUP_PLANES) + 1
        for plane, (one, other) in enumerate(zip(ours, theirs, strict=True)):
            for key in one:
                assert _same(one[key], other[key]), (index, plane, key)


def test_the_production_door_serves_the_device_matrix_natively():
    """SEP 47 phase 4: on a device build ``start_session`` serves an admitted run in
    the native session -- dense output and the low-storage replay among them."""
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
