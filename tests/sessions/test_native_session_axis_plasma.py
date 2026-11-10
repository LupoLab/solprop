# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The on-axis plasma statistics, taken by the native session (SEP 19, SEP 47 phase 4).

The native session evaluates them in its statistics plan
(:mod:`solprop_native._statistics`): the axis row the mode profiles synthesise on the
axis is scanned natively by the node scan's own pass, or a Radau rule's axis node is
read when the evaluation scans the nodes anyway, and the species reduction is finished
natively. The transitional session (the bridge) scans the same row on the host
(:func:`solprop.backends.populations.axis_row_populations`), so the two agree to the
node scans' parity tolerance rather than bit for bit, except where both read the
native node scan of a Radau rule's axis node. An envelope's scan reads the polarisation
ellipse, so on a one-component transform whose modes have a y field on the axis the
session scans the row's two components as one node of two. Each run is also held to
the NumPy reference on the session's exported field at every observation.
"""

import dataclasses
import math
import warnings
from contextlib import ExitStack, closing
from typing import Any, cast

import numpy as np
import pytest

from rate_oracles import adk_oracle
from sessions.executors import (
    DEVICE_GROUP,
    EXECUTORS,
    device_build,
    entrance,
    skip_device_refused,
)
from solprop.backends.driver_session import DriverGroupSession, DriverSession
from solprop.backends.events import ObservationReady, Terminal
from solprop.backends.export import ExportRecipe
from solprop.backends.registry import select
from solprop.exceptions import BackendError, MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.pulses import GaussianPulse
from solprop.stats import ON_AXIS_PLASMA_STATISTICS, StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble

native_session = pytest.importorskip("solprop_native.session")

RAW = ExportRecipe(power=False, optics=False)

PLASMA_RTOL = 1e-10
"""The node scans' parity tolerance, a compiled plasma column against the reference's
multipass scan over the window, which the backends' on-axis statistics are held to
(``test_on_axis_statistics.py``): the native row's scan against the host's."""

LENGTH = 2e-3
"""The capillary's length, m: four equal steps and two saved planes."""


def _capillary(rule_name, n_r):
    """An ionising argon capillary on a given rule, over four equal steps."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45, loss=False)
    table = build_rate_table(adk_oracle(Gas.Ar), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-4).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, request = assemble(
            grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True), PlasmaResponse(Gas.Ar, table)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-16),
            save_planes=[LENGTH / 2, LENGTH],
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 4),
            dz_max=LENGTH,
            dz_init=LENGTH / 4,
            quadrature=transverse_quadrature([mode], n_r=n_r, radial_rule=rule_name),
        )
    return grid, [mode], problem, request


def _mixture():
    """Two synthetic species, each its own population, on a resolved modal rule."""
    from mixture_cases import mixture_case  # noqa: PLC0415

    _, grid, mode, responses, problem = mixture_case()
    problem, request = assemble(
        grid,
        mode,
        list(responses),
        problem.initial.field0,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length / 2, grid.length],
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 4),
        dz_max=grid.length,
        dz_init=grid.length / 4,
        quadrature=transverse_quadrature([mode], n_r=16, radial_rule="gauss_kronrod"),
    )
    assert len(problem.responses.population_channels) == 2
    return grid, [mode], problem, request


def _envelope_y_row():
    """``envelope-rotated-x`` over four equal steps and two saved planes."""
    return _entrance_problem(
        "envelope-rotated-x",
        schedule=FixedSchedule.uniform(0.0, 0.02, 0.005),
        save_planes=(0.01, 0.02),
    )


CASES = {
    # A rule with no node on the axis: the session scans the axis row.
    "row": (lambda: _capillary("gauss_kronrod", 32), ON_AXIS_PLASMA_STATISTICS),
    # A Radau rule whose node scan runs anyway: both read its axis node.
    "radau": (
        lambda: _capillary("field_scale_radau", 5),
        ("species_ionisation", *ON_AXIS_PLASMA_STATISTICS),
    ),
    # Two species, each its own channel: the species forms and their sum.
    "mixture": (_mixture, ("energy", *ON_AXIS_PLASMA_STATISTICS)),
    # An envelope's y row on a one-component transform: the ellipse of both rows.
    "envelope-rotated-x": (_envelope_y_row, ON_AXIS_PLASMA_STATISTICS),
}


@dataclasses.dataclass
class _Seen:
    z: float
    stats: dict[str, np.ndarray]
    field: np.ndarray


def _drive(prepared, problem):
    """Run a prepared execution's session to its end; return it and its observations."""
    seen: list[_Seen] = []
    with prepared.start(entrance(problem), owns=False) as session:
        kind = type(session)
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    field = np.array(session.export(event.ticket, RAW))
                    stats = {k: np.asarray(v) for k, v in event.stats.items()}
                    seen.append(_Seen(event.z, stats, field))
                elif isinstance(event, Terminal):
                    assert event.error is None, event.error
            session.acknowledge(batch.receipt)
    return kind, seen


def _run(built, names, monkeypatch, *, bridge):
    """The session's or the bridge's observations of the run with these statistics."""
    grid, modes, problem, request = built
    run = request.for_run(stats=StatsSpec(names=names))
    with (
        closing(select("native", problem, run, grid=grid, modes=modes)) as backend,
        monkeypatch.context() as patch,
    ):
        if bridge:
            patch.setattr(
                "solprop_native.session.session_refusal", lambda *_: "the bridge"
            )
        kind, seen = _drive(backend.prepare(problem, run), problem)
    assert kind is (DriverSession if bridge else native_session.NativeSession)
    return seen


def _reference(grid, modes, problem, request, names, seen):
    """The NumPy reference's statistics on each of the session's exported fields."""
    run = request.for_run(stats=StatsSpec(names=names))
    start = float(problem.initial.z)
    out = []
    with closing(select("numpy_ref", problem, run, grid=grid, modes=modes)) as backend:
        for record in seen:
            backend.set_initial(record.field, None, z=record.z, resume=record.z > start)
            values = backend.stats(run.observation.stats)
            out.append({name: np.asarray(value) for name, value in values.items()})
    return out


def _within(actual, expected, what):
    """Assert agreement to the node scans' parity tolerance of the largest value."""
    actual = np.asarray(actual).reshape(np.shape(expected))
    bound = PLASMA_RTOL * float(np.max(np.abs(expected), initial=0.0))
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    assert error <= bound, (what, error, bound)


def test_a_run_asking_for_them_opens_the_native_session():
    """SEP 47 phase 4: nothing about the on-axis plasma statistics needs the bridge."""
    grid, modes, problem, request = _capillary("gauss_kronrod", 32)
    run = request.for_run(stats=StatsSpec(names=ON_AXIS_PLASMA_STATISTICS))
    skip_device_refused(problem, run)
    assert native_session.refusal(problem, run, entrance(problem)) is None
    with closing(select("native", problem, run, grid=grid, modes=modes)) as backend:
        prepared = backend.prepare(problem, run)
        with prepared.start(entrance(problem), owns=False) as session:
            assert type(session) is native_session.NativeSession
        assert cast("Any", prepared).refused is None


def test_the_mode_averaged_route_is_still_refused_by_name():
    """Admission is unchanged: a route the backend refuses them on is named."""
    from solprop.backends.populations import axis_refusal  # noqa: PLC0415

    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45, loss=False)
    table = build_rate_table(adk_oracle(Gas.Ar), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-4).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, request = assemble(
            grid,
            mode,
            [PlasmaResponse(Gas.Ar, table)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-16),
            save_planes=[LENGTH],
            dz_max=LENGTH,
            dz_init=LENGTH / 4,
        )
    assert problem.transform.kind == "mode_averaged"
    run = request.for_run(stats=StatsSpec(names=ON_AXIS_PLASMA_STATISTICS))
    expected = axis_refusal(problem, ON_AXIS_PLASMA_STATISTICS)
    assert expected is not None
    assert native_session.refusal(problem, run, entrance(problem)) == expected
    # The plan's own refusal, for a caller that lowers it directly.
    from solprop_native._statistics import lower  # noqa: PLC0415

    with pytest.raises(BackendError, match="mode-averaged route is not qualified"):
        lower(
            problem,
            run.observation.stats,
            ON_AXIS_PLASMA_STATISTICS,
            grid=grid,
            z_end=LENGTH,
        )


def test_the_plan_needs_the_mode_set_for_them():
    """Without the mode set there is no axis synthesis to scan."""
    from solprop_native._statistics import lower  # noqa: PLC0415

    grid, _, problem, request = _capillary("gauss_kronrod", 32)
    spec = request.for_run(stats=StatsSpec(names=ON_AXIS_PLASMA_STATISTICS))
    with pytest.raises(
        BackendError, match="on-axis plasma statistics need the mode set"
    ):
        lower(
            problem,
            spec.observation.stats,
            ON_AXIS_PLASMA_STATISTICS,
            grid=grid,
            z_end=LENGTH,
        )


@pytest.mark.parametrize("case", list(CASES))
def test_the_session_s_axis_statistics_are_the_bridge_s_and_the_reference_s(
    case, monkeypatch
):
    """At every observation: the session against the bridge and the reference.

    The two sessions march the same native steps, so their fields are the same bits
    on one thread and the statistics differ by their evaluation alone: the native
    row's scan against the host's, within the node scans' parity tolerance, and the
    same bits where both read a Radau rule's axis node from the native node scan.
    """
    build, names = CASES[case]
    grid, modes, problem, request = build()
    skip_device_refused(problem, request.for_run(stats=StatsSpec(names=names)))
    built = (grid, modes, problem, request)
    session = _run(built, names, monkeypatch, bridge=False)
    bridge = _run(built, names, monkeypatch, bridge=True)
    reference = _reference(grid, modes, problem, request, names, session)
    assert len(session) == len(bridge) == 3
    same_field = EXECUTORS["native-session"].deterministic
    for mine, theirs, want in zip(session, bridge, reference, strict=True):
        assert mine.z == theirs.z
        if same_field:
            np.testing.assert_array_equal(mine.field, theirs.field)
        for name in ON_AXIS_PLASMA_STATISTICS:
            what = f"{case} {name} at {mine.z}"
            got = mine.stats[name]
            if case == "radau" and same_field:
                np.testing.assert_array_equal(
                    got, np.reshape(theirs.stats[name], np.shape(got)), what
                )
            _within(got, np.asarray(theirs.stats[name]), f"{what} (bridge)")
            _within(got, want[name], f"{what} (numpy_ref)")
    assert max(
        float(np.max(seen.stats["on_axis_electron_density"])) for seen in session
    )
    if case == "mixture":
        # The total is the species forms' sum, in species order.
        for record in session:
            species = record.stats["on_axis_species_electron_density"]
            assert np.shape(species)[-1] == 2
            assert math.fsum(np.ravel(species)) == pytest.approx(
                float(np.ravel(record.stats["on_axis_electron_density"])[0]), rel=1e-15
            )


# --- the plan's axis row at the entrance, on every kind of row ----------------------


def _entrance_problem(kind, schedule=None, save_planes=(0.02,)):
    """An argon capillary of one or two modes at its entrance, built for one row kind.

    ``rotated-x`` carries modes with a y field on the axis on a transform resolving x
    alone, so a field grid's one-component scan reads the magnitude of both rows;
    ``envelope-rotated-x`` is the envelope form, whose scan reads both rows' ellipse.
    """
    from solprop.grid import EnvelopeGrid  # noqa: PLC0415
    from solprop.nonlinear import EnvelopePlasmaResponse  # noqa: PLC0415

    envelope = kind.startswith("envelope")
    grid = (EnvelopeGrid if envelope else FieldGrid).from_limits(
        0.02, 800e-9, (300e-9, 3e-6), 100e-15
    )
    table = build_rate_table(adk_oracle(Gas.Ar), (1e9, 2e11))

    def argon(m=1, phi=0.0):
        return MarcatiliMode(125e-6, "Ar", 1.0, m=m, cladding=1.45, loss=False, phi=phi)

    rotated = kind.endswith("rotated-x")
    modes = (
        [argon(phi=math.pi / 4), argon(2, phi=math.pi / 3)]
        if rotated
        else [argon(1), argon(2)]
        if kind.endswith("two-modes")
        else [argon()]
    )
    plasma = (
        EnvelopePlasmaResponse(Gas.Ar, table)
        if envelope
        else PlasmaResponse(Gas.Ar, table, integration="band")
        if kind == "band"
        else PlasmaResponse(Gas.Ar, table)
    )
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-4).build(grid)
    start = np.concatenate([built, 0.4j * built][: len(modes)], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, request = assemble(
            grid,
            modes,
            [KerrResponse(chi3=2.2e-52, thg=not envelope), plasma],
            start,
            controller=PIController(rtol=1e-8, atol=1e-16),
            save_planes=list(save_planes),
            dz_max=0.002,
            dz_init=1e-4,
            schedule=schedule,
            quadrature=transverse_quadrature(
                modes, n_r=32, radial_rule="gauss_kronrod"
            ),
            components=("x",) if rotated else None,
        )
    return grid, modes, problem, request


ENTRANCE_KINDS = (
    "rotated-x",
    "envelope-rotated-x",
    "band",
    "envelope",
    "envelope-two-modes",
)


@pytest.mark.parametrize("kind", ENTRANCE_KINDS)
def test_the_plan_s_axis_row_is_the_bridge_s_and_the_reference_s(kind):
    """The plan's native row scan, at the entrance, on each kind of row and scan.

    A field grid's one-component scan of two rows (their magnitude), an envelope's
    one-component scan of two rows (their ellipse), the band rule's filtered exposure,
    and an envelope's cycle-averaged scan of one and two modes.
    """
    from solprop_native._statistics import _on_axis, lower  # noqa: PLC0415

    grid, modes, problem, request = _entrance_problem(kind)
    names = ON_AXIS_PLASMA_STATISTICS
    spec = StatsSpec(names=names)
    run = request.for_run(stats=spec)
    if kind.endswith("rotated-x"):
        # A y field on the axis, on a transform resolving x alone.
        assert problem.transform.n_pol == 1
        assert np.any(_on_axis(modes)[1])
    values = {}
    for name in ("numpy_ref", "native"):
        with closing(select(name, problem, run, grid=grid, modes=modes)) as backend:
            backend.set_initial(problem.initial.field0, None)
            values[name] = backend.stats(spec)
    with closing(select("native", problem, run, grid=grid, modes=modes)) as backend:
        backend.set_initial(problem.initial.field0, None)
        lowering = lower(problem, spec, names, grid=grid, modes=modes, z_end=0.02)
        state = backend._native_state  # pyright: ignore[reportAttributeAccessIssue]
        got = lowering.unpack(state.planned_statistics(lowering.plan, 0.0))
    for name in names:
        want = np.reshape(values["numpy_ref"][name], np.shape(got[name]))
        bridge = np.reshape(values["native"][name], np.shape(got[name]))
        assert float(np.max(want)) > 0.0
        _within(got[name], want, f"{kind} {name} (numpy_ref)")
        _within(got[name], bridge, f"{kind} {name} (bridge)")


def test_an_envelope_s_y_row_on_a_one_component_scan_opens_the_native_session():
    """The ellipse of both rows is scanned natively, so nothing sends the run to the
    bridge (SEP 47 phase 4)."""
    grid, modes, problem, request = _entrance_problem("envelope-rotated-x")
    run = request.for_run(stats=StatsSpec(names=ON_AXIS_PLASMA_STATISTICS))
    skip_device_refused(problem, run)
    with closing(select("native", problem, run, grid=grid, modes=modes)) as backend:
        prepared = backend.prepare(problem, run)
        with prepared.start(entrance(problem), owns=False) as session:
            assert type(session) is native_session.NativeSession
        assert cast("Any", prepared).refused is None


def _group_planes(group):
    """Drive a group to its end; return each member's observations."""
    planes = [[] for _ in range(group.n_members)]
    while not group.finished:
        batch = group.advance()
        for index, events in enumerate(batch.members):
            for event in events:
                if isinstance(event, ObservationReady):
                    field = np.array(group.export(index, event.ticket, RAW))
                    stats = {k: np.asarray(v) for k, v in event.stats.items()}
                    planes[index].append(_Seen(event.z, stats, field))
                elif isinstance(event, Terminal):
                    assert event.error is None, event.error
        group.acknowledge(batch.receipt)
    return planes


@pytest.mark.skipif(device_build(), reason=DEVICE_GROUP)
def test_a_native_group_scans_an_envelope_s_y_row():
    """A group of such members opens the native group, whose members' statistics are
    the transitional group's, which scans the row on the host, and the reference's to
    the node scans' parity tolerance."""
    from solprop.backends.session import Start  # noqa: PLC0415
    from solprop.backends.sessions import start_group  # noqa: PLC0415

    group_module = pytest.importorskip("solprop_native.group")
    names = ON_AXIS_PLASMA_STATISTICS
    built = [_envelope_y_row() for _ in range(2)]
    problems = [problem for _, _, problem, _ in built]
    requests = [request.for_run(stats=StatsSpec(names=names)) for *_, request in built]
    recorded = []
    for door in (start_group, DriverGroupSession):
        # A start is handed to one session.
        starts = [Start.entrance(problem) for problem in problems]
        with ExitStack() as cleanup:
            owners = [
                cleanup.enter_context(
                    closing(select("native", problem, run, grid=grid, modes=modes))
                )
                for (grid, modes, problem, _), run in zip(built, requests, strict=True)
            ]
            assert group_module.refusal(problems, requests, starts) is None
            group = door(owners, problems, requests, starts)
            cleanup.callback(group.close)
            expected = (
                group_module.NativeGroupSession
                if door is start_group
                else DriverGroupSession
            )
            assert type(group) is expected
            recorded.append(_group_planes(group))
    same_field = EXECUTORS["native-session"].deterministic
    for index, (ours, theirs) in enumerate(zip(*recorded, strict=True)):
        grid, modes, problem, request = built[index]
        reference = _reference(grid, modes, problem, request, names, ours)
        assert len(ours) == len(theirs) == 3
        for mine, other, want in zip(ours, theirs, reference, strict=True):
            assert mine.z == other.z
            if same_field:
                np.testing.assert_array_equal(mine.field, other.field)
            for name in names:
                what = f"member {index} {name} at {mine.z}"
                got = mine.stats[name]
                _within(got, np.asarray(other.stats[name]), f"{what} (bridge)")
                _within(got, want[name], f"{what} (numpy_ref)")
