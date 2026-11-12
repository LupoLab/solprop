# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The carrier frame on a z table, in the native session (issue #255, SEP 47 6.3).

The scalar carrier frame's shift is the reference element's chord, ``Im (Phi(z + h) -
Phi(z))`` at column 0, polarisation 0 and the reference bin (SEP 35 section 6.1). A z
table's chord is that element's column of the table; the native stage tables carry its
imaginary part as a scalar Chebyshev table of its own and read it with the Clenshaw
helper solprop's tables use, so the shift the native march forms is the one solprop's
march formed on the host. Before, the backend stepped such a run with its Python march
(a ``HostMarchWarning``) and the session left it to the transitional bridge.

The case is a capillary of three gases on independent density ramps, which takes the z
table for both the operator and the prefactor. It is held to ``numpy_ref``: on the
native run's own accepted steps, at a fixed order, the two differ by rounding alone,
512 eps a step relative to the field, the budget of
``tests/backends/test_carrier_frame.py``'s matched windows (SEP 25), and adaptively they
take as many steps, to that budget; on one thread the native run repeats itself bit for
bit, and a run resumed from a mid-run history continues the uninterrupted one. The
carrier approximation itself is unchanged; a device build still
refuses every carrier frame by name.
"""

import dataclasses
import functools
import warnings
from typing import Any, NamedTuple, cast

import numpy as np
import pytest

from bounds import EPS, assert_within
from sessions.executors import (
    EXECUTORS,
    IDENTITY,
    Started,
    _one_thread,  # pyright: ignore[reportPrivateUsage]
    assert_same_run,
    device_build,
    drive,
    entrance,
)
from solprop.backends.events import ObservationReady
from solprop.backends.exponential import carrier_reference
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.grid import FieldGrid
from solprop.output.records import MarchSnapshot
from solprop.physdata import Mixture
from solprop.profiles import Linear
from solprop.stepper import FixedSchedule, Multistep
from solprop.translate.chebyshev import build as build_table
from solprop.translate.linop import delta_phi
from solprop.translate.problem import Flags, PropagationProblem, validate_run
from solprop.translate.run_request import RunRequest

native_session = pytest.importorskip("solprop_native.session")
native_backend = pytest.importorskip("solprop_native.backend")

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        device_build(), reason="device builds refuse the carrier frame by name"
    ),
]

LENGTH = 2e-3
"""The capillary's length, m: about 120 adaptive steps at ``rtol = 1e-8``."""

SCALAR = Multistep(carrier_frame="scalar")


@functools.cache
def capillary(method: Multistep = SCALAR, *, interpolate: bool = False) -> Simulation:
    """Return the capillary of :func:`build`, adaptive, built once per choice.

    Parameters
    ----------
    method : Multistep, optional
        The method.
    interpolate : bool, optional
        Whether planes are interpolated inside steps (dense output).

    Returns
    -------
    solprop.driver.Simulation
        The simulation.
    """
    return build(method, interpolate=interpolate)


def build(
    method: Multistep = SCALAR,
    *,
    interpolate: bool = False,
    schedule: FixedSchedule | None = None,
) -> Simulation:
    """Return a 16 um capillary of H2, He and Ar on independent density ramps.

    Three species on their own paths exceed the two-parameter expansion, so both the
    operator and the prefactor take the z table; the field grid keeps the carrier
    phase whole, so the frame is active. Kerr alone, with a weak temporal collar, as
    ``tests/backends/test_carrier_frame.py``'s capillary.

    Parameters
    ----------
    method : Multistep, optional
        The method.
    interpolate : bool, optional
        Whether planes are interpolated inside steps (dense output).
    schedule : solprop.stepper.FixedSchedule or None, optional
        A replayed itinerary; adaptive by default.

    Returns
    -------
    solprop.driver.Simulation
        The simulation, on ``numpy_ref`` by default.
    """
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (200e-9, 2e-6), 200e-15)
    fill = Mixture.fill({"H2": 0.5, "He": 0.3, "Ar": 0.2}, 40.0)
    fill = fill.with_densities(
        {
            label: Linear((index + 1.0) * 1e26 / 3, (3.5 - index) * 1e26 / 3, LENGTH)
            for index, label in enumerate(fill.labels)
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Simulation.from_capillary(
            16e-6,
            LENGTH,
            fill,
            None,
            grid=grid,
            wavelength=800e-9,
            fwhm=30e-15,
            energy=2e-6,
            rtol=1e-8,
            atol=1e-12,
            noise=None,
            backend="numpy_ref",
            n_save=4,
            integrator=method,
            interpolate_planes=interpolate,
            schedule=schedule,
            raman=False,
            plasma=False,
            absorber=absorber(grid, boundary_length=1.0),
        )


def gradient(
    method: Multistep = SCALAR, *, schedule: FixedSchedule | None = None
) -> Simulation:
    """Return a 16 um hydrogen capillary on a 40 to 20 bar gradient, the frame active.

    A pressure gradient's operator and prefactor are expansions; :func:`on_a_table`
    puts the operator alone on a z table. Kerr alone, with a weak temporal collar, as
    ``tests/backends/test_carrier_frame.py``'s gradient.

    Parameters
    ----------
    method : Multistep, optional
        The method.
    schedule : solprop.stepper.FixedSchedule or None, optional
        A replayed itinerary; adaptive by default.

    Returns
    -------
    solprop.driver.Simulation
        The simulation, on ``numpy_ref`` by default.
    """
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (200e-9, 2e-6), 200e-15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Simulation.from_capillary(
            16e-6,
            LENGTH,
            "H2",
            (40.0, 20.0),
            grid=grid,
            wavelength=800e-9,
            fwhm=30e-15,
            energy=2e-6,
            rtol=1e-8,
            atol=1e-12,
            noise=None,
            backend="numpy_ref",
            n_save=4,
            integrator=method,
            schedule=schedule,
            raman=False,
            plasma=False,
            absorber=absorber(grid, boundary_length=1.0),
        )


class Case(NamedTuple):
    """A problem, its run request, and what its backends take beside them."""

    problem: PropagationProblem
    request: RunRequest
    context: dict[str, Any]


def on_a_table(simulation: Simulation) -> Case:
    """Return `simulation`'s problem with ``Phi`` alone tabulated along z.

    The table is built from the values the expansion gives, as a mode set that
    declares no parameters would present them (``tests/backends/test_native_z_table.py``
    tabulates the same way); the prefactor keeps its own representation.

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation, whose operator is an expansion.

    Returns
    -------
    Case
        The tabulated problem, the simulation's request and context.
    """
    problem = simulation.translate()
    request = simulation._run_request()
    linear = problem.linear
    table = build_table(
        lambda z: np.stack([delta_phi(linear, 0.0, float(one)) for one in z]),
        *linear.z_span,
        tol=1e-10,
    )
    tabulated = dataclasses.replace(
        linear,
        phi_expansion=None,
        phi_representation="table",
        phi_breakpoints=table.breakpoints,
        phi_coefficients=table.coefficients,
        phi_degrees=table.degrees,
        hard_kinks=table.hard_kinks,
        phi_constant=None,
    )
    flags = Flags.from_content(
        problem.shape,
        tabulated,
        problem.route.transform,
        problem.responses,
        problem.initial,
    )
    found = dataclasses.replace(problem, linear=tabulated, flags=flags)
    validate_run(found, request)
    return Case(found, request, _context(simulation))


def _context(simulation: Simulation) -> dict[str, Any]:
    """Return what a backend of `simulation` takes beside its problem.

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation.

    Returns
    -------
    dict
        The grid, the modes and the profiles.
    """
    return {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }


def _run(name: str, simulation: Simulation | Case, start: Start | None = None):
    """Drive one executor of the lifecycle suite over `simulation` to its end.

    Parameters
    ----------
    name : str
        The executor (``sessions.executors.EXECUTORS``).
    simulation : solprop.driver.Simulation or Case
        The simulation, or a problem built from one.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the entrance by default.

    Returns
    -------
    tuple
        The run and the march a driver session's backend took (``None`` for another
        session).
    """
    case = simulation
    if not isinstance(case, Case):
        case = Case(case.translate(), case._run_request(), _context(case))
    started = EXECUTORS[name].start(
        case.problem,
        case.request,
        entrance(case.problem) if start is None else start,
        context=case.context,
    )
    try:
        run = drive(started, export=IDENTITY)
        return run, _march(started)
    finally:
        started.session.close()


def _march(started: Started) -> str | None:
    """Return which march stepped a driver session, or ``None`` for another session.

    Parameters
    ----------
    started : sessions.executors.Started
        The session.

    Returns
    -------
    str or None
        The driver's ``adams_march`` record.
    """
    driver = getattr(started.session, "driver", None)
    return None if driver is None else driver.metadata().get("adams_march")


def test_the_chord_is_a_varying_table_the_session_admits():
    # The operator and the prefactor are z tables, the chord the table's reference
    # element over several intervals, whose rate varies along the capillary; the
    # session admits the run before anything is allocated, and the built backend's
    # own march steps it natively.
    simulation = capillary()
    problem = simulation.translate()
    request = simulation._run_request()
    assert problem.linear.phi_representation == "table"
    assert problem.responses.prefactor_representation == "table"
    reference = carrier_reference(problem, SCALAR)
    assert reference is not None and reference.active
    assert reference.chord is not None and reference.chord.table is not None
    assert reference.chord.table.coefficients.shape[0] > 1
    rates = [
        float(np.imag(reference.chord.exponent(z, z + 1e-4))[0]) / 1e-4
        for z in (0.0, LENGTH - 1e-4)
    ]
    assert abs(rates[0] - rates[1]) > 1e-6 * abs(rates[0])
    assert native_backend.host_rows_refusal(problem) is None
    assert native_session.refusal(problem, request) is None
    # The built native backend, whose march and tables the checks reach.
    backend: Any = select("native", problem, request, **_context(simulation))
    try:
        assert backend._adams._policy.carrier
        # The table is the chord's whole description: a chord of kind ``table``
        # without one is refused when the tables are uploaded.
        tables = backend._stage_tables()
        assert tables is not None
        # A chord kind the build does not know is refused, not cast.
        for chord in (3, 4, -1):
            with pytest.raises(ValueError, match="stage tables are inconsistent"):
                backend._native_state.adams_stage_tables(
                    tables
                    | {
                        "chord": chord,
                        "chord_rate": 0.0,
                        "chord_terms": [],
                        "chord_table": None,
                    }
                )
    finally:
        backend.close()


@pytest.mark.parametrize("interpolate", [False, True], ids=["planes", "dense"])
def test_the_session_takes_the_reference_s_steps_and_repeats_them(interpolate):
    # The native session against the reference's adaptive run, which forms the shift
    # by its own tables: as many steps, each within the accepted-step budget, and the
    # planes within 512 eps a step; dense planes read the accepted step's rotations.
    # On one thread a repeated native run is the same, bitwise.
    simulation = capillary(interpolate=interpolate)
    run, _ = _run("native-session", simulation)
    reference, _ = _run("reference", simulation)
    ends = run.accepted_ends()
    assert len(ends) > 50
    assert len(ends) == len(reference.accepted_ends())
    np.testing.assert_allclose(ends, reference.accepted_ends(), rtol=1e-6)
    assert len(run.fields) == len(reference.fields)
    for one, other in zip(run.fields, reference.fields, strict=True):
        assert_within(one, other, 512 * EPS * len(ends))
    if _one_thread():
        again, _ = _run("native-session", simulation)
        assert_same_run(again, run)


def test_the_session_follows_the_reference_on_its_own_steps():
    # numpy_ref replays the native run's accepted steps at a fixed order, so both run
    # one discrete method and history: rounding alone, 512 eps a step.
    method = Multistep(variable_order=False, carrier_frame="scalar")
    simulation = capillary(method)
    adaptive, _ = _run("native-session", simulation)
    ends = adaptive.accepted_ends()
    steps = np.diff(np.concatenate([[0.0], ends]))
    steps[-1] = LENGTH - float(np.sum(steps[:-1]))
    replayed = build(method, schedule=FixedSchedule(steps))
    native, _ = _run("native-session", replayed)
    reference, _ = _run("reference", replayed)
    assert len(native.fields) == len(reference.fields) > 1
    for one, other in zip(native.fields, reference.fields, strict=True):
        assert_within(
            one, other, len(steps) * 512 * EPS, what="native session against numpy_ref"
        )
    # The adaptive runs agree in their step count; their steps differ by rounding.
    free, _ = _run("reference", simulation)
    assert len(free.accepted_ends()) == len(ends)


def _at_second_plane(simulation: Simulation) -> tuple[MarchSnapshot, dict[str, Any]]:
    """Run the native session to its second plane; return its history and ledger.

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation.

    Returns
    -------
    tuple
        The snapshot and the call ledger's record there.
    """
    problem = simulation.translate()
    started = EXECUTORS["native-session"].start(
        problem,
        simulation._run_request(),
        entrance(problem),
        context=_context(simulation),
    )
    session = started.session
    planes = 0
    try:
        while True:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady) and event.plane_index >= 0:
                    planes += 1
                    if planes == 2:
                        snapshot = session.march_snapshot(event.ticket)
                        assert snapshot is not None
                        record = session.execution_record()["response_calls"]  # pyright: ignore[reportAttributeAccessIssue]
                        assert isinstance(record, dict)
                        return snapshot, cast("dict[str, Any]", record)
            session.acknowledge(batch.receipt)
    finally:
        session.close()


def test_a_resumed_history_continues_the_uninterrupted_run():
    # SEP 47 section 5.5: the history exported at a plane, installed in a new native
    # session, continues as the uninterrupted run goes on from there.
    simulation = capillary()
    snapshot, ledger = _at_second_plane(simulation)
    assert snapshot.z > 0.0 and ledger["accepted_steps"]
    planes = simulation._run_request().stepper.save_planes

    def resumed() -> Start:
        return Start(
            field=np.array(snapshot.state),
            z=snapshot.z,
            plane_index=int(np.searchsorted(planes, snapshot.z, "right")),
            plane_pending=False,
            resume=True,
            march=snapshot,
            ledgers={"logical": ledger},
        )

    run, march = _run("native-session", simulation, resumed())
    assert march is None
    whole, _ = _run("native-session", simulation)
    ends = whole.accepted_ends()
    if _one_thread():
        later = ends[ends > snapshot.z]
        assert run.accepted_ends().tobytes() == later.tobytes()
    # The resumed run ends where the uninterrupted one does, to the march's rounding.
    bound = 512 * EPS * len(ends)
    assert_within(run.fields[-1], whole.fields[-1], bound)


def test_a_tabulated_operator_beside_an_expanded_prefactor():
    # The chord read from the operator's table alone, the prefactor an expansion: the
    # session repeats itself bitwise on one thread, and meets numpy_ref on the
    # session's own steps at a fixed order, 512 eps a step.
    method = Multistep(variable_order=False, carrier_frame="scalar")
    case = on_a_table(gradient(method))
    assert case.problem.linear.phi_representation == "table"
    assert case.problem.responses.prefactor_representation == "expansion"
    reference = carrier_reference(case.problem, method)
    assert reference is not None and reference.chord is not None
    assert reference.chord.table is not None
    assert native_session.refusal(case.problem, case.request) is None
    run, march = _run("native-session", case)
    assert march is None
    ends = run.accepted_ends()
    assert len(ends) > 20
    if _one_thread():
        again, _ = _run("native-session", case)
        assert_same_run(again, run)
    steps = np.diff(np.concatenate([[0.0], ends]))
    steps[-1] = LENGTH - float(np.sum(steps[:-1]))
    replayed = on_a_table(gradient(method, schedule=FixedSchedule(steps)))
    native, _ = _run("native-session", replayed)
    expected, _ = _run("reference", replayed)
    assert len(native.fields) == len(expected.fields) > 1
    for one, other in zip(native.fields, expected.fields, strict=True):
        assert_within(
            one, other, len(steps) * 512 * EPS, what="native session against numpy_ref"
        )
