# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's reading on every backend, against the reference (SEP 21).

A check reads five numbers of the state where the state lives (on native, the native
state's ``monitor_reading``, which its session reads at every check): its energy and
its four weighted energies, the low and high guards, the collar's ``1 - W`` and its
rate. Under the ``"split_step"``
scheme the backend's own finish adds what each accepted step's collar takes, and the
reading drains the sum, beside the energies of what the last step's split found.
``numpy_ref`` forms both on the host
(:meth:`solprop.monitor.BoundaryMonitor.energies`, ``split_sums``), the oracle every
other backend is held to here, on the probes of ``tests/monitor_cases.py``: each route's
tiny case launched with a copy of its pulse near the window's edge, so that every
region holds light.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from backends._session_probe_b1 import (
    bind_native_monitor,
    launched_state,
    native_reading,
)
from backends._session_probe_b1 import run as session_run
from bounds import device_backend, ladder_bound
from capability_routes import LENGTH
from legacy_geometry import bare_absorber
from monitor_cases import (
    MARCHES,
    ROUTE_NAMES,
    SCHEMES,
    STEPS,
    as_driver,
    probe,
    reading,
    record_after_march,
    watched,
    with_copy,
)
from solprop.backends.registry import available, select
from solprop.backends.session import ExecutionOptions, Start
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.translate.problem import PropagationProblem

COMPILED = ("native",)


def _rounding(simulation: Simulation) -> float:
    """Return the reading's rounding budget, relative to the state's energy.

    Each time sample carries the inverse transform's rounding, about ``eps log2
    n_t`` of the field's norm, and each sum adds ``eps`` per term in the worst order
    a threaded reduction may take: ``eps (log2 n_t + n)`` over the ``n`` samples of
    every column, all of them non-negative.
    """
    shape = simulation.translate().shape
    samples = math.prod(shape[:-1]) * simulation.grid.n_t
    return float(np.finfo(np.float64).eps) * (math.log2(simulation.grid.n_t) + samples)


def _require(backend: str) -> None:
    """Skip when a backend is not installed."""
    if backend not in available():
        pytest.skip(f"{backend} is not installed")


def _checkreading(
    backend: str,
    route: str,
    scheme: str,
    problem: Callable[[Simulation], PropagationProblem] | None = None,
    options: ExecutionOptions | None = None,
) -> None:
    """Compare a backend's reading of the edge-reaching launch with the reference's.

    The compiled backend's is its native state's own reduction, as its session reads
    it at a check (:func:`backends._session_probe_b1.native_reading`).
    """
    simulation = probe(route, scheme)
    translated = None if problem is None else problem(simulation)
    reference = watched(simulation, "numpy_ref", translated, options)
    try:
        theirs = reading(reference)
    finally:
        reference.close()
    ours = _launch_reading(simulation, backend, translated, options)
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    largest = monitor.weights.max(axis=1)
    # Light in every region a weight reaches, so that each sum is compared on
    # something; without a collar the rate's row is zero.
    assert np.all(theirs.regions[largest > 0.0] > 1e-6 * theirs.total)
    # A device build adds its transform's declared envelope.
    bound = ladder_bound(backend, _rounding(simulation), simulation.grid.n_t, 1)
    assert ours.total == pytest.approx(theirs.total, rel=bound)
    # Each row's sum is bounded by its largest weight times the energy: the rate's
    # row is energy per metre.
    scale = bound * theirs.total * largest
    weighted = largest > 0.0
    np.testing.assert_array_less(
        np.abs(ours.regions - theirs.regions)[weighted], scale[weighted]
    )
    # A row no weight reaches sums nothing.
    assert np.all(ours.regions[~weighted] == 0.0)
    # Nothing was taken before the first step.
    assert ours.removed == theirs.removed == 0.0


def _built(
    simulation: Simulation,
    backend: str,
    problem: PropagationProblem | None = None,
    options: ExecutionOptions | None = None,
) -> Any:
    """Return a case's backend, selected as ``monitor_cases.watched`` selects it."""
    problem = simulation.translate() if problem is None else problem
    return select(
        backend,
        problem,
        simulation._run_request(),
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options() if options is None else options,
    )


def _launch_reading(
    simulation: Simulation,
    backend: str,
    problem: PropagationProblem | None = None,
    options: ExecutionOptions | None = None,
) -> BoundaryEnergies:
    """Return a native backend's reading of the case's edge-reaching launch."""
    problem = simulation.translate() if problem is None else problem
    made = _built(simulation, backend, problem, options)
    try:
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        bound = bind_native_monitor(made, problem, monitor)
        launch = with_copy(problem.initial.field0, simulation.grid.n_t)
        return native_reading(made, bound, launch)
    finally:
        made.close()


def _check_split(backend: str, route: str, march: str) -> None:
    """Compare a backend's split removal over a fixed march with the reference's."""
    simulation = probe(route, "split_step", MARCHES[march])
    theirs = record_after_march(simulation, "numpy_ref")
    ours = record_after_march(simulation, backend)
    assert theirs["raw_removed_fraction"] > 1e-6
    # Each step's removal carries the reading's rounding, and the matched steps' states
    # agree to the march's own rounding, amplified over the steps; a device build adds
    # its transforms' declared envelope, eight round trips a step at most.
    bound = ladder_bound(
        backend, 1e3 * _rounding(simulation), simulation.grid.n_t, 8 * STEPS
    )
    # The guards are judged on what each step's split found (`split_sums`): the copy's
    # tail puts 3e-15 to 5e-12 of the energy there, whose samples' rounding is a
    # relative 1e-8 of it at most, far inside the bound.
    assert theirs["guard_high_max"] > 1e-15
    names = ("raw_removed_fraction", "removed_fraction", "collar_max", "guard_high_max")
    for name in names:
        assert ours[name] == pytest.approx(theirs[name], rel=bound), name
    assert ours["steps"] == theirs["steps"]
    # The crossing cap is a bucket's: states agreeing to rounding fill the same
    # buckets, but a share that sits on the 1e-6 threshold may tip into the next
    # one, a factor of 10**(9/128) apart.
    assert math.isfinite(theirs["step_cap"])
    assert ours["step_cap"] == pytest.approx(theirs["step_cap"], rel=0.18)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("backend", COMPILED)
def test_a_check_reads_the_reference_s_five_numbers(
    backend: str, route: str, scheme: str
) -> None:
    _require(backend)
    _checkreading(backend, route, scheme)


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("backend", COMPILED)
def test_the_split_s_removal_is_the_reference_s(
    backend: str, route: str, march: str
) -> None:
    # Every accepted step's removal, added up by the backend's own finish on each
    # march; the steps are fixed, so the backends take the same ones.
    _require(backend)
    _check_split(backend, route, march)


def _require_device() -> None:
    """Skip unless the installed native build runs on an accelerator."""
    _require("native")
    if device_backend("native") == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")


@pytest.mark.gpu
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("route", ROUTE_NAMES)
def test_a_device_reads_the_reference_s_five_numbers(route: str, scheme: str) -> None:
    # The reduction runs in the device's memory space; only the numbers cross.
    _require_device()
    _checkreading("native", route, scheme)


@pytest.mark.gpu
@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("route", ROUTE_NAMES)
def test_a_device_s_split_removal_is_the_reference_s(route: str, march: str) -> None:
    _require_device()
    _check_split("native", route, march)


@pytest.mark.parametrize("backend", ["numpy_ref", *COMPILED])
def test_a_reading_needs_a_monitor(backend: str) -> None:
    _require(backend)
    simulation = probe("gnlse", "nonlinear")
    problem = simulation.translate()
    run_request = simulation._run_request()
    made = select(backend, problem, run_request, grid=simulation.grid)
    try:
        if backend == "numpy_ref":
            driver = as_driver(made)
            driver.set_initial(problem.initial.field0, None)
            with pytest.raises(BackendError, match="no boundary monitor"):
                driver._boundary_energies(driver._require_state())
        else:
            # The native state's own reduction, which its session reads at a check.
            launched_state(made, problem.initial.field0)
            with pytest.raises(RuntimeError, match="no boundary monitor is bound"):
                cast("Any", made)._native_state.monitor_reading()
    finally:
        made.close()


@pytest.mark.parametrize("backend", ["numpy_ref", *COMPILED])
def test_a_monitor_for_another_grid_is_refused(backend: str) -> None:
    # Its weights would be read out of bounds by the compiled sums, so every backend
    # refuses them by name when the monitor is attached.
    _require(backend)
    simulation = probe("gnlse", "nonlinear")
    problem = simulation.translate()
    run_request = simulation._run_request()
    other = probe("capillary (envelope)", "nonlinear")._boundary_monitor()
    assert other is not None
    made = select(backend, problem, run_request, grid=simulation.grid)
    try:
        if backend == "numpy_ref":
            driver = as_driver(made)
            driver.set_initial(problem.initial.field0, None)
            with pytest.raises(BackendError, match="do not match the grid"):
                driver.attach_monitor(other)
        else:
            # The native session binds the monitor where its state lives, which
            # refuses the weights' extent before any reading.
            refused = "monitor weights has the wrong extent"
            with pytest.raises(ValueError, match=refused):
                session_run(made, problem, run_request.for_run(monitor=other))
    finally:
        made.close()


PARTIAL = ExecutionOptions(node_chunk=50)
"""Node chunks that leave a partial one on the Cartesian case."""


def _chunked(simulation: Simulation) -> PropagationProblem:
    """Return the Cartesian case's problem, which runs in :data:`PARTIAL` chunks."""
    return simulation.translate()


def _bare(simulation: Simulation) -> PropagationProblem:
    """Return a route's problem without its temporal collar."""
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber=bare_absorber(simulation.grid)
    )
    return simulation.translate()


@pytest.mark.parametrize(
    ("route", "problem"),
    [("cartesian", _chunked), ("gnlse", _bare), ("capillary (multimode)", _bare)],
    ids=["partial-chunk", "single-column-no-collar", "modal-no-collar"],
)
@pytest.mark.parametrize("backend", COMPILED)
def test_a_check_reads_the_reference_s_numbers_at_the_edges(
    backend: str, route: str, problem: Callable[[Simulation], PropagationProblem]
) -> None:
    # A last chunk shorter than the rest, and a run with no collar, for which the
    # backends plan their own coarse transform at the first attach.
    _require(backend)
    _checkreading(
        backend, route, "nonlinear", problem, PARTIAL if problem is _chunked else None
    )


@pytest.mark.parametrize("backend", ["numpy_ref", *COMPILED])
def test_a_second_watched_run_reads_as_the_first(backend: str) -> None:
    # A runner loads its backend again for the next input and attaches a new monitor:
    # the views the first allocated are reused, and nothing taken carries over.
    _require(backend)
    simulation = probe("radial", "split_step", MARCHES["runge_kutta"])
    if backend == "numpy_ref":
        first, again, ledger = _host_second_run(simulation)
    else:
        first, again, ledger = _native_second_run(simulation, backend)
    assert again.total == first.total
    np.testing.assert_array_equal(again.regions, first.regions)
    assert again.removed == 0.0
    assert all(size >= 0 for size in ledger.values())


def _host_second_run(
    simulation: Simulation,
) -> tuple[BoundaryEnergies, BoundaryEnergies, dict[str, int]]:
    """The reference's launch reading, a march, and its reading of a second load."""
    made = watched(simulation, "numpy_ref")
    try:
        driver = as_driver(made)
        first = reading(made)
        made.propagate_to(LENGTH)
        problem = simulation.translate()
        driver.set_initial(with_copy(problem.initial.field0, simulation.grid.n_t), None)
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        driver.attach_monitor(monitor)
        return first, reading(made), made.memory_ledger()
    finally:
        made.close()


def _native_second_run(
    simulation: Simulation, backend: str
) -> tuple[BoundaryEnergies, BoundaryEnergies, dict[str, int]]:
    """A native launch reading, a watched session to the end, and a second reading.

    The session binds its own monitor on the backend's state and marches it; the
    second reading binds a fresh monitor, as the next run's session does.
    """
    problem = simulation.translate()
    launch = with_copy(problem.initial.field0, simulation.grid.n_t)
    made = _built(simulation, backend, problem)
    try:
        first = native_reading(
            made, _fresh_bound(made, simulation, problem), launch.copy()
        )
        boundary = simulation._boundary_monitor()
        assert boundary is not None
        session_run(
            made,
            problem,
            made.request.for_run(monitor=boundary),
            Start(field=launch.copy(), z=0.0),
        )
        again = native_reading(
            made, _fresh_bound(made, simulation, problem), launch.copy()
        )
        return first, again, made.memory_ledger()
    finally:
        made.close()


def _fresh_bound(
    made: Any, simulation: Simulation, problem: PropagationProblem
) -> BoundaryMonitor:
    """Bind a case's fresh monitor in a native backend's state; return it as bound."""
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    return bind_native_monitor(made, problem, monitor)


@pytest.mark.parametrize(
    ("backend", "route", "rows"),
    [
        # The general engine reads through its statistics buffers: only its weights.
        ("native", "gnlse", ("monitor_weights",)),
    ],
    ids=["native"],
)
def test_a_monitor_s_own_transform_is_in_the_ledger(
    backend: str, route: str, rows: tuple[str, ...]
) -> None:
    # Without a collar the backend has no coarse pair: what the monitor allocates at
    # its first attach is counted with every other buffer.
    _require(backend)
    simulation = probe(route, "nonlinear")
    problem = _bare(simulation)
    made = _built(simulation, backend, problem)
    try:
        # What the native session binds for its monitor, where the state lives.
        _fresh_bound(made, simulation, problem)
        ledger = made.memory_ledger()
    finally:
        made.close()
    assert all(ledger[row] > 0 for row in rows), {row: ledger[row] for row in rows}


class _CountingState:
    """A native state that counts its `bind_occupancy` calls."""

    def __init__(self, state: object) -> None:
        """Wrap `state`."""
        self.__dict__["_state"] = state
        self.__dict__["binds"] = 0

    def bind_occupancy(self, *args: object) -> None:
        """Count, then bind."""
        self.__dict__["binds"] += 1
        self.__dict__["_state"].bind_occupancy(*args)

    def __getattr__(self, name: str) -> object:
        """Hand everything else to the state."""
        return getattr(self.__dict__["_state"], name)


def test_a_second_point_keeps_its_bucket_table_and_reads_as_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The table is a sixteenth of a state, and a runner's points share it: a second
    # upload would move it per point. Nothing else may carry over. Each point is a
    # watched session on the one backend, which binds its fresh monitor there.
    _require("native")
    simulation = probe("radial", "split_step", MARCHES["runge_kutta"])
    problem = simulation.translate()
    made = cast("Any", _built(simulation, "native", problem))
    launch = with_copy(problem.initial.field0, simulation.grid.n_t)
    records = []
    try:
        for point in range(2):
            boundary = simulation._boundary_monitor()
            assert boundary is not None
            if point == 1:
                assert made._bound_buckets is not None
                counting = _CountingState(made._native_state)
                bind = made._bind_monitor

                def counted(monitor, *, bind=bind, counting=counting):
                    # The counting state stands in while the session binds alone.
                    state, made._native_state = made._native_state, counting
                    try:
                        bind(monitor)
                    finally:
                        made._native_state = state

                monkeypatch.setattr(made, "_bind_monitor", counted)
            record: dict[str, object] = {}
            session_run(
                made,
                problem,
                made.request.for_run(monitor=boundary),
                Start(field=launch.copy(), z=0.0),
                monitor=record,
            )
            records.append(record)
        assert counting.binds == 0  # pyright: ignore[reportPossiblyUnboundVariable]
    finally:
        made.close()
    # The same launch and march on the same build: the second record is a
    # structural repeat of the first.
    first, again = records
    assert first.keys() == again.keys()
    for name in first:
        assert again[name] == first[name], name


@pytest.mark.parametrize("count", [0, 129])
def test_an_occupancy_outside_its_reduction_is_refused(count: int) -> None:
    _require("native")
    simulation = probe("radial", "split_step", MARCHES["runge_kutta"])
    problem = simulation.translate()
    made = cast("Any", _built(simulation, "native", problem))
    try:
        table = np.zeros(math.prod(problem.shape[0:1]) * simulation.grid.n_omega)
        with pytest.raises(ValueError, match="between 1 and 128 buckets"):
            made._native_state.bind_occupancy(table.astype(np.uint8), count)
    finally:
        made.close()


@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("backend", COMPILED)
def test_successive_occupancies_are_the_reference_s(backend: str, route: str) -> None:
    # Each reading of the state's energy per walk-off bucket stands alone. A host
    # build's scatter once added every reading into the last, and the third read three
    # times the first; a reduction starts from nothing, and this holds it there.
    # The native state reads each of the reference's states in turn, on one bound
    # monitor, as its session reads its own at successive checks.
    _require(backend)
    simulation = probe(route, "split_step", MARCHES["runge_kutta"])
    problem = simulation.translate()
    theirs = watched(simulation, "numpy_ref")
    ours = _built(simulation, backend, problem)
    # The matched march's rounding, amplified over its steps, relative to the energy.
    bound = ladder_bound(
        backend, 1e3 * _rounding(simulation), simulation.grid.n_t, 8 * STEPS
    )
    try:
        monitor = _fresh_bound(ours, simulation, problem)
        for third in (1, 2, 3):
            theirs.propagate_to(LENGTH * third / 3)
            expected = reading(theirs).occupancy
            state = as_driver(theirs)._require_state()[0]
            got = native_reading(ours, monitor, np.array(state)).occupancy
            assert expected is not None and got is not None
            np.testing.assert_allclose(
                got, expected, rtol=0.0, atol=bound * float(expected.sum())
            )
    finally:
        theirs.close()
        ours.close()
