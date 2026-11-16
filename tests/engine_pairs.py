# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Both backends' engines over a loaded workspace, through solprop's protocols.

SEP 52 section 6.5: a parity case is written once, against
:class:`~solprop.backends.engines.RungeKuttaEngine` and
:class:`~solprop.backends.engines.AdamsEngine`, and run on each backend's engines. The
reference's are its own (:class:`~solprop.backends.numpy_ref.engines.NumpyRungeKutta`,
:class:`~solprop.backends.numpy_ref.engines.NumpyLowStorage`,
:class:`~solprop.backends.numpy_ref.multistep.HostHistory`) over its workspace;
native's are its workspace's C++ engines, reached through the private test module
(``_testing.engines``), whose prefixed methods :class:`NativeRungeKutta` and
:class:`NativeAdams` dispatch onto the protocols' names. Each backend is loaded with the
problem's launch, configured with the run's records and, when asked, watched by the
case's boundary monitor, as a session's start leaves it.

- :func:`opened` builds one backend's :class:`Opened` engines for a simulation;
- :func:`record_adams` records the Adams calls the reference's march makes on a run, a
  script :func:`play_adams` replays on either backend's engine.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from builders import load_reference
from sessions.runs import bare, bind_native_monitor, load_native, run
from solprop.backends.adams_core import Stencil
from solprop.backends.engines import (
    AcceptRecord,
    AdamsEngine,
    AttemptOutcome,
    BoundaryReading,
    HistoryEntry,
    RungeKuttaEngine,
)
from solprop.backends.records import run_records
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.monitor import BoundaryMonitor
from solprop.stepper import LowStorageRungeKutta, Multistep

__all__ = [
    "AdamsCall",
    "NativeAdams",
    "NativeRungeKutta",
    "Opened",
    "opened",
    "play_adams",
    "record_adams",
]


class NativeRungeKutta:
    """Native's Runge-Kutta or low-storage engine as a ``RungeKuttaEngine``."""

    def __init__(self, engines: Any) -> None:
        self._engines = engines

    def trial(self, z: float, dz: float) -> float:
        """Form a trial; return its scaled norm (0.0 for the low-storage step)."""
        return float(self._engines.rk_trial(z, dz))

    def accept(self, dz: float) -> AcceptRecord:
        """Finish and commit the trial."""
        return self._engines.rk_accept(dz)

    def extend(self, z: float, dz: float) -> None:
        """Hold the trial as a dense extension."""
        self._engines.rk_extend(z, dz)

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading."""
        return self._engines.rk_reading()

    @property
    def in_flight(self) -> bool:
        """Whether a step that overwrites the state was left partway."""
        return bool(self._engines.rk_in_flight)


class NativeAdams:
    """Native's Adams engine as an ``AdamsEngine``."""

    def __init__(self, engines: Any) -> None:
        self._engines = engines

    def start(self, z: float) -> None:
        """Evaluate the history's first sample into row 0."""
        self._engines.adams_start(z)

    def trial(self, z: float, dz: float, stencil: Stencil) -> np.ndarray:
        """Form a trial; return its scored orders' norms."""
        return np.asarray(self._engines.adams_trial(z, dz, stencil), dtype=np.float64)

    def accept(
        self, dz: float, free: int, stencil: Stencil, kept: Sequence[HistoryEntry]
    ) -> AcceptRecord:
        """Adopt the held trial into row `free`."""
        return self._engines.adams_accept(dz, free, stencil, list(kept))

    def chord_phase(self, z1: float, z2: float) -> float:
        """Return the carrier chord's largest phase over the interval, rad."""
        return float(self._engines.adams_chord_phase(z1, z2))

    def step_carrier(self, order: int) -> np.ndarray:
        """Return the last trial's packed carrier rotations."""
        return np.asarray(self._engines.adams_step_carrier(order), dtype=np.float64)

    def begin_attempt(self, cost: int) -> None:
        """Reserve an attempt's host-source calls."""
        self._engines.adams_begin_attempt(cost)

    def end_attempt(self, outcome: AttemptOutcome) -> None:
        """End an attempt."""
        self._engines.adams_end_attempt(int(outcome))

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading."""
        return self._engines.adams_reading()


@dataclass
class Opened:
    """One backend's engines over its loaded, configured workspace.

    ``field`` reads the committed state and ``stage`` a Runge-Kutta trial's stage
    register, each in the problem's shape; ``monitor`` is the monitor as bound, or
    ``None`` when the case is not watched.
    """

    backend: str
    shape: tuple[int, ...]
    runge_kutta: RungeKuttaEngine | None
    adams: AdamsEngine | None
    field: Callable[[], np.ndarray]
    stage: Callable[[int], np.ndarray]
    monitor: BoundaryMonitor | None


@contextmanager
def opened(
    simulation: Simulation, backend: str, *, watched: bool = True
) -> Iterator[Opened]:
    """Open `backend` on a simulation, its engines over the loaded launch.

    The reference's engines are the ones its next run would build; native's are its
    workspace's, leased for the block (no session can start meanwhile).
    """
    problem = simulation.translate()
    request = simulation._run_request()
    made = cast(
        "Any",
        select(
            backend,
            problem,
            request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
            options=simulation.settings.execution_options(),
        ),
    )
    shape = tuple(problem.shape)
    integrator = request.stepper.integrator
    monitor = simulation._boundary_monitor() if watched else None
    try:
        if backend == "numpy_ref":
            yield _reference(made, problem, integrator, monitor, shape)
        else:
            yield from _native(made, problem, request, integrator, monitor, shape)
    finally:
        made.close()


def _reference(
    made: Any, problem: Any, integrator: Any, monitor: Any, shape: tuple[int, ...]
) -> Opened:
    """The reference's engines over its workspace, loaded and watched."""
    load_reference(made)
    workspace = made.workspace
    if monitor is not None:
        monitor = dataclasses.replace(
            monitor, crossing=problem.split_crossing if monitor.split else None
        )
        workspace.bind_monitor(monitor)
    runge_kutta: Any = None
    adams: Any = None
    if isinstance(integrator, Multistep):
        adams = made.march._engine
    elif isinstance(integrator, LowStorageRungeKutta):
        runge_kutta = made.march.engine
    else:
        runge_kutta = made._runge_kutta(workspace)

    def stage(index: int) -> np.ndarray:
        return np.asarray(runge_kutta.stages[index]).reshape(shape)

    return Opened(
        "numpy_ref",
        shape,
        runge_kutta,
        adams,
        lambda: np.array(workspace.committed()).reshape(shape),
        stage,
        monitor,
    )


def _native(
    made: Any,
    problem: Any,
    request: Any,
    integrator: Any,
    monitor: Any,
    shape: tuple[int, ...],
) -> Iterator[Opened]:
    """Native's engines over its workspace, loaded, configured and watched."""
    _testing = pytest.importorskip("solprop_native._core")._testing
    workspace = load_native(made, problem.initial.field0)
    _testing.configure_march(
        workspace, run_records(problem, request, Start.entrance(problem))
    )
    if monitor is not None:
        monitor = bind_native_monitor(made, problem, monitor)
    engines = _testing.engines(workspace)
    try:
        adams = NativeAdams(engines) if isinstance(integrator, Multistep) else None
        yield Opened(
            "native",
            shape,
            None if adams is not None else NativeRungeKutta(engines),
            adams,
            lambda: np.asarray(engines.field()).reshape(shape),
            lambda index: np.asarray(engines.stage(index)).reshape(shape),
            monitor,
        )
    finally:
        engines.close()


@dataclass(frozen=True)
class AdamsCall:
    """One call an Adams march made on its engine: the method's name and arguments."""

    name: str
    args: tuple[Any, ...]


def record_adams(
    simulation: Simulation, monkeypatch: pytest.MonkeyPatch, *, accepts: int
) -> list[AdamsCall]:
    """Record the calls the reference's Adams march makes, up to `accepts` accepts.

    A run of the simulation on the reference, the calls of its
    :class:`~solprop.backends.numpy_ref.multistep.HostHistory` copied as they are
    made: the startup, every trial (the rejected ones too), every accept with the
    history it keeps, and each carrier read.
    """
    from solprop.backends.numpy_ref.multistep import HostHistory  # noqa: PLC0415

    calls: list[AdamsCall] = []

    def spy(name: str, copy: Callable[..., tuple[Any, ...]]) -> None:
        original = getattr(HostHistory, name)

        def recorded(engine: Any, *args: Any) -> Any:
            calls.append(AdamsCall(name, copy(*args)))
            return original(engine, *args)

        monkeypatch.setattr(HostHistory, name, recorded)

    def stencil(value: Stencil) -> Stencil:
        return Stencil(
            list(value.nodes), np.array(value.rows), value.order, list(value.orders)
        )

    spy("start", lambda z: (float(z),))
    spy("trial", lambda z, dz, held: (float(z), float(dz), stencil(held)))
    spy(
        "accept",
        lambda dz, free, held, kept: (float(dz), int(free), stencil(held), list(kept)),
    )
    spy("chord_phase", lambda z1, z2: (float(z1), float(z2)))
    spy("step_carrier", lambda order: (int(order),))
    problem = simulation.translate()
    request = simulation._run_request()
    made = select(
        "numpy_ref",
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options(),
    )
    run(made, problem, bare(request), owns=True, export=None)
    monkeypatch.undo()
    taken = 0
    for index, call in enumerate(calls):
        taken += call.name == "accept"
        if taken == accepts:
            return calls[: index + 1]
    return calls


def play_adams(opened: Opened, calls: Sequence[AdamsCall]) -> list[tuple[str, Any]]:
    """Replay recorded Adams calls on one backend's engine; what each returned.

    An accept's entry carries its :class:`AcceptRecord` and the committed state after
    it; a startup returns nothing and is not listed.
    """
    engine = opened.adams
    assert engine is not None, "the case is not an Adams march"
    found: list[tuple[str, Any]] = []
    for call in calls:
        value = getattr(engine, call.name)(*call.args)
        if call.name == "accept":
            found.append((call.name, (value, opened.field())))
        elif call.name != "start":
            found.append((call.name, value))
    return found
