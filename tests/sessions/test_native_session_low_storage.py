# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage replay in the native session, against the driver's (SEP 47 6.3).

The native session owns the low-storage loop: each step is one in-place native step
whose stage scalars the native state forms from its tables, and the replay is the
native Runge-Kutta march's. The transitional
:class:`~solprop.backends.driver_session.DriverSession` over the native backend makes
the same native passes from the scalars solprop forms on the host, one return to Python
a step. Their traces and fields are bitwise the same on one thread
(``test_native_session.py``'s ``low-storage`` case, and the lifecycle suite's
``native-session-low-storage`` executor); this file checks what is the low-storage
march's own: that no step returns to Python, and that a step left in flight -- by a
native failure partway, or by a divergence -- poisons the session where the driver's
is poisoned, and the service loop falls back to the latest running checkpoint.
"""

import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from sessions.executors import (
    BETA2,
    GAMMA,
    PLANES,
    TAU0,
    WAVELENGTH,
    Run,
    Started,
    drive,
    entrance,
    fibre,
    trace,
    window,
)
from solprop.backends.driver import StepDriver
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import Eligibility, Outcome
from solprop.backends.registry import select
from solprop.exceptions import ConvergenceError
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.run_request import describe_run

native_session = pytest.importorskip("solprop_native.session")
_counters = pytest.importorskip("solprop_native._counters")

LENGTH = PLANES[-1]
STEPS = 40
FAILING = 6
"""The step a native failure abandons partway: the third after the first plane."""


def _one_thread() -> bool:
    """Return whether the native backend runs one thread in this process.

    Returns
    -------
    bool
        True when ``SOLPROP_NATIVE_THREADS`` or ``SOLPROP_THREADS`` is ``"1"``.
    """
    import os  # noqa: PLC0415 - beside its one use

    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


def _replayed(scale: float = 1.0) -> KernelProblem:
    """Return the soliton replayed by the low-storage march, a row every step.

    Parameters
    ----------
    scale : float, optional
        The peak power in units of the fundamental soliton's: a thousand diverges.

    Returns
    -------
    solprop.translate.problem.KernelProblem
        The problem.
    """
    grid = window()
    peak = scale * abs(BETA2) / (GAMMA * TAU0**2)
    return assemble(
        grid,
        fibre().mode,
        [fibre().kerr],
        SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid),
        save_planes=list(PLANES),
        controller=PIController(rtol=1e-4, atol=1e-16),
        dz_max=LENGTH,
        dz_init=1e-3,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        stats_spec=StatsSpec(
            names=("energy", "peak_power"), periodic=("energy",), period=1
        ),
    )


class _Faulted(native_session.NativeSession):
    """A native session whose state fails partway through its step ``after + 1``.

    The native engine's injected failure, raised after the step's passes have
    overwritten the state and before its finish (``ResidentLowStorage``).

    Parameters
    ----------
    *args, **kwargs
        As :class:`~solprop_native.session.NativeSession` takes them.
    after : int
        The steps that complete before the failing one.
    """

    def __init__(self, *args: Any, after: int, **kwargs: Any) -> None:
        """Hold the failing step, then start the session."""
        self._after = after
        super().__init__(*args, **kwargs)

    def _prepare(self, backend, start):
        """Prepare as the session does, then arm the native failure."""
        prepared = super()._prepare(backend, start)
        backend._native_state.adams_march_fault("trial", self._after, "backend")
        return prepared


def _native(problem: KernelProblem, *, after: int | None = None) -> Any:
    """Start a native session on a problem, failing partway through a step if asked.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    after : int or None, optional
        The steps that complete before a failing one; ``None`` for no failure.

    Returns
    -------
    solprop_native.session.NativeSession
        The session.
    """
    context: dict[str, Any] = {"grid": window(), "modes": [fibre().mode]}
    if after is None:
        return native_session.NativeSession(
            problem, describe_run(problem), entrance(problem), **context
        )
    return _Faulted(
        problem, describe_run(problem), entrance(problem), after=after, **context
    )


class _AbandonedError(RuntimeError):
    """What the driver's engine raises partway through a step."""


def _driver(problem: KernelProblem, *, failing: int | None = None) -> DriverSession:
    """Start the driver session over the native backend, failing a step if asked.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    failing : int or None, optional
        The step whose native passes complete and whose engine then raises, as the
        native failure does; ``None`` for no failure.

    Returns
    -------
    solprop.backends.driver_session.DriverSession
        The session.
    """
    driver = select("native", problem, grid=window(), modes=[fibre().mode])
    if failing is not None:
        march = cast("StepDriver[Any]", driver)._low_storage
        assert march is not None
        engine = march.engine
        engine.step = _FailingStep(engine.step, failing)
    return DriverSession(driver, problem, describe_run(problem), entrance(problem))


class _FailingStep:
    """An engine step that raises after the step `failing` has made its passes."""

    def __init__(self, step: Any, failing: int) -> None:
        """Wrap the engine's step."""
        self._step = step
        self._failing = failing
        self._calls = 0

    def __call__(self, z: float, dz: float) -> float:
        """Take the step; raise once the failing one has overwritten the state."""
        self._calls += 1
        squared = self._step(z, dz)
        if self._calls == self._failing:
            raise _AbandonedError
        return squared


def _closed(session: Any, **drive_options: Any) -> Run:
    """Drive a session to its end, then close it.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The session.
    **drive_options
        Passed to :func:`sessions.executors.drive`.

    Returns
    -------
    sessions.executors.Run
        The run.
    """
    try:
        return drive(Started(session), **drive_options)
    finally:
        session.close()


def test_no_step_returns_to_python():
    """SEP 47 sections 6.3 and 7.3: the stage scalars and the loop are native.

    The driver session returns to Python once a low-storage step, to form its scalars
    and book it (the counters' ``low_storage_step``); the native session returns only
    at its observations and its end, its rows taken natively.
    """
    problem = _replayed()
    counters = _counters.enable()
    try:
        _closed(_driver(problem))
    finally:
        _counters.disable()
    assert counters.record()["native_returns"] == {"low_storage_step": STEPS}
    session = _native(problem)
    counters = _counters.enable()
    try:
        run = drive(Started(session))
        status = session.native.status()
    finally:
        _counters.disable()
        session.close()
    record = counters.record()
    assert record["native_returns"] == {}
    assert record["low_storage_steps"] == record["host_attempts"] == 0
    assert record["statistics_rows"] == record["statistics_evaluations"] == 0
    assert len(run.batches) == len(run.observations) + 1 == status.batches
    rows = sum(event.step_stats.n_rows for event in run.observations)
    assert rows == status.rows_taken == run.attempts == STEPS
    assert not run.histories[-1].n_steps
    assert np.all(np.concatenate([one.error_norm for one in run.histories]) == 0.0)
    assert status.march_calls >= STEPS


def test_a_failure_partway_through_a_step_poisons_the_session():
    """SEP 47 section 5.5: a step abandoned in flight claims no checkpoint.

    The native failure strikes after the step's passes have overwritten the state:
    the terminal event is poisoned, with no ticket, at the last committed endpoint,
    with the attempts before it, as the driver's in-flight failure is reported.
    """
    problem = _replayed()
    pending: list[Any] = []
    run = _closed(
        _native(problem, after=FAILING - 1),
        between=lambda session, batch: pending.append(session.pending_ticket()),
    )
    terminal = run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert terminal.eligibility is Eligibility.POISONED
    assert terminal.ticket is None and pending[-1] is None
    assert isinstance(terminal.error, RuntimeError)
    assert "injected native failure" in str(terminal.error)
    assert run.status.attempts == FAILING - 1
    assert terminal.z == run.accepted_ends()[-1]
    baseline = _closed(_driver(problem, failing=FAILING))
    assert isinstance(baseline.terminal.error, _AbandonedError)
    if _one_thread():
        assert trace(run) == trace(baseline)


def test_a_diverged_step_poisons_the_session_where_the_driver_s_does():
    """A replay cannot reject a step: the divergence is found in the accepted state.

    The state was overwritten in place, so the terminal event is poisoned; the error
    is the low-storage march's own.
    """
    problem = _replayed(1e3)
    run = _closed(_native(problem))
    terminal = run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert terminal.eligibility is Eligibility.POISONED
    assert terminal.ticket is None
    assert isinstance(terminal.error, ConvergenceError)
    assert "non-finite state" in str(terminal.error)
    assert 0 < run.status.attempts < STEPS
    if not _one_thread():
        return
    baseline = _closed(_driver(problem))
    assert trace(run) == trace(baseline)
    assert str(terminal.error) == str(baseline.terminal.error)


# A soliton in a tight window, without a collar: its edges are not what is tested here.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")
def test_a_poisoned_session_falls_back_to_the_latest_running_checkpoint(
    monkeypatch, tmp_path
):
    """SEP 47 section 5.5: the latest running checkpoint, not the last saved plane.

    A snapshot due before the first step is serviced at its end as a running
    checkpoint; the third step fails partway. The service loop records the failure
    with that checkpoint, from which a resume continues to the uninterrupted run's
    planes.
    """
    from solprop.backends import sessions  # noqa: PLC0415
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.output import SnapshotController, open_file  # noqa: PLC0415

    steps = 8
    length = 0.02
    schedule = FixedSchedule.uniform(0.0, length, length / steps, save_planes=[0.01])

    def simulation() -> Simulation:
        built = Simulation.from_gnlse(
            0.11, length, [0.0, 0.0, -11.83e-27], wavelength=835e-9,
            wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=100e-15,
            peak_power=9.8, shape="sech", save_planes=[0.01, length],
            schedule=schedule, integrator=LowStorageRungeKutta(), backend="native",
            dz_max=schedule.max_step, dz_init=schedule.max_step, absorber=False,
            noise=None,
        )  # fmt: skip
        built.settings = dataclasses.replace(built.settings, fields="all")
        return built

    run = simulation()

    def native(backend, problem, request, start, *, listener=None, stop=None, **_):
        del backend
        return _Faulted(
            problem,
            request,
            entrance(problem) if start is None else start,
            after=2,
            grid=run.grid,
            modes=run.modes,
            profiles=run.profiles,
            stop=stop,
            listener=listener,
        )

    path = tmp_path / "run.h5"
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    monkeypatch.setattr(sessions, "start_session", native)
    with pytest.raises(RuntimeError, match="injected native failure") as caught:
        run.run(path, progress="none", snapshots=controller)
    notes = " ".join(caught.value.__notes__)
    assert "abandoned in flight" in notes
    assert "falls back to its latest checkpoint" in notes
    with open_file(path) as stored:
        checkpoint = stored.terminal_checkpoint
        assert checkpoint is not None and checkpoint.plane.z == length / steps
        assert stored.outcome is not None and stored.outcome.status == "failed"
        assert stored.outcome.z == checkpoint.plane.z
        assert stored["z"].tolist() == [0.0]
    monkeypatch.undo()
    simulation().run(path, progress="none", resume=True)
    with open_file(path) as stored:
        assert stored["z"].tolist() == [0.0, 0.01, length]
