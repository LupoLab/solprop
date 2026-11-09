# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native execution session against the driver over the same native marches.

SEP 47 section 7.1 asks for identical-input control traces: given the same numerical
inputs, the native session's accept, reject and landing decisions, its periodic rows,
its plane statistics and its exported fields equal those of the transitional
:class:`~solprop.backends.driver_session.DriverSession` over the native backend, which
runs the same strict marches and the same kernels. On one thread they are bitwise the
same. This is not the reference comparison of section 6.6: both sides are native, and
the lifecycle suite holds each session to the contract alone.

The file also checks what the prototype refuses (by name, before allocating), the
structural counter of section 7.3 -- no return to Python caused by periodic rows -- and
a stop from another thread reaching a march in progress.
"""

import threading
import time

import numpy as np
import pytest

from sessions.executors import (
    CONSUMER,
    PLANES,
    STORED,
    Run,
    Started,
    assert_same_run,
    drive,
    entrance,
    fibre,
    soliton,
    spy_monitor,
    window,
)
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import ObservationReady, Outcome, Progress, Terminal
from solprop.backends.registry import select
from solprop.backends.session import StopToken
from solprop.exceptions import BackendError, StopRequested
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.run_request import describe_run

native_session = pytest.importorskip("solprop_native.session")

LENGTH = PLANES[-1]


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


def _case(  # noqa: PLR0913 - one choice per keyword
    *,
    integrator: Multistep | LowStorageRungeKutta | None = None,
    schedule: FixedSchedule | None = None,
    step_planes: int = 3,
    period: int = 1,
    names: tuple[str, ...] = ("energy", "peak_power", "energy_window"),
    rtol: float = 1e-4,
    interpolate: bool = False,
    controller: PIController | None = None,
    dz_init: float = 1e-3,
    dz_max: float = LENGTH,
) -> KernelProblem:
    """Return the soliton of the lifecycle suite with the method and rows chosen.

    Parameters
    ----------
    integrator : Multistep, LowStorageRungeKutta or None, optional
        The method; the default tableau by default.
    schedule : solprop.stepper.schedule.FixedSchedule or None, optional
        A replayed itinerary.
    step_planes : int, optional
        Observe every ``step_planes``-th accepted step.
    period : int, optional
        The periodic rows' period, accepted steps.
    names : tuple of str, optional
        The plane statistics; ``energy`` and ``peak_power`` are periodic.
    rtol : float, optional
        The controller's tolerance.
    interpolate : bool, optional
        Whether planes are interpolated (dense output).
    controller : PIController or None, optional
        The controller; ``PIController(rtol, 1e-16)`` by default.
    dz_init : float, optional
        The first proposal, m.
    dz_max : float, optional
        The step ceiling, m.

    Returns
    -------
    solprop.translate.problem.KernelProblem
        The problem.
    """
    grid = window()
    from sessions.executors import BETA2, GAMMA, TAU0, WAVELENGTH  # noqa: PLC0415
    from solprop.pulses import SechPulse  # noqa: PLC0415 - beside its one use

    peak = abs(BETA2) / (GAMMA * TAU0**2)
    return assemble(
        grid,
        fibre().mode,
        [fibre().kerr],
        SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid),
        save_planes=list(PLANES),
        controller=controller or PIController(rtol=rtol, atol=1e-16),
        dz_max=dz_max,
        dz_init=dz_init,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        step_planes=step_planes,
        integrator=integrator,
        schedule=schedule,
        interpolate_planes=interpolate,
        stats_spec=StatsSpec(
            names=names,
            periodic=("energy", "peak_power"),
            period=period,
            energy_windows=np.array([[0.99 * grid.omega0, 1.01 * grid.omega0]])
            if "energy_window" in names
            else np.empty((0, 2)),
        ),
    )


def _driver_session(problem, request, **controls) -> Started:
    """Start the transitional driver session over the native backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run request.
    **controls
        ``stop`` and ``listener``.

    Returns
    -------
    sessions.executors.Started
        The session.
    """
    driver = select("native", problem, grid=window(), modes=[fibre().mode])
    return Started(
        DriverSession(driver, problem, request, entrance(problem), **controls)
    )


def _make_native(problem, request, **controls):
    """Return a native execution session.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run request.
    **controls
        ``stop`` and ``listener``.

    Returns
    -------
    solprop_native.session.NativeSession
        The session.
    """
    return native_session.NativeSession(
        problem,
        request,
        entrance(problem),
        grid=window(),
        modes=[fibre().mode],
        **controls,
    )


def _native_session(problem, request, **controls) -> Started:
    """Start the native execution session.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run request.
    **controls
        ``stop`` and ``listener``.

    Returns
    -------
    sessions.executors.Started
        The session.
    """
    session = _make_native(problem, request, **controls)
    return Started(session, field_reads=lambda: session.native.status().exports)


CASES = {
    "rk": {},
    "rk-rejections": {"rtol": 1e-12, "step_planes": 0, "period": 2, "dz_init": 1e-2},
    "adams": {"integrator": Multistep()},
    "adams-period-3": {"integrator": Multistep(), "period": 3, "step_planes": 5},
    "rk-replay": {"schedule": FixedSchedule(np.full(40, LENGTH / 40))},
    "adams-replay": {
        "integrator": Multistep(),
        "schedule": FixedSchedule(np.full(40, LENGTH / 40)),
    },
}
"""Adams and Runge-Kutta, adaptive and replayed, with periodic rows and step planes."""


@pytest.mark.parametrize("recipe", [CONSUMER, STORED], ids=["consumer", "stored"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_identical_inputs_give_the_drivers_trace_and_fields(case, recipe):
    """SEP 47 section 7.1: the native session's control trace is the driver's.

    Every observation's kind, plane and position, every attempt's end, step, norm and
    acceptance, the periodic rows and the plane statistics, and every exported field,
    bitwise, against the driver session over the native backend.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem = _case(**CASES[case])
    request = describe_run(problem)
    baseline = drive(_driver_session(problem, request), export=recipe)
    started = _native_session(problem, request)
    run = drive(started, export=recipe)
    started.session.close()
    assert_same_run(run, baseline)
    assert run.status.attempts == baseline.status.attempts
    assert run.status.rejected == baseline.status.rejected
    if case == "rk-rejections":
        assert run.status.rejected > 0


@pytest.mark.parametrize("case", ["rk", "adams"])
def test_periodic_rows_cause_no_return_to_python(case):
    """SEP 47 section 7.3: zero native returns caused solely by periodic statistics.

    With a row every accepted step the native session returns to Python only at its
    observations and its end, while the driver session returns once a row; the rows
    are taken natively, each once, and arrive in bulk with their observation.
    """
    problem = _case(period=1, step_planes=0, **CASES[case])
    request = describe_run(problem)
    session = _make_native(problem, request)
    run = drive(Started(session))
    status = session.native.status()
    session.close()
    observations = run.observations
    assert len(run.batches) == len(observations) + 1
    assert not any(
        isinstance(event, Progress) for batch in run.batches for event in batch.events
    )
    rows = sum(event.step_stats.n_rows for event in observations)
    accepted = sum(int(history.accepted.sum()) for history in run.histories)
    assert rows == accepted == status.rows_taken > len(PLANES)
    # The marches were re-entered natively for the rows, never from Python.
    assert status.march_calls >= rows
    assert status.batches == len(run.batches)


def test_a_stop_from_another_thread_reaches_a_march_in_progress():
    """SEP 47 section 5.5: the token sets the native flag with no Python lock.

    A long advance runs with its attempt bound and deadline lifted; another thread
    sets the token while the march holds no Python lock. The march stops after the
    attempt in hand, and the session ends stopped at an eligible committed endpoint.
    """
    problem = _case(step_planes=0, period=50, dz_init=2e-6, dz_max=2e-6)
    request = describe_run(problem)
    token = StopToken()
    started = _native_session(problem, request, stop=token)
    session = started.session
    launch = session.advance()
    session.acknowledge(launch.receipt)
    timer = threading.Timer(0.05, token.set)
    timer.start()
    began = time.perf_counter()
    batches = []
    try:
        while not session.finished:
            batch = session.advance(work_limit=10**9, deadline=1e9)
            batches.append(batch)
            session.acknowledge(batch.receipt)
    finally:
        timer.cancel()
        session.close()
    elapsed = time.perf_counter() - began
    terminal = batches[-1].terminal
    assert terminal is not None
    if terminal.outcome is Outcome.COMPLETED:
        pytest.skip(f"the run completed in {elapsed:.3f} s, before the stop was set")
    assert terminal.outcome is Outcome.STOPPED
    assert isinstance(terminal.error, StopRequested)
    assert terminal.ticket is not None
    assert terminal.z < LENGTH
    assert terminal.history is not None and terminal.history.n_steps > 0


def _refused(problem, request, name, monkeypatch) -> None:
    """Assert the session refuses a run by `name`, before building a backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run request.
    name : str
        A phrase the refusal must contain.
    monkeypatch : pytest.MonkeyPatch
        Replaces the backend's constructor by one that fails the test.
    """

    def built(*args, **kwargs):
        del args, kwargs
        pytest.fail("the session built a backend before refusing")

    monkeypatch.setattr(native_session, "NativeBackend", built)
    assert name in (native_session.refusal(problem, request) or "")
    with pytest.raises(BackendError, match=name):
        native_session.NativeSession(
            problem, request, entrance(problem), grid=window(), modes=[fibre().mode]
        )


class _OwnAccept(PIController):
    """A controller that overrides a decision: a reference-only experiment."""

    def accepts(self, norm):
        """Accept as the base does."""
        return super().accepts(norm)


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: (soliton(), {"monitor": spy_monitor([])}), "a boundary monitor"),
        (lambda: (_case(interpolate=True), {}), "dense output"),
        (
            lambda: (
                _case(
                    integrator=LowStorageRungeKutta(),
                    schedule=FixedSchedule(np.full(40, LENGTH / 40)),
                ),
                {},
            ),
            "the low-storage replay",
        ),
        (
            lambda: (
                _case(integrator=ComparisonMultistep(corrector="pece", order=4)),
                {},
            ),
            "the comparison method",
        ),
        (
            lambda: (_case(controller=_OwnAccept(rtol=1e-4, atol=1e-16)), {}),
            "a controller with its own accepts",
        ),
        (
            lambda: (_case(names=("energy", "peak_power", "fwhm_t")), {}),
            "'fwhm_t'",
        ),
    ],
    ids=["monitor", "dense", "low-storage", "comparison", "controller", "statistic"],
)
def test_the_prototype_refuses_by_name_before_allocating(build, name, monkeypatch):
    """SEP 47 section 6.3: every refused feature names itself, before allocation."""
    problem, settings = build()
    _refused(problem, describe_run(problem, **settings), name, monkeypatch)


def test_a_session_leases_its_state_exclusively():
    """SEP 47 section 5.2: the session is the one mutable owner of its state.

    While it lives, the backend's own march refuses to run on the leased state.
    """
    problem = _case()
    session = _make_native(problem, describe_run(problem))
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    state = backend._native_state  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="leased"):
        state.rk_march_run({}, {})
    session.close()


def test_observations_carry_the_rows_since_the_previous_one():
    """SEP 47 section 6.4: rows are sampled at their steps and delivered with the plane.

    Each row's position is its attempt's end and its index points into the same
    observation's history, counted with rejections.
    """
    problem = _case(rtol=1e-9, step_planes=0, period=2)
    started = _native_session(problem, describe_run(problem))
    run: Run = drive(started)
    started.session.close()
    for event in run.observations[1:]:
        assert isinstance(event, ObservationReady)
        rows = event.step_stats
        np.testing.assert_array_equal(event.history.z[rows.step], rows.z)
        assert event.history.accepted[rows.step].all()
        assert rows.values["energy"].shape == (rows.n_rows, 1, 1)
    assert isinstance(run.events[-1], Terminal)
