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
    EXECUTORS,
    IDENTITY,
    PLANES,
    STORED,
    Run,
    Started,
    assert_same_run,
    drive,
    entrance,
    fibre,
    skip_device_refused,
    trace,
    window,
)
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import (
    ObservationReady,
    Outcome,
    Progress,
    Terminal,
)
from solprop.backends.registry import select
from solprop.backends.session import SessionError, StopToken
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
    skip_device_refused(problem, request)
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
    "rk-temporal": {
        "names": (
            "omega0",
            "energy",
            "energy_window",
            "peak_power",
            "peak_power_window",
            "peak_power_total",
            "fwhm_t",
            "fwhm_t_outer",
        )
    },
    "adams-replay": {
        "integrator": Multistep(),
        "schedule": FixedSchedule(np.full(40, LENGTH / 40)),
    },
    "low-storage": {
        "integrator": LowStorageRungeKutta(),
        "schedule": FixedSchedule(np.full(40, LENGTH / 40)),
    },
    "adams-dense": {"integrator": Multistep(), "interpolate": True},
    "adams-interaction-dense": {
        "integrator": Multistep(formulation="interaction"),
        "interpolate": True,
        "step_planes": 2,
        "rtol": 1e-8,
    },
}
"""Adams and Runge-Kutta, adaptive and replayed, and the low-storage replay, with
periodic rows and step planes; every temporal statistic, whose native plan the
driver's backend also reads; and the Adams march's planes interpolated in both
formulations, which the driver also interpolates natively, from weights its Python
forms by the same native basis."""


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
            lambda: (_case(names=("energy", "peak_power", "zdw")), {}),
            "'zdw'",
        ),
    ],
    ids=[
        "comparison",
        "controller",
        "statistic",
    ],
)
def test_the_prototype_refuses_by_name_before_allocating(build, name, monkeypatch):
    """SEP 47 section 6.3: every refused feature names itself, before allocation."""
    problem, settings = build()
    _refused(problem, describe_run(problem, **settings), name, monkeypatch)


@pytest.mark.parametrize(
    ("options", "name"),
    [
        (
            {
                "integrator": LowStorageRungeKutta(),
                "schedule": FixedSchedule(np.full(40, LENGTH / 40)),
            },
            "the low-storage replay on a device build",
        ),
        ({"interpolate": True}, "dense output on a device build"),
    ],
    ids=["low-storage", "dense"],
)
def test_a_device_build_refuses_by_name_what_phase_3_does_not_admit(
    options, name, monkeypatch
):
    """SEP 47 phase 3: the representative device path refuses the rest by name.

    The decision is the problem's and the request's alone, so it is checked here on
    any build, the device patched in; a device build's own lane runs what it admits.
    """
    monkeypatch.setattr(native_session, "_on_device", lambda: True)
    problem = _case(**options)
    _refused(problem, describe_run(problem), name, monkeypatch)


@pytest.mark.parametrize("case", ["rk", "adams", "adams-period-3", "rk-replay"])
def test_a_device_build_admits_the_representative_path(case, monkeypatch):
    """SEP 47 phase 3: Runge-Kutta and Adams on the modal route, with planes, periodic
    rows and step planes, are the device path's; its monitor is admitted too
    (``test_native_session_monitor``)."""
    monkeypatch.setattr(native_session, "_on_device", lambda: True)
    problem = _case(**CASES[case])
    assert native_session.refusal(problem, describe_run(problem)) is None


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


def test_the_native_export_is_apply_recipe_to_the_bit():
    """SEP 47 section 5.4: the fused native pass is ``apply_recipe``, bitwise.

    A launch field with signed zeros in both parts and values of every sign is exported
    with each recipe, into new arrays and into destinations, and compared with the
    reference conversion of the same field at the same position.
    """
    from solprop.backends.export import ExportRecipe, apply_recipe  # noqa: PLC0415
    from solprop.backends.session import Start  # noqa: PLC0415
    from solprop.translate.linop import power_weight_at  # noqa: PLC0415

    problem = _case()
    rng = np.random.default_rng(47)
    field = np.array(problem.initial.field0[0], dtype=np.complex128, copy=True)
    parts = field.view(np.float64).reshape(*field.shape, 2)  # real, imaginary
    parts[..., ::7, 0] = -0.0
    parts[..., ::5, 1] = -0.0
    parts[..., 1::11, 1] = 0.0
    parts[..., 3::13, 0] *= -rng.random(parts[..., 3::13, 0].shape)
    session = native_session.NativeSession(
        problem,
        describe_run(problem),
        Start(field=field, z=0.0),
        grid=window(),
        modes=[fibre().mode],
    )
    batch = session.advance()
    (launch,) = batch.events
    assert isinstance(launch, ObservationReady)
    weight = power_weight_at(problem, 0.0)
    # The loaded state, which the launch's mask may have changed from `field`: the
    # identity export is a lossless copy of it.
    field = np.array(
        session.export(launch.ticket, ExportRecipe(power=False, optics=False))
    )
    for power in (False, True):
        for optics in (False, True):
            recipe = ExportRecipe(power=power, optics=optics)
            expected = apply_recipe(field, recipe, weight=weight if power else None)
            got = session.export(launch.ticket, recipe)
            assert got.tobytes() == expected.tobytes(), recipe
            out = np.empty_like(expected)
            assert session.export(launch.ticket, recipe, out=out) is out
            assert out.tobytes() == expected.tobytes(), recipe
    assert np.signbit(np.real(field)).any() and np.signbit(np.imag(field)).any()
    session.close()


LEASED_CALLS = {
    "set_initial": lambda state, n: state.set_initial(
        np.zeros(n, np.complex128), np.empty(0, np.complex128)
    ),
    "rk_march_run": lambda state, n: state.rk_march_run({}, {}),
    "adams_march_run": lambda state, n: state.adams_march_run({}, {}),
    "table_slots": lambda state, n: state.table_slots(False, False, 1, False),
    "rk_weights_configure": lambda state, n: state.rk_weights_configure([1.0]),
    "low_storage_configure": lambda state, n: state.low_storage_configure(
        1, False, 0.0
    ),
    "flow_guard_configure": lambda state, n: state.flow_guard_configure(
        [1.0], [1.0], 1.0, 1, 1.0
    ),
    "adams_march_fault": lambda state, n: state.adams_march_fault(
        "trial", 0, "backend"
    ),
    "field_to_host": lambda state, n: state.field_to_host(),
    "statistics": lambda state, n: state.statistics(
        2, np.empty(0), np.empty(0), 0.0, 1.0, 0.0, 1.0, np.empty(0)
    ),
    "finish": lambda state, n: state.finish(1e-6),
    "promote_fsal": lambda state, n: state.promote_fsal(),
    "clear_fsal": lambda state, n: state.clear_fsal(),
    "monitor_reading": lambda state, n: state.monitor_reading(),
    "quadrature_peak": lambda state, n: state.quadrature_peak(),
}
"""Methods of a leased state that mutate it or read its live field or workspace."""


@pytest.mark.parametrize("name", sorted(LEASED_CALLS))
def test_a_leased_state_refuses_every_live_use(name):
    """SEP 47 section 5.2: one mutable owner. While a session leases the state, every
    method that mutates it or reads its live field or workspace is refused by name;
    its fixed facts stay readable, so the execution record can be written.
    """
    problem = _case()
    session = _make_native(problem, describe_run(problem))
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    state = backend._native_state  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(RuntimeError, match="leased by an execution session"):
            LEASED_CALLS[name](state, int(np.prod(problem.shape[1:])))
        assert state.memory_ledger()
        assert state.cpu_execution
        assert "native_session" in session.execution_record()
    finally:
        session.close()
    # Closed, the session released its lease.
    with pytest.raises(Exception) as caught:
        state.rk_march_run({}, {})
    assert "leased" not in str(caught.value)


def test_closing_the_state_closes_its_session_first():
    """SEP 47 section 5.2: the state's own close closes the session that leases it,
    which then refuses to advance; nothing uses the closed state.
    """
    problem = _case()
    session = _make_native(problem, describe_run(problem))
    batch = session.advance()
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    backend._native_state.close()  # pyright: ignore[reportPrivateUsage]
    assert session.closed
    with pytest.raises(SessionError):
        session.acknowledge(batch.receipt)
    with pytest.raises(SessionError):
        session.advance()
    session.close()


def test_status_polls_from_another_thread_do_not_wait_for_an_advance():
    """SEP 47 section 5.5: an advance runs without the interpreter lock, and the
    status, a snapshot request and a stop need no session lock: another thread polls
    them while a long advance is in hand.
    """
    problem = _case(step_planes=0, period=50, dz_init=2e-6, dz_max=2e-6)
    session = _make_native(problem, describe_run(problem))
    session.acknowledge(session.advance().receipt)
    polls: list[int] = []
    done = threading.Event()

    def poll() -> None:
        while not done.is_set():
            polls.append(session.status().attempts)
            time.sleep(1e-3)

    poller = threading.Thread(target=poll)
    poller.start()
    try:
        batch = session.advance(work_limit=4000, deadline=1e9)
    finally:
        done.set()
        poller.join()
    attempts = session.status().attempts
    session.acknowledge(batch.receipt)
    session.close()
    during = [count for count in polls if 0 < count < attempts]
    assert during, f"no poll saw the advance in progress ({len(polls)} polls)"


def test_a_full_chunk_buffer_yields_and_changes_nothing(monkeypatch):
    """SEP 47 section 5.3: bounded buffers, exactly-once chunks. With room for three
    attempts and two rows the session yields progress chunks no larger, and the run's
    trace and fields are the unbounded run's.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem = _case(rtol=1e-12, step_planes=0, period=1, dz_init=1e-2)
    request = describe_run(problem)
    baseline = drive(_native_session(problem, request), export=CONSUMER)
    monkeypatch.setattr(native_session, "ATTEMPT_CAPACITY", 3)
    monkeypatch.setattr(native_session, "ROW_CAPACITY", 2)
    run = drive(_native_session(problem, request), export=CONSUMER)
    assert_same_run(run, baseline)
    chunks = [
        event.history.n_steps for event in run.events if isinstance(event, Progress)
    ]
    assert chunks and max(chunks) <= 3
    assert len(run.batches) > len(baseline.batches)


def test_march_admission_is_decided_before_the_backend_is_built(monkeypatch):
    """SEP 47 section 5.2: a march the backend would step on the host is refused from
    the problem alone, and the problem-only predicate agrees with the built backend.
    """
    from solprop_native import backend as native_backend  # noqa: PLC0415

    problem = _case()
    request = describe_run(problem)
    assert native_backend.native_march_refusal(problem) is None
    session = _make_native(problem, request)
    built = session._backend  # pyright: ignore[reportPrivateUsage]
    assert built is not None
    staged = built._stage_tables() is not None  # pyright: ignore[reportPrivateUsage]
    assert staged == native_backend.stage_tables_available(problem)
    session.close()
    monkeypatch.setattr(native_backend, "_RK_ROWS", "host")
    _refused(problem, request, "a trial whose stage rows the host forms", monkeypatch)


DENSE = {
    "runge-kutta": {},
    "runge-kutta-rejections": {"rtol": 1e-12, "period": 2, "dz_init": 1e-2},
    "adams": {"integrator": Multistep()},
    "adams-interaction": {"integrator": Multistep(formulation="interaction")},
}
"""Dense runs: the default tableau's DOPRI5 extension, with and without rejections,
and the Adams march's interpolant in both formulations."""


def _control_trace(run: Run) -> list[tuple[object, ...]]:
    """Return a run's trace without its plane statistics, which read the planes.

    Parameters
    ----------
    run : sessions.executors.Run
        The run.

    Returns
    -------
    list
        Each observation's kind, plane, position, attempts and rows, and the terminal
        event's, as :func:`sessions.executors.trace` takes them, as bytes.
    """
    return [entry[:6] for entry in trace(run)]


def test_runge_kutta_dense_planes_follow_the_drivers_run():
    """SEP 47 section 7.1: the native session's dense Runge-Kutta run is the driver's.

    Both reload the committed state after an observation, the next step evaluating its
    first stage afresh, so every attempt, norm, acceptance, periodic row and
    observation position is the driver's to the bit, whatever FFT plans this process
    chose (a stage reused across an observation instead parted the two by up to 9e-8
    in a norm under some plans, and the test with it). Only the planes themselves
    differ: the driver forms each on the host with NumPy (`StepDriver._dense_record`,
    `_observe_inside`), the session where the state lives, its flow through the native
    exponential; measured 5.5e-16 of the peak, their statistics likewise.
    """
    if not _one_thread():
        pytest.skip("equal decisions are claimed on one thread (SOLPROP_THREADS=1)")
    for options in ({"step_planes": 3, "rtol": 1e-8}, DENSE["runge-kutta-rejections"]):
        problem = _case(interpolate=True, **{"step_planes": 0, **options})
        request = describe_run(problem)
        baseline = drive(_driver_session(problem, request), export=CONSUMER)
        started = _native_session(problem, request)
        run = drive(started, export=CONSUMER)
        started.session.close()
        assert _control_trace(run) == _control_trace(baseline)
        scale = max(float(np.max(np.abs(field))) for field in baseline.fields)
        for field, other in zip(run.fields, baseline.fields, strict=True):
            np.testing.assert_allclose(field, other, rtol=0, atol=1e-14 * scale)
        for got, expected in zip(run.observations, baseline.observations, strict=True):
            for name, value in expected.stats.items():
                np.testing.assert_allclose(got.stats[name], value, rtol=1e-12)


@pytest.mark.parametrize("case", sorted(DENSE))
def test_dense_planes_meet_the_reference(case):
    """SEP 47 section 6.6 and SEP 25: dense planes held to the NumPy reference.

    The native session against the reference session on the same problem, under the
    budgets of the existing dense tests: the same attempts, the planes within 1e-12 of
    the peak (``tests/backends/test_dense_output.py``), and for Adams the accepted
    positions within 1e-6 relative (``tests/backends/test_resident_adams.py``).
    Measured: at most 2.8e-16 of the peak, and 4.8e-11 relative in positions.
    """
    problem = _case(interpolate=True, step_planes=0, **DENSE[case])
    request = describe_run(problem)
    reference = drive(
        EXECUTORS["reference"].start(problem, request, entrance(problem)),
        export=IDENTITY,
    )
    started = _native_session(problem, request)
    run = drive(started, export=IDENTITY)
    started.session.close()
    assert run.status.attempts == reference.status.attempts
    assert [(event.kind, event.z) for event in run.observations] == [
        (event.kind, event.z) for event in reference.observations
    ]
    np.testing.assert_allclose(
        run.accepted_ends(), reference.accepted_ends(), rtol=1e-6
    )
    scale = max(float(np.max(np.abs(field))) for field in reference.fields)
    for got, expected in zip(run.fields, reference.fields, strict=True):
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("case", ["runge-kutta", "adams", "adams-interaction"])
def test_a_dense_plane_crosses_to_the_host_only_as_its_export(case):
    """SEP 47 sections 5.3 and 7.3: dense output is formed where the state lives.

    No stage, field or extension crosses to the host to interpolate a plane, and
    nothing crosses back: the state's crossings are the exported planes' bytes alone.
    """
    problem = _case(interpolate=True, step_planes=0, **DENSE[case])
    session = _make_native(problem, describe_run(problem))
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    state = backend._native_state  # pyright: ignore[reportPrivateUsage]
    before = state.crossed_bytes()
    run = drive(Started(session), export=CONSUMER)
    after = state.crossed_bytes()
    status = session.native.status()
    session.close()
    assert status.exports == len(run.observations)
    assert after[0] - before[0] == status.export_bytes
    assert after[1] == before[1]


def test_the_ledger_counts_runge_kutta_dense_output_before_and_during_a_session():
    """SEP 47 section 5.3: a retained dense-output object appears in the memory ledger.

    A Runge-Kutta run's dense-output storage -- five extension coefficients and the
    held committed state, six fields, and a byte a bin -- is reported by a backend
    built for it before any session allocates it, as an estimate reads it, and is
    then what the session allocated, until it closes; an Adams run's lives in its
    history's views.
    """
    problem = _case(interpolate=True, step_planes=0)
    backend = select("native", problem, grid=window(), modes=[fibre().mode])
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    field = ledger["state"]
    assert ledger["rk_dense"] == 6 * field + field // 16 > 0
    session = _make_native(problem, describe_run(problem))
    live = session._backend  # pyright: ignore[reportPrivateUsage]
    assert live is not None
    assert live.memory_ledger()["rk_dense"] == ledger["rk_dense"]
    state = live._native_state  # pyright: ignore[reportPrivateUsage]
    assert state.memory_ledger()["rk_dense"] == ledger["rk_dense"]
    # Closing the native session releases it, before its backend closes.
    session.native.close()
    assert state.memory_ledger()["rk_dense"] == 0
    session.close()
    landed = _case(step_planes=0)
    backend = select("native", landed, grid=window(), modes=[fibre().mode])
    try:
        assert backend.memory_ledger()["rk_dense"] == 0
    finally:
        backend.close()
