# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Drive a native session or a native group to its end and keep what it observed.

The native test root's copy of solprop's ``tests/sessions/runs.py``, with the same
names and semantics, since this root cannot import solprop's test helpers: :func:`run`
starts a backend's session (:func:`~solprop.backends.sessions.start_session`), drives
it to its end with :func:`drive` and returns its :class:`Run`, whose fields are
exported as the session holds the state (:data:`IDENTITY`); :func:`final_field`,
:func:`launch` and :func:`launch_statistics` read its last state and its launch plane;
:func:`load_native` and :func:`native_trial` reach the native state's own trial, which
the kernel-level checks make where they once called the driver's trial step. A failed
run raises its error by default, as a service loop does.

The group tests compare each member of a native group session
(:class:`solprop_native.group.NativeGroupSession`, started through solprop's
:func:`~solprop.backends.sessions.start_group`) with the same member run alone through
the native session (:func:`alone`): the oracle that replaced the transitional lockstep
batch in solprop SEP 47's phase 5.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from solprop.backends.events import (
    EventBatch,
    ObservationKind,
    ObservationReady,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.session import RunSession, SessionStatus, Start
from solprop.backends.sessions import start_group, start_session

IDENTITY = ExportRecipe(power=False, optics=False)
"""The state as the session holds it, in the internal FFT convention."""

EMPTY = BackendHistory(
    np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=np.bool_), np.empty(0)
)
"""The chunk of an event that carries no attempts."""

_NO_FIELD = np.empty(0, dtype=np.complex128)
"""A plane's field in a run driven without an export recipe."""


@dataclass(frozen=True, eq=False)
class Plane:
    """One observation of a run, with what it delivered since the previous one."""

    event: ObservationReady
    """The observation."""
    field: np.ndarray
    """Its field as the run exported it; empty without an export recipe."""
    history: BackendHistory
    """Every attempt since the previous observation (:meth:`Run.assembled`)."""
    rows: StepStats
    """The periodic rows since the previous observation."""

    @property
    def z(self) -> float:
        """Return the observation's position, m."""
        return float(self.event.z)

    @property
    def kind(self) -> ObservationKind:
        """Return why the plane was observed."""
        return self.event.kind

    @property
    def plane_index(self) -> int:
        """Return the save plane the observation answers, ``-1`` for the launch."""
        return self.event.plane_index

    @property
    def stats(self) -> dict[str, np.ndarray]:
        """Return the observation's statistics by catalogue name."""
        return {name: np.asarray(value) for name, value in self.event.stats.items()}

    @property
    def summary(self) -> StepSummary:
        """Return the cost and tolerance of the propagation that reached the plane."""
        return self.event.summary


@dataclass
class Run:
    """What driving a session to its end produced."""

    batches: list[EventBatch]
    """Every batch, in order."""
    fields: list[np.ndarray]
    """Each observation's field exported with the run's recipe, or nothing."""
    status: SessionStatus
    """The status after the last batch was acknowledged."""

    @property
    def events(self) -> list[object]:
        """Return every event in committed order."""
        return [event for batch in self.batches for event in batch.events]

    @property
    def observations(self) -> list[ObservationReady]:
        """Return the observations in committed order."""
        return [event for event in self.events if isinstance(event, ObservationReady)]

    @property
    def terminal(self) -> Terminal:
        """Return the terminal event, the last of the last batch."""
        terminal = self.batches[-1].terminal
        assert terminal is not None
        return terminal

    @property
    def chunks(self) -> list[tuple[BackendHistory, StepStats]]:
        """Return every event's chunk of attempts and rows, in delivery order.

        Each attempt and row is delivered once, by the first event after it, of any
        kind; an event that carries none gives an empty chunk.
        """
        found: list[tuple[BackendHistory, StepStats]] = []
        for event in self.events:
            history = cast("BackendHistory | None", getattr(event, "history", None))
            rows = cast("StepStats | None", getattr(event, "step_stats", None))
            found.append(
                (
                    EMPTY if history is None else history,
                    StepStats.empty() if rows is None else rows,
                )
            )
        return found

    @property
    def histories(self) -> list[BackendHistory]:
        """Return each attempt record exactly once: every event's chunk."""
        return [history for history, _ in self.chunks]

    def assembled(self) -> list[tuple[object, BackendHistory, StepStats]]:
        """Assemble the chunks per observation and terminal, as a service loop does."""
        found: list[tuple[object, BackendHistory, StepStats]] = []
        held: list[tuple[BackendHistory, StepStats]] = []
        for event, chunk in zip(self.events, self.chunks, strict=True):
            held.append(chunk)
            if isinstance(event, ObservationReady | Terminal):
                found.append((event, *join_attempts(held)))
                held = []
        return found

    @property
    def attempts(self) -> int:
        """Return the attempts the histories report, accepted and rejected."""
        return sum(history.n_steps for history in self.histories)

    @property
    def joined(self) -> tuple[BackendHistory, StepStats]:
        """Return every attempt and every periodic row of the run, joined in order."""
        return join_attempts(self.chunks)

    def all_planes(self) -> list[Plane]:
        """Return every observation, the launch plane's included, in order."""
        fields = (
            list(self.fields) if self.fields else [_NO_FIELD] * len(self.observations)
        )
        found: list[Plane] = []
        for event, history, rows in self.assembled():
            if isinstance(event, ObservationReady):
                found.append(Plane(event, fields[len(found)], history, rows))
        return found

    @property
    def planes(self) -> list[Plane]:
        """Return every observation after the launch plane, in order."""
        return [
            plane
            for plane in self.all_planes()
            if plane.kind is not ObservationKind.LAUNCH
        ]

    @property
    def launch(self) -> Plane | None:
        """Return the launch plane's observation, when the start owed it."""
        for plane in self.all_planes():
            if plane.kind is ObservationKind.LAUNCH:
                return plane
        return None

    @property
    def final(self) -> Plane:
        """Return the last observation after the launch plane."""
        return self.planes[-1]

    @property
    def met_tolerance(self) -> bool:
        """Return whether every propagation between observations met its tolerance."""
        return all(plane.summary.met_tolerance for plane in self.planes)


def committed_steps(run: Run) -> int:
    """Return the accepted attempts a run committed, assembled across its events."""
    return sum(
        int(np.count_nonzero(history.accepted)) for _, history, _ in run.assembled()
    )


def drive(
    session: RunSession,
    *,
    export: ExportRecipe | None = None,
    between: Callable[[RunSession, EventBatch], None] | None = None,
    observations: int | None = None,
    until: float | None = None,
    raises: bool = True,
) -> Run:
    """Advance a started session to its end, acknowledging each batch.

    A warning barrier is emitted as a service loop emits it. The caller closes the
    session. solprop's copy takes the suite's ``Started`` wrapper in place of the
    session.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The started session.
    export : solprop.backends.export.ExportRecipe or None, optional
        Export every observation's field with this recipe, before acknowledging it.
    between : callable or None, optional
        Called with the session and each batch, after any export and before the
        acknowledgment.
    observations : int or None, optional
        Stop once this many observations were reported, the launch plane's included.
    until : float or None, optional
        Stop after the batch of the first observation at this position, m.
    raises : bool, optional
        Raise a failed run's terminal error; ``False`` returns the failed run.

    Returns
    -------
    Run
        What the session reported.
    """
    batches: list[EventBatch] = []
    fields: list[np.ndarray] = []
    seen = 0
    reached = False
    while not session.finished:
        batch = session.advance()
        batches.append(batch)
        for event in batch.events:
            if isinstance(event, ServiceRequired) and event.kind is ServiceKind.WARNING:
                emit_warning(event)
            if isinstance(event, ObservationReady):
                seen += 1
                reached = reached or (until is not None and event.z == until)
        if export is not None:
            fields.extend(
                np.array(session.export(event.ticket, export))
                for event in batch.events
                if isinstance(event, ObservationReady)
            )
        if between is not None:
            between(session, batch)
        session.acknowledge(batch.receipt)
        if reached or (observations is not None and seen >= observations):
            break
    result = Run(batches, fields, session.status())
    if raises and batches and batches[-1].terminal is not None:
        error = result.terminal.error
        if error is not None:
            raise error
    return result


def run(
    backend: Any,
    problem: Any,
    request: Any = None,
    start: Start | None = None,
    *,
    owns: bool = False,
    record: dict[str, object] | None = None,
    monitor: dict[str, object] | None = None,
    raises: bool = True,
    export: ExportRecipe | None = IDENTITY,
    observations: int | None = None,
    until: float | None = None,
    between: Callable[[RunSession, EventBatch], None] | None = None,
) -> Run:
    """Drive a backend's session from a start to the end of its itinerary.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, native or the reference, built for `problem`; it stays the
        caller's unless `owns`.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest or None, optional
        The run's request; the backend's own by default.
    start : solprop.backends.session.Start or None, optional
        Where it starts; the problem's entrance by default.
    owns : bool, optional
        Whether the session closes the backend when it closes.
    record : dict or None, optional
        Updated with the session's execution record before it closes.
    monitor : dict or None, optional
        Updated with the session's boundary monitor record before it closes.
    raises : bool, optional
        Raise a failed run's terminal error; ``False`` returns the failed run.
    export : solprop.backends.export.ExportRecipe or None, optional
        The recipe every observation's field is exported with; :data:`IDENTITY` by
        default, ``None`` exports nothing.
    observations, until, between
        As :func:`drive` takes them.

    Returns
    -------
    Run
        What the session reported; the session is closed.
    """
    session = start_session(
        backend,
        problem,
        backend.request if request is None else request,
        Start.entrance(problem) if start is None else start,
        owns=owns,
    )
    try:
        result = drive(
            session,
            export=export,
            between=between,
            observations=observations,
            until=until,
            raises=False,
        )
        if record is not None:
            record.update(session.execution_record())
        if monitor is not None:
            monitor.update(session.monitor_record() or {})
    finally:
        session.close()
    if raises and result.batches and result.batches[-1].terminal is not None:
        error = result.terminal.error
        if error is not None:
            raise error
    return result


def final_field(
    backend: Any,
    problem: Any,
    request: Any = None,
    start: Start | None = None,
    *,
    z: float | None = None,
    ended: bool = False,
    owns: bool = False,
    **options: Any,
) -> np.ndarray:
    """Return the state at the last observation of a run, as a host copy.

    Parameters
    ----------
    backend, problem, request, start, owns
        As :func:`run` takes them.
    z : float or None, optional
        Where the last observation must be, exactly, m; unchecked by default.
    ended : bool, optional
        Check that the last observation is the request's end, to within a replayed
        schedule's rounding (relative ``1e-12``).
    **options
        Passed to :func:`run`.

    Returns
    -------
    numpy.ndarray
        The field, ``(n_modes, n_pol, n_omega)``, in the FFT convention.
    """
    own = backend.request if request is None else request
    result = run(backend, problem, own, start, owns=owns, **options)
    last = result.observations[-1]
    if z is not None:
        assert last.z == z, (last.z, z)
    if ended:
        end = float(own.stepper.z_end)
        assert math.isclose(last.z, end, rel_tol=1e-12), (last.z, end)
    return result.fields[-1]


def _launch_observation(session: RunSession) -> ObservationReady:
    """Advance `session` to its first observation, the launch plane, unacknowledged."""
    while not session.finished:
        batch = session.advance(work_limit=1)
        for event in batch.events:
            if isinstance(event, Terminal) and event.error is not None:
                raise event.error
            if isinstance(event, ObservationReady):
                return event
        session.acknowledge(batch.receipt)
    msg = "the session observed no launch plane"
    raise AssertionError(msg)


@contextmanager
def launched(
    backend: Any,
    problem: Any,
    request: Any = None,
    start: Start | None = None,
    *,
    field: np.ndarray | None = None,
    owns: bool = False,
) -> Generator[tuple[RunSession, ObservationReady]]:
    """Yield `backend`'s own session and the observation of its launch plane.

    The session starts with the launch plane owed, from `start` or from `field` at the
    problem's entrance, and is closed on exit; its observation is pending, so its
    ticket exports the loaded state as the session holds it. solprop's copy also
    observes a state at another position with other statistics (``stats``, ``z``).
    """
    if start is None:
        launch_field = problem.initial.field0 if field is None else field
        start = Start(field=np.array(launch_field), z=float(problem.initial.z))
    own = backend.request if request is None else request
    session = start_session(backend, problem, own, start, owns=owns)
    try:
        yield session, _launch_observation(session)
    finally:
        session.close()


def launch(
    backend: Any,
    problem: Any,
    request: Any = None,
    start: Start | None = None,
    *,
    field: np.ndarray | None = None,
    owns: bool = False,
) -> Plane:
    """Return the launch plane `backend`'s own session observes, its field exported.

    Nothing is stepped; the arguments are :func:`launched`'s.
    """
    with launched(backend, problem, request, start, field=field, owns=owns) as (
        session,
        observation,
    ):
        held = np.array(session.export(observation.ticket, IDENTITY))
        return Plane(observation, held, observation.history, observation.step_stats)


def launch_statistics(
    backend: Any,
    problem: Any,
    request: Any = None,
    start: Start | None = None,
    *,
    field: np.ndarray | None = None,
    owns: bool = False,
) -> dict[str, np.ndarray]:
    """Return the statistics `backend`'s own session observes at its launch plane.

    Nothing is exported or stepped; the arguments are :func:`launched`'s.
    """
    with launched(backend, problem, request, start, field=field, owns=owns) as (
        _,
        observation,
    ):
        return {name: np.array(value) for name, value in observation.stats.items()}


def load_native(
    backend: Any,
    field: Any,
    noise: Any = None,
    *,
    clear_fsal: bool = True,
) -> Any:
    """Load a host state into a native backend's state, its recycled stage dropped.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The backend; its native state is returned.
    field : numpy.ndarray
        The state, shape ``(n_modes, n_pol, n_omega)``.
    noise : numpy.ndarray or None, optional
        The ``modified`` noise model's spectrum the stages add, in the state's layout,
        or ``None``.
    clear_fsal : bool, optional
        Drop the recycled first-same-as-last stage, as a session's load does; the
        checks that loaded the bare state keep it with ``False``.

    Returns
    -------
    solprop_native._core.State
        The loaded native state.
    """
    state = backend._native_state  # pyright: ignore[reportPrivateUsage]
    flat = np.ascontiguousarray(field, dtype=np.complex128).reshape(-1)
    added = (
        np.empty(0, dtype=np.complex128)
        if noise is None
        else np.ascontiguousarray(noise, dtype=np.complex128).reshape(-1)
    )
    state.set_initial(flat, added)
    if clear_fsal:
        state.clear_fsal()
    return state


def native_trial(backend: Any, z: float, dz: float, *, field: Any = None) -> np.ndarray:
    """Make one native Runge-Kutta trial from `z` on the backend's staged rows.

    At the nodes of the backend's own request's tableau; its stages stay in the
    state. A `field` is loaded first (:func:`load_native`).

    Returns
    -------
    numpy.ndarray
        The trial's scored partials, float64.
    """
    if field is not None:
        load_native(backend, field)
    backend._staged_rows()  # pyright: ignore[reportPrivateUsage]
    nodes = [float(node) for node in backend.request.stepper.tableau.nodes]
    state = backend._native_state  # pyright: ignore[reportPrivateUsage]
    return np.array(state.rk_trial(z, dz, nodes, prepare_only=False), dtype=np.float64)


class _Recorder:
    """Keep one trajectory's observations, each with the attempts and rows since the
    previous one, assembled from every event's chunk (exactly-once delivery).

    The comparable record of a group member and of its lone session, which
    :func:`assert_same` compares.
    """

    def __init__(self) -> None:
        self.planes: list[dict[str, Any]] = []
        self._chunks: list[tuple[Any, Any]] = []

    def __call__(self, session: RunSession, batch: EventBatch) -> None:
        """Record a lone session's batch, as :func:`drive`'s ``between``."""
        self.take(session.export, batch.events)

    def take(self, export: Any, events: Sequence[Any]) -> None:
        """Record a batch's events, exporting each observation's field.

        A run that failed raises its failure here, so that no caller compares a run
        cut short.
        """
        for event in events:
            if isinstance(event, Terminal) and event.error is not None:
                raise event.error
            history = getattr(event, "history", None)
            if history is not None:
                self._chunks.append((history, getattr(event, "step_stats", None)))
            if not isinstance(event, ObservationReady):
                continue
            history, rows = join_attempts(self._chunks)
            self._chunks = []
            self.planes.append(
                {
                    "z": float(event.z),
                    "attempts": {
                        "z": np.asarray(history.z).copy(),
                        "dz": np.asarray(history.dz).copy(),
                        "accepted": np.asarray(history.accepted).copy(),
                        "error_norm": np.asarray(history.error_norm).copy(),
                    },
                    "field": np.array(export(event.ticket, IDENTITY)),
                    "stats": {
                        name: np.asarray(value) for name, value in event.stats.items()
                    },
                    "row_z": np.asarray(rows.z).copy(),
                    "rows": {
                        name: np.asarray(value).copy()
                        for name, value in rows.values.items()
                    },
                }
            )


def alone(owner: Any, problem: Any, request: Any, start: Start) -> list[dict[str, Any]]:
    """Run one member alone through the native session; return its observations.

    The session owns `owner` and closes it; the observations are :func:`grouped`'s.
    """
    recorder = _Recorder()
    run(owner, problem, request, start, owns=True, export=None, between=recorder)
    return recorder.planes


def drain(session: Any) -> Terminal:
    """Advance a started session to its end, exporting nothing; return its terminal.

    A failure is returned in the terminal event, not raised; the caller closes the
    session.
    """
    while True:
        batch = session.advance()
        session.acknowledge(batch.receipt)
        for event in batch.events:
            if isinstance(event, Terminal):
                return event


def grouped(
    owners: Sequence[Any],
    problems: Sequence[Any],
    requests: Sequence[Any],
    starts: Sequence[Start],
) -> list[list[dict[str, Any]]]:
    """Run members as one native group; return each member's observations.

    The group owns `owners` and closes them.
    """
    group = start_group(owners, problems, requests, starts)
    recorders = [_Recorder() for _ in range(group.n_members)]
    try:
        assert type(group).__name__ == "NativeGroupSession"
        while not group.finished:
            batch = group.advance()
            for index, events in enumerate(batch.members):
                recorders[index].take(functools.partial(group.export, index), events)
            group.acknowledge(batch.receipt)
    finally:
        group.close()
    return [recorder.planes for recorder in recorders]


def one_thread() -> bool:
    """Return whether the native backend runs one thread, as the default lane sets it.

    The members of a group are bitwise their lone sessions there, where the
    reductions' order is fixed.
    """
    import os

    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


def assert_same(left: Any, right: Any, what: str = "") -> None:
    """Assert two recorded values agree bitwise, nested dicts and lists included."""
    if isinstance(left, dict):
        mine, theirs = cast("dict[str, Any]", left), cast("dict[str, Any]", right)
        assert mine.keys() == theirs.keys(), what
        for key in mine:
            assert_same(mine[key], theirs[key], f"{what}.{key}")
        return
    if isinstance(left, list):
        mine_list, theirs_list = cast("list[Any]", left), cast("list[Any]", right)
        assert len(mine_list) == len(theirs_list), what
        for index, (one, other) in enumerate(zip(mine_list, theirs_list, strict=True)):
            assert_same(one, other, f"{what}[{index}]")
        return
    assert np.asarray(left).tobytes() == np.asarray(right).tobytes(), what
