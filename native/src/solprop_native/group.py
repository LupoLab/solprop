# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native group session: compatible members marched through one fixed schedule.

:class:`NativeGroupSession` implements solprop's
:class:`~solprop.backends.session.GroupSession` (SEP 47 section 6.7) over one native
execution session per member (:class:`~solprop_native.session.NativeSession`). The
group owns its members explicitly, through their session protocol: each member's
native session is the one mutable owner of that member's trajectory -- its march, its
events and attempts, its periodic rows, its tickets and its boundary monitor -- and the
group only orders their advances, assembles each member's window from its chunks and
maps the group's tickets onto the members'. No member reaches another's state, and the
group reaches no private driver attribute.

**The window.** One :meth:`~NativeGroupSession.advance` runs every running member to
the next save plane of the common schedule (the launch first, the completion last).
Members take their turns in rounds of at most :data:`ROUND_STEPS` common steps: each
running member advances that many steps in turn, natively and without returning to
Python between steps but at the session's bounded yields, before the next takes its
turn. The schedule is common, so after every round the running members stand at one
common step, and the group-wide stop token is polled there.

**Failure** follows section 6.7. A boundary stop -- the monitor's error, or a boundary
warning raised as an error under the caller's filters -- ends its member alone, at the
point its own session reached, and the others go on. Any other failure of a member (a
non-finite norm, a step floor, a trial exception, an in-flight failure, a statistics
failure) ends the group: every member still running ends with that same error, at its
own committed position, with its own checkpoint eligibility. Members ahead of the
failing one in its round have already committed (and recorded) the steps of their turn;
no common cursor or rollback is invented. A failed window delivers each member's
attempts since its previous event in its terminal event, and no observation.

**Tickets.** Each member's latest native batch stays pending until that member's next
turn (acknowledging the group's batch releases the group's tickets only), so a member
the group ended before its turn can still be exported at its committed state, and a
group ticket -- which names its member, since every member's shares one batch and
generation -- is served by that member's own pending ticket. A member with no batch of
its own pending when the group ends (one that never launched, or whose own advance
raised) has nothing to export, and its terminal event is poisoned. An exception that
escapes a member's advance ends the group failed like a member's own failure, the
chunks drained so far delivered; an interrupt then propagates, the group's failed batch
left pending.

**Numerics.** A member's trajectory is its lone native session's: the march, the
monitor, the statistics and the export are the native session's own, so a member's
fields and statistics are those of the same problem run alone through the native
session -- bit for bit at one thread on the native CPU build, which is also what the
transitional lockstep group (:class:`solprop_native.batch.NativeBatchBackend`)
computes. The members share the schedule and nothing mutable; in particular they do not
share a fused accepted update (``_core.BatchState``): the transitional batch fuses that
one elementwise update by splitting every member's step around it, which the native
marches, which own the whole step, do not (SEP 47 section 6.7 asks for coarse groups,
not fused kernels).

:func:`refusal` names what the group does not run, decided from the members' problems,
requests and starts before anything is allocated; solprop's ``start_group`` then starts
the transitional group instead.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from solprop.backends.events import (
    Eligibility,
    Event,
    ExportTicket,
    GroupBatch,
    ObservationReady,
    Outcome,
    Progress,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
    join_attempts,
)
from solprop.backends.session import (
    SessionError,
    SessionLedger,
    SessionState,
    SessionStatus,
    check_group,
    check_member_ticket,
)
from solprop.exceptions import BackendError, BoundaryError, StopRequested

from . import _counters
from .backend import NativeBackend, _on_device  # pyright: ignore[reportPrivateUsage]
from .session import NativeSession
from .session import refusal as session_refusal

if TYPE_CHECKING:
    from types import TracebackType

    import numpy.typing as npt
    from solprop.backends.events import Receipt
    from solprop.backends.export import ExportRecipe
    from solprop.backends.protocol import BackendHistory, StepStats
    from solprop.backends.session import Start
    from solprop.log import StopFlag
    from solprop.translate.problem import KernelProblem
    from solprop.translate.run_request import RunRequest

__all__ = ["ROUND_STEPS", "NativeGroupSession", "refusal"]

ROUND_STEPS = 256
"""The most common steps a member takes in one turn, as one native advance makes at
most (``solprop.backends.session.DEFAULT_WORK_LIMIT``); the group-wide stop is polled
between rounds."""

STRATEGY = "native_group_session"
"""The group's ``batch_strategy`` in each member's execution record."""

_Chunk = tuple["BackendHistory | None", "StepStats | None"]


def refusal(
    problems: Sequence[KernelProblem],
    requests: Sequence[RunRequest],
    starts: Sequence[Start],
) -> str | None:
    """Return why the native group refuses a group, naming the feature, or ``None``.

    Decided from the members' problems, requests and starts alone, before anything is
    allocated. Each member must be a run the native session admits
    (:func:`solprop_native.session.refusal`), replaying a fixed schedule; the members
    must share the schedule, the save planes, the method, the field shape and the
    statistics request, as the lockstep group requires of them.

    Parameters
    ----------
    problems : collections.abc.Sequence of solprop.translate.problem.KernelProblem
        Each member's translated problem, at least one.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request, in member order.
    starts : collections.abc.Sequence of solprop.backends.session.Start
        Each member's start, in member order.

    Returns
    -------
    str or None
        The reason, or ``None`` when the native group admits the group.
    """
    if _on_device():
        return "a device build: the native group session runs on the CPU build only"
    for index, (problem, request, start) in enumerate(
        zip(problems, requests, starts, strict=True)
    ):
        refused = (
            # The group exports no noise phase, which a noisy plane stores.
            f"the {problem.initial.noise_model!r} noise model"
            if problem.initial.noise_model != "none"
            else session_refusal(problem, request, start) or _member_refusal(request)
        )
        if refused is not None:
            return f"{refused} (member {index})"
    first_problem, first = problems[0], requests[0]
    for index, (problem, request) in enumerate(
        zip(problems[1:], requests[1:], strict=True), start=1
    ):
        if not _same_group(problem, request, first_problem, first):
            return (
                f"member {index}, which does not share member 0's schedule, save "
                "planes, method, field shape or statistics request"
            )
    return None


def _member_refusal(request: RunRequest) -> str | None:
    """Return why one member's request falls outside a group's windows, or ``None``.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The member's run request.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    if request.replay is None:
        return "an adaptive march: a group replays one common fixed schedule"
    observation = request.observation
    if observation.step_planes:
        return "step planes, which a group's save-plane windows do not observe"
    if observation.dense:
        return "interpolated planes"
    return None


def _same_group(
    problem: KernelProblem,
    request: RunRequest,
    first_problem: KernelProblem,
    first: RunRequest,
) -> bool:
    """Return whether a member shares member 0's schedule, planes, method and spec.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The member's problem.
    request : solprop.translate.run_request.RunRequest
        The member's request, which replays a schedule.
    first_problem : solprop.translate.problem.KernelProblem
        Member 0's problem.
    first : solprop.translate.run_request.RunRequest
        Member 0's request, which replays a schedule.

    Returns
    -------
    bool
        True when they agree.
    """
    assert request.replay is not None
    assert first.replay is not None
    schedule, reference = request.replay.schedule, first.replay.schedule
    spec, reference_spec = request.observation.stats, first.observation.stats
    return (
        schedule.z_start == reference.z_start
        and np.array_equal(schedule.steps, reference.steps)
        and np.array_equal(
            request.observation.save_planes, first.observation.save_planes
        )
        and problem.stepper.integrator == first_problem.stepper.integrator
        and problem.shape == first_problem.shape
        and (spec.names, spec.periodic, spec.period)
        == (reference_spec.names, reference_spec.periodic, reference_spec.period)
    )


def _bound(event: Event, receipt: Receipt, member: int) -> Event:
    """Return a member's event with its ticket, if any, rebound to the group's batch.

    Parameters
    ----------
    event : solprop.backends.events.Event
        The member's event, its ticket the member session's.
    receipt : solprop.backends.events.Receipt
        The group batch's receipt.
    member : int
        The member's position, which the group's ticket names.

    Returns
    -------
    solprop.backends.events.Event
        The event, its ticket naming the group's batch and the member; the same event
        without one.
    """
    ticket = cast("ExportTicket | None", getattr(event, "ticket", None))
    if ticket is None:
        return event
    return cast(
        "Event",
        dataclasses.replace(
            cast("Any", event),
            ticket=ExportTicket(receipt, ticket.generation, ticket.z, member),
        ),
    )


class _Window:
    """One member's progress through the current window: its chunks and its end.

    Attributes
    ----------
    chunks : list of tuple
        The attempts and rows of each batch the member delivered in this window.
    event : solprop.backends.events.Event or None
        The window's end for the member: its observation or its terminal event, its
        history the window's chunks joined; ``None`` while it runs.
    """

    def __init__(self) -> None:
        """Start an empty window."""
        self.chunks: list[_Chunk] = []
        self.event: Event | None = None

    def joined(self) -> tuple[BackendHistory, StepStats]:
        """Return the window's attempts and rows so far, as one record.

        Returns
        -------
        tuple
            The joined history and rows.
        """
        return join_attempts(self.chunks)


class NativeGroupSession:
    """A :class:`~solprop.backends.session.GroupSession` of native sessions.

    Parameters
    ----------
    members : collections.abc.Sequence of solprop_native.backend.NativeBackend
        One constructed backend per member, each built for its problem and holding no
        state. Once the group is constructed it owns them, each through its member's
        native session, and closes them with itself; a construction that fails closes
        the sessions it started.
    problems : collections.abc.Sequence of solprop.translate.problem.KernelProblem
        Each member's independently translated problem.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request, with its own fresh boundary monitor; their schedule,
        save planes and statistics are the group's common ones.
    starts : collections.abc.Sequence of solprop.backends.session.Start
        Each member's entrance.
    stop : solprop.log.StopFlag or None, optional
        A group-wide stop token, polled between rounds of common steps.

    Raises
    ------
    solprop.backends.session.SessionError
        If the sequences differ in length or are empty, or if a start is not a fresh
        entrance.
    solprop.exceptions.BackendError
        Naming what the native group refuses (:func:`refusal`), before anything is
        allocated, or if a member is not a native backend.
    """

    def __init__(
        self,
        members: Sequence[NativeBackend],
        problems: Sequence[KernelProblem],
        requests: Sequence[RunRequest],
        starts: Sequence[Start],
        *,
        stop: StopFlag | None = None,
    ) -> None:
        """Refuse what the native group does not run, then start each member."""
        check_group(len(members), problems, requests, starts)
        if any(
            not isinstance(cast("object", member), NativeBackend) for member in members
        ):
            msg = "the native group session takes native backends"
            raise BackendError(msg)
        refused = refusal(problems, requests, starts)
        if refused is not None:
            msg = f"the native group session does not run {refused}"
            raise BackendError(msg)
        sessions: list[NativeSession] = []
        try:
            for member, problem, request, start in zip(
                members, problems, requests, starts, strict=True
            ):
                sessions.append(NativeSession(problem, request, start, backend=member))
        except BaseException:
            for session in sessions:
                session.close()
            raise
        self._sessions = tuple(sessions)
        self._planes = int(np.asarray(requests[0].observation.save_planes).size)
        self._stop = stop
        self._ledger = SessionLedger()
        self._launched = False
        self._index = 0
        count = len(self._sessions)
        self._outcomes: list[Outcome | None] = [None] * count
        # Each member's pending native batch, acknowledged at its next turn.
        self._pending: list[Receipt | None] = [None] * count
        # The member tickets the pending group batch's tickets stand for.
        self._tickets: dict[int, ExportTicket] = {}

    # -- protocol -------------------------------------------------------------------

    @property
    def n_members(self) -> int:
        """Return the number of members.

        Returns
        -------
        int
            The members.
        """
        return len(self._sessions)

    @property
    def finished(self) -> bool:
        """Return whether the group's terminal batch was issued.

        Returns
        -------
        bool
            True once every member has ended.
        """
        return self._ledger.finished

    @property
    def closed(self) -> bool:
        """Return whether the group was closed.

        Returns
        -------
        bool
            True once closed.
        """
        return self._ledger.state is SessionState.CLOSED

    def advance(self) -> GroupBatch:
        """Run every running member through the next common window, and report it.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each member's events in the window.
        """
        self._ledger.check_advance()
        self._tickets = {}
        if not self._launched:
            self._launched = True
            return self._window(stepping=False, final=False)
        if self._index >= self._planes:
            return self._window(stepping=False, final=True)
        self._index += 1
        return self._window(stepping=True, final=False)

    def acknowledge(self, receipt: Receipt) -> None:
        """Release the pending batch and allow advancing past it.

        Each member's own pending batch is acknowledged at its next turn, so that a
        member the group ends before then can still be exported.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.
        """
        self._ledger.acknowledge(receipt)
        self._tickets = {}

    def export(
        self,
        member: int,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: npt.NDArray[np.complex128] | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Fill a destination with one member's converted field at a pending ticket.

        Parameters
        ----------
        member : int
            The member's position.
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch, for this member.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination; a new array by default.

        Returns
        -------
        numpy.ndarray
            The member's converted field, exported natively by its session.

        Raises
        ------
        solprop.backends.session.SessionError
            If the ticket expired, or is not one of this member's.
        """
        self._ledger.check_ticket(ticket)
        check_member_ticket(ticket, self._member(member))
        held = self._tickets.get(member)
        if held is None or held.generation != ticket.generation:
            msg = (
                f"the export ticket for z = {ticket.z!r} m is not one of member "
                f"{member}'s in the pending batch"
            )
            raise SessionError(msg)
        return self._sessions[member].export(held, recipe, out)

    def noise_phase(self, member: int, ticket: ExportTicket) -> np.ndarray | None:
        """Return ``None``: the native group refuses noise models (:func:`refusal`).

        Parameters
        ----------
        member : int
            The member's position.
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch.

        Returns
        -------
        None
            No noise phase.
        """
        self._ledger.check_ticket(ticket)
        check_member_ticket(ticket, self._member(member))

    def member_status(self, member: int) -> SessionStatus:
        """Return one member's committed position, counters and outcome.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Its native session's position and counters; a member that ended reports
            its own outcome while the group goes on.
        """
        index = self._member(member)
        status = self._sessions[index].status()
        outcome = self._outcomes[index]
        state = self._ledger.state
        if outcome is not None and state is not SessionState.CLOSED:
            state = SessionState.FINISHED
        return SessionStatus(
            z=status.z,
            plane_index=status.plane_index,
            attempts=status.attempts,
            rejected=status.rejected,
            state=state,
            outcome=outcome,
        )

    def execution_record(self, member: int) -> Mapping[str, object]:
        """Return one member's execution facts, with the group's strategy.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        collections.abc.Mapping
            Its native session's record, with the group's size and strategy.
        """
        record = dict(self._sessions[self._member(member)].execution_record())
        record.update(
            batch_size=len(self._sessions),
            batch_strategy=STRATEGY,
            fused_batch_kernels=False,
        )
        return record

    def monitor_record(self, member: int) -> Mapping[str, object] | None:
        """Return one member's boundary-monitor record, which its session keeps.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        collections.abc.Mapping or None
            Its record, or ``None`` when unwatched.
        """
        return self._sessions[self._member(member)].monitor_record()

    def close(self) -> None:
        """Release every member's session and backend; idempotent."""
        if self._ledger.close():
            self._tickets = {}
            for session in self._sessions:
                session.close()

    def __enter__(self) -> "NativeGroupSession":  # noqa: UP037 - runtime type hook
        """Return the group.

        Returns
        -------
        NativeGroupSession
            This group.
        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the group.

        Parameters
        ----------
        kind : type or None
            The exception type, if one is propagating.
        error : BaseException or None
            The exception, if one is propagating.
        trace : types.TracebackType or None
            Its traceback.
        """
        del kind, error, trace
        self.close()

    # -- internals ------------------------------------------------------------------

    def _member(self, member: int) -> int:
        """Return a valid member position.

        Parameters
        ----------
        member : int
            The position asked for.

        Returns
        -------
        int
            The same position.

        Raises
        ------
        solprop.backends.session.SessionError
            If the group has no such member.
        """
        if not 0 <= member < len(self._sessions):
            msg = f"the group has {len(self._sessions)} members, not a member {member}"
            raise SessionError(msg)
        return member

    def _window(self, *, stepping: bool, final: bool) -> GroupBatch:
        """Run every running member to its window's end, in rounds; report it.

        Parameters
        ----------
        stepping : bool
            Whether the window takes common steps (a save plane's): the stop token is
            polled before its first round. The launch and the completion take none;
            any later round of any window is polled too.
        final : bool
            Whether it is the completion, which ends the group.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each member's events.
        """
        windows = {
            index: _Window()
            for index, outcome in enumerate(self._outcomes)
            if outcome is None
        }
        try:
            rounds = 0
            while active := [i for i, w in windows.items() if w.event is None]:
                # Before every round that may step: a save plane's first, and any
                # later round of any window, which only steps can have made necessary.
                stop = self._stop
                if (stepping or rounds) and stop is not None and stop.is_set():
                    return self._ended(Outcome.STOPPED, StopRequested(stop), windows)
                rounds += 1
                for index in active:
                    error = self._turn(index, windows[index])
                    if error is not None:
                        return self._ended(Outcome.FAILED, error, windows)
        except BaseException as escaped:
            # A failure outside a member's own terminal event -- the session refusing
            # a call, an exception a warning's display raised, an interrupt -- ends
            # the group with the chunks drained so far delivered; an interrupt then
            # propagates, the group's failed batch pending, so it advances no more.
            ended = self._ended(Outcome.FAILED, escaped, windows)
            if isinstance(escaped, Exception):
                return ended
            raise
        events: list[tuple[Event, ...]] = [()] * len(self._sessions)
        steps = 0
        for index, window in windows.items():
            assert window.event is not None
            events[index] = (window.event,)
            if window.event.history is not None:
                steps = max(steps, window.event.history.n_steps)
        if (counters := _counters.active) is not None:
            counters.group_window(steps)
        ended = final or all(outcome is not None for outcome in self._outcomes)
        return self._issue(events, outcome=Outcome.COMPLETED if ended else None)

    def _turn(self, index: int, window: _Window) -> BaseException | None:
        """Advance one member by a round of common steps, or to its window's end.

        Parameters
        ----------
        index : int
            The member's position.
        window : _Window
            Its progress through the window, extended here.

        Returns
        -------
        BaseException or None
            A group-wide failure the member reported, which ends the group; ``None``
            otherwise, its window's end, if reached, in `window`.
        """
        session = self._sessions[index]
        made = 0
        while made < ROUND_STEPS and window.event is None:
            receipt = self._pending[index]
            if receipt is not None:
                session.acknowledge(receipt)
                self._pending[index] = None
            batch = session.advance(work_limit=ROUND_STEPS - made)
            self._pending[index] = batch.receipt
            (event,) = batch.events
            if (counters := _counters.active) is not None:
                _count(counters, event)
            if isinstance(event, Progress):
                window.chunks.append((event.history, event.step_stats))
                made += event.history.n_steps
            elif isinstance(event, ServiceRequired):
                window.chunks.append((event.history, event.step_stats))
                made += 0 if event.history is None else event.history.n_steps
                self._serve(index, window, event)
            elif isinstance(event, ObservationReady):
                history, rows = join_attempts(
                    [*window.chunks, (event.history, event.step_stats)]
                )
                window.event = _with_chunk(event, history, rows)
            else:
                assert isinstance(event, Terminal)
                history, rows = join_attempts(
                    [*window.chunks, (event.history, event.step_stats)]
                )
                window.event = _with_chunk(event, history, rows)
                self._outcomes[index] = event.outcome
                group_wide = event.outcome is not Outcome.COMPLETED and not isinstance(
                    event.error, BoundaryError
                )
                if group_wide:
                    return event.error
        return None

    def _serve(self, index: int, window: _Window, event: ServiceRequired) -> None:
        """Serve a member's barrier: a warning is emitted; one raised ends the member.

        A boundary warning the caller's filters raise ends that member alone, at the
        warning's committed endpoint, with its field there eligible for a checkpoint,
        as a lone run of it ends (SEP 47 section 5.5). The group asks for no snapshot.

        Parameters
        ----------
        index : int
            The member's position.
        window : _Window
            Its window, ended here when the warning is raised.
        event : solprop.backends.events.ServiceRequired
            The barrier.
        """
        if event.kind is not ServiceKind.WARNING:
            return
        try:
            emit_warning(event, stacklevel=4)
        except Warning as raised:
            history, rows = window.joined()
            window.event = Terminal(
                Outcome.FAILED,
                event.z,
                Eligibility.ELIGIBLE,
                raised,
                event.ticket,
                history,
                rows,
            )
            self._outcomes[index] = Outcome.FAILED

    def _ended(
        self, outcome: Outcome, error: BaseException, windows: dict[int, _Window]
    ) -> GroupBatch:
        """End every running member with a group-wide stop or failure.

        Parameters
        ----------
        outcome : solprop.backends.events.Outcome
            Stopped or failed.
        error : BaseException
            The primary failure, which the failing member's own terminal carries, or
            the stop request.
        windows : dict
            The window's progress of each member running when it began.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each running member's terminal event at its own committed position; a
            member that ended in this window keeps its own terminal event.
        """
        events: list[tuple[Event, ...]] = [()] * len(self._sessions)
        for index, window in windows.items():
            if isinstance(window.event, Terminal):
                events[index] = (window.event,)
                continue
            # A member that reached its plane ends there, its observation's attempts
            # delivered; any other at the common step it stood at.
            ended = window.event
            history, rows = (
                (ended.history, ended.step_stats)
                if isinstance(ended, ObservationReady)
                else window.joined()
            )
            # A member holds an exportable state only while a batch of its own is
            # pending: one that never launched, or whose own advance raised, has
            # none, and claims no checkpoint.
            ticket = self._sessions[index].pending_ticket()
            self._outcomes[index] = outcome
            events[index] = (
                Terminal(
                    outcome,
                    self._sessions[index].status().z,
                    Eligibility.POISONED if ticket is None else Eligibility.ELIGIBLE,
                    error,
                    ticket,
                    history,
                    rows,
                ),
            )
        return self._issue(events, outcome=outcome, error=error)

    def _issue(
        self,
        members: list[tuple[Event, ...]],
        *,
        outcome: Outcome | None = None,
        error: BaseException | None = None,
    ) -> GroupBatch:
        """Issue a window's events under a new receipt, its tickets the group's.

        Parameters
        ----------
        members : list
            Each member's events, their tickets the member sessions'.
        outcome : solprop.backends.events.Outcome or None, optional
            The group's outcome, when the batch ends it.
        error : BaseException or None, optional
            A group-wide failure's primary error.

        Returns
        -------
        solprop.backends.events.GroupBatch
            The batch.
        """
        receipt = self._ledger.issue(outcome=outcome)
        for index, events in enumerate(members):
            for event in events:
                ticket = cast("ExportTicket | None", getattr(event, "ticket", None))
                if ticket is not None:
                    self._tickets[index] = ticket
        bound = tuple(
            tuple(_bound(event, receipt, index) for event in events)
            for index, events in enumerate(members)
        )
        return GroupBatch(receipt, bound, outcome, error)


def _count(counters: _counters.Counters, event: Event) -> None:
    """Count a batch a member's native session returned, for the benchmarks.

    Parameters
    ----------
    counters : solprop_native._counters.Counters
        The active counters.
    event : solprop.backends.events.Event
        The batch's event.
    """
    kind = {
        ObservationReady: "observation",
        Progress: "progress",
        ServiceRequired: "warning",
        Terminal: "terminal",
    }[type(event)]
    history = event.history
    attempts = 0 if history is None else history.n_steps
    accepted = 0 if history is None else attempts - history.n_rejected
    counters.session_batch(kind, attempts, accepted)


def _with_chunk(event: Event, history: BackendHistory, rows: StepStats) -> Event:
    """Return a member's window-ending event carrying the whole window's chunk.

    Parameters
    ----------
    event : solprop.backends.events.Event
        The observation or terminal event that ended the member's window.
    history : solprop.backends.protocol.BackendHistory
        The window's attempts, joined.
    rows : solprop.backends.protocol.StepStats
        Its periodic rows, joined.

    Returns
    -------
    solprop.backends.events.Event
        The event with that chunk.
    """
    return cast(
        "Event",
        dataclasses.replace(cast("Any", event), history=history, step_stats=rows),
    )
