# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Detach bounded Julia group transports without exposing numerical owners."""

from dataclasses import dataclass
from typing import Any

from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._calls import EngineReport, _report
from solprop.backends.julia._ownership import JuliaHandle
from solprop.backends.julia._runtime import JuliaRuntime

__all__ = [
    "GroupReport",
    "MemberReport",
    "advance",
    "fail",
    "members",
    "memory",
    "report",
]


@dataclass(frozen=True)
class GroupReport:
    """Retain the detached facts of one pending numerical group transport.

    Attributes
    ----------
    generation : int
        Pending transport identity.
    kind : int
        Zero for progress, one for service, two for window, three for termination.
    window : int
        Minus one for launch, then the zero-based common plane ordinal.
    outcome : int
        Zero for running, one for completed, two for stopped, three for failed.
    primary : int
        Zero-based failing member; minus one for an external cause.
    """

    generation: int
    kind: int
    window: int
    outcome: int
    primary: int


@dataclass(frozen=True)
class MemberReport:
    """Retain a member's event and its independent exactly-once chunk identity.

    Attributes
    ----------
    available, new_event, eligible : bool
        Whether this window holds the member, has a new event and allows export.
    chunk : int
        Positive delivery identity, unchanged by a forced terminal replacement.
    report : EngineReport
        Detached pending member scalars, with the member's own generation.
    """

    available: bool
    new_event: bool
    eligible: bool
    chunk: int
    report: EngineReport


# JuliaCall values are dynamic only within these invoke-guarded helpers.
def _detached(runtime: JuliaRuntime, value: Any) -> GroupReport:
    """Copy a borrowed group report while protected by the bridge guard.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified runtime holding the scalar conversion methods.
    value : object
        Borrowed Julia report; never returned to outer Python frames.

    Returns
    -------
    GroupReport
        Owned Python scalars.
    """
    return GroupReport(
        int(value.generation),
        int(runtime.engine.Int(value.kind)),
        int(value.window),
        int(runtime.engine.Int(value.outcome)),
        int(value.failure_member) - 1,
    )


def _advance(runtime: JuliaRuntime, handle: JuliaHandle) -> GroupReport:
    """Advance and detach one bounded transport inside the bridge guard.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified runtime.
    handle : JuliaHandle
        Group owner without a pending transport.

    Returns
    -------
    GroupReport
        Owned Python scalars from the GIL-free numerical call.
    """
    value = runtime.engine.advance_group_session._jl_call_nogil(
        handle.borrow(), 256, 0.1
    )
    return _detached(runtime, value)


def advance(runtime: JuliaRuntime, handle: JuliaHandle) -> GroupReport:
    """Advance a bounded transport, letting Julia select and run member turns.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified main-thread runtime.
    handle : JuliaHandle
        Group owner with no pending transport.

    Returns
    -------
    GroupReport
        Detached group facts.
    """
    runtime.check()
    return invoke(lambda: _advance(runtime, handle), "Julia group advancement failed")


def report(runtime: JuliaRuntime, handle: JuliaHandle, generation: int) -> GroupReport:
    """Read a pending transport after external service changed its outcome.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified main-thread runtime.
    handle : JuliaHandle
        Group owner.
    generation : int
        Pending group identity.

    Returns
    -------
    GroupReport
        Current detached group facts.
    """
    runtime.check()
    return invoke(
        lambda: _detached(
            runtime, runtime.engine.group_report(handle.borrow(), generation)
        ),
        "Julia group report failed",
    )


def _members(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int, count: int
) -> tuple[MemberReport, ...]:
    """Detach each member report inside one bridge guard.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified runtime.
    handle : JuliaHandle
        Group owner with a held transport.
    generation : int
        Pending group identity.
    count : int
        Number of members.

    Returns
    -------
    tuple of MemberReport
        Owned Python facts in member order.
    """
    group = handle.borrow()
    engine = runtime.engine
    results = []
    for index in range(1, count + 1):
        value = engine.group_member_report(group, generation, index)
        results.append(
            MemberReport(
                bool(value.available),
                bool(value.new_event),
                bool(value.eligible),
                int(value.chunk),
                _report(engine._bridge_report(value.report)),
            )
        )
    return tuple(results)


def members(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int, count: int
) -> tuple[MemberReport, ...]:
    """Copy every member's current scalar event without draining any chunk.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified main-thread runtime.
    handle : JuliaHandle
        Group owner.
    generation : int
        Pending group identity.
    count : int
        Number of independently owned members.

    Returns
    -------
    tuple of MemberReport
        Scalars in member order; no borrowed Julia values escape.
    """
    runtime.check()
    return invoke(
        lambda: _members(runtime, handle, generation, count),
        "Julia group member reports failed",
    )


def fail(runtime: JuliaRuntime, handle: JuliaHandle) -> GroupReport:
    """Hold a failed transport without advancing after an external service failure.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified main-thread runtime.
    handle : JuliaHandle
        Group owner.

    Returns
    -------
    GroupReport
        Pending terminal facts, whether the previous transport was acknowledged or held.
    """
    runtime.check()
    return invoke(
        lambda: _detached(runtime, runtime.engine.fail_group_session(handle.borrow())),
        "Julia group failure publication failed",
    )


def _memory(runtime: JuliaRuntime, handle: JuliaHandle) -> dict[str, int]:
    """Detach the group's numerical allocation counts inside the bridge guard.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified runtime.
    handle : JuliaHandle
        Open group owner.

    Returns
    -------
    dict
        Byte counts excluding member-owned buffers.
    """
    result = runtime.engine.group_memory(handle.borrow())
    return {
        name: int(getattr(result, name))
        for name in ("replay_origins", "plane_edges", "slots", "stop")
    }


def memory(runtime: JuliaRuntime, handle: JuliaHandle) -> dict[str, int]:
    """Read only the group's bounded numerical metadata allocations.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified main-thread runtime.
    handle : JuliaHandle
        Open group owner.

    Returns
    -------
    dict
        Byte counts, excluding buffers owned by individual member sessions.
    """
    runtime.check()
    return invoke(lambda: _memory(runtime, handle), "Julia group memory read failed")
