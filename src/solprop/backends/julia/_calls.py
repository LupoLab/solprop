# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep every temporary Julia wrapper inside a guarded main-thread helper."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Int

from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._ownership import JuliaHandle
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.protocol import BackendHistory

__all__ = [
    "EngineExecution",
    "EngineReport",
    "advance",
    "attempts",
    "budget",
    "call_void",
    "execution",
    "fail",
    "failure",
    "memory",
    "snapshot",
    "status",
]


@dataclass(frozen=True)
class EngineExecution:
    """Hold detached cumulative accounting and one bounded accepted-order chunk.

    Attributes
    ----------
    source_evaluations : int
        Actual source invocations, including calls that failed after entry.
    response_calls : tuple of int or None
        Adams logical counters in the shared CallLedger order, absent for other methods.
    orders : numpy.ndarray
        One integer per completed attempt in the pending chunk, zero for rejections.
        Empty for a method without Adams orders. The receiving session owns encoding.
    """

    source_evaluations: int
    response_calls: tuple[int, ...] | None
    orders: Int[np.ndarray, " n_attempts"]


def _execution(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> EngineExecution:
    """Detach cumulative counters and copy the current pending method-order chunk.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized runtime inside the guarded main-thread call.
    handle : JuliaHandle
        Numerical session with a pending event.
    generation : int
        The bound pending generation.

    Returns
    -------
    EngineExecution
        Python-owned counters and orders, without draining numerical storage.
    """
    session = handle.borrow()
    result = runtime.engine.session_execution(session, generation)
    logical = result[1]
    if logical is None:
        return EngineExecution(int(result[0]), None, np.empty(0, dtype=np.int64))
    counts = tuple(int(value) for value in logical)
    orders = np.empty(int(session.attempts.count), dtype=np.int64)
    getattr(runtime.engine, "session_orders!")(orders, session, generation)
    return EngineExecution(int(result[0]), counts, orders)


def execution(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> EngineExecution:
    """Copy one pending generation's accounting into detached Python storage.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    generation : int
        Current pending event generation.

    Returns
    -------
    EngineExecution
        Logical counts, actual source invocations and bounded accepted-order data.
    """
    runtime.check()
    return invoke(
        lambda: _execution(runtime, handle, generation),
        "the julia execution accounting read failed",
    )


def _budget(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> tuple[int, int, int]:
    """Detach maximum, spent and required charges inside the guarded bridge.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    generation : int
        Pending budget-failure generation.

    Returns
    -------
    tuple of int
        Maximum, spent and required logical calls.
    """
    result = runtime.engine.session_budget(handle.borrow(), generation)
    return int(result[0]), int(result[1]), int(result[2])


def budget(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> tuple[int, int, int]:
    """Read a pending Adams budget refusal without recomputing its admission.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    generation : int
        Pending budget-failure generation.

    Returns
    -------
    tuple of int
        Maximum, spent and required logical calls.
    """
    runtime.check()
    return invoke(
        lambda: _budget(runtime, handle, generation),
        "the julia budget failure read failed",
    )


@dataclass(frozen=True)
class EngineReport:
    """Hold detached scalar facts from a closed Julia numerical event.

    Attributes
    ----------
    kind : int
        Engine event code, from launch through terminal failure.
    z : float
        Committed position, metres.
    plane, generation : int
        Observed plane index and pending export generation.
    attempts, rejected : int
        Cumulative completed attempt counts.
    plane_attempts, plane_rejected : int
        Counts since the previous observation, including intervening yields.
    max_norm, mean_norm : float
        Dimensionless accepted norms since the previous observation.
    wall_time : float
        Numerical work since the previous observation, seconds.
    met_tolerance, poisoned : bool
        Whether accepted norms met tolerance and whether an in-flight failure
        invalidated the committed field.
    """

    kind: int
    z: float
    plane: int
    generation: int
    attempts: int
    rejected: int
    plane_attempts: int
    plane_rejected: int
    max_norm: float
    mean_norm: float
    wall_time: float
    met_tolerance: bool
    poisoned: bool


def _advance(
    runtime: JuliaRuntime, handle: JuliaHandle, work_limit: int, deadline: float
) -> EngineReport:
    # JuliaCall's dynamic values never leave this helper or its sanitized failure.
    """Advance with the GIL released and return only detached event scalars.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    work_limit : int
        Positive attempt bound.
    deadline : float
        Nonnegative finite work duration, seconds.

    Returns
    -------
    EngineReport
        Python-only numerical event facts.
    """
    function = runtime.engine.advance_session
    result = function._jl_call_nogil(handle.borrow(), work_limit, deadline)
    return _report(result)


# The temporary Julia tuple is dynamic and stays within an invoke-guarded helper.
def _report(result: Any) -> EngineReport:
    """Detach the fixed engine report from its temporary Julia tuple.

    Parameters
    ----------
    result : object
        The Julia tuple returned inside a guarded bridge call.

    Returns
    -------
    EngineReport
        Python scalar values, retaining no Julia wrapper.
    """
    return EngineReport(
        int(result[0]),
        float(result[1]),
        int(result[2]),
        int(result[3]),
        int(result[4]),
        int(result[5]),
        int(result[6]),
        int(result[7]),
        float(result[8]),
        float(result[9]),
        float(result[10]),
        bool(result[11]),
        bool(result[12]),
    )


def _fail(runtime: JuliaRuntime, handle: JuliaHandle) -> EngineReport:
    """Bind an external failure to a new terminal barrier without numerical work.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Open numerical session without a pending or terminal event.

    Returns
    -------
    EngineReport
        The terminal generation and unchanged committed-state facts.
    """
    return _report(runtime.engine.fail_session(handle.borrow()))


def fail(runtime: JuliaRuntime, handle: JuliaHandle) -> EngineReport:
    """Bind an external failure to a new terminal barrier without numerical work.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Open numerical session without a pending or terminal event.

    Returns
    -------
    EngineReport
        The terminal generation and unchanged committed-state facts.
    """
    runtime.check()
    return invoke(lambda: _fail(runtime, handle), "the julia failure barrier failed")


def advance(
    runtime: JuliaRuntime, handle: JuliaHandle, work_limit: int, deadline: float
) -> EngineReport:
    """Advance with the GIL released and return only detached event scalars.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    work_limit : int
        Positive attempt bound.
    deadline : float
        Nonnegative finite work duration, seconds.

    Returns
    -------
    EngineReport
        Python-only numerical event facts.
    """
    runtime.check()
    return invoke(
        lambda: _advance(runtime, handle, work_limit, deadline),
        "the julia numerical advance failed",
    )


def _call_void(runtime: JuliaRuntime, name: str, arguments: tuple[object, ...]) -> None:
    """Resolve owned handles and discard a void call's temporary result.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized runtime inside the guarded call.
    name : str
        Engine entry point.
    arguments : tuple of object
        Python-owned values or registry handles.
    """
    resolved = tuple(
        value.borrow() if isinstance(value, JuliaHandle) else value
        for value in arguments
    )
    getattr(runtime.engine, name)(*resolved)


def call_void(runtime: JuliaRuntime, name: str, *arguments: object) -> None:
    """Call a void engine operation without letting its return wrapper escape.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    name : str
        Engine entry point.
    *arguments : object
        Python-owned values or registry handles, resolved only inside the helper.
    """
    runtime.check()
    invoke(
        lambda: _call_void(runtime, name, arguments),
        f"the julia {name} operation failed",
    )


def _attempts(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> BackendHistory:
    """Copy a pending attempt chunk into detached Python-owned arrays.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    generation : int
        Current pending event generation.

    Returns
    -------
    solprop.backends.protocol.BackendHistory
        The pending chunk; no counter or numerical state changes.
    """
    session = handle.borrow()
    count = int(session.attempts.count)
    positions, lengths, norms, elapsed = (np.empty(count) for _ in range(4))
    accepted = np.empty(count, dtype=np.bool_)
    getattr(runtime.engine, "session_attempts!")(
        positions, lengths, norms, accepted, elapsed, session, generation
    )
    return BackendHistory(positions, lengths, norms, accepted, elapsed)


def attempts(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int
) -> BackendHistory:
    """Copy a pending attempt chunk into detached Python-owned arrays.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.
    generation : int
        Current pending event generation.

    Returns
    -------
    solprop.backends.protocol.BackendHistory
        The pending chunk; no counter or numerical state changes.
    """
    runtime.check()
    return invoke(
        lambda: _attempts(runtime, handle, generation),
        "the julia attempt export failed",
    )


def _status(
    runtime: JuliaRuntime, handle: JuliaHandle
) -> tuple[float, int, int, int, int, bool]:
    """Return detached position, plane, counters, generation and poisoning status.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.

    Returns
    -------
    tuple
        Position in metres, next plane, attempts, rejections, generation and poisoning.
    """
    result = runtime.engine.session_status(handle.borrow())
    return (
        float(result[0]),
        int(result[1]),
        int(result[2]),
        int(result[3]),
        int(result[4]),
        bool(result[5]),
    )


def status(
    runtime: JuliaRuntime, handle: JuliaHandle
) -> tuple[float, int, int, int, int, bool]:
    """Return detached position, plane, counters, generation and poisoning status.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.

    Returns
    -------
    tuple
        Position in metres, next plane, attempts, rejections, generation and poisoning.
    """
    runtime.check()
    return invoke(lambda: _status(runtime, handle), "the julia status read failed")


def _snapshot(runtime: JuliaRuntime, handle: JuliaHandle) -> tuple[int, bool]:
    """Latch a snapshot and return its sequence and coalescing flag.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.

    Returns
    -------
    tuple
        Integer request sequence and Boolean coalescing flag.
    """
    result = getattr(runtime.engine, "request_snapshot!")(handle.borrow())
    return int(result[0]), bool(result[1])


def snapshot(runtime: JuliaRuntime, handle: JuliaHandle) -> tuple[int, bool]:
    """Latch a snapshot and return its sequence and coalescing flag.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.

    Returns
    -------
    tuple
        Integer request sequence and Boolean coalescing flag.
    """
    runtime.check()
    return invoke(
        lambda: _snapshot(runtime, handle), "the julia snapshot request failed"
    )


def _failure(
    runtime: JuliaRuntime, handle: JuliaHandle
) -> tuple[str, str, bool, float, float]:
    """Read a terminal failure without retaining Julia's exception or traceback.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.

    Returns
    -------
    tuple
        Failure code/message, in-flight flag and abandoned start/length in metres.
    """
    result = runtime.engine.session_failure(handle.borrow())
    return (
        str(result[0]),
        str(result[1]),
        bool(result[2]),
        float(result[3]),
        float(result[4]),
    )


def failure(
    runtime: JuliaRuntime, handle: JuliaHandle
) -> tuple[str, str, bool, float, float]:
    """Read a terminal failure without retaining Julia's exception or traceback.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session.

    Returns
    -------
    tuple
        Failure code/message, in-flight flag and abandoned start/length in metres.
    """
    runtime.check()
    return invoke(
        lambda: _failure(runtime, handle), "the julia failure record read failed"
    )


def _memory(
    runtime: JuliaRuntime, handle: JuliaHandle, preparation: bool
) -> dict[str, int]:
    """Return detached retained-buffer sizes from a numerical session.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session or immutable preparation.
    preparation : bool, optional
        Read shared preparation storage instead of one session's buffers.

    Returns
    -------
    dict
        Buffer names mapped to bytes; the engine's field never crosses the boundary.
    """
    function = (
        runtime.engine.preparation_memory
        if preparation
        else runtime.engine.session_memory
    )
    result = function(handle.borrow())
    names = runtime.namespace.seval("propertynames")(result)
    return {str(name): int(getattr(result, str(name))) for name in names}


def memory(
    runtime: JuliaRuntime, handle: JuliaHandle, *, preparation: bool = False
) -> dict[str, int]:
    """Return detached retained-buffer sizes from a numerical session.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned numerical session or immutable preparation.
    preparation : bool, optional
        Read shared preparation storage instead of one session's buffers.

    Returns
    -------
    dict
        Buffer names mapped to bytes; the engine's field never crosses the boundary.
    """
    runtime.check()
    return invoke(
        lambda: _memory(runtime, handle, preparation),
        "the julia memory ledger read failed",
    )
