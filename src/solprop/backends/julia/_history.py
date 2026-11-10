# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transfer portable Adams history without exposing borrowed Julia storage."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float, Int, UInt8

from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._lower import RunTransfer
from solprop.backends.julia._ownership import JuliaHandle
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.exceptions import BackendError
from solprop.output.schema import MarchSnapshot

__all__ = [
    "CALL_COUNTERS",
    "CallRestore",
    "HistoryBuffers",
    "HistoryIdentity",
    "HistoryInfo",
    "export_history",
    "history_info",
    "lower_call_restore",
    "restore_history",
]

CALL_COUNTERS = (
    "startup",
    "predictor",
    "corrected",
    "rejected",
    "accepted_steps",
    "rejected_steps",
    "restarts",
    "failure_restarts",
    "dense_observations",
)
_INTEGER_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True, eq=False)
class CallRestore:
    """Keep validated cumulative counts and compact accepted orders.

    Attributes
    ----------
    counts : tuple of int
        Nonnegative counters in shared CallLedger order, fitting the engine's int64.
    orders : numpy.ndarray
        Owned ``[order, count]`` runs, shape ``(n_runs, 2)``. Never expand one value
        per historical step merely to restore a compact checkpoint record.
    """

    counts: tuple[int, ...]
    orders: Int[np.ndarray, "n_runs 2"]


def lower_call_restore(
    records: Mapping[str, object] | None, *, adams: bool
) -> CallRestore | None:
    """Validate saved call ledgers before allocating a numerical session.

    Parameters
    ----------
    records : mapping or None
        Shared start record containing ``logical`` and optional ``atomic`` ledgers.
    adams : bool
        Whether this method retains a logical Adams ledger.

    Returns
    -------
    CallRestore or None
        Detached cumulative counts and compact orders, absent without an Adams ledger.

    Raises
    ------
    BackendError
        For unsupported atomic counts, malformed logical data or int64 overflow.
    """
    if not records:
        return None
    if records.get("atomic"):
        raise BackendError("julia does not restore atomic response call ledgers")
    logical = records.get("logical")
    if not logical or not adams:
        return None
    if not isinstance(logical, Mapping) or set(logical) != {
        *CALL_COUNTERS,
        "orders",
    }:
        raise BackendError("logical call ledger fields do not match this schema")
    counts: list[int] = []
    for name in CALL_COUNTERS:
        value = logical[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= _INTEGER_MAX
        ):
            raise BackendError(f"invalid logical call counter {name}")
        counts.append(value)
    if sum(counts[:3]) > _INTEGER_MAX:
        raise BackendError("logical call ledger total exceeds Julia's int64 capacity")
    orders = _order_runs(logical["orders"], counts[4])
    return CallRestore(tuple(counts), orders)


def _order_runs(runs: object, accepted: int) -> Int[np.ndarray, "n_runs 2"]:
    """Validate and copy compact orders with the shared CallLedger schema.

    Parameters
    ----------
    runs : object
        Persisted list of positive integer ``[order, count]`` pairs.
    accepted : int
        Validated nonnegative accepted-step count.

    Returns
    -------
    numpy.ndarray
        Owned int64 pairs with adjacent equal orders combined.

    Raises
    ------
    BackendError
        For malformed pairs, overflow or a different number of accepted steps.
    """
    if not isinstance(runs, list):
        raise BackendError("logical call ledger orders must be [order, count] pairs")
    owned: list[tuple[int, int]] = []
    counted = 0
    for run in runs:
        if not isinstance(run, list) or len(run) != 2:
            raise BackendError(
                "logical call ledger orders must be [order, count] pairs; "
                "the flat form from earlier development builds is not supported"
            )
        order, count = run
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 < value <= _INTEGER_MAX
            for value in (order, count)
        ):
            raise BackendError("logical call ledger order pairs need positive integers")
        counted += count
        if owned and owned[-1][0] == order:
            owned[-1] = (order, owned[-1][1] + count)
        else:
            owned.append((order, count))
    if counted != accepted:
        raise BackendError(
            f"logical call ledger orders count {counted} steps, and the ledger "
            f"{accepted} accepted steps"
        )
    return np.array(owned, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class HistoryInfo:
    """Hold detached scalar facts about a pending committed Adams history.

    Attributes
    ----------
    z, proposal : float
        Committed position and proposed next length, metres.
    order, rejections : int
        Next method order and consecutive rejection count.
    last_rejected, collar_was_quiet : bool
        Controller flags retained by the portable schema.
    n_entries, n_slots : int
        Selected oldest-first history entries and full engine-store capacity.
    compensated : bool
        Whether sampled states and rates accompany the source store.
    nbytes : int
        Raw state and full numerical stores, excluding the small metadata vectors.
    """

    z: float
    proposal: float
    order: int
    rejections: int
    last_rejected: bool
    collar_was_quiet: bool
    n_entries: int
    n_slots: int
    compensated: bool
    nbytes: int


@dataclass(frozen=True, eq=False)
class HistoryBuffers:
    """Own disjoint C-order buffers borrowed for one Julia history transfer.

    Attributes
    ----------
    state : numpy.ndarray
        Raw internal complex128 state, ``(n_modes, n_pol, n_omega)``.
    positions : numpy.ndarray
        Actual sample positions in metres, ``(n_entries,)``, oldest first.
    identities : numpy.ndarray
        UInt8 sample kinds, accepted zero or predicted one, ``(n_entries,)``.
    slots : numpy.ndarray
        Zero-based int64 store rows, ``(n_entries,)``.
    store, states, rates : numpy.ndarray or None
        Full complex128 stores, ``(n_slots, 1, n_modes, n_pol, n_omega)``.
        Sampled states and rates are absent for plain history. The bridge roots
        these owners while Julia borrows their pointers; sessions retain no buffer.
    """

    state: Complex[np.ndarray, "n_modes n_pol n_omega"]
    positions: Float[np.ndarray, " n_entries"]
    identities: UInt8[np.ndarray, " n_entries"]
    slots: Int[np.ndarray, " n_entries"]
    store: Complex[np.ndarray, "n_slots 1 n_modes n_pol n_omega"]
    states: Complex[np.ndarray, "n_slots 1 n_modes n_pol n_omega"] | None = None
    rates: Complex[np.ndarray, "n_slots 1 n_modes n_pol n_omega"] | None = None


@dataclass(frozen=True)
class HistoryIdentity:
    """Keep portable history provenance without retaining a run's transfer arrays.

    Attributes
    ----------
    method : str
        Original method representation, retained as provenance only.
    key : str or None
        Shared method history key, independent of backend and execution settings.
    """

    method: str
    key: str | None


def _info(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int, z: float
) -> HistoryInfo | None:
    """Detach scalar history metadata within the guarded bridge.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned Adams session.
    generation : int
        Pending export generation.
    z : float
        Observed position in metres; a dense field behind the march has no history.

    Returns
    -------
    HistoryInfo or None
        Python scalars only, without any borrowed Julia wrapper.
    """
    result = runtime.engine.adams_history_info(handle.borrow(), generation, z)
    if result is None:
        return None
    metadata = result[0]
    return HistoryInfo(
        float(metadata.z),
        float(metadata.proposal),
        int(metadata.order),
        int(metadata.rejections),
        bool(metadata.last_rejected),
        bool(metadata.collar_was_quiet),
        int(result[1]),
        int(result[2]),
        bool(result[3]),
        int(result[4]),
    )


def history_info(
    runtime: JuliaRuntime, handle: JuliaHandle, generation: int, z: float
) -> HistoryInfo | None:
    """Read eligible history size and metadata without copying any numerical array.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned Adams session.
    generation : int
        Pending export generation.
    z : float
        Observed position in metres.

    Returns
    -------
    HistoryInfo or None
        Detached metadata, absent before startup or at a dense field behind the march.
    """
    runtime.check()
    return invoke(
        lambda: _info(runtime, handle, generation, z),
        "the julia Adams history metadata read failed",
    )


def export_history(
    runtime: JuliaRuntime,
    handle: JuliaHandle,
    generation: int,
    z: float,
    shape: tuple[int, int, int],
    identity: HistoryIdentity,
) -> MarchSnapshot | None:
    """Copy raw accepted state and all portable history rows into owned Python arrays.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned Adams session.
    generation : int
        Pending export generation.
    z : float
        Observed position in metres.
    shape : tuple of int
        Host state shape ``(n_modes, n_pol, n_omega)``.
    identity : HistoryIdentity
        Method provenance and shared history identity.

    Returns
    -------
    solprop.output.schema.MarchSnapshot or None
        Lossless detached snapshot, absent without eligible history.
    """
    info = history_info(runtime, handle, generation, z)
    if info is None:
        return None
    stored = (info.n_slots, 1, *shape)
    buffers = HistoryBuffers(
        np.empty(shape, dtype=np.complex128),
        np.empty(info.n_entries, dtype=np.float64),
        np.empty(info.n_entries, dtype=np.uint8),
        np.empty(info.n_entries, dtype=np.int64),
        np.empty(stored, dtype=np.complex128),
        np.empty(stored, dtype=np.complex128) if info.compensated else None,
        np.empty(stored, dtype=np.complex128) if info.compensated else None,
    )
    # Each allocation is independent; offset foreign views cannot hide overlapping
    # storage from Julia's Array alias checks after nonowning pointer wrapping.
    from solprop.backends.julia._calls import call_void  # noqa: PLC0415

    call_void(runtime, "export_adams_history!", buffers, handle, generation, z)
    rows = {"store": buffers.store}
    if buffers.states is not None and buffers.rates is not None:
        rows.update(states=buffers.states, rates=buffers.rates)
    return MarchSnapshot(
        z=info.z,
        state=buffers.state,
        proposal=info.proposal,
        method=identity.method,
        order=info.order,
        rejections=info.rejections,
        last_rejected=info.last_rejected,
        positions=buffers.positions,
        identities=tuple(
            "predicted" if kind else "accepted" for kind in buffers.identities
        ),
        slots=buffers.slots,
        rows=rows,
        collar_was_quiet=info.collar_was_quiet,
        history_key=identity.key,
    )


def _history_refusal(
    snapshot: MarchSnapshot, transfer: RunTransfer, shape: tuple[int, int, int]
) -> str | None:
    """Check shared history identity, precision and portable row names.

    Parameters
    ----------
    snapshot : MarchSnapshot
        Portable candidate; its mutable arrays are never retained by Julia.
    transfer : RunTransfer
        Shared method identity and capacity.
    shape : tuple of int
        Required host state shape.

    Returns
    -------
    str or None
        Compatibility reason used by the shared reset record.
    """
    if transfer.adams is None:
        kind = "low-storage" if transfer.low_storage else "Runge-Kutta"
        return f"the run steps with the {kind} march, which keeps no history"
    if snapshot.history_key is None:
        return "the snapshot names no history key"
    if snapshot.history_key != transfer.history_key:
        return (
            f"the snapshot's history was kept by {snapshot.history_key}, and "
            f"this run steps with {transfer.history_key}"
        )
    allowed = {"store", "states", "rates"}
    if foreign := set(snapshot.rows) - allowed:
        return f"the julia Adams engine keeps no {sorted(foreign)} rows"
    if "store" not in snapshot.rows or (
        ("states" in snapshot.rows) != ("rates" in snapshot.rows)
    ):
        return "the snapshot needs a source store and paired sampled states/rates"
    return _storage_refusal(snapshot, transfer.adams.order + 2, shape)


def _storage_refusal(
    snapshot: MarchSnapshot, n_slots: int, shape: tuple[int, int, int]
) -> str | None:
    """Check exact host representation before any narrowing or pointer wrapping.

    Parameters
    ----------
    snapshot : MarchSnapshot
        Candidate raw history.
    n_slots : int
        Full engine store capacity.
    shape : tuple of int
        Required host state shape.

    Returns
    -------
    str or None
        Shape/type refusal, or None; numerical compatibility is checked by Julia.
    """
    if (
        not isinstance(snapshot.state, np.ndarray)
        or snapshot.state.shape != shape
        or snapshot.state.dtype != np.complex128
    ):
        return "the snapshot state needs matching binary64 complex storage"
    stored = (n_slots, 1, *shape)
    for name, row in snapshot.rows.items():
        if (
            not isinstance(row, np.ndarray)
            or row.shape != stored
            or row.dtype != np.complex128
        ):
            return f"the snapshot {name!r} rows do not fit binary64 history {stored}"
    if any(kind not in ("accepted", "predicted") for kind in snapshot.identities):
        return "the snapshot sample kind is neither accepted nor predicted"
    if snapshot.positions.dtype != np.float64 or snapshot.slots.dtype != np.int64:
        return "the snapshot needs binary64 positions and int64 slots"
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= _INTEGER_MAX
        for value in (snapshot.order, snapshot.rejections)
    ):
        return "the snapshot next order or rejection count is not a nonnegative int64"
    return None


def _lower_history(
    snapshot: MarchSnapshot, transfer: RunTransfer, shape: tuple[int, int, int]
) -> HistoryBuffers | str:
    """Detach compatible input arrays before numerical restoration.

    Parameters
    ----------
    snapshot : MarchSnapshot
        Portable candidate; its arrays are never retained by Julia.
    transfer : RunTransfer
        Shared method identity and capacity.
    shape : tuple of int
        Required host state shape.

    Returns
    -------
    HistoryBuffers or str
        Owned buffers, or a compatibility reason before allocating them.
    """
    if reason := _history_refusal(snapshot, transfer, shape):
        return reason
    rows = {
        name: np.array(row, dtype=np.complex128, order="C", copy=True)
        for name, row in snapshot.rows.items()
    }
    return HistoryBuffers(
        np.array(snapshot.state, dtype=np.complex128, order="C", copy=True),
        np.array(snapshot.positions, dtype=np.float64, order="C", copy=True),
        np.array([kind == "predicted" for kind in snapshot.identities], dtype=np.uint8),
        np.array(snapshot.slots, dtype=np.int64, order="C", copy=True),
        rows["store"],
        rows.get("states"),
        rows.get("rates"),
    )


def _restore(
    runtime: JuliaRuntime,
    handle: JuliaHandle,
    snapshot: MarchSnapshot,
    buffers: HistoryBuffers,
) -> str | None:
    """Install already-admitted buffers inside the guarded bridge.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Fresh unadvanced Adams session.
    snapshot : MarchSnapshot
        Scalar metadata to restore.
    buffers : HistoryBuffers
        Detached and disjoint source allocations.

    Returns
    -------
    str or None
        Detached compatibility reason, or None after successful installation.
    """
    reason = getattr(runtime.engine, "restore_adams_history!")(
        handle.borrow(), snapshot, buffers
    )
    return None if reason is None else str(reason)


def restore_history(
    runtime: JuliaRuntime,
    handle: JuliaHandle,
    snapshot: MarchSnapshot,
    transfer: RunTransfer,
    shape: tuple[int, int, int],
) -> str | None:
    """Restore compatible raw history transactionally, preserving field-reset reasons.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Fresh session after its field and cumulative ledger have been installed.
    snapshot : MarchSnapshot
        Portable history, possibly produced by another supported backend.
    transfer : RunTransfer
        Shared method identity and capacity.
    shape : tuple of int
        Required host state shape.

    Returns
    -------
    str or None
        Why history restarts from the loaded field, or None after restoration.

    Raises
    ------
    BackendError
        For an unexpected bridge/copy failure, which must not masquerade as a reset.
    """
    runtime.check()
    buffers = _lower_history(snapshot, transfer, shape)
    if isinstance(buffers, str):
        return buffers
    return invoke(
        lambda: _restore(runtime, handle, snapshot, buffers),
        "the julia Adams history restore failed",
    )
