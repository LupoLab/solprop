# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native march protocol: its returns, its errors and the Adams record.

Why a native march call returned and which error a failure maps to (``march::Reason``,
``march::Category``), the boundary watch's fresh state, the translation of a native
failure into solprop's exceptions, and the Adams march's record that a run starts
from (solprop SEP 39, SEP 47 section 5.2).
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np
from solprop.backends.exponential import carrier_reference
from solprop.backends.ledger import CallLedger, Derivative
from solprop.backends.records import adams_policy
from solprop.exceptions import BackendError, ConvergenceError
from solprop.output.records import MarchSnapshot
from solprop.stepper.multistep import MarchPolicy, Multistep

from . import _core

__all__ = [
    "FRESH_WATCH",
    "_BUDGET",
    "_DIVERGED",
    "_ERROR",
    "_FLOOR",
    "_NativeAdamsMarch",
    "_envelope_errors",
    "_raise_native",
]


class _envelope_errors:
    """Translate refinement failures without importing solprop in the extension.

    A class rather than a generator: it wraps every native step call.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        """Enter the native call."""

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Raise a refinement failure leaving the call as solprop's error."""
        del kind, traceback
        if isinstance(error, _core.EnvelopeConvergenceError):
            raise ConvergenceError(str(error)) from error
        if isinstance(error, _core.BackendRefusal):
            raise BackendError(str(error)) from error


FRESH_WATCH: Mapping[str, object] = types.MappingProxyType(
    {
        "has_last": False,
        "last_z": 0.0,
        "since": 0,
        "busy": False,
        "quiet_run": 0,
        "quiet": False,
        "warned": False,
        "cap": math.inf,
    }
)
"""The boundary watch's state at a run's start (``watch::State``): no check yet; a
session's monitor holds its own from there (solprop SEP 45 section 4.8)."""

_MARCH_COUNTERS = (
    "startup",
    "predictor",
    "rejected",
    "accepted_steps",
    "rejected_steps",
    "restarts",
    "failure_restarts",
    "dense_observations",
)
"""The call ledger's counters the native march keeps."""

_REASONS = (
    "reached",
    "host_step",
    "monitor",
    "bound",
    "budget",
    "floor",
    "error",
    "diverged",
    "watch",
    "capacity",
    "dense",
    "stopped",
    "halted",
)
"""Why a native march call returned (``march::Reason``), by name, in its order."""
(
    _REACHED,
    _HOST_STEP,
    _MONITOR,
    _BOUND,
    _BUDGET,
    _FLOOR,
    _ERROR,
    _DIVERGED,
    _WATCH,
    _CAPACITY,
    _DENSE,
    _STOPPED,
    _HALTED,
) = range(len(_REASONS))

_CONVERGENCE, _BACKEND, _VALUE, _REFUSED = 1, 2, 3, 4
"""Which error a native failure maps to (``march::Category``)."""


class _NativeAdamsMarch:
    """The native Adams march's record: its policy, ledger and history (SEP 39, 47).

    The native workspace steps solprop's exponential Adams march (solprop SEP 39 phase
    2): the controller, the history, the landings and the call ledger's counts advance
    natively, from one policy record (:func:`~solprop.backends.records.adams_policy`).
    This record is what the run starts from: the policy, the history's entries
    (position, identity and engine row), the order, the rejection flags and the
    :class:`~solprop.backends.ledger.CallLedger`, each fresh at a load (:meth:`reset`,
    :meth:`begin_run`) or restored from a resume's snapshot and ledger (:meth:`resume`,
    :meth:`restore_ledger`). A native session hands it to the workspace once, with the
    run's start (:meth:`start_state`), and holds the march's state there from then on
    (solprop SEP 47 section 5.2).

    `rows` are the names of the resident history's persistent rows
    (``Workspace.diagnostics``); `engine` is a scripted harness's own history.
    """

    def __init__(
        self,
        backend: Any,
        method: Multistep,
        rows: Sequence[str] = (),
        *,
        engine: object | None = None,
    ) -> None:
        """Build the record over the backend's resident history, with no history."""
        self._backend = backend
        self._method = method
        self._rows = tuple(str(name) for name in rows)
        self._engine: Any = engine
        stepper = backend._stepper
        problem = backend._problem
        self._history: list[Derivative] = []
        self._rejections = 0
        self._last_rejected = False
        self._order = 1 if method.variable_order else method.order
        self.ledger = CallLedger()
        # A scripted harness's history holds its own frame.
        carrier = (
            carrier_reference(problem, method)
            if engine is None
            else getattr(engine, "carrier", None)
        )
        # Every number the native controller and landings read (SEP 39 section 4.2),
        # formed as the reference forms it.
        self._policy: MarchPolicy = adams_policy(
            problem, stepper, method, carrier=carrier is not None and carrier.active
        )
        # Whether the stages left a quiet collar out at the last attempt.
        self._collar_was_quiet = False
        # The admission refuses a carrier frame on an operator with no chord
        # (``refuse_carrier_frame``).
        assert not (
            self._policy.carrier and carrier is not None and carrier.chord is None
        )

    @property
    def n_slots(self) -> int:
        """The history's rows: the method's order and two more."""
        return self._method.order + 2

    def snapshot_nbytes(self) -> int:
        """Return the bytes a snapshot of the history's rows copies, copying nothing."""
        shape = self._backend._problem.shape
        return len(self._rows) * self.n_slots * math.prod(shape) * 2 * 8

    # -- the record ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the history: the state was loaded outside the march."""
        self._history = []
        self._rejections = 0

    def begin_run(self) -> None:
        """Start a run's accounting afresh: the ledger and the last attempt's flags.

        A backend loaded for another run counts that run's calls alone and chooses its
        first order and step as a fresh backend would (SEP 22 section 4.7, SEP 39 R9).
        A resume restores its persisted ledger and flags afterwards.
        """
        self.ledger = CallLedger()
        self._last_rejected = False
        self._collar_was_quiet = False

    def restore_ledger(
        self, logical: Mapping[str, object], atomic: Mapping[str, object] | None = None
    ) -> None:
        """Restore a resumed run's logical call counts, not its history.

        - ``logical``: the persisted logical counts; empty installs nothing.
        - ``atomic``: persisted atomic counts, which only the TDSE reference keeps.
        """
        if atomic:
            raise BackendError("native does not restore atomic response call ledgers")
        if logical:
            self.ledger = CallLedger.from_record(logical)

    def resume(
        self, snapshot: MarchSnapshot, z: float
    ) -> tuple[str | None, tuple[dict[str, object], ...]]:
        """Take a snapshot's history where the run resumes, at `z`.

        Its entries, order and flags go to this record, and its rows are returned as
        the workspace installs them with the run's start; the snapshot's state and
        proposal are the caller's. A snapshot that cannot continue the march changes
        nothing.

        - ``z``: the loaded position, m.
        - Returns: ``None`` and the rows when the history continues, otherwise why the
          march starts afresh, as a resume did before snapshots were kept, and no rows.
        """
        reason = self._refusal(snapshot, z)
        if reason is not None:
            return reason, ()
        try:
            rows = self._rows_record(snapshot.rows, self._backend._problem.shape)
        except BackendError as error:
            return str(error), ()
        self._history = [
            Derivative(float(z), None, identity, int(slot))
            for z, identity, slot in zip(
                snapshot.positions, snapshot.identities, snapshot.slots, strict=True
            )
        ]
        self._order = snapshot.order
        self._rejections = snapshot.rejections
        self._last_rejected = snapshot.last_rejected
        self._collar_was_quiet = snapshot.collar_was_quiet
        return None, rows

    def _rows_record(
        self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]
    ) -> tuple[dict[str, object], ...]:
        """Return a snapshot's rows as the workspace takes them, for a state of `shape`.

        The rows keep the reference's names and layout (``AdamsRows.export``,
        ``numpy_ref/multistep.py``), ``(n_slots, n_modes, n_pol, n_omega)``:
        ``"store"`` and, compensated, ``"states"`` and
        ``"rates"``. A frozen-residual flow's history holds each sample's generator
        applied to its state, ``"applied"``, where the host engine holds the sample's
        weights (``"weights_<row>"``); the two are not interchangeable, and each engine
        refuses the other's. Everything is checked before any row is copied.
        """
        names = list(self._rows)
        weights = sorted(name for name in rows if name.startswith("weights_"))
        if weights:
            listed = ", ".join(weights)
            raise BackendError(
                "the native backend cannot continue this Adams history: the snapshot "
                f"holds the host engine's frozen-residual weights ({listed}), and the "
                "native engine keeps each sample's applied generator ('applied') "
                "instead"
            )
        foreign = sorted(set(rows) - set(names))
        missing = [name for name in names if name not in rows]
        if foreign or missing:
            raise BackendError(
                "the native backend cannot continue this Adams history: its engine "
                f"keeps the rows {names}, and the snapshot "
                + (f"lacks {missing}" if missing else f"also holds {foreign}")
            )
        expected = (self.n_slots, *shape)
        for name in names:
            row = rows[name]
            if row.shape != expected or not np.iscomplexobj(row):
                raise BackendError(
                    f"the snapshot's {name!r} rows ({row.dtype}, shape {row.shape}) do "
                    f"not fit the native backend's complex store of shape {expected}"
                )
        return tuple(
            {
                "name": name,
                "values": np.ascontiguousarray(rows[name], dtype=np.complex128).reshape(
                    self.n_slots, -1, shape[-1]
                ),
            }
            for name in names
        )

    def _refusal(self, snapshot: MarchSnapshot, z: float) -> str | None:
        """Return why a snapshot cannot continue this march at `z`, or ``None``."""
        shape = tuple(self._backend._problem.shape)
        checks = (
            (
                snapshot.history_key != self._method.history_key,
                f"the snapshot's history was kept by {snapshot.history_key}, and "
                f"this run steps with {self._method.history_key}",
            ),
            (
                snapshot.z != z,
                f"the snapshot is at z = {snapshot.z!r} m and the run resumes at "
                f"z = {z!r} m",
            ),
            (
                snapshot.state.shape != shape,
                f"the snapshot's state has shape {snapshot.state.shape}",
            ),
            (
                # A landing snaps to its plane, a rounding from the step's z + dz.
                not math.isclose(
                    float(snapshot.positions[-1]), snapshot.z, rel_tol=1e-12
                ),
                "the snapshot's newest sample is not at its position",
            ),
            (
                bool(np.any((snapshot.slots < 0) | (snapshot.slots >= self.n_slots))),
                "the snapshot's history names no row of this engine",
            ),
        )
        return next((reason for failed, reason in checks if failed), None)

    def start_state(self, z: float, proposal: float, cursor: int) -> dict[str, object]:
        """Return the march's state at a start, as the native call takes it.

        - ``z``: the loaded position, m.
        - ``proposal``: the next step's proposal, m: ``dz_init``, or a resumed
          snapshot's.
        - Returns: the position, the proposal, the order, the rejection flags, the
          cursor, the history's entries and the ledger's counts.
        """
        ledger = self.ledger
        return {
            "z": float(z),
            "proposal": float(proposal),
            "order": self._order,
            "rejections": self._rejections,
            "last_rejected": self._last_rejected,
            "collar_was_quiet": self._collar_was_quiet,
            "cursor": int(cursor),
            "history": [
                (entry.z, entry.identity, entry.slot) for entry in self._history
            ],
            "ledger": {name: getattr(ledger, name) for name in _MARCH_COUNTERS},
            "watch": dict(FRESH_WATCH),
        }

    def _raise(self, reason: int, result: dict[str, Any]) -> None:
        """Raise the error a native call returned with, if any.

        A session forms the call budget's own error (``NativeSession._budget_error``)
        before it asks here.
        """
        if reason == _FLOOR:
            proposed = float(result["floor_proposal"])
            msg = (
                f"step control asked for {proposed:.3e} m, below the floor dz_min ="
                f" {self._policy.dz_min:.3e} m; the tolerance cannot be met here"
            )
            raise ConvergenceError(msg)
        if reason == _DIVERGED:
            dz, end = float(result["replay_dz"]), float(result["replay_end"])
            norm = float(result["replay_norm"])
            msg = (
                f"{self._backend.name}: a replayed {self._method.name} step of "
                f"{dz:.4g} m ending at z = {end:.6g} m has a non-finite error norm "
                f"({norm!r}): the field has diverged. A replayed schedule takes "
                "its steps as given and cannot reject one, so a step beyond an "
                "explicit stability limit -- the in-stage temporal collar's or a "
                "coupled flow's advisory step -- grows until it overflows; the "
                "translation refuses a declared step past the collar's limit and "
                "warns on one past the flow's. Take shorter steps, or let the "
                "controller choose them"
            )
            raise ConvergenceError(msg)
        if reason == _ERROR:
            _raise_native(result)


def _raise_native(result: Mapping[str, Any]) -> None:
    """Raise a native march's error as the exception class its category names."""
    message = str(result["message"])
    category = int(result["category"])
    if category == _CONVERGENCE:
        raise ConvergenceError(message)
    if category == _VALUE:
        raise ValueError(message)
    if category == _REFUSED:
        raise BackendError(message)
    raise RuntimeError(message)
