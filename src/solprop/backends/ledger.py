# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march's bookkeeping, shared by both backends (SEP 13 section 5, SEP 39).

The call ledger and its record, the budget's error, a history entry and whether a
problem's source is structurally zero: what the native session and the reference's
host march both keep, with no arithmetic of either.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from jaxtyping import Complex

from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.translate.routes import tdse_descriptor

__all__ = ["CallLedger", "Derivative", "budget_error", "source_absent"]

_State = Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]


@dataclass
class CallLedger:
    """Expensive source evaluations of one run, by purpose (SEP 13 section 5).

    Attributes
    ----------
    startup : int
        Evaluations at an accepted field to start or restart the history.
    predictor : int
        Evaluations at a predicted field, accepted or not.
    corrected : int
        Evaluations at a corrected, accepted field; PEC makes none.
    rejected : int
        Evaluations spent on steps that were then rejected (included in the above).
    accepted_steps, rejected_steps : int
        Step counts.
    restarts : int
        History restarts at hard kinks.
    failure_restarts : int
        History restarts after :data:`~solprop.stepper.multistep.RESTART_AFTER`
        consecutive rejections.
    dense_observations : int
        Save planes observed by interpolation instead of a landing; none costs a call.
    orders : list of int
        The predictor order of every accepted step. Its record
        (:meth:`to_record`) run-length encodes it, as ``[order, count]`` pairs: a
        run's order settles, and the record is rewritten at every saved plane.
    """

    startup: int = 0
    predictor: int = 0
    corrected: int = 0
    rejected: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    restarts: int = 0
    failure_restarts: int = 0
    dense_observations: int = 0
    orders: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Start the record's encoding of the orders empty."""
        # The encoded runs of `orders[:_encoded]`, of the list `_source`: the record
        # is rewritten at every saved plane, and re-encoding every step's order each
        # time made a run saved step by step quadratic in its steps.
        self._source: list[int] | None = None
        self._runs: list[list[int]] = []
        self._encoded = 0

    def to_record(self) -> dict[str, object]:
        """Return the ledger as its run record holds it, the orders run-length encoded.

        The encoding is extended from the orders appended since the last record. A
        list of orders that was replaced, or shortened, is encoded afresh.

        Returns
        -------
        dict
            Every counter by name, and ``orders`` as ``[order, count]`` pairs, in
            step order, no two neighbours of the same order.

        Examples
        --------
        >>> from solprop.backends.ledger import CallLedger
        >>> CallLedger(accepted_steps=5, orders=[1, 2, 2, 3, 3]).to_record()["orders"]
        [[1, 1], [2, 2], [3, 2]]
        """
        orders, runs = self.orders, self._runs
        if (
            orders is not self._source
            or len(orders) < self._encoded
            or (self._encoded and orders[self._encoded - 1] != runs[-1][0])
        ):
            self._source, runs, self._encoded = orders, [], 0
            self._runs = runs
        for order in orders[self._encoded :]:
            if runs and runs[-1][0] == order:
                runs[-1][1] += 1
            else:
                runs.append([order, 1])
        self._encoded = len(orders)
        record: dict[str, object] = {
            name: getattr(self, name) for name in _LEDGER_FIELDS
        }
        record["orders"] = [list(run) for run in runs]
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CallLedger":
        """Restore validated cumulative counts without restoring numerical history.

        Parameters
        ----------
        record : mapping
            Persisted counter fields and accepted predictor orders, as
            :meth:`to_record` writes them.

        Returns
        -------
        CallLedger
            Independently owned counters, the orders decoded step by step.

        Raises
        ------
        BackendError
            If a counter is absent, unknown, negative or not an integer, an order run
            is not a pair of positive integers, or the runs do not count one order
            per accepted step.

        Examples
        --------
        >>> from solprop.backends.ledger import CallLedger
        >>> ledger = CallLedger(accepted_steps=3, orders=[1, 2, 2])
        >>> CallLedger.from_record(ledger.to_record()) == ledger
        True
        """
        if set(record) != set(_LEDGER_FIELDS):
            raise BackendError("logical call ledger fields do not match this schema")
        ledger = cls()
        for name, value in record.items():
            if name == "orders":
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BackendError(f"invalid logical call counter {name}")
            setattr(ledger, name, value)
        ledger.orders = _decoded_orders(record["orders"], ledger.accepted_steps)
        return ledger

    @property
    def total(self) -> int:  # numpydoc ignore=RT01
        """Every charged source evaluation.

        Startup, predictor and corrected evaluations together.
        """
        return self.startup + self.predictor + self.corrected


_LEDGER_FIELDS = tuple(one.name for one in fields(CallLedger))
"""The counters and the orders a ledger's record holds, in field order."""


def budget_error(
    max_calls: int | None,
    spent: int,
    z: float,
    cost: int,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> ResponseBudgetError:
    """Return the error of an Adams trial the source-call budget cannot pay for.

    One message for every march that keeps the budget: solprop's and the native
    session's, which charge the same calls (SEP 39 section 4.9).

    Parameters
    ----------
    max_calls : int or None
        The method's budget.
    spent : int
        The calls charged so far (:attr:`CallLedger.total`).
    z : float
        The last accepted position, m.
    cost : int
        The calls the refused trial needs.
    field : numpy.ndarray
        The accepted state there, ``(n_modes, n_pol, n_omega)``, owned by the error.

    Returns
    -------
    solprop.exceptions.ResponseBudgetError
        Carrying the last accepted position and state and the calls spent.
    """
    msg = (
        f"the source-call budget of {max_calls} is spent ({spent} "
        f"charged) at z = {z:.6g} m: the next trial needs {cost}; "
        "the run stops at its last accepted step, unfinished"
    )
    return ResponseBudgetError(msg, z=z, field=field, calls=spent)


def _decoded_orders(runs: object, steps: int) -> list[int]:
    """Return the order of every accepted step from its run-length record.

    - ``runs``: the record's ``orders``: ``[order, count]`` pairs of positive integers.
    """
    if (
        isinstance(runs, list)
        and runs
        and all(isinstance(one, int) and not isinstance(one, bool) for one in runs)
    ):
        # Development builds between two releases share the output schema, and
        # those before the encoding wrote one order per step (#80). Read as pairs,
        # such a list would be refused for its shape alone; it is named instead.
        raise BackendError(
            "the record's predictor orders are in the flat form an earlier "
            "development build wrote, one order per step, not the [order, count] "
            "pairs this build reads: rerun the case, or read it with that build"
        )
    if not isinstance(runs, list) or not all(
        isinstance(run, list)
        and len(run) == 2
        and all(
            isinstance(one, int) and not isinstance(one, bool) and one > 0
            for one in run
        )
        for run in runs
    ):
        raise BackendError(
            "logical call ledger orders must be [order, count] pairs of positive "
            "integers"
        )
    # Checked before the expansion, which the counts bound.
    counted = sum(count for _, count in runs)
    if counted != steps:
        raise BackendError(
            f"logical call ledger orders count {counted} steps, and the ledger "
            f"{steps} accepted steps"
        )
    return [order for order, count in runs for _ in range(count)]


@dataclass
class Derivative:
    """One history entry, in the current anchor's frame."""

    z: float
    value: _State | None
    """The store's row, or ``None`` for an engine that keeps its rows resident."""
    identity: str
    slot: int = -1
    """Its row of the history store."""


def source_absent(problem: Any) -> bool:
    """Return whether a problem's complete source is structurally zero.

    No response is compiled beyond the accumulation, and no in-stage absorber, no
    atomic response and no general or spatial pipeline adds a term: SEP 16 section 4.2
    then requires the exact chord rather than an interpolated compensation, which
    need not cancel even at the endpoint. Decided from the problem's structure,
    never from a sample that happens to vanish.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.

    Returns
    -------
    bool
        Whether every source evaluation is identically zero.
    """
    responses = problem.responses
    geometry = problem.transform.geometry
    in_stage = (
        responses.temporal_absorber_fine is not None
        and responses.absorber_scheme == "nonlinear"
    ) or (geometry is not None and bool(np.any(geometry.absorber_rate)))
    return (
        len(responses.response_order) == 1
        and not in_stage
        and tdse_descriptor(problem.route) is None
        and problem.transform.modal_tables is None
        and problem.linear.separable is None
    )
