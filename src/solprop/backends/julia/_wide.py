# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bind DF32 operand analysis to owned inputs without admitting execution."""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

import numpy as np

from solprop.backends.julia._descriptors import (
    CoefficientTransfer,
    TableTransfer,
    owned,
)
from solprop.backends.julia._lower import RunTransfer
from solprop.backends.julia._phase_bounds import PhaseOperands, phase_operands
from solprop.exceptions import BackendError
from solprop.output.schema import MarchSnapshot

__all__ = ["DF32OperatorBinding", "DF32StartBinding", "bind_df32_start"]


def _table_copy(table: TableTransfer | None) -> TableTransfer | None:
    """Copy a represented table without changing its coordinates or coefficients.

    Parameters
    ----------
    table : TableTransfer or None
        Original represented cumulative data.

    Returns
    -------
    TableTransfer or None
        Owned read-only arrays, or the absent table.
    """
    return (
        None
        if table is None
        else replace(
            table, edges=owned(table.edges), coefficients=owned(table.coefficients)
        )
    )


@dataclass(frozen=True, eq=False)
class DF32OperatorBinding:
    """Copy an immutable linear representation for sharing between actual starts.

    Attributes
    ----------
    data : CoefficientTransfer
        Owned constant generator, cumulative table or real-weight expansion.
        Construction always copies its arrays, even if the input is read-only;
        a borrowed read-only view does not establish immutable ownership.

    Notes
    -----
    Subsequent start bindings may share this record. Mutating its arrays, including
    re-enabling writes, invalidates the preparation. This record is not a phase
    domain or assembly-error certificate. No execution backend is initialized.
    """

    data: CoefficientTransfer

    def __post_init__(self) -> None:
        """Detach every coefficient array from the caller's allocation."""
        data = self.data
        object.__setattr__(
            self,
            "data",
            CoefficientTransfer(
                None if data.values is None else owned(data.values),
                _table_copy(data.table),
                _table_copy(data.weights),
            ),
        )


@dataclass(frozen=True, eq=False)
class DF32StartBinding:
    """Retain the exact inputs used by one start's DF32 operand analysis.

    Attributes
    ----------
    operator : DF32OperatorBinding
        Immutable owned linear representation, optionally shared between starts.
    run : RunTransfer
        Actual resolved method, masks, precision/scales and replay with detached
        arrays. Geometry uses the configured maximum Adams order.
    history : MarchSnapshot or None
        The full owned compatible raw history that must actually be restored.
        Positions and rates are preserved, without snapping or physical scaling.
    operands : PhaseOperands
        Conservative bounds from these exact retained inputs, including the actual
        start coordinate. These are not computed DF32 error bounds or admission.
    origin_wide : {'binary64', 'df32'} or None
        Declared previous wide type. None means unknown for legacy history and
        absent for a fresh start. It never establishes a zero prior rate error.
    """

    operator: DF32OperatorBinding
    run: RunTransfer
    history: MarchSnapshot | None
    operands: PhaseOperands
    origin_wide: Literal["binary64", "df32"] | None


def _run_copy(run: RunTransfer) -> RunTransfer:
    """Detach one resolved request while preserving every scalar method choice.

    Parameters
    ----------
    run : RunTransfer
        Resolved request with possibly borrowed array storage.

    Returns
    -------
    RunTransfer
        Per-start array ownership, including optional controller weights.
    """
    controller = run.control.controller
    control = replace(
        run.control,
        controller=replace(
            controller,
            weights=None if controller.weights is None else owned(controller.weights),
        ),
    )
    return replace(
        run,
        control=control,
        nodes=owned(run.nodes),
        matrix=owned(run.matrix),
        solution=owned(run.solution),
        error=owned(run.error),
        active=owned(run.active),
        observed=owned(run.observed),
        weights=owned(run.weights),
        planes=owned(run.planes),
        breakpoints=owned(run.breakpoints),
        hard_kinks=owned(run.hard_kinks),
        steps=owned(run.steps),
        edges=owned(run.edges),
        dense_weights=None if run.dense_weights is None else owned(run.dense_weights),
    )


def _history_copy(history: MarchSnapshot | None) -> MarchSnapshot | None:
    """Detach already-compatible raw history without changing its numerical values.

    Parameters
    ----------
    history : MarchSnapshot or None
        Actual history selected for restoration, or an explicit fresh start.

    Returns
    -------
    MarchSnapshot or None
        Read-only owned arrays and row mapping, or no history.

    Raises
    ------
    BackendError
        If the portable row dtypes disagree with binary32 state and binary64 rates.
    """
    if history is None:
        return None
    if history.state.dtype != np.complex128 or any(
        row.dtype != (np.complex128 if name == "rates" else np.complex64)
        for name, row in history.rows.items()
    ):
        raise BackendError("DF32 binding needs raw binary32 history and binary64 rates")
    return replace(
        history,
        state=owned(history.state),
        positions=owned(history.positions),
        slots=owned(history.slots),
        rows=MappingProxyType({name: owned(row) for name, row in history.rows.items()}),
    )


def bind_df32_start(
    operator: CoefficientTransfer | DF32OperatorBinding,
    run: RunTransfer,
    start: float,
    *,
    history: MarchSnapshot | None = None,
    origin_wide: Literal["binary64", "df32"] | None = None,
) -> DF32StartBinding:
    """Bind conservative DF32 operands to one actual start and compatible history.

    Parameters
    ----------
    operator : CoefficientTransfer or DF32OperatorBinding
        Constant, represented cumulative or expanded linear data. A raw descriptor
        is copied once; an existing immutable binding is shared.
    run : RunTransfer
        Actual resolved binary32 request. Its arrays are detached per start.
    start : float
        Actual accepted anchor, metres, including an unchanged resumed coordinate.
    history : MarchSnapshot or None, optional
        Already-compatible raw history that will be restored. None explicitly
        selects a fresh or field-only start; incompatibility is never hidden.
    origin_wide : {'binary64', 'df32'} or None, optional
        Previous wide arithmetic, or unknown for an older snapshot.

    Returns
    -------
    DF32StartBinding
        Owned numerical inputs and their operand envelopes. A completed replay may
        have zero future step. No returned field admits execution or certifies a
        propagator. Existing ``transfer_memory`` inventories all retained arrays.

    Raises
    ------
    BackendError
        For a different scientific precision, incompatible history, malformed
        geometry/representation or a nonfinite conservative bound.

    Notes
    -----
    Per-operation assembly/function certificates and pretrial transaction checks
    remain required. Separable spatial operators have no input representation in
    this interface. Public DF32 and device execution remain closed.
    """
    if run.arithmetic.name != "single":
        raise BackendError("DF32 wide binding requires binary32 scientific state")
    if origin_wide not in (None, "binary64", "df32"):
        raise BackendError("DF32 history origin must name binary64, df32 or unknown")
    if history is None and origin_wide is not None:
        raise BackendError("a fresh DF32 start cannot declare a previous wide type")
    bound = (
        operator
        if isinstance(operator, DF32OperatorBinding)
        else DF32OperatorBinding(operator)
    )
    owned_run, owned_history = _run_copy(run), _history_copy(history)
    operands = phase_operands(bound.data, owned_run, start, history=owned_history)
    return DF32StartBinding(bound, owned_run, owned_history, operands, origin_wide)
