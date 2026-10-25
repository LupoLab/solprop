# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The interaction-picture Adams march every CPU backend steps with (SEP 13, SEP 14).

One accepted step of a mature method costs one expensive source evaluation (explicit
Adams-Bashforth and PEC) or two (PECE), against seven for the reference backend's
IP-DOPRI5. The march keeps a short history of **transported** derivatives, re-anchored
at every accepted step, and decides each step on a local-defect estimate formed from
that history without another evaluation (SEP 13 sections 4.1--4.3).

The interaction picture is anchored at the start of each step, ``z_n``. With
``U(z_2, z_1) = exp(Phi(z_2) - Phi(z_1))``, the variable ``v(z) = U(z_n, z) a(z)``
obeys ``v' = g(z) = U(z_n, z) N(U(z, z_n) v, z)``. A history derivative is stored in
the current anchor's frame; accepting a step re-anchors every one of them by the step's
own propagator, which is the multiplication the Runge-Kutta march already performs on
its state, so no table is queried backwards. Every derivative keeps its position and
the identity of the field it was evaluated at -- the accepted field, or the predicted
one PEC stores in its place.

The march is shared: a backend supplies two hooks of the step driver,
``_adams_source`` (one source evaluation at a lab-frame field, returned in the step's
anchor frame) and ``_adams_propagator``, and the driver's own state, landing and
recording do the rest. ``numpy_ref`` evaluates through its reference passes and
``numba_cpu`` through its fused stage kernels (SEP 14 phase 2).

Lifecycle, owned here and not by the shared driver's Runge-Kutta march (SEP 13
section 6.1):

- a trial computes its predictor, its single source call and its estimate without
  touching the history; only an accepted step commits, so a rejection leaves the
  accepted history exactly as it was;
- PECE decides acceptance before its second call, so a rejection costs one call;
- explicit Adams-Bashforth rejects for free on a history estimate before the call,
  and checks the step again afterwards with the new derivative (section 4.3), at the
  cost of the call it has made;
- a hard kink is landed on and the history restarted on its far side; a save plane is
  a pure observation and keeps the history;
- a load or a resume starts afresh, at order one, and its startup calls are counted.
"""

import functools
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Protocol, cast

import numpy as np
from jaxtyping import Complex
from scipy.linalg import expm

from solprop._helpers import NUMBA, Helpers
from solprop._interrupts import committing
from solprop.backends.driver import StepDriver
from solprop.backends.exponential import phi_functions
from solprop.backends.numpy_ref import passes
from solprop.exceptions import BackendError, ConvergenceError, ResponseBudgetError
from solprop.output.schema import MarchSnapshot
from solprop.stepper.controller import limit_step
from solprop.stepper.multistep import (
    PEC_REAL_INTERVAL,
    Multistep,
    integrated_weights,
    lagrange_coefficients,
    milne_factor,
)
from solprop.stepper.schedule import FixedSchedule, LandingTarget, StepTargets, landing
from solprop.translate.linop import phi_rate
from solprop.translate.tdse import TDSEPurpose

__all__ = [
    "COLLAR_STABILITY",
    "ORDER_HYSTERESIS",
    "RESTART_AFTER",
    "AdamsEngine",
    "AdamsKernels",
    "AdamsMarch",
    "CallLedger",
    "CoupledBlock",
    "CoupledExponentialKernels",
    "ExponentialKernels",
    "HostHistory",
    "compensation_tables",
    "coupled_moments",
    "exponential_tables",
    "factored_defect_rows",
    "fused_tables",
    "source_absent",
]

RESTART_AFTER = 3
"""Consecutive rejections after which the history is restarted at order one."""

ORDER_HYSTERESIS = 1.2
"""Factor by which a neighbouring order's predicted step must win to be adopted."""

COLLAR_STABILITY = 1.0
"""Share of an order's negative-real stability interval the order choice allows."""


logger = logging.getLogger(__name__)

_State = Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
_StoreRows = Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"]


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
        PECE's evaluations at the corrected, accepted field.
    rejected : int
        Evaluations spent on steps that were then rejected (included in the above).
    accepted_steps, rejected_steps : int
        Step counts.
    restarts : int
        History restarts at hard kinks.
    failure_restarts : int
        History restarts after :data:`RESTART_AFTER` consecutive rejections.
    dense_observations : int
        Save planes observed by interpolation instead of a landing; none costs a call.
    orders : list of int
        The predictor order of every accepted step.
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

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CallLedger":  # noqa: UP037 - runtime hook
        """Restore validated cumulative counts without restoring numerical history.

        Parameters
        ----------
        record : mapping
            Persisted counter fields and accepted predictor orders.

        Returns
        -------
        CallLedger
            Independently owned counters.

        Raises
        ------
        BackendError
            If a counter is absent, unknown, negative or not an integer.
        """
        if set(record) != {one.name for one in fields(cls)}:
            raise BackendError("logical call ledger fields do not match this schema")
        ledger = cls()
        for name, value in record.items():
            if name == "orders":
                if not isinstance(value, list) or not all(
                    isinstance(one, int) and not isinstance(one, bool) and one > 0
                    for one in value
                ):
                    raise BackendError(
                        "logical call ledger orders must be positive integers"
                    )
                ledger.orders = list(value)
            elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BackendError(f"invalid logical call counter {name}")
            else:
                setattr(ledger, name, value)
        return ledger

    @property
    def total(self) -> int:
        """Return every charged source evaluation.

        Returns
        -------
        int
            Startup, predictor and corrected evaluations together.
        """
        return self.startup + self.predictor + self.corrected


@dataclass
class _Derivative:
    """One history entry, in the current anchor's frame."""

    z: float
    value: _State | None
    """The derivative on the general path; the store's row, or ``None`` for an engine
    that keeps its rows resident, on the fused and exponential paths."""
    identity: str
    slot: int = -1
    """Its row of the fused path's history store; -1 off that path."""


@dataclass
class _DenseStep:
    """The last accepted step, in its anchor's frame, for dense observation."""

    z: float
    dz: float
    start: _State
    end: _State
    start_slope: _State
    end_slope: _State
    nodes: np.ndarray
    slopes: list[_State]


def _hermite(step: _DenseStep, theta: float) -> _State:
    """Return the cubic Hermite interpolant of an accepted step at `theta`.

    The generic option (a2) of SEP 13 section 6.4: the two end states and the two end
    derivatives, all in the step's anchor frame, with local error ``O(h^4)``.

    Parameters
    ----------
    step : _DenseStep
        The accepted step.
    theta : float
        The observation as a fraction of the step, in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The interaction variable at `theta`, in the step's anchor frame.
    """
    theta2 = theta * theta
    theta3 = theta2 * theta
    h00 = 2 * theta3 - 3 * theta2 + 1
    h10 = theta3 - 2 * theta2 + theta
    h01 = -2 * theta3 + 3 * theta2
    h11 = theta3 - theta2
    return (
        h00 * step.start
        + h10 * step.dz * step.start_slope
        + h01 * step.end
        + h11 * step.dz * step.end_slope
    )


@dataclass
class _Trial:
    """One trial step: its stencil, its propagator and its predictor."""

    z: float
    dz: float
    order: int
    nodes: np.ndarray
    values: list[_State]
    state: _State
    propagator: _State
    predictor: _State
    scale: Callable[[_State], float]

    @classmethod
    def start(
        cls,
        history: list[_Derivative],
        order: int,
        z: float,
        dz: float,
        state: _State,
        backend: StepDriver,
    ) -> "_Trial":  # noqa: UP037 - a runtime hook evaluates it
        """Form the predictor of a step from the history, newest derivative first.

        Parameters
        ----------
        history : list
            The accepted derivatives, oldest first, in the anchor's frame.
        order : int
            The order asked for; fewer derivatives during startup lower it.
        z, dz : float
            The anchor and the step, m.
        state : numpy.ndarray
            The accepted state at the anchor.
        backend : solprop.backends.driver.StepDriver
            The backend, for the step's propagator and norm.

        Returns
        -------
        _Trial
            The trial, with its predictor formed.
        """
        recent = history[::-1]
        nodes = np.array([(entry.z - z) / dz for entry in recent])
        values = [cast("_State", entry.value) for entry in recent]
        used = min(order, len(history))
        return cls(
            z=z,
            dz=dz,
            order=used,
            nodes=nodes,
            values=values,
            state=state,
            propagator=backend._adams_propagator(z, z + dz),
            predictor=_combine(
                integrated_weights(nodes[:used], check=False, helpers=backend.helpers),
                values[:used],
                dz,
            ),
            scale=_norm_function(backend, state),
        )

    def lab(self, increment: _State) -> _State:
        """Return the lab-frame field at the end of the step for an increment.

        Parameters
        ----------
        increment : numpy.ndarray
            ``v - a_n`` in the anchor's frame.

        Returns
        -------
        numpy.ndarray
            ``U(z + dz, z) (a_n + increment)``.
        """
        return self.propagator * (self.state + increment)

    def norm(self, defect: _State) -> float:
        """Return the weak norm of a defect, carried to the end of the step.

        Parameters
        ----------
        defect : numpy.ndarray
            A local-defect estimate in the anchor's frame.

        Returns
        -------
        float
            The scaled norm; one is the tolerance.
        """
        return self.scale(self.propagator * defect)


class AdamsKernels:
    """The field arithmetic of a fused Adams step, in NumPy (SEP 14 phase 2).

    Three passes serve a step of the one-order-higher PEC family: the predictor's
    lab-frame field, the corrected field with the defects of the current and the
    neighbouring orders, and the history's transport to the new anchor. The history is
    a store of rows, newest first by ``slots``. ``numba_cpu`` compiles each pass into
    one traversal (:mod:`solprop.backends.numba_cpu.multistep`); this is the reference
    the compiled form is checked against.

    Parameters
    ----------
    helpers : solprop._helpers.Helpers, optional
        The owning backend's implementation of the Adams basis (SEP 29 section 6.6);
        the shared Numba one by default.
    """

    def __init__(self, *, helpers: Helpers = NUMBA) -> None:
        """Hold the backend's basis implementation."""
        self._helpers = helpers

    def weights(
        self, nodes: list[float] | np.ndarray, upper: float = 1.0
    ) -> np.ndarray:
        """Return the integrated weights of an interpolant on distinct nodes.

        Parameters
        ----------
        nodes : list of float or numpy.ndarray
            The normalised nodes, accepted positions and so distinct.
        upper : float, optional
            The integral's upper limit.

        Returns
        -------
        numpy.ndarray
            One weight per node, as ``integrated_weights`` gives them.
        """
        return integrated_weights(nodes, upper, check=False, helpers=self._helpers)

    def predict(
        self,
        state: _State,
        propagator: _State,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        weights: np.ndarray,
        dz: float,
    ) -> _State:
        """Return the predictor's lab-frame field ``U (a_n + dz sum_t w_t g_t)``.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        propagator : numpy.ndarray
            ``U(z_n + dz, z_n)``.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first, ``int64``.
        weights : numpy.ndarray
            One integrated weight per row.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted field at the end of the step.
        """
        total = np.zeros_like(state)
        for weight, slot in zip(weights, slots, strict=True):
            total = passes.accumulate(total, store[slot], float(weight))
        return propagator * (state + dz * total)

    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        propagator: _State,
        source: _State,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        corrector: np.ndarray,
        rows: np.ndarray,
        dz: float,
        mask: np.ndarray,
        divisor: np.ndarray,
    ) -> tuple[_State, np.ndarray]:
        """Return the corrected lab-frame field and the scaled norms of the defects.

        With ``X = (source, store[slots[0]], ...)``, the field is
        ``U (a_n + dz sum_t c_t X_t)`` and each norm is the weak norm of
        ``U dz sum_t rows[r, t] X_t`` over the in-band bins.

        Parameters
        ----------
        state, propagator : numpy.ndarray
            As for :meth:`predict`.
        source : numpy.ndarray
            The source at the predicted field, in the anchor's frame.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first, ``int64``.
        corrector : numpy.ndarray
            The corrector's weights over ``X``, zero past its own nodes.
        rows : numpy.ndarray
            One defect's weights over ``X`` per row, shape ``(n_rows, 1 + n_slots)``.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins, Boolean.
        divisor : numpy.ndarray
            The norm's scale per bin.

        Returns
        -------
        tuple
            The corrected field and one norm per row.
        """
        values = [source, *(store[slot] for slot in slots)]
        total = np.zeros_like(state)
        for weight, value in zip(corrector, values, strict=True):
            total = passes.accumulate(total, value, float(weight))
        norms = np.empty(rows.shape[0])
        for index, row in enumerate(rows):
            defect = np.zeros_like(state)
            for weight, value in zip(row, values, strict=True):
                defect = passes.accumulate(defect, value, float(weight))
            scaled = (propagator * (dz * defect))[..., mask] / divisor[mask]
            norms[index] = float(np.linalg.norm(scaled))
        return propagator * (state + dz * total), norms

    def transport(
        self,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        propagator: _State,
        new_slot: int,
        source: _State,
    ) -> None:
        """Re-anchor the history at the step's end and store the new derivative.

        Parameters
        ----------
        store : numpy.ndarray
            The history rows, updated in place.
        slots : numpy.ndarray
            The rows of the kept history, ``int64``.
        propagator : numpy.ndarray
            ``U(z_n + dz, z_n)``.
        new_slot : int
            The free row the new derivative goes to.
        source : numpy.ndarray
            The new derivative in the step's anchor frame.
        """
        for slot in slots:
            store[slot] *= propagator
        np.multiply(propagator, source, out=store[new_slot])


@dataclass(frozen=True)
class CoupledBlock:
    """A step's frozen coupled block on the bins that carry one (SEP 16 phase 5).

    Beside a coupled flow the cumulative diagonal ``Phi`` is a multiple of the identity
    in the channel axis (``docs/dev/kernel-contract.md``, ``coupled_flow``), so the
    step's reference is ``B = (Delta Phi / h) I + G_s`` with ``G_s = Z T Z^H`` the
    frozen block of the section the step is anchored in, and the two parts commute.

    Attributes
    ----------
    active : numpy.ndarray
        Which bins carry a block, bool, shape ``(n_omega,)``.
    vectors, triangular : numpy.ndarray
        ``Z`` and ``T`` of the section's block, 1/m for ``T``, shape
        ``(n_active, n_modes, n_modes)``.
    section : int
        The section the step is anchored in.
    end_weights : numpy.ndarray or None
        The flow's term weights at the step's end, read on the step's side of a
        section edge (``StageScalars.flow_weights``); ``None`` without a residual.
    residual : callable or None
        ``(state, weights) -> R_s(z) state`` for a frozen-residual flow: the
        generator at the position the weights belong to less this section's frozen
        block, on the active bins the error norm scores and zero elsewhere, as the
        Runge-Kutta stage applies it; ``None`` for a constant-Schur flow.
    basis : tuple or None
        The section's eigenbasis ``(W, Y, lambda, route)``, as
        :func:`solprop.translate.coupled.schur_eigenbasis` returns it: the bins on the
        route take their matrix functions as ``W f(x + h lambda) Y``, the compiled
        backends' evaluation. ``None``, the reference's, takes every bin from the
        augmented exponential.
    """

    active: np.ndarray
    vectors: np.ndarray
    triangular: np.ndarray
    section: int
    end_weights: np.ndarray | None = None
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class _MatrixMoments:
    """A coupled trial's flow and moments: scalar per element, matrices on a block.

    Off the block's bins the step's flow and moments are the diagonal ones, per
    element; on them each is a matrix in the channel axis, applied to the channel
    vector of every batch row and polarisation. ``matrices`` and ``flow_matrix`` are
    laid out by bin, ``(..., n_bins, n_modes, n_modes)``; the products read the same
    matrices with the bin axis last, as the state holds it.
    """

    scalar: np.ndarray
    flow_scalar: np.ndarray
    matrices: np.ndarray
    flow_matrix: np.ndarray
    active: np.ndarray
    _columns: np.ndarray = field(init=False, repr=False, compare=False)
    _flow_columns: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Hold the matrices with the bin axis last, for the products."""
        object.__setattr__(
            self, "_columns", np.ascontiguousarray(np.moveaxis(self.matrices, -3, -1))
        )
        object.__setattr__(
            self,
            "_flow_columns",
            np.ascontiguousarray(np.moveaxis(self.flow_matrix, -3, -1)),
        )

    def _product(
        self, columns: np.ndarray, scalar: np.ndarray, values: _State
    ) -> _State:
        """Return the scalar product off the block and the matrix product on it.

        Parameters
        ----------
        columns : numpy.ndarray
            The block's matrices, ``(n_modes, n_modes, n_bins)``.
        scalar : numpy.ndarray
            The diagonal factor, in the state's layout.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            The product.
        """
        out = scalar * values
        selected = values[..., self.active]
        block = columns[:, 0][np.newaxis, :, np.newaxis] * selected[:, np.newaxis, 0]
        for column in range(1, columns.shape[1]):
            block += (
                columns[:, column][np.newaxis, :, np.newaxis]
                * selected[:, np.newaxis, column]
            )
        out[..., self.active] = block
        return out

    def apply(self, k: int, values: _State) -> _State:
        """Return ``M_k`` applied to a state-shaped array.

        Parameters
        ----------
        k : int
            The moment.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            ``M_k values``.
        """
        columns = self._columns
        return self._product(columns[k], self.scalar[k], values)

    def flow(self, values: _State) -> _State:
        """Return the step's flow ``exp(h B)`` applied to a state-shaped array.

        Parameters
        ----------
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            ``exp(h B) values``.
        """
        columns = self._flow_columns
        return self._product(columns, self.flow_scalar, values)


def _through(
    pair: tuple[np.ndarray, np.ndarray], values: np.ndarray
) -> Complex[np.ndarray, "n_bins n_modes_out n_modes_in"]:
    """Return ``W diag(values) Y`` per bin, for an eigenbasis pair ``(W, Y)``.

    Parameters
    ----------
    pair : tuple of numpy.ndarray
        ``W`` and ``Y``, shape ``(n_bins, n_modes, n_modes)``.
    values : numpy.ndarray
        The function's values at the eigenvalues, shape ``(n_bins, n_modes)``.

    Returns
    -------
    numpy.ndarray
        The matrices, shape ``(n_bins, n_modes, n_modes)``.
    """
    left, right = pair
    return (left * values[:, np.newaxis, :]) @ right


def _eigen_functions(
    shift: np.ndarray,
    dz: float,
    count: int,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    matrices: np.ndarray,
    flow_matrix: np.ndarray,
) -> None:
    """Write the flow and moments of the bins on an eigenbasis route, in NumPy.

    Parameters
    ----------
    shift : numpy.ndarray
        ``Delta Phi`` on each of the block's bins.
    dz : float
        The step, m.
    count : int
        The moments formed.
    basis : tuple of numpy.ndarray
        ``(W, Y, lambda, route)`` of :attr:`CoupledBlock.basis`.
    matrices, flow_matrix : numpy.ndarray
        Written on the route's bins.
    """
    left, right, eigenvalues, route = basis
    if not np.any(route):
        return
    arguments = shift[route, np.newaxis] + dz * eigenvalues[route]
    pair = (left[route], right[route])
    flow_matrix[route] = _through(pair, np.exp(arguments))
    if count:
        phi = phi_functions(arguments, count)
        for k in range(count):
            matrices[k, route] = math.factorial(k) * _through(pair, phi[k + 1])


def coupled_moments(
    exponent: _State,
    count: int,
    block: CoupledBlock,
    dz: float,
    *,
    eigen: Callable[..., None] = _eigen_functions,
    diagonal: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> _MatrixMoments:
    """Return a coupled trial's flow and moments ``k! phi_{k+1}(h B)``, ``k < count``.

    The reference's evaluation (SEP 16 section 4.5): on each bin of the block the
    functions of ``X = Delta Phi I + h T``, upper triangular, are read off the first
    block row of the exponential of the augmented matrix
    ``[[X, I, 0, ...], [0, 0, I, ...], ..., [0, ..., 0]]`` -- ``exp(X)``,
    ``phi_1(X)``, ..., ``phi_count(X)`` -- and carried back by ``Z``; off the block the
    diagonal functions of :class:`ExponentialKernels`. With the block's eigenbasis
    (``CoupledBlock.basis``), the bins on its route instead take ``W f(Delta Phi + h
    lambda) Y``, every bin in one pass: the route and its gate are the compiled flow
    actions' (:func:`solprop.translate.coupled.eigenbasis_limit`), and the bins off it
    keep the augmented exponential.

    Parameters
    ----------
    exponent : numpy.ndarray
        ``Delta Phi`` over the step, the same in every channel.
    count : int
        The moments formed.
    block : CoupledBlock
        The step's frozen block.
    dz : float
        The step, m.
    eigen : callable, optional
        Writes the eigenbasis route's bins, ``(shift, dz, count, basis, matrices,
        flow_matrix)``: ``_eigen_functions``, or a compiled backend's kernel.
    diagonal : callable or None, optional
        ``(exponent, count) -> moments`` off the block; ``None`` for the reference's
        :meth:`ExponentialKernels.moments`.

    Returns
    -------
    _MatrixMoments
        The flow and the moments.
    """
    active = block.active
    # The block's bins take matrix moments below: the scalar ones are formed off it.
    scalar = np.zeros((count, *exponent.shape), dtype=np.complex128)
    if count and not np.all(active):
        moments = ExponentialKernels().moments if diagonal is None else diagonal
        scalar[..., ~active] = moments(
            np.ascontiguousarray(exponent[..., ~active]), count
        )
    flow_scalar = np.exp(exponent)
    n_modes = block.triangular.shape[-1]
    shift = exponent[0, 0, 0, active]
    matrices = np.empty((count, *block.triangular.shape), dtype=np.complex128)
    flow_matrix = np.empty(block.triangular.shape, dtype=np.complex128)
    size = n_modes * (count + 1)
    identity = np.eye(n_modes)
    route = np.zeros(block.triangular.shape[0], dtype=bool)
    if block.basis is not None:
        route = block.basis[3]
        eigen(shift, dz, count, block.basis, matrices, flow_matrix)
    for index in np.flatnonzero(~route):
        augmented = np.zeros((size, size), dtype=np.complex128)
        augmented[:n_modes, :n_modes] = (
            shift[index] * identity + dz * block.triangular[index]
        )
        for row in range(count):
            start = (row + 1) * n_modes
            augmented[row * n_modes : start, start : start + n_modes] = identity
        functions = expm(augmented)[:n_modes]
        vectors = block.vectors[index]
        adjoint = np.conj(vectors.T)
        flow_matrix[index] = vectors @ functions[:, :n_modes] @ adjoint
        for k in range(count):
            column = (k + 1) * n_modes
            matrices[k, index] = (
                math.factorial(k)
                * vectors
                @ functions[:, column : column + n_modes]
                @ adjoint
            )
    return _MatrixMoments(scalar, flow_scalar, matrices, flow_matrix, active)


class ExponentialKernels:
    """The field arithmetic of an exponential Adams step, in NumPy (SEP 16 phase 2).

    The history is a store of **untransported** source rows, newest first by
    ``slots``. A trial forms ``M_k = k! phi_{k+1}(x)``, ``x = h D`` per element, once,
    and applies every formula it needs in the monomial basis of its interpolant,
    ``h sum_k M_k (sum_j c_jk F_j)`` (``solprop.backends.exponential``), so no
    per-node weight volume is formed. Accepting a step re-anchors nothing. The
    compiled backends' kernels are checked against these passes.

    Parameters
    ----------
    helpers : solprop._helpers.Helpers, optional
        The owning backend's implementation of the Adams basis (SEP 29 section 6.6);
        the shared Numba one by default.
    """

    def __init__(self, *, helpers: Helpers = NUMBA) -> None:
        """Hold the backend's basis implementation."""
        self._helpers = helpers

    def coupled(self) -> "CoupledExponentialKernels":  # noqa: UP037 - defined below
        """Return the passes this backend takes beside a coupled flow.

        Returns
        -------
        CoupledExponentialKernels
            The reference's; a compiled backend returns its own.
        """
        return CoupledExponentialKernels(helpers=self._helpers)

    def _times_moment(self, moments: Any, k: int, values: _State) -> _State:
        """Return ``M_k values``: elementwise here, through a block in a subclass.

        Parameters
        ----------
        moments : numpy.ndarray
            The trial's moments.
        k : int
            The moment.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            The product.
        """
        return moments[k] * values

    def _times_flow(self, flow: Any, values: _State) -> _State:
        """Return the step's flow applied to an array, elementwise here.

        Parameters
        ----------
        flow : numpy.ndarray
            The trial's flow.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            The product.
        """
        return flow * values

    def moments(
        self, exponent: _State, count: int
    ) -> Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"]:
        """Return ``k! phi_{k+1}(x)`` for ``k < count`` along a new leading axis.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``, in the canonical layout.
        count : int
            The most interpolation nodes any formula of the trial has.

        Returns
        -------
        numpy.ndarray
            Shape ``(count, *x.shape)``.
        """
        phi = phi_functions(exponent, count)
        factors = np.array([math.factorial(k) for k in range(count)], dtype=np.float64)
        return phi[1:] * factors.reshape((count, *([1] * exponent.ndim)))

    def carry(
        self,
        exponent: _State,
        dz: float,
        rates: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        states: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        spans: np.ndarray,
    ) -> Complex[np.ndarray, "n_used n_batch n_modes n_pol n_omega"]:
        """Return each history sample's compensation carried to the step's end.

        ``exp(s_j x) (D(z_j) - x / dz) u_j`` with ``s_j = 1 - theta_j``, the
        difference of rates formed before the state multiplies it (SEP 16 4.2).

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``.
        dz : float
            The step, m.
        rates, states : numpy.ndarray
            The history's rates and states, by slot.
        slots : numpy.ndarray
            The samples carried, newest first, ``int64``.
        spans : numpy.ndarray
            ``1 - theta_j`` per sample: the flow's length in steps.

        Returns
        -------
        numpy.ndarray
            One carried row per sample.
        """
        reference = exponent / dz
        return np.stack(
            [
                np.exp(span * exponent) * ((rates[slot] - reference) * states[slot])
                for slot, span in zip(slots, spans, strict=True)
            ]
        )

    def flow(self, exponent: _State) -> _State:
        """Return the step's exact linear flow ``exp(x)``.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``.

        Returns
        -------
        numpy.ndarray
            ``exp(x)``, the propagator the interaction formulation would apply.
        """
        return np.exp(exponent)

    def apply(
        self,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"],
        coefficients: np.ndarray,
        values: list[_State],
        dz: float,
    ) -> _State:
        """Return ``dz sum_k M_k (sum_j c_jk X_j)`` over the given rows.

        Parameters
        ----------
        moments : numpy.ndarray
            ``M_k`` from :meth:`moments`.
        coefficients : numpy.ndarray
            ``c_jk``, shape ``(len(values), n_k)`` with ``n_k <= len(moments)``.
        values : list of numpy.ndarray
            The samples ``X_j``.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The increment.
        """
        return self._applied(moments, coefficients, values, dz)

    def _applied(
        self,
        moments: Any,
        coefficients: np.ndarray,
        values: list[_State],
        dz: float,
    ) -> _State:
        """Return :meth:`apply`'s increment; shared with the coupled passes.

        Parameters
        ----------
        moments : numpy.ndarray or _MatrixMoments
            The trial's moments.
        coefficients : numpy.ndarray
            ``c_jk``.
        values : list of numpy.ndarray
            The samples.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The increment.
        """
        total = np.zeros_like(values[0])
        for k in range(coefficients.shape[1]):
            combined = np.zeros_like(values[0])
            for weight, value in zip(coefficients[:, k], values, strict=True):
                if weight != 0.0:
                    combined = passes.accumulate(combined, value, float(weight))
            total += self._times_moment(moments, k, combined)
        return dz * total

    def predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"],
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        coefficients: np.ndarray,
        dz: float,
    ) -> _State:
        """Return the predicted field ``exp(x) u_n + dz sum_k M_k sum_j c_jk F_j``.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow : numpy.ndarray
            ``exp(x)``, the step's exact linear flow.
        moments : numpy.ndarray
            ``M_k`` from :meth:`moments`.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first, ``int64``.
        coefficients : numpy.ndarray
            The predictor's Lagrange coefficients, shape ``(len(slots), len(slots))``.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted lab-frame field at the end of the step.
        """
        return self._predicted(state, flow, moments, store, slots, coefficients, dz)

    def _predicted(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: Any,
        moments: Any,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        coefficients: np.ndarray,
        dz: float,
    ) -> _State:
        """Return :meth:`predict`'s field; shared with the coupled passes.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state.
        flow, moments : numpy.ndarray or _MatrixMoments
            The trial's flow and moments.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows combined.
        coefficients : numpy.ndarray
            The predictor's coefficients.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted field.
        """
        values = [store[slot] for slot in slots]
        return self._times_flow(flow, state) + self._applied(
            moments, coefficients, values, dz
        )

    def defect_rows(
        self, nodes: list[float], orders: list[int], width: int
    ) -> np.ndarray:
        """Return the defects' coefficients in the form :meth:`correct` takes.

        Parameters
        ----------
        nodes : list of float
            The history's normalised positions, newest first.
        orders : list of int
            The orders scored, the step's own first.
        width : int
            ``1 + max(orders)``.

        Returns
        -------
        numpy.ndarray
            The monomial rows of ``_exponential_rows``, shape
            ``(len(orders), width, width)``.
        """
        return _exponential_rows(nodes, orders, width, helpers=self._helpers)[0]

    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"],
        source: _State,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        corrector: np.ndarray,
        rows: np.ndarray,
        dz: float,
        mask: np.ndarray,
        divisor: np.ndarray,
        extra: tuple[_State, list[_State]] | None = None,
    ) -> tuple[_State, np.ndarray]:
        """Return the corrected field and the scaled norms of the defects.

        With ``X = (source, store[slots[0]], ...)``, the field is
        ``exp(x) u_n + dz sum_k M_k sum_j corrector[j, k] X_j`` and each norm is the
        weak norm of ``dz sum_k M_k sum_j rows[r, j, k] X_j`` over the in-band bins;
        both are already at the end of the step. A varying operator's compensation
        adds its own corrector term to the field and one term to each defect.

        Parameters
        ----------
        state, flow, moments : numpy.ndarray
            As for :meth:`predict`.
        source : numpy.ndarray
            The untransported source at the predicted field.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first, ``int64``.
        corrector : numpy.ndarray
            The corrector's coefficients over ``X``, shape ``(width, n_k)``, zero past
            its own nodes.
        rows : numpy.ndarray
            One defect's coefficients over ``X`` per row, shape ``(n_rows, width,
            n_k)``.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins, Boolean.
        divisor : numpy.ndarray
            The norm's scale per bin.
        extra : tuple or None, optional
            The compensation's corrector term and its term in each defect, or
            ``None`` for a constant operator.

        Returns
        -------
        tuple
            The corrected field and one norm per row.
        """
        return self._corrected(
            state,
            flow,
            moments,
            source,
            (store, slots),
            (corrector, rows),
            dz,
            (mask, divisor),
            extra,
        )

    def _corrected(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: Any,
        moments: Any,
        source: _State,
        history: tuple[np.ndarray, np.ndarray],
        tables: tuple[np.ndarray, np.ndarray],
        dz: float,
        scoring: tuple[np.ndarray, np.ndarray],
        extra: tuple[_State, list[_State]] | None,
    ) -> tuple[_State, np.ndarray]:
        """Return :meth:`correct`'s field and norms; shared with the coupled passes.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state.
        flow, moments : numpy.ndarray or _MatrixMoments
            The trial's flow and moments.
        source : numpy.ndarray
            The new sample.
        history : tuple of numpy.ndarray
            The store and the rows after the source.
        tables : tuple of numpy.ndarray
            The corrector's coefficients and the defect rows.
        dz : float
            The step, m.
        scoring : tuple of numpy.ndarray
            The in-band mask and the norm's scale per bin.
        extra : tuple or None
            A varying operator's compensation terms.

        Returns
        -------
        tuple
            The corrected field and one norm per row.
        """
        store, slots = history
        corrector, rows = tables
        mask, divisor = scoring
        values = [source, *(store[slot] for slot in slots)]
        advanced = self._times_flow(flow, state) + self._applied(
            moments, corrector, values, dz
        )
        if extra is not None:
            advanced = advanced + extra[0]
        norms = np.empty(rows.shape[0])
        for index, row in enumerate(rows):
            defect = self._applied(moments, row, values, dz)
            if extra is not None:
                defect = defect + extra[1][index]
            norms[index] = float(np.linalg.norm(defect[..., mask] / divisor[mask]))
        return advanced, norms


class CoupledExponentialKernels(ExponentialKernels):
    """The reference passes beside a coupled flow (SEP 16 phase 5).

    A coupled trial's flow and moments are ``_MatrixMoments``
    (:func:`coupled_moments`): matrices in the channel axis on the bins that carry a
    block, elementwise elsewhere. The passes are :class:`ExponentialKernels`' with
    those products.
    """

    def coupled(self) -> "CoupledExponentialKernels":  # noqa: UP037 - this class
        """Return these passes.

        Returns
        -------
        CoupledExponentialKernels
            This object.
        """
        return self

    def coupled_moments(
        self, exponent: _State, count: int, block: CoupledBlock, dz: float
    ) -> _MatrixMoments:
        """Return a coupled trial's flow and moments (:func:`coupled_moments`).

        Parameters
        ----------
        exponent : numpy.ndarray
            ``Delta Phi`` over the step.
        count : int
            The moments formed.
        block : CoupledBlock
            The step's frozen block.
        dz : float
            The step, m.

        Returns
        -------
        _MatrixMoments
            The flow and the moments.
        """
        return coupled_moments(exponent, count, block, dz)

    def _times_moment(self, moments: Any, k: int, values: _State) -> _State:
        """Return ``M_k values`` through the trial's block.

        Parameters
        ----------
        moments : _MatrixMoments
            The coupled trial's moments.
        k : int
            The moment.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            The product.
        """
        return cast("_MatrixMoments", moments).apply(k, values)

    def _times_flow(self, flow: Any, values: _State) -> _State:
        """Return the coupled trial's flow applied to an array.

        Parameters
        ----------
        flow : _MatrixMoments
            The coupled trial's flow and moments.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            The product.
        """
        return cast("_MatrixMoments", flow).flow(values)

    def carry_coupled(
        self,
        exponent: _State,
        dz: float,
        rows: tuple[_StoreRows, _StoreRows, list[np.ndarray | None]],
        slots: np.ndarray,
        spans: np.ndarray,
        block: CoupledBlock,
    ) -> Complex[np.ndarray, "n_used n_batch n_modes n_pol n_omega"]:
        """Return each sample's compensation carried to the step's end (SEP 16 4.5).

        ``C_j = (D(z_j) - x / h) u_j + R_s(z_j) u_j``: the diagonal chord's
        difference of rates and the frozen-residual's residual relative to the
        **step's** section, formed from the sample's own flow weights -- so that a
        step in a new section rebuilds the history's residual against its block --
        and carried by the reference's forward flow ``exp((1 - theta_j) h B)``.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Delta Phi`` over the step.
        dz : float
            The step, m.
        rows : tuple
            The history's rates and states, by slot, and each slot's flow weights.
        slots : numpy.ndarray
            The samples carried, newest first.
        spans : numpy.ndarray
            ``1 - theta_j`` per sample.
        block : CoupledBlock
            The step's frozen block.

        Returns
        -------
        numpy.ndarray
            One carried row per sample.
        """
        rates, states, weights = rows
        reference = exponent / dz
        carried = []
        for slot, span in zip(slots, spans, strict=True):
            term = (rates[slot] - reference) * states[slot]
            sample_weights = weights[slot]
            if block.residual is not None and sample_weights is not None:
                term = term + block.residual(states[slot], sample_weights)
            if span != 0.0:
                flow = self.coupled_moments(span * exponent, 0, block, span * dz)
                term = flow.flow(term)
            carried.append(term)
        return np.stack(carried)

    def predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State | _MatrixMoments,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"]
        | _MatrixMoments,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        coefficients: np.ndarray,
        dz: float,
    ) -> _State:
        """Return the predicted field through the trial's block.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow, moments : _MatrixMoments or numpy.ndarray
            The coupled trial's flow and moments.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first, ``int64``.
        coefficients : numpy.ndarray
            The predictor's Lagrange coefficients.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted lab-frame field at the end of the step.
        """
        return self._predicted(state, flow, moments, store, slots, coefficients, dz)

    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State | _MatrixMoments,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"]
        | _MatrixMoments,
        source: _State,
        store: Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"],
        slots: np.ndarray,
        corrector: np.ndarray,
        rows: np.ndarray,
        dz: float,
        mask: np.ndarray,
        divisor: np.ndarray,
        extra: tuple[_State, list[_State]] | None = None,
    ) -> tuple[_State, np.ndarray]:
        """Return the corrected field and the defects' norms through the block.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow, moments : _MatrixMoments or numpy.ndarray
            The coupled trial's flow and moments.
        source : numpy.ndarray
            The untransported source at the predicted field.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first, ``int64``.
        corrector : numpy.ndarray
            The corrector's coefficients.
        rows : numpy.ndarray
            One defect's coefficients per row.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins, Boolean.
        divisor : numpy.ndarray
            The norm's scale per bin.
        extra : tuple or None, optional
            A varying operator's compensation terms.

        Returns
        -------
        tuple
            The corrected field and one norm per row.
        """
        return self._corrected(
            state,
            flow,
            moments,
            source,
            (store, slots),
            (corrector, rows),
            dz,
            (mask, divisor),
            extra,
        )


@dataclass
class _ExponentialStep:
    """The last accepted exponential step, for dense observation."""

    z: float
    dz: float
    start: _State
    exponent: _State
    coefficients: np.ndarray
    values: list[_State]
    compensation: tuple[list[float], list[_State]] | None = None


@dataclass
class _ExponentialTrial:
    """An exponential trial held between its norms and its acceptance."""

    z: float
    dz: float
    state: _State
    exponent: _State
    corrector: np.ndarray
    source: _State
    lab: _State
    advanced: _State
    slots: np.ndarray
    order: int
    compensation: "_Compensation | None"  # noqa: UP037 - defined below


@dataclass
class _FusedTrial:
    """An interaction-formulation trial held between its norms and its acceptance."""

    z: float
    dz: float
    state: _State
    propagator: _State
    source: _State
    advanced: _State
    nodes: list[float]
    slots: np.ndarray
    order: int


@dataclass
class _Compensation:
    """One trial's linear compensation ``C_j = (D(z_j) - B_n) u_j`` (SEP 16 4.2).

    The transported treatment integrates it in the reference flow's frame,
    ``h sum_j a_j exp((1 - theta_j) h B_n) C_j`` with the ordinary Adams weights
    ``a_j``; every factor is a forward flow over at least the step. The carried rows
    are formed once per trial (:meth:`ExponentialKernels.carry`) and every formula is
    one weighted sum over them. The untransported treatment, a test witness only
    (Q8), interpolates ``C_j`` with the source's exponential weights instead.
    """

    exponent: _State
    moments: Any
    """The trial's moments, read by the untransported witness only."""
    nodes: list[float]
    carried: Complex[np.ndarray, "n_used n_batch n_modes n_pol n_omega"]
    dz: float
    treatment: str
    helpers: Helpers
    """The owning backend's implementation of the Adams basis."""
    endpoint_rate: _State | None = None
    endpoint: _State | None = None
    _weights: dict[tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def add_endpoint(
        self, rate: _State, state: _State, residual: _State | None = None
    ) -> None:
        """Add the new sample's compensation at the predicted state; no transport.

        Parameters
        ----------
        rate : numpy.ndarray
            ``D(z_n + h)`` from the step's side of the end.
        state : numpy.ndarray
            The predicted state the source was evaluated at.
        residual : numpy.ndarray or None, optional
            A frozen-residual flow's ``R_s(z_n + h)`` applied to that state.
        """
        self.endpoint_rate = rate
        self.endpoint = (rate - self.exponent / self.dz) * state
        if residual is not None:
            self.endpoint = self.endpoint + residual

    def _sum(self, nodes: list[float], first: _State | None, count: int) -> _State:
        """Return ``h`` times the treatment's integral over the given samples.

        Parameters
        ----------
        nodes : list of float
            The formula's nodes, the endpoint's first when `first` is given.
        first : numpy.ndarray or None
            The endpoint sample's compensation, or ``None`` for a predictor.
        count : int
            How many carried history rows the formula reads.

        Returns
        -------
        numpy.ndarray
            The increment.
        """
        rows = self.carried[:count]
        if self.treatment == "transported":
            key = tuple(nodes)
            weights = self._weights.get(key)
            if weights is None:
                weights = integrated_weights(
                    np.asarray(nodes), check=False, helpers=self.helpers
                )
                self._weights[key] = weights
            total = np.tensordot(weights[len(nodes) - count :], rows, axes=1)
            if first is not None:
                total = total + weights[0] * first
            return self.dz * total
        values = ([] if first is None else [first]) + list(rows)
        return ExponentialKernels(helpers=self.helpers).apply(
            self.moments[: len(nodes)],
            lagrange_coefficients(nodes, check=False, helpers=self.helpers),
            values,
            self.dz,
        )

    def predictor(self, order: int) -> _State:
        """Return the predictor's compensation over the ``order`` newest samples.

        Parameters
        ----------
        order : int
            The predictor's order.

        Returns
        -------
        numpy.ndarray
            The increment.
        """
        return self._sum(self.nodes[:order], None, order)

    def corrector(self, order: int) -> _State:
        """Return the corrector's, through the endpoint sample.

        Parameters
        ----------
        order : int
            The step's order; the corrector has one node more.

        Returns
        -------
        numpy.ndarray
            The increment.
        """
        return self._sum([1.0, *self.nodes[:order]], self.endpoint, order)

    def defects(self, orders: list[int]) -> list[_State]:
        """Return each scored order's corrector less predictor compensation.

        Parameters
        ----------
        orders : list of int
            The orders scored, the step's own first.

        Returns
        -------
        list of numpy.ndarray
            One defect term per order.
        """
        return [self.corrector(k) - self.predictor(k) for k in orders]

    def observation(
        self, order: int, history: list[_State]
    ) -> tuple[list[float], list[_State]]:
        """Return the accepted corrector's nodes and untransported compensations.

        Dense observation applies its own flow factors ``exp((xi - theta_j) x)``;
        the history's terms are formed again from the stored rows by the caller,
        rather than carried back by an inverse flow that a damped bin would overflow.

        Parameters
        ----------
        order : int
            The accepted step's order.
        history : list of numpy.ndarray
            ``(D(z_j) - B_n) u_j`` of its ``order`` history samples, newest first.

        Returns
        -------
        tuple
            The nodes ``(1, theta_0, ...)`` and the matching compensations.
        """
        assert self.endpoint is not None
        return [1.0, *self.nodes[:order]], [self.endpoint.copy(), *history]


def source_absent(problem: Any) -> bool:
    """Return whether a problem's complete source is structurally zero.

    No response is compiled beyond the accumulation, and no in-stage absorber, no
    atomic response and no general or spatial pipeline adds a term: SEP 16 section 4.2
    then requires the exact chord rather than an interpolated compensation, which
    need not cancel even at the endpoint. Decided from the problem's structure,
    never from a sample that happens to vanish.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
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
        and problem.tdse is None
        and problem.transform.modal_tables is None
        and problem.linear.separable is None
    )


class AdamsEngine(Protocol):
    """What :class:`AdamsMarch` asks of the engine that holds its history (SEP 16).

    The march keeps the controller; an engine keeps the history's rows and does every
    state-sized operation, so that a backend can keep them where its state lives.
    :class:`HostHistory` is the reference; ``docs/dev/stepper.md`` states the contract.
    """

    dense: Any
    """The general path's last accepted step (a ``_DenseStep`` or ``None``), which the
    march sets for observation."""

    compensated: bool
    """Whether trials carry a varying operator's linear compensation (SEP 16 4.2)."""

    @property
    def n_slots(self) -> int:
        """The history store's rows."""
        ...

    def reset(self) -> None:
        """Forget the last accepted step and any trial in flight."""
        ...

    def hold(self) -> object:
        """Return the accepted state, to restore after a dense observation.

        Returns
        -------
        object
            A handle :meth:`restore` takes.
        """
        ...

    def restore(self, z: float, held: object) -> None:
        """Make a held accepted state current again at `z`.

        Parameters
        ----------
        z : float
            Its position, m.
        held : object
            What :meth:`hold` returned.
        """
        ...

    def start(self, z: float, *, stored: bool) -> _State | None:
        """Evaluate the history's first sample at the accepted state.

        Parameters
        ----------
        z : float
            The accepted position, m.
        stored : bool
            Whether the sample goes to the store's row zero.

        Returns
        -------
        numpy.ndarray or None
            The sample on the host, or ``None`` when the rows are resident.
        """
        ...

    def value(self, slot: int) -> _State | None:
        """Return a store row on the host, or ``None`` when the rows are resident.

        Parameters
        ----------
        slot : int
            The row.

        Returns
        -------
        numpy.ndarray or None
            The row, or ``None``.
        """
        ...

    def fused_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form an interaction-formulation trial; return its scored norms.

        Parameters
        ----------
        z, dz : float
            The anchor and the step, m.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            Their store rows, ``int64``.
        order : int
            The step's order.
        orders : list of int
            The orders scored, the step's own first.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order.
        """
        ...

    def fused_accept(self, keep: np.ndarray, free: int, *, interpolate: bool) -> None:
        """Adopt the held interaction-formulation trial.

        Parameters
        ----------
        keep : numpy.ndarray
            The history rows kept, ``int64``.
        free : int
            The row the new sample takes.
        interpolate : bool
            Whether the step's interpolant is kept for dense observation.
        """
        ...

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form an exponential trial; return its scored norms.

        Parameters
        ----------
        z, dz : float
            The anchor and the step, m.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            Their store rows, ``int64``.
        order : int
            The step's order.
        orders : list of int
            The orders scored, the step's own first.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order.
        """
        ...

    def exponential_accept(
        self, keep: np.ndarray, free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held exponential trial.

        Parameters
        ----------
        keep : numpy.ndarray
            The history rows kept, ``int64``.
        free : int
            The row the new sample takes.
        interpolate : bool
            Whether the step's interpolant is kept for dense observation.
        """
        ...

    def snapshot_rows(self) -> dict[str, np.ndarray]:
        """Return host copies of the rows a resume needs to continue the history.

        Returns
        -------
        dict
            The rows by name, each ``(n_slots, *shape)``, in the engine's own names
            (``HostHistory.snapshot_rows`` states the host engine's).
        """
        ...

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows :meth:`snapshot_rows` returned, refusing rows it does not keep.

        Parameters
        ----------
        rows : collections.abc.Mapping
            The rows, by name.
        shape : tuple of int
            The state's shape.

        Raises
        ------
        BackendError
            If the rows are another engine's, missing, or of the wrong shape; nothing
            is installed then.
        """
        ...

    def observe(self, position: float) -> object:
        """Return the masked field at `position` inside the last accepted step.

        Parameters
        ----------
        position : float
            The observation, m.

        Returns
        -------
        object
            The field, as the backend holds a state.
        """
        ...


class HostHistory:
    """The Adams history and its field arithmetic, on host arrays (SEP 16 phase 3b).

    :class:`AdamsMarch` keeps the controller -- the orders, the slots, rejection,
    restarts, the budget and the ledger -- and hands every state-sized operation to an
    engine: the startup sample, a trial's predictor, source, corrector and defect
    norms, the adoption and transport of an accepted step, and dense observation. This
    engine holds the history in a NumPy store and applies the given kernels (NumPy, or
    ``numba_cpu``'s compiled passes); a backend may supply its own through
    ``solprop.backends.driver.StepDriver._adams_engine``, as Kokkos does to keep
    the history resident in its engine. Between a trial and its acceptance the engine
    holds the trial; nothing else is carried across calls but the store and the last
    accepted step for dense observation.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend whose state, source evaluation and propagators are used.
    method : solprop.stepper.multistep.Multistep
        The method.
    kernels : AdamsKernels
        The interaction formulation's fused arithmetic.
    exponential : ExponentialKernels or None
        The exponential formulation's arithmetic, ``None`` for the interaction one.
    compensation : {'transported', 'untransported'}
        A varying operator's compensation treatment (SEP 16 Q8).
    """

    def __init__(
        self,
        backend: StepDriver,
        method: Multistep,
        kernels: AdamsKernels,
        exponential: ExponentialKernels | None,
        compensation: Literal["transported", "untransported"],
    ) -> None:
        """Hold the backend, the kernels and an empty store."""
        self._backend = backend
        self._method = method
        self._kernels = kernels
        self._exponential = exponential
        self._compensation = compensation
        self._interpolant = getattr(method, "interpolant", "adams")
        linear = backend._problem.linear
        # SEP 16 phase 4: a varying operator's linear compensation, integrated in the
        # reference flow's frame (Q8); the untransported treatment is a test witness.
        # A structurally absent source takes the exact chord instead (section 4.2).
        # SEP 16 phase 5: a frozen-residual coupled flow's residual is linear in the
        # state and joins the compensation, whatever the diagonal's representation.
        flow = linear.coupled_flow
        residual = flow is not None and "frozen_residual" in flow.section_methods
        self.compensated = exponential is not None and (
            residual
            or (
                linear.phi_representation in ("expansion", "table")
                and not source_absent(backend._problem)
            )
        )
        if self.compensated and flow is not None and compensation != "transported":
            msg = "the untransported compensation is a witness for diagonal operators"
            raise BackendError(msg)
        self._weights: list[np.ndarray | None] = []
        self._trial_weights: np.ndarray | None = None
        self._store: np.ndarray | None = None
        self._states: np.ndarray | None = None
        self._rates: np.ndarray | None = None
        self.dense: Any = None
        self._exponential_dense: _ExponentialStep | None = None
        self._fused_trial: _FusedTrial | None = None
        self._exponential_trial: _ExponentialTrial | None = None

    def reset(self) -> None:
        """Forget the last accepted step and any trial in flight."""
        self.dense = None
        self._exponential_dense = None
        self._fused_trial = None
        self._exponential_trial = None

    @property
    def n_slots(self) -> int:
        """The history store's rows: the method's order and two more.

        Returns
        -------
        int
            ``order + 2``.
        """
        return self._method.order + 2

    # --- the accepted state, held across a dense observation --------------------------

    def hold(self) -> object:
        """Return the accepted state, to restore after a dense observation.

        Returns
        -------
        object
            The backend's host copy of the accepted state.
        """
        return self._backend._adams_state()

    def restore(self, z: float, held: object) -> None:
        """Make a held accepted state current again at `z`.

        Parameters
        ----------
        z : float
            Its position, m.
        held : object
            What :meth:`hold` returned.
        """
        self._backend._adams_restore(z, cast("_State", held))

    # --- startup ----------------------------------------------------------------------

    def start(self, z: float, *, stored: bool) -> _State:
        """Evaluate the source at the accepted state, the history's first sample.

        Parameters
        ----------
        z : float
            The accepted position, m.
        stored : bool
            Whether the sample goes to the store's row zero (the fused and exponential
            paths) or is returned alone (the general path).

        Returns
        -------
        numpy.ndarray
            The sample: the transported source in the interaction formulation, the
            untransported one in the exponential formulation.
        """
        backend = self._backend
        state = backend._adams_state()
        backend._hoist(z)
        if self._exponential is not None:
            value = backend._adams_lab_source(state, z, purpose="startup")
        else:
            value, _ = backend._adams_source(state, z, z, purpose="startup")
        if not stored:
            return value
        if self._store is None or self._store.shape[1:] != value.shape:
            # Zeroed, not empty: the rows a short history has not reached go into a
            # resume snapshot as they are, and a file has to be reproducible.
            # In the state's precision: complex64 on SEP 33's single-precision path.
            self._store = np.zeros(
                (self.n_slots, *value.shape),
                dtype=np.result_type(value, np.complex64),
            )
            if self.compensated:
                self._states = np.zeros_like(self._store)
                # The operator's rates are binary64 host data, differenced against
                # the step's exponent, so they stay binary64 under SEP 33 too.
                self._rates = np.zeros(self._store.shape, dtype=np.complex128)
        self._store[0] = value
        if self.compensated:
            assert self._states is not None and self._rates is not None
            # The startup sample's state and rate, read on the far side of a kink
            # that restarted the history.
            self._states[0] = state
            self._rates[0] = phi_rate(
                backend._problem.linear, z, helpers=backend.helpers
            )
            # A frozen-residual flow's weights at the sample, on the far side too.
            start = cast("CoupledBlock | None", backend._adams_block(z, 0.0))
            self._weights = [None] * self.n_slots
            self._weights[0] = None if start is None else start.end_weights
        return self._store[0]

    # --- the interaction formulation's fused path -------------------------------------

    def fused_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form one fused trial and return the scaled norms of its scored defects.

        The predictor's lab-frame field, the source there (one call, labelled a
        predictor), and in one pass the corrected field and each scored order's
        defect: its ``k + 1``-node corrector through the new source less its
        ``k``-node predictor. The trial is held for :meth:`fused_accept`.

        Parameters
        ----------
        z, dz : float
            The anchor and the step, m.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            Their store rows, ``int64``.
        order : int
            The step's order.
        orders : list of int
            The orders scored, the step's own first.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order.
        """
        backend, kernels = self._backend, self._kernels
        store = self._store
        assert store is not None
        state = backend._adams_state()
        backend._hoist(z)
        propagator = backend._adams_propagator(z, z + dz)
        predictor = kernels.weights(nodes[:order])
        lab = kernels.predict(state, propagator, store, slots[:order], predictor, dz)
        source, _ = backend._adams_source(lab, z, z + dz, purpose="predictor")
        rows, corrector = _defect_rows(kernels, nodes, orders)
        width = rows.shape[1]
        mask = backend._problem.grids.norm_mask
        advanced, norms = kernels.correct(
            state,
            propagator,
            source,
            store,
            slots[: width - 1],
            corrector,
            rows,
            dz,
            mask,
            _divisor(backend, state),
        )
        self._fused_trial = _FusedTrial(
            z=z,
            dz=dz,
            state=state,
            propagator=propagator,
            source=source,
            advanced=advanced,
            nodes=nodes,
            slots=slots,
            order=order,
        )
        return norms

    def fused_accept(self, keep: np.ndarray, free: int, *, interpolate: bool) -> None:
        """Adopt the held fused trial: the state, its interpolant, the new sample.

        Parameters
        ----------
        keep : numpy.ndarray
            The history rows kept, ``int64``; each is re-anchored at the step's end.
        free : int
            The row the new sample takes.
        interpolate : bool
            Whether the step's interpolant is kept for dense observation.
        """
        trial = self._fused_trial
        store = self._store
        assert trial is not None and store is not None
        self._fused_trial = None
        finished = self._backend._adams_adopt(trial.advanced, trial.dz)
        order = trial.order
        if interpolate:
            # Copied before the transport re-anchors the rows in place: the step's own
            # interpolant is in its anchor's frame.
            slopes = [
                trial.source,
                *(store[slot].copy() for slot in trial.slots[:order]),
            ]
            self.dense = _DenseStep(
                z=trial.z,
                dz=trial.dz,
                start=trial.state.copy(),
                end=finished / trial.propagator,
                start_slope=slopes[1],
                end_slope=trial.source,
                nodes=np.array([1.0, *trial.nodes[:order]]),
                slopes=slopes,
            )
        self._kernels.transport(store, keep, trial.propagator, free, trial.source)

    def value(self, slot: int) -> _State:
        """Return a store row, for the history entry that names it.

        Parameters
        ----------
        slot : int
            The row.

        Returns
        -------
        numpy.ndarray
            The row itself, not a copy.
        """
        assert self._store is not None
        return self._store[slot]

    # --- persistence across a resume --------------------------------------------------

    def snapshot_rows(self) -> dict[str, np.ndarray]:
        """Return copies of the rows a resume needs to continue the history.

        Returns
        -------
        dict
            ``"store"``, the samples; for a compensated march also ``"states"`` and
            ``"rates"``, and ``"weights_<row>"`` for each row a frozen-residual flow
            weights.
        """
        assert self._store is not None
        rows = {"store": self._store.copy()}
        if self.compensated:
            assert self._states is not None and self._rates is not None
            rows["states"] = self._states.copy()
            rows["rates"] = self._rates.copy()
        for slot, weights in enumerate(self._weights):
            if weights is not None:
                rows[f"weights_{slot}"] = np.array(weights, copy=True)
        return rows

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows :meth:`snapshot_rows` returned, for a state of `shape`.

        Parameters
        ----------
        rows : collections.abc.Mapping
            The rows, by name.
        shape : tuple of int
            The state's shape.

        Raises
        ------
        BackendError
            If a row is missing, does not have the store's shape, or is one this
            history does not keep -- the native engine's applied frozen-residual
            generators (``"applied"``) among them.
        """
        expected = (self.n_slots, *shape)
        names = ("store", "states", "rates") if self.compensated else ("store",)
        # A compensated history keeps a frozen-residual flow's weights per sample.
        kept = set(names) | (
            {f"weights_{slot}" for slot in range(self.n_slots)}
            if self.compensated
            else set()
        )
        foreign = sorted(set(rows) - kept)
        if foreign:
            name = self._backend.name
            if "applied" in foreign:
                msg = (
                    f"the {name} backend cannot continue this Adams history: the "
                    "snapshot holds the native engine's applied frozen-residual "
                    "generators ('applied'), and this engine keeps each sample's "
                    "weights instead"
                )
            else:
                msg = (
                    f"the {name} backend cannot continue this Adams history: its "
                    f"engine keeps no {foreign} rows"
                )
            raise BackendError(msg)
        for name in names:
            row = rows.get(name)
            if row is None or row.shape != expected:
                msg = (
                    f"the snapshot's {name!r} rows do not fit a store of shape "
                    f"{expected}"
                )
                raise BackendError(msg)
        self.reset()
        self._store = np.array(rows["store"], dtype=np.complex128, copy=True)
        if self.compensated:
            self._states = np.array(rows["states"], dtype=np.complex128, copy=True)
            self._rates = np.array(rows["rates"], dtype=np.complex128, copy=True)
        self._weights = [
            None
            if f"weights_{slot}" not in rows
            else np.array(rows[f"weights_{slot}"], copy=True)
            for slot in range(self.n_slots)
        ]

    # --- the exponential formulation --------------------------------------------------

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form one exponential trial and return the scaled norms of its defects.

        The arithmetic of :meth:`fused_trial` with untransported samples: the predictor
        and the corrector integrate their interpolants exactly against the step's flow
        ``exp(x)``, ``x = Phi(z + h) - Phi(z)``; the defects of the neighbouring orders
        are scored from the same moments. A varying operator adds the compensation of
        its samples (SEP 16 section 4.2). The trial is held for
        :meth:`exponential_accept`.

        Parameters
        ----------
        z, dz : float
            The anchor and the step, m.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            Their store rows, ``int64``.
        order : int
            The step's order.
        orders : list of int
            The orders scored, the step's own first.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order.
        """
        backend, kernels = self._backend, self._exponential
        assert kernels is not None
        store = self._store
        assert store is not None
        state = backend._adams_state()
        backend._hoist(z)
        width = 1 + max(orders)
        # The exponent is the backend's own increment of Phi over the step, so the
        # flow below is exactly the propagator the interaction formulation applies.
        # Copied: a z table's host evaluator returns its own reused buffer, which the
        # source evaluation writes again.
        exponent = np.array(backend._delta_phi(backend._stage_scalars(z, z + dz)))
        # A coupled section's step (SEP 16 phase 5): its flow and moments are matrix
        # functions of the frozen block on the bins that carry one, applied by the
        # reference passes.
        block = cast("CoupledBlock | None", backend._adams_block(z, dz))
        moments: Any
        flow: Any
        if block is not None:
            kernels = kernels.coupled()
            moments = kernels.coupled_moments(exponent, width, block, dz)
            flow = moments
        else:
            moments = kernels.moments(exponent, width)
            flow = kernels.flow(exponent)
        if state.dtype == np.complex64:
            # SEP 33: the flow and its moments are evaluated in binary64, like every
            # propagator, and rounded once to the state's binary32.
            moments = np.asarray(moments).astype(np.complex64)
            flow = np.asarray(flow).astype(np.complex64)
        lab = kernels.predict(
            state,
            flow,
            moments,
            store,
            slots[:order],
            lagrange_coefficients(nodes[:order], check=False, helpers=backend.helpers),
            dz,
        )
        compensation = None
        if self.compensated:
            compensation = self._compensation_terms(
                exponent, moments, nodes, slots[: width - 1], dz, block
            )
            lab = lab + compensation.predictor(order)
            if state.dtype == np.complex64:
                # SEP 33: the compensation is formed in binary64 from the binary32
                # history and the prediction rounded once.
                lab = lab.astype(np.complex64)
        source = backend._adams_lab_source(lab, z + dz, purpose="predictor")
        _, corrector = _exponential_rows(nodes, orders, width, helpers=backend.helpers)
        rows = kernels.defect_rows(nodes, orders, width)
        extra = None
        if compensation is not None:
            rate = phi_rate(
                backend._problem.linear, z + dz, before=True, helpers=backend.helpers
            )
            residual = None
            if block is not None and block.residual is not None:
                assert block.end_weights is not None
                residual = block.residual(lab, block.end_weights)
            compensation.add_endpoint(rate, lab, residual)
            self._trial_weights = None if block is None else block.end_weights
            extra = (compensation.corrector(order), compensation.defects(orders))
        mask = backend._problem.grids.norm_mask
        advanced, norms = kernels.correct(
            state,
            flow,
            moments,
            source,
            store,
            slots[: width - 1],
            corrector,
            rows,
            dz,
            mask,
            _divisor(backend, state),
            extra,
        )
        if state.dtype == np.complex64:
            advanced = np.asarray(advanced).astype(np.complex64)
        self._exponential_trial = _ExponentialTrial(
            z=z,
            dz=dz,
            state=state,
            exponent=exponent,
            corrector=corrector,
            source=source,
            lab=lab,
            advanced=advanced,
            slots=slots,
            order=order,
            compensation=compensation,
        )
        return norms

    def exponential_accept(
        self, keep: np.ndarray, free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held exponential trial; the new sample takes row `free`.

        Parameters
        ----------
        keep : numpy.ndarray
            The history rows kept, ``int64``; nothing moves in this formulation.
        free : int
            The row the new sample takes.
        interpolate : bool
            Whether the step's interpolant is kept for dense observation.
        """
        del keep  # untransported samples are never re-anchored
        trial = self._exponential_trial
        store = self._store
        assert trial is not None and store is not None
        self._exponential_trial = None
        self._backend._adams_adopt(trial.advanced, trial.dz)
        z, dz, order, slots = trial.z, trial.dz, trial.order, trial.slots
        compensation = trial.compensation
        if interpolate:
            # Copied: the store's rows are overwritten by later steps.
            self._exponential_dense = _ExponentialStep(
                z=z,
                dz=dz,
                start=trial.state.copy(),
                exponent=trial.exponent,
                coefficients=trial.corrector[: order + 1, : order + 1].copy(),
                values=[
                    trial.source.copy(),
                    *(store[slot].copy() for slot in slots[:order]),
                ],
                compensation=None
                if compensation is None
                else compensation.observation(
                    order, self._history_terms(trial.exponent, dz, slots[:order])
                ),
            )
        store[free] = trial.source
        if compensation is not None:
            assert self._states is not None and self._rates is not None
            assert compensation.endpoint_rate is not None
            self._states[free] = trial.lab
            self._rates[free] = compensation.endpoint_rate
            # One entry per slot, set with the startup sample or a resumed history.
            self._weights[free] = self._trial_weights

    def _history_terms(
        self, exponent: _State, dz: float, slots: np.ndarray
    ) -> list[_State]:
        """Return ``(D(z_j) - B_n) u_j`` of history samples, untransported.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = h B_n``.
        dz : float
            The step, m.
        slots : numpy.ndarray
            The samples, newest first.

        Returns
        -------
        list of numpy.ndarray
            One compensation per sample.
        """
        assert self._states is not None and self._rates is not None
        reference = exponent / dz
        return [(self._rates[slot] - reference) * self._states[slot] for slot in slots]

    def _compensation_terms(
        self,
        exponent: _State,
        moments: Any,
        nodes: list[float],
        slots: np.ndarray,
        dz: float,
        block: CoupledBlock | None = None,
    ) -> "_Compensation":  # noqa: UP037 - defined below
        """Return the trial's linear compensation of its history samples.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = h B_n``, the step's chord exponent.
        moments : numpy.ndarray
            The trial's moments, for the untransported witness.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            The history rows the trial reads, newest first.
        dz : float
            The step, m.
        block : CoupledBlock or None, optional
            A coupled step's frozen block: the samples' residuals against it join
            their compensation, carried by its forward flow.

        Returns
        -------
        _Compensation
            The samples' ``(D(z_j) - B_n) u_j`` and the trial's weights over them.
        """
        assert self._states is not None and self._rates is not None
        assert self._exponential is not None
        used = list(nodes[: len(slots)])
        transported = self._compensation == "transported"
        # The untransported witness keeps the samples in their own frame: a span of 0.
        spans = np.array([1.0 - node if transported else 0.0 for node in used])
        if block is not None:
            carried = self._exponential.coupled().carry_coupled(
                exponent,
                dz,
                (self._rates, self._states, self._weights),
                slots,
                spans,
                block,
            )
        else:
            carried = self._exponential.carry(
                exponent, dz, self._rates, self._states, slots, spans
            )
        return _Compensation(
            exponent=exponent,
            moments=moments,
            nodes=used,
            carried=carried,
            dz=dz,
            treatment=self._compensation,
            helpers=self._backend.helpers,
        )

    # --- dense observation ------------------------------------------------------------

    def observe(self, position: float) -> object:
        """Return the field at `position` inside the last accepted step.

        Parameters
        ----------
        position : float
            The observation, m, inside the last accepted step.

        Returns
        -------
        object
            The masked lab-frame field at `position`, as the backend holds a state:
            a host array, or Kokkos's native state loaded with it.
        """
        if self._exponential is not None:
            return self._observe_exponential(position)
        step = cast("_DenseStep | None", self.dense)
        assert step is not None
        backend = self._backend
        theta = (position - step.z) / step.dz
        if self._interpolant == "adams":
            weights = integrated_weights(
                step.nodes, theta, check=False, helpers=backend.helpers
            )
            value = step.start + _combine(weights, step.slopes, step.dz)
        else:
            value = _hermite(step, theta)
        # A bin the linear flow changes by more than an e-fold over the step (the
        # spectral collar's) takes the anchor's linear flow: no polynomial follows its
        # exponentially growing derivative (as for the Runge-Kutta extension).
        exponent = backend._delta_phi(backend._stage_scalars(step.z, step.z + step.dz))
        value = np.where(np.abs(np.real(exponent)) > 1.0, step.start, value)
        return backend._apply_mask(backend._adams_propagator(step.z, position) * value)

    def _observe_exponential(self, position: float) -> object:
        """Return the field at `position` inside the last accepted exponential step.

        The accepted corrector's interpolant integrated exactly against the flow to the
        fraction ``xi`` of the step (SEP 16 section 4.3):
        ``exp(xi x) u_n + h sum_k k! xi**(k+1) phi_{k+1}(xi x) sum_j c_jk F_j``. Every
        factor is a forward flow, so a strongly damped bin needs no guard; no source
        is evaluated and nothing enters the history.

        Parameters
        ----------
        position : float
            The observation, m, inside the last accepted step.

        Returns
        -------
        object
            The masked lab-frame field at `position`, as the backend holds a state.
        """
        step = self._exponential_dense
        assert step is not None
        # One evaluation per plane: the reference passes serve every backend.
        helpers = self._backend.helpers
        kernels = ExponentialKernels(helpers=helpers)
        xi = (position - step.z) / step.dz
        count = step.coefficients.shape[1]
        moments = kernels.moments(xi * step.exponent, count)
        scaled = moments * np.array([xi ** (k + 1) for k in range(count)]).reshape(
            (count, *([1] * step.start.ndim))
        )
        value = np.exp(xi * step.exponent) * step.start + kernels.apply(
            scaled, step.coefficients, step.values, step.dz
        )
        if step.compensation is not None:
            nodes, terms = step.compensation
            weights = integrated_weights(
                np.asarray(nodes), xi, check=False, helpers=helpers
            )
            # e^{(xi - theta_j) x} is a forward flow for every history sample; for
            # the endpoint sample it is an inverse one, which a bin the step damps by
            # more than an e-fold would amplify: such a bin takes the anchor's flow,
            # as the interaction formulation's interpolant does (ADR-039).
            for weight, node, term in zip(weights, nodes, terms, strict=True):
                value = (
                    value
                    + step.dz * weight * np.exp((xi - node) * step.exponent) * term
                )
            damped = np.real(step.exponent) < -1.0
            value = np.where(damped, np.exp(xi * step.exponent) * step.start, value)
        return self._backend._apply_mask(value)


def _divisor(backend: StepDriver, state: _State) -> np.ndarray:
    """Return the defect norm's scale per bin, the accepted state's before the step.

    See :func:`_norm_function` for why the state before the step is the scale.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend, for the controller and the in-band mask.
    state : numpy.ndarray
        The accepted state before the step.

    Returns
    -------
    numpy.ndarray
        The scale of every bin, shape ``(n_omega,)``.
    """
    mask = backend._problem.grids.norm_mask
    controller = backend._norm_controller
    magnitude = float(np.linalg.norm(state[..., mask]))
    scale = controller.scale(magnitude, magnitude)
    return (
        scale
        if isinstance(scale, np.ndarray)
        else np.full(state.shape[-1], float(scale))
    )


class AdamsMarch:
    """March a reference backend's state with an interaction-picture Adams method.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend whose state, source evaluation and propagators are used. It owns
        the step history and the statistics rows; this object owns the derivative
        history and the call ledger.
    method : solprop.stepper.multistep.Multistep
        The method.
    kernels : AdamsKernels or None, optional
        The field arithmetic of the fused path; ``None`` takes the NumPy reference.
    exponential : ExponentialKernels or None, optional
        The exponential formulation's field arithmetic; ``None`` takes the NumPy
        reference. Used only when the method's formulation is ``"exponential"``.
    compensation : {'transported', 'untransported'}, optional
        How a varying operator's linear compensation is integrated (SEP 16 Q8):
        ``"transported"``, the delivered treatment, or ``"untransported"``, kept as a
        test witness only.
    """

    def __init__(
        self,
        backend: StepDriver,
        method: Multistep,
        kernels: AdamsKernels | None = None,
        exponential: ExponentialKernels | None = None,
        *,
        compensation: Literal["transported", "untransported"] = "transported",
    ) -> None:
        """Hold the backend and start with no history."""
        self._backend = backend
        self._method = method
        # Whether the run interpolates its planes, and with what (ADR-036, ADR-039).
        self._interpolate = backend._problem.stepper.interpolate_planes
        self._interpolant = getattr(method, "interpolant", "adams")
        self._kernels = (
            AdamsKernels(helpers=backend.helpers) if kernels is None else kernels
        )
        # SEP 16's exponential formulation keeps untransported samples in the same
        # store and shares everything else: the lifecycle, the controller, the ledger.
        self._exponential: ExponentialKernels | None = None
        if method.formulation == "exponential":
            self._exponential = (
                ExponentialKernels(helpers=backend.helpers)
                if exponential is None
                else exponential
            )
        # The selected method's family, one-order-higher PEC, takes the fused path:
        # three passes a step over a store of history rows instead of a list of arrays
        # combined one product at a time.
        self._fused = (
            self._exponential is None
            and method.corrector == "pec"
            and all(
                method.corrector_nodes(order) == order + 1
                for order in range(1, method.order + 1)
            )
        )
        # Every state-sized operation goes to an engine (SEP 16 phase 3b): the host
        # one here, or one the backend keeps resident for the fused and exponential
        # paths. The general path's arithmetic stays in this class.
        resident = (
            backend._adams_engine(method, compensation)
            if self._fused or self._exponential is not None
            else None
        )
        self._engine: AdamsEngine = (
            HostHistory(backend, method, self._kernels, self._exponential, compensation)
            if resident is None
            else cast("AdamsEngine", resident)
        )
        self._history: list[_Derivative] = []
        self._rejections = 0
        self._last_rejected = False
        self._order = 1 if method.variable_order else method.order
        self._ahead: tuple[float, object] | None = None
        self.ledger = CallLedger()
        # An in-stage temporal collar adds -alpha(t) / 2 to the derivative: in the
        # exponential formulation, whose linear flow is exact, it is the one stiff term
        # the explicit history integrates, and its largest rate bounds each order's
        # stable step, which the order choice weighs (`_choose_order`).
        responses = backend._problem.responses
        collar = responses.temporal_absorber
        self._collar_rate = (
            0.5 * float(np.max(collar))
            if collar is not None
            and responses.absorber_scheme == "nonlinear"
            and method.formulation == "exponential"
            else 0.0
        )

    def reset(self) -> None:
        """Forget the history: the state was loaded or changed outside the march."""
        self._history = []
        self._rejections = 0
        self._engine.reset()
        self._ahead = None

    def begin_run(self) -> None:
        """Start a run's accounting afresh: the call ledger and the last rejection.

        Called by :meth:`~solprop.backends.driver.StepDriver.set_initial`, so that a
        backend loaded for another run counts that run's calls alone and chooses its
        first order as a fresh backend would (SEP 22 section 4.7). A resume restores
        its persisted ledger afterwards.
        """
        self.ledger = CallLedger()
        self._last_rejected = False

    def snapshot(self) -> MarchSnapshot | None:
        """Return the march's state at its accepted position, for a resume.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            The snapshot, or ``None`` where there is nothing a resume could continue:
            no history yet, the march ahead of an interpolated observation, or a march
            that cannot carry its history (:meth:`export_refusal`).
        """
        if (
            self.export_refusal() is not None
            or not self._history
            or self._ahead is not None
        ):
            return None
        export = self._engine.snapshot_rows
        backend = self._backend
        return MarchSnapshot(
            z=float(backend._z),
            state=np.array(backend._adams_state(), dtype=np.complex128, copy=True),
            proposal=float(backend._proposal),
            method=repr(self._method),
            order=int(self._order),
            rejections=int(self._rejections),
            last_rejected=bool(self._last_rejected),
            positions=np.array([entry.z for entry in self._history], dtype=np.float64),
            identities=tuple(entry.identity for entry in self._history),
            slots=np.array([entry.slot for entry in self._history], dtype=np.int64),
            rows=export(),
        )

    def export_refusal(self) -> str | None:
        """Return why this march cannot carry its history across a resume, or ``None``.

        The fused and exponential paths keep their history in an engine, and every
        engine exports and installs its rows (:class:`AdamsEngine`, SEP 29 section
        6.6). The general path combines a list of host arrays one product at a time
        and keeps no store a snapshot could hold.

        Returns
        -------
        str or None
            The reason, or ``None`` when the march exports and installs its history.
        """
        if not (self._fused or self._exponential is not None):
            return (
                "the general Adams path keeps its history as host arrays a resume "
                "does not carry"
            )
        return None

    def load(self, snapshot: MarchSnapshot) -> str | None:
        """Continue from a snapshot taken where the backend's state now stands.

        The backend has been loaded at the snapshot's position; this installs the
        accepted state exactly as it was held, the history, the order, the rejection
        count and the proposed step, so that the march goes on as the uninterrupted
        one would have. Nothing is changed when the snapshot cannot be used.

        Parameters
        ----------
        snapshot : solprop.output.schema.MarchSnapshot
            What :meth:`snapshot` returned, possibly read back from a file.

        Returns
        -------
        str or None
            ``None`` when the history was continued, otherwise why it starts afresh.
        """
        backend = self._backend
        reason = self._refusal(snapshot)
        if reason is not None:
            return reason
        load_rows = self._engine.load_rows
        shape = tuple(backend._problem.shape)
        try:
            load_rows(snapshot.rows, shape)
        except BackendError as error:
            return str(error)
        backend._adams_restore(snapshot.z, np.array(snapshot.state, copy=True))
        self._history = [
            _Derivative(float(z), self._engine.value(int(slot)), identity, int(slot))
            for z, identity, slot in zip(
                snapshot.positions, snapshot.identities, snapshot.slots, strict=True
            )
        ]
        self._order = snapshot.order
        self._rejections = snapshot.rejections
        self._last_rejected = snapshot.last_rejected
        self._ahead = None
        backend._proposal = snapshot.proposal
        return None

    def _refusal(self, snapshot: MarchSnapshot) -> str | None:
        """Return why a snapshot cannot continue this march, or ``None``.

        Parameters
        ----------
        snapshot : solprop.output.schema.MarchSnapshot
            The candidate.

        Returns
        -------
        str or None
            The reason, or ``None`` when the snapshot fits.
        """
        backend = self._backend
        refusal = self.export_refusal()
        if refusal is not None:
            return refusal
        checks = (
            (
                snapshot.method != repr(self._method),
                f"the snapshot was taken by {snapshot.method}",
            ),
            (
                snapshot.z != backend._z,
                f"the snapshot is at z = {snapshot.z!r} m and the run resumes at "
                f"z = {backend._z!r} m",
            ),
            (
                snapshot.state.shape != tuple(backend._problem.shape),
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
                bool(
                    np.any(
                        (snapshot.slots < 0) | (snapshot.slots >= self._engine.n_slots)
                    )
                ),
                "the snapshot's history names no row of this engine",
            ),
        )
        return next((reason for failed, reason in checks if failed), None)

    # --- evaluation -------------------------------------------------------------------

    def _source(
        self,
        lab: _State,
        anchor: float,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> tuple[_State, _State]:
        """Return the source at a lab-frame field, in the anchor's frame, and U.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`.
        anchor, position : float
            The frame's anchor and the evaluation position, m.
        purpose : str, optional
            Atomic evaluation label, distinguishing startup and predictor work.

        Returns
        -------
        tuple
            ``U(anchor, position) N(lab, position)`` and ``U(position, anchor)``.
        """
        return self._backend._adams_source(lab, anchor, position, purpose=purpose)

    # --- the march --------------------------------------------------------------------

    def march_to(self, target: float) -> None:
        """Step the backend's state adaptively to `target`, landing on it exactly.

        Parameters
        ----------
        target : float
            Position to reach, m.

        Raises
        ------
        ConvergenceError
            If the proposal falls below the contract's ``dz_min``.
        """
        from solprop.backends.driver import reached  # noqa: PLC0415 - import cycle

        backend = self._backend
        settings = backend._problem.stepper
        method = self._method
        dense = self._interpolate
        if self._ahead is not None:
            # The march is ahead of the last observation; resume it there.
            self._engine.restore(*self._ahead)
            self._ahead = None
        if backend._natural_stop_due(target):
            return
        if settings.schedule is not None:
            self._replay_to(settings.schedule, target)
            return
        breakpoints = backend._problem.linear.phi_breakpoints
        if not method.interval_cap:
            breakpoints = np.array([breakpoints[0], breakpoints[-1]])
        targets = StepTargets(
            # Dense observation steps across the save planes (SEP 13 section 6.4);
            # hard kinks and the end of the run are still landed on.
            save_planes=np.empty(0)
            if dense
            else np.asarray(settings.save_planes, dtype=np.float64),
            hard_kinks=backend._problem.linear.hard_kinks,
            breakpoints=breakpoints,
            dz_max=settings.dz_max,
            z_end=settings.z_end if dense else target,
            stretch=settings.stretch,
        )
        while not reached(backend._z, target):
            self._afford()
            try:
                if not self._history:
                    self._start()
                stop = landing(backend._z, backend._proposal, targets)
                dz, end, kink = self._approach(stop, targets, target)
                # The accepted step moves the position with the state, and restarts
                # the history at a kink, in one commit (`_land`).
                accepted, norm = self._attempt(dz, landing=(end, kink))
                backend._record(end, dz, norm, accepted)
                if accepted:
                    backend._take_step_stats()
                    if backend._natural_stop(target):
                        return
            finally:
                backend._release_source_calls()
        if dense and backend._z > target:
            self._ahead = (backend._z, self._engine.hold())
            backend._state = cast("Any", self._observe(target))
            backend._z = target
            self.ledger.dense_observations += 1

    def _replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay a prescribed schedule's steps to `target` (SEP 16 section 5).

        The schedule fixes the accepted positions: every step is taken as given and
        accepted, as the Runge-Kutta replay does (``docs/dev/stepper.md`` section 8),
        while the predictor-corrector estimate is still computed and recorded, so a
        schedule too coarse for its tolerance is reported rather than silent. The
        order is chosen by the same rule as an adaptive run's, from the same
        estimates; with no rejection it never has to wait a step to rise, so a replay
        of an adaptive run's steps can choose differently where that run rejected,
        while a replay of a replay is identical. A hard kink, an edge of the schedule,
        restarts the history as it does in an adaptive run.

        Parameters
        ----------
        schedule : solprop.stepper.schedule.FixedSchedule
            The schedule to replay.
        target : float
            Position to reach, m.

        Raises
        ------
        BackendError
            If the schedule runs out before the target is reached.
        """
        from solprop.backends.driver import replay_tolerance  # noqa: PLC0415 - cycle

        backend = self._backend
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        while (
            backend._schedule_index < schedule.n_steps and backend._z < target - slack
        ):
            dz = float(schedule.steps[backend._schedule_index])
            end = float(edges[backend._schedule_index + 1])
            norm = self.replay_step(dz, end, slack)
            backend._schedule_index += 1
            backend._record(end, dz, norm, accepted=True)
            backend._accepted_norms.append(norm)
            backend._take_step_stats()
            if backend._natural_stop(target):
                return
        if backend._z < target - slack:
            msg = (
                f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                f"cannot reach z = {target:.6g} m"
            )
            raise BackendError(msg)

    def replay_step(self, dz: float, end: float, slack: float) -> float:
        """Take one prescribed step from the backend's position and accept it.

        The unit a lockstep batch advances each member by
        (:class:`solprop.backends.batch.LockstepRunner`): the budget, the startup of
        an empty history, the step and, when `end` is a hard kink, the restart.

        Parameters
        ----------
        dz : float
            The step, m.
        end : float
            The schedule's edge it ends on, m.
        slack : float
            How far an edge may sit from a kink it lands on, m.

        Returns
        -------
        float
            The step's scaled error norm, recorded and not acted on.

        Raises
        ------
        ConvergenceError
            If the norm is not finite: the field has diverged, and a replay cannot
            reject the step.
        """
        self._afford()
        try:
            if not self._history:
                self._start()
            kinks = self._backend._problem.linear.hard_kinks
            kink = bool(np.any(np.abs(np.asarray(kinks) - end) <= slack))
            _, norm = self._attempt(dz, fixed=True, landing=(end, kink))
            if not math.isfinite(norm):
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
        finally:
            self._backend._release_source_calls()
        return norm

    def _observe(self, position: float) -> object:
        """Return the field at `position` inside the last accepted step.

        Interpolated and carried to `position` by the exact linear flow (SEP 13
        section 6.4, SEP 16 section 4.3); never a history point, and no source call.

        Parameters
        ----------
        position : float
            The observation, m, inside the last accepted step.

        Returns
        -------
        object
            The masked lab-frame field at `position`, as the backend holds a state:
            a host array, or Kokkos's native state loaded with it.
        """
        return self._engine.observe(position)

    def _afford(self) -> None:
        """Refuse a trial the call budget cannot pay for, before any call is made.

        Raises
        ------
        ResponseBudgetError
            Carrying the last accepted position and state and the calls spent.
        """
        budget = self._method.max_calls
        cost = (0 if self._history else 1) + (
            2 if self._method.corrector == "pece" else 1
        )
        spent = self.ledger.total
        backend = self._backend
        if budget is not None and spent + cost > budget:
            msg = (
                f"the source-call budget of {budget} is spent ({spent} charged) at "
                f"z = {backend._z:.6g} m: the next trial needs {cost}; "
                "the run stops at "
                "its last accepted step, unfinished"
            )
            raise ResponseBudgetError(
                msg, z=backend._z, field=backend._adams_state().copy(), calls=spent
            )
        backend._reserve_source_calls(cost)

    def _start(self) -> None:
        """Evaluate the source at the accepted field to (re)start the history."""
        backend = self._backend
        z = backend._z
        self.ledger.startup += 1
        stored = self._fused or self._exponential is not None
        value = self._engine.start(z, stored=stored)
        self._history = [_Derivative(z, value, "accepted", 0 if stored else -1)]
        self._order = 1 if self._method.variable_order else self._method.order

    def _approach(
        self, stop: object, targets: StepTargets, target: float
    ) -> tuple[float, float, bool]:
        """Return the step to take, where it ends and whether it lands on a kink.

        Parameters
        ----------
        stop : solprop.stepper.schedule.Landing
            The landing the shared rule chose for the proposal.
        targets : solprop.stepper.schedule.StepTargets
            The caps and landings.
        target : float
            The end of this march, m.

        Returns
        -------
        tuple
            The step, m; its end position, m; and whether that end is a hard kink.
        """
        from solprop.stepper.schedule import Landing  # noqa: PLC0415 - typing only

        assert isinstance(stop, Landing)
        backend = self._backend
        z = backend._z
        landed = stop.z_target is not None
        if not landed and self._method.equal_approach:
            # A full step followed by a sliver to a forced landing is a large step
            # ratio, which variable-step Adams tolerates poorly (SEP 13 section 4.4):
            # between one and two proposals away, take two equal steps instead.
            ahead = _next_landing(z, targets, target)
            if ahead is not None and stop.dz < ahead - z < 2.0 * stop.dz:
                half = 0.5 * (ahead - z)
                return half, z + half, False
        end = stop.z_target if landed else z + stop.dz
        assert end is not None
        # A kink that is also a save plane may be reported as the plane; it still
        # restarts the history, which must not interpolate across a jump in D.
        # The run's own end is not one: nothing follows it to restart.
        tolerance = 8 * float(np.spacing(max(abs(end), 1.0)))
        interior = end < backend._problem.stepper.z_end - tolerance
        on_kink = interior and bool(
            np.any(np.abs(targets.hard_kinks - end) <= tolerance)
        )
        return stop.dz, end, stop.target is LandingTarget.HARD_KINK or on_kink

    def _attempt(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one step of length `dz` and commit it if accepted.

        Parameters
        ----------
        dz : float
            The step, m.
        fixed : bool, optional
            Whether the step is a prescribed schedule's, accepted whatever its norm.
        landing : tuple or None, optional
            Where an accepted step ends, m, and whether that is a hard kink, which
            restarts the history; ``None`` ends it at ``z + dz`` and restarts nothing.

        Returns
        -------
        tuple
            Whether the step was accepted, and its scaled error norm.
        """
        if self._exponential is not None:
            return self._attempt_exponential(dz, fixed=fixed, landing=landing)
        if self._fused:
            return self._attempt_fused(dz, fixed=fixed, landing=landing)
        backend = self._backend
        state = backend._adams_state()
        z = backend._z
        backend._hoist(z)
        trial = _Trial.start(self._history, self._order, z, dz, state, backend)
        if self._method.corrector == "none" and len(trial.values) > trial.order:
            # Free rejection on the history alone: the next Adams-Bashforth formula's
            # difference from this one is this one's defect (SEP 13 section 4.3).
            wider = integrated_weights(
                trial.nodes[: trial.order + 1], check=False, helpers=backend.helpers
            )
            early = _combine(wider, trial.values[: trial.order + 1], dz)
            norm = trial.norm(early - trial.predictor)
            if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
                self._reject(dz, norm, trial.order)
                return False, norm

        self.ledger.predictor += 1
        source, _ = self._source(
            trial.lab(trial.predictor), z, z + dz, purpose="predictor"
        )
        increment, defect, formula = self._correct(trial, source)
        norm = trial.norm(defect)
        if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
            self.ledger.rejected += 1
            self._reject(dz, norm, trial.order)
            return False, norm
        self._commit(trial, trial.lab(increment), source, norm, formula, landing)
        return True, norm

    def _attempt_fused(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one step of the one-order-higher PEC family on the fused path.

        The arithmetic of :meth:`_attempt`, :meth:`_correct`, :meth:`_commit` and
        :meth:`_next_order` for this family, in the engine's passes
        (:meth:`HostHistory.fused_trial`): the defects of the neighbouring orders are
        scored in the same pass as the step's own, before the decision.

        Parameters
        ----------
        dz : float
            The step, m.
        fixed : bool, optional
            Whether the step is a prescribed schedule's, accepted whatever its norm.
        landing : tuple or None, optional
            Where an accepted step ends and whether it is a hard kink
            (:meth:`_attempt`).

        Returns
        -------
        tuple
            Whether the step was accepted, and its scaled error norm.
        """
        z, nodes, slots, order, orders = self._stencil(dz)
        self.ledger.predictor += 1
        norms = self._engine.fused_trial(z, dz, nodes, slots, order, orders)
        return self._decide(dz, order, orders, norms, fixed=fixed, landing=landing)

    def _stencil(
        self, dz: float
    ) -> tuple[float, list[float], np.ndarray, int, list[int]]:
        """Return a trial's anchor, nodes, rows, order and scored orders.

        Each order's defect is its ``k + 1``-node corrector through the new source less
        its ``k``-node predictor; a neighbour is scored when the history can form it,
        and the order never rises straight after a rejection.

        Parameters
        ----------
        dz : float
            The step, m.

        Returns
        -------
        tuple
            The anchor, m; the history's normalised positions and store rows, newest
            first; the step's order; and the orders scored, its own first.
        """
        method = self._method
        z = self._backend._z
        recent = self._history[::-1]
        nodes = [(entry.z - z) / dz for entry in recent]
        slots = np.array([entry.slot for entry in recent], dtype=np.int64)
        held = len(recent)
        order = min(self._order, held)
        rising = not self._last_rejected and held > order
        orders = [order] + (
            [
                other
                for other in (order - 1, order + 1)
                if 1 <= other <= method.order
                and other <= held
                and (other < order or rising)
            ]
            if method.variable_order
            else []
        )
        return z, nodes, slots, order, orders

    def _decide(
        self,
        dz: float,
        order: int,
        orders: list[int],
        norms: np.ndarray,
        *,
        fixed: bool,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Reject a held trial, or accept it and choose the next order and step.

        The new sample takes the oldest row past the method's need.

        Parameters
        ----------
        dz : float
            The step, m.
        order : int
            Its order.
        orders : list of int
            The orders scored, its own first.
        norms : numpy.ndarray
            Their scaled norms.
        fixed : bool
            Whether the step is a prescribed schedule's, accepted whatever its norm.
        landing : tuple or None, optional
            Where an accepted step ends and whether it is a hard kink
            (:meth:`_attempt`).

        Returns
        -------
        tuple
            Whether the step was accepted, and its scaled error norm.
        """
        backend, method, engine = self._backend, self._method, self._engine
        norm = float(norms[0])
        if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
            self.ledger.rejected += 1
            self._reject(dz, norm, order)
            return False, norm
        z = backend._z
        keep = self._history[-method.order :] if method.order else []
        used = {entry.slot for entry in keep}
        free = next(slot for slot in range(engine.n_slots) if slot not in used)
        rows = np.array([entry.slot for entry in keep], dtype=np.int64)
        with committing():
            if self._exponential is None:
                engine.fused_accept(rows, free, interpolate=self._interpolate)
            else:
                engine.exponential_accept(rows, free, interpolate=self._interpolate)
            self._history = [
                *keep,
                _Derivative(z + dz, engine.value(free), "predicted", free),
            ]
            self.ledger.accepted_steps += 1
            self._rejections = 0
            self.ledger.orders.append(order)
            next_order, control = self._choose_order(
                order, dict(zip(orders, norms.tolist(), strict=True)), dz
            )
            backend._proposal = limit_step(
                dz,
                _factor(method, control, next_order, growth=method.max_growth),
                backend._problem.stepper.dz_min,
                backend._problem.stepper.dz_max,
            )
            self._last_rejected = False
            self._land(z, dz, landing)
        return True, norm

    def _land(self, z: float, dz: float, landing: tuple[float, bool] | None) -> None:
        """Move the backend to an accepted step's end, restarting at a kink.

        Called inside the commit that installed the step's state, so that the state
        and its position change together whatever interrupts the march afterwards.

        Parameters
        ----------
        z : float
            The step's start, m.
        dz : float
            The step, m.
        landing : tuple or None
            Where it ends and whether that is a hard kink (:meth:`_attempt`).
        """
        end, kink = (z + dz, False) if landing is None else landing
        self._backend._z = end
        if kink:
            self._history = []
            self.ledger.restarts += 1

    def _attempt_exponential(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one exponential Adams step and commit it if accepted (SEP 16).

        The controller of :meth:`_attempt_fused` with the engine's exponential trial
        (:meth:`HostHistory.exponential_trial`): untransported samples, integrated
        exactly against the step's flow; an accepted step re-anchors nothing.

        Parameters
        ----------
        dz : float
            The step, m.
        fixed : bool, optional
            Whether the step is a prescribed schedule's, accepted whatever its norm.
        landing : tuple or None, optional
            Where an accepted step ends and whether it is a hard kink
            (:meth:`_attempt`).

        Returns
        -------
        tuple
            Whether the step was accepted, and its scaled error norm.
        """
        z, nodes, slots, order, orders = self._stencil(dz)
        self.ledger.predictor += 1
        norms = self._engine.exponential_trial(z, dz, nodes, slots, order, orders)
        return self._decide(dz, order, orders, norms, fixed=fixed, landing=landing)

    def _correct(
        self, trial: _Trial, source: _State
    ) -> tuple[_State, _State, tuple[np.ndarray, list[_State]]]:
        """Return the accepted increment and its local-defect estimate.

        Parameters
        ----------
        trial : _Trial
            The step being tried.
        source : numpy.ndarray
            The source at the predicted field, in the anchor's frame.

        Returns
        -------
        tuple
            The increment ``v - a_n`` the step would accept and the estimate of its
            local defect, both in the anchor's frame, and the nodes and derivatives of
            the formula that produced the increment, for dense observation.
        """
        order, nodes, values, dz = trial.order, trial.nodes, trial.values, trial.dz
        if self._method.corrector == "none":
            # Explicit Adams-Bashforth keeps its predictor; the stencil that ends at the
            # new derivative rechecks it after the call (SEP 13 section 4.3).
            check = np.concatenate(([1.0], nodes[:order]))
            corrected = _combine(
                integrated_weights(check, check=False, helpers=self._backend.helpers),
                [source, *values[:order]],
                dz,
            )
            predictor = (nodes[:order], values[:order])
            return trial.predictor, corrected - trial.predictor, predictor
        count = self._method.corrector_nodes(order)
        corrector_nodes = np.concatenate(([1.0], nodes[: count - 1]))
        corrected = _combine(
            integrated_weights(
                corrector_nodes, check=False, helpers=self._backend.helpers
            ),
            [source, *values[: count - 1]],
            dz,
        )
        difference = corrected - trial.predictor
        formula = (corrector_nodes, [source, *values[: count - 1]])
        if count == order:
            factor = milne_factor(nodes[:order], corrector_nodes)
            return corrected, factor * difference, formula
        # A corrector one order higher: the difference is the predictor's defect, a
        # conservative estimate of the accepted corrector's.
        return corrected, difference, formula

    def _commit(
        self,
        trial: _Trial,
        advanced: _State,
        source: _State,
        norm: float,
        formula: tuple[np.ndarray, list[_State]],
        landing: tuple[float, bool] | None = None,
    ) -> None:
        """Adopt an accepted step: the state, the re-anchored history and the proposal.

        Parameters
        ----------
        trial : _Trial
            The accepted step.
        advanced : numpy.ndarray
            The state at its end, before the boundary treatments.
        source : numpy.ndarray
            The source at the predicted field, in the step's anchor frame.
        norm : float
            The step's scaled error norm.
        formula : tuple
            The nodes and anchor-frame derivatives of the accepted formula.
        landing : tuple or None, optional
            Where the step ends and whether it is a hard kink (:meth:`_attempt`).
        """
        backend = self._backend
        method = self._method
        settings = backend._problem.stepper
        z, dz, propagator = trial.z, trial.dz, trial.propagator
        # The comparison variants' corrected evaluation is inside the commit too.
        with committing():
            finished = backend._adams_adopt(advanced, dz)
            if method.corrector == "pece":
                self.ledger.corrected += 1
                try:
                    stored, _ = self._source(finished, z, z + dz, purpose="corrected")
                except BaseException:
                    # Adoption installs the candidate before its corrected response. A
                    # failed response must leave the checkpoint at the previous state.
                    backend._adams_restore(z, trial.state)
                    raise
                identity = "accepted"
            else:
                stored = source
                identity = "predicted" if method.corrector == "pec" else "accepted"
            for entry in self._history:
                entry.value = propagator * cast("_State", entry.value)
            self._history.append(_Derivative(z + dz, propagator * stored, identity))
            self._history = self._history[-(method.order + 1) :]
            if self._interpolate:
                self._engine.dense = _DenseStep(
                    z=z,
                    dz=dz,
                    start=trial.state,
                    end=finished / propagator,
                    start_slope=trial.values[0],
                    end_slope=stored,
                    nodes=formula[0],
                    slopes=formula[1],
                )
            self.ledger.accepted_steps += 1
            self._rejections = 0
            self.ledger.orders.append(trial.order)
            order, norm = self._next_order(trial, source, norm)
            backend._proposal = limit_step(
                dz,
                _factor(method, norm, order, growth=method.max_growth),
                settings.dz_min,
                settings.dz_max,
            )
            self._last_rejected = False
            self._land(z, dz, landing)

    def _next_order(
        self, trial: _Trial, source: _State, norm: float
    ) -> tuple[int, float]:
        """Choose the next step's order from neighbouring estimates; no extra call.

        For the one-order-higher PEC/PECE family the defect of order ``k`` is the
        difference between the ``k + 1``-derivative corrector through the new source
        and the ``k``-derivative predictor, so orders ``q - 1`` and ``q + 1`` are
        estimated from the same history and source as the step itself (SEP 13 section
        4.4). The order maximising the predicted step wins if it beats the current one
        by :data:`ORDER_HYSTERESIS`; the order never rises straight after a rejection.

        Parameters
        ----------
        trial : _Trial
            The accepted step.
        source : numpy.ndarray
            Its predicted-state source, in the step's anchor frame.
        norm : float
            Its scaled defect at its own order.

        Returns
        -------
        tuple
            The next order and the norm it is controlled with.
        """
        method = self._method
        current = trial.order
        if not method.variable_order:
            self._order = method.order
            return current, norm
        # A neighbour is scored when the history can form it; the order never rises
        # straight after a rejection, nor beyond the derivatives the history holds.
        rising = not self._last_rejected and len(trial.values) > current
        scores = {current: norm} | {
            order: _defect_at(trial, source, order, self._backend.helpers)
            for order in (current - 1, current + 1)
            if 1 <= order <= method.order
            and order <= len(trial.values)
            and (order < current or rising)
        }
        return self._choose_order(current, scores, trial.dz)

    def _stable_factor(self, order: int, dz: float) -> float:
        """Return the largest factor on `dz` an in-stage collar leaves an order stable.

        Parameters
        ----------
        order : int
            The predictor order.
        dz : float
            The step, m.

        Returns
        -------
        float
            ``COLLAR_STABILITY * PEC_REAL_INTERVAL[order - 1] / (dz max(alpha) / 2)``,
            infinite without an in-stage collar.
        """
        if self._collar_rate <= 0.0:
            return math.inf
        interval = COLLAR_STABILITY * PEC_REAL_INTERVAL[order - 1]
        return interval / (self._collar_rate * dz)

    def _order_weight(self, scores: dict[int, float], dz: float, order: int) -> float:
        """Return an order's weight: its defect's step factor, capped for stability.

        Parameters
        ----------
        scores : dict
            The scaled defect of every scored order.
        dz : float
            The accepted step, m.
        order : int
            One of the scored orders.

        Returns
        -------
        float
            ``safety norm ** (-1 / (order + 1))``, at most :meth:`_stable_factor`.
        """
        allowed = self._method.safety * _order_rate(scores, order)
        return min(allowed, self._stable_factor(order, dz))

    def _choose_order(
        self, current: int, scores: dict[int, float], dz: float
    ) -> tuple[int, float]:
        """Adopt the scored order with the largest predicted step, with hysteresis.

        In the exponential formulation each order is weighed by the step its defect
        allows capped at the step an in-stage collar leaves it stable at: near that
        limit the higher orders' defects carry the collar's growing mode, and a choice
        on them alone was decided by rounding -- runs a rounding apart settled in
        different orders, with calls 40 % apart. The cap weighs the choice only; the
        step is the chosen order's defect's, as before.

        Parameters
        ----------
        current : int
            The step's own order, which `scores` contains.
        scores : dict
            The scaled defect of every scored order.
        dz : float
            The accepted step, m.

        Returns
        -------
        tuple
            The next order and the norm it is controlled with.
        """
        if not self._method.variable_order:
            self._order = self._method.order
            return current, scores[current]

        rate = functools.partial(self._order_weight, scores, dz)
        best = max(scores, key=rate)
        if best != current and rate(best) > ORDER_HYSTERESIS * rate(current):
            self._order = best
            return best, scores[best]
        self._order = current
        return current, scores[current]

    def _reject(self, dz: float, norm: float, order: int) -> None:
        """Shrink the proposal after a rejected step; restart a failing history.

        A rejection leaves the accepted history untouched. Repeated rejections mean the
        history itself is the problem -- a parasitic mode of the multistep recurrence
        that a smaller step cannot remove, because every stencil still interpolates it
        -- so after :data:`RESTART_AFTER` in a row the method restarts at order one from
        the accepted state, paying one startup call.

        Parameters
        ----------
        dz : float
            The rejected step, m.
        norm : float
            Its scaled defect, possibly not finite.
        order : int
            Its order.
        """
        backend = self._backend
        settings = backend._problem.stepper
        self.ledger.rejected_steps += 1
        self._rejections += 1
        self._last_rejected = True
        if self._rejections >= RESTART_AFTER and len(self._history) > 1:
            self._history = []
            self._rejections = 0
            self.ledger.failure_restarts += 1
        factor = (
            self._method.min_factor
            if not math.isfinite(norm)
            else _factor(self._method, norm, order, growth=0.9)
        )
        backend._proposal = limit_step(dz, factor, settings.dz_min, settings.dz_max)


def _order_rate(scores: dict[int, float], order: int) -> float:
    """Return the step-size factor an order's defect would allow.

    A module function rather than a closure in :meth:`AdamsMarch._choose_order`, whose
    branch into a nested definition Python 3.14's coverage does not record.

    Parameters
    ----------
    scores : dict
        The scaled defect of every scored order.
    order : int
        One of the scored orders.

    Returns
    -------
    float
        ``norm ** (-1 / (order + 1))``, infinite for a zero defect.
    """
    value = scores[order]
    return math.inf if value <= 0.0 else value ** (-1.0 / (order + 1))


def _defect_rows(
    kernels: AdamsKernels, nodes: list[float], orders: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return each order's defect weights and the first order's corrector weights.

    Over ``(source, newest, ...)``, order ``k``'s defect is its ``k + 1``-node
    corrector through the new source less its ``k``-node predictor.

    Parameters
    ----------
    kernels : AdamsKernels
        The fused path's arithmetic, for its weights.
    nodes : list of float
        The history's normalised positions, newest first.
    orders : list of int
        The orders to score, the step's own first.

    Returns
    -------
    tuple
        The defect rows, shape ``(len(orders), 1 + max(orders))``, and the step's own
        corrector weights, zero past its nodes.
    """
    width = 1 + max(orders)
    rows = np.zeros((len(orders), width))
    corrector = np.zeros(width)
    for index, order in enumerate(orders):
        stencil = nodes[:order]
        through = kernels.weights([1.0, *stencil])
        rows[index, : order + 1] = through
        rows[index, 1 : order + 1] -= kernels.weights(stencil)
        if index == 0:
            corrector[: order + 1] = through
    return rows, corrector


def fused_tables(
    nodes: list[float], order: int, orders: list[int], *, helpers: Helpers
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a fused trial's coefficient tables, for an engine that applies them.

    The interaction formulation's tables of :meth:`HostHistory.fused_trial`, which a
    resident engine (ADR-043) receives instead of forming them.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers
        The engine's backend's implementation of the Adams basis.

    Returns
    -------
    tuple
        The predictor's integrated weights over the ``order`` newest rows; the
        corrector's weights over ``(source, newest, ...)``, zero past its nodes; and
        one defect row per scored order over the same samples.
    """
    kernels = AdamsKernels(helpers=helpers)
    rows, corrector = _defect_rows(kernels, nodes, orders)
    return kernels.weights(nodes[:order]), corrector, rows


def factored_defect_rows(
    nodes: list[float], orders: list[int], width: int
) -> np.ndarray:
    """Return each exponential defect factored: a divided difference and a polynomial.

    Order ``k``'s defect, its ``k + 1``-node corrector through the new sample less its
    ``k``-node predictor, vanishes at the predictor's nodes, so it is
    ``f[1, theta_0, ..., theta_{k-1}] prod_{j<k} (theta - theta_j)``: one divided
    difference of the samples times one polynomial, whose integral against the flow is
    one combination of the moments. That is ``2 (k + 1)`` products an element instead
    of ``(k + 1) n_k``, the monomial rows of ``_exponential_rows``.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    orders : list of int
        The orders scored, the step's own first.
    width : int
        ``1 + max(orders)``: the samples and the moments of the trial.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(orders), 2 width)``: the divided-difference coefficients over
        ``(source, newest, ...)``, then the node polynomial's ascending monomial
        coefficients.
    """
    rows = np.zeros((len(orders), 2 * width))
    for index, order in enumerate(orders):
        points = [1.0, *nodes[:order]]
        for t, node in enumerate(points):
            denominator = 1.0
            for s, other in enumerate(points):
                if s != t:
                    denominator *= node - other
            rows[index, t] = 1.0 / denominator
        rows[index, width : width + order + 1] = np.polynomial.polynomial.polyfromroots(
            nodes[:order]
        )
    return rows


def exponential_tables(
    nodes: list[float], order: int, orders: list[int], *, helpers: Helpers
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exponential trial's coefficient tables, for a resident engine.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers
        The engine's backend's implementation of the Adams basis.

    Returns
    -------
    tuple
        The predictor's Lagrange coefficients over the ``order`` newest rows, shape
        ``(order, order)``; the corrector's over ``(source, newest, ...)``, shape
        ``(width, width)`` with ``width = 1 + max(orders)``; and the factored defect
        rows of :func:`factored_defect_rows`.
    """
    width = 1 + max(orders)
    _, corrector = _exponential_rows(nodes, orders, width, helpers=helpers)
    predictor = lagrange_coefficients(nodes[:order], check=False, helpers=helpers)
    return predictor, corrector, factored_defect_rows(nodes, orders, width)


def compensation_tables(
    nodes: list[float], order: int, orders: list[int], *, helpers: Helpers
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a trial's transported-compensation tables, for a resident engine.

    The weights ``_Compensation`` applies (SEP 16 section 4.2, ADR-042), over the
    carried rows ``exp((1 - theta_j) x) (D(z_j) - x / h) u_j`` of the history samples
    and the endpoint term ``(D(z + h) - x / h) u_pred``, each sum still to be
    multiplied by the step.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers
        The engine's backend's implementation of the Adams basis.

    Returns
    -------
    tuple
        The spans ``1 - theta_j`` of the ``width - 1`` carried samples; the
        predictor's weights over the ``order`` newest of them; the corrector's over
        ``(endpoint, newest, ...)``; and each scored order's defect weights over the
        same, shape ``(len(orders), width)``, with ``width = 1 + max(orders)``.
    """
    width = 1 + max(orders)
    spans = np.array([1.0 - node for node in nodes[: width - 1]])
    predictor = integrated_weights(
        np.asarray(nodes[:order]), check=False, helpers=helpers
    )
    corrector = integrated_weights(
        np.asarray([1.0, *nodes[:order]]), check=False, helpers=helpers
    )
    rows = np.zeros((len(orders), width))
    for index, scored in enumerate(orders):
        rows[index, : scored + 1] = integrated_weights(
            np.asarray([1.0, *nodes[:scored]]), check=False, helpers=helpers
        )
        rows[index, 1 : scored + 1] -= integrated_weights(
            np.asarray(nodes[:scored]), check=False, helpers=helpers
        )
    return spans, predictor, corrector, rows


def _exponential_rows(
    nodes: list[float], orders: list[int], width: int, *, helpers: Helpers
) -> tuple[np.ndarray, np.ndarray]:
    """Return each order's defect coefficients and the step's own corrector's.

    Over ``X = (source, newest, ...)`` in the monomial basis of the moments, order
    ``k``'s defect is its ``k + 1``-node corrector through the new source less its
    ``k``-node predictor, as for :func:`_defect_rows`; the weights are the Lagrange
    coefficients, which the trial's moments turn into exponential weights.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    orders : list of int
        The orders to score, the step's own first.
    width : int
        ``1 + max(orders)``: the samples and the moments the trial forms.
    helpers : solprop._helpers.Helpers
        The owning backend's implementation of the Adams basis.

    Returns
    -------
    tuple
        The defect rows, shape ``(len(orders), width, width)``, and the step's own
        corrector, shape ``(width, width)``, zero past its nodes.
    """
    rows = np.zeros((len(orders), width, width))
    corrector = np.zeros((width, width))
    for index, order in enumerate(orders):
        stencil = nodes[:order]
        through = lagrange_coefficients([1.0, *stencil], check=False, helpers=helpers)
        rows[index, : order + 1, : order + 1] = through
        rows[index, 1 : order + 1, :order] -= lagrange_coefficients(
            stencil, check=False, helpers=helpers
        )
        if index == 0:
            corrector[: order + 1, : order + 1] = through
    return rows, corrector


def _defect_at(trial: _Trial, source: _State, order: int, helpers: Helpers) -> float:
    """Return the scaled defect an accepted step would have had at another order.

    The order's predictor and its one-order-higher corrector are formed from the same
    history and the same predicted-state source, so scoring a neighbour costs no call.

    Parameters
    ----------
    trial : _Trial
        The accepted step.
    source : numpy.ndarray
        Its predicted-state source, in the step's anchor frame.
    order : int
        The neighbouring order, no more than the derivatives the trial holds.
    helpers : solprop._helpers.Helpers
        The owning backend's implementation of the Adams basis.

    Returns
    -------
    float
        The scaled predictor-corrector difference at that order.
    """
    nodes, values = trial.nodes[:order], trial.values[:order]
    predicted = _combine(
        integrated_weights(nodes, check=False, helpers=helpers), values, trial.dz
    )
    corrected = _combine(
        integrated_weights(
            np.concatenate(([1.0], nodes)), check=False, helpers=helpers
        ),
        [source, *values],
        trial.dz,
    )
    return trial.norm(corrected - predicted)


def _factor(method: Multistep, norm: float, order: int, *, growth: float) -> float:
    """Return the elementary step-size factor of an order-`order` local defect.

    The defect scales as ``h^{order+1}`` (SEP 13 section 4.4), so the exponent is
    ``1/(order + 1)``, not the Runge-Kutta pair's fixed one.

    Parameters
    ----------
    method : solprop.stepper.multistep.Multistep
        The method, for its safety and smallest factor.
    norm : float
        The scaled defect.
    order : int
        The order the defect belongs to.
    growth : float
        The largest factor allowed.

    Returns
    -------
    float
        The factor on the step.
    """
    if norm <= 0.0:
        return growth
    raw = method.safety * norm ** (-1.0 / (order + 1))
    return min(growth, max(method.min_factor, raw))


def _combine(weights: np.ndarray, values: list[_State], dz: float) -> _State:
    """Return ``dz * sum_j w_j g_j``.

    Parameters
    ----------
    weights : numpy.ndarray
        The integrated weights.
    values : list
        The derivatives, one per weight.
    dz : float
        The step, m.

    Returns
    -------
    numpy.ndarray
        The increment.
    """
    total = np.zeros_like(values[0])
    for weight, value in zip(weights, values, strict=True):
        total = passes.accumulate(total, value, float(weight))
    return dz * total


def _norm_function(
    backend: StepDriver,
    state: _State,
) -> Callable[[_State], float]:
    """Return the weak norm of a defect, scaled by the accepted state before the step.

    The Runge-Kutta march scales by the larger of the states before and after the step.
    An Adams step outside its stability region can multiply the state many times over
    while its predictor and corrector still agree to a fraction of the *new* state, so
    that scale lets a divergence pass as accurate: SEP 13 phase 2 measured PEC4 growing
    from 1e7 to 1e236 in five accepted steps with every scaled norm below 0.6. The state
    the step starts from is the scale that cannot grow with the step it judges.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend, for the controller and the in-band mask.
    state : numpy.ndarray
        The accepted state before the step.

    Returns
    -------
    callable
        ``defect -> norm`` for a defect carried to the end of the step.
    """
    problem = backend._problem
    mask, controller = problem.grids.norm_mask, problem.stepper.controller
    # The scale is the state's, the same for every defect of the step, so it is
    # reduced once: :func:`passes.error_norm`'s arithmetic, with the two state norms
    # it would take per call hoisted out.
    magnitude = float(np.linalg.norm(state[..., mask]))
    scale = controller.scale(magnitude, magnitude)
    divisor = scale[mask] if isinstance(scale, np.ndarray) else scale

    def scaled(defect: _State) -> float:
        """Return the scaled norm of a defect at the end of the step.

        Parameters
        ----------
        defect : numpy.ndarray
            The defect, carried to the end of the step.

        Returns
        -------
        float
            The scaled norm.
        """
        return float(np.linalg.norm(defect[..., mask] / divisor))

    return scaled


def _next_landing(z: float, targets: StepTargets, target: float) -> float | None:
    """Return the next forced landing after `z`: save plane, hard kink or end.

    Parameters
    ----------
    z : float
        The current position, m.
    targets : solprop.stepper.schedule.StepTargets
        The planes and kinks.
    target : float
        The end of the current march, m.

    Returns
    -------
    float or None
        The nearest landing ahead, or ``None`` when none lies ahead.
    """
    tolerance = 8 * float(np.spacing(max(abs(z), abs(target))))
    candidates = [target]
    for positions in (targets.save_planes, targets.hard_kinks):
        ahead = positions[positions > z + tolerance]
        if ahead.size:
            candidates.append(float(ahead[0]))
    nearest = min(candidates)
    return nearest if nearest > z + tolerance else None
