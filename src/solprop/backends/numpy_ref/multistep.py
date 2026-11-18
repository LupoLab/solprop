# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's Adams engine: the history and its trials (SEP 13, SEP 14, SEP 16).

One accepted step of the PEC method costs one expensive source evaluation, against
DOPRI5's seven (six with first-same-as-last, which the reference does not take). The
engine keeps a short history of source samples and decides each step on a local-defect
estimate formed from that history without another evaluation (SEP 13 sections 4.1--4.3);
a trial touches no row, so a rejection leaves the history as it was.
:class:`HostHistory` is the engine the march
(:class:`solprop.backends.numpy_ref.marches.adams.HostAdamsMarch`) drives, with its two
formulations: :class:`InteractionHistory` keeps **transported** sources, re-anchored at
every accepted step by the step's propagator, and :class:`ExponentialHistory` keeps
untransported ones and integrates them exactly against the step's flow, with a coupled
flow's frozen block, a varying operator's compensation and the carrier frame (SEP 35).
:class:`AdamsRows` is where the rows live, which the session exports and installs. The
field arithmetic is :mod:`~solprop.backends.numpy_ref.adams_kernels`', the coefficients
:mod:`~solprop.backends.numpy_ref.adams_tables`'.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop._helpers import PLAIN
from solprop.backends.adams_core import Stencil
from solprop.backends.engines import (
    AcceptRecord,
    AttemptOutcome,
    BoundaryReading,
    HistoryEntry,
)
from solprop.backends.exponential import CarrierReference, carrier_reference
from solprop.backends.ledger import source_absent
from solprop.backends.numpy_ref.adams_kernels import (
    AdamsKernels,
    CoupledBlock,
    ExponentialKernels,
    coupled_moments,
)
from solprop.backends.numpy_ref.adams_tables import defect_rows, exponential_corrector
from solprop.exceptions import BackendError
from solprop.stepper.multistep import (
    Multistep,
    integrated_weights,
    lagrange_coefficients,
)
from solprop.translate.linop import phi_rate

if TYPE_CHECKING:
    from solprop.backends.numpy_ref.workspace import NumpyWorkspace
    from solprop.stepper.controller import PIController

__all__ = ["AdamsRows", "ExponentialHistory", "HostHistory", "InteractionHistory"]

logger = logging.getLogger(__name__)

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]
_Shift = Complex[np.ndarray, "1 1 #n_omega"]
"""A carrier frame's shift (SEP 35), shape ``(1, 1, 1)``."""
_Factors = Complex[np.ndarray, "n_used 1 1 #n_omega"]
"""A carrier frame's rotation of each history row a trial reads, in its slots' order."""


@dataclass
class _DenseStep:
    """The last accepted step, in its anchor's frame, for dense observation.

    Its slopes, the new source first; the observation's weights are the session's, from
    the march's record of the step
    (:class:`solprop.backends.numpy_ref.marches.adams.AcceptedStep`).
    """

    z: float
    dz: float
    start: _State
    slopes: list[_State]


@dataclass
class _ExponentialStep:
    """The last accepted exponential step, for dense observation.

    Its samples as stored, the new one first; the observation's coefficients and, under
    a carrier frame, its rotations are the session's, from the march's record of the
    step (:class:`solprop.backends.numpy_ref.marches.adams.AcceptedStep`).
    """

    z: float
    dz: float
    start: _State
    exponent: _State
    values: list[_State]
    compensation: tuple[list[float], list[_State]] | None = None


@dataclass
class _ExponentialTrial:
    """An exponential trial held between its norms and its acceptance."""

    z: float
    dz: float
    state: _State
    exponent: _State
    source: _State
    lab: _State
    advanced: _State
    slots: np.ndarray
    order: int
    compensation: "_Compensation | None"


@dataclass
class _FusedTrial:
    """An interaction-formulation trial held between its norms and its acceptance."""

    z: float
    dz: float
    state: _State
    propagator: _State
    source: _State
    advanced: _State
    slots: np.ndarray
    order: int


@dataclass
class _Compensation:
    """One trial's linear compensation ``C_j = (D(z_j) - B_n) u_j`` (SEP 16 4.2).

    The transported treatment integrates it in the reference flow's frame,
    ``h sum_j a_j exp((1 - theta_j) h B_n) C_j`` with the ordinary Adams weights
    ``a_j``; every factor is a forward flow over at least the step. The carried rows
    are formed once per trial (:meth:`ExponentialKernels.carry
    <solprop.backends.numpy_ref.adams_kernels.ExponentialKernels.carry>`) and every
    formula is one weighted sum over them. The untransported treatment, a test witness
    only (Q8), interpolates ``C_j`` with the source's exponential weights instead.
    """

    exponent: _State
    moments: Any
    """The trial's moments, read by the untransported witness only."""
    nodes: list[float]
    carried: Complex[np.ndarray, "n_used n_modes n_pol n_omega"]
    dz: float
    treatment: str
    endpoint_rate: _State | None = None
    endpoint: _State | None = None
    _weights: dict[tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def add_endpoint(
        self, rate: _State, state: _State, residual: _State | None = None
    ) -> None:
        """Add the new sample's compensation at the predicted state; no transport.

        - ``rate``: ``D(z_n + h)`` from the step's side of the end.
        - ``residual``: a frozen-residual flow's ``R_s(z_n + h)`` applied to `state`.
        """
        self.endpoint_rate = rate
        self.endpoint = (rate - self.exponent / self.dz) * state
        if residual is not None:
            self.endpoint = self.endpoint + residual

    def rotate(self, shift: _Shift) -> None:
        """Rotate the untransported witness's terms by a trial's carrier shift.

        The transported treatment's formula is unchanged under a carrier frame (SEP 35
        section 4.5): its terms are carried by the reference flow and integrated with
        the ordinary Adams weights. The untransported witness integrates its terms
        against the trial's moments, which are the shifted ones, so its history terms
        take the samples' factors ``exp(sigma (1 - theta_j))``; the endpoint's is one.
        """
        if self.treatment == "transported":
            return
        self.carried = np.stack(
            [
                np.exp(shift * (1.0 - node)) * row
                for node, row in zip(self.nodes, self.carried, strict=True)
            ]
        )

    def _sum(self, nodes: list[float], first: _State | None, count: int) -> _State:
        """Return ``h`` times the treatment's integral over the given samples.

        - ``nodes``: the formula's nodes, the endpoint's first when `first` is given.
        - ``first``: the endpoint sample's compensation, or ``None`` for a predictor.
        """
        rows = self.carried[:count]
        if self.treatment == "transported":
            key = tuple(nodes)
            weights = self._weights.get(key)
            if weights is None:
                weights = integrated_weights(
                    np.asarray(nodes), check=False, helpers=PLAIN
                )
                self._weights[key] = weights
            total = np.tensordot(weights[len(nodes) - count :], rows, axes=1)
            if first is not None:
                total = total + weights[0] * first
            return self.dz * total
        values = ([] if first is None else [first]) + list(rows)
        return ExponentialKernels().apply(
            self.moments[: len(nodes)],
            lagrange_coefficients(nodes, check=False, helpers=PLAIN),
            values,
            self.dz,
        )

    def predictor(self, order: int) -> _State:
        """Return the predictor's compensation over the ``order`` newest samples."""
        return self._sum(self.nodes[:order], None, order)

    def corrector(self, order: int) -> _State:
        """Return the corrector's, through the endpoint sample."""
        return self._sum([1.0, *self.nodes[:order]], self.endpoint, order)

    def defects(self, orders: list[int]) -> list[_State]:
        """Return each scored order's corrector less predictor compensation."""
        return [self.corrector(k) - self.predictor(k) for k in orders]

    def observation(
        self, order: int, history: list[_State]
    ) -> tuple[list[float], list[_State]]:
        """Return the accepted corrector's nodes and untransported compensations.

        Dense observation applies its own flow factors ``exp((xi - theta_j) x)``;
        the history's terms are formed again from the stored rows by the caller,
        rather than carried back by an inverse flow that a damped bin would overflow.

        - Returns: the nodes ``(1, theta_0, ...)`` and the matching compensations.
        """
        assert self.endpoint is not None
        return [1.0, *self.nodes[:order]], [self.endpoint.copy(), *history]


class AdamsRows:
    """The Adams history's rows, where the state lives (SEP 52 section 5.2).

    The samples a march's entries name, and for a compensated history each sample's
    state, rate and frozen-residual weights: what the engine
    (:class:`HostHistory`) writes at a startup and an accepted step, and what the
    session exports into a snapshot and installs from one, as native's session reads
    its resident rows.

    Parameters
    ----------
    n_slots : int
        The rows: the method's order and two more.
    compensated : bool
        Whether each sample's state, rate and weights are kept too.
    name : str, optional
        The backend's name, for messages.

    Attributes
    ----------
    n_slots : int
        As given.
    compensated : bool
        As given.
    name : str
        As given.
    store : numpy.ndarray or None
        The samples, shape ``(n_slots, n_modes, n_pol, n_omega)``; ``None`` before the
        first startup.
    states, rates : numpy.ndarray or None
        A compensated history's per-sample states and rates, shaped as `store`.
    weights : list
        A frozen-residual flow's weights per sample, ``None`` where a sample has none.
    """

    def __init__(
        self, n_slots: int, compensated: bool, *, name: str = "numpy_ref"
    ) -> None:
        """Hold no rows yet."""
        self.n_slots = n_slots
        self.compensated = compensated
        self.name = name
        self.store: np.ndarray | None = None
        self.states: np.ndarray | None = None
        self.rates: np.ndarray | None = None
        self.weights: list[np.ndarray | None] = []

    def nbytes(self) -> int:
        """Return the bytes :meth:`export` would copy, copying nothing.

        Returns
        -------
        int
            The size of every row.
        """
        assert self.store is not None
        rows = [self.store, *self.weights]
        if self.compensated:
            assert self.states is not None and self.rates is not None
            rows += [self.states, self.rates]
        return sum(int(row.nbytes) for row in rows if row is not None)

    def export(self) -> dict[str, np.ndarray]:
        """Return copies of the rows a resume needs to continue the history.

        Returns
        -------
        dict
            ``"store"``, the samples; for a compensated march also ``"states"`` and
            ``"rates"``, and ``"weights_<row>"`` for each row a frozen-residual flow
            weights.
        """
        assert self.store is not None
        rows = {"store": self.store.copy()}
        if self.compensated:
            assert self.states is not None and self.rates is not None
            rows["states"] = self.states.copy()
            rows["rates"] = self.rates.copy()
        for slot, weights in enumerate(self.weights):
            if weights is not None:
                rows[f"weights_{slot}"] = np.array(weights, copy=True)
        return rows

    def install(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows :meth:`export` returned, for a state of `shape`.

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
            name = self.name
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
        self.store = np.array(rows["store"], dtype=np.complex128, copy=True)
        if self.compensated:
            self.states = np.array(rows["states"], dtype=np.complex128, copy=True)
            self.rates = np.array(rows["rates"], dtype=np.complex128, copy=True)
        self.weights = [
            None
            if f"weights_{slot}" not in rows
            else np.array(rows[f"weights_{slot}"], copy=True)
            for slot in range(self.n_slots)
        ]


class HostHistory:
    """The Adams history and its field arithmetic, on host arrays (SEP 16 phase 3b).

    The reference's :class:`~solprop.backends.engines.AdamsEngine` (native's
    ``march::Engine``, SEP 52 section 5.2). The march
    (:class:`~solprop.backends.numpy_ref.marches.adams.HostAdamsMarch`) keeps the
    controller -- the orders, the entries, rejection, restarts, the budget and the
    ledger -- and hands every state-sized operation to this engine over the workspace:
    the startup sample, a trial's predictor, source, corrector and defect norms, and the
    adoption of an accepted step. The engine writes the history's rows into
    :class:`AdamsRows`, which the session exports and installs; between a trial and its
    acceptance it holds the trial, and an interpolated run's accepted step keeps what
    its dense observation reads (``dense``; :mod:`solprop.backends.numpy_ref.dense`).

    This base holds the lifecycle the two formulations share; each formulation is a
    subclass, :class:`InteractionHistory` or :class:`ExponentialHistory`, which forms
    the startup sample, the trial and the acceptance.

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state, source evaluation and propagators are used.
    method : solprop.stepper.multistep.Multistep
        The method.
    compensated : bool
        Whether the rows keep each sample's state, rate and flow weights.
    interpolate : bool, optional
        Whether an accepted step keeps its interpolant for dense observation.
    rows : AdamsRows or None, optional
        The rows to write; new ones by default.

    Attributes
    ----------
    rows : AdamsRows
        The history's rows.
    dense : object
        The last accepted step, for dense observation; ``None`` before one.
    carrier : solprop.backends.exponential.CarrierReference or None
        The exponential formulation's carrier frame; ``None`` without one.
    """

    def __init__(
        self,
        workspace: "NumpyWorkspace",
        method: Multistep,
        *,
        compensated: bool,
        interpolate: bool = False,
        rows: AdamsRows | None = None,
    ) -> None:
        """Hold the workspace and an empty store."""
        self._workspace = workspace
        self._method = method
        self._interpolate = interpolate
        self._controller = workspace.stepper.controller
        self.rows = (
            AdamsRows(method.order + 2, compensated, name=workspace.name)
            if rows is None
            else rows
        )
        self.dense: Any = None
        self.carrier: CarrierReference | None = None
        # The last trial's carrier shift and row rotations (SEP 35); None without.
        self._carrier_rows: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def n_slots(self) -> int:  # numpydoc ignore=RT01
        """The history store's rows: the method's order and two more."""
        return self.rows.n_slots

    # --- startup ----------------------------------------------------------------------

    def start(self, z: float) -> None:
        """Evaluate the source at the accepted state, the history's first sample.

        The sample goes to the store's row zero (the subclass's ``_startup``).

        Parameters
        ----------
        z : float
            The accepted position, m.
        """
        workspace = self._workspace
        state = workspace.committed()
        workspace.hoist(z)
        value = self._startup(state, z)
        rows = self.rows
        if rows.store is None or rows.store.shape[1:] != value.shape:
            # Zeroed, not empty: the rows a short history has not reached go into a
            # resume snapshot as they are, and a file has to be reproducible.
            rows.store = np.zeros((self.n_slots, *value.shape), dtype=np.complex128)
            if rows.compensated:
                rows.states = np.zeros_like(rows.store)
                rows.rates = np.zeros(rows.store.shape, dtype=np.complex128)
        rows.store[0] = value
        self._started(state, z)

    def _startup(self, state: _State, z: float) -> _State:
        """Return the startup sample at the accepted state, as the store keeps it."""
        raise NotImplementedError

    def _started(self, state: _State, z: float) -> None:
        """Keep what else the startup sample's row needs; nothing by default."""

    # --- the protocol's trial and accept -------------------------------------------

    def trial(
        self, z: float, dz: float, stencil: Stencil
    ) -> Float[np.ndarray, " n_orders"]:
        """Form a trial from `z` over `dz`; return its scored orders' norms.

        The formulation's own trial (the subclass's ``_trial``), held for
        :meth:`accept`.

        Parameters
        ----------
        z : float
            Where the trial starts, m.
        dz : float
            The step, m.
        stencil : solprop.backends.adams_core.Stencil
            The history's nodes and rows, newest first, and the orders scored.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order, the step's own first.
        """
        return self._trial(
            z, dz, stencil.nodes, stencil.rows, stencil.order, stencil.orders
        )

    def _trial(  # noqa: PLR0917 - called positionally
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form and hold one trial; return the scaled norms of its scored defects."""
        raise NotImplementedError

    def accept(
        self, dz: float, free: int, stencil: Stencil, kept: Sequence[HistoryEntry]
    ) -> AcceptRecord:
        """Adopt the held trial; the new sample takes row `free`.

        Parameters
        ----------
        dz : float
            The accepted step, m.
        free : int
            The history row the new sample takes.
        stencil : solprop.backends.adams_core.Stencil
            The trial's stencil.
        kept : collections.abc.Sequence of solprop.backends.engines.HistoryEntry
            The history the step keeps, oldest first, which the interaction
            formulation transports.

        Returns
        -------
        solprop.backends.engines.AcceptRecord
            The finish's split collar's sums.
        """
        del dz, stencil  # the held trial carries both
        rows = np.array([entry.row for entry in kept], dtype=np.int64)
        return self._accept(rows, free)

    def _accept(self, keep: np.ndarray, free: int) -> AcceptRecord:
        """Adopt the held trial, keeping rows `keep`; the new sample takes `free`."""
        raise NotImplementedError

    def chord_phase(self, z1: float, z2: float) -> float:
        """Return the carrier chord's largest phase over ``[z1, z2]``.

        Parameters
        ----------
        z1, z2 : float
            The interval, m.

        Returns
        -------
        float
            The phase, rad; 0.0 without a carrier frame.
        """
        carrier = self.carrier
        return 0.0 if carrier is None else carrier.chord_phase(z1, z2)

    def step_carrier(self, order: int) -> Float[np.ndarray, " n_rotations"]:
        """Return the last trial's carrier rotations over its `order` newest rows.

        Packed as native's are: the shift's imaginary part, then each row's rotation as
        its real and imaginary parts.

        Parameters
        ----------
        order : int
            The rows the dense observation's formula reads.

        Returns
        -------
        numpy.ndarray
            The rotations, empty without an active carrier frame.
        """
        if self._carrier_rows is None:
            return np.zeros(0)
        shift, factors = self._carrier_rows
        rows = [float(np.imag(shift).reshape(-1)[0])]
        for factor in factors[:order]:
            value = complex(np.asarray(factor).reshape(-1)[0])
            rows += [value.real, value.imag]
        return np.array(rows)

    def begin_attempt(self, cost: int) -> None:
        """Reserve the attempt's host-source calls (SEP 49 section 6.2).

        Parameters
        ----------
        cost : int
            The attempt's logical response calls.
        """
        self._workspace.reserve_calls(cost)

    def end_attempt(self, outcome: AttemptOutcome) -> None:
        """Release the reservation's rest and record a decided attempt's outcome.

        Parameters
        ----------
        outcome : solprop.backends.engines.AttemptOutcome
            How the attempt ended; an abandoned one only releases.
        """
        try:
            if outcome != AttemptOutcome.ABANDONED:
                self._workspace.record_trial(
                    accepted=outcome == AttemptOutcome.ACCEPTED
                )
        finally:
            self._workspace.release_calls()

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        solprop.backends.engines.BoundaryReading
            The workspace's reading.
        """
        return self._workspace.reading()

    def _adopt(self, advanced: _State, dz: float) -> AcceptRecord:
        """Finish an accepted step's state and commit it as the workspace's."""
        finished, record = self._workspace.finish(advanced, dz)
        self._workspace.install(finished)
        return record


class InteractionHistory(HostHistory):
    """The interaction formulation's history (SEP 13, SEP 14).

    The history keeps **transported** sources, re-anchored at every accepted step by
    the step's own propagator; a trial's predictor, source and corrector are
    :class:`~solprop.backends.numpy_ref.adams_kernels.AdamsKernels`' fused passes.

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state, source evaluation and propagators are used.
    method : solprop.stepper.multistep.Multistep
        The method.
    interpolate : bool, optional
        Whether an accepted step keeps its interpolant for dense observation.
    rows : AdamsRows or None, optional
        The rows to write; new ones by default.
    """

    def __init__(
        self,
        workspace: "NumpyWorkspace",
        method: Multistep,
        *,
        interpolate: bool = False,
        rows: AdamsRows | None = None,
    ) -> None:
        """Hold the workspace, the kernels and an empty store."""
        super().__init__(
            workspace, method, compensated=False, interpolate=interpolate, rows=rows
        )
        self._kernels = AdamsKernels()
        self._held: _FusedTrial | None = None

    def _startup(self, state: _State, z: float) -> _State:
        """Return the transported source at the accepted state."""
        value, _ = self._workspace.source(state, z, z, purpose="startup")
        return value

    def _trial(  # noqa: PLR0917 - called positionally
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
        ``k``-node predictor. The trial is held for ``_accept``.

        - ``nodes``: the history's normalised positions, newest first; ``slots`` their
          store rows.
        - ``orders``: the orders scored, the step's own (``order``) first.
        - Returns: one scaled norm per scored order.
        """
        workspace, kernels = self._workspace, self._kernels
        store = self.rows.store
        assert store is not None
        state = workspace.committed()
        workspace.hoist(z)
        propagator = workspace.propagator(z, z + dz, hold=True)
        predictor = integrated_weights(nodes[:order], check=False, helpers=PLAIN)
        lab = kernels.predict(state, propagator, store, slots[:order], predictor, dz)
        source, _ = workspace.source(lab, z, z + dz, purpose="predictor")
        rows, corrector = defect_rows(nodes, orders)
        width = rows.shape[1]
        mask = workspace.problem.grids.norm_mask
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
            _divisor(mask, self._controller, state),
        )
        self._held = _FusedTrial(
            z=z,
            dz=dz,
            state=state,
            propagator=propagator,
            source=source,
            advanced=advanced,
            slots=slots,
            order=order,
        )
        return norms

    def _accept(self, keep: np.ndarray, free: int) -> AcceptRecord:
        """Adopt the held fused trial: the state, its interpolant, the new sample.

        - ``keep``: the history rows kept, each re-anchored at the step's end.
        - Returns: the finish's split collar's sums.
        """
        trial = self._held
        store = self.rows.store
        assert trial is not None and store is not None
        self._held = None
        record = self._adopt(trial.advanced, trial.dz)
        order = trial.order
        if self._interpolate:
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
                slopes=slopes,
            )
        self._kernels.transport(store, keep, trial.propagator, free, trial.source)
        return record


class ExponentialHistory(HostHistory):
    """The exponential formulation's history (SEP 16, SEP 35).

    The history keeps **untransported** sources; a trial integrates its interpolants
    exactly against the step's flow ``exp(x)``, ``x = Phi(z + h) - Phi(z)``
    (:class:`~solprop.backends.numpy_ref.adams_kernels.ExponentialKernels`), steps a
    coupled flow's section against its frozen block, adds a varying operator's linear
    compensation of its samples, and under a carrier frame rotates the samples it reads.

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state, source evaluation and propagators are used.
    method : solprop.stepper.multistep.Multistep
        The method.
    compensation : {'transported', 'untransported'}
        A varying operator's compensation treatment (SEP 16 Q8); the untransported
        one is a test witness.
    interpolate : bool, optional
        Whether an accepted step keeps its interpolant for dense observation.
    rows : AdamsRows or None, optional
        The rows to write; new ones by default.

    Attributes
    ----------
    compensated : bool
        Whether a varying operator's linear compensation is integrated.
    """

    def __init__(
        self,
        workspace: "NumpyWorkspace",
        method: Multistep,
        compensation: Literal["transported", "untransported"],
        *,
        interpolate: bool = False,
        rows: AdamsRows | None = None,
    ) -> None:
        """Hold the workspace, the kernels, the carrier frame and an empty store."""
        linear = workspace.problem.linear
        # SEP 16 phase 4: a varying operator's linear compensation, integrated in the
        # reference flow's frame (Q8); the untransported treatment is a test witness.
        # A structurally absent source takes the exact chord instead (section 4.2).
        # SEP 16 phase 5: a frozen-residual coupled flow's residual is linear in the
        # state and joins the compensation, whatever the diagonal's representation.
        flow = linear.coupled_flow
        residual = flow is not None and "frozen_residual" in flow.section_methods
        self.compensated = residual or (
            linear.phi_representation in ("expansion", "table")
            and not source_absent(workspace.problem)
        )
        if self.compensated and flow is not None and compensation != "transported":
            msg = "the untransported compensation is a witness for diagonal operators"
            raise BackendError(msg)
        super().__init__(
            workspace,
            method,
            compensated=self.compensated,
            interpolate=interpolate,
            rows=rows,
        )
        self._kernels = ExponentialKernels()
        self._compensation = compensation
        self._held: _ExponentialTrial | None = None
        # A frozen-residual flow's weights at the held trial's end.
        self._trial_weights: np.ndarray | None = None
        # SEP 35: the carrier phase each trial removes from the samples it reads, from
        # a reference fixed for the run; None when off. An inert one (sigma = 0) takes
        # the unshifted arithmetic exactly.
        self.carrier = carrier_reference(workspace.problem, method)
        if self.carrier is not None:
            logger.info(
                "carrier frame %r: %s", self.carrier.policy, self.carrier.reason
            )

    def _startup(self, state: _State, z: float) -> _State:
        """Return the untransported source at the accepted state."""
        return self._workspace.lab_source(state, z, purpose="startup")

    def _started(self, state: _State, z: float) -> None:
        """Keep a compensated history's startup state, rate and flow weights."""
        rows = self.rows
        if not rows.compensated:
            return
        assert rows.states is not None and rows.rates is not None
        workspace = self._workspace
        # The startup sample's state and rate, read on the far side of a kink that
        # restarted the history.
        rows.states[0] = state
        rows.rates[0] = phi_rate(workspace.problem.linear, z, helpers=PLAIN)
        # A frozen-residual flow's weights at the sample, on the far side too.
        start = workspace.coupled_block(z, 0.0)
        rows.weights = [None] * self.n_slots
        rows.weights[0] = None if start is None else start.end_weights

    def _trial(  # noqa: PLR0917 - called positionally
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form one exponential trial and return the scaled norms of its defects.

        The interaction formulation's arithmetic with untransported samples: the
        predictor and the corrector integrate their interpolants exactly against the
        step's flow ``exp(x)``, ``x = Phi(z + h) - Phi(z)``; the defects of the
        neighbouring orders are scored from the same moments. A varying operator adds
        the compensation of its samples (SEP 16 section 4.2). Under a carrier frame the
        samples read are rotated and the moments shifted (SEP 35 section 4.1); the flow
        is unchanged. The trial is held for ``_accept``.

        - ``nodes``: the history's normalised positions, newest first; ``slots`` their
          store rows.
        - ``orders``: the orders scored, the step's own (``order``) first.
        - Returns: one scaled norm per scored order.
        """
        workspace, kernels = self._workspace, self._kernels
        store = self.rows.store
        assert store is not None
        state = workspace.committed()
        workspace.hoist(z)
        width = 1 + max(orders)
        # The exponent is the backend's own increment of Phi over the step, so the
        # flow below is exactly the propagator the interaction formulation applies.
        exponent = np.array(workspace.delta_phi(workspace.stage_scalars(z, z + dz)))
        # A coupled section's step (SEP 16 phase 5): its flow and moments are matrix
        # functions of the frozen block on the bins that carry one, applied by the
        # reference passes.
        block = workspace.coupled_block(z, dz)
        # SEP 35 section 4.1: with F(z_n + s) = exp(-sigma (1 - s/h)) G(s), the
        # step interpolates G_j = exp(sigma (1 - theta_j)) F_j and integrates it
        # against the moments at x - sigma; exact for any sigma, the flow untouched.
        carrier = self.carrier
        shift = (
            None if carrier is None or not carrier.active else carrier.shift(exponent)
        )
        # The moments at the shifted exponent and the true flow exp(x); a coupled
        # step's are the shifted block's, its flow exp(sigma) times their phi_0, the
        # true exp(h B) (section 4.5): one evaluation, never the shifted flow.
        moments: Any
        flow: Any
        if block is not None:
            moments = coupled_moments(exponent, width, block, dz, carrier=shift)
            flow = moments
        else:
            moments = kernels.moments(exponent, width, shift=shift)
            flow = kernels.flow(exponent)
        rows_read = slots[: width - 1]
        factors = (
            None if shift is None else _carrier_factors(nodes, len(rows_read), shift)
        )
        lab = kernels.predict(
            state,
            flow,
            moments,
            store,
            rows_read[:order],
            lagrange_coefficients(nodes[:order], check=False, helpers=PLAIN),
            dz,
            factors=None if factors is None else factors[:order],
        )
        compensation = None
        if self.compensated:
            compensation = self._compensation_terms(
                exponent, moments, nodes, slots[: width - 1], dz, block
            )
            if shift is not None:
                compensation.rotate(shift)
            lab = lab + compensation.predictor(order)
        source = workspace.lab_source(lab, z + dz, purpose="predictor")
        corrector = exponential_corrector(nodes, order, width)
        rows = kernels.defect_rows(nodes, orders, width)
        extra = None
        if compensation is not None:
            rate = phi_rate(
                workspace.problem.linear, z + dz, before=True, helpers=PLAIN
            )
            residual = None
            if block is not None and block.residual is not None:
                assert block.end_weights is not None
                residual = block.residual(lab, block.end_weights)
            compensation.add_endpoint(rate, lab, residual)
            self._trial_weights = None if block is None else block.end_weights
            extra = (compensation.corrector(order), compensation.defects(orders))
        mask = workspace.problem.grids.norm_mask
        advanced, norms = kernels.correct(
            state,
            flow,
            moments,
            source,
            store,
            rows_read,
            corrector,
            rows,
            dz,
            mask,
            _divisor(mask, self._controller, state),
            extra,
            factors=factors,
        )
        self._held = _ExponentialTrial(
            z=z,
            dz=dz,
            state=state,
            exponent=exponent,
            source=source,
            lab=lab,
            advanced=advanced,
            slots=slots,
            order=order,
            compensation=compensation,
        )
        self._carrier_rows = (
            None if shift is None or factors is None else (shift, factors)
        )
        return norms

    def _accept(self, keep: np.ndarray, free: int) -> AcceptRecord:
        """Adopt the held exponential trial; the new sample takes row `free`.

        - ``keep``: the history rows kept; nothing moves in this formulation.
        - Returns: the finish's split collar's sums.
        """
        del keep  # untransported samples are never re-anchored
        trial = self._held
        store = self.rows.store
        assert trial is not None and store is not None
        self._held = None
        record = self._adopt(trial.advanced, trial.dz)
        z, dz, order, slots = trial.z, trial.dz, trial.order, trial.slots
        compensation = trial.compensation
        if self._interpolate:
            # Dense output refuses a coupled flow, whose residual the history terms
            # below leave out (they are the diagonal chord's alone).
            assert self._workspace.problem.linear.coupled_flow is None
            # Copied: the store's rows are overwritten by later steps.
            self.dense = _ExponentialStep(
                z=z,
                dz=dz,
                start=trial.state.copy(),
                exponent=trial.exponent,
                # The samples as stored: under a carrier frame the observation
                # rotates them by the step's rotations (`AcceptedStep.carrier`).
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
            assert self.rows.states is not None and self.rows.rates is not None
            assert compensation.endpoint_rate is not None
            self.rows.states[free] = trial.lab
            self.rows.rates[free] = compensation.endpoint_rate
            # One entry per slot, set with the startup sample or a resumed history.
            self.rows.weights[free] = self._trial_weights
        return record

    def _history_terms(
        self, exponent: _State, dz: float, slots: np.ndarray
    ) -> list[_State]:
        """Return ``(D(z_j) - B_n) u_j`` of history samples, untransported.

        - ``exponent``: ``x = h B_n``.
        - ``dz``: the step, m.
        """
        assert self.rows.states is not None and self.rows.rates is not None
        reference = exponent / dz
        return [
            (self.rows.rates[slot] - reference) * self.rows.states[slot]
            for slot in slots
        ]

    def _compensation_terms(  # noqa: PLR0917 - called positionally
        self,
        exponent: _State,
        moments: Any,
        nodes: list[float],
        slots: np.ndarray,
        dz: float,
        block: CoupledBlock | None = None,
    ) -> "_Compensation":
        """Return the trial's linear compensation of its history samples.

        - ``exponent``: ``x = h B_n``, the step's chord exponent.
        - ``nodes``: the history's normalised positions, newest first.
        - ``dz``: the step, m.
        - ``block``: a coupled step's frozen block: the samples' residuals against it
          join their compensation, carried by its forward flow.
        """
        rows = self.rows
        assert rows.states is not None and rows.rates is not None
        used = list(nodes[: len(slots)])
        transported = self._compensation == "transported"
        # The untransported witness keeps the samples in their own frame: a span of 0.
        spans = np.array([1.0 - node if transported else 0.0 for node in used])
        if block is not None:
            carried = self._kernels.carry_coupled(
                exponent,
                dz,
                (rows.rates, rows.states, rows.weights),
                slots,
                spans,
                block,
            )
        else:
            carried = self._kernels.carry(
                exponent, dz, rows.rates, rows.states, slots, spans
            )
        return _Compensation(
            exponent=exponent,
            moments=moments,
            nodes=used,
            carried=carried,
            dz=dz,
            treatment=self._compensation,
        )


def _carrier_factors(nodes: list[float], count: int, shift: _Shift) -> _Factors:
    """Return the rotations ``exp(sigma (1 - theta_j))`` of a trial's history rows.

    The rotated-sample form of SEP 35 section 6.2: the passes multiply each row by its
    factor as they read it, so the store keeps the untransported samples and a
    rejection, a resume or a later step with its own shift reads them unchanged.

    - ``nodes``: the history's normalised positions ``theta_j``, newest first.
    - Returns: one factor per row, ``(count, *shift.shape)``.
    """
    return np.stack([np.exp(shift * (1.0 - nodes[index])) for index in range(count)])


def _divisor(
    mask: Bool[np.ndarray, " n_omega"],
    controller: "PIController",
    state: _State,
) -> np.ndarray:
    """Return the defect norm's scale per bin, the accepted state's before the step.

    The Runge-Kutta march scales by the larger of the states before and after the step.
    An Adams step outside its stability region can multiply the state many times over
    while its predictor and corrector still agree to a fraction of the *new* state, so
    that scale lets a divergence pass as accurate: SEP 13 phase 2 measured PEC4 growing
    from 1e7 to 1e236 in five accepted steps with every scaled norm below 0.6. The state
    the step starts from is the scale that cannot grow with the step it judges.

    - Returns: the scale of every bin, shape ``(n_omega,)``.
    """
    magnitude = float(np.linalg.norm(state[..., mask]))
    scale = controller.scale(magnitude, magnitude)
    return (
        scale
        if isinstance(scale, np.ndarray)
        else np.full(state.shape[-1], float(scale))
    )
