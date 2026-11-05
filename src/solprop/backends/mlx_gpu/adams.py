# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The exponential Adams history on the Metal device, for the ``mlx`` backend.

:class:`MlxAdams` is an :class:`~solprop.backends.multistep.AdamsEngine`: the march
(:class:`~solprop.backends.multistep.AdamsMarch`) keeps the controller -- the orders,
the slots, rejection, restarts -- and this engine keeps the history's untransported
source rows on the device and forms each trial there, as
:class:`~solprop.backends.multistep.HostHistory` forms it on the host. The step's flow
``exp(x)`` and its moments ``k! phi_{k+1}(x)`` are evaluated on the host in binary64 and
rounded once to binary32, as the single-precision oracle rounds them; the predictor,
the corrector and the defects are compiled device passes over the rows, with their
Lagrange coefficients as inputs; and a trial reads its norms, the anchor's magnitude,
the trial state's finiteness and the envelope plasma's failures in one
synchronisation. One source evaluation per trial.

It takes the exponential formulation, with or without a carrier frame. A varying
operator (a gradient, a taper) adds the transported compensation of SEP 16 section 4.2:
each history sample keeps its state on the device and its rate ``D(z_j)`` on the host,
the state-independent factor ``exp((1 - theta_j) x) (D(z_j) - x / h)`` is formed in
binary64 and rounded once, and the products with the states and the integrated Adams
weights are device passes. The backend refuses the untransported witness by name.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx  # pyright: ignore[reportMissingImports] - macOS arm64 only
import numpy as np

from solprop.backends.exponential import carrier_reference
from solprop.backends.multistep import (
    ExponentialKernels,
    _carrier_factors,
    _exponential_corrector,
    source_absent,
)
from solprop.exceptions import BackendError
from solprop.stepper.multistep import (
    Multistep,
    integrated_weights,
    lagrange_coefficients,
)
from solprop.translate.linop import phi_rate

if TYPE_CHECKING:
    from solprop.backends.mlx_gpu.backend import MlxBackend

__all__ = ["MlxAdams"]


def _combine(coefficients: mx.array, values: list[mx.array]) -> mx.array:
    """Return ``sum_j c_fj X_j`` for every row ``f`` of `coefficients`, at once.

    One real matrix product over the stacked samples' real and imaginary parts, so a
    compiled graph takes the coefficients as one input rather than one per entry
    (Metal bounds a fused kernel's buffers).

    Parameters
    ----------
    coefficients : mlx.core.array
        ``(n_f, len(values))``, float32.
    values : list of mlx.core.array
        The samples, of one shape.

    Returns
    -------
    mlx.core.array
        ``(n_f, *shape)``.
    """
    shape = values[0].shape
    flat = mx.stack(values).reshape(len(values), -1)
    re = coefficients @ mx.real(flat)
    im = coefficients @ mx.imag(flat)
    return (re + 1j * im).reshape(coefficients.shape[0], *shape)


def _increments(
    moments: mx.array, tables: mx.array, values: list[mx.array], dz: mx.array
) -> mx.array:
    """Return ``dz sum_k M_k (sum_j c_fjk X_j)`` for every formula ``f``.

    Parameters
    ----------
    moments : mlx.core.array
        ``(n_moments, *shape)``.
    tables : mlx.core.array
        ``c_fjk``, float32, ``(n_f, len(values), n_k)`` with ``n_k <= n_moments``.
    values : list of mlx.core.array
        The samples.
    dz : mlx.core.array
        The step, a float32 scalar.

    Returns
    -------
    mlx.core.array
        ``(n_f, *shape)``.
    """
    n_f, n_j, n_k = tables.shape
    rows = mx.transpose(tables, (0, 2, 1)).reshape(n_f * n_k, n_j)
    combined = _combine(rows, values).reshape(n_f, n_k, *values[0].shape)
    return dz * mx.sum(moments[None, :n_k] * combined, axis=1)


def _scaled_square_sum(values: mx.array, index: mx.array) -> list[mx.array]:
    """Return the power of two at or below the peak and the scaled sum of squares.

    SEP 33's rule 6 over the norm's bins, as the Runge-Kutta norm reduces.

    Parameters
    ----------
    values : mlx.core.array
        A state-shaped array.
    index : mlx.core.array
        The norm's bins.

    Returns
    -------
    list of mlx.core.array
        The power and the sum, float32 scalars.
    """
    array = mx.take(values, index, axis=-1)
    peak = mx.max(mx.abs(array))
    power = mx.where(peak > 0, mx.floor(mx.log2(peak)), 0.0)
    scaled = array * mx.power(2.0, -power)
    re, im = mx.real(scaled), mx.imag(scaled)
    return [power, mx.sum(re * re + im * im)]


def _predict(  # noqa: PLR0913, PLR0917 - a kernel's buffers
    state: mx.array,
    flow: mx.array,
    moments: mx.array,
    values: list[mx.array],
    coefficients: mx.array,
    dz: mx.array,
    carried: list[mx.array] | None = None,
    weights: mx.array | None = None,
) -> mx.array:
    """Return the predicted field ``exp(x) u_n + dz sum_k M_k sum_j c_jk F_j``.

    Parameters
    ----------
    state, flow, moments : mlx.core.array
        The anchor, ``exp(x)`` and the moments.
    values : list of mlx.core.array
        The history rows read, newest first, rotated under a carrier frame.
    coefficients : mlx.core.array
        The predictor's Lagrange coefficients, ``(len(values), n_k)``.
    dz : mlx.core.array
        The step.
    carried : list of mlx.core.array or None, optional
        A varying operator's carried compensations.
    weights : mlx.core.array or None, optional
        The predictor's integrated weights over them, ``(len(carried),)``.

    Returns
    -------
    mlx.core.array
        The lab-frame field at the end of the step.
    """
    lab = flow * state + _increments(moments, coefficients[None], values, dz)[0]
    if carried is not None and weights is not None:
        lab = lab + dz * _combine(weights[None], carried)[0]
    return lab


def _correct(  # noqa: PLR0913, PLR0917 - a kernel's buffers
    state: mx.array,
    flow: mx.array,
    moments: mx.array,
    values: list[mx.array],
    corrector: mx.array,
    rows: mx.array,
    dz: mx.array,
    index: mx.array,
    compensation: dict[str, Any] | None = None,
) -> tuple[mx.array, list[mx.array]]:
    """Return the corrected field and the reductions its norms are read from.

    Parameters
    ----------
    state, flow, moments : mlx.core.array
        As for the predictor.
    values : list of mlx.core.array
        The new source, then the history rows read.
    corrector : mlx.core.array
        The corrector's coefficients, ``(width, n_k)``.
    rows : mlx.core.array
        Each scored order's defect coefficients, ``(n_rows, width, n_k)``.
    dz : mlx.core.array
        The step.
    index : mlx.core.array
        The norm's bins.
    compensation : dict or None, optional
        A varying operator's terms: ``carried``, the history's compensations;
        ``endpoint``, the new sample's; ``corrector`` and ``predictor``, each scored
        order's integrated weights, ``(n_rows, 1 + n_carried)`` (the endpoint's first)
        and ``(n_rows, n_carried)``.

    Returns
    -------
    tuple
        The field, and per defect, then for the anchor, a power of two and a scaled
        sum of squares, then whether the field is finite, as float32 scalars.
    """
    tables = mx.concatenate([corrector[None], rows], axis=0)
    increments = _increments(moments, tables, values, dz)
    advanced = flow * state + increments[0]
    defects = increments[1:]
    if compensation is not None:
        carried = compensation["carried"]
        after = dz * _combine(
            compensation["corrector"], [compensation["endpoint"], *carried]
        )
        before = dz * _combine(compensation["predictor"], carried)
        advanced = advanced + after[0]
        defects = defects + (after - before)
    reductions: list[mx.array] = []
    for row in range(rows.shape[0]):
        reductions += _scaled_square_sum(defects[row], index)
    reductions += _scaled_square_sum(state, index)
    reductions.append(mx.all(mx.isfinite(advanced)).astype(mx.float32))
    return advanced, reductions


_predict_fused = mx.compile(_predict)
_correct_fused = mx.compile(_correct)


@dataclass
class _Trial:
    """A formed trial, held until the march accepts or rejects it."""

    dz: float
    source: mx.array
    advanced: mx.array
    lab: mx.array
    rate: np.ndarray | None


class MlxAdams:
    """The exponential Adams history and its arithmetic, on the Metal device.

    Parameters
    ----------
    backend : solprop.backends.mlx_gpu.backend.MlxBackend
        The backend whose state, sources, controller and synchronisation are used.
    method : solprop.stepper.multistep.Multistep
        The method, in the exponential formulation.

    Attributes
    ----------
    dense : None
        No dense observation: ``interpolate_planes`` is refused on ``mlx``.
    compensated : bool
        Whether a varying operator's samples carry the transported compensation.
    carrier : solprop.backends.exponential.CarrierReference or None
        The carrier frame the trials remove (SEP 35), as the host engine's.
    """

    dense: Any = None

    def __init__(self, backend: "MlxBackend", method: Multistep) -> None:  # noqa: UP037 - a type-checking import
        """Hold the backend and an empty history; nothing is read from the device."""
        self._backend = backend
        self._method = method
        self._kernels = ExponentialKernels(helpers=backend.helpers)
        self._rows: list[mx.array | None] = [None] * self.n_slots
        problem = backend._problem
        self.compensated = problem.linear.phi_representation in (
            "expansion",
            "table",
        ) and not source_absent(problem)
        # A compensated history's sample states on the device, rates on the host.
        self._states: list[mx.array | None] = [None] * self.n_slots
        self._rates: list[np.ndarray | None] = [None] * self.n_slots
        self._trial: _Trial | None = None
        self.carrier = carrier_reference(backend._problem, method)

    @property
    def n_slots(self) -> int:
        """The history's rows: the method's order and two more.

        Returns
        -------
        int
            ``order + 2``.
        """
        return self._method.order + 2

    def reset(self) -> None:
        """Forget any trial in flight."""
        self._trial = None

    def hold(self) -> object:
        """Return the accepted state, the device array itself.

        Returns
        -------
        object
            The state.
        """
        return self._backend._require_state()

    def restore(self, z: float, held: object) -> None:
        """Make a held state current again at `z`.

        Parameters
        ----------
        z : float
            Its position, m.
        held : object
            What :meth:`hold` returned.
        """
        self._backend._adams_restore(z, cast(Any, held))

    def start(self, z: float, *, stored: bool) -> None:
        """Evaluate the source at the accepted state into row zero.

        Parameters
        ----------
        z : float
            The accepted position, m.
        stored : bool
            Must be true: the general path runs on the host and is refused.

        Raises
        ------
        BackendError
            For an unstored sample, which only the general path asks for.
        """
        if not stored:
            msg = "the mlx Adams engine keeps every sample in its history"
            raise BackendError(msg)
        backend = self._backend
        backend._hoist(z)
        state = cast(mx.array, backend._require_state())
        self._rows[0] = backend._adams_lab_source(state, z, purpose="startup")
        if self.compensated:
            # The startup sample's state and rate, read on the far side of a kink
            # that restarted the history.
            self._states[0] = state
            self._rates[0] = phi_rate(
                backend._problem.linear, z, helpers=backend.helpers
            )

    def value(self, slot: int) -> None:
        """Return nothing: the history's rows stay on the device.

        Parameters
        ----------
        slot : int
            The row.
        """
        del slot

    def fused_trial(self, *args: object, **kwargs: object) -> np.ndarray:
        """Refuse the interaction formulation, which this engine does not run.

        Parameters
        ----------
        *args, **kwargs : object
            The march's trial arguments.

        Raises
        ------
        BackendError
            Always.
        """
        del args, kwargs
        msg = "the mlx Adams engine runs the exponential formulation only"
        raise BackendError(msg)

    def fused_accept(self, keep: np.ndarray, free: int, *, interpolate: bool) -> None:
        """Refuse the interaction formulation, which this engine does not run.

        Parameters
        ----------
        keep : numpy.ndarray
            The rows kept.
        free : int
            The free row.
        interpolate : bool
            Whether the interpolant is kept.

        Raises
        ------
        BackendError
            Always.
        """
        del keep, free, interpolate
        msg = "the mlx Adams engine runs the exponential formulation only"
        raise BackendError(msg)

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Form one exponential trial on the device and return its defects' norms.

        :meth:`HostHistory.exponential_trial
        <solprop.backends.multistep.HostHistory.exponential_trial>`'s arithmetic over
        device rows, read in one synchronisation.

        Parameters
        ----------
        z, dz : float
            The anchor and the step, m.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            Their rows, ``int64``.
        order : int
            The step's order.
        orders : list of int
            The orders scored, the step's own first.

        Returns
        -------
        numpy.ndarray
            One scaled norm per scored order; infinite when the trial is not finite.
        """
        backend = self._backend
        state = cast(mx.array, backend._require_state())
        backend._hoist(z)
        backend._envelope_failures = []
        width = 1 + max(orders)
        helpers = backend.helpers
        exponent = np.array(backend._delta_phi(backend._stage_scalars(z, z + dz)))
        carrier = self.carrier
        shift = (
            None if carrier is None or not carrier.active else carrier.shift(exponent)
        )
        # SEP 33: the flow and its moments in binary64, each rounded once.
        moments = mx.array(
            np.asarray(self._kernels.moments(exponent, width, shift=shift)).astype(
                np.complex64
            )
        )
        flow = mx.array(np.exp(exponent).astype(np.complex64))
        values: list[mx.array] = []
        for slot in slots[: width - 1]:
            row = self._rows[int(slot)]
            assert row is not None  # every slot the march reads holds a sample
            values.append(row)
        if shift is not None:
            factors = _carrier_factors(nodes, len(values), shift).astype(np.complex64)
            values = [
                mx.array(factor) * value
                for factor, value in zip(factors, values, strict=True)
            ]
        length = mx.array(dz, dtype=mx.float32)
        predictor = lagrange_coefficients(nodes[:order], check=False, helpers=helpers)
        carried = weights = None
        if self.compensated:
            carried = self._carried(exponent, dz, nodes, slots[: width - 1])
            weights = self._weights(nodes, [order], len(carried))[1][0]
        # Chosen here: the march builds the engine before the backend's own tables.
        fused = backend._fused
        lab = (_predict_fused if fused else _predict)(
            state,
            flow,
            moments,
            values[:order],
            mx.array(np.asarray(predictor, dtype=np.float32)),
            length,
            carried,
            weights,
        )
        source = cast(
            mx.array, backend._adams_lab_source(lab, z + dz, purpose="predictor")
        )
        corrector = _exponential_corrector(nodes, order, width, helpers=helpers)
        rows = self._kernels.defect_rows(nodes, orders, width)
        compensation = rate = None
        if carried is not None:
            rate = phi_rate(
                backend._problem.linear, z + dz, before=True, helpers=helpers
            )
            after, before = self._weights(nodes, orders, len(carried))
            compensation = {
                "carried": carried,
                # The new sample's compensation at the predicted state, untransported.
                "endpoint": mx.array((rate - exponent / dz).astype(np.complex64)) * lab,
                "corrector": after,
                "predictor": before,
            }
        advanced, reductions = (_correct_fused if fused else _correct)(
            state,
            flow,
            moments,
            [source, *values],
            mx.array(np.asarray(corrector, dtype=np.float32)),
            mx.array(np.asarray(rows, dtype=np.float32)),
            length,
            backend._norm_index,
            compensation,
        )
        read = backend._synchronised([*reductions, backend._failure_count()])
        backend._raise_on_failures(float(read[-1]))
        self._trial = _Trial(
            dz=dz, source=source, advanced=advanced, lab=lab, rate=rate
        )
        if not np.all(np.isfinite(read[:-1])) or read[-2] != 1.0:
            return np.full(len(orders), math.inf)
        squares = [
            math.ldexp(float(read[2 * k + 1]), 2 * int(read[2 * k]))
            for k in range(len(orders) + 1)
        ]
        magnitude = math.sqrt(squares[-1])
        scale = backend._norm_controller.scale(magnitude, magnitude)
        if isinstance(scale, np.ndarray):
            return self._weighted(
                [source, *values], moments, rows, length, scale, len(orders)
            )
        return np.array([math.sqrt(square) / scale for square in squares[:-1]])

    def _carried(
        self, exponent: np.ndarray, dz: float, nodes: list[float], slots: np.ndarray
    ) -> list[mx.array]:
        """Return the history samples' compensations carried to the step's end.

        ``exp(s_j x) (D(z_j) - x / h) u_j``, ``s_j = 1 - theta_j``: the factor before
        the state is formed in binary64, the difference of rates before anything
        multiplies it (SEP 16 section 4.2), and rounded once.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x``, binary64.
        dz : float
            The step, m.
        nodes : list of float
            The history's normalised positions, newest first.
        slots : numpy.ndarray
            The rows read, newest first.

        Returns
        -------
        list of mlx.core.array
            One carried row per sample.
        """
        reference = exponent / dz
        carried = []
        for node, slot in zip(nodes, slots, strict=False):
            rate, state = self._rates[int(slot)], self._states[int(slot)]
            assert rate is not None and state is not None
            factor = np.exp((1.0 - node) * exponent) * (rate - reference)
            carried.append(mx.array(factor.astype(np.complex64)) * state)
        return carried

    def _weights(
        self, nodes: list[float], orders: list[int], count: int
    ) -> tuple[mx.array, mx.array]:
        """Return each order's corrector and predictor weights over the carried rows.

        The transported treatment's ordinary integrated Adams weights, the
        corrector's through the endpoint sample, padded with zeros to the rows the
        trial carries.

        Parameters
        ----------
        nodes : list of float
            The history's normalised positions, newest first.
        orders : list of int
            The orders.
        count : int
            The carried rows.

        Returns
        -------
        tuple of mlx.core.array
            ``(n_orders, 1 + count)`` and ``(n_orders, count)``, float32.
        """
        helpers = self._backend.helpers
        after = np.zeros((len(orders), 1 + count))
        before = np.zeros((len(orders), count))
        for row, order in enumerate(orders):
            after[row, : order + 1] = integrated_weights(
                np.asarray([1.0, *nodes[:order]]), check=False, helpers=helpers
            )
            before[row, :order] = integrated_weights(
                np.asarray(nodes[:order]), check=False, helpers=helpers
            )
        return (
            mx.array(after.astype(np.float32)),
            mx.array(before.astype(np.float32)),
        )

    def _weighted(
        self,
        values: list[mx.array],
        moments: mx.array,
        rows: np.ndarray,
        length: mx.array,
        scale: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """Return the norms under a controller that scales each bin.

        Parameters
        ----------
        values : list of mlx.core.array
            The new source and the rows read.
        moments : mlx.core.array
            The trial's moments.
        rows : numpy.ndarray
            The defects' coefficients.
        length : mlx.core.array
            The step.
        scale : numpy.ndarray
            The scale of every bin.
        count : int
            The orders scored.

        Returns
        -------
        numpy.ndarray
            One norm per order.
        """
        backend = self._backend
        weights = mx.array(
            (1.0 / scale[backend._problem.grids.norm_mask]).astype(np.float32)
        )
        sums = []
        for row in range(count):
            defect = _increments(
                moments,
                mx.array(rows[row : row + 1].astype(np.float32)),
                values,
                length,
            )[0]
            weighted = mx.take(defect, backend._norm_index, axis=-1) * weights
            re, im = mx.real(weighted), mx.imag(weighted)
            sums.append(mx.sum(re * re + im * im))
        read = np.array(mx.stack(sums), dtype=np.float64)
        return np.sqrt(read)

    def exponential_accept(
        self, keep: np.ndarray, free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held trial; its source takes row `free`.

        Parameters
        ----------
        keep : numpy.ndarray
            The rows kept; nothing moves in this formulation.
        free : int
            The row the new sample takes.
        interpolate : bool
            Must be false: dense observation is refused on ``mlx``.
        """
        del keep, interpolate
        trial = self._trial
        assert trial is not None
        self._trial = None
        self._backend._adams_adopt(trial.advanced, trial.dz)
        self._rows[free] = trial.source
        if self.compensated:
            self._states[free] = trial.lab
            self._rates[free] = trial.rate

    def snapshot_nbytes(self) -> int:
        """Return the bytes :meth:`snapshot_rows` would copy.

        Returns
        -------
        int
            The rows' size, complex64.
        """
        shape = self._backend._require_state().shape
        rows = 2 if self.compensated else 1
        rates = 16 * self.n_slots * math.prod(shape) if self.compensated else 0
        return rows * self.n_slots * math.prod(shape) * 8 + rates

    def snapshot_rows(self) -> dict[str, np.ndarray]:
        """Return host copies of the history's rows, in complex64.

        Returns
        -------
        dict
            ``"store"``: one row per slot, zero where the history has none yet.
        """
        shape = self._backend._require_state().shape
        rows = {"store": self._host(self._rows, shape)}
        if self.compensated:
            rows["states"] = self._host(self._states, shape)
            rates = np.zeros((self.n_slots, *shape), dtype=np.complex128)
            for slot, rate in enumerate(self._rates):
                if rate is not None:
                    rates[slot] = rate
            rows["rates"] = rates
        return rows

    def _host(self, rows: list[mx.array | None], shape: tuple[int, ...]) -> np.ndarray:
        """Return device rows on the host, complex64, zero where there is none.

        Parameters
        ----------
        rows : list of mlx.core.array or None
            One per slot.
        shape : tuple of int
            The state's shape.

        Returns
        -------
        numpy.ndarray
            ``(n_slots, *shape)``.
        """
        host = np.zeros((self.n_slots, *shape), dtype=np.complex64)
        for slot, row in enumerate(rows):
            if row is not None:
                host[slot] = np.array(row)
        return host

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows a snapshot holds, narrowed to complex64 on the device.

        Parameters
        ----------
        rows : collections.abc.Mapping
            The rows, by name.
        shape : tuple of int
            The state's shape.

        Raises
        ------
        BackendError
            If the rows are not a store of this history's shape, or hold any other
            name (a compensated or coupled history's).
        """
        expected = (self.n_slots, *shape)
        names = {"store", "states", "rates"} if self.compensated else {"store"}
        if set(rows) != names or any(rows[name].shape != expected for name in names):
            msg = (
                "the mlx backend cannot continue this Adams history: it keeps "
                f"{sorted(names)} rows of shape {expected} and nothing else"
            )
            raise BackendError(msg)
        self.reset()
        self._rows = [mx.array(row) for row in rows["store"].astype(np.complex64)]
        if self.compensated:
            self._states = [
                mx.array(row) for row in rows["states"].astype(np.complex64)
            ]
            self._rates = [np.array(row, dtype=np.complex128) for row in rows["rates"]]

    def observe(self, position: float) -> object:
        """Refuse dense observation, which ``mlx`` does not run.

        Parameters
        ----------
        position : float
            The plane, m.

        Raises
        ------
        BackendError
            Always.
        """
        del position
        msg = "the mlx backend does not interpolate planes"
        raise BackendError(msg)
