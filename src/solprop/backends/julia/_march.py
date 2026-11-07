# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The exponential Adams march executed by the Julia engine (SEP 39's native march).

The algorithm is :class:`~solprop.backends.multistep.AdamsMarch`'s, which stays the
reference: ``julia/src/march.jl`` transcribes it (by way of native's
``adams_march.cpp``) and performs a run of attempts on the resident engine, returning
to Python only at a host event: the target reached, an accepted step on which the
driver has work (a statistics row, a natural stop, a monitor check, a transverse
read), the call's bound, the budget, the floor or an error. Between calls the march's
state is this object's, as the Python march keeps it, so a snapshot, a reset and a
new run are the reference's own. The call protocol, the bounds and the adoption follow
native's ``_NativeAdamsMarch``.

A configuration the Julia march does not cover runs the Python march
(:attr:`JuliaAdamsMarch.refusal`): a replayed schedule, interpolated planes, the
carrier frame, a compensated trial without an expansion's rate table, or
``SOLPROP_JULIA_MARCH=host``. A split collar's sums are formed from the power each
accepted step's commit found, by the adapter, as its own commit does.
"""

import logging
import math
import os
import time
from typing import Any

import numpy as np

from solprop._interrupts import committing
from solprop.backends.driver import reached
from solprop.backends.julia._runtime import call_julia
from solprop.backends.multistep import (
    AdamsMarch,
    _Derivative,  # pyright: ignore[reportPrivateUsage] - the march's history entries
)
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper import Multistep

logger = logging.getLogger(__name__)

__all__ = ["JuliaAdamsMarch"]

_REACHED, _HOST_STEP, _MONITOR, _BOUND, _BUDGET, _FLOOR, _ERROR, _DIVERGED = range(8)
_CONVERGENCE, _BACKEND, _VALUE = 1, 2, 3
_COUNTERS = (
    "startup",
    "predictor",
    "rejected",
    "accepted_steps",
    "rejected_steps",
    "restarts",
    "failure_restarts",
)
"""The call ledger's counters the Julia march keeps, in its order."""


class JuliaAdamsMarch(AdamsMarch):
    """solprop's exponential Adams march, executed by the Julia engine.

    Parameters
    ----------
    backend : solprop.backends.julia.JuliaBackend
        The backend whose engine steps.
    method : solprop.stepper.Multistep
        The exponential Adams method.
    """

    MAX_ATTEMPTS = 256
    """A call's most attempts, the bound on the wait of a signal or a stop flag."""
    MAX_SECONDS = 0.1
    """A call's most wall time, s, read once per attempt."""

    def __init__(self, backend: Any, method: Multistep) -> None:
        """Build the Python march, then decide which march steps the run."""
        super().__init__(backend, method)
        # The driver's protected bookkeeping, which this march books into exactly as
        # solprop's does (``DRIVER_SURFACE``), as native's march reaches it.
        self._driver: Any = backend
        self._configured: tuple[float | None, bool] | None = None
        self.refusal = self._refusal()
        """Why the Python march steps the run, or ``None`` when Julia's does."""

    @property
    def native(self) -> bool:
        """Whether the Julia march steps the run."""
        return self.refusal is None

    def _refusal(self) -> str | None:
        """Return why the Julia march does not step the run, or ``None``."""
        problem = self._driver._problem
        engine: Any = self._engine
        reasons = (
            (os.environ.get("SOLPROP_JULIA_MARCH") == "host", "asked for"),
            (self._exponential is None, "not the exponential formulation"),
            (problem.stepper.schedule is not None, "a replayed schedule"),
            (self._interpolate, "interpolated planes"),
            (self._policy.carrier, "the carrier frame"),
            (
                engine.compensated and problem.linear.phi_expansion is None,
                "a compensated trial without an expansion",
            ),
        )
        return next((reason for failed, reason in reasons if failed), None)

    def march_to(self, target: float) -> None:
        """Step the backend's state to `target`, by Julia's march when admitted.

        Parameters
        ----------
        target : float
            Position to reach, m.
        """
        if not self.native:
            super().march_to(target)
            return
        backend = self._driver
        if backend._natural_stop_due(target):
            return
        self._configure()
        while not reached(backend._z, target):
            if self._call(target):
                return

    def _configure(self) -> None:
        """Hand the engine the run's policy, again when the monitor's interval moved."""
        from solprop.backends.julia._backend import _splits  # noqa: PLC0415 - cycle

        monitor = self._driver._monitor
        interval = None if monitor is None else float(monitor.interval)
        split = bool(
            _splits(self._driver._problem) and self._driver._split_monitor is not None
        )
        if self._configured == (interval, split):
            return
        policy = self._policy
        backend = self._driver
        problem = backend._problem
        engine: Any = self._engine
        numbers = np.array(
            [
                policy.order,
                policy.variable_order,
                policy.safety,
                policy.max_growth,
                policy.min_factor,
                policy.rejection_growth,
                policy.hysteresis,
                policy.restart_after,
                policy.collar_stability,
                policy.collar_rate,
                policy.max_calls is not None,
                0 if policy.max_calls is None else policy.max_calls,
                policy.equal_approach,
                policy.dz_min,
                policy.dz_max,
                policy.stretch,
                policy.z_end,
                policy.landing_ulps,
                interval is not None,
                0.0 if interval is None else interval,
            ],
            dtype=np.float64,
        )
        edges = coefficients = np.empty(0)
        if engine.compensated:
            edges, _, _, coefficients = problem.linear.rate.packed()
        call_julia(
            backend._jl.configure_march_b,
            backend._engine,
            numbers,
            np.asarray(policy.real_intervals, dtype=np.float64),
            np.asarray(policy.save_planes, dtype=np.float64),
            np.asarray(policy.hard_kinks, dtype=np.float64),
            np.asarray(policy.breakpoints, dtype=np.float64),
            len(problem.stepper.tableau.nodes),
            bool(backend._varying),
            bool(engine.compensated),
            split,
            np.ascontiguousarray(edges, dtype=np.float64),
            np.ascontiguousarray(coefficients, dtype=np.float64)
            if engine.compensated
            else np.zeros((0, 0, 0)),
        )
        self._configured = (interval, split)

    def _bounds(self, target: float) -> tuple[np.ndarray, np.ndarray]:
        """Return a call's bounds: the target, and the steps the driver has work on.

        Returns
        -------
        tuple of numpy.ndarray
            The limits (target, check interval, last check, read position and
            spacing, step cap, seconds) and flags (quiet, host step, monitor, first
            check, a read due, attempts) of ``march_run!``.
        """
        backend = self._driver
        steps: list[int] = []
        if backend._row_spec is not None:
            period = backend._row_period
            steps.append(period - backend._accepted_steps % period)
        every = backend._problem.stepper.step_planes
        if every:
            steps.append(every - backend._natural_count % every)
        monitor = backend._monitor
        checked = None
        read_at = None
        spacing = 0.0
        if monitor is not None:
            steps.append(monitor.steps_to_check())
            checked = monitor.checked_at
            # The transverse edge is read on the host (SEP 40 section 5.3): steps are
            # capped at its spacing, and the march returns before one would carry
            # past its next read.
            edge = monitor.transverse
            if edge is not None:
                if edge.every_step:
                    steps.append(1)
                if math.isfinite(edge.next_read):
                    read_at = float(edge.next_read)
                    spacing = float(edge.ceiling(backend._z))
        limits = np.array(
            [
                target,
                0.0 if monitor is None else float(monitor.interval),
                0.0 if checked is None else float(checked),
                0.0 if read_at is None else read_at,
                spacing,
                math.inf if monitor is None else float(monitor.step_cap),
                self.MAX_SECONDS,
            ]
        )
        flags = np.array(
            [
                bool(backend._collar_quiet),
                int(min(steps)) if steps else 0,
                checked is not None,
                False,
                read_at is not None,
                self.MAX_ATTEMPTS,
            ],
            dtype=np.int64,
        )
        return limits, flags

    def _call(self, target: float) -> bool:
        """Make one Julia call and drain it; return whether a natural stop is due."""
        backend = self._driver
        backend._poll_stop()
        limits, flags = self._bounds(target)
        started = time.perf_counter() - backend._clock
        history = self._history
        with committing():
            result = call_julia(
                backend._jl.march_run_b,
                backend._engine,
                np.array([backend._z, backend._proposal]),
                np.array(
                    [
                        self._order,
                        self._rejections,
                        self._last_rejected,
                        self._collar_was_quiet,
                    ],
                    dtype=np.int64,
                ),
                np.array([entry.z for entry in history], dtype=np.float64),
                np.array([entry.identity == "predicted" for entry in history]),
                np.array([entry.slot for entry in history], dtype=np.int64),
                np.array([getattr(self.ledger, name) for name in _COUNTERS]),
                limits,
                flags,
                nogil=True,
            )
            self._synchronise(result)
            rows = np.asarray(result.rows, dtype=np.float64).reshape(-1, 6)
            reason = int(result.reason)
            # Only the last accepted row of an event can have the driver's work; the
            # others take the skip path, as the reference's would.
            event = reason in (_REACHED, _HOST_STEP, _MONITOR)
            last = len(rows) - 1
            for index, (end, dz, norm, accepted, _, wall) in enumerate(rows.tolist()):
                backend._record_attempt(end, dz, norm, bool(accepted), started + wall)
                final = index == last and (event or reason == _FLOOR)
                if accepted and not final:
                    backend._skip_step(end)
        progress = backend._progress
        for end, dz, norm, accepted, order, _ in rows.tolist():
            progress.step(end, dz, bool(accepted))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "julia Adams attempt to z = %.9g m: dz = %.4g m, order %d, "
                    "norm %.4g, %s",
                    end,
                    dz,
                    int(order),
                    norm,
                    "accepted" if accepted else "rejected",
                )
        if event and rows.size and rows[-1, 3]:
            backend._take_step_stats()
            if backend._natural_stop(target):
                return True
        self._raise(reason, result)
        return False

    def _synchronise(self, result: Any) -> None:
        """Adopt the state a Julia call handed back."""
        backend = self._driver
        z, proposal = np.asarray(result.zs).tolist()
        backend._z, backend._proposal = float(z), float(proposal)
        order, rejections, last_rejected, was_quiet = np.asarray(result.counts).tolist()
        self._order, self._rejections = int(order), int(rejections)
        self._last_rejected, self._collar_was_quiet = (
            bool(last_rejected),
            bool(was_quiet),
        )
        self._history = [
            _Derivative(
                float(z), None, "predicted" if predicted else "accepted", int(row)
            )
            for z, predicted, row in zip(
                np.asarray(result.history_z).tolist(),
                np.asarray(result.history_predicted).tolist(),
                np.asarray(result.history_row).tolist(),
                strict=True,
            )
        ]
        ledger = self.ledger
        for name, value in zip(
            _COUNTERS, np.asarray(result.ledger).tolist(), strict=True
        ):
            setattr(ledger, name, int(value))
        orders = np.asarray(result.orders).tolist()
        ledger.orders.extend(int(order) for order in orders)
        steps = np.asarray(result.split_steps).tolist()
        if steps:
            # Each accepted step's split sums from the power its commit found, by the
            # operations `JuliaBackend._finish` performs, step by step.
            powers = np.asarray(result.split_powers).reshape(len(steps), -1).T
            monitor = backend._split_monitor
            rate = monitor.weights[3]
            for index, dz in enumerate(steps):
                power = np.ascontiguousarray(powers[:, index])
                half = power * np.exp(-0.5 * rate * dz)
                backend._split_removed += float(power @ -np.expm1(-rate * dz))
                backend._split_found = np.concatenate(
                    [[half.sum()], monitor.weights[:3] @ half]
                )
        if orders:
            # The accepted steps' commits, which the engine made: what `_finish` sets.
            backend._alarm = float(result.alarm)
            backend._state = backend._engine
            backend._fsal = backend._recycle
        first = float(result.first_alarm)
        if not math.isnan(first) and not backend._alarm_reported:
            backend._alarm_reported = True
            logger.warning(
                "transverse quadrature error %.2e exceeds 1e-6 at z=%.6g m; "
                "refine the quadrature",
                first,
                float(result.alarm_z),
            )

    def _raise(self, reason: int, result: Any) -> None:
        """Raise the error a call returned with, if any."""
        if reason == _BUDGET:
            raise self._budget_error(int(result.cost))
        if reason == _FLOOR:
            proposed = float(result.floor_proposal)
            msg = (
                f"step control asked for {proposed:.3e} m, below the floor dz_min ="
                f" {self._policy.dz_min:.3e} m; the tolerance cannot be met here"
            )
            raise ConvergenceError(msg)
        if reason == _ERROR:
            message = str(result.message)
            category = int(result.category)
            if category == _CONVERGENCE:
                raise ConvergenceError(message)
            if category == _VALUE:
                raise ValueError(message)
            raise BackendError(f"julia execution failed: {message}")
