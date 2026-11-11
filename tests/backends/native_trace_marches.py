# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native marches' call-by-call adapter, for the decision-trace tests.

The native session runs the compiled Adams and Runge-Kutta marches (``adams_march.cpp``,
``rk_march.cpp``) to an event and holds their state natively. The decision-trace tests
hold those marches to the reference's decisions on identical scripted readings
(``adams_traces.py``, ``rk_traces.py``): a scripted native march
(``_core.ScriptedMarch``, ``_core.ScriptedRkMarch``) is driven a call at a time over a
scripted host backend, each call's attempts booked into the harness as the reference
books its own. That driving
loop is the adapter the native backend used before SEP 47 phase 5 moved every native run
into its session; it lives here, beside the harnesses it drives, and serves only them.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, ClassVar, cast

import numpy as np
import pytest

from solprop._interrupts import committing
from solprop.backends import rk_core
from solprop.backends.multistep import (
    _Derivative,  # pyright: ignore[reportPrivateUsage]
    budget_error,
)
from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule, reached, replay_tolerance

backend_module = pytest.importorskip("solprop_native.backend")
session_module = pytest.importorskip("solprop_native.session")
native = pytest.importorskip("solprop_native")

WATCH_POLICY_FIELDS = native.WATCH_POLICY_FIELDS
WATCH_STATE_FIELDS = native.WATCH_STATE_FIELDS
SPLIT_RATE_BUCKETS = __import__("solprop.boundaries", fromlist=["x"]).SPLIT_RATE_BUCKETS
_REACHED = backend_module._REACHED
_HOST_STEP = backend_module._HOST_STEP
_MONITOR = backend_module._MONITOR
_WATCH = backend_module._WATCH
_FLOOR = backend_module._FLOOR
_BUDGET = backend_module._BUDGET
_DIVERGED = backend_module._DIVERGED
_DENSE = backend_module._DENSE
_ResidentAdams = backend_module._ResidentAdams
_envelope_errors = backend_module._envelope_errors
_raise_rk = session_module._raise_rk

logger = logging.getLogger(__name__)

_VERDICTS = ("none", "stop", "warning", "quiet verdict", "refused crossing", "error")
"""A watch return's verdicts, by ``watch::Verdict`` (solprop SEP 45 section 4.8)."""


def _watch_policy_of(monitor: BoundaryMonitor) -> dict[str, object]:
    """Return a monitor's watch policy, refusing one with other fields (SEP 45 4.8)."""
    policy = monitor.watch_policy()
    if tuple(policy) != WATCH_POLICY_FIELDS:
        raise BackendError(
            "solprop's boundary watch policy has the fields "
            f"{tuple(policy)}, not the {WATCH_POLICY_FIELDS} this build reads"
        )
    return policy


def _readings_of(
    monitor: BoundaryMonitor, result: dict[str, Any]
) -> dict[int, BoundaryEnergies]:
    """Return the readings a native call's checks took, by the row they follow."""
    rows = [int(row) for row in result["reading_rows"]]
    if not rows:
        return {}
    values = np.asarray(result["readings"], dtype=np.float64).reshape(len(rows), -1)
    readings: dict[int, BoundaryEnergies] = {}
    for row, reading in zip(rows, values, strict=True):
        readings[row] = BoundaryEnergies(
            float(reading[0]),
            reading[1:5].copy(),
            removed=float(reading[5]),
            found=reading[6:10].copy() if monitor.split else None,
            occupancy=None
            if monitor.crossing is None
            else reading[10 : 10 + SPLIT_RATE_BUCKETS].copy(),
        )
    return readings


def _verify_watch_of(monitor: BoundaryMonitor, result: dict[str, Any]) -> None:
    """Hold solprop's monitor, a drain done, to a native march's predictions.

    Raises
    ------
    BackendError
        If they disagree: a fault in the native transcription of the monitor's
        checks, which the run cannot continue past (solprop SEP 45, Q4).
    """
    predicted = result["state"]["watch"]
    actual = monitor.watch_state()
    if any(predicted[name] != actual[name] for name in WATCH_STATE_FIELDS):
        raise BackendError(
            "the native march's boundary checks disagree with solprop's monitor: "
            f"predicted {dict(predicted)}, booked {actual}"
        )
    if int(result["reason"]) == _WATCH and int(result["verdict"]) in (1, 4, 5):
        raise BackendError(
            "the native march returned to stop the run on a "
            f"{_VERDICTS[int(result['verdict'])]}, and solprop's monitor did not"
        )


class TraceAdamsMarch(backend_module._NativeAdamsMarch):
    """The native Adams record, driven a call at a time over a scripted harness.

    Parameters
    ----------
    backend : object
        The scripted harness, a host driver whose ``_adams_engine`` gives the history.
    method : solprop.stepper.multistep.Multistep
        The method.
    runner : object
        The scripted native march, in the native state's place.
    """

    def __init__(self, backend: Any, method: Multistep, *, runner: Any) -> None:
        super().__init__(
            backend, method, engine=backend._adams_engine(method, "transported")
        )
        self._driver: Any = backend
        self._runner = runner
        self._ahead: tuple[float, object] | None = None

    def _configure(self) -> None:
        """Configure nothing: the scripted runner takes the policy itself."""

    def _observe(self, position: float) -> object:
        """Return the field at `position` inside the last accepted step."""
        return self._engine.observe(position)

    def _raise(self, reason: int, result: dict[str, Any]) -> None:
        """Raise the error a call returned with, the budget's included."""
        if reason == _BUDGET:
            raise self._budget_error(int(result["cost"]))
        super()._raise(reason, result)

    MAX_ATTEMPTS: ClassVar[int] = 256
    """A call's most attempts, the bound on the wait of a signal or a stop flag (solprop
    SEP 39 Q3)."""

    MAX_SECONDS: ClassVar[float] = 0.1
    """A call's most wall time, s, read once per attempt."""

    def _budget_error(self, cost: int) -> ResponseBudgetError:
        """Return the error of a trial the call budget cannot pay for."""
        backend = self._driver
        return budget_error(
            self._policy.max_calls,
            self.ledger.total,
            backend._z,
            cost,
            backend._adams_state()[0].copy(),
        )

    def march_to(self, target: float) -> None:
        """Step the backend's state to `target`, natively when the march is admitted.

        Parameters
        ----------
        target : float
            Position to reach, m.

        Raises
        ------
        ConvergenceError
            If the proposal falls below the contract's ``dz_min``, or a native
            refinement fails.
        ResponseBudgetError
            If the next trial would exceed the method's call budget.
        """
        backend = self._driver
        if self._ahead is not None:
            # The march is ahead of the last observation; resume it there.
            self._engine.restore(*self._ahead)
            self._ahead = None
        if backend._natural_stop_due(target):
            return
        self._configure()
        schedule = backend._stepper.schedule
        if schedule is not None:
            # A replay takes the schedule's steps as given.
            slack = replay_tolerance(schedule, target)
            while (
                backend._schedule_index < schedule.n_steps
                and backend._z < target - slack
            ):
                if self._call(target, slack):
                    return
            if backend._z < target - slack:
                msg = (
                    f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                    f"cannot reach z = {target:.6g} m"
                )
                raise BackendError(msg)
            return
        while not reached(backend._z, target):
            if self._call(target):
                return
        if self._interpolate and backend._z > target:
            self._ahead = (backend._z, self._engine.hold())
            backend._state = cast("Any", self._observe(target))
            backend._z = target
            self.ledger.dense_observations += 1

    def _watching(self) -> bool:
        """Whether the march takes the boundary checks itself (solprop SEP 45 4.8).

        Every monitored run but one with a transverse edge, whose reads stay the
        host's (SEP 44), and a scripted run, whose engine takes no readings.
        """
        monitor = self._driver._monitor
        return (
            self._runner is None and monitor is not None and monitor.transverse is None
        )

    def _watch_policy(self) -> dict[str, object]:
        """Return the monitor's watch policy, refusing one with other fields."""
        return _watch_policy_of(self._driver._monitor)

    def _bounds(
        self, target: float, slack: float | None, watching: bool = False
    ) -> dict[str, object]:
        """Return a call's bounds: the target, the steps the driver has work on and,
        for a replay, the edges' slack; under the march's own checks, their policy."""
        backend = self._driver
        steps: list[int] = []
        if backend._row_spec is not None:
            period = backend._row_period
            steps.append(period - backend._accepted_steps % period)
        every = backend._stepper.step_planes
        if every:
            steps.append(every - backend._natural_count % every)
        monitor = backend._monitor
        checked = None
        read_at = None
        spacing = 0.0
        if monitor is not None and not watching:
            steps.append(monitor.steps_to_check())
            checked = monitor.checked_at
            # The transverse edge is read on the host (solprop SEP 40 section 5.3):
            # steps are capped at its spacing, the march returns before a step
            # would carry past its next read, and a collar's current is taken at
            # every committed step.
            edge = monitor.transverse
            if edge is not None:
                if edge.every_step:
                    steps.append(1)
                if math.isfinite(edge.next_read):
                    read_at = float(edge.next_read)
                    spacing = float(edge.ceiling(backend._z))
        return {
            "target": float(target),
            "quiet": bool(backend._collar_quiet),
            "host_step": int(min(steps)) if steps else 0,
            "monitor": checked is not None,
            "first_check": False,
            "check_interval": 0.0 if monitor is None else float(monitor.interval),
            "last_check": 0.0 if checked is None else float(checked),
            "read_at": read_at,
            "read_spacing": spacing,
            # The split collar's crossing limit, read at the last check (solprop
            # #440): a ceiling on every adaptive step until the next.
            "step_cap": math.inf if monitor is None else float(monitor.step_cap),
            "max_attempts": self.MAX_ATTEMPTS,
            "max_seconds": self.MAX_SECONDS,
            "replay": slack is not None,
            "slack": 0.0 if slack is None else float(slack),
            "watch": self._watch_policy() if watching else None,
            "watch_capacity": self.MAX_ATTEMPTS,
        }

    def _state(self, watching: bool = False) -> dict[str, object]:
        """Return the bridge march's state as the native call takes it."""
        backend = self._driver
        state = self.start_state(backend._z, backend._proposal, backend._schedule_index)
        if watching:
            state["watch"] = backend._monitor.watch_state()
        return state

    def _readings(self, result: dict[str, Any]) -> dict[int, BoundaryEnergies]:
        """Return the readings a call's checks took, by the row they follow."""
        return _readings_of(self._driver._monitor, result)

    def _verify_watch(self, result: dict[str, Any]) -> None:
        """Hold solprop's monitor, the drain done, to the march's predictions."""
        _verify_watch_of(self._driver._monitor, result)

    def _call(self, target: float, slack: float | None = None) -> bool:  # noqa: PLR0912 - one call drained
        """Make one native call and drain it; return whether a natural stop is due.

        `slack` is a replay's (:func:`solprop.stepper.schedule.replay_tolerance`), or
        ``None`` for an adaptive march.
        """
        backend = self._driver
        backend._poll_stop()
        watching = self._watching()
        bounds = self._bounds(target, slack, watching)
        started = time.perf_counter() - backend._clock
        cursor = int(backend._schedule_index)
        position = float(backend._z)
        with committing():
            result = self._native_run(self._state(watching), bounds)
            self._synchronise(result)
            rows = np.asarray(result["rows"], dtype=np.float64).reshape(-1, 6)
            reason = int(result["reason"])
            readings = self._readings(result) if watching else {}
            # Only the last accepted row of an event can have the driver's work; the
            # others are accounted, each once, with any reading the march took
            # (solprop SEP 45 section 4.10).
            event = reason in (_REACHED, _HOST_STEP, _MONITOR, _WATCH)
            last = len(rows) - 1
            for index, (end, dz, norm, accepted, _, wall) in enumerate(rows.tolist()):
                backend._record_attempt(end, dz, norm, bool(accepted), started + wall)
                if slack is not None:
                    backend._accepted_norms.append(norm)
                if accepted:
                    position = end
                    if slack is not None:
                        cursor += 1
                final = index == last and (event or reason == _FLOOR)
                if accepted and not final:
                    try:
                        backend._account_step(end, cursor, readings.get(index))
                    except Exception as error:
                        if not watching:
                            raise
                        # The march stops at a check whose verdict needs solprop, so
                        # a step it went on past cannot be one.
                        raise BackendError(
                            "solprop's boundary monitor acted on a check the native "
                            f"march went on past, at z = {end:.9g} m: {error}"
                        ) from error
        progress = backend._progress
        for end, dz, norm, accepted, order, _ in rows.tolist():
            progress.step(end, dz, bool(accepted))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "native Adams attempt to z = %.9g m: dz = %.4g m, order %d, "
                    "norm %.4g, %s",
                    end,
                    dz,
                    int(order),
                    norm,
                    "accepted" if accepted else "rejected",
                )
        if event and rows.size and rows[-1, 3]:
            if watching:
                backend._watched_step(position, cursor, readings.get(last))
                backend._statistics_step()
            else:
                backend._take_step_stats()
            if watching:
                self._verify_watch(result)
            if backend._natural_stop(target):
                return True
        elif watching:
            self._verify_watch(result)
        self._raise(reason, result)
        return False

    def _native_run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]:
        """Run the native march from `state` to an event within `bounds`."""
        if self._runner is not None:
            return self._runner.run(state, bounds)
        native = self._driver._native_state
        with _envelope_errors():
            return native.adams_march_run(state, bounds)

    def _synchronise(self, result: dict[str, Any]) -> None:
        """Adopt the state a native call handed back."""
        backend, engine = self._driver, self._engine
        state = result["state"]
        backend._z = float(state["z"])
        backend._proposal = float(state["proposal"])
        backend._schedule_index = int(state["cursor"])
        self._order = int(state["order"])
        self._rejections = int(state["rejections"])
        self._last_rejected = bool(state["last_rejected"])
        self._collar_was_quiet = bool(state["collar_was_quiet"])
        self._history = [
            _Derivative(
                float(z), None, "predicted" if predicted else "accepted", int(slot)
            )
            for z, predicted, slot in state["history"]
        ]
        ledger = self.ledger
        for name, value in state["ledger"].items():
            setattr(ledger, name, int(value))
        ledger.orders.extend(int(order) for order in result["orders"])
        if isinstance(engine, _ResidentAdams):
            step = None
            if result["stepped"]:
                order = int(result["step_order"])
                nodes = [1.0, *(float(node) for node in result["step_nodes"][:order])]
                step = (float(result["step_z"]), float(result["step_dz"]), nodes)
            engine.march_stepped(
                step, fused=self._fused, carrier=list(result["step_carrier"])
            )


class TraceRungeKuttaMarch(session_module._NativeRungeKuttaMarch):
    """The native Runge-Kutta policy, its march driven a call at a time.

    Parameters
    ----------
    backend : object
        The scripted harness, a host driver.
    runner : object or None, optional
        The scripted native march; ``None`` builds the policy alone.
    """

    def __init__(self, backend: Any, runner: Any = None) -> None:
        super().__init__(backend)
        self._runner = runner
        self._replaying: FixedSchedule | None = None

    def _interval(self) -> float | None:
        """Return the harness's monitor's quiet interval, or ``None``."""
        monitor = self._driver._monitor
        return None if monitor is None else float(monitor.interval)

    def _policy(self, interval: float | None = None) -> dict[str, object]:
        """Return the record with the harness's quiet interval."""
        return super()._policy(self._interval() if interval is None else interval)

    def _configure(self) -> None:
        """Configure nothing: the scripted runner takes the policy itself."""

    MAX_ATTEMPTS: int = 256

    MAX_SECONDS: float = 0.1

    def _watching(self) -> bool:
        """Whether the march takes the boundary checks itself (SEP 45 4.8)."""
        monitor = self._driver._monitor
        return (
            self._runner is None and monitor is not None and monitor.transverse is None
        )

    def march_to(self, target: float) -> None:
        """Step the backend's state to `target` on the native march.

        Parameters
        ----------
        target : float
            Position to reach, m.

        Raises
        ------
        ConvergenceError
            If the proposal falls below the contract's ``dz_min``, or a native
            trial fails.
        """
        backend = self._driver
        dense = backend._stepper.interpolate_planes
        if dense and backend._dense_ahead is not None:
            # The march is ahead of the last interpolated plane; resume it there.
            backend._adams_restore(*backend._dense_ahead)
            backend._dense_ahead = None
        if backend._natural_stop_due(target):
            return
        self._configure()
        while not reached(backend._z, target):
            if self._call(target):
                return
        if dense and not reached(target, backend._z):
            backend._observe_inside(target)

    def replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay the contract's schedule to `target` on the native march.

        ``HostDriver._replay_to``, a call of steps at a time: the steps as given, never
        rejected, the edges hit exactly.

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
        ConvergenceError
            If a step's error norm is not finite.
        """
        backend = self._driver
        self._replaying = schedule
        self._configure()
        slack = replay_tolerance(schedule, target)
        while (
            backend._schedule_index < schedule.n_steps and backend._z < target - slack
        ):
            if self._call(target, slack):
                return
        if backend._z < target - slack:
            msg = (
                f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                f"cannot reach z = {target:.6g} m"
            )
            raise BackendError(msg)

    def _bounds(
        self, target: float, watching: bool, slack: float | None = None
    ) -> dict[str, object]:
        """Return a call's bounds: the target and the steps the driver has work on."""
        backend = self._driver
        steps: list[int] = []
        if backend._row_spec is not None:
            period = backend._row_period
            steps.append(period - backend._accepted_steps % period)
        every = backend._stepper.step_planes
        if every:
            steps.append(every - backend._natural_count % every)
        monitor = backend._monitor
        checked = None
        read_at = None
        ceiling = math.inf
        if monitor is not None and not watching:
            steps.append(monitor.steps_to_check())
            checked = monitor.checked_at
            # The transverse edge is read on the host (solprop SEP 40 section 5.3):
            # its ceiling caps every step, the march returns before a step would carry
            # past its next read, and a collar's current is taken at every step.
            edge = monitor.transverse
            if edge is not None:
                ceiling = float(edge.ceiling(backend._z))
                if edge.every_step:
                    steps.append(1)
                if math.isfinite(edge.next_read):
                    read_at = float(edge.next_read)
        return {
            "target": float(target),
            "quiet": bool(backend._collar_quiet),
            "host_step": int(min(steps)) if steps else 0,
            "monitor": checked is not None,
            "first_check": False,
            "check_interval": 0.0 if monitor is None else float(monitor.interval),
            "last_check": 0.0 if checked is None else float(checked),
            "read_at": read_at,
            "read_spacing": ceiling,
            "step_cap": math.inf if monitor is None else float(monitor.step_cap),
            "max_attempts": self.MAX_ATTEMPTS,
            "max_seconds": self.MAX_SECONDS,
            "watch": _watch_policy_of(monitor)
            if watching and monitor is not None
            else None,
            "watch_capacity": self.MAX_ATTEMPTS,
            "replay": slack is not None,
            "slack": 0.0 if slack is None else float(slack),
        }

    def _state(self, watching: bool) -> dict[str, object]:
        """Return the march's state as the native call takes it."""
        backend = self._driver
        state: dict[str, object] = {
            "z": float(backend._z),
            "proposal": float(backend._proposal),
            "norms": [float(norm) for norm in backend._accepted_norms],
            "cursor": int(backend._schedule_index),
        }
        if watching:
            state["watch"] = backend._monitor.watch_state()
        return state

    def _call(self, target: float, slack: float | None = None) -> bool:  # noqa: PLR0912 - one call drained
        """Make one native call and drain it; return whether a natural stop is due."""
        backend = self._driver
        backend._poll_stop()
        watching = self._watching()
        bounds = self._bounds(target, watching, slack)
        started = time.perf_counter() - backend._clock
        cursor = int(backend._schedule_index)
        position = float(backend._z)
        with committing():
            result = self._native_run(self._state(watching), bounds)
            state = result["state"]
            backend._z = float(state["z"])
            backend._schedule_index = int(state["cursor"])
            backend._proposal = float(state["proposal"])
            backend._accepted_norms.clear()
            backend._accepted_norms.extend(float(norm) for norm in state["norms"])
            rows = np.asarray(result["rows"], dtype=np.float64).reshape(-1, 6)
            reason = int(result["reason"])
            readings = _readings_of(backend._monitor, result) if watching else {}
            # Beside a transverse edge the march returns at a check due by distance,
            # which solprop takes on the event step (`_take_step_stats`).
            event = reason in (_REACHED, _HOST_STEP, _MONITOR, _WATCH)
            last = len(rows) - 1
            for index, (end, dz, norm, accepted, _, wall) in enumerate(rows.tolist()):
                backend._record_attempt(end, dz, norm, bool(accepted), started + wall)
                if accepted and not (index == last and reason == _DENSE):
                    position = end
                    if slack is not None:
                        # The row's own replay cursor (SEP 45 4.10).
                        cursor += 1
                final = index == last and (event or reason in (_FLOOR, _DENSE))
                if accepted and not final:
                    try:
                        backend._account_step(end, cursor, readings.get(index))
                    except Exception as error:
                        if not watching:
                            raise
                        raise BackendError(
                            "solprop's boundary monitor acted on a check the native "
                            f"march went on past, at z = {end:.9g} m: {error}"
                        ) from error
        progress = backend._progress
        for end, dz, _, accepted, _, _ in rows.tolist():
            progress.step(end, dz, bool(accepted))
        if event and rows.size and rows[-1, 3]:
            if watching:
                backend._watched_step(position, cursor, readings.get(last))
                backend._statistics_step()
                _verify_watch_of(backend._monitor, result)
            else:
                backend._take_step_stats()
            if backend._natural_stop(target):
                return True
        elif watching:
            _verify_watch_of(backend._monitor, result)
        if reason == _DENSE:
            end, dz = float(rows[-1, 0]), float(rows[-1, 1])
            return self._finish_crossing(
                target, end, dz, float(result["pending_equivalent"])
            )
        self._raise(reason, result)
        return False

    def _finish_crossing(
        self, target: float, end: float, dz: float, equivalent: float
    ) -> bool:
        """Finish the accepted step across an interpolated plane, as the reference.

        `HostDriver._march_to`'s acceptance, line for line: the accepted-step hook
        `_record` makes, the step's extension from the state before it, the finish and
        the commit, the norm, the start-up rule, the proposal, the statistics and the
        natural stop. The native march returned the step accepted and unfinished.
        """
        backend = self._driver
        policy = backend._controller_policy()
        backend._on_accepted_step()
        backend._dense_step = backend._dense_record(backend._z, dz)
        finished = backend._finish(backend._state, dz)
        with committing():
            backend._state = finished
            backend._z = end
        backend._accepted_norms.append(equivalent)
        if policy.auto_init and len(backend._accepted_norms) == 1:
            backend._start_up(equivalent)
        backend._proposal = rk_core.propose(
            policy, backend._proposal, backend._accepted_norms
        )
        backend._take_step_stats()
        return bool(backend._natural_stop(target))

    def _native_run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]:
        """Make one native call: the native state's march, or the scripted one."""
        if self._runner is not None:
            return self._runner.run(state, bounds)
        with _envelope_errors():
            return self._driver._native_state.rk_march_run(state, bounds)

    def _raise(self, reason: int, result: dict[str, Any]) -> None:
        """Raise the error a call returned with, if any."""
        backend = self._driver
        schedule = self._replaying
        _raise_rk(
            reason,
            result,
            name=backend.name,
            dz_min=float(backend._stepper.dz_min),
            at=(float(backend._z), int(backend._schedule_index)),
            n_steps=0 if schedule is None else schedule.n_steps,
        )
