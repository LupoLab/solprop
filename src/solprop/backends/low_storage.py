# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage interaction-picture RK4 march: replay-only, accepted in place.

SEP 24 (``docs/dev/seps/archive/``,
``sep-0024-low-storage-interaction-picture-runge-kutta.md``).
Classical fourth-order Runge-Kutta in the interaction picture anchored at the step's
midpoint, Lawson's transformation [1]_ in Hult's arrangement [2]_. For a step
``[z_n, z_n + h]`` with ``z_m = z_n + h/2``, ``P = U(z_m, z_n)``, ``Q = U(z_n + h,
z_m)`` and ``U(z_2, z_1) = exp(Phi(z_2) - Phi(z_1))``, the step is

``g1 = P F(z_n, A_n)``, ``F2 = F(z_m, P A_n + h/2 g1)``,
``F3 = F(z_m, P A_n + h/2 F2)``, ``F4 = F(z_n + h, Q (P A_n + h F3))``,
``A_{n+1} = Q (P A_n + h/6 g1 + h/3 F2 + h/3 F3) + h/6 F4``,

with ``F`` the complete source (the response, the in-stage absorbers and the
``modified`` noise draw at the evaluation's position). Only forward flows are applied,
four per element, and on a constant operator ``P = Q = exp(h D / 2)``.

**The register form** (SEP 24 section 4.3) holds three field-sized registers beside the
source pipeline's scratch ``S``: the state ``A``, overwritten in place by ``P A`` and
then by the accepted step, the accumulator ``B`` and the stage register ``C``. A step
is four passes, each forming an input, evaluating ``F`` into ``S`` and folding it into
the registers:

====  ==============  ======================================================
pass  evaluation      finish
====  ==============  ======================================================
1     ``F1`` at z_n   ``A <- P A``; ``g <- P S``; ``B <- A + h/6 g``;
                      ``C <- A + h/2 g``
2     ``F2`` at z_m   ``B <- B + h/3 S``; ``C <- A + h/2 S``
3     ``F3`` at z_m   ``B <- B + h/3 S``; ``C <- Q (A + h S)``
4     ``F4`` at end   ``A <- Q B + h/6 S``
====  ==============  ======================================================

**The march** (:class:`LowStorageMarch`) replays a prescribed schedule and never
rejects, so it accepts each step in place: from the step's start to its commit the
backend is *in flight* (``StepDriver._in_flight``) and every entry that reads or
advances the state refuses it, until ``StepDriver.set_initial`` loads a state again
(SEP 24 section 6.2, ADR-050). It records no error estimate: the step history's norm
is 0.0, "not estimated" (section 6.3). Divergence is detected on the accepted state's
norm instead.

**The engine** holds the registers and makes the passes (:class:`LowStorageEngine`); a
backend may keep its own where its state lives (``StepDriver._low_storage_engine``).
:class:`HostLowStorage` is the host engine and the oracle, over the backend's
``_adams_lab_source`` and ``_adams_propagator``.

References
----------
.. [1] J. D. Lawson, "Generalized Runge-Kutta processes for stable systems with large
       Lipschitz constants", SIAM Journal on Numerical Analysis 4, 372 (1967).
       doi:10.1137/0704033
.. [2] J. Hult, "A fourth-order Runge-Kutta in the interaction picture method for
       simulating supercontinuum generation in optical fibers", J. Lightwave Technol.
       25, 3770 (2007). doi:10.1109/JLT.2007.909373
"""

import logging
import math
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float

from solprop._interrupts import committing
from solprop.backends.driver import LANDING_ULPS, StepDriver, replay_tolerance
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import FixedSchedule

__all__ = [
    "HostLowStorage",
    "LowStorageEngine",
    "LowStorageMarch",
    "schedule_key",
    "table_steps",
]

logger = logging.getLogger(__name__)

_State = Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]


def table_steps(schedule: FixedSchedule) -> Float[np.ndarray, " n_steps"]:
    """Return the step each of a schedule's steps builds and keys the flow table on.

    SEP 24 Q10, decided in phase 2: the mean step of the step's **run**, the longest
    stretch of consecutive steps within :data:`~solprop.backends.driver.LANDING_ULPS`
    ulps of the run's first, the ulps being of the run's farthest position (or of the
    step, near the origin). :meth:`~solprop.stepper.schedule.FixedSchedule.uniform`
    takes a segment's steps from differences of rounded positions, which differ in their
    last bits by up to an ulp of the *position*: far from the origin that is many ulps
    of the step, and a table keyed on one step's own value within ulps of the step was
    rebuilt nearly every step. Keyed on the run's mean ``h_bar`` instead, a uniform
    segment builds one table and reuses it; each flow's exponent is then off by ``|D| |h
    - h_bar| / 2``, a few ulps of the position times ``|D|``, and the error does not
    drift: the steps of a run sum to ``n h_bar`` to rounding, so the phase the flows
    apply over a run is the run's own. A schedule whose steps all differ (a discovered
    one) has runs of one step and keys each step on itself.

    Parameters
    ----------
    schedule : solprop.stepper.schedule.FixedSchedule
        The schedule replayed.

    Returns
    -------
    numpy.ndarray
        One key per step, m, shape ``(n_steps,)``: constant over a run.

    Examples
    --------
    >>> from solprop.backends.low_storage import table_steps
    >>> from solprop.stepper import FixedSchedule
    >>> schedule = FixedSchedule.uniform(0.0, 1.0, 0.1, save_planes=[0.25, 1.0])
    >>> keys = table_steps(schedule)
    >>> len(set(keys.tolist())), round(float(keys[0]), 12), round(float(keys[-1]), 12)
    (2, 0.083333333333, 0.09375)
    """
    steps, edges = schedule.steps, schedule.edges
    keys = np.empty_like(steps)
    start = 0
    while start < steps.size:
        end = start + 1
        while end < steps.size:
            reach = max(abs(float(edges[start])), abs(float(edges[end + 1])))
            slack = LANDING_ULPS * float(
                max(np.spacing(reach), np.spacing(steps[start]))
            )
            if abs(float(steps[end] - steps[start])) > slack:
                break
            end += 1
        # A run of one keys on its own step, unrounded by the edges' difference.
        keys[start:end] = (
            steps[start]
            if end - start == 1
            else (edges[end] - edges[start]) / (end - start)
        )
        start = end
    return keys


def schedule_key(
    keys: Float[np.ndarray, " n_steps"],
    edges: Float[np.ndarray, " n_edges"],
    z: float,
    dz: float,
) -> float:
    """Return the table key of the schedule step from `z`, from :func:`table_steps`.

    Found by position rather than by a step counter, which a lockstep member does not
    advance (``LockstepRunner`` keeps the batch's).

    Parameters
    ----------
    keys : numpy.ndarray
        :func:`table_steps` of the schedule.
    edges : numpy.ndarray
        The schedule's edges, m.
    z : float
        Where the step starts, m.
    dz : float
        The step, m.

    Returns
    -------
    float
        The key, m.
    """
    index = int(np.searchsorted(edges, z + 0.5 * dz)) - 1
    return float(keys[min(max(index, 0), keys.size - 1)])


@runtime_checkable
class LowStorageEngine(Protocol):
    """What :class:`LowStorageMarch` asks of the engine that holds its registers.

    Part of the kernel seam (schema 29): the march keeps the schedule, the commit and
    the records; an engine keeps the registers and the flow table and does every
    state-sized operation of a step. :class:`HostLowStorage` is the reference.
    """

    stiffness: float
    """``h rho`` of the last completed step (SEP 24 section 6.3), or NaN where the
    engine does not form it: the march warns when it passes RK4's stability interval."""

    atomic: bool
    """Whether :meth:`step` is one native call that no signal interrupts (schema 32).

    A signal's Python handler runs at the first bytecode after such a call returns,
    which is before the commit, so the march then takes the step, the backend's
    ``_finish`` and the position move in one ``committing()`` block (SEP 24 section
    6.7): only a native exception can abandon the step in flight. An engine whose step
    runs Python between its passes is not atomic, and a signal there abandons the step
    at once rather than waiting up to a step for the commit (section 6.2, Q6)."""

    def step(self, z: float, dz: float) -> float:
        """Take one step in place, leaving the raw accepted state in the backend.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The accepted state's squared norm over the norm mask, before the boundary
            treatments, in the state's units squared; not finite when the field has
            diverged.
        """
        ...

    def reset(self) -> None:
        """Forget what a step left behind: the state was loaded outside the march."""
        ...

    def close(self) -> None:
        """Release the registers and the flow table; the engine is not used again."""
        ...


class HostLowStorage:
    """The host engine: the four register passes over host arrays (SEP 24 6.5).

    The backend's state is the register ``A`` and is updated in place; ``B`` and ``C``
    are this engine's host arrays, allocated at the first step. Each pass makes one
    source evaluation through ``StepDriver._adams_lab_source`` (the complete source,
    the noise draw advanced to the pass's position and the in-stage absorbers
    included) and takes ``P`` and ``Q`` from ``StepDriver._adams_propagator`` or the
    table, allocating transients as the reference does. The step's three positions
    are known before its first pass, so their host scalars are read together
    (``StepDriver._stage_bundles``) and the hooks take the kept bundles.

    **The flows.** ``"tabulate"`` on a constant operator holds one volume,
    ``exp(h_t D / 2)``, built at the first step after a load and reused while the
    steps keep the same key ``h_t``, their run's mean step (:func:`table_steps`, SEP 24
    Q10): a uniform schedule's steps are differences of rounded positions and differ in
    their last bits. The table is built from the key alone, from the operator's origin
    to ``h_t / 2`` -- a constant operator's increment is the distance itself -- so its
    exponent carries one rounding of ``D h_t / 2`` and none of the step's position
    (``fl(z + h/2) - z`` would be off by up to half an ulp of ``z``, an error that
    accumulates). Each reused flow's exponent is then off by ``|D| |h - h_t| / 2``, a
    few ulps of the position times ``|D|``, which does not accumulate over the run. A
    varying operator's
    ``P`` and ``Q`` change every step, and ``"tabulate"`` forms the pair once per step;
    ``"recompute"`` forms each flow in the finish that applies it, from the positions,
    whose differences telescope. The treatments agree to rounding, not bit for bit.

    **The stiffness indicator** (SEP 24 section 6.3, Q1): ``h rho = 2 ||F3 - F2|| /
    ||F2 - g1||`` over the norm mask estimates ``|h lambda|`` of the linearised source
    on the content the step integrates (for ``u' = lambda u`` it is exactly
    ``|h lambda|``), the quantity the stability interval of section 4.6 bounds. It is
    kept in ``stiffness`` and never enters the norm column. It reads the content's
    margin, not the worst case: measured on an ionising capillary it passed 2.785 on
    the step before the replay diverged, and beside an in-stage collar it read 0.17 to
    0.65 of ``h alpha_max / 2`` as the collar held less or more of the field.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend whose state, hooks and boundary treatments are used; its state is
        a host array in the canonical layout.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.

    Attributes
    ----------
    stiffness : float
        ``h rho`` of the last completed step, dimensionless; NaN before one, or when
        ``F2 - g1`` vanished on the norm mask.
    atomic : bool
        Always ``False``: the passes are Python calls a signal may interrupt.

    Notes
    -----
    What the engine holds between steps is the registers ``B`` and ``C`` and, on a
    constant operator tabulating, the table: :meth:`held` names them for the backend's
    memory ledger, and :meth:`close` releases them.
    """

    def __init__(self, backend: StepDriver[Any], method: LowStorageRungeKutta) -> None:
        """Hold the backend with no registers yet."""
        self._backend = backend
        self._method = method
        self._accumulator: _State | None = None
        self._stage: _State | None = None
        self._table: _State | None = None
        self._table_step = math.nan
        self._keys: tuple[np.ndarray, np.ndarray] | None = None
        self.stiffness = math.nan
        self.atomic = False

    def reset(self) -> None:
        """Drop the flow table, which a resume builds again from its own first step."""
        self._table = None
        self._table_step = math.nan
        self.stiffness = math.nan

    def close(self) -> None:
        """Release the registers and the flow table."""
        self.reset()
        self._accumulator = None
        self._stage = None

    def held(self) -> dict[str, int]:
        """Return the bytes of every array the engine holds between steps, by name.

        Returns
        -------
        dict
            ``register_accumulator``, ``register_stage`` and ``propagators`` (the
            table), each zero while it is not allocated.
        """
        return {
            "register_accumulator": _bytes(self._accumulator),
            "register_stage": _bytes(self._stage),
            "propagators": _bytes(self._table),
        }

    def step(self, z: float, dz: float) -> float:
        """Take one RK4IP step in the four passes of the register form.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The accepted state's squared norm over the norm mask, before the boundary
            treatments.
        """
        backend = self._backend
        midpoint, end = z + 0.5 * dz, z + dz
        first, second = self._flows(z, dz)
        mask = backend._problem.grids.norm_mask
        state = cast("_State", backend._state)

        # Pass 1: F1 at z_n; A <- P A, g <- P S, B <- A + h/6 g, C <- A + h/2 g.
        source = backend._adams_lab_source(state, z)
        flow = backend._adams_propagator(z, midpoint) if first is None else first
        if state.flags.writeable:
            np.multiply(state, flow, out=state)
        else:
            # An observation holds the accepted state read-only; it keeps its own.
            state = flow * state
            backend._state = state
        slope = flow * source
        accumulator, stage = self._registers(state)
        np.add(state, (dz / 6.0) * slope, out=accumulator)
        np.add(state, (0.5 * dz) * slope, out=stage)

        # Pass 2: F2 at z_m; B <- B + h/3 S, C <- A + h/2 S.
        source = backend._adams_lab_source(stage, midpoint)
        lower = _masked_norm(source - slope, mask)
        accumulator += (dz / 3.0) * source
        np.add(state, (0.5 * dz) * source, out=stage)
        previous = source

        # Pass 3: F3 at z_m; B <- B + h/3 S, C <- Q (A + h S).
        source = backend._adams_lab_source(stage, midpoint)
        upper = _masked_norm(source - previous, mask)
        accumulator += (dz / 3.0) * source
        flow = backend._adams_propagator(midpoint, end) if second is None else second
        np.multiply(flow, state + dz * source, out=stage)

        # Pass 4: F4 at z_n + h; A <- Q B + h/6 S.
        source = backend._adams_lab_source(stage, end)
        flow = backend._adams_propagator(midpoint, end) if second is None else second
        np.multiply(flow, accumulator, out=state)
        state += (dz / 6.0) * source
        self.stiffness = 2.0 * upper / lower if lower > 0.0 else math.nan
        accepted = state[..., mask]
        return float(np.vdot(accepted, accepted).real)

    def _flows(self, z: float, dz: float) -> tuple[_State | None, _State | None]:
        """Read the step's bundles and return the flows held for it, ``P`` and ``Q``.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        tuple
            ``P`` and ``Q`` when they are held for the step -- the table on a constant
            operator, the pair on a varying one -- or ``None`` each when recomputed in
            the finishes that apply them.
        """
        backend = self._backend
        midpoint, end = z + 0.5 * dz, z + dz
        tabulating = backend._propagator_mode == "tabulate"
        table = tabulating and backend._problem.flags.constant_linop
        backend._hoist(z)
        # Each evaluation reads its position as its own anchor and each flow its
        # interval, so the anchors' bundles are read here, once each.
        backend._stage_bundles(z, [z] if table else [z, midpoint])
        backend._stage_bundles(midpoint, [midpoint] if table else [midpoint, end])
        backend._stage_bundles(end, [end])
        if table:
            if self._keys is None:
                schedule = backend._problem.stepper.schedule
                assert schedule is not None  # the translation refused it without one
                self._keys = (table_steps(schedule), schedule.edges)
            key = schedule_key(*self._keys, z, dz)
            if self._table is None or key != self._table_step:
                # From the operator's origin, where a constant operator's increment is
                # the half step itself, not fl(z + h/2) - z (class docstring).
                origin = backend._problem.linear.z_span[0]
                half = origin + 0.5 * key
                backend._stage_bundles(origin, [half])
                self._table = backend._adams_propagator(origin, half)
                self._table_step = key
            return self._table, self._table
        if tabulating:
            return (
                backend._adams_propagator(z, midpoint),
                backend._adams_propagator(midpoint, end),
            )
        return None, None

    def _registers(self, state: _State) -> tuple[_State, _State]:
        """Return ``B`` and ``C``, allocated like the state at the first step.

        Parameters
        ----------
        state : numpy.ndarray
            The state, whose shape and type the registers take.

        Returns
        -------
        tuple of numpy.ndarray
            The accumulator and the stage register.
        """
        if self._accumulator is None or self._stage is None:
            self._accumulator = np.empty_like(state)
            self._stage = np.empty_like(state)
        return self._accumulator, self._stage


def _bytes(values: np.ndarray | None) -> int:
    """Return an optional array's bytes, zero for ``None``.

    Parameters
    ----------
    values : numpy.ndarray or None
        The array.

    Returns
    -------
    int
        Its ``nbytes``, or 0.
    """
    return 0 if values is None else int(values.nbytes)


def _masked_norm(values: _State, mask: np.ndarray) -> float:
    """Return the 2-norm of an array over the norm mask's bins.

    Parameters
    ----------
    values : numpy.ndarray
        An array of the state's layout.
    mask : numpy.ndarray
        The norm mask, boolean of shape ``(n_omega,)``.

    Returns
    -------
    float
        The norm, in the array's units.
    """
    return float(np.linalg.norm(values[..., mask]))


class LowStorageMarch:
    """Replay a schedule with the low-storage RK4IP march, accepting in place.

    Follows :class:`solprop.backends.multistep.AdamsMarch`: it owns its registers
    through an engine and replays a prescribed schedule, which is the only thing it
    does (SEP 24 section 6.2). Each step calls the engine, applies the backend's
    ``_finish`` (the hard band mask, and the split-step collar where chosen) and moves
    the position in one commit, then records the step through the backend.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend whose state is marched; it owns the step history and the
        statistics rows.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.

    Attributes
    ----------
    source_evaluations : int
        Source evaluations of the committed steps since the state was last loaded
        (``StepDriver.set_initial``, which sets it to zero), four a step: a physical
        count, never a logical response call (SEP 24 section 5). A step that did not
        commit -- abandoned in flight, or diverged -- adds none.

    Notes
    -----
    The march has no error estimate to warn on. It logs a warning, once per call of
    :meth:`replay_to`, when the engine's stiffness indicator passes RK4's
    negative-real stability interval
    (:data:`~solprop.stepper.low_storage.LOW_STORAGE_REAL_INTERVAL`): an explicitly
    integrated damping term the translation cannot bound in advance -- plasma loss --
    is then near the step's stability limit.
    """

    def __init__(self, backend: StepDriver[Any], method: LowStorageRungeKutta) -> None:
        """Hold the backend and its engine."""
        self._backend = backend
        self._method = method
        self._warned = False
        resident = backend._low_storage_engine(method)
        self._engine: LowStorageEngine = (
            HostLowStorage(backend, method)
            if resident is None
            else cast("LowStorageEngine", resident)
        )
        self.source_evaluations = 0

    @property
    def engine(self) -> LowStorageEngine:
        """Return the engine that holds the registers.

        Returns
        -------
        LowStorageEngine
            The host engine, or the backend's own.
        """
        return self._engine

    def reset(self) -> None:
        """Forget what the engine holds: the state was loaded outside the march."""
        self._engine.reset()

    def close(self) -> None:
        """Release the engine's registers and table, with the backend's buffers."""
        self._engine.close()

    def snapshot(self) -> None:
        """Return nothing for a resume to continue: the march keeps no history.

        A resume at a schedule edge needs no history and no recycled stage
        (SEP 24 section 5), as under ``RungeKutta()``.
        """
        return

    def replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay the schedule's steps to `target`, mirroring the Adams replay.

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
        backend = self._backend
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        self._warned = False
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
        (:class:`solprop.backends.batch.LockstepRunner`), with
        :meth:`solprop.backends.multistep.AdamsMarch.replay_step`'s signature.

        Parameters
        ----------
        dz : float
            The step, m.
        end : float
            The schedule's edge it ends on, m.
        slack : float
            How far an edge may sit from a target it lands on, m; unused, since the
            march restarts nothing at a kink.

        Returns
        -------
        float
            The norm the history records: 0.0, "not estimated".

        Raises
        ------
        BackendError
            If an earlier step was abandoned in flight.
        ConvergenceError
            If the accepted state is not finite: the field has diverged, and a replay
            cannot reject the step. Raised in flight, so the state is lost with it.
        """
        del slack
        backend = self._backend
        backend._refuse_in_flight()
        z = backend._z
        backend._in_flight = (z, dz)
        if self._engine.atomic:
            # One native call: a signal waits for its return anyway, so the step,
            # the finish and the move commit together (SEP 24 section 6.7).
            with committing():
                self._move(self._accept(z, dz, self._engine.step(z, dz)), end)
        else:
            squared = self._engine.step(z, dz)
            finished = self._accept(z, dz, squared)
            with committing():
                self._move(finished, end)
        self._check_stiffness(z, dz)
        return 0.0

    def _move(self, finished: object, end: float) -> None:
        """Make the finished state the backend's at the step's end; clear the flag.

        Parameters
        ----------
        finished : object
            The state after the backend's ``_finish``.
        end : float
            The schedule's edge the step ends on, m.
        """
        backend = self._backend
        backend._state = finished
        backend._z = end
        backend._in_flight = None
        self.source_evaluations += self._method.evaluations

    def _accept(self, z: float, dz: float, squared: float) -> object:
        """Refuse a diverged step, or apply the backend's finish to the state.

        Parameters
        ----------
        z : float
            Where the step started, m.
        dz : float
            The step, m.
        squared : float
            The accepted state's squared norm the engine returned.

        Returns
        -------
        object
            The state after the backend's ``_finish``.

        Raises
        ------
        ConvergenceError
            If `squared` is not finite.
        """
        backend = self._backend
        if not math.isfinite(squared):
            msg = (
                f"{backend.name}: a replayed {self._method.name} step of {dz:.4g} m "
                f"from z = {z:.6g} m ended in a non-finite state: the field has "
                "diverged. A replayed schedule takes its steps as given and cannot "
                "reject one, so a step beyond an explicit stability limit -- the "
                "in-stage collars' -- grows until it overflows; the translation "
                "refuses a declared step past the collars' limit. The state was "
                "accepted in place and is lost; take shorter steps"
            )
            raise ConvergenceError(msg)
        if (reader := backend._split_reader(z + dz)) is not None:
            reader.before_split(backend._observed_field(backend._state), dz)
        return backend._finish(backend._state, dz)

    def _check_stiffness(self, z: float, dz: float) -> None:
        """Warn, once per replay, when a step's stiffness passes RK4's interval.

        Parameters
        ----------
        z : float
            Where the step started, m.
        dz : float
            The step, m.
        """
        stiffness = self._engine.stiffness
        interval = self._method.real_stability_interval
        if self._warned or not stiffness > interval:
            return
        self._warned = True
        logger.warning(
            "%s: the %s march's stiffness indicator h rho = %.3g passed RK4's "
            "stability interval %.4g on the step of %.4g m from z = %.6g m; an "
            "explicitly integrated damping term (plasma loss, an in-stage collar) is "
            "near the step's stability limit and may make the replay diverge. Take "
            "shorter steps",
            self._backend.name,
            self._method.name,
            stiffness,
            interval,
            dz,
            z,
        )
