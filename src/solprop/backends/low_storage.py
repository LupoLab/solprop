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

**The march** (the reference's
:class:`solprop.backends.host.low_storage.HostLowStorageMarch`; the native session's
replay) replays a prescribed schedule and never rejects, so it accepts each step in
place: from the start of the step's trial to its accept the engine is *in flight*
and every entry that reads or advances the state refuses it, until a load replaces the
state (SEP 24 section 6.2, ADR-050). It records no error estimate: the step history's
norm is 0.0, "not estimated" (section 6.3). Divergence is detected on the accepted
state's norm instead.

**The engine** holds the registers and makes the passes: it is a
:class:`~solprop.backends.engines.RungeKuttaEngine` with the documented destructive
trial (SEP 52 section 5.2), as native's low-storage engine is.
:class:`HostLowStorage` is the host engine and the oracle, over the workspace's
``lab_source`` and ``propagator``. The engines and the table keys
(:func:`table_steps`, :func:`schedule_key`) are stateless numerical code the reference
and the native backend share (SEP 47 Part III decision 5).

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
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Complex, Float

from solprop.backends.engines import AcceptRecord, BoundaryReading
from solprop.exceptions import BackendError
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import LANDING_ULPS, FixedSchedule

if TYPE_CHECKING:
    from solprop.backends.numpy_ref.workspace import NumpyWorkspace

__all__ = [
    "HostLowStorage",
    "schedule_key",
    "table_steps",
]

logger = logging.getLogger(__name__)

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]


def table_steps(schedule: FixedSchedule) -> Float[np.ndarray, " n_steps"]:
    """Return the step each of a schedule's steps builds and keys the flow table on.

    SEP 24 Q10, decided in phase 2: the mean step of the step's **run**, the longest
    stretch of consecutive steps within :data:`~solprop.stepper.schedule.LANDING_ULPS`
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

    Found by position rather than by a step counter, so that it needs no count of the
    steps taken before.

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


class HostLowStorage:
    """The host engine: the four register passes over host arrays (SEP 24 6.5).

    The workspace's state is the register ``A`` and is updated in place by the
    trial; ``B`` and ``C`` are this engine's host arrays, allocated at the first step.
    Each pass makes one source evaluation through the workspace's ``lab_source`` (the
    complete source, the noise draw advanced to the pass's position and the in-stage
    absorbers included) and takes ``P`` and ``Q`` from its ``propagator`` or the
    table, allocating transients as the reference does. The step's three positions
    are known before its first pass, so their host scalars are read together
    (``stage_bundles``) and the evaluations take the kept bundles.

    **The commit contract** (SEP 52 section 5.2): :meth:`trial` overwrites the state,
    and from its start until :meth:`accept` returns the engine is :attr:`in_flight`.
    An exception or a non-finite result in that interval leaves no committed state at
    that position; only :meth:`reset`, at a load, clears the flag.

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
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The reference's workspace, whose state, evaluations and finish are used. The
        native session keeps its registers resident and never takes this engine.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        The replayed schedule, whose runs key the flow table; ``None`` only for an
        engine that never tabulates.
    propagator_mode : {'tabulate', 'recompute'}
        Whether the step's flows are held for it.

    Attributes
    ----------
    stiffness : float
        ``h rho`` of the last completed trial, dimensionless; NaN before one, or when
        ``F2 - g1`` vanished on the norm mask.

    Notes
    -----
    What the engine holds between steps is the registers ``B`` and ``C`` and, on a
    constant operator tabulating, the table: :meth:`held` names them for the backend's
    memory ledger, and :meth:`close` releases them.
    """

    def __init__(
        self,
        workspace: "NumpyWorkspace",
        method: LowStorageRungeKutta,
        schedule: FixedSchedule | None,
        propagator_mode: str,
    ) -> None:
        """Hold the workspace with no registers yet."""
        self._workspace = workspace
        self._method = method
        self._schedule = schedule
        self._propagator_mode = propagator_mode
        self._accumulator: _State | None = None
        self._stage: _State | None = None
        self._table: _State | None = None
        self._table_step = math.nan
        self._keys: tuple[np.ndarray, np.ndarray] | None = None
        self._in_flight = False
        self.stiffness = math.nan

    @property
    def in_flight(self) -> bool:  # numpydoc ignore=RT01
        """Whether a trial overwrote the state and its accept has not returned."""
        return self._in_flight

    def reset(self) -> None:
        """Drop the flow table and a step left in flight: a state was loaded.

        A resume builds the table again from its own first step.
        """
        self._table = None
        self._table_step = math.nan
        self.stiffness = math.nan
        self._in_flight = False

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

    def trial(self, z: float, dz: float) -> float:
        """Take one RK4IP step in place, in the four passes of the register form.

        The engine is in flight from here until :meth:`accept` returns.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            0.0, "not estimated"; or, when the field has diverged, the accepted
            state's non-finite squared norm over the norm mask.
        """
        workspace = self._workspace
        state = workspace.committed()
        self._in_flight = True
        squared = self._step(state, z, dz)
        return squared if not math.isfinite(squared) else 0.0

    def accept(self, dz: float) -> AcceptRecord:
        """Finish the state the trial left and commit it; the step leaves flight.

        Parameters
        ----------
        dz : float
            The step, m, which the split collar's rate needs.

        Returns
        -------
        solprop.backends.engines.AcceptRecord
            The stiffness indicator and the finish's split collar's sums.
        """
        workspace = self._workspace
        finished, record = workspace.finish(workspace.committed(), dz)
        workspace.install(finished)
        self._in_flight = False
        return AcceptRecord(
            stiffness=self.stiffness, removed=record.removed, found=record.found
        )

    def extend(self, z: float, dz: float) -> None:
        """Refuse: the low-storage march replays and observes no plane densely.

        Parameters
        ----------
        z : float
            Where the step started, m.
        dz : float
            The step, m.

        Raises
        ------
        BackendError
            Always.
        """
        del z, dz
        raise BackendError("the low-storage engine holds no dense extension")

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        solprop.backends.engines.BoundaryReading
            The workspace's reading.
        """
        return self._workspace.reading()

    def _step(self, state: _State, z: float, dz: float) -> float:
        """Take the step's four passes on `state`; return its squared norm."""
        workspace = self._workspace
        midpoint, end = z + 0.5 * dz, z + dz
        first, second = self._flows(z, dz)
        mask = workspace.problem.grids.norm_mask

        # Pass 1: F1 at z_n; A <- P A, g <- P S, B <- A + h/6 g, C <- A + h/2 g.
        source = workspace.lab_source(state, z)
        flow = workspace.propagator(z, midpoint) if first is None else first
        if state.flags.writeable:
            np.multiply(state, flow, out=state)
        else:
            # An observation holds the accepted state read-only; it keeps its own.
            state = flow * state
            workspace.install(state)
        slope = flow * source
        accumulator, stage = self._registers(state)
        np.add(state, (dz / 6.0) * slope, out=accumulator)
        np.add(state, (0.5 * dz) * slope, out=stage)

        # Pass 2: F2 at z_m; B <- B + h/3 S, C <- A + h/2 S.
        source = workspace.lab_source(stage, midpoint)
        lower = _masked_norm(source - slope, mask)
        accumulator += (dz / 3.0) * source
        np.add(state, (0.5 * dz) * source, out=stage)
        previous = source

        # Pass 3: F3 at z_m; B <- B + h/3 S, C <- Q (A + h S).
        source = workspace.lab_source(stage, midpoint)
        upper = _masked_norm(source - previous, mask)
        accumulator += (dz / 3.0) * source
        flow = workspace.propagator(midpoint, end) if second is None else second
        np.multiply(flow, state + dz * source, out=stage)

        # Pass 4: F4 at z_n + h; A <- Q B + h/6 S.
        source = workspace.lab_source(stage, end)
        flow = workspace.propagator(midpoint, end) if second is None else second
        np.multiply(flow, accumulator, out=state)
        state += (dz / 6.0) * source
        self.stiffness = 2.0 * upper / lower if lower > 0.0 else math.nan
        accepted = state[..., mask]
        return float(np.vdot(accepted, accepted).real)

    def _flows(self, z: float, dz: float) -> tuple[_State | None, _State | None]:
        """Read the step's bundles and return the flows held for it, ``P`` and ``Q``.

        - ``z``: where the step starts, m.
        - ``dz``: the step, m, positive.
        - Returns: ``P`` and ``Q`` when they are held for the step -- the table on a
          constant operator, the pair on a varying one -- or ``None`` each when
          recomputed in the finishes that apply them.
        """
        workspace = self._workspace
        midpoint, end = z + 0.5 * dz, z + dz
        tabulating = self._propagator_mode == "tabulate"
        table = tabulating and workspace.problem.flags.constant_linop
        workspace.hoist(z)
        # Each evaluation reads its position as its own anchor and each flow its
        # interval, so the anchors' bundles are read here, once each.
        workspace.stage_bundles(z, [z] if table else [z, midpoint])
        workspace.stage_bundles(midpoint, [midpoint] if table else [midpoint, end])
        workspace.stage_bundles(end, [end])
        if table:
            if self._keys is None:
                schedule = self._schedule
                assert schedule is not None  # the translation refused it without one
                self._keys = (table_steps(schedule), schedule.edges)
            key = schedule_key(*self._keys, z, dz)
            if self._table is None or key != self._table_step:
                # From the operator's origin, where a constant operator's increment is
                # the half step itself, not fl(z + h/2) - z (class docstring).
                origin = workspace.problem.linear.z_span[0]
                half = origin + 0.5 * key
                workspace.stage_bundles(origin, [half])
                self._table = workspace.propagator(origin, half)
                self._table_step = key
            return self._table, self._table
        if tabulating:
            return (
                workspace.propagator(z, midpoint),
                workspace.propagator(midpoint, end),
            )
        return None, None

    def _registers(self, state: _State) -> tuple[_State, _State]:
        """Return ``B`` and ``C``, allocated like the state at the first step.

        - ``state``: the state, whose shape and type the registers take.
        - Returns: the accumulator and the stage register.
        """
        if self._accumulator is None or self._stage is None:
            self._accumulator = np.empty_like(state)
            self._stage = np.empty_like(state)
        return self._accumulator, self._stage


def _bytes(values: np.ndarray | None) -> int:
    """Return an optional array's bytes, zero for ``None``."""
    return 0 if values is None else int(values.nbytes)


def _masked_norm(values: _State, mask: np.ndarray) -> float:
    """Return the 2-norm of an array over the norm mask's bins.

    - ``values``: an array of the state's layout.
    - ``mask``: the norm mask, boolean of shape ``(n_omega,)``.
    """
    return float(np.linalg.norm(values[..., mask]))
